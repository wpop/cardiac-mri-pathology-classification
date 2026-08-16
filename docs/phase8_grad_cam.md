# Phase 8 Grad-CAM Scientific Contract

Phase 8 implements post-hoc 3D Grad-CAM explainability for the frozen
patient-level cardiac pathology classifier. The explainability pipeline uses the
project's custom `ResNet3D18` only. It does not replace the model with
torchvision, MONAI, MedicalNet, or any third-party architecture.

## Model And Target

- Model: custom `ResNet3D18`.
- Input tensor shape: `[N, 2, 14, 144, 144]`.
- Channel semantics: channel `0` is ED and channel `1` is ES.
- Model output: raw logits with shape `[N, 5]`.
- Grad-CAM target layer: `model.layer4[1]`.
- Verified target-layer activation shape: `[N, 512, 7, 5, 5]`.
- Target score: the raw predicted-class logit.
- No softmax is applied inside Grad-CAM.

## Grad-CAM Algorithm

For each patient, Grad-CAM is computed from the selected raw predicted-class
logit using the frozen target layer. Gradients are globally averaged over the
target-layer spatial dimensions `D`, `H`, and `W`, producing one scalar weight
per activation channel.

The frozen Phase 8 CAM construction order is:

1. Weight target-layer activations by the gradient-GAP channel weights.
2. Sum the weighted activations over channels.
3. Apply ReLU.
4. Apply per-patient min-max normalization.
5. Trilinearly upsample to `[14, 144, 144]` with `align_corners=False`.

Degenerate CAM ranges are handled safely so the normalized CAM remains finite
and constrained to `[0, 1]`.

## Attribution Semantics

The Grad-CAM volume is one joint ED+ES model attribution. It is not an
independent ED attribution and not an independent ES attribution. The same CAM
slice is overlaid on ED and ES images to show where the joint two-channel model
attribution localizes on each phase.

Grad-CAM is a post-hoc explanation of model scoring behavior. It is not causal
evidence, a clinical proof, or a substitute for medical interpretation.

## Frozen Case Selection

Phase 8 uses 10 frozen real ACDC cases:

- `patient070`
- `patient063`
- `patient010`
- `patient005`
- `patient023`
- `patient022`
- `patient043`
- `patient044`
- `patient091`
- `patient083`

The deterministic selection rule is one correct and one incorrect case per true
class. Within each subgroup, the selected case is the patient nearest to the
subgroup median confidence, with `patient_id` as the deterministic tie-break.

Each case uses its own outer-test fold checkpoint from Phase 7. All 10 cases
reproduced their stored Phase 7 out-of-fold logits and probabilities exactly
within the frozen verification tolerance before Grad-CAM metadata was accepted.

## Slice Selection

The selected visualization slice is deterministic and restricted to the real
non-padded Z range:

`selected_slice = valid_start + argmax(mean(CAM[valid_start:valid_end], over H/W))`

The valid interval is `[valid_start, valid_end)`, where `valid_end` is exclusive.
Ties resolve to the smallest valid absolute slice index through the natural
`argmax` behavior.

The non-padded restriction is required because the preprocessed tensor has a
fixed depth of 14 slices. Some patients have artificial Z-padding after
preprocessing. Without restricting selection to the real depth range, a high CAM
value in padded slices could select an anatomically artificial slice and produce
misleading QA figures.

## Artifacts

Phase 8 artifacts are stored under:

- `artifacts/phase8_grad_cam/figures/`
- `artifacts/phase8_grad_cam/phase8_grad_cam_metadata.json`

The metadata records the checkpoint, fold, class labels, target class,
activation shape, CAM output shape, normalization rule, upsampling rule, valid
slice interval, selected slice, attribution semantics, and figure path for each
frozen case.

Visual QA passed for the frozen Phase 8 Grad-CAM figures.
