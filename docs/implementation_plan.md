# Implementation Plan

## Project Goal

Build a standalone medical AI pipeline for patient-level cardiac pathology classification from real ACDC cardiac cine MRI using a custom 3D ResNet-18.

The final Python model will later serve as the ML foundation for a separate C++/Qt Medical AI Workstation.

This project does not reuse segmentation masks, segmentation outputs, existing U-Net checkpoints, or segmentation-derived measurements.

---

## Phase 0 — Project Bootstrap

Create the initial professional repository structure.

Deliverables:

* Git repository;
* project configuration;
* class mapping;
* `.gitignore`;
* `pyproject.toml`;
* `Makefile`;
* documentation;
* CI configuration;
* Python package structure;
* test structure.

Do not implement preprocessing, the neural network, or training yet.

---

## Phase 1 — Real ACDC Dataset Inspection — COMPLETE

The real official ACDC training cohort has been inspected and Phase 1 is complete.

Verify:

* actual patient count;
* diagnostic class distribution;
* ED and ES frames;
* H/W dimensions;
* slice counts;
* X/Y/Z spacing;
* physical Z coverage;
* MRI orientation;
* affine information;
* background characteristics;
* NaN/Inf presence.

Analyze whether acquisition variables such as slice count or padding fraction correlate with diagnosis.

Phase 1 froze:

* authoritative standalone ED/ES spatial inputs;
* LPS target orientation;
* target spacing `X=1.50 mm`, `Y=1.50 mm`, `Z=7.50 mm`;
* deterministic geometric FOV-center crop;
* final tensor shape `[2, 14, 144, 144]`;
* joint ED/ES p0.5-p99.5 clipping and z-score normalization;
* post-normalization Z padding to `D=14`;
* anisotropy-aware ResNet3D18 depth-downsampling schedule.

Known residual acquisition/padding shortcut risk remains documented in the preprocessing contract.

---

## Phase 2 — Dataset Indexing and Patient Splits — NEXT

Implement:

* ACDC patient indexing;
* diagnostic metadata parsing;
* ED/ES pairing;
* patient-level representation;
* stratified 5-fold cross-validation splits.

All splits must be patient-level and leakage-safe.

---

## Phase 3 — Deterministic Preprocessing

Implement the final preprocessing pipeline:

```text
load NIfTI
      ↓
read affine and spacing
      ↓
explicit orientation normalization
      ↓
extract ED and ES
      ↓
physical resampling
      ↓
deterministic crop
      ↓
foreground-aware normalization
      ↓
padding
      ↓
stack ED + ES
      ↓
transpose to [D, H, W]
      ↓
float32 tensor
```

ED and ES must undergo the same spatial preprocessing.

The final preprocessing contract must be documented in:

```text
docs/preprocessing_contract.md
```

---

## Phase 4 — Golden Preprocessing References

Create deterministic preprocessing references from selected real ACDC patients.

Store them under:

```text
tests/fixtures/golden/
```

These references will be used for preprocessing regression testing and later Python ↔ C++ reproducibility work.

---

## Phase 5 — Custom 3D ResNet-18

Implement:

* `BasicBlock3D`;
* `ResNet3D18`.

The model input is:

```text
[N, 2, D, H, W]
```

where:

```text
channel 0 = ED
channel 1 = ES
```

The model output is:

```text
[N, 5]
```

containing raw logits for:

```text
NOR
DCM
HCM
MINF
RV
```

The architecture must account for cardiac MRI anisotropy.

---

## Phase 6 — Training Infrastructure and Initialization Pilot

Implement training infrastructure.

Compare:

* random initialization;
* compatible pretrained initialization where technically appropriate.

Run the comparison first on one representative fold.

Use the selected initialization strategy for the full experiment.

---

## Phase 7 — Five-Fold Cross-Validation

Run stratified patient-level 5-fold cross-validation.

Inside every outer training fold, use a separate validation subset for:

* early stopping;
* checkpoint selection.

Report:

* Accuracy mean ± standard deviation;
* Macro F1 mean ± standard deviation;
* per-class Precision;
* per-class Recall;
* per-class F1;
* pooled out-of-fold confusion matrix;
* multiclass one-vs-rest AUROC where statistically meaningful.

Every patient must appear exactly once in the pooled out-of-fold evaluation.

---

## Phase 8 — 3D Grad-CAM

Implement compact 3D Grad-CAM visualization for representative real ACDC patients.

Show:

* ED slice;
* ES slice;
* Grad-CAM overlay;
* true class;
* predicted class;
* confidence.

---

## Phase 9 — Final Production Model

Use the best-epoch information from cross-validation to derive a deterministic final training duration.

Then train one production model using **all labeled ACDC patients**.

Produce:

```text
classifier.pt
```

Cross-validation metrics remain the reported generalization metrics.

---

## Phase 10 — ONNX Deployment

Export the production model as:

```text
classifier.onnx
```

Planned interface:

```text
Input:
cine_mri
float32
[N, 2, D, H, W]

Output:
logits
float32
[N, 5]
```

Softmax remains outside the model.

Validate numerical parity between:

```text
PyTorch
   ↕
ONNX Runtime
```

using real preprocessed ACDC patient inputs.

Finalize:

```text
docs/preprocessing_contract.md
docs/onnx_deployment_contract.md
```

---

## Final Deliverables

The completed project should include:

* `classifier.pt`;
* `classifier.onnx`;
* class mapping;
* preprocessing contract;
* ONNX deployment contract;
* cross-validation metrics;
* pooled out-of-fold predictions;
* confusion matrix;
* ROC curves where appropriate;
* preprocessing QA figures;
* training histories;
* Grad-CAM figures;
* golden preprocessing references;
* automated tests;
* CI pipeline.

---

## Current Status

Current phase:

```text
Phase 0 — Project Bootstrap
```

No preprocessing constants have been frozen.

No neural network has been implemented.

No training has been performed.

The next major phase is:

```text
Phase 1 — Real ACDC Dataset Inspection
```
