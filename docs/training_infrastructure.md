# Phase 6A Training Infrastructure

Phase 6A adds reusable training infrastructure only. It does not run the random
versus pretrained initialization pilot and does not run full cross-validation.

## Scope

The infrastructure trains the committed custom `ResNet3D18` on preprocessed ACDC
patient tensors produced by the frozen preprocessing pipeline. It does not modify
the preprocessing contract, model input geometry, class mapping, outer split
artifact, or model architecture.

Checkpoint selection uses validation Macro F1. The best checkpoint stores state
dictionaries and metadata, not a serialized model object. The checkpoint payload
includes:

* model state dictionary;
* optimizer state dictionary;
* epoch;
* best validation Macro F1;
* training configuration metadata;
* initialization strategy;
* seed;
* initialization compatibility report.

## Initialization

Random initialization is the default and works offline. It preserves the
constructor initialization in `ResNet3D18`.

Pretrained initialization is opt-in. When selected, torchvision `r3d_18`
Kinetics weights are loaded lazily. The custom model remains the target model;
the implementation does not replace it with torchvision's model.

The RGB stem weight is adapted from three source channels to the ED/ES target
channels with the exact transform:

```python
rgb_mean = source_weight.mean(dim=1, keepdim=True)
target_weight = rgb_mean.repeat(1, 2, 1, 1, 1) * (3.0 / 2.0)
```

The classifier head is task-specific and is never imported from pretrained
weights. Compatible parameters are copied explicitly, incompatible parameters are
reported, and the resulting target state dictionary is loaded strictly.

No conclusion has been made about whether pretrained initialization is better
than random initialization. That comparison belongs to the Phase 6 pilot.

## Validation Split

For one requested outer fold, the outer test patients remain reserved for final
evaluation only. The inner validation split is created from the outer development
patients, deterministically, at patient level, and stratified by class label.

The split utility validates:

* inner train and validation patient IDs are disjoint;
* inner train patient IDs do not overlap outer test IDs;
* inner validation patient IDs do not overlap outer test IDs.

## Next Verification Command

The next Phase 6B single-fold pilot can be launched with:

```bash
python scripts/run_training.py --dataset-dir data/raw/acdc/training
```

Use `training.initialization_strategy: random` for the offline baseline and
`training.initialization_strategy: pretrained` for the opt-in pretrained pilot.

## Phase 6 Initialization Pilot Result

The initialization pilot was performed on outer fold 0 using the same:

- seed: 42;
- inner train/validation split;
- deterministic preprocessing;
- approved training-only augmentation;
- optimizer and hyperparameters;
- early-stopping configuration;
- checkpoint-selection metric;
- CUDA execution environment.

The outer test patients were not used for initialization selection.

### Random Initialization

Custom `ResNet3D18` with random initialization:

- best validation Macro F1: `0.274444`;
- best epoch: `4`.

### Compatible Pretrained Initialization

The same custom `ResNet3D18` initialized from compatible torchvision `r3d_18`
Kinetics weights:

- directly transferred state entries: `119`;
- adapted state entries: `1`;
- intentionally skipped classifier entries: `2`;
- missing state entries: `0`;
- best validation Macro F1: `0.593651`;
- best epoch: `11`.

The input stem was adapted from three RGB channels to the two ED/ES channels
using the documented 3-to-2 transformation.

The pretrained classifier head was not transferred.

### Selected Initialization Strategy

Compatible pretrained initialization was selected for Phase 7 because it achieved
the higher inner-validation Macro F1:

`0.593651 > 0.274444`

The neural network remains the project's custom `BasicBlock3D` + `ResNet3D18`.
torchvision is used only as the source of compatible pretrained weights.

### CUDA Reproducibility Note

Both pilot runs used:

`CUBLAS_WORKSPACE_CONFIG=:4096:8`

PyTorch still reports that `max_pool3d_with_indices_backward_cuda` does not have
a deterministic CUDA implementation.

Therefore, the CUDA training path uses fixed seeds and identical execution
conditions but is not claimed to be bitwise deterministic.
