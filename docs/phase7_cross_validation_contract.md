# Phase 7 Cross-Validation Contract

## 1. Status / Freeze Rule

This contract is frozen before Phase 7 Fold 0 training begins. Once Fold 0 starts, the scientific training/evaluation contract must not change. Any required scientific change invalidates the active Phase 7 run and requires a documented restart.

Pre-contract baseline repository revision from which this contract was created:

```text
8a1e4924e687f1c9fbeb7a35924b86c2cc1c61e0
```

This baseline is not the final commit containing this contract.

## 2. Dataset Cohort

Dataset: ACDC official labeled training cohort.

Real indexed labeled patients: 100.

Diagnostic classes:

```text
0 NOR
1 DCM
2 HCM
3 MINF
4 RV
```

Class mapping source: `configs/class_mapping.json`

Class mapping SHA256:

```text
d71fd3a0591ef7318b1683146ae851dee630c306593a49d09a8fbbe7aeb3e17b
```

## 3. Outer Five-Fold Split

Exact persisted artifact: `artifacts/dataset_splits/acdc_5fold_seed42.json`

SHA256:

```text
ee56ce1b581bb1a3cbfe3e304984248464945ce2f05d84453c8e589b3cf8ebf1
```

The experiment uses 5 stratified patient-level folds.

Each fold contains:

- 80 outer-development patients
- 20 outer-test patients
- 16 development patients per class
- 4 outer-test patients per class

Across five folds, every one of the 100 patients appears exactly once in outer-test evaluation.

Outer test is sacred and must not be used for training, early stopping, checkpoint selection, hyperparameter decisions, or initialization selection.

## 4. Inner Validation Split

Exact persisted artifact: `artifacts/dataset_splits/acdc_5fold_inner_validation_seed42.json`

SHA256:

```text
7e492e289a3574ec0dcae0c4b78c63a8d534335ea102db4b58d05c94418d2d33
```

For every outer fold:

- 60 inner-train patients
- 20 inner-validation patients
- 20 outer-test patients

Per class:

- 12 inner-train patients
- 4 inner-validation patients
- 4 outer-test patients

`validation_fraction`: 0.25

Exact inner assignments are persisted and must be reused. They must not be regenerated opportunistically during Phase 7 training.

## 5. Seed / RNG Contract

Experiment seed: 42.

Execution order:

1. Set experiment seed.
2. Construct custom `ResNet3D18`.
3. Apply selected initialization.
4. Reset training RNG streams to the same experiment seed.
5. Begin stochastic training.

Python, NumPy, PyTorch CPU, and CUDA RNG are seeded.

```python
torch.use_deterministic_algorithms(True, warn_only=True)
```

The DataLoader shuffle generator is explicitly seeded from the experiment seed.

## 6. Initialization

Selected strategy: `pretrained`.

This selection comes from the corrected controlled Phase 6 pilot.

Corrected pilot:

| Strategy | Best Validation Macro F1 | Best Epoch |
| --- | ---: | ---: |
| `random` | 0.4222222222222222 | 8 |
| `pretrained` | 0.7677777777777778 | 44 |

The project model remains the custom `BasicBlock3D` + `ResNet3D18`.

`torchvision` is only the compatible pretrained-weight source:

```text
torchvision.models.video.r3d_18
R3D_18_Weights.KINETICS400_V1
```

RGB source stem adaptation:

```python
rgb_mean = source_weight.mean(dim=1, keepdim=True)
target_weight = rgb_mean.repeat(1, 2, 1, 1, 1) * (3.0 / 2.0)
```

Classifier head is not transferred.

No Phase 7 initialization pilot is to be rerun.

## 7. Model Architecture

Custom implementation:

- `src/cardiac_pathology/models/basic_block_3d.py`
- `src/cardiac_pathology/models/resnet3d18.py`

Model history revision:

```text
9a9a717547698c31c361f33592e17d8cf931eb19
```

File SHA256:

```text
ResNet3D18:  fe0e0acb27f302ae0ba73d024d8509bf3741d1a3e2fb1acd3ee6c6474077ce83
BasicBlock3D: af4c2e188c314bb84dd6e0e5b32c414cffcda6e157143291c69d57c9850ca49c
```

Input: `[N, 2, 14, 144, 144]`

Output: `[N, 5]` raw logits

Channel semantics:

```text
channel 0 ED
channel 1 ES
```

Architecture strides:

```text
stem Conv3d: (1, 2, 2)
maxpool: (1, 2, 2)
layer1: (1, 1, 1)
layer2: (1, 2, 2)
layer3: (1, 2, 2)
layer4: (2, 2, 2)
```

Verified feature-map spatial shapes:

```text
input    [14, 144, 144]
stem     [14, 72, 72]
maxpool  [14, 36, 36]
layer1   [14, 36, 36]
layer2   [14, 18, 18]
layer3   [14, 9, 9]
layer4   [7, 5, 5]
```

## 8. Preprocessing Contract

Authoritative document: `docs/preprocessing_contract.md`

History revision:

```text
a654ac14a1d1e306df38de9cb0a7f374adaaaa8d
```

SHA256:

```text
2ea430037849078aaa76f8822d56cdca2e5fb7be5a06a17f742926e1183b310d
```

Currently frozen values from `configs/default.yaml`:

- orientation: LPS
- target spacing Z/Y/X: 7.5 / 1.5 / 1.5 mm
- target shape D/H/W: 14 / 144 / 144
- crop: FOV-center crop
- image interpolation: linear
- normalization: joint ED/ES normalization
- clipping: p0.5 to p99.5
- epsilon: 1e-6
- post-normalization Z padding value: 0.0
- source order: XYZ
- model order: ZYX
- dtype: float32

This section summarizes the frozen preprocessing contract and does not redefine preprocessing beyond `docs/preprocessing_contract.md`.

## 9. Training Augmentation

Training augmentation is enabled for training only.

- rotation: +/-5 degrees in-plane
- translation: +/-5 mm in-plane
- intensity scale: 0.95 to 1.05
- Gaussian noise std: 0.01
- probability: 0.5 independently for each configured transformation
- shared ED/ES spatial transform
- no left-right flip
- no Z rotation
- no Z translation

Validation/test augmentation is disabled.

## 10. Training Hyperparameters

```text
num_folds: 5
initialization_strategy: pretrained
batch_size: 2
max epochs: 50
learning_rate: 1e-4
weight_decay: 1e-4
optimizer: AdamW
early stopping patience: 10
early stopping min_delta: 0.0
num_workers: 0
checkpoint root: artifacts/checkpoints/phase7
device: cuda
```

Loss: `CrossEntropyLoss`

## 11. Checkpoint Selection / Early Stopping

Checkpoint metric: validation Macro F1.

The metric is maximized.

Improvement criterion:

```text
metric > previous_best + min_delta
```

Only the best checkpoint is used for outer-test evaluation.

Outer test must not influence best epoch.

## 12. Metric Definitions

Training/validation Accuracy:

```text
number correct / number samples
```

Training/validation Macro F1 is the unweighted mean of class F1 over all five fixed classes.

For each class:

```text
F1 = 2TP / (2TP + FP + FN)
```

If the denominator is zero, class F1 is 0.0.

Phase 7 final reporting must include:

- per-fold Accuracy
- per-fold Macro F1
- mean +/- standard deviation Accuracy across 5 folds
- mean +/- standard deviation Macro F1 across 5 folds
- per-class Precision
- per-class Recall
- per-class F1
- pooled OOF confusion matrix
- multiclass one-vs-rest AUROC where statistically meaningful

Every patient must appear exactly once in pooled OOF evaluation.

Fold summary statistics are frozen as:

- Accuracy mean: arithmetic mean across the 5 outer-fold Accuracy values
- Accuracy standard deviation: sample standard deviation across those 5 values with `ddof=1`
- Macro F1 mean: arithmetic mean across the 5 outer-fold Macro F1 values
- Macro F1 standard deviation: sample standard deviation across those 5 values with `ddof=1`

Pooled OOF per-class Precision, Recall, and F1 must be computed once from the pooled 100-patient out-of-fold predictions using the fixed authoritative class order:

```text
0 NOR
1 DCM
2 HCM
3 MINF
4 RV
```

Per-class metric definitions:

```text
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2TP / (2TP + FP + FN)
```

When a metric denominator is zero, return 0.0 for that class.

The primary reported per-class table must not average per-class Precision, Recall, or F1 across folds.

The pooled OOF confusion matrix must be computed once from the pooled 100-patient OOF predictions using the fixed class order 0..4.

Pooled one-vs-rest AUROC must use pooled OOF class probabilities, not hard class predictions. For each class, treat that class as positive and all other classes as negative, then compute one-vs-rest AUROC from the pooled OOF probabilities for that class.

Report per-class OvR AUROC where mathematically defined. Also report macro-average OvR AUROC as the unweighted mean of the defined per-class AUROCs.

If an AUROC is undefined because the required positive or negative class is absent, record it as undefined rather than inventing a value. For the frozen 100-patient ACDC pooled OOF design, all five classes are expected to be present, but evaluation code must still validate this condition.

## 13. Runtime Environment

```text
Python: 3.12.3
PyTorch: 2.6.0+cu124
torchvision: 0.21.0+cu124
PyTorch CUDA build: 12.4
cuDNN: 92000
GPU: NVIDIA GeForce RTX 3060
GPU memory: 12288 MiB
NVIDIA driver: 580.173.02
```

## 14. CUDA Reproducibility

Every Phase 7 training command must explicitly use:

```text
CUBLAS_WORKSPACE_CONFIG=:4096:8
```

The current shell environment being unset outside training is not a contract violation.

Known limitation: PyTorch reports that `max_pool3d_with_indices_backward_cuda` does not have a deterministic CUDA implementation.

Therefore:

- fixed seeds are required
- identical execution conditions are required
- deterministic algorithms are requested with `warn_only=True`
- Phase 7 CUDA training must not be described as bitwise deterministic

The neural network must not be redesigned because of this warning.

No multi-seed Phase 7 runs are added.

## 15. Scientific Freeze Statement

Once Fold 0 begins, all of the following are frozen across all five folds:

- dataset cohort
- class mapping
- outer splits
- inner splits
- seed policy
- initialization
- model architecture
- preprocessing
- augmentation
- optimizer
- learning rate
- weight decay
- batch size
- maximum epochs
- early stopping
- checkpoint metric
- metric definitions
- runtime execution policy

No tuning based on outer-test results is permitted.

## 16. GitHub / Quality Baseline

Immediately before creating this contract:

```text
PR #1: Phase 7: Full five-fold cross-validation
CI/quality: successful
0 failing
0 pending
```
