# Implementation Plan

## Project Goal

Build a standalone medical AI pipeline for patient-level cardiac pathology classification from real ACDC cardiac cine MRI using a custom 3D ResNet-18.

The final Python model will later serve as the ML foundation for a separate C++/Qt Medical AI Workstation.

This project does not reuse segmentation masks, segmentation outputs, existing U-Net checkpoints, or segmentation-derived measurements.

---

## Phase 0 — Project Bootstrap — COMPLETE

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

## Phase 2 — Dataset Indexing and Patient Splits — COMPLETE

Phase 2 produced:

* immutable patient-level representation;
* ACDC metadata parser;
* deterministic patient index;
* ED/ES pairing from `Info.cfg`;
* leakage-safe stratified outer 5-fold splits;
* deterministic split artifact;
* automated real-ACDC validation.

The split artifact is:

```text
artifacts/dataset_splits/acdc_5fold_seed42.json
```

All splits are patient-level and leakage-safe.

---

## Phase 3 — Deterministic Preprocessing — COMPLETE

Phase 3 implemented and validated the final preprocessing pipeline:

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
joint valid-data normalization
      ↓
padding
      ↓
stack ED + ES
      ↓
transpose to [D, H, W]
      ↓
float32 tensor
```

ED and ES undergo the same spatial preprocessing grid and crop. The final output contract is `[2, 14, 144, 144]`, `float32`, C-contiguous.

Validation artifact:

```text
artifacts/preprocessing_validation/phase3_validation.csv
```

---

## Phase 4 — Golden Preprocessing References — COMPLETE

Phase 4 produced deterministic preprocessing references from selected real ACDC patients.

Store them under:

```text
tests/fixtures/golden/
```

These references include byte-stable tensors, source checksums, preprocessing metadata, and
an automated validator for preprocessing regression testing and later Python ↔ C++
reproducibility work.

---

## Phase 5 — Custom 3D ResNet-18 — COMPLETE

Implemented:

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

## Phase 6 — Training Infrastructure and Initialization Pilot — COMPLETE

Phase 6 implemented training infrastructure and completed the corrected initialization pilot.

The corrected pilot compared:

* random initialization;
* compatible pretrained initialization where technically appropriate.

Results:

```text
random:     best validation Macro F1 = 0.4222222222222222, best epoch = 8
pretrained: best validation Macro F1 = 0.7677777777777778, best epoch = 44
```

Selected initialization: compatible pretrained Kinetics weights.

Phase 6A has added reusable infrastructure for deterministic seeding, inner
validation splitting, early stopping, validation Macro F1 checkpoint selection,
training and validation loops, checkpoint metadata, and initialization
compatibility reporting.

---

## Phase 7 — Five-Fold Cross-Validation — COMPLETE

Phase 7 completed leakage-safe stratified patient-level 5-fold cross-validation on the real official ACDC dataset with 100 labeled patients.

Inside every outer training fold, a separate validation subset was used for:

* early stopping;
* checkpoint selection.

The pooled out-of-fold evaluation contains 100 unique patients, each exactly once.

Fold Accuracy:

```text
[0.65, 0.50, 0.55, 0.65, 0.50]
```

Fold Macro F1:

```text
[0.6393650793650794, 0.4704761904761905, 0.5556410256410256, 0.6442857142857144, 0.4833333333333333]
```

Summary:

```text
Accuracy mean ± sample std: 0.5700 ± 0.0758
Macro F1 mean ± sample std: 0.5586 ± 0.0826
Pooled Macro OvR AUROC: 0.83375
Best epochs: [27, 38, 11, 25, 12]
```

Pooled per-class Precision / Recall / F1:

```text
NOR:  0.3793 / 0.5500 / 0.4490
DCM:  0.9375 / 0.7500 / 0.8333
HCM:  0.8000 / 0.6000 / 0.6857
MINF: 0.4000 / 0.4000 / 0.4000
RV:   0.5500 / 0.5500 / 0.5500
```

Per-class OvR AUROC:

```text
NOR:  0.8100
DCM:  0.9200
HCM:  0.86625
MINF: 0.683125
RV:   0.889375
```

Generated outputs include:

* per-fold best checkpoints;
* per-fold OOF predictions and summaries;
* pooled OOF predictions;
* final metrics;
* loss history figure;
* Macro F1 history figure;
* pooled confusion matrix figure;
* pooled OvR ROC figure;
* per-class metrics figure.

Cross-validation metrics remain the generalization metrics. Phase 7 CUDA execution must not be described as bitwise deterministic.

---

## Phase 8 — 3D Grad-CAM — COMPLETE

Phase 8 implemented and validated compact 3D Grad-CAM visualization for representative real ACDC patients.

Show:

* ED slice;
* ES slice;
* Grad-CAM overlay;
* true class;
* predicted class;
* confidence.

---

## Phase 9 — Final Production Model — COMPLETE

Phase 9 used the best-epoch information from cross-validation to derive a deterministic final training duration:

```text
best_epochs = [27, 38, 11, 25, 12]
final_epochs = median(best_epochs) = 25
```

One production model was trained using all 100 labeled ACDC patients:

```text
total_patient_entries = 100
unique_patient_ids = 100
NOR  = 20
DCM  = 20
HCM  = 20
MINF = 20
RV   = 20
```

The final model retained the custom `ResNet3D18` architecture and used compatible pretrained initialization.

No Phase 9 validation subset, test subset, early stopping, or checkpoint selection was used.

Produced:

```text
artifacts/checkpoints/phase9/classifier.pt
artifacts/checkpoints/phase9/phase9_training_summary.json
```

The final classifier artifact reloaded successfully and passed real ACDC inference validation:

```text
input shape  = [1, 2, 14, 144, 144]
output shape = [1, 5]
logits       = finite
```

The `classifier.pt` artifact contains no optimizer state.

Phase 9 training diagnostics are engineering QA only. Cross-validation metrics remain the reported generalization metrics.

CUDA execution is not claimed to be bitwise deterministic because PyTorch warned that `max_pool3d_with_indices_backward_cuda` lacks a deterministic implementation under `warn_only=True`.

---

## Phase 10 — ONNX Deployment — COMPLETE

Phase 10 exported the Phase 9 production checkpoint:

```text
artifacts/checkpoints/phase9/classifier.pt
```

to the final ONNX artifact:

```text
artifacts/deployment/classifier.onnx
```

Final ONNX interface:

```text
Input:
cine_mri
float32
[N, 2, 14, 144, 144]
dynamic batch only

Output:
logits
float32
[N, 5]
raw logits
```

Softmax remains outside the model.

Export used ONNX opset 18.

ONNX checker passed.

ONNX Runtime execution passed for batch sizes 1 and 2.

Real ACDC PyTorch ↔ ONNX Runtime parity passed on:

```text
patient001 / DCM
patient021 / HCM
patient041 / MINF
patient061 / NOR
patient081 / RV
```

Predicted class index matched for all five patients.

Worst maximum absolute error:

```text
9.5367431640625e-07
```

Final absolute logit tolerance:

```text
1e-5
```

Phase 10 did not retrain or tune the model.

Phase 10 parity validation is deployment numerical validation only. Phase 7
pooled out-of-fold evaluation remains the generalization evidence.

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
Python AI project functionally complete pending final Phase 10 quality gate / PR / merge / exact-main CI verification.
```

Phase status:

```text
Phase 0 — COMPLETE
Phase 1 — COMPLETE
Phase 2 — COMPLETE
Phase 3 — COMPLETE
Phase 4 — COMPLETE
Phase 5 — COMPLETE
Phase 6 — COMPLETE
Phase 7 — COMPLETE
Phase 8 — COMPLETE
Phase 9 — COMPLETE
Phase 10 — COMPLETE
```

Preprocessing constants are frozen.

Custom `BasicBlock3D` and `ResNet3D18` are implemented.

The corrected Phase 6 initialization pilot selected compatible pretrained Kinetics weights.

Phase 7 leakage-safe stratified patient-level 5-fold cross-validation is complete.

Cross-validation metrics remain the generalization metrics.

Phase 8 compact 3D Grad-CAM is complete and validated on representative real ACDC patients.

Phase 9 final production training is complete using all 100 labeled ACDC patients.

Phase 10 ONNX deployment is complete. The Python AI project is functionally
complete pending final Phase 10 quality gate / PR / merge / exact-main CI
verification.
