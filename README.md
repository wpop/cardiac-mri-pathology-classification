# Cardiac MRI Pathology Classification

Patient-level cardiac pathology classification from cardiac cine MRI using a custom anisotropy-aware 3D ResNet-18 implemented in PyTorch.

This repository is a standalone medical AI project built around the **ACDC — Automated Cardiac Diagnosis Challenge** dataset.

The project focuses not only on neural-network training, but on the complete medical machine-learning pipeline:

* real cardiac MRI data inspection;
* explicit ED/ES extraction and pairing;
* patient-level leakage prevention;
* physical voxel-spacing awareness;
* explicit MRI orientation handling;
* medically defensible preprocessing;
* custom 3D CNN implementation;
* robust evaluation on a very small medical dataset;
* 3D model explainability;
* reproducible preprocessing;
* ONNX deployment;
* numerical PyTorch ↔ ONNX Runtime validation;
* future Python ↔ C++ preprocessing reproducibility.

The trained model is intended to become the ML foundation for a separate future **C++/Qt Medical AI Workstation**.

The workstation itself is deliberately outside the scope of this repository.

---

## Project Motivation

Cardiac cine MRI contains both anatomical and functional information about the heart.

Many pathology-classification prototypes focus primarily on model accuracy while leaving important engineering questions implicit:

* How are ED and ES paired?
* Are train and evaluation patients truly disjoint?
* Is voxel spacing handled consistently?
* Is slice spacing physically comparable between patients?
* Can padding leak acquisition information?
* Is MRI background handled correctly during normalization?
* Is orientation explicitly defined?
* Can preprocessing later be reproduced in another language?
* Does the exported ONNX model numerically agree with PyTorch?

This project treats those questions as first-class engineering requirements.

The goal is therefore not merely:

```text
MRI
  ↓
Neural Network
  ↓
Class
```

but:

```text
Real ACDC cardiac MRI
        ↓
Dataset inspection
        ↓
Verified ED / ES pairing
        ↓
Physical geometry analysis
        ↓
Deterministic preprocessing
        ↓
Patient-level cross-validation
        ↓
Custom anisotropy-aware 3D ResNet-18
        ↓
Out-of-fold evaluation
        ↓
3D Grad-CAM
        ↓
Final model trained on all labeled patients
        ↓
PyTorch checkpoint
        ↓
ONNX export
        ↓
PyTorch ↔ ONNX Runtime parity validation
        ↓
Future C++ / Qt Medical AI Workstation
```

---

## Research and Engineering Value

### Research Novelty

The research novelty is intentionally modest.

A custom 3D ResNet-18 is **not** a new neural-network architecture.

The project does not claim a novel deep-learning method.

Potentially interesting experimental questions include:

* whether ED+ES joint classification is effective for ACDC pathology classification;
* whether anisotropy-aware depth downsampling is preferable for this dataset;
* whether compatible pretrained initialization improves performance over random initialization;
* whether acquisition variables such as slice count or padding fraction become shortcut signals.

These are useful experimental questions, but they do not make the architecture itself novel.

### Engineering Novelty

The stronger contribution is engineering quality.

The project emphasizes:

* explicit physical-space reasoning;
* data-driven preprocessing decisions;
* deterministic preprocessing;
* patient-level leakage prevention;
* cross-validation appropriate for a very small medical dataset;
* explicit deployment contracts;
* golden preprocessing references;
* ONNX numerical parity;
* future cross-language reproducibility.

### Portfolio Value

The project is intended to demonstrate skills relevant to:

* medical AI;
* computer vision;
* scientific software;
* numerical software;
* PyTorch;
* medical image processing;
* deployment-oriented machine learning;
* C++/Qt medical software integration;
* reproducible ML engineering.

It is especially useful as a bridge between Python medical ML development and future production-oriented C++ inference software.

---

## Dataset

### ACDC

Primary dataset:

**ACDC — Automated Cardiac Diagnosis Challenge**

Official dataset page:

https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html

The expected diagnostic classes are:

| Index | Code | Diagnosis                   |
| ----: | ---- | --------------------------- |
|     0 | NOR  | Normal / healthy control    |
|     1 | DCM  | Dilated cardiomyopathy      |
|     2 | HCM  | Hypertrophic cardiomyopathy |
|     3 | MINF | Myocardial infarction       |
|     4 | RV   | Abnormal right ventricle    |

The commonly used labeled ACDC cohort is often described as approximately:

```text
100 patients
5 diagnostic classes
approximately 20 patients per class
```

These numbers are **not hard-coded assumptions**.

The downloaded dataset must be inspected directly before preprocessing or model geometry decisions are finalized.

### Data Policy

Raw ACDC medical data must never be committed to GitHub.

The repository must not contain:

* raw NIfTI volumes;
* downloaded ACDC archives;
* extracted patient datasets;
* large checkpoints;
* temporary preprocessing outputs;
* local machine paths;
* virtual environments;
* IDE-specific files.

The project uses **real ACDC data** for medical-data validation and project tests.

Synthetic medical datasets and synthetic medical test samples are not used.

---

## Classification Task

The task is **patient-level multiclass cardiac pathology classification**.

The project does not perform per-slice classification.

One patient produces one model prediction.

```text
Patient
  │
  ├── End-Diastolic volume (ED)
  │
  └── End-Systolic volume (ES)
            │
            ▼
       One tensor
            │
            ▼
     One prediction
```

---

## Model Input

ED and ES are represented as two input channels:

```text
channel 0 = ED
channel 1 = ES
```

One patient:

```text
[C=2, D, H, W]
```

A batch:

```text
[N, 2, D, H, W]
```

ED and ES must undergo the **same spatial preprocessing window**.

They must not be independently cropped.

Independent spatial processing could destroy the approximate voxel correspondence between the two cardiac phases.

Conceptually:

```text
ED volume ──┐
            ├── same spatial preprocessing ──┐
ES volume ──┘                                │
                                             ▼
                                      [2, D, H, W]
                                             │
                                             ▼
                                      3D ResNet-18
```

---

## Why ED + ES?

End-diastole and end-systole represent two physiologically important states of the cardiac cycle.

Using both phases allows the model to observe:

* cardiac anatomy;
* chamber morphology;
* myocardial appearance;
* differences between contracted and relaxed states;
* pathology-related changes visible across cardiac phases.

Instead of training two independent networks, the project uses ED and ES as two channels of a single 3D patient representation.

The network therefore learns a joint representation of the two phases.

---

## Classification Output

The network produces **five raw logits**.

```text
0 -> NOR
1 -> DCM
2 -> HCM
3 -> MINF
4 -> RV
```

The authoritative mapping is stored in:

```text
configs/class_mapping.json
```

Conceptually:

```text
3D ResNet-18
      ↓
Global Average Pooling
      ↓
512-dimensional representation
      ↓
Linear(512 → 5)
      ↓
raw logits
```

Softmax is deliberately excluded from the neural-network output.

Training:

```text
raw logits
    ↓
CrossEntropyLoss
```

Inference:

```text
raw logits
    ↓
softmax
    ↓
class probabilities
```

Softmax is applied outside the model only when probabilities are needed.

---

## Primary Neural Network

The primary architecture is a custom:

```text
3D ResNet-18
```

The implementation includes:

```text
BasicBlock3D
ResNet3D18
```

with one class per file:

```text
src/cardiac_pathology/models/basic_block_3d.py
src/cardiac_pathology/models/resnet3d18.py
```

The neural-network implementation does not simply import a complete ResNet from:

* torchvision;
* MONAI;
* a third-party repository;
* a model zoo.

Established architectures may be studied and compatible pretrained weights may be reused, but the residual blocks and network class are implemented in this repository.

---

## Why a Custom 3D ResNet-18?

There are two main reasons.

First, the project is intended to demonstrate understanding of the internal 3D CNN architecture rather than treating the neural network as a black-box dependency.

Second, cardiac short-axis MRI is not geometrically equivalent to ordinary RGB video.

A generic video ResNet commonly treats:

```text
D
H
W
```

more symmetrically than is appropriate for anisotropic medical images.

Our architecture must reflect the actual geometry observed in ACDC.

---

## MRI Anisotropy

Cardiac short-axis MRI can be strongly anisotropic.

Typical in-plane voxel spacing may be much smaller than the distance between slices.

Conceptually:

```text
X spacing ≈ 1–2 mm
Y spacing ≈ 1–2 mm
Z spacing = several millimeters
```

Therefore aggressive early depth downsampling can destroy important anatomical information.

A possible architecture might eventually resemble:

```text
stem:
    stride = (1, 2, 2)

layer1:
    no depth downsampling

layer2:
    stride = (1, 2, 2)

layer3:
    possibly (2, 2, 2)

layer4:
    possibly (2, 2, 2)
```

However, these values are **not frozen**.

They will be finalized only after inspecting the real ACDC geometry.

---

## Phase 1: Dataset Inspection Before Model Decisions

The project deliberately refuses to freeze important preprocessing constants from memory or assumptions.

Before final preprocessing or model geometry is implemented, the real ACDC dataset must be inspected.

For every patient, Phase 1 must examine at least:

* patient identifier;
* diagnostic class;
* ED frame;
* ES frame;
* H;
* W;
* number of slices;
* X spacing;
* Y spacing;
* Z spacing or effective inter-slice spacing;
* physical Z coverage;
* orientation;
* affine;
* intensity distribution;
* zero-voxel fraction;
* NaN/Inf presence.

The inspection must produce statistics such as:

* minimum;
* maximum;
* median;
* percentiles;
* histograms;
* distributions by diagnostic class.

### Phase 1 Questions That Must Be Answered

Training must not begin before the following questions have been investigated.

1. How many labeled patients are really present?
2. Are all five classes present and balanced?
3. What are the real H/W distributions?
4. What is the slice-count distribution?
5. What are the X/Y spacing distributions?
6. What is the Z-spacing distribution?
7. What is the physical stack coverage?
8. Does slice count correlate with disease class?
9. Would padding fraction correlate with disease class?
10. Is MRI background exactly zero?
11. How much background exists?
12. Is `128 × 128` a safe crop?
13. Should Z be resampled or kept native?
14. What should the final D be?
15. What should the final H/W be?
16. Where should depth downsampling begin in ResNet?
17. What exact normalization rule should be frozen?

These questions are design gates, not optional exploratory analysis.

---

## Important Risk 1: Variable Z Spacing

Slice count alone is not sufficient to describe a 3D cardiac MRI volume.

For example:

```text
Patient A:
spacing_z = 5 mm

Patient B:
spacing_z = 10 mm
```

Then three voxels along Z represent approximately:

```text
Patient A → 15 mm
Patient B → 30 mm
```

A Conv3D kernel would therefore observe physically different amounts of anatomy.

An anisotropic neural network alone does not solve this problem.

After real dataset inspection, one explicit engineering decision must be made.

### Option A

Resample Z to a common coarse physical spacing.

### Option B

Keep native Z spacing only if the actual ACDC distribution shows that variation is sufficiently small and preserving native slices is preferable.

The choice must be based on measured ACDC statistics rather than memory or assumptions.

The final decision will be documented in:

```text
docs/preprocessing_contract.md
```

---

## Important Risk 2: MRI Background and Normalization

MRI volumes often contain large background regions.

If background voxels are included blindly in intensity statistics, they can dominate:

* mean;
* standard deviation;
* percentile calculations.

Phase 1 therefore investigates:

* exact zero behavior;
* near-zero behavior;
* zero-voxel fraction;
* NaN/Inf values;
* interpolation effects.

A deterministic foreground rule must then be established.

A possible processing sequence is:

```text
foreground selection
        ↓
percentile clipping
        ↓
foreground-only mean / standard deviation
        ↓
z-score normalization
        ↓
padding
```

A possible working hypothesis is approximately:

```text
0.5th percentile
      ↓
clipping
      ↓
99.5th percentile
```

but these percentile values are not frozen before inspecting real ACDC intensity distributions.

The final preprocessing contract must explicitly define:

* foreground rule;
* percentile values;
* ED/ES normalization policy;
* mean calculation;
* standard deviation calculation;
* epsilon;
* background output value;
* padding value.

Artificial padding must not influence normalization statistics.

---

## Important Risk 3: Padding Shortcut Learning

A CNN may learn acquisition characteristics instead of pathology.

For example:

```text
Patient A:
12 real slices
12 padded slices

Patient B:
23 real slices
1 padded slice
```

If slice count is correlated with diagnosis, the network could potentially learn:

```text
large padding fraction
        ↓
particular diagnostic class
```

instead of learning cardiac pathology.

Phase 1 must therefore calculate:

* slice count per patient;
* slice count by class;
* candidate target D;
* number of padded slices;
* number of cropped slices;
* padding fraction;
* padding fraction by class;
* association between acquisition variables and diagnosis.

The analysis must explicitly answer:

> Does padding look like a plausible shortcut signal?

If the answer is yes, the preferred response is to improve preprocessing rather than add unnecessary neural-network complexity.

Possible changes include:

* target D;
* Z-resampling strategy;
* physical coverage policy;
* crop/pad strategy.

---

## Spatial Input Size

The final input shape must not be hard-coded prematurely.

In particular:

```text
[2, 16, 128, 128]
```

may be treated only as a hypothesis.

Final:

```text
D
H
W
```

must be selected based on the real dataset.

Factors include:

* slice-count distribution;
* spacing distribution;
* physical field of view;
* physical Z coverage;
* heart coverage;
* padding fraction;
* cropping fraction;
* GPU memory.

For depth D, a high percentile of the real slice-count distribution may be considered so that:

* most patients require padding;
* only rare patients require limited cropping.

Basal and apical cardiac anatomy should be preserved whenever possible.

---

## Crop Strategy

The preprocessing pipeline must not claim to use a:

```text
heart-centered crop
```

unless the heart is actually localized.

For the MVP, the starting strategy is a deterministic:

```text
field-of-view-center crop
```

provided dataset inspection confirms that cardiac anatomy remains inside the crop.

If that assumption fails, a simple deterministic non-neural localization strategy may be evaluated, for example temporal-variance localization from cine MRI.

A separate localization neural network is outside the MVP scope.

---

## Orientation Contract

MRI orientation must be explicit.

The project must not rely on vague statements such as:

```text
convert to canonical orientation
```

The preprocessing contract must define:

* target physical orientation;
* RAS+ / LPS convention;
* physical direction of every axis;
* original NumPy array order;
* array order after canonicalization;
* transpose into model `[D, H, W]`;
* ED/ES orientation consistency;
* affine handling.

For example, if NiBabel provides an array in:

```text
[X, Y, Z]
```

and the model expects:

```text
[D, H, W] = [Z, Y, X]
```

then the exact transpose must be documented and tested.

The actual semantics must first be verified from the real dataset.

This is especially important because a future C++ implementation may use ITK, where LPS coordinate conventions are common.

The Python preprocessing contract must therefore contain enough information to prevent mirrored, flipped, or transposed anatomy in a future implementation.

---

## Preprocessing Pipeline

The intended processing order is:

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
deterministic spatial crop
      ↓
foreground-aware normalization
      ↓
padding
      ↓
stack ED + ES
      ↓
explicit array transpose
      ↓
float32 contiguous tensor
```

Training-only augmentation is applied afterwards.

Important invariants:

* ED and ES receive identical spatial transformations;
* artificial padding is applied after normalization;
* final tensor layout is explicit;
* physical orientation is explicit;
* preprocessing is deterministic outside training augmentation.

---

## Data Augmentation

Training augmentation must remain conservative and medically defensible.

Candidate transformations include:

* small rotations;
* small translations;
* small intensity scaling;
* mild intensity jitter;
* mild Gaussian noise.

Left-right flipping is **not automatically enabled**.

Cardiac anatomical chirality must be preserved.

Every augmentation must have a physical or medical justification.

---

## Patient-Level Leakage Prevention

All data partitioning is performed at the **patient level**.

No patient may appear simultaneously in training and evaluation partitions.

The project will include automated tests that verify split disjointness.

This is essential because multiple images or cardiac phases originating from the same patient must never be treated as statistically independent samples across train and test partitions.

---

## Cross-Validation

The project uses:

```text
Stratified patient-level 5-fold cross-validation
```

For every outer fold:

```text
all patients
     ↓
outer training/development patients
     +
outer test patients
```

Outer test patients must never participate in:

* gradient training;
* checkpoint selection;
* early stopping;
* hyperparameter decisions.

Inside the development partition, a leakage-safe validation subset is used for:

* early stopping;
* checkpoint selection.

Each patient must appear exactly once in the pooled out-of-fold evaluation.

---

## Evaluation Metrics

The completed Phase 7 cross-validation reports:

* mean ± sample standard deviation Accuracy;
* mean ± sample standard deviation Macro F1;
* per-class Precision;
* per-class Recall;
* per-class F1;
* pooled out-of-fold confusion matrix;
* pooled multiclass one-vs-rest AUROC.

Phase 7 used the real official ACDC dataset with 100 labeled patients and leakage-safe stratified patient-level 5-fold cross-validation. The pooled out-of-fold evaluation contains 100 unique patients, each exactly once.

Phase 7 final results:

```text
Fold Accuracy: [0.65, 0.50, 0.55, 0.65, 0.50]
Fold Macro F1: [0.6393650793650794, 0.4704761904761905, 0.5556410256410256, 0.6442857142857144, 0.4833333333333333]
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

Because ACDC is small, evaluation methodology is considered as important as the network architecture.

---

## Out-of-Fold Predictions

Reported generalization performance comes from pooled out-of-fold predictions.

Conceptually:

```text
Fold 1 test predictions ─┐
Fold 2 test predictions ─┤
Fold 3 test predictions ─┼─→ pooled OOF predictions
Fold 4 test predictions ─┤
Fold 5 test predictions ─┘
```

Every labeled patient appears exactly once in this pooled evaluation.

---

## Production Model

Cross-validation models are evaluation models.

They are **not** the final deployment model.

During cross-validation, the best epoch for every fold is recorded.

Phase 7 best epochs:

```text
Fold 0 → 27
Fold 1 → 38
Fold 2 → 11
Fold 3 → 25
Fold 4 → 12
```

A deterministic final training duration can then be derived, for example:

```text
median(best_epochs)
```

After the entire pipeline is frozen, one final production model is trained using:

```text
ALL labeled ACDC patients
          ↓
fixed number of epochs derived from CV
          ↓
classifier.pt
          ↓
classifier.onnx
```

No patients are retained from gradient training while claiming that the final model was trained on the full labeled cohort.

The final production model's training performance is not reported as a generalization metric.

Generalization metrics come only from out-of-fold evaluation.

---

## Scratch vs Pretrained Initialization

ACDC is very small relative to the capacity of a 3D ResNet-18.

Initialization is therefore treated as an explicit experiment.

At minimum:

### Experiment A

```text
Custom ResNet3D18
+
random initialization
```

### Experiment B

```text
Same custom ResNet3D18
+
compatible Kinetics pretrained weights
```

The custom network implementation remains unchanged.

The project does not replace it with torchvision's implementation.

For the first convolution, two strategies may be compared.

### B1

Reinitialize the two-channel input stem.

### B2

Mathematically adapt compatible three-channel RGB weights to two input channels.

If B2 is used, the exact transformation must be documented.

The corrected Phase 6 initialization pilot has been completed on one representative fold:

```text
random:     best validation Macro F1 = 0.4222222222222222, best epoch = 8
pretrained: best validation Macro F1 = 0.7677777777777778, best epoch = 44
```

The selected initialization strategy is compatible pretrained Kinetics weights. It is used for:

* full 5-fold cross-validation;
* final production training.

Pretrained weights were not assumed to outperform random initialization; the choice was made from the corrected controlled pilot.

Layers are not frozen by default because RGB video and cardiac MRI belong to substantially different domains.

---

## Visualization

Visualization uses Matplotlib and simple Python scripts.

The project does not require:

* a dashboard;
* an interactive 3D visualization framework;
* a web frontend.

Required visualizations include:

1. representative raw ED slices;
2. representative raw ES slices;
3. processed ED/ES side by side;
4. before/after preprocessing QA;
5. class distribution;
6. H/W/D distributions;
7. X/Y/Z spacing distributions;
8. padding-fraction distribution;
9. acquisition-variable distributions by class;
10. training and validation loss;
11. training and validation Macro F1;
12. pooled out-of-fold confusion matrix;
13. ROC curves where statistically useful;
14. correct predictions;
15. incorrect predictions;
16. compact 3D Grad-CAM.

---

## 3D Grad-CAM

Compact 3D Grad-CAM visualization has been implemented and validated on representative real ACDC patients.

Example presentation:

```text
ED slice
ES slice
Grad-CAM overlay
true class
predicted class
confidence
```

Because ED and ES channels are fused by the network, standard Grad-CAM provides a **joint model attribution**.

It must not be presented as independent ED and ES attribution unless a different attribution method is explicitly implemented.

---

## ONNX Deployment

The final deployment model will be:

```text
classifier.onnx
```

Planned input contract:

```text
name:
cine_mri

dtype:
float32

shape:
[N, 2, D, H, W]
```

D, H, and W become fixed only after Phase 1.

Planned output:

```text
name:
logits

dtype:
float32

shape:
[N, 5]
```

Output semantics:

```text
raw logits
```

Softmax remains outside the ONNX graph.

---

## PyTorch ↔ ONNX Runtime Parity

The exported model is not considered deployment-ready merely because ONNX export succeeds.

Automated numerical comparison must verify:

```text
same preprocessed patient tensor
              │
       ┌──────┴──────┐
       ▼             ▼
    PyTorch     ONNX Runtime
       │             │
       └──────┬──────┘
              ▼
       numerical comparison
```

The numerical tolerance is established empirically for the actual exported model.

---

## Class Mapping Contract

The authoritative class mapping is:

```json
{
  "0": "NOR",
  "1": "DCM",
  "2": "HCM",
  "3": "MINF",
  "4": "RV"
}
```

Stored in:

```text
configs/class_mapping.json
```

Training, evaluation, and deployment code must use the same mapping.

A future deployment application must never guess what output index `0`, `1`, `2`, `3`, or `4` means.

---

## Golden Preprocessing References

The project will create deterministic preprocessing reference tensors under:

```text
tests/fixtures/golden/
```

Example:

```text
patient_x.npy
patient_y.npy
manifest.json
```

These references are intended to capture known preprocessing results for selected real ACDC patients.

The manifest will eventually contain information such as:

* patient ID;
* source file checksum;
* preprocessing-contract version;
* source shape;
* source spacing;
* target shape;
* target spacing;
* orientation;
* normalization parameters;
* output tensor checksum.

These references will later support cross-language preprocessing verification.

---

## Numerical Tolerances

Three numerical comparisons are considered separate problems.

### 1. Python Preprocessing Regression

Checks that the Python preprocessing pipeline remains deterministic.

### 2. PyTorch ↔ ONNX Runtime Parity

Checks that exported model inference agrees numerically with PyTorch.

### 3. Future Python ↔ C++ Preprocessing Parity

This tolerance is **not invented in advance**.

A future C++/ITK implementation may differ slightly because of:

* interpolation implementations;
* floating-point behavior;
* boundary handling;
* image-processing library conventions.

That tolerance will be measured only when a real C++ implementation exists.

---

## Future C++ / Qt Integration

The Python repository is designed to provide an explicit ML deployment contract for a future Medical AI Workstation.

Conceptually:

```text
Medical image
      ↓
C++ / ITK preprocessing
      ↓
[1, 2, D, H, W]
      ↓
ONNX Runtime C++
      ↓
classifier.onnx
      ↓
5 raw logits
      ↓
softmax
      ↓
diagnosis probabilities
      ↓
Qt user interface
```

The future C++ implementation must reproduce the Python preprocessing contract closely enough to provide the model with equivalent input tensors.

This is why orientation, array order, interpolation, normalization, padding, and physical coordinate conventions are explicitly documented rather than hidden inside Python code.

The C++/Qt workstation itself is a separate future project.

---

## Testing Strategy

The project uses:

* pytest;
* ruff;
* mypy.

Tests are intended to validate meaningful medical-ML and engineering behavior rather than increase code coverage artificially.

Planned test areas include:

* ACDC metadata parsing;
* diagnosis labels;
* ED/ES pairing;
* patient-level split disjointness;
* cross-validation stratification;
* spacing extraction;
* orientation conversion;
* explicit array-axis transpose;
* resampling;
* foreground/background handling;
* normalization;
* crop behavior;
* padding behavior;
* padding fraction;
* identical ED/ES spatial windows;
* tensor dtype;
* tensor shape;
* BasicBlock3D behavior;
* ResNet3D18 output;
* NaN/Inf prevention;
* evaluation metrics;
* class mapping;
* golden preprocessing regression;
* ONNX export;
* PyTorch ↔ ONNX Runtime parity.

Medical-data validation and project test fixtures use real ACDC data rather than synthetic medical samples.

---

## Software Engineering Principles

The project follows professional Python engineering practices.

Key rules include:

* object-oriented design where appropriate;
* SOLID where it genuinely improves the design;
* no unnecessary architectural complexity;
* one class per file;
* Python type hints;
* `pathlib.Path`;
* dataclasses where appropriate;
* logging instead of uncontrolled `print()` calls inside reusable modules;
* deterministic random seeds;
* explicit error handling;
* configuration instead of magic constants;
* production logic under `src/`;
* lightweight notebooks only for exploration or visualization.

Important project logic must not be hidden inside Jupyter notebooks.

---

## Project Structure

```text
cardiac-mri-pathology-classification/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── configs/
│   ├── default.yaml
│   └── class_mapping.json
│
├── data/
│   └── README.md
│
├── docs/
│   ├── implementation_plan.md
│   ├── dataset.md
│   ├── preprocessing_contract.md
│   └── onnx_deployment_contract.md
│
├── artifacts/
│   └── .gitkeep
│
├── scripts/
│   ├── inspect_dataset.py
│   ├── run_cross_validation.py
│   ├── run_training.py
│   ├── run_evaluation.py
│   ├── run_visualization.py
│   ├── generate_golden_tensors.py
│   └── export_onnx_model.py
│
├── src/
│   └── cardiac_pathology/
│       ├── data/
│       ├── preprocessing/
│       ├── models/
│       ├── training/
│       ├── evaluation/
│       ├── explainability/
│       ├── deployment/
│       └── utils/
│
├── tests/
│   ├── fixtures/
│   │   └── golden/
│   ├── unit/
│   └── integration/
│
├── .gitignore
├── Makefile
├── pyproject.toml
└── README.md
```

---

## Technology Stack

### Core

* Python 3.12;
* PyTorch;
* NumPy;
* SciPy;
* pandas;
* NiBabel;
* scikit-learn;
* Matplotlib;
* PyYAML;
* ONNX;
* ONNX Runtime.

### Development

* pytest;
* pytest-cov;
* ruff;
* mypy.

MONAI is optional and will only be introduced if a concrete medical-image transform or utility provides clear value over a fragile custom implementation.

The project deliberately excludes unnecessary technologies such as:

* FastAPI;
* Flask;
* Django;
* REST API;
* PostgreSQL;
* AWS;
* Docker deployment;
* JAX;
* Transformers;
* custom CUDA kernels;
* Qt;
* C++.

Qt and C++ belong to the future workstation project, not this repository.

---

## Implementation Roadmap

### Phase 0 — Project Bootstrap — COMPLETE

Create:

* repository structure;
* configuration;
* documentation;
* dependency configuration;
* development tooling;
* CI foundation.

### Phase 1 — Real ACDC Dataset Inspection — COMPLETE

Inspect real ACDC data and answer all geometry, orientation, background, crop, padding, and resampling questions before freezing preprocessing constants.

### Phase 2 — Dataset Indexing and Patient-Level Splits — COMPLETE

Implement:

* ACDC metadata indexing;
* patient representation;
* diagnosis extraction;
* ED/ES pairing;
* stratified patient-level cross-validation;
* leakage tests.

### Phase 3 — Deterministic Preprocessing — COMPLETE

Implement:

* orientation handling;
* physical resampling;
* deterministic spatial processing;
* foreground-aware normalization;
* padding;
* ED/ES stacking;
* explicit axis conversion;
* final tensor construction.

### Phase 4 — Golden Preprocessing Fixtures — COMPLETE

Create deterministic preprocessing references from selected real ACDC patients.

### Phase 5 — Custom 3D ResNet-18 — COMPLETE

Implemented:

* `BasicBlock3D`;
* `ResNet3D18`;
* anisotropy-aware downsampling;
* model tests.

### Phase 6 — Initialization Pilot and Training Infrastructure — COMPLETE

Implemented training infrastructure and completed the corrected initialization pilot comparing:

* random initialization;
* compatible pretrained initialization.

Selected initialization: compatible pretrained Kinetics weights.

### Phase 7 — Full Five-Fold Cross-Validation — COMPLETE

Completed leakage-safe stratified patient-level 5-fold cross-validation.

Produced:

* per-fold best checkpoints;
* out-of-fold predictions;
* per-fold summaries;
* pooled out-of-fold predictions;
* final metrics;
* loss history figure;
* Macro F1 history figure;
* pooled confusion matrix figure;
* pooled OvR ROC figure;
* per-class metrics figure.

Cross-validation metrics remain the generalization metrics.

### Phase 8 — 3D Grad-CAM — COMPLETE

Implemented compact model explainability for representative correctly and incorrectly classified real ACDC patients.

### Phase 9 — Final Production Model — COMPLETE

Completed final production training using all 100 labeled ACDC patients.

The production cohort contained 100 unique patients with the following class distribution:

```text
NOR  = 20
DCM  = 20
HCM  = 20
MINF = 20
RV   = 20
```

The fixed duration was derived from the Phase 7 best epochs:

```text
final_epochs = median(27, 38, 11, 25, 12) = 25
```

The final model uses the custom `ResNet3D18` with compatible pretrained initialization.

No Phase 9 validation subset, test subset, early stopping, or checkpoint selection was used.

Produced:

```text
artifacts/checkpoints/phase9/classifier.pt
artifacts/checkpoints/phase9/phase9_training_summary.json
```

Classifier reload validation passed on a real ACDC input:

```text
input shape  = [1, 2, 14, 144, 144]
output shape = [1, 5]
logits       = finite
```

The `classifier.pt` artifact contains no optimizer state.

Phase 9 training diagnostics are engineering QA only. The Phase 7 pooled out-of-fold metrics remain the reported generalization results.

### Phase 10 — ONNX Deployment

Export:

```text
classifier.onnx
```

Validate:

```text
PyTorch
   ↕
ONNX Runtime
```

Finalize deployment and preprocessing documentation.

---

## Expected Final Artifacts

The completed project should produce:

```text
classifier.pt
classifier.onnx
configs/class_mapping.json
docs/implementation_plan.md
docs/dataset.md
docs/preprocessing_contract.md
docs/onnx_deployment_contract.md
```

as well as:

* dataset inspection statistics;
* dataset inspection figures;
* preprocessing QA figures;
* cross-validation metrics;
* pooled out-of-fold predictions;
* confusion matrix;
* ROC curves where appropriate;
* training histories;
* correct-prediction examples;
* incorrect-prediction examples;
* 3D Grad-CAM figures;
* golden preprocessing references;
* automated tests;
* CI pipeline;
* GitHub-quality documentation.

---

## Current Status

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
Phase 10 — NEXT
```

Completed so far:

* repository structure created;
* Git repository initialized;
* class mapping defined;
* initial project configuration defined;
* `.gitignore` configured;
* `pyproject.toml` configured;
* `Makefile` configured;
* real ACDC Phase 1 inspection completed;
* preprocessing contract frozen;
* data-driven ResNet3D18 spatial geometry frozen;
* deterministic patient-level 5-fold split artifact generated and validated;
* deterministic production preprocessing implemented and validated on all 100 real ACDC patients;
* golden preprocessing references generated and validated under `tests/fixtures/golden/`;
* custom `BasicBlock3D` and `ResNet3D18` implemented;
* corrected Phase 6 initialization pilot completed;
* Phase 7 leakage-safe stratified patient-level 5-fold cross-validation completed;
* Phase 7 pooled OOF predictions, final metrics, and figures generated.
* Phase 8 compact 3D Grad-CAM implemented and validated on representative real ACDC patients;
* Phase 9 final production model trained on all 100 labeled ACDC patients and saved as `artifacts/checkpoints/phase9/classifier.pt`;
* Phase 9 classifier reload validation passed with finite `[1, 5]` logits for a real ACDC input.

Frozen preprocessing summary:

```text
orientation: LPS
spacing:     Z=7.50 mm, Y=1.50 mm, X=1.50 mm
shape:       [2, 14, 144, 144]
crop:        deterministic geometric FOV-center
normalizer:  joint ED/ES p0.5-p99.5 clipping and z-score
padding:     post-normalization center Z padding with value 0.0
```

Frozen model input geometry:

```text
[N, 2, 14, 144, 144] -> raw logits [N, 5]
```

Selected initialization:

```text
compatible pretrained Kinetics weights
```

CUDA execution requested deterministic algorithms with `warn_only=True`. During Phase 9 final production training, PyTorch emitted a non-deterministic implementation warning for `max_pool3d_with_indices_backward_cuda`, so CUDA training is not claimed to be bitwise deterministic.

The next phase is **Phase 10 — ONNX Deployment**.
