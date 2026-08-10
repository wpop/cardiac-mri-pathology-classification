# Dataset

## Primary Dataset

This project uses the **ACDC — Automated Cardiac Diagnosis Challenge** dataset as the primary source of cardiac cine MRI data.

Official dataset page:

https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html

The dataset is used for patient-level cardiac pathology classification.

---

## Expected Diagnostic Classes

The expected diagnostic groups are:

| Index | Code | Diagnosis                   |
| ----: | ---- | --------------------------- |
|     0 | NOR  | Normal / healthy control    |
|     1 | DCM  | Dilated cardiomyopathy      |
|     2 | HCM  | Hypertrophic cardiomyopathy |
|     3 | MINF | Myocardial infarction       |
|     4 | RV   | Abnormal right ventricle    |

The authoritative class mapping is stored in:

```text
configs/class_mapping.json
```

The commonly used labeled ACDC cohort is often described as approximately 100 patients with roughly 20 patients per class.

These numbers must not be hard-coded.

The real downloaded dataset will be inspected directly.

---

## Model Sample

One model sample represents one patient.

The project uses:

```text
channel 0 = ED
channel 1 = ES
```

where:

* ED is the end-diastolic cardiac volume;
* ES is the end-systolic cardiac volume.

The two phases are paired for the same patient and later combined into one tensor:

```text
[2, D, H, W]
```

The project does not perform per-slice classification.

---

## Phase 1 Dataset Inspection

Before preprocessing constants are finalized, the real ACDC dataset must be inspected.

For every labeled patient, record:

* patient ID;
* diagnostic class;
* ED frame;
* ES frame;
* image dimensions;
* slice count;
* X spacing;
* Y spacing;
* Z spacing or effective inter-slice spacing;
* physical Z coverage;
* image orientation;
* affine information;
* intensity characteristics;
* fraction of zero or background voxels;
* NaN/Inf presence.

The inspection must also analyze distributions by diagnostic class.

---

## Decisions Driven by the Real Dataset

Phase 1 must determine:

* final model depth `D`;
* final spatial dimensions `H` and `W`;
* whether Z resampling is required;
* target physical spacing;
* deterministic crop strategy;
* foreground/background rule;
* normalization strategy;
* expected padding fraction;
* whether padding or slice count could become a shortcut signal;
* where depth downsampling should begin in the 3D ResNet.

These decisions must be based on measured ACDC data rather than assumptions.

---

## Data Policy

Raw ACDC data must never be committed to Git.

Do not commit:

* NIfTI volumes;
* downloaded archives;
* extracted raw patient data;
* temporary preprocessing outputs.

The repository keeps only source code, configuration, documentation, small reproducibility metadata, and explicitly approved reference artifacts.

Medical-data tests and preprocessing validation use real ACDC data rather than synthetic medical datasets.

---

## Current Status

The ACDC dataset has not yet been inspected in this project.

This document will be updated after Phase 1 with the measured patient counts, class distribution, geometry statistics, spacing statistics, orientation findings, background analysis, and preprocessing decisions.
