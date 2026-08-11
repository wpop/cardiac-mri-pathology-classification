# Dataset

## Primary Dataset

This project uses the **ACDC — Automated Cardiac Diagnosis Challenge** dataset for patient-level cardiac pathology classification from cardiac cine MRI.

Official dataset page:

https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html

Raw ACDC data must never be committed to Git.

---

## Diagnostic Classes

The authoritative class mapping is stored in:

```text
configs/class_mapping.json
```

| Index | Code | Diagnosis                   |
| ----: | ---- | --------------------------- |
|     0 | NOR  | Normal / healthy control    |
|     1 | DCM  | Dilated cardiomyopathy      |
|     2 | HCM  | Hypertrophic cardiomyopathy |
|     3 | MINF | Myocardial infarction       |
|     4 | RV   | Abnormal right ventricle    |

---

## Measured Phase 1 Facts

The official real ACDC training cohort contains:

```text
100 patients
20 DCM
20 HCM
20 MINF
20 NOR
20 RV
```

Standalone ED/ES findings:

* standalone ED/ES volumes exist for all 100 patients;
* standalone ED/ES are the authoritative spatial images;
* standalone ED/ES orientation is LPS for 100/100 patients;
* patient-wise ED/ES geometry is consistent.

4D cine finding:

* raw ED/ES frames extracted from the 4D cine match standalone ED/ES arrays exactly for 100/100 patients;
* the 4D cine affine is not the authoritative preprocessing spatial reference.

Measured geometry:

* depth ranges from 6 to 18 source slices;
* XY spacing varies substantially, approximately 0.703 mm to 1.920 mm;
* Z spacing includes 5.0, 6.5, 7.0, and 10.0 mm;
* source orientations for standalone ED/ES are all LPS.

Z geometry by class showed measurable class correlation. A leakage-safe geometry-only diagnostic baseline found:

```text
Z geometry only OOF Accuracy approximately 0.390
Z geometry only OOF Macro F1 approximately 0.365
Balanced chance accuracy 0.200
```

Background/intensity findings:

* ED/ES contain no NaN, +Inf, -Inf, or negative intensities;
* exact zero is not a reliable universal background marker;
* global minimum is not a reliable background definition;
* boundary intensity is not a reliable background definition.

---

## Frozen Phase 1 Design Decisions

Model samples are patient-level ED/ES pairs:

```text
channel 0 = ED
channel 1 = ES
```

Frozen preprocessing:

```text
target orientation: LPS
target spacing:     X=1.50 mm, Y=1.50 mm, Z=7.50 mm
localization:       deterministic geometric FOV-center crop
final tensor:       [2, 14, 144, 144]
batch tensor:       [N, 2, 14, 144, 144]
```

Rejected localization methods:

* temporal-variance connected-component localization;
* ED-ES connected-component localization.

Normalization decision:

* use all real valid voxels inside the deterministic ROI before artificial Z padding;
* jointly normalize ED and ES per patient;
* clip at joint p0.5 and p99.5;
* use one joint mean/std for both phases.

Known residual shortcut:

```text
Final realized Z-padding diagnostic OOF Accuracy = 0.360
Final realized Z-padding diagnostic OOF Macro F1 = 0.296
Chance accuracy = 0.200
```

This residual acquisition/padding shortcut risk must be considered during evaluation.

---

## Data Policy

Do not commit:

* NIfTI volumes;
* downloaded archives;
* extracted raw patient data;
* temporary preprocessing outputs;
* large checkpoints.

Repository artifacts should be code, configuration, documentation, and approved small analysis outputs.
