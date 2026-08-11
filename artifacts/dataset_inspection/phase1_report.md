# Phase 1 Report

## 1. Cohort

Phase 1 used the official real ACDC training cohort under `data/raw/acdc/training`.

Measured cohort:

```text
100 patients
20 DCM
20 HCM
20 MINF
20 NOR
20 RV
```

Primary artifacts include `patient_geometry.csv`, `orientation_correspondence.csv`, `motion_localization.csv`, `spatial_xy_candidates.csv`, `spatial_z_candidates.csv`, and `final_preprocessing_validation.csv`.

## 2. Geometry

Standalone ED/ES volumes exist for all patients and are patient-wise geometry-consistent.

Measured geometry included source depth 6 to 18 slices, standalone ED/ES LPS orientation for 100/100 patients, XY spacing approximately 0.703 mm to 1.920 mm, and Z spacing values including 5.0, 6.5, 7.0, and 10.0 mm.

## 3. Orientation Audit

Standalone ED/ES are the authoritative spatial images.

Raw ED/ES frames extracted from the 4D cine match standalone ED/ES arrays exactly for 100/100 patients. The 4D cine affine is not used as the authoritative preprocessing spatial reference.

Artifact: `orientation_correspondence.csv`

## 4. Acquisition Shortcut Finding

Acquisition geometry contains measurable diagnosis-related information.

```text
Z geometry only OOF Accuracy approximately 0.390
Z geometry only OOF Macro F1 approximately 0.365
Chance accuracy 0.200
```

This is a shortcut-learning risk, not pathology performance.

## 5. Rejected Localization Methods

Rejected:

* temporal-variance connected-component localization;
* ED-ES connected-component localization.

Artifacts: `temporal_variance_localization.csv`, `motion_localization.csv`, and `motion_localization_qa_contact_sheet.png`.

## 6. Frozen FOV-Center Localization

Localization is frozen as deterministic geometric FOV-center crop. No motion-based localization is used.

## 7. Frozen XY Spacing / H/W

```text
X = 1.50 mm
Y = 1.50 mm
H = 144
W = 144
Physical XY FOV = 216 mm x 216 mm
```

Artifact: `spatial_xy_candidates.csv`

## 8. Frozen Z Spacing / D

```text
Z = 7.50 mm
D = 14
Final center-to-center Z span = (14 - 1) * 7.5 = 97.5 mm
```

Artifact: `spatial_z_candidates.csv`

## 9. Intensity / Background Findings

Rejected as universal foreground/background definitions:

* `image != 0`;
* global minimum;
* boundary minimum.

ED/ES volumes contain no NaN, +Inf, -Inf, or negative intensities.

## 10. Frozen Normalization

Normalization uses all real valid voxels inside the deterministic spatial ROI before artificial Z padding.

ED and ES are normalized jointly per patient:

```text
lower = joint ED+ES p0.5
upper = joint ED+ES p99.5
clipped = clip(values, lower, upper)
normalized = (clipped - joint_mean) / max(joint_std, 1e-6)
```

## 11. Padding Validation

After normalization, Z is center-padded to `D=14` with value `0.0`.

```text
XY padding failures: 0/100
Z cropping failures: 0/100
correct shape: 100/100
float32: 100/100
C-contiguous: 100/100
NaN/+Inf/-Inf failures: 0
```

Real resampled depth:

```text
min 7, p05 8, median 12, mean 11.16, p95 13, max 14
```

Real Z padding:

```text
min 0, p05 1, median 2, mean 2.84, p95 6, max 7
```

Artifact: `final_preprocessing_validation.csv`

## 12. Residual Shortcut Limitation

Final realized Z-padding diagnostic:

```text
OOF Accuracy = 0.360
OOF Macro F1 = 0.296
Chance accuracy = 0.200
```

This residual acquisition/padding shortcut risk remains documented. The frozen policy prioritizes preserving the full resampled cardiac Z stack and avoiding basal/apical cropping.

## 13. Final Tensor Contract

Source array order:

```text
[X, Y, Z]
```

Model transpose:

```python
transpose(2, 1, 0)
```

One patient:

```text
[2, 14, 144, 144]
```

Batch:

```text
[N, 2, 14, 144, 144]
```

dtype is `float32`; tensors must be C-contiguous.

## 14. Frozen ResNet Depth-Downsampling Geometry

Input:

```text
[N, 2, 14, 144, 144]
```

Stem:

```text
Conv3d kernel_size=(3,7,7), stride=(1,2,2), padding=(1,3,3)
No max-pool
```

Stage geometry:

```text
stem   -> [14,72,72]
layer1 -> [14,72,72]
layer2 -> [14,36,36]
layer3 -> [7,18,18]
layer4 -> [4,9,9]
```

Then `AdaptiveAvgPool3d((1,1,1))`, flatten to 512, and `Linear(512,5)` raw logits. No softmax is included in the model.

## 15. Phase 1 Conclusion

Phase 1 is complete. The preprocessing contract and ResNet3D18 spatial/downsampling geometry are frozen.

## 16. Next Phase

```text
Phase 2 — Dataset Indexing and Patient Splits
```
