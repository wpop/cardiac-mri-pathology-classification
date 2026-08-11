# Preprocessing Contract

## Status

**Status: frozen after Phase 1 real ACDC validation.**

This contract defines the deterministic preprocessing input to the model. It is based on the official real ACDC training cohort and must not be changed without a new documented data audit.

---

## Authoritative Source Images

The authoritative spatial inputs are the standalone ACDC ED and ES NIfTI volumes:

```text
patientXXX_frameED.nii.gz
patientXXX_frameES.nii.gz
```

The 4D cine files are not authoritative for preprocessing affine/orientation metadata. Phase 1 verified that raw cine ED/ES frames correspond exactly to standalone ED/ES arrays for 100/100 patients, while some 4D cine affine orientations differ from standalone ED/ES metadata.

Segmentation masks are not used.

---

## Orientation And Axis Order

Target orientation:

```text
LPS
```

NiBabel source array order is:

```text
[X, Y, Z]
```

After preprocessing each phase remains:

```text
[X, Y, Z]
```

The explicit model transpose is:

```python
phase.transpose(2, 1, 0)
```

Therefore model spatial order is:

```text
[D, H, W] = [Z, Y, X]
```

ED and ES must use identical orientation, resampling, crop, and padding geometry.

---

## Spatial Resampling

Target physical spacing:

```text
X = 1.50 mm
Y = 1.50 mm
Z = 7.50 mm
```

Use linear interpolation for MRI intensities.

The destination grid preserves the physical image center. ED and ES must use the same destination physical grid and identical spatial transformation.

The production implementation uses deterministic nearest-boundary extension for sub-voxel edge samples during interpolation. This boundary behavior does not create artificial pre-normalization padding.

Use the voxel-center convention validated in Phase 1:

```text
source_center_span_mm = (source_size - 1) * source_spacing_mm
target_size = round(source_center_span_mm / target_spacing_mm) + 1
```

No artificial padding is introduced before normalization.

---

## XY Crop

After physical resampling, apply deterministic geometric FOV-center cropping in XY:

```text
H = 144
W = 144
```

Physical XY field of view:

```text
216 mm x 216 mm
```

No XY padding is permitted. Phase 1 validation found:

```text
XY padding failures: 0/100
```

---

## Normalization

Do not define foreground by intensity thresholding.

Phase 1 rejected these as universal foreground/background definitions:

* `image != 0`;
* global minimum;
* boundary minimum.

The normalization domain is:

```text
all real valid voxels inside the deterministic spatial ROI before artificial Z padding
```

Normalize ED and ES jointly per patient:

1. Conceptually combine valid ED + ES voxel values.
2. Calculate the joint 0.5 percentile.
3. Calculate the joint 99.5 percentile.
4. Clip ED and ES using the same bounds.
5. Calculate joint mean and standard deviation from clipped valid ED+ES values.
6. Normalize both phases with:

```text
normalized = (clipped - mean) / max(std, 1e-6)
```

Epsilon:

```text
1e-6
```

Artificial padding is excluded from normalization statistics.

Phase 1 validation:

```text
normalization std min:    21.196588
normalization std median: 42.025421
normalization std max:   236.211528
```

All normalized joint means and standard deviations passed validation tolerance.

---

## Z Padding

After normalization, center-pad Z to:

```text
D = 14
```

Padding value:

```text
0.0
```

For odd padding:

```text
lower-index side = floor(total_padding / 2)
upper-index side = remainder
```

No final Z cropping is permitted. Phase 1 validation found:

```text
Z cropping failures: 0/100
```

Real resampled depth:

```text
min     7
p05     8
median 12
mean   11.16
p95    13
max    14
```

Real Z padding:

```text
min     0
p05     1
median  2
mean    2.84
p95     6
max     7
```

---

## Output Contract

One patient tensor:

```text
[2, 14, 144, 144]
```

Batch tensor:

```text
[N, 2, 14, 144, 144]
```

Channel semantics:

```text
channel 0 = ED
channel 1 = ES
```

dtype:

```text
float32
```

Memory:

```text
C-contiguous
```

Phase 1 validation:

```text
correct shape: 100/100
float32:       100/100
C-contiguous:  100/100
NaN failures:  0
+Inf failures: 0
-Inf failures: 0
```

---

## Known Residual Shortcut Risk

Initial real acquisition Z-geometry-only diagnostic:

```text
OOF Accuracy approximately 0.390
OOF Macro F1 approximately 0.365
```

Final realized Z-padding diagnostic:

```text
OOF Accuracy = 0.360
OOF Macro F1 = 0.296
Chance accuracy = 0.200
```

This is a known residual acquisition/padding shortcut risk. The frozen policy prioritizes preserving the full resampled cardiac Z stack and avoiding basal/apical cropping.

---

## Reproducibility

Preprocessing is deterministic outside training-only augmentation.

Future golden references from real ACDC patients should be stored under:

```text
tests/fixtures/golden/
```

They will support Python preprocessing regression testing and future Python-to-C++ preprocessing comparison.
