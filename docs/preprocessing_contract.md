# Preprocessing Contract

## Status

**Status: provisional — not frozen.**

The final preprocessing contract will be established only after Phase 1 inspection of the real ACDC dataset.

No final values are currently defined for:

* target voxel spacing;
* target depth `D`;
* target height `H`;
* target width `W`;
* Z-resampling policy;
* crop dimensions;
* foreground threshold;
* clipping percentiles;
* normalization parameters.

---

## Model Input

One patient is represented by two cardiac phases:

* channel 0: ED;
* channel 1: ES.

Model input layout:

`[N, 2, D, H, W]`

ED and ES must undergo the same spatial preprocessing.

Independent ED and ES cropping is not allowed.

---

## Required Processing Order

The final preprocessing pipeline will follow this conceptual order:

1. Load NIfTI image.
2. Read affine, spacing, and orientation metadata.
3. Normalize image orientation explicitly.
4. Extract the ED and ES volumes.
5. Apply physical resampling according to the Phase 1 decision.
6. Apply the same deterministic spatial crop to ED and ES.
7. Perform foreground-aware intensity normalization.
8. Apply padding after normalization.
9. Stack ED and ES as two channels.
10. Convert the array explicitly to model `[D, H, W]` order.
11. Produce a contiguous `float32` tensor.

Training-only augmentation is applied afterwards.

---

## Orientation Contract

The final implementation must explicitly document:

* target physical orientation;
* RAS+ / LPS convention;
* physical direction of each axis;
* source NumPy array order;
* array order after orientation normalization;
* exact transpose into model `[D, H, W]`;
* affine handling;
* identical orientation handling for ED and ES.

No implementation may rely only on a vague statement such as "convert to canonical orientation."

The contract must later contain enough information for a future C++/ITK implementation to reproduce Python preprocessing without mirrored or transposed anatomy.

---

## Physical Resampling

The Z-axis policy is not yet decided.

Phase 1 must measure the real:

* X spacing;
* Y spacing;
* Z spacing;
* slice count;
* physical Z coverage.

After inspection, the project will explicitly choose either:

* resampling Z to a common physical spacing; or
* preserving native Z spacing if the measured variation justifies it.

The decision must be based on real ACDC geometry.

---

## Spatial Crop and Padding

The initial candidate strategy is a deterministic field-of-view-center crop.

It must not be described as a heart-centered crop unless the heart is actually localized.

Final crop dimensions will be determined from Phase 1 analysis.

The selected policy must preserve cardiac anatomy, including basal and apical regions whenever possible.

Padding must be analyzed as a possible shortcut signal.

The final contract must specify:

* target spatial dimensions;
* crop behavior;
* padding location;
* padding value;
* expected padding fraction;
* handling of volumes larger than the target dimensions.

---

## Intensity Normalization

Normalization must be background-aware.

Phase 1 must first determine:

* whether background voxels are exactly zero;
* near-zero behavior;
* background fraction;
* intensity distribution;
* NaN/Inf presence;
* interpolation effects.

The final normalization contract must specify:

* foreground selection rule;
* clipping percentiles;
* whether ED and ES are normalized separately;
* mean calculation;
* standard deviation calculation;
* numerical epsilon;
* background output value;
* padding value.

Artificial padding must be applied **after** normalization so that padding does not influence intensity statistics.

---

## Output Contract

The final preprocessing output must be:

* dtype: `float32`;
* contiguous in memory;
* tensor layout: `[2, D, H, W]` for one patient;
* batch layout: `[N, 2, D, H, W]`.

The final values of `D`, `H`, and `W` will be frozen only after Phase 1.

---

## Reproducibility

The preprocessing implementation must be deterministic outside training-only augmentation.

Golden preprocessing references from real ACDC patients will later be stored under:

`tests/fixtures/golden/`

They will support:

* Python preprocessing regression testing;
* future Python ↔ C++ preprocessing comparison.

The future Python ↔ C++ numerical tolerance will not be invented in advance. It will be measured after a real C++/ITK implementation exists.

---

## Phase 1 Update Requirement

After real ACDC inspection, this document must be updated with the final:

* orientation convention;
* axis order;
* target spacing;
* Z-resampling decision;
* crop dimensions;
* target `D/H/W`;
* foreground rule;
* clipping percentiles;
* normalization formula;
* padding behavior;
* interpolation method;
* boundary behavior;
* voxel-center convention;
* output dtype.

Until those decisions are supported by real dataset measurements, this contract remains provisional.
