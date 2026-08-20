# ONNX Deployment Contract

## Status

**Status: frozen/final after successful Phase 10 export and parity validation.**

This document defines the finalized ONNX deployment contract for the Phase 9
cardiac MRI pathology classifier. It records only verified Phase 10 deployment
facts and does not change or reinterpret the frozen preprocessing contract.

---

## Artifacts

Source checkpoint:

`artifacts/checkpoints/phase9/classifier.pt`

Final ONNX artifact:

`artifacts/deployment/classifier.onnx`

Architecture:

`ResNet3D18`

Export API:

`torch.onnx.export(..., dynamo=True)`

ONNX opset:

`18`

The ONNX model represents only the neural-network classifier.

---

## Input Contract

Input name:

`cine_mri`

Input dtype:

`float32`

Input tensor layout:

`[N, 2, 14, 144, 144]`

Only the batch dimension `N` is dynamic. Channel, depth, height, and width are
fixed as `[2, 14, 144, 144]`.

Channel semantics:

* channel 0: ED;
* channel 1: ES.

The input tensor must already satisfy the frozen preprocessing contract before
inference.

---

## Output Contract

Output name:

`logits`

Output dtype:

`float32`

Output shape:

`[N, 5]`

The output contains raw, unnormalized class logits. Softmax remains outside the
ONNX graph.

---

## Class Mapping

The authoritative output mapping is stored in:

`configs/class_mapping.json`

Frozen mapping:

| Output Index | Class |
| -----------: | ----- |
|            0 | NOR   |
|            1 | DCM   |
|            2 | HCM   |
|            3 | MINF  |
|            4 | RV    |

Training, evaluation, ONNX validation, and future deployment code must use the
same mapping. A deployment application must never infer class semantics from
output position without this contract.

---

## Inference Flow

The deployment flow is:

```text
medical image
      ↓
frozen preprocessing pipeline
      ↓
float32 tensor [N, 2, 14, 144, 144]
      ↓
artifacts/deployment/classifier.onnx
      ↓
raw logits [N, 5]
      ↓
softmax outside the model
      ↓
class probabilities
```

---

## Relationship to Preprocessing

Medical-image preprocessing remains outside the ONNX graph and is defined by:

`docs/preprocessing_contract.md`

The ONNX model does not define:

* MRI orientation normalization;
* physical resampling;
* crop behavior;
* intensity normalization;
* padding;
* ED/ES extraction;
* axis transposition.

A deployment application must satisfy both this ONNX deployment contract and the
frozen preprocessing contract.

---

## Verified Software Stack

The Phase 10 export and parity validation used:

| Component    | Version      |
| ------------ | ------------ |
| PyTorch      | 2.6.0+cu124  |
| ONNX         | 1.22.0       |
| ONNX Runtime | 1.28.0       |
| ONNX Script  | 0.5.3        |
| ONNX IR      | 0.1.15       |

ONNX Script and ONNX IR are pinned for compatibility with the verified PyTorch
2.6 exporter stack.

ONNX checker passed.

ONNX Runtime execution passed for batch size 1 and batch size 2.

---

## PyTorch ↔ ONNX Runtime Parity

Phase 10 validated deployment numerical parity by comparing the Phase 9 PyTorch
checkpoint and the exported ONNX model on the exact same already-preprocessed
real ACDC tensors.

Validated patient set:

| Patient ID | Class |
| ---------- | ----- |
| patient001 | DCM   |
| patient021 | HCM   |
| patient041 | MINF  |
| patient061 | NOR   |
| patient081 | RV    |

All PyTorch and ONNX predicted class indices matched.

Worst measured errors across the validated patient set:

| Metric                                | Value                  |
| ------------------------------------- | ---------------------- |
| maximum absolute error                | 9.5367431640625e-07    |
| mean absolute error                   | 6.437301749429025e-07  |
| maximum relative error                | 6.35185813280259e-07   |

Final absolute logit parity tolerance:

`1e-5`

The predicted class index must match exactly.

This is deployment numerical parity validation, not a model generalization
evaluation. Phase 7 pooled out-of-fold results remain the generalization
evidence.

---

## Future C++ Integration

A future C++/Qt Medical AI Workstation may use ONNX Runtime C++ for inference.

Conceptually:

```text
C++ / ITK preprocessing
        ↓
float32 [1, 2, 14, 144, 144]
        ↓
ONNX Runtime C++
        ↓
artifacts/deployment/classifier.onnx
        ↓
5 raw logits
```

The portable handoff package for this integration is:

`artifacts/deployment/package/cardiac_mri_pathology/`

Package contents are divided by deployment role:

| Role | Files |
| ---- | ----- |
| runtime-required | `classifier.onnx`, `class_mapping.json`, `deployment.json` |
| validation-only | `golden/` |
| developer-only | `docs/` |

The C++/Qt workstation is outside the scope of this repository.
