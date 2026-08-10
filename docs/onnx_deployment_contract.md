# ONNX Deployment Contract

## Status

**Status: provisional — not frozen.**

The final ONNX deployment contract will be completed after:

* Phase 1 dataset inspection;
* preprocessing design freeze;
* final model training;
* ONNX export;
* PyTorch ↔ ONNX Runtime parity validation.

---

## Deployment Model

Final deployment artifact:

`classifier.onnx`

The ONNX model represents only the neural-network classifier.

Medical-image preprocessing remains outside the ONNX graph.

---

## Input Contract

Planned input name:

`cine_mri`

Planned dtype:

`float32`

Planned tensor layout:

`[N, 2, D, H, W]`

Channel semantics:

* channel 0: ED;
* channel 1: ES.

The final values of `D`, `H`, and `W` will be defined after Phase 1 inspection of the real ACDC dataset.

The input tensor must already satisfy the frozen preprocessing contract before inference.

---

## Output Contract

Output name:

`logits`

Output dtype:

`float32`

Output shape:

`[N, 5]`

The output contains **raw logits**.

Softmax must remain outside the neural network and outside the ONNX graph.

---

## Class Mapping

The authoritative output mapping is stored in:

`configs/class_mapping.json`

Current mapping:

| Output Index | Class |
| -----------: | ----- |
|            0 | NOR   |
|            1 | DCM   |
|            2 | HCM   |
|            3 | MINF  |
|            4 | RV    |

Training, evaluation, ONNX validation, and future deployment code must use the same mapping.

A deployment application must never infer class semantics from output position without this contract.

---

## Inference Flow

The intended deployment flow is:

```text
medical image
      ↓
frozen preprocessing pipeline
      ↓
float32 tensor [N, 2, D, H, W]
      ↓
classifier.onnx
      ↓
raw logits [N, 5]
      ↓
softmax outside the model
      ↓
class probabilities
```

---

## PyTorch ↔ ONNX Runtime Parity

ONNX export alone is not considered sufficient validation.

The project must compare PyTorch and ONNX Runtime outputs using the same real preprocessed ACDC patient input.

The validation must verify that:

* input tensors are identical;
* output shapes are identical;
* class ordering is identical;
* logits are numerically equivalent within an empirically established tolerance.

The parity tolerance will be defined after real export and measurement.

---

## ONNX Opset

The exact ONNX opset is not frozen yet.

Opset 18 may be evaluated as a starting point, but the final version must be compatible with the installed PyTorch, ONNX, and ONNX Runtime stack.

The selected opset must be recorded here after export validation.

---

## Relationship to Preprocessing

The ONNX model does not define:

* MRI orientation normalization;
* physical resampling;
* crop behavior;
* intensity normalization;
* padding;
* ED/ES extraction;
* axis transposition.

Those operations are defined separately in:

`docs/preprocessing_contract.md`

A future deployment implementation must satisfy both contracts.

---

## Future C++ Integration

A future C++/Qt Medical AI Workstation may use ONNX Runtime C++ for inference.

Conceptually:

```text
C++ / ITK preprocessing
        ↓
float32 [1, 2, D, H, W]
        ↓
ONNX Runtime C++
        ↓
classifier.onnx
        ↓
5 raw logits
```

The C++/Qt workstation is outside the scope of this repository.

---

## Final Contract Update

After ONNX export and parity validation, this document must be updated with:

* final model filename;
* ONNX opset;
* fixed `D/H/W`;
* exact input shape;
* exact output shape;
* class mapping version;
* preprocessing contract version;
* validated numerical tolerance;
* tested ONNX Runtime version;
* parity-test results.
