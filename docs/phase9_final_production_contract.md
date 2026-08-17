# Phase 9 Final Production Model Contract

## Status

**Status: final production training completed and validated.**

This contract defines the scientific and reproducibility requirements for Phase 9.

## Final Training Duration

Phase 7 produced the following best epochs:

```text
Fold 0 = 27
Fold 1 = 38
Fold 2 = 11
Fold 3 = 25
Fold 4 = 12
```

The deterministic Phase 9 epoch-derivation rule is:

```text
final_epochs = median(best_epochs)
```

Sorted values:

```text
11, 12, 25, 27, 38
```

Therefore:

```text
final_epochs = 25
```

The final epoch count is frozen before production training begins.

It must not be changed in response to Phase 9 training loss, accuracy, Macro F1, or any other observed training diagnostic.

## Scientific Role

Phase 7 cross-validation models estimate generalization performance.

Phase 9 trains one production model using all labeled ACDC patients.

Phase 9 training metrics are engineering diagnostics only and must not be reported as generalization performance.

The reported generalization evidence remains the Phase 7 pooled out-of-fold evaluation.

## Frozen Production Requirements

The final production run must use:

* all 100 labeled ACDC patients;
* custom `ResNet3D18`;
* `input_channels = 2`;
* `num_classes = 5`;
* `initialization_strategy = pretrained`;
* `seed = 42`;
* `final_epochs = 25`;
* frozen deterministic preprocessing;
* Phase 7 training-only augmentation policy;
* Phase 7 optimizer, loss, and training conventions.

No labeled patient may be held out for Phase 9 validation or testing.

No early stopping, hyperparameter tuning, architecture changes, preprocessing changes, initialization changes, or augmentation changes are permitted during Phase 9.

## Production Cohort

The Phase 9 production cohort is the complete labeled ACDC training cohort.

Exact patient set:

```text
patient001 through patient100
```

Persisted Phase 7 split evidence confirms:

```text
total_patient_entries = 100
unique_patient_ids = 100
duplicate_patient_ids = 0
```

Class distribution:

```text
NOR  = 20
DCM  = 20
HCM  = 20
MINF = 20
RV   = 20
```

No labeled patient is reserved for Phase 9 validation or testing.

The production patient set is derived from the persisted Phase 7 patient-level split evidence and must not be regenerated opportunistically.

## Inherited Training Configuration

Phase 9 inherits the frozen Phase 7 training conventions:

```text
seed = 42
initialization_strategy = pretrained
batch_size = 2
optimizer = AdamW
learning_rate = 1e-4
weight_decay = 1e-4
loss = CrossEntropyLoss
num_workers = 0
device = cuda
```

Training-only augmentation remains:

```text
rotation = +/-5 degrees in-plane
translation = +/-5 mm in-plane
intensity scale = 0.95 to 1.05
Gaussian noise std = 0.01
transform probability = 0.5
shared ED/ES spatial transform
no left-right flip
no Z rotation
no Z translation
```

Phase 7 validation-specific settings such as early stopping are not part of Phase 9 production training.

## Production Output

The final production artifact path is:

```text
artifacts/checkpoints/phase9/classifier.pt
```

The artifact must be deployment-oriented and must not require optimizer state.

Training diagnostics and reproducibility metadata may be stored separately under:

```text
artifacts/checkpoints/phase9/
```

Phase 7 artifacts must not be overwritten.

## Final Artifact

The production model artifact must be named:

```text
classifier.pt
```

ONNX export is outside Phase 9 and belongs to Phase 10.

## Production Training Result

Final production training completed successfully under the frozen Phase 9 contract.

The production run used the complete labeled ACDC cohort:

```text
total_patient_entries = 100
unique_patient_ids = 100
validation_set = none
test_set = none
```

The verified class distribution was:

```text
NOR  = 20
DCM  = 20
HCM  = 20
MINF = 20
RV   = 20
```

The completed training configuration was:

```text
final_epochs = 25
initialization_strategy = pretrained
optimizer = AdamW
learning_rate = 1e-4
weight_decay = 1e-4
batch_size = 2
early_stopping = none
checkpoint_selection = none
```

Phase 9 training diagnostics are retained as engineering QA only. They must not be
reported as generalization performance. The Phase 7 pooled out-of-fold metrics
remain the generalization evidence for the project.

## Final Artifact Validation

The completed Phase 9 artifacts are:

```text
classifier = artifacts/checkpoints/phase9/classifier.pt
training_summary = artifacts/checkpoints/phase9/phase9_training_summary.json
```

The final classifier artifact was reloaded successfully.

Validation of the reloaded classifier confirmed:

```text
real_acdc_validation_input_shape = [1, 2, 14, 144, 144]
reloaded_model_output_shape = [1, 5]
logits = finite
optimizer_state_dict = absent
```

The classifier artifact is deployment-oriented and does not contain optimizer
state. Reproducibility and training diagnostics are stored separately in the
Phase 9 training summary.

## Reproducibility Note

Phase 9 used the frozen seed and deterministic training setup defined in this
contract. During final production training, PyTorch emitted a deterministic
algorithm warning for `max_pool3d_with_indices_backward_cuda` with
`warn_only=True`.

This warning means the run used deterministic controls where PyTorch could apply
them, but Phase 9 does not claim bitwise deterministic CUDA training.
