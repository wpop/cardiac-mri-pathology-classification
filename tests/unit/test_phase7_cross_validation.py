"""Unit tests for Phase 7 cross-validation orchestration."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

import cardiac_pathology.training.phase7_cross_validation as phase7_module
from cardiac_pathology.training.initialization import CompatibilityReport
from cardiac_pathology.training.phase7_cross_validation import (
    EXPECTED_CUBLAS_WORKSPACE_CONFIG,
    Phase7CrossValidationOrchestrator,
    Phase7FoldArtifacts,
    Phase7FoldResult,
    Phase7FoldRunner,
    Phase7FoldSplit,
    Phase7PredictionRecord,
    evaluate_outer_test,
    load_phase7_fold_splits,
    phase7_artifacts,
    select_phase7_folds,
    validate_cuda_runtime_preconditions,
    validate_no_completed_phase7_artifacts,
)
from cardiac_pathology.training.trainer import TrainingHistory


@dataclass(frozen=True, slots=True)
class FakePatient:
    """Label-only patient record for orchestration tests."""

    patient_id: str
    class_index: int


class ConstantLogitModel(nn.Module):
    """Model that emits deterministic five-class logits for software tests."""

    def forward(self, inputs: Tensor) -> Tensor:
        """Return one fixed logit row for each input sample."""
        return torch.tensor([[0.0, 3.0, 1.0, -1.0, 2.0]], dtype=inputs.dtype).repeat(
            inputs.shape[0],
            1,
        )


class FakeOuterTestDataset:
    """Dataset test double that records whether augmentation was disabled."""

    observed_augmentations: list[object] = []

    def __init__(self, patients: Sequence[FakePatient], preprocessor: object, augmentation: object):
        self.patients = tuple(patients)
        self.preprocessor = preprocessor
        self.observed_augmentations.append(augmentation)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        patient = self.patients[index]
        return torch.zeros((2, 14, 144, 144), dtype=torch.float32), torch.tensor(
            patient.class_index,
        )


def test_load_phase7_fold_splits_uses_persisted_inner_assignments(tmp_path: Path) -> None:
    """Persisted inner train/validation IDs are consumed exactly from the manifest."""
    outer_path, inner_path = write_split_manifests(tmp_path)

    folds = load_phase7_fold_splits(
        inner_split_path=inner_path,
        outer_split_path=outer_path,
    )

    assert [fold.fold_index for fold in folds] == [0, 1, 2, 3, 4]
    assert folds[0].inner_train_patient_ids == (
        "patient001",
        "patient002",
        "patient003",
    )
    assert folds[0].inner_validation_patient_ids == ("patient004",)
    assert folds[0].outer_test_patient_ids == ("patient005",)


def test_select_phase7_folds_honors_requested_fold_index() -> None:
    """An explicit fold index selects only that persisted fold."""
    folds = build_fold_splits()

    selected = select_phase7_folds(folds=folds, fold_index=3)

    assert [fold.fold_index for fold in selected] == [3]


def test_select_phase7_folds_defaults_to_all_folds_in_order() -> None:
    """Omitting --fold-index selects folds 0..4 deterministically."""
    folds = tuple(reversed(build_fold_splits()))

    selected = select_phase7_folds(folds=folds, fold_index=None)

    assert [fold.fold_index for fold in selected] == [0, 1, 2, 3, 4]


def test_select_phase7_folds_rejects_invalid_fold_index() -> None:
    """Unknown fold indices fail before any fold runner is created."""
    with pytest.raises(ValueError, match="Fold index 9 is not available"):
        select_phase7_folds(folds=build_fold_splits(), fold_index=9)


def test_orchestrator_runs_only_requested_fold(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The CLI orchestration layer passes only the requested fold to the runner."""
    outer_path, inner_path = write_split_manifests(tmp_path)
    config_path = tmp_path / "default.yaml"
    config_path.write_text("project:\n  seed: 42\ntraining: {}\n", encoding="utf-8")
    observed_fold_indices: list[int] = []

    class FakeRunner:
        def __init__(
            self,
            *,
            repository_root: Path,
            dataset_dir: Path,
            config: Mapping[str, object],
            class_mapping_path: Path,
            fold_split: Phase7FoldSplit,
        ) -> None:
            self.fold_split = fold_split

        def run(self) -> Phase7FoldResult:
            observed_fold_indices.append(self.fold_split.fold_index)
            return Phase7FoldResult(
                summary_path=tmp_path / "summary.json",
                oof_predictions_path=tmp_path / "oof.json",
                best_checkpoint_path=tmp_path / "best.pt",
                outer_test_accuracy=1.0,
                outer_test_macro_f1=1.0,
            )

    monkeypatch.setattr(phase7_module, "Phase7FoldRunner", FakeRunner)

    orchestrator = Phase7CrossValidationOrchestrator(
        repository_root=tmp_path,
        dataset_dir=tmp_path / "dataset",
        config_path=config_path,
        class_mapping_path=tmp_path / "class_mapping.json",
        outer_split_path=outer_path,
        inner_split_path=inner_path,
    )

    orchestrator.run(fold_index=2)

    assert observed_fold_indices == [2]


def test_load_phase7_fold_splits_rejects_outer_test_leakage(tmp_path: Path) -> None:
    """Outer-test IDs cannot appear in inner train or validation assignments."""
    outer_path, inner_path = write_split_manifests(tmp_path)
    artifact = json.loads(inner_path.read_text(encoding="utf-8"))
    artifact["folds"][0]["inner_validation_patient_ids"] = ["patient004", "patient005"]
    inner_path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(ValueError, match="inner validation overlaps outer test"):
        load_phase7_fold_splits(inner_split_path=inner_path, outer_split_path=outer_path)


def test_cuda_precondition_rejects_unavailable_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDA device configuration fails clearly when CUDA is unavailable."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="requires CUDA"):
        validate_cuda_runtime_preconditions({"device": "cuda"})


def test_cuda_precondition_rejects_missing_cublas_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Phase 7 training requires the explicit cuBLAS workspace environment value."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG=:4096:8"):
        validate_cuda_runtime_preconditions({"device": "cuda"})


def test_cuda_precondition_accepts_exact_cublas_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The exact frozen cuBLAS workspace value satisfies the runtime precondition."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", EXPECTED_CUBLAS_WORKSPACE_CONFIG)

    validate_cuda_runtime_preconditions({"device": "cuda"})


def test_phase6_checkpoint_root_is_rejected(tmp_path: Path) -> None:
    """Phase 7 artifact construction refuses Phase 6 checkpoint roots."""
    with pytest.raises(ValueError, match="Phase 6 checkpoint paths"):
        phase7_artifacts(
            checkpoint_root=tmp_path / "artifacts/checkpoints/phase6",
            fold_index=0,
            initialization_strategy="pretrained",
        )


def test_completed_phase7_fold_artifacts_are_not_overwritten(tmp_path: Path) -> None:
    """Existing completed Phase 7 outputs fail fast before training."""
    output_dir = tmp_path / "artifacts/checkpoints/phase7/fold_0/pretrained"
    output_dir.mkdir(parents=True)
    summary_path = output_dir / "phase7_fold_summary.json"
    summary_path.write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exist"):
        validate_no_completed_phase7_artifacts(
            Phase7FoldArtifacts(
                output_dir=output_dir,
                checkpoint_path=output_dir / "best_checkpoint.pt",
                oof_predictions_path=output_dir / "oof_predictions.json",
                summary_path=summary_path,
            )
        )


def test_existing_phase7_best_checkpoint_alone_is_not_overwritten(tmp_path: Path) -> None:
    """A partial prior run with only best_checkpoint.pt still blocks a fresh run."""
    output_dir = tmp_path / "artifacts/checkpoints/phase7/fold_0/pretrained"
    output_dir.mkdir(parents=True)
    checkpoint_path = output_dir / "best_checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")

    with pytest.raises(FileExistsError, match="best_checkpoint.pt"):
        validate_no_completed_phase7_artifacts(
            Phase7FoldArtifacts(
                output_dir=output_dir,
                checkpoint_path=checkpoint_path,
                oof_predictions_path=output_dir / "oof_predictions.json",
                summary_path=output_dir / "phase7_fold_summary.json",
            )
        )


def test_outer_test_evaluation_disables_augmentation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Outer-test prediction records are produced from a dataset with augmentation disabled."""
    FakeOuterTestDataset.observed_augmentations = []
    monkeypatch.setattr(phase7_module, "PreprocessedPatientDataset", FakeOuterTestDataset)

    records = evaluate_outer_test(
        model=ConstantLogitModel(),
        patients=(FakePatient(patient_id="patient001", class_index=1),),
        preprocessor=object(),
        device=torch.device("cpu"),
        num_classes=5,
        fold_index=0,
    )

    assert FakeOuterTestDataset.observed_augmentations == [None]
    assert len(records) == 1
    assert records[0].patient_id == "patient001"
    assert records[0].predicted_class_index == 1
    assert len(records[0].logits) == 5
    assert len(records[0].probabilities) == 5


def test_fold_runner_reloads_best_checkpoint_before_outer_test(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Outer-test evaluation is ordered after validation-selected checkpoint reload."""
    events: list[str] = []

    class FakeTrainer:
        def __init__(self, **kwargs: object) -> None:
            events.append("trainer_created")

        def fit(self, train_loader: object, validation_loader: object) -> TrainingHistory:
            events.append("fit")
            return TrainingHistory(
                best_epoch=3,
                best_validation_macro_f1=0.75,
                stopped_early=False,
            )

    def fake_build_loader(
        patients: Sequence[FakePatient],
        preprocessor: object,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        seed: int,
        augmentation: object,
    ) -> object:
        events.append("train_loader" if shuffle else f"eval_loader_aug_{augmentation!r}")
        return object()

    def fake_load_best_checkpoint(
        model: nn.Module,
        checkpoint_path: Path,
        device: torch.device,
    ) -> None:
        events.append("load_best_checkpoint")

    def fake_evaluate_outer_test(**kwargs: object) -> tuple[Phase7PredictionRecord, ...]:
        events.append("outer_test")
        return (
            Phase7PredictionRecord(
                fold_index=0,
                patient_id="patient005",
                true_class_index=1,
                predicted_class_index=1,
                logits=(0.0, 1.0, 0.0, 0.0, 0.0),
                probabilities=(0.1, 0.6, 0.1, 0.1, 0.1),
            ),
        )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", EXPECTED_CUBLAS_WORKSPACE_CONFIG)
    monkeypatch.setattr(phase7_module, "build_loader", fake_build_loader)
    monkeypatch.setattr(
        phase7_module,
        "initialize_model",
        lambda model, strategy: CompatibilityReport(strategy, ()),
    )
    monkeypatch.setattr(phase7_module, "build_optimizer", lambda **kwargs: object())
    monkeypatch.setattr(phase7_module, "Trainer", FakeTrainer)
    monkeypatch.setattr(phase7_module, "load_best_checkpoint", fake_load_best_checkpoint)
    monkeypatch.setattr(phase7_module, "evaluate_outer_test", fake_evaluate_outer_test)
    monkeypatch.setattr(phase7_module, "save_oof_predictions", lambda path, records: path)
    monkeypatch.setattr(phase7_module, "save_phase7_summary", lambda **kwargs: kwargs["path"])
    monkeypatch.setattr(
        Phase7FoldRunner,
        "_build_model",
        staticmethod(lambda model_config: nn.Linear(1, 1)),
    )
    monkeypatch.setattr(
        Phase7FoldRunner,
        "_index_patients",
        lambda self: (
            FakePatient("patient001", 0),
            FakePatient("patient002", 0),
            FakePatient("patient003", 0),
            FakePatient("patient004", 0),
            FakePatient("patient005", 1),
        ),
    )

    runner = Phase7FoldRunner(
        repository_root=tmp_path,
        dataset_dir=tmp_path / "dataset",
        config=build_phase7_config(),
        class_mapping_path=tmp_path / "class_mapping.json",
        fold_split=Phase7FoldSplit(
            fold_index=0,
            inner_train_patient_ids=("patient001", "patient002", "patient003"),
            inner_validation_patient_ids=("patient004",),
            outer_test_patient_ids=("patient005",),
        ),
    )

    runner.run()

    assert "eval_loader_aug_None" in events
    assert events.index("fit") < events.index("load_best_checkpoint")
    assert events.index("load_best_checkpoint") < events.index("outer_test")


def build_fold_splits() -> tuple[Phase7FoldSplit, ...]:
    """Build five small persisted split records."""
    return tuple(
        Phase7FoldSplit(
            fold_index=fold_index,
            inner_train_patient_ids=("patient001",),
            inner_validation_patient_ids=("patient002",),
            outer_test_patient_ids=("patient003",),
        )
        for fold_index in range(5)
    )


def write_split_manifests(tmp_path: Path) -> tuple[Path, Path]:
    """Write small outer and inner split manifests for orchestration tests."""
    outer_path = tmp_path / "outer.json"
    inner_path = tmp_path / "inner.json"
    outer_folds = []
    inner_folds = []
    for fold_index in range(5):
        offset = fold_index * 5
        development_ids = [f"patient{offset + value:03d}" for value in range(1, 5)]
        test_ids = [f"patient{offset + 5:03d}"]
        outer_folds.append(
            {
                "fold_index": fold_index,
                "train_patient_ids": development_ids,
                "test_patient_ids": test_ids,
            }
        )
        inner_folds.append(
            {
                "fold_index": fold_index,
                "outer_development_patient_ids": development_ids,
                "outer_test_patient_ids": test_ids,
                "inner_train_patient_ids": development_ids[:3],
                "inner_validation_patient_ids": development_ids[3:],
            }
        )
    outer_path.write_text(json.dumps({"folds": outer_folds}), encoding="utf-8")
    inner_path.write_text(json.dumps({"folds": inner_folds}), encoding="utf-8")
    return outer_path, inner_path


def build_phase7_config() -> dict[str, object]:
    """Build a minimal frozen Phase 7 config mapping."""
    return {
        "project": {"seed": 42},
        "preprocessing": {
            "orientation": "LPS",
            "target_spacing_mm": {"x": 1.5, "y": 1.5, "z": 7.5},
            "target_shape": {"d": 14, "h": 144, "w": 144},
            "intensity": {
                "lower_percentile": 0.5,
                "upper_percentile": 99.5,
                "epsilon": 1.0e-6,
            },
            "padding": {"z_value": 0.0},
        },
        "model": {"input_channels": 2, "num_classes": 5},
        "training": {
            "initialization_strategy": "pretrained",
            "batch_size": 2,
            "num_epochs": 50,
            "learning_rate": 1.0e-4,
            "weight_decay": 1.0e-4,
            "optimizer": "adamw",
            "early_stopping_patience": 10,
            "early_stopping_min_delta": 0.0,
            "num_workers": 0,
            "checkpoint_dir": "artifacts/checkpoints/phase7",
            "device": "cuda",
            "augmentation": {
                "enabled": True,
                "rotation_degrees": 5.0,
                "translation_mm": 5.0,
                "intensity_scale_min": 0.95,
                "intensity_scale_max": 1.05,
                "gaussian_noise_std": 0.01,
                "transform_probability": 0.5,
            },
        },
    }
