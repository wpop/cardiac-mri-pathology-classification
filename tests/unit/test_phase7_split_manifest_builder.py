"""Unit tests for Phase 7 inner-validation split manifest building."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import cardiac_pathology.training.phase7_split_manifest_builder as builder_module
from cardiac_pathology.data import PatientFold
from cardiac_pathology.training import Phase7SplitManifestBuilder
from cardiac_pathology.training.splits import InnerValidationSplit


@dataclass(frozen=True, slots=True)
class ManifestPatient:
    """Minimal label-only patient record for manifest-building tests."""

    patient_id: str
    class_index: int


class FakeAcdcDatasetIndexer:
    """Test double that avoids touching real medical image files."""

    patients: tuple[ManifestPatient, ...] = ()
    class_mapping: dict[int, str] = {}

    def __init__(self, dataset_dir: Path, class_mapping_path: Path) -> None:
        self.dataset_dir = dataset_dir
        self.class_mapping_path = class_mapping_path

    def index_patients(self) -> tuple[ManifestPatient, ...]:
        """Return the configured label-only cohort."""
        return self.patients

    def load_class_mapping(self) -> dict[int, str]:
        """Return the configured authoritative class mapping."""
        return self.class_mapping


def test_phase7_manifest_builds_inner_validation_for_all_outer_folds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The manifest records validated, deterministic inner splits for every outer fold."""
    install_test_doubles(monkeypatch=monkeypatch)

    artifact = build_test_builder(tmp_path).build_artifact()

    assert artifact["dataset"] == "ACDC"
    assert artifact["num_patients"] == 100
    assert artifact["num_folds"] == 5
    assert artifact["seed"] == 42
    assert artifact["validation_fraction"] == 0.25
    assert artifact["outer_split_artifact"] == "artifacts/dataset_splits/acdc_5fold_seed42.json"
    assert artifact["class_mapping"] == {
        "0": "NOR",
        "1": "DCM",
        "2": "HCM",
        "3": "MINF",
        "4": "RV",
    }

    folds = require_fold_records(artifact)
    assert [fold["fold_index"] for fold in folds] == [0, 1, 2, 3, 4]
    for fold in folds:
        train_ids = set(require_str_list(fold, "inner_train_patient_ids"))
        validation_ids = set(require_str_list(fold, "inner_validation_patient_ids"))
        outer_development_ids = set(require_str_list(fold, "outer_development_patient_ids"))
        outer_test_ids = set(require_str_list(fold, "outer_test_patient_ids"))

        assert len(train_ids) == 60
        assert len(validation_ids) == 20
        assert len(outer_development_ids) == 80
        assert len(outer_test_ids) == 20
        assert not train_ids & validation_ids
        assert not train_ids & outer_test_ids
        assert not validation_ids & outer_test_ids
        assert train_ids | validation_ids == outer_development_ids
        assert fold["outer_development_class_counts"] == expected_counts(16)
        assert fold["outer_test_class_counts"] == expected_counts(4)
        assert fold["inner_train_class_counts"] == expected_counts(12)
        assert fold["inner_validation_class_counts"] == expected_counts(4)


def test_phase7_manifest_builder_persists_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The manifest builder writes its validated artifact to the requested path."""
    install_test_doubles(monkeypatch=monkeypatch)
    output_path = tmp_path / "nested" / "inner.json"

    saved_path = build_test_builder(tmp_path, output_path=output_path).build_and_save()

    assert saved_path == output_path
    assert output_path.is_file()
    assert output_path.read_text(encoding="utf-8").endswith("\n")


def test_phase7_manifest_builder_rejects_wrong_outer_fold_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The frozen Phase 7 contract requires exactly five outer folds."""
    install_test_doubles(monkeypatch=monkeypatch, outer_folds=build_outer_folds()[:4])

    with pytest.raises(ValueError, match="outer folds size mismatch"):
        build_test_builder(tmp_path).build_artifact()


def test_phase7_manifest_builder_rejects_unknown_patient_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Every partition must reference only IDs from the indexed cohort."""
    outer_folds = list(build_outer_folds())
    first_fold = outer_folds[0]
    outer_folds[0] = PatientFold(
        fold_index=first_fold.fold_index,
        train_patient_ids=first_fold.train_patient_ids,
        test_patient_ids=first_fold.test_patient_ids[:-1] + ("patient999",),
    )
    install_test_doubles(monkeypatch=monkeypatch, outer_folds=tuple(outer_folds))

    with pytest.raises(ValueError, match="outer-test contains unknown patient IDs"):
        build_test_builder(tmp_path).build_artifact()


def test_phase7_manifest_builder_rejects_duplicate_inner_partition_ids(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Inner partition validation catches duplicate IDs before saving."""
    install_test_doubles(monkeypatch=monkeypatch)

    def duplicate_inner_split(
        development_patients: Sequence[ManifestPatient],
        outer_test_patient_ids: Sequence[str],
        validation_fraction: float,
        random_seed: int,
    ) -> InnerValidationSplit:
        development_ids = tuple(patient.patient_id for patient in development_patients)
        return InnerValidationSplit(
            train_patient_ids=development_ids[:59] + (development_ids[0],),
            validation_patient_ids=development_ids[60:80],
            outer_test_patient_ids=tuple(outer_test_patient_ids),
        )

    monkeypatch.setattr(builder_module, "create_inner_validation_split", duplicate_inner_split)

    with pytest.raises(ValueError, match="inner-train contains duplicate patient IDs"):
        build_test_builder(tmp_path).build_artifact()


def test_phase7_manifest_builder_rejects_pooled_outer_test_duplicate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Pooled outer-test IDs must cover the 100-patient cohort exactly once."""
    outer_folds = list(build_outer_folds())
    outer_folds[1] = outer_folds[0]
    install_test_doubles(monkeypatch=monkeypatch, outer_folds=tuple(outer_folds))

    with pytest.raises(ValueError, match="pooled outer-test patient IDs contains duplicate"):
        build_test_builder(tmp_path).build_artifact()


def install_test_doubles(
    monkeypatch: pytest.MonkeyPatch,
    patients: tuple[ManifestPatient, ...] | None = None,
    class_mapping: dict[int, str] | None = None,
    outer_folds: tuple[PatientFold, ...] | None = None,
) -> None:
    """Install label-only indexer and outer-fold doubles."""
    FakeAcdcDatasetIndexer.patients = patients if patients is not None else build_patients()
    FakeAcdcDatasetIndexer.class_mapping = (
        class_mapping if class_mapping is not None else build_class_mapping()
    )
    monkeypatch.setattr(builder_module, "AcdcDatasetIndexer", FakeAcdcDatasetIndexer)
    monkeypatch.setattr(
        builder_module,
        "load_outer_folds",
        lambda path: outer_folds if outer_folds is not None else build_outer_folds(),
    )


def build_test_builder(
    tmp_path: Path,
    output_path: Path | None = None,
) -> Phase7SplitManifestBuilder:
    """Build a test manifest builder with frozen Phase 7 settings."""
    return Phase7SplitManifestBuilder(
        dataset_dir=tmp_path / "dataset",
        class_mapping_path=tmp_path / "class_mapping.json",
        outer_split_path=tmp_path / "outer.json",
        outer_split_reference="artifacts/dataset_splits/acdc_5fold_seed42.json",
        output_path=output_path if output_path is not None else tmp_path / "inner.json",
        seed=42,
        validation_fraction=0.25,
    )


def build_patients() -> tuple[ManifestPatient, ...]:
    """Build the frozen 100-patient, five-class ACDC-shaped cohort."""
    patients: list[ManifestPatient] = []
    for class_index in range(5):
        for offset in range(20):
            patient_number = (class_index * 20) + offset + 1
            patients.append(
                ManifestPatient(
                    patient_id=f"patient{patient_number:03d}",
                    class_index=class_index,
                )
            )
    return tuple(patients)


def build_class_mapping() -> dict[int, str]:
    """Return the canonical five-class ACDC mapping."""
    return {0: "NOR", 1: "DCM", 2: "HCM", 3: "MINF", 4: "RV"}


def build_outer_folds() -> tuple[PatientFold, ...]:
    """Build five deterministic outer folds with four test patients per class."""
    patient_ids_by_class = [
        tuple(f"patient{(class_index * 20) + offset + 1:03d}" for offset in range(20))
        for class_index in range(5)
    ]
    all_patient_ids = tuple(patient_id for group in patient_ids_by_class for patient_id in group)

    folds: list[PatientFold] = []
    for fold_index in range(5):
        test_ids = tuple(
            patient_id
            for group in patient_ids_by_class
            for patient_id in group[fold_index * 4 : (fold_index + 1) * 4]
        )
        test_set = set(test_ids)
        train_ids = tuple(
            patient_id for patient_id in all_patient_ids if patient_id not in test_set
        )
        folds.append(
            PatientFold(
                fold_index=fold_index,
                train_patient_ids=train_ids,
                test_patient_ids=test_ids,
            )
        )
    return tuple(folds)


def expected_counts(count: int) -> dict[str, int]:
    """Return equal expected counts for each canonical ACDC class."""
    return {class_name: count for class_name in build_class_mapping().values()}


def require_fold_records(artifact: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return typed fold records from a generated manifest."""
    value = artifact["folds"]
    assert isinstance(value, list)
    assert all(isinstance(item, dict) for item in value)
    return value


def require_str_list(record: Mapping[str, Any], key: str) -> list[str]:
    """Return a string list from one manifest fold record."""
    value = record[key]
    assert isinstance(value, list)
    assert all(isinstance(item, str) for item in value)
    return value
