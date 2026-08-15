"""Unit tests for Phase 7 pooled OOF prediction aggregation."""

import json
from pathlib import Path

import pytest

from cardiac_pathology.training.phase7_oof_pooling import (
    EXPECTED_NUM_FOLDS,
    EXPECTED_RECORDS_PER_FOLD,
    Phase7OofPredictionPooler,
)


def test_phase7_oof_pooler_writes_records_sorted_by_patient_id(tmp_path: Path) -> None:
    """Validated pooled OOF records are written in deterministic patient order."""
    fold_paths, outer_split_path = write_valid_oof_fixture(tmp_path)
    output_path = tmp_path / "pooled_oof_predictions.json"

    saved_path = Phase7OofPredictionPooler(
        fold_prediction_paths=fold_paths,
        outer_split_path=outer_split_path,
        output_path=output_path,
    ).build_and_save()

    pooled_records = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved_path == output_path
    assert len(pooled_records) == 100
    assert [record["patient_id"] for record in pooled_records] == [
        f"patient{patient_number:03d}" for patient_number in range(1, 101)
    ]
    assert set(pooled_records[0]) == {
        "fold_index",
        "patient_id",
        "true_class_index",
        "predicted_class_index",
        "logits",
        "probabilities",
    }


def test_phase7_oof_pooler_requires_exactly_five_folds(tmp_path: Path) -> None:
    """The pooled artifact must be assembled from exactly folds 0..4."""
    fold_paths, outer_split_path = write_valid_oof_fixture(tmp_path)
    fold_paths.pop(4)

    with pytest.raises(ValueError, match="folds 0..4"):
        Phase7OofPredictionPooler(
            fold_prediction_paths=fold_paths,
            outer_split_path=outer_split_path,
            output_path=tmp_path / "pooled.json",
        ).build_pooled_records()


def test_phase7_oof_pooler_requires_twenty_records_per_fold(tmp_path: Path) -> None:
    """Each fold must contribute exactly 20 OOF prediction records."""
    fold_paths, outer_split_path = write_valid_oof_fixture(tmp_path)
    records = json.loads(fold_paths[0].read_text(encoding="utf-8"))
    fold_paths[0].write_text(json.dumps(records[:-1]), encoding="utf-8")

    with pytest.raises(ValueError, match="fold 0 must contain 20 records"):
        Phase7OofPredictionPooler(
            fold_prediction_paths=fold_paths,
            outer_split_path=outer_split_path,
            output_path=tmp_path / "pooled.json",
        ).build_pooled_records()


def test_phase7_oof_pooler_rejects_duplicate_patient_id(tmp_path: Path) -> None:
    """Every pooled patient ID must be globally unique."""
    fold_paths, outer_split_path = write_valid_oof_fixture(tmp_path)
    records = json.loads(fold_paths[1].read_text(encoding="utf-8"))
    records[0]["patient_id"] = "patient001"
    fold_paths[1].write_text(json.dumps(records), encoding="utf-8")

    with pytest.raises(ValueError, match="globally unique"):
        Phase7OofPredictionPooler(
            fold_prediction_paths=fold_paths,
            outer_split_path=outer_split_path,
            output_path=tmp_path / "pooled.json",
        ).build_pooled_records()


def test_phase7_oof_pooler_rejects_outer_split_mismatch(tmp_path: Path) -> None:
    """Pooled patient IDs must equal the persisted outer-test patient IDs."""
    fold_paths, outer_split_path = write_valid_oof_fixture(tmp_path)
    records = json.loads(fold_paths[0].read_text(encoding="utf-8"))
    records[0]["patient_id"] = "patient999"
    fold_paths[0].write_text(json.dumps(records), encoding="utf-8")

    with pytest.raises(ValueError, match="outer-test IDs"):
        Phase7OofPredictionPooler(
            fold_prediction_paths=fold_paths,
            outer_split_path=outer_split_path,
            output_path=tmp_path / "pooled.json",
        ).build_pooled_records()


def test_phase7_oof_pooler_rejects_record_fold_index_mismatch(tmp_path: Path) -> None:
    """Each record must carry the fold index of its source fold file."""
    fold_paths, outer_split_path = write_valid_oof_fixture(tmp_path)
    records = json.loads(fold_paths[2].read_text(encoding="utf-8"))
    records[0]["fold_index"] = 4
    fold_paths[2].write_text(json.dumps(records), encoding="utf-8")

    with pytest.raises(ValueError, match="expected 2"):
        Phase7OofPredictionPooler(
            fold_prediction_paths=fold_paths,
            outer_split_path=outer_split_path,
            output_path=tmp_path / "pooled.json",
        ).build_pooled_records()


def test_phase7_oof_pooler_rejects_bad_probability_vector(tmp_path: Path) -> None:
    """Probabilities must have length 5 and sum to 1 within tolerance."""
    fold_paths, outer_split_path = write_valid_oof_fixture(tmp_path)
    records = json.loads(fold_paths[0].read_text(encoding="utf-8"))
    records[0]["probabilities"] = [0.5, 0.5, 0.1, 0.0, 0.0]
    fold_paths[0].write_text(json.dumps(records), encoding="utf-8")

    with pytest.raises(ValueError, match="sum to 1"):
        Phase7OofPredictionPooler(
            fold_prediction_paths=fold_paths,
            outer_split_path=outer_split_path,
            output_path=tmp_path / "pooled.json",
        ).build_pooled_records()


def test_phase7_oof_pooler_rejects_predicted_class_not_argmax(tmp_path: Path) -> None:
    """The predicted class must equal argmax over stored probabilities."""
    fold_paths, outer_split_path = write_valid_oof_fixture(tmp_path)
    records = json.loads(fold_paths[0].read_text(encoding="utf-8"))
    records[0]["predicted_class_index"] = 4
    fold_paths[0].write_text(json.dumps(records), encoding="utf-8")

    with pytest.raises(ValueError, match="argmax"):
        Phase7OofPredictionPooler(
            fold_prediction_paths=fold_paths,
            outer_split_path=outer_split_path,
            output_path=tmp_path / "pooled.json",
        ).build_pooled_records()


def write_valid_oof_fixture(tmp_path: Path) -> tuple[dict[int, Path], Path]:
    """Write a 5-fold, 100-record label-only OOF fixture."""
    fold_paths: dict[int, Path] = {}
    outer_folds: list[dict[str, object]] = []
    for fold_index in range(EXPECTED_NUM_FOLDS):
        fold_patient_numbers = list(
            range(
                (fold_index * EXPECTED_RECORDS_PER_FOLD) + 1,
                ((fold_index + 1) * EXPECTED_RECORDS_PER_FOLD) + 1,
            )
        )
        fold_records = [
            build_oof_record(
                fold_index=fold_index,
                patient_id=f"patient{patient_number:03d}",
            )
            for patient_number in reversed(fold_patient_numbers)
        ]
        fold_path = tmp_path / f"fold_{fold_index}" / "pretrained" / "oof_predictions.json"
        fold_path.parent.mkdir(parents=True)
        fold_path.write_text(json.dumps(fold_records), encoding="utf-8")
        fold_paths[fold_index] = fold_path
        outer_folds.append(
            {
                "fold_index": fold_index,
                "train_patient_ids": [],
                "test_patient_ids": [
                    f"patient{patient_number:03d}" for patient_number in fold_patient_numbers
                ],
            }
        )

    outer_split_path = tmp_path / "acdc_5fold_seed42.json"
    outer_split_path.write_text(json.dumps({"folds": outer_folds}), encoding="utf-8")
    return fold_paths, outer_split_path


def build_oof_record(fold_index: int, patient_id: str) -> dict[str, object]:
    """Build one valid OOF record using the frozen schema."""
    return {
        "fold_index": fold_index,
        "patient_id": patient_id,
        "true_class_index": 1,
        "predicted_class_index": 1,
        "logits": [0.0, 3.0, 1.0, -1.0, 2.0],
        "probabilities": [0.03, 0.75, 0.1, 0.02, 0.1],
    }
