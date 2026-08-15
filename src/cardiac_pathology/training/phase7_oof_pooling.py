"""Pool Phase 7 out-of-fold prediction records after all folds finish."""

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from cardiac_pathology.training.splits import load_outer_folds, patient_id_sort_key

EXPECTED_NUM_FOLDS = 5
EXPECTED_RECORDS_PER_FOLD = 20
EXPECTED_TOTAL_RECORDS = 100
EXPECTED_NUM_CLASSES = 5
PROBABILITY_SUM_TOLERANCE = 1.0e-5


@dataclass(frozen=True, slots=True)
class Phase7OofRecord:
    """Loaded Phase 7 patient-level OOF prediction record."""

    fold_index: int
    patient_id: str
    true_class_index: int
    predicted_class_index: int
    logits: tuple[float, ...]
    probabilities: tuple[float, ...]

    def as_metadata(self) -> dict[str, object]:
        """Convert the OOF record to the frozen JSON schema."""
        return {
            "fold_index": self.fold_index,
            "patient_id": self.patient_id,
            "true_class_index": self.true_class_index,
            "predicted_class_index": self.predicted_class_index,
            "logits": list(self.logits),
            "probabilities": list(self.probabilities),
        }


class Phase7OofPredictionPooler:
    """Validate and pool the five Phase 7 per-fold OOF prediction files."""

    def __init__(
        self,
        *,
        fold_prediction_paths: Mapping[int, Path],
        outer_split_path: Path,
        output_path: Path,
    ) -> None:
        self.fold_prediction_paths = dict(fold_prediction_paths)
        self.outer_split_path = outer_split_path
        self.output_path = output_path

    @classmethod
    def from_phase7_checkpoint_root(
        cls,
        *,
        checkpoint_root: Path,
        outer_split_path: Path,
        output_path: Path | None = None,
    ) -> "Phase7OofPredictionPooler":
        """Build a pooler for the frozen Phase 7 checkpoint directory layout."""
        return cls(
            fold_prediction_paths={
                fold_index: checkpoint_root
                / f"fold_{fold_index}"
                / "pretrained"
                / "oof_predictions.json"
                for fold_index in range(EXPECTED_NUM_FOLDS)
            },
            outer_split_path=outer_split_path,
            output_path=(
                output_path
                if output_path is not None
                else checkpoint_root / "pooled_oof_predictions.json"
            ),
        )

    def build_and_save(self) -> Path:
        """Load, validate, sort, and save the pooled OOF prediction artifact."""
        records = self.build_pooled_records()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps([record.as_metadata() for record in records], indent=2) + "\n",
            encoding="utf-8",
        )
        return self.output_path

    def build_pooled_records(self) -> tuple[Phase7OofRecord, ...]:
        """Build validated pooled records sorted deterministically by patient ID."""
        validate_expected_fold_paths(self.fold_prediction_paths)
        expected_outer_test_ids = load_expected_outer_test_patient_ids(self.outer_split_path)

        records: list[Phase7OofRecord] = []
        for fold_index in sorted(self.fold_prediction_paths):
            fold_records = load_fold_oof_records(
                path=self.fold_prediction_paths[fold_index],
                expected_fold_index=fold_index,
            )
            if len(fold_records) != EXPECTED_RECORDS_PER_FOLD:
                raise ValueError(
                    f"fold {fold_index} must contain {EXPECTED_RECORDS_PER_FOLD} records, "
                    f"got {len(fold_records)}"
                )
            records.extend(fold_records)

        validate_pooled_oof_records(
            records=records,
            expected_outer_test_ids=expected_outer_test_ids,
        )
        return tuple(sorted(records, key=lambda record: patient_id_sort_key(record.patient_id)))


def validate_expected_fold_paths(fold_prediction_paths: Mapping[int, Path]) -> None:
    """Validate that exactly fold indices 0..4 are provided."""
    expected_fold_indices = set(range(EXPECTED_NUM_FOLDS))
    actual_fold_indices = set(fold_prediction_paths)
    if actual_fold_indices != expected_fold_indices:
        raise ValueError(
            f"Expected fold prediction paths for folds 0..4, got {sorted(actual_fold_indices)}"
        )


def load_fold_oof_records(
    *,
    path: Path,
    expected_fold_index: int,
) -> tuple[Phase7OofRecord, ...]:
    """Load one per-fold OOF prediction file and validate each record."""
    with path.open("r", encoding="utf-8") as file:
        raw_records = json.load(file)
    if not isinstance(raw_records, list):
        raise ValueError(f"Expected a list of OOF records in {path}")

    records: list[Phase7OofRecord] = []
    for record_index, raw_record in enumerate(raw_records):
        record = oof_record_from_mapping(
            raw_record=raw_record,
            path=path,
            record_index=record_index,
        )
        if record.fold_index != expected_fold_index:
            raise ValueError(
                f"{path} record {record_index} has fold_index {record.fold_index}, "
                f"expected {expected_fold_index}"
            )
        validate_oof_record(record=record, path=path, record_index=record_index)
        records.append(record)
    return tuple(records)


def oof_record_from_mapping(
    *,
    raw_record: object,
    path: Path,
    record_index: int,
) -> Phase7OofRecord:
    """Parse one JSON OOF record without changing its schema."""
    record = require_json_mapping(raw_record, path, record_index)
    return Phase7OofRecord(
        fold_index=require_json_int(record, "fold_index", path, record_index),
        patient_id=require_json_str(record, "patient_id", path, record_index),
        true_class_index=require_json_int(record, "true_class_index", path, record_index),
        predicted_class_index=require_json_int(
            record,
            "predicted_class_index",
            path,
            record_index,
        ),
        logits=require_json_float_tuple(record, "logits", path, record_index),
        probabilities=require_json_float_tuple(record, "probabilities", path, record_index),
    )


def validate_oof_record(record: Phase7OofRecord, path: Path, record_index: int) -> None:
    """Validate shape and probability invariants for one OOF prediction record."""
    if len(record.logits) != EXPECTED_NUM_CLASSES:
        raise ValueError(f"{path} record {record_index} logits must have length 5")
    if len(record.probabilities) != EXPECTED_NUM_CLASSES:
        raise ValueError(f"{path} record {record_index} probabilities must have length 5")
    if not all(math.isfinite(probability) for probability in record.probabilities):
        raise ValueError(f"{path} record {record_index} probabilities must be finite")
    probability_sum = sum(record.probabilities)
    if abs(probability_sum - 1.0) > PROBABILITY_SUM_TOLERANCE:
        raise ValueError(
            f"{path} record {record_index} probabilities must sum to 1 within "
            f"{PROBABILITY_SUM_TOLERANCE}, got {probability_sum}"
        )
    predicted_from_probabilities = max(
        range(EXPECTED_NUM_CLASSES),
        key=lambda class_index: record.probabilities[class_index],
    )
    if record.predicted_class_index != predicted_from_probabilities:
        raise ValueError(
            f"{path} record {record_index} predicted_class_index must equal argmax(probabilities)"
        )


def validate_pooled_oof_records(
    *,
    records: Sequence[Phase7OofRecord],
    expected_outer_test_ids: set[str],
) -> None:
    """Validate pooled 100-patient OOF coverage before writing the artifact."""
    if len(records) != EXPECTED_TOTAL_RECORDS:
        raise ValueError(f"pooled OOF records must contain 100 records, got {len(records)}")

    fold_counts = Counter(record.fold_index for record in records)
    expected_fold_counts = {fold_index: EXPECTED_RECORDS_PER_FOLD for fold_index in range(5)}
    if dict(sorted(fold_counts.items())) != expected_fold_counts:
        raise ValueError(f"pooled OOF fold counts mismatch: {dict(sorted(fold_counts.items()))}")

    patient_ids = [record.patient_id for record in records]
    if len(set(patient_ids)) != len(patient_ids):
        raise ValueError("pooled OOF patient IDs must be globally unique")
    if set(patient_ids) != expected_outer_test_ids:
        raise ValueError("pooled OOF patient IDs do not match persisted outer-test IDs")


def load_expected_outer_test_patient_ids(outer_split_path: Path) -> set[str]:
    """Load the 100 patient IDs expected to appear exactly once in pooled OOF."""
    folds = load_outer_folds(outer_split_path)
    if len(folds) != EXPECTED_NUM_FOLDS:
        raise ValueError(f"outer split must contain {EXPECTED_NUM_FOLDS} folds")
    patient_ids = [patient_id for fold in folds for patient_id in fold.test_patient_ids]
    if len(patient_ids) != EXPECTED_TOTAL_RECORDS:
        raise ValueError("outer split must contain 100 total outer-test patient IDs")
    if len(set(patient_ids)) != len(patient_ids):
        raise ValueError("outer split contains duplicate outer-test patient IDs")
    return set(patient_ids)


def require_json_mapping(value: object, path: Path, record_index: int) -> Mapping[str, object]:
    """Return one parsed OOF record as a string-keyed mapping."""
    if not isinstance(value, dict):
        raise ValueError(f"{path} record {record_index} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{path} record {record_index} must contain string keys")
    return value


def require_json_int(
    mapping: Mapping[str, object],
    key: str,
    path: Path,
    record_index: int,
) -> int:
    """Read one required integer field from an OOF record."""
    value = mapping.get(key)
    if not isinstance(value, int):
        raise ValueError(f"{path} record {record_index} {key} must be an integer")
    return value


def require_json_str(
    mapping: Mapping[str, object],
    key: str,
    path: Path,
    record_index: int,
) -> str:
    """Read one required string field from an OOF record."""
    value = mapping.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{path} record {record_index} {key} must be a string")
    return value


def require_json_float_tuple(
    mapping: Mapping[str, object],
    key: str,
    path: Path,
    record_index: int,
) -> tuple[float, ...]:
    """Read one required numeric sequence from an OOF record."""
    value = mapping.get(key)
    if not isinstance(value, list) or not all(isinstance(item, int | float) for item in value):
        raise ValueError(f"{path} record {record_index} {key} must be a numeric list")
    return tuple(float(item) for item in value)
