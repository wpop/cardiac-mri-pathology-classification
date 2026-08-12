"""Golden preprocessing reference generation and validation utilities."""

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import nibabel as nib
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cardiac_pathology.data import AcdcPatient
from cardiac_pathology.preprocessing.patient_preprocessor import PatientPreprocessor
from cardiac_pathology.preprocessing.preprocessing_result import PreprocessingResult

GOLDEN_FIXTURE_VERSION = 1
TENSOR_SHAPE = (2, 14, 144, 144)
TENSOR_DTYPE = "float32"
SOURCE_ARRAY_ORDER = "XYZ"
MODEL_ARRAY_ORDER = "ZYX"
CHANNEL_SEMANTICS = {"0": "ED", "1": "ES"}
NORMALIZATION_SCOPE = "joint_ed_es"


@dataclass(frozen=True, slots=True)
class GoldenGenerationResult:
    """Summary of a golden reference generation run."""

    manifest_path: Path
    tensor_paths: tuple[Path, ...]
    selected_patient_ids: tuple[str, ...]
    file_hashes: dict[str, str]


class GoldenReferenceGenerator:
    """Generate deterministic golden preprocessing fixtures from real ACDC patients."""

    def __init__(
        self,
        output_dir: Path,
        preprocessing_contract_path: Path,
        phase3_validation_csv_path: Path | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.preprocessing_contract_path = preprocessing_contract_path
        self.phase3_validation_csv_path = phase3_validation_csv_path

    def generate(
        self,
        patients: tuple[AcdcPatient, ...],
        class_mapping: dict[int, str],
        preprocessor: PatientPreprocessor,
    ) -> GoldenGenerationResult:
        """Generate golden tensors and manifest using the production preprocessor."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        patients_by_id = {patient.patient_id: patient for patient in patients}
        selections = self.select_patients(patients, class_mapping, preprocessor)

        manifest_patients: list[dict[str, Any]] = []
        tensor_paths: list[Path] = []
        for patient_id in sorted(selections):
            patient = patients_by_id[patient_id]
            result = preprocessor.preprocess(patient)
            self._validate_tensor(patient_id, result.tensor)
            tensor_filename = f"{patient_id}.npy"
            tensor_path = self.output_dir / tensor_filename
            np.save(tensor_path, result.tensor, allow_pickle=False)
            reloaded_tensor = np.load(tensor_path, allow_pickle=False)
            self._validate_tensor(patient_id, reloaded_tensor)
            if not np.array_equal(result.tensor, reloaded_tensor):
                raise ValueError(f"{patient_id}: saved tensor differs from preprocessing output")
            tensor_sha256 = self.compute_file_sha256(tensor_path)
            manifest_patients.append(
                self.build_patient_manifest_entry(
                    patient=patient,
                    result=result,
                    tensor_filename=tensor_filename,
                    tensor_sha256=tensor_sha256,
                    selection_reasons=selections[patient_id],
                )
            )
            tensor_paths.append(tensor_path)

        manifest = self.build_manifest(manifest_patients)
        manifest_path = self.output_dir / "manifest.json"
        self.write_manifest(manifest_path, manifest)
        file_hashes = self.collect_fixture_hashes(manifest_path)
        return GoldenGenerationResult(
            manifest_path=manifest_path,
            tensor_paths=tuple(tensor_paths),
            selected_patient_ids=tuple(patient["patient_id"] for patient in manifest_patients),
            file_hashes=file_hashes,
        )

    def select_patients(
        self,
        patients: tuple[AcdcPatient, ...],
        class_mapping: dict[int, str],
        preprocessor: PatientPreprocessor,
    ) -> dict[str, list[str]]:
        """Select one representative per class plus global depth extremes."""
        validation = self._load_or_compute_validation_depths(patients, preprocessor)
        patients_by_id = {patient.patient_id: patient for patient in patients}
        missing_from_index = set(validation["patient_id"]) - set(patients_by_id)
        if missing_from_index:
            missing = ", ".join(sorted(missing_from_index))
            raise ValueError(f"Phase 3 validation includes unknown patients: {missing}")
        missing_from_validation = set(patients_by_id) - set(validation["patient_id"])
        if missing_from_validation:
            missing = ", ".join(sorted(missing_from_validation))
            raise ValueError(f"Phase 3 validation missing indexed patients: {missing}")

        selected: dict[str, list[str]] = defaultdict(list)
        for class_index in sorted(class_mapping):
            class_name = class_mapping[class_index]
            class_rows = validation[validation["class"] == class_name].copy()
            if class_rows.empty:
                raise ValueError(f"No Phase 3 validation rows for class: {class_name}")
            median_depth = float(class_rows["valid_depth"].median())
            class_rows["median_distance"] = (
                class_rows["valid_depth"].astype(float) - median_depth
            ).abs()
            representative = (
                class_rows.sort_values(["median_distance", "patient_id"], kind="mergesort")
                .iloc[0]["patient_id"]
            )
            selected[str(representative)].append(f"representative_class_{class_name}")

        minimum_depth = validation["valid_depth"].min()
        maximum_depth = validation["valid_depth"].max()
        min_patient = (
            validation[validation["valid_depth"] == minimum_depth]
            .sort_values("patient_id", kind="mergesort")
            .iloc[0]["patient_id"]
        )
        max_patient = (
            validation[validation["valid_depth"] == maximum_depth]
            .sort_values("patient_id", kind="mergesort")
            .iloc[0]["patient_id"]
        )
        selected[str(min_patient)].append("global_min_valid_depth")
        selected[str(max_patient)].append("global_max_valid_depth")
        return {
            patient_id: sorted(reasons)
            for patient_id, reasons in sorted(selected.items())
        }

    def validate_manifest_files(self, manifest_path: Path) -> None:
        """Validate manifest structure, contract checksum, tensor files, and tensor hashes."""
        manifest = self.read_manifest(manifest_path)
        self._validate_manifest_top_level(manifest)

        expected_contract_hash = self.compute_file_sha256(self.preprocessing_contract_path)
        if manifest["preprocessing_contract_sha256"] != expected_contract_hash:
            raise ValueError("Preprocessing contract SHA-256 mismatch")

        patient_ids: list[str] = []
        class_names: set[str] = set()
        for patient_entry in manifest["patients"]:
            self._validate_patient_manifest_entry(patient_entry)
            patient_ids.append(str(patient_entry["patient_id"]))
            class_names.add(str(patient_entry["class_name"]))
            tensor_path = manifest_path.parent / str(patient_entry["tensor_filename"])
            if not tensor_path.is_file():
                raise FileNotFoundError(f"Missing golden tensor: {tensor_path}")
            if self.compute_file_sha256(tensor_path) != patient_entry["tensor_sha256"]:
                raise ValueError(f"{patient_entry['patient_id']}: tensor SHA-256 mismatch")
            tensor = np.load(tensor_path, allow_pickle=False)
            self._validate_tensor(str(patient_entry["patient_id"]), tensor)

        if len(patient_ids) != len(set(patient_ids)):
            raise ValueError("Duplicate patient IDs in golden manifest")
        if patient_ids != sorted(patient_ids):
            raise ValueError("Golden manifest patients are not sorted by patient ID")
        if len(class_names) < 5:
            raise ValueError("Golden manifest does not represent all five ACDC classes")

    def validate_source_checksums(
        self,
        manifest_path: Path,
        patients: tuple[AcdcPatient, ...],
    ) -> None:
        """Validate manifest source-file checksums against indexed real ACDC patients."""
        manifest = self.read_manifest(manifest_path)
        patients_by_id = {patient.patient_id: patient for patient in patients}
        for patient_entry in manifest["patients"]:
            patient_id = str(patient_entry["patient_id"])
            patient = patients_by_id[patient_id]
            checks = {
                "source_ed_sha256": patient.ed_path,
                "source_es_sha256": patient.es_path,
                "info_sha256": patient.info_path,
            }
            for key, path in checks.items():
                if self.compute_file_sha256(path) != patient_entry[key]:
                    raise ValueError(f"{patient_id}: {key} mismatch")

    def validate_production_parity(
        self,
        manifest_path: Path,
        patients: tuple[AcdcPatient, ...],
        preprocessor: PatientPreprocessor,
    ) -> None:
        """Validate fixtures are byte-exact outputs of current production preprocessing."""
        manifest = self.read_manifest(manifest_path)
        patients_by_id = {patient.patient_id: patient for patient in patients}
        for patient_entry in manifest["patients"]:
            patient_id = str(patient_entry["patient_id"])
            patient = patients_by_id[patient_id]
            result = preprocessor.preprocess(patient)
            tensor_path = manifest_path.parent / str(patient_entry["tensor_filename"])
            tensor = np.load(tensor_path, allow_pickle=False)
            if not np.array_equal(result.tensor, tensor):
                raise ValueError(f"{patient_id}: tensor differs from production preprocessing")
            expected_entry = self.build_patient_manifest_entry(
                patient=patient,
                result=result,
                tensor_filename=str(patient_entry["tensor_filename"]),
                tensor_sha256=str(patient_entry["tensor_sha256"]),
                selection_reasons=list(patient_entry["selection_reasons"]),
            )
            if expected_entry != patient_entry:
                raise ValueError(f"{patient_id}: manifest metadata differs from production output")

    def collect_fixture_hashes(self, manifest_path: Path) -> dict[str, str]:
        """Collect SHA-256 hashes for the manifest and all referenced tensor files."""
        manifest = self.read_manifest(manifest_path)
        hashes = {"manifest.json": self.compute_file_sha256(manifest_path)}
        for patient_entry in manifest["patients"]:
            tensor_filename = str(patient_entry["tensor_filename"])
            hashes[tensor_filename] = self.compute_file_sha256(
                manifest_path.parent / tensor_filename
            )
        return dict(sorted(hashes.items()))

    def build_manifest(self, patient_entries: list[dict[str, Any]]) -> dict[str, Any]:
        """Build the deterministic manifest document."""
        return {
            "dataset": "ACDC",
            "fixture_version": GOLDEN_FIXTURE_VERSION,
            "preprocessing_contract": "docs/preprocessing_contract.md",
            "preprocessing_contract_sha256": self.compute_file_sha256(
                self.preprocessing_contract_path
            ),
            "tensor_shape": list(TENSOR_SHAPE),
            "dtype": TENSOR_DTYPE,
            "source_array_order": SOURCE_ARRAY_ORDER,
            "model_array_order": MODEL_ARRAY_ORDER,
            "channel_semantics": CHANNEL_SEMANTICS,
            "patients": patient_entries,
        }

    def build_patient_manifest_entry(
        self,
        patient: AcdcPatient,
        result: PreprocessingResult,
        tensor_filename: str,
        tensor_sha256: str,
        selection_reasons: list[str],
    ) -> dict[str, Any]:
        """Build one deterministic patient entry for the golden manifest."""
        ed_shape, ed_spacing = self._source_shape_and_spacing(patient.ed_path)
        es_shape, es_spacing = self._source_shape_and_spacing(patient.es_path)
        return {
            "patient_id": patient.patient_id,
            "class_index": patient.class_index,
            "class_name": patient.class_name,
            "ed_frame": patient.ed_frame,
            "es_frame": patient.es_frame,
            "source_ed_filename": patient.ed_path.name,
            "source_es_filename": patient.es_path.name,
            "info_filename": patient.info_path.name,
            "source_ed_sha256": self.compute_file_sha256(patient.ed_path),
            "source_es_sha256": self.compute_file_sha256(patient.es_path),
            "info_sha256": self.compute_file_sha256(patient.info_path),
            "source_ed_shape_xyz": list(ed_shape),
            "source_es_shape_xyz": list(es_shape),
            "source_ed_spacing_xyz_mm": list(ed_spacing),
            "source_es_spacing_xyz_mm": list(es_spacing),
            "resampled_shape_xyz": list(result.resampled_shape_xyz),
            "target_spacing_xyz_mm": list(result.target_spacing_xyz),
            "valid_depth": result.valid_depth,
            "z_padding_lower": result.z_padding_lower,
            "z_padding_upper": result.z_padding_upper,
            "normalization": {
                "scope": NORMALIZATION_SCOPE,
                "clip_lower": result.clip_lower,
                "clip_upper": result.clip_upper,
                "mean": result.normalization_mean,
                "std": result.normalization_std,
            },
            "tensor_filename": tensor_filename,
            "tensor_shape": list(result.tensor.shape),
            "tensor_dtype": str(result.tensor.dtype),
            "tensor_sha256": tensor_sha256,
            "selection_reasons": sorted(selection_reasons),
        }

    @staticmethod
    def read_manifest(manifest_path: Path) -> dict[str, Any]:
        """Read a golden fixture manifest."""
        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
        if not isinstance(manifest, dict):
            raise ValueError(f"Malformed manifest: {manifest_path}")
        return manifest

    @staticmethod
    def write_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
        """Write a manifest with stable formatting."""
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def compute_file_sha256(path: Path) -> str:
        """Compute a SHA-256 hash by streaming file bytes."""
        digest = hashlib.sha256()
        with path.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def _load_or_compute_validation_depths(
        self,
        patients: tuple[AcdcPatient, ...],
        preprocessor: PatientPreprocessor,
    ) -> pd.DataFrame:
        """Load Phase 3 valid-depth metadata or compute it from production preprocessing."""
        if (
            self.phase3_validation_csv_path is not None
            and self.phase3_validation_csv_path.is_file()
        ):
            validation = pd.read_csv(self.phase3_validation_csv_path)
            required_columns = {"patient_id", "class", "valid_depth"}
            missing_columns = required_columns - set(validation.columns)
            if missing_columns:
                missing = ", ".join(sorted(missing_columns))
                raise ValueError(f"Phase 3 validation CSV missing columns: {missing}")
            return validation.loc[:, ["patient_id", "class", "valid_depth"]].copy()

        records = []
        for patient in patients:
            result = preprocessor.preprocess(patient)
            records.append(
                {
                    "patient_id": patient.patient_id,
                    "class": patient.class_name,
                    "valid_depth": result.valid_depth,
                }
            )
        return pd.DataFrame(records)

    def _source_shape_and_spacing(
        self,
        path: Path,
    ) -> tuple[tuple[int, int, int], tuple[float, float, float]]:
        """Read raw source image shape and spacing from a NIfTI header."""
        image = cast(nib.spatialimages.SpatialImage, nib.load(path))
        shape = tuple(int(value) for value in image.shape[:3])
        spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
        if len(shape) != 3 or len(spacing) != 3:
            raise ValueError(f"Expected 3D source image metadata: {path}")
        return shape, spacing

    def _validate_manifest_top_level(self, manifest: dict[str, Any]) -> None:
        """Validate stable top-level manifest fields."""
        expected = {
            "dataset": "ACDC",
            "fixture_version": GOLDEN_FIXTURE_VERSION,
            "preprocessing_contract": "docs/preprocessing_contract.md",
            "tensor_shape": list(TENSOR_SHAPE),
            "dtype": TENSOR_DTYPE,
            "source_array_order": SOURCE_ARRAY_ORDER,
            "model_array_order": MODEL_ARRAY_ORDER,
            "channel_semantics": CHANNEL_SEMANTICS,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f"Manifest field {key!r} mismatch")
        if not isinstance(manifest.get("patients"), list) or not manifest["patients"]:
            raise ValueError("Manifest must contain at least one patient")

    def _validate_patient_manifest_entry(self, patient_entry: dict[str, Any]) -> None:
        """Validate required fields for one manifest patient entry."""
        required_fields = {
            "patient_id",
            "class_index",
            "class_name",
            "ed_frame",
            "es_frame",
            "source_ed_filename",
            "source_es_filename",
            "info_filename",
            "source_ed_sha256",
            "source_es_sha256",
            "info_sha256",
            "source_ed_shape_xyz",
            "source_es_shape_xyz",
            "source_ed_spacing_xyz_mm",
            "source_es_spacing_xyz_mm",
            "resampled_shape_xyz",
            "target_spacing_xyz_mm",
            "valid_depth",
            "z_padding_lower",
            "z_padding_upper",
            "normalization",
            "tensor_filename",
            "tensor_shape",
            "tensor_dtype",
            "tensor_sha256",
            "selection_reasons",
        }
        missing_fields = required_fields - set(patient_entry)
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"Patient manifest entry missing fields: {missing}")
        for filename_field in [
            "source_ed_filename",
            "source_es_filename",
            "info_filename",
            "tensor_filename",
        ]:
            if Path(str(patient_entry[filename_field])).is_absolute():
                raise ValueError(f"{patient_entry['patient_id']}: absolute path in manifest")
        if patient_entry["tensor_shape"] != list(TENSOR_SHAPE):
            raise ValueError(f"{patient_entry['patient_id']}: manifest tensor shape mismatch")
        if patient_entry["tensor_dtype"] != TENSOR_DTYPE:
            raise ValueError(f"{patient_entry['patient_id']}: manifest tensor dtype mismatch")
        normalization = patient_entry["normalization"]
        if not isinstance(normalization, dict) or normalization.get("scope") != NORMALIZATION_SCOPE:
            raise ValueError(f"{patient_entry['patient_id']}: normalization scope mismatch")
        if not patient_entry["selection_reasons"]:
            raise ValueError(f"{patient_entry['patient_id']}: missing selection reason")

    def _validate_tensor(self, patient_id: str, tensor: NDArray[np.float32]) -> None:
        """Validate the golden tensor contract."""
        if tensor.shape != TENSOR_SHAPE:
            raise ValueError(f"{patient_id}: tensor shape mismatch")
        if tensor.dtype != np.float32:
            raise ValueError(f"{patient_id}: tensor dtype mismatch")
        if not tensor.flags.c_contiguous:
            raise ValueError(f"{patient_id}: tensor is not C-contiguous")
        if not np.isfinite(tensor).all():
            raise ValueError(f"{patient_id}: tensor contains non-finite values")
