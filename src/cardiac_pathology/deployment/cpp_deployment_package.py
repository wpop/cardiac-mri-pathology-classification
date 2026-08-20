"""Build and validate the portable C++ deployment package."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import onnx
import onnxruntime as ort  # type: ignore[import-untyped]
import yaml
from onnx import TensorProto

EXPECTED_MODEL_SHA256 = "d308dc7905f633f9f69f5d784cc6a0e04da9e552a2ece5464c8841dbb6328c5b"
MODEL_ID = "cardiac_mri_pathology"
PACKAGE_RELATIVE_PATH = Path("artifacts/deployment/package/cardiac_mri_pathology")
EXPECTED_INPUT_SHAPE = ("N", 2, 14, 144, 144)
EXPECTED_OUTPUT_SHAPE = ("N", 5)
EXPECTED_SAMPLE_SHAPE = (2, 14, 144, 144)
EXPECTED_BATCH_SHAPE = (1, 2, 14, 144, 144)
EXPECTED_LOGITS_SHAPE = (1, 5)


@dataclass(frozen=True, slots=True)
class PackageBuildSummary:
    """Summary of one C++ deployment package build."""

    package_dir: Path
    model_sha256: str
    preprocessing_patient_count: int
    inference_patient_count: int
    input_shape: tuple[int, ...]
    logits_shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PackageValidationSummary:
    """Summary of one C++ deployment package validation."""

    package_dir: Path
    model_sha256: str
    preprocessing_patient_count: int
    inference_patient_count: int
    input_shape: tuple[int, ...]
    logits_shape: tuple[int, ...]
    predicted_class_parity: bool
    onnx_logits_parity: bool


@dataclass(frozen=True, slots=True)
class OnnxTensorContract:
    """ONNX tensor name, dtype, and shape."""

    name: str
    dtype: str
    shape: tuple[str | int, ...]


@dataclass(frozen=True, slots=True)
class OnnxContract:
    """Relevant ONNX graph contract."""

    opset: int
    input: OnnxTensorContract
    output: OnnxTensorContract
    softmax_inside_model: bool


class CppDeploymentPackageBuilder:
    """Build and validate the handoff package for ONNX Runtime C++ integration."""

    def __init__(
        self,
        repository_root: Path,
        package_dir: Path | None = None,
    ) -> None:
        self.repository_root = repository_root.resolve()
        self.package_dir = (
            package_dir.resolve()
            if package_dir is not None
            else self.repository_root / PACKAGE_RELATIVE_PATH
        )
        self.source_onnx_path = self.repository_root / "artifacts/deployment/classifier.onnx"
        self.source_metadata_path = (
            self.repository_root / "artifacts/deployment/phase10_deployment_metadata.json"
        )
        self.config_path = self.repository_root / "configs/default.yaml"
        self.class_mapping_path = self.repository_root / "configs/class_mapping.json"
        self.preprocessing_contract_path = (
            self.repository_root / "docs/preprocessing_contract.md"
        )
        self.onnx_contract_path = self.repository_root / "docs/onnx_deployment_contract.md"
        self.golden_preprocessing_dir = self.repository_root / "tests/fixtures/golden"
        self.golden_preprocessing_manifest_path = self.golden_preprocessing_dir / "manifest.json"

    def build(self) -> PackageBuildSummary:
        """Build the complete deterministic deployment package."""
        source_model_hash = sha256_file(self.source_onnx_path)
        if source_model_hash != EXPECTED_MODEL_SHA256:
            raise ValueError(
                "ONNX SHA-256 mismatch before packaging: "
                f"{source_model_hash} != {EXPECTED_MODEL_SHA256}"
            )

        config = self._load_yaml_mapping(self.config_path)
        metadata = self._load_json_mapping(self.source_metadata_path)
        class_mapping = load_class_mapping(self.class_mapping_path)
        onnx_contract = inspect_onnx_contract(self.source_onnx_path)
        self._validate_source_consistency(
            config=config,
            metadata=metadata,
            class_mapping=class_mapping,
            onnx_contract=onnx_contract,
            model_sha256=source_model_hash,
        )

        if self.package_dir.exists():
            shutil.rmtree(self.package_dir)
        self._create_package_directories()

        self._copy_runtime_files()
        self._write_deployment_json(
            config=config,
            metadata=metadata,
            class_mapping=class_mapping,
            onnx_contract=onnx_contract,
            model_sha256=source_model_hash,
        )
        preprocessing_count = self._copy_golden_preprocessing_references()
        inference_count, input_shape, logits_shape = self._generate_inference_references(
            class_mapping=class_mapping,
            model_sha256=source_model_hash,
            metadata=metadata,
        )
        self.validate()
        return PackageBuildSummary(
            package_dir=self.package_dir,
            model_sha256=source_model_hash,
            preprocessing_patient_count=preprocessing_count,
            inference_patient_count=inference_count,
            input_shape=input_shape,
            logits_shape=logits_shape,
        )

    def validate(self) -> PackageValidationSummary:
        """Validate package contents and ONNX Runtime raw-logit parity."""
        deployment = self._load_json_mapping(self.package_dir / "deployment.json")
        package_model_path = self.package_dir / "classifier.onnx"
        model_sha256 = sha256_file(package_model_path)
        if model_sha256 != EXPECTED_MODEL_SHA256:
            raise ValueError(f"Packaged model SHA-256 mismatch: {model_sha256}")

        onnx_contract = inspect_onnx_contract(package_model_path)
        class_mapping = load_class_mapping(self.package_dir / "class_mapping.json")
        self._validate_deployment_json(
            deployment=deployment,
            class_mapping=class_mapping,
            onnx_contract=onnx_contract,
            model_sha256=model_sha256,
        )
        preprocessing_count = self._validate_preprocessing_package()
        inference_count, input_shape, logits_shape = self._validate_inference_package(
            class_mapping=class_mapping,
            model_sha256=model_sha256,
            deployment=deployment,
        )
        return PackageValidationSummary(
            package_dir=self.package_dir,
            model_sha256=model_sha256,
            preprocessing_patient_count=preprocessing_count,
            inference_patient_count=inference_count,
            input_shape=input_shape,
            logits_shape=logits_shape,
            predicted_class_parity=True,
            onnx_logits_parity=True,
        )

    def collect_package_hashes(self) -> dict[str, str]:
        """Return SHA-256 hashes for every file in the package."""
        if not self.package_dir.is_dir():
            raise FileNotFoundError(f"Package directory is missing: {self.package_dir}")
        hashes: dict[str, str] = {}
        for path in sorted(self.package_dir.rglob("*")):
            if path.is_file():
                relative_path = path.relative_to(self.package_dir).as_posix()
                hashes[relative_path] = sha256_file(path)
        return hashes

    def validate_deterministic_regeneration(self) -> bool:
        """Build the package twice and require identical file hashes."""
        self.build()
        first_hashes = self.collect_package_hashes()
        self.build()
        second_hashes = self.collect_package_hashes()
        return first_hashes == second_hashes

    def _create_package_directories(self) -> None:
        """Create the package directory tree."""
        for relative_path in [
            Path("docs"),
            Path("golden/preprocessing"),
            Path("golden/inference/inputs"),
            Path("golden/inference/logits"),
        ]:
            (self.package_dir / relative_path).mkdir(parents=True, exist_ok=True)

    def _copy_runtime_files(self) -> None:
        """Copy runtime and developer documentation files into the package."""
        copy_file(self.source_onnx_path, self.package_dir / "classifier.onnx")
        copy_file(self.class_mapping_path, self.package_dir / "class_mapping.json")
        copy_file(
            self.preprocessing_contract_path,
            self.package_dir / "docs/preprocessing_contract.md",
        )
        copy_file(
            self.onnx_contract_path,
            self.package_dir / "docs/onnx_deployment_contract.md",
        )

    def _copy_golden_preprocessing_references(self) -> int:
        """Copy existing golden preprocessing fixtures and verify their hashes."""
        manifest = self._load_json_mapping(self.golden_preprocessing_manifest_path)
        patients = require_list(manifest, "patients")
        destination_dir = self.package_dir / "golden/preprocessing"
        copy_file(self.golden_preprocessing_manifest_path, destination_dir / "manifest.json")
        for patient in patients:
            entry = require_mapping_value(patient, "golden patient")
            tensor_filename = require_str(entry, "tensor_filename")
            tensor_path = self.golden_preprocessing_dir / tensor_filename
            if sha256_file(tensor_path) != require_str(entry, "tensor_sha256"):
                raise ValueError(f"Golden preprocessing tensor checksum mismatch: {tensor_path}")
            copy_file(tensor_path, destination_dir / tensor_filename)
            copied_hash = sha256_file(destination_dir / tensor_filename)
            if copied_hash != require_str(entry, "tensor_sha256"):
                raise ValueError(
                    f"Copied preprocessing tensor checksum mismatch: {tensor_filename}"
                )
        return len(patients)

    def _generate_inference_references(
        self,
        *,
        class_mapping: dict[int, str],
        model_sha256: str,
        metadata: Mapping[str, object],
    ) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
        """Generate batched inputs and raw ONNX logits from preprocessing fixtures."""
        preprocessing_manifest = self._load_json_mapping(
            self.package_dir / "golden/preprocessing/manifest.json"
        )
        patients = require_list(preprocessing_manifest, "patients")
        input_dir = self.package_dir / "golden/inference/inputs"
        logits_dir = self.package_dir / "golden/inference/logits"
        session = ort.InferenceSession(
            str(self.package_dir / "classifier.onnx"),
            providers=["CPUExecutionProvider"],
        )
        patient_entries: list[dict[str, object]] = []
        last_input_shape: tuple[int, ...] = EXPECTED_BATCH_SHAPE
        last_logits_shape: tuple[int, ...] = EXPECTED_LOGITS_SHAPE

        for patient in patients:
            entry = require_mapping_value(patient, "golden preprocessing patient")
            patient_id = require_str(entry, "patient_id")
            class_name = require_str(entry, "class_name")
            tensor_filename = require_str(entry, "tensor_filename")
            tensor_path = self.package_dir / "golden/preprocessing" / tensor_filename
            tensor = load_preprocessed_tensor(tensor_path)
            batched_input = np.ascontiguousarray(np.expand_dims(tensor, axis=0))
            if batched_input.shape != EXPECTED_BATCH_SHAPE:
                raise ValueError(f"{patient_id}: unexpected batched input shape")

            outputs = session.run(["logits"], {"cine_mri": batched_input})
            logits = require_numpy_array(outputs[0], f"{patient_id} logits")
            if logits.shape != EXPECTED_LOGITS_SHAPE or logits.dtype != np.float32:
                raise ValueError(f"{patient_id}: unexpected logits contract")
            if not np.isfinite(logits).all():
                raise ValueError(f"{patient_id}: logits contain non-finite values")

            input_filename = f"{patient_id}.npy"
            logits_filename = f"{patient_id}_logits.npy"
            input_path = input_dir / input_filename
            logits_path = logits_dir / logits_filename
            np.save(input_path, batched_input, allow_pickle=False)
            np.save(logits_path, logits, allow_pickle=False)

            predicted_class_index = int(np.argmax(logits, axis=1)[0])
            patient_entries.append(
                {
                    "class_name": class_name,
                    "input_dtype": str(batched_input.dtype),
                    "input_filename": f"inputs/{input_filename}",
                    "input_sha256": sha256_file(input_path),
                    "input_shape": list(batched_input.shape),
                    "logits_dtype": str(logits.dtype),
                    "logits_filename": f"logits/{logits_filename}",
                    "logits_sha256": sha256_file(logits_path),
                    "logits_shape": list(logits.shape),
                    "patient_id": patient_id,
                    "predicted_class_index": predicted_class_index,
                    "predicted_class_name": class_mapping[predicted_class_index],
                }
            )
            last_input_shape = tuple(int(value) for value in batched_input.shape)
            last_logits_shape = tuple(int(value) for value in logits.shape)

        parity = require_mapping(metadata, "parity_validation")
        manifest: dict[str, object] = {
            "absolute_tolerance": require_float(parity, "absolute_logit_tolerance"),
            "class_order": classes_as_list(class_mapping),
            "input_name": "cine_mri",
            "model_sha256": model_sha256,
            "output_name": "logits",
            "patients": patient_entries,
            "relative_tolerance": None,
            "schema_version": 1,
        }
        write_json(self.package_dir / "golden/inference/manifest.json", manifest)
        return len(patient_entries), last_input_shape, last_logits_shape

    def _write_deployment_json(
        self,
        *,
        config: Mapping[str, object],
        metadata: Mapping[str, object],
        class_mapping: dict[int, str],
        onnx_contract: OnnxContract,
        model_sha256: str,
    ) -> None:
        """Write deterministic machine-readable deployment metadata."""
        preprocessing = require_mapping(config, "preprocessing")
        target_spacing = require_mapping(preprocessing, "target_spacing_mm")
        target_shape = require_mapping(preprocessing, "target_shape")
        intensity = require_mapping(preprocessing, "intensity")
        array_config = require_mapping(preprocessing, "array")
        padding = require_mapping(preprocessing, "padding")
        crop = require_mapping(preprocessing, "crop")
        interpolation = require_mapping(preprocessing, "interpolation")
        parity = require_mapping(metadata, "parity_validation")
        deployment = {
            "classes": classes_as_list(class_mapping),
            "model_format": "onnx",
            "model_id": MODEL_ID,
            "onnx": {
                "filename": "classifier.onnx",
                "input": {
                    "dtype": onnx_contract.input.dtype,
                    "name": onnx_contract.input.name,
                    "shape": list(onnx_contract.input.shape),
                },
                "opset": onnx_contract.opset,
                "output": {
                    "dtype": onnx_contract.output.dtype,
                    "name": onnx_contract.output.name,
                    "shape": list(onnx_contract.output.shape),
                },
                "sha256": model_sha256,
                "softmax_inside_model": onnx_contract.softmax_inside_model,
            },
            "preprocessing": {
                "boundary_mode": "nearest",
                "channel_order": ["ED", "ES"],
                "clip_percentiles": [
                    require_float(intensity, "lower_percentile"),
                    require_float(intensity, "upper_percentile"),
                ],
                "final_shape_dhw": [
                    require_int(target_shape, "d"),
                    require_int(target_shape, "h"),
                    require_int(target_shape, "w"),
                ],
                "interpolation": require_str(interpolation, "image"),
                "model_array_order": require_str(array_config, "model_order"),
                "normalization_epsilon": require_float(intensity, "epsilon"),
                "source_array_order": require_str(array_config, "source_order"),
                "target_orientation": require_str(preprocessing, "orientation"),
                "target_spacing_xyz_mm": [
                    require_float(target_spacing, "x"),
                    require_float(target_spacing, "y"),
                    require_float(target_spacing, "z"),
                ],
                "tensor_layout": "NCDHW",
                "xy_crop": f"geometric_{require_str(crop, 'strategy')}",
                "z_padding": "center_after_normalization",
                "z_padding_value": require_float(padding, "z_value"),
            },
            "schema_version": 1,
            "validation": {
                "absolute_logit_tolerance": require_float(parity, "absolute_logit_tolerance"),
                "predicted_class_index_must_match": require_bool(
                    parity,
                    "predicted_class_index_must_match",
                ),
            },
        }
        write_json(self.package_dir / "deployment.json", deployment)

    def _validate_source_consistency(
        self,
        *,
        config: Mapping[str, object],
        metadata: Mapping[str, object],
        class_mapping: dict[int, str],
        onnx_contract: OnnxContract,
        model_sha256: str,
    ) -> None:
        """Validate source configuration, metadata, and ONNX graph agree."""
        deployment = require_mapping(config, "deployment")
        if require_str(deployment, "input_name") != onnx_contract.input.name:
            raise ValueError("default.yaml input name disagrees with ONNX")
        if require_str(deployment, "output_name") != onnx_contract.output.name:
            raise ValueError("default.yaml output name disagrees with ONNX")
        if require_int(metadata, "opset") != onnx_contract.opset:
            raise ValueError("Phase 10 metadata opset disagrees with ONNX")
        if require_str(metadata, "input_name") != onnx_contract.input.name:
            raise ValueError("Phase 10 metadata input name disagrees with ONNX")
        if require_str(metadata, "output_name") != onnx_contract.output.name:
            raise ValueError("Phase 10 metadata output name disagrees with ONNX")
        if tuple(require_list(metadata, "input_shape")) != EXPECTED_INPUT_SHAPE:
            raise ValueError("Phase 10 input shape disagrees with expected contract")
        if tuple(require_list(metadata, "output_shape")) != EXPECTED_OUTPUT_SHAPE:
            raise ValueError("Phase 10 output shape disagrees with expected contract")
        if onnx_contract.input.shape != EXPECTED_INPUT_SHAPE:
            raise ValueError(f"ONNX input shape mismatch: {onnx_contract.input.shape}")
        if onnx_contract.output.shape != EXPECTED_OUTPUT_SHAPE:
            raise ValueError(f"ONNX output shape mismatch: {onnx_contract.output.shape}")
        if onnx_contract.input.dtype != "float32" or onnx_contract.output.dtype != "float32":
            raise ValueError("ONNX dtype mismatch")
        if onnx_contract.softmax_inside_model:
            raise ValueError("ONNX graph unexpectedly contains Softmax")
        if classes_as_mapping(require_mapping(metadata, "class_mapping")) != class_mapping:
            raise ValueError("Phase 10 class mapping disagrees with class_mapping.json")
        if model_sha256 != EXPECTED_MODEL_SHA256:
            raise ValueError("Model SHA-256 disagrees with expected package contract")
        self._validate_preprocessing_config(config)

    def _validate_preprocessing_config(self, config: Mapping[str, object]) -> None:
        """Validate frozen preprocessing constants used by the package."""
        preprocessing = require_mapping(config, "preprocessing")
        target_spacing = require_mapping(preprocessing, "target_spacing_mm")
        target_shape = require_mapping(preprocessing, "target_shape")
        intensity = require_mapping(preprocessing, "intensity")
        padding = require_mapping(preprocessing, "padding")
        array_config = require_mapping(preprocessing, "array")
        if require_str(preprocessing, "orientation") != "LPS":
            raise ValueError("Unexpected target orientation")
        if (
            require_float(target_spacing, "x"),
            require_float(target_spacing, "y"),
            require_float(target_spacing, "z"),
        ) != (1.5, 1.5, 7.5):
            raise ValueError("Unexpected target spacing")
        if (
            require_int(target_shape, "d"),
            require_int(target_shape, "h"),
            require_int(target_shape, "w"),
        ) != (14, 144, 144):
            raise ValueError("Unexpected target shape")
        if (
            require_float(intensity, "lower_percentile"),
            require_float(intensity, "upper_percentile"),
        ) != (0.5, 99.5):
            raise ValueError("Unexpected clipping percentiles")
        if require_float(intensity, "epsilon") != 1.0e-6:
            raise ValueError("Unexpected normalization epsilon")
        if require_float(padding, "z_value") != 0.0:
            raise ValueError("Unexpected Z padding value")
        if (
            require_str(array_config, "source_order"),
            require_str(array_config, "model_order"),
            require_str(array_config, "dtype"),
        ) != ("XYZ", "ZYX", "float32"):
            raise ValueError("Unexpected array contract")

    def _validate_deployment_json(
        self,
        *,
        deployment: Mapping[str, object],
        class_mapping: dict[int, str],
        onnx_contract: OnnxContract,
        model_sha256: str,
    ) -> None:
        """Validate deployment.json against packaged files."""
        if require_int(deployment, "schema_version") != 1:
            raise ValueError("deployment.json schema_version mismatch")
        if require_str(deployment, "model_id") != MODEL_ID:
            raise ValueError("deployment.json model_id mismatch")
        if require_list(deployment, "classes") != classes_as_list(class_mapping):
            raise ValueError("deployment.json class list mismatch")
        onnx_metadata = require_mapping(deployment, "onnx")
        if require_str(onnx_metadata, "sha256") != model_sha256:
            raise ValueError("deployment.json ONNX SHA-256 mismatch")
        if require_int(onnx_metadata, "opset") != onnx_contract.opset:
            raise ValueError("deployment.json opset mismatch")
        if require_bool(onnx_metadata, "softmax_inside_model"):
            raise ValueError("deployment.json softmax flag mismatch")
        if require_mapping(onnx_metadata, "input") != {
            "dtype": onnx_contract.input.dtype,
            "name": onnx_contract.input.name,
            "shape": list(onnx_contract.input.shape),
        }:
            raise ValueError("deployment.json input contract mismatch")
        if require_mapping(onnx_metadata, "output") != {
            "dtype": onnx_contract.output.dtype,
            "name": onnx_contract.output.name,
            "shape": list(onnx_contract.output.shape),
        }:
            raise ValueError("deployment.json output contract mismatch")

    def _validate_preprocessing_package(self) -> int:
        """Validate copied golden preprocessing fixtures."""
        manifest = self._load_json_mapping(self.package_dir / "golden/preprocessing/manifest.json")
        patients = require_list(manifest, "patients")
        for patient in patients:
            entry = require_mapping_value(patient, "preprocessing patient")
            tensor_filename = require_str(entry, "tensor_filename")
            tensor_path = self.package_dir / "golden/preprocessing" / tensor_filename
            if sha256_file(tensor_path) != require_str(entry, "tensor_sha256"):
                raise ValueError(f"Preprocessing package checksum mismatch: {tensor_filename}")
            _tensor = load_preprocessed_tensor(tensor_path)
        return len(patients)

    def _validate_inference_package(
        self,
        *,
        class_mapping: dict[int, str],
        model_sha256: str,
        deployment: Mapping[str, object],
    ) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
        """Validate saved inference inputs, logits, and ONNX Runtime parity."""
        manifest = self._load_json_mapping(self.package_dir / "golden/inference/manifest.json")
        if require_int(manifest, "schema_version") != 1:
            raise ValueError("Inference manifest schema_version mismatch")
        if require_str(manifest, "model_sha256") != model_sha256:
            raise ValueError("Inference manifest model SHA-256 mismatch")
        if require_str(manifest, "input_name") != "cine_mri":
            raise ValueError("Inference manifest input name mismatch")
        if require_str(manifest, "output_name") != "logits":
            raise ValueError("Inference manifest output name mismatch")
        if require_list(manifest, "class_order") != classes_as_list(class_mapping):
            raise ValueError("Inference manifest class order mismatch")
        tolerance = require_float(manifest, "absolute_tolerance")
        validation = require_mapping(deployment, "validation")
        if tolerance != require_float(validation, "absolute_logit_tolerance"):
            raise ValueError("Inference tolerance disagrees with deployment.json")

        session = ort.InferenceSession(
            str(self.package_dir / "classifier.onnx"),
            providers=["CPUExecutionProvider"],
        )
        patients = require_list(manifest, "patients")
        last_input_shape: tuple[int, ...] = EXPECTED_BATCH_SHAPE
        last_logits_shape: tuple[int, ...] = EXPECTED_LOGITS_SHAPE
        for patient in patients:
            entry = require_mapping_value(patient, "inference patient")
            patient_id = require_str(entry, "patient_id")
            input_path = self.package_dir / "golden/inference" / require_str(
                entry,
                "input_filename",
            )
            logits_path = self.package_dir / "golden/inference" / require_str(
                entry,
                "logits_filename",
            )
            if sha256_file(input_path) != require_str(entry, "input_sha256"):
                raise ValueError(f"{patient_id}: inference input checksum mismatch")
            if sha256_file(logits_path) != require_str(entry, "logits_sha256"):
                raise ValueError(f"{patient_id}: inference logits checksum mismatch")

            batched_input = load_batched_input(input_path)
            expected_logits = load_logits(logits_path)
            actual_outputs = session.run(["logits"], {"cine_mri": batched_input})
            actual_logits = require_numpy_array(actual_outputs[0], f"{patient_id} actual logits")
            if actual_logits.shape != expected_logits.shape:
                raise ValueError(f"{patient_id}: logits shape mismatch")
            if not np.allclose(actual_logits, expected_logits, atol=tolerance, rtol=0.0):
                raise ValueError(f"{patient_id}: ONNX Runtime logits parity failed")
            predicted_class_index = int(np.argmax(actual_logits, axis=1)[0])
            if predicted_class_index != require_int(entry, "predicted_class_index"):
                raise ValueError(f"{patient_id}: predicted class index mismatch")
            if class_mapping[predicted_class_index] != require_str(
                entry,
                "predicted_class_name",
            ):
                raise ValueError(f"{patient_id}: predicted class name mismatch")
            last_input_shape = tuple(int(value) for value in batched_input.shape)
            last_logits_shape = tuple(int(value) for value in expected_logits.shape)
        return len(patients), last_input_shape, last_logits_shape

    @staticmethod
    def _load_json_mapping(path: Path) -> Mapping[str, object]:
        """Load one JSON object."""
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object: {path}")
        return cast(Mapping[str, object], value)

    @staticmethod
    def _load_yaml_mapping(path: Path) -> Mapping[str, object]:
        """Load one YAML mapping."""
        with path.open("r", encoding="utf-8") as file:
            value = yaml.safe_load(file)
        if not isinstance(value, dict):
            raise ValueError(f"Expected YAML mapping: {path}")
        return cast(Mapping[str, object], value)


def inspect_onnx_contract(path: Path) -> OnnxContract:
    """Inspect the actual ONNX graph input/output contract."""
    model = onnx.load(path)
    onnx.checker.check_model(model)
    opsets = [opset.version for opset in model.opset_import if opset.domain in ("", "ai.onnx")]
    if len(opsets) != 1:
        raise ValueError(f"Expected one ai.onnx opset import, got {opsets}")
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("Expected one ONNX graph input and one output")
    return OnnxContract(
        opset=int(opsets[0]),
        input=inspect_tensor_contract(model.graph.input[0]),
        output=inspect_tensor_contract(model.graph.output[0]),
        softmax_inside_model=any(node.op_type == "Softmax" for node in model.graph.node),
    )


def inspect_tensor_contract(value_info: onnx.ValueInfoProto) -> OnnxTensorContract:
    """Convert ONNX value info to the package tensor contract."""
    tensor_type = value_info.type.tensor_type
    dtype = tensor_dtype_name(int(tensor_type.elem_type))
    shape: list[str | int] = []
    for index, dimension in enumerate(tensor_type.shape.dim):
        if dimension.dim_value:
            shape.append(int(dimension.dim_value))
        elif dimension.dim_param:
            shape.append("N" if index == 0 else str(dimension.dim_param))
        else:
            shape.append("N" if index == 0 else "?")
    return OnnxTensorContract(name=value_info.name, dtype=dtype, shape=tuple(shape))


def tensor_dtype_name(elem_type: int) -> str:
    """Return the deployment dtype name for one ONNX tensor element type."""
    if elem_type == TensorProto.FLOAT:
        return "float32"
    raise ValueError(f"Unsupported ONNX tensor dtype: {elem_type}")


def load_class_mapping(path: Path) -> dict[int, str]:
    """Load class_mapping.json as an integer-indexed mapping."""
    with path.open("r", encoding="utf-8") as file:
        raw_mapping = json.load(file)
    return classes_as_mapping(require_mapping_value(raw_mapping, str(path)))


def classes_as_mapping(mapping: Mapping[str, object]) -> dict[int, str]:
    """Convert a JSON class mapping to sorted integer keys."""
    class_mapping: dict[int, str] = {}
    for raw_index, raw_name in mapping.items():
        if not isinstance(raw_name, str):
            raise ValueError(f"Class name must be a string: {raw_index}")
        class_mapping[int(raw_index)] = raw_name
    if sorted(class_mapping) != list(range(len(class_mapping))):
        raise ValueError(f"Class mapping indices must be contiguous: {class_mapping}")
    return dict(sorted(class_mapping.items()))


def classes_as_list(class_mapping: Mapping[int, str]) -> list[dict[str, object]]:
    """Return class metadata as a deterministic list."""
    return [
        {"index": class_index, "name": class_mapping[class_index]}
        for class_index in sorted(class_mapping)
    ]


def copy_file(source: Path, destination: Path) -> None:
    """Copy one file without creating symlinks."""
    if not source.is_file():
        raise FileNotFoundError(f"Source file is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def load_preprocessed_tensor(path: Path) -> np.ndarray:
    """Load and validate one golden preprocessing tensor."""
    tensor = require_numpy_array(np.load(path, allow_pickle=False), str(path))
    if tensor.shape != EXPECTED_SAMPLE_SHAPE:
        raise ValueError(f"Unexpected preprocessing tensor shape: {path}: {tensor.shape}")
    if tensor.dtype != np.float32:
        raise ValueError(f"Unexpected preprocessing tensor dtype: {path}: {tensor.dtype}")
    if not tensor.flags.c_contiguous:
        raise ValueError(f"Preprocessing tensor must be C-contiguous: {path}")
    if not np.isfinite(tensor).all():
        raise ValueError(f"Preprocessing tensor contains non-finite values: {path}")
    return tensor


def load_batched_input(path: Path) -> np.ndarray:
    """Load and validate one saved batched inference input."""
    array = require_numpy_array(np.load(path, allow_pickle=False), str(path))
    if array.shape != EXPECTED_BATCH_SHAPE or array.dtype != np.float32:
        raise ValueError(f"Unexpected inference input contract: {path}")
    if not array.flags.c_contiguous or not np.isfinite(array).all():
        raise ValueError(f"Invalid inference input tensor: {path}")
    return array


def load_logits(path: Path) -> np.ndarray:
    """Load and validate one saved raw-logits reference."""
    array = require_numpy_array(np.load(path, allow_pickle=False), str(path))
    if array.shape != EXPECTED_LOGITS_SHAPE or array.dtype != np.float32:
        raise ValueError(f"Unexpected logits contract: {path}")
    if not np.isfinite(array).all():
        raise ValueError(f"Logits contain non-finite values: {path}")
    return array


def require_numpy_array(value: object, name: str) -> np.ndarray:
    """Require a value to be a NumPy array."""
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    return value


def sha256_file(path: Path) -> str:
    """Compute the SHA-256 hash of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write deterministic JSON with a trailing newline."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require_mapping(mapping: Mapping[str, object], key: str) -> Mapping[str, object]:
    """Require a mapping-valued key."""
    value = mapping.get(key)
    return require_mapping_value(value, key)


def require_mapping_value(value: object, name: str) -> Mapping[str, object]:
    """Require one object to be a string-keyed mapping."""
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must contain string keys")
    return cast(Mapping[str, object], value)


def require_list(mapping: Mapping[str, object], key: str) -> list[object]:
    """Require a list-valued key."""
    value = mapping.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return value


def require_str(mapping: Mapping[str, object], key: str) -> str:
    """Require a string-valued key."""
    value = mapping.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def require_int(mapping: Mapping[str, object], key: str) -> int:
    """Require an integer-valued key."""
    value = mapping.get(key)
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def require_float(mapping: Mapping[str, object], key: str) -> float:
    """Require a numeric-valued key."""
    value = mapping.get(key)
    if not isinstance(value, int | float):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def require_bool(mapping: Mapping[str, object], key: str) -> bool:
    """Require a boolean-valued key."""
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value
