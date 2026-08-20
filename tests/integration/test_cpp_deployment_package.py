"""Integration tests for the portable C++ deployment package."""

from pathlib import Path

import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]

from cardiac_pathology.deployment.cpp_deployment_package import (
    EXPECTED_BATCH_SHAPE,
    EXPECTED_LOGITS_SHAPE,
    EXPECTED_MODEL_SHA256,
    CppDeploymentPackageBuilder,
    classes_as_list,
    load_class_mapping,
    load_logits,
    require_list,
    require_mapping,
    require_mapping_value,
    sha256_file,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPOSITORY_ROOT / "artifacts/deployment/package/cardiac_mri_pathology"


def test_cpp_deployment_package_builds_and_validates() -> None:
    """Package contains deterministic runtime and validation artifacts."""
    builder = CppDeploymentPackageBuilder(REPOSITORY_ROOT)
    first_summary = builder.build()
    first_hashes = builder.collect_package_hashes()
    second_summary = builder.build()
    second_hashes = builder.collect_package_hashes()
    validation = builder.validate()

    assert first_summary.package_dir == PACKAGE_DIR
    assert second_summary.model_sha256 == EXPECTED_MODEL_SHA256
    assert validation.model_sha256 == EXPECTED_MODEL_SHA256
    assert first_hashes == second_hashes
    assert validation.preprocessing_patient_count == 7
    assert validation.inference_patient_count == 7
    assert validation.input_shape == EXPECTED_BATCH_SHAPE
    assert validation.logits_shape == EXPECTED_LOGITS_SHAPE
    assert validation.predicted_class_parity
    assert validation.onnx_logits_parity


def test_cpp_deployment_json_and_onnx_runtime_contract() -> None:
    """deployment.json agrees with packaged class mapping and ONNX Runtime I/O."""
    CppDeploymentPackageBuilder(REPOSITORY_ROOT).build()
    deployment = CppDeploymentPackageBuilder._load_json_mapping(PACKAGE_DIR / "deployment.json")
    class_mapping = load_class_mapping(PACKAGE_DIR / "class_mapping.json")
    session = ort.InferenceSession(
        str(PACKAGE_DIR / "classifier.onnx"),
        providers=["CPUExecutionProvider"],
    )

    assert sha256_file(PACKAGE_DIR / "classifier.onnx") == EXPECTED_MODEL_SHA256
    assert require_list(deployment, "classes") == classes_as_list(class_mapping)

    onnx_metadata = require_mapping(deployment, "onnx")
    onnx_input = require_mapping(onnx_metadata, "input")
    onnx_output = require_mapping(onnx_metadata, "output")
    session_input = session.get_inputs()[0]
    session_output = session.get_outputs()[0]

    assert onnx_input["name"] == session_input.name == "cine_mri"
    assert onnx_input["dtype"] == "float32"
    assert onnx_input["shape"] == ["N", 2, 14, 144, 144]
    assert session_input.shape[1:] == [2, 14, 144, 144]
    assert onnx_output["name"] == session_output.name == "logits"
    assert onnx_output["dtype"] == "float32"
    assert onnx_output["shape"] == ["N", 5]
    assert session_output.shape[1:] == [5]


def test_cpp_deployment_golden_inference_logits_parity() -> None:
    """Saved raw logits match ONNX Runtime outputs for real golden tensors."""
    CppDeploymentPackageBuilder(REPOSITORY_ROOT).build()
    manifest = CppDeploymentPackageBuilder._load_json_mapping(
        PACKAGE_DIR / "golden/inference/manifest.json"
    )
    tolerance = float(manifest["absolute_tolerance"])
    session = ort.InferenceSession(
        str(PACKAGE_DIR / "classifier.onnx"),
        providers=["CPUExecutionProvider"],
    )

    for patient in require_list(manifest, "patients"):
        entry = require_mapping_value(patient, "patient")
        input_path = PACKAGE_DIR / "golden/inference" / str(entry["input_filename"])
        logits_path = PACKAGE_DIR / "golden/inference" / str(entry["logits_filename"])
        batched_input = np.load(input_path, allow_pickle=False)
        expected_logits = load_logits(logits_path)
        actual_logits = session.run(["logits"], {"cine_mri": batched_input})[0]

        assert batched_input.shape == EXPECTED_BATCH_SHAPE
        assert batched_input.dtype == np.float32
        assert expected_logits.shape == EXPECTED_LOGITS_SHAPE
        assert np.allclose(actual_logits, expected_logits, atol=tolerance, rtol=0.0)
        assert int(np.argmax(actual_logits, axis=1)[0]) == entry["predicted_class_index"]
