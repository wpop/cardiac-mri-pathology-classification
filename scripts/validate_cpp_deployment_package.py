"""Validate the portable C++ deployment package."""

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPOSITORY_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cardiac_pathology.deployment import CppDeploymentPackageBuilder  # noqa: E402


def main() -> None:
    """Validate package files, checksums, and ONNX Runtime logits parity."""
    summary = CppDeploymentPackageBuilder(REPOSITORY_ROOT).validate()

    print("C++ deployment package validation")
    print("")
    print("Model SHA-256: PASS")
    print("ONNX contract: PASS")
    print("Class mapping: PASS")
    print("Preprocessing contract: PASS")
    print("Golden preprocessing references: PASS")
    print("Golden inference inputs: PASS")
    print("Golden inference logits: PASS")
    print("ONNX Runtime inference parity: PASS")
    print("")
    print(f"Package directory: {summary.package_dir}")
    print(f"ONNX SHA-256: {summary.model_sha256}")
    print(f"Golden preprocessing patients: {summary.preprocessing_patient_count}")
    print(f"Golden inference patients: {summary.inference_patient_count}")
    print(f"Inference input shape: {summary.input_shape}")
    print(f"Logits shape: {summary.logits_shape}")
    print("")
    print("Deployment package: PASS")


if __name__ == "__main__":
    main()
