"""Build the portable C++ deployment package for Medical AI Workstation."""

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPOSITORY_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cardiac_pathology.deployment import CppDeploymentPackageBuilder  # noqa: E402


def main() -> None:
    """Build the package and validate deterministic regeneration."""
    builder = CppDeploymentPackageBuilder(REPOSITORY_ROOT)
    summary = builder.build()
    deterministic = builder.validate_deterministic_regeneration()

    print("C++ deployment package build")
    print(f"\nPackage directory: {summary.package_dir}")
    print(f"ONNX SHA-256: {summary.model_sha256}")
    print(f"Golden preprocessing patients: {summary.preprocessing_patient_count}")
    print(f"Golden inference patients: {summary.inference_patient_count}")
    print(f"Inference input shape: {summary.input_shape}")
    print(f"Logits shape: {summary.logits_shape}")
    print(f"\nDeterministic deployment packaging: {'PASS' if deterministic else 'FAIL'}")
    print("\nDeployment package: PASS")

    if not deterministic:
        raise RuntimeError("Deployment package regeneration was not deterministic")


if __name__ == "__main__":
    main()
