"""Run the integrated Phase 1 spatial-policy analysis."""

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPOSITORY_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cardiac_pathology.analysis.spatial_policy_analyzer import (  # noqa: E402
    SpatialPolicyAnalyzer,
)


def main() -> None:
    """Run spatial candidate analysis and print final summaries."""
    analyzer = SpatialPolicyAnalyzer(
        dataset_dir=REPOSITORY_ROOT / "data/raw/acdc/training",
        geometry_csv=REPOSITORY_ROOT
        / "artifacts/dataset_inspection/patient_geometry.csv",
        class_mapping_path=REPOSITORY_ROOT / "configs/class_mapping.json",
        output_dir=REPOSITORY_ROOT / "artifacts/dataset_inspection",
    )
    results = analyzer.run()
    analyzer.print_final_summaries(results)


if __name__ == "__main__":
    main()
