"""Run the frozen Phase 1 preprocessing-candidate validation."""

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPOSITORY_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cardiac_pathology.analysis.final_preprocessing_candidate_analyzer import (  # noqa: E402
    FinalPreprocessingCandidateAnalyzer,
)


def main() -> None:
    """Run final preprocessing validation and print summaries."""
    analyzer = FinalPreprocessingCandidateAnalyzer(
        dataset_dir=REPOSITORY_ROOT / "data/raw/acdc/training",
        class_mapping_path=REPOSITORY_ROOT / "configs/class_mapping.json",
        output_dir=REPOSITORY_ROOT / "artifacts/dataset_inspection",
    )
    results = analyzer.run()
    analyzer.print_final_summary(results)


if __name__ == "__main__":
    main()
