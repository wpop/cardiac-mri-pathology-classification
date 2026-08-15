"""Pool frozen Phase 7 out-of-fold prediction files."""

import argparse
from pathlib import Path

from cardiac_pathology.training import Phase7OofPredictionPooler

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_ROOT = REPOSITORY_ROOT / "artifacts/checkpoints/phase7"
DEFAULT_OUTER_SPLIT_PATH = REPOSITORY_ROOT / "artifacts/dataset_splits/acdc_5fold_seed42.json"


def main() -> None:
    """Build the deterministic pooled Phase 7 OOF prediction artifact."""
    args = parse_args()
    pooler = Phase7OofPredictionPooler.from_phase7_checkpoint_root(
        checkpoint_root=args.checkpoint_root,
        outer_split_path=args.outer_split_artifact,
        output_path=args.output,
    )
    output_path = pooler.build_and_save()
    print(f"Saved pooled Phase 7 OOF predictions: {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse OOF pooling command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--outer-split-artifact", type=Path, default=DEFAULT_OUTER_SPLIT_PATH)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
