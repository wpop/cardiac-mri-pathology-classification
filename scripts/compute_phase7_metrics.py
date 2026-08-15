"""Compute final frozen Phase 7 cross-validation metrics."""

import argparse
from pathlib import Path

from cardiac_pathology.training import Phase7MetricsBuilder

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_ROOT = REPOSITORY_ROOT / "artifacts/checkpoints/phase7"


def main() -> None:
    """Build the deterministic Phase 7 metrics artifact."""
    args = parse_args()
    builder = Phase7MetricsBuilder.from_phase7_checkpoint_root(
        checkpoint_root=args.checkpoint_root,
        output_path=args.output,
    )
    output_path = builder.build_and_save()
    print(f"Saved Phase 7 metrics: {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse final metric command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
