"""Generate final Phase 7 visualization PNGs."""

import argparse
from pathlib import Path

from cardiac_pathology.training import Phase7VisualizationBuilder

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_ROOT = REPOSITORY_ROOT / "artifacts/checkpoints/phase7"


def main() -> None:
    """Render all required Phase 7 figures."""
    args = parse_args()
    builder = Phase7VisualizationBuilder.from_phase7_checkpoint_root(
        checkpoint_root=args.checkpoint_root,
        figures_dir=args.figures_dir,
    )
    output_paths = builder.build_all()
    for output_path in output_paths:
        print(f"Saved Phase 7 figure: {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse visualization command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--figures-dir", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
