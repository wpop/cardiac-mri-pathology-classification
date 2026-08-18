"""Export the Phase 9 classifier checkpoint to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

from cardiac_pathology.deployment.onnx_exporter import OnnxExporter

DEFAULT_CHECKPOINT_PATH = Path("artifacts/checkpoints/phase9/classifier.pt")
DEFAULT_OUTPUT_PATH = Path("artifacts/deployment/classifier.onnx")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for ONNX export."""
    parser = argparse.ArgumentParser(
        description="Export the Phase 9 cardiac MRI classifier checkpoint to ONNX.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help=f"Path to the Phase 9 checkpoint. Defaults to {DEFAULT_CHECKPOINT_PATH}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Destination ONNX path. Defaults to {DEFAULT_OUTPUT_PATH}.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output ONNX file if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the ONNX export command."""
    args = parse_args()
    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    exported_path = OnnxExporter().export(
        checkpoint_path=args.checkpoint,
        output_path=output_path,
        overwrite=args.overwrite,
    )
    print(f"Exported ONNX model to {exported_path}")


if __name__ == "__main__":
    main()
