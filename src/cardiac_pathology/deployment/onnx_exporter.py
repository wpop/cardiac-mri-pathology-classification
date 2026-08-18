"""ONNX export utilities for the Phase 9 cardiac MRI classifier."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn

from cardiac_pathology.models.resnet3d18 import ResNet3D18

EXPECTED_ARCHITECTURE = "ResNet3D18"
EXPECTED_INPUT_CHANNELS = 2
EXPECTED_NUM_CLASSES = 5
SOFTWARE_FORWARD_INPUT_SHAPE = (1, EXPECTED_INPUT_CHANNELS, 14, 144, 144)
ONNX_EXPORT_INPUT_SHAPE = (2, EXPECTED_INPUT_CHANNELS, 14, 144, 144)
EXPECTED_OUTPUT_SHAPE = (1, EXPECTED_NUM_CLASSES)
ONNX_OPSET_VERSION = 18
ONNX_INPUT_NAME = "cine_mri"
ONNX_OUTPUT_NAME = "logits"


class OnnxExporter:
    """Export the trained Phase 9 ``ResNet3D18`` classifier to ONNX.

    The exported graph contains only the classifier. It expects preprocessed
    float32 tensors with shape ``[N, 2, 14, 144, 144]`` and emits raw logits
    with shape ``[N, 5]``. Only the batch dimension is dynamic.
    """

    def export(
        self,
        checkpoint_path: str | Path,
        output_path: str | Path,
        *,
        overwrite: bool = False,
    ) -> Path:
        """Export a validated Phase 9 checkpoint as a single ONNX file.

        Args:
            checkpoint_path: Path to the Phase 9 PyTorch checkpoint.
            output_path: Destination path for the ONNX file.
            overwrite: If ``True``, allow replacing an existing ONNX file.

        Returns:
            The ONNX output path supplied by the caller.

        Raises:
            FileNotFoundError: If the checkpoint or output directory is missing.
            FileExistsError: If the output path exists and overwrite is disabled.
            IsADirectoryError: If the output path points to a directory.
            TypeError: If the checkpoint payload has an invalid type.
            ValueError: If required checkpoint metadata or software inference is invalid.
        """
        checkpoint = Path(checkpoint_path)
        output = Path(output_path)
        self._validate_paths(checkpoint_path=checkpoint, output_path=output, overwrite=overwrite)

        model = self._load_model(checkpoint)
        self._verify_software_forward(model)
        self._export_model(model=model, output_path=output)

        return output

    def _validate_paths(
        self,
        *,
        checkpoint_path: Path,
        output_path: Path,
        overwrite: bool,
    ) -> None:
        """Validate filesystem preconditions before loading or exporting."""
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Phase 9 checkpoint is missing: {checkpoint_path}")
        if not output_path.parent.is_dir():
            raise FileNotFoundError(f"ONNX output directory is missing: {output_path.parent}")
        if output_path.is_dir():
            raise IsADirectoryError(f"ONNX output path is a directory: {output_path}")
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"ONNX output already exists: {output_path}. Pass overwrite=True to replace it."
            )

    def _load_model(self, checkpoint_path: Path) -> ResNet3D18:
        """Load and validate the Phase 9 checkpoint into ``ResNet3D18``."""
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise TypeError(f"Malformed checkpoint payload: {checkpoint_path}")

        self._validate_checkpoint_metadata(payload=payload, checkpoint_path=checkpoint_path)

        model_state_dict = payload["model_state_dict"]
        if not isinstance(model_state_dict, Mapping):
            raise TypeError(f"Checkpoint model_state_dict is invalid: {checkpoint_path}")

        model = ResNet3D18(
            input_channels=EXPECTED_INPUT_CHANNELS,
            num_classes=EXPECTED_NUM_CLASSES,
        )
        model.load_state_dict(cast(Mapping[str, Tensor], model_state_dict), strict=True)
        model.eval()
        return model

    def _validate_checkpoint_metadata(
        self,
        *,
        payload: Mapping[Any, Any],
        checkpoint_path: Path,
    ) -> None:
        """Validate required Phase 9 checkpoint metadata."""
        self._require_metadata_value(
            payload=payload,
            checkpoint_path=checkpoint_path,
            key="architecture",
            expected=EXPECTED_ARCHITECTURE,
        )
        self._require_metadata_value(
            payload=payload,
            checkpoint_path=checkpoint_path,
            key="input_channels",
            expected=EXPECTED_INPUT_CHANNELS,
        )
        self._require_metadata_value(
            payload=payload,
            checkpoint_path=checkpoint_path,
            key="num_classes",
            expected=EXPECTED_NUM_CLASSES,
        )
        if "model_state_dict" not in payload:
            raise ValueError(f"Checkpoint lacks model_state_dict: {checkpoint_path}")

    def _require_metadata_value(
        self,
        *,
        payload: Mapping[Any, Any],
        checkpoint_path: Path,
        key: str,
        expected: object,
    ) -> None:
        """Require one checkpoint metadata value to match the export contract."""
        actual = payload.get(key)
        if actual != expected:
            raise ValueError(
                f"Checkpoint metadata {key!r} must be {expected!r}, "
                f"got {actual!r}: {checkpoint_path}"
            )

    def _verify_software_forward(self, model: nn.Module) -> None:
        """Verify PyTorch inference before exporting the graph."""
        example_input = torch.randn(SOFTWARE_FORWARD_INPUT_SHAPE, dtype=torch.float32)
        with torch.no_grad():
            logits = model(example_input)

        if tuple(logits.shape) != EXPECTED_OUTPUT_SHAPE:
            raise ValueError(
                f"Software forward pass produced shape {tuple(logits.shape)}, "
                f"expected {EXPECTED_OUTPUT_SHAPE}."
            )
        if not torch.isfinite(logits).all().item():
            raise ValueError("Software forward pass produced non-finite logits.")

    def _export_model(
        self,
        *,
        model: nn.Module,
        output_path: Path,
    ) -> None:
        """Export the validated classifier with a dynamic batch dimension."""
        batch_dim = torch.export.Dim("batch", min=1)
        example_input = torch.randn(ONNX_EXPORT_INPUT_SHAPE, dtype=torch.float32)
        torch.onnx.export(
            model,
            args=(example_input,),
            f=output_path,
            input_names=[ONNX_INPUT_NAME],
            output_names=[ONNX_OUTPUT_NAME],
            opset_version=ONNX_OPSET_VERSION,
            dynamo=True,
            dynamic_shapes=({0: batch_dim},),
            external_data=False,
        )
