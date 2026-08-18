"""PyTorch and ONNX Runtime parity validation for the Phase 9 classifier."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]
import torch
from torch import Tensor

from cardiac_pathology.models.resnet3d18 import ResNet3D18

EXPECTED_ARCHITECTURE = "ResNet3D18"
EXPECTED_INPUT_CHANNELS = 2
EXPECTED_NUM_CLASSES = 5
EXPECTED_SAMPLE_SHAPE = (EXPECTED_INPUT_CHANNELS, 14, 144, 144)
EXPECTED_BATCHED_SHAPE = (1, EXPECTED_INPUT_CHANNELS, 14, 144, 144)
EXPECTED_LOGITS_SHAPE = (1, EXPECTED_NUM_CLASSES)
ONNX_INPUT_NAME = "cine_mri"
ONNX_OUTPUT_NAME = "logits"


@dataclass(frozen=True)
class ParityValidationResult:
    """Measured PyTorch and ONNX Runtime classifier parity results."""

    patient_id: str
    pytorch_output_shape: tuple[int, ...]
    onnx_output_shape: tuple[int, ...]
    pytorch_output_dtype: str
    onnx_output_dtype: str
    pytorch_output_is_finite: bool
    onnx_output_is_finite: bool
    pytorch_predicted_class_index: int
    onnx_predicted_class_index: int
    predicted_class_index_matches: bool
    maximum_absolute_error: float
    mean_absolute_error: float
    maximum_relative_error: float


class ParityValidator:
    """Compare Phase 9 PyTorch and ONNX classifiers on one preprocessed tensor."""

    def validate(
        self,
        *,
        patient_id: str,
        preprocessed_tensor: np.ndarray,
        checkpoint_path: str | Path,
        onnx_model_path: str | Path,
    ) -> ParityValidationResult:
        """Run PyTorch and ONNX Runtime inference and return parity measurements.

        Args:
            patient_id: Identifier for the already-preprocessed patient tensor.
            preprocessed_tensor: Float32 tensor with shape ``[2, 14, 144, 144]``.
            checkpoint_path: Path to the Phase 9 PyTorch checkpoint.
            onnx_model_path: Path to the exported ONNX classifier.

        Returns:
            Immutable parity measurements for the supplied patient.

        Raises:
            FileNotFoundError: If the checkpoint or ONNX model is missing.
            TypeError: If the input tensor or checkpoint payload has an invalid type.
            ValueError: If input, checkpoint metadata, or backend outputs are invalid.
        """
        if not patient_id:
            raise ValueError("patient_id must be non-empty.")

        checkpoint = Path(checkpoint_path)
        onnx_model = Path(onnx_model_path)
        self._validate_paths(checkpoint_path=checkpoint, onnx_model_path=onnx_model)
        batched_input = self._validate_and_batch_input(preprocessed_tensor)

        model = self._load_model(checkpoint)
        pytorch_logits = self._run_pytorch_inference(model=model, batched_input=batched_input)
        onnx_logits = self._run_onnx_inference(
            onnx_model_path=onnx_model,
            batched_input=batched_input,
        )

        self._validate_logits(name="PyTorch", logits=pytorch_logits)
        self._validate_logits(name="ONNX", logits=onnx_logits)

        absolute_error = np.abs(pytorch_logits - onnx_logits)
        relative_denominator = np.maximum(
            np.maximum(np.abs(pytorch_logits), np.abs(onnx_logits)),
            np.finfo(np.float32).eps,
        )
        relative_error = absolute_error / relative_denominator
        pytorch_prediction = int(np.argmax(pytorch_logits, axis=1)[0])
        onnx_prediction = int(np.argmax(onnx_logits, axis=1)[0])

        return ParityValidationResult(
            patient_id=patient_id,
            pytorch_output_shape=tuple(int(dimension) for dimension in pytorch_logits.shape),
            onnx_output_shape=tuple(int(dimension) for dimension in onnx_logits.shape),
            pytorch_output_dtype=str(pytorch_logits.dtype),
            onnx_output_dtype=str(onnx_logits.dtype),
            pytorch_output_is_finite=bool(np.isfinite(pytorch_logits).all()),
            onnx_output_is_finite=bool(np.isfinite(onnx_logits).all()),
            pytorch_predicted_class_index=pytorch_prediction,
            onnx_predicted_class_index=onnx_prediction,
            predicted_class_index_matches=pytorch_prediction == onnx_prediction,
            maximum_absolute_error=float(np.max(absolute_error)),
            mean_absolute_error=float(np.mean(absolute_error)),
            maximum_relative_error=float(np.max(relative_error)),
        )

    def _validate_paths(self, *, checkpoint_path: Path, onnx_model_path: Path) -> None:
        """Validate model artifact paths before inference."""
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Phase 9 checkpoint is missing: {checkpoint_path}")
        if not onnx_model_path.is_file():
            raise FileNotFoundError(f"ONNX model is missing: {onnx_model_path}")

    def _validate_and_batch_input(self, preprocessed_tensor: np.ndarray) -> np.ndarray:
        """Validate the caller-supplied tensor and add a batch dimension."""
        if not isinstance(preprocessed_tensor, np.ndarray):
            raise TypeError("preprocessed_tensor must be a NumPy array.")
        if preprocessed_tensor.shape != EXPECTED_SAMPLE_SHAPE:
            raise ValueError(
                f"preprocessed_tensor must have shape {EXPECTED_SAMPLE_SHAPE}, "
                f"got {preprocessed_tensor.shape}."
            )
        if preprocessed_tensor.dtype != np.float32:
            raise TypeError(
                f"preprocessed_tensor must have dtype float32, got {preprocessed_tensor.dtype}."
            )
        if not np.isfinite(preprocessed_tensor).all():
            raise ValueError("preprocessed_tensor contains non-finite values.")

        batched_input = np.expand_dims(preprocessed_tensor, axis=0)
        if batched_input.shape != EXPECTED_BATCHED_SHAPE:
            raise ValueError(
                f"Batched input must have shape {EXPECTED_BATCHED_SHAPE}, "
                f"got {batched_input.shape}."
            )
        return batched_input

    def _load_model(self, checkpoint_path: Path) -> ResNet3D18:
        """Load and validate a Phase 9 checkpoint into ``ResNet3D18``."""
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
        """Require one checkpoint metadata value to match the parity contract."""
        actual = payload.get(key)
        if actual != expected:
            raise ValueError(
                f"Checkpoint metadata {key!r} must be {expected!r}, "
                f"got {actual!r}: {checkpoint_path}"
            )

    def _run_pytorch_inference(
        self,
        *,
        model: ResNet3D18,
        batched_input: np.ndarray,
    ) -> np.ndarray:
        """Run PyTorch inference and return CPU float32 logits."""
        input_tensor = torch.from_numpy(batched_input)
        with torch.inference_mode():
            logits = model(input_tensor)
        return cast(np.ndarray, logits.detach().cpu().numpy())

    def _run_onnx_inference(
        self,
        *,
        onnx_model_path: Path,
        batched_input: np.ndarray,
    ) -> np.ndarray:
        """Run ONNX Runtime inference and return the logits array."""
        session = ort.InferenceSession(
            str(onnx_model_path),
            providers=["CPUExecutionProvider"],
        )
        outputs = session.run([ONNX_OUTPUT_NAME], {ONNX_INPUT_NAME: batched_input})
        if len(outputs) != 1:
            raise ValueError(f"ONNX Runtime returned {len(outputs)} outputs, expected 1.")
        output = outputs[0]
        if not isinstance(output, np.ndarray):
            raise TypeError(f"ONNX Runtime output {ONNX_OUTPUT_NAME!r} is not a NumPy array.")
        return output

    def _validate_logits(self, *, name: str, logits: np.ndarray) -> None:
        """Validate backend output shape, dtype, and finite values."""
        if logits.shape != EXPECTED_LOGITS_SHAPE:
            raise ValueError(
                f"{name} logits must have shape {EXPECTED_LOGITS_SHAPE}, got {logits.shape}."
            )
        if logits.dtype != np.float32:
            raise TypeError(f"{name} logits must have dtype float32, got {logits.dtype}.")
        if not np.isfinite(logits).all():
            raise ValueError(f"{name} logits contain non-finite values.")
