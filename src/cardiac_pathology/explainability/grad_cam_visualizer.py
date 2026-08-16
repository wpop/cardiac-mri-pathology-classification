"""Reusable Phase 8 3D Grad-CAM visualization infrastructure."""

import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cardiac_pathology_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from torch import Tensor


class GradCamVisualizer:
    """Render deterministic joint ED+ES Grad-CAM overlay figures.

    The visualizer receives already-preprocessed tensors and already-computed
    Grad-CAM volumes. It does not load patient data, checkpoints, model outputs,
    or any project artifacts.
    """

    cam_colormap = "magma"
    overlay_alpha = 0.45
    figure_dpi = 160

    def select_slice(self, cam: Tensor, valid_start: int, valid_end: int) -> int:
        """Select the real depth slice with maximum mean CAM over height and width.

        Args:
            cam: Grad-CAM tensor with shape ``[D, H, W]``.
            valid_start: Inclusive first real, non-padded depth index.
            valid_end: Exclusive final real, non-padded depth index.

        Returns:
            Absolute integer depth index selected by
            ``valid_start + argmax(mean(cam[valid_start:valid_end], dim=(H, W)))``.
            PyTorch's deterministic argmax behavior resolves ties to the first
            smallest valid index.
        """
        self._validate_cam(cam)
        self._validate_valid_depth_range(valid_start, valid_end, depth=cam.shape[0])
        slice_scores = cam[valid_start:valid_end].mean(dim=(1, 2))
        relative_index = int(torch.argmax(slice_scores).detach().cpu())
        return valid_start + relative_index

    def save(
        self,
        *,
        patient_tensor: Tensor,
        cam: Tensor,
        patient_id: str,
        fold_index: int,
        true_class_name: str,
        predicted_class_name: str,
        confidence: float,
        valid_start: int,
        valid_end: int,
        output_path: Path | str,
    ) -> Path:
        """Save a four-panel joint ED+ES Grad-CAM visualization.

        Args:
            patient_tensor: Preprocessed patient tensor with shape ``[2, D, H, W]``.
                Channel ``0`` is ED and channel ``1`` is ES.
            cam: Joint ED+ES Grad-CAM tensor with shape ``[D, H, W]``.
            patient_id: Patient identifier to show in the figure title.
            fold_index: Cross-validation fold index to show in the figure title.
            true_class_name: Ground-truth class name to show in the figure title.
            predicted_class_name: Predicted class name to show in the figure title.
            confidence: Prediction confidence in ``[0, 1]``.
            valid_start: Inclusive first real, non-padded depth index.
            valid_end: Exclusive final real, non-padded depth index.
            output_path: File path where the PNG figure should be written.

        Returns:
            The deterministic output path used for saving.
        """
        self._validate_patient_tensor(patient_tensor)
        self._validate_cam(cam)
        self._validate_metadata(
            patient_id=patient_id,
            fold_index=fold_index,
            true_class_name=true_class_name,
            predicted_class_name=predicted_class_name,
            confidence=confidence,
        )
        if tuple(patient_tensor.shape[1:]) != tuple(cam.shape):
            raise ValueError(
                "cam shape must match patient_tensor spatial shape; "
                f"got cam {tuple(cam.shape)} and patient spatial {tuple(patient_tensor.shape[1:])}."
            )

        selected_slice = self.select_slice(cam, valid_start, valid_end)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        figure, axes = plt.subplots(1, 4, figsize=(10.5, 3.2), constrained_layout=True)
        try:
            ed_slice = patient_tensor[0, selected_slice].detach().cpu()
            es_slice = patient_tensor[1, selected_slice].detach().cpu()
            cam_slice = cam[selected_slice].detach().cpu()

            ed_limits = self._robust_grayscale_limits(ed_slice)
            es_limits = self._robust_grayscale_limits(es_slice)

            panels = (
                ("ED", ed_slice, None, ed_limits),
                ("ES", es_slice, None, es_limits),
                ("Joint Grad-CAM on ED", ed_slice, cam_slice, ed_limits),
                ("Joint Grad-CAM on ES", es_slice, cam_slice, es_limits),
            )
            heatmap = None
            for axis, (title, image_slice, cam_overlay, limits) in zip(axes, panels, strict=True):
                axis.imshow(image_slice.numpy(), cmap="gray", vmin=limits[0], vmax=limits[1])
                if cam_overlay is not None:
                    heatmap = axis.imshow(
                        cam_overlay.numpy(),
                        cmap=self.cam_colormap,
                        vmin=0.0,
                        vmax=1.0,
                        alpha=self.overlay_alpha,
                    )
                axis.set_title(title, fontsize=9)
                axis.set_axis_off()

            if heatmap is not None:
                figure.colorbar(heatmap, ax=axes[2:], fraction=0.046, pad=0.04)

            figure.suptitle(
                "Joint ED+ES model attribution | "
                f"Patient {patient_id} | Fold {fold_index} | True {true_class_name} | "
                f"Predicted {predicted_class_name} | Confidence {confidence:.3f} | "
                f"Slice {selected_slice}",
                fontsize=10,
            )
            figure.savefig(output, dpi=self.figure_dpi)
        finally:
            plt.close(figure)

        return output

    @staticmethod
    def _validate_patient_tensor(patient_tensor: Tensor) -> None:
        if not isinstance(patient_tensor, Tensor):
            raise TypeError("patient_tensor must be a torch.Tensor.")
        if patient_tensor.ndim != 4 or patient_tensor.shape[0] != 2:
            raise ValueError(
                f"patient_tensor must have shape [2, D, H, W]; got {tuple(patient_tensor.shape)}."
            )
        if any(size <= 0 for size in patient_tensor.shape[1:]):
            raise ValueError("patient_tensor spatial dimensions must be positive.")
        GradCamVisualizer._validate_finite("patient_tensor", patient_tensor)

    @staticmethod
    def _validate_cam(cam: Tensor) -> None:
        if not isinstance(cam, Tensor):
            raise TypeError("cam must be a torch.Tensor.")
        if cam.ndim != 3:
            raise ValueError(f"cam must have shape [D, H, W]; got {tuple(cam.shape)}.")
        if any(size <= 0 for size in cam.shape):
            raise ValueError("cam spatial dimensions must be positive.")
        GradCamVisualizer._validate_finite("cam", cam)

    @staticmethod
    def _validate_valid_depth_range(valid_start: int, valid_end: int, *, depth: int) -> None:
        if not isinstance(valid_start, int) or isinstance(valid_start, bool):
            raise TypeError("valid_start must be an integer.")
        if not isinstance(valid_end, int) or isinstance(valid_end, bool):
            raise TypeError("valid_end must be an integer.")
        if not 0 <= valid_start < valid_end <= depth:
            raise ValueError(
                "valid_start and valid_end must satisfy "
                f"0 <= valid_start < valid_end <= D; got valid_start={valid_start}, "
                f"valid_end={valid_end}, D={depth}."
            )

    @staticmethod
    def _validate_metadata(
        *,
        patient_id: str,
        fold_index: int,
        true_class_name: str,
        predicted_class_name: str,
        confidence: float,
    ) -> None:
        if not isinstance(patient_id, str) or not patient_id:
            raise ValueError("patient_id must be a non-empty string.")
        if not isinstance(fold_index, int) or isinstance(fold_index, bool):
            raise TypeError("fold_index must be an integer.")
        if not isinstance(true_class_name, str) or not true_class_name:
            raise ValueError("true_class_name must be a non-empty string.")
        if not isinstance(predicted_class_name, str) or not predicted_class_name:
            raise ValueError("predicted_class_name must be a non-empty string.")
        if not isinstance(confidence, int | float):
            raise TypeError("confidence must be numeric.")
        if not math.isfinite(float(confidence)):
            raise ValueError("confidence must be finite.")
        if not 0.0 <= float(confidence) <= 1.0:
            raise ValueError("confidence must be within [0, 1].")

    @staticmethod
    def _robust_grayscale_limits(image_slice: Tensor) -> tuple[float, float]:
        image_slice = image_slice.to(dtype=torch.float32)
        lower = float(torch.quantile(image_slice, 0.01).detach().cpu())
        upper = float(torch.quantile(image_slice, 0.99).detach().cpu())
        if lower < upper:
            return lower, upper

        center = float(image_slice.flatten()[0].detach().cpu())
        return center - 0.5, center + 0.5

    @staticmethod
    def _validate_finite(name: str, tensor: Tensor) -> None:
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must contain only finite values.")
