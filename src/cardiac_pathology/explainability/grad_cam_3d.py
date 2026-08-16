"""Reusable native PyTorch 3D Grad-CAM implementation."""

from collections.abc import Callable, Sequence
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle


class GradCam3D:
    """Generate 3D Grad-CAM volumes for a selected model layer.

    The wrapped model is expected to consume 5D tensors with shape
    ``[N, C, D, H, W]`` and produce raw class logits with shape
    ``[N, num_classes]``. The target layer must produce a 5D activation tensor
    with shape ``[N, channels, D_layer, H_layer, W_layer]``.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        """Initialize the Grad-CAM generator.

        Args:
            model: Classification model that returns raw logits.
            target_layer: Layer whose 5D activations should be explained.

        Raises:
            TypeError: If ``model`` or ``target_layer`` is not an ``nn.Module``.
        """
        if not isinstance(model, nn.Module):
            raise TypeError("model must be an instance of torch.nn.Module.")
        if not isinstance(target_layer, nn.Module):
            raise TypeError("target_layer must be an instance of torch.nn.Module.")

        self.model = model
        self.target_layer = target_layer

    def generate(
        self,
        input_tensor: Tensor,
        target_class: int | Sequence[int] | Tensor,
    ) -> Tensor:
        """Generate normalized, input-resolution 3D Grad-CAM maps.

        Args:
            input_tensor: Input batch with shape ``[N, C, D, H, W]``.
            target_class: Either a single class index used for every batch item,
                or one class index per batch item.

        Returns:
            CAM tensor with shape ``[N, D, H, W]`` upsampled to the input spatial
            dimensions and normalized independently per batch item to ``[0, 1]``.

        Raises:
            TypeError: If tensor arguments have unsupported types or dtypes.
            ValueError: If input, output, activation, gradient, or class-index
                shapes are invalid, or if required tensors contain non-finite
                values.
        """
        self._validate_input_tensor(input_tensor)
        batch_size = input_tensor.shape[0]
        target_indices = self._prepare_target_indices(target_class, batch_size, input_tensor.device)

        activation: Tensor | None = None
        gradient: Tensor | None = None
        gradient_handle: RemovableHandle | None = None

        def save_activation(
            _module: nn.Module, _inputs: tuple[object, ...], output: object
        ) -> None:
            nonlocal activation, gradient, gradient_handle

            if not isinstance(output, Tensor):
                raise TypeError("target_layer output must be a torch.Tensor.")

            self._validate_activation(output)
            if not output.requires_grad:
                raise ValueError(
                    "target_layer activation does not require gradients; "
                    "Grad-CAM must run with autograd enabled."
                )

            activation = output

            def save_gradient(grad: Tensor) -> None:
                nonlocal gradient
                gradient = grad

            gradient_handle = self._register_tensor_gradient_hook(output, save_gradient)

        previous_training_state = self.model.training
        forward_handle = self.target_layer.register_forward_hook(save_activation)

        try:
            self.model.eval()
            self.model.zero_grad(set_to_none=True)

            logits = self.model(input_tensor)
            self._validate_logits(logits, batch_size)
            self._validate_target_indices(target_indices, logits.shape[1])

            selected_logits = logits.gather(1, target_indices.view(batch_size, 1)).sum()
            selected_logits.backward()

            if activation is None:
                raise ValueError(
                    "target_layer did not produce an activation during the forward pass."
                )
            if gradient is None:
                raise ValueError(
                    "target_layer activation gradient was not captured during backward."
                )

            self._validate_activation(activation)
            if gradient.shape != activation.shape:
                gradient_shape = tuple(gradient.shape)
                activation_shape = tuple(activation.shape)
                raise ValueError(
                    "captured gradient shape must match activation shape; "
                    f"got gradient {gradient_shape} and activation {activation_shape}."
                )
            self._validate_finite("activation", activation)
            self._validate_finite("gradient", gradient)

            output_size = (
                int(input_tensor.shape[-3]),
                int(input_tensor.shape[-2]),
                int(input_tensor.shape[-1]),
            )
            cam = self._compute_cam(activation, gradient, output_size)
            self._validate_finite("Grad-CAM output", cam)

            tolerance = 1e-6
            cam_min = float(cam.min().detach().cpu())
            cam_max = float(cam.max().detach().cpu())
            if cam_min < -tolerance or cam_max > 1.0 + tolerance:
                raise ValueError(
                    "normalized Grad-CAM output must be constrained to [0, 1]; "
                    f"got min={cam_min:.8f}, max={cam_max:.8f}."
                )

            return cam.clamp(0.0, 1.0)
        finally:
            forward_handle.remove()
            if gradient_handle is not None:
                gradient_handle.remove()
            self.model.zero_grad(set_to_none=True)
            self.model.train(previous_training_state)

    @staticmethod
    def _validate_input_tensor(input_tensor: Tensor) -> None:
        if not isinstance(input_tensor, Tensor):
            raise TypeError("input_tensor must be a torch.Tensor.")
        if input_tensor.ndim != 5:
            raise ValueError(
                f"input_tensor must have shape [N, C, D, H, W]; got {tuple(input_tensor.shape)}."
            )
        if input_tensor.shape[0] <= 0:
            raise ValueError("input_tensor batch size N must be greater than zero.")
        if not torch.is_floating_point(input_tensor):
            raise TypeError("input_tensor must be a floating-point tensor.")
        GradCam3D._validate_finite("input_tensor", input_tensor)

    @staticmethod
    def _prepare_target_indices(
        target_class: int | Sequence[int] | Tensor,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        if isinstance(target_class, int):
            target_indices = torch.full(
                (batch_size,),
                target_class,
                dtype=torch.long,
                device=device,
            )
        elif isinstance(target_class, Tensor):
            if not GradCam3D._is_integer_dtype(target_class):
                raise TypeError("target_class tensor must contain integer class indices.")
            if target_class.ndim == 0:
                target_indices = target_class.to(device=device, dtype=torch.long).expand(batch_size)
            elif target_class.ndim == 1:
                target_indices = target_class.to(device=device, dtype=torch.long)
            else:
                raise ValueError(
                    "target_class tensor must be scalar or 1D with one class per batch item; "
                    f"got shape {tuple(target_class.shape)}."
                )
        elif isinstance(target_class, Sequence):
            target_indices = torch.as_tensor(target_class, device=device)
            if target_indices.numel() > 0 and not GradCam3D._is_integer_dtype(target_indices):
                raise TypeError("target_class sequence must contain integer class indices.")
            target_indices = target_indices.to(dtype=torch.long)
        else:
            raise TypeError("target_class must be an int, a sequence of ints, or a torch.Tensor.")

        if target_indices.ndim != 1:
            raise ValueError(
                "target_class must resolve to a 1D tensor of class indices; "
                f"got shape {tuple(target_indices.shape)}."
            )
        if target_indices.numel() != batch_size:
            raise ValueError(
                "target_class must provide exactly one class index per batch item; "
                f"got {target_indices.numel()} indices for batch size {batch_size}."
            )
        return target_indices

    @staticmethod
    def _validate_logits(logits: Tensor, batch_size: int) -> None:
        if not isinstance(logits, Tensor):
            raise TypeError("model output must be a torch.Tensor.")
        if logits.ndim != 2:
            raise ValueError(
                f"model output must have shape [N, num_classes]; got {tuple(logits.shape)}."
            )
        if logits.shape[0] != batch_size:
            raise ValueError(
                "model output batch size must match input batch size; "
                f"got output N={logits.shape[0]} and input N={batch_size}."
            )
        if logits.shape[1] <= 0:
            raise ValueError("model output must contain at least one class logit.")
        GradCam3D._validate_finite("model output", logits)

    @staticmethod
    def _validate_target_indices(target_indices: Tensor, num_classes: int) -> None:
        if target_indices.numel() == 0:
            raise ValueError("target_class must contain at least one class index.")
        if not torch.isfinite(target_indices).all():
            raise ValueError("target_class indices must be finite.")

        min_index = int(target_indices.min().detach().cpu())
        max_index = int(target_indices.max().detach().cpu())
        if min_index < 0 or max_index >= num_classes:
            raise ValueError(
                "target_class indices must be within the model output range "
                f"[0, {num_classes - 1}]; got min={min_index}, max={max_index}."
            )

    @staticmethod
    def _validate_activation(activation: Tensor) -> None:
        if activation.ndim != 5:
            raise ValueError(
                "target_layer activation must have shape [N, C, D, H, W]; "
                f"got {tuple(activation.shape)}."
            )
        if activation.shape[0] <= 0 or activation.shape[1] <= 0:
            raise ValueError(
                "target_layer activation must have positive batch and channel dimensions; "
                f"got {tuple(activation.shape)}."
            )
        if any(size <= 0 for size in activation.shape[-3:]):
            raise ValueError(
                "target_layer activation must have positive spatial dimensions; "
                f"got {tuple(activation.shape)}."
            )

    @staticmethod
    def _compute_cam(
        activation: Tensor, gradient: Tensor, output_size: tuple[int, int, int]
    ) -> Tensor:
        activation = activation.detach()
        gradient = gradient.detach()
        weights = gradient.mean(dim=(2, 3, 4), keepdim=True)
        cam = (weights * activation).sum(dim=1)
        cam = F.relu(cam)
        cam = GradCam3D._normalize_per_item(cam)
        cam = F.interpolate(
            cam.unsqueeze(1),
            size=output_size,
            mode="trilinear",
            align_corners=False,
        ).squeeze(1)
        return cast(Tensor, cam)

    @staticmethod
    def _normalize_per_item(cam: Tensor) -> Tensor:
        flattened = cam.flatten(start_dim=1)
        cam_min = flattened.min(dim=1).values.view(-1, 1, 1, 1)
        cam_max = flattened.max(dim=1).values.view(-1, 1, 1, 1)
        cam_range = cam_max - cam_min
        eps = torch.finfo(cam.dtype).eps
        safe_range = torch.where(cam_range > eps, cam_range, torch.ones_like(cam_range))
        normalized = (cam - cam_min) / safe_range
        return torch.where(cam_range > eps, normalized, torch.zeros_like(normalized))

    @staticmethod
    def _is_integer_dtype(tensor: Tensor) -> bool:
        return tensor.dtype in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }

    @staticmethod
    def _register_tensor_gradient_hook(
        tensor: Tensor,
        hook: Callable[[Tensor], None],
    ) -> RemovableHandle:
        register_hook = cast(
            Callable[[Callable[[Tensor], None]], RemovableHandle],
            tensor.register_hook,
        )
        return register_hook(hook)

    @staticmethod
    def _validate_finite(name: str, tensor: Tensor) -> None:
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must contain only finite values.")
