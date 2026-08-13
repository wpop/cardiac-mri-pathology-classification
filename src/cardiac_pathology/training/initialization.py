"""Model initialization support for Phase 6 training experiments."""

import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import Literal, cast

import numpy as np
import torch
from torch import Tensor, nn

InitializationStrategy = Literal["random", "pretrained"]
CompatibilityAction = Literal["transferred", "adapted", "skipped", "missing"]


@dataclass(frozen=True, slots=True)
class ParameterCompatibility:
    """Compatibility decision for one target model state entry."""

    target_name: str
    action: CompatibilityAction
    source_name: str | None
    source_shape: tuple[int, ...] | None
    target_shape: tuple[int, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class CompatibilityReport:
    """Summary of a pretrained-to-target model initialization attempt."""

    initialization_strategy: InitializationStrategy
    entries: tuple[ParameterCompatibility, ...]

    @property
    def transferred_count(self) -> int:
        """Return the number of directly transferred parameter entries."""
        return self._count_action("transferred")

    @property
    def adapted_count(self) -> int:
        """Return the number of adapted parameter entries."""
        return self._count_action("adapted")

    @property
    def skipped_count(self) -> int:
        """Return the number of skipped parameter entries."""
        return self._count_action("skipped")

    @property
    def missing_count(self) -> int:
        """Return the number of target entries missing from the source model."""
        return self._count_action("missing")

    def as_metadata(self) -> dict[str, object]:
        """Convert the report to checkpoint-safe metadata.

        Returns:
            JSON-compatible metadata describing each transfer, adaptation, skip,
            and missing target entry.
        """
        return {
            "initialization_strategy": self.initialization_strategy,
            "transferred_count": self.transferred_count,
            "adapted_count": self.adapted_count,
            "skipped_count": self.skipped_count,
            "missing_count": self.missing_count,
            "parameters": [
                {
                    "target_name": entry.target_name,
                    "action": entry.action,
                    "source_name": entry.source_name,
                    "source_shape": entry.source_shape,
                    "target_shape": entry.target_shape,
                    "reason": entry.reason,
                }
                for entry in self.entries
            ],
        }

    def _count_action(self, action: CompatibilityAction) -> int:
        """Count entries with the requested compatibility action."""
        return sum(1 for entry in self.entries if entry.action == action)


def set_deterministic_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch execution for reproducible training.

    Args:
        seed: Non-negative integer seed used for Python, NumPy, CPU PyTorch, and
            CUDA PyTorch random number generators. The function also requests
            deterministic PyTorch algorithms when available.

    Returns:
        None.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def adapt_rgb_stem_to_two_channels(source_weight: Tensor) -> Tensor:
    """Adapt a three-channel Kinetics RGB stem weight to ED/ES input channels.

    Args:
        source_weight: Source convolution tensor with shape
            ``[out_channels, 3, D, H, W]`` from a video model trained on RGB
            inputs.

    Returns:
        Adapted target convolution tensor with shape
        ``[out_channels, 2, D, H, W]``. The transform is
        ``source_weight.mean(dim=1, keepdim=True).repeat(1, 2, 1, 1, 1) * (3/2)``.
    """
    if source_weight.ndim != 5:
        raise ValueError("source_weight must have shape [out_channels, 3, D, H, W]")
    if source_weight.shape[1] != 3:
        raise ValueError("source_weight must have exactly three source channels")

    rgb_mean = source_weight.mean(dim=1, keepdim=True)
    return rgb_mean.repeat(1, 2, 1, 1, 1) * (3.0 / 2.0)


def initialize_model(
    model: nn.Module,
    strategy: InitializationStrategy,
    source_state_dict: Mapping[str, Tensor] | None = None,
) -> CompatibilityReport:
    """Apply the requested model initialization strategy.

    Args:
        model: Target custom ``ResNet3D18`` instance or compatible PyTorch
            module. Random initialization uses the model's constructor-provided
            initialization and does not modify parameters.
        strategy: Initialization strategy. Supported values are exactly
            ``"random"`` and ``"pretrained"``.
        source_state_dict: Optional pretrained source state. When omitted for
            ``"pretrained"``, torchvision ``r3d_18`` Kinetics weights are loaded
            lazily, which may download weights through torchvision.

    Returns:
        Compatibility report describing initialization actions.
    """
    if strategy == "random":
        return CompatibilityReport(
            initialization_strategy="random",
            entries=tuple(
                ParameterCompatibility(
                    target_name=name,
                    action="skipped",
                    source_name=None,
                    source_shape=None,
                    target_shape=tuple(tensor.shape),
                    reason="Random model initialization is preserved.",
                )
                for name, tensor in model.state_dict().items()
            ),
        )
    if strategy != "pretrained":
        raise ValueError("initialization strategy must be 'random' or 'pretrained'")

    state_dict = (
        load_torchvision_r3d18_kinetics_state_dict()
        if source_state_dict is None
        else source_state_dict
    )
    return apply_pretrained_initialization(model=model, source_state_dict=state_dict)


def apply_pretrained_initialization(
    model: nn.Module,
    source_state_dict: Mapping[str, Tensor],
) -> CompatibilityReport:
    """Copy compatible pretrained tensors into a target model explicitly.

    Args:
        model: Target custom ``ResNet3D18`` model.
        source_state_dict: Source torchvision ``r3d_18`` state dictionary or a
            compatible mapping from parameter names to tensors.

    Returns:
        Compatibility report with transferred, adapted, skipped, and missing
        parameter decisions. The target state is reloaded with ``strict=True``
        after explicit tensor selection; this function does not use blind
        partial loading.
    """
    target_state = model.state_dict()
    updated_state = dict(target_state)
    entries: list[ParameterCompatibility] = []

    for target_name, target_tensor in target_state.items():
        source_name = source_name_for_target(target_name)
        target_shape = tuple(target_tensor.shape)

        if is_classifier_parameter(target_name):
            entries.append(
                ParameterCompatibility(
                    target_name=target_name,
                    action="skipped",
                    source_name=source_name,
                    source_shape=None,
                    target_shape=target_shape,
                    reason="Classifier head is task-specific and is never imported.",
                )
            )
            continue

        source_tensor = source_state_dict.get(source_name)
        if source_tensor is None:
            entries.append(
                ParameterCompatibility(
                    target_name=target_name,
                    action="missing",
                    source_name=source_name,
                    source_shape=None,
                    target_shape=target_shape,
                    reason="No mapped source tensor is available.",
                )
            )
            continue

        source_shape = tuple(source_tensor.shape)
        if target_name == "conv1.weight" and source_shape[1:2] == (3,):
            adapted_tensor = adapt_rgb_stem_to_two_channels(source_tensor)
            if tuple(adapted_tensor.shape) != target_shape:
                entries.append(
                    ParameterCompatibility(
                        target_name=target_name,
                        action="skipped",
                        source_name=source_name,
                        source_shape=source_shape,
                        target_shape=target_shape,
                        reason="Adapted RGB stem shape does not match target stem shape.",
                    )
                )
                continue
            updated_state[target_name] = adapted_tensor.to(
                dtype=target_tensor.dtype,
                device=target_tensor.device,
            )
            entries.append(
                ParameterCompatibility(
                    target_name=target_name,
                    action="adapted",
                    source_name=source_name,
                    source_shape=source_shape,
                    target_shape=target_shape,
                    reason="RGB stem adapted to ED/ES channels with the documented 3-to-2 formula.",
                )
            )
            continue

        if source_shape == target_shape:
            updated_state[target_name] = source_tensor.to(
                dtype=target_tensor.dtype,
                device=target_tensor.device,
            )
            entries.append(
                ParameterCompatibility(
                    target_name=target_name,
                    action="transferred",
                    source_name=source_name,
                    source_shape=source_shape,
                    target_shape=target_shape,
                    reason="Mapped source tensor shape matches the target tensor shape.",
                )
            )
            continue

        entries.append(
            ParameterCompatibility(
                target_name=target_name,
                action="skipped",
                source_name=source_name,
                source_shape=source_shape,
                target_shape=target_shape,
                reason="Source tensor shape is incompatible with the target tensor shape.",
            )
        )

    model.load_state_dict(updated_state, strict=True)
    return CompatibilityReport(initialization_strategy="pretrained", entries=tuple(entries))


def source_name_for_target(target_name: str) -> str:
    """Map a custom ``ResNet3D18`` state name to a torchvision ``r3d_18`` name.

    Args:
        target_name: Target model state-dictionary name.

    Returns:
        Source torchvision state-dictionary name expected to provide the same
        kind of tensor where the architectures are compatible.
    """
    if target_name == "conv1.weight":
        return "stem.0.weight"
    if target_name.startswith("bn1."):
        return target_name.replace("bn1.", "stem.1.", 1)

    mapped = target_name
    for conv_name in ("conv1", "conv2"):
        mapped = mapped.replace(f".{conv_name}.weight", f".{conv_name}.0.weight")
    mapped = mapped.replace(".bn1.", ".conv1.1.")
    mapped = mapped.replace(".bn2.", ".conv2.1.")
    return mapped


def is_classifier_parameter(target_name: str) -> bool:
    """Return whether a target state entry belongs to the classifier head.

    Args:
        target_name: Target model state-dictionary name.

    Returns:
        True when the state entry belongs to ``fc``; otherwise False.
    """
    return target_name.startswith("fc.")


def load_torchvision_r3d18_kinetics_state_dict() -> Mapping[str, Tensor]:
    """Load torchvision ``r3d_18`` Kinetics weights lazily.

    Returns:
        Mapping from torchvision state-dictionary names to tensors.

    Raises:
        ImportError: If torchvision is not installed. Random initialization does
            not import torchvision or require network access.
    """
    try:
        video_module = import_module("torchvision.models.video")
    except ModuleNotFoundError as error:
        raise ImportError(
            "Pretrained initialization requires torchvision. Install torchvision "
            "or use initialization_strategy='random'."
        ) from error

    weights_enum_attribute = "R3D_18_Weights"
    weights_attribute = "KINETICS400_V1"
    model_attribute = "r3d_18"
    weights_enum = getattr(video_module, weights_enum_attribute)
    weights = getattr(weights_enum, weights_attribute)
    r3d_18 = cast(Callable[..., nn.Module], getattr(video_module, model_attribute))
    source_model = r3d_18(weights=weights)
    return {
        name: tensor
        for name, tensor in source_model.state_dict().items()
        if isinstance(tensor, Tensor)
    }
