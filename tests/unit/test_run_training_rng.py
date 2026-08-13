"""Unit tests for Phase 6 training RNG orchestration."""

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from cardiac_pathology.training import set_deterministic_seed
from scripts.run_training import reset_training_rng_after_initialization


def test_post_initialization_reset_restores_rng_after_different_internal_consumption() -> None:
    """Post-initialization reset equalizes RNG state after branch-specific consumption."""
    set_deterministic_seed(42)
    _ = torch.rand(())
    _ = random.random()
    _ = np.random.random()
    lightly_consumed_state = torch.get_rng_state().clone()
    reset_training_rng_after_initialization(42)
    lightly_consumed_reset_state = torch.get_rng_state().clone()

    set_deterministic_seed(42)
    _ = torch.rand(10)
    _ = [random.random() for _ in range(10)]
    _ = np.random.random(10)
    heavily_consumed_state = torch.get_rng_state().clone()
    reset_training_rng_after_initialization(42)
    heavily_consumed_reset_state = torch.get_rng_state().clone()

    assert not torch.equal(lightly_consumed_state, heavily_consumed_state)
    assert torch.equal(lightly_consumed_reset_state, heavily_consumed_reset_state)
    assert random.random() == 0.6394267984578837
    assert np.random.random() == 0.3745401188473625


def test_training_rng_reset_aligns_augmentation_like_torch_sequence() -> None:
    """Augmentation-style torch random draws match after the training RNG reset."""
    first_sequence = augmentation_like_sequence_after_reset(internal_torch_draws=1)
    second_sequence = augmentation_like_sequence_after_reset(internal_torch_draws=25)

    assert len(first_sequence) == len(second_sequence)
    for first, second in zip(first_sequence, second_sequence, strict=True):
        assert torch.equal(first, second)


def test_dataloader_shuffle_generator_is_independent_of_global_torch_rng() -> None:
    """The explicit DataLoader generator controls shuffle order despite global RNG drift."""
    order_without_global_drift = dataloader_order_after_global_torch_draws(draw_count=0)
    order_with_global_drift = dataloader_order_after_global_torch_draws(draw_count=100)

    assert order_without_global_drift == order_with_global_drift


def augmentation_like_sequence_after_reset(internal_torch_draws: int) -> tuple[torch.Tensor, ...]:
    """Return non-medical torch draws like those used by training augmentation."""
    set_deterministic_seed(42)
    _ = torch.rand(internal_torch_draws)
    reset_training_rng_after_initialization(42)
    return (
        torch.rand(()),
        torch.rand(()),
        torch.rand(()),
        torch.randn(4),
    )


def dataloader_order_after_global_torch_draws(draw_count: int) -> list[int]:
    """Return shuffled scalar order from an explicit generator after global RNG draws."""
    set_deterministic_seed(42)
    generator = torch.Generator()
    generator.manual_seed(42)
    _ = torch.rand(draw_count)
    dataset = TensorDataset(torch.arange(8))
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        generator=generator,
    )
    return [int(value) for batch in loader for value in batch[0]]
