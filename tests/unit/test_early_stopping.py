"""Unit tests for Phase 6 early stopping."""

import pytest

from cardiac_pathology.training.early_stopping import EarlyStopping


def test_early_stopping_tracks_best_macro_f1_and_patience() -> None:
    """EarlyStopping maximizes validation Macro F1 and stops after patience."""
    stopper = EarlyStopping(patience=2, min_delta=0.01)

    assert stopper.step(0.40, epoch=1)
    assert stopper.best_metric == 0.40
    assert stopper.best_epoch == 1
    assert not stopper.step(0.405, epoch=2)
    assert not stopper.should_stop
    assert not stopper.step(0.409, epoch=3)
    assert stopper.should_stop


def test_early_stopping_resets_patience_after_improvement() -> None:
    """EarlyStopping clears the non-improvement count after a new best value."""
    stopper = EarlyStopping(patience=2)

    assert stopper.step(0.2, epoch=1)
    assert not stopper.step(0.1, epoch=2)
    assert stopper.step(0.3, epoch=3)

    assert stopper.non_improving_epochs == 0
    assert stopper.best_metric == 0.3
    assert stopper.best_epoch == 3
    assert not stopper.should_stop


def test_early_stopping_rejects_invalid_configuration() -> None:
    """EarlyStopping validates patience and minimum delta values."""
    with pytest.raises(ValueError, match="patience"):
        EarlyStopping(patience=0)

    with pytest.raises(ValueError, match="min_delta"):
        EarlyStopping(patience=1, min_delta=-0.1)
