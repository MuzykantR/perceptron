"""Tests of the plotting helpers; files go to a temporary folder."""

from pathlib import Path

import numpy as np
import pytest

from perceptron_lab.data import make_blobs
from perceptron_lab.model import Perceptron
from perceptron_lab.plots import boundary_line, plot_training


def test_boundary_points_lie_on_the_line() -> None:
    weights, bias = np.array([2.0, -1.0]), 3.0
    points = boundary_line(weights, bias, -5.0, 5.0)
    assert points is not None
    x1, x2 = points
    assert np.allclose(weights[0] * x1 + weights[1] * x2 + bias, 0.0)


def test_boundary_vertical_and_missing() -> None:
    vertical = boundary_line(np.array([2.0, 0.0]), -4.0, 0.0, 5.0)
    assert vertical is not None
    assert np.allclose(vertical[0], 2.0)
    assert boundary_line(np.array([0.0, 0.0]), 1.0, 0.0, 5.0) is None


@pytest.fixture
def trained() -> tuple[np.ndarray, np.ndarray, Perceptron]:
    X, y = make_blobs(60, seed=0)
    model = Perceptron(2)
    model.train(X, y)
    return X, y, model


def test_plot_file_is_created(
    trained: tuple[np.ndarray, np.ndarray, Perceptron], tmp_path: Path
) -> None:
    X, y, model = trained
    plot_training(X, y, model, tmp_path / "plot.png")
    assert (tmp_path / "plot.png").stat().st_size > 0


def test_plot_untrained_model_raises(tmp_path: Path) -> None:
    X, y = make_blobs(10, seed=0)
    with pytest.raises(ValueError, match="trained"):
        plot_training(X, y, Perceptron(2), tmp_path / "plot.png")
