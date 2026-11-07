"""Tests of the Perceptron class."""

import numpy as np
import pytest

from perceptron_lab.data import make_blobs
from perceptron_lab.model import Perceptron


@pytest.mark.parametrize(("z", "expected"), [(0.0, 1), (2.5, 1), (-1.0, 0), (-1e-9, 0)])
def test_activate_threshold(z: float, expected: int) -> None:
    assert Perceptron(2).activate(z) == expected


@pytest.mark.parametrize("n_features", [1, 2, 4])
def test_weights_have_expected_length(n_features: int) -> None:
    model = Perceptron(n_features)
    assert model.weights.shape == (n_features,)
    assert np.all(model.weights == 0)
    assert model.bias == 0.0


def test_predict_with_known_weights() -> None:
    # boundary x1 + x2 - 4 = 0: class 1 above it, class 0 below
    model = Perceptron(2)
    model.weights = np.array([1.0, 1.0])
    model.bias = -4.0
    assert model.predict(np.array([3.0, 3.0])) == 1
    assert model.predict(np.array([1.0, 1.0])) == 0
    assert model.predict(np.array([2.0, 2.0])) == 1  # exactly on the line: z = 0


def test_training_separates_linearly_separable_classes() -> None:
    X, y = make_blobs(200, seed=0)
    model = Perceptron(2, learning_rate=0.1, epochs=100)
    model.train(X, y)
    assert model.score(X, y) == 1.0


def test_training_stops_after_error_free_epoch() -> None:
    X, y = make_blobs(100, seed=1)
    model = Perceptron(2, epochs=100)
    model.train(X, y)
    errors = [epoch_errors for _, _, epoch_errors in model.history_]
    assert errors[-1] == 0
    assert all(e > 0 for e in errors[:-1])
    assert len(errors) <= 100


def test_history_keeps_a_copy_of_weights() -> None:
    X, y = make_blobs(100, seed=2)
    model = Perceptron(2)
    model.train(X, y)
    first_weights = model.history_[0][0]
    last_weights = model.history_[-1][0]
    assert last_weights is not model.weights
    assert len(model.history_) == 1 or not np.array_equal(first_weights, last_weights)


def test_score_is_share_of_correct_answers() -> None:
    model = Perceptron(1)  # zero weights: z = 0, so every answer is 1
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([1, 1, 1, 0])
    assert model.score(X, y) == 0.75


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_features": 0},
        {"n_features": 2, "learning_rate": 0.0},
        {"n_features": 2, "epochs": -1},
    ],
)
def test_invalid_parameters_raise(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        Perceptron(**kwargs)  # type: ignore[arg-type]


def test_wrong_number_of_features_raises() -> None:
    model = Perceptron(2)
    with pytest.raises(ValueError, match="expected 2 features"):
        model.predict(np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="same length"):
        model.train(np.zeros((3, 2)), np.array([0, 1]))
