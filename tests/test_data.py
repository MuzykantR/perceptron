"""Tests of the datasets and the train/test split."""

import numpy as np
import pytest

from perceptron_lab.data import make_blobs, train_test_split


def test_blobs_shape_and_classes() -> None:
    X, y = make_blobs(101, seed=0)
    assert X.shape == (101, 2)
    assert set(np.unique(y)) == {0, 1}
    assert abs(int(y.sum()) - 50) <= 1


def test_blobs_same_seed_same_data() -> None:
    X1, y1 = make_blobs(50, seed=7)
    X2, y2 = make_blobs(50, seed=7)
    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)


def test_blobs_are_separated_along_x1() -> None:
    X, y = make_blobs(500, seed=3)
    assert X[y == 0, 0].max() < X[y == 1, 0].min()


@pytest.mark.parametrize(("n_samples", "spread"), [(1, 1.0), (10, 0.0), (10, 2.0)])
def test_blobs_invalid_arguments(n_samples: int, spread: float) -> None:
    with pytest.raises(ValueError):
        make_blobs(n_samples, spread=spread)


def test_split_sizes_and_no_lost_samples() -> None:
    X, y = make_blobs(100, seed=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2, seed=0)
    assert (len(X_train), len(X_test)) == (80, 20)
    assert (len(y_train), len(y_test)) == (80, 20)
    together = np.vstack([X_train, X_test])
    assert np.array_equal(np.sort(together, axis=0), np.sort(X, axis=0))
