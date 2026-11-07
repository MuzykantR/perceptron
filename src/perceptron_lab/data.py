"""Datasets for the perceptron: synthetic 2D blobs."""

import numpy as np

from perceptron_lab.model import FloatArray, IntArray

CENTER_ZERO = (1.0, 3.0)
CENTER_ONE = (3.2, 4.0)
MAX_SPREAD = (CENTER_ONE[0] - CENTER_ZERO[0]) / 2


def make_blobs(
    n_samples: int, seed: int | None = None, spread: float = 1.0
) -> tuple[FloatArray, IntArray]:
    """Generate two linearly separable classes on a plane.

    Class 0 lies around (1, 3), class 1 around (3.2, 4). Each point is the
    center plus uniform noise in [-spread, spread] on both axes. While
    spread < 1.1 the two noise squares do not overlap along x1, so a
    straight line always separates the classes. Neither center is at the
    origin, so the perceptron needs the bias to find that line.

    Args:
        n_samples: Total number of points, at least 2.
        seed: Seed of the random generator, the same seed gives the same data.
        spread: Half-width of the noise square around each center.

    Returns:
        X with shape (n_samples, 2) and y with classes 0 and 1, shuffled.

    Raises:
        ValueError: If n_samples < 2 or spread is not in (0, 1.1).
    """
    if n_samples < 2:
        raise ValueError(f"n_samples must be at least 2, got {n_samples}")
    if not 0 < spread < MAX_SPREAD:
        raise ValueError(f"spread must be in (0, {MAX_SPREAD:g}), got {spread}")

    rng = np.random.default_rng(seed)
    n_zeros = n_samples // 2
    n_ones = n_samples - n_zeros

    x_zeros = rng.uniform(-spread, spread, size=(n_zeros, 2)) + CENTER_ZERO
    x_ones = rng.uniform(-spread, spread, size=(n_ones, 2)) + CENTER_ONE
    X = np.vstack([x_zeros, x_ones])
    y = np.concatenate(
        [np.zeros(n_zeros, dtype=np.int64), np.ones(n_ones, dtype=np.int64)]
    )

    order = rng.permutation(n_samples)
    return X[order], y[order]


def train_test_split(
    X: FloatArray,
    y: IntArray,
    test_ratio: float = 0.2,
    seed: int | None = None,
) -> tuple[FloatArray, FloatArray, IntArray, IntArray]:
    """Shuffle the samples and split them into training and test parts.

    Args:
        X: Samples, shape (n_samples, n_features).
        y: Classes, shape (n_samples,).
        test_ratio: Share of samples that go to the test part.
        seed: Seed of the shuffle.

    Returns:
        X_train, X_test, y_train, y_test.

    Raises:
        ValueError: If test_ratio is not in (0, 1) or a part would be empty.
    """
    if not 0 < test_ratio < 1:
        raise ValueError(f"test_ratio must be in (0, 1), got {test_ratio}")
    n_test = round(len(X) * test_ratio)
    if n_test == 0 or n_test == len(X):
        raise ValueError(f"cannot split {len(X)} samples with ratio {test_ratio}")

    order = np.random.default_rng(seed).permutation(len(X))
    test_idx, train_idx = order[:n_test], order[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
