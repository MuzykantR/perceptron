"""Tests of the optional extensions."""

import numpy as np
from sklearn.datasets import load_iris

from perceptron_lab.data import make_blobs, train_test_split
from perceptron_lab.extras.compare import compare_with_sklearn
from perceptron_lab.extras.multiclass import OneVsRestPerceptron


def test_one_vs_rest_on_three_iris_classes() -> None:
    iris = load_iris()
    X = np.asarray(iris.data, dtype=np.float64)
    y = np.asarray(iris.target, dtype=np.int64)
    model = OneVsRestPerceptron(n_features=4, n_classes=3, epochs=100)
    model.train(X, y)
    assert len(model.models) == 3
    assert model.score(X, y) > 0.8


def test_ours_and_sklearn_on_separable_data() -> None:
    X, y = make_blobs(200, seed=0)
    ours, theirs = compare_with_sklearn(*train_test_split(X, y, seed=0))
    assert ours == 1.0
    assert theirs >= 0.95
