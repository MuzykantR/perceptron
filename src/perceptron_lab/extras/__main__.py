"""Demo of the extensions: one-vs-rest on all of iris and the sklearn comparison."""

import numpy as np
from sklearn.datasets import load_iris

from perceptron_lab.data import load_iris_binary, make_blobs, train_test_split
from perceptron_lab.extras.compare import compare_with_sklearn
from perceptron_lab.extras.multiclass import OneVsRestPerceptron

SEED = 42


def demo_one_vs_rest() -> None:
    """Train one-vs-rest on 3 iris species with 4 features and print accuracy."""
    iris = load_iris()
    X = np.asarray(iris.data, dtype=np.float64)
    y = np.asarray(iris.target, dtype=np.int64)
    X_train, X_test, y_train, y_test = train_test_split(X, y, seed=SEED)

    model = OneVsRestPerceptron(n_features=4, n_classes=3, epochs=100)
    model.train(X_train, y_train)
    print("One-vs-rest, iris, 3 classes, 4 features")
    print(f"  train accuracy: {model.score(X_train, y_train):.3f}")
    print(f"  test accuracy:  {model.score(X_test, y_test):.3f}")


def demo_compare() -> None:
    """Print test accuracy of our perceptron and sklearn on two datasets."""
    print("Our perceptron vs sklearn.linear_model.Perceptron (test accuracy)")
    datasets = {
        "synthetic blobs": make_blobs(200, seed=SEED),
        "iris binary": load_iris_binary(),
    }
    for name, (X, y) in datasets.items():
        split = train_test_split(X, y, seed=SEED)
        ours, theirs = compare_with_sklearn(*split)
        print(f"  {name:16} ours {ours:.3f}   sklearn {theirs:.3f}")


def main() -> None:
    """Run both demos."""
    demo_one_vs_rest()
    print()
    demo_compare()


if __name__ == "__main__":
    main()
