"""Compare our perceptron with sklearn.linear_model.Perceptron on the same data."""

import numpy as np
from sklearn.linear_model import Perceptron as SklearnPerceptron

from perceptron_lab.model import FloatArray, IntArray, Perceptron


def compare_with_sklearn(
    X_train: FloatArray,
    X_test: FloatArray,
    y_train: IntArray,
    y_test: IntArray,
    learning_rate: float = 0.1,
    epochs: int = 50,
) -> tuple[float, float]:
    """Train both models on the same split and return their test accuracy.

    sklearn is told to go through the samples in the same order (no
    shuffle) and without regularization, so both use the same rule.

    Args:
        X_train: Training samples.
        X_test: Test samples.
        y_train: Training classes 0 and 1.
        y_test: Test classes 0 and 1.
        learning_rate: Step size for both models.
        epochs: Maximum number of epochs for both models.

    Returns:
        Test accuracy of our perceptron and of the sklearn one.
    """
    ours = Perceptron(X_train.shape[1], learning_rate=learning_rate, epochs=epochs)
    ours.train(X_train, y_train)

    theirs = SklearnPerceptron(
        eta0=learning_rate, max_iter=epochs, shuffle=False, tol=None
    )
    theirs.fit(X_train, y_train)
    theirs_accuracy = float(np.mean(theirs.predict(X_test) == y_test))

    return ours.score(X_test, y_test), theirs_accuracy
