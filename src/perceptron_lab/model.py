"""Single-layer perceptron for binary classification (classes 0 and 1)."""

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int64]


class Perceptron:
    """Binary linear classifier trained with the perceptron learning rule.

    The model computes z = w . x + b and returns 1 if z >= 0, otherwise 0.

    Attributes:
        weights: Weights w, one per feature.
        bias: Bias b (threshold shift).
        learning_rate: Step size of each weight update.
        epochs: Maximum number of passes over the training set.
        history_: Weights, bias and number of errors after each epoch.
    """

    def __init__(
        self, n_features: int, learning_rate: float = 0.1, epochs: int = 50
    ) -> None:
        """Create a perceptron with zero weights and zero bias.

        Args:
            n_features: Number of input features.
            learning_rate: Step size of each weight update, must be positive.
            epochs: Maximum number of training epochs, must be positive.

        Raises:
            ValueError: If any argument is not positive.
        """
        if n_features <= 0:
            raise ValueError(f"n_features must be positive, got {n_features}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if epochs <= 0:
            raise ValueError(f"epochs must be positive, got {epochs}")

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights: FloatArray = np.zeros(n_features)
        self.bias: float = 0.0
        self.history_: list[tuple[FloatArray, float, int]] = []

    def activate(self, z: float) -> int:
        """Apply the step function.

        Args:
            z: Weighted sum w . x + b.

        Returns:
            1 if z >= 0, otherwise 0.
        """
        return 1 if z >= 0 else 0

    def net_input(self, x: FloatArray) -> float:
        """Compute the weighted sum z = w . x + b for one sample.

        Args:
            x: Feature vector of length n_features.

        Returns:
            The weighted sum before the step function.

        Raises:
            ValueError: If the length of x does not match the number of weights.
        """
        if len(x) != len(self.weights):
            raise ValueError(f"expected {len(self.weights)} features, got {len(x)}")
        return float(self.weights @ x + self.bias)

    def predict(self, x: FloatArray) -> int:
        """Predict the class of one sample.

        Args:
            x: Feature vector of length n_features.

        Returns:
            Predicted class, 0 or 1.

        Raises:
            ValueError: If the length of x does not match the number of weights.
        """
        return self.activate(self.net_input(x))

    def _check_dataset(self, X: FloatArray, y: IntArray) -> None:
        """Check that X and y fit the model and each other.

        Args:
            X: Samples, shape (n_samples, n_features).
            y: True classes, shape (n_samples,).

        Raises:
            ValueError: If the shapes do not match.
        """
        if X.ndim != 2 or X.shape[1] != len(self.weights):
            raise ValueError(
                f"X must have shape (n_samples, {len(self.weights)}), got {X.shape}"
            )
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have the same length, got {len(X)} and {len(y)}"
            )

    def train(self, X: FloatArray, y: IntArray) -> None:
        """Fit the weights with the perceptron learning rule.

        For every sample the weights move by lr * (target - pred) * x.
        Training stops early when an epoch has no errors.
        After each epoch a copy of the weights, the bias and the number
        of errors is appended to history_.

        Args:
            X: Training samples, shape (n_samples, n_features).
            y: True classes 0 or 1, shape (n_samples,).

        Raises:
            ValueError: If the shapes of X and y do not match.
        """
        self._check_dataset(X, y)
        self.history_ = []
        for _ in range(self.epochs):
            errors = 0
            for x, target in zip(X, y, strict=True):
                pred = self.predict(x)
                delta = self.learning_rate * (target - pred)
                if delta != 0:
                    self.weights = self.weights + delta * x
                    self.bias = float(self.bias + delta)
                    errors += 1
            self.history_.append((self.weights.copy(), self.bias, errors))
            if errors == 0:
                break

    def score(self, X: FloatArray, y: IntArray) -> float:
        """Return the share of correctly classified samples.

        Args:
            X: Samples, shape (n_samples, n_features).
            y: True classes 0 or 1, shape (n_samples,).

        Returns:
            Accuracy from 0.0 to 1.0.

        Raises:
            ValueError: If the shapes of X and y do not match.
        """
        self._check_dataset(X, y)

        correct = 0
        for x, target in zip(X, y, strict=True):
            if self.predict(x) == target:
                correct += 1
        return correct / len(y)
