"""Single-layer perceptron for binary classification (classes 0 and 1)."""

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]


class Perceptron:
    """Binary linear classifier trained with the perceptron learning rule.

    The model computes z = w . x + b and returns 1 if z >= 0, otherwise 0.

    Attributes:
        w: Weights, one per feature.
        b: Bias (threshold shift).
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
        self.w: FloatArray = np.zeros(n_features)
        self.b: float = 0.0
        self.history_: list[tuple[FloatArray, float, int]] = []

    def activate(self, z: float) -> int:
        """Apply the step function.

        Args:
            z: Weighted sum w . x + b.

        Returns:
            1 if z >= 0, otherwise 0.
        """
        return 1 if z >= 0 else 0

    def predict(self, x: FloatArray) -> int:
        """Predict the class of one sample.

        Args:
            x: Feature vector of length n_features.

        Returns:
            Predicted class, 0 or 1.

        Raises:
            ValueError: If the length of x does not match the number of weights.
        """
        if len(x) != len(self.w):
            raise ValueError(
                f"expected {len(self.w)} features, got {len(x)}"
            )

        z = float(self.w @ x + self.b)
        return self.activate(z)
