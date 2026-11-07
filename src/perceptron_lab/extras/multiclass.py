"""One-vs-rest: several binary perceptrons for more than two classes.

Some "class k or not" tasks are not linearly separable (versicolor against
the other two iris species). Then the perceptron never reaches an epoch
without errors and its last weights can be poor, so each binary model is a
pocket perceptron: it keeps the best weights seen during training.
Features are standardized (mean 0, std 1), so that the weighted sums of
different perceptrons are on a comparable scale.
"""

import numpy as np

from perceptron_lab.model import FloatArray, IntArray, Perceptron


class PocketPerceptron(Perceptron):
    """Perceptron that ends with the best weights from its history."""

    def train(self, X: FloatArray, y: IntArray) -> None:
        """Train as usual, then restore the epoch with the best accuracy.

        Args:
            X: Training samples, shape (n_samples, n_features).
            y: True classes 0 or 1.
        """
        super().train(X, y)
        best_accuracy = -1.0
        best_weights, best_bias = self.weights, self.bias
        for weights, bias, _ in self.history_:
            self.weights, self.bias = weights, bias
            accuracy = self.score(X, y)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weights, best_bias = weights, bias
        self.weights, self.bias = best_weights.copy(), best_bias


class OneVsRestPerceptron:
    """Multiclass classifier built from one pocket perceptron per class.

    Perceptron number k learns "class k or not". A sample gets the class
    whose perceptron gives the largest weighted sum z = w . x + b.

    Attributes:
        models: One trained perceptron per class.
        mean_: Feature means of the training data.
        std_: Feature standard deviations of the training data.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        learning_rate: float = 0.1,
        epochs: int = 50,
    ) -> None:
        """Create n_classes untrained perceptrons.

        Args:
            n_features: Number of input features.
            n_classes: Number of classes, at least 2.
            learning_rate: Step size of each perceptron.
            epochs: Maximum number of epochs of each perceptron.

        Raises:
            ValueError: If n_classes < 2 or another argument is not positive.
        """
        if n_classes < 2:
            raise ValueError(f"n_classes must be at least 2, got {n_classes}")
        self.models: list[Perceptron] = [
            PocketPerceptron(n_features, learning_rate=learning_rate, epochs=epochs)
            for _ in range(n_classes)
        ]
        self.mean_: FloatArray = np.zeros(n_features)
        self.std_: FloatArray = np.ones(n_features)

    def train(self, X: FloatArray, y: IntArray) -> None:
        """Train each perceptron on the labels "class k" = 1, others = 0.

        Args:
            X: Samples, shape (n_samples, n_features).
            y: Classes from 0 to n_classes - 1.
        """
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.std_ = np.where(std > 0, std, 1.0)
        X_scaled = (X - self.mean_) / self.std_
        for cls, model in enumerate(self.models):
            model.train(X_scaled, (y == cls).astype(np.int64))

    def predict(self, x: FloatArray) -> int:
        """Return the class whose perceptron is the most confident.

        Args:
            x: Feature vector.

        Returns:
            Predicted class.
        """
        x_scaled = (x - self.mean_) / self.std_
        sums = [model.net_input(x_scaled) for model in self.models]
        return int(np.argmax(sums))

    def score(self, X: FloatArray, y: IntArray) -> float:
        """Return the share of correctly classified samples.

        Args:
            X: Samples, shape (n_samples, n_features).
            y: True classes.

        Returns:
            Accuracy from 0.0 to 1.0.
        """
        correct = 0
        for x, target in zip(X, y, strict=True):
            if self.predict(x) == target:
                correct += 1
        return correct / len(y)
