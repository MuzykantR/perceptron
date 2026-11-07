"""Plots of the perceptron: decision boundary, errors per epoch, animation.

Figures are built without pyplot and only saved to files, so the module
works on a server without a display (for example in CI).
"""

from pathlib import Path

import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from perceptron_lab.model import FloatArray, IntArray, Perceptron

CLASS_COLORS = ("tab:blue", "tab:orange")


def boundary_line(
    weights: FloatArray, bias: float, x_min: float, x_max: float
) -> tuple[FloatArray, FloatArray] | None:
    """Find two points of the line w1 * x1 + w2 * x2 + b = 0.

    Args:
        weights: Two weights w1 and w2.
        bias: Bias b.
        x_min: Left edge of the plot on the x1 axis.
        x_max: Right edge of the plot on the x1 axis.

    Returns:
        Coordinates (x1, x2) of two points of the line, or None if both
        weights are zero and the line does not exist.

    Raises:
        ValueError: If the number of weights is not 2.
    """
    if len(weights) != 2:
        raise ValueError(f"a line needs exactly 2 weights, got {len(weights)}")
    w1, w2 = float(weights[0]), float(weights[1])
    if w2 != 0:
        x1 = np.array([x_min, x_max])
        return x1, -(w1 * x1 + bias) / w2
    if w1 != 0:
        x1_value = -bias / w1
        return np.array([x1_value, x1_value]), np.array([-1e6, 1e6])
    return None


def _draw_points(
    ax: Axes,
    X: FloatArray,
    y: IntArray,
    feature_names: tuple[str, str],
) -> None:
    """Scatter the samples coloured by class and set the axis limits.

    Args:
        ax: Axes to draw on.
        X: Samples, shape (n_samples, 2).
        y: Classes 0 and 1.
        feature_names: Labels of the two axes.
    """
    for cls, color in enumerate(CLASS_COLORS):
        points = X[y == cls]
        ax.scatter(points[:, 0], points[:, 1], s=18, c=color, label=f"class {cls}")
    margin = 0.5
    ax.set_xlim(X[:, 0].min() - margin, X[:, 0].max() + margin)
    ax.set_ylim(X[:, 1].min() - margin, X[:, 1].max() + margin)
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])


def _set_line(line: Line2D, ax: Axes, weights: FloatArray, bias: float) -> None:
    """Move a line object to the decision boundary for given weights.

    Args:
        line: Line to update.
        ax: Axes with the points, used for the x range.
        weights: Two weights.
        bias: Bias.
    """
    x_min, x_max = ax.get_xlim()
    points = boundary_line(weights, bias, x_min, x_max)
    if points is None:
        line.set_data([], [])
    else:
        line.set_data(points[0], points[1])


def _setup_errors_axes(ax: Axes, n_epochs: int, max_errors: int) -> None:
    """Prepare the axes of the errors-per-epoch plot.

    Args:
        ax: Axes to set up.
        n_epochs: Number of epochs on the x axis.
        max_errors: Largest number of errors, for the y range.
    """
    ax.set_xlim(0.5, n_epochs + 0.5)
    ax.set_ylim(0, max_errors + 1)
    ax.set_xlabel("epoch")
    ax.set_ylabel("misclassified points")
    ax.set_title("Errors per epoch")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.3)


def plot_training(
    X: FloatArray,
    y: IntArray,
    model: Perceptron,
    path: Path,
    feature_names: tuple[str, str] = ("x1", "x2"),
) -> None:
    """Save a figure: data with the final boundary and errors per epoch.

    Args:
        X: Samples with two features.
        y: Classes 0 and 1.
        model: Trained perceptron with two weights.
        path: Output image file, for example result.png.
        feature_names: Labels of the two axes.

    Raises:
        ValueError: If the model has not been trained.
    """
    if not model.history_:
        raise ValueError("the model must be trained before plotting")
    errors = [epoch_errors for _, _, epoch_errors in model.history_]

    fig = Figure(figsize=(11, 4.5), layout="constrained")
    ax_data, ax_errors = fig.subplots(1, 2)

    _draw_points(ax_data, X, y, feature_names)
    (line,) = ax_data.plot([], [], "k-", linewidth=2, label="decision boundary")
    _set_line(line, ax_data, model.weights, model.bias)
    ax_data.set_title(f"Decision boundary, accuracy {model.score(X, y):.3f}")
    ax_data.legend(loc="best")

    epochs = range(1, len(errors) + 1)
    _setup_errors_axes(ax_errors, len(errors), max(errors))
    ax_errors.plot(epochs, errors, "o-", color="tab:red")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def animate_training(
    X: FloatArray,
    y: IntArray,
    model: Perceptron,
    path: Path,
    feature_names: tuple[str, str] = ("x1", "x2"),
    fps: int = 2,
) -> None:
    """Save a GIF: the boundary moves epoch by epoch, errors are drawn alongside.

    Each frame is one record of model.history_.

    Args:
        X: Samples with two features.
        y: Classes 0 and 1.
        model: Trained perceptron with two weights.
        path: Output GIF file.
        feature_names: Labels of the two axes.
        fps: Frames (epochs) per second.

    Raises:
        ValueError: If the model has not been trained.
    """
    if not model.history_:
        raise ValueError("the model must be trained before animating")
    history = model.history_
    errors = [epoch_errors for _, _, epoch_errors in history]

    fig = Figure(figsize=(11, 4.5), layout="constrained")
    ax_data, ax_errors = fig.subplots(1, 2)
    _draw_points(ax_data, X, y, feature_names)
    (boundary,) = ax_data.plot([], [], "k-", linewidth=2, label="decision boundary")
    ax_data.legend(loc="best")
    _setup_errors_axes(ax_errors, len(errors), max(errors))
    (error_line,) = ax_errors.plot([], [], "o-", color="tab:red")

    def update(frame: int) -> tuple[Line2D, Line2D]:
        """Draw the state after epoch number frame + 1."""
        weights, bias, _ = history[frame]
        _set_line(boundary, ax_data, weights, bias)
        ax_data.set_title(f"Epoch {frame + 1} of {len(history)}")
        error_line.set_data(range(1, frame + 2), errors[: frame + 1])
        return boundary, error_line

    animation = FuncAnimation(fig, update, frames=len(history), blit=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(str(path), writer=PillowWriter(fps=fps))
