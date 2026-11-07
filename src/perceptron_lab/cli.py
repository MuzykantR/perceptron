"""Command-line interface: generate data, train the perceptron, print accuracy."""

import argparse
import sys
from pathlib import Path

from perceptron_lab.data import (
    IRIS_FEATURES,
    load_iris_binary,
    make_blobs,
    train_test_split,
)
from perceptron_lab.model import FloatArray, IntArray, Perceptron
from perceptron_lab.plots import animate_training, plot_training


def positive_int(text: str) -> int:
    """Parse a positive integer for argparse.

    Args:
        text: Raw command-line value.

    Returns:
        The parsed number.

    Raises:
        argparse.ArgumentTypeError: If the value is not a positive integer.
    """
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}") from None
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {value}")
    return value


def positive_float(text: str) -> float:
    """Parse a positive number for argparse.

    Args:
        text: Raw command-line value.

    Returns:
        The parsed number.

    Raises:
        argparse.ArgumentTypeError: If the value is not a positive number.
    """
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, got {text!r}") from None
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    """Describe the command-line options.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="perceptron",
        description="Train a perceptron on two linearly separable classes.",
    )
    parser.add_argument(
        "--samples", type=positive_int, default=200, help="number of points"
    )
    parser.add_argument("--lr", type=positive_float, default=0.1, help="learning rate")
    parser.add_argument(
        "--epochs", type=positive_int, default=50, help="maximum number of epochs"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--iris",
        action="store_true",
        help="use iris (setosa vs versicolor) instead of synthetic data",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        metavar="FILE.png",
        help="save the decision boundary and the errors per epoch",
    )
    parser.add_argument(
        "--animate",
        type=Path,
        metavar="FILE.gif",
        help="save an animation of the training, one frame per epoch",
    )
    return parser


def load_dataset(
    args: argparse.Namespace,
) -> tuple[FloatArray, IntArray, str, tuple[str, str]]:
    """Load the dataset chosen on the command line.

    Args:
        args: Parsed command-line arguments.

    Returns:
        X, y, a short name of the data and the names of the two features.

    Raises:
        ValueError: If the synthetic data cannot be generated.
    """
    if args.iris:
        X, y = load_iris_binary()
        return X, y, "iris: setosa vs versicolor", IRIS_FEATURES
    X, y = make_blobs(args.samples, seed=args.seed)
    return X, y, "synthetic blobs", ("x1", "x2")


def print_report(
    model: Perceptron,
    split: tuple[FloatArray, FloatArray, IntArray, IntArray],
    source: str,
) -> None:
    """Print the training summary and the accuracy.

    Args:
        model: Trained perceptron.
        split: X_train, X_test, y_train, y_test.
        source: Short name of the data.
    """
    X_train, X_test, y_train, y_test = split
    n_total = len(X_train) + len(X_test)
    print(f"Data: {source}, {n_total} samples")
    print(f"      {len(X_train)} for training, {len(X_test)} for testing")
    print(f"Epochs run: {len(model.history_)} of {model.epochs}")
    print(f"Errors per epoch: {[errors for _, _, errors in model.history_]}")
    print(f"Weights: {model.weights.round(3)}, bias: {model.bias:.3f}")
    print(f"Train accuracy: {model.score(X_train, y_train):.3f}")
    print(f"Test accuracy:  {model.score(X_test, y_test):.3f}")


def main(argv: list[str] | None = None) -> int:
    """Run the program: load data, train, print accuracy, save plots.

    Args:
        argv: Command-line arguments without the program name;
            None means sys.argv[1:].

    Returns:
        Exit code: 0 on success, 1 on an error in the data, the model
        or while saving a file.
    """
    args = build_parser().parse_args(argv)
    try:
        X, y, source, feature_names = load_dataset(args)
        split = train_test_split(X, y, seed=args.seed)
        model = Perceptron(X.shape[1], learning_rate=args.lr, epochs=args.epochs)
        model.train(split[0], split[2])
        print_report(model, split, source)

        if args.plot is not None:
            plot_training(X, y, model, args.plot, feature_names)
            print(f"Plot saved to {args.plot}")
        if args.animate is not None:
            animate_training(X, y, model, args.animate, feature_names)
            print(f"Animation saved to {args.animate}")
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0
