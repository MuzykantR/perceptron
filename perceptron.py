"""Entry point: uv run python perceptron.py --samples 200 --lr 0.1 --epochs 50."""

import sys

from perceptron_lab.cli import main

if __name__ == "__main__":
    sys.exit(main())
