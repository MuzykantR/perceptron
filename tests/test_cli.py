"""Tests of the command-line interface."""

from pathlib import Path

import pytest

from perceptron_lab.cli import main


def test_cli_prints_accuracy(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--samples", "100", "--lr", "0.1", "--epochs", "50", "--seed", "1"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Train accuracy: 1.000" in out
    assert "Test accuracy:" in out


def test_cli_iris(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--iris"]) == 0
    assert "iris" in capsys.readouterr().out


def test_cli_saves_plot(tmp_path: Path) -> None:
    path = tmp_path / "out" / "plot.png"
    assert main(["--samples", "40", "--plot", str(path)]) == 0
    assert path.exists()


@pytest.mark.parametrize("args", [["--samples", "-5"], ["--lr", "abc"], ["--lr", "0"]])
def test_cli_rejects_bad_arguments(
    args: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(args)
    assert exit_info.value.code == 2
    assert "error" in capsys.readouterr().err


def test_cli_reports_data_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--samples", "1"]) == 1
    assert "Error: n_samples must be at least 2" in capsys.readouterr().err
