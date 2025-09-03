import os
import re
import sys
import subprocess

import pytest


# Skip these tests entirely if scikit-learn is not available in the env.
pytest.importorskip("sklearn")


def _run_script() -> subprocess.CompletedProcess:
    """Run the Linear_regression script and return the completed process.

    Runs using the current Python interpreter to avoid PATH issues on Windows.
    Captures stdout/stderr as text for parsing.
    """
    return subprocess.run(
        [sys.executable, "-u", "Linear_regression.py"],
        cwd=os.getcwd(),
        capture_output=True,
        text=True,
        check=False,
    )


def _extract_metric_lines(stdout: str):
    """Return the lines from stdout that look like per-target metric rows."""
    lines = [ln for ln in stdout.splitlines() if "MSE=" in ln and "MAE=" in ln]
    return lines


def _parse_metrics(line: str):
    """Extract (mse, mae, r2) floats from a metric line regardless of labels.

    The source file may contain non-ASCII label text (e.g., corrupted 'R^2').
    We therefore grab the first three floats present on the line in order.
    """
    nums = re.findall(r"[-+]?(?:\d+\.\d+|\d+)", line)
    assert len(nums) >= 3, f"Could not parse three metrics from line: {line!r}"
    mse, mae, r2 = map(float, nums[:3])
    return mse, mae, r2


def test_script_runs_and_reports_two_targets():
    """Script executes successfully and prints two per-target metric rows."""
    proc = _run_script()
    assert (
        proc.returncode == 0
    ), f"Script failed with code {proc.returncode}:\nSTDERR:\n{proc.stderr}"

    metric_lines = _extract_metric_lines(proc.stdout)
    assert len(metric_lines) == 2, f"Expected 2 metric lines, got {len(metric_lines)}\n{proc.stdout}"

    for line in metric_lines:
        mse, mae, r2 = _parse_metrics(line)
        # Basic sanity checks on metric values
        assert mse >= 0.0, f"MSE should be non-negative, got {mse}"
        assert mae >= 0.0, f"MAE should be non-negative, got {mae}"
        assert r2 <= 1.0 + 1e-9, f"R^2 should be <= 1.0, got {r2}"


def test_metrics_are_deterministic_with_fixed_seed():
    """With fixed random_state in the script, outputs should be identical across runs."""
    first = _run_script()
    second = _run_script()

    assert first.returncode == 0, f"First run failed:\n{first.stderr}"
    assert second.returncode == 0, f"Second run failed:\n{second.stderr}"

    # Extract only the metric lines (ignore header and any other noise)
    lines1 = _extract_metric_lines(first.stdout)
    lines2 = _extract_metric_lines(second.stdout)
    assert lines1 == lines2, "Metric lines differ between runs with fixed seed."

