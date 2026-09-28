"""
tests/test_run_tests_semantics.py — Regression tests for Issue #185
Verifies that tests/run_tests.py fails closed when required native test
stages (Stage 4, Stage 5) are executed without their native library artifact,
while pure Python stages (Stage 2, Stage 3) continue to succeed.
"""

import os
import shutil
import subprocess
import sys
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_TESTS_SCRIPT = os.path.join(REPO_ROOT, "tests", "run_tests.py")


def _run_stage(stage_num, cwd=None, script_path=None):
    cmd = [
        sys.executable,
        script_path or RUN_TESTS_SCRIPT,
    ]
    if stage_num > 0:
        cmd.extend(["--stage", str(stage_num)])
    return subprocess.run(
        cmd,
        cwd=cwd or REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def test_stage4_missing_native_fails():
    """Test A: Stage 4 must exit non-zero and report failure when native library is absent."""
    res = _run_stage(4)
    assert res.returncode != 0, f"Expected non-zero exit code, got {res.returncode}\nOutput:\n{res.stdout}"
    assert "[FAIL]" in res.stdout or "[ FAIL ]" in res.stdout
    assert "*** ALL STAGES PASS ***" not in res.stdout
    assert "C ABI integration: required native library not found" in res.stdout or "ERROR:" in res.stdout


def test_stage5_missing_native_fails():
    """Test B: Stage 5 must exit non-zero and report failure when native library is absent."""
    res = _run_stage(5)
    assert res.returncode != 0, f"Expected non-zero exit code, got {res.returncode}\nOutput:\n{res.stdout}"
    assert "[FAIL]" in res.stdout or "[ FAIL ]" in res.stdout
    assert "*** ALL STAGES PASS ***" not in res.stdout
    assert "Cross-validation: required native library not found" in res.stdout or "ERROR:" in res.stdout


def test_stage2_pure_python_succeeds():
    """Test C: Stage 2 (MSE calibration) is pure Python and must succeed with exit code 0."""
    res = _run_stage(2)
    assert res.returncode == 0, f"Expected exit code 0, got {res.returncode}\nOutput:\n{res.stdout}\nStderr:\n{res.stderr}"
    assert "[PASS]" in res.stdout or "[ PASS ]" in res.stdout
    assert "*** ALL STAGES PASS" in res.stdout


def test_stage3_pure_python_succeeds():
    """Test D: Stage 3 (Python reference benchmark) is pure Python and must succeed with exit code 0."""
    res = _run_stage(3)
    assert res.returncode == 0, f"Expected exit code 0, got {res.returncode}\nOutput:\n{res.stdout}\nStderr:\n{res.stderr}"
    assert "[PASS]" in res.stdout or "[ PASS ]" in res.stdout
    assert "*** ALL STAGES PASS" in res.stdout


def test_full_runner_without_native_fails():
    """Full runner execution without native libraries must fail and never report all stages pass."""
    res = _run_stage(0)
    assert res.returncode != 0, f"Expected non-zero exit code for full suite when native is missing\nOutput:\n{res.stdout}"
    assert "*** ALL STAGES PASS ***" not in res.stdout
    assert "*** SOME STAGES FAILED" in res.stdout


def test_isolated_cleanroom_missing_native(tmp_path):
    """Cleanroom test: runs run_tests.py in an isolated directory with guaranteed absence of build artifacts."""
    isolated_tests = tmp_path / "tests"
    isolated_tests.mkdir()
    isolated_script = isolated_tests / "run_tests.py"
    shutil.copyfile(RUN_TESTS_SCRIPT, isolated_script)

    # Stage 4 in cleanroom
    res4 = _run_stage(4, cwd=str(tmp_path), script_path=str(isolated_script))
    assert res4.returncode != 0
    assert "[FAIL]" in res4.stdout or "[ FAIL ]" in res4.stdout
    assert "*** ALL STAGES PASS ***" not in res4.stdout

    # Stage 5 in cleanroom
    res5 = _run_stage(5, cwd=str(tmp_path), script_path=str(isolated_script))
    assert res5.returncode != 0
    assert "[FAIL]" in res5.stdout or "[ FAIL ]" in res5.stdout
    assert "*** ALL STAGES PASS ***" not in res5.stdout

    # Stage 2 in cleanroom
    res2 = _run_stage(2, cwd=str(tmp_path), script_path=str(isolated_script))
    assert res2.returncode == 0
    assert "[PASS]" in res2.stdout or "[ PASS ]" in res2.stdout
