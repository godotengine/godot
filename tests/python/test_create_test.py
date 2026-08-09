#!/usr/bin/env python3
"""Tests for tests/create_test.py."""

import os
import subprocess
import sys
import tempfile


def test_create_test_uses_callers_cwd():
    """
    create_test.py chdir()s into its own directory before resolving the
    "path" argument. Make sure that a relative default path ('.') is
    resolved against the caller's original working directory, not against
    the tests/ folder where the script itself lives.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    script_path = os.path.join(repo_root, "tests", "create_test.py")

    with tempfile.TemporaryDirectory() as tmp_dir:
        result = subprocess.run(
            [sys.executable, script_path, "MyDummyThing"],
            cwd=tmp_dir,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"

        expected_file = os.path.join(tmp_dir, "test_my_dummy_thing.cpp")
        assert os.path.isfile(expected_file), (
            f"Expected scaffolded file at {expected_file}, but it was not created there.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

        # Make sure it was NOT created inside the tests/ folder instead.
        wrong_file = os.path.join(repo_root, "tests", "test_my_dummy_thing.cpp")
        assert not os.path.isfile(wrong_file), (
            "The scaffolded file was incorrectly created inside tests/ instead of the caller's cwd."
        )
