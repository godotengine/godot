#!/usr/bin/env python3
"""Tests for tests/create_test.py."""

import os
import subprocess
import sys
import tempfile


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_create_test_default_path_relative_to_tests_dir():
    """
    The default path ('.') must always resolve relative to the tests/
    folder, regardless of the caller's current working directory, since
    create_test.py chdir()s into tests/ and the path argument is documented
    as relative to the tests/ folder.
    """
    repo_root = _repo_root()
    script_path = os.path.join(repo_root, "tests", "create_test.py")
    tests_dir = os.path.join(repo_root, "tests")

    with tempfile.TemporaryDirectory() as tmp_dir:
        expected_file = os.path.join(tests_dir, "test_my_dummy_thing.cpp")
        try:
            result = subprocess.run(
                [sys.executable, script_path, "MyDummyThing"],
                cwd=tmp_dir,
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"

            assert os.path.isfile(expected_file), (
                f"Expected scaffolded file at {expected_file}, but it was not created there.\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )

            # Make sure it was NOT created inside the caller's cwd instead.
            wrong_file = os.path.join(tmp_dir, "test_my_dummy_thing.cpp")
            assert not os.path.isfile(wrong_file), (
                "The scaffolded file was incorrectly created in the caller's cwd instead of tests/."
            )
        finally:
            if os.path.isfile(expected_file):
                os.remove(expected_file)


def test_create_test_explicit_relative_path_relative_to_tests_dir():
    """
    An explicit relative path argument (e.g. "core/math") must resolve
    relative to the tests/ folder, not relative to the caller's original
    working directory (e.g. the repo root).
    """
    repo_root = _repo_root()
    script_path = os.path.join(repo_root, "tests", "create_test.py")
    tests_dir = os.path.join(repo_root, "tests")

    expected_file = os.path.join(tests_dir, "core", "math", "test_my_other_thing.cpp")
    wrong_file = os.path.join(repo_root, "core", "math", "test_my_other_thing.cpp")
    try:
        result = subprocess.run(
            [sys.executable, script_path, "MyOtherThing", "core/math"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"

        assert os.path.isfile(expected_file), (
            f"Expected scaffolded file at {expected_file}, but it was not created there.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

        assert not os.path.isfile(wrong_file), (
            "The scaffolded file was incorrectly created relative to the repo root "
            "instead of relative to tests/."
        )
    finally:
        if os.path.isfile(expected_file):
            os.remove(expected_file)
        if os.path.isfile(wrong_file):
            os.remove(wrong_file)
