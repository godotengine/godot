# Git hooks for Godot Engine

This folder contains Git hooks meant to be installed locally by Godot Engine
contributors to make sure they comply with our requirements.

## List of hooks

- Pre-commit hook for clang-format: Applies clang-format to the staged files
  before accepting a commit; blocks the commit and generates a patch if the
  style is not respected.
  Should work on Linux and macOS. You may need to edit the file if your
  clang-format binary is not in the `$PATH`, or if you want to enable colored
  output with pygmentize.
- Pre-commit hook for makerst: Checks the class reference syntax using `makerst.py`.
  Should work on Linux and macOS.

## Installation

Copy all the files from this folder into your `.git/hooks` folder, and make sure
the hooks and helper scripts are executable.
