# Git hooks for Godot Engine

This folder contains Git hooks meant to be installed locally by Godot Engine
contributors to make sure they comply with our requirements.

## List of hooks

- Pre-commit hook for `clang-format`: Applies `clang-format` to the staged
  files before accepting a commit; blocks the commit and generates a patch if
  the style is not respected.
  You may need to edit the file if your `clang-format` binary is not in the
  `PATH`, or if you want to enable colored output with `pygmentize`.
- Pre-commit hook for `black`: Applies `black` to the staged Python files
  before accepting a commit.
- Pre-commit hook for `make_rst`: Checks the class reference syntax using
  `make_rst.py`.

## Installation

Copy all the files from this folder into your `.git/hooks` folder, and make
sure the hooks and helper scripts are executable.

#### Linux/MacOS

The hooks rely on bash scripts and tools which should be in the system `PATH`,
so they should work out of the box on Linux/macOS.

#### Windows

##### clang-format
- Download LLVM for Windows (version 13 or later) from
  <https://releases.llvm.org/download.html>
- Make sure LLVM is added to the `PATH` during installation

##### black
- Python installation: make sure Python is added to the `PATH`
- Install `black` - in any console: `pip3 install black`
