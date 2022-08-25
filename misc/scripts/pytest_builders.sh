#!/usr/bin/env bash
set -uo pipefail

echo "Running Python checks for builder system"
pytest ./tests/python_build
