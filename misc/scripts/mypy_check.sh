#!/usr/bin/env bash

set -uo pipefail

echo -e "Python: mypy static analysis..."
mypy --config-file=./misc/scripts/mypy.ini .
