#!/usr/bin/env bash
# Focused port checks; the compiler suite requires bin/tint_convert_cli (drivers/webgpu/tint_cli/build.sh).
set -euo pipefail
cd "$(dirname "$0")/../../.."

for tool in uv node; do
    command -v "$tool" >/dev/null || { echo "Missing required tool: $tool" >&2; exit 1; }
done

uv run --no-project python -m unittest discover -s drivers/webgpu/tests -p 'test_*.py' -v
node --test drivers/webgpu/tests/engine_boot.test.mjs
node drivers/webgpu/tests/preprocessing_tests/run_tests.mjs
