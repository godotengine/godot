#!/bin/bash
set -eux
bash git/re2/kokoro/bazel.sh
exit $?
