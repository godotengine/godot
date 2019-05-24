#!/bin/bash
set -eux

cd git/re2

bazel clean
bazel build --compilation_mode=dbg -- //...
bazel test  --compilation_mode=dbg --test_output=errors -- //... \
  -//:dfa_test \
  -//:exhaustive1_test \
  -//:exhaustive2_test \
  -//:exhaustive3_test \
  -//:exhaustive_test \
  -//:random_test

bazel clean
bazel build --compilation_mode=opt -- //...
bazel test  --compilation_mode=opt --test_output=errors -- //... \
  -//:dfa_test \
  -//:exhaustive1_test \
  -//:exhaustive2_test \
  -//:exhaustive3_test \
  -//:exhaustive_test \
  -//:random_test

exit 0
