#!/bin/sh
##
##  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
##
##  Use of this source code is governed by a BSD-style license
##  that can be found in the LICENSE file in the root of the source
##  tree. An additional intellectual property rights grant can be found
##  in the file PATENTS.  All contributing project authors may
##  be found in the AUTHORS file in the root of the source tree.
##
##  This file runs all of the tests for the libvpx examples.
##
. $(dirname $0)/tools_common.sh

example_tests=$(ls $(dirname $0)/*.sh)

# List of script names to exclude.
exclude_list="examples stress tools_common"

# Filter out the scripts in $exclude_list.
for word in ${exclude_list}; do
  example_tests=$(filter_strings "${example_tests}" "${word}" exclude)
done

for test in ${example_tests}; do
  # Source each test script so that exporting variables can be avoided.
  VPX_TEST_NAME="$(basename ${test%.*})"
  . "${test}"
done
