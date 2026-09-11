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
##  This file tests the libvpx postproc example code. To add new tests to this
##  file, do the following:
##    1. Write a shell function (this is your test).
##    2. Add the function to postproc_tests (on a new line).
##
. $(dirname $0)/tools_common.sh

# Environment check: Make sure input is available:
#   $VP8_IVF_FILE and $VP9_IVF_FILE are required.
postproc_verify_environment() {
  if [ ! -e "${VP8_IVF_FILE}" ] || [ ! -e "${VP9_IVF_FILE}" ]; then
    echo "Libvpx test data must exist in LIBVPX_TEST_DATA_PATH."
    return 1
  fi
}

# Runs postproc using $1 as input file. $2 is the codec name, and is used
# solely to name the output file.
postproc() {
  local decoder="${LIBVPX_BIN_PATH}/postproc${VPX_TEST_EXE_SUFFIX}"
  local input_file="$1"
  local codec="$2"
  local output_file="${VPX_TEST_OUTPUT_DIR}/postproc_${codec}.raw"

  if [ ! -x "${decoder}" ]; then
    elog "${decoder} does not exist or is not executable."
    return 1
  fi

  eval "${VPX_TEST_PREFIX}" "${decoder}" "${input_file}" "${output_file}" \
      ${devnull} || return 1

  [ -e "${output_file}" ] || return 1
}

postproc_vp8() {
  if [ "$(vp8_decode_available)" = "yes" ]; then
    postproc "${VP8_IVF_FILE}" vp8 || return 1
  fi
}

postproc_vp9() {
  if [ "$(vpx_config_option_enabled CONFIG_VP9_POSTPROC)" = "yes" ]; then
    if [ "$(vp9_decode_available)" = "yes" ]; then
      postproc "${VP9_IVF_FILE}" vp9 || return 1
    fi
  fi
}

postproc_tests="postproc_vp8
                postproc_vp9"

run_tests postproc_verify_environment "${postproc_tests}"
