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
##  This file tests the libvpx simple_decoder example code. To add new tests to
##  this file, do the following:
##    1. Write a shell function (this is your test).
##    2. Add the function to simple_decoder_tests (on a new line).
##
. $(dirname $0)/tools_common.sh

# Environment check: Make sure input is available:
#   $VP8_IVF_FILE and $VP9_IVF_FILE are required.
simple_decoder_verify_environment() {
  if [ ! -e "${VP8_IVF_FILE}" ] || [ ! -e "${VP9_IVF_FILE}" ]; then
    echo "Libvpx test data must exist in LIBVPX_TEST_DATA_PATH."
    return 1
  fi
}

# Runs simple_decoder using $1 as input file. $2 is the codec name, and is used
# solely to name the output file.
simple_decoder() {
  local decoder="${LIBVPX_BIN_PATH}/simple_decoder${VPX_TEST_EXE_SUFFIX}"
  local input_file="$1"
  local codec="$2"
  local output_file="${VPX_TEST_OUTPUT_DIR}/simple_decoder_${codec}.raw"

  if [ ! -x "${decoder}" ]; then
    elog "${decoder} does not exist or is not executable."
    return 1
  fi

  eval "${VPX_TEST_PREFIX}" "${decoder}" "${input_file}" "${output_file}" \
      ${devnull} || return 1

  [ -e "${output_file}" ] || return 1
}

simple_decoder_vp8() {
  if [ "$(vp8_decode_available)" = "yes" ]; then
    simple_decoder "${VP8_IVF_FILE}" vp8 || return 1
  fi
}

simple_decoder_vp9() {
  if [ "$(vp9_decode_available)" = "yes" ]; then
    simple_decoder "${VP9_IVF_FILE}" vp9 || return 1
  fi
}

simple_decoder_tests="simple_decoder_vp8
                      simple_decoder_vp9"

run_tests simple_decoder_verify_environment "${simple_decoder_tests}"
