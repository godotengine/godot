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
##  This file tests the libvpx twopass_encoder example. To add new tests to this
##  file, do the following:
##    1. Write a shell function (this is your test).
##    2. Add the function to twopass_encoder_tests (on a new line).
##
. $(dirname $0)/tools_common.sh

# Environment check: $YUV_RAW_INPUT is required.
twopass_encoder_verify_environment() {
  if [ ! -e "${YUV_RAW_INPUT}" ]; then
    echo "Libvpx test data must exist in LIBVPX_TEST_DATA_PATH."
    return 1
  fi
}

# Runs twopass_encoder using the codec specified by $1 with a frame limit of
# 100.
twopass_encoder() {
  local encoder="${LIBVPX_BIN_PATH}/twopass_encoder${VPX_TEST_EXE_SUFFIX}"
  local codec="$1"
  local output_file="${VPX_TEST_OUTPUT_DIR}/twopass_encoder_${codec}.ivf"

  if [ ! -x "${encoder}" ]; then
    elog "${encoder} does not exist or is not executable."
    return 1
  fi

  eval "${VPX_TEST_PREFIX}" "${encoder}" "${codec}" "${YUV_RAW_INPUT_WIDTH}" \
      "${YUV_RAW_INPUT_HEIGHT}" "${YUV_RAW_INPUT}" "${output_file}" 100 \
      ${devnull} || return 1

  [ -e "${output_file}" ] || return 1
}

twopass_encoder_vp8() {
  if [ "$(vp8_encode_available)" = "yes" ]; then
    twopass_encoder vp8 || return 1
  fi
}

twopass_encoder_vp9() {
  if [ "$(vp9_encode_available)" = "yes" ]; then
    twopass_encoder vp9 || return 1
  fi
}


if [ "$(vpx_config_option_enabled CONFIG_REALTIME_ONLY)" != "yes" ]; then
  twopass_encoder_tests="twopass_encoder_vp8
                         twopass_encoder_vp9"

  run_tests twopass_encoder_verify_environment "${twopass_encoder_tests}"
fi
