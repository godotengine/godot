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
##  This file tests the libvpx decode_to_md5 example. To add new tests to this
##  file, do the following:
##    1. Write a shell function (this is your test).
##    2. Add the function to decode_to_md5_tests (on a new line).
##
. $(dirname $0)/tools_common.sh

# Environment check: Make sure input is available:
#   $VP8_IVF_FILE and $VP9_IVF_FILE are required.
decode_to_md5_verify_environment() {
  if [ ! -e "${VP8_IVF_FILE}" ] || [ ! -e "${VP9_IVF_FILE}" ]; then
    echo "Libvpx test data must exist in LIBVPX_TEST_DATA_PATH."
    return 1
  fi
}

# Runs decode_to_md5 on $1 and captures the md5 sum for the final frame. $2 is
# interpreted as codec name and used solely to name the output file. $3 is the
# expected md5 sum: It must match that of the final frame.
decode_to_md5() {
  local decoder="${LIBVPX_BIN_PATH}/decode_to_md5${VPX_TEST_EXE_SUFFIX}"
  local input_file="$1"
  local codec="$2"
  local expected_md5="$3"
  local output_file="${VPX_TEST_OUTPUT_DIR}/decode_to_md5_${codec}"

  if [ ! -x "${decoder}" ]; then
    elog "${decoder} does not exist or is not executable."
    return 1
  fi

  eval "${VPX_TEST_PREFIX}" "${decoder}" "${input_file}" "${output_file}" \
      ${devnull} || return 1

  [ -e "${output_file}" ] || return 1

  local md5_last_frame="$(tail -n1 "${output_file}" | awk '{print $1}')"
  local actual_md5="$(echo "${md5_last_frame}" | awk '{print $1}')"
  [ "${actual_md5}" = "${expected_md5}" ] || return 1
}

decode_to_md5_vp8() {
  # expected MD5 sum for the last frame.
  local expected_md5="56794d911b02190212bca92f88ad60c6"

  if [ "$(vp8_decode_available)" = "yes" ]; then
    decode_to_md5 "${VP8_IVF_FILE}" "vp8" "${expected_md5}"
  fi
}

decode_to_md5_vp9() {
  # expected MD5 sum for the last frame.
  local expected_md5="2952c0eae93f3dadd1aa84c50d3fd6d2"

  if [ "$(vp9_decode_available)" = "yes" ]; then
    decode_to_md5 "${VP9_IVF_FILE}" "vp9" "${expected_md5}"
  fi
}

decode_to_md5_tests="decode_to_md5_vp8
                     decode_to_md5_vp9"

run_tests decode_to_md5_verify_environment "${decode_to_md5_tests}"
