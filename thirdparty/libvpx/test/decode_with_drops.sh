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
##  This file tests the libvpx decode_with_drops example. To add new tests to
##  this file, do the following:
##    1. Write a shell function (this is your test).
##    2. Add the function to decode_with_drops_tests (on a new line).
##
. $(dirname $0)/tools_common.sh

# Environment check: Make sure input is available:
#   $VP8_IVF_FILE and $VP9_IVF_FILE are required.
decode_with_drops_verify_environment() {
  if [ ! -e "${VP8_IVF_FILE}" ] || [ ! -e "${VP9_IVF_FILE}" ]; then
    echo "Libvpx test data must exist in LIBVPX_TEST_DATA_PATH."
    return 1
  fi
}

# Runs decode_with_drops on $1, $2 is interpreted as codec name and used solely
# to name the output file. $3 is the drop mode, and is passed directly to
# decode_with_drops.
decode_with_drops() {
  local decoder="${LIBVPX_BIN_PATH}/decode_with_drops${VPX_TEST_EXE_SUFFIX}"
  local input_file="$1"
  local codec="$2"
  local output_file="${VPX_TEST_OUTPUT_DIR}/decode_with_drops_${codec}"
  local drop_mode="$3"

  if [ ! -x "${decoder}" ]; then
    elog "${decoder} does not exist or is not executable."
    return 1
  fi

  eval "${VPX_TEST_PREFIX}" "${decoder}" "${input_file}" "${output_file}" \
      "${drop_mode}" ${devnull} || return 1

  [ -e "${output_file}" ] || return 1
}

# Decodes $VP8_IVF_FILE while dropping frames, twice: once in sequence mode,
# and once in pattern mode.
# Note: This test assumes that $VP8_IVF_FILE has exactly 29 frames, and could
# break if the file is modified.
decode_with_drops_vp8() {
  if [ "$(vp8_decode_available)" = "yes" ]; then
    # Test sequence mode: Drop frames 2-28.
    decode_with_drops "${VP8_IVF_FILE}" "vp8" "2-28" || return 1

    # Test pattern mode: Drop 3 of every 4 frames.
    decode_with_drops "${VP8_IVF_FILE}" "vp8" "3/4" || return 1
  fi
}

# Decodes $VP9_IVF_FILE while dropping frames, twice: once in sequence mode,
# and once in pattern mode.
# Note: This test assumes that $VP9_IVF_FILE has exactly 20 frames, and could
# break if the file is modified.
decode_with_drops_vp9() {
  if [ "$(vp9_decode_available)" = "yes" ]; then
    # Test sequence mode: Drop frames 2-28.
    decode_with_drops "${VP9_IVF_FILE}" "vp9" "2-19" || return 1

    # Test pattern mode: Drop 3 of every 4 frames.
    decode_with_drops "${VP9_IVF_FILE}" "vp9" "3/4" || return 1
  fi
}

decode_with_drops_tests="decode_with_drops_vp8
                         decode_with_drops_vp9"

run_tests decode_with_drops_verify_environment "${decode_with_drops_tests}"
