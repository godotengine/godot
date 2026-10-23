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
##  This file tests the libvpx set_maps example. To add new tests to this file,
##  do the following:
##    1. Write a shell function (this is your test).
##    2. Add the function to set_maps_tests (on a new line).
##
. $(dirname $0)/tools_common.sh

# Environment check: $YUV_RAW_INPUT is required, and set_maps must exist in
# $LIBVPX_BIN_PATH.
set_maps_verify_environment() {
  if [ ! -e "${YUV_RAW_INPUT}" ]; then
    echo "Libvpx test data must exist in LIBVPX_TEST_DATA_PATH."
    return 1
  fi
  if [ -z "$(vpx_tool_path set_maps)" ]; then
    elog "set_maps not found. It must exist in LIBVPX_BIN_PATH or its parent."
    return 1
  fi
}

# Runs set_maps using the codec specified by $1.
set_maps() {
  local encoder="$(vpx_tool_path set_maps)"
  local codec="$1"
  local output_file="${VPX_TEST_OUTPUT_DIR}/set_maps_${codec}.ivf"

  eval "${VPX_TEST_PREFIX}" "${encoder}" "${codec}" "${YUV_RAW_INPUT_WIDTH}" \
      "${YUV_RAW_INPUT_HEIGHT}" "${YUV_RAW_INPUT}" "${output_file}" \
      ${devnull} || return 1

  [ -e "${output_file}" ] || return 1
}

set_maps_vp8() {
  if [ "$(vp8_encode_available)" = "yes" ]; then
    set_maps vp8 || return 1
  fi
}

set_maps_vp9() {
  if [ "$(vp9_encode_available)" = "yes" ]; then
    set_maps vp9 || return 1
  fi
}

set_maps_tests="set_maps_vp8
                set_maps_vp9"

run_tests set_maps_verify_environment "${set_maps_tests}"
