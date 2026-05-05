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
##  This file tests vpxdec. To add new tests to this file, do the following:
##    1. Write a shell function (this is your test).
##    2. Add the function to vpxdec_tests (on a new line).
##
. $(dirname $0)/tools_common.sh

# Environment check: Make sure input is available.
vpxdec_verify_environment() {
  if [ ! -e "${VP8_IVF_FILE}" ] || [ ! -e "${VP9_WEBM_FILE}" ] || \
    [ ! -e "${VP9_FPM_WEBM_FILE}" ] || \
    [ ! -e "${VP9_LT_50_FRAMES_WEBM_FILE}" ] || \
    [ ! -e "${VP9_RAW_FILE}" ]; then
    elog "Libvpx test data must exist in LIBVPX_TEST_DATA_PATH."
    return 1
  fi
  if [ -z "$(vpx_tool_path vpxdec)" ]; then
    elog "vpxdec not found. It must exist in LIBVPX_BIN_PATH or its parent."
    return 1
  fi
}

# Wrapper function for running vpxdec with pipe input. Requires that
# LIBVPX_BIN_PATH points to the directory containing vpxdec. $1 is used as the
# input file path and shifted away. All remaining parameters are passed through
# to vpxdec.
vpxdec_pipe() {
  local decoder="$(vpx_tool_path vpxdec)"
  local input="$1"
  shift
  cat "${input}" | eval "${VPX_TEST_PREFIX}" "${decoder}" - "$@" ${devnull}
}

# Wrapper function for running vpxdec. Requires that LIBVPX_BIN_PATH points to
# the directory containing vpxdec. $1 one is used as the input file path and
# shifted away. All remaining parameters are passed through to vpxdec.
vpxdec() {
  local decoder="$(vpx_tool_path vpxdec)"
  local input="$1"
  shift
  eval "${VPX_TEST_PREFIX}" "${decoder}" "$input" "$@" ${devnull}
}

vpxdec_can_decode_vp8() {
  if [ "$(vp8_decode_available)" = "yes" ]; then
    echo yes
  fi
}

vpxdec_can_decode_vp9() {
  if [ "$(vp9_decode_available)" = "yes" ]; then
    echo yes
  fi
}

vpxdec_vp8_ivf() {
  if [ "$(vpxdec_can_decode_vp8)" = "yes" ]; then
    vpxdec "${VP8_IVF_FILE}" --summary --noblit
  fi
}

vpxdec_vp8_ivf_pipe_input() {
  if [ "$(vpxdec_can_decode_vp8)" = "yes" ]; then
    vpxdec_pipe "${VP8_IVF_FILE}" --summary --noblit
  fi
}

vpxdec_vp9_webm() {
  if [ "$(vpxdec_can_decode_vp9)" = "yes" ] && \
     [ "$(webm_io_available)" = "yes" ]; then
    vpxdec "${VP9_WEBM_FILE}" --summary --noblit
  fi
}

vpxdec_vp9_webm_frame_parallel() {
  if [ "$(vpxdec_can_decode_vp9)" = "yes" ] && \
     [ "$(webm_io_available)" = "yes" ]; then
    for threads in 2 3 4 5 6 7 8; do
      vpxdec "${VP9_FPM_WEBM_FILE}" --summary --noblit --threads=$threads \
        --frame-parallel || return 1
    done
  fi
}

vpxdec_vp9_webm_less_than_50_frames() {
  # ensure that reaching eof in webm_guess_framerate doesn't result in invalid
  # frames in actual webm_read_frame calls.
  if [ "$(vpxdec_can_decode_vp9)" = "yes" ] && \
     [ "$(webm_io_available)" = "yes" ]; then
    local decoder="$(vpx_tool_path vpxdec)"
    local expected=10
    local num_frames=$(${VPX_TEST_PREFIX} "${decoder}" \
      "${VP9_LT_50_FRAMES_WEBM_FILE}" --summary --noblit 2>&1 \
      | awk '/^[0-9]+ decoded frames/ { print $1 }')
    if [ "$num_frames" -ne "$expected" ]; then
      elog "Output frames ($num_frames) != expected ($expected)"
      return 1
    fi
  fi
}

# Ensures VP9_RAW_FILE correctly produces 1 frame instead of causing a hang.
vpxdec_vp9_raw_file() {
  # Ensure a raw file properly reports eof and doesn't cause a hang.
  if [ "$(vpxdec_can_decode_vp9)" = "yes" ]; then
    local decoder="$(vpx_tool_path vpxdec)"
    local expected=1
    [ -x /usr/bin/timeout ] && local TIMEOUT="/usr/bin/timeout 30s"
    local num_frames=$(${TIMEOUT} ${VPX_TEST_PREFIX} "${decoder}" \
      "${VP9_RAW_FILE}" --summary --noblit 2>&1 \
      | awk '/^[0-9]+ decoded frames/ { print $1 }')
    if [ -z "$num_frames" ] || [ "$num_frames" -ne "$expected" ]; then
      elog "Output frames ($num_frames) != expected ($expected)"
      return 1
    fi
  fi
}

vpxdec_tests="vpxdec_vp8_ivf
              vpxdec_vp8_ivf_pipe_input
              vpxdec_vp9_webm
              vpxdec_vp9_webm_frame_parallel
              vpxdec_vp9_webm_less_than_50_frames
              vpxdec_vp9_raw_file"

run_tests vpxdec_verify_environment "${vpxdec_tests}"
