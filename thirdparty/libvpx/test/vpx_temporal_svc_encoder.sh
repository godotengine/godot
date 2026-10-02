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
##  This file tests the libvpx vpx_temporal_svc_encoder example. To add new
##  tests to this file, do the following:
##    1. Write a shell function (this is your test).
##    2. Add the function to vpx_tsvc_encoder_tests (on a new line).
##
. $(dirname $0)/tools_common.sh

# Environment check: $YUV_RAW_INPUT is required.
vpx_tsvc_encoder_verify_environment() {
  if [ ! -e "${YUV_RAW_INPUT}" ]; then
    echo "Libvpx test data must exist in LIBVPX_TEST_DATA_PATH."
    return 1
  fi
  if [ "$(vpx_config_option_enabled CONFIG_TEMPORAL_DENOISING)" != "yes" ]; then
    elog "Warning: Temporal denoising is disabled! Spatial denoising will be " \
      "used instead, which is probably not what you want for this test."
  fi
}

# Runs vpx_temporal_svc_encoder using the codec specified by $1 and output file
# name by $2. Additional positional parameters are passed directly to
# vpx_temporal_svc_encoder.
vpx_tsvc_encoder() {
  local encoder="${LIBVPX_BIN_PATH}/vpx_temporal_svc_encoder"
  encoder="${encoder}${VPX_TEST_EXE_SUFFIX}"
  local codec="$1"
  local output_file_base="$2"
  local output_file="${VPX_TEST_OUTPUT_DIR}/${output_file_base}"
  local timebase_num="1"
  local timebase_den="1000"
  local timebase_den_y4m="30"
  local speed="6"
  local frame_drop_thresh="30"
  local max_threads="4"
  local error_resilient="1"

  shift 2

  if [ ! -x "${encoder}" ]; then
    elog "${encoder} does not exist or is not executable."
    return 1
  fi

  # TODO(tomfinegan): Verify file output for all thread runs.
  for threads in $(seq $max_threads); do
    if [ "$(vpx_config_option_enabled CONFIG_VP9_HIGHBITDEPTH)" != "yes" ]; then
      eval "${VPX_TEST_PREFIX}" "${encoder}" "${YUV_RAW_INPUT}" \
        "${output_file}" "${codec}" "${YUV_RAW_INPUT_WIDTH}" \
        "${YUV_RAW_INPUT_HEIGHT}" "${timebase_num}" "${timebase_den}" \
        "${speed}" "${frame_drop_thresh}" "${error_resilient}" "${threads}" \
        "$@" ${devnull} || return 1
      # Test for y4m input.
      eval "${VPX_TEST_PREFIX}" "${encoder}" "${Y4M_720P_INPUT}" \
        "${output_file}" "${codec}" "${Y4M_720P_INPUT_WIDTH}" \
        "${Y4M_720P_INPUT_HEIGHT}" "${timebase_num}" "${timebase_den_y4m}" \
        "${speed}" "${frame_drop_thresh}" "${error_resilient}" "${threads}" \
        "$@" ${devnull} || return 1
    else
      eval "${VPX_TEST_PREFIX}" "${encoder}" "${YUV_RAW_INPUT}" \
        "${output_file}" "${codec}" "${YUV_RAW_INPUT_WIDTH}" \
        "${YUV_RAW_INPUT_HEIGHT}" "${timebase_num}" "${timebase_den}" \
        "${speed}" "${frame_drop_thresh}" "${error_resilient}" "${threads}" \
        "$@" "8" ${devnull} || return 1
    fi
  done
}

# Confirms that all expected output files exist given the output file name
# passed to vpx_temporal_svc_encoder.
# The file name passed to vpx_temporal_svc_encoder is joined with the stream
# number and the extension .ivf to produce per stream output files.  Here $1 is
# file name, and $2 is expected number of files.
files_exist() {
  local file_name="${VPX_TEST_OUTPUT_DIR}/$1"
  local num_files="$(($2 - 1))"
  for stream_num in $(seq 0 ${num_files}); do
    [ -e "${file_name}_${stream_num}.ivf" ] || return 1
  done
}

# Run vpx_temporal_svc_encoder in all supported modes for vp8 and vp9.

vpx_tsvc_encoder_vp8_mode_0() {
  if [ "$(vp8_encode_available)" = "yes" ]; then
    local output_basename="vpx_tsvc_encoder_vp8_mode_0"
    vpx_tsvc_encoder vp8 "${output_basename}" 0 200 || return 1
    # Mode 0 produces 1 stream
    files_exist "${output_basename}" 1 || return 1
  fi
}

vpx_tsvc_encoder_vp8_mode_1() {
  if [ "$(vp8_encode_available)" = "yes" ]; then
    local output_basename="vpx_tsvc_encoder_vp8_mode_1"
    vpx_tsvc_encoder vp8 "${output_basename}" 1 200 400 || return 1
    # Mode 1 produces 2 streams
    files_exist "${output_basename}" 2 || return 1
  fi
}

vpx_tsvc_encoder_vp8_mode_2() {
  if [ "$(vp8_encode_available)" = "yes" ]; then
    local output_basename="vpx_tsvc_encoder_vp8_mode_2"
    vpx_tsvc_encoder vp8 "${output_basename}" 2 200 400 || return 1
    # Mode 2 produces 2 streams
    files_exist "${output_basename}" 2 || return 1
  fi
}

vpx_tsvc_encoder_vp8_mode_3() {
  if [ "$(vp8_encode_available)" = "yes" ]; then
    local output_basename="vpx_tsvc_encoder_vp8_mode_3"
    vpx_tsvc_encoder vp8 "${output_basename}" 3 200 400 600 || return 1
    # Mode 3 produces 3 streams
    files_exist "${output_basename}" 3 || return 1
  fi
}

vpx_tsvc_encoder_vp8_mode_4() {
  if [ "$(vp8_encode_available)" = "yes" ]; then
    local output_basename="vpx_tsvc_encoder_vp8_mode_4"
    vpx_tsvc_encoder vp8 "${output_basename}" 4 200 400 600 || return 1
    # Mode 4 produces 3 streams
    files_exist "${output_basename}" 3 || return 1
  fi
}

vpx_tsvc_encoder_vp8_mode_5() {
  if [ "$(vp8_encode_available)" = "yes" ]; then
    local output_basename="vpx_tsvc_encoder_vp8_mode_5"
    vpx_tsvc_encoder vp8 "${output_basename}" 5 200 400 600 || return 1
    # Mode 5 produces 3 streams
    files_exist "${output_basename}" 3 || return 1
  fi
}

vpx_tsvc_encoder_vp8_mode_6() {
  if [ "$(vp8_encode_available)" = "yes" ]; then
    local output_basename="vpx_tsvc_encoder_vp8_mode_6"
    vpx_tsvc_encoder vp8 "${output_basename}" 6 200 400 600 || return 1
    # Mode 6 produces 3 streams
    files_exist "${output_basename}" 3 || return 1
  fi
}

vpx_tsvc_encoder_vp8_mode_7() {
  if [ "$(vp8_encode_available)" = "yes" ]; then
    local output_basename="vpx_tsvc_encoder_vp8_mode_7"
    vpx_tsvc_encoder vp8 "${output_basename}" 7 200 400 600 800 1000 || return 1
    # Mode 7 produces 5 streams
    files_exist "${output_basename}" 5 || return 1
  fi
}

vpx_tsvc_encoder_vp8_mode_8() {
  if [ "$(vp8_encode_available)" = "yes" ]; then
    local output_basename="vpx_tsvc_encoder_vp8_mode_8"
    vpx_tsvc_encoder vp8 "${output_basename}" 8 200 400 || return 1
    # Mode 8 produces 2 streams
    files_exist "${output_basename}" 2 || return 1
  fi
}

vpx_tsvc_encoder_vp8_mode_9() {
  if [ "$(vp8_encode_available)" = "yes" ]; then
    local output_basename="vpx_tsvc_encoder_vp8_mode_9"
    vpx_tsvc_encoder vp8 "${output_basename}" 9 200 400 600 || return 1
    # Mode 9 produces 3 streams
    files_exist "${output_basename}" 3 || return 1
  fi
}

vpx_tsvc_encoder_vp8_mode_10() {
  if [ "$(vp8_encode_available)" = "yes" ]; then
    local output_basename="vpx_tsvc_encoder_vp8_mode_10"
    vpx_tsvc_encoder vp8 "${output_basename}" 10 200 400 600 || return 1
    # Mode 10 produces 3 streams
    files_exist "${output_basename}" 3 || return 1
  fi
}

vpx_tsvc_encoder_vp8_mode_11() {
  if [ "$(vp8_encode_available)" = "yes" ]; then
    local output_basename="vpx_tsvc_encoder_vp8_mode_11"
    vpx_tsvc_encoder vp8 "${output_basename}" 11 200 400 600 || return 1
    # Mode 11 produces 3 streams
    files_exist "${output_basename}" 3 || return 1
  fi
}

vpx_tsvc_encoder_vp9_mode_0() {
  if [ "$(vp9_encode_available)" = "yes" ]; then
    local output_basename="vpx_tsvc_encoder_vp9_mode_0"
    vpx_tsvc_encoder vp9 "${output_basename}" 0 200 || return 1
    # Mode 0 produces 1 stream
    files_exist "${output_basename}" 1 || return 1
  fi
}

vpx_tsvc_encoder_vp9_mode_1() {
  if [ "$(vp9_encode_available)" = "yes" ]; then
    local output_basename="vpx_tsvc_encoder_vp9_mode_1"
    vpx_tsvc_encoder vp9 "${output_basename}" 1 200 400 || return 1
    # Mode 1 produces 2 streams
    files_exist "${output_basename}" 2 || return 1
  fi
}

vpx_tsvc_encoder_vp9_mode_2() {
  if [ "$(vp9_encode_available)" = "yes" ]; then
    local output_basename="vpx_tsvc_encoder_vp9_mode_2"
    vpx_tsvc_encoder vp9 "${output_basename}" 2 200 400 || return 1
    # Mode 2 produces 2 streams
    files_exist "${output_basename}" 2 || return 1
  fi
}

vpx_tsvc_encoder_vp9_mode_3() {
  if [ "$(vp9_encode_available)" = "yes" ]; then
    local output_basename="vpx_tsvc_encoder_vp9_mode_3"
    vpx_tsvc_encoder vp9 "${output_basename}" 3 200 400 600 || return 1
    # Mode 3 produces 3 streams
    files_exist "${output_basename}" 3 || return 1
  fi
}

vpx_tsvc_encoder_vp9_mode_4() {
  if [ "$(vp9_encode_available)" = "yes" ]; then
    local output_basename="vpx_tsvc_encoder_vp9_mode_4"
    vpx_tsvc_encoder vp9 "${output_basename}" 4 200 400 600 || return 1
    # Mode 4 produces 3 streams
    files_exist "${output_basename}" 3 || return 1
  fi
}

vpx_tsvc_encoder_vp9_mode_5() {
  if [ "$(vp9_encode_available)" = "yes" ]; then
    local output_basename="vpx_tsvc_encoder_vp9_mode_5"
    vpx_tsvc_encoder vp9 "${output_basename}" 5 200 400 600 || return 1
    # Mode 5 produces 3 streams
    files_exist "${output_basename}" 3 || return 1
  fi
}

vpx_tsvc_encoder_vp9_mode_6() {
  if [ "$(vp9_encode_available)" = "yes" ]; then
    local output_basename="vpx_tsvc_encoder_vp9_mode_6"
    vpx_tsvc_encoder vp9 "${output_basename}" 6 200 400 600 || return 1
    # Mode 6 produces 3 streams
    files_exist "${output_basename}" 3 || return 1
  fi
}

vpx_tsvc_encoder_vp9_mode_7() {
  if [ "$(vp9_encode_available)" = "yes" ]; then
    local output_basename="vpx_tsvc_encoder_vp9_mode_7"
    vpx_tsvc_encoder vp9 "${output_basename}" 7 200 400 600 800 1000 || return 1
    # Mode 7 produces 5 streams
    files_exist "${output_basename}" 5 || return 1
  fi
}

vpx_tsvc_encoder_vp9_mode_8() {
  if [ "$(vp9_encode_available)" = "yes" ]; then
    local output_basename="vpx_tsvc_encoder_vp9_mode_8"
    vpx_tsvc_encoder vp9 "${output_basename}" 8 200 400 || return 1
    # Mode 8 produces 2 streams
    files_exist "${output_basename}" 2 || return 1
  fi
}

vpx_tsvc_encoder_vp9_mode_9() {
  if [ "$(vp9_encode_available)" = "yes" ]; then
    local output_basename="vpx_tsvc_encoder_vp9_mode_9"
    vpx_tsvc_encoder vp9 "${output_basename}" 9 200 400 600 || return 1
    # Mode 9 produces 3 streams
    files_exist "${output_basename}" 3 || return 1
  fi
}

vpx_tsvc_encoder_vp9_mode_10() {
  if [ "$(vp9_encode_available)" = "yes" ]; then
    local output_basename="vpx_tsvc_encoder_vp9_mode_10"
    vpx_tsvc_encoder vp9 "${output_basename}" 10 200 400 600 || return 1
    # Mode 10 produces 3 streams
    files_exist "${output_basename}" 3 || return 1
  fi
}

vpx_tsvc_encoder_vp9_mode_11() {
  if [ "$(vp9_encode_available)" = "yes" ]; then
    local output_basename="vpx_tsvc_encoder_vp9_mode_11"
    vpx_tsvc_encoder vp9 "${output_basename}" 11 200 400 600 || return 1
    # Mode 11 produces 3 streams
    files_exist "${output_basename}" 3 || return 1
  fi
}

vpx_tsvc_encoder_tests="vpx_tsvc_encoder_vp8_mode_0
                        vpx_tsvc_encoder_vp8_mode_1
                        vpx_tsvc_encoder_vp8_mode_2
                        vpx_tsvc_encoder_vp8_mode_3
                        vpx_tsvc_encoder_vp8_mode_4
                        vpx_tsvc_encoder_vp8_mode_5
                        vpx_tsvc_encoder_vp8_mode_6
                        vpx_tsvc_encoder_vp8_mode_7
                        vpx_tsvc_encoder_vp8_mode_8
                        vpx_tsvc_encoder_vp8_mode_9
                        vpx_tsvc_encoder_vp8_mode_10
                        vpx_tsvc_encoder_vp8_mode_11
                        vpx_tsvc_encoder_vp9_mode_0
                        vpx_tsvc_encoder_vp9_mode_1
                        vpx_tsvc_encoder_vp9_mode_2
                        vpx_tsvc_encoder_vp9_mode_3
                        vpx_tsvc_encoder_vp9_mode_4
                        vpx_tsvc_encoder_vp9_mode_5
                        vpx_tsvc_encoder_vp9_mode_6
                        vpx_tsvc_encoder_vp9_mode_7
                        vpx_tsvc_encoder_vp9_mode_8
                        vpx_tsvc_encoder_vp9_mode_9
                        vpx_tsvc_encoder_vp9_mode_10
                        vpx_tsvc_encoder_vp9_mode_11"

run_tests vpx_tsvc_encoder_verify_environment "${vpx_tsvc_encoder_tests}"
