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
##  This file tests the libvpx vp8_multi_resolution_encoder example. To add new
##  tests to this file, do the following:
##    1. Write a shell function (this is your test).
##    2. Add the function to vp8_mre_tests (on a new line).
##
. $(dirname $0)/tools_common.sh

# Environment check: $YUV_RAW_INPUT is required.
vp8_multi_resolution_encoder_verify_environment() {
  if [ "$(vpx_config_option_enabled CONFIG_MULTI_RES_ENCODING)" = "yes" ]; then
    if [ ! -e "${YUV_RAW_INPUT}" ]; then
      elog "Libvpx test data must exist in LIBVPX_TEST_DATA_PATH."
      return 1
    fi
    local app="vp8_multi_resolution_encoder"
    if [ -z "$(vpx_tool_path "${app}")" ]; then
      elog "${app} not found. It must exist in LIBVPX_BIN_PATH or its parent."
      return 1
    fi
  fi
}

# Runs vp8_multi_resolution_encoder. Simply forwards all arguments to
# vp8_multi_resolution_encoder after building path to the executable.
vp8_mre() {
  local encoder="$(vpx_tool_path vp8_multi_resolution_encoder)"
  if [ ! -x "${encoder}" ]; then
    elog "${encoder} does not exist or is not executable."
    return 1
  fi

  eval "${VPX_TEST_PREFIX}" "${encoder}" "$@" ${devnull}
}

vp8_multi_resolution_encoder_three_formats() {
  local output_files="${VPX_TEST_OUTPUT_DIR}/vp8_mre_0.ivf
                      ${VPX_TEST_OUTPUT_DIR}/vp8_mre_1.ivf
                      ${VPX_TEST_OUTPUT_DIR}/vp8_mre_2.ivf"
  local layer_bitrates="150 80 50"
  local keyframe_insert="200"
  local temporal_layers="3 3 3"
  local framerate="30"

  if [ "$(vpx_config_option_enabled CONFIG_MULTI_RES_ENCODING)" = "yes" ]; then
    if [ "$(vp8_encode_available)" = "yes" ]; then
      # Param order:
      #  Input width
      #  Input height
      #  Framerate
      #  Input file path
      #  Output file names
      #  Layer bitrates
      #  Temporal layers
      #  Keyframe insert
      #  Output PSNR
      vp8_mre "${YUV_RAW_INPUT_WIDTH}" \
        "${YUV_RAW_INPUT_HEIGHT}" \
        "${framerate}" \
        "${YUV_RAW_INPUT}" \
        ${output_files} \
        ${layer_bitrates} \
        ${temporal_layers} \
        "${keyframe_insert}" \
        0 || return 1

      for output_file in ${output_files}; do
        if [ ! -e "${output_file}" ]; then
          elog "Missing output file: ${output_file}"
          return 1
        fi
      done
    fi
  fi
}

vp8_mre_tests="vp8_multi_resolution_encoder_three_formats"
run_tests vp8_multi_resolution_encoder_verify_environment "${vp8_mre_tests}"
