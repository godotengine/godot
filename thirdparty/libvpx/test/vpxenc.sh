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
##  This file tests vpxenc using hantro_collage_w352h288.yuv as input. To add
##  new tests to this file, do the following:
##    1. Write a shell function (this is your test).
##    2. Add the function to vpxenc_tests (on a new line).
##
. $(dirname $0)/tools_common.sh

readonly TEST_FRAMES=10

# Environment check: Make sure input is available.
vpxenc_verify_environment() {
  if [ ! -e "${YUV_RAW_INPUT}" ]; then
    elog "The file ${YUV_RAW_INPUT##*/} must exist in LIBVPX_TEST_DATA_PATH."
    return 1
  fi
  if [ "$(vpxenc_can_encode_vp9)" = "yes" ]; then
    if [ ! -e "${Y4M_NOSQ_PAR_INPUT}" ]; then
      elog "The file ${Y4M_NOSQ_PAR_INPUT##*/} must exist in"
      elog "LIBVPX_TEST_DATA_PATH."
      return 1
    fi
  fi
  if [ -z "$(vpx_tool_path vpxenc)" ]; then
    elog "vpxenc not found. It must exist in LIBVPX_BIN_PATH or its parent."
    return 1
  fi
}

vpxenc_can_encode_vp8() {
  if [ "$(vp8_encode_available)" = "yes" ]; then
    echo yes
  fi
}

vpxenc_can_encode_vp9() {
  if [ "$(vp9_encode_available)" = "yes" ]; then
    echo yes
  fi
}

# Echo vpxenc command line parameters allowing use of
# hantro_collage_w352h288.yuv as input.
yuv_input_hantro_collage() {
  echo ""${YUV_RAW_INPUT}"
       --width="${YUV_RAW_INPUT_WIDTH}"
       --height="${YUV_RAW_INPUT_HEIGHT}""
}

y4m_input_non_square_par() {
  echo ""${Y4M_NOSQ_PAR_INPUT}""
}

y4m_input_720p() {
  echo ""${Y4M_720P_INPUT}""
}

# Echo default vpxenc real time encoding params. $1 is the codec, which defaults
# to vp8 if unspecified.
vpxenc_rt_params() {
  local codec="${1:-vp8}"
  echo "--codec=${codec}
    --buf-initial-sz=500
    --buf-optimal-sz=600
    --buf-sz=1000
    --cpu-used=-6
    --end-usage=cbr
    --error-resilient=1
    --kf-max-dist=90000
    --lag-in-frames=0
    --max-intra-rate=300
    --max-q=56
    --min-q=2
    --noise-sensitivity=0
    --overshoot-pct=50
    --passes=1
    --profile=0
    --resize-allowed=0
    --rt
    --static-thresh=0
    --undershoot-pct=50"
}

# Forces --passes to 1 with CONFIG_REALTIME_ONLY.
vpxenc_passes_param() {
  if [ "$(vpx_config_option_enabled CONFIG_REALTIME_ONLY)" = "yes" ]; then
    echo "--passes=1"
  else
    echo "--passes=2"
  fi
}

# Wrapper function for running vpxenc with pipe input. Requires that
# LIBVPX_BIN_PATH points to the directory containing vpxenc. $1 is used as the
# input file path and shifted away. All remaining parameters are passed through
# to vpxenc.
vpxenc_pipe() {
  local encoder="$(vpx_tool_path vpxenc)"
  local input="$1"
  shift
  cat "${input}" | eval "${VPX_TEST_PREFIX}" "${encoder}" - \
    --test-decode=fatal \
    "$@" ${devnull}
}

# Wrapper function for running vpxenc. Requires that LIBVPX_BIN_PATH points to
# the directory containing vpxenc. $1 one is used as the input file path and
# shifted away. All remaining parameters are passed through to vpxenc.
vpxenc() {
  local encoder="$(vpx_tool_path vpxenc)"
  local input="$1"
  shift
  eval "${VPX_TEST_PREFIX}" "${encoder}" "${input}" \
    --test-decode=fatal \
    "$@" ${devnull}
}

vpxenc_vp8_ivf() {
  if [ "$(vpxenc_can_encode_vp8)" = "yes" ]; then
    local output="${VPX_TEST_OUTPUT_DIR}/vp8.ivf"
    vpxenc $(yuv_input_hantro_collage) \
      --codec=vp8 \
      --limit="${TEST_FRAMES}" \
      --ivf \
      --output="${output}" || return 1

    if [ ! -e "${output}" ]; then
      elog "Output file does not exist."
      return 1
    fi
  fi
}

vpxenc_vp8_webm() {
  if [ "$(vpxenc_can_encode_vp8)" = "yes" ] && \
     [ "$(webm_io_available)" = "yes" ]; then
    local output="${VPX_TEST_OUTPUT_DIR}/vp8.webm"
    vpxenc $(yuv_input_hantro_collage) \
      --codec=vp8 \
      --limit="${TEST_FRAMES}" \
      --output="${output}" || return 1

    if [ ! -e "${output}" ]; then
      elog "Output file does not exist."
      return 1
    fi
  fi
}

vpxenc_vp8_webm_rt() {
  if [ "$(vpxenc_can_encode_vp8)" = "yes" ] && \
     [ "$(webm_io_available)" = "yes" ]; then
    local output="${VPX_TEST_OUTPUT_DIR}/vp8_rt.webm"
    vpxenc $(yuv_input_hantro_collage) \
      $(vpxenc_rt_params vp8) \
      --output="${output}" || return 1

    if [ ! -e "${output}" ]; then
      elog "Output file does not exist."
      return 1
    fi
  fi
}

vpxenc_vp8_webm_2pass() {
  if [ "$(vpxenc_can_encode_vp8)" = "yes" ] && \
     [ "$(webm_io_available)" = "yes" ]; then
    local output="${VPX_TEST_OUTPUT_DIR}/vp8.webm"
    vpxenc $(yuv_input_hantro_collage) \
      --codec=vp8 \
      --limit="${TEST_FRAMES}" \
      --output="${output}" \
      --passes=2 || return 1

    if [ ! -e "${output}" ]; then
      elog "Output file does not exist."
      return 1
    fi
  fi
}

vpxenc_vp8_webm_lag10_frames20() {
  if [ "$(vpxenc_can_encode_vp8)" = "yes" ] && \
     [ "$(webm_io_available)" = "yes" ]; then
    local lag_total_frames=20
    local lag_frames=10
    local output="${VPX_TEST_OUTPUT_DIR}/vp8_lag10_frames20.webm"
    vpxenc $(yuv_input_hantro_collage) \
      --codec=vp8 \
      --limit="${lag_total_frames}" \
      --lag-in-frames="${lag_frames}" \
      --output="${output}" \
      --auto-alt-ref=1 \
      --passes=2 || return 1

    if [ ! -e "${output}" ]; then
      elog "Output file does not exist."
      return 1
    fi
  fi
}

vpxenc_vp8_ivf_piped_input() {
  if [ "$(vpxenc_can_encode_vp8)" = "yes" ]; then
    local output="${VPX_TEST_OUTPUT_DIR}/vp8_piped_input.ivf"
    vpxenc_pipe $(yuv_input_hantro_collage) \
      --codec=vp8 \
      --limit="${TEST_FRAMES}" \
      --ivf \
      --output="${output}" || return 1

    if [ ! -e "${output}" ]; then
      elog "Output file does not exist."
      return 1
    fi
  fi
}

vpxenc_vp9_ivf() {
  if [ "$(vpxenc_can_encode_vp9)" = "yes" ]; then
    local output="${VPX_TEST_OUTPUT_DIR}/vp9.ivf"
    local passes=$(vpxenc_passes_param)
    vpxenc $(yuv_input_hantro_collage) \
      --codec=vp9 \
      --limit="${TEST_FRAMES}" \
      "${passes}" \
      --ivf \
      --output="${output}" || return 1

    if [ ! -e "${output}" ]; then
      elog "Output file does not exist."
      return 1
    fi
  fi
}

vpxenc_vp9_webm() {
  if [ "$(vpxenc_can_encode_vp9)" = "yes" ] && \
     [ "$(webm_io_available)" = "yes" ]; then
    local output="${VPX_TEST_OUTPUT_DIR}/vp9.webm"
    local passes=$(vpxenc_passes_param)
    vpxenc $(yuv_input_hantro_collage) \
      --codec=vp9 \
      --limit="${TEST_FRAMES}" \
      "${passes}" \
      --output="${output}" || return 1

    if [ ! -e "${output}" ]; then
      elog "Output file does not exist."
      return 1
    fi
  fi
}

vpxenc_vp9_webm_rt() {
  if [ "$(vpxenc_can_encode_vp9)" = "yes" ] && \
     [ "$(webm_io_available)" = "yes" ]; then
    local output="${VPX_TEST_OUTPUT_DIR}/vp9_rt.webm"
    vpxenc $(yuv_input_hantro_collage) \
      $(vpxenc_rt_params vp9) \
      --output="${output}" || return 1

    if [ ! -e "${output}" ]; then
      elog "Output file does not exist."
      return 1
    fi
  fi
}

vpxenc_vp9_webm_rt_multithread_tiled() {
  if [ "$(vpxenc_can_encode_vp9)" = "yes" ] && \
     [ "$(webm_io_available)" = "yes" ]; then
    local output="${VPX_TEST_OUTPUT_DIR}/vp9_rt_multithread_tiled.webm"
    local tilethread_min=2
    local tilethread_max=4
    local num_threads="$(seq ${tilethread_min} ${tilethread_max})"
    local num_tile_cols="$(seq ${tilethread_min} ${tilethread_max})"

    for threads in ${num_threads}; do
      for tile_cols in ${num_tile_cols}; do
        vpxenc $(y4m_input_720p) \
          $(vpxenc_rt_params vp9) \
          --threads=${threads} \
          --tile-columns=${tile_cols} \
          --output="${output}" || return 1

        if [ ! -e "${output}" ]; then
          elog "Output file does not exist."
          return 1
        fi
        rm "${output}"
      done
    done
  fi
}

vpxenc_vp9_webm_rt_multithread_tiled_frameparallel() {
  if [ "$(vpxenc_can_encode_vp9)" = "yes" ] && \
     [ "$(webm_io_available)" = "yes" ]; then
    local output="${VPX_TEST_OUTPUT_DIR}/vp9_rt_mt_t_fp.webm"
    local tilethread_min=2
    local tilethread_max=4
    local num_threads="$(seq ${tilethread_min} ${tilethread_max})"
    local num_tile_cols="$(seq ${tilethread_min} ${tilethread_max})"

    for threads in ${num_threads}; do
      for tile_cols in ${num_tile_cols}; do
        vpxenc $(y4m_input_720p) \
          $(vpxenc_rt_params vp9) \
          --threads=${threads} \
          --tile-columns=${tile_cols} \
          --frame-parallel=1 \
          --output="${output}" || return 1

        if [ ! -e "${output}" ]; then
          elog "Output file does not exist."
          return 1
        fi
        rm "${output}"
      done
    done
  fi
}

vpxenc_vp9_webm_2pass() {
  if [ "$(vpxenc_can_encode_vp9)" = "yes" ] && \
     [ "$(webm_io_available)" = "yes" ]; then
    local output="${VPX_TEST_OUTPUT_DIR}/vp9.webm"
    vpxenc $(yuv_input_hantro_collage) \
      --codec=vp9 \
      --limit="${TEST_FRAMES}" \
      --output="${output}" \
      --passes=2 || return 1

    if [ ! -e "${output}" ]; then
      elog "Output file does not exist."
      return 1
    fi
  fi
}

vpxenc_vp9_ivf_lossless() {
  if [ "$(vpxenc_can_encode_vp9)" = "yes" ]; then
    local output="${VPX_TEST_OUTPUT_DIR}/vp9_lossless.ivf"
    local passes=$(vpxenc_passes_param)
    vpxenc $(yuv_input_hantro_collage) \
      --codec=vp9 \
      --limit="${TEST_FRAMES}" \
      --ivf \
      --output="${output}" \
      "${passes}" \
      --lossless=1 || return 1

    if [ ! -e "${output}" ]; then
      elog "Output file does not exist."
      return 1
    fi
  fi
}

vpxenc_vp9_ivf_minq0_maxq0() {
  if [ "$(vpxenc_can_encode_vp9)" = "yes" ]; then
    local output="${VPX_TEST_OUTPUT_DIR}/vp9_lossless_minq0_maxq0.ivf"
    local passes=$(vpxenc_passes_param)
    vpxenc $(yuv_input_hantro_collage) \
      --codec=vp9 \
      --limit="${TEST_FRAMES}" \
      --ivf \
      --output="${output}" \
      "${passes}" \
      --min-q=0 \
      --max-q=0 || return 1

    if [ ! -e "${output}" ]; then
      elog "Output file does not exist."
      return 1
    fi
  fi
}

vpxenc_vp9_webm_lag10_frames20() {
  if [ "$(vpxenc_can_encode_vp9)" = "yes" ] && \
     [ "$(webm_io_available)" = "yes" ]; then
    local lag_total_frames=20
    local lag_frames=10
    local output="${VPX_TEST_OUTPUT_DIR}/vp9_lag10_frames20.webm"
    local passes=$(vpxenc_passes_param)
    vpxenc $(yuv_input_hantro_collage) \
      --codec=vp9 \
      --limit="${lag_total_frames}" \
      --lag-in-frames="${lag_frames}" \
      --output="${output}" \
      "${passes}" \
      --auto-alt-ref=1 || return 1

    if [ ! -e "${output}" ]; then
      elog "Output file does not exist."
      return 1
    fi
  fi
}

# TODO(fgalligan): Test that DisplayWidth is different than video width.
vpxenc_vp9_webm_non_square_par() {
  if [ "$(vpxenc_can_encode_vp9)" = "yes" ] && \
     [ "$(webm_io_available)" = "yes" ]; then
    local output="${VPX_TEST_OUTPUT_DIR}/vp9_non_square_par.webm"
    local passes=$(vpxenc_passes_param)
    vpxenc $(y4m_input_non_square_par) \
      --codec=vp9 \
      --limit="${TEST_FRAMES}" \
      "${passes}" \
      --output="${output}" || return 1

    if [ ! -e "${output}" ]; then
      elog "Output file does not exist."
      return 1
    fi
  fi
}

vpxenc_vp9_webm_sharpness() {
  if [ "$(vpxenc_can_encode_vp9)" = "yes" ]; then
    local sharpnesses="0 1 2 3 4 5 6 7"
    local output="${VPX_TEST_OUTPUT_DIR}/vpxenc_vp9_webm_sharpness.ivf"
    local last_size=0
    local this_size=0

    for sharpness in ${sharpnesses}; do

      vpxenc $(yuv_input_hantro_collage) \
        --sharpness="${sharpness}" \
        --codec=vp9 \
        --limit=1 \
        --cpu-used=2 \
        --end-usage=q \
        --cq-level=40 \
        --output="${output}" \
        "${passes}" || return 1

      if [ ! -e "${output}" ]; then
        elog "Output file does not exist."
        return 1
      fi

      this_size=$(stat -c '%s' "${output}")
      if [ "${this_size}" -lt "${last_size}" ]; then
        elog "Higher sharpness value yielded lower file size."
        echo "${this_size}" " < " "${last_size}"
        return 1
      fi
      last_size="${this_size}"

    done
  fi
}

vpxenc_tests="vpxenc_vp8_ivf
              vpxenc_vp8_webm
              vpxenc_vp8_webm_rt
              vpxenc_vp8_ivf_piped_input
              vpxenc_vp9_ivf
              vpxenc_vp9_webm
              vpxenc_vp9_webm_rt
              vpxenc_vp9_webm_rt_multithread_tiled
              vpxenc_vp9_webm_rt_multithread_tiled_frameparallel
              vpxenc_vp9_ivf_lossless
              vpxenc_vp9_ivf_minq0_maxq0
              vpxenc_vp9_webm_lag10_frames20
              vpxenc_vp9_webm_non_square_par
              vpxenc_vp9_webm_sharpness"

if [ "$(vpx_config_option_enabled CONFIG_REALTIME_ONLY)" != "yes" ]; then
  vpxenc_tests="$vpxenc_tests
                vpxenc_vp8_webm_2pass
                vpxenc_vp8_webm_lag10_frames20
                vpxenc_vp9_webm_2pass"
fi

run_tests vpxenc_verify_environment "${vpxenc_tests}"
