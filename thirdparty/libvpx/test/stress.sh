#!/bin/sh
##
##  Copyright (c) 2016 The WebM project authors. All Rights Reserved.
##
##  Use of this source code is governed by a BSD-style license
##  that can be found in the LICENSE file in the root of the source
##  tree. An additional intellectual property rights grant can be found
##  in the file PATENTS.  All contributing project authors may
##  be found in the AUTHORS file in the root of the source tree.
##
##  This file performs a stress test. It runs (STRESS_ONEPASS_MAX_JOBS,
##  default=5) one, (STRESS_TWOPASS_MAX_JOBS, default=5) two pass &
##  (STRESS_RT_MAX_JOBS, default=5) encodes and (STRESS_<codec>_DECODE_MAX_JOBS,
##  default=30) decodes in parallel.

. $(dirname $0)/tools_common.sh

YUV="${LIBVPX_TEST_DATA_PATH}/niklas_1280_720_30.yuv"
VP8="${LIBVPX_TEST_DATA_PATH}/tos_vp8.webm"
VP9="${LIBVPX_TEST_DATA_PATH}/vp90-2-sintel_1920x818_tile_1x4_fpm_2279kbps.webm"
DATA_URL="https://storage.googleapis.com/downloads.webmproject.org/test_data/libvpx/"
SHA1_FILE="$(dirname $0)/test-data.sha1"

# Set sha1sum to proper sha program (sha1sum, shasum, sha1). This code is
# cribbed from libs.mk.
[ -x "$(which sha1sum)" ] && sha1sum=sha1sum
[ -x "$(which shasum)" ] && sha1sum=shasum
[ -x "$(which sha1)" ] && sha1sum=sha1

# Download a file from the url and check its sha1sum.
download_and_check_file() {
  # Get the file from the file path.
  local root="${1#${LIBVPX_TEST_DATA_PATH}/}"

  # Download the file using curl. Trap to insure non partial file.
  (trap "rm -f $1" INT TERM \
    && eval "curl --retry 1 -L -o $1 ${DATA_URL}${root} ${devnull}")

  # Check the sha1 sum of the file.
  if [ -n "${sha1sum}" ]; then
    set -e
    grep ${root} ${SHA1_FILE} \
      | (cd ${LIBVPX_TEST_DATA_PATH}; ${sha1sum} -c);
  fi
}

# Environment check: Make sure input is available.
stress_verify_environment() {
  if [ ! -e "${SHA1_FILE}" ] ; then
    echo "Missing ${SHA1_FILE}"
    return 1
  fi
  for file in "${YUV}" "${VP8}" "${VP9}"; do
    if [ ! -e "${file}" ] ; then
      download_and_check_file "${file}" || return 1
    fi
  done
  if [ ! -e "${YUV}" ] || [ ! -e "${VP8}" ] || [ ! -e "${VP9}" ] ; then
    elog "Libvpx test data must exist in LIBVPX_TEST_DATA_PATH."
    return 1
  fi
  if [ -z "$(vpx_tool_path vpxenc)" ]; then
    elog "vpxenc not found. It must exist in LIBVPX_BIN_PATH or its parent."
    return 1
  fi
  if [ -z "$(vpx_tool_path vpxdec)" ]; then
    elog "vpxdec not found. It must exist in LIBVPX_BIN_PATH or its parent."
    return 1
  fi
}

# This function runs tests on libvpx that run multiple encodes and decodes
# in parallel in hopes of catching synchronization and/or threading issues.
stress() {
  local decoder="$(vpx_tool_path vpxdec)"
  local encoder="$(vpx_tool_path vpxenc)"
  local codec="$1"
  local webm="$2"
  local decode_count="$3"
  local threads="$4"
  local enc_args="$5"
  local pids=""
  local rt_max_jobs=${STRESS_RT_MAX_JOBS:-5}
  local onepass_max_jobs=${STRESS_ONEPASS_MAX_JOBS:-5}
  local twopass_max_jobs=${STRESS_TWOPASS_MAX_JOBS:-5}

  # Enable job control, so we can run multiple processes.
  set -m

  # Start $onepass_max_jobs encode jobs in parallel.
  for i in $(seq ${onepass_max_jobs}); do
    bitrate=$(($i * 20 + 300))
    eval "${VPX_TEST_PREFIX}" "${encoder}" "--codec=${codec} -w 1280 -h 720" \
      "${YUV}" "-t ${threads} --limit=150 --test-decode=fatal --passes=1" \
      "--target-bitrate=${bitrate} -o ${VPX_TEST_OUTPUT_DIR}/${i}.1pass.webm" \
      "${enc_args}" ${devnull} &
    pids="${pids} $!"
  done

  # Start $twopass_max_jobs encode jobs in parallel.
  for i in $(seq ${twopass_max_jobs}); do
    bitrate=$(($i * 20 + 300))
    eval "${VPX_TEST_PREFIX}" "${encoder}" "--codec=${codec} -w 1280 -h 720" \
      "${YUV}" "-t ${threads} --limit=150 --test-decode=fatal --passes=2" \
      "--target-bitrate=${bitrate} -o ${VPX_TEST_OUTPUT_DIR}/${i}.2pass.webm" \
      "${enc_args}" ${devnull} &
    pids="${pids} $!"
  done

  # Start $rt_max_jobs rt encode jobs in parallel.
  for i in $(seq ${rt_max_jobs}); do
    bitrate=$(($i * 20 + 300))
    eval "${VPX_TEST_PREFIX}" "${encoder}" "--codec=${codec} -w 1280 -h 720" \
      "${YUV}" "-t ${threads} --limit=150 --test-decode=fatal " \
      "--target-bitrate=${bitrate} --lag-in-frames=0 --error-resilient=1" \
      "--kf-min-dist=3000 --kf-max-dist=3000 --cpu-used=-6 --static-thresh=1" \
      "--end-usage=cbr --min-q=2 --max-q=56 --undershoot-pct=100" \
      "--overshoot-pct=15 --buf-sz=1000 --buf-initial-sz=500" \
      "--buf-optimal-sz=600 --max-intra-rate=900 --resize-allowed=0" \
      "--drop-frame=0 --passes=1 --rt --noise-sensitivity=4" \
      "-o ${VPX_TEST_OUTPUT_DIR}/${i}.rt.webm" ${devnull} &
    pids="${pids} $!"
  done

  # Start $decode_count decode jobs in parallel.
  for i in $(seq "${decode_count}"); do
    eval "${decoder}" "-t ${threads}" "${webm}" "--noblit" ${devnull} &
    pids="${pids} $!"
  done

  # Wait for all parallel jobs to finish.
  fail=0
  for job in "${pids}"; do
    wait $job || fail=$(($fail + 1))
  done
  return $fail
}

vp8_stress_test() {
  local vp8_max_jobs=${STRESS_VP8_DECODE_MAX_JOBS:-40}
  if [ "$(vp8_decode_available)" = "yes" -a \
       "$(vp8_encode_available)" = "yes" ]; then
    stress vp8 "${VP8}" "${vp8_max_jobs}" 4
  fi
}

vp8_stress_test_token_parititions() {
  local vp8_max_jobs=${STRESS_VP8_DECODE_MAX_JOBS:-40}
  if [ "$(vp8_decode_available)" = "yes" -a \
       "$(vp8_encode_available)" = "yes" ]; then
    for threads in 2 4 8; do
      for token_partitions in 1 2 3; do
        stress vp8 "${VP8}" "${vp8_max_jobs}" ${threads} \
          "--token-parts=$token_partitions"
      done
    done
  fi
}

vp9_stress() {
  local vp9_max_jobs=${STRESS_VP9_DECODE_MAX_JOBS:-25}

  if [ "$(vp9_decode_available)" = "yes" -a \
       "$(vp9_encode_available)" = "yes" ]; then
    stress vp9 "${VP9}" "${vp9_max_jobs}" "$@"
  fi
}

vp9_stress_test() {
  for threads in 4 8 64; do
    vp9_stress "$threads" "--row-mt=0"
  done
}

vp9_stress_test_row_mt() {
  for threads in 4 8 64; do
    vp9_stress "$threads" "--row-mt=1"
  done
}

run_tests stress_verify_environment \
  "vp8_stress_test vp8_stress_test_token_parititions
   vp9_stress_test vp9_stress_test_row_mt"
