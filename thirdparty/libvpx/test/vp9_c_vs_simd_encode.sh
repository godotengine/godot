#!/bin/sh
##
##  Copyright (c) 2023 The WebM project authors. All Rights Reserved.
##
##  Use of this source code is governed by a BSD-style license
##  that can be found in the LICENSE file in the root of the source
##  tree. An additional intellectual property rights grant can be found
##  in the file PATENTS.  All contributing project authors may
##  be found in the AUTHORS file in the root of the source tree.
##
##  This script checks the bit exactness between C and SIMD
##  implementations of VP9 encoder.
##
. $(dirname $0)/tools_common.sh

TEST_BITRATES="1600 6400"
PRESETS="good rt"
TEST_CLIPS="yuv_raw_input y4m_360p_10bit_input yuv_480p_raw_input y4m_720p_input"
OUT_FILE_SUFFIX=".ivf"
SCRIPT_DIR=$(dirname "$0")
LIBVPX_SOURCE_DIR=$(cd "${SCRIPT_DIR}/.."; pwd)

# Clips used in test.
YUV_RAW_INPUT="${LIBVPX_TEST_DATA_PATH}/hantro_collage_w352h288.yuv"
YUV_480P_RAW_INPUT="${LIBVPX_TEST_DATA_PATH}/niklas_640_480_30.yuv"
Y4M_360P_10BIT_INPUT="${LIBVPX_TEST_DATA_PATH}/crowd_run_360p_10_150f.y4m"
Y4M_720P_INPUT="${LIBVPX_TEST_DATA_PATH}/niklas_1280_720_30.y4m"

# Number of frames to test.
VP9_ENCODE_C_VS_SIMD_TEST_FRAME_LIMIT=20

# Create a temporary directory for output files.
if [ -n "${TMPDIR}" ]; then
  VPX_TEST_TEMP_ROOT="${TMPDIR}"
elif [ -n "${TEMPDIR}" ]; then
  VPX_TEST_TEMP_ROOT="${TEMPDIR}"
else
  VPX_TEST_TEMP_ROOT=/tmp
fi

VPX_TEST_OUTPUT_DIR="${VPX_TEST_TEMP_ROOT}/vpx_test_$$"

if ! mkdir -p "${VPX_TEST_OUTPUT_DIR}" || \
   [ ! -d "${VPX_TEST_OUTPUT_DIR}" ]; then
  echo "${0##*/}: Cannot create output directory, giving up."
  echo "${0##*/}:   VPX_TEST_OUTPUT_DIR=${VPX_TEST_OUTPUT_DIR}"
  exit 1
fi

elog() {
  echo "$@" 1>&2
}

# Echoes path to $1 when it's executable and exists in ${VPX_TEST_OUTPUT_DIR},
# or an empty string. Caller is responsible for testing the string once the
# function returns.
vp9_enc_tool_path() {
  local target="$1"
  local tool_path="${VPX_TEST_OUTPUT_DIR}/build_target_${target}/vpxenc"

  if [ ! -x "${tool_path}" ]; then
    tool_path=""
  fi
  echo "${tool_path}"
}

# Environment check: Make sure input and source directories are available.
vp9_c_vs_simd_enc_verify_environment() {
  if [ ! -e "${YUV_RAW_INPUT}" ]; then
    elog "libvpx test data must exist in LIBVPX_TEST_DATA_PATH."
    return 1
  fi
  if [ ! -e "${YUV_480P_RAW_INPUT}" ]; then
    elog "libvpx test data must exist in LIBVPX_TEST_DATA_PATH."
    return 1
  fi
  if [ ! -e "${Y4M_720P_INPUT}" ]; then
    elog "libvpx test data must exist in LIBVPX_TEST_DATA_PATH."
    return 1
  fi
  if [ ! -e "${Y4M_360P_10BIT_INPUT}" ]; then
    elog "libvpx test data must exist in LIBVPX_TEST_DATA_PATH."
    return 1
  fi
  if [ ! -d "$LIBVPX_SOURCE_DIR" ]; then
    elog "LIBVPX_SOURCE_DIR does not exist."
    return 1
  fi
}

# This is not needed since tools_common.sh does the same cleanup.
# Keep the code here for our reference.
# cleanup() {
#   rm -rf  ${VPX_TEST_OUTPUT_DIR}
# }

# Echo VPX_SIMD_CAPS_MASK for different instruction set architecture.
avx512f() {
   echo "0x1FF"
}

avx2() {
   echo "0x0FF"
}

sse4_1() {
   echo "0x03F"
}

ssse3() {
   echo "0x01F"
}

sse2() {
   echo "0x007"
}

# Echo clip details to be used as input to vpxenc.
yuv_raw_input() {
  echo ""${YUV_RAW_INPUT}"
       --width=352
       --height=288
       --bit-depth=8
       --profile=0"
}

yuv_480p_raw_input() {
  echo ""${YUV_480P_RAW_INPUT}"
       --width=640
       --height=480
       --bit-depth=8
       --profile=0"
}

y4m_720p_input() {
  echo ""${Y4M_720P_INPUT}"
       --bit-depth=8
       --profile=0"
}

y4m_360p_10bit_input() {
  echo ""${Y4M_360P_10BIT_INPUT}"
       --bit-depth=10
       --profile=2"
}

has_x86_isa_extn() {
  instruction_set=$1
  if ! grep -q "$instruction_set" /proc/cpuinfo; then
    # This instruction_set is not supported.
    return 1
  fi
  # This instruction_set is supported.
  return 0
}

# Echo good encode params for use with VP9 encoder.
vp9_encode_good_params() {
  echo "--codec=vp9 \
  --good \
  --test-decode=fatal \
  --ivf \
  --threads=1 \
  --static-thresh=0 \
  --tile-columns=0 \
  --end-usage=vbr \
  --kf-max-dist=160 \
  --kf-min-dist=0 \
  --lag-in-frames=19 \
  --max-q=63 \
  --min-q=0 \
  --passes=2 \
  --undershoot-pct=100 \
  --overshoot-pct=100 \
  --verbose \
  --auto-alt-ref=1 \
  --drop-frame=0 \
  --bias-pct=50 \
  --minsection-pct=0 \
  --maxsection-pct=2000 \
  --arnr-maxframes=7 \
  --arnr-strength=5 \
  --sharpness=0 \
  --frame-parallel=0"
}

# Echo realtime encode params for use with VP9 encoder.
vp9_encode_rt_params() {
  echo "--codec=vp9 \
  --rt \
  --test-decode=fatal \
  --ivf \
  --threads=1 \
  --static-thresh=0 \
  --tile-columns=0 \
  --tile-rows=0 \
  --end-usage=cbr \
  --kf-max-dist=90000 \
  --lag-in-frames=0 \
  --max-q=58 \
  --min-q=2 \
  --passes=1 \
  --undershoot-pct=50 \
  --overshoot-pct=50 \
  --verbose \
  --row-mt=0 \
  --buf-sz=1000 \
  --buf-initial-sz=500 \
  --buf-optimal-sz=600 \
  --max-intra-rate=300 \
  --resize-allowed=0 \
  --noise-sensitivity=0 \
  --aq-mode=3 \
  --error-resilient=0"
}

# Configures for the given target in the
# ${VPX_TEST_OUTPUT_DIR}/build_target_${target} directory.
vp9_enc_build() {
  local target=$1
  local configure="$2"
  local tmp_build_dir=${VPX_TEST_OUTPUT_DIR}/build_target_${target}
  mkdir -p "$tmp_build_dir"
  local save_dir="$PWD"
  cd "$tmp_build_dir"

  echo "Building target: ${target}"
  local config_args="--disable-install-docs \
             --enable-unit-tests \
             --enable-debug \
             --enable-postproc \
             --enable-vp9-postproc \
             --enable-vp9-temporal-denoising \
             --enable-vp9-highbitdepth"

  eval "$configure" --target="${target}" "${config_args}" ${devnull}
  eval make -j$(nproc) ${devnull}
  echo "Done building target: ${target}"
  cd "${save_dir}"
}

compare_enc_output() {
  local target=$1
  local cpu=$2
  local clip=$3
  local bitrate=$4
  local preset=$5
  if ! diff -q ${VPX_TEST_OUTPUT_DIR}/Out-generic-gnu-"${clip}"-${preset}-${bitrate}kbps-cpu${cpu}${OUT_FILE_SUFFIX} \
       ${VPX_TEST_OUTPUT_DIR}/Out-${target}-"${clip}"-${preset}-${bitrate}kbps-cpu${cpu}${OUT_FILE_SUFFIX}; then
    elog "C vs ${target} encode mismatches for ${clip}, at ${bitrate} kbps, speed ${cpu}, ${preset} preset"
    return 1
  fi
}

vp9_enc_test() {
  local encoder="$1"
  local target=$2
  if [ -z "$(vp9_enc_tool_path "${target}")" ]; then
    elog "vpxenc not found. It must exist in ${VPX_TEST_OUTPUT_DIR}/build_target_${target} path"
    return 1
  fi

  local tmp_build_dir=${VPX_TEST_OUTPUT_DIR}/build_target_${target}
  local save_dir="$PWD"
  cd "$tmp_build_dir"
  for preset in ${PRESETS}; do
    if [ "${preset}" = "good" ]; then
      local max_cpu_used=5
      local test_params=vp9_encode_good_params
    elif [ "${preset}" = "rt" ]; then
      local max_cpu_used=9
      local test_params=vp9_encode_rt_params
    else
      elog "Invalid preset"
      cd "${save_dir}"
      return 1
    fi

    # Enable armv8 test for real-time only
    if [ "${preset}" = "good" ] && [ "${target}" = "armv8-linux-gcc" ]; then
      continue
    fi

    for cpu in $(seq 0 $max_cpu_used); do
      for clip in ${TEST_CLIPS}; do
        for bitrate in ${TEST_BITRATES}; do
          eval "${encoder}" $($clip) $($test_params) \
          "--limit=${VP9_ENCODE_C_VS_SIMD_TEST_FRAME_LIMIT}" \
          "--cpu-used=${cpu}" "--target-bitrate=${bitrate}" "-o" \
          ${VPX_TEST_OUTPUT_DIR}/Out-${target}-"${clip}"-${preset}-${bitrate}kbps-cpu${cpu}${OUT_FILE_SUFFIX} \
          ${devnull}

          if [ "${target}" != "generic-gnu" ]; then
            if ! compare_enc_output ${target} $cpu ${clip} $bitrate ${preset}; then
              # Find the mismatch
              cd "${save_dir}"
              return 1
            fi
          fi
        done
      done
    done
  done
  cd "${save_dir}"
}

vp9_test_generic() {
  local configure="$LIBVPX_SOURCE_DIR/configure"
  local target="generic-gnu"

  echo "Build for: ${target}"
  vp9_enc_build ${target} ${configure}
  local encoder="$(vp9_enc_tool_path "${target}")"
  vp9_enc_test $encoder "${target}"
}

# This function encodes VP9 bitstream by enabling SSE2, SSSE3, SSE4_1, AVX2, AVX512f as there are
# no functions with MMX, SSE, SSE3 and AVX specialization.
# The value of environment variable 'VPX_SIMD_CAPS' controls enabling of different instruction
# set extension optimizations. The value of the flag 'VPX_SIMD_CAPS' and the corresponding
# instruction set extension optimization enabled are as follows:
# AVX512 AVX2 AVX SSE4_1 SSSE3 SSE3 SSE2 SSE MMX
#   1     1    1    1      1    1    1    1   1  -> 0x1FF -> Enable AVX512 and lower variants
#   0     1    1    1      1    1    1    1   1  -> 0x0FF -> Enable AVX2 and lower variants
#   0     0    1    1      1    1    1    1   1  -> 0x07F -> Enable AVX and lower variants
#   0     0    0    1      1    1    1    1   1  -> 0x03F  -> Enable SSE4_1 and lower variants
#   0     0    0    0      1    1    1    1   1  -> 0x01F  -> Enable SSSE3 and lower variants
#   0     0    0    0      0    1    1    1   1  -> 0x00F  -> Enable SSE3 and lower variants
#   0     0    0    0      0    0    1    1   1  -> 0x007  -> Enable SSE2 and lower variants
#   0     0    0    0      0    0    0    1   1  -> 0x003  -> Enable SSE and lower variants
#   0     0    0    0      0    0    0    0   1  -> 0x001  -> Enable MMX
## NOTE: In x86_64 platform, it is not possible to enable sse/mmx/c using "VPX_SIMD_CAPS_MASK" as
#  all x86_64 platforms implement sse2.
vp9_test_x86() {
  local arch=$1

  if ! uname -m | grep -q "x86"; then
    elog "Machine architecture is not x86 or x86_64"
    return 0
  fi

  if [ $arch = "x86" ]; then
    local target="x86-linux-gcc"
  elif [ $arch = "x86_64" ]; then
    local target="x86_64-linux-gcc"
  fi

  local x86_isa_variants="avx512f avx2 sse4_1 ssse3 sse2"
  local configure="$LIBVPX_SOURCE_DIR/configure"

  echo "Build for x86: ${target}"
  vp9_enc_build ${target} ${configure}
  local encoder="$(vp9_enc_tool_path "${target}")"
  for isa in $x86_isa_variants; do
    # Note that if has_x86_isa_extn returns 1, it is false, and vice versa.
    if ! has_x86_isa_extn $isa; then
      echo "${isa} is not supported in this machine"
      continue
    fi
    export VPX_SIMD_CAPS_MASK=$($isa)
    if ! vp9_enc_test $encoder ${target}; then
      # Find the mismatch
      return 1
    fi
    unset VPX_SIMD_CAPS_MASK
  done
}

vp9_test_arm() {
  local target="armv8-linux-gcc"
  local configure="CROSS=aarch64-linux-gnu- $LIBVPX_SOURCE_DIR/configure --extra-cflags=-march=armv8.4-a \
          --extra-cxxflags=-march=armv8.4-a"
  echo "Build for arm64: ${target}"
  vp9_enc_build ${target} "${configure}"

  local encoder="$(vp9_enc_tool_path "${target}")"
  if ! vp9_enc_test "qemu-aarch64 -L /usr/aarch64-linux-gnu ${encoder}" ${target}; then
    # Find the mismatch
    return 1
  fi
}

vp9_c_vs_simd_enc_test() {
  # Test Generic
  vp9_test_generic

  # Test x86 (32 bit)
  echo "vp9 test for x86 (32 bit): Started."
  if ! vp9_test_x86 "x86"; then
    echo "vp9 test for x86 (32 bit): Done, test failed."
    return 1
  else
    echo "vp9 test for x86 (32 bit): Done, all tests passed."
  fi

  # Test x86_64 (64 bit)
  if [ "$(eval uname -m)" = "x86_64" ]; then
    echo "vp9 test for x86_64 (64 bit): Started."
    if ! vp9_test_x86 "x86_64"; then
      echo "vp9 test for x86_64 (64 bit): Done, test failed."
      return 1
    else
      echo "vp9 test for x86_64 (64 bit): Done, all tests passed."
    fi
  fi

  # Test ARM
  echo "vp9_test_arm: Started."
  if ! vp9_test_arm; then
    echo "vp9 test for arm: Done, test failed."
    return 1
  else
    echo "vp9 test for arm: Done, all tests passed."
  fi
}

# Setup a trap function to clean up build, and output files after tests complete.
# trap cleanup EXIT

run_tests vp9_c_vs_simd_enc_verify_environment vp9_c_vs_simd_enc_test
