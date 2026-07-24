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
##  This file contains shell code shared by test scripts for libvpx tools.

# Use $VPX_TEST_TOOLS_COMMON_SH as a pseudo include guard.
if [ -z "${VPX_TEST_TOOLS_COMMON_SH}" ]; then
VPX_TEST_TOOLS_COMMON_SH=included

set -e
devnull='> /dev/null 2>&1'
VPX_TEST_PREFIX=""

elog() {
  echo "$@" 1>&2
}

vlog() {
  if [ "${VPX_TEST_VERBOSE_OUTPUT}" = "yes" ]; then
    echo "$@"
  fi
}

# Sets $VPX_TOOL_TEST to the name specified by positional parameter one.
test_begin() {
  VPX_TOOL_TEST="${1}"
}

# Clears the VPX_TOOL_TEST variable after confirming that $VPX_TOOL_TEST matches
# positional parameter one.
test_end() {
  if [ "$1" != "${VPX_TOOL_TEST}" ]; then
    echo "FAIL completed test mismatch!."
    echo "  completed test: ${1}"
    echo "  active test: ${VPX_TOOL_TEST}."
    return 1
  fi
  VPX_TOOL_TEST='<unset>'
}

# Echoes the target configuration being tested.
test_configuration_target() {
  vpx_config_mk="${LIBVPX_CONFIG_PATH}/config.mk"
  # Find the TOOLCHAIN line, split it using ':=' as the field separator, and
  # print the last field to get the value. Then pipe the value to tr to consume
  # any leading/trailing spaces while allowing tr to echo the output to stdout.
  awk -F ':=' '/TOOLCHAIN/ { print $NF }' "${vpx_config_mk}" | tr -d ' '
}

# Trap function used for failure reports and tool output directory removal.
# When the contents of $VPX_TOOL_TEST do not match the string '<unset>', reports
# failure of test stored in $VPX_TOOL_TEST.
cleanup() {
  if [ -n "${VPX_TOOL_TEST}" ] && [ "${VPX_TOOL_TEST}" != '<unset>' ]; then
    echo "FAIL: $VPX_TOOL_TEST"
  fi
  if [ -n "${VPX_TEST_OUTPUT_DIR}" ] && [ -d "${VPX_TEST_OUTPUT_DIR}" ]; then
    rm -rf "${VPX_TEST_OUTPUT_DIR}"
  fi
}

# Echoes the git hash portion of the VERSION_STRING variable defined in
# $LIBVPX_CONFIG_PATH/config.mk to stdout, or the version number string when
# no git hash is contained in VERSION_STRING.
config_hash() {
  vpx_config_mk="${LIBVPX_CONFIG_PATH}/config.mk"
  # Find VERSION_STRING line, split it with "-g" and print the last field to
  # output the git hash to stdout.
  vpx_version=$(awk -F -g '/VERSION_STRING/ {print $NF}' "${vpx_config_mk}")
  # Handle two situations here:
  # 1. The default case: $vpx_version is a git hash, so echo it unchanged.
  # 2. When being run a non-dev tree, the -g portion is not present in the
  #    version string: It's only the version number.
  #    In this case $vpx_version is something like 'VERSION_STRING=v1.3.0', so
  #    we echo only what is after the '='.
  echo "${vpx_version##*=}"
}

# Echoes the short form of the current git hash.
current_hash() {
  if git --version > /dev/null 2>&1; then
    (cd "$(dirname "${0}")"
    git rev-parse --short HEAD)
  else
    # Return the config hash if git is unavailable: Fail silently, git hashes
    # are used only for warnings.
    config_hash
  fi
}

# Echoes warnings to stdout when git hash in vpx_config.h does not match the
# current git hash.
check_git_hashes() {
  hash_at_configure_time=$(config_hash)
  hash_now=$(current_hash)

  if [ "${hash_at_configure_time}" != "${hash_now}" ]; then
    echo "Warning: git hash has changed since last configure."
  fi
}

# $1 is the name of an environment variable containing a directory name to
# test.
test_env_var_dir() {
  local dir=$(eval echo "\${$1}")
  if [ ! -d "${dir}" ]; then
    elog "'${dir}': No such directory"
    elog "The $1 environment variable must be set to a valid directory."
    return 1
  fi
}

# This script requires that the LIBVPX_BIN_PATH, LIBVPX_CONFIG_PATH, and
# LIBVPX_TEST_DATA_PATH variables are in the environment: Confirm that
# the variables are set and that they all evaluate to directory paths.
verify_vpx_test_environment() {
  test_env_var_dir "LIBVPX_BIN_PATH" \
    && test_env_var_dir "LIBVPX_CONFIG_PATH" \
    && test_env_var_dir "LIBVPX_TEST_DATA_PATH"
}

# Greps vpx_config.h in LIBVPX_CONFIG_PATH for positional parameter one, which
# should be a LIBVPX preprocessor flag. Echoes yes to stdout when the feature
# is available.
vpx_config_option_enabled() {
  vpx_config_option="${1}"
  vpx_config_file="${LIBVPX_CONFIG_PATH}/vpx_config.h"
  config_line=$(grep "${vpx_config_option}" "${vpx_config_file}")
  if echo "${config_line}" | grep -E -q '1$'; then
    echo yes
  fi
}

# Echoes yes when output of test_configuration_target() contains win32 or win64.
is_windows_target() {
  if test_configuration_target \
     | grep -q -e win32 -e win64 > /dev/null 2>&1; then
    echo yes
  fi
}

# Echoes path to $1 when it's executable and exists in ${LIBVPX_BIN_PATH}, or an
# empty string. Caller is responsible for testing the string once the function
# returns.
vpx_tool_path() {
  local tool_name="$1"
  local tool_path="${LIBVPX_BIN_PATH}/${tool_name}${VPX_TEST_EXE_SUFFIX}"
  if [ ! -x "${tool_path}" ]; then
    # Try one directory up: when running via examples.sh the tool could be in
    # the parent directory of $LIBVPX_BIN_PATH.
    tool_path="${LIBVPX_BIN_PATH}/../${tool_name}${VPX_TEST_EXE_SUFFIX}"
  fi

  if [ ! -x "${tool_path}" ]; then
    tool_path=""
  fi
  echo "${tool_path}"
}

# Echoes yes to stdout when the file named by positional parameter one exists
# in LIBVPX_BIN_PATH, and is executable.
vpx_tool_available() {
  local tool_name="$1"
  local tool="${LIBVPX_BIN_PATH}/${tool_name}${VPX_TEST_EXE_SUFFIX}"
  [ -x "${tool}" ] && echo yes
}

# Echoes yes to stdout when vpx_config_option_enabled() reports yes for
# CONFIG_VP8_DECODER.
vp8_decode_available() {
  [ "$(vpx_config_option_enabled CONFIG_VP8_DECODER)" = "yes" ] && echo yes
}

# Echoes yes to stdout when vpx_config_option_enabled() reports yes for
# CONFIG_VP8_ENCODER.
vp8_encode_available() {
  [ "$(vpx_config_option_enabled CONFIG_VP8_ENCODER)" = "yes" ] && echo yes
}

# Echoes yes to stdout when vpx_config_option_enabled() reports yes for
# CONFIG_VP9_DECODER.
vp9_decode_available() {
  [ "$(vpx_config_option_enabled CONFIG_VP9_DECODER)" = "yes" ] && echo yes
}

# Echoes yes to stdout when vpx_config_option_enabled() reports yes for
# CONFIG_VP9_ENCODER.
vp9_encode_available() {
  [ "$(vpx_config_option_enabled CONFIG_VP9_ENCODER)" = "yes" ] && echo yes
}

# Echoes yes to stdout when vpx_config_option_enabled() reports yes for
# CONFIG_WEBM_IO.
webm_io_available() {
  [ "$(vpx_config_option_enabled CONFIG_WEBM_IO)" = "yes" ] && echo yes
}

# Filters strings from $1 using the filter specified by $2. Filter behavior
# depends on the presence of $3. When $3 is present, strings that match the
# filter are excluded. When $3 is omitted, strings matching the filter are
# included.
# The filtered result is echoed to stdout.
filter_strings() {
  strings=${1}
  filter=${2}
  exclude=${3}

  if [ -n "${exclude}" ]; then
    # When positional parameter three exists the caller wants to remove strings.
    # Tell grep to invert matches using the -v argument.
    exclude='-v'
  else
    unset exclude
  fi

  if [ -n "${filter}" ]; then
    for s in ${strings}; do
      if echo "${s}" | grep -E -q ${exclude} "${filter}" > /dev/null 2>&1; then
        filtered_strings="${filtered_strings} ${s}"
      fi
    done
  else
    filtered_strings="${strings}"
  fi
  echo "${filtered_strings}"
}

# Runs user test functions passed via positional parameters one and two.
# Functions in positional parameter one are treated as environment verification
# functions and are run unconditionally. Functions in positional parameter two
# are run according to the rules specified in vpx_test_usage().
run_tests() {
  local env_tests="verify_vpx_test_environment $1"
  local tests_to_filter="$2"
  local test_name="${VPX_TEST_NAME}"

  if [ -z "${test_name}" ]; then
    test_name="$(basename "${0%.*}")"
  fi

  if [ "${VPX_TEST_RUN_DISABLED_TESTS}" != "yes" ]; then
    # Filter out DISABLED tests.
    tests_to_filter=$(filter_strings "${tests_to_filter}" ^DISABLED exclude)
  fi

  if [ -n "${VPX_TEST_FILTER}" ]; then
    # Remove tests not matching the user's filter.
    tests_to_filter=$(filter_strings "${tests_to_filter}" ${VPX_TEST_FILTER})
  fi

  # User requested test listing: Dump test names and return.
  if [ "${VPX_TEST_LIST_TESTS}" = "yes" ]; then
    for test_name in $tests_to_filter; do
      echo ${test_name}
    done
    return
  fi

  # Don't bother with the environment tests if everything else was disabled.
  [ -z "${tests_to_filter}" ] && return

  # Combine environment and actual tests.
  local tests_to_run="${env_tests} ${tests_to_filter}"

  check_git_hashes

  # Run tests.
  for test in ${tests_to_run}; do
    test_begin "${test}"
    vlog "  RUN  ${test}"
    "${test}"
    vlog "  PASS ${test}"
    test_end "${test}"
  done

  # C vs SIMD tests are run for x86 32-bit, 64-bit and ARM platform
  if [ "${test_name}" = "vp9_c_vs_simd_encode" ]; then
    local tested_config="$(current_hash)"
  else
    local tested_config="$(test_configuration_target) @ $(current_hash)"
  fi
  echo "${test_name}: Done, all tests pass for ${tested_config}."
}

vpx_test_usage() {
cat << EOF
  Usage: ${0##*/} [arguments]
    --bin-path <path to libvpx binaries directory>
    --config-path <path to libvpx config directory>
    --filter <filter>: User test filter. Only tests matching filter are run.
    --run-disabled-tests: Run disabled tests.
    --help: Display this message and exit.
    --test-data-path <path to libvpx test data directory>
    --show-program-output: Shows output from all programs being tested.
    --prefix: Allows for a user specified prefix to be inserted before all test
              programs. Grants the ability, for example, to run test programs
              within valgrind.
    --list-tests: List all test names and exit without actually running tests.
    --verbose: Verbose output.

    When the --bin-path option is not specified the script attempts to use
    \$LIBVPX_BIN_PATH and then the current directory.

    When the --config-path option is not specified the script attempts to use
    \$LIBVPX_CONFIG_PATH and then the current directory.

    When the -test-data-path option is not specified the script attempts to use
    \$LIBVPX_TEST_DATA_PATH and then the current directory.
EOF
}

# Returns non-zero (failure) when required environment variables are empty
# strings.
vpx_test_check_environment() {
  if [ -z "${LIBVPX_BIN_PATH}" ] || \
     [ -z "${LIBVPX_CONFIG_PATH}" ] || \
     [ -z "${LIBVPX_TEST_DATA_PATH}" ]; then
    return 1
  fi
}

# Parse the command line.
while [ -n "$1" ]; do
  case "$1" in
    --bin-path)
      LIBVPX_BIN_PATH="$2"
      shift
      ;;
    --config-path)
      LIBVPX_CONFIG_PATH="$2"
      shift
      ;;
    --filter)
      VPX_TEST_FILTER="$2"
      shift
      ;;
    --run-disabled-tests)
      VPX_TEST_RUN_DISABLED_TESTS=yes
      ;;
    --help)
      vpx_test_usage
      exit
      ;;
    --test-data-path)
      LIBVPX_TEST_DATA_PATH="$2"
      shift
      ;;
    --prefix)
      VPX_TEST_PREFIX="$2"
      shift
      ;;
    --verbose)
      VPX_TEST_VERBOSE_OUTPUT=yes
      ;;
    --show-program-output)
      devnull=
      ;;
    --list-tests)
      VPX_TEST_LIST_TESTS=yes
      ;;
    *)
      vpx_test_usage
      exit 1
      ;;
  esac
  shift
done

# Handle running the tests from a build directory without arguments when running
# the tests on *nix/macosx.
LIBVPX_BIN_PATH="${LIBVPX_BIN_PATH:-.}"
LIBVPX_CONFIG_PATH="${LIBVPX_CONFIG_PATH:-.}"
LIBVPX_TEST_DATA_PATH="${LIBVPX_TEST_DATA_PATH:-.}"

# Create a temporary directory for output files, and a trap to clean it up.
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

if [ "$(is_windows_target)" = "yes" ]; then
  VPX_TEST_EXE_SUFFIX=".exe"
fi

# Variables shared by tests.
VP8_IVF_FILE="${LIBVPX_TEST_DATA_PATH}/vp80-00-comprehensive-001.ivf"
VP9_IVF_FILE="${LIBVPX_TEST_DATA_PATH}/vp90-2-09-subpixel-00.ivf"

VP9_WEBM_FILE="${LIBVPX_TEST_DATA_PATH}/vp90-2-00-quantizer-00.webm"
VP9_FPM_WEBM_FILE="${LIBVPX_TEST_DATA_PATH}/vp90-2-07-frame_parallel-1.webm"
VP9_LT_50_FRAMES_WEBM_FILE="${LIBVPX_TEST_DATA_PATH}/vp90-2-02-size-32x08.webm"

VP9_RAW_FILE="${LIBVPX_TEST_DATA_PATH}/crbug-1539.rawfile"

YUV_RAW_INPUT="${LIBVPX_TEST_DATA_PATH}/hantro_collage_w352h288.yuv"
YUV_RAW_INPUT_WIDTH=352
YUV_RAW_INPUT_HEIGHT=288

Y4M_NOSQ_PAR_INPUT="${LIBVPX_TEST_DATA_PATH}/park_joy_90p_8_420_a10-1.y4m"
Y4M_720P_INPUT="${LIBVPX_TEST_DATA_PATH}/niklas_1280_720_30.y4m"
Y4M_720P_INPUT_WIDTH=1280
Y4M_720P_INPUT_HEIGHT=720

# Setup a trap function to clean up after tests complete.
trap cleanup EXIT

vlog "$(basename "${0%.*}") test configuration:
  LIBVPX_BIN_PATH=${LIBVPX_BIN_PATH}
  LIBVPX_CONFIG_PATH=${LIBVPX_CONFIG_PATH}
  LIBVPX_TEST_DATA_PATH=${LIBVPX_TEST_DATA_PATH}
  VP8_IVF_FILE=${VP8_IVF_FILE}
  VP9_IVF_FILE=${VP9_IVF_FILE}
  VP9_WEBM_FILE=${VP9_WEBM_FILE}
  VPX_TEST_EXE_SUFFIX=${VPX_TEST_EXE_SUFFIX}
  VPX_TEST_FILTER=${VPX_TEST_FILTER}
  VPX_TEST_LIST_TESTS=${VPX_TEST_LIST_TESTS}
  VPX_TEST_OUTPUT_DIR=${VPX_TEST_OUTPUT_DIR}
  VPX_TEST_PREFIX=${VPX_TEST_PREFIX}
  VPX_TEST_RUN_DISABLED_TESTS=${VPX_TEST_RUN_DISABLED_TESTS}
  VPX_TEST_SHOW_PROGRAM_OUTPUT=${VPX_TEST_SHOW_PROGRAM_OUTPUT}
  VPX_TEST_TEMP_ROOT=${VPX_TEST_TEMP_ROOT}
  VPX_TEST_VERBOSE_OUTPUT=${VPX_TEST_VERBOSE_OUTPUT}
  YUV_RAW_INPUT=${YUV_RAW_INPUT}
  YUV_RAW_INPUT_WIDTH=${YUV_RAW_INPUT_WIDTH}
  YUV_RAW_INPUT_HEIGHT=${YUV_RAW_INPUT_HEIGHT}
  Y4M_NOSQ_PAR_INPUT=${Y4M_NOSQ_PAR_INPUT}"

fi  # End $VPX_TEST_TOOLS_COMMON_SH pseudo include guard.
