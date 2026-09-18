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
##
## This script generates 'VPX.framework'. An iOS app can encode and decode VPx
## video by including 'VPX.framework'.
##
## Run iosbuild.sh to create 'VPX.framework' in the current directory.
##
set -e
devnull='> /dev/null 2>&1'

BUILD_ROOT="_iosbuild"
CONFIGURE_ARGS="--disable-docs
                --disable-examples
                --disable-libyuv
                --disable-unit-tests"
DIST_DIR="_dist"
FRAMEWORK_DIR="VPX.framework"
FRAMEWORK_LIB="VPX.framework/VPX"
HEADER_DIR="${FRAMEWORK_DIR}/Headers/vpx"
SCRIPT_DIR=$(dirname "$0")
LIBVPX_SOURCE_DIR=$(cd ${SCRIPT_DIR}/../..; pwd)
LIPO=$(xcrun -sdk iphoneos${SDK} -find lipo)
ORIG_PWD="$(pwd)"
ARM_TARGETS="arm64-darwin-gcc"
SIM_TARGETS="x86_64-iphonesimulator-gcc"
OSX_TARGETS="x86_64-darwin16-gcc"
TARGETS="${ARM_TARGETS} ${SIM_TARGETS}"

# Configures for the target specified by $1, and invokes make with the dist
# target using $DIST_DIR as the distribution output directory.
build_target() {
  local target="$1"
  local old_pwd="$(pwd)"
  local target_specific_flags=""

  vlog "***Building target: ${target}***"

  case "${target}" in
    x86-*)
      target_specific_flags="--enable-pic"
      vlog "Enabled PIC for ${target}"
      ;;
  esac

  mkdir "${target}"
  cd "${target}"
  eval "${LIBVPX_SOURCE_DIR}/configure" --target="${target}" \
    ${CONFIGURE_ARGS} ${EXTRA_CONFIGURE_ARGS} ${target_specific_flags} \
    ${devnull}
  export DIST_DIR
  eval make dist ${devnull}
  cd "${old_pwd}"

  vlog "***Done building target: ${target}***"
}

# Returns the preprocessor symbol for the target specified by $1.
target_to_preproc_symbol() {
  target="$1"
  case "${target}" in
    arm64-*)
      echo "__aarch64__"
      ;;
    armv7-*)
      echo "__ARM_ARCH_7A__"
      ;;
    armv7s-*)
      echo "__ARM_ARCH_7S__"
      ;;
    x86-*)
      echo "__i386__"
      ;;
    x86_64-*)
      echo "__x86_64__"
      ;;
    *)
      echo "#error ${target} unknown/unsupported"
      return 1
      ;;
  esac
}

# Create a vpx_config.h shim that, based on preprocessor settings for the
# current target CPU, includes the real vpx_config.h for the current target.
# $1 is the list of targets.
create_vpx_framework_config_shim() {
  local targets="$1"
  local config_file="${HEADER_DIR}/vpx_config.h"
  local preproc_symbol=""
  local target=""
  local include_guard="VPX_FRAMEWORK_HEADERS_VPX_VPX_CONFIG_H_"

  local file_header="/*
 *  Copyright (c) $(date +%Y) The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

/* GENERATED FILE: DO NOT EDIT! */

#ifndef ${include_guard}
#define ${include_guard}

#if defined"

  printf "%s" "${file_header}" > "${config_file}"
  for target in ${targets}; do
    preproc_symbol=$(target_to_preproc_symbol "${target}")
    printf " ${preproc_symbol}\n" >> "${config_file}"
    printf "#define VPX_FRAMEWORK_TARGET \"${target}\"\n" >> "${config_file}"
    printf "#include \"VPX/vpx/${target}/vpx_config.h\"\n" >> "${config_file}"
    printf "#elif defined" >> "${config_file}"
    mkdir "${HEADER_DIR}/${target}"
    cp -p "${BUILD_ROOT}/${target}/vpx_config.h" "${HEADER_DIR}/${target}"
  done

  # Consume the last line of output from the loop: We don't want it.
  sed -i.bak -e '$d' "${config_file}"
  rm "${config_file}.bak"

  printf "#endif\n\n" >> "${config_file}"
  printf "#endif  // ${include_guard}" >> "${config_file}"
}

# Verifies that $FRAMEWORK_LIB fat library contains requested builds.
verify_framework_targets() {
  local requested_cpus=""
  local cpu=""

  # Extract CPU from full target name.
  for target; do
    cpu="${target%%-*}"
    if [ "${cpu}" = "x86" ]; then
      # lipo -info outputs i386 for libvpx x86 targets.
      cpu="i386"
    fi
    requested_cpus="${requested_cpus}${cpu} "
  done

  # Get target CPUs present in framework library.
  local targets_built=$(${LIPO} -info ${FRAMEWORK_LIB})

  # $LIPO -info outputs a string like the following:
  #   Architectures in the fat file: $FRAMEWORK_LIB <architectures>
  # Capture only the architecture strings.
  targets_built=${targets_built##*: }

  # Sort CPU strings to make the next step a simple string compare.
  local actual=$(echo ${targets_built} | tr " " "\n" | sort | tr "\n" " ")
  local requested=$(echo ${requested_cpus} | tr " " "\n" | sort | tr "\n" " ")

  vlog "Requested ${FRAMEWORK_LIB} CPUs: ${requested}"
  vlog "Actual ${FRAMEWORK_LIB} CPUs: ${actual}"

  if [ "${requested}" != "${actual}" ]; then
    elog "Actual ${FRAMEWORK_LIB} targets do not match requested target list."
    elog "  Requested target CPUs: ${requested}"
    elog "  Actual target CPUs: ${actual}"
    return 1
  fi
}

# Configures and builds each target specified by $1, and then builds
# VPX.framework.
build_framework() {
  local lib_list=""
  local targets="$1"
  local target=""
  local target_dist_dir=""

  # Clean up from previous build(s).
  rm -rf "${BUILD_ROOT}" "${FRAMEWORK_DIR}"

  # Create output dirs.
  mkdir -p "${BUILD_ROOT}"
  mkdir -p "${HEADER_DIR}"

  cd "${BUILD_ROOT}"

  for target in ${targets}; do
    build_target "${target}"
    target_dist_dir="${BUILD_ROOT}/${target}/${DIST_DIR}"
    if [ "${ENABLE_SHARED}" = "yes" ]; then
      local suffix="dylib"
    else
      local suffix="a"
    fi
    lib_list="${lib_list} ${target_dist_dir}/lib/libvpx.${suffix}"
  done

  cd "${ORIG_PWD}"

  # The basic libvpx API includes are all the same; just grab the most recent
  # set.
  cp -p "${target_dist_dir}"/include/vpx/* "${HEADER_DIR}"

  # Build the fat library.
  ${LIPO} -create ${lib_list} -output ${FRAMEWORK_DIR}/VPX

  # Create the vpx_config.h shim that allows usage of vpx_config.h from
  # within VPX.framework.
  create_vpx_framework_config_shim "${targets}"

  # Copy in vpx_version.h.
  cp -p "${BUILD_ROOT}/${target}/vpx_version.h" "${HEADER_DIR}"

  if [ "${ENABLE_SHARED}" = "yes" ]; then
    # Adjust the dylib's name so dynamic linking in apps works as expected.
    install_name_tool -id '@rpath/VPX.framework/VPX' ${FRAMEWORK_DIR}/VPX

    # Copy in Info.plist.
    cat "${SCRIPT_DIR}/ios-Info.plist" \
      | sed "s/\${FULLVERSION}/${FULLVERSION}/g" \
      | sed "s/\${VERSION}/${VERSION}/g" \
      | sed "s/\${IOS_VERSION_MIN}/${IOS_VERSION_MIN}/g" \
      > "${FRAMEWORK_DIR}/Info.plist"
  fi

  # Confirm VPX.framework/VPX contains the targets requested.
  verify_framework_targets ${targets}

  vlog "Created fat library ${FRAMEWORK_LIB} containing:"
  for lib in ${lib_list}; do
    vlog "  $(echo ${lib} | awk -F / '{print $2, $NF}')"
  done
}

# Trap function. Cleans up the subtree used to build all targets contained in
# $TARGETS.
cleanup() {
  local res=$?
  cd "${ORIG_PWD}"

  if [ $res -ne 0 ]; then
    elog "build exited with error ($res)"
  fi

  if [ "${PRESERVE_BUILD_OUTPUT}" != "yes" ]; then
    rm -rf "${BUILD_ROOT}"
  fi
}

print_list() {
  local indent="$1"
  shift
  local list="$@"
  for entry in ${list}; do
    echo "${indent}${entry}"
  done
}

iosbuild_usage() {
cat << EOF
  Usage: ${0##*/} [arguments]
    --help: Display this message and exit.
    --enable-shared: Build a dynamic framework for use on iOS 8 or later.
    --extra-configure-args <args>: Extra args to pass when configuring libvpx.
    --macosx: Uses darwin16 targets instead of iphonesimulator targets for x86
              and x86_64. Allows linking to framework when builds target MacOSX
              instead of iOS.
    --preserve-build-output: Do not delete the build directory.
    --show-build-output: Show output from each library build.
    --targets <targets>: Override default target list. Defaults:
$(print_list "        " ${TARGETS})
    --test-link: Confirms all targets can be linked. Functionally identical to
                 passing --enable-examples via --extra-configure-args.
    --verbose: Output information about the environment and each stage of the
               build.
EOF
}

elog() {
  echo "${0##*/} failed because: $@" 1>&2
}

vlog() {
  if [ "${VERBOSE}" = "yes" ]; then
    echo "$@"
  fi
}

trap cleanup EXIT

# Parse the command line.
while [ -n "$1" ]; do
  case "$1" in
    --extra-configure-args)
      EXTRA_CONFIGURE_ARGS="$2"
      shift
      ;;
    --help)
      iosbuild_usage
      exit
      ;;
    --enable-shared)
      ENABLE_SHARED=yes
      ;;
    --preserve-build-output)
      PRESERVE_BUILD_OUTPUT=yes
      ;;
    --show-build-output)
      devnull=
      ;;
    --test-link)
      EXTRA_CONFIGURE_ARGS="${EXTRA_CONFIGURE_ARGS} --enable-examples"
      ;;
    --targets)
      TARGETS="$2"
      shift
      ;;
    --macosx)
      TARGETS="${ARM_TARGETS} ${OSX_TARGETS}"
      ;;
    --verbose)
      VERBOSE=yes
      ;;
    *)
      iosbuild_usage
      exit 1
      ;;
  esac
  shift
done

if [ "${ENABLE_SHARED}" = "yes" ]; then
  CONFIGURE_ARGS="--enable-shared ${CONFIGURE_ARGS}"
fi

FULLVERSION=$("${SCRIPT_DIR}"/version.sh --bare "${LIBVPX_SOURCE_DIR}")
VERSION=$(echo "${FULLVERSION}" | sed -E 's/^v([0-9]+\.[0-9]+\.[0-9]+).*$/\1/')

if [ "$ENABLE_SHARED" = "yes" ]; then
  IOS_VERSION_OPTIONS="--enable-shared"
  IOS_VERSION_MIN="8.0"
else
  IOS_VERSION_OPTIONS=""
  IOS_VERSION_MIN="7.0"
fi

if [ "${VERBOSE}" = "yes" ]; then
cat << EOF
  BUILD_ROOT=${BUILD_ROOT}
  DIST_DIR=${DIST_DIR}
  CONFIGURE_ARGS=${CONFIGURE_ARGS}
  EXTRA_CONFIGURE_ARGS=${EXTRA_CONFIGURE_ARGS}
  FRAMEWORK_DIR=${FRAMEWORK_DIR}
  FRAMEWORK_LIB=${FRAMEWORK_LIB}
  HEADER_DIR=${HEADER_DIR}
  LIBVPX_SOURCE_DIR=${LIBVPX_SOURCE_DIR}
  LIPO=${LIPO}
  MAKEFLAGS=${MAKEFLAGS}
  ORIG_PWD=${ORIG_PWD}
  PRESERVE_BUILD_OUTPUT=${PRESERVE_BUILD_OUTPUT}
  TARGETS="$(print_list "" ${TARGETS})"
  ENABLE_SHARED=${ENABLE_SHARED}
  OSX_TARGETS="${OSX_TARGETS}"
  SIM_TARGETS="${SIM_TARGETS}"
  SCRIPT_DIR="${SCRIPT_DIR}"
  FULLVERSION="${FULLVERSION}"
  VERSION="${VERSION}"
  IOS_VERSION_MIN="${IOS_VERSION_MIN}"
EOF
fi

build_framework "${TARGETS}"
echo "Successfully built '${FRAMEWORK_DIR}' for:"
print_list "" ${TARGETS}
