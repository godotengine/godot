#! /usr/bin/env bash

# Copyright (c) 2018-2022 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Released under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.

################################################################################
# Thrust and CUB build script for gpuCI
################################################################################

set -e # Stop on errors.

# append variable value
# Appends ${value} to ${variable}, adding a space before ${value} if
# ${variable} is not empty.
function append {
  tmp="${!1:+${!1} }${2}"
  eval "${1}=\${tmp}"
}

# log args...
# Prints out ${args[*]} with a gpuCI log prefix and a newline before and after.
function log() {
  printf "\n>>>> %s\n\n" "${*}"
}

# print_with_trailing_blank_line args...
# Prints ${args[*]} with one blank line following, preserving newlines within
# ${args[*]} but stripping any preceding ${args[*]}.
function print_with_trailing_blank_line {
  printf "%s\n\n" "${*}"
}

# echo_and_run name args...
# Echo ${args[@]}, then execute ${args[@]}
function echo_and_run {
  echo "${1}: ${@:2}"
  ${@:2}
}

# echo_and_run_timed name args...
# Echo ${args[@]}, then execute ${args[@]} and report how long it took,
# including ${name} in the output of the time.
function echo_and_run_timed {
  echo "${@:2}"
  TIMEFORMAT=$'\n'"${1} Time: %lR"
  time ${@:2}
}

# join_delimit <delimiter> [value [value [...]]]
# Combine all values into a single string, separating each by a single character
# delimiter. Eg:
# foo=(bar baz kramble)
# joined_foo=$(join_delimit "|" "${foo[@]}")
# echo joined_foo # "bar|baz|kramble"
function join_delimit {
  local IFS="${1}"
  shift
  echo "${*}"
}

################################################################################
# VARIABLES - Set up bash and environmental variables.
################################################################################

# Get the variables the Docker container set up for us: ${CXX}, ${CUDACXX}, etc.
set +e # Don't stop on errors from /etc/cccl.bashrc.
source /etc/cccl.bashrc
set -e # Stop on errors.

# Configure sccache.
if [[ "${CXX_TYPE}" == "nvcxx" ]]; then
  log "Disabling sccache (nvcxx not supported)"
  unset ENABLE_SCCACHE
elif [[ "${BUILD_MODE}" == "pull-request" || "${BUILD_MODE}" == "branch" ]]; then
  # gpuCI builds cache in S3.
  export ENABLE_SCCACHE="gpuCI"
  # Change to 'thrust-aarch64' if we add aarch64 builds to gpuCI:
  export SCCACHE_S3_KEY_PREFIX=thrust-linux64 # [linux64]
  export SCCACHE_BUCKET=rapids-sccache
  export SCCACHE_REGION=us-west-2
  export SCCACHE_IDLE_TIMEOUT=32768
else
  export ENABLE_SCCACHE="local"
  # local builds cache locally
  export SCCACHE_DIR="${WORKSPACE}/build-sccache"
fi

# Set sccache compiler flags
if [[ -n "${ENABLE_SCCACHE}" ]]; then
  export CMAKE_CUDA_COMPILER_LAUNCHER="sccache"
  export CMAKE_CXX_COMPILER_LAUNCHER="sccache"
  export CMAKE_C_COMPILER_LAUNCHER="sccache"
fi

# Set path.
export PATH=/usr/local/cuda/bin:${PATH}

# Set home to the job's workspace.
export HOME=${WORKSPACE}

# Per-process memory util logs:
MEMMON_LOG=${WORKSPACE}/build/memmon_log

# Switch to the build directory.
cd ${WORKSPACE}
mkdir -p build
cd build

# Remove any old .ninja_log file so the PrintNinjaBuildTimes step is accurate:
rm -f .ninja_log

if [[ -z "${CMAKE_BUILD_TYPE}" ]]; then
  CMAKE_BUILD_TYPE="Release"
fi

CMAKE_BUILD_FLAGS="--"

# The Docker image sets up `${CXX}` and `${CUDACXX}`.
append CMAKE_FLAGS "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
append CMAKE_FLAGS "-DCMAKE_CUDA_COMPILER='${CUDACXX}'"

if [[ "${CXX_TYPE}" == "nvcxx" ]]; then
  # NVC++ isn't properly detected by CMake, so we have to tell CMake to ignore
  # detection and explicit provide the compiler ID. Ninja currently isn't
  # supported, so we just use makefiles.
  append CMAKE_FLAGS "-DCMAKE_CUDA_COMPILER_FORCED=ON"
  append CMAKE_FLAGS "-DCMAKE_CUDA_COMPILER_ID=NVCXX"
  # We use NVC++ "slim" image which only contain a single CUDA toolkit version.
  # When using NVC++ in an environment without GPUs (like our CPU-only
  # builders) it unfortunately defaults to the oldest CUDA toolkit version it
  # supports, even if that version is not in the image. So, we have to
  # explicitly tell NVC++ it which CUDA toolkit version to use.
  CUDA_VER=$(echo ${SDK_VER} | sed 's/.*\(cuda[0-9]\+\.[0-9]\+\)/\1/')
  append CMAKE_FLAGS "-DCMAKE_CUDA_FLAGS=-gpu=${CUDA_VER}"
  # Don't stop on build failures.
  append CMAKE_BUILD_FLAGS "-k"
else
  if [[ "${CXX_TYPE}" == "icc" ]]; then
    # Only the latest version of the Intel C++ compiler, which NVCC doesn't
    # officially support yet, is freely available.
    append CMAKE_FLAGS "-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler"
  fi
  # We're using NVCC so we need to set the host compiler.
  append CMAKE_FLAGS "-DCMAKE_CXX_COMPILER='${CXX}'"
  append CMAKE_FLAGS "-G Ninja"
  # Don't stop on build failures.
  append CMAKE_BUILD_FLAGS "-k0"
fi

DETERMINE_PARALLELISM_FLAGS=""

# Used to limit the number of default build threads. Any build/link
# steps that exceed this limit will cause this script to report a
# failure. Tune this using the memmon logs printed after each run.
#
# Build steps that take more memory than this limit should
# be split into multiple steps/translation units. Any temporary
# increases to this threshold should be reverted ASAP. The goal
# to do decrease this as much as possible and not increase it.
if [[ -z "${MIN_MEMORY_PER_THREAD}" ]]; then
  if [[ "${CXX_TYPE}" == "nvcxx" ]]; then
      MIN_MEMORY_PER_THREAD=3.0 # GiB
  elif [[ "${CXX_TYPE}" == "icc" ]]; then
      MIN_MEMORY_PER_THREAD=2.5 # GiB
  else
      MIN_MEMORY_PER_THREAD=2.0 # GiB
  fi
fi
append DETERMINE_PARALLELISM_FLAGS "--min-memory-per-thread ${MIN_MEMORY_PER_THREAD}"

if [[ -n "${PARALLEL_LEVEL}" ]]; then
  append DETERMINE_PARALLELISM_FLAGS "-j ${PARALLEL_LEVEL}"
fi

# COVERAGE_PLAN options:
# * Exhaustive
# * Thorough
# * Minimal
if [[ -z "${COVERAGE_PLAN}" ]]; then
  # `ci/local/build.bash` always sets a coverage plan, so we can assume we're
  # in gpuCI if one was not set.
  if [[ "${CXX_TYPE}" == "nvcxx" ]]; then
    # Today, NVC++ builds take too long to do anything more than Minimal.
    COVERAGE_PLAN="Minimal"
  elif [[ "${BUILD_TYPE}" == "cpu" ]] && [[ "${BUILD_MODE}" == "branch" ]]; then
    # Post-commit CPU CI builds.
    COVERAGE_PLAN="Exhaustive"
  elif [[ "${BUILD_TYPE}" == "cpu" ]]; then
    # Pre-commit CPU CI builds.
    COVERAGE_PLAN="Thorough"
  elif [[ "${BUILD_TYPE}" == "gpu" ]]; then
    # Pre- and post-commit GPU CI builds.
    COVERAGE_PLAN="Minimal"
  fi
fi

case "${COVERAGE_PLAN}" in
  Exhaustive)
    append CMAKE_FLAGS "-DTHRUST_ENABLE_MULTICONFIG=ON"
    append CMAKE_FLAGS "-DTHRUST_IGNORE_DEPRECATED_CPP_11=ON"
    append CMAKE_FLAGS "-DTHRUST_MULTICONFIG_ENABLE_DIALECT_ALL=ON"
    append CMAKE_FLAGS "-DTHRUST_MULTICONFIG_ENABLE_SYSTEM_CPP=ON"
    append CMAKE_FLAGS "-DTHRUST_MULTICONFIG_ENABLE_SYSTEM_TBB=ON"
    append CMAKE_FLAGS "-DTHRUST_MULTICONFIG_ENABLE_SYSTEM_OMP=ON"
    append CMAKE_FLAGS "-DTHRUST_MULTICONFIG_ENABLE_SYSTEM_CUDA=ON"
    append CMAKE_FLAGS "-DTHRUST_INCLUDE_CUB_CMAKE=ON"
    append CMAKE_FLAGS "-DTHRUST_MULTICONFIG_WORKLOAD=LARGE"
    ;;
  Thorough)
    # Build the legacy bench.cu. We'll probably want to remove this when we
    # switch to the new, heavier thrust_benchmarks project.
    append CMAKE_FLAGS "-DTHRUST_ENABLE_BENCHMARKS=ON"
    append CMAKE_FLAGS "-DTHRUST_ENABLE_MULTICONFIG=ON"
    append CMAKE_FLAGS "-DTHRUST_IGNORE_DEPRECATED_CPP_11=ON"
    append CMAKE_FLAGS "-DTHRUST_MULTICONFIG_ENABLE_DIALECT_ALL=ON"
    append CMAKE_FLAGS "-DTHRUST_MULTICONFIG_ENABLE_SYSTEM_CPP=ON"
    append CMAKE_FLAGS "-DTHRUST_MULTICONFIG_ENABLE_SYSTEM_TBB=ON"
    append CMAKE_FLAGS "-DTHRUST_MULTICONFIG_ENABLE_SYSTEM_OMP=ON"
    append CMAKE_FLAGS "-DTHRUST_MULTICONFIG_ENABLE_SYSTEM_CUDA=ON"
    append CMAKE_FLAGS "-DTHRUST_MULTICONFIG_WORKLOAD=SMALL"
    append CMAKE_FLAGS "-DTHRUST_INCLUDE_CUB_CMAKE=ON"
    append CMAKE_FLAGS "-DTHRUST_AUTO_DETECT_COMPUTE_ARCHS=ON"
    if [[ "${CXX_TYPE}" != "nvcxx" ]]; then
      # NVC++ can currently only target one compute architecture at a time.
      append CMAKE_FLAGS "-DTHRUST_ENABLE_COMPUTE_50=ON"
      append CMAKE_FLAGS "-DTHRUST_ENABLE_COMPUTE_60=ON"
      append CMAKE_FLAGS "-DTHRUST_ENABLE_COMPUTE_70=ON"
    fi
    append CMAKE_FLAGS "-DTHRUST_ENABLE_COMPUTE_80=ON"
    ;;
  Minimal)
    append CMAKE_FLAGS "-DTHRUST_ENABLE_MULTICONFIG=ON"
    append CMAKE_FLAGS "-DTHRUST_MULTICONFIG_ENABLE_DIALECT_LATEST=ON"
    append CMAKE_FLAGS "-DTHRUST_MULTICONFIG_ENABLE_SYSTEM_CPP=ON"
    append CMAKE_FLAGS "-DTHRUST_MULTICONFIG_ENABLE_SYSTEM_TBB=OFF"
    append CMAKE_FLAGS "-DTHRUST_MULTICONFIG_ENABLE_SYSTEM_OMP=OFF"
    append CMAKE_FLAGS "-DTHRUST_MULTICONFIG_ENABLE_SYSTEM_CUDA=ON"
    append CMAKE_FLAGS "-DTHRUST_MULTICONFIG_WORKLOAD=SMALL"
    append CMAKE_FLAGS "-DTHRUST_INCLUDE_CUB_CMAKE=ON"
    append CMAKE_FLAGS "-DTHRUST_AUTO_DETECT_COMPUTE_ARCHS=ON"
    if [[ "${BUILD_TYPE}" == "cpu" ]] && [[ "${CXX_TYPE}" == "nvcxx" ]]; then
      # If no GPU is automatically detected, NVC++ insists that you explicitly
      # provide an architecture.
      # TODO: This logic should really be moved into CMake, but it will be
      # tricky to do that until CMake officially supports NVC++.
      append CMAKE_FLAGS "-DTHRUST_ENABLE_COMPUTE_80=ON"
    fi
    ;;
esac

if [[ -n "${@}" ]]; then
  append CMAKE_BUILD_FLAGS "${@}"
fi

append CTEST_FLAGS "--output-on-failure"

CTEST_EXCLUSION_REGEXES=()

if [[ "${BUILD_TYPE}" == "cpu" ]]; then
  CTEST_EXCLUSION_REGEXES+=("^cub" "^thrust.*cuda")
fi

if [[ -n "${CTEST_EXCLUSION_REGEXES[@]}" ]]; then
  CTEST_EXCLUSION_REGEX=$(join_delimit "|" "${CTEST_EXCLUSION_REGEXES[@]}")
  append CTEST_FLAGS "-E ${CTEST_EXCLUSION_REGEX}"
fi

if [[ -n "${@}" ]]; then
  CTEST_INCLUSION_REGEX=$(join_delimit "|" "${@}")
  append CTEST_FLAGS "-R ^${CTEST_INCLUSION_REGEX[@]}$"
fi

# Export variables so they'll show up in the logs when we report the environment.
export COVERAGE_PLAN
export CMAKE_FLAGS
export CMAKE_BUILD_FLAGS
export CTEST_FLAGS

################################################################################
# ENVIRONMENT - Configure and print out information about the environment.
################################################################################

log "Determine system topology..."

# Set `${PARALLEL_LEVEL}` if it is unset; otherwise, this just reports the
# system topology.
source ${WORKSPACE}/ci/common/determine_build_parallelism.bash ${DETERMINE_PARALLELISM_FLAGS}

log "Get environment..."

env | sort

log "Check versions..."

# We use sed and echo below to ensure there is always one and only trailing
# line following the output from each tool.

${CXX} --version 2>&1 | sed -Ez '$ s/\n*$/\n/'

echo

${CUDACXX} --version 2>&1 | sed -Ez '$ s/\n*$/\n/'

echo

cmake --version 2>&1 | sed -Ez '$ s/\n*$/\n/'

if [[ "${BUILD_TYPE}" == "gpu" ]]; then
  echo
  nvidia-smi 2>&1 | sed -Ez '$ s/\n*$/\n/'
fi

if [[ -n "${ENABLE_SCCACHE}" ]]; then
  echo
  # Set sccache statistics to zero to capture clean run.
  sccache --version
  sccache --zero-stats | grep location
fi

################################################################################
# BUILD - Build Thrust and CUB examples and tests.
################################################################################

log "Configure Thrust and CUB..."

echo_and_run_timed "Configure" cmake .. --log-level=VERBOSE ${CMAKE_FLAGS}
configure_status=$?

log "Build Thrust and CUB..."

# ${PARALLEL_LEVEL} needs to be passed after we run
# determine_build_parallelism.bash, so it can't be part of ${CMAKE_BUILD_FLAGS}.
set +e # Don't stop on build failures.

# Monitor memory usage. Thresholds in GiB:
python3 ${WORKSPACE}/ci/common/memmon.py \
	--log-threshold 0.0 \
	--fail-threshold ${MIN_MEMORY_PER_THREAD} \
	--log-file ${MEMMON_LOG} \
        &
memmon_pid=$!

echo_and_run_timed "Build" cmake --build . ${CMAKE_BUILD_FLAGS} -j ${PARALLEL_LEVEL}
build_status=$?

# Stop memmon:
kill -s SIGINT ${memmon_pid}

# Re-enable exit on failure:
set -e

################################################################################
# TEST - Run Thrust and CUB examples and tests.
################################################################################

log "Test Thrust and CUB..."

(
  # Make sure test_status captures ctest, not tee:
  # https://stackoverflow.com/a/999259/11130318
  set -o pipefail
  echo_and_run_timed "Test" ctest ${CTEST_FLAGS} | tee ctest_log
)
test_status=$?

################################################################################
# COMPILATION STATS
################################################################################

if [[ -n "${ENABLE_SCCACHE}" ]]; then
  # Get sccache stats after the compile is completed
  COMPILE_REQUESTS=$(sccache -s | grep "Compile requests \+ [0-9]\+$" | awk '{ print $NF }')
  CACHE_HITS=$(sccache -s | grep "Cache hits \+ [0-9]\+$" | awk '{ print $NF }')
  HIT_RATE=$(echo - | awk "{printf \"%.2f\n\", $CACHE_HITS / $COMPILE_REQUESTS * 100}")
  log "sccache stats (${HIT_RATE}% hit):"
  sccache -s
fi

################################################################################
# COMPILE TIME INFO: Print the 20 longest running build steps (ninja only)
################################################################################

if [[ -f ".ninja_log" ]]; then
  log "Checking slowest build steps:"
  echo_and_run "CompileTimeInfo" cmake -P ../cmake/PrintNinjaBuildTimes.cmake | head -n 23
fi

################################################################################
# RUNTIME INFO: Print the 20 longest running test steps
################################################################################

if [[ -f "ctest_log" ]]; then
  log "Checking slowest test steps:"
  echo_and_run "TestTimeInfo" cmake -DLOGFILE=ctest_log -P ../cmake/PrintCTestRunTimes.cmake | head -n 20
fi

################################################################################
# MEMORY_USAGE
################################################################################

memmon_status=0
if [[ -f "${MEMMON_LOG}" ]]; then
  log "Checking memmon logfile: ${MEMMON_LOG}"

  if [[ -n "$(grep -E "^FAIL" ${MEMMON_LOG})" ]]; then
    log "error: Some build steps exceeded memory threshold (${MIN_MEMORY_PER_THREAD} GiB):"
    grep -E "^FAIL" ${MEMMON_LOG}
    memmon_status=1
  else
    log "Top memory usage per build step (all less than limit of ${MIN_MEMORY_PER_THREAD} GiB):"
    if [[ -s ${MEMMON_LOG} ]]; then
      # Not empty:
      head -n5 ${MEMMON_LOG}
    else
      echo "None detected above logging threshold."
    fi
  fi
fi

################################################################################
# SUMMARY - Print status of each step and exit with failure if needed.
################################################################################

log "Summary:"
echo "Warnings:"
# Not currently a failure; sccache makes these unreliable and intermittent:
echo "- Build Memory Check: ${memmon_status}"
echo "Failures:"
echo "- Configure Error Code: ${configure_status}"
echo "- Build Error Code: ${build_status}"
echo "- Test Error Code: ${test_status}"

if [[ "${configure_status}" != "0" ]] || \
   [[ "${build_status}" != "0" ]] || \
   [[ "${test_status}" != "0" ]]; then
     exit 1
fi
