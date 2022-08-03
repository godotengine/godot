#! /usr/bin/env bash

# Copyright (c) 2018-2020 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Released under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.

function usage {
  echo "Usage: ${0} [flags...]"
  echo
  echo "Examine the system topology to determine a reasonable amount of build"
  echo "parallelism."
  echo
  echo "Exported variables:"
  echo "  \${LOGICAL_CPUS}          : Logical processors (e.g. threads)."
  echo "  \${PHYSICAL_CPUS}         : Physical processors (e.g. cores)."
  echo "  \${TOTAL_MEM}             : Total system memory [GB]."
  echo "  \${MAX_THREADS_PER_CORE}  : Maximum threads per core allowed."
  echo "  \${MIN_MEMORY_PER_THREAD} : Minimum memory [GB] per thread allowed."
  echo "  \${CPU_BOUND_THREADS}     : # of build threads constrained by processors."
  echo "  \${MEM_BOUND_THREADS}     : # of build threads constrained by memory [GB]."
  echo "  \${PARALLEL_LEVEL}        : Determined # of build threads."
  echo "  \${MEM_PER_THREAD}        : Memory [GB] per build thread."
  echo
  echo "-h, -help, --help"
  echo "  Print this message."
  echo
  echo "-q, --quiet"
  echo "  Print nothing and only export variables."
  echo
  echo "-j <threads>, --jobs <threads>"
  echo "  Explicitly set the number of build threads to use."
  echo
  echo "--max-threads-per-core <threads>"
  echo "  Specify the maximum threads per core allowed (default: ${MAX_THREADS_PER_CORE} [threads/core])."
  echo
  echo "--min-memory-per-thread <gigabytes>"
  echo "  Specify the minimum memory per thread allowed (default: ${MIN_MEMORY_PER_THREAD} [GBs/thread])."

  exit -3
}

QUIET=0

export MAX_THREADS_PER_CORE=2
export MIN_MEMORY_PER_THREAD=4 # [GB]

while test ${#} != 0
do
  case "${1}" in
  -h) ;&
  -help) ;&
  --help) usage ;;
  -q) ;&
  --quiet) QUIET=1 ;;
  -j) ;&
  --jobs)
    shift # The next argument is the number of threads.
    PARALLEL_LEVEL="${1}"
    ;;
  --max-threads-per-core)
    shift # The next argument is the number of threads per core.
    MAX_THREADS_PER_CORE="${1}"
    ;;
  --min-memory-per-thread)
    shift # The next argument is the amount of memory per thread.
    MIN_MEMORY_PER_THREAD="${1}"
    ;;
  esac
  shift
done

# https://stackoverflow.com/a/23378780
if [ $(uname) == "Darwin" ]; then
  export LOGICAL_CPUS=$(sysctl -n hw.logicalcpu_max)
  export PHYSICAL_CPUS=$(sysctl -n hw.physicalcpu_max)
else
  export LOGICAL_CPUS=$(lscpu -p | egrep -v '^#' | wc -l)
  export PHYSICAL_CPUS=$(lscpu -p | egrep -v '^#' | sort -u -t, -k 2,4 | wc -l)
fi

export TOTAL_MEM=$(awk "BEGIN { printf \"%0.4g\", $(grep MemTotal /proc/meminfo | awk '{ print $2 }') / (1024 * 1024) }")

export CPU_BOUND_THREADS=$(awk "BEGIN { printf \"%.04g\", int(${PHYSICAL_CPUS} * ${MAX_THREADS_PER_CORE}) }")
export MEM_BOUND_THREADS=$(awk "BEGIN { printf \"%.04g\", int(${TOTAL_MEM} / ${MIN_MEMORY_PER_THREAD}) }")

if [[ -z "${PARALLEL_LEVEL}" ]]; then
  # Pick the smaller of the two as the default.
  if [[ "${MEM_BOUND_THREADS}" -lt "${CPU_BOUND_THREADS}" ]]; then
    export PARALLEL_LEVEL=${MEM_BOUND_THREADS}
  else
    export PARALLEL_LEVEL=${CPU_BOUND_THREADS}
  fi
else
  EXPLICIT_PARALLEL_LEVEL=1
fi

# This can be a floating point number.
export MEM_PER_THREAD=$(awk "BEGIN { printf \"%.04g\", ${TOTAL_MEM} / ${PARALLEL_LEVEL} }")

if [[ "${QUIET}" == 0 ]]; then
  echo    "Logical CPUs:           ${LOGICAL_CPUS} [threads]"
  echo    "Physical CPUs:          ${PHYSICAL_CPUS} [cores]"
  echo    "Total Mem:              ${TOTAL_MEM} [GBs]"
  echo    "Max Threads Per Core:   ${MAX_THREADS_PER_CORE} [threads/core]"
  echo    "Min Memory Per Threads: ${MIN_MEMORY_PER_THREAD} [GBs/thread]"
  echo    "CPU Bound Threads:      ${CPU_BOUND_THREADS} [threads]"
  echo    "Mem Bound Threads:      ${MEM_BOUND_THREADS} [threads]"

  echo -n "Parallel Level:         ${PARALLEL_LEVEL} [threads]"
  if [[ -n "${EXPLICIT_PARALLEL_LEVEL}" ]]; then
    echo " (explicitly set)"
  else
    echo
  fi

  echo    "Mem Per Thread:         ${MEM_PER_THREAD} [GBs/thread]"
fi

