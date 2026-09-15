##  Copyright (c) 2018 The WebM project authors. All Rights Reserved.
##
##  Use of this source code is governed by a BSD-style license
##  that can be found in the LICENSE file in the root of the source
##  tree. An additional intellectual property rights grant can be found
##  in the file PATENTS.  All contributing project authors may
##  be found in the AUTHORS file in the root of the source tree.
##
##  Sourcing this file sets environment variables to simplify setting up
##  sanitizer builds and testing.

sanitizer="${1}"

case "${sanitizer}" in
  address) ;;
  cfi) ;;
  integer) ;;
  memory) ;;
  thread) ;;
  undefined) ;;
  clear)
    echo "Clearing environment:"
    set -x
    unset CC CXX LD AR
    unset CFLAGS CXXFLAGS LDFLAGS
    unset ASAN_OPTIONS MSAN_OPTIONS TSAN_OPTIONS UBSAN_OPTIONS
    set +x
    return
    ;;
  *)
    echo "Usage: source set_analyzer_env.sh [<sanitizer>|clear]"
    echo "  Supported sanitizers:"
    echo "    address cfi integer memory thread undefined"
    return 1
    ;;
esac

if [ ! $(which clang) ]; then
  # TODO(johannkoenig): Support gcc analyzers.
  echo "ERROR: 'clang' must be in your PATH"
  return 1
fi

# Warnings.
if [ "${sanitizer}" = "undefined" -o "${sanitizer}" = "integer" ]; then
  echo "WARNING: When building the ${sanitizer} sanitizer for 32 bit targets"
  echo "you must run:"
  echo "export LDFLAGS=\"\${LDFLAGS} --rtlib=compiler-rt -lgcc_s\""
  echo "See http://llvm.org/bugs/show_bug.cgi?id=17693 for details."
fi

if [ "${sanitizer}" = "undefined" ]; then
  major_version=$(clang --version | head -n 1 \
    | grep -o -E "[[:digit:]]\.[[:digit:]]\.[[:digit:]]" | cut -f1 -d.)
  if [ ${major_version} -eq 5 ]; then
    echo "WARNING: clang v5 has a problem with vp9 x86_64 high bit depth"
    echo "configurations. It can take ~40 minutes to compile"
    echo "vpx_dsp/x86/fwd_txfm_sse2.c"
    echo "clang v4 did not have this issue."
  fi
fi

echo "It is recommended to configure with '--enable-debug' to improve stack"
echo "traces. On mac builds, run 'dysmutil' on the output binaries (vpxenc,"
echo "test_libvpx, etc) to link the stack traces to source code lines."

# Build configuration.
cflags="-fsanitize=${sanitizer}"
ldflags="-fsanitize=${sanitizer}"

# Useful backtraces.
cflags="${cflags} -fno-omit-frame-pointer"
# Exact backtraces.
cflags="${cflags} -fno-optimize-sibling-calls"

case "${sanitizer}" in
  cfi)
    # https://clang.llvm.org/docs/ControlFlowIntegrity.html
    cflags="${cflags} -fno-sanitize-trap=cfi -flto -fvisibility=hidden"
    ldflags="${ldflags} -fno-sanitize-trap=cfi -flto -fuse-ld=gold"
    export AR="llvm-ar"
    ;;
  integer|undefined)
    # https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html
    cflags="${cflags} -fsanitize=float-cast-overflow"
    ;;
esac

set -x
export CC="clang"
export CXX="clang++"
export LD="clang++"

export CFLAGS="${cflags}"
export CXXFLAGS="${cflags}"
export LDFLAGS="${ldflags}"
set +x

# Execution configuration.
sanitizer_options=""
sanitizer_options="${sanitizer_options}:handle_segv=1"
sanitizer_options="${sanitizer_options}:handle_abort=1"
sanitizer_options="${sanitizer_options}:handle_sigfpe=1"
sanitizer_options="${sanitizer_options}:fast_unwind_on_fatal=1"
sanitizer_options="${sanitizer_options}:allocator_may_return_null=1"

case "${sanitizer}" in
  address)
    sanitizer_options="${sanitizer_options}:detect_stack_use_after_return=1"
    sanitizer_options="${sanitizer_options}:max_uar_stack_size_log=17"
    set -x
    export ASAN_OPTIONS="${sanitizer_options}"
    set +x
    ;;
  cfi)
    # No environment settings
    ;;
  memory)
    set -x
    export MSAN_OPTIONS="${sanitizer_options}"
    set +x
    ;;
  thread)
    # The thread sanitizer uses an entirely independent set of options.
    set -x
    export TSAN_OPTIONS="halt_on_error=1"
    set +x
    ;;
  undefined|integer)
    sanitizer_options="${sanitizer_options}:print_stacktrace=1"
    set -x
    export UBSAN_OPTIONS="${sanitizer_options}"
    set +x
    ;;
esac
