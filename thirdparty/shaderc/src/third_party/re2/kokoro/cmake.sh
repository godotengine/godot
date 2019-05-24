#!/bin/bash
set -eux

cd git/re2

case "${KOKORO_JOB_NAME}" in
  */windows-*)
    CMAKE_G_A_FLAGS=('-G' 'Visual Studio 14 2015' '-A' 'x64')
    ;;
  *)
    CMAKE_G_A_FLAGS=()
    # Work around a bug in older versions of bash. :/
    set +u
    ;;
esac

cmake -D CMAKE_BUILD_TYPE=Debug "${CMAKE_G_A_FLAGS[@]}" .
cmake --build . --config Debug --clean-first
ctest -C Debug --output-on-failure -E 'dfa|exhaustive|random'

cmake -D CMAKE_BUILD_TYPE=Release "${CMAKE_G_A_FLAGS[@]}" .
cmake --build . --config Release --clean-first
ctest -C Release --output-on-failure -E 'dfa|exhaustive|random'

exit 0
