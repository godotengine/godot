#!/bin/sh
#
# Builds the fuzzers
#
# By Paul Dreik 20200923
set -exu

CATCHROOT=$(readlink -f $(dirname $0)/..)


BUILDDIR=$CATCHROOT/build-fuzzers
mkdir -p $BUILDDIR
cd $BUILDDIR

if which /usr/lib/ccache/clang++ >/dev/null 2>&1 ; then
 CXX=/usr/lib/ccache/clang++
else
 CXX=clang++
fi

cmake $CATCHROOT \
  -DCMAKE_CXX_COMPILER=$CXX \
  -DCMAKE_CXX_FLAGS="-fsanitize=fuzzer-no-link,address,undefined -O3 -g" \
  -DCATCH_DEVELOPMENT_BUILD=On \
  -DCATCH_BUILD_EXAMPLES=Off \
  -DCATCH_BUILD_EXTRA_TESTS=Off \
  -DCATCH_BUILD_TESTING=Off \
  -DBUILD_TESTING=Off \
  -DCATCH_ENABLE_WERROR=Off \
  -DCATCH_BUILD_FUZZERS=On

cmake --build . -j $(nproc)

