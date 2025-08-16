#!/usr/bin/env bash
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 build_dir"
    exit 1
fi

BUILD_DIR=$1

# Make the build directory
rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR/out

cd $BUILD_DIR

# Build the version
git checkout $BUILD_DIR -q

cmake \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-g -Og" \
  -DBUILD_SHARED_LIBS=ON \
  -DHTTPLIB_COMPILE=ON \
  -DCMAKE_INSTALL_PREFIX=./out \
  ../..

cmake --build . --target install
cmake --build . --target clean

