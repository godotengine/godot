#!/bin/bash

PROFILE=Release

if [ ! -z $1 ]; then
	PROFILE=$1
fi

if [ ! -z $2 ]; then
	NPROC="--parallel $2"
fi

echo "Building glslang."
mkdir -p external/glslang-build
cd external/glslang-build
cmake ../glslang -DCMAKE_BUILD_TYPE=$PROFILE -DCMAKE_INSTALL_PREFIX=output
cmake --build . --config $PROFILE --target install ${NPROC}
cd ../..

echo "Building SPIRV-Tools."
mkdir -p external/spirv-tools-build
cd external/spirv-tools-build
cmake ../spirv-tools -DCMAKE_BUILD_TYPE=$PROFILE -DSPIRV_WERROR=OFF -DCMAKE_INSTALL_PREFIX=output
cmake --build . --config $PROFILE --target install ${NPROC}
cd ../..

