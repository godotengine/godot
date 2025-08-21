#!/bin/bash
set -e
GCC_TRIPLE="riscv64-linux-gnu"
# Detect RISC-V compiler
export CC=$GCC_TRIPLE-gcc-10
if command -v "$GCC_TRIPLE-gcc-11" &> /dev/null; then
	export CC="$GCC_TRIPLE-gcc-11"
fi
NIMCPU="--cpu=riscv64"
NIMFILE="$PWD/${1:-hello.nim}"

mkdir -p $GCC_TRIPLE
pushd $GCC_TRIPLE

NIMCACHE=$PWD/nimcache
mkdir -p $NIMCACHE

# find nim and replace /bin/nim with /lib
NIM_LIBS=`whereis nim`
NIM_LIBS="${NIM_LIBS##*: }"
NIM_LIBS="${NIM_LIBS/bin*/lib}"

if [[ -z "${DEBUG}" ]]; then
	nim c --nimcache:$NIMCACHE $NIMCPU --colors:on --os:linux --gc:arc -d:release -c ${NIMFILE}
	jq '.compile[] [0]' $NIMCACHE/*.json > buildfiles.txt

	cmake .. -DGCC_TRIPLE=$GCC_TRIPLE -DNIM_LIBS=$NIM_LIBS -DCMAKE_BUILD_TYPE=Release -DDEBUGGING=OFF
else
	nim c --nimcache:$NIMCACHE $NIMCPU --colors:on --os:linux --gc:arc -d:useMalloc=true --debugger=native -c ${NIMFILE}
	jq '.compile[] [0]' $NIMCACHE/*.json > buildfiles.txt

	cmake .. -DGCC_TRIPLE=$GCC_TRIPLE -DNIM_LIBS=$NIM_LIBS -DCMAKE_BUILD_TYPE=Debug -DDEBUGGING=ON
fi
make -j4
popd

# print the filename
echo $GCC_TRIPLE/`cat $GCC_TRIPLE/program.txt`
