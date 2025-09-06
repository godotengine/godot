FOLDER=build_unittests
set -e
source scripts/find_compiler.sh

mkdir -p $FOLDER
pushd $FOLDER
cmake .. -DCMAKE_BUILD_TYPE=Debug -DRISCV_MEMORY_TRAPS=ON -DRISCV_THREADED=OFF -DRISCV_FLAT_RW_ARENA=OFF
make -j4
ctest --verbose -j4 . $@
popd
