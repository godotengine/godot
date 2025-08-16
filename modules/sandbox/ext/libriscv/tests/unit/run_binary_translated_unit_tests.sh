FOLDER=build_tr
set -e
source scripts/find_compiler.sh
#export RCC="riscv64-unknown-elf-gcc"
#export RCXX="riscv64-unknown-elf-g++"


mkdir -p $FOLDER
pushd $FOLDER
cmake .. -DCMAKE_BUILD_TYPE=Debug -DRISCV_BINARY_TRANSLATION=ON -DRISCV_EXT_C=ON -DRISCV_MEMORY_TRAPS=ON -DRISCV_THREADED=ON
make -j4
ctest --verbose -j4 . $@
popd
