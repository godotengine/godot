FOLDER=build_unittests_exp
set -e
source scripts/find_compiler.sh

mkdir -p $FOLDER
pushd $FOLDER
cmake .. -DCMAKE_BUILD_TYPE=Debug -DRISCV_MEMORY_TRAPS=ON -DRISCV_THREADED=OFF -DRISCV_TAILCALL_DISPATCH=ON -DRISCV_EXPERIMENTAL=ON
make -j4
ctest --verbose -j4 . $@
popd
