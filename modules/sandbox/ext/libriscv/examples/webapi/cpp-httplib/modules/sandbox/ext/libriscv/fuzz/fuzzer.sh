export ASAN_OPTIONS=disable_coredump=0::unmap_shadow_on_exit=1::handle_segv=0::handle_sigfpe=0

set -e
mkdir -p build
pushd build
cmake .. -DRISCV_128I=ON
make -j4
popd

echo "Example: ./build/vmfuzzer32 -fork=1 -handle_fpe=0"
