export ASAN_OPTIONS=disable_coredump=0::unmap_shadow_on_exit=1::handle_segv=0::handle_sigfpe=0

./build/vmfuzzer32 -handle_fpe=0 $@
#./build/vmfuzzer64 -handle_fpe=0 $@
