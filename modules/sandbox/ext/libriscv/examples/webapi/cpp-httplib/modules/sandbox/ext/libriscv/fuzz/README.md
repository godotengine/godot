# libriscv fuzzing system

## Building

To build all the available fuzzers run:
```
./fuzzer.sh
```

All the fuzzers are built in the build folder:
```
la -ls build/*fuzzer*
```
There are 3 fuzzers of each kind. A 32-bit, a 64--bit and a 128-bit fuzzer. Normally we would try to fuzz as much as possible, however the fuzzer is ineffective when it fuzzes different machines and loaders at the same time.

## Fuzzing

Example starting a fuzzer:
```
./build/vmfuzzer32 -N4 -handle_fpe=0
```
You may want to make sure coredumps are enabled, through ASAN_OPTIONS:
```
export ASAN_OPTIONS=disable_coredump=0::unmap_shadow_on_exit=1::handle_segv=0::handle_sigfpe=0
```

[libfuzzer](https://llvm.org/docs/LibFuzzer.html) is being employed, which is a part of LLVM-Clang.
