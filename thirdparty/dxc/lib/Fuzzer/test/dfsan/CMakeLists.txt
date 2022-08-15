# These tests depend on both coverage and dfsan instrumentation.

set(CMAKE_CXX_FLAGS_RELEASE
  "${LIBFUZZER_FLAGS_BASE} -O0 -fno-sanitize=all -fsanitize=dataflow")

foreach(Test ${DFSanTests})
  add_executable(LLVMFuzzer-${Test}-DFSan
    ../${Test}.cpp
    )
  target_link_libraries(LLVMFuzzer-${Test}-DFSan
    LLVMFuzzer
    )
endforeach()

