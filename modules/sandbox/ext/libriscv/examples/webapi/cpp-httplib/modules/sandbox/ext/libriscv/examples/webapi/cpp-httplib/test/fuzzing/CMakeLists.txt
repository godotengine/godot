file(GLOB HTTPLIB_CORPUS corpus/*)
add_executable(httplib-test-fuzz
    server_fuzzer.cc
    standalone_fuzz_target_runner.cpp
)
target_link_libraries(httplib-test-fuzz PRIVATE httplib)
add_test(
    NAME httplib-test-fuzz
    COMMAND httplib-test-fuzz ${HTTPLIB_CORPUS}
)
