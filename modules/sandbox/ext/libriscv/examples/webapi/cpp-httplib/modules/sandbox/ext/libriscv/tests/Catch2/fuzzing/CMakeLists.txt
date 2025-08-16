# License: Boost 1.0
# By Paul Dreik 2020

# add a library that brings in the main() function from libfuzzer
# and has all the dependencies, so the individual fuzzers can be
# added one line each.
add_library(fuzzhelper NullOStream.h NullOStream.cpp)
target_link_libraries(fuzzhelper PUBLIC Catch2::Catch2)

# use C++17 so we can get string_view
target_compile_features(fuzzhelper PUBLIC cxx_std_17)

# This should be possible to set from the outside to be oss-fuzz compatible,
# fix later. For now, target libFuzzer only.
target_link_options(fuzzhelper PUBLIC "-fsanitize=fuzzer")

foreach(fuzzer TestSpecParser XmlWriter textflow)
  add_executable(fuzz_${fuzzer} fuzz_${fuzzer}.cpp)
  target_link_libraries(fuzz_${fuzzer} PRIVATE fuzzhelper)
endforeach()
