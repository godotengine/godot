cmake_minimum_required(VERSION 3.16)
project(CatchCoverageHelper LANGUAGES CXX)

add_executable(CoverageHelper coverage-helper.cpp)
target_compile_features(CoverageHelper PRIVATE cxx_std_11)
if(MSVC)
  target_compile_options(CoverageHelper PRIVATE /W4 /w44265 /WX /w44061 /w44062)
endif()
