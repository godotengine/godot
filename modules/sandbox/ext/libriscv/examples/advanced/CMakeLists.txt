cmake_minimum_required(VERSION 3.14)
project(example LANGUAGES C CXX)

include(FetchContent)
FetchContent_Declare(libriscv
  GIT_REPOSITORY https://github.com/fwsGonzo/libriscv
  GIT_TAG        master
  )
FetchContent_Declare(libfmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt
  GIT_TAG        master
  )

#FetchContent_MakeAvailable(libriscv)
FetchContent_MakeAvailable(libfmt)

set(CMAKE_CXX_STANDARD 20)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../c libriscv_capi)

add_executable(cxx_example src/example.cpp)
target_link_libraries(cxx_example riscv fmt)

add_executable(c_example src/example.c)
target_link_libraries(c_example riscv_capi)
