cmake_minimum_required(VERSION 3.14)
project(example LANGUAGES CXX)

include(FetchContent)
FetchContent_Declare(libriscv
  GIT_REPOSITORY https://github.com/fwsGonzo/libriscv
  GIT_TAG        master
  )

FetchContent_MakeAvailable(libriscv)

add_executable(example example.cpp)
target_link_libraries(example riscv)
