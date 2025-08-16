cmake_minimum_required(VERSION 3.14)
project(noexcept LANGUAGES CXX)

add_subdirectory(../../lib riscv)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -fno-exceptions -fno-rtti -fno-unwind-tables -fno-asynchronous-unwind-tables")

add_executable(example example.cpp)
target_link_libraries(example riscv)
