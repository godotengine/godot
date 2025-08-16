cmake_minimum_required(VERSION 3.5)
project(example LANGUAGES CXX)

find_package(PkgConfig REQUIRED)
pkg_check_modules(libriscv REQUIRED IMPORTED_TARGET libriscv)

add_executable(example example.cpp)
target_link_libraries(example PkgConfig::libriscv)
