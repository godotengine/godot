cmake_minimum_required(VERSION 3.9.4)
project(riscv CXX)

set(SOURCES
	main.cpp
)

# Always enable 128-bit ISA
option(RISCV_128I "" ON)
add_subdirectory(../../lib lib)

add_executable(buildconfig ${SOURCES})
target_link_libraries(buildconfig riscv)
