cmake_minimum_required(VERSION 3.9)
project(riscv CXX)

set(SOURCES
	main.cpp
)
add_executable(elfverify ${SOURCES})

# Enable FCSR emulation
option(RISCV_FCSR "" ON)

add_subdirectory(../../lib libriscv)
target_link_libraries(elfverify PRIVATE riscv)
