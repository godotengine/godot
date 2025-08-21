cmake_minimum_required(VERSION 3.9.4)
project(riscv)

# Disable memory traps and 32-bit
option(RISCV_32I "" OFF)
option(RISCV_MEMORY_TRAPS "" OFF)
add_subdirectory(.. riscv)

add_executable(test test.c)
target_link_libraries(test riscv_capi)
