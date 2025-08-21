#
# libriscv RISC-V emulator library
#   C API
#

add_subdirectory(../lib libriscv)

add_library(riscv_capi STATIC libriscv.cpp libriscv.h)
target_include_directories(riscv_capi PUBLIC .)
target_link_libraries(riscv_capi PRIVATE riscv)
