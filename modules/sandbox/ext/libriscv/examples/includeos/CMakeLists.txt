cmake_minimum_required(VERSION 3.0)
project (includeos_riscvemu)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake OPTIONAL RESULT_VARIABLE HAS_CONAN)
if (NOT HAS_CONAN)
	message(FATAL_ERROR "Missing conanbuildinfo.cmake did you forget to run conan install?")
endif()
conan_basic_setup()
include(os)

set(SOURCES
    service.cpp
	server.cpp
	../src/linux.cpp
	../src/syscalls.cpp
	../src/threads.cpp
)

os_add_executable(riscvemu "RISC-V emulator in IncludeOS" ${SOURCES})
os_add_stdout(riscvemu default_stdout)
os_add_drivers(riscvemu e1000)
os_add_plugins(riscvemu autoconf) # for config.json
os_include_directories(riscvemu PRIVATE "../src")

add_subdirectory(../../lib libriscv)
target_compile_options(riscv PUBLIC -O2)
os_link_libraries(riscvemu riscv)
