cmake_minimum_required(VERSION 3.9.4)
project(riscv CXX)

set(CMAKE_CXX_STANDARD 20)

set(SOURCES
	nested.cpp
)

# Link Time Optimization
include(CheckIPOSupported)
check_ipo_supported(RESULT supported OUTPUT error)
if (supported)
	message(STATUS "IPO / LTO enabled")
	set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
	message(STATUS "IPO / LTO not supported: <${error}>")
endif()

option(RISCV_EXT_C "" OFF)
option(RISCV_EXT_V "" ON)
option(RISCV_EXPERIMENTAL "" ON)
option(RISCV_ENCOMPASSING_ARENA "" ON)
set(RISCV_ENCOMPASSING_ARENA_BITS "30" CACHE STRING "Encompassing arena bits")
add_subdirectory(../../lib lib)

add_executable(nested ${SOURCES})
target_link_libraries(nested riscv -static)
