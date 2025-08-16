cmake_minimum_required(VERSION 3.9)
project(webapi CXX)

option(LTO         "Enable interprocedural optimizations" ON)
option(NATIVE      "Enable native instructions" ON)

add_subdirectory(../../lib libriscv)

set(SOURCES
	server.cpp
	compile.cpp
	execute.cpp
)

add_executable(webapi ${SOURCES})
set_target_properties(webapi PROPERTIES CXX_STANDARD 17)
target_link_libraries(webapi riscv pthread m)
target_include_directories(webapi PRIVATE "cpp-httplib")
target_include_directories(webapi PRIVATE ${CMAKE_BINARY_DIR})
target_include_directories(webapi PRIVATE ${CMAKE_SOURCE_DIR}/../emulator/src)
target_compile_definitions(webapi PRIVATE
			"CPPHTTPLIB_KEEPALIVE_TIMEOUT_SECOND=30"
			"CPPHTTPLIB_READ_TIMEOUT_SECOND=30"
			"CPPHTTPLIB_THREAD_POOL_COUNT=16")

if (LTO)
	include(CheckIPOSupported)
	check_ipo_supported(RESULT supported OUTPUT error)
	if (supported)
		message(STATUS "IPO / LTO enabled")
		set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
	else()
		message(STATUS "IPO / LTO not supported: <${error}>")
	endif()
endif()
if (NATIVE)
	target_compile_options(riscv PUBLIC "-march=native")
endif()
target_compile_options(riscv PUBLIC -Ofast)
