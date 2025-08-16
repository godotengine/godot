cmake_minimum_required(VERSION 3.1.0)
project(hello_world C CXX)

option(MINIMAL     "Minimal build" ON)
option(LTO         "Link-time optimizations" ON)
option(GCSECTIONS  "Garbage collect empty sections" ON)

if (RISCV_ARCH STREQUAL "32")
	if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
		set(RISCV_ABI "-target riscv32 -march=rv32imfd -mabi=ilp32d")
		set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld")
	endif()
else()
	if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
		set(RISCV_ABI "-target riscv64 -march=rv64imfd -mabi=lp64d")
		set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld")
	endif()
endif()

set(WARNINGS  "-Wall -Wextra")
set(COMMON    "-fstack-protector")
if (NOT LIBC_USE_STDLIB)
	set(COMMON "${COMMON} -nostdlib")
else()
	#set(COMMON "${COMMON} -fno-builtin")
endif()
set(CMAKE_C_FLAGS "${WARNINGS} ${RISCV_ABI} -std=c11 ${COMMON}")
set(CMAKE_CXX_FLAGS "${WARNINGS} ${RISCV_ABI} -std=c++17 ${COMMON} -fno-exceptions -fno-rtti")

# Priority: Debug > Minimal > Release
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -ggdb3")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -ggdb3")
elseif (MINIMAL)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Os -fno-stack-protector")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Os -fno-stack-protector")
elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
endif()

if (LTO)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -flto -ffat-lto-objects")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto -ffat-lto-objects")
endif()

if (GCSECTIONS)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffunction-sections -fdata-sections")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffunction-sections -fdata-sections")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-gc-sections")
endif()


set(BUILD_SHARED_LIBS OFF)
set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "") # remove -rdynamic

add_subdirectory(ext)
add_subdirectory(libc)

function (add_micro_executable NAME)
	add_executable(${NAME} ${ARGN})
	if (LIBC_WRAP_NATIVE)
	# Accelerated wrappers for standard memory and string functions
	target_link_libraries(${NAME} -Wl,--wrap,memset,--wrap,memcpy,--wrap,memmove,--wrap,memcmp)
	target_link_libraries(${NAME} -Wl,--wrap,strlen,--wrap,strcmp,--wrap,strncmp)
	# Accelerated wrappers for heap management
	target_link_libraries(${NAME} -Wl,--wrap,malloc,--wrap,calloc,--wrap,realloc,--wrap,free)
	endif()
	# Add the whole libc directly as source files
	target_link_libraries(${NAME} -static -Wl,--whole-archive libc -Wl,--no-whole-archive)
	# Place ELF binary into the top build folder
	set_target_properties(${NAME}
		PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
	)
endfunction()

add_micro_executable(hello_world
	src/hello_world.cpp
)
