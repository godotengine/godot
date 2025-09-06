cmake_minimum_required(VERSION 3.1.0)
project(hello_world CXX)

option(MINIMAL     "Minimal build" ON)
option(DEBUGGING   "Debugging build" OFF)
option(LTO         "Link-time optimizations" ON)
option(GCSECTIONS  "Garbage collect empty sections" ON)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
	set(RISCV_ABI "-target riscv64 -march=rv64imfd")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld")
else()
	set(RISCV_ABI "-march=rv64imfd -mabi=lp64d")
endif()

set(WARNINGS  "-Wall -Wextra")
set(COMMON    "-O2 -fno-stack-protector -nostdlib -nostdinc -fno-builtin")
set(CMAKE_CXX_FLAGS "${WARNINGS} ${RISCV_ABI} -std=c++17 ${COMMON} -fno-exceptions -fno-rtti")

if (DEBUGGING)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
elseif (MINIMAL)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Os -fno-stack-protector")
endif()

if (LTO)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto")
endif()

if (GCSECTIONS)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffunction-sections -fdata-sections")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-gc-sections")
endif()


set(SOURCES
	src/hello_world.cpp
	src/start.cpp
)

set(BUILD_SHARED_LIBS OFF)
set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "") # remove -rdynamic

add_executable(hello_world ${SOURCES})
target_link_libraries(hello_world -static)
