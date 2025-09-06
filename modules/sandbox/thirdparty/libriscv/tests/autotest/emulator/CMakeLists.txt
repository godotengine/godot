cmake_minimum_required(VERSION 3.9)
project(emulator CXX)

add_subdirectory(../../../lib lib)

set(SOURCES
	src/main.cpp
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld")
endif()

set(NAME emulator)
add_executable(${NAME} ${SOURCES})
target_link_libraries(${NAME} riscv)
target_include_directories(${NAME} PRIVATE .)
