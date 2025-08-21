cmake_minimum_required(VERSION 3.1.0)
project(hello_world C)

option(MINIMAL     "Minimal build" OFF)
option(DEBUGGING   "Debugging build" OFF)
option(GCSECTIONS  "Garbage collect empty sections" ON)

set(RISCV_ABI "-march=rv64g -mabi=lp64d")
set(WARNINGS  "-Wall -Wextra")
set(COMMON    "-O2")
set(CMAKE_C_FLAGS "${WARNINGS} ${RISCV_ABI} -std=c11 ${COMMON}")

if (DEBUGGING)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -g")
elseif (MINIMAL)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Os")
endif()

if (GCSECTIONS)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffunction-sections -fdata-sections")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-gc-sections")
endif()

set(SOURCES
	src/main.c
)

set(BUILD_SHARED_LIBS OFF)
set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS "") # remove -rdynamic

add_executable(tcpserver ${SOURCES})
target_link_libraries(tcpserver -static)
# place ELF into the top binary folder
set_target_properties(tcpserver
	PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
