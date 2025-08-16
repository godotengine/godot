cmake_minimum_required(VERSION 3.1.0)
project(hello_world CXX)

option(MINIMAL     "Minimal build" OFF)
option(DEBUGGING   "Debugging build" OFF)
option(GCSECTIONS  "Garbage collect empty sections" ON)

set(WARNINGS  "-Wall -Wextra")
set(COMMON    "-O2")
set(CMAKE_CXX_FLAGS "${WARNINGS} -std=c++17 ${COMMON}")

if (DEBUGGING)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
elseif (MINIMAL)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Os")
endif()

if (GCSECTIONS)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffunction-sections -fdata-sections")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-gc-sections")
endif()

set(SOURCES
	src/hello_world.cpp
	src/test_threads.cpp
)

add_executable(hello_world ${SOURCES})
target_link_libraries(hello_world -static pthread)

# place ELF into the top binary folder
set_target_properties(hello_world
	PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
