cmake_minimum_required(VERSION 3.1.0)
project(testsuite CXX)

option(MINIMAL     "Minimal build" OFF)
option(DEBUGGING   "Debugging build" OFF)
option(GCSECTIONS  "Garbage collect empty sections" ON)
set(VERSION_FILE   "symbols.map" CACHE STRING "Retained symbols file")

set(RISCV_ABI "-march=rv32g -mabi=ilp32d")
set(WARNINGS  "-Wall -Wextra")
set(COMMON    "-O2 -fno-math-errno -fstack-protector")
set(CMAKE_CXX_FLAGS "${WARNINGS} ${RISCV_ABI} -std=c++2a ${COMMON}")

if (DEBUGGING)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
elseif (MINIMAL)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Os")
endif()

if (GCSECTIONS)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffunction-sections -fdata-sections")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-gc-sections")
	file(STRINGS "${CMAKE_SOURCE_DIR}/${VERSION_FILE}" SYMBOLS)
	foreach(SYMBOL ${SYMBOLS})
		if (NOT ${SYMBOL} STREQUAL "")
			set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--undefined=${SYMBOL}")
		endif()
	endforeach()
endif()


set(SOURCES
	src/main.cpp
	src/floating-point.cpp
	src/special-instructions.cpp
)

set(BUILD_SHARED_LIBS OFF)
set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "") # remove -rdynamic

add_executable(testsuite ${SOURCES})
target_link_libraries(testsuite -static)

if (NOT DEBUGGING)
	# remove all symbols except the public API
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--retain-symbols-file=${CMAKE_SOURCE_DIR}/${VERSION_FILE}")
	set_target_properties(testsuite PROPERTIES LINK_DEPENDS ${CMAKE_SOURCE_DIR}/${VERSION_FILE})
endif()
