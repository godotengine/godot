cmake_minimum_required(VERSION 3.11.0)
project(rvv CXX)

option(LTO         "Link-time optimizations" ON)
option(GCSECTIONS  "Garbage collect empty sections" ON)
set(VERSION_FILE   "symbols.map" CACHE STRING "Retained symbols file")
option(STRIP_SYMBOLS "Remove all symbols except the public API" OFF)

#
# Build configuration
#
if (GCC_TRIPLE STREQUAL "riscv32-unknown-elf")
	set(RISCV_ABI "-march=rv32gv -mabi=ilp32d")
else()
	set(RISCV_ABI "-march=rv64gv -mabi=lp64d")
endif()
set(WARNINGS  "-Wall -Wextra")
set(COMMON    "-fno-math-errno -fno-threadsafe-statics")
set(COMMON    "-O2 -fno-stack-protector ${COMMON}")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
	set (COMMON "${COMMON} -ggdb3 -O0 -fstack-protector")
	set(DEBUGGING TRUE)
endif()
set(CMAKE_CXX_FLAGS "${WARNINGS} ${RISCV_ABI} -std=c++20 ${COMMON}")

if (LTO)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto -ffat-lto-objects")
endif()

if (GCSECTIONS AND NOT DEBUGGING)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffunction-sections -fdata-sections")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-gc-sections")
endif()

set(BUILD_SHARED_LIBS OFF)
set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "") # remove -rdynamic

function (add_verfile NAME VERFILE)
	set_target_properties(${NAME} PROPERTIES LINK_DEPENDS ${VERFILE})
	target_link_libraries(${NAME} "-Wl,--retain-symbols-file=${VERFILE}")
endfunction()

function (add_rvv_binary NAME)
	# the micro binary
	add_executable(${NAME} ${ARGN})
	# Add the whole libc directly as source files
	target_link_libraries(${NAME} -static)
	# strip symbols but keep public API
	if (STRIP_SYMBOLS AND NOT DEBUGGING)
		set(VERFILE "${CMAKE_CURRENT_SOURCE_DIR}/${VERSION_FILE}")
		if (EXISTS "${VERFILE}")
			add_verfile(${NAME} ${VERFILE})
		else()
			add_verfile(${NAME} ${CMAKE_SOURCE_DIR}/${VERSION_FILE})
		endif()
		add_custom_command(TARGET ${NAME} POST_BUILD
		COMMAND ${CMAKE_STRIP} --strip-debug -R .note -R .comment -- ${CMAKE_CURRENT_SOURCE_DIR}/${NAME})
	endif()
endfunction()

add_rvv_binary(simple "src/simple.cpp")
