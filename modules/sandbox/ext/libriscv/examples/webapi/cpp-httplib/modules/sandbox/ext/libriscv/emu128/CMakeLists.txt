cmake_minimum_required(VERSION 3.9)
project(emulator CXX)

option(GPROF        "Enable profiling with gprof" OFF)
option(SANITIZE     "Enable sanitizers" OFF)

option(RISCV_32I   "" OFF)
option(RISCV_64I   "" OFF)
option(RISCV_128I  "" ON)
option(RISCV_EXT_C "" OFF)
add_subdirectory(../lib lib)

set(SOURCES
	src/main.cpp
)

if (SANITIZE)
	target_compile_options(riscv PUBLIC "-fsanitize=address,undefined")
endif()
if (GPROF)
	target_compile_options(riscv PUBLIC -pg -g -fno-inline)
endif()

function (add_emulator NAME)
	add_executable(${NAME} ${SOURCES})
	target_link_libraries(${NAME} riscv)

	if (SANITIZE)
		target_link_libraries(${NAME} "-fsanitize=address,undefined")
	endif()
	if (GPROF)
		target_link_libraries(${NAME} "-pg")
	endif()

endfunction()

add_emulator(emulator)
