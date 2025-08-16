cmake_minimum_required(VERSION 3.11.0)
project(builder C)
#set (CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(LTO         "Link-time optimizations" ON)
option(GCSECTIONS  "Garbage collect empty sections" ON)
option(DEBUGGING   "Add debugging information" OFF)

#
# Build configuration
#
if (GCC_TRIPLE STREQUAL "riscv32-linux-gnu")
	set(RISCV_ABI "-march=rv32gc -mabi=ilp32d")
else()
	set(RISCV_ABI "-march=rv64gc -mabi=lp64d")
endif()
set(WARNINGS  "-Wall -Wextra -Werror=return-type -Wno-unused -Wno-strict-aliasing -Wno-parentheses")
set(COMMON    "-O2 -fno-math-errno")
if (DEBUGGING)
	set (COMMON "${COMMON} -ggdb3 -O0")
endif()
set(FLAGS "${WARNINGS} ${RISCV_ABI} ${COMMON}")

if (LTO)
	set(FLAGS "${FLAGS} -flto")
endif()

if (GCSECTIONS)
	set(FLAGS "${FLAGS} -ffunction-sections -fdata-sections")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-gc-sections")
endif()

set(CMAKE_C_FLAGS   "-std=c11 ${FLAGS}")
set(BUILD_SHARED_LIBS OFF)

function (add_micronim_binary NAME)
	add_executable(${NAME} ${ARGN}
		env/stdio.c
	)
	target_link_libraries(${NAME} -static)
	target_include_directories(${NAME} PRIVATE "env")
	# Strip most symbols when not debugging
	if (NOT DEBUGGING)
		target_link_libraries(${NAME} -Wl,-S,-x)
	endif()
	file(WRITE "${CMAKE_BINARY_DIR}/program.txt" ${NAME})
endfunction()
