cmake_minimum_required(VERSION 3.14)
project(example LANGUAGES CXX)

option(LTO         "Link-time optimizations" ON)
option(STRIP_SYMBOLS "Remove all symbols except the public API" ON)

#
# Build configuration
#
set(WARNINGS  "-Wall -Wextra -ggdb3")
set(COMMON    "-fno-math-errno -fno-threadsafe-statics")
set(COMMON    "-fno-stack-protector ${COMMON}")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
	set (COMMON "${COMMON} -ggdb3 -O0 -fstack-protector")
	set(DEBUGGING TRUE)
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${WARNINGS} -std=gnu++20 ${COMMON}")

if (LTO)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto -ffat-lto-objects")
endif()

if (GCSECTIONS AND NOT DEBUGGING)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffunction-sections -fdata-sections")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-gc-sections")
endif()

set(BUILD_SHARED_LIBS OFF)
set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "") # remove -rdynamic


function (add_micro_binary NAME ORG)
	# the micro binary
	add_executable(${NAME} ${ARGN} env.cpp)
	# Add the whole libc directly as source files
	target_link_libraries(${NAME} -static)
	target_link_libraries(${NAME} "-Wl,-Ttext-segment=${ORG}") # --image-base on LLVM
	target_link_libraries(${NAME} "-Wl,--undefined=fast_exit")
	target_link_libraries(${NAME} "-Wl,--wrap=memcpy,--wrap=memset,--wrap=memcmp,--wrap=memmove")
	target_link_libraries(${NAME} "-Wl,--wrap=malloc,--wrap=free,--wrap=calloc,--wrap=realloc")
	target_link_libraries(${NAME} "-Wl,--wrap=strlen,--wrap=strcmp,--wrap=strncmp")
	# place ELF into the sub-projects source folder
	set_target_properties(${NAME}
		PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
	)
	# strip symbols
	if (STRIP_SYMBOLS AND NOT DEBUGGING)
		target_link_libraries(${NAME} "-Wl,-x,-S")
	endif()

endfunction()


add_micro_binary(micro 0x400000
	program.cpp
)
