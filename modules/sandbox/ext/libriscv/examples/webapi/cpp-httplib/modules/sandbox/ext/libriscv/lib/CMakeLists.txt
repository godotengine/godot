#
# C++20 RISC-V emulator library
#

# DEBUG allows memory alignment checks and other things
option(RISCV_DEBUG  "Enable extra checks in the RISC-V machine" OFF)
# Enable and disable various RISC-V instruction
# set extensions. Not recommended to disable any.
option(RISCV_EXT_A  "Enable RISC-V atomic instructions" ON)
option(RISCV_EXT_C  "Enable RISC-V compressed instructions" ON)
option(RISCV_EXT_V  "Enable RISC-V vector instructions" OFF)
# Enable 32-, 64- and 128-bit architecture emulation
option(RISCV_32I    "Enable 32-bit RISC-V" ON)
option(RISCV_64I    "Enable 64-bit RISC-V" ON)
option(RISCV_128I   "Enable 128-bit RISC-V" OFF)
# Enable Floating-point Control and Status Register emulation
option(RISCV_FCSR   "Enable FCSR emulation" OFF)
# EXPERIMENTAL enables some high-performance interpreter
# features that may be unstable.
option(RISCV_EXPERIMENTAL  "Enable experimental features" OFF)
# MEMORY_TRAPS allows you to trap writes to uncacheable
# pages in memory. Cached pages can only be trapped once.
option(RISCV_MEMORY_TRAPS  "Enable memory page traps" ON)
# BINARY_TRANSLATION will make libriscv generate portable C-code,
# and invoke a system compiler to execute a program much faster.
option(RISCV_BINARY_TRANSLATION  "Enable exp. binary translation" ON)
# FLAT_RW_ARENA simplifies the program arena, making the heap area always read-write.
# Memory before the heap and outside of the arena behaves like before.
option(RISCV_FLAT_RW_ARENA       "Enable performant flat read-write arena" ON)

set(THREADED_IS_DEFAULT OFF)
# Threaded simulation uses computed goto, and is not supported
# on all compilers/systems.
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang"
	OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
	option(RISCV_THREADED "Enable fastest simulation mode" ON)
else()
	option(RISCV_THREADED "Enable fastest simulation mode" OFF)
endif()

# TAILCALL_DISPATCH enables clang-based compilers to use musttail dispatch.
option(RISCV_TAILCALL_DISPATCH   "Enable exp. tailcall dispatch" OFF)

if (RISCV_EXPERIMENTAL)
	# RISCV_ENCOMPASSING_ARENA allows the memory arena to encompass
	# the entire memory space, allowing for more efficient memory
	# accesses, but is only available for 32-bit RISC-V.
	option(RISCV_ENCOMPASSING_ARENA  "Enable encompassing memory arena" OFF)
	if (RISCV_ENCOMPASSING_ARENA)
		# Encompassing arena defaults to 32-bit address space, but it is configurable (power of 2).
		set(RISCV_ENCOMPASSING_ARENA_BITS "32" CACHE STRING "Encompassing arena address space bits")
		# Encompassing arena implies flat read-write arena
		set(RISCV_FLAT_RW_ARENA ON)
	else()
		unset(RISCV_ENCOMPASSING_ARENA_BITS CACHE)
	endif()
	# RISCV_ASM_DISPATCH enables a custom assembly dispatch
	option(RISCV_ASM_DISPATCH        "Enable assembly dispatch" OFF)
else()
	unset(RISCV_ENCOMPASSING_ARENA CACHE)
	unset(RISCV_ENCOMPASSING_ARENA_BITS CACHE)
endif()
if (RISCV_BINARY_TRANSLATION)
	# LIBTCC will embed the TCC compiler library, using it for binary translation.
	option(RISCV_LIBTCC              "Enable binary translation with libtcc" ON)
endif()

# Version information from git tags
set(RISCV_VERSION_MAJOR 1)
set(RISCV_VERSION_MINOR 11)

set (SOURCES
		libriscv/cpu.cpp
		libriscv/debug.cpp
		libriscv/decode_bytecodes.cpp
		libriscv/decoder_cache.cpp
		libriscv/machine.cpp
		libriscv/machine_defaults.cpp
		libriscv/memory.cpp
		libriscv/memory_elf.cpp
		libriscv/memory_mmap.cpp
		libriscv/memory_rw.cpp
		libriscv/native_libc.cpp
		libriscv/native_threads.cpp
		libriscv/posix/minimal.cpp
		libriscv/posix/signals.cpp
		libriscv/posix/threads.cpp
		libriscv/posix/socket_calls.cpp
		libriscv/serialize.cpp
		libriscv/util/crc32c.cpp
	)
if (RISCV_32I)
	list(APPEND SOURCES
		${CMAKE_CURRENT_LIST_DIR}/libriscv/rv32i.cpp
	)
endif()
if (RISCV_64I)
	list(APPEND SOURCES
		${CMAKE_CURRENT_LIST_DIR}/libriscv/rv64i.cpp
	)
endif()
if (RISCV_128I)
	list(APPEND SOURCES
		${CMAKE_CURRENT_LIST_DIR}/libriscv/rv128i.cpp
	)
endif()

if (MINGW_TOOLCHAIN OR MINGW OR WIN32)
	list(APPEND SOURCES
		libriscv/win32/system_calls.cpp
	)
elseif(UNIX)
	list(APPEND SOURCES
		libriscv/linux/system_calls.cpp
	)
endif()
if (RISCV_EXPERIMENTAL AND RISCV_ASM_DISPATCH)
	# ASM_DISPATCH uses assembly to dispatch instructions.
	# It is currently only available on the AMD64 (x86_64) Linux platform.
	if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64"
	 OR CMAKE_SYSTEM_PROCESSOR STREQUAL "amd64")
		message(STATUS "libriscv: Assembly dispatch enabled")
		list(APPEND SOURCES
			libriscv/amd64/asm_dispatch.cpp
			libriscv/threaded_dispatch.cpp
			libriscv/threaded_inaccurate_dispatch.cpp
			#libriscv/amd64/accurate_dispatch_rv64gb.o
			libriscv/amd64/inaccurate_dispatch_rv64gb.o
		)
	else()
		message(FATAL_ERROR "libriscv: Assembly dispatch not supported on this platform")
	endif()
elseif (RISCV_THREADED OR RISCV_TAILCALL_DISPATCH)
	if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
	 AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0
	 AND RISCV_TAILCALL_DISPATCH)
	 	# Experimental tail-call dispatch
		message(STATUS "libriscv: Tail-call dispatch enabled")
		list(APPEND SOURCES
			libriscv/tailcall_dispatch.cpp
		)
	else()
		message(STATUS "libriscv: Threaded dispatch enabled")
		list(APPEND SOURCES
			libriscv/threaded_dispatch.cpp
			libriscv/threaded_inaccurate_dispatch.cpp
		)
	endif()
else()
	message(STATUS "libriscv: Switch-based dispatch enabled")
	list(APPEND SOURCES
		libriscv/bytecode_dispatch.cpp
	)
endif()
if (RISCV_BINARY_TRANSLATION)
	list(APPEND SOURCES
		libriscv/tr_api.cpp
		libriscv/tr_emit.cpp
		libriscv/tr_translate.cpp
	)
	if (MSVC)
		list(APPEND SOURCES libriscv/win32/tr_msvc.cpp)
	else()
		list(APPEND SOURCES libriscv/tr_compiler.cpp)
		if (RISCV_LIBTCC)
			list(APPEND SOURCES libriscv/tr_tcc.cpp)
		endif()
	endif()
endif()

# Generate the settings header
# RISCV_VERSION_MAJOR and RISCV_VERSION_MINOR are required by libriscv_settings.h.in
configure_file(libriscv_settings.h.in ${CMAKE_CURRENT_BINARY_DIR}/libriscv_settings.h)
configure_file(libriscv.pc.in ${CMAKE_CURRENT_BINARY_DIR}/libriscv.pc)

add_library(riscv ${SOURCES})
if (COSMOPOLITAN)
	target_compile_options(riscv PUBLIC -fexceptions)
endif()

if(APPLE)
    find_library(SECURITY_FRAMEWORK Security)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    mark_as_advanced(SECURITY_FRAMEWORK FOUNDATION_FRAMEWORK)

    target_link_libraries(riscv PUBLIC 
        "${SECURITY_FRAMEWORK}"
        "${FOUNDATION_FRAMEWORK}"
    )
endif()

target_compile_features(riscv PUBLIC cxx_std_20)
target_include_directories(riscv PUBLIC .)
target_include_directories(riscv PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

if (NOT WIN32 OR MINGW_TOOLCHAIN)
	target_compile_options(riscv PRIVATE -Wall -Wextra)
endif()

if (RISCV_EXPERIMENTAL AND RISCV_ENCOMPASSING_ARENA)
	target_compile_definitions(riscv PUBLIC
		RISCV_ENCOMPASSING_ARENA_BITS=${RISCV_ENCOMPASSING_ARENA_BITS}
	)
endif()
if (RISCV_EXPERIMENTAL AND RISCV_ASM_DISPATCH)
	target_compile_definitions(riscv PUBLIC
		RISCV_ASM_DISPATCH=1
	)
endif()

if (RISCV_MULTIPROCESS)
	find_package(Threads REQUIRED)
	target_link_libraries(riscv PUBLIC Threads::Threads)
endif()

if (WIN32 OR MINGW_TOOLCHAIN)
	target_link_libraries(riscv PUBLIC wsock32 ws2_32)
	if (RISCV_BINARY_TRANSLATION)
		target_sources(riscv PRIVATE libriscv/win32/dlfcn.cpp)
	endif()
elseif (RISCV_BINARY_TRANSLATION)
	target_link_libraries(riscv PUBLIC dl)
endif()


if (RISCV_BINARY_TRANSLATION AND RISCV_LIBTCC)
	enable_language(C)
	include(FetchContent)
	FetchContent_Declare(tinycc
		GIT_REPOSITORY https://github.com/fwsGonzo/tinycc.git
		GIT_TAG        mob
	)
	FetchContent_MakeAvailable(tinycc)

	set(TINYCC_LOCATION    "${CMAKE_BINARY_DIR}/_deps/tinycc-src")
	set(CONFIG_H_LOCATION "${TINYCC_LOCATION}/config.h")
	configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.in ${CONFIG_H_LOCATION})
	set_source_files_properties(${CONFIG_H_LOCATION} PROPERTIES GENERATED TRUE)
	file(COPY "${TINYCC_LOCATION}/libtcc.h" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
	file(COPY "${TINYCC_LOCATION}/libtcc1.h" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
	file(COPY "${TINYCC_LOCATION}/lib-arm64.h" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

	target_sources(riscv PRIVATE
		${tinycc_SOURCE_DIR}/libtcc.c
		${tinycc_SOURCE_DIR}/libtcc.h
		${CONFIG_H_LOCATION}
	)
	target_compile_definitions(riscv PRIVATE
		CONFIG_TCC_PREDEFS=1
		TCC_VERSION="0.9.27"
		TCC_IS_NATIVE=1
		CONFIG_TCC_BACKTRACE=0
		CONFIG_TCC_BCHECK=0
	)
	# We support: Windows, Linux, macOS and Android
	if (ANDROID_TOOLCHAIN)
		target_compile_definitions(riscv PRIVATE
			TARGETOS_ANDROID=1
			TCC_TARGET_ARM64=1
		)
	elseif (CMAKE_HOST_APPLE OR APPLE)
		target_compile_definitions(riscv PRIVATE
			TARGETOS_Darwin=1
			TCC_TARGET_MACHO=1
			TCC_TARGET_ARM64=1
		)
	elseif (CMAKE_HOST_WIN32 OR MINGW_TOOLCHAIN)
		target_compile_definitions(riscv PRIVATE
			TARGETOS_Windows=1
			TCC_TARGET_PE=1
			TCC_TARGET_X86_64=1
		)
	elseif (CMAKE_SYSTEM_NAME STREQUAL FreeBSD)
		target_compile_definitions(riscv PRIVATE
			TARGETOS_FreeBSD=1
			TCC_TARGET_X86_64=1
		)
	elseif (RISCV_TOOLCHAIN)
		target_compile_definitions(riscv PRIVATE
			TARGETOS_Linux=1
			TCC_TARGET_RISCV64=1
		)
	elseif (CMAKE_HOST_UNIX)
		target_compile_definitions(riscv PRIVATE
			TARGETOS_Linux=1
			TCC_TARGET_X86_64=1
		)
	endif()
	target_include_directories(riscv PUBLIC
		"${CMAKE_CURRENT_BINARY_DIR}"
	)
endif()

if (${CMAKE_PROJECT_NAME} STREQUAL "libriscv")
	install(TARGETS riscv)
	install(FILES
		${CMAKE_CURRENT_BINARY_DIR}/libriscv_settings.h
		DESTINATION include/${PROJECT_NAME}
	)
	install(FILES
		libriscv/cached_address.hpp
		libriscv/common.hpp
		libriscv/cpu.hpp
		libriscv/cpu_inline.hpp
		libriscv/debug.hpp
		libriscv/decoded_exec_segment.hpp
		libriscv/decoder_cache.hpp
		libriscv/elf.hpp
		libriscv/guest_datatypes.hpp
		libriscv/instr_helpers.hpp
		libriscv/instruction_counter.hpp
		libriscv/instruction_list.hpp
		libriscv/machine.hpp
		libriscv/machine_inline.hpp
		libriscv/machine_vmcall.hpp
		libriscv/memory.hpp
		libriscv/memory_helpers_paging.hpp
		libriscv/memory_inline.hpp
		libriscv/memory_inline_pages.hpp
		libriscv/mmap_cache.hpp
		libriscv/native_heap.hpp
		libriscv/page.hpp
		libriscv/prepared_call.hpp
		libriscv/registers.hpp
		libriscv/rvv_registers.hpp
		libriscv/riscvbase.hpp
		libriscv/rv32i_instr.hpp
		libriscv/rva.hpp
		libriscv/rvc.hpp
		libriscv/rvfd.hpp
		libriscv/rsp_server.hpp
		libriscv/threads.hpp
		libriscv/types.hpp

		DESTINATION include/${PROJECT_NAME}
	)
	install(FILES
		libriscv/linux/rsp_server.hpp

		DESTINATION include/${PROJECT_NAME}/linux
	)
	install(FILES
		libriscv/posix/filedesc.hpp
		libriscv/posix/signals.hpp

		DESTINATION include/${PROJECT_NAME}/posix
	)
	install(FILES
		libriscv/util/buffer.hpp
		libriscv/util/function.hpp

		DESTINATION include/${PROJECT_NAME}/util
	)
	install(
		FILES ${CMAKE_CURRENT_BINARY_DIR}/libriscv.pc
		DESTINATION lib/pkgconfig
	)
endif()
