cmake_minimum_required(VERSION 3.9)
project(riscv CXX)

#set(FUZZER_MODE "memory,undefined" "-fsanitize-memory-track-origins")
set(FUZZER_MODE "address,undefined")

option(RISCV_128I "" ON)
add_subdirectory(../lib lib)
target_compile_options(riscv PUBLIC -ggdb3 -O0)
target_compile_options(riscv PUBLIC "-fsanitize=${FUZZER_MODE},fuzzer")
target_compile_options(riscv PUBLIC "--coverage" "-fprofile-instr-generate" "-fcoverage-mapping")
target_compile_definitions(riscv PUBLIC FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION=1)

if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
	message(FATAL_ERROR "libfuzzer is part of the Clang compiler suite.")
endif()

set(SOURCES
	fuzz.cpp
)

function(add_fuzzer NAME ARCH MODE)
	add_executable(${NAME} ${SOURCES})
	target_link_libraries(${NAME} riscv)
	target_link_libraries(${NAME} "-fsanitize=${FUZZER_MODE},fuzzer")
	target_link_libraries(${NAME} "-fprofile-instr-generate" "-fcoverage-mapping" "-Wl,-Bsymbolic-functions" "-Wl,-z,relro" "-fuse-ld=lld")
	target_compile_definitions(${NAME} PRIVATE RISCV_ARCH=${ARCH})
	target_compile_definitions(${NAME} PRIVATE ${MODE}=1)
endfunction()

add_fuzzer(vmfuzzer32  4   FUZZ_VM)
add_fuzzer(vmfuzzer64  8   FUZZ_VM)
add_fuzzer(vmfuzzer128 16  FUZZ_VM)

add_fuzzer(elffuzzer32  4   FUZZ_ELF)
add_fuzzer(elffuzzer64  8   FUZZ_ELF)
add_fuzzer(elffuzzer128 16  FUZZ_ELF)

add_fuzzer(natsysfuzzer32  4   FUZZ_NAT)
add_fuzzer(natsysfuzzer64  8   FUZZ_NAT)

add_fuzzer(syshelperfuzzer32  4   FUZZ_SYSH)
add_fuzzer(syshelperfuzzer64  8   FUZZ_SYSH)
