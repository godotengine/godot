cmake_minimum_required(VERSION 3.9.4)
project(riscv CXX)

option(SANITIZE    "Enable sanitizers" OFF)
option(STATIC_BUILD "Build statically" OFF)

set(SOURCES
	src/cpp/resource_loader_cpp.cpp
	src/cpp/resource_saver_cpp.cpp
	src/cpp/script_cpp.cpp
	src/cpp/script_cpp_instance.cpp
	src/cpp/script_language_cpp.cpp
	src/elf/resource_loader_elf.cpp
	src/elf/resource_saver_elf.cpp
	src/elf/script_elf.cpp
	src/elf/script_instance.cpp
	src/elf/script_language_elf.cpp
	src/rust/resource_loader_rust.cpp
	src/rust/resource_saver_rust.cpp
	src/rust/script_rust.cpp
	src/rust/script_language_rust.cpp
	src/zig/resource_loader_zig.cpp
	src/zig/resource_saver_zig.cpp
	src/zig/script_zig.cpp
	src/zig/script_language_zig.cpp
	src/docker.cpp
	src/godot/script_instance.cpp
	src/guest_variant.cpp
	src/register_types.cpp
	src/sandbox.cpp
	src/sandbox_bintr.cpp
	src/sandbox_debug.cpp
	src/sandbox_exception.cpp
	src/sandbox_functions.cpp
	src/sandbox_globals.cpp
	src/sandbox_generated_api.cpp
	src/sandbox_profiling.cpp
	src/sandbox_programs.cpp
	src/sandbox_project_settings.cpp
	src/sandbox_restrictions.cpp
	src/sandbox_shm.cpp
	src/sandbox_syscalls.cpp
	src/sandbox_syscalls_2d.cpp
	src/sandbox_syscalls_3d.cpp
	src/override_libriscv.cpp

	src/tests/assault.cpp
)

# Add bintr sources
file(GLOB BINTR_FILES src/bintr/*.cpp)
list(APPEND SOURCES ${BINTR_FILES})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fvisibility=hidden -fvisibility-inlines-hidden")

if (MINGW OR MINGW_TOOLCHAIN)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libgcc -static-libstdc++ -static")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -static-libgcc -static-libstdc++ -static")
endif()

add_subdirectory(ext)

add_library(godot-riscv SHARED ${SOURCES})
target_link_libraries(godot-riscv PUBLIC riscv godot-cpp)

if (STATIC_BUILD)
	target_link_libraries(godot-riscv PUBLIC -static)
endif()
if (SANITIZE)
	#target_compile_options(godot-riscv PUBLIC -fsanitize=scudo,undefined)
	#target_link_libraries(godot-riscv PUBLIC -fsanitize=scudo,undefined -L/usr/lib/llvm-18/lib/clang/18/lib/linux -l:libclang_rt.ubsan_standalone-x86_64.so -l:libclang_rt.scudo_standalone-x86_64.so -Wl,-rpath=/usr/lib/llvm-18/lib/clang/18/lib/linux)
	target_compile_options(godot-riscv PUBLIC -fsanitize=undefined -fno-sanitize=vptr)
	target_link_libraries(godot-riscv PUBLIC -fsanitize=undefined -L/usr/lib/llvm-18/lib/clang/18/lib/linux -l:libclang_rt.ubsan_standalone-x86_64.a)
endif()
if (ANDROID_TOOLCHAIN)
	target_link_libraries(godot-riscv PUBLIC android log)
endif()
