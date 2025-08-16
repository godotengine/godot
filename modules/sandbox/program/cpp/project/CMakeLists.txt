cmake_minimum_required(VERSION 3.18)
project (godot-sandbox-programs)

# Fetch godot-sandbox for add_sandbox_program and other functions
include(FetchContent)
FetchContent_Declare(
	godot-sandbox
	GIT_REPOSITORY https://github.com/libriscv/godot-sandbox.git
	GIT_TAG        main
	SOURCE_SUBDIR  "program/cpp/cmake"
)
FetchContent_MakeAvailable(godot-sandbox)

# Build the godot-sandbox program `sandbox_program.elf`:
add_sandbox_program(sandbox_program.elf
	"src/program.cpp"
)
