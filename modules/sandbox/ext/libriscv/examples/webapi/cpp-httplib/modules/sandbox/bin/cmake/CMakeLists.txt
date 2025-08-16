cmake_minimum_required(VERSION 3.10)
project(example LANGUAGES CXX)

# Fetch godot-sandbox C++ API
include(FetchContent)
FetchContent_Declare(
	godot-sandbox
	GIT_REPOSITORY https://github.com/libriscv/godot-sandbox.git
	GIT_TAG        main
	GIT_SHALLOW    TRUE
	GIT_SUBMODULES ""
	SOURCE_SUBDIR  "program/cpp/cmake"
)
FetchContent_MakeAvailable(godot-sandbox)

# Create example.elf in the src folder
add_sandbox_program_at(example.elf ../src
	../src/example.cpp
)
