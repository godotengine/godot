cmake_minimum_required(VERSION 3.10)
project(example LANGUAGES CXX)

add_subdirectory(../ cmake)

# Fetch fmt from GitHub
include(FetchContent)
FetchContent_Declare(libfmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt
  GIT_TAG        master
  )
FetchContent_MakeAvailable(libfmt)

add_sandbox_program(example
  "example.cpp"
)
target_link_libraries(example PRIVATE fmt::fmt)
