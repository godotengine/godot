cmake_minimum_required (VERSION 3.14)
project ("libriscv_example")

option(RISCV_THREADED "" ON)

include(FetchContent)
FetchContent_Declare(libriscv
  GIT_REPOSITORY https://github.com/fwsGonzo/libriscv
  GIT_TAG        master
  )
FetchContent_MakeAvailable(libriscv)

add_executable (example
	"main.cpp"
)
target_link_libraries(example PRIVATE riscv)
