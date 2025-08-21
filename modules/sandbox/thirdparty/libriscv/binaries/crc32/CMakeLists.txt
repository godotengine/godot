cmake_minimum_required(VERSION 3.5)
project(crc32 CXX)

set(WARNINGS  "-Wall -Wextra")
set(COMMON    "-O2 -fstack-protector")
set(RISCV_ABI "-march=rv64g_zba_zbb_zbc_zbs -mabi=lp64d")

set(CMAKE_CXX_FLAGS "${WARNINGS} ${RISCV_ABI} -std=c++2a ${COMMON}")

set(SOURCES
	crc32.cpp
)

add_executable(crc32 ${SOURCES})
target_link_libraries(crc32 -static)
