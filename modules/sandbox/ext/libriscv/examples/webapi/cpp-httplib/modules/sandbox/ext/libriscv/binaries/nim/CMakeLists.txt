cmake_minimum_required(VERSION 3.11.0)
project(programs C)

include(cmake/micronim.cmake)

file(READ "${CMAKE_BINARY_DIR}/buildfiles.txt" BUILD_LIST)
string(REPLACE "\"" "" BUILD_LIST ${BUILD_LIST})
string(REPLACE "\n" ";" BUILD_LIST ${BUILD_LIST})

add_micronim_binary(hello
	${BUILD_LIST}
)
target_include_directories(hello PRIVATE ${NIM_LIBS})
