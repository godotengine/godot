cmake_minimum_required(VERSION 3.10)
project(unittests LANGUAGES CXX)

add_subdirectory(../program/cpp/cmake sandbox_api)

add_sandbox_program(unittests
	"tests/test_basic.cpp"
	"tests/test_math.cpp"
	"tests/test_properties.cpp"
	"tests/test_shm.cpp"
)
