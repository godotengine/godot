cmake_minimum_required(VERSION 3.5)
project(wasm_example)

add_subdirectory(../../lib libriscv)
set(CMAKE_EXECUTABLE_SUFFIX ".html") # emcc
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s TOTAL_MEMORY=300MB -fexceptions")

file(GLOB BINTR_SOURCES
	"bintr/*.c"
	"bintr/*.cpp"
)
foreach(src ${BINTR_SOURCES})
	message(STATUS "Adding translation file: ${src}")
	# Avoid logspam
	set_source_files_properties(${src} PROPERTIES
		COMPILE_FLAGS "-Wno-constant-conversion -Wno-unused-function -Wno-unused-variable -Wno-unused-parameter -Wno-sign-compare"
	)
endforeach()

add_executable(wasm_example main.cpp ${BINTR_SOURCES})
target_link_libraries(wasm_example PRIVATE riscv)
set_target_properties(wasm_example PROPERTIES
	CXX_STANDARD 20
	CXX_STANDARD_REQUIRED ON
	CXX_EXTENSIONS ON
)

# Set the output directory for the executable
#set_target_properties(wasm_example PROPERTIES
#	RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
#)
