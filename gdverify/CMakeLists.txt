cmake_minimum_required(VERSION 3.12)
project(gdverify)

option(GODOT_SRC_DIR "Path to Godot source" "${CMAKE_CURRENT_SOURCE_DIR}/../godot")

add_compile_definitions(TOOLS_ENABLED=0 DEBUG_ENABLED=0)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${GODOT_SRC_DIR})

add_library(gdscript STATIC
    ${GODOT_SRC_DIR}/modules/gdscript/gdscript_parser.cpp
    ${GODOT_SRC_DIR}/modules/gdscript/gdscript_tokenizer.cpp
)

add_executable(gdverify src/main.cpp)
target_link_libraries(gdverify gdscript)
