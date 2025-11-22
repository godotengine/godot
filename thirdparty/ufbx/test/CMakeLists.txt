cmake_minimum_required(VERSION 3.15)

project(ufbxw_tests)

option(USE_ASAN "Enable address sanitizer" OFF)

add_library(ufbx_write)

set(SRC_UFBX_WRITE
    ../ufbx_write.c
    ../ufbx_write.h
    ../misc/ufbxw.natvis
    ../misc/ufbxwi.natvis
)

if(MSVC)
    string(REPLACE "/Ob1" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /Ob2")
    string(REPLACE "/Ob1" "" CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO}")
    set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} /Ob2")
endif()

if(USE_ASAN)
    add_compile_options("-fsanitize=address")
    add_link_options("-fsanitize=address")
endif()

target_sources(ufbx_write PRIVATE ${SRC_UFBX_WRITE})

add_subdirectory(ufbx)
add_subdirectory(extra)
add_subdirectory(util)
add_subdirectory(test)
add_subdirectory(unit)
add_subdirectory(roundtrip)
add_subdirectory(compare_fbx)

if (MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-Wall -Wextra -Werror -Wno-format-truncation)
    target_link_libraries(ufbxw_test PRIVATE m)
    target_link_libraries(ufbxw_unit PRIVATE m)
endif()

target_compile_definitions(ufbx PRIVATE UFBX_DEV=1)
target_compile_definitions(ufbx_write PRIVATE UFBXW_DEV=1 UFBX_DEV=1)

get_filename_component(ROOT_PATH "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ufbxw_test)
set_target_properties(ufbxw_test PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY ${ROOT_PATH})
set_target_properties(ufbxw_unit PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY ${ROOT_PATH})
set_target_properties(roundtrip PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY ${ROOT_PATH})
set_target_properties(compare_fbx PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY ${ROOT_PATH})

file(MAKE_DIRECTORY "${ROOT_PATH}/output")

set(CMAKE_SUPPRESS_REGENERATION true)
