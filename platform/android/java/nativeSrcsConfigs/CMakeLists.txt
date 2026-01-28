# Non functional cmake build file used to provide Android Studio editor support to the project.
cmake_minimum_required(VERSION 3.10)
project(godot)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(GODOT_ROOT_DIR ../../../..)
set(ANDROID_ROOT_DIR "${GODOT_ROOT_DIR}/platform/android" CACHE STRING "")
set(OPENXR_INCLUDE_DIR "${GODOT_ROOT_DIR}/thirdparty/openxr/include" CACHE STRING "")

# Get sources
file(GLOB_RECURSE SOURCES ${GODOT_ROOT_DIR}/*.c**)
file(GLOB_RECURSE HEADERS ${GODOT_ROOT_DIR}/*.h**)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})
target_include_directories(${PROJECT_NAME}
        SYSTEM PUBLIC
        ${GODOT_ROOT_DIR}
        ${ANDROID_ROOT_DIR}
        ${OPENXR_INCLUDE_DIR})

add_definitions(
        -DUNIX_ENABLED
        -DVULKAN_ENABLED
        -DANDROID_ENABLED
        -DGLES3_ENABLED
        -DTOOLS_ENABLED
        -DDEBUG_ENABLED
        -DRD_ENABLED
)
