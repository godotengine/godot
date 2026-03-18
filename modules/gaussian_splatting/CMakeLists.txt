# Auto-generated for IDE support. Canonical build uses SCsub. Run generate_cmake.py to refresh.
#
# WARNING: This file uses file(GLOB ...) to collect sources. CMake does not
# automatically detect newly added files when using GLOB. If you add or remove
# .cpp/.h files, re-run cmake configuration (or generate_cmake.py) to pick them up.
#
# CMakeLists.txt for Godot Gaussian Splatting Module
# This file is for IDE support (Rider) only - actual building is done via SCons
cmake_minimum_required(VERSION 3.16)
project(GodotGaussianSplatting)

# Set C++ standard to match Godot 4.x requirements
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Platform-specific settings
if(WIN32)
    add_definitions(-DWINDOWS_ENABLED)
    add_definitions(-DWASAPI_ENABLED)
    add_definitions(-DWINMIDI_ENABLED)
    add_definitions(-DTYPED_METHOD_BIND)
    add_definitions(-DNOMINMAX)  # Prevent Windows.h min/max macros
elseif(UNIX AND NOT APPLE)
    add_definitions(-DX11_ENABLED)
    add_definitions(-DUNIX_ENABLED)
elseif(APPLE)
    add_definitions(-DOSX_ENABLED)
    add_definitions(-DUNIX_ENABLED)
endif()

# Godot-specific definitions
add_definitions(-DGAUSSIAN_SPLATTING_ENABLED)
add_definitions(-DTOOLS_ENABLED)  # Enable editor features
add_definitions(-DDEBUG_ENABLED)  # Enable debug features
add_definitions(-DDEV_ENABLED)    # Enable development features

# Set Godot source path for the flattened fork layout.
set(GODOT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../..")

# Repository root for cross-platform helper scripts.
get_filename_component(GODOTGS_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

# Include directories for proper IntelliSense
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GODOT_SOURCE_DIR}
    ${GODOT_SOURCE_DIR}/core
    ${GODOT_SOURCE_DIR}/core/extension
    ${GODOT_SOURCE_DIR}/core/input
    ${GODOT_SOURCE_DIR}/core/io
    ${GODOT_SOURCE_DIR}/core/math
    ${GODOT_SOURCE_DIR}/core/object
    ${GODOT_SOURCE_DIR}/core/os
    ${GODOT_SOURCE_DIR}/core/string
    ${GODOT_SOURCE_DIR}/core/templates
    ${GODOT_SOURCE_DIR}/core/variant
    ${GODOT_SOURCE_DIR}/drivers
    ${GODOT_SOURCE_DIR}/editor
    ${GODOT_SOURCE_DIR}/modules
    ${GODOT_SOURCE_DIR}/platform
    ${GODOT_SOURCE_DIR}/scene
    ${GODOT_SOURCE_DIR}/scene/2d
    ${GODOT_SOURCE_DIR}/scene/3d
    ${GODOT_SOURCE_DIR}/scene/animation
    ${GODOT_SOURCE_DIR}/scene/gui
    ${GODOT_SOURCE_DIR}/scene/main
    ${GODOT_SOURCE_DIR}/scene/resources
    ${GODOT_SOURCE_DIR}/servers
    ${GODOT_SOURCE_DIR}/servers/rendering
    ${GODOT_SOURCE_DIR}/servers/rendering/renderer_rd
    ${GODOT_SOURCE_DIR}/servers/rendering/renderer_rd/storage_rd
    ${GODOT_SOURCE_DIR}/servers/rendering_server
    ${GODOT_SOURCE_DIR}/thirdparty
)

# =============================================================================
# Source file organization by component
# =============================================================================

# Core Components
file(GLOB CORE_SOURCES
    "core/*.cpp"
    "core/*.h"
)

# Animation System
file(GLOB ANIMATION_SOURCES
    "animation/*.cpp"
    "animation/*.h"
)

# Asset Management
file(GLOB ASSET_MANAGEMENT_SOURCES
    "asset_management/*.cpp"
    "asset_management/*.h"
)

# Compute Infrastructure
file(GLOB COMPUTE_SOURCES
    "compute/*.cpp"
    "compute/*.h"
)

# Editor Integration
file(GLOB EDITOR_SOURCES
    "editor/*.cpp"
    "editor/*.h"
)

# I/O and Loading
file(GLOB IO_SOURCES
    "io/*.cpp"
    "io/*.h"
)

# Interface Contracts and Bridges
file(GLOB INTERFACE_SOURCES
    "interfaces/*.cpp"
    "interfaces/*.h"
)

# Level of Detail System
file(GLOB LOD_SOURCES
    "lod/*.cpp"
    "lod/*.h"
)

# Logging and Diagnostics
file(GLOB LOGGER_SOURCES
    "logger/*.cpp"
    "logger/*.h"
)

# Scene Nodes
file(GLOB NODE_SOURCES
    "nodes/*.cpp"
    "nodes/*.h"
)

# Painterly Rendering
file(GLOB PAINTERLY_SOURCES
    "painterly/*.cpp"
    "painterly/*.h"
)

# Persistence System
file(GLOB PERSISTENCE_SOURCES
    "persistence/*.cpp"
    "persistence/*.h"
)

# Rendering Pipeline
file(GLOB RENDERER_SOURCES
    "renderer/*.cpp"
    "renderer/*.h"
)

# Resources
file(GLOB RESOURCE_SOURCES
    "resources/*.cpp"
    "resources/*.h"
)

# Module Registration
set(REGISTRATION_SOURCES
    "register_types.cpp"
    "register_types.h"
)

# Tests (optional - only included in test builds)
file(GLOB TEST_SOURCES
    "tests/*.cpp"
    "tests/*.h"
)

# Shaders (for IDE visibility)
file(GLOB_RECURSE SHADER_FILES
    "shaders/*.glsl"
    "shaders/*.vert"
    "shaders/*.frag"
    "shaders/*.comp"
)

# GDScript files (for reference)
file(GLOB_RECURSE GDSCRIPT_FILES
    "*.gd"
    "tests/*.gd"
)

# Documentation and configuration
file(GLOB DOC_FILES
    "*.md"
    "docs/*.md"
    "tests/*.md"
)

# =============================================================================
# Create IDE Target Groups for Better Organization
# =============================================================================

# Main module library (for IntelliSense and navigation)
add_library(gaussian_splatting_core STATIC
    ${CORE_SOURCES}
    ${REGISTRATION_SOURCES}
)

# Rendering subsystem
add_library(gaussian_splatting_rendering STATIC
    ${RENDERER_SOURCES}
    ${PAINTERLY_SOURCES}
    ${COMPUTE_SOURCES}
)

# Data management subsystem
add_library(gaussian_splatting_data STATIC
    ${IO_SOURCES}
    ${RESOURCE_SOURCES}
    ${ASSET_MANAGEMENT_SOURCES}
    ${PERSISTENCE_SOURCES}
)

# Scene integration
add_library(gaussian_splatting_scene STATIC
    ${NODE_SOURCES}
    ${ANIMATION_SOURCES}
)

# Systems and utilities
add_library(gaussian_splatting_systems STATIC
    ${LOD_SOURCES}
    ${INTERFACE_SOURCES}
    ${LOGGER_SOURCES}
)

# Editor components (only in tools builds)
if(TOOLS_ENABLED)
    add_library(gaussian_splatting_editor STATIC
        ${EDITOR_SOURCES}
    )
    set_target_properties(gaussian_splatting_editor PROPERTIES
        FOLDER "Editor"
    )
endif()

# Test executable (optional)
option(BUILD_TESTS "Build test executables" OFF)
if(BUILD_TESTS)
    add_executable(gaussian_splatting_tests
        ${TEST_SOURCES}
    )
    set_target_properties(gaussian_splatting_tests PROPERTIES
        FOLDER "Tests"
    )
endif()

# Custom target for shaders (visible in IDE but not compiled)
add_custom_target(gaussian_splatting_shaders
    SOURCES ${SHADER_FILES}
)

# Custom target for GDScript files
add_custom_target(gaussian_splatting_gdscripts
    SOURCES ${GDSCRIPT_FILES}
)

# Custom target for documentation
add_custom_target(gaussian_splatting_docs
    SOURCES ${DOC_FILES}
)

# =============================================================================
# IDE Organization
# =============================================================================

# Set folder properties for better IDE organization
set_target_properties(gaussian_splatting_core PROPERTIES
    FOLDER "Core"
)

set_target_properties(gaussian_splatting_rendering PROPERTIES
    FOLDER "Rendering"
)

set_target_properties(gaussian_splatting_data PROPERTIES
    FOLDER "Data Management"
)

set_target_properties(gaussian_splatting_scene PROPERTIES
    FOLDER "Scene Integration"
)

set_target_properties(gaussian_splatting_systems PROPERTIES
    FOLDER "Systems"
)

set_target_properties(gaussian_splatting_shaders PROPERTIES
    FOLDER "Resources/Shaders"
)

set_target_properties(gaussian_splatting_gdscripts PROPERTIES
    FOLDER "Resources/Scripts"
)

set_target_properties(gaussian_splatting_docs PROPERTIES
    FOLDER "Documentation"
)

# =============================================================================
# Source Groups for File Organization in IDE
# =============================================================================

source_group("Core" FILES ${CORE_SOURCES})
source_group("Core\\Registration" FILES ${REGISTRATION_SOURCES})
source_group("Animation" FILES ${ANIMATION_SOURCES})
source_group("Asset Management" FILES ${ASSET_MANAGEMENT_SOURCES})
source_group("Compute" FILES ${COMPUTE_SOURCES})
source_group("Editor" FILES ${EDITOR_SOURCES})
source_group("IO" FILES ${IO_SOURCES})
source_group("Interfaces" FILES ${INTERFACE_SOURCES})
source_group("LOD" FILES ${LOD_SOURCES})
source_group("Logger" FILES ${LOGGER_SOURCES})
source_group("Nodes" FILES ${NODE_SOURCES})
source_group("Painterly" FILES ${PAINTERLY_SOURCES})
source_group("Persistence" FILES ${PERSISTENCE_SOURCES})
source_group("Renderer" FILES ${RENDERER_SOURCES})
source_group("Resources" FILES ${RESOURCE_SOURCES})
source_group("Tests" FILES ${TEST_SOURCES})
source_group("Shaders" FILES ${SHADER_FILES})
source_group("Scripts" FILES ${GDSCRIPT_FILES})
source_group("Documentation" FILES ${DOC_FILES})

# =============================================================================
# Compiler Settings for Better IDE Experience
# =============================================================================

# Enable all warnings for development
if(MSVC)
    target_compile_options(gaussian_splatting_core PRIVATE /W4)
    # Disable specific warnings that Godot triggers
    target_compile_options(gaussian_splatting_core PRIVATE
        /wd4244  # conversion from 'type1' to 'type2'
        /wd4305  # truncation from 'type1' to 'type2'
        /wd4267  # conversion from 'size_t' to 'type'
        /wd4100  # unreferenced formal parameter
    )
else()
    target_compile_options(gaussian_splatting_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

# =============================================================================
# Custom Build Commands (for reference - actual build uses SCons)
# =============================================================================

find_program(SCONS_EXECUTABLE NAMES scons scons.py)
if(SCONS_EXECUTABLE)
    if(WIN32)
        set(GS_SCONS_PLATFORM "windows")
    elseif(APPLE)
        set(GS_SCONS_PLATFORM "macos")
    else()
        set(GS_SCONS_PLATFORM "linuxbsd")
    endif()

    add_custom_target(build_with_scons
        COMMAND ${SCONS_EXECUTABLE}
                platform=${GS_SCONS_PLATFORM}
                target=editor
                dev_build=yes
                modules/gaussian_splatting
        WORKING_DIRECTORY ${GODOT_SOURCE_DIR}
        COMMENT "Building Gaussian Splatting module with SCons"
    )
else()
    message(STATUS "SCons not found in PATH; skipping build_with_scons target.")
endif()

find_program(PYTHON_EXECUTABLE NAMES python3 python)
if(PYTHON_EXECUTABLE)
    add_custom_target(validate_build_metadata
        COMMAND ${PYTHON_EXECUTABLE} modules/gaussian_splatting/tests/check_build_metadata_consistency.py
        WORKING_DIRECTORY ${GODOTGS_REPO_ROOT}
        COMMENT "Validating Gaussian Splatting SCsub/CMake/doc metadata consistency"
    )

    add_custom_target(run_module_tests
        COMMAND ${PYTHON_EXECUTABLE} tests/ci/run_module_tests.py
        WORKING_DIRECTORY ${GODOTGS_REPO_ROOT}
        COMMENT "Running Gaussian Splatting module tests and CI guards"
    )
else()
    message(STATUS "Python not found in PATH; skipping validation/test helper targets.")
endif()

# =============================================================================
# Export compile commands for additional tools
# =============================================================================

# This generates compile_commands.json for tools like clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Print configuration summary
message(STATUS "========================================")
message(STATUS "Godot Gaussian Splatting Module Configuration")
message(STATUS "========================================")
message(STATUS "Godot Source: ${GODOT_SOURCE_DIR}")
message(STATUS "Module Path: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Tests: ${BUILD_TESTS}")
message(STATUS "========================================")
