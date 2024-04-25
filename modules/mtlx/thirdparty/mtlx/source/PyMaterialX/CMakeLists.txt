include_directories(
    ${EXTERNAL_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Apply Python version and location requests from the user.
set(PYBIND11_PYTHON_VERSION ${MATERIALX_PYTHON_VERSION})
set(PYTHON_EXECUTABLE ${MATERIALX_PYTHON_EXECUTABLE})

# First look for a PyBind11 package via CMake.
if(MATERIALX_PYTHON_PYBIND11_DIR STREQUAL "")
    find_package(pybind11 QUIET)
    if(pybind11_FOUND)
        include_directories(${PYBIND11_INCLUDE_DIR})
    else()
        set(MATERIALX_PYTHON_PYBIND11_DIR "${CMAKE_CURRENT_SOURCE_DIR}/External/PyBind11")
    endif()
endif()

# Then look for PyBind11 at its native or user-specified location.
if(NOT pybind11_FOUND)
    add_subdirectory(${MATERIALX_PYTHON_PYBIND11_DIR})
    include_directories(${PYBIND11_INCLUDE_DIR})
endif()

if(NOT MATERIALX_PYTHON_LTO)
    set(PYBIND11_MODULE_FLAGS "NO_EXTRAS")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES Clang)
    include(CheckCXXCompilerFlag)
    CHECK_CXX_COMPILER_FLAG(-Wno-undefined-var-template UNDEFINED_VAR_TEMPLATE_FLAG)
    if(UNDEFINED_VAR_TEMPLATE_FLAG)
        add_compile_options(-Wno-undefined-var-template)
    endif()
endif()

# Add proper postfix for Python 2.7 on Windows when using a Debug interpreter:
set(MATERIALX_PYTHON_DEBUG_POSTFIX "")
if(MSVC AND PYTHON_IS_DEBUG AND (PYBIND11_PYTHON_VERSION VERSION_LESS 3.0))
    set(MATERIALX_PYTHON_DEBUG_POSTFIX "_d")
endif()

add_subdirectory(PyMaterialXCore)
add_subdirectory(PyMaterialXFormat)
if (MATERIALX_BUILD_GEN_GLSL OR MATERIALX_BUILD_GEN_OSL OR MATERIALX_BUILD_GEN_MDL OR MATERIALX_BUILD_GEN_MSL)
    add_subdirectory(PyMaterialXGenShader)
    if (MATERIALX_BUILD_GEN_GLSL)
        add_subdirectory(PyMaterialXGenGlsl)
    endif()
    if (MATERIALX_BUILD_GEN_MSL)
        add_subdirectory(PyMaterialXGenMsl)
    endif()
    if (MATERIALX_BUILD_GEN_OSL)
        add_subdirectory(PyMaterialXGenOsl)
    endif()
    if (MATERIALX_BUILD_GEN_MDL)
        add_subdirectory(PyMaterialXGenMdl)
    endif()
endif()
if (MATERIALX_BUILD_RENDER)
    add_subdirectory(PyMaterialXRender)
    if (MATERIALX_BUILD_GEN_GLSL)
        add_subdirectory(PyMaterialXRenderGlsl)
    endif()
    if (MATERIALX_BUILD_GEN_OSL)
        add_subdirectory(PyMaterialXRenderOsl)
    endif()
    if (APPLE AND MATERIALX_BUILD_GEN_MSL)
        add_subdirectory(PyMaterialXRenderMsl)
    endif()
endif()
