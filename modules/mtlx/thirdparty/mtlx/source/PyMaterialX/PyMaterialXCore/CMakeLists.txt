file(GLOB pymaterialxcore_source "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB pymaterialxcore_headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

pybind11_add_module(PyMaterialXCore SHARED ${PYBIND11_MODULE_FLAGS} ${pymaterialxcore_source} ${pymaterialxcore_headers})

if(APPLE)
    set_target_properties(PyMaterialXCore PROPERTIES CXX_VISIBILITY_PRESET "default")
endif()

set_target_properties(
    PyMaterialXCore
    PROPERTIES
    OUTPUT_NAME PyMaterialXCore
    COMPILE_FLAGS "${EXTERNAL_COMPILE_FLAGS}"
    LINK_FLAGS "${EXTERNAL_LINK_FLAGS}"
    INSTALL_RPATH "${MATERIALX_UP_TWO_RPATH}"
    DEBUG_POSTFIX "${MATERIALX_PYTHON_DEBUG_POSTFIX}")

target_link_libraries(
    PyMaterialXCore
    PUBLIC MaterialXCore
    PRIVATE ${CMAKE_DL_LIBS})

install(TARGETS PyMaterialXCore
        DESTINATION "${MATERIALX_PYTHON_FOLDER_NAME}")
