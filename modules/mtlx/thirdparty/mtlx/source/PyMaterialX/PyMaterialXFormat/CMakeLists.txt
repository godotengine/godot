file(GLOB pymaterialxformat_source "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB pymaterialxformat_headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

pybind11_add_module(PyMaterialXFormat SHARED ${PYBIND11_MODULE_FLAGS} ${pymaterialxformat_source} ${pymaterialxformat_headers})

if(APPLE)
    set_target_properties(PyMaterialXFormat PROPERTIES CXX_VISIBILITY_PRESET "default")
endif()

set_target_properties(
    PyMaterialXFormat
    PROPERTIES
    OUTPUT_NAME PyMaterialXFormat
    COMPILE_FLAGS "${EXTERNAL_COMPILE_FLAGS}"
    LINK_FLAGS "${EXTERNAL_LINK_FLAGS}"
    INSTALL_RPATH "${MATERIALX_UP_TWO_RPATH}"
    DEBUG_POSTFIX "${MATERIALX_PYTHON_DEBUG_POSTFIX}")

target_link_libraries(
    PyMaterialXFormat
    PUBLIC PyMaterialXCore
           MaterialXFormat
    PRIVATE ${CMAKE_DL_LIBS})

install(TARGETS PyMaterialXFormat
        DESTINATION "${MATERIALX_PYTHON_FOLDER_NAME}")
