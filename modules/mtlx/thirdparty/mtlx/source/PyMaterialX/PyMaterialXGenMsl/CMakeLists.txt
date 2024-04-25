file(GLOB pymaterialxgenmsl_source "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB pymaterialxgenmsl_headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

pybind11_add_module(PyMaterialXGenMsl SHARED ${PYBIND11_MODULE_FLAGS} ${pymaterialxgenmsl_source} ${pymaterialxgenmsl_headers})

if(APPLE)
    set_target_properties(PyMaterialXGenMsl PROPERTIES CXX_VISIBILITY_PRESET "default")
endif()

set_target_properties(
    PyMaterialXGenMsl
    PROPERTIES
    OUTPUT_NAME PyMaterialXGenMsl
    COMPILE_FLAGS "${EXTERNAL_COMPILE_FLAGS}"
    LINK_FLAGS "${EXTERNAL_LINK_FLAGS}"
    INSTALL_RPATH "${MATERIALX_UP_TWO_RPATH}"
    DEBUG_POSTFIX "${MATERIALX_PYTHON_DEBUG_POSTFIX}")

target_link_libraries(
    PyMaterialXGenMsl
    PUBLIC PyMaterialXGenShader
           MaterialXGenMsl
    PRIVATE ${CMAKE_DL_LIBS})

install(TARGETS PyMaterialXGenMsl
        DESTINATION "${MATERIALX_PYTHON_FOLDER_NAME}")
