file(GLOB pymaterialxgenglsl_source "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB pymaterialxgenglsl_headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

pybind11_add_module(PyMaterialXGenGlsl SHARED ${PYBIND11_MODULE_FLAGS} ${pymaterialxgenglsl_source} ${pymaterialxgenglsl_headers})

if(APPLE)
    set_target_properties(PyMaterialXGenGlsl PROPERTIES CXX_VISIBILITY_PRESET "default")
endif()

set_target_properties(
    PyMaterialXGenGlsl
    PROPERTIES
    OUTPUT_NAME PyMaterialXGenGlsl
    COMPILE_FLAGS "${EXTERNAL_COMPILE_FLAGS}"
    LINK_FLAGS "${EXTERNAL_LINK_FLAGS}"
    INSTALL_RPATH "${MATERIALX_UP_TWO_RPATH}"
    DEBUG_POSTFIX "${MATERIALX_PYTHON_DEBUG_POSTFIX}")

target_link_libraries(
    PyMaterialXGenGlsl
    PUBLIC PyMaterialXGenShader
           MaterialXGenGlsl
    PRIVATE ${CMAKE_DL_LIBS})

install(TARGETS PyMaterialXGenGlsl
        DESTINATION "${MATERIALX_PYTHON_FOLDER_NAME}")
