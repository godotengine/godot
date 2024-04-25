file(GLOB pymaterialxrenderglsl_source "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB pymaterialxrenderglsl_headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

pybind11_add_module(PyMaterialXRenderGlsl SHARED ${PYBIND11_MODULE_FLAGS} ${pymaterialxrenderglsl_source} ${pymaterialxrenderglsl_headers})

if(APPLE)
    set_target_properties(PyMaterialXRenderGlsl PROPERTIES CXX_VISIBILITY_PRESET "default")
endif()

set_target_properties(
    PyMaterialXRenderGlsl
    PROPERTIES
    OUTPUT_NAME PyMaterialXRenderGlsl
    COMPILE_FLAGS "${EXTERNAL_COMPILE_FLAGS}"
    LINK_FLAGS "${EXTERNAL_LINK_FLAGS}"
    INSTALL_RPATH "${MATERIALX_UP_TWO_RPATH}"
    DEBUG_POSTFIX "${MATERIALX_PYTHON_DEBUG_POSTFIX}")

target_link_libraries(
    PyMaterialXRenderGlsl
    PUBLIC PyMaterialXRender
           MaterialXRenderGlsl
           PyMaterialXGenGlsl
           MaterialXGenGlsl
    PRIVATE ${CMAKE_DL_LIBS})

install(TARGETS PyMaterialXRenderGlsl
        DESTINATION "${MATERIALX_PYTHON_FOLDER_NAME}")
