file(GLOB pymaterialxrendermsl_source "${CMAKE_CURRENT_SOURCE_DIR}/*.mm")
file(GLOB pymaterialxrendermsl_headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

pybind11_add_module(PyMaterialXRenderMsl SHARED ${PYBIND11_MODULE_FLAGS} ${pymaterialxrendermsl_source} ${pymaterialxrendermsl_headers})

if(APPLE)
    set_target_properties(PyMaterialXRenderMsl PROPERTIES CXX_VISIBILITY_PRESET "default")
endif()

set_target_properties(
    PyMaterialXRenderMsl
    PROPERTIES
    OUTPUT_NAME PyMaterialXRenderMsl
    COMPILE_FLAGS "${EXTERNAL_COMPILE_FLAGS}"
    LINK_FLAGS "${EXTERNAL_LINK_FLAGS}"
    INSTALL_RPATH "${MATERIALX_UP_TWO_RPATH}"
    DEBUG_POSTFIX "${MATERIALX_PYTHON_DEBUG_POSTFIX}")

target_link_libraries(
    PyMaterialXRenderMsl
    PUBLIC PyMaterialXRender
           MaterialXRenderMsl
           PyMaterialXGenMsl
           MaterialXGenMsl
    PRIVATE ${CMAKE_DL_LIBS})

install(TARGETS PyMaterialXRenderMsl
        DESTINATION "${MATERIALX_PYTHON_FOLDER_NAME}")
