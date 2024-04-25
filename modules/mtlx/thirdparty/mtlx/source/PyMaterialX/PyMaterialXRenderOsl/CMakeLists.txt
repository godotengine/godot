file(GLOB pymaterialxrenderosl_source "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB pymaterialxrenderosl_headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

pybind11_add_module(PyMaterialXRenderOsl SHARED ${PYBIND11_MODULE_FLAGS} ${pymaterialxrenderosl_source} ${pymaterialxrenderosl_headers})

if(APPLE)
    set_target_properties(PyMaterialXRenderOsl PROPERTIES CXX_VISIBILITY_PRESET "default")
endif()

set_target_properties(
    PyMaterialXRenderOsl
    PROPERTIES
    OUTPUT_NAME PyMaterialXRenderOsl
    COMPILE_FLAGS "${EXTERNAL_COMPILE_FLAGS}"
    LINK_FLAGS "${EXTERNAL_LINK_FLAGS}"
    INSTALL_RPATH "${MATERIALX_UP_TWO_RPATH}"
    DEBUG_POSTFIX "${MATERIALX_PYTHON_DEBUG_POSTFIX}")

target_link_libraries(
    PyMaterialXRenderOsl
    PUBLIC PyMaterialXRender
           MaterialXRenderOsl
    PRIVATE ${CMAKE_DL_LIBS})

install(TARGETS PyMaterialXRenderOsl
        DESTINATION "${MATERIALX_PYTHON_FOLDER_NAME}")
