file(GLOB pymaterialxrender_source "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB pymaterialxrender_headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
if(NOT MATERIALX_BUILD_OIIO)
    list(REMOVE_ITEM pymaterialxrender_source "${CMAKE_CURRENT_SOURCE_DIR}/PyOiioImageLoader.cpp")
endif()

pybind11_add_module(PyMaterialXRender SHARED ${PYBIND11_MODULE_FLAGS} ${pymaterialxrender_source} ${pymaterialxrender_headers})

if(APPLE)
    set_target_properties(PyMaterialXRender PROPERTIES CXX_VISIBILITY_PRESET "default")
endif()

set_target_properties(
    PyMaterialXRender
    PROPERTIES
    OUTPUT_NAME PyMaterialXRender
    COMPILE_FLAGS "${EXTERNAL_COMPILE_FLAGS}"
    LINK_FLAGS "${EXTERNAL_LINK_FLAGS}"
    INSTALL_RPATH "${MATERIALX_UP_TWO_RPATH}"
    DEBUG_POSTFIX "${MATERIALX_PYTHON_DEBUG_POSTFIX}")

target_link_libraries(
    PyMaterialXRender
    PUBLIC PyMaterialXGenShader
           MaterialXRender
    PRIVATE ${CMAKE_DL_LIBS})

install(TARGETS PyMaterialXRender
        DESTINATION "${MATERIALX_PYTHON_FOLDER_NAME}")
