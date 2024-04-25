file(GLOB pymaterialxgenosl_source "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB pymaterialxgenosl_headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

pybind11_add_module(PyMaterialXGenOsl SHARED ${PYBIND11_MODULE_FLAGS} ${pymaterialxgenosl_source} ${pymaterialxgenosl_headers})

if(APPLE)
    set_target_properties(PyMaterialXGenOsl PROPERTIES CXX_VISIBILITY_PRESET "default")
endif()

set_target_properties(
    PyMaterialXGenOsl
    PROPERTIES
    OUTPUT_NAME PyMaterialXGenOsl
    COMPILE_FLAGS "${EXTERNAL_COMPILE_FLAGS}"
    LINK_FLAGS "${EXTERNAL_LINK_FLAGS}"
    INSTALL_RPATH "${MATERIALX_UP_TWO_RPATH}"
    DEBUG_POSTFIX "${MATERIALX_PYTHON_DEBUG_POSTFIX}")

target_link_libraries(
    PyMaterialXGenOsl
    PUBLIC PyMaterialXGenShader
           MaterialXGenOsl
    PRIVATE ${CMAKE_DL_LIBS})

install(TARGETS PyMaterialXGenOsl
        DESTINATION "${MATERIALX_PYTHON_FOLDER_NAME}")
