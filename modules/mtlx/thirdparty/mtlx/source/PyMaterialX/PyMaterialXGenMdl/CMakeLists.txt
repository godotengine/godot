file(GLOB pymaterialxgenmdl_source "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB pymaterialxgenmdl_headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

pybind11_add_module(PyMaterialXGenMdl SHARED ${PYBIND11_MODULE_FLAGS} ${pymaterialxgenmdl_source} ${pymaterialxgenmdl_headers})

if(APPLE)
    set_target_properties(PyMaterialXGenMdl PROPERTIES CXX_VISIBILITY_PRESET "default")
endif()

set_target_properties(
    PyMaterialXGenMdl
    PROPERTIES
    OUTPUT_NAME PyMaterialXGenMdl
    COMPILE_FLAGS "${EXTERNAL_COMPILE_FLAGS}"
    LINK_FLAGS "${EXTERNAL_LINK_FLAGS}"
    INSTALL_RPATH "${MATERIALX_UP_TWO_RPATH}"
    DEBUG_POSTFIX "${MATERIALX_PYTHON_DEBUG_POSTFIX}")

target_link_libraries(
    PyMaterialXGenMdl
    PUBLIC MaterialXGenMdl
    PRIVATE ${CMAKE_DL_LIBS})

install(TARGETS PyMaterialXGenMdl
        DESTINATION "${MATERIALX_PYTHON_FOLDER_NAME}")
