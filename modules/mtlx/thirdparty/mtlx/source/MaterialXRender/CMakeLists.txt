set(MATERIALX_MODULE_NAME MaterialXRender)

include_directories(
    ${EXTERNAL_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../)

file(GLOB_RECURSE materialx_source "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB_RECURSE materialx_inlined "${CMAKE_CURRENT_SOURCE_DIR}/*.inl")
file(GLOB_RECURSE materialx_headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h*")

if(NOT MATERIALX_BUILD_OIIO)
    list(REMOVE_ITEM materialx_source "${CMAKE_CURRENT_SOURCE_DIR}/OiioImageLoader.cpp")
endif()

assign_source_group("Source Files" ${materialx_source})
assign_source_group("Source Files" ${materialx_inlined})
assign_source_group("Header Files" ${materialx_headers})

if(UNIX)
    add_compile_options(-Wno-unused-function)
endif()

add_library(${MATERIALX_MODULE_NAME} ${materialx_source} ${materialx_headers} ${materialx_inlined})

add_definitions(-DMATERIALX_RENDER_EXPORTS)

# Create version resource
if(MATERIALX_BUILD_SHARED_LIBS AND MSVC)
    configure_file(${CMAKE_SOURCE_DIR}/cmake/modules/MaterialXVersion.rc.in ${CMAKE_CURRENT_BINARY_DIR}/version.rc)
    target_sources(${MATERIALX_MODULE_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/version.rc)
endif()

target_link_libraries(
    ${MATERIALX_MODULE_NAME}
    MaterialXGenShader
    ${CMAKE_DL_LIBS})

if(MATERIALX_BUILD_OIIO)
    set(OPENIMAGEIO_ROOT_DIR ${MATERIALX_OIIO_DIR})
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/External/OpenImageIO")
    find_package(OpenImageIO REQUIRED)
    if(OPENIMAGEIO_FOUND)
        include_directories(${OPENIMAGEIO_INCLUDE_DIR})
        target_link_libraries(${MATERIALX_MODULE_NAME} ${OPENIMAGEIO_LIBRARIES})
        # Also needed by MaterialXTest:
        set(OPENIMAGEIO_FOUND "${OPENIMAGEIO_FOUND}" PARENT_SCOPE)
        set(OPENIMAGEIO_INCLUDE_DIR "${OPENIMAGEIO_INCLUDE_DIR}" PARENT_SCOPE)
        set(OPENIMAGEIO_LIBRARIES "${OPENIMAGEIO_LIBRARIES}" PARENT_SCOPE)
        set(OPENIMAGEIO_LIBRARY_DIRS "${OPENIMAGEIO_LIBRARY_DIRS}" PARENT_SCOPE)
        set(OPENIMAGEIO_ROOT_DIR "${OPENIMAGEIO_ROOT_DIR}" PARENT_SCOPE)
    endif()
endif()

set_target_properties(
    ${MATERIALX_MODULE_NAME} PROPERTIES
    OUTPUT_NAME ${MATERIALX_MODULE_NAME}${MATERIALX_LIBNAME_SUFFIX}
    COMPILE_FLAGS "${EXTERNAL_COMPILE_FLAGS}"
    LINK_FLAGS "${EXTERNAL_LINK_FLAGS}"
    INSTALL_RPATH "${MATERIALX_SAME_DIR_RPATH}"
    VERSION "${MATERIALX_LIBRARY_VERSION}"
    SOVERSION "${MATERIALX_MAJOR_VERSION}")

if(NOT SKBUILD)
    install(TARGETS ${MATERIALX_MODULE_NAME}
        EXPORT MaterialX
        ARCHIVE DESTINATION ${MATERIALX_INSTALL_LIB_PATH}
        LIBRARY DESTINATION ${MATERIALX_INSTALL_LIB_PATH}
        RUNTIME DESTINATION bin)

    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
        DESTINATION ${MATERIALX_INSTALL_INCLUDE_PATH}/${MATERIALX_MODULE_NAME}/ MESSAGE_NEVER
        FILES_MATCHING
        PATTERN "*.h*"
        PATTERN "*.inl")

    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/${MATERIALX_MODULE_NAME}.pdb"
        DESTINATION "${MATERIALX_INSTALL_LIB_PATH}/" OPTIONAL)
endif()
