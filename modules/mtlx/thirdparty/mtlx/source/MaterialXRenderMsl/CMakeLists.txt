set(MATERIALX_MODULE_NAME MaterialXRenderMsl)

file(GLOB_RECURSE materialx_source "${CMAKE_CURRENT_SOURCE_DIR}/*.m*")
file(GLOB_RECURSE materialx_headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h*")

assign_source_group("Source Files" ${materialx_source})
assign_source_group("Header Files" ${materialx_headers})

if(POLICY CMP0072)
    cmake_policy(SET CMP0072 NEW)
endif()

if(APPLE)
if(NOT MATERIALX_BUILD_IOS)
    find_library(COCOA_FRAMEWORK Cocoa)
    find_package(OpenGL REQUIRED)
endif()
    file(GLOB_RECURSE materialx_source_oc "${CMAKE_CURRENT_SOURCE_DIR}/*.m")
    message("Objective C files: " ${materialx_source_oc})
    set_source_files_properties(${materialx_source_oc} PROPERTIES
        COMPILE_FLAGS "-x objective-c++")
    set(materialx_source ${materialx_source} ${materialx_source_oc})
    add_compile_options(-DGL_SILENCE_DEPRECATION)
elseif(UNIX)
    find_package(X11 REQUIRED)
    # Note - can't just require the Xt component because FindX11 in cmake 3.1
    # doesn't support it
    if(NOT X11_Xt_FOUND)
        message(FATAL_ERROR "Error in building MaterialXRenderMsl: Xt was not found")
    endif()

    find_package(OpenGL REQUIRED)
    include_directories(${X11_INCLUDE_DIR})
endif()

# Disable OpenGL deprecation warnings on Clang.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wno-deprecated-declarations)
endif()

add_library(${MATERIALX_MODULE_NAME} ${materialx_source} ${materialx_headers})

add_definitions(-DMATERIALX_RENDERMSL_EXPORTS)

target_link_libraries(
    ${MATERIALX_MODULE_NAME}
    MaterialXRenderHw
    MaterialXGenMsl
    ${CMAKE_DL_LIBS})

if(MSVC)
    target_link_libraries(
        ${MATERIALX_MODULE_NAME}
        Opengl32)
elseif(APPLE)
    if(NOT MATERIALX_BUILD_IOS)
        target_link_libraries(
            ${MATERIALX_MODULE_NAME}
            "-framework Cocoa"
            OpenGL::GL)
    endif()
    target_link_libraries(
        ${MATERIALX_MODULE_NAME}
        "-framework Foundation"
        "-framework Metal")
elseif(UNIX)
    target_link_libraries(
        ${MATERIALX_MODULE_NAME}
        OpenGL::GL
        X11::X11
        X11::Xt)
endif()

set_target_properties(
    ${MATERIALX_MODULE_NAME} PROPERTIES
    OUTPUT_NAME ${MATERIALX_MODULE_NAME}${MATERIALX_LIBNAME_SUFFIX}
    COMPILE_FLAGS "${EXTERNAL_COMPILE_FLAGS}"
    LINK_FLAGS "${EXTERNAL_LINK_FLAGS}"
    VERSION "${MATERIALX_LIBRARY_VERSION}"
    SOVERSION "${MATERIALX_MAJOR_VERSION}")

target_include_directories(${MATERIALX_MODULE_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>
        $<INSTALL_INTERFACE:${MATERIALX_INSTALL_INCLUDE_PATH}>
    PRIVATE
        ${EXTERNAL_INCLUDE_DIRS})

if(NOT SKBUILD)
    install(TARGETS ${MATERIALX_MODULE_NAME}
        EXPORT MaterialX
        ARCHIVE DESTINATION ${MATERIALX_INSTALL_LIB_PATH}
        LIBRARY DESTINATION ${MATERIALX_INSTALL_LIB_PATH}
        RUNTIME DESTINATION bin)

    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
        DESTINATION ${MATERIALX_INSTALL_INCLUDE_PATH}/${MATERIALX_MODULE_NAME}/ MESSAGE_NEVER
        FILES_MATCHING PATTERN "*.h*")

    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/${MATERIALX_MODULE_NAME}.pdb"
        DESTINATION "${MATERIALX_INSTALL_LIB_PATH}/" OPTIONAL)
endif()
