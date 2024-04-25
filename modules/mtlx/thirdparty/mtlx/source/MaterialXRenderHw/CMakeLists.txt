set(MATERIALX_MODULE_NAME MaterialXRenderHw)

file(GLOB materialx_source "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB materialx_headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h*")

if(APPLE)
if (NOT MATERIALX_BUILD_IOS)
  find_library(COCOA_FRAMEWORK Cocoa)
endif()
  file(GLOB materialx_source_oc "${CMAKE_CURRENT_SOURCE_DIR}/*.m")
  message("Objective C files: " ${materialx_source_oc})
  set_source_files_properties(${materialx_source_oc} PROPERTIES
                              COMPILE_FLAGS "-x objective-c++")
  set(materialx_source ${materialx_source} ${materialx_source_oc})
elseif(UNIX)
  find_package(X11 REQUIRED)
  # Note - can't just require the Xt component because FindX11 in cmake 3.1
  # doesn't support it
  if(NOT X11_Xt_FOUND)
    message(FATAL_ERROR "Error in building MaterialXRenderHw: Xt was not found")
  endif()
endif()

assign_source_group("Source Files" ${materialx_source})
assign_source_group("Header Files" ${materialx_headers})

add_library(${MATERIALX_MODULE_NAME} ${materialx_source} ${materialx_headers})

add_definitions(-DMATERIALX_RENDERHW_EXPORTS)

if(APPLE AND NOT MATERIALX_BUILD_IOS)
set(CMAKE_DL_LIBS
${CMAKE_DL_LIBS}
"-framework Cocoa")
endif()

# Create version resource
if(MATERIALX_BUILD_SHARED_LIBS AND MSVC)
    configure_file(${CMAKE_SOURCE_DIR}/cmake/modules/MaterialXVersion.rc.in ${CMAKE_CURRENT_BINARY_DIR}/version.rc)
    target_sources(${MATERIALX_MODULE_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/version.rc)
endif()

if(MSVC)
    target_link_libraries(
        ${MATERIALX_MODULE_NAME}
        MaterialXRender
        ${CMAKE_DL_LIBS})
elseif(APPLE)
    target_link_libraries(
        ${MATERIALX_MODULE_NAME}
        MaterialXRender
        ${CMAKE_DL_LIBS}
        "-framework Foundation"
        "-framework Metal")
elseif(UNIX)
    target_link_libraries(
        ${MATERIALX_MODULE_NAME}
        MaterialXRender
        ${CMAKE_DL_LIBS}
        X11::X11
        X11::Xt)
endif()

set_target_properties(
    ${MATERIALX_MODULE_NAME} PROPERTIES
    OUTPUT_NAME ${MATERIALX_MODULE_NAME}${MATERIALX_LIBNAME_SUFFIX}
    COMPILE_FLAGS "${EXTERNAL_COMPILE_FLAGS}"
    LINK_FLAGS "${EXTERNAL_LINK_FLAGS}"
    INSTALL_RPATH "${MATERIALX_SAME_DIR_RPATH}"
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
