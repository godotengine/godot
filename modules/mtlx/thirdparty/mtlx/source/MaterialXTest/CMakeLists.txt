file(GLOB materialx_source "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB materialx_headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h*")
list(APPEND materialx_headers "${CMAKE_CURRENT_SOURCE_DIR}/External/Catch/catch.hpp")

assign_source_group("Source Files" ${materialx_source})
assign_source_group("Header Files" ${materialx_headers})

# Discover all tests and allow them to be run in parallel (ctest -j20):
function(add_tests _sources)
  foreach(src_file ${_sources})
    file(STRINGS ${src_file} matched_lines REGEX "TEST_CASE")
    foreach(matched_line ${matched_lines})
      string(REGEX REPLACE "(TEST_CASE[( \"]+)" "" test_name ${matched_line})
      string(REGEX REPLACE "(\".*)" "" test_name ${test_name})
      string(REGEX REPLACE "[^A-Za-z0-9_]+" "_" test_safe_name ${test_name})
      add_test(NAME "MaterialXTest_${test_safe_name}"
          COMMAND MaterialXTest ${test_name}
          WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
      if(MATERIALX_BUILD_OIIO AND MSVC)
        # Add path to OIIO library so it can be found for the test.
        # On windows we have to escape the semicolons, otherwise only
        # the first path entry will be passed to the test executable
        STRING(REPLACE ";" "\\;" TESTPATH "$ENV{PATH}")
        STRING(APPEND TESTPATH "\\;${OPENIMAGEIO_ROOT_DIR}/bin")
        STRING(REPLACE "/" "\\" TESTPATH "${TESTPATH}")
        set_tests_properties("MaterialXTest_${test_safe_name}" PROPERTIES
                             ENVIRONMENT "PATH=${TESTPATH}")
      endif()
    endforeach()
  endforeach()
endfunction()

add_executable(MaterialXTest ${materialx_source} ${materialx_headers})

target_include_directories( MaterialXTest PUBLIC
    ${EXTERNAL_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../)

add_subdirectory(MaterialXCore)
target_link_libraries(MaterialXTest MaterialXCore)
add_subdirectory(MaterialXFormat)
target_link_libraries(MaterialXTest MaterialXFormat)

add_subdirectory(MaterialXGenShader)
target_link_libraries(MaterialXTest MaterialXGenShader)
if(MATERIALX_BUILD_GEN_GLSL OR MATERIALX_BUILD_GEN_OSL OR MATERIALX_BUILD_GEN_MDL OR MATERIALX_BUILD_GEN_MSL)
  if(MATERIALX_BUILD_GEN_GLSL)
    add_subdirectory(MaterialXGenGlsl)
    target_link_libraries(MaterialXTest MaterialXGenGlsl)
  endif()
  if(MATERIALX_BUILD_GEN_OSL)
    add_subdirectory(MaterialXGenOsl)
    target_link_libraries(MaterialXTest MaterialXGenOsl)
  endif()
  if(MATERIALX_BUILD_GEN_MDL)
    add_subdirectory(MaterialXGenMdl)
    target_link_libraries(MaterialXTest MaterialXGenMdl)
    get_target_property(MaterialXGenMdl_SOURCE_DIR MaterialXGenMdl SOURCE_DIR)
    target_compile_definitions(MaterialXTest PRIVATE -DMATERIALX_MDL_IMPL_MODULE_PATH=\"${MaterialXGenMdl_SOURCE_DIR}/mdl\")
  endif()
  if(MATERIALX_BUILD_GEN_MSL)
    add_subdirectory(MaterialXGenMsl)
    target_link_libraries(MaterialXTest MaterialXGenMsl)
  endif()
endif()

if(MATERIALX_BUILD_RENDER)
  add_subdirectory(MaterialXRender)
  target_link_libraries(MaterialXTest MaterialXRender)
  if (MATERIALX_BUILD_OIIO)
    target_link_libraries(MaterialXTest ${OPENIMAGEIO_LIBRARIES})
  endif()
  if(MATERIALX_BUILD_GEN_GLSL)
    add_subdirectory(MaterialXRenderGlsl)
    target_link_libraries(MaterialXTest MaterialXRenderGlsl)
  endif()
  if(MATERIALX_BUILD_GEN_OSL)
    add_subdirectory(MaterialXRenderOsl)
    target_link_libraries(MaterialXTest MaterialXRenderOsl)
  endif()
  if(APPLE AND MATERIALX_BUILD_GEN_MSL)
    add_subdirectory(MaterialXRenderMsl)
    target_link_libraries(MaterialXTest MaterialXRenderMsl)
  endif()
  if(MATERIALX_BUILD_OIIO AND OPENIMAGEIO_ROOT_DIR)
      add_custom_command(TARGET MaterialXTest POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_directory
      ${OPENIMAGEIO_ROOT_DIR}/bin ${CMAKE_CURRENT_BINARY_DIR})
  endif()
endif()

# TODO: Only do this stuff when it's relevant

add_custom_command(TARGET MaterialXTest POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/../../libraries ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/libraries)
if(MATERIALX_OSL_LEGACY_CLOSURES)
    add_custom_command(TARGET MaterialXTest POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E rename
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/libraries/pbrlib/genosl/pbrlib_genosl_impl.legacy ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/libraries/pbrlib/genosl/pbrlib_genosl_impl.mtlx)
endif()

if(MATERIALX_BUILD_GEN_MDL)
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../source/MaterialXGenMdl/mdl/"
        DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${MATERIALX_INSTALL_MDL_MODULE_PATH}/mdl)
endif()

add_custom_command(TARGET MaterialXTest POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/../../resources ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/resources)

add_custom_command(TARGET MaterialXTest POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/MaterialXRenderOsl/Utilities ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/resources/Utilities)

set_target_properties(
    MaterialXTest PROPERTIES
    OUTPUT_NAME MaterialXTest
    COMPILE_FLAGS "${EXTERNAL_COMPILE_FLAGS}"
    LINK_FLAGS "${EXTERNAL_LINK_FLAGS}"
    VERSION "${MATERIALX_LIBRARY_VERSION}"
    SOVERSION "${MATERIALX_MAJOR_VERSION}")

if(MATERIALX_BUILD_BENCHMARK_TESTS)
    target_compile_definitions(MaterialXTest PRIVATE -DCATCH_CONFIG_ENABLE_BENCHMARKING)
endif()

target_link_libraries(
    MaterialXTest
    ${CMAKE_DL_LIBS})
