find_package(Threads)

add_library(glslc STATIC
  src/file_compiler.cc
  src/file_compiler.h
  src/file.cc
  src/file.h
  src/file_includer.cc
  src/file_includer.h
  src/resource_parse.h
  src/resource_parse.cc
  src/shader_stage.cc
  src/shader_stage.h
  src/dependency_info.cc
  src/dependency_info.h
)

shaderc_default_compile_options(glslc)
target_include_directories(glslc PUBLIC ${glslang_SOURCE_DIR})
target_link_libraries(glslc PRIVATE glslang OSDependent OGLCompiler
  HLSL glslang SPIRV ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(glslc PRIVATE shaderc_util shaderc)

add_executable(glslc_exe src/main.cc)
shaderc_default_compile_options(glslc_exe)
target_include_directories(glslc_exe PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/.. ${spirv-tools_SOURCE_DIR}/include)
set_target_properties(glslc_exe PROPERTIES OUTPUT_NAME glslc)
target_link_libraries(glslc_exe PRIVATE glslc shaderc_util shaderc)
add_dependencies(glslc_exe build-version)

shaderc_add_tests(
  TEST_PREFIX glslc
  LINK_LIBS glslc shaderc_util shaderc
  TEST_NAMES
    file
    resource_parse
    stage)

shaderc_add_asciidoc(glslc_doc_README README)

if(SHADERC_ENABLE_INSTALL)
  install(TARGETS glslc_exe
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif(SHADERC_ENABLE_INSTALL)

add_subdirectory(test)
