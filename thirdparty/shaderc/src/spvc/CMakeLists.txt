add_executable(spvc_exe src/main.cc)
shaderc_default_compile_options(spvc_exe)
target_include_directories(spvc_exe PRIVATE ${shaderc_SOURCE_DIR}/libshaderc/include ${spirv-tools_SOURCE_DIR}/include)
set_target_properties(spvc_exe PROPERTIES OUTPUT_NAME spvc)
target_link_libraries(spvc_exe PRIVATE shaderc_spvc shaderc_util)
add_dependencies(spvc_exe build-version)

shaderc_add_asciidoc(spvc_doc_README README)

if(SHADERC_ENABLE_INSTALL)
  install(TARGETS spvc_exe
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif(SHADERC_ENABLE_INSTALL)

if(${SHADERC_ENABLE_TESTS})
  add_subdirectory(test)
endif()
