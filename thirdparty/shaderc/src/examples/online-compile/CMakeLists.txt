add_executable(shaderc-online-compile main.cc)
shaderc_default_compile_options(shaderc-online-compile)
target_include_directories(shaderc-online-compile PUBLIC ${shaderc_SOURCE_DIR}/libshaderc_util/include)
target_link_libraries(shaderc-online-compile PRIVATE shaderc)
