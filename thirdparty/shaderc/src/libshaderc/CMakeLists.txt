project(libshaderc)

# Even though shaderc.hpp is a headers-only library, adding
# a dependency here will force clients of the library to rebuild
# when it changes.
set(SHADERC_SOURCES
  include/shaderc/shaderc.h
  include/shaderc/shaderc.hpp
  src/shaderc.cc
  src/shaderc_private.h
)

add_library(shaderc STATIC ${SHADERC_SOURCES})
shaderc_default_compile_options(shaderc)
target_include_directories(shaderc PUBLIC include PRIVATE ${glslang_SOURCE_DIR})

add_library(shaderc_shared SHARED ${SHADERC_SOURCES})
shaderc_default_compile_options(shaderc_shared)
target_include_directories(shaderc_shared PUBLIC include PRIVATE ${glslang_SOURCE_DIR})
target_compile_definitions(shaderc_shared
    PRIVATE SHADERC_IMPLEMENTATION
    PUBLIC SHADERC_SHAREDLIB
)
set_target_properties(shaderc_shared PROPERTIES SOVERSION 1)

if(SHADERC_ENABLE_INSTALL)
  install(
    FILES
      include/shaderc/env.h
      include/shaderc/status.h
      include/shaderc/visibility.h
      include/shaderc/shaderc.h
      include/shaderc/shaderc.hpp
    DESTINATION
      ${CMAKE_INSTALL_INCLUDEDIR}/shaderc)

  install(TARGETS shaderc shaderc_shared
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif(SHADERC_ENABLE_INSTALL)

find_package(Threads)
set(SHADERC_LIBS
  glslang OSDependent OGLCompiler glslang ${CMAKE_THREAD_LIBS_INIT}
  shaderc_util
  SPIRV # from glslang
  SPIRV-Tools
)

target_link_libraries(shaderc PRIVATE ${SHADERC_LIBS})
target_link_libraries(shaderc_shared PRIVATE ${SHADERC_LIBS})

shaderc_add_tests(
  TEST_PREFIX shaderc
  LINK_LIBS shaderc
  INCLUDE_DIRS include ${shaderc_SOURCE_DIR}/libshaderc_util/include ${glslang_SOURCE_DIR}
               ${spirv-tools_SOURCE_DIR}/include
  TEST_NAMES
    shaderc
    shaderc_cpp
    shaderc_private)

shaderc_add_tests(
  TEST_PREFIX shaderc_shared
  LINK_LIBS shaderc_shared SPIRV-Tools
  INCLUDE_DIRS include ${shaderc_SOURCE_DIR}/libshaderc_util/include ${glslang_SOURCE_DIR}
               ${spirv-tools_SOURCE_DIR}/include
  TEST_NAMES
    shaderc
    shaderc_cpp
    shaderc_private)

shaderc_combine_static_lib(shaderc_combined shaderc)

if(SHADERC_ENABLE_INSTALL)
  # Since shaderc_combined is defined as an imported library, we cannot use the
  # install() directive to install it. Install it like a normal file.
  get_target_property(generated_location shaderc_combined LOCATION)
  string(REGEX MATCH "Visual Studio .*" vs_generator "${CMAKE_GENERATOR}")
  if (NOT "${vs_generator}" STREQUAL "")
    # With Visual Studio generators, the LOCATION property is not properly
    # expanded according to the current build configuration. We need to work
    # around this problem by manually substitution.
    string(REPLACE "$(Configuration)" "\${CMAKE_INSTALL_CONFIG_NAME}"
      install_location "${generated_location}")
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${install_location} DESTINATION ${CMAKE_INSTALL_LIBDIR})
  else()
    install(FILES ${generated_location} DESTINATION ${CMAKE_INSTALL_LIBDIR})
  endif()
endif(SHADERC_ENABLE_INSTALL)

shaderc_add_tests(
  TEST_PREFIX shaderc_combined
  LINK_LIBS shaderc_combined ${CMAKE_THREAD_LIBS_INIT}
  INCLUDE_DIRS include ${shaderc_SOURCE_DIR}/libshaderc_util/include ${glslang_SOURCE_DIR}
               ${spirv-tools_SOURCE_DIR}/include
  TEST_NAMES
    shaderc
    shaderc_cpp)

if(${SHADERC_ENABLE_TESTS})
  add_executable(shaderc_c_smoke_test ./src/shaderc_c_smoke_test.c)
  shaderc_default_c_compile_options(shaderc_c_smoke_test)
  target_include_directories(shaderc_c_smoke_test PUBLIC ${shaderc_SOURCE_DIR}/libshaderc_util/include)
  target_link_libraries(shaderc_c_smoke_test PRIVATE shaderc)
  add_test(NAME shaderc_c_smoke_test COMMAND shaderc_c_smoke_test)
endif()
