project(libshaderc)

# Even though shaderc.hpp is a headers-only library, adding
# a dependency here will force clients of the library to rebuild
# when it changes.
set(SPVC_SOURCES
  include/shaderc/spvc.h
  include/shaderc/spvc.hpp
  src/spvc.cc
)

add_library(shaderc_spvc STATIC ${SPVC_SOURCES})
shaderc_default_compile_options(shaderc_spvc)
target_include_directories(shaderc_spvc PUBLIC include PRIVATE ${shaderc_SOURCE_DIR}/libshaderc/include ${shaderc_SOURCE_DIR}/libshaderc_util/include ${spirv-tools_SOURCE_DIR}/include ${SPIRV-Cross_SOURCE_DIR}/..)

add_library(shaderc_spvc_shared SHARED ${SPVC_SOURCES})
shaderc_default_compile_options(shaderc_spvc_shared)
target_include_directories(shaderc_spvc_shared PUBLIC include PRIVATE ${shaderc_SOURCE_DIR}/libshaderc/include ${shaderc_SOURCE_DIR}/libshaderc_util/include ${spirv-tools_SOURCE_DIR}/include ${SPIRV-Cross_SOURCE_DIR}/..)

target_compile_definitions(shaderc_spvc_shared
    PRIVATE SHADERC_IMPLEMENTATION
    PUBLIC SHADERC_SHAREDLIB
)

if (DISABLE_EXCEPTIONS)
  target_compile_definitions(shaderc_spvc PRIVATE SPIRV_CROSS_EXCEPTIONS_TO_ASSERTIONS)
  target_compile_definitions(shaderc_spvc_shared PRIVATE SPIRV_CROSS_EXCEPTIONS_TO_ASSERTIONS)
endif (DISABLE_EXCEPTIONS)

set_target_properties(shaderc_spvc_shared PROPERTIES SOVERSION 1)

if(SHADERC_ENABLE_INSTALL)
  install(
    FILES
      include/shaderc/spvc.h
      include/shaderc/spvc.hpp
    DESTINATION
      ${CMAKE_INSTALL_INCLUDEDIR}/shaderc)

  install(TARGETS shaderc_spvc shaderc_spvc_shared
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif(SHADERC_ENABLE_INSTALL)

find_package(Threads)
set(SPVC_LIBS
  ${CMAKE_THREAD_LIBS_INIT}
  SPIRV-Tools
  spirv-cross-glsl
  spirv-cross-hlsl
  spirv-cross-msl
)

target_link_libraries(shaderc_spvc PRIVATE ${SPVC_LIBS})
target_link_libraries(shaderc_spvc_shared PRIVATE ${SPVC_LIBS})

shaderc_add_tests(
  TEST_PREFIX shaderc
  LINK_LIBS shaderc_spvc
  INCLUDE_DIRS include ${shaderc_SOURCE_DIR}/libshaderc/include ${SPIRV-Cross_SOURCE_DIR}/..
  TEST_NAMES
    spvc
    spvc_cpp)

shaderc_add_tests(
  TEST_PREFIX shaderc_shared
  LINK_LIBS shaderc_spvc_shared SPIRV-Tools
  INCLUDE_DIRS include ${shaderc_SOURCE_DIR}/libshaderc/include ${SPIRV-Cross_SOURCE_DIR}/..
  TEST_NAMES
    spvc
    spvc_cpp)

shaderc_combine_static_lib(shaderc_spvc_combined shaderc_spvc)

if(SHADERC_ENABLE_INSTALL)
  # Since shaderc_combined is defined as an imported library, we cannot use the
  # install() directive to install it. Install it like a normal file.
  get_target_property(generated_location shaderc_spvc_combined LOCATION)
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
  TEST_PREFIX shaderc_spvc_combined
  LINK_LIBS shaderc_spvc_combined ${CMAKE_THREAD_LIBS_INIT} shaderc_util
  INCLUDE_DIRS include ${shaderc_SOURCE_DIR}/libshaderc/include ${spirv-tools_SOURCE_DIR}/include
  TEST_NAMES
    spvc
    spvc_cpp)

if(${SHADERC_ENABLE_TESTS})
  add_executable(spvc_c_smoke_test ./src/spvc_c_smoke_test.c)
  target_include_directories(spvc_c_smoke_test PUBLIC include PRIVATE ${shaderc_SOURCE_DIR}/libshaderc/include)
  shaderc_default_c_compile_options(spvc_c_smoke_test)
  target_link_libraries(spvc_c_smoke_test PRIVATE shaderc_spvc shaderc)
  add_test(NAME spvc_c_smoke_test COMMAND spvc_c_smoke_test)
endif()
