# utility functions

function (shaderc_use_gmock TARGET)
  target_include_directories(${TARGET} PRIVATE
    ${gmock_SOURCE_DIR}/include
    ${gtest_SOURCE_DIR}/include)
  target_link_libraries(${TARGET} PRIVATE gmock gtest_main)
endfunction(shaderc_use_gmock)

function(shaderc_default_c_compile_options TARGET)
  if (NOT "${MSVC}")
    target_compile_options(${TARGET} PRIVATE -Wall -Werror -fvisibility=hidden)
    if (NOT "${MINGW}")
      target_compile_options(${TARGET} PRIVATE -fPIC)
    endif()
    if (ENABLE_CODE_COVERAGE)
      # The --coverage option is a synonym for -fprofile-arcs -ftest-coverage
      # when compiling.
      target_compile_options(${TARGET} PRIVATE -g -O0 --coverage)
      # The --coverage option is a synonym for -lgcov when linking for gcc.
      # For clang, it links in a different library, libclang_rt.profile, which
      # requires clang to be built with compiler-rt.
      target_link_libraries(${TARGET} PRIVATE --coverage)
    endif()
    if (NOT SHADERC_ENABLE_SHARED_CRT)
      if (WIN32)
        # For MinGW cross compile, statically link to the libgcc runtime.
        # But it still depends on MSVCRT.dll.
        set_target_properties(${TARGET} PROPERTIES
          LINK_FLAGS "-static -static-libgcc")
      endif(WIN32)
    endif(NOT SHADERC_ENABLE_SHARED_CRT)
  else()
    # disable warning C4800: 'int' : forcing value to bool 'true' or 'false'
    # (performance warning)
    target_compile_options(${TARGET} PRIVATE /wd4800)
  endif()
endfunction(shaderc_default_c_compile_options)

function(shaderc_default_compile_options TARGET)
  shaderc_default_c_compile_options(${TARGET})
  if (NOT "${MSVC}")
    target_compile_options(${TARGET} PRIVATE -std=c++11)
    if (NOT SHADERC_ENABLE_SHARED_CRT)
      if (WIN32)
        # For MinGW cross compile, statically link to the C++ runtime.
        # But it still depends on MSVCRT.dll.
        set_target_properties(${TARGET} PROPERTIES
          LINK_FLAGS "-static -static-libgcc -static-libstdc++")
      endif(WIN32)
    endif(NOT SHADERC_ENABLE_SHARED_CRT)
  endif()
endfunction(shaderc_default_compile_options)

# Build an asciidoc file; additional arguments past the base filename specify
# additional dependencies for the file.
function(shaderc_add_asciidoc TARGET FILE)
  if (ASCIIDOCTOR_EXE)
    set(DEST ${CMAKE_CURRENT_BINARY_DIR}/${FILE}.html)
    add_custom_command(
      COMMAND ${ASCIIDOCTOR_EXE} -a toc -o ${DEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/${FILE}.asciidoc
      DEPENDS ${FILE}.asciidoc ${ARGN}
      OUTPUT ${DEST})
    # Create the target, but the default build target does not depend on it.
    # Some Asciidoctor installations are mysteriously broken, and it's hard
    # to detect those cases.  Generating HTML is not critical by default.
    add_custom_target(${TARGET} DEPENDS ${DEST})
  endif(ASCIIDOCTOR_EXE)
endfunction()

# Run nosetests on file ${PREFIX}_nosetest.py. Nosetests will look for classes
# and functions whose names start with "nosetest". The test name will be
# ${PREFIX}_nosetests.
function(shaderc_add_nosetests PREFIX)
  if("${SHADERC_ENABLE_TESTS}" AND NOSETESTS_EXE)
    add_test(
      NAME ${PREFIX}_nosetests
      COMMAND ${NOSETESTS_EXE} -m "^[Nn]ose[Tt]est" -v
        ${CMAKE_CURRENT_SOURCE_DIR}/${PREFIX}_nosetest.py)
  endif()
endfunction()

# Adds a set of tests.
# This function accepts the following parameters:
# TEST_PREFIX:  a prefix for each test target name
# TEST_NAMES:   a list of test names where each TEST_NAME has a corresponding
#               file residing at src/${TEST_NAME}_test.cc
# LINK_LIBS:    (optional) a list of libraries to be linked to the test target
# INCLUDE_DIRS: (optional) a list of include directories to be searched
#               for header files.
function(shaderc_add_tests)
  if(${SHADERC_ENABLE_TESTS})
    cmake_parse_arguments(PARSED_ARGS
      ""
      "TEST_PREFIX"
      "TEST_NAMES;LINK_LIBS;INCLUDE_DIRS"
      ${ARGN})
    if (NOT PARSED_ARGS_TEST_NAMES)
      message(FATAL_ERROR "Tests must have a target")
    endif()
    if (NOT PARSED_ARGS_TEST_PREFIX)
      message(FATAL_ERROR "Tests must have a prefix")
    endif()
    foreach(TARGET ${PARSED_ARGS_TEST_NAMES})
      set(TEST_NAME ${PARSED_ARGS_TEST_PREFIX}_${TARGET}_test)
      add_executable(${TEST_NAME} src/${TARGET}_test.cc)
      shaderc_default_compile_options(${TEST_NAME})
      if (MINGW)
        target_compile_options(${TEST_NAME} PRIVATE -DSHADERC_DISABLE_THREADED_TESTS)
      endif()
      if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        # Disable this warning, which is useless in test code.
        # Fixes https://github.com/google/shaderc/issues/334
        target_compile_options(${TEST_NAME} PRIVATE -Wno-noexcept-type)
      endif()
      if (PARSED_ARGS_LINK_LIBS)
        target_link_libraries(${TEST_NAME} PRIVATE
        ${PARSED_ARGS_LINK_LIBS})
      endif()
      if (PARSED_ARGS_INCLUDE_DIRS)
        target_include_directories(${TEST_NAME} PRIVATE
        ${PARSED_ARGS_INCLUDE_DIRS})
      endif()
      shaderc_use_gmock(${TEST_NAME})
      add_test(
        NAME ${PARSED_ARGS_TEST_PREFIX}_${TARGET}
        COMMAND ${TEST_NAME})
    endforeach()
  endif(${SHADERC_ENABLE_TESTS})
endfunction(shaderc_add_tests)

# Finds all transitive static library dependencies of a given target
# including possibly the target itself.
# This will skip libraries that were statically linked that were not
# built by CMake, for example -lpthread.
macro(shaderc_get_transitive_libs target out_list)
  if (TARGET ${target})
    get_target_property(libtype ${target} TYPE)
    # If this target is a static library, get anything it depends on.
    if ("${libtype}" STREQUAL "STATIC_LIBRARY")
      list(INSERT ${out_list} 0 "${target}")
      get_target_property(libs ${target} LINK_LIBRARIES)
      if (libs)
        foreach(lib ${libs})
          shaderc_get_transitive_libs(${lib} ${out_list})
        endforeach()
      endif()
    endif()
  endif()
  # If we know the location (i.e. if it was made with CMake) then we
  # can add it to our list.
  LIST(REMOVE_DUPLICATES ${out_list})
endmacro()

# Combines the static library "target" with all of its transitive static
# library dependencies into a single static library "new_target".
function(shaderc_combine_static_lib new_target target)

  set(all_libs "")
  shaderc_get_transitive_libs(${target} all_libs)

  set(libname
      ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${new_target}${CMAKE_STATIC_LIBRARY_SUFFIX})

  if (MSVC)
    string(REPLACE ";" ">;$<TARGET_FILE:" temp_string "${all_libs}")
    set(lib_target_list "$<TARGET_FILE:${temp_string}>")

    add_custom_command(OUTPUT ${libname}
      DEPENDS ${all_libs}
      COMMAND lib.exe ${lib_target_list} /OUT:${libname} /NOLOGO)
  elseif(APPLE)
    string(REPLACE ";" ">;$<TARGET_FILE:" temp_string "${all_libs}")
    set(lib_target_list "$<TARGET_FILE:${temp_string}>")

    add_custom_command(OUTPUT ${libname}
      DEPENDS ${all_libs}
      COMMAND libtool -static -o ${libname} ${lib_target_list})
  else()
    string(REPLACE ";" "> \naddlib $<TARGET_FILE:" temp_string "${all_libs}")
    set(start_of_file
      "create ${libname}\naddlib $<TARGET_FILE:${temp_string}>")
    set(build_script_file "${start_of_file}\nsave\nend\n")

    file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${new_target}.ar"
        CONTENT ${build_script_file}
        CONDITION 1)

    add_custom_command(OUTPUT  ${libname}
      DEPENDS ${all_libs}
      COMMAND ${CMAKE_AR} -M < ${new_target}.ar)
  endif()

  add_custom_target(${new_target}_genfile ALL
    DEPENDS ${libname})

  # CMake needs to be able to see this as another normal library,
  # so import the newly created library as an imported library,
  # and set up the dependencies on the custom target.
  add_library(${new_target} STATIC IMPORTED)
  set_target_properties(${new_target}
    PROPERTIES IMPORTED_LOCATION ${libname})
  add_dependencies(${new_target} ${new_target}_genfile)
endfunction()
