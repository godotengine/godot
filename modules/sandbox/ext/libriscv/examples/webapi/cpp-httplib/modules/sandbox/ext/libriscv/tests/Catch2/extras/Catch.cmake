# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
Catch
-----

This module defines a function to help use the Catch test framework.

The :command:`catch_discover_tests` discovers tests by asking the compiled test
executable to enumerate its tests.  This does not require CMake to be re-run
when tests change.  However, it may not work in a cross-compiling environment,
and setting test properties is less convenient.

This command is intended to replace use of :command:`add_test` to register
tests, and will create a separate CTest test for each Catch test case.  Note
that this is in some cases less efficient, as common set-up and tear-down logic
cannot be shared by multiple test cases executing in the same instance.
However, it provides more fine-grained pass/fail information to CTest, which is
usually considered as more beneficial.  By default, the CTest test name is the
same as the Catch name; see also ``TEST_PREFIX`` and ``TEST_SUFFIX``.

.. command:: catch_discover_tests

  Automatically add tests with CTest by querying the compiled test executable
  for available tests::

    catch_discover_tests(target
                         [TEST_SPEC arg1...]
                         [EXTRA_ARGS arg1...]
                         [WORKING_DIRECTORY dir]
                         [TEST_PREFIX prefix]
                         [TEST_SUFFIX suffix]
                         [PROPERTIES name1 value1...]
                         [TEST_LIST var]
                         [REPORTER reporter]
                         [OUTPUT_DIR dir]
                         [OUTPUT_PREFIX prefix]
                         [OUTPUT_SUFFIX suffix]
                         [DISCOVERY_MODE <POST_BUILD|PRE_TEST>]
                         [SKIP_IS_FAILURE]
                         [ADD_TAGS_AS_LABELS]
    )

  ``catch_discover_tests`` sets up a post-build command on the test executable
  that generates the list of tests by parsing the output from running the test
  with the ``--list-test-names-only`` argument.  This ensures that the full
  list of tests is obtained.  Since test discovery occurs at build time, it is
  not necessary to re-run CMake when the list of tests changes.
  However, it requires that :prop_tgt:`CROSSCOMPILING_EMULATOR` is properly set
  in order to function in a cross-compiling environment.

  Additionally, setting properties on tests is somewhat less convenient, since
  the tests are not available at CMake time.  Additional test properties may be
  assigned to the set of tests as a whole using the ``PROPERTIES`` option.  If
  more fine-grained test control is needed, custom content may be provided
  through an external CTest script using the :prop_dir:`TEST_INCLUDE_FILES`
  directory property.  The set of discovered tests is made accessible to such a
  script via the ``<target>_TESTS`` variable.

  The options are:

  ``target``
    Specifies the Catch executable, which must be a known CMake executable
    target.  CMake will substitute the location of the built executable when
    running the test.

  ``TEST_SPEC arg1...``
    Specifies test cases, wildcarded test cases, tags and tag expressions to
    pass to the Catch executable with the ``--list-test-names-only`` argument.

  ``EXTRA_ARGS arg1...``
    Any extra arguments to pass on the command line to each test case.

  ``WORKING_DIRECTORY dir``
    Specifies the directory in which to run the discovered test cases.  If this
    option is not provided, the current binary directory is used.

  ``TEST_PREFIX prefix``
    Specifies a ``prefix`` to be prepended to the name of each discovered test
    case.  This can be useful when the same test executable is being used in
    multiple calls to ``catch_discover_tests()`` but with different
    ``TEST_SPEC`` or ``EXTRA_ARGS``.

  ``TEST_SUFFIX suffix``
    Similar to ``TEST_PREFIX`` except the ``suffix`` is appended to the name of
    every discovered test case.  Both ``TEST_PREFIX`` and ``TEST_SUFFIX`` may
    be specified.

  ``PROPERTIES name1 value1...``
    Specifies additional properties to be set on all tests discovered by this
    invocation of ``catch_discover_tests``.

  ``TEST_LIST var``
    Make the list of tests available in the variable ``var``, rather than the
    default ``<target>_TESTS``.  This can be useful when the same test
    executable is being used in multiple calls to ``catch_discover_tests()``.
    Note that this variable is only available in CTest.

  ``REPORTER reporter``
    Use the specified reporter when running the test case. The reporter will
    be passed to the Catch executable as ``--reporter reporter``.

  ``OUTPUT_DIR dir``
    If specified, the parameter is passed along as
    ``--out dir/<test_name>`` to Catch executable. The actual file name is the
    same as the test name. This should be used instead of
    ``EXTRA_ARGS --out foo`` to avoid race conditions writing the result output
    when using parallel test execution.

  ``OUTPUT_PREFIX prefix``
    May be used in conjunction with ``OUTPUT_DIR``.
    If specified, ``prefix`` is added to each output file name, like so
    ``--out dir/prefix<test_name>``.

  ``OUTPUT_SUFFIX suffix``
    May be used in conjunction with ``OUTPUT_DIR``.
    If specified, ``suffix`` is added to each output file name, like so
    ``--out dir/<test_name>suffix``. This can be used to add a file extension to
    the output e.g. ".xml".

  ``DL_PATHS path...``
    Specifies paths that need to be set for the dynamic linker to find shared
    libraries/DLLs when running the test executable (PATH/LD_LIBRARY_PATH respectively).
    These paths will both be set when retrieving the list of test cases from the
    test executable and when the tests are executed themselves. This requires
    cmake/ctest >= 3.22.

  ``DL_FRAMEWORK_PATHS path...``
    Specifies paths that need to be set for the dynamic linker to find libraries
    packaged as frameworks on Apple platforms when running the test executable
    (DYLD_FRAMEWORK_PATH). These paths will both be set when retrieving the list
    of test cases from the test executable and when the tests are executed themselves.
    This requires cmake/ctest >= 3.22.

  ``DISCOVERY_MODE mode``
    Provides control over when ``catch_discover_tests`` performs test discovery.
    By default, ``POST_BUILD`` sets up a post-build command to perform test discovery
    at build time. In certain scenarios, like cross-compiling, this ``POST_BUILD``
    behavior is not desirable. By contrast, ``PRE_TEST`` delays test discovery until
    just prior to test execution. This way test discovery occurs in the target environment
    where the test has a better chance at finding appropriate runtime dependencies.

    ``DISCOVERY_MODE`` defaults to the value of the
    ``CMAKE_CATCH_DISCOVER_TESTS_DISCOVERY_MODE`` variable if it is not passed when
    calling ``catch_discover_tests``. This provides a mechanism for globally selecting
    a preferred test discovery behavior without having to modify each call site.

  ``SKIP_IS_FAILURE``
    Disables skipped test detection.

  ``ADD_TAGS_AS_LABELS``
    Adds all test tags as CTest labels.

#]=======================================================================]

#------------------------------------------------------------------------------
function(catch_discover_tests TARGET)

  cmake_parse_arguments(
    ""
    "SKIP_IS_FAILURE;ADD_TAGS_AS_LABELS"
    "TEST_PREFIX;TEST_SUFFIX;WORKING_DIRECTORY;TEST_LIST;REPORTER;OUTPUT_DIR;OUTPUT_PREFIX;OUTPUT_SUFFIX;DISCOVERY_MODE"
    "TEST_SPEC;EXTRA_ARGS;PROPERTIES;DL_PATHS;DL_FRAMEWORK_PATHS"
    ${ARGN}
  )

  if(${CMAKE_VERSION} VERSION_LESS "3.19")
    message(FATAL_ERROR "This script requires JSON support from CMake version 3.19 or greater.")
  endif()

  if(NOT _WORKING_DIRECTORY)
    set(_WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
  endif()
  if(NOT _TEST_LIST)
    set(_TEST_LIST ${TARGET}_TESTS)
  endif()
  if(_DL_PATHS AND ${CMAKE_VERSION} VERSION_LESS "3.22.0")
    message(FATAL_ERROR "The DL_PATHS option requires at least cmake 3.22")
  endif()
  if(_DL_FRAMEWORK_PATHS AND ${CMAKE_VERSION} VERSION_LESS "3.22.0")
    message(FATAL_ERROR "The DL_FRAMEWORK_PATHS option requires at least cmake 3.22")
  endif()
  if(NOT _DISCOVERY_MODE)
    if(NOT CMAKE_CATCH_DISCOVER_TESTS_DISCOVERY_MODE)
      set(CMAKE_CATCH_DISCOVER_TESTS_DISCOVERY_MODE "POST_BUILD")
    endif()
    set(_DISCOVERY_MODE ${CMAKE_CATCH_DISCOVER_TESTS_DISCOVERY_MODE})
  endif()
  if(NOT _DISCOVERY_MODE MATCHES "^(POST_BUILD|PRE_TEST)$")
    message(FATAL_ERROR "Unknown DISCOVERY_MODE: ${_DISCOVERY_MODE}")
  endif()

  ## Generate a unique name based on the extra arguments
  string(SHA1 args_hash "${_TEST_SPEC} ${_EXTRA_ARGS} ${_REPORTER} ${_OUTPUT_DIR} ${_OUTPUT_PREFIX} ${_OUTPUT_SUFFIX}")
  string(SUBSTRING ${args_hash} 0 7 args_hash)

  # Define rule to generate test list for aforementioned test executable
  set(ctest_file_base "${CMAKE_CURRENT_BINARY_DIR}/${TARGET}-${args_hash}")
  set(ctest_include_file "${ctest_file_base}_include.cmake")
  set(ctest_tests_file "${ctest_file_base}_tests.cmake")

  get_property(crosscompiling_emulator
    TARGET ${TARGET}
    PROPERTY CROSSCOMPILING_EMULATOR
  )
  if(NOT _SKIP_IS_FAILURE)
    set(_PROPERTIES ${_PROPERTIES} SKIP_RETURN_CODE 4)
  endif()

  if(_DISCOVERY_MODE STREQUAL "POST_BUILD")
    add_custom_command(
      TARGET ${TARGET} POST_BUILD
      BYPRODUCTS "${ctest_tests_file}"
      COMMAND "${CMAKE_COMMAND}"
              -D "TEST_TARGET=${TARGET}"
              -D "TEST_EXECUTABLE=$<TARGET_FILE:${TARGET}>"
              -D "TEST_EXECUTOR=${crosscompiling_emulator}"
              -D "TEST_WORKING_DIR=${_WORKING_DIRECTORY}"
              -D "TEST_SPEC=${_TEST_SPEC}"
              -D "TEST_EXTRA_ARGS=${_EXTRA_ARGS}"
              -D "TEST_PROPERTIES=${_PROPERTIES}"
              -D "TEST_PREFIX=${_TEST_PREFIX}"
              -D "TEST_SUFFIX=${_TEST_SUFFIX}"
              -D "TEST_LIST=${_TEST_LIST}"
              -D "TEST_REPORTER=${_REPORTER}"
              -D "TEST_OUTPUT_DIR=${_OUTPUT_DIR}"
              -D "TEST_OUTPUT_PREFIX=${_OUTPUT_PREFIX}"
              -D "TEST_OUTPUT_SUFFIX=${_OUTPUT_SUFFIX}"
              -D "TEST_DL_PATHS=${_DL_PATHS}"
              -D "TEST_DL_FRAMEWORK_PATHS=${_DL_FRAMEWORK_PATHS}"
              -D "CTEST_FILE=${ctest_tests_file}"
              -D "ADD_TAGS_AS_LABELS=${_ADD_TAGS_AS_LABELS}"
              -P "${_CATCH_DISCOVER_TESTS_SCRIPT}"
      VERBATIM
    )

    file(WRITE "${ctest_include_file}"
      "if(EXISTS \"${ctest_tests_file}\")\n"
      "  include(\"${ctest_tests_file}\")\n"
      "else()\n"
      "  add_test(${TARGET}_NOT_BUILT-${args_hash} ${TARGET}_NOT_BUILT-${args_hash})\n"
      "endif()\n"
    )

  elseif(_DISCOVERY_MODE STREQUAL "PRE_TEST")

    get_property(GENERATOR_IS_MULTI_CONFIG GLOBAL
        PROPERTY GENERATOR_IS_MULTI_CONFIG
    )

    if(GENERATOR_IS_MULTI_CONFIG)
      set(ctest_tests_file "${ctest_file_base}_tests-$<CONFIG>.cmake")
    endif()

    string(CONCAT ctest_include_content
      "if(EXISTS \"$<TARGET_FILE:${TARGET}>\")"                                    "\n"
      "  if(NOT EXISTS \"${ctest_tests_file}\" OR"                                 "\n"
      "     NOT \"${ctest_tests_file}\" IS_NEWER_THAN \"$<TARGET_FILE:${TARGET}>\" OR\n"
      "     NOT \"${ctest_tests_file}\" IS_NEWER_THAN \"\${CMAKE_CURRENT_LIST_FILE}\")\n"
      "    include(\"${_CATCH_DISCOVER_TESTS_SCRIPT}\")"                           "\n"
      "    catch_discover_tests_impl("                                             "\n"
      "      TEST_EXECUTABLE"        " [==[" "$<TARGET_FILE:${TARGET}>"   "]==]"   "\n"
      "      TEST_EXECUTOR"          " [==[" "${crosscompiling_emulator}" "]==]"   "\n"
      "      TEST_WORKING_DIR"       " [==[" "${_WORKING_DIRECTORY}"      "]==]"   "\n"
      "      TEST_SPEC"              " [==[" "${_TEST_SPEC}"              "]==]"   "\n"
      "      TEST_EXTRA_ARGS"        " [==[" "${_EXTRA_ARGS}"             "]==]"   "\n"
      "      TEST_PROPERTIES"        " [==[" "${_PROPERTIES}"             "]==]"   "\n"
      "      TEST_PREFIX"            " [==[" "${_TEST_PREFIX}"            "]==]"   "\n"
      "      TEST_SUFFIX"            " [==[" "${_TEST_SUFFIX}"            "]==]"   "\n"
      "      TEST_LIST"              " [==[" "${_TEST_LIST}"              "]==]"   "\n"
      "      TEST_REPORTER"          " [==[" "${_REPORTER}"               "]==]"   "\n"
      "      TEST_OUTPUT_DIR"        " [==[" "${_OUTPUT_DIR}"             "]==]"   "\n"
      "      TEST_OUTPUT_PREFIX"     " [==[" "${_OUTPUT_PREFIX}"          "]==]"   "\n"
      "      TEST_OUTPUT_SUFFIX"     " [==[" "${_OUTPUT_SUFFIX}"          "]==]"   "\n"
      "      CTEST_FILE"             " [==[" "${ctest_tests_file}"        "]==]"   "\n"
      "      TEST_DL_PATHS"          " [==[" "${_DL_PATHS}"               "]==]"   "\n"
      "      TEST_DL_FRAMEWORK_PATHS" " [==[" "${_DL_FRAMEWORK_PATHS}"     "]==]"   "\n"
      "      ADD_TAGS_AS_LABELS"     " [==[" "${_ADD_TAGS_AS_LABELS}"     "]==]"   "\n"
      "    )"                                                                      "\n"
      "  endif()"                                                                  "\n"
      "  include(\"${ctest_tests_file}\")"                                         "\n"
      "else()"                                                                     "\n"
      "  add_test(${TARGET}_NOT_BUILT ${TARGET}_NOT_BUILT)"                        "\n"
      "endif()"                                                                    "\n"
    )

    if(GENERATOR_IS_MULTI_CONFIG)
      foreach(_config ${CMAKE_CONFIGURATION_TYPES})
        file(GENERATE OUTPUT "${ctest_file_base}_include-${_config}.cmake" CONTENT "${ctest_include_content}" CONDITION $<CONFIG:${_config}>)
      endforeach()
      string(CONCAT ctest_include_multi_content
        "if(NOT CTEST_CONFIGURATION_TYPE)"                                              "\n"
        "  message(\"No configuration for testing specified, use '-C <cfg>'.\")"        "\n"
        "else()"                                                                        "\n"
        "  include(\"${ctest_file_base}_include-\${CTEST_CONFIGURATION_TYPE}.cmake\")"  "\n"
        "endif()"                                                                       "\n"
      )
      file(GENERATE OUTPUT "${ctest_include_file}" CONTENT "${ctest_include_multi_content}")
    else()
      file(GENERATE OUTPUT "${ctest_file_base}_include.cmake" CONTENT "${ctest_include_content}")
      file(WRITE "${ctest_include_file}" "include(\"${ctest_file_base}_include.cmake\")")
    endif()
  endif()

  # Add discovered tests to directory TEST_INCLUDE_FILES
  set_property(DIRECTORY
    APPEND PROPERTY TEST_INCLUDE_FILES "${ctest_include_file}"
  )

endfunction()

###############################################################################

set(_CATCH_DISCOVER_TESTS_SCRIPT
  ${CMAKE_CURRENT_LIST_DIR}/CatchAddTests.cmake
  CACHE INTERNAL "Catch2 full path to CatchAddTests.cmake helper file"
)
