#==================================================================================================#
#  supported macros                                                                                #
#    - TEST_CASE,                                                                                  #
#    - TEMPLATE_TEST_CASE                                                                          #
#    - SCENARIO,                                                                                   #
#    - TEST_CASE_METHOD,                                                                           #
#    - CATCH_TEST_CASE,                                                                            #
#    - CATCH_TEMPLATE_TEST_CASE                                                                    #
#    - CATCH_SCENARIO,                                                                             #
#    - CATCH_TEST_CASE_METHOD.                                                                     #
#                                                                                                  #
#  Usage                                                                                           #
# 1. make sure this module is in the path or add this otherwise:                                   #
#    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake.modules/")              #
# 2. make sure that you've enabled testing option for the project by the call:                     #
#    enable_testing()                                                                              #
# 3. add the lines to the script for testing target (sample CMakeLists.txt):                       #
#        project(testing_target)                                                                   #
#        set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake.modules/")          #
#        enable_testing()                                                                          #
#                                                                                                  #
#        find_path(CATCH_INCLUDE_DIR "catch.hpp")                                                  #
#        include_directories(${INCLUDE_DIRECTORIES} ${CATCH_INCLUDE_DIR})                          #
#                                                                                                  #
#        file(GLOB SOURCE_FILES "*.cpp")                                                           #
#        add_executable(${PROJECT_NAME} ${SOURCE_FILES})                                           #
#                                                                                                  #
#        include(ParseAndAddCatchTests)                                                            #
#        ParseAndAddCatchTests(${PROJECT_NAME})                                                    #
#                                                                                                  #
# The following variables affect the behavior of the script:                                       #
#                                                                                                  #
#    PARSE_CATCH_TESTS_VERBOSE (Default OFF)                                                       #
#    -- enables debug messages                                                                     #
#    PARSE_CATCH_TESTS_NO_HIDDEN_TESTS (Default OFF)                                               #
#    -- excludes tests marked with [!hide], [.] or [.foo] tags                                     #
#    PARSE_CATCH_TESTS_ADD_FIXTURE_IN_TEST_NAME (Default ON)                                       #
#    -- adds fixture class name to the test name                                                   #
#    PARSE_CATCH_TESTS_ADD_TARGET_IN_TEST_NAME (Default ON)                                        #
#    -- adds cmake target name to the test name                                                    #
#    PARSE_CATCH_TESTS_ADD_TO_CONFIGURE_DEPENDS (Default OFF)                                      #
#    -- causes CMake to rerun when file with tests changes so that new tests will be discovered    #
#                                                                                                  #
# One can also set (locally) the optional variable OptionalCatchTestLauncher to precise the way    #
# a test should be run. For instance to use test MPI, one can write                                #
#     set(OptionalCatchTestLauncher ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${NUMPROC})                 #
# just before calling this ParseAndAddCatchTests function                                          #
#                                                                                                  #
# The AdditionalCatchParameters optional variable can be used to pass extra argument to the test   #
# command. For example, to include successful tests in the output, one can write                   #
#     set(AdditionalCatchParameters --success)                                                     #
#                                                                                                  #
# After the script, the ParseAndAddCatchTests_TESTS property for the target, and for each source   #
# file in the target is set, and contains the list of the tests extracted from that target, or     #
# from that file. This is useful, for example to add further labels or properties to the tests.    #
#                                                                                                  #
#==================================================================================================#

if(CMAKE_MINIMUM_REQUIRED_VERSION VERSION_LESS 2.8.8)
  message(FATAL_ERROR "ParseAndAddCatchTests requires CMake 2.8.8 or newer")
endif()

option(PARSE_CATCH_TESTS_VERBOSE "Print Catch to CTest parser debug messages" OFF)
option(PARSE_CATCH_TESTS_NO_HIDDEN_TESTS "Exclude tests with [!hide], [.] or [.foo] tags" OFF)
option(PARSE_CATCH_TESTS_ADD_FIXTURE_IN_TEST_NAME "Add fixture class name to the test name" ON)
option(PARSE_CATCH_TESTS_ADD_TARGET_IN_TEST_NAME "Add target name to the test name" ON)
option(PARSE_CATCH_TESTS_ADD_TO_CONFIGURE_DEPENDS "Add test file to CMAKE_CONFIGURE_DEPENDS property" OFF)

function(ParseAndAddCatchTests_PrintDebugMessage)
  if(PARSE_CATCH_TESTS_VERBOSE)
    message(STATUS "ParseAndAddCatchTests: ${ARGV}")
  endif()
endfunction()

# This removes the contents between
#  - block comments (i.e. /* ... */)
#  - full line comments (i.e. // ... )
# contents have been read into '${CppCode}'.
# !keep partial line comments
function(ParseAndAddCatchTests_RemoveComments CppCode)
  string(ASCII 2 CMakeBeginBlockComment)
  string(ASCII 3 CMakeEndBlockComment)
  string(REGEX REPLACE "/\\*" "${CMakeBeginBlockComment}" ${CppCode} "${${CppCode}}")
  string(REGEX REPLACE "\\*/" "${CMakeEndBlockComment}" ${CppCode} "${${CppCode}}")
  string(REGEX REPLACE "${CMakeBeginBlockComment}[^${CMakeEndBlockComment}]*${CMakeEndBlockComment}" "" ${CppCode} "${${CppCode}}")
  string(REGEX REPLACE "\n[ \t]*//+[^\n]+" "\n" ${CppCode} "${${CppCode}}")

  set(${CppCode} "${${CppCode}}" PARENT_SCOPE)
endfunction()

# Worker function
function(ParseAndAddCatchTests_ParseFile SourceFile TestTarget)
  # If SourceFile is an object library, do not scan it (as it is not a file). Exit without giving a warning about a missing file.
  if(SourceFile MATCHES "\\\$<TARGET_OBJECTS:.+>")
    ParseAndAddCatchTests_PrintDebugMessage("Detected OBJECT library: ${SourceFile} this will not be scanned for tests.")
    return()
  endif()
  # According to CMake docs EXISTS behavior is well-defined only for full paths.
  get_filename_component(SourceFile ${SourceFile} ABSOLUTE)
  if(NOT EXISTS ${SourceFile})
    message(WARNING "Cannot find source file: ${SourceFile}")
    return()
  endif()
  ParseAndAddCatchTests_PrintDebugMessage("parsing ${SourceFile}")
  file(STRINGS ${SourceFile} Contents NEWLINE_CONSUME)

  # Remove block and fullline comments
  ParseAndAddCatchTests_RemoveComments(Contents)

  # Find definition of test names
  # https://regex101.com/r/JygOND/1
  string(REGEX MATCHALL "[ \t]*(CATCH_)?(TEMPLATE_)?(TEST_CASE_METHOD|SCENARIO|TEST_CASE)[ \t]*\\([ \t\n]*\"[^\"]*\"[ \t\n]*(,[ \t\n]*\"[^\"]*\")?(,[ \t\n]*[^\,\)]*)*\\)[ \t\n]*\{+[ \t]*(//[^\n]*[Tt][Ii][Mm][Ee][Oo][Uu][Tt][ \t]*[0-9]+)*" Tests "${Contents}")

  if(PARSE_CATCH_TESTS_ADD_TO_CONFIGURE_DEPENDS AND Tests)
    ParseAndAddCatchTests_PrintDebugMessage("Adding ${SourceFile} to CMAKE_CONFIGURE_DEPENDS property")
    set_property(
      DIRECTORY
      APPEND
      PROPERTY CMAKE_CONFIGURE_DEPENDS ${SourceFile}
    )
  endif()

  # check CMP0110 policy for new add_test() behavior
  if(POLICY CMP0110)
    cmake_policy(GET CMP0110 _cmp0110_value) # new add_test() behavior
  else()
    # just to be thorough explicitly set the variable
    set(_cmp0110_value)
  endif()

  foreach(TestName ${Tests})
    # Strip newlines
    string(REGEX REPLACE "\\\\\n|\n" "" TestName "${TestName}")

    # Get test type and fixture if applicable
    string(REGEX MATCH "(CATCH_)?(TEMPLATE_)?(TEST_CASE_METHOD|SCENARIO|TEST_CASE)[ \t]*\\([^,^\"]*" TestTypeAndFixture "${TestName}")
    string(REGEX MATCH "(CATCH_)?(TEMPLATE_)?(TEST_CASE_METHOD|SCENARIO|TEST_CASE)" TestType "${TestTypeAndFixture}")
    string(REGEX REPLACE "${TestType}\\([ \t]*" "" TestFixture "${TestTypeAndFixture}")

    # Get string parts of test definition
    string(REGEX MATCHALL "\"+([^\\^\"]|\\\\\")+\"+" TestStrings "${TestName}")

    # Strip wrapping quotation marks
    string(REGEX REPLACE "^\"(.*)\"$" "\\1" TestStrings "${TestStrings}")
    string(REPLACE "\";\"" ";" TestStrings "${TestStrings}")

    # Validate that a test name and tags have been provided
    list(LENGTH TestStrings TestStringsLength)
    if(TestStringsLength GREATER 2 OR TestStringsLength LESS 1)
      message(FATAL_ERROR "You must provide a valid test name and tags for all tests in ${SourceFile}")
    endif()

    # Assign name and tags
    list(GET TestStrings 0 Name)
    if("${TestType}" STREQUAL "SCENARIO")
      set(Name "Scenario: ${Name}")
    endif()
    if(PARSE_CATCH_TESTS_ADD_FIXTURE_IN_TEST_NAME AND "${TestType}" MATCHES "(CATCH_)?TEST_CASE_METHOD" AND TestFixture)
      set(CTestName "${TestFixture}:${Name}")
    else()
      set(CTestName "${Name}")
    endif()
    if(PARSE_CATCH_TESTS_ADD_TARGET_IN_TEST_NAME)
      set(CTestName "${TestTarget}:${CTestName}")
    endif()
    # add target to labels to enable running all tests added from this target
    set(Labels ${TestTarget})
    if(TestStringsLength EQUAL 2)
      list(GET TestStrings 1 Tags)
      string(TOLOWER "${Tags}" Tags)
      # remove target from labels if the test is hidden
      if("${Tags}" MATCHES ".*\\[!?(hide|\\.)\\].*")
        list(REMOVE_ITEM Labels ${TestTarget})
      endif()
      string(REPLACE "]" ";" Tags "${Tags}")
      string(REPLACE "[" "" Tags "${Tags}")
    else()
      # unset tags variable from previous loop
      unset(Tags)
    endif()

    list(APPEND Labels ${Tags})

    set(HiddenTagFound OFF)
    foreach(label ${Labels})
      string(REGEX MATCH "^!hide|^\\." result ${label})
      if(result)
        set(HiddenTagFound ON)
        break()
      endif()
    endforeach(label)
    if(PARSE_CATCH_TESTS_NO_HIDDEN_TESTS AND ${HiddenTagFound} AND ${CMAKE_VERSION} VERSION_LESS "3.9")
      ParseAndAddCatchTests_PrintDebugMessage("Skipping test \"${CTestName}\" as it has [!hide], [.] or [.foo] label")
    else()
      ParseAndAddCatchTests_PrintDebugMessage("Adding test \"${CTestName}\"")
      if(Labels)
        ParseAndAddCatchTests_PrintDebugMessage("Setting labels to ${Labels}")
      endif()

      # Escape commas in the test spec
      string(REPLACE "," "\\," Name ${Name})

      # Work around CMake 3.18.0 change in `add_test()`, before the escaped quotes were necessary,
      # only with CMake 3.18.0 the escaped double quotes confuse the call. This change is reverted in 3.18.1
      # And properly introduced in 3.19 with the CMP0110 policy
      if(_cmp0110_value STREQUAL "NEW" OR ${CMAKE_VERSION} VERSION_EQUAL "3.18")
        ParseAndAddCatchTests_PrintDebugMessage("CMP0110 set to NEW, no need for add_test(\"\") workaround")
      else()
        ParseAndAddCatchTests_PrintDebugMessage("CMP0110 set to OLD adding \"\" for add_test() workaround")
        set(CTestName "\"${CTestName}\"")
      endif()

      # Handle template test cases
      if("${TestTypeAndFixture}" MATCHES ".*TEMPLATE_.*")
        set(Name "${Name} - *")
      endif()

      # Add the test and set its properties
      add_test(NAME "${CTestName}" COMMAND ${OptionalCatchTestLauncher} $<TARGET_FILE:${TestTarget}> ${Name} ${AdditionalCatchParameters})
      # Old CMake versions do not document VERSION_GREATER_EQUAL, so we use VERSION_GREATER with 3.8 instead
      if(PARSE_CATCH_TESTS_NO_HIDDEN_TESTS AND ${HiddenTagFound} AND ${CMAKE_VERSION} VERSION_GREATER "3.8")
        ParseAndAddCatchTests_PrintDebugMessage("Setting DISABLED test property")
        set_tests_properties("${CTestName}" PROPERTIES DISABLED ON)
      else()
        set_tests_properties("${CTestName}" PROPERTIES FAIL_REGULAR_EXPRESSION "No tests ran"
                                                LABELS "${Labels}")
      endif()
      set_property(
        TARGET ${TestTarget}
        APPEND
        PROPERTY ParseAndAddCatchTests_TESTS "${CTestName}")
      set_property(
        SOURCE ${SourceFile}
        APPEND
        PROPERTY ParseAndAddCatchTests_TESTS "${CTestName}")
    endif()
  endforeach()
endfunction()

# entry point
function(ParseAndAddCatchTests TestTarget)
  message(DEPRECATION "ParseAndAddCatchTest: function deprecated because of possibility of missed test cases. Consider using 'catch_discover_tests' from 'Catch.cmake'")
  ParseAndAddCatchTests_PrintDebugMessage("Started parsing ${TestTarget}")
  get_target_property(SourceFiles ${TestTarget} SOURCES)
  ParseAndAddCatchTests_PrintDebugMessage("Found the following sources: ${SourceFiles}")
  foreach(SourceFile ${SourceFiles})
    ParseAndAddCatchTests_ParseFile(${SourceFile} ${TestTarget})
  endforeach()
  ParseAndAddCatchTests_PrintDebugMessage("Finished parsing ${TestTarget}")
endfunction()
