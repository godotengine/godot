include(CatchMiscFunctions)

if(CATCH_BUILD_SURROGATES)
  message(STATUS "Configuring targets for surrogate TUs")

  # If the folder does not exist before we ask for output redirect to
  # a file, it won't work.
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/surrogates)

  # Creates target to generate the surrogate TU for provided header.
  # Returns the path to the generated file.
  function(createSurrogateFileTarget sourceHeader pathToFile)
    set(pathPrefix ${PROJECT_SOURCE_DIR}/src)
    file(RELATIVE_PATH includePath ${pathPrefix} ${sourceHeader})

    get_filename_component(basicFileName "${sourceHeader}" NAME_WE)
    set(surrogateFilePath ${CMAKE_CURRENT_BINARY_DIR}/surrogates/surrogate_${basicFileName}.cpp)

    add_custom_command(
      OUTPUT ${surrogateFilePath}
      COMMAND cmake -E echo "\#include <${includePath}>" > "${surrogateFilePath}"
      VERBATIM
    )

    set(${pathToFile} ${surrogateFilePath} PARENT_SCOPE)
  endfunction()

  # Extracts all non-helper (e.g. catch_all.hpp) headers from the
  # Catch2 target, and returns them through the argument.
  function(ExtractCatch2Headers OutArg)
    get_target_property(targetSources Catch2 SOURCES)
    foreach(Source ${targetSources})
      string(REGEX MATCH "^.*\\.hpp$" isHeader ${Source})
      string(REGEX MATCH "_all.hpp" isAllHeader ${Source})
      if(isHeader AND NOT isAllHeader)
        list(APPEND AllHeaders ${Source})
      endif()
    endforeach()
    set(${OutArg} ${AllHeaders} PARENT_SCOPE)
  endfunction()

  ExtractCatch2Headers(mainHeaders)

  if(NOT mainHeaders)
    message(FATAL_ERROR "No headers in the main target were detected. Something is broken.")
  endif()

  foreach(header ${mainHeaders})
    createSurrogateFileTarget(${header} pathToGeneratedFile)
    list(APPEND surrogateFiles ${pathToGeneratedFile})
  endforeach()

  add_executable(Catch2SurrogateTarget
    ${surrogateFiles}
  )
  target_link_libraries(Catch2SurrogateTarget PRIVATE Catch2WithMain)
endif()

####
# Temporary workaround for VS toolset changes in 2017
# We need to disable <UseFullPaths> property, but CMake doesn't support it
# until 3.13 (not yet released)
####
if(MSVC)
  configure_file(${CATCH_DIR}/tools/misc/SelfTest.vcxproj.user ${CMAKE_BINARY_DIR}/tests COPYONLY)
endif() #Temporary workaround

# define the sources of the self test
# Please keep these ordered alphabetically
set(TEST_SOURCES
  ${SELF_TEST_DIR}/TestRegistrations.cpp
  ${SELF_TEST_DIR}/IntrospectiveTests/Algorithms.tests.cpp
  ${SELF_TEST_DIR}/IntrospectiveTests/AssertionHandler.tests.cpp
  ${SELF_TEST_DIR}/IntrospectiveTests/Clara.tests.cpp
  ${SELF_TEST_DIR}/IntrospectiveTests/CmdLine.tests.cpp
  ${SELF_TEST_DIR}/IntrospectiveTests/CmdLineHelpers.tests.cpp
  ${SELF_TEST_DIR}/IntrospectiveTests/ColourImpl.tests.cpp
  ${SELF_TEST_DIR}/IntrospectiveTests/Details.tests.cpp
  ${SELF_TEST_DIR}/IntrospectiveTests/FloatingPoint.tests.cpp
  ${SELF_TEST_DIR}/IntrospectiveTests/GeneratorsImpl.tests.cpp
  ${SELF_TEST_DIR}/IntrospectiveTests/Integer.tests.cpp
  ${SELF_TEST_DIR}/IntrospectiveTests/InternalBenchmark.tests.cpp
  ${SELF_TEST_DIR}/IntrospectiveTests/Json.tests.cpp
  ${SELF_TEST_DIR}/IntrospectiveTests/Parse.tests.cpp
  ${SELF_TEST_DIR}/IntrospectiveTests/PartTracker.tests.cpp
  ${SELF_TEST_DIR}/IntrospectiveTests/RandomNumberGeneration.tests.cpp
  ${SELF_TEST_DIR}/IntrospectiveTests/Reporters.tests.cpp
  ${SELF_TEST_DIR}/IntrospectiveTests/Tag.tests.cpp
  ${SELF_TEST_DIR}/IntrospectiveTests/TestCaseInfoHasher.tests.cpp
  ${SELF_TEST_DIR}/IntrospectiveTests/TestSpec.tests.cpp
  ${SELF_TEST_DIR}/IntrospectiveTests/TestSpecParser.tests.cpp
  ${SELF_TEST_DIR}/IntrospectiveTests/TextFlow.tests.cpp
  ${SELF_TEST_DIR}/IntrospectiveTests/Sharding.tests.cpp
  ${SELF_TEST_DIR}/IntrospectiveTests/Stream.tests.cpp
  ${SELF_TEST_DIR}/IntrospectiveTests/String.tests.cpp
  ${SELF_TEST_DIR}/IntrospectiveTests/StringManip.tests.cpp
  ${SELF_TEST_DIR}/IntrospectiveTests/Xml.tests.cpp
  ${SELF_TEST_DIR}/IntrospectiveTests/Traits.tests.cpp
  ${SELF_TEST_DIR}/IntrospectiveTests/ToString.tests.cpp
  ${SELF_TEST_DIR}/IntrospectiveTests/UniquePtr.tests.cpp
  ${SELF_TEST_DIR}/helpers/parse_test_spec.cpp
  ${SELF_TEST_DIR}/TimingTests/Sleep.tests.cpp
  ${SELF_TEST_DIR}/UsageTests/Approx.tests.cpp
  ${SELF_TEST_DIR}/UsageTests/BDD.tests.cpp
  ${SELF_TEST_DIR}/UsageTests/Benchmark.tests.cpp
  ${SELF_TEST_DIR}/UsageTests/Class.tests.cpp
  ${SELF_TEST_DIR}/UsageTests/Compilation.tests.cpp
  ${SELF_TEST_DIR}/UsageTests/Condition.tests.cpp
  ${SELF_TEST_DIR}/UsageTests/Decomposition.tests.cpp
  ${SELF_TEST_DIR}/UsageTests/EnumToString.tests.cpp
  ${SELF_TEST_DIR}/UsageTests/Exception.tests.cpp
  ${SELF_TEST_DIR}/UsageTests/Generators.tests.cpp
  ${SELF_TEST_DIR}/UsageTests/Message.tests.cpp
  ${SELF_TEST_DIR}/UsageTests/Misc.tests.cpp
  ${SELF_TEST_DIR}/UsageTests/Skip.tests.cpp
  ${SELF_TEST_DIR}/UsageTests/ToStringByte.tests.cpp
  ${SELF_TEST_DIR}/UsageTests/ToStringChrono.tests.cpp
  ${SELF_TEST_DIR}/UsageTests/ToStringGeneral.tests.cpp
  ${SELF_TEST_DIR}/UsageTests/ToStringOptional.tests.cpp
  ${SELF_TEST_DIR}/UsageTests/ToStringPair.tests.cpp
  ${SELF_TEST_DIR}/UsageTests/ToStringTuple.tests.cpp
  ${SELF_TEST_DIR}/UsageTests/ToStringVariant.tests.cpp
  ${SELF_TEST_DIR}/UsageTests/ToStringVector.tests.cpp
  ${SELF_TEST_DIR}/UsageTests/ToStringWhich.tests.cpp
  ${SELF_TEST_DIR}/UsageTests/Tricky.tests.cpp
  ${SELF_TEST_DIR}/UsageTests/VariadicMacros.tests.cpp
  ${SELF_TEST_DIR}/UsageTests/MatchersRanges.tests.cpp
  ${SELF_TEST_DIR}/UsageTests/Matchers.tests.cpp
)

set(TEST_HEADERS
  ${SELF_TEST_DIR}/helpers/parse_test_spec.hpp
  ${SELF_TEST_DIR}/helpers/range_test_helpers.hpp
  ${SELF_TEST_DIR}/helpers/type_with_lit_0_comparisons.hpp
)

# Specify the headers, too, so CLion recognises them as project files
set(HEADERS
  ${TOP_LEVEL_HEADERS}
  ${EXTERNAL_HEADERS}
  ${INTERNAL_HEADERS}
  ${REPORTER_HEADERS}
  ${BENCHMARK_HEADERS}
  ${BENCHMARK_SOURCES}
)

# Provide some groupings for IDEs
#source_group("benchmark" FILES ${BENCHMARK_HEADERS} ${BENCHMARK_SOURCES})
#source_group("Tests" FILES ${TEST_SOURCES})

include(CTest)

add_executable(SelfTest ${TEST_SOURCES} ${TEST_HEADERS})
target_include_directories(SelfTest PRIVATE ${SELF_TEST_DIR})
target_link_libraries(SelfTest PRIVATE Catch2WithMain)
if(BUILD_SHARED_LIBS AND WIN32)
  add_custom_command(TARGET SelfTest PRE_LINK
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:Catch2>
    $<TARGET_FILE:Catch2WithMain> $<TARGET_FILE_DIR:SelfTest>
  )
endif()

if(CATCH_ENABLE_COVERAGE)
  set(ENABLE_COVERAGE ON CACHE BOOL "Enable coverage build." FORCE)
  find_package(codecov)
  add_coverage(SelfTest)
  list(APPEND LCOV_REMOVE_PATTERNS "'/usr/*'")
  coverage_evaluate()
endif()

# configure unit tests via CTest
add_test(NAME RunTests COMMAND $<TARGET_FILE:SelfTest> --order rand --rng-seed time)
set_tests_properties(RunTests PROPERTIES
  FAIL_REGULAR_EXPRESSION "Filters:"
  COST 15
)

# Because CTest does not allow us to check both return code _and_ expected
# output in one test, we run these commands twice. First time we check
# the output, the second time we check the exit code.
add_test(NAME List::Tests::Output COMMAND $<TARGET_FILE:SelfTest> --list-tests --verbosity high)
set_tests_properties(List::Tests::Output PROPERTIES
  PASS_REGULAR_EXPRESSION "[0-9]+ test cases"
  FAIL_REGULAR_EXPRESSION "Hidden Test"
)
# This should be equivalent to the old --list-test-names-only and be usable
# with --input-file.
add_test(NAME List::Tests::Quiet COMMAND $<TARGET_FILE:SelfTest> --list-tests --verbosity quiet)
# Sadly we cannot ask for start-of-line and end-of-line in a ctest regex,
# so we fail if we see space/tab at the start...
set_tests_properties(List::Tests::Quiet PROPERTIES
  PASS_REGULAR_EXPRESSION "\"#1905 -- test spec parser properly clears internal state between compound tests\"[\r\n]"
  FAIL_REGULAR_EXPRESSION "[ \t]\"#1905 -- test spec parser properly clears internal state between compound tests\""
)
add_test(NAME List::Tests::ExitCode COMMAND $<TARGET_FILE:SelfTest> --list-tests --verbosity high)
add_test(NAME List::Tests::XmlOutput COMMAND $<TARGET_FILE:SelfTest> --list-tests --verbosity high -r xml)
set_tests_properties(List::Tests::XmlOutput PROPERTIES
  PASS_REGULAR_EXPRESSION "<Line>[0-9]+</Line>"
  FAIL_REGULAR_EXPRESSION "[0-9]+ test cases"
)

add_test(NAME List::Tags::Output COMMAND $<TARGET_FILE:SelfTest> --list-tags)
set_tests_properties(List::Tags::Output PROPERTIES
  PASS_REGULAR_EXPRESSION "[0-9]+ tags"
  FAIL_REGULAR_EXPRESSION "\\[\\.\\]")
add_test(NAME List::Tags::ExitCode COMMAND $<TARGET_FILE:SelfTest> --list-tags)
add_test(NAME List::Tags::XmlOutput COMMAND $<TARGET_FILE:SelfTest> --list-tags -r xml)
set_tests_properties(List::Tags::XmlOutput PROPERTIES
  PASS_REGULAR_EXPRESSION "<Count>18</Count>"
  FAIL_REGULAR_EXPRESSION "[0-9]+ tags"
)

add_test(NAME List::Reporters::Output COMMAND $<TARGET_FILE:SelfTest> --list-reporters)
set_tests_properties(List::Reporters::Output PROPERTIES PASS_REGULAR_EXPRESSION "Available reporters:")
add_test(NAME List::Reporters::ExitCode COMMAND $<TARGET_FILE:SelfTest> --list-reporters)
add_test(NAME List::Reporters::XmlOutput COMMAND $<TARGET_FILE:SelfTest> --list-reporters -r xml)
set_tests_properties(List::Reporters::XmlOutput PROPERTIES
  PASS_REGULAR_EXPRESSION "<Name>compact</Name>"
  FAIL_REGULAR_EXPRESSION "Available reporters:"
)

add_test(NAME List::Listeners::Output
  COMMAND
    $<TARGET_FILE:SelfTest> --list-listeners
)
set_tests_properties(List::Listeners::Output
  PROPERTIES
    PASS_REGULAR_EXPRESSION "Registered listeners:"
)
add_test(NAME List::Listeners::ExitCode
  COMMAND
    $<TARGET_FILE:SelfTest> --list-listeners
)
add_test(NAME List::Listeners::XmlOutput
  COMMAND
    $<TARGET_FILE:SelfTest>
      --list-listeners
      --reporter xml
)
set_tests_properties(List::Listeners::XmlOutput
  PROPERTIES
    PASS_REGULAR_EXPRESSION "<RegisteredListeners>"
    FAIL_REGULAR_EXPRESSION "Registered listeners:"
)

add_test(NAME NoAssertions COMMAND $<TARGET_FILE:SelfTest> -w NoAssertions "An empty test with no assertions")
set_tests_properties(NoAssertions PROPERTIES PASS_REGULAR_EXPRESSION "No assertions in test case")

# We cannot combine a regular expression on output with return code check
# in one test, so we register two instead of making a checking script because
# the runtime overhead is small enough.
add_test(NAME TestSpecs::CombiningMatchingAndNonMatchingIsOk-1 COMMAND $<TARGET_FILE:SelfTest> Tracker, "___nonexistent_test___")

add_test(NAME TestSpecs::CombiningMatchingAndNonMatchingIsOk-2 COMMAND $<TARGET_FILE:SelfTest> Tracker, "___nonexistent_test___")
set_tests_properties(TestSpecs::CombiningMatchingAndNonMatchingIsOk-2 PROPERTIES
  PASS_REGULAR_EXPRESSION "No test cases matched '\"___nonexistent_test___\"'"
  FAIL_REGULAR_EXPRESSION "No tests ran"
)

add_test(NAME TestSpecs::NoMatchedTestsFail
  COMMAND $<TARGET_FILE:SelfTest> "___nonexistent_test___"
)
set_tests_properties(TestSpecs::NoMatchedTestsFail
  PROPERTIES
    WILL_FAIL ON
)
add_test(NAME TestSpecs::OverrideFailureWithNoMatchedTests
  COMMAND $<TARGET_FILE:SelfTest> "___nonexistent_test___" --allow-running-no-tests
)

add_test(NAME TestSpecs::OverrideAllSkipFailure
  COMMAND $<TARGET_FILE:SelfTest> "tests can be skipped dynamically at runtime" --allow-running-no-tests
)

add_test(NAME TestSpecs::NonMatchingTestSpecIsRoundTrippable
  COMMAND $<TARGET_FILE:SelfTest> Tracker, "this test does not exist" "[nor does this tag]"
)
set_tests_properties(TestSpecs::NonMatchingTestSpecIsRoundTrippable
  PROPERTIES
    PASS_REGULAR_EXPRESSION "No test cases matched '\"this test does not exist\" \\[nor does this tag\\]'"
)

add_test(NAME Warnings::UnmatchedTestSpecIsAccepted
  COMMAND $<TARGET_FILE:SelfTest> Tracker --warn UnmatchedTestSpec
)
set_tests_properties(Warnings::UnmatchedTestSpecIsAccepted
  PROPERTIES
    FAIL_REGULAR_EXPRESSION "Unrecognised warning option: "
)

add_test(NAME Warnings::MultipleWarningsCanBeSpecified
  COMMAND
    $<TARGET_FILE:SelfTest> Tracker
      --warn NoAssertions
      --warn UnmatchedTestSpec
)

add_test(NAME TestSpecs::WarnUnmatchedTestSpecFailsWithUnmatchedTestSpec
  COMMAND
    $<TARGET_FILE:SelfTest> Tracker, "___nonexistent_test___" --warn UnmatchedTestSpec
)
set_tests_properties(TestSpecs::WarnUnmatchedTestSpecFailsWithUnmatchedTestSpec
  PROPERTIES
    WILL_FAIL ON
)

add_test(NAME UnmatchedOutputFilter COMMAND $<TARGET_FILE:SelfTest> [this-tag-does-not-exist])
set_tests_properties(UnmatchedOutputFilter
  PROPERTIES
    PASS_REGULAR_EXPRESSION "No test cases matched '\\[this-tag-does-not-exist\\]'"
)

add_test(NAME FilteredSection-1 COMMAND $<TARGET_FILE:SelfTest> \#1394 -c RunSection)
set_tests_properties(FilteredSection-1 PROPERTIES FAIL_REGULAR_EXPRESSION "No tests ran")
add_test(NAME FilteredSection-2 COMMAND $<TARGET_FILE:SelfTest> \#1394\ nested -c NestedRunSection -c s1)
set_tests_properties(FilteredSection-2 PROPERTIES FAIL_REGULAR_EXPRESSION "No tests ran")

add_test(
  NAME
    FilteredSection::GeneratorsDontCauseInfiniteLoop-1
  COMMAND
    $<TARGET_FILE:SelfTest> "#2025: original repro" -c "fov_0"
)
set_tests_properties(FilteredSection::GeneratorsDontCauseInfiniteLoop-1
  PROPERTIES
    PASS_REGULAR_EXPRESSION "inside with fov: 0" # This should happen
    FAIL_REGULAR_EXPRESSION "inside with fov: 1" # This would mean there was no filtering
)

# GENERATE between filtered sections (both are selected)
add_test(
  NAME
    FilteredSection::GeneratorsDontCauseInfiniteLoop-2
  COMMAND
    $<TARGET_FILE:SelfTest> "#2025: same-level sections"
    -c "A"
    -c "B"
    --colour-mode none
)
set_tests_properties(FilteredSection::GeneratorsDontCauseInfiniteLoop-2
  PROPERTIES
    PASS_REGULAR_EXPRESSION "All tests passed \\(4 assertions in 1 test case\\)"
)

add_test(NAME ApprovalTests
  COMMAND
    Python3::Interpreter
    ${CATCH_DIR}/tools/scripts/approvalTests.py
    $<TARGET_FILE:SelfTest>
    "${CMAKE_CURRENT_BINARY_DIR}"
)

set_tests_properties(ApprovalTests
  PROPERTIES
    FAIL_REGULAR_EXPRESSION "Results differed"

    # This is the most expensive test in the basic test suite, so we give
    # it high cost estimate so that CI runs it as one of the first ones,
    # for better parallelization.
    COST 30
    LABELS "uses-python"
)

add_test(NAME RegressionCheck-1670 COMMAND $<TARGET_FILE:SelfTest> "#1670 regression check" -c A -r compact)
set_tests_properties(RegressionCheck-1670 PROPERTIES PASS_REGULAR_EXPRESSION "All tests passed \\(2 assertions in 1 test case\\)")

add_test(NAME VersionCheck COMMAND $<TARGET_FILE:SelfTest> -h)
set_tests_properties(VersionCheck PROPERTIES PASS_REGULAR_EXPRESSION "Catch2 v${PROJECT_VERSION}")

add_test(NAME LibIdentityTest COMMAND $<TARGET_FILE:SelfTest> --libidentify)
set_tests_properties(LibIdentityTest PROPERTIES PASS_REGULAR_EXPRESSION "description:    A Catch2 test executable")

add_test(NAME FilenameAsTagsTest COMMAND $<TARGET_FILE:SelfTest> -\# --list-tags)
set_tests_properties(FilenameAsTagsTest PROPERTIES PASS_REGULAR_EXPRESSION "\\[#Approx.tests\\]")

# Check that the filename tags can also be matched against (#2064)
add_test(NAME FilenameAsTagsMatching COMMAND $<TARGET_FILE:SelfTest> -\# --list-tags [\#Approx.tests])
set_tests_properties(FilenameAsTagsMatching
  PROPERTIES
    PASS_REGULAR_EXPRESSION "\\[#Approx.tests\\]"
    # Avoids false positives by looking for start of line (newline) before the 0
    FAIL_REGULAR_EXPRESSION "[\r\n]0 tag"
)

add_test(NAME EscapeSpecialCharactersInTestNames COMMAND $<TARGET_FILE:SelfTest> "Test with special\\, characters \"in name")
set_tests_properties(EscapeSpecialCharactersInTestNames PROPERTIES PASS_REGULAR_EXPRESSION "1 assertion in 1 test case")

add_test(NAME NegativeSpecNoHiddenTests COMMAND $<TARGET_FILE:SelfTest> --list-tests ~[approval])
set_tests_properties(NegativeSpecNoHiddenTests PROPERTIES FAIL_REGULAR_EXPRESSION "\\[\\.\\]")

add_test(NAME TestsInFile::SimpleSpecs COMMAND $<TARGET_FILE:SelfTest> "-f ${SELF_TEST_DIR}/Misc/plain-old-tests.input")
set_tests_properties(TestsInFile::SimpleSpecs PROPERTIES PASS_REGULAR_EXPRESSION "6 assertions in 2 test cases")

add_test(NAME TestsInFile::EscapeSpecialCharacters COMMAND $<TARGET_FILE:SelfTest> "-f ${SELF_TEST_DIR}/Misc/special-characters-in-file.input")
set_tests_properties(TestsInFile::EscapeSpecialCharacters PROPERTIES PASS_REGULAR_EXPRESSION "1 assertion in 1 test case")

add_test(NAME TestsInFile::InvalidTestNames-1 COMMAND $<TARGET_FILE:SelfTest> "-f ${SELF_TEST_DIR}/Misc/invalid-test-names.input")
set_tests_properties(TestsInFile::InvalidTestNames-1
  PROPERTIES
    PASS_REGULAR_EXPRESSION "Invalid Filter: \"Test with special, characters in \\\\\" name\""
    FAIL_REGULAR_EXPRESSION "No tests ran"
)

add_test(NAME TagAlias COMMAND $<TARGET_FILE:SelfTest> [@tricky] --list-tests)
set_tests_properties(TagAlias PROPERTIES
  PASS_REGULAR_EXPRESSION "[0-9]+ matching test cases"
  FAIL_REGULAR_EXPRESSION "0 matching test cases"
)

add_test(NAME RandomTestOrdering COMMAND Python3::Interpreter
  ${CATCH_DIR}/tests/TestScripts/testRandomOrder.py $<TARGET_FILE:SelfTest>)
set_tests_properties(RandomTestOrdering
  PROPERTIES
    LABELS "uses-python"
)

add_test(NAME CheckConvenienceHeaders
  COMMAND
    Python3::Interpreter ${CATCH_DIR}/tools/scripts/checkConvenienceHeaders.py
)
set_tests_properties(CheckConvenienceHeaders
  PROPERTIES
    LABELS "uses-python"
)

add_test(NAME "Benchmarking::SkipBenchmarkMacros"
  COMMAND
    $<TARGET_FILE:SelfTest> "Skip benchmark macros"
      --reporter console
      --skip-benchmarks
      --colour-mode none
)
set_tests_properties("Benchmarking::SkipBenchmarkMacros"
  PROPERTIES
    PASS_REGULAR_EXPRESSION "All tests passed \\(2 assertions in 1 test case\\)"
    FAIL_REGULAR_EXPRESSION "benchmark name"
)

add_test(NAME "Benchmarking::FailureReporting::OptimizedOut"
  COMMAND
    $<TARGET_FILE:SelfTest> "Failing benchmarks" -c "empty" -r xml
  # This test only makes sense with the optimizer being enabled when
  # the tests are being compiled.
  CONFIGURATIONS Release
)
set_tests_properties("Benchmarking::FailureReporting::OptimizedOut"
  PROPERTIES
    PASS_REGULAR_EXPRESSION "could not measure benchmark\, maybe it was optimized away"
    FAIL_REGULAR_EXPRESSION "successes=\"1\""
)

add_test(NAME "Benchmarking::FailureReporting::ThrowingBenchmark"
  COMMAND
    $<TARGET_FILE:SelfTest> "Failing benchmarks" -c "throw" -r xml
)
set_tests_properties("Benchmarking::FailureReporting::ThrowingBenchmark"
  PROPERTIES
    PASS_REGULAR_EXPRESSION "<failed message=\"just a plain literal"
    FAIL_REGULAR_EXPRESSION "successes=\"1\""
)

add_test(NAME "Benchmarking::FailureReporting::FailedAssertion"
  COMMAND
    $<TARGET_FILE:SelfTest> "Failing benchmarks" -c "assert" -r xml
)
set_tests_properties("Benchmarking::FailureReporting::FailedAssertion"
  PROPERTIES
    PASS_REGULAR_EXPRESSION "<Expression success=\"false\""
    FAIL_REGULAR_EXPRESSION "successes=\"1\""
)

add_test(NAME "Benchmarking::FailureReporting::FailMacro"
  COMMAND
    $<TARGET_FILE:SelfTest> "Failing benchmarks" -c "fail" -r xml
)
set_tests_properties("Benchmarking::FailureReporting::FailMacro"
  PROPERTIES
    PASS_REGULAR_EXPRESSION "This benchmark only fails\, nothing else"
    FAIL_REGULAR_EXPRESSION "successes=\"1\""
)

add_test(NAME "Benchmarking::FailureReporting::ShouldFailIsRespected"
  COMMAND
    $<TARGET_FILE:SelfTest> "Failing benchmark respects should-fail"
)
set_tests_properties("Benchmarking::FailureReporting::ShouldFailIsRespected"
  PROPERTIES
    PASS_REGULAR_EXPRESSION "1 failed as expected"
)

add_test(NAME "ErrorHandling::InvalidTestSpecExitsEarly"
  COMMAND
    $<TARGET_FILE:SelfTest> "[aa,a]"
)
set_tests_properties("ErrorHandling::InvalidTestSpecExitsEarly"
  PROPERTIES
    PASS_REGULAR_EXPRESSION "Invalid Filter: \\[aa\,a\\]"
    FAIL_REGULAR_EXPRESSION "No tests ran"
)

if(MSVC)
  set(_NullFile "NUL")
else()
  set(_NullFile "/dev/null")
endif()

# This test checks that there is nothing written out from the process,
# but if CMake is running the tests under Valgrind or similar tool, then
# that will write its own output to stdout and the test would fail.
if(NOT MEMORYCHECK_COMMAND)
  add_test(NAME "MultiReporter::CapturingReportersDontPropagateStdOut"
    COMMAND
      $<TARGET_FILE:SelfTest> "Sends stuff to stdout and stderr"
        --reporter xml::out=${_NullFile}
        --reporter junit::out=${_NullFile}
  )
  set_tests_properties("MultiReporter::CapturingReportersDontPropagateStdOut"
    PROPERTIES
      FAIL_REGULAR_EXPRESSION ".+"
  )
endif()

add_test(NAME "MultiReporter::NonCapturingReportersPropagateStdout"
  COMMAND
    $<TARGET_FILE:SelfTest> "Sends stuff to stdout and stderr"
      --reporter xml::out=${_NullFile}
      --reporter console::out=${_NullFile}
)
set_tests_properties("MultiReporter::NonCapturingReportersPropagateStdout"
  PROPERTIES
    PASS_REGULAR_EXPRESSION "A string sent to stderr via clog"
)

add_test(NAME "Outputs::DashAsOutLocationSendsOutputToStdout"
  COMMAND
    $<TARGET_FILE:SelfTest> "Factorials are computed"
      --out=-
      --colour-mode none
)
set_tests_properties("Outputs::DashAsOutLocationSendsOutputToStdout"
  PROPERTIES
    PASS_REGULAR_EXPRESSION "All tests passed \\(5 assertions in 1 test case\\)"
)

add_test(NAME "Reporters::DashAsLocationInReporterSpecSendsOutputToStdout"
  COMMAND
    $<TARGET_FILE:SelfTest> "Factorials are computed"
      --reporter console::out=-
      --colour-mode none
)
set_tests_properties("Reporters::DashAsLocationInReporterSpecSendsOutputToStdout"
  PROPERTIES
    PASS_REGULAR_EXPRESSION "All tests passed \\(5 assertions in 1 test case\\)"
)

add_test(NAME "Reporters::ReporterSpecificColourOverridesDefaultColour"
  COMMAND
    $<TARGET_FILE:SelfTest> "Factorials are computed"
      --reporter console::colour-mode=ansi
      --colour-mode none
)
set_tests_properties("Reporters::ReporterSpecificColourOverridesDefaultColour"
  PROPERTIES
    PASS_REGULAR_EXPRESSION "\\[1\;32mAll tests passed"
)

add_test(NAME "Reporters::UnrecognizedOptionInSpecCausesError"
  COMMAND
    $<TARGET_FILE:SelfTest> "Factorials are computed"
      --reporter console::bad-option=ansi
)
set_tests_properties("Reporters::UnrecognizedOptionInSpecCausesError"
  PROPERTIES
    WILL_FAIL ON
)

add_test(NAME "Colours::ColourModeCanBeExplicitlySetToAnsi"
  COMMAND
    $<TARGET_FILE:SelfTest> "Factorials are computed"
      --reporter console
      --colour-mode ansi
)
set_tests_properties("Colours::ColourModeCanBeExplicitlySetToAnsi"
  PROPERTIES
    PASS_REGULAR_EXPRESSION "\\[1\;32mAll tests passed"
)

add_test(NAME "Reporters::JUnit::NamespacesAreNormalized"
  COMMAND
    $<TARGET_FILE:SelfTest>
      --reporter junit
      "A TEST_CASE_METHOD testing junit classname normalization"
)
set_tests_properties("Reporters::JUnit::NamespacesAreNormalized"
  PROPERTIES
    PASS_REGULAR_EXPRESSION "testcase classname=\"SelfTest(\.exe)?\\.A\\.B\\.TestClass\""
)

if(CATCH_ENABLE_CONFIGURE_TESTS)
  foreach(testName "DefaultReporter" "Disable" "DisableStringification"
                    "ExperimentalRedirect")
    add_test(NAME "CMakeConfig::${testName}"
      COMMAND
        Python3::Interpreter "${CMAKE_CURRENT_LIST_DIR}/TestScripts/testConfigure${testName}.py" "${CATCH_DIR}" "${CMAKE_CURRENT_BINARY_DIR}"
    )
    set_tests_properties("CMakeConfig::${testName}"
      PROPERTIES
        COST 240
        LABELS "uses-python"
    )
  endforeach()
endif()

if(CATCH_ENABLE_CMAKE_HELPER_TESTS)
  add_test(NAME "CMakeHelper::DiscoverTests"
    COMMAND
      Python3::Interpreter "${CMAKE_CURRENT_LIST_DIR}/TestScripts/DiscoverTests/VerifyRegistration.py" "${CATCH_DIR}" "${CMAKE_CURRENT_BINARY_DIR}"
  )
  set_tests_properties("CMakeHelper::DiscoverTests"
    PROPERTIES
      COST 240
      LABELS "uses-python"
  )
endif()

foreach(reporterName # "Automake" - the simple .trs format does not support any kind of comments/metadata
                       "compact"
                       "console"
                       "JUnit"
                       "SonarQube"
                       "TAP"
                     # "TeamCity" - does not seem to support test suite-level metadata/comments
                       "XML"
                       "JSON")
  add_test(NAME "Reporters:Filters:${reporterName}"
    COMMAND
      $<TARGET_FILE:SelfTest> [comparisons][string-case] "CaseInsensitiveLess is case insensitive"
        --reporter ${reporterName}
  )
  # Different regex for these two reporters, because the commas end up xml-escaped
  if(reporterName MATCHES "JUnit|XML")
    set(testCaseNameFormat "&quot;CaseInsensitiveLess is case insensitive&quot;")
  elseif(reporterName MATCHES "JSON")
    set(testCaseNameFormat "\\\\\"CaseInsensitiveLess is case insensitive\\\\\"")
  else()
    set(testCaseNameFormat "\"CaseInsensitiveLess is case insensitive\"")
  endif()
  set_tests_properties("Reporters:Filters:${reporterName}"
    PROPERTIES
      PASS_REGULAR_EXPRESSION "[fF]ilters.+\\[comparisons\\] \\[string-case\\] ${testCaseNameFormat}"
  )

  add_test(NAME "Reporters:RngSeed:${reporterName}"
    COMMAND
      $<TARGET_FILE:SelfTest> "Factorials are computed"
        --reporter ${reporterName}
        --rng-seed 18181818
  )
  set_tests_properties("Reporters:RngSeed:${reporterName}"
    PROPERTIES
      PASS_REGULAR_EXPRESSION "18181818"
  )
endforeach()

list(APPEND CATCH_TEST_TARGETS SelfTest)
set(CATCH_TEST_TARGETS ${CATCH_TEST_TARGETS} PARENT_SCOPE)
