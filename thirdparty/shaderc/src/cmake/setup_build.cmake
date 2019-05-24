# Find nosetests; see shaderc_add_nosetests() from utils.cmake for opting in to
# nosetests in a specific directory.

if(NOT COMMAND find_host_package)
  macro(find_host_package)
    find_package(${ARGN})
  endmacro()
endif()
if(NOT COMMAND find_host_program)
  macro(find_host_program)
    find_program(${ARGN})
  endmacro()
endif()

find_program(NOSETESTS_EXE NAMES nosetests PATHS $ENV{PYTHON_PACKAGE_PATH})
if (NOT NOSETESTS_EXE)
  message(STATUS "nosetests was not found - python code will not be tested")
endif()

# Find asciidoctor; see shaderc_add_asciidoc() from utils.cmake for
# adding documents.
find_program(ASCIIDOCTOR_EXE NAMES asciidoctor)
if (NOT ASCIIDOCTOR_EXE)
  message(STATUS "asciidoctor was not found - no documentation will be"
          " generated")
endif()

# On Windows, CMake by default compiles with the shared CRT.
# Ensure that gmock compiles the same, otherwise failures will occur.
if(WIN32)
  # TODO(awoloszyn): Once we support selecting CRT versions,
  # make sure this matches correctly.
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
endif(WIN32)

if (ANDROID)
# For android let's preemptively find the correct packages so that
# child projects (glslang, googletest) do not fail to find them.
find_host_package(PythonInterp)
find_host_package(BISON)
endif()

foreach(PROGRAM echo python)
  string(TOUPPER ${PROGRAM} PROG_UC)
  if (ANDROID)
    find_host_program(${PROG_UC}_EXE ${PROGRAM} REQUIRED)
  else()
    find_program(${PROG_UC}_EXE ${PROGRAM} REQUIRED)
  endif()
endforeach(PROGRAM)

option(ENABLE_CODE_COVERAGE "Enable collecting code coverage." OFF)
if (ENABLE_CODE_COVERAGE)
  message(STATUS "Shaderc: code coverage report is on.")
  if (NOT UNIX)
    message(FATAL_ERROR "Code coverage on non-UNIX system not supported yet.")
  endif()
  if (NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    message(FATAL_ERROR "Code coverage with non-Debug build can be misleading.")
  endif()
  find_program(LCOV_EXE NAMES lcov)
  if (NOT LCOV_EXE)
    message(FATAL_ERROR "lcov was not found")
  endif()
  find_program(GENHTML_EXE NAMES genhtml)
  if (NOT GENHTML_EXE)
    message(FATAL_ERROR "genhtml was not found")
  endif()

  set(LCOV_BASE_DIR ${CMAKE_BINARY_DIR})
  set(LCOV_INFO_FILE ${LCOV_BASE_DIR}/lcov.info)
  set(COVERAGE_STAT_HTML_DIR ${LCOV_BASE_DIR}/coverage-report)

  add_custom_target(clean-coverage
    # Remove all gcov .gcda files in the directory recursively.
    COMMAND ${LCOV_EXE} --directory . --zerocounters -q
    # Remove all lcov .info files.
    COMMAND ${CMAKE_COMMAND} -E remove ${LCOV_INFO_FILE}
    # Remove all html report files.
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${COVERAGE_STAT_HTML_DIR}
    # TODO(antiagainst): the following two commands are here to remedy the
    # problem of "reached unexpected end of file" experienced by lcov.
    # The symptom is that some .gcno files are wrong after code change and
    # recompiling. We don't know the exact reason yet. Figure it out.
    # Remove all .gcno files in the directory recursively.
    COMMAND ${PYTHON_EXE}
    ${shaderc_SOURCE_DIR}/utils/remove-file-by-suffix.py . ".gcno"
    # .gcno files are not tracked by CMake. So no recompiling is triggered
    # even if they are missing. Unfortunately, we just removed all of them
    # in the above.
    COMMAND ${CMAKE_COMMAND} --build . --target clean
    WORKING_DIRECTORY ${LCOV_BASE_DIR}
    COMMENT "Clean coverage files"
  )

  add_custom_target(report-coverage
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
    # Run all tests.
    COMMAND ctest --output-on-failure
    # Collect coverage data from gcov .gcda files.
    COMMAND ${LCOV_EXE} --directory . --capture -o ${LCOV_INFO_FILE}
    # Remove coverage info for system header files.
    COMMAND ${LCOV_EXE}
      --remove ${LCOV_INFO_FILE} '/usr/include/*' -o ${LCOV_INFO_FILE}
    # Remove coverage info for external and third_party code.
    COMMAND ${LCOV_EXE}
      --remove ${LCOV_INFO_FILE} '${shaderc_SOURCE_DIR}/ext/*'
      -o ${LCOV_INFO_FILE}

    COMMAND ${LCOV_EXE}
      --remove ${LCOV_INFO_FILE} '${shaderc_SOURCE_DIR}/third_party/*'
      -o ${LCOV_INFO_FILE}
    # Remove coverage info for tests.
    COMMAND ${LCOV_EXE}
      --remove ${LCOV_INFO_FILE} '*_test.cc' -o ${LCOV_INFO_FILE}
    # Generate html report file.
    COMMAND ${GENHTML_EXE}
      ${LCOV_INFO_FILE} -t "Coverage Report" -o ${COVERAGE_STAT_HTML_DIR}
    DEPENDS clean-coverage
    WORKING_DIRECTORY ${LCOV_BASE_DIR}
    COMMENT "Collect and analyze coverage data"
  )
endif()

option(DISABLE_RTTI "Disable RTTI in builds")
if(DISABLE_RTTI)
  if(UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -fno-rtti")
  endif(UNIX)
endif(DISABLE_RTTI)

option(DISABLE_EXCEPTIONS "Disables exceptions in builds")
if(DISABLE_EXCEPTIONS)
  if(UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions")
  endif(UNIX)
endif(DISABLE_EXCEPTIONS)
