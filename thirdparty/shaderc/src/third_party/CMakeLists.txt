# Suppress all warnings from third-party projects.
set_property(DIRECTORY APPEND PROPERTY COMPILE_OPTIONS -w)

set(SHADERC_THIRD_PARTY_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" CACHE STRING
  "Root location of all third_party projects")
set(SHADERC_GOOGLE_TEST_DIR "${SHADERC_THIRD_PARTY_ROOT_DIR}/googletest" CACHE STRING
  "Location of googletest source")
set(SHADERC_SPIRV_TOOLS_DIR "${SHADERC_THIRD_PARTY_ROOT_DIR}/spirv-tools" CACHE STRING
  "Location of spirv-tools source")
set(SHADERC_SPIRV_HEADERS_DIR "${SHADERC_THIRD_PARTY_ROOT_DIR}/spirv-headers" CACHE STRING
  "Location of spirv-headers source")
set(SHADERC_SPIRV_CROSS_DIR "${SHADERC_THIRD_PARTY_ROOT_DIR}/spirv-cross" CACHE STRING
  "Location of SPIRV-Cross source")
set(SHADERC_GLSLANG_DIR "${SHADERC_THIRD_PARTY_ROOT_DIR}/glslang" CACHE STRING
  "Location of glslang source")
set(SHADERC_EFFCEE_DIR "${SHADERC_THIRD_PARTY_ROOT_DIR}/effcee" CACHE STRING
  "Location of effcee source")
set(SHADERC_RE2_DIR "${SHADERC_THIRD_PARTY_ROOT_DIR}/re2" CACHE STRING
  "Location of re2 source")

set( SKIP_GLSLANG_INSTALL ${SHADERC_SKIP_INSTALL} )
set( SKIP_SPIRV_TOOLS_INSTALL ${SHADERC_SKIP_INSTALL} )
set( SKIP_GOOGLETEST_INSTALL ${SHADERC_SKIP_INSTALL} )

# Configure third party projects.
if(${SHADERC_ENABLE_TESTS})
  if (IS_DIRECTORY ${SHADERC_GOOGLE_TEST_DIR})
    add_subdirectory(${SHADERC_GOOGLE_TEST_DIR} googletest)
  endif()
  if (NOT TARGET gmock)
    message(FATAL_ERROR "gmock was not found - required for tests")
  endif()
endif()

set(OLD_PLATFORM_TOOLSET ${CMAKE_GENERATOR_TOOLSET})


if (IS_DIRECTORY ${SHADERC_SPIRV_HEADERS_DIR})
  set(SPIRV_HEADERS_SKIP_EXAMPLES ON)
  add_subdirectory(${SHADERC_SPIRV_HEADERS_DIR} spirv-headers)
endif()

if (NOT TARGET SPIRV-Tools)
  # Check SPIRV-Tools before glslang so that it is linked into glslang.
  # we control optimizations via glslang API calls directly.
  if (IS_DIRECTORY ${SHADERC_SPIRV_TOOLS_DIR})
    if ("${SHADERC_SKIP_TESTS}")
      # Also skip building tests in SPIRV-Tools.
      set(SPIRV_SKIP_TESTS ON CACHE BOOL "Skip building SPIRV-Tools tests")
    elseif(NOT "${SPIRV_SKIP_TESTS}")
      # SPIRV-Tools requires effcee and re2 to build tests.
      add_subdirectory(${SHADERC_RE2_DIR} re2)
      add_subdirectory(${SHADERC_EFFCEE_DIR} effcee)
    endif()
    add_subdirectory(${SHADERC_SPIRV_TOOLS_DIR} spirv-tools)
  endif()
  if (NOT TARGET SPIRV-Tools)
    message(FATAL_ERROR "SPIRV-Tools was not found - required for compilation")
  endif()
endif()

if (NOT TARGET glslang)
  if (IS_DIRECTORY ${SHADERC_GLSLANG_DIR})
    add_subdirectory(${SHADERC_GLSLANG_DIR} glslang)
  endif()
  if (NOT TARGET glslang)
    message(FATAL_ERROR "glslang was not found - required for compilation")
  endif()
  if(WIN32)
      # This is unfortunate but glslang forces our
      # platform toolset to be v110, which we may not even have
      # installed, undo anything glslang has done to it.
      set(CMAKE_GENERATOR_TOOLSET "${OLD_PLATFORM_TOOLSET}" CACHE STRING
        "Platform Toolset" FORCE)
  endif()
endif()

if (SHADERC_ENABLE_SPVC)
  if (NOT TARGET spirv-cross-core)
    if (IS_DIRECTORY ${SHADERC_SPIRV_CROSS_DIR})
      set(SPIRV_CROSS_EXCEPTIONS_TO_ASSERTIONS ${DISABLE_EXCEPTIONS} CACHE BOOL "tell SPIRV-Cross not to throw" FORCE)

      # Add -fPIC to SPIRV-Cross build.
      set(CXX_BACK ${CMAKE_CXX_FLAGS})
      set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-fPIC")
      # cmake inserts a semicolon, change it to a space.
      string(REGEX REPLACE ";" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
      add_subdirectory(${SHADERC_SPIRV_CROSS_DIR} spirv-cross)
      set(CMAKE_CXX_FLAGS ${CXX_BACK})
    endif()
    if (NOT TARGET spirv-cross-core)
      message(FATAL_ERROR "SPIRV-Cross was not found - required for compilation")
    endif()
  endif()
endif (SHADERC_ENABLE_SPVC)

if(${SHADERC_ENABLE_TESTS})
  # Configure out-of-source-directory tests for glslang.
  # The glslang project uses a bash script called "runtests" to run tests.
  # The runtests script assumes the glslangValidator executable exists in
  # a location inside the source tree, but we build it elsewhere.
  # We need to copy the test files, fix the path references, and then run tests.
  # Use test directory named "Test" to match Glslangs Test directory, so
  # that we get the right relative path names in the "include" test output.
  set(GLSLANG_TEST_SRC_DIR ${SHADERC_GLSLANG_DIR}/Test)
  set(GLSLANG_TEST_BIN_DIR
        ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/Test)

  # If we are building in a multi-configuration setting we have
  # to put the glslang tests into their respective subdirectories.
  if (CMAKE_CONFIGURATION_TYPES)
    set(GLSLANG_CONFIGURATION_DIR ${CMAKE_CFG_INTDIR})
  endif()

  add_custom_target(copy-tests-if-necessary ALL
    COMMAND ${PYTHON_EXE}
      ${shaderc_SOURCE_DIR}/utils/copy-tests-if-necessary.py
      ${GLSLANG_TEST_SRC_DIR} ${GLSLANG_TEST_BIN_DIR} ${GLSLANG_CONFIGURATION_DIR}
    COMMENT "Copying and patching glslang tests if needed")

  # glslang-testsuite runs a bash script on Windows.
  # Make sure to use '-o igncr' flag to ignore carriage returns (\r).
  set(IGNORE_CR_FLAG "")
  if(WIN32)
    set(IGNORE_CR_FLAG -o igncr)
  endif()

  if (CMAKE_CONFIGURATION_TYPES)
    # If we are running a multi-configuration project,
    # the tests will be in ${Configuration}/Test
    add_test(NAME glslang-testsuite
      COMMAND bash ${IGNORE_CR_FLAG} runtests
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIGURATION>/Test
    )
  else()
    add_test(NAME glslang-testsuite
      COMMAND bash ${IGNORE_CR_FLAG} runtests
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/Test/
    )
  endif()
endif()
