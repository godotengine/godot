cmake_minimum_required(VERSION 2.8.12)
project(shaderc)
enable_testing()

if ("${CMAKE_BUILD_TYPE}" STREQUAL "")
  message(STATUS "No build type selected, default to Debug")
  set(CMAKE_BUILD_TYPE "Debug")
endif()

message(STATUS "Shaderc: build type is \"${CMAKE_BUILD_TYPE}\".")

option(SHADERC_ENABLE_SPVC "Enable libshaderc_spvc" OFF)

option(SHADERC_SKIP_INSTALL "Skip installation" ${SHADERC_SKIP_INSTALL})
if(NOT ${SHADERC_SKIP_INSTALL})
  set(SHADERC_ENABLE_INSTALL ON)
endif()

option(SHADERC_SKIP_TESTS "Skip building tests" ${SHADERC_SKIP_TESTS})
if(NOT ${SHADERC_SKIP_TESTS})
  set(SHADERC_ENABLE_TESTS ON)
endif()
if(${SHADERC_ENABLE_TESTS})
  message(STATUS "Configuring Shaderc to build tests.")
else()
  message(STATUS "Configuring Shaderc to avoid building tests.")
endif()

set (CMAKE_CXX_STANDARD 11)

option(SHADERC_ENABLE_NV_EXTENSIONS "Enables support of Nvidia-specific extensions" ON)

include(GNUInstallDirs)
include(cmake/setup_build.cmake)
include(cmake/utils.cmake)


if(SHADERC_ENABLE_NV_EXTENSIONS)
    add_definitions(-DNV_EXTENSIONS)
endif(SHADERC_ENABLE_NV_EXTENSIONS)

add_custom_target(check-copyright ALL
  ${PYTHON_EXE} ${CMAKE_CURRENT_SOURCE_DIR}/utils/add_copyright.py
  --check
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Check copyright")

add_custom_target(add-copyright
  ${PYTHON_EXE} ${CMAKE_CURRENT_SOURCE_DIR}/utils/add_copyright.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Add copyright")

if(MSVC)
  option(SHADERC_ENABLE_SHARED_CRT
          "Use the shared CRT instead of the static CRT"
          ${SHADERC_ENABLE_SHARED_CRT})
  if (NOT SHADERC_ENABLE_SHARED_CRT)
    # Link executables statically by replacing /MD with /MT everywhere.
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
  endif(NOT SHADERC_ENABLE_SHARED_CRT)
endif(MSVC)

# Configure subdirectories.
# We depend on these for later projects, so they should come first.
add_subdirectory(third_party)

if(SHADERC_ENABLE_SPVC)
add_subdirectory(libshaderc_spvc)
add_subdirectory(spvc)
endif()
add_subdirectory(libshaderc_util)
add_subdirectory(libshaderc)
add_subdirectory(glslc)
add_subdirectory(examples)

add_custom_target(build-version
  ${PYTHON_EXE}
  ${CMAKE_CURRENT_SOURCE_DIR}/utils/update_build_version.py
  ${shaderc_SOURCE_DIR} ${spirv-tools_SOURCE_DIR} ${glslang_SOURCE_DIR}
  COMMENT "Update build-version.inc in the Shaderc build directory (if necessary).")

function(define_pkg_config_file NAME LIBS)
  add_custom_target(${NAME}-pkg-config ALL
    COMMAND ${CMAKE_COMMAND}
      -DCHANGES_FILE=${CMAKE_CURRENT_SOURCE_DIR}/CHANGES
      -DTEMPLATE_FILE=${CMAKE_CURRENT_SOURCE_DIR}/cmake/${NAME}.pc.in
      -DOUT_FILE=${CMAKE_CURRENT_BINARY_DIR}/${NAME}.pc
      -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
      -DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}
      -DCMAKE_INSTALL_INCLUDEDIR=${CMAKE_INSTALL_INCLUDEDIR}
      -DLIBS=${LIBS}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/write_pkg_config.cmake
  DEPENDS "CHANGES" "cmake/${NAME}.pc.in" "cmake/write_pkg_config.cmake")

  if (SHADERC_ENABLE_INSTALL)
    install(
      FILES
        ${CMAKE_CURRENT_BINARY_DIR}/${NAME}.pc
      DESTINATION
        ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
  endif()
endfunction()

if(SHADERC_ENABLE_SPVC)
  set(EXTRA_STATIC_PKGCONFIG_LIBS -lshaderc_spvc)
endif()
define_pkg_config_file(shaderc -lshaderc_shared)
define_pkg_config_file(shaderc_static "-lshaderc ${EXTRA_STATIC_PKGCONFIG_LIBS} -lshaderc_util")
define_pkg_config_file(shaderc_combined -lshaderc_combined)
