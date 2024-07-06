# Copyright (c) 2015-2024 The Khronos Group Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and/or associated documentation files (the
# "Materials"), to deal in the Materials without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Materials, and to
# permit persons to whom the Materials are furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Materials.
#
# MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS
# KHRONOS STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS
# SPECIFICATIONS AND HEADER INFORMATION ARE LOCATED AT
#    https://www.khronos.org/registry/
#
# THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
cmake_minimum_required(VERSION 3.14)
project(SPIRV-Headers LANGUAGES CXX VERSION 1.5.5)

if (CMAKE_VERSION VERSION_LESS "3.21")
    # https://cmake.org/cmake/help/latest/variable/PROJECT_IS_TOP_LEVEL.html
    string(COMPARE EQUAL ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR} PROJECT_IS_TOP_LEVEL)
endif()

add_library(SPIRV-Headers INTERFACE)
add_library(SPIRV-Headers::SPIRV-Headers ALIAS SPIRV-Headers)
target_include_directories(SPIRV-Headers INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

option(SPIRV_HEADERS_ENABLE_TESTS "Test SPIRV-Headers" ${PROJECT_IS_TOP_LEVEL})
option(SPIRV_HEADERS_ENABLE_INSTALL "Install SPIRV-Headers" ${PROJECT_IS_TOP_LEVEL})

if(SPIRV_HEADERS_ENABLE_TESTS)
    add_subdirectory(tests)
endif()

if(SPIRV_HEADERS_ENABLE_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/spirv DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

    set(cmake_install_dir "${CMAKE_INSTALL_DATADIR}/cmake/SPIRV-Headers")
    set(version_config "${CMAKE_CURRENT_BINARY_DIR}/generated/SPIRV-HeadersConfigVersion.cmake")

    write_basic_package_version_file("${version_config}" COMPATIBILITY SameMajorVersion ARCH_INDEPENDENT)
    install(FILES "${version_config}" DESTINATION "${cmake_install_dir}")

    install(TARGETS SPIRV-Headers EXPORT "SPIRV-HeadersConfig" INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(EXPORT "SPIRV-HeadersConfig" NAMESPACE "SPIRV-Headers::" DESTINATION "${cmake_install_dir}")

    if (IS_ABSOLUTE ${CMAKE_INSTALL_INCLUDEDIR})
        set(SPIRV_HEADERS_PKGCONFIG_INCLUDE_DIR ${CMAKE_INSTALL_INCLUDEDIR})
    else()
        set(SPIRV_HEADERS_PKGCONFIG_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR})
    endif()
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/SPIRV-Headers.pc.in ${CMAKE_CURRENT_BINARY_DIR}/SPIRV-Headers.pc @ONLY)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/SPIRV-Headers.pc" DESTINATION ${CMAKE_INSTALL_DATADIR}/pkgconfig)
endif()
