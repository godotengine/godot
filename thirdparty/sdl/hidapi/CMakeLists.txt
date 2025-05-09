cmake_minimum_required(VERSION 3.1.3...3.25 FATAL_ERROR)

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    add_subdirectory(src)
    # compatibility with find_package() vs add_subdirectory
    set(hidapi_VERSION "${hidapi_VERSION}" PARENT_SCOPE)
    return()
endif()
# All of the below in this file is meant for a standalone build.
# When building as a subdirectory of a larger project, most of the options may not make sense for it,
# so it is up to developer to configure those, e.g.:
#
# # a subfolder of a master project, e.g.: 3rdparty/hidapi/CMakeLists.txt
#
# set(HIDAPI_WITH_HIDRAW OFF)
# set(CMAKE_FRAMEWORK ON)
# # and keep everything else to their defaults
# add_subdirectory(hidapi)
#

set(DEFAULT_CMAKE_BUILD_TYPES "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
if(NOT DEFINED CMAKE_BUILD_TYPE OR NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "${DEFAULT_CMAKE_BUILD_TYPES}" FORCE)
endif()
# This part is for convenience, when used one of the standard build types with cmake-gui
list(FIND DEFAULT_CMAKE_BUILD_TYPES "${CMAKE_BUILD_TYPE}" _build_type_index)
if(${_build_type_index} GREATER -1)
    # set it optionally, so a custom CMAKE_BUILD_TYPE can be used as well, if needed
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${DEFAULT_CMAKE_BUILD_TYPES})
endif()
unset(_build_type_index)
#

project(hidapi LANGUAGES C)

if(APPLE)
    if(NOT CMAKE_VERSION VERSION_LESS "3.15")
        option(CMAKE_FRAMEWORK "Build macOS/iOS Framework version of the library" OFF)
    endif()
elseif(NOT WIN32)
    if(CMAKE_SYSTEM_NAME MATCHES "Linux")
        option(HIDAPI_WITH_HIDRAW "Build HIDRAW-based implementation of HIDAPI" ON)
        option(HIDAPI_WITH_LIBUSB "Build LIBUSB-based implementation of HIDAPI" ON)
    endif()
    if(CMAKE_SYSTEM_NAME MATCHES "NetBSD")
        option(HIDAPI_WITH_NETBSD "Build NetBSD/UHID implementation of HIDAPI" ON)
    endif()
endif()

option(BUILD_SHARED_LIBS "Build shared version of the libraries, otherwise build statically" ON)

set(HIDAPI_INSTALL_TARGETS ON)
set(HIDAPI_PRINT_VERSION ON)

set(IS_DEBUG_BUILD OFF)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(IS_DEBUG_BUILD ON)
endif()

option(HIDAPI_ENABLE_ASAN "Build HIDAPI with ASAN address sanitizer instrumentation" OFF)

if(HIDAPI_ENABLE_ASAN)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address")
    if(MSVC)
        # the default is to have "/INCREMENTAL" which causes a warning when "-fsanitize=address" is present
        set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} /INCREMENTAL:NO")
        set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} /INCREMENTAL:NO")
        set(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO} /INCREMENTAL:NO")
        set(CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO} /INCREMENTAL:NO")
    endif()
endif()

if(WIN32)
    # so far only Windows has tests
    option(HIDAPI_WITH_TESTS "Build HIDAPI (unit-)tests" ${IS_DEBUG_BUILD})
else()
    set(HIDAPI_WITH_TESTS OFF)
endif()

if(HIDAPI_WITH_TESTS)
    enable_testing()
endif()

if(WIN32)
    option(HIDAPI_BUILD_PP_DATA_DUMP "Build small Windows console application pp_data_dump.exe" ${IS_DEBUG_BUILD})
endif()

add_subdirectory(src)

option(HIDAPI_BUILD_HIDTEST "Build small console test application hidtest" ${IS_DEBUG_BUILD})
if(HIDAPI_BUILD_HIDTEST)
    add_subdirectory(hidtest)
endif()

if(HIDAPI_ENABLE_ASAN)
    if(NOT MSVC)
        # MSVC doesn't recognize those options, other compilers - requiring it
        foreach(HIDAPI_TARGET hidapi_winapi hidapi_darwin hidapi_hidraw hidapi_libusb hidtest_hidraw hidtest_libusb hidtest)
            if(TARGET ${HIDAPI_TARGET})
                if(BUILD_SHARED_LIBS)
                    target_link_options(${HIDAPI_TARGET} PRIVATE -fsanitize=address)
                else()
                    target_link_options(${HIDAPI_TARGET} PUBLIC -fsanitize=address)
                endif()
            endif()
        endforeach()
    endif()
endif()
