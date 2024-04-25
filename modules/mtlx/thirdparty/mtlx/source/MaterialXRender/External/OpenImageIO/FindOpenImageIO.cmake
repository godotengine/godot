###########################################################################
# OpenImageIO   https://www.openimageio.org
# Copyright 2008-2018 Larry Gritz et al. All rights reserved.
# BSD 3-clause license:
#   https://github.com/OpenImageIO/oiio/blob/master/LICENSE
# For an up-to-date version of this file, see:
#   https://github.com/OpenImageIO/oiio/blob/master/src/cmake/Modules/FindOpenImageIO.cmake
#
###########################################################################
#
# CMake module to find OpenImageIO
#
# This module will set
#   OPENIMAGEIO_FOUND          True, if found
#   OPENIMAGEIO_INCLUDE_DIR    directory where headers are found
#   OPENIMAGEIO_LIBRARIES      libraries for OIIO
#   OPENIMAGEIO_LIBRARY_DIRS   library dirs for OIIO
#   OPENIMAGEIO_VERSION        Version ("major.minor.patch")
#   OPENIMAGEIO_VERSION_MAJOR  Version major number
#   OPENIMAGEIO_VERSION_MINOR  Version minor number
#   OPENIMAGEIO_VERSION_PATCH  Version minor patch
#   OIIOTOOL_BIN               Path to oiiotool executable
#
# Special inputs:
#   OPENIMAGEIO_ROOT_DIR - custom "prefix" location of OIIO installation
#                          (expecting bin, lib, include subdirectories)
#   OpenImageIO_FIND_QUIETLY - if set, print minimal console output
#   OIIO_LIBNAME_SUFFIX - if set, optional nonstandard library suffix
#
###########################################################################


# If 'OPENIMAGE_HOME' not set, use the env variable of that name if available
if (NOT OPENIMAGEIO_ROOT_DIR AND NOT $ENV{OPENIMAGEIO_ROOT_DIR} STREQUAL "")
    set (OPENIMAGEIO_ROOT_DIR $ENV{OPENIMAGEIO_ROOT_DIR})
endif ()


if (NOT OpenImageIO_FIND_QUIETLY)
    message ( STATUS "OPENIMAGEIO_ROOT_DIR = ${OPENIMAGEIO_ROOT_DIR}" )
endif ()

find_library ( OPENIMAGEIO_LIBRARY
               NAMES OpenImageIO${OIIO_LIBNAME_SUFFIX}
               HINTS ${OPENIMAGEIO_ROOT_DIR}/lib
               PATH_SUFFIXES lib64 lib
               PATHS "${OPENIMAGEIO_ROOT_DIR}/lib" )

find_library ( OPENIMAGEIO_UTIL_LIBRARY
               NAMES OpenImageIO_Util${OIIO_LIBNAME_SUFFIX}
               HINTS ${OPENIMAGEIO_ROOT_DIR}/lib
               PATH_SUFFIXES lib64 lib
               PATHS "${OPENIMAGEIO_ROOT_DIR}/lib" )
               
find_path ( OPENIMAGEIO_INCLUDE_DIR
            NAMES OpenImageIO/imageio.h
            HINTS ${OPENIMAGEIO_ROOT_DIR}/include
            PATH_SUFFIXES include )
find_program ( OIIOTOOL_BIN
               NAMES oiiotool oiiotool.exe
               HINTS ${OPENIMAGEIO_ROOT_DIR}/bin
               PATH_SUFFIXES bin )

# Try to figure out version number
set (OIIO_VERSION_HEADER "${OPENIMAGEIO_INCLUDE_DIR}/OpenImageIO/oiioversion.h")
if (EXISTS "${OIIO_VERSION_HEADER}")
    file (STRINGS "${OIIO_VERSION_HEADER}" TMP REGEX "^#define OIIO_VERSION_MAJOR .*$")
    string (REGEX MATCHALL "[0-9]+" OPENIMAGEIO_VERSION_MAJOR ${TMP})
    file (STRINGS "${OIIO_VERSION_HEADER}" TMP REGEX "^#define OIIO_VERSION_MINOR .*$")
    string (REGEX MATCHALL "[0-9]+" OPENIMAGEIO_VERSION_MINOR ${TMP})
    file (STRINGS "${OIIO_VERSION_HEADER}" TMP REGEX "^#define OIIO_VERSION_PATCH .*$")
    string (REGEX MATCHALL "[0-9]+" OPENIMAGEIO_VERSION_PATCH ${TMP})
    set (OPENIMAGEIO_VERSION "${OPENIMAGEIO_VERSION_MAJOR}.${OPENIMAGEIO_VERSION_MINOR}.${OPENIMAGEIO_VERSION_PATCH}")
endif ()

set ( OPENIMAGEIO_LIBRARIES ${OPENIMAGEIO_LIBRARY} ${OPENIMAGEIO_UTIL_LIBRARY})
get_filename_component (OPENIMAGEIO_LIBRARY_DIRS "${OPENIMAGEIO_LIBRARY}" DIRECTORY CACHE)

if (NOT OpenImageIO_FIND_QUIETLY)
    message ( STATUS "OpenImageIO includes     = ${OPENIMAGEIO_INCLUDE_DIR}" )
    message ( STATUS "OpenImageIO libraries    = ${OPENIMAGEIO_LIBRARIES}" )
    message ( STATUS "OpenImageIO library_dirs = ${OPENIMAGEIO_LIBRARY_DIRS}" )
    message ( STATUS "OpenImageIO oiiiotool    = ${OIIOTOOL_BIN}" )
endif ()

include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (OpenImageIO
    FOUND_VAR     OPENIMAGEIO_FOUND
    REQUIRED_VARS OPENIMAGEIO_INCLUDE_DIR OPENIMAGEIO_LIBRARIES
                  OPENIMAGEIO_LIBRARY_DIRS OPENIMAGEIO_VERSION
    VERSION_VAR   OPENIMAGEIO_VERSION
    )

mark_as_advanced (
    OPENIMAGEIO_INCLUDE_DIR
    OPENIMAGEIO_LIBRARIES
    OPENIMAGEIO_LIBRARY_DIRS
    OPENIMAGEIO_VERSION
    OPENIMAGEIO_VERSION_MAJOR
    OPENIMAGEIO_VERSION_MINOR
    OPENIMAGEIO_VERSION_PATCH
    )
