/*
---------------------------------------------------------------------------
Open Asset Import Library (assimp)
---------------------------------------------------------------------------

Copyright (c) 2006-2019, assimp team



All rights reserved.

Redistribution and use of this software in source and binary forms,
with or without modification, are permitted provided that the following
conditions are met:

* Redistributions of source code must retain the above
  copyright notice, this list of conditions and the
  following disclaimer.

* Redistributions in binary form must reproduce the above
  copyright notice, this list of conditions and the
  following disclaimer in the documentation and/or other
  materials provided with the distribution.

* Neither the name of the assimp team, nor the names of its
  contributors may be used to endorse or promote products
  derived from this software without specific prior
  written permission of the assimp team.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
---------------------------------------------------------------------------
*/

/** @file defs.h
 *  @brief Assimp build configuration setup. See the notes in the comment
 *  blocks to find out how to customize _your_ Assimp build.
 */

#pragma once
#ifndef AI_DEFINES_H_INC
#define AI_DEFINES_H_INC

#include <assimp/config.h>

//////////////////////////////////////////////////////////////////////////
/* Define ASSIMP_BUILD_NO_XX_IMPORTER to disable a specific
 * file format loader. The loader is be excluded from the
 * build in this case. 'XX' stands for the most common file
 * extension of the file format. E.g.:
 * ASSIMP_BUILD_NO_X_IMPORTER disables the X loader.
 *
 * If you're unsure about that, take a look at the implementation of the
 * import plugin you wish to disable. You'll find the right define in the
 * first lines of the corresponding unit.
 *
 * Other (mixed) configuration switches are listed here:
 *    ASSIMP_BUILD_NO_COMPRESSED_X
 *      - Disable support for compressed X files (zip)
 *    ASSIMP_BUILD_NO_COMPRESSED_BLEND
 *      - Disable support for compressed Blender files (zip)
 *    ASSIMP_BUILD_NO_COMPRESSED_IFC
 *      - Disable support for IFCZIP files (unzip)
 */
//////////////////////////////////////////////////////////////////////////

#ifndef ASSIMP_BUILD_NO_COMPRESSED_X
#   define ASSIMP_BUILD_NEED_Z_INFLATE
#endif

#ifndef ASSIMP_BUILD_NO_COMPRESSED_BLEND
#   define ASSIMP_BUILD_NEED_Z_INFLATE
#endif

#ifndef ASSIMP_BUILD_NO_COMPRESSED_IFC
#   define ASSIMP_BUILD_NEED_Z_INFLATE
#   define ASSIMP_BUILD_NEED_UNZIP
#endif

#ifndef ASSIMP_BUILD_NO_Q3BSP_IMPORTER
#   define ASSIMP_BUILD_NEED_Z_INFLATE
#   define ASSIMP_BUILD_NEED_UNZIP
#endif

//////////////////////////////////////////////////////////////////////////
/* Define ASSIMP_BUILD_NO_XX_PROCESS to disable a specific
 * post processing step. This is the current list of process names ('XX'):
 * CALCTANGENTS
 * JOINVERTICES
 * TRIANGULATE
 * DROPFACENORMALS
 * GENFACENORMALS
 * GENVERTEXNORMALS
 * REMOVEVC
 * SPLITLARGEMESHES
 * PRETRANSFORMVERTICES
 * LIMITBONEWEIGHTS
 * VALIDATEDS
 * IMPROVECACHELOCALITY
 * FIXINFACINGNORMALS
 * REMOVE_REDUNDANTMATERIALS
 * OPTIMIZEGRAPH
 * SORTBYPTYPE
 * FINDINVALIDDATA
 * TRANSFORMTEXCOORDS
 * GENUVCOORDS
 * ENTITYMESHBUILDER
 * EMBEDTEXTURES
 * MAKELEFTHANDED
 * FLIPUVS
 * FLIPWINDINGORDER
 * OPTIMIZEMESHES
 * OPTIMIZEANIMS
 * OPTIMIZEGRAPH
 * GENENTITYMESHES
 * FIXTEXTUREPATHS */
//////////////////////////////////////////////////////////////////////////

#ifdef _MSC_VER
#   undef ASSIMP_API

    //////////////////////////////////////////////////////////////////////////
    /* Define 'ASSIMP_BUILD_DLL_EXPORT' to build a DLL of the library */
    //////////////////////////////////////////////////////////////////////////
#   ifdef ASSIMP_BUILD_DLL_EXPORT
#       define ASSIMP_API __declspec(dllexport)
#       define ASSIMP_API_WINONLY __declspec(dllexport)
#       pragma warning (disable : 4251)

    //////////////////////////////////////////////////////////////////////////
    /* Define 'ASSIMP_DLL' before including Assimp to link to ASSIMP in
     * an external DLL under Windows. Default is static linkage. */
    //////////////////////////////////////////////////////////////////////////
#   elif (defined ASSIMP_DLL)
#       define ASSIMP_API __declspec(dllimport)
#       define ASSIMP_API_WINONLY __declspec(dllimport)
#   else
#       define ASSIMP_API
#       define ASSIMP_API_WINONLY
#   endif

    /* Force the compiler to inline a function, if possible
     */
#   define AI_FORCE_INLINE __forceinline

    /* Tells the compiler that a function never returns. Used in code analysis
     * to skip dead paths (e.g. after an assertion evaluated to false). */
#   define AI_WONT_RETURN __declspec(noreturn)

#elif defined(SWIG)

    /* Do nothing, the relevant defines are all in AssimpSwigPort.i */

#else

#   define AI_WONT_RETURN

#   define ASSIMP_API __attribute__ ((visibility("default")))
#   define ASSIMP_API_WINONLY
#   define AI_FORCE_INLINE inline
#endif // (defined _MSC_VER)

#ifdef __GNUC__
#   define AI_WONT_RETURN_SUFFIX  __attribute__((noreturn))
#else
#   define AI_WONT_RETURN_SUFFIX
#endif // (defined __clang__)

#ifdef __cplusplus
    /* No explicit 'struct' and 'enum' tags for C++, this keeps showing up
     * in doxydocs.
     */
#   define C_STRUCT
#   define C_ENUM
#else
    //////////////////////////////////////////////////////////////////////////
    /* To build the documentation, make sure ASSIMP_DOXYGEN_BUILD
     * is defined by Doxygen's preprocessor. The corresponding
     * entries in the DOXYFILE are: */
    //////////////////////////////////////////////////////////////////////////
#if 0
    ENABLE_PREPROCESSING   = YES
    MACRO_EXPANSION        = YES
    EXPAND_ONLY_PREDEF     = YES
    SEARCH_INCLUDES        = YES
    INCLUDE_PATH           =
    INCLUDE_FILE_PATTERNS  =
    PREDEFINED             = ASSIMP_DOXYGEN_BUILD=1
    EXPAND_AS_DEFINED      = C_STRUCT C_ENUM
    SKIP_FUNCTION_MACROS   = YES
#endif
    //////////////////////////////////////////////////////////////////////////
    /* Doxygen gets confused if we use c-struct typedefs to avoid
     * the explicit 'struct' notation. This trick here has the same
     * effect as the TYPEDEF_HIDES_STRUCT option, but we don't need
     * to typedef all structs/enums. */
     //////////////////////////////////////////////////////////////////////////
#   if (defined ASSIMP_DOXYGEN_BUILD)
#       define C_STRUCT
#       define C_ENUM
#   else
#       define C_STRUCT struct
#       define C_ENUM   enum
#   endif
#endif

#if (defined(__BORLANDC__) || defined (__BCPLUSPLUS__))
#error Currently, Borland is unsupported. Feel free to port Assimp.

// "W8059 Packgröße der Struktur geändert"

#endif


    //////////////////////////////////////////////////////////////////////////
    /* Define ASSIMP_BUILD_SINGLETHREADED to compile assimp
     * without threading support. The library doesn't utilize
     * threads then and is itself not threadsafe. */
    //////////////////////////////////////////////////////////////////////////
#ifndef ASSIMP_BUILD_SINGLETHREADED
#   define ASSIMP_BUILD_SINGLETHREADED
#endif

#if defined(_DEBUG) || ! defined(NDEBUG)
#   define ASSIMP_BUILD_DEBUG
#endif

    //////////////////////////////////////////////////////////////////////////
    /* Define ASSIMP_DOUBLE_PRECISION to compile assimp
     * with double precision support (64-bit). */
    //////////////////////////////////////////////////////////////////////////

#ifdef ASSIMP_DOUBLE_PRECISION
    typedef double ai_real;
    typedef signed long long int ai_int;
    typedef unsigned long long int ai_uint;
#else // ASSIMP_DOUBLE_PRECISION
    typedef float ai_real;
    typedef signed int ai_int;
    typedef unsigned int ai_uint;
#endif // ASSIMP_DOUBLE_PRECISION

    //////////////////////////////////////////////////////////////////////////
    /* Useful constants */
    //////////////////////////////////////////////////////////////////////////

/* This is PI. Hi PI. */
#define AI_MATH_PI          (3.141592653589793238462643383279 )
#define AI_MATH_TWO_PI      (AI_MATH_PI * 2.0)
#define AI_MATH_HALF_PI     (AI_MATH_PI * 0.5)

/* And this is to avoid endless casts to float */
#define AI_MATH_PI_F        (3.1415926538f)
#define AI_MATH_TWO_PI_F    (AI_MATH_PI_F * 2.0f)
#define AI_MATH_HALF_PI_F   (AI_MATH_PI_F * 0.5f)

/* Tiny macro to convert from radians to degrees and back */
#define AI_DEG_TO_RAD(x) ((x)*(ai_real)0.0174532925)
#define AI_RAD_TO_DEG(x) ((x)*(ai_real)57.2957795)

/* Support for big-endian builds */
#if defined(__BYTE_ORDER__)
#   if (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
#       if !defined(__BIG_ENDIAN__)
#           define __BIG_ENDIAN__
#       endif
#   else /* little endian */
#       if defined (__BIG_ENDIAN__)
#           undef __BIG_ENDIAN__
#       endif
#   endif
#endif
#if defined(__BIG_ENDIAN__)
#   define AI_BUILD_BIG_ENDIAN
#endif


/* To avoid running out of memory
 * This can be adjusted for specific use cases
 * It's NOT a total limit, just a limit for individual allocations
 */
#define AI_MAX_ALLOC(type) ((256U * 1024 * 1024) / sizeof(type))

#ifndef _MSC_VER
#  define AI_NO_EXCEPT noexcept
#else
#  if (_MSC_VER == 1915 )
#    define AI_NO_EXCEPT noexcept
#  else
#    define AI_NO_EXCEPT
#  endif
#endif

#endif // !! AI_DEFINES_H_INC
