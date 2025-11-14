/****************************************************************************
 *
 * config/public-macros.h
 *
 *   Define a set of compiler macros used in public FreeType headers.
 *
 * Copyright (C) 2020-2025 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */

  /*
   * The definitions in this file are used by the public FreeType headers
   * and thus should be considered part of the public API.
   *
   * Other compiler-specific macro definitions that are not exposed by the
   * FreeType API should go into
   * `include/freetype/internal/compiler-macros.h` instead.
   */
#ifndef FREETYPE_CONFIG_PUBLIC_MACROS_H_
#define FREETYPE_CONFIG_PUBLIC_MACROS_H_

  /*
   * `FT_BEGIN_HEADER` and `FT_END_HEADER` might have already been defined
   * by `freetype/config/ftheader.h`, but we don't want to include this
   * header here, so redefine the macros here only when needed.  Their
   * definition is very stable, so keeping them in sync with the ones in the
   * header should not be a maintenance issue.
   */
#ifndef FT_BEGIN_HEADER
#ifdef __cplusplus
#define FT_BEGIN_HEADER  extern "C" {
#else
#define FT_BEGIN_HEADER  /* empty */
#endif
#endif  /* FT_BEGIN_HEADER */

#ifndef FT_END_HEADER
#ifdef __cplusplus
#define FT_END_HEADER  }
#else
#define FT_END_HEADER  /* empty */
#endif
#endif  /* FT_END_HEADER */


FT_BEGIN_HEADER

  /*
   * Mark a function declaration as public.  This ensures it will be
   * properly exported to client code.  Place this before a function
   * declaration.
   *
   * NOTE: This macro should be considered an internal implementation
   * detail, and not part of the FreeType API.  It is only defined here
   * because it is needed by `FT_EXPORT`.
   */

  /* Visual C, MinGW, Cygwin */
#if defined( _WIN32 ) || defined( __CYGWIN__ )

#if defined( FT2_BUILD_LIBRARY ) && defined( DLL_EXPORT )
#define FT_PUBLIC_FUNCTION_ATTRIBUTE  __declspec( dllexport )
#elif defined( DLL_IMPORT )
#define FT_PUBLIC_FUNCTION_ATTRIBUTE  __declspec( dllimport )
#endif

  /* gcc, clang */
#elif ( defined( __GNUC__ ) && __GNUC__ >= 4 ) || defined( __clang__ )
#define FT_PUBLIC_FUNCTION_ATTRIBUTE \
          __attribute__(( visibility( "default" ) ))

  /* Sun */
#elif defined( __SUNPRO_C ) && __SUNPRO_C >= 0x550
#define FT_PUBLIC_FUNCTION_ATTRIBUTE  __global
#endif


#ifndef FT_PUBLIC_FUNCTION_ATTRIBUTE
#define FT_PUBLIC_FUNCTION_ATTRIBUTE  /* empty */
#endif


  /*
   * Define a public FreeType API function.  This ensures it is properly
   * exported or imported at build time.  The macro parameter is the
   * function's return type as in:
   *
   *   FT_EXPORT( FT_Bool )
   *   FT_Object_Method( FT_Object  obj,
   *                     ... );
   *
   * NOTE: This requires that all `FT_EXPORT` uses are inside
   * `FT_BEGIN_HEADER ... FT_END_HEADER` blocks.  This guarantees that the
   * functions are exported with C linkage, even when the header is included
   * by a C++ source file.
   */
#define FT_EXPORT( x )  FT_PUBLIC_FUNCTION_ATTRIBUTE extern x


  /*
   * `FT_UNUSED` indicates that a given parameter is not used -- this is
   * only used to get rid of unpleasant compiler warnings.
   *
   * Technically, this was not meant to be part of the public API, but some
   * third-party code depends on it.
   */
#ifndef FT_UNUSED
#define FT_UNUSED( arg )  ( (arg) = (arg) )
#endif


  /*
   * Support for casts in both C and C++.
   */
#ifdef __cplusplus
#define FT_STATIC_CAST( type, var )       static_cast<type>(var)
#define FT_REINTERPRET_CAST( type, var )  reinterpret_cast<type>(var)

#define FT_STATIC_BYTE_CAST( type, var )                         \
          static_cast<type>( static_cast<unsigned char>( var ) )
#else
#define FT_STATIC_CAST( type, var )       (type)(var)
#define FT_REINTERPRET_CAST( type, var )  (type)(var)

#define FT_STATIC_BYTE_CAST( type, var )  (type)(unsigned char)(var)
#endif


FT_END_HEADER

#endif  /* FREETYPE_CONFIG_PUBLIC_MACROS_H_ */
