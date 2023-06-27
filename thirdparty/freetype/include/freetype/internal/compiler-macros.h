/****************************************************************************
 *
 * internal/compiler-macros.h
 *
 *   Compiler-specific macro definitions used internally by FreeType.
 *
 * Copyright (C) 2020-2023 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */

#ifndef INTERNAL_COMPILER_MACROS_H_
#define INTERNAL_COMPILER_MACROS_H_

#include <freetype/config/public-macros.h>

FT_BEGIN_HEADER

  /* Fix compiler warning with sgi compiler. */
#if defined( __sgi ) && !defined( __GNUC__ )
#  if defined( _COMPILER_VERSION ) && ( _COMPILER_VERSION >= 730 )
#    pragma set woff 3505
#  endif
#endif

  /* Fix compiler warning with sgi compiler. */
#if defined( __sgi ) && !defined( __GNUC__ )
#  if defined( _COMPILER_VERSION ) && ( _COMPILER_VERSION >= 730 )
#    pragma set woff 3505
#  endif
#endif

  /* Newer compilers warn for fall-through case statements. */
#ifndef FALL_THROUGH
#  if ( defined( __STDC_VERSION__ ) && __STDC_VERSION__ > 201710L ) || \
      ( defined( __cplusplus ) && __cplusplus > 201402L )
#    define FALL_THROUGH  [[__fallthrough__]]
#  elif ( defined( __GNUC__ ) && __GNUC__ >= 7 )          || \
        ( defined( __clang__ ) && __clang_major__ >= 10 )
#    define FALL_THROUGH  __attribute__(( __fallthrough__ ))
#  else
#    define FALL_THROUGH  ( (void)0 )
#  endif
#endif

  /*
   * When defining a macro that expands to a non-trivial C statement, use
   * FT_BEGIN_STMNT and FT_END_STMNT to enclose the macro's body.  This
   * ensures there are no surprises when the macro is invoked in conditional
   * branches.
   *
   * Example:
   *
   *   #define  LOG( ... )        \
   *     FT_BEGIN_STMNT           \
   *       if ( logging_enabled ) \
   *         log( __VA_ARGS__ );  \
   *     FT_END_STMNT
   */
#define FT_BEGIN_STMNT  do {
#define FT_END_STMNT    } while ( 0 )

  /*
   * FT_DUMMY_STMNT expands to an empty C statement.  Useful for
   * conditionally defined statement macros.
   *
   * Example:
   *
   *   #ifdef BUILD_CONFIG_LOGGING
   *   #define  LOG( ... )         \
   *      FT_BEGIN_STMNT           \
   *        if ( logging_enabled ) \
   *          log( __VA_ARGS__ );  \
   *      FT_END_STMNT
   *   #else
   *   #  define LOG( ... )  FT_DUMMY_STMNT
   *   #endif
   */
#define FT_DUMMY_STMNT  FT_BEGIN_STMNT FT_END_STMNT

#ifdef __UINTPTR_TYPE__
  /*
   * GCC and Clang both provide a `__UINTPTR_TYPE__` that can be used to
   * avoid a dependency on `stdint.h`.
   */
#  define FT_UINT_TO_POINTER( x )  (void *)(__UINTPTR_TYPE__)(x)
#elif defined( _WIN64 )
  /* only 64bit Windows uses the LLP64 data model, i.e., */
  /* 32-bit integers, 64-bit pointers.                   */
#  define FT_UINT_TO_POINTER( x )  (void *)(unsigned __int64)(x)
#else
#  define FT_UINT_TO_POINTER( x )  (void *)(unsigned long)(x)
#endif

  /*
   * Use `FT_TYPEOF( type )` to cast a value to `type`.  This is useful to
   * suppress signedness compilation warnings in macros.
   *
   * Example:
   *
   *   #define PAD_( x, n )  ( (x) & ~FT_TYPEOF( x )( (n) - 1 ) )
   *
   * (The `typeof` condition is taken from gnulib's `intprops.h` header
   * file.)
   */
#if ( ( defined( __GNUC__ ) && __GNUC__ >= 2 )                       || \
      ( defined( __IBMC__ ) && __IBMC__ >= 1210 &&                      \
        defined( __IBM__TYPEOF__ ) )                                 || \
      ( defined( __SUNPRO_C ) && __SUNPRO_C >= 0x5110 && !__STDC__ ) )
#define FT_TYPEOF( type )  ( __typeof__ ( type ) )
#else
#define FT_TYPEOF( type )  /* empty */
#endif

  /*
   * Mark a function declaration as internal to the library.  This ensures
   * that it will not be exposed by default to client code, and helps
   * generate smaller and faster code on ELF-based platforms.  Place this
   * before a function declaration.
   */

  /* Visual C, mingw */
#if defined( _WIN32 )
#define FT_INTERNAL_FUNCTION_ATTRIBUTE  /* empty */

  /* gcc, clang */
#elif ( defined( __GNUC__ ) && __GNUC__ >= 4 ) || defined( __clang__ )
#define FT_INTERNAL_FUNCTION_ATTRIBUTE  \
          __attribute__(( visibility( "hidden" ) ))

  /* Sun */
#elif defined( __SUNPRO_C ) && __SUNPRO_C >= 0x550
#define FT_INTERNAL_FUNCTION_ATTRIBUTE  __hidden

#else
#define FT_INTERNAL_FUNCTION_ATTRIBUTE  /* empty */
#endif

  /*
   * FreeType supports compilation of its C sources with a C++ compiler (in
   * C++ mode); this introduces a number of subtle issues.
   *
   * The main one is that a C++ function declaration and its definition must
   * have the same 'linkage'.  Because all FreeType headers declare their
   * functions with C linkage (i.e., within an `extern "C" { ... }` block
   * due to the magic of FT_BEGIN_HEADER and FT_END_HEADER), their
   * definition in FreeType sources should also be prefixed with `extern
   * "C"` when compiled in C++ mode.
   *
   * The `FT_FUNCTION_DECLARATION` and `FT_FUNCTION_DEFINITION` macros are
   * provided to deal with this case, as well as `FT_CALLBACK_DEF` and its
   * siblings below.
   */

  /*
   * `FT_FUNCTION_DECLARATION( type )` can be used to write a C function
   * declaration to ensure it will have C linkage when the library is built
   * with a C++ compiler.  The parameter is the function's return type, so a
   * declaration would look like
   *
   *    FT_FUNCTION_DECLARATION( int )
   *    foo( int x );
   *
   * NOTE: This requires that all uses are inside of `FT_BEGIN_HEADER ...
   * FT_END_HEADER` blocks, which guarantees that the declarations have C
   * linkage when the headers are included by C++ sources.
   *
   * NOTE: Do not use directly.  Use `FT_LOCAL`, `FT_BASE`, and `FT_EXPORT`
   * instead.
   */
#define FT_FUNCTION_DECLARATION( x )  extern x

  /*
   * Same as `FT_FUNCTION_DECLARATION`, but for function definitions instead.
   *
   * NOTE: Do not use directly.  Use `FT_LOCAL_DEF`, `FT_BASE_DEF`, and
   * `FT_EXPORT_DEF` instead.
   */
#ifdef __cplusplus
#define FT_FUNCTION_DEFINITION( x )  extern "C" x
#else
#define FT_FUNCTION_DEFINITION( x )  x
#endif

  /*
   * Use `FT_LOCAL` and `FT_LOCAL_DEF` to declare and define, respectively,
   * an internal FreeType function that is only used by the sources of a
   * single `src/module/` directory.  This ensures that the functions are
   * turned into static ones at build time, resulting in smaller and faster
   * code.
   */
#ifdef FT_MAKE_OPTION_SINGLE_OBJECT

#define FT_LOCAL( x )      static x
#define FT_LOCAL_DEF( x )  static x

#else

#define FT_LOCAL( x )      FT_INTERNAL_FUNCTION_ATTRIBUTE \
                           FT_FUNCTION_DECLARATION( x )
#define FT_LOCAL_DEF( x )  FT_FUNCTION_DEFINITION( x )

#endif  /* FT_MAKE_OPTION_SINGLE_OBJECT */

  /*
   * Use `FT_LOCAL_ARRAY` and `FT_LOCAL_ARRAY_DEF` to declare and define,
   * respectively, a constant array that must be accessed from several
   * sources in the same `src/module/` sub-directory, and which are internal
   * to the library.
   */
#define FT_LOCAL_ARRAY( x )      FT_INTERNAL_FUNCTION_ATTRIBUTE \
                                 extern const x
#define FT_LOCAL_ARRAY_DEF( x )  FT_FUNCTION_DEFINITION( const x )

  /*
   * `Use FT_BASE` and `FT_BASE_DEF` to declare and define, respectively, an
   * internal library function that is used by more than a single module.
   */
#define FT_BASE( x )      FT_INTERNAL_FUNCTION_ATTRIBUTE \
                          FT_FUNCTION_DECLARATION( x )
#define FT_BASE_DEF( x )  FT_FUNCTION_DEFINITION( x )


  /*
   * NOTE: Conditionally define `FT_EXPORT_VAR` due to its definition in
   * `src/smooth/ftgrays.h` to make the header more portable.
   */
#ifndef FT_EXPORT_VAR
#define FT_EXPORT_VAR( x )  FT_FUNCTION_DECLARATION( x )
#endif

  /*
   * When compiling FreeType as a DLL or DSO with hidden visibility,
   * some systems/compilers need a special attribute in front OR after
   * the return type of function declarations.
   *
   * Two macros are used within the FreeType source code to define
   * exported library functions: `FT_EXPORT` and `FT_EXPORT_DEF`.
   *
   * - `FT_EXPORT( return_type )`
   *
   *   is used in a function declaration, as in
   *
   *   ```
   *     FT_EXPORT( FT_Error )
   *     FT_Init_FreeType( FT_Library*  alibrary );
   *   ```
   *
   * - `FT_EXPORT_DEF( return_type )`
   *
   *   is used in a function definition, as in
   *
   *   ```
   *     FT_EXPORT_DEF( FT_Error )
   *     FT_Init_FreeType( FT_Library*  alibrary )
   *     {
   *       ... some code ...
   *       return FT_Err_Ok;
   *     }
   *   ```
   *
   * You can provide your own implementation of `FT_EXPORT` and
   * `FT_EXPORT_DEF` here if you want.
   *
   * To export a variable, use `FT_EXPORT_VAR`.
   */

  /* See `freetype/config/public-macros.h` for the `FT_EXPORT` definition */
#define FT_EXPORT_DEF( x )  FT_FUNCTION_DEFINITION( x )

  /*
   * The following macros are needed to compile the library with a
   * C++ compiler and with 16bit compilers.
   */

  /*
   * This is special.  Within C++, you must specify `extern "C"` for
   * functions which are used via function pointers, and you also
   * must do that for structures which contain function pointers to
   * assure C linkage -- it's not possible to have (local) anonymous
   * functions which are accessed by (global) function pointers.
   *
   *
   * FT_CALLBACK_DEF is used to _define_ a callback function,
   * located in the same source code file as the structure that uses
   * it.  FT_COMPARE_DEF, in addition, ensures the `cdecl` calling
   * convention on x86, required by the C library function `qsort`.
   *
   * FT_BASE_CALLBACK and FT_BASE_CALLBACK_DEF are used to declare
   * and define a callback function, respectively, in a similar way
   * as FT_BASE and FT_BASE_DEF work.
   *
   * FT_CALLBACK_TABLE is used to _declare_ a constant variable that
   * contains pointers to callback functions.
   *
   * FT_CALLBACK_TABLE_DEF is used to _define_ a constant variable
   * that contains pointers to callback functions.
   *
   *
   * Some 16bit compilers have to redefine these macros to insert
   * the infamous `_cdecl` or `__fastcall` declarations.
   */
#ifdef __cplusplus
#define FT_CALLBACK_DEF( x )  extern "C"  x
#else
#define FT_CALLBACK_DEF( x )  static  x
#endif

#if defined( __GNUC__ ) && defined( __i386__ )
#define FT_COMPARE_DEF( x )  FT_CALLBACK_DEF( x ) __attribute__(( cdecl ))
#elif defined( _MSC_VER ) && defined( _M_IX86 )
#define FT_COMPARE_DEF( x )  FT_CALLBACK_DEF( x ) __cdecl
#elif defined( __WATCOMC__ ) && __WATCOMC__ >= 1240
#define FT_COMPARE_DEF( x )  FT_CALLBACK_DEF( x ) __watcall
#else
#define FT_COMPARE_DEF( x )  FT_CALLBACK_DEF( x )
#endif

#define FT_BASE_CALLBACK( x )      FT_FUNCTION_DECLARATION( x )
#define FT_BASE_CALLBACK_DEF( x )  FT_FUNCTION_DEFINITION( x )

#ifndef FT_CALLBACK_TABLE
#ifdef __cplusplus
#define FT_CALLBACK_TABLE      extern "C"
#define FT_CALLBACK_TABLE_DEF  extern "C"
#else
#define FT_CALLBACK_TABLE      extern
#define FT_CALLBACK_TABLE_DEF  /* nothing */
#endif
#endif /* FT_CALLBACK_TABLE */

FT_END_HEADER

#endif  /* INTERNAL_COMPILER_MACROS_H_ */
