/****************************************************************************
 *
 * config/integer-types.h
 *
 *   FreeType integer types definitions.
 *
 * Copyright (C) 1996-2021 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */
#ifndef FREETYPE_CONFIG_INTEGER_TYPES_H_
#define FREETYPE_CONFIG_INTEGER_TYPES_H_

  /* There are systems (like the Texas Instruments 'C54x) where a `char`  */
  /* has 16~bits.  ANSI~C says that `sizeof(char)` is always~1.  Since an */
  /* `int` has 16~bits also for this system, `sizeof(int)` gives~1 which  */
  /* is probably unexpected.                                              */
  /*                                                                      */
  /* `CHAR_BIT` (defined in `limits.h`) gives the number of bits in a     */
  /* `char` type.                                                         */

#ifndef FT_CHAR_BIT
#define FT_CHAR_BIT  CHAR_BIT
#endif

#ifndef FT_SIZEOF_INT

  /* The size of an `int` type. */
#if                                 FT_UINT_MAX == 0xFFFFUL
#define FT_SIZEOF_INT  ( 16 / FT_CHAR_BIT )
#elif                               FT_UINT_MAX == 0xFFFFFFFFUL
#define FT_SIZEOF_INT  ( 32 / FT_CHAR_BIT )
#elif FT_UINT_MAX > 0xFFFFFFFFUL && FT_UINT_MAX == 0xFFFFFFFFFFFFFFFFUL
#define FT_SIZEOF_INT  ( 64 / FT_CHAR_BIT )
#else
#error "Unsupported size of `int' type!"
#endif

#endif  /* !defined(FT_SIZEOF_INT) */

#ifndef FT_SIZEOF_LONG

  /* The size of a `long` type.  A five-byte `long` (as used e.g. on the */
  /* DM642) is recognized but avoided.                                   */
#if                                  FT_ULONG_MAX == 0xFFFFFFFFUL
#define FT_SIZEOF_LONG  ( 32 / FT_CHAR_BIT )
#elif FT_ULONG_MAX > 0xFFFFFFFFUL && FT_ULONG_MAX == 0xFFFFFFFFFFUL
#define FT_SIZEOF_LONG  ( 32 / FT_CHAR_BIT )
#elif FT_ULONG_MAX > 0xFFFFFFFFUL && FT_ULONG_MAX == 0xFFFFFFFFFFFFFFFFUL
#define FT_SIZEOF_LONG  ( 64 / FT_CHAR_BIT )
#else
#error "Unsupported size of `long' type!"
#endif

#endif /* !defined(FT_SIZEOF_LONG) */

#ifndef FT_SIZEOF_LONG_LONG

  /* The size of a `long long` type if available */
#if defined( FT_ULLONG_MAX ) && FT_ULLONG_MAX >= 0xFFFFFFFFFFFFFFFFULL
#define FT_SIZEOF_LONG_LONG  ( 64 / FT_CHAR_BIT )
#else
#define FT_SIZEOF_LONG_LONG  0
#endif

#endif /* !defined(FT_SIZEOF_LONG_LONG) */


  /**************************************************************************
   *
   * @section:
   *   basic_types
   *
   */


  /**************************************************************************
   *
   * @type:
   *   FT_Int16
   *
   * @description:
   *   A typedef for a 16bit signed integer type.
   */
  typedef signed short  FT_Int16;


  /**************************************************************************
   *
   * @type:
   *   FT_UInt16
   *
   * @description:
   *   A typedef for a 16bit unsigned integer type.
   */
  typedef unsigned short  FT_UInt16;

  /* */


  /* this #if 0 ... #endif clause is for documentation purposes */
#if 0

  /**************************************************************************
   *
   * @type:
   *   FT_Int32
   *
   * @description:
   *   A typedef for a 32bit signed integer type.  The size depends on the
   *   configuration.
   */
  typedef signed XXX  FT_Int32;


  /**************************************************************************
   *
   * @type:
   *   FT_UInt32
   *
   *   A typedef for a 32bit unsigned integer type.  The size depends on the
   *   configuration.
   */
  typedef unsigned XXX  FT_UInt32;


  /**************************************************************************
   *
   * @type:
   *   FT_Int64
   *
   *   A typedef for a 64bit signed integer type.  The size depends on the
   *   configuration.  Only defined if there is real 64bit support;
   *   otherwise, it gets emulated with a structure (if necessary).
   */
  typedef signed XXX  FT_Int64;


  /**************************************************************************
   *
   * @type:
   *   FT_UInt64
   *
   *   A typedef for a 64bit unsigned integer type.  The size depends on the
   *   configuration.  Only defined if there is real 64bit support;
   *   otherwise, it gets emulated with a structure (if necessary).
   */
  typedef unsigned XXX  FT_UInt64;

  /* */

#endif

#if FT_SIZEOF_INT == ( 32 / FT_CHAR_BIT )

  typedef signed int      FT_Int32;
  typedef unsigned int    FT_UInt32;

#elif FT_SIZEOF_LONG == ( 32 / FT_CHAR_BIT )

  typedef signed long     FT_Int32;
  typedef unsigned long   FT_UInt32;

#else
#error "no 32bit type found -- please check your configuration files"
#endif


  /* look up an integer type that is at least 32~bits */
#if FT_SIZEOF_INT >= ( 32 / FT_CHAR_BIT )

  typedef int            FT_Fast;
  typedef unsigned int   FT_UFast;

#elif FT_SIZEOF_LONG >= ( 32 / FT_CHAR_BIT )

  typedef long           FT_Fast;
  typedef unsigned long  FT_UFast;

#endif


  /* determine whether we have a 64-bit integer type */
#if FT_SIZEOF_LONG == ( 64 / FT_CHAR_BIT )

#define FT_INT64   long
#define FT_UINT64  unsigned long

#elif FT_SIZEOF_LONG_LONG >= ( 64 / FT_CHAR_BIT )

#define FT_INT64   long long int
#define FT_UINT64  unsigned long long int

  /**************************************************************************
   *
   * A 64-bit data type may create compilation problems if you compile in
   * strict ANSI mode.  To avoid them, we disable other 64-bit data types if
   * `__STDC__` is defined.  You can however ignore this rule by defining the
   * `FT_CONFIG_OPTION_FORCE_INT64` configuration macro.
   */
#elif !defined( __STDC__ ) || defined( FT_CONFIG_OPTION_FORCE_INT64 )

#if defined( _MSC_VER ) && _MSC_VER >= 900 /* Visual C++ (and Intel C++) */

  /* this compiler provides the `__int64` type */
#define FT_INT64   __int64
#define FT_UINT64  unsigned __int64

#elif defined( __BORLANDC__ )  /* Borland C++ */

  /* XXXX: We should probably check the value of `__BORLANDC__` in order */
  /*       to test the compiler version.                                 */

  /* this compiler provides the `__int64` type */
#define FT_INT64   __int64
#define FT_UINT64  unsigned __int64

#elif defined( __WATCOMC__ )   /* Watcom C++ */

  /* Watcom doesn't provide 64-bit data types */

#elif defined( __MWERKS__ )    /* Metrowerks CodeWarrior */

#define FT_INT64   long long int
#define FT_UINT64  unsigned long long int

#elif defined( __GNUC__ )

  /* GCC provides the `long long` type */
#define FT_INT64   long long int
#define FT_UINT64  unsigned long long int

#endif /* !__STDC__ */

#endif /* FT_SIZEOF_LONG == (64 / FT_CHAR_BIT) */

#ifdef FT_INT64
  typedef FT_INT64   FT_Int64;
  typedef FT_UINT64  FT_UInt64;
#endif


#endif  /* FREETYPE_CONFIG_INTEGER_TYPES_H_ */
