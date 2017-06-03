/***************************************************************************/
/*                                                                         */
/*  ftconfig.h                                                             */
/*                                                                         */
/*    ANSI-specific configuration file (specification only).               */
/*                                                                         */
/*  Copyright 1996-2016 by                                                 */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


  /*************************************************************************/
  /*                                                                       */
  /* This header file contains a number of macro definitions that are used */
  /* by the rest of the engine.  Most of the macros here are automatically */
  /* determined at compile time, and you should not need to change it to   */
  /* port FreeType, except to compile the library with a non-ANSI          */
  /* compiler.                                                             */
  /*                                                                       */
  /* Note however that if some specific modifications are needed, we       */
  /* advise you to place a modified copy in your build directory.          */
  /*                                                                       */
  /* The build directory is usually `builds/<system>', and contains        */
  /* system-specific files that are always included first when building    */
  /* the library.                                                          */
  /*                                                                       */
  /* This ANSI version should stay in `include/config/'.                   */
  /*                                                                       */
  /*************************************************************************/

#ifndef FTCONFIG_H_
#define FTCONFIG_H_

#include <ft2build.h>
#include FT_CONFIG_OPTIONS_H
#include FT_CONFIG_STANDARD_LIBRARY_H


FT_BEGIN_HEADER


  /*************************************************************************/
  /*                                                                       */
  /*               PLATFORM-SPECIFIC CONFIGURATION MACROS                  */
  /*                                                                       */
  /* These macros can be toggled to suit a specific system.  The current   */
  /* ones are defaults used to compile FreeType in an ANSI C environment   */
  /* (16bit compilers are also supported).  Copy this file to your own     */
  /* `builds/<system>' directory, and edit it to port the engine.          */
  /*                                                                       */
  /*************************************************************************/


  /* There are systems (like the Texas Instruments 'C54x) where a `char' */
  /* has 16 bits.  ANSI C says that sizeof(char) is always 1.  Since an  */
  /* `int' has 16 bits also for this system, sizeof(int) gives 1 which   */
  /* is probably unexpected.                                             */
  /*                                                                     */
  /* `CHAR_BIT' (defined in limits.h) gives the number of bits in a      */
  /* `char' type.                                                        */

#ifndef FT_CHAR_BIT
#define FT_CHAR_BIT  CHAR_BIT
#endif


  /* The size of an `int' type.  */
#if                                 FT_UINT_MAX == 0xFFFFUL
#define FT_SIZEOF_INT  (16 / FT_CHAR_BIT)
#elif                               FT_UINT_MAX == 0xFFFFFFFFUL
#define FT_SIZEOF_INT  (32 / FT_CHAR_BIT)
#elif FT_UINT_MAX > 0xFFFFFFFFUL && FT_UINT_MAX == 0xFFFFFFFFFFFFFFFFUL
#define FT_SIZEOF_INT  (64 / FT_CHAR_BIT)
#else
#error "Unsupported size of `int' type!"
#endif

  /* The size of a `long' type.  A five-byte `long' (as used e.g. on the */
  /* DM642) is recognized but avoided.                                   */
#if                                  FT_ULONG_MAX == 0xFFFFFFFFUL
#define FT_SIZEOF_LONG  (32 / FT_CHAR_BIT)
#elif FT_ULONG_MAX > 0xFFFFFFFFUL && FT_ULONG_MAX == 0xFFFFFFFFFFUL
#define FT_SIZEOF_LONG  (32 / FT_CHAR_BIT)
#elif FT_ULONG_MAX > 0xFFFFFFFFUL && FT_ULONG_MAX == 0xFFFFFFFFFFFFFFFFUL
#define FT_SIZEOF_LONG  (64 / FT_CHAR_BIT)
#else
#error "Unsupported size of `long' type!"
#endif


  /* FT_UNUSED is a macro used to indicate that a given parameter is not  */
  /* used -- this is only used to get rid of unpleasant compiler warnings */
#ifndef FT_UNUSED
#define FT_UNUSED( arg )  ( (arg) = (arg) )
#endif


  /*************************************************************************/
  /*                                                                       */
  /*                     AUTOMATIC CONFIGURATION MACROS                    */
  /*                                                                       */
  /* These macros are computed from the ones defined above.  Don't touch   */
  /* their definition, unless you know precisely what you are doing.  No   */
  /* porter should need to mess with them.                                 */
  /*                                                                       */
  /*************************************************************************/


  /*************************************************************************/
  /*                                                                       */
  /* Mac support                                                           */
  /*                                                                       */
  /*   This is the only necessary change, so it is defined here instead    */
  /*   providing a new configuration file.                                 */
  /*                                                                       */
#if defined( __APPLE__ ) || ( defined( __MWERKS__ ) && defined( macintosh ) )
  /* no Carbon frameworks for 64bit 10.4.x */
  /* AvailabilityMacros.h is available since Mac OS X 10.2,        */
  /* so guess the system version by maximum errno before inclusion */
#include <errno.h>
#ifdef ECANCELED /* defined since 10.2 */
#include "AvailabilityMacros.h"
#endif
#if defined( __LP64__ ) && \
    ( MAC_OS_X_VERSION_MIN_REQUIRED <= MAC_OS_X_VERSION_10_4 )
#undef FT_MACINTOSH
#endif

#elif defined( __SC__ ) || defined( __MRC__ )
  /* Classic MacOS compilers */
#include "ConditionalMacros.h"
#if TARGET_OS_MAC
#define FT_MACINTOSH 1
#endif

#endif


  /*************************************************************************/
  /*                                                                       */
  /* <Section>                                                             */
  /*    basic_types                                                        */
  /*                                                                       */
  /*************************************************************************/


  /*************************************************************************/
  /*                                                                       */
  /* <Type>                                                                */
  /*    FT_Int16                                                           */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A typedef for a 16bit signed integer type.                         */
  /*                                                                       */
  typedef signed short  FT_Int16;


  /*************************************************************************/
  /*                                                                       */
  /* <Type>                                                                */
  /*    FT_UInt16                                                          */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A typedef for a 16bit unsigned integer type.                       */
  /*                                                                       */
  typedef unsigned short  FT_UInt16;

  /* */


  /* this #if 0 ... #endif clause is for documentation purposes */
#if 0

  /*************************************************************************/
  /*                                                                       */
  /* <Type>                                                                */
  /*    FT_Int32                                                           */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A typedef for a 32bit signed integer type.  The size depends on    */
  /*    the configuration.                                                 */
  /*                                                                       */
  typedef signed XXX  FT_Int32;


  /*************************************************************************/
  /*                                                                       */
  /* <Type>                                                                */
  /*    FT_UInt32                                                          */
  /*                                                                       */
  /*    A typedef for a 32bit unsigned integer type.  The size depends on  */
  /*    the configuration.                                                 */
  /*                                                                       */
  typedef unsigned XXX  FT_UInt32;


  /*************************************************************************/
  /*                                                                       */
  /* <Type>                                                                */
  /*    FT_Int64                                                           */
  /*                                                                       */
  /*    A typedef for a 64bit signed integer type.  The size depends on    */
  /*    the configuration.  Only defined if there is real 64bit support;   */
  /*    otherwise, it gets emulated with a structure (if necessary).       */
  /*                                                                       */
  typedef signed XXX  FT_Int64;


  /*************************************************************************/
  /*                                                                       */
  /* <Type>                                                                */
  /*    FT_UInt64                                                          */
  /*                                                                       */
  /*    A typedef for a 64bit unsigned integer type.  The size depends on  */
  /*    the configuration.  Only defined if there is real 64bit support;   */
  /*    otherwise, it gets emulated with a structure (if necessary).       */
  /*                                                                       */
  typedef unsigned XXX  FT_UInt64;

  /* */

#endif

#if FT_SIZEOF_INT == (32 / FT_CHAR_BIT)

  typedef signed int      FT_Int32;
  typedef unsigned int    FT_UInt32;

#elif FT_SIZEOF_LONG == (32 / FT_CHAR_BIT)

  typedef signed long     FT_Int32;
  typedef unsigned long   FT_UInt32;

#else
#error "no 32bit type found -- please check your configuration files"
#endif


  /* look up an integer type that is at least 32 bits */
#if FT_SIZEOF_INT >= (32 / FT_CHAR_BIT)

  typedef int            FT_Fast;
  typedef unsigned int   FT_UFast;

#elif FT_SIZEOF_LONG >= (32 / FT_CHAR_BIT)

  typedef long           FT_Fast;
  typedef unsigned long  FT_UFast;

#endif


  /* determine whether we have a 64-bit int type for platforms without */
  /* Autoconf                                                          */
#if FT_SIZEOF_LONG == (64 / FT_CHAR_BIT)

  /* FT_LONG64 must be defined if a 64-bit type is available */
#define FT_LONG64
#define FT_INT64   long
#define FT_UINT64  unsigned long

  /*************************************************************************/
  /*                                                                       */
  /* A 64-bit data type may create compilation problems if you compile     */
  /* in strict ANSI mode.  To avoid them, we disable other 64-bit data     */
  /* types if __STDC__ is defined.  You can however ignore this rule       */
  /* by defining the FT_CONFIG_OPTION_FORCE_INT64 configuration macro.     */
  /*                                                                       */
#elif !defined( __STDC__ ) || defined( FT_CONFIG_OPTION_FORCE_INT64 )

#if defined( __STDC_VERSION__ ) && __STDC_VERSION__ >= 199901L

#define FT_LONG64
#define FT_INT64   long long int
#define FT_UINT64  unsigned long long int

#elif defined( _MSC_VER ) && _MSC_VER >= 900  /* Visual C++ (and Intel C++) */

  /* this compiler provides the __int64 type */
#define FT_LONG64
#define FT_INT64   __int64
#define FT_UINT64  unsigned __int64

#elif defined( __BORLANDC__ )  /* Borland C++ */

  /* XXXX: We should probably check the value of __BORLANDC__ in order */
  /*       to test the compiler version.                               */

  /* this compiler provides the __int64 type */
#define FT_LONG64
#define FT_INT64   __int64
#define FT_UINT64  unsigned __int64

#elif defined( __WATCOMC__ )   /* Watcom C++ */

  /* Watcom doesn't provide 64-bit data types */

#elif defined( __MWERKS__ )    /* Metrowerks CodeWarrior */

#define FT_LONG64
#define FT_INT64   long long int
#define FT_UINT64  unsigned long long int

#elif defined( __GNUC__ )

  /* GCC provides the `long long' type */
#define FT_LONG64
#define FT_INT64   long long int
#define FT_UINT64  unsigned long long int

#endif /* __STDC_VERSION__ >= 199901L */

#endif /* FT_SIZEOF_LONG == (64 / FT_CHAR_BIT) */

#ifdef FT_LONG64
  typedef FT_INT64   FT_Int64;
  typedef FT_UINT64  FT_UInt64;
#endif


  /*************************************************************************/
  /*                                                                       */
  /* miscellaneous                                                         */
  /*                                                                       */
  /*************************************************************************/


#define FT_BEGIN_STMNT  do {
#define FT_END_STMNT    } while ( 0 )
#define FT_DUMMY_STMNT  FT_BEGIN_STMNT FT_END_STMNT


  /* typeof condition taken from gnulib's `intprops.h' header file */
#if ( __GNUC__ >= 2                         || \
      defined( __IBM__TYPEOF__ )            || \
      ( __SUNPRO_C >= 0x5110 && !__STDC__ ) )
#define FT_TYPEOF( type )  (__typeof__ (type))
#else
#define FT_TYPEOF( type )  /* empty */
#endif


#ifdef FT_MAKE_OPTION_SINGLE_OBJECT

#define FT_LOCAL( x )      static  x
#define FT_LOCAL_DEF( x )  static  x

#else

#ifdef __cplusplus
#define FT_LOCAL( x )      extern "C"  x
#define FT_LOCAL_DEF( x )  extern "C"  x
#else
#define FT_LOCAL( x )      extern  x
#define FT_LOCAL_DEF( x )  x
#endif

#endif /* FT_MAKE_OPTION_SINGLE_OBJECT */

#define FT_LOCAL_ARRAY( x )      extern const  x
#define FT_LOCAL_ARRAY_DEF( x )  const  x


#ifndef FT_BASE

#ifdef __cplusplus
#define FT_BASE( x )  extern "C"  x
#else
#define FT_BASE( x )  extern  x
#endif

#endif /* !FT_BASE */


#ifndef FT_BASE_DEF

#ifdef __cplusplus
#define FT_BASE_DEF( x )  x
#else
#define FT_BASE_DEF( x )  x
#endif

#endif /* !FT_BASE_DEF */


#ifndef FT_EXPORT

#ifdef __cplusplus
#define FT_EXPORT( x )  extern "C"  x
#else
#define FT_EXPORT( x )  extern  x
#endif

#endif /* !FT_EXPORT */


#ifndef FT_EXPORT_DEF

#ifdef __cplusplus
#define FT_EXPORT_DEF( x )  extern "C"  x
#else
#define FT_EXPORT_DEF( x )  extern  x
#endif

#endif /* !FT_EXPORT_DEF */


#ifndef FT_EXPORT_VAR

#ifdef __cplusplus
#define FT_EXPORT_VAR( x )  extern "C"  x
#else
#define FT_EXPORT_VAR( x )  extern  x
#endif

#endif /* !FT_EXPORT_VAR */

  /* The following macros are needed to compile the library with a   */
  /* C++ compiler and with 16bit compilers.                          */
  /*                                                                 */

  /* This is special.  Within C++, you must specify `extern "C"' for */
  /* functions which are used via function pointers, and you also    */
  /* must do that for structures which contain function pointers to  */
  /* assure C linkage -- it's not possible to have (local) anonymous */
  /* functions which are accessed by (global) function pointers.     */
  /*                                                                 */
  /*                                                                 */
  /* FT_CALLBACK_DEF is used to _define_ a callback function.        */
  /*                                                                 */
  /* FT_CALLBACK_TABLE is used to _declare_ a constant variable that */
  /* contains pointers to callback functions.                        */
  /*                                                                 */
  /* FT_CALLBACK_TABLE_DEF is used to _define_ a constant variable   */
  /* that contains pointers to callback functions.                   */
  /*                                                                 */
  /*                                                                 */
  /* Some 16bit compilers have to redefine these macros to insert    */
  /* the infamous `_cdecl' or `__fastcall' declarations.             */
  /*                                                                 */
#ifndef FT_CALLBACK_DEF
#ifdef __cplusplus
#define FT_CALLBACK_DEF( x )  extern "C"  x
#else
#define FT_CALLBACK_DEF( x )  static  x
#endif
#endif /* FT_CALLBACK_DEF */

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


#endif /* FTCONFIG_H_ */


/* END */
