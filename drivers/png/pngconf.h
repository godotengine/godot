
/* pngconf.h - machine configurable file for libpng
 *
 * libpng version 1.5.27, May 26, 2016
 *
 * Copyright (c) 1998-2002,2004,2006-2013 Glenn Randers-Pehrson
 * (Version 0.96 Copyright (c) 1996, 1997 Andreas Dilger)
 * (Version 0.88 Copyright (c) 1995, 1996 Guy Eric Schalnat, Group 42, Inc.)
 *
 * This code is released under the libpng license.
 * For conditions of distribution and use, see the disclaimer
 * and license in png.h
 *
 * Any machine specific code is near the front of this file, so if you
 * are configuring libpng for a machine, you may want to read the section
 * starting here down to where it starts to typedef png_color, png_text,
 * and png_info.
 */

#ifndef PNGCONF_H
#define PNGCONF_H

#ifndef PNG_BUILDING_SYMBOL_TABLE
/* PNG_NO_LIMITS_H may be used to turn off the use of the standard C
 * definition file for  machine specific limits, this may impact the
 * correctness of the definitions below (see uses of INT_MAX).
 */
#  ifndef PNG_NO_LIMITS_H
#    include <limits.h>
#  endif

/* For the memory copy APIs (i.e. the standard definitions of these),
 * because this file defines png_memcpy and so on the base APIs must
 * be defined here.
 */
#  ifdef BSD
#    include <strings.h>
#  else
#    include <string.h>
#  endif

/* For png_FILE_p - this provides the standard definition of a
 * FILE
 */
#  ifdef PNG_STDIO_SUPPORTED
#    include <stdio.h>
#  endif
#endif

/* This controls optimization of the reading of 16 and 32 bit values
 * from PNG files.  It can be set on a per-app-file basis - it
 * just changes whether a macro is used when the function is called.
 * The library builder sets the default; if read functions are not
 * built into the library the macro implementation is forced on.
 */
#ifndef PNG_READ_INT_FUNCTIONS_SUPPORTED
#  define PNG_USE_READ_MACROS
#endif
#if !defined(PNG_NO_USE_READ_MACROS) && !defined(PNG_USE_READ_MACROS)
#  if PNG_DEFAULT_READ_MACROS
#    define PNG_USE_READ_MACROS
#  endif
#endif

/* COMPILER SPECIFIC OPTIONS.
 *
 * These options are provided so that a variety of difficult compilers
 * can be used.  Some are fixed at build time (e.g. PNG_API_RULE
 * below) but still have compiler specific implementations, others
 * may be changed on a per-file basis when compiling against libpng.
 */

/* The PNGARG macro protects us against machines that don't have function
 * prototypes (ie K&R style headers).  If your compiler does not handle
 * function prototypes, define this macro and use the included ansi2knr.
 * I've always been able to use _NO_PROTO as the indicator, but you may
 * need to drag the empty declaration out in front of here, or change the
 * ifdef to suit your own needs.
 */
#ifndef PNGARG

#  ifdef OF /* zlib prototype munger */
#    define PNGARG(arglist) OF(arglist)
#  else

#    ifdef _NO_PROTO
#      define PNGARG(arglist) ()
#    else
#      define PNGARG(arglist) arglist
#    endif /* _NO_PROTO */

#  endif /* OF */

#endif /* PNGARG */

/* Function calling conventions.
 * =============================
 * Normally it is not necessary to specify to the compiler how to call
 * a function - it just does it - however on x86 systems derived from
 * Microsoft and Borland C compilers ('IBM PC', 'DOS', 'Windows' systems
 * and some others) there are multiple ways to call a function and the
 * default can be changed on the compiler command line.  For this reason
 * libpng specifies the calling convention of every exported function and
 * every function called via a user supplied function pointer.  This is
 * done in this file by defining the following macros:
 *
 * PNGAPI    Calling convention for exported functions.
 * PNGCBAPI  Calling convention for user provided (callback) functions.
 * PNGCAPI   Calling convention used by the ANSI-C library (required
 *           for longjmp callbacks and sometimes used internally to
 *           specify the calling convention for zlib).
 *
 * These macros should never be overridden.  If it is necessary to
 * change calling convention in a private build this can be done
 * by setting PNG_API_RULE (which defaults to 0) to one of the values
 * below to select the correct 'API' variants.
 *
 * PNG_API_RULE=0 Use PNGCAPI - the 'C' calling convention - throughout.
 *                This is correct in every known environment.
 * PNG_API_RULE=1 Use the operating system convention for PNGAPI and
 *                the 'C' calling convention (from PNGCAPI) for
 *                callbacks (PNGCBAPI).  This is no longer required
 *                in any known environment - if it has to be used
 *                please post an explanation of the problem to the
 *                libpng mailing list.
 *
 * These cases only differ if the operating system does not use the C
 * calling convention, at present this just means the above cases
 * (x86 DOS/Windows sytems) and, even then, this does not apply to
 * Cygwin running on those systems.
 *
 * Note that the value must be defined in pnglibconf.h so that what
 * the application uses to call the library matches the conventions
 * set when building the library.
 */

/* Symbol export
 * =============
 * When building a shared library it is almost always necessary to tell
 * the compiler which symbols to export.  The png.h macro 'PNG_EXPORT'
 * is used to mark the symbols.  On some systems these symbols can be
 * extracted at link time and need no special processing by the compiler,
 * on other systems the symbols are flagged by the compiler and just
 * the declaration requires a special tag applied (unfortunately) in a
 * compiler dependent way.  Some systems can do either.
 *
 * A small number of older systems also require a symbol from a DLL to
 * be flagged to the program that calls it.  This is a problem because
 * we do not know in the header file included by application code that
 * the symbol will come from a shared library, as opposed to a statically
 * linked one.  For this reason the application must tell us by setting
 * the magic flag PNG_USE_DLL to turn on the special processing before
 * it includes png.h.
 *
 * Four additional macros are used to make this happen:
 *
 * PNG_IMPEXP The magic (if any) to cause a symbol to be exported from
 *            the build or imported if PNG_USE_DLL is set - compiler
 *            and system specific.
 *
 * PNG_EXPORT_TYPE(type) A macro that pre or appends PNG_IMPEXP to
 *                       'type', compiler specific.
 *
 * PNG_DLL_EXPORT Set to the magic to use during a libpng build to
 *                make a symbol exported from the DLL.  Not used in the
 *                public header files; see pngpriv.h for how it is used
 *                in the libpng build.
 *
 * PNG_DLL_IMPORT Set to the magic to force the libpng symbols to come
 *                from a DLL - used to define PNG_IMPEXP when
 *                PNG_USE_DLL is set.
 */

/* System specific discovery.
 * ==========================
 * This code is used at build time to find PNG_IMPEXP, the API settings
 * and PNG_EXPORT_TYPE(), it may also set a macro to indicate the DLL
 * import processing is possible.  On Windows systems it also sets
 * compiler-specific macros to the values required to change the calling
 * conventions of the various functions.
 */
#if defined(_Windows) || defined(_WINDOWS) || defined(WIN32) ||\
    defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
  /* Windows system (DOS doesn't support DLLs).  Includes builds under Cygwin or
   * MinGW on any architecture currently supported by Windows.  Also includes
   * Watcom builds but these need special treatment because they are not
   * compatible with GCC or Visual C because of different calling conventions.
   */
#  if PNG_API_RULE == 2
    /* If this line results in an error, either because __watcall is not
     * understood or because of a redefine just below you cannot use *this*
     * build of the library with the compiler you are using.  *This* build was
     * build using Watcom and applications must also be built using Watcom!
     */
#    define PNGCAPI __watcall
#  endif

#  if defined(__GNUC__) || (defined(_MSC_VER) && (_MSC_VER >= 800))
#    define PNGCAPI __cdecl
#    if PNG_API_RULE == 1
       /* If this line results in an error __stdcall is not understood and
        * PNG_API_RULE should not have been set to '1'.
        */
#      define PNGAPI __stdcall
#    endif
#  else
    /* An older compiler, or one not detected (erroneously) above,
     * if necessary override on the command line to get the correct
     * variants for the compiler.
     */
#    ifndef PNGCAPI
#      define PNGCAPI _cdecl
#    endif
#    if PNG_API_RULE == 1 && !defined(PNGAPI)
#      define PNGAPI _stdcall
#    endif
#  endif /* compiler/api */
  /* NOTE: PNGCBAPI always defaults to PNGCAPI. */

#  if defined(PNGAPI) && !defined(PNG_USER_PRIVATEBUILD)
   ERROR: PNG_USER_PRIVATEBUILD must be defined if PNGAPI is changed
#  endif

#  if (defined(_MSC_VER) && _MSC_VER < 800) ||\
      (defined(__BORLANDC__) && __BORLANDC__ < 0x500)
    /* older Borland and MSC
     * compilers used '__export' and required this to be after
     * the type.
     */
#    ifndef PNG_EXPORT_TYPE
#      define PNG_EXPORT_TYPE(type) type PNG_IMPEXP
#    endif
#    define PNG_DLL_EXPORT __export
#  else /* newer compiler */
#    define PNG_DLL_EXPORT __declspec(dllexport)
#    ifndef PNG_DLL_IMPORT
#      define PNG_DLL_IMPORT __declspec(dllimport)
#    endif
#  endif /* compiler */

#else /* !Windows */
#  if (defined(__IBMC__) || defined(__IBMCPP__)) && defined(__OS2__)
#    define PNGAPI _System
#  else /* !Windows/x86 && !OS/2 */
    /* Use the defaults, or define PNG*API on the command line (but
     * this will have to be done for every compile!)
     */
#  endif /* other system, !OS/2 */
#endif /* !Windows/x86 */

/* Now do all the defaulting . */
#ifndef PNGCAPI
#  define PNGCAPI
#endif
#ifndef PNGCBAPI
#  define PNGCBAPI PNGCAPI
#endif
#ifndef PNGAPI
#  define PNGAPI PNGCAPI
#endif

/* PNG_IMPEXP may be set on the compilation system command line or (if not set)
 * then in an internal header file when building the library, otherwise (when
 * using the library) it is set here.
 */
#ifndef PNG_IMPEXP
#  if defined(PNG_USE_DLL) && defined(PNG_DLL_IMPORT)
     /* This forces use of a DLL, disallowing static linking */
#    define PNG_IMPEXP PNG_DLL_IMPORT
#  endif

#  ifndef PNG_IMPEXP
#    define PNG_IMPEXP
#  endif
#endif

/* In 1.5.2 the definition of PNG_FUNCTION has been changed to always treat
 * 'attributes' as a storage class - the attributes go at the start of the
 * function definition, and attributes are always appended regardless of the
 * compiler.  This considerably simplifies these macros but may cause problems
 * if any compilers both need function attributes and fail to handle them as
 * a storage class (this is unlikely.)
 */
#ifndef PNG_FUNCTION
#  define PNG_FUNCTION(type, name, args, attributes) attributes type name args
#endif

#ifndef PNG_EXPORT_TYPE
#  define PNG_EXPORT_TYPE(type) PNG_IMPEXP type
#endif

   /* The ordinal value is only relevant when preprocessing png.h for symbol
    * table entries, so we discard it here.  See the .dfn files in the
    * scripts directory.
    */
#ifndef PNG_EXPORTA

#  define PNG_EXPORTA(ordinal, type, name, args, attributes)\
      PNG_FUNCTION(PNG_EXPORT_TYPE(type),(PNGAPI name),PNGARG(args), \
        extern attributes)
#endif

/* ANSI-C (C90) does not permit a macro to be invoked with an empty argument,
 * so make something non-empty to satisfy the requirement:
 */
#define PNG_EMPTY /*empty list*/

#define PNG_EXPORT(ordinal, type, name, args)\
   PNG_EXPORTA(ordinal, type, name, args, PNG_EMPTY)

/* Use PNG_REMOVED to comment out a removed interface. */
#ifndef PNG_REMOVED
#  define PNG_REMOVED(ordinal, type, name, args, attributes)
#endif

#ifndef PNG_CALLBACK
#  define PNG_CALLBACK(type, name, args) type (PNGCBAPI name) PNGARG(args)
#endif

/* Support for compiler specific function attributes.  These are used
 * so that where compiler support is available incorrect use of API
 * functions in png.h will generate compiler warnings.
 *
 * Added at libpng-1.2.41.
 */

#ifndef PNG_NO_PEDANTIC_WARNINGS
#  ifndef PNG_PEDANTIC_WARNINGS_SUPPORTED
#    define PNG_PEDANTIC_WARNINGS_SUPPORTED
#  endif
#endif

#ifdef PNG_PEDANTIC_WARNINGS_SUPPORTED
  /* Support for compiler specific function attributes.  These are used
   * so that where compiler support is available incorrect use of API
   * functions in png.h will generate compiler warnings.  Added at libpng
   * version 1.2.41.
   */

#  if defined(__GNUC__)
#    ifndef PNG_USE_RESULT
#      define PNG_USE_RESULT __attribute__((__warn_unused_result__))
#    endif
#    ifndef PNG_NORETURN
#      define PNG_NORETURN   __attribute__((__noreturn__))
#    endif
#    if __GNUC__ >= 3
#      ifndef PNG_ALLOCATED
#        define PNG_ALLOCATED  __attribute__((__malloc__))
#      endif
#      ifndef PNG_DEPRECATED
#        define PNG_DEPRECATED __attribute__((__deprecated__))
#      endif
#      ifndef PNG_PRIVATE
#        if 0 /* Doesn't work so we use deprecated instead*/
#          define PNG_PRIVATE \
            __attribute__((warning("This function is not exported by libpng.")))
#        else
#          define PNG_PRIVATE \
            __attribute__((__deprecated__))
#        endif
#      endif
#    endif /* __GNUC__ >= 3 */
#  endif /* __GNUC__ */

#  if defined(_MSC_VER)  && (_MSC_VER >= 1300)
#    ifndef PNG_USE_RESULT
#      define PNG_USE_RESULT /* not supported */
#    endif
#    ifndef PNG_NORETURN
#      define PNG_NORETURN __declspec(noreturn)
#    endif
#    ifndef PNG_ALLOCATED
#      if (_MSC_VER >= 1400)
#        define PNG_ALLOCATED __declspec(restrict)
#      endif
#    endif
#    ifndef PNG_DEPRECATED
#      define PNG_DEPRECATED __declspec(deprecated)
#    endif
#    ifndef PNG_PRIVATE
#      define PNG_PRIVATE __declspec(deprecated)
#    endif
#  endif /* _MSC_VER */
#endif /* PNG_PEDANTIC_WARNINGS */

#ifndef PNG_DEPRECATED
#  define PNG_DEPRECATED  /* Use of this function is deprecated */
#endif
#ifndef PNG_USE_RESULT
#  define PNG_USE_RESULT  /* The result of this function must be checked */
#endif
#ifndef PNG_NORETURN
#  define PNG_NORETURN    /* This function does not return */
#endif
#ifndef PNG_ALLOCATED
#  define PNG_ALLOCATED   /* The result of the function is new memory */
#endif
#ifndef PNG_PRIVATE
#  define PNG_PRIVATE     /* This is a private libpng function */
#endif
#ifndef PNG_FP_EXPORT     /* A floating point API. */
#  ifdef PNG_FLOATING_POINT_SUPPORTED
#     define PNG_FP_EXPORT(ordinal, type, name, args)\
         PNG_EXPORT(ordinal, type, name, args);
#  else                   /* No floating point APIs */
#     define PNG_FP_EXPORT(ordinal, type, name, args)
#  endif
#endif
#ifndef PNG_FIXED_EXPORT  /* A fixed point API. */
#  ifdef PNG_FIXED_POINT_SUPPORTED
#     define PNG_FIXED_EXPORT(ordinal, type, name, args)\
         PNG_EXPORT(ordinal, type, name, args);
#  else                   /* No fixed point APIs */
#     define PNG_FIXED_EXPORT(ordinal, type, name, args)
#  endif
#endif

/* The following uses const char * instead of char * for error
 * and warning message functions, so some compilers won't complain.
 * If you do not want to use const, define PNG_NO_CONST here.
 *
 * This should not change how the APIs are called, so it can be done
 * on a per-file basis in the application.
 */
#ifndef PNG_CONST
#  ifndef PNG_NO_CONST
#    define PNG_CONST const
#  else
#    define PNG_CONST
#  endif
#endif

/* Some typedefs to get us started.  These should be safe on most of the
 * common platforms.  The typedefs should be at least as large as the
 * numbers suggest (a png_uint_32 must be at least 32 bits long), but they
 * don't have to be exactly that size.  Some compilers dislike passing
 * unsigned shorts as function parameters, so you may be better off using
 * unsigned int for png_uint_16.
 */

#if defined(INT_MAX) && (INT_MAX > 0x7ffffffeL)
typedef unsigned int png_uint_32;
typedef int png_int_32;
#else
typedef unsigned long png_uint_32;
typedef long png_int_32;
#endif
typedef unsigned short png_uint_16;
typedef short png_int_16;
typedef unsigned char png_byte;

#ifdef PNG_NO_SIZE_T
typedef unsigned int png_size_t;
#else
typedef size_t png_size_t;
#endif
#define png_sizeof(x) (sizeof (x))

/* The following is needed for medium model support.  It cannot be in the
 * pngpriv.h header.  Needs modification for other compilers besides
 * MSC.  Model independent support declares all arrays and pointers to be
 * large using the far keyword.  The zlib version used must also support
 * model independent data.  As of version zlib 1.0.4, the necessary changes
 * have been made in zlib.  The USE_FAR_KEYWORD define triggers other
 * changes that are needed. (Tim Wegner)
 */

/* Separate compiler dependencies (problem here is that zlib.h always
 * defines FAR. (SJT)
 */
#ifdef __BORLANDC__
#  if defined(__LARGE__) || defined(__HUGE__) || defined(__COMPACT__)
#    define LDATA 1
#  else
#    define LDATA 0
#  endif
  /* GRR:  why is Cygwin in here?  Cygwin is not Borland C... */
#  if !defined(__WIN32__) && !defined(__FLAT__) && !defined(__CYGWIN__)
#    define PNG_MAX_MALLOC_64K /* only used in build */
#    if (LDATA != 1)
#      ifndef FAR
#        define FAR __far
#      endif
#      define USE_FAR_KEYWORD
#    endif   /* LDATA != 1 */
         /* Possibly useful for moving data out of default segment.
          * Uncomment it if you want. Could also define FARDATA as
          * const if your compiler supports it. (SJT)
#        define FARDATA FAR
          */
#  endif  /* __WIN32__, __FLAT__, __CYGWIN__ */
#endif   /* __BORLANDC__ */


/* Suggest testing for specific compiler first before testing for
 * FAR.  The Watcom compiler defines both __MEDIUM__ and M_I86MM,
 * making reliance oncertain keywords suspect. (SJT)
 */

/* MSC Medium model */
#ifdef FAR
#  ifdef M_I86MM
#    define USE_FAR_KEYWORD
#    define FARDATA FAR
#    include <dos.h>
#  endif
#endif

/* SJT: default case */
#ifndef FAR
#  define FAR
#endif

/* At this point FAR is always defined */
#ifndef FARDATA
#  define FARDATA
#endif

/* Typedef for floating-point numbers that are converted
 * to fixed-point with a multiple of 100,000, e.g., gamma
 */
typedef png_int_32 png_fixed_point;

/* Add typedefs for pointers */
typedef void                      FAR * png_voidp;
typedef PNG_CONST void            FAR * png_const_voidp;
typedef png_byte                  FAR * png_bytep;
typedef PNG_CONST png_byte        FAR * png_const_bytep;
typedef png_uint_32               FAR * png_uint_32p;
typedef PNG_CONST png_uint_32     FAR * png_const_uint_32p;
typedef png_int_32                FAR * png_int_32p;
typedef PNG_CONST png_int_32      FAR * png_const_int_32p;
typedef png_uint_16               FAR * png_uint_16p;
typedef PNG_CONST png_uint_16     FAR * png_const_uint_16p;
typedef png_int_16                FAR * png_int_16p;
typedef PNG_CONST png_int_16      FAR * png_const_int_16p;
typedef char                      FAR * png_charp;
typedef PNG_CONST char            FAR * png_const_charp;
typedef png_fixed_point           FAR * png_fixed_point_p;
typedef PNG_CONST png_fixed_point FAR * png_const_fixed_point_p;
typedef png_size_t                FAR * png_size_tp;
typedef PNG_CONST png_size_t      FAR * png_const_size_tp;

#ifdef PNG_STDIO_SUPPORTED
typedef FILE            * png_FILE_p;
#endif

#ifdef PNG_FLOATING_POINT_SUPPORTED
typedef double           FAR * png_doublep;
typedef PNG_CONST double FAR * png_const_doublep;
#endif

/* Pointers to pointers; i.e. arrays */
typedef png_byte        FAR * FAR * png_bytepp;
typedef png_uint_32     FAR * FAR * png_uint_32pp;
typedef png_int_32      FAR * FAR * png_int_32pp;
typedef png_uint_16     FAR * FAR * png_uint_16pp;
typedef png_int_16      FAR * FAR * png_int_16pp;
typedef PNG_CONST char  FAR * FAR * png_const_charpp;
typedef char            FAR * FAR * png_charpp;
typedef png_fixed_point FAR * FAR * png_fixed_point_pp;
#ifdef PNG_FLOATING_POINT_SUPPORTED
typedef double          FAR * FAR * png_doublepp;
#endif

/* Pointers to pointers to pointers; i.e., pointer to array */
typedef char            FAR * FAR * FAR * png_charppp;

/* png_alloc_size_t is guaranteed to be no smaller than png_size_t,
 * and no smaller than png_uint_32.  Casts from png_size_t or png_uint_32
 * to png_alloc_size_t are not necessary; in fact, it is recommended
 * not to use them at all so that the compiler can complain when something
 * turns out to be problematic.
 * Casts in the other direction (from png_alloc_size_t to png_size_t or
 * png_uint_32) should be explicitly applied; however, we do not expect
 * to encounter practical situations that require such conversions.
 */
#if defined(__TURBOC__) && !defined(__FLAT__)
   typedef unsigned long png_alloc_size_t;
#else
#  if defined(_MSC_VER) && defined(MAXSEG_64K)
     typedef unsigned long    png_alloc_size_t;
#  else
     /* This is an attempt to detect an old Windows system where (int) is
      * actually 16 bits, in that case png_malloc must have an argument with a
      * bigger size to accomodate the requirements of the library.
      */
#    if (defined(_Windows) || defined(_WINDOWS) || defined(_WINDOWS_)) && \
        (!defined(INT_MAX) || INT_MAX <= 0x7ffffffeL)
       typedef DWORD         png_alloc_size_t;
#    else
       typedef png_size_t    png_alloc_size_t;
#    endif
#  endif
#endif

#endif /* PNGCONF_H */
