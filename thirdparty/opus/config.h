/* Opus configuration header */
/* Based on the output of libopus configure script */

/* Define to 1 if you have the <dlfcn.h> header file. */
#define HAVE_DLFCN_H 1

/* Define to 1 if you have the <inttypes.h> header file. */
#define HAVE_INTTYPES_H 1

#if (!defined( _MSC_VER ) || ( _MSC_VER >= 1800 ))

/* Define to 1 if you have the `lrint' function. */
#define HAVE_LRINT 1

/* Define to 1 if you have the `lrintf' function. */
#define HAVE_LRINTF 1

#endif

/* Define to 1 if you have the <memory.h> header file. */
#define HAVE_MEMORY_H 1

/* Define to 1 if you have the <stdint.h> header file. */
#define HAVE_STDINT_H 1

/* Define to 1 if you have the <stdlib.h> header file. */
#define HAVE_STDLIB_H 1

/* Define to 1 if you have the <strings.h> header file. */
#define HAVE_STRINGS_H 1

/* Define to 1 if you have the <string.h> header file. */
#define HAVE_STRING_H 1

/* Define to 1 if you have the <sys/stat.h> header file. */
#define HAVE_SYS_STAT_H 1

/* Define to 1 if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* Define to 1 if you have the <unistd.h> header file. */
#define HAVE_UNISTD_H 1

/* Define to the sub-directory in which libtool stores uninstalled libraries.
   */
#define LT_OBJDIR ".libs/"

#ifdef OPUS_ARM_OPT
/* Make use of ARM asm optimization */
#define OPUS_ARM_ASM 1

/* Use generic ARMv4 inline asm optimizations */
#define OPUS_ARM_INLINE_ASM 1

/* Use ARMv5E inline asm optimizations */
#define OPUS_ARM_INLINE_EDSP 1

/* Use ARMv6 inline asm optimizations */
#define OPUS_ARM_INLINE_MEDIA 1

/* Use ARM NEON inline asm optimizations */
#define OPUS_ARM_INLINE_NEON 1

/* Define if assembler supports EDSP instructions */
#define OPUS_ARM_MAY_HAVE_EDSP 1

/* Define if assembler supports ARMv6 media instructions */
#define OPUS_ARM_MAY_HAVE_MEDIA 1

/* Define if compiler supports NEON instructions */
#define OPUS_ARM_MAY_HAVE_NEON 1
#endif // OPUS_ARM_OPT

#ifdef OPUS_ARM64_OPT
/* Make use of ARM asm optimization */
#define OPUS_ARM_ASM 1

/* Use ARMv6 inline asm optimizations */
#define OPUS_ARM_INLINE_MEDIA 1 // work

/* Use ARM NEON inline asm optimizations */
#define OPUS_ARM_INLINE_NEON 1 // work

/* Define if assembler supports EDSP instructions */
#define OPUS_ARM_MAY_HAVE_EDSP 1 // work

/* Define if assembler supports ARMv6 media instructions */
#define OPUS_ARM_MAY_HAVE_MEDIA 1 // work

/* Define if compiler supports NEON instructions */
#define OPUS_ARM_MAY_HAVE_NEON 1

#endif // OPUS_ARM64_OPT

/* This is a build of OPUS */
#define OPUS_BUILD /**/

#ifndef WIN32
	/* Use C99 variable-size arrays */
	#define VAR_ARRAYS 1
#else
	/* Fixes VS 2013 compile error */
	#define USE_ALLOCA 1
#endif

#ifndef OPUS_FIXED_POINT
#define FLOAT_APPROX 1
#endif


/* Define to `__inline__' or `__inline' if that's what the C compiler
   calls it, or to nothing if 'inline' is not supported under any name.  */
#ifndef __cplusplus
/* #undef inline */
#endif

/* Define to the equivalent of the C99 'restrict' keyword, or to
   nothing if this is not supported.  Do not define if restrict is
   supported directly.  */
#if (!defined( _MSC_VER ) || ( _MSC_VER >= 1800 ))
#define restrict __restrict
#else
#undef restrict
#endif
/* Work around a bug in Sun C++: it does not support _Restrict or
   __restrict__, even though the corresponding Sun C compiler ends up with
   "#define restrict _Restrict" or "#define restrict __restrict__" in the
   previous line.  Perhaps some future version of Sun C++ will work with
   restrict; if so, hopefully it defines __RESTRICT like Sun C does.  */
#if defined __SUNPRO_CC && !defined __RESTRICT
# define _Restrict
# define __restrict__
#endif
