/* Opus configuration header */
/* Based on the output of libopus configure script (https://github.com/xiph/opus/blob/main/configure.ac) */

/* Define to 1 if you have the <dlfcn.h> header file. */
#define HAVE_DLFCN_H 1

/* Define to 1 if you have the <inttypes.h> header file. */
#define HAVE_INTTYPES_H 1

#if (!defined(_MSC_VER) || (_MSC_VER >= 1800))
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

#ifdef OPUS_ARM32_OPT
#endif // OPUS_ARM32_OPT

#ifdef OPUS_ARM64_OPT
/* Use NEON optimizations */
/* Supported by all ARM64 devices by requirement */
#define OPUS_ARM_MAY_HAVE_NEON 1
#define OPUS_ARM_PRESUME_NEON 1

/* Use NEON intrinsics optimizations */
/* Supported by all ARM64 devices by requirement */
#define OPUS_ARM_MAY_HAVE_NEON_INTR 1
#define OPUS_ARM_PRESUME_NEON_INTR 1

/* Use 64-bit NEON intrinsics optimizations */
/* Supported by all ARM64 devices by requirement */
#define OPUS_ARM_MAY_HAVE_AARCH64_NEON_INTR 1
#define OPUS_ARM_PRESUME_AARCH64_NEON_INTR 1
#endif // OPUS_ARM64_OPT

#ifdef OPUS_X32_OPT
/* Use SSE SIMD optimizations */
/* Supported by Godot's minimum system requirements for x32 */
#define OPUS_X86_MAY_HAVE_SSE 1
#define OPUS_X86_PRESUME_SSE 1

/* Use SSE2 SIMD optimizations */
/* Supported by Godot's minimum system requirements for x32 */
#define OPUS_X86_MAY_HAVE_SSE2 1
#define OPUS_X86_PRESUME_SSE2 1

/* Use SSE4.1 SIMD optimizations */
/* Not enabled by Godot for x32: https://github.com/godotengine/godot/blob/master/SConstruct#L798 */
//#define OPUS_X86_MAY_HAVE_SSE4_1 1
//#define OPUS_X86_PRESUME_SSE4_1 1

/* Use AVX2 SIMD optimizations */
/* Not enabled by Godot for x32: https://github.com/godotengine/godot/blob/master/SConstruct#L798 */
//#define OPUS_X86_MAY_HAVE_AVX2 1
//#define OPUS_X86_PRESUME_AVX2 1
#endif // OPUS_X32_OPT

#ifdef OPUS_X64_OPT
/* Use SSE SIMD optimizations */
/* Supported by Godot's minimum system requirements for x64 */
#define OPUS_X86_MAY_HAVE_SSE 1
#define OPUS_X86_PRESUME_SSE 1

/* Use SSE2 SIMD optimizations */
/* Supported by Godot's minimum system requirements for x64 */
#define OPUS_X86_MAY_HAVE_SSE2 1
#define OPUS_X86_PRESUME_SSE2 1

/* Use SSE4.1 SIMD optimizations */
/* Supported by Godot's minimum system requirements for x64 */
#define OPUS_X86_MAY_HAVE_SSE4_1 1
#define OPUS_X86_PRESUME_SSE4_1 1

/* Use AVX2 SIMD optimizations */
/* Not enabled by Godot for x64: https://github.com/godotengine/godot/blob/master/SConstruct#L798 */
//#define OPUS_X86_MAY_HAVE_AVX2 1
//#define OPUS_X86_PRESUME_AVX2 1
#endif // OPUS_X64_OPT

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
#if (!defined(_MSC_VER) || (_MSC_VER >= 1800))
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
#define _Restrict
#define __restrict__
#endif