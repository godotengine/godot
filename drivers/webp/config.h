/* src/webp/config.h.in.  Generated from configure.ac by autoheader.  */

/* Define if building universal (internal helper macro) */
#undef AC_APPLE_UNIVERSAL_BUILD

/* Set to 1 if __builtin_bswap16 is available */
#undef HAVE_BUILTIN_BSWAP16

/* Set to 1 if __builtin_bswap32 is available */
#undef HAVE_BUILTIN_BSWAP32

/* Set to 1 if __builtin_bswap64 is available */
#undef HAVE_BUILTIN_BSWAP64

/* Define to 1 if you have the <dlfcn.h> header file. */
#undef HAVE_DLFCN_H

/* Define to 1 if you have the <GLUT/glut.h> header file. */
#undef HAVE_GLUT_GLUT_H

/* Define to 1 if you have the <GL/glut.h> header file. */
#undef HAVE_GL_GLUT_H

/* Define to 1 if you have the <inttypes.h> header file. */
#undef HAVE_INTTYPES_H

/* Define to 1 if you have the <memory.h> header file. */
#undef HAVE_MEMORY_H

/* Define to 1 if you have the <OpenGL/glut.h> header file. */
#undef HAVE_OPENGL_GLUT_H

/* Have PTHREAD_PRIO_INHERIT. */
#undef HAVE_PTHREAD_PRIO_INHERIT

/* Define to 1 if you have the <shlwapi.h> header file. */
#undef HAVE_SHLWAPI_H

/* Define to 1 if you have the <stdint.h> header file. */
#undef HAVE_STDINT_H

/* Define to 1 if you have the <stdlib.h> header file. */
#undef HAVE_STDLIB_H

/* Define to 1 if you have the <strings.h> header file. */
#undef HAVE_STRINGS_H

/* Define to 1 if you have the <string.h> header file. */
#undef HAVE_STRING_H

/* Define to 1 if you have the <sys/stat.h> header file. */
#undef HAVE_SYS_STAT_H

/* Define to 1 if you have the <sys/types.h> header file. */
#undef HAVE_SYS_TYPES_H

/* Define to 1 if you have the <unistd.h> header file. */
#undef HAVE_UNISTD_H

/* Define to 1 if you have the <wincodec.h> header file. */
#undef HAVE_WINCODEC_H

/* Define to 1 if you have the <windows.h> header file. */
#undef HAVE_WINDOWS_H

/* Define to the sub-directory in which libtool stores uninstalled libraries.
   */
#undef LT_OBJDIR

/* Name of package */
#undef PACKAGE

/* Define to the address where bug reports for this package should be sent. */
#undef PACKAGE_BUGREPORT

/* Define to the full name of this package. */
#undef PACKAGE_NAME

/* Define to the full name and version of this package. */
#undef PACKAGE_STRING

/* Define to the one symbol short name of this package. */
#undef PACKAGE_TARNAME

/* Define to the home page for this package. */
#undef PACKAGE_URL

/* Define to the version of this package. */
#undef PACKAGE_VERSION

/* Define to necessary symbol if this constant uses a non-standard name on
   your system. */
#undef PTHREAD_CREATE_JOINABLE

/* Define to 1 if you have the ANSI C header files. */
#undef STDC_HEADERS

/* Version number of package */
#undef VERSION

/* Enable experimental code */
#undef WEBP_EXPERIMENTAL_FEATURES

/* Define to 1 to force aligned memory operations */
#undef WEBP_FORCE_ALIGNED

/* Set to 1 if AVX2 is supported */
#undef WEBP_HAVE_AVX2

/* Set to 1 if GIF library is installed */
#undef WEBP_HAVE_GIF

/* Set to 1 if OpenGL is supported */
#undef WEBP_HAVE_GL

/* Set to 1 if JPEG library is installed */
#undef WEBP_HAVE_JPEG

/* Set to 1 if NEON is supported */
#undef WEBP_HAVE_NEON

/* Set to 1 if runtime detection of NEON is enabled */
#undef WEBP_HAVE_NEON_RTCD

/* Set to 1 if PNG library is installed */
#undef WEBP_HAVE_PNG

/* Set to 1 if SSE2 is supported */
#undef WEBP_HAVE_SSE2

/* Set to 1 if SSE4.1 is supported */
#undef WEBP_HAVE_SSE41

/* Set to 1 if TIFF library is installed */
#undef WEBP_HAVE_TIFF

/* Undefine this to disable thread support. */
#undef WEBP_USE_THREAD

/* Define WORDS_BIGENDIAN to 1 if your processor stores words with the most
   significant byte first (like Motorola and SPARC, unlike Intel). */
#if defined AC_APPLE_UNIVERSAL_BUILD
# if defined __BIG_ENDIAN__
#  define WORDS_BIGENDIAN 1
# endif
#else
# ifndef WORDS_BIGENDIAN
#  undef WORDS_BIGENDIAN
# endif
#endif
