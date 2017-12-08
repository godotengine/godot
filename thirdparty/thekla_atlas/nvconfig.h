#ifndef NV_CONFIG
#define NV_CONFIG

#if NV_OS_DARWIN

// Hardcoded.

#define NV_HAVE_UNISTD_H
#define NV_HAVE_STDARG_H
#define NV_HAVE_SIGNAL_H
#define NV_HAVE_EXECINFO_H
//#define NV_HAVE_MALLOC_H

#else

//#define HAVE_UNISTD_H
#define NV_HAVE_STDARG_H
//#define HAVE_SIGNAL_H
//#define HAVE_EXECINFO_H
//#define HAVE_MALLOC_H

#endif

//#define HAVE_OPENMP // Only in MSVC pro edition.

//#cmakedefine HAVE_PNG
//#cmakedefine HAVE_JPEG
//#cmakedefine HAVE_TIFF
//#cmakedefine HAVE_OPENEXR
//#cmakedefine HAVE_FREEIMAGE
#if !NV_OS_IOS
#define NV_HAVE_STBIMAGE
#endif

//#cmakedefine HAVE_MAYA

#endif // NV_CONFIG
