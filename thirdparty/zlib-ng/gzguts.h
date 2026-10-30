#ifndef GZGUTS_H_
#define GZGUTS_H_
/* gzguts.h -- zlib internal header definitions for gz* operations
 * Copyright (C) 2004-2024 Mark Adler
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#ifdef _LARGEFILE64_SOURCE
#  ifndef _LARGEFILE_SOURCE
#    define _LARGEFILE_SOURCE 1
#  endif
#  undef _FILE_OFFSET_BITS
#  undef _TIME_BITS
#endif

#if defined(HAVE_VISIBILITY_INTERNAL)
#  define Z_INTERNAL __attribute__((visibility ("internal")))
#elif defined(HAVE_VISIBILITY_HIDDEN)
#  define Z_INTERNAL __attribute__((visibility ("hidden")))
#else
#  define Z_INTERNAL
#endif

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>
#include <fcntl.h>

#if defined(ZLIB_COMPAT)
#  include "zlib.h"
#else
#  include "zlib-ng.h"
#endif

#ifdef _WIN32
#  include <stddef.h>
#endif

#if defined(_WIN32)
#  include <io.h>
#  define WIDECHAR
#endif

#ifdef WINAPI_FAMILY
#  define open _open
#  define read _read
#  define write _write
#  define close _close
#endif

/* In Win32, vsnprintf is available as the "non-ANSI" _vsnprintf. */
#if !defined(STDC99) && !defined(__CYGWIN__) && !defined(__MINGW__) && defined(_WIN32)
#  if !defined(vsnprintf)
#    if !defined(_MSC_VER) || ( defined(_MSC_VER) && _MSC_VER < 1500 )
#       define vsnprintf _vsnprintf
#    endif
#  endif
#endif

/* unlike snprintf (which is required in C99), _snprintf does not guarantee
   null termination of the result -- however this is only used in gzlib.c
   where the result is assured to fit in the space provided */
#if defined(_MSC_VER) && _MSC_VER < 1900
#  define snprintf _snprintf
#endif

/* get errno and strerror definition */
#ifndef NO_STRERROR
#  include <errno.h>
#  define zstrerror() strerror(errno)
#else
#  define zstrerror() "stdio error (consult errno)"
#endif

/* default memLevel */
#if MAX_MEM_LEVEL >= 8
#  define DEF_MEM_LEVEL 8
#else
#  define DEF_MEM_LEVEL  MAX_MEM_LEVEL
#endif

/* default i/o buffer size -- double this for output when reading (this and
   twice this must be able to fit in an unsigned type) */
#ifndef GZBUFSIZE
#  define GZBUFSIZE 131072
#endif

/* gzip modes, also provide a little integrity check on the passed structure */
#define GZ_NONE 0
#define GZ_READ 7247
#define GZ_WRITE 31153
#define GZ_APPEND 1     /* mode set to GZ_WRITE after the file is opened */

/* values for gz_state how */
#define LOOK 0      /* look for a gzip header */
#define COPY 1      /* copy input directly */
#define GZIP 2      /* decompress a gzip stream */

/* internal gzip file state data structure */
typedef struct {
        /* exposed contents for gzgetc() macro */
    struct gzFile_s x;      /* "x" for exposed */
                            /* x.have: number of bytes available at x.next */
                            /* x.next: next output data to deliver or write */
                            /* x.pos: current position in uncompressed data */
        /* used for both reading and writing */
    int mode;               /* see gzip modes above */
    int fd;                 /* file descriptor */
    char *path;             /* path or fd for error messages */
    unsigned size;          /* buffer size, zero if not allocated yet */
    unsigned want;          /* requested buffer size, default is GZBUFSIZE */
    unsigned char *in;      /* input buffer (double-sized when writing) */
    unsigned char *out;     /* output buffer (double-sized when reading) */
    unsigned char *buffers; /* Pointer to the real input/output buffer allocation */
    int direct;             /* 0 if processing gzip, 1 if transparent */
        /* just for reading */
    int how;                /* 0: get header, 1: copy, 2: decompress */
    z_off64_t start;        /* where the gzip data started, for rewinding */
    int eof;                /* true if end of input file reached */
    int past;               /* true if read requested past end */
        /* just for writing */
    int level;              /* compression level */
    int strategy;           /* compression strategy */
    int reset;              /* true if a reset is pending after a Z_FINISH */
        /* seek request */
    z_off64_t skip;         /* amount to skip (already rewound if backwards) */
    int seek;               /* true if seek request pending */
        /* error information */
    int err;                /* error code */
    char *msg;              /* error message */
        /* zlib inflate or deflate stream */
    PREFIX3(stream) strm;  /* stream structure in-place (not a pointer) */
} gz_state;
typedef gz_state *gz_statep;

/* shared functions */
void Z_INTERNAL PREFIX(gz_error)(gz_state *, int, const char *);
int  Z_INTERNAL gz_buffer_alloc(gz_state *state);
void Z_INTERNAL gz_buffer_free(gz_state *state);
void Z_INTERNAL gz_state_free(gz_state *state);

#ifdef ZLIB_COMPAT
unsigned Z_INTERNAL gz_intmax(void);
#endif
/* GT_OFF(x), where x is an unsigned value, is true if x > maximum z_off64_t
   value -- needed when comparing unsigned to z_off64_t, which is signed
   (possible z_off64_t types off_t, off64_t, and long are all signed) */
#define GT_OFF(x) (sizeof(int) == sizeof(z_off64_t) && (x) > INT_MAX)

#endif /* GZGUTS_H_ */
