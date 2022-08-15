/*===-- include/Support/DataTypes.h - Define fixed size types -----*- C -*-===*\
|*                                                                            *|
|*                     The LLVM Compiler Infrastructure                       *|
|*                                                                            *|
|* This file is distributed under the University of Illinois Open Source      *|
|* License. See LICENSE.TXT for details.                                      *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This file contains definitions to figure out the size of _HOST_ data types.*|
|* This file is important because different host OS's define different macros,*|
|* which makes portability tough.  This file exports the following            *|
|* definitions:                                                               *|
|*                                                                            *|
|*   [u]int(32|64)_t : typedefs for signed and unsigned 32/64 bit system types*|
|*   [U]INT(8|16|32|64)_(MIN|MAX) : Constants for the min and max values.     *|
|*                                                                            *|
|* No library is required when using these functions.                         *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*/

/* Please leave this file C-compatible. */

/* Please keep this file in sync with DataTypes.h.in */

#ifndef SUPPORT_DATATYPES_H
#define SUPPORT_DATATYPES_H

#cmakedefine HAVE_INTTYPES_H ${HAVE_INTTYPES_H}
#cmakedefine HAVE_STDINT_H ${HAVE_STDINT_H}
#cmakedefine HAVE_UINT64_T ${HAVE_UINT64_T}
#cmakedefine HAVE_U_INT64_T ${HAVE_U_INT64_T}

#ifdef __cplusplus
#include <cmath>
#else
#include <math.h>
#endif

#ifdef HAVE_INTTYPES_H
#include <inttypes.h>
#endif

#ifdef HAVE_STDINT_H
#include <stdint.h>
#else
#error "Compiler must provide an implementation of stdint.h"
#endif

#ifndef _MSC_VER

/* Note that this header's correct operation depends on __STDC_LIMIT_MACROS
   being defined.  We would define it here, but in order to prevent Bad Things
   happening when system headers or C++ STL headers include stdint.h before we
   define it here, we define it on the g++ command line (in Makefile.rules). */
#if !defined(__STDC_LIMIT_MACROS)
# error "Must #define __STDC_LIMIT_MACROS before #including Support/DataTypes.h"
#endif

#if !defined(__STDC_CONSTANT_MACROS)
# error "Must #define __STDC_CONSTANT_MACROS before " \
        "#including Support/DataTypes.h"
#endif

/* Note that <inttypes.h> includes <stdint.h>, if this is a C99 system. */
#include <sys/types.h>

#ifdef _AIX
#include "llvm/Support/AIXDataTypesFix.h"
#endif

/* Handle incorrect definition of uint64_t as u_int64_t */
#ifndef HAVE_UINT64_T
#ifdef HAVE_U_INT64_T
typedef u_int64_t uint64_t;
#else
# error "Don't have a definition for uint64_t on this platform"
#endif
#endif

#else /* _MSC_VER */
#include <stdlib.h>
#include <stddef.h>
#include <sys/types.h>
#ifdef __cplusplus
#include <cmath>
#else
#include <math.h>
#endif

#if defined(_WIN64)
typedef signed __int64 ssize_t;
#else
typedef signed int ssize_t;
#endif /* _WIN64 */

#ifndef HAVE_INTTYPES_H
#define PRId64 "I64d"
#define PRIi64 "I64i"
#define PRIo64 "I64o"
#define PRIu64 "I64u"
#define PRIx64 "I64x"
#define PRIX64 "I64X"

#define PRId32 "d"
#define PRIi32 "i"
#define PRIo32 "o"
#define PRIu32 "u"
#define PRIx32 "x"
#define PRIX32 "X"
#endif /* HAVE_INTTYPES_H */

#endif /* _MSC_VER */

/* Set defaults for constants which we cannot find. */
#if !defined(INT64_MAX)
# define INT64_MAX 9223372036854775807LL
#endif
#if !defined(INT64_MIN)
# define INT64_MIN ((-INT64_MAX)-1)
#endif
#if !defined(UINT64_MAX)
# define UINT64_MAX 0xffffffffffffffffULL
#endif

#ifndef HUGE_VALF
#define HUGE_VALF (float)HUGE_VAL
#endif

#endif  /* SUPPORT_DATATYPES_H */
