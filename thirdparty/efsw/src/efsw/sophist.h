/* sophist.h - 0.3 - public domain - Sean Barrett 2010
** Knowledge drawn from Brian Hook's posh.h and http://predef.sourceforge.net
** Sophist provides portable types; you typedef/#define them to your own names
**
** defines:
**   - SOPHIST_endian    - either SOPHIST_little_endian or SOPHIST_big_endian
**   - SOPHIST_has_64    - either 0 or 1; if 0, int64 types aren't defined
**   - SOPHIST_pointer64 - either 0 or 1; if 1, pointer is 64-bit
**
**   - SOPHIST_intptr, SOPHIST_uintptr - integer same size as pointer
**   - SOPHIST_int8,  SOPHIST_uint8,  SOPHIST_int16, SOPHIST_uint16
**   - SOPHIST_int32, SOPHIST_uint32, SOPHIST_int64, SOPHIST_uint64
**   - SOPHIST_int64_constant(number)      - macros for creating 64-bit
**   - SOPHIST_uint64_constant(number)       integer constants
**   - SOPHIST_printf_format64 - string for printf format for int64
*/

#ifndef __INCLUDE_SOPHIST_H__
#define __INCLUDE_SOPHIST_H__

#define SOPHIST_compiletime_assert(name,val) \
            typedef int SOPHIST__assert##name[(val) ? 1 : -1]

/* define a couple synthetic rules to make code more readable */
#if (defined(__sparc__) || defined(__sparc)) && \
    (defined(__arch64__) || defined(__sparcv9) || defined(__sparc_v9__))
  #define SOPHIST_sparc64
#endif

#if (defined(linux) || defined(__linux__)) &&   \
    (defined(__alpha)||defined(__alpha__)||defined(__x86_64__)||defined(_M_X64))
  #define SOPHIST_linux64
#endif

/* basic types */
typedef   signed  char SOPHIST_int8;
typedef unsigned  char SOPHIST_uint8;

typedef   signed short SOPHIST_int16;
typedef unsigned short SOPHIST_uint16;

#ifdef __palmos__
  typedef   signed long SOPHIST_int32;
  typedef unsigned long SOPHIST_uint32;
#else
  typedef   signed  int SOPHIST_int32;
  typedef unsigned  int SOPHIST_uint32;
#endif

#ifndef SOPHIST_NO_64
   #if defined(_MSC_VER) || defined(__WATCOMC__) || defined(__BORLANDC__)     \
       || (defined(__alpha) && defined(__DECC))

     typedef   signed __int64 SOPHIST_int64;
     typedef unsigned __int64 SOPHIST_uint64;
     #define SOPHIST_has_64              1
     #define SOPHIST_int64_constant(x)   (x##i64)
     #define SOPHIST_uint64_constant(x)  (x##ui64)
     #define SOPHIST_printf_format64     "I64"

   #elif defined(__LP64__) || defined(__powerpc64__) || defined(SOPHIST_sparc64)

     typedef   signed long    SOPHIST_int64;
     typedef unsigned long    SOPHIST_uint64;

     #define SOPHIST_has_64              1
     #define SOPHIST_int64_constant(x)   ((SOPHIST_int64) x)
     #define SOPHIST_uint64_constant(x)  ((SOPHIST_uint64) x)
     #define SOPHIST_printf_format64     "l"

   #elif defined(_LONG_LONG) || defined(__SUNPRO_C) || defined(__SUNPRO_CC)  \
       || defined(__GNUC__)  || defined(__MWERKS__) || defined(__APPLE_CC__) \
       || defined(sgi)       || defined (__sgi)     || defined(__sgi__)      \
       || defined(_CRAYC)

     typedef   signed long long SOPHIST_int64;
     typedef unsigned long long SOPHIST_uint64;

     #define SOPHIST_has_64              1
     #define SOPHIST_int64_constant(x)   (x##LL)
     #define SOPHIST_uint64_constant(x)  (x##ULL)
     #define SOPHIST_printf_format64     "ll"
   #endif
#endif

#ifndef SOPHIST_has_64
#define SOPHIST_has_64 0
#endif

SOPHIST_compiletime_assert( int8 , sizeof(SOPHIST_int8 ) == 1);
SOPHIST_compiletime_assert(uint16, sizeof(SOPHIST_int16) == 2);
SOPHIST_compiletime_assert( int32, sizeof(SOPHIST_int32 ) == 4);
SOPHIST_compiletime_assert(uint32, sizeof(SOPHIST_uint32) == 4);

#if SOPHIST_has_64
  SOPHIST_compiletime_assert( int64, sizeof(SOPHIST_int64 ) == 8);
  SOPHIST_compiletime_assert(uint64, sizeof(SOPHIST_uint64) == 8);
#endif

/* determine whether pointers are 64-bit */

#if    defined(SOPHIST_linux64) || defined(SOPHIST_sparc64)      \
    || defined(__osf__) || (defined(_WIN64) && !defined(_XBOX))  \
    || defined(__64BIT__)                                        \
    || defined(__LP64)  || defined(__LP64__) || defined(_LP64)   \
    || defined(_ADDR64) || defined(_CRAYC)                       \

  #define SOPHIST_pointer64 1

  SOPHIST_compiletime_assert(pointer64, sizeof(void*) == 8);  

  typedef SOPHIST_int64  SOPHIST_intptr;
  typedef SOPHIST_uint64 SOPHIST_uintptr;
#else

  #define SOPHIST_pointer64 0

  SOPHIST_compiletime_assert(pointer64, sizeof(void*) <= 4);  
  
  /* do we care about pointers that are only 16-bit? */
  typedef SOPHIST_int32  SOPHIST_intptr;
  typedef SOPHIST_uint32 SOPHIST_uintptr;

#endif 

SOPHIST_compiletime_assert(intptr, sizeof(SOPHIST_intptr) == sizeof(char *));

/* enumerate known little endian cases; fallback to big-endian */

#define SOPHIST_little_endian 1
#define SOPHIST_big_endian 2

#if    defined(__386__) || defined(i386)    || defined(__i386__)  \
    || defined(__X86)   || defined(_M_IX86)                       \
    || defined(_M_X64)  || defined(__x86_64__)                    \
    || defined(alpha)   || defined(__alpha) || defined(__alpha__) \
    || defined(_M_ALPHA)                                          \
    || defined(ARM)     || defined(_ARM)    || defined(__arm__)   \
    || defined(WIN32)   || defined(_WIN32)  || defined(__WIN32__) \
    || defined(_WIN32_WCE) || defined(__NT__)                     \
    || defined(__MIPSEL__)
  #define SOPHIST_endian  SOPHIST_little_endian
#else
  #define SOPHIST_endian  SOPHIST_big_endian
#endif

#endif /* __INCLUDE_SOPHIST_H__ */
