/**
 * This file has no copyright assigned and is placed in the Public Domain.
 * This file is part of the mingw-w64 runtime package.
 */

#ifndef _INTSAFE_H_INCLUDED_
#define _INTSAFE_H_INCLUDED_

#include <winapifamily.h>

#if WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_APP)

#include <wtypesbase.h>
#include <specstrings.h>

#define INTSAFE_E_ARITHMETIC_OVERFLOW ((HRESULT)0x80070216)

#ifndef S_OK
#define S_OK ((HRESULT)0)
#endif

#ifdef __clang__
#if __has_builtin(__builtin_add_overflow)
#define __MINGW_INTSAFE_WORKS
#endif
#elif __GNUC__ >= 5
#define __MINGW_INTSAFE_WORKS
#endif

#ifdef __MINGW_INTSAFE_WORKS

#ifndef __MINGW_INTSAFE_API
#define __MINGW_INTSAFE_API FORCEINLINE
#endif

/** If CHAR is unsigned, use static inline for functions that operate
on chars.  This avoids the risk of linking to the wrong function when
different translation units with different types of chars are linked
together, and code using signed chars will not be affected. */
#ifndef __MINGW_INTSAFE_CHAR_API
#ifdef __CHAR_UNSIGNED__
#define __MINGW_INTSAFE_CHAR_API static inline
#else
#define __MINGW_INTSAFE_CHAR_API __MINGW_INTSAFE_API
#endif
#endif

#define __MINGW_INTSAFE_BODY(operation, x, y) \
{ \
  if (__builtin_##operation##_overflow(x, y, result)) \
  { \
      *result = 0; \
      return INTSAFE_E_ARITHMETIC_OVERFLOW; \
  } \
  return S_OK; \
}

#define __MINGW_INTSAFE_CONV(name, type_src, type_dest) \
    HRESULT name(type_src operand, type_dest * result) \
    __MINGW_INTSAFE_BODY(add, operand, 0)

#define __MINGW_INTSAFE_MATH(name, type, operation) \
    HRESULT name(type x, type y, type * result) \
    __MINGW_INTSAFE_BODY(operation, x, y)

__MINGW_INTSAFE_API __MINGW_INTSAFE_CONV(SizeTToULong, size_t, ULONG)
__MINGW_INTSAFE_API __MINGW_INTSAFE_CONV(SizeTToInt, size_t, INT)

#endif /* __GNUC__ >= 5 */
#endif /* WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_APP) */
#endif /* _INTSAFE_H_INCLUDED_ */
