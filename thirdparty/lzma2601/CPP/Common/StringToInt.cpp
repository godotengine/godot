// Common/StringToInt.cpp

#include "StdAfx.h"

#include <limits.h>
#if defined(_MSC_VER) && (_MSC_VER >= 1600)
#include <stdint.h> // for WCHAR_MAX in vs2022
#endif

#include "StringToInt.h"

static const UInt32 k_UInt32_max = 0xFFFFFFFF;
static const UInt64 k_UInt64_max = UINT64_CONST(0xFFFFFFFFFFFFFFFF);
// static const UInt64 k_UInt64_max = (UInt64)(Int64)-1;

#define DIGIT_TO_VALUE(charTypeUnsigned, digit)   ((unsigned)(charTypeUnsigned)digit - '0')
// #define DIGIT_TO_VALUE(charTypeUnsigned, digit)   ((unsigned)digit - '0')
// #define DIGIT_TO_VALUE(charTypeUnsigned, digit)   ((unsigned)(digit - '0'))

#define CONVERT_STRING_TO_UINT_FUNC(uintType, charType, charTypeUnsigned) \
uintType ConvertStringTo ## uintType(const charType *s, const charType **end) throw() { \
    if (end) *end = s; \
    uintType res = 0; \
    for (;; s++) { \
      const unsigned v = DIGIT_TO_VALUE(charTypeUnsigned, *s); \
      if (v > 9) { if (end) *end = s; return res; } \
      if (res > (k_ ## uintType ## _max) / 10) return 0; \
      res *= 10; \
      if (res > (k_ ## uintType ## _max) - v) return 0; \
      res += v; }}

// arm-linux-gnueabi GCC compilers give (WCHAR_MAX > UINT_MAX) by some unknown reason
// so we don't use this branch
#if 0 && WCHAR_MAX > UINT_MAX
/*
   if (sizeof(wchar_t) > sizeof(unsigned)
      we must use CONVERT_STRING_TO_UINT_FUNC_SLOW
   But we just stop compiling instead.
   We need some real cases to test this code.
*/
#error Stop_Compiling_WCHAR_MAX_IS_LARGER_THAN_UINT_MAX
#define CONVERT_STRING_TO_UINT_FUNC_SLOW(uintType, charType, charTypeUnsigned) \
uintType ConvertStringTo ## uintType(const charType *s, const charType **end) throw() { \
    if (end) *end = s; \
    uintType res = 0; \
    for (;; s++) { \
      const charTypeUnsigned c = (charTypeUnsigned)*s; \
      if (c < '0' || c > '9') { if (end) *end = s; return res; } \
      if (res > (k_ ## uintType ## _max) / 10) return 0; \
      res *= 10; \
      const unsigned v = (unsigned)(c - '0'); \
      if (res > (k_ ## uintType ## _max) - v) return 0; \
      res += v; }}
#endif


CONVERT_STRING_TO_UINT_FUNC(UInt32, char, Byte)
CONVERT_STRING_TO_UINT_FUNC(UInt32, wchar_t, wchar_t)
CONVERT_STRING_TO_UINT_FUNC(UInt64, char, Byte)
CONVERT_STRING_TO_UINT_FUNC(UInt64, wchar_t, wchar_t)

Int32 ConvertStringToInt32(const wchar_t *s, const wchar_t **end) throw()
{
  if (end)
    *end = s;
  const wchar_t *s2 = s;
  if (*s == '-')
    s2++;
  const wchar_t *end2;
  UInt32 res = ConvertStringToUInt32(s2, &end2);
  if (s2 == end2)
    return 0;
  if (s != s2)
  {
    if (res > (UInt32)1 << (32 - 1))
      return 0;
    res = 0 - res;
  }
  else
  {
    if (res & (UInt32)1 << (32 - 1))
      return 0;
  }
  if (end)
    *end = end2;
  return (Int32)res;
}


#define CONVERT_OCT_STRING_TO_UINT_FUNC(uintType) \
uintType ConvertOctStringTo ## uintType(const char *s, const char **end) throw() \
{ \
  if (end) *end = s; \
  uintType res = 0; \
  for (;; s++) { \
    const unsigned c = (unsigned)(Byte)*s - '0'; \
    if (c > 7) { \
      if (end) \
        *end = s; \
      return res; \
    } \
    if (res & (uintType)7 << (sizeof(uintType) * 8 - 3)) \
      return 0; \
    res <<= 3; \
    res |= c; \
  } \
}

CONVERT_OCT_STRING_TO_UINT_FUNC(UInt32)
CONVERT_OCT_STRING_TO_UINT_FUNC(UInt64)


#define CONVERT_HEX_STRING_TO_UINT_FUNC(uintType) \
uintType ConvertHexStringTo ## uintType(const char *s, const char **end) throw() \
{ \
  if (end) *end = s; \
  uintType res = 0; \
  for (;; s++) { \
    unsigned c = (unsigned)(Byte)*s; \
    Z7_PARSE_HEX_DIGIT(c, { if (end) *end = s;  return res; }) \
    if (res & (uintType)0xF << (sizeof(uintType) * 8 - 4)) \
      return 0; \
    res <<= 4; \
    res |= c; \
  } \
}

CONVERT_HEX_STRING_TO_UINT_FUNC(UInt32)
CONVERT_HEX_STRING_TO_UINT_FUNC(UInt64)

const char *FindNonHexChar(const char *s) throw()
{
  for (;;)
  {
    unsigned c = (Byte)*s++; // pointer can go 1 byte after end
    c -= '0';
    if (c <= 9)
      continue;
    c -= 'A' - '0';
    c &= ~0x20u;
    if (c > 5)
      return s - 1;
  }
}

Byte *ParseHexString(const char *s, Byte *dest) throw()
{
  for (;;)
  {
    unsigned v0 = (Byte)s[0];         Z7_PARSE_HEX_DIGIT(v0, return dest;)
    unsigned v1 = (Byte)s[1]; s += 2; Z7_PARSE_HEX_DIGIT(v1, return dest;)
    *dest++ = (Byte)(v1 | (v0 << 4));
  }
}
