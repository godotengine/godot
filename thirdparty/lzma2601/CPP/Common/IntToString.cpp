// Common/IntToString.cpp

#include "StdAfx.h"

#include "../../C/CpuArch.h"

#include "IntToString.h"

#define CONVERT_INT_TO_STR(charType, tempSize) \
  if (val < 10) \
    *s++ = (charType)('0' + (unsigned)val); \
  else { \
    Byte temp[tempSize]; \
    size_t i = 0; \
    do { \
      temp[++i] = (Byte)('0' + (unsigned)(val % 10)); \
      val /= 10; } \
    while (val >= 10); \
    *s++ = (charType)('0' + (unsigned)val); \
    do { *s++ = (charType)temp[i]; } \
    while (--i); \
  } \
  *s = 0; \
  return s;

char * ConvertUInt32ToString(UInt32 val, char *s) throw()
{
  CONVERT_INT_TO_STR(char, 16)
}

char * ConvertUInt64ToString(UInt64 val, char *s) throw()
{
  if (val <= (UInt32)0xFFFFFFFF)
    return ConvertUInt32ToString((UInt32)val, s);
  CONVERT_INT_TO_STR(char, 24)
}

wchar_t * ConvertUInt32ToString(UInt32 val, wchar_t *s) throw()
{
  CONVERT_INT_TO_STR(wchar_t, 16)
}

wchar_t * ConvertUInt64ToString(UInt64 val, wchar_t *s) throw()
{
  if (val <= (UInt32)0xFFFFFFFF)
    return ConvertUInt32ToString((UInt32)val, s);
  CONVERT_INT_TO_STR(wchar_t, 24)
}

void ConvertInt64ToString(Int64 val, char *s) throw()
{
  if (val < 0)
  {
    *s++ = '-';
    val = -val;
  }
  ConvertUInt64ToString((UInt64)val, s);
}

void ConvertInt64ToString(Int64 val, wchar_t *s) throw()
{
  if (val < 0)
  {
    *s++ = L'-';
    val = -val;
  }
  ConvertUInt64ToString((UInt64)val, s);
}


void ConvertUInt64ToOct(UInt64 val, char *s) throw()
{
  {
    UInt64 v = val;
    do
      s++;
    while (v >>= 3);
  }
  *s = 0;
  do
  {
    const unsigned t = (unsigned)val & 7;
    *--s = (char)('0' + t);
  }
  while (val >>= 3);
}

MY_ALIGN(16) const char k_Hex_Upper[16] =
 { '0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F' };
MY_ALIGN(16) const char k_Hex_Lower[16] =
 { '0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f' };

void ConvertUInt32ToHex(UInt32 val, char *s) throw()
{
  {
    UInt32 v = val;
    do
      s++;
    while (v >>= 4);
  }
  *s = 0;
  do
  {
    const unsigned t = (unsigned)val & 0xF;
    *--s = GET_HEX_CHAR_UPPER(t);
  }
  while (val >>= 4);
}

void ConvertUInt64ToHex(UInt64 val, char *s) throw()
{
  {
    UInt64 v = val;
    do
      s++;
    while (v >>= 4);
  }
  *s = 0;
  do
  {
    const unsigned t = (unsigned)val & 0xF;
    *--s = GET_HEX_CHAR_UPPER(t);
  }
  while (val >>= 4);
}

void ConvertUInt32ToHex8Digits(UInt32 val, char *s) throw()
{
  s[8] = 0;
  int i = 7;
  do
  {
    { const unsigned t = (unsigned)val & 0xF;       s[i--] = GET_HEX_CHAR_UPPER(t); }
    { const unsigned t = (Byte)val >> 4; val >>= 8; s[i--] = GET_HEX_CHAR_UPPER(t); }
  }
  while (i >= 0);
}

/*
void ConvertUInt32ToHex8Digits(UInt32 val, wchar_t *s)
{
  s[8] = 0;
  for (int i = 7; i >= 0; i--)
  {
    const unsigned t = (unsigned)val & 0xF;
    val >>= 4;
    s[i] = GET_HEX_CHAR(t);
  }
}
*/


MY_ALIGN(16) static const Byte k_Guid_Pos[] =
  { 6,4,2,0, 11,9, 16,14, 19,21, 24,26,28,30,32,34 };

char *RawLeGuidToString(const Byte *g, char *s) throw()
{
  s[ 8] = '-';
  s[13] = '-';
  s[18] = '-';
  s[23] = '-';
  s[36] = 0;
  for (unsigned i = 0; i < 16; i++)
  {
    char *s2 = s + k_Guid_Pos[i];
    const unsigned v = g[i];
    s2[0] = GET_HEX_CHAR_UPPER(v >> 4);
    s2[1] = GET_HEX_CHAR_UPPER(v & 0xF);
  }
  return s + 36;
}

char *RawLeGuidToString_Braced(const Byte *g, char *s) throw()
{
  *s++ = '{';
  s = RawLeGuidToString(g, s);
  *s++ = '}';
  *s = 0;
  return s;
}


void ConvertDataToHex_Lower(char *dest, const Byte *src, size_t size) throw()
{
  if (size)
  {
    const Byte *lim = src + size;
    do
    {
      const unsigned b = *src++;
      dest[0] = GET_HEX_CHAR_LOWER(b >> 4);
      dest[1] = GET_HEX_CHAR_LOWER(b & 0xF);
      dest += 2;
    }
    while (src != lim);
  }
  *dest = 0;
}

void ConvertDataToHex_Upper(char *dest, const Byte *src, size_t size) throw()
{
  if (size)
  {
    const Byte *lim = src + size;
    do
    {
      const unsigned b = *src++;
      dest[0] = GET_HEX_CHAR_UPPER(b >> 4);
      dest[1] = GET_HEX_CHAR_UPPER(b & 0xF);
      dest += 2;
    }
    while (src != lim);
  }
  *dest = 0;
}
