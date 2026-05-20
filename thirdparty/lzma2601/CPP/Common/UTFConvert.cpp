// UTFConvert.cpp

#include "StdAfx.h"

// #include <stdio.h>

#include "MyTypes.h"
#include "UTFConvert.h"


#ifndef Z7_WCHART_IS_16BIT
#ifndef __APPLE__
  // we define it if the system supports files with non-utf8 symbols:
  #define MY_UTF8_RAW_NON_UTF8_SUPPORTED
#endif
#endif

/*
  MY_UTF8_START(n) - is a base value for start byte (head), if there are (n) additional bytes after start byte
  
  n : MY_UTF8_START(n) : Bits of code point

  0 : 0x80 :    : unused
  1 : 0xC0 : 11 :
  2 : 0xE0 : 16 : Basic Multilingual Plane
  3 : 0xF0 : 21 : Unicode space
  4 : 0xF8 : 26 :
  5 : 0xFC : 31 : UCS-4 : wcstombs() in ubuntu is limited to that value
  6 : 0xFE : 36 : We can use it, if we want to encode any 32-bit value
  7 : 0xFF :
*/

#define MY_UTF8_START(n) (0x100 - (1 << (7 - (n))))

#define MY_UTF8_HEAD_PARSE2(n) \
    if (c < MY_UTF8_START((n) + 1)) \
    { numBytes = (n); val -= MY_UTF8_START(n); }

#ifndef Z7_WCHART_IS_16BIT

/*
   if (wchar_t is 32-bit), we can support large points in long UTF-8 sequence,
   when we convert wchar_t strings to UTF-8:
     (_UTF8_NUM_TAIL_BYTES_MAX == 3) : (21-bits points) - Unicode
     (_UTF8_NUM_TAIL_BYTES_MAX == 5) : (31-bits points) - UCS-4
     (_UTF8_NUM_TAIL_BYTES_MAX == 6) : (36-bit hack)
*/

#define MY_UTF8_NUM_TAIL_BYTES_MAX 5
#endif

/*
#define MY_UTF8_HEAD_PARSE \
    UInt32 val = c; \
         MY_UTF8_HEAD_PARSE2(1) \
    else MY_UTF8_HEAD_PARSE2(2) \
    else MY_UTF8_HEAD_PARSE2(3) \
    else MY_UTF8_HEAD_PARSE2(4) \
    else MY_UTF8_HEAD_PARSE2(5) \
  #if MY_UTF8_NUM_TAIL_BYTES_MAX >= 6
    else MY_UTF8_HEAD_PARSE2(6)
  #endif
*/

#define MY_UTF8_HEAD_PARSE_MAX_3_BYTES \
    UInt32 val = c; \
         MY_UTF8_HEAD_PARSE2(1) \
    else MY_UTF8_HEAD_PARSE2(2) \
    else { numBytes = 3; val -= MY_UTF8_START(3); }


#define MY_UTF8_RANGE(n) (((UInt32)1) << ((n) * 5 + 6))


#define START_POINT_FOR_SURROGATE 0x10000


/* we use 128 bytes block in 16-bit BMP-PLANE to encode non-UTF-8 Escapes
   Also we can use additional HIGH-PLANE (we use 21-bit points above 0x1f0000)
   to simplify internal intermediate conversion in Linux:
   RAW-UTF-8 <-> internal wchar_t utf-16 strings <-> RAW-UTF-UTF-8
*/
 

#if defined(Z7_WCHART_IS_16BIT)

#define UTF_ESCAPE_PLANE 0

#else

/*
we can place 128 ESCAPE chars to
   ef 80 -    ee be 80 (3-bytes utf-8) : similar to WSL
   ef ff -    ee bf bf

1f ef 80 - f7 be be 80 (4-bytes utf-8) : last  4-bytes utf-8 plane (out of Unicode)
1f ef ff - f7 be bf bf (4-bytes utf-8) : last  4-bytes utf-8 plane (out of Unicode)
*/

// #define UTF_ESCAPE_PLANE_HIGH  (0x1f << 16)
// #define UTF_ESCAPE_PLANE        UTF_ESCAPE_PLANE_HIGH
#define UTF_ESCAPE_PLANE 0

/*
  if (Z7_UTF_FLAG_FROM_UTF8_USE_ESCAPE is set)
  {
    if (UTF_ESCAPE_PLANE is UTF_ESCAPE_PLANE_HIGH)
    {
      we can restore any 8-bit Escape from ESCAPE-PLANE-21 plane.
      But ESCAPE-PLANE-21 point cannot be stored to utf-16 (7z archive)
      So we still need a way to extract 8-bit Escapes and BMP-Escapes-8
      from same BMP-Escapes-16 stored in 7z.
      And if we want to restore any 8-bit from 7z archive,
      we still must use Z7_UTF_FLAG_FROM_UTF8_BMP_ESCAPE_CONVERT for (utf-8 -> utf-16)
      Also we need additional Conversions to tranform from utf-16 to utf-16-With-Escapes-21
    }
    else (UTF_ESCAPE_PLANE == 0)
    {
      we must convert original 3-bytes utf-8 BMP-Escape point to sequence
      of 3 BMP-Escape-16 points with Z7_UTF_FLAG_FROM_UTF8_BMP_ESCAPE_CONVERT
      so we can extract original RAW-UTF-8 from UTFD-16 later.
    }
  }
*/

#endif



#define UTF_ESCAPE_BASE 0xef00


#ifdef UTF_ESCAPE_BASE
#define IS_ESCAPE_POINT(v, plane) (((v) & (UInt32)0xffffff80) == (plane) + UTF_ESCAPE_BASE + 0x80)
#endif

#define IS_SURROGATE_POINT(v)     (((v) & (UInt32)0xfffff800) == 0xd800)
#define IS_LOW_SURROGATE_POINT(v) (((v) & (UInt32)0xfffffc00) == 0xdc00)


#define UTF_ERROR_UTF8_CHECK \
  { NonUtf = true; continue; }

void CUtf8Check::Check_Buf(const char *src, size_t size) throw()
{
  Clear();
  // Byte maxByte = 0;

  for (;;)
  {
    if (size == 0)
      break;

    const Byte c = (Byte)(*src++);
    size--;

    if (c == 0)
    {
      ZeroChar = true;
      continue;
    }

    /*
    if (c > maxByte)
      maxByte = c;
    */

    if (c < 0x80)
      continue;
    
    if (c < 0xc0 + 2)
      UTF_ERROR_UTF8_CHECK

    unsigned numBytes;
    UInt32 val = c;
         MY_UTF8_HEAD_PARSE2(1)
    else MY_UTF8_HEAD_PARSE2(2)
    else MY_UTF8_HEAD_PARSE2(3)
    else MY_UTF8_HEAD_PARSE2(4)
    else MY_UTF8_HEAD_PARSE2(5)
    else
    {
      UTF_ERROR_UTF8_CHECK
    }

    unsigned pos = 0;
    do
    {
      if (pos == size)
        break;
      unsigned c2 = (Byte)src[pos];
      c2 -= 0x80;
      if (c2 >= 0x40)
        break;
      val <<= 6;
      val |= c2;
      if (pos == 0)
        if (val < (((unsigned)1 << 7) >> numBytes))
          break;
      pos++;
    }
    while (--numBytes);

    if (numBytes != 0)
    {
      if (pos == size)
        Truncated = true;
      else
        UTF_ERROR_UTF8_CHECK
    }

    #ifdef UTF_ESCAPE_BASE
      if (IS_ESCAPE_POINT(val, 0))
        Escape = true;
    #endif

    if (MaxHighPoint < val)
      MaxHighPoint = val;

    if (IS_SURROGATE_POINT(val))
      SingleSurrogate = true;
    
    src += pos;
    size -= pos;
  }

  // MaxByte = maxByte;
}

bool Check_UTF8_Buf(const char *src, size_t size, bool allowReduced) throw()
{
  CUtf8Check check;
  check.Check_Buf(src, size);
  return check.IsOK(allowReduced);
}

/*
bool CheckUTF8_chars(const char *src, bool allowReduced) throw()
{
  CUtf8Check check;
  check.CheckBuf(src, strlen(src));
  return check.IsOK(allowReduced);
}
*/

bool CheckUTF8_AString(const AString &s) throw()
{
  CUtf8Check check;
  check.Check_AString(s);
  return check.IsOK();
}


/*
bool CheckUTF8(const char *src, bool allowReduced) throw()
{
  // return Check_UTF8_Buf(src, strlen(src), allowReduced);

  for (;;)
  {
    const Byte c = (Byte)(*src++);
    if (c == 0)
      return true;

    if (c < 0x80)
      continue;
    if (c < 0xC0 + 2 || c >= 0xf5)
      return false;
    
    unsigned numBytes;
    MY_UTF8_HEAD_PARSE
    else
      return false;

    unsigned pos = 0;
    
    do
    {
      Byte c2 = (Byte)(*src++);
      if (c2 < 0x80 || c2 >= 0xC0)
        return allowReduced && c2 == 0;
      val <<= 6;
      val |= (c2 - 0x80);
      pos++;
    }
    while (--numBytes);

    if (val < MY_UTF8_RANGE(pos - 1))
      return false;

    if (val >= 0x110000)
      return false;
  }
}
*/

// in case of UTF-8 error we have two ways:
// 21.01- : old : 0xfffd: REPLACEMENT CHARACTER : old version
// 21.02+ : new : 0xef00 + (c) : similar to WSL scheme for low symbols

#define UTF_REPLACEMENT_CHAR  0xfffd



#define UTF_ESCAPE(c) \
   ((flags & Z7_UTF_FLAG_FROM_UTF8_USE_ESCAPE) ? \
    UTF_ESCAPE_PLANE + UTF_ESCAPE_BASE + (c) : UTF_REPLACEMENT_CHAR)

/*
#define UTF_HARD_ERROR_UTF8
  { if (dest) dest[destPos] = (wchar_t)UTF_ESCAPE(c); \
    destPos++; ok = false; continue; }
*/

// we ignore utf errors, and don't change (ok) variable!

#define UTF_ERROR_UTF8 \
  { if (dest) dest[destPos] = (wchar_t)UTF_ESCAPE(c); \
    destPos++; continue; }

// we store UTF-16 in wchar_t strings. So we use surrogates for big unicode points:

// for debug puposes only we can store UTF-32 in wchar_t:
// #define START_POINT_FOR_SURROGATE ((UInt32)0 - 1)


/*
  WIN32 MultiByteToWideChar(CP_UTF8) emits 0xfffd point, if utf-8 error was found.
  Ant it can emit single 0xfffd from 2 src bytes.
  It doesn't emit single 0xfffd from 3-4 src bytes.
  We can
    1) emit Escape point for each incorrect byte. So we can data recover later
    2) emit 0xfffd for each incorrect byte.
       That scheme is similar to Escape scheme, but we emit 0xfffd
       instead of each Escape point.
    3) emit single 0xfffd from 1-2 incorrect bytes, as WIN32 MultiByteToWideChar scheme
*/

static bool Utf8_To_Utf16(wchar_t *dest, size_t *destLen, const char *src, const char *srcLim, unsigned flags) throw()
{
  size_t destPos = 0;
  bool ok = true;

  for (;;)
  {
    if (src == srcLim)
    {
      *destLen = destPos;
      return ok;
    }
    
    const Byte c = (Byte)(*src++);

    if (c < 0x80)
    {
      if (dest)
        dest[destPos] = (wchar_t)c;
      destPos++;
      continue;
    }
    
    if (c < 0xc0 + 2
      || c >= 0xf5) // it's limit for 0x140000 unicode codes : win32 compatibility
    {
      UTF_ERROR_UTF8
    }

    unsigned numBytes;

    MY_UTF8_HEAD_PARSE_MAX_3_BYTES

    unsigned pos = 0;
    do
    {
      if (src + pos == srcLim)
        break;
      unsigned c2 = (Byte)src[pos];
      c2 -= 0x80;
      if (c2 >= 0x40)
        break;
      val <<= 6;
      val |= c2;
      pos++;
      if (pos == 1)
      {
        if (val < (((unsigned)1 << 7) >> numBytes))
          break;
        if (numBytes == 2)
        {
          if (flags & Z7_UTF_FLAG_FROM_UTF8_SURROGATE_ERROR)
            if ((val & (0xF800 >> 6)) == (0xd800 >> 6))
              break;
        }
        else if (numBytes == 3 && val >= (0x110000 >> 12))
          break;
      }
    }
    while (--numBytes);

    if (numBytes != 0)
    {
      if ((flags & Z7_UTF_FLAG_FROM_UTF8_USE_ESCAPE) == 0)
      {
        // the following code to emit the 0xfffd chars as win32 Utf8 function.
        // disable the folling line, if you need 0xfffd for each incorrect byte as in Escape mode
        src += pos;
      }
      UTF_ERROR_UTF8
    }

    /*
    if (val < MY_UTF8_RANGE(pos - 1))
      UTF_ERROR_UTF8
    */

    #ifdef UTF_ESCAPE_BASE
    
      if ((flags & Z7_UTF_FLAG_FROM_UTF8_BMP_ESCAPE_CONVERT)
          && IS_ESCAPE_POINT(val, 0))
      {
        // We will emit 3 utf16-Escape-16-21 points from one Escape-16 point (3 bytes)
        UTF_ERROR_UTF8
      }
    
    #endif

    /*
       We don't expect virtual Escape-21 points in UTF-8 stream.
       And we don't check for Escape-21.
       So utf8-Escape-21 will be converted to another 3 utf16-Escape-21 points.
       Maybe we could convert virtual utf8-Escape-21 to one utf16-Escape-21 point in some cases?
    */
    
    if (val < START_POINT_FOR_SURROGATE)
    {
      /*
      if ((flags & Z7_UTF_FLAG_FROM_UTF8_SURROGATE_ERROR)
          && IS_SURROGATE_POINT(val))
      {
        // We will emit 3 utf16-Escape-16-21 points from one Surrogate-16 point (3 bytes)
        UTF_ERROR_UTF8
      }
      */
      if (dest)
        dest[destPos] = (wchar_t)val;
      destPos++;
    }
    else
    {
      /*
      if (val >= 0x110000)
      {
        // We will emit utf16-Escape-16-21 point from each source byte
        UTF_ERROR_UTF8
      }
      */
      if (dest)
      {
        dest[destPos + 0] = (wchar_t)(0xd800 - (0x10000 >> 10) + (val >> 10));
        dest[destPos + 1] = (wchar_t)(0xdc00 + (val & 0x3ff));
      }
      destPos += 2;
    }
    src += pos;
  }
}



#define MY_UTF8_HEAD(n, val) ((char)(MY_UTF8_START(n) + (val >> (6 * (n)))))
#define MY_UTF8_CHAR(n, val) ((char)(0x80 + (((val) >> (6 * (n))) & 0x3F)))

static size_t Utf16_To_Utf8_Calc(const wchar_t *src, const wchar_t *srcLim, unsigned flags)
{
  size_t size = (size_t)(srcLim - src);
  for (;;)
  {
    if (src == srcLim)
      return size;
    
    UInt32 val = (UInt32)(*src++);
   
    if (val < 0x80)
      continue;

    if (val < MY_UTF8_RANGE(1))
    {
      size++;
      continue;
    }

    #ifdef UTF_ESCAPE_BASE
    
    #if UTF_ESCAPE_PLANE != 0
    if (flags & Z7_UTF_FLAG_TO_UTF8_PARSE_HIGH_ESCAPE)
      if (IS_ESCAPE_POINT(val, UTF_ESCAPE_PLANE))
        continue;
    #endif
    
    if (flags & Z7_UTF_FLAG_TO_UTF8_EXTRACT_BMP_ESCAPE)
      if (IS_ESCAPE_POINT(val, 0))
        continue;
    
    #endif

    if (IS_SURROGATE_POINT(val))
    {
      // it's hack to UTF-8 encoding

      if (val < 0xdc00 && src != srcLim)
      {
        const UInt32 c2 = (UInt32)*src;
        if (c2 >= 0xdc00 && c2 < 0xe000)
          src++;
      }
      size += 2;
      continue;
    }

    #ifdef Z7_WCHART_IS_16BIT
    
    size += 2;
    
    #else

         if (val < MY_UTF8_RANGE(2)) size += 2;
    else if (val < MY_UTF8_RANGE(3)) size += 3;
    else if (val < MY_UTF8_RANGE(4)) size += 4;
    else if (val < MY_UTF8_RANGE(5)) size += 5;
    else
    #if MY_UTF8_NUM_TAIL_BYTES_MAX >= 6
      size += 6;
    #else
      size += 3;
    #endif
    
    #endif
  }
}


static char *Utf16_To_Utf8(char *dest, const wchar_t *src, const wchar_t *srcLim, unsigned flags)
{
  for (;;)
  {
    if (src == srcLim)
      return dest;
    
    UInt32 val = (UInt32)*src++;
    
    if (val < 0x80)
    {
      *dest++ = (char)val;
      continue;
    }

    if (val < MY_UTF8_RANGE(1))
    {
      dest[0] = MY_UTF8_HEAD(1, val);
      dest[1] = MY_UTF8_CHAR(0, val);
      dest += 2;
      continue;
    }

    #ifdef UTF_ESCAPE_BASE
    
    #if UTF_ESCAPE_PLANE != 0
    /*
       if (wchar_t is 32-bit)
            && (Z7_UTF_FLAG_TO_UTF8_PARSE_HIGH_ESCAPE is set)
            && (point is virtual escape plane)
          we extract 8-bit byte from virtual HIGH-ESCAPE PLANE.
    */
    if (flags & Z7_UTF_FLAG_TO_UTF8_PARSE_HIGH_ESCAPE)
      if (IS_ESCAPE_POINT(val, UTF_ESCAPE_PLANE))
      {
        *dest++ = (char)(val);
        continue;
      }
    #endif // UTF_ESCAPE_PLANE != 0

    /* if (Z7_UTF_FLAG_TO_UTF8_EXTRACT_BMP_ESCAPE is defined)
          we extract 8-bit byte from BMP-ESCAPE PLANE. */

    if (flags & Z7_UTF_FLAG_TO_UTF8_EXTRACT_BMP_ESCAPE)
      if (IS_ESCAPE_POINT(val, 0))
      {
        *dest++ = (char)(val);
        continue;
      }
    
    #endif // UTF_ESCAPE_BASE

    if (IS_SURROGATE_POINT(val))
    {
      // it's hack to UTF-8 encoding
      if (val < 0xdc00 && src != srcLim)
      {
        const UInt32 c2 = (UInt32)*src;
        if (IS_LOW_SURROGATE_POINT(c2))
        {
          src++;
          val = (((val - 0xd800) << 10) | (c2 - 0xdc00)) + 0x10000;
          dest[0] = MY_UTF8_HEAD(3, val);
          dest[1] = MY_UTF8_CHAR(2, val);
          dest[2] = MY_UTF8_CHAR(1, val);
          dest[3] = MY_UTF8_CHAR(0, val);
          dest += 4;
          continue;
        }
      }
      if (flags & Z7_UTF_FLAG_TO_UTF8_SURROGATE_ERROR)
        val = UTF_REPLACEMENT_CHAR; // WIN32 function does it
    }

    #ifndef Z7_WCHART_IS_16BIT
    if (val < MY_UTF8_RANGE(2))
    #endif
    {
      dest[0] = MY_UTF8_HEAD(2, val);
      dest[1] = MY_UTF8_CHAR(1, val);
      dest[2] = MY_UTF8_CHAR(0, val);
      dest += 3;
      continue;
    }
    
    #ifndef Z7_WCHART_IS_16BIT

    // we don't expect this case. so we can throw exception
    // throw 20210407;
   
    char b;
    unsigned numBits;
         if (val < MY_UTF8_RANGE(3)) { numBits = 6 * 3; b = MY_UTF8_HEAD(3, val); }
    else if (val < MY_UTF8_RANGE(4)) { numBits = 6 * 4; b = MY_UTF8_HEAD(4, val); }
    else if (val < MY_UTF8_RANGE(5)) { numBits = 6 * 5; b = MY_UTF8_HEAD(5, val); }
    #if MY_UTF8_NUM_TAIL_BYTES_MAX >= 6
    else                           { numBits = 6 * 6; b = (char)MY_UTF8_START(6); }
    #else
    else
    {
      val = UTF_REPLACEMENT_CHAR;
                                   { numBits = 6 * 3; b = MY_UTF8_HEAD(3, val); }
    }
    #endif

    *dest++ = b;
    
    do
    {
      numBits -= 6;
      *dest++ = (char)(0x80 + ((val >> numBits) & 0x3F));
    }
    while (numBits != 0);

    #endif
  }
}

bool Convert_UTF8_Buf_To_Unicode(const char *src, size_t srcSize, UString &dest, unsigned flags)
{
  dest.Empty();
  size_t destLen = 0;
  Utf8_To_Utf16(NULL, &destLen, src, src + srcSize, flags);
  bool res = Utf8_To_Utf16(dest.GetBuf((unsigned)destLen), &destLen, src, src + srcSize, flags);
  dest.ReleaseBuf_SetEnd((unsigned)destLen);
  return res;
}

bool ConvertUTF8ToUnicode_Flags(const AString &src, UString &dest, unsigned flags)
{
  return Convert_UTF8_Buf_To_Unicode(src, src.Len(), dest,  flags);
}


static
unsigned g_UTF8_To_Unicode_Flags =
    Z7_UTF_FLAG_FROM_UTF8_USE_ESCAPE
  #ifndef Z7_WCHART_IS_16BIT
    | Z7_UTF_FLAG_FROM_UTF8_SURROGATE_ERROR
  #ifdef MY_UTF8_RAW_NON_UTF8_SUPPORTED
    | Z7_UTF_FLAG_FROM_UTF8_BMP_ESCAPE_CONVERT
  #endif
  #endif
    ;
    

/*
bool ConvertUTF8ToUnicode_boolRes(const AString &src, UString &dest)
{
  return ConvertUTF8ToUnicode_Flags(src, dest, g_UTF8_To_Unicode_Flags);
}
*/

bool ConvertUTF8ToUnicode(const AString &src, UString &dest)
{
  return ConvertUTF8ToUnicode_Flags(src, dest, g_UTF8_To_Unicode_Flags);
}

void Print_UString(const UString &a);

void ConvertUnicodeToUTF8_Flags(const UString &src, AString &dest, unsigned flags)
{
  /*
  if (src.Len()== 24)
    throw "202104";
  */
  dest.Empty();
  const size_t destLen = Utf16_To_Utf8_Calc(src, src.Ptr(src.Len()), flags);
  char *destStart = dest.GetBuf((unsigned)destLen);
  const char *destEnd = Utf16_To_Utf8(destStart, src, src.Ptr(src.Len()), flags);
  dest.ReleaseBuf_SetEnd((unsigned)destLen);
  // printf("\nlen = %d\n", src.Len());
  if (destLen != (size_t)(destEnd - destStart))
  {
    /*
    // dest.ReleaseBuf_SetEnd((unsigned)(destEnd - destStart));
    printf("\nlen = %d\n", (unsigned)destLen);
    printf("\n(destEnd - destStart) = %d\n", (unsigned)(destEnd - destStart));
    printf("\n");
    // Print_UString(src);
    printf("\n");
    // printf("\nlen = %d\n", destLen);
    */
    throw 20210406;
  }
}



unsigned g_Unicode_To_UTF8_Flags =
      // Z7_UTF_FLAG_TO_UTF8_PARSE_HIGH_ESCAPE
      0
  #ifndef _WIN32
    #ifdef MY_UTF8_RAW_NON_UTF8_SUPPORTED
      | Z7_UTF_FLAG_TO_UTF8_EXTRACT_BMP_ESCAPE
    #else
      | Z7_UTF_FLAG_TO_UTF8_SURROGATE_ERROR
    #endif
  #endif
    ;

void ConvertUnicodeToUTF8(const UString &src, AString &dest)
{
  ConvertUnicodeToUTF8_Flags(src, dest, g_Unicode_To_UTF8_Flags);
}

void Convert_Unicode_To_UTF8_Buf(const UString &src, CByteBuffer &dest)
{
  const unsigned flags = g_Unicode_To_UTF8_Flags;
  dest.Free();
  const size_t destLen = Utf16_To_Utf8_Calc(src, src.Ptr(src.Len()), flags);
  dest.Alloc(destLen);
  const char *destEnd = Utf16_To_Utf8((char *)(void *)(Byte *)dest, src, src.Ptr(src.Len()), flags);
  if (destLen != (size_t)(destEnd - (char *)(void *)(Byte *)dest))
    throw 202104;
}

/*

#ifndef _WIN32
void Convert_UTF16_To_UTF32(const UString &src, UString &dest)
{
  dest.Empty();
  for (size_t i = 0; i < src.Len();)
  {
    wchar_t c = src[i++];
    if (c >= 0xd800 && c < 0xdc00 && i < src.Len())
    {
      const wchar_t c2 = src[i];
      if (c2 >= 0xdc00 && c2 < 0xe000)
      {
        // printf("\nSurragate [%d]: %4x %4x -> ", i, (int)c, (int)c2);
        c = 0x10000 + ((c & 0x3ff) << 10) + (c2 & 0x3ff);
        // printf("%4x\n", (int)c);
        i++;
      }
    }
    dest += c;
  }
}

void Convert_UTF32_To_UTF16(const UString &src, UString &dest)
{
  dest.Empty();
  for (size_t i = 0; i < src.Len();)
  {
    wchar_t w = src[i++];
    if (w >= 0x10000 && w < 0x110000)
    {
      w -= 0x10000;
      dest += (wchar_t)((unsigned)0xd800 + (((unsigned)w >> 10) & 0x3ff));
      w = 0xdc00 + (w & 0x3ff);
    }
    dest += w;
  }
}

bool UTF32_IsThere_BigPoint(const UString &src)
{
  for (size_t i = 0; i < src.Len();)
  {
    const UInt32 c = (UInt32)src[i++];
    if (c >= 0x110000)
      return true;
  }
  return false;
}

bool Unicode_IsThere_BmpEscape(const UString &src)
{
  for (size_t i = 0; i < src.Len();)
  {
    const UInt32 c = (UInt32)src[i++];
    if (IS_ESCAPE_POINT(c, 0))
      return true;
  }
  return false;
}


#endif

bool Unicode_IsThere_Utf16SurrogateError(const UString &src)
{
  for (size_t i = 0; i < src.Len();)
  {
    const UInt32 val = (UInt32)src[i++];
    if (IS_SURROGATE_POINT(val))
    {
      // it's hack to UTF-8 encoding
      if (val >= 0xdc00 || i == src.Len())
        return true;
      const UInt32 c2 = (UInt32)*src;
      if (!IS_LOW_SURROGATE_POINT(c2))
        return true;
    }
  }
  return false;
}
*/

#ifndef Z7_WCHART_IS_16BIT

void Convert_UnicodeEsc16_To_UnicodeEscHigh
#if UTF_ESCAPE_PLANE == 0
    (UString &) {}
#else
    (UString &s)
{
  const unsigned len = s.Len();
  for (unsigned i = 0; i < len; i++)
  {
    wchar_t c = s[i];
    if (IS_ESCAPE_POINT(c, 0))
    {
      c += UTF_ESCAPE_PLANE;
      s.ReplaceOneCharAtPos(i, c);
    }
  }
}
#endif
#endif
