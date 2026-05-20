// Windows/ResourceString.cpp

#include "StdAfx.h"

#ifndef _UNICODE
#include "../Common/StringConvert.h"
#endif

#include "ResourceString.h"

extern HINSTANCE g_hInstance;
#ifndef _UNICODE
extern bool g_IsNT;
#endif

namespace NWindows {

#ifndef _UNICODE

static CSysString MyLoadStringA(HINSTANCE hInstance, UINT resourceID)
{
  CSysString s;
  int size = 128;
  int len;
  do
  {
    size <<= 1;
    len = ::LoadString(hInstance, resourceID, s.GetBuf((unsigned)size - 1), size);
  }
  while (size - len <= 1);
  s.ReleaseBuf_CalcLen((unsigned)len);
  return s;
}

#endif

static const int kStartSize = 256;

static void MyLoadString2(HINSTANCE hInstance, UINT resourceID, UString &s)
{
  int size = kStartSize;
  int len;
  do
  {
    size <<= 1;
    len = ::LoadStringW(hInstance, resourceID, s.GetBuf((unsigned)size - 1), size);
  }
  while (size - len <= 1);
  s.ReleaseBuf_CalcLen((unsigned)len);
}

// NT4 doesn't support LoadStringW(,,, 0) to get pointer to resource string. So we don't use it.

UString MyLoadString(UINT resourceID)
{
  #ifndef _UNICODE
  if (!g_IsNT)
    return GetUnicodeString(MyLoadStringA(g_hInstance, resourceID));
  else
  #endif
  {
    {
      wchar_t s[kStartSize];
      s[0] = 0;
      int len = ::LoadStringW(g_hInstance, resourceID, s, kStartSize);
      if (kStartSize - len > 1)
        return s;
    }
    UString dest;
    MyLoadString2(g_hInstance, resourceID, dest);
    return dest;
  }
}

void MyLoadString(HINSTANCE hInstance, UINT resourceID, UString &dest)
{
  dest.Empty();
  #ifndef _UNICODE
  if (!g_IsNT)
    MultiByteToUnicodeString2(dest, MyLoadStringA(hInstance, resourceID));
  else
  #endif
  {
    {
      wchar_t s[kStartSize];
      s[0] = 0;
      int len = ::LoadStringW(hInstance, resourceID, s, kStartSize);
      if (kStartSize - len > 1)
      {
        dest = s;
        return;
      }
    }
    MyLoadString2(hInstance, resourceID, dest);
  }
}

void MyLoadString(UINT resourceID, UString &dest)
{
  MyLoadString(g_hInstance, resourceID, dest);
}

}
