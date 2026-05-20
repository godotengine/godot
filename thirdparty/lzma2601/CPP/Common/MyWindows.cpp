// MyWindows.cpp

#include "StdAfx.h"

#ifndef _WIN32

#include <stdlib.h>
#include <time.h>
#ifdef __GNUC__
#include <sys/time.h>
#endif

#include "MyWindows.h"

static inline void *AllocateForBSTR(size_t cb) { return ::malloc(cb); }
static inline void FreeForBSTR(void *pv) { ::free(pv);}

/* Win32 uses DWORD (32-bit) type to store size of string before (OLECHAR *) string.
  We must select CBstrSizeType for another systems (not Win32):

    if (CBstrSizeType is UINT32),
          then we support only strings smaller than 4 GB.
          Win32 version always has that limitation.
  
    if (CBstrSizeType is UINT),
          (UINT can be 16/32/64-bit)
          We can support strings larger than 4 GB (if UINT is 64-bit),
          but sizeof(UINT) can be different in parts compiled by
          different compilers/settings,
          and we can't send such BSTR strings between such parts.
*/

typedef UINT32 CBstrSizeType;
// typedef UINT CBstrSizeType;

#define k_BstrSize_Max 0xFFFFFFFF
// #define k_BstrSize_Max UINT_MAX
// #define k_BstrSize_Max ((UINT)(INT)-1)

BSTR SysAllocStringByteLen(LPCSTR s, UINT len)
{
  /* Original SysAllocStringByteLen in Win32 maybe fills only unaligned null OLECHAR at the end.
     We provide also aligned null OLECHAR at the end. */

  if (len >= (k_BstrSize_Max - (UINT)sizeof(OLECHAR) - (UINT)sizeof(OLECHAR) - (UINT)sizeof(CBstrSizeType)))
    return NULL;

  UINT size = (len + (UINT)sizeof(OLECHAR) + (UINT)sizeof(OLECHAR) - 1) & ~((UINT)sizeof(OLECHAR) - 1);
  void *p = AllocateForBSTR(size + (UINT)sizeof(CBstrSizeType));
  if (!p)
    return NULL;
  *(CBstrSizeType *)p = (CBstrSizeType)len;
  BSTR bstr = (BSTR)((CBstrSizeType *)p + 1);
  if (s)
    memcpy(bstr, s, len);
  for (; len < size; len++)
    ((Byte *)bstr)[len] = 0;
  return bstr;
}

BSTR SysAllocStringLen(const OLECHAR *s, UINT len)
{
  if (len >= (k_BstrSize_Max - (UINT)sizeof(OLECHAR) - (UINT)sizeof(CBstrSizeType)) / (UINT)sizeof(OLECHAR))
    return NULL;

  UINT size = len * (UINT)sizeof(OLECHAR);
  void *p = AllocateForBSTR(size + (UINT)sizeof(CBstrSizeType) + (UINT)sizeof(OLECHAR));
  if (!p)
    return NULL;
  *(CBstrSizeType *)p = (CBstrSizeType)size;
  BSTR bstr = (BSTR)((CBstrSizeType *)p + 1);
  if (s)
    memcpy(bstr, s, size);
  bstr[len] = 0;
  return bstr;
}

BSTR SysAllocString(const OLECHAR *s)
{
  if (!s)
    return NULL;
  const OLECHAR *s2 = s;
  while (*s2 != 0)
    s2++;
  return SysAllocStringLen(s, (UINT)(s2 - s));
}

void SysFreeString(BSTR bstr)
{
  if (bstr)
    FreeForBSTR((CBstrSizeType *)(void *)bstr - 1);
}

UINT SysStringByteLen(BSTR bstr)
{
  if (!bstr)
    return 0;
  return *((CBstrSizeType *)(void *)bstr - 1);
}

UINT SysStringLen(BSTR bstr)
{
  if (!bstr)
    return 0;
  return *((CBstrSizeType *)(void *)bstr - 1) / (UINT)sizeof(OLECHAR);
}


HRESULT VariantClear(VARIANTARG *prop)
{
  if (prop->vt == VT_BSTR)
    SysFreeString(prop->bstrVal);
  prop->vt = VT_EMPTY;
  return S_OK;
}

HRESULT VariantCopy(VARIANTARG *dest, const VARIANTARG *src)
{
  HRESULT res = ::VariantClear(dest);
  if (res != S_OK)
    return res;
  if (src->vt == VT_BSTR)
  {
    dest->bstrVal = SysAllocStringByteLen((LPCSTR)src->bstrVal,
        SysStringByteLen(src->bstrVal));
    if (!dest->bstrVal)
      return E_OUTOFMEMORY;
    dest->vt = VT_BSTR;
  }
  else
    *dest = *src;
  return S_OK;
}

LONG CompareFileTime(const FILETIME* ft1, const FILETIME* ft2)
{
  if (ft1->dwHighDateTime < ft2->dwHighDateTime) return -1;
  if (ft1->dwHighDateTime > ft2->dwHighDateTime) return 1;
  if (ft1->dwLowDateTime < ft2->dwLowDateTime) return -1;
  if (ft1->dwLowDateTime > ft2->dwLowDateTime) return 1;
  return 0;
}

DWORD GetLastError()
{
  return (DWORD)errno;
}

void SetLastError(DWORD dw)
{
  errno = (int)dw;
}


static LONG TIME_GetBias()
{
  time_t utc = time(NULL);
  struct tm *ptm = localtime(&utc);
  int localdaylight = ptm->tm_isdst; /* daylight for local timezone */
  ptm = gmtime(&utc);
  ptm->tm_isdst = localdaylight; /* use local daylight, not that of Greenwich */
  LONG bias = (int)(mktime(ptm)-utc);
  return bias;
}

#define TICKS_PER_SEC 10000000
/*
#define SECS_PER_DAY (24 * 60 * 60)
#define SECS_1601_TO_1970  ((369 * 365 + 89) * (UInt64)SECS_PER_DAY)
#define TICKS_1601_TO_1970 (SECS_1601_TO_1970 * TICKS_PER_SEC)
*/

#define GET_TIME_64(pft) ((pft)->dwLowDateTime | ((UInt64)(pft)->dwHighDateTime << 32))

#define SET_FILETIME(ft, v64) \
   (ft)->dwLowDateTime = (DWORD)v64; \
   (ft)->dwHighDateTime = (DWORD)(v64 >> 32);


BOOL WINAPI FileTimeToLocalFileTime(const FILETIME *fileTime, FILETIME *localFileTime)
{
  UInt64 v = GET_TIME_64(fileTime);
  v = (UInt64)((Int64)v - (Int64)TIME_GetBias() * TICKS_PER_SEC);
  SET_FILETIME(localFileTime, v)
  return TRUE;
}

BOOL WINAPI LocalFileTimeToFileTime(const FILETIME *localFileTime, FILETIME *fileTime)
{
  UInt64 v = GET_TIME_64(localFileTime);
  v = (UInt64)((Int64)v + (Int64)TIME_GetBias() * TICKS_PER_SEC);
  SET_FILETIME(fileTime, v)
  return TRUE;
}

/*
VOID WINAPI GetSystemTimeAsFileTime(FILETIME *ft)
{
  UInt64 t = 0;
  timeval tv;
  if (gettimeofday(&tv, NULL) == 0)
  {
    t = tv.tv_sec * (UInt64)TICKS_PER_SEC + TICKS_1601_TO_1970;
    t += tv.tv_usec * 10;
  }
  SET_FILETIME(ft, t)
}
*/

DWORD WINAPI GetTickCount(VOID)
{
  #ifndef _WIN32
  // gettimeofday() doesn't work in some MINGWs by unknown reason
  timeval tv;
  if (gettimeofday(&tv, NULL) == 0)
  {
    // tv_sec and tv_usec are (long)
    return (DWORD)((UInt64)(Int64)tv.tv_sec * (UInt64)1000 + (UInt64)(Int64)tv.tv_usec / 1000);
  }
  #endif
  return (DWORD)time(NULL) * 1000;
}


#define PERIOD_4 (4 * 365 + 1)
#define PERIOD_100 (PERIOD_4 * 25 - 1)
#define PERIOD_400 (PERIOD_100 * 4 + 1)

BOOL WINAPI FileTimeToSystemTime(const FILETIME *ft, SYSTEMTIME *st)
{
  UInt32 v;
  UInt64 v64 = GET_TIME_64(ft);
  v64 /= 10000;
  st->wMilliseconds = (WORD)(v64 % 1000); v64 /= 1000;
  st->wSecond       = (WORD)(v64 %   60); v64 /= 60;
  st->wMinute       = (WORD)(v64 %   60); v64 /= 60;
  v = (UInt32)v64;
  st->wHour         = (WORD)(v %   24); v /= 24;

  // 1601-01-01 was Monday
  st->wDayOfWeek = (WORD)((v + 1) % 7);

  UInt32 leaps, year, day, mon;
  leaps = (3 * ((4 * v + (365 - 31 - 28) * 4 + 3) / PERIOD_400) + 3) / 4;
  v += 28188 + leaps;
  // leaps - the number of exceptions from PERIOD_4 rules starting from 1600-03-01
  // (1959 / 64) - converts day from 03-01 to month
  year = (20 * v - 2442) / (5 * PERIOD_4);
  day = v - (year * PERIOD_4) / 4;
  mon = (64 * day) / 1959;
  st->wDay = (WORD)(day - (1959 * mon) / 64);
  mon -= 1;
  year += 1524;
  if (mon > 12)
  {
    mon -= 12;
    year++;
  }
  st->wMonth = (WORD)mon;
  st->wYear = (WORD)year;

  /*
  unsigned year, mon;
  unsigned char ms[] = { 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };
  unsigned t;

  year = (WORD)(1601 + v / PERIOD_400 * 400);
  v %= PERIOD_400;

  t = v / PERIOD_100; if (t ==  4) t =  3; year += t * 100; v -= t * PERIOD_100;
  t = v / PERIOD_4;   if (t == 25) t = 24; year += t * 4;   v -= t * PERIOD_4;
  t = v / 365;        if (t ==  4) t =  3; year += t;       v -= t * 365;

  st->wYear = (WORD)year;

  if (year % 4 == 0 && (year % 100 != 0 || year % 400 == 0))
    ms[1] = 29;
  for (mon = 0;; mon++)
  {
    unsigned d = ms[mon];
    if (v < d)
      break;
    v -= d;
  }
  st->wDay = (WORD)(v + 1);
  st->wMonth = (WORD)(mon + 1);
  */

  return TRUE;
}

#endif
