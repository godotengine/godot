// Windows/PropVariantConv.h

#ifndef ZIP7_INC_PROP_VARIANT_CONV_H
#define ZIP7_INC_PROP_VARIANT_CONV_H

#include "../Common/MyTypes.h"

// provide at least 32 bytes for buffer including zero-end

extern bool g_Timestamp_Show_UTC;

#define kTimestampPrintLevel_DAY -3
// #define kTimestampPrintLevel_HOUR -2
#define kTimestampPrintLevel_MIN -1
#define kTimestampPrintLevel_SEC  0
#define kTimestampPrintLevel_NTFS 7
#define kTimestampPrintLevel_NS   9


#define kTimestampPrintFlags_Force_UTC   (1 << 0)
#define kTimestampPrintFlags_Force_LOCAL (1 << 1)
#define kTimestampPrintFlags_DisableZ    (1 << 4)

bool ConvertUtcFileTimeToString(const FILETIME &ft, char *s, int level = kTimestampPrintLevel_SEC) throw();
bool ConvertUtcFileTimeToString(const FILETIME &ft, wchar_t *s, int level = kTimestampPrintLevel_SEC) throw();
bool ConvertUtcFileTimeToString2(const FILETIME &ft, unsigned ns100, char *s, int level = kTimestampPrintLevel_SEC, unsigned flags = 0) throw();
bool ConvertUtcFileTimeToString2(const FILETIME &ft, unsigned ns100, wchar_t *s, int level = kTimestampPrintLevel_SEC) throw();

// provide at least 32 bytes for buffer including zero-end
// don't send VT_BSTR to these functions
void ConvertPropVariantToShortString(const PROPVARIANT &prop, char *dest) throw();
void ConvertPropVariantToShortString(const PROPVARIANT &prop, wchar_t *dest) throw();

inline bool ConvertPropVariantToUInt64(const PROPVARIANT &prop, UInt64 &value)
{
  switch (prop.vt)
  {
    case VT_UI8: value = (UInt64)prop.uhVal.QuadPart; return true;
    case VT_UI4: value = prop.ulVal; return true;
    case VT_UI2: value = prop.uiVal; return true;
    case VT_UI1: value = prop.bVal; return true;
    case VT_EMPTY: return false;
    default: throw 151199;
  }
}

#endif
