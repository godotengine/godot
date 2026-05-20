// FormatUtils.cpp

#include "StdAfx.h"

#include "../../../Common/IntToString.h"

#include "FormatUtils.h"

#include "LangUtils.h"

UString NumberToString(UInt64 number)
{
  wchar_t numberString[32];
  ConvertUInt64ToString(number, numberString);
  return numberString;
}

UString MyFormatNew(const UString &format, const UString &argument)
{
  UString result = format;
  result.Replace(L"{0}", argument);
  return result;
}

UString MyFormatNew(UINT resourceID, const UString &argument)
{
  return MyFormatNew(LangString(resourceID), argument);
}
