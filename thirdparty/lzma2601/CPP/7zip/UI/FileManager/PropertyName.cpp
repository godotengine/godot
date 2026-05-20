// PropertyName.cpp

#include "StdAfx.h"

#include "../../../Common/IntToString.h"

#include "LangUtils.h"
#include "PropertyName.h"

UString GetNameOfProperty(PROPID propID, const wchar_t *name)
{
  if (propID < 1000)
  {
    UString s = LangString(1000 + propID);
    if (!s.IsEmpty())
      return s;
  }
  if (name)
    return name;
  wchar_t temp[16];
  ConvertUInt32ToString(propID, temp);
  return temp;
}
