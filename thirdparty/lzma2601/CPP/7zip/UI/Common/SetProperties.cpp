// SetProperties.cpp

#include "StdAfx.h"

#include "../../../Common/MyCom.h"
#include "../../../Common/MyString.h"
#include "../../../Common/StringToInt.h"

#include "../../../Windows/PropVariant.h"

#include "../../Archive/IArchive.h"

#include "SetProperties.h"

using namespace NWindows;
using namespace NCOM;

static void ParseNumberString(const UString &s, NCOM::CPropVariant &prop)
{
  const wchar_t *end;
  const UInt64 result = ConvertStringToUInt64(s, &end);
  if (*end != 0 || s.IsEmpty())
    prop = s;
  else if (result <= (UInt32)0xFFFFFFFF)
    prop = (UInt32)result;
  else
    prop = result;
}


struct CPropPropetiesVector
{
  CPropVariant *values;
  CPropPropetiesVector(unsigned num)
  {
    values = new CPropVariant[num];
  }
  ~CPropPropetiesVector()
  {
    delete []values;
  }
};


HRESULT SetProperties(IUnknown *unknown, const CObjectVector<CProperty> &properties)
{
  if (properties.IsEmpty())
    return S_OK;
  Z7_DECL_CMyComPtr_QI_FROM(
      ISetProperties,
      setProperties, unknown)
  if (!setProperties)
    return S_OK;

  UStringVector realNames;
  CPropPropetiesVector values(properties.Size());
  {
    unsigned i;
    for (i = 0; i < properties.Size(); i++)
    {
      const CProperty &property = properties[i];
      NCOM::CPropVariant propVariant;
      UString name = property.Name;
      if (property.Value.IsEmpty())
      {
        if (!name.IsEmpty())
        {
          const wchar_t c = name.Back();
          if (c == L'-')
            propVariant = false;
          else if (c == L'+')
            propVariant = true;
          if (propVariant.vt != VT_EMPTY)
            name.DeleteBack();
        }
      }
      else
        ParseNumberString(property.Value, propVariant);
      realNames.Add(name);
      values.values[i] = propVariant;
    }
    CRecordVector<const wchar_t *> names;
    for (i = 0; i < realNames.Size(); i++)
      names.Add((const wchar_t *)realNames[i]);
    
    return setProperties->SetProperties(names.ConstData(), values.values, names.Size());
  }
}
