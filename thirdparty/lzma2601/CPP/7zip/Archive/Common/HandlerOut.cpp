// HandlerOut.cpp

#include "StdAfx.h"

#include "../../../Common/StringToInt.h"

#include "HandlerOut.h"

namespace NArchive {

bool ParseSizeString(const wchar_t *s, const PROPVARIANT &prop, UInt64 percentsBase, UInt64 &res)
{
  if (*s == 0)
  {
    switch (prop.vt)
    {
      case VT_UI4: res = prop.ulVal; return true;
      case VT_UI8: res = prop.uhVal.QuadPart; return true;
      case VT_BSTR:
        s = prop.bstrVal;
        break;
      default: return false;
    }
  }
  else if (prop.vt != VT_EMPTY)
    return false;

  bool percentMode = false;
  {
    const wchar_t c = *s;
    if (MyCharLower_Ascii(c) == 'p')
    {
      percentMode = true;
      s++;
    }
  }

  const wchar_t *end;
  const UInt64 v = ConvertStringToUInt64(s, &end);
  if (s == end)
    return false;
  const wchar_t c = *end;

  if (percentMode)
  {
    if (c != 0)
      return false;
    res = Calc_From_Val_Percents(percentsBase, v);
    return true;
  }

  if (c == 0)
  {
    res = v;
    return true;
  }
  if (end[1] != 0)
    return false;

  if (c == '%')
  {
    res = Calc_From_Val_Percents(percentsBase, v);
    return true;
  }

  unsigned numBits;
  switch (MyCharLower_Ascii(c))
  {
    case 'b': numBits =  0; break;
    case 'k': numBits = 10; break;
    case 'm': numBits = 20; break;
    case 'g': numBits = 30; break;
    case 't': numBits = 40; break;
    default: return false;
  }
  const UInt64 val2 = v << numBits;
  if ((val2 >> numBits) != v)
    return false;
  res = val2;
  return true;
}


bool CCommonMethodProps::SetCommonProperty(const UString &name, const PROPVARIANT &value, HRESULT &hres)
{
  hres = S_OK;

  if (name.IsPrefixedBy_Ascii_NoCase("mt"))
  {
    #ifndef Z7_ST
    _numThreads = _numProcessors;
    _numThreads_WasForced = false;
    hres = ParseMtProp2(name.Ptr(2), value, _numThreads, _numThreads_WasForced);
    // "mt" means "_numThreads_WasForced = false" here
    #endif
    return true;
  }
  
  if (name.IsPrefixedBy_Ascii_NoCase("memuse"))
  {
    UInt64 v;
    if (!ParseSizeString(name.Ptr(6), value, _memAvail, v))
      hres = E_INVALIDARG;
    _memUsage_Decompress = v;
    _memUsage_Compress = v;
    _memUsage_WasSet = true;
    return true;
  }

  return false;
}


#ifndef Z7_EXTRACT_ONLY

static void SetMethodProp32(CMethodProps &m, PROPID propID, UInt32 value)
{
  if (m.FindProp(propID) < 0)
    m.AddProp32(propID, value);
}

void CMultiMethodProps::SetGlobalLevelTo(COneMethodInfo &oneMethodInfo) const
{
  UInt32 level = _level;
  if (level != (UInt32)(Int32)-1)
    SetMethodProp32(oneMethodInfo, NCoderPropID::kLevel, (UInt32)level);
}

#ifndef Z7_ST

static void SetMethodProp32_Replace(CMethodProps &m, PROPID propID, UInt32 value)
{
  const int i = m.FindProp(propID);
  if (i >= 0)
  {
    NWindows::NCOM::CPropVariant &val = m.Props[(unsigned)i].Value;
    val = (UInt32)value;
    return;
  }
  m.AddProp32(propID, value);
}

void CMultiMethodProps::SetMethodThreadsTo_IfNotFinded(CMethodProps &oneMethodInfo, UInt32 numThreads)
{
  SetMethodProp32(oneMethodInfo, NCoderPropID::kNumThreads, numThreads);
}

void CMultiMethodProps::SetMethodThreadsTo_Replace(CMethodProps &oneMethodInfo, UInt32 numThreads)
{
  SetMethodProp32_Replace(oneMethodInfo, NCoderPropID::kNumThreads, numThreads);
}

void CMultiMethodProps::Set_Method_NumThreadGroups_IfNotFinded(CMethodProps &oneMethodInfo, UInt32 numThreadGroups)
{
  SetMethodProp32(oneMethodInfo, NCoderPropID::kNumThreadGroups, numThreadGroups);
}

#endif // Z7_ST


void CMultiMethodProps::InitMulti()
{
  _level = (UInt32)(Int32)-1;
  _analysisLevel = -1;
  _crcSize = 4;
  _autoFilter = true;
}

void CMultiMethodProps::Init()
{
  InitCommon();
  InitMulti();
  _methods.Clear();
  _filterMethod.Clear();
}


HRESULT CMultiMethodProps::SetProperty(const wchar_t *nameSpec, const PROPVARIANT &value)
{
  UString name = nameSpec;
  name.MakeLower_Ascii();
  if (name.IsEmpty())
    return E_INVALIDARG;
  
  if (name[0] == 'x')
  {
    name.Delete(0);
    _level = 9;
    return ParsePropToUInt32(name, value, _level);
  }

  if (name.IsPrefixedBy_Ascii_NoCase("yx"))
  {
    name.Delete(0, 2);
    UInt32 v = 9;
    RINOK(ParsePropToUInt32(name, value, v))
    _analysisLevel = (int)v;
    return S_OK;
  }
  
  if (name.IsPrefixedBy_Ascii_NoCase("crc"))
  {
    name.Delete(0, 3);
    _crcSize = 4;
    return ParsePropToUInt32(name, value, _crcSize);
  }

  {
    HRESULT hres;
    if (SetCommonProperty(name, value, hres))
      return hres;
  }
  
  UInt32 number;
  const unsigned index = ParseStringToUInt32(name, number);
  const UString realName = name.Ptr(index);
  if (index == 0)
  {
    if (name.IsEqualTo("f"))
    {
      const HRESULT res = PROPVARIANT_to_bool(value, _autoFilter);
      if (res == S_OK)
        return res;
      if (value.vt != VT_BSTR)
        return E_INVALIDARG;
      return _filterMethod.ParseMethodFromPROPVARIANT(UString(), value);
    }
    number = 0;
  }
  if (number > 64)
    return E_INVALIDARG;
  for (unsigned j = _methods.Size(); j <= number; j++)
    _methods.AddNew();
  return _methods[number].ParseMethodFromPROPVARIANT(realName, value);
}



void CSingleMethodProps::Init()
{
  InitCommon();
  InitSingle();
  Clear();
}


HRESULT CSingleMethodProps::SetProperty(const wchar_t *name2, const PROPVARIANT &value)
{
  // processed = false;
  UString name = name2;
  name.MakeLower_Ascii();
  if (name.IsEmpty())
    return E_INVALIDARG;
  if (name.IsPrefixedBy_Ascii_NoCase("x"))
  {
    UInt32 a = 9;
    RINOK(ParsePropToUInt32(name.Ptr(1), value, a))
    _level = a;
    AddProp_Level(a);
    // processed = true;
    return S_OK;
  }
  {
    HRESULT hres;
    if (SetCommonProperty(name, value, hres))
    {
      // processed = true;
      return S_OK;
    }
  }
  RINOK(ParseMethodFromPROPVARIANT(name, value))
  return S_OK;
}


HRESULT CSingleMethodProps::SetProperties(const wchar_t * const *names, const PROPVARIANT *values, UInt32 numProps)
{
  Init();
  
  for (UInt32 i = 0; i < numProps; i++)
  {
    RINOK(SetProperty(names[i], values[i]))
  }

  return S_OK;
}

#endif


static HRESULT PROPVARIANT_to_BoolPair(const PROPVARIANT &prop, CBoolPair &dest)
{
  RINOK(PROPVARIANT_to_bool(prop, dest.Val))
  dest.Def = true;
  return S_OK;
}

HRESULT CHandlerTimeOptions::Parse(const UString &name, const PROPVARIANT &prop, bool &processed)
{
  processed = true;
  if (name.IsEqualTo_Ascii_NoCase("tm")) { return PROPVARIANT_to_BoolPair(prop, Write_MTime); }
  if (name.IsEqualTo_Ascii_NoCase("ta")) { return PROPVARIANT_to_BoolPair(prop, Write_ATime); }
  if (name.IsEqualTo_Ascii_NoCase("tc")) { return PROPVARIANT_to_BoolPair(prop, Write_CTime); }
  if (name.IsPrefixedBy_Ascii_NoCase("tp"))
  {
    UInt32 v = 0;
    RINOK(ParsePropToUInt32(name.Ptr(2), prop, v))
    Prec = v;
    return S_OK;
  }
  processed = false;
  return S_OK;
}

}
