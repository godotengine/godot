// CodecExports.cpp

#include "StdAfx.h"

#include "../../../C/CpuArch.h"
#include "../../../C/7zVersion.h"

#include "../../Common/ComTry.h"
#include "../../Common/MyCom.h"

#include "../../Windows/Defs.h"
#include "../../Windows/PropVariant.h"

#include "../ICoder.h"

#include "../Common/RegisterCodec.h"

extern unsigned g_NumCodecs;
extern const CCodecInfo *g_Codecs[];

extern unsigned g_NumHashers;
extern const CHasherInfo *g_Hashers[];

static void SetPropFromAscii(const char *s, PROPVARIANT *prop) throw()
{
  const UINT len = (UINT)strlen(s);
  BSTR dest = ::SysAllocStringLen(NULL, len);
  if (dest)
  {
    for (UINT i = 0; i <= len; i++)
      dest[i] = (Byte)s[i];
    prop->bstrVal = dest;
    prop->vt = VT_BSTR;
  }
}

static inline HRESULT SetPropGUID(const GUID &guid, PROPVARIANT *value) throw()
{
  if ((value->bstrVal = ::SysAllocStringByteLen((const char *)&guid, sizeof(guid))) != NULL)
    value->vt = VT_BSTR;
  return S_OK;
}

static HRESULT MethodToClassID(UInt16 typeId, CMethodId id, PROPVARIANT *value) throw()
{
  GUID clsId;
  clsId.Data1 = k_7zip_GUID_Data1;
  clsId.Data2 = k_7zip_GUID_Data2;
  clsId.Data3 = typeId;
  SetUi64(clsId.Data4, id)
  return SetPropGUID(clsId, value);
}

static HRESULT FindCodecClassId(const GUID *clsid, bool isCoder2, bool isFilter, bool &encode, int &index) throw()
{
  index = -1;
  if (clsid->Data1 != k_7zip_GUID_Data1 ||
      clsid->Data2 != k_7zip_GUID_Data2)
    return S_OK;
  
  encode = true;
  
       if (clsid->Data3 == k_7zip_GUID_Data3_Decoder) encode = false;
  else if (clsid->Data3 != k_7zip_GUID_Data3_Encoder) return S_OK;
  
  const UInt64 id = GetUi64(clsid->Data4);
  
  for (unsigned i = 0; i < g_NumCodecs; i++)
  {
    const CCodecInfo &codec = *g_Codecs[i];
    
    if (id != codec.Id
        || (encode ? !codec.CreateEncoder : !codec.CreateDecoder)
        || (isFilter ? !codec.IsFilter : codec.IsFilter))
      continue;

    if (codec.NumStreams == 1 ? isCoder2 : !isCoder2)
      return E_NOINTERFACE;
    
    index = (int)i;
    return S_OK;
  }
  
  return S_OK;
}

/*
#ifdef __GNUC__
#ifndef __clang__
#pragma GCC diagnostic ignored "-Wduplicated-branches"
#endif
#endif
*/

static HRESULT CreateCoderMain(unsigned index, bool encode, void **coder)
{
  COM_TRY_BEGIN
  
  const CCodecInfo &codec = *g_Codecs[index];
  
  void *c;
  if (encode)
    c = codec.CreateEncoder();
  else
    c = codec.CreateDecoder();
  
  if (c)
  {
    IUnknown *unk;
    unk = (IUnknown *)c;
    /*
    if (codec.IsFilter)
      unk = (IUnknown *)(ICompressFilter *)c;
    else if (codec.NumStreams != 1)
      unk = (IUnknown *)(ICompressCoder2 *)c;
    else
      unk = (IUnknown *)(ICompressCoder *)c;
    */
    unk->AddRef();
    *coder = c;
  }
  return S_OK;
  
  COM_TRY_END
}

static HRESULT CreateCoder2(bool encode, UInt32 index, const GUID *iid, void **outObject)
{
  *outObject = NULL;

  const CCodecInfo &codec = *g_Codecs[index];

  if (encode ? !codec.CreateEncoder : !codec.CreateDecoder)
    return CLASS_E_CLASSNOTAVAILABLE;

  if (codec.IsFilter)
  {
    if (*iid != IID_ICompressFilter) return E_NOINTERFACE;
  }
  else if (codec.NumStreams != 1)
  {
    if (*iid != IID_ICompressCoder2) return E_NOINTERFACE;
  }
  else
  {
    if (*iid != IID_ICompressCoder) return E_NOINTERFACE;
  }
  
  return CreateCoderMain(index, encode, outObject);
}


STDAPI CreateDecoder(UInt32 index, const GUID *iid, void **outObject);
STDAPI CreateDecoder(UInt32 index, const GUID *iid, void **outObject)
{
  return CreateCoder2(false, index, iid, outObject);
}


STDAPI CreateEncoder(UInt32 index, const GUID *iid, void **outObject);
STDAPI CreateEncoder(UInt32 index, const GUID *iid, void **outObject)
{
  return CreateCoder2(true, index, iid, outObject);
}


STDAPI CreateCoder(const GUID *clsid, const GUID *iid, void **outObject);
STDAPI CreateCoder(const GUID *clsid, const GUID *iid, void **outObject)
{
  *outObject = NULL;

  bool isFilter = false;
  bool isCoder2 = false;
  const bool isCoder = (*iid == IID_ICompressCoder) != 0;
  if (!isCoder)
  {
    isFilter = (*iid == IID_ICompressFilter) != 0;
    if (!isFilter)
    {
      isCoder2 = (*iid == IID_ICompressCoder2) != 0;
      if (!isCoder2)
        return E_NOINTERFACE;
    }
  }
  
  bool encode;
  int codecIndex;
  const HRESULT res = FindCodecClassId(clsid, isCoder2, isFilter, encode, codecIndex);
  if (res != S_OK)
    return res;
  if (codecIndex < 0)
    return CLASS_E_CLASSNOTAVAILABLE;

  return CreateCoderMain((unsigned)codecIndex, encode, outObject);
}
 

STDAPI GetMethodProperty(UInt32 codecIndex, PROPID propID, PROPVARIANT *value);
STDAPI GetMethodProperty(UInt32 codecIndex, PROPID propID, PROPVARIANT *value)
{
  ::VariantClear((VARIANTARG *)value);
  const CCodecInfo &codec = *g_Codecs[codecIndex];
  switch (propID)
  {
    case NMethodPropID::kID:
      value->uhVal.QuadPart = (UInt64)codec.Id;
      value->vt = VT_UI8;
      break;
    case NMethodPropID::kName:
      SetPropFromAscii(codec.Name, value);
      break;
    case NMethodPropID::kDecoder:
      if (codec.CreateDecoder)
        return MethodToClassID(k_7zip_GUID_Data3_Decoder, codec.Id, value);
      break;
    case NMethodPropID::kEncoder:
      if (codec.CreateEncoder)
        return MethodToClassID(k_7zip_GUID_Data3_Encoder, codec.Id, value);
      break;
    case NMethodPropID::kDecoderIsAssigned:
        value->vt = VT_BOOL;
        value->boolVal = BoolToVARIANT_BOOL(codec.CreateDecoder != NULL);
      break;
    case NMethodPropID::kEncoderIsAssigned:
        value->vt = VT_BOOL;
        value->boolVal = BoolToVARIANT_BOOL(codec.CreateEncoder != NULL);
      break;
    case NMethodPropID::kPackStreams:
      if (codec.NumStreams != 1)
      {
        value->vt = VT_UI4;
        value->ulVal = (ULONG)codec.NumStreams;
      }
      break;
    case NMethodPropID::kIsFilter:
      {
        value->vt = VT_BOOL;
        value->boolVal = BoolToVARIANT_BOOL(codec.IsFilter);
      }
      break;
    /*
    case NMethodPropID::kDecoderFlags:
      {
        value->vt = VT_UI4;
        value->ulVal = (ULONG)codec.DecoderFlags;
      }
      break;
    case NMethodPropID::kEncoderFlags:
      {
        value->vt = VT_UI4;
        value->ulVal = (ULONG)codec.EncoderFlags;
      }
      break;
    */
  }
  return S_OK;
}


STDAPI GetNumberOfMethods(UInt32 *numCodecs);
STDAPI GetNumberOfMethods(UInt32 *numCodecs)
{
  *numCodecs = g_NumCodecs;
  return S_OK;
}


// ---------- Hashers ----------

static int FindHasherClassId(const GUID *clsid) throw()
{
  if (clsid->Data1 != k_7zip_GUID_Data1 ||
      clsid->Data2 != k_7zip_GUID_Data2 ||
      clsid->Data3 != k_7zip_GUID_Data3_Hasher)
    return -1;
  const UInt64 id = GetUi64(clsid->Data4);
  for (unsigned i = 0; i < g_NumCodecs; i++)
    if (id == g_Hashers[i]->Id)
      return (int)i;
  return -1;
}

static HRESULT CreateHasher2(UInt32 index, IHasher **hasher)
{
  COM_TRY_BEGIN
  *hasher = g_Hashers[index]->CreateHasher();
  if (*hasher)
    (*hasher)->AddRef();
  return S_OK;
  COM_TRY_END
}

STDAPI CreateHasher(const GUID *clsid, IHasher **outObject);
STDAPI CreateHasher(const GUID *clsid, IHasher **outObject)
{
  COM_TRY_BEGIN
  *outObject = NULL;
  const int index = FindHasherClassId(clsid);
  if (index < 0)
    return CLASS_E_CLASSNOTAVAILABLE;
  return CreateHasher2((UInt32)(unsigned)index, outObject);
  COM_TRY_END
}

STDAPI GetHasherProp(UInt32 codecIndex, PROPID propID, PROPVARIANT *value);
STDAPI GetHasherProp(UInt32 codecIndex, PROPID propID, PROPVARIANT *value)
{
  ::VariantClear((VARIANTARG *)value);
  const CHasherInfo &codec = *g_Hashers[codecIndex];
  switch (propID)
  {
    case NMethodPropID::kID:
      value->uhVal.QuadPart = (UInt64)codec.Id;
      value->vt = VT_UI8;
      break;
    case NMethodPropID::kName:
      SetPropFromAscii(codec.Name, value);
      break;
    case NMethodPropID::kEncoder:
      if (codec.CreateHasher)
        return MethodToClassID(k_7zip_GUID_Data3_Hasher, codec.Id, value);
      break;
    case NMethodPropID::kDigestSize:
      value->ulVal = (ULONG)codec.DigestSize;
      value->vt = VT_UI4;
      break;
  }
  return S_OK;
}

Z7_CLASS_IMP_COM_1(CHashers, IHashers) };

STDAPI GetHashers(IHashers **hashers);
STDAPI GetHashers(IHashers **hashers)
{
  COM_TRY_BEGIN
  *hashers = new CHashers;
  if (*hashers)
    (*hashers)->AddRef();
  return S_OK;
  COM_TRY_END
}

Z7_COM7F_IMF2(UInt32, CHashers::GetNumHashers())
{
  return g_NumHashers;
}

Z7_COM7F_IMF(CHashers::GetHasherProp(UInt32 index, PROPID propID, PROPVARIANT *value))
{
  return ::GetHasherProp(index, propID, value);
}

Z7_COM7F_IMF(CHashers::CreateHasher(UInt32 index, IHasher **hasher))
{
  return ::CreateHasher2(index, hasher);
}


STDAPI GetModuleProp(PROPID propID, PROPVARIANT *value);
STDAPI GetModuleProp(PROPID propID, PROPVARIANT *value)
{
  ::VariantClear((VARIANTARG *)value);
  switch (propID)
  {
    case NModulePropID::kInterfaceType:
    {
      NWindows::NCOM::PropVarEm_Set_UInt32(value, NModuleInterfaceType::k_IUnknown_VirtDestructor_ThisModule);
      break;
    }
    case NModulePropID::kVersion:
    {
      NWindows::NCOM::PropVarEm_Set_UInt32(value, (MY_VER_MAJOR << 16) | MY_VER_MINOR);
      break;
    }
  }
  return S_OK;
}
