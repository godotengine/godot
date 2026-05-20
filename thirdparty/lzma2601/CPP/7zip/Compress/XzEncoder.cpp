// XzEncoder.cpp

#include "StdAfx.h"

#include "../../../C/Alloc.h"

#include "../../Common/MyString.h"
#include "../../Common/StringToInt.h"

#include "../Common/CWrappers.h"
#include "../Common/StreamUtils.h"

#include "XzEncoder.h"

namespace NCompress {

namespace NLzma2 {
HRESULT SetLzma2Prop(PROPID propID, const PROPVARIANT &prop, CLzma2EncProps &lzma2Props);
}

namespace NXz {

void CEncoder::InitCoderProps()
{
  XzProps_Init(&xzProps);
}

CEncoder::CEncoder()
{
  XzProps_Init(&xzProps);
  _encoder = NULL;
  _encoder = XzEnc_Create(&g_Alloc, &g_BigAlloc);
  if (!_encoder)
    throw 1;
}

CEncoder::~CEncoder()
{
  if (_encoder)
    XzEnc_Destroy(_encoder);
}


struct CMethodNamePair
{
  UInt32 Id;
  const char *Name;
};

static const CMethodNamePair g_NamePairs[] =
{
  { XZ_ID_Delta, "Delta" },
  { XZ_ID_X86, "BCJ" },
  { XZ_ID_PPC, "PPC" },
  { XZ_ID_IA64, "IA64" },
  { XZ_ID_ARM, "ARM" },
  { XZ_ID_ARMT, "ARMT" },
  { XZ_ID_SPARC, "SPARC" }
  // { XZ_ID_LZMA2, "LZMA2" }
};

static int FilterIdFromName(const wchar_t *name)
{
  for (unsigned i = 0; i < Z7_ARRAY_SIZE(g_NamePairs); i++)
  {
    const CMethodNamePair &pair = g_NamePairs[i];
    if (StringsAreEqualNoCase_Ascii(name, pair.Name))
      return (int)pair.Id;
  }
  return -1;
}


HRESULT CEncoder::SetCheckSize(UInt32 checkSizeInBytes)
{
  unsigned id;
  switch (checkSizeInBytes)
  {
    case  0: id = XZ_CHECK_NO; break;
    case  4: id = XZ_CHECK_CRC32; break;
    case  8: id = XZ_CHECK_CRC64; break;
    case 32: id = XZ_CHECK_SHA256; break;
    default: return E_INVALIDARG;
  }
  xzProps.checkId = id;
  return S_OK;
}


HRESULT CEncoder::SetCoderProp(PROPID propID, const PROPVARIANT &prop)
{
  if (propID == NCoderPropID::kNumThreads)
  {
    if (prop.vt != VT_UI4)
      return E_INVALIDARG;
    xzProps.numTotalThreads = (int)(prop.ulVal);
    return S_OK;
  }

  if (propID == NCoderPropID::kCheckSize)
  {
    if (prop.vt != VT_UI4)
      return E_INVALIDARG;
    return SetCheckSize(prop.ulVal);
  }

  if (propID == NCoderPropID::kBlockSize2)
  {
    if (prop.vt == VT_UI4)
      xzProps.blockSize = prop.ulVal;
    else if (prop.vt == VT_UI8)
      xzProps.blockSize = prop.uhVal.QuadPart;
    else
      return E_INVALIDARG;
    return S_OK;
  }

  if (propID == NCoderPropID::kReduceSize)
  {
    if (prop.vt == VT_UI8)
      xzProps.reduceSize = prop.uhVal.QuadPart;
    else
      return E_INVALIDARG;
    return S_OK;
  }
 
  if (propID == NCoderPropID::kFilter)
  {
    if (prop.vt == VT_UI4)
    {
      const UInt32 id32 = prop.ulVal;
      if (id32 == XZ_ID_Delta)
        return E_INVALIDARG;
      xzProps.filterProps.id = prop.ulVal;
    }
    else
    {
      if (prop.vt != VT_BSTR)
        return E_INVALIDARG;
      
      const wchar_t *name = prop.bstrVal;
      const wchar_t *end;

      UInt32 id32 = ConvertStringToUInt32(name, &end);
      
      if (end != name)
        name = end;
      else
      {
        if (IsString1PrefixedByString2_NoCase_Ascii(name, "Delta"))
        {
          name += 5; // strlen("Delta");
          id32 = XZ_ID_Delta;
        }
        else
        {
          const int filterId = FilterIdFromName(prop.bstrVal);
          if (filterId < 0 /* || filterId == XZ_ID_LZMA2 */)
            return E_INVALIDARG;
          id32 = (UInt32)(unsigned)filterId;
        }
      }
      
      if (id32 == XZ_ID_Delta)
      {
        const wchar_t c = *name;
        if (c != '-' && c != ':')
          return E_INVALIDARG;
        name++;
        const UInt32 delta = ConvertStringToUInt32(name, &end);
        if (end == name || *end != 0 || delta == 0 || delta > 256)
          return E_INVALIDARG;
        xzProps.filterProps.delta = delta;
      }
      
      xzProps.filterProps.id = id32;
    }
    
    return S_OK;
  }

  return NLzma2::SetLzma2Prop(propID, prop, xzProps.lzma2Props);
}


Z7_COM7F_IMF(CEncoder::SetCoderProperties(const PROPID *propIDs,
    const PROPVARIANT *coderProps, UInt32 numProps))
{
  XzProps_Init(&xzProps);

  for (UInt32 i = 0; i < numProps; i++)
  {
    RINOK(SetCoderProp(propIDs[i], coderProps[i]))
  }
  
  return S_OK;
  // return SResToHRESULT(XzEnc_SetProps(_encoder, &xzProps));
}


Z7_COM7F_IMF(CEncoder::SetCoderPropertiesOpt(const PROPID *propIDs,
    const PROPVARIANT *coderProps, UInt32 numProps))
{
  for (UInt32 i = 0; i < numProps; i++)
  {
    const PROPVARIANT &prop = coderProps[i];
    const PROPID propID = propIDs[i];
    if (propID == NCoderPropID::kExpectedDataSize)
      if (prop.vt == VT_UI8)
        XzEnc_SetDataSize(_encoder, prop.uhVal.QuadPart);
  }
  return S_OK;
}


#define RET_IF_WRAP_ERROR(wrapRes, sRes, sResErrorCode) \
  if (wrapRes != S_OK /* && (sRes == SZ_OK || sRes == sResErrorCode) */) return wrapRes;

Z7_COM7F_IMF(CEncoder::Code(ISequentialInStream *inStream, ISequentialOutStream *outStream,
    const UInt64 * /* inSize */, const UInt64 * /* outSize */, ICompressProgressInfo *progress))
{
  CSeqInStreamWrap inWrap;
  CSeqOutStreamWrap outWrap;
  CCompressProgressWrap progressWrap;

  inWrap.Init(inStream);
  outWrap.Init(outStream);
  progressWrap.Init(progress);

  SRes res = XzEnc_SetProps(_encoder, &xzProps);
  if (res == SZ_OK)
    res = XzEnc_Encode(_encoder, &outWrap.vt, &inWrap.vt, progress ? &progressWrap.vt : NULL);

  // SRes res = Xz_Encode(&outWrap.vt, &inWrap.vt, &xzProps, progress ? &progressWrap.vt : NULL);

  RET_IF_WRAP_ERROR(inWrap.Res, res, SZ_ERROR_READ)
  RET_IF_WRAP_ERROR(outWrap.Res, res, SZ_ERROR_WRITE)
  RET_IF_WRAP_ERROR(progressWrap.Res, res, SZ_ERROR_PROGRESS)

  return SResToHRESULT(res);
}
  
}}
