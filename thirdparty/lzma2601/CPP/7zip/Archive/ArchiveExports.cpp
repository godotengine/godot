// ArchiveExports.cpp

#include "StdAfx.h"

#include "../../../C/7zVersion.h"

#include "../../Common/ComTry.h"

#include "../../Windows/PropVariant.h"

#include "../Common/RegisterArc.h"

static const unsigned kNumArcsMax = 72;
static unsigned g_NumArcs = 0;
static unsigned g_DefaultArcIndex = 0;
static const CArcInfo *g_Arcs[kNumArcsMax];

void RegisterArc(const CArcInfo *arcInfo) throw()
{
  if (g_NumArcs < kNumArcsMax)
  {
    const char *p = arcInfo->Name;
    if (p[0] == '7' && p[1] == 'z' && p[2] == 0)
      g_DefaultArcIndex = g_NumArcs;
    g_Arcs[g_NumArcs++] = arcInfo;
  }
  // else throw 1;
}

Z7_DEFINE_GUID(CLSID_CArchiveHandler,
    k_7zip_GUID_Data1,
    k_7zip_GUID_Data2,
    k_7zip_GUID_Data3_Common,
    0x10, 0x00, 0x00, 0x01, 0x10, 0x00, 0x00, 0x00);

#define CLS_ARC_ID_ITEM(cls) ((cls).Data4[5])

static inline HRESULT SetPropStrFromBin(const char *s, unsigned size, PROPVARIANT *value)
{
  if ((value->bstrVal = ::SysAllocStringByteLen(s, size)) != NULL)
    value->vt = VT_BSTR;
  return S_OK;
}

static inline HRESULT SetPropGUID(const GUID &guid, PROPVARIANT *value)
{
  return SetPropStrFromBin((const char *)&guid, sizeof(guid), value);
}

static int FindFormatCalssId(const GUID *clsid)
{
  GUID cls = *clsid;
  CLS_ARC_ID_ITEM(cls) = 0;
  if (cls != CLSID_CArchiveHandler)
    return -1;
  const Byte id = CLS_ARC_ID_ITEM(*clsid);
  for (unsigned i = 0; i < g_NumArcs; i++)
    if (g_Arcs[i]->Id == id)
      return (int)i;
  return -1;
}

STDAPI CreateArchiver(const GUID *clsid, const GUID *iid, void **outObject);
STDAPI CreateArchiver(const GUID *clsid, const GUID *iid, void **outObject)
{
  COM_TRY_BEGIN
  {
    const int needIn = (*iid == IID_IInArchive);
    const int needOut = (*iid == IID_IOutArchive);
    if (!needIn && !needOut)
      return E_NOINTERFACE;
    const int formatIndex = FindFormatCalssId(clsid);
    if (formatIndex < 0)
      return CLASS_E_CLASSNOTAVAILABLE;
    
    const CArcInfo &arc = *g_Arcs[formatIndex];
    if (needIn)
    {
      *outObject = arc.CreateInArchive();
      ((IInArchive *)*outObject)->AddRef();
    }
    else
    {
      if (!arc.CreateOutArchive)
        return CLASS_E_CLASSNOTAVAILABLE;
      *outObject = arc.CreateOutArchive();
      ((IOutArchive *)*outObject)->AddRef();
    }
  }
  COM_TRY_END
  return S_OK;
}

STDAPI GetHandlerProperty2(UInt32 formatIndex, PROPID propID, PROPVARIANT *value);
STDAPI GetHandlerProperty2(UInt32 formatIndex, PROPID propID, PROPVARIANT *value)
{
  COM_TRY_BEGIN
  NWindows::NCOM::PropVariant_Clear(value);
  if (formatIndex >= g_NumArcs)
    return E_INVALIDARG;
  const CArcInfo &arc = *g_Arcs[formatIndex];
  NWindows::NCOM::CPropVariant prop;
  switch (propID)
  {
    case NArchive::NHandlerPropID::kName: prop = arc.Name; break;
    case NArchive::NHandlerPropID::kClassID:
    {
      GUID clsId = CLSID_CArchiveHandler;
      CLS_ARC_ID_ITEM(clsId) = arc.Id;
      return SetPropGUID(clsId, value);
    }
    case NArchive::NHandlerPropID::kExtension: if (arc.Ext) prop = arc.Ext; break;
    case NArchive::NHandlerPropID::kAddExtension: if (arc.AddExt) prop = arc.AddExt; break;
    case NArchive::NHandlerPropID::kUpdate: prop = (bool)(arc.CreateOutArchive != NULL); break;
    case NArchive::NHandlerPropID::kKeepName:   prop = ((arc.Flags & NArcInfoFlags::kKeepName) != 0); break;
    case NArchive::NHandlerPropID::kAltStreams: prop = ((arc.Flags & NArcInfoFlags::kAltStreams) != 0); break;
    case NArchive::NHandlerPropID::kNtSecure:   prop = ((arc.Flags & NArcInfoFlags::kNtSecure) != 0); break;
    case NArchive::NHandlerPropID::kFlags: prop = (UInt32)arc.Flags; break;
    case NArchive::NHandlerPropID::kTimeFlags: prop = (UInt32)arc.TimeFlags; break;
    case NArchive::NHandlerPropID::kSignatureOffset: prop = (UInt32)arc.SignatureOffset; break;
    // case NArchive::NHandlerPropID::kVersion: prop = (UInt32)MY_VER_MIX; break;

    case NArchive::NHandlerPropID::kSignature:
      if (arc.SignatureSize != 0 && !arc.IsMultiSignature())
        return SetPropStrFromBin((const char *)arc.Signature, arc.SignatureSize, value);
      break;
    case NArchive::NHandlerPropID::kMultiSignature:
      if (arc.SignatureSize != 0 && arc.IsMultiSignature())
        return SetPropStrFromBin((const char *)arc.Signature, arc.SignatureSize, value);
      break;
  }
  prop.Detach(value);
  return S_OK;
  COM_TRY_END
}

STDAPI GetHandlerProperty(PROPID propID, PROPVARIANT *value);
STDAPI GetHandlerProperty(PROPID propID, PROPVARIANT *value)
{
  return GetHandlerProperty2(g_DefaultArcIndex, propID, value);
}

STDAPI GetNumberOfFormats(UINT32 *numFormats);
STDAPI GetNumberOfFormats(UINT32 *numFormats)
{
  *numFormats = g_NumArcs;
  return S_OK;
}

STDAPI GetIsArc(UInt32 formatIndex, Func_IsArc *isArc);
STDAPI GetIsArc(UInt32 formatIndex, Func_IsArc *isArc)
{
  *isArc = NULL;
  if (formatIndex >= g_NumArcs)
    return E_INVALIDARG;
  *isArc = g_Arcs[formatIndex]->IsArc;
  return S_OK;
}
