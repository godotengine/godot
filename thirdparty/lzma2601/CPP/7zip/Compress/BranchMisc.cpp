// BranchMisc.cpp

#include "StdAfx.h"

#include "../../../C/CpuArch.h"

#include "../Common/StreamUtils.h"

#include "BranchMisc.h"

namespace NCompress {
namespace NBranch {

Z7_COM7F_IMF(CCoder::Init())
{
  _pc = 0;
  return S_OK;
}


Z7_COM7F_IMF2(UInt32, CCoder::Filter(Byte *data, UInt32 size))
{
  const UInt32 processed = (UInt32)(size_t)(BraFunc(data, size, _pc) - data);
  _pc += processed;
  return processed;
}


#ifndef Z7_EXTRACT_ONLY

Z7_COM7F_IMF(CEncoder::Init())
{
  _pc = _pc_Init;
  return S_OK;
}

Z7_COM7F_IMF2(UInt32, CEncoder::Filter(Byte *data, UInt32 size))
{
  const UInt32 processed = (UInt32)(size_t)(BraFunc(data, size, _pc) - data);
  _pc += processed;
  return processed;
}

Z7_COM7F_IMF(CEncoder::SetCoderProperties(const PROPID *propIDs, const PROPVARIANT *props, UInt32 numProps))
{
  UInt32 pc = 0;
  for (UInt32 i = 0; i < numProps; i++)
  {
    const PROPID propID = propIDs[i];
    if (propID == NCoderPropID::kDefaultProp ||
        propID == NCoderPropID::kBranchOffset)
    {
      const PROPVARIANT &prop = props[i];
      if (prop.vt != VT_UI4)
        return E_INVALIDARG;
      pc = prop.ulVal;
      if (pc & _alignment)
        return E_INVALIDARG;
    }
  }
  _pc_Init = pc;
  return S_OK;
}


Z7_COM7F_IMF(CEncoder::WriteCoderProperties(ISequentialOutStream *outStream))
{
  if (_pc_Init == 0)
    return S_OK;
  UInt32 buf32[1];
  SetUi32(buf32, _pc_Init)
  return WriteStream(outStream, buf32, 4);
}

#endif


Z7_COM7F_IMF(CDecoder::Init())
{
  _pc = _pc_Init;
  return S_OK;
}

Z7_COM7F_IMF2(UInt32, CDecoder::Filter(Byte *data, UInt32 size))
{
  const UInt32 processed = (UInt32)(size_t)(BraFunc(data, size, _pc) - data);
  _pc += processed;
  return processed;
}

Z7_COM7F_IMF(CDecoder::SetDecoderProperties2(const Byte *props, UInt32 size))
{
  UInt32 val = 0;
  if (size != 0)
  {
    if (size != 4)
      return E_NOTIMPL;
    val = GetUi32(props);
    if (val & _alignment)
      return E_NOTIMPL;
  }
  _pc_Init = val;
  return S_OK;
}

}}
