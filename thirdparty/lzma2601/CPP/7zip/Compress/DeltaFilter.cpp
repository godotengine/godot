// DeltaFilter.cpp

#include "StdAfx.h"

#include "../../../C/Delta.h"

#include "../../Common/MyCom.h"

#include "../ICoder.h"

#include "../Common/RegisterCodec.h"

namespace NCompress {
namespace NDelta {

struct CDelta
{
  unsigned _delta;
  Byte _state[DELTA_STATE_SIZE];

  CDelta(): _delta(1) {}
  void DeltaInit() { Delta_Init(_state); }
};


#ifndef Z7_EXTRACT_ONLY

class CEncoder Z7_final:
  public ICompressFilter,
  public ICompressSetCoderProperties,
  public ICompressWriteCoderProperties,
  public CMyUnknownImp,
  CDelta
{
  Z7_IFACES_IMP_UNK_3(
      ICompressFilter,
      ICompressSetCoderProperties,
      ICompressWriteCoderProperties)
};

Z7_COM7F_IMF(CEncoder::Init())
{
  DeltaInit();
  return S_OK;
}

Z7_COM7F_IMF2(UInt32, CEncoder::Filter(Byte *data, UInt32 size))
{
  Delta_Encode(_state, _delta, data, size);
  return size;
}

Z7_COM7F_IMF(CEncoder::SetCoderProperties(const PROPID *propIDs, const PROPVARIANT *props, UInt32 numProps))
{
  unsigned delta = _delta;
  for (UInt32 i = 0; i < numProps; i++)
  {
    const PROPVARIANT &prop = props[i];
    const PROPID propID = propIDs[i];
    if (propID >= NCoderPropID::kReduceSize)
      continue;
    if (prop.vt != VT_UI4)
      return E_INVALIDARG;
    switch (propID)
    {
      case NCoderPropID::kDefaultProp:
        if (prop.ulVal < 1 || prop.ulVal > 256)
          return E_INVALIDARG;
        delta = prop.ulVal;
        break;
      case NCoderPropID::kNumThreads: break;
      case NCoderPropID::kLevel: break;
      default: return E_INVALIDARG;
    }
  }
  _delta = delta;
  return S_OK;
}

Z7_COM7F_IMF(CEncoder::WriteCoderProperties(ISequentialOutStream *outStream))
{
  const Byte prop = (Byte)(_delta - 1);
  return outStream->Write(&prop, 1, NULL);
}

#endif


class CDecoder Z7_final:
  public ICompressFilter,
  public ICompressSetDecoderProperties2,
  public CMyUnknownImp,
  CDelta
{
  Z7_IFACES_IMP_UNK_2(
      ICompressFilter,
      ICompressSetDecoderProperties2)
};

Z7_COM7F_IMF(CDecoder::Init())
{
  DeltaInit();
  return S_OK;
}

Z7_COM7F_IMF2(UInt32, CDecoder::Filter(Byte *data, UInt32 size))
{
  Delta_Decode(_state, _delta, data, size);
  return size;
}

Z7_COM7F_IMF(CDecoder::SetDecoderProperties2(const Byte *props, UInt32 size))
{
  if (size != 1)
    return E_INVALIDARG;
  _delta = (unsigned)props[0] + 1;
  return S_OK;
}


REGISTER_FILTER_E(Delta,
    CDecoder(),
    CEncoder(),
    3, "Delta")

}}
