// PpmdEncoder.cpp

#include "StdAfx.h"

#include "../../../C/Alloc.h"

#include "../Common/StreamUtils.h"

#include "PpmdEncoder.h"

namespace NCompress {
namespace NPpmd {

static const UInt32 kBufSize = (1 << 20);

static const Byte kOrders[10] = { 3, 4, 4, 5, 5, 6, 8, 16, 24, 32 };

void CEncProps::Normalize(int level)
{
  if (level < 0) level = 5;
  if (level > 9) level = 9;
  if (MemSize == (UInt32)(Int32)-1)
    MemSize = (UInt32)1 << (level + 19);
  const unsigned kMult = 16;
  if (MemSize / kMult > ReduceSize)
  {
    for (unsigned i = 16; i < 32; i++)
    {
      UInt32 m = (UInt32)1 << i;
      if (ReduceSize <= m / kMult)
      {
        if (MemSize > m)
          MemSize = m;
        break;
      }
    }
  }
  if (Order == -1) Order = kOrders[(unsigned)level];
}

CEncoder::CEncoder():
  _inBuf(NULL)
{
  _props.Normalize(-1);
  Ppmd7_Construct(&_ppmd);
  _ppmd.rc.enc.Stream = &_outStream.vt;
}

CEncoder::~CEncoder()
{
  ::MidFree(_inBuf);
  Ppmd7_Free(&_ppmd, &g_BigAlloc);
}

Z7_COM7F_IMF(CEncoder::SetCoderProperties(const PROPID *propIDs, const PROPVARIANT *coderProps, UInt32 numProps))
{
  int level = -1;
  CEncProps props;
  for (UInt32 i = 0; i < numProps; i++)
  {
    const PROPVARIANT &prop = coderProps[i];
    const PROPID propID = propIDs[i];
    if (propID > NCoderPropID::kReduceSize)
      continue;
    if (propID == NCoderPropID::kReduceSize)
    {
      if (prop.vt == VT_UI8 && prop.uhVal.QuadPart < (UInt32)(Int32)-1)
        props.ReduceSize = (UInt32)prop.uhVal.QuadPart;
      continue;
    }

    if (propID == NCoderPropID::kUsedMemorySize)
    {
      // here we have selected (4 GiB - 1 KiB) as replacement for (4 GiB) MEM_SIZE.
      const UInt32 kPpmd_Default_4g = (UInt32)0 - ((UInt32)1 << 10);
      UInt32 v;
      if (prop.vt == VT_UI8)
      {
        // 21.03 : we support 64-bit values (for 4 GiB value)
        const UInt64 v64 = prop.uhVal.QuadPart;
        if (v64 > ((UInt64)1 << 32))
          return E_INVALIDARG;
        if (v64 == ((UInt64)1 << 32))
          v = kPpmd_Default_4g;
        else
          v = (UInt32)v64;
      }
      else if (prop.vt == VT_UI4)
        v = (UInt32)prop.ulVal;
      else
        return E_INVALIDARG;
      if (v > PPMD7_MAX_MEM_SIZE)
        v = kPpmd_Default_4g;

      /* here we restrict MEM_SIZE for Encoder.
         It's for better performance of encoding and decoding.
         The Decoder still supports more MEM_SIZE values. */
      if (v < ((UInt32)1 << 16) || (v & 3) != 0)
        return E_INVALIDARG;
      // if (v < PPMD7_MIN_MEM_SIZE) return E_INVALIDARG; // (1 << 11)
      /*
        Supported MEM_SIZE range :
        [ (1 << 11) , 0xFFFFFFFF - 12 * 3 ] - current 7-Zip's Ppmd7 constants
        [ 1824      , 0xFFFFFFFF          ] - real limits of Ppmd7 code
      */
      props.MemSize = v;
      continue;
    }

    if (prop.vt != VT_UI4)
      return E_INVALIDARG;
    const UInt32 v = (UInt32)prop.ulVal;
    switch (propID)
    {
      case NCoderPropID::kOrder:
        if (v < 2 || v > 32)
          return E_INVALIDARG;
        props.Order = (Byte)v;
        break;
      case NCoderPropID::kNumThreads: break;
      case NCoderPropID::kLevel: level = (int)v; break;
      default: return E_INVALIDARG;
    }
  }
  props.Normalize(level);
  _props = props;
  return S_OK;
}

Z7_COM7F_IMF(CEncoder::WriteCoderProperties(ISequentialOutStream *outStream))
{
  const UInt32 kPropSize = 5;
  Byte props[kPropSize];
  props[0] = (Byte)_props.Order;
  SetUi32(props + 1, _props.MemSize)
  return WriteStream(outStream, props, kPropSize);
}

Z7_COM7F_IMF(CEncoder::Code(ISequentialInStream *inStream, ISequentialOutStream *outStream,
    const UInt64 * /* inSize */, const UInt64 * /* outSize */, ICompressProgressInfo *progress))
{
  if (!_inBuf)
  {
    _inBuf = (Byte *)::MidAlloc(kBufSize);
    if (!_inBuf)
      return E_OUTOFMEMORY;
  }
  if (!_outStream.Alloc(1 << 20))
    return E_OUTOFMEMORY;
  if (!Ppmd7_Alloc(&_ppmd, _props.MemSize, &g_BigAlloc))
    return E_OUTOFMEMORY;

  _outStream.Stream = outStream;
  _outStream.Init();

  Ppmd7z_Init_RangeEnc(&_ppmd);
  Ppmd7_Init(&_ppmd, (unsigned)_props.Order);

  UInt64 processed = 0;
  for (;;)
  {
    UInt32 size;
    RINOK(inStream->Read(_inBuf, kBufSize, &size))
    if (size == 0)
    {
      // We don't write EndMark in PPMD-7z.
      // Ppmd7z_EncodeSymbol(&_ppmd, -1);
      Ppmd7z_Flush_RangeEnc(&_ppmd);
      return _outStream.Flush();
    }
    const Byte *buf = _inBuf;
    const Byte *lim = buf + size;
    /*
    for (; buf < lim; buf++)
    {
      Ppmd7z_EncodeSymbol(&_ppmd, *buf);
      RINOK(_outStream.Res);
    }
    */

    Ppmd7z_EncodeSymbols(&_ppmd, buf, lim);
    RINOK(_outStream.Res)

    processed += size;
    if (progress)
    {
      const UInt64 outSize = _outStream.GetProcessed();
      RINOK(progress->SetRatioInfo(&processed, &outSize))
    }
  }
}

}}
