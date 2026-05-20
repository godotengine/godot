// PpmdDecoder.cpp

#include "StdAfx.h"

#include "../../../C/Alloc.h"
#include "../../../C/CpuArch.h"

#include "../Common/StreamUtils.h"

#include "PpmdDecoder.h"

namespace NCompress {
namespace NPpmd {

static const UInt32 kBufSize = (1 << 16);

enum
{
  kStatus_NeedInit,
  kStatus_Normal,
  kStatus_Finished_With_Mark,
  kStatus_Error
};

CDecoder::~CDecoder()
{
  ::MidFree(_outBuf);
  Ppmd7_Free(&_ppmd, &g_BigAlloc);
}

Z7_COM7F_IMF(CDecoder::SetDecoderProperties2(const Byte *props, UInt32 size))
{
  if (size < 5)
    return E_INVALIDARG;
  _order = props[0];
  const UInt32 memSize = GetUi32(props + 1);
  if (_order < PPMD7_MIN_ORDER ||
      _order > PPMD7_MAX_ORDER ||
      memSize < PPMD7_MIN_MEM_SIZE ||
      memSize > PPMD7_MAX_MEM_SIZE)
    return E_NOTIMPL;
  if (!_inStream.Alloc(1 << 20))
    return E_OUTOFMEMORY;
  if (!Ppmd7_Alloc(&_ppmd, memSize, &g_BigAlloc))
    return E_OUTOFMEMORY;
  return S_OK;
}

#define MY_rangeDec  _ppmd.rc.dec

#define CHECK_EXTRA_ERROR \
    if (_inStream.Extra) { \
      _status = kStatus_Error; \
      return (_res = (_inStream.Res != SZ_OK ? _inStream.Res: S_FALSE)); }


HRESULT CDecoder::CodeSpec(Byte *memStream, UInt32 size)
{
  if (_res != S_OK)
    return _res;
  
  switch (_status)
  {
    case kStatus_Finished_With_Mark: return S_OK;
    case kStatus_Error: return S_FALSE;
    case kStatus_NeedInit:
      _inStream.Init();
      if (!Ppmd7z_RangeDec_Init(&MY_rangeDec))
      {
        _status = kStatus_Error;
        return (_res = S_FALSE);
      }
      CHECK_EXTRA_ERROR
      _status = kStatus_Normal;
      Ppmd7_Init(&_ppmd, _order);
      break;
    default: break;
  }
  
  if (_outSizeDefined)
  {
    const UInt64 rem = _outSize - _processedSize;
    if (size > rem)
      size = (UInt32)rem;
  }

  int sym = 0;
  {
    Byte *buf = memStream;
    const Byte *lim = buf + size;
    for (; buf != lim; buf++)
    {
      sym = Ppmd7z_DecodeSymbol(&_ppmd);
      if (_inStream.Extra || sym < 0)
        break;
      *buf = (Byte)sym;
    }
    /*
    buf = Ppmd7z_DecodeSymbols(&_ppmd, buf, lim);
    sym = _ppmd.LastSymbol;
    */
    _processedSize += (size_t)(buf - memStream);
  }

  CHECK_EXTRA_ERROR
  
  if (sym >= 0)
  {
    if (!FinishStream
        || !_outSizeDefined
        || _outSize != _processedSize
        || MY_rangeDec.Code == 0)
      return S_OK;
    /*
    // We can decode additional End Marker here:
    sym = Ppmd7z_DecodeSymbol(&_ppmd);
    CHECK_EXTRA_ERROR
    */
  }

  if (sym != PPMD7_SYM_END || MY_rangeDec.Code != 0)
  {
    _status = kStatus_Error;
    return (_res = S_FALSE);
  }
  
  _status = kStatus_Finished_With_Mark;
  return S_OK;
}



Z7_COM7F_IMF(CDecoder::Code(ISequentialInStream *inStream, ISequentialOutStream *outStream,
    const UInt64 *inSize, const UInt64 *outSize, ICompressProgressInfo *progress))
{
  if (!_outBuf)
  {
    _outBuf = (Byte *)::MidAlloc(kBufSize);
    if (!_outBuf)
      return E_OUTOFMEMORY;
  }
  
  _inStream.Stream = inStream;
  SetOutStreamSize(outSize);

  do
  {
    const UInt64 startPos = _processedSize;
    const HRESULT res = CodeSpec(_outBuf, kBufSize);
    const size_t processed = (size_t)(_processedSize - startPos);
    RINOK(WriteStream(outStream, _outBuf, processed))
    RINOK(res)
    if (_status == kStatus_Finished_With_Mark)
      break;
    if (progress)
    {
      const UInt64 inProcessed = _inStream.GetProcessed();
      RINOK(progress->SetRatioInfo(&inProcessed, &_processedSize))
    }
  }
  while (!_outSizeDefined || _processedSize < _outSize);

  if (FinishStream && inSize && *inSize != _inStream.GetProcessed())
    return S_FALSE;

  return S_OK;
}


Z7_COM7F_IMF(CDecoder::SetOutStreamSize(const UInt64 *outSize))
{
  _outSizeDefined = (outSize != NULL);
  if (_outSizeDefined)
    _outSize = *outSize;
  _processedSize = 0;
  _status = kStatus_NeedInit;
  _res = SZ_OK;
  return S_OK;
}

Z7_COM7F_IMF(CDecoder::SetFinishMode(UInt32 finishMode))
{
  FinishStream = (finishMode != 0);
  return S_OK;
}

Z7_COM7F_IMF(CDecoder::GetInStreamProcessedSize(UInt64 *value))
{
  *value = _inStream.GetProcessed();
  return S_OK;
}

#ifndef Z7_NO_READ_FROM_CODER

Z7_COM7F_IMF(CDecoder::SetInStream(ISequentialInStream *inStream))
{
  InSeqStream = inStream;
  _inStream.Stream = inStream;
  return S_OK;
}

Z7_COM7F_IMF(CDecoder::ReleaseInStream())
{
  InSeqStream.Release();
  return S_OK;
}

Z7_COM7F_IMF(CDecoder::Read(void *data, UInt32 size, UInt32 *processedSize))
{
  const UInt64 startPos = _processedSize;
  const HRESULT res = CodeSpec((Byte *)data, size);
  if (processedSize)
    *processedSize = (UInt32)(_processedSize - startPos);
  return res;
}

#endif

}}
