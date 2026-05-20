// LzmaDecoder.cpp

#include "StdAfx.h"

#include "../../../C/Alloc.h"

#include "../Common/StreamUtils.h"

#include "LzmaDecoder.h"

static HRESULT SResToHRESULT(SRes res)
{
  switch (res)
  {
    case SZ_OK: return S_OK;
    case SZ_ERROR_MEM: return E_OUTOFMEMORY;
    case SZ_ERROR_PARAM: return E_INVALIDARG;
    case SZ_ERROR_UNSUPPORTED: return E_NOTIMPL;
    case SZ_ERROR_DATA: return S_FALSE;
    default: break;
  }
  return E_FAIL;
}

namespace NCompress {
namespace NLzma {

CDecoder::CDecoder():
    FinishStream(false),
    _propsWereSet(false),
    _outSizeDefined(false),
    _outStep(1 << 20),
    _inBufSize(0),
    _inBufSizeNew(1 << 20),
    _lzmaStatus(LZMA_STATUS_NOT_SPECIFIED),
    _inBuf(NULL)
{
  _inProcessed = 0;
  _inPos = _inLim = 0;

  /*
  AlignOffsetAlloc_CreateVTable(&_alloc);
  _alloc.numAlignBits = 7;
  _alloc.offset = 0;
  */
  LzmaDec_CONSTRUCT(&_state)
}

CDecoder::~CDecoder()
{
  LzmaDec_Free(&_state, &g_AlignedAlloc); // &_alloc.vt
  MyFree(_inBuf);
}

Z7_COM7F_IMF(CDecoder::SetInBufSize(UInt32 , UInt32 size))
  { _inBufSizeNew = size; return S_OK; }
Z7_COM7F_IMF(CDecoder::SetOutBufSize(UInt32 , UInt32 size))
  { _outStep = size; return S_OK; }

HRESULT CDecoder::CreateInputBuffer()
{
  if (!_inBuf || _inBufSizeNew != _inBufSize)
  {
    MyFree(_inBuf);
    _inBufSize = 0;
    _inBuf = (Byte *)MyAlloc(_inBufSizeNew);
    if (!_inBuf)
      return E_OUTOFMEMORY;
    _inBufSize = _inBufSizeNew;
  }
  return S_OK;
}


Z7_COM7F_IMF(CDecoder::SetDecoderProperties2(const Byte *prop, UInt32 size))
{
  RINOK(SResToHRESULT(LzmaDec_Allocate(&_state, prop, size, &g_AlignedAlloc))) // &_alloc.vt
  _propsWereSet = true;
  return CreateInputBuffer();
}


void CDecoder::SetOutStreamSizeResume(const UInt64 *outSize)
{
  _outSizeDefined = (outSize != NULL);
  _outSize = 0;
  if (_outSizeDefined)
    _outSize = *outSize;
  _outProcessed = 0;
  _lzmaStatus = LZMA_STATUS_NOT_SPECIFIED;

  LzmaDec_Init(&_state);
}


Z7_COM7F_IMF(CDecoder::SetOutStreamSize(const UInt64 *outSize))
{
  _inProcessed = 0;
  _inPos = _inLim = 0;
  SetOutStreamSizeResume(outSize);
  return S_OK;
}


Z7_COM7F_IMF(CDecoder::SetFinishMode(UInt32 finishMode))
{
  FinishStream = (finishMode != 0);
  return S_OK;
}


Z7_COM7F_IMF(CDecoder::GetInStreamProcessedSize(UInt64 *value))
{
  *value = _inProcessed;
  return S_OK;
}


HRESULT CDecoder::CodeSpec(ISequentialInStream *inStream, ISequentialOutStream *outStream, ICompressProgressInfo *progress)
{
  if (!_inBuf || !_propsWereSet)
    return S_FALSE;
  
  const UInt64 startInProgress = _inProcessed;
  SizeT wrPos = _state.dicPos;
  HRESULT readRes = S_OK;

  for (;;)
  {
    if (_inPos == _inLim && readRes == S_OK)
    {
      _inPos = _inLim = 0;
      readRes = inStream->Read(_inBuf, _inBufSize, &_inLim);
    }

    const SizeT dicPos = _state.dicPos;
    SizeT size;
    {
      SizeT next = _state.dicBufSize;
      if (next - wrPos > _outStep)
        next = wrPos + _outStep;
      size = next - dicPos;
    }

    ELzmaFinishMode finishMode = LZMA_FINISH_ANY;
    if (_outSizeDefined)
    {
      const UInt64 rem = _outSize - _outProcessed;
      if (size >= rem)
      {
        size = (SizeT)rem;
        if (FinishStream)
          finishMode = LZMA_FINISH_END;
      }
    }

    SizeT inProcessed = _inLim - _inPos;
    ELzmaStatus status;

    const SRes res = LzmaDec_DecodeToDic(&_state, dicPos + size, _inBuf + _inPos, &inProcessed, finishMode, &status);

    _lzmaStatus = status;
    _inPos += (UInt32)inProcessed;
    _inProcessed += inProcessed;
    const SizeT outProcessed = _state.dicPos - dicPos;
    _outProcessed += outProcessed;

    // we check for LZMA_STATUS_NEEDS_MORE_INPUT to allow RangeCoder initialization, if (_outSizeDefined && _outSize == 0)
    const bool outFinished = (_outSizeDefined && _outProcessed >= _outSize);

    const bool needStop = (res != 0
        || (inProcessed == 0 && outProcessed == 0)
        || status == LZMA_STATUS_FINISHED_WITH_MARK
        || (outFinished && status != LZMA_STATUS_NEEDS_MORE_INPUT));

    if (needStop || outProcessed >= size)
    {
      const HRESULT res2 = WriteStream(outStream, _state.dic + wrPos, _state.dicPos - wrPos);

      if (_state.dicPos == _state.dicBufSize)
        _state.dicPos = 0;
      wrPos = _state.dicPos;
      
      RINOK(res2)

      if (needStop)
      {
        if (res != 0)
        {
          // return SResToHRESULT(res);
          return S_FALSE;
        }

        if (status == LZMA_STATUS_FINISHED_WITH_MARK)
        {
          if (FinishStream)
            if (_outSizeDefined && _outSize != _outProcessed)
              return S_FALSE;
          return readRes;
        }
        
        if (outFinished && status != LZMA_STATUS_NEEDS_MORE_INPUT)
          if (!FinishStream || status == LZMA_STATUS_MAYBE_FINISHED_WITHOUT_MARK)
            return readRes;
          
        return S_FALSE;
      }
    }
    
    if (progress)
    {
      const UInt64 inSize = _inProcessed - startInProgress;
      RINOK(progress->SetRatioInfo(&inSize, &_outProcessed))
    }
  }
}


Z7_COM7F_IMF(CDecoder::Code(ISequentialInStream *inStream, ISequentialOutStream *outStream,
    const UInt64 *inSize, const UInt64 *outSize, ICompressProgressInfo *progress))
{
  if (!_inBuf)
    return E_INVALIDARG;
  SetOutStreamSize(outSize);
  HRESULT res = CodeSpec(inStream, outStream, progress);
  if (res == S_OK)
    if (FinishStream && inSize && *inSize != _inProcessed)
      res = S_FALSE;
  return res;
}


#ifndef Z7_NO_READ_FROM_CODER

Z7_COM7F_IMF(CDecoder::SetInStream(ISequentialInStream *inStream))
  { _inStream = inStream; return S_OK; }
Z7_COM7F_IMF(CDecoder::ReleaseInStream())
  { _inStream.Release(); return S_OK; }

Z7_COM7F_IMF(CDecoder::Read(void *data, UInt32 size, UInt32 *processedSize))
{
  if (processedSize)
    *processedSize = 0;

  ELzmaFinishMode finishMode = LZMA_FINISH_ANY;
  if (_outSizeDefined)
  {
    const UInt64 rem = _outSize - _outProcessed;
    if (size >= rem)
    {
      size = (UInt32)rem;
      if (FinishStream)
        finishMode = LZMA_FINISH_END;
    }
  }

  HRESULT readRes = S_OK;
  
  for (;;)
  {
    if (_inPos == _inLim && readRes == S_OK)
    {
      _inPos = _inLim = 0;
      readRes = _inStream->Read(_inBuf, _inBufSize, &_inLim);
    }

    SizeT inProcessed = _inLim - _inPos;
    SizeT outProcessed = size;
    ELzmaStatus status;
    
    const SRes res = LzmaDec_DecodeToBuf(&_state, (Byte *)data, &outProcessed,
        _inBuf + _inPos, &inProcessed, finishMode, &status);
    
    _lzmaStatus = status;
    _inPos += (UInt32)inProcessed;
    _inProcessed += inProcessed;
    _outProcessed += outProcessed;
    size -= (UInt32)outProcessed;
    data = (Byte *)data + outProcessed;
    if (processedSize)
      *processedSize += (UInt32)outProcessed;

    if (res != 0)
      return S_FALSE;
    
    /*
    if (status == LZMA_STATUS_FINISHED_WITH_MARK)
      return readRes;

    if (size == 0 && status != LZMA_STATUS_NEEDS_MORE_INPUT)
    {
      if (FinishStream
          && _outSizeDefined && _outProcessed >= _outSize
          && status != LZMA_STATUS_MAYBE_FINISHED_WITHOUT_MARK)
        return S_FALSE;
      return readRes;
    }
    */

    if (inProcessed == 0 && outProcessed == 0)
      return readRes;
  }
}


HRESULT CDecoder::CodeResume(ISequentialOutStream *outStream, const UInt64 *outSize, ICompressProgressInfo *progress)
{
  SetOutStreamSizeResume(outSize);
  return CodeSpec(_inStream, outStream, progress);
}


HRESULT CDecoder::ReadFromInputStream(void *data, UInt32 size, UInt32 *processedSize)
{
  RINOK(CreateInputBuffer())
  
  if (processedSize)
    *processedSize = 0;

  HRESULT readRes = S_OK;

  while (size != 0)
  {
    if (_inPos == _inLim)
    {
      _inPos = _inLim = 0;
      if (readRes == S_OK)
        readRes = _inStream->Read(_inBuf, _inBufSize, &_inLim);
      if (_inLim == 0)
        break;
    }

    UInt32 cur = _inLim - _inPos;
    if (cur > size)
      cur = size;
    memcpy(data, _inBuf + _inPos, cur);
    _inPos += cur;
    _inProcessed += cur;
    size -= cur;
    data = (Byte *)data + cur;
    if (processedSize)
      *processedSize += cur;
  }
  
  return readRes;
}

#endif

}}
