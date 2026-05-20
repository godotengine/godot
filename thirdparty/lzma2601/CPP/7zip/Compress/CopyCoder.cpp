// Compress/CopyCoder.cpp

#include "StdAfx.h"

#include "../../../C/Alloc.h"

#include "CopyCoder.h"

namespace NCompress {

static const UInt32 kBufSize = 1 << 17;

CCopyCoder::~CCopyCoder()
{
  ::MidFree(_buf);
}

Z7_COM7F_IMF(CCopyCoder::SetFinishMode(UInt32 /* finishMode */))
{
  return S_OK;
}

Z7_COM7F_IMF(CCopyCoder::Code(ISequentialInStream *inStream,
    ISequentialOutStream *outStream,
    const UInt64 * /* inSize */, const UInt64 *outSize,
    ICompressProgressInfo *progress))
{
  if (!_buf)
  {
    _buf = (Byte *)::MidAlloc(kBufSize);
    if (!_buf)
      return E_OUTOFMEMORY;
  }

  TotalSize = 0;
  
  for (;;)
  {
    UInt32 size = kBufSize;
    if (outSize)
    {
      const UInt64 rem = *outSize - TotalSize;
      if (size > rem)
      {
        size = (UInt32)rem;
        if (size == 0)
        {
          /* if we enable the following check,
             we will make one call of Read(_buf, 0) for empty stream */
          // if (TotalSize != 0)
          return S_OK;
        }
      }
    }
    
    HRESULT readRes;
    {
      UInt32 pos = 0;
      do
      {
        const UInt32 curSize = size - pos;
        UInt32 processed = 0;
        readRes = inStream->Read(_buf + pos, curSize, &processed);
        if (processed > curSize)
          return E_FAIL; // internal code failure
        pos += processed;
        if (readRes != S_OK || processed == 0)
          break;
      }
      while (pos < kBufSize);
      size = pos;
    }

    if (size == 0)
      return readRes;

    if (outStream)
    {
      UInt32 pos = 0;
      do
      {
        const UInt32 curSize = size - pos;
        UInt32 processed = 0;
        const HRESULT res = outStream->Write(_buf + pos, curSize, &processed);
        if (processed > curSize)
          return E_FAIL; // internal code failure
        pos += processed;
        TotalSize += processed;
        RINOK(res)
        if (processed == 0)
          return E_FAIL;
      }
      while (pos < size);
    }
    else
      TotalSize += size;

    RINOK(readRes)

    if (size != kBufSize)
      return S_OK;

    if (progress && (TotalSize & (((UInt32)1 << 22) - 1)) == 0)
    {
      RINOK(progress->SetRatioInfo(&TotalSize, &TotalSize))
    }
  }
}

Z7_COM7F_IMF(CCopyCoder::SetInStream(ISequentialInStream *inStream))
{
  _inStream = inStream;
  TotalSize = 0;
  return S_OK;
}

Z7_COM7F_IMF(CCopyCoder::ReleaseInStream())
{
  _inStream.Release();
  return S_OK;
}

Z7_COM7F_IMF(CCopyCoder::Read(void *data, UInt32 size, UInt32 *processedSize))
{
  UInt32 realProcessedSize = 0;
  HRESULT res = _inStream->Read(data, size, &realProcessedSize);
  TotalSize += realProcessedSize;
  if (processedSize)
    *processedSize = realProcessedSize;
  return res;
}

Z7_COM7F_IMF(CCopyCoder::GetInStreamProcessedSize(UInt64 *value))
{
  *value = TotalSize;
  return S_OK;
}

HRESULT CopyStream(ISequentialInStream *inStream, ISequentialOutStream *outStream, ICompressProgressInfo *progress)
{
  CMyComPtr<ICompressCoder> copyCoder = new CCopyCoder;
  return copyCoder->Code(inStream, outStream, NULL, NULL, progress);
}

HRESULT CopyStream_ExactSize(ISequentialInStream *inStream, ISequentialOutStream *outStream, UInt64 size, ICompressProgressInfo *progress)
{
  NCompress::CCopyCoder *copyCoderSpec = new NCompress::CCopyCoder;
  CMyComPtr<ICompressCoder> copyCoder = copyCoderSpec;
  RINOK(copyCoder->Code(inStream, outStream, NULL, &size, progress))
  return copyCoderSpec->TotalSize == size ? S_OK : E_FAIL;
}

}
