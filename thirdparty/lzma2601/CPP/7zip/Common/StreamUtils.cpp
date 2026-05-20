// StreamUtils.cpp

#include "StdAfx.h"

#include "../../Common/MyCom.h"

#include "StreamUtils.h"

static const UInt32 kBlockSize = ((UInt32)1 << 31);


HRESULT InStream_SeekToBegin(IInStream *stream) throw()
{
  return InStream_SeekSet(stream, 0);
}


HRESULT InStream_AtBegin_GetSize(IInStream *stream, UInt64 &sizeRes) throw()
{
#ifdef _WIN32
  {
    Z7_DECL_CMyComPtr_QI_FROM(
        IStreamGetSize,
        streamGetSize, stream)
    if (streamGetSize && streamGetSize->GetSize(&sizeRes) == S_OK)
      return S_OK;
  }
#endif
  const HRESULT hres = InStream_GetSize_SeekToEnd(stream, sizeRes);
  const HRESULT hres2 = InStream_SeekToBegin(stream);
  return hres != S_OK ? hres : hres2;
}


HRESULT InStream_GetPos_GetSize(IInStream *stream, UInt64 &curPosRes, UInt64 &sizeRes) throw()
{
  RINOK(InStream_GetPos(stream, curPosRes))
#ifdef _WIN32
  {
    Z7_DECL_CMyComPtr_QI_FROM(
        IStreamGetSize,
        streamGetSize, stream)
    if (streamGetSize && streamGetSize->GetSize(&sizeRes) == S_OK)
      return S_OK;
  }
#endif
  const HRESULT hres = InStream_GetSize_SeekToEnd(stream, sizeRes);
  const HRESULT hres2 = InStream_SeekSet(stream, curPosRes);
  return hres != S_OK ? hres : hres2;
}



HRESULT ReadStream(ISequentialInStream *stream, void *data, size_t *processedSize) throw()
{
  size_t size = *processedSize;
  *processedSize = 0;
  while (size != 0)
  {
    UInt32 curSize = (size < kBlockSize) ? (UInt32)size : kBlockSize;
    UInt32 processedSizeLoc;
    HRESULT res = stream->Read(data, curSize, &processedSizeLoc);
    *processedSize += processedSizeLoc;
    data = (void *)((Byte *)data + processedSizeLoc);
    size -= processedSizeLoc;
    RINOK(res)
    if (processedSizeLoc == 0)
      return S_OK;
  }
  return S_OK;
}

HRESULT ReadStream_FALSE(ISequentialInStream *stream, void *data, size_t size) throw()
{
  size_t processedSize = size;
  RINOK(ReadStream(stream, data, &processedSize))
  return (size == processedSize) ? S_OK : S_FALSE;
}

HRESULT ReadStream_FAIL(ISequentialInStream *stream, void *data, size_t size) throw()
{
  size_t processedSize = size;
  RINOK(ReadStream(stream, data, &processedSize))
  return (size == processedSize) ? S_OK : E_FAIL;
}

HRESULT WriteStream(ISequentialOutStream *stream, const void *data, size_t size) throw()
{
  while (size != 0)
  {
    UInt32 curSize = (size < kBlockSize) ? (UInt32)size : kBlockSize;
    UInt32 processedSizeLoc;
    HRESULT res = stream->Write(data, curSize, &processedSizeLoc);
    data = (const void *)((const Byte *)data + processedSizeLoc);
    size -= processedSizeLoc;
    RINOK(res)
    if (processedSizeLoc == 0)
      return E_FAIL;
  }
  return S_OK;
}
