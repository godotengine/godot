// 7zSpecStream.cpp

#include "StdAfx.h"

#include "7zSpecStream.h"

/*
Z7_COM7F_IMF(CSequentialInStreamSizeCount2::Read(void *data, UInt32 size, UInt32 *processedSize))
{
  UInt32 realProcessedSize;
  const HRESULT result = _stream->Read(data, size, &realProcessedSize);
  _size += realProcessedSize;
  if (processedSize)
    *processedSize = realProcessedSize;
  return result;
}

Z7_COM7F_IMF(CSequentialInStreamSizeCount2::GetSubStreamSize(UInt64 subStream, UInt64 *value))
{
  if (!_getSubStreamSize)
    return E_NOTIMPL;
  return _getSubStreamSize->GetSubStreamSize(subStream, value);
}

Z7_COM7F_IMF(CSequentialInStreamSizeCount2::GetNextInSubStream(UInt64 *streamIndexRes, ISequentialInStream **stream))
{
  if (!_compressGetSubStreamSize)
    return E_NOTIMPL;
  return _compressGetSubStreamSize->GetNextInSubStream(streamIndexRes, stream);
}
*/
