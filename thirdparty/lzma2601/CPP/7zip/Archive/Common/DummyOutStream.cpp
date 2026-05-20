// DummyOutStream.cpp

#include "StdAfx.h"

#include "DummyOutStream.h"

Z7_COM7F_IMF(CDummyOutStream::Write(const void *data, UInt32 size, UInt32 *processedSize))
{
  UInt32 realProcessedSize = size;
  HRESULT res = S_OK;
  if (_stream)
    res = _stream->Write(data, size, &realProcessedSize);
  _size += realProcessedSize;
  if (processedSize)
    *processedSize = realProcessedSize;
  return res;
}
