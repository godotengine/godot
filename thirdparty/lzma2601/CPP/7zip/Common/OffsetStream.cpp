// OffsetStream.cpp

#include "StdAfx.h"

#include "OffsetStream.h"

HRESULT COffsetOutStream::Init(IOutStream *stream, UInt64 offset)
{
  _offset = offset;
  _stream = stream;
  return _stream->Seek((Int64)offset, STREAM_SEEK_SET, NULL);
}

Z7_COM7F_IMF(COffsetOutStream::Write(const void *data, UInt32 size, UInt32 *processedSize))
{
  return _stream->Write(data, size, processedSize);
}

Z7_COM7F_IMF(COffsetOutStream::Seek(Int64 offset, UInt32 seekOrigin, UInt64 *newPosition))
{
  if (seekOrigin == STREAM_SEEK_SET)
  {
    if (offset < 0)
      return HRESULT_WIN32_ERROR_NEGATIVE_SEEK;
    offset += _offset;
  }
  UInt64 absoluteNewPosition = 0; // =0 for gcc-10
  const HRESULT result = _stream->Seek(offset, seekOrigin, &absoluteNewPosition);
  if (newPosition)
    *newPosition = absoluteNewPosition - _offset;
  return result;
}

Z7_COM7F_IMF(COffsetOutStream::SetSize(UInt64 newSize))
{
  return _stream->SetSize(_offset + newSize);
}
