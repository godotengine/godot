// OutStreamWithCRC.cpp

#include "StdAfx.h"

#include "OutStreamWithCRC.h"

Z7_COM7F_IMF(COutStreamWithCRC::Write(const void *data, UInt32 size, UInt32 *processedSize))
{
  HRESULT result = S_OK;
  if (_stream)
    result = _stream->Write(data, size, &size);
  if (_calculate)
    _crc = CrcUpdate(_crc, data, size);
  _size += size;
  if (processedSize)
    *processedSize = size;
  return result;
}
