// LimitedStreams.cpp

#include "StdAfx.h"

#include <string.h>

#include "LimitedStreams.h"

Z7_COM7F_IMF(CLimitedSequentialInStream::Read(void *data, UInt32 size, UInt32 *processedSize))
{
  UInt32 realProcessedSize = 0;
  {
    const UInt64 rem = _size - _pos;
    if (size > rem)
      size = (UInt32)rem;
  }
  HRESULT result = S_OK;
  if (size != 0)
  {
    result = _stream->Read(data, size, &realProcessedSize);
    _pos += realProcessedSize;
    if (realProcessedSize == 0)
      _wasFinished = true;
  }
  if (processedSize)
    *processedSize = realProcessedSize;
  return result;
}

Z7_COM7F_IMF(CLimitedInStream::Read(void *data, UInt32 size, UInt32 *processedSize))
{
  if (processedSize)
    *processedSize = 0;
  if (_virtPos >= _size)
  {
    // 9.31: Fixed. Windows doesn't return error in ReadFile and IStream->Read in that case.
    return S_OK;
    // return (_virtPos == _size) ? S_OK: E_FAIL; // ERROR_HANDLE_EOF
  }
  {
    const UInt64 rem = _size - _virtPos;
    if (size > rem)
      size = (UInt32)rem;
  }
  UInt64 newPos = _startOffset + _virtPos;
  if (newPos != _physPos)
  {
    _physPos = newPos;
    RINOK(SeekToPhys())
  }
  HRESULT res = _stream->Read(data, size, &size);
  if (processedSize)
    *processedSize = size;
  _physPos += size;
  _virtPos += size;
  return res;
}

Z7_COM7F_IMF(CLimitedInStream::Seek(Int64 offset, UInt32 seekOrigin, UInt64 *newPosition))
{
  switch (seekOrigin)
  {
    case STREAM_SEEK_SET: break;
    case STREAM_SEEK_CUR: offset += _virtPos; break;
    case STREAM_SEEK_END: offset += _size; break;
    default: return STG_E_INVALIDFUNCTION;
  }
  if (offset < 0)
    return HRESULT_WIN32_ERROR_NEGATIVE_SEEK;
  _virtPos = (UInt64)offset;
  if (newPosition)
    *newPosition = _virtPos;
  return S_OK;
}

HRESULT CreateLimitedInStream(IInStream *inStream, UInt64 pos, UInt64 size, ISequentialInStream **resStream)
{
  *resStream = NULL;
  CLimitedInStream *streamSpec = new CLimitedInStream;
  CMyComPtr<ISequentialInStream> streamTemp = streamSpec;
  streamSpec->SetStream(inStream);
  RINOK(streamSpec->InitAndSeek(pos, size))
  streamSpec->SeekToStart();
  *resStream = streamTemp.Detach();
  return S_OK;
}

Z7_COM7F_IMF(CClusterInStream::Read(void *data, UInt32 size, UInt32 *processedSize))
{
  if (processedSize)
    *processedSize = 0;
  if (_virtPos >= Size)
    return S_OK;
  {
    UInt64 rem = Size - _virtPos;
    if (size > rem)
      size = (UInt32)rem;
  }
  if (size == 0)
    return S_OK;

  if (_curRem == 0)
  {
    const UInt32 blockSize = (UInt32)1 << BlockSizeLog;
    const UInt32 virtBlock = (UInt32)(_virtPos >> BlockSizeLog);
    const UInt32 offsetInBlock = (UInt32)_virtPos & (blockSize - 1);
    const UInt32 phyBlock = Vector[virtBlock];
    
    UInt64 newPos = StartOffset + ((UInt64)phyBlock << BlockSizeLog) + offsetInBlock;
    if (newPos != _physPos)
    {
      _physPos = newPos;
      RINOK(SeekToPhys())
    }

    _curRem = blockSize - offsetInBlock;
    
    for (unsigned i = 1; i < 64 && (virtBlock + i) < (UInt32)Vector.Size() && phyBlock + i == Vector[virtBlock + i]; i++)
      _curRem += (UInt32)1 << BlockSizeLog;
  }
  
  if (size > _curRem)
    size = _curRem;
  HRESULT res = Stream->Read(data, size, &size);
  if (processedSize)
    *processedSize = size;
  _physPos += size;
  _virtPos += size;
  _curRem -= size;
  return res;
}
 
Z7_COM7F_IMF(CClusterInStream::Seek(Int64 offset, UInt32 seekOrigin, UInt64 *newPosition))
{
  switch (seekOrigin)
  {
    case STREAM_SEEK_SET: break;
    case STREAM_SEEK_CUR: offset += _virtPos; break;
    case STREAM_SEEK_END: offset += Size; break;
    default: return STG_E_INVALIDFUNCTION;
  }
  if (offset < 0)
    return HRESULT_WIN32_ERROR_NEGATIVE_SEEK;
  if (_virtPos != (UInt64)offset)
    _curRem = 0;
  _virtPos = (UInt64)offset;
  if (newPosition)
    *newPosition = (UInt64)offset;
  return S_OK;
}


Z7_COM7F_IMF(CExtentsStream::Read(void *data, UInt32 size, UInt32 *processedSize))
{
  if (processedSize)
    *processedSize = 0;
  const UInt64 virt = _virtPos;
  if (virt >= Extents.Back().Virt)
    return S_OK;
  if (size == 0)
    return S_OK;
  
  unsigned left = _prevExtentIndex;
  if (virt <  Extents[left].Virt ||
      virt >= Extents[left + 1].Virt)
  {
    left = 0;
    unsigned right = Extents.Size() - 1;
    for (;;)
    {
      const unsigned mid = (unsigned)(((size_t)left + (size_t)right) / 2);
      if (mid == left)
        break;
      if (virt < Extents[mid].Virt)
        right = mid;
      else
        left = mid;
    }
    _prevExtentIndex = left;
  }
  
  {
    const UInt64 rem = Extents[left + 1].Virt - virt;
    if (size > rem)
      size = (UInt32)rem;
  }
  
  const CSeekExtent &extent = Extents[left];
  
  if (extent.Is_ZeroFill())
  {
    memset(data, 0, size);
    _virtPos += size;
    if (processedSize)
      *processedSize = size;
    return S_OK;
  }

  {
    const UInt64 phy = extent.Phy + (virt - extent.Virt);
    if (_phyPos != phy)
    {
      _phyPos = (UInt64)0 - 1;  // we don't trust seek_pos in case of error
      RINOK(InStream_SeekSet(Stream, phy))
      _phyPos = phy;
    }
  }
  
  const HRESULT res = Stream->Read(data, size, &size);
  _virtPos += size;
  if (res == S_OK)
    _phyPos += size;
  else
    _phyPos = (UInt64)0 - 1;  // we don't trust seek_pos in case of error
  if (processedSize)
    *processedSize = size;
  return res;
}


Z7_COM7F_IMF(CExtentsStream::Seek(Int64 offset, UInt32 seekOrigin, UInt64 *newPosition))
{
  switch (seekOrigin)
  {
    case STREAM_SEEK_SET: break;
    case STREAM_SEEK_CUR: offset += _virtPos; break;
    case STREAM_SEEK_END: offset += Extents.Back().Virt; break;
    default: return STG_E_INVALIDFUNCTION;
  }
  if (offset < 0)
    return HRESULT_WIN32_ERROR_NEGATIVE_SEEK;
  _virtPos = (UInt64)offset;
  if (newPosition)
    *newPosition = _virtPos;
  return S_OK;
}


Z7_COM7F_IMF(CLimitedSequentialOutStream::Write(const void *data, UInt32 size, UInt32 *processedSize))
{
  HRESULT result = S_OK;
  if (processedSize)
    *processedSize = 0;
  if (size > _size)
  {
    if (_size == 0)
    {
      _overflow = true;
      if (!_overflowIsAllowed)
        return E_FAIL;
      if (processedSize)
        *processedSize = size;
      return S_OK;
    }
    size = (UInt32)_size;
  }
  if (_stream)
    result = _stream->Write(data, size, &size);
  _size -= size;
  if (processedSize)
    *processedSize = size;
  return result;
}


Z7_COM7F_IMF(CTailInStream::Read(void *data, UInt32 size, UInt32 *processedSize))
{
  UInt32 cur;
  HRESULT res = Stream->Read(data, size, &cur);
  if (processedSize)
    *processedSize = cur;
  _virtPos += cur;
  return res;
}
  
Z7_COM7F_IMF(CTailInStream::Seek(Int64 offset, UInt32 seekOrigin, UInt64 *newPosition))
{
  switch (seekOrigin)
  {
    case STREAM_SEEK_SET: break;
    case STREAM_SEEK_CUR: offset += _virtPos; break;
    case STREAM_SEEK_END:
    {
      UInt64 pos = 0;
      RINOK(Stream->Seek(offset, STREAM_SEEK_END, &pos))
      if (pos < Offset)
        return HRESULT_WIN32_ERROR_NEGATIVE_SEEK;
      _virtPos = pos - Offset;
      if (newPosition)
        *newPosition = _virtPos;
      return S_OK;
    }
    default: return STG_E_INVALIDFUNCTION;
  }
  if (offset < 0)
    return HRESULT_WIN32_ERROR_NEGATIVE_SEEK;
  _virtPos = (UInt64)offset;
  if (newPosition)
    *newPosition = _virtPos;
  return InStream_SeekSet(Stream, Offset + _virtPos);
}

Z7_COM7F_IMF(CLimitedCachedInStream::Read(void *data, UInt32 size, UInt32 *processedSize))
{
  if (processedSize)
    *processedSize = 0;
  if (_virtPos >= _size)
  {
    // 9.31: Fixed. Windows doesn't return error in ReadFile and IStream->Read in that case.
    return S_OK;
    // return (_virtPos == _size) ? S_OK: E_FAIL; // ERROR_HANDLE_EOF
  }
  UInt64 rem = _size - _virtPos;
  if (rem < size)
    size = (UInt32)rem;

  UInt64 newPos = _startOffset + _virtPos;
  UInt64 offsetInCache = newPos - _cachePhyPos;
  HRESULT res = S_OK;
  if (newPos >= _cachePhyPos &&
      offsetInCache <= _cacheSize &&
      size <= _cacheSize - (size_t)offsetInCache)
  {
    if (size != 0)
      memcpy(data, _cache + (size_t)offsetInCache, size);
  }
  else
  {
    if (newPos != _physPos)
    {
      _physPos = newPos;
      RINOK(SeekToPhys())
    }
    res = _stream->Read(data, size, &size);
    _physPos += size;
  }
  if (processedSize)
    *processedSize = size;
  _virtPos += size;
  return res;
}

Z7_COM7F_IMF(CLimitedCachedInStream::Seek(Int64 offset, UInt32 seekOrigin, UInt64 *newPosition))
{
  switch (seekOrigin)
  {
    case STREAM_SEEK_SET: break;
    case STREAM_SEEK_CUR: offset += _virtPos; break;
    case STREAM_SEEK_END: offset += _size; break;
    default: return STG_E_INVALIDFUNCTION;
  }
  if (offset < 0)
    return HRESULT_WIN32_ERROR_NEGATIVE_SEEK;
  _virtPos = (UInt64)offset;
  if (newPosition)
    *newPosition = _virtPos;
  return S_OK;
}

Z7_COM7F_IMF(CTailOutStream::Write(const void *data, UInt32 size, UInt32 *processedSize))
{
  UInt32 cur;
  HRESULT res = Stream->Write(data, size, &cur);
  if (processedSize)
    *processedSize = cur;
  _virtPos += cur;
  if (_virtSize < _virtPos)
    _virtSize = _virtPos;
  return res;
}
  
Z7_COM7F_IMF(CTailOutStream::Seek(Int64 offset, UInt32 seekOrigin, UInt64 *newPosition))
{
  switch (seekOrigin)
  {
    case STREAM_SEEK_SET: break;
    case STREAM_SEEK_CUR: offset += _virtPos; break;
    case STREAM_SEEK_END: offset += _virtSize; break;
    default: return STG_E_INVALIDFUNCTION;
  }
  if (offset < 0)
    return HRESULT_WIN32_ERROR_NEGATIVE_SEEK;
  _virtPos = (UInt64)offset;
  if (newPosition)
    *newPosition = _virtPos;
  return Stream->Seek((Int64)(Offset + _virtPos), STREAM_SEEK_SET, NULL);
}

Z7_COM7F_IMF(CTailOutStream::SetSize(UInt64 newSize))
{
  _virtSize = newSize;
  return Stream->SetSize(Offset + newSize);
}
