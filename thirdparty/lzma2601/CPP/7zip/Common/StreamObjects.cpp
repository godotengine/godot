// StreamObjects.cpp

#include "StdAfx.h"

#include "../../../C/Alloc.h"

#include "StreamObjects.h"

Z7_COM7F_IMF(CBufferInStream::Read(void *data, UInt32 size, UInt32 *processedSize))
{
  if (processedSize)
    *processedSize = 0;
  if (size == 0)
    return S_OK;
  if (_pos >= Buf.Size())
    return S_OK;
  size_t rem = Buf.Size() - (size_t)_pos;
  if (rem > size)
    rem = (size_t)size;
  memcpy(data, (const Byte *)Buf + (size_t)_pos, rem);
  _pos += rem;
  if (processedSize)
    *processedSize = (UInt32)rem;
  return S_OK;
}

Z7_COM7F_IMF(CBufferInStream::Seek(Int64 offset, UInt32 seekOrigin, UInt64 *newPosition))
{
  switch (seekOrigin)
  {
    case STREAM_SEEK_SET: break;
    case STREAM_SEEK_CUR: offset += _pos; break;
    case STREAM_SEEK_END: offset += Buf.Size(); break;
    default: return STG_E_INVALIDFUNCTION;
  }
  if (offset < 0)
    return HRESULT_WIN32_ERROR_NEGATIVE_SEEK;
  _pos = (UInt64)offset;
  if (newPosition)
    *newPosition = (UInt64)offset;
  return S_OK;
}

Z7_COM7F_IMF(CBufInStream::Read(void *data, UInt32 size, UInt32 *processedSize))
{
  if (processedSize)
    *processedSize = 0;
  if (size == 0)
    return S_OK;
  if (_pos >= _size)
    return S_OK;
  size_t rem = _size - (size_t)_pos;
  if (rem > size)
    rem = (size_t)size;
  memcpy(data, _data + (size_t)_pos, rem);
  _pos += rem;
  if (processedSize)
    *processedSize = (UInt32)rem;
  return S_OK;
}

Z7_COM7F_IMF(CBufInStream::Seek(Int64 offset, UInt32 seekOrigin, UInt64 *newPosition))
{
  switch (seekOrigin)
  {
    case STREAM_SEEK_SET: break;
    case STREAM_SEEK_CUR: offset += _pos; break;
    case STREAM_SEEK_END: offset += _size; break;
    default: return STG_E_INVALIDFUNCTION;
  }
  if (offset < 0)
    return HRESULT_WIN32_ERROR_NEGATIVE_SEEK;
  _pos = (UInt64)offset;
  if (newPosition)
    *newPosition = (UInt64)offset;
  return S_OK;
}

void Create_BufInStream_WithReference(const void *data, size_t size, IUnknown *ref, ISequentialInStream **stream)
{
  *stream = NULL;
  CBufInStream *inStreamSpec = new CBufInStream;
  CMyComPtr<ISequentialInStream> streamTemp = inStreamSpec;
  inStreamSpec->Init((const Byte *)data, size, ref);
  *stream = streamTemp.Detach();
}

void Create_BufInStream_WithNewBuffer(const void *data, size_t size, ISequentialInStream **stream)
{
  *stream = NULL;
  CBufferInStream *inStreamSpec = new CBufferInStream;
  CMyComPtr<ISequentialInStream> streamTemp = inStreamSpec;
  inStreamSpec->Buf.CopyFrom((const Byte *)data, size);
  inStreamSpec->Init();
  *stream = streamTemp.Detach();
}

void CByteDynBuffer::Free() throw()
{
  MyFree(_buf);
  _buf = NULL;
  _capacity = 0;
}

bool CByteDynBuffer::EnsureCapacity(size_t cap) throw()
{
  if (cap <= _capacity)
    return true;
  const size_t cap2 = _capacity + _capacity / 4;
  if (cap < cap2)
    cap = cap2;
  Byte *buf = (Byte *)MyRealloc(_buf, cap);
  if (!buf)
    return false;
  _buf = buf;
  _capacity = cap;
  return true;
}

Byte *CDynBufSeqOutStream::GetBufPtrForWriting(size_t addSize)
{
  addSize += _size;
  if (addSize < _size)
    return NULL;
  if (!_buffer.EnsureCapacity(addSize))
    return NULL;
  return (Byte *)_buffer + _size;
}

void CDynBufSeqOutStream::CopyToBuffer(CByteBuffer &dest) const
{
  dest.CopyFrom((const Byte *)_buffer, _size);
}

Z7_COM7F_IMF(CDynBufSeqOutStream::Write(const void *data, UInt32 size, UInt32 *processedSize))
{
  if (processedSize)
    *processedSize = 0;
  if (size == 0)
    return S_OK;
  Byte *buf = GetBufPtrForWriting(size);
  if (!buf)
    return E_OUTOFMEMORY;
  memcpy(buf, data, size);
  UpdateSize(size);
  if (processedSize)
    *processedSize = size;
  return S_OK;
}

Z7_COM7F_IMF(CBufPtrSeqOutStream::Write(const void *data, UInt32 size, UInt32 *processedSize))
{
  size_t rem = _size - _pos;
  if (rem > size)
    rem = (size_t)size;
  if (rem != 0)
  {
    memcpy(_buffer + _pos, data, rem);
    _pos += rem;
  }
  if (processedSize)
    *processedSize = (UInt32)rem;
  return (rem != 0 || size == 0) ? S_OK : E_FAIL;
}

Z7_COM7F_IMF(CSequentialOutStreamSizeCount::Write(const void *data, UInt32 size, UInt32 *processedSize))
{
  UInt32 realProcessedSize;
  HRESULT result = _stream->Write(data, size, &realProcessedSize);
  _size += realProcessedSize;
  if (processedSize)
    *processedSize = realProcessedSize;
  return result;
}

static const UInt64 kEmptyTag = (UInt64)(Int64)-1;

void CCachedInStream::Free() throw()
{
  MyFree(_tags);
  _tags = NULL;
  MidFree(_data);
  _data = NULL;
}

bool CCachedInStream::Alloc(unsigned blockSizeLog, unsigned numBlocksLog) throw()
{
  unsigned sizeLog = blockSizeLog + numBlocksLog;
  if (sizeLog >= sizeof(size_t) * 8)
    return false;
  size_t dataSize = (size_t)1 << sizeLog;
  if (!_data || dataSize != _dataSize)
  {
    MidFree(_data);
    _data = (Byte *)MidAlloc(dataSize);
    if (!_data)
      return false;
    _dataSize = dataSize;
  }
  if (!_tags || numBlocksLog != _numBlocksLog)
  {
    MyFree(_tags);
    _tags = (UInt64 *)MyAlloc(sizeof(UInt64) << numBlocksLog);
    if (!_tags)
      return false;
    _numBlocksLog = numBlocksLog;
  }
  _blockSizeLog = blockSizeLog;
  return true;
}

void CCachedInStream::Init(UInt64 size) throw()
{
  _size = size;
  _pos = 0;
  const size_t numBlocks = (size_t)1 << _numBlocksLog;
  for (size_t i = 0; i < numBlocks; i++)
    _tags[i] = kEmptyTag;
}

Z7_COM7F_IMF(CCachedInStream::Read(void *data, UInt32 size, UInt32 *processedSize))
{
  if (processedSize)
    *processedSize = 0;
  if (size == 0)
    return S_OK;
  if (_pos >= _size)
    return S_OK;

  {
    const UInt64 rem = _size - _pos;
    if (size > rem)
      size = (UInt32)rem;
  }

  while (size != 0)
  {
    const UInt64 cacheTag = _pos >> _blockSizeLog;
    const size_t cacheIndex = (size_t)cacheTag & (((size_t)1 << _numBlocksLog) - 1);
    Byte *p = _data + (cacheIndex << _blockSizeLog);

    if (_tags[cacheIndex] != cacheTag)
    {
      _tags[cacheIndex] = kEmptyTag;
      const UInt64 remInBlock = _size - (cacheTag << _blockSizeLog);
      size_t blockSize = (size_t)1 << _blockSizeLog;
      if (blockSize > remInBlock)
        blockSize = (size_t)remInBlock;
      
      RINOK(ReadBlock(cacheTag, p, blockSize))
      
      _tags[cacheIndex] = cacheTag;
    }
    
    const size_t kBlockSize = (size_t)1 << _blockSizeLog;
    const size_t offset = (size_t)_pos & (kBlockSize - 1);
    UInt32 cur = size;
    const size_t rem = kBlockSize - offset;
    if (cur > rem)
      cur = (UInt32)rem;
    
    memcpy(data, p + offset, cur);
    
    if (processedSize)
      *processedSize += cur;
    data = (void *)((const Byte *)data + cur);
    _pos += cur;
    size -= cur;
  }

  return S_OK;
}

 
Z7_COM7F_IMF(CCachedInStream::Seek(Int64 offset, UInt32 seekOrigin, UInt64 *newPosition))
{
  switch (seekOrigin)
  {
    case STREAM_SEEK_SET: break;
    case STREAM_SEEK_CUR: offset += _pos; break;
    case STREAM_SEEK_END: offset += _size; break;
    default: return STG_E_INVALIDFUNCTION;
  }
  if (offset < 0)
    return HRESULT_WIN32_ERROR_NEGATIVE_SEEK;
  _pos = (UInt64)offset;
  if (newPosition)
    *newPosition = (UInt64)offset;
  return S_OK;
}
