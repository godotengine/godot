// InBuffer.cpp

#include "StdAfx.h"

#include "../../../C/Alloc.h"

#include "InBuffer.h"

CInBufferBase::CInBufferBase() throw():
  _buf(NULL),
  _bufLim(NULL),
  _bufBase(NULL),
  _stream(NULL),
  _processedSize(0),
  _bufSize(0),
  _wasFinished(false),
  NumExtraBytes(0)
{}

bool CInBuffer::Create(size_t bufSize) throw()
{
  const unsigned kMinBlockSize = 1;
  if (bufSize < kMinBlockSize)
    bufSize = kMinBlockSize;
  if (_bufBase != NULL && _bufSize == bufSize)
    return true;
  Free();
  _bufSize = bufSize;
  _bufBase = (Byte *)::MidAlloc(bufSize);
  return (_bufBase != NULL);
}

void CInBuffer::Free() throw()
{
  ::MidFree(_bufBase);
  _bufBase = NULL;
}

void CInBufferBase::Init() throw()
{
  _processedSize = 0;
  _buf = _bufBase;
  _bufLim = _buf;
  _wasFinished = false;
  #ifdef Z7_NO_EXCEPTIONS
  ErrorCode = S_OK;
  #endif
  NumExtraBytes = 0;
}

bool CInBufferBase::ReadBlock()
{
  #ifdef Z7_NO_EXCEPTIONS
  if (ErrorCode != S_OK)
    return false;
  #endif
  if (_wasFinished)
    return false;
  _processedSize += (size_t)(_buf - _bufBase);
  _buf = _bufBase;
  _bufLim = _bufBase;
  UInt32 processed;
  // FIX_ME: we can improve it to support (_bufSize >= (1 << 32))
  const HRESULT result = _stream->Read(_bufBase, (UInt32)_bufSize, &processed);
  #ifdef Z7_NO_EXCEPTIONS
  ErrorCode = result;
  #else
  if (result != S_OK)
    throw CInBufferException(result);
  #endif
  _bufLim = _buf + processed;
  _wasFinished = (processed == 0);
  return !_wasFinished;
}

bool CInBufferBase::ReadByte_FromNewBlock(Byte &b)
{
  if (!ReadBlock())
  {
    // 22.00: we don't increment (NumExtraBytes) here
    // NumExtraBytes++;
    b = 0xFF;
    return false;
  }
  b = *_buf++;
  return true;
}

Byte CInBufferBase::ReadByte_FromNewBlock()
{
  if (!ReadBlock())
  {
    NumExtraBytes++;
    return 0xFF;
  }
  return *_buf++;
}

size_t CInBufferBase::ReadBytesPart(Byte *buf, size_t size)
{
  if (size == 0)
    return 0;
  size_t rem = (size_t)(_bufLim - _buf);
  if (rem == 0)
  {
    if (!ReadBlock())
      return 0;
    rem = (size_t)(_bufLim - _buf);
  }
  if (size > rem)
      size = rem;
  memcpy(buf, _buf, size);
  _buf += size;
  return size;
}

size_t CInBufferBase::ReadBytes(Byte *buf, size_t size)
{
  size_t num = 0;
  for (;;)
  {
    const size_t rem = (size_t)(_bufLim - _buf);
    if (size <= rem)
    {
      if (size != 0)
      {
        memcpy(buf, _buf, size);
        _buf += size;
        num += size;
      }
      return num;
    }
    if (rem != 0)
    {
      memcpy(buf, _buf, rem);
      _buf += rem;
      buf += rem;
      num += rem;
      size -= rem;
    }
    if (!ReadBlock())
      return num;
  }

  /*
  if ((size_t)(_bufLim - _buf) >= size)
  {
    const Byte *src = _buf;
    for (size_t i = 0; i < size; i++)
      buf[i] = src[i];
    _buf += size;
    return size;
  }
  for (size_t i = 0; i < size; i++)
  {
    if (_buf >= _bufLim)
      if (!ReadBlock())
        return i;
    buf[i] = *_buf++;
  }
  return size;
  */
}

size_t CInBufferBase::Skip(size_t size)
{
  size_t processed = 0;
  for (;;)
  {
    const size_t rem = (size_t)(_bufLim - _buf);
    if (rem >= size)
    {
      _buf += size;
      return processed + size;
    }
    _buf += rem;
    processed += rem;
    size -= rem;
    if (!ReadBlock())
      return processed;
  }
}
