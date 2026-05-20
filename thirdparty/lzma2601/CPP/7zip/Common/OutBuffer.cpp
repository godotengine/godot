// OutBuffer.cpp

#include "StdAfx.h"

#include "../../../C/Alloc.h"

#include "OutBuffer.h"

bool COutBuffer::Create(UInt32 bufSize) throw()
{
  const UInt32 kMinBlockSize = 1;
  if (bufSize < kMinBlockSize)
    bufSize = kMinBlockSize;
  if (_buf && _bufSize == bufSize)
    return true;
  Free();
  _bufSize = bufSize;
  _buf = (Byte *)::MidAlloc(bufSize);
  return (_buf != NULL);
}

void COutBuffer::Free() throw()
{
  ::MidFree(_buf);
  _buf = NULL;
}

void COutBuffer::Init() throw()
{
  _streamPos = 0;
  _limitPos = _bufSize;
  _pos = 0;
  _processedSize = 0;
  _overDict = false;
  #ifdef Z7_NO_EXCEPTIONS
  ErrorCode = S_OK;
  #endif
}

UInt64 COutBuffer::GetProcessedSize() const throw()
{
  UInt64 res = _processedSize + _pos - _streamPos;
  if (_streamPos > _pos)
    res += _bufSize;
  return res;
}


HRESULT COutBuffer::FlushPart() throw()
{
  // _streamPos < _bufSize
  UInt32 size = (_streamPos >= _pos) ? (_bufSize - _streamPos) : (_pos - _streamPos);
  HRESULT result = S_OK;
  #ifdef Z7_NO_EXCEPTIONS
  result = ErrorCode;
  #endif
  if (_buf2)
  {
    memcpy(_buf2, _buf + _streamPos, size);
    _buf2 += size;
  }

  if (_stream
      #ifdef Z7_NO_EXCEPTIONS
      && (ErrorCode == S_OK)
      #endif
     )
  {
    UInt32 processedSize = 0;
    result = _stream->Write(_buf + _streamPos, size, &processedSize);
    size = processedSize;
  }
  _streamPos += size;
  if (_streamPos == _bufSize)
    _streamPos = 0;
  if (_pos == _bufSize)
  {
    _overDict = true;
    _pos = 0;
  }
  _limitPos = (_streamPos > _pos) ? _streamPos : _bufSize;
  _processedSize += size;
  return result;
}

HRESULT COutBuffer::Flush() throw()
{
  #ifdef Z7_NO_EXCEPTIONS
  if (ErrorCode != S_OK)
    return ErrorCode;
  #endif

  while (_streamPos != _pos)
  {
    const HRESULT result = FlushPart();
    if (result != S_OK)
      return result;
  }
  return S_OK;
}

void COutBuffer::FlushWithCheck()
{
  const HRESULT result = Flush();
  #ifdef Z7_NO_EXCEPTIONS
  ErrorCode = result;
  #else
  if (result != S_OK)
    throw COutBufferException(result);
  #endif
}
