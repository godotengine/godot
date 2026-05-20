// Common/DynLimBuf.cpp

#include "StdAfx.h"

#include "DynLimBuf.h"
#include "MyString.h"

CDynLimBuf::CDynLimBuf(size_t limit) throw()
{
  _chars = NULL;
  _pos = 0;
  _size = 0;
  _sizeLimit = limit;
  _error = true;
  unsigned size = 1 << 4;
  if (size > limit)
    size = (unsigned)limit;
  _chars = (Byte *)MyAlloc(size);
  if (_chars)
  {
    _size = size;
    _error = false;
  }
}

CDynLimBuf & CDynLimBuf::operator+=(char c) throw()
{
  if (_error)
    return *this;
  if (_size == _pos)
  {
    size_t n = _sizeLimit - _size;
    if (n == 0)
    {
      _error = true;
      return *this;
    }
    if (n > _size)
      n = _size;

    n += _pos;

    Byte *newBuf = (Byte *)MyAlloc(n);
    if (!newBuf)
    {
      _error = true;
      return *this;
    }
    memcpy(newBuf, _chars, _pos);
    MyFree(_chars);
    _chars = newBuf;
    _size = n;
  }
  _chars[_pos++] = (Byte)c;
  return *this;
}

CDynLimBuf &CDynLimBuf::operator+=(const char *s) throw()
{
  if (_error)
    return *this;
  unsigned len = MyStringLen(s);
  size_t rem = _sizeLimit - _pos;
  if (rem < len)
  {
    len = (unsigned)rem;
    _error = true;
  }
  if (_size - _pos < len)
  {
    size_t n = _pos + len;
    if (n - _size < _size)
    {
      n = _sizeLimit;
      if (n - _size > _size)
        n = _size * 2;
    }

    Byte *newBuf = (Byte *)MyAlloc(n);
    if (!newBuf)
    {
      _error = true;
      return *this;
    }
    memcpy(newBuf, _chars, _pos);
    MyFree(_chars);
    _chars = newBuf;
    _size = n;
  }
  memcpy(_chars + _pos, s, len);
  _pos += len;
  return *this;
}
