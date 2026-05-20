// Common/DynLimBuf.h

#ifndef ZIP7_INC_COMMON_DYN_LIM_BUF_H
#define ZIP7_INC_COMMON_DYN_LIM_BUF_H

#include <string.h>

#include "../../C/Alloc.h"

#include "MyString.h"

class CDynLimBuf
{
  Byte *_chars;
  size_t _pos;
  size_t _size;
  size_t _sizeLimit;
  bool _error;

  CDynLimBuf(const CDynLimBuf &s);

  // ---------- forbidden functions ----------
  CDynLimBuf &operator+=(wchar_t c);

public:
  CDynLimBuf(size_t limit) throw();
  ~CDynLimBuf() { MyFree(_chars); }

  size_t Len() const { return _pos; }
  bool IsError() const { return _error; }
  void Empty() { _pos = 0; _error = false; }

  operator const Byte *() const { return _chars; }
  // const char *Ptr() const { return _chars; }

  CDynLimBuf &operator+=(char c) throw();
  CDynLimBuf &operator+=(const char *s) throw();
};


#endif
