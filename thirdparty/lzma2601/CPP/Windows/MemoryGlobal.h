// Windows/MemoryGlobal.h

#ifndef ZIP7_INC_WINDOWS_MEMORY_GLOBAL_H
#define ZIP7_INC_WINDOWS_MEMORY_GLOBAL_H

#include "../Common/MyWindows.h"

namespace NWindows {
namespace NMemory {

class CGlobal
{
  HGLOBAL _global;
public:
  CGlobal(): _global(NULL) {}
  ~CGlobal() { Free(); }
  operator HGLOBAL() const { return _global; }
  void Attach(HGLOBAL hGlobal)
  {
    Free();
    _global = hGlobal;
  }
  HGLOBAL Detach()
  {
    const HGLOBAL h = _global;
    _global = NULL;
    return h;
  }
  bool Alloc(UINT flags, SIZE_T size) throw();
  bool Free() throw();
  LPVOID Lock() const { return GlobalLock(_global); }
  void Unlock() const { GlobalUnlock(_global); }
  bool ReAlloc(SIZE_T size) throw();
};

class CGlobalLock
{
  HGLOBAL _global;
  LPVOID _ptr;
public:
  LPVOID GetPointer() const { return _ptr; }
  CGlobalLock(HGLOBAL hGlobal): _global(hGlobal)
  {
    _ptr = GlobalLock(hGlobal);
  }
  ~CGlobalLock()
  {
    if (_ptr)
      GlobalUnlock(_global);
  }
};

}}

#endif
