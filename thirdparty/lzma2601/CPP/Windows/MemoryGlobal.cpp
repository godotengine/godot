// Windows/MemoryGlobal.cpp

#include "StdAfx.h"

#include "MemoryGlobal.h"

namespace NWindows {
namespace NMemory {

bool CGlobal::Alloc(UINT flags, SIZE_T size) throw()
{
  HGLOBAL newBlock = ::GlobalAlloc(flags, size);
  if (newBlock == NULL)
    return false;
  _global = newBlock;
  return true;
}

bool CGlobal::Free() throw()
{
  if (_global == NULL)
    return true;
  _global = ::GlobalFree(_global);
  return (_global == NULL);
}

bool CGlobal::ReAlloc(SIZE_T size) throw()
{
  HGLOBAL newBlock = ::GlobalReAlloc(_global, size, GMEM_MOVEABLE);
  if (newBlock == NULL)
    return false;
  _global = newBlock;
  return true;
}

}}
