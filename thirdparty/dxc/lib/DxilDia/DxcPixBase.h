#pragma once

#include "dxc/Support/WinIncludes.h"

namespace dxil_debug_info
{
template <typename T>
T *Initialize(T *Obj)
{
  if (Obj != nullptr)
  {
    Obj->AddRef();
  }
  return Obj;
}

template <typename T, typename O, typename... A>
HRESULT NewDxcPixDxilDebugInfoObjectOrThrow(O **pOut, A... args)
{
  if ((*pOut = Initialize(T::Alloc(args...))) == nullptr)
  {
    throw std::bad_alloc();
  }

  return S_OK;
}
}  // namespace dxil_debug_info
