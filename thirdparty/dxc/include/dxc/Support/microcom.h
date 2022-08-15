///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// microcom.h                                                                //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Provides support for basic COM-like constructs.                           //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#ifndef __DXC_MICROCOM__
#define __DXC_MICROCOM__

#include <atomic>
#include "llvm/Support/Atomic.h"

template <typename TIface>
class CComInterfaceArray {
private:
  TIface **m_pData;
  unsigned m_length;
public:
  CComInterfaceArray() : m_pData(nullptr), m_length(0) { }
  ~CComInterfaceArray() {
    clear();
  }
  bool empty() const { return m_length == 0; }
  unsigned size() const { return m_length; }
  TIface ***data_ref() { return &m_pData; }
  unsigned *size_ref() { return &m_length; }
  TIface **begin() {
    return m_pData;
  }
  TIface **end() {
    return m_pData + m_length;
  }
  void clear() {
    if (m_length) {
      for (unsigned i = 0; i < m_length; ++i) {
        if (m_pData[i] != nullptr) {
          m_pData[i]->Release();
          m_pData[i] = nullptr;
        }
      }
      m_length = 0;
    }
    if (m_pData) {
      CoTaskMemFree(m_pData);
      m_pData = nullptr;
    }
  }
  HRESULT alloc(unsigned count) {
    clear();
    m_pData = (TIface**)CoTaskMemAlloc(sizeof(TIface*) * count);
    if (m_pData == nullptr)
      return E_OUTOFMEMORY;
    m_length = count;
    ZeroMemory(m_pData, sizeof(TIface*) * count);
    return S_OK;
  }
  TIface **get_address_of(unsigned index) {
    return &(m_pData[index]);
  }
  TIface **release() {
    TIface **result = m_pData;
    m_pData = nullptr;
    m_length = 0;
    return result;
  }
  void release(TIface ***pValues, unsigned *length) {
    *pValues = m_pData;
    m_pData = nullptr;
    *length = m_length;
    m_length = 0;
  }
};

#define DXC_MICROCOM_REF_FIELD(m_dwRef)                                        \
  volatile std::atomic<llvm::sys::cas_flag> m_dwRef = {0};
#define DXC_MICROCOM_ADDREF_IMPL(m_dwRef)                                      \
  ULONG STDMETHODCALLTYPE AddRef() override {                                  \
    return (ULONG)++m_dwRef;                                                   \
  }
#define DXC_MICROCOM_ADDREF_RELEASE_IMPL(m_dwRef)                              \
  DXC_MICROCOM_ADDREF_IMPL(m_dwRef)                                            \
  ULONG STDMETHODCALLTYPE Release() override {                                 \
    ULONG result = (ULONG)--m_dwRef;                                           \
    if (result == 0)                                                           \
      delete this;                                                             \
    return result;                                                             \
  }

template <typename T, typename... Args>
inline T *CreateOnMalloc(IMalloc * pMalloc, Args&&... args) {
  void *P = pMalloc->Alloc(sizeof(T));
  try { if (P) new (P)T(pMalloc, std::forward<Args>(args)...); }
  catch (...) { pMalloc->Free(P); throw; }
  return (T *)P;
}

template<typename T>
void DxcCallDestructor(T *obj) {
  obj->~T();
}

// The "TM" version keep an IMalloc field that, if not null, indicate
// ownership of 'this' and of any allocations used during release.
#define DXC_MICROCOM_TM_REF_FIELDS()                                           \
  volatile std::atomic<llvm::sys::cas_flag> m_dwRef = {0};                     \
  CComPtr<IMalloc> m_pMalloc;

#define DXC_MICROCOM_TM_ADDREF_RELEASE_IMPL()                                  \
  DXC_MICROCOM_ADDREF_IMPL(m_dwRef)                                            \
  ULONG STDMETHODCALLTYPE Release() override {                                 \
    ULONG result = (ULONG)--m_dwRef;                                           \
    if (result == 0) {                                                         \
      CComPtr<IMalloc> pTmp(m_pMalloc);                                        \
      DxcThreadMalloc M(pTmp);                                                 \
      DxcCallDestructor(this);                                                 \
      pTmp->Free(this);                                                        \
    }                                                                          \
    return result;                                                             \
  }

#define DXC_MICROCOM_TM_CTOR(T)                                                \
  DXC_MICROCOM_TM_CTOR_ONLY(T)                                                 \
  DXC_MICROCOM_TM_ALLOC(T)
#define DXC_MICROCOM_TM_CTOR_ONLY(T)                                           \
  T(IMalloc *pMalloc) : m_dwRef(0), m_pMalloc(pMalloc) {}
#define DXC_MICROCOM_TM_ALLOC(T)                                               \
  template <typename... Args>                                                  \
  static T *Alloc(IMalloc *pMalloc, Args &&... args) {                         \
    void *P = pMalloc->Alloc(sizeof(T));                                       \
    try {                                                                      \
      if (P)                                                                   \
        new (P) T(pMalloc, std::forward<Args>(args)...);                       \
    } catch (...) {                                                            \
      pMalloc->Free(P);                                                        \
      throw;                                                                   \
    }                                                                          \
    return (T *)P;                                                             \
  }

/// <summary>
/// Provides a QueryInterface implementation for a class that supports
/// any number of interfaces in addition to IUnknown.
/// </summary>
/// <remarks>
/// This implementation will also report the instance as not supporting
/// marshaling. This will help catch marshaling problems early or avoid
/// them altogether.
/// </remarks>
template<typename TObject>
HRESULT DoBasicQueryInterface_recurse(TObject* self, REFIID iid, void** ppvObject) {
  return E_NOINTERFACE;
}
template<typename TObject, typename TInterface, typename... Ts>
HRESULT DoBasicQueryInterface_recurse(TObject* self, REFIID iid, void** ppvObject) {
  if (ppvObject == nullptr) return E_POINTER;
  if (IsEqualIID(iid, __uuidof(TInterface))) {
    *(TInterface**)ppvObject = self;
    self->AddRef();
    return S_OK;
  }
  return DoBasicQueryInterface_recurse<TObject, Ts...>(self, iid, ppvObject);
}
template<typename... Ts, typename TObject>
HRESULT DoBasicQueryInterface(TObject* self, REFIID iid, void** ppvObject) {
  if (ppvObject == nullptr) return E_POINTER;

  // Support INoMarshal to void GIT shenanigans.
  if (IsEqualIID(iid, __uuidof(IUnknown)) ||
    IsEqualIID(iid, __uuidof(INoMarshal))) {
    *ppvObject = reinterpret_cast<IUnknown*>(self);
    reinterpret_cast<IUnknown*>(self)->AddRef();
    return S_OK;
  }

  return DoBasicQueryInterface_recurse<TObject, Ts...>(self, iid, ppvObject);
}

template <typename T>
HRESULT AssignToOut(T value, _Out_ T* pResult) {
  if (pResult == nullptr)
    return E_POINTER;
  *pResult = value;
  return S_OK;
}
template <typename T>
HRESULT AssignToOut(nullptr_t value, _Out_ T* pResult) {
  if (pResult == nullptr)
    return E_POINTER;
  *pResult = value;
  return S_OK;
}
template <typename T>
HRESULT ZeroMemoryToOut(_Out_ T* pResult) {
  if (pResult == nullptr)
    return E_POINTER;
  ZeroMemory(pResult, sizeof(*pResult));
  return S_OK;
}

template <typename T>
void AssignToOutOpt(T value, _Out_opt_ T* pResult) {
  if (pResult != nullptr)
    *pResult = value;
}
template <typename T>
void AssignToOutOpt(nullptr_t value, _Out_opt_ T* pResult) {
  if (pResult != nullptr)
    *pResult = value;
}

#endif
