///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// dxcmem.cpp                                                                //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Provides support for a thread-local allocator.                            //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/Support/Global.h"
#ifdef _WIN32
#include <specstrings.h>
#endif

#include "dxc/Support/WinIncludes.h"
#include "dxc/Support/WinFunctions.h"
#include "llvm/Support/ThreadLocal.h"
#include <memory>

static llvm::sys::ThreadLocal<IMalloc> *g_ThreadMallocTls;
static IMalloc *g_pDefaultMalloc;

HRESULT DxcInitThreadMalloc() throw() {
  // Allow a default malloc from a previous call to Init.
  // This will not be cleaned up in the call to Cleanup because
  // it can still be referenced after Cleanup is called.
  if (g_pDefaultMalloc) {
    g_pDefaultMalloc->AddRef();
  }
  else {
    // We capture the default malloc early to avoid potential failures later on.
    HRESULT hrMalloc = CoGetMalloc(1, &g_pDefaultMalloc);
    if (FAILED(hrMalloc)) return hrMalloc;
  }
  DXASSERT(g_ThreadMallocTls == nullptr, "else InitThreadMalloc already called");

  g_ThreadMallocTls = (llvm::sys::ThreadLocal<IMalloc>*)g_pDefaultMalloc->Alloc(sizeof(llvm::sys::ThreadLocal<IMalloc>));
  if (g_ThreadMallocTls == nullptr) {
    g_pDefaultMalloc->Release();
    g_pDefaultMalloc = nullptr;
    return E_OUTOFMEMORY;
  }
  g_ThreadMallocTls = new(g_ThreadMallocTls) llvm::sys::ThreadLocal<IMalloc>;

  return S_OK;
}

void DxcCleanupThreadMalloc() throw() {
  if (g_ThreadMallocTls) {
    DXASSERT(g_pDefaultMalloc, "else DxcInitThreadMalloc didn't work/fail atomically");
    g_ThreadMallocTls->~ThreadLocal();
    g_pDefaultMalloc->Free(g_ThreadMallocTls);
    g_ThreadMallocTls = nullptr;
  }
}

IMalloc *DxcGetThreadMallocNoRef() throw() {
  if (g_ThreadMallocTls == nullptr) {
    return g_pDefaultMalloc;
  }

  return g_ThreadMallocTls->get();
}

void DxcClearThreadMalloc() throw() {
  if (g_ThreadMallocTls != nullptr) {
    IMalloc *pMalloc = DxcGetThreadMallocNoRef();
    g_ThreadMallocTls->erase();
    if (pMalloc != nullptr) {
      pMalloc->Release();
    }
  }
}

void DxcSetThreadMallocToDefault() throw() {
  DXASSERT(g_ThreadMallocTls != nullptr, "else prior to DxcInitThreadMalloc or after DxcCleanupThreadMalloc");
  DXASSERT(DxcGetThreadMallocNoRef() == nullptr, "else nested allocation invoked");
  g_ThreadMallocTls->set(g_pDefaultMalloc);
  g_pDefaultMalloc->AddRef();
}

static IMalloc *DxcSwapThreadMalloc(IMalloc *pMalloc, IMalloc **ppPrior) throw() {
  DXASSERT(g_ThreadMallocTls != nullptr, "else prior to DxcInitThreadMalloc or after DxcCleanupThreadMalloc");
  IMalloc *pPrior = DxcGetThreadMallocNoRef();
  if (ppPrior) {
    *ppPrior = pPrior;
  }
  g_ThreadMallocTls->set(pMalloc);
  return pMalloc;
}

DxcThreadMalloc::DxcThreadMalloc(IMalloc *pMallocOrNull) throw() {
    p = DxcSwapThreadMalloc(pMallocOrNull ? pMallocOrNull : g_pDefaultMalloc, &pPrior);
}

DxcThreadMalloc::~DxcThreadMalloc() {
    DxcSwapThreadMalloc(pPrior, nullptr);
}
