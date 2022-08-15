//===-- WinAdapter.cpp - Windows Adapter for other platforms ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef _WIN32

#include "dxc/Support/WinAdapter.h"
#include "dxc/Support/WinFunctions.h"

//===--------------------------- IUnknown ---------------------------------===//

ULONG IUnknown::AddRef() {
  ++m_count;
  return m_count;
}
ULONG IUnknown::Release() {
  ULONG result = --m_count;
  if (m_count == 0) {
    delete this;
  }
  return result;
}
IUnknown::~IUnknown() {}

//===--------------------------- IMalloc ----------------------------------===//

void *IMalloc::Alloc(size_t size) { return malloc(size); }
void *IMalloc::Realloc(void *ptr, size_t size) { return realloc(ptr, size); }
void IMalloc::Free(void *ptr) { free(ptr); }
HRESULT IMalloc::QueryInterface(REFIID riid, void **ppvObject) {
  assert(false && "QueryInterface not implemented for IMalloc.");
  return E_NOINTERFACE;
}

//===--------------------------- CAllocator -------------------------------===//

void *CAllocator::Reallocate(void *p, size_t nBytes) throw() {
  return realloc(p, nBytes);
}
void *CAllocator::Allocate(size_t nBytes) throw() { return malloc(nBytes); }
void CAllocator::Free(void *p) throw() { free(p); }

//===--------------------------- BSTR Allocation --------------------------===//

void SysFreeString(BSTR bstrString) {
  if (bstrString)
    free((void *)((uintptr_t)bstrString - sizeof(uint32_t)));
}

// Allocate string with length prefix
// https://docs.microsoft.com/en-us/previous-versions/windows/desktop/automat/bstr
BSTR SysAllocStringLen(const OLECHAR *strIn, UINT ui) {
  uint32_t *blobOut =
      (uint32_t *)malloc(sizeof(uint32_t) + (ui + 1) * sizeof(OLECHAR));

  if (!blobOut)
    return nullptr;

  // Size in bytes without trailing NULL character
  blobOut[0] = ui * sizeof(OLECHAR);

  BSTR strOut = (BSTR)&blobOut[1];

  if (strIn)
    memcpy(strOut, strIn, blobOut[0]);

  // Write trailing NULL character:
  strOut[ui] = 0;

  return strOut;
}

//===---------------------- Char converstion ------------------------------===//

const char *CPToLocale(uint32_t CodePage) {
#ifdef __APPLE__
  static const char *utf8 = "en_US.UTF-8";
  static const char *iso88591 = "en_US.ISO8859-1";
#else
  static const char *utf8 = "en_US.utf8";
  static const char *iso88591 = "en_US.iso88591";
#endif
  if (CodePage == CP_UTF8) {
    return utf8;
  } else if (CodePage == CP_ACP) {
    // Experimentation suggests that ACP is expected to be ISO-8859-1
    return iso88591;
  }
  return nullptr;
}

//===--------------------------- CHandle -------------------------------===//

CHandle::CHandle(HANDLE h) { m_h = h; }
CHandle::~CHandle() { CloseHandle(m_h); }
CHandle::operator HANDLE() const throw() { return m_h; }

#endif
