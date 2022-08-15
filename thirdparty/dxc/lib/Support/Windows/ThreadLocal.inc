//= llvm/Support/Win32/ThreadLocal.inc - Win32 Thread Local Data -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Win32 specific (non-pthread) ThreadLocal class.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only generic Win32 code that
//===          is guaranteed to work on *all* Win32 variants.
//===----------------------------------------------------------------------===//

#include "WindowsSupport.h"
#include "llvm/Support/ThreadLocal.h"

namespace llvm {
using namespace sys;

ThreadLocalImpl::ThreadLocalImpl() : data() {
  static_assert(sizeof(DWORD) <= sizeof(data), "size too big");
  DWORD* tls = reinterpret_cast<DWORD*>(&data);
  *tls = TlsAlloc();
  assert(*tls != TLS_OUT_OF_INDEXES);
}

ThreadLocalImpl::~ThreadLocalImpl() {
  DWORD* tls = reinterpret_cast<DWORD*>(&data);
  TlsFree(*tls);
}

void *ThreadLocalImpl::getInstance() {
  DWORD* tls = reinterpret_cast<DWORD*>(&data);
  return TlsGetValue(*tls);
}

void ThreadLocalImpl::setInstance(const void* d){
  DWORD* tls = reinterpret_cast<DWORD*>(&data);
  int errorcode = TlsSetValue(*tls, const_cast<void*>(d));
  assert(errorcode != 0);
  (void)errorcode;
}

void ThreadLocalImpl::removeInstance() {
  setInstance(0);
}

}
