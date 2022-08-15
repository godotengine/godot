//===- llvm/Support/Win32/Mutex.inc - Win32 Mutex Implementation -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Win32 specific (non-pthread) Mutex class.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only generic Win32 code that
//===          is guaranteed to work on *all* Win32 variants.
//===----------------------------------------------------------------------===//

#include "WindowsSupport.h"
#include "llvm/Support/Mutex.h"

namespace llvm {
using namespace sys;

MutexImpl::MutexImpl(bool /*recursive*/)
{
  C_ASSERT(sizeof(data_) == sizeof(CRITICAL_SECTION));
  // data_ = new CRITICAL_SECTION; // HLSL Change
  InitializeCriticalSection((LPCRITICAL_SECTION)data_);
}

MutexImpl::~MutexImpl()
{
  DeleteCriticalSection((LPCRITICAL_SECTION)data_);
  // delete (LPCRITICAL_SECTION)data_; // HLSL Change
  // data_ = 0; // HLSL Change
}

bool
MutexImpl::acquire()
{
  EnterCriticalSection((LPCRITICAL_SECTION)data_);
  return true;
}

bool
MutexImpl::release()
{
  LeaveCriticalSection((LPCRITICAL_SECTION)data_);
  return true;
}

bool
MutexImpl::tryacquire()
{
  return TryEnterCriticalSection((LPCRITICAL_SECTION)data_);
}

}
