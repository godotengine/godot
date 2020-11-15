// Copyright 2006-2008 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// This file is used for debugging assertion support.  The Lock class
// is functionally a wrapper around the LockImpl class, so the only
// real intelligence in the class is in the debugging logic.

#include "base/synchronization/lock.h"

#include "base/logging.h"

#ifndef NDEBUG

namespace base {

namespace {

ThreadRefType GetCurrentThreadRef() {
#if defined(OS_WIN)
  return GetCurrentThreadId();
#elif defined(OS_POSIX)
  return pthread_self();
#endif
}

}  // namespace

Lock::Lock() : owning_thread_(), lock_() {
}

Lock::~Lock() {
  DCHECK_EQ(owning_thread_, ThreadRefType());
}

void Lock::AssertAcquired() const {
  DCHECK_EQ(owning_thread_, GetCurrentThreadRef());
}

void Lock::CheckHeldAndUnmark() {
  DCHECK_EQ(owning_thread_, GetCurrentThreadRef());
  owning_thread_ = ThreadRefType();
}

void Lock::CheckUnheldAndMark() {
  DCHECK_EQ(owning_thread_, ThreadRefType());
  owning_thread_ = GetCurrentThreadRef();
}

}  // namespace base

#endif
