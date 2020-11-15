// Copyright 2008 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef MINI_CHROMIUM_BASE_MAC_SCOPED_NSAUTORELEASE_POOL_H_
#define MINI_CHROMIUM_BASE_MAC_SCOPED_NSAUTORELEASE_POOL_H_

#include "base/macros.h"

#if defined(OS_MACOSX)
#if defined(__OBJC__)
@class NSAutoreleasePool;
#else  // __OBJC__
class NSAutoreleasePool;
#endif  // __OBJC__
#endif  // OS_MACOSX

namespace base {
namespace mac {

// On the Mac, ScopedNSAutoreleasePool allocates an NSAutoreleasePool when
// instantiated and sends it a -drain message when destroyed.  This allows an
// autorelease pool to be maintained in ordinary C++ code without bringing in
// any direct Objective-C dependency.
//
// On other platforms, ScopedNSAutoreleasePool is an empty object with no
// effects.  This allows it to be used directly in cross-platform code without
// ugly #ifdefs.
class ScopedNSAutoreleasePool {
 public:
#if !defined(OS_MACOSX)
  ScopedNSAutoreleasePool() {}
  void Recycle() { }
#else  // OS_MACOSX
  ScopedNSAutoreleasePool();
  ~ScopedNSAutoreleasePool();

  // Clear out the pool in case its position on the stack causes it to be
  // alive for long periods of time (such as the entire length of the app).
  // Only use then when you're certain the items currently in the pool are
  // no longer needed.
  void Recycle();
 private:
  NSAutoreleasePool* autorelease_pool_;
#endif  // OS_MACOSX

 private:
  DISALLOW_COPY_AND_ASSIGN(ScopedNSAutoreleasePool);
};

}  // namespace mac
}  // namespace base

#endif  // MINI_CHROMIUM_BASE_MAC_SCOPED_NSAUTORELEASE_POOL_H_
