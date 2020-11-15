// Copyright 2010 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef MINI_CHROMIUM_BASE_MAC_SCOPED_IOOBJECT_H_
#define MINI_CHROMIUM_BASE_MAC_SCOPED_IOOBJECT_H_

#include <IOKit/IOKitLib.h>

#include "base/mac/scoped_typeref.h"

namespace base {
namespace mac {

namespace internal {

template <typename IOT>
struct ScopedIOObjectTraits {
  static IOT InvalidValue() { return IO_OBJECT_NULL; }
  static IOT Retain(IOT iot) {
    IOObjectRetain(iot);
    return iot;
  }
  static void Release(IOT iot) { IOObjectRelease(iot); }
};

}  // namespce internal

template <typename IOT>
using ScopedIOObject = ScopedTypeRef<IOT, internal::ScopedIOObjectTraits<IOT>>;

}  // namespace mac
}  // namespace base

#endif  // MINI_CHROMIUM_BASE_MAC_SCOPED_IOOBJECT_H_
