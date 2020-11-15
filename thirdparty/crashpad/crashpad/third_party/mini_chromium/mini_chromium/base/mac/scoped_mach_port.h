// Copyright 2012 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef MINI_CHROMIUM_BASE_MAC_SCOPED_MACH_PORT_H_
#define MINI_CHROMIUM_BASE_MAC_SCOPED_MACH_PORT_H_

#include <mach/mach.h>

#include "base/scoped_generic.h"

namespace base {
namespace mac {

namespace internal {

struct SendRightTraits {
  static mach_port_t InvalidValue() {
    return MACH_PORT_NULL;
  }
  static void Free(mach_port_t port);
};

struct ReceiveRightTraits {
  static mach_port_t InvalidValue() {
    return MACH_PORT_NULL;
  }
  static void Free(mach_port_t port);
};

struct PortSetTraits {
  static mach_port_t InvalidValue() {
    return MACH_PORT_NULL;
  }
  static void Free(mach_port_t port);
};

}  // namespace internal

using ScopedMachSendRight =
    ScopedGeneric<mach_port_t, internal::SendRightTraits>;
using ScopedMachReceiveRight =
    ScopedGeneric<mach_port_t, internal::ReceiveRightTraits>;
using ScopedMachPortSet = ScopedGeneric<mach_port_t, internal::PortSetTraits>;

}  // namespace mac
}  // namespace base

#endif  // MINI_CHROMIUM_BASE_MAC_SCOPED_MACH_PORT_H_
