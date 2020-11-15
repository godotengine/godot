// Copyright 2011 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef MINI_CHROMIUM_BASE_MAC_SCOPED_LAUNCH_DATA_H_
#define MINI_CHROMIUM_BASE_MAC_SCOPED_LAUNCH_DATA_H_

#include <launch.h>

#include "base/scoped_generic.h"

namespace base {
namespace mac {

namespace internal {

struct ScopedLaunchDataTraits {
  static launch_data_t InvalidValue() { return nullptr; }

  static void Free(launch_data_t ldt) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    launch_data_free(ldt);
#pragma clang diagnostic pop
  }
};

}  // namespace internal

using ScopedLaunchData =
    ScopedGeneric<launch_data_t, internal::ScopedLaunchDataTraits>;

}  // namespace mac
}  // namespace base

#endif  // MINI_CHROMIUM_BASE_MAC_SCOPED_LAUNCH_DATA_H_
