// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_BASE_OVERRIDE_H_
#define LIB_JXL_BASE_OVERRIDE_H_

#include <cstdint>

// 'Trool' for command line arguments: force enable/disable, or use default.

namespace jxl {

// No effect if kDefault, otherwise forces a feature (typically a FrameHeader
// flag) on or off.
enum class Override : int8_t { kOn = 1, kOff = 0, kDefault = -1 };

static inline Override OverrideFromBool(bool flag) {
  return flag ? Override::kOn : Override::kOff;
}

static inline bool ApplyOverride(Override o, bool default_condition) {
  if (o == Override::kOn) return true;
  if (o == Override::kOff) return false;
  return default_condition;
}

}  // namespace jxl

#endif  // LIB_JXL_BASE_OVERRIDE_H_
