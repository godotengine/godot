// Copyright 2017 The Chromium OS Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef _BSDIFF_CONTROL_ENTRY_H_
#define _BSDIFF_CONTROL_ENTRY_H_

#include <stdint.h>

struct ControlEntry {
  constexpr ControlEntry(uint64_t diff_size,
                         uint64_t extra_size,
                         int64_t offset_increment)
      : diff_size(diff_size),
        extra_size(extra_size),
        offset_increment(offset_increment) {}
  constexpr ControlEntry() = default;

  // The number of bytes to copy from the source and diff stream.
  uint64_t diff_size{0};

  // The number of bytes to copy from the extra stream.
  uint64_t extra_size{0};

  // The value to add to the source pointer after patching from the diff stream.
  int64_t offset_increment{0};

  [[nodiscard]] bool operator==(const ControlEntry& o) const {
    return diff_size == o.diff_size && extra_size == o.extra_size &&
           offset_increment == o.offset_increment;
  }
};

#endif  // _BSDIFF_CONTROL_ENTRY_H_
