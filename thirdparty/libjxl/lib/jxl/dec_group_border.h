// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_DEC_GROUP_BORDER_H_
#define LIB_JXL_DEC_GROUP_BORDER_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "lib/jxl/base/rect.h"
#include "lib/jxl/frame_dimensions.h"

namespace jxl {

class GroupBorderAssigner {
 public:
  // Prepare the GroupBorderAssigner to handle a given frame.
  void Init(const FrameDimensions& frame_dim);
  // Marks a group as done, and returns the (at most 3) rects to run
  // FinalizeImageRect on. `block_rect` must be the rect corresponding
  // to the given `group_id`, measured in blocks.
  void GroupDone(size_t group_id, size_t padx, size_t pady,
                 Rect* rects_to_finalize, size_t* num_to_finalize);
  // Marks a group as not-done, for running re-paints.
  void ClearDone(size_t group_id);

  static constexpr size_t kMaxToFinalize = 3;

 private:
  FrameDimensions frame_dim_;
  std::vector<std::atomic<uint8_t>> counters_;

  // Constants to identify group positions relative to the corners.
  static constexpr uint8_t kTopLeft = 0x01;
  static constexpr uint8_t kTopRight = 0x02;
  static constexpr uint8_t kBottomRight = 0x04;
  static constexpr uint8_t kBottomLeft = 0x08;
};

}  // namespace jxl

#endif  // LIB_JXL_DEC_GROUP_BORDER_H_
