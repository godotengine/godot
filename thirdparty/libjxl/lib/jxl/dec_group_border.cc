// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/dec_group_border.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/frame_dimensions.h"

namespace jxl {

void GroupBorderAssigner::Init(const FrameDimensions& frame_dim) {
  frame_dim_ = frame_dim;
  size_t num_corners =
      (frame_dim_.xsize_groups + 1) * (frame_dim_.ysize_groups + 1);
  { std::vector<std::atomic<uint8_t>>(num_corners).swap(counters_); }
  // Initialize counters.
  for (size_t y = 0; y < frame_dim_.ysize_groups + 1; y++) {
    for (size_t x = 0; x < frame_dim_.xsize_groups + 1; x++) {
      // Counters at image borders don't have anything on the other side, we
      // pre-fill their value to have more uniform handling afterwards.
      uint8_t init_value = 0;
      if (x == 0) {
        init_value |= kTopLeft | kBottomLeft;
      }
      if (x == frame_dim_.xsize_groups) {
        init_value |= kTopRight | kBottomRight;
      }
      if (y == 0) {
        init_value |= kTopLeft | kTopRight;
      }
      if (y == frame_dim_.ysize_groups) {
        init_value |= kBottomLeft | kBottomRight;
      }
      counters_[y * (frame_dim_.xsize_groups + 1) + x] = init_value;
    }
  }
}

void GroupBorderAssigner::ClearDone(size_t group_id) {
  size_t x = group_id % frame_dim_.xsize_groups;
  size_t y = group_id / frame_dim_.xsize_groups;
  size_t top_left_idx = y * (frame_dim_.xsize_groups + 1) + x;
  size_t top_right_idx = y * (frame_dim_.xsize_groups + 1) + x + 1;
  size_t bottom_right_idx = (y + 1) * (frame_dim_.xsize_groups + 1) + x + 1;
  size_t bottom_left_idx = (y + 1) * (frame_dim_.xsize_groups + 1) + x;
  counters_[top_left_idx].fetch_and(~kBottomRight);
  counters_[top_right_idx].fetch_and(~kBottomLeft);
  counters_[bottom_left_idx].fetch_and(~kTopRight);
  counters_[bottom_right_idx].fetch_and(~kTopLeft);
}

// Looking at each corner between groups, we can guarantee that the four
// involved groups will agree between each other regarding the order in which
// each of the four groups terminated. Thus, the last of the four groups
// gets the responsibility of handling the corner. For borders, every border
// is assigned to its top corner (for vertical borders) or to its left corner
// (for horizontal borders): the order as seen on those corners will decide who
// handles that border.

void GroupBorderAssigner::GroupDone(size_t group_id, size_t padx, size_t pady,
                                    Rect* rects_to_finalize,
                                    size_t* num_to_finalize) {
  size_t x = group_id % frame_dim_.xsize_groups;
  size_t y = group_id / frame_dim_.xsize_groups;
  Rect block_rect(x * frame_dim_.group_dim / kBlockDim,
                  y * frame_dim_.group_dim / kBlockDim,
                  frame_dim_.group_dim / kBlockDim,
                  frame_dim_.group_dim / kBlockDim, frame_dim_.xsize_blocks,
                  frame_dim_.ysize_blocks);

  size_t top_left_idx = y * (frame_dim_.xsize_groups + 1) + x;
  size_t top_right_idx = y * (frame_dim_.xsize_groups + 1) + x + 1;
  size_t bottom_right_idx = (y + 1) * (frame_dim_.xsize_groups + 1) + x + 1;
  size_t bottom_left_idx = (y + 1) * (frame_dim_.xsize_groups + 1) + x;

  auto fetch_status = [this](size_t idx, uint8_t bit) {
    // Note that the acq-rel semantics of this fetch are actually needed to
    // ensure that the pixel data of the group is already written to memory.
    size_t status = counters_[idx].fetch_or(bit);
    JXL_DASSERT((bit & status) == 0);
    return bit | status;
  };

  size_t top_left_status = fetch_status(top_left_idx, kBottomRight);
  size_t top_right_status = fetch_status(top_right_idx, kBottomLeft);
  size_t bottom_right_status = fetch_status(bottom_right_idx, kTopLeft);
  size_t bottom_left_status = fetch_status(bottom_left_idx, kTopRight);

  size_t x1 = block_rect.x0() + block_rect.xsize();
  size_t y1 = block_rect.y0() + block_rect.ysize();

  bool is_last_group_x = frame_dim_.xsize_groups == x + 1;
  bool is_last_group_y = frame_dim_.ysize_groups == y + 1;

  // Start of border of neighbouring group, end of border of this group, start
  // of border of this group (on the other side), end of border of next group.
  size_t xpos[4] = {
      block_rect.x0() == 0 ? 0 : block_rect.x0() * kBlockDim - padx,
      block_rect.x0() == 0
          ? 0
          : std::min(frame_dim_.xsize, block_rect.x0() * kBlockDim + padx),
      is_last_group_x ? frame_dim_.xsize : x1 * kBlockDim - padx,
      std::min(frame_dim_.xsize, x1 * kBlockDim + padx)};
  size_t ypos[4] = {
      block_rect.y0() == 0 ? 0 : block_rect.y0() * kBlockDim - pady,
      block_rect.y0() == 0
          ? 0
          : std::min(frame_dim_.ysize, block_rect.y0() * kBlockDim + pady),
      is_last_group_y ? frame_dim_.ysize : y1 * kBlockDim - pady,
      std::min(frame_dim_.ysize, y1 * kBlockDim + pady)};

  *num_to_finalize = 0;
  auto append_rect = [&](size_t x0, size_t x1, size_t y0, size_t y1) {
    Rect rect(xpos[x0], ypos[y0], xpos[x1] - xpos[x0], ypos[y1] - ypos[y0]);
    if (rect.xsize() == 0 || rect.ysize() == 0) return;
    JXL_DASSERT(*num_to_finalize < kMaxToFinalize);
    rects_to_finalize[(*num_to_finalize)++] = rect;
  };

  // Because of how group borders are assigned, it is impossible that we need to
  // process the left and right side of some area but not the center area. Thus,
  // we compute the first/last part to process in every horizontal strip and
  // merge them together. We first collect a mask of what parts should be
  // processed.
  // We do this horizontally rather than vertically because horizontal borders
  // are larger.
  bool available_parts_mask[3][3] = {};  // [x][y]
  // Center
  available_parts_mask[1][1] = true;
  // Corners
  if (top_left_status == 0xF) available_parts_mask[0][0] = true;
  if (top_right_status == 0xF) available_parts_mask[2][0] = true;
  if (bottom_right_status == 0xF) available_parts_mask[2][2] = true;
  if (bottom_left_status == 0xF) available_parts_mask[0][2] = true;
  // Other borders
  if (top_left_status & kTopRight) available_parts_mask[1][0] = true;
  if (top_left_status & kBottomLeft) available_parts_mask[0][1] = true;
  if (top_right_status & kBottomRight) available_parts_mask[2][1] = true;
  if (bottom_left_status & kBottomRight) available_parts_mask[1][2] = true;

  // Collect horizontal ranges.
  constexpr size_t kNoSegment = 3;
  std::pair<size_t, size_t> horizontal_segments[3] = {{kNoSegment, kNoSegment},
                                                      {kNoSegment, kNoSegment},
                                                      {kNoSegment, kNoSegment}};
  for (size_t y = 0; y < 3; y++) {
    for (size_t x = 0; x < 3; x++) {
      if (!available_parts_mask[x][y]) continue;
      JXL_DASSERT(horizontal_segments[y].second == kNoSegment ||
                  horizontal_segments[y].second == x);
      JXL_DASSERT((horizontal_segments[y].first == kNoSegment) ==
                  (horizontal_segments[y].second == kNoSegment));
      if (horizontal_segments[y].first == kNoSegment) {
        horizontal_segments[y].first = x;
      }
      horizontal_segments[y].second = x + 1;
    }
  }
  if (horizontal_segments[0] == horizontal_segments[1] &&
      horizontal_segments[0] == horizontal_segments[2]) {
    append_rect(horizontal_segments[0].first, horizontal_segments[0].second, 0,
                3);
  } else if (horizontal_segments[0] == horizontal_segments[1]) {
    append_rect(horizontal_segments[0].first, horizontal_segments[0].second, 0,
                2);
    append_rect(horizontal_segments[2].first, horizontal_segments[2].second, 2,
                3);
  } else if (horizontal_segments[1] == horizontal_segments[2]) {
    append_rect(horizontal_segments[0].first, horizontal_segments[0].second, 0,
                1);
    append_rect(horizontal_segments[1].first, horizontal_segments[1].second, 1,
                3);
  } else {
    append_rect(horizontal_segments[0].first, horizontal_segments[0].second, 0,
                1);
    append_rect(horizontal_segments[1].first, horizontal_segments[1].second, 1,
                2);
    append_rect(horizontal_segments[2].first, horizontal_segments[2].second, 2,
                3);
  }
}

}  // namespace jxl
