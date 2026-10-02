// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_FRAME_DIMENSIONS_H_
#define LIB_JXL_FRAME_DIMENSIONS_H_

// FrameDimensions struct, block and group dimensions constants.

#include <cstddef>

#include "lib/jxl/base/common.h"
#include "lib/jxl/base/rect.h"

namespace jxl {
// Some enums and typedefs used by more than one header file.

// Block is the square grid of pixels to which an "energy compaction"
// transformation (e.g. DCT) is applied. Each block has its own AC quantizer.
constexpr size_t kBlockDim = 8;

constexpr size_t kDCTBlockSize = kBlockDim * kBlockDim;

constexpr size_t kGroupDim = 256;
static_assert(kGroupDim % kBlockDim == 0,
              "Group dim should be divisible by block dim");
constexpr size_t kGroupDimInBlocks = kGroupDim / kBlockDim;

// Dimensions of a frame, in pixels, and other derived dimensions.
// Computed from FrameHeader.
// TODO(veluca): add extra channels.
struct FrameDimensions {
  void Set(size_t xsize, size_t ysize, size_t group_size_shift,
           size_t max_hshift, size_t max_vshift, bool modular_mode,
           size_t upsampling) {
    group_dim = (kGroupDim >> 1) << group_size_shift;
    dc_group_dim = group_dim * kBlockDim;
    xsize_upsampled = xsize;
    ysize_upsampled = ysize;
    this->xsize = DivCeil(xsize, upsampling);
    this->ysize = DivCeil(ysize, upsampling);
    xsize_blocks = DivCeil(this->xsize, kBlockDim << max_hshift) << max_hshift;
    ysize_blocks = DivCeil(this->ysize, kBlockDim << max_vshift) << max_vshift;
    xsize_padded = xsize_blocks * kBlockDim;
    ysize_padded = ysize_blocks * kBlockDim;
    if (modular_mode) {
      // Modular mode doesn't have any padding.
      xsize_padded = this->xsize;
      ysize_padded = this->ysize;
    }
    xsize_upsampled_padded = xsize_padded * upsampling;
    ysize_upsampled_padded = ysize_padded * upsampling;
    xsize_groups = DivCeil(this->xsize, group_dim);
    ysize_groups = DivCeil(this->ysize, group_dim);
    xsize_dc_groups = DivCeil(xsize_blocks, group_dim);
    ysize_dc_groups = DivCeil(ysize_blocks, group_dim);
    num_groups = xsize_groups * ysize_groups;
    num_dc_groups = xsize_dc_groups * ysize_dc_groups;
  }

  Rect GroupRect(size_t group_index) const {
    const size_t gx = group_index % xsize_groups;
    const size_t gy = group_index / xsize_groups;
    const Rect rect(gx * group_dim, gy * group_dim, group_dim, group_dim, xsize,
                    ysize);
    return rect;
  }

  Rect BlockGroupRect(size_t group_index) const {
    const size_t gx = group_index % xsize_groups;
    const size_t gy = group_index / xsize_groups;
    const Rect rect(gx * (group_dim >> 3), gy * (group_dim >> 3),
                    group_dim >> 3, group_dim >> 3, xsize_blocks, ysize_blocks);
    return rect;
  }

  Rect DCGroupRect(size_t group_index) const {
    const size_t gx = group_index % xsize_dc_groups;
    const size_t gy = group_index / xsize_dc_groups;
    const Rect rect(gx * group_dim, gy * group_dim, group_dim, group_dim,
                    xsize_blocks, ysize_blocks);
    return rect;
  }

  // Image size without any upsampling, i.e. original_size / upsampling.
  size_t xsize;
  size_t ysize;
  // Original image size.
  size_t xsize_upsampled;
  size_t ysize_upsampled;
  // Image size after upsampling the padded image.
  size_t xsize_upsampled_padded;
  size_t ysize_upsampled_padded;
  // Image size after padding to a multiple of kBlockDim (if VarDCT mode).
  size_t xsize_padded;
  size_t ysize_padded;
  // Image size in kBlockDim blocks.
  size_t xsize_blocks;
  size_t ysize_blocks;
  // Image size in number of groups.
  size_t xsize_groups;
  size_t ysize_groups;
  // Image size in number of DC groups.
  size_t xsize_dc_groups;
  size_t ysize_dc_groups;
  // Number of AC or DC groups.
  size_t num_groups;
  size_t num_dc_groups;
  // Size of a group.
  size_t group_dim;
  size_t dc_group_dim;
};

}  // namespace jxl

#endif  // LIB_JXL_FRAME_DIMENSIONS_H_
