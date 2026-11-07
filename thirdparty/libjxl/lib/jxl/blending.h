// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_BLENDING_H_
#define LIB_JXL_BLENDING_H_

#include <jxl/memory_manager.h>

#include <cstddef>
#include <vector>

#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_patch_dictionary.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/image_metadata.h"

namespace jxl {

bool NeedsBlending(const FrameHeader& frame_header);

Status PerformBlending(JxlMemoryManager* memory_manager, const float* const* bg,
                       const float* const* fg, float* const* out, size_t x0,
                       size_t xsize, const PatchBlending& color_blending,
                       const PatchBlending* ec_blending,
                       const std::vector<ExtraChannelInfo>& extra_channel_info);

}  // namespace jxl

#endif  // LIB_JXL_BLENDING_H_
