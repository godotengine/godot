// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_RENDER_PIPELINE_STAGE_CHROMA_UPSAMPLING_H_
#define LIB_JXL_RENDER_PIPELINE_STAGE_CHROMA_UPSAMPLING_H_
#include <math.h>
#include <stdint.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "lib/jxl/loop_filter.h"
#include "lib/jxl/render_pipeline/render_pipeline_stage.h"

namespace jxl {

// Applies simple upsampling, either horizontal or vertical, to the given
// channel.
std::unique_ptr<RenderPipelineStage> GetChromaUpsamplingStage(size_t channel,
                                                              bool horizontal);
}  // namespace jxl

#endif  // LIB_JXL_RENDER_PIPELINE_STAGE_CHROMA_UPSAMPLING_H_
