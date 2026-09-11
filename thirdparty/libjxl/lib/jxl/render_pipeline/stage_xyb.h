// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_RENDER_PIPELINE_STAGE_XYB_H_
#define LIB_JXL_RENDER_PIPELINE_STAGE_XYB_H_
#include <stdint.h>

#include "lib/jxl/dec_xyb.h"
#include "lib/jxl/render_pipeline/render_pipeline_stage.h"

namespace jxl {

// Converts the color channels from XYB to linear with appropriate primaries.
std::unique_ptr<RenderPipelineStage> GetXYBStage(
    const OutputEncodingInfo& output_encoding_info);

// Gets a stage to convert with fixed point arithmetic from XYB to sRGB8 and
// write to a uint8 buffer.
std::unique_ptr<RenderPipelineStage> GetFastXYBTosRGB8Stage(
    uint8_t* rgb, size_t stride, size_t width, size_t height, bool rgba,
    bool has_alpha, size_t alpha_c);
}  // namespace jxl

#endif  // LIB_JXL_RENDER_PIPELINE_STAGE_XYB_H_
