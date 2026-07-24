// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_RENDER_PIPELINE_STAGE_BLENDING_H_
#define LIB_JXL_RENDER_PIPELINE_STAGE_BLENDING_H_

#include <memory>

#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/dec_cache.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/render_pipeline/render_pipeline_stage.h"

namespace jxl {

// Applies blending if applicable.
std::unique_ptr<RenderPipelineStage> GetBlendingStage(
    const FrameHeader& frame_header, const PassesDecoderState* dec_state,
    const ColorEncoding& frame_color_encoding);

}  // namespace jxl

#endif  // LIB_JXL_RENDER_PIPELINE_STAGE_BLENDING_H_
