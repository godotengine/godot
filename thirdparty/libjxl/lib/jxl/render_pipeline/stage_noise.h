// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_RENDER_PIPELINE_STAGE_NOISE_H_
#define LIB_JXL_RENDER_PIPELINE_STAGE_NOISE_H_

#include <cstddef>
#include <memory>

#include "lib/jxl/chroma_from_luma.h"
#include "lib/jxl/noise.h"
#include "lib/jxl/render_pipeline/render_pipeline_stage.h"

namespace jxl {

// Adds noise to color channels.
std::unique_ptr<RenderPipelineStage> GetAddNoiseStage(
    const NoiseParams& noise_params, const ColorCorrelation& color_correlation,
    size_t noise_c_start);

// Applies a 5x5 subtract-box-filter convolution to the noise input channels.
std::unique_ptr<RenderPipelineStage> GetConvolveNoiseStage(
    size_t noise_c_start);

}  // namespace jxl

#endif  // LIB_JXL_RENDER_PIPELINE_STAGE_NOISE_H_
