// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_RENDER_PIPELINE_STAGE_UPSAMPLING_H_
#define LIB_JXL_RENDER_PIPELINE_STAGE_UPSAMPLING_H_
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "lib/jxl/image_metadata.h"
#include "lib/jxl/render_pipeline/render_pipeline_stage.h"

namespace jxl {

// Upsamples the given channel by the given factor.
std::unique_ptr<RenderPipelineStage> GetUpsamplingStage(
    const CustomTransformData& ups_factors, size_t c, size_t shift);
}  // namespace jxl

#endif  // LIB_JXL_RENDER_PIPELINE_STAGE_UPSAMPLING_H_
