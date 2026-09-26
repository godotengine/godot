// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_RENDER_PIPELINE_STAGE_TONE_MAPPING_H_
#define LIB_JXL_RENDER_PIPELINE_STAGE_TONE_MAPPING_H_
#include <math.h>
#include <stdint.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "lib/jxl/dec_xyb.h"
#include "lib/jxl/render_pipeline/render_pipeline_stage.h"

namespace jxl {

// Tone maps the image if appropriate. It must be in linear space and
// `output_encoding_info.luminances` must contain the luminance for the
// primaries of that space. It must also be encoded such that (1, 1, 1)
// represents `output_encoding_info.orig_intensity_target` nits, unless
// `output_encoding_info.color_encoding.tf.IsPQ()`, in which case (1, 1, 1) must
// represent 10000 nits. This corresponds to what XYBStage outputs. After this
// stage, (1, 1, 1) will represent
// `output_encoding_info.desired_intensity_target` nits, except in the PQ
// special case in which it remains 10000.
//
// If no tone mapping is necessary, this will return nullptr.
std::unique_ptr<RenderPipelineStage> GetToneMappingStage(
    const OutputEncodingInfo& output_encoding_info);

}  // namespace jxl

#endif  // LIB_JXL_RENDER_PIPELINE_STAGE_TONE_MAPPING_H_
