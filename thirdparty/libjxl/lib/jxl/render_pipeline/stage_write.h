// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_RENDER_PIPELINE_STAGE_WRITE_H_
#define LIB_JXL_RENDER_PIPELINE_STAGE_WRITE_H_

#include <jxl/memory_manager.h>

#include <cstddef>
#include <memory>
#include <vector>

#include "lib/jxl/dec_cache.h"
#include "lib/jxl/dec_xyb.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/image_metadata.h"
#include "lib/jxl/render_pipeline/render_pipeline_stage.h"

namespace jxl {

std::unique_ptr<RenderPipelineStage> GetWriteToImageBundleStage(
    ImageBundle* image_bundle, const OutputEncodingInfo& output_encoding_info);

// Gets a stage to write color channels to an Image3F.
std::unique_ptr<RenderPipelineStage> GetWriteToImage3FStage(
    JxlMemoryManager* memory_manager, Image3F* image);

// Gets a stage to write to a pixel callback or image buffer.
std::unique_ptr<RenderPipelineStage> GetWriteToOutputStage(
    const ImageOutput& main_output, size_t width, size_t height, bool has_alpha,
    bool unpremul_alpha, size_t alpha_c, Orientation undo_orientation,
    std::vector<ImageOutput>& extra_output, JxlMemoryManager* memory_manager);

}  // namespace jxl

#endif  // LIB_JXL_RENDER_PIPELINE_STAGE_WRITE_H_
