// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_RENDER_PIPELINE_STAGE_PATCHES_H_
#define LIB_JXL_RENDER_PIPELINE_STAGE_PATCHES_H_

#include <memory>
#include <vector>

#include "lib/jxl/dec_patch_dictionary.h"
#include "lib/jxl/image_metadata.h"
#include "lib/jxl/render_pipeline/render_pipeline_stage.h"

namespace jxl {

// Draws patches if applicable.
std::unique_ptr<RenderPipelineStage> GetPatchesStage(
    const PatchDictionary* patches,
    const std::vector<ExtraChannelInfo>* extra_channel_info);

}  // namespace jxl

#endif  // LIB_JXL_RENDER_PIPELINE_STAGE_PATCHES_H_
