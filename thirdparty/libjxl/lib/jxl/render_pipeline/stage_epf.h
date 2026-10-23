// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_RENDER_PIPELINE_STAGE_EPF_H_
#define LIB_JXL_RENDER_PIPELINE_STAGE_EPF_H_

#include <cstdint>
#include <memory>

#include "lib/jxl/image.h"
#include "lib/jxl/loop_filter.h"
#include "lib/jxl/render_pipeline/render_pipeline_stage.h"

namespace jxl {

enum class EpfStage : uint8_t { Zero, One, Two };

// Applies the `epf_stage`-th EPF step with the given settings and `sigma`.
// `sigma` will be accessed with an offset of (kSigmaPadding, kSigmaPadding),
// and should have (kSigmaBorder, kSigmaBorder) mirrored sigma values available
// around the main image. See also filters.(h|cc)
std::unique_ptr<RenderPipelineStage> GetEPFStage(const LoopFilter& lf,
                                                 const ImageF& sigma,
                                                 EpfStage epf_stage);
}  // namespace jxl

#endif  // LIB_JXL_RENDER_PIPELINE_STAGE_EPF_H_
