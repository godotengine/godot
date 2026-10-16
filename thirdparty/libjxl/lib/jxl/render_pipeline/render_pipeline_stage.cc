// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/render_pipeline/render_pipeline_stage.h"

#include <cstddef>
#include <utility>
#include <vector>

#include "lib/jxl/base/status.h"
#include "lib/jxl/frame_header.h"

namespace jxl {

Status RenderPipelineStage::SetInputSizes(
    const std::vector<std::pair<size_t, size_t>>& input_sizes) {
  return true;
}

Status RenderPipelineStage::PrepareForThreads(size_t num_threads) {
  return true;
}

Status RenderPipelineStage::IsInitialized() const { return true; }

RenderPipelineStage::~RenderPipelineStage() = default;

void RenderPipelineStage::ProcessPaddingRow(const RowInfo& output_rows,
                                            size_t xsize, size_t xpos,
                                            size_t ypos) const {}

void RenderPipelineStage::GetImageDimensions(size_t* xsize, size_t* ysize,
                                             FrameOrigin* frame_origin) const {}

bool RenderPipelineStage::SwitchToImageDimensions() const { return false; }

}  // namespace jxl
