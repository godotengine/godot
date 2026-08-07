// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/render_pipeline/render_pipeline.h"

#include <jxl/memory_manager.h>

#include <memory>
#include <utility>

#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/sanitizers.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/render_pipeline/low_memory_render_pipeline.h"
#include "lib/jxl/render_pipeline/simple_render_pipeline.h"

namespace jxl {

Status RenderPipeline::Builder::AddStage(
    std::unique_ptr<RenderPipelineStage> stage) {
  if (!stage) return JXL_FAILURE("internal: no stage to add");
  stages_.push_back(std::move(stage));
  return true;
}

StatusOr<std::unique_ptr<RenderPipeline>> RenderPipeline::Builder::Finalize(
    FrameDimensions frame_dimensions) && {
  // Check that the last stage is not a kInOut stage for any channel, and that
  // there is at least one stage.
  JXL_ENSURE(!stages_.empty());
  for (size_t c = 0; c < num_c_; c++) {
    JXL_ENSURE(stages_.back()->GetChannelMode(c) !=
               RenderPipelineChannelMode::kInOut);
  }

  std::unique_ptr<RenderPipeline> res;
  if (use_simple_implementation_) {
    res = jxl::make_unique<SimpleRenderPipeline>(memory_manager_);
  } else {
    res = jxl::make_unique<LowMemoryRenderPipeline>(memory_manager_);
  }

  res->padding_.resize(stages_.size());
  for (size_t i = stages_.size(); i-- > 0;) {
    const auto& stage = stages_[i];
    res->padding_[i].resize(num_c_);
    if (i + 1 == stages_.size()) {
      continue;
    }
    for (size_t c = 0; c < num_c_; c++) {
      if (stage->GetChannelMode(c) == RenderPipelineChannelMode::kInOut) {
        res->padding_[i][c].first = DivCeil(res->padding_[i + 1][c].first,
                                            1 << stage->settings_.shift_x) +
                                    stage->settings_.border_x;
        res->padding_[i][c].second = DivCeil(res->padding_[i + 1][c].second,
                                             1 << stage->settings_.shift_y) +
                                     stage->settings_.border_y;
      } else {
        res->padding_[i][c] = res->padding_[i + 1][c];
      }
    }
  }

  res->frame_dimensions_ = frame_dimensions;
  res->group_completed_passes_.resize(frame_dimensions.num_groups);
  res->channel_shifts_.resize(stages_.size());
  res->channel_shifts_[0].resize(num_c_);
  for (size_t i = 1; i < stages_.size(); i++) {
    auto& stage = stages_[i - 1];
    for (size_t c = 0; c < num_c_; c++) {
      if (stage->GetChannelMode(c) == RenderPipelineChannelMode::kInOut) {
        res->channel_shifts_[0][c].first += stage->settings_.shift_x;
        res->channel_shifts_[0][c].second += stage->settings_.shift_y;
      }
    }
  }
  for (size_t i = 1; i < stages_.size(); i++) {
    auto& stage = stages_[i - 1];
    res->channel_shifts_[i].resize(num_c_);
    for (size_t c = 0; c < num_c_; c++) {
      if (stage->GetChannelMode(c) == RenderPipelineChannelMode::kInOut) {
        res->channel_shifts_[i][c].first =
            res->channel_shifts_[i - 1][c].first - stage->settings_.shift_x;
        res->channel_shifts_[i][c].second =
            res->channel_shifts_[i - 1][c].second - stage->settings_.shift_y;
      } else {
        res->channel_shifts_[i][c].first = res->channel_shifts_[i - 1][c].first;
        res->channel_shifts_[i][c].second =
            res->channel_shifts_[i - 1][c].second;
      }
    }
  }
  res->stages_ = std::move(stages_);
  JXL_RETURN_IF_ERROR(res->Init());
  return res;
}

RenderPipelineInput RenderPipeline::GetInputBuffers(size_t group_id,
                                                    size_t thread_id) {
  RenderPipelineInput ret;
  JXL_DASSERT(group_id < group_completed_passes_.size());
  ret.group_id_ = group_id;
  ret.thread_id_ = thread_id;
  ret.pipeline_ = this;
  ret.buffers_ = PrepareBuffers(group_id, thread_id);
  return ret;
}

Status RenderPipeline::InputReady(
    size_t group_id, size_t thread_id,
    const std::vector<std::pair<ImageF*, Rect>>& buffers) {
  JXL_ENSURE(group_id < group_completed_passes_.size());
  group_completed_passes_[group_id]++;
  for (size_t i = 0; i < buffers.size(); ++i) {
    (void)i;
    JXL_CHECK_PLANE_INITIALIZED(*buffers[i].first, buffers[i].second, i);
  }

  JXL_RETURN_IF_ERROR(ProcessBuffers(group_id, thread_id));
  return true;
}

Status RenderPipeline::PrepareForThreads(size_t num, bool use_group_ids) {
  for (const auto& stage : stages_) {
    JXL_RETURN_IF_ERROR(stage->PrepareForThreads(num));
  }
  JXL_RETURN_IF_ERROR(PrepareForThreadsInternal(num, use_group_ids));
  return true;
}

Status RenderPipelineInput::Done() {
  JXL_ENSURE(pipeline_);
  JXL_RETURN_IF_ERROR(pipeline_->InputReady(group_id_, thread_id_, buffers_));
  return true;
}

}  // namespace jxl
