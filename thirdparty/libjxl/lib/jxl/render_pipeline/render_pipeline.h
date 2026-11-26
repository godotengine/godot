// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_RENDER_PIPELINE_RENDER_PIPELINE_H_
#define LIB_JXL_RENDER_PIPELINE_RENDER_PIPELINE_H_

#include <jxl/memory_manager.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/image.h"
#include "lib/jxl/render_pipeline/render_pipeline_stage.h"

namespace jxl {

// Interface to provide input to the rendering pipeline. When this object is
// destroyed, all the data in the provided ImageF's Rects must have been
// initialized.
class RenderPipelineInput {
 public:
  RenderPipelineInput(const RenderPipelineInput&) = delete;
  RenderPipelineInput(RenderPipelineInput&& other) noexcept {
    *this = std::move(other);
  }
  RenderPipelineInput& operator=(RenderPipelineInput&& other) noexcept {
    pipeline_ = other.pipeline_;
    group_id_ = other.group_id_;
    thread_id_ = other.thread_id_;
    buffers_ = std::move(other.buffers_);
    other.pipeline_ = nullptr;
    return *this;
  }

  RenderPipelineInput() = default;
  Status Done();

  const std::pair<ImageF*, Rect>& GetBuffer(size_t c) const {
    JXL_DASSERT(c < buffers_.size());
    return buffers_[c];
  }

 private:
  RenderPipeline* pipeline_ = nullptr;
  size_t group_id_;
  size_t thread_id_;
  std::vector<std::pair<ImageF*, Rect>> buffers_;
  friend class RenderPipeline;
};

class RenderPipeline {
 public:
  class Builder {
   public:
    explicit Builder(JxlMemoryManager* memory_manager, size_t num_c)
        : memory_manager_(memory_manager), num_c_(num_c) {
      JXL_DASSERT(num_c > 0);
    }

    // Adds a stage to the pipeline. Must be called at least once; the last
    // added stage cannot have kInOut channels.
    Status AddStage(std::unique_ptr<RenderPipelineStage> stage);

    // Enables using the simple (i.e. non-memory-efficient) implementation of
    // the pipeline.
    void UseSimpleImplementation() { use_simple_implementation_ = true; }

    // Finalizes setup of the pipeline. Shifts for all channels should be 0 at
    // this point.
    StatusOr<std::unique_ptr<RenderPipeline>> Finalize(
        FrameDimensions frame_dimensions) &&;

   private:
    JxlMemoryManager* memory_manager_;
    std::vector<std::unique_ptr<RenderPipelineStage>> stages_;
    size_t num_c_;
    bool use_simple_implementation_ = false;
  };

  friend class Builder;

  virtual ~RenderPipeline() = default;

  Status IsInitialized() const {
    for (const auto& stage : stages_) {
      JXL_RETURN_IF_ERROR(stage->IsInitialized());
    }
    return true;
  }

  // Allocates storage to run with `num` threads. If `use_group_ids` is true,
  // storage is allocated for each group, not each thread. The behaviour is
  // undefined if calling this function multiple times with a different value
  // for `use_group_ids`.
  Status PrepareForThreads(size_t num, bool use_group_ids);

  // Retrieves a buffer where input data should be stored by the callee. When
  // input has been provided for all buffers, the pipeline will complete its
  // processing. This method may be called multiple times concurrently from
  // different threads, provided that a different `thread_id` is given.
  RenderPipelineInput GetInputBuffers(size_t group_id, size_t thread_id);

  size_t PassesWithAllInput() const {
    return *std::min_element(group_completed_passes_.begin(),
                             group_completed_passes_.end());
  }

  virtual void ClearDone(size_t i) {}

 protected:
  explicit RenderPipeline(JxlMemoryManager* memory_manager)
      : memory_manager_(memory_manager) {}
  JxlMemoryManager* memory_manager_;

  std::vector<std::unique_ptr<RenderPipelineStage>> stages_;
  // Shifts for every channel at the input of each stage.
  std::vector<std::vector<std::pair<size_t, size_t>>> channel_shifts_;

  // Amount of (cumulative) padding required by each stage and channel, in
  // either direction.
  std::vector<std::vector<std::pair<size_t, size_t>>> padding_;

  FrameDimensions frame_dimensions_;

  std::vector<uint8_t> group_completed_passes_;

  friend class RenderPipelineInput;

 private:
  Status InputReady(size_t group_id, size_t thread_id,
                    const std::vector<std::pair<ImageF*, Rect>>& buffers);

  virtual std::vector<std::pair<ImageF*, Rect>> PrepareBuffers(
      size_t group_id, size_t thread_id) = 0;

  virtual Status ProcessBuffers(size_t group_id, size_t thread_id) = 0;

  // Note that this method may be called multiple times with different (or
  // equal) `num`.
  virtual Status PrepareForThreadsInternal(size_t num, bool use_group_ids) = 0;

  // Called once frame dimensions and stages are known.
  virtual Status Init() { return true; }
};

}  // namespace jxl

#endif  // LIB_JXL_RENDER_PIPELINE_RENDER_PIPELINE_H_
