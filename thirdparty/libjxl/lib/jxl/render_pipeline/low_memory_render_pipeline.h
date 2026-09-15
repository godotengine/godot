// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_RENDER_PIPELINE_LOW_MEMORY_RENDER_PIPELINE_H_
#define LIB_JXL_RENDER_PIPELINE_LOW_MEMORY_RENDER_PIPELINE_H_

#include <jxl/memory_manager.h>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_group_border.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/image.h"
#include "lib/jxl/render_pipeline/render_pipeline.h"

namespace jxl {

// A multithreaded, low-memory rendering pipeline that only allocates a minimal
// amount of buffers.
class LowMemoryRenderPipeline final : public RenderPipeline {
 public:
  explicit LowMemoryRenderPipeline(JxlMemoryManager* memory_manager)
      : RenderPipeline(memory_manager) {}

 private:
  std::vector<std::pair<ImageF*, Rect>> PrepareBuffers(
      size_t group_id, size_t thread_id) override;

  Status PrepareForThreadsInternal(size_t num, bool use_group_ids) override;

  Status ProcessBuffers(size_t group_id, size_t thread_id) override;

  void ClearDone(size_t i) override { group_border_assigner_.ClearDone(i); }

  Status Init() override;

  Status EnsureBordersStorage();
  size_t GroupInputXSize(size_t c) const;
  size_t GroupInputYSize(size_t c) const;
  Status RenderRect(size_t thread_id, std::vector<ImageF>& input_data,
                    Rect data_max_color_channel_rect,
                    Rect image_max_color_channel_rect);
  Status RenderPadding(size_t thread_id, Rect rect);

  Status SaveBorders(size_t group_id, size_t c, const ImageF& in);
  Status LoadBorders(size_t group_id, size_t c, const Rect& r, ImageF* out);

  std::pair<size_t, size_t> ColorDimensionsToChannelDimensions(
      std::pair<size_t, size_t> in, size_t c, size_t stage) const;

  std::pair<size_t, size_t> BorderToStore(size_t c) const;

  bool use_group_ids_;

  // Storage for borders between groups. Borders of adjacent groups are stacked
  // together, e.g. bottom border of current group is followed by top border
  // of next group.
  std::vector<ImageF> borders_horizontal_;
  std::vector<ImageF> borders_vertical_;

  // Manages the status of borders.
  GroupBorderAssigner group_border_assigner_;

  // Size (in color-channel-pixels) of the border around each group that might
  // be assigned to that group.
  std::pair<size_t, size_t> group_border_;
  // base_color_shift_ defines the size of groups in terms of final image
  // pixels.
  size_t base_color_shift_;

  // Buffer for decoded pixel data for a group, indexed by [thread][channel] or
  // [group][channel] depending on `use_group_ids_`.
  std::vector<std::vector<ImageF>> group_data_;

  // Borders for storing group data.
  size_t group_data_x_border_;
  size_t group_data_y_border_;

  // Buffers for intermediate rows for the various stages, indexed by
  // [thread][channel][stage].
  std::vector<std::vector<std::vector<ImageF>>> stage_data_;

  // Buffers for out-of-frame data, indexed by [thread]; every row is a
  // different channel.
  std::vector<ImageF> out_of_frame_data_;

  // For each stage, a non-kIgnored channel.
  std::vector<int32_t> anyc_;

  // Size of the image at each stage.
  std::vector<Rect> image_rect_;

  // For each stage, for each channel, keep track of the kInOut stage that
  // produced the input to that stage (which corresponds to the buffer index
  // containing the data). -1 if data comes from the original input.
  std::vector<std::vector<int32_t>> stage_input_for_channel_;

  // Number of (virtual) extra rows that must be processed at each stage
  // to produce sufficient output for future stages.
  std::vector<int> virtual_ypadding_for_output_;

  // Same thing for columns, except these are real columns and not virtual ones.
  std::vector<int> xpadding_for_output_;

  // First stage that doesn't have any kInOut channel.
  size_t first_trailing_stage_;

  // Origin and size of the frame after switching to image dimensions.
  FrameOrigin frame_origin_;
  size_t full_image_xsize_;
  size_t full_image_ysize_;
  size_t first_image_dim_stage_;
};

}  // namespace jxl

#endif  // LIB_JXL_RENDER_PIPELINE_LOW_MEMORY_RENDER_PIPELINE_H_
