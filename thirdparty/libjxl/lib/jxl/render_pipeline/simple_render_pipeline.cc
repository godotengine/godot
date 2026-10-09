// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/render_pipeline/simple_render_pipeline.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <hwy/base.h>
#include <utility>
#include <vector>

#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/sanitizers.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/image_ops.h"
#include "lib/jxl/render_pipeline/render_pipeline_stage.h"

namespace jxl {

Status SimpleRenderPipeline::PrepareForThreadsInternal(size_t num,
                                                       bool use_group_ids) {
  if (!channel_data_.empty()) {
    return true;
  }
  auto ch_size = [](size_t frame_size, size_t shift) {
    return DivCeil(frame_size, 1 << shift) + kRenderPipelineXOffset * 2;
  };
  for (auto& entry : channel_shifts_[0]) {
    JXL_ASSIGN_OR_RETURN(
        ImageF ch,
        ImageF::Create(
            memory_manager_,
            ch_size(frame_dimensions_.xsize_upsampled, entry.first),
            ch_size(frame_dimensions_.ysize_upsampled, entry.second)));
    channel_data_.push_back(std::move(ch));
    msan::PoisonImage(channel_data_.back());
  }
  return true;
}

Rect SimpleRenderPipeline::MakeChannelRect(size_t group_id, size_t channel) {
  size_t base_color_shift =
      CeilLog2Nonzero(frame_dimensions_.xsize_upsampled_padded /
                      frame_dimensions_.xsize_padded);

  const size_t gx = group_id % frame_dimensions_.xsize_groups;
  const size_t gy = group_id / frame_dimensions_.xsize_groups;
  size_t xgroupdim = (frame_dimensions_.group_dim << base_color_shift) >>
                     channel_shifts_[0][channel].first;
  size_t ygroupdim = (frame_dimensions_.group_dim << base_color_shift) >>
                     channel_shifts_[0][channel].second;
  return Rect(
      kRenderPipelineXOffset + gx * xgroupdim,
      kRenderPipelineXOffset + gy * ygroupdim, xgroupdim, ygroupdim,
      kRenderPipelineXOffset + DivCeil(frame_dimensions_.xsize_upsampled,
                                       1 << channel_shifts_[0][channel].first),
      kRenderPipelineXOffset +
          DivCeil(frame_dimensions_.ysize_upsampled,
                  1 << channel_shifts_[0][channel].second));
}

std::vector<std::pair<ImageF*, Rect>> SimpleRenderPipeline::PrepareBuffers(
    size_t group_id, size_t thread_id) {
  std::vector<std::pair<ImageF*, Rect>> ret;
  for (size_t c = 0; c < channel_data_.size(); c++) {
    ret.emplace_back(&channel_data_[c], MakeChannelRect(group_id, c));
  }
  return ret;
}

Status SimpleRenderPipeline::ProcessBuffers(size_t group_id, size_t thread_id) {
  for (size_t c = 0; c < channel_data_.size(); c++) {
    Rect r = MakeChannelRect(group_id, c);
    (void)r;
    JXL_CHECK_PLANE_INITIALIZED(channel_data_[c], r, c);
  }

  if (PassesWithAllInput() <= processed_passes_) return true;
  processed_passes_++;

  for (size_t stage_id = 0; stage_id < stages_.size(); stage_id++) {
    const auto& stage = stages_[stage_id];
    // Prepare buffers for kInOut channels.
    std::vector<ImageF> new_channels(channel_data_.size());
    std::vector<ImageF*> output_channels(channel_data_.size());

    std::vector<std::pair<size_t, size_t>> input_sizes(channel_data_.size());
    for (size_t c = 0; c < channel_data_.size(); c++) {
      input_sizes[c] =
          std::make_pair(channel_data_[c].xsize() - kRenderPipelineXOffset * 2,
                         channel_data_[c].ysize() - kRenderPipelineXOffset * 2);
    }

    for (size_t c = 0; c < channel_data_.size(); c++) {
      if (stage->GetChannelMode(c) != RenderPipelineChannelMode::kInOut) {
        continue;
      }
      // Ensure that the newly allocated channels are large enough to avoid
      // problems with padding.
      JXL_ASSIGN_OR_RETURN(
          new_channels[c],
          ImageF::Create(memory_manager_,
                         frame_dimensions_.xsize_upsampled_padded +
                             kRenderPipelineXOffset * 2 +
                             hwy::kMaxVectorSize * 8,
                         frame_dimensions_.ysize_upsampled_padded +
                             kRenderPipelineXOffset * 2));
      JXL_RETURN_IF_ERROR(new_channels[c].ShrinkTo(
          (input_sizes[c].first << stage->settings_.shift_x) +
              kRenderPipelineXOffset * 2,
          (input_sizes[c].second << stage->settings_.shift_y) +
              kRenderPipelineXOffset * 2));
      output_channels[c] = &new_channels[c];
    }

    auto get_row = [&](size_t c, int64_t y) {
      return channel_data_[c].Row(kRenderPipelineXOffset + y) +
             kRenderPipelineXOffset;
    };

    // Add mirrored pixels to all kInOut channels.
    for (size_t c = 0; c < channel_data_.size(); c++) {
      if (stage->GetChannelMode(c) != RenderPipelineChannelMode::kInOut) {
        continue;
      }
      // Horizontal mirroring.
      for (size_t y = 0; y < input_sizes[c].second; y++) {
        float* row = get_row(c, y);
        for (size_t ix = 0; ix < stage->settings_.border_x; ix++) {
          *(row - ix - 1) =
              row[Mirror(-static_cast<ssize_t>(ix) - 1, input_sizes[c].first)];
        }
        for (size_t ix = 0; ix < stage->settings_.border_x; ix++) {
          *(row + ix + input_sizes[c].first) =
              row[Mirror(ix + input_sizes[c].first, input_sizes[c].first)];
        }
      }
      // Vertical mirroring.
      for (int y = 0; y < static_cast<int>(stage->settings_.border_y); y++) {
        memcpy(get_row(c, -y - 1) - stage->settings_.border_x,
               get_row(c, Mirror(-static_cast<ssize_t>(y) - 1,
                                 input_sizes[c].second)) -
                   stage->settings_.border_x,
               sizeof(float) *
                   (input_sizes[c].first + 2 * stage->settings_.border_x));
      }
      for (int y = 0; y < static_cast<int>(stage->settings_.border_y); y++) {
        memcpy(
            get_row(c, input_sizes[c].second + y) - stage->settings_.border_x,
            get_row(c,
                    Mirror(input_sizes[c].second + y, input_sizes[c].second)) -
                stage->settings_.border_x,
            sizeof(float) *
                (input_sizes[c].first + 2 * stage->settings_.border_x));
      }
    }

    size_t ysize = 0;
    size_t xsize = 0;
    for (size_t c = 0; c < channel_data_.size(); c++) {
      if (stage->GetChannelMode(c) == RenderPipelineChannelMode::kIgnored) {
        continue;
      }
      ysize = std::max(input_sizes[c].second, ysize);
      xsize = std::max(input_sizes[c].first, xsize);
    }

    JXL_ENSURE(ysize != 0);
    JXL_ENSURE(xsize != 0);

    RenderPipelineStage::RowInfo input_rows(channel_data_.size());
    RenderPipelineStage::RowInfo output_rows(channel_data_.size());

    // Run the pipeline.
    {
      JXL_RETURN_IF_ERROR(stage->SetInputSizes(input_sizes));
      int border_y = stage->settings_.border_y;
      for (size_t y = 0; y < ysize; y++) {
        // Prepare input rows.
        for (size_t c = 0; c < channel_data_.size(); c++) {
          if (stage->GetChannelMode(c) == RenderPipelineChannelMode::kIgnored) {
            continue;
          }
          input_rows[c].resize(2 * border_y + 1);
          for (int iy = -border_y; iy <= border_y; iy++) {
            input_rows[c][iy + border_y] =
                channel_data_[c].Row(y + kRenderPipelineXOffset + iy);
          }
        }
        // Prepare output rows.
        for (size_t c = 0; c < channel_data_.size(); c++) {
          if (!output_channels[c]) continue;
          output_rows[c].resize(1 << stage->settings_.shift_y);
          for (size_t iy = 0; iy < output_rows[c].size(); iy++) {
            output_rows[c][iy] = output_channels[c]->Row(
                (y << stage->settings_.shift_y) + iy + kRenderPipelineXOffset);
          }
        }
        JXL_RETURN_IF_ERROR(stage->ProcessRow(input_rows, output_rows,
                                              /*xextra=*/0, xsize,
                                              /*xpos=*/0, y, thread_id));
      }
    }

    // Move new channels to current channels.
    for (size_t c = 0; c < channel_data_.size(); c++) {
      if (stage->GetChannelMode(c) != RenderPipelineChannelMode::kInOut) {
        continue;
      }
      channel_data_[c] = std::move(new_channels[c]);
    }
    for (size_t c = 0; c < channel_data_.size(); c++) {
      size_t next_stage = std::min(stage_id + 1, channel_shifts_.size() - 1);
      size_t xsize = DivCeil(frame_dimensions_.xsize_upsampled,
                             1 << channel_shifts_[next_stage][c].first);
      size_t ysize = DivCeil(frame_dimensions_.ysize_upsampled,
                             1 << channel_shifts_[next_stage][c].second);
      JXL_RETURN_IF_ERROR(
          channel_data_[c].ShrinkTo(xsize + 2 * kRenderPipelineXOffset,
                                    ysize + 2 * kRenderPipelineXOffset));
      JXL_CHECK_PLANE_INITIALIZED(
          channel_data_[c],
          Rect(kRenderPipelineXOffset, kRenderPipelineXOffset, xsize, ysize),
          c);
    }

    if (stage->SwitchToImageDimensions()) {
      size_t image_xsize;
      size_t image_ysize;
      FrameOrigin frame_origin;
      stage->GetImageDimensions(&image_xsize, &image_ysize, &frame_origin);
      frame_dimensions_.Set(image_xsize, image_ysize, 0, 0, 0, false, 1);
      std::vector<ImageF> old_channels = std::move(channel_data_);
      channel_data_.clear();
      channel_data_.reserve(old_channels.size());
      for (size_t c = 0; c < old_channels.size(); c++) {
        JXL_ASSIGN_OR_RETURN(
            ImageF ch,
            ImageF::Create(memory_manager_,
                           2 * kRenderPipelineXOffset + image_xsize,
                           2 * kRenderPipelineXOffset + image_ysize));
        channel_data_.emplace_back(std::move(ch));
      }
      for (size_t y = 0; y < image_ysize; ++y) {
        for (size_t c = 0; c < channel_data_.size(); c++) {
          output_rows[c].resize(1);
          output_rows[c][0] = channel_data_[c].Row(kRenderPipelineXOffset + y);
        }
        // TODO(sboukortt): consider doing this only on the parts of the
        // background that won't be occluded.
        stage->ProcessPaddingRow(output_rows, image_xsize, 0, y);
      }
      ssize_t x0 = frame_origin.x0;
      ssize_t y0 = frame_origin.y0;
      size_t x0_fg = 0;
      size_t y0_fg = 0;
      if (x0 < 0) {
        xsize += x0;
        x0_fg -= x0;
        x0 = 0;
      }
      if (x0 + xsize > image_xsize) {
        xsize = image_xsize - x0;
      }
      if (y0 < 0) {
        ysize += y0;
        y0_fg -= x0;
        y0 = 0;
      }
      if (y0 + ysize > image_ysize) {
        ysize = image_ysize - y0;
      }
      const Rect rect_fg_relative_to_image =
          Rect(x0, y0, xsize, ysize)
              .Translate(kRenderPipelineXOffset, kRenderPipelineXOffset);
      const Rect rect_fg =
          Rect(x0_fg, y0_fg, xsize, ysize)
              .Translate(kRenderPipelineXOffset, kRenderPipelineXOffset);
      for (size_t c = 0; c < channel_data_.size(); c++) {
        JXL_RETURN_IF_ERROR(CopyImageTo(rect_fg, old_channels[c],
                                        rect_fg_relative_to_image,
                                        &channel_data_[c]));
      }
    }
  }
  return true;
}
}  // namespace jxl
