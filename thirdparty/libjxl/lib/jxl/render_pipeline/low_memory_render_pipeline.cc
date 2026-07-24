// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/render_pipeline/low_memory_render_pipeline.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "lib/jxl/base/arch_macros.h"
#include "lib/jxl/base/common.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_ops.h"

namespace jxl {
std::pair<size_t, size_t>
LowMemoryRenderPipeline::ColorDimensionsToChannelDimensions(
    std::pair<size_t, size_t> in, size_t c, size_t stage) const {
  std::pair<size_t, size_t> ret;
  std::pair<size_t, size_t> shift = channel_shifts_[stage][c];
  ret.first =
      ((in.first << base_color_shift_) + (1 << shift.first) - 1) >> shift.first;
  ret.second = ((in.second << base_color_shift_) + (1 << shift.second) - 1) >>
               shift.second;
  return ret;
}

std::pair<size_t, size_t> LowMemoryRenderPipeline::BorderToStore(
    size_t c) const {
  auto ret = ColorDimensionsToChannelDimensions(group_border_, c, 0);
  ret.first += padding_[0][c].first;
  ret.second += padding_[0][c].second;
  return ret;
}

Status LowMemoryRenderPipeline::SaveBorders(size_t group_id, size_t c,
                                            const ImageF& in) {
  size_t gy = group_id / frame_dimensions_.xsize_groups;
  size_t gx = group_id % frame_dimensions_.xsize_groups;
  size_t hshift = channel_shifts_[0][c].first;
  size_t vshift = channel_shifts_[0][c].second;
  size_t x0 = gx * GroupInputXSize(c);
  size_t x1 = std::min((gx + 1) * GroupInputXSize(c),
                       DivCeil(frame_dimensions_.xsize_upsampled, 1 << hshift));
  size_t y0 = gy * GroupInputYSize(c);
  size_t y1 = std::min((gy + 1) * GroupInputYSize(c),
                       DivCeil(frame_dimensions_.ysize_upsampled, 1 << vshift));

  auto borders = BorderToStore(c);
  size_t borderx_write = borders.first;
  size_t bordery_write = borders.second;

  if (gy > 0) {
    Rect from(group_data_x_border_, group_data_y_border_, x1 - x0,
              bordery_write);
    Rect to(x0, (gy * 2 - 1) * bordery_write, x1 - x0, bordery_write);
    JXL_RETURN_IF_ERROR(CopyImageTo(from, in, to, &borders_horizontal_[c]));
  }
  if (gy + 1 < frame_dimensions_.ysize_groups) {
    Rect from(group_data_x_border_,
              group_data_y_border_ + y1 - y0 - bordery_write, x1 - x0,
              bordery_write);
    Rect to(x0, (gy * 2) * bordery_write, x1 - x0, bordery_write);
    JXL_RETURN_IF_ERROR(CopyImageTo(from, in, to, &borders_horizontal_[c]));
  }
  if (gx > 0) {
    Rect from(group_data_x_border_, group_data_y_border_, borderx_write,
              y1 - y0);
    Rect to((gx * 2 - 1) * borderx_write, y0, borderx_write, y1 - y0);
    JXL_RETURN_IF_ERROR(CopyImageTo(from, in, to, &borders_vertical_[c]));
  }
  if (gx + 1 < frame_dimensions_.xsize_groups) {
    Rect from(group_data_x_border_ + x1 - x0 - borderx_write,
              group_data_y_border_, borderx_write, y1 - y0);
    Rect to((gx * 2) * borderx_write, y0, borderx_write, y1 - y0);
    JXL_RETURN_IF_ERROR(CopyImageTo(from, in, to, &borders_vertical_[c]));
  }
  return true;
}

Status LowMemoryRenderPipeline::LoadBorders(size_t group_id, size_t c,
                                            const Rect& r, ImageF* out) {
  size_t gy = group_id / frame_dimensions_.xsize_groups;
  size_t gx = group_id % frame_dimensions_.xsize_groups;
  size_t hshift = channel_shifts_[0][c].first;
  size_t vshift = channel_shifts_[0][c].second;
  // Coordinates of the group in the image.
  size_t x0 = gx * GroupInputXSize(c);
  size_t x1 = std::min((gx + 1) * GroupInputXSize(c),
                       DivCeil(frame_dimensions_.xsize_upsampled, 1 << hshift));
  size_t y0 = gy * GroupInputYSize(c);
  size_t y1 = std::min((gy + 1) * GroupInputYSize(c),
                       DivCeil(frame_dimensions_.ysize_upsampled, 1 << vshift));

  size_t paddingx = padding_[0][c].first;
  size_t paddingy = padding_[0][c].second;

  auto borders = BorderToStore(c);
  size_t borderx_write = borders.first;
  size_t bordery_write = borders.second;

  // Limits of the area to copy from, in image coordinates.
  JXL_ENSURE(r.x0() == 0 || (r.x0() << base_color_shift_) >= paddingx);
  size_t x0src = DivCeil(r.x0() << base_color_shift_, 1 << hshift);
  if (x0src != 0) {
    x0src -= paddingx;
  }
  // r may be such that r.x1 (namely x0() + xsize()) is within paddingx of the
  // right side of the image, so we use min() here.
  size_t x1src =
      DivCeil((r.x0() + r.xsize()) << base_color_shift_, 1 << hshift);
  x1src = std::min(x1src + paddingx,
                   DivCeil(frame_dimensions_.xsize_upsampled, 1 << hshift));

  // Similar computation for y.
  JXL_ENSURE(r.y0() == 0 || (r.y0() << base_color_shift_) >= paddingy);
  size_t y0src = DivCeil(r.y0() << base_color_shift_, 1 << vshift);
  if (y0src != 0) {
    y0src -= paddingy;
  }
  size_t y1src =
      DivCeil((r.y0() + r.ysize()) << base_color_shift_, 1 << vshift);
  y1src = std::min(y1src + paddingy,
                   DivCeil(frame_dimensions_.ysize_upsampled, 1 << vshift));

  // Copy other groups' borders from the border storage.
  if (y0src < y0) {
    JXL_ENSURE(gy > 0);
    JXL_RETURN_IF_ERROR(CopyImageTo(
        Rect(x0src, (gy * 2 - 2) * bordery_write, x1src - x0src, bordery_write),
        borders_horizontal_[c],
        Rect(group_data_x_border_ + x0src - x0,
             group_data_y_border_ - bordery_write, x1src - x0src,
             bordery_write),
        out));
  }
  if (y1src > y1) {
    // When copying the bottom border we must not be on the bottom groups.
    JXL_ENSURE(gy + 1 < frame_dimensions_.ysize_groups);
    JXL_RETURN_IF_ERROR(CopyImageTo(
        Rect(x0src, (gy * 2 + 1) * bordery_write, x1src - x0src, bordery_write),
        borders_horizontal_[c],
        Rect(group_data_x_border_ + x0src - x0, group_data_y_border_ + y1 - y0,
             x1src - x0src, bordery_write),
        out));
  }
  if (x0src < x0) {
    JXL_ENSURE(gx > 0);
    JXL_RETURN_IF_ERROR(CopyImageTo(
        Rect((gx * 2 - 2) * borderx_write, y0src, borderx_write, y1src - y0src),
        borders_vertical_[c],
        Rect(group_data_x_border_ - borderx_write,
             group_data_y_border_ + y0src - y0, borderx_write, y1src - y0src),
        out));
  }
  if (x1src > x1) {
    // When copying the right border we must not be on the rightmost groups.
    JXL_ENSURE(gx + 1 < frame_dimensions_.xsize_groups);
    JXL_RETURN_IF_ERROR(CopyImageTo(
        Rect((gx * 2 + 1) * borderx_write, y0src, borderx_write, y1src - y0src),
        borders_vertical_[c],
        Rect(group_data_x_border_ + x1 - x0, group_data_y_border_ + y0src - y0,
             borderx_write, y1src - y0src),
        out));
  }
  return true;
}

size_t LowMemoryRenderPipeline::GroupInputXSize(size_t c) const {
  return (frame_dimensions_.group_dim << base_color_shift_) >>
         channel_shifts_[0][c].first;
}

size_t LowMemoryRenderPipeline::GroupInputYSize(size_t c) const {
  return (frame_dimensions_.group_dim << base_color_shift_) >>
         channel_shifts_[0][c].second;
}

Status LowMemoryRenderPipeline::EnsureBordersStorage() {
  const auto& shifts = channel_shifts_[0];
  if (borders_horizontal_.size() < shifts.size()) {
    borders_horizontal_.resize(shifts.size());
    borders_vertical_.resize(shifts.size());
  }
  for (size_t c = 0; c < shifts.size(); c++) {
    auto borders = BorderToStore(c);
    size_t borderx = borders.first;
    size_t bordery = borders.second;
    JXL_ENSURE(frame_dimensions_.xsize_groups > 0);
    size_t num_xborders = (frame_dimensions_.xsize_groups - 1) * 2;
    JXL_ENSURE(frame_dimensions_.ysize_groups > 0);
    size_t num_yborders = (frame_dimensions_.ysize_groups - 1) * 2;
    size_t downsampled_xsize =
        DivCeil(frame_dimensions_.xsize_upsampled_padded, 1 << shifts[c].first);
    size_t downsampled_ysize = DivCeil(frame_dimensions_.ysize_upsampled_padded,
                                       1 << shifts[c].second);
    Rect horizontal = Rect(0, 0, downsampled_xsize, bordery * num_yborders);
    if (!SameSize(horizontal, borders_horizontal_[c])) {
      JXL_ASSIGN_OR_RETURN(borders_horizontal_[c],
                           ImageF::Create(memory_manager_, horizontal.xsize(),
                                          horizontal.ysize()));
    }
    Rect vertical = Rect(0, 0, borderx * num_xborders, downsampled_ysize);
    if (!SameSize(vertical, borders_vertical_[c])) {
      JXL_ASSIGN_OR_RETURN(
          borders_vertical_[c],
          ImageF::Create(memory_manager_, vertical.xsize(), vertical.ysize()));
    }
  }
  return true;
}

Status LowMemoryRenderPipeline::Init() {
  group_border_ = {0, 0};
  base_color_shift_ = CeilLog2Nonzero(frame_dimensions_.xsize_upsampled_padded /
                                      frame_dimensions_.xsize_padded);

  const auto& shifts = channel_shifts_[0];

  // Ensure that each channel has enough many border pixels.
  for (size_t c = 0; c < shifts.size(); c++) {
    group_border_.first =
        std::max(group_border_.first,
                 DivCeil(padding_[0][c].first << channel_shifts_[0][c].first,
                         1 << base_color_shift_));
    group_border_.second =
        std::max(group_border_.second,
                 DivCeil(padding_[0][c].second << channel_shifts_[0][c].second,
                         1 << base_color_shift_));
  }

  // Ensure that all channels have an integer number of border pixels in the
  // input.
  for (size_t c = 0; c < shifts.size(); c++) {
    if (channel_shifts_[0][c].first >= base_color_shift_) {
      group_border_.first =
          RoundUpTo(group_border_.first,
                    1 << (channel_shifts_[0][c].first - base_color_shift_));
    }
    if (channel_shifts_[0][c].second >= base_color_shift_) {
      group_border_.second =
          RoundUpTo(group_border_.second,
                    1 << (channel_shifts_[0][c].second - base_color_shift_));
    }
  }
  // Ensure that the X border on color channels is a multiple of kBlockDim or
  // the vector size (required for EPF stages). Vectors on ARM NEON are never
  // wider than 4 floats, so rounding to multiples of 4 is enough.
#if JXL_ARCH_ARM
  constexpr size_t kGroupXAlign = 4;
#else
  constexpr size_t kGroupXAlign = 16;
#endif
  group_border_.first = RoundUpTo(group_border_.first, kGroupXAlign);
  // Allocate borders in group images that are just enough for storing the
  // borders to be copied in, plus any rounding to ensure alignment.
  std::pair<size_t, size_t> max_border = {0, 0};
  for (size_t c = 0; c < shifts.size(); c++) {
    max_border.first = std::max(BorderToStore(c).first, max_border.first);
    max_border.second = std::max(BorderToStore(c).second, max_border.second);
  }
  group_data_x_border_ = RoundUpTo(max_border.first, kGroupXAlign);
  group_data_y_border_ = max_border.second;

  JXL_RETURN_IF_ERROR(EnsureBordersStorage());
  group_border_assigner_.Init(frame_dimensions_);

  for (first_trailing_stage_ = stages_.size(); first_trailing_stage_ > 0;
       first_trailing_stage_--) {
    bool has_inout_c = false;
    for (size_t c = 0; c < shifts.size(); c++) {
      if (stages_[first_trailing_stage_ - 1]->GetChannelMode(c) ==
          RenderPipelineChannelMode::kInOut) {
        has_inout_c = true;
      }
    }
    if (has_inout_c) {
      break;
    }
  }

  first_image_dim_stage_ = stages_.size();
  for (size_t i = 0; i < stages_.size(); i++) {
    std::vector<std::pair<size_t, size_t>> input_sizes(shifts.size());
    for (size_t c = 0; c < shifts.size(); c++) {
      input_sizes[c] =
          std::make_pair(DivCeil(frame_dimensions_.xsize_upsampled,
                                 1 << channel_shifts_[i][c].first),
                         DivCeil(frame_dimensions_.ysize_upsampled,
                                 1 << channel_shifts_[i][c].second));
    }
    JXL_RETURN_IF_ERROR(stages_[i]->SetInputSizes(input_sizes));
    if (stages_[i]->SwitchToImageDimensions()) {
      // We don't allow kInOut after switching to image dimensions.
      JXL_ENSURE(i >= first_trailing_stage_);
      first_image_dim_stage_ = i + 1;
      stages_[i]->GetImageDimensions(&full_image_xsize_, &full_image_ysize_,
                                     &frame_origin_);
      break;
    }
  }
  for (size_t i = first_image_dim_stage_; i < stages_.size(); i++) {
    if (stages_[i]->SwitchToImageDimensions()) {
      return JXL_UNREACHABLE(
          "cannot switch to image dimensions multiple times");
    }
    std::vector<std::pair<size_t, size_t>> input_sizes(shifts.size());
    for (size_t c = 0; c < shifts.size(); c++) {
      input_sizes[c] = {full_image_xsize_, full_image_ysize_};
    }
    JXL_RETURN_IF_ERROR(stages_[i]->SetInputSizes(input_sizes));
  }

  anyc_.resize(stages_.size());
  for (size_t i = 0; i < stages_.size(); i++) {
    for (size_t c = 0; c < shifts.size(); c++) {
      if (stages_[i]->GetChannelMode(c) !=
          RenderPipelineChannelMode::kIgnored) {
        anyc_[i] = c;
      }
    }
  }

  stage_input_for_channel_ = std::vector<std::vector<int32_t>>(
      stages_.size(), std::vector<int32_t>(shifts.size()));
  for (size_t c = 0; c < shifts.size(); c++) {
    int input = -1;
    for (size_t i = 0; i < stages_.size(); i++) {
      stage_input_for_channel_[i][c] = input;
      if (stages_[i]->GetChannelMode(c) == RenderPipelineChannelMode::kInOut) {
        input = i;
      }
    }
  }

  image_rect_.resize(stages_.size());
  for (size_t i = 0; i < stages_.size(); i++) {
    size_t x1 = DivCeil(frame_dimensions_.xsize_upsampled,
                        1 << channel_shifts_[i][anyc_[i]].first);
    size_t y1 = DivCeil(frame_dimensions_.ysize_upsampled,
                        1 << channel_shifts_[i][anyc_[i]].second);
    image_rect_[i] = Rect(0, 0, x1, y1);
  }

  virtual_ypadding_for_output_.resize(stages_.size());
  xpadding_for_output_.resize(stages_.size());
  for (size_t c = 0; c < shifts.size(); c++) {
    int ypad = 0;
    int xpad = 0;
    for (size_t i = stages_.size(); i-- > 0;) {
      if (stages_[i]->GetChannelMode(c) !=
          RenderPipelineChannelMode::kIgnored) {
        virtual_ypadding_for_output_[i] =
            std::max(ypad, virtual_ypadding_for_output_[i]);
        xpadding_for_output_[i] = std::max(xpad, xpadding_for_output_[i]);
      }
      if (stages_[i]->GetChannelMode(c) == RenderPipelineChannelMode::kInOut) {
        ypad = (DivCeil(ypad, 1 << channel_shifts_[i][c].second) +
                stages_[i]->settings_.border_y)
               << channel_shifts_[i][c].second;
        xpad = DivCeil(xpad, 1 << stages_[i]->settings_.shift_x) +
               stages_[i]->settings_.border_x;
      }
    }
  }
  return true;
}

Status LowMemoryRenderPipeline::PrepareForThreadsInternal(size_t num,
                                                          bool use_group_ids) {
  const auto& shifts = channel_shifts_[0];
  use_group_ids_ = use_group_ids;
  size_t num_buffers = use_group_ids_ ? frame_dimensions_.num_groups : num;
  for (size_t t = group_data_.size(); t < num_buffers; t++) {
    group_data_.emplace_back();
    group_data_[t].resize(shifts.size());
    for (size_t c = 0; c < shifts.size(); c++) {
      JXL_ASSIGN_OR_RETURN(
          group_data_[t][c],
          ImageF::Create(memory_manager_,
                         GroupInputXSize(c) + group_data_x_border_ * 2,
                         GroupInputYSize(c) + group_data_y_border_ * 2,
                         kRenderPipelineXOffset));
    }
  }
  // TODO(veluca): avoid reallocating buffers if not needed.
  stage_data_.resize(num);
  size_t upsampling = 1u << base_color_shift_;
  size_t group_dim = frame_dimensions_.group_dim * upsampling;
  size_t padding =
      2 * group_data_x_border_ * upsampling +  // maximum size of a rect
      2 * kRenderPipelineXOffset;              // extra padding for processing
  size_t stage_buffer_xsize = group_dim + padding;
  for (size_t t = 0; t < num; t++) {
    stage_data_[t].resize(shifts.size());
    for (size_t c = 0; c < shifts.size(); c++) {
      stage_data_[t][c].resize(stages_.size());
      size_t next_y_border = 0;
      for (size_t i = stages_.size(); i-- > 0;) {
        if (stages_[i]->GetChannelMode(c) ==
            RenderPipelineChannelMode::kInOut) {
          size_t stage_buffer_ysize =
              2 * next_y_border + (1 << stages_[i]->settings_.shift_y);
          stage_buffer_ysize = 1 << CeilLog2Nonzero(stage_buffer_ysize);
          next_y_border = stages_[i]->settings_.border_y;
          JXL_ASSIGN_OR_RETURN(
              stage_data_[t][c][i],
              ImageF::Create(memory_manager_, stage_buffer_xsize,
                             stage_buffer_ysize));
        }
      }
    }
  }
  if (first_image_dim_stage_ != stages_.size()) {
    RectT<ssize_t> image_rect(0, 0, frame_dimensions_.xsize_upsampled,
                              frame_dimensions_.ysize_upsampled);
    RectT<ssize_t> full_image_rect(0, 0, full_image_xsize_, full_image_ysize_);
    image_rect = image_rect.Translate(frame_origin_.x0, frame_origin_.y0);
    image_rect = image_rect.Intersection(full_image_rect);
    if (image_rect.xsize() == 0 || image_rect.ysize() == 0) {
      image_rect = RectT<ssize_t>(0, 0, 0, 0);
    }
    size_t left_padding = image_rect.x0();
    size_t middle_padding = group_dim;
    size_t right_padding = full_image_xsize_ - image_rect.x1();
    size_t out_of_frame_xsize =
        padding +
        std::max(left_padding, std::max(middle_padding, right_padding));
    out_of_frame_data_.resize(num);
    for (size_t t = 0; t < num; t++) {
      JXL_ASSIGN_OR_RETURN(
          out_of_frame_data_[t],
          ImageF::Create(memory_manager_, out_of_frame_xsize, shifts.size()));
    }
  }
  return true;
}

std::vector<std::pair<ImageF*, Rect>> LowMemoryRenderPipeline::PrepareBuffers(
    size_t group_id, size_t thread_id) {
  std::vector<std::pair<ImageF*, Rect>> ret(channel_shifts_[0].size());
  const size_t gx = group_id % frame_dimensions_.xsize_groups;
  const size_t gy = group_id / frame_dimensions_.xsize_groups;
  for (size_t c = 0; c < channel_shifts_[0].size(); c++) {
    ret[c].first = &group_data_[use_group_ids_ ? group_id : thread_id][c];
    ret[c].second = Rect(group_data_x_border_, group_data_y_border_,
                         GroupInputXSize(c), GroupInputYSize(c),
                         DivCeil(frame_dimensions_.xsize_upsampled,
                                 1 << channel_shifts_[0][c].first) -
                             gx * GroupInputXSize(c) + group_data_x_border_,
                         DivCeil(frame_dimensions_.ysize_upsampled,
                                 1 << channel_shifts_[0][c].second) -
                             gy * GroupInputYSize(c) + group_data_y_border_);
  }
  return ret;
}

namespace {

JXL_INLINE int GetMirroredY(int y, ssize_t group_y0, ssize_t image_ysize) {
  if (group_y0 == 0 && (y < 0 || y + group_y0 >= image_ysize)) {
    return Mirror(y, image_ysize);
  }
  if (y + group_y0 >= image_ysize) {
    // Here we know that the one mirroring step is sufficient.
    return 2 * image_ysize - (y + group_y0) - 1 - group_y0;
  }
  return y;
}

JXL_INLINE void ApplyXMirroring(float* row, ssize_t borderx, ssize_t group_x0,
                                ssize_t group_xsize, ssize_t image_xsize) {
  if (image_xsize <= borderx) {
    if (group_x0 == 0) {
      for (ssize_t ix = 0; ix < borderx; ix++) {
        row[kRenderPipelineXOffset - ix - 1] =
            row[kRenderPipelineXOffset + Mirror(-ix - 1, image_xsize)];
      }
    }
    if (group_xsize + borderx + group_x0 >= image_xsize) {
      for (ssize_t ix = 0; ix < borderx; ix++) {
        row[kRenderPipelineXOffset + image_xsize + ix - group_x0] =
            row[kRenderPipelineXOffset + Mirror(image_xsize + ix, image_xsize) -
                group_x0];
      }
    }
  } else {
    // Here we know that the one mirroring step is sufficient.
    if (group_x0 == 0) {
      for (ssize_t ix = 0; ix < borderx; ix++) {
        row[kRenderPipelineXOffset - ix - 1] = row[kRenderPipelineXOffset + ix];
      }
    }
    if (group_xsize + borderx + group_x0 >= image_xsize) {
      for (ssize_t ix = 0; ix < borderx; ix++) {
        row[kRenderPipelineXOffset + image_xsize - group_x0 + ix] =
            row[kRenderPipelineXOffset + image_xsize - group_x0 - ix - 1];
      }
    }
  }
}

// Information about where the *output* of each stage is stored.
class Rows {
 public:
  static StatusOr<Rows> Create(
      const std::vector<std::unique_ptr<RenderPipelineStage>>& stages,
      const Rect data_max_color_channel_rect, int group_data_x_border,
      int group_data_y_border,
      const std::vector<std::pair<size_t, size_t>>& group_data_shift,
      size_t base_color_shift, std::vector<std::vector<ImageF>>& thread_data,
      std::vector<ImageF>& input_data) {
    size_t num_stages = stages.size();
    size_t num_channels = input_data.size();

    JXL_ENSURE(thread_data.size() == num_channels);
    JXL_ENSURE(group_data_shift.size() == num_channels);

    for (const auto& td : thread_data) {
      JXL_ENSURE(td.size() == num_stages);
    }

    std::vector<std::vector<RowInfo>> rows;
    rows.resize(num_stages + 1, std::vector<RowInfo>(num_channels));

    for (size_t i = 0; i < num_stages; i++) {
      for (size_t c = 0; c < input_data.size(); c++) {
        if (stages[i]->GetChannelMode(c) == RenderPipelineChannelMode::kInOut) {
          rows[i + 1][c].ymod_minus_1 = thread_data[c][i].ysize() - 1;
          rows[i + 1][c].base_ptr = thread_data[c][i].Row(0);
          rows[i + 1][c].stride = thread_data[c][i].PixelsPerRow();
        }
      }
    }

    for (size_t c = 0; c < input_data.size(); c++) {
      auto tmp = data_max_color_channel_rect.As<ssize_t>()
                     .Translate(-group_data_x_border, -group_data_y_border)
                     .ShiftLeft(base_color_shift);
      JXL_ASSIGN_OR_RETURN(tmp, tmp.CeilShiftRight(group_data_shift[c]));
      auto channel_group_data_rect = tmp.Translate(
          group_data_x_border - static_cast<ssize_t>(kRenderPipelineXOffset),
          group_data_y_border);
      rows[0][c].base_ptr = channel_group_data_rect.Row(&input_data[c], 0);
      rows[0][c].stride = input_data[c].PixelsPerRow();
      rows[0][c].ymod_minus_1 = -1;
    }
    return Rows(std::move(rows));
  }

  // Stage -1 refers to the input data; all other values must be nonnegative and
  // refer to the data for the output of that stage.
  JXL_INLINE float* GetBuffer(int stage, int y, size_t c) const {
    JXL_DASSERT(stage >= -1);
    const RowInfo& info = rows_[stage + 1][c];
    return info.base_ptr +
           static_cast<ssize_t>(info.stride) * (y & info.ymod_minus_1);
  }

 private:
  struct RowInfo {
    // Pointer to beginning of the first row.
    float* base_ptr;
    // Modulo value for the y axis minus 1 (ymod is guaranteed to be a power of
    // 2, which allows efficient mod computation by masking).
    int ymod_minus_1;
    // Number of floats per row.
    size_t stride;
  };

  explicit Rows(std::vector<std::vector<RowInfo>>&& rows)
      : rows_(std::move(rows)) {}

  std::vector<std::vector<RowInfo>> rows_;
};

}  // namespace

Status LowMemoryRenderPipeline::RenderRect(size_t thread_id,
                                           std::vector<ImageF>& input_data,
                                           Rect data_max_color_channel_rect,
                                           Rect image_max_color_channel_rect) {
  // For each stage, the rect corresponding to the image area currently being
  // processed, in the coordinates of that stage (i.e. with the scaling factor
  // that that stage has).
  std::vector<Rect> group_rect;
  group_rect.resize(stages_.size());
  Rect image_area_rect =
      image_max_color_channel_rect.ShiftLeft(base_color_shift_)
          .Crop(frame_dimensions_.xsize_upsampled,
                frame_dimensions_.ysize_upsampled);
  for (size_t i = 0; i < stages_.size(); i++) {
    JXL_ASSIGN_OR_RETURN(group_rect[i], image_area_rect.CeilShiftRight(
                                            channel_shifts_[i][anyc_[i]]));
  }

  ssize_t frame_x0 = frame_origin_.x0;
  ssize_t frame_y0 = frame_origin_.y0;

  // Compute actual x-axis bounds for the current image area in the context of
  // the full image this frame is part of. As the left boundary may be negative,
  // we also create the x_pixels_skip value, defined as follows:
  // - both x_pixels_skip and full_image_x0 are >= 0, and at least one is 0;
  // - full_image_x0 - x_pixels_skip is the position of the current frame area
  //   in the full image.
  ssize_t full_image_x0 = frame_x0 + image_area_rect.x0();
  ssize_t x_pixels_skip = 0;
  if (full_image_x0 < 0) {
    x_pixels_skip = -full_image_x0;
    full_image_x0 = 0;
  }
  ssize_t full_image_x1 = frame_x0 + image_area_rect.x1();

  std::vector<Rect> span(stages_.size());
  for (size_t i = 0; i < stages_.size(); ++i) {
    if (i < first_image_dim_stage_) {
      span[i] = Rect(group_rect[i].x0(), 0, group_rect[i].xsize(),
                     image_rect_[i].ysize());
    } else {
      size_t x0 = full_image_x0;
      size_t x1 = full_image_x1;
      size_t x_max = full_image_xsize_;
      size_t cropped_x1 = std::min<ssize_t>(x1, x_max);
      span[i] =
          Rect(x0, 0, std::max<ssize_t>(0, cropped_x1 - x0), full_image_ysize_);
    }
  }

  // Data structures to hold information about input/output rows and their
  // buffers.
  JXL_ASSIGN_OR_RETURN(
      Rows rows,
      Rows::Create(stages_, data_max_color_channel_rect, group_data_x_border_,
                   group_data_y_border_, channel_shifts_[0], base_color_shift_,
                   stage_data_[thread_id], input_data));

  std::vector<RenderPipelineStage::RowInfo> input_rows(first_trailing_stage_ +
                                                       1);
  for (size_t i = 0; i < first_trailing_stage_; i++) {
    input_rows[i].resize(input_data.size());
  }
  input_rows[first_trailing_stage_].resize(input_data.size(),
                                           std::vector<float*>(1));

  // Maximum possible shift is 3.
  RenderPipelineStage::RowInfo output_rows(input_data.size(),
                                           std::vector<float*>(8));

  // Fills in input_rows and output_rows for a given y value (relative to the
  // start of the group, measured in actual pixels at the appropriate vertical
  // scaling factor) and a given stage, applying mirroring if necessary. This
  // function is somewhat inefficient for trailing kInOut or kInput stages,
  // where just filling the input row once ought to be sufficient.
  auto prepare_io_rows = [&](int y, size_t i) {
    ssize_t bordery = stages_[i]->settings_.border_y;
    size_t shifty = stages_[i]->settings_.shift_y;
    auto make_row = [&](size_t c, ssize_t iy) {
      size_t mirrored_y = GetMirroredY(y + iy - bordery, group_rect[i].y0(),
                                       image_rect_[i].ysize());
      input_rows[i][c][iy] =
          rows.GetBuffer(stage_input_for_channel_[i][c], mirrored_y, c);
      ApplyXMirroring(input_rows[i][c][iy], stages_[i]->settings_.border_x,
                      group_rect[i].x0(), group_rect[i].xsize(),
                      image_rect_[i].xsize());
    };
    for (size_t c = 0; c < input_data.size(); c++) {
      RenderPipelineChannelMode mode = stages_[i]->GetChannelMode(c);
      if (mode == RenderPipelineChannelMode::kIgnored) {
        continue;
      }
      // If we already have rows from a previous iteration, we can just shift
      // the rows by 1 and insert the new one.
      if (input_rows[i][c].size() == 2 * static_cast<size_t>(bordery) + 1) {
        for (ssize_t iy = 0; iy < 2 * bordery; iy++) {
          input_rows[i][c][iy] = input_rows[i][c][iy + 1];
        }
        make_row(c, bordery * 2);
      } else {
        input_rows[i][c].resize(2 * bordery + 1);
        for (ssize_t iy = 0; iy < 2 * bordery + 1; iy++) {
          make_row(c, iy);
        }
      }

      // If necessary, get the output buffers.
      if (mode == RenderPipelineChannelMode::kInOut) {
        for (size_t iy = 0; iy < (1u << shifty); iy++) {
          output_rows[c][iy] = rows.GetBuffer(i, y * (1 << shifty) + iy, c);
        }
      }
    }
  };

  // We pretend that every stage has a vertical shift of 0, i.e. it is as tall
  // as the final image.
  // We call each such row a "virtual" row, because it may or may not correspond
  // to an actual row of the current processing stage; actual processing happens
  // when vy % (1<<vshift) == 0.

  int num_extra_rows = *std::max_element(virtual_ypadding_for_output_.begin(),
                                         virtual_ypadding_for_output_.end());

  for (int vy = -num_extra_rows;
       vy < static_cast<int>(image_area_rect.ysize()) + num_extra_rows; vy++) {
    for (size_t i = 0; i < first_trailing_stage_; i++) {
      int stage_vy = vy - num_extra_rows + virtual_ypadding_for_output_[i];

      if (stage_vy % (1 << channel_shifts_[i][anyc_[i]].second) != 0) {
        continue;
      }

      if (stage_vy < -virtual_ypadding_for_output_[i]) {
        continue;
      }

      int y = stage_vy >> channel_shifts_[i][anyc_[i]].second;

      ssize_t image_y = static_cast<ssize_t>(group_rect[i].y0()) + y;
      // Do not produce rows in out-of-bounds areas.
      if (image_y < 0) continue;
      if (image_y >= static_cast<ssize_t>(span[i].y1())) continue;

      // Get the input/output rows and potentially apply mirroring to the input.
      prepare_io_rows(y, i);

      // Produce output rows.
      if (span[i].xsize() == 0) continue;
      JXL_RETURN_IF_ERROR(stages_[i]->ProcessRow(
          input_rows[i], output_rows, xpadding_for_output_[i], span[i].xsize(),
          span[i].x0(), image_y, thread_id));
    }

    // Process trailing stages, i.e. the final set of non-kInOut stages; they
    // all have the same input buffer and no need to use any mirroring.

    int y = vy - num_extra_rows;

    for (size_t c = 0; c < input_data.size(); c++) {
      input_rows[first_trailing_stage_][c][0] = rows.GetBuffer(
          stage_input_for_channel_[first_trailing_stage_][c], y, c);
    }

    // Check that we are not outside of the bounds for the current rendering
    // rect. Not doing so might result in overwriting some rows that have been
    // written (or will be written) by other threads.
    if (y < 0 || y >= static_cast<ssize_t>(image_area_rect.ysize())) {
      continue;
    }

    for (size_t i = first_trailing_stage_; i < first_image_dim_stage_; i++) {
      if (span[i].xsize() == 0) continue;
      size_t y0 = image_area_rect.y0() + y;
      if (y0 >= span[i].y1()) continue;
      JXL_RETURN_IF_ERROR(stages_[i]->ProcessRow(
          input_rows[first_trailing_stage_], output_rows,
          /*xextra=*/0, span[i].xsize(), span[i].x0(), y0, thread_id));
    }

    if (first_image_dim_stage_ == stages_.size()) continue;

    // Skip pixels that are not part of the actual final image area.
    for (size_t c = 0; c < input_data.size(); c++) {
      input_rows[first_trailing_stage_][c][0] += x_pixels_skip;
    }
    // Avoid running pipeline stages on pixels that are outside the full image
    // area. As trailing stages have no borders, this is a free optimization
    // (and may be necessary for correctness, as some stages assume coordinates
    // are within bounds).
    ssize_t full_image_y = frame_y0 + image_area_rect.y0() + y;
    if (full_image_y < 0) continue;

    for (size_t i = first_image_dim_stage_; i < stages_.size(); i++) {
      if (span[i].xsize() == 0) continue;
      if (full_image_y >= static_cast<ssize_t>(span[i].y1())) continue;
      JXL_RETURN_IF_ERROR(
          stages_[i]->ProcessRow(input_rows[first_trailing_stage_], output_rows,
                                 /*xextra=*/0, span[i].xsize(), span[i].x0(),
                                 full_image_y, thread_id));
    }
  }
  return true;
}

Status LowMemoryRenderPipeline::RenderPadding(size_t thread_id, Rect rect) {
  if (rect.xsize() == 0) return true;
  size_t numc = channel_shifts_[0].size();
  RenderPipelineStage::RowInfo input_rows(numc, std::vector<float*>(1));
  RenderPipelineStage::RowInfo output_rows;

  for (size_t c = 0; c < numc; c++) {
    input_rows[c][0] = out_of_frame_data_[thread_id].Row(c);
  }

  for (size_t y = 0; y < rect.ysize(); y++) {
    stages_[first_image_dim_stage_ - 1]->ProcessPaddingRow(
        input_rows, rect.xsize(), rect.x0(), rect.y0() + y);
    for (size_t i = first_image_dim_stage_; i < stages_.size(); i++) {
      JXL_RETURN_IF_ERROR(stages_[i]->ProcessRow(
          input_rows, output_rows,
          /*xextra=*/0, rect.xsize(), rect.x0(), rect.y0() + y, thread_id));
    }
  }
  return true;
}

Status LowMemoryRenderPipeline::ProcessBuffers(size_t group_id,
                                               size_t thread_id) {
  std::vector<ImageF>& input_data =
      group_data_[use_group_ids_ ? group_id : thread_id];

  // Copy the group borders to the border storage.
  for (size_t c = 0; c < input_data.size(); c++) {
    JXL_RETURN_IF_ERROR(SaveBorders(group_id, c, input_data[c]));
  }

  size_t gy = group_id / frame_dimensions_.xsize_groups;
  size_t gx = group_id % frame_dimensions_.xsize_groups;

  if (first_image_dim_stage_ != stages_.size()) {
    size_t group_dim = frame_dimensions_.group_dim << base_color_shift_;
    RectT<ssize_t> group_rect(gx * group_dim, gy * group_dim, group_dim,
                              group_dim);
    RectT<ssize_t> image_rect(0, 0, frame_dimensions_.xsize_upsampled,
                              frame_dimensions_.ysize_upsampled);
    RectT<ssize_t> full_image_rect(0, 0, full_image_xsize_, full_image_ysize_);
    group_rect = group_rect.Translate(frame_origin_.x0, frame_origin_.y0);
    image_rect = image_rect.Translate(frame_origin_.x0, frame_origin_.y0);
    image_rect = image_rect.Intersection(full_image_rect);
    group_rect = group_rect.Intersection(image_rect);
    size_t x0 = group_rect.x0();
    size_t y0 = group_rect.y0();
    size_t x1 = group_rect.x1();
    size_t y1 = group_rect.y1();
    JXL_DEBUG_V(6,
                "Rendering padding for full image rect %s "
                "outside group rect %s",
                Description(full_image_rect).c_str(),
                Description(group_rect).c_str());

    if (group_id == 0 && (image_rect.xsize() == 0 || image_rect.ysize() == 0)) {
      // If this frame does not intersect with the full image, we have to
      // initialize the whole image area with RenderPadding.
      JXL_RETURN_IF_ERROR(RenderPadding(
          thread_id, Rect(0, 0, full_image_xsize_, full_image_ysize_)));
    }

    // Render padding for groups that intersect with the full image. The case
    // where no groups intersect was handled above.
    if (group_rect.xsize() > 0 && group_rect.ysize() > 0) {
      if (gx == 0 && gy == 0) {
        JXL_RETURN_IF_ERROR(RenderPadding(thread_id, Rect(0, 0, x0, y0)));
      }
      if (gy == 0) {
        JXL_RETURN_IF_ERROR(RenderPadding(thread_id, Rect(x0, 0, x1 - x0, y0)));
      }
      if (gx == 0) {
        JXL_RETURN_IF_ERROR(RenderPadding(thread_id, Rect(0, y0, x0, y1 - y0)));
      }
      if (gx == 0 && gy + 1 == frame_dimensions_.ysize_groups) {
        JXL_RETURN_IF_ERROR(
            RenderPadding(thread_id, Rect(0, y1, x0, full_image_ysize_ - y1)));
      }
      if (gy + 1 == frame_dimensions_.ysize_groups) {
        JXL_RETURN_IF_ERROR(RenderPadding(
            thread_id, Rect(x0, y1, x1 - x0, full_image_ysize_ - y1)));
      }
      if (gy == 0 && gx + 1 == frame_dimensions_.xsize_groups) {
        JXL_RETURN_IF_ERROR(
            RenderPadding(thread_id, Rect(x1, 0, full_image_xsize_ - x1, y0)));
      }
      if (gx + 1 == frame_dimensions_.xsize_groups) {
        JXL_RETURN_IF_ERROR(RenderPadding(
            thread_id, Rect(x1, y0, full_image_xsize_ - x1, y1 - y0)));
      }
      if (gy + 1 == frame_dimensions_.ysize_groups &&
          gx + 1 == frame_dimensions_.xsize_groups) {
        JXL_RETURN_IF_ERROR(RenderPadding(
            thread_id,
            Rect(x1, y1, full_image_xsize_ - x1, full_image_ysize_ - y1)));
      }
    }
  }

  Rect ready_rects[GroupBorderAssigner::kMaxToFinalize];
  size_t num_ready_rects = 0;
  group_border_assigner_.GroupDone(group_id, group_border_.first,
                                   group_border_.second, ready_rects,
                                   &num_ready_rects);
  for (size_t i = 0; i < num_ready_rects; i++) {
    const Rect& image_max_color_channel_rect = ready_rects[i];
    for (size_t c = 0; c < input_data.size(); c++) {
      JXL_RETURN_IF_ERROR(LoadBorders(group_id, c, image_max_color_channel_rect,
                                      &input_data[c]));
    }
    Rect data_max_color_channel_rect(
        group_data_x_border_ + image_max_color_channel_rect.x0() -
            gx * frame_dimensions_.group_dim,
        group_data_y_border_ + image_max_color_channel_rect.y0() -
            gy * frame_dimensions_.group_dim,
        image_max_color_channel_rect.xsize(),
        image_max_color_channel_rect.ysize());
    JXL_RETURN_IF_ERROR(RenderRect(thread_id, input_data,
                                   data_max_color_channel_rect,
                                   image_max_color_channel_rect));
  }
  return true;
}
}  // namespace jxl
