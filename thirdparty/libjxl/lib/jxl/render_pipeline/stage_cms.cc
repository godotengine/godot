// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/render_pipeline/stage_cms.h"

#include <memory>

#include "lib/jxl/base/status.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/dec_xyb.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/render_pipeline/stage_cms.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "lib/jxl/dec_xyb-inl.h"

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

class CmsStage : public RenderPipelineStage {
 public:
  explicit CmsStage(OutputEncodingInfo output_encoding_info)
      : RenderPipelineStage(RenderPipelineStage::Settings()),
        output_encoding_info_(std::move(output_encoding_info)) {
    c_src_ = output_encoding_info_.linear_color_encoding;
  }

  bool IsNeeded() const {
    const size_t channels_src = (c_src_.IsCMYK() ? 4 : c_src_.Channels());
    const size_t channels_dst = output_encoding_info_.color_encoding.Channels();
    const bool not_mixing_color_and_grey =
        (channels_src == channels_dst ||
         (channels_src == 4 && channels_dst == 3));
    return (output_encoding_info_.cms_set) &&
           !c_src_.SameColorEncoding(output_encoding_info_.color_encoding) &&
           not_mixing_color_and_grey;
  }

  Status ProcessRow(const RowInfo& input_rows, const RowInfo& output_rows,
                    size_t xextra, size_t xsize, size_t xpos, size_t ypos,
                    size_t thread_id) const final {
    JXL_ENSURE(xsize <= xsize_);
    bool gray_src = (c_src_.Channels() == 1);
    bool gray_dst = (output_encoding_info_.color_encoding.Channels() == 1);
    float* mutable_buf_src = color_space_transform->BufSrc(thread_id);
    float* JXL_RESTRICT buf_dst = color_space_transform->BufDst(thread_id);
    //  interleave
    if (gray_src) {
      float* JXL_RESTRICT row0 = GetInputRow(input_rows, 0, 0);
      memcpy(mutable_buf_src, row0, xsize * sizeof(float));
    } else {
      float* JXL_RESTRICT row0 = GetInputRow(input_rows, 0, 0);
      float* JXL_RESTRICT row1 = GetInputRow(input_rows, 1, 0);
      float* JXL_RESTRICT row2 = GetInputRow(input_rows, 2, 0);

      for (size_t x = 0; x < xsize; x++) {
        mutable_buf_src[3 * x + 0] = row0[x];
        mutable_buf_src[3 * x + 1] = row1[x];
        mutable_buf_src[3 * x + 2] = row2[x];
      }
    }
    const float* buf_src = mutable_buf_src;
    JXL_RETURN_IF_ERROR(
        color_space_transform->Run(thread_id, buf_src, buf_dst, xsize));
    // de-interleave
    if (gray_dst) {
      float* JXL_RESTRICT row0 = GetInputRow(input_rows, 0, 0);
      float* JXL_RESTRICT row1 = GetInputRow(input_rows, 1, 0);
      float* JXL_RESTRICT row2 = GetInputRow(input_rows, 2, 0);
      memcpy(row0, buf_dst, xsize * sizeof(float));
      memcpy(row1, buf_dst, xsize * sizeof(float));
      memcpy(row2, buf_dst, xsize * sizeof(float));
    } else {
      float* JXL_RESTRICT row0 = GetInputRow(input_rows, 0, 0);
      float* JXL_RESTRICT row1 = GetInputRow(input_rows, 1, 0);
      float* JXL_RESTRICT row2 = GetInputRow(input_rows, 2, 0);
      for (size_t x = 0; x < xsize; x++) {
        row0[x] = buf_dst[3 * x + 0];
        row1[x] = buf_dst[3 * x + 1];
        row2[x] = buf_dst[3 * x + 2];
      }
    }
    return true;
  }
  RenderPipelineChannelMode GetChannelMode(size_t c) const final {
    return c < 3 ? RenderPipelineChannelMode::kInPlace
                 : RenderPipelineChannelMode::kIgnored;
  }

  const char* GetName() const override { return "Cms"; }

 private:
  OutputEncodingInfo output_encoding_info_;
  size_t xsize_;
  std::unique_ptr<jxl::ColorSpaceTransform> color_space_transform;
  ColorEncoding c_src_;

  Status SetInputSizes(
      const std::vector<std::pair<size_t, size_t>>& input_sizes) override {
    JXL_ENSURE(input_sizes.size() >= 3);
    for (size_t c = 1; c < input_sizes.size(); c++) {
      JXL_ENSURE(input_sizes[c].first == input_sizes[0].first);
      JXL_ENSURE(input_sizes[c].second == input_sizes[0].second);
    }
    xsize_ = input_sizes[0].first;
    return true;
  }

  Status PrepareForThreads(size_t num_threads) override {
    color_space_transform = jxl::make_unique<jxl::ColorSpaceTransform>(
        output_encoding_info_.color_management_system);
    JXL_RETURN_IF_ERROR(color_space_transform->Init(
        c_src_, output_encoding_info_.color_encoding,
        output_encoding_info_.desired_intensity_target, xsize_, num_threads));
    return true;
  }
};

std::unique_ptr<RenderPipelineStage> GetCmsStage(
    const OutputEncodingInfo& output_encoding_info) {
  auto stage = jxl::make_unique<CmsStage>(output_encoding_info);
  if (!stage->IsNeeded()) return nullptr;
  return stage;
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {

HWY_EXPORT(GetCmsStage);

std::unique_ptr<RenderPipelineStage> GetCmsStage(
    const OutputEncodingInfo& output_encoding_info) {
  return HWY_DYNAMIC_DISPATCH(GetCmsStage)(output_encoding_info);
}

}  // namespace jxl
#endif
