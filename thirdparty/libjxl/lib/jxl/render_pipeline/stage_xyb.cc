// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/render_pipeline/stage_xyb.h"

#include "lib/jxl/base/common.h"
#include "lib/jxl/base/sanitizers.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/render_pipeline/stage_xyb.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "lib/jxl/cms/opsin_params.h"
#include "lib/jxl/common.h"  // JXL_HIGH_PRECISION
#include "lib/jxl/dec_xyb-inl.h"

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

class XYBStage : public RenderPipelineStage {
 public:
  explicit XYBStage(const OutputEncodingInfo& output_encoding_info)
      : RenderPipelineStage(RenderPipelineStage::Settings()),
        opsin_params_(output_encoding_info.opsin_params),
        output_is_xyb_(output_encoding_info.color_encoding.GetColorSpace() ==
                       ColorSpace::kXYB) {}

  Status ProcessRow(const RowInfo& input_rows, const RowInfo& output_rows,
                    size_t xextra, size_t xsize, size_t xpos, size_t ypos,
                    size_t thread_id) const final {
    const HWY_FULL(float) d;
    JXL_ENSURE(xextra == 0);
    const size_t xsize_v = RoundUpTo(xsize, Lanes(d));
    float* JXL_RESTRICT row0 = GetInputRow(input_rows, 0, 0);
    float* JXL_RESTRICT row1 = GetInputRow(input_rows, 1, 0);
    float* JXL_RESTRICT row2 = GetInputRow(input_rows, 2, 0);
    // All calculations are lane-wise, still some might require
    // value-dependent behaviour (e.g. NearestInt). Temporary unpoison last
    // vector tail.
    msan::UnpoisonMemory(row0 + xsize, sizeof(float) * (xsize_v - xsize));
    msan::UnpoisonMemory(row1 + xsize, sizeof(float) * (xsize_v - xsize));
    msan::UnpoisonMemory(row2 + xsize, sizeof(float) * (xsize_v - xsize));
    // TODO(eustas): when using frame origin, addresses might be unaligned;
    //               making them aligned will void performance penalty.
    if (output_is_xyb_) {
      const auto scale_x = Set(d, jxl::cms::kScaledXYBScale[0]);
      const auto scale_y = Set(d, jxl::cms::kScaledXYBScale[1]);
      const auto scale_bmy = Set(d, jxl::cms::kScaledXYBScale[2]);
      const auto offset_x = Set(d, jxl::cms::kScaledXYBOffset[0]);
      const auto offset_y = Set(d, jxl::cms::kScaledXYBOffset[1]);
      const auto offset_bmy = Set(d, jxl::cms::kScaledXYBOffset[2]);
      for (ssize_t x = -xextra; x < static_cast<ssize_t>(xsize + xextra);
           x += Lanes(d)) {
        const auto in_x = LoadU(d, row0 + x);
        const auto in_y = LoadU(d, row1 + x);
        const auto in_b = LoadU(d, row2 + x);
        auto out_x = Mul(Add(in_x, offset_x), scale_x);
        auto out_y = Mul(Add(in_y, offset_y), scale_y);
        auto out_b = Mul(Add(Sub(in_b, in_y), offset_bmy), scale_bmy);
        StoreU(out_x, d, row0 + x);
        StoreU(out_y, d, row1 + x);
        StoreU(out_b, d, row2 + x);
      }
    } else {
      for (ssize_t x = -xextra; x < static_cast<ssize_t>(xsize + xextra);
           x += Lanes(d)) {
        const auto in_opsin_x = LoadU(d, row0 + x);
        const auto in_opsin_y = LoadU(d, row1 + x);
        const auto in_opsin_b = LoadU(d, row2 + x);
        auto r = Undefined(d);
        auto g = Undefined(d);
        auto b = Undefined(d);
        XybToRgb(d, in_opsin_x, in_opsin_y, in_opsin_b, opsin_params_, &r, &g,
                 &b);
        StoreU(r, d, row0 + x);
        StoreU(g, d, row1 + x);
        StoreU(b, d, row2 + x);
      }
    }
    msan::PoisonMemory(row0 + xsize, sizeof(float) * (xsize_v - xsize));
    msan::PoisonMemory(row1 + xsize, sizeof(float) * (xsize_v - xsize));
    msan::PoisonMemory(row2 + xsize, sizeof(float) * (xsize_v - xsize));
    return true;
  }

  RenderPipelineChannelMode GetChannelMode(size_t c) const final {
    return c < 3 ? RenderPipelineChannelMode::kInPlace
                 : RenderPipelineChannelMode::kIgnored;
  }

  const char* GetName() const override { return "XYB"; }

 private:
  const OpsinParams opsin_params_;
  const bool output_is_xyb_;
};

std::unique_ptr<RenderPipelineStage> GetXYBStage(
    const OutputEncodingInfo& output_encoding_info) {
  return jxl::make_unique<XYBStage>(output_encoding_info);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {

HWY_EXPORT(GetXYBStage);

std::unique_ptr<RenderPipelineStage> GetXYBStage(
    const OutputEncodingInfo& output_encoding_info) {
  return HWY_DYNAMIC_DISPATCH(GetXYBStage)(output_encoding_info);
}

#if !JXL_HIGH_PRECISION
namespace {
class FastXYBStage : public RenderPipelineStage {
 public:
  FastXYBStage(uint8_t* rgb, size_t stride, size_t width, size_t height,
               bool rgba, bool has_alpha, size_t alpha_c)
      : RenderPipelineStage(RenderPipelineStage::Settings()),
        rgb_(rgb),
        stride_(stride),
        width_(width),
        height_(height),
        rgba_(rgba),
        has_alpha_(has_alpha),
        alpha_c_(alpha_c) {}

  Status ProcessRow(const RowInfo& input_rows, const RowInfo& output_rows,
                    size_t xextra, size_t xsize, size_t xpos, size_t ypos,
                    size_t thread_id) const final {
    if (ypos >= height_) return true;
    JXL_ENSURE(xextra == 0);
    const float* xyba[4] = {
        GetInputRow(input_rows, 0, 0), GetInputRow(input_rows, 1, 0),
        GetInputRow(input_rows, 2, 0),
        has_alpha_ ? GetInputRow(input_rows, alpha_c_, 0) : nullptr};
    uint8_t* out_buf = rgb_ + stride_ * ypos + (rgba_ ? 4 : 3) * xpos;
    return FastXYBTosRGB8(xyba, out_buf, rgba_,
                          xsize + xpos <= width_ ? xsize : width_ - xpos);
  }

  RenderPipelineChannelMode GetChannelMode(size_t c) const final {
    return c < 3 || (has_alpha_ && c == alpha_c_)
               ? RenderPipelineChannelMode::kInput
               : RenderPipelineChannelMode::kIgnored;
  }

  const char* GetName() const override { return "FastXYB"; }

 private:
  uint8_t* rgb_;
  size_t stride_;
  size_t width_;
  size_t height_;
  bool rgba_;
  bool has_alpha_;
  size_t alpha_c_;
  std::vector<float> opaque_alpha_;
};

}  // namespace

std::unique_ptr<RenderPipelineStage> GetFastXYBTosRGB8Stage(
    uint8_t* rgb, size_t stride, size_t width, size_t height, bool rgba,
    bool has_alpha, size_t alpha_c) {
  if (!HasFastXYBTosRGB8()) return nullptr;
  return make_unique<FastXYBStage>(rgb, stride, width, height, rgba, has_alpha,
                                   alpha_c);
}
#endif

}  // namespace jxl
#endif
