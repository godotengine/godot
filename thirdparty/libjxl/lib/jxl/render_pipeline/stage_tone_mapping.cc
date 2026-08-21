// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/render_pipeline/stage_tone_mapping.h"

#include "lib/jxl/base/sanitizers.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/render_pipeline/stage_tone_mapping.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "lib/jxl/cms/tone_mapping-inl.h"
#include "lib/jxl/dec_xyb-inl.h"

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

class ToneMappingStage : public RenderPipelineStage {
 public:
  explicit ToneMappingStage(OutputEncodingInfo output_encoding_info)
      : RenderPipelineStage(RenderPipelineStage::Settings()),
        output_encoding_info_(std::move(output_encoding_info)) {
    if (output_encoding_info_.desired_intensity_target ==
        output_encoding_info_.orig_intensity_target) {
      // No tone mapping requested.
      return;
    }
    const auto& orig_tf = output_encoding_info_.orig_color_encoding.Tf();
    const auto& dest_tf = output_encoding_info_.color_encoding.Tf();
    if (orig_tf.IsPQ() && output_encoding_info_.desired_intensity_target <
                              output_encoding_info_.orig_intensity_target) {
      tone_mapper_ = jxl::make_unique<ToneMapper>(
          /*source_range=*/std::pair<float, float>(
              0.0f, output_encoding_info_.orig_intensity_target),
          /*target_range=*/
          std::pair<float, float>(
              0.0f, output_encoding_info_.desired_intensity_target),
          output_encoding_info_.luminances);
    } else if (orig_tf.IsHLG() && !dest_tf.IsHLG()) {
      hlg_ootf_ = jxl::make_unique<HlgOOTF>(
          /*source_luminance=*/output_encoding_info_.orig_intensity_target,
          /*target_luminance=*/output_encoding_info_.desired_intensity_target,
          output_encoding_info_.luminances);
    }

    if (dest_tf.IsPQ() && (tone_mapper_ || hlg_ootf_)) {
      to_intensity_target_ =
          10000.f / output_encoding_info_.orig_intensity_target;
      from_desired_intensity_target_ =
          output_encoding_info_.desired_intensity_target / 10000.f;
    }
  }

  bool IsNeeded() const { return tone_mapper_ || hlg_ootf_; }

  Status ProcessRow(const RowInfo& input_rows, const RowInfo& output_rows,
                    size_t xextra, size_t xsize, size_t xpos, size_t ypos,
                    size_t thread_id) const final {
    if (!tone_mapper_ && !hlg_ootf_) return true;
    // I.e. we have tone_mapper_ or hlg_ootf_ or both.
    // Other inveriants: !tone_mapper_ means !!hlg_ootf_ and vice-versa.

    const HWY_FULL(float) d;
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
    for (ssize_t x = -xextra; x < static_cast<ssize_t>(xsize + xextra);
         x += Lanes(d)) {
      auto r = LoadU(d, row0 + x);
      auto g = LoadU(d, row1 + x);
      auto b = LoadU(d, row2 + x);
      r = Mul(r, Set(d, to_intensity_target_));
      g = Mul(g, Set(d, to_intensity_target_));
      b = Mul(b, Set(d, to_intensity_target_));
      if (tone_mapper_) {
        tone_mapper_->ToneMap(&r, &g, &b);
      } else {
        hlg_ootf_->Apply(&r, &g, &b);
      }
      if (tone_mapper_ || hlg_ootf_->WarrantsGamutMapping()) {
        GamutMap(&r, &g, &b, output_encoding_info_.luminances);
      }
      r = Mul(r, Set(d, from_desired_intensity_target_));
      g = Mul(g, Set(d, from_desired_intensity_target_));
      b = Mul(b, Set(d, from_desired_intensity_target_));
      StoreU(r, d, row0 + x);
      StoreU(g, d, row1 + x);
      StoreU(b, d, row2 + x);
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

  const char* GetName() const override { return "ToneMapping"; }

 private:
  using ToneMapper = Rec2408ToneMapper<HWY_FULL(float)>;
  OutputEncodingInfo output_encoding_info_;
  std::unique_ptr<ToneMapper> tone_mapper_;
  std::unique_ptr<HlgOOTF> hlg_ootf_;
  // When the target colorspace is PQ, 1 represents 10000 nits instead of
  // orig_intensity_target. This temporarily changes this if the tone mappers
  // require it.
  float to_intensity_target_ = 1.f;
  float from_desired_intensity_target_ = 1.f;
};

std::unique_ptr<RenderPipelineStage> GetToneMappingStage(
    const OutputEncodingInfo& output_encoding_info) {
  auto stage = jxl::make_unique<ToneMappingStage>(output_encoding_info);
  if (!stage->IsNeeded()) return nullptr;
  return stage;
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {

HWY_EXPORT(GetToneMappingStage);

std::unique_ptr<RenderPipelineStage> GetToneMappingStage(
    const OutputEncodingInfo& output_encoding_info) {
  return HWY_DYNAMIC_DISPATCH(GetToneMappingStage)(output_encoding_info);
}

}  // namespace jxl
#endif
