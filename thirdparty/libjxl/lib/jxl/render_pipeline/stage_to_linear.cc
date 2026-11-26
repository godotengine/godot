// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/render_pipeline/stage_to_linear.h"

#include <cstddef>
#include <cstdio>
#include <memory>

#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/sanitizers.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_xyb.h"
#include "lib/jxl/render_pipeline/render_pipeline_stage.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/render_pipeline/stage_to_linear.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "lib/jxl/cms/tone_mapping-inl.h"
#include "lib/jxl/cms/transfer_functions-inl.h"

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {
namespace {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::IfThenZeroElse;

struct OpLinear {
  explicit OpLinear(const OutputEncodingInfo& output_encoding_info) {}
  template <typename D, typename T>
  void Transform(D d, T* r, T* g, T* b) const {}
};

struct OpRgb {
  explicit OpRgb(const OutputEncodingInfo& output_encoding_info) {}
  template <typename D, typename T>
  void Transform(D d, T* r, T* g, T* b) const {
    for (T* val : {r, g, b}) {
      *val = TF_SRGB().DisplayFromEncoded(*val);
    }
  }
};

struct OpPq {
  explicit OpPq(const OutputEncodingInfo& output_encoding_info)
      : tf_pq_(output_encoding_info.orig_intensity_target) {}
  template <typename D, typename T>
  void Transform(D d, T* r, T* g, T* b) const {
    for (T* val : {r, g, b}) {
      *val = tf_pq_.DisplayFromEncoded(d, *val);
    }
  }
  TF_PQ tf_pq_;
};

struct OpHlg {
  explicit OpHlg(const OutputEncodingInfo& output_encoding_info)
      : hlg_ootf_(
            HlgOOTF::FromSceneLight(output_encoding_info.orig_intensity_target,
                                    output_encoding_info.luminances)) {}

  template <typename D, typename T>
  void Transform(D d, T* r, T* g, T* b) const {
    for (T* val : {r, g, b}) {
      *val = TF_HLG().DisplayFromEncoded(d, *val);
    }
    hlg_ootf_.Apply(r, g, b);
  }
  HlgOOTF hlg_ootf_;
};

struct Op709 {
  explicit Op709(const OutputEncodingInfo& output_encoding_info) {}
  template <typename D, typename T>
  void Transform(D d, T* r, T* g, T* b) const {
    for (T* val : {r, g, b}) {
      *val = TF_709().DisplayFromEncoded(d, *val);
    }
  }
};

struct OpGamma {
  explicit OpGamma(const OutputEncodingInfo& output_encoding_info)
      : gamma(1.f / output_encoding_info.inverse_gamma) {}
  const float gamma;
  template <typename D, typename T>
  void Transform(D d, T* r, T* g, T* b) const {
    for (T* val : {r, g, b}) {
      *val = IfThenZeroElse(Le(*val, Set(d, 1e-5f)),
                            FastPowf(d, *val, Set(d, gamma)));
    }
  }
};

struct OpInvalid {
  explicit OpInvalid(const OutputEncodingInfo& output_encoding_info) {}
  template <typename D, typename T>
  void Transform(D d, T* r, T* g, T* b) const {}
};

template <typename Op>
class ToLinearStage : public RenderPipelineStage {
 public:
  explicit ToLinearStage(const OutputEncodingInfo& output_encoding_info)
      : RenderPipelineStage(RenderPipelineStage::Settings()),
        op_(output_encoding_info) {}

  explicit ToLinearStage()
      : RenderPipelineStage(RenderPipelineStage::Settings()), valid_(false) {}

  Status ProcessRow(const RowInfo& input_rows, const RowInfo& output_rows,
                    size_t xextra_left, size_t xextra_right, size_t xsize,
                    size_t xpos, size_t ypos, size_t thread_id) const final {
    const HWY_FULL(float) d;
    JXL_ENSURE(xextra_left == 0 && xextra_right == 0);
    size_t x_span_tail = RoundUpTo(xsize, Lanes(d)) - xsize;
    float* JXL_RESTRICT row0 = GetInputRow(input_rows, 0, 0);
    float* JXL_RESTRICT row1 = GetInputRow(input_rows, 1, 0);
    float* JXL_RESTRICT row2 = GetInputRow(input_rows, 2, 0);
    // All calculations are lane-wise, still some might require
    // value-dependent behaviour (e.g. NearestInt). Temporary unpoison last
    // vector tail.
    msan::UnpoisonMemory(row0 + xsize, sizeof(float) * x_span_tail);
    msan::UnpoisonMemory(row1 + xsize, sizeof(float) * x_span_tail);
    msan::UnpoisonMemory(row2 + xsize, sizeof(float) * x_span_tail);
    // TODO(eustas): why unaligned load/store?
    for (size_t x = 0; x < xsize; x += Lanes(d)) {
      auto r = LoadU(d, row0 + x);
      auto g = LoadU(d, row1 + x);
      auto b = LoadU(d, row2 + x);
      op_.Transform(d, &r, &g, &b);
      StoreU(r, d, row0 + x);
      StoreU(g, d, row1 + x);
      StoreU(b, d, row2 + x);
    }
    msan::PoisonMemory(row0 + xsize, sizeof(float) * x_span_tail);
    msan::PoisonMemory(row1 + xsize, sizeof(float) * x_span_tail);
    msan::PoisonMemory(row2 + xsize, sizeof(float) * x_span_tail);
    return true;
  }

  RenderPipelineChannelMode GetChannelMode(size_t c) const final {
    return c < 3 ? RenderPipelineChannelMode::kInPlace
                 : RenderPipelineChannelMode::kIgnored;
  }

  const char* GetName() const override { return "ToLinear"; }

 private:
  Status IsInitialized() const override { return valid_; }

  Op op_;
  bool valid_ = true;
};

template <typename Op>
std::unique_ptr<ToLinearStage<Op>> MakeToLinearStage(
    const OutputEncodingInfo& output_encoding_info) {
  return jxl::make_unique<ToLinearStage<Op>>(output_encoding_info);
}

std::unique_ptr<RenderPipelineStage> GetToLinearStage(
    const OutputEncodingInfo& output_encoding_info) {
  const auto& tf = output_encoding_info.color_encoding.Tf();
  if (tf.IsLinear()) {
    return MakeToLinearStage<OpLinear>(output_encoding_info);
  } else if (tf.IsSRGB()) {
    return MakeToLinearStage<OpRgb>(output_encoding_info);
  } else if (tf.IsPQ()) {
    return MakeToLinearStage<OpPq>(output_encoding_info);
  } else if (tf.IsHLG()) {
    return MakeToLinearStage<OpHlg>(output_encoding_info);
  } else if (tf.Is709()) {
    return MakeToLinearStage<Op709>(output_encoding_info);
  } else if (tf.have_gamma || tf.IsDCI()) {
    return MakeToLinearStage<OpGamma>(output_encoding_info);
  } else {
    return MakeToLinearStage<OpInvalid>(output_encoding_info);
  }
}

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {

HWY_EXPORT(GetToLinearStage);

std::unique_ptr<RenderPipelineStage> GetToLinearStage(
    const OutputEncodingInfo& output_encoding_info) {
  return HWY_DYNAMIC_DISPATCH(GetToLinearStage)(output_encoding_info);
}

}  // namespace jxl
#endif
