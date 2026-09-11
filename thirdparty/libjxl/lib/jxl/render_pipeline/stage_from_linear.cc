// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/render_pipeline/stage_from_linear.h"

#include "lib/jxl/base/status.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/render_pipeline/stage_from_linear.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "lib/jxl/base/sanitizers.h"
#include "lib/jxl/cms/tone_mapping-inl.h"
#include "lib/jxl/cms/transfer_functions-inl.h"
#include "lib/jxl/common.h"  // JXL_HIGH_PRECISION

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {
namespace {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::IfThenZeroElse;

template <typename Op>
struct PerChannelOp {
  explicit PerChannelOp(Op op) : op(op) {}
  template <typename D, typename T>
  void Transform(D d, T* r, T* g, T* b) const {
    *r = op.Transform(d, *r);
    *g = op.Transform(d, *g);
    *b = op.Transform(d, *b);
  }

  Op op;
};
template <typename Op>
PerChannelOp<Op> MakePerChannelOp(Op&& op) {
  return PerChannelOp<Op>(std::forward<Op>(op));
}

struct OpLinear {
  template <typename D, typename T>
  T Transform(D d, const T& linear) const {
    return linear;
  }
};

struct OpRgb {
  template <typename D, typename T>
  T Transform(D d, const T& linear) const {
#if JXL_HIGH_PRECISION
    return TF_SRGB().EncodedFromDisplay(d, linear);
#else
    return FastLinearToSRGB(d, linear);
#endif
  }
};

struct OpPq {
  explicit OpPq(const float intensity_target) : tf_pq_(intensity_target) {}
  template <typename D, typename T>
  T Transform(D d, const T& linear) const {
    return tf_pq_.EncodedFromDisplay(d, linear);
  }
  TF_PQ tf_pq_;
};

struct OpHlg {
  explicit OpHlg(const Vector3& luminances, const float intensity_target)
      : hlg_ootf_(HlgOOTF::ToSceneLight(/*display_luminance=*/intensity_target,
                                        luminances)) {}

  template <typename D, typename T>
  void Transform(D d, T* r, T* g, T* b) const {
    hlg_ootf_.Apply(r, g, b);
    *r = TF_HLG().EncodedFromDisplay(d, *r);
    *g = TF_HLG().EncodedFromDisplay(d, *g);
    *b = TF_HLG().EncodedFromDisplay(d, *b);
  }
  HlgOOTF hlg_ootf_;
};

struct Op709 {
  template <typename D, typename T>
  T Transform(D d, const T& linear) const {
    return TF_709().EncodedFromDisplay(d, linear);
  }
};

struct OpGamma {
  const float inverse_gamma;
  template <typename D, typename T>
  T Transform(D d, const T& linear) const {
    return IfThenZeroElse(Le(linear, Set(d, 1e-5f)),
                          FastPowf(d, linear, Set(d, inverse_gamma)));
  }
};

template <typename Op>
class FromLinearStage : public RenderPipelineStage {
 public:
  explicit FromLinearStage(Op op)
      : RenderPipelineStage(RenderPipelineStage::Settings()),
        op_(std::move(op)) {}

  Status ProcessRow(const RowInfo& input_rows, const RowInfo& output_rows,
                    size_t xextra, size_t xsize, size_t xpos, size_t ypos,
                    size_t thread_id) const final {
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
      op_.Transform(d, &r, &g, &b);
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

  const char* GetName() const override { return "FromLinear"; }

 private:
  Op op_;
};

template <typename Op>
std::unique_ptr<FromLinearStage<Op>> MakeFromLinearStage(Op&& op) {
  return jxl::make_unique<FromLinearStage<Op>>(std::forward<Op>(op));
}

std::unique_ptr<RenderPipelineStage> GetFromLinearStage(
    const OutputEncodingInfo& output_encoding_info) {
  const auto& tf = output_encoding_info.color_encoding.Tf();
  if (tf.IsLinear()) {
    return MakeFromLinearStage(MakePerChannelOp(OpLinear()));
  } else if (tf.IsSRGB()) {
    return MakeFromLinearStage(MakePerChannelOp(OpRgb()));
  } else if (tf.IsPQ()) {
    return MakeFromLinearStage(
        MakePerChannelOp(OpPq(output_encoding_info.orig_intensity_target)));
  } else if (tf.IsHLG()) {
    return MakeFromLinearStage(
        OpHlg(output_encoding_info.luminances,
              output_encoding_info.desired_intensity_target));
  } else if (tf.Is709()) {
    return MakeFromLinearStage(MakePerChannelOp(Op709()));
  } else if (tf.have_gamma || tf.IsDCI()) {
    return MakeFromLinearStage(
        MakePerChannelOp(OpGamma{output_encoding_info.inverse_gamma}));
  } else {
    // This is a programming error.
    JXL_DEBUG_ABORT("Invalid target encoding");
    return nullptr;
  }
}

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {

HWY_EXPORT(GetFromLinearStage);

std::unique_ptr<RenderPipelineStage> GetFromLinearStage(
    const OutputEncodingInfo& output_encoding_info) {
  return HWY_DYNAMIC_DISPATCH(GetFromLinearStage)(output_encoding_info);
}

}  // namespace jxl
#endif
