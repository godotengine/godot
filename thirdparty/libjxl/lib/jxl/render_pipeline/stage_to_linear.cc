// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/render_pipeline/stage_to_linear.h"

#include "lib/jxl/base/sanitizers.h"

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
  T Transform(D d, const T& encoded) const {
    return encoded;
  }
};

struct OpRgb {
  template <typename D, typename T>
  T Transform(D d, const T& encoded) const {
    return TF_SRGB().DisplayFromEncoded(encoded);
  }
};

struct OpPq {
  explicit OpPq(const float intensity_target) : tf_pq_(intensity_target) {}
  template <typename D, typename T>
  T Transform(D d, const T& encoded) const {
    return tf_pq_.DisplayFromEncoded(d, encoded);
  }
  TF_PQ tf_pq_;
};

struct OpHlg {
  explicit OpHlg(const Vector3& luminances, const float intensity_target)
      : hlg_ootf_(HlgOOTF::FromSceneLight(
            /*display_luminance=*/intensity_target, luminances)) {}

  template <typename D, typename T>
  void Transform(D d, T* r, T* g, T* b) const {
    for (T* val : {r, g, b}) {
      HWY_ALIGN float vals[MaxLanes(d)];
      Store(*val, d, vals);
      for (size_t i = 0; i < Lanes(d); ++i) {
        vals[i] = TF_HLG_Base::DisplayFromEncoded(vals[i]);
      }
      *val = Load(d, vals);
    }
    hlg_ootf_.Apply(r, g, b);
  }
  HlgOOTF hlg_ootf_;
};

struct Op709 {
  template <typename D, typename T>
  T Transform(D d, const T& encoded) const {
    return TF_709().DisplayFromEncoded(d, encoded);
  }
};

struct OpGamma {
  const float gamma;
  template <typename D, typename T>
  T Transform(D d, const T& encoded) const {
    return IfThenZeroElse(Le(encoded, Set(d, 1e-5f)),
                          FastPowf(d, encoded, Set(d, gamma)));
  }
};

struct OpInvalid {
  template <typename D, typename T>
  void Transform(D d, T* r, T* g, T* b) const {}
};

template <typename Op>
class ToLinearStage : public RenderPipelineStage {
 public:
  explicit ToLinearStage(Op op)
      : RenderPipelineStage(RenderPipelineStage::Settings()),
        op_(std::move(op)) {}

  explicit ToLinearStage()
      : RenderPipelineStage(RenderPipelineStage::Settings()), valid_(false) {}

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

  const char* GetName() const override { return "ToLinear"; }

 private:
  Status IsInitialized() const override { return valid_; }

  Op op_;
  bool valid_ = true;
};

template <typename Op>
std::unique_ptr<ToLinearStage<Op>> MakeToLinearStage(Op&& op) {
  return jxl::make_unique<ToLinearStage<Op>>(std::forward<Op>(op));
}

std::unique_ptr<RenderPipelineStage> GetToLinearStage(
    const OutputEncodingInfo& output_encoding_info) {
  const auto& tf = output_encoding_info.color_encoding.Tf();
  if (tf.IsLinear()) {
    return MakeToLinearStage(MakePerChannelOp(OpLinear()));
  } else if (tf.IsSRGB()) {
    return MakeToLinearStage(MakePerChannelOp(OpRgb()));
  } else if (tf.IsPQ()) {
    return MakeToLinearStage(
        MakePerChannelOp(OpPq(output_encoding_info.orig_intensity_target)));
  } else if (tf.IsHLG()) {
    return MakeToLinearStage(OpHlg(output_encoding_info.luminances,
                                   output_encoding_info.orig_intensity_target));
  } else if (tf.Is709()) {
    return MakeToLinearStage(MakePerChannelOp(Op709()));
  } else if (tf.have_gamma || tf.IsDCI()) {
    return MakeToLinearStage(
        MakePerChannelOp(OpGamma{1.f / output_encoding_info.inverse_gamma}));
  } else {
    return jxl::make_unique<ToLinearStage<OpInvalid>>();
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
