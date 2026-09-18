// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#if defined(LIB_JXL_CMS_TONE_MAPPING_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef LIB_JXL_CMS_TONE_MAPPING_INL_H_
#undef LIB_JXL_CMS_TONE_MAPPING_INL_H_
#else
#define LIB_JXL_CMS_TONE_MAPPING_INL_H_
#endif

#include <hwy/highway.h>

#include "lib/jxl/cms/tone_mapping.h"
#include "lib/jxl/cms/transfer_functions-inl.h"

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {
namespace {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::Clamp;
using hwy::HWY_NAMESPACE::Max;
using hwy::HWY_NAMESPACE::ZeroIfNegative;

template <typename D>
class Rec2408ToneMapper : Rec2408ToneMapperBase {
 private:
  using V = hwy::HWY_NAMESPACE::Vec<D>;

 public:
  using Rec2408ToneMapperBase::Rec2408ToneMapperBase;

  void ToneMap(V* red, V* green, V* blue) const {
    const V luminance = Mul(Set(df_, source_range_.second),
                            (MulAdd(Set(df_, red_Y_), *red,
                                    MulAdd(Set(df_, green_Y_), *green,
                                           Mul(Set(df_, blue_Y_), *blue)))));
    const V pq_mastering_min = Set(df_, pq_mastering_min_);
    const V inv_pq_mastering_range = Set(df_, inv_pq_mastering_range_);
    const V normalized_pq = Min(
        Set(df_, 1.f),
        Mul(Sub(InvEOTF(luminance), pq_mastering_min), inv_pq_mastering_range));
    const V ks = Set(df_, ks_);
    const V e2 =
        IfThenElse(Lt(normalized_pq, ks), normalized_pq, P(normalized_pq));
    const V one_minus_e2 = Sub(Set(df_, 1), e2);
    const V one_minus_e2_2 = Mul(one_minus_e2, one_minus_e2);
    const V one_minus_e2_4 = Mul(one_minus_e2_2, one_minus_e2_2);
    const V b = Set(df_, min_lum_);
    const V e3 = MulAdd(b, one_minus_e2_4, e2);
    const V pq_mastering_range = Set(df_, pq_mastering_range_);
    const V e4 = MulAdd(e3, pq_mastering_range, pq_mastering_min);
    const V new_luminance =
        Min(Set(df_, target_range_.second),
            ZeroIfNegative(tf_pq_.DisplayFromEncoded(df_, e4)));
    const V min_luminance = Set(df_, 1e-6f);
    const auto use_cap = Le(luminance, min_luminance);
    const V ratio = Div(new_luminance, Max(luminance, min_luminance));
    const V cap = Mul(new_luminance, Set(df_, inv_target_peak_));
    const V normalizer = Set(df_, normalizer_);
    const V multiplier = Mul(ratio, normalizer);
    for (V* const val : {red, green, blue}) {
      *val = IfThenElse(use_cap, cap, Mul(*val, multiplier));
    }
  }

 private:
  V InvEOTF(const V luminance) const {
    return tf_pq_.EncodedFromDisplay(df_, luminance);
  }
  V T(const V a) const {
    const V ks = Set(df_, ks_);
    const V inv_one_minus_ks = Set(df_, inv_one_minus_ks_);
    return Mul(Sub(a, ks), inv_one_minus_ks);
  }
  V P(const V b) const {
    const V t_b = T(b);
    const V t_b_2 = Mul(t_b, t_b);
    const V t_b_3 = Mul(t_b_2, t_b);
    const V ks = Set(df_, ks_);
    const V max_lum = Set(df_, max_lum_);
    return MulAdd(
        MulAdd(Set(df_, 2), t_b_3, MulAdd(Set(df_, -3), t_b_2, Set(df_, 1))),
        ks,
        MulAdd(Add(t_b_3, MulAdd(Set(df_, -2), t_b_2, t_b)),
               Sub(Set(df_, 1), ks),
               Mul(MulAdd(Set(df_, -2), t_b_3, Mul(Set(df_, 3), t_b_2)),
                   max_lum)));
  }

  D df_;
  const TF_PQ tf_pq_ = TF_PQ(/*display_intensity_target=*/1.0);
};

class HlgOOTF : HlgOOTF_Base {
 public:
  using HlgOOTF_Base::HlgOOTF_Base;

  static HlgOOTF FromSceneLight(float display_luminance,
                                const Vector3& primaries_luminances) {
    return HlgOOTF(/*gamma=*/1.2f *
                       std::pow(1.111f, std::log2(display_luminance / 1000.f)),
                   primaries_luminances);
  }

  static HlgOOTF ToSceneLight(float display_luminance,
                              const Vector3& primaries_luminances) {
    return HlgOOTF(
        /*gamma=*/(1 / 1.2f) *
            std::pow(1.111f, -std::log2(display_luminance / 1000.f)),
        primaries_luminances);
  }

  template <typename V>
  void Apply(V* red, V* green, V* blue) const {
    hwy::HWY_NAMESPACE::DFromV<V> df;
    if (!apply_ootf_) return;
    const V luminance =
        MulAdd(Set(df, red_Y_), *red,
               MulAdd(Set(df, green_Y_), *green, Mul(Set(df, blue_Y_), *blue)));
    const V ratio =
        Min(FastPowf(df, luminance, Set(df, exponent_)), Set(df, 1e9));
    *red = Mul(*red, ratio);
    *green = Mul(*green, ratio);
    *blue = Mul(*blue, ratio);
  }

  bool WarrantsGamutMapping() const { return apply_ootf_ && exponent_ < 0; }
};

template <typename V>
void GamutMap(V* red, V* green, V* blue, const Vector3& primaries_luminances,
              float preserve_saturation = 0.1f) {
  hwy::HWY_NAMESPACE::DFromV<V> df;
  const V luminance =
      MulAdd(Set(df, primaries_luminances[0]), *red,
             MulAdd(Set(df, primaries_luminances[1]), *green,
                    Mul(Set(df, primaries_luminances[2]), *blue)));

  // Desaturate out-of-gamut pixels. This is done by mixing each pixel
  // with just enough gray of the target luminance to make all
  // components non-negative.
  // - For saturation preservation, if a component is still larger than
  // 1 then the pixel is normalized to have a maximum component of 1.
  // That will reduce its luminance.
  // - For luminance preservation, getting all components below 1 is
  // done by mixing in yet more gray. That will desaturate it further.
  const V zero = Zero(df);
  const V one = Set(df, 1);
  V gray_mix_saturation = zero;
  V gray_mix_luminance = zero;
  for (const V* ch : {red, green, blue}) {
    const V& val = *ch;
    const V val_minus_gray = Sub(val, luminance);
    const V inv_val_minus_gray =
        Div(one, IfThenElse(Eq(val_minus_gray, zero), one, val_minus_gray));
    const V val_over_val_minus_gray = Mul(val, inv_val_minus_gray);
    gray_mix_saturation =
        IfThenElse(Ge(val_minus_gray, zero), gray_mix_saturation,
                   Max(gray_mix_saturation, val_over_val_minus_gray));
    gray_mix_luminance =
        Max(gray_mix_luminance,
            IfThenElse(Le(val_minus_gray, zero), gray_mix_saturation,
                       Sub(val_over_val_minus_gray, inv_val_minus_gray)));
  }
  const V gray_mix = Clamp(
      MulAdd(Set(df, preserve_saturation),
             Sub(gray_mix_saturation, gray_mix_luminance), gray_mix_luminance),
      zero, one);
  for (V* const ch : {red, green, blue}) {
    V& val = *ch;
    val = MulAdd(gray_mix, Sub(luminance, val), val);
  }
  const V max_clr = Max(Max(one, *red), Max(*green, *blue));
  const V normalizer = Div(one, max_clr);
  for (V* const ch : {red, green, blue}) {
    V& val = *ch;
    val = Mul(val, normalizer);
  }
}

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#endif  // LIB_JXL_CMS_TONE_MAPPING_INL_H_
