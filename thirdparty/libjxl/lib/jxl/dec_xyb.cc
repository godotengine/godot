// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/dec_xyb.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iterator>

#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/image_metadata.h"
#include "lib/jxl/image_ops.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/dec_xyb.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/matrix_ops.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/sanitizers.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/cms/color_encoding_cms.h"
#include "lib/jxl/cms/jxl_cms_internal.h"
#include "lib/jxl/cms/opsin_params.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/common.h"
#include "lib/jxl/dec_xyb-inl.h"
#include "lib/jxl/image.h"
#include "lib/jxl/opsin_params.h"
#include "lib/jxl/quantizer.h"
HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::MulAdd;

// Same, but not in-place.
Status OpsinToLinear(const Image3F& opsin, const Rect& rect, ThreadPool* pool,
                     Image3F* JXL_RESTRICT linear,
                     const OpsinParams& opsin_params) {
  JXL_ENSURE(SameSize(rect, *linear));
  JXL_CHECK_IMAGE_INITIALIZED(opsin, rect);

  const auto process_row = [&](const uint32_t task,
                               size_t /*thread*/) -> Status {
    const size_t y = static_cast<size_t>(task);

    // Faster than adding via ByteOffset at end of loop.
    const float* JXL_RESTRICT row_opsin_0 = rect.ConstPlaneRow(opsin, 0, y);
    const float* JXL_RESTRICT row_opsin_1 = rect.ConstPlaneRow(opsin, 1, y);
    const float* JXL_RESTRICT row_opsin_2 = rect.ConstPlaneRow(opsin, 2, y);
    float* JXL_RESTRICT row_linear_0 = linear->PlaneRow(0, y);
    float* JXL_RESTRICT row_linear_1 = linear->PlaneRow(1, y);
    float* JXL_RESTRICT row_linear_2 = linear->PlaneRow(2, y);

    const HWY_FULL(float) d;

    for (size_t x = 0; x < rect.xsize(); x += Lanes(d)) {
      const auto in_opsin_x = Load(d, row_opsin_0 + x);
      const auto in_opsin_y = Load(d, row_opsin_1 + x);
      const auto in_opsin_b = Load(d, row_opsin_2 + x);
      auto linear_r = Undefined(d);
      auto linear_g = Undefined(d);
      auto linear_b = Undefined(d);
      XybToRgb(d, in_opsin_x, in_opsin_y, in_opsin_b, opsin_params, &linear_r,
               &linear_g, &linear_b);

      Store(linear_r, d, row_linear_0 + x);
      Store(linear_g, d, row_linear_1 + x);
      Store(linear_b, d, row_linear_2 + x);
    }
    return true;
  };
  JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, static_cast<int>(rect.ysize()),
                                ThreadPool::NoInit, process_row,
                                "OpsinToLinear(Rect)"));
  JXL_CHECK_IMAGE_INITIALIZED(*linear, rect);
  return true;
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {

HWY_EXPORT(OpsinToLinear);
Status OpsinToLinear(const Image3F& opsin, const Rect& rect, ThreadPool* pool,
                     Image3F* JXL_RESTRICT linear,
                     const OpsinParams& opsin_params) {
  return HWY_DYNAMIC_DISPATCH(OpsinToLinear)(opsin, rect, pool, linear,
                                             opsin_params);
}

#if !JXL_HIGH_PRECISION
HWY_EXPORT(HasFastXYBTosRGB8);
bool HasFastXYBTosRGB8() { return HWY_DYNAMIC_DISPATCH(HasFastXYBTosRGB8)(); }

HWY_EXPORT(FastXYBTosRGB8);
Status FastXYBTosRGB8(const float* input[4], uint8_t* output, bool is_rgba,
                      size_t xsize) {
  return HWY_DYNAMIC_DISPATCH(FastXYBTosRGB8)(input, output, is_rgba, xsize);
}
#endif  // !JXL_HIGH_PRECISION

void OpsinParams::Init(float intensity_target) {
  InitSIMDInverseMatrix(GetOpsinAbsorbanceInverseMatrix(), inverse_opsin_matrix,
                        intensity_target);
  memcpy(opsin_biases, jxl::cms::kNegOpsinAbsorbanceBiasRGB.data(),
         sizeof(jxl::cms::kNegOpsinAbsorbanceBiasRGB));
  memcpy(quant_biases, kDefaultQuantBias, sizeof(kDefaultQuantBias));
  for (size_t c = 0; c < 4; c++) {
    opsin_biases_cbrt[c] = cbrtf(opsin_biases[c]);
  }
}

bool CanOutputToColorEncoding(const ColorEncoding& c_desired) {
  if (!c_desired.HaveFields()) {
    return false;
  }
  // TODO(veluca): keep in sync with dec_reconstruct.cc
  const auto& tf = c_desired.Tf();
  if (!tf.IsPQ() && !tf.IsSRGB() && !tf.have_gamma && !tf.IsLinear() &&
      !tf.IsHLG() && !tf.IsDCI() && !tf.Is709()) {
    return false;
  }
  if (c_desired.IsGray() && c_desired.GetWhitePointType() != WhitePoint::kD65) {
    // TODO(veluca): figure out what should happen here.
    return false;
  }
  return true;
}

Status OutputEncodingInfo::SetFromMetadata(const CodecMetadata& metadata) {
  orig_color_encoding = metadata.m.color_encoding;
  orig_intensity_target = metadata.m.IntensityTarget();
  desired_intensity_target = orig_intensity_target;
  const auto& im = metadata.transform_data.opsin_inverse_matrix;
  orig_inverse_matrix = im.inverse_matrix;
  default_transform = im.all_default;
  xyb_encoded = metadata.m.xyb_encoded;
  std::copy(std::begin(im.opsin_biases), std::end(im.opsin_biases),
            opsin_params.opsin_biases);
  for (int i = 0; i < 3; ++i) {
    opsin_params.opsin_biases_cbrt[i] = cbrtf(opsin_params.opsin_biases[i]);
  }
  opsin_params.opsin_biases_cbrt[3] = opsin_params.opsin_biases[3] = 1;
  std::copy(std::begin(im.quant_biases), std::end(im.quant_biases),
            opsin_params.quant_biases);
  bool orig_ok = CanOutputToColorEncoding(orig_color_encoding);
  bool orig_grey = orig_color_encoding.IsGray();
  return SetColorEncoding(!xyb_encoded || orig_ok
                              ? orig_color_encoding
                              : ColorEncoding::LinearSRGB(orig_grey));
}

Status OutputEncodingInfo::MaybeSetColorEncoding(
    const ColorEncoding& c_desired) {
  if (c_desired.GetColorSpace() == ColorSpace::kXYB &&
      ((color_encoding.GetColorSpace() == ColorSpace::kRGB &&
        color_encoding.GetPrimariesType() != Primaries::kSRGB) ||
       color_encoding.Tf().IsPQ())) {
    return false;
  }
  if (!xyb_encoded && !CanOutputToColorEncoding(c_desired)) {
    return false;
  }
  return SetColorEncoding(c_desired);
}

Status OutputEncodingInfo::SetColorEncoding(const ColorEncoding& c_desired) {
  color_encoding = c_desired;
  linear_color_encoding = color_encoding;
  linear_color_encoding.Tf().SetTransferFunction(TransferFunction::kLinear);
  color_encoding_is_original = orig_color_encoding.SameColorEncoding(c_desired);

  // Compute the opsin inverse matrix and luminances based on primaries and
  // white point.
  Matrix3x3 inverse_matrix;
  bool inverse_matrix_is_default = default_transform;
  inverse_matrix = orig_inverse_matrix;
  constexpr Vector3 kSRGBLuminances{0.2126, 0.7152, 0.0722};
  luminances = kSRGBLuminances;
  if ((c_desired.GetPrimariesType() != Primaries::kSRGB ||
       c_desired.GetWhitePointType() != WhitePoint::kD65) &&
      !c_desired.IsGray()) {
    Matrix3x3 srgb_to_xyzd50;
    const auto& srgb = ColorEncoding::SRGB(/*is_gray=*/false);
    PrimariesCIExy p;
    JXL_RETURN_IF_ERROR(srgb.GetPrimaries(p));
    CIExy w = srgb.GetWhitePoint();
    JXL_RETURN_IF_ERROR(PrimariesToXYZD50(p.r.x, p.r.y, p.g.x, p.g.y, p.b.x,
                                          p.b.y, w.x, w.y, srgb_to_xyzd50));
    Matrix3x3 original_to_xyz;
    JXL_RETURN_IF_ERROR(c_desired.GetPrimaries(p));
    w = c_desired.GetWhitePoint();
    if (!PrimariesToXYZ(p.r.x, p.r.y, p.g.x, p.g.y, p.b.x, p.b.y, w.x, w.y,
                        original_to_xyz)) {
      return JXL_FAILURE("PrimariesToXYZ failed");
    }
    luminances = original_to_xyz[1];
    if (xyb_encoded) {
      Matrix3x3 adapt_to_d50;
      if (!AdaptToXYZD50(c_desired.GetWhitePoint().x,
                         c_desired.GetWhitePoint().y, adapt_to_d50)) {
        return JXL_FAILURE("AdaptToXYZD50 failed");
      }
      Matrix3x3 xyzd50_to_original;
      Mul3x3Matrix(adapt_to_d50, original_to_xyz, xyzd50_to_original);
      JXL_RETURN_IF_ERROR(Inv3x3Matrix(xyzd50_to_original));
      Matrix3x3 srgb_to_original;
      Mul3x3Matrix(xyzd50_to_original, srgb_to_xyzd50, srgb_to_original);
      Mul3x3Matrix(srgb_to_original, orig_inverse_matrix, inverse_matrix);
      inverse_matrix_is_default = false;
    }
  }

  if (c_desired.IsGray()) {
    Matrix3x3 tmp_inv_matrix = inverse_matrix;
    Matrix3x3 srgb_to_luma{luminances, luminances, luminances};
    Mul3x3Matrix(srgb_to_luma, tmp_inv_matrix, inverse_matrix);
  }

  // The internal XYB color space uses absolute luminance, so we scale back the
  // opsin inverse matrix to relative luminance where 1.0 corresponds to the
  // original intensity target.
  if (xyb_encoded) {
    InitSIMDInverseMatrix(inverse_matrix, opsin_params.inverse_opsin_matrix,
                          orig_intensity_target);
    all_default_opsin = (std::abs(orig_intensity_target - 255.0) <= 0.1f &&
                         inverse_matrix_is_default);
  }

  // Set the inverse gamma based on color space transfer function.
  const auto& tf = c_desired.Tf();
  inverse_gamma = (tf.have_gamma ? tf.GetGamma()
                   : tf.IsDCI()  ? 1.0f / 2.6f
                                 : 1.0);
  return true;
}

}  // namespace jxl
#endif  // HWY_ONCE
