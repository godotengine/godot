// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_QUANTIZER_H_
#define LIB_JXL_QUANTIZER_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/field_encodings.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/image.h"
#include "lib/jxl/quant_weights.h"

// Quantizes DC and AC coefficients, with separate quantization tables according
// to the quant_kind (which is currently computed from the AC strategy and the
// block index inside that strategy).

namespace jxl {

enum class AcStrategyType : uint32_t;

static constexpr int kGlobalScaleDenom = 1 << 16;
static constexpr int kGlobalScaleNumerator = 4096;

// zero-biases for quantizing channels X, Y, B
static constexpr float kZeroBiasDefault[3] = {0.5f, 0.5f, 0.5f};

// Returns adjusted version of a quantized integer, such that its value is
// closer to the expected value of the original.
// The residuals of AC coefficients that we quantize are not uniformly
// distributed. Numerical experiments show that they have a distribution with
// the "shape" of 1/(1+x^2) [up to some coefficients]. This means that the
// expected value of a coefficient that gets quantized to x will not be x
// itself, but (at least with reasonable approximation):
// - 0 if x is 0
// - x * biases[c] if x is 1 or -1
// - x - biases[3]/x otherwise
// This follows from computing the distribution of the quantization bias, which
// can be approximated fairly well by <constant>/x when |x| is at least two.
static constexpr float kBiasNumerator = 0.145f;

static constexpr float kDefaultQuantBias[4] = {
    1.0f - 0.05465007330715401f,
    1.0f - 0.07005449891748593f,
    1.0f - 0.049935103337343655f,
    0.145f,
};

struct QuantizerParams;

class Quantizer {
 public:
  explicit Quantizer(const DequantMatrices& dequant);
  Quantizer(const DequantMatrices& dequant, int quant_dc, int global_scale);

  static constexpr int32_t kQuantMax = 256;

  static JXL_INLINE int32_t ClampVal(float val) {
    return static_cast<int32_t>(
        std::max(1.0f, std::min<float>(val, kQuantMax)));
  }

  float ScaleGlobalScale(const float scale) {
    int new_global_scale = static_cast<int>(std::lround(global_scale_ * scale));
    float scale_out = new_global_scale * 1.0f / global_scale_;
    global_scale_ = new_global_scale;
    RecomputeFromGlobalScale();
    return scale_out;
  }

  // Recomputes other derived fields after global_scale_ has changed.
  void RecomputeFromGlobalScale() {
    global_scale_float_ = global_scale_ * (1.0 / kGlobalScaleDenom);
    inv_global_scale_ = 1.0 * kGlobalScaleDenom / global_scale_;
    inv_quant_dc_ = inv_global_scale_ / quant_dc_;
    for (size_t c = 0; c < 3; c++) {
      mul_dc_[c] = GetDcStep(c);
      inv_mul_dc_[c] = GetInvDcStep(c);
    }
  }

  // Returns scaling factor such that Scale() * (RawDC() or RawQuantField())
  // pixels yields the same float values returned by GetQuantField.
  JXL_INLINE float Scale() const { return global_scale_float_; }

  // Reciprocal of Scale().
  JXL_INLINE float InvGlobalScale() const { return inv_global_scale_; }

  void SetQuantFieldRect(const ImageF& qf, const Rect& rect,
                         ImageI* JXL_RESTRICT raw_quant_field) const;

  Status SetQuantField(float quant_dc, const ImageF& qf,
                       ImageI* JXL_RESTRICT raw_quant_field);

  void SetQuant(float quant_dc, float quant_ac,
                ImageI* JXL_RESTRICT raw_quant_field);

  // Returns the DC quantization base value, which is currently global (not
  // adaptive). The actual scale factor used to dequantize pixels in channel c
  // is: inv_quant_dc() * dequant_->DCQuant(c).
  float inv_quant_dc() const { return inv_quant_dc_; }

  // Dequantize by multiplying with this times dequant_matrix.
  float inv_quant_ac(int32_t quant) const { return inv_global_scale_ / quant; }

  QuantizerParams GetParams() const;

  Status Decode(BitReader* reader);

  void DumpQuantizationMap(const ImageI& raw_quant_field) const;

  JXL_INLINE const float* DequantMatrix(AcStrategyType quant_kind,
                                        size_t c) const {
    return dequant_->Matrix(quant_kind, c);
  }

  JXL_INLINE const float* InvDequantMatrix(AcStrategyType quant_kind,
                                           size_t c) const {
    return dequant_->InvMatrix(quant_kind, c);
  }

  // Calculates DC quantization step.
  JXL_INLINE float GetDcStep(size_t c) const {
    return inv_quant_dc_ * dequant_->DCQuant(c);
  }
  JXL_INLINE float GetInvDcStep(size_t c) const {
    return dequant_->InvDCQuant(c) * (global_scale_float_ * quant_dc_);
  }

  JXL_INLINE const float* MulDC() const { return mul_dc_; }
  JXL_INLINE const float* InvMulDC() const { return inv_mul_dc_; }

  JXL_INLINE void ClearDCMul() {
    std::fill(mul_dc_, mul_dc_ + 4, 1.f);
    std::fill(inv_mul_dc_, inv_mul_dc_ + 4, 1.f);
  }

  void ComputeGlobalScaleAndQuant(float quant_dc, float quant_median,
                                  float quant_median_absd);

 private:
  float mul_dc_[4];
  float inv_mul_dc_[4];

  // These are serialized:
  int global_scale_;
  int quant_dc_;

  // These are derived from global_scale_:
  float inv_global_scale_;
  float global_scale_float_;  // reciprocal of inv_global_scale_
  float inv_quant_dc_;

  float zero_bias_[3];
  const DequantMatrices* dequant_;
};

struct QuantizerParams : public Fields {
  QuantizerParams() { Bundle::Init(this); }
  JXL_FIELDS_NAME(QuantizerParams)

  Status VisitFields(Visitor* JXL_RESTRICT visitor) override;

  uint32_t global_scale;
  uint32_t quant_dc;
};

}  // namespace jxl

#endif  // LIB_JXL_QUANTIZER_H_
