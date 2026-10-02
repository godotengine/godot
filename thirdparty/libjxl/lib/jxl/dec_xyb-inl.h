// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// XYB -> linear sRGB helper function.

#if defined(LIB_JXL_DEC_XYB_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef LIB_JXL_DEC_XYB_INL_H_
#undef LIB_JXL_DEC_XYB_INL_H_
#else
#define LIB_JXL_DEC_XYB_INL_H_
#endif

#include <hwy/highway.h>

#include "lib/jxl/dec_xyb.h"
HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {
namespace {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::Add;
using hwy::HWY_NAMESPACE::Broadcast;
using hwy::HWY_NAMESPACE::Mul;
using hwy::HWY_NAMESPACE::MulAdd;
using hwy::HWY_NAMESPACE::Sub;

// Inverts the pixel-wise RGB->XYB conversion in OpsinDynamicsImage() (including
// the gamma mixing and simple gamma). Avoids clamping to [0, 1] - out of (sRGB)
// gamut values may be in-gamut after transforming to a wider space.
// "inverse_matrix" points to 9 broadcasted vectors, which are the 3x3 entries
// of the (row-major) opsin absorbance matrix inverse. Pre-multiplying its
// entries by c is equivalent to multiplying linear_* by c afterwards.
template <class D, class V>
HWY_INLINE HWY_MAYBE_UNUSED void XybToRgb(D d, const V opsin_x, const V opsin_y,
                                          const V opsin_b,
                                          const OpsinParams& opsin_params,
                                          V* const HWY_RESTRICT linear_r,
                                          V* const HWY_RESTRICT linear_g,
                                          V* const HWY_RESTRICT linear_b) {
#if HWY_TARGET == HWY_SCALAR
  const auto neg_bias_r = Set(d, opsin_params.opsin_biases[0]);
  const auto neg_bias_g = Set(d, opsin_params.opsin_biases[1]);
  const auto neg_bias_b = Set(d, opsin_params.opsin_biases[2]);
#else
  const auto neg_bias_rgb = LoadDup128(d, opsin_params.opsin_biases);
  const auto neg_bias_r = Broadcast<0>(neg_bias_rgb);
  const auto neg_bias_g = Broadcast<1>(neg_bias_rgb);
  const auto neg_bias_b = Broadcast<2>(neg_bias_rgb);
#endif

  // Color space: XYB -> RGB
  auto gamma_r = Add(opsin_y, opsin_x);
  auto gamma_g = Sub(opsin_y, opsin_x);
  auto gamma_b = opsin_b;

  gamma_r = Sub(gamma_r, Set(d, opsin_params.opsin_biases_cbrt[0]));
  gamma_g = Sub(gamma_g, Set(d, opsin_params.opsin_biases_cbrt[1]));
  gamma_b = Sub(gamma_b, Set(d, opsin_params.opsin_biases_cbrt[2]));

  // Undo gamma compression: linear = gamma^3 for efficiency.
  const auto gamma_r2 = Mul(gamma_r, gamma_r);
  const auto gamma_g2 = Mul(gamma_g, gamma_g);
  const auto gamma_b2 = Mul(gamma_b, gamma_b);
  const auto mixed_r = MulAdd(gamma_r2, gamma_r, neg_bias_r);
  const auto mixed_g = MulAdd(gamma_g2, gamma_g, neg_bias_g);
  const auto mixed_b = MulAdd(gamma_b2, gamma_b, neg_bias_b);

  const float* HWY_RESTRICT inverse_matrix = opsin_params.inverse_opsin_matrix;

  // Unmix (multiply by 3x3 inverse_matrix)
  // TODO(eustas): ref would be more readable than pointer
  *linear_r = Mul(LoadDup128(d, &inverse_matrix[0 * 4]), mixed_r);
  *linear_g = Mul(LoadDup128(d, &inverse_matrix[3 * 4]), mixed_r);
  *linear_b = Mul(LoadDup128(d, &inverse_matrix[6 * 4]), mixed_r);
  *linear_r = MulAdd(LoadDup128(d, &inverse_matrix[1 * 4]), mixed_g, *linear_r);
  *linear_g = MulAdd(LoadDup128(d, &inverse_matrix[4 * 4]), mixed_g, *linear_g);
  *linear_b = MulAdd(LoadDup128(d, &inverse_matrix[7 * 4]), mixed_g, *linear_b);
  *linear_r = MulAdd(LoadDup128(d, &inverse_matrix[2 * 4]), mixed_b, *linear_r);
  *linear_g = MulAdd(LoadDup128(d, &inverse_matrix[5 * 4]), mixed_b, *linear_g);
  *linear_b = MulAdd(LoadDup128(d, &inverse_matrix[8 * 4]), mixed_b, *linear_b);
}

inline HWY_MAYBE_UNUSED bool HasFastXYBTosRGB8() {
#if HWY_TARGET == HWY_NEON
  return true;
#else
  return false;
#endif
}

inline HWY_MAYBE_UNUSED Status FastXYBTosRGB8(const float* input[4],
                                              uint8_t* output, bool is_rgba,
                                              size_t xsize) {
  // This function is very NEON-specific. As such, it uses intrinsics directly.
#if HWY_TARGET == HWY_NEON
  // WARNING: doing fixed point arithmetic correctly is very complicated.
  // Changes to this function should be thoroughly tested.

  // Note that the input is assumed to have 13 bits of mantissa, and the output
  // will have 14 bits.
  auto srgb_tf = [&](int16x8_t v16) {
    int16x8_t clz = vclzq_s16(v16);
    // Convert to [0.25, 0.5) range.
    int16x8_t v025_05_16 = vqshlq_s16(v16, vqsubq_s16(clz, vdupq_n_s16(2)));

    // third degree polynomial approximation between 0.25 and 0.5
    // of 1.055/2^(7/2.4) * x^(1/2.4) / 32.
    // poly ~ ((0.95x-1.75)*x+1.72)*x+0.29
    // We actually compute ~ ((0.47x-0.87)*x+0.86)*(2x)+0.29 as 1.75 and 1.72
    // overflow our fixed point representation.

    int16x8_t twov = vqaddq_s16(v025_05_16, v025_05_16);

    // 0.47 * x
    int16x8_t step1 = vqrdmulhq_n_s16(v025_05_16, 15706);
    // - 0.87
    int16x8_t step2 = vsubq_s16(step1, vdupq_n_s16(28546));
    // * x
    int16x8_t step3 = vqrdmulhq_s16(step2, v025_05_16);
    // + 0.86
    int16x8_t step4 = vaddq_s16(step3, vdupq_n_s16(28302));
    // * 2x
    int16x8_t step5 = vqrdmulhq_s16(step4, twov);
    // + 0.29
    int16x8_t mul16 = vaddq_s16(step5, vdupq_n_s16(9485));

    int16x8_t exp16 = vsubq_s16(vdupq_n_s16(11), clz);
    // Compute 2**(1/2.4*exp16)/32. Values of exp16 that would overflow are
    // capped to 1.
    // Generated with the following Python script:
    // a = []
    // b = []
    //
    // for i in range(0, 16):
    //   v = 2**(5/12.*i)
    //   v /= 16
    //   v *= 256 * 128
    //   v = int(v)
    //   a.append(v // 256)
    //   b.append(v % 256)
    //
    // print(", ".join("0x%02x" % x for x in a))
    //
    // print(", ".join("0x%02x" % x for x in b))

    HWY_ALIGN constexpr uint8_t k2to512powersm1div32_high[16] = {
        0x08, 0x0a, 0x0e, 0x13, 0x19, 0x21, 0x2d, 0x3c,
        0x50, 0x6b, 0x8f, 0x8f, 0x8f, 0x8f, 0x8f, 0x8f,
    };
    HWY_ALIGN constexpr uint8_t k2to512powersm1div32_low[16] = {
        0x00, 0xad, 0x41, 0x06, 0x65, 0xe7, 0x41, 0x68,
        0xa2, 0xa2, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    };
    // Using the highway implementation here since vqtbl1q is aarch64-only.
    using hwy::HWY_NAMESPACE::Vec128;
    uint8x16_t pow_low =
        TableLookupBytes(
            Vec128<uint8_t, 16>(vld1q_u8(k2to512powersm1div32_low)),
            Vec128<uint8_t, 16>(vreinterpretq_u8_s16(exp16)))
            .raw;
    uint8x16_t pow_high =
        TableLookupBytes(
            Vec128<uint8_t, 16>(vld1q_u8(k2to512powersm1div32_high)),
            Vec128<uint8_t, 16>(vreinterpretq_u8_s16(exp16)))
            .raw;
    int16x8_t pow16 = vreinterpretq_s16_u16(vsliq_n_u16(
        vreinterpretq_u16_u8(pow_low), vreinterpretq_u16_u8(pow_high), 8));

    // approximation of v * 12.92, divided by 2
    // Note that our input is using 13 mantissa bits instead of 15.
    int16x8_t v16_linear = vrshrq_n_s16(vmulq_n_s16(v16, 826), 5);
    // 1.055*pow(v, 1/2.4) - 0.055, divided by 2
    auto v16_pow = vsubq_s16(vqrdmulhq_s16(mul16, pow16), vdupq_n_s16(901));
    // > 0.0031308f (note that v16 has 13 mantissa bits)
    return vbslq_s16(vcgeq_s16(v16, vdupq_n_s16(26)), v16_pow, v16_linear);
  };

  const float* JXL_RESTRICT row_in_x = input[0];
  const float* JXL_RESTRICT row_in_y = input[1];
  const float* JXL_RESTRICT row_in_b = input[2];
  const float* JXL_RESTRICT row_in_a = input[3];
  for (size_t x = 0; x < xsize; x += 8) {
    // Normal ranges for xyb for in-gamut sRGB colors:
    // x: -0.015386 0.028100
    // y: 0.000000 0.845308
    // b: 0.000000 0.845308

    // We actually want x * 8 to have some extra precision.
    // TODO(veluca): consider different approaches here, like vld1q_f32_x2.
    float32x4_t opsin_x_left = vld1q_f32(row_in_x + x);
    int16x4_t opsin_x16_times8_left =
        vqmovn_s32(vcvtq_n_s32_f32(opsin_x_left, 18));
    float32x4_t opsin_x_right =
        vld1q_f32(row_in_x + x + (x + 4 < xsize ? 4 : 0));
    int16x4_t opsin_x16_times8_right =
        vqmovn_s32(vcvtq_n_s32_f32(opsin_x_right, 18));
    int16x8_t opsin_x16_times8 =
        vcombine_s16(opsin_x16_times8_left, opsin_x16_times8_right);

    float32x4_t opsin_y_left = vld1q_f32(row_in_y + x);
    int16x4_t opsin_y16_left = vqmovn_s32(vcvtq_n_s32_f32(opsin_y_left, 15));
    float32x4_t opsin_y_right =
        vld1q_f32(row_in_y + x + (x + 4 < xsize ? 4 : 0));
    int16x4_t opsin_y16_right = vqmovn_s32(vcvtq_n_s32_f32(opsin_y_right, 15));
    int16x8_t opsin_y16 = vcombine_s16(opsin_y16_left, opsin_y16_right);

    float32x4_t opsin_b_left = vld1q_f32(row_in_b + x);
    int16x4_t opsin_b16_left = vqmovn_s32(vcvtq_n_s32_f32(opsin_b_left, 15));
    float32x4_t opsin_b_right =
        vld1q_f32(row_in_b + x + (x + 4 < xsize ? 4 : 0));
    int16x4_t opsin_b16_right = vqmovn_s32(vcvtq_n_s32_f32(opsin_b_right, 15));
    int16x8_t opsin_b16 = vcombine_s16(opsin_b16_left, opsin_b16_right);

    int16x8_t neg_bias16 = vdupq_n_s16(-124);        // -0.0037930732552754493
    int16x8_t neg_bias_cbrt16 = vdupq_n_s16(-5110);  // -0.155954201
    int16x8_t neg_bias_half16 = vdupq_n_s16(-62);

    // Color space: XYB -> RGB
    // Compute ((y+x-bias_cbrt)^3-(y-x-bias_cbrt)^3)/2,
    // ((y+x-bias_cbrt)^3+(y-x-bias_cbrt)^3)/2+bias, (b-bias_cbrt)^3+bias.
    // Note that ignoring x2 in the formulas below (as x << y) results in
    // errors of at least 3 in the final sRGB values.
    int16x8_t opsin_yp16 = vqsubq_s16(opsin_y16, neg_bias_cbrt16);
    int16x8_t ysq16 = vqrdmulhq_s16(opsin_yp16, opsin_yp16);
    int16x8_t twentyfourx16 = vmulq_n_s16(opsin_x16_times8, 3);
    int16x8_t twentyfourxy16 = vqrdmulhq_s16(opsin_yp16, twentyfourx16);
    int16x8_t threexsq16 =
        vrshrq_n_s16(vqrdmulhq_s16(opsin_x16_times8, twentyfourx16), 6);

    // We can ignore x^3 here. Note that this is multiplied by 8.
    int16x8_t mixed_rmg16 = vqrdmulhq_s16(twentyfourxy16, opsin_yp16);

    int16x8_t mixed_rpg_sos_half = vhaddq_s16(ysq16, threexsq16);
    int16x8_t mixed_rpg16 = vhaddq_s16(
        vqrdmulhq_s16(opsin_yp16, mixed_rpg_sos_half), neg_bias_half16);

    int16x8_t gamma_b16 = vqsubq_s16(opsin_b16, neg_bias_cbrt16);
    int16x8_t gamma_bsq16 = vqrdmulhq_s16(gamma_b16, gamma_b16);
    int16x8_t gamma_bcb16 = vqrdmulhq_s16(gamma_bsq16, gamma_b16);
    int16x8_t mixed_b16 = vqaddq_s16(gamma_bcb16, neg_bias16);
    // mixed_rpg and mixed_b are in 0-1 range.
    // mixed_rmg has a smaller range (-0.035 to 0.035 for valid sRGB). Note
    // that at this point it is already multiplied by 8.

    // We multiply all the mixed values by 1/4 (i.e. shift them to 13-bit
    // fixed point) to ensure intermediate quantities are in range. Note that
    // r-g is not shifted, and was x8 before here; this corresponds to a x32
    // overall multiplicative factor and ensures that all the matrix constants
    // are in 0-1 range.
    // Similarly, mixed_rpg16 is already multiplied by 1/4 because of the two
    // vhadd + using neg_bias_half.
    mixed_b16 = vshrq_n_s16(mixed_b16, 2);

    // Unmix (multiply by 3x3 inverse_matrix)
    // For increased precision, we use a matrix for converting from
    // ((mixed_r - mixed_g)/2, (mixed_r + mixed_g)/2, mixed_b) to rgb. This
    // avoids cancellation effects when computing (y+x)^3-(y-x)^3.
    // We compute mixed_rpg - mixed_b because the (1+c)*mixed_rpg - c *
    // mixed_b pattern is repeated frequently in the code below. This allows
    // us to save a multiply per channel, and removes the presence of
    // some constants above 1. Moreover, mixed_rmg - mixed_b is in (-1, 1)
    // range, so the subtraction is safe.
    // All the magic-looking constants here are derived by computing the
    // inverse opsin matrix for the transformation modified as described
    // above.

    // Precomputation common to multiple color values.
    int16x8_t mixed_rpgmb16 = vqsubq_s16(mixed_rpg16, mixed_b16);
    int16x8_t mixed_rpgmb_times_016 = vqrdmulhq_n_s16(mixed_rpgmb16, 5394);
    int16x8_t mixed_rg16 = vqaddq_s16(mixed_rpgmb_times_016, mixed_rpg16);

    // R
    int16x8_t linear_r16 =
        vqaddq_s16(mixed_rg16, vqrdmulhq_n_s16(mixed_rmg16, 21400));

    // G
    int16x8_t linear_g16 =
        vqaddq_s16(mixed_rg16, vqrdmulhq_n_s16(mixed_rmg16, -7857));

    // B
    int16x8_t linear_b16 = vqrdmulhq_n_s16(mixed_rpgmb16, -30996);
    linear_b16 = vqaddq_s16(linear_b16, mixed_b16);
    linear_b16 = vqaddq_s16(linear_b16, vqrdmulhq_n_s16(mixed_rmg16, -6525));

    // Apply SRGB transfer function.
    int16x8_t r = srgb_tf(linear_r16);
    int16x8_t g = srgb_tf(linear_g16);
    int16x8_t b = srgb_tf(linear_b16);

    uint8x8_t r8 =
        vqmovun_s16(vrshrq_n_s16(vsubq_s16(r, vshrq_n_s16(r, 8)), 6));
    uint8x8_t g8 =
        vqmovun_s16(vrshrq_n_s16(vsubq_s16(g, vshrq_n_s16(g, 8)), 6));
    uint8x8_t b8 =
        vqmovun_s16(vrshrq_n_s16(vsubq_s16(b, vshrq_n_s16(b, 8)), 6));

    size_t n = xsize - x;
    if (is_rgba) {
      float32x4_t a_f32_left =
          row_in_a ? vld1q_f32(row_in_a + x) : vdupq_n_f32(1.0f);
      float32x4_t a_f32_right =
          row_in_a ? vld1q_f32(row_in_a + x + (x + 4 < xsize ? 4 : 0))
                   : vdupq_n_f32(1.0f);
      int16x4_t a16_left = vqmovn_s32(vcvtq_n_s32_f32(a_f32_left, 8));
      int16x4_t a16_right = vqmovn_s32(vcvtq_n_s32_f32(a_f32_right, 8));
      uint8x8_t a8 = vqmovun_s16(vcombine_s16(a16_left, a16_right));
      uint8_t* buf = output + 4 * x;
      uint8x8x4_t data = {r8, g8, b8, a8};
      if (n >= 8) {
        vst4_u8(buf, data);
      } else {
        uint8_t tmp[8 * 4];
        vst4_u8(tmp, data);
        memcpy(buf, tmp, n * 4);
      }
    } else {
      uint8_t* buf = output + 3 * x;
      uint8x8x3_t data = {r8, g8, b8};
      if (n >= 8) {
        vst3_u8(buf, data);
      } else {
        uint8_t tmp[8 * 3];
        vst3_u8(tmp, data);
        memcpy(buf, tmp, n * 3);
      }
    }
  }
  return true;
#else
  (void)input;
  (void)output;
  (void)is_rgba;
  (void)xsize;
  return JXL_UNREACHABLE("unsupported platform");
#endif
}

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#endif  // LIB_JXL_DEC_XYB_INL_H_
