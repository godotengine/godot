// Copyright 2022 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Gamma correction utilities.

#include "sharpyuv/sharpyuv_gamma.h"

#include <assert.h>
#include <float.h>
#include <math.h>

#include "sharpyuv/sharpyuv.h"
#include "src/webp/types.h"

// Gamma correction compensates loss of resolution during chroma subsampling.
// Size of pre-computed table for converting from gamma to linear.
#define GAMMA_TO_LINEAR_TAB_BITS 10
#define GAMMA_TO_LINEAR_TAB_SIZE (1 << GAMMA_TO_LINEAR_TAB_BITS)
static uint32_t kGammaToLinearTabS[GAMMA_TO_LINEAR_TAB_SIZE + 2];
#define LINEAR_TO_GAMMA_TAB_BITS 9
#define LINEAR_TO_GAMMA_TAB_SIZE (1 << LINEAR_TO_GAMMA_TAB_BITS)
static uint32_t kLinearToGammaTabS[LINEAR_TO_GAMMA_TAB_SIZE + 2];

#if defined(_MSC_VER)
static const double kGammaF = 2.222222222222222;
#else
static const double kGammaF = 1. / 0.45;
#endif
#define GAMMA_TO_LINEAR_BITS 16

static volatile int kGammaTablesSOk = 0;
void SharpYuvInitGammaTables(void) {
  assert(GAMMA_TO_LINEAR_BITS <= 16);
  if (!kGammaTablesSOk) {
    int v;
    const double a = 0.09929682680944;
    const double thresh = 0.018053968510807;
    const double final_scale = 1 << GAMMA_TO_LINEAR_BITS;
    // Precompute gamma to linear table.
    {
      const double norm = 1. / GAMMA_TO_LINEAR_TAB_SIZE;
      const double a_rec = 1. / (1. + a);
      for (v = 0; v <= GAMMA_TO_LINEAR_TAB_SIZE; ++v) {
        const double g = norm * v;
        double value;
        if (g <= thresh * 4.5) {
          value = g / 4.5;
        } else {
          value = pow(a_rec * (g + a), kGammaF);
        }
        kGammaToLinearTabS[v] = (uint32_t)(value * final_scale + .5);
      }
      // to prevent small rounding errors to cause read-overflow:
      kGammaToLinearTabS[GAMMA_TO_LINEAR_TAB_SIZE + 1] =
          kGammaToLinearTabS[GAMMA_TO_LINEAR_TAB_SIZE];
    }
    // Precompute linear to gamma table.
    {
      const double scale = 1. / LINEAR_TO_GAMMA_TAB_SIZE;
      for (v = 0; v <= LINEAR_TO_GAMMA_TAB_SIZE; ++v) {
        const double g = scale * v;
        double value;
        if (g <= thresh) {
          value = 4.5 * g;
        } else {
          value = (1. + a) * pow(g, 1. / kGammaF) - a;
        }
        kLinearToGammaTabS[v] =
            (uint32_t)(final_scale * value + 0.5);
      }
      // to prevent small rounding errors to cause read-overflow:
      kLinearToGammaTabS[LINEAR_TO_GAMMA_TAB_SIZE + 1] =
          kLinearToGammaTabS[LINEAR_TO_GAMMA_TAB_SIZE];
    }
    kGammaTablesSOk = 1;
  }
}

static WEBP_INLINE int Shift(int v, int shift) {
  return (shift >= 0) ? (v << shift) : (v >> -shift);
}

static WEBP_INLINE uint32_t FixedPointInterpolation(int v, uint32_t* tab,
                                                    int tab_pos_shift_right,
                                                    int tab_value_shift) {
  const uint32_t tab_pos = Shift(v, -tab_pos_shift_right);
  // fractional part, in 'tab_pos_shift' fixed-point precision
  const uint32_t x = v - (tab_pos << tab_pos_shift_right);  // fractional part
  // v0 / v1 are in kGammaToLinearBits fixed-point precision (range [0..1])
  const uint32_t v0 = Shift(tab[tab_pos + 0], tab_value_shift);
  const uint32_t v1 = Shift(tab[tab_pos + 1], tab_value_shift);
  // Final interpolation.
  const uint32_t v2 = (v1 - v0) * x;  // note: v1 >= v0.
  const int half =
      (tab_pos_shift_right > 0) ? 1 << (tab_pos_shift_right - 1) : 0;
  const uint32_t result = v0 + ((v2 + half) >> tab_pos_shift_right);
  return result;
}

static uint32_t ToLinearSrgb(uint16_t v, int bit_depth) {
  const int shift = GAMMA_TO_LINEAR_TAB_BITS - bit_depth;
  if (shift > 0) {
    return kGammaToLinearTabS[v << shift];
  }
  return FixedPointInterpolation(v, kGammaToLinearTabS, -shift, 0);
}

static uint16_t FromLinearSrgb(uint32_t value, int bit_depth) {
  return FixedPointInterpolation(
      value, kLinearToGammaTabS,
      (GAMMA_TO_LINEAR_BITS - LINEAR_TO_GAMMA_TAB_BITS),
      bit_depth - GAMMA_TO_LINEAR_BITS);
}

////////////////////////////////////////////////////////////////////////////////

#define CLAMP(x, low, high) \
  (((x) < (low)) ? (low) : (((high) < (x)) ? (high) : (x)))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

static WEBP_INLINE float Roundf(float x) {
  if (x < 0)
    return (float)ceil((double)(x - 0.5f));
  else
    return (float)floor((double)(x + 0.5f));
}

static WEBP_INLINE float Powf(float base, float exp) {
  return (float)pow((double)base, (double)exp);
}

static WEBP_INLINE float Log10f(float x) { return (float)log10((double)x); }

static float ToLinear709(float gamma) {
  if (gamma < 0.f) {
    return 0.f;
  } else if (gamma < 4.5f * 0.018053968510807f) {
    return gamma / 4.5f;
  } else if (gamma < 1.f) {
    return Powf((gamma + 0.09929682680944f) / 1.09929682680944f, 1.f / 0.45f);
  }
  return 1.f;
}

static float FromLinear709(float linear) {
  if (linear < 0.f) {
    return 0.f;
  } else if (linear < 0.018053968510807f) {
    return linear * 4.5f;
  } else if (linear < 1.f) {
    return 1.09929682680944f * Powf(linear, 0.45f) - 0.09929682680944f;
  }
  return 1.f;
}

static float ToLinear470M(float gamma) {
  return Powf(CLAMP(gamma, 0.f, 1.f), 2.2f);
}

static float FromLinear470M(float linear) {
  return Powf(CLAMP(linear, 0.f, 1.f), 1.f / 2.2f);
}

static float ToLinear470Bg(float gamma) {
  return Powf(CLAMP(gamma, 0.f, 1.f), 2.8f);
}

static float FromLinear470Bg(float linear) {
  return Powf(CLAMP(linear, 0.f, 1.f), 1.f / 2.8f);
}

static float ToLinearSmpte240(float gamma) {
  if (gamma < 0.f) {
    return 0.f;
  } else if (gamma < 4.f * 0.022821585529445f) {
    return gamma / 4.f;
  } else if (gamma < 1.f) {
    return Powf((gamma + 0.111572195921731f) / 1.111572195921731f, 1.f / 0.45f);
  }
  return 1.f;
}

static float FromLinearSmpte240(float linear) {
  if (linear < 0.f) {
    return 0.f;
  } else if (linear < 0.022821585529445f) {
    return linear * 4.f;
  } else if (linear < 1.f) {
    return 1.111572195921731f * Powf(linear, 0.45f) - 0.111572195921731f;
  }
  return 1.f;
}

static float ToLinearLog100(float gamma) {
  // The function is non-bijective so choose the middle of [0, 0.01].
  const float mid_interval = 0.01f / 2.f;
  return (gamma <= 0.0f) ? mid_interval
                          : Powf(10.0f, 2.f * (MIN(gamma, 1.f) - 1.0f));
}

static float FromLinearLog100(float linear) {
  return (linear < 0.01f) ? 0.0f : 1.0f + Log10f(MIN(linear, 1.f)) / 2.0f;
}

static float ToLinearLog100Sqrt10(float gamma) {
  // The function is non-bijective so choose the middle of [0, 0.00316227766f[.
  const float mid_interval = 0.00316227766f / 2.f;
  return (gamma <= 0.0f) ? mid_interval
                          : Powf(10.0f, 2.5f * (MIN(gamma, 1.f) - 1.0f));
}

static float FromLinearLog100Sqrt10(float linear) {
  return (linear < 0.00316227766f) ? 0.0f
                                  : 1.0f + Log10f(MIN(linear, 1.f)) / 2.5f;
}

static float ToLinearIec61966(float gamma) {
  if (gamma <= -4.5f * 0.018053968510807f) {
    return Powf((-gamma + 0.09929682680944f) / -1.09929682680944f, 1.f / 0.45f);
  } else if (gamma < 4.5f * 0.018053968510807f) {
    return gamma / 4.5f;
  }
  return Powf((gamma + 0.09929682680944f) / 1.09929682680944f, 1.f / 0.45f);
}

static float FromLinearIec61966(float linear) {
  if (linear <= -0.018053968510807f) {
    return -1.09929682680944f * Powf(-linear, 0.45f) + 0.09929682680944f;
  } else if (linear < 0.018053968510807f) {
    return linear * 4.5f;
  }
  return 1.09929682680944f * Powf(linear, 0.45f) - 0.09929682680944f;
}

static float ToLinearBt1361(float gamma) {
  if (gamma < -0.25f) {
    return -0.25f;
  } else if (gamma < 0.f) {
    return Powf((gamma - 0.02482420670236f) / -0.27482420670236f, 1.f / 0.45f) /
           -4.f;
  } else if (gamma < 4.5f * 0.018053968510807f) {
    return gamma / 4.5f;
  } else if (gamma < 1.f) {
    return Powf((gamma + 0.09929682680944f) / 1.09929682680944f, 1.f / 0.45f);
  }
  return 1.f;
}

static float FromLinearBt1361(float linear) {
  if (linear < -0.25f) {
    return -0.25f;
  } else if (linear < 0.f) {
    return -0.27482420670236f * Powf(-4.f * linear, 0.45f) + 0.02482420670236f;
  } else if (linear < 0.018053968510807f) {
    return linear * 4.5f;
  } else if (linear < 1.f) {
    return 1.09929682680944f * Powf(linear, 0.45f) - 0.09929682680944f;
  }
  return 1.f;
}

static float ToLinearPq(float gamma) {
  if (gamma > 0.f) {
    const float pow_gamma = Powf(gamma, 32.f / 2523.f);
    const float num = MAX(pow_gamma - 107.f / 128.f, 0.0f);
    const float den = MAX(2413.f / 128.f - 2392.f / 128.f * pow_gamma, FLT_MIN);
    return Powf(num / den, 4096.f / 653.f);
  }
  return 0.f;
}

static float FromLinearPq(float linear) {
  if (linear > 0.f) {
    const float pow_linear = Powf(linear, 653.f / 4096.f);
    const float num = 107.f / 128.f + 2413.f / 128.f * pow_linear;
    const float den = 1.0f + 2392.f / 128.f * pow_linear;
    return Powf(num / den, 2523.f / 32.f);
  }
  return 0.f;
}

static float ToLinearSmpte428(float gamma) {
  return Powf(MAX(gamma, 0.f), 2.6f) / 0.91655527974030934f;
}

static float FromLinearSmpte428(float linear) {
  return Powf(0.91655527974030934f * MAX(linear, 0.f), 1.f / 2.6f);
}

// Conversion in BT.2100 requires RGB info. Simplify to gamma correction here.
static float ToLinearHlg(float gamma) {
  if (gamma < 0.f) {
    return 0.f;
  } else if (gamma <= 0.5f) {
    return Powf((gamma * gamma) * (1.f / 3.f), 1.2f);
  }
  return Powf((expf((gamma - 0.55991073f) / 0.17883277f) + 0.28466892f) / 12.0f,
              1.2f);
}

static float FromLinearHlg(float linear) {
  linear = Powf(linear, 1.f / 1.2f);
  if (linear < 0.f) {
    return 0.f;
  } else if (linear <= (1.f / 12.f)) {
    return sqrtf(3.f * linear);
  }
  return 0.17883277f * logf(12.f * linear - 0.28466892f) + 0.55991073f;
}

uint32_t SharpYuvGammaToLinear(uint16_t v, int bit_depth,
                               SharpYuvTransferFunctionType transfer_type) {
  float v_float, linear;
  if (transfer_type == kSharpYuvTransferFunctionSrgb) {
    return ToLinearSrgb(v, bit_depth);
  }
  v_float = (float)v / ((1 << bit_depth) - 1);
  switch (transfer_type) {
    case kSharpYuvTransferFunctionBt709:
    case kSharpYuvTransferFunctionBt601:
    case kSharpYuvTransferFunctionBt2020_10Bit:
    case kSharpYuvTransferFunctionBt2020_12Bit:
      linear = ToLinear709(v_float);
      break;
    case kSharpYuvTransferFunctionBt470M:
      linear = ToLinear470M(v_float);
      break;
    case kSharpYuvTransferFunctionBt470Bg:
      linear = ToLinear470Bg(v_float);
      break;
    case kSharpYuvTransferFunctionSmpte240:
      linear = ToLinearSmpte240(v_float);
      break;
    case kSharpYuvTransferFunctionLinear:
      return v;
    case kSharpYuvTransferFunctionLog100:
      linear = ToLinearLog100(v_float);
      break;
    case kSharpYuvTransferFunctionLog100_Sqrt10:
      linear = ToLinearLog100Sqrt10(v_float);
      break;
    case kSharpYuvTransferFunctionIec61966:
      linear = ToLinearIec61966(v_float);
      break;
    case kSharpYuvTransferFunctionBt1361:
      linear = ToLinearBt1361(v_float);
      break;
    case kSharpYuvTransferFunctionSmpte2084:
      linear = ToLinearPq(v_float);
      break;
    case kSharpYuvTransferFunctionSmpte428:
      linear = ToLinearSmpte428(v_float);
      break;
    case kSharpYuvTransferFunctionHlg:
      linear = ToLinearHlg(v_float);
      break;
    default:
      assert(0);
      linear = 0;
      break;
  }
  return (uint32_t)Roundf(linear * ((1 << 16) - 1));
}

uint16_t SharpYuvLinearToGamma(uint32_t v, int bit_depth,
                               SharpYuvTransferFunctionType transfer_type) {
  float v_float, linear;
  if (transfer_type == kSharpYuvTransferFunctionSrgb) {
    return FromLinearSrgb(v, bit_depth);
  }
  v_float = (float)v / ((1 << 16) - 1);
  switch (transfer_type) {
    case kSharpYuvTransferFunctionBt709:
    case kSharpYuvTransferFunctionBt601:
    case kSharpYuvTransferFunctionBt2020_10Bit:
    case kSharpYuvTransferFunctionBt2020_12Bit:
      linear = FromLinear709(v_float);
      break;
    case kSharpYuvTransferFunctionBt470M:
      linear = FromLinear470M(v_float);
      break;
    case kSharpYuvTransferFunctionBt470Bg:
      linear = FromLinear470Bg(v_float);
      break;
    case kSharpYuvTransferFunctionSmpte240:
      linear = FromLinearSmpte240(v_float);
      break;
    case kSharpYuvTransferFunctionLinear:
      return v;
    case kSharpYuvTransferFunctionLog100:
      linear = FromLinearLog100(v_float);
      break;
    case kSharpYuvTransferFunctionLog100_Sqrt10:
      linear = FromLinearLog100Sqrt10(v_float);
      break;
    case kSharpYuvTransferFunctionIec61966:
      linear = FromLinearIec61966(v_float);
      break;
    case kSharpYuvTransferFunctionBt1361:
      linear = FromLinearBt1361(v_float);
      break;
    case kSharpYuvTransferFunctionSmpte2084:
      linear = FromLinearPq(v_float);
      break;
    case kSharpYuvTransferFunctionSmpte428:
      linear = FromLinearSmpte428(v_float);
      break;
    case kSharpYuvTransferFunctionHlg:
      linear = FromLinearHlg(v_float);
      break;
    default:
      assert(0);
      linear = 0;
      break;
  }
  return (uint16_t)Roundf(linear * ((1 << bit_depth) - 1));
}
