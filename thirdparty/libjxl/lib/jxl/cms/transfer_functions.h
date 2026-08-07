// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Transfer functions for color encodings.

#ifndef LIB_JXL_CMS_TRANSFER_FUNCTIONS_H_
#define LIB_JXL_CMS_TRANSFER_FUNCTIONS_H_

#include <algorithm>
#include <cmath>

#include "lib/jxl/base/status.h"

namespace jxl {

// Definitions for BT.2100-2 transfer functions (used inside/outside SIMD):
// "display" is linear light (nits) normalized to [0, 1].
// "encoded" is a nonlinear encoding (e.g. PQ) in [0, 1].
// "scene" is a linear function of photon counts, normalized to [0, 1].

// Despite the stated ranges, we need unbounded transfer functions: see
// http://www.littlecms.com/CIC18_UnboundedCMM.pdf. Inputs can be negative or
// above 1 due to chromatic adaptation. To avoid severe round-trip errors caused
// by clamping, we mirror negative inputs via copysign (f(-x) = -f(x), see
// https://developer.apple.com/documentation/coregraphics/cgcolorspace/1644735-extendedsrgb)
// and extend the function domains above 1.

// Hybrid Log-Gamma.
class TF_HLG_Base {
 public:
  // EOTF. e = encoded.
  static double DisplayFromEncoded(const double e) { return OOTF(InvOETF(e)); }

  // Inverse EOTF. d = display.
  static double EncodedFromDisplay(const double d) { return OETF(InvOOTF(d)); }

 private:
  // OETF (defines the HLG approach). s = scene, returns encoded.
  static double OETF(double s) {
    if (s == 0.0) return 0.0;
    const double original_sign = s;
    s = std::abs(s);

    if (s <= kDiv12) return copysignf(std::sqrt(3.0 * s), original_sign);

    const double e = kA * std::log(12 * s - kB) + kC;
    JXL_DASSERT(e > 0.0);
    return copysignf(e, original_sign);
  }

  // e = encoded, returns scene.
  static double InvOETF(double e) {
    if (e == 0.0) return 0.0;
    const double original_sign = e;
    e = std::abs(e);

    if (e <= 0.5) return copysignf(e * e * (1.0 / 3), original_sign);

    const double s = (std::exp((e - kC) * kRA) + kB) * kDiv12;
    JXL_DASSERT(s >= 0);
    return copysignf(s, original_sign);
  }

  // s = scene, returns display.
  static double OOTF(const double s) {
    // The actual (red channel) OOTF is RD = alpha * YS^(gamma-1) * RS, where
    // YS = 0.2627 * RS + 0.6780 * GS + 0.0593 * BS. Let alpha = 1 so we return
    // "display" (normalized [0, 1]) instead of nits. Our transfer function
    // interface does not allow a dependency on YS. Fortunately, the system
    // gamma at 334 nits is 1.0, so this reduces to RD = RS.
    return s;
  }

  // d = display, returns scene.
  static double InvOOTF(const double d) {
    return d;  // see OOTF().
  }

 protected:
  static constexpr double kA = 0.17883277;
  static constexpr double kRA = 1.0 / kA;
  static constexpr double kB = 1 - 4 * kA;
  static constexpr double kC = 0.5599107295;
  static constexpr double kDiv12 = 1.0 / 12;
};

// Perceptual Quantization
class TF_PQ_Base {
 public:
  static double DisplayFromEncoded(float display_intensity_target, double e) {
    if (e == 0.0) return 0.0;
    const double original_sign = e;
    e = std::abs(e);

    const double xp = std::pow(e, 1.0 / kM2);
    const double num = std::max(xp - kC1, 0.0);
    const double den = kC2 - kC3 * xp;
    JXL_DASSERT(den != 0.0);
    const double d = std::pow(num / den, 1.0 / kM1);
    JXL_DASSERT(d >= 0.0);  // Equal for e ~= 1E-9
    return copysignf(d * (10000.0f / display_intensity_target), original_sign);
  }

  // Inverse EOTF. d = display.
  static double EncodedFromDisplay(float display_intensity_target, double d) {
    if (d == 0.0) return 0.0;
    const double original_sign = d;
    d = std::abs(d);

    const double xp =
        std::pow(d * (display_intensity_target * (1.0f / 10000.0f)), kM1);
    const double num = kC1 + xp * kC2;
    const double den = 1.0 + xp * kC3;
    const double e = std::pow(num / den, kM2);
    JXL_DASSERT(e > 0.0);
    return copysignf(e, original_sign);
  }

 protected:
  static constexpr double kM1 = 2610.0 / 16384;
  static constexpr double kM2 = (2523.0 / 4096) * 128;
  static constexpr double kC1 = 3424.0 / 4096;
  static constexpr double kC2 = (2413.0 / 4096) * 32;
  static constexpr double kC3 = (2392.0 / 4096) * 32;
};

}  // namespace jxl

#endif  // LIB_JXL_CMS_TRANSFER_FUNCTIONS_H_
