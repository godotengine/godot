// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_photon_noise.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "lib/jxl/base/common.h"
#include "lib/jxl/cms/opsin_params.h"
#include "lib/jxl/noise.h"

namespace jxl {

namespace {

// Assumes a daylight-like spectrum.
// https://www.strollswithmydog.com/effective-quantum-efficiency-of-sensor/#:~:text=11%2C260%20photons/um%5E2/lx-s
constexpr float kPhotonsPerLxSPerUm2 = 11260;

// Order of magnitude for cameras in the 2010-2020 decade, taking the CFA into
// account.
constexpr float kEffectiveQuantumEfficiency = 0.20;

// TODO(sboukortt): reevaluate whether these are good defaults, notably whether
// it would be worth making read noise higher at lower ISO settings.
constexpr float kPhotoResponseNonUniformity = 0.005;
constexpr float kInputReferredReadNoise = 3;

// Assumes a 35mm sensor.
constexpr float kSensorAreaUm2 = 36000.f * 24000;

template <typename T>
inline constexpr T Square(const T x) {
  return x * x;
}
template <typename T>
inline constexpr T Cube(const T x) {
  return x * x * x;
}

}  // namespace

NoiseParams SimulatePhotonNoise(const size_t xsize, const size_t ysize,
                                const float iso) {
  const float kOpsinAbsorbanceBiasCbrt =
      std::cbrt(jxl::cms::kOpsinAbsorbanceBias[1]);

  // Focal plane exposure for 18% of kDefaultIntensityTarget, in lx·s.
  // (ISO = 10 lx·s ÷ H)
  const float h_18 = 10 / iso;

  const float pixel_area_um2 = kSensorAreaUm2 / (xsize * ysize);

  const float electrons_per_pixel_18 = kEffectiveQuantumEfficiency *
                                       kPhotonsPerLxSPerUm2 * h_18 *
                                       pixel_area_um2;

  NoiseParams params;

  for (size_t i = 0; i < NoiseParams::kNumNoisePoints; ++i) {
    const float scaled_index = i / (NoiseParams::kNumNoisePoints - 2.f);
    // scaled_index is used for XYB = (0, 2·scaled_index, 2·scaled_index)
    const float y = 2 * scaled_index;
    // 1 = default intensity target
    const float linear = std::max(0.f, Cube(y - kOpsinAbsorbanceBiasCbrt) +
                                           jxl::cms::kOpsinAbsorbanceBias[1]);
    const float electrons_per_pixel = electrons_per_pixel_18 * (linear / 0.18f);
    // Quadrature sum of read noise, photon shot noise (sqrt(S) so simply not
    // squared here) and photo response non-uniformity.
    // https://doi.org/10.1117/3.725073
    // Units are electrons rms.
    const float noise =
        std::sqrt(Square(kInputReferredReadNoise) + electrons_per_pixel +
                  Square(kPhotoResponseNonUniformity * electrons_per_pixel));
    const float linear_noise = noise * (0.18f / electrons_per_pixel_18);
    const float opsin_derivative =
        (1.f / 3) /
        Square(std::cbrt(linear - jxl::cms::kOpsinAbsorbanceBias[1]));
    const float opsin_noise = linear_noise * opsin_derivative;

    // TODO(sboukortt): verify more thoroughly whether the denominator is
    // correct.
    params.lut[i] =
        Clamp1(opsin_noise /
                   (0.22f             // norm_const
                    * std::sqrt(2.f)  // red_noise + green_noise
                    * 1.13f  // standard deviation of a plane of generated noise
                    ),
               0.f, kNoiseLutMax);
  }

  return params;
}

}  // namespace jxl
