// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_MODULAR_TRANSFORM_PALETTE_H_
#define LIB_JXL_MODULAR_TRANSFORM_PALETTE_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/modular/encoding/context_predict.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/options.h"

namespace jxl {

namespace palette_internal {

static constexpr int kMaxPaletteLookupTableSize = 1 << 16;

static constexpr int kRgbChannels = 3;

// 5x5x5 color cube for the larger cube.
static constexpr int kLargeCube = 5;

// Smaller interleaved color cube to fill the holes of the larger cube.
static constexpr int kSmallCube = 4;
static constexpr int kSmallCubeBits = 2;
// kSmallCube ** 3
static constexpr int kLargeCubeOffset = kSmallCube * kSmallCube * kSmallCube;
static constexpr int kImplicitPaletteSize =
    kLargeCubeOffset + kLargeCube * kLargeCube * kLargeCube;

template <int denom>
static inline pixel_type Scale(uint64_t value, uint64_t bit_depth) {
  // return (value * ((static_cast<pixel_type_w>(1) << bit_depth) - 1)) / denom;
  // We only call this function with kSmallCube or kLargeCube - 1 as denom,
  // allowing us to avoid a division here.
  static_assert(denom == 4);
  return (value * ((static_cast<uint64_t>(1) << bit_depth) - 1)) >> 2;
}

// The purpose of this function is solely to extend the interpretation of
// palette indices to implicit values. If index < nb_deltas, indicating that the
// result is a delta palette entry, it is the responsibility of the caller to
// treat it as such.
static JXL_MAYBE_UNUSED pixel_type
GetPaletteValue(const pixel_type *const palette, int index, const size_t c,
                const int palette_size, const int onerow, const int bit_depth) {
  if (index < 0) {
    static constexpr std::array<std::array<pixel_type, 3>, 72> kDeltaPalette = {
        {
            {{0, 0, 0}},       {{4, 4, 4}},       {{11, 0, 0}},
            {{0, 0, -13}},     {{0, -12, 0}},     {{-10, -10, -10}},
            {{-18, -18, -18}}, {{-27, -27, -27}}, {{-18, -18, 0}},
            {{0, 0, -32}},     {{-32, 0, 0}},     {{-37, -37, -37}},
            {{0, -32, -32}},   {{24, 24, 45}},    {{50, 50, 50}},
            {{-45, -24, -24}}, {{-24, -45, -45}}, {{0, -24, -24}},
            {{-34, -34, 0}},   {{-24, 0, -24}},   {{-45, -45, -24}},
            {{64, 64, 64}},    {{-32, 0, -32}},   {{0, -32, 0}},
            {{-32, 0, 32}},    {{-24, -45, -24}}, {{45, 24, 45}},
            {{24, -24, -45}},  {{-45, -24, 24}},  {{80, 80, 80}},
            {{64, 0, 0}},      {{0, 0, -64}},     {{0, -64, -64}},
            {{-24, -24, 45}},  {{96, 96, 96}},    {{64, 64, 0}},
            {{45, -24, -24}},  {{34, -34, 0}},    {{112, 112, 112}},
            {{24, -45, -45}},  {{45, 45, -24}},   {{0, -32, 32}},
            {{24, -24, 45}},   {{0, 96, 96}},     {{45, -24, 24}},
            {{24, -45, -24}},  {{-24, -45, 24}},  {{0, -64, 0}},
            {{96, 0, 0}},      {{128, 128, 128}}, {{64, 0, 64}},
            {{144, 144, 144}}, {{96, 96, 0}},     {{-36, -36, 36}},
            {{45, -24, -45}},  {{45, -45, -24}},  {{0, 0, -96}},
            {{0, 128, 128}},   {{0, 96, 0}},      {{45, 24, -45}},
            {{-128, 0, 0}},    {{24, -45, 24}},   {{-45, 24, -45}},
            {{64, 0, -64}},    {{64, -64, -64}},  {{96, 0, 96}},
            {{45, -45, 24}},   {{24, 45, -45}},   {{64, 64, -64}},
            {{128, 128, 0}},   {{0, 0, -128}},    {{-24, 45, -45}},
        }};
    if (c >= kRgbChannels) {
      return 0;
    }
    // Do not open the brackets, otherwise INT32_MIN negation could overflow.
    index = -(index + 1);
    index %= 1 + 2 * (kDeltaPalette.size() - 1);
    static constexpr int kMultiplier[] = {-1, 1};
    pixel_type result =
        kDeltaPalette[((index + 1) >> 1)][c] * kMultiplier[index & 1];
    if (bit_depth > 8) {
      result *= static_cast<pixel_type>(1) << (bit_depth - 8);
    }
    return result;
  } else if (palette_size <= index && index < palette_size + kLargeCubeOffset) {
    if (c >= kRgbChannels) return 0;
    index -= palette_size;
    index >>= c * kSmallCubeBits;
    return Scale<kSmallCube>(index % kSmallCube, bit_depth) +
           (1 << (std::max(0, bit_depth - 3)));
  } else if (palette_size + kLargeCubeOffset <= index) {
    if (c >= kRgbChannels) return 0;
    index -= palette_size + kLargeCubeOffset;
    // TODO(eustas): should we take care of ambiguity created by
    //               index >= kLargeCube ** 3 ?
    switch (c) {
      case 0:
      default:
        break;
      case 1:
        index /= kLargeCube;
        break;
      case 2:
        index /= kLargeCube * kLargeCube;
        break;
    }
    return Scale<kLargeCube - 1>(index % kLargeCube, bit_depth);
  }
  return palette[c * onerow + static_cast<size_t>(index)];
}

}  // namespace palette_internal

Status InvPalette(Image &input, uint32_t begin_c, uint32_t nb_colors,
                  uint32_t nb_deltas, Predictor predictor,
                  const weighted::Header &wp_header, ThreadPool *pool);

Status MetaPalette(Image &input, uint32_t begin_c, uint32_t end_c,
                   uint32_t nb_colors, uint32_t nb_deltas, bool lossy);

}  // namespace jxl

#endif  // LIB_JXL_MODULAR_TRANSFORM_PALETTE_H_
