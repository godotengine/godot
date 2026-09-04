// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_PROGRESSIVE_SPLIT_H_
#define LIB_JXL_PROGRESSIVE_SPLIT_H_

#include <cstddef>
#include <cstdint>
#include <limits>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/common.h"  // kMaxNumPasses
#include "lib/jxl/frame_header.h"

// Functions to split DCT coefficients in multiple passes. All the passes of a
// single frame are added together.

namespace jxl {

class AcStrategy;

constexpr size_t kNoDownsamplingFactor = std::numeric_limits<size_t>::max();

struct PassDefinition {
  // Side of the square of the coefficients that should be kept in each 8x8
  // block. Must be greater than 1, and at most 8. Should be in non-decreasing
  // order.
  size_t num_coefficients;

  // How much to shift the encoded values by, with rounding.
  size_t shift;

  // If specified, this indicates that if the requested downsampling factor is
  // sufficiently high, then it is fine to stop decoding after this pass.
  // By default, passes are not marked as being suitable for any downsampling.
  size_t suitable_for_downsampling_of_at_least;
};

struct ProgressiveMode {
  size_t num_passes = 1;
  PassDefinition passes[kMaxNumPasses] = {
      PassDefinition{/*num_coefficients=*/8, /*shift=*/0,
                     /*suitable_for_downsampling_of_at_least=*/1}};

  ProgressiveMode() = default;

  template <size_t nump>
  explicit ProgressiveMode(const PassDefinition (&p)[nump]) {
    static_assert(nump <= kMaxNumPasses);
    num_passes = nump;
    PassDefinition previous_pass{
        /*num_coefficients=*/1, /*shift=*/0,
        /*suitable_for_downsampling_of_at_least=*/kNoDownsamplingFactor};
    size_t last_downsampling_factor = kNoDownsamplingFactor;
    for (size_t i = 0; i < nump; i++) {
      JXL_DASSERT(p[i].num_coefficients > previous_pass.num_coefficients ||
                  (p[i].num_coefficients == previous_pass.num_coefficients &&
                   p[i].shift < previous_pass.shift));
      JXL_DASSERT(p[i].suitable_for_downsampling_of_at_least ==
                      kNoDownsamplingFactor ||
                  p[i].suitable_for_downsampling_of_at_least <=
                      last_downsampling_factor);
      // Only used inside assert.
      (void)last_downsampling_factor;
      if (p[i].suitable_for_downsampling_of_at_least != kNoDownsamplingFactor) {
        last_downsampling_factor = p[i].suitable_for_downsampling_of_at_least;
      }
      previous_pass = passes[i] = p[i];
    }
  }
};

class ProgressiveSplitter {
 public:
  void SetProgressiveMode(ProgressiveMode mode) { mode_ = mode; }

  size_t GetNumPasses() const { return mode_.num_passes; }

  Status InitPasses(Passes* JXL_RESTRICT passes) const {
    passes->num_passes = static_cast<uint32_t>(GetNumPasses());
    passes->num_downsample = 0;
    JXL_ENSURE(passes->num_passes != 0);
    passes->shift[passes->num_passes - 1] = 0;
    if (passes->num_passes == 1) return true;  // Done, arrays are empty

    for (uint32_t i = 0; i < mode_.num_passes - 1; ++i) {
      const size_t min_downsampling_factor =
          mode_.passes[i].suitable_for_downsampling_of_at_least;
      passes->shift[i] = mode_.passes[i].shift;
      if (1 < min_downsampling_factor &&
          min_downsampling_factor != kNoDownsamplingFactor) {
        passes->downsample[passes->num_downsample] = min_downsampling_factor;
        passes->last_pass[passes->num_downsample] = i;
        if (mode_.passes[i + 1].suitable_for_downsampling_of_at_least <
            min_downsampling_factor) {
          passes->num_downsample += 1;
        }
      }
    }
    return true;
  }

  template <typename T>
  void SplitACCoefficients(const T* JXL_RESTRICT block, const AcStrategy& acs,
                           size_t bx, size_t by,
                           T* JXL_RESTRICT output[kMaxNumPasses]);

 private:
  ProgressiveMode mode_;
};

extern template void ProgressiveSplitter::SplitACCoefficients<int32_t>(
    const int32_t* JXL_RESTRICT, const AcStrategy&, size_t, size_t,
    int32_t* JXL_RESTRICT[kMaxNumPasses]);

extern template void ProgressiveSplitter::SplitACCoefficients<int16_t>(
    const int16_t* JXL_RESTRICT, const AcStrategy&, size_t, size_t,
    int16_t* JXL_RESTRICT[kMaxNumPasses]);

}  // namespace jxl

#endif  // LIB_JXL_PROGRESSIVE_SPLIT_H_
