// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_LOOP_FILTER_H_
#define LIB_JXL_LOOP_FILTER_H_

// Parameters for loop filter(s), stored in each frame.

#include <cstddef>
#include <cstdint>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/field_encodings.h"

namespace jxl {

struct LoopFilter : public Fields {
  LoopFilter();
  JXL_FIELDS_NAME(LoopFilter)

  Status VisitFields(Visitor* JXL_RESTRICT visitor) override;

  size_t Padding() const {
    static const size_t padding_per_epf_iter[4] = {0, 2, 3, 6};
    return padding_per_epf_iter[epf_iters] + (gab ? 1 : 0);
  }

  mutable bool all_default;

  // --- Gaborish convolution
  bool gab;

  bool gab_custom;
  float gab_x_weight1;
  float gab_x_weight2;
  float gab_y_weight1;
  float gab_y_weight2;
  float gab_b_weight1;
  float gab_b_weight2;

  // --- Edge-preserving filter

  // Number of EPF stages to apply. 0 means EPF disabled. 1 applies only the
  // first stage, 2 applies both stages and 3 applies the first stage twice and
  // the second stage once.
  uint32_t epf_iters;

  bool epf_sharp_custom;
  enum { kEpfSharpEntries = 8 };
  float epf_sharp_lut[kEpfSharpEntries];

  bool epf_weight_custom;      // Custom weight params
  float epf_channel_scale[3];  // Relative weight of each channel
  float epf_pass1_zeroflush;   // Minimum weight for first pass
  float epf_pass2_zeroflush;   // Minimum weight for second pass

  bool epf_sigma_custom;        // Custom sigma parameters
  float epf_quant_mul;          // Sigma is ~ this * quant
  float epf_pass0_sigma_scale;  // Multiplier for sigma in pass 0
  float epf_pass2_sigma_scale;  // Multiplier for sigma in the second pass
  float epf_border_sad_mul;     // (inverse) multiplier for sigma on borders

  float epf_sigma_for_modular;

  uint64_t extensions;

  bool nonserialized_is_modular = false;
};

}  // namespace jxl

#endif  // LIB_JXL_LOOP_FILTER_H_
