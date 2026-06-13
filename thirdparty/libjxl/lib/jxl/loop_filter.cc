// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/loop_filter.h"

#include <cmath>

#include "lib/jxl/base/status.h"
#include "lib/jxl/fields.h"

namespace jxl {

LoopFilter::LoopFilter() { Bundle::Init(this); }
Status LoopFilter::VisitFields(Visitor* JXL_RESTRICT visitor) {
  // Must come before AllDefault.

  if (visitor->AllDefault(*this, &all_default)) {
    // Overwrite all serialized fields, but not any nonserialized_*.
    visitor->SetDefault(this);
    return true;
  }

  JXL_QUIET_RETURN_IF_ERROR(visitor->Bool(true, &gab));
  if (visitor->Conditional(gab)) {
    JXL_QUIET_RETURN_IF_ERROR(visitor->Bool(false, &gab_custom));
    if (visitor->Conditional(gab_custom)) {
      JXL_QUIET_RETURN_IF_ERROR(
          visitor->F16(1.1 * 0.104699568f, &gab_x_weight1));
      JXL_QUIET_RETURN_IF_ERROR(
          visitor->F16(1.1 * 0.055680538f, &gab_x_weight2));
      if (std::abs(1.0f + (gab_x_weight1 + gab_x_weight2) * 4) < 1e-8) {
        return JXL_FAILURE(
            "Gaborish x weights lead to near 0 unnormalized kernel");
      }
      JXL_QUIET_RETURN_IF_ERROR(
          visitor->F16(1.1 * 0.104699568f, &gab_y_weight1));
      JXL_QUIET_RETURN_IF_ERROR(
          visitor->F16(1.1 * 0.055680538f, &gab_y_weight2));
      if (std::abs(1.0f + (gab_y_weight1 + gab_y_weight2) * 4) < 1e-8) {
        return JXL_FAILURE(
            "Gaborish y weights lead to near 0 unnormalized kernel");
      }
      JXL_QUIET_RETURN_IF_ERROR(
          visitor->F16(1.1 * 0.104699568f, &gab_b_weight1));
      JXL_QUIET_RETURN_IF_ERROR(
          visitor->F16(1.1 * 0.055680538f, &gab_b_weight2));
      if (std::abs(1.0f + (gab_b_weight1 + gab_b_weight2) * 4) < 1e-8) {
        return JXL_FAILURE(
            "Gaborish b weights lead to near 0 unnormalized kernel");
      }
    }
  }

  JXL_QUIET_RETURN_IF_ERROR(visitor->Bits(2, 2, &epf_iters));
  if (visitor->Conditional(epf_iters > 0)) {
    if (visitor->Conditional(!nonserialized_is_modular)) {
      JXL_QUIET_RETURN_IF_ERROR(visitor->Bool(false, &epf_sharp_custom));
      if (visitor->Conditional(epf_sharp_custom)) {
        for (size_t i = 0; i < kEpfSharpEntries; ++i) {
          JXL_QUIET_RETURN_IF_ERROR(visitor->F16(
              float(i) / float(kEpfSharpEntries - 1), &epf_sharp_lut[i]));
        }
      }
    }

    JXL_QUIET_RETURN_IF_ERROR(visitor->Bool(false, &epf_weight_custom));
    if (visitor->Conditional(epf_weight_custom)) {
      JXL_QUIET_RETURN_IF_ERROR(visitor->F16(40.0f, &epf_channel_scale[0]));
      JXL_QUIET_RETURN_IF_ERROR(visitor->F16(5.0f, &epf_channel_scale[1]));
      JXL_QUIET_RETURN_IF_ERROR(visitor->F16(3.5f, &epf_channel_scale[2]));
      JXL_QUIET_RETURN_IF_ERROR(visitor->F16(0.45f, &epf_pass1_zeroflush));
      JXL_QUIET_RETURN_IF_ERROR(visitor->F16(0.6f, &epf_pass2_zeroflush));
    }

    JXL_QUIET_RETURN_IF_ERROR(visitor->Bool(false, &epf_sigma_custom));
    if (visitor->Conditional(epf_sigma_custom)) {
      if (visitor->Conditional(!nonserialized_is_modular)) {
        JXL_QUIET_RETURN_IF_ERROR(visitor->F16(0.46f, &epf_quant_mul));
      }
      JXL_QUIET_RETURN_IF_ERROR(visitor->F16(0.9f, &epf_pass0_sigma_scale));
      JXL_QUIET_RETURN_IF_ERROR(visitor->F16(6.5f, &epf_pass2_sigma_scale));
      JXL_QUIET_RETURN_IF_ERROR(
          visitor->F16(0.6666666666666666f, &epf_border_sad_mul));
    }
    if (visitor->Conditional(nonserialized_is_modular)) {
      JXL_QUIET_RETURN_IF_ERROR(visitor->F16(1.0f, &epf_sigma_for_modular));
      if (epf_sigma_for_modular < 1e-8) {
        return JXL_FAILURE("EPF: sigma for modular is too small");
      }
    }
  }

  JXL_QUIET_RETURN_IF_ERROR(visitor->BeginExtensions(&extensions));
  // Extensions: in chronological order of being added to the format.
  return visitor->EndExtensions();
}

}  // namespace jxl
