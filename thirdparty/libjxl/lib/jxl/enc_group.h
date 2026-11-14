// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_GROUP_H_
#define LIB_JXL_ENC_GROUP_H_

#include <cstddef>

#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/image.h"

namespace jxl {

struct AuxOut;
struct PassesEncoderState;

// Fills DC
Status ComputeCoefficients(size_t group_idx, PassesEncoderState* enc_state,
                           const Image3F& opsin, const Rect& rect, Image3F* dc);

Status EncodeGroupTokenizedCoefficients(size_t group_idx, size_t pass_idx,
                                        size_t histogram_idx,
                                        const PassesEncoderState& enc_state,
                                        BitWriter* writer, AuxOut* aux_out);

}  // namespace jxl

#endif  // LIB_JXL_ENC_GROUP_H_
