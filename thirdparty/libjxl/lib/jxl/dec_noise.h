// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_DEC_NOISE_H_
#define LIB_JXL_DEC_NOISE_H_

// Noise synthesis. Currently disabled.

#include <cstddef>

#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/dec_cache.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/noise.h"

namespace jxl {

void PrepareNoiseInput(const PassesDecoderState& dec_state,
                       const FrameDimensions& frame_dim,
                       const FrameHeader& frame_header, size_t group_index,
                       size_t thread);

// Must only call if FrameHeader.flags.kNoise.
Status DecodeNoise(BitReader* br, NoiseParams* noise_params);

}  // namespace jxl

#endif  // LIB_JXL_DEC_NOISE_H_
