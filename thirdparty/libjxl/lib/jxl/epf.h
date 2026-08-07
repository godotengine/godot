// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_EPF_H_
#define LIB_JXL_EPF_H_

// Fast SIMD "in-loop" edge preserving filter (adaptive, nonlinear).

#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_cache.h"
#include "lib/jxl/loop_filter.h"

namespace jxl {

// 4 * (sqrt(0.5)-1), so that Weight(sigma) = 0.5.
static constexpr float kInvSigmaNum = -1.1715728752538099024f;

// kInvSigmaNum / 0.3
constexpr float kMinSigma = -3.90524291751269967465540850526868f;

// Fills the `state->filter_weights.sigma` image with the precomputed sigma
// values in the area inside `block_rect`. Accesses the AC strategy, quant field
// and epf_sharpness fields in the corresponding positions.
Status ComputeSigma(const LoopFilter& lf, const Rect& block_rect,
                    PassesDecoderState* state);

}  // namespace jxl

#endif  // LIB_JXL_EPF_H_
