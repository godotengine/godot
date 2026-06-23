// Copyright 2016 The Draco Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// File provides shared functions for adaptive rANS bit coding.
#ifndef DRACO_COMPRESSION_BIT_CODERS_ADAPTIVE_RANS_BIT_CODING_SHARED_H_
#define DRACO_COMPRESSION_BIT_CODERS_ADAPTIVE_RANS_BIT_CODING_SHARED_H_

#include "draco/core/macros.h"

namespace draco {

// Clamp the probability p to a uint8_t in the range [1,255].
inline uint8_t clamp_probability(double p) {
  DRACO_DCHECK_LE(p, 1.0);
  DRACO_DCHECK_LE(0.0, p);
  uint32_t p_int = static_cast<uint32_t>((p * 256) + 0.5);
  p_int -= (p_int == 256);
  p_int += (p_int == 0);
  return static_cast<uint8_t>(p_int);
}

// Update the probability according to new incoming bit.
inline double update_probability(double old_p, bool bit) {
  static constexpr double w = 128.0;
  static constexpr double w0 = (w - 1.0) / w;
  static constexpr double w1 = 1.0 / w;
  return old_p * w0 + (!bit) * w1;
}

}  // namespace draco

#endif  // DRACO_COMPRESSION_BIT_CODERS_ADAPTIVE_RANS_BIT_CODING_SHARED_H_
