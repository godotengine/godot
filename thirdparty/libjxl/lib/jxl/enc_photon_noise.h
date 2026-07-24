// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_PHOTON_NOISE_H_
#define LIB_JXL_ENC_PHOTON_NOISE_H_

#include <cstddef>

#include "lib/jxl/noise.h"

namespace jxl {

// Constructs a NoiseParams representing the noise that would be seen at the
// selected nominal exposure on a last-decade (as of 2021) color camera with a
// 36×24mm sensor (“35mm format”).
NoiseParams SimulatePhotonNoise(size_t xsize, size_t ysize, float iso);

}  // namespace jxl

#endif  // LIB_JXL_ENC_PHOTON_NOISE_H_
