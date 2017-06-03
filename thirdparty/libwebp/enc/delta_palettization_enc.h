// Copyright 2015 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Author: Mislav Bradac (mislavm@google.com)
//

#ifndef WEBP_ENC_DELTA_PALETTIZATION_H_
#define WEBP_ENC_DELTA_PALETTIZATION_H_

#include "../webp/encode.h"
#include "../enc/vp8li_enc.h"

// Replaces enc->argb_[] input by a palettizable approximation of it,
// and generates optimal enc->palette_[].
// This function can revert enc->use_palette_ / enc->use_predict_ flag
// if delta-palettization is not producing expected saving.
WebPEncodingError WebPSearchOptimalDeltaPalette(VP8LEncoder* const enc);

#endif  // WEBP_ENC_DELTA_PALETTIZATION_H_
