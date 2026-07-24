// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_GABORISH_H_
#define LIB_JXL_GABORISH_H_

// Linear smoothing (3x3 convolution) for deblocking without too much blur.

#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/image.h"

namespace jxl {

// Used in encoder to reduce the impact of the decoder's smoothing.
// This is not exact. Works in-place to reduce memory use.
// The input is typically in XYB space.
Status GaborishInverse(Image3F* in_out, const Rect& rect, const float mul[3],
                       ThreadPool* pool);

}  // namespace jxl

#endif  // LIB_JXL_GABORISH_H_
