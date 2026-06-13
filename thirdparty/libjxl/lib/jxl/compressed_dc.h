// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_COMPRESSED_DC_H_
#define LIB_JXL_COMPRESSED_DC_H_

#include <jxl/memory_manager.h>

#include "lib/jxl/ac_context.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/image.h"
#include "lib/jxl/modular/modular_image.h"

// DC handling functions: encoding and decoding of DC to and from bitstream, and
// related function to initialize the per-group decoder cache.

namespace jxl {

// Smooth DC in already-smooth areas, to counteract banding.
Status AdaptiveDCSmoothing(JxlMemoryManager* memory_manager,
                           const float* dc_factors, Image3F* dc,
                           ThreadPool* pool);

void DequantDC(const Rect& r, Image3F* dc, ImageB* quant_dc, const Image& in,
               const float* dc_factors, float mul, const float* cfl_factors,
               const YCbCrChromaSubsampling& chroma_subsampling,
               const BlockCtxMap& bctx);

}  // namespace jxl

#endif  // LIB_JXL_COMPRESSED_DC_H_
