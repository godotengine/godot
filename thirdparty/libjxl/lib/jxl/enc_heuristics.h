// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_HEURISTICS_H_
#define LIB_JXL_ENC_HEURISTICS_H_

// Hook for custom encoder heuristics (VarDCT only for now).

#include <jxl/cms_interface.h>

#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/image.h"

namespace jxl {

struct AuxOut;
struct PassesEncoderState;
class DequantMatrices;
class ImageBundle;
class ModularFrameEncoder;

// Initializes encoder structures in `enc_state` using the original image data
// in `original_pixels`, and the XYB image data in `opsin`. Also modifies the
// `opsin` image by applying Gaborish, and doing other modifications if
// necessary. `pool` is used for running the computations on multiple threads.
// `aux_out` collects statistics and can be used to print debug images.
Status LossyFrameHeuristics(const FrameHeader& frame_header,
                            PassesEncoderState* enc_state,
                            ModularFrameEncoder* modular_frame_encoder,
                            const Image3F* linear, Image3F* opsin,
                            const Rect& rect, const JxlCmsInterface& cms,
                            ThreadPool* pool, AuxOut* aux_out);

Status ComputeARHeuristics(const FrameHeader& frame_header,
                           PassesEncoderState* enc_state,
                           const Image3F& orig_opsin, const Rect& rect,
                           ThreadPool* pool);

void FindBestBlockEntropyModel(PassesEncoderState& enc_state);

Status DownsampleImage2_Iterative(Image3F* opsin);
Status DownsampleImage2_Sharper(Image3F* opsin);

}  // namespace jxl

#endif  // LIB_JXL_ENC_HEURISTICS_H_
