// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_DOT_DICTIONARY_H_
#define LIB_JXL_ENC_DOT_DICTIONARY_H_

// Dots are stored in a dictionary to avoid storing similar dots multiple
// times.

#include <vector>

#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/chroma_from_luma.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/enc_patch_dictionary.h"
#include "lib/jxl/image.h"

namespace jxl {

StatusOr<std::vector<PatchInfo>> FindDotDictionary(
    const CompressParams& cparams, const Image3F& opsin, const Rect& rect,
    const ColorCorrelation& color_correlation, ThreadPool* pool);

}  // namespace jxl

#endif  // LIB_JXL_ENC_DOT_DICTIONARY_H_
