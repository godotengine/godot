// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Functions for clustering similar histograms together.

#ifndef LIB_JXL_ENC_CLUSTER_H_
#define LIB_JXL_ENC_CLUSTER_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "lib/jxl/base/status.h"
#include "lib/jxl/enc_ans_params.h"

namespace jxl {

Status ClusterHistograms(const HistogramParams& params,
                         const std::vector<Histogram>& in,
                         size_t max_histograms, std::vector<Histogram>* out,
                         std::vector<uint32_t>* histogram_symbols);
}  // namespace jxl

#endif  // LIB_JXL_ENC_CLUSTER_H_
