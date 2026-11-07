// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_dot_dictionary.h"

#include <array>
#include <cstddef>
#include <cstring>
#include <vector>

#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/override.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/chroma_from_luma.h"
#include "lib/jxl/enc_detect_dots.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/enc_patch_dictionary.h"
#include "lib/jxl/image.h"

namespace jxl {

// Private implementation of Dictionary Encode/Decode
namespace {

/* Quantization constants for Ellipse dots */
const size_t kEllipsePosQ = 2;        // Quantization level for the position
const double kEllipseMinSigma = 0.1;  // Minimum sigma value
const double kEllipseMaxSigma = 3.1;  // Maximum Sigma value
const size_t kEllipseSigmaQ = 16;     // Number of quantization levels for sigma
const size_t kEllipseAngleQ = 8;      // Quantization level for the angle
// TODO(user): fix these values.
const std::array<double, 3> kEllipseMinIntensity{{-0.05, 0.0, -0.5}};
const std::array<double, 3> kEllipseMaxIntensity{{0.05, 1.0, 0.4}};
const std::array<size_t, 3> kEllipseIntensityQ{{10, 36, 10}};
}  // namespace

StatusOr<std::vector<PatchInfo>> FindDotDictionary(
    const CompressParams& cparams, const Image3F& opsin, const Rect& rect,
    const ColorCorrelation& color_correlation, ThreadPool* pool) {
  if (ApplyOverride(cparams.dots,
                    cparams.butteraugli_distance >= kMinButteraugliForDots)) {
    GaussianDetectParams ellipse_params;
    ellipse_params.t_high = 0.04;
    ellipse_params.t_low = 0.02;
    ellipse_params.maxWinSize = 5;
    ellipse_params.maxL2Loss = 0.005;
    ellipse_params.maxCustomLoss = 300;
    ellipse_params.minIntensity = 0.12;
    ellipse_params.maxDistMeanMode = 1.0;
    ellipse_params.maxNegPixels = 0;
    ellipse_params.minScore = 12.0;
    ellipse_params.maxCC = 100;
    ellipse_params.percCC = 100;
    EllipseQuantParams qParams{rect.xsize(),
                               rect.ysize(),
                               kEllipsePosQ,
                               kEllipseMinSigma,
                               kEllipseMaxSigma,
                               kEllipseSigmaQ,
                               kEllipseAngleQ,
                               kEllipseMinIntensity,
                               kEllipseMaxIntensity,
                               kEllipseIntensityQ,
                               kEllipsePosQ <= 5,
                               color_correlation.YtoXRatio(0),
                               color_correlation.YtoBRatio(0)};

    return DetectGaussianEllipses(opsin, rect, ellipse_params, qParams, pool);
  }
  std::vector<PatchInfo> nothing;
  return nothing;
}
}  // namespace jxl
