// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "lib/jxl/base/status.h"
#include "lib/jxl/enc_ans.h"
#include "lib/jxl/enc_ans_params.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/image.h"
#include "lib/jxl/pack_signed.h"
#include "lib/jxl/splines.h"

namespace jxl {

struct AuxOut;
enum class LayerType : uint8_t;

class QuantizedSplineEncoder {
 public:
  // Only call if HasAny().
  static void Tokenize(const QuantizedSpline& spline,
                       std::vector<Token>* const tokens) {
    tokens->emplace_back(static_cast<uint32_t>(kNumControlPointsContext),
                         static_cast<uint32_t>(spline.control_points_.size()));
    for (const auto& point : spline.control_points_) {
      tokens->emplace_back(kControlPointsContext, PackSigned(point.first));
      tokens->emplace_back(kControlPointsContext, PackSigned(point.second));
    }
    const auto encode_dct = [tokens](const int dct[32]) {
      for (int i = 0; i < 32; ++i) {
        tokens->emplace_back(kDCTContext, PackSigned(dct[i]));
      }
    };
    for (const auto& dct : spline.color_dct_) {
      encode_dct(dct);
    }
    encode_dct(spline.sigma_dct_);
  }
};

namespace {

void EncodeAllStartingPoints(Span<const Spline::Point> points,
                             std::vector<Token>* tokens) {
  int64_t last_x = 0;
  int64_t last_y = 0;
  for (size_t i = 0; i < points.size(); i++) {
    const int64_t x = std::lround(points[i].x);
    const int64_t y = std::lround(points[i].y);
    if (i == 0) {
      tokens->emplace_back(static_cast<uint32_t>(kStartingPositionContext),
                           static_cast<uint32_t>(x));
      tokens->emplace_back(static_cast<uint32_t>(kStartingPositionContext),
                           static_cast<uint32_t>(y));
    } else {
      tokens->emplace_back(static_cast<uint32_t>(kStartingPositionContext),
                           PackSigned(x - last_x));
      tokens->emplace_back(static_cast<uint32_t>(kStartingPositionContext),
                           PackSigned(y - last_y));
    }
    last_x = x;
    last_y = y;
  }
}

}  // namespace

Status EncodeSplines(const Splines& splines, BitWriter* writer,
                     const LayerType layer,
                     const HistogramParams& histogram_params, AuxOut* aux_out) {
  JXL_ENSURE(splines.HasAny());

  Span<const QuantizedSpline> quantized_splines = splines.QuantizedSplines();
  std::vector<std::vector<Token>> tokens(1);
  tokens[0].emplace_back(static_cast<uint32_t>(kNumSplinesContext),
                         static_cast<uint32_t>(quantized_splines.size() - 1));
  EncodeAllStartingPoints(splines.StartingPoints(), tokens.data());

  tokens[0].emplace_back(kQuantizationAdjustmentContext,
                         PackSigned(splines.GetQuantizationAdjustment()));

  for (const QuantizedSpline& spline : quantized_splines) {
    QuantizedSplineEncoder::Tokenize(spline, tokens.data());
  }

  EntropyEncodingData codes;
  JXL_ASSIGN_OR_RETURN(
      size_t cost,
      BuildAndEncodeHistograms(writer->memory_manager(), histogram_params,
                               kNumSplineContexts, tokens, &codes, writer,
                               layer, aux_out));
  (void)cost;
  JXL_RETURN_IF_ERROR(WriteTokens(tokens[0], codes, 0, writer, layer, aux_out));
  return true;
}

Splines FindSplines(const Image3F& opsin) {
  // TODO(user): implement spline detection.
  return Splines{opsin.memory_manager()};
}

}  // namespace jxl
