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
#ifndef DRACO_COMPRESSION_POINT_CLOUD_ALGORITHMS_QUANTIZE_POINTS_3_H_
#define DRACO_COMPRESSION_POINT_CLOUD_ALGORITHMS_QUANTIZE_POINTS_3_H_

#include <inttypes.h>

#include "draco/compression/point_cloud/algorithms/point_cloud_types.h"
#include "draco/core/quantization_utils.h"

namespace draco {

// TODO(b/199760123): Make this a stable bounding box.
struct QuantizationInfo {
  uint32_t quantization_bits;
  float range;
};

template <class PointIterator, class OutputIterator>
OutputIterator QuantizePoints3(const PointIterator &begin,
                               const PointIterator &end, QuantizationInfo *info,
                               OutputIterator oit) {
  DRACO_DCHECK_GE(info->quantization_bits, 0);

  float max_range = 0;
  for (auto it = begin; it != end; ++it) {
    max_range = std::max(std::fabs((*it)[0]), max_range);
    max_range = std::max(std::fabs((*it)[1]), max_range);
    max_range = std::max(std::fabs((*it)[2]), max_range);
  }

  const uint32_t max_quantized_value((1u << info->quantization_bits) - 1);
  Quantizer quantize;
  quantize.Init(max_range, max_quantized_value);
  info->range = max_range;

  Point3ui qpoint;
  for (auto it = begin; it != end; ++it) {
    // Quantize and all positive.
    qpoint[0] = quantize((*it)[0]) + max_quantized_value;
    qpoint[1] = quantize((*it)[1]) + max_quantized_value;
    qpoint[2] = quantize((*it)[2]) + max_quantized_value;
    *oit++ = (qpoint);
  }

  return oit;
}

template <class QPointIterator, class OutputIterator>
void DequantizePoints3(const QPointIterator &begin, const QPointIterator &end,
                       const QuantizationInfo &info, OutputIterator &oit) {
  DRACO_DCHECK_GE(info.quantization_bits, 0);
  DRACO_DCHECK_GE(info.range, 0);

  const uint32_t quantization_bits = info.quantization_bits;
  const float range = info.range;
  const uint32_t max_quantized_value((1u << quantization_bits) - 1);
  Dequantizer dequantize;
  dequantize.Init(range, max_quantized_value);

  for (auto it = begin; it != end; ++it) {
    const float x = dequantize((*it)[0] - max_quantized_value);
    const float y = dequantize((*it)[1] - max_quantized_value);
    const float z = dequantize((*it)[2] - max_quantized_value);
    *oit = Point3f(x, y, z);
    ++oit;
  }
}

}  // namespace draco

#endif  // DRACO_COMPRESSION_POINT_CLOUD_ALGORITHMS_QUANTIZE_POINTS_3_H_
