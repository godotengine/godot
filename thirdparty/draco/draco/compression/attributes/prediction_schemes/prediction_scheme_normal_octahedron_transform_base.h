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
#ifndef DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_PREDICTION_SCHEME_NORMAL_OCTAHEDRON_TRANSFORM_BASE_H_
#define DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_PREDICTION_SCHEME_NORMAL_OCTAHEDRON_TRANSFORM_BASE_H_

#include <cmath>

#include "draco/compression/attributes/normal_compression_utils.h"
#include "draco/compression/config/compression_shared.h"
#include "draco/core/bit_utils.h"
#include "draco/core/macros.h"
#include "draco/core/vector_d.h"

namespace draco {

// Base class containing shared functionality used by both encoding and decoding
// octahedral normal prediction scheme transforms. See the encoding transform
// for more details about the method.
template <typename DataTypeT>
class PredictionSchemeNormalOctahedronTransformBase {
 public:
  typedef VectorD<DataTypeT, 2> Point2;
  typedef DataTypeT DataType;

  PredictionSchemeNormalOctahedronTransformBase() {}
  // We expect the mod value to be of the form 2^b-1.
  explicit PredictionSchemeNormalOctahedronTransformBase(
      DataType max_quantized_value) {
    this->set_max_quantized_value(max_quantized_value);
  }

  static constexpr PredictionSchemeTransformType GetType() {
    return PREDICTION_TRANSFORM_NORMAL_OCTAHEDRON;
  }

  // We can return true as we keep correction values positive.
  bool AreCorrectionsPositive() const { return true; }

  inline DataTypeT max_quantized_value() const {
    return octahedron_tool_box_.max_quantized_value();
  }
  inline DataTypeT center_value() const {
    return octahedron_tool_box_.center_value();
  }
  inline int32_t quantization_bits() const {
    return octahedron_tool_box_.quantization_bits();
  }

 protected:
  inline bool set_max_quantized_value(DataTypeT max_quantized_value) {
    if (max_quantized_value % 2 == 0) {
      return false;
    }
    int q = MostSignificantBit(max_quantized_value) + 1;
    return octahedron_tool_box_.SetQuantizationBits(q);
  }

  bool IsInDiamond(DataTypeT s, DataTypeT t) const {
    return octahedron_tool_box_.IsInDiamond(s, t);
  }
  void InvertDiamond(DataTypeT *s, DataTypeT *t) const {
    return octahedron_tool_box_.InvertDiamond(s, t);
  }

  int32_t ModMax(int32_t x) const { return octahedron_tool_box_.ModMax(x); }

  // For correction values.
  int32_t MakePositive(int32_t x) const {
    return octahedron_tool_box_.MakePositive(x);
  }

 private:
  OctahedronToolBox octahedron_tool_box_;
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_PREDICTION_SCHEME_NORMAL_OCTAHEDRON_TRANSFORM_BASE_H_
