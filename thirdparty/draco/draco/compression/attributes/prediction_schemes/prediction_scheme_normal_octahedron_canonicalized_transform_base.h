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
#ifndef DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_PREDICTION_SCHEME_NORMAL_OCTAHEDRON_CANONICALIZED_TRANSFORM_BASE_H_
#define DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_PREDICTION_SCHEME_NORMAL_OCTAHEDRON_CANONICALIZED_TRANSFORM_BASE_H_

#include <cmath>

#include "draco/compression/attributes/normal_compression_utils.h"
#include "draco/compression/attributes/prediction_schemes/prediction_scheme_normal_octahedron_transform_base.h"
#include "draco/compression/config/compression_shared.h"
#include "draco/core/bit_utils.h"
#include "draco/core/macros.h"
#include "draco/core/vector_d.h"

namespace draco {

// Base class containing shared functionality used by both encoding and decoding
// canonicalized normal octahedron prediction scheme transforms. See the
// encoding transform for more details about the method.
template <typename DataTypeT>
class PredictionSchemeNormalOctahedronCanonicalizedTransformBase
    : public PredictionSchemeNormalOctahedronTransformBase<DataTypeT> {
 public:
  typedef PredictionSchemeNormalOctahedronTransformBase<DataTypeT> Base;
  typedef VectorD<DataTypeT, 2> Point2;
  typedef DataTypeT DataType;

  PredictionSchemeNormalOctahedronCanonicalizedTransformBase() : Base() {}
  // We expect the mod value to be of the form 2^b-1.
  explicit PredictionSchemeNormalOctahedronCanonicalizedTransformBase(
      DataType mod_value)
      : Base(mod_value) {}

  static constexpr PredictionSchemeTransformType GetType() {
    return PREDICTION_TRANSFORM_NORMAL_OCTAHEDRON_CANONICALIZED;
  }

  int32_t GetRotationCount(Point2 pred) const {
    const DataType sign_x = pred[0];
    const DataType sign_y = pred[1];

    int32_t rotation_count = 0;
    if (sign_x == 0) {
      if (sign_y == 0) {
        rotation_count = 0;
      } else if (sign_y > 0) {
        rotation_count = 3;
      } else {
        rotation_count = 1;
      }
    } else if (sign_x > 0) {
      if (sign_y >= 0) {
        rotation_count = 2;
      } else {
        rotation_count = 1;
      }
    } else {
      if (sign_y <= 0) {
        rotation_count = 0;
      } else {
        rotation_count = 3;
      }
    }
    return rotation_count;
  }

  Point2 RotatePoint(Point2 p, int32_t rotation_count) const {
    switch (rotation_count) {
      case 1:
        return Point2(p[1], -p[0]);
      case 2:
        return Point2(-p[0], -p[1]);
      case 3:
        return Point2(-p[1], p[0]);
      default:
        return p;
    }
  }

  bool IsInBottomLeft(const Point2 &p) const {
    if (p[0] == 0 && p[1] == 0) {
      return true;
    }
    return (p[0] < 0 && p[1] <= 0);
  }
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_PREDICTION_SCHEME_NORMAL_OCTAHEDRON_CANONICALIZED_TRANSFORM_BASE_H_
