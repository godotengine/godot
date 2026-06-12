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
#ifndef DRACO_COMPRESSION_POINT_CLOUD_ALGORITHMS_POINT_CLOUD_TYPES_H_
#define DRACO_COMPRESSION_POINT_CLOUD_ALGORITHMS_POINT_CLOUD_TYPES_H_

#include <inttypes.h>

#include <vector>

#include "draco/core/vector_d.h"

namespace draco {

// Using Eigen as this is favored by project Cartographer.
typedef Vector3f Point3f;
typedef Vector4f Point4f;
typedef Vector3ui Point3ui;
typedef Vector4ui Point4ui;
typedef Vector5ui Point5ui;
typedef Vector6ui Point6ui;
typedef Vector7ui Point7ui;

typedef std::vector<Point3f> PointCloud3f;

template <class PointDT>
struct PointDLess;

template <class CoeffT, int dimension_t>
struct PointDLess<VectorD<CoeffT, dimension_t>> {
  bool operator()(const VectorD<CoeffT, dimension_t> &a,
                  const VectorD<CoeffT, dimension_t> &b) const {
    return a < b;
  }
};

template <class PointDT>
class PointTraits {};

template <class CoordinateTypeT, int dimension_t>
class PointTraits<VectorD<CoordinateTypeT, dimension_t>> {
 public:
  typedef VectorD<CoordinateTypeT, dimension_t> PointD;
  typedef CoordinateTypeT CoordinateType;

  static constexpr uint32_t Dimension() { return dimension_t; }
  static PointD Origin() {
    PointD origin;
    for (uint32_t i = 0; i < dimension_t; i++) {
      origin(i) = 0;
    }
    return origin;
  }
  static std::array<uint32_t, dimension_t> ZeroArray() {
    std::array<uint32_t, dimension_t> zero;
    for (uint32_t i = 0; i < dimension_t; i++) {
      zero[i] = 0;
    }
    return zero;
  }
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_POINT_CLOUD_ALGORITHMS_POINT_CLOUD_TYPES_H_
