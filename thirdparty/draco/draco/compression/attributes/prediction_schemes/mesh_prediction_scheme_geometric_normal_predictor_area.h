// Copyright 2017 The Draco Authors.
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
#ifndef DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_MESH_PREDICTION_SCHEME_GEOMETRIC_NORMAL_PREDICTOR_AREA_H_
#define DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_MESH_PREDICTION_SCHEME_GEOMETRIC_NORMAL_PREDICTOR_AREA_H_

#include "draco/compression/attributes/prediction_schemes/mesh_prediction_scheme_geometric_normal_predictor_base.h"

namespace draco {

// This predictor estimates the normal via the surrounding triangles of the
// given corner. Triangles are weighted according to their area.
template <typename DataTypeT, class TransformT, class MeshDataT>
class MeshPredictionSchemeGeometricNormalPredictorArea
    : public MeshPredictionSchemeGeometricNormalPredictorBase<
          DataTypeT, TransformT, MeshDataT> {
  typedef MeshPredictionSchemeGeometricNormalPredictorBase<
      DataTypeT, TransformT, MeshDataT>
      Base;

 public:
  explicit MeshPredictionSchemeGeometricNormalPredictorArea(const MeshDataT &md)
      : Base(md) {
    this->SetNormalPredictionMode(TRIANGLE_AREA);
  };
  virtual ~MeshPredictionSchemeGeometricNormalPredictorArea() {}

  // Computes predicted octahedral coordinates on a given corner.
  void ComputePredictedValue(CornerIndex corner_id,
                             DataTypeT *prediction) override {
    DRACO_DCHECK(this->IsInitialized());
    typedef typename MeshDataT::CornerTable CornerTable;
    const CornerTable *const corner_table = this->mesh_data_.corner_table();
    // Going to compute the predicted normal from the surrounding triangles
    // according to the connectivity of the given corner table.
    VertexCornersIterator<CornerTable> cit(corner_table, corner_id);
    // Position of central vertex does not change in loop.
    const VectorD<int64_t, 3> pos_cent = this->GetPositionForCorner(corner_id);
    // Computing normals for triangles and adding them up.

    VectorD<int64_t, 3> normal;
    CornerIndex c_next, c_prev;
    while (!cit.End()) {
      // Getting corners.
      if (this->normal_prediction_mode_ == ONE_TRIANGLE) {
        c_next = corner_table->Next(corner_id);
        c_prev = corner_table->Previous(corner_id);
      } else {
        c_next = corner_table->Next(cit.Corner());
        c_prev = corner_table->Previous(cit.Corner());
      }
      const VectorD<int64_t, 3> pos_next = this->GetPositionForCorner(c_next);
      const VectorD<int64_t, 3> pos_prev = this->GetPositionForCorner(c_prev);

      // Computing delta vectors to next and prev.
      const VectorD<int64_t, 3> delta_next = pos_next - pos_cent;
      const VectorD<int64_t, 3> delta_prev = pos_prev - pos_cent;

      // Computing cross product.
      const VectorD<int64_t, 3> cross = CrossProduct(delta_next, delta_prev);

      // Prevent signed integer overflows by doing math as unsigned.
      auto normal_data = reinterpret_cast<uint64_t *>(normal.data());
      auto cross_data = reinterpret_cast<const uint64_t *>(cross.data());
      normal_data[0] = normal_data[0] + cross_data[0];
      normal_data[1] = normal_data[1] + cross_data[1];
      normal_data[2] = normal_data[2] + cross_data[2];

      cit.Next();
    }

    // Convert to int32_t, make sure entries are not too large.
    constexpr int64_t upper_bound = 1 << 29;
    if (this->normal_prediction_mode_ == ONE_TRIANGLE) {
      const int32_t abs_sum = static_cast<int32_t>(normal.AbsSum());
      if (abs_sum > upper_bound) {
        const int64_t quotient = abs_sum / upper_bound;
        normal = normal / quotient;
      }
    } else {
      const int64_t abs_sum = normal.AbsSum();
      if (abs_sum > upper_bound) {
        const int64_t quotient = abs_sum / upper_bound;
        normal = normal / quotient;
      }
    }
    DRACO_DCHECK_LE(normal.AbsSum(), upper_bound);
    prediction[0] = static_cast<int32_t>(normal[0]);
    prediction[1] = static_cast<int32_t>(normal[1]);
    prediction[2] = static_cast<int32_t>(normal[2]);
  }
  bool SetNormalPredictionMode(NormalPredictionMode mode) override {
    if (mode == ONE_TRIANGLE) {
      this->normal_prediction_mode_ = mode;
      return true;
    } else if (mode == TRIANGLE_AREA) {
      this->normal_prediction_mode_ = mode;
      return true;
    }
    return false;
  }
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_MESH_PREDICTION_SCHEME_GEOMETRIC_NORMAL_PREDICTOR_AREA_H_
