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
#ifndef DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_MESH_PREDICTION_SCHEME_GEOMETRIC_NORMAL_PREDICTOR_BASE_H_
#define DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_MESH_PREDICTION_SCHEME_GEOMETRIC_NORMAL_PREDICTOR_BASE_H_

#include <math.h>

#include "draco/attributes/point_attribute.h"
#include "draco/compression/attributes/normal_compression_utils.h"
#include "draco/compression/config/compression_shared.h"
#include "draco/core/math_utils.h"
#include "draco/core/vector_d.h"
#include "draco/mesh/corner_table.h"
#include "draco/mesh/corner_table_iterators.h"

namespace draco {

// Base class for geometric normal predictors using position attribute.
template <typename DataTypeT, class TransformT, class MeshDataT>
class MeshPredictionSchemeGeometricNormalPredictorBase {
 protected:
  explicit MeshPredictionSchemeGeometricNormalPredictorBase(const MeshDataT &md)
      : pos_attribute_(nullptr),
        entry_to_point_id_map_(nullptr),
        mesh_data_(md) {}
  virtual ~MeshPredictionSchemeGeometricNormalPredictorBase() {}

 public:
  void SetPositionAttribute(const PointAttribute &position_attribute) {
    pos_attribute_ = &position_attribute;
  }
  void SetEntryToPointIdMap(const PointIndex *map) {
    entry_to_point_id_map_ = map;
  }
  bool IsInitialized() const {
    if (pos_attribute_ == nullptr) {
      return false;
    }
    if (entry_to_point_id_map_ == nullptr) {
      return false;
    }
    return true;
  }

  virtual bool SetNormalPredictionMode(NormalPredictionMode mode) = 0;
  virtual NormalPredictionMode GetNormalPredictionMode() const {
    return normal_prediction_mode_;
  }

 protected:
  VectorD<int64_t, 3> GetPositionForDataId(int data_id) const {
    DRACO_DCHECK(this->IsInitialized());
    const auto point_id = entry_to_point_id_map_[data_id];
    const auto pos_val_id = pos_attribute_->mapped_index(point_id);
    VectorD<int64_t, 3> pos;
    pos_attribute_->ConvertValue(pos_val_id, &pos[0]);
    return pos;
  }
  VectorD<int64_t, 3> GetPositionForCorner(CornerIndex ci) const {
    DRACO_DCHECK(this->IsInitialized());
    const auto corner_table = mesh_data_.corner_table();
    const auto vert_id = corner_table->Vertex(ci).value();
    const auto data_id = mesh_data_.vertex_to_data_map()->at(vert_id);
    return GetPositionForDataId(data_id);
  }
  VectorD<int32_t, 2> GetOctahedralCoordForDataId(int data_id,
                                                  const DataTypeT *data) const {
    DRACO_DCHECK(this->IsInitialized());
    const int data_offset = data_id * 2;
    return VectorD<int32_t, 2>(data[data_offset], data[data_offset + 1]);
  }
  // Computes predicted octahedral coordinates on a given corner.
  virtual void ComputePredictedValue(CornerIndex corner_id,
                                     DataTypeT *prediction) = 0;

  const PointAttribute *pos_attribute_;
  const PointIndex *entry_to_point_id_map_;
  MeshDataT mesh_data_;
  NormalPredictionMode normal_prediction_mode_;
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_MESH_PREDICTION_SCHEME_GEOMETRIC_NORMAL_PREDICTOR_BASE_H_
