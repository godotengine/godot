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
#ifndef DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_MESH_PREDICTION_SCHEME_DATA_H_
#define DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_MESH_PREDICTION_SCHEME_DATA_H_

#include "draco/mesh/corner_table.h"
#include "draco/mesh/mesh.h"

namespace draco {

// Class stores data about the connectivity data of the mesh and information
// about how the connectivity was encoded/decoded.
template <class CornerTableT>
class MeshPredictionSchemeData {
 public:
  typedef CornerTableT CornerTable;
  MeshPredictionSchemeData()
      : mesh_(nullptr),
        corner_table_(nullptr),
        vertex_to_data_map_(nullptr),
        data_to_corner_map_(nullptr) {}

  void Set(const Mesh *mesh, const CornerTable *table,
           const std::vector<CornerIndex> *data_to_corner_map,
           const std::vector<int32_t> *vertex_to_data_map) {
    mesh_ = mesh;
    corner_table_ = table;
    data_to_corner_map_ = data_to_corner_map;
    vertex_to_data_map_ = vertex_to_data_map;
  }

  const Mesh *mesh() const { return mesh_; }
  const CornerTable *corner_table() const { return corner_table_; }
  const std::vector<int32_t> *vertex_to_data_map() const {
    return vertex_to_data_map_;
  }
  const std::vector<CornerIndex> *data_to_corner_map() const {
    return data_to_corner_map_;
  }
  bool IsInitialized() const {
    return mesh_ != nullptr && corner_table_ != nullptr &&
           vertex_to_data_map_ != nullptr && data_to_corner_map_ != nullptr;
  }

 private:
  const Mesh *mesh_;
  const CornerTable *corner_table_;

  // Mapping between vertices and their encoding order. I.e. when an attribute
  // entry on a given vertex was encoded.
  const std::vector<int32_t> *vertex_to_data_map_;

  // Array that stores which corner was processed when a given attribute entry
  // was encoded or decoded.
  const std::vector<CornerIndex> *data_to_corner_map_;
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_MESH_PREDICTION_SCHEME_DATA_H_
