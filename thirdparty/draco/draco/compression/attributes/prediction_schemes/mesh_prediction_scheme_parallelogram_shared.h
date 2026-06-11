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
// Shared functionality for different parallelogram prediction schemes.

#ifndef DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_MESH_PREDICTION_SCHEME_PARALLELOGRAM_SHARED_H_
#define DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_MESH_PREDICTION_SCHEME_PARALLELOGRAM_SHARED_H_

#include "draco/mesh/corner_table.h"
#include "draco/mesh/mesh.h"

namespace draco {

// TODO(draco-eng) consolidate Vertex/next/previous queries to one call
// (performance).
template <class CornerTableT>
inline void GetParallelogramEntries(
    const CornerIndex ci, const CornerTableT *table,
    const std::vector<int32_t> &vertex_to_data_map, int *opp_entry,
    int *next_entry, int *prev_entry) {
  // One vertex of the input |table| correspond to exactly one attribute value
  // entry. The |table| can be either CornerTable for per-vertex attributes,
  // or MeshAttributeCornerTable for attributes with interior seams.
  *opp_entry = vertex_to_data_map[table->Vertex(ci).value()];
  *next_entry = vertex_to_data_map[table->Vertex(table->Next(ci)).value()];
  *prev_entry = vertex_to_data_map[table->Vertex(table->Previous(ci)).value()];
}

// Computes parallelogram prediction for a given corner and data entry id.
// The prediction is stored in |out_prediction|.
// Function returns false when the prediction couldn't be computed, e.g. because
// not all entry points were available.
template <class CornerTableT, typename DataTypeT>
inline bool ComputeParallelogramPrediction(
    int data_entry_id, const CornerIndex ci, const CornerTableT *table,
    const std::vector<int32_t> &vertex_to_data_map, const DataTypeT *in_data,
    int num_components, DataTypeT *out_prediction) {
  const CornerIndex oci = table->Opposite(ci);
  if (oci == kInvalidCornerIndex) {
    return false;
  }
  int vert_opp, vert_next, vert_prev;
  GetParallelogramEntries<CornerTableT>(oci, table, vertex_to_data_map,
                                        &vert_opp, &vert_next, &vert_prev);
  if (vert_opp < data_entry_id && vert_next < data_entry_id &&
      vert_prev < data_entry_id) {
    // Apply the parallelogram prediction.
    const int v_opp_off = vert_opp * num_components;
    const int v_next_off = vert_next * num_components;
    const int v_prev_off = vert_prev * num_components;
    for (int c = 0; c < num_components; ++c) {
      const int64_t in_data_next_off = in_data[v_next_off + c];
      const int64_t in_data_prev_off = in_data[v_prev_off + c];
      const int64_t in_data_opp_off = in_data[v_opp_off + c];
      const int64_t result =
          (in_data_next_off + in_data_prev_off) - in_data_opp_off;

      out_prediction[c] = static_cast<DataTypeT>(result);
    }
    return true;
  }
  return false;  // Not all data is available for prediction
}

}  // namespace draco

#endif  // DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_MESH_PREDICTION_SCHEME_PARALLELOGRAM_SHARED_H_
