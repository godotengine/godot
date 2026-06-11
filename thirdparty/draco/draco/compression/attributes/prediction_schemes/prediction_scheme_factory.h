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
// Functions for creating prediction schemes from a provided prediction method
// name. The functions in this file can create only basic prediction schemes
// that don't require any encoder or decoder specific data. To create more
// sophisticated prediction schemes, use functions from either
// prediction_scheme_encoder_factory.h  or,
// prediction_scheme_decoder_factory.h.

#ifndef DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_PREDICTION_SCHEME_FACTORY_H_
#define DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_PREDICTION_SCHEME_FACTORY_H_

#include "draco/compression/attributes/mesh_attribute_indices_encoding_data.h"
#include "draco/compression/attributes/prediction_schemes/mesh_prediction_scheme_data.h"
#include "draco/compression/config/compression_shared.h"
#include "draco/mesh/mesh_attribute_corner_table.h"

namespace draco {

template <class EncodingDataSourceT, class PredictionSchemeT,
          class MeshPredictionSchemeFactoryT>
std::unique_ptr<PredictionSchemeT> CreateMeshPredictionScheme(
    const EncodingDataSourceT *source, PredictionSchemeMethod method,
    int att_id, const typename PredictionSchemeT::Transform &transform,
    uint16_t bitstream_version) {
  const PointAttribute *const att = source->point_cloud()->attribute(att_id);
  if (source->GetGeometryType() == TRIANGULAR_MESH &&
      (method == MESH_PREDICTION_PARALLELOGRAM ||
       method == MESH_PREDICTION_MULTI_PARALLELOGRAM ||
       method == MESH_PREDICTION_CONSTRAINED_MULTI_PARALLELOGRAM ||
       method == MESH_PREDICTION_TEX_COORDS_PORTABLE ||
       method == MESH_PREDICTION_GEOMETRIC_NORMAL ||
       method == MESH_PREDICTION_TEX_COORDS_DEPRECATED)) {
    const CornerTable *const ct = source->GetCornerTable();
    const MeshAttributeIndicesEncodingData *const encoding_data =
        source->GetAttributeEncodingData(att_id);
    if (ct == nullptr || encoding_data == nullptr) {
      // No connectivity data found.
      return nullptr;
    }
    // Connectivity data exists.
    const MeshAttributeCornerTable *const att_ct =
        source->GetAttributeCornerTable(att_id);
    if (att_ct != nullptr) {
      typedef MeshPredictionSchemeData<MeshAttributeCornerTable> MeshData;
      MeshData md;
      md.Set(source->mesh(), att_ct,
             &encoding_data->encoded_attribute_value_index_to_corner_map,
             &encoding_data->vertex_to_encoded_attribute_value_index_map);
      MeshPredictionSchemeFactoryT factory;
      auto ret = factory(method, att, transform, md, bitstream_version);
      if (ret) {
        return ret;
      }
    } else {
      typedef MeshPredictionSchemeData<CornerTable> MeshData;
      MeshData md;
      md.Set(source->mesh(), ct,
             &encoding_data->encoded_attribute_value_index_to_corner_map,
             &encoding_data->vertex_to_encoded_attribute_value_index_map);
      MeshPredictionSchemeFactoryT factory;
      auto ret = factory(method, att, transform, md, bitstream_version);
      if (ret) {
        return ret;
      }
    }
  }
  return nullptr;
}

}  // namespace draco

#endif  // DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_PREDICTION_SCHEME_FACTORY_H_
