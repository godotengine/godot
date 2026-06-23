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
#ifdef DRACO_BACKWARDS_COMPATIBILITY_SUPPORTED
#ifndef DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_MESH_PREDICTION_SCHEME_MULTI_PARALLELOGRAM_DECODER_H_
#define DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_MESH_PREDICTION_SCHEME_MULTI_PARALLELOGRAM_DECODER_H_

#include "draco/compression/attributes/prediction_schemes/mesh_prediction_scheme_decoder.h"
#include "draco/compression/attributes/prediction_schemes/mesh_prediction_scheme_parallelogram_shared.h"
#include "draco/core/math_utils.h"
#include "draco/draco_features.h"

namespace draco {

// Decoder for predictions encoded by multi-parallelogram encoding scheme.
// See the corresponding encoder for method description.
template <typename DataTypeT, class TransformT, class MeshDataT>
class MeshPredictionSchemeMultiParallelogramDecoder
    : public MeshPredictionSchemeDecoder<DataTypeT, TransformT, MeshDataT> {
 public:
  using CorrType =
      typename PredictionSchemeDecoder<DataTypeT, TransformT>::CorrType;
  using CornerTable = typename MeshDataT::CornerTable;

  explicit MeshPredictionSchemeMultiParallelogramDecoder(
      const PointAttribute *attribute)
      : MeshPredictionSchemeDecoder<DataTypeT, TransformT, MeshDataT>(
            attribute) {}
  MeshPredictionSchemeMultiParallelogramDecoder(const PointAttribute *attribute,
                                                const TransformT &transform,
                                                const MeshDataT &mesh_data)
      : MeshPredictionSchemeDecoder<DataTypeT, TransformT, MeshDataT>(
            attribute, transform, mesh_data) {}

  bool ComputeOriginalValues(const CorrType *in_corr, DataTypeT *out_data,
                             int size, int num_components,
                             const PointIndex *entry_to_point_id_map) override;
  PredictionSchemeMethod GetPredictionMethod() const override {
    return MESH_PREDICTION_MULTI_PARALLELOGRAM;
  }

  bool IsInitialized() const override {
    return this->mesh_data().IsInitialized();
  }
};

template <typename DataTypeT, class TransformT, class MeshDataT>
bool MeshPredictionSchemeMultiParallelogramDecoder<DataTypeT, TransformT,
                                                   MeshDataT>::
    ComputeOriginalValues(const CorrType *in_corr, DataTypeT *out_data,
                          int /* size */, int num_components,
                          const PointIndex * /* entry_to_point_id_map */) {
  this->transform().Init(num_components);

  // For storage of prediction values (already initialized to zero).
  std::unique_ptr<DataTypeT[]> pred_vals(new DataTypeT[num_components]());
  std::unique_ptr<DataTypeT[]> parallelogram_pred_vals(
      new DataTypeT[num_components]());

  this->transform().ComputeOriginalValue(pred_vals.get(), in_corr, out_data);

  const CornerTable *const table = this->mesh_data().corner_table();
  const std::vector<int32_t> *const vertex_to_data_map =
      this->mesh_data().vertex_to_data_map();

  const int corner_map_size =
      static_cast<int>(this->mesh_data().data_to_corner_map()->size());
  for (int p = 1; p < corner_map_size; ++p) {
    const CornerIndex start_corner_id =
        this->mesh_data().data_to_corner_map()->at(p);

    CornerIndex corner_id(start_corner_id);
    int num_parallelograms = 0;
    for (int i = 0; i < num_components; ++i) {
      pred_vals[i] = static_cast<DataTypeT>(0);
    }
    while (corner_id != kInvalidCornerIndex) {
      if (ComputeParallelogramPrediction(
              p, corner_id, table, *vertex_to_data_map, out_data,
              num_components, parallelogram_pred_vals.get())) {
        for (int c = 0; c < num_components; ++c) {
          pred_vals[c] =
              AddAsUnsigned(pred_vals[c], parallelogram_pred_vals[c]);
        }
        ++num_parallelograms;
      }

      // Proceed to the next corner attached to the vertex.
      corner_id = table->SwingRight(corner_id);
      if (corner_id == start_corner_id) {
        corner_id = kInvalidCornerIndex;
      }
    }

    const int dst_offset = p * num_components;
    if (num_parallelograms == 0) {
      // No parallelogram was valid.
      // We use the last decoded point as a reference.
      const int src_offset = (p - 1) * num_components;
      this->transform().ComputeOriginalValue(
          out_data + src_offset, in_corr + dst_offset, out_data + dst_offset);
    } else {
      // Compute the correction from the predicted value.
      for (int c = 0; c < num_components; ++c) {
        pred_vals[c] /= num_parallelograms;
      }
      this->transform().ComputeOriginalValue(
          pred_vals.get(), in_corr + dst_offset, out_data + dst_offset);
    }
  }
  return true;
}

}  // namespace draco

#endif  // DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_MESH_PREDICTION_SCHEME_MULTI_PARALLELOGRAM_DECODER_H_
#endif
