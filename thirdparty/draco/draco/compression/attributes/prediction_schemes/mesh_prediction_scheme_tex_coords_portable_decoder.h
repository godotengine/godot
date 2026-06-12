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
#ifndef DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_MESH_PREDICTION_SCHEME_TEX_COORDS_PORTABLE_DECODER_H_
#define DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_MESH_PREDICTION_SCHEME_TEX_COORDS_PORTABLE_DECODER_H_

#include "draco/compression/attributes/prediction_schemes/mesh_prediction_scheme_decoder.h"
#include "draco/compression/attributes/prediction_schemes/mesh_prediction_scheme_tex_coords_portable_predictor.h"
#include "draco/compression/bit_coders/rans_bit_decoder.h"

namespace draco {

// Decoder for predictions of UV coordinates encoded by our specialized and
// portable texture coordinate predictor. See the corresponding encoder for more
// details.
template <typename DataTypeT, class TransformT, class MeshDataT>
class MeshPredictionSchemeTexCoordsPortableDecoder
    : public MeshPredictionSchemeDecoder<DataTypeT, TransformT, MeshDataT> {
 public:
  using CorrType = typename MeshPredictionSchemeDecoder<DataTypeT, TransformT,
                                                        MeshDataT>::CorrType;
  MeshPredictionSchemeTexCoordsPortableDecoder(const PointAttribute *attribute,
                                               const TransformT &transform,
                                               const MeshDataT &mesh_data)
      : MeshPredictionSchemeDecoder<DataTypeT, TransformT, MeshDataT>(
            attribute, transform, mesh_data),
        predictor_(mesh_data) {}

  bool ComputeOriginalValues(const CorrType *in_corr, DataTypeT *out_data,
                             int size, int num_components,
                             const PointIndex *entry_to_point_id_map) override;

  bool DecodePredictionData(DecoderBuffer *buffer) override;

  PredictionSchemeMethod GetPredictionMethod() const override {
    return MESH_PREDICTION_TEX_COORDS_PORTABLE;
  }

  bool IsInitialized() const override {
    if (!predictor_.IsInitialized()) {
      return false;
    }
    if (!this->mesh_data().IsInitialized()) {
      return false;
    }
    return true;
  }

  int GetNumParentAttributes() const override { return 1; }

  GeometryAttribute::Type GetParentAttributeType(int i) const override {
    DRACO_DCHECK_EQ(i, 0);
    (void)i;
    return GeometryAttribute::POSITION;
  }

  bool SetParentAttribute(const PointAttribute *att) override {
    if (!att || att->attribute_type() != GeometryAttribute::POSITION) {
      return false;  // Invalid attribute type.
    }
    if (att->num_components() != 3) {
      return false;  // Currently works only for 3 component positions.
    }
    predictor_.SetPositionAttribute(*att);
    return true;
  }

 private:
  MeshPredictionSchemeTexCoordsPortablePredictor<DataTypeT, MeshDataT>
      predictor_;
};

template <typename DataTypeT, class TransformT, class MeshDataT>
bool MeshPredictionSchemeTexCoordsPortableDecoder<
    DataTypeT, TransformT,
    MeshDataT>::ComputeOriginalValues(const CorrType *in_corr,
                                      DataTypeT *out_data, int /* size */,
                                      int num_components,
                                      const PointIndex *entry_to_point_id_map) {
  if (num_components != MeshPredictionSchemeTexCoordsPortablePredictor<
                            DataTypeT, MeshDataT>::kNumComponents) {
    return false;
  }
  predictor_.SetEntryToPointIdMap(entry_to_point_id_map);
  this->transform().Init(num_components);

  const int corner_map_size =
      static_cast<int>(this->mesh_data().data_to_corner_map()->size());
  for (int p = 0; p < corner_map_size; ++p) {
    const CornerIndex corner_id = this->mesh_data().data_to_corner_map()->at(p);
    if (!predictor_.template ComputePredictedValue<false>(corner_id, out_data,
                                                          p)) {
      return false;
    }

    const int dst_offset = p * num_components;
    this->transform().ComputeOriginalValue(predictor_.predicted_value(),
                                           in_corr + dst_offset,
                                           out_data + dst_offset);
  }
  return true;
}

template <typename DataTypeT, class TransformT, class MeshDataT>
bool MeshPredictionSchemeTexCoordsPortableDecoder<
    DataTypeT, TransformT, MeshDataT>::DecodePredictionData(DecoderBuffer
                                                                *buffer) {
  // Decode the delta coded orientations.
  int32_t num_orientations = 0;
  if (!buffer->Decode(&num_orientations) || num_orientations < 0) {
    return false;
  }
  predictor_.ResizeOrientations(num_orientations);
  bool last_orientation = true;
  RAnsBitDecoder decoder;
  if (!decoder.StartDecoding(buffer)) {
    return false;
  }
  for (int i = 0; i < num_orientations; ++i) {
    if (!decoder.DecodeNextBit()) {
      last_orientation = !last_orientation;
    }
    predictor_.set_orientation(i, last_orientation);
  }
  decoder.EndDecoding();
  return MeshPredictionSchemeDecoder<DataTypeT, TransformT,
                                     MeshDataT>::DecodePredictionData(buffer);
}

}  // namespace draco

#endif  // DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_MESH_PREDICTION_SCHEME_TEX_COORDS_PORTABLE_DECODER_H_
