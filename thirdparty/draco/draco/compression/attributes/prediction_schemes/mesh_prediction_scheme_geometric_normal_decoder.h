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
#ifndef DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_MESH_PREDICTION_SCHEME_GEOMETRIC_NORMAL_DECODER_H_
#define DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_MESH_PREDICTION_SCHEME_GEOMETRIC_NORMAL_DECODER_H_

#include "draco/compression/attributes/prediction_schemes/mesh_prediction_scheme_decoder.h"
#include "draco/compression/attributes/prediction_schemes/mesh_prediction_scheme_geometric_normal_predictor_area.h"
#include "draco/compression/bit_coders/rans_bit_decoder.h"
#include "draco/draco_features.h"

namespace draco {

// See MeshPredictionSchemeGeometricNormalEncoder for documentation.
template <typename DataTypeT, class TransformT, class MeshDataT>
class MeshPredictionSchemeGeometricNormalDecoder
    : public MeshPredictionSchemeDecoder<DataTypeT, TransformT, MeshDataT> {
 public:
  using CorrType = typename MeshPredictionSchemeDecoder<DataTypeT, TransformT,
                                                        MeshDataT>::CorrType;
  MeshPredictionSchemeGeometricNormalDecoder(const PointAttribute *attribute,
                                             const TransformT &transform,
                                             const MeshDataT &mesh_data)
      : MeshPredictionSchemeDecoder<DataTypeT, TransformT, MeshDataT>(
            attribute, transform, mesh_data),
        predictor_(mesh_data) {}

 private:
  MeshPredictionSchemeGeometricNormalDecoder() {}

 public:
  bool ComputeOriginalValues(const CorrType *in_corr, DataTypeT *out_data,
                             int size, int num_components,
                             const PointIndex *entry_to_point_id_map) override;

  bool DecodePredictionData(DecoderBuffer *buffer) override;

  PredictionSchemeMethod GetPredictionMethod() const override {
    return MESH_PREDICTION_GEOMETRIC_NORMAL;
  }

  bool IsInitialized() const override {
    if (!predictor_.IsInitialized()) {
      return false;
    }
    if (!this->mesh_data().IsInitialized()) {
      return false;
    }
    if (!octahedron_tool_box_.IsInitialized()) {
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
    if (att->attribute_type() != GeometryAttribute::POSITION) {
      return false;  // Invalid attribute type.
    }
    if (att->num_components() != 3) {
      return false;  // Currently works only for 3 component positions.
    }
    predictor_.SetPositionAttribute(*att);
    return true;
  }
  void SetQuantizationBits(int q) {
    octahedron_tool_box_.SetQuantizationBits(q);
  }

 private:
  MeshPredictionSchemeGeometricNormalPredictorArea<DataTypeT, TransformT,
                                                   MeshDataT>
      predictor_;
  OctahedronToolBox octahedron_tool_box_;
  RAnsBitDecoder flip_normal_bit_decoder_;
};

template <typename DataTypeT, class TransformT, class MeshDataT>
bool MeshPredictionSchemeGeometricNormalDecoder<
    DataTypeT, TransformT,
    MeshDataT>::ComputeOriginalValues(const CorrType *in_corr,
                                      DataTypeT *out_data, int /* size */,
                                      int num_components,
                                      const PointIndex *entry_to_point_id_map) {
  this->SetQuantizationBits(this->transform().quantization_bits());
  predictor_.SetEntryToPointIdMap(entry_to_point_id_map);
  DRACO_DCHECK(this->IsInitialized());

  // Expecting in_data in octahedral coordinates, i.e., portable attribute.
  DRACO_DCHECK_EQ(num_components, 2);

  const int corner_map_size =
      static_cast<int>(this->mesh_data().data_to_corner_map()->size());

  VectorD<int32_t, 3> pred_normal_3d;
  int32_t pred_normal_oct[2];

  for (int data_id = 0; data_id < corner_map_size; ++data_id) {
    const CornerIndex corner_id =
        this->mesh_data().data_to_corner_map()->at(data_id);
    predictor_.ComputePredictedValue(corner_id, pred_normal_3d.data());

    // Compute predicted octahedral coordinates.
    octahedron_tool_box_.CanonicalizeIntegerVector(pred_normal_3d.data());
    DRACO_DCHECK_EQ(pred_normal_3d.AbsSum(),
                    octahedron_tool_box_.center_value());
    if (flip_normal_bit_decoder_.DecodeNextBit()) {
      pred_normal_3d = -pred_normal_3d;
    }
    octahedron_tool_box_.IntegerVectorToQuantizedOctahedralCoords(
        pred_normal_3d.data(), pred_normal_oct, pred_normal_oct + 1);

    const int data_offset = data_id * 2;
    this->transform().ComputeOriginalValue(
        pred_normal_oct, in_corr + data_offset, out_data + data_offset);
  }
  flip_normal_bit_decoder_.EndDecoding();
  return true;
}

template <typename DataTypeT, class TransformT, class MeshDataT>
bool MeshPredictionSchemeGeometricNormalDecoder<
    DataTypeT, TransformT, MeshDataT>::DecodePredictionData(DecoderBuffer
                                                                *buffer) {
  // Get data needed for transform
  if (!this->transform().DecodeTransformData(buffer)) {
    return false;
  }

#ifdef DRACO_BACKWARDS_COMPATIBILITY_SUPPORTED
  if (buffer->bitstream_version() < DRACO_BITSTREAM_VERSION(2, 2)) {
    uint8_t prediction_mode;
    if (!buffer->Decode(&prediction_mode)) {
      return false;
    }
    if (prediction_mode > TRIANGLE_AREA) {
      // Invalid prediction mode.
      return false;
    }

    if (!predictor_.SetNormalPredictionMode(
            NormalPredictionMode(prediction_mode))) {
      return false;
    }
  }
#endif

  // Init normal flips.
  if (!flip_normal_bit_decoder_.StartDecoding(buffer)) {
    return false;
  }

  return true;
}

}  // namespace draco

#endif  // DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_MESH_PREDICTION_SCHEME_GEOMETRIC_NORMAL_DECODER_H_
