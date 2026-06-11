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
#ifndef DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_PREDICTION_SCHEME_DECODER_INTERFACE_H_
#define DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_PREDICTION_SCHEME_DECODER_INTERFACE_H_

#include "draco/compression/attributes/prediction_schemes/prediction_scheme_interface.h"
#include "draco/core/decoder_buffer.h"

// Prediction schemes can be used during encoding and decoding of attributes
// to predict attribute values based on the previously encoded/decoded data.
// See prediction_scheme.h for more details.
namespace draco {

// Abstract interface for all prediction schemes used during attribute encoding.
class PredictionSchemeDecoderInterface : public PredictionSchemeInterface {
 public:
  // Method that can be used to decode any prediction scheme specific data
  // from the input buffer.
  virtual bool DecodePredictionData(DecoderBuffer *buffer) = 0;
};

// A specialized version of the prediction scheme interface for specific
// input and output data types.
// |entry_to_point_id_map| is the mapping between value entries to point ids
// of the associated point cloud, where one entry is defined as |num_components|
// values of the |in_data|.
// DataTypeT is the data type of input and predicted values.
// CorrTypeT is the data type used for storing corrected values.
template <typename DataTypeT, typename CorrTypeT = DataTypeT>
class PredictionSchemeTypedDecoderInterface
    : public PredictionSchemeDecoderInterface {
 public:
  // Reverts changes made by the prediction scheme during encoding.
  virtual bool ComputeOriginalValues(
      const CorrTypeT *in_corr, DataTypeT *out_data, int size,
      int num_components, const PointIndex *entry_to_point_id_map) = 0;
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_PREDICTION_SCHEME_DECODER_INTERFACE_H_
