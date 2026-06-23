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
#ifndef DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_PREDICTION_SCHEME_DELTA_DECODER_H_
#define DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_PREDICTION_SCHEME_DELTA_DECODER_H_

#include "draco/compression/attributes/prediction_schemes/prediction_scheme_decoder.h"

namespace draco {

// Decoder for values encoded with delta coding. See the corresponding encoder
// for more details.
template <typename DataTypeT, class TransformT>
class PredictionSchemeDeltaDecoder
    : public PredictionSchemeDecoder<DataTypeT, TransformT> {
 public:
  using CorrType =
      typename PredictionSchemeDecoder<DataTypeT, TransformT>::CorrType;
  // Initialized the prediction scheme.
  explicit PredictionSchemeDeltaDecoder(const PointAttribute *attribute)
      : PredictionSchemeDecoder<DataTypeT, TransformT>(attribute) {}
  PredictionSchemeDeltaDecoder(const PointAttribute *attribute,
                               const TransformT &transform)
      : PredictionSchemeDecoder<DataTypeT, TransformT>(attribute, transform) {}

  bool ComputeOriginalValues(const CorrType *in_corr, DataTypeT *out_data,
                             int size, int num_components,
                             const PointIndex *entry_to_point_id_map) override;
  PredictionSchemeMethod GetPredictionMethod() const override {
    return PREDICTION_DIFFERENCE;
  }
  bool IsInitialized() const override { return true; }
};

template <typename DataTypeT, class TransformT>
bool PredictionSchemeDeltaDecoder<DataTypeT, TransformT>::ComputeOriginalValues(
    const CorrType *in_corr, DataTypeT *out_data, int size, int num_components,
    const PointIndex *) {
  this->transform().Init(num_components);
  // Decode the original value for the first element.
  std::unique_ptr<DataTypeT[]> zero_vals(new DataTypeT[num_components]());
  this->transform().ComputeOriginalValue(zero_vals.get(), in_corr, out_data);

  // Decode data from the front using D(i) = D(i) + D(i - 1).
  for (int i = num_components; i < size; i += num_components) {
    this->transform().ComputeOriginalValue(out_data + i - num_components,
                                           in_corr + i, out_data + i);
  }
  return true;
}

}  // namespace draco

#endif  // DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_PREDICTION_SCHEME_DELTA_DECODER_H_
