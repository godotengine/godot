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
#ifndef DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_PREDICTION_SCHEME_DECODING_TRANSFORM_H_
#define DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_PREDICTION_SCHEME_DECODING_TRANSFORM_H_

#include "draco/compression/config/compression_shared.h"
#include "draco/core/decoder_buffer.h"

namespace draco {

// PredictionSchemeDecodingTransform is used to transform predicted values and
// correction values into the final original attribute values.
// DataTypeT is the data type of predicted values.
// CorrTypeT is the data type used for storing corrected values. It allows
// transforms to store corrections into a different type or format compared to
// the predicted data.
template <typename DataTypeT, typename CorrTypeT>
class PredictionSchemeDecodingTransform {
 public:
  typedef CorrTypeT CorrType;
  PredictionSchemeDecodingTransform() : num_components_(0) {}

  void Init(int num_components) { num_components_ = num_components; }

  // Computes the original value from the input predicted value and the decoded
  // corrections. The default implementation is equal to std:plus.
  inline void ComputeOriginalValue(const DataTypeT *predicted_vals,
                                   const CorrTypeT *corr_vals,
                                   DataTypeT *out_original_vals) const {
    static_assert(std::is_same<DataTypeT, CorrTypeT>::value,
                  "For the default prediction transform, correction and input "
                  "data must be of the same type.");
    for (int i = 0; i < num_components_; ++i) {
      out_original_vals[i] = predicted_vals[i] + corr_vals[i];
    }
  }

  // Decodes any transform specific data. Called before Init() method.
  bool DecodeTransformData(DecoderBuffer * /* buffer */) { return true; }

  // Should return true if all corrected values are guaranteed to be positive.
  bool AreCorrectionsPositive() const { return false; }

 protected:
  int num_components() const { return num_components_; }

 private:
  int num_components_;
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_PREDICTION_SCHEME_DECODING_TRANSFORM_H_
