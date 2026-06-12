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
#ifndef DRACO_COMPRESSION_ATTRIBUTES_SEQUENTIAL_INTEGER_ATTRIBUTE_DECODER_H_
#define DRACO_COMPRESSION_ATTRIBUTES_SEQUENTIAL_INTEGER_ATTRIBUTE_DECODER_H_

#include "draco/compression/attributes/prediction_schemes/prediction_scheme_decoder.h"
#include "draco/compression/attributes/sequential_attribute_decoder.h"
#include "draco/draco_features.h"

namespace draco {

// Decoder for attributes encoded with the SequentialIntegerAttributeEncoder.
class SequentialIntegerAttributeDecoder : public SequentialAttributeDecoder {
 public:
  SequentialIntegerAttributeDecoder();
  bool Init(PointCloudDecoder *decoder, int attribute_id) override;

  bool TransformAttributeToOriginalFormat(
      const std::vector<PointIndex> &point_ids) override;

 protected:
  bool DecodeValues(const std::vector<PointIndex> &point_ids,
                    DecoderBuffer *in_buffer) override;
  virtual bool DecodeIntegerValues(const std::vector<PointIndex> &point_ids,
                                   DecoderBuffer *in_buffer);

  // Returns a prediction scheme that should be used for decoding of the
  // integer values.
  virtual std::unique_ptr<PredictionSchemeTypedDecoderInterface<int32_t>>
  CreateIntPredictionScheme(PredictionSchemeMethod method,
                            PredictionSchemeTransformType transform_type);

  // Returns the number of integer attribute components. In general, this
  // can be different from the number of components of the input attribute.
  virtual int32_t GetNumValueComponents() const {
    return attribute()->num_components();
  }

  // Called after all integer values are decoded. The implementation should
  // use this method to store the values into the attribute.
  virtual bool StoreValues(uint32_t num_values);

  void PreparePortableAttribute(int num_entries, int num_components);

  int32_t *GetPortableAttributeData() {
    if (portable_attribute()->size() == 0) {
      return nullptr;
    }
    return reinterpret_cast<int32_t *>(
        portable_attribute()->GetAddress(AttributeValueIndex(0)));
  }

 private:
  // Stores decoded values into the attribute with a data type AttributeTypeT.
  template <typename AttributeTypeT>
  void StoreTypedValues(uint32_t num_values);

  std::unique_ptr<PredictionSchemeTypedDecoderInterface<int32_t>>
      prediction_scheme_;
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_ATTRIBUTES_SEQUENTIAL_INTEGER_ATTRIBUTE_DECODER_H_
