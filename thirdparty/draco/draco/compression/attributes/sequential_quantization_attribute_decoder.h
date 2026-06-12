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
#ifndef DRACO_COMPRESSION_ATTRIBUTES_SEQUENTIAL_QUANTIZATION_ATTRIBUTE_DECODER_H_
#define DRACO_COMPRESSION_ATTRIBUTES_SEQUENTIAL_QUANTIZATION_ATTRIBUTE_DECODER_H_

#include "draco/attributes/attribute_quantization_transform.h"
#include "draco/compression/attributes/sequential_integer_attribute_decoder.h"
#include "draco/draco_features.h"

namespace draco {

// Decoder for attribute values encoded with the
// SequentialQuantizationAttributeEncoder.
class SequentialQuantizationAttributeDecoder
    : public SequentialIntegerAttributeDecoder {
 public:
  SequentialQuantizationAttributeDecoder();
  bool Init(PointCloudDecoder *decoder, int attribute_id) override;

 protected:
  bool DecodeIntegerValues(const std::vector<PointIndex> &point_ids,
                           DecoderBuffer *in_buffer) override;
  bool DecodeDataNeededByPortableTransform(
      const std::vector<PointIndex> &point_ids,
      DecoderBuffer *in_buffer) override;
  bool StoreValues(uint32_t num_points) override;

  // Decodes data necessary for dequantizing the encoded values.
  virtual bool DecodeQuantizedDataInfo();

  // Dequantizes all values and stores them into the output attribute.
  virtual bool DequantizeValues(uint32_t num_values);

 private:
  AttributeQuantizationTransform quantization_transform_;
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_ATTRIBUTES_SEQUENTIAL_QUANTIZATION_ATTRIBUTE_DECODER_H_
