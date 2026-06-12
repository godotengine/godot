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
#include "draco/compression/attributes/sequential_quantization_attribute_decoder.h"

#include "draco/core/quantization_utils.h"

namespace draco {

SequentialQuantizationAttributeDecoder::
    SequentialQuantizationAttributeDecoder() {}

bool SequentialQuantizationAttributeDecoder::Init(PointCloudDecoder *decoder,
                                                  int attribute_id) {
  if (!SequentialIntegerAttributeDecoder::Init(decoder, attribute_id)) {
    return false;
  }
  const PointAttribute *const attribute =
      decoder->point_cloud()->attribute(attribute_id);
  // Currently we can quantize only floating point arguments.
  if (attribute->data_type() != DT_FLOAT32) {
    return false;
  }
  return true;
}

bool SequentialQuantizationAttributeDecoder::DecodeIntegerValues(
    const std::vector<PointIndex> &point_ids, DecoderBuffer *in_buffer) {
#ifdef DRACO_BACKWARDS_COMPATIBILITY_SUPPORTED
  if (decoder()->bitstream_version() < DRACO_BITSTREAM_VERSION(2, 0) &&
      !DecodeQuantizedDataInfo()) {
    return false;
  }
#endif
  return SequentialIntegerAttributeDecoder::DecodeIntegerValues(point_ids,
                                                                in_buffer);
}

bool SequentialQuantizationAttributeDecoder::
    DecodeDataNeededByPortableTransform(
        const std::vector<PointIndex> &point_ids, DecoderBuffer *in_buffer) {
  if (decoder()->bitstream_version() >= DRACO_BITSTREAM_VERSION(2, 0)) {
    // Decode quantization data here only for files with bitstream version 2.0+
    if (!DecodeQuantizedDataInfo()) {
      return false;
    }
  }

  // Store the decoded transform data in portable attribute;
  return quantization_transform_.TransferToAttribute(portable_attribute());
}

bool SequentialQuantizationAttributeDecoder::StoreValues(uint32_t num_points) {
  return DequantizeValues(num_points);
}

bool SequentialQuantizationAttributeDecoder::DecodeQuantizedDataInfo() {
  // Get attribute used as source for decoding.
  auto att = GetPortableAttribute();
  if (att == nullptr) {
    // This should happen only in the backward compatibility mode. It will still
    // work fine for this case because the only thing the quantization transform
    // cares about is the number of components that is the same for both source
    // and target attributes.
    att = attribute();
  }
  return quantization_transform_.DecodeParameters(*att, decoder()->buffer());
}

bool SequentialQuantizationAttributeDecoder::DequantizeValues(
    uint32_t num_values) {
  // Convert all quantized values back to floats.
  return quantization_transform_.InverseTransformAttribute(
      *GetPortableAttribute(), attribute());
}

}  // namespace draco
