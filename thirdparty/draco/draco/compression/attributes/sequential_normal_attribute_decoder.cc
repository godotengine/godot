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
#include "draco/compression/attributes/sequential_normal_attribute_decoder.h"

#include "draco/compression/attributes/normal_compression_utils.h"

namespace draco {

SequentialNormalAttributeDecoder::SequentialNormalAttributeDecoder() {}

bool SequentialNormalAttributeDecoder::Init(PointCloudDecoder *decoder,
                                            int attribute_id) {
  if (!SequentialIntegerAttributeDecoder::Init(decoder, attribute_id)) {
    return false;
  }
  // Currently, this encoder works only for 3-component normal vectors.
  if (attribute()->num_components() != 3) {
    return false;
  }
  // Also the data type must be DT_FLOAT32.
  if (attribute()->data_type() != DT_FLOAT32) {
    return false;
  }
  return true;
}

bool SequentialNormalAttributeDecoder::DecodeIntegerValues(
    const std::vector<PointIndex> &point_ids, DecoderBuffer *in_buffer) {
#ifdef DRACO_BACKWARDS_COMPATIBILITY_SUPPORTED
  if (decoder()->bitstream_version() < DRACO_BITSTREAM_VERSION(2, 0)) {
    // Note: in older bitstreams, we do not have a PortableAttribute() decoded
    // at this stage so we cannot pass it down to the DecodeParameters() call.
    // It still works fine for octahedral transform because it does not need to
    // use any data from the attribute.
    if (!octahedral_transform_.DecodeParameters(*attribute(), in_buffer)) {
      return false;
    }
  }
#endif
  return SequentialIntegerAttributeDecoder::DecodeIntegerValues(point_ids,
                                                                in_buffer);
}

bool SequentialNormalAttributeDecoder::DecodeDataNeededByPortableTransform(
    const std::vector<PointIndex> &point_ids, DecoderBuffer *in_buffer) {
  if (decoder()->bitstream_version() >= DRACO_BITSTREAM_VERSION(2, 0)) {
    // For newer file version, decode attribute transform data here.
    if (!octahedral_transform_.DecodeParameters(*GetPortableAttribute(),
                                                in_buffer)) {
      return false;
    }
  }

  // Store the decoded transform data in portable attribute.
  return octahedral_transform_.TransferToAttribute(portable_attribute());
}

bool SequentialNormalAttributeDecoder::StoreValues(uint32_t num_points) {
  // Convert all quantized values back to floats.
  return octahedral_transform_.InverseTransformAttribute(
      *GetPortableAttribute(), attribute());
}

}  // namespace draco
