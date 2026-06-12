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
#include "draco/compression/attributes/attributes_decoder.h"

#include "draco/core/varint_decoding.h"

namespace draco {

AttributesDecoder::AttributesDecoder()
    : point_cloud_decoder_(nullptr), point_cloud_(nullptr) {}

bool AttributesDecoder::Init(PointCloudDecoder *decoder, PointCloud *pc) {
  point_cloud_decoder_ = decoder;
  point_cloud_ = pc;
  return true;
}

bool AttributesDecoder::DecodeAttributesDecoderData(DecoderBuffer *in_buffer) {
  // Decode and create attributes.
  uint32_t num_attributes;
#ifdef DRACO_BACKWARDS_COMPATIBILITY_SUPPORTED
  if (point_cloud_decoder_->bitstream_version() <
      DRACO_BITSTREAM_VERSION(2, 0)) {
    if (!in_buffer->Decode(&num_attributes)) {
      return false;
    }
  } else
#endif
  {
    if (!DecodeVarint(&num_attributes, in_buffer)) {
      return false;
    }
  }

  // Check that decoded number of attributes is valid.
  if (num_attributes == 0) {
    return false;
  }
  if (num_attributes > 5 * in_buffer->remaining_size()) {
    // The decoded number of attributes is unreasonably high, because at least
    // five bytes of attribute descriptor data per attribute are expected.
    return false;
  }

  // Decode attribute descriptor data.
  point_attribute_ids_.resize(num_attributes);
  PointCloud *pc = point_cloud_;
  for (uint32_t i = 0; i < num_attributes; ++i) {
    // Decode attribute descriptor data.
    uint8_t att_type, data_type, num_components, normalized;
    if (!in_buffer->Decode(&att_type)) {
      return false;
    }
    if (!in_buffer->Decode(&data_type)) {
      return false;
    }
    if (!in_buffer->Decode(&num_components)) {
      return false;
    }
    if (!in_buffer->Decode(&normalized)) {
      return false;
    }
    if (att_type >= GeometryAttribute::NAMED_ATTRIBUTES_COUNT) {
      return false;
    }
    if (data_type == DT_INVALID || data_type >= DT_TYPES_COUNT) {
      return false;
    }

    // Check decoded attribute descriptor data.
    if (num_components == 0) {
      return false;
    }

    // Add the attribute to the point cloud.
    const DataType draco_dt = static_cast<DataType>(data_type);
    GeometryAttribute ga;
    ga.Init(static_cast<GeometryAttribute::Type>(att_type), nullptr,
            num_components, draco_dt, normalized > 0,
            DataTypeLength(draco_dt) * num_components, 0);
    uint32_t unique_id;
#ifdef DRACO_BACKWARDS_COMPATIBILITY_SUPPORTED
    if (point_cloud_decoder_->bitstream_version() <
        DRACO_BITSTREAM_VERSION(1, 3)) {
      uint16_t custom_id;
      if (!in_buffer->Decode(&custom_id)) {
        return false;
      }
      // TODO(draco-eng): Add "custom_id" to attribute metadata.
      unique_id = static_cast<uint32_t>(custom_id);
      ga.set_unique_id(unique_id);
    } else
#endif
    {
      if (!DecodeVarint(&unique_id, in_buffer)) {
        return false;
      }
      ga.set_unique_id(unique_id);
    }
    const int att_id = pc->AddAttribute(
        std::unique_ptr<PointAttribute>(new PointAttribute(ga)));
    pc->attribute(att_id)->set_unique_id(unique_id);
    point_attribute_ids_[i] = att_id;

    // Update the inverse map.
    if (att_id >=
        static_cast<int32_t>(point_attribute_to_local_id_map_.size())) {
      point_attribute_to_local_id_map_.resize(att_id + 1, -1);
    }
    point_attribute_to_local_id_map_[att_id] = i;
  }
  return true;
}

}  // namespace draco
