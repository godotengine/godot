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

#include "draco/attributes/attribute_octahedron_transform.h"

#include "draco/attributes/attribute_transform_type.h"
#include "draco/compression/attributes/normal_compression_utils.h"

namespace draco {

bool AttributeOctahedronTransform::InitFromAttribute(
    const PointAttribute &attribute) {
  const AttributeTransformData *const transform_data =
      attribute.GetAttributeTransformData();
  if (!transform_data ||
      transform_data->transform_type() != ATTRIBUTE_OCTAHEDRON_TRANSFORM) {
    return false;  // Wrong transform type.
  }
  quantization_bits_ = transform_data->GetParameterValue<int32_t>(0);
  return true;
}

void AttributeOctahedronTransform::CopyToAttributeTransformData(
    AttributeTransformData *out_data) const {
  out_data->set_transform_type(ATTRIBUTE_OCTAHEDRON_TRANSFORM);
  out_data->AppendParameterValue(quantization_bits_);
}

bool AttributeOctahedronTransform::TransformAttribute(
    const PointAttribute &attribute, const std::vector<PointIndex> &point_ids,
    PointAttribute *target_attribute) {
  return GeneratePortableAttribute(attribute, point_ids,
                                   target_attribute->size(), target_attribute);
}

bool AttributeOctahedronTransform::InverseTransformAttribute(
    const PointAttribute &attribute, PointAttribute *target_attribute) {
  if (target_attribute->data_type() != DT_FLOAT32) {
    return false;
  }

  const int num_points = target_attribute->size();
  const int num_components = target_attribute->num_components();
  if (num_components != 3) {
    return false;
  }
  constexpr int kEntrySize = sizeof(float) * 3;
  float att_val[3];
  const int32_t *source_attribute_data = reinterpret_cast<const int32_t *>(
      attribute.GetAddress(AttributeValueIndex(0)));
  uint8_t *target_address =
      target_attribute->GetAddress(AttributeValueIndex(0));
  OctahedronToolBox octahedron_tool_box;
  if (!octahedron_tool_box.SetQuantizationBits(quantization_bits_)) {
    return false;
  }
  for (uint32_t i = 0; i < num_points; ++i) {
    const int32_t s = *source_attribute_data++;
    const int32_t t = *source_attribute_data++;
    octahedron_tool_box.QuantizedOctahedralCoordsToUnitVector(s, t, att_val);

    // Store the decoded floating point values into the attribute buffer.
    std::memcpy(target_address, att_val, kEntrySize);
    target_address += kEntrySize;
  }
  return true;
}

void AttributeOctahedronTransform::SetParameters(int quantization_bits) {
  quantization_bits_ = quantization_bits;
}

bool AttributeOctahedronTransform::EncodeParameters(
    EncoderBuffer *encoder_buffer) const {
  if (is_initialized()) {
    encoder_buffer->Encode(static_cast<uint8_t>(quantization_bits_));
    return true;
  }
  return false;
}

bool AttributeOctahedronTransform::DecodeParameters(
    const PointAttribute &attribute, DecoderBuffer *decoder_buffer) {
  uint8_t quantization_bits;
  if (!decoder_buffer->Decode(&quantization_bits)) {
    return false;
  }
  quantization_bits_ = quantization_bits;
  return true;
}

bool AttributeOctahedronTransform::GeneratePortableAttribute(
    const PointAttribute &attribute, const std::vector<PointIndex> &point_ids,
    int num_points, PointAttribute *target_attribute) const {
  DRACO_DCHECK(is_initialized());

  // Quantize all values in the order given by point_ids into portable
  // attribute.
  int32_t *const portable_attribute_data = reinterpret_cast<int32_t *>(
      target_attribute->GetAddress(AttributeValueIndex(0)));
  float att_val[3];
  int32_t dst_index = 0;
  OctahedronToolBox converter;
  if (!converter.SetQuantizationBits(quantization_bits_)) {
    return false;
  }
  if (!point_ids.empty()) {
    for (uint32_t i = 0; i < point_ids.size(); ++i) {
      const AttributeValueIndex att_val_id =
          attribute.mapped_index(point_ids[i]);
      attribute.GetValue(att_val_id, att_val);
      // Encode the vector into a s and t octahedral coordinates.
      int32_t s, t;
      converter.FloatVectorToQuantizedOctahedralCoords(att_val, &s, &t);
      portable_attribute_data[dst_index++] = s;
      portable_attribute_data[dst_index++] = t;
    }
  } else {
    for (PointIndex i(0); i < num_points; ++i) {
      const AttributeValueIndex att_val_id = attribute.mapped_index(i);
      attribute.GetValue(att_val_id, att_val);
      // Encode the vector into a s and t octahedral coordinates.
      int32_t s, t;
      converter.FloatVectorToQuantizedOctahedralCoords(att_val, &s, &t);
      portable_attribute_data[dst_index++] = s;
      portable_attribute_data[dst_index++] = t;
    }
  }

  return true;
}

}  // namespace draco
