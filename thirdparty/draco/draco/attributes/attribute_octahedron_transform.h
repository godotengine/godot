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

#ifndef DRACO_ATTRIBUTES_ATTRIBUTE_OCTAHEDRON_TRANSFORM_H_
#define DRACO_ATTRIBUTES_ATTRIBUTE_OCTAHEDRON_TRANSFORM_H_

#include "draco/attributes/attribute_transform.h"
#include "draco/attributes/point_attribute.h"
#include "draco/core/encoder_buffer.h"

namespace draco {

// Attribute transform for attributes transformed to octahedral coordinates.
class AttributeOctahedronTransform : public AttributeTransform {
 public:
  AttributeOctahedronTransform() : quantization_bits_(-1) {}

  // Return attribute transform type.
  AttributeTransformType Type() const override {
    return ATTRIBUTE_OCTAHEDRON_TRANSFORM;
  }
  // Try to init transform from attribute.
  bool InitFromAttribute(const PointAttribute &attribute) override;
  // Copy parameter values into the provided AttributeTransformData instance.
  void CopyToAttributeTransformData(
      AttributeTransformData *out_data) const override;

  bool TransformAttribute(const PointAttribute &attribute,
                          const std::vector<PointIndex> &point_ids,
                          PointAttribute *target_attribute) override;

  bool InverseTransformAttribute(const PointAttribute &attribute,
                                 PointAttribute *target_attribute) override;

  // Set number of quantization bits.
  void SetParameters(int quantization_bits);

  // Encode relevant parameters into buffer.
  bool EncodeParameters(EncoderBuffer *encoder_buffer) const override;

  bool DecodeParameters(const PointAttribute &attribute,
                        DecoderBuffer *decoder_buffer) override;

  bool is_initialized() const { return quantization_bits_ != -1; }
  int32_t quantization_bits() const { return quantization_bits_; }

 protected:
  DataType GetTransformedDataType(
      const PointAttribute &attribute) const override {
    return DT_UINT32;
  }
  int GetTransformedNumComponents(
      const PointAttribute &attribute) const override {
    return 2;
  }

  // Perform the actual transformation.
  bool GeneratePortableAttribute(const PointAttribute &attribute,
                                 const std::vector<PointIndex> &point_ids,
                                 int num_points,
                                 PointAttribute *target_attribute) const;

 private:
  int32_t quantization_bits_;
};

}  // namespace draco

#endif  // DRACO_ATTRIBUTES_ATTRIBUTE_OCTAHEDRON_TRANSFORM_H_
