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
#ifndef DRACO_ATTRIBUTES_ATTRIBUTE_QUANTIZATION_TRANSFORM_H_
#define DRACO_ATTRIBUTES_ATTRIBUTE_QUANTIZATION_TRANSFORM_H_

#include <vector>

#include "draco/attributes/attribute_transform.h"
#include "draco/attributes/point_attribute.h"
#include "draco/core/encoder_buffer.h"

namespace draco {

// Attribute transform for quantized attributes.
class AttributeQuantizationTransform : public AttributeTransform {
 public:
  AttributeQuantizationTransform() : quantization_bits_(-1), range_(0.f) {}
  // Return attribute transform type.
  AttributeTransformType Type() const override {
    return ATTRIBUTE_QUANTIZATION_TRANSFORM;
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

  bool SetParameters(int quantization_bits, const float *min_values,
                     int num_components, float range);

  bool ComputeParameters(const PointAttribute &attribute,
                         const int quantization_bits);

  // Encode relevant parameters into buffer.
  bool EncodeParameters(EncoderBuffer *encoder_buffer) const override;

  bool DecodeParameters(const PointAttribute &attribute,
                        DecoderBuffer *decoder_buffer) override;

  int32_t quantization_bits() const { return quantization_bits_; }
  float min_value(int axis) const { return min_values_[axis]; }
  const std::vector<float> &min_values() const { return min_values_; }
  float range() const { return range_; }
  bool is_initialized() const { return quantization_bits_ != -1; }

 protected:
  // Create portable attribute using 1:1 mapping between points in the input and
  // output attribute.
  void GeneratePortableAttribute(const PointAttribute &attribute,
                                 int num_points,
                                 PointAttribute *target_attribute) const;

  // Create portable attribute using custom mapping between input and output
  // points.
  void GeneratePortableAttribute(const PointAttribute &attribute,
                                 const std::vector<PointIndex> &point_ids,
                                 int num_points,
                                 PointAttribute *target_attribute) const;

  DataType GetTransformedDataType(
      const PointAttribute &attribute) const override {
    return DT_UINT32;
  }
  int GetTransformedNumComponents(
      const PointAttribute &attribute) const override {
    return attribute.num_components();
  }

  static bool IsQuantizationValid(int quantization_bits);

 private:
  int32_t quantization_bits_;

  // Minimal dequantized value for each component of the attribute.
  std::vector<float> min_values_;

  // Bounds of the dequantized attribute (max delta over all components).
  float range_;
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_ATTRIBUTES_ATTRIBUTE_DEQUANTIZATION_TRANSFORM_H_
