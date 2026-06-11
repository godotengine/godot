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
#ifndef DRACO_ATTRIBUTES_ATTRIBUTE_TRANSFORM_H_
#define DRACO_ATTRIBUTES_ATTRIBUTE_TRANSFORM_H_

#include "draco/attributes/attribute_transform_data.h"
#include "draco/attributes/point_attribute.h"
#include "draco/core/decoder_buffer.h"
#include "draco/core/encoder_buffer.h"

namespace draco {

// Virtual base class for various attribute transforms, enforcing common
// interface where possible.
class AttributeTransform {
 public:
  virtual ~AttributeTransform() = default;

  // Return attribute transform type.
  virtual AttributeTransformType Type() const = 0;
  // Try to init transform from attribute.
  virtual bool InitFromAttribute(const PointAttribute &attribute) = 0;
  // Copy parameter values into the provided AttributeTransformData instance.
  virtual void CopyToAttributeTransformData(
      AttributeTransformData *out_data) const = 0;
  bool TransferToAttribute(PointAttribute *attribute) const;

  // Applies the transform to |attribute| and stores the result in
  // |target_attribute|. |point_ids| is an optional vector that can be used to
  // remap values during the transform.
  virtual bool TransformAttribute(const PointAttribute &attribute,
                                  const std::vector<PointIndex> &point_ids,
                                  PointAttribute *target_attribute) = 0;

  // Applies an inverse transform to |attribute| and stores the result in
  // |target_attribute|. In this case, |attribute| is an attribute that was
  // already transformed (e.g. quantized) and |target_attribute| is the
  // attribute before the transformation.
  virtual bool InverseTransformAttribute(const PointAttribute &attribute,
                                         PointAttribute *target_attribute) = 0;

  // Encodes all data needed by the transformation into the |encoder_buffer|.
  virtual bool EncodeParameters(EncoderBuffer *encoder_buffer) const = 0;

  // Decodes all data needed to transform |attribute| back to the original
  // format.
  virtual bool DecodeParameters(const PointAttribute &attribute,
                                DecoderBuffer *decoder_buffer) = 0;

  // Initializes a transformed attribute that can be used as target in the
  // TransformAttribute() function call.
  virtual std::unique_ptr<PointAttribute> InitTransformedAttribute(
      const PointAttribute &src_attribute, int num_entries);

 protected:
  virtual DataType GetTransformedDataType(
      const PointAttribute &attribute) const = 0;
  virtual int GetTransformedNumComponents(
      const PointAttribute &attribute) const = 0;
};

}  // namespace draco

#endif  // DRACO_ATTRIBUTES_ATTRIBUTE_OCTAHEDRON_TRANSFORM_H_
