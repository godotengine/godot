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
#include "draco/attributes/attribute_transform.h"

namespace draco {

bool AttributeTransform::TransferToAttribute(PointAttribute *attribute) const {
  std::unique_ptr<AttributeTransformData> transform_data(
      new AttributeTransformData());
  this->CopyToAttributeTransformData(transform_data.get());
  attribute->SetAttributeTransformData(std::move(transform_data));
  return true;
}

std::unique_ptr<PointAttribute> AttributeTransform::InitTransformedAttribute(
    const PointAttribute &src_attribute, int num_entries) {
  const int num_components = GetTransformedNumComponents(src_attribute);
  const DataType dt = GetTransformedDataType(src_attribute);
  GeometryAttribute ga;
  ga.Init(src_attribute.attribute_type(), nullptr, num_components, dt, false,
          num_components * DataTypeLength(dt), 0);
  std::unique_ptr<PointAttribute> transformed_attribute(new PointAttribute(ga));
  transformed_attribute->Reset(num_entries);
  transformed_attribute->SetIdentityMapping();
  transformed_attribute->set_unique_id(src_attribute.unique_id());
  return transformed_attribute;
}

}  // namespace draco
