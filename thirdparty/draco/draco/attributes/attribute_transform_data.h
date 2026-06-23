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
#ifndef DRACO_ATTRIBUTES_ATTRIBUTE_TRANSFORM_DATA_H_
#define DRACO_ATTRIBUTES_ATTRIBUTE_TRANSFORM_DATA_H_

#include <memory>

#include "draco/attributes/attribute_transform_type.h"
#include "draco/core/data_buffer.h"

namespace draco {

// Class for holding parameter values for an attribute transform of a
// PointAttribute. This can be for example quantization data for an attribute
// that holds quantized values. This class provides only a basic storage for
// attribute transform parameters and it should be accessed only through wrapper
// classes for a specific transform (e.g. AttributeQuantizationTransform).
class AttributeTransformData {
 public:
  AttributeTransformData() : transform_type_(ATTRIBUTE_INVALID_TRANSFORM) {}
  AttributeTransformData(const AttributeTransformData &data) = default;

  // Returns the type of the attribute transform that is described by the class.
  AttributeTransformType transform_type() const { return transform_type_; }
  void set_transform_type(AttributeTransformType type) {
    transform_type_ = type;
  }

  // Returns a parameter value on a given |byte_offset|.
  template <typename DataTypeT>
  DataTypeT GetParameterValue(int byte_offset) const {
    DataTypeT out_data;
    buffer_.Read(byte_offset, &out_data, sizeof(DataTypeT));
    return out_data;
  }

  // Sets a parameter value on a given |byte_offset|.
  template <typename DataTypeT>
  void SetParameterValue(int byte_offset, const DataTypeT &in_data) {
    if (byte_offset + sizeof(DataTypeT) > buffer_.data_size()) {
      buffer_.Resize(byte_offset + sizeof(DataTypeT));
    }
    buffer_.Write(byte_offset, &in_data, sizeof(DataTypeT));
  }

  // Sets a parameter value at the end of the |buffer_|.
  template <typename DataTypeT>
  void AppendParameterValue(const DataTypeT &in_data) {
    SetParameterValue(static_cast<int>(buffer_.data_size()), in_data);
  }

 private:
  AttributeTransformType transform_type_;
  DataBuffer buffer_;
};

}  // namespace draco

#endif  // DRACO_ATTRIBUTES_ATTRIBUTE_TRANSFORM_DATA_H_
