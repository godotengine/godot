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
#include "draco/attributes/geometry_attribute.h"

namespace draco {

GeometryAttribute::GeometryAttribute()
    : buffer_(nullptr),
      num_components_(1),
      data_type_(DT_FLOAT32),
      byte_stride_(0),
      byte_offset_(0),
      attribute_type_(INVALID),
      unique_id_(0) {}

void GeometryAttribute::Init(GeometryAttribute::Type attribute_type,
                             DataBuffer *buffer, uint8_t num_components,
                             DataType data_type, bool normalized,
                             int64_t byte_stride, int64_t byte_offset) {
  buffer_ = buffer;
  if (buffer) {
    buffer_descriptor_.buffer_id = buffer->buffer_id();
    buffer_descriptor_.buffer_update_count = buffer->update_count();
  }
  num_components_ = num_components;
  data_type_ = data_type;
  normalized_ = normalized;
  byte_stride_ = byte_stride;
  byte_offset_ = byte_offset;
  attribute_type_ = attribute_type;
}

bool GeometryAttribute::CopyFrom(const GeometryAttribute &src_att) {
  num_components_ = src_att.num_components_;
  data_type_ = src_att.data_type_;
  normalized_ = src_att.normalized_;
  byte_stride_ = src_att.byte_stride_;
  byte_offset_ = src_att.byte_offset_;
  attribute_type_ = src_att.attribute_type_;
  buffer_descriptor_ = src_att.buffer_descriptor_;
  unique_id_ = src_att.unique_id_;
  if (src_att.buffer_ == nullptr) {
    buffer_ = nullptr;
  } else {
    if (buffer_ == nullptr) {
      return false;
    }
    buffer_->Update(src_att.buffer_->data(), src_att.buffer_->data_size());
  }
#ifdef DRACO_TRANSCODER_SUPPORTED
  name_ = src_att.name_;
#endif
  return true;
}

bool GeometryAttribute::operator==(const GeometryAttribute &va) const {
  if (attribute_type_ != va.attribute_type_) {
    return false;
  }
  // It's OK to compare just the buffer descriptors here. We don't need to
  // compare the buffers themselves.
  if (buffer_descriptor_.buffer_id != va.buffer_descriptor_.buffer_id) {
    return false;
  }
  if (buffer_descriptor_.buffer_update_count !=
      va.buffer_descriptor_.buffer_update_count) {
    return false;
  }
  if (num_components_ != va.num_components_) {
    return false;
  }
  if (data_type_ != va.data_type_) {
    return false;
  }
  if (byte_stride_ != va.byte_stride_) {
    return false;
  }
  if (byte_offset_ != va.byte_offset_) {
    return false;
  }
#ifdef DRACO_TRANSCODER_SUPPORTED
  if (name_ != va.name_) {
    return false;
  }
#endif
  return true;
}

void GeometryAttribute::ResetBuffer(DataBuffer *buffer, int64_t byte_stride,
                                    int64_t byte_offset) {
  buffer_ = buffer;
  buffer_descriptor_.buffer_id = buffer->buffer_id();
  buffer_descriptor_.buffer_update_count = buffer->update_count();
  byte_stride_ = byte_stride;
  byte_offset_ = byte_offset;
}

}  // namespace draco
