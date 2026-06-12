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
#include "draco/core/data_buffer.h"

#include <algorithm>

namespace draco {

DataBuffer::DataBuffer() {}

bool DataBuffer::Update(const void *data, int64_t size) {
  const int64_t offset = 0;
  return this->Update(data, size, offset);
}

bool DataBuffer::Update(const void *data, int64_t size, int64_t offset) {
  if (data == nullptr) {
    if (size + offset < 0) {
      return false;
    }
    // If no data is provided, just resize the buffer.
    data_.resize(size + offset);
  } else {
    if (size < 0) {
      return false;
    }
    if (size + offset > static_cast<int64_t>(data_.size())) {
      data_.resize(size + offset);
    }
    const uint8_t *const byte_data = static_cast<const uint8_t *>(data);
    std::copy(byte_data, byte_data + size, data_.data() + offset);
  }
  descriptor_.buffer_update_count++;
  return true;
}

void DataBuffer::Resize(int64_t size) {
  data_.resize(size);
  descriptor_.buffer_update_count++;
}

void DataBuffer::WriteDataToStream(std::ostream &stream) {
  if (data_.empty()) {
    return;
  }
  stream.write(reinterpret_cast<char *>(data_.data()), data_.size());
}

}  // namespace draco
