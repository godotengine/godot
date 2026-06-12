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
#ifndef DRACO_CORE_DATA_BUFFER_H_
#define DRACO_CORE_DATA_BUFFER_H_

#include <cstring>
#include <ostream>
#include <vector>

#include "draco/core/draco_types.h"

namespace draco {

// Buffer descriptor servers as a unique identifier of a buffer.
struct DataBufferDescriptor {
  DataBufferDescriptor() : buffer_id(0), buffer_update_count(0) {}
  // Id of the data buffer.
  int64_t buffer_id;
  // The number of times the buffer content was updated.
  int64_t buffer_update_count;
};

// Class used for storing raw buffer data.
class DataBuffer {
 public:
  DataBuffer();
  bool Update(const void *data, int64_t size);
  bool Update(const void *data, int64_t size, int64_t offset);

  // Reallocate the buffer storage to a new size keeping the data unchanged.
  void Resize(int64_t new_size);
  void WriteDataToStream(std::ostream &stream);
  // Reads data from the buffer. Potentially unsafe, called needs to ensure
  // the accessed memory is valid.
  void Read(int64_t byte_pos, void *out_data, size_t data_size) const {
    memcpy(out_data, data() + byte_pos, data_size);
  }

  // Writes data to the buffer. Unsafe, caller must ensure the accessed memory
  // is valid.
  void Write(int64_t byte_pos, const void *in_data, size_t data_size) {
    memcpy(const_cast<uint8_t *>(data()) + byte_pos, in_data, data_size);
  }

  // Copies data from another buffer to this buffer.
  void Copy(int64_t dst_offset, const DataBuffer *src_buf, int64_t src_offset,
            int64_t size) {
    memcpy(const_cast<uint8_t *>(data()) + dst_offset,
           src_buf->data() + src_offset, size);
  }

  void set_update_count(int64_t buffer_update_count) {
    descriptor_.buffer_update_count = buffer_update_count;
  }
  int64_t update_count() const { return descriptor_.buffer_update_count; }
  size_t data_size() const { return data_.size(); }
  const uint8_t *data() const { return data_.data(); }
  uint8_t *data() { return data_.data(); }
  int64_t buffer_id() const { return descriptor_.buffer_id; }
  void set_buffer_id(int64_t buffer_id) { descriptor_.buffer_id = buffer_id; }

 private:
  std::vector<uint8_t> data_;
  // Counter incremented by Update() calls.
  DataBufferDescriptor descriptor_;
};

}  // namespace draco

#endif  // DRACO_CORE_DATA_BUFFER_H_
