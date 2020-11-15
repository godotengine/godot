// Copyright 2017 The Crashpad Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "minidump/test/minidump_user_extension_stream_util.h"

#include <string.h>

namespace crashpad {
namespace test {

BufferExtensionStreamDataSource::BufferExtensionStreamDataSource(
    uint32_t stream_type,
    const void* data,
    size_t data_size)
    : MinidumpUserExtensionStreamDataSource(stream_type) {
  data_.resize(data_size);

  if (data_size)
    memcpy(data_.data(), data, data_size);
}

size_t BufferExtensionStreamDataSource::StreamDataSize() {
  return data_.size();
}

bool BufferExtensionStreamDataSource::ReadStreamData(Delegate* delegate) {
  return delegate->ExtensionStreamDataSourceRead(
      data_.size() ? data_.data() : nullptr, data_.size());
}

}  // namespace test
}  // namespace crashpad
