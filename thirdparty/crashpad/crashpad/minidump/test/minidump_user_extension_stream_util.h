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

#ifndef CRASHPAD_MINIDUMP_TEST_MINIDUMP_USER_EXTENSION_STREAM_UTIL_H_
#define CRASHPAD_MINIDUMP_TEST_MINIDUMP_USER_EXTENSION_STREAM_UTIL_H_

#include "minidump/minidump_user_extension_stream_data_source.h"

#include <stdint.h>
#include <sys/types.h>

#include <vector>

namespace crashpad {
namespace test {

//! \brief A user extension data source that wraps a buffer.
class BufferExtensionStreamDataSource final
    : public MinidumpUserExtensionStreamDataSource {
 public:
  //! \brief Creates a data source with \a stream_type.
  //!
  //! param[in] stream_type The type of the stream.
  //! param[in] data The data of the stream.
  //! param[in] data_size The length of \a data.
  BufferExtensionStreamDataSource(uint32_t stream_type,
                                  const void* data,
                                  size_t data_size);

  size_t StreamDataSize() override;
  bool ReadStreamData(Delegate* delegate) override;

 private:
  std::vector<uint8_t> data_;

  DISALLOW_COPY_AND_ASSIGN(BufferExtensionStreamDataSource);
};

}  // namespace test
}  // namespace crashpad

#endif  // CRASHPAD_MINIDUMP_TEST_MINIDUMP_USER_EXTENSION_STREAM_UTIL_H_
