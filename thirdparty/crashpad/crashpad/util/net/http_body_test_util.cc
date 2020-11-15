// Copyright 2014 The Crashpad Authors. All rights reserved.
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

#include "util/net/http_body_test_util.h"

#include <stdint.h>

#include <memory>

#include "gtest/gtest.h"
#include "util/file/file_io.h"
#include "util/net/http_body.h"

namespace crashpad {
namespace test {

std::string ReadStreamToString(HTTPBodyStream* stream) {
  return ReadStreamToString(stream, 32);
}

std::string ReadStreamToString(HTTPBodyStream* stream, size_t buffer_size) {
  std::unique_ptr<uint8_t[]> buf(new uint8_t[buffer_size]);
  std::string result;

  FileOperationResult bytes_read;
  while ((bytes_read = stream->GetBytesBuffer(buf.get(), buffer_size)) != 0) {
    if (bytes_read < 0) {
      ADD_FAILURE() << "Failed to read from stream: " << bytes_read;
      return std::string();
    }

    result.append(reinterpret_cast<char*>(buf.get()), bytes_read);
  }

  return result;
}

}  // namespace test
}  // namespace crashpad
