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

#include "minidump/test/minidump_byte_array_writer_test_util.h"

#include "minidump/test/minidump_writable_test_util.h"

namespace crashpad {
namespace test {

std::vector<uint8_t> MinidumpByteArrayAtRVA(const std::string& file_contents,
                                            RVA rva) {
  auto* minidump_byte_array =
      MinidumpWritableAtRVA<MinidumpByteArray>(file_contents, rva);
  if (!minidump_byte_array) {
    return {};
  }
  auto* data = static_cast<const uint8_t*>(minidump_byte_array->data);
  const uint8_t* data_end = data + minidump_byte_array->length;
  return std::vector<uint8_t>(data, data_end);
}


}  // namespace test
}  // namespace crashpad
