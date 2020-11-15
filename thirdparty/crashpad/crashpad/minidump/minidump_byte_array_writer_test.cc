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

#include "minidump/minidump_byte_array_writer.h"

#include <memory>

#include "base/format_macros.h"
#include "base/strings/stringprintf.h"
#include "gtest/gtest.h"
#include "minidump/test/minidump_writable_test_util.h"
#include "util/file/string_file.h"

namespace crashpad {
namespace test {
namespace {

TEST(MinidumpByteArrayWriter, Write) {
  const std::vector<uint8_t> kTests[] = {
      {'h', 'e', 'l', 'l', 'o'},
      {0x42, 0x99, 0x00, 0xbe},
      {0x00},
      {},
  };

  for (size_t i = 0; i < arraysize(kTests); ++i) {
    SCOPED_TRACE(base::StringPrintf("index %" PRIuS, i));

    StringFile string_file;

    crashpad::MinidumpByteArrayWriter writer;
    writer.set_data(kTests[i]);
    EXPECT_TRUE(writer.WriteEverything(&string_file));

    ASSERT_EQ(string_file.string().size(),
              sizeof(MinidumpByteArray) + kTests[i].size());

    auto byte_array = std::make_unique<MinidumpByteArray>();
    EXPECT_EQ(string_file.Seek(0, SEEK_SET), 0);
    string_file.Read(byte_array.get(), sizeof(*byte_array));

    EXPECT_EQ(byte_array->length, kTests[i].size());

    std::vector<uint8_t> data(byte_array->length);
    string_file.Read(data.data(), byte_array->length);

    EXPECT_EQ(data, kTests[i]);
  }
}

TEST(MinidumpByteArrayWriter, SetData) {
  const std::vector<uint8_t> kTests[] = {
    {1, 2, 3, 4, 5},
    {0x0},
    {},
  };

  for (size_t i = 0; i < arraysize(kTests); ++i) {
    SCOPED_TRACE(base::StringPrintf("index %" PRIuS, i));

    crashpad::MinidumpByteArrayWriter writer;
    writer.set_data(kTests[i].data(), kTests[i].size());
    EXPECT_EQ(writer.data(), kTests[i]);
  }
}

}  // namespace
}  // namespace test
}  // namespace crashpad
