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

#include "minidump/test/minidump_file_writer_test_util.h"

#include "gtest/gtest.h"
#include "minidump/test/minidump_writable_test_util.h"

namespace crashpad {
namespace test {

const MINIDUMP_HEADER* MinidumpHeaderAtStart(
    const std::string& file_contents,
    const MINIDUMP_DIRECTORY** directory) {
  MINIDUMP_LOCATION_DESCRIPTOR location_descriptor;
  location_descriptor.DataSize = sizeof(MINIDUMP_HEADER);
  location_descriptor.Rva = 0;

  const MINIDUMP_HEADER* header =
      MinidumpWritableAtLocationDescriptor<MINIDUMP_HEADER>(
          file_contents, location_descriptor);

  if (header) {
    location_descriptor.DataSize =
        header->NumberOfStreams * sizeof(MINIDUMP_DIRECTORY);
    location_descriptor.Rva = header->StreamDirectoryRva;
    *directory = MinidumpWritableAtLocationDescriptor<MINIDUMP_DIRECTORY>(
        file_contents, location_descriptor);
  } else {
    *directory = nullptr;
  }

  return header;
}

void VerifyMinidumpHeader(const MINIDUMP_HEADER* header,
                          uint32_t streams,
                          uint32_t timestamp) {
  ASSERT_TRUE(header);
  ASSERT_EQ(header->NumberOfStreams, streams);
  ASSERT_EQ(header->StreamDirectoryRva, streams ? sizeof(MINIDUMP_HEADER) : 0u);
  EXPECT_EQ(header->CheckSum, 0u);
  EXPECT_EQ(header->TimeDateStamp, timestamp);
  EXPECT_EQ(static_cast<MINIDUMP_TYPE>(header->Flags), MiniDumpNormal);
}

}  // namespace test
}  // namespace crashpad
