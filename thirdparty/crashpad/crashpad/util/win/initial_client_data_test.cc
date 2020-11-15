// Copyright 2016 The Crashpad Authors. All rights reserved.
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

#include "util/win/initial_client_data.h"

#include "gtest/gtest.h"

namespace crashpad {
namespace test {
namespace {

TEST(InitialClientData, Validity) {
  InitialClientData icd1;
  EXPECT_FALSE(icd1.IsValid());

  InitialClientData icd2(
      reinterpret_cast<HANDLE>(0x123),
      reinterpret_cast<HANDLE>(0x456),
      reinterpret_cast<HANDLE>(0x789),
      reinterpret_cast<HANDLE>(0xabc),
      reinterpret_cast<HANDLE>(0xdef),
      0x7fff000012345678ull,
      0x100000ull,
      0xccccddddeeeeffffull);
  EXPECT_TRUE(icd2.IsValid());
}

TEST(InitialClientData, RoundTrip) {
  InitialClientData first(
      reinterpret_cast<HANDLE>(0x123),
      reinterpret_cast<HANDLE>(0x456),
      reinterpret_cast<HANDLE>(0x789),
      reinterpret_cast<HANDLE>(0xabc),
      reinterpret_cast<HANDLE>(0xdef),
      0x7fff000012345678ull,
      0x100000ull,
      0xccccddddeeeeffffull);

  std::string as_string = first.StringRepresentation();
  EXPECT_EQ(as_string,
            "0x123,0x456,0x789,0xabc,0xdef,"
            "0x7fff000012345678,0x100000,0xccccddddeeeeffff");

  InitialClientData second;
  ASSERT_TRUE(second.InitializeFromString(as_string));
  EXPECT_EQ(second.request_crash_dump(), first.request_crash_dump());
  EXPECT_EQ(second.request_non_crash_dump(), first.request_non_crash_dump());
  EXPECT_EQ(second.non_crash_dump_completed(),
            first.non_crash_dump_completed());
  EXPECT_EQ(second.first_pipe_instance(), first.first_pipe_instance());
  EXPECT_EQ(second.client_process(), first.client_process());
  EXPECT_EQ(second.crash_exception_information(),
            first.crash_exception_information());
  EXPECT_EQ(second.non_crash_exception_information(),
            first.non_crash_exception_information());
  EXPECT_EQ(second.debug_critical_section_address(),
            first.debug_critical_section_address());
}

}  // namespace
}  // namespace test
}  // namespace crashpad
