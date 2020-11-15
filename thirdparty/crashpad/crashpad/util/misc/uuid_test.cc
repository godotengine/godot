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

#include "util/misc/uuid.h"

#include <string.h>
#include <sys/types.h>

#include <string>

#include "base/format_macros.h"
#include "base/macros.h"
#include "base/scoped_generic.h"
#include "base/strings/stringprintf.h"
#include "gtest/gtest.h"

namespace crashpad {
namespace test {
namespace {

TEST(UUID, UUID) {
  UUID uuid_zero;
  uuid_zero.InitializeToZero();
  EXPECT_EQ(uuid_zero.data_1, 0u);
  EXPECT_EQ(uuid_zero.data_2, 0u);
  EXPECT_EQ(uuid_zero.data_3, 0u);
  EXPECT_EQ(uuid_zero.data_4[0], 0u);
  EXPECT_EQ(uuid_zero.data_4[1], 0u);
  EXPECT_EQ(uuid_zero.data_5[0], 0u);
  EXPECT_EQ(uuid_zero.data_5[1], 0u);
  EXPECT_EQ(uuid_zero.data_5[2], 0u);
  EXPECT_EQ(uuid_zero.data_5[3], 0u);
  EXPECT_EQ(uuid_zero.data_5[4], 0u);
  EXPECT_EQ(uuid_zero.data_5[5], 0u);
  EXPECT_EQ(uuid_zero.ToString(), "00000000-0000-0000-0000-000000000000");

  static constexpr uint8_t kBytes[16] = {0x00,
                                         0x01,
                                         0x02,
                                         0x03,
                                         0x04,
                                         0x05,
                                         0x06,
                                         0x07,
                                         0x08,
                                         0x09,
                                         0x0a,
                                         0x0b,
                                         0x0c,
                                         0x0d,
                                         0x0e,
                                         0x0f};
  UUID uuid;
  uuid.InitializeFromBytes(kBytes);
  EXPECT_EQ(uuid.data_1, 0x00010203u);
  EXPECT_EQ(uuid.data_2, 0x0405u);
  EXPECT_EQ(uuid.data_3, 0x0607u);
  EXPECT_EQ(uuid.data_4[0], 0x08u);
  EXPECT_EQ(uuid.data_4[1], 0x09u);
  EXPECT_EQ(uuid.data_5[0], 0x0au);
  EXPECT_EQ(uuid.data_5[1], 0x0bu);
  EXPECT_EQ(uuid.data_5[2], 0x0cu);
  EXPECT_EQ(uuid.data_5[3], 0x0du);
  EXPECT_EQ(uuid.data_5[4], 0x0eu);
  EXPECT_EQ(uuid.data_5[5], 0x0fu);
  EXPECT_EQ(uuid.ToString(), "00010203-0405-0607-0809-0a0b0c0d0e0f");

  // Test both operator== and operator!=.
  EXPECT_FALSE(uuid == uuid_zero);
  EXPECT_NE(uuid, uuid_zero);

  UUID uuid_2;
  uuid_2.InitializeFromBytes(kBytes);
  EXPECT_EQ(uuid_2, uuid);
  EXPECT_FALSE(uuid != uuid_2);

  // Make sure that operator== and operator!= check the entire UUID.
  ++uuid.data_1;
  EXPECT_NE(uuid, uuid_2);
  --uuid.data_1;
  ++uuid.data_2;
  EXPECT_NE(uuid, uuid_2);
  --uuid.data_2;
  ++uuid.data_3;
  EXPECT_NE(uuid, uuid_2);
  --uuid.data_3;
  for (size_t index = 0; index < arraysize(uuid.data_4); ++index) {
    ++uuid.data_4[index];
    EXPECT_NE(uuid, uuid_2);
    --uuid.data_4[index];
  }
  for (size_t index = 0; index < arraysize(uuid.data_5); ++index) {
    ++uuid.data_5[index];
    EXPECT_NE(uuid, uuid_2);
    --uuid.data_5[index];
  }

  // Make sure that the UUIDs are equal again, otherwise the test above may not
  // have been valid.
  EXPECT_EQ(uuid_2, uuid);

  static constexpr uint8_t kMoreBytes[16] = {0xff,
                                             0xee,
                                             0xdd,
                                             0xcc,
                                             0xbb,
                                             0xaa,
                                             0x99,
                                             0x88,
                                             0x77,
                                             0x66,
                                             0x55,
                                             0x44,
                                             0x33,
                                             0x22,
                                             0x11,
                                             0x00};
  uuid.InitializeFromBytes(kMoreBytes);
  EXPECT_EQ(uuid.data_1, 0xffeeddccu);
  EXPECT_EQ(uuid.data_2, 0xbbaau);
  EXPECT_EQ(uuid.data_3, 0x9988u);
  EXPECT_EQ(uuid.data_4[0], 0x77u);
  EXPECT_EQ(uuid.data_4[1], 0x66u);
  EXPECT_EQ(uuid.data_5[0], 0x55u);
  EXPECT_EQ(uuid.data_5[1], 0x44u);
  EXPECT_EQ(uuid.data_5[2], 0x33u);
  EXPECT_EQ(uuid.data_5[3], 0x22u);
  EXPECT_EQ(uuid.data_5[4], 0x11u);
  EXPECT_EQ(uuid.data_5[5], 0x00u);
  EXPECT_EQ(uuid.ToString(), "ffeeddcc-bbaa-9988-7766-554433221100");

  EXPECT_NE(uuid, uuid_2);
  EXPECT_NE(uuid, uuid_zero);

  // Test that UUID is standard layout.
  memset(&uuid, 0x45, 16);
  EXPECT_EQ(uuid.data_1, 0x45454545u);
  EXPECT_EQ(uuid.data_2, 0x4545u);
  EXPECT_EQ(uuid.data_3, 0x4545u);
  EXPECT_EQ(uuid.data_4[0], 0x45u);
  EXPECT_EQ(uuid.data_4[1], 0x45u);
  EXPECT_EQ(uuid.data_5[0], 0x45u);
  EXPECT_EQ(uuid.data_5[1], 0x45u);
  EXPECT_EQ(uuid.data_5[2], 0x45u);
  EXPECT_EQ(uuid.data_5[3], 0x45u);
  EXPECT_EQ(uuid.data_5[4], 0x45u);
  EXPECT_EQ(uuid.data_5[5], 0x45u);
  EXPECT_EQ(uuid.ToString(), "45454545-4545-4545-4545-454545454545");

  UUID initialized_generated;
  initialized_generated.InitializeWithNew();
  EXPECT_NE(initialized_generated, uuid_zero);
}

TEST(UUID, FromString) {
  static constexpr struct TestCase {
    const char* uuid_string;
    bool success;
  } kCases[] = {
    // Valid:
    {"c6849cb5-fe14-4a79-8978-9ae6034c521d", true},
    {"00000000-0000-0000-0000-000000000000", true},
    {"ffffffff-ffff-ffff-ffff-ffffffffffff", true},
    // Outside HEX range:
    {"7318z10b-c453-4cef-9dc8-015655cb4bbc", false},
    {"7318a10b-c453-4cef-9dz8-015655cb4bbc", false},
    // Incomplete:
    {"15655cb4-", false},
    {"7318f10b-c453-4cef-9dc8-015655cb4bb", false},
    {"318f10b-c453-4cef-9dc8-015655cb4bb2", false},
    {"7318f10b-c453-4ef-9dc8-015655cb4bb2", false},
    {"", false},
    {"abcd", false},
    // Trailing data:
    {"6d247a34-53d5-40ec-a90d-d8dea9e94cc01", false}
  };

  UUID uuid_zero;
  uuid_zero.InitializeToZero();
  const std::string empty_uuid = uuid_zero.ToString();

  for (size_t index = 0; index < arraysize(kCases); ++index) {
    const TestCase& test_case = kCases[index];
    SCOPED_TRACE(base::StringPrintf(
        "index %" PRIuS ": %s", index, test_case.uuid_string));

    UUID uuid;
    uuid.InitializeToZero();
    EXPECT_EQ(uuid.InitializeFromString(test_case.uuid_string),
              test_case.success);
    if (test_case.success) {
      EXPECT_EQ(uuid.ToString(), test_case.uuid_string);
    } else {
      EXPECT_EQ(uuid.ToString(), empty_uuid);
    }
  }

  // Test for case insensitivty.
  UUID uuid;
  uuid.InitializeFromString("F32E5BDC-2681-4C73-A4E6-911FFD89B846");
  EXPECT_EQ(uuid.ToString(), "f32e5bdc-2681-4c73-a4e6-911ffd89b846");

  // Mixed case.
  uuid.InitializeFromString("5762C15D-50b5-4171-a2e9-7429C9EC6CAB");
  EXPECT_EQ(uuid.ToString(), "5762c15d-50b5-4171-a2e9-7429c9ec6cab");

  // Test accepting a StringPiece16.
  // clang-format off
  static constexpr base::char16 kChar16UUID[] = {
      'f', '3', '2', 'e', '5', 'b', 'd', 'c', '-',
      '2', '6', '8', '1', '-',
      '4', 'c', '7', '3', '-',
      'a', '4', 'e', '6', '-',
      '3', '3', '3', 'f', 'f', 'd', '3', '3', 'b', '3', '3', '3',
  };
  // clang-format on
  EXPECT_TRUE(uuid.InitializeFromString(
      base::StringPiece16(kChar16UUID, arraysize(kChar16UUID))));
  EXPECT_EQ(uuid.ToString(), "f32e5bdc-2681-4c73-a4e6-333ffd33b333");

#if defined(OS_WIN)
  // Test accepting a StringPiece16 via L"" literals on Windows.
  EXPECT_TRUE(
      uuid.InitializeFromString(L"F32E5BDC-2681-4C73-A4E6-444FFD44B444"));
  EXPECT_EQ(uuid.ToString(), "f32e5bdc-2681-4c73-a4e6-444ffd44b444");

  EXPECT_TRUE(
      uuid.InitializeFromString(L"5762C15D-50b5-4171-a2e9-5555C5EC5CAB"));
  EXPECT_EQ(uuid.ToString(), "5762c15d-50b5-4171-a2e9-5555c5ec5cab");
#endif  // OS_WIN
}

#if defined(OS_WIN)

TEST(UUID, FromSystem) {
  ::GUID system_uuid;
  ASSERT_EQ(UuidCreate(&system_uuid), RPC_S_OK);

  UUID uuid;
  uuid.InitializeFromSystemUUID(&system_uuid);

  RPC_WSTR system_string;
  ASSERT_EQ(UuidToString(&system_uuid, &system_string), RPC_S_OK);

  struct ScopedRpcStringFreeTraits {
    static RPC_WSTR* InvalidValue() { return nullptr; }
    static void Free(RPC_WSTR* rpc_string) {
      EXPECT_EQ(RpcStringFree(rpc_string), RPC_S_OK);
    }
  };
  using ScopedRpcString =
      base::ScopedGeneric<RPC_WSTR*, ScopedRpcStringFreeTraits>;
  ScopedRpcString scoped_system_string(&system_string);

  EXPECT_EQ(uuid.ToString16(), reinterpret_cast<wchar_t*>(system_string));
}

#endif  // OS_WIN

}  // namespace
}  // namespace test
}  // namespace crashpad
