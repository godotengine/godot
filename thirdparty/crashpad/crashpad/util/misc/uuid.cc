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

#if !defined(__STDC_FORMAT_MACROS)
#define __STDC_FORMAT_MACROS
#endif

#include "util/misc/uuid.h"

#include <inttypes.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include <type_traits>

#include "base/logging.h"
#include "base/rand_util.h"
#include "base/strings/stringprintf.h"
#include "base/strings/utf_string_conversions.h"
#include "base/sys_byteorder.h"

#if defined(OS_MACOSX)
#include <uuid/uuid.h>
#endif  // OS_MACOSX

namespace crashpad {

static_assert(sizeof(UUID) == 16, "UUID must be 16 bytes");
static_assert(std::is_pod<UUID>::value, "UUID must be POD");

bool UUID::operator==(const UUID& that) const {
  return memcmp(this, &that, sizeof(*this)) == 0;
}

void UUID::InitializeToZero() {
  memset(this, 0, sizeof(*this));
}

void UUID::InitializeFromBytes(const uint8_t* bytes) {
  memcpy(this, bytes, sizeof(*this));
  data_1 = base::NetToHost32(data_1);
  data_2 = base::NetToHost16(data_2);
  data_3 = base::NetToHost16(data_3);
}

bool UUID::InitializeFromString(const base::StringPiece& string) {
  if (string.length() != 36)
    return false;

  UUID temp;
  static constexpr char kScanFormat[] =
      "%08" SCNx32 "-%04" SCNx16 "-%04" SCNx16
      "-%02" SCNx8 "%02" SCNx8
      "-%02" SCNx8 "%02" SCNx8 "%02" SCNx8 "%02" SCNx8 "%02" SCNx8 "%02" SCNx8;
  int rv = sscanf(string.data(),
                  kScanFormat,
                  &temp.data_1,
                  &temp.data_2,
                  &temp.data_3,
                  &temp.data_4[0],
                  &temp.data_4[1],
                  &temp.data_5[0],
                  &temp.data_5[1],
                  &temp.data_5[2],
                  &temp.data_5[3],
                  &temp.data_5[4],
                  &temp.data_5[5]);
  if (rv != 11)
    return false;

  *this = temp;
  return true;
}

bool UUID::InitializeFromString(const base::StringPiece16& string) {
  return InitializeFromString(UTF16ToUTF8(string));
}

bool UUID::InitializeWithNew() {
#if defined(OS_MACOSX)
  uuid_t uuid;
  uuid_generate(uuid);
  InitializeFromBytes(uuid);
  return true;
#elif defined(OS_WIN) || defined(OS_LINUX) || defined(OS_ANDROID) || \
    defined(OS_FUCHSIA)
  // Linux, Android, and Fuchsia do not provide a UUID generator in a
  // widely-available system library. On Linux and Android, uuid_generate()
  // from libuuid is not available everywhere.
  // On Windows, do not use UuidCreate() to avoid a dependency on rpcrt4, so
  // that this function is usable early in DllMain().
  base::RandBytes(this, sizeof(*this));

  // Set six bits per RFC 4122 §4.4 to identify this as a pseudo-random UUID.
  data_3 = (4 << 12) | (data_3 & 0x0fff);  // §4.1.3
  data_4[0] = 0x80 | (data_4[0] & 0x3f);  // §4.1.1

  return true;
#else
#error Port.
#endif  // OS_MACOSX
}

#if defined(OS_WIN)
void UUID::InitializeFromSystemUUID(const ::UUID* system_uuid) {
  static_assert(sizeof(::UUID) == sizeof(UUID),
                "unexpected system uuid size");
  static_assert(offsetof(::UUID, Data1) == offsetof(UUID, data_1),
                "unexpected system uuid layout");
  memcpy(this, system_uuid, sizeof(*this));
}
#endif  // OS_WIN

std::string UUID::ToString() const {
  return base::StringPrintf("%08x-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                            data_1,
                            data_2,
                            data_3,
                            data_4[0],
                            data_4[1],
                            data_5[0],
                            data_5[1],
                            data_5[2],
                            data_5[3],
                            data_5[4],
                            data_5[5]);
}

#if defined(OS_WIN)
base::string16 UUID::ToString16() const {
  return base::UTF8ToUTF16(ToString());
}
#endif  // OS_WIN

}  // namespace crashpad
