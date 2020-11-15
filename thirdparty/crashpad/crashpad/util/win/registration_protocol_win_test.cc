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

#include "util/win/registration_protocol_win.h"

#include <windows.h>
#include <sddl.h>
#include <string.h>

#include "gtest/gtest.h"
#include "test/errors.h"
#include "util/win/scoped_local_alloc.h"

namespace crashpad {
namespace test {
namespace {

TEST(SecurityDescriptor, MatchesAdvapi32) {
  // This security descriptor is built manually in the connection code to avoid
  // calling the advapi32 functions. Verify that it returns the same thing as
  // ConvertStringSecurityDescriptorToSecurityDescriptor() would.

  // Mandatory Label, no ACE flags, no ObjectType, integrity level
  // untrusted.
  static constexpr wchar_t kSddl[] = L"S:(ML;;;;;S-1-16-0)";
  PSECURITY_DESCRIPTOR sec_desc;
  ULONG sec_desc_len;
  ASSERT_TRUE(ConvertStringSecurityDescriptorToSecurityDescriptor(
      kSddl, SDDL_REVISION_1, &sec_desc, &sec_desc_len))
      << ErrorMessage("ConvertStringSecurityDescriptorToSecurityDescriptor");
  ScopedLocalAlloc sec_desc_owner(sec_desc);

  size_t created_len;
  const void* const created =
      GetSecurityDescriptorForNamedPipeInstance(&created_len);
  ASSERT_EQ(created_len, sec_desc_len);
  EXPECT_EQ(memcmp(sec_desc, created, sec_desc_len), 0);
}

}  // namespace
}  // namespace test
}  // namespace crashpad
