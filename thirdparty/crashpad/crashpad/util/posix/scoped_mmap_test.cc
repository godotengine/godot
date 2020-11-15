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

#include "util/posix/scoped_mmap.h"

#include <stdint.h>
#include <sys/types.h>
#include <unistd.h>

#include "base/numerics/safe_conversions.h"
#include "base/rand_util.h"
#include "base/strings/stringprintf.h"
#include "gtest/gtest.h"
#include "test/gtest_death.h"

namespace crashpad {
namespace test {
namespace {

bool ScopedMmapResetMmap(ScopedMmap* mapping, size_t len) {
  return mapping->ResetMmap(
      nullptr, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
}

void* BareMmap(size_t len) {
  return mmap(
      nullptr, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
}

// A weird class. This is used to test that memory-mapped regions are freed
// as expected by calling munmap(). This is difficult to test well because once
// a region has been unmapped, the address space it formerly occupied becomes
// eligible for reuse.
//
// The strategy taken here is that a random 64-bit cookie value is written into
// a mapped region by SetUp(). While the mapping is active, Check() should not
// crash, or for a gtest expectation, Expected() and Observed() should not crash
// and should be equal. After the region is unmapped, Check() should crash,
// either because the region has been unmapped and the address not reused, the
// address has been reused but is protected against reading (unlikely), or
// because the address has been reused but the cookie value is no longer present
// there.
class TestCookie {
 public:
  // A weird constructor for a weird class. The member variable initialization
  // assures that Check() won’t crash if called on an object that hasn’t had
  // SetUp() called on it.
  explicit TestCookie() : address_(&cookie_), cookie_(0) {}

  ~TestCookie() {}

  void SetUp(uint64_t* address) {
    address_ = address, cookie_ = base::RandUint64();
    *address_ = cookie_;
  }

  uint64_t Expected() const { return cookie_; }
  uint64_t Observed() const { return *address_; }

  void Check() const {
    if (Observed() != Expected()) {
      __builtin_trap();
    }
  }

 private:
  uint64_t* address_;
  uint64_t cookie_;

  DISALLOW_COPY_AND_ASSIGN(TestCookie);
};

TEST(ScopedMmap, Mmap) {
  TestCookie cookie;

  ScopedMmap mapping;
  EXPECT_FALSE(mapping.is_valid());
  EXPECT_EQ(mapping.addr(), MAP_FAILED);
  EXPECT_EQ(mapping.len(), 0u);

  ASSERT_TRUE(mapping.Reset());
  EXPECT_FALSE(mapping.is_valid());

  const size_t kPageSize = base::checked_cast<size_t>(getpagesize());
  ASSERT_TRUE(ScopedMmapResetMmap(&mapping, kPageSize));
  EXPECT_TRUE(mapping.is_valid());
  EXPECT_NE(mapping.addr(), MAP_FAILED);
  EXPECT_EQ(mapping.len(), kPageSize);

  cookie.SetUp(mapping.addr_as<uint64_t*>());
  EXPECT_EQ(cookie.Observed(), cookie.Expected());

  ASSERT_TRUE(mapping.Reset());
  EXPECT_FALSE(mapping.is_valid());
}

TEST(ScopedMmapDeathTest, Destructor) {
  TestCookie cookie;
  {
    ScopedMmap mapping;

    const size_t kPageSize = base::checked_cast<size_t>(getpagesize());
    ASSERT_TRUE(ScopedMmapResetMmap(&mapping, kPageSize));
    EXPECT_TRUE(mapping.is_valid());
    EXPECT_NE(mapping.addr(), MAP_FAILED);
    EXPECT_EQ(mapping.len(), kPageSize);

    cookie.SetUp(mapping.addr_as<uint64_t*>());
  }

  EXPECT_DEATH_CRASH(cookie.Check(), "");
}

TEST(ScopedMmapDeathTest, Reset) {
  ScopedMmap mapping;

  const size_t kPageSize = base::checked_cast<size_t>(getpagesize());
  ASSERT_TRUE(ScopedMmapResetMmap(&mapping, kPageSize));
  EXPECT_TRUE(mapping.is_valid());
  EXPECT_NE(mapping.addr(), MAP_FAILED);
  EXPECT_EQ(mapping.len(), kPageSize);

  TestCookie cookie;
  cookie.SetUp(mapping.addr_as<uint64_t*>());

  ASSERT_TRUE(mapping.Reset());

  EXPECT_DEATH_CRASH(cookie.Check(), "");
}

TEST(ScopedMmapDeathTest, ResetAddrLen_Shrink) {
  ScopedMmap mapping;

  // Start with three pages mapped.
  const size_t kPageSize = base::checked_cast<size_t>(getpagesize());
  ASSERT_TRUE(ScopedMmapResetMmap(&mapping, 3 * kPageSize));
  EXPECT_TRUE(mapping.is_valid());
  EXPECT_NE(mapping.addr(), MAP_FAILED);
  EXPECT_EQ(mapping.len(), 3 * kPageSize);

  TestCookie cookies[3];
  for (size_t index = 0; index < arraysize(cookies); ++index) {
    cookies[index].SetUp(reinterpret_cast<uint64_t*>(
        mapping.addr_as<uintptr_t>() + index * kPageSize));
  }

  // Reset to the second page. The first and third pages should be unmapped.
  void* const new_addr =
      reinterpret_cast<void*>(mapping.addr_as<uintptr_t>() + kPageSize);
  ASSERT_TRUE(mapping.ResetAddrLen(new_addr, kPageSize));
  EXPECT_TRUE(mapping.is_valid());
  EXPECT_EQ(mapping.addr(), new_addr);
  EXPECT_EQ(mapping.len(), kPageSize);

  EXPECT_EQ(cookies[1].Observed(), cookies[1].Expected());

  EXPECT_DEATH_CRASH(cookies[0].Check(), "");
  EXPECT_DEATH_CRASH(cookies[2].Check(), "");
}

TEST(ScopedMmap, ResetAddrLen_Grow) {
  // Start with three pages mapped, but ScopedMmap only aware of the the second
  // page.
  const size_t kPageSize = base::checked_cast<size_t>(getpagesize());
  void* pages = BareMmap(3 * kPageSize);
  ASSERT_NE(pages, MAP_FAILED);

  ScopedMmap mapping;
  void* const old_addr =
      reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(pages) + kPageSize);
  ASSERT_TRUE(mapping.ResetAddrLen(old_addr, kPageSize));
  EXPECT_TRUE(mapping.is_valid());
  EXPECT_EQ(mapping.addr(), old_addr);
  EXPECT_EQ(mapping.len(), kPageSize);

  TestCookie cookies[3];
  for (size_t index = 0; index < arraysize(cookies); ++index) {
    cookies[index].SetUp(reinterpret_cast<uint64_t*>(
        reinterpret_cast<uintptr_t>(pages) + index * kPageSize));
  }

  // Reset to all three pages. Nothing should be unmapped until destruction.
  ASSERT_TRUE(mapping.ResetAddrLen(pages, 3 * kPageSize));
  EXPECT_TRUE(mapping.is_valid());
  EXPECT_EQ(mapping.addr(), pages);
  EXPECT_EQ(mapping.len(), 3 * kPageSize);

  for (size_t index = 0; index < arraysize(cookies); ++index) {
    SCOPED_TRACE(base::StringPrintf("index %zu", index));
    EXPECT_EQ(cookies[index].Observed(), cookies[index].Expected());
  }
}

TEST(ScopedMmapDeathTest, ResetAddrLen_MoveDownAndGrow) {
  // Start with three pages mapped, but ScopedMmap only aware of the third page.
  const size_t kPageSize = base::checked_cast<size_t>(getpagesize());
  void* pages = BareMmap(3 * kPageSize);
  ASSERT_NE(pages, MAP_FAILED);

  ScopedMmap mapping;
  void* const old_addr = reinterpret_cast<void*>(
      reinterpret_cast<uintptr_t>(pages) + 2 * kPageSize);
  ASSERT_TRUE(mapping.ResetAddrLen(old_addr, kPageSize));
  EXPECT_TRUE(mapping.is_valid());
  EXPECT_EQ(mapping.addr(), old_addr);
  EXPECT_EQ(mapping.len(), kPageSize);

  TestCookie cookies[3];
  for (size_t index = 0; index < arraysize(cookies); ++index) {
    cookies[index].SetUp(reinterpret_cast<uint64_t*>(
        reinterpret_cast<uintptr_t>(pages) + index * kPageSize));
  }

  // Reset to the first two pages. The third page should be unmapped.
  ASSERT_TRUE(mapping.ResetAddrLen(pages, 2 * kPageSize));
  EXPECT_TRUE(mapping.is_valid());
  EXPECT_EQ(mapping.addr(), pages);
  EXPECT_EQ(mapping.len(), 2 * kPageSize);

  EXPECT_EQ(cookies[0].Observed(), cookies[0].Expected());
  EXPECT_EQ(cookies[1].Observed(), cookies[1].Expected());

  EXPECT_DEATH_CRASH(cookies[2].Check(), "");
}

TEST(ScopedMmapDeathTest, ResetAddrLen_MoveUpAndShrink) {
  // Start with three pages mapped, but ScopedMmap only aware of the first two
  // pages.
  const size_t kPageSize = base::checked_cast<size_t>(getpagesize());
  void* pages = BareMmap(3 * kPageSize);
  ASSERT_NE(pages, MAP_FAILED);

  ScopedMmap mapping;
  ASSERT_TRUE(mapping.ResetAddrLen(pages, 2 * kPageSize));
  EXPECT_TRUE(mapping.is_valid());
  EXPECT_EQ(mapping.addr(), pages);
  EXPECT_EQ(mapping.len(), 2 * kPageSize);

  TestCookie cookies[3];
  for (size_t index = 0; index < arraysize(cookies); ++index) {
    cookies[index].SetUp(reinterpret_cast<uint64_t*>(
        reinterpret_cast<uintptr_t>(pages) + index * kPageSize));
  }

  // Reset to the third page. The first two pages should be unmapped.
  void* const new_addr =
      reinterpret_cast<void*>(mapping.addr_as<uintptr_t>() + 2 * kPageSize);
  ASSERT_TRUE(mapping.ResetAddrLen(new_addr, kPageSize));
  EXPECT_TRUE(mapping.is_valid());
  EXPECT_EQ(mapping.addr(), new_addr);
  EXPECT_EQ(mapping.len(), kPageSize);

  EXPECT_EQ(cookies[2].Observed(), cookies[2].Expected());

  EXPECT_DEATH_CRASH(cookies[0].Check(), "");
  EXPECT_DEATH_CRASH(cookies[1].Check(), "");
}

TEST(ScopedMmapDeathTest, ResetMmap) {
  ScopedMmap mapping;

  // Calling ScopedMmap::ResetMmap() frees the existing mapping before
  // establishing the new one, so the new one may wind up at the same address as
  // the old. In fact, this is likely. Create a two-page mapping and replace it
  // with a single-page mapping, so that the test can assure that the second
  // page isn’t mapped after establishing the second mapping.
  const size_t kPageSize = base::checked_cast<size_t>(getpagesize());
  ASSERT_TRUE(ScopedMmapResetMmap(&mapping, 2 * kPageSize));
  EXPECT_TRUE(mapping.is_valid());
  EXPECT_NE(mapping.addr(), MAP_FAILED);
  EXPECT_EQ(mapping.len(), 2 * kPageSize);

  TestCookie cookie;
  cookie.SetUp(
      reinterpret_cast<uint64_t*>(mapping.addr_as<char*>() + kPageSize));

  ASSERT_TRUE(ScopedMmapResetMmap(&mapping, kPageSize));
  EXPECT_TRUE(mapping.is_valid());
  EXPECT_NE(mapping.addr(), MAP_FAILED);
  EXPECT_EQ(mapping.len(), kPageSize);

  EXPECT_DEATH_CRASH(cookie.Check(), "");
}

TEST(ScopedMmapDeathTest, Mprotect) {
  ScopedMmap mapping;

  const size_t kPageSize = base::checked_cast<size_t>(getpagesize());
  ASSERT_TRUE(ScopedMmapResetMmap(&mapping, kPageSize));
  EXPECT_TRUE(mapping.is_valid());
  EXPECT_NE(mapping.addr(), MAP_FAILED);
  EXPECT_EQ(mapping.len(), kPageSize);

  char* addr = mapping.addr_as<char*>();
  *addr = 1;

  ASSERT_TRUE(mapping.Mprotect(PROT_READ));

  EXPECT_DEATH_CRASH(*addr = 0, "");

  ASSERT_TRUE(mapping.Mprotect(PROT_READ | PROT_WRITE));
  EXPECT_EQ(*addr, 1);
  *addr = 2;
}

}  // namespace
}  // namespace test
}  // namespace crashpad
