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

#include "util/linux/auxiliary_vector.h"

#include <linux/auxvec.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <limits>

#include "base/bit_cast.h"
#include "base/macros.h"
#include "build/build_config.h"
#include "gtest/gtest.h"
#include "test/errors.h"
#include "test/linux/fake_ptrace_connection.h"
#include "test/main_arguments.h"
#include "test/multiprocess.h"
#include "util/linux/address_types.h"
#include "util/linux/memory_map.h"
#include "util/misc/from_pointer_cast.h"
#include "util/numeric/int128.h"
#include "util/process/process_memory_linux.h"

#if !defined(OS_ANDROID)
// TODO(jperaza): This symbol isn't defined when building in chromium for
// Android. There may be another symbol to use.
extern "C" {
#if defined(ARCH_CPU_MIPS_FAMILY)
#define START_SYMBOL __start
#else
#define START_SYMBOL _start
#endif
extern void START_SYMBOL();
}  // extern "C"
#endif

namespace crashpad {
namespace test {
namespace {

void TestAgainstCloneOrSelf(pid_t pid) {
  FakePtraceConnection connection;
  ASSERT_TRUE(connection.Initialize(pid));

  AuxiliaryVector aux;
  ASSERT_TRUE(aux.Initialize(&connection));

  MemoryMap mappings;
  ASSERT_TRUE(mappings.Initialize(&connection));

  LinuxVMAddress phdrs;
  ASSERT_TRUE(aux.GetValue(AT_PHDR, &phdrs));
  EXPECT_TRUE(mappings.FindMapping(phdrs));

  int pagesize;
  ASSERT_TRUE(aux.GetValue(AT_PAGESZ, &pagesize));
  EXPECT_EQ(pagesize, getpagesize());

  LinuxVMAddress interp_base;
  ASSERT_TRUE(aux.GetValue(AT_BASE, &interp_base));
  EXPECT_TRUE(mappings.FindMapping(interp_base));

#if !defined(OS_ANDROID)
  LinuxVMAddress entry_addr;
  ASSERT_TRUE(aux.GetValue(AT_ENTRY, &entry_addr));
  EXPECT_EQ(entry_addr, FromPointerCast<LinuxVMAddress>(START_SYMBOL));
#endif

  uid_t uid;
  ASSERT_TRUE(aux.GetValue(AT_UID, &uid));
  EXPECT_EQ(uid, getuid());

  uid_t euid;
  ASSERT_TRUE(aux.GetValue(AT_EUID, &euid));
  EXPECT_EQ(euid, geteuid());

  gid_t gid;
  ASSERT_TRUE(aux.GetValue(AT_GID, &gid));
  EXPECT_EQ(gid, getgid());

  gid_t egid;
  ASSERT_TRUE(aux.GetValue(AT_EGID, &egid));
  EXPECT_EQ(egid, getegid());

  ProcessMemoryLinux memory;
  ASSERT_TRUE(memory.Initialize(pid));

  LinuxVMAddress platform_addr;
  ASSERT_TRUE(aux.GetValue(AT_PLATFORM, &platform_addr));
  std::string platform;
  ASSERT_TRUE(memory.ReadCStringSizeLimited(platform_addr, 10, &platform));
#if defined(ARCH_CPU_X86)
  EXPECT_STREQ(platform.c_str(), "i686");
#elif defined(ARCH_CPU_X86_64)
  EXPECT_STREQ(platform.c_str(), "x86_64");
#elif defined(ARCH_CPU_ARMEL)
  // Machine name and platform are set in Linux:/arch/arm/kernel/setup.c
  // Machine typically looks like "armv7l".
  // Platform typically looks like "v7l".
  utsname sys_names;
  ASSERT_EQ(uname(&sys_names), 0);
  std::string machine_name(sys_names.machine);
  EXPECT_NE(machine_name.find(platform), std::string::npos);
#elif defined(ARCH_CPU_ARM64)
  EXPECT_STREQ(platform.c_str(), "aarch64");
#endif  // ARCH_CPU_X86

#if defined(AT_SYSINFO_EHDR)
  LinuxVMAddress vdso_addr;
  ASSERT_TRUE(aux.GetValue(AT_SYSINFO_EHDR, &vdso_addr));
  EXPECT_TRUE(mappings.FindMapping(vdso_addr));
#endif  // AT_SYSINFO_EHDR

#if defined(AT_EXECFN)
  LinuxVMAddress filename_addr;
  ASSERT_TRUE(aux.GetValue(AT_EXECFN, &filename_addr));
  std::string filename;
  ASSERT_TRUE(memory.ReadCStringSizeLimited(filename_addr, 4096, &filename));
  EXPECT_TRUE(filename.find(GetMainArguments()[0]) != std::string::npos);
#endif  // AT_EXECFN

  int ignore;
  EXPECT_FALSE(aux.GetValue(AT_NULL, &ignore));

  char too_small;
  EXPECT_FALSE(aux.GetValue(AT_PAGESZ, &too_small));

  uint128_struct big_dest;
  memset(&big_dest, 0xf, sizeof(big_dest));
  ASSERT_TRUE(aux.GetValue(AT_PHDR, &big_dest));
  EXPECT_EQ(big_dest.lo, phdrs);
}

TEST(AuxiliaryVector, ReadSelf) {
  TestAgainstCloneOrSelf(getpid());
}

class ReadChildTest : public Multiprocess {
 public:
  ReadChildTest() : Multiprocess() {}
  ~ReadChildTest() {}

 private:
  void MultiprocessParent() override { TestAgainstCloneOrSelf(ChildPID()); }

  void MultiprocessChild() override { CheckedReadFileAtEOF(ReadPipeHandle()); }

  DISALLOW_COPY_AND_ASSIGN(ReadChildTest);
};

TEST(AuxiliaryVector, ReadChild) {
  ReadChildTest test;
  test.Run();
}

class AuxVecTester : public AuxiliaryVector {
 public:
  AuxVecTester() : AuxiliaryVector() {}
  void Insert(uint64_t type, uint64_t value) { values_[type] = value; }
};

TEST(AuxiliaryVector, SignedBit) {
  FakePtraceConnection connection;
  ASSERT_TRUE(connection.Initialize(getpid()));

  AuxVecTester aux;
  ASSERT_TRUE(&connection);
  constexpr uint64_t type = 0x0000000012345678;

  constexpr int32_t neg1_32 = -1;
  aux.Insert(type, bit_cast<uint32_t>(neg1_32));
  int32_t outval32s;
  ASSERT_TRUE(aux.GetValue(type, &outval32s));
  EXPECT_EQ(outval32s, neg1_32);

  constexpr int32_t int32_max = std::numeric_limits<int32_t>::max();
  aux.Insert(type, bit_cast<uint32_t>(int32_max));
  ASSERT_TRUE(aux.GetValue(type, &outval32s));
  EXPECT_EQ(outval32s, int32_max);

  constexpr uint32_t uint32_max = std::numeric_limits<uint32_t>::max();
  aux.Insert(type, uint32_max);
  uint32_t outval32u;
  ASSERT_TRUE(aux.GetValue(type, &outval32u));
  EXPECT_EQ(outval32u, uint32_max);

  constexpr int64_t neg1_64 = -1;
  aux.Insert(type, bit_cast<uint64_t>(neg1_64));
  int64_t outval64s;
  ASSERT_TRUE(aux.GetValue(type, &outval64s));
  EXPECT_EQ(outval64s, neg1_64);

  constexpr int64_t int64_max = std::numeric_limits<int64_t>::max();
  aux.Insert(type, bit_cast<uint64_t>(int64_max));
  ASSERT_TRUE(aux.GetValue(type, &outval64s));
  EXPECT_EQ(outval64s, int64_max);

  constexpr uint64_t uint64_max = std::numeric_limits<uint64_t>::max();
  aux.Insert(type, uint64_max);
  uint64_t outval64u;
  ASSERT_TRUE(aux.GetValue(type, &outval64u));
  EXPECT_EQ(outval64u, uint64_max);
}

}  // namespace
}  // namespace test
}  // namespace crashpad
