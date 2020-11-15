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

#include "snapshot/linux/debug_rendezvous.h"

#include <linux/auxvec.h>
#include <string.h>
#include <unistd.h>

#include <limits>

#include "base/format_macros.h"
#include "base/strings/string_number_conversions.h"
#include "base/strings/string_piece.h"
#include "base/strings/stringprintf.h"
#include "build/build_config.h"
#include "gtest/gtest.h"
#include "snapshot/elf/elf_image_reader.h"
#include "test/linux/fake_ptrace_connection.h"
#include "test/multiprocess.h"
#include "util/linux/address_types.h"
#include "util/linux/auxiliary_vector.h"
#include "util/linux/direct_ptrace_connection.h"
#include "util/linux/memory_map.h"
#include "util/process/process_memory_linux.h"
#include "util/process/process_memory_range.h"

#if defined(OS_ANDROID)
#include <sys/system_properties.h>
#endif

namespace crashpad {
namespace test {
namespace {

#if defined(OS_ANDROID)
int AndroidRuntimeAPI() {
  char api_string[PROP_VALUE_MAX];
  int length = __system_property_get("ro.build.version.sdk", api_string);
  if (length <= 0) {
    return -1;
  }

  int api_level;
  bool success =
      base::StringToInt(base::StringPiece(api_string, length), &api_level);
  return success ? api_level : -1;
}
#endif  // OS_ANDROID

void TestAgainstTarget(PtraceConnection* connection) {
  // Use ElfImageReader on the main executable which can tell us the debug
  // address. glibc declares the symbol _r_debug in link.h which we can use to
  // get the address, but Android does not.
  AuxiliaryVector aux;
  ASSERT_TRUE(aux.Initialize(connection));

  LinuxVMAddress phdrs;
  ASSERT_TRUE(aux.GetValue(AT_PHDR, &phdrs));

  MemoryMap mappings;
  ASSERT_TRUE(mappings.Initialize(connection));

  const MemoryMap::Mapping* phdr_mapping = mappings.FindMapping(phdrs);
  ASSERT_TRUE(phdr_mapping);
  std::vector<const MemoryMap::Mapping*> exe_mappings =
      mappings.FindFilePossibleMmapStarts(*phdr_mapping);
  ASSERT_EQ(exe_mappings.size(), 1u);
  LinuxVMAddress elf_address = exe_mappings[0]->range.Base();

  ProcessMemoryLinux memory;
  ASSERT_TRUE(memory.Initialize(connection->GetProcessID()));
  ProcessMemoryRange range;
  ASSERT_TRUE(range.Initialize(&memory, connection->Is64Bit()));

  ElfImageReader exe_reader;
  ASSERT_TRUE(exe_reader.Initialize(range, elf_address));
  LinuxVMAddress debug_address;
  ASSERT_TRUE(exe_reader.GetDebugAddress(&debug_address));

  // start the actual tests
  DebugRendezvous debug;
  ASSERT_TRUE(debug.Initialize(range, debug_address));

#if defined(OS_ANDROID)
  const int android_runtime_api = AndroidRuntimeAPI();
  ASSERT_GE(android_runtime_api, 1);

  EXPECT_NE(debug.Executable()->name.find("crashpad_snapshot_test"),
            std::string::npos);

  // Android's loader never sets the dynamic array for the executable.
  EXPECT_EQ(debug.Executable()->dynamic_array, 0u);
#else
  // glibc's loader implements most of the tested features that Android's was
  // missing but has since gained.
  const int android_runtime_api = std::numeric_limits<int>::max();

  // glibc's loader does not set the name for the executable.
  EXPECT_TRUE(debug.Executable()->name.empty());
  CheckedLinuxAddressRange exe_range(
      connection->Is64Bit(), exe_reader.Address(), exe_reader.Size());
  EXPECT_TRUE(exe_range.ContainsValue(debug.Executable()->dynamic_array));
#endif  // OS_ANDROID

  // Android's loader doesn't set the load bias until Android 4.3 (API 18).
  if (android_runtime_api >= 18) {
    EXPECT_EQ(debug.Executable()->load_bias, exe_reader.GetLoadBias());
  } else {
    EXPECT_EQ(debug.Executable()->load_bias, 0);
  }

  for (const DebugRendezvous::LinkEntry& module : debug.Modules()) {
    SCOPED_TRACE(base::StringPrintf("name %s, load_bias 0x%" PRIx64
                                    ", dynamic_array 0x%" PRIx64,
                                    module.name.c_str(),
                                    module.load_bias,
                                    module.dynamic_array));
    const bool is_android_loader = (module.name == "/system/bin/linker" ||
                                    module.name == "/system/bin/linker64");

    // Android's loader doesn't set its own dynamic array until Android 4.2
    // (API 17).
    if (is_android_loader && android_runtime_api < 17) {
      EXPECT_EQ(module.dynamic_array, 0u);
      EXPECT_EQ(module.load_bias, 0);
      continue;
    }

    ASSERT_TRUE(module.dynamic_array);
    const MemoryMap::Mapping* dyn_mapping =
        mappings.FindMapping(module.dynamic_array);
    ASSERT_TRUE(dyn_mapping);

    std::vector<const MemoryMap::Mapping*> possible_mappings =
        mappings.FindFilePossibleMmapStarts(*dyn_mapping);
    ASSERT_GE(possible_mappings.size(), 1u);

    std::unique_ptr<ElfImageReader> module_reader;
    const MemoryMap::Mapping* module_mapping = nullptr;
    for (const auto mapping : possible_mappings) {
      auto parsed_module = std::make_unique<ElfImageReader>();
      VMAddress dynamic_address;
      if (parsed_module->Initialize(range, mapping->range.Base()) &&
          parsed_module->GetDynamicArrayAddress(&dynamic_address) &&
          dynamic_address == module.dynamic_array) {
        module_reader = std::move(parsed_module);
        module_mapping = mapping;
        break;
      }
    }
    ASSERT_TRUE(module_reader.get());

#if defined(OS_ANDROID)
    EXPECT_FALSE(module.name.empty());
#else
    // glibc's loader doesn't always set the name in the link map for the vdso.
    EXPECT_PRED4(
        [](const std::string mapping_name,
           int device,
           int inode,
           const std::string& module_name) {
          const bool is_vdso_mapping =
              device == 0 && inode == 0 && mapping_name == "[vdso]";
          static constexpr char kPrefix[] = "linux-vdso.so.";
          return is_vdso_mapping ==
                 (module_name.empty() ||
                  module_name.compare(0, strlen(kPrefix), kPrefix) == 0);
        },
        module_mapping->name,
        module_mapping->device,
        module_mapping->inode,
        module.name);
#endif  // OS_ANDROID

    // Android's loader stops setting its own load bias after Android 4.4.4
    // (API 20) until Android 6.0 (API 23).
    if (is_android_loader && android_runtime_api > 20 &&
        android_runtime_api < 23) {
      EXPECT_EQ(module.load_bias, 0);
    } else {
      EXPECT_EQ(module.load_bias, module_reader->GetLoadBias());
    }

    CheckedLinuxAddressRange module_range(
        connection->Is64Bit(), module_reader->Address(), module_reader->Size());
    EXPECT_TRUE(module_range.ContainsValue(module.dynamic_array));
  }
}

TEST(DebugRendezvous, Self) {
  FakePtraceConnection connection;
  ASSERT_TRUE(connection.Initialize(getpid()));

  TestAgainstTarget(&connection);
}

class ChildTest : public Multiprocess {
 public:
  ChildTest() {}
  ~ChildTest() {}

 private:
  void MultiprocessParent() {
    DirectPtraceConnection connection;
    ASSERT_TRUE(connection.Initialize(ChildPID()));

    TestAgainstTarget(&connection);
  }

  void MultiprocessChild() { CheckedReadFileAtEOF(ReadPipeHandle()); }

  DISALLOW_COPY_AND_ASSIGN(ChildTest);
};

TEST(DebugRendezvous, Child) {
  ChildTest test;
  test.Run();
}

}  // namespace
}  // namespace test
}  // namespace crashpad
