// Copyright 2015 The Crashpad Authors. All rights reserved.
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

#include "snapshot/crashpad_info_client_options.h"

#include "base/auto_reset.h"
#include "base/files/file_path.h"
#include "base/macros.h"
#include "base/strings/utf_string_conversions.h"
#include "build/build_config.h"
#include "client/crashpad_info.h"
#include "gtest/gtest.h"
#include "test/errors.h"
#include "test/scoped_module_handle.h"
#include "test/test_paths.h"

#if defined(OS_MACOSX)
#include <dlfcn.h>
#include "snapshot/mac/process_snapshot_mac.h"
#elif defined(OS_WIN)
#include <windows.h>
#include "snapshot/win/process_snapshot_win.h"
#elif defined(OS_FUCHSIA)
#include <lib/zx/process.h>
#include "snapshot/fuchsia/process_snapshot_fuchsia.h"
#endif

namespace crashpad {
namespace test {
namespace {

TEST(CrashpadInfoClientOptions, TriStateFromCrashpadInfo) {
  EXPECT_EQ(CrashpadInfoClientOptions::TriStateFromCrashpadInfo(0),
            TriState::kUnset);
  EXPECT_EQ(CrashpadInfoClientOptions::TriStateFromCrashpadInfo(1),
            TriState::kEnabled);
  EXPECT_EQ(CrashpadInfoClientOptions::TriStateFromCrashpadInfo(2),
            TriState::kDisabled);

  // These will produce log messages but should result in kUnset being returned.
  EXPECT_EQ(CrashpadInfoClientOptions::TriStateFromCrashpadInfo(3),
            TriState::kUnset);
  EXPECT_EQ(CrashpadInfoClientOptions::TriStateFromCrashpadInfo(4),
            TriState::kUnset);
  EXPECT_EQ(CrashpadInfoClientOptions::TriStateFromCrashpadInfo(0xff),
            TriState::kUnset);
}

class ScopedUnsetCrashpadInfoOptions {
 public:
  explicit ScopedUnsetCrashpadInfoOptions(CrashpadInfo* crashpad_info)
      : crashpad_info_(crashpad_info) {
  }

  ~ScopedUnsetCrashpadInfoOptions() {
    crashpad_info_->set_crashpad_handler_behavior(TriState::kUnset);
    crashpad_info_->set_system_crash_reporter_forwarding(TriState::kUnset);
    crashpad_info_->set_gather_indirectly_referenced_memory(TriState::kUnset,
                                                            0);
  }

 private:
  CrashpadInfo* crashpad_info_;

  DISALLOW_COPY_AND_ASSIGN(ScopedUnsetCrashpadInfoOptions);
};

CrashpadInfoClientOptions SelfProcessSnapshotAndGetCrashpadOptions() {
#if defined(OS_MACOSX)
  ProcessSnapshotMac process_snapshot;
  EXPECT_TRUE(process_snapshot.Initialize(mach_task_self()));
#elif defined(OS_WIN)
  ProcessSnapshotWin process_snapshot;
  EXPECT_TRUE(process_snapshot.Initialize(
      GetCurrentProcess(), ProcessSuspensionState::kRunning, 0, 0));
#elif defined(OS_FUCHSIA)
  ProcessSnapshotFuchsia process_snapshot;
  EXPECT_TRUE(process_snapshot.Initialize(*zx::process::self()));
#else
#error Port.
#endif  // OS_MACOSX

  CrashpadInfoClientOptions options;
  process_snapshot.GetCrashpadOptions(&options);
  return options;
}

TEST(CrashpadInfoClientOptions, OneModule) {
  // Make sure that the initial state has all values unset.
  auto options = SelfProcessSnapshotAndGetCrashpadOptions();

  EXPECT_EQ(options.crashpad_handler_behavior, TriState::kUnset);
  EXPECT_EQ(options.system_crash_reporter_forwarding, TriState::kUnset);
  EXPECT_EQ(options.gather_indirectly_referenced_memory, TriState::kUnset);
  EXPECT_EQ(options.indirectly_referenced_memory_cap, 0u);

  CrashpadInfo* crashpad_info = CrashpadInfo::GetCrashpadInfo();
  ASSERT_TRUE(crashpad_info);

  {
    ScopedUnsetCrashpadInfoOptions unset(crashpad_info);

    crashpad_info->set_crashpad_handler_behavior(TriState::kEnabled);

    options = SelfProcessSnapshotAndGetCrashpadOptions();
    EXPECT_EQ(options.crashpad_handler_behavior, TriState::kEnabled);
    EXPECT_EQ(options.system_crash_reporter_forwarding, TriState::kUnset);
    EXPECT_EQ(options.gather_indirectly_referenced_memory, TriState::kUnset);
    EXPECT_EQ(options.indirectly_referenced_memory_cap, 0u);
  }

  {
    ScopedUnsetCrashpadInfoOptions unset(crashpad_info);

    crashpad_info->set_system_crash_reporter_forwarding(TriState::kDisabled);

    options = SelfProcessSnapshotAndGetCrashpadOptions();
    EXPECT_EQ(options.crashpad_handler_behavior, TriState::kUnset);
    EXPECT_EQ(options.system_crash_reporter_forwarding, TriState::kDisabled);
    EXPECT_EQ(options.gather_indirectly_referenced_memory, TriState::kUnset);
    EXPECT_EQ(options.indirectly_referenced_memory_cap, 0u);
  }

  {
    ScopedUnsetCrashpadInfoOptions unset(crashpad_info);

    crashpad_info->set_gather_indirectly_referenced_memory(TriState::kEnabled,
                                                           1234);

    options = SelfProcessSnapshotAndGetCrashpadOptions();
    EXPECT_EQ(options.crashpad_handler_behavior, TriState::kUnset);
    EXPECT_EQ(options.system_crash_reporter_forwarding, TriState::kUnset);
    EXPECT_EQ(options.gather_indirectly_referenced_memory, TriState::kEnabled);
    EXPECT_EQ(options.indirectly_referenced_memory_cap, 1234u);
  }
}

TEST(CrashpadInfoClientOptions, TwoModules) {
  // Open the module, which has its own CrashpadInfo structure.
  base::FilePath module_path =
      TestPaths::BuildArtifact(FILE_PATH_LITERAL("snapshot"),
                               FILE_PATH_LITERAL("module"),
                               TestPaths::FileType::kLoadableModule);
#if defined(OS_POSIX)
  ScopedModuleHandle module(
      dlopen(module_path.value().c_str(), RTLD_LAZY | RTLD_LOCAL));
  ASSERT_TRUE(module.valid()) << "dlopen " << module_path.value() << ": "
                              << dlerror();
#elif defined(OS_WIN)
  ScopedModuleHandle module(LoadLibrary(module_path.value().c_str()));
  ASSERT_TRUE(module.valid()) << "LoadLibrary "
                              << base::UTF16ToUTF8(module_path.value()) << ": "
                              << ErrorMessage();
#else
#error Port.
#endif  // OS_MACOSX

  // Get the function pointer from the module. This wraps GetCrashpadInfo(), but
  // because it runs in the module, it returns the remote module’s CrashpadInfo
  // structure.
  CrashpadInfo* (*TestModule_GetCrashpadInfo)() =
      module.LookUpSymbol<CrashpadInfo* (*)()>("TestModule_GetCrashpadInfo");
  ASSERT_TRUE(TestModule_GetCrashpadInfo);

  auto options = SelfProcessSnapshotAndGetCrashpadOptions();

  // Make sure that the initial state has all values unset.
  EXPECT_EQ(options.crashpad_handler_behavior, TriState::kUnset);
  EXPECT_EQ(options.system_crash_reporter_forwarding, TriState::kUnset);
  EXPECT_EQ(options.gather_indirectly_referenced_memory, TriState::kUnset);

  // Get both CrashpadInfo structures.
  CrashpadInfo* local_crashpad_info = CrashpadInfo::GetCrashpadInfo();
  ASSERT_TRUE(local_crashpad_info);

  CrashpadInfo* remote_crashpad_info = TestModule_GetCrashpadInfo();
  ASSERT_TRUE(remote_crashpad_info);

  {
    ScopedUnsetCrashpadInfoOptions unset_local(local_crashpad_info);
    ScopedUnsetCrashpadInfoOptions unset_remote(remote_crashpad_info);

    // When only one module sets a value, it applies to the entire process.
    remote_crashpad_info->set_crashpad_handler_behavior(TriState::kEnabled);

    options = SelfProcessSnapshotAndGetCrashpadOptions();
    EXPECT_EQ(options.crashpad_handler_behavior, TriState::kEnabled);
    EXPECT_EQ(options.system_crash_reporter_forwarding, TriState::kUnset);
    EXPECT_EQ(options.gather_indirectly_referenced_memory, TriState::kUnset);

    // When more than one module sets a value, the first one in the module list
    // applies to the process. The local module should appear before the remote
    // module, because the local module loaded the remote module.
    local_crashpad_info->set_crashpad_handler_behavior(TriState::kDisabled);

    options = SelfProcessSnapshotAndGetCrashpadOptions();
    EXPECT_EQ(options.crashpad_handler_behavior, TriState::kDisabled);
    EXPECT_EQ(options.system_crash_reporter_forwarding, TriState::kUnset);
    EXPECT_EQ(options.gather_indirectly_referenced_memory, TriState::kUnset);
  }

  {
    ScopedUnsetCrashpadInfoOptions unset_local(local_crashpad_info);
    ScopedUnsetCrashpadInfoOptions unset_remote(remote_crashpad_info);

    // When only one module sets a value, it applies to the entire process.
    remote_crashpad_info->set_system_crash_reporter_forwarding(
        TriState::kDisabled);

    options = SelfProcessSnapshotAndGetCrashpadOptions();
    EXPECT_EQ(options.crashpad_handler_behavior, TriState::kUnset);
    EXPECT_EQ(options.system_crash_reporter_forwarding, TriState::kDisabled);
    EXPECT_EQ(options.gather_indirectly_referenced_memory, TriState::kUnset);

    // When more than one module sets a value, the first one in the module list
    // applies to the process. The local module should appear before the remote
    // module, because the local module loaded the remote module.
    local_crashpad_info->set_system_crash_reporter_forwarding(
        TriState::kEnabled);

    options = SelfProcessSnapshotAndGetCrashpadOptions();
    EXPECT_EQ(options.crashpad_handler_behavior, TriState::kUnset);
    EXPECT_EQ(options.system_crash_reporter_forwarding, TriState::kEnabled);
    EXPECT_EQ(options.gather_indirectly_referenced_memory, TriState::kUnset);
  }
}

class CrashpadInfoSizes_ClientOptions
    : public testing::TestWithParam<base::FilePath::StringType> {};

TEST_P(CrashpadInfoSizes_ClientOptions, DifferentlySizedStruct) {
  base::FilePath::StringType artifact(FILE_PATH_LITERAL("module_"));
  artifact += GetParam();

  // Open the module, which has a CrashpadInfo-like structure that’s smaller or
  // larger than the current version’s CrashpadInfo structure defined in the
  // client library.
  base::FilePath module_path =
      TestPaths::BuildArtifact(FILE_PATH_LITERAL("snapshot"),
                               artifact,
                               TestPaths::FileType::kLoadableModule);
#if defined(OS_POSIX)
  ScopedModuleHandle module(
      dlopen(module_path.value().c_str(), RTLD_LAZY | RTLD_LOCAL));
  ASSERT_TRUE(module.valid())
      << "dlopen " << module_path.value() << ": " << dlerror();
#elif defined(OS_WIN)
  ScopedModuleHandle module(LoadLibrary(module_path.value().c_str()));
  ASSERT_TRUE(module.valid())
      << "LoadLibrary " << base::UTF16ToUTF8(module_path.value()) << ": "
      << ErrorMessage();
#else
#error Port.
#endif  // OS_MACOSX

  // Get the function pointer from the module.
  CrashpadInfo* (*TestModule_GetCrashpadInfo)() =
      module.LookUpSymbol<CrashpadInfo* (*)()>("TestModule_GetCrashpadInfo");
  ASSERT_TRUE(TestModule_GetCrashpadInfo);

  auto options = SelfProcessSnapshotAndGetCrashpadOptions();

  // Make sure that the initial state has all values unset.
  EXPECT_EQ(options.crashpad_handler_behavior, TriState::kUnset);
  EXPECT_EQ(options.system_crash_reporter_forwarding, TriState::kUnset);
  EXPECT_EQ(options.gather_indirectly_referenced_memory, TriState::kUnset);

  // Get the remote CrashpadInfo structure.
  CrashpadInfo* remote_crashpad_info = TestModule_GetCrashpadInfo();
  ASSERT_TRUE(remote_crashpad_info);

  {
    ScopedUnsetCrashpadInfoOptions unset_remote(remote_crashpad_info);

    // Make sure that a change in the remote structure can be read back out,
    // even though it’s a different size.
    remote_crashpad_info->set_crashpad_handler_behavior(TriState::kEnabled);
    remote_crashpad_info->set_system_crash_reporter_forwarding(
        TriState::kDisabled);

    options = SelfProcessSnapshotAndGetCrashpadOptions();
    EXPECT_EQ(options.crashpad_handler_behavior, TriState::kEnabled);
    EXPECT_EQ(options.system_crash_reporter_forwarding, TriState::kDisabled);
    EXPECT_EQ(options.gather_indirectly_referenced_memory, TriState::kUnset);
  }

  {
    ScopedUnsetCrashpadInfoOptions unset_remote(remote_crashpad_info);

    // Make sure that the portion of the remote structure lying beyond its
    // declared size reads as zero.

    // 4 = offsetof(CrashpadInfo, size_), but it’s private.
    uint32_t* size = reinterpret_cast<uint32_t*>(
        reinterpret_cast<char*>(remote_crashpad_info) + 4);

    // 21 = offsetof(CrashpadInfo, system_crash_reporter_forwarding_, but it’s
    // private.
    base::AutoReset<uint32_t> reset_size(size, 21);

    // system_crash_reporter_forwarding_ is now beyond the struct’s declared
    // size. Storage has actually been allocated for it, so it’s safe to set
    // here.
    remote_crashpad_info->set_crashpad_handler_behavior(TriState::kEnabled);
    remote_crashpad_info->set_system_crash_reporter_forwarding(
        TriState::kDisabled);

    // Since system_crash_reporter_forwarding_ is beyond the struct’s declared
    // size, it should read as 0 (TriState::kUnset), even though it was set to
    // a different value above.
    options = SelfProcessSnapshotAndGetCrashpadOptions();
    EXPECT_EQ(options.crashpad_handler_behavior, TriState::kEnabled);
    EXPECT_EQ(options.system_crash_reporter_forwarding, TriState::kUnset);
    EXPECT_EQ(options.gather_indirectly_referenced_memory, TriState::kUnset);
  }
}

INSTANTIATE_TEST_CASE_P(CrashpadInfoSizes_ClientOptions,
                        CrashpadInfoSizes_ClientOptions,
                        testing::Values(FILE_PATH_LITERAL("small"),
                                        FILE_PATH_LITERAL("large")));

}  // namespace
}  // namespace test
}  // namespace crashpad
