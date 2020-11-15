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

#include "snapshot/win/pe_image_reader.h"

#define PSAPI_VERSION 1
#include <psapi.h>

#include "base/files/file_path.h"
#include "base/strings/stringprintf.h"
#include "base/strings/utf_string_conversions.h"
#include "gtest/gtest.h"
#include "snapshot/win/process_reader_win.h"
#include "test/errors.h"
#include "test/scoped_module_handle.h"
#include "test/test_paths.h"
#include "util/misc/from_pointer_cast.h"
#include "util/win/get_module_information.h"
#include "util/win/module_version.h"
#include "util/win/process_info.h"

namespace crashpad {
namespace test {
namespace {

TEST(PEImageReader, DebugDirectory) {
  base::FilePath module_path =
      TestPaths::BuildArtifact(L"snapshot",
                               L"image_reader_module",
                               TestPaths::FileType::kLoadableModule);
  ScopedModuleHandle module_handle(LoadLibrary(module_path.value().c_str()));
  ASSERT_TRUE(module_handle.valid()) << ErrorMessage("LoadLibrary");

  PEImageReader pe_image_reader;
  ProcessReaderWin process_reader;
  ASSERT_TRUE(process_reader.Initialize(GetCurrentProcess(),
                                        ProcessSuspensionState::kRunning));

  MODULEINFO module_info;
  ASSERT_TRUE(CrashpadGetModuleInformation(GetCurrentProcess(),
                                           module_handle.get(),
                                           &module_info,
                                           sizeof(module_info)))
      << ErrorMessage("GetModuleInformation");
  EXPECT_EQ(module_info.lpBaseOfDll, module_handle.get());

  base::FilePath module_basename = module_path.BaseName();
  ASSERT_TRUE(pe_image_reader.Initialize(
      &process_reader,
      FromPointerCast<WinVMAddress>(module_handle.get()),
      module_info.SizeOfImage,
      base::UTF16ToUTF8(module_basename.value())));

  UUID uuid;
  DWORD age;
  std::string pdbname;
  ASSERT_TRUE(pe_image_reader.DebugDirectoryInformation(&uuid, &age, &pdbname));
  std::string module_name =
      base::UTF16ToUTF8(module_basename.RemoveFinalExtension().value());
  EXPECT_NE(pdbname.find(module_name), std::string::npos);
  const std::string suffix(".pdb");
  EXPECT_EQ(
      pdbname.compare(pdbname.size() - suffix.size(), suffix.size(), suffix),
      0);
}

void TestVSFixedFileInfo(ProcessReaderWin* process_reader,
                         const ProcessInfo::Module& module,
                         bool known_dll) {
  PEImageReader pe_image_reader;
  ASSERT_TRUE(pe_image_reader.Initialize(process_reader,
                                         module.dll_base,
                                         module.size,
                                         base::UTF16ToUTF8(module.name)));

  VS_FIXEDFILEINFO observed;
  const bool observed_rv = pe_image_reader.VSFixedFileInfo(&observed);
  ASSERT_TRUE(observed_rv || !known_dll);

  if (observed_rv) {
    EXPECT_EQ(observed.dwSignature, static_cast<DWORD>(VS_FFI_SIGNATURE));
    EXPECT_EQ(observed.dwStrucVersion, static_cast<DWORD>(VS_FFI_STRUCVERSION));
    EXPECT_EQ(observed.dwFileFlags & ~observed.dwFileFlagsMask, 0u);
    if (known_dll) {
      EXPECT_EQ(observed.dwFileOS, static_cast<DWORD>(VOS_NT_WINDOWS32));
      EXPECT_EQ(observed.dwFileType, static_cast<DWORD>(VFT_DLL));
    } else {
      EXPECT_NE(observed.dwFileOS & VOS_NT_WINDOWS32, 0u);

      // VFT_DRV/VFT2_DRV_NETWORK is for nsi.dll, “network service interface.”
      // It’s not normally loaded, but has been observed to be loaded in some
      // cases.
      EXPECT_TRUE(observed.dwFileType == VFT_APP ||
                  observed.dwFileType == VFT_DLL ||
                  (observed.dwFileType == VFT_DRV &&
                   observed.dwFileSubtype == VFT2_DRV_NETWORK))
          << base::StringPrintf("type 0x%lx, subtype 0x%lx",
                                observed.dwFileType,
                                observed.dwFileSubtype);
    }
  }

  base::FilePath module_path(module.name);

  const DWORD version = GetVersion();
  const int major_version = LOBYTE(LOWORD(version));
  const int minor_version = HIBYTE(LOWORD(version));
  if (major_version > 6 || (major_version == 6 && minor_version >= 2)) {
    // Windows 8 or later.
    //
    // Use BaseName() to ensure that GetModuleVersionAndType() finds the
    // already-loaded module with the specified name. Otherwise, dwFileVersionMS
    // may not match. This appears to be related to the changes made in Windows
    // 8.1 to GetVersion() and GetVersionEx() for non-manifested applications
    module_path = module_path.BaseName();
  }

  VS_FIXEDFILEINFO expected;
  const bool expected_rv = GetModuleVersionAndType(module_path, &expected);
  ASSERT_TRUE(expected_rv || !known_dll);

  EXPECT_EQ(observed_rv, expected_rv);

  if (observed_rv && expected_rv) {
    EXPECT_EQ(observed.dwSignature, expected.dwSignature);
    EXPECT_EQ(observed.dwStrucVersion, expected.dwStrucVersion);
    EXPECT_EQ(observed.dwFileVersionMS, expected.dwFileVersionMS);
    EXPECT_EQ(observed.dwFileVersionLS, expected.dwFileVersionLS);
    EXPECT_EQ(observed.dwProductVersionMS, expected.dwProductVersionMS);
    EXPECT_EQ(observed.dwProductVersionLS, expected.dwProductVersionLS);
    EXPECT_EQ(observed.dwFileFlagsMask, expected.dwFileFlagsMask);
    EXPECT_EQ(observed.dwFileFlags, expected.dwFileFlags);
    EXPECT_EQ(observed.dwFileOS, expected.dwFileOS);
    EXPECT_EQ(observed.dwFileType, expected.dwFileType);
    EXPECT_EQ(observed.dwFileSubtype, expected.dwFileSubtype);
    EXPECT_EQ(observed.dwFileDateMS, expected.dwFileDateMS);
    EXPECT_EQ(observed.dwFileDateLS, expected.dwFileDateLS);
  }
}

TEST(PEImageReader, VSFixedFileInfo_OneModule) {
  ProcessReaderWin process_reader;
  ASSERT_TRUE(process_reader.Initialize(GetCurrentProcess(),
                                        ProcessSuspensionState::kRunning));

  static constexpr wchar_t kModuleName[] = L"kernel32.dll";
  const HMODULE module_handle = GetModuleHandle(kModuleName);
  ASSERT_TRUE(module_handle) << ErrorMessage("GetModuleHandle");

  MODULEINFO module_info;
  ASSERT_TRUE(CrashpadGetModuleInformation(
      GetCurrentProcess(), module_handle, &module_info, sizeof(module_info)))
      << ErrorMessage("GetModuleInformation");
  EXPECT_EQ(module_info.lpBaseOfDll, module_handle);

  ProcessInfo::Module module;
  module.name = kModuleName;
  module.dll_base = FromPointerCast<WinVMAddress>(module_info.lpBaseOfDll);
  module.size = module_info.SizeOfImage;

  TestVSFixedFileInfo(&process_reader, module, true);
}

TEST(PEImageReader, VSFixedFileInfo_AllModules) {
  ProcessReaderWin process_reader;
  ASSERT_TRUE(process_reader.Initialize(GetCurrentProcess(),
                                        ProcessSuspensionState::kRunning));

  const std::vector<ProcessInfo::Module>& modules = process_reader.Modules();
  EXPECT_GT(modules.size(), 2u);

  for (const auto& module : modules) {
    SCOPED_TRACE(base::UTF16ToUTF8(module.name));
    TestVSFixedFileInfo(&process_reader, module, false);
  }
}

}  // namespace
}  // namespace test
}  // namespace crashpad
