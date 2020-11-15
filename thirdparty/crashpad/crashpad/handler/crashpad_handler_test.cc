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

#include <windows.h>
#include <dbghelp.h>
#include <string.h>

#include <map>
#include <string>
#include <vector>

#include "base/files/file_path.h"
#include "client/crash_report_database.h"
#include "client/crashpad_client.h"
#include "gtest/gtest.h"
#include "minidump/test/minidump_file_writer_test_util.h"
#include "test/test_paths.h"
#include "test/win/win_multiprocess_with_temp_dir.h"
#include "util/file/file_reader.h"
#include "util/misc/capture_context.h"

namespace crashpad {
namespace test {
namespace {

constexpr DWORD kExpectedExitCode = 0x1CEB00DA;

void StartAndCrashWithExtendedHandler(const base::FilePath& temp_dir) {
  base::FilePath handler_path = TestPaths::BuildArtifact(
      L"handler", L"extended_handler", TestPaths::FileType::kExecutable);

  CrashpadClient client;
  ASSERT_TRUE(client.StartHandler(handler_path,
                                  temp_dir,
                                  base::FilePath(),
                                  "",
                                  std::map<std::string, std::string>(),
                                  std::vector<std::string>(),
                                  false,
                                  false));

  // It appears that the GoogleTest fixture will catch and handle exceptions
  // from here. Hence the fabricated crash in favor of raising an exception.
  EXCEPTION_RECORD exception_record = {kExpectedExitCode,
                                       EXCEPTION_NONCONTINUABLE};
  CONTEXT context;
  CaptureContext(&context);
  EXCEPTION_POINTERS exception_pointers = {&exception_record, &context};
  CrashpadClient::DumpAndCrash(&exception_pointers);
}

class CrashWithExtendedHandler final : public WinMultiprocessWithTempDir {
 public:
  CrashWithExtendedHandler() : WinMultiprocessWithTempDir() {}
  ~CrashWithExtendedHandler() {}

 private:
  void ValidateGeneratedDump();

  void WinMultiprocessParent() override {
    SetExpectedChildExitCode(kExpectedExitCode);
  }

  void WinMultiprocessChild() override {
    StartAndCrashWithExtendedHandler(GetTempDirPath());
  }

  void WinMultiprocessParentAfterChild(HANDLE child) override {
    // At this point the child has exited, which means the crash report should
    // have been written.
    ValidateGeneratedDump();

    // Delegate the cleanup to the superclass.
    WinMultiprocessWithTempDir::WinMultiprocessParentAfterChild(child);
  }
};

void CrashWithExtendedHandler::ValidateGeneratedDump() {
  // Open the database and find the sole dump that should have been created.
  std::unique_ptr<CrashReportDatabase> database(
      CrashReportDatabase::Initialize(GetTempDirPath()));
  ASSERT_TRUE(database);

  std::vector<CrashReportDatabase::Report> reports;
  ASSERT_EQ(database->GetPendingReports(&reports),
            CrashReportDatabase::kNoError);
  ASSERT_EQ(reports.size(), 1u);

  // Open the dump and validate that it has the extension stream with the
  // expected contents.
  FileReader reader;
  ASSERT_TRUE(reader.Open(reports[0].file_path));

  // Read the header.
  MINIDUMP_HEADER header = {};
  ASSERT_TRUE(reader.ReadExactly(&header, sizeof(header)));

  // Read the directory.
  std::vector<MINIDUMP_DIRECTORY> directory(header.NumberOfStreams);
  ASSERT_TRUE(reader.SeekSet(header.StreamDirectoryRva));
  ASSERT_TRUE(reader.ReadExactly(directory.data(),
                                 directory.size() * sizeof(directory[0])));

  // Search for the extension stream.
  size_t found_extension_streams = 0;
  for (const auto& entry : directory) {
    if (entry.StreamType == 0xCAFEBABE) {
      ++found_extension_streams;

      ASSERT_TRUE(reader.SeekSet(entry.Location.Rva));

      std::vector<char> data;
      data.resize(entry.Location.DataSize);

      ASSERT_TRUE(reader.ReadExactly(data.data(), data.size()));

      static constexpr char kExpectedData[] = "Injected extension stream!";
      EXPECT_EQ(memcmp(kExpectedData, data.data(), sizeof(kExpectedData)), 0);
    }
  }

  EXPECT_EQ(found_extension_streams, 1u);
}

#if defined(ADDRESS_SANITIZER)
// https://crbug.com/845011
#define MAYBE_ExtensibilityCalloutsWork DISABLED_ExtensibilityCalloutsWork
#else
#define MAYBE_ExtensibilityCalloutsWork ExtensibilityCalloutsWork
#endif
TEST(CrashpadHandler, MAYBE_ExtensibilityCalloutsWork) {
  WinMultiprocessWithTempDir::Run<CrashWithExtendedHandler>();
}

}  // namespace
}  // namespace test
}  // namespace crashpad
