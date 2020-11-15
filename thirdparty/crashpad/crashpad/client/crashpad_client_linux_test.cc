// Copyright 2018 The Crashpad Authors. All rights reserved.
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

#include "client/crashpad_client.h"

#include <stdlib.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#include "base/logging.h"
#include "client/annotation.h"
#include "client/annotation_list.h"
#include "client/crash_report_database.h"
#include "client/simulate_crash.h"
#include "gtest/gtest.h"
#include "snapshot/annotation_snapshot.h"
#include "snapshot/minidump/process_snapshot_minidump.h"
#include "snapshot/sanitized/sanitization_information.h"
#include "test/multiprocess.h"
#include "test/multiprocess_exec.h"
#include "test/scoped_temp_dir.h"
#include "test/test_paths.h"
#include "util/file/file_io.h"
#include "util/file/filesystem.h"
#include "util/linux/exception_handler_client.h"
#include "util/linux/exception_information.h"
#include "util/misc/address_types.h"
#include "util/misc/from_pointer_cast.h"
#include "util/posix/signals.h"

namespace crashpad {
namespace test {
namespace {

bool HandleCrashSuccessfully(int, siginfo_t*, ucontext_t*) {
  return true;
}

TEST(CrashpadClient, SimulateCrash) {
  ScopedTempDir temp_dir;

  base::FilePath handler_path = TestPaths::Executable().DirName().Append(
      FILE_PATH_LITERAL("crashpad_handler"));

  crashpad::CrashpadClient client;
  ASSERT_TRUE(client.StartHandlerAtCrash(handler_path,
                                         base::FilePath(temp_dir.path()),
                                         base::FilePath(),
                                         "",
                                         std::map<std::string, std::string>(),
                                         std::vector<std::string>()));

  auto database =
      CrashReportDatabase::InitializeWithoutCreating(temp_dir.path());
  ASSERT_TRUE(database);

  {
    CrashpadClient::SetFirstChanceExceptionHandler(HandleCrashSuccessfully);

    CRASHPAD_SIMULATE_CRASH();

    std::vector<CrashReportDatabase::Report> reports;
    ASSERT_EQ(database->GetPendingReports(&reports),
              CrashReportDatabase::kNoError);
    EXPECT_EQ(reports.size(), 0u);

    reports.clear();
    ASSERT_EQ(database->GetCompletedReports(&reports),
              CrashReportDatabase::kNoError);
    EXPECT_EQ(reports.size(), 0u);
  }

  {
    CrashpadClient::SetFirstChanceExceptionHandler(nullptr);

    CRASHPAD_SIMULATE_CRASH();

    std::vector<CrashReportDatabase::Report> reports;
    ASSERT_EQ(database->GetPendingReports(&reports),
              CrashReportDatabase::kNoError);
    EXPECT_EQ(reports.size(), 1u);

    reports.clear();
    ASSERT_EQ(database->GetCompletedReports(&reports),
              CrashReportDatabase::kNoError);
    EXPECT_EQ(reports.size(), 0u);
  }
}

constexpr char kTestAnnotationName[] = "name_of_annotation";
constexpr char kTestAnnotationValue[] = "value_of_annotation";

void ValidateDump(const CrashReportDatabase::UploadReport* report) {
  ProcessSnapshotMinidump minidump_snapshot;
  ASSERT_TRUE(minidump_snapshot.Initialize(report->Reader()));

  for (const ModuleSnapshot* module : minidump_snapshot.Modules()) {
    for (const AnnotationSnapshot& annotation : module->AnnotationObjects()) {
      if (static_cast<Annotation::Type>(annotation.type) !=
          Annotation::Type::kString) {
        continue;
      }

      if (annotation.name == kTestAnnotationName) {
        std::string value(
            reinterpret_cast<const char*>(annotation.value.data()),
            annotation.value.size());
        EXPECT_EQ(value, kTestAnnotationValue);
        return;
      }
    }
  }
  ADD_FAILURE();
}

CRASHPAD_CHILD_TEST_MAIN(StartHandlerAtCrashChild) {
  FileHandle in = StdioFileHandle(StdioStream::kStandardInput);

  VMSize temp_dir_length;
  CheckedReadFileExactly(in, &temp_dir_length, sizeof(temp_dir_length));

  std::string temp_dir(temp_dir_length, '\0');
  CheckedReadFileExactly(in, &temp_dir[0], temp_dir_length);

  base::FilePath handler_path = TestPaths::Executable().DirName().Append(
      FILE_PATH_LITERAL("crashpad_handler"));

  crashpad::AnnotationList::Register();

  static StringAnnotation<32> test_annotation(kTestAnnotationName);
  test_annotation.Set(kTestAnnotationValue);

  crashpad::CrashpadClient client;
  if (!client.StartHandlerAtCrash(handler_path,
                                  base::FilePath(temp_dir),
                                  base::FilePath(),
                                  "",
                                  std::map<std::string, std::string>(),
                                  std::vector<std::string>())) {
    return EXIT_FAILURE;
  }

  __builtin_trap();

  NOTREACHED();
  return EXIT_SUCCESS;
}

class StartHandlerAtCrashTest : public MultiprocessExec {
 public:
  StartHandlerAtCrashTest() : MultiprocessExec() {
    SetChildTestMainFunction("StartHandlerAtCrashChild");
    SetExpectedChildTerminationBuiltinTrap();
  }

 private:
  void MultiprocessParent() override {
    ScopedTempDir temp_dir;
    VMSize temp_dir_length = temp_dir.path().value().size();
    ASSERT_TRUE(LoggingWriteFile(
        WritePipeHandle(), &temp_dir_length, sizeof(temp_dir_length)));
    ASSERT_TRUE(LoggingWriteFile(
        WritePipeHandle(), temp_dir.path().value().data(), temp_dir_length));

    // Wait for child to finish.
    CheckedReadFileAtEOF(ReadPipeHandle());

    auto database = CrashReportDatabase::Initialize(temp_dir.path());
    ASSERT_TRUE(database);

    std::vector<CrashReportDatabase::Report> reports;
    ASSERT_EQ(database->GetCompletedReports(&reports),
              CrashReportDatabase::kNoError);
    EXPECT_EQ(reports.size(), 0u);

    reports.clear();
    ASSERT_EQ(database->GetPendingReports(&reports),
              CrashReportDatabase::kNoError);
    EXPECT_EQ(reports.size(), 1u);

    std::unique_ptr<const CrashReportDatabase::UploadReport> report;
    ASSERT_EQ(database->GetReportForUploading(reports[0].uuid, &report),
              CrashReportDatabase::kNoError);
    ValidateDump(report.get());
  }

  DISALLOW_COPY_AND_ASSIGN(StartHandlerAtCrashTest);
};

TEST(CrashpadClient, StartHandlerAtCrash) {
  StartHandlerAtCrashTest test;
  test.Run();
}

// Test state for starting the handler for another process.
class StartHandlerForClientTest {
 public:
  StartHandlerForClientTest() = default;
  ~StartHandlerForClientTest() = default;

  bool Initialize(bool sanitize) {
    sanitize_ = sanitize;

    int socks[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, socks) != 0) {
      PLOG(ERROR) << "socketpair";
      return false;
    }
    client_sock_.reset(socks[0]);
    server_sock_.reset(socks[1]);

    return true;
  }

  bool StartHandlerOnDemand() {
    char c;
    if (!LoggingReadFileExactly(server_sock_.get(), &c, sizeof(c))) {
      ADD_FAILURE();
      return false;
    }

    base::FilePath handler_path = TestPaths::Executable().DirName().Append(
        FILE_PATH_LITERAL("crashpad_handler"));

    CrashpadClient client;
    if (!client.StartHandlerForClient(handler_path,
                                        temp_dir_.path(),
                                        base::FilePath(),
                                        "",
                                        std::map<std::string, std::string>(),
                                        std::vector<std::string>(),
                                        server_sock_.get())) {
      ADD_FAILURE();
      return false;
    }

    return true;
  }

  void ExpectReport() {
    auto database =
        CrashReportDatabase::InitializeWithoutCreating(temp_dir_.path());
    ASSERT_TRUE(database);

    std::vector<CrashReportDatabase::Report> reports;
    ASSERT_EQ(database->GetCompletedReports(&reports),
              CrashReportDatabase::kNoError);
    EXPECT_EQ(reports.size(), 0u);

    reports.clear();
    ASSERT_EQ(database->GetPendingReports(&reports),
              CrashReportDatabase::kNoError);
    if (sanitize_) {
      EXPECT_EQ(reports.size(), 0u);
    } else {
      EXPECT_EQ(reports.size(), 1u);
    }
  }

  bool InstallHandler() {
    auto signal_handler = SandboxedHandler::Get();
    return signal_handler->Initialize(client_sock_.get(), sanitize_);
  }

 private:
  // A signal handler that defers handler process startup to another, presumably
  // more privileged, process.
  class SandboxedHandler {
   public:
    static SandboxedHandler* Get() {
      static SandboxedHandler* instance = new SandboxedHandler();
      return instance;
    }

    bool Initialize(FileHandle client_sock, bool sanitize) {
      client_sock_ = client_sock;
      sanitize_ = sanitize;
      return Signals::InstallCrashHandlers(HandleCrash, 0, nullptr);
    }

   private:
    SandboxedHandler() = default;
    ~SandboxedHandler() = delete;

    static void HandleCrash(int signo, siginfo_t* siginfo, void* context) {
      auto state = Get();

      char c;
      CHECK(LoggingWriteFile(state->client_sock_, &c, sizeof(c)));

      ExceptionInformation exception_information;
      exception_information.siginfo_address =
          FromPointerCast<decltype(exception_information.siginfo_address)>(
              siginfo);
      exception_information.context_address =
          FromPointerCast<decltype(exception_information.context_address)>(
              context);
      exception_information.thread_id = syscall(SYS_gettid);

      ClientInformation info;
      info.exception_information_address =
          FromPointerCast<decltype(info.exception_information_address)>(
              &exception_information);

      SanitizationInformation sanitization_info = {};
      if (state->sanitize_) {
        info.sanitization_information_address =
            FromPointerCast<VMAddress>(&sanitization_info);
        // Target a non-module address to prevent a crash dump.
        sanitization_info.target_module_address =
            FromPointerCast<VMAddress>(&sanitization_info);
      }

      ExceptionHandlerClient handler_client(state->client_sock_);
      CHECK_EQ(handler_client.RequestCrashDump(info), 0);

      Signals::RestoreHandlerAndReraiseSignalOnReturn(siginfo, nullptr);
    }

    FileHandle client_sock_;
    bool sanitize_;

    DISALLOW_COPY_AND_ASSIGN(SandboxedHandler);
  };

  ScopedTempDir temp_dir_;
  ScopedFileHandle client_sock_;
  ScopedFileHandle server_sock_;
  bool sanitize_;

  DISALLOW_COPY_AND_ASSIGN(StartHandlerForClientTest);
};

// Tests starting the handler for a child process.
class StartHandlerForChildTest : public Multiprocess {
 public:
  StartHandlerForChildTest() = default;
  ~StartHandlerForChildTest() = default;

  bool Initialize(bool sanitize) {
    SetExpectedChildTerminationBuiltinTrap();
    return test_state_.Initialize(sanitize);
  }

 private:
  void MultiprocessParent() {
    ASSERT_TRUE(test_state_.StartHandlerOnDemand());

    // Wait for chlid to finish.
    CheckedReadFileAtEOF(ReadPipeHandle());

    test_state_.ExpectReport();
  }

  void MultiprocessChild() {
    CHECK(test_state_.InstallHandler());

    __builtin_trap();

    NOTREACHED();
  }

  StartHandlerForClientTest test_state_;

  DISALLOW_COPY_AND_ASSIGN(StartHandlerForChildTest);
};

TEST(CrashpadClient, StartHandlerForChild) {
  StartHandlerForChildTest test;
  ASSERT_TRUE(test.Initialize(/* sanitize= */ false));
  test.Run();
}

TEST(CrashpadClient, SanitizedChild) {
  StartHandlerForChildTest test;
  ASSERT_TRUE(test.Initialize(/* sanitize= */ true));
  test.Run();
}

}  // namespace
}  // namespace test
}  // namespace crashpad
