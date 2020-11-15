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

#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>

#include <memory>
#include <string>

#include "base/logging.h"
#include "base/strings/stringprintf.h"
#include "build/build_config.h"
#include "minidump/minidump_file_writer.h"
#include "tools/tool_support.h"
#include "util/file/file_writer.h"
#include "util/posix/drop_privileges.h"
#include "util/stdlib/string_number_conversion.h"

#if defined(OS_POSIX)
#include <unistd.h>
#endif

#if defined(OS_MACOSX)
#include <mach/mach.h>

#include "base/mac/scoped_mach_port.h"
#include "snapshot/mac/process_snapshot_mac.h"
#include "util/mach/scoped_task_suspend.h"
#include "util/mach/task_for_pid.h"
#elif defined(OS_WIN)
#include "base/strings/utf_string_conversions.h"
#include "snapshot/win/process_snapshot_win.h"
#include "util/win/scoped_process_suspend.h"
#include "util/win/xp_compat.h"
#elif defined(OS_FUCHSIA)
#include <lib/zx/process.h>

#include "snapshot/fuchsia/process_snapshot_fuchsia.h"
#include "util/fuchsia/koid_utilities.h"
#include "util/fuchsia/scoped_task_suspend.h"
#elif defined(OS_LINUX) || defined(OS_ANDROID)
#include "snapshot/linux/process_snapshot_linux.h"
#endif  // OS_MACOSX

namespace crashpad {
namespace {

void Usage(const base::FilePath& me) {
  fprintf(stderr,
"Usage: %" PRFilePath " [OPTION]... PID\n"
"Generate a minidump file containing a snapshot of a running process.\n"
"\n"
"  -r, --no-suspend   don't suspend the target process during dump generation\n"
"  -o, --output=FILE  write the minidump to FILE instead of minidump.PID\n"
"      --help         display this help and exit\n"
"      --version      output version information and exit\n",
          me.value().c_str());
  ToolSupport::UsageTail(me);
}

int GenerateDumpMain(int argc, char* argv[]) {
  const base::FilePath argv0(
      ToolSupport::CommandLineArgumentToFilePathStringType(argv[0]));
  const base::FilePath me(argv0.BaseName());

  enum OptionFlags {
    // “Short” (single-character) options.
    kOptionOutput = 'o',
    kOptionNoSuspend = 'r',

    // Long options without short equivalents.
    kOptionLastChar = 255,

    // Standard options.
    kOptionHelp = -2,
    kOptionVersion = -3,
  };

  struct {
    std::string dump_path;
    pid_t pid;
    bool suspend;
  } options = {};
  options.suspend = true;

  static constexpr option long_options[] = {
      {"no-suspend", no_argument, nullptr, kOptionNoSuspend},
      {"output", required_argument, nullptr, kOptionOutput},
      {"help", no_argument, nullptr, kOptionHelp},
      {"version", no_argument, nullptr, kOptionVersion},
      {nullptr, 0, nullptr, 0},
  };

  int opt;
  while ((opt = getopt_long(argc, argv, "o:r", long_options, nullptr)) != -1) {
    switch (opt) {
      case kOptionOutput:
        options.dump_path = optarg;
        break;
      case kOptionNoSuspend:
        options.suspend = false;
        break;
      case kOptionHelp:
        Usage(me);
        return EXIT_SUCCESS;
      case kOptionVersion:
        ToolSupport::Version(me);
        return EXIT_SUCCESS;
      default:
        ToolSupport::UsageHint(me, nullptr);
        return EXIT_FAILURE;
    }
  }
  argc -= optind;
  argv += optind;

  if (argc != 1) {
    ToolSupport::UsageHint(me, "PID is required");
    return EXIT_FAILURE;
  }

  if (!StringToNumber(argv[0], &options.pid) || options.pid <= 0) {
    fprintf(stderr,
            "%" PRFilePath ": invalid PID: %s\n",
            me.value().c_str(),
            argv[0]);
    return EXIT_FAILURE;
  }

#if defined(OS_MACOSX)
  task_t task = TaskForPID(options.pid);
  if (task == TASK_NULL) {
    return EXIT_FAILURE;
  }
  base::mac::ScopedMachSendRight task_owner(task);

  // This tool may have been installed as a setuid binary so that TaskForPID()
  // could succeed. Drop any privileges now that they’re no longer necessary.
  DropPrivileges();

  if (options.pid == getpid()) {
    if (options.suspend) {
      LOG(ERROR) << "cannot suspend myself";
      return EXIT_FAILURE;
    }
    LOG(WARNING) << "operating on myself";
  }
#elif defined(OS_WIN)
  ScopedKernelHANDLE process(
      OpenProcess(kXPProcessAllAccess, false, options.pid));
  if (!process.is_valid()) {
    PLOG(ERROR) << "could not open process " << options.pid;
    return EXIT_FAILURE;
  }
#elif defined(OS_FUCHSIA)
  zx::process process = GetProcessFromKoid(options.pid);
  if (!process.is_valid()) {
    LOG(ERROR) << "could not open process " << options.pid;
    return EXIT_FAILURE;
  }
#endif  // OS_MACOSX

  if (options.dump_path.empty()) {
    options.dump_path = base::StringPrintf("minidump.%d", options.pid);
  }

  {
#if defined(OS_MACOSX)
    std::unique_ptr<ScopedTaskSuspend> suspend;
    if (options.suspend) {
      suspend.reset(new ScopedTaskSuspend(task));
    }
#elif defined(OS_WIN)
    std::unique_ptr<ScopedProcessSuspend> suspend;
    if (options.suspend) {
      suspend.reset(new ScopedProcessSuspend(process.get()));
    }
#elif defined(OS_FUCHSIA)
    std::unique_ptr<ScopedTaskSuspend> suspend;
    if (options.suspend) {
      suspend.reset(new ScopedTaskSuspend(process));
    }
#endif  // OS_MACOSX

#if defined(OS_MACOSX)
    ProcessSnapshotMac process_snapshot;
    if (!process_snapshot.Initialize(task)) {
      return EXIT_FAILURE;
    }
#elif defined(OS_WIN)
    ProcessSnapshotWin process_snapshot;
    if (!process_snapshot.Initialize(process.get(),
                                     options.suspend
                                         ? ProcessSuspensionState::kSuspended
                                         : ProcessSuspensionState::kRunning,
                                     0,
                                     0)) {
      return EXIT_FAILURE;
    }
#elif defined(OS_FUCHSIA)
    ProcessSnapshotFuchsia process_snapshot;
    if (!process_snapshot.Initialize(process)) {
      return EXIT_FAILURE;
    }
#elif defined(OS_LINUX) || defined(OS_ANDROID)
    // TODO(jperaza): https://crashpad.chromium.org/bug/30.
    ProcessSnapshotLinux process_snapshot;
    if (!process_snapshot.Initialize(nullptr)) {
      return EXIT_FAILURE;
    }
#endif  // OS_MACOSX

    FileWriter file_writer;
    base::FilePath dump_path(
        ToolSupport::CommandLineArgumentToFilePathStringType(
            options.dump_path));
    if (!file_writer.Open(dump_path,
                          FileWriteMode::kTruncateOrCreate,
                          FilePermissions::kWorldReadable)) {
      return EXIT_FAILURE;
    }

    MinidumpFileWriter minidump;
    minidump.InitializeFromSnapshot(&process_snapshot);
    if (!minidump.WriteEverything(&file_writer)) {
      file_writer.Close();
      if (unlink(options.dump_path.c_str()) != 0) {
        PLOG(ERROR) << "unlink";
      }
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}

}  // namespace
}  // namespace crashpad

#if defined(OS_POSIX)
int main(int argc, char* argv[]) {
  return crashpad::GenerateDumpMain(argc, argv);
}
#elif defined(OS_WIN)
int wmain(int argc, wchar_t* argv[]) {
  return crashpad::ToolSupport::Wmain(argc, argv, crashpad::GenerateDumpMain);
}
#endif  // OS_POSIX
