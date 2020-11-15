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

#include "handler/handler_main.h"

#include <errno.h>
#include <getopt.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/types.h>

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "base/auto_reset.h"
#include "base/compiler_specific.h"
#include "base/files/file_path.h"
#include "base/files/scoped_file.h"
#include "base/logging.h"
#include "base/metrics/persistent_histogram_allocator.h"
#include "base/scoped_generic.h"
#include "base/strings/string_number_conversions.h"
#include "base/strings/stringprintf.h"
#include "base/strings/utf_string_conversions.h"
#include "build/build_config.h"
#include "client/crash_report_database.h"
#include "client/crashpad_client.h"
#include "client/crashpad_info.h"
#include "client/prune_crash_reports.h"
#include "client/simple_string_dictionary.h"
#include "handler/crash_report_upload_thread.h"
#include "handler/prune_crash_reports_thread.h"
#include "tools/tool_support.h"
#include "util/file/file_io.h"
#include "util/misc/address_types.h"
#include "util/misc/metrics.h"
#include "util/misc/paths.h"
#include "util/numeric/in_range_cast.h"
#include "util/stdlib/map_insert.h"
#include "util/stdlib/string_number_conversion.h"
#include "util/string/split_string.h"
#include "util/synchronization/semaphore.h"

#if defined(OS_LINUX) || defined(OS_ANDROID)
#include <unistd.h>

#include "handler/linux/crash_report_exception_handler.h"
#include "handler/linux/exception_handler_server.h"
#include "util/posix/signals.h"
#elif defined(OS_MACOSX)
#include <libgen.h>
#include <signal.h>

#include "base/mac/scoped_mach_port.h"
#include "handler/mac/crash_report_exception_handler.h"
#include "handler/mac/exception_handler_server.h"
#include "handler/mac/file_limit_annotation.h"
#include "util/mach/child_port_handshake.h"
#include "util/mach/mach_extensions.h"
#include "util/posix/close_stdio.h"
#include "util/posix/signals.h"
#elif defined(OS_WIN)
#include <windows.h>

#include "handler/win/crash_report_exception_handler.h"
#include "util/win/exception_handler_server.h"
#include "util/win/handle.h"
#include "util/win/initial_client_data.h"
#include "util/win/session_end_watcher.h"
#elif defined(OS_FUCHSIA)
#include <zircon/process.h>
#include <zircon/processargs.h>

#include "handler/fuchsia/crash_report_exception_handler.h"
#include "handler/fuchsia/exception_handler_server.h"
#elif defined(OS_LINUX)
#include "handler/linux/crash_report_exception_handler.h"
#include "handler/linux/exception_handler_server.h"
#endif  // OS_MACOSX

namespace crashpad {

namespace {

void Usage(const base::FilePath& me) {
  fprintf(stderr,
"Usage: %" PRFilePath " [OPTION]...\n"
"Crashpad's exception handler server.\n"
"\n"
"      --annotation=KEY=VALUE  set a process annotation in each crash report\n"
"      --database=PATH         store the crash report database at PATH\n"
#if defined(OS_MACOSX)
"      --handshake-fd=FD       establish communication with the client over FD\n"
#endif  // OS_MACOSX
#if defined(OS_WIN)
"      --initial-client-data=HANDLE_request_crash_dump,\n"
"                            HANDLE_request_non_crash_dump,\n"
"                            HANDLE_non_crash_dump_completed,\n"
"                            HANDLE_pipe,\n"
"                            HANDLE_client_process,\n"
"                            Address_crash_exception_information,\n"
"                            Address_non_crash_exception_information,\n"
"                            Address_debug_critical_section\n"
"                              use precreated data to register initial client\n"
#endif  // OS_WIN
#if defined(OS_MACOSX)
"      --mach-service=SERVICE  register SERVICE with the bootstrap server\n"
#endif  // OS_MACOSX
"      --metrics-dir=DIR       store metrics files in DIR (only in Chromium)\n"
"      --monitor-self          run a second handler to catch crashes in the first\n"
"      --monitor-self-annotation=KEY=VALUE\n"
"                              set a module annotation in the handler\n"
"      --monitor-self-argument=ARGUMENT\n"
"                              provide additional arguments to the second handler\n"
"      --no-identify-client-via-url\n"
"                              when uploading crash report, don't add\n"
"                              client-identifying arguments to URL\n"
"      --no-periodic-tasks     don't scan for new reports or prune the database\n"
"      --no-rate-limit         don't rate limit crash uploads\n"
"      --no-upload-gzip        don't use gzip compression when uploading\n"
#if defined(OS_WIN)
"      --pipe-name=PIPE        communicate with the client over PIPE\n"
#endif  // OS_WIN
#if defined(OS_MACOSX)
"      --reset-own-crash-exception-port-to-system-default\n"
"                              reset the server's exception handler to default\n"
#endif  // OS_MACOSX
#if defined(OS_LINUX) || defined(OS_ANDROID)
"      --trace-parent-with-exception=EXCEPTION_INFORMATION_ADDRESS\n"
"                              request a dump for the handler's parent process\n"
"      --initial-client-fd=FD  a socket connected to a client.\n"
"      --sanitization_information=SANITIZATION_INFORMATION_ADDRESS\n"
"                              the address of a SanitizationInformation struct.\n"
#endif  // OS_LINUX || OS_ANDROID
"      --url=URL               send crash reports to this Breakpad server URL,\n"
"                              only if uploads are enabled for the database\n"
"      --help                  display this help and exit\n"
"      --version               output version information and exit\n",
          me.value().c_str());
  ToolSupport::UsageTail(me);
}

struct Options {
  std::map<std::string, std::string> annotations;
  std::map<std::string, std::string> monitor_self_annotations;
  std::string url;
  base::FilePath database;
  base::FilePath metrics_dir;
  std::vector<std::string> monitor_self_arguments;
#if defined(OS_MACOSX)
  std::string mach_service;
  int handshake_fd;
  bool reset_own_crash_exception_port_to_system_default;
#elif defined(OS_LINUX) || defined(OS_ANDROID)
  VMAddress exception_information_address;
  int initial_client_fd;
  VMAddress sanitization_information_address;
#elif defined(OS_WIN)
  std::string pipe_name;
  InitialClientData initial_client_data;
#endif  // OS_MACOSX
  bool identify_client_via_url;
  bool monitor_self;
  bool periodic_tasks;
  bool rate_limit;
  bool upload_gzip;
};

// Splits |key_value| on '=' and inserts the resulting key and value into |map|.
// If |key_value| has the wrong format, logs an error and returns false. If the
// key is already in the map, logs a warning, replaces the existing value, and
// returns true. If the key and value were inserted into the map, returns true.
// |argument| is used to give context to logged messages.
bool AddKeyValueToMap(std::map<std::string, std::string>* map,
                      const std::string& key_value,
                      const char* argument) {
  std::string key;
  std::string value;
  if (!SplitStringFirst(key_value, '=', &key, &value)) {
    LOG(ERROR) << argument << " requires KEY=VALUE";
    return false;
  }

  std::string old_value;
  if (!MapInsertOrReplace(map, key, value, &old_value)) {
    LOG(WARNING) << argument << " has duplicate key " << key
                 << ", discarding value " << old_value;
  }
  return true;
}

// Calls Metrics::HandlerLifetimeMilestone, but only on the first call. This is
// to prevent multiple exit events from inadvertently being recorded, which
// might happen if a crash occurs during destruction in what would otherwise be
// a normal exit, or if a CallMetricsRecordNormalExit object is destroyed after
// something else logs an exit event.
void MetricsRecordExit(Metrics::LifetimeMilestone milestone) {
  static bool once = [](Metrics::LifetimeMilestone milestone) {
    Metrics::HandlerLifetimeMilestone(milestone);
    return true;
  }(milestone);
  ALLOW_UNUSED_LOCAL(once);
}

// Calls MetricsRecordExit() to record a failure, and returns EXIT_FAILURE for
// the convenience of callers in main() which can simply write “return
// ExitFailure();”.
int ExitFailure() {
  MetricsRecordExit(Metrics::LifetimeMilestone::kFailed);
  return EXIT_FAILURE;
}

class CallMetricsRecordNormalExit {
 public:
  CallMetricsRecordNormalExit() {}
  ~CallMetricsRecordNormalExit() {
    MetricsRecordExit(Metrics::LifetimeMilestone::kExitedNormally);
  }

 private:
  DISALLOW_COPY_AND_ASSIGN(CallMetricsRecordNormalExit);
};

#if defined(OS_MACOSX) || defined(OS_LINUX) || defined(OS_ANDROID)

Signals::OldActions g_old_crash_signal_handlers;

void HandleCrashSignal(int sig, siginfo_t* siginfo, void* context) {
  MetricsRecordExit(Metrics::LifetimeMilestone::kCrashed);

  // Is siginfo->si_code useful? The only interesting values on macOS are 0 (not
  // useful, signals generated asynchronously such as by kill() or raise()) and
  // small positive numbers (useful, signal generated via a hardware fault). The
  // standard specifies these other constants, and while xnu never uses them,
  // they are intended to denote signals generated asynchronously and are
  // included here. Additionally, existing practice on other systems
  // (acknowledged by the standard) is for negative numbers to indicate that a
  // signal was generated asynchronously. Although xnu does not do this, allow
  // for the possibility for completeness.
  bool si_code_valid = !(siginfo->si_code <= 0 ||
                         siginfo->si_code == SI_USER ||
                         siginfo->si_code == SI_QUEUE ||
                         siginfo->si_code == SI_TIMER ||
                         siginfo->si_code == SI_ASYNCIO ||
                         siginfo->si_code == SI_MESGQ);

  // 0x5343 = 'SC', signifying “signal and code”, disambiguates from the schema
  // used by ExceptionCodeForMetrics(). That system primarily uses Mach
  // exception types and codes, which are not available to a POSIX signal
  // handler. It does provide a way to encode only signal numbers, but does so
  // with the understanding that certain “raw” signals would not be encountered
  // without a Mach exception. Furthermore, it does not allow siginfo->si_code
  // to be encoded, because that’s not available to Mach exception handlers. It
  // would be a shame to lose that information available to a POSIX signal
  // handler.
  int metrics_code = 0x53430000 | (InRangeCast<uint8_t>(sig, 0xff) << 8);
  if (si_code_valid) {
    metrics_code |= InRangeCast<uint8_t>(siginfo->si_code, 0xff);
  }
  Metrics::HandlerCrashed(metrics_code);

  struct sigaction* old_action =
      g_old_crash_signal_handlers.ActionForSignal(sig);
  Signals::RestoreHandlerAndReraiseSignalOnReturn(siginfo, old_action);
}

void HandleTerminateSignal(int sig, siginfo_t* siginfo, void* context) {
  MetricsRecordExit(Metrics::LifetimeMilestone::kTerminated);
  Signals::RestoreHandlerAndReraiseSignalOnReturn(siginfo, nullptr);
}

#if defined(OS_MACOSX)

void ReinstallCrashHandler() {
  // This is used to re-enable the metrics-recording crash handler after
  // MonitorSelf() sets up a Crashpad exception handler. On macOS, the
  // metrics-recording handler uses signals and the Crashpad handler uses Mach
  // exceptions, so there’s nothing to re-enable.
}

void InstallCrashHandler() {
  Signals::InstallCrashHandlers(HandleCrashSignal, 0, nullptr);

  // Not a crash handler, but close enough.
  Signals::InstallTerminateHandlers(HandleTerminateSignal, 0, nullptr);
}

struct ResetSIGTERMTraits {
  static struct sigaction* InvalidValue() {
    return nullptr;
  }

  static void Free(struct sigaction* sa) {
    int rv = sigaction(SIGTERM, sa, nullptr);
    PLOG_IF(ERROR, rv != 0) << "sigaction";
  }
};
using ScopedResetSIGTERM =
    base::ScopedGeneric<struct sigaction*, ResetSIGTERMTraits>;

ExceptionHandlerServer* g_exception_handler_server;

// This signal handler is only operative when being run from launchd.
void HandleSIGTERM(int sig, siginfo_t* siginfo, void* context) {
  // Don’t call MetricsRecordExit(). This is part of the normal exit path when
  // running from launchd.

  DCHECK(g_exception_handler_server);
  g_exception_handler_server->Stop();
}

#else

void ReinstallCrashHandler() {
  // This is used to re-enable the metrics-recording crash handler after
  // MonitorSelf() sets up a Crashpad signal handler.
  Signals::InstallCrashHandlers(
      HandleCrashSignal, 0, &g_old_crash_signal_handlers);
}

void InstallCrashHandler() {
  ReinstallCrashHandler();

  Signals::InstallTerminateHandlers(HandleTerminateSignal, 0, nullptr);
}

#endif  // OS_MACOSX

#elif defined(OS_WIN)

LONG(WINAPI* g_original_exception_filter)(EXCEPTION_POINTERS*) = nullptr;

LONG WINAPI UnhandledExceptionHandler(EXCEPTION_POINTERS* exception_pointers) {
  MetricsRecordExit(Metrics::LifetimeMilestone::kCrashed);
  Metrics::HandlerCrashed(exception_pointers->ExceptionRecord->ExceptionCode);

  if (g_original_exception_filter)
    return g_original_exception_filter(exception_pointers);
  else
    return EXCEPTION_CONTINUE_SEARCH;
}

// Handles events like Control-C and Control-Break on a console.
BOOL WINAPI ConsoleHandler(DWORD console_event) {
  MetricsRecordExit(Metrics::LifetimeMilestone::kTerminated);
  return false;
}

// Handles a WM_ENDSESSION message sent when the user session is ending.
class TerminateHandler final : public SessionEndWatcher {
 public:
  TerminateHandler() : SessionEndWatcher() {}
  ~TerminateHandler() override {}

 private:
  // SessionEndWatcher:
  void SessionEnding() override {
    MetricsRecordExit(Metrics::LifetimeMilestone::kTerminated);
  }

  DISALLOW_COPY_AND_ASSIGN(TerminateHandler);
};

void ReinstallCrashHandler() {
  // This is used to re-enable the metrics-recording crash handler after
  // MonitorSelf() sets up a Crashpad exception handler. The Crashpad handler
  // takes over the UnhandledExceptionFilter, so reinstall the metrics-recording
  // one.
  g_original_exception_filter =
      SetUnhandledExceptionFilter(&UnhandledExceptionHandler);
}

void InstallCrashHandler() {
  ReinstallCrashHandler();

  // These are termination handlers, not crash handlers, but that’s close
  // enough. Note that destroying the TerminateHandler would wait for its thread
  // to exit, which isn’t necessary or desirable.
  SetConsoleCtrlHandler(ConsoleHandler, true);
  static TerminateHandler* terminate_handler = new TerminateHandler();
  ALLOW_UNUSED_LOCAL(terminate_handler);
}

#elif defined(OS_FUCHSIA)

void InstallCrashHandler() {
  // There's nothing to do here. Crashes in this process will already be caught
  // here because this handler process is in the same job that has had its
  // exception port bound.

  // TODO(scottmg): This should collect metrics on handler crashes, at a
  // minimum. https://crashpad.chromium.org/bug/230.
}

void ReinstallCrashHandler() {
  // TODO(scottmg): Fuchsia: https://crashpad.chromium.org/bug/196
  NOTREACHED();
}

#endif  // OS_MACOSX

void MonitorSelf(const Options& options) {
  base::FilePath executable_path;
  if (!Paths::Executable(&executable_path)) {
    return;
  }

  if (std::find(options.monitor_self_arguments.begin(),
                options.monitor_self_arguments.end(),
                "--monitor-self") != options.monitor_self_arguments.end()) {
    LOG(WARNING) << "--monitor-self-argument=--monitor-self is not supported";
    return;
  }
  std::vector<std::string> extra_arguments(options.monitor_self_arguments);
  if (!options.identify_client_via_url) {
    extra_arguments.push_back("--no-identify-client-via-url");
  }
  extra_arguments.push_back("--no-periodic-tasks");
  if (!options.rate_limit) {
    extra_arguments.push_back("--no-rate-limit");
  }
  if (!options.upload_gzip) {
    extra_arguments.push_back("--no-upload-gzip");
  }
  for (const auto& iterator : options.monitor_self_annotations) {
    extra_arguments.push_back(
        base::StringPrintf("--monitor-self-annotation=%s=%s",
                           iterator.first.c_str(),
                           iterator.second.c_str()));
  }

  // Don’t use options.metrics_dir. The current implementation only allows one
  // instance of crashpad_handler to be writing metrics at a time, and it should
  // be the primary instance.
  CrashpadClient crashpad_client;
#if defined(OS_LINUX) || defined(OS_ANDROID)
  if (!crashpad_client.StartHandlerAtCrash(executable_path,
                                           options.database,
                                           base::FilePath(),
                                           options.url,
                                           options.annotations,
                                           extra_arguments)) {
    return;
  }
#else
  if (!crashpad_client.StartHandler(executable_path,
                                    options.database,
                                    base::FilePath(),
                                    options.url,
                                    options.annotations,
                                    extra_arguments,
                                    true,
                                    false)) {
    return;
  }
#endif

  // Make sure that appropriate metrics will be recorded on crash before this
  // process is terminated.
  ReinstallCrashHandler();
}

class ScopedStoppable {
 public:
  ScopedStoppable() = default;

  ~ScopedStoppable() {
    if (stoppable_) {
      stoppable_->Stop();
    }
  }

  void Reset(Stoppable* stoppable) { stoppable_.reset(stoppable); }

  Stoppable* Get() { return stoppable_.get(); }

 private:
  std::unique_ptr<Stoppable> stoppable_;

  DISALLOW_COPY_AND_ASSIGN(ScopedStoppable);
};

}  // namespace

int HandlerMain(int argc,
                char* argv[],
                const UserStreamDataSources* user_stream_sources) {
  InstallCrashHandler();
  CallMetricsRecordNormalExit metrics_record_normal_exit;

  const base::FilePath argv0(
      ToolSupport::CommandLineArgumentToFilePathStringType(argv[0]));
  const base::FilePath me(argv0.BaseName());

  enum OptionFlags {
    // Long options without short equivalents.
    kOptionLastChar = 255,
    kOptionAnnotation,
    kOptionDatabase,
#if defined(OS_MACOSX)
    kOptionHandshakeFD,
#endif  // OS_MACOSX
#if defined(OS_WIN)
    kOptionInitialClientData,
#endif  // OS_WIN
#if defined(OS_MACOSX)
    kOptionMachService,
#endif  // OS_MACOSX
    kOptionMetrics,
    kOptionMonitorSelf,
    kOptionMonitorSelfAnnotation,
    kOptionMonitorSelfArgument,
    kOptionNoIdentifyClientViaUrl,
    kOptionNoPeriodicTasks,
    kOptionNoRateLimit,
    kOptionNoUploadGzip,
#if defined(OS_WIN)
    kOptionPipeName,
#endif  // OS_WIN
#if defined(OS_MACOSX)
    kOptionResetOwnCrashExceptionPortToSystemDefault,
#endif  // OS_MACOSX
#if defined(OS_LINUX) || defined(OS_ANDROID)
    kOptionTraceParentWithException,
    kOptionInitialClientFD,
    kOptionSanitizationInformation,
#endif
    kOptionURL,

    // Standard options.
    kOptionHelp = -2,
    kOptionVersion = -3,
  };

  static constexpr option long_options[] = {
    {"annotation", required_argument, nullptr, kOptionAnnotation},
    {"database", required_argument, nullptr, kOptionDatabase},
#if defined(OS_MACOSX)
    {"handshake-fd", required_argument, nullptr, kOptionHandshakeFD},
#endif  // OS_MACOSX
#if defined(OS_WIN)
    {"initial-client-data",
     required_argument,
     nullptr,
     kOptionInitialClientData},
#endif  // OS_MACOSX
#if defined(OS_MACOSX)
    {"mach-service", required_argument, nullptr, kOptionMachService},
#endif  // OS_MACOSX
    {"metrics-dir", required_argument, nullptr, kOptionMetrics},
    {"monitor-self", no_argument, nullptr, kOptionMonitorSelf},
    {"monitor-self-annotation",
     required_argument,
     nullptr,
     kOptionMonitorSelfAnnotation},
    {"monitor-self-argument",
     required_argument,
     nullptr,
     kOptionMonitorSelfArgument},
    {"no-identify-client-via-url",
     no_argument,
     nullptr,
     kOptionNoIdentifyClientViaUrl},
    {"no-periodic-tasks", no_argument, nullptr, kOptionNoPeriodicTasks},
    {"no-rate-limit", no_argument, nullptr, kOptionNoRateLimit},
    {"no-upload-gzip", no_argument, nullptr, kOptionNoUploadGzip},
#if defined(OS_WIN)
    {"pipe-name", required_argument, nullptr, kOptionPipeName},
#endif  // OS_WIN
#if defined(OS_MACOSX)
    {"reset-own-crash-exception-port-to-system-default",
     no_argument,
     nullptr,
     kOptionResetOwnCrashExceptionPortToSystemDefault},
#endif  // OS_MACOSX
#if defined(OS_LINUX) || defined(OS_ANDROID)
    {"trace-parent-with-exception",
     required_argument,
     nullptr,
     kOptionTraceParentWithException},
    {"initial-client-fd", required_argument, nullptr, kOptionInitialClientFD},
    {"sanitization-information",
     required_argument,
     nullptr,
     kOptionSanitizationInformation},
#endif  // OS_LINUX || OS_ANDROID
    {"url", required_argument, nullptr, kOptionURL},
    {"help", no_argument, nullptr, kOptionHelp},
    {"version", no_argument, nullptr, kOptionVersion},
    {nullptr, 0, nullptr, 0},
  };

  Options options = {};
#if defined(OS_MACOSX)
  options.handshake_fd = -1;
#endif
  options.identify_client_via_url = true;
  options.periodic_tasks = true;
  options.rate_limit = true;
  options.upload_gzip = true;
#if defined(OS_LINUX) || defined(OS_ANDROID)
  options.exception_information_address = 0;
  options.initial_client_fd = kInvalidFileHandle;
  options.sanitization_information_address = 0;
#endif

  int opt;
  while ((opt = getopt_long(argc, argv, "", long_options, nullptr)) != -1) {
    switch (opt) {
      case kOptionAnnotation: {
        if (!AddKeyValueToMap(&options.annotations, optarg, "--annotation")) {
          return ExitFailure();
        }
        break;
      }
      case kOptionDatabase: {
        options.database = base::FilePath(
            ToolSupport::CommandLineArgumentToFilePathStringType(optarg));
        break;
      }
#if defined(OS_MACOSX)
      case kOptionHandshakeFD: {
        if (!StringToNumber(optarg, &options.handshake_fd) ||
            options.handshake_fd < 0) {
          ToolSupport::UsageHint(me,
                                 "--handshake-fd requires a file descriptor");
          return ExitFailure();
        }
        break;
      }
      case kOptionMachService: {
        options.mach_service = optarg;
        break;
      }
#endif  // OS_MACOSX
#if defined(OS_WIN)
      case kOptionInitialClientData: {
        if (!options.initial_client_data.InitializeFromString(optarg)) {
          ToolSupport::UsageHint(
              me, "failed to parse --initial-client-data");
          return ExitFailure();
        }
        break;
      }
#endif  // OS_WIN
      case kOptionMetrics: {
        options.metrics_dir = base::FilePath(
            ToolSupport::CommandLineArgumentToFilePathStringType(optarg));
        break;
      }
      case kOptionMonitorSelf: {
        options.monitor_self = true;
        break;
      }
      case kOptionMonitorSelfAnnotation: {
        if (!AddKeyValueToMap(&options.monitor_self_annotations,
                              optarg,
                              "--monitor-self-annotation")) {
          return ExitFailure();
        }
        break;
      }
      case kOptionMonitorSelfArgument: {
        options.monitor_self_arguments.push_back(optarg);
        break;
      }
      case kOptionNoIdentifyClientViaUrl: {
        options.identify_client_via_url = false;
        break;
      }
      case kOptionNoPeriodicTasks: {
        options.periodic_tasks = false;
        break;
      }
      case kOptionNoRateLimit: {
        options.rate_limit = false;
        break;
      }
      case kOptionNoUploadGzip: {
        options.upload_gzip = false;
        break;
      }
#if defined(OS_WIN)
      case kOptionPipeName: {
        options.pipe_name = optarg;
        break;
      }
#endif  // OS_WIN
#if defined(OS_MACOSX)
      case kOptionResetOwnCrashExceptionPortToSystemDefault: {
        options.reset_own_crash_exception_port_to_system_default = true;
        break;
      }
#endif  // OS_MACOSX
#if defined(OS_LINUX) || defined(OS_ANDROID)
      case kOptionTraceParentWithException: {
        if (!StringToNumber(optarg, &options.exception_information_address)) {
          ToolSupport::UsageHint(
              me, "failed to parse --trace-parent-with-exception");
          return ExitFailure();
        }
        break;
      }
      case kOptionInitialClientFD: {
        if (!base::StringToInt(optarg, &options.initial_client_fd)) {
          ToolSupport::UsageHint(me, "failed to parse --initial-client-fd");
          return ExitFailure();
        }
        break;
      }
      case kOptionSanitizationInformation: {
        if (!StringToNumber(optarg,
                            &options.sanitization_information_address)) {
          ToolSupport::UsageHint(me,
                                 "failed to parse --sanitization-information");
          return ExitFailure();
        }
        break;
      }
#endif  // OS_LINUX || OS_ANDROID
      case kOptionURL: {
        options.url = optarg;
        break;
      }
      case kOptionHelp: {
        Usage(me);
        MetricsRecordExit(Metrics::LifetimeMilestone::kExitedEarly);
        return EXIT_SUCCESS;
      }
      case kOptionVersion: {
        ToolSupport::Version(me);
        MetricsRecordExit(Metrics::LifetimeMilestone::kExitedEarly);
        return EXIT_SUCCESS;
      }
      default: {
        ToolSupport::UsageHint(me, nullptr);
        return ExitFailure();
      }
    }
  }
  argc -= optind;
  argv += optind;

#if defined(OS_MACOSX)
  if (options.handshake_fd < 0 && options.mach_service.empty()) {
    ToolSupport::UsageHint(me, "--handshake-fd or --mach-service is required");
    return ExitFailure();
  }
  if (options.handshake_fd >= 0 && !options.mach_service.empty()) {
    ToolSupport::UsageHint(
        me, "--handshake-fd and --mach-service are incompatible");
    return ExitFailure();
  }
#elif defined(OS_WIN)
  if (!options.initial_client_data.IsValid() && options.pipe_name.empty()) {
    ToolSupport::UsageHint(me,
                           "--initial-client-data or --pipe-name is required");
    return ExitFailure();
  }
  if (options.initial_client_data.IsValid() && !options.pipe_name.empty()) {
    ToolSupport::UsageHint(
        me, "--initial-client-data and --pipe-name are incompatible");
    return ExitFailure();
  }
#elif defined(OS_LINUX) || defined(OS_ANDROID)
  if (!options.exception_information_address &&
      options.initial_client_fd == kInvalidFileHandle) {
    ToolSupport::UsageHint(
        me, "--trace-parent-with-exception or --initial_client_fd is required");
    return ExitFailure();
  }
  if (options.sanitization_information_address &&
      !options.exception_information_address) {
    ToolSupport::UsageHint(
        me,
        "--sanitization_information requires --trace-parent-with-exception");
    return ExitFailure();
  }
#endif  // OS_MACOSX

  if (options.database.empty()) {
    ToolSupport::UsageHint(me, "--database is required");
    return ExitFailure();
  }

  if (argc) {
    ToolSupport::UsageHint(me, nullptr);
    return ExitFailure();
  }

#if defined(OS_MACOSX)
  if (options.reset_own_crash_exception_port_to_system_default) {
    CrashpadClient::UseSystemDefaultHandler();
  }
#endif  // OS_MACOSX

  if (options.monitor_self) {
    MonitorSelf(options);
  }

  if (!options.monitor_self_annotations.empty()) {
    // Establish these annotations even if --monitor-self is not present, in
    // case something such as generate_dump wants to try to access them later.
    //
    // If the handler is part of a multi-purpose executable, simple annotations
    // may already be present for this module. If they are, use them.
    CrashpadInfo* crashpad_info = CrashpadInfo::GetCrashpadInfo();
    SimpleStringDictionary* module_annotations =
        crashpad_info->simple_annotations();
    if (!module_annotations) {
      module_annotations = new SimpleStringDictionary();
      crashpad_info->set_simple_annotations(module_annotations);
    }

    for (const auto& iterator : options.monitor_self_annotations) {
      module_annotations->SetKeyValue(iterator.first.c_str(),
                                      iterator.second.c_str());
    }
  }

  std::unique_ptr<CrashReportDatabase> database(
      CrashReportDatabase::Initialize(options.database));
  if (!database) {
    return ExitFailure();
  }

  ScopedStoppable upload_thread;
  if (!options.url.empty()) {
    // TODO(scottmg): options.rate_limit should be removed when we have a
    // configurable database setting to control upload limiting.
    // See https://crashpad.chromium.org/bug/23.
    CrashReportUploadThread::Options upload_thread_options;
    upload_thread_options.identify_client_via_url =
        options.identify_client_via_url;
    upload_thread_options.rate_limit = options.rate_limit;
    upload_thread_options.upload_gzip = options.upload_gzip;
    upload_thread_options.watch_pending_reports = options.periodic_tasks;

    upload_thread.Reset(new CrashReportUploadThread(
        database.get(), options.url, upload_thread_options));
    upload_thread.Get()->Start();
  }

  CrashReportExceptionHandler exception_handler(
      database.get(),
      static_cast<CrashReportUploadThread*>(upload_thread.Get()),
      &options.annotations,
#if defined(OS_FUCHSIA)
      // TODO(scottmg): Process level file attachments, and for all platforms.
      nullptr,
#endif
      user_stream_sources);

 #if defined(OS_LINUX) || defined(OS_ANDROID)
  if (options.exception_information_address) {
    ClientInformation info;
    info.exception_information_address = options.exception_information_address;
    info.sanitization_information_address =
        options.sanitization_information_address;
    return exception_handler.HandleException(getppid(), info) ? EXIT_SUCCESS
                                                              : ExitFailure();
  }
#endif  // OS_LINUX || OS_ANDROID

  ScopedStoppable prune_thread;
  if (options.periodic_tasks) {
    prune_thread.Reset(new PruneCrashReportThread(
        database.get(), PruneCondition::GetDefault()));
    prune_thread.Get()->Start();
  }

#if defined(OS_MACOSX)
  if (options.mach_service.empty()) {
    // Don’t do this when being run by launchd. See launchd.plist(5).
    CloseStdinAndStdout();
  }

  base::mac::ScopedMachReceiveRight receive_right;

  if (options.handshake_fd >= 0) {
    receive_right.reset(
        ChildPortHandshake::RunServerForFD(
            base::ScopedFD(options.handshake_fd),
            ChildPortHandshake::PortRightType::kReceiveRight));
  } else if (!options.mach_service.empty()) {
    receive_right = BootstrapCheckIn(options.mach_service);
  }

  if (!receive_right.is_valid()) {
    return ExitFailure();
  }

  ExceptionHandlerServer exception_handler_server(
      std::move(receive_right), !options.mach_service.empty());
  base::AutoReset<ExceptionHandlerServer*> reset_g_exception_handler_server(
      &g_exception_handler_server, &exception_handler_server);

  struct sigaction old_sigterm_action;
  ScopedResetSIGTERM reset_sigterm;
  if (!options.mach_service.empty()) {
    // When running from launchd, no no-senders notification could ever be
    // triggered, because launchd maintains a send right to the service. When
    // launchd wants the job to exit, it will send a SIGTERM. See
    // launchd.plist(5).
    //
    // Set up a SIGTERM handler that will call exception_handler_server.Stop().
    // This replaces the HandleTerminateSignal handler for SIGTERM.
    if (Signals::InstallHandler(
            SIGTERM, HandleSIGTERM, 0, &old_sigterm_action)) {
      reset_sigterm.reset(&old_sigterm_action);
    }
  }

  RecordFileLimitAnnotation();
#elif defined(OS_WIN)
  // Shut down as late as possible relative to programs we're watching.
  if (!SetProcessShutdownParameters(0x100, SHUTDOWN_NORETRY))
    PLOG(ERROR) << "SetProcessShutdownParameters";

  ExceptionHandlerServer exception_handler_server(!options.pipe_name.empty());

  if (!options.pipe_name.empty()) {
    exception_handler_server.SetPipeName(base::UTF8ToUTF16(options.pipe_name));
  }
#elif defined(OS_FUCHSIA)
  // These handles are logically "moved" into these variables when retrieved by
  // zx_take_startup_handle(). Both are given to ExceptionHandlerServer which
  // owns them in this process. There is currently no "connect-later" mode on
  // Fuchsia, all the binding must be done by the client before starting
  // crashpad_handler.
  zx::job root_job(zx_take_startup_handle(PA_HND(PA_USER0, 0)));
  if (!root_job.is_valid()) {
    LOG(ERROR) << "no process handle passed in startup handle 0";
    return EXIT_FAILURE;
  }

  zx::port exception_port(zx_take_startup_handle(PA_HND(PA_USER0, 1)));
  if (!exception_port.is_valid()) {
    LOG(ERROR) << "no exception port handle passed in startup handle 1";
    return EXIT_FAILURE;
  }

  ExceptionHandlerServer exception_handler_server(std::move(root_job),
                                                  std::move(exception_port));
#elif defined(OS_LINUX) || defined(OS_ANDROID)
  ExceptionHandlerServer exception_handler_server;
#endif  // OS_MACOSX

  base::GlobalHistogramAllocator* histogram_allocator = nullptr;
  if (!options.metrics_dir.empty()) {
    static constexpr char kMetricsName[] = "CrashpadMetrics";
    constexpr size_t kMetricsFileSize = 1 << 20;
    if (base::GlobalHistogramAllocator::CreateWithActiveFileInDir(
            options.metrics_dir, kMetricsFileSize, 0, kMetricsName)) {
      histogram_allocator = base::GlobalHistogramAllocator::Get();
      histogram_allocator->CreateTrackingHistograms(kMetricsName);
    }
  }

  Metrics::HandlerLifetimeMilestone(Metrics::LifetimeMilestone::kStarted);

#if defined(OS_WIN)
  if (options.initial_client_data.IsValid()) {
    exception_handler_server.InitializeWithInheritedDataForInitialClient(
        options.initial_client_data, &exception_handler);
  }
#elif defined(OS_LINUX) || defined(OS_ANDROID)
  if (options.initial_client_fd == kInvalidFileHandle ||
             !exception_handler_server.InitializeWithClient(
                 ScopedFileHandle(options.initial_client_fd))) {
    return ExitFailure();
  }
#endif  // OS_WIN

  exception_handler_server.Run(&exception_handler);

  return EXIT_SUCCESS;
}

}  // namespace crashpad
