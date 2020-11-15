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

#include <errno.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/types.h>
#include <time.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "base/files/file_path.h"
#include "base/logging.h"
#include "base/macros.h"
#include "base/numerics/safe_conversions.h"
#include "base/strings/utf_string_conversions.h"
#include "build/build_config.h"
#include "client/crash_report_database.h"
#include "client/settings.h"
#include "tools/tool_support.h"
#include "util/file/file_io.h"
#include "util/file/file_reader.h"
#include "util/misc/uuid.h"
#include "util/stdlib/string_number_conversion.h"

namespace crashpad {
namespace {

void Usage(const base::FilePath& me) {
  fprintf(stderr,
"Usage: %" PRFilePath " [OPTION]... PID\n"
"Operate on Crashpad crash report databases.\n"
"\n"
"      --create                    allow database at PATH to be created\n"
"  -d, --database=PATH             operate on the crash report database at PATH\n"
"      --show-client-id            show the client ID\n"
"      --show-uploads-enabled      show whether uploads are enabled\n"
"      --show-last-upload-attempt-time\n"
"                                  show the last-upload-attempt time\n"
"      --show-pending-reports      show reports eligible for upload\n"
"      --show-completed-reports    show reports not eligible for upload\n"
"      --show-all-report-info      with --show-*-reports, show more information\n"
"      --show-report=UUID          show report stored under UUID\n"
"      --set-uploads-enabled=BOOL  enable or disable uploads\n"
"      --set-last-upload-attempt-time=TIME\n"
"                                  set the last-upload-attempt time to TIME\n"
"      --new-report=PATH           submit a new report at PATH, or - for stdin\n"
"      --utc                       show and set UTC times instead of local\n"
"      --help                      display this help and exit\n"
"      --version                   output version information and exit\n",
          me.value().c_str());
  ToolSupport::UsageTail(me);
}

struct Options {
  std::vector<UUID> show_reports;
  std::vector<base::FilePath> new_report_paths;
  const char* database;
  const char* set_last_upload_attempt_time_string;
  time_t set_last_upload_attempt_time;
  bool create;
  bool show_client_id;
  bool show_uploads_enabled;
  bool show_last_upload_attempt_time;
  bool show_pending_reports;
  bool show_completed_reports;
  bool show_all_report_info;
  bool set_uploads_enabled;
  bool has_set_uploads_enabled;
  bool utc;
};

// Converts |string| to |boolean|, returning true if a conversion could be
// performed, and false without setting |boolean| if no conversion could be
// performed. Various string representations of a boolean are recognized
// case-insensitively.
bool StringToBool(const char* string, bool* boolean) {
  static constexpr const char* kFalseWords[] = {
      "0",
      "false",
      "no",
      "off",
      "disabled",
      "clear",
  };
  static constexpr const char* kTrueWords[] = {
      "1",
      "true",
      "yes",
      "on",
      "enabled",
      "set",
  };

  for (size_t index = 0; index < arraysize(kFalseWords); ++index) {
    if (strcasecmp(string, kFalseWords[index]) == 0) {
      *boolean = false;
      return true;
    }
  }

  for (size_t index = 0; index < arraysize(kTrueWords); ++index) {
    if (strcasecmp(string, kTrueWords[index]) == 0) {
      *boolean = true;
      return true;
    }
  }

  return false;
}

// Converts |boolean| to a string, either "true" or "false".
std::string BoolToString(bool boolean) {
  return std::string(boolean ? "true" : "false");
}

// Converts |string| to |out_time|, returning true if a conversion could be
// performed, and false without setting |boolean| if no conversion could be
// performed. Various time formats are recognized, including several string
// representations and a numeric time_t representation. The special |string|
// "never" is recognized as converted to a |out_time| value of 0; "now" is
// converted to the current time. |utc|, when true, causes |string| to be
// interpreted as a UTC time rather than a local time when the time zone is
// ambiguous.
bool StringToTime(const char* string, time_t* out_time, bool utc) {
  if (strcasecmp(string, "never") == 0) {
    *out_time = 0;
    return true;
  }

  if (strcasecmp(string, "now") == 0) {
    errno = 0;
    PCHECK(time(out_time) != -1 || errno == 0);
    return true;
  }

  const char* end = string + strlen(string);

  static constexpr const char* kFormats[] = {
      "%Y-%m-%d %H:%M:%S %Z",
      "%Y-%m-%d %H:%M:%S",
      "%+",
  };

  for (size_t index = 0; index < arraysize(kFormats); ++index) {
    tm time_tm;
    const char* strptime_result = strptime(string, kFormats[index], &time_tm);
    if (strptime_result == end) {
      time_t test_out_time;
      if (utc) {
        test_out_time = timegm(&time_tm);
      } else {
        test_out_time = mktime(&time_tm);
      }

      // mktime() is supposed to set errno in the event of an error, but support
      // for this is spotty, so there’s no way to distinguish between a true
      // time_t of -1 (1969-12-31 23:59:59 UTC) and an error. Assume error.
      //
      // See 10.11.5 Libc-1082.50.1/stdtime/FreeBSD/localtime.c and
      // glibc-2.24/time/mktime.c, which don’t set errno or save and restore
      // errno. Post-Android 7.1.0 Bionic is even more hopeless, setting errno
      // whenever the time conversion returns -1, even for valid input. See
      // libc/tzcode/localtime.c mktime(). Windows seems to get it right: see
      // 10.0.14393 SDK Source/ucrt/time/mktime.cpp.
      if (test_out_time != -1) {
        *out_time = test_out_time;
        return true;
      }
    }
  }

  int64_t int64_result;
  if (StringToNumber(string, &int64_result) &&
      base::IsValueInRangeForNumericType<time_t>(int64_result)) {
    *out_time = int64_result;
    return true;
  }

  return false;
}

// Converts |out_time| to a string, and returns it. |utc| determines whether the
// converted time will reference local time or UTC. If |out_time| is 0, the
// string "never" will be returned as a special case.
std::string TimeToString(time_t out_time, bool utc) {
  if (out_time == 0) {
    return std::string("never");
  }

  tm time_tm;
  if (utc) {
    PCHECK(gmtime_r(&out_time, &time_tm));
  } else {
    PCHECK(localtime_r(&out_time, &time_tm));
  }

  char string[64];
  CHECK_NE(
      strftime(string, arraysize(string), "%Y-%m-%d %H:%M:%S %Z", &time_tm),
      0u);

  return std::string(string);
}

// Shows information about a single |report|. |space_count| is the number of
// spaces to print before each line that is printed. |utc| determines whether
// times should be shown in UTC or the local time zone.
void ShowReport(const CrashReportDatabase::Report& report,
                size_t space_count,
                bool utc) {
  std::string spaces(space_count, ' ');

  printf("%sPath: %" PRFilePath "\n",
         spaces.c_str(),
         report.file_path.value().c_str());
  if (!report.id.empty()) {
    printf("%sRemote ID: %s\n", spaces.c_str(), report.id.c_str());
  }
  printf("%sCreation time: %s\n",
         spaces.c_str(),
         TimeToString(report.creation_time, utc).c_str());
  printf("%sUploaded: %s\n",
         spaces.c_str(),
         BoolToString(report.uploaded).c_str());
  printf("%sLast upload attempt time: %s\n",
         spaces.c_str(),
         TimeToString(report.last_upload_attempt_time, utc).c_str());
  printf("%sUpload attempts: %d\n", spaces.c_str(), report.upload_attempts);
}

// Shows information about a vector of |reports|. |space_count| is the number of
// spaces to print before each line that is printed. |options| will be consulted
// to determine whether to show expanded information
// (options.show_all_report_info) and what time zone to use when showing
// expanded information (options.utc).
void ShowReports(const std::vector<CrashReportDatabase::Report>& reports,
                 size_t space_count,
                 const Options& options) {
  std::string spaces(space_count, ' ');
  const char* colon = options.show_all_report_info ? ":" : "";

  for (const CrashReportDatabase::Report& report : reports) {
    printf("%s%s%s\n", spaces.c_str(), report.uuid.ToString().c_str(), colon);
    if (options.show_all_report_info) {
      ShowReport(report, space_count + 2, options.utc);
    }
  }
}

int DatabaseUtilMain(int argc, char* argv[]) {
  const base::FilePath argv0(
      ToolSupport::CommandLineArgumentToFilePathStringType(argv[0]));
  const base::FilePath me(argv0.BaseName());

  enum OptionFlags {
    // “Short” (single-character) options.
    kOptionDatabase = 'd',

    // Long options without short equivalents.
    kOptionLastChar = 255,
    kOptionCreate,
    kOptionShowClientID,
    kOptionShowUploadsEnabled,
    kOptionShowLastUploadAttemptTime,
    kOptionShowPendingReports,
    kOptionShowCompletedReports,
    kOptionShowAllReportInfo,
    kOptionShowReport,
    kOptionSetUploadsEnabled,
    kOptionSetLastUploadAttemptTime,
    kOptionNewReport,
    kOptionUTC,

    // Standard options.
    kOptionHelp = -2,
    kOptionVersion = -3,
  };

  static constexpr option long_options[] = {
      {"create", no_argument, nullptr, kOptionCreate},
      {"database", required_argument, nullptr, kOptionDatabase},
      {"show-client-id", no_argument, nullptr, kOptionShowClientID},
      {"show-uploads-enabled", no_argument, nullptr, kOptionShowUploadsEnabled},
      {"show-last-upload-attempt-time",
       no_argument,
       nullptr,
       kOptionShowLastUploadAttemptTime},
      {"show-pending-reports", no_argument, nullptr, kOptionShowPendingReports},
      {"show-completed-reports",
       no_argument,
       nullptr,
       kOptionShowCompletedReports},
      {"show-all-report-info", no_argument, nullptr, kOptionShowAllReportInfo},
      {"show-report", required_argument, nullptr, kOptionShowReport},
      {"set-uploads-enabled",
       required_argument,
       nullptr,
       kOptionSetUploadsEnabled},
      {"set-last-upload-attempt-time",
       required_argument,
       nullptr,
       kOptionSetLastUploadAttemptTime},
      {"new-report", required_argument, nullptr, kOptionNewReport},
      {"utc", no_argument, nullptr, kOptionUTC},
      {"help", no_argument, nullptr, kOptionHelp},
      {"version", no_argument, nullptr, kOptionVersion},
      {nullptr, 0, nullptr, 0},
  };

  Options options = {};

  int opt;
  while ((opt = getopt_long(argc, argv, "d:", long_options, nullptr)) != -1) {
    switch (opt) {
      case kOptionCreate: {
        options.create = true;
        break;
      }
      case kOptionDatabase: {
        options.database = optarg;
        break;
      }
      case kOptionShowClientID: {
        options.show_client_id = true;
        break;
      }
      case kOptionShowUploadsEnabled: {
        options.show_uploads_enabled = true;
        break;
      }
      case kOptionShowLastUploadAttemptTime: {
        options.show_last_upload_attempt_time = true;
        break;
      }
      case kOptionShowPendingReports: {
        options.show_pending_reports = true;
        break;
      }
      case kOptionShowCompletedReports: {
        options.show_completed_reports = true;
        break;
      }
      case kOptionShowAllReportInfo: {
        options.show_all_report_info = true;
        break;
      }
      case kOptionShowReport: {
        UUID uuid;
        if (!uuid.InitializeFromString(optarg)) {
          ToolSupport::UsageHint(me, "--show-report requires a UUID");
          return EXIT_FAILURE;
        }
        options.show_reports.push_back(uuid);
        break;
      }
      case kOptionSetUploadsEnabled: {
        if (!StringToBool(optarg, &options.set_uploads_enabled)) {
          ToolSupport::UsageHint(me, "--set-uploads-enabled requires a BOOL");
          return EXIT_FAILURE;
        }
        options.has_set_uploads_enabled = true;
        break;
      }
      case kOptionSetLastUploadAttemptTime: {
        options.set_last_upload_attempt_time_string = optarg;
        break;
      }
      case kOptionNewReport: {
        options.new_report_paths.push_back(base::FilePath(
            ToolSupport::CommandLineArgumentToFilePathStringType(optarg)));
        break;
      }
      case kOptionUTC: {
        options.utc = true;
        break;
      }
      case kOptionHelp: {
        Usage(me);
        return EXIT_SUCCESS;
      }
      case kOptionVersion: {
        ToolSupport::Version(me);
        return EXIT_SUCCESS;
      }
      default: {
        ToolSupport::UsageHint(me, nullptr);
        return EXIT_FAILURE;
      }
    }
  }
  argc -= optind;
  argv += optind;

  if (!options.database) {
    ToolSupport::UsageHint(me, "--database is required");
    return EXIT_FAILURE;
  }

  // This conversion couldn’t happen in the option-processing loop above because
  // it depends on options.utc, which may have been set after
  // options.set_last_upload_attempt_time_string.
  if (options.set_last_upload_attempt_time_string) {
    if (!StringToTime(options.set_last_upload_attempt_time_string,
                      &options.set_last_upload_attempt_time,
                      options.utc)) {
      ToolSupport::UsageHint(me,
                             "--set-last-upload-attempt-time requires a TIME");
      return EXIT_FAILURE;
    }
  }

  // --new-report is treated as a show operation because it produces output.
  const size_t show_operations = options.show_client_id +
                                 options.show_uploads_enabled +
                                 options.show_last_upload_attempt_time +
                                 options.show_pending_reports +
                                 options.show_completed_reports +
                                 options.show_reports.size() +
                                 options.new_report_paths.size();
  const size_t set_operations =
      options.has_set_uploads_enabled +
      (options.set_last_upload_attempt_time_string != nullptr);

  if ((options.create ? 1 : 0) + show_operations + set_operations == 0) {
    ToolSupport::UsageHint(me, "nothing to do");
    return EXIT_FAILURE;
  }

  std::unique_ptr<CrashReportDatabase> database;
  base::FilePath database_path = base::FilePath(
      ToolSupport::CommandLineArgumentToFilePathStringType(options.database));
  if (options.create) {
    database = CrashReportDatabase::Initialize(database_path);
  } else {
    database = CrashReportDatabase::InitializeWithoutCreating(database_path);
  }
  if (!database) {
    return EXIT_FAILURE;
  }

  Settings* settings = database->GetSettings();

  // Handle the “show” options before the “set” options so that when they’re
  // specified together, the “show” option reflects the initial state.

  if (options.show_client_id) {
    UUID client_id;
    if (!settings->GetClientID(&client_id)) {
      return EXIT_FAILURE;
    }

    const char* prefix = (show_operations > 1) ? "Client ID: " : "";

    printf("%s%s\n", prefix, client_id.ToString().c_str());
  }

  if (options.show_uploads_enabled) {
    bool uploads_enabled;
    if (!settings->GetUploadsEnabled(&uploads_enabled)) {
      return EXIT_FAILURE;
    }

    const char* prefix = (show_operations > 1) ? "Uploads enabled: " : "";

    printf("%s%s\n", prefix, BoolToString(uploads_enabled).c_str());
  }

  if (options.show_last_upload_attempt_time) {
    time_t last_upload_attempt_time;
    if (!settings->GetLastUploadAttemptTime(&last_upload_attempt_time)) {
      return EXIT_FAILURE;
    }

    const char* prefix =
        (show_operations > 1) ? "Last upload attempt time: " : "";

    printf("%s%s (%ld)\n",
           prefix,
           TimeToString(last_upload_attempt_time, options.utc).c_str(),
           static_cast<long>(last_upload_attempt_time));
  }

  if (options.show_pending_reports) {
    std::vector<CrashReportDatabase::Report> pending_reports;
    if (database->GetPendingReports(&pending_reports) !=
        CrashReportDatabase::kNoError) {
      return EXIT_FAILURE;
    }

    if (show_operations > 1) {
      printf("Pending reports:\n");
    }

    ShowReports(pending_reports, show_operations > 1 ? 2 : 0, options);
  }

  if (options.show_completed_reports) {
    std::vector<CrashReportDatabase::Report> completed_reports;
    if (database->GetCompletedReports(&completed_reports) !=
        CrashReportDatabase::kNoError) {
      return EXIT_FAILURE;
    }

    if (show_operations > 1) {
      printf("Completed reports:\n");
    }

    ShowReports(completed_reports, show_operations > 1 ? 2 : 0, options);
  }

  for (const UUID& uuid : options.show_reports) {
    CrashReportDatabase::Report report;
    CrashReportDatabase::OperationStatus status =
        database->LookUpCrashReport(uuid, &report);
    if (status == CrashReportDatabase::kNoError) {
      if (show_operations > 1) {
        printf("Report %s:\n", uuid.ToString().c_str());
      }
      ShowReport(report, show_operations > 1 ? 2 : 0, options.utc);
    } else if (status == CrashReportDatabase::kReportNotFound) {
      // If only asked to do one thing, a failure to find the single requested
      // report should result in a failure exit status.
      if (show_operations + set_operations == 1) {
        fprintf(
            stderr, "%" PRFilePath ": Report not found\n", me.value().c_str());
        return EXIT_FAILURE;
      }
      printf("Report %s not found\n", uuid.ToString().c_str());
    } else {
      return EXIT_FAILURE;
    }
  }

  if (options.has_set_uploads_enabled &&
      !settings->SetUploadsEnabled(options.set_uploads_enabled)) {
    return EXIT_FAILURE;
  }

  if (options.set_last_upload_attempt_time_string &&
      !settings->SetLastUploadAttemptTime(
          options.set_last_upload_attempt_time)) {
    return EXIT_FAILURE;
  }

  bool used_stdin = false;
  for (const base::FilePath new_report_path : options.new_report_paths) {
    std::unique_ptr<FileReaderInterface> file_reader;

    if (new_report_path.value() == FILE_PATH_LITERAL("-")) {
      if (used_stdin) {
        fprintf(stderr,
                "%" PRFilePath
                ": Only one --new-report may be read from standard input\n",
                me.value().c_str());
        return EXIT_FAILURE;
      }
      used_stdin = true;
      file_reader.reset(new WeakFileHandleFileReader(
          StdioFileHandle(StdioStream::kStandardInput)));
    } else {
      std::unique_ptr<FileReader> file_path_reader(new FileReader());
      if (!file_path_reader->Open(new_report_path)) {
        return EXIT_FAILURE;
      }

      file_reader = std::move(file_path_reader);
    }

    std::unique_ptr<CrashReportDatabase::NewReport> new_report;
    CrashReportDatabase::OperationStatus status =
        database->PrepareNewCrashReport(&new_report);
    if (status != CrashReportDatabase::kNoError) {
      return EXIT_FAILURE;
    }

    char buf[4096];
    FileOperationResult read_result;
    do {
      read_result = file_reader->Read(buf, sizeof(buf));
      if (read_result < 0) {
        return EXIT_FAILURE;
      }
      if (read_result > 0 && !new_report->Writer()->Write(buf, read_result)) {
        return EXIT_FAILURE;
      }
    } while (read_result > 0);

    UUID uuid;
    status = database->FinishedWritingCrashReport(std::move(new_report), &uuid);
    if (status != CrashReportDatabase::kNoError) {
      return EXIT_FAILURE;
    }

    const char* prefix = (show_operations > 1) ? "New report ID: " : "";
    printf("%s%s\n", prefix, uuid.ToString().c_str());
  }

  return EXIT_SUCCESS;
}

}  // namespace
}  // namespace crashpad

#if defined(OS_POSIX)
int main(int argc, char* argv[]) {
  return crashpad::DatabaseUtilMain(argc, argv);
}
#elif defined(OS_WIN)
int wmain(int argc, wchar_t* argv[]) {
  return crashpad::ToolSupport::Wmain(argc, argv, crashpad::DatabaseUtilMain);
}
#endif  // OS_POSIX
