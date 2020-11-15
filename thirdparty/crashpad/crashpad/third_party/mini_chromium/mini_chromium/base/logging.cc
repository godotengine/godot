// Copyright 2006-2008 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "base/logging.h"

#include <stdio.h>
#include <stdlib.h>

#include <iomanip>

#if defined(OS_POSIX)
#include <paths.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include "base/posix/safe_strerror.h"
#endif  // OS_POSIX

#if defined(OS_MACOSX)
#include <AvailabilityMacros.h>
#include <CoreFoundation/CoreFoundation.h>
#include <pthread.h>
#if !defined(MAC_OS_X_VERSION_10_12) || \
    MAC_OS_X_VERSION_MIN_REQUIRED < MAC_OS_X_VERSION_10_12
#include <asl.h>
#else
#include <os/log.h>
#endif
#elif defined(OS_LINUX)
#include <sys/syscall.h>
#include <sys/types.h>
#elif defined(OS_WIN)
#include <intrin.h>
#include <windows.h>
#elif defined(OS_FUCHSIA)
#include <zircon/process.h>
#include <zircon/syscalls.h>
#endif

#include "base/strings/string_util.h"
#include "base/strings/stringprintf.h"
#include "base/strings/utf_string_conversions.h"

namespace logging {

namespace {

const char* const log_severity_names[] = {
  "INFO",
  "WARNING",
  "ERROR",
  "ERROR_REPORT",
  "FATAL"
};

LogMessageHandlerFunction g_log_message_handler = nullptr;

}  // namespace

void SetLogMessageHandler(LogMessageHandlerFunction log_message_handler) {
  g_log_message_handler = log_message_handler;
}

LogMessageHandlerFunction GetLogMessageHandler() {
  return g_log_message_handler;
}

#if defined(OS_WIN)
std::string SystemErrorCodeToString(unsigned long error_code) {
  wchar_t msgbuf[256];
  DWORD flags = FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS |
                FORMAT_MESSAGE_MAX_WIDTH_MASK;
  DWORD len = FormatMessage(
      flags, nullptr, error_code, 0, msgbuf, arraysize(msgbuf), nullptr);
  if (len) {
    // Most system messages end in a period and a space. Remove the space if
    // it’s there, because the following StringPrintf() includes one.
    if (len >= 1 && msgbuf[len - 1] == ' ') {
      msgbuf[len - 1] = '\0';
    }
    return base::StringPrintf("%s (%u)",
                              base::UTF16ToUTF8(msgbuf).c_str(), error_code);
  }
  return base::StringPrintf("Error %u while retrieving error %u",
                            GetLastError(),
                            error_code);
}
#endif  // OS_WIN

#if defined(OS_FUCHSIA)
zx_koid_t GetKoidForHandle(zx_handle_t handle) {
  // Get the 64-bit koid (unique kernel object ID) of the given handle.
  zx_koid_t koid = 0;
  zx_info_handle_basic_t info;
  if (zx_object_get_info(handle,
                         ZX_INFO_HANDLE_BASIC,
                         &info,
                         sizeof(info),
                         nullptr,
                         nullptr) == ZX_OK) {
    // If this fails, there's not much that can be done. As this is used only
    // for logging, leave it as 0, which is not a valid koid.
    koid = info.koid;
  }
  return koid;
}
#endif  // OS_FUCHSIA

LogMessage::LogMessage(const char* function,
                       const char* file_path,
                       int line,
                       LogSeverity severity)
    : stream_(),
      file_path_(file_path),
      message_start_(0),
      line_(line),
      severity_(severity) {
  Init(function);
}

LogMessage::LogMessage(const char* function,
                       const char* file_path,
                       int line,
                       std::string* result)
    : stream_(),
      file_path_(file_path),
      message_start_(0),
      line_(line),
      severity_(LOG_FATAL) {
  Init(function);
  stream_ << "Check failed: " << *result << ". ";
  delete result;
}

LogMessage::~LogMessage() {
  stream_ << std::endl;
  std::string str_newline(stream_.str());

  if (g_log_message_handler &&
      g_log_message_handler(
          severity_, file_path_, line_, message_start_, str_newline)) {
    return;
  }

  fprintf(stderr, "%s", str_newline.c_str());
  fflush(stderr);

#if defined(OS_MACOSX)
  const bool log_to_system = []() {
    struct stat stderr_stat;
    if (fstat(fileno(stderr), &stderr_stat) == -1) {
      return true;
    }
    if (!S_ISCHR(stderr_stat.st_mode)) {
      return false;
    }

    struct stat dev_null_stat;
    if (stat(_PATH_DEVNULL, &dev_null_stat) == -1) {
      return true;
    }

    return !S_ISCHR(dev_null_stat.st_mode) ||
           stderr_stat.st_rdev == dev_null_stat.st_rdev;
  }();

  if (log_to_system) {
    CFBundleRef main_bundle = CFBundleGetMainBundle();
    CFStringRef main_bundle_id_cf =
        main_bundle ? CFBundleGetIdentifier(main_bundle) : nullptr;

    std::string main_bundle_id_buf;
    const char* main_bundle_id = nullptr;

    if (main_bundle_id_cf) {
      main_bundle_id =
          CFStringGetCStringPtr(main_bundle_id_cf, kCFStringEncodingUTF8);
      if (!main_bundle_id) {
        // 1024 is from 10.10.5 CF-1153.18/CFBundle.c __CFBundleMainID__ (at
        // the point of use, not declaration).
        main_bundle_id_buf.resize(1024);
        if (!CFStringGetCString(main_bundle_id_cf,
                                &main_bundle_id_buf[0],
                                main_bundle_id_buf.size(),
                                kCFStringEncodingUTF8)) {
          main_bundle_id_buf.clear();
        } else {
          main_bundle_id = &main_bundle_id_buf[0];
        }
      }
    }

#if !defined(MAC_OS_X_VERSION_10_12) || \
    MAC_OS_X_VERSION_MIN_REQUIRED < MAC_OS_X_VERSION_10_12
    // Use ASL when this might run on pre-10.12 systems. Unified Logging
    // (os_log) was introduced in 10.12.

    const class ASLClient {
     public:
      explicit ASLClient(const char* asl_facility)
          : client_(asl_open(nullptr, asl_facility, ASL_OPT_NO_DELAY)) {}
      ~ASLClient() { asl_close(client_); }

      aslclient get() const { return client_; }

     private:
      aslclient client_;
      DISALLOW_COPY_AND_ASSIGN(ASLClient);
    } asl_client(main_bundle_id ? main_bundle_id : "com.apple.console");

    const class ASLMessage {
     public:
      ASLMessage() : message_(asl_new(ASL_TYPE_MSG)) {}
      ~ASLMessage() { asl_free(message_); }

      aslmsg get() const { return message_; }

     private:
      aslmsg message_;
      DISALLOW_COPY_AND_ASSIGN(ASLMessage);
    } asl_message;

    // By default, messages are only readable by the admin group. Explicitly
    // make them readable by the user generating the messages.
    char euid_string[12];
    snprintf(euid_string, arraysize(euid_string), "%d", geteuid());
    asl_set(asl_message.get(), ASL_KEY_READ_UID, euid_string);

    // Map Chrome log severities to ASL log levels.
    const char* const asl_level_string = [](LogSeverity severity) {
#define ASL_LEVEL_STR(level) ASL_LEVEL_STR_X(level)
#define ASL_LEVEL_STR_X(level) #level
        switch (severity) {
          case LOG_INFO:
            return ASL_LEVEL_STR(ASL_LEVEL_INFO);
          case LOG_WARNING:
            return ASL_LEVEL_STR(ASL_LEVEL_WARNING);
          case LOG_ERROR:
            return ASL_LEVEL_STR(ASL_LEVEL_ERR);
          case LOG_FATAL:
            return ASL_LEVEL_STR(ASL_LEVEL_CRIT);
          default:
            return severity < 0 ? ASL_LEVEL_STR(ASL_LEVEL_DEBUG)
                                : ASL_LEVEL_STR(ASL_LEVEL_NOTICE);
        }
#undef ASL_LEVEL_STR
#undef ASL_LEVEL_STR_X
    }(severity_);
    asl_set(asl_message.get(), ASL_KEY_LEVEL, asl_level_string);

    asl_set(asl_message.get(), ASL_KEY_MSG, str_newline.c_str());

    asl_send(asl_client.get(), asl_message.get());
#else
    // Use Unified Logging (os_log) when this will only run on 10.12 and later.
    // ASL is deprecated in 10.12.

    const class OSLog {
     public:
      explicit OSLog(const char* subsystem)
          : os_log_(subsystem ? os_log_create(subsystem, "chromium_logging")
                              : OS_LOG_DEFAULT) {}
      ~OSLog() {
        if (os_log_ != OS_LOG_DEFAULT) {
          os_release(os_log_);
        }
      }

      os_log_t get() const { return os_log_; }

     private:
      os_log_t os_log_;
      DISALLOW_COPY_AND_ASSIGN(OSLog);
    } log(main_bundle_id);

    const os_log_type_t os_log_type = [](LogSeverity severity) {
      switch (severity) {
        case LOG_INFO:
          return OS_LOG_TYPE_INFO;
        case LOG_WARNING:
          return OS_LOG_TYPE_DEFAULT;
        case LOG_ERROR:
          return OS_LOG_TYPE_ERROR;
        case LOG_FATAL:
          return OS_LOG_TYPE_FAULT;
        default:
          return severity < 0 ? OS_LOG_TYPE_DEBUG : OS_LOG_TYPE_DEFAULT;
      }
    }(severity_);

    os_log_with_type(log.get(), os_log_type, "%{public}s", str_newline.c_str());
#endif
  }
#elif defined(OS_WIN)
  OutputDebugString(base::UTF8ToUTF16(str_newline).c_str());
#endif  // OS_MACOSX

  if (severity_ == LOG_FATAL) {
#if defined(COMPILER_MSVC)
    __debugbreak();
    __ud2();
#elif defined(ARCH_CPU_X86_FAMILY)
    asm("int3; ud2;");
#elif defined(ARCH_CPU_ARMEL)
    asm("bkpt #0; udf #0;");
#elif defined(ARCH_CPU_ARM64)
    asm("brk #0; hlt #0;");
#else
    __builtin_trap();
#endif
  }
}

void LogMessage::Init(const char* function) {
  std::string file_name(file_path_);
#if defined(OS_WIN)
  size_t last_slash = file_name.find_last_of("\\/");
#else
  size_t last_slash = file_name.find_last_of('/');
#endif
  if (last_slash != std::string::npos) {
    file_name.assign(file_name.substr(last_slash + 1));
  }

#if defined(OS_FUCHSIA)
  zx_koid_t pid = GetKoidForHandle(zx_process_self());
#elif defined(OS_POSIX)
  pid_t pid = getpid();
#elif defined(OS_WIN)
  DWORD pid = GetCurrentProcessId();
#endif

#if defined(OS_MACOSX)
  uint64_t thread;
  pthread_threadid_np(pthread_self(), &thread);
#elif defined(OS_ANDROID)
  pid_t thread = gettid();
#elif defined(OS_LINUX)
  pid_t thread = syscall(__NR_gettid);
#elif defined(OS_WIN)
  DWORD thread = GetCurrentThreadId();
#elif defined(OS_FUCHSIA)
  zx_koid_t thread = GetKoidForHandle(zx_thread_self());
#endif

  stream_ << '['
          << pid
          << ':'
          << thread
          << ':'
          << std::setfill('0');

#if defined(OS_POSIX)
  timeval tv;
  gettimeofday(&tv, nullptr);
  tm local_time;
  localtime_r(&tv.tv_sec, &local_time);
  stream_ << std::setw(4) << local_time.tm_year + 1900
          << std::setw(2) << local_time.tm_mon + 1
          << std::setw(2) << local_time.tm_mday
          << ','
          << std::setw(2) << local_time.tm_hour
          << std::setw(2) << local_time.tm_min
          << std::setw(2) << local_time.tm_sec
          << '.'
          << std::setw(6) << tv.tv_usec;
#elif defined(OS_WIN)
  SYSTEMTIME local_time;
  GetLocalTime(&local_time);
  stream_ << std::setw(4) << local_time.wYear
          << std::setw(2) << local_time.wMonth
          << std::setw(2) << local_time.wDay
          << ','
          << std::setw(2) << local_time.wHour
          << std::setw(2) << local_time.wMinute
          << std::setw(2) << local_time.wSecond
          << '.'
          << std::setw(3) << local_time.wMilliseconds;
#endif

  stream_ << ':';

  if (severity_ >= 0) {
    stream_ << log_severity_names[severity_];
  } else {
    stream_ << "VERBOSE" << -severity_;
  }

  stream_ << ' '
          << file_name
          << ':'
          << line_
          << "] ";

  message_start_ = stream_.str().size();
}

#if defined(OS_WIN)

unsigned long GetLastSystemErrorCode() {
  return GetLastError();
}

Win32ErrorLogMessage::Win32ErrorLogMessage(const char* function,
                                           const char* file_path,
                                           int line,
                                           LogSeverity severity,
                                           unsigned long err)
    : LogMessage(function, file_path, line, severity), err_(err) {
}

Win32ErrorLogMessage::~Win32ErrorLogMessage() {
  stream() << ": " << SystemErrorCodeToString(err_);
}

#elif defined(OS_POSIX)

ErrnoLogMessage::ErrnoLogMessage(const char* function,
                                 const char* file_path,
                                 int line,
                                 LogSeverity severity,
                                 int err)
    : LogMessage(function, file_path, line, severity),
      err_(err) {
}

ErrnoLogMessage::~ErrnoLogMessage() {
  stream() << ": "
           << base::safe_strerror(err_)
           << " ("
           << err_
           << ")";
}

#endif  // OS_POSIX

}  // namespace logging
