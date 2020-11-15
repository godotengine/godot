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

#ifndef CRASHPAD_UTIL_WIN_NTSTATUS_LOGGING_H_
#define CRASHPAD_UTIL_WIN_NTSTATUS_LOGGING_H_

#include <windows.h>

#include "base/logging.h"
#include "base/macros.h"

namespace logging {

class NtstatusLogMessage : public logging::LogMessage {
 public:
  NtstatusLogMessage(
#if defined(MINI_CHROMIUM_BASE_LOGGING_H_)
      const char* function,
#endif
      const char* file_path,
      int line,
      LogSeverity severity,
      DWORD ntstatus);
  ~NtstatusLogMessage();

 private:
  DWORD ntstatus_;

  DISALLOW_COPY_AND_ASSIGN(NtstatusLogMessage);
};

}  // namespace logging

#define NTSTATUS_LOG_STREAM(severity, ntstatus) \
  COMPACT_GOOGLE_LOG_EX_##severity(NtstatusLogMessage, ntstatus).stream()

#if defined(MINI_CHROMIUM_BASE_LOGGING_H_)

#define NTSTATUS_VLOG_STREAM(verbose_level, ntstatus)                    \
  logging::NtstatusLogMessage(                                           \
      __PRETTY_FUNCTION__, __FILE__, __LINE__, -verbose_level, ntstatus) \
      .stream()

#else

#define NTSTATUS_VLOG_STREAM(verbose_level, ntstatus)                       \
  logging::NtstatusLogMessage(__FILE__, __LINE__, -verbose_level, ntstatus) \
      .stream()

#endif  // MINI_CHROMIUM_BASE_LOGGING_H_

#define NTSTATUS_LOG(severity, ntstatus) \
  LAZY_STREAM(NTSTATUS_LOG_STREAM(severity, ntstatus), LOG_IS_ON(severity))
#define NTSTATUS_LOG_IF(severity, condition, ntstatus) \
  LAZY_STREAM(NTSTATUS_LOG_STREAM(severity, ntstatus), \
              LOG_IS_ON(severity) && (condition))

#define NTSTATUS_VLOG(verbose_level, ntstatus)               \
  LAZY_STREAM(NTSTATUS_VLOG_STREAM(verbose_level, ntstatus), \
              VLOG_IS_ON(verbose_level))
#define NTSTATUS_VLOG_IF(verbose_level, condition, ntstatus) \
  LAZY_STREAM(NTSTATUS_VLOG_STREAM(verbose_level, ntstatus), \
              VLOG_IS_ON(verbose_level) && (condition))

#define NTSTATUS_CHECK(condition, ntstatus)                       \
  LAZY_STREAM(NTSTATUS_LOG_STREAM(FATAL, ntstatus), !(condition)) \
      << "Check failed: " #condition << ". "

#define NTSTATUS_DLOG(severity, ntstatus) \
  LAZY_STREAM(NTSTATUS_LOG_STREAM(severity, ntstatus), DLOG_IS_ON(severity))
#define NTSTATUS_DLOG_IF(severity, condition, ntstatus) \
  LAZY_STREAM(NTSTATUS_LOG_STREAM(severity, ntstatus),  \
              DLOG_IS_ON(severity) && (condition))

#define NTSTATUS_DVLOG(verbose_level, ntstatus)              \
  LAZY_STREAM(NTSTATUS_VLOG_STREAM(verbose_level, ntstatus), \
              DVLOG_IS_ON(verbose_level))
#define NTSTATUS_DVLOG_IF(verbose_level, condition, ntstatus) \
  LAZY_STREAM(NTSTATUS_VLOG_STREAM(verbose_level, ntstatus),  \
              DVLOG_IS_ON(verbose_level) && (condition))

#define NTSTATUS_DCHECK(condition, ntstatus)        \
  LAZY_STREAM(NTSTATUS_LOG_STREAM(FATAL, ntstatus), \
              DCHECK_IS_ON && !(condition))         \
      << "Check failed: " #condition << ". "

#endif  // CRASHPAD_UTIL_WIN_NTSTATUS_LOGGING_H_
