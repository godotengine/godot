// Copyright 2016 The Crashpad Authors. All rights reserved.
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

#ifndef CRASHPAD_UTIL_MISC_METRICS_H_
#define CRASHPAD_UTIL_MISC_METRICS_H_

#include <inttypes.h>

#include "base/macros.h"
#include "util/file/file_io.h"

namespace crashpad {

//! \brief Container class to hold shared UMA metrics integration points.
//!
//! Each static function in this class will call a `UMA_*` from
//! `base/metrics/histogram_macros.h`. When building Crashpad standalone,
//! (against mini_chromium), these macros do nothing. When built against
//! Chromium's base, they allow integration with its metrics system.
class Metrics {
 public:
  //! \brief Values for CrashReportPending().
  //!
  //! \note These are used as metrics enumeration values, so new values should
  //!     always be added at the end, before PendingReportReason::kMaxValue.
  enum class PendingReportReason : int32_t {
    //! \brief A report was newly created and is ready for upload.
    kNewlyCreated = 0,

    //! \brief The user manually requested the report be uploaded.
    kUserInitiated = 1,

    //! \brief The number of values in this enumeration; not a valid value.
    kMaxValue
  };

  //! \brief Reports when a crash upload has entered the pending state.
  static void CrashReportPending(PendingReportReason reason);

  //! \brief Reports the size of a crash report file in bytes. Should be called
  //!     when a new report is written to disk.
  static void CrashReportSize(FileOffset size);

  //! \brief Reports on a crash upload attempt, and if it succeeded.
  static void CrashUploadAttempted(bool successful);

  //! \brief Values for CrashUploadSkipped().
  //!
  //! \note These are used as metrics enumeration values, so new values should
  //!     always be added at the end, before CrashSkippedReason::kMaxValue.
  enum class CrashSkippedReason : int32_t {
    //! \brief Crash uploading is disabled.
    kUploadsDisabled = 0,

    //! \brief There was another upload too recently, so this one was throttled.
    kUploadThrottled = 1,

    //! \brief The report had an unexpected timestamp.
    kUnexpectedTime = 2,

    //! \brief The database reported an error, likely due to a filesystem
    //!     problem.
    kDatabaseError = 3,

    //! \brief The upload of the crash failed during communication with the
    //!     server.
    kUploadFailed = 4,

    //! \brief There was an error between accessing the report from the database
    //!     and uploading it to the crash server.
    kPrepareForUploadFailed = 5,

    //! \brief The number of values in this enumeration; not a valid value.
    kMaxValue
  };

  //! \brief Reports when a report is moved to the completed state in the
  //!     database, without the report being uploadad.
  static void CrashUploadSkipped(CrashSkippedReason reason);

  //! \brief The result of capturing an exception.
  //!
  //! \note These are used as metrics enumeration values, so new values should
  //!     always be added at the end, before CaptureResult::kMaxValue.
  enum class CaptureResult : int32_t {
    //! \brief The exception capture succeeded normally.
    kSuccess = 0,

    //! \brief Unexpected exception behavior.
    //!
    //! This value is only used on macOS.
    kUnexpectedExceptionBehavior = 1,

    //! \brief Failed due to attempt to suspend self.
    //!
    //! This value is only used on macOS.
    kFailedDueToSuspendSelf = 2,

    //! \brief The process snapshot could not be captured.
    kSnapshotFailed = 3,

    //! \brief The exception could not be initialized.
    kExceptionInitializationFailed = 4,

    //! \brief The attempt to prepare a new crash report in the crash database
    //!     failed.
    kPrepareNewCrashReportFailed = 5,

    //! \brief Writing the minidump to disk failed.
    kMinidumpWriteFailed = 6,

    //! \brief There was a database error in attempt to complete the report.
    kFinishedWritingCrashReportFailed = 7,

    //! \brief An attempt to directly `ptrace` the target failed.
    //!
    //! This value is only used on Linux/Android.
    kDirectPtraceFailed = 8,

    //! \brief An attempt to `ptrace` via a PtraceBroker failed.
    //!
    //! This value is only used on Linux/Android.
    kBrokeredPtraceFailed = 9,

    //! \brief Sanitization was requested but could not be initialized.
    kSanitizationInitializationFailed = 10,

    //! \brief Sanitization caused this crash dump to be skipped.
    kSkippedDueToSanitization = 11,

    //! \brief The number of values in this enumeration; not a valid value.
    kMaxValue
  };

  //! \brief Reports on the outcome of capturing a report in the exception
  //!     handler. Should be called on all capture completion paths.
  static void ExceptionCaptureResult(CaptureResult result);

  //! \brief The exception code for an exception was retrieved.
  //!
  //! These values are OS-specific, and correspond to
  //! MINIDUMP_EXCEPTION::ExceptionCode.
  static void ExceptionCode(uint32_t exception_code);

  //! \brief The exception handler server started capturing an exception.
  static void ExceptionEncountered();

  //! \brief An important event in a handler process’ lifetime.
  //!
  //! \note These are used as metrics enumeration values, so new values should
  //!     always be added at the end, before LifetimeMilestone::kMaxValue.
  enum class LifetimeMilestone : int32_t {
    //! \brief The handler process started.
    kStarted = 0,

    //! \brief The handler process exited normally and cleanly.
    kExitedNormally,

    //! \brief The handler process exited early, but was successful in
    //!     performing some non-default action on user request.
    kExitedEarly,

    //! \brief The handler process exited with a failure code.
    kFailed,

    //! \brief The handler process was forcibly terminated.
    kTerminated,

    //! \brief The handler process crashed.
    kCrashed,

    //! \brief The number of values in this enumeration; not a valid value.
    kMaxValue
  };

  //! \brief Records a handler start/exit/crash event.
  static void HandlerLifetimeMilestone(LifetimeMilestone milestone);

  //! \brief The handler process crashed with the given exception code.
  //!
  //! This is currently only reported on Windows.
  static void HandlerCrashed(uint32_t exception_code);

 private:
  DISALLOW_IMPLICIT_CONSTRUCTORS(Metrics);
};

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_MISC_METRICS_H_
