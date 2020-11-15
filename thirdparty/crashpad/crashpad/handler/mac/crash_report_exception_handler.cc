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

#include "handler/mac/crash_report_exception_handler.h"

#include <utility>
#include <vector>

#include "base/logging.h"
#include "base/mac/mach_logging.h"
#include "base/mac/scoped_mach_port.h"
#include "base/strings/stringprintf.h"
#include "client/settings.h"
#include "handler/mac/file_limit_annotation.h"
#include "minidump/minidump_file_writer.h"
#include "minidump/minidump_user_extension_stream_data_source.h"
#include "snapshot/crashpad_info_client_options.h"
#include "snapshot/mac/process_snapshot_mac.h"
#include "util/file/file_writer.h"
#include "util/mach/exc_client_variants.h"
#include "util/mach/exception_behaviors.h"
#include "util/mach/exception_types.h"
#include "util/mach/mach_extensions.h"
#include "util/mach/mach_message.h"
#include "util/mach/scoped_task_suspend.h"
#include "util/mach/symbolic_constants_mach.h"
#include "util/misc/metrics.h"
#include "util/misc/tri_state.h"
#include "util/misc/uuid.h"

namespace crashpad {

CrashReportExceptionHandler::CrashReportExceptionHandler(
    CrashReportDatabase* database,
    CrashReportUploadThread* upload_thread,
    const std::map<std::string, std::string>* process_annotations,
    const UserStreamDataSources* user_stream_data_sources)
    : database_(database),
      upload_thread_(upload_thread),
      process_annotations_(process_annotations),
      user_stream_data_sources_(user_stream_data_sources) {}

CrashReportExceptionHandler::~CrashReportExceptionHandler() {
}

kern_return_t CrashReportExceptionHandler::CatchMachException(
    exception_behavior_t behavior,
    exception_handler_t exception_port,
    thread_t thread,
    task_t task,
    exception_type_t exception,
    const mach_exception_data_type_t* code,
    mach_msg_type_number_t code_count,
    thread_state_flavor_t* flavor,
    ConstThreadState old_state,
    mach_msg_type_number_t old_state_count,
    thread_state_t new_state,
    mach_msg_type_number_t* new_state_count,
    const mach_msg_trailer_t* trailer,
    bool* destroy_complex_request) {
  RecordFileLimitAnnotation();
  Metrics::ExceptionEncountered();
  Metrics::ExceptionCode(ExceptionCodeForMetrics(exception, code[0]));
  *destroy_complex_request = true;

  // The expected behavior is EXCEPTION_STATE_IDENTITY | MACH_EXCEPTION_CODES,
  // but it’s possible to deal with any exception behavior as long as it
  // carries identity information (valid thread and task ports).
  if (!ExceptionBehaviorHasIdentity(behavior)) {
    LOG(ERROR) << base::StringPrintf(
        "unexpected exception behavior %s, rejecting",
        ExceptionBehaviorToString(
            behavior, kUseFullName | kUnknownIsNumeric | kUseOr).c_str());
    Metrics::ExceptionCaptureResult(
        Metrics::CaptureResult::kUnexpectedExceptionBehavior);
    return KERN_FAILURE;
  } else if (behavior != (EXCEPTION_STATE_IDENTITY | kMachExceptionCodes)) {
    LOG(WARNING) << base::StringPrintf(
        "unexpected exception behavior %s, proceeding",
        ExceptionBehaviorToString(
            behavior, kUseFullName | kUnknownIsNumeric | kUseOr).c_str());
  }

  if (task == mach_task_self()) {
    LOG(ERROR) << "cannot suspend myself";
    Metrics::ExceptionCaptureResult(
        Metrics::CaptureResult::kFailedDueToSuspendSelf);
    return KERN_FAILURE;
  }

  ScopedTaskSuspend suspend(task);

  ProcessSnapshotMac process_snapshot;
  if (!process_snapshot.Initialize(task)) {
    Metrics::ExceptionCaptureResult(Metrics::CaptureResult::kSnapshotFailed);
    return KERN_FAILURE;
  }

  // Check for suspicious message sources. A suspicious exception message comes
  // from a source other than the kernel or the process that the exception
  // purportedly occurred in.
  //
  // TODO(mark): Consider exceptions outside of the range (0, 32) from the
  // kernel to be suspicious, and exceptions other than kMachExceptionSimulated
  // from the process itself to be suspicious.
  const pid_t pid = process_snapshot.ProcessID();
  pid_t audit_pid = AuditPIDFromMachMessageTrailer(trailer);
  if (audit_pid != -1 && audit_pid != 0) {
    if (audit_pid != pid) {
      LOG(WARNING) << "exception for pid " << pid << " sent by pid "
                   << audit_pid;
    }
  }

  CrashpadInfoClientOptions client_options;
  process_snapshot.GetCrashpadOptions(&client_options);

  if (client_options.crashpad_handler_behavior != TriState::kDisabled &&
      !IsExceptionNonfatalResource(exception, code[0], pid)) {
    // Non-fatal resource exceptions are never user-visible and are not
    // currently of interest to Crashpad.

    if (!process_snapshot.InitializeException(behavior,
                                              thread,
                                              exception,
                                              code,
                                              code_count,
                                              *flavor,
                                              old_state,
                                              old_state_count)) {
      Metrics::ExceptionCaptureResult(
          Metrics::CaptureResult::kExceptionInitializationFailed);
      return KERN_FAILURE;
    }

    UUID client_id;
    Settings* const settings = database_->GetSettings();
    if (settings) {
      // If GetSettings() or GetClientID() fails, something else will log a
      // message and client_id will be left at its default value, all zeroes,
      // which is appropriate.
      settings->GetClientID(&client_id);
    }

    process_snapshot.SetClientID(client_id);
    process_snapshot.SetAnnotationsSimpleMap(*process_annotations_);

    std::unique_ptr<CrashReportDatabase::NewReport> new_report;
    CrashReportDatabase::OperationStatus database_status =
        database_->PrepareNewCrashReport(&new_report);
    if (database_status != CrashReportDatabase::kNoError) {
      Metrics::ExceptionCaptureResult(
          Metrics::CaptureResult::kPrepareNewCrashReportFailed);
      return KERN_FAILURE;
    }

    process_snapshot.SetReportID(new_report->ReportID());

    MinidumpFileWriter minidump;
    minidump.InitializeFromSnapshot(&process_snapshot);
    AddUserExtensionStreams(
        user_stream_data_sources_, &process_snapshot, &minidump);

    if (!minidump.WriteEverything(new_report->Writer())) {
      Metrics::ExceptionCaptureResult(
          Metrics::CaptureResult::kMinidumpWriteFailed);
      return KERN_FAILURE;
    }

    UUID uuid;
    database_status =
        database_->FinishedWritingCrashReport(std::move(new_report), &uuid);
    if (database_status != CrashReportDatabase::kNoError) {
      Metrics::ExceptionCaptureResult(
          Metrics::CaptureResult::kFinishedWritingCrashReportFailed);
      return KERN_FAILURE;
    }

    if (upload_thread_) {
      upload_thread_->ReportPending(uuid);
    }
  }

  if (client_options.system_crash_reporter_forwarding != TriState::kDisabled &&
      (exception == EXC_CRASH ||
       exception == EXC_RESOURCE ||
       exception == EXC_GUARD)) {
    // Don’t forward simulated exceptions such as kMachExceptionSimulated to the
    // system crash reporter. Only forward the types of exceptions that it would
    // receive under normal conditions. Although the system crash reporter is
    // able to deal with other exceptions including simulated ones, forwarding
    // them to the system crash reporter could present the system’s crash UI for
    // processes that haven’t actually crashed, and could result in reports not
    // actually associated with crashes being sent to the operating system
    // vendor.
    base::mac::ScopedMachSendRight
        system_crash_reporter_handler(SystemCrashReporterHandler());
    if (system_crash_reporter_handler.get()) {
      // Make copies of mutable out parameters so that the system crash reporter
      // can’t influence the state returned by this method.
      thread_state_flavor_t flavor_forward = *flavor;
      mach_msg_type_number_t new_state_forward_count = *new_state_count;
      std::vector<natural_t> new_state_forward(
          new_state, new_state + new_state_forward_count);

      // The system crash reporter requires the behavior to be
      // EXCEPTION_STATE_IDENTITY | MACH_EXCEPTION_CODES. It uses the identity
      // parameters but doesn’t appear to use the state parameters, including
      // |flavor|, and doesn’t care if they are 0 or invalid. As long as an
      // identity is available (checked above), any other exception behavior is
      // converted to what the system crash reporter wants, with the caveat that
      // problems may arise if the state wasn’t available and the system crash
      // reporter changes in the future to use it. However, normally, the state
      // will be available.
      kern_return_t kr = UniversalExceptionRaise(
          EXCEPTION_STATE_IDENTITY | MACH_EXCEPTION_CODES,
          system_crash_reporter_handler.get(),
          thread,
          task,
          exception,
          code,
          code_count,
          &flavor_forward,
          old_state,
          old_state_count,
          new_state_forward_count ? &new_state_forward[0] : nullptr,
          &new_state_forward_count);
      MACH_LOG_IF(WARNING, kr != KERN_SUCCESS, kr) << "UniversalExceptionRaise";
    }
  }

  ExcServerCopyState(
      behavior, old_state, old_state_count, new_state, new_state_count);

  Metrics::ExceptionCaptureResult(Metrics::CaptureResult::kSuccess);
  return ExcServerSuccessfulReturnValue(exception, behavior, false);
}

}  // namespace crashpad
