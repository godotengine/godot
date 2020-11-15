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

#include "handler/linux/crash_report_exception_handler.h"

#include <vector>

#include "base/logging.h"
#include "client/settings.h"
#include "minidump/minidump_file_writer.h"
#include "snapshot/crashpad_info_client_options.h"
#include "snapshot/linux/process_snapshot_linux.h"
#include "snapshot/sanitized/process_snapshot_sanitized.h"
#include "snapshot/sanitized/sanitization_information.h"
#include "util/linux/direct_ptrace_connection.h"
#include "util/linux/ptrace_client.h"
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

CrashReportExceptionHandler::~CrashReportExceptionHandler() = default;

bool CrashReportExceptionHandler::HandleException(
    pid_t client_process_id,
    const ClientInformation& info) {
  Metrics::ExceptionEncountered();

  DirectPtraceConnection connection;
  if (!connection.Initialize(client_process_id)) {
    Metrics::ExceptionCaptureResult(
        Metrics::CaptureResult::kDirectPtraceFailed);
    return false;
  }

  return HandleExceptionWithConnection(&connection, info);
}

bool CrashReportExceptionHandler::HandleExceptionWithBroker(
    pid_t client_process_id,
    const ClientInformation& info,
    int broker_sock) {
  Metrics::ExceptionEncountered();

  PtraceClient client;
  if (!client.Initialize(broker_sock, client_process_id)) {
    Metrics::ExceptionCaptureResult(
        Metrics::CaptureResult::kBrokeredPtraceFailed);
    return false;
  }

  return HandleExceptionWithConnection(&client, info);
}

bool CrashReportExceptionHandler::HandleExceptionWithConnection(
    PtraceConnection* connection,
    const ClientInformation& info) {
  ProcessSnapshotLinux process_snapshot;
  if (!process_snapshot.Initialize(connection)) {
    Metrics::ExceptionCaptureResult(Metrics::CaptureResult::kSnapshotFailed);
    return false;
  }

  if (!process_snapshot.InitializeException(
          info.exception_information_address)) {
    Metrics::ExceptionCaptureResult(
        Metrics::CaptureResult::kExceptionInitializationFailed);
    return false;
  }

  Metrics::ExceptionCode(process_snapshot.Exception()->Exception());

  CrashpadInfoClientOptions client_options;
  process_snapshot.GetCrashpadOptions(&client_options);
  if (client_options.crashpad_handler_behavior != TriState::kDisabled) {
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
      LOG(ERROR) << "PrepareNewCrashReport failed";
      Metrics::ExceptionCaptureResult(
          Metrics::CaptureResult::kPrepareNewCrashReportFailed);
      return false;
    }

    process_snapshot.SetReportID(new_report->ReportID());

    ProcessSnapshot* snapshot = nullptr;
    ProcessSnapshotSanitized sanitized;
    std::vector<std::string> whitelist;
    if (info.sanitization_information_address) {
      SanitizationInformation sanitization_info;
      ProcessMemoryRange range;
      if (!range.Initialize(connection->Memory(), connection->Is64Bit()) ||
          !range.Read(info.sanitization_information_address,
                      sizeof(sanitization_info),
                      &sanitization_info)) {
        Metrics::ExceptionCaptureResult(
            Metrics::CaptureResult::kSanitizationInitializationFailed);
        return false;
      }

      if (sanitization_info.annotations_whitelist_address &&
          !ReadAnnotationsWhitelist(
              range,
              sanitization_info.annotations_whitelist_address,
              &whitelist)) {
        Metrics::ExceptionCaptureResult(
            Metrics::CaptureResult::kSanitizationInitializationFailed);
        return false;
      }

      if (!sanitized.Initialize(&process_snapshot,
                                sanitization_info.annotations_whitelist_address
                                    ? &whitelist
                                    : nullptr,
                                sanitization_info.target_module_address,
                                sanitization_info.sanitize_stacks)) {
        Metrics::ExceptionCaptureResult(
            Metrics::CaptureResult::kSkippedDueToSanitization);
        return true;
      }

      snapshot = &sanitized;
    } else {
      snapshot = &process_snapshot;
    }

    MinidumpFileWriter minidump;
    minidump.InitializeFromSnapshot(snapshot);
    AddUserExtensionStreams(user_stream_data_sources_, snapshot, &minidump);

    if (!minidump.WriteEverything(new_report->Writer())) {
      LOG(ERROR) << "WriteEverything failed";
      Metrics::ExceptionCaptureResult(
          Metrics::CaptureResult::kMinidumpWriteFailed);
      return false;
    }

    UUID uuid;
    database_status =
        database_->FinishedWritingCrashReport(std::move(new_report), &uuid);
    if (database_status != CrashReportDatabase::kNoError) {
      LOG(ERROR) << "FinishedWritingCrashReport failed";
      Metrics::ExceptionCaptureResult(
          Metrics::CaptureResult::kFinishedWritingCrashReportFailed);
      return false;
    }

    if (upload_thread_) {
      upload_thread_->ReportPending(uuid);
    }
  }

  Metrics::ExceptionCaptureResult(Metrics::CaptureResult::kSuccess);
  return true;
}

}  // namespace crashpad
