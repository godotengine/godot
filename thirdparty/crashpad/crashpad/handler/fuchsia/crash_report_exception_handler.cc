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

#include "handler/fuchsia/crash_report_exception_handler.h"

#include <lib/zx/thread.h>
#include <zircon/syscalls/exception.h>

#include "base/fuchsia/fuchsia_logging.h"
#include "client/settings.h"
#include "minidump/minidump_file_writer.h"
#include "minidump/minidump_user_extension_stream_data_source.h"
#include "snapshot/fuchsia/process_snapshot_fuchsia.h"
#include "util/fuchsia/koid_utilities.h"

namespace crashpad {

namespace {

class ScopedThreadResumeAfterException {
 public:
  ScopedThreadResumeAfterException(const zx::thread& thread)
      : thread_(thread) {}
  ~ScopedThreadResumeAfterException() {
    DCHECK(thread_->is_valid());
    // Resuming with ZX_RESUME_TRY_NEXT chains to the next handler. In normal
    // operation, there won't be another beyond this one, which will result in
    // the kernel terminating the process.
    zx_status_t status =
        thread_->resume(ZX_RESUME_EXCEPTION | ZX_RESUME_TRY_NEXT);
    ZX_LOG_IF(ERROR, status != ZX_OK, status) << "zx_task_resume";
  }

 private:
  zx::unowned_thread thread_;
  DISALLOW_COPY_AND_ASSIGN(ScopedThreadResumeAfterException);
};

}  // namespace

CrashReportExceptionHandler::CrashReportExceptionHandler(
    CrashReportDatabase* database,
    CrashReportUploadThread* upload_thread,
    const std::map<std::string, std::string>* process_annotations,
    const std::map<std::string, base::FilePath>* process_attachments,
    const UserStreamDataSources* user_stream_data_sources)
    : database_(database),
      upload_thread_(upload_thread),
      process_annotations_(process_annotations),
      process_attachments_(process_attachments),
      user_stream_data_sources_(user_stream_data_sources) {}

CrashReportExceptionHandler::~CrashReportExceptionHandler() {}

bool CrashReportExceptionHandler::HandleException(uint64_t process_id,
                                                  uint64_t thread_id) {
  // TODO(scottmg): This function needs to be instrumented with metrics calls,
  // https://crashpad.chromium.org/bug/230.

  zx::process process(GetProcessFromKoid(process_id));
  if (!process.is_valid()) {
    // There's no way to zx_task_resume() the thread if the process retrieval
    // fails. Assume that the process has been already killed, and bail.
    return false;
  }

  zx::thread thread(GetThreadHandleByKoid(process, thread_id));
  if (!thread.is_valid()) {
    return false;
  }

  return HandleExceptionHandles(process, thread);
}

bool CrashReportExceptionHandler::HandleExceptionHandles(
    const zx::process& process,
    const zx::thread& thread) {
  // Now that the thread has been successfully retrieved, it is possible to
  // correctly call zx_task_resume() to continue exception processing, even if
  // something else during this function fails.
  ScopedThreadResumeAfterException resume(thread);

  ProcessSnapshotFuchsia process_snapshot;
  if (!process_snapshot.Initialize(process)) {
    return false;
  }

  CrashpadInfoClientOptions client_options;
  process_snapshot.GetCrashpadOptions(&client_options);

  if (client_options.crashpad_handler_behavior != TriState::kDisabled) {
    zx_exception_report_t report;
    zx_status_t status = thread.get_info(ZX_INFO_THREAD_EXCEPTION_REPORT,
                                         &report,
                                         sizeof(report),
                                         nullptr,
                                         nullptr);
    if (status != ZX_OK) {
      ZX_LOG(ERROR, status)
          << "zx_object_get_info ZX_INFO_THREAD_EXCEPTION_REPORT";
      return false;
    }

    zx_koid_t thread_id = GetKoidForHandle(thread);
    if (!process_snapshot.InitializeException(thread_id, report)) {
      return false;
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
      return false;
    }

    process_snapshot.SetReportID(new_report->ReportID());

    MinidumpFileWriter minidump;
    minidump.InitializeFromSnapshot(&process_snapshot);
    AddUserExtensionStreams(
        user_stream_data_sources_, &process_snapshot, &minidump);

    if (!minidump.WriteEverything(new_report->Writer())) {
      return false;
    }

    if (process_attachments_) {
      // Note that attachments are read at this point each time rather than once
      // so that if the contents of the file has changed it will be re-read for
      // each upload (e.g. in the case of a log file).
      for (const auto& it : *process_attachments_) {
        FileWriter* writer = new_report->AddAttachment(it.first);
        if (writer) {
          std::string contents;
          if (!LoggingReadEntireFile(it.second, &contents)) {
            // Not being able to read the file isn't considered fatal, and
            // should not prevent the report from being processed.
            continue;
          }
          writer->Write(contents.data(), contents.size());
        }
      }
    }

    UUID uuid;
    database_status =
        database_->FinishedWritingCrashReport(std::move(new_report), &uuid);
    if (database_status != CrashReportDatabase::kNoError) {
      return false;
    }

    if (upload_thread_) {
      upload_thread_->ReportPending(uuid);
    }
  }

  return true;
}

}  // namespace crashpad
