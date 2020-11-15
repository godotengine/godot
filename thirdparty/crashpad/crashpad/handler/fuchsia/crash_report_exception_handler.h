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

#ifndef CRASHPAD_HANDLER_FUCHSIA_CRASH_REPORT_EXCEPTION_HANDLER_H_
#define CRASHPAD_HANDLER_FUCHSIA_CRASH_REPORT_EXCEPTION_HANDLER_H_

#include <lib/zx/process.h>
#include <lib/zx/thread.h>
#include <stdint.h>
#include <zircon/types.h>

#include <map>
#include <string>

#include "base/files/file_path.h"
#include "base/macros.h"
#include "client/crash_report_database.h"
#include "handler/crash_report_upload_thread.h"
#include "handler/user_stream_data_source.h"

namespace crashpad {

//! \brief An exception handler that writes crash reports for exception messages
//!     to a CrashReportDatabase.
class CrashReportExceptionHandler {
 public:
  //! \brief Creates a new object that will store crash reports in \a database.
  //!
  //! \param[in] database The database to store crash reports in. Weak.
  //! \param[in] upload_thread The upload thread to notify when a new crash
  //!     report is written into \a database.
  //! \param[in] process_annotations A map of annotations to insert as
  //!     process-level annotations into each crash report that is written. Do
  //!     not confuse this with module-level annotations, which are under the
  //!     control of the crashing process, and are used to implement Chrome's
  //!     "crash keys." Process-level annotations are those that are beyond the
  //!     control of the crashing process, which must reliably be set even if
  //!     the process crashes before it’s able to establish its own annotations.
  //!     To interoperate with Breakpad servers, the recommended practice is to
  //!     specify values for the `"prod"` and `"ver"` keys as process
  //!     annotations.
  //! \param[in] process_attachments A map of file name keys to file paths to be
  //!     included in the report. Each time a report is written, the file paths
  //!     will be read in their entirety and included in the report using the
  //!     file name key as the name in the http upload.
  //! \param[in] user_stream_data_sources Data sources to be used to extend
  //!     crash reports. For each crash report that is written, the data sources
  //!     are called in turn. These data sources may contribute additional
  //!     minidump streams. `nullptr` if not required.
  CrashReportExceptionHandler(
      CrashReportDatabase* database,
      CrashReportUploadThread* upload_thread,
      const std::map<std::string, std::string>* process_annotations,
      const std::map<std::string, base::FilePath>* process_attachments,
      const UserStreamDataSources* user_stream_data_sources);

  ~CrashReportExceptionHandler();

  //! \brief Called when the exception handler server has caught an exception
  //!     and wants a crash dump to be taken.
  //!
  //! This function is expected to call `zx_task_resume()` in order to complete
  //! handling of the exception.
  //!
  //! \param[in] process_id The koid of the process which sustained the
  //!     exception.
  //! \param[in] thread_id The koid of the thread which sustained the exception.
  //! \return `true` on success, or `false` with an error logged.
  bool HandleException(uint64_t process_id, uint64_t thread_id);

  //! \brief Called when the exception handler server has caught an exception
  //!     and wants a crash dump to be taken.
  //!
  //! This function is expected to call `zx_task_resume()` in order to complete
  //! handling of the exception.
  //!
  //! \param[in] process The handle to the process which sustained the
  //!     exception.
  //! \param[in] thread The handle to the thread of \a process which sustained
  //!     the exception.
  //! \return `true` on success, or `false` with an error logged.
  bool HandleExceptionHandles(const zx::process& process,
                              const zx::thread& thread);

 private:
  CrashReportDatabase* database_;  // weak
  CrashReportUploadThread* upload_thread_;  // weak
  const std::map<std::string, std::string>* process_annotations_;  // weak
  const std::map<std::string, base::FilePath>* process_attachments_;  // weak
  const UserStreamDataSources* user_stream_data_sources_;  // weak

  DISALLOW_COPY_AND_ASSIGN(CrashReportExceptionHandler);
};

}  // namespace crashpad

#endif  // CRASHPAD_HANDLER_FUCHSIA_CRASH_REPORT_EXCEPTION_HANDLER_H_
