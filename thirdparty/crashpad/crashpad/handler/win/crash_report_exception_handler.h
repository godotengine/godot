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

#ifndef CRASHPAD_HANDLER_WIN_CRASH_REPORT_EXCEPTION_HANDLER_H_
#define CRASHPAD_HANDLER_WIN_CRASH_REPORT_EXCEPTION_HANDLER_H_

#include <windows.h>

#include <map>
#include <string>

#include "base/macros.h"
#include "handler/user_stream_data_source.h"
#include "util/win/exception_handler_server.h"

namespace crashpad {

class CrashReportDatabase;
class CrashReportUploadThread;

//! \brief An exception handler that writes crash reports for exception messages
//!     to a CrashReportDatabase.
class CrashReportExceptionHandler : public ExceptionHandlerServer::Delegate {
 public:
  //! \brief Creates a new object that will store crash reports in \a database.
  //!
  //! \param[in] database The database to store crash reports in. Weak.
  //! \param[in] upload_thread The upload thread to notify when a new crash
  //!     report is written into \a database. Report upload is skipped if this
  //!     value is `nullptr`.
  //! \param[in] process_annotations A map of annotations to insert as
  //!     process-level annotations into each crash report that is written. Do
  //!     not confuse this with module-level annotations, which are under the
  //!     control of the crashing process, and are used to implement Chrome's
  //!     "crash keys." Process-level annotations are those that are beyond the
  //!     control of the crashing process, which must reliably be set even if
  //!     the process crashes before it's able to establish its own annotations.
  //!     To interoperate with Breakpad servers, the recommended practice is to
  //!     specify values for the `"prod"` and `"ver"` keys as process
  //!     annotations.
  //! \param[in] user_stream_data_sources Data sources to be used to extend
  //!     crash reports. For each crash report that is written, the data sources
  //!     are called in turn. These data sources may contribute additional
  //!     minidump streams. `nullptr` if not required.
  CrashReportExceptionHandler(
      CrashReportDatabase* database,
      CrashReportUploadThread* upload_thread,
      const std::map<std::string, std::string>* process_annotations,
      const UserStreamDataSources* user_stream_data_sources);

  ~CrashReportExceptionHandler();

  // ExceptionHandlerServer::Delegate:

  //! \brief Processes an exception message by writing a crash report to this
  //!     object's CrashReportDatabase.
  void ExceptionHandlerServerStarted() override;
  unsigned int ExceptionHandlerServerException(
      HANDLE process,
      WinVMAddress exception_information_address,
      WinVMAddress debug_critical_section_address) override;

 private:
  CrashReportDatabase* database_;  // weak
  CrashReportUploadThread* upload_thread_;  // weak
  const std::map<std::string, std::string>* process_annotations_;  // weak
  const UserStreamDataSources* user_stream_data_sources_;  // weak

  DISALLOW_COPY_AND_ASSIGN(CrashReportExceptionHandler);
};

}  // namespace crashpad

#endif  // CRASHPAD_HANDLER_WIN_CRASH_REPORT_EXCEPTION_HANDLER_H_
