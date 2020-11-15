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

#include "client/crash_report_database.h"

#include "base/logging.h"
#include "build/build_config.h"

namespace crashpad {

CrashReportDatabase::Report::Report()
    : uuid(),
      file_path(),
      id(),
      creation_time(0),
      uploaded(false),
      last_upload_attempt_time(0),
      upload_attempts(0),
      upload_explicitly_requested(false) {}

CrashReportDatabase::NewReport::NewReport()
    : writer_(std::make_unique<FileWriter>()),
      file_remover_(),
      attachment_writers_(),
      attachment_removers_(),
      uuid_(),
      database_() {}

CrashReportDatabase::NewReport::~NewReport() = default;

bool CrashReportDatabase::NewReport::Initialize(
    CrashReportDatabase* database,
    const base::FilePath& directory,
    const base::FilePath::StringType& extension) {
  database_ = database;

  if (!uuid_.InitializeWithNew()) {
    return false;
  }

#if defined(OS_WIN)
  const std::wstring uuid_string = uuid_.ToString16();
#else
  const std::string uuid_string = uuid_.ToString();
#endif

  const base::FilePath path = directory.Append(uuid_string + extension);
  if (!writer_->Open(
          path, FileWriteMode::kCreateOrFail, FilePermissions::kOwnerOnly)) {
    return false;
  }
  file_remover_.reset(path);
  return true;
}

CrashReportDatabase::UploadReport::UploadReport()
    : Report(),
      reader_(std::make_unique<FileReader>()),
      database_(nullptr),
      attachment_readers_(),
      attachment_map_(),
      report_metrics_(false) {}

CrashReportDatabase::UploadReport::~UploadReport() {
  if (database_) {
    database_->RecordUploadAttempt(this, false, std::string());
  }
}

bool CrashReportDatabase::UploadReport::Initialize(const base::FilePath path,
                                                   CrashReportDatabase* db) {
  database_ = db;
  InitializeAttachments();
  return reader_->Open(path);
}

CrashReportDatabase::OperationStatus CrashReportDatabase::RecordUploadComplete(
    std::unique_ptr<const UploadReport> report_in,
    const std::string& id) {
  UploadReport* report = const_cast<UploadReport*>(report_in.get());

  report->database_ = nullptr;
  return RecordUploadAttempt(report, true, id);
}

}  // namespace crashpad
