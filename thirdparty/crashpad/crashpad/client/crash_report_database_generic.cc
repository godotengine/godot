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

#include "client/crash_report_database.h"

#include <stdint.h>
#include <sys/types.h>

#include <utility>

#include "base/logging.h"
#include "build/build_config.h"
#include "client/settings.h"
#include "util/file/directory_reader.h"
#include "util/file/filesystem.h"
#include "util/misc/initialization_state_dcheck.h"

namespace crashpad {

namespace {

base::FilePath ReplaceFinalExtension(
    const base::FilePath& path,
    const base::FilePath::StringType extension) {
  return base::FilePath(path.RemoveFinalExtension().value() + extension);
}

UUID UUIDFromReportPath(const base::FilePath& path) {
  UUID uuid;
  uuid.InitializeFromString(path.RemoveFinalExtension().BaseName().value());
  return uuid;
}

bool AttachmentNameIsOK(const std::string& name) {
  for (const char c : name) {
    if (c != '_' && c != '-' && c != '.' && !isalnum(c))
      return false;
  }
  return true;
}

using OperationStatus = CrashReportDatabase::OperationStatus;

constexpr base::FilePath::CharType kSettings[] =
    FILE_PATH_LITERAL("settings.dat");

constexpr base::FilePath::CharType kCrashReportExtension[] =
    FILE_PATH_LITERAL(".dmp");
constexpr base::FilePath::CharType kMetadataExtension[] =
    FILE_PATH_LITERAL(".meta");
constexpr base::FilePath::CharType kLockExtension[] =
    FILE_PATH_LITERAL(".lock");

constexpr base::FilePath::CharType kNewDirectory[] = FILE_PATH_LITERAL("new");
constexpr base::FilePath::CharType kPendingDirectory[] =
    FILE_PATH_LITERAL("pending");
constexpr base::FilePath::CharType kCompletedDirectory[] =
    FILE_PATH_LITERAL("completed");
constexpr base::FilePath::CharType kAttachmentsDirectory[] =
    FILE_PATH_LITERAL("attachments");

constexpr const base::FilePath::CharType* kReportDirectories[] = {
    kNewDirectory,
    kPendingDirectory,
    kCompletedDirectory,
};

enum {
  //! \brief Corresponds to uploaded bit of the report state.
  kAttributeUploaded = 1 << 0,

  //! \brief Corresponds to upload_explicity_requested bit of the report state.
  kAttributeUploadExplicitlyRequested = 1 << 1,
};

struct ReportMetadata {
  static constexpr int32_t kVersion = 1;

  int32_t version = kVersion;
  int32_t upload_attempts = 0;
  int64_t last_upload_attempt_time = 0;
  time_t creation_time = 0;
  uint8_t attributes = 0;
};

// A lock held while using database resources.
class ScopedLockFile {
 public:
  ScopedLockFile() = default;
  ~ScopedLockFile() = default;

  ScopedLockFile& operator=(ScopedLockFile&& other) {
    lock_file_.reset(other.lock_file_.release());
    return *this;
  }

  // Attempt to acquire a lock for the report at report_path.
  // Return `true` on success, otherwise `false`.
  bool ResetAcquire(const base::FilePath& report_path) {
    lock_file_.reset();

    base::FilePath lock_path(report_path.RemoveFinalExtension().value() +
                             kLockExtension);
    ScopedFileHandle lock_fd(LoggingOpenFileForWrite(
        lock_path, FileWriteMode::kCreateOrFail, FilePermissions::kOwnerOnly));
    if (!lock_fd.is_valid()) {
      return false;
    }
    lock_file_.reset(lock_path);

    time_t timestamp = time(nullptr);
    if (!LoggingWriteFile(lock_fd.get(), &timestamp, sizeof(timestamp))) {
      return false;
    }

    return true;
  }

  // Returns `true` if the lock is held.
  bool is_valid() const { return lock_file_.is_valid(); }

  // Returns `true` if the lockfile at lock_path has expired.
  static bool IsExpired(const base::FilePath& lock_path, time_t lockfile_ttl) {
    time_t now = time(nullptr);

    timespec filetime;
    if (FileModificationTime(lock_path, &filetime) &&
        filetime.tv_sec > now + lockfile_ttl) {
      return false;
    }

    ScopedFileHandle lock_fd(LoggingOpenFileForReadAndWrite(
        lock_path, FileWriteMode::kReuseOrFail, FilePermissions::kOwnerOnly));
    if (!lock_fd.is_valid()) {
      return false;
    }

    time_t timestamp;
    if (!LoggingReadFileExactly(lock_fd.get(), &timestamp, sizeof(timestamp))) {
      return false;
    }

    return now >= timestamp + lockfile_ttl;
  }

 private:
  ScopedRemoveFile lock_file_;

  DISALLOW_COPY_AND_ASSIGN(ScopedLockFile);
};

}  // namespace

class CrashReportDatabaseGeneric : public CrashReportDatabase {
 public:
  CrashReportDatabaseGeneric();
  ~CrashReportDatabaseGeneric() override;

  bool Initialize(const base::FilePath& path, bool may_create);

  // CrashReportDatabase:
  Settings* GetSettings() override;
  OperationStatus PrepareNewCrashReport(
      std::unique_ptr<NewReport>* report) override;
  OperationStatus FinishedWritingCrashReport(std::unique_ptr<NewReport> report,
                                             UUID* uuid) override;
  OperationStatus LookUpCrashReport(const UUID& uuid, Report* report) override;
  OperationStatus GetPendingReports(std::vector<Report>* reports) override;
  OperationStatus GetCompletedReports(std::vector<Report>* reports) override;
  OperationStatus GetReportForUploading(
      const UUID& uuid,
      std::unique_ptr<const UploadReport>* report,
      bool report_metrics) override;
  OperationStatus SkipReportUpload(const UUID& uuid,
                                   Metrics::CrashSkippedReason reason) override;
  OperationStatus DeleteReport(const UUID& uuid) override;
  OperationStatus RequestUpload(const UUID& uuid) override;
  int CleanDatabase(time_t lockfile_ttl) override;

  // Build a filepath for the directory for the report to hold attachments.
  base::FilePath AttachmentsPath(const UUID& uuid);

 private:
  struct LockfileUploadReport : public UploadReport {
    ScopedLockFile lock_file;
  };

  enum ReportState : int32_t {
    kUninitialized = -1,

    // Being created by a caller of PrepareNewCrashReport().
    kNew,

    // Created by FinishedWritingCrashReport(), but not yet uploaded.
    kPending,

    // Upload completed or skipped.
    kCompleted,

    // Specifies either kPending or kCompleted.
    kSearchable,
  };

  // CrashReportDatabase:
  OperationStatus RecordUploadAttempt(UploadReport* report,
                                      bool successful,
                                      const std::string& id) override;

  // Builds a filepath for the report with the specified uuid and state.
  base::FilePath ReportPath(const UUID& uuid, ReportState state);

  // Locates the report with id uuid and returns its file path in path and a
  // lock for the report in lock_file. This method succeeds as long as the
  // report file exists and the lock can be acquired. No validation is done on
  // the existence or content of the metadata file.
  OperationStatus LocateAndLockReport(const UUID& uuid,
                                      ReportState state,
                                      base::FilePath* path,
                                      ScopedLockFile* lock_file);

  // Locates, locks, and reads the metadata for the report with the specified
  // uuid and state. This method will fail and may remove reports if invalid
  // metadata is detected. state may be kPending, kCompleted, or kSearchable.
  OperationStatus CheckoutReport(const UUID& uuid,
                                 ReportState state,
                                 base::FilePath* path,
                                 ScopedLockFile* lock_file,
                                 Report* report);

  // Reads metadata for all reports in state and returns it in reports.
  OperationStatus ReportsInState(ReportState state,
                                 std::vector<Report>* reports);

  // Cleans lone metadata, reports, or expired locks in a particular state.
  int CleanReportsInState(ReportState state, time_t lockfile_ttl);

  // Cleans any attachments that have no associated report in any state.
  void CleanOrphanedAttachments();

  // Attempt to remove any attachments associated with the given report UUID.
  // There may not be any, so failing is not an error.
  void RemoveAttachmentsByUUID(const UUID& uuid);

  // Reads the metadata for a report from path and returns it in report.
  static bool ReadMetadata(const base::FilePath& path, Report* report);

  // Wraps ReadMetadata and removes the report from the database on failure.
  bool CleaningReadMetadata(const base::FilePath& path, Report* report);

  // Writes metadata for a new report to the filesystem at path.
  static bool WriteNewMetadata(const base::FilePath& path);

  // Writes the metadata for report to the filesystem at path.
  static bool WriteMetadata(const base::FilePath& path, const Report& report);

  base::FilePath base_dir_;
  Settings settings_;
  InitializationStateDcheck initialized_;

  DISALLOW_COPY_AND_ASSIGN(CrashReportDatabaseGeneric);
};

FileWriter* CrashReportDatabase::NewReport::AddAttachment(
    const std::string& name) {
  if (!AttachmentNameIsOK(name)) {
    LOG(ERROR) << "invalid name for attachment " << name;
    return nullptr;
  }

  base::FilePath attachments_dir =
      static_cast<CrashReportDatabaseGeneric*>(database_)->AttachmentsPath(
          uuid_);
  if (!LoggingCreateDirectory(
          attachments_dir, FilePermissions::kOwnerOnly, true)) {
    return nullptr;
  }

  base::FilePath path = attachments_dir.Append(name);

  auto writer = std::make_unique<FileWriter>();
  if (!writer->Open(
          path, FileWriteMode::kCreateOrFail, FilePermissions::kOwnerOnly)) {
    LOG(ERROR) << "could not open " << path.value();
    return nullptr;
  }
  attachment_writers_.emplace_back(std::move(writer));
  attachment_removers_.emplace_back(ScopedRemoveFile(path));
  return attachment_writers_.back().get();
}

void CrashReportDatabase::UploadReport::InitializeAttachments() {
  base::FilePath attachments_dir =
      static_cast<CrashReportDatabaseGeneric*>(database_)->AttachmentsPath(
          uuid);
  DirectoryReader reader;
  if (!reader.Open(attachments_dir)) {
    return;
  }

  base::FilePath filename;
  DirectoryReader::Result dir_result;
  while ((dir_result = reader.NextFile(&filename)) ==
         DirectoryReader::Result::kSuccess) {
    const base::FilePath filepath(attachments_dir.Append(filename));
    std::unique_ptr<FileReader> reader(std::make_unique<FileReader>());
    if (!reader->Open(filepath)) {
      LOG(ERROR) << "attachment " << filepath.value()
                 << " couldn't be opened, skipping";
      continue;
    }
    attachment_readers_.emplace_back(std::move(reader));
    attachment_map_[filename.value()] = attachment_readers_.back().get();
  }
}

CrashReportDatabaseGeneric::CrashReportDatabaseGeneric() = default;

CrashReportDatabaseGeneric::~CrashReportDatabaseGeneric() = default;

bool CrashReportDatabaseGeneric::Initialize(const base::FilePath& path,
                                            bool may_create) {
  INITIALIZATION_STATE_SET_INITIALIZING(initialized_);
  base_dir_ = path;

  if (!IsDirectory(base_dir_, true) &&
      !(may_create &&
        LoggingCreateDirectory(base_dir_, FilePermissions::kOwnerOnly, true))) {
    return false;
  }

  for (const base::FilePath::CharType* subdir : kReportDirectories) {
    if (!LoggingCreateDirectory(
            base_dir_.Append(subdir), FilePermissions::kOwnerOnly, true)) {
      return false;
    }
  }

  if (!LoggingCreateDirectory(base_dir_.Append(kAttachmentsDirectory),
                              FilePermissions::kOwnerOnly,
                              true)) {
    return false;
  }

  if (!settings_.Initialize(base_dir_.Append(kSettings))) {
    return false;
  }

  INITIALIZATION_STATE_SET_VALID(initialized_);
  return true;
}

// static
std::unique_ptr<CrashReportDatabase> CrashReportDatabase::Initialize(
    const base::FilePath& path) {
  auto database = std::make_unique<CrashReportDatabaseGeneric>();
  return database->Initialize(path, true) ? std::move(database) : nullptr;
}

// static
std::unique_ptr<CrashReportDatabase>
CrashReportDatabase::InitializeWithoutCreating(const base::FilePath& path) {
  auto database = std::make_unique<CrashReportDatabaseGeneric>();
  return database->Initialize(path, false) ? std::move(database) : nullptr;
}

Settings* CrashReportDatabaseGeneric::GetSettings() {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return &settings_;
}

OperationStatus CrashReportDatabaseGeneric::PrepareNewCrashReport(
    std::unique_ptr<NewReport>* report) {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  auto new_report = std::make_unique<NewReport>();
  if (!new_report->Initialize(
          this, base_dir_.Append(kNewDirectory), kCrashReportExtension)) {
    return kFileSystemError;
  }

  report->reset(new_report.release());
  return kNoError;
}

OperationStatus CrashReportDatabaseGeneric::FinishedWritingCrashReport(
    std::unique_ptr<NewReport> report,
    UUID* uuid) {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  base::FilePath path = ReportPath(report->ReportID(), kPending);
  ScopedLockFile lock_file;
  if (!lock_file.ResetAcquire(path)) {
    return kBusyError;
  }

  if (!WriteNewMetadata(ReplaceFinalExtension(path, kMetadataExtension))) {
    return kDatabaseError;
  }

  FileOffset size = report->Writer()->Seek(0, SEEK_END);

  report->Writer()->Close();
  if (!MoveFileOrDirectory(report->file_remover_.get(), path)) {
    return kFileSystemError;
  }
  // We've moved the report to pending, so it no longer needs to be removed.
  ignore_result(report->file_remover_.release());

  // Close all the attachments and disarm their removers too.
  for (auto& writer : report->attachment_writers_) {
    writer->Close();
  }
  for (auto& remover : report->attachment_removers_) {
    ignore_result(remover.release());
  }

  *uuid = report->ReportID();

  Metrics::CrashReportPending(Metrics::PendingReportReason::kNewlyCreated);
  Metrics::CrashReportSize(size);

  return kNoError;
}

OperationStatus CrashReportDatabaseGeneric::LookUpCrashReport(const UUID& uuid,
                                                              Report* report) {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  ScopedLockFile lock_file;
  base::FilePath path;
  return CheckoutReport(uuid, kSearchable, &path, &lock_file, report);
}

OperationStatus CrashReportDatabaseGeneric::GetPendingReports(
    std::vector<Report>* reports) {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return ReportsInState(kPending, reports);
}

OperationStatus CrashReportDatabaseGeneric::GetCompletedReports(
    std::vector<Report>* reports) {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return ReportsInState(kCompleted, reports);
}

OperationStatus CrashReportDatabaseGeneric::GetReportForUploading(
    const UUID& uuid,
    std::unique_ptr<const UploadReport>* report,
    bool report_metrics) {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  auto upload_report = std::make_unique<LockfileUploadReport>();

  base::FilePath path;
  OperationStatus os = CheckoutReport(
      uuid, kPending, &path, &upload_report->lock_file, upload_report.get());
  if (os != kNoError) {
    return os;
  }

  if (!upload_report->Initialize(path, this)) {
    return kFileSystemError;
  }
  upload_report->report_metrics_ = report_metrics;

  report->reset(upload_report.release());
  return kNoError;
}

OperationStatus CrashReportDatabaseGeneric::SkipReportUpload(
    const UUID& uuid,
    Metrics::CrashSkippedReason reason) {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  Metrics::CrashUploadSkipped(reason);

  base::FilePath path;
  ScopedLockFile lock_file;
  Report report;
  OperationStatus os =
      CheckoutReport(uuid, kPending, &path, &lock_file, &report);
  if (os != kNoError) {
    return os;
  }

  base::FilePath completed_path(ReportPath(uuid, kCompleted));
  ScopedLockFile completed_lock_file;
  if (!completed_lock_file.ResetAcquire(completed_path)) {
    return kBusyError;
  }

  report.upload_explicitly_requested = false;
  if (!WriteMetadata(completed_path, report)) {
    return kDatabaseError;
  }

  if (!MoveFileOrDirectory(path, completed_path)) {
    return kFileSystemError;
  }

  if (!LoggingRemoveFile(ReplaceFinalExtension(path, kMetadataExtension))) {
    return kDatabaseError;
  }

  return kNoError;
}

OperationStatus CrashReportDatabaseGeneric::DeleteReport(const UUID& uuid) {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  base::FilePath path;
  ScopedLockFile lock_file;
  OperationStatus os =
      LocateAndLockReport(uuid, kSearchable, &path, &lock_file);
  if (os != kNoError) {
    return os;
  }

  if (!LoggingRemoveFile(path)) {
    return kFileSystemError;
  }

  if (!LoggingRemoveFile(ReplaceFinalExtension(path, kMetadataExtension))) {
    return kDatabaseError;
  }

  RemoveAttachmentsByUUID(uuid);

  return kNoError;
}

OperationStatus CrashReportDatabaseGeneric::RequestUpload(const UUID& uuid) {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  base::FilePath path;
  ScopedLockFile lock_file;
  Report report;
  OperationStatus os =
      CheckoutReport(uuid, kSearchable, &path, &lock_file, &report);
  if (os != kNoError) {
    return os;
  }

  if (report.uploaded) {
    return kCannotRequestUpload;
  }

  report.upload_explicitly_requested = true;
  base::FilePath pending_path = ReportPath(uuid, kPending);
  if (!MoveFileOrDirectory(path, pending_path)) {
    return kFileSystemError;
  }

  if (!WriteMetadata(pending_path, report)) {
    return kDatabaseError;
  }

  if (pending_path != path) {
    if (!LoggingRemoveFile(ReplaceFinalExtension(path, kMetadataExtension))) {
      return kDatabaseError;
    }
  }

  Metrics::CrashReportPending(Metrics::PendingReportReason::kUserInitiated);
  return kNoError;
}

int CrashReportDatabaseGeneric::CleanDatabase(time_t lockfile_ttl) {
  int removed = 0;
  time_t now = time(nullptr);

  DirectoryReader reader;
  const base::FilePath new_dir(base_dir_.Append(kNewDirectory));
  if (reader.Open(new_dir)) {
    base::FilePath filename;
    DirectoryReader::Result result;
    while ((result = reader.NextFile(&filename)) ==
           DirectoryReader::Result::kSuccess) {
      const base::FilePath filepath(new_dir.Append(filename));
      timespec filetime;
      if (!FileModificationTime(filepath, &filetime)) {
        continue;
      }
      if (filetime.tv_sec <= now - lockfile_ttl) {
        if (LoggingRemoveFile(filepath)) {
          ++removed;
        }
      }
    }
  }

  removed += CleanReportsInState(kPending, lockfile_ttl);
  removed += CleanReportsInState(kCompleted, lockfile_ttl);
  CleanOrphanedAttachments();
  return removed;
}

OperationStatus CrashReportDatabaseGeneric::RecordUploadAttempt(
    UploadReport* report,
    bool successful,
    const std::string& id) {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  if (report->report_metrics_) {
    Metrics::CrashUploadAttempted(successful);
  }
  time_t now = time(nullptr);

  report->id = id;
  report->uploaded = successful;
  report->last_upload_attempt_time = now;
  ++report->upload_attempts;

  base::FilePath report_path(report->file_path);

  ScopedLockFile lock_file;
  if (successful) {
    report->upload_explicitly_requested = false;

    base::FilePath completed_report_path = ReportPath(report->uuid, kCompleted);

    if (!lock_file.ResetAcquire(completed_report_path)) {
      return kBusyError;
    }

    report->Reader()->Close();
    if (!MoveFileOrDirectory(report_path, completed_report_path)) {
      return kFileSystemError;
    }

    LoggingRemoveFile(ReplaceFinalExtension(report_path, kMetadataExtension));
    report_path = completed_report_path;
  }

  if (!WriteMetadata(report_path, *report)) {
    return kDatabaseError;
  }

  if (!settings_.SetLastUploadAttemptTime(now)) {
    return kDatabaseError;
  }

  return kNoError;
}

base::FilePath CrashReportDatabaseGeneric::ReportPath(const UUID& uuid,
                                                      ReportState state) {
  DCHECK_NE(state, kUninitialized);
  DCHECK_NE(state, kSearchable);

#if defined(OS_WIN)
  const std::wstring uuid_string = uuid.ToString16();
#else
  const std::string uuid_string = uuid.ToString();
#endif

  return base_dir_.Append(kReportDirectories[state])
      .Append(uuid_string + kCrashReportExtension);
}

base::FilePath CrashReportDatabaseGeneric::AttachmentsPath(const UUID& uuid) {
#if defined(OS_WIN)
  const std::wstring uuid_string = uuid.ToString16();
#else
  const std::string uuid_string = uuid.ToString();
#endif

  return base_dir_.Append(kAttachmentsDirectory).Append(uuid_string);
}

OperationStatus CrashReportDatabaseGeneric::LocateAndLockReport(
    const UUID& uuid,
    ReportState desired_state,
    base::FilePath* path,
    ScopedLockFile* lock_file) {
  std::vector<ReportState> searchable_states;
  if (desired_state == kSearchable) {
    searchable_states.push_back(kPending);
    searchable_states.push_back(kCompleted);
  } else {
    DCHECK(desired_state == kPending || desired_state == kCompleted);
    searchable_states.push_back(desired_state);
  }

  for (const ReportState state : searchable_states) {
    base::FilePath local_path(ReportPath(uuid, state));
    ScopedLockFile local_lock;
    if (!local_lock.ResetAcquire(local_path)) {
      return kBusyError;
    }

    if (!IsRegularFile(local_path)) {
      continue;
    }

    *path = local_path;
    *lock_file = std::move(local_lock);
    return kNoError;
  }

  return kReportNotFound;
}

OperationStatus CrashReportDatabaseGeneric::CheckoutReport(
    const UUID& uuid,
    ReportState state,
    base::FilePath* path,
    ScopedLockFile* lock_file,
    Report* report) {
  ScopedLockFile local_lock;
  base::FilePath local_path;
  OperationStatus os =
      LocateAndLockReport(uuid, state, &local_path, &local_lock);
  if (os != kNoError) {
    return os;
  }

  if (!CleaningReadMetadata(local_path, report)) {
    return kDatabaseError;
  }

  *path = local_path;
  *lock_file = std::move(local_lock);
  return kNoError;
}

OperationStatus CrashReportDatabaseGeneric::ReportsInState(
    ReportState state,
    std::vector<Report>* reports) {
  DCHECK(reports->empty());
  DCHECK_NE(state, kUninitialized);
  DCHECK_NE(state, kSearchable);
  DCHECK_NE(state, kNew);

  const base::FilePath dir_path(base_dir_.Append(kReportDirectories[state]));
  DirectoryReader reader;
  if (!reader.Open(dir_path)) {
    return kDatabaseError;
  }

  base::FilePath filename;
  DirectoryReader::Result result;
  while ((result = reader.NextFile(&filename)) ==
         DirectoryReader::Result::kSuccess) {
    const base::FilePath::StringType extension(filename.FinalExtension());
    if (extension.compare(kCrashReportExtension) != 0) {
      continue;
    }

    const base::FilePath filepath(dir_path.Append(filename));
    ScopedLockFile lock_file;
    if (!lock_file.ResetAcquire(filepath)) {
      continue;
    }

    Report report;
    if (!CleaningReadMetadata(filepath, &report)) {
      continue;
    }
    reports->push_back(report);
    reports->back().file_path = filepath;
  }
  return kNoError;
}

int CrashReportDatabaseGeneric::CleanReportsInState(ReportState state,
                                                    time_t lockfile_ttl) {
  const base::FilePath dir_path(base_dir_.Append(kReportDirectories[state]));
  DirectoryReader reader;
  if (!reader.Open(dir_path)) {
    return 0;
  }

  int removed = 0;
  base::FilePath filename;
  DirectoryReader::Result result;
  while ((result = reader.NextFile(&filename)) ==
         DirectoryReader::Result::kSuccess) {
    const base::FilePath::StringType extension(filename.FinalExtension());
    const base::FilePath filepath(dir_path.Append(filename));

    // Remove any report files without metadata.
    if (extension.compare(kCrashReportExtension) == 0) {
      const base::FilePath metadata_path(
          ReplaceFinalExtension(filepath, kMetadataExtension));
      ScopedLockFile report_lock;
      if (report_lock.ResetAcquire(filepath) && !IsRegularFile(metadata_path) &&
          LoggingRemoveFile(filepath)) {
        ++removed;
        RemoveAttachmentsByUUID(UUIDFromReportPath(filepath));
      }
      continue;
    }

    // Remove any metadata files without report files.
    if (extension.compare(kMetadataExtension) == 0) {
      const base::FilePath report_path(
          ReplaceFinalExtension(filepath, kCrashReportExtension));
      ScopedLockFile report_lock;
      if (report_lock.ResetAcquire(report_path) &&
          !IsRegularFile(report_path) && LoggingRemoveFile(filepath)) {
        ++removed;
        RemoveAttachmentsByUUID(UUIDFromReportPath(filepath));
      }
      continue;
    }

    // Remove any expired locks only if we can remove the report and metadata.
    if (extension.compare(kLockExtension) == 0 &&
        ScopedLockFile::IsExpired(filepath, lockfile_ttl)) {
      const base::FilePath no_ext(filepath.RemoveFinalExtension());
      const base::FilePath report_path(no_ext.value() + kCrashReportExtension);
      const base::FilePath metadata_path(no_ext.value() + kMetadataExtension);
      if ((IsRegularFile(report_path) && !LoggingRemoveFile(report_path)) ||
          (IsRegularFile(metadata_path) && !LoggingRemoveFile(metadata_path))) {
        continue;
      }

      if (LoggingRemoveFile(filepath)) {
        ++removed;
        RemoveAttachmentsByUUID(UUIDFromReportPath(filepath));
      }
      continue;
    }
  }

  return removed;
}

void CrashReportDatabaseGeneric::CleanOrphanedAttachments() {
  base::FilePath root_attachments_dir(base_dir_.Append(kAttachmentsDirectory));
  DirectoryReader reader;
  if (!reader.Open(root_attachments_dir)) {
    LOG(ERROR) << "no attachments dir";
    return;
  }

  base::FilePath filename;
  DirectoryReader::Result result;
  while ((result = reader.NextFile(&filename)) ==
         DirectoryReader::Result::kSuccess) {
    const base::FilePath path(root_attachments_dir.Append(filename));
    if (IsDirectory(path, false)) {
      UUID uuid;
      if (!uuid.InitializeFromString(filename.value())) {
        LOG(ERROR) << "unexpected attachment dir name " << filename.value();
        continue;
      }

      // Check to see if the report is being created in "new".
      base::FilePath new_dir_path =
          base_dir_.Append(kNewDirectory)
              .Append(uuid.ToString() + kCrashReportExtension);
      if (IsRegularFile(new_dir_path)) {
        continue;
      }

      // Check to see if the report is in "pending" or "completed".
      ScopedLockFile local_lock;
      base::FilePath local_path;
      OperationStatus os =
          LocateAndLockReport(uuid, kSearchable, &local_path, &local_lock);
      if (os != kReportNotFound) {
        continue;
      }

      // Couldn't find a report, assume these attachments are orphaned.
      RemoveAttachmentsByUUID(uuid);
    }
  }
}

void CrashReportDatabaseGeneric::RemoveAttachmentsByUUID(const UUID& uuid) {
  base::FilePath attachments_dir = AttachmentsPath(uuid);
  DirectoryReader reader;
  if (!reader.Open(attachments_dir)) {
    return;
  }

  base::FilePath filename;
  DirectoryReader::Result result;
  while ((result = reader.NextFile(&filename)) ==
         DirectoryReader::Result::kSuccess) {
    const base::FilePath filepath(attachments_dir.Append(filename));
    LoggingRemoveFile(filepath);
  }

  LoggingRemoveDirectory(attachments_dir);
}

// static
bool CrashReportDatabaseGeneric::ReadMetadata(const base::FilePath& path,
                                              Report* report) {
  const base::FilePath metadata_path(
      ReplaceFinalExtension(path, kMetadataExtension));

  ScopedFileHandle handle(LoggingOpenFileForRead(metadata_path));
  if (!handle.is_valid()) {
    return false;
  }

  if (!report->uuid.InitializeFromString(
          path.BaseName().RemoveFinalExtension().value())) {
    LOG(ERROR) << "Couldn't interpret report uuid";
    return false;
  }

  ReportMetadata metadata;
  if (!LoggingReadFileExactly(handle.get(), &metadata, sizeof(metadata))) {
    return false;
  }

  if (metadata.version != ReportMetadata::kVersion) {
    LOG(ERROR) << "metadata version mismatch";
    return false;
  }

  if (!LoggingReadToEOF(handle.get(), &report->id)) {
    return false;
  }

  report->upload_attempts = metadata.upload_attempts;
  report->last_upload_attempt_time = metadata.last_upload_attempt_time;
  report->creation_time = metadata.creation_time;
  report->uploaded = (metadata.attributes & kAttributeUploaded) != 0;
  report->upload_explicitly_requested =
      (metadata.attributes & kAttributeUploadExplicitlyRequested) != 0;
  report->file_path = path;
  return true;
}

bool CrashReportDatabaseGeneric::CleaningReadMetadata(
    const base::FilePath& path,
    Report* report) {
  if (ReadMetadata(path, report)) {
    return true;
  }

  LoggingRemoveFile(path);
  LoggingRemoveFile(ReplaceFinalExtension(path, kMetadataExtension));
  RemoveAttachmentsByUUID(report->uuid);
  return false;
}

// static
bool CrashReportDatabaseGeneric::WriteNewMetadata(const base::FilePath& path) {
  const base::FilePath metadata_path(
      ReplaceFinalExtension(path, kMetadataExtension));

  ScopedFileHandle handle(LoggingOpenFileForWrite(metadata_path,
                                                  FileWriteMode::kCreateOrFail,
                                                  FilePermissions::kOwnerOnly));
  if (!handle.is_valid()) {
    return false;
  }

  ReportMetadata metadata;
  metadata.creation_time = time(nullptr);

  return LoggingWriteFile(handle.get(), &metadata, sizeof(metadata));
}

// static
bool CrashReportDatabaseGeneric::WriteMetadata(const base::FilePath& path,
                                               const Report& report) {
  const base::FilePath metadata_path(
      ReplaceFinalExtension(path, kMetadataExtension));

  ScopedFileHandle handle(
      LoggingOpenFileForWrite(metadata_path,
                              FileWriteMode::kTruncateOrCreate,
                              FilePermissions::kOwnerOnly));
  if (!handle.is_valid()) {
    return false;
  }

  ReportMetadata metadata;
  metadata.creation_time = report.creation_time;
  metadata.last_upload_attempt_time = report.last_upload_attempt_time;
  metadata.upload_attempts = report.upload_attempts;
  metadata.attributes =
      (report.uploaded ? kAttributeUploaded : 0) |
      (report.upload_explicitly_requested ? kAttributeUploadExplicitlyRequested
                                          : 0);

  return LoggingWriteFile(handle.get(), &metadata, sizeof(metadata)) &&
         LoggingWriteFile(handle.get(), report.id.c_str(), report.id.size());
}

}  // namespace crashpad
