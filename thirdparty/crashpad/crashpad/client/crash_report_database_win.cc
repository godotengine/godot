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

#include <windows.h>
#include <stdint.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>
#include <wchar.h>

#include <utility>

#include "base/logging.h"
#include "base/numerics/safe_math.h"
#include "base/strings/string16.h"
#include "base/strings/stringprintf.h"
#include "base/strings/utf_string_conversions.h"
#include "client/settings.h"
#include "util/misc/implicit_cast.h"
#include "util/misc/initialization_state_dcheck.h"
#include "util/misc/metrics.h"

namespace crashpad {

namespace {

constexpr wchar_t kReportsDirectory[] = L"reports";
constexpr wchar_t kMetadataFileName[] = L"metadata";

constexpr wchar_t kSettings[] = L"settings.dat";

constexpr wchar_t kCrashReportFileExtension[] = L"dmp";

constexpr uint32_t kMetadataFileHeaderMagic = 'CPAD';
constexpr uint32_t kMetadataFileVersion = 1;

using OperationStatus = CrashReportDatabase::OperationStatus;

// Helpers ---------------------------------------------------------------------

// Adds a string to the string table and returns the byte index where it was
// added.
uint32_t AddStringToTable(std::string* string_table, const std::string& str) {
  uint32_t offset = base::checked_cast<uint32_t>(string_table->size());
  *string_table += str;
  *string_table += '\0';
  return offset;
}

// Converts |str| to UTF8, adds the result to the string table and returns the
// byte index where it was added.
uint32_t AddStringToTable(std::string* string_table,
                          const base::string16& str) {
  return AddStringToTable(string_table, base::UTF16ToUTF8(str));
}

// Reads from the current file position to EOF and returns as a string of bytes.
std::string ReadRestOfFileAsString(FileHandle file) {
  FileOffset read_from = LoggingSeekFile(file, 0, SEEK_CUR);
  FileOffset end = LoggingSeekFile(file, 0, SEEK_END);
  FileOffset original = LoggingSeekFile(file, read_from, SEEK_SET);
  if (read_from == -1 || end == -1 || original == -1 || read_from == end)
    return std::string();
  DCHECK_EQ(read_from, original);
  DCHECK_GT(end, read_from);
  size_t data_length = static_cast<size_t>(end - read_from);
  std::string buffer(data_length, '\0');
  return LoggingReadFileExactly(file, &buffer[0], data_length) ? buffer
                                                               : std::string();
}

// Helper structures, and conversions ------------------------------------------

// The format of the on disk metadata file is a MetadataFileHeader, followed by
// a number of fixed size records of MetadataFileReportRecord, followed by a
// string table in UTF8 format, where each string is \0 terminated.
struct MetadataFileHeader {
  uint32_t magic;
  uint32_t version;
  uint32_t num_records;
  uint32_t padding;
};

struct ReportDisk;

enum class ReportState {
  //! \brief Created and filled out by caller, owned by database.
  kPending,
  //! \brief In the process of uploading, owned by caller.
  kUploading,
  //! \brief Upload completed or skipped, owned by database.
  kCompleted,
};

enum {
  //! \brief Corresponds to uploaded bit of the report state.
  kAttributeUploaded = 1 << 0,

  //! \brief Corresponds to upload_explicity_requested bit of the report state.
  kAttributeUploadExplicitlyRequested = 1 << 1,
};

struct MetadataFileReportRecord {
  // Note that this default constructor does no initialization. It is used only
  // to create an array of records that are immediately initialized by reading
  // from disk in Metadata::Read().
  MetadataFileReportRecord() {}

  // Constructs from a ReportDisk, adding to |string_table| and storing indices
  // as strings into that table.
  MetadataFileReportRecord(const ReportDisk& report, std::string* string_table);

  UUID uuid;  // UUID is a 16 byte, standard layout structure.
  uint32_t file_path_index;  // Index into string table. File name is relative
                             // to the reports directory when on disk.
  uint32_t id_index;  // Index into string table.
  int64_t creation_time;  // Holds a time_t.
  int64_t last_upload_attempt_time;  // Holds a time_t.
  int32_t upload_attempts;
  int32_t state;  // A ReportState.
  uint8_t attributes;  // Bitfield of kAttribute*.
  uint8_t padding[7];
};

//! \brief A private extension of the Report class that includes additional data
//!     that's stored on disk in the metadata file.
struct ReportDisk : public CrashReportDatabase::Report {
  ReportDisk(const MetadataFileReportRecord& record,
             const base::FilePath& report_dir,
             const std::string& string_table);

  ReportDisk(const UUID& uuid,
             const base::FilePath& path,
             time_t creation_tim,
             ReportState state);

  //! \brief The current state of the report.
  ReportState state;
};

MetadataFileReportRecord::MetadataFileReportRecord(const ReportDisk& report,
                                                   std::string* string_table)
    : uuid(report.uuid),
      file_path_index(
          AddStringToTable(string_table, report.file_path.BaseName().value())),
      id_index(AddStringToTable(string_table, report.id)),
      creation_time(report.creation_time),
      last_upload_attempt_time(report.last_upload_attempt_time),
      upload_attempts(report.upload_attempts),
      state(static_cast<uint32_t>(report.state)),
      attributes((report.uploaded ? kAttributeUploaded : 0) |
                 (report.upload_explicitly_requested
                      ? kAttributeUploadExplicitlyRequested
                      : 0)) {
  memset(&padding, 0, sizeof(padding));
}

ReportDisk::ReportDisk(const MetadataFileReportRecord& record,
                       const base::FilePath& report_dir,
                       const std::string& string_table)
    : Report() {
  uuid = record.uuid;
  file_path = report_dir.Append(
      base::UTF8ToUTF16(&string_table[record.file_path_index]));
  id = &string_table[record.id_index];
  creation_time = record.creation_time;
  last_upload_attempt_time = record.last_upload_attempt_time;
  upload_attempts = record.upload_attempts;
  state = static_cast<ReportState>(record.state);
  uploaded = (record.attributes & kAttributeUploaded) != 0;
  upload_explicitly_requested =
      (record.attributes & kAttributeUploadExplicitlyRequested) != 0;
}

ReportDisk::ReportDisk(const UUID& uuid,
                       const base::FilePath& path,
                       time_t creation_time,
                       ReportState state)
    : Report() {
  this->uuid = uuid;
  this->file_path = path;
  this->creation_time = creation_time;
  this->state = state;
}

// Metadata --------------------------------------------------------------------

//! \brief Manages the metadata for the set of reports, handling serialization
//!     to disk, and queries.
class Metadata {
 public:
  //! \brief Writes any changes if necessary, unlocks and closes the file
  //!     handle.
  ~Metadata();

  static std::unique_ptr<Metadata> Create(const base::FilePath& metadata_file,
                                          const base::FilePath& report_dir);

  //! \brief Adds a new report to the set.
  //!
  //! \param[in] new_report_disk The record to add. The #state field must be set
  //!     to kPending.
  void AddNewRecord(const ReportDisk& new_report_disk);

  //! \brief Finds all reports in a given state. The \a reports vector is only
  //!     valid when CrashReportDatabase::kNoError is returned.
  //!
  //! \param[in] desired_state The state to match.
  //! \param[out] reports Matching reports, must be empty on entry.
  OperationStatus FindReports(
      ReportState desired_state,
      std::vector<CrashReportDatabase::Report>* reports) const;

  //! \brief Finds the report matching the given UUID.
  //!
  //! The returned report is only valid if CrashReportDatabase::kNoError is
  //! returned.
  //!
  //! \param[in] uuid The report identifier.
  //! \param[out] report_disk The found report, valid only if
  //!     CrashReportDatabase::kNoError is returned. Ownership is not
  //!     transferred to the caller, and the report may not be modified.
  OperationStatus FindSingleReport(const UUID& uuid,
                                   const ReportDisk** report_disk) const;

  //! \brief Finds a single report matching the given UUID and in the desired
  //!     state, and returns a mutable ReportDisk* if found.
  //!
  //! This marks the metadata as dirty, and on destruction, changes will be
  //! written to disk via Write().
  //!
  //! \return #kNoError on success. #kReportNotFound if there was no report with
  //!     the specified UUID, or if the report was not in the specified state
  //!     and was not uploading. #kBusyError if the report was not in the
  //!     specified state and was uploading.
  OperationStatus FindSingleReportAndMarkDirty(const UUID& uuid,
                                               ReportState desired_state,
                                               ReportDisk** report_disk);

  //! \brief Removes a report from the metadata database, without touching the
  //!     on-disk file.
  //!
  //! The returned report is only valid if CrashReportDatabase::kNoError is
  //! returned. This will mark the database as dirty. Future metadata
  //! operations for this report will not succeed.
  //!
  //! \param[in] uuid The report identifier to remove.
  //! \param[out] report_path The found report's file_path, valid only if
  //!     CrashReportDatabase::kNoError is returned.
  OperationStatus DeleteReport(const UUID& uuid,
                               base::FilePath* report_path);

 private:
  Metadata(FileHandle handle, const base::FilePath& report_dir);

  bool Rewind();

  void Read();
  void Write();

  //! \brief Confirms that the corresponding report actually exists on disk
  //!     (that is, the dump file has not been removed), and that the report is
  //!     in the given state.
  static OperationStatus VerifyReport(const ReportDisk& report_disk,
                                      ReportState desired_state);
  //! \brief Confirms that the corresponding report actually exists on disk
  //!     (that is, the dump file has not been removed).
  static OperationStatus VerifyReportAnyState(const ReportDisk& report_disk);

  ScopedFileHandle handle_;
  const base::FilePath report_dir_;
  bool dirty_;  //! \brief `true` when a Write() is required on destruction.
  std::vector<ReportDisk> reports_;

  DISALLOW_COPY_AND_ASSIGN(Metadata);
};

Metadata::~Metadata() {
  if (dirty_)
    Write();
  // Not actually async, UnlockFileEx requires the Offset fields.
  OVERLAPPED overlapped = {0};
  if (!UnlockFileEx(handle_.get(), 0, MAXDWORD, MAXDWORD, &overlapped))
    PLOG(ERROR) << "UnlockFileEx";
}

// static
std::unique_ptr<Metadata> Metadata::Create(const base::FilePath& metadata_file,
                                           const base::FilePath& report_dir) {
  // It is important that dwShareMode be non-zero so that concurrent access to
  // this file results in a successful open. This allows us to get to LockFileEx
  // which then blocks to guard access.
  FileHandle handle = CreateFile(metadata_file.value().c_str(),
                                 GENERIC_READ | GENERIC_WRITE,
                                 FILE_SHARE_READ | FILE_SHARE_WRITE,
                                 nullptr,
                                 OPEN_ALWAYS,
                                 FILE_ATTRIBUTE_NORMAL,
                                 nullptr);
  if (handle == kInvalidFileHandle)
    return std::unique_ptr<Metadata>();
  // Not actually async, LockFileEx requires the Offset fields.
  OVERLAPPED overlapped = {0};
  if (!LockFileEx(handle,
                  LOCKFILE_EXCLUSIVE_LOCK,
                  0,
                  MAXDWORD,
                  MAXDWORD,
                  &overlapped)) {
    PLOG(ERROR) << "LockFileEx";
    return std::unique_ptr<Metadata>();
  }

  std::unique_ptr<Metadata> metadata(new Metadata(handle, report_dir));
  // If Read() fails, for whatever reason (corruption, etc.) metadata will not
  // have been modified and will be in a clean empty state. We continue on and
  // return an empty database to hopefully recover. This means that existing
  // crash reports have been orphaned.
  metadata->Read();
  return metadata;
}

void Metadata::AddNewRecord(const ReportDisk& new_report_disk) {
  DCHECK(new_report_disk.state == ReportState::kPending);
  reports_.push_back(new_report_disk);
  dirty_ = true;
}

OperationStatus Metadata::FindReports(
    ReportState desired_state,
    std::vector<CrashReportDatabase::Report>* reports) const {
  DCHECK(reports->empty());
  for (const auto& report : reports_) {
    if (report.state == desired_state &&
        VerifyReport(report, desired_state) == CrashReportDatabase::kNoError) {
      reports->push_back(report);
    }
  }
  return CrashReportDatabase::kNoError;
}

OperationStatus Metadata::FindSingleReport(
    const UUID& uuid,
    const ReportDisk** out_report) const {
  auto report_iter = std::find_if(
      reports_.begin(), reports_.end(), [uuid](const ReportDisk& report) {
        return report.uuid == uuid;
      });
  if (report_iter == reports_.end())
    return CrashReportDatabase::kReportNotFound;
  OperationStatus os = VerifyReportAnyState(*report_iter);
  if (os == CrashReportDatabase::kNoError)
    *out_report = &*report_iter;
  return os;
}

OperationStatus Metadata::FindSingleReportAndMarkDirty(
    const UUID& uuid,
    ReportState desired_state,
    ReportDisk** report_disk) {
  auto report_iter = std::find_if(
      reports_.begin(), reports_.end(), [uuid](const ReportDisk& report) {
        return report.uuid == uuid;
      });
  if (report_iter == reports_.end())
    return CrashReportDatabase::kReportNotFound;
  OperationStatus os = VerifyReport(*report_iter, desired_state);
  if (os == CrashReportDatabase::kNoError) {
    dirty_ = true;
    *report_disk = &*report_iter;
  }
  return os;
}

OperationStatus Metadata::DeleteReport(const UUID& uuid,
                                       base::FilePath* report_path) {
  auto report_iter = std::find_if(
      reports_.begin(), reports_.end(), [uuid](const ReportDisk& report) {
        return report.uuid == uuid;
      });
  if (report_iter == reports_.end())
    return CrashReportDatabase::kReportNotFound;
  *report_path = report_iter->file_path;
  reports_.erase(report_iter);
  dirty_ = true;
  return CrashReportDatabase::kNoError;
}

Metadata::Metadata(FileHandle handle, const base::FilePath& report_dir)
    : handle_(handle), report_dir_(report_dir), dirty_(false), reports_() {
}

bool Metadata::Rewind() {
  FileOffset result = LoggingSeekFile(handle_.get(), 0, SEEK_SET);
  DCHECK_EQ(result, 0);
  return result == 0;
}

void Metadata::Read() {
  FileOffset length = LoggingSeekFile(handle_.get(), 0, SEEK_END);
  if (length <= 0)  // Failed, or empty: Abort.
    return;
  if (!Rewind()) {
    LOG(ERROR) << "failed to rewind to read";
    return;
  }

  MetadataFileHeader header;
  if (!LoggingReadFileExactly(handle_.get(), &header, sizeof(header))) {
    LOG(ERROR) << "failed to read header";
    return;
  }
  if (header.magic != kMetadataFileHeaderMagic ||
      header.version != kMetadataFileVersion) {
    LOG(ERROR) << "unexpected header";
    return;
  }

  base::CheckedNumeric<uint32_t> records_size =
      base::CheckedNumeric<uint32_t>(header.num_records) *
      static_cast<uint32_t>(sizeof(MetadataFileReportRecord));
  if (!records_size.IsValid()) {
    LOG(ERROR) << "record size out of range";
    return;
  }

  std::vector<ReportDisk> reports;
  if (header.num_records > 0) {
    std::vector<MetadataFileReportRecord> records(header.num_records);
    if (!LoggingReadFileExactly(
            handle_.get(), &records[0], records_size.ValueOrDie())) {
      LOG(ERROR) << "failed to read records";
      return;
    }

    std::string string_table = ReadRestOfFileAsString(handle_.get());
    if (string_table.empty() || string_table.back() != '\0') {
      LOG(ERROR) << "bad string table";
      return;
    }

    for (const auto& record : records) {
      if (record.file_path_index >= string_table.size() ||
          record.id_index >= string_table.size()) {
        LOG(ERROR) << "invalid string table index";
        return;
      }
      reports.push_back(ReportDisk(record, report_dir_, string_table));
    }
  }
  reports_.swap(reports);
}

void Metadata::Write() {
  if (!Rewind()) {
    LOG(ERROR) << "failed to rewind to write";
    return;
  }

  // Truncate to ensure that a partial write doesn't cause a mix of old and new
  // data causing an incorrect interpretation on read.
  if (!SetEndOfFile(handle_.get())) {
    PLOG(ERROR) << "failed to truncate";
    return;
  }

  size_t num_records = reports_.size();

  // Fill and write out the header.
  MetadataFileHeader header = {0};
  header.magic = kMetadataFileHeaderMagic;
  header.version = kMetadataFileVersion;
  header.num_records = base::checked_cast<uint32_t>(num_records);
  if (!LoggingWriteFile(handle_.get(), &header, sizeof(header))) {
    LOG(ERROR) << "failed to write header";
    return;
  }

  if (num_records == 0)
    return;

  // Build the records and string table we're going to write.
  std::string string_table;
  std::vector<MetadataFileReportRecord> records;
  records.reserve(num_records);
  for (const auto& report : reports_) {
    const base::FilePath& path = report.file_path;
    if (path.DirName() != report_dir_) {
      LOG(ERROR) << path.value().c_str() << " expected to start with "
                 << base::UTF16ToUTF8(report_dir_.value());
      return;
    }
    records.push_back(MetadataFileReportRecord(report, &string_table));
  }

  if (!LoggingWriteFile(handle_.get(),
                        &records[0],
                        records.size() * sizeof(MetadataFileReportRecord))) {
    LOG(ERROR) << "failed to write records";
    return;
  }
  if (!LoggingWriteFile(
          handle_.get(), string_table.c_str(), string_table.size())) {
    LOG(ERROR) << "failed to write string table";
    return;
  }
}

// static
OperationStatus Metadata::VerifyReportAnyState(const ReportDisk& report_disk) {
  DWORD fileattr = GetFileAttributes(report_disk.file_path.value().c_str());
  if (fileattr == INVALID_FILE_ATTRIBUTES)
    return CrashReportDatabase::kReportNotFound;
  return (fileattr & FILE_ATTRIBUTE_DIRECTORY)
             ? CrashReportDatabase::kFileSystemError
             : CrashReportDatabase::kNoError;
}

// static
OperationStatus Metadata::VerifyReport(const ReportDisk& report_disk,
                                       ReportState desired_state) {
  if (report_disk.state == desired_state) {
    return VerifyReportAnyState(report_disk);
  }

  return report_disk.state == ReportState::kUploading
             ? CrashReportDatabase::kBusyError
             : CrashReportDatabase::kReportNotFound;
}

bool EnsureDirectory(const base::FilePath& path) {
  DWORD fileattr = GetFileAttributes(path.value().c_str());
  if (fileattr == INVALID_FILE_ATTRIBUTES) {
    PLOG(ERROR) << "GetFileAttributes " << base::UTF16ToUTF8(path.value());
    return false;
  }
  if ((fileattr & FILE_ATTRIBUTE_DIRECTORY) == 0) {
    LOG(ERROR) << "GetFileAttributes "
               << base::UTF16ToUTF8(path.value())
               << ": not a directory";
    return false;
  }
  return true;
}

//! \brief Ensures that the node at path is a directory, and creates it if it
//!     does not exist.
//!
//! \return If the path points to a file, rather than a directory, or the
//!     directory could not be created, returns `false`. Otherwise, returns
//!     `true`, indicating that path already was or now is a directory.
bool CreateDirectoryIfNecessary(const base::FilePath& path) {
  if (CreateDirectory(path.value().c_str(), nullptr))
    return true;
  if (GetLastError() != ERROR_ALREADY_EXISTS) {
    PLOG(ERROR) << "CreateDirectory " << base::UTF16ToUTF8(path.value());
    return false;
  }
  return EnsureDirectory(path);
}

}  // namespace

// CrashReportDatabaseWin ------------------------------------------------------

class CrashReportDatabaseWin : public CrashReportDatabase {
 public:
  explicit CrashReportDatabaseWin(const base::FilePath& path);
  ~CrashReportDatabaseWin() override;

  bool Initialize(bool may_create);

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

 private:
  // CrashReportDatabase:
  OperationStatus RecordUploadAttempt(UploadReport* report,
                                      bool successful,
                                      const std::string& id) override;

  std::unique_ptr<Metadata> AcquireMetadata();

  base::FilePath base_dir_;
  Settings settings_;
  InitializationStateDcheck initialized_;

  DISALLOW_COPY_AND_ASSIGN(CrashReportDatabaseWin);
};

FileWriter* CrashReportDatabase::NewReport::AddAttachment(
    const std::string& name) {
  // Attachments aren't implemented in the Windows database yet.
  return nullptr;
}

void CrashReportDatabase::UploadReport::InitializeAttachments() {
  // Attachments aren't implemented in the Windows database yet.
}

CrashReportDatabaseWin::CrashReportDatabaseWin(const base::FilePath& path)
    : CrashReportDatabase(), base_dir_(path), settings_(), initialized_() {}

CrashReportDatabaseWin::~CrashReportDatabaseWin() {
}

bool CrashReportDatabaseWin::Initialize(bool may_create) {
  INITIALIZATION_STATE_SET_INITIALIZING(initialized_);

  // Ensure the database directory exists.
  if (may_create) {
    if (!CreateDirectoryIfNecessary(base_dir_))
      return false;
  } else if (!EnsureDirectory(base_dir_)) {
    return false;
  }

  // Ensure that the report subdirectory exists.
  if (!CreateDirectoryIfNecessary(base_dir_.Append(kReportsDirectory)))
    return false;

  if (!settings_.Initialize(base_dir_.Append(kSettings)))
    return false;

  INITIALIZATION_STATE_SET_VALID(initialized_);
  return true;
}

Settings* CrashReportDatabaseWin::GetSettings() {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return &settings_;
}

OperationStatus CrashReportDatabaseWin::PrepareNewCrashReport(
    std::unique_ptr<NewReport>* report) {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  std::unique_ptr<NewReport> new_report(new NewReport());
  if (!new_report->Initialize(this,
                              base_dir_.Append(kReportsDirectory),
                              std::wstring(L".") + kCrashReportFileExtension)) {
    return kFileSystemError;
  }

  report->reset(new_report.release());
  return kNoError;
}

OperationStatus CrashReportDatabaseWin::FinishedWritingCrashReport(
    std::unique_ptr<NewReport> report,
    UUID* uuid) {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  std::unique_ptr<Metadata> metadata(AcquireMetadata());
  if (!metadata)
    return kDatabaseError;
  metadata->AddNewRecord(ReportDisk(report->ReportID(),
                                    report->file_remover_.get(),
                                    time(nullptr),
                                    ReportState::kPending));

  ignore_result(report->file_remover_.release());

  *uuid = report->ReportID();

  Metrics::CrashReportPending(Metrics::PendingReportReason::kNewlyCreated);
  Metrics::CrashReportSize(report->Writer()->Seek(0, SEEK_END));

  return kNoError;
}

OperationStatus CrashReportDatabaseWin::LookUpCrashReport(const UUID& uuid,
                                                          Report* report) {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  std::unique_ptr<Metadata> metadata(AcquireMetadata());
  if (!metadata)
    return kDatabaseError;
  // Find and return a copy of the matching report.
  const ReportDisk* report_disk;
  OperationStatus os = metadata->FindSingleReport(uuid, &report_disk);
  if (os == kNoError)
    *report = *report_disk;
  return os;
}

OperationStatus CrashReportDatabaseWin::GetPendingReports(
    std::vector<Report>* reports) {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  std::unique_ptr<Metadata> metadata(AcquireMetadata());
  return metadata ? metadata->FindReports(ReportState::kPending, reports)
                  : kDatabaseError;
}

OperationStatus CrashReportDatabaseWin::GetCompletedReports(
    std::vector<Report>* reports) {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  std::unique_ptr<Metadata> metadata(AcquireMetadata());
  return metadata ? metadata->FindReports(ReportState::kCompleted, reports)
                  : kDatabaseError;
}

OperationStatus CrashReportDatabaseWin::GetReportForUploading(
    const UUID& uuid,
    std::unique_ptr<const UploadReport>* report,
    bool report_metrics) {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  std::unique_ptr<Metadata> metadata(AcquireMetadata());
  if (!metadata)
    return kDatabaseError;

  ReportDisk* report_disk;
  OperationStatus os = metadata->FindSingleReportAndMarkDirty(
      uuid, ReportState::kPending, &report_disk);
  if (os == kNoError) {
    report_disk->state = ReportState::kUploading;
    auto upload_report = std::make_unique<UploadReport>();
    *implicit_cast<Report*>(upload_report.get()) = *report_disk;

    if (!upload_report->Initialize(upload_report->file_path, this)) {
      return kFileSystemError;
    }
    upload_report->report_metrics_ = report_metrics;

    report->reset(upload_report.release());
  }
  return os;
}

OperationStatus CrashReportDatabaseWin::RecordUploadAttempt(
    UploadReport* report,
    bool successful,
    const std::string& id) {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  if (report->report_metrics_) {
    Metrics::CrashUploadAttempted(successful);
  }

  std::unique_ptr<Metadata> metadata(AcquireMetadata());
  if (!metadata)
    return kDatabaseError;
  ReportDisk* report_disk;
  OperationStatus os = metadata->FindSingleReportAndMarkDirty(
      report->uuid, ReportState::kUploading, &report_disk);
  if (os != kNoError)
    return os;

  time_t now = time(nullptr);

  report_disk->uploaded = successful;
  report_disk->id = id;
  report_disk->last_upload_attempt_time = now;
  report_disk->upload_attempts++;
  if (successful) {
    report_disk->state = ReportState::kCompleted;
    report_disk->upload_explicitly_requested = false;
  } else {
    report_disk->state = ReportState::kPending;
    report_disk->upload_explicitly_requested =
        report->upload_explicitly_requested;
  }

  if (!settings_.SetLastUploadAttemptTime(now))
    return kDatabaseError;

  return kNoError;
}

OperationStatus CrashReportDatabaseWin::DeleteReport(const UUID& uuid) {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  std::unique_ptr<Metadata> metadata(AcquireMetadata());
  if (!metadata)
    return kDatabaseError;

  base::FilePath report_path;
  OperationStatus os = metadata->DeleteReport(uuid, &report_path);
  if (os != kNoError)
    return os;

  if (!DeleteFile(report_path.value().c_str())) {
    PLOG(ERROR) << "DeleteFile "
                << base::UTF16ToUTF8(report_path.value());
    return kFileSystemError;
  }
  return kNoError;
}

OperationStatus CrashReportDatabaseWin::SkipReportUpload(
    const UUID& uuid,
    Metrics::CrashSkippedReason reason) {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  Metrics::CrashUploadSkipped(reason);

  std::unique_ptr<Metadata> metadata(AcquireMetadata());
  if (!metadata)
    return kDatabaseError;
  ReportDisk* report_disk;
  OperationStatus os = metadata->FindSingleReportAndMarkDirty(
      uuid, ReportState::kPending, &report_disk);
  if (os == kNoError) {
    report_disk->state = ReportState::kCompleted;
    report_disk->upload_explicitly_requested = false;
  }
  return os;
}

std::unique_ptr<Metadata> CrashReportDatabaseWin::AcquireMetadata() {
  base::FilePath metadata_file = base_dir_.Append(kMetadataFileName);
  return Metadata::Create(metadata_file, base_dir_.Append(kReportsDirectory));
}

std::unique_ptr<CrashReportDatabase> InitializeInternal(
    const base::FilePath& path,
    bool may_create) {
  std::unique_ptr<CrashReportDatabaseWin> database_win(
      new CrashReportDatabaseWin(path));
  return database_win->Initialize(may_create)
             ? std::move(database_win)
             : std::unique_ptr<CrashReportDatabaseWin>();
}

OperationStatus CrashReportDatabaseWin::RequestUpload(const UUID& uuid) {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  std::unique_ptr<Metadata> metadata(AcquireMetadata());
  if (!metadata)
    return kDatabaseError;

  ReportDisk* report_disk;
  // TODO(gayane): Search for the report only once regardless of its state.
  OperationStatus os = metadata->FindSingleReportAndMarkDirty(
      uuid, ReportState::kCompleted, &report_disk);
  if (os == kReportNotFound) {
    os = metadata->FindSingleReportAndMarkDirty(
        uuid, ReportState::kPending, &report_disk);
  }

  if (os != kNoError)
    return os;

  // If the crash report has already been uploaded, don't request new upload.
  if (report_disk->uploaded)
    return kCannotRequestUpload;

  // Mark the crash report as having upload explicitly requested by the user,
  // and move it to the pending state.
  report_disk->upload_explicitly_requested = true;
  report_disk->state = ReportState::kPending;

  Metrics::CrashReportPending(Metrics::PendingReportReason::kUserInitiated);

  return kNoError;
}

// static
std::unique_ptr<CrashReportDatabase> CrashReportDatabase::Initialize(
    const base::FilePath& path) {
  return InitializeInternal(path, true);
}

// static
std::unique_ptr<CrashReportDatabase>
CrashReportDatabase::InitializeWithoutCreating(const base::FilePath& path) {
  return InitializeInternal(path, false);
}

}  // namespace crashpad
