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

#ifndef CRASHPAD_CLIENT_CRASH_REPORT_DATABASE_H_
#define CRASHPAD_CLIENT_CRASH_REPORT_DATABASE_H_

#include <time.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "base/files/file_path.h"
#include "base/macros.h"
#include "util/file/file_io.h"
#include "util/file/file_reader.h"
#include "util/file/file_writer.h"
#include "util/file/scoped_remove_file.h"
#include "util/misc/metrics.h"
#include "util/misc/uuid.h"

namespace crashpad {

class Settings;

//! \brief An interface for managing a collection of crash report files and
//!     metadata associated with the crash reports.
//!
//! All Report objects that are returned by this class are logically const.
//! They are snapshots of the database at the time the query was run, and the
//! data returned is liable to change after the query is executed.
//!
//! The lifecycle of a crash report has three stages:
//!
//!   1. New: A crash report is created with PrepareNewCrashReport(), the
//!      the client then writes the report, and then calls
//!      FinishedWritingCrashReport() to make the report Pending.
//!   2. Pending: The report has been written but has not been locally
//!      processed, or it was has been brought back from 'Completed' state by
//!      user request.
//!   3. Completed: The report has been locally processed, either by uploading
//!      it to a collection server and calling RecordUploadComplete(), or by
//!      calling SkipReportUpload().
class CrashReportDatabase {
 public:
  //! \brief A crash report record.
  //!
  //! This represents the metadata for a crash report, as well as the location
  //! of the report itself. A CrashReportDatabase maintains at least this
  //! information.
  struct Report {
    Report();

    //! A unique identifier by which this report will always be known to the
    //! database.
    UUID uuid;

    //! The current location of the crash report on the client’s filesystem.
    //! The location of a crash report may change over time, so the UUID should
    //! be used as the canonical identifier.
    base::FilePath file_path;

    //! An identifier issued to this crash report by a collection server.
    std::string id;

    //! The time at which the report was generated.
    time_t creation_time;

    //! Whether this crash report was successfully uploaded to a collection
    //! server.
    bool uploaded;

    //! The last timestamp at which an attempt was made to submit this crash
    //! report to a collection server. If this is zero, then the report has
    //! never been uploaded. If #uploaded is true, then this timestamp is the
    //! time at which the report was uploaded, and no other attempts to upload
    //! this report will be made.
    time_t last_upload_attempt_time;

    //! The number of times an attempt was made to submit this report to
    //! a collection server. If this is more than zero, then
    //! #last_upload_attempt_time will be set to the timestamp of the most
    //! recent attempt.
    int upload_attempts;

    //! Whether this crash report was explicitly requested by user to be
    //! uploaded. This can be true only if report is in the 'pending' state.
    bool upload_explicitly_requested;
  };

  //! \brief A crash report that is in the process of being written.
  //!
  //! An instance of this class should be created via PrepareNewCrashReport().
  class NewReport {
   public:
    NewReport();
    ~NewReport();

    //! An open FileWriter with which to write the report.
    FileWriter* Writer() const { return writer_.get(); }

    //! A unique identifier by which this report will always be known to the
    //! database.
    const UUID& ReportID() const { return uuid_; }

    //! \brief Adds an attachment to the report.
    //!
    //! \note This function is not yet implemented on macOS or Windows.
    //!
    //! \param[in] name The key and name for the attachment, which will be
    //!     included in the http upload. The attachment will not appear in the
    //!     minidump report. \a name should only use characters from the set
    //!     `[a-zA-Z0-9._-]`.
    //! \return A FileWriter that the caller should use to write the contents of
    //!     the attachment, or `nullptr` on failure with an error logged.
    FileWriter* AddAttachment(const std::string& name);

   private:
    friend class CrashReportDatabaseGeneric;
    friend class CrashReportDatabaseMac;
    friend class CrashReportDatabaseWin;

    bool Initialize(CrashReportDatabase* database,
                    const base::FilePath& directory,
                    const base::FilePath::StringType& extension);

    std::unique_ptr<FileWriter> writer_;
    ScopedRemoveFile file_remover_;
    std::vector<std::unique_ptr<FileWriter>> attachment_writers_;
    std::vector<ScopedRemoveFile> attachment_removers_;
    UUID uuid_;
    CrashReportDatabase* database_;

    DISALLOW_COPY_AND_ASSIGN(NewReport);
  };

  //! \brief A crash report that is in the process of being uploaded.
  //!
  //! An instance of this class should be created via GetReportForUploading().
  class UploadReport : public Report {
   public:
    UploadReport();
    virtual ~UploadReport();

    //! \brief An open FileReader with which to read the report.
    FileReader* Reader() const { return reader_.get(); }

    //! \brief Obtains a mapping of names to file readers for any attachments
    //!     for the report.
    //!
    //! This is not implemented on macOS or Windows.
    std::map<std::string, FileReader*> GetAttachments() const {
      return attachment_map_;
    };

   private:
    friend class CrashReportDatabase;
    friend class CrashReportDatabaseGeneric;
    friend class CrashReportDatabaseMac;
    friend class CrashReportDatabaseWin;

    bool Initialize(const base::FilePath path, CrashReportDatabase* database);
    void InitializeAttachments();

    std::unique_ptr<FileReader> reader_;
    CrashReportDatabase* database_;
    std::vector<std::unique_ptr<FileReader>> attachment_readers_;
    std::map<std::string, FileReader*> attachment_map_;
    bool report_metrics_;

    DISALLOW_COPY_AND_ASSIGN(UploadReport);
  };

  //! \brief The result code for operations performed on a database.
  enum OperationStatus {
    //! \brief No error occurred.
    kNoError = 0,

    //! \brief The report that was requested could not be located.
    //!
    //! This may occur when the report is present in the database but not in a
    //! state appropriate for the requested operation, for example, if
    //! GetReportForUploading() is called to obtain report that’s already in the
    //! completed state.
    kReportNotFound,

    //! \brief An error occured while performing a file operation on a crash
    //!     report.
    //!
    //! A database is responsible for managing both the metadata about a report
    //! and the actual crash report itself. This error is returned when an
    //! error occurred when managing the report file. Additional information
    //! will be logged.
    kFileSystemError,

    //! \brief An error occured while recording metadata for a crash report or
    //!     database-wide settings.
    //!
    //! A database is responsible for managing both the metadata about a report
    //! and the actual crash report itself. This error is returned when an
    //! error occurred when managing the metadata about a crash report or
    //! database-wide settings. Additional information will be logged.
    kDatabaseError,

    //! \brief The operation could not be completed because a concurrent
    //!     operation affecting the report is occurring.
    kBusyError,

    //! \brief The report cannot be uploaded by user request as it has already
    //!     been uploaded.
    kCannotRequestUpload,
  };

  virtual ~CrashReportDatabase() {}

  //! \brief Opens a database of crash reports, possibly creating it.
  //!
  //! \param[in] path A path to the database to be created or opened. If the
  //!     database does not yet exist, it will be created if possible. Note that
  //!     for databases implemented as directory structures, existence refers
  //!     solely to the outermost directory.
  //!
  //! \return A database object on success, `nullptr` on failure with an error
  //!     logged.
  //!
  //! \sa InitializeWithoutCreating
  static std::unique_ptr<CrashReportDatabase> Initialize(
      const base::FilePath& path);

  //! \brief Opens an existing database of crash reports.
  //!
  //! \param[in] path A path to the database to be opened. If the database does
  //!     not yet exist, it will not be created. Note that for databases
  //!     implemented as directory structures, existence refers solely to the
  //!     outermost directory. On such databases, as long as the outermost
  //!     directory is present, this method will create the inner structure.
  //!
  //! \return A database object on success, `nullptr` on failure with an error
  //!     logged.
  //!
  //! \sa Initialize
  static std::unique_ptr<CrashReportDatabase> InitializeWithoutCreating(
      const base::FilePath& path);

  //! \brief Returns the Settings object for this database.
  //!
  //! \return A weak pointer to the Settings object, which is owned by the
  //!     database.
  virtual Settings* GetSettings() = 0;

  //! \brief Creates a record of a new crash report.
  //!
  //! Callers should write the crash report using the FileWriter provided.
  //! Callers should then call FinishedWritingCrashReport() to complete report
  //! creation. If an error is encountered while writing the crash report, no
  //! special action needs to be taken. If FinishedWritingCrashReport() is not
  //! called, the report will be removed from the database when \a report is
  //! destroyed.
  //!
  //! \param[out] report A NewReport object containing a FileWriter with which
  //!     to write the report data. Only valid if this returns #kNoError.
  //!
  //! \return The operation status code.
  virtual OperationStatus PrepareNewCrashReport(
      std::unique_ptr<NewReport>* report) = 0;

  //! \brief Informs the database that a crash report has been successfully
  //!     written.
  //!
  //! \param[in] report A NewReport obtained with PrepareNewCrashReport(). The
  //!     NewReport object will be invalidated as part of this call.
  //! \param[out] uuid The UUID of this crash report.
  //!
  //! \return The operation status code.
  virtual OperationStatus FinishedWritingCrashReport(
      std::unique_ptr<NewReport> report,
      UUID* uuid) = 0;

  //! \brief Returns the crash report record for the unique identifier.
  //!
  //! \param[in] uuid The crash report record unique identifier.
  //! \param[out] report A crash report record. Only valid if this returns
  //!     #kNoError.
  //!
  //! \return The operation status code.
  virtual OperationStatus LookUpCrashReport(const UUID& uuid,
                                            Report* report) = 0;

  //! \brief Returns a list of crash report records that have not been uploaded.
  //!
  //! \param[out] reports A list of crash report record objects. This must be
  //!     empty on entry. Only valid if this returns #kNoError.
  //!
  //! \return The operation status code.
  virtual OperationStatus GetPendingReports(std::vector<Report>* reports) = 0;

  //! \brief Returns a list of crash report records that have been completed,
  //!     either by being uploaded or by skipping upload.
  //!
  //! \param[out] reports A list of crash report record objects. This must be
  //!     empty on entry. Only valid if this returns #kNoError.
  //!
  //! \return The operation status code.
  virtual OperationStatus GetCompletedReports(std::vector<Report>* reports) = 0;

  //! \brief Obtains and locks a report object for uploading to a collection
  //!     server.
  //!
  //! Callers should upload the crash report using the FileReader provided.
  //! Callers should then call RecordUploadComplete() to record a successful
  //! upload. If RecordUploadComplete() is not called, the upload attempt will
  //! be recorded as unsuccessful and the report lock released when \a report is
  //! destroyed.
  //!
  //! \param[in] uuid The unique identifier for the crash report record.
  //! \param[out] report A crash report record for the report to be uploaded.
  //!     Only valid if this returns #kNoError.
  //! \param[in] report_metrics If `false`, metrics will not be recorded for
  //!     this upload attempt when RecordUploadComplete() is called or \a report
  //!     is destroyed. Metadata for the upload attempt will still be recorded
  //!     in the database.
  //!
  //! \return The operation status code.
  virtual OperationStatus GetReportForUploading(
      const UUID& uuid,
      std::unique_ptr<const UploadReport>* report,
      bool report_metrics = true) = 0;

  //! \brief Records a successful upload for a report and updates the last
  //!     upload attempt time as returned by
  //!     Settings::GetLastUploadAttemptTime().
  //!
  //! \param[in] report A UploadReport object obtained from
  //!     GetReportForUploading(). The UploadReport object will be invalidated
  //!     and the report unlocked as part of this call.
  //! \param[in] id The possibly empty identifier assigned to this crash report
  //!     by the collection server.
  //!
  //! \return The operation status code.
  OperationStatus RecordUploadComplete(
      std::unique_ptr<const UploadReport> report,
      const std::string& id);

  //! \brief Moves a report from the pending state to the completed state, but
  //!     without the report being uploaded.
  //!
  //! This can be used if the user has disabled crash report collection, but
  //! crash generation is still enabled in the product.
  //!
  //! \param[in] uuid The unique identifier for the crash report record.
  //! \param[in] reason The reason the report upload is being skipped for
  //!     metrics tracking purposes.
  //!
  //! \return The operation status code.
  virtual OperationStatus SkipReportUpload(
      const UUID& uuid,
      Metrics::CrashSkippedReason reason) = 0;

  //! \brief Deletes a crash report file and its associated metadata.
  //!
  //! \param[in] uuid The UUID of the report to delete.
  //!
  //! \return The operation status code.
  virtual OperationStatus DeleteReport(const UUID& uuid) = 0;

  //! \brief Marks a crash report as explicitly requested to be uploaded by the
  //!     user and moves it to 'pending' state.
  //!
  //! \param[in] uuid The unique identifier for the crash report record.
  //!
  //! \return The operation status code.
  virtual OperationStatus RequestUpload(const UUID& uuid) = 0;

  //! \brief Cleans the database of expired lockfiles, metadata without report
  //!     files, and report files without metadata.
  //!
  //! This method does nothing on the macOS and Windows implementations of the
  //! database.
  //!
  //! \param[in] lockfile_ttl The number of seconds at which lockfiles or new
  //!     report files are considered expired.
  //! \return The number of reports cleaned.
  virtual int CleanDatabase(time_t lockfile_ttl) { return 0; }

 protected:
  CrashReportDatabase() {}

 private:
  //! \brief Adjusts a crash report record’s metadata to account for an upload
  //!     attempt, and updates the last upload attempt time as returned by
  //!     Settings::GetLastUploadAttemptTime().
  //!
  //! \param[in] report The report object obtained from
  //!     GetReportForUploading().
  //! \param[in] successful Whether the upload attempt was successful.
  //! \param[in] id The identifier assigned to this crash report by the
  //!     collection server. Must be empty if \a successful is `false`; may be
  //!     empty if it is `true`.
  //!
  //! \return The operation status code.
  virtual OperationStatus RecordUploadAttempt(UploadReport* report,
                                              bool successful,
                                              const std::string& id) = 0;

  DISALLOW_COPY_AND_ASSIGN(CrashReportDatabase);
};

}  // namespace crashpad

#endif  // CRASHPAD_CLIENT_CRASH_REPORT_DATABASE_H_
