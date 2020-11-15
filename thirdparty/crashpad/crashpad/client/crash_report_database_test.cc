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

#include "build/build_config.h"
#include "client/settings.h"
#include "gtest/gtest.h"
#include "test/errors.h"
#include "test/file.h"
#include "test/filesystem.h"
#include "test/gtest_disabled.h"
#include "test/scoped_temp_dir.h"
#include "util/file/file_io.h"
#include "util/file/filesystem.h"

namespace crashpad {
namespace test {
namespace {

class CrashReportDatabaseTest : public testing::Test {
 public:
  CrashReportDatabaseTest() {
  }

 protected:
  // testing::Test:
  void SetUp() override {
    db_ = CrashReportDatabase::Initialize(path());
    ASSERT_TRUE(db_);
  }

  void ResetDatabase() {
    db_.reset();
  }

  CrashReportDatabase* db() { return db_.get(); }
  base::FilePath path() const {
    return temp_dir_.path().Append(FILE_PATH_LITERAL("crashpad_test_database"));
  }

  void CreateCrashReport(CrashReportDatabase::Report* report) {
    std::unique_ptr<CrashReportDatabase::NewReport> new_report;
    ASSERT_EQ(db_->PrepareNewCrashReport(&new_report),
              CrashReportDatabase::kNoError);
    static constexpr char kTest[] = "test";
    ASSERT_TRUE(new_report->Writer()->Write(kTest, sizeof(kTest)));

    UUID uuid;
    EXPECT_EQ(db_->FinishedWritingCrashReport(std::move(new_report), &uuid),
              CrashReportDatabase::kNoError);

    EXPECT_EQ(db_->LookUpCrashReport(uuid, report),
              CrashReportDatabase::kNoError);
    ExpectPreparedCrashReport(*report);
  }

  void UploadReport(const UUID& uuid, bool successful, const std::string& id) {
    Settings* const settings = db_->GetSettings();
    ASSERT_TRUE(settings);
    time_t times[2];
    ASSERT_TRUE(settings->GetLastUploadAttemptTime(&times[0]));

    std::unique_ptr<const CrashReportDatabase::UploadReport> report;
    ASSERT_EQ(db_->GetReportForUploading(uuid, &report),
              CrashReportDatabase::kNoError);
    EXPECT_NE(report->uuid, UUID());
    EXPECT_FALSE(report->file_path.empty());
    EXPECT_TRUE(FileExists(report->file_path)) << report->file_path.value();
    EXPECT_GT(report->creation_time, 0);
    if (successful) {
      EXPECT_EQ(db_->RecordUploadComplete(std::move(report), id),
                CrashReportDatabase::kNoError);
    } else {
      report.reset();
    }

    ASSERT_TRUE(settings->GetLastUploadAttemptTime(&times[1]));
    EXPECT_NE(times[1], 0);
    EXPECT_GE(times[1], times[0]);
  }

  void ExpectPreparedCrashReport(const CrashReportDatabase::Report& report) {
    EXPECT_NE(report.uuid, UUID());
    EXPECT_FALSE(report.file_path.empty());
    EXPECT_TRUE(FileExists(report.file_path)) << report.file_path.value();
    EXPECT_TRUE(report.id.empty());
    EXPECT_GT(report.creation_time, 0);
    EXPECT_FALSE(report.uploaded);
    EXPECT_EQ(report.last_upload_attempt_time, 0);
    EXPECT_EQ(report.upload_attempts, 0);
    EXPECT_FALSE(report.upload_explicitly_requested);
  }

  void RelocateDatabase() {
    ResetDatabase();
    temp_dir_.Rename();
    SetUp();
  }

  CrashReportDatabase::OperationStatus RequestUpload(const UUID& uuid) {
    CrashReportDatabase::OperationStatus os = db()->RequestUpload(uuid);

    CrashReportDatabase::Report report;
    EXPECT_EQ(db_->LookUpCrashReport(uuid, &report),
              CrashReportDatabase::kNoError);

    return os;
  }

 private:
  ScopedTempDir temp_dir_;
  std::unique_ptr<CrashReportDatabase> db_;

  DISALLOW_COPY_AND_ASSIGN(CrashReportDatabaseTest);
};

TEST_F(CrashReportDatabaseTest, Initialize) {
  // Initialize the database for the first time, creating it.
  ASSERT_TRUE(db());

  Settings* settings = db()->GetSettings();

  UUID client_ids[3];
  ASSERT_TRUE(settings->GetClientID(&client_ids[0]));
  EXPECT_NE(client_ids[0], UUID());

  time_t last_upload_attempt_time;
  ASSERT_TRUE(settings->GetLastUploadAttemptTime(&last_upload_attempt_time));
  EXPECT_EQ(last_upload_attempt_time, 0);

  // Close and reopen the database at the same path.
  ResetDatabase();
  EXPECT_FALSE(db());
  auto db = CrashReportDatabase::InitializeWithoutCreating(path());
  ASSERT_TRUE(db);

  settings = db->GetSettings();

  ASSERT_TRUE(settings->GetClientID(&client_ids[1]));
  EXPECT_EQ(client_ids[1], client_ids[0]);

  ASSERT_TRUE(settings->GetLastUploadAttemptTime(&last_upload_attempt_time));
  EXPECT_EQ(last_upload_attempt_time, 0);

  // Check that the database can also be opened by the method that is permitted
  // to create it.
  db = CrashReportDatabase::Initialize(path());
  ASSERT_TRUE(db);

  settings = db->GetSettings();

  ASSERT_TRUE(settings->GetClientID(&client_ids[2]));
  EXPECT_EQ(client_ids[2], client_ids[0]);

  ASSERT_TRUE(settings->GetLastUploadAttemptTime(&last_upload_attempt_time));
  EXPECT_EQ(last_upload_attempt_time, 0);

  std::vector<CrashReportDatabase::Report> reports;
  EXPECT_EQ(db->GetPendingReports(&reports), CrashReportDatabase::kNoError);
  EXPECT_TRUE(reports.empty());
  reports.clear();
  EXPECT_EQ(db->GetCompletedReports(&reports), CrashReportDatabase::kNoError);
  EXPECT_TRUE(reports.empty());

  // InitializeWithoutCreating() shouldn’t create a nonexistent database.
  base::FilePath non_database_path =
      path().DirName().Append(FILE_PATH_LITERAL("not_a_database"));
  db = CrashReportDatabase::InitializeWithoutCreating(non_database_path);
  EXPECT_FALSE(db);
}

TEST_F(CrashReportDatabaseTest, NewCrashReport) {
  std::unique_ptr<CrashReportDatabase::NewReport> new_report;
  EXPECT_EQ(db()->PrepareNewCrashReport(&new_report),
            CrashReportDatabase::kNoError);
  UUID expect_uuid = new_report->ReportID();
  UUID uuid;
  EXPECT_EQ(db()->FinishedWritingCrashReport(std::move(new_report), &uuid),
            CrashReportDatabase::kNoError);
  EXPECT_EQ(uuid, expect_uuid);

  CrashReportDatabase::Report report;
  EXPECT_EQ(db()->LookUpCrashReport(uuid, &report),
            CrashReportDatabase::kNoError);
  ExpectPreparedCrashReport(report);

  std::vector<CrashReportDatabase::Report> reports;
  EXPECT_EQ(db()->GetPendingReports(&reports), CrashReportDatabase::kNoError);
  ASSERT_EQ(reports.size(), 1u);
  EXPECT_EQ(reports[0].uuid, report.uuid);

  reports.clear();
  EXPECT_EQ(db()->GetCompletedReports(&reports), CrashReportDatabase::kNoError);
  EXPECT_TRUE(reports.empty());
}

TEST_F(CrashReportDatabaseTest, LookUpCrashReport) {
  UUID uuid;

  {
    CrashReportDatabase::Report report;
    CreateCrashReport(&report);
    uuid = report.uuid;
  }

  {
    CrashReportDatabase::Report report;
    EXPECT_EQ(db()->LookUpCrashReport(uuid, &report),
              CrashReportDatabase::kNoError);
    EXPECT_EQ(report.uuid, uuid);
    EXPECT_NE(report.file_path.value().find(path().value()), std::string::npos);
    EXPECT_EQ(report.id, std::string());
    EXPECT_FALSE(report.uploaded);
    EXPECT_EQ(report.last_upload_attempt_time, 0);
    EXPECT_EQ(report.upload_attempts, 0);
    EXPECT_FALSE(report.upload_explicitly_requested);
  }

  UploadReport(uuid, true, "test");

  {
    CrashReportDatabase::Report report;
    EXPECT_EQ(db()->LookUpCrashReport(uuid, &report),
              CrashReportDatabase::kNoError);
    EXPECT_EQ(report.uuid, uuid);
    EXPECT_NE(report.file_path.value().find(path().value()), std::string::npos);
    EXPECT_EQ(report.id, "test");
    EXPECT_TRUE(report.uploaded);
    EXPECT_NE(report.last_upload_attempt_time, 0);
    EXPECT_EQ(report.upload_attempts, 1);
    EXPECT_FALSE(report.upload_explicitly_requested);
  }
}

TEST_F(CrashReportDatabaseTest, RecordUploadAttempt) {
  std::vector<CrashReportDatabase::Report> reports(3);
  CreateCrashReport(&reports[0]);
  CreateCrashReport(&reports[1]);
  CreateCrashReport(&reports[2]);

  // Record two attempts: one successful, one not.
  UploadReport(reports[1].uuid, false, std::string());
  UploadReport(reports[2].uuid, true, "abc123");

  std::vector<CrashReportDatabase::Report> query(3);
  EXPECT_EQ(db()->LookUpCrashReport(reports[0].uuid, &query[0]),
            CrashReportDatabase::kNoError);
  EXPECT_EQ(db()->LookUpCrashReport(reports[1].uuid, &query[1]),
            CrashReportDatabase::kNoError);
  EXPECT_EQ(db()->LookUpCrashReport(reports[2].uuid, &query[2]),
            CrashReportDatabase::kNoError);

  EXPECT_EQ(query[0].id, std::string());
  EXPECT_EQ(query[1].id, std::string());
  EXPECT_EQ(query[2].id, "abc123");

  EXPECT_FALSE(query[0].uploaded);
  EXPECT_FALSE(query[1].uploaded);
  EXPECT_TRUE(query[2].uploaded);

  EXPECT_EQ(query[0].last_upload_attempt_time, 0);
  EXPECT_NE(query[1].last_upload_attempt_time, 0);
  EXPECT_NE(query[2].last_upload_attempt_time, 0);

  EXPECT_EQ(query[0].upload_attempts, 0);
  EXPECT_EQ(query[1].upload_attempts, 1);
  EXPECT_EQ(query[2].upload_attempts, 1);

  // Attempt to upload and fail again.
  UploadReport(reports[1].uuid, false, std::string());

  time_t report_2_upload_time = query[2].last_upload_attempt_time;

  EXPECT_EQ(db()->LookUpCrashReport(reports[0].uuid, &query[0]),
            CrashReportDatabase::kNoError);
  EXPECT_EQ(db()->LookUpCrashReport(reports[1].uuid, &query[1]),
            CrashReportDatabase::kNoError);
  EXPECT_EQ(db()->LookUpCrashReport(reports[2].uuid, &query[2]),
            CrashReportDatabase::kNoError);

  EXPECT_FALSE(query[0].uploaded);
  EXPECT_FALSE(query[1].uploaded);
  EXPECT_TRUE(query[2].uploaded);

  EXPECT_EQ(query[0].last_upload_attempt_time, 0);
  EXPECT_GE(query[1].last_upload_attempt_time, report_2_upload_time);
  EXPECT_EQ(query[2].last_upload_attempt_time, report_2_upload_time);

  EXPECT_EQ(query[0].upload_attempts, 0);
  EXPECT_EQ(query[1].upload_attempts, 2);
  EXPECT_EQ(query[2].upload_attempts, 1);

  // Third time's the charm: upload and succeed.
  UploadReport(reports[1].uuid, true, "666hahaha");

  time_t report_1_upload_time = query[1].last_upload_attempt_time;

  EXPECT_EQ(db()->LookUpCrashReport(reports[0].uuid, &query[0]),
            CrashReportDatabase::kNoError);
  EXPECT_EQ(db()->LookUpCrashReport(reports[1].uuid, &query[1]),
            CrashReportDatabase::kNoError);
  EXPECT_EQ(db()->LookUpCrashReport(reports[2].uuid, &query[2]),
            CrashReportDatabase::kNoError);

  EXPECT_FALSE(query[0].uploaded);
  EXPECT_TRUE(query[1].uploaded);
  EXPECT_TRUE(query[2].uploaded);

  EXPECT_EQ(query[0].last_upload_attempt_time, 0);
  EXPECT_GE(query[1].last_upload_attempt_time, report_1_upload_time);
  EXPECT_EQ(query[2].last_upload_attempt_time, report_2_upload_time);

  EXPECT_EQ(query[0].upload_attempts, 0);
  EXPECT_EQ(query[1].upload_attempts, 3);
  EXPECT_EQ(query[2].upload_attempts, 1);
}

// This test covers both query functions since they are related.
TEST_F(CrashReportDatabaseTest, GetCompletedAndNotUploadedReports) {
  std::vector<CrashReportDatabase::Report> reports(5);
  CreateCrashReport(&reports[0]);
  CreateCrashReport(&reports[1]);
  CreateCrashReport(&reports[2]);
  CreateCrashReport(&reports[3]);
  CreateCrashReport(&reports[4]);

  const UUID& report_0_uuid = reports[0].uuid;
  const UUID& report_1_uuid = reports[1].uuid;
  const UUID& report_2_uuid = reports[2].uuid;
  const UUID& report_3_uuid = reports[3].uuid;
  const UUID& report_4_uuid = reports[4].uuid;

  std::vector<CrashReportDatabase::Report> pending;
  EXPECT_EQ(db()->GetPendingReports(&pending), CrashReportDatabase::kNoError);

  std::vector<CrashReportDatabase::Report> completed;
  EXPECT_EQ(db()->GetCompletedReports(&completed),
            CrashReportDatabase::kNoError);

  EXPECT_EQ(pending.size(), reports.size());
  EXPECT_EQ(completed.size(), 0u);

  // Upload one report successfully.
  UploadReport(report_1_uuid, true, "report1");

  pending.clear();
  EXPECT_EQ(db()->GetPendingReports(&pending), CrashReportDatabase::kNoError);
  completed.clear();
  EXPECT_EQ(db()->GetCompletedReports(&completed),
            CrashReportDatabase::kNoError);

  EXPECT_EQ(pending.size(), 4u);
  ASSERT_EQ(completed.size(), 1u);

  for (const auto& report : pending) {
    EXPECT_NE(report.uuid, report_1_uuid);
    EXPECT_FALSE(report.file_path.empty());
  }
  EXPECT_EQ(completed[0].uuid, report_1_uuid);
  EXPECT_EQ(completed[0].id, "report1");
  EXPECT_EQ(completed[0].uploaded, true);
  EXPECT_GT(completed[0].last_upload_attempt_time, 0);
  EXPECT_EQ(completed[0].upload_attempts, 1);

  // Fail to upload one report.
  UploadReport(report_2_uuid, false, std::string());

  pending.clear();
  EXPECT_EQ(db()->GetPendingReports(&pending), CrashReportDatabase::kNoError);
  completed.clear();
  EXPECT_EQ(db()->GetCompletedReports(&completed),
            CrashReportDatabase::kNoError);

  EXPECT_EQ(pending.size(), 4u);
  ASSERT_EQ(completed.size(), 1u);

  for (const auto& report : pending) {
    if (report.upload_attempts != 0) {
      EXPECT_EQ(report.uuid, report_2_uuid);
      EXPECT_GT(report.last_upload_attempt_time, 0);
      EXPECT_FALSE(report.uploaded);
      EXPECT_TRUE(report.id.empty());
    }
    EXPECT_FALSE(report.file_path.empty());
  }

  // Upload a second report.
  UploadReport(report_4_uuid, true, "report_4");

  pending.clear();
  EXPECT_EQ(db()->GetPendingReports(&pending), CrashReportDatabase::kNoError);
  completed.clear();
  EXPECT_EQ(db()->GetCompletedReports(&completed),
            CrashReportDatabase::kNoError);

  EXPECT_EQ(pending.size(), 3u);
  ASSERT_EQ(completed.size(), 2u);

  // Succeed the failed report.
  UploadReport(report_2_uuid, true, "report 2");

  pending.clear();
  EXPECT_EQ(db()->GetPendingReports(&pending), CrashReportDatabase::kNoError);
  completed.clear();
  EXPECT_EQ(db()->GetCompletedReports(&completed),
            CrashReportDatabase::kNoError);

  EXPECT_EQ(pending.size(), 2u);
  ASSERT_EQ(completed.size(), 3u);

  for (const auto& report : pending) {
    EXPECT_TRUE(report.uuid == report_0_uuid || report.uuid == report_3_uuid);
    EXPECT_FALSE(report.file_path.empty());
  }

  // Skip upload for one report.
  EXPECT_EQ(db()->SkipReportUpload(
                report_3_uuid, Metrics::CrashSkippedReason::kUploadsDisabled),
            CrashReportDatabase::kNoError);

  pending.clear();
  EXPECT_EQ(db()->GetPendingReports(&pending), CrashReportDatabase::kNoError);
  completed.clear();
  EXPECT_EQ(db()->GetCompletedReports(&completed),
            CrashReportDatabase::kNoError);

  ASSERT_EQ(pending.size(), 1u);
  ASSERT_EQ(completed.size(), 4u);

  EXPECT_EQ(pending[0].uuid, report_0_uuid);

  for (const auto& report : completed) {
    if (report.uuid == report_3_uuid) {
      EXPECT_FALSE(report.uploaded);
      EXPECT_EQ(report.upload_attempts, 0);
      EXPECT_EQ(report.last_upload_attempt_time, 0);
    } else {
      EXPECT_TRUE(report.uploaded);
      EXPECT_GT(report.upload_attempts, 0);
      EXPECT_GT(report.last_upload_attempt_time, 0);
    }
    EXPECT_FALSE(report.file_path.empty());
  }
}

TEST_F(CrashReportDatabaseTest, DuelingUploads) {
  CrashReportDatabase::Report report;
  CreateCrashReport(&report);

  std::unique_ptr<const CrashReportDatabase::UploadReport> upload_report;
  EXPECT_EQ(db()->GetReportForUploading(report.uuid, &upload_report),
            CrashReportDatabase::kNoError);

  std::unique_ptr<const CrashReportDatabase::UploadReport> upload_report_2;
  EXPECT_EQ(db()->GetReportForUploading(report.uuid, &upload_report_2),
            CrashReportDatabase::kBusyError);
  EXPECT_FALSE(upload_report_2);

  EXPECT_EQ(db()->RecordUploadComplete(std::move(upload_report), std::string()),
            CrashReportDatabase::kNoError);
}

TEST_F(CrashReportDatabaseTest, UploadAlreadyUploaded) {
  CrashReportDatabase::Report report;
  CreateCrashReport(&report);

  std::unique_ptr<const CrashReportDatabase::UploadReport> upload_report;
  EXPECT_EQ(db()->GetReportForUploading(report.uuid, &upload_report),
            CrashReportDatabase::kNoError);
  EXPECT_EQ(db()->RecordUploadComplete(std::move(upload_report), std::string()),
            CrashReportDatabase::kNoError);

  std::unique_ptr<const CrashReportDatabase::UploadReport> upload_report_2;
  EXPECT_EQ(db()->GetReportForUploading(report.uuid, &upload_report_2),
            CrashReportDatabase::kReportNotFound);
  EXPECT_FALSE(upload_report_2.get());
}

TEST_F(CrashReportDatabaseTest, MoveDatabase) {
  std::unique_ptr<CrashReportDatabase::NewReport> new_report;
  EXPECT_EQ(db()->PrepareNewCrashReport(&new_report),
            CrashReportDatabase::kNoError);
  UUID uuid;
  EXPECT_EQ(db()->FinishedWritingCrashReport(std::move(new_report), &uuid),
            CrashReportDatabase::kNoError);

  RelocateDatabase();

  CrashReportDatabase::Report report;
  EXPECT_EQ(db()->LookUpCrashReport(uuid, &report),
            CrashReportDatabase::kNoError);
  ExpectPreparedCrashReport(report);
}

TEST_F(CrashReportDatabaseTest, ReportRemoved) {
  std::unique_ptr<CrashReportDatabase::NewReport> new_report;
  EXPECT_EQ(db()->PrepareNewCrashReport(&new_report),
            CrashReportDatabase::kNoError);

  UUID uuid;
  EXPECT_EQ(db()->FinishedWritingCrashReport(std::move(new_report), &uuid),
            CrashReportDatabase::kNoError);

  CrashReportDatabase::Report report;
  EXPECT_EQ(db()->LookUpCrashReport(uuid, &report),
            CrashReportDatabase::kNoError);

  EXPECT_TRUE(LoggingRemoveFile(report.file_path));

  EXPECT_EQ(db()->LookUpCrashReport(uuid, &report),
            CrashReportDatabase::kReportNotFound);
}

TEST_F(CrashReportDatabaseTest, DeleteReport) {
  CrashReportDatabase::Report keep_pending;
  CrashReportDatabase::Report delete_pending;
  CrashReportDatabase::Report keep_completed;
  CrashReportDatabase::Report delete_completed;

  CreateCrashReport(&keep_pending);
  CreateCrashReport(&delete_pending);
  CreateCrashReport(&keep_completed);
  CreateCrashReport(&delete_completed);

  EXPECT_TRUE(FileExists(keep_pending.file_path));
  EXPECT_TRUE(FileExists(delete_pending.file_path));
  EXPECT_TRUE(FileExists(keep_completed.file_path));
  EXPECT_TRUE(FileExists(delete_completed.file_path));

  UploadReport(keep_completed.uuid, true, "1");
  UploadReport(delete_completed.uuid, true, "2");

  EXPECT_EQ(db()->LookUpCrashReport(keep_completed.uuid, &keep_completed),
            CrashReportDatabase::kNoError);
  EXPECT_EQ(db()->LookUpCrashReport(delete_completed.uuid, &delete_completed),
            CrashReportDatabase::kNoError);

  EXPECT_TRUE(FileExists(keep_completed.file_path));
  EXPECT_TRUE(FileExists(delete_completed.file_path));

  EXPECT_EQ(db()->DeleteReport(delete_pending.uuid),
            CrashReportDatabase::kNoError);
  EXPECT_FALSE(FileExists(delete_pending.file_path));
  EXPECT_EQ(db()->LookUpCrashReport(delete_pending.uuid, &delete_pending),
            CrashReportDatabase::kReportNotFound);
  EXPECT_EQ(db()->DeleteReport(delete_pending.uuid),
            CrashReportDatabase::kReportNotFound);

  EXPECT_EQ(db()->DeleteReport(delete_completed.uuid),
            CrashReportDatabase::kNoError);
  EXPECT_FALSE(FileExists(delete_completed.file_path));
  EXPECT_EQ(db()->LookUpCrashReport(delete_completed.uuid, &delete_completed),
            CrashReportDatabase::kReportNotFound);
  EXPECT_EQ(db()->DeleteReport(delete_completed.uuid),
            CrashReportDatabase::kReportNotFound);

  EXPECT_EQ(db()->LookUpCrashReport(keep_pending.uuid, &keep_pending),
            CrashReportDatabase::kNoError);
  EXPECT_EQ(db()->LookUpCrashReport(keep_completed.uuid, &keep_completed),
            CrashReportDatabase::kNoError);
}

TEST_F(CrashReportDatabaseTest, DeleteReportEmptyingDatabase) {
  CrashReportDatabase::Report report;
  CreateCrashReport(&report);

  EXPECT_TRUE(FileExists(report.file_path));

  UploadReport(report.uuid, true, "1");

  EXPECT_EQ(db()->LookUpCrashReport(report.uuid, &report),
            CrashReportDatabase::kNoError);

  EXPECT_TRUE(FileExists(report.file_path));

  // This causes an empty database to be written, make sure this is handled.
  EXPECT_EQ(db()->DeleteReport(report.uuid), CrashReportDatabase::kNoError);
  EXPECT_FALSE(FileExists(report.file_path));
}

TEST_F(CrashReportDatabaseTest, ReadEmptyDatabase) {
  CrashReportDatabase::Report report;
  CreateCrashReport(&report);
  EXPECT_EQ(db()->DeleteReport(report.uuid), CrashReportDatabase::kNoError);

  // Deleting and the creating another report causes an empty database to be
  // loaded. Make sure this is handled.

  CrashReportDatabase::Report report2;
  CreateCrashReport(&report2);
}

TEST_F(CrashReportDatabaseTest, RequestUpload) {
  std::vector<CrashReportDatabase::Report> reports(2);
  CreateCrashReport(&reports[0]);
  CreateCrashReport(&reports[1]);

  const UUID& report_0_uuid = reports[0].uuid;
  const UUID& report_1_uuid = reports[1].uuid;

  // Skipped report gets back to pending state after RequestUpload is called.
  EXPECT_EQ(db()->SkipReportUpload(
                report_1_uuid, Metrics::CrashSkippedReason::kUploadsDisabled),
            CrashReportDatabase::kNoError);

  std::vector<CrashReportDatabase::Report> pending_reports;
  CrashReportDatabase::OperationStatus os =
      db()->GetPendingReports(&pending_reports);
  EXPECT_EQ(os, CrashReportDatabase::kNoError);
  ASSERT_EQ(pending_reports.size(), 1u);
  EXPECT_EQ(report_0_uuid, pending_reports[0].uuid);

  pending_reports.clear();
  EXPECT_EQ(RequestUpload(report_1_uuid), CrashReportDatabase::kNoError);
  os = db()->GetPendingReports(&pending_reports);
  EXPECT_EQ(os, CrashReportDatabase::kNoError);
  ASSERT_EQ(pending_reports.size(), 2u);

  // Check individual reports.
  const CrashReportDatabase::Report* explicitly_requested_report;
  const CrashReportDatabase::Report* pending_report;
  if (pending_reports[0].uuid == report_0_uuid) {
    pending_report = &pending_reports[0];
    explicitly_requested_report = &pending_reports[1];
  } else {
    pending_report = &pending_reports[1];
    explicitly_requested_report = &pending_reports[0];
  }

  EXPECT_EQ(pending_report->uuid, report_0_uuid);
  EXPECT_FALSE(pending_report->upload_explicitly_requested);

  EXPECT_EQ(explicitly_requested_report->uuid, report_1_uuid);
  EXPECT_TRUE(explicitly_requested_report->upload_explicitly_requested);

  // Explicitly requested reports will not have upload_explicitly_requested bit
  // after getting skipped.
  EXPECT_EQ(db()->SkipReportUpload(
                report_1_uuid, Metrics::CrashSkippedReason::kUploadsDisabled),
            CrashReportDatabase::kNoError);
  CrashReportDatabase::Report report;
  EXPECT_EQ(db()->LookUpCrashReport(report_1_uuid, &report),
            CrashReportDatabase::kNoError);
  EXPECT_FALSE(report.upload_explicitly_requested);

  // Pending report gets correctly affected after RequestUpload is called.
  pending_reports.clear();
  EXPECT_EQ(RequestUpload(report_0_uuid), CrashReportDatabase::kNoError);
  os = db()->GetPendingReports(&pending_reports);
  EXPECT_EQ(os, CrashReportDatabase::kNoError);
  EXPECT_EQ(pending_reports.size(), 1u);
  EXPECT_EQ(report_0_uuid, pending_reports[0].uuid);
  EXPECT_TRUE(pending_reports[0].upload_explicitly_requested);

  // Already uploaded report cannot be requested for the new upload.
  UploadReport(report_0_uuid, true, "1");
  EXPECT_EQ(RequestUpload(report_0_uuid),
            CrashReportDatabase::kCannotRequestUpload);
}

TEST_F(CrashReportDatabaseTest, Attachments) {
#if defined(OS_MACOSX) || defined(OS_WIN)
  // Attachments aren't supported on Mac and Windows yet.
  DISABLED_TEST();
#else
  std::unique_ptr<CrashReportDatabase::NewReport> new_report;
  ASSERT_EQ(db()->PrepareNewCrashReport(&new_report),
            CrashReportDatabase::kNoError);

  FileWriter* attach_some_file = new_report->AddAttachment("some_file");
  ASSERT_NE(attach_some_file, nullptr);
  static constexpr char test_data[] = "test data";
  attach_some_file->Write(test_data, sizeof(test_data));

  FileWriter* failed_attach = new_report->AddAttachment("not/a valid fi!e");
  EXPECT_EQ(failed_attach, nullptr);

  UUID expect_uuid = new_report->ReportID();
  UUID uuid;
  ASSERT_EQ(db()->FinishedWritingCrashReport(std::move(new_report), &uuid),
            CrashReportDatabase::kNoError);
  EXPECT_EQ(uuid, expect_uuid);

  CrashReportDatabase::Report report;
  EXPECT_EQ(db()->LookUpCrashReport(uuid, &report),
            CrashReportDatabase::kNoError);
  ExpectPreparedCrashReport(report);

  std::vector<CrashReportDatabase::Report> reports;
  EXPECT_EQ(db()->GetPendingReports(&reports), CrashReportDatabase::kNoError);
  ASSERT_EQ(reports.size(), 1u);
  EXPECT_EQ(reports[0].uuid, report.uuid);

  std::unique_ptr<const CrashReportDatabase::UploadReport> upload_report;
  ASSERT_EQ(db()->GetReportForUploading(reports[0].uuid, &upload_report),
            CrashReportDatabase::kNoError);
  std::map<std::string, FileReader*> result_attachments =
      upload_report->GetAttachments();
  EXPECT_EQ(result_attachments.size(), 1u);
  EXPECT_NE(result_attachments.find("some_file"), result_attachments.end());
  char result_buffer[sizeof(test_data)];
  result_attachments["some_file"]->Read(result_buffer, sizeof(result_buffer));
  EXPECT_EQ(memcmp(test_data, result_buffer, sizeof(test_data)), 0);
#endif
}

TEST_F(CrashReportDatabaseTest, OrphanedAttachments) {
#if defined(OS_MACOSX) || defined(OS_WIN)
  // Attachments aren't supported on Mac and Windows yet.
  DISABLED_TEST();
#else
  // TODO: This is using paths that are specific to the generic implementation
  // and will need to be generalized for Mac and Windows.
  std::unique_ptr<CrashReportDatabase::NewReport> new_report;
  ASSERT_EQ(db()->PrepareNewCrashReport(&new_report),
            CrashReportDatabase::kNoError);

  FileWriter* file1 = new_report->AddAttachment("file1");
  ASSERT_NE(file1, nullptr);
  FileWriter* file2 = new_report->AddAttachment("file2");
  ASSERT_NE(file2, nullptr);

  UUID expect_uuid = new_report->ReportID();
  UUID uuid;
  ASSERT_EQ(db()->FinishedWritingCrashReport(std::move(new_report), &uuid),
            CrashReportDatabase::kNoError);
  EXPECT_EQ(uuid, expect_uuid);

  CrashReportDatabase::Report report;
  ASSERT_EQ(db()->LookUpCrashReport(uuid, &report),
            CrashReportDatabase::kNoError);

  ASSERT_TRUE(LoggingRemoveFile(report.file_path));

  ASSERT_TRUE(LoggingRemoveFile(base::FilePath(
      report.file_path.RemoveFinalExtension().value() + ".meta")));

  ASSERT_EQ(db()->LookUpCrashReport(uuid, &report),
            CrashReportDatabase::kReportNotFound);

  base::FilePath report_attachments_dir(
      path().Append("attachments").Append(uuid.ToString()));
  base::FilePath file_path1(report_attachments_dir.Append("file1"));
  base::FilePath file_path2(report_attachments_dir.Append("file2"));
  EXPECT_TRUE(FileExists(file_path1));
  EXPECT_TRUE(FileExists(file_path1));

  EXPECT_EQ(db()->CleanDatabase(0), 0);

  EXPECT_FALSE(FileExists(file_path1));
  EXPECT_FALSE(FileExists(file_path2));
  EXPECT_FALSE(FileExists(report_attachments_dir));
#endif
}

// This test uses knowledge of the database format to break it, so it only
// applies to the unfified database implementation.
#if !defined(OS_MACOSX) && !defined(OS_WIN)
TEST_F(CrashReportDatabaseTest, CleanBrokenDatabase) {
  // Remove report files if metadata goes missing.
  CrashReportDatabase::Report report;
  ASSERT_NO_FATAL_FAILURE(CreateCrashReport(&report));

  const base::FilePath metadata(
      report.file_path.RemoveFinalExtension().value() +
      FILE_PATH_LITERAL(".meta"));
  ASSERT_TRUE(PathExists(report.file_path));
  ASSERT_TRUE(PathExists(metadata));

  ASSERT_TRUE(LoggingRemoveFile(metadata));
  EXPECT_EQ(db()->CleanDatabase(0), 1);

  EXPECT_FALSE(PathExists(report.file_path));
  EXPECT_FALSE(PathExists(metadata));

  // Remove metadata files if reports go missing.
  ASSERT_NO_FATAL_FAILURE(CreateCrashReport(&report));
  const base::FilePath metadata2(
      report.file_path.RemoveFinalExtension().value() +
      FILE_PATH_LITERAL(".meta"));
  ASSERT_TRUE(PathExists(report.file_path));
  ASSERT_TRUE(PathExists(metadata2));

  ASSERT_TRUE(LoggingRemoveFile(report.file_path));
  EXPECT_EQ(db()->CleanDatabase(0), 1);

  EXPECT_FALSE(PathExists(report.file_path));
  EXPECT_FALSE(PathExists(metadata2));

  // Remove stale new files.
  std::unique_ptr<CrashReportDatabase::NewReport> new_report;
  EXPECT_EQ(db()->PrepareNewCrashReport(&new_report),
            CrashReportDatabase::kNoError);
  new_report->Writer()->Close();
  EXPECT_EQ(db()->CleanDatabase(0), 1);

  // Remove stale lock files and their associated reports.
  ASSERT_NO_FATAL_FAILURE(CreateCrashReport(&report));
  const base::FilePath metadata3(
      report.file_path.RemoveFinalExtension().value() +
      FILE_PATH_LITERAL(".meta"));
  ASSERT_TRUE(PathExists(report.file_path));
  ASSERT_TRUE(PathExists(metadata3));

  const base::FilePath lockpath(
      report.file_path.RemoveFinalExtension().value() +
      FILE_PATH_LITERAL(".lock"));
  ScopedFileHandle handle(LoggingOpenFileForWrite(
      lockpath, FileWriteMode::kCreateOrFail, FilePermissions::kOwnerOnly));
  ASSERT_TRUE(handle.is_valid());

  time_t expired_timestamp = time(nullptr) - 60 * 60 * 24 * 3;

  ASSERT_TRUE(LoggingWriteFile(
      handle.get(), &expired_timestamp, sizeof(expired_timestamp)));
  ASSERT_TRUE(LoggingCloseFile(handle.get()));
  ignore_result(handle.release());

  EXPECT_EQ(db()->CleanDatabase(0), 1);

  EXPECT_FALSE(PathExists(report.file_path));
  EXPECT_FALSE(PathExists(metadata3));
}
#endif  // !OS_MACOSX && !OS_WIN

}  // namespace
}  // namespace test
}  // namespace crashpad
