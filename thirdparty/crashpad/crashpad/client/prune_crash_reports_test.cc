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

#include "client/prune_crash_reports.h"

#include <stddef.h>
#include <stdlib.h>

#include <algorithm>
#include <random>
#include <string>
#include <vector>

#include "base/numerics/safe_conversions.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "test/scoped_temp_dir.h"
#include "util/file/file_io.h"

namespace crashpad {
namespace test {
namespace {

class MockDatabase : public CrashReportDatabase {
 public:
  // CrashReportDatabase:
  MOCK_METHOD0(GetSettings, Settings*());
  MOCK_METHOD1(PrepareNewCrashReport,
               OperationStatus(std::unique_ptr<NewReport>*));
  MOCK_METHOD2(LookUpCrashReport, OperationStatus(const UUID&, Report*));
  MOCK_METHOD1(GetPendingReports, OperationStatus(std::vector<Report>*));
  MOCK_METHOD1(GetCompletedReports, OperationStatus(std::vector<Report>*));
  MOCK_METHOD3(GetReportForUploading,
               OperationStatus(const UUID&,
                               std::unique_ptr<const UploadReport>*,
                               bool report_metrics));
  MOCK_METHOD3(RecordUploadAttempt,
               OperationStatus(UploadReport*, bool, const std::string&));
  MOCK_METHOD2(SkipReportUpload,
               OperationStatus(const UUID&, Metrics::CrashSkippedReason));
  MOCK_METHOD1(DeleteReport, OperationStatus(const UUID&));
  MOCK_METHOD1(RequestUpload, OperationStatus(const UUID&));

  // gmock doesn't support mocking methods with non-copyable types such as
  // unique_ptr.
  OperationStatus FinishedWritingCrashReport(std::unique_ptr<NewReport> report,
                                             UUID* uuid) override {
    return kNoError;
  }
};

time_t NDaysAgo(int num_days) {
  return time(nullptr) - (num_days * 60 * 60 * 24);
}

TEST(PruneCrashReports, AgeCondition) {
  CrashReportDatabase::Report report_80_days;
  report_80_days.creation_time = NDaysAgo(80);

  CrashReportDatabase::Report report_10_days;
  report_10_days.creation_time = NDaysAgo(10);

  CrashReportDatabase::Report report_30_days;
  report_30_days.creation_time = NDaysAgo(30);

  AgePruneCondition condition(30);
  EXPECT_TRUE(condition.ShouldPruneReport(report_80_days));
  EXPECT_FALSE(condition.ShouldPruneReport(report_10_days));
  EXPECT_FALSE(condition.ShouldPruneReport(report_30_days));
}

TEST(PruneCrashReports, SizeCondition) {
  ScopedTempDir temp_dir;

  CrashReportDatabase::Report report_1k;
  report_1k.file_path = temp_dir.path().Append(FILE_PATH_LITERAL("file1024"));
  CrashReportDatabase::Report report_3k;
  report_3k.file_path = temp_dir.path().Append(FILE_PATH_LITERAL("file3072"));

  {
    ScopedFileHandle scoped_file_1k(
        LoggingOpenFileForWrite(report_1k.file_path,
                                FileWriteMode::kCreateOrFail,
                                FilePermissions::kOwnerOnly));
    ASSERT_TRUE(scoped_file_1k.is_valid());

    std::string string;
    for (int i = 0; i < 128; ++i)
      string.push_back(static_cast<char>(i));

    for (size_t i = 0; i < 1024; i += string.size()) {
      ASSERT_TRUE(LoggingWriteFile(scoped_file_1k.get(),
                                   string.c_str(), string.length()));
    }

    ScopedFileHandle scoped_file_3k(
        LoggingOpenFileForWrite(report_3k.file_path,
                                FileWriteMode::kCreateOrFail,
                                FilePermissions::kOwnerOnly));
    ASSERT_TRUE(scoped_file_3k.is_valid());

    for (size_t i = 0; i < 3072; i += string.size()) {
      ASSERT_TRUE(LoggingWriteFile(scoped_file_3k.get(),
                                   string.c_str(), string.length()));
    }
  }

  {
    DatabaseSizePruneCondition condition(1);
    EXPECT_FALSE(condition.ShouldPruneReport(report_1k));
    EXPECT_TRUE(condition.ShouldPruneReport(report_1k));
  }

  {
    DatabaseSizePruneCondition condition(1);
    EXPECT_TRUE(condition.ShouldPruneReport(report_3k));
  }

  {
    DatabaseSizePruneCondition condition(6);
    EXPECT_FALSE(condition.ShouldPruneReport(report_3k));
    EXPECT_FALSE(condition.ShouldPruneReport(report_3k));
    EXPECT_TRUE(condition.ShouldPruneReport(report_1k));
  }
}

class StaticCondition final : public PruneCondition {
 public:
  explicit StaticCondition(bool value) : value_(value), did_execute_(false) {}
  ~StaticCondition() {}

  bool ShouldPruneReport(const CrashReportDatabase::Report& report) override {
    did_execute_ = true;
    return value_;
  }

  bool did_execute() const { return did_execute_; }

 private:
  const bool value_;
  bool did_execute_;

  DISALLOW_COPY_AND_ASSIGN(StaticCondition);
};

TEST(PruneCrashReports, BinaryCondition) {
  static constexpr struct {
    const char* name;
    BinaryPruneCondition::Operator op;
    bool lhs_value;
    bool rhs_value;
    bool cond_result;
    bool lhs_executed;
    bool rhs_executed;
  } kTests[] = {
    {"false && false",
     BinaryPruneCondition::AND, false, false,
     false, true, false},
    {"false && true",
     BinaryPruneCondition::AND, false, true,
     false, true, false},
    {"true && false",
     BinaryPruneCondition::AND, true, false,
     false, true, true},
    {"true && true",
     BinaryPruneCondition::AND, true, true,
     true, true, true},
    {"false || false",
     BinaryPruneCondition::OR, false, false,
     false, true, true},
    {"false || true",
     BinaryPruneCondition::OR, false, true,
     true, true, true},
    {"true || false",
     BinaryPruneCondition::OR, true, false,
     true, true, false},
    {"true || true",
     BinaryPruneCondition::OR, true, true,
     true, true, false},
  };
  for (const auto& test : kTests) {
    SCOPED_TRACE(test.name);
    auto lhs = new StaticCondition(test.lhs_value);
    auto rhs = new StaticCondition(test.rhs_value);
    BinaryPruneCondition condition(test.op, lhs, rhs);
    CrashReportDatabase::Report report;
    EXPECT_EQ(condition.ShouldPruneReport(report), test.cond_result);
    EXPECT_EQ(lhs->did_execute(), test.lhs_executed);
    EXPECT_EQ(rhs->did_execute(), test.rhs_executed);
  }
}

MATCHER_P(TestUUID, data_1, "") {
  return arg.data_1 == data_1;
}

TEST(PruneCrashReports, PruneOrder) {
  using ::testing::_;
  using ::testing::DoAll;
  using ::testing::Return;
  using ::testing::SetArgPointee;

  std::vector<CrashReportDatabase::Report> reports;
  for (int i = 0; i < 10; ++i) {
    CrashReportDatabase::Report temp;
    temp.uuid.data_1 = i;
    temp.creation_time = NDaysAgo(i * 10);
    reports.push_back(temp);
  }
  std::mt19937 urng(std::random_device{}());
  std::shuffle(reports.begin(), reports.end(), urng);
  std::vector<CrashReportDatabase::Report> pending_reports(
      reports.begin(), reports.begin() + 5);
  std::vector<CrashReportDatabase::Report> completed_reports(
      reports.begin() + 5, reports.end());

  MockDatabase db;
  EXPECT_CALL(db, GetPendingReports(_)).WillOnce(DoAll(
      SetArgPointee<0>(pending_reports),
      Return(CrashReportDatabase::kNoError)));
  EXPECT_CALL(db, GetCompletedReports(_)).WillOnce(DoAll(
      SetArgPointee<0>(completed_reports),
      Return(CrashReportDatabase::kNoError)));
  for (size_t i = 0; i < reports.size(); ++i) {
    EXPECT_CALL(db, DeleteReport(TestUUID(i)))
        .WillOnce(Return(CrashReportDatabase::kNoError));
  }

  StaticCondition delete_all(true);
  PruneCrashReportDatabase(&db, &delete_all);
}

}  // namespace
}  // namespace test
}  // namespace crashpad
