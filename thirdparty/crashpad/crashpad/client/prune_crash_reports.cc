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

#include <sys/stat.h>

#include <algorithm>
#include <vector>

#include "base/logging.h"
#include "build/build_config.h"

namespace crashpad {

void PruneCrashReportDatabase(CrashReportDatabase* database,
                              PruneCondition* condition) {
  std::vector<CrashReportDatabase::Report> all_reports;
  CrashReportDatabase::OperationStatus status;

  status = database->GetPendingReports(&all_reports);
  if (status != CrashReportDatabase::kNoError) {
    LOG(ERROR) << "PruneCrashReportDatabase: Failed to get pending reports";
    return;
  }

  std::vector<CrashReportDatabase::Report> completed_reports;
  status = database->GetCompletedReports(&completed_reports);
  if (status != CrashReportDatabase::kNoError) {
    LOG(ERROR) << "PruneCrashReportDatabase: Failed to get completed reports";
    return;
  }
  all_reports.insert(all_reports.end(), completed_reports.begin(),
                     completed_reports.end());

  std::sort(all_reports.begin(), all_reports.end(),
      [](const CrashReportDatabase::Report& lhs,
         const CrashReportDatabase::Report& rhs) {
        return lhs.creation_time > rhs.creation_time;
      });

  for (const auto& report : all_reports) {
    if (condition->ShouldPruneReport(report)) {
      status = database->DeleteReport(report.uuid);
      if (status != CrashReportDatabase::kNoError) {
        LOG(ERROR) << "Database Pruning: Failed to remove report "
                   << report.uuid.ToString();
      }
    }
  }

  // TODO(rsesek): For databases that do not use a directory structure, it is
  // possible for the metadata sidecar to become corrupted and thus leave
  // orphaned crash report files on-disk. https://crashpad.chromium.org/bug/66
}

// static
std::unique_ptr<PruneCondition> PruneCondition::GetDefault() {
  // DatabaseSizePruneCondition must be the LHS so that it is always evaluated,
  // due to the short-circuting behavior of BinaryPruneCondition.
  return std::make_unique<BinaryPruneCondition>(
      BinaryPruneCondition::OR,
      new DatabaseSizePruneCondition(1024 * 128),
      new AgePruneCondition(365));
}

static const time_t kSecondsInDay = 60 * 60 * 24;

AgePruneCondition::AgePruneCondition(int max_age_in_days)
    : oldest_report_time_(
        ((time(nullptr) - (max_age_in_days * kSecondsInDay))
             / kSecondsInDay) * kSecondsInDay) {}

AgePruneCondition::~AgePruneCondition() {}

bool AgePruneCondition::ShouldPruneReport(
    const CrashReportDatabase::Report& report) {
  return report.creation_time < oldest_report_time_;
}

DatabaseSizePruneCondition::DatabaseSizePruneCondition(size_t max_size_in_kb)
    : max_size_in_kb_(max_size_in_kb), measured_size_in_kb_(0) {}

DatabaseSizePruneCondition::~DatabaseSizePruneCondition() {}

bool DatabaseSizePruneCondition::ShouldPruneReport(
    const CrashReportDatabase::Report& report) {
#if defined(OS_POSIX)
  struct stat statbuf;
  if (stat(report.file_path.value().c_str(), &statbuf) == 0) {
#elif defined(OS_WIN)
  struct _stati64 statbuf;
  if (_wstat64(report.file_path.value().c_str(), &statbuf) == 0) {
#else
#error "Not implemented"
#endif
    // Round up fractional KB to the next 1-KB boundary.
    measured_size_in_kb_ +=
        static_cast<size_t>((statbuf.st_size + 1023) / 1024);
  }
  return measured_size_in_kb_ > max_size_in_kb_;
}

BinaryPruneCondition::BinaryPruneCondition(
    Operator op, PruneCondition* lhs, PruneCondition* rhs)
    : op_(op), lhs_(lhs), rhs_(rhs) {}

BinaryPruneCondition::~BinaryPruneCondition() {}

bool BinaryPruneCondition::ShouldPruneReport(
    const CrashReportDatabase::Report& report) {
  switch (op_) {
    case AND:
      return lhs_->ShouldPruneReport(report) && rhs_->ShouldPruneReport(report);
    case OR:
      return lhs_->ShouldPruneReport(report) || rhs_->ShouldPruneReport(report);
    default:
      NOTREACHED();
      return false;
  }
}

}  // namespace crashpad
