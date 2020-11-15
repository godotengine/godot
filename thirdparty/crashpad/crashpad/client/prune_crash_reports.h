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

#ifndef CRASHPAD_CLIENT_PRUNE_CRASH_REPORTS_H_
#define CRASHPAD_CLIENT_PRUNE_CRASH_REPORTS_H_

#include <sys/types.h>
#include <time.h>

#include <memory>

#include "base/macros.h"
#include "client/crash_report_database.h"

namespace crashpad {

class PruneCondition;

//! \brief Deletes crash reports from \a database that match \a condition.
//!
//! This function can be used to remove old or large reports from the database.
//! The \a condition will be evaluated against each report in the \a database,
//! sorted in descending order by CrashReportDatabase::Report::creation_time.
//! This guarantee allows conditions to be stateful.
//!
//! \param[in] database The database from which crash reports will be deleted.
//! \param[in] condition The condition against which all reports in the database
//!     will be evaluated.
void PruneCrashReportDatabase(CrashReportDatabase* database,
                              PruneCondition* condition);

std::unique_ptr<PruneCondition> GetDefaultDatabasePruneCondition();

//! \brief An abstract base class for evaluating crash reports for deletion.
//!
//! When passed to PruneCrashReportDatabase(), each crash report in the
//! database will be evaluated according to ShouldPruneReport(). The reports
//! are evaluated serially in descending sort order by
//! CrashReportDatabase::Report::creation_time.
class PruneCondition {
 public:
  //! \brief Returns a sensible default condition for removing obsolete crash
  //!     reports.
  //!
  //! The default is to keep reports for one year or a maximum database size
  //! of 128 MB.
  //!
  //! \return A PruneCondition for use with PruneCrashReportDatabase().
  static std::unique_ptr<PruneCondition> GetDefault();

  virtual ~PruneCondition() {}

  //! \brief Evaluates a crash report for deletion.
  //!
  //! \param[in] report The crash report to evaluate.
  //!
  //! \return `true` if the crash report should be deleted, `false` if it
  //!     should be kept.
  virtual bool ShouldPruneReport(const CrashReportDatabase::Report& report) = 0;
};

//! \brief A PruneCondition that deletes reports older than the specified number
//!     days.
class AgePruneCondition final : public PruneCondition {
 public:
  //! \brief Creates a PruneCondition based on Report::creation_time.
  //!
  //! \param[in] max_age_in_days Reports created more than this many days ago
  //!     will be deleted.
  explicit AgePruneCondition(int max_age_in_days);
  ~AgePruneCondition();

  bool ShouldPruneReport(const CrashReportDatabase::Report& report) override;

 private:
  const time_t oldest_report_time_;

  DISALLOW_COPY_AND_ASSIGN(AgePruneCondition);
};

//! \brief A PruneCondition that deletes older reports to keep the total
//!     Crashpad database size under the specified limit.
class DatabaseSizePruneCondition final : public PruneCondition {
 public:
  //! \brief Creates a PruneCondition that will keep newer reports, until the
  //!     sum of the size of all reports is not smaller than \a max_size_in_kb.
  //!     After the limit is reached, older reports will be pruned.
  //!
  //! \param[in] max_size_in_kb The maximum number of kilobytes that all crash
  //!     reports should consume.
  explicit DatabaseSizePruneCondition(size_t max_size_in_kb);
  ~DatabaseSizePruneCondition();

  bool ShouldPruneReport(const CrashReportDatabase::Report& report) override;

 private:
  const size_t max_size_in_kb_;
  size_t measured_size_in_kb_;

  DISALLOW_COPY_AND_ASSIGN(DatabaseSizePruneCondition);
};

//! \brief A PruneCondition that conjoins two other PruneConditions.
class BinaryPruneCondition final : public PruneCondition {
 public:
  enum Operator {
    AND,
    OR,
  };

  //! \brief Evaluates two sub-conditions according to the specified logical
  //!     operator.
  //!
  //! This implements left-to-right evaluation. For Operator::AND, this means
  //! if the \a lhs is `false`, the \a rhs will not be consulted. Similarly,
  //! with Operator::OR, if the \a lhs is `true`, the \a rhs will not be
  //! consulted.
  //!
  //! \param[in] op The logical operator to apply on \a lhs and \a rhs.
  //! \param[in] lhs The left-hand side of \a op. This class takes ownership.
  //! \param[in] rhs The right-hand side of \a op. This class takes ownership.
  BinaryPruneCondition(Operator op, PruneCondition* lhs, PruneCondition* rhs);
  ~BinaryPruneCondition();

  bool ShouldPruneReport(const CrashReportDatabase::Report& report) override;

 private:
  const Operator op_;
  std::unique_ptr<PruneCondition> lhs_;
  std::unique_ptr<PruneCondition> rhs_;

  DISALLOW_COPY_AND_ASSIGN(BinaryPruneCondition);
};

}  // namespace crashpad

#endif  // CRASHPAD_CLIENT_PRUNE_CRASH_REPORTS_H_
