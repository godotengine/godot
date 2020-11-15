// Copyright 2016 The Crashpad Authors. All rights reserved.
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

#include "handler/prune_crash_reports_thread.h"

#include <utility>

#include "client/prune_crash_reports.h"

namespace crashpad {

PruneCrashReportThread::PruneCrashReportThread(
    CrashReportDatabase* database,
    std::unique_ptr<PruneCondition> condition)
    : thread_(60 * 60 * 24, this),
      condition_(std::move(condition)),
      database_(database) {}

PruneCrashReportThread::~PruneCrashReportThread() {}

void PruneCrashReportThread::Start() {
  thread_.Start(60 * 10);
}

void PruneCrashReportThread::Stop() {
  thread_.Stop();
}

void PruneCrashReportThread::DoWork(const WorkerThread* thread) {
  database_->CleanDatabase(60 * 60 * 24 * 3);
  PruneCrashReportDatabase(database_, condition_.get());
}

}  // namespace crashpad
