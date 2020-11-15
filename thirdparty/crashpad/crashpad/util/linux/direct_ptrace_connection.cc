// Copyright 2017 The Crashpad Authors. All rights reserved.
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

#include "util/linux/direct_ptrace_connection.h"

#include <stdio.h>

#include <utility>

#include "base/logging.h"
#include "base/strings/string_number_conversions.h"
#include "util/file/directory_reader.h"
#include "util/file/file_io.h"
#include "util/misc/as_underlying_type.h"

namespace crashpad {

DirectPtraceConnection::DirectPtraceConnection()
    : PtraceConnection(),
      attachments_(),
      memory_(),
      pid_(-1),
      ptracer_(/* can_log= */ true),
      initialized_() {}

DirectPtraceConnection::~DirectPtraceConnection() {}

bool DirectPtraceConnection::Initialize(pid_t pid) {
  INITIALIZATION_STATE_SET_INITIALIZING(initialized_);

  if (!Attach(pid) || !ptracer_.Initialize(pid)) {
    return false;
  }
  pid_ = pid;

  if (!memory_.Initialize(pid)) {
    return false;
  }

  INITIALIZATION_STATE_SET_VALID(initialized_);
  return true;
}

pid_t DirectPtraceConnection::GetProcessID() {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return pid_;
}

bool DirectPtraceConnection::Attach(pid_t tid) {
  std::unique_ptr<ScopedPtraceAttach> attach(new ScopedPtraceAttach);
  if (!attach->ResetAttach(tid)) {
    return false;
  }
  attachments_.push_back(std::move(attach));
  return true;
}

bool DirectPtraceConnection::Is64Bit() {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return ptracer_.Is64Bit();
}

bool DirectPtraceConnection::GetThreadInfo(pid_t tid, ThreadInfo* info) {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return ptracer_.GetThreadInfo(tid, info);
}

bool DirectPtraceConnection::ReadFileContents(const base::FilePath& path,
                                              std::string* contents) {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return LoggingReadEntireFile(path, contents);
}

ProcessMemory* DirectPtraceConnection::Memory() {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return &memory_;
}

bool DirectPtraceConnection::Threads(std::vector<pid_t>* threads) {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  DCHECK(threads->empty());

  char path[32];
  snprintf(path, arraysize(path), "/proc/%d/task", pid_);
  DirectoryReader reader;
  if (!reader.Open(base::FilePath(path))) {
    return false;
  }

  std::vector<pid_t> local_threads;
  base::FilePath tid_str;
  DirectoryReader::Result result;
  while ((result = reader.NextFile(&tid_str)) ==
         DirectoryReader::Result::kSuccess) {
    pid_t tid;
    if (!base::StringToInt(tid_str.value(), &tid)) {
      LOG(ERROR) << "format error";
      continue;
    }

    local_threads.push_back(tid);
  }
  DCHECK_EQ(AsUnderlyingType(result),
            AsUnderlyingType(DirectoryReader::Result::kNoMoreFiles));

  threads->swap(local_threads);
  return true;
}

}  // namespace crashpad
