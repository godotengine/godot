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

#include "test/linux/fake_ptrace_connection.h"

#include <utility>

#include "build/build_config.h"
#include "gtest/gtest.h"
#include "util/file/file_io.h"

namespace crashpad {
namespace test {

FakePtraceConnection::FakePtraceConnection()
    : PtraceConnection(),
      attachments_(),
      pid_(-1),
      is_64_bit_(false),
      initialized_() {}

FakePtraceConnection::~FakePtraceConnection() {}

bool FakePtraceConnection::Initialize(pid_t pid) {
  INITIALIZATION_STATE_SET_INITIALIZING(initialized_);

  if (!Attach(pid)) {
    return false;
  }
  pid_ = pid;

#if defined(ARCH_CPU_64_BITS)
  is_64_bit_ = true;
#else
  is_64_bit_ = false;
#endif

  INITIALIZATION_STATE_SET_VALID(initialized_);
  return true;
}

pid_t FakePtraceConnection::GetProcessID() {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return pid_;
}

bool FakePtraceConnection::Attach(pid_t tid) {
  bool inserted = attachments_.insert(tid).second;
  EXPECT_TRUE(inserted);
  return inserted;
}

bool FakePtraceConnection::Is64Bit() {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return is_64_bit_;
}

bool FakePtraceConnection::GetThreadInfo(pid_t tid, ThreadInfo* info) {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  bool attached = attachments_.find(tid) != attachments_.end();
  EXPECT_TRUE(attached);
  return attached;
}

bool FakePtraceConnection::ReadFileContents(const base::FilePath& path,
                                            std::string* contents) {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return LoggingReadEntireFile(path, contents);
}

ProcessMemory* FakePtraceConnection::Memory() {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  if (!memory_) {
    auto mem = std::make_unique<ProcessMemoryLinux>();
    if (mem->Initialize(pid_)) {
      memory_ = std::move(mem);
    } else {
      ADD_FAILURE();
    }
  }
  return memory_.get();
}

bool FakePtraceConnection::Threads(std::vector<pid_t>* threads) {
  // TODO(jperaza): Implement this if/when it's needed.
  NOTREACHED();
  return false;
}

}  // namespace test
}  // namespace crashpad
