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

#include "test/win/win_multiprocess.h"

#include <shellapi.h>

#include "base/logging.h"
#include "base/scoped_generic.h"
#include "base/strings/stringprintf.h"
#include "base/strings/utf_string_conversions.h"
#include "util/stdlib/string_number_conversion.h"
#include "util/string/split_string.h"

namespace crashpad {
namespace test {

WinMultiprocess::WinMultiprocess()
    : exit_code_(EXIT_SUCCESS),
      child_handles_(nullptr),
      child_process_helper_(nullptr) {}

WinMultiprocess::~WinMultiprocess() {
}

void WinMultiprocess::SetExpectedChildExitCode(unsigned int exit_code) {
  exit_code_ = exit_code;
}

FileHandle WinMultiprocess::ReadPipeHandle() const {
  if (child_handles_)
    return child_handles_->read.get();
  CHECK(child_process_helper_);
  return child_process_helper_->ReadPipeHandleForwarder();
}

FileHandle WinMultiprocess::WritePipeHandle() const {
  if (child_handles_)
    return child_handles_->write.get();
  CHECK(child_process_helper_);
  return child_process_helper_->WritePipeHandleForwarder();
}

void WinMultiprocess::CloseReadPipe() {
  if (child_handles_) {
    child_handles_->read.reset();
  } else {
    CHECK(child_process_helper_);
    child_process_helper_->CloseReadPipeForwarder();
  }
}

void WinMultiprocess::CloseWritePipe() {
  if (child_handles_) {
    child_handles_->write.reset();
  } else {
    CHECK(child_process_helper_);
    child_process_helper_->CloseWritePipeForwarder();
  }
}

HANDLE WinMultiprocess::ChildProcess() const {
  CHECK(child_handles_);
  return child_handles_->process.get();
}

}  // namespace test
}  // namespace crashpad
