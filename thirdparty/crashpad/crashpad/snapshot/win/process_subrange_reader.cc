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

#include "snapshot/win/process_subrange_reader.h"

#include "base/logging.h"
#include "snapshot/win/process_reader_win.h"

namespace crashpad {

ProcessSubrangeReader::ProcessSubrangeReader()
    : name_(),
      range_(),
      process_reader_(nullptr) {
}

ProcessSubrangeReader::~ProcessSubrangeReader() {
}

bool ProcessSubrangeReader::Initialize(ProcessReaderWin* process_reader,
                                       WinVMAddress base,
                                       WinVMSize size,
                                       const std::string& name) {
  INITIALIZATION_STATE_SET_INITIALIZING(initialized_);

  if (!InitializeInternal(process_reader, base, size, name)) {
    return false;
  }

  INITIALIZATION_STATE_SET_VALID(initialized_);
  return true;
}

bool ProcessSubrangeReader::InitializeSubrange(
    const ProcessSubrangeReader& that,
    WinVMAddress base,
    WinVMSize size,
    const std::string& sub_name) {
  INITIALIZATION_STATE_SET_INITIALIZING(initialized_);
  INITIALIZATION_STATE_DCHECK_VALID(that.initialized_);

  if (!InitializeInternal(
          that.process_reader_, base, size, that.name_ + " " + sub_name)) {
    return false;
  }

  if (!that.range_.ContainsRange(range_)) {
    LOG(WARNING) << "range " << range_.AsString() << " outside of  range "
                 << that.range_.AsString() << " for " << name_;
    return false;
  }

  INITIALIZATION_STATE_SET_VALID(initialized_);
  return true;
}

bool ProcessSubrangeReader::ReadMemory(WinVMAddress address,
                                       WinVMSize size,
                                       void* into) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  CheckedWinAddressRange read_range(process_reader_->Is64Bit(), address, size);
  if (!read_range.IsValid()) {
    LOG(WARNING) << "invalid read range " << read_range.AsString();
    return false;
  }

  if (!range_.ContainsRange(read_range)) {
    LOG(WARNING) << "attempt to read outside of " << name_ << " range "
                 << range_.AsString() << " at range " << read_range.AsString();
    return false;
  }

  return process_reader_->ReadMemory(address, size, into);
}

bool ProcessSubrangeReader::InitializeInternal(ProcessReaderWin* process_reader,
                                               WinVMAddress base,
                                               WinVMSize size,
                                               const std::string& name) {
  range_.SetRange(process_reader->Is64Bit(), base, size);
  if (!range_.IsValid()) {
    LOG(WARNING) << "invalid range " << range_.AsString() << " for " << name;
    return false;
  }

  name_ = name;
  process_reader_ = process_reader;

  return true;
}

}  // namespace crashpad
