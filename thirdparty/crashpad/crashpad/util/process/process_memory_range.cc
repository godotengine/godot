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

#include "util/process/process_memory_range.h"

#include <algorithm>
#include <limits>

#include "base/logging.h"

namespace crashpad {

ProcessMemoryRange::ProcessMemoryRange()
    : memory_(nullptr), range_(), initialized_() {}

ProcessMemoryRange::~ProcessMemoryRange() {}

bool ProcessMemoryRange::Initialize(const ProcessMemory* memory,
                                    bool is_64_bit,
                                    VMAddress base,
                                    VMSize size) {
  INITIALIZATION_STATE_SET_INITIALIZING(initialized_);
  memory_ = memory;
  range_.SetRange(is_64_bit, base, size);
  if (!range_.IsValid()) {
    LOG(ERROR) << "invalid range";
    return false;
  }
  INITIALIZATION_STATE_SET_VALID(initialized_);
  return true;
}

bool ProcessMemoryRange::Initialize(const ProcessMemory* memory,
                                    bool is_64_bit) {
  VMSize max = is_64_bit ? std::numeric_limits<uint64_t>::max()
                         : std::numeric_limits<uint32_t>::max();
  return Initialize(memory, is_64_bit, 0, max);
}

bool ProcessMemoryRange::Initialize(const ProcessMemoryRange& other) {
  return Initialize(other.memory_,
                    other.range_.Is64Bit(),
                    other.range_.Base(),
                    other.range_.Size());
}

bool ProcessMemoryRange::RestrictRange(VMAddress base, VMSize size) {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  CheckedVMAddressRange new_range(range_.Is64Bit(), base, size);
  if (!new_range.IsValid() || !range_.ContainsRange(new_range)) {
    LOG(ERROR) << "invalid range";
    return false;
  }
  range_ = new_range;
  return true;
}

bool ProcessMemoryRange::Read(VMAddress address,
                              size_t size,
                              void* buffer) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  CheckedVMAddressRange read_range(range_.Is64Bit(), address, size);
  if (!read_range.IsValid() || !range_.ContainsRange(read_range)) {
    LOG(ERROR) << "read out of range";
    return false;
  }
  return memory_->Read(address, size, buffer);
}

bool ProcessMemoryRange::ReadCStringSizeLimited(VMAddress address,
                                                size_t size,
                                                std::string* string) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  if (!range_.ContainsValue(address)) {
    LOG(ERROR) << "read out of range";
    return false;
  }
  size = std::min(static_cast<VMSize>(size), range_.End() - address);
  return memory_->ReadCStringSizeLimited(address, size, string);
}

}  // namespace crashpad
