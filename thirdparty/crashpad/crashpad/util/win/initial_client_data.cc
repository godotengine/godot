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

#include "util/win/initial_client_data.h"

#include <vector>

#include "base/logging.h"
#include "base/strings/stringprintf.h"
#include "util/stdlib/string_number_conversion.h"
#include "util/string/split_string.h"
#include "util/win/handle.h"

namespace crashpad {

namespace {

bool HandleFromString(const std::string& str, HANDLE* handle) {
  unsigned int handle_uint;
  if (!StringToNumber(str, &handle_uint) ||
      (*handle = IntToHandle(handle_uint)) == INVALID_HANDLE_VALUE) {
    LOG(ERROR) << "could not convert '" << str << "' to HANDLE";
    return false;
  }
  return true;
}

bool AddressFromString(const std::string& str, WinVMAddress* address) {
  if (!StringToNumber(str, address)) {
    LOG(ERROR) << "could not convert '" << str << "' to WinVMAddress";
    return false;
  }
  return true;
}

}  // namespace

InitialClientData::InitialClientData()
    : crash_exception_information_(0),
      non_crash_exception_information_(0),
      debug_critical_section_address_(0),
      request_crash_dump_(nullptr),
      request_non_crash_dump_(nullptr),
      non_crash_dump_completed_(nullptr),
      first_pipe_instance_(INVALID_HANDLE_VALUE),
      client_process_(nullptr),
      is_valid_(false) {}

InitialClientData::InitialClientData(
    HANDLE request_crash_dump,
    HANDLE request_non_crash_dump,
    HANDLE non_crash_dump_completed,
    HANDLE first_pipe_instance,
    HANDLE client_process,
    WinVMAddress crash_exception_information,
    WinVMAddress non_crash_exception_information,
    WinVMAddress debug_critical_section_address)
    : crash_exception_information_(crash_exception_information),
      non_crash_exception_information_(non_crash_exception_information),
      debug_critical_section_address_(debug_critical_section_address),
      request_crash_dump_(request_crash_dump),
      request_non_crash_dump_(request_non_crash_dump),
      non_crash_dump_completed_(non_crash_dump_completed),
      first_pipe_instance_(first_pipe_instance),
      client_process_(client_process),
      is_valid_(true) {}

bool InitialClientData::InitializeFromString(const std::string& str) {
  std::vector<std::string> parts(SplitString(str, ','));
  if (parts.size() != 8) {
    LOG(ERROR) << "expected 8 comma separated arguments";
    return false;
  }

  if (!HandleFromString(parts[0], &request_crash_dump_) ||
      !HandleFromString(parts[1], &request_non_crash_dump_) ||
      !HandleFromString(parts[2], &non_crash_dump_completed_) ||
      !HandleFromString(parts[3], &first_pipe_instance_) ||
      !HandleFromString(parts[4], &client_process_) ||
      !AddressFromString(parts[5], &crash_exception_information_) ||
      !AddressFromString(parts[6], &non_crash_exception_information_) ||
      !AddressFromString(parts[7], &debug_critical_section_address_)) {
    return false;
  }

  is_valid_ = true;
  return true;
}

std::string InitialClientData::StringRepresentation() const {
  return base::StringPrintf("0x%x,0x%x,0x%x,0x%x,0x%x,0x%I64x,0x%I64x,0x%I64x",
                            HandleToInt(request_crash_dump_),
                            HandleToInt(request_non_crash_dump_),
                            HandleToInt(non_crash_dump_completed_),
                            HandleToInt(first_pipe_instance_),
                            HandleToInt(client_process_),
                            crash_exception_information_,
                            non_crash_exception_information_,
                            debug_critical_section_address_);
}

}  // namespace crashpad
