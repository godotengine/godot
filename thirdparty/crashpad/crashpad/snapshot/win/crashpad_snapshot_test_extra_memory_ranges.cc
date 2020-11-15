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

#include <windows.h>

#include "base/logging.h"
#include "client/crashpad_info.h"
#include "util/file/file_io.h"


int wmain(int argc, wchar_t* argv[]) {
  using namespace crashpad;

  CrashpadInfo* crashpad_info = CrashpadInfo::GetCrashpadInfo();

  // This is "leaked" to crashpad_info.
  SimpleAddressRangeBag* extra_ranges = new SimpleAddressRangeBag();
  extra_ranges->Insert(CheckedRange<uint64_t>(0, 1));
  extra_ranges->Insert(CheckedRange<uint64_t>(1, 0));
  extra_ranges->Insert(CheckedRange<uint64_t>(0x1000000000ULL, 0x1000));
  extra_ranges->Insert(CheckedRange<uint64_t>(0x2000, 0x2000000000ULL));
  extra_ranges->Insert(CheckedRange<uint64_t>(1234, 5678));
  extra_ranges->Insert(CheckedRange<uint64_t>(1234, 5678));
  extra_ranges->Insert(CheckedRange<uint64_t>(1234, 5678));
  crashpad_info->set_extra_memory_ranges(extra_ranges);

  // Tell the parent that the environment has been set up.
  HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
  PCHECK(out != INVALID_HANDLE_VALUE) << "GetStdHandle";
  char c = ' ';
  CheckedWriteFile(out, &c, sizeof(c));

  HANDLE in = GetStdHandle(STD_INPUT_HANDLE);
  PCHECK(in != INVALID_HANDLE_VALUE) << "GetStdHandle";
  CheckedReadFileExactly(in, &c, sizeof(c));
  CHECK(c == 'd' || c == ' ');

  // If 'd' we crash with a debug break, otherwise exit normally.
  if (c == 'd')
    __debugbreak();

  return 0;
}
