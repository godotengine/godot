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

#include "util/win/ntstatus_logging.h"

#include <string>

#include "base/strings/stringprintf.h"

namespace {

std::string FormatNtstatus(DWORD ntstatus) {
  char msgbuf[256];
  DWORD len = FormatMessageA(
      FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS |
          FORMAT_MESSAGE_MAX_WIDTH_MASK | FORMAT_MESSAGE_FROM_HMODULE,
      GetModuleHandle(L"ntdll.dll"),
      ntstatus,
      0,
      msgbuf,
      arraysize(msgbuf),
      nullptr);
  if (len) {
    // Most system messages end in a period and a space. Remove the space if
    // it’s there, because ~NtstatusLogMessage() includes one.
    if (len >= 1 && msgbuf[len - 1] == ' ') {
      msgbuf[len - 1] = '\0';
    }
    return msgbuf;
  } else {
    return base::StringPrintf("<failed to retrieve error message (0x%lx)>",
                              GetLastError());
  }
}

}  // namespace

namespace logging {

NtstatusLogMessage::NtstatusLogMessage(
#if defined(MINI_CHROMIUM_BASE_LOGGING_H_)
    const char* function,
#endif
    const char* file_path,
    int line,
    LogSeverity severity,
    DWORD ntstatus)
    : LogMessage(
#if defined(MINI_CHROMIUM_BASE_LOGGING_H_)
          function,
#endif
          file_path,
          line,
          severity),
      ntstatus_(ntstatus) {
}

NtstatusLogMessage::~NtstatusLogMessage() {
  stream() << ": " << FormatNtstatus(ntstatus_)
           << base::StringPrintf(" (0x%08lx)", ntstatus_);
}

}  // namespace logging
