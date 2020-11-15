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

#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <windows.h>

namespace {

// This custom logging function is to avoid taking a dependency on base in this
// standalone DLL. Don’t call this directly, use the LOG_FATAL() and
// PLOG_FATAL() macros below.
__declspec(noreturn) void LogFatal(const char* file,
                                   int line,
                                   bool get_last_error,
                                   const char* format,
                                   ...) {
  DWORD last_error = ERROR_SUCCESS;
  if (get_last_error) {
    last_error = GetLastError();
  }

  char fname[_MAX_FNAME];
  char ext[_MAX_EXT];
  if (_splitpath_s(file,
                   nullptr,
                   0,
                   nullptr,
                   0,
                   fname,
                   sizeof(fname),
                   ext,
                   sizeof(ext)) == 0) {
    fprintf(stderr, "%s%s", fname, ext);
  } else {
    fputs(file, stderr);
  }
  fprintf(stderr, ":%d: ", line);

  va_list va;
  va_start(va, format);
  vfprintf(stderr, format, va);
  va_end(va);

  if (get_last_error) {
    fprintf(stderr, ": error %lu", last_error);
  }

  fputs("\n", stderr);
  fflush(stderr);

  TerminateProcess(GetCurrentProcess(), 1);
  __fastfail(FAST_FAIL_FATAL_APP_EXIT);
}

#define LOG_FATAL(...)                                \
  do {                                                \
    LogFatal(__FILE__, __LINE__, false, __VA_ARGS__); \
  } while (false)
#define PLOG_FATAL(...)                              \
  do {                                               \
    LogFatal(__FILE__, __LINE__, true, __VA_ARGS__); \
  } while (false)

}  // namespace

// This program intentionally blocks in DllMain which is executed with the
// loader lock locked. This allows us to test that
// CrashpadClient::DumpAndCrashTargetProcess() can still dump the target in this
// case.
BOOL WINAPI DllMain(HINSTANCE, DWORD reason, LPVOID) {
  switch (reason) {
    case DLL_PROCESS_DETACH:
    case DLL_THREAD_DETACH: {
      // Recover the event handle stashed by the main executable.
      static constexpr size_t kEventStringSize = 19;
      wchar_t event_string[kEventStringSize];
      SetLastError(ERROR_SUCCESS);
      DWORD size = GetEnvironmentVariable(
          L"CRASHPAD_TEST_DLL_EVENT", event_string, kEventStringSize);
      if (size == 0 && GetLastError() != ERROR_SUCCESS) {
        PLOG_FATAL("GetEnvironmentVariable");
      }
      if (size == 0 || size >= kEventStringSize) {
        LOG_FATAL("GetEnvironmentVariable: size %u", size);
      }

      HANDLE event;
      int converted = swscanf(event_string, L"%p", &event);
      if (converted != 1) {
        LOG_FATAL("swscanf: converted %d", converted);
      }

      // Let the main thread know that the loader lock is and will remain held.
      if (!SetEvent(event)) {
        PLOG_FATAL("SetEvent");
      }

      Sleep(INFINITE);

      break;
    }
  }

  return TRUE;
}
