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

#include <stdio.h>
#include <windows.h>

#include "base/debug/alias.h"
#include "base/logging.h"
#include "base/macros.h"
#include "base/strings/stringprintf.h"
#include "base/strings/utf_string_conversions.h"
#include "client/crashpad_client.h"
#include "client/crashpad_info.h"

namespace {

DWORD WINAPI Thread1(LPVOID context) {
  HANDLE event = context;

  // Increase the thread priority as a hacky way to signal to
  // crash_other_program.exe that this is the thread to dump.
  PCHECK(SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL));

  // Let the main thread proceed.
  PCHECK(SetEvent(event));

  Sleep(INFINITE);

  NOTREACHED();
  return 0;
}

DWORD WINAPI Thread2(LPVOID dummy) {
  Sleep(INFINITE);
  NOTREACHED();
  return 0;
}

DWORD WINAPI Thread3(LPVOID context) {
  // This is a convenient way to pass the event handle to loader_lock_dll.dll.
  HANDLE event = context;
  PCHECK(SetEnvironmentVariable(
      L"CRASHPAD_TEST_DLL_EVENT",
      base::UTF8ToUTF16(base::StringPrintf("%p", event)).c_str()));

  HMODULE dll = LoadLibrary(L"loader_lock_dll.dll");
  if (!dll)
    PLOG(FATAL) << "LoadLibrary";

  // This call is not expected to return.
  if (!FreeLibrary(dll))
    PLOG(FATAL) << "FreeLibrary";

  NOTREACHED();
  return 0;
}

}  // namespace

int wmain(int argc, wchar_t* argv[]) {
  crashpad::CrashpadClient client;

  if (argc == 2) {
    if (!client.SetHandlerIPCPipe(argv[1])) {
      LOG(ERROR) << "SetHandlerIPCPipe";
      return EXIT_FAILURE;
    }
  } else {
    fprintf(stderr, "Usage: %ls <server_pipe_name>\n", argv[0]);
    return EXIT_FAILURE;
  }

  // Make sure this module has a CrashpadInfo structure.
  crashpad::CrashpadInfo* crashpad_info =
      crashpad::CrashpadInfo::GetCrashpadInfo();
  base::debug::Alias(crashpad_info);

  HANDLE event = CreateEvent(nullptr,
                             false,  // bManualReset
                             false,
                             nullptr);
  if (!event) {
    PLOG(ERROR) << "CreateEvent";
    return EXIT_FAILURE;
  }

  HANDLE threads[3];
  threads[0] = CreateThread(nullptr, 0, Thread1, event, 0, nullptr);

  threads[1] = CreateThread(nullptr, 0, Thread2, nullptr, 0, nullptr);

  // Wait for Thread1() to complete its work and reach its Sleep() before
  // starting the next thread, which will hold the loader lock and potentially
  // block any further progress.
  if (WaitForSingleObject(event, INFINITE) != WAIT_OBJECT_0) {
    PLOG(ERROR) << "WaitForSingleObject";
    return EXIT_FAILURE;
  }

  // Use the same event object, which was automatically reset.
  threads[2] = CreateThread(nullptr, 0, Thread3, event, 0, nullptr);

  // Wait for loader_lock_dll.dll to signal that the loader lock is held and
  // won’t be released.
  if (WaitForSingleObject(event, INFINITE) != WAIT_OBJECT_0) {
    PLOG(ERROR) << "WaitForSingleObject";
    return EXIT_FAILURE;
  }

  // Signal to the parent that everything is ready.
  fprintf(stdout, " ");
  fflush(stdout);

  // This is not expected to return.
  DWORD count =
      WaitForMultipleObjects(arraysize(threads), threads, true, INFINITE);
  if (count == WAIT_FAILED) {
    PLOG(ERROR) << "WaitForMultipleObjects";
  } else {
    LOG(ERROR) << "WaitForMultipleObjects: " << count;
  }

  return EXIT_FAILURE;
}
