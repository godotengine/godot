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
#include <stdlib.h>
#include <string.h>
#include <windows.h>

#include "base/logging.h"
#include "client/crashpad_client.h"

namespace crashpad {
namespace {

enum WhereToSignalFrom {
  kUnknown = -1,
  kMain = 0,
  kBackground = 1,
};

WhereToSignalFrom MainOrBackground(wchar_t* name) {
  if (wcscmp(name, L"main") == 0)
    return kMain;
  if (wcscmp(name, L"background") == 0)
    return kBackground;
  return kUnknown;
}

DWORD WINAPI BackgroundThread(void* arg) {
  abort();
  return 0;
}

int CrashySignalMain(int argc, wchar_t* argv[]) {
  CrashpadClient client;

  WhereToSignalFrom from;
  if (argc == 3 && (from = MainOrBackground(argv[2])) != kUnknown) {
    if (!client.SetHandlerIPCPipe(argv[1])) {
      LOG(ERROR) << "SetHandler";
      return EXIT_FAILURE;
    }
  } else {
    fprintf(stderr, "Usage: %ls <server_pipe_name> main|background\n", argv[0]);
    return EXIT_FAILURE;
  }

  // In debug builds part of abort() is to open a dialog. We don't want tests to
  // block at that dialog, so disable it.
  _set_abort_behavior(0, _WRITE_ABORT_MSG);

  if (from == kBackground) {
    HANDLE thread = CreateThread(nullptr,
                                 0,
                                 &BackgroundThread,
                                 nullptr,
                                 0,
                                 nullptr);
    if (!thread) {
      PLOG(ERROR) << "CreateThread";
      return EXIT_FAILURE;
    }
    if (WaitForSingleObject(thread, INFINITE) != WAIT_OBJECT_0) {
      PLOG(ERROR) << "WaitForSingleObject";
      return EXIT_FAILURE;
    }
  } else {
    abort();
  }

  return EXIT_SUCCESS;
}

}  // namespace
}  // namespace crashpad

int wmain(int argc, wchar_t* argv[]) {
  return crashpad::CrashySignalMain(argc, argv);
}
