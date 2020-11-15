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

#include <windows.h>

#include "base/logging.h"
#include "client/crashpad_info.h"
#include "util/file/file_io.h"
#include "util/synchronization/semaphore.h"
#include "util/win/scoped_handle.h"

namespace {

DWORD WINAPI LotsOfReferencesThreadProc(void* param) {
  crashpad::Semaphore* semaphore =
      reinterpret_cast<crashpad::Semaphore*>(param);

  // Allocate a bunch of pointers to things on the stack.
  int* pointers[1000];
  for (size_t i = 0; i < arraysize(pointers); ++i) {
    pointers[i] = new int[2048];
  }

  semaphore->Signal();
  Sleep(INFINITE);
  return 0;
}

}  // namespace

int wmain(int argc, wchar_t* argv[]) {
  CHECK_EQ(argc, 2);

  crashpad::ScopedKernelHANDLE done(CreateEvent(nullptr, true, false, argv[1]));
  PCHECK(done.is_valid()) << "CreateEvent";

  PCHECK(LoadLibrary(L"crashpad_snapshot_test_image_reader_module.dll"))
      << "LoadLibrary";

  // Create threads with lots of stack pointers to memory. This is used to
  // verify the cap on pointed-to memory.
  crashpad::Semaphore semaphore(0);
  crashpad::ScopedKernelHANDLE threads[100];
  for (size_t i = 0; i < arraysize(threads); ++i) {
    threads[i].reset(CreateThread(nullptr,
                                  0,
                                  &LotsOfReferencesThreadProc,
                                  reinterpret_cast<void*>(&semaphore),
                                  0,
                                  nullptr));
    if (!threads[i].is_valid()) {
      PLOG(ERROR) << "CreateThread";
      return 1;
    }
  }

  for (size_t i = 0; i < arraysize(threads); ++i) {
    semaphore.Wait();
  }

  crashpad::CrashpadInfo* crashpad_info =
      crashpad::CrashpadInfo::GetCrashpadInfo();
  crashpad_info->set_gather_indirectly_referenced_memory(
      crashpad::TriState::kEnabled, 100000);

  // Tell the parent process we're ready to proceed.
  HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
  PCHECK(out != INVALID_HANDLE_VALUE) << "GetStdHandle";
  char c = ' ';
  crashpad::CheckedWriteFile(out, &c, sizeof(c));

  // Parent process says we can exit.
  PCHECK(WaitForSingleObject(done.get(), INFINITE) == WAIT_OBJECT_0)
      << "WaitForSingleObject";

  return 0;
}
