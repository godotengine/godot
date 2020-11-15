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

#include <intrin.h>
#include <windows.h>

#include "base/logging.h"
#include "build/build_config.h"
#include "client/crashpad_client.h"
#include "util/misc/capture_context.h"
#include "util/win/address_types.h"

int wmain(int argc, wchar_t* argv[]) {
  CHECK_EQ(argc, 2);

  crashpad::CrashpadClient client;
  CHECK(client.SetHandlerIPCPipe(argv[1]));

  HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
  PCHECK(out != INVALID_HANDLE_VALUE) << "GetStdHandle";

  CONTEXT context;
  crashpad::CaptureContext(&context);
#if defined(ARCH_CPU_64_BITS)
  crashpad::WinVMAddress break_address = context.Rip;
#else
  crashpad::WinVMAddress break_address = context.Eip;
#endif

  // This does not used CheckedWriteFile() because at high optimization
  // settings, a lot of logging code can be inlined, causing there to be a large
  // number of instructions between where the IP is captured and the actual
  // __debugbreak(). Instead call Windows' WriteFile() to minimize the amount of
  // code here. Because the next line is going to crash in any case, there's
  // minimal difference in behavior aside from an indication of what broke when
  // the other end experiences a ReadFile() error.
  DWORD bytes_written;
  WriteFile(
      out, &break_address, sizeof(break_address), &bytes_written, nullptr);

  __debugbreak();

  return 0;
}
