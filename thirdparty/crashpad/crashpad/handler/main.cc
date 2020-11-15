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

#include "handler/handler_main.h"

#include "build/build_config.h"
#include "tools/tool_support.h"

#if defined(OS_WIN)
#include <windows.h>
#endif

#if defined(OS_POSIX)

int main(int argc, char* argv[]) {
  return crashpad::HandlerMain(argc, argv, nullptr);
}

#elif defined(OS_WIN)

namespace {

int HandlerMainAdaptor(int argc, char* argv[]) {
  return crashpad::HandlerMain(argc, argv, nullptr);
}

}  // namespace

// The default entry point for /subsystem:windows. In Crashpad’s own build, this
// is used by crashpad_handler.exe. It’s also used by crashpad_handler.com when
// produced by editbin from a copy of crashpad_handler.exe.
int APIENTRY wWinMain(HINSTANCE, HINSTANCE, wchar_t*, int) {
  return crashpad::ToolSupport::Wmain(__argc, __wargv, HandlerMainAdaptor);
}

// The default entry point for /subsystem:console. This is not currently used by
// Crashpad’s own build, but may be used by other builds.
int wmain(int argc, wchar_t* argv[]) {
  return crashpad::ToolSupport::Wmain(argc, argv, HandlerMainAdaptor);
}

#endif  // OS_POSIX
