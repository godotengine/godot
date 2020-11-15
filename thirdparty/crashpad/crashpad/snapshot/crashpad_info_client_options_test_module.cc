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

#include "build/build_config.h"
#include "client/crashpad_info.h"

#if defined(OS_POSIX)
#define EXPORT __attribute__((visibility("default")))
#elif defined(OS_WIN)
#include <windows.h>
#define EXPORT __declspec(dllexport)
#endif  // OS_POSIX

extern "C" {

// Returns the module’s CrashpadInfo structure. Assuming that this file is built
// into a loadable_module with a distinct static copy of the Crashpad client
// library from the copy built into the loader of this loadable_module, this
// will return a different CrashpadInfo structure than the one that the loader
// uses. Having an extra CrashpadInfo structure makes it possible to test
// behaviors that are relevant in the presence of multiple Crashpad
// client-enabled modules.
//
// This function is used by the CrashpadInfoClientOptions.TwoModules test in
// crashpad_info_client_options_test.cc.
EXPORT crashpad::CrashpadInfo* TestModule_GetCrashpadInfo() {
  return crashpad::CrashpadInfo::GetCrashpadInfo();
}

}  // extern "C"

#if defined(OS_WIN)
BOOL WINAPI DllMain(HINSTANCE hinstance, DWORD reason, LPVOID reserved) {
  return TRUE;
}
#endif  // OS_WIN
