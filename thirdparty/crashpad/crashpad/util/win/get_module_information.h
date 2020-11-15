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

#ifndef CRASHPAD_UTIL_WIN_GET_MODULE_INFORMATION_H_
#define CRASHPAD_UTIL_WIN_GET_MODULE_INFORMATION_H_

#include <windows.h>

#define PSAPI_VERSION 1
#include <psapi.h>

namespace crashpad {

//! \brief Proxy function for `GetModuleInformation()`.
BOOL CrashpadGetModuleInformation(HANDLE process,
                                  HMODULE module,
                                  MODULEINFO* module_info,
                                  DWORD cb);

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_WIN_GET_MODULE_INFORMATION_H_
