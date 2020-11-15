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

#include "util/win/critical_section_with_debug_info.h"

#include "base/logging.h"
#include "util/win/get_function.h"

namespace crashpad {

namespace {

bool CrashpadInitializeCriticalSectionEx(
    CRITICAL_SECTION* critical_section,
    DWORD spin_count,
    DWORD flags) {
  static const auto initialize_critical_section_ex =
      GET_FUNCTION_REQUIRED(L"kernel32.dll", ::InitializeCriticalSectionEx);
  BOOL ret =
      initialize_critical_section_ex(critical_section, spin_count, flags);
  if (!ret) {
    PLOG(ERROR) << "InitializeCriticalSectionEx";
    return false;
  }
  return true;
}

}  // namespace

bool InitializeCriticalSectionWithDebugInfoIfPossible(
    CRITICAL_SECTION* critical_section) {
  // On XP and Vista, a plain initialization causes the CRITICAL_SECTION to be
  // allocated with .DebugInfo. On 8 and above, we can pass an additional flag
  // to InitializeCriticalSectionEx() to force the .DebugInfo on. Before Win 8,
  // that flag causes InitializeCriticalSectionEx() to fail. So, for XP, Vista,
  // and 7 we use InitializeCriticalSection(), and for 8 and above,
  // InitializeCriticalSectionEx() with the additional flag.
  //
  // TODO(scottmg): Try to find a solution for Win 7. It's unclear how to force
  // it on for Win 7, however the Loader Lock does have .DebugInfo so there may
  // be a way to do it. The comments in winnt.h imply that perhaps it's passed
  // to InitializeCriticalSectionAndSpinCount() as the top bits of the spin
  // count, but that doesn't appear to work. For now, we initialize a valid
  // CRITICAL_SECTION, but without .DebugInfo.

  const DWORD version = GetVersion();
  const DWORD major_version = LOBYTE(LOWORD(version));
  const DWORD minor_version = HIBYTE(LOWORD(version));
  const bool win7_or_lower =
      major_version < 6 || (major_version == 6 && minor_version <= 1);

  if (win7_or_lower) {
    InitializeCriticalSection(critical_section);
    return true;
  }

  return CrashpadInitializeCriticalSectionEx(
      critical_section, 0, RTL_CRITICAL_SECTION_FLAG_FORCE_DEBUG_INFO);
}

}  // namespace crashpad
