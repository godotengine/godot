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

#ifndef CRASHPAD_UTIL_WIN_XP_COMPAT_H_
#define CRASHPAD_UTIL_WIN_XP_COMPAT_H_

#include <windows.h>

namespace crashpad {

enum {
  //! \brief This is the XP-suitable value of `PROCESS_ALL_ACCESS`.
  //!
  //! Requesting `PROCESS_ALL_ACCESS` with the value defined when building
  //! against a Vista+ SDK results in `ERROR_ACCESS_DENIED` when running on XP.
  //! See https://msdn.microsoft.com/library/ms684880.aspx.
  kXPProcessAllAccess = STANDARD_RIGHTS_REQUIRED | SYNCHRONIZE | 0xFFF,

  //! \brief This is the XP-suitable value of `THREAD_ALL_ACCESS`.
  //!
  //! Requesting `THREAD_ALL_ACCESS` with the value defined when building
  //! against a Vista+ SDK results in `ERROR_ACCESS_DENIED` when running on XP.
  //! See https://msdn.microsoft.com/library/ms686769.aspx.
  kXPThreadAllAccess = STANDARD_RIGHTS_REQUIRED | SYNCHRONIZE | 0x3FF,
};

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_WIN_XP_COMPAT_H_
