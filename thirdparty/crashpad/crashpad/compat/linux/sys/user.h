// Copyright 2018 The Crashpad Authors. All rights reserved.
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

#ifndef CRASHPAD_COMPAT_LINUX_SYS_USER_H_
#define CRASHPAD_COMPAT_LINUX_SYS_USER_H_

#include_next <sys/user.h>

#include <features.h>

// glibc for 64-bit ARM uses different names for these structs prior to 2.20.
#if defined(__aarch64__) && defined(__GLIBC__)
#if !__GLIBC_PREREQ(2, 20)
using user_regs_struct = user_pt_regs;
using user_fpsimd_struct = user_fpsimd_state;
#endif
#endif

#endif  // CRASHPAD_COMPAT_LINUX_SYS_USER_H_
