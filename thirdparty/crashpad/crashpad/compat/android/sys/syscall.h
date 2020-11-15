// Copyright 2017 The Crashpad Authors. All rights reserved.
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

#ifndef CRASHPAD_COMPAT_ANDROID_SYS_SYSCALL_H_
#define CRASHPAD_COMPAT_ANDROID_SYS_SYSCALL_H_

#include_next <sys/syscall.h>

// Android 5.0.0 (API 21) NDK

#if !defined(SYS_epoll_create1)
#define SYS_epoll_create1 __NR_epoll_create1
#endif

#if !defined(SYS_gettid)
#define SYS_gettid __NR_gettid
#endif

#if !defined(SYS_timer_create)
#define SYS_timer_create __NR_timer_create
#endif

#if !defined(SYS_timer_getoverrun)
#define SYS_timer_getoverrun __NR_timer_getoverrun
#endif

#if !defined(SYS_timer_settime)
#define SYS_timer_settime __NR_timer_settime
#endif

#endif  // CRASHPAD_COMPAT_ANDROID_SYS_SYSCALL_H_
