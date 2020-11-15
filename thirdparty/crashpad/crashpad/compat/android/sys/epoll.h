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

#ifndef CRASHPAD_COMPAT_ANDROID_SYS_EPOLL_H_
#define CRASHPAD_COMPAT_ANDROID_SYS_EPOLL_H_

#include_next <sys/epoll.h>

#include <android/api-level.h>
#include <fcntl.h>

// This is missing from traditional headers before API 21.
#if !defined(EPOLLRDHUP)
#define EPOLLRDHUP 0x00002000
#endif

// EPOLL_CLOEXEC is undefined in traditional headers before API 21 and removed
// from unified headers at API levels < 21 as a means to indicate that
// epoll_create1 is missing from the C library, but the raw system call should
// still be available.
#if !defined(EPOLL_CLOEXEC)
#define EPOLL_CLOEXEC O_CLOEXEC
#endif

#if __ANDROID_API__ < 21

#ifdef __cplusplus
extern "C" {
#endif

int epoll_create1(int flags);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // __ANDROID_API__ < 21

#endif  // CRASHPAD_COMPAT_ANDROID_SYS_EPOLL_H_
