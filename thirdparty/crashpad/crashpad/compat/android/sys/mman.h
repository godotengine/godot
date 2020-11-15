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

#ifndef CRASHPAD_COMPAT_ANDROID_SYS_MMAN_H_
#define CRASHPAD_COMPAT_ANDROID_SYS_MMAN_H_

#include_next <sys/mman.h>

#include <android/api-level.h>
#include <sys/cdefs.h>

// There’s no mmap() wrapper compatible with a 64-bit off_t for 32-bit code
// until API 21 (Android 5.0/“Lollipop”). A custom mmap() wrapper is provided
// here. Note that this scenario is only possible with NDK unified headers.
//
// https://android.googlesource.com/platform/bionic/+/0bfcbaf4d069e005d6e959d97f8d11c77722b70d/docs/32-bit-abi.md#is-32_bit-1

#if defined(__USE_FILE_OFFSET64) && __ANDROID_API__ < 21

#ifdef __cplusplus
extern "C" {
#endif

void* mmap(void* addr, size_t size, int prot, int flags, int fd, off_t offset);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // defined(__USE_FILE_OFFSET64) && __ANDROID_API__ < 21

#endif  // CRASHPAD_COMPAT_ANDROID_SYS_MMAN_H_
