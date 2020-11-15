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

#ifndef CRASHPAD_COMPAT_ANDROID_LINUX_ELF_H_
#define CRASHPAD_COMPAT_ANDROID_LINUX_ELF_H_

#include_next <linux/elf.h>

// Android 5.0.0 (API 21) NDK

#if defined(__i386__) || defined(__x86_64__)
#if !defined(NT_386_TLS)
#define NT_386_TLS 0x200
#endif
#endif  // __i386__ || __x86_64__

#if defined(__ARMEL__) || defined(__aarch64__)
#if !defined(NT_ARM_VFP)
#define NT_ARM_VFP 0x400
#endif

#if !defined(NT_ARM_TLS)
#define NT_ARM_TLS 0x401
#endif
#endif  // __ARMEL__ || __aarch64__

#endif  // CRASHPAD_COMPAT_ANDROID_LINUX_ELF_H_
