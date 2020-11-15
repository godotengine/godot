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

#ifndef CRASHPAD_COMPAT_LINUX_SIGNAL_H_
#define CRASHPAD_COMPAT_LINUX_SIGNAL_H_

#include_next <signal.h>

// Missing from glibc and bionic-x86_64

#if defined(__x86_64__) || defined(__i386__)
#if !defined(X86_FXSR_MAGIC)
#define X86_FXSR_MAGIC 0x0000
#endif
#endif  // __x86_64__ || __i386__

#if defined(__aarch64__) || defined(__arm__)

#if !defined(FPSIMD_MAGIC)
#define FPSIMD_MAGIC 0x46508001
#endif

#if !defined(ESR_MAGIC)
#define ESR_MAGIC 0x45535201
#endif

#if !defined(EXTRA_MAGIC)
#define EXTRA_MAGIC 0x45585401
#endif

#if !defined(VFP_MAGIC)
#define VFP_MAGIC 0x56465001
#endif

#if !defined(CRUNCH_MAGIC)
#define CRUNCH_MAGIC 0x5065cf03
#endif

#if !defined(DUMMY_MAGIC)
#define DUMMY_MAGIC 0xb0d9ed01
#endif

#if !defined(IWMMXT_MAGIC)
#define IWMMXT_MAGIC 0x12ef842a
#endif

#endif  // __aarch64__ || __arm__

#endif  // CRASHPAD_COMPAT_LINUX_SIGNAL_H_
