// Copyright 2014 The Crashpad Authors. All rights reserved.
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

#ifndef CRASHPAD_COMPAT_MAC_MACH_MACH_H_
#define CRASHPAD_COMPAT_MAC_MACH_MACH_H_

#include_next <mach/mach.h>

// <mach/exception_types.h>

// 10.8 SDK

#ifndef EXC_RESOURCE
#define EXC_RESOURCE 11
#endif

#ifndef EXC_MASK_RESOURCE
#define EXC_MASK_RESOURCE (1 << EXC_RESOURCE)
#endif

// 10.9 SDK

#ifndef EXC_GUARD
#define EXC_GUARD 12
#endif

#ifndef EXC_MASK_GUARD
#define EXC_MASK_GUARD (1 << EXC_GUARD)
#endif

// 10.11 SDK

#ifndef EXC_CORPSE_NOTIFY
#define EXC_CORPSE_NOTIFY 13
#endif

#ifndef EXC_MASK_CORPSE_NOTIFY
#define EXC_MASK_CORPSE_NOTIFY (1 << EXC_CORPSE_NOTIFY)
#endif

// Don’t expose EXC_MASK_ALL at all, because its definition varies with SDK, and
// older kernels will reject values that they don’t understand. Instead, use
// crashpad::ExcMaskAll(), which computes the correct value of EXC_MASK_ALL for
// the running system.
#undef EXC_MASK_ALL

#if defined(__i386__) || defined(__x86_64__)

// <mach/i386/exception.h>

// 10.11 SDK

#if EXC_TYPES_COUNT > 14  // Definition varies with SDK
#error Update this file for new exception types
#elif EXC_TYPES_COUNT != 14
#undef EXC_TYPES_COUNT
#define EXC_TYPES_COUNT 14
#endif

// <mach/i386/thread_status.h>

// 10.6 SDK
//
// Earlier versions of this SDK didn’t have AVX definitions. They didn’t appear
// until the version of the 10.6 SDK that shipped with Xcode 4.2, although
// versions of this SDK appeared with Xcode releases as early as Xcode 3.2.
// Similarly, the kernel didn’t handle AVX state until Mac OS X 10.6.8
// (xnu-1504.15.3) and presumably the hardware-specific versions of Mac OS X
// 10.6.7 intended to run on processors with AVX.

#ifndef x86_AVX_STATE32
#define x86_AVX_STATE32 16
#endif

#ifndef x86_AVX_STATE64
#define x86_AVX_STATE64 17
#endif

// 10.8 SDK

#ifndef x86_AVX_STATE
#define x86_AVX_STATE 18
#endif

// 10.13 SDK

#ifndef x86_AVX512_STATE32
#define x86_AVX512_STATE32 19
#endif

#ifndef x86_AVX512_STATE64
#define x86_AVX512_STATE64 20
#endif

#ifndef x86_AVX512_STATE
#define x86_AVX512_STATE 21
#endif

#endif  // defined(__i386__) || defined(__x86_64__)

// <mach/thread_status.h>

// 10.8 SDK

#ifndef THREAD_STATE_FLAVOR_LIST_10_9
#define THREAD_STATE_FLAVOR_LIST_10_9 129
#endif

#endif  // CRASHPAD_COMPAT_MAC_MACH_MACH_H_
