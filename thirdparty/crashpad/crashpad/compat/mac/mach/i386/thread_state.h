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

#ifndef CRASHPAD_COMPAT_MAC_MACH_I386_THREAD_STATE_H_
#define CRASHPAD_COMPAT_MAC_MACH_I386_THREAD_STATE_H_

#include_next <mach/i386/thread_state.h>

// 10.13 SDK
//
// This was defined as 244 in the 10.7 through 10.12 SDKs, and 144 previously.
#if I386_THREAD_STATE_MAX < 614
#undef I386_THREAD_STATE_MAX
#define I386_THREAD_STATE_MAX (614)
#endif

#endif  // CRASHPAD_COMPAT_MAC_MACH_I386_THREAD_STATE_H_
