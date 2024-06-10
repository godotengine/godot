// Copyright 2012 Google LLC
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google LLC nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef GOOGLE_BREAKPAD_COMMON_ANDROID_INCLUDE_SYS_USER_H
#define GOOGLE_BREAKPAD_COMMON_ANDROID_INCLUDE_SYS_USER_H

// The purpose of this file is to glue the mismatching headers (Android NDK vs
// glibc) and therefore avoid doing otherwise awkward #ifdefs in the code.
// The following quirks are currently handled by this file:
// - i386: Use the Android NDK but alias user_fxsr_struct > user_fpxregs_struct.

// TODO(primiano): remove these changes after Chromium has stably rolled to
// an NDK with the appropriate fixes. https://crbug.com/358831

// With traditional headers, <sys/user.h> forgot to do this. Unified headers get
// it right.
#include <sys/types.h>

#include_next <sys/user.h>

#include <android/api-level.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#if defined(__i386__)
#if __ANDROID_API__ < 21 && !defined(__ANDROID_API_N__)

// user_fpxregs_struct was called user_fxsr_struct in traditional headers before
// API level 21. Unified headers call it user_fpxregs_struct regardless of the
// chosen API level. __ANDROID_API_N__ is a proxy for determining whether
// unified headers are in use. It’s only defined by unified headers.
typedef struct user_fxsr_struct user_fpxregs_struct;

#endif  // __ANDROID_API__ < 21 && !defined(__ANDROID_API_N__)
#endif  // defined(__i386__)

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // GOOGLE_BREAKPAD_COMMON_ANDROID_INCLUDE_SYS_USER_H
