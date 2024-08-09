/* Copyright 2019 Google LLC
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *     * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following disclaimer
 * in the documentation and/or other materials provided with the
 * distribution.
 *     * Neither the name of Google LLC nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. */

/* minidump_exception_fuchsia.h: A definition of exception codes for Fuchsia.
 *
 * Author: Ivan Penkov */

#ifndef GOOGLE_BREAKPAD_COMMON_MINIDUMP_EXCEPTION_FUCHSIA_H_
#define GOOGLE_BREAKPAD_COMMON_MINIDUMP_EXCEPTION_FUCHSIA_H_

#include <stddef.h>

#include "google_breakpad/common/breakpad_types.h"

// Based on zircon/system/public/zircon/syscalls/exception.h
typedef enum {
  // Architectural exceptions
  MD_EXCEPTION_CODE_FUCHSIA_GENERAL = 0x8,
  MD_EXCEPTION_CODE_FUCHSIA_FATAL_PAGE_FAULT = 0x108,
  MD_EXCEPTION_CODE_FUCHSIA_UNDEFINED_INSTRUCTION = 0x208,
  MD_EXCEPTION_CODE_FUCHSIA_SW_BREAKPOINT = 0x308,
  MD_EXCEPTION_CODE_FUCHSIA_HW_BREAKPOINT = 0x408,
  MD_EXCEPTION_CODE_FUCHSIA_UNALIGNED_ACCESS = 0x508,
  //
  // Synthetic exceptions
  MD_EXCEPTION_CODE_FUCHSIA_THREAD_STARTING = 0x8008,
  MD_EXCEPTION_CODE_FUCHSIA_THREAD_EXITING = 0x8108,
  MD_EXCEPTION_CODE_FUCHSIA_POLICY_ERROR = 0x8208,
  MD_EXCEPTION_CODE_FUCHSIA_PROCESS_STARTING = 0x8308,
} MDExceptionCodeFuchsia;

#endif  // GOOGLE_BREAKPAD_COMMON_MINIDUMP_EXCEPTION_FUCHSIA_H_
