// Copyright 2010 Google LLC
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

#ifndef COMMON_LINUX_EINTR_WRAPPER_H_
#define COMMON_LINUX_EINTR_WRAPPER_H_

#include <errno.h>

// This provides a wrapper around system calls which may be interrupted by a
// signal and return EINTR. See man 7 signal.
//

#define HANDLE_EINTR(x) ({ \
  __typeof__(x) eintr_wrapper_result; \
  do { \
    eintr_wrapper_result = (x); \
  } while (eintr_wrapper_result == -1 && errno == EINTR); \
  eintr_wrapper_result; \
})

#define IGNORE_EINTR(x) ({ \
  __typeof__(x) eintr_wrapper_result; \
  do { \
    eintr_wrapper_result = (x); \
    if (eintr_wrapper_result == -1 && errno == EINTR) { \
      eintr_wrapper_result = 0; \
    } \
  } while (0); \
  eintr_wrapper_result; \
})

#endif  // COMMON_LINUX_EINTR_WRAPPER_H_
