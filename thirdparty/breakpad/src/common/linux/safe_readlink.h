// Copyright (c) 2011, Google Inc.
// All rights reserved.
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
//     * Neither the name of Google Inc. nor the names of its
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

// safe_readlink.h: Define the google_breakpad::SafeReadLink function,
// which wraps sys_readlink and gurantees the result is NULL-terminated.

#ifndef COMMON_LINUX_SAFE_READLINK_H_
#define COMMON_LINUX_SAFE_READLINK_H_

#include <stddef.h>

namespace google_breakpad {

// This function wraps sys_readlink() and performs the same functionalty,
// but guarantees |buffer| is NULL-terminated if sys_readlink() returns
// no error. It takes the same arguments as sys_readlink(), but unlike
// sys_readlink(), it returns true on success.
//
// |buffer_size| specifies the size of |buffer| in bytes. As this function
// always NULL-terminates |buffer| on success, |buffer_size| should be
// at least one byte longer than the expected path length (e.g. PATH_MAX,
// which is typically defined as the maximum length of a path name
// including the NULL byte).
//
// The implementation of this function calls sys_readlink() instead of
// readlink(), it can thus be used in the context where calling to libc
// functions is discouraged.
bool SafeReadLink(const char* path, char* buffer, size_t buffer_size);

// Same as the three-argument version of SafeReadLink() but deduces the
// size of |buffer| if it is a char array of known size.
template <size_t N>
bool SafeReadLink(const char* path, char (&buffer)[N]) {
  return SafeReadLink(path, buffer, sizeof(buffer));
}

}  // namespace google_breakpad

#endif  // COMMON_LINUX_SAFE_READLINK_H_
