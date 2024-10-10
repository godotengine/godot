// Copyright 2011 Google LLC
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

// safe_readlink.cc: Implement google_breakpad::SafeReadLink.
// See safe_readlink.h for details.

#ifdef HAVE_CONFIG_H
#include <config.h>  // Must come first
#endif

#include <stddef.h>

#include "third_party/lss/linux_syscall_support.h"

namespace google_breakpad {

bool SafeReadLink(const char* path, char* buffer, size_t buffer_size) {
  // sys_readlink() does not add a NULL byte to |buffer|. In order to return
  // a NULL-terminated string in |buffer|, |buffer_size| should be at least
  // one byte longer than the expected path length. Also, sys_readlink()
  // returns the actual path length on success, which does not count the
  // NULL byte, so |result_size| should be less than |buffer_size|.
  ssize_t result_size = sys_readlink(path, buffer, buffer_size);
  if (result_size >= 0 && static_cast<size_t>(result_size) < buffer_size) {
    buffer[result_size] = '\0';
    return true;
  }
  return false;
}

}  // namespace google_breakpad
