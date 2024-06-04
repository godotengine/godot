// Copyright 2014 Google Inc.
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

#include "common/linux/crc32.h"

namespace google_breakpad {

// This implementation is based on the sample implementation in RFC 1952.

// CRC32 polynomial, in reversed form.
// See RFC 1952, or http://en.wikipedia.org/wiki/Cyclic_redundancy_check
static const uint32_t kCrc32Polynomial = 0xEDB88320;
static uint32_t kCrc32Table[256] = { 0 };

#define arraysize(f) (sizeof(f) / sizeof(*f))

static void EnsureCrc32TableInited() {
  if (kCrc32Table[arraysize(kCrc32Table) - 1])
    return;  // already inited
  for (uint32_t i = 0; i < arraysize(kCrc32Table); ++i) {
    uint32_t c = i;
    for (size_t j = 0; j < 8; ++j) {
      if (c & 1) {
        c = kCrc32Polynomial ^ (c >> 1);
      } else {
        c >>= 1;
      }
    }
    kCrc32Table[i] = c;
  }
}

uint32_t UpdateCrc32(uint32_t start, const void* buf, size_t len) {
  EnsureCrc32TableInited();

  uint32_t c = start ^ 0xFFFFFFFF;
  const uint8_t* u = static_cast<const uint8_t*>(buf);
  for (size_t i = 0; i < len; ++i) {
    c = kCrc32Table[(c ^ u[i]) & 0xFF] ^ (c >> 8);
  }
  return c ^ 0xFFFFFFFF;
}

}  // namespace google_breakpad
