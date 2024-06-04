// Copyright (c) 2007, Google Inc.
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
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. */

// minidump_size.h: Provides a C++ template for programmatic access to
// the sizes of various types defined in minidump_format.h.
//
// Author: Mark Mentovai

#ifndef GOOGLE_BREAKPAD_COMMON_MINIDUMP_SIZE_H__
#define GOOGLE_BREAKPAD_COMMON_MINIDUMP_SIZE_H__

#include <sys/types.h>

#include "google_breakpad/common/minidump_format.h"

namespace google_breakpad {

template<typename T>
class minidump_size {
 public:
  static size_t size() { return sizeof(T); }
};

// Explicit specializations for variable-length types.  The size returned
// for these should be the size for an object without its variable-length
// section.

template<>
class minidump_size<MDString> {
 public:
  static size_t size() { return MDString_minsize; }
};

template<>
class minidump_size<MDRawThreadList> {
 public:
  static size_t size() { return MDRawThreadList_minsize; }
};

template<>
class minidump_size<MDCVInfoPDB20> {
 public:
  static size_t size() { return MDCVInfoPDB20_minsize; }
};

template<>
class minidump_size<MDCVInfoPDB70> {
 public:
  static size_t size() { return MDCVInfoPDB70_minsize; }
};

template<>
class minidump_size<MDCVInfoELF> {
 public:
  static size_t size() { return MDCVInfoELF_minsize; }
};

template<>
class minidump_size<MDImageDebugMisc> {
 public:
  static size_t size() { return MDImageDebugMisc_minsize; }
};

template<>
class minidump_size<MDRawModuleList> {
 public:
  static size_t size() { return MDRawModuleList_minsize; }
};

template<>
class minidump_size<MDRawMemoryList> {
 public:
  static size_t size() { return MDRawMemoryList_minsize; }
};

// Explicit specialization for MDRawModule, for which sizeof may include
// tail-padding on some architectures but not others.

template<>
class minidump_size<MDRawModule> {
 public:
  static size_t size() { return MD_MODULE_SIZE; }
};

}  // namespace google_breakpad

#endif  // GOOGLE_BREAKPAD_COMMON_MINIDUMP_SIZE_H__
