// Copyright 2008 Google LLC
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

#ifndef CLIENT_WINDOWS_COMMON_AUTO_CRITICAL_SECTION_H__
#define CLIENT_WINDOWS_COMMON_AUTO_CRITICAL_SECTION_H__

#include <windows.h>

namespace google_breakpad {

// Automatically enters the critical section in the constructor and leaves
// the critical section in the destructor.
class AutoCriticalSection {
 public:
  // Creates a new instance with the given critical section object
  // and enters the critical section immediately.
  explicit AutoCriticalSection(CRITICAL_SECTION* cs) : cs_(cs), taken_(false) {
    assert(cs_);
    Acquire();
  }

  // Destructor: leaves the critical section.
  ~AutoCriticalSection() {
    if (taken_) {
      Release();
    }
  }

  // Enters the critical section. Recursive Acquire() calls are not allowed.
  void Acquire() {
    assert(!taken_);
    EnterCriticalSection(cs_);
    taken_ = true;
  }

  // Leaves the critical section. The caller should not call Release() unless
  // the critical seciton has been entered already.
  void Release() {
    assert(taken_);
    taken_ = false;
    LeaveCriticalSection(cs_);
  }

 private:
  // Disable copy ctor and operator=.
  AutoCriticalSection(const AutoCriticalSection&);
  AutoCriticalSection& operator=(const AutoCriticalSection&);

  CRITICAL_SECTION* cs_;
  bool taken_;
};

}  // namespace google_breakpad

#endif  // CLIENT_WINDOWS_COMMON_AUTO_CRITICAL_SECTION_H__
