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
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Author: Alfred Peng

#ifndef COMMON_SOLARIS_MESSAGE_OUTPUT_H__
#define COMMON_SOLARIS_MESSAGE_OUTPUT_H__

namespace google_breakpad {

const int MESSAGE_MAX = 1000;

// Message output macros.
// snprintf doesn't operate heap on Solaris, while printf and fprintf do.
// Use snprintf here to avoid heap allocation.
#define print_message1(std, message) \
  char buffer[MESSAGE_MAX]; \
  int len = snprintf(buffer, MESSAGE_MAX, message); \
  write(std, buffer, len)

#define print_message2(std, message, para) \
  char buffer[MESSAGE_MAX]; \
  int len = snprintf(buffer, MESSAGE_MAX, message, para); \
  write(std, buffer, len);

}  // namespace google_breakpad

#endif  // COMMON_SOLARIS_MESSAGE_OUTPUT_H__
