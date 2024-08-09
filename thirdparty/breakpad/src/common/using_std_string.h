// -*- mode: C++ -*-

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

// Original author: Ivan Penkov

// using_std_string.h: Allows building this code in environments where
//                     global string (::string) exists.
//
// The problem:
// -------------
// Let's say you want to build this code in an environment where a global
// string type is defined (i.e. ::string).  Now, let's suppose that ::string
// is different that std::string and you'd like to have the option to easily
// choose between the two string types.  Ideally you'd like to control which
// string type is chosen by simply #defining an identifier.
//
// The solution:
// -------------
// #define HAS_GLOBAL_STRING somewhere in a global header file and then
// globally replace std::string with string.  Then include this header
// file everywhere where string is used.  If you want to revert back to
// using std::string, simply remove the #define (HAS_GLOBAL_STRING).

#ifndef THIRD_PARTY_BREAKPAD_SRC_COMMON_USING_STD_STRING_H_
#define THIRD_PARTY_BREAKPAD_SRC_COMMON_USING_STD_STRING_H_

#ifdef HAS_GLOBAL_STRING
  typedef ::string google_breakpad_string;
#else
#include <string>
  using std::string;
  typedef std::string google_breakpad_string;
#endif

// Inicates that type google_breakpad_string is defined
#define HAS_GOOGLE_BREAKPAD_STRING

#endif  // THIRD_PARTY_BREAKPAD_SRC_COMMON_USING_STD_STRING_H_
