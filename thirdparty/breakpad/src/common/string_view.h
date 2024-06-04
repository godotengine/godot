// Copyright (c) 2021 Google Inc.
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

#ifndef COMMON_STRING_VIEW_H__
#define COMMON_STRING_VIEW_H__

#include <cassert>
#include <cstring>
#include <ostream>
#include "common/using_std_string.h"

namespace google_breakpad {

// A StringView is a string reference to a string object, but not own the
// string object. It's a compatibile layer until we can use std::string_view in
// C++17.
class StringView {
 private:
  // The start of the string, in an external buffer. It doesn't have to be
  // null-terminated.
  const char* data_ = "";

  size_t length_ = 0;

 public:
  // Construct an empty StringView.
  StringView() = default;

  // Disallow construct StringView from nullptr.
  StringView(std::nullptr_t) = delete;

  // Construct a StringView from a cstring.
  StringView(const char* str) : data_(str) {
    assert(str);
    length_ = strlen(str);
  }

  // Construct a StringView from a cstring with fixed length.
  StringView(const char* str, size_t length) : data_(str), length_(length) {
    assert(str);
  }

  // Construct a StringView from an std::string.
  StringView(const string& str) : data_(str.data()), length_(str.length()) {}

  string str() const { return string(data_, length_); }

  const char* data() const { return data_; }

  bool empty() const { return length_ == 0; }

  size_t size() const { return length_; }

  int compare(StringView rhs) const {
    size_t min_len = std::min(size(), rhs.size());
    int res = memcmp(data_, rhs.data(), min_len);
    if (res != 0)
      return res;
    if (size() == rhs.size())
      return 0;
    return size() < rhs.size() ? -1 : 1;
  }
};

inline bool operator==(StringView lhs, StringView rhs) {
  return lhs.compare(rhs) == 0;
}

inline bool operator!=(StringView lhs, StringView rhs) {
  return lhs.compare(rhs) != 0;
}

inline bool operator<(StringView lhs, StringView rhs) {
  return lhs.compare(rhs) < 0;
}

inline bool operator>(StringView lhs, StringView rhs) {
  return lhs.compare(rhs) > 0;
}

inline std::ostream& operator<<(std::ostream& os, StringView s) {
  os << s.str();
  return os;
}

}  // namespace google_breakpad

#endif  // COMMON_STRING_VIEW_H__
