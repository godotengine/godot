// Copyright 2006 Google LLC
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

#ifdef HAVE_CONFIG_H
#include <config.h>  // Must come first
#endif

#include "common/scoped_ptr.h"
#include "common/mac/string_utilities.h"

namespace MacStringUtils {

using google_breakpad::scoped_array;

std::string ConvertToString(CFStringRef str) {
  CFIndex length = CFStringGetLength(str);
  std::string result;

  if (!length)
    return result;

  CFIndex maxUTF8Length =
    CFStringGetMaximumSizeForEncoding(length, kCFStringEncodingUTF8);
  scoped_array<UInt8> buffer(new UInt8[maxUTF8Length + 1]);
  CFIndex actualUTF8Length;
  CFStringGetBytes(str, CFRangeMake(0, length), kCFStringEncodingUTF8, 0,
                   false, buffer.get(), maxUTF8Length, &actualUTF8Length);
  buffer[actualUTF8Length] = 0;
  result.assign((const char*)buffer.get());

  return result;
}

unsigned int IntegerValueAtIndex(string& str, unsigned int idx) {
  string digits("0123456789"), temp;
  size_t start = 0;
  size_t end;
  size_t found = 0;
  unsigned int result = 0;

  for (; found <= idx; ++found) {
    end = str.find_first_not_of(digits, start);

    if (end == string::npos)
      end = str.size();

    temp = str.substr(start, end - start);

    if (found == idx) {
      result = atoi(temp.c_str());
    }

    start = str.find_first_of(digits, end + 1);

    if (start == string::npos)
      break;
  }

  return result;
}

}  // namespace MacStringUtils
