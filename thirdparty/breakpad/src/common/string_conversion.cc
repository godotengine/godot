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

#include <string.h>

#include "common/convert_UTF.h"
#include "common/scoped_ptr.h"
#include "common/string_conversion.h"
#include "common/using_std_string.h"

namespace google_breakpad {

using std::vector;

void UTF8ToUTF16(const char* in, vector<uint16_t>* out) {
  size_t source_length = strlen(in);
  const UTF8* source_ptr = reinterpret_cast<const UTF8*>(in);
  const UTF8* source_end_ptr = source_ptr + source_length;
  // Erase the contents and zero fill to the expected size
  out->clear();
  out->insert(out->begin(), source_length, 0);
  uint16_t* target_ptr = &(*out)[0];
  uint16_t* target_end_ptr = target_ptr + out->capacity();
  ConversionResult result = ConvertUTF8toUTF16(&source_ptr, source_end_ptr,
                                               &target_ptr, target_end_ptr,
                                               strictConversion);

  // Resize to be the size of the # of converted characters + NULL
  out->resize(result == conversionOK ? target_ptr - &(*out)[0] + 1: 0);
}

int UTF8ToUTF16Char(const char* in, int in_length, uint16_t out[2]) {
  const UTF8* source_ptr = reinterpret_cast<const UTF8*>(in);
  const UTF8* source_end_ptr = source_ptr + 1;
  uint16_t* target_ptr = out;
  uint16_t* target_end_ptr = target_ptr + 2;
  out[0] = out[1] = 0;

  // Process one character at a time
  while (1) {
    ConversionResult result = ConvertUTF8toUTF16(&source_ptr, source_end_ptr,
                                                 &target_ptr, target_end_ptr,
                                                 strictConversion);

    if (result == conversionOK)
      return static_cast<int>(source_ptr - reinterpret_cast<const UTF8*>(in));

    // Add another character to the input stream and try again
    source_ptr = reinterpret_cast<const UTF8*>(in);
    ++source_end_ptr;

    if (source_end_ptr > reinterpret_cast<const UTF8*>(in) + in_length)
      break;
  }

  return 0;
}

void UTF32ToUTF16(const wchar_t* in, vector<uint16_t>* out) {
  size_t source_length = wcslen(in);
  const UTF32* source_ptr = reinterpret_cast<const UTF32*>(in);
  const UTF32* source_end_ptr = source_ptr + source_length;
  // Erase the contents and zero fill to the expected size
  out->clear();
  out->insert(out->begin(), source_length, 0);
  uint16_t* target_ptr = &(*out)[0];
  uint16_t* target_end_ptr = target_ptr + out->capacity();
  ConversionResult result = ConvertUTF32toUTF16(&source_ptr, source_end_ptr,
                                                &target_ptr, target_end_ptr,
                                                strictConversion);

  // Resize to be the size of the # of converted characters + NULL
  out->resize(result == conversionOK ? target_ptr - &(*out)[0] + 1: 0);
}

void UTF32ToUTF16Char(wchar_t in, uint16_t out[2]) {
  const UTF32* source_ptr = reinterpret_cast<const UTF32*>(&in);
  const UTF32* source_end_ptr = source_ptr + 1;
  uint16_t* target_ptr = out;
  uint16_t* target_end_ptr = target_ptr + 2;
  out[0] = out[1] = 0;
  ConversionResult result = ConvertUTF32toUTF16(&source_ptr, source_end_ptr,
                                                &target_ptr, target_end_ptr,
                                                strictConversion);

  if (result != conversionOK) {
    out[0] = out[1] = 0;
  }
}

static inline uint16_t Swap(uint16_t value) {
  return (value >> 8) | static_cast<uint16_t>(value << 8);
}

string UTF16ToUTF8(const vector<uint16_t>& in, bool swap) {
  const UTF16* source_ptr = &in[0];
  scoped_array<uint16_t> source_buffer;

  // If we're to swap, we need to make a local copy and swap each byte pair
  if (swap) {
    int idx = 0;
    source_buffer.reset(new uint16_t[in.size()]);
    UTF16* source_buffer_ptr = source_buffer.get();
    for (vector<uint16_t>::const_iterator it = in.begin();
         it != in.end(); ++it, ++idx)
      source_buffer_ptr[idx] = Swap(*it);

    source_ptr = source_buffer.get();
  }

  // The maximum expansion would be 4x the size of the input string.
  const UTF16* source_end_ptr = source_ptr + in.size();
  size_t target_capacity = in.size() * 4;
  scoped_array<UTF8> target_buffer(new UTF8[target_capacity]);
  UTF8* target_ptr = target_buffer.get();
  UTF8* target_end_ptr = target_ptr + target_capacity;
  ConversionResult result = ConvertUTF16toUTF8(&source_ptr, source_end_ptr,
                                               &target_ptr, target_end_ptr,
                                               strictConversion);

  if (result == conversionOK) {
    const char* targetPtr = reinterpret_cast<const char*>(target_buffer.get());
    return targetPtr;
  }

  return "";
}

}  // namespace google_breakpad
