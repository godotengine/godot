// Copyright 2017 Google LLC
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

#include "common/long_string_dictionary.h"

#include <assert.h>
#include <string.h>

#include <algorithm>
#include <string>

#include "common/simple_string_dictionary.h"

#define arraysize(f) (sizeof(f) / sizeof(*f))

namespace {
// Suffixes for segment keys.
const char* const kSuffixes[] = {"__1", "__2", "__3", "__4", "__5", "__6",
    "__7", "__8", "__9", "__10"};
#if !defined(NDEBUG)
// The maximum suffix string length.
const size_t kMaxSuffixLength = 4;
#endif
} // namespace

namespace google_breakpad {

using std::string;

void LongStringDictionary::SetKeyValue(const char* key, const char* value) {
  assert(key);
  if (!key)
    return;

  RemoveKey(key);

  if (!value) {
    return;
  }

  // Key must not be an empty string.
  assert(key[0] != '\0');
  if (key[0] == '\0')
    return;

  // If the value is not valid for segmentation, forwards the key and the value
  // to SetKeyValue of SimpleStringDictionary and returns.
  size_t value_length = strlen(value);
  if (value_length <= (value_size - 1)) {
    SimpleStringDictionary::SetKeyValue(key, value);
    return;
  }

  size_t key_length = strlen(key);
  assert(key_length + kMaxSuffixLength <= (key_size - 1));

  char segment_key[key_size];
  char segment_value[value_size];

  strcpy(segment_key, key);

  const char* remain_value = value;
  size_t remain_value_length = strlen(value);

  for (unsigned long i = 0; i < arraysize(kSuffixes); i++) {
    if (remain_value_length == 0) {
      return;
    }

    strcpy(segment_key + key_length, kSuffixes[i]);

    size_t segment_value_length =
        std::min(remain_value_length, value_size - 1);

    strncpy(segment_value, remain_value, segment_value_length);
    segment_value[segment_value_length] = '\0';

    remain_value += segment_value_length;
    remain_value_length -= segment_value_length;

    SimpleStringDictionary::SetKeyValue(segment_key, segment_value);
  }
}

bool LongStringDictionary::RemoveKey(const char* key) {
  assert(key);
  if (!key)
    return false;

  if (SimpleStringDictionary::RemoveKey(key)) {
    return true;
  }

  size_t key_length = strlen(key);
  assert(key_length + kMaxSuffixLength <= (key_size - 1));

  char segment_key[key_size];
  strcpy(segment_key, key);

  unsigned long i = 0;
  for (; i < arraysize(kSuffixes); i++) {
    strcpy(segment_key + key_length, kSuffixes[i]);
    if (!SimpleStringDictionary::RemoveKey(segment_key)) {
      break;
    }
  }
  return i != 0;
}

const string LongStringDictionary::GetValueForKey(const char* key) const {
  assert(key);
  if (!key)
    return "";

  // Key must not be an empty string.
  assert(key[0] != '\0');
  if (key[0] == '\0')
    return "";

  const char* value = SimpleStringDictionary::GetValueForKey(key);
  if (value)
    return string(value);

  size_t key_length = strlen(key);
  assert(key_length + kMaxSuffixLength <= (key_size - 1));

  bool found_segment = false;
  char segment_key[key_size];
  string return_value;

  strcpy(segment_key, key);
  for (unsigned long i = 0; i < arraysize(kSuffixes); i++) {
    strcpy(segment_key + key_length, kSuffixes[i]);

    const char* segment_value =
        SimpleStringDictionary::GetValueForKey(segment_key);

    if (segment_value != NULL) {
      found_segment = true;
      return_value.append(segment_value);
    } else {
      break;
    }
  }

  if (found_segment) {
    return return_value;
  }
  return "";
}

}  // namespace google_breakpad
