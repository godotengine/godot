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

#ifndef COMMON_LONG_STRING_DICTIONARY_H_
#define COMMON_LONG_STRING_DICTIONARY_H_

#include <string>

#include "common/simple_string_dictionary.h"

namespace google_breakpad {
// key_size is the maxium size that |key| can take in
// SimpleStringDictionary which is defined in simple_string_dictionary.h.
//
// value_size is the maxium size that |value| can take in
// SimpleStringDictionary which is defined in simple_string_dictionary.h.
//
// LongStringDictionary is a subclass of SimpleStringDictionary which supports
// longer values to be stored in the dictionary. The maximum length supported is
// (value_size - 1) * 10.
//
// For example, LongStringDictionary will store long value with key 'abc' into
// segment values with segment keys 'abc__1', 'abc__2', 'abc__3', ...
//
// Clients must avoid using the same suffixes as their key's suffix when
// LongStringDictionary is used.
class LongStringDictionary : public SimpleStringDictionary {
 public:
  // Stores |value| into |key|, or segment values into segment keys. The maxium
  // number of segments is 10. If |value| can not be stored in 10 segments, it
  // will be truncated. Replacing the existing value if |key| is already present
  // and replacing the existing segment values if segment keys are already
  // present.
  //
  // |key| must not be NULL. If the |value| need to be divided into segments,
  // the lengh of |key| must be smaller enough so that lengths of segment keys
  // which are key with suffixes are all samller than (key_size - 1). Currently,
  // the max length of suffixes are 4.
  //
  // If |value| is NULL, the key and its corresponding segment keys are removed
  // from the map. If there is no more space in the map, then the operation
  // silently fails.
  void SetKeyValue(const char* key, const char* value);

  // Given |key|, removes any associated value or associated segment values.
  // |key| must not be NULL. If the key is not found, searchs its segment keys
  // and removes corresponding segment values if found.
  bool RemoveKey(const char* key);

  // Given |key|, returns its corresponding |value|. |key| must not be NULL. If
  // the key is found, its corresponding |value| is returned.
  //
  // If no corresponding |value| is found, segment keys of the given |key| will
  // be used to search for corresponding segment values. If segment values
  // exist, assembled value from them is returned. If no segment value exists,
  // NULL is returned.
  const std::string GetValueForKey(const char* key) const;
};
} // namespace google_breakpad

#endif // COMMON_LONG_STRING_DICTIONARY_H_
