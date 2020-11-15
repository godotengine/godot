// Copyright 2015 The Crashpad Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "snapshot/minidump/minidump_simple_string_dictionary_reader.h"

#include <stdint.h>

#include <utility>
#include <vector>

#include "base/logging.h"
#include "minidump/minidump_extensions.h"
#include "snapshot/minidump/minidump_string_reader.h"

namespace crashpad {
namespace internal {

bool ReadMinidumpSimpleStringDictionary(
    FileReaderInterface* file_reader,
    const MINIDUMP_LOCATION_DESCRIPTOR& location,
    std::map<std::string, std::string>* dictionary) {
  if (location.Rva == 0) {
    dictionary->clear();
    return true;
  }

  if (location.DataSize < sizeof(MinidumpSimpleStringDictionary)) {
    LOG(ERROR) << "simple_string_dictionary size mismatch";
    return false;
  }

  if (!file_reader->SeekSet(location.Rva)) {
    return false;
  }

  uint32_t entry_count;
  if (!file_reader->ReadExactly(&entry_count, sizeof(entry_count))) {
    return false;
  }

  if (location.DataSize !=
      sizeof(MinidumpSimpleStringDictionary) +
          entry_count * sizeof(MinidumpSimpleStringDictionaryEntry)) {
    LOG(ERROR) << "simple_string_dictionary size mismatch";
    return false;
  }

  std::vector<MinidumpSimpleStringDictionaryEntry> entries(entry_count);
  if (!file_reader->ReadExactly(&entries[0],
                                entry_count * sizeof(entries[0]))) {
    return false;
  }

  std::map<std::string, std::string> local_dictionary;
  for (const MinidumpSimpleStringDictionaryEntry& entry : entries) {
    std::string key;
    if (!ReadMinidumpUTF8String(file_reader, entry.key, &key)) {
      return false;
    }

    std::string value;
    if (!ReadMinidumpUTF8String(file_reader, entry.value, &value)) {
      return false;
    }

    if (!local_dictionary.insert(std::make_pair(key, value)).second) {
      LOG(ERROR) << "duplicate key " << key;
      return false;
    }
  }

  dictionary->swap(local_dictionary);
  return true;
}

}  // namespace internal
}  // namespace crashpad
