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

#include "snapshot/minidump/minidump_string_list_reader.h"

#include <stdint.h>

#include "base/logging.h"
#include "minidump/minidump_extensions.h"
#include "snapshot/minidump/minidump_string_reader.h"

namespace crashpad {
namespace internal {

bool ReadMinidumpStringList(FileReaderInterface* file_reader,
                            const MINIDUMP_LOCATION_DESCRIPTOR& location,
                            std::vector<std::string>* list) {
  if (location.Rva == 0) {
    list->clear();
    return true;
  }

  if (location.DataSize < sizeof(MinidumpRVAList)) {
    LOG(ERROR) << "string_list size mismatch";
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
      sizeof(MinidumpRVAList) + entry_count * sizeof(RVA)) {
    LOG(ERROR) << "string_list size mismatch";
    return false;
  }

  std::vector<RVA> rvas(entry_count);
  if (!file_reader->ReadExactly(&rvas[0], entry_count * sizeof(rvas[0]))) {
    return false;
  }

  std::vector<std::string> local_list;
  for (RVA rva : rvas) {
    std::string element;
    if (!ReadMinidumpUTF8String(file_reader, rva, &element)) {
      return false;
    }

    local_list.push_back(element);
  }

  list->swap(local_list);
  return true;
}

}  // namespace internal
}  // namespace crashpad
