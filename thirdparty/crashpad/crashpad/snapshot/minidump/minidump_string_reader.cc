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

#include "snapshot/minidump/minidump_string_reader.h"

#include <stdint.h>

#include "base/logging.h"
#include "minidump/minidump_extensions.h"

namespace crashpad {
namespace internal {

bool ReadMinidumpUTF8String(FileReaderInterface* file_reader,
                            RVA rva,
                            std::string* string) {
  if (rva == 0) {
    string->clear();
    return true;
  }

  if (!file_reader->SeekSet(rva)) {
    return false;
  }

  uint32_t string_size;
  if (!file_reader->ReadExactly(&string_size, sizeof(string_size))) {
    return false;
  }

  std::string local_string(string_size, '\0');
  if (!file_reader->ReadExactly(&local_string[0], string_size)) {
    return false;
  }

  string->swap(local_string);
  return true;
}

}  // namespace internal
}  // namespace crashpad
