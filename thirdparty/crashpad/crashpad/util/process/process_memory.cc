// Copyright 2018 The Crashpad Authors. All rights reserved.
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

#include "util/process/process_memory.h"

#include "base/logging.h"

namespace crashpad {

bool ProcessMemory::Read(VMAddress address, size_t size, void* buffer) const {
  char* buffer_c = static_cast<char*>(buffer);
  while (size > 0) {
    ssize_t bytes_read = ReadUpTo(address, size, buffer_c);
    if (bytes_read < 0) {
      return false;
    }
    if (bytes_read == 0) {
      LOG(ERROR) << "short read";
      return false;
    }
    DCHECK_LE(static_cast<size_t>(bytes_read), size);
    size -= bytes_read;
    address += bytes_read;
    buffer_c += bytes_read;
  }
  return true;
}

bool ProcessMemory::ReadCStringInternal(VMAddress address,
                                        bool has_size,
                                        size_t size,
                                        std::string* string) const {
  string->clear();

  char buffer[4096];
  do {
    size_t read_size;
    if (has_size) {
      read_size = std::min(sizeof(buffer), size);
    } else {
      read_size = sizeof(buffer);
    }

    ssize_t bytes_read = ReadUpTo(address, read_size, buffer);
    if (bytes_read < 0) {
      return false;
    }
    if (bytes_read == 0) {
      break;
    }

    char* nul = static_cast<char*>(memchr(buffer, '\0', bytes_read));
    if (nul != nullptr) {
      string->append(buffer, nul - buffer);
      return true;
    }
    string->append(buffer, bytes_read);

    address += bytes_read;
    size -= bytes_read;
  } while (!has_size || size > 0);

  LOG(ERROR) << "unterminated string";
  return false;
}

}  // namespace crashpad
