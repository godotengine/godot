// Copyright 2017 The Crashpad Authors. All rights reserved.
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

#include "util/misc/reinterpret_bytes.h"

#include <string.h>

#include <algorithm>

#include "base/logging.h"

namespace crashpad {
namespace internal {

bool ReinterpretBytesImpl(const char* data,
                          size_t data_size,
                          char* dest,
                          size_t dest_size) {
  // Verify that any unused bytes from data are zero.
  // The unused bytes are at the start of the data buffer for big-endian and the
  // end of the buffer for little-endian.
  if (dest_size < data_size) {
    auto extra_bytes = data;
#if defined(ARCH_CPU_LITTLE_ENDIAN)
    extra_bytes += dest_size;
#endif  // ARCH_CPU_LITTLE_ENDIAN

    uint64_t zero = 0;
    size_t extra_bytes_size = data_size - dest_size;
    while (extra_bytes_size > 0) {
      size_t to_check = std::min(extra_bytes_size, sizeof(zero));
      if (memcmp(extra_bytes, &zero, to_check) != 0) {
        LOG(ERROR) << "information loss";
        return false;
      }
      extra_bytes += to_check;
      extra_bytes_size -= to_check;
    }
  }

  // Zero out the destination, in case it is larger than data.
  memset(dest, 0, dest_size);

#if defined(ARCH_CPU_LITTLE_ENDIAN)
  // Copy a prefix of data to a prefix of dest for little-endian
  memcpy(dest, data, std::min(dest_size, data_size));
#else
  // or the suffix of data to the suffix of dest for big-endian
  if (data_size >= dest_size) {
    memcpy(dest, data + data_size - dest_size, dest_size);
  } else {
    memcpy(dest + dest_size - data_size, data, data_size);
  }
#endif  // ARCH_CPU_LITTLE_ENDIAN
  return true;
}

}  // namespace internal
}  // namespace crashpad
