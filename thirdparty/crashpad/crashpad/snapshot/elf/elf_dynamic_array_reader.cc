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

#include "snapshot/elf/elf_dynamic_array_reader.h"

#include <elf.h>

#include <type_traits>

#include "util/stdlib/map_insert.h"

namespace crashpad {

namespace {

template <typename DynType>
bool Read(const ProcessMemoryRange& memory,
          VMAddress address,
          VMSize size,
          std::map<uint64_t, uint64_t>* values) {
  std::map<uint64_t, uint64_t> local_values;

  while (size > 0) {
    DynType entry;
    if (!memory.Read(address, sizeof(entry), &entry)) {
      return false;
    }
    size -= sizeof(entry);
    address += sizeof(entry);

    switch (entry.d_tag) {
      case DT_NULL:
        values->swap(local_values);
        if (size != 0) {
          LOG(WARNING) << size << " trailing bytes not read";
        }
        return true;
      case DT_NEEDED:
        // Skip these entries for now.
        break;
      default:
        static_assert(std::is_unsigned<decltype(entry.d_un.d_ptr)>::value,
                      "type must be unsigned");
        static_assert(static_cast<void*>(&entry.d_un.d_ptr) ==
                              static_cast<void*>(&entry.d_un.d_val) &&
                          sizeof(entry.d_un.d_ptr) == sizeof(entry.d_un.d_val),
                      "d_ptr and d_val must be aliases");
        if (!MapInsertOrReplace(
                &local_values, entry.d_tag, entry.d_un.d_ptr, nullptr)) {
          LOG(ERROR) << "duplicate dynamic array entry";
          return false;
        }
    }
  }
  LOG(ERROR) << "missing DT_NULL";
  return false;
}

}  // namespace

ElfDynamicArrayReader::ElfDynamicArrayReader() : values_() {}

ElfDynamicArrayReader::~ElfDynamicArrayReader() {}

bool ElfDynamicArrayReader::Initialize(const ProcessMemoryRange& memory,
                                       VMAddress address,
                                       VMSize size) {
  return memory.Is64Bit() ? Read<Elf64_Dyn>(memory, address, size, &values_)
                          : Read<Elf32_Dyn>(memory, address, size, &values_);
}

}  // namespace crashpad
