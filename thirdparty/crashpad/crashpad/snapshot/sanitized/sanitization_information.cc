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

#include "snapshot/sanitized/sanitization_information.h"

#include "client/annotation.h"

namespace crashpad {

namespace {

template <typename Pointer>
bool ReadWhitelist(const ProcessMemoryRange& memory,
                   VMAddress whitelist_address,
                   std::vector<std::string>* whitelist) {
  if (!whitelist_address) {
    return true;
  }

  std::vector<std::string> local_whitelist;
  Pointer name_address;
  while (memory.Read(whitelist_address, sizeof(name_address), &name_address)) {
    if (!name_address) {
      whitelist->swap(local_whitelist);
      return true;
    }

    std::string name;
    if (!memory.ReadCStringSizeLimited(
            name_address, Annotation::kNameMaxLength, &name)) {
      return false;
    }
    local_whitelist.push_back(name);
    whitelist_address += sizeof(Pointer);
  }

  return false;
}

}  // namespace

bool ReadAnnotationsWhitelist(const ProcessMemoryRange& memory,
                              VMAddress whitelist_address,
                              std::vector<std::string>* whitelist) {
  return memory.Is64Bit()
             ? ReadWhitelist<uint64_t>(memory, whitelist_address, whitelist)
             : ReadWhitelist<uint32_t>(memory, whitelist_address, whitelist);
}

}  // namespace crashpad
