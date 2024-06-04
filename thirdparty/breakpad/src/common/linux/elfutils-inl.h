// Copyright (c) 2012, Google Inc.
// All rights reserved.
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
//     * Neither the name of Google Inc. nor the names of its
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

#ifndef COMMON_LINUX_ELFUTILS_INL_H__
#define COMMON_LINUX_ELFUTILS_INL_H__

#include "common/linux/linux_libc_support.h"
#include "elfutils.h"

namespace google_breakpad {

template<typename ElfClass, typename T>
const T* GetOffset(const typename ElfClass::Ehdr* elf_header,
                   typename ElfClass::Off offset) {
  return reinterpret_cast<const T*>(reinterpret_cast<uintptr_t>(elf_header) +
                                    offset);
}

template<typename ElfClass>
const typename ElfClass::Shdr* FindElfSectionByName(
    const char* name,
    typename ElfClass::Word section_type,
    const typename ElfClass::Shdr* sections,
    const char* section_names,
    const char* names_end,
    int nsection) {
  assert(name != NULL);
  assert(sections != NULL);
  assert(nsection > 0);

  int name_len = my_strlen(name);
  if (name_len == 0)
    return NULL;

  for (int i = 0; i < nsection; ++i) {
    const char* section_name = section_names + sections[i].sh_name;
    if (sections[i].sh_type == section_type &&
        names_end - section_name >= name_len + 1 &&
        my_strcmp(name, section_name) == 0) {
      return sections + i;
    }
  }
  return NULL;
}

}  // namespace google_breakpad

#endif  // COMMON_LINUX_ELFUTILS_INL_H__
