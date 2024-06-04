// -*- mode: c++ -*-

// Copyright (c) 2011 Google Inc. All Rights Reserved.
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

// Original author: Ted Mielczarek <ted.mielczarek@gmail.com>

// elf_symbols_to_module.h: Exposes ELFSymbolsToModule, a function
// for reading ELF symbol tables and inserting exported symbol names
// into a google_breakpad::Module as Extern definitions.

#ifndef BREAKPAD_COMMON_LINUX_ELF_SYMBOLS_TO_MODULE_H_
#define BREAKPAD_COMMON_LINUX_ELF_SYMBOLS_TO_MODULE_H_

#include <stddef.h>
#include <stdint.h>

namespace google_breakpad {

class Module;

bool ELFSymbolsToModule(const uint8_t* symtab_section,
                        size_t symtab_size,
                        const uint8_t* string_section,
                        size_t string_size,
                        const bool big_endian,
                        size_t value_size,
                        Module* module);

}  // namespace google_breakpad


#endif  // BREAKPAD_COMMON_LINUX_ELF_SYMBOLS_TO_MODULE_H_
