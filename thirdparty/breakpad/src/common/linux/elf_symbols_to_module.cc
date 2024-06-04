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

#include "common/linux/elf_symbols_to_module.h"

#include <cxxabi.h>
#include <elf.h>
#include <string.h>

#include "common/byte_cursor.h"
#include "common/module.h"

namespace google_breakpad {

class ELFSymbolIterator {
public:
  // The contents of an ELF symbol, adjusted for the host's endianness,
  // word size, and so on. Corresponds to the data in Elf32_Sym / Elf64_Sym.
  struct Symbol {
    // True if this iterator has reached the end of the symbol array. When
    // this is set, the other members of this structure are not valid.
    bool at_end;

    // The number of this symbol within the list.
    size_t index;

    // The current symbol's name offset. This is the offset within the
    // string table.
    size_t name_offset;

    // The current symbol's value, size, info and shndx fields.
    uint64_t value;
    uint64_t size;
    unsigned char info;
    uint16_t shndx;
  };

  // Create an ELFSymbolIterator walking the symbols in BUFFER. Treat the
  // symbols as big-endian if BIG_ENDIAN is true, as little-endian
  // otherwise. Assume each symbol has a 'value' field whose size is
  // VALUE_SIZE.
  //
  ELFSymbolIterator(const ByteBuffer* buffer, bool big_endian,
                    size_t value_size)
    : value_size_(value_size), cursor_(buffer, big_endian) {
    // Actually, weird sizes could be handled just fine, but they're
    // probably mistakes --- expressed in bits, say.
    assert(value_size == 4 || value_size == 8);
    symbol_.index = 0;
    Fetch();
  }

  // Move to the next symbol. This function's behavior is undefined if
  // at_end() is true when it is called.
  ELFSymbolIterator& operator++() { Fetch(); symbol_.index++; return *this; }

  // Dereferencing this iterator produces a reference to an Symbol structure
  // that holds the current symbol's values. The symbol is owned by this
  // SymbolIterator, and will be invalidated at the next call to operator++.
  const Symbol& operator*() const { return symbol_; }
  const Symbol* operator->() const { return &symbol_; }

private:
  // Read the symbol at cursor_, and set symbol_ appropriately.
  void Fetch() {
    // Elf32_Sym and Elf64_Sym have different layouts.
    unsigned char other;
    if (value_size_ == 4) {
      // Elf32_Sym
      cursor_
        .Read(4, false, &symbol_.name_offset)
        .Read(4, false, &symbol_.value)
        .Read(4, false, &symbol_.size)
        .Read(1, false, &symbol_.info)
        .Read(1, false, &other)
        .Read(2, false, &symbol_.shndx);
    } else {
      // Elf64_Sym
      cursor_
        .Read(4, false, &symbol_.name_offset)
        .Read(1, false, &symbol_.info)
        .Read(1, false, &other)
        .Read(2, false, &symbol_.shndx)
        .Read(8, false, &symbol_.value)
        .Read(8, false, &symbol_.size);
    }
    symbol_.at_end = !cursor_;
  }

  // The size of symbols' value field, in bytes.
  size_t value_size_;

  // A byte cursor traversing buffer_.
  ByteCursor cursor_;

  // Values for the symbol this iterator refers to.
  Symbol symbol_;
};

const char* SymbolString(ptrdiff_t offset, ByteBuffer& strings) {
  if (offset < 0 || (size_t) offset >= strings.Size()) {
    // Return the null string.
    offset = 0;
  }
  return reinterpret_cast<const char*>(strings.start + offset);
}

bool ELFSymbolsToModule(const uint8_t* symtab_section,
                        size_t symtab_size,
                        const uint8_t* string_section,
                        size_t string_size,
                        const bool big_endian,
                        size_t value_size,
                        Module* module) {
  ByteBuffer symbols(symtab_section, symtab_size);
  // Ensure that the string section is null-terminated.
  if (string_section[string_size - 1] != '\0') {
    const void* null_terminator = memrchr(string_section, '\0', string_size);
    string_size = reinterpret_cast<const uint8_t*>(null_terminator)
      - string_section;
  }
  ByteBuffer strings(string_section, string_size);

  // The iterator walking the symbol table.
  ELFSymbolIterator iterator(&symbols, big_endian, value_size);

  while(!iterator->at_end) {
    if (ELF32_ST_TYPE(iterator->info) == STT_FUNC &&
        iterator->shndx != SHN_UNDEF) {
      Module::Extern* ext = new Module::Extern(iterator->value);
      ext->name = SymbolString(iterator->name_offset, strings);
#if !defined(__ANDROID__)  // Android NDK doesn't provide abi::__cxa_demangle.
      int status = 0;
      char* demangled =
          abi::__cxa_demangle(ext->name.c_str(), NULL, NULL, &status);
      if (demangled) {
        if (status == 0)
          ext->name = demangled;
        free(demangled);
      }
#endif
      module->AddExtern(ext);
    }
    ++iterator;
  }
  return true;
}

}  // namespace google_breakpad
