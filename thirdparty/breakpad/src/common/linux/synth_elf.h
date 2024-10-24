// -*- mode: C++ -*-

// Copyright 2011 Google LLC
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
//     * Neither the name of Google LLC nor the names of its
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

// synth_elf.h: Interface to synth_elf::ELF: fake ELF generator.

#ifndef COMMON_LINUX_SYNTH_ELF_H_
#define COMMON_LINUX_SYNTH_ELF_H_

#include "common/test_assembler.h"

#include <list>
#include <vector>
#include <map>
#include <string>
#include <utility>

#include "common/using_std_string.h"

namespace google_breakpad {
namespace synth_elf {

using std::list;
using std::vector;
using std::map;
using std::pair;
using test_assembler::Endianness;
using test_assembler::kLittleEndian;
using test_assembler::kUnsetEndian;
using test_assembler::Label;
using test_assembler::Section;

// String tables are common in ELF headers, so subclass Section
// to make them easy to generate.
class StringTable : public Section {
public:
  StringTable(Endianness endianness = kUnsetEndian)
  : Section(endianness) {
    start() = 0;
    empty_string = Add("");
  }

  // Add the string s to the string table, and return
  // a label containing the offset into the string table
  // at which it was added.
  Label Add(const string& s) {
    if (strings_.find(s) != strings_.end())
      return strings_[s];

    Label string_label(Here());
    AppendCString(s);
    strings_[s] = string_label;
    return string_label;
  }

  // All StringTables contain an empty string as their first
  // entry.
  Label empty_string;

  // Avoid inserting duplicate strings.
  map<string,Label> strings_;
};

// A Section representing an entire ELF file.
class ELF : public Section {
 public:
  ELF(uint16_t machine,    // EM_386, etc
      uint8_t file_class,  // ELFCLASS{32,64}
      Endianness endianness = kLittleEndian);

  // Add the Section section to the section header table and append it
  // to the file. Returns the index of the section in the section
  // header table.
  int AddSection(const string& name, const Section& section,
                 uint32_t type, uint32_t flags = 0, uint64_t addr = 0,
                 uint32_t link = 0, uint64_t entsize = 0, uint64_t offset = 0);
                  
  // Add a segment containing from section index start to section index end.
  // The indexes must have been gotten from AddSection.
  void AddSegment(int start, int end, uint32_t type, uint32_t flags = 0);

  // Write out all data. GetContents may be used after this.
  void Finish();

 private:
  // Size of an address, in bytes.
  const size_t addr_size_;

  // Offset to the program header table.
  Label program_header_label_;
  // Number of entries in the program header table.
  int program_count_;
  Label program_count_label_;
  // The program header table itself.
  Section program_header_table_;

  // Offset to the section header table.
  Label section_header_label_;
  // Number of entries in the section header table.
  int section_count_;
  Label section_count_label_;
  // The section header table itself.
  Section section_header_table_;

  // Index of the section header string table in the section
  // header table.
  Label section_header_string_index_;
  // Section containing the names of section header table entries.
  StringTable section_header_strings_;

  // Record of an added section
  struct ElfSection : public Section {
    ElfSection(const Section& section, uint32_t type, uint32_t addr,
               uint32_t offset, Label offset_label, uint32_t size)
    : Section(section), type_(type), addr_(addr), offset_(offset)
    , offset_label_(offset_label), size_(size) {
    }

    uint32_t type_;
    uint32_t addr_;
    uint32_t offset_;
    Label offset_label_;
    uint32_t size_;
  };

  vector<ElfSection> sections_;

  void AppendSection(ElfSection& section);
};

// A class to build .symtab or .dynsym sections.
class SymbolTable : public Section {
 public:
  // table is the StringTable that contains symbol names. The caller
  // must ensure that it remains alive for the life of the
  // SymbolTable.
  SymbolTable(Endianness endianness, size_t addr_size, StringTable& table);

  // Add an Elf32_Sym.
  void AddSymbol(const string& name, uint32_t value,
                 uint32_t size, unsigned info, uint16_t shndx);
  // Add an Elf64_Sym.
  void AddSymbol(const string& name, uint64_t value,
                 uint64_t size, unsigned info, uint16_t shndx);

 private:
#ifndef NDEBUG
  size_t addr_size_;
#endif
  StringTable& table_;
};

// A class for note sections
class Notes : public Section {
public:
  Notes(Endianness endianness)
  : Section(endianness) {
  }

  // Add a note.
  void AddNote(int type, const string& name, const uint8_t* desc_bytes,
               size_t desc_size);
};

}  // namespace synth_elf
}  // namespace google_breakpad

#endif  // COMMON_LINUX_SYNTH_ELF_H_
