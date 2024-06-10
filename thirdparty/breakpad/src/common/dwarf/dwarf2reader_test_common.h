// -*- mode: c++ -*-

// Copyright 2012 Google LLC
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

// Original author: Jim Blandy <jimb@mozilla.com> <jimb@red-bean.com>

// dwarf2reader_test_common.h: Define TestCompilationUnit and
// TestAbbrevTable, classes for creating properly (and improperly)
// formatted DWARF compilation unit data for unit tests.

#ifndef COMMON_DWARF_DWARF2READER_TEST_COMMON_H__
#define COMMON_DWARF_DWARF2READER_TEST_COMMON_H__

#include "common/test_assembler.h"
#include "common/dwarf/dwarf2enums.h"

// A subclass of test_assembler::Section, specialized for constructing
// DWARF compilation units.
class TestCompilationUnit: public google_breakpad::test_assembler::Section {
 public:
  typedef google_breakpad::DwarfTag DwarfTag;
  typedef google_breakpad::DwarfAttribute DwarfAttribute;
  typedef google_breakpad::DwarfForm DwarfForm;
  typedef google_breakpad::test_assembler::Label Label;

  // Set the section's DWARF format size (the 32-bit DWARF format or the
  // 64-bit DWARF format, for lengths and section offsets --- not the
  // address size) to format_size.
  void set_format_size(size_t format_size) {
    assert(format_size == 4 || format_size == 8);
    format_size_ = format_size;
  }

  // Append a DWARF section offset value, of the appropriate size for this
  // compilation unit.
  template<typename T>
  void SectionOffset(T offset) {
    if (format_size_ == 4)
      D32(offset);
    else
      D64(offset);
  }

  // Append a DWARF compilation unit header to the section, with the given
  // DWARF version, abbrev table offset, and address size.
  TestCompilationUnit& Header(int version, const Label& abbrev_offset,
                              size_t address_size, int header_type) {
    if (format_size_ == 4) {
      D32(length_);
    } else {
      D32(0xffffffff);
      D64(length_);
    }
    post_length_offset_ = Size();
    D16(version);
    if (version <= 4) {
      SectionOffset(abbrev_offset);
      D8(address_size);
    } else {
      D8(header_type);  // DW_UT_compile, DW_UT_type, etc.
      D8(address_size);
      SectionOffset(abbrev_offset);
      if (header_type == google_breakpad::DW_UT_type) {
        uint64_t dummy_type_signature = 0xdeadbeef;
        uint64_t dummy_type_offset = 0x2b;
        D64(dummy_type_signature);
        if (format_size_ == 4)
          D32(dummy_type_offset);
        else
          D64(dummy_type_offset);
      }
    }
    return *this;
  }

  // Mark the end of this header's DIEs.
  TestCompilationUnit& Finish() {
    length_ = Size() - post_length_offset_;
    return *this;
  }

 private:
  // The DWARF format size for this compilation unit.
  size_t format_size_;

  // The offset of the point in the compilation unit header immediately
  // after the initial length field.
  uint64_t post_length_offset_;

  // The length of the compilation unit, not including the initial length field.
  Label length_;
};

// A subclass of test_assembler::Section specialized for constructing DWARF
// abbreviation tables.
class TestAbbrevTable: public google_breakpad::test_assembler::Section {
 public:
  typedef google_breakpad::DwarfTag DwarfTag;
  typedef google_breakpad::DwarfAttribute DwarfAttribute;
  typedef google_breakpad::DwarfForm DwarfForm;
  typedef google_breakpad::DwarfHasChild DwarfHasChild;
  typedef google_breakpad::test_assembler::Label Label;

  // Start a new abbreviation table entry for abbreviation code |code|,
  // encoding a DIE whose tag is |tag|, and which has children if and only
  // if |has_children| is true.
  TestAbbrevTable& Abbrev(int code, DwarfTag tag, DwarfHasChild has_children) {
    assert(code != 0);
    ULEB128(code);
    ULEB128(static_cast<unsigned>(tag));
    D8(static_cast<unsigned>(has_children));
    return *this;
  };

  // Add an attribute to the current abbreviation code whose name is |name|
  // and whose form is |form|.
  TestAbbrevTable& Attribute(DwarfAttribute name, DwarfForm form) {
    ULEB128(static_cast<unsigned>(name));
    ULEB128(static_cast<unsigned>(form));
    return *this;
  }

  // Finish the current abbreviation code.
  TestAbbrevTable& EndAbbrev() {
    ULEB128(0);
    ULEB128(0);
    return *this;
  }

  // Finish the current abbreviation table.
  TestAbbrevTable& EndTable() {
    ULEB128(0);
    return *this;
  }
};

#endif // COMMON_DWARF_DWARF2READER_TEST_COMMON_H__
