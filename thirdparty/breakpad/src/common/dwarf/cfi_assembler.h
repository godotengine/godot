// -*- mode: C++ -*-

// Copyright 2010 Google LLC
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

// cfi_assembler.h: Define CFISection, a class for creating properly
// (and improperly) formatted DWARF CFI data for unit tests.

#ifndef PROCESSOR_CFI_ASSEMBLER_H_
#define PROCESSOR_CFI_ASSEMBLER_H_

#include <string>

#include "common/dwarf/dwarf2enums.h"
#include "common/test_assembler.h"
#include "common/using_std_string.h"
#include "google_breakpad/common/breakpad_types.h"

namespace google_breakpad {

using google_breakpad::test_assembler::Label;
using google_breakpad::test_assembler::Section;

class CFISection: public Section {
 public:

  // CFI augmentation strings beginning with 'z', defined by the
  // Linux/IA-64 C++ ABI, can specify interesting encodings for
  // addresses appearing in FDE headers and call frame instructions (and
  // for additional fields whose presence the augmentation string
  // specifies). In particular, pointers can be specified to be relative
  // to various base address: the start of the .text section, the
  // location holding the address itself, and so on. These allow the
  // frame data to be position-independent even when they live in
  // write-protected pages. These variants are specified at the
  // following two URLs:
  //
  // http://refspecs.linux-foundation.org/LSB_4.0.0/LSB-Core-generic/LSB-Core-generic/dwarfext.html
  // http://refspecs.linux-foundation.org/LSB_4.0.0/LSB-Core-generic/LSB-Core-generic/ehframechpt.html
  //
  // CFISection leaves the production of well-formed 'z'-augmented CIEs and
  // FDEs to the user, but does provide EncodedPointer, to emit
  // properly-encoded addresses for a given pointer encoding.
  // EncodedPointer uses an instance of this structure to find the base
  // addresses it should use; you can establish a default for all encoded
  // pointers appended to this section with SetEncodedPointerBases.
  struct EncodedPointerBases {
    EncodedPointerBases() : cfi(), text(), data() { }

    // The starting address of this CFI section in memory, for
    // DW_EH_PE_pcrel. DW_EH_PE_pcrel pointers may only be used in data
    // that has is loaded into the program's address space.
    uint64_t cfi;

    // The starting address of this file's .text section, for DW_EH_PE_textrel.
    uint64_t text;

    // The starting address of this file's .got or .eh_frame_hdr section,
    // for DW_EH_PE_datarel.
    uint64_t data;
  };

  // Create a CFISection whose endianness is ENDIANNESS, and where
  // machine addresses are ADDRESS_SIZE bytes long. If EH_FRAME is
  // true, use the .eh_frame format, as described by the Linux
  // Standards Base Core Specification, instead of the DWARF CFI
  // format.
  CFISection(google_breakpad::test_assembler::Endianness endianness, size_t address_size,
             bool eh_frame = false)
      : Section(endianness), address_size_(address_size), eh_frame_(eh_frame),
        pointer_encoding_(DW_EH_PE_absptr),
        encoded_pointer_bases_(), entry_length_(NULL), in_fde_(false) {
    // The 'start', 'Here', and 'Mark' members of a CFISection all refer
    // to section offsets.
    start() = 0;
  }

  // Return this CFISection's address size.
  size_t AddressSize() const { return address_size_; }

  // Return true if this CFISection uses the .eh_frame format, or
  // false if it contains ordinary DWARF CFI data.
  bool ContainsEHFrame() const { return eh_frame_; }

  // Use ENCODING for pointers in calls to FDEHeader and EncodedPointer.
  void SetPointerEncoding(DwarfPointerEncoding encoding) {
    pointer_encoding_ = encoding;
  }

  // Use the addresses in BASES as the base addresses for encoded
  // pointers in subsequent calls to FDEHeader or EncodedPointer.
  // This function makes a copy of BASES.
  void SetEncodedPointerBases(const EncodedPointerBases& bases) {
    encoded_pointer_bases_ = bases;
  }

  // Append a Common Information Entry header to this section with the
  // given values. If dwarf64 is true, use the 64-bit DWARF initial
  // length format for the CIE's initial length. Return a reference to
  // this section. You should call FinishEntry after writing the last
  // instruction for the CIE.
  //
  // Before calling this function, you will typically want to use Mark
  // or Here to make a label to pass to FDEHeader that refers to this
  // CIE's position in the section.
  CFISection& CIEHeader(uint64_t code_alignment_factor,
                        int data_alignment_factor,
                        unsigned return_address_register,
                        uint8_t version = 3,
                        const string& augmentation = "",
                        bool dwarf64 = false,
                        uint8_t address_size = 8,
                        uint8_t segment_size = 0);

  // Append a Frame Description Entry header to this section with the
  // given values. If dwarf64 is true, use the 64-bit DWARF initial
  // length format for the CIE's initial length. Return a reference to
  // this section. You should call FinishEntry after writing the last
  // instruction for the CIE.
  //
  // This function doesn't support entries that are longer than
  // 0xffffff00 bytes. (The "initial length" is always a 32-bit
  // value.) Nor does it support .debug_frame sections longer than
  // 0xffffff00 bytes.
  CFISection& FDEHeader(Label cie_pointer,
                        uint64_t initial_location,
                        uint64_t address_range,
                        bool dwarf64 = false);

  // Note the current position as the end of the last CIE or FDE we
  // started, after padding with DW_CFA_nops for alignment. This
  // defines the label representing the entry's length, cited in the
  // entry's header. Return a reference to this section.
  CFISection& FinishEntry();

  // Append the contents of BLOCK as a DW_FORM_block value: an
  // unsigned LEB128 length, followed by that many bytes of data.
  CFISection& Block(const string& block) {
    ULEB128(block.size());
    Append(block);
    return *this;
  }

  // Append ADDRESS to this section, in the appropriate size and
  // endianness. Return a reference to this section.
  CFISection& Address(uint64_t address) {
    Section::Append(endianness(), address_size_, address);
    return *this;
  }
  CFISection& Address(Label address) {
    Section::Append(endianness(), address_size_, address);
    return *this;
  }

  // Append ADDRESS to this section, using ENCODING and BASES. ENCODING
  // defaults to this section's default encoding, established by
  // SetPointerEncoding. BASES defaults to this section's bases, set by
  // SetEncodedPointerBases. If the DW_EH_PE_indirect bit is set in the
  // encoding, assume that ADDRESS is where the true address is stored.
  // Return a reference to this section.
  // 
  // (C++ doesn't let me use default arguments here, because I want to
  // refer to members of *this in the default argument expression.)
  CFISection& EncodedPointer(uint64_t address) {
    return EncodedPointer(address, pointer_encoding_, encoded_pointer_bases_);
  }
  CFISection& EncodedPointer(uint64_t address, DwarfPointerEncoding encoding) {
    return EncodedPointer(address, encoding, encoded_pointer_bases_);
  }
  CFISection& EncodedPointer(uint64_t address, DwarfPointerEncoding encoding,
                             const EncodedPointerBases& bases);

  // Restate some member functions, to keep chaining working nicely.
  CFISection& Mark(Label* label)  { Section::Mark(label); return *this; }
  CFISection& D8(uint8_t v)       { Section::D8(v);       return *this; }
  CFISection& D16(uint16_t v)     { Section::D16(v);      return *this; }
  CFISection& D16(Label v)        { Section::D16(v);      return *this; }
  CFISection& D32(uint32_t v)     { Section::D32(v);      return *this; }
  CFISection& D32(const Label& v) { Section::D32(v);      return *this; }
  CFISection& D64(uint64_t v)     { Section::D64(v);      return *this; }
  CFISection& D64(const Label& v) { Section::D64(v);      return *this; }
  CFISection& LEB128(long long v) { Section::LEB128(v);   return *this; }
  CFISection& ULEB128(uint64_t v) { Section::ULEB128(v);  return *this; }

 private:
  // A length value that we've appended to the section, but is not yet
  // known. LENGTH is the appended value; START is a label referring
  // to the start of the data whose length was cited.
  struct PendingLength {
    Label length;
    Label start;
  };

  // Constants used in CFI/.eh_frame data:

  // If the first four bytes of an "initial length" are this constant, then
  // the data uses the 64-bit DWARF format, and the length itself is the
  // subsequent eight bytes.
  static const uint32_t kDwarf64InitialLengthMarker = 0xffffffffU;

  // The CIE identifier for 32- and 64-bit DWARF CFI and .eh_frame data.
  static const uint32_t kDwarf32CIEIdentifier = ~(uint32_t)0;
  static const uint64_t kDwarf64CIEIdentifier = ~(uint64_t)0;
  static const uint32_t kEHFrame32CIEIdentifier = 0;
  static const uint64_t kEHFrame64CIEIdentifier = 0;

  // The size of a machine address for the data in this section.
  size_t address_size_;

  // If true, we are generating a Linux .eh_frame section, instead of
  // a standard DWARF .debug_frame section.
  bool eh_frame_;

  // The encoding to use for FDE pointers.
  DwarfPointerEncoding pointer_encoding_;

  // The base addresses to use when emitting encoded pointers.
  EncodedPointerBases encoded_pointer_bases_;

  // The length value for the current entry.
  //
  // Oddly, this must be dynamically allocated. Labels never get new
  // values; they only acquire constraints on the value they already
  // have, or assert if you assign them something incompatible. So
  // each header needs truly fresh Label objects to cite in their
  // headers and track their positions. The alternative is explicit
  // destructor invocation and a placement new. Ick.
  PendingLength *entry_length_;

  // True if we are currently emitting an FDE --- that is, we have
  // called FDEHeader but have not yet called FinishEntry.
  bool in_fde_;

  // If in_fde_ is true, this is its starting address. We use this for
  // emitting DW_EH_PE_funcrel pointers.
  uint64_t fde_start_address_;
};

}  // namespace google_breakpad

#endif  // PROCESSOR_CFI_ASSEMBLER_H_
