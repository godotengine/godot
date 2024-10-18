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

#ifdef HAVE_CONFIG_H
#include <config.h>  // Must come first
#endif

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

#include "common/dwarf/bytereader-inl.h"
#include "common/dwarf/bytereader.h"

namespace google_breakpad {

ByteReader::ByteReader(enum Endianness endian)
    :offset_reader_(NULL), address_reader_(NULL), endian_(endian),
     address_size_(0), offset_size_(0),
     have_section_base_(), have_text_base_(), have_data_base_(),
     have_function_base_() { }

ByteReader::~ByteReader() { }

void ByteReader::SetOffsetSize(uint8_t size) {
  offset_size_ = size;
  assert(size == 4 || size == 8);
  if (size == 4) {
    this->offset_reader_ = &ByteReader::ReadFourBytes;
  } else {
    this->offset_reader_ = &ByteReader::ReadEightBytes;
  }
}

void ByteReader::SetAddressSize(uint8_t size) {
  address_size_ = size;
  assert(size == 4 || size == 8);
  if (size == 4) {
    this->address_reader_ = &ByteReader::ReadFourBytes;
  } else {
    this->address_reader_ = &ByteReader::ReadEightBytes;
  }
}

uint64_t ByteReader::ReadInitialLength(const uint8_t* start, size_t* len) {
  const uint64_t initial_length = ReadFourBytes(start);
  start += 4;

  // In DWARF2/3, if the initial length is all 1 bits, then the offset
  // size is 8 and we need to read the next 8 bytes for the real length.
  if (initial_length == 0xffffffff) {
    SetOffsetSize(8);
    *len = 12;
    return ReadOffset(start);
  } else {
    SetOffsetSize(4);
    *len = 4;
  }
  return initial_length;
}

bool ByteReader::ValidEncoding(DwarfPointerEncoding encoding) const {
  if (encoding == DW_EH_PE_omit) return true;
  if (encoding == DW_EH_PE_aligned) return true;
  if ((encoding & 0x7) > DW_EH_PE_udata8)
    return false;
  if ((encoding & 0x70) > DW_EH_PE_funcrel)
    return false;
  return true;
}

bool ByteReader::UsableEncoding(DwarfPointerEncoding encoding) const {
  switch (encoding & 0x70) {
    case DW_EH_PE_absptr:  return true;
    case DW_EH_PE_pcrel:   return have_section_base_;
    case DW_EH_PE_textrel: return have_text_base_;
    case DW_EH_PE_datarel: return have_data_base_;
    case DW_EH_PE_funcrel: return have_function_base_;
    default:               return false;
  }
}

uint64_t ByteReader::ReadEncodedPointer(const uint8_t* buffer,
                                      DwarfPointerEncoding encoding,
                                      size_t* len) const {
  // UsableEncoding doesn't approve of DW_EH_PE_omit, so we shouldn't
  // see it here.
  assert(encoding != DW_EH_PE_omit);

  // The Linux Standards Base 4.0 does not make this clear, but the
  // GNU tools (gcc/unwind-pe.h; readelf/dwarf.c; gdb/dwarf2-frame.c)
  // agree that aligned pointers are always absolute, machine-sized,
  // machine-signed pointers.
  if (encoding == DW_EH_PE_aligned) {
    assert(have_section_base_);

    // We don't need to align BUFFER in *our* address space. Rather, we
    // need to find the next position in our buffer that would be aligned
    // when the .eh_frame section the buffer contains is loaded into the
    // program's memory. So align assuming that buffer_base_ gets loaded at
    // address section_base_, where section_base_ itself may or may not be
    // aligned.

    // First, find the offset to START from the closest prior aligned
    // address.
    uint64_t skew = section_base_ & (AddressSize() - 1);
    // Now find the offset from that aligned address to buffer.
    uint64_t offset = skew + (buffer - buffer_base_);
    // Round up to the next boundary.
    uint64_t aligned = (offset + AddressSize() - 1) & -AddressSize();
    // Convert back to a pointer.
    const uint8_t* aligned_buffer = buffer_base_ + (aligned - skew);
    // Finally, store the length and actually fetch the pointer.
    *len = aligned_buffer - buffer + AddressSize();
    return ReadAddress(aligned_buffer);
  }

  // Extract the value first, ignoring whether it's a pointer or an
  // offset relative to some base.
  uint64_t offset;
  switch (encoding & 0x0f) {
    case DW_EH_PE_absptr:
      // DW_EH_PE_absptr is weird, as it is used as a meaningful value for
      // both the high and low nybble of encoding bytes. When it appears in
      // the high nybble, it means that the pointer is absolute, not an
      // offset from some base address. When it appears in the low nybble,
      // as here, it means that the pointer is stored as a normal
      // machine-sized and machine-signed address. A low nybble of
      // DW_EH_PE_absptr does not imply that the pointer is absolute; it is
      // correct for us to treat the value as an offset from a base address
      // if the upper nybble is not DW_EH_PE_absptr.
      offset = ReadAddress(buffer);
      *len = AddressSize();
      break;

    case DW_EH_PE_uleb128:
      offset = ReadUnsignedLEB128(buffer, len);
      break;

    case DW_EH_PE_udata2:
      offset = ReadTwoBytes(buffer);
      *len = 2;
      break;

    case DW_EH_PE_udata4:
      offset = ReadFourBytes(buffer);
      *len = 4;
      break;

    case DW_EH_PE_udata8:
      offset = ReadEightBytes(buffer);
      *len = 8;
      break;

    case DW_EH_PE_sleb128:
      offset = ReadSignedLEB128(buffer, len);
      break;

    case DW_EH_PE_sdata2:
      offset = ReadTwoBytes(buffer);
      // Sign-extend from 16 bits.
      offset = (offset ^ 0x8000) - 0x8000;
      *len = 2;
      break;

    case DW_EH_PE_sdata4:
      offset = ReadFourBytes(buffer);
      // Sign-extend from 32 bits.
      offset = (offset ^ 0x80000000ULL) - 0x80000000ULL;
      *len = 4;
      break;

    case DW_EH_PE_sdata8:
      // No need to sign-extend; this is the full width of our type.
      offset = ReadEightBytes(buffer);
      *len = 8;
      break;

    default:
      abort();
  }

  // Find the appropriate base address.
  uint64_t base;
  switch (encoding & 0x70) {
    case DW_EH_PE_absptr:
      base = 0;
      break;

    case DW_EH_PE_pcrel:
      assert(have_section_base_);
      base = section_base_ + (buffer - buffer_base_);
      break;

    case DW_EH_PE_textrel:
      assert(have_text_base_);
      base = text_base_;
      break;

    case DW_EH_PE_datarel:
      assert(have_data_base_);
      base = data_base_;
      break;

    case DW_EH_PE_funcrel:
      assert(have_function_base_);
      base = function_base_;
      break;

    default:
      abort();
  }

  uint64_t pointer = base + offset;

  // Remove inappropriate upper bits.
  if (AddressSize() == 4)
    pointer = pointer & 0xffffffff;
  else
    assert(AddressSize() == sizeof(uint64_t));

  return pointer;
}

Endianness ByteReader::GetEndianness() const {
  return endian_;
}

}  // namespace google_breakpad
