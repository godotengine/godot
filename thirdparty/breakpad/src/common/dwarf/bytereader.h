// -*- mode: C++ -*-

// Copyright (c) 2010 Google Inc. All Rights Reserved.
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

#ifndef COMMON_DWARF_BYTEREADER_H__
#define COMMON_DWARF_BYTEREADER_H__

#include <stdint.h>

#include <string>

#include "common/dwarf/types.h"
#include "common/dwarf/dwarf2enums.h"

namespace google_breakpad {

// We can't use the obvious name of LITTLE_ENDIAN and BIG_ENDIAN
// because it conflicts with a macro
enum Endianness {
  ENDIANNESS_BIG,
  ENDIANNESS_LITTLE
};

// A ByteReader knows how to read single- and multi-byte values of
// various endiannesses, sizes, and encodings, as used in DWARF
// debugging information and Linux C++ exception handling data.
class ByteReader {
 public:
  // Construct a ByteReader capable of reading one-, two-, four-, and
  // eight-byte values according to ENDIANNESS, absolute machine-sized
  // addresses, DWARF-style "initial length" values, signed and
  // unsigned LEB128 numbers, and Linux C++ exception handling data's
  // encoded pointers.
  explicit ByteReader(enum Endianness endianness);
  virtual ~ByteReader();

  // Read a single byte from BUFFER and return it as an unsigned 8 bit
  // number.
  uint8_t ReadOneByte(const uint8_t* buffer) const;

  // Read two bytes from BUFFER and return them as an unsigned 16 bit
  // number, using this ByteReader's endianness.
  uint16_t ReadTwoBytes(const uint8_t* buffer) const;

  // Read three bytes from BUFFER and return them as an unsigned 64 bit
  // number, using this ByteReader's endianness. DWARF 5 uses this encoding
  // for various index-related DW_FORMs.
  uint64_t ReadThreeBytes(const uint8_t* buffer) const;

  // Read four bytes from BUFFER and return them as an unsigned 32 bit
  // number, using this ByteReader's endianness. This function returns
  // a uint64_t so that it is compatible with ReadAddress and
  // ReadOffset. The number it returns will never be outside the range
  // of an unsigned 32 bit integer.
  uint64_t ReadFourBytes(const uint8_t* buffer) const;

  // Read eight bytes from BUFFER and return them as an unsigned 64
  // bit number, using this ByteReader's endianness.
  uint64_t ReadEightBytes(const uint8_t* buffer) const;

  // Read an unsigned LEB128 (Little Endian Base 128) number from
  // BUFFER and return it as an unsigned 64 bit integer. Set LEN to
  // the number of bytes read.
  //
  // The unsigned LEB128 representation of an integer N is a variable
  // number of bytes:
  //
  // - If N is between 0 and 0x7f, then its unsigned LEB128
  //   representation is a single byte whose value is N.
  //
  // - Otherwise, its unsigned LEB128 representation is (N & 0x7f) |
  //   0x80, followed by the unsigned LEB128 representation of N /
  //   128, rounded towards negative infinity.
  //
  // In other words, we break VALUE into groups of seven bits, put
  // them in little-endian order, and then write them as eight-bit
  // bytes with the high bit on all but the last.
  uint64_t ReadUnsignedLEB128(const uint8_t* buffer, size_t* len) const;

  // Read a signed LEB128 number from BUFFER and return it as an
  // signed 64 bit integer. Set LEN to the number of bytes read.
  //
  // The signed LEB128 representation of an integer N is a variable
  // number of bytes:
  //
  // - If N is between -0x40 and 0x3f, then its signed LEB128
  //   representation is a single byte whose value is N in two's
  //   complement.
  //
  // - Otherwise, its signed LEB128 representation is (N & 0x7f) |
  //   0x80, followed by the signed LEB128 representation of N / 128,
  //   rounded towards negative infinity.
  //
  // In other words, we break VALUE into groups of seven bits, put
  // them in little-endian order, and then write them as eight-bit
  // bytes with the high bit on all but the last.
  int64_t ReadSignedLEB128(const uint8_t* buffer, size_t* len) const;

  // Indicate that addresses on this architecture are SIZE bytes long. SIZE
  // must be either 4 or 8. (DWARF allows addresses to be any number of
  // bytes in length from 1 to 255, but we only support 32- and 64-bit
  // addresses at the moment.) You must call this before using the
  // ReadAddress member function.
  //
  // For data in a .debug_info section, or something that .debug_info
  // refers to like line number or macro data, the compilation unit
  // header's address_size field indicates the address size to use. Call
  // frame information doesn't indicate its address size (a shortcoming of
  // the spec); you must supply the appropriate size based on the
  // architecture of the target machine.
  void SetAddressSize(uint8_t size);

  // Return the current address size, in bytes. This is either 4,
  // indicating 32-bit addresses, or 8, indicating 64-bit addresses.
  uint8_t AddressSize() const { return address_size_; }

  // Read an address from BUFFER and return it as an unsigned 64 bit
  // integer, respecting this ByteReader's endianness and address size. You
  // must call SetAddressSize before calling this function.
  uint64_t ReadAddress(const uint8_t* buffer) const;

  // DWARF actually defines two slightly different formats: 32-bit DWARF
  // and 64-bit DWARF. This is *not* related to the size of registers or
  // addresses on the target machine; it refers only to the size of section
  // offsets and data lengths appearing in the DWARF data. One only needs
  // 64-bit DWARF when the debugging data itself is larger than 4GiB.
  // 32-bit DWARF can handle x86_64 or PPC64 code just fine, unless the
  // debugging data itself is very large.
  //
  // DWARF information identifies itself as 32-bit or 64-bit DWARF: each
  // compilation unit and call frame information entry begins with an
  // "initial length" field, which, in addition to giving the length of the
  // data, also indicates the size of section offsets and lengths appearing
  // in that data. The ReadInitialLength member function, below, reads an
  // initial length and sets the ByteReader's offset size as a side effect.
  // Thus, in the normal process of reading DWARF data, the appropriate
  // offset size is set automatically. So, you should only need to call
  // SetOffsetSize if you are using the same ByteReader to jump from the
  // midst of one block of DWARF data into another.

  // Read a DWARF "initial length" field from START, and return it as
  // an unsigned 64 bit integer, respecting this ByteReader's
  // endianness. Set *LEN to the length of the initial length in
  // bytes, either four or twelve. As a side effect, set this
  // ByteReader's offset size to either 4 (if we see a 32-bit DWARF
  // initial length) or 8 (if we see a 64-bit DWARF initial length).
  //
  // A DWARF initial length is either:
  //
  // - a byte count stored as an unsigned 32-bit value less than
  //   0xffffff00, indicating that the data whose length is being
  //   measured uses the 32-bit DWARF format, or
  //
  // - The 32-bit value 0xffffffff, followed by a 64-bit byte count,
  //   indicating that the data whose length is being measured uses
  //   the 64-bit DWARF format.
  uint64_t ReadInitialLength(const uint8_t* start, size_t* len);

  // Read an offset from BUFFER and return it as an unsigned 64 bit
  // integer, respecting the ByteReader's endianness. In 32-bit DWARF, the
  // offset is 4 bytes long; in 64-bit DWARF, the offset is eight bytes
  // long. You must call ReadInitialLength or SetOffsetSize before calling
  // this function; see the comments above for details.
  uint64_t ReadOffset(const uint8_t* buffer) const;

  // Return the current offset size, in bytes.
  // A return value of 4 indicates that we are reading 32-bit DWARF.
  // A return value of 8 indicates that we are reading 64-bit DWARF.
  uint8_t OffsetSize() const { return offset_size_; }

  // Indicate that section offsets and lengths are SIZE bytes long. SIZE
  // must be either 4 (meaning 32-bit DWARF) or 8 (meaning 64-bit DWARF).
  // Usually, you should not call this function yourself; instead, let a
  // call to ReadInitialLength establish the data's offset size
  // automatically.
  void SetOffsetSize(uint8_t size);

  // The Linux C++ ABI uses a variant of DWARF call frame information
  // for exception handling. This data is included in the program's
  // address space as the ".eh_frame" section, and intepreted at
  // runtime to walk the stack, find exception handlers, and run
  // cleanup code. The format is mostly the same as DWARF CFI, with
  // some adjustments made to provide the additional
  // exception-handling data, and to make the data easier to work with
  // in memory --- for example, to allow it to be placed in read-only
  // memory even when describing position-independent code.
  //
  // In particular, exception handling data can select a number of
  // different encodings for pointers that appear in the data, as
  // described by the DwarfPointerEncoding enum. There are actually
  // four axes(!) to the encoding:
  //
  // - The pointer size: pointers can be 2, 4, or 8 bytes long, or use
  //   the DWARF LEB128 encoding.
  //
  // - The pointer's signedness: pointers can be signed or unsigned.
  //
  // - The pointer's base address: the data stored in the exception
  //   handling data can be the actual address (that is, an absolute
  //   pointer), or relative to one of a number of different base
  //   addreses --- including that of the encoded pointer itself, for
  //   a form of "pc-relative" addressing.
  //
  // - The pointer may be indirect: it may be the address where the
  //   true pointer is stored. (This is used to refer to things via
  //   global offset table entries, program linkage table entries, or
  //   other tricks used in position-independent code.)
  //
  // There are also two options that fall outside that matrix
  // altogether: the pointer may be omitted, or it may have padding to
  // align it on an appropriate address boundary. (That last option
  // may seem like it should be just another axis, but it is not.)

  // Indicate that the exception handling data is loaded starting at
  // SECTION_BASE, and that the start of its buffer in our own memory
  // is BUFFER_BASE. This allows us to find the address that a given
  // byte in our buffer would have when loaded into the program the
  // data describes. We need this to resolve DW_EH_PE_pcrel pointers.
  void SetCFIDataBase(uint64_t section_base, const uint8_t* buffer_base);

  // Indicate that the base address of the program's ".text" section
  // is TEXT_BASE. We need this to resolve DW_EH_PE_textrel pointers.
  void SetTextBase(uint64_t text_base);

  // Indicate that the base address for DW_EH_PE_datarel pointers is
  // DATA_BASE. The proper value depends on the ABI; it is usually the
  // address of the global offset table, held in a designated register in
  // position-independent code. You will need to look at the startup code
  // for the target system to be sure. I tried; my eyes bled.
  void SetDataBase(uint64_t data_base);

  // Indicate that the base address for the FDE we are processing is
  // FUNCTION_BASE. This is the start address of DW_EH_PE_funcrel
  // pointers. (This encoding does not seem to be used by the GNU
  // toolchain.)
  void SetFunctionBase(uint64_t function_base);

  // Indicate that we are no longer processing any FDE, so any use of
  // a DW_EH_PE_funcrel encoding is an error.
  void ClearFunctionBase();

  // Return true if ENCODING is a valid pointer encoding.
  bool ValidEncoding(DwarfPointerEncoding encoding) const;

  // Return true if we have all the information we need to read a
  // pointer that uses ENCODING. This checks that the appropriate
  // SetFooBase function for ENCODING has been called.
  bool UsableEncoding(DwarfPointerEncoding encoding) const;

  // Read an encoded pointer from BUFFER using ENCODING; return the
  // absolute address it represents, and set *LEN to the pointer's
  // length in bytes, including any padding for aligned pointers.
  //
  // This function calls 'abort' if ENCODING is invalid or refers to a
  // base address this reader hasn't been given, so you should check
  // with ValidEncoding and UsableEncoding first if you would rather
  // die in a more helpful way.
  uint64_t ReadEncodedPointer(const uint8_t* buffer,
                            DwarfPointerEncoding encoding,
                            size_t* len) const;

  Endianness GetEndianness() const;
 private:

  // Function pointer type for our address and offset readers.
  typedef uint64_t (ByteReader::*AddressReader)(const uint8_t*) const;

  // Read an offset from BUFFER and return it as an unsigned 64 bit
  // integer.  DWARF2/3 define offsets as either 4 or 8 bytes,
  // generally depending on the amount of DWARF2/3 info present.
  // This function pointer gets set by SetOffsetSize.
  AddressReader offset_reader_;

  // Read an address from BUFFER and return it as an unsigned 64 bit
  // integer.  DWARF2/3 allow addresses to be any size from 0-255
  // bytes currently.  Internally we support 4 and 8 byte addresses,
  // and will CHECK on anything else.
  // This function pointer gets set by SetAddressSize.
  AddressReader address_reader_;

  Endianness endian_;
  uint8_t address_size_;
  uint8_t offset_size_;

  // Base addresses for Linux C++ exception handling data's encoded pointers.
  bool have_section_base_, have_text_base_, have_data_base_;
  bool have_function_base_;
  uint64_t section_base_, text_base_, data_base_, function_base_;
  const uint8_t* buffer_base_;
};

}  // namespace google_breakpad

#endif  // COMMON_DWARF_BYTEREADER_H__
