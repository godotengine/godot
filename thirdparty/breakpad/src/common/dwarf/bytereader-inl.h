// Copyright 2006 Google Inc. All Rights Reserved.
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

#ifndef UTIL_DEBUGINFO_BYTEREADER_INL_H__
#define UTIL_DEBUGINFO_BYTEREADER_INL_H__

#include "common/dwarf/bytereader.h"

#include <assert.h>
#include <stdint.h>

namespace google_breakpad {

inline uint8_t ByteReader::ReadOneByte(const uint8_t* buffer) const {
  return buffer[0];
}

inline uint16_t ByteReader::ReadTwoBytes(const uint8_t* buffer) const {
  const uint16_t buffer0 = buffer[0];
  const uint16_t buffer1 = buffer[1];
  if (endian_ == ENDIANNESS_LITTLE) {
    return buffer0 | buffer1 << 8;
  } else {
    return buffer1 | buffer0 << 8;
  }
}

inline uint64_t ByteReader::ReadThreeBytes(const uint8_t* buffer) const {
  const uint32_t buffer0 = buffer[0];
  const uint32_t buffer1 = buffer[1];
  const uint32_t buffer2 = buffer[2];
  if (endian_ == ENDIANNESS_LITTLE) {
    return buffer0 | buffer1 << 8 | buffer2 << 16;
  } else {
    return buffer2 | buffer1 << 8 | buffer0 << 16;
  }
}

inline uint64_t ByteReader::ReadFourBytes(const uint8_t* buffer) const {
  const uint32_t buffer0 = buffer[0];
  const uint32_t buffer1 = buffer[1];
  const uint32_t buffer2 = buffer[2];
  const uint32_t buffer3 = buffer[3];
  if (endian_ == ENDIANNESS_LITTLE) {
    return buffer0 | buffer1 << 8 | buffer2 << 16 | buffer3 << 24;
  } else {
    return buffer3 | buffer2 << 8 | buffer1 << 16 | buffer0 << 24;
  }
}

inline uint64_t ByteReader::ReadEightBytes(const uint8_t* buffer) const {
  const uint64_t buffer0 = buffer[0];
  const uint64_t buffer1 = buffer[1];
  const uint64_t buffer2 = buffer[2];
  const uint64_t buffer3 = buffer[3];
  const uint64_t buffer4 = buffer[4];
  const uint64_t buffer5 = buffer[5];
  const uint64_t buffer6 = buffer[6];
  const uint64_t buffer7 = buffer[7];
  if (endian_ == ENDIANNESS_LITTLE) {
    return buffer0 | buffer1 << 8 | buffer2 << 16 | buffer3 << 24 |
      buffer4 << 32 | buffer5 << 40 | buffer6 << 48 | buffer7 << 56;
  } else {
    return buffer7 | buffer6 << 8 | buffer5 << 16 | buffer4 << 24 |
      buffer3 << 32 | buffer2 << 40 | buffer1 << 48 | buffer0 << 56;
  }
}

// Read an unsigned LEB128 number.  Each byte contains 7 bits of
// information, plus one bit saying whether the number continues or
// not.

inline uint64_t ByteReader::ReadUnsignedLEB128(const uint8_t* buffer,
                                             size_t* len) const {
  uint64_t result = 0;
  size_t num_read = 0;
  unsigned int shift = 0;
  uint8_t byte;

  do {
    byte = *buffer++;
    num_read++;

    result |= (static_cast<uint64_t>(byte & 0x7f)) << shift;

    shift += 7;

  } while (byte & 0x80);

  *len = num_read;

  return result;
}

// Read a signed LEB128 number.  These are like regular LEB128
// numbers, except the last byte may have a sign bit set.

inline int64_t ByteReader::ReadSignedLEB128(const uint8_t* buffer,
                                          size_t* len) const {
  int64_t result = 0;
  unsigned int shift = 0;
  size_t num_read = 0;
  uint8_t byte;

  do {
      byte = *buffer++;
      num_read++;
      result |= (static_cast<uint64_t>(byte & 0x7f) << shift);
      shift += 7;
  } while (byte & 0x80);

  if ((shift < 8 * sizeof (result)) && (byte & 0x40))
    result |= -((static_cast<int64_t>(1)) << shift);
  *len = num_read;
  return result;
}

inline uint64_t ByteReader::ReadOffset(const uint8_t* buffer) const {
  assert(this->offset_reader_);
  return (this->*offset_reader_)(buffer);
}

inline uint64_t ByteReader::ReadAddress(const uint8_t* buffer) const {
  assert(this->address_reader_);
  return (this->*address_reader_)(buffer);
}

inline void ByteReader::SetCFIDataBase(uint64_t section_base,
                                       const uint8_t* buffer_base) {
  section_base_ = section_base;
  buffer_base_ = buffer_base;
  have_section_base_ = true;
}

inline void ByteReader::SetTextBase(uint64_t text_base) {
  text_base_ = text_base;
  have_text_base_ = true;
}

inline void ByteReader::SetDataBase(uint64_t data_base) {
  data_base_ = data_base;
  have_data_base_ = true;
}

inline void ByteReader::SetFunctionBase(uint64_t function_base) {
  function_base_ = function_base;
  have_function_base_ = true;
}

inline void ByteReader::ClearFunctionBase() {
  have_function_base_ = false;
}

}  // namespace google_breakpad

#endif  // UTIL_DEBUGINFO_BYTEREADER_INL_H__
