//===- llvm/Support/LEB128.h - [SU]LEB128 utility functions -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares some utility functions for encoding SLEB128 and
// ULEB128 values.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_LEB128_H
#define LLVM_SUPPORT_LEB128_H

#include "llvm/Support/raw_ostream.h"

namespace llvm {

/// Utility function to encode a SLEB128 value to an output stream.
inline void encodeSLEB128(int64_t Value, raw_ostream &OS) {
  bool More;
  do {
    uint8_t Byte = Value & 0x7f;
    // NOTE: this assumes that this signed shift is an arithmetic right shift.
    Value >>= 7;
    More = !((((Value == 0 ) && ((Byte & 0x40) == 0)) ||
              ((Value == -1) && ((Byte & 0x40) != 0))));
    if (More)
      Byte |= 0x80; // Mark this byte to show that more bytes will follow.
    OS << char(Byte);
  } while (More);
}

/// Utility function to encode a ULEB128 value to an output stream.
inline void encodeULEB128(uint64_t Value, raw_ostream &OS,
                          unsigned Padding = 0) {
  do {
    uint8_t Byte = Value & 0x7f;
    Value >>= 7;
    if (Value != 0 || Padding != 0)
      Byte |= 0x80; // Mark this byte to show that more bytes will follow.
    OS << char(Byte);
  } while (Value != 0);

  // Pad with 0x80 and emit a null byte at the end.
  if (Padding != 0) {
    for (; Padding != 1; --Padding)
      OS << '\x80';
    OS << '\x00';
  }
}

/// Utility function to encode a ULEB128 value to a buffer. Returns
/// the length in bytes of the encoded value.
inline unsigned encodeULEB128(uint64_t Value, uint8_t *p,
                              unsigned Padding = 0) {
  uint8_t *orig_p = p;
  do {
    uint8_t Byte = Value & 0x7f;
    Value >>= 7;
    if (Value != 0 || Padding != 0)
      Byte |= 0x80; // Mark this byte to show that more bytes will follow.
    *p++ = Byte;
  } while (Value != 0);

  // Pad with 0x80 and emit a null byte at the end.
  if (Padding != 0) {
    for (; Padding != 1; --Padding)
      *p++ = '\x80';
    *p++ = '\x00';
  }
  return (unsigned)(p - orig_p);
}


/// Utility function to decode a ULEB128 value.
inline uint64_t decodeULEB128(const uint8_t *p, unsigned *n = nullptr) {
  const uint8_t *orig_p = p;
  uint64_t Value = 0;
  unsigned Shift = 0;
  do {
    Value += uint64_t(*p & 0x7f) << Shift;
    Shift += 7;
  } while (*p++ >= 128);
  if (n)
    *n = (unsigned)(p - orig_p);
  return Value;
}

/// Utility function to decode a SLEB128 value.
inline int64_t decodeSLEB128(const uint8_t *p, unsigned *n = nullptr) {
  const uint8_t *orig_p = p;
  int64_t Value = 0;
  unsigned Shift = 0;
  uint8_t Byte;
  do {
    Byte = *p++;
    Value |= (((uint64_t)(Byte & 0x7f)) << Shift); // HLSL Change - use 64 bits before casting, not after
    Shift += 7;
  } while (Byte >= 128);
  // Sign extend negative numbers.
  if (Byte & 0x40)
    Value |= (-1ULL) << Shift;
  if (n)
    *n = (unsigned)(p - orig_p);
  return Value;
}


/// Utility function to get the size of the ULEB128-encoded value.
extern unsigned getULEB128Size(uint64_t Value);

/// Utility function to get the size of the SLEB128-encoded value.
extern unsigned getSLEB128Size(int64_t Value);

}  // namespace llvm

#endif  // LLVM_SYSTEM_LEB128_H
