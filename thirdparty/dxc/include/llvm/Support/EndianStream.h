//===- EndianStream.h - Stream ops with endian specific data ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines utilities for operating on streams that have endian
// specific data.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_ENDIANSTREAM_H
#define LLVM_SUPPORT_ENDIANSTREAM_H

#include "llvm/Support/Endian.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace support {

namespace endian {
/// Adapter to write values to a stream in a particular byte order.
template <endianness endian> struct Writer {
  raw_ostream &OS;
  Writer(raw_ostream &OS) : OS(OS) {}
  template <typename value_type> void write(value_type Val) {
    Val = byte_swap<value_type, endian>(Val);
    OS.write((const char *)&Val, sizeof(value_type));
  }
};

template <>
template <>
inline void Writer<little>::write<float>(float Val) {
  write(FloatToBits(Val));
}

template <>
template <>
inline void Writer<little>::write<double>(double Val) {
  write(DoubleToBits(Val));
}

template <>
template <>
inline void Writer<big>::write<float>(float Val) {
  write(FloatToBits(Val));
}

template <>
template <>
inline void Writer<big>::write<double>(double Val) {
  write(DoubleToBits(Val));
}

} // end namespace endian

} // end namespace support
} // end namespace llvm

#endif
