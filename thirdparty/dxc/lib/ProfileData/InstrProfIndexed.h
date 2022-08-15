//=-- InstrProfIndexed.h - Indexed profiling format support -------*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Shared header for the instrumented profile data reader and writer.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_PROFILEDATA_INSTRPROFINDEXED_H
#define LLVM_LIB_PROFILEDATA_INSTRPROFINDEXED_H

#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MD5.h"

namespace llvm {

namespace IndexedInstrProf {
enum class HashT : uint32_t {
  MD5,

  Last = MD5
};

static inline uint64_t MD5Hash(StringRef Str) {
  MD5 Hash;
  Hash.update(Str);
  llvm::MD5::MD5Result Result;
  Hash.final(Result);
  // Return the least significant 8 bytes. Our MD5 implementation returns the
  // result in little endian, so we may need to swap bytes.
  using namespace llvm::support;
  return endian::read<uint64_t, little, unaligned>(Result);
}

static inline uint64_t ComputeHash(HashT Type, StringRef K) {
  switch (Type) {
  case HashT::MD5:
    return IndexedInstrProf::MD5Hash(K);
  }
  llvm_unreachable("Unhandled hash type");
}

const uint64_t Magic = 0x8169666f72706cff; // "\xfflprofi\x81"
const uint64_t Version = 2;
const HashT HashType = HashT::MD5;
}

} // end namespace llvm

#endif
