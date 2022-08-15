//===-- DWARFRelocMap.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_DEBUGINFO_DWARFRELOCMAP_H
#define LLVM_LIB_DEBUGINFO_DWARFRELOCMAP_H

#include "llvm/ADT/DenseMap.h"

namespace llvm {

typedef DenseMap<uint64_t, std::pair<uint8_t, int64_t> > RelocAddrMap;

} // namespace llvm

#endif

