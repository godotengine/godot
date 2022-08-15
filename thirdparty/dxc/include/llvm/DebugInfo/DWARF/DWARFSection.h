//===-- DWARFSection.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_DEBUGINFO_DWARFSECTION_H
#define LLVM_LIB_DEBUGINFO_DWARFSECTION_H

#include "llvm/DebugInfo/DWARF/DWARFRelocMap.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {

struct DWARFSection {
  StringRef Data;
  RelocAddrMap Relocs;
};

}

#endif
