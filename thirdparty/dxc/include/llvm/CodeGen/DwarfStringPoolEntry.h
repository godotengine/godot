//===- llvm/CodeGen/DwarfStringPoolEntry.h - String pool entry --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_DWARFSTRINGPOOLENTRY_H
#define LLVM_CODEGEN_DWARFSTRINGPOOLENTRY_H

#include "llvm/ADT/StringMap.h"

namespace llvm {

class MCSymbol;

/// Data for a string pool entry.
struct DwarfStringPoolEntry {
  MCSymbol *Symbol;
  unsigned Offset;
  unsigned Index;
};

/// String pool entry reference.
struct DwarfStringPoolEntryRef {
  const StringMapEntry<DwarfStringPoolEntry> *I = nullptr;

public:
  DwarfStringPoolEntryRef() = default;
  explicit DwarfStringPoolEntryRef(
      const StringMapEntry<DwarfStringPoolEntry> &I)
      : I(&I) {}

  explicit operator bool() const { return I; }
  MCSymbol *getSymbol() const {
    assert(I->second.Symbol && "No symbol available!");
    return I->second.Symbol;
  }
  unsigned getOffset() const { return I->second.Offset; }
  unsigned getIndex() const { return I->second.Index; }
  StringRef getString() const { return I->first(); }

  bool operator==(const DwarfStringPoolEntryRef &X) const { return I == X.I; }
  bool operator!=(const DwarfStringPoolEntryRef &X) const { return I != X.I; }
};

} // end namespace llvm

#endif
