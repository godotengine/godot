//===-- llvm/CodeGen/AddressPool.h - Dwarf Debug Framework -----*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_ASMPRINTER_ADDRESSPOOL_H
#define LLVM_LIB_CODEGEN_ASMPRINTER_ADDRESSPOOL_H

#include "llvm/ADT/DenseMap.h"

namespace llvm {
class MCSection;
class MCSymbol;
class AsmPrinter;
// Collection of addresses for this unit and assorted labels.
// A Symbol->unsigned mapping of addresses used by indirect
// references.
class AddressPool {
  struct AddressPoolEntry {
    unsigned Number;
    bool TLS;
    AddressPoolEntry(unsigned Number, bool TLS) : Number(Number), TLS(TLS) {}
  };
  DenseMap<const MCSymbol *, AddressPoolEntry> Pool;

  /// Record whether the AddressPool has been queried for an address index since
  /// the last "resetUsedFlag" call. Used to implement type unit fallback - a
  /// type that references addresses cannot be placed in a type unit when using
  /// fission.
  bool HasBeenUsed;

public:
  AddressPool() : HasBeenUsed(false) {}

  /// \brief Returns the index into the address pool with the given
  /// label/symbol.
  unsigned getIndex(const MCSymbol *Sym, bool TLS = false);

  void emit(AsmPrinter &Asm, MCSection *AddrSection);

  bool isEmpty() { return Pool.empty(); }

  bool hasBeenUsed() const { return HasBeenUsed; }

  void resetUsedFlag() { HasBeenUsed = false; }
};
}
#endif
