//===- PDBSymbolBlock.h - Accessors for querying PDB blocks -------------*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_DEBUGINFO_PDB_PDBSYMBOLBLOCK_H
#define LLVM_DEBUGINFO_PDB_PDBSYMBOLBLOCK_H

#include "PDBSymbol.h"
#include "PDBTypes.h"
#include <string>

namespace llvm {

class raw_ostream;

class PDBSymbolBlock : public PDBSymbol {
public:
  PDBSymbolBlock(const IPDBSession &PDBSession,
                 std::unique_ptr<IPDBRawSymbol> Symbol);

  DECLARE_PDB_SYMBOL_CONCRETE_TYPE(PDB_SymType::Block)

  void dump(PDBSymDumper &Dumper) const override;

  FORWARD_SYMBOL_METHOD(getAddressOffset)
  FORWARD_SYMBOL_METHOD(getAddressSection)
  FORWARD_SYMBOL_METHOD(getLength)
  FORWARD_SYMBOL_METHOD(getLexicalParentId)
  FORWARD_SYMBOL_METHOD(getLocationType)
  FORWARD_SYMBOL_METHOD(getName)
  FORWARD_SYMBOL_METHOD(getRelativeVirtualAddress)
  FORWARD_SYMBOL_METHOD(getSymIndexId)
  FORWARD_SYMBOL_METHOD(getVirtualAddress)
};
}

#endif // LLVM_DEBUGINFO_PDB_PDBSYMBOLBLOCK_H
