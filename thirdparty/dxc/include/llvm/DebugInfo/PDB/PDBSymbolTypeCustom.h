//===- PDBSymbolTypeCustom.h - custom compiler type information -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_PDBSYMBOLTYPECUSTOM_H
#define LLVM_DEBUGINFO_PDB_PDBSYMBOLTYPECUSTOM_H

#include "PDBSymbol.h"
#include "PDBTypes.h"

namespace llvm {

class raw_ostream;

class PDBSymbolTypeCustom : public PDBSymbol {
public:
  PDBSymbolTypeCustom(const IPDBSession &PDBSession,
                      std::unique_ptr<IPDBRawSymbol> Symbol);

  DECLARE_PDB_SYMBOL_CONCRETE_TYPE(PDB_SymType::CustomType)

  void dump(PDBSymDumper &Dumper) const override;

  FORWARD_SYMBOL_METHOD(getOemId)
  FORWARD_SYMBOL_METHOD(getOemSymbolId)
  FORWARD_SYMBOL_METHOD(getSymIndexId)
};

} // namespace llvm

#endif // LLVM_DEBUGINFO_PDB_PDBSYMBOLTYPECUSTOM_H
