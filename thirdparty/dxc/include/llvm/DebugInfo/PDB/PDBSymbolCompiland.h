//===- PDBSymbolCompiland.h - Accessors for querying PDB compilands -----*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_DEBUGINFO_PDB_PDBSYMBOLCOMPILAND_H
#define LLVM_DEBUGINFO_PDB_PDBSYMBOLCOMPILAND_H

#include "PDBSymbol.h"
#include "PDBTypes.h"
#include <string>

namespace llvm {

class raw_ostream;

class PDBSymbolCompiland : public PDBSymbol {
public:
  PDBSymbolCompiland(const IPDBSession &PDBSession,
                     std::unique_ptr<IPDBRawSymbol> CompilandSymbol);

  DECLARE_PDB_SYMBOL_CONCRETE_TYPE(PDB_SymType::Compiland)

  void dump(PDBSymDumper &Dumper) const override;

  FORWARD_SYMBOL_METHOD(isEditAndContinueEnabled)
  FORWARD_SYMBOL_METHOD(getLexicalParentId)
  FORWARD_SYMBOL_METHOD(getLibraryName)
  FORWARD_SYMBOL_METHOD(getName)
  FORWARD_SYMBOL_METHOD(getSourceFileName)
  FORWARD_SYMBOL_METHOD(getSymIndexId)
};
}

#endif // LLVM_DEBUGINFO_PDB_PDBSYMBOLCOMPILAND_H
