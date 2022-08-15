//===- PDBSymbolTypeVTableShape.h - VTable shape info -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_PDBSYMBOLTYPEVTABLESHAPE_H
#define LLVM_DEBUGINFO_PDB_PDBSYMBOLTYPEVTABLESHAPE_H

#include "PDBSymbol.h"
#include "PDBTypes.h"

namespace llvm {

class raw_ostream;

class PDBSymbolTypeVTableShape : public PDBSymbol {
public:
  PDBSymbolTypeVTableShape(const IPDBSession &PDBSession,
                           std::unique_ptr<IPDBRawSymbol> VtblShapeSymbol);

  DECLARE_PDB_SYMBOL_CONCRETE_TYPE(PDB_SymType::VTableShape)

  void dump(PDBSymDumper &Dumper) const override;

  FORWARD_SYMBOL_METHOD(isConstType)
  FORWARD_SYMBOL_METHOD(getCount)
  FORWARD_SYMBOL_METHOD(getLexicalParentId)
  FORWARD_SYMBOL_METHOD(getSymIndexId)
  FORWARD_SYMBOL_METHOD(isUnalignedType)
  FORWARD_SYMBOL_METHOD(isVolatileType)
};

} // namespace llvm

#endif // LLVM_DEBUGINFO_PDB_PDBSYMBOLTYPEVTABLESHAPE_H
