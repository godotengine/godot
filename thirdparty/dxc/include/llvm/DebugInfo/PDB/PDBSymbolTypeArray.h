//===- PDBSymbolTypeArray.h - array type information ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_PDBSYMBOLTYPEARRAY_H
#define LLVM_DEBUGINFO_PDB_PDBSYMBOLTYPEARRAY_H

#include "PDBSymbol.h"
#include "PDBTypes.h"

namespace llvm {

class raw_ostream;

class PDBSymbolTypeArray : public PDBSymbol {
public:
  PDBSymbolTypeArray(const IPDBSession &PDBSession,
                     std::unique_ptr<IPDBRawSymbol> ArrayTypeSymbol);

  DECLARE_PDB_SYMBOL_CONCRETE_TYPE(PDB_SymType::ArrayType)

  std::unique_ptr<PDBSymbol> getElementType() const;

  void dump(PDBSymDumper &Dumper) const override;

  FORWARD_SYMBOL_METHOD(getArrayIndexTypeId)
  FORWARD_SYMBOL_METHOD(isConstType)
  FORWARD_SYMBOL_METHOD(getCount)
  FORWARD_SYMBOL_METHOD(getLength)
  FORWARD_SYMBOL_METHOD(getLexicalParentId)
  FORWARD_SYMBOL_METHOD(getRank)
  FORWARD_SYMBOL_METHOD(getSymIndexId)
  FORWARD_SYMBOL_METHOD(getTypeId)
  FORWARD_SYMBOL_METHOD(isUnalignedType)
  FORWARD_SYMBOL_METHOD(isVolatileType)
};

} // namespace llvm

#endif // LLVM_DEBUGINFO_PDB_PDBSYMBOLTYPEARRAY_H
