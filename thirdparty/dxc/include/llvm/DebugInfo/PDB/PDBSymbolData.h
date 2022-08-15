//===- PDBSymbolData.h - PDB data (e.g. variable) accessors -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_PDBSYMBOLDATA_H
#define LLVM_DEBUGINFO_PDB_PDBSYMBOLDATA_H

#include "PDBSymbol.h"
#include "PDBTypes.h"

namespace llvm {

class raw_ostream;

class PDBSymbolData : public PDBSymbol {
public:
  PDBSymbolData(const IPDBSession &PDBSession,
                std::unique_ptr<IPDBRawSymbol> DataSymbol);

  DECLARE_PDB_SYMBOL_CONCRETE_TYPE(PDB_SymType::Data)

  std::unique_ptr<PDBSymbol> getType() const;

  void dump(PDBSymDumper &Dumper) const override;

  FORWARD_SYMBOL_METHOD(getAccess)
  FORWARD_SYMBOL_METHOD(getAddressOffset)
  FORWARD_SYMBOL_METHOD(getAddressSection)
  FORWARD_SYMBOL_METHOD(getAddressTaken)
  FORWARD_SYMBOL_METHOD(getBitPosition)
  FORWARD_SYMBOL_METHOD(getClassParentId)
  FORWARD_SYMBOL_METHOD(isCompilerGenerated)
  FORWARD_SYMBOL_METHOD(isConstType)
  FORWARD_SYMBOL_METHOD(getDataKind)
  FORWARD_SYMBOL_METHOD(isAggregated)
  FORWARD_SYMBOL_METHOD(isSplitted)
  FORWARD_SYMBOL_METHOD(getLength)
  FORWARD_SYMBOL_METHOD(getLexicalParentId)
  FORWARD_SYMBOL_METHOD(getLocationType)
  FORWARD_SYMBOL_METHOD(getName)
  FORWARD_SYMBOL_METHOD(getOffset)
  FORWARD_SYMBOL_METHOD(getRegisterId)
  FORWARD_SYMBOL_METHOD(getRelativeVirtualAddress)
  FORWARD_SYMBOL_METHOD(getSlot)
  FORWARD_SYMBOL_METHOD(getSymIndexId)
  FORWARD_SYMBOL_METHOD(getToken)
  FORWARD_SYMBOL_METHOD(getTypeId)
  FORWARD_SYMBOL_METHOD(isUnalignedType)
  FORWARD_SYMBOL_METHOD(getValue)
  FORWARD_SYMBOL_METHOD(getVirtualAddress)
  FORWARD_SYMBOL_METHOD(isVolatileType)
};

} // namespace llvm

#endif // LLVM_DEBUGINFO_PDB_PDBSYMBOLDATA_H
