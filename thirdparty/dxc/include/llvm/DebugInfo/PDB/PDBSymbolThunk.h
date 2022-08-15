//===- PDBSymbolThunk.h - Support for querying PDB thunks ---------------*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_PDBSYMBOLTHUNK_H
#define LLVM_DEBUGINFO_PDB_PDBSYMBOLTHUNK_H

#include "PDBSymbol.h"
#include "PDBTypes.h"
#include <string>

namespace llvm {

class raw_ostream;

class PDBSymbolThunk : public PDBSymbol {
public:
  PDBSymbolThunk(const IPDBSession &PDBSession,
                 std::unique_ptr<IPDBRawSymbol> ThunkSymbol);

  DECLARE_PDB_SYMBOL_CONCRETE_TYPE(PDB_SymType::Thunk)

  void dump(PDBSymDumper &Dumper) const override;

  FORWARD_SYMBOL_METHOD(getAccess)
  FORWARD_SYMBOL_METHOD(getAddressOffset)
  FORWARD_SYMBOL_METHOD(getAddressSection)
  FORWARD_SYMBOL_METHOD(getClassParentId)
  FORWARD_SYMBOL_METHOD(isConstType)
  FORWARD_SYMBOL_METHOD(isIntroVirtualFunction)
  FORWARD_SYMBOL_METHOD(isStatic)
  FORWARD_SYMBOL_METHOD(getLength)
  FORWARD_SYMBOL_METHOD(getLexicalParentId)
  FORWARD_SYMBOL_METHOD(getName)
  FORWARD_SYMBOL_METHOD(isPureVirtual)
  FORWARD_SYMBOL_METHOD(getRelativeVirtualAddress)
  FORWARD_SYMBOL_METHOD(getSymIndexId)
  FORWARD_SYMBOL_METHOD(getTargetOffset)
  FORWARD_SYMBOL_METHOD(getTargetRelativeVirtualAddress)
  FORWARD_SYMBOL_METHOD(getTargetVirtualAddress)
  FORWARD_SYMBOL_METHOD(getTargetSection)
  FORWARD_SYMBOL_METHOD(getThunkOrdinal)
  FORWARD_SYMBOL_METHOD(getTypeId)
  FORWARD_SYMBOL_METHOD(isUnalignedType)
  FORWARD_SYMBOL_METHOD(isVirtual)
  FORWARD_SYMBOL_METHOD(getVirtualAddress)
  FORWARD_SYMBOL_METHOD(getVirtualBaseOffset)
  FORWARD_SYMBOL_METHOD(isVolatileType)
};
} // namespace llvm

#endif // LLVM_DEBUGINFO_PDB_PDBSYMBOLTHUNK_H
