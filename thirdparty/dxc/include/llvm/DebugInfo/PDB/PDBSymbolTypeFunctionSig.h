//===- PDBSymbolTypeFunctionSig.h - function signature type info *- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_PDBSYMBOLTYPEFUNCTIONSIG_H
#define LLVM_DEBUGINFO_PDB_PDBSYMBOLTYPEFUNCTIONSIG_H

#include "PDBSymbol.h"
#include "PDBTypes.h"

namespace llvm {

class raw_ostream;

class PDBSymbolTypeFunctionSig : public PDBSymbol {
public:
  PDBSymbolTypeFunctionSig(const IPDBSession &PDBSession,
                           std::unique_ptr<IPDBRawSymbol> Symbol);

  DECLARE_PDB_SYMBOL_CONCRETE_TYPE(PDB_SymType::FunctionSig)

  std::unique_ptr<PDBSymbol> getReturnType() const;
  std::unique_ptr<IPDBEnumSymbols> getArguments() const;
  std::unique_ptr<PDBSymbol> getClassParent() const;

  void dump(PDBSymDumper &Dumper) const override;
  void dumpArgList(raw_ostream &OS) const;

  FORWARD_SYMBOL_METHOD(getCallingConvention)
  FORWARD_SYMBOL_METHOD(getClassParentId)
  FORWARD_SYMBOL_METHOD(getUnmodifiedTypeId)
  FORWARD_SYMBOL_METHOD(isConstType)
  FORWARD_SYMBOL_METHOD(getCount)
  FORWARD_SYMBOL_METHOD(getLexicalParentId)
  // FORWARD_SYMBOL_METHOD(getObjectPointerType)
  FORWARD_SYMBOL_METHOD(getSymIndexId)
  FORWARD_SYMBOL_METHOD(getThisAdjust)
  FORWARD_SYMBOL_METHOD(getTypeId)
  FORWARD_SYMBOL_METHOD(isUnalignedType)
  FORWARD_SYMBOL_METHOD(isVolatileType)
};

} // namespace llvm

#endif // LLVM_DEBUGINFO_PDB_PDBSYMBOLTYPEFUNCTIONSIG_H
