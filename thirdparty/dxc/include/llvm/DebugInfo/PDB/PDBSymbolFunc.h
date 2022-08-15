//===- PDBSymbolFunc.h - class representing a function instance -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_PDBSYMBOLFUNC_H
#define LLVM_DEBUGINFO_PDB_PDBSYMBOLFUNC_H

#include "PDBSymbol.h"
#include "PDBTypes.h"

namespace llvm {

class raw_ostream;

class PDBSymbolFunc : public PDBSymbol {
public:
  PDBSymbolFunc(const IPDBSession &PDBSession,
                std::unique_ptr<IPDBRawSymbol> FuncSymbol);

  void dump(PDBSymDumper &Dumper) const override;

  std::unique_ptr<PDBSymbolTypeFunctionSig> getSignature() const;
  std::unique_ptr<PDBSymbolTypeUDT> getClassParent() const;
  std::unique_ptr<IPDBEnumChildren<PDBSymbolData>> getArguments() const;

  DECLARE_PDB_SYMBOL_CONCRETE_TYPE(PDB_SymType::Function)

  FORWARD_SYMBOL_METHOD(getAccess)
  FORWARD_SYMBOL_METHOD(getAddressOffset)
  FORWARD_SYMBOL_METHOD(getAddressSection)
  FORWARD_SYMBOL_METHOD(getClassParentId)
  FORWARD_SYMBOL_METHOD(isCompilerGenerated)
  FORWARD_SYMBOL_METHOD(isConstType)
  FORWARD_SYMBOL_METHOD(hasCustomCallingConvention)
  FORWARD_SYMBOL_METHOD(hasFarReturn)
  FORWARD_SYMBOL_METHOD(hasAlloca)
  FORWARD_SYMBOL_METHOD(hasEH)
  FORWARD_SYMBOL_METHOD(hasEHa)
  FORWARD_SYMBOL_METHOD(hasInlAsm)
  FORWARD_SYMBOL_METHOD(hasLongJump)
  FORWARD_SYMBOL_METHOD(hasSEH)
  FORWARD_SYMBOL_METHOD(hasSecurityChecks)
  FORWARD_SYMBOL_METHOD(hasSetJump)
  FORWARD_SYMBOL_METHOD(hasInterruptReturn)
  FORWARD_SYMBOL_METHOD(isIntroVirtualFunction)
  FORWARD_SYMBOL_METHOD(hasInlineAttribute)
  FORWARD_SYMBOL_METHOD(isNaked)
  FORWARD_SYMBOL_METHOD(isStatic)
  FORWARD_SYMBOL_METHOD(getLength)
  FORWARD_SYMBOL_METHOD(getLexicalParentId)
  FORWARD_SYMBOL_METHOD(getLocalBasePointerRegisterId)
  FORWARD_SYMBOL_METHOD(getLocationType)
  FORWARD_SYMBOL_METHOD(getName)
  FORWARD_SYMBOL_METHOD(hasFramePointer)
  FORWARD_SYMBOL_METHOD(hasNoInlineAttribute)
  FORWARD_SYMBOL_METHOD(hasNoReturnAttribute)
  FORWARD_SYMBOL_METHOD(isUnreached)
  FORWARD_SYMBOL_METHOD(getNoStackOrdering)
  FORWARD_SYMBOL_METHOD(hasOptimizedCodeDebugInfo)
  FORWARD_SYMBOL_METHOD(isPureVirtual)
  FORWARD_SYMBOL_METHOD(getRelativeVirtualAddress)
  FORWARD_SYMBOL_METHOD(getSymIndexId)
  FORWARD_SYMBOL_METHOD(getToken)
  FORWARD_SYMBOL_METHOD(getTypeId)
  FORWARD_SYMBOL_METHOD(isUnalignedType)
  FORWARD_SYMBOL_METHOD(getUndecoratedName)
  FORWARD_SYMBOL_METHOD(isVirtual)
  FORWARD_SYMBOL_METHOD(getVirtualAddress)
  FORWARD_SYMBOL_METHOD(getVirtualBaseOffset)
  FORWARD_SYMBOL_METHOD(isVolatileType)
};

} // namespace llvm

#endif // LLVM_DEBUGINFO_PDB_PDBSYMBOLFUNC_H
