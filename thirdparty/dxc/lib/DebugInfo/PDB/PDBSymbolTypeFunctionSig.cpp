//===- PDBSymbolTypeFunctionSig.cpp - --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/PDBSymbolTypeFunctionSig.h"

#include "llvm/DebugInfo/PDB/ConcreteSymbolEnumerator.h"
#include "llvm/DebugInfo/PDB/IPDBEnumChildren.h"
#include "llvm/DebugInfo/PDB/IPDBSession.h"
#include "llvm/DebugInfo/PDB/PDBSymbol.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeFunctionArg.h"
#include "llvm/DebugInfo/PDB/PDBSymDumper.h"

#include <utility>

using namespace llvm;

namespace {
class FunctionArgEnumerator : public IPDBEnumSymbols {
public:
  typedef ConcreteSymbolEnumerator<PDBSymbolTypeFunctionArg> ArgEnumeratorType;

  FunctionArgEnumerator(const IPDBSession &PDBSession,
                        const PDBSymbolTypeFunctionSig &Sig)
      : Session(PDBSession),
        Enumerator(Sig.findAllChildren<PDBSymbolTypeFunctionArg>()) {}

  FunctionArgEnumerator(const IPDBSession &PDBSession,
                        std::unique_ptr<ArgEnumeratorType> ArgEnumerator)
      : Session(PDBSession), Enumerator(std::move(ArgEnumerator)) {}

  uint32_t getChildCount() const override {
    return Enumerator->getChildCount();
  }

  std::unique_ptr<PDBSymbol> getChildAtIndex(uint32_t Index) const override {
    auto FunctionArgSymbol = Enumerator->getChildAtIndex(Index);
    if (!FunctionArgSymbol)
      return nullptr;
    return Session.getSymbolById(FunctionArgSymbol->getTypeId());
  }

  std::unique_ptr<PDBSymbol> getNext() override {
    auto FunctionArgSymbol = Enumerator->getNext();
    if (!FunctionArgSymbol)
      return nullptr;
    return Session.getSymbolById(FunctionArgSymbol->getTypeId());
  }

  void reset() override { Enumerator->reset(); }

  MyType *clone() const override {
    std::unique_ptr<ArgEnumeratorType> Clone(Enumerator->clone());
    return new FunctionArgEnumerator(Session, std::move(Clone));
  }

private:
  const IPDBSession &Session;
  std::unique_ptr<ArgEnumeratorType> Enumerator;
};
}

PDBSymbolTypeFunctionSig::PDBSymbolTypeFunctionSig(
    const IPDBSession &PDBSession, std::unique_ptr<IPDBRawSymbol> Symbol)
    : PDBSymbol(PDBSession, std::move(Symbol)) {}

std::unique_ptr<PDBSymbol> PDBSymbolTypeFunctionSig::getReturnType() const {
  return Session.getSymbolById(getTypeId());
}

std::unique_ptr<IPDBEnumSymbols>
PDBSymbolTypeFunctionSig::getArguments() const {
  return llvm::make_unique<FunctionArgEnumerator>(Session, *this);
}

std::unique_ptr<PDBSymbol> PDBSymbolTypeFunctionSig::getClassParent() const {
  uint32_t ClassId = getClassParentId();
  if (ClassId == 0)
    return nullptr;
  return Session.getSymbolById(ClassId);
}

void PDBSymbolTypeFunctionSig::dump(PDBSymDumper &Dumper) const {
  Dumper.dump(*this);
}
