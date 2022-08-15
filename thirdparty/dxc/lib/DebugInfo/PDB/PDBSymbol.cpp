//===- PDBSymbol.cpp - base class for user-facing symbol types --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/PDBSymbol.h"

#include "llvm/DebugInfo/PDB/IPDBEnumChildren.h"
#include "llvm/DebugInfo/PDB/IPDBRawSymbol.h"
#include "llvm/DebugInfo/PDB/PDBSymbolAnnotation.h"
#include "llvm/DebugInfo/PDB/PDBSymbolBlock.h"
#include "llvm/DebugInfo/PDB/PDBSymbolCompiland.h"
#include "llvm/DebugInfo/PDB/PDBSymbolCompilandDetails.h"
#include "llvm/DebugInfo/PDB/PDBSymbolCompilandEnv.h"
#include "llvm/DebugInfo/PDB/PDBSymbolCustom.h"
#include "llvm/DebugInfo/PDB/PDBSymbolData.h"
#include "llvm/DebugInfo/PDB/PDBSymbolExe.h"
#include "llvm/DebugInfo/PDB/PDBSymbolFunc.h"
#include "llvm/DebugInfo/PDB/PDBSymbolFuncDebugEnd.h"
#include "llvm/DebugInfo/PDB/PDBSymbolFuncDebugStart.h"
#include "llvm/DebugInfo/PDB/PDBSymbolLabel.h"
#include "llvm/DebugInfo/PDB/PDBSymbolPublicSymbol.h"
#include "llvm/DebugInfo/PDB/PDBSymbolThunk.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeArray.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeBaseClass.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeBuiltin.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeCustom.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeDimension.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeEnum.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeFriend.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeFunctionArg.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeFunctionSig.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeManaged.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypePointer.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeTypedef.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeUDT.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeVTable.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeVTableShape.h"
#include "llvm/DebugInfo/PDB/PDBSymbolUnknown.h"
#include "llvm/DebugInfo/PDB/PDBSymbolUsingNamespace.h"
#include "llvm/DebugInfo/PDB/PDBSymDumper.h"
#include <memory>
#include <utility>

#include <memory>
#include <utility>

using namespace llvm;

PDBSymbol::PDBSymbol(const IPDBSession &PDBSession,
                     std::unique_ptr<IPDBRawSymbol> Symbol)
    : Session(PDBSession), RawSymbol(std::move(Symbol)) {}

PDBSymbol::~PDBSymbol() {}

#define FACTORY_SYMTAG_CASE(Tag, Type)                                         \
  case PDB_SymType::Tag:                                                       \
    return std::unique_ptr<PDBSymbol>(new Type(PDBSession, std::move(Symbol)));

std::unique_ptr<PDBSymbol>
PDBSymbol::create(const IPDBSession &PDBSession,
                  std::unique_ptr<IPDBRawSymbol> Symbol) {
  switch (Symbol->getSymTag()) {
    FACTORY_SYMTAG_CASE(Exe, PDBSymbolExe)
    FACTORY_SYMTAG_CASE(Compiland, PDBSymbolCompiland)
    FACTORY_SYMTAG_CASE(CompilandDetails, PDBSymbolCompilandDetails)
    FACTORY_SYMTAG_CASE(CompilandEnv, PDBSymbolCompilandEnv)
    FACTORY_SYMTAG_CASE(Function, PDBSymbolFunc)
    FACTORY_SYMTAG_CASE(Block, PDBSymbolBlock)
    FACTORY_SYMTAG_CASE(Data, PDBSymbolData)
    FACTORY_SYMTAG_CASE(Annotation, PDBSymbolAnnotation)
    FACTORY_SYMTAG_CASE(Label, PDBSymbolLabel)
    FACTORY_SYMTAG_CASE(PublicSymbol, PDBSymbolPublicSymbol)
    FACTORY_SYMTAG_CASE(UDT, PDBSymbolTypeUDT)
    FACTORY_SYMTAG_CASE(Enum, PDBSymbolTypeEnum)
    FACTORY_SYMTAG_CASE(FunctionSig, PDBSymbolTypeFunctionSig)
    FACTORY_SYMTAG_CASE(PointerType, PDBSymbolTypePointer)
    FACTORY_SYMTAG_CASE(ArrayType, PDBSymbolTypeArray)
    FACTORY_SYMTAG_CASE(BuiltinType, PDBSymbolTypeBuiltin)
    FACTORY_SYMTAG_CASE(Typedef, PDBSymbolTypeTypedef)
    FACTORY_SYMTAG_CASE(BaseClass, PDBSymbolTypeBaseClass)
    FACTORY_SYMTAG_CASE(Friend, PDBSymbolTypeFriend)
    FACTORY_SYMTAG_CASE(FunctionArg, PDBSymbolTypeFunctionArg)
    FACTORY_SYMTAG_CASE(FuncDebugStart, PDBSymbolFuncDebugStart)
    FACTORY_SYMTAG_CASE(FuncDebugEnd, PDBSymbolFuncDebugEnd)
    FACTORY_SYMTAG_CASE(UsingNamespace, PDBSymbolUsingNamespace)
    FACTORY_SYMTAG_CASE(VTableShape, PDBSymbolTypeVTableShape)
    FACTORY_SYMTAG_CASE(VTable, PDBSymbolTypeVTable)
    FACTORY_SYMTAG_CASE(Custom, PDBSymbolCustom)
    FACTORY_SYMTAG_CASE(Thunk, PDBSymbolThunk)
    FACTORY_SYMTAG_CASE(CustomType, PDBSymbolTypeCustom)
    FACTORY_SYMTAG_CASE(ManagedType, PDBSymbolTypeManaged)
    FACTORY_SYMTAG_CASE(Dimension, PDBSymbolTypeDimension)
  default:
    return std::unique_ptr<PDBSymbol>(
        new PDBSymbolUnknown(PDBSession, std::move(Symbol)));
  }
}

#define TRY_DUMP_TYPE(Type)                                                    \
  if (const Type *DerivedThis = dyn_cast<Type>(this))                          \
    Dumper.dump(OS, Indent, *DerivedThis);

#define ELSE_TRY_DUMP_TYPE(Type, Dumper) else TRY_DUMP_TYPE(Type, Dumper)

void PDBSymbol::defaultDump(raw_ostream &OS, int Indent) const {
  RawSymbol->dump(OS, Indent);
}

PDB_SymType PDBSymbol::getSymTag() const { return RawSymbol->getSymTag(); }

std::unique_ptr<IPDBEnumSymbols> PDBSymbol::findAllChildren() const {
  return findAllChildren(PDB_SymType::None);
}

std::unique_ptr<IPDBEnumSymbols>
PDBSymbol::findAllChildren(PDB_SymType Type) const {
  return RawSymbol->findChildren(Type);
}

std::unique_ptr<IPDBEnumSymbols>
PDBSymbol::findChildren(PDB_SymType Type, StringRef Name,
                        PDB_NameSearchFlags Flags) const {
  return RawSymbol->findChildren(Type, Name, Flags);
}

std::unique_ptr<IPDBEnumSymbols>
PDBSymbol::findChildrenByRVA(PDB_SymType Type, StringRef Name,
                             PDB_NameSearchFlags Flags, uint32_t RVA) const {
  return RawSymbol->findChildrenByRVA(Type, Name, Flags, RVA);
}

std::unique_ptr<IPDBEnumSymbols>
PDBSymbol::findInlineFramesByRVA(uint32_t RVA) const {
  return RawSymbol->findInlineFramesByRVA(RVA);
}

std::unique_ptr<IPDBEnumSymbols>
PDBSymbol::getChildStats(TagStats &Stats) const {
  std::unique_ptr<IPDBEnumSymbols> Result(findAllChildren());
  Stats.clear();
  while (auto Child = Result->getNext()) {
    ++Stats[Child->getSymTag()];
  }
  Result->reset();
  return Result;
}
