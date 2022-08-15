//===- PDBSymbolTypePointer.cpp -----------------------------------*- C++ -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/PDBSymbolTypePointer.h"

#include "llvm/DebugInfo/PDB/IPDBSession.h"
#include "llvm/DebugInfo/PDB/PDBSymDumper.h"

#include <utility>

using namespace llvm;

PDBSymbolTypePointer::PDBSymbolTypePointer(
    const IPDBSession &PDBSession, std::unique_ptr<IPDBRawSymbol> Symbol)
    : PDBSymbol(PDBSession, std::move(Symbol)) {}

std::unique_ptr<PDBSymbol> PDBSymbolTypePointer::getPointeeType() const {
  return Session.getSymbolById(getTypeId());
}

void PDBSymbolTypePointer::dump(PDBSymDumper &Dumper) const {
  Dumper.dump(*this);
}
