//===- PDBSymbolData.cpp - PDB data (e.g. variable) accessors ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/PDBSymbolData.h"

#include "llvm/DebugInfo/PDB/IPDBSession.h"
#include "llvm/DebugInfo/PDB/PDBSymDumper.h"

#include <utility>

using namespace llvm;

PDBSymbolData::PDBSymbolData(const IPDBSession &PDBSession,
                             std::unique_ptr<IPDBRawSymbol> DataSymbol)
    : PDBSymbol(PDBSession, std::move(DataSymbol)) {}

std::unique_ptr<PDBSymbol> PDBSymbolData::getType() const {
  return Session.getSymbolById(getTypeId());
}

void PDBSymbolData::dump(PDBSymDumper &Dumper) const { Dumper.dump(*this); }
