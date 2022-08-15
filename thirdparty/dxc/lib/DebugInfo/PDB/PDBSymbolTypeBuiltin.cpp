//===- PDBSymbolTypeBuiltin.cpp - ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/PDBSymbolTypeBuiltin.h"

#include "llvm/DebugInfo/PDB/PDBSymDumper.h"

#include <utility>

using namespace llvm;

PDBSymbolTypeBuiltin::PDBSymbolTypeBuiltin(
    const IPDBSession &PDBSession, std::unique_ptr<IPDBRawSymbol> Symbol)
    : PDBSymbol(PDBSession, std::move(Symbol)) {}

void PDBSymbolTypeBuiltin::dump(PDBSymDumper &Dumper) const {
  Dumper.dump(*this);
}
