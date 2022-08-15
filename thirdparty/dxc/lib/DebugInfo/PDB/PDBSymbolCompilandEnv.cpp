//===- PDBSymbolCompilandEnv.cpp - compiland env variables ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/PDBSymbolCompilandEnv.h"

#include "llvm/DebugInfo/PDB/IPDBRawSymbol.h"
#include "llvm/DebugInfo/PDB/PDBSymbol.h"
#include "llvm/DebugInfo/PDB/PDBSymDumper.h"

#include <utility>

using namespace llvm;

PDBSymbolCompilandEnv::PDBSymbolCompilandEnv(
    const IPDBSession &PDBSession, std::unique_ptr<IPDBRawSymbol> Symbol)
    : PDBSymbol(PDBSession, std::move(Symbol)) {}

std::string PDBSymbolCompilandEnv::getValue() const {
  // call RawSymbol->getValue() and convert the result to an std::string.
  return std::string();
}

void PDBSymbolCompilandEnv::dump(PDBSymDumper &Dumper) const {
  Dumper.dump(*this);
}
