//===- PDBSymbolTypeUDT.cpp - --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/PDBSymbolTypeUDT.h"

#include "llvm/DebugInfo/PDB/PDBSymDumper.h"

#include <utility>

using namespace llvm;

PDBSymbolTypeUDT::PDBSymbolTypeUDT(const IPDBSession &PDBSession,
                                   std::unique_ptr<IPDBRawSymbol> Symbol)
    : PDBSymbol(PDBSession, std::move(Symbol)) {}

void PDBSymbolTypeUDT::dump(PDBSymDumper &Dumper) const { Dumper.dump(*this); }
