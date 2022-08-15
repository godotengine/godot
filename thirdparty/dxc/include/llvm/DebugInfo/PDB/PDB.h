//===- PDB.h - base header file for creating a PDB reader -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_PDB_H
#define LLVM_DEBUGINFO_PDB_PDB_H

#include "PDBTypes.h"
#include <memory>

namespace llvm {
class StringRef;

PDB_ErrorCode loadDataForPDB(PDB_ReaderType Type, StringRef Path,
                             std::unique_ptr<IPDBSession> &Session);

PDB_ErrorCode loadDataForEXE(PDB_ReaderType Type, StringRef Path,
                             std::unique_ptr<IPDBSession> &Session);
}

#endif
