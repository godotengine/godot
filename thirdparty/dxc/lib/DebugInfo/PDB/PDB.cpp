//===- PDB.cpp - base header file for creating a PDB reader -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/PDB.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Config/config.h"
#include "llvm/DebugInfo/PDB/IPDBSession.h"
#include "llvm/DebugInfo/PDB/PDB.h"

#if HAVE_DIA_SDK
#include "llvm/DebugInfo/PDB/DIA/DIASession.h"
#endif

using namespace llvm;

PDB_ErrorCode llvm::loadDataForPDB(PDB_ReaderType Type, StringRef Path,
                                   std::unique_ptr<IPDBSession> &Session) {
  // Create the correct concrete instance type based on the value of Type.
#if HAVE_DIA_SDK
  return DIASession::createFromPdb(Path, Session);
#endif
  return PDB_ErrorCode::NoPdbImpl;
}

PDB_ErrorCode llvm::loadDataForEXE(PDB_ReaderType Type, StringRef Path,
                                   std::unique_ptr<IPDBSession> &Session) {
// Create the correct concrete instance type based on the value of Type.
#if HAVE_DIA_SDK
  return DIASession::createFromExe(Path, Session);
#endif
  return PDB_ErrorCode::NoPdbImpl;
}
