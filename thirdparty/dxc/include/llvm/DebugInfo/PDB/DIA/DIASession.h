//===- DIASession.h - DIA implementation of IPDBSession ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_DIA_DIASESSION_H
#define LLVM_DEBUGINFO_PDB_DIA_DIASESSION_H

#include "DIASupport.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/PDB/IPDBSession.h"

namespace llvm {
class DIASession : public IPDBSession {
public:
  explicit DIASession(CComPtr<IDiaSession> DiaSession);

  static PDB_ErrorCode createFromPdb(StringRef Path,
                                     std::unique_ptr<IPDBSession> &Session);
  static PDB_ErrorCode createFromExe(StringRef Path,
                                     std::unique_ptr<IPDBSession> &Session);

  uint64_t getLoadAddress() const override;
  void setLoadAddress(uint64_t Address) override;
  std::unique_ptr<PDBSymbolExe> getGlobalScope() const override;
  std::unique_ptr<PDBSymbol> getSymbolById(uint32_t SymbolId) const override;

  std::unique_ptr<PDBSymbol>
  findSymbolByAddress(uint64_t Address, PDB_SymType Type) const override;

  std::unique_ptr<IPDBEnumLineNumbers>
  findLineNumbersByAddress(uint64_t Address, uint32_t Length) const override;

  std::unique_ptr<IPDBEnumSourceFiles> getAllSourceFiles() const override;
  std::unique_ptr<IPDBEnumSourceFiles> getSourceFilesForCompiland(
      const PDBSymbolCompiland &Compiland) const override;
  std::unique_ptr<IPDBSourceFile>
  getSourceFileById(uint32_t FileId) const override;

  std::unique_ptr<IPDBEnumDataStreams> getDebugStreams() const override;

private:
  CComPtr<IDiaSession> Session;
};
}

#endif
