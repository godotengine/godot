//===- IPDBSession.h - base interface for a PDB symbol context --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_IPDBSESSION_H
#define LLVM_DEBUGINFO_PDB_IPDBSESSION_H

#include "PDBTypes.h"
#include "llvm/Support/Casting.h"
#include <memory>

namespace llvm {

class PDBSymbolCompiland;
class PDBSymbolExe;

/// IPDBSession defines an interface used to provide a context for querying
/// debug information from a debug data source (for example, a PDB).
class IPDBSession {
public:
  virtual ~IPDBSession();

  virtual uint64_t getLoadAddress() const = 0;
  virtual void setLoadAddress(uint64_t Address) = 0;
  virtual std::unique_ptr<PDBSymbolExe> getGlobalScope() const = 0;
  virtual std::unique_ptr<PDBSymbol> getSymbolById(uint32_t SymbolId) const = 0;

  template <typename T>
  std::unique_ptr<T> getConcreteSymbolById(uint32_t SymbolId) const {
    auto Symbol(getSymbolById(SymbolId));
    if (!Symbol)
      return nullptr;

    T *ConcreteSymbol = dyn_cast<T>(Symbol.get());
    if (!ConcreteSymbol)
      return nullptr;
    Symbol.release();
    return std::unique_ptr<T>(ConcreteSymbol);
  }

  virtual std::unique_ptr<PDBSymbol>
  findSymbolByAddress(uint64_t Address, PDB_SymType Type) const = 0;
  virtual std::unique_ptr<IPDBEnumLineNumbers>
  findLineNumbersByAddress(uint64_t Address, uint32_t Length) const = 0;

  virtual std::unique_ptr<IPDBEnumSourceFiles> getAllSourceFiles() const = 0;
  virtual std::unique_ptr<IPDBEnumSourceFiles>
  getSourceFilesForCompiland(const PDBSymbolCompiland &Compiland) const = 0;
  virtual std::unique_ptr<IPDBSourceFile>
  getSourceFileById(uint32_t FileId) const = 0;

  virtual std::unique_ptr<IPDBEnumDataStreams> getDebugStreams() const = 0;
};
}

#endif
