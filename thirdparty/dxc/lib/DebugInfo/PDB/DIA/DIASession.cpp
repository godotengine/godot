//===- DIASession.cpp - DIA implementation of IPDBSession -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/DebugInfo/PDB/DIA/DIAEnumDebugStreams.h"
#include "llvm/DebugInfo/PDB/DIA/DIAEnumLineNumbers.h"
#include "llvm/DebugInfo/PDB/DIA/DIAEnumSourceFiles.h"
#include "llvm/DebugInfo/PDB/DIA/DIARawSymbol.h"
#include "llvm/DebugInfo/PDB/DIA/DIASession.h"
#include "llvm/DebugInfo/PDB/DIA/DIASourceFile.h"
#include "llvm/DebugInfo/PDB/PDBSymbolCompiland.h"
#include "llvm/DebugInfo/PDB/PDBSymbolExe.h"
#include "llvm/Support/ConvertUTF.h"

using namespace llvm;

namespace {}

DIASession::DIASession(CComPtr<IDiaSession> DiaSession) : Session(DiaSession) {}

PDB_ErrorCode DIASession::createFromPdb(StringRef Path,
                                        std::unique_ptr<IPDBSession> &Session) {
  CComPtr<IDiaDataSource> DiaDataSource;
  CComPtr<IDiaSession> DiaSession;

  // We assume that CoInitializeEx has already been called by the executable.
  HRESULT Result = ::CoCreateInstance(
      CLSID_DiaSource, nullptr, CLSCTX_INPROC_SERVER, IID_IDiaDataSource,
      reinterpret_cast<LPVOID *>(&DiaDataSource));
  if (FAILED(Result))
    return PDB_ErrorCode::NoPdbImpl;

  llvm::SmallVector<UTF16, 128> Path16;
  if (!llvm::convertUTF8ToUTF16String(Path, Path16))
    return PDB_ErrorCode::InvalidPath;

  const wchar_t *Path16Str = reinterpret_cast<const wchar_t*>(Path16.data());
  if (FAILED(Result = DiaDataSource->loadDataFromPdb(Path16Str))) {
    if (Result == E_PDB_NOT_FOUND)
      return PDB_ErrorCode::InvalidPath;
    else if (Result == E_PDB_FORMAT)
      return PDB_ErrorCode::InvalidFileFormat;
    else if (Result == E_INVALIDARG)
      return PDB_ErrorCode::InvalidParameter;
    else if (Result == E_UNEXPECTED)
      return PDB_ErrorCode::AlreadyLoaded;
    else
      return PDB_ErrorCode::UnknownError;
  }

  if (FAILED(Result = DiaDataSource->openSession(&DiaSession))) {
    if (Result == E_OUTOFMEMORY)
      return PDB_ErrorCode::NoMemory;
    else
      return PDB_ErrorCode::UnknownError;
  }

  Session.reset(new DIASession(DiaSession));
  return PDB_ErrorCode::Success;
}

PDB_ErrorCode DIASession::createFromExe(StringRef Path,
                                        std::unique_ptr<IPDBSession> &Session) {
  CComPtr<IDiaDataSource> DiaDataSource;
  CComPtr<IDiaSession> DiaSession;

  // We assume that CoInitializeEx has already been called by the executable.
  HRESULT Result = ::CoCreateInstance(
      CLSID_DiaSource, nullptr, CLSCTX_INPROC_SERVER, IID_IDiaDataSource,
      reinterpret_cast<LPVOID *>(&DiaDataSource));
  if (FAILED(Result))
    return PDB_ErrorCode::NoPdbImpl;

  llvm::SmallVector<UTF16, 128> Path16;
  if (!llvm::convertUTF8ToUTF16String(Path, Path16))
    return PDB_ErrorCode::InvalidPath;

  const wchar_t *Path16Str = reinterpret_cast<const wchar_t *>(Path16.data());
  if (FAILED(Result =
                 DiaDataSource->loadDataForExe(Path16Str, nullptr, nullptr))) {
    if (Result == E_PDB_NOT_FOUND)
      return PDB_ErrorCode::InvalidPath;
    else if (Result == E_PDB_FORMAT)
      return PDB_ErrorCode::InvalidFileFormat;
    else if (Result == E_PDB_INVALID_SIG || Result == E_PDB_INVALID_AGE)
      return PDB_ErrorCode::DebugInfoMismatch;
    else if (Result == E_INVALIDARG)
      return PDB_ErrorCode::InvalidParameter;
    else if (Result == E_UNEXPECTED)
      return PDB_ErrorCode::AlreadyLoaded;
    else
      return PDB_ErrorCode::UnknownError;
  }

  if (FAILED(Result = DiaDataSource->openSession(&DiaSession))) {
    if (Result == E_OUTOFMEMORY)
      return PDB_ErrorCode::NoMemory;
    else
      return PDB_ErrorCode::UnknownError;
  }

  Session.reset(new DIASession(DiaSession));
  return PDB_ErrorCode::Success;
}

uint64_t DIASession::getLoadAddress() const {
  uint64_t LoadAddress;
  bool success = (S_OK == Session->get_loadAddress(&LoadAddress));
  return (success) ? LoadAddress : 0;
}

void DIASession::setLoadAddress(uint64_t Address) {
  Session->put_loadAddress(Address);
}

std::unique_ptr<PDBSymbolExe> DIASession::getGlobalScope() const {
  CComPtr<IDiaSymbol> GlobalScope;
  if (S_OK != Session->get_globalScope(&GlobalScope))
    return nullptr;

  auto RawSymbol = llvm::make_unique<DIARawSymbol>(*this, GlobalScope);
  auto PdbSymbol(PDBSymbol::create(*this, std::move(RawSymbol)));
  std::unique_ptr<PDBSymbolExe> ExeSymbol(
      static_cast<PDBSymbolExe *>(PdbSymbol.release()));
  return ExeSymbol;
}

std::unique_ptr<PDBSymbol> DIASession::getSymbolById(uint32_t SymbolId) const {
  CComPtr<IDiaSymbol> LocatedSymbol;
  if (S_OK != Session->symbolById(SymbolId, &LocatedSymbol))
    return nullptr;

  auto RawSymbol = llvm::make_unique<DIARawSymbol>(*this, LocatedSymbol);
  return PDBSymbol::create(*this, std::move(RawSymbol));
}

std::unique_ptr<PDBSymbol>
DIASession::findSymbolByAddress(uint64_t Address, PDB_SymType Type) const {
  enum SymTagEnum EnumVal = static_cast<enum SymTagEnum>(Type);

  CComPtr<IDiaSymbol> Symbol;
  if (S_OK != Session->findSymbolByVA(Address, EnumVal, &Symbol)) {
    ULONGLONG LoadAddr = 0;
    if (S_OK != Session->get_loadAddress(&LoadAddr))
      return nullptr;
    DWORD RVA = static_cast<DWORD>(Address - LoadAddr);
    if (S_OK != Session->findSymbolByRVA(RVA, EnumVal, &Symbol))
      return nullptr;
  }
  auto RawSymbol = llvm::make_unique<DIARawSymbol>(*this, Symbol);
  return PDBSymbol::create(*this, std::move(RawSymbol));
}

std::unique_ptr<IPDBEnumLineNumbers>
DIASession::findLineNumbersByAddress(uint64_t Address, uint32_t Length) const {
  CComPtr<IDiaEnumLineNumbers> LineNumbers;
  if (S_OK != Session->findLinesByVA(Address, Length, &LineNumbers))
    return nullptr;

  return llvm::make_unique<DIAEnumLineNumbers>(LineNumbers);
}

std::unique_ptr<IPDBEnumSourceFiles> DIASession::getAllSourceFiles() const {
  CComPtr<IDiaEnumSourceFiles> Files;
  if (S_OK != Session->findFile(nullptr, nullptr, nsNone, &Files))
    return nullptr;

  return llvm::make_unique<DIAEnumSourceFiles>(*this, Files);
}

std::unique_ptr<IPDBEnumSourceFiles> DIASession::getSourceFilesForCompiland(
    const PDBSymbolCompiland &Compiland) const {
  CComPtr<IDiaEnumSourceFiles> Files;

  const DIARawSymbol &RawSymbol =
      static_cast<const DIARawSymbol &>(Compiland.getRawSymbol());
  if (S_OK !=
      Session->findFile(RawSymbol.getDiaSymbol(), nullptr, nsNone, &Files))
    return nullptr;

  return llvm::make_unique<DIAEnumSourceFiles>(*this, Files);
}

std::unique_ptr<IPDBSourceFile>
DIASession::getSourceFileById(uint32_t FileId) const {
  CComPtr<IDiaSourceFile> LocatedFile;
  if (S_OK != Session->findFileById(FileId, &LocatedFile))
    return nullptr;

  return llvm::make_unique<DIASourceFile>(*this, LocatedFile);
}

std::unique_ptr<IPDBEnumDataStreams> DIASession::getDebugStreams() const {
  CComPtr<IDiaEnumDebugStreams> DiaEnumerator;
  if (S_OK != Session->getEnumDebugStreams(&DiaEnumerator))
    return nullptr;

  return llvm::make_unique<DIAEnumDebugStreams>(DiaEnumerator);
}
