//===- DIASourceFile.cpp - DIA implementation of IPDBSourceFile -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/DIA/DIAEnumSymbols.h"
#include "llvm/DebugInfo/PDB/DIA/DIASession.h"
#include "llvm/DebugInfo/PDB/DIA/DIASourceFile.h"
#include "llvm/Support/ConvertUTF.h"

using namespace llvm;

DIASourceFile::DIASourceFile(const DIASession &PDBSession,
                             CComPtr<IDiaSourceFile> DiaSourceFile)
    : Session(PDBSession), SourceFile(DiaSourceFile) {}

std::string DIASourceFile::getFileName() const {
  CComBSTR FileName16;
  HRESULT Result = SourceFile->get_fileName(&FileName16);
  if (S_OK != Result)
    return std::string();

  std::string FileName8;
  llvm::ArrayRef<char> FileNameBytes(reinterpret_cast<char *>(FileName16.m_str),
                                     FileName16.ByteLength());
  llvm::convertUTF16ToUTF8String(FileNameBytes, FileName8);
  return FileName8;
}

uint32_t DIASourceFile::getUniqueId() const {
  DWORD Id;
  return (S_OK == SourceFile->get_uniqueId(&Id)) ? Id : 0;
}

std::string DIASourceFile::getChecksum() const {
  DWORD ByteSize = 0;
  HRESULT Result = SourceFile->get_checksum(0, &ByteSize, nullptr);
  if (ByteSize == 0)
    return std::string();
  std::vector<BYTE> ChecksumBytes(ByteSize);
  Result = SourceFile->get_checksum(ByteSize, &ByteSize, &ChecksumBytes[0]);
  if (S_OK != Result)
    return std::string();
  return std::string(ChecksumBytes.begin(), ChecksumBytes.end());
}

PDB_Checksum DIASourceFile::getChecksumType() const {
  DWORD Type;
  HRESULT Result = SourceFile->get_checksumType(&Type);
  if (S_OK != Result)
    return PDB_Checksum::None;
  return static_cast<PDB_Checksum>(Type);
}

std::unique_ptr<IPDBEnumSymbols> DIASourceFile::getCompilands() const {
  CComPtr<IDiaEnumSymbols> DiaEnumerator;
  HRESULT Result = SourceFile->get_compilands(&DiaEnumerator);
  if (S_OK != Result)
    return nullptr;

  return std::unique_ptr<IPDBEnumSymbols>(
      new DIAEnumSymbols(Session, DiaEnumerator));
}
