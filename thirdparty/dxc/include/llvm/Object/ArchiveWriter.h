//===- ArchiveWriter.h - ar archive file format writer ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Declares the writeArchive function for writing an archive file.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_ARCHIVEWRITER_H
#define LLVM_OBJECT_ARCHIVEWRITER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Object/Archive.h"
#include "llvm/Support/FileSystem.h"

namespace llvm {

class NewArchiveIterator {
  bool IsNewMember;
  StringRef Name;

  object::Archive::child_iterator OldI;

  StringRef NewFilename;

public:
  NewArchiveIterator(object::Archive::child_iterator I, StringRef Name);
  NewArchiveIterator(StringRef I, StringRef Name);
  bool isNewMember() const;
  StringRef getName() const;

  object::Archive::child_iterator getOld() const;

  StringRef getNew() const;
  llvm::ErrorOr<int> getFD(sys::fs::file_status &NewStatus) const;
  const sys::fs::file_status &getStatus() const;
};

std::pair<StringRef, std::error_code>
writeArchive(StringRef ArcName, std::vector<NewArchiveIterator> &NewMembers,
             bool WriteSymtab, object::Archive::Kind Kind, bool Deterministic);
}

#endif
