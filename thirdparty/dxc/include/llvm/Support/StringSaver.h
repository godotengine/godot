//===- llvm/Support/StringSaver.h -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_STRINGSAVER_H
#define LLVM_SUPPORT_STRINGSAVER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Allocator.h"

namespace llvm {

/// \brief Saves strings in the inheritor's stable storage and returns a stable
/// raw character pointer.
class StringSaver {
protected:
  ~StringSaver() {}
  virtual const char *saveImpl(StringRef S);

public:
  StringSaver(BumpPtrAllocator &Alloc) : Alloc(Alloc) {}
  const char *save(const char *S) { return save(StringRef(S)); }
  const char *save(StringRef S) { return saveImpl(S); }
  const char *save(const Twine &S) { return save(StringRef(S.str())); }
  const char *save(std::string &S) { return save(StringRef(S)); }

private:
  BumpPtrAllocator &Alloc;
};

class BumpPtrStringSaver final : public StringSaver {
public:
  BumpPtrStringSaver(BumpPtrAllocator &Alloc) : StringSaver(Alloc) {}
};
}
#endif
