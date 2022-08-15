//===--- StringSet.h - The LLVM Compiler Driver -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open
// Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  StringSet - A set-like wrapper for the StringMap.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_STRINGSET_H
#define LLVM_ADT_STRINGSET_H

#include "llvm/ADT/StringMap.h"

namespace llvm {

  /// StringSet - A wrapper for StringMap that provides set-like functionality.
  template <class AllocatorTy = llvm::MallocAllocator>
  class StringSet : public llvm::StringMap<char, AllocatorTy> {
    typedef llvm::StringMap<char, AllocatorTy> base;
  public:

    std::pair<typename base::iterator, bool> insert(StringRef Key) {
      assert(!Key.empty());
      return base::insert(std::make_pair(Key, '\0'));
    }
  };
}

#endif // LLVM_ADT_STRINGSET_H
