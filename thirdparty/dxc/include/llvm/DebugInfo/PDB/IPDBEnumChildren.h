//===- IPDBEnumChildren.h - base interface for child enumerator -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_IPDBENUMCHILDREN_H
#define LLVM_DEBUGINFO_PDB_IPDBENUMCHILDREN_H

#include "PDBTypes.h"
#include <memory>

namespace llvm {

template <typename ChildType> class IPDBEnumChildren {
public:
  typedef std::unique_ptr<ChildType> ChildTypePtr;
  typedef IPDBEnumChildren<ChildType> MyType;

  virtual ~IPDBEnumChildren() {}

  virtual uint32_t getChildCount() const = 0;
  virtual ChildTypePtr getChildAtIndex(uint32_t Index) const = 0;
  virtual ChildTypePtr getNext() = 0;
  virtual void reset() = 0;
  virtual MyType *clone() const = 0;
};
}

#endif
