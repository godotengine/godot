//===- SMLoc.h - Source location for use with diagnostics -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the SMLoc class.  This class encapsulates a location in
// source code for use in diagnostics.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_SMLOC_H
#define LLVM_SUPPORT_SMLOC_H

#include <cassert>

namespace llvm {

/// Represents a location in source code.
class SMLoc {
  const char *Ptr;
public:
  SMLoc() : Ptr(nullptr) {}

  bool isValid() const { return Ptr != nullptr; }

  bool operator==(const SMLoc &RHS) const { return RHS.Ptr == Ptr; }
  bool operator!=(const SMLoc &RHS) const { return RHS.Ptr != Ptr; }

  const char *getPointer() const { return Ptr; }

  static SMLoc getFromPointer(const char *Ptr) {
    SMLoc L;
    L.Ptr = Ptr;
    return L;
  }
};

/// Represents a range in source code.
///
/// SMRange is implemented using a half-open range, as is the convention in C++.
/// In the string "abc", the range (1,3] represents the substring "bc", and the
/// range (2,2] represents an empty range between the characters "b" and "c".
class SMRange {
public:
  SMLoc Start, End;

  SMRange() {}
  SMRange(SMLoc St, SMLoc En) : Start(St), End(En) {
    assert(Start.isValid() == End.isValid() &&
           "Start and end should either both be valid or both be invalid!");
  }
  
  bool isValid() const { return Start.isValid(); }
};
  
} // end namespace llvm

#endif

