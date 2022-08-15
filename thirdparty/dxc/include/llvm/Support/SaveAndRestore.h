//===-- SaveAndRestore.h - Utility  -------------------------------*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides utility classes that use RAII to save and restore
/// values.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_SAVEANDRESTORE_H
#define LLVM_SUPPORT_SAVEANDRESTORE_H

namespace llvm {

/// A utility class that uses RAII to save and restore the value of a variable.
template <typename T> struct SaveAndRestore {
  SaveAndRestore(T &X) : X(X), OldValue(X) {}
  SaveAndRestore(T &X, const T &NewValue) : X(X), OldValue(X) {
    X = NewValue;
  }
  ~SaveAndRestore() { X = OldValue; }
  T get() { return OldValue; }

private:
  T &X;
  T OldValue;
};

/// Similar to \c SaveAndRestore.  Operates only on bools; the old value of a
/// variable is saved, and during the dstor the old value is or'ed with the new
/// value.
struct SaveOr {
  SaveOr(bool &X) : X(X), OldValue(X) { X = false; }
  ~SaveOr() { X |= OldValue; }

private:
  bool &X;
  const bool OldValue;
};

} // namespace llvm

#endif
