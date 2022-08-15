//===--- OptSpecifier.h - Option Specifiers ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OPTION_OPTSPECIFIER_H
#define LLVM_OPTION_OPTSPECIFIER_H

#include "llvm/Support/Compiler.h"

namespace llvm {
namespace opt {
  class Option;

  /// OptSpecifier - Wrapper class for abstracting references to option IDs.
  class OptSpecifier {
    unsigned ID;

  private:
    explicit OptSpecifier(bool) = delete;

  public:
    OptSpecifier() : ID(0) {}
    /*implicit*/ OptSpecifier(unsigned ID) : ID(ID) {}
    /*implicit*/ OptSpecifier(const Option *Opt);

    bool isValid() const { return ID != 0; }

    unsigned getID() const { return ID; }

    bool operator==(OptSpecifier Opt) const { return ID == Opt.getID(); }
    bool operator!=(OptSpecifier Opt) const { return !(*this == Opt); }
  };
}
}

#endif
