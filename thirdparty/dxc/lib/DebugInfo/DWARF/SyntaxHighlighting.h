//===-- SyntaxHighlighting.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_DEBUGINFO_SYNTAXHIGHLIGHTING_H
#define LLVM_LIB_DEBUGINFO_SYNTAXHIGHLIGHTING_H

#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace dwarf {
namespace syntax {

// Symbolic names for various syntax elements.
enum HighlightColor { Address, String, Tag, Attribute, Enumerator };

/// An RAII object that temporarily switches an output stream to a
/// specific color.
class WithColor {
  llvm::raw_ostream &OS;

public:
  /// To be used like this: WithColor(OS, syntax::String) << "text";
  WithColor(llvm::raw_ostream &OS, enum HighlightColor Type);
  ~WithColor();

  llvm::raw_ostream& get() { return OS; }
  operator llvm::raw_ostream& () { return OS; }
};
}
}
}

#endif
