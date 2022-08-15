//===- lib/MC/MCLabel.cpp - MCLabel implementation ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCLabel.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

void MCLabel::print(raw_ostream &OS) const {
  OS << '"' << getInstance() << '"';
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void MCLabel::dump() const {
  print(dbgs());
}
#endif
