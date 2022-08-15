//===-- DebugLoc.cpp - Implement DebugLoc class ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DebugLoc.h"
#include "LLVMContextImpl.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/IR/DebugInfo.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
// DebugLoc Implementation
//===----------------------------------------------------------------------===//
DebugLoc::DebugLoc(const DILocation *L) : Loc(const_cast<DILocation *>(L)) {}
DebugLoc::DebugLoc(const MDNode *L) : Loc(const_cast<MDNode *>(L)) {}

DILocation *DebugLoc::get() const {
  return cast_or_null<DILocation>(Loc.get());
}

unsigned DebugLoc::getLine() const {
  assert(get() && "Expected valid DebugLoc");
  return get()->getLine();
}

unsigned DebugLoc::getCol() const {
  assert(get() && "Expected valid DebugLoc");
  return get()->getColumn();
}

MDNode *DebugLoc::getScope() const {
  assert(get() && "Expected valid DebugLoc");
  return get()->getScope();
}

DILocation *DebugLoc::getInlinedAt() const {
  assert(get() && "Expected valid DebugLoc");
  return get()->getInlinedAt();
}

MDNode *DebugLoc::getInlinedAtScope() const {
  return cast<DILocation>(Loc)->getInlinedAtScope();
}

DebugLoc DebugLoc::getFnDebugLoc() const {
  // FIXME: Add a method on \a DILocation that does this work.
  const MDNode *Scope = getInlinedAtScope();
  if (auto *SP = getDISubprogram(Scope))
    return DebugLoc::get(SP->getScopeLine(), 0, SP);

  return DebugLoc();
}

DebugLoc DebugLoc::get(unsigned Line, unsigned Col, const MDNode *Scope,
                       const MDNode *InlinedAt) {
  // If no scope is available, this is an unknown location.
  if (!Scope)
    return DebugLoc();

  return DILocation::get(Scope->getContext(), Line, Col,
                         const_cast<MDNode *>(Scope),
                         const_cast<MDNode *>(InlinedAt));
}

void DebugLoc::dump() const {
#ifndef NDEBUG
  if (!Loc)
    return;

  dbgs() << getLine();
  if (getCol() != 0)
    dbgs() << ',' << getCol();
  if (DebugLoc InlinedAtDL = DebugLoc(getInlinedAt())) {
    dbgs() << " @ ";
    InlinedAtDL.dump();
  } else
    dbgs() << "\n";
#endif
}

void DebugLoc::print(raw_ostream &OS) const {
  if (!Loc)
    return;

  // Print source line info.
  auto *Scope = cast<DIScope>(getScope());
  OS << Scope->getFilename();
  OS << ':' << getLine();
  if (getCol() != 0)
    OS << ':' << getCol();

  if (DebugLoc InlinedAtDL = getInlinedAt()) {
    OS << " @[ ";
    InlinedAtDL.print(OS);
    OS << " ]";
  }
}
