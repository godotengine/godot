//===-- User.cpp - Implement the User class -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/User.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Operator.h"

namespace llvm {
class BasicBlock;

//===----------------------------------------------------------------------===//
//                                 User Class
//===----------------------------------------------------------------------===//

void User::anchor() {}

void User::replaceUsesOfWith(Value *From, Value *To) {
  if (From == To) return;   // Duh what?

  assert((!isa<Constant>(this) || isa<GlobalValue>(this)) &&
         "Cannot call User::replaceUsesOfWith on a constant!");

  for (unsigned i = 0, E = getNumOperands(); i != E; ++i)
    if (getOperand(i) == From) {  // Is This operand is pointing to oldval?
      // The side effects of this setOperand call include linking to
      // "To", adding "this" to the uses list of To, and
      // most importantly, removing "this" from the use list of "From".
      setOperand(i, To); // Fix it now...
    }
}

//===----------------------------------------------------------------------===//
//                         User allocHungoffUses Implementation
//===----------------------------------------------------------------------===//

void User::allocHungoffUses(unsigned N, bool IsPhi) {
  assert(HasHungOffUses && "alloc must have hung off uses");

  static_assert(AlignOf<Use>::Alignment >= AlignOf<Use::UserRef>::Alignment,
                "Alignment is insufficient for 'hung-off-uses' pieces");
  static_assert(AlignOf<Use::UserRef>::Alignment >=
                    AlignOf<BasicBlock *>::Alignment,
                "Alignment is insufficient for 'hung-off-uses' pieces");

  // Allocate the array of Uses, followed by a pointer (with bottom bit set) to
  // the User.
  size_t size = N * sizeof(Use) + sizeof(Use::UserRef);
  if (IsPhi)
    size += N * sizeof(BasicBlock *);
  Use *Begin = static_cast<Use*>(::operator new(size));
  Use *End = Begin + N;
  (void) new(End) Use::UserRef(const_cast<User*>(this), 1);
  setOperandList(Use::initTags(Begin, End));
}

void User::growHungoffUses(unsigned NewNumUses, bool IsPhi) {
  assert(HasHungOffUses && "realloc must have hung off uses");

  unsigned OldNumUses = getNumOperands();

  // We don't support shrinking the number of uses.  We wouldn't have enough
  // space to copy the old uses in to the new space.
  assert(NewNumUses > OldNumUses && "realloc must grow num uses");

  Use *OldOps = getOperandList();
  allocHungoffUses(NewNumUses, IsPhi);
  Use *NewOps = getOperandList();

  // Now copy from the old operands list to the new one.
  std::copy(OldOps, OldOps + OldNumUses, NewOps);

  // If this is a Phi, then we need to copy the BB pointers too.
  if (IsPhi) {
    auto *OldPtr =
        reinterpret_cast<char *>(OldOps + OldNumUses) + sizeof(Use::UserRef);
    auto *NewPtr =
        reinterpret_cast<char *>(NewOps + NewNumUses) + sizeof(Use::UserRef);
    std::copy(OldPtr, OldPtr + (OldNumUses * sizeof(BasicBlock *)), NewPtr);
  }
  Use::zap(OldOps, OldOps + OldNumUses, true);
}

//===----------------------------------------------------------------------===//
//                         User operator new Implementations
//===----------------------------------------------------------------------===//

void *User::operator new(size_t Size, unsigned Us) {
  assert(Us < (1u << NumUserOperandsBits) && "Too many operands");
  void *Storage = ::operator new(Size + sizeof(Use) * Us);
  Use *Start = static_cast<Use*>(Storage);
  Use *End = Start + Us;
  User *Obj = reinterpret_cast<User*>(End);
  Obj->NumUserOperands = Us;
  Obj->HasHungOffUses = false;
  Use::initTags(Start, End);
  return Obj;
}

void *User::operator new(size_t Size) {
  // Allocate space for a single Use*
  void *Storage = ::operator new(Size + sizeof(Use *));
  Use **HungOffOperandList = static_cast<Use **>(Storage);
  User *Obj = reinterpret_cast<User *>(HungOffOperandList + 1);
  Obj->NumUserOperands = 0;
  Obj->HasHungOffUses = true;
  *HungOffOperandList = nullptr;
  return Obj;
}

//===----------------------------------------------------------------------===//
//                         User operator delete Implementation
//===----------------------------------------------------------------------===//

void User::operator delete(void *Usr) {
  // Hung off uses use a single Use* before the User, while other subclasses
  // use a Use[] allocated prior to the user.
  User *Obj = static_cast<User *>(Usr);
  if (Obj->HasHungOffUses) {
    Use **HungOffOperandList = static_cast<Use **>(Usr) - 1;
    // drop the hung off uses.
    Use::zap(*HungOffOperandList, *HungOffOperandList + Obj->NumUserOperands,
             /* Delete */ true);
    ::operator delete(HungOffOperandList);
  } else {
    Use *Storage = static_cast<Use *>(Usr) - Obj->NumUserOperands;
    Use::zap(Storage, Storage + Obj->NumUserOperands,
             /* Delete */ false);
    ::operator delete(Storage);
  }
}

// HLSL Change Starts
void User::operator delete(void *Usr, unsigned NumUserOperands) {
  // Fun fact: during construction Obj->NumUserOperands is overwritten
  Use *Storage = static_cast<Use *>(Usr) - NumUserOperands;
  Use::zap(Storage, Storage + NumUserOperands, /* Delete */ false);
  ::operator delete(Storage);
}
// HLSL Change Ends

//===----------------------------------------------------------------------===//
//                             Operator Class
//===----------------------------------------------------------------------===//

Operator::~Operator() {
  llvm_unreachable("should never destroy an Operator");
}

} // End llvm namespace
