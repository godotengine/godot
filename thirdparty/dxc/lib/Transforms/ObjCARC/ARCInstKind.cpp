//===- ARCInstKind.cpp - ObjC ARC Optimization ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file defines several utility functions used by various ARC
/// optimizations which are IMHO too big to be in a header file.
///
/// WARNING: This file knows about certain library functions. It recognizes them
/// by name, and hardwires knowledge of their semantics.
///
/// WARNING: This file knows about how certain Objective-C library functions are
/// used. Naive LLVM IR transformations which would otherwise be
/// behavior-preserving may break these assumptions.
///
//===----------------------------------------------------------------------===//

#include "ObjCARC.h"
#include "llvm/IR/Intrinsics.h"

using namespace llvm;
using namespace llvm::objcarc;

raw_ostream &llvm::objcarc::operator<<(raw_ostream &OS,
                                       const ARCInstKind Class) {
  switch (Class) {
  case ARCInstKind::Retain:
    return OS << "ARCInstKind::Retain";
  case ARCInstKind::RetainRV:
    return OS << "ARCInstKind::RetainRV";
  case ARCInstKind::RetainBlock:
    return OS << "ARCInstKind::RetainBlock";
  case ARCInstKind::Release:
    return OS << "ARCInstKind::Release";
  case ARCInstKind::Autorelease:
    return OS << "ARCInstKind::Autorelease";
  case ARCInstKind::AutoreleaseRV:
    return OS << "ARCInstKind::AutoreleaseRV";
  case ARCInstKind::AutoreleasepoolPush:
    return OS << "ARCInstKind::AutoreleasepoolPush";
  case ARCInstKind::AutoreleasepoolPop:
    return OS << "ARCInstKind::AutoreleasepoolPop";
  case ARCInstKind::NoopCast:
    return OS << "ARCInstKind::NoopCast";
  case ARCInstKind::FusedRetainAutorelease:
    return OS << "ARCInstKind::FusedRetainAutorelease";
  case ARCInstKind::FusedRetainAutoreleaseRV:
    return OS << "ARCInstKind::FusedRetainAutoreleaseRV";
  case ARCInstKind::LoadWeakRetained:
    return OS << "ARCInstKind::LoadWeakRetained";
  case ARCInstKind::StoreWeak:
    return OS << "ARCInstKind::StoreWeak";
  case ARCInstKind::InitWeak:
    return OS << "ARCInstKind::InitWeak";
  case ARCInstKind::LoadWeak:
    return OS << "ARCInstKind::LoadWeak";
  case ARCInstKind::MoveWeak:
    return OS << "ARCInstKind::MoveWeak";
  case ARCInstKind::CopyWeak:
    return OS << "ARCInstKind::CopyWeak";
  case ARCInstKind::DestroyWeak:
    return OS << "ARCInstKind::DestroyWeak";
  case ARCInstKind::StoreStrong:
    return OS << "ARCInstKind::StoreStrong";
  case ARCInstKind::CallOrUser:
    return OS << "ARCInstKind::CallOrUser";
  case ARCInstKind::Call:
    return OS << "ARCInstKind::Call";
  case ARCInstKind::User:
    return OS << "ARCInstKind::User";
  case ARCInstKind::IntrinsicUser:
    return OS << "ARCInstKind::IntrinsicUser";
  case ARCInstKind::None:
    return OS << "ARCInstKind::None";
  }
  llvm_unreachable("Unknown instruction class!");
}

ARCInstKind llvm::objcarc::GetFunctionClass(const Function *F) {
  Function::const_arg_iterator AI = F->arg_begin(), AE = F->arg_end();

  // No (mandatory) arguments.
  if (AI == AE)
    return StringSwitch<ARCInstKind>(F->getName())
        .Case("objc_autoreleasePoolPush", ARCInstKind::AutoreleasepoolPush)
        .Case("clang.arc.use", ARCInstKind::IntrinsicUser)
        .Default(ARCInstKind::CallOrUser);

  // One argument.
  const Argument *A0 = AI++;
  if (AI == AE)
    // Argument is a pointer.
    if (PointerType *PTy = dyn_cast<PointerType>(A0->getType())) {
      Type *ETy = PTy->getElementType();
      // Argument is i8*.
      if (ETy->isIntegerTy(8))
        return StringSwitch<ARCInstKind>(F->getName())
            .Case("objc_retain", ARCInstKind::Retain)
            .Case("objc_retainAutoreleasedReturnValue", ARCInstKind::RetainRV)
            .Case("objc_retainBlock", ARCInstKind::RetainBlock)
            .Case("objc_release", ARCInstKind::Release)
            .Case("objc_autorelease", ARCInstKind::Autorelease)
            .Case("objc_autoreleaseReturnValue", ARCInstKind::AutoreleaseRV)
            .Case("objc_autoreleasePoolPop", ARCInstKind::AutoreleasepoolPop)
            .Case("objc_retainedObject", ARCInstKind::NoopCast)
            .Case("objc_unretainedObject", ARCInstKind::NoopCast)
            .Case("objc_unretainedPointer", ARCInstKind::NoopCast)
            .Case("objc_retain_autorelease",
                  ARCInstKind::FusedRetainAutorelease)
            .Case("objc_retainAutorelease", ARCInstKind::FusedRetainAutorelease)
            .Case("objc_retainAutoreleaseReturnValue",
                  ARCInstKind::FusedRetainAutoreleaseRV)
            .Case("objc_sync_enter", ARCInstKind::User)
            .Case("objc_sync_exit", ARCInstKind::User)
            .Default(ARCInstKind::CallOrUser);

      // Argument is i8**
      if (PointerType *Pte = dyn_cast<PointerType>(ETy))
        if (Pte->getElementType()->isIntegerTy(8))
          return StringSwitch<ARCInstKind>(F->getName())
              .Case("objc_loadWeakRetained", ARCInstKind::LoadWeakRetained)
              .Case("objc_loadWeak", ARCInstKind::LoadWeak)
              .Case("objc_destroyWeak", ARCInstKind::DestroyWeak)
              .Default(ARCInstKind::CallOrUser);
    }

  // Two arguments, first is i8**.
  const Argument *A1 = AI++;
  if (AI == AE)
    if (PointerType *PTy = dyn_cast<PointerType>(A0->getType()))
      if (PointerType *Pte = dyn_cast<PointerType>(PTy->getElementType()))
        if (Pte->getElementType()->isIntegerTy(8))
          if (PointerType *PTy1 = dyn_cast<PointerType>(A1->getType())) {
            Type *ETy1 = PTy1->getElementType();
            // Second argument is i8*
            if (ETy1->isIntegerTy(8))
              return StringSwitch<ARCInstKind>(F->getName())
                  .Case("objc_storeWeak", ARCInstKind::StoreWeak)
                  .Case("objc_initWeak", ARCInstKind::InitWeak)
                  .Case("objc_storeStrong", ARCInstKind::StoreStrong)
                  .Default(ARCInstKind::CallOrUser);
            // Second argument is i8**.
            if (PointerType *Pte1 = dyn_cast<PointerType>(ETy1))
              if (Pte1->getElementType()->isIntegerTy(8))
                return StringSwitch<ARCInstKind>(F->getName())
                    .Case("objc_moveWeak", ARCInstKind::MoveWeak)
                    .Case("objc_copyWeak", ARCInstKind::CopyWeak)
                    // Ignore annotation calls. This is important to stop the
                    // optimizer from treating annotations as uses which would
                    // make the state of the pointers they are attempting to
                    // elucidate to be incorrect.
                    .Case("llvm.arc.annotation.topdown.bbstart",
                          ARCInstKind::None)
                    .Case("llvm.arc.annotation.topdown.bbend",
                          ARCInstKind::None)
                    .Case("llvm.arc.annotation.bottomup.bbstart",
                          ARCInstKind::None)
                    .Case("llvm.arc.annotation.bottomup.bbend",
                          ARCInstKind::None)
                    .Default(ARCInstKind::CallOrUser);
          }

  // Anything else.
  return ARCInstKind::CallOrUser;
}

// A whitelist of intrinsics that we know do not use objc pointers or decrement
// ref counts.
static bool isInertIntrinsic(unsigned ID) {
  // TODO: Make this into a covered switch.
  switch (ID) {
  case Intrinsic::returnaddress:
  case Intrinsic::frameaddress:
  case Intrinsic::stacksave:
  case Intrinsic::stackrestore:
  case Intrinsic::vastart:
  case Intrinsic::vacopy:
  case Intrinsic::vaend:
  case Intrinsic::objectsize:
  case Intrinsic::prefetch:
  case Intrinsic::stackprotector:
  case Intrinsic::eh_return_i32:
  case Intrinsic::eh_return_i64:
  case Intrinsic::eh_typeid_for:
  case Intrinsic::eh_dwarf_cfa:
  case Intrinsic::eh_sjlj_lsda:
  case Intrinsic::eh_sjlj_functioncontext:
  case Intrinsic::init_trampoline:
  case Intrinsic::adjust_trampoline:
  case Intrinsic::lifetime_start:
  case Intrinsic::lifetime_end:
  case Intrinsic::invariant_start:
  case Intrinsic::invariant_end:
  // Don't let dbg info affect our results.
  case Intrinsic::dbg_declare:
  case Intrinsic::dbg_value:
    // Short cut: Some intrinsics obviously don't use ObjC pointers.
    return true;
  default:
    return false;
  }
}

// A whitelist of intrinsics that we know do not use objc pointers or decrement
// ref counts.
static bool isUseOnlyIntrinsic(unsigned ID) {
  // We are conservative and even though intrinsics are unlikely to touch
  // reference counts, we white list them for safety.
  //
  // TODO: Expand this into a covered switch. There is a lot more here.
  switch (ID) {
  case Intrinsic::memcpy:
  case Intrinsic::memmove:
  case Intrinsic::memset:
    return true;
  default:
    return false;
  }
}

/// \brief Determine what kind of construct V is.
ARCInstKind llvm::objcarc::GetARCInstKind(const Value *V) {
  if (const Instruction *I = dyn_cast<Instruction>(V)) {
    // Any instruction other than bitcast and gep with a pointer operand have a
    // use of an objc pointer. Bitcasts, GEPs, Selects, PHIs transfer a pointer
    // to a subsequent use, rather than using it themselves, in this sense.
    // As a short cut, several other opcodes are known to have no pointer
    // operands of interest. And ret is never followed by a release, so it's
    // not interesting to examine.
    switch (I->getOpcode()) {
    case Instruction::Call: {
      const CallInst *CI = cast<CallInst>(I);
      // See if we have a function that we know something about.
      if (const Function *F = CI->getCalledFunction()) {
        ARCInstKind Class = GetFunctionClass(F);
        if (Class != ARCInstKind::CallOrUser)
          return Class;
        Intrinsic::ID ID = F->getIntrinsicID();
        if (isInertIntrinsic(ID))
          return ARCInstKind::None;
        if (isUseOnlyIntrinsic(ID))
          return ARCInstKind::User;
      }

      // Otherwise, be conservative.
      return GetCallSiteClass(CI);
    }
    case Instruction::Invoke:
      // Otherwise, be conservative.
      return GetCallSiteClass(cast<InvokeInst>(I));
    case Instruction::BitCast:
    case Instruction::GetElementPtr:
    case Instruction::Select:
    case Instruction::PHI:
    case Instruction::Ret:
    case Instruction::Br:
    case Instruction::Switch:
    case Instruction::IndirectBr:
    case Instruction::Alloca:
    case Instruction::VAArg:
    case Instruction::Add:
    case Instruction::FAdd:
    case Instruction::Sub:
    case Instruction::FSub:
    case Instruction::Mul:
    case Instruction::FMul:
    case Instruction::SDiv:
    case Instruction::UDiv:
    case Instruction::FDiv:
    case Instruction::SRem:
    case Instruction::URem:
    case Instruction::FRem:
    case Instruction::Shl:
    case Instruction::LShr:
    case Instruction::AShr:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor:
    case Instruction::SExt:
    case Instruction::ZExt:
    case Instruction::Trunc:
    case Instruction::IntToPtr:
    case Instruction::FCmp:
    case Instruction::FPTrunc:
    case Instruction::FPExt:
    case Instruction::FPToUI:
    case Instruction::FPToSI:
    case Instruction::UIToFP:
    case Instruction::SIToFP:
    case Instruction::InsertElement:
    case Instruction::ExtractElement:
    case Instruction::ShuffleVector:
    case Instruction::ExtractValue:
      break;
    case Instruction::ICmp:
      // Comparing a pointer with null, or any other constant, isn't an
      // interesting use, because we don't care what the pointer points to, or
      // about the values of any other dynamic reference-counted pointers.
      if (IsPotentialRetainableObjPtr(I->getOperand(1)))
        return ARCInstKind::User;
      break;
    default:
      // For anything else, check all the operands.
      // Note that this includes both operands of a Store: while the first
      // operand isn't actually being dereferenced, it is being stored to
      // memory where we can no longer track who might read it and dereference
      // it, so we have to consider it potentially used.
      for (User::const_op_iterator OI = I->op_begin(), OE = I->op_end();
           OI != OE; ++OI)
        if (IsPotentialRetainableObjPtr(*OI))
          return ARCInstKind::User;
    }
  }

  // Otherwise, it's totally inert for ARC purposes.
  return ARCInstKind::None;
}

/// \brief Test if the given class is a kind of user.
bool llvm::objcarc::IsUser(ARCInstKind Class) {
  switch (Class) {
  case ARCInstKind::User:
  case ARCInstKind::CallOrUser:
  case ARCInstKind::IntrinsicUser:
    return true;
  case ARCInstKind::Retain:
  case ARCInstKind::RetainRV:
  case ARCInstKind::RetainBlock:
  case ARCInstKind::Release:
  case ARCInstKind::Autorelease:
  case ARCInstKind::AutoreleaseRV:
  case ARCInstKind::AutoreleasepoolPush:
  case ARCInstKind::AutoreleasepoolPop:
  case ARCInstKind::NoopCast:
  case ARCInstKind::FusedRetainAutorelease:
  case ARCInstKind::FusedRetainAutoreleaseRV:
  case ARCInstKind::LoadWeakRetained:
  case ARCInstKind::StoreWeak:
  case ARCInstKind::InitWeak:
  case ARCInstKind::LoadWeak:
  case ARCInstKind::MoveWeak:
  case ARCInstKind::CopyWeak:
  case ARCInstKind::DestroyWeak:
  case ARCInstKind::StoreStrong:
  case ARCInstKind::Call:
  case ARCInstKind::None:
    return false;
  }
  llvm_unreachable("covered switch isn't covered?");
}

/// \brief Test if the given class is objc_retain or equivalent.
bool llvm::objcarc::IsRetain(ARCInstKind Class) {
  switch (Class) {
  case ARCInstKind::Retain:
  case ARCInstKind::RetainRV:
    return true;
  // I believe we treat retain block as not a retain since it can copy its
  // block.
  case ARCInstKind::RetainBlock:
  case ARCInstKind::Release:
  case ARCInstKind::Autorelease:
  case ARCInstKind::AutoreleaseRV:
  case ARCInstKind::AutoreleasepoolPush:
  case ARCInstKind::AutoreleasepoolPop:
  case ARCInstKind::NoopCast:
  case ARCInstKind::FusedRetainAutorelease:
  case ARCInstKind::FusedRetainAutoreleaseRV:
  case ARCInstKind::LoadWeakRetained:
  case ARCInstKind::StoreWeak:
  case ARCInstKind::InitWeak:
  case ARCInstKind::LoadWeak:
  case ARCInstKind::MoveWeak:
  case ARCInstKind::CopyWeak:
  case ARCInstKind::DestroyWeak:
  case ARCInstKind::StoreStrong:
  case ARCInstKind::IntrinsicUser:
  case ARCInstKind::CallOrUser:
  case ARCInstKind::Call:
  case ARCInstKind::User:
  case ARCInstKind::None:
    return false;
  }
  llvm_unreachable("covered switch isn't covered?");
}

/// \brief Test if the given class is objc_autorelease or equivalent.
bool llvm::objcarc::IsAutorelease(ARCInstKind Class) {
  switch (Class) {
  case ARCInstKind::Autorelease:
  case ARCInstKind::AutoreleaseRV:
    return true;
  case ARCInstKind::Retain:
  case ARCInstKind::RetainRV:
  case ARCInstKind::RetainBlock:
  case ARCInstKind::Release:
  case ARCInstKind::AutoreleasepoolPush:
  case ARCInstKind::AutoreleasepoolPop:
  case ARCInstKind::NoopCast:
  case ARCInstKind::FusedRetainAutorelease:
  case ARCInstKind::FusedRetainAutoreleaseRV:
  case ARCInstKind::LoadWeakRetained:
  case ARCInstKind::StoreWeak:
  case ARCInstKind::InitWeak:
  case ARCInstKind::LoadWeak:
  case ARCInstKind::MoveWeak:
  case ARCInstKind::CopyWeak:
  case ARCInstKind::DestroyWeak:
  case ARCInstKind::StoreStrong:
  case ARCInstKind::IntrinsicUser:
  case ARCInstKind::CallOrUser:
  case ARCInstKind::Call:
  case ARCInstKind::User:
  case ARCInstKind::None:
    return false;
  }
  llvm_unreachable("covered switch isn't covered?");
}

/// \brief Test if the given class represents instructions which return their
/// argument verbatim.
bool llvm::objcarc::IsForwarding(ARCInstKind Class) {
  switch (Class) {
  case ARCInstKind::Retain:
  case ARCInstKind::RetainRV:
  case ARCInstKind::Autorelease:
  case ARCInstKind::AutoreleaseRV:
  case ARCInstKind::NoopCast:
    return true;
  case ARCInstKind::RetainBlock:
  case ARCInstKind::Release:
  case ARCInstKind::AutoreleasepoolPush:
  case ARCInstKind::AutoreleasepoolPop:
  case ARCInstKind::FusedRetainAutorelease:
  case ARCInstKind::FusedRetainAutoreleaseRV:
  case ARCInstKind::LoadWeakRetained:
  case ARCInstKind::StoreWeak:
  case ARCInstKind::InitWeak:
  case ARCInstKind::LoadWeak:
  case ARCInstKind::MoveWeak:
  case ARCInstKind::CopyWeak:
  case ARCInstKind::DestroyWeak:
  case ARCInstKind::StoreStrong:
  case ARCInstKind::IntrinsicUser:
  case ARCInstKind::CallOrUser:
  case ARCInstKind::Call:
  case ARCInstKind::User:
  case ARCInstKind::None:
    return false;
  }
  llvm_unreachable("covered switch isn't covered?");
}

/// \brief Test if the given class represents instructions which do nothing if
/// passed a null pointer.
bool llvm::objcarc::IsNoopOnNull(ARCInstKind Class) {
  switch (Class) {
  case ARCInstKind::Retain:
  case ARCInstKind::RetainRV:
  case ARCInstKind::Release:
  case ARCInstKind::Autorelease:
  case ARCInstKind::AutoreleaseRV:
  case ARCInstKind::RetainBlock:
    return true;
  case ARCInstKind::AutoreleasepoolPush:
  case ARCInstKind::AutoreleasepoolPop:
  case ARCInstKind::FusedRetainAutorelease:
  case ARCInstKind::FusedRetainAutoreleaseRV:
  case ARCInstKind::LoadWeakRetained:
  case ARCInstKind::StoreWeak:
  case ARCInstKind::InitWeak:
  case ARCInstKind::LoadWeak:
  case ARCInstKind::MoveWeak:
  case ARCInstKind::CopyWeak:
  case ARCInstKind::DestroyWeak:
  case ARCInstKind::StoreStrong:
  case ARCInstKind::IntrinsicUser:
  case ARCInstKind::CallOrUser:
  case ARCInstKind::Call:
  case ARCInstKind::User:
  case ARCInstKind::None:
  case ARCInstKind::NoopCast:
    return false;
  }
  llvm_unreachable("covered switch isn't covered?");
}

/// \brief Test if the given class represents instructions which are always safe
/// to mark with the "tail" keyword.
bool llvm::objcarc::IsAlwaysTail(ARCInstKind Class) {
  // ARCInstKind::RetainBlock may be given a stack argument.
  switch (Class) {
  case ARCInstKind::Retain:
  case ARCInstKind::RetainRV:
  case ARCInstKind::AutoreleaseRV:
    return true;
  case ARCInstKind::Release:
  case ARCInstKind::Autorelease:
  case ARCInstKind::RetainBlock:
  case ARCInstKind::AutoreleasepoolPush:
  case ARCInstKind::AutoreleasepoolPop:
  case ARCInstKind::FusedRetainAutorelease:
  case ARCInstKind::FusedRetainAutoreleaseRV:
  case ARCInstKind::LoadWeakRetained:
  case ARCInstKind::StoreWeak:
  case ARCInstKind::InitWeak:
  case ARCInstKind::LoadWeak:
  case ARCInstKind::MoveWeak:
  case ARCInstKind::CopyWeak:
  case ARCInstKind::DestroyWeak:
  case ARCInstKind::StoreStrong:
  case ARCInstKind::IntrinsicUser:
  case ARCInstKind::CallOrUser:
  case ARCInstKind::Call:
  case ARCInstKind::User:
  case ARCInstKind::None:
  case ARCInstKind::NoopCast:
    return false;
  }
  llvm_unreachable("covered switch isn't covered?");
}

/// \brief Test if the given class represents instructions which are never safe
/// to mark with the "tail" keyword.
bool llvm::objcarc::IsNeverTail(ARCInstKind Class) {
  /// It is never safe to tail call objc_autorelease since by tail calling
  /// objc_autorelease: fast autoreleasing causing our object to be potentially
  /// reclaimed from the autorelease pool which violates the semantics of
  /// __autoreleasing types in ARC.
  switch (Class) {
  case ARCInstKind::Autorelease:
    return true;
  case ARCInstKind::Retain:
  case ARCInstKind::RetainRV:
  case ARCInstKind::AutoreleaseRV:
  case ARCInstKind::Release:
  case ARCInstKind::RetainBlock:
  case ARCInstKind::AutoreleasepoolPush:
  case ARCInstKind::AutoreleasepoolPop:
  case ARCInstKind::FusedRetainAutorelease:
  case ARCInstKind::FusedRetainAutoreleaseRV:
  case ARCInstKind::LoadWeakRetained:
  case ARCInstKind::StoreWeak:
  case ARCInstKind::InitWeak:
  case ARCInstKind::LoadWeak:
  case ARCInstKind::MoveWeak:
  case ARCInstKind::CopyWeak:
  case ARCInstKind::DestroyWeak:
  case ARCInstKind::StoreStrong:
  case ARCInstKind::IntrinsicUser:
  case ARCInstKind::CallOrUser:
  case ARCInstKind::Call:
  case ARCInstKind::User:
  case ARCInstKind::None:
  case ARCInstKind::NoopCast:
    return false;
  }
  llvm_unreachable("covered switch isn't covered?");
}

/// \brief Test if the given class represents instructions which are always safe
/// to mark with the nounwind attribute.
bool llvm::objcarc::IsNoThrow(ARCInstKind Class) {
  // objc_retainBlock is not nounwind because it calls user copy constructors
  // which could theoretically throw.
  switch (Class) {
  case ARCInstKind::Retain:
  case ARCInstKind::RetainRV:
  case ARCInstKind::Release:
  case ARCInstKind::Autorelease:
  case ARCInstKind::AutoreleaseRV:
  case ARCInstKind::AutoreleasepoolPush:
  case ARCInstKind::AutoreleasepoolPop:
    return true;
  case ARCInstKind::RetainBlock:
  case ARCInstKind::FusedRetainAutorelease:
  case ARCInstKind::FusedRetainAutoreleaseRV:
  case ARCInstKind::LoadWeakRetained:
  case ARCInstKind::StoreWeak:
  case ARCInstKind::InitWeak:
  case ARCInstKind::LoadWeak:
  case ARCInstKind::MoveWeak:
  case ARCInstKind::CopyWeak:
  case ARCInstKind::DestroyWeak:
  case ARCInstKind::StoreStrong:
  case ARCInstKind::IntrinsicUser:
  case ARCInstKind::CallOrUser:
  case ARCInstKind::Call:
  case ARCInstKind::User:
  case ARCInstKind::None:
  case ARCInstKind::NoopCast:
    return false;
  }
  llvm_unreachable("covered switch isn't covered?");
}

/// Test whether the given instruction can autorelease any pointer or cause an
/// autoreleasepool pop.
///
/// This means that it *could* interrupt the RV optimization.
bool llvm::objcarc::CanInterruptRV(ARCInstKind Class) {
  switch (Class) {
  case ARCInstKind::AutoreleasepoolPop:
  case ARCInstKind::CallOrUser:
  case ARCInstKind::Call:
  case ARCInstKind::Autorelease:
  case ARCInstKind::AutoreleaseRV:
  case ARCInstKind::FusedRetainAutorelease:
  case ARCInstKind::FusedRetainAutoreleaseRV:
    return true;
  case ARCInstKind::Retain:
  case ARCInstKind::RetainRV:
  case ARCInstKind::Release:
  case ARCInstKind::AutoreleasepoolPush:
  case ARCInstKind::RetainBlock:
  case ARCInstKind::LoadWeakRetained:
  case ARCInstKind::StoreWeak:
  case ARCInstKind::InitWeak:
  case ARCInstKind::LoadWeak:
  case ARCInstKind::MoveWeak:
  case ARCInstKind::CopyWeak:
  case ARCInstKind::DestroyWeak:
  case ARCInstKind::StoreStrong:
  case ARCInstKind::IntrinsicUser:
  case ARCInstKind::User:
  case ARCInstKind::None:
  case ARCInstKind::NoopCast:
    return false;
  }
  llvm_unreachable("covered switch isn't covered?");
}

bool llvm::objcarc::CanDecrementRefCount(ARCInstKind Kind) {
  switch (Kind) {
  case ARCInstKind::Retain:
  case ARCInstKind::RetainRV:
  case ARCInstKind::Autorelease:
  case ARCInstKind::AutoreleaseRV:
  case ARCInstKind::NoopCast:
  case ARCInstKind::FusedRetainAutorelease:
  case ARCInstKind::FusedRetainAutoreleaseRV:
  case ARCInstKind::IntrinsicUser:
  case ARCInstKind::User:
  case ARCInstKind::None:
    return false;

  // The cases below are conservative.

  // RetainBlock can result in user defined copy constructors being called
  // implying releases may occur.
  case ARCInstKind::RetainBlock:
  case ARCInstKind::Release:
  case ARCInstKind::AutoreleasepoolPush:
  case ARCInstKind::AutoreleasepoolPop:
  case ARCInstKind::LoadWeakRetained:
  case ARCInstKind::StoreWeak:
  case ARCInstKind::InitWeak:
  case ARCInstKind::LoadWeak:
  case ARCInstKind::MoveWeak:
  case ARCInstKind::CopyWeak:
  case ARCInstKind::DestroyWeak:
  case ARCInstKind::StoreStrong:
  case ARCInstKind::CallOrUser:
  case ARCInstKind::Call:
    return true;
  }

  llvm_unreachable("covered switch isn't covered?");
}
