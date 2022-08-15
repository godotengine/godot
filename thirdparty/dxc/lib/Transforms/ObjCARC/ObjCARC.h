//===- ObjCARC.h - ObjC ARC Optimization --------------*- C++ -*-----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file defines common definitions/declarations used by the ObjC ARC
/// Optimizer. ARC stands for Automatic Reference Counting and is a system for
/// managing reference counts for objects in Objective C.
///
/// WARNING: This file knows about certain library functions. It recognizes them
/// by name, and hardwires knowledge of their semantics.
///
/// WARNING: This file knows about how certain Objective-C library functions are
/// used. Naive LLVM IR transformations which would otherwise be
/// behavior-preserving may break these assumptions.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TRANSFORMS_OBJCARC_OBJCARC_H
#define LLVM_LIB_TRANSFORMS_OBJCARC_OBJCARC_H

#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/ObjCARC.h"
#include "llvm/Transforms/Utils/Local.h"
#include "ARCInstKind.h"

namespace llvm {
class raw_ostream;
}

namespace llvm {
namespace objcarc {

/// \brief A handy option to enable/disable all ARC Optimizations.
extern bool EnableARCOpts;

/// \brief Test if the given module looks interesting to run ARC optimization
/// on.
static inline bool ModuleHasARC(const Module &M) {
  return
    M.getNamedValue("objc_retain") ||
    M.getNamedValue("objc_release") ||
    M.getNamedValue("objc_autorelease") ||
    M.getNamedValue("objc_retainAutoreleasedReturnValue") ||
    M.getNamedValue("objc_retainBlock") ||
    M.getNamedValue("objc_autoreleaseReturnValue") ||
    M.getNamedValue("objc_autoreleasePoolPush") ||
    M.getNamedValue("objc_loadWeakRetained") ||
    M.getNamedValue("objc_loadWeak") ||
    M.getNamedValue("objc_destroyWeak") ||
    M.getNamedValue("objc_storeWeak") ||
    M.getNamedValue("objc_initWeak") ||
    M.getNamedValue("objc_moveWeak") ||
    M.getNamedValue("objc_copyWeak") ||
    M.getNamedValue("objc_retainedObject") ||
    M.getNamedValue("objc_unretainedObject") ||
    M.getNamedValue("objc_unretainedPointer") ||
    M.getNamedValue("clang.arc.use");
}

/// \brief This is a wrapper around getUnderlyingObject which also knows how to
/// look through objc_retain and objc_autorelease calls, which we know to return
/// their argument verbatim.
static inline const Value *GetUnderlyingObjCPtr(const Value *V,
                                                const DataLayout &DL) {
  for (;;) {
    V = GetUnderlyingObject(V, DL);
    if (!IsForwarding(GetBasicARCInstKind(V)))
      break;
    V = cast<CallInst>(V)->getArgOperand(0);
  }

  return V;
}

/// The RCIdentity root of a value \p V is a dominating value U for which
/// retaining or releasing U is equivalent to retaining or releasing V. In other
/// words, ARC operations on \p V are equivalent to ARC operations on \p U.
///
/// We use this in the ARC optimizer to make it easier to match up ARC
/// operations by always mapping ARC operations to RCIdentityRoots instead of
/// pointers themselves.
///
/// The two ways that we see RCIdentical values in ObjC are via:
///
///   1. PointerCasts
///   2. Forwarding Calls that return their argument verbatim.
///
/// Thus this function strips off pointer casts and forwarding calls. *NOTE*
/// This implies that two RCIdentical values must alias.
static inline const Value *GetRCIdentityRoot(const Value *V) {
  for (;;) {
    V = V->stripPointerCasts();
    if (!IsForwarding(GetBasicARCInstKind(V)))
      break;
    V = cast<CallInst>(V)->getArgOperand(0);
  }
  return V;
}

/// Helper which calls const Value *GetRCIdentityRoot(const Value *V) and just
/// casts away the const of the result. For documentation about what an
/// RCIdentityRoot (and by extension GetRCIdentityRoot is) look at that
/// function.
static inline Value *GetRCIdentityRoot(Value *V) {
  return const_cast<Value *>(GetRCIdentityRoot((const Value *)V));
}

/// \brief Assuming the given instruction is one of the special calls such as
/// objc_retain or objc_release, return the RCIdentity root of the argument of
/// the call.
static inline Value *GetArgRCIdentityRoot(Value *Inst) {
  return GetRCIdentityRoot(cast<CallInst>(Inst)->getArgOperand(0));
}

static inline bool IsNullOrUndef(const Value *V) {
  return isa<ConstantPointerNull>(V) || isa<UndefValue>(V);
}

static inline bool IsNoopInstruction(const Instruction *I) {
  return isa<BitCastInst>(I) ||
    (isa<GetElementPtrInst>(I) &&
     cast<GetElementPtrInst>(I)->hasAllZeroIndices());
}


/// \brief Erase the given instruction.
///
/// Many ObjC calls return their argument verbatim,
/// so if it's such a call and the return value has users, replace them with the
/// argument value.
///
static inline void EraseInstruction(Instruction *CI) {
  Value *OldArg = cast<CallInst>(CI)->getArgOperand(0);

  bool Unused = CI->use_empty();

  if (!Unused) {
    // Replace the return value with the argument.
    assert((IsForwarding(GetBasicARCInstKind(CI)) ||
            (IsNoopOnNull(GetBasicARCInstKind(CI)) &&
             isa<ConstantPointerNull>(OldArg))) &&
           "Can't delete non-forwarding instruction with users!");
    CI->replaceAllUsesWith(OldArg);
  }

  CI->eraseFromParent();

  if (Unused)
    RecursivelyDeleteTriviallyDeadInstructions(OldArg);
}

/// \brief Test whether the given value is possible a retainable object pointer.
static inline bool IsPotentialRetainableObjPtr(const Value *Op) {
  // Pointers to static or stack storage are not valid retainable object
  // pointers.
  if (isa<Constant>(Op) || isa<AllocaInst>(Op))
    return false;
  // Special arguments can not be a valid retainable object pointer.
  if (const Argument *Arg = dyn_cast<Argument>(Op))
    if (Arg->hasByValAttr() ||
        Arg->hasInAllocaAttr() ||
        Arg->hasNestAttr() ||
        Arg->hasStructRetAttr())
      return false;
  // Only consider values with pointer types.
  //
  // It seemes intuitive to exclude function pointer types as well, since
  // functions are never retainable object pointers, however clang occasionally
  // bitcasts retainable object pointers to function-pointer type temporarily.
  PointerType *Ty = dyn_cast<PointerType>(Op->getType());
  if (!Ty)
    return false;
  // Conservatively assume anything else is a potential retainable object
  // pointer.
  return true;
}

static inline bool IsPotentialRetainableObjPtr(const Value *Op,
                                               AliasAnalysis &AA) {
  // First make the rudimentary check.
  if (!IsPotentialRetainableObjPtr(Op))
    return false;

  // Objects in constant memory are not reference-counted.
  if (AA.pointsToConstantMemory(Op))
    return false;

  // Pointers in constant memory are not pointing to reference-counted objects.
  if (const LoadInst *LI = dyn_cast<LoadInst>(Op))
    if (AA.pointsToConstantMemory(LI->getPointerOperand()))
      return false;

  // Otherwise assume the worst.
  return true;
}

/// \brief Helper for GetARCInstKind. Determines what kind of construct CS
/// is.
static inline ARCInstKind GetCallSiteClass(ImmutableCallSite CS) {
  for (ImmutableCallSite::arg_iterator I = CS.arg_begin(), E = CS.arg_end();
       I != E; ++I)
    if (IsPotentialRetainableObjPtr(*I))
      return CS.onlyReadsMemory() ? ARCInstKind::User : ARCInstKind::CallOrUser;

  return CS.onlyReadsMemory() ? ARCInstKind::None : ARCInstKind::Call;
}

/// \brief Return true if this value refers to a distinct and identifiable
/// object.
///
/// This is similar to AliasAnalysis's isIdentifiedObject, except that it uses
/// special knowledge of ObjC conventions.
static inline bool IsObjCIdentifiedObject(const Value *V) {
  // Assume that call results and arguments have their own "provenance".
  // Constants (including GlobalVariables) and Allocas are never
  // reference-counted.
  if (isa<CallInst>(V) || isa<InvokeInst>(V) ||
      isa<Argument>(V) || isa<Constant>(V) ||
      isa<AllocaInst>(V))
    return true;

  if (const LoadInst *LI = dyn_cast<LoadInst>(V)) {
    const Value *Pointer =
      GetRCIdentityRoot(LI->getPointerOperand());
    if (const GlobalVariable *GV = dyn_cast<GlobalVariable>(Pointer)) {
      // A constant pointer can't be pointing to an object on the heap. It may
      // be reference-counted, but it won't be deleted.
      if (GV->isConstant())
        return true;
      StringRef Name = GV->getName();
      // These special variables are known to hold values which are not
      // reference-counted pointers.
      if (Name.startswith("\01l_objc_msgSend_fixup_"))
        return true;

      StringRef Section = GV->getSection();
      if (Section.find("__message_refs") != StringRef::npos ||
          Section.find("__objc_classrefs") != StringRef::npos ||
          Section.find("__objc_superrefs") != StringRef::npos ||
          Section.find("__objc_methname") != StringRef::npos ||
          Section.find("__cstring") != StringRef::npos)
        return true;
    }
  }

  return false;
}

enum class ARCMDKindID {
  ImpreciseRelease,
  CopyOnEscape,
  NoObjCARCExceptions,
};

/// A cache of MDKinds used by various ARC optimizations.
class ARCMDKindCache {
  Module *M;

  /// The Metadata Kind for clang.imprecise_release metadata.
  llvm::Optional<unsigned> ImpreciseReleaseMDKind;

  /// The Metadata Kind for clang.arc.copy_on_escape metadata.
  llvm::Optional<unsigned> CopyOnEscapeMDKind;

  /// The Metadata Kind for clang.arc.no_objc_arc_exceptions metadata.
  llvm::Optional<unsigned> NoObjCARCExceptionsMDKind;

public:
  void init(Module *Mod) {
    M = Mod;
    ImpreciseReleaseMDKind = NoneType::None;
    CopyOnEscapeMDKind = NoneType::None;
    NoObjCARCExceptionsMDKind = NoneType::None;
  }

  unsigned get(ARCMDKindID ID) {
    switch (ID) {
    case ARCMDKindID::ImpreciseRelease:
      if (!ImpreciseReleaseMDKind)
        ImpreciseReleaseMDKind =
            M->getContext().getMDKindID("clang.imprecise_release");
      return *ImpreciseReleaseMDKind;
    case ARCMDKindID::CopyOnEscape:
      if (!CopyOnEscapeMDKind)
        CopyOnEscapeMDKind =
            M->getContext().getMDKindID("clang.arc.copy_on_escape");
      return *CopyOnEscapeMDKind;
    case ARCMDKindID::NoObjCARCExceptions:
      if (!NoObjCARCExceptionsMDKind)
        NoObjCARCExceptionsMDKind =
            M->getContext().getMDKindID("clang.arc.no_objc_arc_exceptions");
      return *NoObjCARCExceptionsMDKind;
    }
    llvm_unreachable("Covered switch isn't covered?!");
  }
};

} // end namespace objcarc
} // end namespace llvm

#endif
