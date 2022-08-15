//===- AliasAnalysis.cpp - Generic Alias Analysis Interface Implementation -==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the generic AliasAnalysis interface which is used as the
// common interface used by all clients and implementations of alias analysis.
//
// This file also implements the default version of the AliasAnalysis interface
// that is to be used when no other implementation is specified.  This does some
// simple tests that detect obvious cases: two different global pointers cannot
// alias, a global cannot alias a malloc, two different mallocs cannot alias,
// etc.
//
// This alias analysis implementation really isn't very good for anything, but
// it is very fast, and makes a nice clean default implementation.  Because it
// handles lots of little corner cases, other, more complex, alias analysis
// implementations may choose to rely on this pass to resolve these simple and
// easy cases.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Type.h"
#include "llvm/Pass.h"
using namespace llvm;

// Register the AliasAnalysis interface, providing a nice name to refer to.
INITIALIZE_ANALYSIS_GROUP(AliasAnalysis, "Alias Analysis", NoAA)
char AliasAnalysis::ID = 0;

//===----------------------------------------------------------------------===//
// Default chaining methods
//===----------------------------------------------------------------------===//

AliasResult AliasAnalysis::alias(const MemoryLocation &LocA,
                                 const MemoryLocation &LocB) {
  assert(AA && "AA didn't call InitializeAliasAnalysis in its run method!");
  return AA->alias(LocA, LocB);
}

bool AliasAnalysis::pointsToConstantMemory(const MemoryLocation &Loc,
                                           bool OrLocal) {
  assert(AA && "AA didn't call InitializeAliasAnalysis in its run method!");
  return AA->pointsToConstantMemory(Loc, OrLocal);
}

AliasAnalysis::ModRefResult
AliasAnalysis::getArgModRefInfo(ImmutableCallSite CS, unsigned ArgIdx) {
  assert(AA && "AA didn't call InitializeAliasAnalysis in its run method!");
  return AA->getArgModRefInfo(CS, ArgIdx);
}

void AliasAnalysis::deleteValue(Value *V) {
  assert(AA && "AA didn't call InitializeAliasAnalysis in its run method!");
  AA->deleteValue(V);
}

void AliasAnalysis::addEscapingUse(Use &U) {
  assert(AA && "AA didn't call InitializeAliasAnalysis in its run method!");
  AA->addEscapingUse(U);
}

AliasAnalysis::ModRefResult
AliasAnalysis::getModRefInfo(Instruction *I, ImmutableCallSite Call) {
  // We may have two calls
  if (auto CS = ImmutableCallSite(I)) {
    // Check if the two calls modify the same memory
    return getModRefInfo(Call, CS);
  } else {
    // Otherwise, check if the call modifies or references the
    // location this memory access defines.  The best we can say
    // is that if the call references what this instruction
    // defines, it must be clobbered by this location.
    const MemoryLocation DefLoc = MemoryLocation::get(I);
    if (getModRefInfo(Call, DefLoc) != AliasAnalysis::NoModRef)
      return AliasAnalysis::ModRef;
  }
  return AliasAnalysis::NoModRef;
}

AliasAnalysis::ModRefResult
AliasAnalysis::getModRefInfo(ImmutableCallSite CS, const MemoryLocation &Loc) {
  assert(AA && "AA didn't call InitializeAliasAnalysis in its run method!");

  ModRefBehavior MRB = getModRefBehavior(CS);
  if (MRB == DoesNotAccessMemory)
    return NoModRef;

  ModRefResult Mask = ModRef;
  if (onlyReadsMemory(MRB))
    Mask = Ref;

  if (onlyAccessesArgPointees(MRB)) {
    bool doesAlias = false;
    ModRefResult AllArgsMask = NoModRef;
    if (doesAccessArgPointees(MRB)) {
      for (ImmutableCallSite::arg_iterator AI = CS.arg_begin(), AE = CS.arg_end();
           AI != AE; ++AI) {
        const Value *Arg = *AI;
        if (!Arg->getType()->isPointerTy())
          continue;
        unsigned ArgIdx = std::distance(CS.arg_begin(), AI);
        MemoryLocation ArgLoc =
            MemoryLocation::getForArgument(CS, ArgIdx, *TLI);
        if (!isNoAlias(ArgLoc, Loc)) {
          ModRefResult ArgMask = getArgModRefInfo(CS, ArgIdx);
          doesAlias = true;
          AllArgsMask = ModRefResult(AllArgsMask | ArgMask);
        }
      }
    }
    if (!doesAlias)
      return NoModRef;
    Mask = ModRefResult(Mask & AllArgsMask);
  }

  // If Loc is a constant memory location, the call definitely could not
  // modify the memory location.
  if ((Mask & Mod) && pointsToConstantMemory(Loc))
    Mask = ModRefResult(Mask & ~Mod);

  // If this is the end of the chain, don't forward.
  if (!AA) return Mask;

  // Otherwise, fall back to the next AA in the chain. But we can merge
  // in any mask we've managed to compute.
  return ModRefResult(AA->getModRefInfo(CS, Loc) & Mask);
}

AliasAnalysis::ModRefResult
AliasAnalysis::getModRefInfo(ImmutableCallSite CS1, ImmutableCallSite CS2) {
  assert(AA && "AA didn't call InitializeAliasAnalysis in its run method!");

  // If CS1 or CS2 are readnone, they don't interact.
  ModRefBehavior CS1B = getModRefBehavior(CS1);
  if (CS1B == DoesNotAccessMemory) return NoModRef;

  ModRefBehavior CS2B = getModRefBehavior(CS2);
  if (CS2B == DoesNotAccessMemory) return NoModRef;

  // If they both only read from memory, there is no dependence.
  if (onlyReadsMemory(CS1B) && onlyReadsMemory(CS2B))
    return NoModRef;

  AliasAnalysis::ModRefResult Mask = ModRef;

  // If CS1 only reads memory, the only dependence on CS2 can be
  // from CS1 reading memory written by CS2.
  if (onlyReadsMemory(CS1B))
    Mask = ModRefResult(Mask & Ref);

  // If CS2 only access memory through arguments, accumulate the mod/ref
  // information from CS1's references to the memory referenced by
  // CS2's arguments.
  if (onlyAccessesArgPointees(CS2B)) {
    AliasAnalysis::ModRefResult R = NoModRef;
    if (doesAccessArgPointees(CS2B)) {
      for (ImmutableCallSite::arg_iterator
           I = CS2.arg_begin(), E = CS2.arg_end(); I != E; ++I) {
        const Value *Arg = *I;
        if (!Arg->getType()->isPointerTy())
          continue;
        unsigned CS2ArgIdx = std::distance(CS2.arg_begin(), I);
        auto CS2ArgLoc = MemoryLocation::getForArgument(CS2, CS2ArgIdx, *TLI);

        // ArgMask indicates what CS2 might do to CS2ArgLoc, and the dependence of
        // CS1 on that location is the inverse.
        ModRefResult ArgMask = getArgModRefInfo(CS2, CS2ArgIdx);
        if (ArgMask == Mod)
          ArgMask = ModRef;
        else if (ArgMask == Ref)
          ArgMask = Mod;

        R = ModRefResult((R | (getModRefInfo(CS1, CS2ArgLoc) & ArgMask)) & Mask);
        if (R == Mask)
          break;
      }
    }
    return R;
  }

  // If CS1 only accesses memory through arguments, check if CS2 references
  // any of the memory referenced by CS1's arguments. If not, return NoModRef.
  if (onlyAccessesArgPointees(CS1B)) {
    AliasAnalysis::ModRefResult R = NoModRef;
    if (doesAccessArgPointees(CS1B)) {
      for (ImmutableCallSite::arg_iterator
           I = CS1.arg_begin(), E = CS1.arg_end(); I != E; ++I) {
        const Value *Arg = *I;
        if (!Arg->getType()->isPointerTy())
          continue;
        unsigned CS1ArgIdx = std::distance(CS1.arg_begin(), I);
        auto CS1ArgLoc = MemoryLocation::getForArgument(CS1, CS1ArgIdx, *TLI);

        // ArgMask indicates what CS1 might do to CS1ArgLoc; if CS1 might Mod
        // CS1ArgLoc, then we care about either a Mod or a Ref by CS2. If CS1
        // might Ref, then we care only about a Mod by CS2.
        ModRefResult ArgMask = getArgModRefInfo(CS1, CS1ArgIdx);
        ModRefResult ArgR = getModRefInfo(CS2, CS1ArgLoc);
        if (((ArgMask & Mod) != NoModRef && (ArgR & ModRef) != NoModRef) ||
            ((ArgMask & Ref) != NoModRef && (ArgR & Mod)    != NoModRef))
          R = ModRefResult((R | ArgMask) & Mask);

        if (R == Mask)
          break;
      }
    }
    return R;
  }

  // If this is the end of the chain, don't forward.
  if (!AA) return Mask;

  // Otherwise, fall back to the next AA in the chain. But we can merge
  // in any mask we've managed to compute.
  return ModRefResult(AA->getModRefInfo(CS1, CS2) & Mask);
}

AliasAnalysis::ModRefBehavior
AliasAnalysis::getModRefBehavior(ImmutableCallSite CS) {
  assert(AA && "AA didn't call InitializeAliasAnalysis in its run method!");

  ModRefBehavior Min = UnknownModRefBehavior;

  // Call back into the alias analysis with the other form of getModRefBehavior
  // to see if it can give a better response.
  if (const Function *F = CS.getCalledFunction())
    Min = getModRefBehavior(F);

  // If this is the end of the chain, don't forward.
  if (!AA) return Min;

  // Otherwise, fall back to the next AA in the chain. But we can merge
  // in any result we've managed to compute.
  return ModRefBehavior(AA->getModRefBehavior(CS) & Min);
}

AliasAnalysis::ModRefBehavior
AliasAnalysis::getModRefBehavior(const Function *F) {
  assert(AA && "AA didn't call InitializeAliasAnalysis in its run method!");
  return AA->getModRefBehavior(F);
}

//===----------------------------------------------------------------------===//
// AliasAnalysis non-virtual helper method implementation
//===----------------------------------------------------------------------===//

AliasAnalysis::ModRefResult
AliasAnalysis::getModRefInfo(const LoadInst *L, const MemoryLocation &Loc) {
  // Be conservative in the face of volatile/atomic.
  if (!L->isUnordered())
    return ModRef;

  // If the load address doesn't alias the given address, it doesn't read
  // or write the specified memory.
  if (Loc.Ptr && !alias(MemoryLocation::get(L), Loc))
    return NoModRef;

  // Otherwise, a load just reads.
  return Ref;
}

AliasAnalysis::ModRefResult
AliasAnalysis::getModRefInfo(const StoreInst *S, const MemoryLocation &Loc) {
  // Be conservative in the face of volatile/atomic.
  if (!S->isUnordered())
    return ModRef;

  if (Loc.Ptr) {
    // If the store address cannot alias the pointer in question, then the
    // specified memory cannot be modified by the store.
    if (!alias(MemoryLocation::get(S), Loc))
      return NoModRef;

    // If the pointer is a pointer to constant memory, then it could not have
    // been modified by this store.
    if (pointsToConstantMemory(Loc))
      return NoModRef;

  }

  // Otherwise, a store just writes.
  return Mod;
}

AliasAnalysis::ModRefResult
AliasAnalysis::getModRefInfo(const VAArgInst *V, const MemoryLocation &Loc) {

  if (Loc.Ptr) {
    // If the va_arg address cannot alias the pointer in question, then the
    // specified memory cannot be accessed by the va_arg.
    if (!alias(MemoryLocation::get(V), Loc))
      return NoModRef;

    // If the pointer is a pointer to constant memory, then it could not have
    // been modified by this va_arg.
    if (pointsToConstantMemory(Loc))
      return NoModRef;
  }

  // Otherwise, a va_arg reads and writes.
  return ModRef;
}

AliasAnalysis::ModRefResult
AliasAnalysis::getModRefInfo(const AtomicCmpXchgInst *CX,
                             const MemoryLocation &Loc) {
  // Acquire/Release cmpxchg has properties that matter for arbitrary addresses.
  if (CX->getSuccessOrdering() > Monotonic)
    return ModRef;

  // If the cmpxchg address does not alias the location, it does not access it.
  if (Loc.Ptr && !alias(MemoryLocation::get(CX), Loc))
    return NoModRef;

  return ModRef;
}

AliasAnalysis::ModRefResult
AliasAnalysis::getModRefInfo(const AtomicRMWInst *RMW,
                             const MemoryLocation &Loc) {
  // Acquire/Release atomicrmw has properties that matter for arbitrary addresses.
  if (RMW->getOrdering() > Monotonic)
    return ModRef;

  // If the atomicrmw address does not alias the location, it does not access it.
  if (Loc.Ptr && !alias(MemoryLocation::get(RMW), Loc))
    return NoModRef;

  return ModRef;
}

// FIXME: this is really just shoring-up a deficiency in alias analysis.
// BasicAA isn't willing to spend linear time determining whether an alloca
// was captured before or after this particular call, while we are. However,
// with a smarter AA in place, this test is just wasting compile time.
AliasAnalysis::ModRefResult AliasAnalysis::callCapturesBefore(
    const Instruction *I, const MemoryLocation &MemLoc, DominatorTree *DT) {
  if (!DT)
    return AliasAnalysis::ModRef;

  const Value *Object = GetUnderlyingObject(MemLoc.Ptr, *DL);
  if (!isIdentifiedObject(Object) || isa<GlobalValue>(Object) ||
      isa<Constant>(Object))
    return AliasAnalysis::ModRef;

  ImmutableCallSite CS(I);
  if (!CS.getInstruction() || CS.getInstruction() == Object)
    return AliasAnalysis::ModRef;

  if (llvm::PointerMayBeCapturedBefore(Object, /* ReturnCaptures */ true,
                                       /* StoreCaptures */ true, I, DT,
                                       /* include Object */ true))
    return AliasAnalysis::ModRef;

  unsigned ArgNo = 0;
  AliasAnalysis::ModRefResult R = AliasAnalysis::NoModRef;
  for (ImmutableCallSite::arg_iterator CI = CS.arg_begin(), CE = CS.arg_end();
       CI != CE; ++CI, ++ArgNo) {
    // Only look at the no-capture or byval pointer arguments.  If this
    // pointer were passed to arguments that were neither of these, then it
    // couldn't be no-capture.
    if (!(*CI)->getType()->isPointerTy() ||
        (!CS.doesNotCapture(ArgNo) && !CS.isByValArgument(ArgNo)))
      continue;

    // If this is a no-capture pointer argument, see if we can tell that it
    // is impossible to alias the pointer we're checking.  If not, we have to
    // assume that the call could touch the pointer, even though it doesn't
    // escape.
    if (isNoAlias(MemoryLocation(*CI), MemoryLocation(Object)))
      continue;
    if (CS.doesNotAccessMemory(ArgNo))
      continue;
    if (CS.onlyReadsMemory(ArgNo)) {
      R = AliasAnalysis::Ref;
      continue;
    }
    return AliasAnalysis::ModRef;
  }
  return R;
}

// AliasAnalysis destructor: DO NOT move this to the header file for
// AliasAnalysis or else clients of the AliasAnalysis class may not depend on
// the AliasAnalysis.o file in the current .a file, causing alias analysis
// support to not be included in the tool correctly!
//
AliasAnalysis::~AliasAnalysis() {}

/// InitializeAliasAnalysis - Subclasses must call this method to initialize the
/// AliasAnalysis interface before any other methods are called.
///
void AliasAnalysis::InitializeAliasAnalysis(Pass *P, const DataLayout *NewDL) {
  DL = NewDL;
  auto *TLIP = P->getAnalysisIfAvailable<TargetLibraryInfoWrapperPass>();
  TLI = TLIP ? &TLIP->getTLI() : nullptr;
  AA = &P->getAnalysis<AliasAnalysis>();
}

// getAnalysisUsage - All alias analysis implementations should invoke this
// directly (using AliasAnalysis::getAnalysisUsage(AU)).
void AliasAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<AliasAnalysis>();         // All AA's chain
}

/// getTypeStoreSize - Return the DataLayout store size for the given type,
/// if known, or a conservative value otherwise.
///
uint64_t AliasAnalysis::getTypeStoreSize(Type *Ty) {
  return DL ? DL->getTypeStoreSize(Ty) : MemoryLocation::UnknownSize;
}

/// canBasicBlockModify - Return true if it is possible for execution of the
/// specified basic block to modify the location Loc.
///
bool AliasAnalysis::canBasicBlockModify(const BasicBlock &BB,
                                        const MemoryLocation &Loc) {
  return canInstructionRangeModRef(BB.front(), BB.back(), Loc, Mod);
}

/// canInstructionRangeModRef - Return true if it is possible for the
/// execution of the specified instructions to mod\ref (according to the
/// mode) the location Loc. The instructions to consider are all
/// of the instructions in the range of [I1,I2] INCLUSIVE.
/// I1 and I2 must be in the same basic block.
bool AliasAnalysis::canInstructionRangeModRef(const Instruction &I1,
                                              const Instruction &I2,
                                              const MemoryLocation &Loc,
                                              const ModRefResult Mode) {
  assert(I1.getParent() == I2.getParent() &&
         "Instructions not in same basic block!");
  BasicBlock::const_iterator I = &I1;
  BasicBlock::const_iterator E = &I2;
  ++E;  // Convert from inclusive to exclusive range.

  for (; I != E; ++I) // Check every instruction in range
    if (getModRefInfo(I, Loc) & Mode)
      return true;
  return false;
}

/// isNoAliasCall - Return true if this pointer is returned by a noalias
/// function.
bool llvm::isNoAliasCall(const Value *V) {
  if (isa<CallInst>(V) || isa<InvokeInst>(V))
    return ImmutableCallSite(cast<Instruction>(V))
      .paramHasAttr(0, Attribute::NoAlias);
  return false;
}

/// isNoAliasArgument - Return true if this is an argument with the noalias
/// attribute.
bool llvm::isNoAliasArgument(const Value *V)
{
  if (const Argument *A = dyn_cast<Argument>(V))
    return A->hasNoAliasAttr();
  return false;
}

/// isIdentifiedObject - Return true if this pointer refers to a distinct and
/// identifiable object.  This returns true for:
///    Global Variables and Functions (but not Global Aliases)
///    Allocas and Mallocs
///    ByVal and NoAlias Arguments
///    NoAlias returns
///
bool llvm::isIdentifiedObject(const Value *V) {
  if (isa<AllocaInst>(V))
    return true;
  if (isa<GlobalValue>(V) && !isa<GlobalAlias>(V))
    return true;
  if (isNoAliasCall(V))
    return true;
  if (const Argument *A = dyn_cast<Argument>(V))
    return A->hasNoAliasAttr() || A->hasByValAttr();
  return false;
}

/// isIdentifiedFunctionLocal - Return true if V is umabigously identified
/// at the function-level. Different IdentifiedFunctionLocals can't alias.
/// Further, an IdentifiedFunctionLocal can not alias with any function
/// arguments other than itself, which is not necessarily true for
/// IdentifiedObjects.
bool llvm::isIdentifiedFunctionLocal(const Value *V)
{
  return isa<AllocaInst>(V) || isNoAliasCall(V) || isNoAliasArgument(V);
}
