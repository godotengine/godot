//===- BasicAliasAnalysis.cpp - Stateless Alias Analysis Impl -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the primary stateless implementation of the
// Alias Analysis interface that implements identities (two different
// globals cannot alias, etc), but does no stateful analysis.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Passes.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Operator.h"
#include "llvm/Pass.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
using namespace llvm;

/// Cutoff after which to stop analysing a set of phi nodes potentially involved
/// in a cycle. Because we are analysing 'through' phi nodes we need to be
/// careful with value equivalence. We use reachability to make sure a value
/// cannot be involved in a cycle.
const unsigned MaxNumPhiBBsValueReachabilityCheck = 20;

// The max limit of the search depth in DecomposeGEPExpression() and
// GetUnderlyingObject(), both functions need to use the same search
// depth otherwise the algorithm in aliasGEP will assert.
static const unsigned MaxLookupSearchDepth = 6;

//===----------------------------------------------------------------------===//
// Useful predicates
//===----------------------------------------------------------------------===//

/// isNonEscapingLocalObject - Return true if the pointer is to a function-local
/// object that never escapes from the function.
static bool isNonEscapingLocalObject(const Value *V) {
  // If this is a local allocation, check to see if it escapes.
  if (isa<AllocaInst>(V) || isNoAliasCall(V))
    // Set StoreCaptures to True so that we can assume in our callers that the
    // pointer is not the result of a load instruction. Currently
    // PointerMayBeCaptured doesn't have any special analysis for the
    // StoreCaptures=false case; if it did, our callers could be refined to be
    // more precise.
    return !PointerMayBeCaptured(V, false, /*StoreCaptures=*/true);

  // If this is an argument that corresponds to a byval or noalias argument,
  // then it has not escaped before entering the function.  Check if it escapes
  // inside the function.
  if (const Argument *A = dyn_cast<Argument>(V))
    if (A->hasByValAttr() || A->hasNoAliasAttr())
      // Note even if the argument is marked nocapture we still need to check
      // for copies made inside the function. The nocapture attribute only
      // specifies that there are no copies made that outlive the function.
      return !PointerMayBeCaptured(V, false, /*StoreCaptures=*/true);

  return false;
}

/// isEscapeSource - Return true if the pointer is one which would have
/// been considered an escape by isNonEscapingLocalObject.
static bool isEscapeSource(const Value *V) {
  if (isa<CallInst>(V) || isa<InvokeInst>(V) || isa<Argument>(V))
    return true;

  // The load case works because isNonEscapingLocalObject considers all
  // stores to be escapes (it passes true for the StoreCaptures argument
  // to PointerMayBeCaptured).
  if (isa<LoadInst>(V))
    return true;

  return false;
}

/// getObjectSize - Return the size of the object specified by V, or
/// UnknownSize if unknown.
static uint64_t getObjectSize(const Value *V, const DataLayout &DL,
                              const TargetLibraryInfo &TLI,
                              bool RoundToAlign = false) {
  uint64_t Size;
  if (getObjectSize(V, Size, DL, &TLI, RoundToAlign))
    return Size;
  return MemoryLocation::UnknownSize;
}

/// isObjectSmallerThan - Return true if we can prove that the object specified
/// by V is smaller than Size.
static bool isObjectSmallerThan(const Value *V, uint64_t Size,
                                const DataLayout &DL,
                                const TargetLibraryInfo &TLI) {
  // Note that the meanings of the "object" are slightly different in the
  // following contexts:
  //    c1: llvm::getObjectSize()
  //    c2: llvm.objectsize() intrinsic
  //    c3: isObjectSmallerThan()
  // c1 and c2 share the same meaning; however, the meaning of "object" in c3
  // refers to the "entire object".
  //
  //  Consider this example:
  //     char *p = (char*)malloc(100)
  //     char *q = p+80;
  //
  //  In the context of c1 and c2, the "object" pointed by q refers to the
  // stretch of memory of q[0:19]. So, getObjectSize(q) should return 20.
  //
  //  However, in the context of c3, the "object" refers to the chunk of memory
  // being allocated. So, the "object" has 100 bytes, and q points to the middle
  // the "object". In case q is passed to isObjectSmallerThan() as the 1st
  // parameter, before the llvm::getObjectSize() is called to get the size of
  // entire object, we should:
  //    - either rewind the pointer q to the base-address of the object in
  //      question (in this case rewind to p), or
  //    - just give up. It is up to caller to make sure the pointer is pointing
  //      to the base address the object.
  //
  // We go for 2nd option for simplicity.
  if (!isIdentifiedObject(V))
    return false;

  // This function needs to use the aligned object size because we allow
  // reads a bit past the end given sufficient alignment.
  uint64_t ObjectSize = getObjectSize(V, DL, TLI, /*RoundToAlign*/true);

  return ObjectSize != MemoryLocation::UnknownSize && ObjectSize < Size;
}

/// isObjectSize - Return true if we can prove that the object specified
/// by V has size Size.
static bool isObjectSize(const Value *V, uint64_t Size,
                         const DataLayout &DL, const TargetLibraryInfo &TLI) {
  uint64_t ObjectSize = getObjectSize(V, DL, TLI);
  return ObjectSize != MemoryLocation::UnknownSize && ObjectSize == Size;
}

//===----------------------------------------------------------------------===//
// GetElementPtr Instruction Decomposition and Analysis
//===----------------------------------------------------------------------===//

namespace {
  enum ExtensionKind {
    EK_NotExtended,
    EK_SignExt,
    EK_ZeroExt
  };

  struct VariableGEPIndex {
    const Value *V;
    ExtensionKind Extension;
    int64_t Scale;

    bool operator==(const VariableGEPIndex &Other) const {
      return V == Other.V && Extension == Other.Extension &&
        Scale == Other.Scale;
    }

    bool operator!=(const VariableGEPIndex &Other) const {
      return !operator==(Other);
    }
  };
}


/// GetLinearExpression - Analyze the specified value as a linear expression:
/// "A*V + B", where A and B are constant integers.  Return the scale and offset
/// values as APInts and return V as a Value*, and return whether we looked
/// through any sign or zero extends.  The incoming Value is known to have
/// IntegerType and it may already be sign or zero extended.
///
/// Note that this looks through extends, so the high bits may not be
/// represented in the result.
static Value *GetLinearExpression(Value *V, APInt &Scale, APInt &Offset,
                                  ExtensionKind &Extension,
                                  const DataLayout &DL, unsigned Depth,
                                  AssumptionCache *AC, DominatorTree *DT) {
  assert(V->getType()->isIntegerTy() && "Not an integer value");

  // Limit our recursion depth.
  if (Depth == 6) {
    Scale = 1;
    Offset = 0;
    return V;
  }

  if (BinaryOperator *BOp = dyn_cast<BinaryOperator>(V)) {
    if (ConstantInt *RHSC = dyn_cast<ConstantInt>(BOp->getOperand(1))) {
      switch (BOp->getOpcode()) {
      default: break;
      case Instruction::Or:
        // X|C == X+C if all the bits in C are unset in X.  Otherwise we can't
        // analyze it.
        if (!MaskedValueIsZero(BOp->getOperand(0), RHSC->getValue(), DL, 0, AC,
                               BOp, DT))
          break;
        // FALL THROUGH.
      case Instruction::Add:
        V = GetLinearExpression(BOp->getOperand(0), Scale, Offset, Extension,
                                DL, Depth + 1, AC, DT);
        Offset += RHSC->getValue();
        return V;
      case Instruction::Mul:
        V = GetLinearExpression(BOp->getOperand(0), Scale, Offset, Extension,
                                DL, Depth + 1, AC, DT);
        Offset *= RHSC->getValue();
        Scale *= RHSC->getValue();
        return V;
      case Instruction::Shl:
        V = GetLinearExpression(BOp->getOperand(0), Scale, Offset, Extension,
                                DL, Depth + 1, AC, DT);
        Offset <<= RHSC->getValue().getLimitedValue();
        Scale <<= RHSC->getValue().getLimitedValue();
        return V;
      }
    }
  }

  // Since GEP indices are sign extended anyway, we don't care about the high
  // bits of a sign or zero extended value - just scales and offsets.  The
  // extensions have to be consistent though.
  if ((isa<SExtInst>(V) && Extension != EK_ZeroExt) ||
      (isa<ZExtInst>(V) && Extension != EK_SignExt)) {
    Value *CastOp = cast<CastInst>(V)->getOperand(0);
    unsigned OldWidth = Scale.getBitWidth();
    unsigned SmallWidth = CastOp->getType()->getPrimitiveSizeInBits();
    Scale = Scale.trunc(SmallWidth);
    Offset = Offset.trunc(SmallWidth);
    Extension = isa<SExtInst>(V) ? EK_SignExt : EK_ZeroExt;

    Value *Result = GetLinearExpression(CastOp, Scale, Offset, Extension, DL,
                                        Depth + 1, AC, DT);
    Scale = Scale.zext(OldWidth);
    Offset = Offset.zext(OldWidth);

    return Result;
  }

  Scale = 1;
  Offset = 0;
  return V;
}

/// DecomposeGEPExpression - If V is a symbolic pointer expression, decompose it
/// into a base pointer with a constant offset and a number of scaled symbolic
/// offsets.
///
/// The scaled symbolic offsets (represented by pairs of a Value* and a scale in
/// the VarIndices vector) are Value*'s that are known to be scaled by the
/// specified amount, but which may have other unrepresented high bits. As such,
/// the gep cannot necessarily be reconstructed from its decomposed form.
///
/// When DataLayout is around, this function is capable of analyzing everything
/// that GetUnderlyingObject can look through. To be able to do that
/// GetUnderlyingObject and DecomposeGEPExpression must use the same search
/// depth (MaxLookupSearchDepth).
/// When DataLayout not is around, it just looks through pointer casts.
///
static const Value *
DecomposeGEPExpression(const Value *V, int64_t &BaseOffs,
                       SmallVectorImpl<VariableGEPIndex> &VarIndices,
                       bool &MaxLookupReached, const DataLayout &DL,
                       AssumptionCache *AC, DominatorTree *DT) {
  // Limit recursion depth to limit compile time in crazy cases.
  unsigned MaxLookup = MaxLookupSearchDepth;
  MaxLookupReached = false;

  BaseOffs = 0;
  do {
    // See if this is a bitcast or GEP.
    const Operator *Op = dyn_cast<Operator>(V);
    if (!Op) {
      // The only non-operator case we can handle are GlobalAliases.
      if (const GlobalAlias *GA = dyn_cast<GlobalAlias>(V)) {
        if (!GA->mayBeOverridden()) {
          V = GA->getAliasee();
          continue;
        }
      }
      return V;
    }

    if (Op->getOpcode() == Instruction::BitCast ||
        Op->getOpcode() == Instruction::AddrSpaceCast) {
      V = Op->getOperand(0);
      continue;
    }

    const GEPOperator *GEPOp = dyn_cast<GEPOperator>(Op);
    if (!GEPOp) {
      // If it's not a GEP, hand it off to SimplifyInstruction to see if it
      // can come up with something. This matches what GetUnderlyingObject does.
      if (const Instruction *I = dyn_cast<Instruction>(V))
        // TODO: Get a DominatorTree and AssumptionCache and use them here
        // (these are both now available in this function, but this should be
        // updated when GetUnderlyingObject is updated). TLI should be
        // provided also.
        if (const Value *Simplified =
              SimplifyInstruction(const_cast<Instruction *>(I), DL)) {
          V = Simplified;
          continue;
        }

      return V;
    }

    // Don't attempt to analyze GEPs over unsized objects.
    if (!GEPOp->getOperand(0)->getType()->getPointerElementType()->isSized())
      return V;

    unsigned AS = GEPOp->getPointerAddressSpace();
    // Walk the indices of the GEP, accumulating them into BaseOff/VarIndices.
    gep_type_iterator GTI = gep_type_begin(GEPOp);
    for (User::const_op_iterator I = GEPOp->op_begin()+1,
         E = GEPOp->op_end(); I != E; ++I) {
      Value *Index = *I;
      // Compute the (potentially symbolic) offset in bytes for this index.
      if (StructType *STy = dyn_cast<StructType>(*GTI++)) {
        // For a struct, add the member offset.
        unsigned FieldNo = cast<ConstantInt>(Index)->getZExtValue();
        if (FieldNo == 0) continue;

        BaseOffs += DL.getStructLayout(STy)->getElementOffset(FieldNo);
        continue;
      }

      // For an array/pointer, add the element offset, explicitly scaled.
      if (ConstantInt *CIdx = dyn_cast<ConstantInt>(Index)) {
        if (CIdx->isZero()) continue;
        BaseOffs += DL.getTypeAllocSize(*GTI) * CIdx->getSExtValue();
        continue;
      }

      uint64_t Scale = DL.getTypeAllocSize(*GTI);
      ExtensionKind Extension = EK_NotExtended;

      // If the integer type is smaller than the pointer size, it is implicitly
      // sign extended to pointer size.
      unsigned Width = Index->getType()->getIntegerBitWidth();
      if (DL.getPointerSizeInBits(AS) > Width)
        Extension = EK_SignExt;

      // Use GetLinearExpression to decompose the index into a C1*V+C2 form.
      APInt IndexScale(Width, 0), IndexOffset(Width, 0);
      Index = GetLinearExpression(Index, IndexScale, IndexOffset, Extension, DL,
                                  0, AC, DT);

      // The GEP index scale ("Scale") scales C1*V+C2, yielding (C1*V+C2)*Scale.
      // This gives us an aggregate computation of (C1*Scale)*V + C2*Scale.
      BaseOffs += IndexOffset.getSExtValue()*Scale;
      Scale *= IndexScale.getSExtValue();

      // If we already had an occurrence of this index variable, merge this
      // scale into it.  For example, we want to handle:
      //   A[x][x] -> x*16 + x*4 -> x*20
      // This also ensures that 'x' only appears in the index list once.
      for (unsigned i = 0, e = VarIndices.size(); i != e; ++i) {
        if (VarIndices[i].V == Index &&
            VarIndices[i].Extension == Extension) {
          Scale += VarIndices[i].Scale;
          VarIndices.erase(VarIndices.begin()+i);
          break;
        }
      }

      // Make sure that we have a scale that makes sense for this target's
      // pointer size.
      if (unsigned ShiftBits = 64 - DL.getPointerSizeInBits(AS)) {
        Scale <<= ShiftBits;
        Scale = (int64_t)Scale >> ShiftBits;
      }

      if (Scale) {
        VariableGEPIndex Entry = {Index, Extension,
                                  static_cast<int64_t>(Scale)};
        VarIndices.push_back(Entry);
      }
    }

    // Analyze the base pointer next.
    V = GEPOp->getOperand(0);
  } while (--MaxLookup);

  // If the chain of expressions is too deep, just return early.
  MaxLookupReached = true;
  return V;
}

//===----------------------------------------------------------------------===//
// BasicAliasAnalysis Pass
//===----------------------------------------------------------------------===//

#ifndef NDEBUG
static const Function *getParent(const Value *V) {
  if (const Instruction *inst = dyn_cast<Instruction>(V))
    return inst->getParent()->getParent();

  if (const Argument *arg = dyn_cast<Argument>(V))
    return arg->getParent();

  return nullptr;
}

static bool notDifferentParent(const Value *O1, const Value *O2) {

  const Function *F1 = getParent(O1);
  const Function *F2 = getParent(O2);

  return !F1 || !F2 || F1 == F2;
}
#endif

namespace {
  /// BasicAliasAnalysis - This is the primary alias analysis implementation.
  struct BasicAliasAnalysis : public ImmutablePass, public AliasAnalysis {
    static char ID; // Class identification, replacement for typeinfo
    BasicAliasAnalysis() : ImmutablePass(ID) {
      initializeBasicAliasAnalysisPass(*PassRegistry::getPassRegistry());
    }

    bool doInitialization(Module &M) override;

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.addRequired<AliasAnalysis>();
      AU.addRequired<AssumptionCacheTracker>();
      AU.addRequired<TargetLibraryInfoWrapperPass>();
    }

    AliasResult alias(const MemoryLocation &LocA,
                      const MemoryLocation &LocB) override {
      assert(AliasCache.empty() && "AliasCache must be cleared after use!");
      assert(notDifferentParent(LocA.Ptr, LocB.Ptr) &&
             "BasicAliasAnalysis doesn't support interprocedural queries.");
      AliasResult Alias = aliasCheck(LocA.Ptr, LocA.Size, LocA.AATags,
                                     LocB.Ptr, LocB.Size, LocB.AATags);
      // AliasCache rarely has more than 1 or 2 elements, always use
      // shrink_and_clear so it quickly returns to the inline capacity of the
      // SmallDenseMap if it ever grows larger.
      // FIXME: This should really be shrink_to_inline_capacity_and_clear().
      AliasCache.shrink_and_clear();
      VisitedPhiBBs.clear();
      return Alias;
    }

    ModRefResult getModRefInfo(ImmutableCallSite CS,
                               const MemoryLocation &Loc) override;

    ModRefResult getModRefInfo(ImmutableCallSite CS1,
                               ImmutableCallSite CS2) override;

    /// pointsToConstantMemory - Chase pointers until we find a (constant
    /// global) or not.
    bool pointsToConstantMemory(const MemoryLocation &Loc,
                                bool OrLocal) override;

    /// Get the location associated with a pointer argument of a callsite.
    ModRefResult getArgModRefInfo(ImmutableCallSite CS,
                                  unsigned ArgIdx) override;

    /// getModRefBehavior - Return the behavior when calling the given
    /// call site.
    ModRefBehavior getModRefBehavior(ImmutableCallSite CS) override;

    /// getModRefBehavior - Return the behavior when calling the given function.
    /// For use when the call site is not known.
    ModRefBehavior getModRefBehavior(const Function *F) override;

    /// getAdjustedAnalysisPointer - This method is used when a pass implements
    /// an analysis interface through multiple inheritance.  If needed, it
    /// should override this to adjust the this pointer as needed for the
    /// specified pass info.
    void *getAdjustedAnalysisPointer(const void *ID) override {
      if (ID == &AliasAnalysis::ID)
        return (AliasAnalysis*)this;
      return this;
    }

  private:
    // AliasCache - Track alias queries to guard against recursion.
    typedef std::pair<MemoryLocation, MemoryLocation> LocPair;
    typedef SmallDenseMap<LocPair, AliasResult, 8> AliasCacheTy;
    AliasCacheTy AliasCache;

    /// \brief Track phi nodes we have visited. When interpret "Value" pointer
    /// equality as value equality we need to make sure that the "Value" is not
    /// part of a cycle. Otherwise, two uses could come from different
    /// "iterations" of a cycle and see different values for the same "Value"
    /// pointer.
    /// The following example shows the problem:
    ///   %p = phi(%alloca1, %addr2)
    ///   %l = load %ptr
    ///   %addr1 = gep, %alloca2, 0, %l
    ///   %addr2 = gep  %alloca2, 0, (%l + 1)
    ///      alias(%p, %addr1) -> MayAlias !
    ///   store %l, ...
    SmallPtrSet<const BasicBlock*, 8> VisitedPhiBBs;

    // Visited - Track instructions visited by pointsToConstantMemory.
    SmallPtrSet<const Value*, 16> Visited;

    /// \brief Check whether two Values can be considered equivalent.
    ///
    /// In addition to pointer equivalence of \p V1 and \p V2 this checks
    /// whether they can not be part of a cycle in the value graph by looking at
    /// all visited phi nodes an making sure that the phis cannot reach the
    /// value. We have to do this because we are looking through phi nodes (That
    /// is we say noalias(V, phi(VA, VB)) if noalias(V, VA) and noalias(V, VB).
    bool isValueEqualInPotentialCycles(const Value *V1, const Value *V2);

    /// \brief Dest and Src are the variable indices from two decomposed
    /// GetElementPtr instructions GEP1 and GEP2 which have common base
    /// pointers.  Subtract the GEP2 indices from GEP1 to find the symbolic
    /// difference between the two pointers.
    void GetIndexDifference(SmallVectorImpl<VariableGEPIndex> &Dest,
                            const SmallVectorImpl<VariableGEPIndex> &Src);

    // aliasGEP - Provide a bunch of ad-hoc rules to disambiguate a GEP
    // instruction against another.
    AliasResult aliasGEP(const GEPOperator *V1, uint64_t V1Size,
                         const AAMDNodes &V1AAInfo,
                         const Value *V2, uint64_t V2Size,
                         const AAMDNodes &V2AAInfo,
                         const Value *UnderlyingV1, const Value *UnderlyingV2);

    // aliasPHI - Provide a bunch of ad-hoc rules to disambiguate a PHI
    // instruction against another.
    AliasResult aliasPHI(const PHINode *PN, uint64_t PNSize,
                         const AAMDNodes &PNAAInfo,
                         const Value *V2, uint64_t V2Size,
                         const AAMDNodes &V2AAInfo);

    /// aliasSelect - Disambiguate a Select instruction against another value.
    AliasResult aliasSelect(const SelectInst *SI, uint64_t SISize,
                            const AAMDNodes &SIAAInfo,
                            const Value *V2, uint64_t V2Size,
                            const AAMDNodes &V2AAInfo);

    AliasResult aliasCheck(const Value *V1, uint64_t V1Size,
                           AAMDNodes V1AATag,
                           const Value *V2, uint64_t V2Size,
                           AAMDNodes V2AATag);
  };
}  // End of anonymous namespace

// Register this pass...
char BasicAliasAnalysis::ID = 0;
INITIALIZE_AG_PASS_BEGIN(BasicAliasAnalysis, AliasAnalysis, "basicaa",
                   "Basic Alias Analysis (stateless AA impl)",
                   false, true, false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_AG_PASS_END(BasicAliasAnalysis, AliasAnalysis, "basicaa",
                   "Basic Alias Analysis (stateless AA impl)",
                   false, true, false)


ImmutablePass *llvm::createBasicAliasAnalysisPass() {
  return new BasicAliasAnalysis();
}

/// pointsToConstantMemory - Returns whether the given pointer value
/// points to memory that is local to the function, with global constants being
/// considered local to all functions.
bool BasicAliasAnalysis::pointsToConstantMemory(const MemoryLocation &Loc,
                                                bool OrLocal) {
  assert(Visited.empty() && "Visited must be cleared after use!");

  unsigned MaxLookup = 8;
  SmallVector<const Value *, 16> Worklist;
  Worklist.push_back(Loc.Ptr);
  do {
    const Value *V = GetUnderlyingObject(Worklist.pop_back_val(), *DL);
    if (!Visited.insert(V).second) {
      Visited.clear();
      return AliasAnalysis::pointsToConstantMemory(Loc, OrLocal);
    }

    // An alloca instruction defines local memory.
    if (OrLocal && isa<AllocaInst>(V))
      continue;

    // A global constant counts as local memory for our purposes.
    if (const GlobalVariable *GV = dyn_cast<GlobalVariable>(V)) {
      // Note: this doesn't require GV to be "ODR" because it isn't legal for a
      // global to be marked constant in some modules and non-constant in
      // others.  GV may even be a declaration, not a definition.
      if (!GV->isConstant()) {
        Visited.clear();
        return AliasAnalysis::pointsToConstantMemory(Loc, OrLocal);
      }
      continue;
    }

    // If both select values point to local memory, then so does the select.
    if (const SelectInst *SI = dyn_cast<SelectInst>(V)) {
      Worklist.push_back(SI->getTrueValue());
      Worklist.push_back(SI->getFalseValue());
      continue;
    }

    // If all values incoming to a phi node point to local memory, then so does
    // the phi.
    if (const PHINode *PN = dyn_cast<PHINode>(V)) {
      // Don't bother inspecting phi nodes with many operands.
      if (PN->getNumIncomingValues() > MaxLookup) {
        Visited.clear();
        return AliasAnalysis::pointsToConstantMemory(Loc, OrLocal);
      }
      for (Value *IncValue : PN->incoming_values())
        Worklist.push_back(IncValue);
      continue;
    }

    // Otherwise be conservative.
    Visited.clear();
    return AliasAnalysis::pointsToConstantMemory(Loc, OrLocal);

  } while (!Worklist.empty() && --MaxLookup);

  Visited.clear();
  return Worklist.empty();
}

// FIXME: This code is duplicated with MemoryLocation and should be hoisted to
// some common utility location.
static bool isMemsetPattern16(const Function *MS,
                              const TargetLibraryInfo &TLI) {
  if (TLI.has(LibFunc::memset_pattern16) &&
      MS->getName() == "memset_pattern16") {
    FunctionType *MemsetType = MS->getFunctionType();
    if (!MemsetType->isVarArg() && MemsetType->getNumParams() == 3 &&
        isa<PointerType>(MemsetType->getParamType(0)) &&
        isa<PointerType>(MemsetType->getParamType(1)) &&
        isa<IntegerType>(MemsetType->getParamType(2)))
      return true;
  }

  return false;
}

/// getModRefBehavior - Return the behavior when calling the given call site.
AliasAnalysis::ModRefBehavior
BasicAliasAnalysis::getModRefBehavior(ImmutableCallSite CS) {
  if (CS.doesNotAccessMemory())
    // Can't do better than this.
    return DoesNotAccessMemory;

  ModRefBehavior Min = UnknownModRefBehavior;

  // If the callsite knows it only reads memory, don't return worse
  // than that.
  if (CS.onlyReadsMemory())
    Min = OnlyReadsMemory;

  if (CS.onlyAccessesArgMemory())
    Min = ModRefBehavior(Min & OnlyAccessesArgumentPointees);

  // The AliasAnalysis base class has some smarts, lets use them.
  return ModRefBehavior(AliasAnalysis::getModRefBehavior(CS) & Min);
}

/// getModRefBehavior - Return the behavior when calling the given function.
/// For use when the call site is not known.
AliasAnalysis::ModRefBehavior
BasicAliasAnalysis::getModRefBehavior(const Function *F) {
  // If the function declares it doesn't access memory, we can't do better.
  if (F->doesNotAccessMemory())
    return DoesNotAccessMemory;

  // For intrinsics, we can check the table.
  if (Intrinsic::ID iid = F->getIntrinsicID()) {
#define GET_INTRINSIC_MODREF_BEHAVIOR
#include "llvm/IR/Intrinsics.gen"
#undef GET_INTRINSIC_MODREF_BEHAVIOR
  }

  ModRefBehavior Min = UnknownModRefBehavior;

  // If the function declares it only reads memory, go with that.
  if (F->onlyReadsMemory())
    Min = OnlyReadsMemory;

  if (F->onlyAccessesArgMemory())
    Min = ModRefBehavior(Min & OnlyAccessesArgumentPointees);

  const TargetLibraryInfo &TLI =
      getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();
  if (isMemsetPattern16(F, TLI))
    Min = OnlyAccessesArgumentPointees;

  // Otherwise be conservative.
  return ModRefBehavior(AliasAnalysis::getModRefBehavior(F) & Min);
}

AliasAnalysis::ModRefResult
BasicAliasAnalysis::getArgModRefInfo(ImmutableCallSite CS, unsigned ArgIdx) {
  if (const IntrinsicInst *II = dyn_cast<IntrinsicInst>(CS.getInstruction()))
    switch (II->getIntrinsicID()) {
    default:
      break;
    case Intrinsic::memset:
    case Intrinsic::memcpy:
    case Intrinsic::memmove:
      assert((ArgIdx == 0 || ArgIdx == 1) &&
             "Invalid argument index for memory intrinsic");
      return ArgIdx ? Ref : Mod;
    }

  // We can bound the aliasing properties of memset_pattern16 just as we can
  // for memcpy/memset.  This is particularly important because the
  // LoopIdiomRecognizer likes to turn loops into calls to memset_pattern16
  // whenever possible.
  if (CS.getCalledFunction() &&
      isMemsetPattern16(CS.getCalledFunction(), *TLI)) {
    assert((ArgIdx == 0 || ArgIdx == 1) &&
           "Invalid argument index for memset_pattern16");
    return ArgIdx ? Ref : Mod;
  }
  // FIXME: Handle memset_pattern4 and memset_pattern8 also.

  return AliasAnalysis::getArgModRefInfo(CS, ArgIdx);
}

static bool isAssumeIntrinsic(ImmutableCallSite CS) {
  const IntrinsicInst *II = dyn_cast<IntrinsicInst>(CS.getInstruction());
  if (II && II->getIntrinsicID() == Intrinsic::assume)
    return true;

  return false;
}

bool BasicAliasAnalysis::doInitialization(Module &M) {
  InitializeAliasAnalysis(this, &M.getDataLayout());
  return true;
}

/// getModRefInfo - Check to see if the specified callsite can clobber the
/// specified memory object.  Since we only look at local properties of this
/// function, we really can't say much about this query.  We do, however, use
/// simple "address taken" analysis on local objects.
AliasAnalysis::ModRefResult
BasicAliasAnalysis::getModRefInfo(ImmutableCallSite CS,
                                  const MemoryLocation &Loc) {
  assert(notDifferentParent(CS.getInstruction(), Loc.Ptr) &&
         "AliasAnalysis query involving multiple functions!");

  const Value *Object = GetUnderlyingObject(Loc.Ptr, *DL);

  // If this is a tail call and Loc.Ptr points to a stack location, we know that
  // the tail call cannot access or modify the local stack.
  // We cannot exclude byval arguments here; these belong to the caller of
  // the current function not to the current function, and a tail callee
  // may reference them.
  if (isa<AllocaInst>(Object))
    if (const CallInst *CI = dyn_cast<CallInst>(CS.getInstruction()))
      if (CI->isTailCall())
        return NoModRef;

  // If the pointer is to a locally allocated object that does not escape,
  // then the call can not mod/ref the pointer unless the call takes the pointer
  // as an argument, and itself doesn't capture it.
  if (!isa<Constant>(Object) && CS.getInstruction() != Object &&
      isNonEscapingLocalObject(Object)) {
    bool PassedAsArg = false;
    unsigned ArgNo = 0;
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
      if (!isNoAlias(MemoryLocation(*CI), MemoryLocation(Object))) {
        PassedAsArg = true;
        break;
      }
    }

    if (!PassedAsArg)
      return NoModRef;
  }

  // While the assume intrinsic is marked as arbitrarily writing so that
  // proper control dependencies will be maintained, it never aliases any
  // particular memory location.
  if (isAssumeIntrinsic(CS))
    return NoModRef;

  // The AliasAnalysis base class has some smarts, lets use them.
  return AliasAnalysis::getModRefInfo(CS, Loc);
}

AliasAnalysis::ModRefResult
BasicAliasAnalysis::getModRefInfo(ImmutableCallSite CS1,
                                  ImmutableCallSite CS2) {
  // While the assume intrinsic is marked as arbitrarily writing so that
  // proper control dependencies will be maintained, it never aliases any
  // particular memory location.
  if (isAssumeIntrinsic(CS1) || isAssumeIntrinsic(CS2))
    return NoModRef;

  // The AliasAnalysis base class has some smarts, lets use them.
  return AliasAnalysis::getModRefInfo(CS1, CS2);
}

/// \brief Provide ad-hoc rules to disambiguate accesses through two GEP
/// operators, both having the exact same pointer operand.
static AliasResult aliasSameBasePointerGEPs(const GEPOperator *GEP1,
                                            uint64_t V1Size,
                                            const GEPOperator *GEP2,
                                            uint64_t V2Size,
                                            const DataLayout &DL) {

  assert(GEP1->getPointerOperand() == GEP2->getPointerOperand() &&
         "Expected GEPs with the same pointer operand");

  // Try to determine whether GEP1 and GEP2 index through arrays, into structs,
  // such that the struct field accesses provably cannot alias.
  // We also need at least two indices (the pointer, and the struct field).
  if (GEP1->getNumIndices() != GEP2->getNumIndices() ||
      GEP1->getNumIndices() < 2)
    return MayAlias;

  // If we don't know the size of the accesses through both GEPs, we can't
  // determine whether the struct fields accessed can't alias.
  if (V1Size == MemoryLocation::UnknownSize ||
      V2Size == MemoryLocation::UnknownSize)
    return MayAlias;

  ConstantInt *C1 =
      dyn_cast<ConstantInt>(GEP1->getOperand(GEP1->getNumOperands() - 1));
  ConstantInt *C2 =
      dyn_cast<ConstantInt>(GEP2->getOperand(GEP2->getNumOperands() - 1));

  // If the last (struct) indices aren't constants, we can't say anything.
  // If they're identical, the other indices might be also be dynamically
  // equal, so the GEPs can alias.
  if (!C1 || !C2 || C1 == C2)
    return MayAlias;

  // Find the last-indexed type of the GEP, i.e., the type you'd get if
  // you stripped the last index.
  // On the way, look at each indexed type.  If there's something other
  // than an array, different indices can lead to different final types.
  SmallVector<Value *, 8> IntermediateIndices;

  // Insert the first index; we don't need to check the type indexed
  // through it as it only drops the pointer indirection.
  assert(GEP1->getNumIndices() > 1 && "Not enough GEP indices to examine");
  IntermediateIndices.push_back(GEP1->getOperand(1));

  // Insert all the remaining indices but the last one.
  // Also, check that they all index through arrays.
  for (unsigned i = 1, e = GEP1->getNumIndices() - 1; i != e; ++i) {
    if (!isa<ArrayType>(GetElementPtrInst::getIndexedType(
            GEP1->getSourceElementType(), IntermediateIndices)))
      return MayAlias;
    IntermediateIndices.push_back(GEP1->getOperand(i + 1));
  }

  StructType *LastIndexedStruct =
      dyn_cast<StructType>(GetElementPtrInst::getIndexedType(
          GEP1->getSourceElementType(), IntermediateIndices));

  if (!LastIndexedStruct)
    return MayAlias;

  // We know that:
  // - both GEPs begin indexing from the exact same pointer;
  // - the last indices in both GEPs are constants, indexing into a struct;
  // - said indices are different, hence, the pointed-to fields are different;
  // - both GEPs only index through arrays prior to that.
  //
  // This lets us determine that the struct that GEP1 indexes into and the
  // struct that GEP2 indexes into must either precisely overlap or be
  // completely disjoint.  Because they cannot partially overlap, indexing into
  // different non-overlapping fields of the struct will never alias.

  // Therefore, the only remaining thing needed to show that both GEPs can't
  // alias is that the fields are not overlapping.
  const StructLayout *SL = DL.getStructLayout(LastIndexedStruct);
  const uint64_t StructSize = SL->getSizeInBytes();
  const uint64_t V1Off = SL->getElementOffset(C1->getZExtValue());
  const uint64_t V2Off = SL->getElementOffset(C2->getZExtValue());

  auto EltsDontOverlap = [StructSize](uint64_t V1Off, uint64_t V1Size,
                                      uint64_t V2Off, uint64_t V2Size) {
    return V1Off < V2Off && V1Off + V1Size <= V2Off &&
           ((V2Off + V2Size <= StructSize) ||
            (V2Off + V2Size - StructSize <= V1Off));
  };

  if (EltsDontOverlap(V1Off, V1Size, V2Off, V2Size) ||
      EltsDontOverlap(V2Off, V2Size, V1Off, V1Size))
    return NoAlias;

  return MayAlias;
}

/// aliasGEP - Provide a bunch of ad-hoc rules to disambiguate a GEP instruction
/// against another pointer.  We know that V1 is a GEP, but we don't know
/// anything about V2.  UnderlyingV1 is GetUnderlyingObject(GEP1, DL),
/// UnderlyingV2 is the same for V2.
///
AliasResult BasicAliasAnalysis::aliasGEP(
    const GEPOperator *GEP1, uint64_t V1Size, const AAMDNodes &V1AAInfo,
    const Value *V2, uint64_t V2Size, const AAMDNodes &V2AAInfo,
    const Value *UnderlyingV1, const Value *UnderlyingV2) {
  int64_t GEP1BaseOffset;
  bool GEP1MaxLookupReached;
  SmallVector<VariableGEPIndex, 4> GEP1VariableIndices;

  // We have to get two AssumptionCaches here because GEP1 and V2 may be from
  // different functions.
  // FIXME: This really doesn't make any sense. We get a dominator tree below
  // that can only refer to a single function. But this function (aliasGEP) is
  // a method on an immutable pass that can be called when there *isn't*
  // a single function. The old pass management layer makes this "work", but
  // this isn't really a clean solution.
  AssumptionCacheTracker &ACT = getAnalysis<AssumptionCacheTracker>();
  AssumptionCache *AC1 = nullptr, *AC2 = nullptr;
  if (auto *GEP1I = dyn_cast<Instruction>(GEP1))
    AC1 = &ACT.getAssumptionCache(
        const_cast<Function &>(*GEP1I->getParent()->getParent()));
  if (auto *I2 = dyn_cast<Instruction>(V2))
    AC2 = &ACT.getAssumptionCache(
        const_cast<Function &>(*I2->getParent()->getParent()));

  DominatorTreeWrapperPass *DTWP =
      getAnalysisIfAvailable<DominatorTreeWrapperPass>();
  DominatorTree *DT = DTWP ? &DTWP->getDomTree() : nullptr;

  // If we have two gep instructions with must-alias or not-alias'ing base
  // pointers, figure out if the indexes to the GEP tell us anything about the
  // derived pointer.
  if (const GEPOperator *GEP2 = dyn_cast<GEPOperator>(V2)) {
    // Do the base pointers alias?
    AliasResult BaseAlias =
        aliasCheck(UnderlyingV1, MemoryLocation::UnknownSize, AAMDNodes(),
                   UnderlyingV2, MemoryLocation::UnknownSize, AAMDNodes());

    // Check for geps of non-aliasing underlying pointers where the offsets are
    // identical.
    if ((BaseAlias == MayAlias) && V1Size == V2Size) {
      // Do the base pointers alias assuming type and size.
      AliasResult PreciseBaseAlias = aliasCheck(UnderlyingV1, V1Size,
                                                V1AAInfo, UnderlyingV2,
                                                V2Size, V2AAInfo);
      if (PreciseBaseAlias == NoAlias) {
        // See if the computed offset from the common pointer tells us about the
        // relation of the resulting pointer.
        int64_t GEP2BaseOffset;
        bool GEP2MaxLookupReached;
        SmallVector<VariableGEPIndex, 4> GEP2VariableIndices;
        const Value *GEP2BasePtr =
            DecomposeGEPExpression(GEP2, GEP2BaseOffset, GEP2VariableIndices,
                                   GEP2MaxLookupReached, *DL, AC2, DT);
        const Value *GEP1BasePtr =
            DecomposeGEPExpression(GEP1, GEP1BaseOffset, GEP1VariableIndices,
                                   GEP1MaxLookupReached, *DL, AC1, DT);
        // DecomposeGEPExpression and GetUnderlyingObject should return the
        // same result except when DecomposeGEPExpression has no DataLayout.
        if (GEP1BasePtr != UnderlyingV1 || GEP2BasePtr != UnderlyingV2) {
          assert(!DL &&
                 "DecomposeGEPExpression and GetUnderlyingObject disagree!");
          return MayAlias;
        }
        // If the max search depth is reached the result is undefined
        if (GEP2MaxLookupReached || GEP1MaxLookupReached)
          return MayAlias;

        // Same offsets.
        if (GEP1BaseOffset == GEP2BaseOffset &&
            GEP1VariableIndices == GEP2VariableIndices)
          return NoAlias;
        GEP1VariableIndices.clear();
      }
    }

    // If we get a No or May, then return it immediately, no amount of analysis
    // will improve this situation.
    if (BaseAlias != MustAlias) return BaseAlias;

    // Otherwise, we have a MustAlias.  Since the base pointers alias each other
    // exactly, see if the computed offset from the common pointer tells us
    // about the relation of the resulting pointer.
    const Value *GEP1BasePtr =
        DecomposeGEPExpression(GEP1, GEP1BaseOffset, GEP1VariableIndices,
                               GEP1MaxLookupReached, *DL, AC1, DT);

    int64_t GEP2BaseOffset;
    bool GEP2MaxLookupReached;
    SmallVector<VariableGEPIndex, 4> GEP2VariableIndices;
    const Value *GEP2BasePtr =
        DecomposeGEPExpression(GEP2, GEP2BaseOffset, GEP2VariableIndices,
                               GEP2MaxLookupReached, *DL, AC2, DT);

    // DecomposeGEPExpression and GetUnderlyingObject should return the
    // same result except when DecomposeGEPExpression has no DataLayout.
    if (GEP1BasePtr != UnderlyingV1 || GEP2BasePtr != UnderlyingV2) {
      assert(!DL &&
             "DecomposeGEPExpression and GetUnderlyingObject disagree!");
      return MayAlias;
    }

    // If we know the two GEPs are based off of the exact same pointer (and not
    // just the same underlying object), see if that tells us anything about
    // the resulting pointers.
    if (DL && GEP1->getPointerOperand() == GEP2->getPointerOperand()) {
      AliasResult R = aliasSameBasePointerGEPs(GEP1, V1Size, GEP2, V2Size, *DL);
      // If we couldn't find anything interesting, don't abandon just yet.
      if (R != MayAlias)
        return R;
    }

    // If the max search depth is reached the result is undefined
    if (GEP2MaxLookupReached || GEP1MaxLookupReached)
      return MayAlias;

    // Subtract the GEP2 pointer from the GEP1 pointer to find out their
    // symbolic difference.
    GEP1BaseOffset -= GEP2BaseOffset;
    GetIndexDifference(GEP1VariableIndices, GEP2VariableIndices);

  } else {
    // Check to see if these two pointers are related by the getelementptr
    // instruction.  If one pointer is a GEP with a non-zero index of the other
    // pointer, we know they cannot alias.

    // If both accesses are unknown size, we can't do anything useful here.
    if (V1Size == MemoryLocation::UnknownSize &&
        V2Size == MemoryLocation::UnknownSize)
      return MayAlias;

    AliasResult R = aliasCheck(UnderlyingV1, MemoryLocation::UnknownSize,
                               AAMDNodes(), V2, V2Size, V2AAInfo);
    if (R != MustAlias)
      // If V2 may alias GEP base pointer, conservatively returns MayAlias.
      // If V2 is known not to alias GEP base pointer, then the two values
      // cannot alias per GEP semantics: "A pointer value formed from a
      // getelementptr instruction is associated with the addresses associated
      // with the first operand of the getelementptr".
      return R;

    const Value *GEP1BasePtr =
        DecomposeGEPExpression(GEP1, GEP1BaseOffset, GEP1VariableIndices,
                               GEP1MaxLookupReached, *DL, AC1, DT);

    // DecomposeGEPExpression and GetUnderlyingObject should return the
    // same result except when DecomposeGEPExpression has no DataLayout.
    if (GEP1BasePtr != UnderlyingV1) {
      assert(!DL &&
             "DecomposeGEPExpression and GetUnderlyingObject disagree!");
      return MayAlias;
    }
    // If the max search depth is reached the result is undefined
    if (GEP1MaxLookupReached)
      return MayAlias;
  }

  // In the two GEP Case, if there is no difference in the offsets of the
  // computed pointers, the resultant pointers are a must alias.  This
  // hapens when we have two lexically identical GEP's (for example).
  //
  // In the other case, if we have getelementptr <ptr>, 0, 0, 0, 0, ... and V2
  // must aliases the GEP, the end result is a must alias also.
  if (GEP1BaseOffset == 0 && GEP1VariableIndices.empty())
    return MustAlias;

  // If there is a constant difference between the pointers, but the difference
  // is less than the size of the associated memory object, then we know
  // that the objects are partially overlapping.  If the difference is
  // greater, we know they do not overlap.
  if (GEP1BaseOffset != 0 && GEP1VariableIndices.empty()) {
    if (GEP1BaseOffset >= 0) {
      if (V2Size != MemoryLocation::UnknownSize) {
        if ((uint64_t)GEP1BaseOffset < V2Size)
          return PartialAlias;
        return NoAlias;
      }
    } else {
      // We have the situation where:
      // +                +
      // | BaseOffset     |
      // ---------------->|
      // |-->V1Size       |-------> V2Size
      // GEP1             V2
      // We need to know that V2Size is not unknown, otherwise we might have
      // stripped a gep with negative index ('gep <ptr>, -1, ...).
      if (V1Size != MemoryLocation::UnknownSize &&
          V2Size != MemoryLocation::UnknownSize) {
        if (-(uint64_t)GEP1BaseOffset < V1Size)
          return PartialAlias;
        return NoAlias;
      }
    }
  }

  // Try to distinguish something like &A[i][1] against &A[42][0].
  // Grab the least significant bit set in any of the scales.
  if (!GEP1VariableIndices.empty()) {
    uint64_t Modulo = 0;
    for (unsigned i = 0, e = GEP1VariableIndices.size(); i != e; ++i)
      Modulo |= (uint64_t) GEP1VariableIndices[i].Scale;
    Modulo = Modulo ^ (Modulo & (Modulo - 1));

    // We can compute the difference between the two addresses
    // mod Modulo. Check whether that difference guarantees that the
    // two locations do not alias.
    uint64_t ModOffset = (uint64_t)GEP1BaseOffset & (Modulo - 1);
    if (V1Size != MemoryLocation::UnknownSize &&
        V2Size != MemoryLocation::UnknownSize && ModOffset >= V2Size &&
        V1Size <= Modulo - ModOffset)
      return NoAlias;
  }

  // Statically, we can see that the base objects are the same, but the
  // pointers have dynamic offsets which we can't resolve. And none of our
  // little tricks above worked.
  //
  // TODO: Returning PartialAlias instead of MayAlias is a mild hack; the
  // practical effect of this is protecting TBAA in the case of dynamic
  // indices into arrays of unions or malloc'd memory.
  return PartialAlias;
}

static AliasResult MergeAliasResults(AliasResult A, AliasResult B) {
  // If the results agree, take it.
  if (A == B)
    return A;
  // A mix of PartialAlias and MustAlias is PartialAlias.
  if ((A == PartialAlias && B == MustAlias) ||
      (B == PartialAlias && A == MustAlias))
    return PartialAlias;
  // Otherwise, we don't know anything.
  return MayAlias;
}

/// aliasSelect - Provide a bunch of ad-hoc rules to disambiguate a Select
/// instruction against another.
AliasResult BasicAliasAnalysis::aliasSelect(const SelectInst *SI,
                                            uint64_t SISize,
                                            const AAMDNodes &SIAAInfo,
                                            const Value *V2, uint64_t V2Size,
                                            const AAMDNodes &V2AAInfo) {
  // If the values are Selects with the same condition, we can do a more precise
  // check: just check for aliases between the values on corresponding arms.
  if (const SelectInst *SI2 = dyn_cast<SelectInst>(V2))
    if (SI->getCondition() == SI2->getCondition()) {
      AliasResult Alias =
        aliasCheck(SI->getTrueValue(), SISize, SIAAInfo,
                   SI2->getTrueValue(), V2Size, V2AAInfo);
      if (Alias == MayAlias)
        return MayAlias;
      AliasResult ThisAlias =
        aliasCheck(SI->getFalseValue(), SISize, SIAAInfo,
                   SI2->getFalseValue(), V2Size, V2AAInfo);
      return MergeAliasResults(ThisAlias, Alias);
    }

  // If both arms of the Select node NoAlias or MustAlias V2, then returns
  // NoAlias / MustAlias. Otherwise, returns MayAlias.
  AliasResult Alias =
    aliasCheck(V2, V2Size, V2AAInfo, SI->getTrueValue(), SISize, SIAAInfo);
  if (Alias == MayAlias)
    return MayAlias;

  AliasResult ThisAlias =
    aliasCheck(V2, V2Size, V2AAInfo, SI->getFalseValue(), SISize, SIAAInfo);
  return MergeAliasResults(ThisAlias, Alias);
}

// aliasPHI - Provide a bunch of ad-hoc rules to disambiguate a PHI instruction
// against another.
AliasResult BasicAliasAnalysis::aliasPHI(const PHINode *PN, uint64_t PNSize,
                                         const AAMDNodes &PNAAInfo,
                                         const Value *V2, uint64_t V2Size,
                                         const AAMDNodes &V2AAInfo) {
  // Track phi nodes we have visited. We use this information when we determine
  // value equivalence.
  VisitedPhiBBs.insert(PN->getParent());

  // If the values are PHIs in the same block, we can do a more precise
  // as well as efficient check: just check for aliases between the values
  // on corresponding edges.
  if (const PHINode *PN2 = dyn_cast<PHINode>(V2))
    if (PN2->getParent() == PN->getParent()) {
      LocPair Locs(MemoryLocation(PN, PNSize, PNAAInfo),
                   MemoryLocation(V2, V2Size, V2AAInfo));
      if (PN > V2)
        std::swap(Locs.first, Locs.second);
      // Analyse the PHIs' inputs under the assumption that the PHIs are
      // NoAlias.
      // If the PHIs are May/MustAlias there must be (recursively) an input
      // operand from outside the PHIs' cycle that is MayAlias/MustAlias or
      // there must be an operation on the PHIs within the PHIs' value cycle
      // that causes a MayAlias.
      // Pretend the phis do not alias.
      AliasResult Alias = NoAlias;
      assert(AliasCache.count(Locs) &&
             "There must exist an entry for the phi node");
      AliasResult OrigAliasResult = AliasCache[Locs];
      AliasCache[Locs] = NoAlias;

      for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
        AliasResult ThisAlias =
          aliasCheck(PN->getIncomingValue(i), PNSize, PNAAInfo,
                     PN2->getIncomingValueForBlock(PN->getIncomingBlock(i)),
                     V2Size, V2AAInfo);
        Alias = MergeAliasResults(ThisAlias, Alias);
        if (Alias == MayAlias)
          break;
      }

      // Reset if speculation failed.
      if (Alias != NoAlias)
        AliasCache[Locs] = OrigAliasResult;

      return Alias;
    }

  SmallPtrSet<Value*, 4> UniqueSrc;
  SmallVector<Value*, 4> V1Srcs;
  for (Value *PV1 : PN->incoming_values()) {
    if (isa<PHINode>(PV1))
      // If any of the source itself is a PHI, return MayAlias conservatively
      // to avoid compile time explosion. The worst possible case is if both
      // sides are PHI nodes. In which case, this is O(m x n) time where 'm'
      // and 'n' are the number of PHI sources.
      return MayAlias;
    if (UniqueSrc.insert(PV1).second)
      V1Srcs.push_back(PV1);
  }

  AliasResult Alias = aliasCheck(V2, V2Size, V2AAInfo,
                                 V1Srcs[0], PNSize, PNAAInfo);
  // Early exit if the check of the first PHI source against V2 is MayAlias.
  // Other results are not possible.
  if (Alias == MayAlias)
    return MayAlias;

  // If all sources of the PHI node NoAlias or MustAlias V2, then returns
  // NoAlias / MustAlias. Otherwise, returns MayAlias.
  for (unsigned i = 1, e = V1Srcs.size(); i != e; ++i) {
    Value *V = V1Srcs[i];

    AliasResult ThisAlias = aliasCheck(V2, V2Size, V2AAInfo,
                                       V, PNSize, PNAAInfo);
    Alias = MergeAliasResults(ThisAlias, Alias);
    if (Alias == MayAlias)
      break;
  }

  return Alias;
}

// aliasCheck - Provide a bunch of ad-hoc rules to disambiguate in common cases,
// such as array references.
//
AliasResult BasicAliasAnalysis::aliasCheck(const Value *V1, uint64_t V1Size,
                                           AAMDNodes V1AAInfo, const Value *V2,
                                           uint64_t V2Size,
                                           AAMDNodes V2AAInfo) {
  // If either of the memory references is empty, it doesn't matter what the
  // pointer values are.
  if (V1Size == 0 || V2Size == 0)
    return NoAlias;

  // Strip off any casts if they exist.
  V1 = V1->stripPointerCasts();
  V2 = V2->stripPointerCasts();

  // If V1 or V2 is undef, the result is NoAlias because we can always pick a
  // value for undef that aliases nothing in the program.
  if (isa<UndefValue>(V1) || isa<UndefValue>(V2))
    return NoAlias;

  // Are we checking for alias of the same value?
  // Because we look 'through' phi nodes we could look at "Value" pointers from
  // different iterations. We must therefore make sure that this is not the
  // case. The function isValueEqualInPotentialCycles ensures that this cannot
  // happen by looking at the visited phi nodes and making sure they cannot
  // reach the value.
  if (isValueEqualInPotentialCycles(V1, V2))
    return MustAlias;

  if (!V1->getType()->isPointerTy() || !V2->getType()->isPointerTy())
    return NoAlias;  // Scalars cannot alias each other

  // Figure out what objects these things are pointing to if we can.
  const Value *O1 = GetUnderlyingObject(V1, *DL, MaxLookupSearchDepth);
  const Value *O2 = GetUnderlyingObject(V2, *DL, MaxLookupSearchDepth);

  // Null values in the default address space don't point to any object, so they
  // don't alias any other pointer.
  if (const ConstantPointerNull *CPN = dyn_cast<ConstantPointerNull>(O1))
    if (CPN->getType()->getAddressSpace() == 0)
      return NoAlias;
  if (const ConstantPointerNull *CPN = dyn_cast<ConstantPointerNull>(O2))
    if (CPN->getType()->getAddressSpace() == 0)
      return NoAlias;

  if (O1 != O2) {
    // If V1/V2 point to two different objects we know that we have no alias.
    if (isIdentifiedObject(O1) && isIdentifiedObject(O2))
      return NoAlias;

    // Constant pointers can't alias with non-const isIdentifiedObject objects.
    if ((isa<Constant>(O1) && isIdentifiedObject(O2) && !isa<Constant>(O2)) ||
        (isa<Constant>(O2) && isIdentifiedObject(O1) && !isa<Constant>(O1)))
      return NoAlias;

    // Function arguments can't alias with things that are known to be
    // unambigously identified at the function level.
    if ((isa<Argument>(O1) && isIdentifiedFunctionLocal(O2)) ||
        (isa<Argument>(O2) && isIdentifiedFunctionLocal(O1)))
      return NoAlias;

    // Most objects can't alias null.
    if ((isa<ConstantPointerNull>(O2) && isKnownNonNull(O1)) ||
        (isa<ConstantPointerNull>(O1) && isKnownNonNull(O2)))
      return NoAlias;

    // If one pointer is the result of a call/invoke or load and the other is a
    // non-escaping local object within the same function, then we know the
    // object couldn't escape to a point where the call could return it.
    //
    // Note that if the pointers are in different functions, there are a
    // variety of complications. A call with a nocapture argument may still
    // temporary store the nocapture argument's value in a temporary memory
    // location if that memory location doesn't escape. Or it may pass a
    // nocapture value to other functions as long as they don't capture it.
    if (isEscapeSource(O1) && isNonEscapingLocalObject(O2))
      return NoAlias;
    if (isEscapeSource(O2) && isNonEscapingLocalObject(O1))
      return NoAlias;
  }

  // If the size of one access is larger than the entire object on the other
  // side, then we know such behavior is undefined and can assume no alias.
  if (DL)
    if ((V1Size != MemoryLocation::UnknownSize &&
         isObjectSmallerThan(O2, V1Size, *DL, *TLI)) ||
        (V2Size != MemoryLocation::UnknownSize &&
         isObjectSmallerThan(O1, V2Size, *DL, *TLI)))
      return NoAlias;

  // Check the cache before climbing up use-def chains. This also terminates
  // otherwise infinitely recursive queries.
  LocPair Locs(MemoryLocation(V1, V1Size, V1AAInfo),
               MemoryLocation(V2, V2Size, V2AAInfo));
  if (V1 > V2)
    std::swap(Locs.first, Locs.second);
  std::pair<AliasCacheTy::iterator, bool> Pair =
    AliasCache.insert(std::make_pair(Locs, MayAlias));
  if (!Pair.second)
    return Pair.first->second;

  // FIXME: This isn't aggressively handling alias(GEP, PHI) for example: if the
  // GEP can't simplify, we don't even look at the PHI cases.
  if (!isa<GEPOperator>(V1) && isa<GEPOperator>(V2)) {
    std::swap(V1, V2);
    std::swap(V1Size, V2Size);
    std::swap(O1, O2);
    std::swap(V1AAInfo, V2AAInfo);
  }
  if (const GEPOperator *GV1 = dyn_cast<GEPOperator>(V1)) {
    AliasResult Result = aliasGEP(GV1, V1Size, V1AAInfo, V2, V2Size, V2AAInfo, O1, O2);
    if (Result != MayAlias) return AliasCache[Locs] = Result;
  }

  if (isa<PHINode>(V2) && !isa<PHINode>(V1)) {
    std::swap(V1, V2);
    std::swap(V1Size, V2Size);
    std::swap(V1AAInfo, V2AAInfo);
  }
  if (const PHINode *PN = dyn_cast<PHINode>(V1)) {
    AliasResult Result = aliasPHI(PN, V1Size, V1AAInfo,
                                  V2, V2Size, V2AAInfo);
    if (Result != MayAlias) return AliasCache[Locs] = Result;
  }

  if (isa<SelectInst>(V2) && !isa<SelectInst>(V1)) {
    std::swap(V1, V2);
    std::swap(V1Size, V2Size);
    std::swap(V1AAInfo, V2AAInfo);
  }
  if (const SelectInst *S1 = dyn_cast<SelectInst>(V1)) {
    AliasResult Result = aliasSelect(S1, V1Size, V1AAInfo,
                                     V2, V2Size, V2AAInfo);
    if (Result != MayAlias) return AliasCache[Locs] = Result;
  }

  // If both pointers are pointing into the same object and one of them
  // accesses is accessing the entire object, then the accesses must
  // overlap in some way.
  if (DL && O1 == O2)
    if ((V1Size != MemoryLocation::UnknownSize &&
         isObjectSize(O1, V1Size, *DL, *TLI)) ||
        (V2Size != MemoryLocation::UnknownSize &&
         isObjectSize(O2, V2Size, *DL, *TLI)))
      return AliasCache[Locs] = PartialAlias;

  AliasResult Result =
      AliasAnalysis::alias(MemoryLocation(V1, V1Size, V1AAInfo),
                           MemoryLocation(V2, V2Size, V2AAInfo));
  return AliasCache[Locs] = Result;
}

bool BasicAliasAnalysis::isValueEqualInPotentialCycles(const Value *V,
                                                       const Value *V2) {
  if (V != V2)
    return false;

  const Instruction *Inst = dyn_cast<Instruction>(V);
  if (!Inst)
    return true;

  if (VisitedPhiBBs.empty())
    return true;

  if (VisitedPhiBBs.size() > MaxNumPhiBBsValueReachabilityCheck)
    return false;

  // Use dominance or loop info if available.
  DominatorTreeWrapperPass *DTWP =
      getAnalysisIfAvailable<DominatorTreeWrapperPass>();
  DominatorTree *DT = DTWP ? &DTWP->getDomTree() : nullptr;
  auto *LIWP = getAnalysisIfAvailable<LoopInfoWrapperPass>();
  LoopInfo *LI = LIWP ? &LIWP->getLoopInfo() : nullptr;

  // Make sure that the visited phis cannot reach the Value. This ensures that
  // the Values cannot come from different iterations of a potential cycle the
  // phi nodes could be involved in.
  for (auto *P : VisitedPhiBBs)
    if (isPotentiallyReachable(P->begin(), Inst, DT, LI))
      return false;

  return true;
}

/// GetIndexDifference - Dest and Src are the variable indices from two
/// decomposed GetElementPtr instructions GEP1 and GEP2 which have common base
/// pointers.  Subtract the GEP2 indices from GEP1 to find the symbolic
/// difference between the two pointers.
void BasicAliasAnalysis::GetIndexDifference(
    SmallVectorImpl<VariableGEPIndex> &Dest,
    const SmallVectorImpl<VariableGEPIndex> &Src) {
  if (Src.empty())
    return;

  for (unsigned i = 0, e = Src.size(); i != e; ++i) {
    const Value *V = Src[i].V;
    ExtensionKind Extension = Src[i].Extension;
    int64_t Scale = Src[i].Scale;

    // Find V in Dest.  This is N^2, but pointer indices almost never have more
    // than a few variable indexes.
    for (unsigned j = 0, e = Dest.size(); j != e; ++j) {
      if (!isValueEqualInPotentialCycles(Dest[j].V, V) ||
          Dest[j].Extension != Extension)
        continue;

      // If we found it, subtract off Scale V's from the entry in Dest.  If it
      // goes to zero, remove the entry.
      if (Dest[j].Scale != Scale)
        Dest[j].Scale -= Scale;
      else
        Dest.erase(Dest.begin() + j);
      Scale = 0;
      break;
    }

    // If we didn't consume this entry, add it to the end of the Dest list.
    if (Scale) {
      VariableGEPIndex Entry = { V, Extension, -Scale };
      Dest.push_back(Entry);
    }
  }
}
