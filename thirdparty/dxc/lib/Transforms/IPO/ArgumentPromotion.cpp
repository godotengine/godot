//===-- ArgumentPromotion.cpp - Promote by-reference arguments ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass promotes "by reference" arguments to be "by value" arguments.  In
// practice, this means looking for internal functions that have pointer
// arguments.  If it can prove, through the use of alias analysis, that an
// argument is *only* loaded, then it can pass the value into the function
// instead of the address of the value.  This can cause recursive simplification
// of code and lead to the elimination of allocas (especially in C++ template
// code like the STL).
//
// This pass also handles aggregate arguments that are passed into a function,
// scalarizing them if the elements of the aggregate are only loaded.  Note that
// by default it refuses to scalarize aggregates which would require passing in
// more than three operands to the function, because passing thousands of
// operands for a large array or structure is unprofitable! This limit can be
// configured or disabled, however.
//
// Note that this transformation could also be done for arguments that are only
// stored to (returning the value instead), but does not currently.  This case
// would be best handled when and if LLVM begins supporting multiple return
// values from functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/CallGraphSCCPass.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <set>
using namespace llvm;

#define DEBUG_TYPE "argpromotion"

STATISTIC(NumArgumentsPromoted , "Number of pointer arguments promoted");
STATISTIC(NumAggregatesPromoted, "Number of aggregate arguments promoted");
STATISTIC(NumByValArgsPromoted , "Number of byval arguments promoted");
STATISTIC(NumArgumentsDead     , "Number of dead pointer args eliminated");

namespace {
  /// ArgPromotion - The 'by reference' to 'by value' argument promotion pass.
  ///
  struct ArgPromotion : public CallGraphSCCPass {
    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.addRequired<AliasAnalysis>();
      CallGraphSCCPass::getAnalysisUsage(AU);
    }

    bool runOnSCC(CallGraphSCC &SCC) override;
    static char ID; // Pass identification, replacement for typeid
    explicit ArgPromotion(unsigned maxElements = 3)
        : CallGraphSCCPass(ID), maxElements(maxElements) {
      initializeArgPromotionPass(*PassRegistry::getPassRegistry());
    }

    /// A vector used to hold the indices of a single GEP instruction
    typedef std::vector<uint64_t> IndicesVector;

  private:
    bool isDenselyPacked(Type *type, const DataLayout &DL);
    bool canPaddingBeAccessed(Argument *Arg);
    CallGraphNode *PromoteArguments(CallGraphNode *CGN);
    bool isSafeToPromoteArgument(Argument *Arg, bool isByVal) const;
    CallGraphNode *DoPromotion(Function *F,
                              SmallPtrSetImpl<Argument*> &ArgsToPromote,
                              SmallPtrSetImpl<Argument*> &ByValArgsToTransform);
    
    using llvm::Pass::doInitialization;
    bool doInitialization(CallGraph &CG) override;
    /// The maximum number of elements to expand, or 0 for unlimited.
    unsigned maxElements;
    DenseMap<const Function *, DISubprogram *> FunctionDIs;
  };
}

char ArgPromotion::ID = 0;
INITIALIZE_PASS_BEGIN(ArgPromotion, "argpromotion",
                "Promote 'by reference' arguments to scalars", false, false)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_DEPENDENCY(CallGraphWrapperPass)
INITIALIZE_PASS_END(ArgPromotion, "argpromotion",
                "Promote 'by reference' arguments to scalars", false, false)

Pass *llvm::createArgumentPromotionPass(unsigned maxElements) {
  return new ArgPromotion(maxElements);
}

bool ArgPromotion::runOnSCC(CallGraphSCC &SCC) {
  bool Changed = false, LocalChange;

  do {  // Iterate until we stop promoting from this SCC.
    LocalChange = false;
    // Attempt to promote arguments from all functions in this SCC.
    for (CallGraphSCC::iterator I = SCC.begin(), E = SCC.end(); I != E; ++I) {
      if (CallGraphNode *CGN = PromoteArguments(*I)) {
        LocalChange = true;
        SCC.ReplaceNode(*I, CGN);
      }
    }
    Changed |= LocalChange;               // Remember that we changed something.
  } while (LocalChange);
  
  return Changed;
}

/// \brief Checks if a type could have padding bytes.
bool ArgPromotion::isDenselyPacked(Type *type, const DataLayout &DL) {

  // There is no size information, so be conservative.
  if (!type->isSized())
    return false;

  // If the alloc size is not equal to the storage size, then there are padding
  // bytes. For x86_fp80 on x86-64, size: 80 alloc size: 128.
  if (DL.getTypeSizeInBits(type) != DL.getTypeAllocSizeInBits(type))
    return false;

  if (!isa<CompositeType>(type))
    return true;

  // For homogenous sequential types, check for padding within members.
  if (SequentialType *seqTy = dyn_cast<SequentialType>(type))
    return isa<PointerType>(seqTy) ||
           isDenselyPacked(seqTy->getElementType(), DL);

  // Check for padding within and between elements of a struct.
  StructType *StructTy = cast<StructType>(type);
  const StructLayout *Layout = DL.getStructLayout(StructTy);
  uint64_t StartPos = 0;
  for (unsigned i = 0, E = StructTy->getNumElements(); i < E; ++i) {
    Type *ElTy = StructTy->getElementType(i);
    if (!isDenselyPacked(ElTy, DL))
      return false;
    if (StartPos != Layout->getElementOffsetInBits(i))
      return false;
    StartPos += DL.getTypeAllocSizeInBits(ElTy);
  }

  return true;
}

/// \brief Checks if the padding bytes of an argument could be accessed.
bool ArgPromotion::canPaddingBeAccessed(Argument *arg) {

  assert(arg->hasByValAttr());

  // Track all the pointers to the argument to make sure they are not captured.
  SmallPtrSet<Value *, 16> PtrValues;
  PtrValues.insert(arg);

  // Track all of the stores.
  SmallVector<StoreInst *, 16> Stores;

  // Scan through the uses recursively to make sure the pointer is always used
  // sanely.
  SmallVector<Value *, 16> WorkList;
  WorkList.insert(WorkList.end(), arg->user_begin(), arg->user_end());
  while (!WorkList.empty()) {
    Value *V = WorkList.back();
    WorkList.pop_back();
    if (isa<GetElementPtrInst>(V) || isa<PHINode>(V)) {
      if (PtrValues.insert(V).second)
        WorkList.insert(WorkList.end(), V->user_begin(), V->user_end());
    } else if (StoreInst *Store = dyn_cast<StoreInst>(V)) {
      Stores.push_back(Store);
    } else if (!isa<LoadInst>(V)) {
      return true;
    }
  }

// Check to make sure the pointers aren't captured
  for (StoreInst *Store : Stores)
    if (PtrValues.count(Store->getValueOperand()))
      return true;

  return false;
}

/// PromoteArguments - This method checks the specified function to see if there
/// are any promotable arguments and if it is safe to promote the function (for
/// example, all callers are direct).  If safe to promote some arguments, it
/// calls the DoPromotion method.
///
CallGraphNode *ArgPromotion::PromoteArguments(CallGraphNode *CGN) {
  Function *F = CGN->getFunction();

  // Make sure that it is local to this module.
  if (!F || !F->hasLocalLinkage()) return nullptr;

  // Don't promote arguments for variadic functions. Adding, removing, or
  // changing non-pack parameters can change the classification of pack
  // parameters. Frontends encode that classification at the call site in the
  // IR, while in the callee the classification is determined dynamically based
  // on the number of registers consumed so far.
  if (F->isVarArg()) return nullptr;

  // First check: see if there are any pointer arguments!  If not, quick exit.
  SmallVector<Argument*, 16> PointerArgs;
  for (Function::arg_iterator I = F->arg_begin(), E = F->arg_end(); I != E; ++I)
    if (I->getType()->isPointerTy())
      PointerArgs.push_back(I);
  if (PointerArgs.empty()) return nullptr;

  // Second check: make sure that all callers are direct callers.  We can't
  // transform functions that have indirect callers.  Also see if the function
  // is self-recursive.
  bool isSelfRecursive = false;
  for (Use &U : F->uses()) {
    CallSite CS(U.getUser());
    // Must be a direct call.
    if (CS.getInstruction() == nullptr || !CS.isCallee(&U)) return nullptr;
    
    if (CS.getInstruction()->getParent()->getParent() == F)
      isSelfRecursive = true;
  }
  
  const DataLayout &DL = F->getParent()->getDataLayout();

  // Check to see which arguments are promotable.  If an argument is promotable,
  // add it to ArgsToPromote.
  SmallPtrSet<Argument*, 8> ArgsToPromote;
  SmallPtrSet<Argument*, 8> ByValArgsToTransform;
  for (unsigned i = 0, e = PointerArgs.size(); i != e; ++i) {
    Argument *PtrArg = PointerArgs[i];
    Type *AgTy = cast<PointerType>(PtrArg->getType())->getElementType();

    // Replace sret attribute with noalias. This reduces register pressure by
    // avoiding a register copy.
    if (PtrArg->hasStructRetAttr()) {
      unsigned ArgNo = PtrArg->getArgNo();
      F->setAttributes(
          F->getAttributes()
              .removeAttribute(F->getContext(), ArgNo + 1, Attribute::StructRet)
              .addAttribute(F->getContext(), ArgNo + 1, Attribute::NoAlias));
      for (Use &U : F->uses()) {
        CallSite CS(U.getUser());
        CS.setAttributes(
            CS.getAttributes()
                .removeAttribute(F->getContext(), ArgNo + 1,
                                 Attribute::StructRet)
                .addAttribute(F->getContext(), ArgNo + 1, Attribute::NoAlias));
      }
    }

    // If this is a byval argument, and if the aggregate type is small, just
    // pass the elements, which is always safe, if the passed value is densely
    // packed or if we can prove the padding bytes are never accessed. This does
    // not apply to inalloca.
    bool isSafeToPromote =
        PtrArg->hasByValAttr() &&
        (isDenselyPacked(AgTy, DL) || !canPaddingBeAccessed(PtrArg));
    if (isSafeToPromote) {
      if (StructType *STy = dyn_cast<StructType>(AgTy)) {
        if (maxElements > 0 && STy->getNumElements() > maxElements) {
          DEBUG(dbgs() << "argpromotion disable promoting argument '"
                << PtrArg->getName() << "' because it would require adding more"
                << " than " << maxElements << " arguments to the function.\n");
          continue;
        }
        
        // If all the elements are single-value types, we can promote it.
        bool AllSimple = true;
        for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i) {
          if (!STy->getElementType(i)->isSingleValueType()) {
            AllSimple = false;
            break;
          }
        }

        // Safe to transform, don't even bother trying to "promote" it.
        // Passing the elements as a scalar will allow scalarrepl to hack on
        // the new alloca we introduce.
        if (AllSimple) {
          ByValArgsToTransform.insert(PtrArg);
          continue;
        }
      }
    }

    // If the argument is a recursive type and we're in a recursive
    // function, we could end up infinitely peeling the function argument.
    if (isSelfRecursive) {
      if (StructType *STy = dyn_cast<StructType>(AgTy)) {
        bool RecursiveType = false;
        for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i) {
          if (STy->getElementType(i) == PtrArg->getType()) {
            RecursiveType = true;
            break;
          }
        }
        if (RecursiveType)
          continue;
      }
    }
    
    // Otherwise, see if we can promote the pointer to its value.
    if (isSafeToPromoteArgument(PtrArg, PtrArg->hasByValOrInAllocaAttr()))
      ArgsToPromote.insert(PtrArg);
  }

  // No promotable pointer arguments.
  if (ArgsToPromote.empty() && ByValArgsToTransform.empty()) 
    return nullptr;

  return DoPromotion(F, ArgsToPromote, ByValArgsToTransform);
}

/// AllCallersPassInValidPointerForArgument - Return true if we can prove that
/// all callees pass in a valid pointer for the specified function argument.
static bool AllCallersPassInValidPointerForArgument(Argument *Arg) {
  Function *Callee = Arg->getParent();
  const DataLayout &DL = Callee->getParent()->getDataLayout();

  unsigned ArgNo = Arg->getArgNo();

  // Look at all call sites of the function.  At this pointer we know we only
  // have direct callees.
  for (User *U : Callee->users()) {
    CallSite CS(U);
    assert(CS && "Should only have direct calls!");

    if (!isDereferenceablePointer(CS.getArgument(ArgNo), DL))
      return false;
  }
  return true;
}

/// Returns true if Prefix is a prefix of longer. That means, Longer has a size
/// that is greater than or equal to the size of prefix, and each of the
/// elements in Prefix is the same as the corresponding elements in Longer.
///
/// This means it also returns true when Prefix and Longer are equal!
static bool IsPrefix(const ArgPromotion::IndicesVector &Prefix,
                     const ArgPromotion::IndicesVector &Longer) {
  if (Prefix.size() > Longer.size())
    return false;
  return std::equal(Prefix.begin(), Prefix.end(), Longer.begin());
}


/// Checks if Indices, or a prefix of Indices, is in Set.
static bool PrefixIn(const ArgPromotion::IndicesVector &Indices,
                     std::set<ArgPromotion::IndicesVector> &Set) {
    std::set<ArgPromotion::IndicesVector>::iterator Low;
    Low = Set.upper_bound(Indices);
    if (Low != Set.begin())
      Low--;
    // Low is now the last element smaller than or equal to Indices. This means
    // it points to a prefix of Indices (possibly Indices itself), if such
    // prefix exists.
    //
    // This load is safe if any prefix of its operands is safe to load.
    return Low != Set.end() && IsPrefix(*Low, Indices);
}

/// Mark the given indices (ToMark) as safe in the given set of indices
/// (Safe). Marking safe usually means adding ToMark to Safe. However, if there
/// is already a prefix of Indices in Safe, Indices are implicitely marked safe
/// already. Furthermore, any indices that Indices is itself a prefix of, are
/// removed from Safe (since they are implicitely safe because of Indices now).
static void MarkIndicesSafe(const ArgPromotion::IndicesVector &ToMark,
                            std::set<ArgPromotion::IndicesVector> &Safe) {
  std::set<ArgPromotion::IndicesVector>::iterator Low;
  Low = Safe.upper_bound(ToMark);
  // Guard against the case where Safe is empty
  if (Low != Safe.begin())
    Low--;
  // Low is now the last element smaller than or equal to Indices. This
  // means it points to a prefix of Indices (possibly Indices itself), if
  // such prefix exists.
  if (Low != Safe.end()) {
    if (IsPrefix(*Low, ToMark))
      // If there is already a prefix of these indices (or exactly these
      // indices) marked a safe, don't bother adding these indices
      return;

    // Increment Low, so we can use it as a "insert before" hint
    ++Low;
  }
  // Insert
  Low = Safe.insert(Low, ToMark);
  ++Low;
  // If there we're a prefix of longer index list(s), remove those
  std::set<ArgPromotion::IndicesVector>::iterator End = Safe.end();
  while (Low != End && IsPrefix(ToMark, *Low)) {
    std::set<ArgPromotion::IndicesVector>::iterator Remove = Low;
    ++Low;
    Safe.erase(Remove);
  }
}

/// isSafeToPromoteArgument - As you might guess from the name of this method,
/// it checks to see if it is both safe and useful to promote the argument.
/// This method limits promotion of aggregates to only promote up to three
/// elements of the aggregate in order to avoid exploding the number of
/// arguments passed in.
bool ArgPromotion::isSafeToPromoteArgument(Argument *Arg,
                                           bool isByValOrInAlloca) const {
  typedef std::set<IndicesVector> GEPIndicesSet;

  // Quick exit for unused arguments
  if (Arg->use_empty())
    return true;

  // We can only promote this argument if all of the uses are loads, or are GEP
  // instructions (with constant indices) that are subsequently loaded.
  //
  // Promoting the argument causes it to be loaded in the caller
  // unconditionally. This is only safe if we can prove that either the load
  // would have happened in the callee anyway (ie, there is a load in the entry
  // block) or the pointer passed in at every call site is guaranteed to be
  // valid.
  // In the former case, invalid loads can happen, but would have happened
  // anyway, in the latter case, invalid loads won't happen. This prevents us
  // from introducing an invalid load that wouldn't have happened in the
  // original code.
  //
  // This set will contain all sets of indices that are loaded in the entry
  // block, and thus are safe to unconditionally load in the caller.
  //
  // This optimization is also safe for InAlloca parameters, because it verifies
  // that the address isn't captured.
  GEPIndicesSet SafeToUnconditionallyLoad;

  // This set contains all the sets of indices that we are planning to promote.
  // This makes it possible to limit the number of arguments added.
  GEPIndicesSet ToPromote;

  // If the pointer is always valid, any load with first index 0 is valid.
  if (isByValOrInAlloca || AllCallersPassInValidPointerForArgument(Arg))
    SafeToUnconditionallyLoad.insert(IndicesVector(1, 0));

  // First, iterate the entry block and mark loads of (geps of) arguments as
  // safe.
  BasicBlock *EntryBlock = Arg->getParent()->begin();
  // Declare this here so we can reuse it
  IndicesVector Indices;
  for (BasicBlock::iterator I = EntryBlock->begin(), E = EntryBlock->end();
       I != E; ++I)
    if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
      Value *V = LI->getPointerOperand();
      if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(V)) {
        V = GEP->getPointerOperand();
        if (V == Arg) {
          // This load actually loads (part of) Arg? Check the indices then.
          Indices.reserve(GEP->getNumIndices());
          for (User::op_iterator II = GEP->idx_begin(), IE = GEP->idx_end();
               II != IE; ++II)
            if (ConstantInt *CI = dyn_cast<ConstantInt>(*II))
              Indices.push_back(CI->getSExtValue());
            else
              // We found a non-constant GEP index for this argument? Bail out
              // right away, can't promote this argument at all.
              return false;

          // Indices checked out, mark them as safe
          MarkIndicesSafe(Indices, SafeToUnconditionallyLoad);
          Indices.clear();
        }
      } else if (V == Arg) {
        // Direct loads are equivalent to a GEP with a single 0 index.
        MarkIndicesSafe(IndicesVector(1, 0), SafeToUnconditionallyLoad);
      }
    }

  // Now, iterate all uses of the argument to see if there are any uses that are
  // not (GEP+)loads, or any (GEP+)loads that are not safe to promote.
  SmallVector<LoadInst*, 16> Loads;
  IndicesVector Operands;
  for (Use &U : Arg->uses()) {
    User *UR = U.getUser();
    Operands.clear();
    if (LoadInst *LI = dyn_cast<LoadInst>(UR)) {
      // Don't hack volatile/atomic loads
      if (!LI->isSimple()) return false;
      Loads.push_back(LI);
      // Direct loads are equivalent to a GEP with a zero index and then a load.
      Operands.push_back(0);
    } else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(UR)) {
      if (GEP->use_empty()) {
        // Dead GEP's cause trouble later.  Just remove them if we run into
        // them.
        getAnalysis<AliasAnalysis>().deleteValue(GEP);
        GEP->eraseFromParent();
        // TODO: This runs the above loop over and over again for dead GEPs
        // Couldn't we just do increment the UI iterator earlier and erase the
        // use?
        return isSafeToPromoteArgument(Arg, isByValOrInAlloca);
      }

      // Ensure that all of the indices are constants.
      for (User::op_iterator i = GEP->idx_begin(), e = GEP->idx_end();
        i != e; ++i)
        if (ConstantInt *C = dyn_cast<ConstantInt>(*i))
          Operands.push_back(C->getSExtValue());
        else
          return false;  // Not a constant operand GEP!

      // Ensure that the only users of the GEP are load instructions.
      for (User *GEPU : GEP->users())
        if (LoadInst *LI = dyn_cast<LoadInst>(GEPU)) {
          // Don't hack volatile/atomic loads
          if (!LI->isSimple()) return false;
          Loads.push_back(LI);
        } else {
          // Other uses than load?
          return false;
        }
    } else {
      return false;  // Not a load or a GEP.
    }

    // Now, see if it is safe to promote this load / loads of this GEP. Loading
    // is safe if Operands, or a prefix of Operands, is marked as safe.
    if (!PrefixIn(Operands, SafeToUnconditionallyLoad))
      return false;

    // See if we are already promoting a load with these indices. If not, check
    // to make sure that we aren't promoting too many elements.  If so, nothing
    // to do.
    if (ToPromote.find(Operands) == ToPromote.end()) {
      if (maxElements > 0 && ToPromote.size() == maxElements) {
        DEBUG(dbgs() << "argpromotion not promoting argument '"
              << Arg->getName() << "' because it would require adding more "
              << "than " << maxElements << " arguments to the function.\n");
        // We limit aggregate promotion to only promoting up to a fixed number
        // of elements of the aggregate.
        return false;
      }
      ToPromote.insert(std::move(Operands));
    }
  }

  if (Loads.empty()) return true;  // No users, this is a dead argument.

  // Okay, now we know that the argument is only used by load instructions and
  // it is safe to unconditionally perform all of them. Use alias analysis to
  // check to see if the pointer is guaranteed to not be modified from entry of
  // the function to each of the load instructions.

  // Because there could be several/many load instructions, remember which
  // blocks we know to be transparent to the load.
  SmallPtrSet<BasicBlock*, 16> TranspBlocks;

  AliasAnalysis &AA = getAnalysis<AliasAnalysis>();

  for (unsigned i = 0, e = Loads.size(); i != e; ++i) {
    // Check to see if the load is invalidated from the start of the block to
    // the load itself.
    LoadInst *Load = Loads[i];
    BasicBlock *BB = Load->getParent();

    MemoryLocation Loc = MemoryLocation::get(Load);
    if (AA.canInstructionRangeModRef(BB->front(), *Load, Loc,
        AliasAnalysis::Mod))
      return false;  // Pointer is invalidated!

    // Now check every path from the entry block to the load for transparency.
    // To do this, we perform a depth first search on the inverse CFG from the
    // loading block.
    for (BasicBlock *P : predecessors(BB)) {
      for (BasicBlock *TranspBB : inverse_depth_first_ext(P, TranspBlocks))
        if (AA.canBasicBlockModify(*TranspBB, Loc))
          return false;
    }
  }

  // If the path from the entry of the function to each load is free of
  // instructions that potentially invalidate the load, we can make the
  // transformation!
  return true;
}

/// DoPromotion - This method actually performs the promotion of the specified
/// arguments, and returns the new function.  At this point, we know that it's
/// safe to do so.
CallGraphNode *ArgPromotion::DoPromotion(Function *F,
                             SmallPtrSetImpl<Argument*> &ArgsToPromote,
                             SmallPtrSetImpl<Argument*> &ByValArgsToTransform) {

  // Start by computing a new prototype for the function, which is the same as
  // the old function, but has modified arguments.
  FunctionType *FTy = F->getFunctionType();
  std::vector<Type*> Params;

  typedef std::set<std::pair<Type *, IndicesVector>> ScalarizeTable;

  // ScalarizedElements - If we are promoting a pointer that has elements
  // accessed out of it, keep track of which elements are accessed so that we
  // can add one argument for each.
  //
  // Arguments that are directly loaded will have a zero element value here, to
  // handle cases where there are both a direct load and GEP accesses.
  //
  std::map<Argument*, ScalarizeTable> ScalarizedElements;

  // OriginalLoads - Keep track of a representative load instruction from the
  // original function so that we can tell the alias analysis implementation
  // what the new GEP/Load instructions we are inserting look like.
  // We need to keep the original loads for each argument and the elements
  // of the argument that are accessed.
  std::map<std::pair<Argument*, IndicesVector>, LoadInst*> OriginalLoads;

  // Attribute - Keep track of the parameter attributes for the arguments
  // that we are *not* promoting. For the ones that we do promote, the parameter
  // attributes are lost
  SmallVector<AttributeSet, 8> AttributesVec;
  const AttributeSet &PAL = F->getAttributes();

  // Add any return attributes.
  if (PAL.hasAttributes(AttributeSet::ReturnIndex))
    AttributesVec.push_back(AttributeSet::get(F->getContext(),
                                              PAL.getRetAttributes()));

  // First, determine the new argument list
  unsigned ArgIndex = 1;
  for (Function::arg_iterator I = F->arg_begin(), E = F->arg_end(); I != E;
       ++I, ++ArgIndex) {
    if (ByValArgsToTransform.count(I)) {
      // Simple byval argument? Just add all the struct element types.
      Type *AgTy = cast<PointerType>(I->getType())->getElementType();
      StructType *STy = cast<StructType>(AgTy);
      Params.insert(Params.end(), STy->element_begin(), STy->element_end());
      ++NumByValArgsPromoted;
    } else if (!ArgsToPromote.count(I)) {
      // Unchanged argument
      Params.push_back(I->getType());
      AttributeSet attrs = PAL.getParamAttributes(ArgIndex);
      if (attrs.hasAttributes(ArgIndex)) {
        AttrBuilder B(attrs, ArgIndex);
        AttributesVec.
          push_back(AttributeSet::get(F->getContext(), Params.size(), B));
      }
    } else if (I->use_empty()) {
      // Dead argument (which are always marked as promotable)
      ++NumArgumentsDead;
    } else {
      // Okay, this is being promoted. This means that the only uses are loads
      // or GEPs which are only used by loads

      // In this table, we will track which indices are loaded from the argument
      // (where direct loads are tracked as no indices).
      ScalarizeTable &ArgIndices = ScalarizedElements[I];
      for (User *U : I->users()) {
        Instruction *UI = cast<Instruction>(U);
        Type *SrcTy;
        if (LoadInst *L = dyn_cast<LoadInst>(UI))
          SrcTy = L->getType();
        else
          SrcTy = cast<GetElementPtrInst>(UI)->getSourceElementType();
        IndicesVector Indices;
        Indices.reserve(UI->getNumOperands() - 1);
        // Since loads will only have a single operand, and GEPs only a single
        // non-index operand, this will record direct loads without any indices,
        // and gep+loads with the GEP indices.
        for (User::op_iterator II = UI->op_begin() + 1, IE = UI->op_end();
             II != IE; ++II)
          Indices.push_back(cast<ConstantInt>(*II)->getSExtValue());
        // GEPs with a single 0 index can be merged with direct loads
        if (Indices.size() == 1 && Indices.front() == 0)
          Indices.clear();
        ArgIndices.insert(std::make_pair(SrcTy, Indices));
        LoadInst *OrigLoad;
        if (LoadInst *L = dyn_cast<LoadInst>(UI))
          OrigLoad = L;
        else
          // Take any load, we will use it only to update Alias Analysis
          OrigLoad = cast<LoadInst>(UI->user_back());
        OriginalLoads[std::make_pair(I, Indices)] = OrigLoad;
      }

      // Add a parameter to the function for each element passed in.
      for (ScalarizeTable::iterator SI = ArgIndices.begin(),
             E = ArgIndices.end(); SI != E; ++SI) {
        // not allowed to dereference ->begin() if size() is 0
        Params.push_back(GetElementPtrInst::getIndexedType(
            cast<PointerType>(I->getType()->getScalarType())->getElementType(),
            SI->second));
        assert(Params.back());
      }

      if (ArgIndices.size() == 1 && ArgIndices.begin()->second.empty())
        ++NumArgumentsPromoted;
      else
        ++NumAggregatesPromoted;
    }
  }

  // Add any function attributes.
  if (PAL.hasAttributes(AttributeSet::FunctionIndex))
    AttributesVec.push_back(AttributeSet::get(FTy->getContext(),
                                              PAL.getFnAttributes()));

  Type *RetTy = FTy->getReturnType();

  // Construct the new function type using the new arguments.
  FunctionType *NFTy = FunctionType::get(RetTy, Params, FTy->isVarArg());

  // Create the new function body and insert it into the module.
  Function *NF = Function::Create(NFTy, F->getLinkage(), F->getName());
  NF->copyAttributesFrom(F);

  // Patch the pointer to LLVM function in debug info descriptor.
  auto DI = FunctionDIs.find(F);
  if (DI != FunctionDIs.end()) {
    DISubprogram *SP = DI->second;
    SP->replaceFunction(NF);
    // Ensure the map is updated so it can be reused on subsequent argument
    // promotions of the same function.
    FunctionDIs.erase(DI);
    FunctionDIs[NF] = SP;
  }

  DEBUG(dbgs() << "ARG PROMOTION:  Promoting to:" << *NF << "\n"
        << "From: " << *F);
  
  // Recompute the parameter attributes list based on the new arguments for
  // the function.
  NF->setAttributes(AttributeSet::get(F->getContext(), AttributesVec));
  AttributesVec.clear();

  F->getParent()->getFunctionList().insert(F, NF);
  NF->takeName(F);

  // Get the alias analysis information that we need to update to reflect our
  // changes.
  AliasAnalysis &AA = getAnalysis<AliasAnalysis>();

  // Get the callgraph information that we need to update to reflect our
  // changes.
  CallGraph &CG = getAnalysis<CallGraphWrapperPass>().getCallGraph();

  // Get a new callgraph node for NF.
  CallGraphNode *NF_CGN = CG.getOrInsertFunction(NF);

  // Loop over all of the callers of the function, transforming the call sites
  // to pass in the loaded pointers.
  //
  SmallVector<Value*, 16> Args;
  while (!F->use_empty()) {
    CallSite CS(F->user_back());
    assert(CS.getCalledFunction() == F);
    Instruction *Call = CS.getInstruction();
    const AttributeSet &CallPAL = CS.getAttributes();

    // Add any return attributes.
    if (CallPAL.hasAttributes(AttributeSet::ReturnIndex))
      AttributesVec.push_back(AttributeSet::get(F->getContext(),
                                                CallPAL.getRetAttributes()));

    // Loop over the operands, inserting GEP and loads in the caller as
    // appropriate.
    CallSite::arg_iterator AI = CS.arg_begin();
    ArgIndex = 1;
    for (Function::arg_iterator I = F->arg_begin(), E = F->arg_end();
         I != E; ++I, ++AI, ++ArgIndex)
      if (!ArgsToPromote.count(I) && !ByValArgsToTransform.count(I)) {
        Args.push_back(*AI);          // Unmodified argument

        if (CallPAL.hasAttributes(ArgIndex)) {
          AttrBuilder B(CallPAL, ArgIndex);
          AttributesVec.
            push_back(AttributeSet::get(F->getContext(), Args.size(), B));
        }
      } else if (ByValArgsToTransform.count(I)) {
        // Emit a GEP and load for each element of the struct.
        Type *AgTy = cast<PointerType>(I->getType())->getElementType();
        StructType *STy = cast<StructType>(AgTy);
        Value *Idxs[2] = {
              ConstantInt::get(Type::getInt32Ty(F->getContext()), 0), nullptr };
        for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i) {
          Idxs[1] = ConstantInt::get(Type::getInt32Ty(F->getContext()), i);
          Value *Idx = GetElementPtrInst::Create(
              STy, *AI, Idxs, (*AI)->getName() + "." + Twine(i), Call);
          // TODO: Tell AA about the new values?
          Args.push_back(new LoadInst(Idx, Idx->getName()+".val", Call));
        }
      } else if (!I->use_empty()) {
        // Non-dead argument: insert GEPs and loads as appropriate.
        ScalarizeTable &ArgIndices = ScalarizedElements[I];
        // Store the Value* version of the indices in here, but declare it now
        // for reuse.
        std::vector<Value*> Ops;
        for (ScalarizeTable::iterator SI = ArgIndices.begin(),
               E = ArgIndices.end(); SI != E; ++SI) {
          Value *V = *AI;
          LoadInst *OrigLoad = OriginalLoads[std::make_pair(I, SI->second)];
          if (!SI->second.empty()) {
            Ops.reserve(SI->second.size());
            Type *ElTy = V->getType();
            for (IndicesVector::const_iterator II = SI->second.begin(),
                                               IE = SI->second.end();
                 II != IE; ++II) {
              // Use i32 to index structs, and i64 for others (pointers/arrays).
              // This satisfies GEP constraints.
              Type *IdxTy = (ElTy->isStructTy() ?
                    Type::getInt32Ty(F->getContext()) : 
                    Type::getInt64Ty(F->getContext()));
              Ops.push_back(ConstantInt::get(IdxTy, *II));
              // Keep track of the type we're currently indexing.
              ElTy = cast<CompositeType>(ElTy)->getTypeAtIndex(*II);
            }
            // And create a GEP to extract those indices.
            V = GetElementPtrInst::Create(SI->first, V, Ops,
                                          V->getName() + ".idx", Call);
            Ops.clear();
          }
          // Since we're replacing a load make sure we take the alignment
          // of the previous load.
          LoadInst *newLoad = new LoadInst(V, V->getName()+".val", Call);
          newLoad->setAlignment(OrigLoad->getAlignment());
          // Transfer the AA info too.
          AAMDNodes AAInfo;
          OrigLoad->getAAMetadata(AAInfo);
          newLoad->setAAMetadata(AAInfo);

          Args.push_back(newLoad);
        }
      }

    // Push any varargs arguments on the list.
    for (; AI != CS.arg_end(); ++AI, ++ArgIndex) {
      Args.push_back(*AI);
      if (CallPAL.hasAttributes(ArgIndex)) {
        AttrBuilder B(CallPAL, ArgIndex);
        AttributesVec.
          push_back(AttributeSet::get(F->getContext(), Args.size(), B));
      }
    }

    // Add any function attributes.
    if (CallPAL.hasAttributes(AttributeSet::FunctionIndex))
      AttributesVec.push_back(AttributeSet::get(Call->getContext(),
                                                CallPAL.getFnAttributes()));

    Instruction *New;
    if (InvokeInst *II = dyn_cast<InvokeInst>(Call)) {
      New = InvokeInst::Create(NF, II->getNormalDest(), II->getUnwindDest(),
                               Args, "", Call);
      cast<InvokeInst>(New)->setCallingConv(CS.getCallingConv());
      cast<InvokeInst>(New)->setAttributes(AttributeSet::get(II->getContext(),
                                                            AttributesVec));
    } else {
      New = CallInst::Create(NF, Args, "", Call);
      cast<CallInst>(New)->setCallingConv(CS.getCallingConv());
      cast<CallInst>(New)->setAttributes(AttributeSet::get(New->getContext(),
                                                          AttributesVec));
      if (cast<CallInst>(Call)->isTailCall())
        cast<CallInst>(New)->setTailCall();
    }
    New->setDebugLoc(Call->getDebugLoc());
    Args.clear();
    AttributesVec.clear();

    // Update the alias analysis implementation to know that we are replacing
    // the old call with a new one.
    AA.replaceWithNewValue(Call, New);

    // Update the callgraph to know that the callsite has been transformed.
    CallGraphNode *CalleeNode = CG[Call->getParent()->getParent()];
    CalleeNode->replaceCallEdge(CS, CallSite(New), NF_CGN);

    if (!Call->use_empty()) {
      Call->replaceAllUsesWith(New);
      New->takeName(Call);
    }

    // Finally, remove the old call from the program, reducing the use-count of
    // F.
    Call->eraseFromParent();
  }

  // Since we have now created the new function, splice the body of the old
  // function right into the new function, leaving the old rotting hulk of the
  // function empty.
  NF->getBasicBlockList().splice(NF->begin(), F->getBasicBlockList());

  // Loop over the argument list, transferring uses of the old arguments over to
  // the new arguments, also transferring over the names as well.
  //
  for (Function::arg_iterator I = F->arg_begin(), E = F->arg_end(),
       I2 = NF->arg_begin(); I != E; ++I) {
    if (!ArgsToPromote.count(I) && !ByValArgsToTransform.count(I)) {
      // If this is an unmodified argument, move the name and users over to the
      // new version.
      I->replaceAllUsesWith(I2);
      I2->takeName(I);
      AA.replaceWithNewValue(I, I2);
      ++I2;
      continue;
    }

    if (ByValArgsToTransform.count(I)) {
      // In the callee, we create an alloca, and store each of the new incoming
      // arguments into the alloca.
      Instruction *InsertPt = NF->begin()->begin();

      // Just add all the struct element types.
      Type *AgTy = cast<PointerType>(I->getType())->getElementType();
      Value *TheAlloca = new AllocaInst(AgTy, nullptr, "", InsertPt);
      StructType *STy = cast<StructType>(AgTy);
      Value *Idxs[2] = {
            ConstantInt::get(Type::getInt32Ty(F->getContext()), 0), nullptr };

      for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i) {
        Idxs[1] = ConstantInt::get(Type::getInt32Ty(F->getContext()), i);
        Value *Idx = GetElementPtrInst::Create(
            AgTy, TheAlloca, Idxs, TheAlloca->getName() + "." + Twine(i),
            InsertPt);
        I2->setName(I->getName()+"."+Twine(i));
        new StoreInst(I2++, Idx, InsertPt);
      }

      // Anything that used the arg should now use the alloca.
      I->replaceAllUsesWith(TheAlloca);
      TheAlloca->takeName(I);
      AA.replaceWithNewValue(I, TheAlloca);

      // If the alloca is used in a call, we must clear the tail flag since
      // the callee now uses an alloca from the caller.
      for (User *U : TheAlloca->users()) {
        CallInst *Call = dyn_cast<CallInst>(U);
        if (!Call)
          continue;
        Call->setTailCall(false);
      }
      continue;
    }

    if (I->use_empty()) {
      AA.deleteValue(I);
      continue;
    }

    // Otherwise, if we promoted this argument, then all users are load
    // instructions (or GEPs with only load users), and all loads should be
    // using the new argument that we added.
    ScalarizeTable &ArgIndices = ScalarizedElements[I];

    while (!I->use_empty()) {
      if (LoadInst *LI = dyn_cast<LoadInst>(I->user_back())) {
        assert(ArgIndices.begin()->second.empty() &&
               "Load element should sort to front!");
        I2->setName(I->getName()+".val");
        LI->replaceAllUsesWith(I2);
        AA.replaceWithNewValue(LI, I2);
        LI->eraseFromParent();
        DEBUG(dbgs() << "*** Promoted load of argument '" << I->getName()
              << "' in function '" << F->getName() << "'\n");
      } else {
        GetElementPtrInst *GEP = cast<GetElementPtrInst>(I->user_back());
        IndicesVector Operands;
        Operands.reserve(GEP->getNumIndices());
        for (User::op_iterator II = GEP->idx_begin(), IE = GEP->idx_end();
             II != IE; ++II)
          Operands.push_back(cast<ConstantInt>(*II)->getSExtValue());

        // GEPs with a single 0 index can be merged with direct loads
        if (Operands.size() == 1 && Operands.front() == 0)
          Operands.clear();

        Function::arg_iterator TheArg = I2;
        for (ScalarizeTable::iterator It = ArgIndices.begin();
             It->second != Operands; ++It, ++TheArg) {
          assert(It != ArgIndices.end() && "GEP not handled??");
        }

        std::string NewName = I->getName();
        for (unsigned i = 0, e = Operands.size(); i != e; ++i) {
            NewName += "." + utostr(Operands[i]);
        }
        NewName += ".val";
        TheArg->setName(NewName);

        DEBUG(dbgs() << "*** Promoted agg argument '" << TheArg->getName()
              << "' of function '" << NF->getName() << "'\n");

        // All of the uses must be load instructions.  Replace them all with
        // the argument specified by ArgNo.
        while (!GEP->use_empty()) {
          LoadInst *L = cast<LoadInst>(GEP->user_back());
          L->replaceAllUsesWith(TheArg);
          AA.replaceWithNewValue(L, TheArg);
          L->eraseFromParent();
        }
        AA.deleteValue(GEP);
        GEP->eraseFromParent();
      }
    }

    // Increment I2 past all of the arguments added for this promoted pointer.
    std::advance(I2, ArgIndices.size());
  }

  // Tell the alias analysis that the old function is about to disappear.
  AA.replaceWithNewValue(F, NF);

  
  NF_CGN->stealCalledFunctionsFrom(CG[F]);
  
  // Now that the old function is dead, delete it.  If there is a dangling
  // reference to the CallgraphNode, just leave the dead function around for
  // someone else to nuke.
  CallGraphNode *CGN = CG[F];
  if (CGN->getNumReferences() == 0)
    delete CG.removeFunctionFromModule(CGN);
  else
    F->setLinkage(Function::ExternalLinkage);
  
  return NF_CGN;
}

bool ArgPromotion::doInitialization(CallGraph &CG) {
  FunctionDIs = makeSubprogramMap(CG.getModule());
  return CallGraphSCCPass::doInitialization(CG);
}
