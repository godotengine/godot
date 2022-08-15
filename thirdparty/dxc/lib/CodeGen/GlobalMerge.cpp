//===-- GlobalMerge.cpp - Internal globals merging  -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This pass merges globals with internal linkage into one. This way all the
// globals which were merged into a biggest one can be addressed using offsets
// from the same base pointer (no need for separate base pointer for each of the
// global). Such a transformation can significantly reduce the register pressure
// when many globals are involved.
//
// For example, consider the code which touches several global variables at
// once:
//
// static int foo[N], bar[N], baz[N];
//
// for (i = 0; i < N; ++i) {
//    foo[i] = bar[i] * baz[i];
// }
//
//  On ARM the addresses of 3 arrays should be kept in the registers, thus
//  this code has quite large register pressure (loop body):
//
//  ldr     r1, [r5], #4
//  ldr     r2, [r6], #4
//  mul     r1, r2, r1
//  str     r1, [r0], #4
//
//  Pass converts the code to something like:
//
//  static struct {
//    int foo[N];
//    int bar[N];
//    int baz[N];
//  } merged;
//
//  for (i = 0; i < N; ++i) {
//    merged.foo[i] = merged.bar[i] * merged.baz[i];
//  }
//
//  and in ARM code this becomes:
//
//  ldr     r0, [r5, #40]
//  ldr     r1, [r5, #80]
//  mul     r0, r1, r0
//  str     r0, [r5], #4
//
//  note that we saved 2 registers here almostly "for free".
//
// However, merging globals can have tradeoffs:
// - it confuses debuggers, tools, and users
// - it makes linker optimizations less useful (order files, LOHs, ...)
// - it forces usage of indexed addressing (which isn't necessarily "free")
// - it can increase register pressure when the uses are disparate enough.
// 
// We use heuristics to discover the best global grouping we can (cf cl::opts).
// ===---------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include <algorithm>
using namespace llvm;

#define DEBUG_TYPE "global-merge"

// FIXME: This is only useful as a last-resort way to disable the pass.
static cl::opt<bool>
EnableGlobalMerge("enable-global-merge", cl::Hidden,
                  cl::desc("Enable the global merge pass"),
                  cl::init(true));

static cl::opt<bool> GlobalMergeGroupByUse(
    "global-merge-group-by-use", cl::Hidden,
    cl::desc("Improve global merge pass to look at uses"), cl::init(true));

static cl::opt<bool> GlobalMergeIgnoreSingleUse(
    "global-merge-ignore-single-use", cl::Hidden,
    cl::desc("Improve global merge pass to ignore globals only used alone"),
    cl::init(true));

static cl::opt<bool>
EnableGlobalMergeOnConst("global-merge-on-const", cl::Hidden,
                         cl::desc("Enable global merge pass on constants"),
                         cl::init(false));

// FIXME: this could be a transitional option, and we probably need to remove
// it if only we are sure this optimization could always benefit all targets.
static cl::opt<bool>
EnableGlobalMergeOnExternal("global-merge-on-external", cl::Hidden,
     cl::desc("Enable global merge pass on external linkage"),
     cl::init(false));

STATISTIC(NumMerged, "Number of globals merged");
namespace {
  class GlobalMerge : public FunctionPass {
    const TargetMachine *TM;
    // FIXME: Infer the maximum possible offset depending on the actual users
    // (these max offsets are different for the users inside Thumb or ARM
    // functions), see the code that passes in the offset in the ARM backend
    // for more information.
    unsigned MaxOffset;

    /// Whether we should try to optimize for size only.
    /// Currently, this applies a dead simple heuristic: only consider globals
    /// used in minsize functions for merging.
    /// FIXME: This could learn about optsize, and be used in the cost model.
    bool OnlyOptimizeForSize;

    bool doMerge(SmallVectorImpl<GlobalVariable*> &Globals,
                 Module &M, bool isConst, unsigned AddrSpace) const;
    /// \brief Merge everything in \p Globals for which the corresponding bit
    /// in \p GlobalSet is set.
    bool doMerge(SmallVectorImpl<GlobalVariable *> &Globals,
                 const BitVector &GlobalSet, Module &M, bool isConst,
                 unsigned AddrSpace) const;

    /// \brief Check if the given variable has been identified as must keep
    /// \pre setMustKeepGlobalVariables must have been called on the Module that
    ///      contains GV
    bool isMustKeepGlobalVariable(const GlobalVariable *GV) const {
      return MustKeepGlobalVariables.count(GV);
    }

    /// Collect every variables marked as "used" or used in a landing pad
    /// instruction for this Module.
    void setMustKeepGlobalVariables(Module &M);

    /// Collect every variables marked as "used"
    void collectUsedGlobalVariables(Module &M);

    /// Keep track of the GlobalVariable that must not be merged away
    SmallPtrSet<const GlobalVariable *, 16> MustKeepGlobalVariables;

  public:
    static char ID;             // Pass identification, replacement for typeid.
    explicit GlobalMerge(const TargetMachine *TM = nullptr,
                         unsigned MaximalOffset = 0,
                         bool OnlyOptimizeForSize = false)
        : FunctionPass(ID), TM(TM), MaxOffset(MaximalOffset),
          OnlyOptimizeForSize(OnlyOptimizeForSize) {
      initializeGlobalMergePass(*PassRegistry::getPassRegistry());
    }

    bool doInitialization(Module &M) override;
    bool runOnFunction(Function &F) override;
    bool doFinalization(Module &M) override;

    StringRef getPassName() const override {
      return "Merge internal globals";
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesCFG();
      FunctionPass::getAnalysisUsage(AU);
    }
  };
} // end anonymous namespace

char GlobalMerge::ID = 0;
INITIALIZE_PASS_BEGIN(GlobalMerge, "global-merge", "Merge global variables",
                      false, false)
INITIALIZE_PASS_END(GlobalMerge, "global-merge", "Merge global variables",
                    false, false)

bool GlobalMerge::doMerge(SmallVectorImpl<GlobalVariable*> &Globals,
                          Module &M, bool isConst, unsigned AddrSpace) const {
  auto &DL = M.getDataLayout();
  // FIXME: Find better heuristics
  std::stable_sort(
      Globals.begin(), Globals.end(),
      [&DL](const GlobalVariable *GV1, const GlobalVariable *GV2) {
        Type *Ty1 = cast<PointerType>(GV1->getType())->getElementType();
        Type *Ty2 = cast<PointerType>(GV2->getType())->getElementType();

        return (DL.getTypeAllocSize(Ty1) < DL.getTypeAllocSize(Ty2));
      });

  // If we want to just blindly group all globals together, do so.
  if (!GlobalMergeGroupByUse) {
    BitVector AllGlobals(Globals.size());
    AllGlobals.set();
    return doMerge(Globals, AllGlobals, M, isConst, AddrSpace);
  }

  // If we want to be smarter, look at all uses of each global, to try to
  // discover all sets of globals used together, and how many times each of
  // these sets occured.
  //
  // Keep this reasonably efficient, by having an append-only list of all sets
  // discovered so far (UsedGlobalSet), and mapping each "together-ness" unit of
  // code (currently, a Function) to the set of globals seen so far that are
  // used together in that unit (GlobalUsesByFunction).
  //
  // When we look at the Nth global, we now that any new set is either:
  // - the singleton set {N}, containing this global only, or
  // - the union of {N} and a previously-discovered set, containing some
  //   combination of the previous N-1 globals.
  // Using that knowledge, when looking at the Nth global, we can keep:
  // - a reference to the singleton set {N} (CurGVOnlySetIdx)
  // - a list mapping each previous set to its union with {N} (EncounteredUGS),
  //   if it actually occurs.

  // We keep track of the sets of globals used together "close enough".
  struct UsedGlobalSet {
    UsedGlobalSet(size_t Size) : Globals(Size), UsageCount(1) {}
    BitVector Globals;
    unsigned UsageCount;
  };

  // Each set is unique in UsedGlobalSets.
  std::vector<UsedGlobalSet> UsedGlobalSets;

  // Avoid repeating the create-global-set pattern.
  auto CreateGlobalSet = [&]() -> UsedGlobalSet & {
    UsedGlobalSets.emplace_back(Globals.size());
    return UsedGlobalSets.back();
  };

  // The first set is the empty set.
  CreateGlobalSet().UsageCount = 0;

  // We define "close enough" to be "in the same function".
  // FIXME: Grouping uses by function is way too aggressive, so we should have
  // a better metric for distance between uses.
  // The obvious alternative would be to group by BasicBlock, but that's in
  // turn too conservative..
  // Anything in between wouldn't be trivial to compute, so just stick with
  // per-function grouping.

  // The value type is an index into UsedGlobalSets.
  // The default (0) conveniently points to the empty set.
  DenseMap<Function *, size_t /*UsedGlobalSetIdx*/> GlobalUsesByFunction;

  // Now, look at each merge-eligible global in turn.

  // Keep track of the sets we already encountered to which we added the
  // current global.
  // Each element matches the same-index element in UsedGlobalSets.
  // This lets us efficiently tell whether a set has already been expanded to
  // include the current global.
  std::vector<size_t> EncounteredUGS;

  for (size_t GI = 0, GE = Globals.size(); GI != GE; ++GI) {
    GlobalVariable *GV = Globals[GI];

    // Reset the encountered sets for this global...
    std::fill(EncounteredUGS.begin(), EncounteredUGS.end(), 0);
    // ...and grow it in case we created new sets for the previous global.
    EncounteredUGS.resize(UsedGlobalSets.size());

    // We might need to create a set that only consists of the current global.
    // Keep track of its index into UsedGlobalSets.
    size_t CurGVOnlySetIdx = 0;

    // For each global, look at all its Uses.
    for (auto &U : GV->uses()) {
      // This Use might be a ConstantExpr.  We're interested in Instruction
      // users, so look through ConstantExpr...
      Use *UI, *UE;
      if (ConstantExpr *CE = dyn_cast<ConstantExpr>(U.getUser())) {
        if (CE->use_empty())
          continue;
        UI = &*CE->use_begin();
        UE = nullptr;
      } else if (isa<Instruction>(U.getUser())) {
        UI = &U;
        UE = UI->getNext();
      } else {
        continue;
      }

      // ...to iterate on all the instruction users of the global.
      // Note that we iterate on Uses and not on Users to be able to getNext().
      for (; UI != UE; UI = UI->getNext()) {
        Instruction *I = dyn_cast<Instruction>(UI->getUser());
        if (!I)
          continue;

        Function *ParentFn = I->getParent()->getParent();

        // If we're only optimizing for size, ignore non-minsize functions.
        if (OnlyOptimizeForSize &&
            !ParentFn->hasFnAttribute(Attribute::MinSize))
          continue;

        size_t UGSIdx = GlobalUsesByFunction[ParentFn];

        // If this is the first global the basic block uses, map it to the set
        // consisting of this global only.
        if (!UGSIdx) {
          // If that set doesn't exist yet, create it.
          if (!CurGVOnlySetIdx) {
            CurGVOnlySetIdx = UsedGlobalSets.size();
            CreateGlobalSet().Globals.set(GI);
          } else {
            ++UsedGlobalSets[CurGVOnlySetIdx].UsageCount;
          }

          GlobalUsesByFunction[ParentFn] = CurGVOnlySetIdx;
          continue;
        }

        // If we already encountered this BB, just increment the counter.
        if (UsedGlobalSets[UGSIdx].Globals.test(GI)) {
          ++UsedGlobalSets[UGSIdx].UsageCount;
          continue;
        }

        // If not, the previous set wasn't actually used in this function.
        --UsedGlobalSets[UGSIdx].UsageCount;

        // If we already expanded the previous set to include this global, just
        // reuse that expanded set.
        if (size_t ExpandedIdx = EncounteredUGS[UGSIdx]) {
          ++UsedGlobalSets[ExpandedIdx].UsageCount;
          GlobalUsesByFunction[ParentFn] = ExpandedIdx;
          continue;
        }

        // If not, create a new set consisting of the union of the previous set
        // and this global.  Mark it as encountered, so we can reuse it later.
        GlobalUsesByFunction[ParentFn] = EncounteredUGS[UGSIdx] =
            UsedGlobalSets.size();

        UsedGlobalSet &NewUGS = CreateGlobalSet();
        NewUGS.Globals.set(GI);
        NewUGS.Globals |= UsedGlobalSets[UGSIdx].Globals;
      }
    }
  }

  // Now we found a bunch of sets of globals used together.  We accumulated
  // the number of times we encountered the sets (i.e., the number of blocks
  // that use that exact set of globals).
  //
  // Multiply that by the size of the set to give us a crude profitability
  // metric.
  std::sort(UsedGlobalSets.begin(), UsedGlobalSets.end(),
            [](const UsedGlobalSet &UGS1, const UsedGlobalSet &UGS2) {
              return UGS1.Globals.count() * UGS1.UsageCount <
                     UGS2.Globals.count() * UGS2.UsageCount;
            });

  // We can choose to merge all globals together, but ignore globals never used
  // with another global.  This catches the obviously non-profitable cases of
  // having a single global, but is aggressive enough for any other case.
  if (GlobalMergeIgnoreSingleUse) {
    BitVector AllGlobals(Globals.size());
    for (size_t i = 0, e = UsedGlobalSets.size(); i != e; ++i) {
      const UsedGlobalSet &UGS = UsedGlobalSets[e - i - 1];
      if (UGS.UsageCount == 0)
        continue;
      if (UGS.Globals.count() > 1)
        AllGlobals |= UGS.Globals;
    }
    return doMerge(Globals, AllGlobals, M, isConst, AddrSpace);
  }

  // Starting from the sets with the best (=biggest) profitability, find a
  // good combination.
  // The ideal (and expensive) solution can only be found by trying all
  // combinations, looking for the one with the best profitability.
  // Don't be smart about it, and just pick the first compatible combination,
  // starting with the sets with the best profitability.
  BitVector PickedGlobals(Globals.size());
  bool Changed = false;

  for (size_t i = 0, e = UsedGlobalSets.size(); i != e; ++i) {
    const UsedGlobalSet &UGS = UsedGlobalSets[e - i - 1];
    if (UGS.UsageCount == 0)
      continue;
    if (PickedGlobals.anyCommon(UGS.Globals))
      continue;
    PickedGlobals |= UGS.Globals;
    // If the set only contains one global, there's no point in merging.
    // Ignore the global for inclusion in other sets though, so keep it in
    // PickedGlobals.
    if (UGS.Globals.count() < 2)
      continue;
    Changed |= doMerge(Globals, UGS.Globals, M, isConst, AddrSpace);
  }

  return Changed;
}

bool GlobalMerge::doMerge(SmallVectorImpl<GlobalVariable *> &Globals,
                          const BitVector &GlobalSet, Module &M, bool isConst,
                          unsigned AddrSpace) const {

  Type *Int32Ty = Type::getInt32Ty(M.getContext());
  auto &DL = M.getDataLayout();

  assert(Globals.size() > 1);

  DEBUG(dbgs() << " Trying to merge set, starts with #"
               << GlobalSet.find_first() << "\n");

  ssize_t i = GlobalSet.find_first();
  while (i != -1) {
    ssize_t j = 0;
    uint64_t MergedSize = 0;
    std::vector<Type*> Tys;
    std::vector<Constant*> Inits;

    bool HasExternal = false;
    GlobalVariable *TheFirstExternal = 0;
    for (j = i; j != -1; j = GlobalSet.find_next(j)) {
      Type *Ty = Globals[j]->getType()->getElementType();
      MergedSize += DL.getTypeAllocSize(Ty);
      if (MergedSize > MaxOffset) {
        break;
      }
      Tys.push_back(Ty);
      Inits.push_back(Globals[j]->getInitializer());

      if (Globals[j]->hasExternalLinkage() && !HasExternal) {
        HasExternal = true;
        TheFirstExternal = Globals[j];
      }
    }

    // If merged variables doesn't have external linkage, we needn't to expose
    // the symbol after merging.
    GlobalValue::LinkageTypes Linkage = HasExternal
                                            ? GlobalValue::ExternalLinkage
                                            : GlobalValue::InternalLinkage;

    StructType *MergedTy = StructType::get(M.getContext(), Tys);
    Constant *MergedInit = ConstantStruct::get(MergedTy, Inits);

    // If merged variables have external linkage, we use symbol name of the
    // first variable merged as the suffix of global symbol name. This would
    // be able to avoid the link-time naming conflict for globalm symbols.
    GlobalVariable *MergedGV = new GlobalVariable(
        M, MergedTy, isConst, Linkage, MergedInit,
        HasExternal ? "_MergedGlobals_" + TheFirstExternal->getName()
                    : "_MergedGlobals",
        nullptr, GlobalVariable::NotThreadLocal, AddrSpace);

    for (ssize_t k = i, idx = 0; k != j; k = GlobalSet.find_next(k)) {
      GlobalValue::LinkageTypes Linkage = Globals[k]->getLinkage();
      std::string Name = Globals[k]->getName();

      Constant *Idx[2] = {
        ConstantInt::get(Int32Ty, 0),
        ConstantInt::get(Int32Ty, idx++)
      };
      Constant *GEP =
          ConstantExpr::getInBoundsGetElementPtr(MergedTy, MergedGV, Idx);
      Globals[k]->replaceAllUsesWith(GEP);
      Globals[k]->eraseFromParent();

      if (Linkage != GlobalValue::InternalLinkage) {
        // Generate a new alias...
        auto *PTy = cast<PointerType>(GEP->getType());
        GlobalAlias::create(PTy, Linkage, Name, GEP, &M);
      }

      NumMerged++;
    }
    i = j;
  }

  return true;
}

void GlobalMerge::collectUsedGlobalVariables(Module &M) {
  // Extract global variables from llvm.used array
  const GlobalVariable *GV = M.getGlobalVariable("llvm.used");
  if (!GV || !GV->hasInitializer()) return;

  // Should be an array of 'i8*'.
  const ConstantArray *InitList = cast<ConstantArray>(GV->getInitializer());

  for (unsigned i = 0, e = InitList->getNumOperands(); i != e; ++i)
    if (const GlobalVariable *G =
        dyn_cast<GlobalVariable>(InitList->getOperand(i)->stripPointerCasts()))
      MustKeepGlobalVariables.insert(G);
}

void GlobalMerge::setMustKeepGlobalVariables(Module &M) {
  collectUsedGlobalVariables(M);

  for (Module::iterator IFn = M.begin(), IEndFn = M.end(); IFn != IEndFn;
       ++IFn) {
    for (Function::iterator IBB = IFn->begin(), IEndBB = IFn->end();
         IBB != IEndBB; ++IBB) {
      // Follow the invoke link to find the landing pad instruction
      const InvokeInst *II = dyn_cast<InvokeInst>(IBB->getTerminator());
      if (!II) continue;

      const LandingPadInst *LPInst = II->getUnwindDest()->getLandingPadInst();
      // Look for globals in the clauses of the landing pad instruction
      for (unsigned Idx = 0, NumClauses = LPInst->getNumClauses();
           Idx != NumClauses; ++Idx)
        if (const GlobalVariable *GV =
            dyn_cast<GlobalVariable>(LPInst->getClause(Idx)
                                     ->stripPointerCasts()))
          MustKeepGlobalVariables.insert(GV);
    }
  }
}

bool GlobalMerge::doInitialization(Module &M) {
  if (!EnableGlobalMerge)
    return false;

  auto &DL = M.getDataLayout();
  DenseMap<unsigned, SmallVector<GlobalVariable*, 16> > Globals, ConstGlobals,
                                                        BSSGlobals;
  bool Changed = false;
  setMustKeepGlobalVariables(M);

  // Grab all non-const globals.
  for (Module::global_iterator I = M.global_begin(),
         E = M.global_end(); I != E; ++I) {
    // Merge is safe for "normal" internal or external globals only
    if (I->isDeclaration() || I->isThreadLocal() || I->hasSection())
      continue;

    if (!(EnableGlobalMergeOnExternal && I->hasExternalLinkage()) &&
        !I->hasInternalLinkage())
      continue;

    PointerType *PT = dyn_cast<PointerType>(I->getType());
    assert(PT && "Global variable is not a pointer!");

    unsigned AddressSpace = PT->getAddressSpace();

    // Ignore fancy-aligned globals for now.
    unsigned Alignment = DL.getPreferredAlignment(I);
    Type *Ty = I->getType()->getElementType();
    if (Alignment > DL.getABITypeAlignment(Ty))
      continue;

    // Ignore all 'special' globals.
    if (I->getName().startswith("llvm.") ||
        I->getName().startswith(".llvm."))
      continue;

    // Ignore all "required" globals:
    if (isMustKeepGlobalVariable(I))
      continue;

    if (DL.getTypeAllocSize(Ty) < MaxOffset) {
      if (TargetLoweringObjectFile::getKindForGlobal(I, *TM).isBSSLocal())
        BSSGlobals[AddressSpace].push_back(I);
      else if (I->isConstant())
        ConstGlobals[AddressSpace].push_back(I);
      else
        Globals[AddressSpace].push_back(I);
    }
  }

  for (DenseMap<unsigned, SmallVector<GlobalVariable*, 16> >::iterator
       I = Globals.begin(), E = Globals.end(); I != E; ++I)
    if (I->second.size() > 1)
      Changed |= doMerge(I->second, M, false, I->first);

  for (DenseMap<unsigned, SmallVector<GlobalVariable*, 16> >::iterator
       I = BSSGlobals.begin(), E = BSSGlobals.end(); I != E; ++I)
    if (I->second.size() > 1)
      Changed |= doMerge(I->second, M, false, I->first);

  if (EnableGlobalMergeOnConst)
    for (DenseMap<unsigned, SmallVector<GlobalVariable*, 16> >::iterator
         I = ConstGlobals.begin(), E = ConstGlobals.end(); I != E; ++I)
      if (I->second.size() > 1)
        Changed |= doMerge(I->second, M, true, I->first);

  return Changed;
}

bool GlobalMerge::runOnFunction(Function &F) {
  return false;
}

bool GlobalMerge::doFinalization(Module &M) {
  MustKeepGlobalVariables.clear();
  return false;
}

Pass *llvm::createGlobalMergePass(const TargetMachine *TM, unsigned Offset,
                                  bool OnlyOptimizeForSize) {
  return new GlobalMerge(TM, Offset, OnlyOptimizeForSize);
}
