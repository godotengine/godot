//===- NaryReassociate.cpp - Reassociate n-ary expressions ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass reassociates n-ary add expressions and eliminates the redundancy
// exposed by the reassociation.
//
// A motivating example:
//
//   void foo(int a, int b) {
//     bar(a + b);
//     bar((a + 2) + b);
//   }
//
// An ideal compiler should reassociate (a + 2) + b to (a + b) + 2 and simplify
// the above code to
//
//   int t = a + b;
//   bar(t);
//   bar(t + 2);
//
// However, the Reassociate pass is unable to do that because it processes each
// instruction individually and believes (a + 2) + b is the best form according
// to its rank system.
//
// To address this limitation, NaryReassociate reassociates an expression in a
// form that reuses existing instructions. As a result, NaryReassociate can
// reassociate (a + 2) + b in the example to (a + b) + 2 because it detects that
// (a + b) is computed before.
//
// NaryReassociate works as follows. For every instruction in the form of (a +
// b) + c, it checks whether a + c or b + c is already computed by a dominating
// instruction. If so, it then reassociates (a + b) + c into (a + c) + b or (b +
// c) + a and removes the redundancy accordingly. To efficiently look up whether
// an expression is computed before, we store each instruction seen and its SCEV
// into an SCEV-to-instruction map.
//
// Although the algorithm pattern-matches only ternary additions, it
// automatically handles many >3-ary expressions by walking through the function
// in the depth-first order. For example, given
//
//   (a + c) + d
//   ((a + b) + c) + d
//
// NaryReassociate first rewrites (a + b) + c to (a + c) + b, and then rewrites
// ((a + c) + b) + d into ((a + c) + d) + b.
//
// Finally, the above dominator-based algorithm may need to be run multiple
// iterations before emitting optimal code. One source of this need is that we
// only split an operand when it is used only once. The above algorithm can
// eliminate an instruction and decrease the usage count of its operands. As a
// result, an instruction that previously had multiple uses may become a
// single-use instruction and thus eligible for split consideration. For
// example,
//
//   ac = a + c
//   ab = a + b
//   abc = ab + c
//   ab2 = ab + b
//   ab2c = ab2 + c
//
// In the first iteration, we cannot reassociate abc to ac+b because ab is used
// twice. However, we can reassociate ab2c to abc+b in the first iteration. As a
// result, ab2 becomes dead and ab will be used only once in the second
// iteration.
//
// Limitations and TODO items:
//
// 1) We only considers n-ary adds for now. This should be extended and
// generalized.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Local.h"
using namespace llvm;
using namespace PatternMatch;

#define DEBUG_TYPE "nary-reassociate"

namespace {
class NaryReassociate : public FunctionPass {
public:
  static char ID;

  NaryReassociate(): FunctionPass(ID) {
    initializeNaryReassociatePass(*PassRegistry::getPassRegistry());
  }

  bool doInitialization(Module &M) override {
    DL = &M.getDataLayout();
    return false;
  }
  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addPreserved<DominatorTreeWrapperPass>();
    AU.addPreserved<ScalarEvolution>();
    AU.addPreserved<TargetLibraryInfoWrapperPass>();
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<ScalarEvolution>();
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.setPreservesCFG();
  }

private:
  // Runs only one iteration of the dominator-based algorithm. See the header
  // comments for why we need multiple iterations.
  bool doOneIteration(Function &F);

  // Reassociates I for better CSE.
  Instruction *tryReassociate(Instruction *I);

  // Reassociate GEP for better CSE.
  Instruction *tryReassociateGEP(GetElementPtrInst *GEP);
  // Try splitting GEP at the I-th index and see whether either part can be
  // CSE'ed. This is a helper function for tryReassociateGEP.
  //
  // \p IndexedType The element type indexed by GEP's I-th index. This is
  //                equivalent to
  //                  GEP->getIndexedType(GEP->getPointerOperand(), 0-th index,
  //                                      ..., i-th index).
  GetElementPtrInst *tryReassociateGEPAtIndex(GetElementPtrInst *GEP,
                                              unsigned I, Type *IndexedType);
  // Given GEP's I-th index = LHS + RHS, see whether &Base[..][LHS][..] or
  // &Base[..][RHS][..] can be CSE'ed and rewrite GEP accordingly.
  GetElementPtrInst *tryReassociateGEPAtIndex(GetElementPtrInst *GEP,
                                              unsigned I, Value *LHS,
                                              Value *RHS, Type *IndexedType);

  // Reassociate Add for better CSE.
  Instruction *tryReassociateAdd(BinaryOperator *I);
  // A helper function for tryReassociateAdd. LHS and RHS are explicitly passed.
  Instruction *tryReassociateAdd(Value *LHS, Value *RHS, Instruction *I);
  // Rewrites I to LHS + RHS if LHS is computed already.
  Instruction *tryReassociatedAdd(const SCEV *LHS, Value *RHS, Instruction *I);

  // Returns the closest dominator of \c Dominatee that computes
  // \c CandidateExpr. Returns null if not found.
  Instruction *findClosestMatchingDominator(const SCEV *CandidateExpr,
                                            Instruction *Dominatee);
  // GetElementPtrInst implicitly sign-extends an index if the index is shorter
  // than the pointer size. This function returns whether Index is shorter than
  // GEP's pointer size, i.e., whether Index needs to be sign-extended in order
  // to be an index of GEP.
  bool requiresSignExtension(Value *Index, GetElementPtrInst *GEP);
  // Returns whether V is known to be non-negative at context \c Ctxt.
  bool isKnownNonNegative(Value *V, Instruction *Ctxt);
  // Returns whether AO may sign overflow at context \c Ctxt. It computes a
  // conservative result -- it answers true when not sure.
  bool maySignOverflow(AddOperator *AO, Instruction *Ctxt);

  AssumptionCache *AC;
  const DataLayout *DL;
  DominatorTree *DT;
  ScalarEvolution *SE;
  TargetLibraryInfo *TLI;
  TargetTransformInfo *TTI;
  // A lookup table quickly telling which instructions compute the given SCEV.
  // Note that there can be multiple instructions at different locations
  // computing to the same SCEV, so we map a SCEV to an instruction list.  For
  // example,
  //
  //   if (p1)
  //     foo(a + b);
  //   if (p2)
  //     bar(a + b);
  DenseMap<const SCEV *, SmallVector<Instruction *, 2>> SeenExprs;
};
} // anonymous namespace

char NaryReassociate::ID = 0;
INITIALIZE_PASS_BEGIN(NaryReassociate, "nary-reassociate", "Nary reassociation",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolution)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_END(NaryReassociate, "nary-reassociate", "Nary reassociation",
                    false, false)

FunctionPass *llvm::createNaryReassociatePass() {
  return new NaryReassociate();
}

bool NaryReassociate::runOnFunction(Function &F) {
  if (skipOptnoneFunction(F))
    return false;

  AC = &getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  SE = &getAnalysis<ScalarEvolution>();
  TLI = &getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();
  TTI = &getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);

  bool Changed = false, ChangedInThisIteration;
  do {
    ChangedInThisIteration = doOneIteration(F);
    Changed |= ChangedInThisIteration;
  } while (ChangedInThisIteration);
  return Changed;
}

// Whitelist the instruction types NaryReassociate handles for now.
static bool isPotentiallyNaryReassociable(Instruction *I) {
  switch (I->getOpcode()) {
  case Instruction::Add:
  case Instruction::GetElementPtr:
    return true;
  default:
    return false;
  }
}

bool NaryReassociate::doOneIteration(Function &F) {
  bool Changed = false;
  SeenExprs.clear();
  // Process the basic blocks in pre-order of the dominator tree. This order
  // ensures that all bases of a candidate are in Candidates when we process it.
  for (auto Node = GraphTraits<DominatorTree *>::nodes_begin(DT);
       Node != GraphTraits<DominatorTree *>::nodes_end(DT); ++Node) {
    BasicBlock *BB = Node->getBlock();
    for (auto I = BB->begin(); I != BB->end(); ++I) {
      if (SE->isSCEVable(I->getType()) && isPotentiallyNaryReassociable(I)) {
        const SCEV *OldSCEV = SE->getSCEV(I);
        if (Instruction *NewI = tryReassociate(I)) {
          Changed = true;
          SE->forgetValue(I);
          I->replaceAllUsesWith(NewI);
          RecursivelyDeleteTriviallyDeadInstructions(I, TLI);
          I = NewI;
        }
        // Add the rewritten instruction to SeenExprs; the original instruction
        // is deleted.
        const SCEV *NewSCEV = SE->getSCEV(I);
        SeenExprs[NewSCEV].push_back(I);
        // Ideally, NewSCEV should equal OldSCEV because tryReassociate(I)
        // is equivalent to I. However, ScalarEvolution::getSCEV may
        // weaken nsw causing NewSCEV not to equal OldSCEV. For example, suppose
        // we reassociate
        //   I = &a[sext(i +nsw j)] // assuming sizeof(a[0]) = 4
        // to
        //   NewI = &a[sext(i)] + sext(j).
        //
        // ScalarEvolution computes
        //   getSCEV(I)    = a + 4 * sext(i + j)
        //   getSCEV(newI) = a + 4 * sext(i) + 4 * sext(j)
        // which are different SCEVs.
        //
        // To alleviate this issue of ScalarEvolution not always capturing
        // equivalence, we add I to SeenExprs[OldSCEV] as well so that we can
        // map both SCEV before and after tryReassociate(I) to I.
        //
        // This improvement is exercised in @reassociate_gep_nsw in nary-gep.ll.
        if (NewSCEV != OldSCEV)
          SeenExprs[OldSCEV].push_back(I);
      }
    }
  }
  return Changed;
}

Instruction *NaryReassociate::tryReassociate(Instruction *I) {
  switch (I->getOpcode()) {
  case Instruction::Add:
    return tryReassociateAdd(cast<BinaryOperator>(I));
  case Instruction::GetElementPtr:
    return tryReassociateGEP(cast<GetElementPtrInst>(I));
  default:
    llvm_unreachable("should be filtered out by isPotentiallyNaryReassociable");
  }
}

// FIXME: extract this method into TTI->getGEPCost.
static bool isGEPFoldable(GetElementPtrInst *GEP,
                          const TargetTransformInfo *TTI,
                          const DataLayout *DL) {
  GlobalVariable *BaseGV = nullptr;
  int64_t BaseOffset = 0;
  bool HasBaseReg = false;
  int64_t Scale = 0;

  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(GEP->getPointerOperand()))
    BaseGV = GV;
  else
    HasBaseReg = true;

  gep_type_iterator GTI = gep_type_begin(GEP);
  for (auto I = GEP->idx_begin(); I != GEP->idx_end(); ++I, ++GTI) {
    if (isa<SequentialType>(*GTI)) {
      int64_t ElementSize = DL->getTypeAllocSize(GTI.getIndexedType());
      if (ConstantInt *ConstIdx = dyn_cast<ConstantInt>(*I)) {
        BaseOffset += ConstIdx->getSExtValue() * ElementSize;
      } else {
        // Needs scale register.
        if (Scale != 0) {
          // No addressing mode takes two scale registers.
          return false;
        }
        Scale = ElementSize;
      }
    } else {
      StructType *STy = cast<StructType>(*GTI);
      uint64_t Field = cast<ConstantInt>(*I)->getZExtValue();
      BaseOffset += DL->getStructLayout(STy)->getElementOffset(Field);
    }
  }

  unsigned AddrSpace = GEP->getPointerAddressSpace();
  return TTI->isLegalAddressingMode(GEP->getType()->getElementType(), BaseGV,
                                    BaseOffset, HasBaseReg, Scale, AddrSpace);
}

Instruction *NaryReassociate::tryReassociateGEP(GetElementPtrInst *GEP) {
  // Not worth reassociating GEP if it is foldable.
  if (isGEPFoldable(GEP, TTI, DL))
    return nullptr;

  gep_type_iterator GTI = gep_type_begin(*GEP);
  for (unsigned I = 1, E = GEP->getNumOperands(); I != E; ++I) {
    if (isa<SequentialType>(*GTI++)) {
      if (auto *NewGEP = tryReassociateGEPAtIndex(GEP, I - 1, *GTI)) {
        return NewGEP;
      }
    }
  }
  return nullptr;
}

bool NaryReassociate::requiresSignExtension(Value *Index,
                                            GetElementPtrInst *GEP) {
  unsigned PointerSizeInBits =
      DL->getPointerSizeInBits(GEP->getType()->getPointerAddressSpace());
  return cast<IntegerType>(Index->getType())->getBitWidth() < PointerSizeInBits;
}

bool NaryReassociate::isKnownNonNegative(Value *V, Instruction *Ctxt) {
  bool NonNegative, Negative;
  // TODO: ComputeSignBits is expensive. Consider caching the results.
  ComputeSignBit(V, NonNegative, Negative, *DL, 0, AC, Ctxt, DT);
  return NonNegative;
}

bool NaryReassociate::maySignOverflow(AddOperator *AO, Instruction *Ctxt) {
  if (AO->hasNoSignedWrap())
    return false;

  Value *LHS = AO->getOperand(0), *RHS = AO->getOperand(1);
  // If LHS or RHS has the same sign as the sum, AO doesn't sign overflow.
  // TODO: handle the negative case as well.
  if (isKnownNonNegative(AO, Ctxt) &&
      (isKnownNonNegative(LHS, Ctxt) || isKnownNonNegative(RHS, Ctxt)))
    return false;

  return true;
}

GetElementPtrInst *
NaryReassociate::tryReassociateGEPAtIndex(GetElementPtrInst *GEP, unsigned I,
                                          Type *IndexedType) {
  Value *IndexToSplit = GEP->getOperand(I + 1);
  if (SExtInst *SExt = dyn_cast<SExtInst>(IndexToSplit)) {
    IndexToSplit = SExt->getOperand(0);
  } else if (ZExtInst *ZExt = dyn_cast<ZExtInst>(IndexToSplit)) {
    // zext can be treated as sext if the source is non-negative.
    if (isKnownNonNegative(ZExt->getOperand(0), GEP))
      IndexToSplit = ZExt->getOperand(0);
  }

  if (AddOperator *AO = dyn_cast<AddOperator>(IndexToSplit)) {
    // If the I-th index needs sext and the underlying add is not equipped with
    // nsw, we cannot split the add because
    //   sext(LHS + RHS) != sext(LHS) + sext(RHS).
    if (requiresSignExtension(IndexToSplit, GEP) && maySignOverflow(AO, GEP))
      return nullptr;
    Value *LHS = AO->getOperand(0), *RHS = AO->getOperand(1);
    // IndexToSplit = LHS + RHS.
    if (auto *NewGEP = tryReassociateGEPAtIndex(GEP, I, LHS, RHS, IndexedType))
      return NewGEP;
    // Symmetrically, try IndexToSplit = RHS + LHS.
    if (LHS != RHS) {
      if (auto *NewGEP =
              tryReassociateGEPAtIndex(GEP, I, RHS, LHS, IndexedType))
        return NewGEP;
    }
  }
  return nullptr;
}

GetElementPtrInst *NaryReassociate::tryReassociateGEPAtIndex(
    GetElementPtrInst *GEP, unsigned I, Value *LHS, Value *RHS,
    Type *IndexedType) {
  // Look for GEP's closest dominator that has the same SCEV as GEP except that
  // the I-th index is replaced with LHS.
  SmallVector<const SCEV *, 4> IndexExprs;
  for (auto Index = GEP->idx_begin(); Index != GEP->idx_end(); ++Index)
    IndexExprs.push_back(SE->getSCEV(*Index));
  // Replace the I-th index with LHS.
  IndexExprs[I] = SE->getSCEV(LHS);
  if (isKnownNonNegative(LHS, GEP) &&
      DL->getTypeSizeInBits(LHS->getType()) <
          DL->getTypeSizeInBits(GEP->getOperand(I)->getType())) {
    // Zero-extend LHS if it is non-negative. InstCombine canonicalizes sext to
    // zext if the source operand is proved non-negative. We should do that
    // consistently so that CandidateExpr more likely appears before. See
    // @reassociate_gep_assume for an example of this canonicalization.
    IndexExprs[I] =
        SE->getZeroExtendExpr(IndexExprs[I], GEP->getOperand(I)->getType());
  }
  const SCEV *CandidateExpr = SE->getGEPExpr(
      GEP->getSourceElementType(), SE->getSCEV(GEP->getPointerOperand()),
      IndexExprs, GEP->isInBounds());

  auto *Candidate = findClosestMatchingDominator(CandidateExpr, GEP);
  if (Candidate == nullptr)
    return nullptr;

  PointerType *TypeOfCandidate = dyn_cast<PointerType>(Candidate->getType());
  // Pretty rare but theoretically possible when a numeric value happens to
  // share CandidateExpr.
  if (TypeOfCandidate == nullptr)
    return nullptr;

  // NewGEP = (char *)Candidate + RHS * sizeof(IndexedType)
  uint64_t IndexedSize = DL->getTypeAllocSize(IndexedType);
  Type *ElementType = TypeOfCandidate->getElementType();
  uint64_t ElementSize = DL->getTypeAllocSize(ElementType);
  // Another less rare case: because I is not necessarily the last index of the
  // GEP, the size of the type at the I-th index (IndexedSize) is not
  // necessarily divisible by ElementSize. For example,
  //
  // #pragma pack(1)
  // struct S {
  //   int a[3];
  //   int64 b[8];
  // };
  // #pragma pack()
  //
  // sizeof(S) = 100 is indivisible by sizeof(int64) = 8.
  //
  // TODO: bail out on this case for now. We could emit uglygep.
  if (IndexedSize % ElementSize != 0)
    return nullptr;

  // NewGEP = &Candidate[RHS * (sizeof(IndexedType) / sizeof(Candidate[0])));
  IRBuilder<> Builder(GEP);
  Type *IntPtrTy = DL->getIntPtrType(TypeOfCandidate);
  if (RHS->getType() != IntPtrTy)
    RHS = Builder.CreateSExtOrTrunc(RHS, IntPtrTy);
  if (IndexedSize != ElementSize) {
    RHS = Builder.CreateMul(
        RHS, ConstantInt::get(IntPtrTy, IndexedSize / ElementSize));
  }
  GetElementPtrInst *NewGEP =
      cast<GetElementPtrInst>(Builder.CreateGEP(Candidate, RHS));
  NewGEP->setIsInBounds(GEP->isInBounds());
  NewGEP->takeName(GEP);
  return NewGEP;
}

Instruction *NaryReassociate::tryReassociateAdd(BinaryOperator *I) {
  Value *LHS = I->getOperand(0), *RHS = I->getOperand(1);
  if (auto *NewI = tryReassociateAdd(LHS, RHS, I))
    return NewI;
  if (auto *NewI = tryReassociateAdd(RHS, LHS, I))
    return NewI;
  return nullptr;
}

Instruction *NaryReassociate::tryReassociateAdd(Value *LHS, Value *RHS,
                                                Instruction *I) {
  Value *A = nullptr, *B = nullptr;
  // To be conservative, we reassociate I only when it is the only user of A+B.
  if (LHS->hasOneUse() && match(LHS, m_Add(m_Value(A), m_Value(B)))) {
    // I = (A + B) + RHS
    //   = (A + RHS) + B or (B + RHS) + A
    const SCEV *AExpr = SE->getSCEV(A), *BExpr = SE->getSCEV(B);
    const SCEV *RHSExpr = SE->getSCEV(RHS);
    if (BExpr != RHSExpr) {
      if (auto *NewI = tryReassociatedAdd(SE->getAddExpr(AExpr, RHSExpr), B, I))
        return NewI;
    }
    if (AExpr != RHSExpr) {
      if (auto *NewI = tryReassociatedAdd(SE->getAddExpr(BExpr, RHSExpr), A, I))
        return NewI;
    }
  }
  return nullptr;
}

Instruction *NaryReassociate::tryReassociatedAdd(const SCEV *LHSExpr,
                                                 Value *RHS, Instruction *I) {
  auto Pos = SeenExprs.find(LHSExpr);
  // Bail out if LHSExpr is not previously seen.
  if (Pos == SeenExprs.end())
    return nullptr;

  // Look for the closest dominator LHS of I that computes LHSExpr, and replace
  // I with LHS + RHS.
  auto *LHS = findClosestMatchingDominator(LHSExpr, I);
  if (LHS == nullptr)
    return nullptr;

  Instruction *NewI = BinaryOperator::CreateAdd(LHS, RHS, "", I);
  NewI->takeName(I);
  return NewI;
}

Instruction *
NaryReassociate::findClosestMatchingDominator(const SCEV *CandidateExpr,
                                              Instruction *Dominatee) {
  auto Pos = SeenExprs.find(CandidateExpr);
  if (Pos == SeenExprs.end())
    return nullptr;

  auto &Candidates = Pos->second;
  // Because we process the basic blocks in pre-order of the dominator tree, a
  // candidate that doesn't dominate the current instruction won't dominate any
  // future instruction either. Therefore, we pop it out of the stack. This
  // optimization makes the algorithm O(n).
  while (!Candidates.empty()) {
    Instruction *Candidate = Candidates.back();
    if (DT->dominates(Candidate, Dominatee))
      return Candidate;
    Candidates.pop_back();
  }
  return nullptr;
}
