//===- CorrelatedValuePropagation.cpp - Propagate CFG-derived info --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Correlated Value Propagation pass.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LazyValueInfo.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Local.h"
using namespace llvm;

#define DEBUG_TYPE "correlated-value-propagation"

STATISTIC(NumPhis,      "Number of phis propagated");
STATISTIC(NumSelects,   "Number of selects propagated");
STATISTIC(NumMemAccess, "Number of memory access targets propagated");
STATISTIC(NumCmps,      "Number of comparisons propagated");
STATISTIC(NumDeadCases, "Number of switch cases removed");

namespace {
  class CorrelatedValuePropagation : public FunctionPass {
    LazyValueInfo *LVI;

    bool processSelect(SelectInst *SI);
    bool processPHI(PHINode *P);
    bool processMemAccess(Instruction *I);
    bool processCmp(CmpInst *C);
    bool processSwitch(SwitchInst *SI);

  public:
    static char ID;
    CorrelatedValuePropagation(): FunctionPass(ID) {
     initializeCorrelatedValuePropagationPass(*PassRegistry::getPassRegistry());
    }

    bool runOnFunction(Function &F) override;

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.addRequired<LazyValueInfo>();
    }
  };
}

char CorrelatedValuePropagation::ID = 0;
INITIALIZE_PASS_BEGIN(CorrelatedValuePropagation, "correlated-propagation",
                "Value Propagation", false, false)
INITIALIZE_PASS_DEPENDENCY(LazyValueInfo)
INITIALIZE_PASS_END(CorrelatedValuePropagation, "correlated-propagation",
                "Value Propagation", false, false)

// Public interface to the Value Propagation pass
Pass *llvm::createCorrelatedValuePropagationPass() {
  return new CorrelatedValuePropagation();
}

bool CorrelatedValuePropagation::processSelect(SelectInst *S) {
  if (S->getType()->isVectorTy()) return false;
  if (isa<Constant>(S->getOperand(0))) return false;

  Constant *C = LVI->getConstant(S->getOperand(0), S->getParent(), S);
  if (!C) return false;

  ConstantInt *CI = dyn_cast<ConstantInt>(C);
  if (!CI) return false;

  Value *ReplaceWith = S->getOperand(1);
  Value *Other = S->getOperand(2);
  if (!CI->isOne()) std::swap(ReplaceWith, Other);
  if (ReplaceWith == S) ReplaceWith = UndefValue::get(S->getType());

  S->replaceAllUsesWith(ReplaceWith);
  S->eraseFromParent();

  ++NumSelects;

  return true;
}

bool CorrelatedValuePropagation::processPHI(PHINode *P) {
  bool Changed = false;

  BasicBlock *BB = P->getParent();
  for (unsigned i = 0, e = P->getNumIncomingValues(); i < e; ++i) {
    Value *Incoming = P->getIncomingValue(i);
    if (isa<Constant>(Incoming)) continue;

    Value *V = LVI->getConstantOnEdge(Incoming, P->getIncomingBlock(i), BB, P);

    // Look if the incoming value is a select with a scalar condition for which
    // LVI can tells us the value. In that case replace the incoming value with
    // the appropriate value of the select. This often allows us to remove the
    // select later.
    if (!V) {
      SelectInst *SI = dyn_cast<SelectInst>(Incoming);
      if (!SI) continue;

      Value *Condition = SI->getCondition();
      if (!Condition->getType()->isVectorTy()) {
        if (Constant *C = LVI->getConstantOnEdge(
                Condition, P->getIncomingBlock(i), BB, P)) {
          if (C->isOneValue()) {
            V = SI->getTrueValue();
          } else if (C->isZeroValue()) {
            V = SI->getFalseValue();
          }
          // Once LVI learns to handle vector types, we could also add support
          // for vector type constants that are not all zeroes or all ones.
        }
      }

      // Look if the select has a constant but LVI tells us that the incoming
      // value can never be that constant. In that case replace the incoming
      // value with the other value of the select. This often allows us to
      // remove the select later.
      if (!V) {
        Constant *C = dyn_cast<Constant>(SI->getFalseValue());
        if (!C) continue;

        if (LVI->getPredicateOnEdge(ICmpInst::ICMP_EQ, SI, C,
              P->getIncomingBlock(i), BB, P) !=
            LazyValueInfo::False)
          continue;
        V = SI->getTrueValue();
      }

      DEBUG(dbgs() << "CVP: Threading PHI over " << *SI << '\n');
    }

    P->setIncomingValue(i, V);
    Changed = true;
  }

  // FIXME: Provide TLI, DT, AT to SimplifyInstruction.
  const DataLayout &DL = BB->getModule()->getDataLayout();
  if (Value *V = SimplifyInstruction(P, DL)) {
    P->replaceAllUsesWith(V);
    P->eraseFromParent();
    Changed = true;
  }

  if (Changed)
    ++NumPhis;

  return Changed;
}

bool CorrelatedValuePropagation::processMemAccess(Instruction *I) {
  Value *Pointer = nullptr;
  if (LoadInst *L = dyn_cast<LoadInst>(I))
    Pointer = L->getPointerOperand();
  else
    Pointer = cast<StoreInst>(I)->getPointerOperand();

  if (isa<Constant>(Pointer)) return false;

  Constant *C = LVI->getConstant(Pointer, I->getParent(), I);
  if (!C) return false;

  ++NumMemAccess;
  I->replaceUsesOfWith(Pointer, C);
  return true;
}

/// processCmp - If the value of this comparison could be determined locally,
/// constant propagation would already have figured it out.  Instead, walk
/// the predecessors and statically evaluate the comparison based on information
/// available on that edge.  If a given static evaluation is true on ALL
/// incoming edges, then it's true universally and we can simplify the compare.
bool CorrelatedValuePropagation::processCmp(CmpInst *C) {
  Value *Op0 = C->getOperand(0);
  if (isa<Instruction>(Op0) &&
      cast<Instruction>(Op0)->getParent() == C->getParent())
    return false;

  Constant *Op1 = dyn_cast<Constant>(C->getOperand(1));
  if (!Op1) return false;

  pred_iterator PI = pred_begin(C->getParent()), PE = pred_end(C->getParent());
  if (PI == PE) return false;

  LazyValueInfo::Tristate Result = LVI->getPredicateOnEdge(C->getPredicate(),
                                    C->getOperand(0), Op1, *PI,
                                    C->getParent(), C);
  if (Result == LazyValueInfo::Unknown) return false;

  ++PI;
  while (PI != PE) {
    LazyValueInfo::Tristate Res = LVI->getPredicateOnEdge(C->getPredicate(),
                                    C->getOperand(0), Op1, *PI,
                                    C->getParent(), C);
    if (Res != Result) return false;
    ++PI;
  }

  ++NumCmps;

  if (Result == LazyValueInfo::True)
    C->replaceAllUsesWith(ConstantInt::getTrue(C->getContext()));
  else
    C->replaceAllUsesWith(ConstantInt::getFalse(C->getContext()));

  C->eraseFromParent();

  return true;
}

/// processSwitch - Simplify a switch instruction by removing cases which can
/// never fire.  If the uselessness of a case could be determined locally then
/// constant propagation would already have figured it out.  Instead, walk the
/// predecessors and statically evaluate cases based on information available
/// on that edge.  Cases that cannot fire no matter what the incoming edge can
/// safely be removed.  If a case fires on every incoming edge then the entire
/// switch can be removed and replaced with a branch to the case destination.
bool CorrelatedValuePropagation::processSwitch(SwitchInst *SI) {
  Value *Cond = SI->getCondition();
  BasicBlock *BB = SI->getParent();

  // If the condition was defined in same block as the switch then LazyValueInfo
  // currently won't say anything useful about it, though in theory it could.
  if (isa<Instruction>(Cond) && cast<Instruction>(Cond)->getParent() == BB)
    return false;

  // If the switch is unreachable then trying to improve it is a waste of time.
  pred_iterator PB = pred_begin(BB), PE = pred_end(BB);
  if (PB == PE) return false;

  // Analyse each switch case in turn.  This is done in reverse order so that
  // removing a case doesn't cause trouble for the iteration.
  bool Changed = false;
  for (SwitchInst::CaseIt CI = SI->case_end(), CE = SI->case_begin(); CI-- != CE;
       ) {
    ConstantInt *Case = CI.getCaseValue();

    // Check to see if the switch condition is equal to/not equal to the case
    // value on every incoming edge, equal/not equal being the same each time.
    LazyValueInfo::Tristate State = LazyValueInfo::Unknown;
    for (pred_iterator PI = PB; PI != PE; ++PI) {
      // Is the switch condition equal to the case value?
      LazyValueInfo::Tristate Value = LVI->getPredicateOnEdge(CmpInst::ICMP_EQ,
                                                              Cond, Case, *PI,
                                                              BB, SI);
      // Give up on this case if nothing is known.
      if (Value == LazyValueInfo::Unknown) {
        State = LazyValueInfo::Unknown;
        break;
      }

      // If this was the first edge to be visited, record that all other edges
      // need to give the same result.
      if (PI == PB) {
        State = Value;
        continue;
      }

      // If this case is known to fire for some edges and known not to fire for
      // others then there is nothing we can do - give up.
      if (Value != State) {
        State = LazyValueInfo::Unknown;
        break;
      }
    }

    if (State == LazyValueInfo::False) {
      // This case never fires - remove it.
      CI.getCaseSuccessor()->removePredecessor(BB);
      SI->removeCase(CI); // Does not invalidate the iterator.

      // The condition can be modified by removePredecessor's PHI simplification
      // logic.
      Cond = SI->getCondition();

      ++NumDeadCases;
      Changed = true;
    } else if (State == LazyValueInfo::True) {
      // This case always fires.  Arrange for the switch to be turned into an
      // unconditional branch by replacing the switch condition with the case
      // value.
      SI->setCondition(Case);
      NumDeadCases += SI->getNumCases();
      Changed = true;
      break;
    }
  }

  if (Changed)
    // If the switch has been simplified to the point where it can be replaced
    // by a branch then do so now.
    ConstantFoldTerminator(BB);

  return Changed;
}

bool CorrelatedValuePropagation::runOnFunction(Function &F) {
  if (skipOptnoneFunction(F))
    return false;

  LVI = &getAnalysis<LazyValueInfo>();

  bool FnChanged = false;

  for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE; ++FI) {
    bool BBChanged = false;
    for (BasicBlock::iterator BI = FI->begin(), BE = FI->end(); BI != BE; ) {
      Instruction *II = BI++;
      switch (II->getOpcode()) {
      case Instruction::Select:
        BBChanged |= processSelect(cast<SelectInst>(II));
        break;
      case Instruction::PHI:
        BBChanged |= processPHI(cast<PHINode>(II));
        break;
      case Instruction::ICmp:
      case Instruction::FCmp:
        BBChanged |= processCmp(cast<CmpInst>(II));
        break;
      case Instruction::Load:
      case Instruction::Store:
        BBChanged |= processMemAccess(II);
        break;
      }
    }

    Instruction *Term = FI->getTerminator();
    switch (Term->getOpcode()) {
    case Instruction::Switch:
      BBChanged |= processSwitch(cast<SwitchInst>(Term));
      break;
    }

    FnChanged |= BBChanged;
  }

  return FnChanged;
}
