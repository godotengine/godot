//===- ScalarEvolution.cpp - Scalar Evolution Analysis --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the scalar evolution analysis
// engine, which is used primarily to analyze expressions involving induction
// variables in loops.
//
// There are several aspects to this library.  First is the representation of
// scalar expressions, which are represented as subclasses of the SCEV class.
// These classes are used to represent certain types of subexpressions that we
// can handle. We only create one SCEV of a particular shape, so
// pointer-comparisons for equality are legal.
//
// One important aspect of the SCEV objects is that they are never cyclic, even
// if there is a cycle in the dataflow for an expression (ie, a PHI node).  If
// the PHI node is one of the idioms that we can represent (e.g., a polynomial
// recurrence) then we represent it directly as a recurrence node, otherwise we
// represent it as a SCEVUnknown node.
//
// In addition to being able to represent expressions of various types, we also
// have folders that are used to build the *canonical* representation for a
// particular expression.  These folders are capable of using a variety of
// rewrite rules to simplify the expressions.
//
// Once the folders are defined, we can implement the more interesting
// higher-level code, such as the code that recognizes PHI nodes of various
// types, computes the execution count of a loop, etc.
//
// TODO: We should use these routines and value representations to implement
// dependence analysis!
//
//===----------------------------------------------------------------------===//
//
// There are several good references for the techniques used in this analysis.
//
//  Chains of recurrences -- a method to expedite the evaluation
//  of closed-form functions
//  Olaf Bachmann, Paul S. Wang, Eugene V. Zima
//
//  On computational properties of chains of recurrences
//  Eugene V. Zima
//
//  Symbolic Evaluation of Chains of Recurrences for Loop Optimization
//  Robert A. van Engelen
//
//  Efficient Symbolic Analysis for Optimizing Compilers
//  Robert A. van Engelen
//
//  Using the chains of recurrences algebra for data dependence testing and
//  induction variable substitution
//  MS Thesis, Johnie Birch
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Analysis/DxilValueCache.h" // HLSL Change

#include <algorithm>
using namespace llvm;

#define DEBUG_TYPE "scalar-evolution"

STATISTIC(NumArrayLenItCounts,
          "Number of trip counts computed with array length");
STATISTIC(NumTripCountsComputed,
          "Number of loops with predictable loop counts");
STATISTIC(NumTripCountsNotComputed,
          "Number of loops without predictable loop counts");
STATISTIC(NumBruteForceTripCountsComputed,
          "Number of loops with trip counts computed by force");

#if 0 // HLSL Change Starts - option pending
static cl::opt<unsigned>
MaxBruteForceIterations("scalar-evolution-max-iterations", cl::ReallyHidden,
                        cl::desc("Maximum number of iterations SCEV will "
                                 "symbolically execute a constant "
                                 "derived loop"),
                        cl::init(100));

// FIXME: Enable this with XDEBUG when the test suite is clean.
static cl::opt<bool>
VerifySCEV("verify-scev",
           cl::desc("Verify ScalarEvolution's backedge taken counts (slow)"));
#else
static const unsigned MaxBruteForceIterations = 100;
static const bool VerifySCEV = false;
#endif // HLSL Change Ends

INITIALIZE_PASS_BEGIN(ScalarEvolution, "scalar-evolution",
                "Scalar Evolution Analysis", false, true)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DxilValueCache) // HLSL Change
INITIALIZE_PASS_END(ScalarEvolution, "scalar-evolution",
                "Scalar Evolution Analysis", false, true)
char ScalarEvolution::ID = 0;

//===----------------------------------------------------------------------===//
//                           SCEV class definitions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Implementation of the SCEV class.
//

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void SCEV::dump() const {
  print(dbgs());
  dbgs() << '\n';
}
#endif

void SCEV::print(raw_ostream &OS) const {
  switch (static_cast<SCEVTypes>(getSCEVType())) {
  case scConstant:
    cast<SCEVConstant>(this)->getValue()->printAsOperand(OS, false);
    return;
  case scTruncate: {
    const SCEVTruncateExpr *Trunc = cast<SCEVTruncateExpr>(this);
    const SCEV *Op = Trunc->getOperand();
    OS << "(trunc " << *Op->getType() << " " << *Op << " to "
       << *Trunc->getType() << ")";
    return;
  }
  case scZeroExtend: {
    const SCEVZeroExtendExpr *ZExt = cast<SCEVZeroExtendExpr>(this);
    const SCEV *Op = ZExt->getOperand();
    OS << "(zext " << *Op->getType() << " " << *Op << " to "
       << *ZExt->getType() << ")";
    return;
  }
  case scSignExtend: {
    const SCEVSignExtendExpr *SExt = cast<SCEVSignExtendExpr>(this);
    const SCEV *Op = SExt->getOperand();
    OS << "(sext " << *Op->getType() << " " << *Op << " to "
       << *SExt->getType() << ")";
    return;
  }
  case scAddRecExpr: {
    const SCEVAddRecExpr *AR = cast<SCEVAddRecExpr>(this);
    OS << "{" << *AR->getOperand(0);
    for (unsigned i = 1, e = AR->getNumOperands(); i != e; ++i)
      OS << ",+," << *AR->getOperand(i);
    OS << "}<";
    if (AR->getNoWrapFlags(FlagNUW))
      OS << "nuw><";
    if (AR->getNoWrapFlags(FlagNSW))
      OS << "nsw><";
    if (AR->getNoWrapFlags(FlagNW) &&
        !AR->getNoWrapFlags((NoWrapFlags)(FlagNUW | FlagNSW)))
      OS << "nw><";
    AR->getLoop()->getHeader()->printAsOperand(OS, /*PrintType=*/false);
    OS << ">";
    return;
  }
  case scAddExpr:
  case scMulExpr:
  case scUMaxExpr:
  case scSMaxExpr: {
    const SCEVNAryExpr *NAry = cast<SCEVNAryExpr>(this);
    const char *OpStr = nullptr;
    switch (NAry->getSCEVType()) {
    case scAddExpr: OpStr = " + "; break;
    case scMulExpr: OpStr = " * "; break;
    case scUMaxExpr: OpStr = " umax "; break;
    case scSMaxExpr: OpStr = " smax "; break;
    }
    OS << "(";
    for (SCEVNAryExpr::op_iterator I = NAry->op_begin(), E = NAry->op_end();
         I != E; ++I) {
      OS << **I;
      if (std::next(I) != E)
        OS << OpStr;
    }
    OS << ")";
    switch (NAry->getSCEVType()) {
    case scAddExpr:
    case scMulExpr:
      if (NAry->getNoWrapFlags(FlagNUW))
        OS << "<nuw>";
      if (NAry->getNoWrapFlags(FlagNSW))
        OS << "<nsw>";
    }
    return;
  }
  case scUDivExpr: {
    const SCEVUDivExpr *UDiv = cast<SCEVUDivExpr>(this);
    OS << "(" << *UDiv->getLHS() << " /u " << *UDiv->getRHS() << ")";
    return;
  }
  case scUnknown: {
    const SCEVUnknown *U = cast<SCEVUnknown>(this);
    Type *AllocTy;
    if (U->isSizeOf(AllocTy)) {
      OS << "sizeof(" << *AllocTy << ")";
      return;
    }
    if (U->isAlignOf(AllocTy)) {
      OS << "alignof(" << *AllocTy << ")";
      return;
    }

    Type *CTy;
    Constant *FieldNo;
    if (U->isOffsetOf(CTy, FieldNo)) {
      OS << "offsetof(" << *CTy << ", ";
      FieldNo->printAsOperand(OS, false);
      OS << ")";
      return;
    }

    // Otherwise just print it normally.
    U->getValue()->printAsOperand(OS, false);
    return;
  }
  case scCouldNotCompute:
    OS << "***COULDNOTCOMPUTE***";
    return;
  }
  llvm_unreachable("Unknown SCEV kind!");
}

Type *SCEV::getType() const {
  switch (static_cast<SCEVTypes>(getSCEVType())) {
  case scConstant:
    return cast<SCEVConstant>(this)->getType();
  case scTruncate:
  case scZeroExtend:
  case scSignExtend:
    return cast<SCEVCastExpr>(this)->getType();
  case scAddRecExpr:
  case scMulExpr:
  case scUMaxExpr:
  case scSMaxExpr:
    return cast<SCEVNAryExpr>(this)->getType();
  case scAddExpr:
    return cast<SCEVAddExpr>(this)->getType();
  case scUDivExpr:
    return cast<SCEVUDivExpr>(this)->getType();
  case scUnknown:
    return cast<SCEVUnknown>(this)->getType();
  case scCouldNotCompute:
    llvm_unreachable("Attempt to use a SCEVCouldNotCompute object!");
  }
  llvm_unreachable("Unknown SCEV kind!");
}

bool SCEV::isZero() const {
  if (const SCEVConstant *SC = dyn_cast<SCEVConstant>(this))
    return SC->getValue()->isZero();
  return false;
}

bool SCEV::isOne() const {
  if (const SCEVConstant *SC = dyn_cast<SCEVConstant>(this))
    return SC->getValue()->isOne();
  return false;
}

bool SCEV::isAllOnesValue() const {
  if (const SCEVConstant *SC = dyn_cast<SCEVConstant>(this))
    return SC->getValue()->isAllOnesValue();
  return false;
}

/// isNonConstantNegative - Return true if the specified scev is negated, but
/// not a constant.
bool SCEV::isNonConstantNegative() const {
  const SCEVMulExpr *Mul = dyn_cast<SCEVMulExpr>(this);
  if (!Mul) return false;

  // If there is a constant factor, it will be first.
  const SCEVConstant *SC = dyn_cast<SCEVConstant>(Mul->getOperand(0));
  if (!SC) return false;

  // Return true if the value is negative, this matches things like (-42 * V).
  return SC->getValue()->getValue().isNegative();
}

SCEVCouldNotCompute::SCEVCouldNotCompute() :
  SCEV(FoldingSetNodeIDRef(), scCouldNotCompute) {}

bool SCEVCouldNotCompute::classof(const SCEV *S) {
  return S->getSCEVType() == scCouldNotCompute;
}

const SCEV *ScalarEvolution::getConstant(ConstantInt *V) {
  FoldingSetNodeID ID;
  ID.AddInteger(scConstant);
  ID.AddPointer(V);
  void *IP = nullptr;
  if (const SCEV *S = UniqueSCEVs.FindNodeOrInsertPos(ID, IP)) return S;
  SCEV *S = new (SCEVAllocator) SCEVConstant(ID.Intern(SCEVAllocator), V);
  UniqueSCEVs.InsertNode(S, IP);
  return S;
}

const SCEV *ScalarEvolution::getConstant(const APInt &Val) {
  return getConstant(ConstantInt::get(getContext(), Val));
}

const SCEV *
ScalarEvolution::getConstant(Type *Ty, uint64_t V, bool isSigned) {
  IntegerType *ITy = cast<IntegerType>(getEffectiveSCEVType(Ty));
  return getConstant(ConstantInt::get(ITy, V, isSigned));
}

SCEVCastExpr::SCEVCastExpr(const FoldingSetNodeIDRef ID,
                           unsigned SCEVTy, const SCEV *op, Type *ty)
  : SCEV(ID, SCEVTy), Op(op), Ty(ty) {}

SCEVTruncateExpr::SCEVTruncateExpr(const FoldingSetNodeIDRef ID,
                                   const SCEV *op, Type *ty)
  : SCEVCastExpr(ID, scTruncate, op, ty) {
  assert((Op->getType()->isIntegerTy() || Op->getType()->isPointerTy()) &&
         (Ty->isIntegerTy() || Ty->isPointerTy()) &&
         "Cannot truncate non-integer value!");
}

SCEVZeroExtendExpr::SCEVZeroExtendExpr(const FoldingSetNodeIDRef ID,
                                       const SCEV *op, Type *ty)
  : SCEVCastExpr(ID, scZeroExtend, op, ty) {
  assert((Op->getType()->isIntegerTy() || Op->getType()->isPointerTy()) &&
         (Ty->isIntegerTy() || Ty->isPointerTy()) &&
         "Cannot zero extend non-integer value!");
}

SCEVSignExtendExpr::SCEVSignExtendExpr(const FoldingSetNodeIDRef ID,
                                       const SCEV *op, Type *ty)
  : SCEVCastExpr(ID, scSignExtend, op, ty) {
  assert((Op->getType()->isIntegerTy() || Op->getType()->isPointerTy()) &&
         (Ty->isIntegerTy() || Ty->isPointerTy()) &&
         "Cannot sign extend non-integer value!");
}

void SCEVUnknown::deleted() {
  // Clear this SCEVUnknown from various maps.
  SE->forgetMemoizedResults(this);

  // Remove this SCEVUnknown from the uniquing map.
  SE->UniqueSCEVs.RemoveNode(this);

  // Release the value.
  setValPtr(nullptr);
}

void SCEVUnknown::allUsesReplacedWith(Value *New) {
  // Clear this SCEVUnknown from various maps.
  SE->forgetMemoizedResults(this);

  // Remove this SCEVUnknown from the uniquing map.
  SE->UniqueSCEVs.RemoveNode(this);

  // Update this SCEVUnknown to point to the new value. This is needed
  // because there may still be outstanding SCEVs which still point to
  // this SCEVUnknown.
  setValPtr(New);
}

bool SCEVUnknown::isSizeOf(Type *&AllocTy) const {
  if (ConstantExpr *VCE = dyn_cast<ConstantExpr>(getValue()))
    if (VCE->getOpcode() == Instruction::PtrToInt)
      if (ConstantExpr *CE = dyn_cast<ConstantExpr>(VCE->getOperand(0)))
        if (CE->getOpcode() == Instruction::GetElementPtr &&
            CE->getOperand(0)->isNullValue() &&
            CE->getNumOperands() == 2)
          if (ConstantInt *CI = dyn_cast<ConstantInt>(CE->getOperand(1)))
            if (CI->isOne()) {
              AllocTy = cast<PointerType>(CE->getOperand(0)->getType())
                                 ->getElementType();
              return true;
            }

  return false;
}

bool SCEVUnknown::isAlignOf(Type *&AllocTy) const {
  if (ConstantExpr *VCE = dyn_cast<ConstantExpr>(getValue()))
    if (VCE->getOpcode() == Instruction::PtrToInt)
      if (ConstantExpr *CE = dyn_cast<ConstantExpr>(VCE->getOperand(0)))
        if (CE->getOpcode() == Instruction::GetElementPtr &&
            CE->getOperand(0)->isNullValue()) {
          Type *Ty =
            cast<PointerType>(CE->getOperand(0)->getType())->getElementType();
          if (StructType *STy = dyn_cast<StructType>(Ty))
            if (!STy->isPacked() &&
                CE->getNumOperands() == 3 &&
                CE->getOperand(1)->isNullValue()) {
              if (ConstantInt *CI = dyn_cast<ConstantInt>(CE->getOperand(2)))
                if (CI->isOne() &&
                    STy->getNumElements() == 2 &&
                    STy->getElementType(0)->isIntegerTy(1)) {
                  AllocTy = STy->getElementType(1);
                  return true;
                }
            }
        }

  return false;
}

bool SCEVUnknown::isOffsetOf(Type *&CTy, Constant *&FieldNo) const {
  if (ConstantExpr *VCE = dyn_cast<ConstantExpr>(getValue()))
    if (VCE->getOpcode() == Instruction::PtrToInt)
      if (ConstantExpr *CE = dyn_cast<ConstantExpr>(VCE->getOperand(0)))
        if (CE->getOpcode() == Instruction::GetElementPtr &&
            CE->getNumOperands() == 3 &&
            CE->getOperand(0)->isNullValue() &&
            CE->getOperand(1)->isNullValue()) {
          Type *Ty =
            cast<PointerType>(CE->getOperand(0)->getType())->getElementType();
          // Ignore vector types here so that ScalarEvolutionExpander doesn't
          // emit getelementptrs that index into vectors.
          if (Ty->isStructTy() || Ty->isArrayTy()) {
            CTy = Ty;
            FieldNo = CE->getOperand(2);
            return true;
          }
        }

  return false;
}

//===----------------------------------------------------------------------===//
//                               SCEV Utilities
//===----------------------------------------------------------------------===//

namespace {
  /// SCEVComplexityCompare - Return true if the complexity of the LHS is less
  /// than the complexity of the RHS.  This comparator is used to canonicalize
  /// expressions.
  class SCEVComplexityCompare {
    const LoopInfo *const LI;
  public:
    explicit SCEVComplexityCompare(const LoopInfo *li) : LI(li) {}

    // Return true or false if LHS is less than, or at least RHS, respectively.
    bool operator()(const SCEV *LHS, const SCEV *RHS) const {
      return compare(LHS, RHS) < 0;
    }

    // Return negative, zero, or positive, if LHS is less than, equal to, or
    // greater than RHS, respectively. A three-way result allows recursive
    // comparisons to be more efficient.
    int compare(const SCEV *LHS, const SCEV *RHS) const {
      // Fast-path: SCEVs are uniqued so we can do a quick equality check.
      if (LHS == RHS)
        return 0;

      // Primarily, sort the SCEVs by their getSCEVType().
      unsigned LType = LHS->getSCEVType(), RType = RHS->getSCEVType();
      if (LType != RType)
        return (int)LType - (int)RType;

      // Aside from the getSCEVType() ordering, the particular ordering
      // isn't very important except that it's beneficial to be consistent,
      // so that (a + b) and (b + a) don't end up as different expressions.
      switch (static_cast<SCEVTypes>(LType)) {
      case scUnknown: {
        const SCEVUnknown *LU = cast<SCEVUnknown>(LHS);
        const SCEVUnknown *RU = cast<SCEVUnknown>(RHS);

        // Sort SCEVUnknown values with some loose heuristics. TODO: This is
        // not as complete as it could be.
        const Value *LV = LU->getValue(), *RV = RU->getValue();

        // Order pointer values after integer values. This helps SCEVExpander
        // form GEPs.
        bool LIsPointer = LV->getType()->isPointerTy(),
             RIsPointer = RV->getType()->isPointerTy();
        if (LIsPointer != RIsPointer)
          return (int)LIsPointer - (int)RIsPointer;

        // Compare getValueID values.
        unsigned LID = LV->getValueID(),
                 RID = RV->getValueID();
        if (LID != RID)
          return (int)LID - (int)RID;

        // Sort arguments by their position.
        if (const Argument *LA = dyn_cast<Argument>(LV)) {
          const Argument *RA = cast<Argument>(RV);
          unsigned LArgNo = LA->getArgNo(), RArgNo = RA->getArgNo();
          return (int)LArgNo - (int)RArgNo;
        }

        // For instructions, compare their loop depth, and their operand
        // count.  This is pretty loose.
        if (const Instruction *LInst = dyn_cast<Instruction>(LV)) {
          const Instruction *RInst = cast<Instruction>(RV);

          // Compare loop depths.
          const BasicBlock *LParent = LInst->getParent(),
                           *RParent = RInst->getParent();
          if (LParent != RParent) {
            unsigned LDepth = LI->getLoopDepth(LParent),
                     RDepth = LI->getLoopDepth(RParent);
            if (LDepth != RDepth)
              return (int)LDepth - (int)RDepth;
          }

          // Compare the number of operands.
          unsigned LNumOps = LInst->getNumOperands(),
                   RNumOps = RInst->getNumOperands();
          return (int)LNumOps - (int)RNumOps;
        }

        return 0;
      }

      case scConstant: {
        const SCEVConstant *LC = cast<SCEVConstant>(LHS);
        const SCEVConstant *RC = cast<SCEVConstant>(RHS);

        // Compare constant values.
        const APInt &LA = LC->getValue()->getValue();
        const APInt &RA = RC->getValue()->getValue();
        unsigned LBitWidth = LA.getBitWidth(), RBitWidth = RA.getBitWidth();
        if (LBitWidth != RBitWidth)
          return (int)LBitWidth - (int)RBitWidth;
        return LA.ult(RA) ? -1 : 1;
      }

      case scAddRecExpr: {
        const SCEVAddRecExpr *LA = cast<SCEVAddRecExpr>(LHS);
        const SCEVAddRecExpr *RA = cast<SCEVAddRecExpr>(RHS);

        // Compare addrec loop depths.
        const Loop *LLoop = LA->getLoop(), *RLoop = RA->getLoop();
        if (LLoop != RLoop) {
          unsigned LDepth = LLoop->getLoopDepth(),
                   RDepth = RLoop->getLoopDepth();
          if (LDepth != RDepth)
            return (int)LDepth - (int)RDepth;
        }

        // Addrec complexity grows with operand count.
        unsigned LNumOps = LA->getNumOperands(), RNumOps = RA->getNumOperands();
        if (LNumOps != RNumOps)
          return (int)LNumOps - (int)RNumOps;

        // Lexicographically compare.
        for (unsigned i = 0; i != LNumOps; ++i) {
          long X = compare(LA->getOperand(i), RA->getOperand(i));
          if (X != 0)
            return X;
        }

        return 0;
      }

      case scAddExpr:
      case scMulExpr:
      case scSMaxExpr:
      case scUMaxExpr: {
        const SCEVNAryExpr *LC = cast<SCEVNAryExpr>(LHS);
        const SCEVNAryExpr *RC = cast<SCEVNAryExpr>(RHS);

        // Lexicographically compare n-ary expressions.
        unsigned LNumOps = LC->getNumOperands(), RNumOps = RC->getNumOperands();
        if (LNumOps != RNumOps)
          return (int)LNumOps - (int)RNumOps;

        for (unsigned i = 0; i != LNumOps; ++i) {
          if (i >= RNumOps)
            return 1;
          long X = compare(LC->getOperand(i), RC->getOperand(i));
          if (X != 0)
            return X;
        }
        return (int)LNumOps - (int)RNumOps;
      }

      case scUDivExpr: {
        const SCEVUDivExpr *LC = cast<SCEVUDivExpr>(LHS);
        const SCEVUDivExpr *RC = cast<SCEVUDivExpr>(RHS);

        // Lexicographically compare udiv expressions.
        long X = compare(LC->getLHS(), RC->getLHS());
        if (X != 0)
          return X;
        return compare(LC->getRHS(), RC->getRHS());
      }

      case scTruncate:
      case scZeroExtend:
      case scSignExtend: {
        const SCEVCastExpr *LC = cast<SCEVCastExpr>(LHS);
        const SCEVCastExpr *RC = cast<SCEVCastExpr>(RHS);

        // Compare cast expressions by operand.
        return compare(LC->getOperand(), RC->getOperand());
      }

      case scCouldNotCompute:
        llvm_unreachable("Attempt to use a SCEVCouldNotCompute object!");
      }
      llvm_unreachable("Unknown SCEV kind!");
    }
  };
}

/// GroupByComplexity - Given a list of SCEV objects, order them by their
/// complexity, and group objects of the same complexity together by value.
/// When this routine is finished, we know that any duplicates in the vector are
/// consecutive and that complexity is monotonically increasing.
///
/// Note that we go take special precautions to ensure that we get deterministic
/// results from this routine.  In other words, we don't want the results of
/// this to depend on where the addresses of various SCEV objects happened to
/// land in memory.
///
static void GroupByComplexity(SmallVectorImpl<const SCEV *> &Ops,
                              LoopInfo *LI) {
  if (Ops.size() < 2) return;  // Noop
  if (Ops.size() == 2) {
    // This is the common case, which also happens to be trivially simple.
    // Special case it.
    const SCEV *&LHS = Ops[0], *&RHS = Ops[1];
    if (SCEVComplexityCompare(LI)(RHS, LHS))
      std::swap(LHS, RHS);
    return;
  }

  // Do the rough sort by complexity.
  std::stable_sort(Ops.begin(), Ops.end(), SCEVComplexityCompare(LI));

  // Now that we are sorted by complexity, group elements of the same
  // complexity.  Note that this is, at worst, N^2, but the vector is likely to
  // be extremely short in practice.  Note that we take this approach because we
  // do not want to depend on the addresses of the objects we are grouping.
  for (unsigned i = 0, e = Ops.size(); i != e-2; ++i) {
    const SCEV *S = Ops[i];
    unsigned Complexity = S->getSCEVType();

    // If there are any objects of the same complexity and same value as this
    // one, group them.
    for (unsigned j = i+1; j != e && Ops[j]->getSCEVType() == Complexity; ++j) {
      if (Ops[j] == S) { // Found a duplicate.
        // Move it to immediately after i'th element.
        std::swap(Ops[i+1], Ops[j]);
        ++i;   // no need to rescan it.
        if (i == e-2) return;  // Done!
      }
    }
  }
}

namespace {
struct FindSCEVSize {
  int Size;
  FindSCEVSize() : Size(0) {}

  bool follow(const SCEV *S) {
    ++Size;
    // Keep looking at all operands of S.
    return true;
  }
  bool isDone() const {
    return false;
  }
};
}

// Returns the size of the SCEV S.
static inline int sizeOfSCEV(const SCEV *S) {
  FindSCEVSize F;
  SCEVTraversal<FindSCEVSize> ST(F);
  ST.visitAll(S);
  return F.Size;
}

namespace {

struct SCEVDivision : public SCEVVisitor<SCEVDivision, void> {
public:
  // Computes the Quotient and Remainder of the division of Numerator by
  // Denominator.
  static void divide(ScalarEvolution &SE, const SCEV *Numerator,
                     const SCEV *Denominator, const SCEV **Quotient,
                     const SCEV **Remainder) {
    assert(Numerator && Denominator && "Uninitialized SCEV");

    SCEVDivision D(SE, Numerator, Denominator);

    // Check for the trivial case here to avoid having to check for it in the
    // rest of the code.
    if (Numerator == Denominator) {
      *Quotient = D.One;
      *Remainder = D.Zero;
      return;
    }

    if (Numerator->isZero()) {
      *Quotient = D.Zero;
      *Remainder = D.Zero;
      return;
    }

    // A simple case when N/1. The quotient is N.
    if (Denominator->isOne()) {
      *Quotient = Numerator;
      *Remainder = D.Zero;
      return;
    }

    // Split the Denominator when it is a product.
    if (const SCEVMulExpr *T = dyn_cast<const SCEVMulExpr>(Denominator)) {
      const SCEV *Q, *R;
      *Quotient = Numerator;
      for (const SCEV *Op : T->operands()) {
        divide(SE, *Quotient, Op, &Q, &R);
        *Quotient = Q;

        // Bail out when the Numerator is not divisible by one of the terms of
        // the Denominator.
        if (!R->isZero()) {
          *Quotient = D.Zero;
          *Remainder = Numerator;
          return;
        }
      }
      *Remainder = D.Zero;
      return;
    }

    D.visit(Numerator);
    *Quotient = D.Quotient;
    *Remainder = D.Remainder;
  }

  // Except in the trivial case described above, we do not know how to divide
  // Expr by Denominator for the following functions with empty implementation.
  void visitTruncateExpr(const SCEVTruncateExpr *Numerator) {}
  void visitZeroExtendExpr(const SCEVZeroExtendExpr *Numerator) {}
  void visitSignExtendExpr(const SCEVSignExtendExpr *Numerator) {}
  void visitUDivExpr(const SCEVUDivExpr *Numerator) {}
  void visitSMaxExpr(const SCEVSMaxExpr *Numerator) {}
  void visitUMaxExpr(const SCEVUMaxExpr *Numerator) {}
  void visitUnknown(const SCEVUnknown *Numerator) {}
  void visitCouldNotCompute(const SCEVCouldNotCompute *Numerator) {}

  void visitConstant(const SCEVConstant *Numerator) {
    if (const SCEVConstant *D = dyn_cast<SCEVConstant>(Denominator)) {
      APInt NumeratorVal = Numerator->getValue()->getValue();
      APInt DenominatorVal = D->getValue()->getValue();
      uint32_t NumeratorBW = NumeratorVal.getBitWidth();
      uint32_t DenominatorBW = DenominatorVal.getBitWidth();

      if (NumeratorBW > DenominatorBW)
        DenominatorVal = DenominatorVal.sext(NumeratorBW);
      else if (NumeratorBW < DenominatorBW)
        NumeratorVal = NumeratorVal.sext(DenominatorBW);

      APInt QuotientVal(NumeratorVal.getBitWidth(), 0);
      APInt RemainderVal(NumeratorVal.getBitWidth(), 0);
      APInt::sdivrem(NumeratorVal, DenominatorVal, QuotientVal, RemainderVal);
      Quotient = SE.getConstant(QuotientVal);
      Remainder = SE.getConstant(RemainderVal);
      return;
    }
  }

  void visitAddRecExpr(const SCEVAddRecExpr *Numerator) {
    const SCEV *StartQ, *StartR, *StepQ, *StepR;
    assert(Numerator->isAffine() && "Numerator should be affine");
    divide(SE, Numerator->getStart(), Denominator, &StartQ, &StartR);
    divide(SE, Numerator->getStepRecurrence(SE), Denominator, &StepQ, &StepR);
    // Bail out if the types do not match.
    Type *Ty = Denominator->getType();
    if (Ty != StartQ->getType() || Ty != StartR->getType() ||
        Ty != StepQ->getType() || Ty != StepR->getType()) {
      Quotient = Zero;
      Remainder = Numerator;
      return;
    }
    Quotient = SE.getAddRecExpr(StartQ, StepQ, Numerator->getLoop(),
                                Numerator->getNoWrapFlags());
    Remainder = SE.getAddRecExpr(StartR, StepR, Numerator->getLoop(),
                                 Numerator->getNoWrapFlags());
  }

  void visitAddExpr(const SCEVAddExpr *Numerator) {
    SmallVector<const SCEV *, 2> Qs, Rs;
    Type *Ty = Denominator->getType();

    for (const SCEV *Op : Numerator->operands()) {
      const SCEV *Q, *R;
      divide(SE, Op, Denominator, &Q, &R);

      // Bail out if types do not match.
      if (Ty != Q->getType() || Ty != R->getType()) {
        Quotient = Zero;
        Remainder = Numerator;
        return;
      }

      Qs.push_back(Q);
      Rs.push_back(R);
    }

    if (Qs.size() == 1) {
      Quotient = Qs[0];
      Remainder = Rs[0];
      return;
    }

    Quotient = SE.getAddExpr(Qs);
    Remainder = SE.getAddExpr(Rs);
  }

  void visitMulExpr(const SCEVMulExpr *Numerator) {
    SmallVector<const SCEV *, 2> Qs;
    Type *Ty = Denominator->getType();

    bool FoundDenominatorTerm = false;
    for (const SCEV *Op : Numerator->operands()) {
      // Bail out if types do not match.
      if (Ty != Op->getType()) {
        Quotient = Zero;
        Remainder = Numerator;
        return;
      }

      if (FoundDenominatorTerm) {
        Qs.push_back(Op);
        continue;
      }

      // Check whether Denominator divides one of the product operands.
      const SCEV *Q, *R;
      divide(SE, Op, Denominator, &Q, &R);
      if (!R->isZero()) {
        Qs.push_back(Op);
        continue;
      }

      // Bail out if types do not match.
      if (Ty != Q->getType()) {
        Quotient = Zero;
        Remainder = Numerator;
        return;
      }

      FoundDenominatorTerm = true;
      Qs.push_back(Q);
    }

    if (FoundDenominatorTerm) {
      Remainder = Zero;
      if (Qs.size() == 1)
        Quotient = Qs[0];
      else
        Quotient = SE.getMulExpr(Qs);
      return;
    }

    if (!isa<SCEVUnknown>(Denominator)) {
      Quotient = Zero;
      Remainder = Numerator;
      return;
    }

    // The Remainder is obtained by replacing Denominator by 0 in Numerator.
    ValueToValueMap RewriteMap;
    RewriteMap[cast<SCEVUnknown>(Denominator)->getValue()] =
        cast<SCEVConstant>(Zero)->getValue();
    Remainder = SCEVParameterRewriter::rewrite(Numerator, SE, RewriteMap, true);

    if (Remainder->isZero()) {
      // The Quotient is obtained by replacing Denominator by 1 in Numerator.
      RewriteMap[cast<SCEVUnknown>(Denominator)->getValue()] =
          cast<SCEVConstant>(One)->getValue();
      Quotient =
          SCEVParameterRewriter::rewrite(Numerator, SE, RewriteMap, true);
      return;
    }

    // Quotient is (Numerator - Remainder) divided by Denominator.
    const SCEV *Q, *R;
    const SCEV *Diff = SE.getMinusSCEV(Numerator, Remainder);
    if (sizeOfSCEV(Diff) > sizeOfSCEV(Numerator)) {
      // This SCEV does not seem to simplify: fail the division here.
      Quotient = Zero;
      Remainder = Numerator;
      return;
    }
    divide(SE, Diff, Denominator, &Q, &R);
    assert(R == Zero &&
           "(Numerator - Remainder) should evenly divide Denominator");
    Quotient = Q;
  }

private:
  SCEVDivision(ScalarEvolution &S, const SCEV *Numerator,
               const SCEV *Denominator)
      : SE(S), Denominator(Denominator) {
    Zero = SE.getConstant(Denominator->getType(), 0);
    One = SE.getConstant(Denominator->getType(), 1);

    // By default, we don't know how to divide Expr by Denominator.
    // Providing the default here simplifies the rest of the code.
    Quotient = Zero;
    Remainder = Numerator;
  }

  ScalarEvolution &SE;
  const SCEV *Denominator, *Quotient, *Remainder, *Zero, *One;
};

}

//===----------------------------------------------------------------------===//
//                      Simple SCEV method implementations
//===----------------------------------------------------------------------===//

/// BinomialCoefficient - Compute BC(It, K).  The result has width W.
/// Assume, K > 0.
static const SCEV *BinomialCoefficient(const SCEV *It, unsigned K,
                                       ScalarEvolution &SE,
                                       Type *ResultTy) {
  // Handle the simplest case efficiently.
  if (K == 1)
    return SE.getTruncateOrZeroExtend(It, ResultTy);

  // We are using the following formula for BC(It, K):
  //
  //   BC(It, K) = (It * (It - 1) * ... * (It - K + 1)) / K!
  //
  // Suppose, W is the bitwidth of the return value.  We must be prepared for
  // overflow.  Hence, we must assure that the result of our computation is
  // equal to the accurate one modulo 2^W.  Unfortunately, division isn't
  // safe in modular arithmetic.
  //
  // However, this code doesn't use exactly that formula; the formula it uses
  // is something like the following, where T is the number of factors of 2 in
  // K! (i.e. trailing zeros in the binary representation of K!), and ^ is
  // exponentiation:
  //
  //   BC(It, K) = (It * (It - 1) * ... * (It - K + 1)) / 2^T / (K! / 2^T)
  //
  // This formula is trivially equivalent to the previous formula.  However,
  // this formula can be implemented much more efficiently.  The trick is that
  // K! / 2^T is odd, and exact division by an odd number *is* safe in modular
  // arithmetic.  To do exact division in modular arithmetic, all we have
  // to do is multiply by the inverse.  Therefore, this step can be done at
  // width W.
  //
  // The next issue is how to safely do the division by 2^T.  The way this
  // is done is by doing the multiplication step at a width of at least W + T
  // bits.  This way, the bottom W+T bits of the product are accurate. Then,
  // when we perform the division by 2^T (which is equivalent to a right shift
  // by T), the bottom W bits are accurate.  Extra bits are okay; they'll get
  // truncated out after the division by 2^T.
  //
  // In comparison to just directly using the first formula, this technique
  // is much more efficient; using the first formula requires W * K bits,
  // but this formula less than W + K bits. Also, the first formula requires
  // a division step, whereas this formula only requires multiplies and shifts.
  //
  // It doesn't matter whether the subtraction step is done in the calculation
  // width or the input iteration count's width; if the subtraction overflows,
  // the result must be zero anyway.  We prefer here to do it in the width of
  // the induction variable because it helps a lot for certain cases; CodeGen
  // isn't smart enough to ignore the overflow, which leads to much less
  // efficient code if the width of the subtraction is wider than the native
  // register width.
  //
  // (It's possible to not widen at all by pulling out factors of 2 before
  // the multiplication; for example, K=2 can be calculated as
  // It/2*(It+(It*INT_MIN/INT_MIN)+-1). However, it requires
  // extra arithmetic, so it's not an obvious win, and it gets
  // much more complicated for K > 3.)

  // Protection from insane SCEVs; this bound is conservative,
  // but it probably doesn't matter.
  if (K > 1000)
    return SE.getCouldNotCompute();

  unsigned W = SE.getTypeSizeInBits(ResultTy);

  // Calculate K! / 2^T and T; we divide out the factors of two before
  // multiplying for calculating K! / 2^T to avoid overflow.
  // Other overflow doesn't matter because we only care about the bottom
  // W bits of the result.
  APInt OddFactorial(W, 1);
  unsigned T = 1;
  for (unsigned i = 3; i <= K; ++i) {
    APInt Mult(W, i);
    unsigned TwoFactors = Mult.countTrailingZeros();
    T += TwoFactors;
    Mult = Mult.lshr(TwoFactors);
    OddFactorial *= Mult;
  }

  // We need at least W + T bits for the multiplication step
  unsigned CalculationBits = W + T;

  // Calculate 2^T, at width T+W.
  APInt DivFactor = APInt::getOneBitSet(CalculationBits, T);

  // Calculate the multiplicative inverse of K! / 2^T;
  // this multiplication factor will perform the exact division by
  // K! / 2^T.
  APInt Mod = APInt::getSignedMinValue(W+1);
  APInt MultiplyFactor = OddFactorial.zext(W+1);
  MultiplyFactor = MultiplyFactor.multiplicativeInverse(Mod);
  MultiplyFactor = MultiplyFactor.trunc(W);

  // Calculate the product, at width T+W
  IntegerType *CalculationTy = IntegerType::get(SE.getContext(),
                                                      CalculationBits);
  const SCEV *Dividend = SE.getTruncateOrZeroExtend(It, CalculationTy);
  for (unsigned i = 1; i != K; ++i) {
    const SCEV *S = SE.getMinusSCEV(It, SE.getConstant(It->getType(), i));
    Dividend = SE.getMulExpr(Dividend,
                             SE.getTruncateOrZeroExtend(S, CalculationTy));
  }

  // Divide by 2^T
  const SCEV *DivResult = SE.getUDivExpr(Dividend, SE.getConstant(DivFactor));

  // Truncate the result, and divide by K! / 2^T.

  return SE.getMulExpr(SE.getConstant(MultiplyFactor),
                       SE.getTruncateOrZeroExtend(DivResult, ResultTy));
}

/// evaluateAtIteration - Return the value of this chain of recurrences at
/// the specified iteration number.  We can evaluate this recurrence by
/// multiplying each element in the chain by the binomial coefficient
/// corresponding to it.  In other words, we can evaluate {A,+,B,+,C,+,D} as:
///
///   A*BC(It, 0) + B*BC(It, 1) + C*BC(It, 2) + D*BC(It, 3)
///
/// where BC(It, k) stands for binomial coefficient.
///
const SCEV *SCEVAddRecExpr::evaluateAtIteration(const SCEV *It,
                                                ScalarEvolution &SE) const {
  const SCEV *Result = getStart();
  for (unsigned i = 1, e = getNumOperands(); i != e; ++i) {
    // The computation is correct in the face of overflow provided that the
    // multiplication is performed _after_ the evaluation of the binomial
    // coefficient.
    const SCEV *Coeff = BinomialCoefficient(It, i, SE, getType());
    if (isa<SCEVCouldNotCompute>(Coeff))
      return Coeff;

    Result = SE.getAddExpr(Result, SE.getMulExpr(getOperand(i), Coeff));
  }
  return Result;
}

//===----------------------------------------------------------------------===//
//                    SCEV Expression folder implementations
//===----------------------------------------------------------------------===//

const SCEV *ScalarEvolution::getTruncateExpr(const SCEV *Op,
                                             Type *Ty) {
  assert(getTypeSizeInBits(Op->getType()) > getTypeSizeInBits(Ty) &&
         "This is not a truncating conversion!");
  assert(isSCEVable(Ty) &&
         "This is not a conversion to a SCEVable type!");
  Ty = getEffectiveSCEVType(Ty);

  FoldingSetNodeID ID;
  ID.AddInteger(scTruncate);
  ID.AddPointer(Op);
  ID.AddPointer(Ty);
  void *IP = nullptr;
  if (const SCEV *S = UniqueSCEVs.FindNodeOrInsertPos(ID, IP)) return S;

  // Fold if the operand is constant.
  if (const SCEVConstant *SC = dyn_cast<SCEVConstant>(Op))
    return getConstant(
      cast<ConstantInt>(ConstantExpr::getTrunc(SC->getValue(), Ty)));

  // trunc(trunc(x)) --> trunc(x)
  if (const SCEVTruncateExpr *ST = dyn_cast<SCEVTruncateExpr>(Op))
    return getTruncateExpr(ST->getOperand(), Ty);

  // trunc(sext(x)) --> sext(x) if widening or trunc(x) if narrowing
  if (const SCEVSignExtendExpr *SS = dyn_cast<SCEVSignExtendExpr>(Op))
    return getTruncateOrSignExtend(SS->getOperand(), Ty);

  // trunc(zext(x)) --> zext(x) if widening or trunc(x) if narrowing
  if (const SCEVZeroExtendExpr *SZ = dyn_cast<SCEVZeroExtendExpr>(Op))
    return getTruncateOrZeroExtend(SZ->getOperand(), Ty);

  // trunc(x1+x2+...+xN) --> trunc(x1)+trunc(x2)+...+trunc(xN) if we can
  // eliminate all the truncates, or we replace other casts with truncates.
  if (const SCEVAddExpr *SA = dyn_cast<SCEVAddExpr>(Op)) {
    SmallVector<const SCEV *, 4> Operands;
    bool hasTrunc = false;
    for (unsigned i = 0, e = SA->getNumOperands(); i != e && !hasTrunc; ++i) {
      const SCEV *S = getTruncateExpr(SA->getOperand(i), Ty);
      if (!isa<SCEVCastExpr>(SA->getOperand(i)))
        hasTrunc = isa<SCEVTruncateExpr>(S);
      Operands.push_back(S);
    }
    if (!hasTrunc)
      return getAddExpr(Operands);
    UniqueSCEVs.FindNodeOrInsertPos(ID, IP);  // Mutates IP, returns NULL.
  }

  // trunc(x1*x2*...*xN) --> trunc(x1)*trunc(x2)*...*trunc(xN) if we can
  // eliminate all the truncates, or we replace other casts with truncates.
  if (const SCEVMulExpr *SM = dyn_cast<SCEVMulExpr>(Op)) {
    SmallVector<const SCEV *, 4> Operands;
    bool hasTrunc = false;
    for (unsigned i = 0, e = SM->getNumOperands(); i != e && !hasTrunc; ++i) {
      const SCEV *S = getTruncateExpr(SM->getOperand(i), Ty);
      if (!isa<SCEVCastExpr>(SM->getOperand(i)))
        hasTrunc = isa<SCEVTruncateExpr>(S);
      Operands.push_back(S);
    }
    if (!hasTrunc)
      return getMulExpr(Operands);
    UniqueSCEVs.FindNodeOrInsertPos(ID, IP);  // Mutates IP, returns NULL.
  }

  // If the input value is a chrec scev, truncate the chrec's operands.
  if (const SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(Op)) {
    SmallVector<const SCEV *, 4> Operands;
    for (unsigned i = 0, e = AddRec->getNumOperands(); i != e; ++i)
      Operands.push_back(getTruncateExpr(AddRec->getOperand(i), Ty));
    return getAddRecExpr(Operands, AddRec->getLoop(), SCEV::FlagAnyWrap);
  }

  // The cast wasn't folded; create an explicit cast node. We can reuse
  // the existing insert position since if we get here, we won't have
  // made any changes which would invalidate it.
  SCEV *S = new (SCEVAllocator) SCEVTruncateExpr(ID.Intern(SCEVAllocator),
                                                 Op, Ty);
  UniqueSCEVs.InsertNode(S, IP);
  return S;
}

// Get the limit of a recurrence such that incrementing by Step cannot cause
// signed overflow as long as the value of the recurrence within the
// loop does not exceed this limit before incrementing.
static const SCEV *getSignedOverflowLimitForStep(const SCEV *Step,
                                                 ICmpInst::Predicate *Pred,
                                                 ScalarEvolution *SE) {
  unsigned BitWidth = SE->getTypeSizeInBits(Step->getType());
  if (SE->isKnownPositive(Step)) {
    *Pred = ICmpInst::ICMP_SLT;
    return SE->getConstant(APInt::getSignedMinValue(BitWidth) -
                           SE->getSignedRange(Step).getSignedMax());
  }
  if (SE->isKnownNegative(Step)) {
    *Pred = ICmpInst::ICMP_SGT;
    return SE->getConstant(APInt::getSignedMaxValue(BitWidth) -
                           SE->getSignedRange(Step).getSignedMin());
  }
  return nullptr;
}

// Get the limit of a recurrence such that incrementing by Step cannot cause
// unsigned overflow as long as the value of the recurrence within the loop does
// not exceed this limit before incrementing.
static const SCEV *getUnsignedOverflowLimitForStep(const SCEV *Step,
                                                   ICmpInst::Predicate *Pred,
                                                   ScalarEvolution *SE) {
  unsigned BitWidth = SE->getTypeSizeInBits(Step->getType());
  *Pred = ICmpInst::ICMP_ULT;

  return SE->getConstant(APInt::getMinValue(BitWidth) -
                         SE->getUnsignedRange(Step).getUnsignedMax());
}

namespace {

struct ExtendOpTraitsBase {
  typedef const SCEV *(ScalarEvolution::*GetExtendExprTy)(const SCEV *, Type *);
};

// Used to make code generic over signed and unsigned overflow.
template <typename ExtendOp> struct ExtendOpTraits {
  // Members present:
  //
  // static const SCEV::NoWrapFlags WrapType;
  //
  // static const ExtendOpTraitsBase::GetExtendExprTy GetExtendExpr;
  //
  // static const SCEV *getOverflowLimitForStep(const SCEV *Step,
  //                                           ICmpInst::Predicate *Pred,
  //                                           ScalarEvolution *SE);
};

template <>
struct ExtendOpTraits<SCEVSignExtendExpr> : public ExtendOpTraitsBase {
  static const SCEV::NoWrapFlags WrapType = SCEV::FlagNSW;

  static const GetExtendExprTy GetExtendExpr;

  static const SCEV *getOverflowLimitForStep(const SCEV *Step,
                                             ICmpInst::Predicate *Pred,
                                             ScalarEvolution *SE) {
    return getSignedOverflowLimitForStep(Step, Pred, SE);
  }
};

const ExtendOpTraitsBase::GetExtendExprTy ExtendOpTraits<
    SCEVSignExtendExpr>::GetExtendExpr = &ScalarEvolution::getSignExtendExpr;

template <>
struct ExtendOpTraits<SCEVZeroExtendExpr> : public ExtendOpTraitsBase {
  static const SCEV::NoWrapFlags WrapType = SCEV::FlagNUW;

  static const GetExtendExprTy GetExtendExpr;

  static const SCEV *getOverflowLimitForStep(const SCEV *Step,
                                             ICmpInst::Predicate *Pred,
                                             ScalarEvolution *SE) {
    return getUnsignedOverflowLimitForStep(Step, Pred, SE);
  }
};

const ExtendOpTraitsBase::GetExtendExprTy ExtendOpTraits<
    SCEVZeroExtendExpr>::GetExtendExpr = &ScalarEvolution::getZeroExtendExpr;
}

// The recurrence AR has been shown to have no signed/unsigned wrap or something
// close to it. Typically, if we can prove NSW/NUW for AR, then we can just as
// easily prove NSW/NUW for its preincrement or postincrement sibling. This
// allows normalizing a sign/zero extended AddRec as such: {sext/zext(Step +
// Start),+,Step} => {(Step + sext/zext(Start),+,Step} As a result, the
// expression "Step + sext/zext(PreIncAR)" is congruent with
// "sext/zext(PostIncAR)"
template <typename ExtendOpTy>
static const SCEV *getPreStartForExtend(const SCEVAddRecExpr *AR, Type *Ty,
                                        ScalarEvolution *SE) {
  auto WrapType = ExtendOpTraits<ExtendOpTy>::WrapType;
  auto GetExtendExpr = ExtendOpTraits<ExtendOpTy>::GetExtendExpr;

  const Loop *L = AR->getLoop();
  const SCEV *Start = AR->getStart();
  const SCEV *Step = AR->getStepRecurrence(*SE);

  // Check for a simple looking step prior to loop entry.
  const SCEVAddExpr *SA = dyn_cast<SCEVAddExpr>(Start);
  if (!SA)
    return nullptr;

  // Create an AddExpr for "PreStart" after subtracting Step. Full SCEV
  // subtraction is expensive. For this purpose, perform a quick and dirty
  // difference, by checking for Step in the operand list.
  SmallVector<const SCEV *, 4> DiffOps;
  for (const SCEV *Op : SA->operands())
    if (Op != Step)
      DiffOps.push_back(Op);

  if (DiffOps.size() == SA->getNumOperands())
    return nullptr;

  // Try to prove `WrapType` (SCEV::FlagNSW or SCEV::FlagNUW) on `PreStart` +
  // `Step`:

  // 1. NSW/NUW flags on the step increment.
  const SCEV *PreStart = SE->getAddExpr(DiffOps, SA->getNoWrapFlags());
  const SCEVAddRecExpr *PreAR = dyn_cast<SCEVAddRecExpr>(
      SE->getAddRecExpr(PreStart, Step, L, SCEV::FlagAnyWrap));

  // "{S,+,X} is <nsw>/<nuw>" and "the backedge is taken at least once" implies
  // "S+X does not sign/unsign-overflow".
  //

  const SCEV *BECount = SE->getBackedgeTakenCount(L);
  if (PreAR && PreAR->getNoWrapFlags(WrapType) &&
      !isa<SCEVCouldNotCompute>(BECount) && SE->isKnownPositive(BECount))
    return PreStart;

  // 2. Direct overflow check on the step operation's expression.
  unsigned BitWidth = SE->getTypeSizeInBits(AR->getType());
  Type *WideTy = IntegerType::get(SE->getContext(), BitWidth * 2);
  const SCEV *OperandExtendedStart =
      SE->getAddExpr((SE->*GetExtendExpr)(PreStart, WideTy),
                     (SE->*GetExtendExpr)(Step, WideTy));
  if ((SE->*GetExtendExpr)(Start, WideTy) == OperandExtendedStart) {
    if (PreAR && AR->getNoWrapFlags(WrapType)) {
      // If we know `AR` == {`PreStart`+`Step`,+,`Step`} is `WrapType` (FlagNSW
      // or FlagNUW) and that `PreStart` + `Step` is `WrapType` too, then
      // `PreAR` == {`PreStart`,+,`Step`} is also `WrapType`.  Cache this fact.
      const_cast<SCEVAddRecExpr *>(PreAR)->setNoWrapFlags(WrapType);
    }
    return PreStart;
  }

  // 3. Loop precondition.
  ICmpInst::Predicate Pred;
  const SCEV *OverflowLimit =
      ExtendOpTraits<ExtendOpTy>::getOverflowLimitForStep(Step, &Pred, SE);

  if (OverflowLimit &&
      SE->isLoopEntryGuardedByCond(L, Pred, PreStart, OverflowLimit)) {
    return PreStart;
  }
  return nullptr;
}

// Get the normalized zero or sign extended expression for this AddRec's Start.
template <typename ExtendOpTy>
static const SCEV *getExtendAddRecStart(const SCEVAddRecExpr *AR, Type *Ty,
                                        ScalarEvolution *SE) {
  auto GetExtendExpr = ExtendOpTraits<ExtendOpTy>::GetExtendExpr;

  const SCEV *PreStart = getPreStartForExtend<ExtendOpTy>(AR, Ty, SE);
  if (!PreStart)
    return (SE->*GetExtendExpr)(AR->getStart(), Ty);

  return SE->getAddExpr((SE->*GetExtendExpr)(AR->getStepRecurrence(*SE), Ty),
                        (SE->*GetExtendExpr)(PreStart, Ty));
}

// Try to prove away overflow by looking at "nearby" add recurrences.  A
// motivating example for this rule: if we know `{0,+,4}` is `ult` `-1` and it
// does not itself wrap then we can conclude that `{1,+,4}` is `nuw`.
//
// Formally:
//
//     {S,+,X} == {S-T,+,X} + T
//  => Ext({S,+,X}) == Ext({S-T,+,X} + T)
//
// If ({S-T,+,X} + T) does not overflow  ... (1)
//
//  RHS == Ext({S-T,+,X} + T) == Ext({S-T,+,X}) + Ext(T)
//
// If {S-T,+,X} does not overflow  ... (2)
//
//  RHS == Ext({S-T,+,X}) + Ext(T) == {Ext(S-T),+,Ext(X)} + Ext(T)
//      == {Ext(S-T)+Ext(T),+,Ext(X)}
//
// If (S-T)+T does not overflow  ... (3)
//
//  RHS == {Ext(S-T)+Ext(T),+,Ext(X)} == {Ext(S-T+T),+,Ext(X)}
//      == {Ext(S),+,Ext(X)} == LHS
//
// Thus, if (1), (2) and (3) are true for some T, then
//   Ext({S,+,X}) == {Ext(S),+,Ext(X)}
//
// (3) is implied by (1) -- "(S-T)+T does not overflow" is simply "({S-T,+,X}+T)
// does not overflow" restricted to the 0th iteration.  Therefore we only need
// to check for (1) and (2).
//
// In the current context, S is `Start`, X is `Step`, Ext is `ExtendOpTy` and T
// is `Delta` (defined below).
//
template <typename ExtendOpTy>
bool ScalarEvolution::proveNoWrapByVaryingStart(const SCEV *Start,
                                                const SCEV *Step,
                                                const Loop *L) {
  auto WrapType = ExtendOpTraits<ExtendOpTy>::WrapType;

  // We restrict `Start` to a constant to prevent SCEV from spending too much
  // time here.  It is correct (but more expensive) to continue with a
  // non-constant `Start` and do a general SCEV subtraction to compute
  // `PreStart` below.
  //
  const SCEVConstant *StartC = dyn_cast<SCEVConstant>(Start);
  if (!StartC)
    return false;

  APInt StartAI = StartC->getValue()->getValue();

  for (unsigned Delta : {-2, -1, 1, 2}) {
    const SCEV *PreStart = getConstant(StartAI - Delta);

    // Give up if we don't already have the add recurrence we need because
    // actually constructing an add recurrence is relatively expensive.
    const SCEVAddRecExpr *PreAR = [&]() {
      FoldingSetNodeID ID;
      ID.AddInteger(scAddRecExpr);
      ID.AddPointer(PreStart);
      ID.AddPointer(Step);
      ID.AddPointer(L);
      void *IP = nullptr;
      return static_cast<SCEVAddRecExpr *>(
          this->UniqueSCEVs.FindNodeOrInsertPos(ID, IP));
    }();

    if (PreAR && PreAR->getNoWrapFlags(WrapType)) {  // proves (2)
      const SCEV *DeltaS = getConstant(StartC->getType(), Delta);
      ICmpInst::Predicate Pred = ICmpInst::BAD_ICMP_PREDICATE;
      const SCEV *Limit = ExtendOpTraits<ExtendOpTy>::getOverflowLimitForStep(
          DeltaS, &Pred, this);
      if (Limit && isKnownPredicate(Pred, PreAR, Limit))  // proves (1)
        return true;
    }
  }

  return false;
}

const SCEV *ScalarEvolution::getZeroExtendExpr(const SCEV *Op,
                                               Type *Ty) {
  assert(getTypeSizeInBits(Op->getType()) < getTypeSizeInBits(Ty) &&
         "This is not an extending conversion!");
  assert(isSCEVable(Ty) &&
         "This is not a conversion to a SCEVable type!");
  Ty = getEffectiveSCEVType(Ty);

  // Fold if the operand is constant.
  if (const SCEVConstant *SC = dyn_cast<SCEVConstant>(Op))
    return getConstant(
      cast<ConstantInt>(ConstantExpr::getZExt(SC->getValue(), Ty)));

  // zext(zext(x)) --> zext(x)
  if (const SCEVZeroExtendExpr *SZ = dyn_cast<SCEVZeroExtendExpr>(Op))
    return getZeroExtendExpr(SZ->getOperand(), Ty);

  // Before doing any expensive analysis, check to see if we've already
  // computed a SCEV for this Op and Ty.
  FoldingSetNodeID ID;
  ID.AddInteger(scZeroExtend);
  ID.AddPointer(Op);
  ID.AddPointer(Ty);
  void *IP = nullptr;
  if (const SCEV *S = UniqueSCEVs.FindNodeOrInsertPos(ID, IP)) return S;

  // zext(trunc(x)) --> zext(x) or x or trunc(x)
  if (const SCEVTruncateExpr *ST = dyn_cast<SCEVTruncateExpr>(Op)) {
    // It's possible the bits taken off by the truncate were all zero bits. If
    // so, we should be able to simplify this further.
    const SCEV *X = ST->getOperand();
    ConstantRange CR = getUnsignedRange(X);
    unsigned TruncBits = getTypeSizeInBits(ST->getType());
    unsigned NewBits = getTypeSizeInBits(Ty);
    if (CR.truncate(TruncBits).zeroExtend(NewBits).contains(
            CR.zextOrTrunc(NewBits)))
      return getTruncateOrZeroExtend(X, Ty);
  }

  // If the input value is a chrec scev, and we can prove that the value
  // did not overflow the old, smaller, value, we can zero extend all of the
  // operands (often constants).  This allows analysis of something like
  // this:  for (unsigned char X = 0; X < 100; ++X) { int Y = X; }
  if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(Op))
    if (AR->isAffine()) {
      const SCEV *Start = AR->getStart();
      const SCEV *Step = AR->getStepRecurrence(*this);
      unsigned BitWidth = getTypeSizeInBits(AR->getType());
      const Loop *L = AR->getLoop();

      // If we have special knowledge that this addrec won't overflow,
      // we don't need to do any further analysis.
      if (AR->getNoWrapFlags(SCEV::FlagNUW))
        return getAddRecExpr(
            getExtendAddRecStart<SCEVZeroExtendExpr>(AR, Ty, this),
            getZeroExtendExpr(Step, Ty), L, AR->getNoWrapFlags());

      // Check whether the backedge-taken count is SCEVCouldNotCompute.
      // Note that this serves two purposes: It filters out loops that are
      // simply not analyzable, and it covers the case where this code is
      // being called from within backedge-taken count analysis, such that
      // attempting to ask for the backedge-taken count would likely result
      // in infinite recursion. In the later case, the analysis code will
      // cope with a conservative value, and it will take care to purge
      // that value once it has finished.
      const SCEV *MaxBECount = getMaxBackedgeTakenCount(L);
      if (!isa<SCEVCouldNotCompute>(MaxBECount)) {
        // Manually compute the final value for AR, checking for
        // overflow.

        // Check whether the backedge-taken count can be losslessly casted to
        // the addrec's type. The count is always unsigned.
        const SCEV *CastedMaxBECount =
          getTruncateOrZeroExtend(MaxBECount, Start->getType());
        const SCEV *RecastedMaxBECount =
          getTruncateOrZeroExtend(CastedMaxBECount, MaxBECount->getType());
        if (MaxBECount == RecastedMaxBECount) {
          Type *WideTy = IntegerType::get(getContext(), BitWidth * 2);
          // Check whether Start+Step*MaxBECount has no unsigned overflow.
          const SCEV *ZMul = getMulExpr(CastedMaxBECount, Step);
          const SCEV *ZAdd = getZeroExtendExpr(getAddExpr(Start, ZMul), WideTy);
          const SCEV *WideStart = getZeroExtendExpr(Start, WideTy);
          const SCEV *WideMaxBECount =
            getZeroExtendExpr(CastedMaxBECount, WideTy);
          const SCEV *OperandExtendedAdd =
            getAddExpr(WideStart,
                       getMulExpr(WideMaxBECount,
                                  getZeroExtendExpr(Step, WideTy)));
          if (ZAdd == OperandExtendedAdd) {
            // Cache knowledge of AR NUW, which is propagated to this AddRec.
            const_cast<SCEVAddRecExpr *>(AR)->setNoWrapFlags(SCEV::FlagNUW);
            // Return the expression with the addrec on the outside.
            return getAddRecExpr(
                getExtendAddRecStart<SCEVZeroExtendExpr>(AR, Ty, this),
                getZeroExtendExpr(Step, Ty), L, AR->getNoWrapFlags());
          }
          // Similar to above, only this time treat the step value as signed.
          // This covers loops that count down.
          OperandExtendedAdd =
            getAddExpr(WideStart,
                       getMulExpr(WideMaxBECount,
                                  getSignExtendExpr(Step, WideTy)));
          if (ZAdd == OperandExtendedAdd) {
            // Cache knowledge of AR NW, which is propagated to this AddRec.
            // Negative step causes unsigned wrap, but it still can't self-wrap.
            const_cast<SCEVAddRecExpr *>(AR)->setNoWrapFlags(SCEV::FlagNW);
            // Return the expression with the addrec on the outside.
            return getAddRecExpr(
                getExtendAddRecStart<SCEVZeroExtendExpr>(AR, Ty, this),
                getSignExtendExpr(Step, Ty), L, AR->getNoWrapFlags());
          }
        }

        // If the backedge is guarded by a comparison with the pre-inc value
        // the addrec is safe. Also, if the entry is guarded by a comparison
        // with the start value and the backedge is guarded by a comparison
        // with the post-inc value, the addrec is safe.
        if (isKnownPositive(Step)) {
          const SCEV *N = getConstant(APInt::getMinValue(BitWidth) -
                                      getUnsignedRange(Step).getUnsignedMax());
          if (isLoopBackedgeGuardedByCond(L, ICmpInst::ICMP_ULT, AR, N) ||
              (isLoopEntryGuardedByCond(L, ICmpInst::ICMP_ULT, Start, N) &&
               isLoopBackedgeGuardedByCond(L, ICmpInst::ICMP_ULT,
                                           AR->getPostIncExpr(*this), N))) {
            // Cache knowledge of AR NUW, which is propagated to this AddRec.
            const_cast<SCEVAddRecExpr *>(AR)->setNoWrapFlags(SCEV::FlagNUW);
            // Return the expression with the addrec on the outside.
            return getAddRecExpr(
                getExtendAddRecStart<SCEVZeroExtendExpr>(AR, Ty, this),
                getZeroExtendExpr(Step, Ty), L, AR->getNoWrapFlags());
          }
        } else if (isKnownNegative(Step)) {
          const SCEV *N = getConstant(APInt::getMaxValue(BitWidth) -
                                      getSignedRange(Step).getSignedMin());
          if (isLoopBackedgeGuardedByCond(L, ICmpInst::ICMP_UGT, AR, N) ||
              (isLoopEntryGuardedByCond(L, ICmpInst::ICMP_UGT, Start, N) &&
               isLoopBackedgeGuardedByCond(L, ICmpInst::ICMP_UGT,
                                           AR->getPostIncExpr(*this), N))) {
            // Cache knowledge of AR NW, which is propagated to this AddRec.
            // Negative step causes unsigned wrap, but it still can't self-wrap.
            const_cast<SCEVAddRecExpr *>(AR)->setNoWrapFlags(SCEV::FlagNW);
            // Return the expression with the addrec on the outside.
            return getAddRecExpr(
                getExtendAddRecStart<SCEVZeroExtendExpr>(AR, Ty, this),
                getSignExtendExpr(Step, Ty), L, AR->getNoWrapFlags());
          }
        }
      }

      if (proveNoWrapByVaryingStart<SCEVZeroExtendExpr>(Start, Step, L)) {
        const_cast<SCEVAddRecExpr *>(AR)->setNoWrapFlags(SCEV::FlagNUW);
        return getAddRecExpr(
            getExtendAddRecStart<SCEVZeroExtendExpr>(AR, Ty, this),
            getZeroExtendExpr(Step, Ty), L, AR->getNoWrapFlags());
      }
    }

  // The cast wasn't folded; create an explicit cast node.
  // Recompute the insert position, as it may have been invalidated.
  if (const SCEV *S = UniqueSCEVs.FindNodeOrInsertPos(ID, IP)) return S;
  SCEV *S = new (SCEVAllocator) SCEVZeroExtendExpr(ID.Intern(SCEVAllocator),
                                                   Op, Ty);
  UniqueSCEVs.InsertNode(S, IP);
  return S;
}

const SCEV *ScalarEvolution::getSignExtendExpr(const SCEV *Op,
                                               Type *Ty) {
  assert(getTypeSizeInBits(Op->getType()) < getTypeSizeInBits(Ty) &&
         "This is not an extending conversion!");
  assert(isSCEVable(Ty) &&
         "This is not a conversion to a SCEVable type!");
  Ty = getEffectiveSCEVType(Ty);

  // Fold if the operand is constant.
  if (const SCEVConstant *SC = dyn_cast<SCEVConstant>(Op))
    return getConstant(
      cast<ConstantInt>(ConstantExpr::getSExt(SC->getValue(), Ty)));

  // sext(sext(x)) --> sext(x)
  if (const SCEVSignExtendExpr *SS = dyn_cast<SCEVSignExtendExpr>(Op))
    return getSignExtendExpr(SS->getOperand(), Ty);

  // sext(zext(x)) --> zext(x)
  if (const SCEVZeroExtendExpr *SZ = dyn_cast<SCEVZeroExtendExpr>(Op))
    return getZeroExtendExpr(SZ->getOperand(), Ty);

  // Before doing any expensive analysis, check to see if we've already
  // computed a SCEV for this Op and Ty.
  FoldingSetNodeID ID;
  ID.AddInteger(scSignExtend);
  ID.AddPointer(Op);
  ID.AddPointer(Ty);
  void *IP = nullptr;
  if (const SCEV *S = UniqueSCEVs.FindNodeOrInsertPos(ID, IP)) return S;

  // If the input value is provably positive, build a zext instead.
  if (isKnownNonNegative(Op))
    return getZeroExtendExpr(Op, Ty);

  // sext(trunc(x)) --> sext(x) or x or trunc(x)
  if (const SCEVTruncateExpr *ST = dyn_cast<SCEVTruncateExpr>(Op)) {
    // It's possible the bits taken off by the truncate were all sign bits. If
    // so, we should be able to simplify this further.
    const SCEV *X = ST->getOperand();
    ConstantRange CR = getSignedRange(X);
    unsigned TruncBits = getTypeSizeInBits(ST->getType());
    unsigned NewBits = getTypeSizeInBits(Ty);
    if (CR.truncate(TruncBits).signExtend(NewBits).contains(
            CR.sextOrTrunc(NewBits)))
      return getTruncateOrSignExtend(X, Ty);
  }

  // sext(C1 + (C2 * x)) --> C1 + sext(C2 * x) if C1 < C2
  if (auto SA = dyn_cast<SCEVAddExpr>(Op)) {
    if (SA->getNumOperands() == 2) {
      auto SC1 = dyn_cast<SCEVConstant>(SA->getOperand(0));
      auto SMul = dyn_cast<SCEVMulExpr>(SA->getOperand(1));
      if (SMul && SC1) {
        if (auto SC2 = dyn_cast<SCEVConstant>(SMul->getOperand(0))) {
          const APInt &C1 = SC1->getValue()->getValue();
          const APInt &C2 = SC2->getValue()->getValue();
          if (C1.isStrictlyPositive() && C2.isStrictlyPositive() &&
              C2.ugt(C1) && C2.isPowerOf2())
            return getAddExpr(getSignExtendExpr(SC1, Ty),
                              getSignExtendExpr(SMul, Ty));
        }
      }
    }
  }
  // If the input value is a chrec scev, and we can prove that the value
  // did not overflow the old, smaller, value, we can sign extend all of the
  // operands (often constants).  This allows analysis of something like
  // this:  for (signed char X = 0; X < 100; ++X) { int Y = X; }
  if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(Op))
    if (AR->isAffine()) {
      const SCEV *Start = AR->getStart();
      const SCEV *Step = AR->getStepRecurrence(*this);
      unsigned BitWidth = getTypeSizeInBits(AR->getType());
      const Loop *L = AR->getLoop();

      // If we have special knowledge that this addrec won't overflow,
      // we don't need to do any further analysis.
      if (AR->getNoWrapFlags(SCEV::FlagNSW))
        return getAddRecExpr(
            getExtendAddRecStart<SCEVSignExtendExpr>(AR, Ty, this),
            getSignExtendExpr(Step, Ty), L, SCEV::FlagNSW);

      // Check whether the backedge-taken count is SCEVCouldNotCompute.
      // Note that this serves two purposes: It filters out loops that are
      // simply not analyzable, and it covers the case where this code is
      // being called from within backedge-taken count analysis, such that
      // attempting to ask for the backedge-taken count would likely result
      // in infinite recursion. In the later case, the analysis code will
      // cope with a conservative value, and it will take care to purge
      // that value once it has finished.
      const SCEV *MaxBECount = getMaxBackedgeTakenCount(L);
      if (!isa<SCEVCouldNotCompute>(MaxBECount)) {
        // Manually compute the final value for AR, checking for
        // overflow.

        // Check whether the backedge-taken count can be losslessly casted to
        // the addrec's type. The count is always unsigned.
        const SCEV *CastedMaxBECount =
          getTruncateOrZeroExtend(MaxBECount, Start->getType());
        const SCEV *RecastedMaxBECount =
          getTruncateOrZeroExtend(CastedMaxBECount, MaxBECount->getType());
        if (MaxBECount == RecastedMaxBECount) {
          Type *WideTy = IntegerType::get(getContext(), BitWidth * 2);
          // Check whether Start+Step*MaxBECount has no signed overflow.
          const SCEV *SMul = getMulExpr(CastedMaxBECount, Step);
          const SCEV *SAdd = getSignExtendExpr(getAddExpr(Start, SMul), WideTy);
          const SCEV *WideStart = getSignExtendExpr(Start, WideTy);
          const SCEV *WideMaxBECount =
            getZeroExtendExpr(CastedMaxBECount, WideTy);
          const SCEV *OperandExtendedAdd =
            getAddExpr(WideStart,
                       getMulExpr(WideMaxBECount,
                                  getSignExtendExpr(Step, WideTy)));
          if (SAdd == OperandExtendedAdd) {
            // Cache knowledge of AR NSW, which is propagated to this AddRec.
            const_cast<SCEVAddRecExpr *>(AR)->setNoWrapFlags(SCEV::FlagNSW);
            // Return the expression with the addrec on the outside.
            return getAddRecExpr(
                getExtendAddRecStart<SCEVSignExtendExpr>(AR, Ty, this),
                getSignExtendExpr(Step, Ty), L, AR->getNoWrapFlags());
          }
          // Similar to above, only this time treat the step value as unsigned.
          // This covers loops that count up with an unsigned step.
          OperandExtendedAdd =
            getAddExpr(WideStart,
                       getMulExpr(WideMaxBECount,
                                  getZeroExtendExpr(Step, WideTy)));
          if (SAdd == OperandExtendedAdd) {
            // If AR wraps around then
            //
            //    abs(Step) * MaxBECount > unsigned-max(AR->getType())
            // => SAdd != OperandExtendedAdd
            //
            // Thus (AR is not NW => SAdd != OperandExtendedAdd) <=>
            // (SAdd == OperandExtendedAdd => AR is NW)

            const_cast<SCEVAddRecExpr *>(AR)->setNoWrapFlags(SCEV::FlagNW);

            // Return the expression with the addrec on the outside.
            return getAddRecExpr(
                getExtendAddRecStart<SCEVSignExtendExpr>(AR, Ty, this),
                getZeroExtendExpr(Step, Ty), L, AR->getNoWrapFlags());
          }
        }

        // If the backedge is guarded by a comparison with the pre-inc value
        // the addrec is safe. Also, if the entry is guarded by a comparison
        // with the start value and the backedge is guarded by a comparison
        // with the post-inc value, the addrec is safe.
        ICmpInst::Predicate Pred;
        const SCEV *OverflowLimit =
            getSignedOverflowLimitForStep(Step, &Pred, this);
        if (OverflowLimit &&
            (isLoopBackedgeGuardedByCond(L, Pred, AR, OverflowLimit) ||
             (isLoopEntryGuardedByCond(L, Pred, Start, OverflowLimit) &&
              isLoopBackedgeGuardedByCond(L, Pred, AR->getPostIncExpr(*this),
                                          OverflowLimit)))) {
          // Cache knowledge of AR NSW, then propagate NSW to the wide AddRec.
          const_cast<SCEVAddRecExpr *>(AR)->setNoWrapFlags(SCEV::FlagNSW);
          return getAddRecExpr(
              getExtendAddRecStart<SCEVSignExtendExpr>(AR, Ty, this),
              getSignExtendExpr(Step, Ty), L, AR->getNoWrapFlags());
        }
      }
      // If Start and Step are constants, check if we can apply this
      // transformation:
      // sext{C1,+,C2} --> C1 + sext{0,+,C2} if C1 < C2
      auto SC1 = dyn_cast<SCEVConstant>(Start);
      auto SC2 = dyn_cast<SCEVConstant>(Step);
      if (SC1 && SC2) {
        const APInt &C1 = SC1->getValue()->getValue();
        const APInt &C2 = SC2->getValue()->getValue();
        if (C1.isStrictlyPositive() && C2.isStrictlyPositive() && C2.ugt(C1) &&
            C2.isPowerOf2()) {
          Start = getSignExtendExpr(Start, Ty);
          const SCEV *NewAR = getAddRecExpr(getConstant(AR->getType(), 0), Step,
                                            L, AR->getNoWrapFlags());
          return getAddExpr(Start, getSignExtendExpr(NewAR, Ty));
        }
      }

      if (proveNoWrapByVaryingStart<SCEVSignExtendExpr>(Start, Step, L)) {
        const_cast<SCEVAddRecExpr *>(AR)->setNoWrapFlags(SCEV::FlagNSW);
        return getAddRecExpr(
            getExtendAddRecStart<SCEVSignExtendExpr>(AR, Ty, this),
            getSignExtendExpr(Step, Ty), L, AR->getNoWrapFlags());
      }
    }

  // The cast wasn't folded; create an explicit cast node.
  // Recompute the insert position, as it may have been invalidated.
  if (const SCEV *S = UniqueSCEVs.FindNodeOrInsertPos(ID, IP)) return S;
  SCEV *S = new (SCEVAllocator) SCEVSignExtendExpr(ID.Intern(SCEVAllocator),
                                                   Op, Ty);
  UniqueSCEVs.InsertNode(S, IP);
  return S;
}

/// getAnyExtendExpr - Return a SCEV for the given operand extended with
/// unspecified bits out to the given type.
///
const SCEV *ScalarEvolution::getAnyExtendExpr(const SCEV *Op,
                                              Type *Ty) {
  assert(getTypeSizeInBits(Op->getType()) < getTypeSizeInBits(Ty) &&
         "This is not an extending conversion!");
  assert(isSCEVable(Ty) &&
         "This is not a conversion to a SCEVable type!");
  Ty = getEffectiveSCEVType(Ty);

  // Sign-extend negative constants.
  if (const SCEVConstant *SC = dyn_cast<SCEVConstant>(Op))
    if (SC->getValue()->getValue().isNegative())
      return getSignExtendExpr(Op, Ty);

  // Peel off a truncate cast.
  if (const SCEVTruncateExpr *T = dyn_cast<SCEVTruncateExpr>(Op)) {
    const SCEV *NewOp = T->getOperand();
    if (getTypeSizeInBits(NewOp->getType()) < getTypeSizeInBits(Ty))
      return getAnyExtendExpr(NewOp, Ty);
    return getTruncateOrNoop(NewOp, Ty);
  }

  // Next try a zext cast. If the cast is folded, use it.
  const SCEV *ZExt = getZeroExtendExpr(Op, Ty);
  if (!isa<SCEVZeroExtendExpr>(ZExt))
    return ZExt;

  // Next try a sext cast. If the cast is folded, use it.
  const SCEV *SExt = getSignExtendExpr(Op, Ty);
  if (!isa<SCEVSignExtendExpr>(SExt))
    return SExt;

  // Force the cast to be folded into the operands of an addrec.
  if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(Op)) {
    SmallVector<const SCEV *, 4> Ops;
    for (const SCEV *Op : AR->operands())
      Ops.push_back(getAnyExtendExpr(Op, Ty));
    return getAddRecExpr(Ops, AR->getLoop(), SCEV::FlagNW);
  }

  // If the expression is obviously signed, use the sext cast value.
  if (isa<SCEVSMaxExpr>(Op))
    return SExt;

  // Absent any other information, use the zext cast value.
  return ZExt;
}

/// CollectAddOperandsWithScales - Process the given Ops list, which is
/// a list of operands to be added under the given scale, update the given
/// map. This is a helper function for getAddRecExpr. As an example of
/// what it does, given a sequence of operands that would form an add
/// expression like this:
///
///    m + n + 13 + (A * (o + p + (B * (q + m + 29)))) + r + (-1 * r)
///
/// where A and B are constants, update the map with these values:
///
///    (m, 1+A*B), (n, 1), (o, A), (p, A), (q, A*B), (r, 0)
///
/// and add 13 + A*B*29 to AccumulatedConstant.
/// This will allow getAddRecExpr to produce this:
///
///    13+A*B*29 + n + (m * (1+A*B)) + ((o + p) * A) + (q * A*B)
///
/// This form often exposes folding opportunities that are hidden in
/// the original operand list.
///
/// Return true iff it appears that any interesting folding opportunities
/// may be exposed. This helps getAddRecExpr short-circuit extra work in
/// the common case where no interesting opportunities are present, and
/// is also used as a check to avoid infinite recursion.
///
static bool
CollectAddOperandsWithScales(DenseMap<const SCEV *, APInt> &M,
                             SmallVectorImpl<const SCEV *> &NewOps,
                             APInt &AccumulatedConstant,
                             const SCEV *const *Ops, size_t NumOperands,
                             const APInt &Scale,
                             ScalarEvolution &SE) {
  bool Interesting = false;

  // Iterate over the add operands. They are sorted, with constants first.
  unsigned i = 0;
  while (const SCEVConstant *C = dyn_cast<SCEVConstant>(Ops[i])) {
    ++i;
    // Pull a buried constant out to the outside.
    if (Scale != 1 || AccumulatedConstant != 0 || C->getValue()->isZero())
      Interesting = true;
    AccumulatedConstant += Scale * C->getValue()->getValue();
  }

  // Next comes everything else. We're especially interested in multiplies
  // here, but they're in the middle, so just visit the rest with one loop.
  for (; i != NumOperands; ++i) {
    const SCEVMulExpr *Mul = dyn_cast<SCEVMulExpr>(Ops[i]);
    if (Mul && isa<SCEVConstant>(Mul->getOperand(0))) {
      APInt NewScale =
        Scale * cast<SCEVConstant>(Mul->getOperand(0))->getValue()->getValue();
      if (Mul->getNumOperands() == 2 && isa<SCEVAddExpr>(Mul->getOperand(1))) {
        // A multiplication of a constant with another add; recurse.
        const SCEVAddExpr *Add = cast<SCEVAddExpr>(Mul->getOperand(1));
        Interesting |=
          CollectAddOperandsWithScales(M, NewOps, AccumulatedConstant,
                                       Add->op_begin(), Add->getNumOperands(),
                                       NewScale, SE);
      } else {
        // A multiplication of a constant with some other value. Update
        // the map.
        SmallVector<const SCEV *, 4> MulOps(Mul->op_begin()+1, Mul->op_end());
        const SCEV *Key = SE.getMulExpr(MulOps);
        std::pair<DenseMap<const SCEV *, APInt>::iterator, bool> Pair =
          M.insert(std::make_pair(Key, NewScale));
        if (Pair.second) {
          NewOps.push_back(Pair.first->first);
        } else {
          Pair.first->second += NewScale;
          // The map already had an entry for this value, which may indicate
          // a folding opportunity.
          Interesting = true;
        }
      }
    } else {
      // An ordinary operand. Update the map.
      std::pair<DenseMap<const SCEV *, APInt>::iterator, bool> Pair =
        M.insert(std::make_pair(Ops[i], Scale));
      if (Pair.second) {
        NewOps.push_back(Pair.first->first);
      } else {
        Pair.first->second += Scale;
        // The map already had an entry for this value, which may indicate
        // a folding opportunity.
        Interesting = true;
      }
    }
  }

  return Interesting;
}

namespace {
  struct APIntCompare {
    bool operator()(const APInt &LHS, const APInt &RHS) const {
      return LHS.ult(RHS);
    }
  };
}

// We're trying to construct a SCEV of type `Type' with `Ops' as operands and
// `OldFlags' as can't-wrap behavior.  Infer a more aggressive set of
// can't-overflow flags for the operation if possible.
static SCEV::NoWrapFlags
StrengthenNoWrapFlags(ScalarEvolution *SE, SCEVTypes Type,
                      const SmallVectorImpl<const SCEV *> &Ops,
                      SCEV::NoWrapFlags OldFlags) {
  using namespace std::placeholders;

  bool CanAnalyze =
      Type == scAddExpr || Type == scAddRecExpr || Type == scMulExpr;
  (void)CanAnalyze;
  assert(CanAnalyze && "don't call from other places!");

  int SignOrUnsignMask = SCEV::FlagNUW | SCEV::FlagNSW;
  SCEV::NoWrapFlags SignOrUnsignWrap =
      ScalarEvolution::maskFlags(OldFlags, SignOrUnsignMask);

  // If FlagNSW is true and all the operands are non-negative, infer FlagNUW.
  auto IsKnownNonNegative =
    std::bind(std::mem_fn(&ScalarEvolution::isKnownNonNegative), SE, _1);

  if (SignOrUnsignWrap == SCEV::FlagNSW &&
      std::all_of(Ops.begin(), Ops.end(), IsKnownNonNegative))
    return ScalarEvolution::setFlags(OldFlags,
                                     (SCEV::NoWrapFlags)SignOrUnsignMask);

  return OldFlags;
}

/// getAddExpr - Get a canonical add expression, or something simpler if
/// possible.
const SCEV *ScalarEvolution::getAddExpr(SmallVectorImpl<const SCEV *> &Ops,
                                        SCEV::NoWrapFlags Flags) {
  assert(!(Flags & ~(SCEV::FlagNUW | SCEV::FlagNSW)) &&
         "only nuw or nsw allowed");
  assert(!Ops.empty() && "Cannot get empty add!");
  if (Ops.size() == 1) return Ops[0];
#ifndef NDEBUG
  Type *ETy = getEffectiveSCEVType(Ops[0]->getType());
  for (unsigned i = 1, e = Ops.size(); i != e; ++i)
    assert(getEffectiveSCEVType(Ops[i]->getType()) == ETy &&
           "SCEVAddExpr operand types don't match!");
#endif

  Flags = StrengthenNoWrapFlags(this, scAddExpr, Ops, Flags);

  // Sort by complexity, this groups all similar expression types together.
  GroupByComplexity(Ops, LI);

  // If there are any constants, fold them together.
  unsigned Idx = 0;
  if (const SCEVConstant *LHSC = dyn_cast<SCEVConstant>(Ops[0])) {
    ++Idx;
    assert(Idx < Ops.size());
    while (const SCEVConstant *RHSC = dyn_cast<SCEVConstant>(Ops[Idx])) {
      // We found two constants, fold them together!
      Ops[0] = getConstant(LHSC->getValue()->getValue() +
                           RHSC->getValue()->getValue());
      if (Ops.size() == 2) return Ops[0];
      Ops.erase(Ops.begin()+1);  // Erase the folded element
      LHSC = cast<SCEVConstant>(Ops[0]);
    }

    // If we are left with a constant zero being added, strip it off.
    if (LHSC->getValue()->isZero()) {
      Ops.erase(Ops.begin());
      --Idx;
    }

    if (Ops.size() == 1) return Ops[0];
  }

  // Okay, check to see if the same value occurs in the operand list more than
  // once.  If so, merge them together into an multiply expression.  Since we
  // sorted the list, these values are required to be adjacent.
  Type *Ty = Ops[0]->getType();
  bool FoundMatch = false;
  for (unsigned i = 0, e = Ops.size(); i != e-1; ++i)
    if (Ops[i] == Ops[i+1]) {      //  X + Y + Y  -->  X + Y*2
      // Scan ahead to count how many equal operands there are.
      unsigned Count = 2;
      while (i+Count != e && Ops[i+Count] == Ops[i])
        ++Count;
      // Merge the values into a multiply.
      const SCEV *Scale = getConstant(Ty, Count);
      const SCEV *Mul = getMulExpr(Scale, Ops[i]);
      if (Ops.size() == Count)
        return Mul;
      Ops[i] = Mul;
      Ops.erase(Ops.begin()+i+1, Ops.begin()+i+Count);
      --i; e -= Count - 1;
      FoundMatch = true;
    }
  if (FoundMatch)
    return getAddExpr(Ops, Flags);

  // Check for truncates. If all the operands are truncated from the same
  // type, see if factoring out the truncate would permit the result to be
  // folded. eg., trunc(x) + m*trunc(n) --> trunc(x + trunc(m)*n)
  // if the contents of the resulting outer trunc fold to something simple.
  for (; Idx < Ops.size() && isa<SCEVTruncateExpr>(Ops[Idx]); ++Idx) {
    const SCEVTruncateExpr *Trunc = cast<SCEVTruncateExpr>(Ops[Idx]);
    Type *DstType = Trunc->getType();
    Type *SrcType = Trunc->getOperand()->getType();
    SmallVector<const SCEV *, 8> LargeOps;
    bool Ok = true;
    // Check all the operands to see if they can be represented in the
    // source type of the truncate.
    for (unsigned i = 0, e = Ops.size(); i != e; ++i) {
      if (const SCEVTruncateExpr *T = dyn_cast<SCEVTruncateExpr>(Ops[i])) {
        if (T->getOperand()->getType() != SrcType) {
          Ok = false;
          break;
        }
        LargeOps.push_back(T->getOperand());
      } else if (const SCEVConstant *C = dyn_cast<SCEVConstant>(Ops[i])) {
        LargeOps.push_back(getAnyExtendExpr(C, SrcType));
      } else if (const SCEVMulExpr *M = dyn_cast<SCEVMulExpr>(Ops[i])) {
        SmallVector<const SCEV *, 8> LargeMulOps;
        for (unsigned j = 0, f = M->getNumOperands(); j != f && Ok; ++j) {
          if (const SCEVTruncateExpr *T =
                dyn_cast<SCEVTruncateExpr>(M->getOperand(j))) {
            if (T->getOperand()->getType() != SrcType) {
              Ok = false;
              break;
            }
            LargeMulOps.push_back(T->getOperand());
          } else if (const SCEVConstant *C =
                       dyn_cast<SCEVConstant>(M->getOperand(j))) {
            LargeMulOps.push_back(getAnyExtendExpr(C, SrcType));
          } else {
            Ok = false;
            break;
          }
        }
        if (Ok)
          LargeOps.push_back(getMulExpr(LargeMulOps));
      } else {
        Ok = false;
        break;
      }
    }
    if (Ok) {
      // Evaluate the expression in the larger type.
      const SCEV *Fold = getAddExpr(LargeOps, Flags);
      // If it folds to something simple, use it. Otherwise, don't.
      if (isa<SCEVConstant>(Fold) || isa<SCEVUnknown>(Fold))
        return getTruncateExpr(Fold, DstType);
    }
  }

  // Skip past any other cast SCEVs.
  while (Idx < Ops.size() && Ops[Idx]->getSCEVType() < scAddExpr)
    ++Idx;

  // If there are add operands they would be next.
  if (Idx < Ops.size()) {
    bool DeletedAdd = false;
    while (const SCEVAddExpr *Add = dyn_cast<SCEVAddExpr>(Ops[Idx])) {
      // If we have an add, expand the add operands onto the end of the operands
      // list.
      Ops.erase(Ops.begin()+Idx);
      Ops.append(Add->op_begin(), Add->op_end());
      DeletedAdd = true;
    }

    // If we deleted at least one add, we added operands to the end of the list,
    // and they are not necessarily sorted.  Recurse to resort and resimplify
    // any operands we just acquired.
    if (DeletedAdd)
      return getAddExpr(Ops);
  }

  // Skip over the add expression until we get to a multiply.
  while (Idx < Ops.size() && Ops[Idx]->getSCEVType() < scMulExpr)
    ++Idx;

  // Check to see if there are any folding opportunities present with
  // operands multiplied by constant values.
  if (Idx < Ops.size() && isa<SCEVMulExpr>(Ops[Idx])) {
    uint64_t BitWidth = getTypeSizeInBits(Ty);
    DenseMap<const SCEV *, APInt> M;
    SmallVector<const SCEV *, 8> NewOps;
    APInt AccumulatedConstant(BitWidth, 0);
    if (CollectAddOperandsWithScales(M, NewOps, AccumulatedConstant,
                                     Ops.data(), Ops.size(),
                                     APInt(BitWidth, 1), *this)) {
      // Some interesting folding opportunity is present, so its worthwhile to
      // re-generate the operands list. Group the operands by constant scale,
      // to avoid multiplying by the same constant scale multiple times.
      std::map<APInt, SmallVector<const SCEV *, 4>, APIntCompare> MulOpLists;
      for (SmallVectorImpl<const SCEV *>::const_iterator I = NewOps.begin(),
           E = NewOps.end(); I != E; ++I)
        MulOpLists[M.find(*I)->second].push_back(*I);
      // Re-generate the operands list.
      Ops.clear();
      if (AccumulatedConstant != 0)
        Ops.push_back(getConstant(AccumulatedConstant));
      for (std::map<APInt, SmallVector<const SCEV *, 4>, APIntCompare>::iterator
           I = MulOpLists.begin(), E = MulOpLists.end(); I != E; ++I)
        if (I->first != 0)
          Ops.push_back(getMulExpr(getConstant(I->first),
                                   getAddExpr(I->second)));
      if (Ops.empty())
        return getConstant(Ty, 0);
      if (Ops.size() == 1)
        return Ops[0];
      return getAddExpr(Ops);
    }
  }

  // If we are adding something to a multiply expression, make sure the
  // something is not already an operand of the multiply.  If so, merge it into
  // the multiply.
  for (; Idx < Ops.size() && isa<SCEVMulExpr>(Ops[Idx]); ++Idx) {
    const SCEVMulExpr *Mul = cast<SCEVMulExpr>(Ops[Idx]);
    for (unsigned MulOp = 0, e = Mul->getNumOperands(); MulOp != e; ++MulOp) {
      const SCEV *MulOpSCEV = Mul->getOperand(MulOp);
      if (isa<SCEVConstant>(MulOpSCEV))
        continue;
      for (unsigned AddOp = 0, e = Ops.size(); AddOp != e; ++AddOp)
        if (MulOpSCEV == Ops[AddOp]) {
          // Fold W + X + (X * Y * Z)  -->  W + (X * ((Y*Z)+1))
          const SCEV *InnerMul = Mul->getOperand(MulOp == 0);
          if (Mul->getNumOperands() != 2) {
            // If the multiply has more than two operands, we must get the
            // Y*Z term.
            SmallVector<const SCEV *, 4> MulOps(Mul->op_begin(),
                                                Mul->op_begin()+MulOp);
            MulOps.append(Mul->op_begin()+MulOp+1, Mul->op_end());
            InnerMul = getMulExpr(MulOps);
          }
          const SCEV *One = getConstant(Ty, 1);
          const SCEV *AddOne = getAddExpr(One, InnerMul);
          const SCEV *OuterMul = getMulExpr(AddOne, MulOpSCEV);
          if (Ops.size() == 2) return OuterMul;
          if (AddOp < Idx) {
            Ops.erase(Ops.begin()+AddOp);
            Ops.erase(Ops.begin()+Idx-1);
          } else {
            Ops.erase(Ops.begin()+Idx);
            Ops.erase(Ops.begin()+AddOp-1);
          }
          Ops.push_back(OuterMul);
          return getAddExpr(Ops);
        }

      // Check this multiply against other multiplies being added together.
      for (unsigned OtherMulIdx = Idx+1;
           OtherMulIdx < Ops.size() && isa<SCEVMulExpr>(Ops[OtherMulIdx]);
           ++OtherMulIdx) {
        const SCEVMulExpr *OtherMul = cast<SCEVMulExpr>(Ops[OtherMulIdx]);
        // If MulOp occurs in OtherMul, we can fold the two multiplies
        // together.
        for (unsigned OMulOp = 0, e = OtherMul->getNumOperands();
             OMulOp != e; ++OMulOp)
          if (OtherMul->getOperand(OMulOp) == MulOpSCEV) {
            // Fold X + (A*B*C) + (A*D*E) --> X + (A*(B*C+D*E))
            const SCEV *InnerMul1 = Mul->getOperand(MulOp == 0);
            if (Mul->getNumOperands() != 2) {
              SmallVector<const SCEV *, 4> MulOps(Mul->op_begin(),
                                                  Mul->op_begin()+MulOp);
              MulOps.append(Mul->op_begin()+MulOp+1, Mul->op_end());
              InnerMul1 = getMulExpr(MulOps);
            }
            const SCEV *InnerMul2 = OtherMul->getOperand(OMulOp == 0);
            if (OtherMul->getNumOperands() != 2) {
              SmallVector<const SCEV *, 4> MulOps(OtherMul->op_begin(),
                                                  OtherMul->op_begin()+OMulOp);
              MulOps.append(OtherMul->op_begin()+OMulOp+1, OtherMul->op_end());
              InnerMul2 = getMulExpr(MulOps);
            }
            const SCEV *InnerMulSum = getAddExpr(InnerMul1,InnerMul2);
            const SCEV *OuterMul = getMulExpr(MulOpSCEV, InnerMulSum);
            if (Ops.size() == 2) return OuterMul;
            Ops.erase(Ops.begin()+Idx);
            Ops.erase(Ops.begin()+OtherMulIdx-1);
            Ops.push_back(OuterMul);
            return getAddExpr(Ops);
          }
      }
    }
  }

  // If there are any add recurrences in the operands list, see if any other
  // added values are loop invariant.  If so, we can fold them into the
  // recurrence.
  while (Idx < Ops.size() && Ops[Idx]->getSCEVType() < scAddRecExpr)
    ++Idx;

  // Scan over all recurrences, trying to fold loop invariants into them.
  for (; Idx < Ops.size() && isa<SCEVAddRecExpr>(Ops[Idx]); ++Idx) {
    // Scan all of the other operands to this add and add them to the vector if
    // they are loop invariant w.r.t. the recurrence.
    SmallVector<const SCEV *, 8> LIOps;
    const SCEVAddRecExpr *AddRec = cast<SCEVAddRecExpr>(Ops[Idx]);
    const Loop *AddRecLoop = AddRec->getLoop();
    for (unsigned i = 0, e = Ops.size(); i != e; ++i)
      if (isLoopInvariant(Ops[i], AddRecLoop)) {
        LIOps.push_back(Ops[i]);
        Ops.erase(Ops.begin()+i);
        --i; --e;
      }

    // If we found some loop invariants, fold them into the recurrence.
    if (!LIOps.empty()) {
      //  NLI + LI + {Start,+,Step}  -->  NLI + {LI+Start,+,Step}
      LIOps.push_back(AddRec->getStart());

      SmallVector<const SCEV *, 4> AddRecOps(AddRec->op_begin(),
                                             AddRec->op_end());
      AddRecOps[0] = getAddExpr(LIOps);

      // Build the new addrec. Propagate the NUW and NSW flags if both the
      // outer add and the inner addrec are guaranteed to have no overflow.
      // Always propagate NW.
      Flags = AddRec->getNoWrapFlags(setFlags(Flags, SCEV::FlagNW));
      const SCEV *NewRec = getAddRecExpr(AddRecOps, AddRecLoop, Flags);

      // If all of the other operands were loop invariant, we are done.
      if (Ops.size() == 1) return NewRec;

      // Otherwise, add the folded AddRec by the non-invariant parts.
      for (unsigned i = 0;; ++i)
        if (Ops[i] == AddRec) {
          Ops[i] = NewRec;
          break;
        }
      return getAddExpr(Ops);
    }

    // Okay, if there weren't any loop invariants to be folded, check to see if
    // there are multiple AddRec's with the same loop induction variable being
    // added together.  If so, we can fold them.
    for (unsigned OtherIdx = Idx+1;
         OtherIdx < Ops.size() && isa<SCEVAddRecExpr>(Ops[OtherIdx]);
         ++OtherIdx)
      if (AddRecLoop == cast<SCEVAddRecExpr>(Ops[OtherIdx])->getLoop()) {
        // Other + {A,+,B}<L> + {C,+,D}<L>  -->  Other + {A+C,+,B+D}<L>
        SmallVector<const SCEV *, 4> AddRecOps(AddRec->op_begin(),
                                               AddRec->op_end());
        for (; OtherIdx != Ops.size() && isa<SCEVAddRecExpr>(Ops[OtherIdx]);
             ++OtherIdx)
          if (const SCEVAddRecExpr *OtherAddRec =
                dyn_cast<SCEVAddRecExpr>(Ops[OtherIdx]))
            if (OtherAddRec->getLoop() == AddRecLoop) {
              for (unsigned i = 0, e = OtherAddRec->getNumOperands();
                   i != e; ++i) {
                if (i >= AddRecOps.size()) {
                  AddRecOps.append(OtherAddRec->op_begin()+i,
                                   OtherAddRec->op_end());
                  break;
                }
                AddRecOps[i] = getAddExpr(AddRecOps[i],
                                          OtherAddRec->getOperand(i));
              }
              Ops.erase(Ops.begin() + OtherIdx); --OtherIdx;
            }
        // Step size has changed, so we cannot guarantee no self-wraparound.
        Ops[Idx] = getAddRecExpr(AddRecOps, AddRecLoop, SCEV::FlagAnyWrap);
        return getAddExpr(Ops);
      }

    // Otherwise couldn't fold anything into this recurrence.  Move onto the
    // next one.
  }

  // Okay, it looks like we really DO need an add expr.  Check to see if we
  // already have one, otherwise create a new one.
  FoldingSetNodeID ID;
  ID.AddInteger(scAddExpr);
  for (unsigned i = 0, e = Ops.size(); i != e; ++i)
    ID.AddPointer(Ops[i]);
  void *IP = nullptr;
  SCEVAddExpr *S =
    static_cast<SCEVAddExpr *>(UniqueSCEVs.FindNodeOrInsertPos(ID, IP));
  if (!S) {
    const SCEV **O = SCEVAllocator.Allocate<const SCEV *>(Ops.size());
    std::uninitialized_copy(Ops.begin(), Ops.end(), O);
    S = new (SCEVAllocator) SCEVAddExpr(ID.Intern(SCEVAllocator),
                                        O, Ops.size());
    UniqueSCEVs.InsertNode(S, IP);
  }
  S->setNoWrapFlags(Flags);
  return S;
}

static uint64_t umul_ov(uint64_t i, uint64_t j, bool &Overflow) {
  uint64_t k = i*j;
  if (j > 1 && k / j != i) Overflow = true;
  return k;
}

/// Compute the result of "n choose k", the binomial coefficient.  If an
/// intermediate computation overflows, Overflow will be set and the return will
/// be garbage. Overflow is not cleared on absence of overflow.
static uint64_t Choose(uint64_t n, uint64_t k, bool &Overflow) {
  // We use the multiplicative formula:
  //     n(n-1)(n-2)...(n-(k-1)) / k(k-1)(k-2)...1 .
  // At each iteration, we take the n-th term of the numeral and divide by the
  // (k-n)th term of the denominator.  This division will always produce an
  // integral result, and helps reduce the chance of overflow in the
  // intermediate computations. However, we can still overflow even when the
  // final result would fit.

  if (n == 0 || n == k) return 1;
  if (k > n) return 0;

  if (k > n/2)
    k = n-k;

  uint64_t r = 1;
  for (uint64_t i = 1; i <= k; ++i) {
    r = umul_ov(r, n-(i-1), Overflow);
    r /= i;
  }
  return r;
}

/// Determine if any of the operands in this SCEV are a constant or if
/// any of the add or multiply expressions in this SCEV contain a constant.
static bool containsConstantSomewhere(const SCEV *StartExpr) {
  SmallVector<const SCEV *, 4> Ops;
  Ops.push_back(StartExpr);
  while (!Ops.empty()) {
    const SCEV *CurrentExpr = Ops.pop_back_val();
    if (isa<SCEVConstant>(*CurrentExpr))
      return true;

    if (isa<SCEVAddExpr>(*CurrentExpr) || isa<SCEVMulExpr>(*CurrentExpr)) {
      const auto *CurrentNAry = cast<SCEVNAryExpr>(CurrentExpr);
      Ops.append(CurrentNAry->op_begin(), CurrentNAry->op_end());
    }
  }
  return false;
}

/// getMulExpr - Get a canonical multiply expression, or something simpler if
/// possible.
const SCEV *ScalarEvolution::getMulExpr(SmallVectorImpl<const SCEV *> &Ops,
                                        SCEV::NoWrapFlags Flags) {
  assert(Flags == maskFlags(Flags, SCEV::FlagNUW | SCEV::FlagNSW) &&
         "only nuw or nsw allowed");
  assert(!Ops.empty() && "Cannot get empty mul!");
  if (Ops.size() == 1) return Ops[0];
#ifndef NDEBUG
  Type *ETy = getEffectiveSCEVType(Ops[0]->getType());
  for (unsigned i = 1, e = Ops.size(); i != e; ++i)
    assert(getEffectiveSCEVType(Ops[i]->getType()) == ETy &&
           "SCEVMulExpr operand types don't match!");
#endif

  Flags = StrengthenNoWrapFlags(this, scMulExpr, Ops, Flags);

  // Sort by complexity, this groups all similar expression types together.
  GroupByComplexity(Ops, LI);

  // If there are any constants, fold them together.
  unsigned Idx = 0;
  if (const SCEVConstant *LHSC = dyn_cast<SCEVConstant>(Ops[0])) {

    // C1*(C2+V) -> C1*C2 + C1*V
    if (Ops.size() == 2)
        if (const SCEVAddExpr *Add = dyn_cast<SCEVAddExpr>(Ops[1]))
          // If any of Add's ops are Adds or Muls with a constant,
          // apply this transformation as well.
          if (Add->getNumOperands() == 2)
            if (containsConstantSomewhere(Add))
              return getAddExpr(getMulExpr(LHSC, Add->getOperand(0)),
                                getMulExpr(LHSC, Add->getOperand(1)));

    ++Idx;
    while (const SCEVConstant *RHSC = dyn_cast<SCEVConstant>(Ops[Idx])) {
      // We found two constants, fold them together!
      ConstantInt *Fold = ConstantInt::get(getContext(),
                                           LHSC->getValue()->getValue() *
                                           RHSC->getValue()->getValue());
      Ops[0] = getConstant(Fold);
      Ops.erase(Ops.begin()+1);  // Erase the folded element
      if (Ops.size() == 1) return Ops[0];
      LHSC = cast<SCEVConstant>(Ops[0]);
    }

    // If we are left with a constant one being multiplied, strip it off.
    if (cast<SCEVConstant>(Ops[0])->getValue()->equalsInt(1)) {
      Ops.erase(Ops.begin());
      --Idx;
    } else if (cast<SCEVConstant>(Ops[0])->getValue()->isZero()) {
      // If we have a multiply of zero, it will always be zero.
      return Ops[0];
    } else if (Ops[0]->isAllOnesValue()) {
      // If we have a mul by -1 of an add, try distributing the -1 among the
      // add operands.
      if (Ops.size() == 2) {
        if (const SCEVAddExpr *Add = dyn_cast<SCEVAddExpr>(Ops[1])) {
          SmallVector<const SCEV *, 4> NewOps;
          bool AnyFolded = false;
          for (SCEVAddRecExpr::op_iterator I = Add->op_begin(),
                 E = Add->op_end(); I != E; ++I) {
            const SCEV *Mul = getMulExpr(Ops[0], *I);
            if (!isa<SCEVMulExpr>(Mul)) AnyFolded = true;
            NewOps.push_back(Mul);
          }
          if (AnyFolded)
            return getAddExpr(NewOps);
        }
        else if (const SCEVAddRecExpr *
                 AddRec = dyn_cast<SCEVAddRecExpr>(Ops[1])) {
          // Negation preserves a recurrence's no self-wrap property.
          SmallVector<const SCEV *, 4> Operands;
          for (SCEVAddRecExpr::op_iterator I = AddRec->op_begin(),
                 E = AddRec->op_end(); I != E; ++I) {
            Operands.push_back(getMulExpr(Ops[0], *I));
          }
          return getAddRecExpr(Operands, AddRec->getLoop(),
                               AddRec->getNoWrapFlags(SCEV::FlagNW));
        }
      }
    }

    if (Ops.size() == 1)
      return Ops[0];
  }

  // Skip over the add expression until we get to a multiply.
  while (Idx < Ops.size() && Ops[Idx]->getSCEVType() < scMulExpr)
    ++Idx;

  // If there are mul operands inline them all into this expression.
  if (Idx < Ops.size()) {
    bool DeletedMul = false;
    while (const SCEVMulExpr *Mul = dyn_cast<SCEVMulExpr>(Ops[Idx])) {
      // If we have an mul, expand the mul operands onto the end of the operands
      // list.
      Ops.erase(Ops.begin()+Idx);
      Ops.append(Mul->op_begin(), Mul->op_end());
      DeletedMul = true;
    }

    // If we deleted at least one mul, we added operands to the end of the list,
    // and they are not necessarily sorted.  Recurse to resort and resimplify
    // any operands we just acquired.
    if (DeletedMul)
      return getMulExpr(Ops);
  }

  // If there are any add recurrences in the operands list, see if any other
  // added values are loop invariant.  If so, we can fold them into the
  // recurrence.
  while (Idx < Ops.size() && Ops[Idx]->getSCEVType() < scAddRecExpr)
    ++Idx;

  // Scan over all recurrences, trying to fold loop invariants into them.
  for (; Idx < Ops.size() && isa<SCEVAddRecExpr>(Ops[Idx]); ++Idx) {
    // Scan all of the other operands to this mul and add them to the vector if
    // they are loop invariant w.r.t. the recurrence.
    SmallVector<const SCEV *, 8> LIOps;
    const SCEVAddRecExpr *AddRec = cast<SCEVAddRecExpr>(Ops[Idx]);
    const Loop *AddRecLoop = AddRec->getLoop();
    for (unsigned i = 0, e = Ops.size(); i != e; ++i)
      if (isLoopInvariant(Ops[i], AddRecLoop)) {
        LIOps.push_back(Ops[i]);
        Ops.erase(Ops.begin()+i);
        --i; --e;
      }

    // If we found some loop invariants, fold them into the recurrence.
    if (!LIOps.empty()) {
      //  NLI * LI * {Start,+,Step}  -->  NLI * {LI*Start,+,LI*Step}
      SmallVector<const SCEV *, 4> NewOps;
      NewOps.reserve(AddRec->getNumOperands());
      const SCEV *Scale = getMulExpr(LIOps);
      for (unsigned i = 0, e = AddRec->getNumOperands(); i != e; ++i)
        NewOps.push_back(getMulExpr(Scale, AddRec->getOperand(i)));

      // Build the new addrec. Propagate the NUW and NSW flags if both the
      // outer mul and the inner addrec are guaranteed to have no overflow.
      //
      // No self-wrap cannot be guaranteed after changing the step size, but
      // will be inferred if either NUW or NSW is true.
      Flags = AddRec->getNoWrapFlags(clearFlags(Flags, SCEV::FlagNW));
      const SCEV *NewRec = getAddRecExpr(NewOps, AddRecLoop, Flags);

      // If all of the other operands were loop invariant, we are done.
      if (Ops.size() == 1) return NewRec;

      // Otherwise, multiply the folded AddRec by the non-invariant parts.
      for (unsigned i = 0;; ++i)
        if (Ops[i] == AddRec) {
          Ops[i] = NewRec;
          break;
        }
      return getMulExpr(Ops);
    }

    // Okay, if there weren't any loop invariants to be folded, check to see if
    // there are multiple AddRec's with the same loop induction variable being
    // multiplied together.  If so, we can fold them.

    // {A1,+,A2,+,...,+,An}<L> * {B1,+,B2,+,...,+,Bn}<L>
    // = {x=1 in [ sum y=x..2x [ sum z=max(y-x, y-n)..min(x,n) [
    //       choose(x, 2x)*choose(2x-y, x-z)*A_{y-z}*B_z
    //   ]]],+,...up to x=2n}.
    // Note that the arguments to choose() are always integers with values
    // known at compile time, never SCEV objects.
    //
    // The implementation avoids pointless extra computations when the two
    // addrec's are of different length (mathematically, it's equivalent to
    // an infinite stream of zeros on the right).
    bool OpsModified = false;
    for (unsigned OtherIdx = Idx+1;
         OtherIdx != Ops.size() && isa<SCEVAddRecExpr>(Ops[OtherIdx]);
         ++OtherIdx) {
      const SCEVAddRecExpr *OtherAddRec =
        dyn_cast<SCEVAddRecExpr>(Ops[OtherIdx]);
      if (!OtherAddRec || OtherAddRec->getLoop() != AddRecLoop)
        continue;

      bool Overflow = false;
      Type *Ty = AddRec->getType();
      bool LargerThan64Bits = getTypeSizeInBits(Ty) > 64;
      SmallVector<const SCEV*, 7> AddRecOps;
      for (int x = 0, xe = AddRec->getNumOperands() +
             OtherAddRec->getNumOperands() - 1; x != xe && !Overflow; ++x) {
        const SCEV *Term = getConstant(Ty, 0);
        for (int y = x, ye = 2*x+1; y != ye && !Overflow; ++y) {
          uint64_t Coeff1 = Choose(x, 2*x - y, Overflow);
          for (int z = std::max(y-x, y-(int)AddRec->getNumOperands()+1),
                 ze = std::min(x+1, (int)OtherAddRec->getNumOperands());
               z < ze && !Overflow; ++z) {
            uint64_t Coeff2 = Choose(2*x - y, x-z, Overflow);
            uint64_t Coeff;
            if (LargerThan64Bits)
              Coeff = umul_ov(Coeff1, Coeff2, Overflow);
            else
              Coeff = Coeff1*Coeff2;
            const SCEV *CoeffTerm = getConstant(Ty, Coeff);
            const SCEV *Term1 = AddRec->getOperand(y-z);
            const SCEV *Term2 = OtherAddRec->getOperand(z);
            Term = getAddExpr(Term, getMulExpr(CoeffTerm, Term1,Term2));
          }
        }
        AddRecOps.push_back(Term);
      }
      if (!Overflow) {
        const SCEV *NewAddRec = getAddRecExpr(AddRecOps, AddRec->getLoop(),
                                              SCEV::FlagAnyWrap);
        if (Ops.size() == 2) return NewAddRec;
        Ops[Idx] = NewAddRec;
        Ops.erase(Ops.begin() + OtherIdx); --OtherIdx;
        OpsModified = true;
        AddRec = dyn_cast<SCEVAddRecExpr>(NewAddRec);
        if (!AddRec)
          break;
      }
    }
    if (OpsModified)
      return getMulExpr(Ops);

    // Otherwise couldn't fold anything into this recurrence.  Move onto the
    // next one.
  }

  // Okay, it looks like we really DO need an mul expr.  Check to see if we
  // already have one, otherwise create a new one.
  FoldingSetNodeID ID;
  ID.AddInteger(scMulExpr);
  for (unsigned i = 0, e = Ops.size(); i != e; ++i)
    ID.AddPointer(Ops[i]);
  void *IP = nullptr;
  SCEVMulExpr *S =
    static_cast<SCEVMulExpr *>(UniqueSCEVs.FindNodeOrInsertPos(ID, IP));
  if (!S) {
    const SCEV **O = SCEVAllocator.Allocate<const SCEV *>(Ops.size());
    std::uninitialized_copy(Ops.begin(), Ops.end(), O);
    S = new (SCEVAllocator) SCEVMulExpr(ID.Intern(SCEVAllocator),
                                        O, Ops.size());
    UniqueSCEVs.InsertNode(S, IP);
  }
  S->setNoWrapFlags(Flags);
  return S;
}

/// getUDivExpr - Get a canonical unsigned division expression, or something
/// simpler if possible.
const SCEV *ScalarEvolution::getUDivExpr(const SCEV *LHS,
                                         const SCEV *RHS) {
  assert(getEffectiveSCEVType(LHS->getType()) ==
         getEffectiveSCEVType(RHS->getType()) &&
         "SCEVUDivExpr operand types don't match!");

  if (const SCEVConstant *RHSC = dyn_cast<SCEVConstant>(RHS)) {
    if (RHSC->getValue()->equalsInt(1))
      return LHS;                               // X udiv 1 --> x
    // If the denominator is zero, the result of the udiv is undefined. Don't
    // try to analyze it, because the resolution chosen here may differ from
    // the resolution chosen in other parts of the compiler.
    if (!RHSC->getValue()->isZero()) {
      // Determine if the division can be folded into the operands of
      // its operands.
      // TODO: Generalize this to non-constants by using known-bits information.
      Type *Ty = LHS->getType();
      unsigned LZ = RHSC->getValue()->getValue().countLeadingZeros();
      unsigned MaxShiftAmt = getTypeSizeInBits(Ty) - LZ - 1;
      // For non-power-of-two values, effectively round the value up to the
      // nearest power of two.
      if (!RHSC->getValue()->getValue().isPowerOf2())
        ++MaxShiftAmt;
      IntegerType *ExtTy =
        IntegerType::get(getContext(), getTypeSizeInBits(Ty) + MaxShiftAmt);
      if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(LHS))
        if (const SCEVConstant *Step =
            dyn_cast<SCEVConstant>(AR->getStepRecurrence(*this))) {
          // {X,+,N}/C --> {X/C,+,N/C} if safe and N/C can be folded.
          const APInt &StepInt = Step->getValue()->getValue();
          const APInt &DivInt = RHSC->getValue()->getValue();
          if (!StepInt.urem(DivInt) &&
              getZeroExtendExpr(AR, ExtTy) ==
              getAddRecExpr(getZeroExtendExpr(AR->getStart(), ExtTy),
                            getZeroExtendExpr(Step, ExtTy),
                            AR->getLoop(), SCEV::FlagAnyWrap)) {
            SmallVector<const SCEV *, 4> Operands;
            for (unsigned i = 0, e = AR->getNumOperands(); i != e; ++i)
              Operands.push_back(getUDivExpr(AR->getOperand(i), RHS));
            return getAddRecExpr(Operands, AR->getLoop(),
                                 SCEV::FlagNW);
          }
          /// Get a canonical UDivExpr for a recurrence.
          /// {X,+,N}/C => {Y,+,N}/C where Y=X-(X%N). Safe when C%N=0.
          // We can currently only fold X%N if X is constant.
          const SCEVConstant *StartC = dyn_cast<SCEVConstant>(AR->getStart());
          if (StartC && !DivInt.urem(StepInt) &&
              getZeroExtendExpr(AR, ExtTy) ==
              getAddRecExpr(getZeroExtendExpr(AR->getStart(), ExtTy),
                            getZeroExtendExpr(Step, ExtTy),
                            AR->getLoop(), SCEV::FlagAnyWrap)) {
            const APInt &StartInt = StartC->getValue()->getValue();
            const APInt &StartRem = StartInt.urem(StepInt);
            if (StartRem != 0)
              LHS = getAddRecExpr(getConstant(StartInt - StartRem), Step,
                                  AR->getLoop(), SCEV::FlagNW);
          }
        }
      // (A*B)/C --> A*(B/C) if safe and B/C can be folded.
      if (const SCEVMulExpr *M = dyn_cast<SCEVMulExpr>(LHS)) {
        SmallVector<const SCEV *, 4> Operands;
        for (unsigned i = 0, e = M->getNumOperands(); i != e; ++i)
          Operands.push_back(getZeroExtendExpr(M->getOperand(i), ExtTy));
        if (getZeroExtendExpr(M, ExtTy) == getMulExpr(Operands))
          // Find an operand that's safely divisible.
          for (unsigned i = 0, e = M->getNumOperands(); i != e; ++i) {
            const SCEV *Op = M->getOperand(i);
            const SCEV *Div = getUDivExpr(Op, RHSC);
            if (!isa<SCEVUDivExpr>(Div) && getMulExpr(Div, RHSC) == Op) {
              Operands = SmallVector<const SCEV *, 4>(M->op_begin(),
                                                      M->op_end());
              Operands[i] = Div;
              return getMulExpr(Operands);
            }
          }
      }
      // (A+B)/C --> (A/C + B/C) if safe and A/C and B/C can be folded.
      if (const SCEVAddExpr *A = dyn_cast<SCEVAddExpr>(LHS)) {
        SmallVector<const SCEV *, 4> Operands;
        for (unsigned i = 0, e = A->getNumOperands(); i != e; ++i)
          Operands.push_back(getZeroExtendExpr(A->getOperand(i), ExtTy));
        if (getZeroExtendExpr(A, ExtTy) == getAddExpr(Operands)) {
          Operands.clear();
          for (unsigned i = 0, e = A->getNumOperands(); i != e; ++i) {
            const SCEV *Op = getUDivExpr(A->getOperand(i), RHS);
            if (isa<SCEVUDivExpr>(Op) ||
                getMulExpr(Op, RHS) != A->getOperand(i))
              break;
            Operands.push_back(Op);
          }
          if (Operands.size() == A->getNumOperands())
            return getAddExpr(Operands);
        }
      }

      // Fold if both operands are constant.
      if (const SCEVConstant *LHSC = dyn_cast<SCEVConstant>(LHS)) {
        Constant *LHSCV = LHSC->getValue();
        Constant *RHSCV = RHSC->getValue();
        return getConstant(cast<ConstantInt>(ConstantExpr::getUDiv(LHSCV,
                                                                   RHSCV)));
      }
    }
  }

  FoldingSetNodeID ID;
  ID.AddInteger(scUDivExpr);
  ID.AddPointer(LHS);
  ID.AddPointer(RHS);
  void *IP = nullptr;
  if (const SCEV *S = UniqueSCEVs.FindNodeOrInsertPos(ID, IP)) return S;
  SCEV *S = new (SCEVAllocator) SCEVUDivExpr(ID.Intern(SCEVAllocator),
                                             LHS, RHS);
  UniqueSCEVs.InsertNode(S, IP);
  return S;
}

static const APInt gcd(const SCEVConstant *C1, const SCEVConstant *C2) {
  APInt A = C1->getValue()->getValue().abs();
  APInt B = C2->getValue()->getValue().abs();
  uint32_t ABW = A.getBitWidth();
  uint32_t BBW = B.getBitWidth();

  if (ABW > BBW)
    B = B.zext(ABW);
  else if (ABW < BBW)
    A = A.zext(BBW);

  return APIntOps::GreatestCommonDivisor(A, B);
}

/// getUDivExactExpr - Get a canonical unsigned division expression, or
/// something simpler if possible. There is no representation for an exact udiv
/// in SCEV IR, but we can attempt to remove factors from the LHS and RHS.
/// We can't do this when it's not exact because the udiv may be clearing bits.
const SCEV *ScalarEvolution::getUDivExactExpr(const SCEV *LHS,
                                              const SCEV *RHS) {
  // TODO: we could try to find factors in all sorts of things, but for now we
  // just deal with u/exact (multiply, constant). See SCEVDivision towards the
  // end of this file for inspiration.

  const SCEVMulExpr *Mul = dyn_cast<SCEVMulExpr>(LHS);
  if (!Mul)
    return getUDivExpr(LHS, RHS);

  if (const SCEVConstant *RHSCst = dyn_cast<SCEVConstant>(RHS)) {
    // If the mulexpr multiplies by a constant, then that constant must be the
    // first element of the mulexpr.
    if (const SCEVConstant *LHSCst =
            dyn_cast<SCEVConstant>(Mul->getOperand(0))) {
      if (LHSCst == RHSCst) {
        SmallVector<const SCEV *, 2> Operands;
        Operands.append(Mul->op_begin() + 1, Mul->op_end());
        return getMulExpr(Operands);
      }

      // We can't just assume that LHSCst divides RHSCst cleanly, it could be
      // that there's a factor provided by one of the other terms. We need to
      // check.
      APInt Factor = gcd(LHSCst, RHSCst);
      if (!Factor.isIntN(1)) {
        LHSCst = cast<SCEVConstant>(
            getConstant(LHSCst->getValue()->getValue().udiv(Factor)));
        RHSCst = cast<SCEVConstant>(
            getConstant(RHSCst->getValue()->getValue().udiv(Factor)));
        SmallVector<const SCEV *, 2> Operands;
        Operands.push_back(LHSCst);
        Operands.append(Mul->op_begin() + 1, Mul->op_end());
        LHS = getMulExpr(Operands);
        RHS = RHSCst;
        Mul = dyn_cast<SCEVMulExpr>(LHS);
        if (!Mul)
          return getUDivExactExpr(LHS, RHS);
      }
    }
  }

  for (int i = 0, e = Mul->getNumOperands(); i != e; ++i) {
    if (Mul->getOperand(i) == RHS) {
      SmallVector<const SCEV *, 2> Operands;
      Operands.append(Mul->op_begin(), Mul->op_begin() + i);
      Operands.append(Mul->op_begin() + i + 1, Mul->op_end());
      return getMulExpr(Operands);
    }
  }

  return getUDivExpr(LHS, RHS);
}

/// getAddRecExpr - Get an add recurrence expression for the specified loop.
/// Simplify the expression as much as possible.
const SCEV *ScalarEvolution::getAddRecExpr(const SCEV *Start, const SCEV *Step,
                                           const Loop *L,
                                           SCEV::NoWrapFlags Flags) {
  SmallVector<const SCEV *, 4> Operands;
  Operands.push_back(Start);
  if (const SCEVAddRecExpr *StepChrec = dyn_cast<SCEVAddRecExpr>(Step))
    if (StepChrec->getLoop() == L) {
      Operands.append(StepChrec->op_begin(), StepChrec->op_end());
      return getAddRecExpr(Operands, L, maskFlags(Flags, SCEV::FlagNW));
    }

  Operands.push_back(Step);
  return getAddRecExpr(Operands, L, Flags);
}

/// getAddRecExpr - Get an add recurrence expression for the specified loop.
/// Simplify the expression as much as possible.
const SCEV *
ScalarEvolution::getAddRecExpr(SmallVectorImpl<const SCEV *> &Operands,
                               const Loop *L, SCEV::NoWrapFlags Flags) {
  if (Operands.size() == 1) return Operands[0];
#ifndef NDEBUG
  Type *ETy = getEffectiveSCEVType(Operands[0]->getType());
  for (unsigned i = 1, e = Operands.size(); i != e; ++i)
    assert(getEffectiveSCEVType(Operands[i]->getType()) == ETy &&
           "SCEVAddRecExpr operand types don't match!");
  for (unsigned i = 0, e = Operands.size(); i != e; ++i)
    assert(isLoopInvariant(Operands[i], L) &&
           "SCEVAddRecExpr operand is not loop-invariant!");
#endif

  if (Operands.back()->isZero()) {
    Operands.pop_back();
    return getAddRecExpr(Operands, L, SCEV::FlagAnyWrap); // {X,+,0}  -->  X
  }

  // It's tempting to want to call getMaxBackedgeTakenCount count here and
  // use that information to infer NUW and NSW flags. However, computing a
  // BE count requires calling getAddRecExpr, so we may not yet have a
  // meaningful BE count at this point (and if we don't, we'd be stuck
  // with a SCEVCouldNotCompute as the cached BE count).

  Flags = StrengthenNoWrapFlags(this, scAddRecExpr, Operands, Flags);

  // Canonicalize nested AddRecs in by nesting them in order of loop depth.
  if (const SCEVAddRecExpr *NestedAR = dyn_cast<SCEVAddRecExpr>(Operands[0])) {
    const Loop *NestedLoop = NestedAR->getLoop();
    if (L->contains(NestedLoop) ?
        (L->getLoopDepth() < NestedLoop->getLoopDepth()) :
        (!NestedLoop->contains(L) &&
         DT->dominates(L->getHeader(), NestedLoop->getHeader()))) {
      SmallVector<const SCEV *, 4> NestedOperands(NestedAR->op_begin(),
                                                  NestedAR->op_end());
      Operands[0] = NestedAR->getStart();
      // AddRecs require their operands be loop-invariant with respect to their
      // loops. Don't perform this transformation if it would break this
      // requirement.
      bool AllInvariant = true;
      for (unsigned i = 0, e = Operands.size(); i != e; ++i)
        if (!isLoopInvariant(Operands[i], L)) {
          AllInvariant = false;
          break;
        }
      if (AllInvariant) {
        // Create a recurrence for the outer loop with the same step size.
        //
        // The outer recurrence keeps its NW flag but only keeps NUW/NSW if the
        // inner recurrence has the same property.
        SCEV::NoWrapFlags OuterFlags =
          maskFlags(Flags, SCEV::FlagNW | NestedAR->getNoWrapFlags());

        NestedOperands[0] = getAddRecExpr(Operands, L, OuterFlags);
        AllInvariant = true;
        for (unsigned i = 0, e = NestedOperands.size(); i != e; ++i)
          if (!isLoopInvariant(NestedOperands[i], NestedLoop)) {
            AllInvariant = false;
            break;
          }
        if (AllInvariant) {
          // Ok, both add recurrences are valid after the transformation.
          //
          // The inner recurrence keeps its NW flag but only keeps NUW/NSW if
          // the outer recurrence has the same property.
          SCEV::NoWrapFlags InnerFlags =
            maskFlags(NestedAR->getNoWrapFlags(), SCEV::FlagNW | Flags);
          return getAddRecExpr(NestedOperands, NestedLoop, InnerFlags);
        }
      }
      // Reset Operands to its original state.
      Operands[0] = NestedAR;
    }
  }

  // Okay, it looks like we really DO need an addrec expr.  Check to see if we
  // already have one, otherwise create a new one.
  FoldingSetNodeID ID;
  ID.AddInteger(scAddRecExpr);
  for (unsigned i = 0, e = Operands.size(); i != e; ++i)
    ID.AddPointer(Operands[i]);
  ID.AddPointer(L);
  void *IP = nullptr;
  SCEVAddRecExpr *S =
    static_cast<SCEVAddRecExpr *>(UniqueSCEVs.FindNodeOrInsertPos(ID, IP));
  if (!S) {
    const SCEV **O = SCEVAllocator.Allocate<const SCEV *>(Operands.size());
    std::uninitialized_copy(Operands.begin(), Operands.end(), O);
    S = new (SCEVAllocator) SCEVAddRecExpr(ID.Intern(SCEVAllocator),
                                           O, Operands.size(), L);
    UniqueSCEVs.InsertNode(S, IP);
  }
  S->setNoWrapFlags(Flags);
  return S;
}

const SCEV *
ScalarEvolution::getGEPExpr(Type *PointeeType, const SCEV *BaseExpr,
                            const SmallVectorImpl<const SCEV *> &IndexExprs,
                            bool InBounds) {
  // getSCEV(Base)->getType() has the same address space as Base->getType()
  // because SCEV::getType() preserves the address space.
  Type *IntPtrTy = getEffectiveSCEVType(BaseExpr->getType());
  // FIXME(PR23527): Don't blindly transfer the inbounds flag from the GEP
  // instruction to its SCEV, because the Instruction may be guarded by control
  // flow and the no-overflow bits may not be valid for the expression in any
  // context.
  SCEV::NoWrapFlags Wrap = InBounds ? SCEV::FlagNSW : SCEV::FlagAnyWrap;

  const SCEV *TotalOffset = getConstant(IntPtrTy, 0);
  // The address space is unimportant. The first thing we do on CurTy is getting
  // its element type.
  Type *CurTy = PointerType::getUnqual(PointeeType);
  for (const SCEV *IndexExpr : IndexExprs) {
    // Compute the (potentially symbolic) offset in bytes for this index.
    if (StructType *STy = dyn_cast<StructType>(CurTy)) {
      // For a struct, add the member offset.
      ConstantInt *Index = cast<SCEVConstant>(IndexExpr)->getValue();
      unsigned FieldNo = Index->getZExtValue();
      const SCEV *FieldOffset = getOffsetOfExpr(IntPtrTy, STy, FieldNo);

      // Add the field offset to the running total offset.
      TotalOffset = getAddExpr(TotalOffset, FieldOffset);

      // Update CurTy to the type of the field at Index.
      CurTy = STy->getTypeAtIndex(Index);
    } else {
      // Update CurTy to its element type.
      CurTy = cast<SequentialType>(CurTy)->getElementType();
      // For an array, add the element offset, explicitly scaled.
      const SCEV *ElementSize = getSizeOfExpr(IntPtrTy, CurTy);
      // Getelementptr indices are signed.
      IndexExpr = getTruncateOrSignExtend(IndexExpr, IntPtrTy);

      // Multiply the index by the element size to compute the element offset.
      const SCEV *LocalOffset = getMulExpr(IndexExpr, ElementSize, Wrap);

      // Add the element offset to the running total offset.
      TotalOffset = getAddExpr(TotalOffset, LocalOffset);
    }
  }

  // Add the total offset from all the GEP indices to the base.
  return getAddExpr(BaseExpr, TotalOffset, Wrap);
}

const SCEV *ScalarEvolution::getSMaxExpr(const SCEV *LHS,
                                         const SCEV *RHS) {
  SmallVector<const SCEV *, 2> Ops;
  Ops.push_back(LHS);
  Ops.push_back(RHS);
  return getSMaxExpr(Ops);
}

const SCEV *
ScalarEvolution::getSMaxExpr(SmallVectorImpl<const SCEV *> &Ops) {
  assert(!Ops.empty() && "Cannot get empty smax!");
  if (Ops.size() == 1) return Ops[0];
#ifndef NDEBUG
  Type *ETy = getEffectiveSCEVType(Ops[0]->getType());
  for (unsigned i = 1, e = Ops.size(); i != e; ++i)
    assert(getEffectiveSCEVType(Ops[i]->getType()) == ETy &&
           "SCEVSMaxExpr operand types don't match!");
#endif

  // Sort by complexity, this groups all similar expression types together.
  GroupByComplexity(Ops, LI);

  // If there are any constants, fold them together.
  unsigned Idx = 0;
  if (const SCEVConstant *LHSC = dyn_cast<SCEVConstant>(Ops[0])) {
    ++Idx;
    assert(Idx < Ops.size());
    while (const SCEVConstant *RHSC = dyn_cast<SCEVConstant>(Ops[Idx])) {
      // We found two constants, fold them together!
      ConstantInt *Fold = ConstantInt::get(getContext(),
                              APIntOps::smax(LHSC->getValue()->getValue(),
                                             RHSC->getValue()->getValue()));
      Ops[0] = getConstant(Fold);
      Ops.erase(Ops.begin()+1);  // Erase the folded element
      if (Ops.size() == 1) return Ops[0];
      LHSC = cast<SCEVConstant>(Ops[0]);
    }

    // If we are left with a constant minimum-int, strip it off.
    if (cast<SCEVConstant>(Ops[0])->getValue()->isMinValue(true)) {
      Ops.erase(Ops.begin());
      --Idx;
    } else if (cast<SCEVConstant>(Ops[0])->getValue()->isMaxValue(true)) {
      // If we have an smax with a constant maximum-int, it will always be
      // maximum-int.
      return Ops[0];
    }

    if (Ops.size() == 1) return Ops[0];
  }

  // Find the first SMax
  while (Idx < Ops.size() && Ops[Idx]->getSCEVType() < scSMaxExpr)
    ++Idx;

  // Check to see if one of the operands is an SMax. If so, expand its operands
  // onto our operand list, and recurse to simplify.
  if (Idx < Ops.size()) {
    bool DeletedSMax = false;
    while (const SCEVSMaxExpr *SMax = dyn_cast<SCEVSMaxExpr>(Ops[Idx])) {
      Ops.erase(Ops.begin()+Idx);
      Ops.append(SMax->op_begin(), SMax->op_end());
      DeletedSMax = true;
    }

    if (DeletedSMax)
      return getSMaxExpr(Ops);
  }

  // Okay, check to see if the same value occurs in the operand list twice.  If
  // so, delete one.  Since we sorted the list, these values are required to
  // be adjacent.
  for (unsigned i = 0, e = Ops.size()-1; i != e; ++i)
    //  X smax Y smax Y  -->  X smax Y
    //  X smax Y         -->  X, if X is always greater than Y
    if (Ops[i] == Ops[i+1] ||
        isKnownPredicate(ICmpInst::ICMP_SGE, Ops[i], Ops[i+1])) {
      Ops.erase(Ops.begin()+i+1, Ops.begin()+i+2);
      --i; --e;
    } else if (isKnownPredicate(ICmpInst::ICMP_SLE, Ops[i], Ops[i+1])) {
      Ops.erase(Ops.begin()+i, Ops.begin()+i+1);
      --i; --e;
    }

  if (Ops.size() == 1) return Ops[0];

  assert(!Ops.empty() && "Reduced smax down to nothing!");

  // Okay, it looks like we really DO need an smax expr.  Check to see if we
  // already have one, otherwise create a new one.
  FoldingSetNodeID ID;
  ID.AddInteger(scSMaxExpr);
  for (unsigned i = 0, e = Ops.size(); i != e; ++i)
    ID.AddPointer(Ops[i]);
  void *IP = nullptr;
  if (const SCEV *S = UniqueSCEVs.FindNodeOrInsertPos(ID, IP)) return S;
  const SCEV **O = SCEVAllocator.Allocate<const SCEV *>(Ops.size());
  std::uninitialized_copy(Ops.begin(), Ops.end(), O);
  SCEV *S = new (SCEVAllocator) SCEVSMaxExpr(ID.Intern(SCEVAllocator),
                                             O, Ops.size());
  UniqueSCEVs.InsertNode(S, IP);
  return S;
}

const SCEV *ScalarEvolution::getUMaxExpr(const SCEV *LHS,
                                         const SCEV *RHS) {
  SmallVector<const SCEV *, 2> Ops;
  Ops.push_back(LHS);
  Ops.push_back(RHS);
  return getUMaxExpr(Ops);
}

const SCEV *
ScalarEvolution::getUMaxExpr(SmallVectorImpl<const SCEV *> &Ops) {
  assert(!Ops.empty() && "Cannot get empty umax!");
  if (Ops.size() == 1) return Ops[0];
#ifndef NDEBUG
  Type *ETy = getEffectiveSCEVType(Ops[0]->getType());
  for (unsigned i = 1, e = Ops.size(); i != e; ++i)
    assert(getEffectiveSCEVType(Ops[i]->getType()) == ETy &&
           "SCEVUMaxExpr operand types don't match!");
#endif

  // Sort by complexity, this groups all similar expression types together.
  GroupByComplexity(Ops, LI);

  // If there are any constants, fold them together.
  unsigned Idx = 0;
  if (const SCEVConstant *LHSC = dyn_cast<SCEVConstant>(Ops[0])) {
    ++Idx;
    assert(Idx < Ops.size());
    while (const SCEVConstant *RHSC = dyn_cast<SCEVConstant>(Ops[Idx])) {
      // We found two constants, fold them together!
      ConstantInt *Fold = ConstantInt::get(getContext(),
                              APIntOps::umax(LHSC->getValue()->getValue(),
                                             RHSC->getValue()->getValue()));
      Ops[0] = getConstant(Fold);
      Ops.erase(Ops.begin()+1);  // Erase the folded element
      if (Ops.size() == 1) return Ops[0];
      LHSC = cast<SCEVConstant>(Ops[0]);
    }

    // If we are left with a constant minimum-int, strip it off.
    if (cast<SCEVConstant>(Ops[0])->getValue()->isMinValue(false)) {
      Ops.erase(Ops.begin());
      --Idx;
    } else if (cast<SCEVConstant>(Ops[0])->getValue()->isMaxValue(false)) {
      // If we have an umax with a constant maximum-int, it will always be
      // maximum-int.
      return Ops[0];
    }

    if (Ops.size() == 1) return Ops[0];
  }

  // Find the first UMax
  while (Idx < Ops.size() && Ops[Idx]->getSCEVType() < scUMaxExpr)
    ++Idx;

  // Check to see if one of the operands is a UMax. If so, expand its operands
  // onto our operand list, and recurse to simplify.
  if (Idx < Ops.size()) {
    bool DeletedUMax = false;
    while (const SCEVUMaxExpr *UMax = dyn_cast<SCEVUMaxExpr>(Ops[Idx])) {
      Ops.erase(Ops.begin()+Idx);
      Ops.append(UMax->op_begin(), UMax->op_end());
      DeletedUMax = true;
    }

    if (DeletedUMax)
      return getUMaxExpr(Ops);
  }

  // Okay, check to see if the same value occurs in the operand list twice.  If
  // so, delete one.  Since we sorted the list, these values are required to
  // be adjacent.
  for (unsigned i = 0, e = Ops.size()-1; i != e; ++i)
    //  X umax Y umax Y  -->  X umax Y
    //  X umax Y         -->  X, if X is always greater than Y
    if (Ops[i] == Ops[i+1] ||
        isKnownPredicate(ICmpInst::ICMP_UGE, Ops[i], Ops[i+1])) {
      Ops.erase(Ops.begin()+i+1, Ops.begin()+i+2);
      --i; --e;
    } else if (isKnownPredicate(ICmpInst::ICMP_ULE, Ops[i], Ops[i+1])) {
      Ops.erase(Ops.begin()+i, Ops.begin()+i+1);
      --i; --e;
    }

  if (Ops.size() == 1) return Ops[0];

  assert(!Ops.empty() && "Reduced umax down to nothing!");

  // Okay, it looks like we really DO need a umax expr.  Check to see if we
  // already have one, otherwise create a new one.
  FoldingSetNodeID ID;
  ID.AddInteger(scUMaxExpr);
  for (unsigned i = 0, e = Ops.size(); i != e; ++i)
    ID.AddPointer(Ops[i]);
  void *IP = nullptr;
  if (const SCEV *S = UniqueSCEVs.FindNodeOrInsertPos(ID, IP)) return S;
  const SCEV **O = SCEVAllocator.Allocate<const SCEV *>(Ops.size());
  std::uninitialized_copy(Ops.begin(), Ops.end(), O);
  SCEV *S = new (SCEVAllocator) SCEVUMaxExpr(ID.Intern(SCEVAllocator),
                                             O, Ops.size());
  UniqueSCEVs.InsertNode(S, IP);
  return S;
}

const SCEV *ScalarEvolution::getSMinExpr(const SCEV *LHS,
                                         const SCEV *RHS) {
  // ~smax(~x, ~y) == smin(x, y).
  return getNotSCEV(getSMaxExpr(getNotSCEV(LHS), getNotSCEV(RHS)));
}

const SCEV *ScalarEvolution::getUMinExpr(const SCEV *LHS,
                                         const SCEV *RHS) {
  // ~umax(~x, ~y) == umin(x, y)
  return getNotSCEV(getUMaxExpr(getNotSCEV(LHS), getNotSCEV(RHS)));
}

const SCEV *ScalarEvolution::getSizeOfExpr(Type *IntTy, Type *AllocTy) {
  // We can bypass creating a target-independent
  // constant expression and then folding it back into a ConstantInt.
  // This is just a compile-time optimization.
  return getConstant(IntTy,
                     F->getParent()->getDataLayout().getTypeAllocSize(AllocTy));
}

const SCEV *ScalarEvolution::getOffsetOfExpr(Type *IntTy,
                                             StructType *STy,
                                             unsigned FieldNo) {
  // We can bypass creating a target-independent
  // constant expression and then folding it back into a ConstantInt.
  // This is just a compile-time optimization.
  return getConstant(
      IntTy,
      F->getParent()->getDataLayout().getStructLayout(STy)->getElementOffset(
          FieldNo));
}

const SCEV *ScalarEvolution::getUnknown(Value *V) {
  // Don't attempt to do anything other than create a SCEVUnknown object
  // here.  createSCEV only calls getUnknown after checking for all other
  // interesting possibilities, and any other code that calls getUnknown
  // is doing so in order to hide a value from SCEV canonicalization.

  FoldingSetNodeID ID;
  ID.AddInteger(scUnknown);
  ID.AddPointer(V);
  void *IP = nullptr;
  if (SCEV *S = UniqueSCEVs.FindNodeOrInsertPos(ID, IP)) {
    assert(cast<SCEVUnknown>(S)->getValue() == V &&
           "Stale SCEVUnknown in uniquing map!");
    return S;
  }
  SCEV *S = new (SCEVAllocator) SCEVUnknown(ID.Intern(SCEVAllocator), V, this,
                                            FirstUnknown);
  FirstUnknown = cast<SCEVUnknown>(S);
  UniqueSCEVs.InsertNode(S, IP);
  return S;
}

//===----------------------------------------------------------------------===//
//            Basic SCEV Analysis and PHI Idiom Recognition Code
//

/// isSCEVable - Test if values of the given type are analyzable within
/// the SCEV framework. This primarily includes integer types, and it
/// can optionally include pointer types if the ScalarEvolution class
/// has access to target-specific information.
bool ScalarEvolution::isSCEVable(Type *Ty) const {
  // Integers and pointers are always SCEVable.
  return Ty->isIntegerTy() || Ty->isPointerTy();
}

/// getTypeSizeInBits - Return the size in bits of the specified type,
/// for which isSCEVable must return true.
uint64_t ScalarEvolution::getTypeSizeInBits(Type *Ty) const {
  assert(isSCEVable(Ty) && "Type is not SCEVable!");
  return F->getParent()->getDataLayout().getTypeSizeInBits(Ty);
}

/// getEffectiveSCEVType - Return a type with the same bitwidth as
/// the given type and which represents how SCEV will treat the given
/// type, for which isSCEVable must return true. For pointer types,
/// this is the pointer-sized integer type.
Type *ScalarEvolution::getEffectiveSCEVType(Type *Ty) const {
  assert(isSCEVable(Ty) && "Type is not SCEVable!");

  if (Ty->isIntegerTy()) {
    return Ty;
  }

  // The only other support type is pointer.
  assert(Ty->isPointerTy() && "Unexpected non-pointer non-integer type!");
  return F->getParent()->getDataLayout().getIntPtrType(Ty);
}

const SCEV *ScalarEvolution::getCouldNotCompute() {
  return &CouldNotCompute;
}

namespace {
  // Helper class working with SCEVTraversal to figure out if a SCEV contains
  // a SCEVUnknown with null value-pointer. FindInvalidSCEVUnknown::FindOne
  // is set iff if find such SCEVUnknown.
  //
  struct FindInvalidSCEVUnknown {
    bool FindOne;
    FindInvalidSCEVUnknown() { FindOne = false; }
    bool follow(const SCEV *S) {
      switch (static_cast<SCEVTypes>(S->getSCEVType())) {
      case scConstant:
        return false;
      case scUnknown:
        if (!cast<SCEVUnknown>(S)->getValue())
          FindOne = true;
        return false;
      default:
        return true;
      }
    }
    bool isDone() const { return FindOne; }
  };
}

bool ScalarEvolution::checkValidity(const SCEV *S) const {
  FindInvalidSCEVUnknown F;
  SCEVTraversal<FindInvalidSCEVUnknown> ST(F);
  ST.visitAll(S);

  return !F.FindOne;
}

/// getSCEV - Return an existing SCEV if it exists, otherwise analyze the
/// expression and create a new one.
const SCEV *ScalarEvolution::getSCEV(Value *V) {
  assert(isSCEVable(V->getType()) && "Value is not SCEVable!");

  ValueExprMapType::iterator I = ValueExprMap.find_as(V);
  if (I != ValueExprMap.end()) {
    const SCEV *S = I->second;
    if (checkValidity(S))
      return S;
    else
      ValueExprMap.erase(I);
  }
  const SCEV *S = createSCEV(V);

  // The process of creating a SCEV for V may have caused other SCEVs
  // to have been created, so it's necessary to insert the new entry
  // from scratch, rather than trying to remember the insert position
  // above.
  ValueExprMap.insert(std::make_pair(SCEVCallbackVH(V, this), S));
  return S;
}

/// getNegativeSCEV - Return a SCEV corresponding to -V = -1*V
///
const SCEV *ScalarEvolution::getNegativeSCEV(const SCEV *V) {
  if (const SCEVConstant *VC = dyn_cast<SCEVConstant>(V))
    return getConstant(
               cast<ConstantInt>(ConstantExpr::getNeg(VC->getValue())));

  Type *Ty = V->getType();
  Ty = getEffectiveSCEVType(Ty);
  return getMulExpr(V,
                  getConstant(cast<ConstantInt>(Constant::getAllOnesValue(Ty))));
}

/// getNotSCEV - Return a SCEV corresponding to ~V = -1-V
const SCEV *ScalarEvolution::getNotSCEV(const SCEV *V) {
  if (const SCEVConstant *VC = dyn_cast<SCEVConstant>(V))
    return getConstant(
                cast<ConstantInt>(ConstantExpr::getNot(VC->getValue())));

  Type *Ty = V->getType();
  Ty = getEffectiveSCEVType(Ty);
  const SCEV *AllOnes =
                   getConstant(cast<ConstantInt>(Constant::getAllOnesValue(Ty)));
  return getMinusSCEV(AllOnes, V);
}

/// getMinusSCEV - Return LHS-RHS.  Minus is represented in SCEV as A+B*-1.
const SCEV *ScalarEvolution::getMinusSCEV(const SCEV *LHS, const SCEV *RHS,
                                          SCEV::NoWrapFlags Flags) {
  assert(!maskFlags(Flags, SCEV::FlagNUW) && "subtraction does not have NUW");

  // Fast path: X - X --> 0.
  if (LHS == RHS)
    return getConstant(LHS->getType(), 0);

  // X - Y --> X + -Y.
  // X -(nsw || nuw) Y --> X + -Y.
  return getAddExpr(LHS, getNegativeSCEV(RHS));
}

/// getTruncateOrZeroExtend - Return a SCEV corresponding to a conversion of the
/// input value to the specified type.  If the type must be extended, it is zero
/// extended.
const SCEV *
ScalarEvolution::getTruncateOrZeroExtend(const SCEV *V, Type *Ty) {
  Type *SrcTy = V->getType();
  assert((SrcTy->isIntegerTy() || SrcTy->isPointerTy()) &&
         (Ty->isIntegerTy() || Ty->isPointerTy()) &&
         "Cannot truncate or zero extend with non-integer arguments!");
  if (getTypeSizeInBits(SrcTy) == getTypeSizeInBits(Ty))
    return V;  // No conversion
  if (getTypeSizeInBits(SrcTy) > getTypeSizeInBits(Ty))
    return getTruncateExpr(V, Ty);
  return getZeroExtendExpr(V, Ty);
}

/// getTruncateOrSignExtend - Return a SCEV corresponding to a conversion of the
/// input value to the specified type.  If the type must be extended, it is sign
/// extended.
const SCEV *
ScalarEvolution::getTruncateOrSignExtend(const SCEV *V,
                                         Type *Ty) {
  Type *SrcTy = V->getType();
  assert((SrcTy->isIntegerTy() || SrcTy->isPointerTy()) &&
         (Ty->isIntegerTy() || Ty->isPointerTy()) &&
         "Cannot truncate or zero extend with non-integer arguments!");
  if (getTypeSizeInBits(SrcTy) == getTypeSizeInBits(Ty))
    return V;  // No conversion
  if (getTypeSizeInBits(SrcTy) > getTypeSizeInBits(Ty))
    return getTruncateExpr(V, Ty);
  return getSignExtendExpr(V, Ty);
}

/// getNoopOrZeroExtend - Return a SCEV corresponding to a conversion of the
/// input value to the specified type.  If the type must be extended, it is zero
/// extended.  The conversion must not be narrowing.
const SCEV *
ScalarEvolution::getNoopOrZeroExtend(const SCEV *V, Type *Ty) {
  Type *SrcTy = V->getType();
  assert((SrcTy->isIntegerTy() || SrcTy->isPointerTy()) &&
         (Ty->isIntegerTy() || Ty->isPointerTy()) &&
         "Cannot noop or zero extend with non-integer arguments!");
  assert(getTypeSizeInBits(SrcTy) <= getTypeSizeInBits(Ty) &&
         "getNoopOrZeroExtend cannot truncate!");
  if (getTypeSizeInBits(SrcTy) == getTypeSizeInBits(Ty))
    return V;  // No conversion
  return getZeroExtendExpr(V, Ty);
}

/// getNoopOrSignExtend - Return a SCEV corresponding to a conversion of the
/// input value to the specified type.  If the type must be extended, it is sign
/// extended.  The conversion must not be narrowing.
const SCEV *
ScalarEvolution::getNoopOrSignExtend(const SCEV *V, Type *Ty) {
  Type *SrcTy = V->getType();
  assert((SrcTy->isIntegerTy() || SrcTy->isPointerTy()) &&
         (Ty->isIntegerTy() || Ty->isPointerTy()) &&
         "Cannot noop or sign extend with non-integer arguments!");
  assert(getTypeSizeInBits(SrcTy) <= getTypeSizeInBits(Ty) &&
         "getNoopOrSignExtend cannot truncate!");
  if (getTypeSizeInBits(SrcTy) == getTypeSizeInBits(Ty))
    return V;  // No conversion
  return getSignExtendExpr(V, Ty);
}

/// getNoopOrAnyExtend - Return a SCEV corresponding to a conversion of
/// the input value to the specified type. If the type must be extended,
/// it is extended with unspecified bits. The conversion must not be
/// narrowing.
const SCEV *
ScalarEvolution::getNoopOrAnyExtend(const SCEV *V, Type *Ty) {
  Type *SrcTy = V->getType();
  assert((SrcTy->isIntegerTy() || SrcTy->isPointerTy()) &&
         (Ty->isIntegerTy() || Ty->isPointerTy()) &&
         "Cannot noop or any extend with non-integer arguments!");
  assert(getTypeSizeInBits(SrcTy) <= getTypeSizeInBits(Ty) &&
         "getNoopOrAnyExtend cannot truncate!");
  if (getTypeSizeInBits(SrcTy) == getTypeSizeInBits(Ty))
    return V;  // No conversion
  return getAnyExtendExpr(V, Ty);
}

/// getTruncateOrNoop - Return a SCEV corresponding to a conversion of the
/// input value to the specified type.  The conversion must not be widening.
const SCEV *
ScalarEvolution::getTruncateOrNoop(const SCEV *V, Type *Ty) {
  Type *SrcTy = V->getType();
  assert((SrcTy->isIntegerTy() || SrcTy->isPointerTy()) &&
         (Ty->isIntegerTy() || Ty->isPointerTy()) &&
         "Cannot truncate or noop with non-integer arguments!");
  assert(getTypeSizeInBits(SrcTy) >= getTypeSizeInBits(Ty) &&
         "getTruncateOrNoop cannot extend!");
  if (getTypeSizeInBits(SrcTy) == getTypeSizeInBits(Ty))
    return V;  // No conversion
  return getTruncateExpr(V, Ty);
}

/// getUMaxFromMismatchedTypes - Promote the operands to the wider of
/// the types using zero-extension, and then perform a umax operation
/// with them.
const SCEV *ScalarEvolution::getUMaxFromMismatchedTypes(const SCEV *LHS,
                                                        const SCEV *RHS) {
  const SCEV *PromotedLHS = LHS;
  const SCEV *PromotedRHS = RHS;

  if (getTypeSizeInBits(LHS->getType()) > getTypeSizeInBits(RHS->getType()))
    PromotedRHS = getZeroExtendExpr(RHS, LHS->getType());
  else
    PromotedLHS = getNoopOrZeroExtend(LHS, RHS->getType());

  return getUMaxExpr(PromotedLHS, PromotedRHS);
}

/// getUMinFromMismatchedTypes - Promote the operands to the wider of
/// the types using zero-extension, and then perform a umin operation
/// with them.
const SCEV *ScalarEvolution::getUMinFromMismatchedTypes(const SCEV *LHS,
                                                        const SCEV *RHS) {
  const SCEV *PromotedLHS = LHS;
  const SCEV *PromotedRHS = RHS;

  if (getTypeSizeInBits(LHS->getType()) > getTypeSizeInBits(RHS->getType()))
    PromotedRHS = getZeroExtendExpr(RHS, LHS->getType());
  else
    PromotedLHS = getNoopOrZeroExtend(LHS, RHS->getType());

  return getUMinExpr(PromotedLHS, PromotedRHS);
}

/// getPointerBase - Transitively follow the chain of pointer-type operands
/// until reaching a SCEV that does not have a single pointer operand. This
/// returns a SCEVUnknown pointer for well-formed pointer-type expressions,
/// but corner cases do exist.
const SCEV *ScalarEvolution::getPointerBase(const SCEV *V) {
  // A pointer operand may evaluate to a nonpointer expression, such as null.
  if (!V->getType()->isPointerTy())
    return V;

  if (const SCEVCastExpr *Cast = dyn_cast<SCEVCastExpr>(V)) {
    return getPointerBase(Cast->getOperand());
  }
  else if (const SCEVNAryExpr *NAry = dyn_cast<SCEVNAryExpr>(V)) {
    const SCEV *PtrOp = nullptr;
    for (SCEVNAryExpr::op_iterator I = NAry->op_begin(), E = NAry->op_end();
         I != E; ++I) {
      if ((*I)->getType()->isPointerTy()) {
        // Cannot find the base of an expression with multiple pointer operands.
        if (PtrOp)
          return V;
        PtrOp = *I;
      }
    }
    if (!PtrOp)
      return V;
    return getPointerBase(PtrOp);
  }
  return V;
}

/// PushDefUseChildren - Push users of the given Instruction
/// onto the given Worklist.
static void
PushDefUseChildren(Instruction *I,
                   SmallVectorImpl<Instruction *> &Worklist) {
  // Push the def-use children onto the Worklist stack.
  for (User *U : I->users())
    Worklist.push_back(cast<Instruction>(U));
}

/// ForgetSymbolicValue - This looks up computed SCEV values for all
/// instructions that depend on the given instruction and removes them from
/// the ValueExprMapType map if they reference SymName. This is used during PHI
/// resolution.
void
ScalarEvolution::ForgetSymbolicName(Instruction *PN, const SCEV *SymName) {
  SmallVector<Instruction *, 16> Worklist;
  PushDefUseChildren(PN, Worklist);

  SmallPtrSet<Instruction *, 8> Visited;
  Visited.insert(PN);
  while (!Worklist.empty()) {
    Instruction *I = Worklist.pop_back_val();
    if (!Visited.insert(I).second)
      continue;

    ValueExprMapType::iterator It =
      ValueExprMap.find_as(static_cast<Value *>(I));
    if (It != ValueExprMap.end()) {
      const SCEV *Old = It->second;

      // Short-circuit the def-use traversal if the symbolic name
      // ceases to appear in expressions.
      if (Old != SymName && !hasOperand(Old, SymName))
        continue;

      // SCEVUnknown for a PHI either means that it has an unrecognized
      // structure, it's a PHI that's in the progress of being computed
      // by createNodeForPHI, or it's a single-value PHI. In the first case,
      // additional loop trip count information isn't going to change anything.
      // In the second case, createNodeForPHI will perform the necessary
      // updates on its own when it gets to that point. In the third, we do
      // want to forget the SCEVUnknown.
      if (!isa<PHINode>(I) ||
          !isa<SCEVUnknown>(Old) ||
          (I != PN && Old == SymName)) {
        forgetMemoizedResults(Old);
        ValueExprMap.erase(It);
      }
    }

    PushDefUseChildren(I, Worklist);
  }
}

/// createNodeForPHI - PHI nodes have two cases.  Either the PHI node exists in
/// a loop header, making it a potential recurrence, or it doesn't.
///
const SCEV *ScalarEvolution::createNodeForPHI(PHINode *PN) {
  if (const Loop *L = LI->getLoopFor(PN->getParent()))
    if (L->getHeader() == PN->getParent()) {
      // The loop may have multiple entrances or multiple exits; we can analyze
      // this phi as an addrec if it has a unique entry value and a unique
      // backedge value.
      Value *BEValueV = nullptr, *StartValueV = nullptr;
      for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
        Value *V = PN->getIncomingValue(i);
        if (L->contains(PN->getIncomingBlock(i))) {
          if (!BEValueV) {
            BEValueV = V;
          } else if (BEValueV != V) {
            BEValueV = nullptr;
            break;
          }
        } else if (!StartValueV) {
          StartValueV = V;
        } else if (StartValueV != V) {
          StartValueV = nullptr;
          break;
        }
      }
      if (BEValueV && StartValueV) {
        // While we are analyzing this PHI node, handle its value symbolically.
        const SCEV *SymbolicName = getUnknown(PN);
        assert(ValueExprMap.find_as(PN) == ValueExprMap.end() &&
               "PHI node already processed?");
        ValueExprMap.insert(std::make_pair(SCEVCallbackVH(PN, this), SymbolicName));

        // Using this symbolic name for the PHI, analyze the value coming around
        // the back-edge.
        const SCEV *BEValue = getSCEV(BEValueV);

        // NOTE: If BEValue is loop invariant, we know that the PHI node just
        // has a special value for the first iteration of the loop.

        // If the value coming around the backedge is an add with the symbolic
        // value we just inserted, then we found a simple induction variable!
        if (const SCEVAddExpr *Add = dyn_cast<SCEVAddExpr>(BEValue)) {
          // If there is a single occurrence of the symbolic value, replace it
          // with a recurrence.
          unsigned FoundIndex = Add->getNumOperands();
          for (unsigned i = 0, e = Add->getNumOperands(); i != e; ++i)
            if (Add->getOperand(i) == SymbolicName)
              if (FoundIndex == e) {
                FoundIndex = i;
                break;
              }

          if (FoundIndex != Add->getNumOperands()) {
            // Create an add with everything but the specified operand.
            SmallVector<const SCEV *, 8> Ops;
            for (unsigned i = 0, e = Add->getNumOperands(); i != e; ++i)
              if (i != FoundIndex)
                Ops.push_back(Add->getOperand(i));
            const SCEV *Accum = getAddExpr(Ops);

            // This is not a valid addrec if the step amount is varying each
            // loop iteration, but is not itself an addrec in this loop.
            if (isLoopInvariant(Accum, L) ||
                (isa<SCEVAddRecExpr>(Accum) &&
                 cast<SCEVAddRecExpr>(Accum)->getLoop() == L)) {
              SCEV::NoWrapFlags Flags = SCEV::FlagAnyWrap;

              // If the increment doesn't overflow, then neither the addrec nor
              // the post-increment will overflow.
              if (const AddOperator *OBO = dyn_cast<AddOperator>(BEValueV)) {
                if (OBO->getOperand(0) == PN) {
                  if (OBO->hasNoUnsignedWrap())
                    Flags = setFlags(Flags, SCEV::FlagNUW);
                  if (OBO->hasNoSignedWrap())
                    Flags = setFlags(Flags, SCEV::FlagNSW);
                }
              } else if (GEPOperator *GEP = dyn_cast<GEPOperator>(BEValueV)) {
                // If the increment is an inbounds GEP, then we know the address
                // space cannot be wrapped around. We cannot make any guarantee
                // about signed or unsigned overflow because pointers are
                // unsigned but we may have a negative index from the base
                // pointer. We can guarantee that no unsigned wrap occurs if the
                // indices form a positive value.
                if (GEP->isInBounds() && GEP->getOperand(0) == PN) {
                  Flags = setFlags(Flags, SCEV::FlagNW);

                  const SCEV *Ptr = getSCEV(GEP->getPointerOperand());
                  if (isKnownPositive(getMinusSCEV(getSCEV(GEP), Ptr)))
                    Flags = setFlags(Flags, SCEV::FlagNUW);
                }

                // We cannot transfer nuw and nsw flags from subtraction
                // operations -- sub nuw X, Y is not the same as add nuw X, -Y
                // for instance.
              }

              const SCEV *StartVal = getSCEV(StartValueV);
              const SCEV *PHISCEV = getAddRecExpr(StartVal, Accum, L, Flags);

              // Since the no-wrap flags are on the increment, they apply to the
              // post-incremented value as well.
              if (isLoopInvariant(Accum, L))
                (void)getAddRecExpr(getAddExpr(StartVal, Accum),
                                    Accum, L, Flags);

              // Okay, for the entire analysis of this edge we assumed the PHI
              // to be symbolic.  We now need to go back and purge all of the
              // entries for the scalars that use the symbolic expression.
              ForgetSymbolicName(PN, SymbolicName);
              ValueExprMap[SCEVCallbackVH(PN, this)] = PHISCEV;
              return PHISCEV;
            }
          }
        } else if (const SCEVAddRecExpr *AddRec =
                     dyn_cast<SCEVAddRecExpr>(BEValue)) {
          // Otherwise, this could be a loop like this:
          //     i = 0;  for (j = 1; ..; ++j) { ....  i = j; }
          // In this case, j = {1,+,1}  and BEValue is j.
          // Because the other in-value of i (0) fits the evolution of BEValue
          // i really is an addrec evolution.
          if (AddRec->getLoop() == L && AddRec->isAffine()) {
            const SCEV *StartVal = getSCEV(StartValueV);

            // If StartVal = j.start - j.stride, we can use StartVal as the
            // initial step of the addrec evolution.
            if (StartVal == getMinusSCEV(AddRec->getOperand(0),
                                         AddRec->getOperand(1))) {
              // FIXME: For constant StartVal, we should be able to infer
              // no-wrap flags.
              const SCEV *PHISCEV =
                getAddRecExpr(StartVal, AddRec->getOperand(1), L,
                              SCEV::FlagAnyWrap);

              // Okay, for the entire analysis of this edge we assumed the PHI
              // to be symbolic.  We now need to go back and purge all of the
              // entries for the scalars that use the symbolic expression.
              ForgetSymbolicName(PN, SymbolicName);
              ValueExprMap[SCEVCallbackVH(PN, this)] = PHISCEV;
              return PHISCEV;
            }
          }
        }
      }
    }

  // If the PHI has a single incoming value, follow that value, unless the
  // PHI's incoming blocks are in a different loop, in which case doing so
  // risks breaking LCSSA form. Instcombine would normally zap these, but
  // it doesn't have DominatorTree information, so it may miss cases.
  if (Value *V =
          SimplifyInstruction(PN, F->getParent()->getDataLayout(), TLI, DT, AC))
    if (LI->replacementPreservesLCSSAForm(PN, V))
      return getSCEV(V);

  // If it's not a loop phi, we can't handle it yet.
  return getUnknown(PN);
}

/// createNodeForGEP - Expand GEP instructions into add and multiply
/// operations. This allows them to be analyzed by regular SCEV code.
///
const SCEV *ScalarEvolution::createNodeForGEP(GEPOperator *GEP) {
  Value *Base = GEP->getOperand(0);
  // Don't attempt to analyze GEPs over unsized objects.
  if (!Base->getType()->getPointerElementType()->isSized())
    return getUnknown(GEP);

  SmallVector<const SCEV *, 4> IndexExprs;
  for (auto Index = GEP->idx_begin(); Index != GEP->idx_end(); ++Index)
    IndexExprs.push_back(getSCEV(*Index));
  return getGEPExpr(GEP->getSourceElementType(), getSCEV(Base), IndexExprs,
                    GEP->isInBounds());
}

/// GetMinTrailingZeros - Determine the minimum number of zero bits that S is
/// guaranteed to end in (at every loop iteration).  It is, at the same time,
/// the minimum number of times S is divisible by 2.  For example, given {4,+,8}
/// it returns 2.  If S is guaranteed to be 0, it returns the bitwidth of S.
uint32_t
ScalarEvolution::GetMinTrailingZeros(const SCEV *S) {
  if (const SCEVConstant *C = dyn_cast<SCEVConstant>(S))
    return C->getValue()->getValue().countTrailingZeros();

  if (const SCEVTruncateExpr *T = dyn_cast<SCEVTruncateExpr>(S))
    return std::min(GetMinTrailingZeros(T->getOperand()),
                    (uint32_t)getTypeSizeInBits(T->getType()));

  if (const SCEVZeroExtendExpr *E = dyn_cast<SCEVZeroExtendExpr>(S)) {
    uint32_t OpRes = GetMinTrailingZeros(E->getOperand());
    return OpRes == getTypeSizeInBits(E->getOperand()->getType()) ?
             getTypeSizeInBits(E->getType()) : OpRes;
  }

  if (const SCEVSignExtendExpr *E = dyn_cast<SCEVSignExtendExpr>(S)) {
    uint32_t OpRes = GetMinTrailingZeros(E->getOperand());
    return OpRes == getTypeSizeInBits(E->getOperand()->getType()) ?
             getTypeSizeInBits(E->getType()) : OpRes;
  }

  if (const SCEVAddExpr *A = dyn_cast<SCEVAddExpr>(S)) {
    // The result is the min of all operands results.
    uint32_t MinOpRes = GetMinTrailingZeros(A->getOperand(0));
    for (unsigned i = 1, e = A->getNumOperands(); MinOpRes && i != e; ++i)
      MinOpRes = std::min(MinOpRes, GetMinTrailingZeros(A->getOperand(i)));
    return MinOpRes;
  }

  if (const SCEVMulExpr *M = dyn_cast<SCEVMulExpr>(S)) {
    // The result is the sum of all operands results.
    uint32_t SumOpRes = GetMinTrailingZeros(M->getOperand(0));
    uint32_t BitWidth = getTypeSizeInBits(M->getType());
    for (unsigned i = 1, e = M->getNumOperands();
         SumOpRes != BitWidth && i != e; ++i)
      SumOpRes = std::min(SumOpRes + GetMinTrailingZeros(M->getOperand(i)),
                          BitWidth);
    return SumOpRes;
  }

  if (const SCEVAddRecExpr *A = dyn_cast<SCEVAddRecExpr>(S)) {
    // The result is the min of all operands results.
    uint32_t MinOpRes = GetMinTrailingZeros(A->getOperand(0));
    for (unsigned i = 1, e = A->getNumOperands(); MinOpRes && i != e; ++i)
      MinOpRes = std::min(MinOpRes, GetMinTrailingZeros(A->getOperand(i)));
    return MinOpRes;
  }

  if (const SCEVSMaxExpr *M = dyn_cast<SCEVSMaxExpr>(S)) {
    // The result is the min of all operands results.
    uint32_t MinOpRes = GetMinTrailingZeros(M->getOperand(0));
    for (unsigned i = 1, e = M->getNumOperands(); MinOpRes && i != e; ++i)
      MinOpRes = std::min(MinOpRes, GetMinTrailingZeros(M->getOperand(i)));
    return MinOpRes;
  }

  if (const SCEVUMaxExpr *M = dyn_cast<SCEVUMaxExpr>(S)) {
    // The result is the min of all operands results.
    uint32_t MinOpRes = GetMinTrailingZeros(M->getOperand(0));
    for (unsigned i = 1, e = M->getNumOperands(); MinOpRes && i != e; ++i)
      MinOpRes = std::min(MinOpRes, GetMinTrailingZeros(M->getOperand(i)));
    return MinOpRes;
  }

  if (const SCEVUnknown *U = dyn_cast<SCEVUnknown>(S)) {
    // For a SCEVUnknown, ask ValueTracking.
    unsigned BitWidth = getTypeSizeInBits(U->getType());
    APInt Zeros(BitWidth, 0), Ones(BitWidth, 0);
    computeKnownBits(U->getValue(), Zeros, Ones,
                     F->getParent()->getDataLayout(), 0, AC, nullptr, DT);
    return Zeros.countTrailingOnes();
  }

  // SCEVUDivExpr
  return 0;
}

/// GetRangeFromMetadata - Helper method to assign a range to V from
/// metadata present in the IR.
static Optional<ConstantRange> GetRangeFromMetadata(Value *V) {
  if (Instruction *I = dyn_cast<Instruction>(V)) {
    if (MDNode *MD = I->getMetadata(LLVMContext::MD_range)) {
      ConstantRange TotalRange(
          cast<IntegerType>(I->getType())->getBitWidth(), false);

      unsigned NumRanges = MD->getNumOperands() / 2;
      assert(NumRanges >= 1);

      for (unsigned i = 0; i < NumRanges; ++i) {
        ConstantInt *Lower =
            mdconst::extract<ConstantInt>(MD->getOperand(2 * i + 0));
        ConstantInt *Upper =
            mdconst::extract<ConstantInt>(MD->getOperand(2 * i + 1));
        ConstantRange Range(Lower->getValue(), Upper->getValue());
        TotalRange = TotalRange.unionWith(Range);
      }

      return TotalRange;
    }
  }

  return None;
}

/// getRange - Determine the range for a particular SCEV.  If SignHint is
/// HINT_RANGE_UNSIGNED (resp. HINT_RANGE_SIGNED) then getRange prefers ranges
/// with a "cleaner" unsigned (resp. signed) representation.
///
ConstantRange
ScalarEvolution::getRange(const SCEV *S,
                          ScalarEvolution::RangeSignHint SignHint) {
  DenseMap<const SCEV *, ConstantRange> &Cache =
      SignHint == ScalarEvolution::HINT_RANGE_UNSIGNED ? UnsignedRanges
                                                       : SignedRanges;

  // See if we've computed this range already.
  DenseMap<const SCEV *, ConstantRange>::iterator I = Cache.find(S);
  if (I != Cache.end())
    return I->second;

  if (const SCEVConstant *C = dyn_cast<SCEVConstant>(S))
    return setRange(C, SignHint, ConstantRange(C->getValue()->getValue()));

  unsigned BitWidth = getTypeSizeInBits(S->getType());
  ConstantRange ConservativeResult(BitWidth, /*isFullSet=*/true);

  // If the value has known zeros, the maximum value will have those known zeros
  // as well.
  uint32_t TZ = GetMinTrailingZeros(S);
  if (TZ != 0) {
    if (SignHint == ScalarEvolution::HINT_RANGE_UNSIGNED)
      ConservativeResult =
          ConstantRange(APInt::getMinValue(BitWidth),
                        APInt::getMaxValue(BitWidth).lshr(TZ).shl(TZ) + 1);
    else
      ConservativeResult = ConstantRange(
          APInt::getSignedMinValue(BitWidth),
          APInt::getSignedMaxValue(BitWidth).ashr(TZ).shl(TZ) + 1);
  }

  if (const SCEVAddExpr *Add = dyn_cast<SCEVAddExpr>(S)) {
    ConstantRange X = getRange(Add->getOperand(0), SignHint);
    for (unsigned i = 1, e = Add->getNumOperands(); i != e; ++i)
      X = X.add(getRange(Add->getOperand(i), SignHint));
    return setRange(Add, SignHint, ConservativeResult.intersectWith(X));
  }

  if (const SCEVMulExpr *Mul = dyn_cast<SCEVMulExpr>(S)) {
    ConstantRange X = getRange(Mul->getOperand(0), SignHint);
    for (unsigned i = 1, e = Mul->getNumOperands(); i != e; ++i)
      X = X.multiply(getRange(Mul->getOperand(i), SignHint));
    return setRange(Mul, SignHint, ConservativeResult.intersectWith(X));
  }

  if (const SCEVSMaxExpr *SMax = dyn_cast<SCEVSMaxExpr>(S)) {
    ConstantRange X = getRange(SMax->getOperand(0), SignHint);
    for (unsigned i = 1, e = SMax->getNumOperands(); i != e; ++i)
      X = X.smax(getRange(SMax->getOperand(i), SignHint));
    return setRange(SMax, SignHint, ConservativeResult.intersectWith(X));
  }

  if (const SCEVUMaxExpr *UMax = dyn_cast<SCEVUMaxExpr>(S)) {
    ConstantRange X = getRange(UMax->getOperand(0), SignHint);
    for (unsigned i = 1, e = UMax->getNumOperands(); i != e; ++i)
      X = X.umax(getRange(UMax->getOperand(i), SignHint));
    return setRange(UMax, SignHint, ConservativeResult.intersectWith(X));
  }

  if (const SCEVUDivExpr *UDiv = dyn_cast<SCEVUDivExpr>(S)) {
    ConstantRange X = getRange(UDiv->getLHS(), SignHint);
    ConstantRange Y = getRange(UDiv->getRHS(), SignHint);
    return setRange(UDiv, SignHint,
                    ConservativeResult.intersectWith(X.udiv(Y)));
  }

  if (const SCEVZeroExtendExpr *ZExt = dyn_cast<SCEVZeroExtendExpr>(S)) {
    ConstantRange X = getRange(ZExt->getOperand(), SignHint);
    return setRange(ZExt, SignHint,
                    ConservativeResult.intersectWith(X.zeroExtend(BitWidth)));
  }

  if (const SCEVSignExtendExpr *SExt = dyn_cast<SCEVSignExtendExpr>(S)) {
    ConstantRange X = getRange(SExt->getOperand(), SignHint);
    return setRange(SExt, SignHint,
                    ConservativeResult.intersectWith(X.signExtend(BitWidth)));
  }

  if (const SCEVTruncateExpr *Trunc = dyn_cast<SCEVTruncateExpr>(S)) {
    ConstantRange X = getRange(Trunc->getOperand(), SignHint);
    return setRange(Trunc, SignHint,
                    ConservativeResult.intersectWith(X.truncate(BitWidth)));
  }

  if (const SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(S)) {
    // If there's no unsigned wrap, the value will never be less than its
    // initial value.
    if (AddRec->getNoWrapFlags(SCEV::FlagNUW))
      if (const SCEVConstant *C = dyn_cast<SCEVConstant>(AddRec->getStart()))
        if (!C->getValue()->isZero())
          ConservativeResult =
            ConservativeResult.intersectWith(
              ConstantRange(C->getValue()->getValue(), APInt(BitWidth, 0)));

    // If there's no signed wrap, and all the operands have the same sign or
    // zero, the value won't ever change sign.
    if (AddRec->getNoWrapFlags(SCEV::FlagNSW)) {
      bool AllNonNeg = true;
      bool AllNonPos = true;
      for (unsigned i = 0, e = AddRec->getNumOperands(); i != e; ++i) {
        if (!isKnownNonNegative(AddRec->getOperand(i))) AllNonNeg = false;
        if (!isKnownNonPositive(AddRec->getOperand(i))) AllNonPos = false;
      }
      if (AllNonNeg)
        ConservativeResult = ConservativeResult.intersectWith(
          ConstantRange(APInt(BitWidth, 0),
                        APInt::getSignedMinValue(BitWidth)));
      else if (AllNonPos)
        ConservativeResult = ConservativeResult.intersectWith(
          ConstantRange(APInt::getSignedMinValue(BitWidth),
                        APInt(BitWidth, 1)));
    }

    // TODO: non-affine addrec
    if (AddRec->isAffine()) {
      Type *Ty = AddRec->getType();
      const SCEV *MaxBECount = getMaxBackedgeTakenCount(AddRec->getLoop());
      if (!isa<SCEVCouldNotCompute>(MaxBECount) &&
          getTypeSizeInBits(MaxBECount->getType()) <= BitWidth) {

        // Check for overflow.  This must be done with ConstantRange arithmetic
        // because we could be called from within the ScalarEvolution overflow
        // checking code.

        MaxBECount = getNoopOrZeroExtend(MaxBECount, Ty);
        ConstantRange MaxBECountRange = getUnsignedRange(MaxBECount);
        ConstantRange ZExtMaxBECountRange =
            MaxBECountRange.zextOrTrunc(BitWidth * 2 + 1);

        const SCEV *Start = AddRec->getStart();
        const SCEV *Step = AddRec->getStepRecurrence(*this);
        ConstantRange StepSRange = getSignedRange(Step);
        ConstantRange SExtStepSRange = StepSRange.sextOrTrunc(BitWidth * 2 + 1);

        ConstantRange StartURange = getUnsignedRange(Start);
        ConstantRange EndURange =
            StartURange.add(MaxBECountRange.multiply(StepSRange));

        // Check for unsigned overflow.
        ConstantRange ZExtStartURange =
            StartURange.zextOrTrunc(BitWidth * 2 + 1);
        ConstantRange ZExtEndURange = EndURange.zextOrTrunc(BitWidth * 2 + 1);
        if (ZExtStartURange.add(ZExtMaxBECountRange.multiply(SExtStepSRange)) ==
            ZExtEndURange) {
          APInt Min = APIntOps::umin(StartURange.getUnsignedMin(),
                                     EndURange.getUnsignedMin());
          APInt Max = APIntOps::umax(StartURange.getUnsignedMax(),
                                     EndURange.getUnsignedMax());
          bool IsFullRange = Min.isMinValue() && Max.isMaxValue();
          if (!IsFullRange)
            ConservativeResult =
                ConservativeResult.intersectWith(ConstantRange(Min, Max + 1));
        }

        ConstantRange StartSRange = getSignedRange(Start);
        ConstantRange EndSRange =
            StartSRange.add(MaxBECountRange.multiply(StepSRange));

        // Check for signed overflow. This must be done with ConstantRange
        // arithmetic because we could be called from within the ScalarEvolution
        // overflow checking code.
        ConstantRange SExtStartSRange =
            StartSRange.sextOrTrunc(BitWidth * 2 + 1);
        ConstantRange SExtEndSRange = EndSRange.sextOrTrunc(BitWidth * 2 + 1);
        if (SExtStartSRange.add(ZExtMaxBECountRange.multiply(SExtStepSRange)) ==
            SExtEndSRange) {
          APInt Min = APIntOps::smin(StartSRange.getSignedMin(),
                                     EndSRange.getSignedMin());
          APInt Max = APIntOps::smax(StartSRange.getSignedMax(),
                                     EndSRange.getSignedMax());
          bool IsFullRange = Min.isMinSignedValue() && Max.isMaxSignedValue();
          if (!IsFullRange)
            ConservativeResult =
                ConservativeResult.intersectWith(ConstantRange(Min, Max + 1));
        }
      }
    }

    return setRange(AddRec, SignHint, ConservativeResult);
  }

  if (const SCEVUnknown *U = dyn_cast<SCEVUnknown>(S)) {
    // Check if the IR explicitly contains !range metadata.
    Optional<ConstantRange> MDRange = GetRangeFromMetadata(U->getValue());
    if (MDRange.hasValue())
      ConservativeResult = ConservativeResult.intersectWith(MDRange.getValue());

    // Split here to avoid paying the compile-time cost of calling both
    // computeKnownBits and ComputeNumSignBits.  This restriction can be lifted
    // if needed.
    const DataLayout &DL = F->getParent()->getDataLayout();
    if (SignHint == ScalarEvolution::HINT_RANGE_UNSIGNED) {
      // For a SCEVUnknown, ask ValueTracking.
      APInt Zeros(BitWidth, 0), Ones(BitWidth, 0);
      computeKnownBits(U->getValue(), Zeros, Ones, DL, 0, AC, nullptr, DT);
      if (Ones != ~Zeros + 1)
        ConservativeResult =
            ConservativeResult.intersectWith(ConstantRange(Ones, ~Zeros + 1));
    } else {
      assert(SignHint == ScalarEvolution::HINT_RANGE_SIGNED &&
             "generalize as needed!");
      unsigned NS = ComputeNumSignBits(U->getValue(), DL, 0, AC, nullptr, DT);
      if (NS > 1)
        ConservativeResult = ConservativeResult.intersectWith(
            ConstantRange(APInt::getSignedMinValue(BitWidth).ashr(NS - 1),
                          APInt::getSignedMaxValue(BitWidth).ashr(NS - 1) + 1));
    }

    return setRange(U, SignHint, ConservativeResult);
  }

  return setRange(S, SignHint, ConservativeResult);
}

/// createSCEV - We know that there is no SCEV for the specified value.
/// Analyze the expression.
///
const SCEV *ScalarEvolution::createSCEV(Value *V) {
  if (!isSCEVable(V->getType()))
    return getUnknown(V);

  unsigned Opcode = Instruction::UserOp1;
  if (Instruction *I = dyn_cast<Instruction>(V)) {
    Opcode = I->getOpcode();

    // Don't attempt to analyze instructions in blocks that aren't
    // reachable. Such instructions don't matter, and they aren't required
    // to obey basic rules for definitions dominating uses which this
    // analysis depends on.
    if (!DT->isReachableFromEntry(I->getParent()))
      return getUnknown(V);
  } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(V))
    Opcode = CE->getOpcode();
  else if (ConstantInt *CI = dyn_cast<ConstantInt>(V))
    return getConstant(CI);
  else if (isa<ConstantPointerNull>(V))
    return getConstant(V->getType(), 0);
  else if (GlobalAlias *GA = dyn_cast<GlobalAlias>(V))
    return GA->mayBeOverridden() ? getUnknown(V) : getSCEV(GA->getAliasee());
  else
    return getUnknown(V);

  Operator *U = cast<Operator>(V);
  switch (Opcode) {
  case Instruction::Add: {
    // The simple thing to do would be to just call getSCEV on both operands
    // and call getAddExpr with the result. However if we're looking at a
    // bunch of things all added together, this can be quite inefficient,
    // because it leads to N-1 getAddExpr calls for N ultimate operands.
    // Instead, gather up all the operands and make a single getAddExpr call.
    // LLVM IR canonical form means we need only traverse the left operands.
    //
    // Don't apply this instruction's NSW or NUW flags to the new
    // expression. The instruction may be guarded by control flow that the
    // no-wrap behavior depends on. Non-control-equivalent instructions can be
    // mapped to the same SCEV expression, and it would be incorrect to transfer
    // NSW/NUW semantics to those operations.
    SmallVector<const SCEV *, 4> AddOps;
    AddOps.push_back(getSCEV(U->getOperand(1)));
    for (Value *Op = U->getOperand(0); ; Op = U->getOperand(0)) {
      unsigned Opcode = Op->getValueID() - Value::InstructionVal;
      if (Opcode != Instruction::Add && Opcode != Instruction::Sub)
        break;
      U = cast<Operator>(Op);
      const SCEV *Op1 = getSCEV(U->getOperand(1));
      if (Opcode == Instruction::Sub)
        AddOps.push_back(getNegativeSCEV(Op1));
      else
        AddOps.push_back(Op1);
    }
    AddOps.push_back(getSCEV(U->getOperand(0)));
    return getAddExpr(AddOps);
  }
  case Instruction::Mul: {
    // Don't transfer NSW/NUW for the same reason as AddExpr.
    SmallVector<const SCEV *, 4> MulOps;
    MulOps.push_back(getSCEV(U->getOperand(1)));
    for (Value *Op = U->getOperand(0);
         Op->getValueID() == Instruction::Mul + Value::InstructionVal;
         Op = U->getOperand(0)) {
      U = cast<Operator>(Op);
      MulOps.push_back(getSCEV(U->getOperand(1)));
    }
    MulOps.push_back(getSCEV(U->getOperand(0)));
    return getMulExpr(MulOps);
  }
  case Instruction::UDiv:
    return getUDivExpr(getSCEV(U->getOperand(0)),
                       getSCEV(U->getOperand(1)));
  case Instruction::Sub:
    return getMinusSCEV(getSCEV(U->getOperand(0)),
                        getSCEV(U->getOperand(1)));
  case Instruction::And:
    // For an expression like x&255 that merely masks off the high bits,
    // use zext(trunc(x)) as the SCEV expression.
    if (ConstantInt *CI = dyn_cast<ConstantInt>(U->getOperand(1))) {
      if (CI->isNullValue())
        return getSCEV(U->getOperand(1));
      if (CI->isAllOnesValue())
        return getSCEV(U->getOperand(0));
      const APInt &A = CI->getValue();

      // Instcombine's ShrinkDemandedConstant may strip bits out of
      // constants, obscuring what would otherwise be a low-bits mask.
      // Use computeKnownBits to compute what ShrinkDemandedConstant
      // knew about to reconstruct a low-bits mask value.
      unsigned LZ = A.countLeadingZeros();
      unsigned TZ = A.countTrailingZeros();
      unsigned BitWidth = A.getBitWidth();
      APInt KnownZero(BitWidth, 0), KnownOne(BitWidth, 0);
      computeKnownBits(U->getOperand(0), KnownZero, KnownOne,
                       F->getParent()->getDataLayout(), 0, AC, nullptr, DT);

      APInt EffectiveMask =
          APInt::getLowBitsSet(BitWidth, BitWidth - LZ - TZ).shl(TZ);
      if ((LZ != 0 || TZ != 0) && !((~A & ~KnownZero) & EffectiveMask)) {
        const SCEV *MulCount = getConstant(
            ConstantInt::get(getContext(), APInt::getOneBitSet(BitWidth, TZ)));
        return getMulExpr(
            getZeroExtendExpr(
                getTruncateExpr(
                    getUDivExactExpr(getSCEV(U->getOperand(0)), MulCount),
                    IntegerType::get(getContext(), BitWidth - LZ - TZ)),
                U->getType()),
            MulCount);
      }
    }
    break;

  case Instruction::Or:
    // If the RHS of the Or is a constant, we may have something like:
    // X*4+1 which got turned into X*4|1.  Handle this as an Add so loop
    // optimizations will transparently handle this case.
    //
    // In order for this transformation to be safe, the LHS must be of the
    // form X*(2^n) and the Or constant must be less than 2^n.
    if (ConstantInt *CI = dyn_cast<ConstantInt>(U->getOperand(1))) {
      const SCEV *LHS = getSCEV(U->getOperand(0));
      const APInt &CIVal = CI->getValue();
      if (GetMinTrailingZeros(LHS) >=
          (CIVal.getBitWidth() - CIVal.countLeadingZeros())) {
        // Build a plain add SCEV.
        const SCEV *S = getAddExpr(LHS, getSCEV(CI));
        // If the LHS of the add was an addrec and it has no-wrap flags,
        // transfer the no-wrap flags, since an or won't introduce a wrap.
        if (const SCEVAddRecExpr *NewAR = dyn_cast<SCEVAddRecExpr>(S)) {
          const SCEVAddRecExpr *OldAR = cast<SCEVAddRecExpr>(LHS);
          const_cast<SCEVAddRecExpr *>(NewAR)->setNoWrapFlags(
            OldAR->getNoWrapFlags());
        }
        return S;
      }
    }
    break;
  case Instruction::Xor:
    if (ConstantInt *CI = dyn_cast<ConstantInt>(U->getOperand(1))) {
      // If the RHS of the xor is a signbit, then this is just an add.
      // Instcombine turns add of signbit into xor as a strength reduction step.
      if (CI->getValue().isSignBit())
        return getAddExpr(getSCEV(U->getOperand(0)),
                          getSCEV(U->getOperand(1)));

      // If the RHS of xor is -1, then this is a not operation.
      if (CI->isAllOnesValue())
        return getNotSCEV(getSCEV(U->getOperand(0)));

      // Model xor(and(x, C), C) as and(~x, C), if C is a low-bits mask.
      // This is a variant of the check for xor with -1, and it handles
      // the case where instcombine has trimmed non-demanded bits out
      // of an xor with -1.
      if (BinaryOperator *BO = dyn_cast<BinaryOperator>(U->getOperand(0)))
        if (ConstantInt *LCI = dyn_cast<ConstantInt>(BO->getOperand(1)))
          if (BO->getOpcode() == Instruction::And &&
              LCI->getValue() == CI->getValue())
            if (const SCEVZeroExtendExpr *Z =
                  dyn_cast<SCEVZeroExtendExpr>(getSCEV(U->getOperand(0)))) {
              Type *UTy = U->getType();
              const SCEV *Z0 = Z->getOperand();
              Type *Z0Ty = Z0->getType();
              unsigned Z0TySize = getTypeSizeInBits(Z0Ty);

              // If C is a low-bits mask, the zero extend is serving to
              // mask off the high bits. Complement the operand and
              // re-apply the zext.
              if (APIntOps::isMask(Z0TySize, CI->getValue()))
                return getZeroExtendExpr(getNotSCEV(Z0), UTy);

              // If C is a single bit, it may be in the sign-bit position
              // before the zero-extend. In this case, represent the xor
              // using an add, which is equivalent, and re-apply the zext.
              APInt Trunc = CI->getValue().trunc(Z0TySize);
              if (Trunc.zext(getTypeSizeInBits(UTy)) == CI->getValue() &&
                  Trunc.isSignBit())
                return getZeroExtendExpr(getAddExpr(Z0, getConstant(Trunc)),
                                         UTy);
            }
    }
    break;

  case Instruction::Shl:
    // Turn shift left of a constant amount into a multiply.
    if (ConstantInt *SA = dyn_cast<ConstantInt>(U->getOperand(1))) {
      uint32_t BitWidth = cast<IntegerType>(U->getType())->getBitWidth();

      // If the shift count is not less than the bitwidth, the result of
      // the shift is undefined. Don't try to analyze it, because the
      // resolution chosen here may differ from the resolution chosen in
      // other parts of the compiler.
      if (SA->getValue().uge(BitWidth))
        break;

      Constant *X = ConstantInt::get(getContext(),
        APInt::getOneBitSet(BitWidth, SA->getZExtValue()));
      return getMulExpr(getSCEV(U->getOperand(0)), getSCEV(X));
    }
    break;

  case Instruction::LShr:
    // Turn logical shift right of a constant into a unsigned divide.
    if (ConstantInt *SA = dyn_cast<ConstantInt>(U->getOperand(1))) {
      uint32_t BitWidth = cast<IntegerType>(U->getType())->getBitWidth();

      // If the shift count is not less than the bitwidth, the result of
      // the shift is undefined. Don't try to analyze it, because the
      // resolution chosen here may differ from the resolution chosen in
      // other parts of the compiler.
      if (SA->getValue().uge(BitWidth))
        break;

      Constant *X = ConstantInt::get(getContext(),
        APInt::getOneBitSet(BitWidth, SA->getZExtValue()));
      return getUDivExpr(getSCEV(U->getOperand(0)), getSCEV(X));
    }
    break;

  case Instruction::AShr:
    // For a two-shift sext-inreg, use sext(trunc(x)) as the SCEV expression.
    if (ConstantInt *CI = dyn_cast<ConstantInt>(U->getOperand(1)))
      if (Operator *L = dyn_cast<Operator>(U->getOperand(0)))
        if (L->getOpcode() == Instruction::Shl &&
            L->getOperand(1) == U->getOperand(1)) {
          uint64_t BitWidth = getTypeSizeInBits(U->getType());

          // If the shift count is not less than the bitwidth, the result of
          // the shift is undefined. Don't try to analyze it, because the
          // resolution chosen here may differ from the resolution chosen in
          // other parts of the compiler.
          if (CI->getValue().uge(BitWidth))
            break;

          uint64_t Amt = BitWidth - CI->getZExtValue();
          if (Amt == BitWidth)
            return getSCEV(L->getOperand(0));       // shift by zero --> noop
          return
            getSignExtendExpr(getTruncateExpr(getSCEV(L->getOperand(0)),
                                              IntegerType::get(getContext(),
                                                               Amt)),
                              U->getType());
        }
    break;

  case Instruction::Trunc:
    return getTruncateExpr(getSCEV(U->getOperand(0)), U->getType());

  case Instruction::ZExt:
    return getZeroExtendExpr(getSCEV(U->getOperand(0)), U->getType());

  case Instruction::SExt:
    return getSignExtendExpr(getSCEV(U->getOperand(0)), U->getType());

  case Instruction::BitCast:
    // BitCasts are no-op casts so we just eliminate the cast.
    if (isSCEVable(U->getType()) && isSCEVable(U->getOperand(0)->getType()))
      return getSCEV(U->getOperand(0));
    break;

  // It's tempting to handle inttoptr and ptrtoint as no-ops, however this can
  // lead to pointer expressions which cannot safely be expanded to GEPs,
  // because ScalarEvolution doesn't respect the GEP aliasing rules when
  // simplifying integer expressions.

  case Instruction::GetElementPtr:
    return createNodeForGEP(cast<GEPOperator>(U));

  case Instruction::PHI:
    return createNodeForPHI(cast<PHINode>(U));

  case Instruction::Select:
    // This could be a smax or umax that was lowered earlier.
    // Try to recover it.
    if (ICmpInst *ICI = dyn_cast<ICmpInst>(U->getOperand(0))) {
      Value *LHS = ICI->getOperand(0);
      Value *RHS = ICI->getOperand(1);
      switch (ICI->getPredicate()) {
      case ICmpInst::ICMP_SLT:
      case ICmpInst::ICMP_SLE:
        std::swap(LHS, RHS);
        // fall through
      case ICmpInst::ICMP_SGT:
      case ICmpInst::ICMP_SGE:
        // a >s b ? a+x : b+x  ->  smax(a, b)+x
        // a >s b ? b+x : a+x  ->  smin(a, b)+x
        if (getTypeSizeInBits(LHS->getType()) <=
            getTypeSizeInBits(U->getType())) {
          const SCEV *LS = getNoopOrSignExtend(getSCEV(LHS), U->getType());
          const SCEV *RS = getNoopOrSignExtend(getSCEV(RHS), U->getType());
          const SCEV *LA = getSCEV(U->getOperand(1));
          const SCEV *RA = getSCEV(U->getOperand(2));
          const SCEV *LDiff = getMinusSCEV(LA, LS);
          const SCEV *RDiff = getMinusSCEV(RA, RS);
          if (LDiff == RDiff)
            return getAddExpr(getSMaxExpr(LS, RS), LDiff);
          LDiff = getMinusSCEV(LA, RS);
          RDiff = getMinusSCEV(RA, LS);
          if (LDiff == RDiff)
            return getAddExpr(getSMinExpr(LS, RS), LDiff);
        }
        break;
      case ICmpInst::ICMP_ULT:
      case ICmpInst::ICMP_ULE:
        std::swap(LHS, RHS);
        // fall through
      case ICmpInst::ICMP_UGT:
      case ICmpInst::ICMP_UGE:
        // a >u b ? a+x : b+x  ->  umax(a, b)+x
        // a >u b ? b+x : a+x  ->  umin(a, b)+x
        if (getTypeSizeInBits(LHS->getType()) <=
            getTypeSizeInBits(U->getType())) {
          const SCEV *LS = getNoopOrZeroExtend(getSCEV(LHS), U->getType());
          const SCEV *RS = getNoopOrZeroExtend(getSCEV(RHS), U->getType());
          const SCEV *LA = getSCEV(U->getOperand(1));
          const SCEV *RA = getSCEV(U->getOperand(2));
          const SCEV *LDiff = getMinusSCEV(LA, LS);
          const SCEV *RDiff = getMinusSCEV(RA, RS);
          if (LDiff == RDiff)
            return getAddExpr(getUMaxExpr(LS, RS), LDiff);
          LDiff = getMinusSCEV(LA, RS);
          RDiff = getMinusSCEV(RA, LS);
          if (LDiff == RDiff)
            return getAddExpr(getUMinExpr(LS, RS), LDiff);
        }
        break;
      case ICmpInst::ICMP_NE:
        // n != 0 ? n+x : 1+x  ->  umax(n, 1)+x
        if (getTypeSizeInBits(LHS->getType()) <=
                getTypeSizeInBits(U->getType()) &&
            isa<ConstantInt>(RHS) && cast<ConstantInt>(RHS)->isZero()) {
          const SCEV *One = getConstant(U->getType(), 1);
          const SCEV *LS = getNoopOrZeroExtend(getSCEV(LHS), U->getType());
          const SCEV *LA = getSCEV(U->getOperand(1));
          const SCEV *RA = getSCEV(U->getOperand(2));
          const SCEV *LDiff = getMinusSCEV(LA, LS);
          const SCEV *RDiff = getMinusSCEV(RA, One);
          if (LDiff == RDiff)
            return getAddExpr(getUMaxExpr(One, LS), LDiff);
        }
        break;
      case ICmpInst::ICMP_EQ:
        // n == 0 ? 1+x : n+x  ->  umax(n, 1)+x
        if (getTypeSizeInBits(LHS->getType()) <=
                getTypeSizeInBits(U->getType()) &&
            isa<ConstantInt>(RHS) && cast<ConstantInt>(RHS)->isZero()) {
          const SCEV *One = getConstant(U->getType(), 1);
          const SCEV *LS = getNoopOrZeroExtend(getSCEV(LHS), U->getType());
          const SCEV *LA = getSCEV(U->getOperand(1));
          const SCEV *RA = getSCEV(U->getOperand(2));
          const SCEV *LDiff = getMinusSCEV(LA, One);
          const SCEV *RDiff = getMinusSCEV(RA, LS);
          if (LDiff == RDiff)
            return getAddExpr(getUMaxExpr(One, LS), LDiff);
        }
        break;
      default:
        break;
      }
    }

  default: // We cannot analyze this expression.
    break;
  }

  return getUnknown(V);
}



//===----------------------------------------------------------------------===//
//                   Iteration Count Computation Code
//

unsigned ScalarEvolution::getSmallConstantTripCount(Loop *L) {
  if (BasicBlock *ExitingBB = L->getExitingBlock())
    return getSmallConstantTripCount(L, ExitingBB);

  // No trip count information for multiple exits.
  return 0;
}

/// getSmallConstantTripCount - Returns the maximum trip count of this loop as a
/// normal unsigned value. Returns 0 if the trip count is unknown or not
/// constant. Will also return 0 if the maximum trip count is very large (>=
/// 2^32).
///
/// This "trip count" assumes that control exits via ExitingBlock. More
/// precisely, it is the number of times that control may reach ExitingBlock
/// before taking the branch. For loops with multiple exits, it may not be the
/// number times that the loop header executes because the loop may exit
/// prematurely via another branch.
unsigned ScalarEvolution::getSmallConstantTripCount(Loop *L,
                                                    BasicBlock *ExitingBlock) {
  assert(ExitingBlock && "Must pass a non-null exiting block!");
  assert(L->isLoopExiting(ExitingBlock) &&
         "Exiting block must actually branch out of the loop!");
  const SCEVConstant *ExitCount =
      dyn_cast<SCEVConstant>(getExitCount(L, ExitingBlock));
  if (!ExitCount)
    return 0;

  ConstantInt *ExitConst = ExitCount->getValue();

  // Guard against huge trip counts.
  if (ExitConst->getValue().getActiveBits() > 32)
    return 0;

  // In case of integer overflow, this returns 0, which is correct.
  return ((unsigned)ExitConst->getZExtValue()) + 1;
}

unsigned ScalarEvolution::getSmallConstantTripMultiple(Loop *L) {
  if (BasicBlock *ExitingBB = L->getExitingBlock())
    return getSmallConstantTripMultiple(L, ExitingBB);

  // No trip multiple information for multiple exits.
  return 0;
}

/// getSmallConstantTripMultiple - Returns the largest constant divisor of the
/// trip count of this loop as a normal unsigned value, if possible. This
/// means that the actual trip count is always a multiple of the returned
/// value (don't forget the trip count could very well be zero as well!).
///
/// Returns 1 if the trip count is unknown or not guaranteed to be the
/// multiple of a constant (which is also the case if the trip count is simply
/// constant, use getSmallConstantTripCount for that case), Will also return 1
/// if the trip count is very large (>= 2^32).
///
/// As explained in the comments for getSmallConstantTripCount, this assumes
/// that control exits the loop via ExitingBlock.
unsigned
ScalarEvolution::getSmallConstantTripMultiple(Loop *L,
                                              BasicBlock *ExitingBlock) {
  assert(ExitingBlock && "Must pass a non-null exiting block!");
  assert(L->isLoopExiting(ExitingBlock) &&
         "Exiting block must actually branch out of the loop!");
  const SCEV *ExitCount = getExitCount(L, ExitingBlock);
  if (ExitCount == getCouldNotCompute())
    return 1;

  // Get the trip count from the BE count by adding 1.
  const SCEV *TCMul = getAddExpr(ExitCount,
                                 getConstant(ExitCount->getType(), 1));
  // FIXME: SCEV distributes multiplication as V1*C1 + V2*C1. We could attempt
  // to factor simple cases.
  if (const SCEVMulExpr *Mul = dyn_cast<SCEVMulExpr>(TCMul))
    TCMul = Mul->getOperand(0);

  const SCEVConstant *MulC = dyn_cast<SCEVConstant>(TCMul);
  if (!MulC)
    return 1;

  ConstantInt *Result = MulC->getValue();

  // Guard against huge trip counts (this requires checking
  // for zero to handle the case where the trip count == -1 and the
  // addition wraps).
  if (!Result || Result->getValue().getActiveBits() > 32 ||
      Result->getValue().getActiveBits() == 0)
    return 1;

  return (unsigned)Result->getZExtValue();
}

// getExitCount - Get the expression for the number of loop iterations for which
// this loop is guaranteed not to exit via ExitingBlock. Otherwise return
// SCEVCouldNotCompute.
const SCEV *ScalarEvolution::getExitCount(Loop *L, BasicBlock *ExitingBlock) {
  return getBackedgeTakenInfo(L).getExact(ExitingBlock, this);
}

/// getBackedgeTakenCount - If the specified loop has a predictable
/// backedge-taken count, return it, otherwise return a SCEVCouldNotCompute
/// object. The backedge-taken count is the number of times the loop header
/// will be branched to from within the loop. This is one less than the
/// trip count of the loop, since it doesn't count the first iteration,
/// when the header is branched to from outside the loop.
///
/// Note that it is not valid to call this method on a loop without a
/// loop-invariant backedge-taken count (see
/// hasLoopInvariantBackedgeTakenCount).
///
const SCEV *ScalarEvolution::getBackedgeTakenCount(const Loop *L) {
  return getBackedgeTakenInfo(L).getExact(this);
}

/// getMaxBackedgeTakenCount - Similar to getBackedgeTakenCount, except
/// return the least SCEV value that is known never to be less than the
/// actual backedge taken count.
const SCEV *ScalarEvolution::getMaxBackedgeTakenCount(const Loop *L) {
  return getBackedgeTakenInfo(L).getMax(this);
}

/// PushLoopPHIs - Push PHI nodes in the header of the given loop
/// onto the given Worklist.
static void
PushLoopPHIs(const Loop *L, SmallVectorImpl<Instruction *> &Worklist) {
  BasicBlock *Header = L->getHeader();

  // Push all Loop-header PHIs onto the Worklist stack.
  for (BasicBlock::iterator I = Header->begin();
       PHINode *PN = dyn_cast<PHINode>(I); ++I)
    Worklist.push_back(PN);
}

const ScalarEvolution::BackedgeTakenInfo &
ScalarEvolution::getBackedgeTakenInfo(const Loop *L) {
  // Initially insert an invalid entry for this loop. If the insertion
  // succeeds, proceed to actually compute a backedge-taken count and
  // update the value. The temporary CouldNotCompute value tells SCEV
  // code elsewhere that it shouldn't attempt to request a new
  // backedge-taken count, which could result in infinite recursion.
  std::pair<DenseMap<const Loop *, BackedgeTakenInfo>::iterator, bool> Pair =
    BackedgeTakenCounts.insert(std::make_pair(L, BackedgeTakenInfo()));
  if (!Pair.second)
    return Pair.first->second;

  // ComputeBackedgeTakenCount may allocate memory for its result. Inserting it
  // into the BackedgeTakenCounts map transfers ownership. Otherwise, the result
  // must be cleared in this scope.
  BackedgeTakenInfo Result = ComputeBackedgeTakenCount(L);

  if (Result.getExact(this) != getCouldNotCompute()) {
    assert(isLoopInvariant(Result.getExact(this), L) &&
           isLoopInvariant(Result.getMax(this), L) &&
           "Computed backedge-taken count isn't loop invariant for loop!");
    ++NumTripCountsComputed;
  }
  else if (Result.getMax(this) == getCouldNotCompute() &&
           isa<PHINode>(L->getHeader()->begin())) {
    // Only count loops that have phi nodes as not being computable.
    ++NumTripCountsNotComputed;
  }

  // Now that we know more about the trip count for this loop, forget any
  // existing SCEV values for PHI nodes in this loop since they are only
  // conservative estimates made without the benefit of trip count
  // information. This is similar to the code in forgetLoop, except that
  // it handles SCEVUnknown PHI nodes specially.
  if (Result.hasAnyInfo()) {
    SmallVector<Instruction *, 16> Worklist;
    PushLoopPHIs(L, Worklist);

    SmallPtrSet<Instruction *, 8> Visited;
    while (!Worklist.empty()) {
      Instruction *I = Worklist.pop_back_val();
      if (!Visited.insert(I).second)
        continue;

      ValueExprMapType::iterator It =
        ValueExprMap.find_as(static_cast<Value *>(I));
      if (It != ValueExprMap.end()) {
        const SCEV *Old = It->second;

        // SCEVUnknown for a PHI either means that it has an unrecognized
        // structure, or it's a PHI that's in the progress of being computed
        // by createNodeForPHI.  In the former case, additional loop trip
        // count information isn't going to change anything. In the later
        // case, createNodeForPHI will perform the necessary updates on its
        // own when it gets to that point.
        if (!isa<PHINode>(I) || !isa<SCEVUnknown>(Old)) {
          forgetMemoizedResults(Old);
          ValueExprMap.erase(It);
        }
        if (PHINode *PN = dyn_cast<PHINode>(I))
          ConstantEvolutionLoopExitValue.erase(PN);
      }

      PushDefUseChildren(I, Worklist);
    }
  }

  // Re-lookup the insert position, since the call to
  // ComputeBackedgeTakenCount above could result in a
  // recusive call to getBackedgeTakenInfo (on a different
  // loop), which would invalidate the iterator computed
  // earlier.
  return BackedgeTakenCounts.find(L)->second = Result;
}

/// forgetLoop - This method should be called by the client when it has
/// changed a loop in a way that may effect ScalarEvolution's ability to
/// compute a trip count, or if the loop is deleted.
void ScalarEvolution::forgetLoop(const Loop *L) {
  // Drop any stored trip count value.
  DenseMap<const Loop*, BackedgeTakenInfo>::iterator BTCPos =
    BackedgeTakenCounts.find(L);
  if (BTCPos != BackedgeTakenCounts.end()) {
    BTCPos->second.clear();
    BackedgeTakenCounts.erase(BTCPos);
  }

  // Drop information about expressions based on loop-header PHIs.
  SmallVector<Instruction *, 16> Worklist;
  PushLoopPHIs(L, Worklist);

  SmallPtrSet<Instruction *, 8> Visited;
  while (!Worklist.empty()) {
    Instruction *I = Worklist.pop_back_val();
    if (!Visited.insert(I).second)
      continue;

    ValueExprMapType::iterator It =
      ValueExprMap.find_as(static_cast<Value *>(I));
    if (It != ValueExprMap.end()) {
      forgetMemoizedResults(It->second);
      ValueExprMap.erase(It);
      if (PHINode *PN = dyn_cast<PHINode>(I))
        ConstantEvolutionLoopExitValue.erase(PN);
    }

    PushDefUseChildren(I, Worklist);
  }

  // Forget all contained loops too, to avoid dangling entries in the
  // ValuesAtScopes map.
  for (Loop::iterator I = L->begin(), E = L->end(); I != E; ++I)
    forgetLoop(*I);
}

/// forgetValue - This method should be called by the client when it has
/// changed a value in a way that may effect its value, or which may
/// disconnect it from a def-use chain linking it to a loop.
void ScalarEvolution::forgetValue(Value *V) {
  Instruction *I = dyn_cast<Instruction>(V);
  if (!I) return;

  // Drop information about expressions based on loop-header PHIs.
  SmallVector<Instruction *, 16> Worklist;
  Worklist.push_back(I);

  SmallPtrSet<Instruction *, 8> Visited;
  while (!Worklist.empty()) {
    I = Worklist.pop_back_val();
    if (!Visited.insert(I).second)
      continue;

    ValueExprMapType::iterator It =
      ValueExprMap.find_as(static_cast<Value *>(I));
    if (It != ValueExprMap.end()) {
      forgetMemoizedResults(It->second);
      ValueExprMap.erase(It);
      if (PHINode *PN = dyn_cast<PHINode>(I))
        ConstantEvolutionLoopExitValue.erase(PN);
    }

    PushDefUseChildren(I, Worklist);
  }
}

/// getExact - Get the exact loop backedge taken count considering all loop
/// exits. A computable result can only be return for loops with a single exit.
/// Returning the minimum taken count among all exits is incorrect because one
/// of the loop's exit limit's may have been skipped. HowFarToZero assumes that
/// the limit of each loop test is never skipped. This is a valid assumption as
/// long as the loop exits via that test. For precise results, it is the
/// caller's responsibility to specify the relevant loop exit using
/// getExact(ExitingBlock, SE).
const SCEV *
ScalarEvolution::BackedgeTakenInfo::getExact(ScalarEvolution *SE) const {
  // If any exits were not computable, the loop is not computable.
  if (!ExitNotTaken.isCompleteList()) return SE->getCouldNotCompute();

  // We need exactly one computable exit.
  if (!ExitNotTaken.ExitingBlock) return SE->getCouldNotCompute();
  assert(ExitNotTaken.ExactNotTaken && "uninitialized not-taken info");

  const SCEV *BECount = nullptr;
  for (const ExitNotTakenInfo *ENT = &ExitNotTaken;
       ENT != nullptr; ENT = ENT->getNextExit()) {

    assert(ENT->ExactNotTaken != SE->getCouldNotCompute() && "bad exit SCEV");

    if (!BECount)
      BECount = ENT->ExactNotTaken;
    else if (BECount != ENT->ExactNotTaken)
      return SE->getCouldNotCompute();
  }
  assert(BECount && "Invalid not taken count for loop exit");
  return BECount;
}

/// getExact - Get the exact not taken count for this loop exit.
const SCEV *
ScalarEvolution::BackedgeTakenInfo::getExact(BasicBlock *ExitingBlock,
                                             ScalarEvolution *SE) const {
  for (const ExitNotTakenInfo *ENT = &ExitNotTaken;
       ENT != nullptr; ENT = ENT->getNextExit()) {

    if (ENT->ExitingBlock == ExitingBlock)
      return ENT->ExactNotTaken;
  }
  return SE->getCouldNotCompute();
}

/// getMax - Get the max backedge taken count for the loop.
const SCEV *
ScalarEvolution::BackedgeTakenInfo::getMax(ScalarEvolution *SE) const {
  return Max ? Max : SE->getCouldNotCompute();
}

bool ScalarEvolution::BackedgeTakenInfo::hasOperand(const SCEV *S,
                                                    ScalarEvolution *SE) const {
  if (Max && Max != SE->getCouldNotCompute() && SE->hasOperand(Max, S))
    return true;

  if (!ExitNotTaken.ExitingBlock)
    return false;

  for (const ExitNotTakenInfo *ENT = &ExitNotTaken;
       ENT != nullptr; ENT = ENT->getNextExit()) {

    if (ENT->ExactNotTaken != SE->getCouldNotCompute()
        && SE->hasOperand(ENT->ExactNotTaken, S)) {
      return true;
    }
  }
  return false;
}

/// Allocate memory for BackedgeTakenInfo and copy the not-taken count of each
/// computable exit into a persistent ExitNotTakenInfo array.
ScalarEvolution::BackedgeTakenInfo::BackedgeTakenInfo(
  SmallVectorImpl< std::pair<BasicBlock *, const SCEV *> > &ExitCounts,
  bool Complete, const SCEV *MaxCount) : Max(MaxCount) {

  if (!Complete)
    ExitNotTaken.setIncomplete();

  unsigned NumExits = ExitCounts.size();
  if (NumExits == 0) return;

  ExitNotTaken.ExitingBlock = ExitCounts[0].first;
  ExitNotTaken.ExactNotTaken = ExitCounts[0].second;
  if (NumExits == 1) return;

  // Handle the rare case of multiple computable exits.
  ExitNotTakenInfo *ENT = new ExitNotTakenInfo[NumExits-1];

  ExitNotTakenInfo *PrevENT = &ExitNotTaken;
  for (unsigned i = 1; i < NumExits; ++i, PrevENT = ENT, ++ENT) {
    PrevENT->setNextExit(ENT);
    ENT->ExitingBlock = ExitCounts[i].first;
    ENT->ExactNotTaken = ExitCounts[i].second;
  }
}

/// clear - Invalidate this result and free the ExitNotTakenInfo array.
void ScalarEvolution::BackedgeTakenInfo::clear() {
  ExitNotTaken.ExitingBlock = nullptr;
  ExitNotTaken.ExactNotTaken = nullptr;
  delete[] ExitNotTaken.getNextExit();
}

/// ComputeBackedgeTakenCount - Compute the number of times the backedge
/// of the specified loop will execute.
ScalarEvolution::BackedgeTakenInfo
ScalarEvolution::ComputeBackedgeTakenCount(const Loop *L) {
  SmallVector<BasicBlock *, 8> ExitingBlocks;
  L->getExitingBlocks(ExitingBlocks);

  SmallVector<std::pair<BasicBlock *, const SCEV *>, 4> ExitCounts;
  bool CouldComputeBECount = true;
  BasicBlock *Latch = L->getLoopLatch(); // may be NULL.
  const SCEV *MustExitMaxBECount = nullptr;
  const SCEV *MayExitMaxBECount = nullptr;

  // Compute the ExitLimit for each loop exit. Use this to populate ExitCounts
  // and compute maxBECount.
  for (unsigned i = 0, e = ExitingBlocks.size(); i != e; ++i) {
    BasicBlock *ExitBB = ExitingBlocks[i];
    ExitLimit EL = ComputeExitLimit(L, ExitBB);

    // 1. For each exit that can be computed, add an entry to ExitCounts.
    // CouldComputeBECount is true only if all exits can be computed.
    if (EL.Exact == getCouldNotCompute())
      // We couldn't compute an exact value for this exit, so
      // we won't be able to compute an exact value for the loop.
      CouldComputeBECount = false;
    else
      ExitCounts.push_back(std::make_pair(ExitBB, EL.Exact));

    // 2. Derive the loop's MaxBECount from each exit's max number of
    // non-exiting iterations. Partition the loop exits into two kinds:
    // LoopMustExits and LoopMayExits.
    //
    // If the exit dominates the loop latch, it is a LoopMustExit otherwise it
    // is a LoopMayExit.  If any computable LoopMustExit is found, then
    // MaxBECount is the minimum EL.Max of computable LoopMustExits. Otherwise,
    // MaxBECount is conservatively the maximum EL.Max, where CouldNotCompute is
    // considered greater than any computable EL.Max.
    if (EL.Max != getCouldNotCompute() && Latch &&
        DT->dominates(ExitBB, Latch)) {
      if (!MustExitMaxBECount)
        MustExitMaxBECount = EL.Max;
      else {
        MustExitMaxBECount =
          getUMinFromMismatchedTypes(MustExitMaxBECount, EL.Max);
      }
    } else if (MayExitMaxBECount != getCouldNotCompute()) {
      if (!MayExitMaxBECount || EL.Max == getCouldNotCompute())
        MayExitMaxBECount = EL.Max;
      else {
        MayExitMaxBECount =
          getUMaxFromMismatchedTypes(MayExitMaxBECount, EL.Max);
      }
    }
  }
  const SCEV *MaxBECount = MustExitMaxBECount ? MustExitMaxBECount :
    (MayExitMaxBECount ? MayExitMaxBECount : getCouldNotCompute());
  return BackedgeTakenInfo(ExitCounts, CouldComputeBECount, MaxBECount);
}

/// ComputeExitLimit - Compute the number of times the backedge of the specified
/// loop will execute if it exits via the specified block.
ScalarEvolution::ExitLimit
ScalarEvolution::ComputeExitLimit(const Loop *L, BasicBlock *ExitingBlock) {

  // Okay, we've chosen an exiting block.  See what condition causes us to
  // exit at this block and remember the exit block and whether all other targets
  // lead to the loop header.
  bool MustExecuteLoopHeader = true;
  BasicBlock *Exit = nullptr;
  for (succ_iterator SI = succ_begin(ExitingBlock), SE = succ_end(ExitingBlock);
       SI != SE; ++SI)
    if (!L->contains(*SI)) {
      if (Exit) // Multiple exit successors.
        return getCouldNotCompute();
      Exit = *SI;
    } else if (*SI != L->getHeader()) {
      MustExecuteLoopHeader = false;
    }

  // At this point, we know we have a conditional branch that determines whether
  // the loop is exited.  However, we don't know if the branch is executed each
  // time through the loop.  If not, then the execution count of the branch will
  // not be equal to the trip count of the loop.
  //
  // Currently we check for this by checking to see if the Exit branch goes to
  // the loop header.  If so, we know it will always execute the same number of
  // times as the loop.  We also handle the case where the exit block *is* the
  // loop header.  This is common for un-rotated loops.
  //
  // If both of those tests fail, walk up the unique predecessor chain to the
  // header, stopping if there is an edge that doesn't exit the loop. If the
  // header is reached, the execution count of the branch will be equal to the
  // trip count of the loop.
  //
  //  More extensive analysis could be done to handle more cases here.
  //
  if (!MustExecuteLoopHeader && ExitingBlock != L->getHeader()) {
    // The simple checks failed, try climbing the unique predecessor chain
    // up to the header.
    bool Ok = false;
    for (BasicBlock *BB = ExitingBlock; BB; ) {
      BasicBlock *Pred = BB->getUniquePredecessor();
      if (!Pred)
        return getCouldNotCompute();
      TerminatorInst *PredTerm = Pred->getTerminator();
      for (unsigned i = 0, e = PredTerm->getNumSuccessors(); i != e; ++i) {
        BasicBlock *PredSucc = PredTerm->getSuccessor(i);
        if (PredSucc == BB)
          continue;
        // If the predecessor has a successor that isn't BB and isn't
        // outside the loop, assume the worst.
        if (L->contains(PredSucc))
          return getCouldNotCompute();
      }
      if (Pred == L->getHeader()) {
        Ok = true;
        break;
      }
      BB = Pred;
    }
    if (!Ok)
      return getCouldNotCompute();
  }

  bool IsOnlyExit = (L->getExitingBlock() != nullptr);
  TerminatorInst *Term = ExitingBlock->getTerminator();
  if (BranchInst *BI = dyn_cast<BranchInst>(Term)) {
    assert(BI->isConditional() && "If unconditional, it can't be in loop!");
    // Proceed to the next level to examine the exit condition expression.
    return ComputeExitLimitFromCond(L, BI->getCondition(), BI->getSuccessor(0),
                                    BI->getSuccessor(1),
                                    /*ControlsExit=*/IsOnlyExit);
  }

  if (SwitchInst *SI = dyn_cast<SwitchInst>(Term))
    return ComputeExitLimitFromSingleExitSwitch(L, SI, Exit,
                                                /*ControlsExit=*/IsOnlyExit);

  return getCouldNotCompute();
}

/// ComputeExitLimitFromCond - Compute the number of times the
/// backedge of the specified loop will execute if its exit condition
/// were a conditional branch of ExitCond, TBB, and FBB.
///
/// @param ControlsExit is true if ExitCond directly controls the exit
/// branch. In this case, we can assume that the loop exits only if the
/// condition is true and can infer that failing to meet the condition prior to
/// integer wraparound results in undefined behavior.
ScalarEvolution::ExitLimit
ScalarEvolution::ComputeExitLimitFromCond(const Loop *L,
                                          Value *ExitCond,
                                          BasicBlock *TBB,
                                          BasicBlock *FBB,
                                          bool ControlsExit) {
  // Check if the controlling expression for this loop is an And or Or.
  if (BinaryOperator *BO = dyn_cast<BinaryOperator>(ExitCond)) {
    if (BO->getOpcode() == Instruction::And) {
      // Recurse on the operands of the and.
      bool EitherMayExit = L->contains(TBB);
      ExitLimit EL0 = ComputeExitLimitFromCond(L, BO->getOperand(0), TBB, FBB,
                                               ControlsExit && !EitherMayExit);
      ExitLimit EL1 = ComputeExitLimitFromCond(L, BO->getOperand(1), TBB, FBB,
                                               ControlsExit && !EitherMayExit);
      const SCEV *BECount = getCouldNotCompute();
      const SCEV *MaxBECount = getCouldNotCompute();
      if (EitherMayExit) {
        // Both conditions must be true for the loop to continue executing.
        // Choose the less conservative count.
        if (EL0.Exact == getCouldNotCompute() ||
            EL1.Exact == getCouldNotCompute())
          BECount = getCouldNotCompute();
        else
          BECount = getUMinFromMismatchedTypes(EL0.Exact, EL1.Exact);
        if (EL0.Max == getCouldNotCompute())
          MaxBECount = EL1.Max;
        else if (EL1.Max == getCouldNotCompute())
          MaxBECount = EL0.Max;
        else
          MaxBECount = getUMinFromMismatchedTypes(EL0.Max, EL1.Max);
      } else {
        // Both conditions must be true at the same time for the loop to exit.
        // For now, be conservative.
        assert(L->contains(FBB) && "Loop block has no successor in loop!");
        if (EL0.Max == EL1.Max)
          MaxBECount = EL0.Max;
        if (EL0.Exact == EL1.Exact)
          BECount = EL0.Exact;
      }

      return ExitLimit(BECount, MaxBECount);
    }
    if (BO->getOpcode() == Instruction::Or) {
      // Recurse on the operands of the or.
      bool EitherMayExit = L->contains(FBB);
      ExitLimit EL0 = ComputeExitLimitFromCond(L, BO->getOperand(0), TBB, FBB,
                                               ControlsExit && !EitherMayExit);
      ExitLimit EL1 = ComputeExitLimitFromCond(L, BO->getOperand(1), TBB, FBB,
                                               ControlsExit && !EitherMayExit);
      const SCEV *BECount = getCouldNotCompute();
      const SCEV *MaxBECount = getCouldNotCompute();
      if (EitherMayExit) {
        // Both conditions must be false for the loop to continue executing.
        // Choose the less conservative count.
        if (EL0.Exact == getCouldNotCompute() ||
            EL1.Exact == getCouldNotCompute())
          BECount = getCouldNotCompute();
        else
          BECount = getUMinFromMismatchedTypes(EL0.Exact, EL1.Exact);
        if (EL0.Max == getCouldNotCompute())
          MaxBECount = EL1.Max;
        else if (EL1.Max == getCouldNotCompute())
          MaxBECount = EL0.Max;
        else
          MaxBECount = getUMinFromMismatchedTypes(EL0.Max, EL1.Max);
      } else {
        // Both conditions must be false at the same time for the loop to exit.
        // For now, be conservative.
        assert(L->contains(TBB) && "Loop block has no successor in loop!");
        if (EL0.Max == EL1.Max)
          MaxBECount = EL0.Max;
        if (EL0.Exact == EL1.Exact)
          BECount = EL0.Exact;
      }

      return ExitLimit(BECount, MaxBECount);
    }
  }

  // With an icmp, it may be feasible to compute an exact backedge-taken count.
  // Proceed to the next level to examine the icmp.
  if (ICmpInst *ExitCondICmp = dyn_cast<ICmpInst>(ExitCond))
    return ComputeExitLimitFromICmp(L, ExitCondICmp, TBB, FBB, ControlsExit);

  // Check for a constant condition. These are normally stripped out by
  // SimplifyCFG, but ScalarEvolution may be used by a pass which wishes to
  // preserve the CFG and is temporarily leaving constant conditions
  // in place.
  if (ConstantInt *CI = dyn_cast<ConstantInt>(ExitCond)) {
    if (L->contains(FBB) == !CI->getZExtValue())
      // The backedge is always taken.
      return getCouldNotCompute();
    else
      // The backedge is never taken.
      return getConstant(CI->getType(), 0);
  }

  // If it's not an integer or pointer comparison then compute it the hard way.
  return ComputeExitCountExhaustively(L, ExitCond, !L->contains(TBB));
}

/// ComputeExitLimitFromICmp - Compute the number of times the
/// backedge of the specified loop will execute if its exit condition
/// were a conditional branch of the ICmpInst ExitCond, TBB, and FBB.
ScalarEvolution::ExitLimit
ScalarEvolution::ComputeExitLimitFromICmp(const Loop *L,
                                          ICmpInst *ExitCond,
                                          BasicBlock *TBB,
                                          BasicBlock *FBB,
                                          bool ControlsExit) {

  // If the condition was exit on true, convert the condition to exit on false
  ICmpInst::Predicate Cond;
  if (!L->contains(FBB))
    Cond = ExitCond->getPredicate();
  else
    Cond = ExitCond->getInversePredicate();

  // Handle common loops like: for (X = "string"; *X; ++X)
  if (LoadInst *LI = dyn_cast<LoadInst>(ExitCond->getOperand(0)))
    if (Constant *RHS = dyn_cast<Constant>(ExitCond->getOperand(1))) {
      ExitLimit ItCnt =
        ComputeLoadConstantCompareExitLimit(LI, RHS, L, Cond);
      if (ItCnt.hasAnyInfo())
        return ItCnt;
    }

  const SCEV *LHS = getSCEV(ExitCond->getOperand(0));
  const SCEV *RHS = getSCEV(ExitCond->getOperand(1));

  // Try to evaluate any dependencies out of the loop.
  LHS = getSCEVAtScope(LHS, L);
  RHS = getSCEVAtScope(RHS, L);

  // At this point, we would like to compute how many iterations of the
  // loop the predicate will return true for these inputs.
  if (isLoopInvariant(LHS, L) && !isLoopInvariant(RHS, L)) {
    // If there is a loop-invariant, force it into the RHS.
    std::swap(LHS, RHS);
    Cond = ICmpInst::getSwappedPredicate(Cond);
  }

  // Simplify the operands before analyzing them.
  (void)SimplifyICmpOperands(Cond, LHS, RHS);

  // If we have a comparison of a chrec against a constant, try to use value
  // ranges to answer this query.
  if (const SCEVConstant *RHSC = dyn_cast<SCEVConstant>(RHS))
    if (const SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(LHS))
      if (AddRec->getLoop() == L) {
        // Form the constant range.
        ConstantRange CompRange(
            ICmpInst::makeConstantRange(Cond, RHSC->getValue()->getValue()));

        const SCEV *Ret = AddRec->getNumIterationsInRange(CompRange, *this);
        if (!isa<SCEVCouldNotCompute>(Ret)) return Ret;
      }

  // HLSL Change - begin
  // Try to compute the value exhaustively *right now*. Before trying the more pessimistic
  // partial evaluation.
  const SCEV *AggresiveResult = ComputeExitCountExhaustively(L, ExitCond, !L->contains(TBB));
  if (AggresiveResult != getCouldNotCompute())
    return AggresiveResult;
  // HLSL Change - end

  switch (Cond) {
  case ICmpInst::ICMP_NE: {                     // while (X != Y)
    // Convert to: while (X-Y != 0)
    ExitLimit EL = HowFarToZero(getMinusSCEV(LHS, RHS), L, ControlsExit);
    if (EL.hasAnyInfo()) return EL;
    break;
  }
  case ICmpInst::ICMP_EQ: {                     // while (X == Y)
    // Convert to: while (X-Y == 0)
    ExitLimit EL = HowFarToNonZero(getMinusSCEV(LHS, RHS), L);
    if (EL.hasAnyInfo()) return EL;
    break;
  }
  case ICmpInst::ICMP_SLT:
  case ICmpInst::ICMP_ULT: {                    // while (X < Y)
    bool IsSigned = Cond == ICmpInst::ICMP_SLT;
    ExitLimit EL = HowManyLessThans(LHS, RHS, L, IsSigned, ControlsExit);
    if (EL.hasAnyInfo()) return EL;
    break;
  }
  case ICmpInst::ICMP_SGT:
  case ICmpInst::ICMP_UGT: {                    // while (X > Y)
    bool IsSigned = Cond == ICmpInst::ICMP_SGT;
    ExitLimit EL = HowManyGreaterThans(LHS, RHS, L, IsSigned, ControlsExit);
    if (EL.hasAnyInfo()) return EL;
    break;
  }
  default:
#if 0
    dbgs() << "ComputeBackedgeTakenCount ";
    if (ExitCond->getOperand(0)->getType()->isUnsigned())
      dbgs() << "[unsigned] ";
    dbgs() << *LHS << "   "
         << Instruction::getOpcodeName(Instruction::ICmp)
         << "   " << *RHS << "\n";
#endif
    break;
  }
  // return ComputeExitCountExhaustively(L, ExitCond, !L->contains(TBB)); // HLSL Change
  return getCouldNotCompute(); // HLSL Change - We already tried the exhaustive approach earlier, so don't try again and just give up.
}

ScalarEvolution::ExitLimit
ScalarEvolution::ComputeExitLimitFromSingleExitSwitch(const Loop *L,
                                                      SwitchInst *Switch,
                                                      BasicBlock *ExitingBlock,
                                                      bool ControlsExit) {
  assert(!L->contains(ExitingBlock) && "Not an exiting block!");

  // Give up if the exit is the default dest of a switch.
  if (Switch->getDefaultDest() == ExitingBlock)
    return getCouldNotCompute();

  assert(L->contains(Switch->getDefaultDest()) &&
         "Default case must not exit the loop!");
  const SCEV *LHS = getSCEVAtScope(Switch->getCondition(), L);
  const SCEV *RHS = getConstant(Switch->findCaseDest(ExitingBlock));

  // while (X != Y) --> while (X-Y != 0)
  ExitLimit EL = HowFarToZero(getMinusSCEV(LHS, RHS), L, ControlsExit);
  if (EL.hasAnyInfo())
    return EL;

  return getCouldNotCompute();
}

static ConstantInt *
EvaluateConstantChrecAtConstant(const SCEVAddRecExpr *AddRec, ConstantInt *C,
                                ScalarEvolution &SE) {
  const SCEV *InVal = SE.getConstant(C);
  const SCEV *Val = AddRec->evaluateAtIteration(InVal, SE);
  assert(isa<SCEVConstant>(Val) &&
         "Evaluation of SCEV at constant didn't fold correctly?");
  return cast<SCEVConstant>(Val)->getValue();
}

/// ComputeLoadConstantCompareExitLimit - Given an exit condition of
/// 'icmp op load X, cst', try to see if we can compute the backedge
/// execution count.
ScalarEvolution::ExitLimit
ScalarEvolution::ComputeLoadConstantCompareExitLimit(
  LoadInst *LI,
  Constant *RHS,
  const Loop *L,
  ICmpInst::Predicate predicate) {

  if (LI->isVolatile()) return getCouldNotCompute();

  // Check to see if the loaded pointer is a getelementptr of a global.
  // TODO: Use SCEV instead of manually grubbing with GEPs.
  GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(LI->getOperand(0));
  if (!GEP) return getCouldNotCompute();

  // Make sure that it is really a constant global we are gepping, with an
  // initializer, and make sure the first IDX is really 0.
  GlobalVariable *GV = dyn_cast<GlobalVariable>(GEP->getOperand(0));
  if (!GV || !GV->isConstant() || !GV->hasDefinitiveInitializer() ||
      GEP->getNumOperands() < 3 || !isa<Constant>(GEP->getOperand(1)) ||
      !cast<Constant>(GEP->getOperand(1))->isNullValue())
    return getCouldNotCompute();

  // Okay, we allow one non-constant index into the GEP instruction.
  Value *VarIdx = nullptr;
  std::vector<Constant*> Indexes;
  unsigned VarIdxNum = 0;
  for (unsigned i = 2, e = GEP->getNumOperands(); i != e; ++i)
    if (ConstantInt *CI = dyn_cast<ConstantInt>(GEP->getOperand(i))) {
      Indexes.push_back(CI);
    } else if (!isa<ConstantInt>(GEP->getOperand(i))) {
      if (VarIdx) return getCouldNotCompute();  // Multiple non-constant idx's.
      VarIdx = GEP->getOperand(i);
      VarIdxNum = i-2;
      Indexes.push_back(nullptr);
    }

  // Loop-invariant loads may be a byproduct of loop optimization. Skip them.
  if (!VarIdx)
    return getCouldNotCompute();

  // Okay, we know we have a (load (gep GV, 0, X)) comparison with a constant.
  // Check to see if X is a loop variant variable value now.
  const SCEV *Idx = getSCEV(VarIdx);
  Idx = getSCEVAtScope(Idx, L);

  // We can only recognize very limited forms of loop index expressions, in
  // particular, only affine AddRec's like {C1,+,C2}.
  const SCEVAddRecExpr *IdxExpr = dyn_cast<SCEVAddRecExpr>(Idx);
  if (!IdxExpr || !IdxExpr->isAffine() || isLoopInvariant(IdxExpr, L) ||
      !isa<SCEVConstant>(IdxExpr->getOperand(0)) ||
      !isa<SCEVConstant>(IdxExpr->getOperand(1)))
    return getCouldNotCompute();

  unsigned MaxSteps = MaxBruteForceIterations;
  for (unsigned IterationNum = 0; IterationNum != MaxSteps; ++IterationNum) {
    ConstantInt *ItCst = ConstantInt::get(
                           cast<IntegerType>(IdxExpr->getType()), IterationNum);
    ConstantInt *Val = EvaluateConstantChrecAtConstant(IdxExpr, ItCst, *this);

    // Form the GEP offset.
    Indexes[VarIdxNum] = Val;

    Constant *Result = ConstantFoldLoadThroughGEPIndices(GV->getInitializer(),
                                                         Indexes);
    if (!Result) break;  // Cannot compute!

    // Evaluate the condition for this iteration.
    Result = ConstantExpr::getICmp(predicate, Result, RHS);
    if (!isa<ConstantInt>(Result)) break;  // Couldn't decide for sure
    if (cast<ConstantInt>(Result)->getValue().isMinValue()) {
#if 0
      dbgs() << "\n***\n*** Computed loop count " << *ItCst
             << "\n*** From global " << *GV << "*** BB: " << *L->getHeader()
             << "***\n";
#endif
      ++NumArrayLenItCounts;
      return getConstant(ItCst);   // Found terminating iteration!
    }
  }
  return getCouldNotCompute();
}


/// CanConstantFold - Return true if we can constant fold an instruction of the
/// specified type, assuming that all operands were constants.
static bool CanConstantFold(const Instruction *I) {
  if (isa<BinaryOperator>(I) || isa<CmpInst>(I) ||
      isa<SelectInst>(I) || isa<CastInst>(I) || isa<GetElementPtrInst>(I) ||
      isa<LoadInst>(I))
    return true;

  if (const CallInst *CI = dyn_cast<CallInst>(I))
    if (const Function *F = CI->getCalledFunction())
      return canConstantFoldCallTo(F);
  return false;
}

/// Determine whether this instruction can constant evolve within this loop
/// assuming its operands can all constant evolve.
static bool canConstantEvolve(Instruction *I, const Loop *L) {
  // An instruction outside of the loop can't be derived from a loop PHI.
  if (!L->contains(I)) return false;

  if (isa<PHINode>(I)) {
    // We don't currently keep track of the control flow needed to evaluate
    // PHIs, so we cannot handle PHIs inside of loops.
    return L->getHeader() == I->getParent();
  }

  // If we won't be able to constant fold this expression even if the operands
  // are constants, bail early.
  return CanConstantFold(I);
}

/// getConstantEvolvingPHIOperands - Implement getConstantEvolvingPHI by
/// recursing through each instruction operand until reaching a loop header phi.
static PHINode *
getConstantEvolvingPHIOperands(Instruction *UseInst, const Loop *L,
                               DxilValueCache *DVC, // HLSL Change
                               DenseMap<Instruction *, PHINode *> &PHIMap) {

  // Otherwise, we can evaluate this instruction if all of its operands are
  // constant or derived from a PHI node themselves.
  PHINode *PHI = nullptr;
  for (Instruction::op_iterator OpI = UseInst->op_begin(),
         OpE = UseInst->op_end(); OpI != OpE; ++OpI) {

    if (isa<Constant>(*OpI)) continue;

    // HLSL Change begin
    if (DVC->GetConstValue(*OpI)) continue;
    // HLSL Change end

    Instruction *OpInst = dyn_cast<Instruction>(*OpI);
    if (!OpInst || !canConstantEvolve(OpInst, L)) return nullptr;

    PHINode *P = dyn_cast<PHINode>(OpInst);
    if (!P)
      // If this operand is already visited, reuse the prior result.
      // We may have P != PHI if this is the deepest point at which the
      // inconsistent paths meet.
      P = PHIMap.lookup(OpInst);
    if (!P) {
      // Recurse and memoize the results, whether a phi is found or not.
      // This recursive call invalidates pointers into PHIMap.
      //P = getConstantEvolvingPHIOperands(OpInst, L, PHIMap); // HLSL Change
      P = getConstantEvolvingPHIOperands(OpInst, L, DVC, PHIMap); // HLSL Change - Pass DVC
      PHIMap[OpInst] = P;
    }
    if (!P)
      return nullptr;  // Not evolving from PHI
    if (PHI && PHI != P)
      return nullptr;  // Evolving from multiple different PHIs.
    PHI = P;
  }
  // This is a expression evolving from a constant PHI!
  return PHI;
}

/// getConstantEvolvingPHI - Given an LLVM value and a loop, return a PHI node
/// in the loop that V is derived from.  We allow arbitrary operations along the
/// way, but the operands of an operation must either be constants or a value
/// derived from a constant PHI.  If this expression does not fit with these
/// constraints, return null.
// static PHINode *getConstantEvolvingPHI(Value *V, const Loop *L) { // HLSL Change
static PHINode *getConstantEvolvingPHI(Value *V, const Loop *L, DxilValueCache *DVC) { // HLSL Change
  Instruction *I = dyn_cast<Instruction>(V);
  if (!I || !canConstantEvolve(I, L)) return nullptr;

  if (PHINode *PN = dyn_cast<PHINode>(I)) {
    return PN;
  }

  // Record non-constant instructions contained by the loop.
  DenseMap<Instruction *, PHINode *> PHIMap;
  // return getConstantEvolvingPHIOperands(I, L, PHIMap); // HLSL Change
  return getConstantEvolvingPHIOperands(I, L, DVC, PHIMap); // HLSL Change
}

/// EvaluateExpression - Given an expression that passes the
/// getConstantEvolvingPHI predicate, evaluate its value assuming the PHI node
/// in the loop has the value PHIVal.  If we can't fold this expression for some
/// reason, return null.
static Constant *EvaluateExpression(Value *V, const Loop *L,
                                    DenseMap<Instruction *, Constant *> &Vals,
                                    const DataLayout &DL,
                                    const TargetLibraryInfo *TLI) {
  // Convenient constant check, but redundant for recursive calls.
  if (Constant *C = dyn_cast<Constant>(V)) return C;
  Instruction *I = dyn_cast<Instruction>(V);
  if (!I) return nullptr;

  if (Constant *C = Vals.lookup(I)) return C;

  // An instruction inside the loop depends on a value outside the loop that we
  // weren't given a mapping for, or a value such as a call inside the loop.
  if (!canConstantEvolve(I, L)) return nullptr;

  // An unmapped PHI can be due to a branch or another loop inside this loop,
  // or due to this not being the initial iteration through a loop where we
  // couldn't compute the evolution of this particular PHI last time.
  if (isa<PHINode>(I)) return nullptr;

  std::vector<Constant*> Operands(I->getNumOperands());

  for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i) {
    Instruction *Operand = dyn_cast<Instruction>(I->getOperand(i));
    if (!Operand) {
      Operands[i] = dyn_cast<Constant>(I->getOperand(i));
      if (!Operands[i]) return nullptr;
      continue;
    }
    Constant *C = EvaluateExpression(Operand, L, Vals, DL, TLI);
    Vals[Operand] = C;
    if (!C) return nullptr;
    Operands[i] = C;
  }

  if (CmpInst *CI = dyn_cast<CmpInst>(I))
    return ConstantFoldCompareInstOperands(CI->getPredicate(), Operands[0],
                                           Operands[1], DL, TLI);
  if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
    if (!LI->isVolatile())
      return ConstantFoldLoadFromConstPtr(Operands[0], DL);
  }
  return ConstantFoldInstOperands(I->getOpcode(), I->getType(), Operands, DL,
                                  TLI);
}

/// getConstantEvolutionLoopExitValue - If we know that the specified Phi is
/// in the header of its containing loop, we know the loop executes a
/// constant number of times, and the PHI node is just a recurrence
/// involving constants, fold it.
Constant *
ScalarEvolution::getConstantEvolutionLoopExitValue(PHINode *PN,
                                                   const APInt &BEs,
                                                   const Loop *L) {
  DenseMap<PHINode*, Constant*>::const_iterator I =
    ConstantEvolutionLoopExitValue.find(PN);
  if (I != ConstantEvolutionLoopExitValue.end())
    return I->second;

  if (BEs.ugt(MaxBruteForceIterations))
    return ConstantEvolutionLoopExitValue[PN] = nullptr;  // Not going to evaluate it.

  Constant *&RetVal = ConstantEvolutionLoopExitValue[PN];

  DenseMap<Instruction *, Constant *> CurrentIterVals;
  BasicBlock *Header = L->getHeader();
  assert(PN->getParent() == Header && "Can't evaluate PHI not in loop header!");

  // Since the loop is canonicalized, the PHI node must have two entries.  One
  // entry must be a constant (coming in from outside of the loop), and the
  // second must be derived from the same PHI.
  bool SecondIsBackedge = L->contains(PN->getIncomingBlock(1));
  PHINode *PHI = nullptr;
  for (BasicBlock::iterator I = Header->begin();
       (PHI = dyn_cast<PHINode>(I)); ++I) {
    Constant *StartCST =
      dyn_cast<Constant>(PHI->getIncomingValue(!SecondIsBackedge));
    if (!StartCST) continue;
    CurrentIterVals[PHI] = StartCST;
  }
  if (!CurrentIterVals.count(PN))
    return RetVal = nullptr;

  Value *BEValue = PN->getIncomingValue(SecondIsBackedge);

  // Execute the loop symbolically to determine the exit value.
  if (BEs.getActiveBits() >= 32)
    return RetVal = nullptr; // More than 2^32-1 iterations?? Not doing it!

  unsigned NumIterations = BEs.getZExtValue(); // must be in range
  unsigned IterationNum = 0;
  const DataLayout &DL = F->getParent()->getDataLayout();
  for (; ; ++IterationNum) {
    if (IterationNum == NumIterations)
      return RetVal = CurrentIterVals[PN];  // Got exit value!

    // Compute the value of the PHIs for the next iteration.
    // EvaluateExpression adds non-phi values to the CurrentIterVals map.
    DenseMap<Instruction *, Constant *> NextIterVals;
    Constant *NextPHI =
        EvaluateExpression(BEValue, L, CurrentIterVals, DL, TLI);
    if (!NextPHI)
      return nullptr;        // Couldn't evaluate!
    NextIterVals[PN] = NextPHI;

    bool StoppedEvolving = NextPHI == CurrentIterVals[PN];

    // Also evaluate the other PHI nodes.  However, we don't get to stop if we
    // cease to be able to evaluate one of them or if they stop evolving,
    // because that doesn't necessarily prevent us from computing PN.
    SmallVector<std::pair<PHINode *, Constant *>, 8> PHIsToCompute;
    for (DenseMap<Instruction *, Constant *>::const_iterator
           I = CurrentIterVals.begin(), E = CurrentIterVals.end(); I != E; ++I){
      PHINode *PHI = dyn_cast<PHINode>(I->first);
      if (!PHI || PHI == PN || PHI->getParent() != Header) continue;
      PHIsToCompute.push_back(std::make_pair(PHI, I->second));
    }
    // We use two distinct loops because EvaluateExpression may invalidate any
    // iterators into CurrentIterVals.
    for (SmallVectorImpl<std::pair<PHINode *, Constant*> >::const_iterator
             I = PHIsToCompute.begin(), E = PHIsToCompute.end(); I != E; ++I) {
      PHINode *PHI = I->first;
      Constant *&NextPHI = NextIterVals[PHI];
      if (!NextPHI) {   // Not already computed.
        Value *BEValue = PHI->getIncomingValue(SecondIsBackedge);
        NextPHI = EvaluateExpression(BEValue, L, CurrentIterVals, DL, TLI);
      }
      if (NextPHI != I->second)
        StoppedEvolving = false;
    }

    // If all entries in CurrentIterVals == NextIterVals then we can stop
    // iterating, the loop can't continue to change.
    if (StoppedEvolving)
      return RetVal = CurrentIterVals[PN];

    CurrentIterVals.swap(NextIterVals);
  }
}

/// ComputeExitCountExhaustively - If the loop is known to execute a
/// constant number of times (the condition evolves only from constants),
/// try to evaluate a few iterations of the loop until we get the exit
/// condition gets a value of ExitWhen (true or false).  If we cannot
/// evaluate the trip count of the loop, return getCouldNotCompute().
const SCEV *ScalarEvolution::ComputeExitCountExhaustively(const Loop *L,
                                                          Value *Cond,
                                                          bool ExitWhen) {
  // PHINode *PN = getConstantEvolvingPHI(Cond, L); // HLSL Change
  PHINode *PN = getConstantEvolvingPHI(Cond, L, &getAnalysis<DxilValueCache>()); // HLSL Change
  if (!PN) return getCouldNotCompute();

  // If the loop is canonicalized, the PHI will have exactly two entries.
  // That's the only form we support here.
  if (PN->getNumIncomingValues() != 2) return getCouldNotCompute();

  DenseMap<Instruction *, Constant *> CurrentIterVals;
  BasicBlock *Header = L->getHeader();
  assert(PN->getParent() == Header && "Can't evaluate PHI not in loop header!");

  // One entry must be a constant (coming in from outside of the loop), and the
  // second must be derived from the same PHI.
  bool SecondIsBackedge = L->contains(PN->getIncomingBlock(1));
  PHINode *PHI = nullptr;
  for (BasicBlock::iterator I = Header->begin();
       (PHI = dyn_cast<PHINode>(I)); ++I) {

    Constant *StartCST =
      dyn_cast<Constant>(PHI->getIncomingValue(!SecondIsBackedge));

    // HLSL Change begin
    // If we don't have a constant, try getting a constant from the value cache.
    if (!StartCST)
      if (Constant *C = getAnalysis<DxilValueCache>().GetConstValue(PHI->getIncomingValue(!SecondIsBackedge)))
        StartCST = C;
    // HLSL Change end

    if (!StartCST) continue;
    CurrentIterVals[PHI] = StartCST;
  }
  if (!CurrentIterVals.count(PN))
    return getCouldNotCompute();

  // HLSL Change begin
  SmallVector<std::pair<Instruction *, Constant *>, 4> KnownInvariantOps;
  if (Instruction *CondI = dyn_cast<Instruction>(Cond)) {
    SmallVector<Instruction *, 4> Worklist;
    DxilValueCache *DVC = &getAnalysis<DxilValueCache>();

    Worklist.push_back(CondI);
    while (Worklist.size()) {
      Instruction *I = Worklist.pop_back_val();

      if (Constant *C = DVC->GetConstValue(I)) {
        KnownInvariantOps.push_back({ I, C });
      }
      else if (CurrentIterVals.count(I)) {
        continue;
      }
      else if (L->contains(I)) {
        for (Use &U : I->operands()) {
          if (Instruction *OpI = dyn_cast<Instruction>(U.get())) {
            Worklist.push_back(OpI);
          }
        }
      }
    }
  }
  // HLSL Change end

  // Okay, we find a PHI node that defines the trip count of this loop.  Execute
  // the loop symbolically to determine when the condition gets a value of
  // "ExitWhen".
  unsigned MaxIterations = MaxBruteForceIterations;   // Limit analysis.
  const DataLayout &DL = F->getParent()->getDataLayout();
  for (unsigned IterationNum = 0; IterationNum != MaxIterations;++IterationNum){
    // HLSL Change begin
    for (std::pair<Instruction *, Constant *> &Pair : KnownInvariantOps)
      CurrentIterVals[Pair.first] = Pair.second;
    // HLSL Change end

    ConstantInt *CondVal = dyn_cast_or_null<ConstantInt>(
        EvaluateExpression(Cond, L, CurrentIterVals, DL, TLI));

    // Couldn't symbolically evaluate.
    if (!CondVal) return getCouldNotCompute();

    if (CondVal->getValue() == uint64_t(ExitWhen)) {
      ++NumBruteForceTripCountsComputed;
      return getConstant(Type::getInt32Ty(getContext()), IterationNum);
    }

    // Update all the PHI nodes for the next iteration.
    DenseMap<Instruction *, Constant *> NextIterVals;

    // Create a list of which PHIs we need to compute. We want to do this before
    // calling EvaluateExpression on them because that may invalidate iterators
    // into CurrentIterVals.
    SmallVector<PHINode *, 8> PHIsToCompute;
    for (DenseMap<Instruction *, Constant *>::const_iterator
           I = CurrentIterVals.begin(), E = CurrentIterVals.end(); I != E; ++I){
      PHINode *PHI = dyn_cast<PHINode>(I->first);
      if (!PHI || PHI->getParent() != Header) continue;
      PHIsToCompute.push_back(PHI);
    }
    for (SmallVectorImpl<PHINode *>::const_iterator I = PHIsToCompute.begin(),
             E = PHIsToCompute.end(); I != E; ++I) {
      PHINode *PHI = *I;
      Constant *&NextPHI = NextIterVals[PHI];
      if (NextPHI) continue;    // Already computed!

      Value *BEValue = PHI->getIncomingValue(SecondIsBackedge);
      NextPHI = EvaluateExpression(BEValue, L, CurrentIterVals, DL, TLI);
    }
    CurrentIterVals.swap(NextIterVals);
  }

  // Too many iterations were needed to evaluate.
  return getCouldNotCompute();
}

/// getSCEVAtScope - Return a SCEV expression for the specified value
/// at the specified scope in the program.  The L value specifies a loop
/// nest to evaluate the expression at, where null is the top-level or a
/// specified loop is immediately inside of the loop.
///
/// This method can be used to compute the exit value for a variable defined
/// in a loop by querying what the value will hold in the parent loop.
///
/// In the case that a relevant loop exit value cannot be computed, the
/// original value V is returned.
const SCEV *ScalarEvolution::getSCEVAtScope(const SCEV *V, const Loop *L) {
  // Check to see if we've folded this expression at this loop before.
  SmallVector<std::pair<const Loop *, const SCEV *>, 2> &Values = ValuesAtScopes[V];
  for (unsigned u = 0; u < Values.size(); u++) {
    if (Values[u].first == L)
      return Values[u].second ? Values[u].second : V;
  }
  Values.push_back(std::make_pair(L, static_cast<const SCEV *>(nullptr)));
  // Otherwise compute it.
  const SCEV *C = computeSCEVAtScope(V, L);
  SmallVector<std::pair<const Loop *, const SCEV *>, 2> &Values2 = ValuesAtScopes[V];
  for (unsigned u = Values2.size(); u > 0; u--) {
    if (Values2[u - 1].first == L) {
      Values2[u - 1].second = C;
      break;
    }
  }
  return C;
}

/// This builds up a Constant using the ConstantExpr interface.  That way, we
/// will return Constants for objects which aren't represented by a
/// SCEVConstant, because SCEVConstant is restricted to ConstantInt.
/// Returns NULL if the SCEV isn't representable as a Constant.
static Constant *BuildConstantFromSCEV(const SCEV *V) {
  switch (static_cast<SCEVTypes>(V->getSCEVType())) {
    case scCouldNotCompute:
    case scAddRecExpr:
      break;
    case scConstant:
      return cast<SCEVConstant>(V)->getValue();
    case scUnknown:
      return dyn_cast<Constant>(cast<SCEVUnknown>(V)->getValue());
    case scSignExtend: {
      const SCEVSignExtendExpr *SS = cast<SCEVSignExtendExpr>(V);
      if (Constant *CastOp = BuildConstantFromSCEV(SS->getOperand()))
        return ConstantExpr::getSExt(CastOp, SS->getType());
      break;
    }
    case scZeroExtend: {
      const SCEVZeroExtendExpr *SZ = cast<SCEVZeroExtendExpr>(V);
      if (Constant *CastOp = BuildConstantFromSCEV(SZ->getOperand()))
        return ConstantExpr::getZExt(CastOp, SZ->getType());
      break;
    }
    case scTruncate: {
      const SCEVTruncateExpr *ST = cast<SCEVTruncateExpr>(V);
      if (Constant *CastOp = BuildConstantFromSCEV(ST->getOperand()))
        return ConstantExpr::getTrunc(CastOp, ST->getType());
      break;
    }
    case scAddExpr: {
      const SCEVAddExpr *SA = cast<SCEVAddExpr>(V);
      if (Constant *C = BuildConstantFromSCEV(SA->getOperand(0))) {
        if (PointerType *PTy = dyn_cast<PointerType>(C->getType())) {
          unsigned AS = PTy->getAddressSpace();
          Type *DestPtrTy = Type::getInt8PtrTy(C->getContext(), AS);
          C = ConstantExpr::getBitCast(C, DestPtrTy);
        }
        for (unsigned i = 1, e = SA->getNumOperands(); i != e; ++i) {
          Constant *C2 = BuildConstantFromSCEV(SA->getOperand(i));
          if (!C2) return nullptr;

          // First pointer!
          if (!C->getType()->isPointerTy() && C2->getType()->isPointerTy()) {
            unsigned AS = C2->getType()->getPointerAddressSpace();
            std::swap(C, C2);
            Type *DestPtrTy = Type::getInt8PtrTy(C->getContext(), AS);
            // The offsets have been converted to bytes.  We can add bytes to an
            // i8* by GEP with the byte count in the first index.
            C = ConstantExpr::getBitCast(C, DestPtrTy);
          }

          // Don't bother trying to sum two pointers. We probably can't
          // statically compute a load that results from it anyway.
          if (C2->getType()->isPointerTy())
            return nullptr;

          if (PointerType *PTy = dyn_cast<PointerType>(C->getType())) {
            if (PTy->getElementType()->isStructTy())
              C2 = ConstantExpr::getIntegerCast(
                  C2, Type::getInt32Ty(C->getContext()), true);
            C = ConstantExpr::getGetElementPtr(PTy->getElementType(), C, C2);
          } else
            C = ConstantExpr::getAdd(C, C2);
        }
        return C;
      }
      break;
    }
    case scMulExpr: {
      const SCEVMulExpr *SM = cast<SCEVMulExpr>(V);
      if (Constant *C = BuildConstantFromSCEV(SM->getOperand(0))) {
        // Don't bother with pointers at all.
        if (C->getType()->isPointerTy()) return nullptr;
        for (unsigned i = 1, e = SM->getNumOperands(); i != e; ++i) {
          Constant *C2 = BuildConstantFromSCEV(SM->getOperand(i));
          if (!C2 || C2->getType()->isPointerTy()) return nullptr;
          C = ConstantExpr::getMul(C, C2);
        }
        return C;
      }
      break;
    }
    case scUDivExpr: {
      const SCEVUDivExpr *SU = cast<SCEVUDivExpr>(V);
      if (Constant *LHS = BuildConstantFromSCEV(SU->getLHS()))
        if (Constant *RHS = BuildConstantFromSCEV(SU->getRHS()))
          if (LHS->getType() == RHS->getType())
            return ConstantExpr::getUDiv(LHS, RHS);
      break;
    }
    case scSMaxExpr:
    case scUMaxExpr:
      break; // TODO: smax, umax.
  }
  return nullptr;
}

const SCEV *ScalarEvolution::computeSCEVAtScope(const SCEV *V, const Loop *L) {
  if (isa<SCEVConstant>(V)) return V;

  // If this instruction is evolved from a constant-evolving PHI, compute the
  // exit value from the loop without using SCEVs.
  if (const SCEVUnknown *SU = dyn_cast<SCEVUnknown>(V)) {
    if (Instruction *I = dyn_cast<Instruction>(SU->getValue())) {
      const Loop *LI = (*this->LI)[I->getParent()];
      if (LI && LI->getParentLoop() == L)  // Looking for loop exit value.
        if (PHINode *PN = dyn_cast<PHINode>(I))
          if (PN->getParent() == LI->getHeader()) {
            // Okay, there is no closed form solution for the PHI node.  Check
            // to see if the loop that contains it has a known backedge-taken
            // count.  If so, we may be able to force computation of the exit
            // value.
            const SCEV *BackedgeTakenCount = getBackedgeTakenCount(LI);
            if (const SCEVConstant *BTCC =
                  dyn_cast<SCEVConstant>(BackedgeTakenCount)) {
              // Okay, we know how many times the containing loop executes.  If
              // this is a constant evolving PHI node, get the final value at
              // the specified iteration number.
              Constant *RV = getConstantEvolutionLoopExitValue(PN,
                                                   BTCC->getValue()->getValue(),
                                                               LI);
              if (RV) return getSCEV(RV);
            }
          }

      // Okay, this is an expression that we cannot symbolically evaluate
      // into a SCEV.  Check to see if it's possible to symbolically evaluate
      // the arguments into constants, and if so, try to constant propagate the
      // result.  This is particularly useful for computing loop exit values.
      if (CanConstantFold(I)) {
        SmallVector<Constant *, 4> Operands;
        bool MadeImprovement = false;
        for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i) {
          Value *Op = I->getOperand(i);
          if (Constant *C = dyn_cast<Constant>(Op)) {
            Operands.push_back(C);
            continue;
          }

          // If any of the operands is non-constant and if they are
          // non-integer and non-pointer, don't even try to analyze them
          // with scev techniques.
          if (!isSCEVable(Op->getType()))
            return V;

          const SCEV *OrigV = getSCEV(Op);
          const SCEV *OpV = getSCEVAtScope(OrigV, L);
          MadeImprovement |= OrigV != OpV;

          Constant *C = BuildConstantFromSCEV(OpV);
          if (!C) return V;
          if (C->getType() != Op->getType())
            C = ConstantExpr::getCast(CastInst::getCastOpcode(C, false,
                                                              Op->getType(),
                                                              false),
                                      C, Op->getType());
          Operands.push_back(C);
        }

        // Check to see if getSCEVAtScope actually made an improvement.
        if (MadeImprovement) {
          Constant *C = nullptr;
          const DataLayout &DL = F->getParent()->getDataLayout();
          if (const CmpInst *CI = dyn_cast<CmpInst>(I))
            C = ConstantFoldCompareInstOperands(CI->getPredicate(), Operands[0],
                                                Operands[1], DL, TLI);
          else if (const LoadInst *LI = dyn_cast<LoadInst>(I)) {
            if (!LI->isVolatile())
              C = ConstantFoldLoadFromConstPtr(Operands[0], DL);
          } else
            C = ConstantFoldInstOperands(I->getOpcode(), I->getType(), Operands,
                                         DL, TLI);
          if (!C) return V;
          return getSCEV(C);
        }
      }
    }

    // This is some other type of SCEVUnknown, just return it.
    return V;
  }

  if (const SCEVCommutativeExpr *Comm = dyn_cast<SCEVCommutativeExpr>(V)) {
    // Avoid performing the look-up in the common case where the specified
    // expression has no loop-variant portions.
    for (unsigned i = 0, e = Comm->getNumOperands(); i != e; ++i) {
      const SCEV *OpAtScope = getSCEVAtScope(Comm->getOperand(i), L);
      if (OpAtScope != Comm->getOperand(i)) {
        // Okay, at least one of these operands is loop variant but might be
        // foldable.  Build a new instance of the folded commutative expression.
        SmallVector<const SCEV *, 8> NewOps(Comm->op_begin(),
                                            Comm->op_begin()+i);
        NewOps.push_back(OpAtScope);

        for (++i; i != e; ++i) {
          OpAtScope = getSCEVAtScope(Comm->getOperand(i), L);
          NewOps.push_back(OpAtScope);
        }
        if (isa<SCEVAddExpr>(Comm))
          return getAddExpr(NewOps);
        if (isa<SCEVMulExpr>(Comm))
          return getMulExpr(NewOps);
        if (isa<SCEVSMaxExpr>(Comm))
          return getSMaxExpr(NewOps);
        if (isa<SCEVUMaxExpr>(Comm))
          return getUMaxExpr(NewOps);
        llvm_unreachable("Unknown commutative SCEV type!");
      }
    }
    // If we got here, all operands are loop invariant.
    return Comm;
  }

  if (const SCEVUDivExpr *Div = dyn_cast<SCEVUDivExpr>(V)) {
    const SCEV *LHS = getSCEVAtScope(Div->getLHS(), L);
    const SCEV *RHS = getSCEVAtScope(Div->getRHS(), L);
    if (LHS == Div->getLHS() && RHS == Div->getRHS())
      return Div;   // must be loop invariant
    return getUDivExpr(LHS, RHS);
  }

  // If this is a loop recurrence for a loop that does not contain L, then we
  // are dealing with the final value computed by the loop.
  if (const SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(V)) {
    // First, attempt to evaluate each operand.
    // Avoid performing the look-up in the common case where the specified
    // expression has no loop-variant portions.
    for (unsigned i = 0, e = AddRec->getNumOperands(); i != e; ++i) {
      const SCEV *OpAtScope = getSCEVAtScope(AddRec->getOperand(i), L);
      if (OpAtScope == AddRec->getOperand(i))
        continue;

      // Okay, at least one of these operands is loop variant but might be
      // foldable.  Build a new instance of the folded commutative expression.
      SmallVector<const SCEV *, 8> NewOps(AddRec->op_begin(),
                                          AddRec->op_begin()+i);
      NewOps.push_back(OpAtScope);
      for (++i; i != e; ++i)
        NewOps.push_back(getSCEVAtScope(AddRec->getOperand(i), L));

      const SCEV *FoldedRec =
        getAddRecExpr(NewOps, AddRec->getLoop(),
                      AddRec->getNoWrapFlags(SCEV::FlagNW));
      AddRec = dyn_cast<SCEVAddRecExpr>(FoldedRec);
      // The addrec may be folded to a nonrecurrence, for example, if the
      // induction variable is multiplied by zero after constant folding. Go
      // ahead and return the folded value.
      if (!AddRec)
        return FoldedRec;
      break;
    }

    // If the scope is outside the addrec's loop, evaluate it by using the
    // loop exit value of the addrec.
    if (!AddRec->getLoop()->contains(L)) {
      // To evaluate this recurrence, we need to know how many times the AddRec
      // loop iterates.  Compute this now.
      const SCEV *BackedgeTakenCount = getBackedgeTakenCount(AddRec->getLoop());
      if (BackedgeTakenCount == getCouldNotCompute()) return AddRec;

      // Then, evaluate the AddRec.
      return AddRec->evaluateAtIteration(BackedgeTakenCount, *this);
    }

    return AddRec;
  }

  if (const SCEVZeroExtendExpr *Cast = dyn_cast<SCEVZeroExtendExpr>(V)) {
    const SCEV *Op = getSCEVAtScope(Cast->getOperand(), L);
    if (Op == Cast->getOperand())
      return Cast;  // must be loop invariant
    return getZeroExtendExpr(Op, Cast->getType());
  }

  if (const SCEVSignExtendExpr *Cast = dyn_cast<SCEVSignExtendExpr>(V)) {
    const SCEV *Op = getSCEVAtScope(Cast->getOperand(), L);
    if (Op == Cast->getOperand())
      return Cast;  // must be loop invariant
    return getSignExtendExpr(Op, Cast->getType());
  }

  if (const SCEVTruncateExpr *Cast = dyn_cast<SCEVTruncateExpr>(V)) {
    const SCEV *Op = getSCEVAtScope(Cast->getOperand(), L);
    if (Op == Cast->getOperand())
      return Cast;  // must be loop invariant
    return getTruncateExpr(Op, Cast->getType());
  }

  llvm_unreachable("Unknown SCEV type!");
}

/// getSCEVAtScope - This is a convenience function which does
/// getSCEVAtScope(getSCEV(V), L).
const SCEV *ScalarEvolution::getSCEVAtScope(Value *V, const Loop *L) {
  return getSCEVAtScope(getSCEV(V), L);
}

/// SolveLinEquationWithOverflow - Finds the minimum unsigned root of the
/// following equation:
///
///     A * X = B (mod N)
///
/// where N = 2^BW and BW is the common bit width of A and B. The signedness of
/// A and B isn't important.
///
/// If the equation does not have a solution, SCEVCouldNotCompute is returned.
static const SCEV *SolveLinEquationWithOverflow(const APInt &A, const APInt &B,
                                               ScalarEvolution &SE) {
  uint32_t BW = A.getBitWidth();
  assert(BW == B.getBitWidth() && "Bit widths must be the same.");
  assert(A != 0 && "A must be non-zero.");

  // 1. D = gcd(A, N)
  //
  // The gcd of A and N may have only one prime factor: 2. The number of
  // trailing zeros in A is its multiplicity
  uint32_t Mult2 = A.countTrailingZeros();
  // D = 2^Mult2

  // 2. Check if B is divisible by D.
  //
  // B is divisible by D if and only if the multiplicity of prime factor 2 for B
  // is not less than multiplicity of this prime factor for D.
  if (B.countTrailingZeros() < Mult2)
    return SE.getCouldNotCompute();

  // 3. Compute I: the multiplicative inverse of (A / D) in arithmetic
  // modulo (N / D).
  //
  // (N / D) may need BW+1 bits in its representation.  Hence, we'll use this
  // bit width during computations.
  APInt AD = A.lshr(Mult2).zext(BW + 1);  // AD = A / D
  APInt Mod(BW + 1, 0);
  Mod.setBit(BW - Mult2);  // Mod = N / D
  APInt I = AD.multiplicativeInverse(Mod);

  // 4. Compute the minimum unsigned root of the equation:
  // I * (B / D) mod (N / D)
  APInt Result = (I * B.lshr(Mult2).zext(BW + 1)).urem(Mod);

  // The result is guaranteed to be less than 2^BW so we may truncate it to BW
  // bits.
  return SE.getConstant(Result.trunc(BW));
}

/// SolveQuadraticEquation - Find the roots of the quadratic equation for the
/// given quadratic chrec {L,+,M,+,N}.  This returns either the two roots (which
/// might be the same) or two SCEVCouldNotCompute objects.
///
static std::pair<const SCEV *,const SCEV *>
SolveQuadraticEquation(const SCEVAddRecExpr *AddRec, ScalarEvolution &SE) {
  assert(AddRec->getNumOperands() == 3 && "This is not a quadratic chrec!");
  const SCEVConstant *LC = dyn_cast<SCEVConstant>(AddRec->getOperand(0));
  const SCEVConstant *MC = dyn_cast<SCEVConstant>(AddRec->getOperand(1));
  const SCEVConstant *NC = dyn_cast<SCEVConstant>(AddRec->getOperand(2));

  // We currently can only solve this if the coefficients are constants.
  if (!LC || !MC || !NC) {
    const SCEV *CNC = SE.getCouldNotCompute();
    return std::make_pair(CNC, CNC);
  }

  uint32_t BitWidth = LC->getValue()->getValue().getBitWidth();
  const APInt &L = LC->getValue()->getValue();
  const APInt &M = MC->getValue()->getValue();
  const APInt &N = NC->getValue()->getValue();
  APInt Two(BitWidth, 2);
  APInt Four(BitWidth, 4);

  {
    using namespace APIntOps;
    const APInt& C = L;
    // Convert from chrec coefficients to polynomial coefficients AX^2+BX+C
    // The B coefficient is M-N/2
    APInt B(M);
    B -= sdiv(N,Two);

    // The A coefficient is N/2
    APInt A(N.sdiv(Two));

    // Compute the B^2-4ac term.
    APInt SqrtTerm(B);
    SqrtTerm *= B;
    SqrtTerm -= Four * (A * C);

    if (SqrtTerm.isNegative()) {
      // The loop is provably infinite.
      const SCEV *CNC = SE.getCouldNotCompute();
      return std::make_pair(CNC, CNC);
    }

    // Compute sqrt(B^2-4ac). This is guaranteed to be the nearest
    // integer value or else APInt::sqrt() will assert.
    APInt SqrtVal(SqrtTerm.sqrt());

    // Compute the two solutions for the quadratic formula.
    // The divisions must be performed as signed divisions.
    APInt NegB(-B);
    APInt TwoA(A << 1);
    if (TwoA.isMinValue()) {
      const SCEV *CNC = SE.getCouldNotCompute();
      return std::make_pair(CNC, CNC);
    }

    LLVMContext &Context = SE.getContext();

    ConstantInt *Solution1 =
      ConstantInt::get(Context, (NegB + SqrtVal).sdiv(TwoA));
    ConstantInt *Solution2 =
      ConstantInt::get(Context, (NegB - SqrtVal).sdiv(TwoA));

    return std::make_pair(SE.getConstant(Solution1),
                          SE.getConstant(Solution2));
  } // end APIntOps namespace
}

/// HowFarToZero - Return the number of times a backedge comparing the specified
/// value to zero will execute.  If not computable, return CouldNotCompute.
///
/// This is only used for loops with a "x != y" exit test. The exit condition is
/// now expressed as a single expression, V = x-y. So the exit test is
/// effectively V != 0.  We know and take advantage of the fact that this
/// expression only being used in a comparison by zero context.
ScalarEvolution::ExitLimit
ScalarEvolution::HowFarToZero(const SCEV *V, const Loop *L, bool ControlsExit) {
  // If the value is a constant
  if (const SCEVConstant *C = dyn_cast<SCEVConstant>(V)) {
    // If the value is already zero, the branch will execute zero times.
    if (C->getValue()->isZero()) return C;
    return getCouldNotCompute();  // Otherwise it will loop infinitely.
  }

  const SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(V);
  if (!AddRec || AddRec->getLoop() != L)
    return getCouldNotCompute();

  // If this is a quadratic (3-term) AddRec {L,+,M,+,N}, find the roots of
  // the quadratic equation to solve it.
  if (AddRec->isQuadratic() && AddRec->getType()->isIntegerTy()) {
    std::pair<const SCEV *,const SCEV *> Roots =
      SolveQuadraticEquation(AddRec, *this);
    const SCEVConstant *R1 = dyn_cast<SCEVConstant>(Roots.first);
    const SCEVConstant *R2 = dyn_cast<SCEVConstant>(Roots.second);
    if (R1 && R2) {
#if 0
      dbgs() << "HFTZ: " << *V << " - sol#1: " << *R1
             << "  sol#2: " << *R2 << "\n";
#endif
      // Pick the smallest positive root value.
      if (ConstantInt *CB =
          dyn_cast<ConstantInt>(ConstantExpr::getICmp(CmpInst::ICMP_ULT,
                                                      R1->getValue(),
                                                      R2->getValue()))) {
        if (!CB->getZExtValue())
          std::swap(R1, R2);   // R1 is the minimum root now.

        // We can only use this value if the chrec ends up with an exact zero
        // value at this index.  When solving for "X*X != 5", for example, we
        // should not accept a root of 2.
        const SCEV *Val = AddRec->evaluateAtIteration(R1, *this);
        if (Val->isZero())
          return R1;  // We found a quadratic root!
      }
    }
    return getCouldNotCompute();
  }

  // Otherwise we can only handle this if it is affine.
  if (!AddRec->isAffine())
    return getCouldNotCompute();

  // If this is an affine expression, the execution count of this branch is
  // the minimum unsigned root of the following equation:
  //
  //     Start + Step*N = 0 (mod 2^BW)
  //
  // equivalent to:
  //
  //             Step*N = -Start (mod 2^BW)
  //
  // where BW is the common bit width of Start and Step.

  // Get the initial value for the loop.
  const SCEV *Start = getSCEVAtScope(AddRec->getStart(), L->getParentLoop());
  const SCEV *Step = getSCEVAtScope(AddRec->getOperand(1), L->getParentLoop());

  // For now we handle only constant steps.
  //
  // TODO: Handle a nonconstant Step given AddRec<NUW>. If the
  // AddRec is NUW, then (in an unsigned sense) it cannot be counting up to wrap
  // to 0, it must be counting down to equal 0. Consequently, N = Start / -Step.
  // We have not yet seen any such cases.
  const SCEVConstant *StepC = dyn_cast<SCEVConstant>(Step);
  if (!StepC || StepC->getValue()->equalsInt(0))
    return getCouldNotCompute();

  // For positive steps (counting up until unsigned overflow):
  //   N = -Start/Step (as unsigned)
  // For negative steps (counting down to zero):
  //   N = Start/-Step
  // First compute the unsigned distance from zero in the direction of Step.
  bool CountDown = StepC->getValue()->getValue().isNegative();
  const SCEV *Distance = CountDown ? Start : getNegativeSCEV(Start);

  // Handle unitary steps, which cannot wraparound.
  // 1*N = -Start; -1*N = Start (mod 2^BW), so:
  //   N = Distance (as unsigned)
  if (StepC->getValue()->equalsInt(1) || StepC->getValue()->isAllOnesValue()) {
    ConstantRange CR = getUnsignedRange(Start);
    const SCEV *MaxBECount;
    if (!CountDown && CR.getUnsignedMin().isMinValue())
      // When counting up, the worst starting value is 1, not 0.
      MaxBECount = CR.getUnsignedMax().isMinValue()
        ? getConstant(APInt::getMinValue(CR.getBitWidth()))
        : getConstant(APInt::getMaxValue(CR.getBitWidth()));
    else
      MaxBECount = getConstant(CountDown ? CR.getUnsignedMax()
                                         : -CR.getUnsignedMin());
    return ExitLimit(Distance, MaxBECount);
  }

  // As a special case, handle the instance where Step is a positive power of
  // two. In this case, determining whether Step divides Distance evenly can be
  // done by counting and comparing the number of trailing zeros of Step and
  // Distance.
  if (!CountDown) {
    const APInt &StepV = StepC->getValue()->getValue();
    // StepV.isPowerOf2() returns true if StepV is an positive power of two.  It
    // also returns true if StepV is maximally negative (eg, INT_MIN), but that
    // case is not handled as this code is guarded by !CountDown.
    if (StepV.isPowerOf2() &&
        GetMinTrailingZeros(Distance) >= StepV.countTrailingZeros())
      return getUDivExactExpr(Distance, Step);
  }

  // If the condition controls loop exit (the loop exits only if the expression
  // is true) and the addition is no-wrap we can use unsigned divide to
  // compute the backedge count.  In this case, the step may not divide the
  // distance, but we don't care because if the condition is "missed" the loop
  // will have undefined behavior due to wrapping.
  if (ControlsExit && AddRec->getNoWrapFlags(SCEV::FlagNW)) {
    const SCEV *Exact =
        getUDivExpr(Distance, CountDown ? getNegativeSCEV(Step) : Step);
    return ExitLimit(Exact, Exact);
  }

  // Then, try to solve the above equation provided that Start is constant.
  if (const SCEVConstant *StartC = dyn_cast<SCEVConstant>(Start))
    return SolveLinEquationWithOverflow(StepC->getValue()->getValue(),
                                        -StartC->getValue()->getValue(),
                                        *this);
  return getCouldNotCompute();
}

/// HowFarToNonZero - Return the number of times a backedge checking the
/// specified value for nonzero will execute.  If not computable, return
/// CouldNotCompute
ScalarEvolution::ExitLimit
ScalarEvolution::HowFarToNonZero(const SCEV *V, const Loop *L) {
  // Loops that look like: while (X == 0) are very strange indeed.  We don't
  // handle them yet except for the trivial case.  This could be expanded in the
  // future as needed.

  // If the value is a constant, check to see if it is known to be non-zero
  // already.  If so, the backedge will execute zero times.
  if (const SCEVConstant *C = dyn_cast<SCEVConstant>(V)) {
    if (!C->getValue()->isNullValue())
      return getConstant(C->getType(), 0);
    return getCouldNotCompute();  // Otherwise it will loop infinitely.
  }

  // We could implement others, but I really doubt anyone writes loops like
  // this, and if they did, they would already be constant folded.
  return getCouldNotCompute();
}

/// getPredecessorWithUniqueSuccessorForBB - Return a predecessor of BB
/// (which may not be an immediate predecessor) which has exactly one
/// successor from which BB is reachable, or null if no such block is
/// found.
///
std::pair<BasicBlock *, BasicBlock *>
ScalarEvolution::getPredecessorWithUniqueSuccessorForBB(BasicBlock *BB) {
  // If the block has a unique predecessor, then there is no path from the
  // predecessor to the block that does not go through the direct edge
  // from the predecessor to the block.
  if (BasicBlock *Pred = BB->getSinglePredecessor())
    return std::make_pair(Pred, BB);

  // A loop's header is defined to be a block that dominates the loop.
  // If the header has a unique predecessor outside the loop, it must be
  // a block that has exactly one successor that can reach the loop.
  if (Loop *L = LI->getLoopFor(BB))
    return std::make_pair(L->getLoopPredecessor(), L->getHeader());

  return std::pair<BasicBlock *, BasicBlock *>();
}

/// HasSameValue - SCEV structural equivalence is usually sufficient for
/// testing whether two expressions are equal, however for the purposes of
/// looking for a condition guarding a loop, it can be useful to be a little
/// more general, since a front-end may have replicated the controlling
/// expression.
///
static bool HasSameValue(const SCEV *A, const SCEV *B) {
  // Quick check to see if they are the same SCEV.
  if (A == B) return true;

  // Otherwise, if they're both SCEVUnknown, it's possible that they hold
  // two different instructions with the same value. Check for this case.
  if (const SCEVUnknown *AU = dyn_cast<SCEVUnknown>(A))
    if (const SCEVUnknown *BU = dyn_cast<SCEVUnknown>(B))
      if (const Instruction *AI = dyn_cast<Instruction>(AU->getValue()))
        if (const Instruction *BI = dyn_cast<Instruction>(BU->getValue()))
          if (AI->isIdenticalTo(BI) && !AI->mayReadFromMemory())
            return true;

  // Otherwise assume they may have a different value.
  return false;
}

/// SimplifyICmpOperands - Simplify LHS and RHS in a comparison with
/// predicate Pred. Return true iff any changes were made.
///
bool ScalarEvolution::SimplifyICmpOperands(ICmpInst::Predicate &Pred,
                                           const SCEV *&LHS, const SCEV *&RHS,
                                           unsigned Depth) {
  bool Changed = false;

  // If we hit the max recursion limit bail out.
  if (Depth >= 3)
    return false;

  // Canonicalize a constant to the right side.
  if (const SCEVConstant *LHSC = dyn_cast<SCEVConstant>(LHS)) {
    // Check for both operands constant.
    if (const SCEVConstant *RHSC = dyn_cast<SCEVConstant>(RHS)) {
      if (ConstantExpr::getICmp(Pred,
                                LHSC->getValue(),
                                RHSC->getValue())->isNullValue())
        goto trivially_false;
      else
        goto trivially_true;
    }
    // Otherwise swap the operands to put the constant on the right.
    std::swap(LHS, RHS);
    Pred = ICmpInst::getSwappedPredicate(Pred);
    Changed = true;
  }

  // If we're comparing an addrec with a value which is loop-invariant in the
  // addrec's loop, put the addrec on the left. Also make a dominance check,
  // as both operands could be addrecs loop-invariant in each other's loop.
  if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(RHS)) {
    const Loop *L = AR->getLoop();
    if (isLoopInvariant(LHS, L) && properlyDominates(LHS, L->getHeader())) {
      std::swap(LHS, RHS);
      Pred = ICmpInst::getSwappedPredicate(Pred);
      Changed = true;
    }
  }

  // If there's a constant operand, canonicalize comparisons with boundary
  // cases, and canonicalize *-or-equal comparisons to regular comparisons.
  if (const SCEVConstant *RC = dyn_cast<SCEVConstant>(RHS)) {
    const APInt &RA = RC->getValue()->getValue();
    switch (Pred) {
    default: llvm_unreachable("Unexpected ICmpInst::Predicate value!");
    case ICmpInst::ICMP_EQ:
    case ICmpInst::ICMP_NE:
      // Fold ((-1) * %a) + %b == 0 (equivalent to %b-%a == 0) into %a == %b.
      if (!RA)
        if (const SCEVAddExpr *AE = dyn_cast<SCEVAddExpr>(LHS))
          if (const SCEVMulExpr *ME = dyn_cast<SCEVMulExpr>(AE->getOperand(0)))
            if (AE->getNumOperands() == 2 && ME->getNumOperands() == 2 &&
                ME->getOperand(0)->isAllOnesValue()) {
              RHS = AE->getOperand(1);
              LHS = ME->getOperand(1);
              Changed = true;
            }
      break;
    case ICmpInst::ICMP_UGE:
      if ((RA - 1).isMinValue()) {
        Pred = ICmpInst::ICMP_NE;
        RHS = getConstant(RA - 1);
        Changed = true;
        break;
      }
      if (RA.isMaxValue()) {
        Pred = ICmpInst::ICMP_EQ;
        Changed = true;
        break;
      }
      if (RA.isMinValue()) goto trivially_true;

      Pred = ICmpInst::ICMP_UGT;
      RHS = getConstant(RA - 1);
      Changed = true;
      break;
    case ICmpInst::ICMP_ULE:
      if ((RA + 1).isMaxValue()) {
        Pred = ICmpInst::ICMP_NE;
        RHS = getConstant(RA + 1);
        Changed = true;
        break;
      }
      if (RA.isMinValue()) {
        Pred = ICmpInst::ICMP_EQ;
        Changed = true;
        break;
      }
      if (RA.isMaxValue()) goto trivially_true;

      Pred = ICmpInst::ICMP_ULT;
      RHS = getConstant(RA + 1);
      Changed = true;
      break;
    case ICmpInst::ICMP_SGE:
      if ((RA - 1).isMinSignedValue()) {
        Pred = ICmpInst::ICMP_NE;
        RHS = getConstant(RA - 1);
        Changed = true;
        break;
      }
      if (RA.isMaxSignedValue()) {
        Pred = ICmpInst::ICMP_EQ;
        Changed = true;
        break;
      }
      if (RA.isMinSignedValue()) goto trivially_true;

      Pred = ICmpInst::ICMP_SGT;
      RHS = getConstant(RA - 1);
      Changed = true;
      break;
    case ICmpInst::ICMP_SLE:
      if ((RA + 1).isMaxSignedValue()) {
        Pred = ICmpInst::ICMP_NE;
        RHS = getConstant(RA + 1);
        Changed = true;
        break;
      }
      if (RA.isMinSignedValue()) {
        Pred = ICmpInst::ICMP_EQ;
        Changed = true;
        break;
      }
      if (RA.isMaxSignedValue()) goto trivially_true;

      Pred = ICmpInst::ICMP_SLT;
      RHS = getConstant(RA + 1);
      Changed = true;
      break;
    case ICmpInst::ICMP_UGT:
      if (RA.isMinValue()) {
        Pred = ICmpInst::ICMP_NE;
        Changed = true;
        break;
      }
      if ((RA + 1).isMaxValue()) {
        Pred = ICmpInst::ICMP_EQ;
        RHS = getConstant(RA + 1);
        Changed = true;
        break;
      }
      if (RA.isMaxValue()) goto trivially_false;
      break;
    case ICmpInst::ICMP_ULT:
      if (RA.isMaxValue()) {
        Pred = ICmpInst::ICMP_NE;
        Changed = true;
        break;
      }
      if ((RA - 1).isMinValue()) {
        Pred = ICmpInst::ICMP_EQ;
        RHS = getConstant(RA - 1);
        Changed = true;
        break;
      }
      if (RA.isMinValue()) goto trivially_false;
      break;
    case ICmpInst::ICMP_SGT:
      if (RA.isMinSignedValue()) {
        Pred = ICmpInst::ICMP_NE;
        Changed = true;
        break;
      }
      if ((RA + 1).isMaxSignedValue()) {
        Pred = ICmpInst::ICMP_EQ;
        RHS = getConstant(RA + 1);
        Changed = true;
        break;
      }
      if (RA.isMaxSignedValue()) goto trivially_false;
      break;
    case ICmpInst::ICMP_SLT:
      if (RA.isMaxSignedValue()) {
        Pred = ICmpInst::ICMP_NE;
        Changed = true;
        break;
      }
      if ((RA - 1).isMinSignedValue()) {
       Pred = ICmpInst::ICMP_EQ;
       RHS = getConstant(RA - 1);
        Changed = true;
       break;
      }
      if (RA.isMinSignedValue()) goto trivially_false;
      break;
    }
  }

  // Check for obvious equality.
  if (HasSameValue(LHS, RHS)) {
    if (ICmpInst::isTrueWhenEqual(Pred))
      goto trivially_true;
    if (ICmpInst::isFalseWhenEqual(Pred))
      goto trivially_false;
  }

  // If possible, canonicalize GE/LE comparisons to GT/LT comparisons, by
  // adding or subtracting 1 from one of the operands.
  switch (Pred) {
  case ICmpInst::ICMP_SLE:
    if (!getSignedRange(RHS).getSignedMax().isMaxSignedValue()) {
      RHS = getAddExpr(getConstant(RHS->getType(), 1, true), RHS,
                       SCEV::FlagNSW);
      Pred = ICmpInst::ICMP_SLT;
      Changed = true;
    } else if (!getSignedRange(LHS).getSignedMin().isMinSignedValue()) {
      LHS = getAddExpr(getConstant(RHS->getType(), (uint64_t)-1, true), LHS,
                       SCEV::FlagNSW);
      Pred = ICmpInst::ICMP_SLT;
      Changed = true;
    }
    break;
  case ICmpInst::ICMP_SGE:
    if (!getSignedRange(RHS).getSignedMin().isMinSignedValue()) {
      RHS = getAddExpr(getConstant(RHS->getType(), (uint64_t)-1, true), RHS,
                       SCEV::FlagNSW);
      Pred = ICmpInst::ICMP_SGT;
      Changed = true;
    } else if (!getSignedRange(LHS).getSignedMax().isMaxSignedValue()) {
      LHS = getAddExpr(getConstant(RHS->getType(), 1, true), LHS,
                       SCEV::FlagNSW);
      Pred = ICmpInst::ICMP_SGT;
      Changed = true;
    }
    break;
  case ICmpInst::ICMP_ULE:
    if (!getUnsignedRange(RHS).getUnsignedMax().isMaxValue()) {
      RHS = getAddExpr(getConstant(RHS->getType(), 1, true), RHS,
                       SCEV::FlagNUW);
      Pred = ICmpInst::ICMP_ULT;
      Changed = true;
    } else if (!getUnsignedRange(LHS).getUnsignedMin().isMinValue()) {
      LHS = getAddExpr(getConstant(RHS->getType(), (uint64_t)-1, true), LHS,
                       SCEV::FlagNUW);
      Pred = ICmpInst::ICMP_ULT;
      Changed = true;
    }
    break;
  case ICmpInst::ICMP_UGE:
    if (!getUnsignedRange(RHS).getUnsignedMin().isMinValue()) {
      RHS = getAddExpr(getConstant(RHS->getType(), (uint64_t)-1, true), RHS,
                       SCEV::FlagNUW);
      Pred = ICmpInst::ICMP_UGT;
      Changed = true;
    } else if (!getUnsignedRange(LHS).getUnsignedMax().isMaxValue()) {
      LHS = getAddExpr(getConstant(RHS->getType(), 1, true), LHS,
                       SCEV::FlagNUW);
      Pred = ICmpInst::ICMP_UGT;
      Changed = true;
    }
    break;
  default:
    break;
  }

  // TODO: More simplifications are possible here.

  // Recursively simplify until we either hit a recursion limit or nothing
  // changes.
  if (Changed)
    return SimplifyICmpOperands(Pred, LHS, RHS, Depth+1);

  return Changed;

trivially_true:
  // Return 0 == 0.
  LHS = RHS = getConstant(ConstantInt::getFalse(getContext()));
  Pred = ICmpInst::ICMP_EQ;
  return true;

trivially_false:
  // Return 0 != 0.
  LHS = RHS = getConstant(ConstantInt::getFalse(getContext()));
  Pred = ICmpInst::ICMP_NE;
  return true;
}

bool ScalarEvolution::isKnownNegative(const SCEV *S) {
  return getSignedRange(S).getSignedMax().isNegative();
}

bool ScalarEvolution::isKnownPositive(const SCEV *S) {
  return getSignedRange(S).getSignedMin().isStrictlyPositive();
}

bool ScalarEvolution::isKnownNonNegative(const SCEV *S) {
  return !getSignedRange(S).getSignedMin().isNegative();
}

bool ScalarEvolution::isKnownNonPositive(const SCEV *S) {
  return !getSignedRange(S).getSignedMax().isStrictlyPositive();
}

bool ScalarEvolution::isKnownNonZero(const SCEV *S) {
  return isKnownNegative(S) || isKnownPositive(S);
}

bool ScalarEvolution::isKnownPredicate(ICmpInst::Predicate Pred,
                                       const SCEV *LHS, const SCEV *RHS) {
  // Canonicalize the inputs first.
  (void)SimplifyICmpOperands(Pred, LHS, RHS);

  // If LHS or RHS is an addrec, check to see if the condition is true in
  // every iteration of the loop.
  // If LHS and RHS are both addrec, both conditions must be true in
  // every iteration of the loop.
  const SCEVAddRecExpr *LAR = dyn_cast<SCEVAddRecExpr>(LHS);
  const SCEVAddRecExpr *RAR = dyn_cast<SCEVAddRecExpr>(RHS);
  bool LeftGuarded = false;
  bool RightGuarded = false;
  if (LAR) {
    const Loop *L = LAR->getLoop();
    if (isLoopEntryGuardedByCond(L, Pred, LAR->getStart(), RHS) &&
        isLoopBackedgeGuardedByCond(L, Pred, LAR->getPostIncExpr(*this), RHS)) {
      if (!RAR) return true;
      LeftGuarded = true;
    }
  }
  if (RAR) {
    const Loop *L = RAR->getLoop();
    if (isLoopEntryGuardedByCond(L, Pred, LHS, RAR->getStart()) &&
        isLoopBackedgeGuardedByCond(L, Pred, LHS, RAR->getPostIncExpr(*this))) {
      if (!LAR) return true;
      RightGuarded = true;
    }
  }
  if (LeftGuarded && RightGuarded)
    return true;

  // Otherwise see what can be done with known constant ranges.
  return isKnownPredicateWithRanges(Pred, LHS, RHS);
}

bool
ScalarEvolution::isKnownPredicateWithRanges(ICmpInst::Predicate Pred,
                                            const SCEV *LHS, const SCEV *RHS) {
  if (HasSameValue(LHS, RHS))
    return ICmpInst::isTrueWhenEqual(Pred);

  // This code is split out from isKnownPredicate because it is called from
  // within isLoopEntryGuardedByCond.
  switch (Pred) {
  default:
    llvm_unreachable("Unexpected ICmpInst::Predicate value!");
  case ICmpInst::ICMP_SGT:
    std::swap(LHS, RHS);
  case ICmpInst::ICMP_SLT: {
    ConstantRange LHSRange = getSignedRange(LHS);
    ConstantRange RHSRange = getSignedRange(RHS);
    if (LHSRange.getSignedMax().slt(RHSRange.getSignedMin()))
      return true;
    if (LHSRange.getSignedMin().sge(RHSRange.getSignedMax()))
      return false;
    break;
  }
  case ICmpInst::ICMP_SGE:
    std::swap(LHS, RHS);
  case ICmpInst::ICMP_SLE: {
    ConstantRange LHSRange = getSignedRange(LHS);
    ConstantRange RHSRange = getSignedRange(RHS);
    if (LHSRange.getSignedMax().sle(RHSRange.getSignedMin()))
      return true;
    if (LHSRange.getSignedMin().sgt(RHSRange.getSignedMax()))
      return false;
    break;
  }
  case ICmpInst::ICMP_UGT:
    std::swap(LHS, RHS);
  case ICmpInst::ICMP_ULT: {
    ConstantRange LHSRange = getUnsignedRange(LHS);
    ConstantRange RHSRange = getUnsignedRange(RHS);
    if (LHSRange.getUnsignedMax().ult(RHSRange.getUnsignedMin()))
      return true;
    if (LHSRange.getUnsignedMin().uge(RHSRange.getUnsignedMax()))
      return false;
    break;
  }
  case ICmpInst::ICMP_UGE:
    std::swap(LHS, RHS);
  case ICmpInst::ICMP_ULE: {
    ConstantRange LHSRange = getUnsignedRange(LHS);
    ConstantRange RHSRange = getUnsignedRange(RHS);
    if (LHSRange.getUnsignedMax().ule(RHSRange.getUnsignedMin()))
      return true;
    if (LHSRange.getUnsignedMin().ugt(RHSRange.getUnsignedMax()))
      return false;
    break;
  }
  case ICmpInst::ICMP_NE: {
    if (getUnsignedRange(LHS).intersectWith(getUnsignedRange(RHS)).isEmptySet())
      return true;
    if (getSignedRange(LHS).intersectWith(getSignedRange(RHS)).isEmptySet())
      return true;

    const SCEV *Diff = getMinusSCEV(LHS, RHS);
    if (isKnownNonZero(Diff))
      return true;
    break;
  }
  case ICmpInst::ICMP_EQ:
    // The check at the top of the function catches the case where
    // the values are known to be equal.
    break;
  }
  return false;
}

/// isLoopBackedgeGuardedByCond - Test whether the backedge of the loop is
/// protected by a conditional between LHS and RHS.  This is used to
/// to eliminate casts.
bool
ScalarEvolution::isLoopBackedgeGuardedByCond(const Loop *L,
                                             ICmpInst::Predicate Pred,
                                             const SCEV *LHS, const SCEV *RHS) {
  // Interpret a null as meaning no loop, where there is obviously no guard
  // (interprocedural conditions notwithstanding).
  if (!L) return true;

  if (isKnownPredicateWithRanges(Pred, LHS, RHS)) return true;

  BasicBlock *Latch = L->getLoopLatch();
  if (!Latch)
    return false;

  BranchInst *LoopContinuePredicate =
    dyn_cast<BranchInst>(Latch->getTerminator());
  if (LoopContinuePredicate && LoopContinuePredicate->isConditional() &&
      isImpliedCond(Pred, LHS, RHS,
                    LoopContinuePredicate->getCondition(),
                    LoopContinuePredicate->getSuccessor(0) != L->getHeader()))
    return true;

  // Check conditions due to any @llvm.assume intrinsics.
  for (auto &AssumeVH : AC->assumptions()) {
    if (!AssumeVH)
      continue;
    auto *CI = cast<CallInst>(AssumeVH);
    if (!DT->dominates(CI, Latch->getTerminator()))
      continue;

    if (isImpliedCond(Pred, LHS, RHS, CI->getArgOperand(0), false))
      return true;
  }

  struct ClearWalkingBEDominatingCondsOnExit {
    ScalarEvolution &SE;

    explicit ClearWalkingBEDominatingCondsOnExit(ScalarEvolution &SE)
        : SE(SE){};

    ~ClearWalkingBEDominatingCondsOnExit() {
      SE.WalkingBEDominatingConds = false;
    }
  };

  // We don't want more than one activation of the following loop on the stack
  // -- that can lead to O(n!) time complexity.
  if (WalkingBEDominatingConds)
    return false;

  WalkingBEDominatingConds = true;
  ClearWalkingBEDominatingCondsOnExit ClearOnExit(*this);

  // If the loop is not reachable from the entry block, we risk running into an
  // infinite loop as we walk up into the dom tree.  These loops do not matter
  // anyway, so we just return a conservative answer when we see them.
  if (!DT->isReachableFromEntry(L->getHeader()))
    return false;

  for (DomTreeNode *DTN = (*DT)[Latch], *HeaderDTN = (*DT)[L->getHeader()];
       DTN != HeaderDTN;
       DTN = DTN->getIDom()) {

    assert(DTN && "should reach the loop header before reaching the root!");

    BasicBlock *BB = DTN->getBlock();
    BasicBlock *PBB = BB->getSinglePredecessor();
    if (!PBB)
      continue;

    BranchInst *ContinuePredicate = dyn_cast<BranchInst>(PBB->getTerminator());
    if (!ContinuePredicate || !ContinuePredicate->isConditional())
      continue;

    Value *Condition = ContinuePredicate->getCondition();

    // If we have an edge `E` within the loop body that dominates the only
    // latch, the condition guarding `E` also guards the backedge.  This
    // reasoning works only for loops with a single latch.

    BasicBlockEdge DominatingEdge(PBB, BB);
    if (DominatingEdge.isSingleEdge()) {
      // We're constructively (and conservatively) enumerating edges within the
      // loop body that dominate the latch.  The dominator tree better agree
      // with us on this:
      assert(DT->dominates(DominatingEdge, Latch) && "should be!");

      if (isImpliedCond(Pred, LHS, RHS, Condition,
                        BB != ContinuePredicate->getSuccessor(0)))
        return true;
    }
  }

  return false;
}

/// isLoopEntryGuardedByCond - Test whether entry to the loop is protected
/// by a conditional between LHS and RHS.  This is used to help avoid max
/// expressions in loop trip counts, and to eliminate casts.
bool
ScalarEvolution::isLoopEntryGuardedByCond(const Loop *L,
                                          ICmpInst::Predicate Pred,
                                          const SCEV *LHS, const SCEV *RHS) {
  // Interpret a null as meaning no loop, where there is obviously no guard
  // (interprocedural conditions notwithstanding).
  if (!L) return false;

  if (isKnownPredicateWithRanges(Pred, LHS, RHS)) return true;

  // Starting at the loop predecessor, climb up the predecessor chain, as long
  // as there are predecessors that can be found that have unique successors
  // leading to the original header.
  for (std::pair<BasicBlock *, BasicBlock *>
         Pair(L->getLoopPredecessor(), L->getHeader());
       Pair.first;
       Pair = getPredecessorWithUniqueSuccessorForBB(Pair.first)) {

    BranchInst *LoopEntryPredicate =
      dyn_cast<BranchInst>(Pair.first->getTerminator());
    if (!LoopEntryPredicate ||
        LoopEntryPredicate->isUnconditional())
      continue;

    if (isImpliedCond(Pred, LHS, RHS,
                      LoopEntryPredicate->getCondition(),
                      LoopEntryPredicate->getSuccessor(0) != Pair.second))
      return true;
  }

  // Check conditions due to any @llvm.assume intrinsics.
  for (auto &AssumeVH : AC->assumptions()) {
    if (!AssumeVH)
      continue;
    auto *CI = cast<CallInst>(AssumeVH);
    if (!DT->dominates(CI, L->getHeader()))
      continue;

    if (isImpliedCond(Pred, LHS, RHS, CI->getArgOperand(0), false))
      return true;
  }

  return false;
}

/// RAII wrapper to prevent recursive application of isImpliedCond.
/// ScalarEvolution's PendingLoopPredicates set must be empty unless we are
/// currently evaluating isImpliedCond.
struct MarkPendingLoopPredicate {
  Value *Cond;
  DenseSet<Value*> &LoopPreds;
  bool Pending;

  MarkPendingLoopPredicate(Value *C, DenseSet<Value*> &LP)
    : Cond(C), LoopPreds(LP) {
    Pending = !LoopPreds.insert(Cond).second;
  }
  ~MarkPendingLoopPredicate() {
    if (!Pending)
      LoopPreds.erase(Cond);
  }
};

/// isImpliedCond - Test whether the condition described by Pred, LHS,
/// and RHS is true whenever the given Cond value evaluates to true.
bool ScalarEvolution::isImpliedCond(ICmpInst::Predicate Pred,
                                    const SCEV *LHS, const SCEV *RHS,
                                    Value *FoundCondValue,
                                    bool Inverse) {
  MarkPendingLoopPredicate Mark(FoundCondValue, PendingLoopPredicates);
  if (Mark.Pending)
    return false;

  // Recursively handle And and Or conditions.
  if (BinaryOperator *BO = dyn_cast<BinaryOperator>(FoundCondValue)) {
    if (BO->getOpcode() == Instruction::And) {
      if (!Inverse)
        return isImpliedCond(Pred, LHS, RHS, BO->getOperand(0), Inverse) ||
               isImpliedCond(Pred, LHS, RHS, BO->getOperand(1), Inverse);
    } else if (BO->getOpcode() == Instruction::Or) {
      if (Inverse)
        return isImpliedCond(Pred, LHS, RHS, BO->getOperand(0), Inverse) ||
               isImpliedCond(Pred, LHS, RHS, BO->getOperand(1), Inverse);
    }
  }

  ICmpInst *ICI = dyn_cast<ICmpInst>(FoundCondValue);
  if (!ICI) return false;

  // Now that we found a conditional branch that dominates the loop or controls
  // the loop latch. Check to see if it is the comparison we are looking for.
  ICmpInst::Predicate FoundPred;
  if (Inverse)
    FoundPred = ICI->getInversePredicate();
  else
    FoundPred = ICI->getPredicate();

  const SCEV *FoundLHS = getSCEV(ICI->getOperand(0));
  const SCEV *FoundRHS = getSCEV(ICI->getOperand(1));

  // Balance the types.
  if (getTypeSizeInBits(LHS->getType()) <
      getTypeSizeInBits(FoundLHS->getType())) {
    if (CmpInst::isSigned(Pred)) {
      LHS = getSignExtendExpr(LHS, FoundLHS->getType());
      RHS = getSignExtendExpr(RHS, FoundLHS->getType());
    } else {
      LHS = getZeroExtendExpr(LHS, FoundLHS->getType());
      RHS = getZeroExtendExpr(RHS, FoundLHS->getType());
    }
  } else if (getTypeSizeInBits(LHS->getType()) >
      getTypeSizeInBits(FoundLHS->getType())) {
    if (CmpInst::isSigned(FoundPred)) {
      FoundLHS = getSignExtendExpr(FoundLHS, LHS->getType());
      FoundRHS = getSignExtendExpr(FoundRHS, LHS->getType());
    } else {
      FoundLHS = getZeroExtendExpr(FoundLHS, LHS->getType());
      FoundRHS = getZeroExtendExpr(FoundRHS, LHS->getType());
    }
  }

  // Canonicalize the query to match the way instcombine will have
  // canonicalized the comparison.
  if (SimplifyICmpOperands(Pred, LHS, RHS))
    if (LHS == RHS)
      return CmpInst::isTrueWhenEqual(Pred);
  if (SimplifyICmpOperands(FoundPred, FoundLHS, FoundRHS))
    if (FoundLHS == FoundRHS)
      return CmpInst::isFalseWhenEqual(FoundPred);

  // Check to see if we can make the LHS or RHS match.
  if (LHS == FoundRHS || RHS == FoundLHS) {
    if (isa<SCEVConstant>(RHS)) {
      std::swap(FoundLHS, FoundRHS);
      FoundPred = ICmpInst::getSwappedPredicate(FoundPred);
    } else {
      std::swap(LHS, RHS);
      Pred = ICmpInst::getSwappedPredicate(Pred);
    }
  }

  // Check whether the found predicate is the same as the desired predicate.
  if (FoundPred == Pred)
    return isImpliedCondOperands(Pred, LHS, RHS, FoundLHS, FoundRHS);

  // Check whether swapping the found predicate makes it the same as the
  // desired predicate.
  if (ICmpInst::getSwappedPredicate(FoundPred) == Pred) {
    if (isa<SCEVConstant>(RHS))
      return isImpliedCondOperands(Pred, LHS, RHS, FoundRHS, FoundLHS);
    else
      return isImpliedCondOperands(ICmpInst::getSwappedPredicate(Pred),
                                   RHS, LHS, FoundLHS, FoundRHS);
  }

  // Check if we can make progress by sharpening ranges.
  if (FoundPred == ICmpInst::ICMP_NE &&
      (isa<SCEVConstant>(FoundLHS) || isa<SCEVConstant>(FoundRHS))) {

    const SCEVConstant *C = nullptr;
    const SCEV *V = nullptr;

    if (isa<SCEVConstant>(FoundLHS)) {
      C = cast<SCEVConstant>(FoundLHS);
      V = FoundRHS;
    } else {
      C = cast<SCEVConstant>(FoundRHS);
      V = FoundLHS;
    }

    // The guarding predicate tells us that C != V. If the known range
    // of V is [C, t), we can sharpen the range to [C + 1, t).  The
    // range we consider has to correspond to same signedness as the
    // predicate we're interested in folding.

    APInt Min = ICmpInst::isSigned(Pred) ?
        getSignedRange(V).getSignedMin() : getUnsignedRange(V).getUnsignedMin();

    if (Min == C->getValue()->getValue()) {
      // Given (V >= Min && V != Min) we conclude V >= (Min + 1).
      // This is true even if (Min + 1) wraps around -- in case of
      // wraparound, (Min + 1) < Min, so (V >= Min => V >= (Min + 1)).

      APInt SharperMin = Min + 1;

      switch (Pred) {
        case ICmpInst::ICMP_SGE:
        case ICmpInst::ICMP_UGE:
          // We know V `Pred` SharperMin.  If this implies LHS `Pred`
          // RHS, we're done.
          if (isImpliedCondOperands(Pred, LHS, RHS, V,
                                    getConstant(SharperMin)))
            return true;

        case ICmpInst::ICMP_SGT:
        case ICmpInst::ICMP_UGT:
          // We know from the range information that (V `Pred` Min ||
          // V == Min).  We know from the guarding condition that !(V
          // == Min).  This gives us
          //
          //       V `Pred` Min || V == Min && !(V == Min)
          //   =>  V `Pred` Min
          //
          // If V `Pred` Min implies LHS `Pred` RHS, we're done.

          if (isImpliedCondOperands(Pred, LHS, RHS, V, getConstant(Min)))
            return true;

        default:
          // No change
          break;
      }
    }
  }

  // Check whether the actual condition is beyond sufficient.
  if (FoundPred == ICmpInst::ICMP_EQ)
    if (ICmpInst::isTrueWhenEqual(Pred))
      if (isImpliedCondOperands(Pred, LHS, RHS, FoundLHS, FoundRHS))
        return true;
  if (Pred == ICmpInst::ICMP_NE)
    if (!ICmpInst::isTrueWhenEqual(FoundPred))
      if (isImpliedCondOperands(FoundPred, LHS, RHS, FoundLHS, FoundRHS))
        return true;

  // Otherwise assume the worst.
  return false;
}

/// isImpliedCondOperands - Test whether the condition described by Pred,
/// LHS, and RHS is true whenever the condition described by Pred, FoundLHS,
/// and FoundRHS is true.
bool ScalarEvolution::isImpliedCondOperands(ICmpInst::Predicate Pred,
                                            const SCEV *LHS, const SCEV *RHS,
                                            const SCEV *FoundLHS,
                                            const SCEV *FoundRHS) {
  if (isImpliedCondOperandsViaRanges(Pred, LHS, RHS, FoundLHS, FoundRHS))
    return true;

  return isImpliedCondOperandsHelper(Pred, LHS, RHS,
                                     FoundLHS, FoundRHS) ||
         // ~x < ~y --> x > y
         isImpliedCondOperandsHelper(Pred, LHS, RHS,
                                     getNotSCEV(FoundRHS),
                                     getNotSCEV(FoundLHS));
}


/// If Expr computes ~A, return A else return nullptr
static const SCEV *MatchNotExpr(const SCEV *Expr) {
  const SCEVAddExpr *Add = dyn_cast<SCEVAddExpr>(Expr);
  if (!Add || Add->getNumOperands() != 2) return nullptr;

  const SCEVConstant *AddLHS = dyn_cast<SCEVConstant>(Add->getOperand(0));
  if (!(AddLHS && AddLHS->getValue()->getValue().isAllOnesValue()))
    return nullptr;

  const SCEVMulExpr *AddRHS = dyn_cast<SCEVMulExpr>(Add->getOperand(1));
  if (!AddRHS || AddRHS->getNumOperands() != 2) return nullptr;

  const SCEVConstant *MulLHS = dyn_cast<SCEVConstant>(AddRHS->getOperand(0));
  if (!(MulLHS && MulLHS->getValue()->getValue().isAllOnesValue()))
    return nullptr;

  return AddRHS->getOperand(1);
}


/// Is MaybeMaxExpr an SMax or UMax of Candidate and some other values?
template<typename MaxExprType>
static bool IsMaxConsistingOf(const SCEV *MaybeMaxExpr,
                              const SCEV *Candidate) {
  const MaxExprType *MaxExpr = dyn_cast<MaxExprType>(MaybeMaxExpr);
  if (!MaxExpr) return false;

  auto It = std::find(MaxExpr->op_begin(), MaxExpr->op_end(), Candidate);
  return It != MaxExpr->op_end();
}


/// Is MaybeMinExpr an SMin or UMin of Candidate and some other values?
template<typename MaxExprType>
static bool IsMinConsistingOf(ScalarEvolution &SE,
                              const SCEV *MaybeMinExpr,
                              const SCEV *Candidate) {
  const SCEV *MaybeMaxExpr = MatchNotExpr(MaybeMinExpr);
  if (!MaybeMaxExpr)
    return false;

  return IsMaxConsistingOf<MaxExprType>(MaybeMaxExpr, SE.getNotSCEV(Candidate));
}


/// Is LHS `Pred` RHS true on the virtue of LHS or RHS being a Min or Max
/// expression?
static bool IsKnownPredicateViaMinOrMax(ScalarEvolution &SE,
                                        ICmpInst::Predicate Pred,
                                        const SCEV *LHS, const SCEV *RHS) {
  switch (Pred) {
  default:
    return false;

  case ICmpInst::ICMP_SGE:
    std::swap(LHS, RHS);
    // fall through
  case ICmpInst::ICMP_SLE:
    return
      // min(A, ...) <= A
      IsMinConsistingOf<SCEVSMaxExpr>(SE, LHS, RHS) ||
      // A <= max(A, ...)
      IsMaxConsistingOf<SCEVSMaxExpr>(RHS, LHS);

  case ICmpInst::ICMP_UGE:
    std::swap(LHS, RHS);
    // fall through
  case ICmpInst::ICMP_ULE:
    return
      // min(A, ...) <= A
      IsMinConsistingOf<SCEVUMaxExpr>(SE, LHS, RHS) ||
      // A <= max(A, ...)
      IsMaxConsistingOf<SCEVUMaxExpr>(RHS, LHS);
  }

  llvm_unreachable("covered switch fell through?!");
}

/// isImpliedCondOperandsHelper - Test whether the condition described by
/// Pred, LHS, and RHS is true whenever the condition described by Pred,
/// FoundLHS, and FoundRHS is true.
bool
ScalarEvolution::isImpliedCondOperandsHelper(ICmpInst::Predicate Pred,
                                             const SCEV *LHS, const SCEV *RHS,
                                             const SCEV *FoundLHS,
                                             const SCEV *FoundRHS) {
  auto IsKnownPredicateFull =
      [this](ICmpInst::Predicate Pred, const SCEV *LHS, const SCEV *RHS) {
    return isKnownPredicateWithRanges(Pred, LHS, RHS) ||
        IsKnownPredicateViaMinOrMax(*this, Pred, LHS, RHS);
  };

  switch (Pred) {
  default: llvm_unreachable("Unexpected ICmpInst::Predicate value!");
  case ICmpInst::ICMP_EQ:
  case ICmpInst::ICMP_NE:
    if (HasSameValue(LHS, FoundLHS) && HasSameValue(RHS, FoundRHS))
      return true;
    break;
  case ICmpInst::ICMP_SLT:
  case ICmpInst::ICMP_SLE:
    if (IsKnownPredicateFull(ICmpInst::ICMP_SLE, LHS, FoundLHS) &&
        IsKnownPredicateFull(ICmpInst::ICMP_SGE, RHS, FoundRHS))
      return true;
    break;
  case ICmpInst::ICMP_SGT:
  case ICmpInst::ICMP_SGE:
    if (IsKnownPredicateFull(ICmpInst::ICMP_SGE, LHS, FoundLHS) &&
        IsKnownPredicateFull(ICmpInst::ICMP_SLE, RHS, FoundRHS))
      return true;
    break;
  case ICmpInst::ICMP_ULT:
  case ICmpInst::ICMP_ULE:
    if (IsKnownPredicateFull(ICmpInst::ICMP_ULE, LHS, FoundLHS) &&
        IsKnownPredicateFull(ICmpInst::ICMP_UGE, RHS, FoundRHS))
      return true;
    break;
  case ICmpInst::ICMP_UGT:
  case ICmpInst::ICMP_UGE:
    if (IsKnownPredicateFull(ICmpInst::ICMP_UGE, LHS, FoundLHS) &&
        IsKnownPredicateFull(ICmpInst::ICMP_ULE, RHS, FoundRHS))
      return true;
    break;
  }

  return false;
}

/// isImpliedCondOperandsViaRanges - helper function for isImpliedCondOperands.
/// Tries to get cases like "X `sgt` 0 => X - 1 `sgt` -1".
bool ScalarEvolution::isImpliedCondOperandsViaRanges(ICmpInst::Predicate Pred,
                                                     const SCEV *LHS,
                                                     const SCEV *RHS,
                                                     const SCEV *FoundLHS,
                                                     const SCEV *FoundRHS) {
  if (!isa<SCEVConstant>(RHS) || !isa<SCEVConstant>(FoundRHS))
    // The restriction on `FoundRHS` be lifted easily -- it exists only to
    // reduce the compile time impact of this optimization.
    return false;

  const SCEVAddExpr *AddLHS = dyn_cast<SCEVAddExpr>(LHS);
  if (!AddLHS || AddLHS->getOperand(1) != FoundLHS ||
      !isa<SCEVConstant>(AddLHS->getOperand(0)))
    return false;

  APInt ConstFoundRHS = cast<SCEVConstant>(FoundRHS)->getValue()->getValue();

  // `FoundLHSRange` is the range we know `FoundLHS` to be in by virtue of the
  // antecedent "`FoundLHS` `Pred` `FoundRHS`".
  ConstantRange FoundLHSRange =
      ConstantRange::makeAllowedICmpRegion(Pred, ConstFoundRHS);

  // Since `LHS` is `FoundLHS` + `AddLHS->getOperand(0)`, we can compute a range
  // for `LHS`:
  APInt Addend =
      cast<SCEVConstant>(AddLHS->getOperand(0))->getValue()->getValue();
  ConstantRange LHSRange = FoundLHSRange.add(ConstantRange(Addend));

  // We can also compute the range of values for `LHS` that satisfy the
  // consequent, "`LHS` `Pred` `RHS`":
  APInt ConstRHS = cast<SCEVConstant>(RHS)->getValue()->getValue();
  ConstantRange SatisfyingLHSRange =
      ConstantRange::makeSatisfyingICmpRegion(Pred, ConstRHS);

  // The antecedent implies the consequent if every value of `LHS` that
  // satisfies the antecedent also satisfies the consequent.
  return SatisfyingLHSRange.contains(LHSRange);
}

// Verify if an linear IV with positive stride can overflow when in a
// less-than comparison, knowing the invariant term of the comparison, the
// stride and the knowledge of NSW/NUW flags on the recurrence.
bool ScalarEvolution::doesIVOverflowOnLT(const SCEV *RHS, const SCEV *Stride,
                                         bool IsSigned, bool NoWrap) {
  if (NoWrap) return false;

  unsigned BitWidth = getTypeSizeInBits(RHS->getType());
  const SCEV *One = getConstant(Stride->getType(), 1);

  if (IsSigned) {
    APInt MaxRHS = getSignedRange(RHS).getSignedMax();
    APInt MaxValue = APInt::getSignedMaxValue(BitWidth);
    APInt MaxStrideMinusOne = getSignedRange(getMinusSCEV(Stride, One))
                                .getSignedMax();

    // SMaxRHS + SMaxStrideMinusOne > SMaxValue => overflow!
    return (MaxValue - MaxStrideMinusOne).slt(MaxRHS);
  }

  APInt MaxRHS = getUnsignedRange(RHS).getUnsignedMax();
  APInt MaxValue = APInt::getMaxValue(BitWidth);
  APInt MaxStrideMinusOne = getUnsignedRange(getMinusSCEV(Stride, One))
                              .getUnsignedMax();

  // UMaxRHS + UMaxStrideMinusOne > UMaxValue => overflow!
  return (MaxValue - MaxStrideMinusOne).ult(MaxRHS);
}

// Verify if an linear IV with negative stride can overflow when in a
// greater-than comparison, knowing the invariant term of the comparison,
// the stride and the knowledge of NSW/NUW flags on the recurrence.
bool ScalarEvolution::doesIVOverflowOnGT(const SCEV *RHS, const SCEV *Stride,
                                         bool IsSigned, bool NoWrap) {
  if (NoWrap) return false;

  unsigned BitWidth = getTypeSizeInBits(RHS->getType());
  const SCEV *One = getConstant(Stride->getType(), 1);

  if (IsSigned) {
    APInt MinRHS = getSignedRange(RHS).getSignedMin();
    APInt MinValue = APInt::getSignedMinValue(BitWidth);
    APInt MaxStrideMinusOne = getSignedRange(getMinusSCEV(Stride, One))
                               .getSignedMax();

    // SMinRHS - SMaxStrideMinusOne < SMinValue => overflow!
    return (MinValue + MaxStrideMinusOne).sgt(MinRHS);
  }

  APInt MinRHS = getUnsignedRange(RHS).getUnsignedMin();
  APInt MinValue = APInt::getMinValue(BitWidth);
  APInt MaxStrideMinusOne = getUnsignedRange(getMinusSCEV(Stride, One))
                            .getUnsignedMax();

  // UMinRHS - UMaxStrideMinusOne < UMinValue => overflow!
  return (MinValue + MaxStrideMinusOne).ugt(MinRHS);
}

// Compute the backedge taken count knowing the interval difference, the
// stride and presence of the equality in the comparison.
const SCEV *ScalarEvolution::computeBECount(const SCEV *Delta, const SCEV *Step,
                                            bool Equality) {
  const SCEV *One = getConstant(Step->getType(), 1);
  Delta = Equality ? getAddExpr(Delta, Step)
                   : getAddExpr(Delta, getMinusSCEV(Step, One));
  return getUDivExpr(Delta, Step);
}

/// HowManyLessThans - Return the number of times a backedge containing the
/// specified less-than comparison will execute.  If not computable, return
/// CouldNotCompute.
///
/// @param ControlsExit is true when the LHS < RHS condition directly controls
/// the branch (loops exits only if condition is true). In this case, we can use
/// NoWrapFlags to skip overflow checks.
ScalarEvolution::ExitLimit
ScalarEvolution::HowManyLessThans(const SCEV *LHS, const SCEV *RHS,
                                  const Loop *L, bool IsSigned,
                                  bool ControlsExit) {
  // We handle only IV < Invariant
  if (!isLoopInvariant(RHS, L))
    return getCouldNotCompute();

  const SCEVAddRecExpr *IV = dyn_cast<SCEVAddRecExpr>(LHS);

  // Avoid weird loops
  if (!IV || IV->getLoop() != L || !IV->isAffine())
    return getCouldNotCompute();

  bool NoWrap = ControlsExit &&
                IV->getNoWrapFlags(IsSigned ? SCEV::FlagNSW : SCEV::FlagNUW);

  const SCEV *Stride = IV->getStepRecurrence(*this);

  // Avoid negative or zero stride values
  if (!isKnownPositive(Stride))
    return getCouldNotCompute();

  // Avoid proven overflow cases: this will ensure that the backedge taken count
  // will not generate any unsigned overflow. Relaxed no-overflow conditions
  // exploit NoWrapFlags, allowing to optimize in presence of undefined
  // behaviors like the case of C language.
  if (!Stride->isOne() && doesIVOverflowOnLT(RHS, Stride, IsSigned, NoWrap))
    return getCouldNotCompute();

  ICmpInst::Predicate Cond = IsSigned ? ICmpInst::ICMP_SLT
                                      : ICmpInst::ICMP_ULT;
  const SCEV *Start = IV->getStart();
  const SCEV *End = RHS;
  if (!isLoopEntryGuardedByCond(L, Cond, getMinusSCEV(Start, Stride), RHS)) {
    const SCEV *Diff = getMinusSCEV(RHS, Start);
    // If we have NoWrap set, then we can assume that the increment won't
    // overflow, in which case if RHS - Start is a constant, we don't need to
    // do a max operation since we can just figure it out statically
    if (NoWrap && isa<SCEVConstant>(Diff)) {
      APInt D = dyn_cast<const SCEVConstant>(Diff)->getValue()->getValue();
      if (D.isNegative())
        End = Start;
    } else
      End = IsSigned ? getSMaxExpr(RHS, Start)
                     : getUMaxExpr(RHS, Start);
  }

  const SCEV *BECount = computeBECount(getMinusSCEV(End, Start), Stride, false);

  APInt MinStart = IsSigned ? getSignedRange(Start).getSignedMin()
                            : getUnsignedRange(Start).getUnsignedMin();

  APInt MinStride = IsSigned ? getSignedRange(Stride).getSignedMin()
                             : getUnsignedRange(Stride).getUnsignedMin();

  unsigned BitWidth = getTypeSizeInBits(LHS->getType());
  APInt Limit = IsSigned ? APInt::getSignedMaxValue(BitWidth) - (MinStride - 1)
                         : APInt::getMaxValue(BitWidth) - (MinStride - 1);

  // Although End can be a MAX expression we estimate MaxEnd considering only
  // the case End = RHS. This is safe because in the other case (End - Start)
  // is zero, leading to a zero maximum backedge taken count.
  APInt MaxEnd =
    IsSigned ? APIntOps::smin(getSignedRange(RHS).getSignedMax(), Limit)
             : APIntOps::umin(getUnsignedRange(RHS).getUnsignedMax(), Limit);

  const SCEV *MaxBECount;
  if (isa<SCEVConstant>(BECount))
    MaxBECount = BECount;
  else
    MaxBECount = computeBECount(getConstant(MaxEnd - MinStart),
                                getConstant(MinStride), false);

  if (isa<SCEVCouldNotCompute>(MaxBECount))
    MaxBECount = BECount;

  return ExitLimit(BECount, MaxBECount);
}

ScalarEvolution::ExitLimit
ScalarEvolution::HowManyGreaterThans(const SCEV *LHS, const SCEV *RHS,
                                     const Loop *L, bool IsSigned,
                                     bool ControlsExit) {
  // We handle only IV > Invariant
  if (!isLoopInvariant(RHS, L))
    return getCouldNotCompute();

  const SCEVAddRecExpr *IV = dyn_cast<SCEVAddRecExpr>(LHS);

  // Avoid weird loops
  if (!IV || IV->getLoop() != L || !IV->isAffine())
    return getCouldNotCompute();

  bool NoWrap = ControlsExit &&
                IV->getNoWrapFlags(IsSigned ? SCEV::FlagNSW : SCEV::FlagNUW);

  const SCEV *Stride = getNegativeSCEV(IV->getStepRecurrence(*this));

  // Avoid negative or zero stride values
  if (!isKnownPositive(Stride))
    return getCouldNotCompute();

  // Avoid proven overflow cases: this will ensure that the backedge taken count
  // will not generate any unsigned overflow. Relaxed no-overflow conditions
  // exploit NoWrapFlags, allowing to optimize in presence of undefined
  // behaviors like the case of C language.
  if (!Stride->isOne() && doesIVOverflowOnGT(RHS, Stride, IsSigned, NoWrap))
    return getCouldNotCompute();

  ICmpInst::Predicate Cond = IsSigned ? ICmpInst::ICMP_SGT
                                      : ICmpInst::ICMP_UGT;

  const SCEV *Start = IV->getStart();
  const SCEV *End = RHS;
  if (!isLoopEntryGuardedByCond(L, Cond, getAddExpr(Start, Stride), RHS)) {
    const SCEV *Diff = getMinusSCEV(RHS, Start);
    // If we have NoWrap set, then we can assume that the increment won't
    // overflow, in which case if RHS - Start is a constant, we don't need to
    // do a max operation since we can just figure it out statically
    if (NoWrap && isa<SCEVConstant>(Diff)) {
      APInt D = dyn_cast<const SCEVConstant>(Diff)->getValue()->getValue();
      if (!D.isNegative())
        End = Start;
    } else
      End = IsSigned ? getSMinExpr(RHS, Start)
                     : getUMinExpr(RHS, Start);
  }

  const SCEV *BECount = computeBECount(getMinusSCEV(Start, End), Stride, false);

  APInt MaxStart = IsSigned ? getSignedRange(Start).getSignedMax()
                            : getUnsignedRange(Start).getUnsignedMax();

  APInt MinStride = IsSigned ? getSignedRange(Stride).getSignedMin()
                             : getUnsignedRange(Stride).getUnsignedMin();

  unsigned BitWidth = getTypeSizeInBits(LHS->getType());
  APInt Limit = IsSigned ? APInt::getSignedMinValue(BitWidth) + (MinStride - 1)
                         : APInt::getMinValue(BitWidth) + (MinStride - 1);

  // Although End can be a MIN expression we estimate MinEnd considering only
  // the case End = RHS. This is safe because in the other case (Start - End)
  // is zero, leading to a zero maximum backedge taken count.
  APInt MinEnd =
    IsSigned ? APIntOps::smax(getSignedRange(RHS).getSignedMin(), Limit)
             : APIntOps::umax(getUnsignedRange(RHS).getUnsignedMin(), Limit);


  const SCEV *MaxBECount = getCouldNotCompute();
  if (isa<SCEVConstant>(BECount))
    MaxBECount = BECount;
  else
    MaxBECount = computeBECount(getConstant(MaxStart - MinEnd),
                                getConstant(MinStride), false);

  if (isa<SCEVCouldNotCompute>(MaxBECount))
    MaxBECount = BECount;

  return ExitLimit(BECount, MaxBECount);
}

/// getNumIterationsInRange - Return the number of iterations of this loop that
/// produce values in the specified constant range.  Another way of looking at
/// this is that it returns the first iteration number where the value is not in
/// the condition, thus computing the exit count. If the iteration count can't
/// be computed, an instance of SCEVCouldNotCompute is returned.
const SCEV *SCEVAddRecExpr::getNumIterationsInRange(ConstantRange Range,
                                                    ScalarEvolution &SE) const {
  if (Range.isFullSet())  // Infinite loop.
    return SE.getCouldNotCompute();

  // If the start is a non-zero constant, shift the range to simplify things.
  if (const SCEVConstant *SC = dyn_cast<SCEVConstant>(getStart()))
    if (!SC->getValue()->isZero()) {
      SmallVector<const SCEV *, 4> Operands(op_begin(), op_end());
      Operands[0] = SE.getConstant(SC->getType(), 0);
      const SCEV *Shifted = SE.getAddRecExpr(Operands, getLoop(),
                                             getNoWrapFlags(FlagNW));
      if (const SCEVAddRecExpr *ShiftedAddRec =
            dyn_cast<SCEVAddRecExpr>(Shifted))
        return ShiftedAddRec->getNumIterationsInRange(
                           Range.subtract(SC->getValue()->getValue()), SE);
      // This is strange and shouldn't happen.
      return SE.getCouldNotCompute();
    }

  // The only time we can solve this is when we have all constant indices.
  // Otherwise, we cannot determine the overflow conditions.
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
    if (!isa<SCEVConstant>(getOperand(i)))
      return SE.getCouldNotCompute();


  // Okay at this point we know that all elements of the chrec are constants and
  // that the start element is zero.

  // First check to see if the range contains zero.  If not, the first
  // iteration exits.
  unsigned BitWidth = SE.getTypeSizeInBits(getType());
  if (!Range.contains(APInt(BitWidth, 0)))
    return SE.getConstant(getType(), 0);

  if (isAffine()) {
    // If this is an affine expression then we have this situation:
    //   Solve {0,+,A} in Range  ===  Ax in Range

    // We know that zero is in the range.  If A is positive then we know that
    // the upper value of the range must be the first possible exit value.
    // If A is negative then the lower of the range is the last possible loop
    // value.  Also note that we already checked for a full range.
    APInt One(BitWidth,1);
    APInt A     = cast<SCEVConstant>(getOperand(1))->getValue()->getValue();
    APInt End = A.sge(One) ? (Range.getUpper() - One) : Range.getLower();

    // The exit value should be (End+A)/A.
    APInt ExitVal = (End + A).udiv(A);
    ConstantInt *ExitValue = ConstantInt::get(SE.getContext(), ExitVal);

    // Evaluate at the exit value.  If we really did fall out of the valid
    // range, then we computed our trip count, otherwise wrap around or other
    // things must have happened.
    ConstantInt *Val = EvaluateConstantChrecAtConstant(this, ExitValue, SE);
    if (Range.contains(Val->getValue()))
      return SE.getCouldNotCompute();  // Something strange happened

    // Ensure that the previous value is in the range.  This is a sanity check.
    assert(Range.contains(
           EvaluateConstantChrecAtConstant(this,
           ConstantInt::get(SE.getContext(), ExitVal - One), SE)->getValue()) &&
           "Linear scev computation is off in a bad way!");
    return SE.getConstant(ExitValue);
  } else if (isQuadratic()) {
    // If this is a quadratic (3-term) AddRec {L,+,M,+,N}, find the roots of the
    // quadratic equation to solve it.  To do this, we must frame our problem in
    // terms of figuring out when zero is crossed, instead of when
    // Range.getUpper() is crossed.
    SmallVector<const SCEV *, 4> NewOps(op_begin(), op_end());
    NewOps[0] = SE.getNegativeSCEV(SE.getConstant(Range.getUpper()));
    const SCEV *NewAddRec = SE.getAddRecExpr(NewOps, getLoop(),
                                             // getNoWrapFlags(FlagNW)
                                             FlagAnyWrap);

    // Next, solve the constructed addrec
    std::pair<const SCEV *,const SCEV *> Roots =
      SolveQuadraticEquation(cast<SCEVAddRecExpr>(NewAddRec), SE);
    const SCEVConstant *R1 = dyn_cast<SCEVConstant>(Roots.first);
    const SCEVConstant *R2 = dyn_cast<SCEVConstant>(Roots.second);
    if (R1) {
      // Pick the smallest positive root value.
      if (ConstantInt *CB =
          dyn_cast<ConstantInt>(ConstantExpr::getICmp(ICmpInst::ICMP_ULT,
                         R1->getValue(), R2->getValue()))) {
        if (!CB->getZExtValue())
          std::swap(R1, R2);   // R1 is the minimum root now.

        // Make sure the root is not off by one.  The returned iteration should
        // not be in the range, but the previous one should be.  When solving
        // for "X*X < 5", for example, we should not return a root of 2.
        ConstantInt *R1Val = EvaluateConstantChrecAtConstant(this,
                                                             R1->getValue(),
                                                             SE);
        if (Range.contains(R1Val->getValue())) {
          // The next iteration must be out of the range...
          ConstantInt *NextVal =
                ConstantInt::get(SE.getContext(), R1->getValue()->getValue()+1);

          R1Val = EvaluateConstantChrecAtConstant(this, NextVal, SE);
          if (!Range.contains(R1Val->getValue()))
            return SE.getConstant(NextVal);
          return SE.getCouldNotCompute();  // Something strange happened
        }

        // If R1 was not in the range, then it is a good return value.  Make
        // sure that R1-1 WAS in the range though, just in case.
        ConstantInt *NextVal =
               ConstantInt::get(SE.getContext(), R1->getValue()->getValue()-1);
        R1Val = EvaluateConstantChrecAtConstant(this, NextVal, SE);
        if (Range.contains(R1Val->getValue()))
          return R1;
        return SE.getCouldNotCompute();  // Something strange happened
      }
    }
  }

  return SE.getCouldNotCompute();
}

namespace {
struct FindUndefs {
  bool Found;
  FindUndefs() : Found(false) {}

  bool follow(const SCEV *S) {
    if (const SCEVUnknown *C = dyn_cast<SCEVUnknown>(S)) {
      if (isa<UndefValue>(C->getValue()))
        Found = true;
    } else if (const SCEVConstant *C = dyn_cast<SCEVConstant>(S)) {
      if (isa<UndefValue>(C->getValue()))
        Found = true;
    }

    // Keep looking if we haven't found it yet.
    return !Found;
  }
  bool isDone() const {
    // Stop recursion if we have found an undef.
    return Found;
  }
};
}

// Return true when S contains at least an undef value.
static inline bool
containsUndefs(const SCEV *S) {
  FindUndefs F;
  SCEVTraversal<FindUndefs> ST(F);
  ST.visitAll(S);

  return F.Found;
}

namespace {
// Collect all steps of SCEV expressions.
struct SCEVCollectStrides {
  ScalarEvolution &SE;
  SmallVectorImpl<const SCEV *> &Strides;

  SCEVCollectStrides(ScalarEvolution &SE, SmallVectorImpl<const SCEV *> &S)
      : SE(SE), Strides(S) {}

  bool follow(const SCEV *S) {
    if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(S))
      Strides.push_back(AR->getStepRecurrence(SE));
    return true;
  }
  bool isDone() const { return false; }
};

// Collect all SCEVUnknown and SCEVMulExpr expressions.
struct SCEVCollectTerms {
  SmallVectorImpl<const SCEV *> &Terms;

  SCEVCollectTerms(SmallVectorImpl<const SCEV *> &T)
      : Terms(T) {}

  bool follow(const SCEV *S) {
    if (isa<SCEVUnknown>(S) || isa<SCEVMulExpr>(S)) {
      if (!containsUndefs(S))
        Terms.push_back(S);

      // Stop recursion: once we collected a term, do not walk its operands.
      return false;
    }

    // Keep looking.
    return true;
  }
  bool isDone() const { return false; }
};
}

/// Find parametric terms in this SCEVAddRecExpr.
void ScalarEvolution::collectParametricTerms(const SCEV *Expr,
    SmallVectorImpl<const SCEV *> &Terms) {
  SmallVector<const SCEV *, 4> Strides;
  SCEVCollectStrides StrideCollector(*this, Strides);
  visitAll(Expr, StrideCollector);

  DEBUG({
      dbgs() << "Strides:\n";
      for (const SCEV *S : Strides)
        dbgs() << *S << "\n";
    });

  for (const SCEV *S : Strides) {
    SCEVCollectTerms TermCollector(Terms);
    visitAll(S, TermCollector);
  }

  DEBUG({
      dbgs() << "Terms:\n";
      for (const SCEV *T : Terms)
        dbgs() << *T << "\n";
    });
}

static bool findArrayDimensionsRec(ScalarEvolution &SE,
                                   SmallVectorImpl<const SCEV *> &Terms,
                                   SmallVectorImpl<const SCEV *> &Sizes) {
  int Last = Terms.size() - 1;
  const SCEV *Step = Terms[Last];

  // End of recursion.
  if (Last == 0) {
    if (const SCEVMulExpr *M = dyn_cast<SCEVMulExpr>(Step)) {
      SmallVector<const SCEV *, 2> Qs;
      for (const SCEV *Op : M->operands())
        if (!isa<SCEVConstant>(Op))
          Qs.push_back(Op);

      Step = SE.getMulExpr(Qs);
    }

    Sizes.push_back(Step);
    return true;
  }

  for (const SCEV *&Term : Terms) {
    // Normalize the terms before the next call to findArrayDimensionsRec.
    const SCEV *Q, *R;
    SCEVDivision::divide(SE, Term, Step, &Q, &R);

    // Bail out when GCD does not evenly divide one of the terms.
    if (!R->isZero())
      return false;

    Term = Q;
  }

  // Remove all SCEVConstants.
  Terms.erase(std::remove_if(Terms.begin(), Terms.end(), [](const SCEV *E) {
                return isa<SCEVConstant>(E);
              }),
              Terms.end());

  if (Terms.size() > 0)
    if (!findArrayDimensionsRec(SE, Terms, Sizes))
      return false;

  Sizes.push_back(Step);
  return true;
}

namespace {
struct FindParameter {
  bool FoundParameter;
  FindParameter() : FoundParameter(false) {}

  bool follow(const SCEV *S) {
    if (isa<SCEVUnknown>(S)) {
      FoundParameter = true;
      // Stop recursion: we found a parameter.
      return false;
    }
    // Keep looking.
    return true;
  }
  bool isDone() const {
    // Stop recursion if we have found a parameter.
    return FoundParameter;
  }
};
}

// Returns true when S contains at least a SCEVUnknown parameter.
static inline bool
containsParameters(const SCEV *S) {
  FindParameter F;
  SCEVTraversal<FindParameter> ST(F);
  ST.visitAll(S);

  return F.FoundParameter;
}

// Returns true when one of the SCEVs of Terms contains a SCEVUnknown parameter.
static inline bool
containsParameters(SmallVectorImpl<const SCEV *> &Terms) {
  for (const SCEV *T : Terms)
    if (containsParameters(T))
      return true;
  return false;
}

// Return the number of product terms in S.
static inline int numberOfTerms(const SCEV *S) {
  if (const SCEVMulExpr *Expr = dyn_cast<SCEVMulExpr>(S))
    return Expr->getNumOperands();
  return 1;
}

static const SCEV *removeConstantFactors(ScalarEvolution &SE, const SCEV *T) {
  if (isa<SCEVConstant>(T))
    return nullptr;

  if (isa<SCEVUnknown>(T))
    return T;

  if (const SCEVMulExpr *M = dyn_cast<SCEVMulExpr>(T)) {
    SmallVector<const SCEV *, 2> Factors;
    for (const SCEV *Op : M->operands())
      if (!isa<SCEVConstant>(Op))
        Factors.push_back(Op);

    return SE.getMulExpr(Factors);
  }

  return T;
}

/// Return the size of an element read or written by Inst.
const SCEV *ScalarEvolution::getElementSize(Instruction *Inst) {
  Type *Ty;
  if (StoreInst *Store = dyn_cast<StoreInst>(Inst))
    Ty = Store->getValueOperand()->getType();
  else if (LoadInst *Load = dyn_cast<LoadInst>(Inst))
    Ty = Load->getType();
  else
    return nullptr;

  Type *ETy = getEffectiveSCEVType(PointerType::getUnqual(Ty));
  return getSizeOfExpr(ETy, Ty);
}

/// Second step of delinearization: compute the array dimensions Sizes from the
/// set of Terms extracted from the memory access function of this SCEVAddRec.
void ScalarEvolution::findArrayDimensions(SmallVectorImpl<const SCEV *> &Terms,
                                          SmallVectorImpl<const SCEV *> &Sizes,
                                          const SCEV *ElementSize) const {

  if (Terms.size() < 1 || !ElementSize)
    return;

  // Early return when Terms do not contain parameters: we do not delinearize
  // non parametric SCEVs.
  if (!containsParameters(Terms))
    return;

  DEBUG({
      dbgs() << "Terms:\n";
      for (const SCEV *T : Terms)
        dbgs() << *T << "\n";
    });

  // Remove duplicates.
  std::sort(Terms.begin(), Terms.end());
  Terms.erase(std::unique(Terms.begin(), Terms.end()), Terms.end());

  // Put larger terms first.
  std::sort(Terms.begin(), Terms.end(), [](const SCEV *LHS, const SCEV *RHS) {
    return numberOfTerms(LHS) > numberOfTerms(RHS);
  });

  ScalarEvolution &SE = *const_cast<ScalarEvolution *>(this);

  // Divide all terms by the element size.
  for (const SCEV *&Term : Terms) {
    const SCEV *Q, *R;
    SCEVDivision::divide(SE, Term, ElementSize, &Q, &R);
    Term = Q;
  }

  SmallVector<const SCEV *, 4> NewTerms;

  // Remove constant factors.
  for (const SCEV *T : Terms)
    if (const SCEV *NewT = removeConstantFactors(SE, T))
      NewTerms.push_back(NewT);

  DEBUG({
      dbgs() << "Terms after sorting:\n";
      for (const SCEV *T : NewTerms)
        dbgs() << *T << "\n";
    });

  if (NewTerms.empty() ||
      !findArrayDimensionsRec(SE, NewTerms, Sizes)) {
    Sizes.clear();
    return;
  }

  // The last element to be pushed into Sizes is the size of an element.
  Sizes.push_back(ElementSize);

  DEBUG({
      dbgs() << "Sizes:\n";
      for (const SCEV *S : Sizes)
        dbgs() << *S << "\n";
    });
}

/// Third step of delinearization: compute the access functions for the
/// Subscripts based on the dimensions in Sizes.
void ScalarEvolution::computeAccessFunctions(
    const SCEV *Expr, SmallVectorImpl<const SCEV *> &Subscripts,
    SmallVectorImpl<const SCEV *> &Sizes) {

  // Early exit in case this SCEV is not an affine multivariate function.
  if (Sizes.empty())
    return;

  if (auto AR = dyn_cast<SCEVAddRecExpr>(Expr))
    if (!AR->isAffine())
      return;

  const SCEV *Res = Expr;
  int Last = Sizes.size() - 1;
  for (int i = Last; i >= 0; i--) {
    const SCEV *Q, *R;
    SCEVDivision::divide(*this, Res, Sizes[i], &Q, &R);

    DEBUG({
        dbgs() << "Res: " << *Res << "\n";
        dbgs() << "Sizes[i]: " << *Sizes[i] << "\n";
        dbgs() << "Res divided by Sizes[i]:\n";
        dbgs() << "Quotient: " << *Q << "\n";
        dbgs() << "Remainder: " << *R << "\n";
      });

    Res = Q;

    // Do not record the last subscript corresponding to the size of elements in
    // the array.
    if (i == Last) {

      // Bail out if the remainder is too complex.
      if (isa<SCEVAddRecExpr>(R)) {
        Subscripts.clear();
        Sizes.clear();
        return;
      }

      continue;
    }

    // Record the access function for the current subscript.
    Subscripts.push_back(R);
  }

  // Also push in last position the remainder of the last division: it will be
  // the access function of the innermost dimension.
  Subscripts.push_back(Res);

  std::reverse(Subscripts.begin(), Subscripts.end());

  DEBUG({
      dbgs() << "Subscripts:\n";
      for (const SCEV *S : Subscripts)
        dbgs() << *S << "\n";
    });
}

/// Splits the SCEV into two vectors of SCEVs representing the subscripts and
/// sizes of an array access. Returns the remainder of the delinearization that
/// is the offset start of the array.  The SCEV->delinearize algorithm computes
/// the multiples of SCEV coefficients: that is a pattern matching of sub
/// expressions in the stride and base of a SCEV corresponding to the
/// computation of a GCD (greatest common divisor) of base and stride.  When
/// SCEV->delinearize fails, it returns the SCEV unchanged.
///
/// For example: when analyzing the memory access A[i][j][k] in this loop nest
///
///  void foo(long n, long m, long o, double A[n][m][o]) {
///
///    for (long i = 0; i < n; i++)
///      for (long j = 0; j < m; j++)
///        for (long k = 0; k < o; k++)
///          A[i][j][k] = 1.0;
///  }
///
/// the delinearization input is the following AddRec SCEV:
///
///  AddRec: {{{%A,+,(8 * %m * %o)}<%for.i>,+,(8 * %o)}<%for.j>,+,8}<%for.k>
///
/// From this SCEV, we are able to say that the base offset of the access is %A
/// because it appears as an offset that does not divide any of the strides in
/// the loops:
///
///  CHECK: Base offset: %A
///
/// and then SCEV->delinearize determines the size of some of the dimensions of
/// the array as these are the multiples by which the strides are happening:
///
///  CHECK: ArrayDecl[UnknownSize][%m][%o] with elements of sizeof(double) bytes.
///
/// Note that the outermost dimension remains of UnknownSize because there are
/// no strides that would help identifying the size of the last dimension: when
/// the array has been statically allocated, one could compute the size of that
/// dimension by dividing the overall size of the array by the size of the known
/// dimensions: %m * %o * 8.
///
/// Finally delinearize provides the access functions for the array reference
/// that does correspond to A[i][j][k] of the above C testcase:
///
///  CHECK: ArrayRef[{0,+,1}<%for.i>][{0,+,1}<%for.j>][{0,+,1}<%for.k>]
///
/// The testcases are checking the output of a function pass:
/// DelinearizationPass that walks through all loads and stores of a function
/// asking for the SCEV of the memory access with respect to all enclosing
/// loops, calling SCEV->delinearize on that and printing the results.

void ScalarEvolution::delinearize(const SCEV *Expr,
                                 SmallVectorImpl<const SCEV *> &Subscripts,
                                 SmallVectorImpl<const SCEV *> &Sizes,
                                 const SCEV *ElementSize) {
  // First step: collect parametric terms.
  SmallVector<const SCEV *, 4> Terms;
  collectParametricTerms(Expr, Terms);

  if (Terms.empty())
    return;

  // Second step: find subscript sizes.
  findArrayDimensions(Terms, Sizes, ElementSize);

  if (Sizes.empty())
    return;

  // Third step: compute the access functions for each subscript.
  computeAccessFunctions(Expr, Subscripts, Sizes);

  if (Subscripts.empty())
    return;

  DEBUG({
      dbgs() << "succeeded to delinearize " << *Expr << "\n";
      dbgs() << "ArrayDecl[UnknownSize]";
      for (const SCEV *S : Sizes)
        dbgs() << "[" << *S << "]";

      dbgs() << "\nArrayRef";
      for (const SCEV *S : Subscripts)
        dbgs() << "[" << *S << "]";
      dbgs() << "\n";
    });
}

//===----------------------------------------------------------------------===//
//                   SCEVCallbackVH Class Implementation
//===----------------------------------------------------------------------===//

void ScalarEvolution::SCEVCallbackVH::deleted() {
  assert(SE && "SCEVCallbackVH called with a null ScalarEvolution!");
  if (PHINode *PN = dyn_cast<PHINode>(getValPtr()))
    SE->ConstantEvolutionLoopExitValue.erase(PN);
  SE->ValueExprMap.erase(getValPtr());
  // this now dangles!
}

void ScalarEvolution::SCEVCallbackVH::allUsesReplacedWith(Value *V) {
  assert(SE && "SCEVCallbackVH called with a null ScalarEvolution!");

  // Forget all the expressions associated with users of the old value,
  // so that future queries will recompute the expressions using the new
  // value.
  Value *Old = getValPtr();
  SmallVector<User *, 16> Worklist(Old->user_begin(), Old->user_end());
  SmallPtrSet<User *, 8> Visited;
  while (!Worklist.empty()) {
    User *U = Worklist.pop_back_val();
    // Deleting the Old value will cause this to dangle. Postpone
    // that until everything else is done.
    if (U == Old)
      continue;
    if (!Visited.insert(U).second)
      continue;
    if (PHINode *PN = dyn_cast<PHINode>(U))
      SE->ConstantEvolutionLoopExitValue.erase(PN);
    SE->ValueExprMap.erase(U);
    Worklist.insert(Worklist.end(), U->user_begin(), U->user_end());
  }
  // Delete the Old value.
  if (PHINode *PN = dyn_cast<PHINode>(Old))
    SE->ConstantEvolutionLoopExitValue.erase(PN);
  SE->ValueExprMap.erase(Old);
  // this now dangles!
}

ScalarEvolution::SCEVCallbackVH::SCEVCallbackVH(Value *V, ScalarEvolution *se)
  : CallbackVH(V), SE(se) {}

//===----------------------------------------------------------------------===//
//                   ScalarEvolution Class Implementation
//===----------------------------------------------------------------------===//

ScalarEvolution::ScalarEvolution()
    : FunctionPass(ID), WalkingBEDominatingConds(false), ValuesAtScopes(64),
      LoopDispositions(64), BlockDispositions(64), FirstUnknown(nullptr) {
  initializeScalarEvolutionPass(*PassRegistry::getPassRegistry());
}

bool ScalarEvolution::runOnFunction(Function &F) {
  this->F = &F;
  AC = &getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
  LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  TLI = &getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();
  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  return false;
}

void ScalarEvolution::releaseMemory() {
  // Iterate through all the SCEVUnknown instances and call their
  // destructors, so that they release their references to their values.
  for (SCEVUnknown *U = FirstUnknown; U; U = U->Next)
    U->~SCEVUnknown();
  FirstUnknown = nullptr;

  ValueExprMap.clear();

  // Free any extra memory created for ExitNotTakenInfo in the unlikely event
  // that a loop had multiple computable exits.
  for (DenseMap<const Loop*, BackedgeTakenInfo>::iterator I =
         BackedgeTakenCounts.begin(), E = BackedgeTakenCounts.end();
       I != E; ++I) {
    I->second.clear();
  }

  assert(PendingLoopPredicates.empty() && "isImpliedCond garbage");
  assert(!WalkingBEDominatingConds && "isLoopBackedgeGuardedByCond garbage!");

  BackedgeTakenCounts.clear();
  ConstantEvolutionLoopExitValue.clear();
  ValuesAtScopes.clear();
  LoopDispositions.clear();
  BlockDispositions.clear();
  UnsignedRanges.clear();
  SignedRanges.clear();
  UniqueSCEVs.clear();
  SCEVAllocator.Reset();
}

void ScalarEvolution::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<AssumptionCacheTracker>();
  AU.addRequiredTransitive<LoopInfoWrapperPass>();
  AU.addRequiredTransitive<DominatorTreeWrapperPass>();
  AU.addRequired<TargetLibraryInfoWrapperPass>();
  AU.addRequired<DxilValueCache>(); // HLSL Change
}

bool ScalarEvolution::hasLoopInvariantBackedgeTakenCount(const Loop *L) {
  return !isa<SCEVCouldNotCompute>(getBackedgeTakenCount(L));
}

static void PrintLoopInfo(raw_ostream &OS, ScalarEvolution *SE,
                          const Loop *L) {
  // Print all inner loops first
  for (Loop::iterator I = L->begin(), E = L->end(); I != E; ++I)
    PrintLoopInfo(OS, SE, *I);

  OS << "Loop ";
  L->getHeader()->printAsOperand(OS, /*PrintType=*/false);
  OS << ": ";

  SmallVector<BasicBlock *, 8> ExitBlocks;
  L->getExitBlocks(ExitBlocks);
  if (ExitBlocks.size() != 1)
    OS << "<multiple exits> ";

  if (SE->hasLoopInvariantBackedgeTakenCount(L)) {
    OS << "backedge-taken count is " << *SE->getBackedgeTakenCount(L);
  } else {
    OS << "Unpredictable backedge-taken count. ";
  }

  OS << "\n"
        "Loop ";
  L->getHeader()->printAsOperand(OS, /*PrintType=*/false);
  OS << ": ";

  if (!isa<SCEVCouldNotCompute>(SE->getMaxBackedgeTakenCount(L))) {
    OS << "max backedge-taken count is " << *SE->getMaxBackedgeTakenCount(L);
  } else {
    OS << "Unpredictable max backedge-taken count. ";
  }

  OS << "\n";
}

void ScalarEvolution::print(raw_ostream &OS, const Module *) const {
  // ScalarEvolution's implementation of the print method is to print
  // out SCEV values of all instructions that are interesting. Doing
  // this potentially causes it to create new SCEV objects though,
  // which technically conflicts with the const qualifier. This isn't
  // observable from outside the class though, so casting away the
  // const isn't dangerous.
  ScalarEvolution &SE = *const_cast<ScalarEvolution *>(this);

  OS << "Classifying expressions for: ";
  F->printAsOperand(OS, /*PrintType=*/false);
  OS << "\n";
  for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I)
    if (isSCEVable(I->getType()) && !isa<CmpInst>(*I)) {
      OS << *I << '\n';
      OS << "  -->  ";
      const SCEV *SV = SE.getSCEV(&*I);
      SV->print(OS);
      if (!isa<SCEVCouldNotCompute>(SV)) {
        OS << " U: ";
        SE.getUnsignedRange(SV).print(OS);
        OS << " S: ";
        SE.getSignedRange(SV).print(OS);
      }

      const Loop *L = LI->getLoopFor((*I).getParent());

      const SCEV *AtUse = SE.getSCEVAtScope(SV, L);
      if (AtUse != SV) {
        OS << "  -->  ";
        AtUse->print(OS);
        if (!isa<SCEVCouldNotCompute>(AtUse)) {
          OS << " U: ";
          SE.getUnsignedRange(AtUse).print(OS);
          OS << " S: ";
          SE.getSignedRange(AtUse).print(OS);
        }
      }

      if (L) {
        OS << "\t\t" "Exits: ";
        const SCEV *ExitValue = SE.getSCEVAtScope(SV, L->getParentLoop());
        if (!SE.isLoopInvariant(ExitValue, L)) {
          OS << "<<Unknown>>";
        } else {
          OS << *ExitValue;
        }
      }

      OS << "\n";
    }

  OS << "Determining loop execution counts for: ";
  F->printAsOperand(OS, /*PrintType=*/false);
  OS << "\n";
  for (LoopInfo::iterator I = LI->begin(), E = LI->end(); I != E; ++I)
    PrintLoopInfo(OS, &SE, *I);
}

ScalarEvolution::LoopDisposition
ScalarEvolution::getLoopDisposition(const SCEV *S, const Loop *L) {
  auto &Values = LoopDispositions[S];
  for (auto &V : Values) {
    if (V.getPointer() == L)
      return V.getInt();
  }
  Values.emplace_back(L, LoopVariant);
  LoopDisposition D = computeLoopDisposition(S, L);
  auto &Values2 = LoopDispositions[S];
  for (auto &V : make_range(Values2.rbegin(), Values2.rend())) {
    if (V.getPointer() == L) {
      V.setInt(D);
      break;
    }
  }
  return D;
}

ScalarEvolution::LoopDisposition
ScalarEvolution::computeLoopDisposition(const SCEV *S, const Loop *L) {
  switch (static_cast<SCEVTypes>(S->getSCEVType())) {
  case scConstant:
    return LoopInvariant;
  case scTruncate:
  case scZeroExtend:
  case scSignExtend:
    return getLoopDisposition(cast<SCEVCastExpr>(S)->getOperand(), L);
  case scAddRecExpr: {
    const SCEVAddRecExpr *AR = cast<SCEVAddRecExpr>(S);

    // If L is the addrec's loop, it's computable.
    if (AR->getLoop() == L)
      return LoopComputable;

    // Add recurrences are never invariant in the function-body (null loop).
    if (!L)
      return LoopVariant;

    // This recurrence is variant w.r.t. L if L contains AR's loop.
    if (L->contains(AR->getLoop()))
      return LoopVariant;

    // This recurrence is invariant w.r.t. L if AR's loop contains L.
    if (AR->getLoop()->contains(L))
      return LoopInvariant;

    // This recurrence is variant w.r.t. L if any of its operands
    // are variant.
    for (SCEVAddRecExpr::op_iterator I = AR->op_begin(), E = AR->op_end();
         I != E; ++I)
      if (!isLoopInvariant(*I, L))
        return LoopVariant;

    // Otherwise it's loop-invariant.
    return LoopInvariant;
  }
  case scAddExpr:
  case scMulExpr:
  case scUMaxExpr:
  case scSMaxExpr: {
    const SCEVNAryExpr *NAry = cast<SCEVNAryExpr>(S);
    bool HasVarying = false;
    for (SCEVNAryExpr::op_iterator I = NAry->op_begin(), E = NAry->op_end();
         I != E; ++I) {
      LoopDisposition D = getLoopDisposition(*I, L);
      if (D == LoopVariant)
        return LoopVariant;
      if (D == LoopComputable)
        HasVarying = true;
    }
    return HasVarying ? LoopComputable : LoopInvariant;
  }
  case scUDivExpr: {
    const SCEVUDivExpr *UDiv = cast<SCEVUDivExpr>(S);
    LoopDisposition LD = getLoopDisposition(UDiv->getLHS(), L);
    if (LD == LoopVariant)
      return LoopVariant;
    LoopDisposition RD = getLoopDisposition(UDiv->getRHS(), L);
    if (RD == LoopVariant)
      return LoopVariant;
    return (LD == LoopInvariant && RD == LoopInvariant) ?
           LoopInvariant : LoopComputable;
  }
  case scUnknown:
    // All non-instruction values are loop invariant.  All instructions are loop
    // invariant if they are not contained in the specified loop.
    // Instructions are never considered invariant in the function body
    // (null loop) because they are defined within the "loop".
    if (Instruction *I = dyn_cast<Instruction>(cast<SCEVUnknown>(S)->getValue()))
      return (L && !L->contains(I)) ? LoopInvariant : LoopVariant;
    return LoopInvariant;
  case scCouldNotCompute:
    llvm_unreachable("Attempt to use a SCEVCouldNotCompute object!");
  }
  llvm_unreachable("Unknown SCEV kind!");
}

bool ScalarEvolution::isLoopInvariant(const SCEV *S, const Loop *L) {
  return getLoopDisposition(S, L) == LoopInvariant;
}

bool ScalarEvolution::hasComputableLoopEvolution(const SCEV *S, const Loop *L) {
  return getLoopDisposition(S, L) == LoopComputable;
}

ScalarEvolution::BlockDisposition
ScalarEvolution::getBlockDisposition(const SCEV *S, const BasicBlock *BB) {
  auto &Values = BlockDispositions[S];
  for (auto &V : Values) {
    if (V.getPointer() == BB)
      return V.getInt();
  }
  Values.emplace_back(BB, DoesNotDominateBlock);
  BlockDisposition D = computeBlockDisposition(S, BB);
  auto &Values2 = BlockDispositions[S];
  for (auto &V : make_range(Values2.rbegin(), Values2.rend())) {
    if (V.getPointer() == BB) {
      V.setInt(D);
      break;
    }
  }
  return D;
}

ScalarEvolution::BlockDisposition
ScalarEvolution::computeBlockDisposition(const SCEV *S, const BasicBlock *BB) {
  switch (static_cast<SCEVTypes>(S->getSCEVType())) {
  case scConstant:
    return ProperlyDominatesBlock;
  case scTruncate:
  case scZeroExtend:
  case scSignExtend:
    return getBlockDisposition(cast<SCEVCastExpr>(S)->getOperand(), BB);
  case scAddRecExpr: {
    // This uses a "dominates" query instead of "properly dominates" query
    // to test for proper dominance too, because the instruction which
    // produces the addrec's value is a PHI, and a PHI effectively properly
    // dominates its entire containing block.
    const SCEVAddRecExpr *AR = cast<SCEVAddRecExpr>(S);
    if (!DT->dominates(AR->getLoop()->getHeader(), BB))
      return DoesNotDominateBlock;
  }
  // FALL THROUGH into SCEVNAryExpr handling.
  case scAddExpr:
  case scMulExpr:
  case scUMaxExpr:
  case scSMaxExpr: {
    const SCEVNAryExpr *NAry = cast<SCEVNAryExpr>(S);
    bool Proper = true;
    for (SCEVNAryExpr::op_iterator I = NAry->op_begin(), E = NAry->op_end();
         I != E; ++I) {
      BlockDisposition D = getBlockDisposition(*I, BB);
      if (D == DoesNotDominateBlock)
        return DoesNotDominateBlock;
      if (D == DominatesBlock)
        Proper = false;
    }
    return Proper ? ProperlyDominatesBlock : DominatesBlock;
  }
  case scUDivExpr: {
    const SCEVUDivExpr *UDiv = cast<SCEVUDivExpr>(S);
    const SCEV *LHS = UDiv->getLHS(), *RHS = UDiv->getRHS();
    BlockDisposition LD = getBlockDisposition(LHS, BB);
    if (LD == DoesNotDominateBlock)
      return DoesNotDominateBlock;
    BlockDisposition RD = getBlockDisposition(RHS, BB);
    if (RD == DoesNotDominateBlock)
      return DoesNotDominateBlock;
    return (LD == ProperlyDominatesBlock && RD == ProperlyDominatesBlock) ?
      ProperlyDominatesBlock : DominatesBlock;
  }
  case scUnknown:
    if (Instruction *I =
          dyn_cast<Instruction>(cast<SCEVUnknown>(S)->getValue())) {
      if (I->getParent() == BB)
        return DominatesBlock;
      if (DT->properlyDominates(I->getParent(), BB))
        return ProperlyDominatesBlock;
      return DoesNotDominateBlock;
    }
    return ProperlyDominatesBlock;
  case scCouldNotCompute:
    llvm_unreachable("Attempt to use a SCEVCouldNotCompute object!");
  }
  llvm_unreachable("Unknown SCEV kind!");
}

bool ScalarEvolution::dominates(const SCEV *S, const BasicBlock *BB) {
  return getBlockDisposition(S, BB) >= DominatesBlock;
}

bool ScalarEvolution::properlyDominates(const SCEV *S, const BasicBlock *BB) {
  return getBlockDisposition(S, BB) == ProperlyDominatesBlock;
}

namespace {
// Search for a SCEV expression node within an expression tree.
// Implements SCEVTraversal::Visitor.
struct SCEVSearch {
  const SCEV *Node;
  bool IsFound;

  SCEVSearch(const SCEV *N): Node(N), IsFound(false) {}

  bool follow(const SCEV *S) {
    IsFound |= (S == Node);
    return !IsFound;
  }
  bool isDone() const { return IsFound; }
};
}

bool ScalarEvolution::hasOperand(const SCEV *S, const SCEV *Op) const {
  SCEVSearch Search(Op);
  visitAll(S, Search);
  return Search.IsFound;
}

void ScalarEvolution::forgetMemoizedResults(const SCEV *S) {
  ValuesAtScopes.erase(S);
  LoopDispositions.erase(S);
  BlockDispositions.erase(S);
  UnsignedRanges.erase(S);
  SignedRanges.erase(S);

  for (DenseMap<const Loop*, BackedgeTakenInfo>::iterator I =
         BackedgeTakenCounts.begin(), E = BackedgeTakenCounts.end(); I != E; ) {
    BackedgeTakenInfo &BEInfo = I->second;
    if (BEInfo.hasOperand(S, this)) {
      BEInfo.clear();
      BackedgeTakenCounts.erase(I++);
    }
    else
      ++I;
  }
}

typedef DenseMap<const Loop *, std::string> VerifyMap;

/// replaceSubString - Replaces all occurrences of From in Str with To.
static void replaceSubString(std::string &Str, StringRef From, StringRef To) {
  size_t Pos = 0;
  while ((Pos = Str.find(From, Pos)) != std::string::npos) {
    Str.replace(Pos, From.size(), To.data(), To.size());
    Pos += To.size();
  }
}

/// getLoopBackedgeTakenCounts - Helper method for verifyAnalysis.
static void
getLoopBackedgeTakenCounts(Loop *L, VerifyMap &Map, ScalarEvolution &SE) {
  for (Loop::reverse_iterator I = L->rbegin(), E = L->rend(); I != E; ++I) {
    getLoopBackedgeTakenCounts(*I, Map, SE); // recurse.

    std::string &S = Map[L];
    if (S.empty()) {
      raw_string_ostream OS(S);
      SE.getBackedgeTakenCount(L)->print(OS);

      // false and 0 are semantically equivalent. This can happen in dead loops.
      replaceSubString(OS.str(), "false", "0");
      // Remove wrap flags, their use in SCEV is highly fragile.
      // FIXME: Remove this when SCEV gets smarter about them.
      replaceSubString(OS.str(), "<nw>", "");
      replaceSubString(OS.str(), "<nsw>", "");
      replaceSubString(OS.str(), "<nuw>", "");
    }
  }
}

void ScalarEvolution::verifyAnalysis() const {
  if (!VerifySCEV)
    return;

  ScalarEvolution &SE = *const_cast<ScalarEvolution *>(this);

  // Gather stringified backedge taken counts for all loops using SCEV's caches.
  // FIXME: It would be much better to store actual values instead of strings,
  //        but SCEV pointers will change if we drop the caches.
  VerifyMap BackedgeDumpsOld, BackedgeDumpsNew;
  for (LoopInfo::reverse_iterator I = LI->rbegin(), E = LI->rend(); I != E; ++I)
    getLoopBackedgeTakenCounts(*I, BackedgeDumpsOld, SE);

  // Gather stringified backedge taken counts for all loops without using
  // SCEV's caches.
  SE.releaseMemory();
  for (LoopInfo::reverse_iterator I = LI->rbegin(), E = LI->rend(); I != E; ++I)
    getLoopBackedgeTakenCounts(*I, BackedgeDumpsNew, SE);

  // Now compare whether they're the same with and without caches. This allows
  // verifying that no pass changed the cache.
  assert(BackedgeDumpsOld.size() == BackedgeDumpsNew.size() &&
         "New loops suddenly appeared!");

  for (VerifyMap::iterator OldI = BackedgeDumpsOld.begin(),
                           OldE = BackedgeDumpsOld.end(),
                           NewI = BackedgeDumpsNew.begin();
       OldI != OldE; ++OldI, ++NewI) {
    assert(OldI->first == NewI->first && "Loop order changed!");

    // Compare the stringified SCEVs. We don't care if undef backedgetaken count
    // changes.
    // FIXME: We currently ignore SCEV changes from/to CouldNotCompute. This
    // means that a pass is buggy or SCEV has to learn a new pattern but is
    // usually not harmful.
    if (OldI->second != NewI->second &&
        OldI->second.find("undef") == std::string::npos &&
        NewI->second.find("undef") == std::string::npos &&
        OldI->second != "***COULDNOTCOMPUTE***" &&
        NewI->second != "***COULDNOTCOMPUTE***") {
      dbgs() << "SCEVValidator: SCEV for loop '"
             << OldI->first->getHeader()->getName()
             << "' changed from '" << OldI->second
             << "' to '" << NewI->second << "'!\n";
      std::abort();
    }
  }

  // TODO: Verify more things.
}
