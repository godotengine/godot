//===- llvm/Transforms/Utils/LoopUtils.h - Loop utilities -*- C++ -*-=========//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines some loop transformation utilities.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_LOOPUTILS_H
#define LLVM_TRANSFORMS_UTILS_LOOPUTILS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"

namespace llvm {
class AliasAnalysis;
class AliasSet;
class AliasSetTracker;
class AssumptionCache;
class BasicBlock;
class DataLayout;
class DominatorTree;
class Loop;
class LoopInfo;
class Pass;
class PredIteratorCache;
class ScalarEvolution;
class TargetLibraryInfo;

/// \brief Captures loop safety information.
/// It keep information for loop & its header may throw exception.
struct LICMSafetyInfo {
  bool MayThrow;           // The current loop contains an instruction which
                           // may throw.
  bool HeaderMayThrow;     // Same as previous, but specific to loop header
  LICMSafetyInfo() : MayThrow(false), HeaderMayThrow(false)
  {}
};

/// The RecurrenceDescriptor is used to identify recurrences variables in a
/// loop. Reduction is a special case of recurrence that has uses of the
/// recurrence variable outside the loop. The method isReductionPHI identifies
/// reductions that are basic recurrences.
///
/// Basic recurrences are defined as the summation, product, OR, AND, XOR, min,
/// or max of a set of terms. For example: for(i=0; i<n; i++) { total +=
/// array[i]; } is a summation of array elements. Basic recurrences are a
/// special case of chains of recurrences (CR). See ScalarEvolution for CR
/// references.

/// This struct holds information about recurrence variables.
class RecurrenceDescriptor {

public:
  /// This enum represents the kinds of recurrences that we support.
  enum RecurrenceKind {
    RK_NoRecurrence,  ///< Not a recurrence.
    RK_IntegerAdd,    ///< Sum of integers.
    RK_IntegerMult,   ///< Product of integers.
    RK_IntegerOr,     ///< Bitwise or logical OR of numbers.
    RK_IntegerAnd,    ///< Bitwise or logical AND of numbers.
    RK_IntegerXor,    ///< Bitwise or logical XOR of numbers.
    RK_IntegerMinMax, ///< Min/max implemented in terms of select(cmp()).
    RK_FloatAdd,      ///< Sum of floats.
    RK_FloatMult,     ///< Product of floats.
    RK_FloatMinMax    ///< Min/max implemented in terms of select(cmp()).
  };

  // This enum represents the kind of minmax recurrence.
  enum MinMaxRecurrenceKind {
    MRK_Invalid,
    MRK_UIntMin,
    MRK_UIntMax,
    MRK_SIntMin,
    MRK_SIntMax,
    MRK_FloatMin,
    MRK_FloatMax
  };

  RecurrenceDescriptor()
      : StartValue(nullptr), LoopExitInstr(nullptr), Kind(RK_NoRecurrence),
        MinMaxKind(MRK_Invalid) {}

  RecurrenceDescriptor(Value *Start, Instruction *Exit, RecurrenceKind K,
                       MinMaxRecurrenceKind MK)
      : StartValue(Start), LoopExitInstr(Exit), Kind(K), MinMaxKind(MK) {}

  /// This POD struct holds information about a potential recurrence operation.
  class InstDesc {

  public:
    InstDesc(bool IsRecur, Instruction *I)
        : IsRecurrence(IsRecur), PatternLastInst(I), MinMaxKind(MRK_Invalid) {}

    InstDesc(Instruction *I, MinMaxRecurrenceKind K)
        : IsRecurrence(true), PatternLastInst(I), MinMaxKind(K) {}

    bool isRecurrence() { return IsRecurrence; }

    MinMaxRecurrenceKind getMinMaxKind() { return MinMaxKind; }

    Instruction *getPatternInst() { return PatternLastInst; }

  private:
    // Is this instruction a recurrence candidate.
    bool IsRecurrence;
    // The last instruction in a min/max pattern (select of the select(icmp())
    // pattern), or the current recurrence instruction otherwise.
    Instruction *PatternLastInst;
    // If this is a min/max pattern the comparison predicate.
    MinMaxRecurrenceKind MinMaxKind;
  };

  /// Returns a struct describing if the instruction 'I' can be a recurrence
  /// variable of type 'Kind'. If the recurrence is a min/max pattern of
  /// select(icmp()) this function advances the instruction pointer 'I' from the
  /// compare instruction to the select instruction and stores this pointer in
  /// 'PatternLastInst' member of the returned struct.
  static InstDesc isRecurrenceInstr(Instruction *I, RecurrenceKind Kind,
                                    InstDesc &Prev, bool HasFunNoNaNAttr);

  /// Returns true if instuction I has multiple uses in Insts
  static bool hasMultipleUsesOf(Instruction *I,
                                SmallPtrSetImpl<Instruction *> &Insts);

  /// Returns true if all uses of the instruction I is within the Set.
  static bool areAllUsesIn(Instruction *I, SmallPtrSetImpl<Instruction *> &Set);

  /// Returns a struct describing if the instruction if the instruction is a
  /// Select(ICmp(X, Y), X, Y) instruction pattern corresponding to a min(X, Y)
  /// or max(X, Y).
  static InstDesc isMinMaxSelectCmpPattern(Instruction *I, InstDesc &Prev);

  /// Returns identity corresponding to the RecurrenceKind.
  static Constant *getRecurrenceIdentity(RecurrenceKind K, Type *Tp);

  /// Returns the opcode of binary operation corresponding to the
  /// RecurrenceKind.
  static unsigned getRecurrenceBinOp(RecurrenceKind Kind);

  /// Returns a Min/Max operation corresponding to MinMaxRecurrenceKind.
  static Value *createMinMaxOp(IRBuilder<> &Builder, MinMaxRecurrenceKind RK,
                               Value *Left, Value *Right);

  /// Returns true if Phi is a reduction of type Kind and adds it to the
  /// RecurrenceDescriptor.
  static bool AddReductionVar(PHINode *Phi, RecurrenceKind Kind, Loop *TheLoop,
                              bool HasFunNoNaNAttr,
                              RecurrenceDescriptor &RedDes);

  /// Returns true if Phi is a reduction in TheLoop. The RecurrenceDescriptor is
  /// returned in RedDes.
  static bool isReductionPHI(PHINode *Phi, Loop *TheLoop,
                             RecurrenceDescriptor &RedDes);

  RecurrenceKind getRecurrenceKind() { return Kind; }

  MinMaxRecurrenceKind getMinMaxRecurrenceKind() { return MinMaxKind; }

  TrackingVH<Value> getRecurrenceStartValue() { return StartValue; }

  Instruction *getLoopExitInstr() { return LoopExitInstr; }

private:
  // The starting value of the recurrence.
  // It does not have to be zero!
  TrackingVH<Value> StartValue;
  // The instruction who's value is used outside the loop.
  Instruction *LoopExitInstr;
  // The kind of the recurrence.
  RecurrenceKind Kind;
  // If this a min/max recurrence the kind of recurrence.
  MinMaxRecurrenceKind MinMaxKind;
};

BasicBlock *InsertPreheaderForLoop(Loop *L, Pass *P);

/// \brief Simplify each loop in a loop nest recursively.
///
/// This takes a potentially un-simplified loop L (and its children) and turns
/// it into a simplified loop nest with preheaders and single backedges. It
/// will optionally update \c AliasAnalysis and \c ScalarEvolution analyses if
/// passed into it.
bool simplifyLoop(Loop *L, DominatorTree *DT, LoopInfo *LI, Pass *PP,
                  AliasAnalysis *AA = nullptr, ScalarEvolution *SE = nullptr,
                  AssumptionCache *AC = nullptr);

/// \brief Put loop into LCSSA form.
///
/// Looks at all instructions in the loop which have uses outside of the
/// current loop. For each, an LCSSA PHI node is inserted and the uses outside
/// the loop are rewritten to use this node.
///
/// LoopInfo and DominatorTree are required and preserved.
///
/// If ScalarEvolution is passed in, it will be preserved.
///
/// Returns true if any modifications are made to the loop.
bool formLCSSA(Loop &L, DominatorTree &DT, LoopInfo *LI,
               ScalarEvolution *SE = nullptr);

/// \brief Put a loop nest into LCSSA form.
///
/// This recursively forms LCSSA for a loop nest.
///
/// LoopInfo and DominatorTree are required and preserved.
///
/// If ScalarEvolution is passed in, it will be preserved.
///
/// Returns true if any modifications are made to the loop.
bool formLCSSARecursively(Loop &L, DominatorTree &DT, LoopInfo *LI,
                          ScalarEvolution *SE = nullptr);

/// \brief Walk the specified region of the CFG (defined by all blocks
/// dominated by the specified block, and that are in the current loop) in
/// reverse depth first order w.r.t the DominatorTree. This allows us to visit
/// uses before definitions, allowing us to sink a loop body in one pass without
/// iteration. Takes DomTreeNode, AliasAnalysis, LoopInfo, DominatorTree,
/// DataLayout, TargetLibraryInfo, Loop, AliasSet information for all
/// instructions of the loop and loop safety information as arguments.
/// It returns changed status.
bool sinkRegion(DomTreeNode *, AliasAnalysis *, LoopInfo *, DominatorTree *,
                TargetLibraryInfo *, Loop *, AliasSetTracker *,
                LICMSafetyInfo *);

/// \brief Walk the specified region of the CFG (defined by all blocks
/// dominated by the specified block, and that are in the current loop) in depth
/// first order w.r.t the DominatorTree.  This allows us to visit definitions
/// before uses, allowing us to hoist a loop body in one pass without iteration.
/// Takes DomTreeNode, AliasAnalysis, LoopInfo, DominatorTree, DataLayout,
/// TargetLibraryInfo, Loop, AliasSet information for all instructions of the
/// loop and loop safety information as arguments. It returns changed status.
bool hoistRegion(DomTreeNode *, AliasAnalysis *, LoopInfo *, DominatorTree *,
                 TargetLibraryInfo *, Loop *, AliasSetTracker *,
                 LICMSafetyInfo *);

/// \brief Try to promote memory values to scalars by sinking stores out of
/// the loop and moving loads to before the loop.  We do this by looping over
/// the stores in the loop, looking for stores to Must pointers which are 
/// loop invariant. It takes AliasSet, Loop exit blocks vector, loop exit blocks
/// insertion point vector, PredIteratorCache, LoopInfo, DominatorTree, Loop,
/// AliasSet information for all instructions of the loop and loop safety 
/// information as arguments. It returns changed status.
bool promoteLoopAccessesToScalars(AliasSet &, SmallVectorImpl<BasicBlock*> &,
                                  SmallVectorImpl<Instruction*> &,
                                  PredIteratorCache &, LoopInfo *,
                                  DominatorTree *, Loop *, AliasSetTracker *,
                                  LICMSafetyInfo *);

/// \brief Computes safety information for a loop
/// checks loop body & header for the possiblity of may throw
/// exception, it takes LICMSafetyInfo and loop as argument.
/// Updates safety information in LICMSafetyInfo argument.
void computeLICMSafetyInfo(LICMSafetyInfo *, Loop *);

/// \brief Checks if the given PHINode in a loop header is an induction
/// variable. Returns true if this is an induction PHI along with the step
/// value.
bool isInductionPHI(PHINode *, ScalarEvolution *, ConstantInt *&);
}

#endif
