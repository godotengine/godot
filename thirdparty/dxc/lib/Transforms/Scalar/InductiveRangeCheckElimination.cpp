//===-- InductiveRangeCheckElimination.cpp - ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// The InductiveRangeCheckElimination pass splits a loop's iteration space into
// three disjoint ranges.  It does that in a way such that the loop running in
// the middle loop provably does not need range checks. As an example, it will
// convert
//
//   len = < known positive >
//   for (i = 0; i < n; i++) {
//     if (0 <= i && i < len) {
//       do_something();
//     } else {
//       throw_out_of_bounds();
//     }
//   }
//
// to
//
//   len = < known positive >
//   limit = smin(n, len)
//   // no first segment
//   for (i = 0; i < limit; i++) {
//     if (0 <= i && i < len) { // this check is fully redundant
//       do_something();
//     } else {
//       throw_out_of_bounds();
//     }
//   }
//   for (i = limit; i < n; i++) {
//     if (0 <= i && i < len) {
//       do_something();
//     } else {
//       throw_out_of_bounds();
//     }
//   }
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Optional.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/SimplifyIndVar.h"
#include "llvm/Transforms/Utils/UnrollLoop.h"
#include <array>

using namespace llvm;

static cl::opt<unsigned> LoopSizeCutoff("irce-loop-size-cutoff", cl::Hidden,
                                        cl::init(64));

static cl::opt<bool> PrintChangedLoops("irce-print-changed-loops", cl::Hidden,
                                       cl::init(false));

static cl::opt<bool> PrintRangeChecks("irce-print-range-checks", cl::Hidden,
                                      cl::init(false));

static cl::opt<int> MaxExitProbReciprocal("irce-max-exit-prob-reciprocal",
                                          cl::Hidden, cl::init(10));

#define DEBUG_TYPE "irce"

namespace {

/// An inductive range check is conditional branch in a loop with
///
///  1. a very cold successor (i.e. the branch jumps to that successor very
///     rarely)
///
///  and
///
///  2. a condition that is provably true for some contiguous range of values
///     taken by the containing loop's induction variable.
///
class InductiveRangeCheck {
  // Classifies a range check
  enum RangeCheckKind : unsigned {
    // Range check of the form "0 <= I".
    RANGE_CHECK_LOWER = 1,

    // Range check of the form "I < L" where L is known positive.
    RANGE_CHECK_UPPER = 2,

    // The logical and of the RANGE_CHECK_LOWER and RANGE_CHECK_UPPER
    // conditions.
    RANGE_CHECK_BOTH = RANGE_CHECK_LOWER | RANGE_CHECK_UPPER,

    // Unrecognized range check condition.
    RANGE_CHECK_UNKNOWN = (unsigned)-1
  };

  static const char *rangeCheckKindToStr(RangeCheckKind);

  const SCEV *Offset;
  const SCEV *Scale;
  Value *Length;
  BranchInst *Branch;
  RangeCheckKind Kind;

  static RangeCheckKind parseRangeCheckICmp(Loop *L, ICmpInst *ICI,
                                            ScalarEvolution &SE, Value *&Index,
                                            Value *&Length);

  static InductiveRangeCheck::RangeCheckKind
  parseRangeCheck(Loop *L, ScalarEvolution &SE, Value *Condition,
                  const SCEV *&Index, Value *&UpperLimit);

  InductiveRangeCheck() :
    Offset(nullptr), Scale(nullptr), Length(nullptr), Branch(nullptr) { }

public:
  const SCEV *getOffset() const { return Offset; }
  const SCEV *getScale() const { return Scale; }
  Value *getLength() const { return Length; }

  void print(raw_ostream &OS) const {
    OS << "InductiveRangeCheck:\n";
    OS << "  Kind: " << rangeCheckKindToStr(Kind) << "\n";
    OS << "  Offset: ";
    Offset->print(OS);
    OS << "  Scale: ";
    Scale->print(OS);
    OS << "  Length: ";
    if (Length)
      Length->print(OS);
    else
      OS << "(null)";
    OS << "\n  Branch: ";
    getBranch()->print(OS);
    OS << "\n";
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  void dump() {
    print(dbgs());
  }
#endif

  BranchInst *getBranch() const { return Branch; }

  /// Represents an signed integer range [Range.getBegin(), Range.getEnd()).  If
  /// R.getEnd() sle R.getBegin(), then R denotes the empty range.

  class Range {
    const SCEV *Begin;
    const SCEV *End;

  public:
    Range(const SCEV *Begin, const SCEV *End) : Begin(Begin), End(End) {
      assert(Begin->getType() == End->getType() && "ill-typed range!");
    }

    Type *getType() const { return Begin->getType(); }
    const SCEV *getBegin() const { return Begin; }
    const SCEV *getEnd() const { return End; }
  };

  typedef SpecificBumpPtrAllocator<InductiveRangeCheck> AllocatorTy;

  /// This is the value the condition of the branch needs to evaluate to for the
  /// branch to take the hot successor (see (1) above).
  bool getPassingDirection() { return true; }

  /// Computes a range for the induction variable (IndVar) in which the range
  /// check is redundant and can be constant-folded away.  The induction
  /// variable is not required to be the canonical {0,+,1} induction variable.
  Optional<Range> computeSafeIterationSpace(ScalarEvolution &SE,
                                            const SCEVAddRecExpr *IndVar,
                                            IRBuilder<> &B) const;

  /// Create an inductive range check out of BI if possible, else return
  /// nullptr.
  static InductiveRangeCheck *create(AllocatorTy &Alloc, BranchInst *BI,
                                     Loop *L, ScalarEvolution &SE,
                                     BranchProbabilityInfo &BPI);
};

class InductiveRangeCheckElimination : public LoopPass {
  InductiveRangeCheck::AllocatorTy Allocator;

public:
  static char ID;
  InductiveRangeCheckElimination() : LoopPass(ID) {
    initializeInductiveRangeCheckEliminationPass(
        *PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequiredID(LoopSimplifyID);
    AU.addRequiredID(LCSSAID);
    AU.addRequired<ScalarEvolution>();
    AU.addRequired<BranchProbabilityInfo>();
  }

  bool runOnLoop(Loop *L, LPPassManager &LPM) override;
};

char InductiveRangeCheckElimination::ID = 0;
}

INITIALIZE_PASS(InductiveRangeCheckElimination, "irce",
                "Inductive range check elimination", false, false)

const char *InductiveRangeCheck::rangeCheckKindToStr(
    InductiveRangeCheck::RangeCheckKind RCK) {
  switch (RCK) {
  case InductiveRangeCheck::RANGE_CHECK_UNKNOWN:
    return "RANGE_CHECK_UNKNOWN";

  case InductiveRangeCheck::RANGE_CHECK_UPPER:
    return "RANGE_CHECK_UPPER";

  case InductiveRangeCheck::RANGE_CHECK_LOWER:
    return "RANGE_CHECK_LOWER";

  case InductiveRangeCheck::RANGE_CHECK_BOTH:
    return "RANGE_CHECK_BOTH";
  }

  llvm_unreachable("unknown range check type!");
}

/// Parse a single ICmp instruction, `ICI`, into a range check.  If `ICI`
/// cannot
/// be interpreted as a range check, return `RANGE_CHECK_UNKNOWN` and set
/// `Index` and `Length` to `nullptr`.  Otherwise set `Index` to the value
/// being
/// range checked, and set `Length` to the upper limit `Index` is being range
/// checked with if (and only if) the range check type is stronger or equal to
/// RANGE_CHECK_UPPER.
///
InductiveRangeCheck::RangeCheckKind
InductiveRangeCheck::parseRangeCheckICmp(Loop *L, ICmpInst *ICI,
                                         ScalarEvolution &SE, Value *&Index,
                                         Value *&Length) {

  auto IsNonNegativeAndNotLoopVarying = [&SE, L](Value *V) {
    const SCEV *S = SE.getSCEV(V);
    if (isa<SCEVCouldNotCompute>(S))
      return false;

    return SE.getLoopDisposition(S, L) == ScalarEvolution::LoopInvariant &&
           SE.isKnownNonNegative(S);
  };

  using namespace llvm::PatternMatch;

  ICmpInst::Predicate Pred = ICI->getPredicate();
  Value *LHS = ICI->getOperand(0);
  Value *RHS = ICI->getOperand(1);

  switch (Pred) {
  default:
    return RANGE_CHECK_UNKNOWN;

  case ICmpInst::ICMP_SLE:
    std::swap(LHS, RHS);
  // fallthrough
  case ICmpInst::ICMP_SGE:
    if (match(RHS, m_ConstantInt<0>())) {
      Index = LHS;
      return RANGE_CHECK_LOWER;
    }
    return RANGE_CHECK_UNKNOWN;

  case ICmpInst::ICMP_SLT:
    std::swap(LHS, RHS);
  // fallthrough
  case ICmpInst::ICMP_SGT:
    if (match(RHS, m_ConstantInt<-1>())) {
      Index = LHS;
      return RANGE_CHECK_LOWER;
    }

    if (IsNonNegativeAndNotLoopVarying(LHS)) {
      Index = RHS;
      Length = LHS;
      return RANGE_CHECK_UPPER;
    }
    return RANGE_CHECK_UNKNOWN;

  case ICmpInst::ICMP_ULT:
    std::swap(LHS, RHS);
  // fallthrough
  case ICmpInst::ICMP_UGT:
    if (IsNonNegativeAndNotLoopVarying(LHS)) {
      Index = RHS;
      Length = LHS;
      return RANGE_CHECK_BOTH;
    }
    return RANGE_CHECK_UNKNOWN;
  }

  llvm_unreachable("default clause returns!");
}

/// Parses an arbitrary condition into a range check.  `Length` is set only if
/// the range check is recognized to be `RANGE_CHECK_UPPER` or stronger.
InductiveRangeCheck::RangeCheckKind
InductiveRangeCheck::parseRangeCheck(Loop *L, ScalarEvolution &SE,
                                     Value *Condition, const SCEV *&Index,
                                     Value *&Length) {
  using namespace llvm::PatternMatch;

  Value *A = nullptr;
  Value *B = nullptr;

  if (match(Condition, m_And(m_Value(A), m_Value(B)))) {
    Value *IndexA = nullptr, *IndexB = nullptr;
    Value *LengthA = nullptr, *LengthB = nullptr;
    ICmpInst *ICmpA = dyn_cast<ICmpInst>(A), *ICmpB = dyn_cast<ICmpInst>(B);

    if (!ICmpA || !ICmpB)
      return InductiveRangeCheck::RANGE_CHECK_UNKNOWN;

    auto RCKindA = parseRangeCheckICmp(L, ICmpA, SE, IndexA, LengthA);
    auto RCKindB = parseRangeCheckICmp(L, ICmpB, SE, IndexB, LengthB);

    if (RCKindA == InductiveRangeCheck::RANGE_CHECK_UNKNOWN ||
        RCKindB == InductiveRangeCheck::RANGE_CHECK_UNKNOWN)
      return InductiveRangeCheck::RANGE_CHECK_UNKNOWN;

    if (IndexA != IndexB)
      return InductiveRangeCheck::RANGE_CHECK_UNKNOWN;

    if (LengthA != nullptr && LengthB != nullptr && LengthA != LengthB)
      return InductiveRangeCheck::RANGE_CHECK_UNKNOWN;

    Index = SE.getSCEV(IndexA);
    if (isa<SCEVCouldNotCompute>(Index))
      return InductiveRangeCheck::RANGE_CHECK_UNKNOWN;

    Length = LengthA == nullptr ? LengthB : LengthA;

    return (InductiveRangeCheck::RangeCheckKind)(RCKindA | RCKindB);
  }

  if (ICmpInst *ICI = dyn_cast<ICmpInst>(Condition)) {
    Value *IndexVal = nullptr;

    auto RCKind = parseRangeCheckICmp(L, ICI, SE, IndexVal, Length);

    if (RCKind == InductiveRangeCheck::RANGE_CHECK_UNKNOWN)
      return InductiveRangeCheck::RANGE_CHECK_UNKNOWN;

    Index = SE.getSCEV(IndexVal);
    if (isa<SCEVCouldNotCompute>(Index))
      return InductiveRangeCheck::RANGE_CHECK_UNKNOWN;

    return RCKind;
  }

  return InductiveRangeCheck::RANGE_CHECK_UNKNOWN;
}


InductiveRangeCheck *
InductiveRangeCheck::create(InductiveRangeCheck::AllocatorTy &A, BranchInst *BI,
                            Loop *L, ScalarEvolution &SE,
                            BranchProbabilityInfo &BPI) {

  if (BI->isUnconditional() || BI->getParent() == L->getLoopLatch())
    return nullptr;

  BranchProbability LikelyTaken(15, 16);

  if (BPI.getEdgeProbability(BI->getParent(), (unsigned) 0) < LikelyTaken)
    return nullptr;

  Value *Length = nullptr;
  const SCEV *IndexSCEV = nullptr;

  auto RCKind = InductiveRangeCheck::parseRangeCheck(L, SE, BI->getCondition(),
                                                     IndexSCEV, Length);

  if (RCKind == InductiveRangeCheck::RANGE_CHECK_UNKNOWN)
    return nullptr;

  assert(IndexSCEV && "contract with SplitRangeCheckCondition!");
  assert((!(RCKind & InductiveRangeCheck::RANGE_CHECK_UPPER) || Length) &&
         "contract with SplitRangeCheckCondition!");

  const SCEVAddRecExpr *IndexAddRec = dyn_cast<SCEVAddRecExpr>(IndexSCEV);
  bool IsAffineIndex =
      IndexAddRec && (IndexAddRec->getLoop() == L) && IndexAddRec->isAffine();

  if (!IsAffineIndex)
    return nullptr;

  InductiveRangeCheck *IRC = new (A.Allocate()) InductiveRangeCheck;
  IRC->Length = Length;
  IRC->Offset = IndexAddRec->getStart();
  IRC->Scale = IndexAddRec->getStepRecurrence(SE);
  IRC->Branch = BI;
  IRC->Kind = RCKind;
  return IRC;
}

namespace {

// Keeps track of the structure of a loop.  This is similar to llvm::Loop,
// except that it is more lightweight and can track the state of a loop through
// changing and potentially invalid IR.  This structure also formalizes the
// kinds of loops we can deal with -- ones that have a single latch that is also
// an exiting block *and* have a canonical induction variable.
struct LoopStructure {
  const char *Tag;

  BasicBlock *Header;
  BasicBlock *Latch;

  // `Latch's terminator instruction is `LatchBr', and it's `LatchBrExitIdx'th
  // successor is `LatchExit', the exit block of the loop.
  BranchInst *LatchBr;
  BasicBlock *LatchExit;
  unsigned LatchBrExitIdx;

  Value *IndVarNext;
  Value *IndVarStart;
  Value *LoopExitAt;
  bool IndVarIncreasing;

  LoopStructure()
      : Tag(""), Header(nullptr), Latch(nullptr), LatchBr(nullptr),
        LatchExit(nullptr), LatchBrExitIdx(-1), IndVarNext(nullptr),
        IndVarStart(nullptr), LoopExitAt(nullptr), IndVarIncreasing(false) {}

  template <typename M> LoopStructure map(M Map) const {
    LoopStructure Result;
    Result.Tag = Tag;
    Result.Header = cast<BasicBlock>(Map(Header));
    Result.Latch = cast<BasicBlock>(Map(Latch));
    Result.LatchBr = cast<BranchInst>(Map(LatchBr));
    Result.LatchExit = cast<BasicBlock>(Map(LatchExit));
    Result.LatchBrExitIdx = LatchBrExitIdx;
    Result.IndVarNext = Map(IndVarNext);
    Result.IndVarStart = Map(IndVarStart);
    Result.LoopExitAt = Map(LoopExitAt);
    Result.IndVarIncreasing = IndVarIncreasing;
    return Result;
  }

  static Optional<LoopStructure> parseLoopStructure(ScalarEvolution &,
                                                    BranchProbabilityInfo &BPI,
                                                    Loop &,
                                                    const char *&);
};

/// This class is used to constrain loops to run within a given iteration space.
/// The algorithm this class implements is given a Loop and a range [Begin,
/// End).  The algorithm then tries to break out a "main loop" out of the loop
/// it is given in a way that the "main loop" runs with the induction variable
/// in a subset of [Begin, End).  The algorithm emits appropriate pre and post
/// loops to run any remaining iterations.  The pre loop runs any iterations in
/// which the induction variable is < Begin, and the post loop runs any
/// iterations in which the induction variable is >= End.
///
class LoopConstrainer {
  // The representation of a clone of the original loop we started out with.
  struct ClonedLoop {
    // The cloned blocks
    std::vector<BasicBlock *> Blocks;

    // `Map` maps values in the clonee into values in the cloned version
    ValueToValueMapTy Map;

    // An instance of `LoopStructure` for the cloned loop
    LoopStructure Structure;
  };

  // Result of rewriting the range of a loop.  See changeIterationSpaceEnd for
  // more details on what these fields mean.
  struct RewrittenRangeInfo {
    BasicBlock *PseudoExit;
    BasicBlock *ExitSelector;
    std::vector<PHINode *> PHIValuesAtPseudoExit;
    PHINode *IndVarEnd;

    RewrittenRangeInfo()
        : PseudoExit(nullptr), ExitSelector(nullptr), IndVarEnd(nullptr) {}
  };

  // Calculated subranges we restrict the iteration space of the main loop to.
  // See the implementation of `calculateSubRanges' for more details on how
  // these fields are computed.  `LowLimit` is None if there is no restriction
  // on low end of the restricted iteration space of the main loop.  `HighLimit`
  // is None if there is no restriction on high end of the restricted iteration
  // space of the main loop.

  struct SubRanges {
    Optional<const SCEV *> LowLimit;
    Optional<const SCEV *> HighLimit;
  };

  // A utility function that does a `replaceUsesOfWith' on the incoming block
  // set of a `PHINode' -- replaces instances of `Block' in the `PHINode's
  // incoming block list with `ReplaceBy'.
  static void replacePHIBlock(PHINode *PN, BasicBlock *Block,
                              BasicBlock *ReplaceBy);

  // Compute a safe set of limits for the main loop to run in -- effectively the
  // intersection of `Range' and the iteration space of the original loop.
  // Return None if unable to compute the set of subranges.
  //
  Optional<SubRanges> calculateSubRanges() const;

  // Clone `OriginalLoop' and return the result in CLResult.  The IR after
  // running `cloneLoop' is well formed except for the PHI nodes in CLResult --
  // the PHI nodes say that there is an incoming edge from `OriginalPreheader`
  // but there is no such edge.
  //
  void cloneLoop(ClonedLoop &CLResult, const char *Tag) const;

  // Rewrite the iteration space of the loop denoted by (LS, Preheader). The
  // iteration space of the rewritten loop ends at ExitLoopAt.  The start of the
  // iteration space is not changed.  `ExitLoopAt' is assumed to be slt
  // `OriginalHeaderCount'.
  //
  // If there are iterations left to execute, control is made to jump to
  // `ContinuationBlock', otherwise they take the normal loop exit.  The
  // returned `RewrittenRangeInfo' object is populated as follows:
  //
  //  .PseudoExit is a basic block that unconditionally branches to
  //      `ContinuationBlock'.
  //
  //  .ExitSelector is a basic block that decides, on exit from the loop,
  //      whether to branch to the "true" exit or to `PseudoExit'.
  //
  //  .PHIValuesAtPseudoExit are PHINodes in `PseudoExit' that compute the value
  //      for each PHINode in the loop header on taking the pseudo exit.
  //
  // After changeIterationSpaceEnd, `Preheader' is no longer a legitimate
  // preheader because it is made to branch to the loop header only
  // conditionally.
  //
  RewrittenRangeInfo
  changeIterationSpaceEnd(const LoopStructure &LS, BasicBlock *Preheader,
                          Value *ExitLoopAt,
                          BasicBlock *ContinuationBlock) const;

  // The loop denoted by `LS' has `OldPreheader' as its preheader.  This
  // function creates a new preheader for `LS' and returns it.
  //
  BasicBlock *createPreheader(const LoopStructure &LS, BasicBlock *OldPreheader,
                              const char *Tag) const;

  // `ContinuationBlockAndPreheader' was the continuation block for some call to
  // `changeIterationSpaceEnd' and is the preheader to the loop denoted by `LS'.
  // This function rewrites the PHI nodes in `LS.Header' to start with the
  // correct value.
  void rewriteIncomingValuesForPHIs(
      LoopStructure &LS, BasicBlock *ContinuationBlockAndPreheader,
      const LoopConstrainer::RewrittenRangeInfo &RRI) const;

  // Even though we do not preserve any passes at this time, we at least need to
  // keep the parent loop structure consistent.  The `LPPassManager' seems to
  // verify this after running a loop pass.  This function adds the list of
  // blocks denoted by BBs to this loops parent loop if required.
  void addToParentLoopIfNeeded(ArrayRef<BasicBlock *> BBs);

  // Some global state.
  Function &F;
  LLVMContext &Ctx;
  ScalarEvolution &SE;

  // Information about the original loop we started out with.
  Loop &OriginalLoop;
  LoopInfo &OriginalLoopInfo;
  const SCEV *LatchTakenCount;
  BasicBlock *OriginalPreheader;

  // The preheader of the main loop.  This may or may not be different from
  // `OriginalPreheader'.
  BasicBlock *MainLoopPreheader;

  // The range we need to run the main loop in.
  InductiveRangeCheck::Range Range;

  // The structure of the main loop (see comment at the beginning of this class
  // for a definition)
  LoopStructure MainLoopStructure;

public:
  LoopConstrainer(Loop &L, LoopInfo &LI, const LoopStructure &LS,
                  ScalarEvolution &SE, InductiveRangeCheck::Range R)
      : F(*L.getHeader()->getParent()), Ctx(L.getHeader()->getContext()),
        SE(SE), OriginalLoop(L), OriginalLoopInfo(LI), LatchTakenCount(nullptr),
        OriginalPreheader(nullptr), MainLoopPreheader(nullptr), Range(R),
        MainLoopStructure(LS) {}

  // Entry point for the algorithm.  Returns true on success.
  bool run();
};

}

void LoopConstrainer::replacePHIBlock(PHINode *PN, BasicBlock *Block,
                                      BasicBlock *ReplaceBy) {
  for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
    if (PN->getIncomingBlock(i) == Block)
      PN->setIncomingBlock(i, ReplaceBy);
}

static bool CanBeSMax(ScalarEvolution &SE, const SCEV *S) {
  APInt SMax =
      APInt::getSignedMaxValue(cast<IntegerType>(S->getType())->getBitWidth());
  return SE.getSignedRange(S).contains(SMax) &&
         SE.getUnsignedRange(S).contains(SMax);
}

static bool CanBeSMin(ScalarEvolution &SE, const SCEV *S) {
  APInt SMin =
      APInt::getSignedMinValue(cast<IntegerType>(S->getType())->getBitWidth());
  return SE.getSignedRange(S).contains(SMin) &&
         SE.getUnsignedRange(S).contains(SMin);
}

Optional<LoopStructure>
LoopStructure::parseLoopStructure(ScalarEvolution &SE, BranchProbabilityInfo &BPI,
                                  Loop &L, const char *&FailureReason) {
  assert(L.isLoopSimplifyForm() && "should follow from addRequired<>");

  BasicBlock *Latch = L.getLoopLatch();
  if (!L.isLoopExiting(Latch)) {
    FailureReason = "no loop latch";
    return None;
  }

  BasicBlock *Header = L.getHeader();
  BasicBlock *Preheader = L.getLoopPreheader();
  if (!Preheader) {
    FailureReason = "no preheader";
    return None;
  }

  BranchInst *LatchBr = dyn_cast<BranchInst>(&*Latch->rbegin());
  if (!LatchBr || LatchBr->isUnconditional()) {
    FailureReason = "latch terminator not conditional branch";
    return None;
  }

  unsigned LatchBrExitIdx = LatchBr->getSuccessor(0) == Header ? 1 : 0;

  BranchProbability ExitProbability =
    BPI.getEdgeProbability(LatchBr->getParent(), LatchBrExitIdx);

  if (ExitProbability > BranchProbability(1, MaxExitProbReciprocal)) {
    FailureReason = "short running loop, not profitable";
    return None;
  }

  ICmpInst *ICI = dyn_cast<ICmpInst>(LatchBr->getCondition());
  if (!ICI || !isa<IntegerType>(ICI->getOperand(0)->getType())) {
    FailureReason = "latch terminator branch not conditional on integral icmp";
    return None;
  }

  const SCEV *LatchCount = SE.getExitCount(&L, Latch);
  if (isa<SCEVCouldNotCompute>(LatchCount)) {
    FailureReason = "could not compute latch count";
    return None;
  }

  ICmpInst::Predicate Pred = ICI->getPredicate();
  Value *LeftValue = ICI->getOperand(0);
  const SCEV *LeftSCEV = SE.getSCEV(LeftValue);
  IntegerType *IndVarTy = cast<IntegerType>(LeftValue->getType());

  Value *RightValue = ICI->getOperand(1);
  const SCEV *RightSCEV = SE.getSCEV(RightValue);

  // We canonicalize `ICI` such that `LeftSCEV` is an add recurrence.
  if (!isa<SCEVAddRecExpr>(LeftSCEV)) {
    if (isa<SCEVAddRecExpr>(RightSCEV)) {
      std::swap(LeftSCEV, RightSCEV);
      std::swap(LeftValue, RightValue);
      Pred = ICmpInst::getSwappedPredicate(Pred);
    } else {
      FailureReason = "no add recurrences in the icmp";
      return None;
    }
  }

  auto HasNoSignedWrap = [&](const SCEVAddRecExpr *AR) {
    if (AR->getNoWrapFlags(SCEV::FlagNSW))
      return true;

    IntegerType *Ty = cast<IntegerType>(AR->getType());
    IntegerType *WideTy =
        IntegerType::get(Ty->getContext(), Ty->getBitWidth() * 2);

    const SCEVAddRecExpr *ExtendAfterOp =
        dyn_cast<SCEVAddRecExpr>(SE.getSignExtendExpr(AR, WideTy));
    if (ExtendAfterOp) {
      const SCEV *ExtendedStart = SE.getSignExtendExpr(AR->getStart(), WideTy);
      const SCEV *ExtendedStep =
          SE.getSignExtendExpr(AR->getStepRecurrence(SE), WideTy);

      bool NoSignedWrap = ExtendAfterOp->getStart() == ExtendedStart &&
                          ExtendAfterOp->getStepRecurrence(SE) == ExtendedStep;

      if (NoSignedWrap)
        return true;
    }

    // We may have proved this when computing the sign extension above.
    return AR->getNoWrapFlags(SCEV::FlagNSW) != SCEV::FlagAnyWrap;
  };

  auto IsInductionVar = [&](const SCEVAddRecExpr *AR, bool &IsIncreasing) {
    if (!AR->isAffine())
      return false;

    // Currently we only work with induction variables that have been proved to
    // not wrap.  This restriction can potentially be lifted in the future.

    if (!HasNoSignedWrap(AR))
      return false;

    if (const SCEVConstant *StepExpr =
            dyn_cast<SCEVConstant>(AR->getStepRecurrence(SE))) {
      ConstantInt *StepCI = StepExpr->getValue();
      if (StepCI->isOne() || StepCI->isMinusOne()) {
        IsIncreasing = StepCI->isOne();
        return true;
      }
    }

    return false;
  };

  // `ICI` is interpreted as taking the backedge if the *next* value of the
  // induction variable satisfies some constraint.

  const SCEVAddRecExpr *IndVarNext = cast<SCEVAddRecExpr>(LeftSCEV);
  bool IsIncreasing = false;
  if (!IsInductionVar(IndVarNext, IsIncreasing)) {
    FailureReason = "LHS in icmp not induction variable";
    return None;
  }

  ConstantInt *One = ConstantInt::get(IndVarTy, 1);
  // TODO: generalize the predicates here to also match their unsigned variants.
  if (IsIncreasing) {
    bool FoundExpectedPred =
        (Pred == ICmpInst::ICMP_SLT && LatchBrExitIdx == 1) ||
        (Pred == ICmpInst::ICMP_SGT && LatchBrExitIdx == 0);

    if (!FoundExpectedPred) {
      FailureReason = "expected icmp slt semantically, found something else";
      return None;
    }

    if (LatchBrExitIdx == 0) {
      if (CanBeSMax(SE, RightSCEV)) {
        // TODO: this restriction is easily removable -- we just have to
        // remember that the icmp was an slt and not an sle.
        FailureReason = "limit may overflow when coercing sle to slt";
        return None;
      }

      IRBuilder<> B(&*Preheader->rbegin());
      RightValue = B.CreateAdd(RightValue, One);
    }

  } else {
    bool FoundExpectedPred =
        (Pred == ICmpInst::ICMP_SGT && LatchBrExitIdx == 1) ||
        (Pred == ICmpInst::ICMP_SLT && LatchBrExitIdx == 0);

    if (!FoundExpectedPred) {
      FailureReason = "expected icmp sgt semantically, found something else";
      return None;
    }

    if (LatchBrExitIdx == 0) {
      if (CanBeSMin(SE, RightSCEV)) {
        // TODO: this restriction is easily removable -- we just have to
        // remember that the icmp was an sgt and not an sge.
        FailureReason = "limit may overflow when coercing sge to sgt";
        return None;
      }

      IRBuilder<> B(&*Preheader->rbegin());
      RightValue = B.CreateSub(RightValue, One);
    }
  }

  const SCEV *StartNext = IndVarNext->getStart();
  const SCEV *Addend = SE.getNegativeSCEV(IndVarNext->getStepRecurrence(SE));
  const SCEV *IndVarStart = SE.getAddExpr(StartNext, Addend);

  BasicBlock *LatchExit = LatchBr->getSuccessor(LatchBrExitIdx);

  assert(SE.getLoopDisposition(LatchCount, &L) ==
             ScalarEvolution::LoopInvariant &&
         "loop variant exit count doesn't make sense!");

  assert(!L.contains(LatchExit) && "expected an exit block!");
  const DataLayout &DL = Preheader->getModule()->getDataLayout();
  Value *IndVarStartV =
      SCEVExpander(SE, DL, "irce")
          .expandCodeFor(IndVarStart, IndVarTy, &*Preheader->rbegin());
  IndVarStartV->setName("indvar.start");

  LoopStructure Result;

  Result.Tag = "main";
  Result.Header = Header;
  Result.Latch = Latch;
  Result.LatchBr = LatchBr;
  Result.LatchExit = LatchExit;
  Result.LatchBrExitIdx = LatchBrExitIdx;
  Result.IndVarStart = IndVarStartV;
  Result.IndVarNext = LeftValue;
  Result.IndVarIncreasing = IsIncreasing;
  Result.LoopExitAt = RightValue;

  FailureReason = nullptr;

  return Result;
}

Optional<LoopConstrainer::SubRanges>
LoopConstrainer::calculateSubRanges() const {
  IntegerType *Ty = cast<IntegerType>(LatchTakenCount->getType());

  if (Range.getType() != Ty)
    return None;

  LoopConstrainer::SubRanges Result;

  // I think we can be more aggressive here and make this nuw / nsw if the
  // addition that feeds into the icmp for the latch's terminating branch is nuw
  // / nsw.  In any case, a wrapping 2's complement addition is safe.
  ConstantInt *One = ConstantInt::get(Ty, 1);
  const SCEV *Start = SE.getSCEV(MainLoopStructure.IndVarStart);
  const SCEV *End = SE.getSCEV(MainLoopStructure.LoopExitAt);

  bool Increasing = MainLoopStructure.IndVarIncreasing;

  // We compute `Smallest` and `Greatest` such that [Smallest, Greatest) is the
  // range of values the induction variable takes.

  const SCEV *Smallest = nullptr, *Greatest = nullptr;

  if (Increasing) {
    Smallest = Start;
    Greatest = End;
  } else {
    // These two computations may sign-overflow.  Here is why that is okay:
    //
    // We know that the induction variable does not sign-overflow on any
    // iteration except the last one, and it starts at `Start` and ends at
    // `End`, decrementing by one every time.
    //
    //  * if `Smallest` sign-overflows we know `End` is `INT_SMAX`. Since the
    //    induction variable is decreasing we know that that the smallest value
    //    the loop body is actually executed with is `INT_SMIN` == `Smallest`.
    //
    //  * if `Greatest` sign-overflows, we know it can only be `INT_SMIN`.  In
    //    that case, `Clamp` will always return `Smallest` and
    //    [`Result.LowLimit`, `Result.HighLimit`) = [`Smallest`, `Smallest`)
    //    will be an empty range.  Returning an empty range is always safe.
    //

    Smallest = SE.getAddExpr(End, SE.getSCEV(One));
    Greatest = SE.getAddExpr(Start, SE.getSCEV(One));
  }

  auto Clamp = [this, Smallest, Greatest](const SCEV *S) {
    return SE.getSMaxExpr(Smallest, SE.getSMinExpr(Greatest, S));
  };

  // In some cases we can prove that we don't need a pre or post loop

  bool ProvablyNoPreloop =
      SE.isKnownPredicate(ICmpInst::ICMP_SLE, Range.getBegin(), Smallest);
  if (!ProvablyNoPreloop)
    Result.LowLimit = Clamp(Range.getBegin());

  bool ProvablyNoPostLoop =
      SE.isKnownPredicate(ICmpInst::ICMP_SLE, Greatest, Range.getEnd());
  if (!ProvablyNoPostLoop)
    Result.HighLimit = Clamp(Range.getEnd());

  return Result;
}

void LoopConstrainer::cloneLoop(LoopConstrainer::ClonedLoop &Result,
                                const char *Tag) const {
  for (BasicBlock *BB : OriginalLoop.getBlocks()) {
    BasicBlock *Clone = CloneBasicBlock(BB, Result.Map, Twine(".") + Tag, &F);
    Result.Blocks.push_back(Clone);
    Result.Map[BB] = Clone;
  }

  auto GetClonedValue = [&Result](Value *V) {
    assert(V && "null values not in domain!");
    auto It = Result.Map.find(V);
    if (It == Result.Map.end())
      return V;
    return static_cast<Value *>(It->second);
  };

  Result.Structure = MainLoopStructure.map(GetClonedValue);
  Result.Structure.Tag = Tag;

  for (unsigned i = 0, e = Result.Blocks.size(); i != e; ++i) {
    BasicBlock *ClonedBB = Result.Blocks[i];
    BasicBlock *OriginalBB = OriginalLoop.getBlocks()[i];

    assert(Result.Map[OriginalBB] == ClonedBB && "invariant!");

    for (Instruction &I : *ClonedBB)
      RemapInstruction(&I, Result.Map,
                       RF_NoModuleLevelChanges | RF_IgnoreMissingEntries);

    // Exit blocks will now have one more predecessor and their PHI nodes need
    // to be edited to reflect that.  No phi nodes need to be introduced because
    // the loop is in LCSSA.

    for (auto SBBI = succ_begin(OriginalBB), SBBE = succ_end(OriginalBB);
         SBBI != SBBE; ++SBBI) {

      if (OriginalLoop.contains(*SBBI))
        continue; // not an exit block

      for (Instruction &I : **SBBI) {
        if (!isa<PHINode>(&I))
          break;

        PHINode *PN = cast<PHINode>(&I);
        Value *OldIncoming = PN->getIncomingValueForBlock(OriginalBB);
        PN->addIncoming(GetClonedValue(OldIncoming), ClonedBB);
      }
    }
  }
}

LoopConstrainer::RewrittenRangeInfo LoopConstrainer::changeIterationSpaceEnd(
    const LoopStructure &LS, BasicBlock *Preheader, Value *ExitSubloopAt,
    BasicBlock *ContinuationBlock) const {

  // We start with a loop with a single latch:
  //
  //    +--------------------+
  //    |                    |
  //    |     preheader      |
  //    |                    |
  //    +--------+-----------+
  //             |      ----------------\
  //             |     /                |
  //    +--------v----v------+          |
  //    |                    |          |
  //    |      header        |          |
  //    |                    |          |
  //    +--------------------+          |
  //                                    |
  //            .....                   |
  //                                    |
  //    +--------------------+          |
  //    |                    |          |
  //    |       latch        >----------/
  //    |                    |
  //    +-------v------------+
  //            |
  //            |
  //            |   +--------------------+
  //            |   |                    |
  //            +--->   original exit    |
  //                |                    |
  //                +--------------------+
  //
  // We change the control flow to look like
  //
  //
  //    +--------------------+
  //    |                    |
  //    |     preheader      >-------------------------+
  //    |                    |                         |
  //    +--------v-----------+                         |
  //             |    /-------------+                  |
  //             |   /              |                  |
  //    +--------v--v--------+      |                  |
  //    |                    |      |                  |
  //    |      header        |      |   +--------+     |
  //    |                    |      |   |        |     |
  //    +--------------------+      |   |  +-----v-----v-----------+
  //                                |   |  |                       |
  //                                |   |  |     .pseudo.exit      |
  //                                |   |  |                       |
  //                                |   |  +-----------v-----------+
  //                                |   |              |
  //            .....               |   |              |
  //                                |   |     +--------v-------------+
  //    +--------------------+      |   |     |                      |
  //    |                    |      |   |     |   ContinuationBlock  |
  //    |       latch        >------+   |     |                      |
  //    |                    |          |     +----------------------+
  //    +---------v----------+          |
  //              |                     |
  //              |                     |
  //              |     +---------------^-----+
  //              |     |                     |
  //              +----->    .exit.selector   |
  //                    |                     |
  //                    +----------v----------+
  //                               |
  //     +--------------------+    |
  //     |                    |    |
  //     |   original exit    <----+
  //     |                    |
  //     +--------------------+
  //

  RewrittenRangeInfo RRI;

  auto BBInsertLocation = std::next(Function::iterator(LS.Latch));
  RRI.ExitSelector = BasicBlock::Create(Ctx, Twine(LS.Tag) + ".exit.selector",
                                        &F, BBInsertLocation);
  RRI.PseudoExit = BasicBlock::Create(Ctx, Twine(LS.Tag) + ".pseudo.exit", &F,
                                      BBInsertLocation);

  BranchInst *PreheaderJump = cast<BranchInst>(&*Preheader->rbegin());
  bool Increasing = LS.IndVarIncreasing;

  IRBuilder<> B(PreheaderJump);

  // EnterLoopCond - is it okay to start executing this `LS'?
  Value *EnterLoopCond = Increasing
                             ? B.CreateICmpSLT(LS.IndVarStart, ExitSubloopAt)
                             : B.CreateICmpSGT(LS.IndVarStart, ExitSubloopAt);

  B.CreateCondBr(EnterLoopCond, LS.Header, RRI.PseudoExit);
  PreheaderJump->eraseFromParent();

  LS.LatchBr->setSuccessor(LS.LatchBrExitIdx, RRI.ExitSelector);
  B.SetInsertPoint(LS.LatchBr);
  Value *TakeBackedgeLoopCond =
      Increasing ? B.CreateICmpSLT(LS.IndVarNext, ExitSubloopAt)
                 : B.CreateICmpSGT(LS.IndVarNext, ExitSubloopAt);
  Value *CondForBranch = LS.LatchBrExitIdx == 1
                             ? TakeBackedgeLoopCond
                             : B.CreateNot(TakeBackedgeLoopCond);

  LS.LatchBr->setCondition(CondForBranch);

  B.SetInsertPoint(RRI.ExitSelector);

  // IterationsLeft - are there any more iterations left, given the original
  // upper bound on the induction variable?  If not, we branch to the "real"
  // exit.
  Value *IterationsLeft = Increasing
                              ? B.CreateICmpSLT(LS.IndVarNext, LS.LoopExitAt)
                              : B.CreateICmpSGT(LS.IndVarNext, LS.LoopExitAt);
  B.CreateCondBr(IterationsLeft, RRI.PseudoExit, LS.LatchExit);

  BranchInst *BranchToContinuation =
      BranchInst::Create(ContinuationBlock, RRI.PseudoExit);

  // We emit PHI nodes into `RRI.PseudoExit' that compute the "latest" value of
  // each of the PHI nodes in the loop header.  This feeds into the initial
  // value of the same PHI nodes if/when we continue execution.
  for (Instruction &I : *LS.Header) {
    if (!isa<PHINode>(&I))
      break;

    PHINode *PN = cast<PHINode>(&I);

    PHINode *NewPHI = PHINode::Create(PN->getType(), 2, PN->getName() + ".copy",
                                      BranchToContinuation);

    NewPHI->addIncoming(PN->getIncomingValueForBlock(Preheader), Preheader);
    NewPHI->addIncoming(PN->getIncomingValueForBlock(LS.Latch),
                        RRI.ExitSelector);
    RRI.PHIValuesAtPseudoExit.push_back(NewPHI);
  }

  RRI.IndVarEnd = PHINode::Create(LS.IndVarNext->getType(), 2, "indvar.end",
                                  BranchToContinuation);
  RRI.IndVarEnd->addIncoming(LS.IndVarStart, Preheader);
  RRI.IndVarEnd->addIncoming(LS.IndVarNext, RRI.ExitSelector);

  // The latch exit now has a branch from `RRI.ExitSelector' instead of
  // `LS.Latch'.  The PHI nodes need to be updated to reflect that.
  for (Instruction &I : *LS.LatchExit) {
    if (PHINode *PN = dyn_cast<PHINode>(&I))
      replacePHIBlock(PN, LS.Latch, RRI.ExitSelector);
    else
      break;
  }

  return RRI;
}

void LoopConstrainer::rewriteIncomingValuesForPHIs(
    LoopStructure &LS, BasicBlock *ContinuationBlock,
    const LoopConstrainer::RewrittenRangeInfo &RRI) const {

  unsigned PHIIndex = 0;
  for (Instruction &I : *LS.Header) {
    if (!isa<PHINode>(&I))
      break;

    PHINode *PN = cast<PHINode>(&I);

    for (unsigned i = 0, e = PN->getNumIncomingValues(); i < e; ++i)
      if (PN->getIncomingBlock(i) == ContinuationBlock)
        PN->setIncomingValue(i, RRI.PHIValuesAtPseudoExit[PHIIndex++]);
  }

  LS.IndVarStart = RRI.IndVarEnd;
}

BasicBlock *LoopConstrainer::createPreheader(const LoopStructure &LS,
                                             BasicBlock *OldPreheader,
                                             const char *Tag) const {

  BasicBlock *Preheader = BasicBlock::Create(Ctx, Tag, &F, LS.Header);
  BranchInst::Create(LS.Header, Preheader);

  for (Instruction &I : *LS.Header) {
    if (!isa<PHINode>(&I))
      break;

    PHINode *PN = cast<PHINode>(&I);
    for (unsigned i = 0, e = PN->getNumIncomingValues(); i < e; ++i)
      replacePHIBlock(PN, OldPreheader, Preheader);
  }

  return Preheader;
}

void LoopConstrainer::addToParentLoopIfNeeded(ArrayRef<BasicBlock *> BBs) {
  Loop *ParentLoop = OriginalLoop.getParentLoop();
  if (!ParentLoop)
    return;

  for (BasicBlock *BB : BBs)
    ParentLoop->addBasicBlockToLoop(BB, OriginalLoopInfo);
}

bool LoopConstrainer::run() {
  BasicBlock *Preheader = nullptr;
  LatchTakenCount = SE.getExitCount(&OriginalLoop, MainLoopStructure.Latch);
  Preheader = OriginalLoop.getLoopPreheader();
  assert(!isa<SCEVCouldNotCompute>(LatchTakenCount) && Preheader != nullptr &&
         "preconditions!");

  OriginalPreheader = Preheader;
  MainLoopPreheader = Preheader;

  Optional<SubRanges> MaybeSR = calculateSubRanges();
  if (!MaybeSR.hasValue()) {
    DEBUG(dbgs() << "irce: could not compute subranges\n");
    return false;
  }

  SubRanges SR = MaybeSR.getValue();
  bool Increasing = MainLoopStructure.IndVarIncreasing;
  IntegerType *IVTy =
      cast<IntegerType>(MainLoopStructure.IndVarNext->getType());

  SCEVExpander Expander(SE, F.getParent()->getDataLayout(), "irce");
  Instruction *InsertPt = OriginalPreheader->getTerminator();

  // It would have been better to make `PreLoop' and `PostLoop'
  // `Optional<ClonedLoop>'s, but `ValueToValueMapTy' does not have a copy
  // constructor.
  ClonedLoop PreLoop, PostLoop;
  bool NeedsPreLoop =
      Increasing ? SR.LowLimit.hasValue() : SR.HighLimit.hasValue();
  bool NeedsPostLoop =
      Increasing ? SR.HighLimit.hasValue() : SR.LowLimit.hasValue();

  Value *ExitPreLoopAt = nullptr;
  Value *ExitMainLoopAt = nullptr;
  const SCEVConstant *MinusOneS =
      cast<SCEVConstant>(SE.getConstant(IVTy, -1, true /* isSigned */));

  if (NeedsPreLoop) {
    const SCEV *ExitPreLoopAtSCEV = nullptr;

    if (Increasing)
      ExitPreLoopAtSCEV = *SR.LowLimit;
    else {
      if (CanBeSMin(SE, *SR.HighLimit)) {
        DEBUG(dbgs() << "irce: could not prove no-overflow when computing "
                     << "preloop exit limit.  HighLimit = " << *(*SR.HighLimit)
                     << "\n");
        return false;
      }
      ExitPreLoopAtSCEV = SE.getAddExpr(*SR.HighLimit, MinusOneS);
    }

    ExitPreLoopAt = Expander.expandCodeFor(ExitPreLoopAtSCEV, IVTy, InsertPt);
    ExitPreLoopAt->setName("exit.preloop.at");
  }

  if (NeedsPostLoop) {
    const SCEV *ExitMainLoopAtSCEV = nullptr;

    if (Increasing)
      ExitMainLoopAtSCEV = *SR.HighLimit;
    else {
      if (CanBeSMin(SE, *SR.LowLimit)) {
        DEBUG(dbgs() << "irce: could not prove no-overflow when computing "
                     << "mainloop exit limit.  LowLimit = " << *(*SR.LowLimit)
                     << "\n");
        return false;
      }
      ExitMainLoopAtSCEV = SE.getAddExpr(*SR.LowLimit, MinusOneS);
    }

    ExitMainLoopAt = Expander.expandCodeFor(ExitMainLoopAtSCEV, IVTy, InsertPt);
    ExitMainLoopAt->setName("exit.mainloop.at");
  }

  // We clone these ahead of time so that we don't have to deal with changing
  // and temporarily invalid IR as we transform the loops.
  if (NeedsPreLoop)
    cloneLoop(PreLoop, "preloop");
  if (NeedsPostLoop)
    cloneLoop(PostLoop, "postloop");

  RewrittenRangeInfo PreLoopRRI;

  if (NeedsPreLoop) {
    Preheader->getTerminator()->replaceUsesOfWith(MainLoopStructure.Header,
                                                  PreLoop.Structure.Header);

    MainLoopPreheader =
        createPreheader(MainLoopStructure, Preheader, "mainloop");
    PreLoopRRI = changeIterationSpaceEnd(PreLoop.Structure, Preheader,
                                         ExitPreLoopAt, MainLoopPreheader);
    rewriteIncomingValuesForPHIs(MainLoopStructure, MainLoopPreheader,
                                 PreLoopRRI);
  }

  BasicBlock *PostLoopPreheader = nullptr;
  RewrittenRangeInfo PostLoopRRI;

  if (NeedsPostLoop) {
    PostLoopPreheader =
        createPreheader(PostLoop.Structure, Preheader, "postloop");
    PostLoopRRI = changeIterationSpaceEnd(MainLoopStructure, MainLoopPreheader,
                                          ExitMainLoopAt, PostLoopPreheader);
    rewriteIncomingValuesForPHIs(PostLoop.Structure, PostLoopPreheader,
                                 PostLoopRRI);
  }

  BasicBlock *NewMainLoopPreheader =
      MainLoopPreheader != Preheader ? MainLoopPreheader : nullptr;
  BasicBlock *NewBlocks[] = {PostLoopPreheader,        PreLoopRRI.PseudoExit,
                             PreLoopRRI.ExitSelector,  PostLoopRRI.PseudoExit,
                             PostLoopRRI.ExitSelector, NewMainLoopPreheader};

  // Some of the above may be nullptr, filter them out before passing to
  // addToParentLoopIfNeeded.
  auto NewBlocksEnd =
      std::remove(std::begin(NewBlocks), std::end(NewBlocks), nullptr);

  addToParentLoopIfNeeded(makeArrayRef(std::begin(NewBlocks), NewBlocksEnd));
  addToParentLoopIfNeeded(PreLoop.Blocks);
  addToParentLoopIfNeeded(PostLoop.Blocks);

  return true;
}

/// Computes and returns a range of values for the induction variable (IndVar)
/// in which the range check can be safely elided.  If it cannot compute such a
/// range, returns None.
Optional<InductiveRangeCheck::Range>
InductiveRangeCheck::computeSafeIterationSpace(ScalarEvolution &SE,
                                               const SCEVAddRecExpr *IndVar,
                                               IRBuilder<> &) const {
  // IndVar is of the form "A + B * I" (where "I" is the canonical induction
  // variable, that may or may not exist as a real llvm::Value in the loop) and
  // this inductive range check is a range check on the "C + D * I" ("C" is
  // getOffset() and "D" is getScale()).  We rewrite the value being range
  // checked to "M + N * IndVar" where "N" = "D * B^(-1)" and "M" = "C - NA".
  // Currently we support this only for "B" = "D" = { 1 or -1 }, but the code
  // can be generalized as needed.
  //
  // The actual inequalities we solve are of the form
  //
  //   0 <= M + 1 * IndVar < L given L >= 0  (i.e. N == 1)
  //
  // The inequality is satisfied by -M <= IndVar < (L - M) [^1].  All additions
  // and subtractions are twos-complement wrapping and comparisons are signed.
  //
  // Proof:
  //
  //   If there exists IndVar such that -M <= IndVar < (L - M) then it follows
  //   that -M <= (-M + L) [== Eq. 1].  Since L >= 0, if (-M + L) sign-overflows
  //   then (-M + L) < (-M).  Hence by [Eq. 1], (-M + L) could not have
  //   overflown.
  //
  //   This means IndVar = t + (-M) for t in [0, L).  Hence (IndVar + M) = t.
  //   Hence 0 <= (IndVar + M) < L

  // [^1]: Note that the solution does _not_ apply if L < 0; consider values M =
  // 127, IndVar = 126 and L = -2 in an i8 world.

  if (!IndVar->isAffine())
    return None;

  const SCEV *A = IndVar->getStart();
  const SCEVConstant *B = dyn_cast<SCEVConstant>(IndVar->getStepRecurrence(SE));
  if (!B)
    return None;

  const SCEV *C = getOffset();
  const SCEVConstant *D = dyn_cast<SCEVConstant>(getScale());
  if (D != B)
    return None;

  ConstantInt *ConstD = D->getValue();
  if (!(ConstD->isMinusOne() || ConstD->isOne()))
    return None;

  const SCEV *M = SE.getMinusSCEV(C, A);

  const SCEV *Begin = SE.getNegativeSCEV(M);
  const SCEV *UpperLimit = nullptr;

  // We strengthen "0 <= I" to "0 <= I < INT_SMAX" and "I < L" to "0 <= I < L".
  // We can potentially do much better here.
  if (Value *V = getLength()) {
    UpperLimit = SE.getSCEV(V);
  } else {
    assert(Kind == InductiveRangeCheck::RANGE_CHECK_LOWER && "invariant!");
    unsigned BitWidth = cast<IntegerType>(IndVar->getType())->getBitWidth();
    UpperLimit = SE.getConstant(APInt::getSignedMaxValue(BitWidth));
  }

  const SCEV *End = SE.getMinusSCEV(UpperLimit, M);
  return InductiveRangeCheck::Range(Begin, End);
}

static Optional<InductiveRangeCheck::Range>
IntersectRange(ScalarEvolution &SE,
               const Optional<InductiveRangeCheck::Range> &R1,
               const InductiveRangeCheck::Range &R2, IRBuilder<> &B) {
  if (!R1.hasValue())
    return R2;
  auto &R1Value = R1.getValue();

  // TODO: we could widen the smaller range and have this work; but for now we
  // bail out to keep things simple.
  if (R1Value.getType() != R2.getType())
    return None;

  const SCEV *NewBegin = SE.getSMaxExpr(R1Value.getBegin(), R2.getBegin());
  const SCEV *NewEnd = SE.getSMinExpr(R1Value.getEnd(), R2.getEnd());

  return InductiveRangeCheck::Range(NewBegin, NewEnd);
}

bool InductiveRangeCheckElimination::runOnLoop(Loop *L, LPPassManager &LPM) {
  if (L->getBlocks().size() >= LoopSizeCutoff) {
    DEBUG(dbgs() << "irce: giving up constraining loop, too large\n";);
    return false;
  }

  BasicBlock *Preheader = L->getLoopPreheader();
  if (!Preheader) {
    DEBUG(dbgs() << "irce: loop has no preheader, leaving\n");
    return false;
  }

  LLVMContext &Context = Preheader->getContext();
  InductiveRangeCheck::AllocatorTy IRCAlloc;
  SmallVector<InductiveRangeCheck *, 16> RangeChecks;
  ScalarEvolution &SE = getAnalysis<ScalarEvolution>();
  BranchProbabilityInfo &BPI = getAnalysis<BranchProbabilityInfo>();

  for (auto BBI : L->getBlocks())
    if (BranchInst *TBI = dyn_cast<BranchInst>(BBI->getTerminator()))
      if (InductiveRangeCheck *IRC =
          InductiveRangeCheck::create(IRCAlloc, TBI, L, SE, BPI))
        RangeChecks.push_back(IRC);

  if (RangeChecks.empty())
    return false;

  auto PrintRecognizedRangeChecks = [&](raw_ostream &OS) {
    OS << "irce: looking at loop "; L->print(OS);
    OS << "irce: loop has " << RangeChecks.size()
       << " inductive range checks: \n";
    for (InductiveRangeCheck *IRC : RangeChecks)
      IRC->print(OS);
  };

  DEBUG(PrintRecognizedRangeChecks(dbgs()));

  if (PrintRangeChecks)
    PrintRecognizedRangeChecks(errs());

  const char *FailureReason = nullptr;
  Optional<LoopStructure> MaybeLoopStructure =
      LoopStructure::parseLoopStructure(SE, BPI, *L, FailureReason);
  if (!MaybeLoopStructure.hasValue()) {
    DEBUG(dbgs() << "irce: could not parse loop structure: " << FailureReason
                 << "\n";);
    return false;
  }
  LoopStructure LS = MaybeLoopStructure.getValue();
  bool Increasing = LS.IndVarIncreasing;
  const SCEV *MinusOne =
      SE.getConstant(LS.IndVarNext->getType(), Increasing ? -1 : 1, true);
  const SCEVAddRecExpr *IndVar =
      cast<SCEVAddRecExpr>(SE.getAddExpr(SE.getSCEV(LS.IndVarNext), MinusOne));

  Optional<InductiveRangeCheck::Range> SafeIterRange;
  Instruction *ExprInsertPt = Preheader->getTerminator();

  SmallVector<InductiveRangeCheck *, 4> RangeChecksToEliminate;

  IRBuilder<> B(ExprInsertPt);
  for (InductiveRangeCheck *IRC : RangeChecks) {
    auto Result = IRC->computeSafeIterationSpace(SE, IndVar, B);
    if (Result.hasValue()) {
      auto MaybeSafeIterRange =
        IntersectRange(SE, SafeIterRange, Result.getValue(), B);
      if (MaybeSafeIterRange.hasValue()) {
        RangeChecksToEliminate.push_back(IRC);
        SafeIterRange = MaybeSafeIterRange.getValue();
      }
    }
  }

  if (!SafeIterRange.hasValue())
    return false;

  LoopConstrainer LC(*L, getAnalysis<LoopInfoWrapperPass>().getLoopInfo(), LS,
                     SE, SafeIterRange.getValue());
  bool Changed = LC.run();

  if (Changed) {
    auto PrintConstrainedLoopInfo = [L]() {
      dbgs() << "irce: in function ";
      dbgs() << L->getHeader()->getParent()->getName() << ": ";
      dbgs() << "constrained ";
      L->print(dbgs());
    };

    DEBUG(PrintConstrainedLoopInfo());

    if (PrintChangedLoops)
      PrintConstrainedLoopInfo();

    // Optimize away the now-redundant range checks.

    for (InductiveRangeCheck *IRC : RangeChecksToEliminate) {
      ConstantInt *FoldedRangeCheck = IRC->getPassingDirection()
                                          ? ConstantInt::getTrue(Context)
                                          : ConstantInt::getFalse(Context);
      IRC->getBranch()->setCondition(FoldedRangeCheck);
    }
  }

  return Changed;
}

Pass *llvm::createInductiveRangeCheckEliminationPass() {
  return new InductiveRangeCheckElimination;
}
