//===- PlaceSafepoints.cpp - Place GC Safepoints --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Place garbage collection safepoints at appropriate locations in the IR. This
// does not make relocation semantics or variable liveness explicit.  That's
// done by RewriteStatepointsForGC.
//
// Terminology:
// - A call is said to be "parseable" if there is a stack map generated for the
// return PC of the call.  A runtime can determine where values listed in the
// deopt arguments and (after RewriteStatepointsForGC) gc arguments are located
// on the stack when the code is suspended inside such a call.  Every parse
// point is represented by a call wrapped in an gc.statepoint intrinsic.
// - A "poll" is an explicit check in the generated code to determine if the
// runtime needs the generated code to cooperate by calling a helper routine
// and thus suspending its execution at a known state. The call to the helper
// routine will be parseable.  The (gc & runtime specific) logic of a poll is
// assumed to be provided in a function of the name "gc.safepoint_poll".
//
// We aim to insert polls such that running code can quickly be brought to a
// well defined state for inspection by the collector.  In the current
// implementation, this is done via the insertion of poll sites at method entry
// and the backedge of most loops.  We try to avoid inserting more polls than
// are neccessary to ensure a finite period between poll sites.  This is not
// because the poll itself is expensive in the generated code; it's not.  Polls
// do tend to impact the optimizer itself in negative ways; we'd like to avoid
// perturbing the optimization of the method as much as we can.
//
// We also need to make most call sites parseable.  The callee might execute a
// poll (or otherwise be inspected by the GC).  If so, the entire stack
// (including the suspended frame of the current method) must be parseable.
//
// This pass will insert:
// - Call parse points ("call safepoints") for any call which may need to
// reach a safepoint during the execution of the callee function.
// - Backedge safepoint polls and entry safepoint polls to ensure that
// executing code reaches a safepoint poll in a finite amount of time.
//
// We do not currently support return statepoints, but adding them would not
// be hard.  They are not required for correctness - entry safepoints are an
// alternative - but some GCs may prefer them.  Patches welcome.
//
//===----------------------------------------------------------------------===//

#include "llvm/Pass.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Statepoint.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"

#define DEBUG_TYPE "safepoint-placement"
STATISTIC(NumEntrySafepoints, "Number of entry safepoints inserted");
STATISTIC(NumCallSafepoints, "Number of call safepoints inserted");
STATISTIC(NumBackedgeSafepoints, "Number of backedge safepoints inserted");

STATISTIC(CallInLoop, "Number of loops w/o safepoints due to calls in loop");
STATISTIC(FiniteExecution, "Number of loops w/o safepoints finite execution");

using namespace llvm;

// Ignore oppurtunities to avoid placing safepoints on backedges, useful for
// validation
static cl::opt<bool> AllBackedges("spp-all-backedges", cl::Hidden,
                                  cl::init(false));

/// If true, do not place backedge safepoints in counted loops.
static cl::opt<bool> SkipCounted("spp-counted", cl::Hidden, cl::init(true));

// If true, split the backedge of a loop when placing the safepoint, otherwise
// split the latch block itself.  Both are useful to support for
// experimentation, but in practice, it looks like splitting the backedge
// optimizes better.
static cl::opt<bool> SplitBackedge("spp-split-backedge", cl::Hidden,
                                   cl::init(false));

// Print tracing output
static cl::opt<bool> TraceLSP("spp-trace", cl::Hidden, cl::init(false));

namespace {

/// An analysis pass whose purpose is to identify each of the backedges in
/// the function which require a safepoint poll to be inserted.
struct PlaceBackedgeSafepointsImpl : public FunctionPass {
  static char ID;

  /// The output of the pass - gives a list of each backedge (described by
  /// pointing at the branch) which need a poll inserted.
  std::vector<TerminatorInst *> PollLocations;

  /// True unless we're running spp-no-calls in which case we need to disable
  /// the call dependend placement opts.
  bool CallSafepointsEnabled;

  ScalarEvolution *SE = nullptr;
  DominatorTree *DT = nullptr;
  LoopInfo *LI = nullptr;

  PlaceBackedgeSafepointsImpl(bool CallSafepoints = false)
      : FunctionPass(ID), CallSafepointsEnabled(CallSafepoints) {
    initializePlaceBackedgeSafepointsImplPass(*PassRegistry::getPassRegistry());
  }

  bool runOnLoop(Loop *);
  void runOnLoopAndSubLoops(Loop *L) {
    // Visit all the subloops
    for (auto I = L->begin(), E = L->end(); I != E; I++)
      runOnLoopAndSubLoops(*I);
    runOnLoop(L);
  }

  bool runOnFunction(Function &F) override {
    SE = &getAnalysis<ScalarEvolution>();
    DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    for (auto I = LI->begin(), E = LI->end(); I != E; I++) {
      runOnLoopAndSubLoops(*I);
    }
    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<ScalarEvolution>();
    AU.addRequired<LoopInfoWrapperPass>();
    // We no longer modify the IR at all in this pass.  Thus all
    // analysis are preserved.
    AU.setPreservesAll();
  }
};
}

static cl::opt<bool> NoEntry("spp-no-entry", cl::Hidden, cl::init(false));
static cl::opt<bool> NoCall("spp-no-call", cl::Hidden, cl::init(false));
static cl::opt<bool> NoBackedge("spp-no-backedge", cl::Hidden, cl::init(false));

namespace {
struct PlaceSafepoints : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid

  PlaceSafepoints() : FunctionPass(ID) {
    initializePlaceSafepointsPass(*PassRegistry::getPassRegistry());
  }
  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    // We modify the graph wholesale (inlining, block insertion, etc).  We
    // preserve nothing at the moment.  We could potentially preserve dom tree
    // if that was worth doing
  }
};
}

// Insert a safepoint poll immediately before the given instruction.  Does
// not handle the parsability of state at the runtime call, that's the
// callers job.
static void
InsertSafepointPoll(Instruction *InsertBefore,
                    std::vector<CallSite> &ParsePointsNeeded /*rval*/);

static bool isGCLeafFunction(const CallSite &CS);

static bool needsStatepoint(const CallSite &CS) {
  if (isGCLeafFunction(CS))
    return false;
  if (CS.isCall()) {
    CallInst *call = cast<CallInst>(CS.getInstruction());
    if (call->isInlineAsm())
      return false;
  }
  if (isStatepoint(CS) || isGCRelocate(CS) || isGCResult(CS)) {
    return false;
  }
  return true;
}

static Value *ReplaceWithStatepoint(const CallSite &CS, Pass *P);

/// Returns true if this loop is known to contain a call safepoint which
/// must unconditionally execute on any iteration of the loop which returns
/// to the loop header via an edge from Pred.  Returns a conservative correct
/// answer; i.e. false is always valid.
static bool containsUnconditionalCallSafepoint(Loop *L, BasicBlock *Header,
                                               BasicBlock *Pred,
                                               DominatorTree &DT) {
  // In general, we're looking for any cut of the graph which ensures
  // there's a call safepoint along every edge between Header and Pred.
  // For the moment, we look only for the 'cuts' that consist of a single call
  // instruction in a block which is dominated by the Header and dominates the
  // loop latch (Pred) block.  Somewhat surprisingly, walking the entire chain
  // of such dominating blocks gets substaintially more occurences than just
  // checking the Pred and Header blocks themselves.  This may be due to the
  // density of loop exit conditions caused by range and null checks.
  // TODO: structure this as an analysis pass, cache the result for subloops,
  // avoid dom tree recalculations
  assert(DT.dominates(Header, Pred) && "loop latch not dominated by header?");

  BasicBlock *Current = Pred;
  while (true) {
    for (Instruction &I : *Current) {
      if (auto CS = CallSite(&I))
        // Note: Technically, needing a safepoint isn't quite the right
        // condition here.  We should instead be checking if the target method
        // has an
        // unconditional poll. In practice, this is only a theoretical concern
        // since we don't have any methods with conditional-only safepoint
        // polls.
        if (needsStatepoint(CS))
          return true;
    }

    if (Current == Header)
      break;
    Current = DT.getNode(Current)->getIDom()->getBlock();
  }

  return false;
}

/// Returns true if this loop is known to terminate in a finite number of
/// iterations.  Note that this function may return false for a loop which
/// does actual terminate in a finite constant number of iterations due to
/// conservatism in the analysis.
static bool mustBeFiniteCountedLoop(Loop *L, ScalarEvolution *SE,
                                    BasicBlock *Pred) {
  // Only used when SkipCounted is off
  const unsigned upperTripBound = 8192;

  // A conservative bound on the loop as a whole.
  const SCEV *MaxTrips = SE->getMaxBackedgeTakenCount(L);
  if (MaxTrips != SE->getCouldNotCompute()) {
    if (SE->getUnsignedRange(MaxTrips).getUnsignedMax().ult(upperTripBound))
      return true;
    if (SkipCounted &&
        SE->getUnsignedRange(MaxTrips).getUnsignedMax().isIntN(32))
      return true;
  }

  // If this is a conditional branch to the header with the alternate path
  // being outside the loop, we can ask questions about the execution frequency
  // of the exit block.
  if (L->isLoopExiting(Pred)) {
    // This returns an exact expression only.  TODO: We really only need an
    // upper bound here, but SE doesn't expose that.
    const SCEV *MaxExec = SE->getExitCount(L, Pred);
    if (MaxExec != SE->getCouldNotCompute()) {
      if (SE->getUnsignedRange(MaxExec).getUnsignedMax().ult(upperTripBound))
        return true;
      if (SkipCounted &&
          SE->getUnsignedRange(MaxExec).getUnsignedMax().isIntN(32))
        return true;
    }
  }

  return /* not finite */ false;
}

static void scanOneBB(Instruction *start, Instruction *end,
                      std::vector<CallInst *> &calls,
                      std::set<BasicBlock *> &seen,
                      std::vector<BasicBlock *> &worklist) {
  for (BasicBlock::iterator itr(start);
       itr != start->getParent()->end() && itr != BasicBlock::iterator(end);
       itr++) {
    if (CallInst *CI = dyn_cast<CallInst>(&*itr)) {
      calls.push_back(CI);
    }
    // FIXME: This code does not handle invokes
    assert(!dyn_cast<InvokeInst>(&*itr) &&
           "support for invokes in poll code needed");
    // Only add the successor blocks if we reach the terminator instruction
    // without encountering end first
    if (itr->isTerminator()) {
      BasicBlock *BB = itr->getParent();
      for (BasicBlock *Succ : successors(BB)) {
        if (seen.count(Succ) == 0) {
          worklist.push_back(Succ);
          seen.insert(Succ);
        }
      }
    }
  }
}
static void scanInlinedCode(Instruction *start, Instruction *end,
                            std::vector<CallInst *> &calls,
                            std::set<BasicBlock *> &seen) {
  calls.clear();
  std::vector<BasicBlock *> worklist;
  seen.insert(start->getParent());
  scanOneBB(start, end, calls, seen, worklist);
  while (!worklist.empty()) {
    BasicBlock *BB = worklist.back();
    worklist.pop_back();
    scanOneBB(&*BB->begin(), end, calls, seen, worklist);
  }
}

bool PlaceBackedgeSafepointsImpl::runOnLoop(Loop *L) {
  // Loop through all loop latches (branches controlling backedges).  We need
  // to place a safepoint on every backedge (potentially).
  // Note: In common usage, there will be only one edge due to LoopSimplify
  // having run sometime earlier in the pipeline, but this code must be correct
  // w.r.t. loops with multiple backedges.
  BasicBlock *header = L->getHeader();
  SmallVector<BasicBlock*, 16> LoopLatches;
  L->getLoopLatches(LoopLatches);
  for (BasicBlock *pred : LoopLatches) {
    assert(L->contains(pred));

    // Make a policy decision about whether this loop needs a safepoint or
    // not.  Note that this is about unburdening the optimizer in loops, not
    // avoiding the runtime cost of the actual safepoint.
    if (!AllBackedges) {
      if (mustBeFiniteCountedLoop(L, SE, pred)) {
        if (TraceLSP)
          errs() << "skipping safepoint placement in finite loop\n";
        FiniteExecution++;
        continue;
      }
      if (CallSafepointsEnabled &&
          containsUnconditionalCallSafepoint(L, header, pred, *DT)) {
        // Note: This is only semantically legal since we won't do any further
        // IPO or inlining before the actual call insertion..  If we hadn't, we
        // might latter loose this call safepoint.
        if (TraceLSP)
          errs() << "skipping safepoint placement due to unconditional call\n";
        CallInLoop++;
        continue;
      }
    }

    // TODO: We can create an inner loop which runs a finite number of
    // iterations with an outer loop which contains a safepoint.  This would
    // not help runtime performance that much, but it might help our ability to
    // optimize the inner loop.

    // Safepoint insertion would involve creating a new basic block (as the
    // target of the current backedge) which does the safepoint (of all live
    // variables) and branches to the true header
    TerminatorInst *term = pred->getTerminator();

    if (TraceLSP) {
      errs() << "[LSP] terminator instruction: ";
      term->dump();
    }

    PollLocations.push_back(term);
  }

  return false;
}

/// Returns true if an entry safepoint is not required before this callsite in
/// the caller function.
static bool doesNotRequireEntrySafepointBefore(const CallSite &CS) {
  Instruction *Inst = CS.getInstruction();
  if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(Inst)) {
    switch (II->getIntrinsicID()) {
    case Intrinsic::experimental_gc_statepoint:
    case Intrinsic::experimental_patchpoint_void:
    case Intrinsic::experimental_patchpoint_i64:
      // The can wrap an actual call which may grow the stack by an unbounded
      // amount or run forever.
      return false;
    default:
      // Most LLVM intrinsics are things which do not expand to actual calls, or
      // at least if they do, are leaf functions that cause only finite stack
      // growth.  In particular, the optimizer likes to form things like memsets
      // out of stores in the original IR.  Another important example is
      // llvm.localescape which must occur in the entry block.  Inserting a
      // safepoint before it is not legal since it could push the localescape
      // out of the entry block.
      return true;
    }
  }
  return false;
}

static Instruction *findLocationForEntrySafepoint(Function &F,
                                                  DominatorTree &DT) {

  // Conceptually, this poll needs to be on method entry, but in
  // practice, we place it as late in the entry block as possible.  We
  // can place it as late as we want as long as it dominates all calls
  // that can grow the stack.  This, combined with backedge polls,
  // give us all the progress guarantees we need.

  // hasNextInstruction and nextInstruction are used to iterate
  // through a "straight line" execution sequence.

  auto hasNextInstruction = [](Instruction *I) {
    if (!I->isTerminator()) {
      return true;
    }
    BasicBlock *nextBB = I->getParent()->getUniqueSuccessor();
    return nextBB && (nextBB->getUniquePredecessor() != nullptr);
  };

  auto nextInstruction = [&hasNextInstruction](Instruction *I) {
    assert(hasNextInstruction(I) &&
           "first check if there is a next instruction!");
    (void)hasNextInstruction; // HLSL Change - unused var
    if (I->isTerminator()) {
      return I->getParent()->getUniqueSuccessor()->begin();
    } else {
      return std::next(BasicBlock::iterator(I));
    }
  };

  Instruction *cursor = nullptr;
  for (cursor = F.getEntryBlock().begin(); hasNextInstruction(cursor);
       cursor = nextInstruction(cursor)) {

    // We need to ensure a safepoint poll occurs before any 'real' call.  The
    // easiest way to ensure finite execution between safepoints in the face of
    // recursive and mutually recursive functions is to enforce that each take
    // a safepoint.  Additionally, we need to ensure a poll before any call
    // which can grow the stack by an unbounded amount.  This isn't required
    // for GC semantics per se, but is a common requirement for languages
    // which detect stack overflow via guard pages and then throw exceptions.
    if (auto CS = CallSite(cursor)) {
      if (doesNotRequireEntrySafepointBefore(CS))
        continue;
      break;
    }
  }

  assert((hasNextInstruction(cursor) || cursor->isTerminator()) &&
         "either we stopped because of a call, or because of terminator");

  return cursor;
}

/// Identify the list of call sites which need to be have parseable state
static void findCallSafepoints(Function &F,
                               std::vector<CallSite> &Found /*rval*/) {
  assert(Found.empty() && "must be empty!");
  for (Instruction &I : inst_range(F)) {
    Instruction *inst = &I;
    if (isa<CallInst>(inst) || isa<InvokeInst>(inst)) {
      CallSite CS(inst);

      // No safepoint needed or wanted
      if (!needsStatepoint(CS)) {
        continue;
      }

      Found.push_back(CS);
    }
  }
}

/// Implement a unique function which doesn't require we sort the input
/// vector.  Doing so has the effect of changing the output of a couple of
/// tests in ways which make them less useful in testing fused safepoints.
template <typename T> static void unique_unsorted(std::vector<T> &vec) {
  std::set<T> seen;
  std::vector<T> tmp;
  vec.reserve(vec.size());
  std::swap(tmp, vec);
  for (auto V : tmp) {
    if (seen.insert(V).second) {
      vec.push_back(V);
    }
  }
}

static const char *const GCSafepointPollName = "gc.safepoint_poll";

static bool isGCSafepointPoll(Function &F) {
  return F.getName().equals(GCSafepointPollName);
}

/// Returns true if this function should be rewritten to include safepoint
/// polls and parseable call sites.  The main point of this function is to be
/// an extension point for custom logic.
static bool shouldRewriteFunction(Function &F) {
  // TODO: This should check the GCStrategy
  if (F.hasGC()) {
    const char *FunctionGCName = F.getGC();
    const StringRef StatepointExampleName("statepoint-example");
    const StringRef CoreCLRName("coreclr");
    return (StatepointExampleName == FunctionGCName) ||
           (CoreCLRName == FunctionGCName);
  } else
    return false;
}

// TODO: These should become properties of the GCStrategy, possibly with
// command line overrides.
static bool enableEntrySafepoints(Function &F) { return !NoEntry; }
static bool enableBackedgeSafepoints(Function &F) { return !NoBackedge; }
static bool enableCallSafepoints(Function &F) { return !NoCall; }

// Normalize basic block to make it ready to be target of invoke statepoint.
// Ensure that 'BB' does not have phi nodes. It may require spliting it.
static BasicBlock *normalizeForInvokeSafepoint(BasicBlock *BB,
                                               BasicBlock *InvokeParent) {
  BasicBlock *ret = BB;

  if (!BB->getUniquePredecessor()) {
    ret = SplitBlockPredecessors(BB, InvokeParent, "");
  }

  // Now that 'ret' has unique predecessor we can safely remove all phi nodes
  // from it
  FoldSingleEntryPHINodes(ret);
  assert(!isa<PHINode>(ret->begin()));

  return ret;
}

bool PlaceSafepoints::runOnFunction(Function &F) {
  if (F.isDeclaration() || F.empty()) {
    // This is a declaration, nothing to do.  Must exit early to avoid crash in
    // dom tree calculation
    return false;
  }

  if (isGCSafepointPoll(F)) {
    // Given we're inlining this inside of safepoint poll insertion, this
    // doesn't make any sense.  Note that we do make any contained calls
    // parseable after we inline a poll.
    return false;
  }

  if (!shouldRewriteFunction(F))
    return false;

  bool modified = false;

  // In various bits below, we rely on the fact that uses are reachable from
  // defs.  When there are basic blocks unreachable from the entry, dominance
  // and reachablity queries return non-sensical results.  Thus, we preprocess
  // the function to ensure these properties hold.
  modified |= removeUnreachableBlocks(F);

  // STEP 1 - Insert the safepoint polling locations.  We do not need to
  // actually insert parse points yet.  That will be done for all polls and
  // calls in a single pass.

  DominatorTree DT;
  DT.recalculate(F);

  SmallVector<Instruction *, 16> PollsNeeded;
  std::vector<CallSite> ParsePointNeeded;

  if (enableBackedgeSafepoints(F)) {
    // Construct a pass manager to run the LoopPass backedge logic.  We
    // need the pass manager to handle scheduling all the loop passes
    // appropriately.  Doing this by hand is painful and just not worth messing
    // with for the moment.
    legacy::FunctionPassManager FPM(F.getParent());
    bool CanAssumeCallSafepoints = enableCallSafepoints(F);
    PlaceBackedgeSafepointsImpl *PBS =
      new PlaceBackedgeSafepointsImpl(CanAssumeCallSafepoints);
    FPM.add(PBS);
    FPM.run(F);

    // We preserve dominance information when inserting the poll, otherwise
    // we'd have to recalculate this on every insert
    DT.recalculate(F);

    auto &PollLocations = PBS->PollLocations;

    auto OrderByBBName = [](Instruction *a, Instruction *b) {
      return a->getParent()->getName() < b->getParent()->getName();
    };
    // We need the order of list to be stable so that naming ends up stable
    // when we split edges.  This makes test cases much easier to write.
    std::sort(PollLocations.begin(), PollLocations.end(), OrderByBBName);

    // We can sometimes end up with duplicate poll locations.  This happens if
    // a single loop is visited more than once.   The fact this happens seems
    // wrong, but it does happen for the split-backedge.ll test case.
    PollLocations.erase(std::unique(PollLocations.begin(),
                                    PollLocations.end()),
                        PollLocations.end());

    // Insert a poll at each point the analysis pass identified
    // The poll location must be the terminator of a loop latch block.
    for (TerminatorInst *Term : PollLocations) {
      // We are inserting a poll, the function is modified
      modified = true;

      if (SplitBackedge) {
        // Split the backedge of the loop and insert the poll within that new
        // basic block.  This creates a loop with two latches per original
        // latch (which is non-ideal), but this appears to be easier to
        // optimize in practice than inserting the poll immediately before the
        // latch test.

        // Since this is a latch, at least one of the successors must dominate
        // it. Its possible that we have a) duplicate edges to the same header
        // and b) edges to distinct loop headers.  We need to insert pools on
        // each.
        SetVector<BasicBlock *> Headers;
        for (unsigned i = 0; i < Term->getNumSuccessors(); i++) {
          BasicBlock *Succ = Term->getSuccessor(i);
          if (DT.dominates(Succ, Term->getParent())) {
            Headers.insert(Succ);
          }
        }
        assert(!Headers.empty() && "poll location is not a loop latch?");

        // The split loop structure here is so that we only need to recalculate
        // the dominator tree once.  Alternatively, we could just keep it up to
        // date and use a more natural merged loop.
        SetVector<BasicBlock *> SplitBackedges;
        for (BasicBlock *Header : Headers) {
          BasicBlock *NewBB = SplitEdge(Term->getParent(), Header, &DT);
          PollsNeeded.push_back(NewBB->getTerminator());
          NumBackedgeSafepoints++;
        }
      } else {
        // Split the latch block itself, right before the terminator.
        PollsNeeded.push_back(Term);
        NumBackedgeSafepoints++;
      }
    }
  }

  if (enableEntrySafepoints(F)) {
    Instruction *Location = findLocationForEntrySafepoint(F, DT);
    if (!Location) {
      // policy choice not to insert?
    } else {
      PollsNeeded.push_back(Location);
      modified = true;
      NumEntrySafepoints++;
    }
  }

  // Now that we've identified all the needed safepoint poll locations, insert
  // safepoint polls themselves.
  for (Instruction *PollLocation : PollsNeeded) {
    std::vector<CallSite> RuntimeCalls;
    InsertSafepointPoll(PollLocation, RuntimeCalls);
    ParsePointNeeded.insert(ParsePointNeeded.end(), RuntimeCalls.begin(),
                            RuntimeCalls.end());
  }
  PollsNeeded.clear(); // make sure we don't accidentally use
  // The dominator tree has been invalidated by the inlining performed in the
  // above loop.  TODO: Teach the inliner how to update the dom tree?
  DT.recalculate(F);

  if (enableCallSafepoints(F)) {
    std::vector<CallSite> Calls;
    findCallSafepoints(F, Calls);
    NumCallSafepoints += Calls.size();
    ParsePointNeeded.insert(ParsePointNeeded.end(), Calls.begin(), Calls.end());
  }

  // Unique the vectors since we can end up with duplicates if we scan the call
  // site for call safepoints after we add it for entry or backedge.  The
  // only reason we need tracking at all is that some functions might have
  // polls but not call safepoints and thus we might miss marking the runtime
  // calls for the polls. (This is useful in test cases!)
  unique_unsorted(ParsePointNeeded);

  // Any parse point (no matter what source) will be handled here

  // We're about to start modifying the function
  if (!ParsePointNeeded.empty())
    modified = true;

  // Now run through and insert the safepoints, but do _NOT_ update or remove
  // any existing uses.  We have references to live variables that need to
  // survive to the last iteration of this loop.
  std::vector<Value *> Results;
  Results.reserve(ParsePointNeeded.size());
  for (size_t i = 0; i < ParsePointNeeded.size(); i++) {
    CallSite &CS = ParsePointNeeded[i];

    // For invoke statepoints we need to remove all phi nodes at the normal
    // destination block.
    // Reason for this is that we can place gc_result only after last phi node
    // in basic block. We will get malformed code after RAUW for the
    // gc_result if one of this phi nodes uses result from the invoke.
    if (InvokeInst *Invoke = dyn_cast<InvokeInst>(CS.getInstruction())) {
      normalizeForInvokeSafepoint(Invoke->getNormalDest(),
                                  Invoke->getParent());
    }

    Value *GCResult = ReplaceWithStatepoint(CS, nullptr);
    Results.push_back(GCResult);
  }
  assert(Results.size() == ParsePointNeeded.size());

  // Adjust all users of the old call sites to use the new ones instead
  for (size_t i = 0; i < ParsePointNeeded.size(); i++) {
    CallSite &CS = ParsePointNeeded[i];
    Value *GCResult = Results[i];
    if (GCResult) {
      // Can not RAUW for the invoke gc result in case of phi nodes preset.
      assert(CS.isCall() || !isa<PHINode>(cast<Instruction>(GCResult)->getParent()->begin()));

      // Replace all uses with the new call
      CS.getInstruction()->replaceAllUsesWith(GCResult);
    }

    // Now that we've handled all uses, remove the original call itself
    // Note: The insert point can't be the deleted instruction!
    CS.getInstruction()->eraseFromParent();
  }
  return modified;
}

char PlaceBackedgeSafepointsImpl::ID = 0;
char PlaceSafepoints::ID = 0;

FunctionPass *llvm::createPlaceSafepointsPass() {
  return new PlaceSafepoints();
}

INITIALIZE_PASS_BEGIN(PlaceBackedgeSafepointsImpl,
                      "place-backedge-safepoints-impl",
                      "Place Backedge Safepoints", false, false)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolution)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_END(PlaceBackedgeSafepointsImpl,
                    "place-backedge-safepoints-impl",
                    "Place Backedge Safepoints", false, false)

INITIALIZE_PASS_BEGIN(PlaceSafepoints, "place-safepoints", "Place Safepoints",
                      false, false)
INITIALIZE_PASS_END(PlaceSafepoints, "place-safepoints", "Place Safepoints",
                    false, false)

static bool isGCLeafFunction(const CallSite &CS) {
  Instruction *inst = CS.getInstruction();
  if (isa<IntrinsicInst>(inst)) {
    // Most LLVM intrinsics are things which can never take a safepoint.
    // As a result, we don't need to have the stack parsable at the
    // callsite.  This is a highly useful optimization since intrinsic
    // calls are fairly prevelent, particularly in debug builds.
    return true;
  }

  // If this function is marked explicitly as a leaf call, we don't need to
  // place a safepoint of it.  In fact, for correctness we *can't* in many
  // cases.  Note: Indirect calls return Null for the called function,
  // these obviously aren't runtime functions with attributes
  // TODO: Support attributes on the call site as well.
  const Function *F = CS.getCalledFunction();
  bool isLeaf =
      F &&
      F->getFnAttribute("gc-leaf-function").getValueAsString().equals("true");
  if (isLeaf) {
    return true;
  }
  return false;
}

static void
InsertSafepointPoll(Instruction *InsertBefore,
                    std::vector<CallSite> &ParsePointsNeeded /*rval*/) {
  BasicBlock *OrigBB = InsertBefore->getParent();
  Module *M = InsertBefore->getModule();
  assert(M && "must be part of a module");

  // Inline the safepoint poll implementation - this will get all the branch,
  // control flow, etc..  Most importantly, it will introduce the actual slow
  // path call - where we need to insert a safepoint (parsepoint).

  auto *F = M->getFunction(GCSafepointPollName);
  assert(F->getType()->getElementType() ==
         FunctionType::get(Type::getVoidTy(M->getContext()), false) &&
         "gc.safepoint_poll declared with wrong type");
  assert(!F->empty() && "gc.safepoint_poll must be a non-empty function");
  CallInst *PollCall = CallInst::Create(F, "", InsertBefore);

  // Record some information about the call site we're replacing
  BasicBlock::iterator before(PollCall), after(PollCall);
  bool isBegin(false);
  if (before == OrigBB->begin()) {
    isBegin = true;
  } else {
    before--;
  }
  after++;
  assert(after != OrigBB->end() && "must have successor");

  // do the actual inlining
  InlineFunctionInfo IFI;
  bool InlineStatus = InlineFunction(PollCall, IFI);
  assert(InlineStatus && "inline must succeed");
  (void)InlineStatus; // suppress warning in release-asserts

  // Check post conditions
  assert(IFI.StaticAllocas.empty() && "can't have allocs");

  std::vector<CallInst *> calls; // new calls
  std::set<BasicBlock *> BBs;    // new BBs + insertee
  // Include only the newly inserted instructions, Note: begin may not be valid
  // if we inserted to the beginning of the basic block
  BasicBlock::iterator start;
  if (isBegin) {
    start = OrigBB->begin();
  } else {
    start = before;
    start++;
  }

  // If your poll function includes an unreachable at the end, that's not
  // valid.  Bugpoint likes to create this, so check for it.
  assert(isPotentiallyReachable(&*start, &*after, nullptr, nullptr) &&
         "malformed poll function");

  scanInlinedCode(&*(start), &*(after), calls, BBs);
  assert(!calls.empty() && "slow path not found for safepoint poll");

  // Record the fact we need a parsable state at the runtime call contained in
  // the poll function.  This is required so that the runtime knows how to
  // parse the last frame when we actually take  the safepoint (i.e. execute
  // the slow path)
  assert(ParsePointsNeeded.empty());
  for (size_t i = 0; i < calls.size(); i++) {

    // No safepoint needed or wanted
    if (!needsStatepoint(calls[i])) {
      continue;
    }

    // These are likely runtime calls.  Should we assert that via calling
    // convention or something?
    ParsePointsNeeded.push_back(CallSite(calls[i]));
  }
  assert(ParsePointsNeeded.size() <= calls.size());
}

/// Replaces the given call site (Call or Invoke) with a gc.statepoint
/// intrinsic with an empty deoptimization arguments list.  This does
/// NOT do explicit relocation for GC support.
static Value *ReplaceWithStatepoint(const CallSite &CS, /* to replace */
                                    Pass *P) {
  assert(CS.getInstruction()->getParent()->getParent()->getParent() &&
         "must be set");

  // TODO: technically, a pass is not allowed to get functions from within a
  // function pass since it might trigger a new function addition.  Refactor
  // this logic out to the initialization of the pass.  Doesn't appear to
  // matter in practice.

  // Then go ahead and use the builder do actually do the inserts.  We insert
  // immediately before the previous instruction under the assumption that all
  // arguments will be available here.  We can't insert afterwards since we may
  // be replacing a terminator.
  IRBuilder<> Builder(CS.getInstruction());

  // Note: The gc args are not filled in at this time, that's handled by
  // RewriteStatepointsForGC (which is currently under review).

  // Create the statepoint given all the arguments
  Instruction *Token = nullptr;

  uint64_t ID;
  uint32_t NumPatchBytes;

  AttributeSet OriginalAttrs = CS.getAttributes();
  Attribute AttrID =
      OriginalAttrs.getAttribute(AttributeSet::FunctionIndex, "statepoint-id");
  Attribute AttrNumPatchBytes = OriginalAttrs.getAttribute(
      AttributeSet::FunctionIndex, "statepoint-num-patch-bytes");

  AttrBuilder AttrsToRemove;
  bool HasID = AttrID.isStringAttribute() &&
               !AttrID.getValueAsString().getAsInteger(10, ID);

  if (HasID)
    AttrsToRemove.addAttribute("statepoint-id");
  else
    ID = 0xABCDEF00;

  bool HasNumPatchBytes =
      AttrNumPatchBytes.isStringAttribute() &&
      !AttrNumPatchBytes.getValueAsString().getAsInteger(10, NumPatchBytes);

  if (HasNumPatchBytes)
    AttrsToRemove.addAttribute("statepoint-num-patch-bytes");
  else
    NumPatchBytes = 0;

  OriginalAttrs = OriginalAttrs.removeAttributes(
      CS.getInstruction()->getContext(), AttributeSet::FunctionIndex,
      AttrsToRemove);

  Value *StatepointTarget = NumPatchBytes == 0
                                ? CS.getCalledValue()
                                : ConstantPointerNull::get(cast<PointerType>(
                                      CS.getCalledValue()->getType()));

  if (CS.isCall()) {
    CallInst *ToReplace = cast<CallInst>(CS.getInstruction());
    CallInst *Call = Builder.CreateGCStatepointCall(
        ID, NumPatchBytes, StatepointTarget,
        makeArrayRef(CS.arg_begin(), CS.arg_end()), None, None,
        "safepoint_token");
    Call->setTailCall(ToReplace->isTailCall());
    Call->setCallingConv(ToReplace->getCallingConv());

    // In case if we can handle this set of attributes - set up function
    // attributes directly on statepoint and return attributes later for
    // gc_result intrinsic.
    Call->setAttributes(OriginalAttrs.getFnAttributes());

    Token = Call;

    // Put the following gc_result and gc_relocate calls immediately after the
    // the old call (which we're about to delete).
    assert(ToReplace->getNextNode() && "not a terminator, must have next");
    Builder.SetInsertPoint(ToReplace->getNextNode());
    Builder.SetCurrentDebugLocation(ToReplace->getNextNode()->getDebugLoc());
  } else if (CS.isInvoke()) {
    InvokeInst *ToReplace = cast<InvokeInst>(CS.getInstruction());

    // Insert the new invoke into the old block.  We'll remove the old one in a
    // moment at which point this will become the new terminator for the
    // original block.
    Builder.SetInsertPoint(ToReplace->getParent());
    InvokeInst *Invoke = Builder.CreateGCStatepointInvoke(
        ID, NumPatchBytes, StatepointTarget, ToReplace->getNormalDest(),
        ToReplace->getUnwindDest(), makeArrayRef(CS.arg_begin(), CS.arg_end()),
        None, None, "safepoint_token");

    Invoke->setCallingConv(ToReplace->getCallingConv());

    // In case if we can handle this set of attributes - set up function
    // attributes directly on statepoint and return attributes later for
    // gc_result intrinsic.
    Invoke->setAttributes(OriginalAttrs.getFnAttributes());

    Token = Invoke;

    // We'll insert the gc.result into the normal block
    BasicBlock *NormalDest = ToReplace->getNormalDest();
    // Can not insert gc.result in case of phi nodes preset.
    // Should have removed this cases prior to runnning this function
    assert(!isa<PHINode>(NormalDest->begin()));
    Instruction *IP = &*(NormalDest->getFirstInsertionPt());
    Builder.SetInsertPoint(IP);
  } else {
    llvm_unreachable("unexpect type of CallSite");
  }
  assert(Token);

  // Handle the return value of the original call - update all uses to use a
  // gc_result hanging off the statepoint node we just inserted

  // Only add the gc_result iff there is actually a used result
  if (!CS.getType()->isVoidTy() && !CS.getInstruction()->use_empty()) {
    std::string TakenName =
        CS.getInstruction()->hasName() ? CS.getInstruction()->getName() : "";
    CallInst *GCResult = Builder.CreateGCResult(Token, CS.getType(), TakenName);
    GCResult->setAttributes(OriginalAttrs.getRetAttributes());
    return GCResult;
  } else {
    // No return value for the call.
    return nullptr;
  }
}
