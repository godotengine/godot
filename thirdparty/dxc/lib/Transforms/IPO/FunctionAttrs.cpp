//===- FunctionAttrs.cpp - Pass which marks functions attributes ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a simple interprocedural pass which walks the
// call-graph, looking for functions which do not access or only read
// non-local memory, and marking them readnone/readonly.  It does the
// same with function arguments independently, marking them readonly/
// readnone/nocapture.  Finally, well-known library call declarations
// are marked with all attributes that are consistent with the
// function's standard definition. This pass is implemented as a
// bottom-up traversal of the call-graph.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/CallGraphSCCPass.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
using namespace llvm;

#define DEBUG_TYPE "functionattrs"

STATISTIC(NumReadNone, "Number of functions marked readnone");
STATISTIC(NumReadOnly, "Number of functions marked readonly");
STATISTIC(NumNoCapture, "Number of arguments marked nocapture");
STATISTIC(NumReadNoneArg, "Number of arguments marked readnone");
STATISTIC(NumReadOnlyArg, "Number of arguments marked readonly");
STATISTIC(NumNoAlias, "Number of function returns marked noalias");
STATISTIC(NumAnnotated, "Number of attributes added to library functions");

namespace {
  struct FunctionAttrs : public CallGraphSCCPass {
    static char ID; // Pass identification, replacement for typeid
    FunctionAttrs() : CallGraphSCCPass(ID), AA(nullptr) {
      initializeFunctionAttrsPass(*PassRegistry::getPassRegistry());
    }

    // runOnSCC - Analyze the SCC, performing the transformation if possible.
    bool runOnSCC(CallGraphSCC &SCC) override;

    // AddReadAttrs - Deduce readonly/readnone attributes for the SCC.
    bool AddReadAttrs(const CallGraphSCC &SCC);

    // AddArgumentAttrs - Deduce nocapture attributes for the SCC.
    bool AddArgumentAttrs(const CallGraphSCC &SCC);

    // IsFunctionMallocLike - Does this function allocate new memory?
    bool IsFunctionMallocLike(Function *F,
                              SmallPtrSet<Function*, 8> &) const;

    // AddNoAliasAttrs - Deduce noalias attributes for the SCC.
    bool AddNoAliasAttrs(const CallGraphSCC &SCC);

    // Utility methods used by inferPrototypeAttributes to add attributes
    // and maintain annotation statistics.

    void setDoesNotAccessMemory(Function &F) {
      if (!F.doesNotAccessMemory()) {
        F.setDoesNotAccessMemory();
        ++NumAnnotated;
      }
    }

    void setOnlyReadsMemory(Function &F) {
      if (!F.onlyReadsMemory()) {
        F.setOnlyReadsMemory();
        ++NumAnnotated;
      }
    }

    void setDoesNotThrow(Function &F) {
      if (!F.doesNotThrow()) {
        F.setDoesNotThrow();
        ++NumAnnotated;
      }
    }

    void setDoesNotCapture(Function &F, unsigned n) {
      if (!F.doesNotCapture(n)) {
        F.setDoesNotCapture(n);
        ++NumAnnotated;
      }
    }

    void setOnlyReadsMemory(Function &F, unsigned n) {
      if (!F.onlyReadsMemory(n)) {
        F.setOnlyReadsMemory(n);
        ++NumAnnotated;
      }
    }

    void setDoesNotAlias(Function &F, unsigned n) {
      if (!F.doesNotAlias(n)) {
        F.setDoesNotAlias(n);
        ++NumAnnotated;
      }
    }

    // inferPrototypeAttributes - Analyze the name and prototype of the
    // given function and set any applicable attributes.  Returns true
    // if any attributes were set and false otherwise.
    bool inferPrototypeAttributes(Function &F);

    // annotateLibraryCalls - Adds attributes to well-known standard library
    // call declarations.
    bool annotateLibraryCalls(const CallGraphSCC &SCC);

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesCFG();
      AU.addRequired<AliasAnalysis>();
      AU.addRequired<TargetLibraryInfoWrapperPass>();
      CallGraphSCCPass::getAnalysisUsage(AU);
    }

  private:
    AliasAnalysis *AA;
    TargetLibraryInfo *TLI;
  };
}

char FunctionAttrs::ID = 0;
INITIALIZE_PASS_BEGIN(FunctionAttrs, "functionattrs",
                "Deduce function attributes", false, false)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_DEPENDENCY(CallGraphWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(FunctionAttrs, "functionattrs",
                "Deduce function attributes", false, false)

Pass *llvm::createFunctionAttrsPass() { return new FunctionAttrs(); }


/// AddReadAttrs - Deduce readonly/readnone attributes for the SCC.
bool FunctionAttrs::AddReadAttrs(const CallGraphSCC &SCC) {
  SmallPtrSet<Function*, 8> SCCNodes;

  // Fill SCCNodes with the elements of the SCC.  Used for quickly
  // looking up whether a given CallGraphNode is in this SCC.
  for (CallGraphSCC::iterator I = SCC.begin(), E = SCC.end(); I != E; ++I)
    SCCNodes.insert((*I)->getFunction());

  // Check if any of the functions in the SCC read or write memory.  If they
  // write memory then they can't be marked readnone or readonly.
  bool ReadsMemory = false;
  for (CallGraphSCC::iterator I = SCC.begin(), E = SCC.end(); I != E; ++I) {
    Function *F = (*I)->getFunction();

    if (!F || F->hasFnAttribute(Attribute::OptimizeNone))
      // External node or node we don't want to optimize - assume it may write
      // memory and give up.
      return false;

    AliasAnalysis::ModRefBehavior MRB = AA->getModRefBehavior(F);
    if (MRB == AliasAnalysis::DoesNotAccessMemory)
      // Already perfect!
      continue;

    // Definitions with weak linkage may be overridden at linktime with
    // something that writes memory, so treat them like declarations.
    if (F->isDeclaration() || F->mayBeOverridden()) {
      if (!AliasAnalysis::onlyReadsMemory(MRB))
        // May write memory.  Just give up.
        return false;

      ReadsMemory = true;
      continue;
    }

    // Scan the function body for instructions that may read or write memory.
    for (inst_iterator II = inst_begin(F), E = inst_end(F); II != E; ++II) {
      Instruction *I = &*II;

      // Some instructions can be ignored even if they read or write memory.
      // Detect these now, skipping to the next instruction if one is found.
      CallSite CS(cast<Value>(I));
      if (CS) {
        // Ignore calls to functions in the same SCC.
        if (CS.getCalledFunction() && SCCNodes.count(CS.getCalledFunction()))
          continue;
        AliasAnalysis::ModRefBehavior MRB = AA->getModRefBehavior(CS);
        // If the call doesn't access arbitrary memory, we may be able to
        // figure out something.
        if (AliasAnalysis::onlyAccessesArgPointees(MRB)) {
          // If the call does access argument pointees, check each argument.
          if (AliasAnalysis::doesAccessArgPointees(MRB))
            // Check whether all pointer arguments point to local memory, and
            // ignore calls that only access local memory.
            for (CallSite::arg_iterator CI = CS.arg_begin(), CE = CS.arg_end();
                 CI != CE; ++CI) {
              Value *Arg = *CI;
              if (Arg->getType()->isPointerTy()) {
                AAMDNodes AAInfo;
                I->getAAMetadata(AAInfo);

                MemoryLocation Loc(Arg, MemoryLocation::UnknownSize, AAInfo);
                if (!AA->pointsToConstantMemory(Loc, /*OrLocal=*/true)) {
                  if (MRB & AliasAnalysis::Mod)
                    // Writes non-local memory.  Give up.
                    return false;
                  if (MRB & AliasAnalysis::Ref)
                    // Ok, it reads non-local memory.
                    ReadsMemory = true;
                }
              }
            }
          continue;
        }
        // The call could access any memory. If that includes writes, give up.
        if (MRB & AliasAnalysis::Mod)
          return false;
        // If it reads, note it.
        if (MRB & AliasAnalysis::Ref)
          ReadsMemory = true;
        continue;
      } else if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
        // Ignore non-volatile loads from local memory. (Atomic is okay here.)
        if (!LI->isVolatile()) {
          MemoryLocation Loc = MemoryLocation::get(LI);
          if (AA->pointsToConstantMemory(Loc, /*OrLocal=*/true))
            continue;
        }
      } else if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
        // Ignore non-volatile stores to local memory. (Atomic is okay here.)
        if (!SI->isVolatile()) {
          MemoryLocation Loc = MemoryLocation::get(SI);
          if (AA->pointsToConstantMemory(Loc, /*OrLocal=*/true))
            continue;
        }
      } else if (VAArgInst *VI = dyn_cast<VAArgInst>(I)) {
        // Ignore vaargs on local memory.
        MemoryLocation Loc = MemoryLocation::get(VI);
        if (AA->pointsToConstantMemory(Loc, /*OrLocal=*/true))
          continue;
      }

      // Any remaining instructions need to be taken seriously!  Check if they
      // read or write memory.
      if (I->mayWriteToMemory())
        // Writes memory.  Just give up.
        return false;

      // If this instruction may read memory, remember that.
      ReadsMemory |= I->mayReadFromMemory();
    }
  }

  // Success!  Functions in this SCC do not access memory, or only read memory.
  // Give them the appropriate attribute.
  bool MadeChange = false;
  for (CallGraphSCC::iterator I = SCC.begin(), E = SCC.end(); I != E; ++I) {
    Function *F = (*I)->getFunction();

    if (F->doesNotAccessMemory())
      // Already perfect!
      continue;

    if (F->onlyReadsMemory() && ReadsMemory)
      // No change.
      continue;

    MadeChange = true;

    // Clear out any existing attributes.
    AttrBuilder B;
    B.addAttribute(Attribute::ReadOnly)
      .addAttribute(Attribute::ReadNone);
    F->removeAttributes(AttributeSet::FunctionIndex,
                        AttributeSet::get(F->getContext(),
                                          AttributeSet::FunctionIndex, B));

    // Add in the new attribute.
    F->addAttribute(AttributeSet::FunctionIndex,
                    ReadsMemory ? Attribute::ReadOnly : Attribute::ReadNone);

    if (ReadsMemory)
      ++NumReadOnly;
    else
      ++NumReadNone;
  }

  return MadeChange;
}

namespace {
  // For a given pointer Argument, this retains a list of Arguments of functions
  // in the same SCC that the pointer data flows into. We use this to build an
  // SCC of the arguments.
  struct ArgumentGraphNode {
    Argument *Definition;
    SmallVector<ArgumentGraphNode*, 4> Uses;
  };

  class ArgumentGraph {
    // We store pointers to ArgumentGraphNode objects, so it's important that
    // that they not move around upon insert.
    typedef std::map<Argument*, ArgumentGraphNode> ArgumentMapTy;

    ArgumentMapTy ArgumentMap;

    // There is no root node for the argument graph, in fact:
    //   void f(int *x, int *y) { if (...) f(x, y); }
    // is an example where the graph is disconnected. The SCCIterator requires a
    // single entry point, so we maintain a fake ("synthetic") root node that
    // uses every node. Because the graph is directed and nothing points into
    // the root, it will not participate in any SCCs (except for its own).
    ArgumentGraphNode SyntheticRoot;

  public:
    ArgumentGraph() { SyntheticRoot.Definition = nullptr; }

    typedef SmallVectorImpl<ArgumentGraphNode*>::iterator iterator;

    iterator begin() { return SyntheticRoot.Uses.begin(); }
    iterator end() { return SyntheticRoot.Uses.end(); }
    ArgumentGraphNode *getEntryNode() { return &SyntheticRoot; }

    ArgumentGraphNode *operator[](Argument *A) {
      ArgumentGraphNode &Node = ArgumentMap[A];
      Node.Definition = A;
      SyntheticRoot.Uses.push_back(&Node);
      return &Node;
    }
  };

  // This tracker checks whether callees are in the SCC, and if so it does not
  // consider that a capture, instead adding it to the "Uses" list and
  // continuing with the analysis.
  struct ArgumentUsesTracker : public CaptureTracker {
    ArgumentUsesTracker(const SmallPtrSet<Function*, 8> &SCCNodes)
      : Captured(false), SCCNodes(SCCNodes) {}

    void tooManyUses() override { Captured = true; }

    bool captured(const Use *U) override {
      CallSite CS(U->getUser());
      if (!CS.getInstruction()) { Captured = true; return true; }

      Function *F = CS.getCalledFunction();
      if (!F || !SCCNodes.count(F)) { Captured = true; return true; }

      bool Found = false;
      Function::arg_iterator AI = F->arg_begin(), AE = F->arg_end();
      for (CallSite::arg_iterator PI = CS.arg_begin(), PE = CS.arg_end();
           PI != PE; ++PI, ++AI) {
        if (AI == AE) {
          assert(F->isVarArg() && "More params than args in non-varargs call");
          Captured = true;
          return true;
        }
        if (PI == U) {
          Uses.push_back(AI);
          Found = true;
          break;
        }
      }
      assert(Found && "Capturing call-site captured nothing?");
      (void)Found;
      return false;
    }

    bool Captured;  // True only if certainly captured (used outside our SCC).
    SmallVector<Argument*, 4> Uses;  // Uses within our SCC.

    const SmallPtrSet<Function*, 8> &SCCNodes;
  };
}

namespace llvm {
  template<> struct GraphTraits<ArgumentGraphNode*> {
    typedef ArgumentGraphNode NodeType;
    typedef SmallVectorImpl<ArgumentGraphNode*>::iterator ChildIteratorType;

    static inline NodeType *getEntryNode(NodeType *A) { return A; }
    static inline ChildIteratorType child_begin(NodeType *N) {
      return N->Uses.begin();
    }
    static inline ChildIteratorType child_end(NodeType *N) {
      return N->Uses.end();
    }
  };
  template<> struct GraphTraits<ArgumentGraph*>
    : public GraphTraits<ArgumentGraphNode*> {
    static NodeType *getEntryNode(ArgumentGraph *AG) {
      return AG->getEntryNode();
    }
    static ChildIteratorType nodes_begin(ArgumentGraph *AG) {
      return AG->begin();
    }
    static ChildIteratorType nodes_end(ArgumentGraph *AG) {
      return AG->end();
    }
  };
}

// Returns Attribute::None, Attribute::ReadOnly or Attribute::ReadNone.
static Attribute::AttrKind
determinePointerReadAttrs(Argument *A,
                          const SmallPtrSet<Argument*, 8> &SCCNodes) {
                                                       
  SmallVector<Use*, 32> Worklist;
  SmallSet<Use*, 32> Visited;
  int Count = 0;

  // inalloca arguments are always clobbered by the call.
  if (A->hasInAllocaAttr())
    return Attribute::None;

  bool IsRead = false;
  // We don't need to track IsWritten. If A is written to, return immediately.

  for (Use &U : A->uses()) {
    if (Count++ >= 20)
      return Attribute::None;

    Visited.insert(&U);
    Worklist.push_back(&U);
  }

  while (!Worklist.empty()) {
    Use *U = Worklist.pop_back_val();
    Instruction *I = cast<Instruction>(U->getUser());
    Value *V = U->get();

    switch (I->getOpcode()) {
    case Instruction::BitCast:
    case Instruction::GetElementPtr:
    case Instruction::PHI:
    case Instruction::Select:
    case Instruction::AddrSpaceCast:
      // The original value is not read/written via this if the new value isn't.
      for (Use &UU : I->uses())
        if (Visited.insert(&UU).second)
          Worklist.push_back(&UU);
      break;

    case Instruction::Call:
    case Instruction::Invoke: {
      bool Captures = true;

      if (I->getType()->isVoidTy())
        Captures = false;

      auto AddUsersToWorklistIfCapturing = [&] {
        if (Captures)
          for (Use &UU : I->uses())
            if (Visited.insert(&UU).second)
              Worklist.push_back(&UU);
      };

      CallSite CS(I);
      if (CS.doesNotAccessMemory()) {
        AddUsersToWorklistIfCapturing();
        continue;
      }

      Function *F = CS.getCalledFunction();
      if (!F) {
        if (CS.onlyReadsMemory()) {
          IsRead = true;
          AddUsersToWorklistIfCapturing();
          continue;
        }
        return Attribute::None;
      }

      Function::arg_iterator AI = F->arg_begin(), AE = F->arg_end();
      CallSite::arg_iterator B = CS.arg_begin(), E = CS.arg_end();
      for (CallSite::arg_iterator A = B; A != E; ++A, ++AI) {
        if (A->get() == V) {
          if (AI == AE) {
            assert(F->isVarArg() &&
                   "More params than args in non-varargs call.");
            return Attribute::None;
          }
          Captures &= !CS.doesNotCapture(A - B);
          if (SCCNodes.count(AI))
            continue;
          if (!CS.onlyReadsMemory() && !CS.onlyReadsMemory(A - B))
            return Attribute::None;
          if (!CS.doesNotAccessMemory(A - B))
            IsRead = true;
        }
      }
      AddUsersToWorklistIfCapturing();
      break;
    }

    case Instruction::Load:
      IsRead = true;
      break;

    case Instruction::ICmp:
    case Instruction::Ret:
      break;

    default:
      return Attribute::None;
    }
  }

  return IsRead ? Attribute::ReadOnly : Attribute::ReadNone;
}

/// AddArgumentAttrs - Deduce nocapture attributes for the SCC.
bool FunctionAttrs::AddArgumentAttrs(const CallGraphSCC &SCC) {
  bool Changed = false;

  SmallPtrSet<Function*, 8> SCCNodes;

  // Fill SCCNodes with the elements of the SCC.  Used for quickly
  // looking up whether a given CallGraphNode is in this SCC.
  for (CallGraphSCC::iterator I = SCC.begin(), E = SCC.end(); I != E; ++I) {
    Function *F = (*I)->getFunction();
    if (F && !F->isDeclaration() && !F->mayBeOverridden() &&
        !F->hasFnAttribute(Attribute::OptimizeNone))
      SCCNodes.insert(F);
  }

  ArgumentGraph AG;

  AttrBuilder B;
  B.addAttribute(Attribute::NoCapture);

  // Check each function in turn, determining which pointer arguments are not
  // captured.
  for (CallGraphSCC::iterator I = SCC.begin(), E = SCC.end(); I != E; ++I) {
    Function *F = (*I)->getFunction();

    if (!F || F->hasFnAttribute(Attribute::OptimizeNone))
      // External node or function we're trying not to optimize - only a problem
      // for arguments that we pass to it.
      continue;

    // Definitions with weak linkage may be overridden at linktime with
    // something that captures pointers, so treat them like declarations.
    if (F->isDeclaration() || F->mayBeOverridden())
      continue;

    // Functions that are readonly (or readnone) and nounwind and don't return
    // a value can't capture arguments. Don't analyze them.
    if (F->onlyReadsMemory() && F->doesNotThrow() &&
        F->getReturnType()->isVoidTy()) {
      for (Function::arg_iterator A = F->arg_begin(), E = F->arg_end();
           A != E; ++A) {
        if (A->getType()->isPointerTy() && !A->hasNoCaptureAttr()) {
          A->addAttr(AttributeSet::get(F->getContext(), A->getArgNo() + 1, B));
          ++NumNoCapture;
          Changed = true;
        }
      }
      continue;
    }

    for (Function::arg_iterator A = F->arg_begin(), E = F->arg_end();
         A != E; ++A) {
      if (!A->getType()->isPointerTy()) continue;
      bool HasNonLocalUses = false;
      if (!A->hasNoCaptureAttr()) {
        ArgumentUsesTracker Tracker(SCCNodes);
        PointerMayBeCaptured(A, &Tracker);
        if (!Tracker.Captured) {
          if (Tracker.Uses.empty()) {
            // If it's trivially not captured, mark it nocapture now.
            A->addAttr(AttributeSet::get(F->getContext(), A->getArgNo()+1, B));
            ++NumNoCapture;
            Changed = true;
          } else {
            // If it's not trivially captured and not trivially not captured,
            // then it must be calling into another function in our SCC. Save
            // its particulars for Argument-SCC analysis later.
            ArgumentGraphNode *Node = AG[A];
            for (SmallVectorImpl<Argument*>::iterator UI = Tracker.Uses.begin(),
                     UE = Tracker.Uses.end(); UI != UE; ++UI) {
              Node->Uses.push_back(AG[*UI]);
              if (*UI != A)
                HasNonLocalUses = true;
            }
          }
        }
        // Otherwise, it's captured. Don't bother doing SCC analysis on it.
      }
      if (!HasNonLocalUses && !A->onlyReadsMemory()) {
        // Can we determine that it's readonly/readnone without doing an SCC?
        // Note that we don't allow any calls at all here, or else our result
        // will be dependent on the iteration order through the functions in the
        // SCC.
        SmallPtrSet<Argument*, 8> Self;
        Self.insert(A);
        Attribute::AttrKind R = determinePointerReadAttrs(A, Self);
        if (R != Attribute::None) {
          AttrBuilder B;
          B.addAttribute(R);
          A->addAttr(AttributeSet::get(A->getContext(), A->getArgNo() + 1, B));
          Changed = true;
          R == Attribute::ReadOnly ? ++NumReadOnlyArg : ++NumReadNoneArg;
        }
      }
    }
  }

  // The graph we've collected is partial because we stopped scanning for
  // argument uses once we solved the argument trivially. These partial nodes
  // show up as ArgumentGraphNode objects with an empty Uses list, and for
  // these nodes the final decision about whether they capture has already been
  // made.  If the definition doesn't have a 'nocapture' attribute by now, it
  // captures.

  for (scc_iterator<ArgumentGraph*> I = scc_begin(&AG); !I.isAtEnd(); ++I) {
    const std::vector<ArgumentGraphNode *> &ArgumentSCC = *I;
    if (ArgumentSCC.size() == 1) {
      if (!ArgumentSCC[0]->Definition) continue;  // synthetic root node

      // eg. "void f(int* x) { if (...) f(x); }"
      if (ArgumentSCC[0]->Uses.size() == 1 &&
          ArgumentSCC[0]->Uses[0] == ArgumentSCC[0]) {
        Argument *A = ArgumentSCC[0]->Definition;
        A->addAttr(AttributeSet::get(A->getContext(), A->getArgNo() + 1, B));
        ++NumNoCapture;
        Changed = true;
      }
      continue;
    }

    bool SCCCaptured = false;
    for (auto I = ArgumentSCC.begin(), E = ArgumentSCC.end();
         I != E && !SCCCaptured; ++I) {
      ArgumentGraphNode *Node = *I;
      if (Node->Uses.empty()) {
        if (!Node->Definition->hasNoCaptureAttr())
          SCCCaptured = true;
      }
    }
    if (SCCCaptured) continue;

    SmallPtrSet<Argument*, 8> ArgumentSCCNodes;
    // Fill ArgumentSCCNodes with the elements of the ArgumentSCC.  Used for
    // quickly looking up whether a given Argument is in this ArgumentSCC.
    for (auto I = ArgumentSCC.begin(), E = ArgumentSCC.end(); I != E; ++I) {
      ArgumentSCCNodes.insert((*I)->Definition);
    }

    for (auto I = ArgumentSCC.begin(), E = ArgumentSCC.end();
         I != E && !SCCCaptured; ++I) {
      ArgumentGraphNode *N = *I;
      for (SmallVectorImpl<ArgumentGraphNode*>::iterator UI = N->Uses.begin(),
             UE = N->Uses.end(); UI != UE; ++UI) {
        Argument *A = (*UI)->Definition;
        if (A->hasNoCaptureAttr() || ArgumentSCCNodes.count(A))
          continue;
        SCCCaptured = true;
        break;
      }
    }
    if (SCCCaptured) continue;

    for (unsigned i = 0, e = ArgumentSCC.size(); i != e; ++i) {
      Argument *A = ArgumentSCC[i]->Definition;
      A->addAttr(AttributeSet::get(A->getContext(), A->getArgNo() + 1, B));
      ++NumNoCapture;
      Changed = true;
    }

    // We also want to compute readonly/readnone. With a small number of false
    // negatives, we can assume that any pointer which is captured isn't going
    // to be provably readonly or readnone, since by definition we can't
    // analyze all uses of a captured pointer.
    //
    // The false negatives happen when the pointer is captured by a function
    // that promises readonly/readnone behaviour on the pointer, then the
    // pointer's lifetime ends before anything that writes to arbitrary memory.
    // Also, a readonly/readnone pointer may be returned, but returning a
    // pointer is capturing it.

    Attribute::AttrKind ReadAttr = Attribute::ReadNone;
    for (unsigned i = 0, e = ArgumentSCC.size(); i != e; ++i) {
      Argument *A = ArgumentSCC[i]->Definition;
      Attribute::AttrKind K = determinePointerReadAttrs(A, ArgumentSCCNodes);
      if (K == Attribute::ReadNone)
        continue;
      if (K == Attribute::ReadOnly) {
        ReadAttr = Attribute::ReadOnly;
        continue;
      }
      ReadAttr = K;
      break;
    }

    if (ReadAttr != Attribute::None) {
      AttrBuilder B, R;
      B.addAttribute(ReadAttr);
      R.addAttribute(Attribute::ReadOnly)
        .addAttribute(Attribute::ReadNone);
      for (unsigned i = 0, e = ArgumentSCC.size(); i != e; ++i) {
        Argument *A = ArgumentSCC[i]->Definition;
        // Clear out existing readonly/readnone attributes
        A->removeAttr(AttributeSet::get(A->getContext(), A->getArgNo() + 1, R));
        A->addAttr(AttributeSet::get(A->getContext(), A->getArgNo() + 1, B));
        ReadAttr == Attribute::ReadOnly ? ++NumReadOnlyArg : ++NumReadNoneArg;
        Changed = true;
      }
    }
  }

  return Changed;
}

/// IsFunctionMallocLike - A function is malloc-like if it returns either null
/// or a pointer that doesn't alias any other pointer visible to the caller.
bool FunctionAttrs::IsFunctionMallocLike(Function *F,
                              SmallPtrSet<Function*, 8> &SCCNodes) const {
  SmallSetVector<Value *, 8> FlowsToReturn;
  for (Function::iterator I = F->begin(), E = F->end(); I != E; ++I)
    if (ReturnInst *Ret = dyn_cast<ReturnInst>(I->getTerminator()))
      FlowsToReturn.insert(Ret->getReturnValue());

  for (unsigned i = 0; i != FlowsToReturn.size(); ++i) {
    Value *RetVal = FlowsToReturn[i];

    if (Constant *C = dyn_cast<Constant>(RetVal)) {
      if (!C->isNullValue() && !isa<UndefValue>(C))
        return false;

      continue;
    }

    if (isa<Argument>(RetVal))
      return false;

    if (Instruction *RVI = dyn_cast<Instruction>(RetVal))
      switch (RVI->getOpcode()) {
        // Extend the analysis by looking upwards.
        case Instruction::BitCast:
        case Instruction::GetElementPtr:
        case Instruction::AddrSpaceCast:
          FlowsToReturn.insert(RVI->getOperand(0));
          continue;
        case Instruction::Select: {
          SelectInst *SI = cast<SelectInst>(RVI);
          FlowsToReturn.insert(SI->getTrueValue());
          FlowsToReturn.insert(SI->getFalseValue());
          continue;
        }
        case Instruction::PHI: {
          PHINode *PN = cast<PHINode>(RVI);
          for (Value *IncValue : PN->incoming_values())
            FlowsToReturn.insert(IncValue);
          continue;
        }

        // Check whether the pointer came from an allocation.
        case Instruction::Alloca:
          break;
        case Instruction::Call:
        case Instruction::Invoke: {
          CallSite CS(RVI);
          if (CS.paramHasAttr(0, Attribute::NoAlias))
            break;
          if (CS.getCalledFunction() &&
              SCCNodes.count(CS.getCalledFunction()))
            break;
        } // fall-through
        default:
          return false;  // Did not come from an allocation.
      }

    if (PointerMayBeCaptured(RetVal, false, /*StoreCaptures=*/false))
      return false;
  }

  return true;
}

/// AddNoAliasAttrs - Deduce noalias attributes for the SCC.
bool FunctionAttrs::AddNoAliasAttrs(const CallGraphSCC &SCC) {
  SmallPtrSet<Function*, 8> SCCNodes;

  // Fill SCCNodes with the elements of the SCC.  Used for quickly
  // looking up whether a given CallGraphNode is in this SCC.
  for (CallGraphSCC::iterator I = SCC.begin(), E = SCC.end(); I != E; ++I)
    SCCNodes.insert((*I)->getFunction());

  // Check each function in turn, determining which functions return noalias
  // pointers.
  for (CallGraphSCC::iterator I = SCC.begin(), E = SCC.end(); I != E; ++I) {
    Function *F = (*I)->getFunction();

    if (!F || F->hasFnAttribute(Attribute::OptimizeNone))
      // External node or node we don't want to optimize - skip it;
      return false;

    // Already noalias.
    if (F->doesNotAlias(0))
      continue;

    // Definitions with weak linkage may be overridden at linktime, so
    // treat them like declarations.
    if (F->isDeclaration() || F->mayBeOverridden())
      return false;

    // We annotate noalias return values, which are only applicable to 
    // pointer types.
    if (!F->getReturnType()->isPointerTy())
      continue;

    if (!IsFunctionMallocLike(F, SCCNodes))
      return false;
  }

  bool MadeChange = false;
  for (CallGraphSCC::iterator I = SCC.begin(), E = SCC.end(); I != E; ++I) {
    Function *F = (*I)->getFunction();
    if (F->doesNotAlias(0) || !F->getReturnType()->isPointerTy())
      continue;

    F->setDoesNotAlias(0);
    ++NumNoAlias;
    MadeChange = true;
  }

  return MadeChange;
}

/// inferPrototypeAttributes - Analyze the name and prototype of the
/// given function and set any applicable attributes.  Returns true
/// if any attributes were set and false otherwise.
bool FunctionAttrs::inferPrototypeAttributes(Function &F) {
  if (F.hasFnAttribute(Attribute::OptimizeNone))
    return false;

  FunctionType *FTy = F.getFunctionType();
  LibFunc::Func TheLibFunc;
  if (!(TLI->getLibFunc(F.getName(), TheLibFunc) && TLI->has(TheLibFunc)))
    return false;

  switch (TheLibFunc) {
  case LibFunc::strlen:
    if (FTy->getNumParams() != 1 || !FTy->getParamType(0)->isPointerTy())
      return false;
    setOnlyReadsMemory(F);
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    break;
  case LibFunc::strchr:
  case LibFunc::strrchr:
    if (FTy->getNumParams() != 2 ||
        !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(1)->isIntegerTy())
      return false;
    setOnlyReadsMemory(F);
    setDoesNotThrow(F);
    break;
  case LibFunc::strtol:
  case LibFunc::strtod:
  case LibFunc::strtof:
  case LibFunc::strtoul:
  case LibFunc::strtoll:
  case LibFunc::strtold:
  case LibFunc::strtoull:
    if (FTy->getNumParams() < 2 ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 2);
    setOnlyReadsMemory(F, 1);
    break;
  case LibFunc::strcpy:
  case LibFunc::stpcpy:
  case LibFunc::strcat:
  case LibFunc::strncat:
  case LibFunc::strncpy:
  case LibFunc::stpncpy:
    if (FTy->getNumParams() < 2 ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 2);
    setOnlyReadsMemory(F, 2);
    break;
  case LibFunc::strxfrm:
    if (FTy->getNumParams() != 3 ||
        !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    setDoesNotCapture(F, 2);
    setOnlyReadsMemory(F, 2);
    break;
  case LibFunc::strcmp: //0,1
    case LibFunc::strspn: // 0,1
    case LibFunc::strncmp: // 0,1
    case LibFunc::strcspn: //0,1
    case LibFunc::strcoll: //0,1
    case LibFunc::strcasecmp:  // 0,1
    case LibFunc::strncasecmp: // 
    if (FTy->getNumParams() < 2 ||
        !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    setOnlyReadsMemory(F);
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    setDoesNotCapture(F, 2);
    break;
  case LibFunc::strstr:
  case LibFunc::strpbrk:
    if (FTy->getNumParams() != 2 || !FTy->getParamType(1)->isPointerTy())
      return false;
    setOnlyReadsMemory(F);
    setDoesNotThrow(F);
    setDoesNotCapture(F, 2);
    break;
  case LibFunc::strtok:
  case LibFunc::strtok_r:
    if (FTy->getNumParams() < 2 || !FTy->getParamType(1)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 2);
    setOnlyReadsMemory(F, 2);
    break;
  case LibFunc::scanf:
    if (FTy->getNumParams() < 1 || !FTy->getParamType(0)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    setOnlyReadsMemory(F, 1);
    break;
  case LibFunc::setbuf:
  case LibFunc::setvbuf:
    if (FTy->getNumParams() < 1 || !FTy->getParamType(0)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    break;
  case LibFunc::strdup:
  case LibFunc::strndup:
    if (FTy->getNumParams() < 1 || !FTy->getReturnType()->isPointerTy() ||
        !FTy->getParamType(0)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotAlias(F, 0);
    setDoesNotCapture(F, 1);
    setOnlyReadsMemory(F, 1);
    break;
  case LibFunc::stat:
  case LibFunc::statvfs:
    if (FTy->getNumParams() < 2 ||
        !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    setDoesNotCapture(F, 2);
    setOnlyReadsMemory(F, 1);
    break;
  case LibFunc::sscanf:
    if (FTy->getNumParams() < 2 ||
        !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    setDoesNotCapture(F, 2);
    setOnlyReadsMemory(F, 1);
    setOnlyReadsMemory(F, 2);
    break;
  case LibFunc::sprintf:
    if (FTy->getNumParams() < 2 ||
        !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    setDoesNotCapture(F, 2);
    setOnlyReadsMemory(F, 2);
    break;
  case LibFunc::snprintf:
    if (FTy->getNumParams() != 3 ||
        !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(2)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    setDoesNotCapture(F, 3);
    setOnlyReadsMemory(F, 3);
    break;
  case LibFunc::setitimer:
    if (FTy->getNumParams() != 3 ||
        !FTy->getParamType(1)->isPointerTy() ||
        !FTy->getParamType(2)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 2);
    setDoesNotCapture(F, 3);
    setOnlyReadsMemory(F, 2);
    break;
  case LibFunc::system:
    if (FTy->getNumParams() != 1 ||
        !FTy->getParamType(0)->isPointerTy())
      return false;
    // May throw; "system" is a valid pthread cancellation point.
    setDoesNotCapture(F, 1);
    setOnlyReadsMemory(F, 1);
    break;
  case LibFunc::malloc:
    if (FTy->getNumParams() != 1 ||
        !FTy->getReturnType()->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotAlias(F, 0);
    break;
  case LibFunc::memcmp:
    if (FTy->getNumParams() != 3 ||
        !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    setOnlyReadsMemory(F);
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    setDoesNotCapture(F, 2);
    break;
  case LibFunc::memchr:
  case LibFunc::memrchr:
    if (FTy->getNumParams() != 3)
      return false;
    setOnlyReadsMemory(F);
    setDoesNotThrow(F);
    break;
  case LibFunc::modf:
  case LibFunc::modff:
  case LibFunc::modfl:
    if (FTy->getNumParams() < 2 ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 2);
    break;
  case LibFunc::memcpy:
  case LibFunc::memccpy:
  case LibFunc::memmove:
    if (FTy->getNumParams() < 2 ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 2);
    setOnlyReadsMemory(F, 2);
    break;
  case LibFunc::memalign:
    if (!FTy->getReturnType()->isPointerTy())
      return false;
    setDoesNotAlias(F, 0);
    break;
  case LibFunc::mkdir:
    if (FTy->getNumParams() == 0 ||
        !FTy->getParamType(0)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    setOnlyReadsMemory(F, 1);
    break;
  case LibFunc::mktime:
    if (FTy->getNumParams() == 0 ||
        !FTy->getParamType(0)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    break;
  case LibFunc::realloc:
    if (FTy->getNumParams() != 2 ||
        !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getReturnType()->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotAlias(F, 0);
    setDoesNotCapture(F, 1);
    break;
  case LibFunc::read:
    if (FTy->getNumParams() != 3 ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    // May throw; "read" is a valid pthread cancellation point.
    setDoesNotCapture(F, 2);
    break;
  case LibFunc::rewind:
    if (FTy->getNumParams() < 1 ||
        !FTy->getParamType(0)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    break;
  case LibFunc::rmdir:
  case LibFunc::remove:
  case LibFunc::realpath:
    if (FTy->getNumParams() < 1 ||
        !FTy->getParamType(0)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    setOnlyReadsMemory(F, 1);
    break;
  case LibFunc::rename:
    if (FTy->getNumParams() < 2 ||
        !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    setDoesNotCapture(F, 2);
    setOnlyReadsMemory(F, 1);
    setOnlyReadsMemory(F, 2);
    break;
  case LibFunc::readlink:
    if (FTy->getNumParams() < 2 ||
        !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    setDoesNotCapture(F, 2);
    setOnlyReadsMemory(F, 1);
    break;
  case LibFunc::write:
    if (FTy->getNumParams() != 3 || !FTy->getParamType(1)->isPointerTy())
      return false;
    // May throw; "write" is a valid pthread cancellation point.
    setDoesNotCapture(F, 2);
    setOnlyReadsMemory(F, 2);
    break;
  case LibFunc::bcopy:
    if (FTy->getNumParams() != 3 ||
        !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    setDoesNotCapture(F, 2);
    setOnlyReadsMemory(F, 1);
    break;
  case LibFunc::bcmp:
    if (FTy->getNumParams() != 3 ||
        !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setOnlyReadsMemory(F);
    setDoesNotCapture(F, 1);
    setDoesNotCapture(F, 2);
    break;
  case LibFunc::bzero:
    if (FTy->getNumParams() != 2 || !FTy->getParamType(0)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    break;
  case LibFunc::calloc:
    if (FTy->getNumParams() != 2 ||
        !FTy->getReturnType()->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotAlias(F, 0);
    break;
  case LibFunc::chmod:
  case LibFunc::chown:
    if (FTy->getNumParams() == 0 || !FTy->getParamType(0)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    setOnlyReadsMemory(F, 1);
    break;
  case LibFunc::ctermid:
  case LibFunc::clearerr:
  case LibFunc::closedir:
    if (FTy->getNumParams() == 0 || !FTy->getParamType(0)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    break;
  case LibFunc::atoi:
  case LibFunc::atol:
  case LibFunc::atof:
  case LibFunc::atoll:
    if (FTy->getNumParams() != 1 || !FTy->getParamType(0)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setOnlyReadsMemory(F);
    setDoesNotCapture(F, 1);
    break;
  case LibFunc::access:
    if (FTy->getNumParams() != 2 || !FTy->getParamType(0)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    setOnlyReadsMemory(F, 1);
    break;
  case LibFunc::fopen:
    if (FTy->getNumParams() != 2 ||
        !FTy->getReturnType()->isPointerTy() ||
        !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotAlias(F, 0);
    setDoesNotCapture(F, 1);
    setDoesNotCapture(F, 2);
    setOnlyReadsMemory(F, 1);
    setOnlyReadsMemory(F, 2);
    break;
  case LibFunc::fdopen:
    if (FTy->getNumParams() != 2 ||
        !FTy->getReturnType()->isPointerTy() ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotAlias(F, 0);
    setDoesNotCapture(F, 2);
    setOnlyReadsMemory(F, 2);
    break;
  case LibFunc::feof:
  case LibFunc::free:
  case LibFunc::fseek:
  case LibFunc::ftell:
  case LibFunc::fgetc:
  case LibFunc::fseeko:
  case LibFunc::ftello:
  case LibFunc::fileno:
  case LibFunc::fflush:
  case LibFunc::fclose:
  case LibFunc::fsetpos:
  case LibFunc::flockfile:
  case LibFunc::funlockfile:
  case LibFunc::ftrylockfile:
    if (FTy->getNumParams() == 0 || !FTy->getParamType(0)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    break;
  case LibFunc::ferror:
    if (FTy->getNumParams() != 1 || !FTy->getParamType(0)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    setOnlyReadsMemory(F);
    break;
  case LibFunc::fputc:
  case LibFunc::fstat:
  case LibFunc::frexp:
  case LibFunc::frexpf:
  case LibFunc::frexpl:
  case LibFunc::fstatvfs:
    if (FTy->getNumParams() != 2 || !FTy->getParamType(1)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 2);
    break;
  case LibFunc::fgets:
    if (FTy->getNumParams() != 3 ||
        !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(2)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 3);
    break;
  case LibFunc::fread:
    if (FTy->getNumParams() != 4 ||
        !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(3)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    setDoesNotCapture(F, 4);
    break;
  case LibFunc::fwrite:
    if (FTy->getNumParams() != 4 ||
        !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(3)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    setDoesNotCapture(F, 4);
    break;
  case LibFunc::fputs:
    if (FTy->getNumParams() < 2 ||
        !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    setDoesNotCapture(F, 2);
    setOnlyReadsMemory(F, 1);
    break;
  case LibFunc::fscanf:
  case LibFunc::fprintf:
    if (FTy->getNumParams() < 2 ||
        !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    setDoesNotCapture(F, 2);
    setOnlyReadsMemory(F, 2);
    break;
  case LibFunc::fgetpos:
    if (FTy->getNumParams() < 2 ||
        !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    setDoesNotCapture(F, 2);
    break;
  case LibFunc::getc:
  case LibFunc::getlogin_r:
  case LibFunc::getc_unlocked:
    if (FTy->getNumParams() == 0 || !FTy->getParamType(0)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    break;
  case LibFunc::getenv:
    if (FTy->getNumParams() != 1 || !FTy->getParamType(0)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setOnlyReadsMemory(F);
    setDoesNotCapture(F, 1);
    break;
  case LibFunc::gets:
  case LibFunc::getchar:
    setDoesNotThrow(F);
    break;
  case LibFunc::getitimer:
    if (FTy->getNumParams() != 2 || !FTy->getParamType(1)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 2);
    break;
  case LibFunc::getpwnam:
    if (FTy->getNumParams() != 1 || !FTy->getParamType(0)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    setOnlyReadsMemory(F, 1);
    break;
  case LibFunc::ungetc:
    if (FTy->getNumParams() != 2 || !FTy->getParamType(1)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 2);
    break;
  case LibFunc::uname:
    if (FTy->getNumParams() != 1 || !FTy->getParamType(0)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    break;
  case LibFunc::unlink:
    if (FTy->getNumParams() != 1 || !FTy->getParamType(0)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    setOnlyReadsMemory(F, 1);
    break;
  case LibFunc::unsetenv:
    if (FTy->getNumParams() != 1 || !FTy->getParamType(0)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    setOnlyReadsMemory(F, 1);
    break;
  case LibFunc::utime:
  case LibFunc::utimes:
    if (FTy->getNumParams() != 2 ||
        !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    setDoesNotCapture(F, 2);
    setOnlyReadsMemory(F, 1);
    setOnlyReadsMemory(F, 2);
    break;
  case LibFunc::putc:
    if (FTy->getNumParams() != 2 || !FTy->getParamType(1)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 2);
    break;
  case LibFunc::puts:
  case LibFunc::printf:
  case LibFunc::perror:
    if (FTy->getNumParams() != 1 || !FTy->getParamType(0)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    setOnlyReadsMemory(F, 1);
    break;
  case LibFunc::pread:
    if (FTy->getNumParams() != 4 || !FTy->getParamType(1)->isPointerTy())
      return false;
    // May throw; "pread" is a valid pthread cancellation point.
    setDoesNotCapture(F, 2);
    break;
  case LibFunc::pwrite:
    if (FTy->getNumParams() != 4 || !FTy->getParamType(1)->isPointerTy())
      return false;
    // May throw; "pwrite" is a valid pthread cancellation point.
    setDoesNotCapture(F, 2);
    setOnlyReadsMemory(F, 2);
    break;
  case LibFunc::putchar:
    setDoesNotThrow(F);
    break;
  case LibFunc::popen:
    if (FTy->getNumParams() != 2 ||
        !FTy->getReturnType()->isPointerTy() ||
        !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotAlias(F, 0);
    setDoesNotCapture(F, 1);
    setDoesNotCapture(F, 2);
    setOnlyReadsMemory(F, 1);
    setOnlyReadsMemory(F, 2);
    break;
  case LibFunc::pclose:
    if (FTy->getNumParams() != 1 || !FTy->getParamType(0)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    break;
  case LibFunc::vscanf:
    if (FTy->getNumParams() != 2 || !FTy->getParamType(1)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    setOnlyReadsMemory(F, 1);
    break;
  case LibFunc::vsscanf:
    if (FTy->getNumParams() != 3 ||
        !FTy->getParamType(1)->isPointerTy() ||
        !FTy->getParamType(2)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    setDoesNotCapture(F, 2);
    setOnlyReadsMemory(F, 1);
    setOnlyReadsMemory(F, 2);
    break;
  case LibFunc::vfscanf:
    if (FTy->getNumParams() != 3 ||
        !FTy->getParamType(1)->isPointerTy() ||
        !FTy->getParamType(2)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    setDoesNotCapture(F, 2);
    setOnlyReadsMemory(F, 2);
    break;
  case LibFunc::valloc:
    if (!FTy->getReturnType()->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotAlias(F, 0);
    break;
  case LibFunc::vprintf:
    if (FTy->getNumParams() != 2 || !FTy->getParamType(0)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    setOnlyReadsMemory(F, 1);
    break;
  case LibFunc::vfprintf:
  case LibFunc::vsprintf:
    if (FTy->getNumParams() != 3 ||
        !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    setDoesNotCapture(F, 2);
    setOnlyReadsMemory(F, 2);
    break;
  case LibFunc::vsnprintf:
    if (FTy->getNumParams() != 4 ||
        !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(2)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    setDoesNotCapture(F, 3);
    setOnlyReadsMemory(F, 3);
    break;
  case LibFunc::open:
    if (FTy->getNumParams() < 2 || !FTy->getParamType(0)->isPointerTy())
      return false;
    // May throw; "open" is a valid pthread cancellation point.
    setDoesNotCapture(F, 1);
    setOnlyReadsMemory(F, 1);
    break;
  case LibFunc::opendir:
    if (FTy->getNumParams() != 1 ||
        !FTy->getReturnType()->isPointerTy() ||
        !FTy->getParamType(0)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotAlias(F, 0);
    setDoesNotCapture(F, 1);
    setOnlyReadsMemory(F, 1);
    break;
  case LibFunc::tmpfile:
    if (!FTy->getReturnType()->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotAlias(F, 0);
    break;
  case LibFunc::times:
    if (FTy->getNumParams() != 1 || !FTy->getParamType(0)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    break;
  case LibFunc::htonl:
  case LibFunc::htons:
  case LibFunc::ntohl:
  case LibFunc::ntohs:
    setDoesNotThrow(F);
    setDoesNotAccessMemory(F);
    break;
  case LibFunc::lstat:
    if (FTy->getNumParams() != 2 ||
        !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    setDoesNotCapture(F, 2);
    setOnlyReadsMemory(F, 1);
    break;
  case LibFunc::lchown:
    if (FTy->getNumParams() != 3 || !FTy->getParamType(0)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    setOnlyReadsMemory(F, 1);
    break;
  case LibFunc::qsort:
    if (FTy->getNumParams() != 4 || !FTy->getParamType(3)->isPointerTy())
      return false;
    // May throw; places call through function pointer.
    setDoesNotCapture(F, 4);
    break;
  case LibFunc::dunder_strdup:
  case LibFunc::dunder_strndup:
    if (FTy->getNumParams() < 1 ||
        !FTy->getReturnType()->isPointerTy() ||
        !FTy->getParamType(0)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotAlias(F, 0);
    setDoesNotCapture(F, 1);
    setOnlyReadsMemory(F, 1);
    break;
  case LibFunc::dunder_strtok_r:
    if (FTy->getNumParams() != 3 ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 2);
    setOnlyReadsMemory(F, 2);
    break;
  case LibFunc::under_IO_getc:
    if (FTy->getNumParams() != 1 || !FTy->getParamType(0)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    break;
  case LibFunc::under_IO_putc:
    if (FTy->getNumParams() != 2 || !FTy->getParamType(1)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 2);
    break;
  case LibFunc::dunder_isoc99_scanf:
    if (FTy->getNumParams() < 1 ||
        !FTy->getParamType(0)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    setOnlyReadsMemory(F, 1);
    break;
  case LibFunc::stat64:
  case LibFunc::lstat64:
  case LibFunc::statvfs64:
    if (FTy->getNumParams() < 1 ||
        !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    setDoesNotCapture(F, 2);
    setOnlyReadsMemory(F, 1);
    break;
  case LibFunc::dunder_isoc99_sscanf:
    if (FTy->getNumParams() < 1 ||
        !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    setDoesNotCapture(F, 2);
    setOnlyReadsMemory(F, 1);
    setOnlyReadsMemory(F, 2);
    break;
#if 0 // HLSL Change Starts - Exclude potentially duplicate 64bit versions
  case LibFunc::fopen64:
    if (FTy->getNumParams() != 2 ||
        !FTy->getReturnType()->isPointerTy() ||
        !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotAlias(F, 0);
    setDoesNotCapture(F, 1);
    setDoesNotCapture(F, 2);
    setOnlyReadsMemory(F, 1);
    setOnlyReadsMemory(F, 2);
    break;
  case LibFunc::fseeko64:
  case LibFunc::ftello64:
    if (FTy->getNumParams() == 0 || !FTy->getParamType(0)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    break;
  case LibFunc::tmpfile64:
    if (!FTy->getReturnType()->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotAlias(F, 0);
    break;
#endif // HLSL Change Ends - Exclude potentially duplicate 64bit versions
  case LibFunc::fstat64:
  case LibFunc::fstatvfs64:
    if (FTy->getNumParams() != 2 || !FTy->getParamType(1)->isPointerTy())
      return false;
    setDoesNotThrow(F);
    setDoesNotCapture(F, 2);
    break;
  case LibFunc::open64:
    if (FTy->getNumParams() < 2 || !FTy->getParamType(0)->isPointerTy())
      return false;
    // May throw; "open" is a valid pthread cancellation point.
    setDoesNotCapture(F, 1);
    setOnlyReadsMemory(F, 1);
    break;
  case LibFunc::gettimeofday:
    if (FTy->getNumParams() != 2 || !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    // Currently some platforms have the restrict keyword on the arguments to
    // gettimeofday. To be conservative, do not add noalias to gettimeofday's
    // arguments.
    setDoesNotThrow(F);
    setDoesNotCapture(F, 1);
    setDoesNotCapture(F, 2);
    break;
  default:
    // Didn't mark any attributes.
    return false;
  }

  return true;
}

/// annotateLibraryCalls - Adds attributes to well-known standard library
/// call declarations.
bool FunctionAttrs::annotateLibraryCalls(const CallGraphSCC &SCC) {
  bool MadeChange = false;

  // Check each function in turn annotating well-known library function
  // declarations with attributes.
  for (CallGraphSCC::iterator I = SCC.begin(), E = SCC.end(); I != E; ++I) {
    Function *F = (*I)->getFunction();

    if (F && F->isDeclaration())
      MadeChange |= inferPrototypeAttributes(*F);
  }

  return MadeChange;
}

bool FunctionAttrs::runOnSCC(CallGraphSCC &SCC) {
  AA = &getAnalysis<AliasAnalysis>();
  TLI = &getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();

  bool Changed = annotateLibraryCalls(SCC);
  Changed |= AddReadAttrs(SCC);
  Changed |= AddArgumentAttrs(SCC);
  Changed |= AddNoAliasAttrs(SCC);
  return Changed;
}
