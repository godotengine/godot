//===- EarlyCSE.cpp - Simple and fast CSE pass ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass performs a simple dominator tree walk that eliminates trivially
// redundant instructions.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/RecyclingAllocator.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Local.h"
#include <deque>
using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "early-cse"

STATISTIC(NumSimplify, "Number of instructions simplified or DCE'd");
STATISTIC(NumCSE,      "Number of instructions CSE'd");
STATISTIC(NumCSELoad,  "Number of load instructions CSE'd");
STATISTIC(NumCSECall,  "Number of call instructions CSE'd");
STATISTIC(NumDSE,      "Number of trivial dead stores removed");

//===----------------------------------------------------------------------===//
// SimpleValue
//===----------------------------------------------------------------------===//

namespace {
/// \brief Struct representing the available values in the scoped hash table.
struct SimpleValue {
  Instruction *Inst;

  SimpleValue(Instruction *I) : Inst(I) {
    assert((isSentinel() || canHandle(I)) && "Inst can't be handled!");
  }

  bool isSentinel() const {
    return Inst == DenseMapInfo<Instruction *>::getEmptyKey() ||
           Inst == DenseMapInfo<Instruction *>::getTombstoneKey();
  }

  static bool canHandle(Instruction *Inst) {
    // This can only handle non-void readnone functions.
    if (CallInst *CI = dyn_cast<CallInst>(Inst))
      return CI->doesNotAccessMemory() && !CI->getType()->isVoidTy();
    return isa<CastInst>(Inst) || isa<BinaryOperator>(Inst) ||
           isa<GetElementPtrInst>(Inst) || isa<CmpInst>(Inst) ||
           isa<SelectInst>(Inst) || isa<ExtractElementInst>(Inst) ||
           isa<InsertElementInst>(Inst) || isa<ShuffleVectorInst>(Inst) ||
           isa<ExtractValueInst>(Inst) || isa<InsertValueInst>(Inst);
  }
};
}

namespace llvm {
template <> struct DenseMapInfo<SimpleValue> {
  static inline SimpleValue getEmptyKey() {
    return DenseMapInfo<Instruction *>::getEmptyKey();
  }
  static inline SimpleValue getTombstoneKey() {
    return DenseMapInfo<Instruction *>::getTombstoneKey();
  }
  static unsigned getHashValue(SimpleValue Val);
  static bool isEqual(SimpleValue LHS, SimpleValue RHS);
};
}

unsigned DenseMapInfo<SimpleValue>::getHashValue(SimpleValue Val) {
  Instruction *Inst = Val.Inst;
  // Hash in all of the operands as pointers.
  if (BinaryOperator *BinOp = dyn_cast<BinaryOperator>(Inst)) {
    Value *LHS = BinOp->getOperand(0);
    Value *RHS = BinOp->getOperand(1);
    if (BinOp->isCommutative() && BinOp->getOperand(0) > BinOp->getOperand(1))
      std::swap(LHS, RHS);

    if (isa<OverflowingBinaryOperator>(BinOp)) {
      // Hash the overflow behavior
      unsigned Overflow =
          BinOp->hasNoSignedWrap() * OverflowingBinaryOperator::NoSignedWrap |
          BinOp->hasNoUnsignedWrap() *
              OverflowingBinaryOperator::NoUnsignedWrap;
      return hash_combine(BinOp->getOpcode(), Overflow, LHS, RHS);
    }

    return hash_combine(BinOp->getOpcode(), LHS, RHS);
  }

  if (CmpInst *CI = dyn_cast<CmpInst>(Inst)) {
    Value *LHS = CI->getOperand(0);
    Value *RHS = CI->getOperand(1);
    CmpInst::Predicate Pred = CI->getPredicate();
    if (Inst->getOperand(0) > Inst->getOperand(1)) {
      std::swap(LHS, RHS);
      Pred = CI->getSwappedPredicate();
    }
    return hash_combine(Inst->getOpcode(), Pred, LHS, RHS);
  }

  if (CastInst *CI = dyn_cast<CastInst>(Inst))
    return hash_combine(CI->getOpcode(), CI->getType(), CI->getOperand(0));

  if (const ExtractValueInst *EVI = dyn_cast<ExtractValueInst>(Inst))
    return hash_combine(EVI->getOpcode(), EVI->getOperand(0),
                        hash_combine_range(EVI->idx_begin(), EVI->idx_end()));

  if (const InsertValueInst *IVI = dyn_cast<InsertValueInst>(Inst))
    return hash_combine(IVI->getOpcode(), IVI->getOperand(0),
                        IVI->getOperand(1),
                        hash_combine_range(IVI->idx_begin(), IVI->idx_end()));

  assert((isa<CallInst>(Inst) || isa<BinaryOperator>(Inst) ||
          isa<GetElementPtrInst>(Inst) || isa<SelectInst>(Inst) ||
          isa<ExtractElementInst>(Inst) || isa<InsertElementInst>(Inst) ||
          isa<ShuffleVectorInst>(Inst)) &&
         "Invalid/unknown instruction");

  // Mix in the opcode.
  return hash_combine(
      Inst->getOpcode(),
      hash_combine_range(Inst->value_op_begin(), Inst->value_op_end()));
}

bool DenseMapInfo<SimpleValue>::isEqual(SimpleValue LHS, SimpleValue RHS) {
  Instruction *LHSI = LHS.Inst, *RHSI = RHS.Inst;

  if (LHS.isSentinel() || RHS.isSentinel())
    return LHSI == RHSI;

  if (LHSI->getOpcode() != RHSI->getOpcode())
    return false;
  if (LHSI->isIdenticalTo(RHSI))
    return true;

  // If we're not strictly identical, we still might be a commutable instruction
  if (BinaryOperator *LHSBinOp = dyn_cast<BinaryOperator>(LHSI)) {
    if (!LHSBinOp->isCommutative())
      return false;

    assert(isa<BinaryOperator>(RHSI) &&
           "same opcode, but different instruction type?");
    BinaryOperator *RHSBinOp = cast<BinaryOperator>(RHSI);

    // Check overflow attributes
    if (isa<OverflowingBinaryOperator>(LHSBinOp)) {
      assert(isa<OverflowingBinaryOperator>(RHSBinOp) &&
             "same opcode, but different operator type?");
      if (LHSBinOp->hasNoUnsignedWrap() != RHSBinOp->hasNoUnsignedWrap() ||
          LHSBinOp->hasNoSignedWrap() != RHSBinOp->hasNoSignedWrap())
        return false;
    }

    // Commuted equality
    return LHSBinOp->getOperand(0) == RHSBinOp->getOperand(1) &&
           LHSBinOp->getOperand(1) == RHSBinOp->getOperand(0);
  }
  if (CmpInst *LHSCmp = dyn_cast<CmpInst>(LHSI)) {
    assert(isa<CmpInst>(RHSI) &&
           "same opcode, but different instruction type?");
    CmpInst *RHSCmp = cast<CmpInst>(RHSI);
    // Commuted equality
    return LHSCmp->getOperand(0) == RHSCmp->getOperand(1) &&
           LHSCmp->getOperand(1) == RHSCmp->getOperand(0) &&
           LHSCmp->getSwappedPredicate() == RHSCmp->getPredicate();
  }

  return false;
}

//===----------------------------------------------------------------------===//
// CallValue
//===----------------------------------------------------------------------===//

namespace {
/// \brief Struct representing the available call values in the scoped hash
/// table.
struct CallValue {
  Instruction *Inst;

  CallValue(Instruction *I) : Inst(I) {
    assert((isSentinel() || canHandle(I)) && "Inst can't be handled!");
  }

  bool isSentinel() const {
    return Inst == DenseMapInfo<Instruction *>::getEmptyKey() ||
           Inst == DenseMapInfo<Instruction *>::getTombstoneKey();
  }

  static bool canHandle(Instruction *Inst) {
    // Don't value number anything that returns void.
    if (Inst->getType()->isVoidTy())
      return false;

    CallInst *CI = dyn_cast<CallInst>(Inst);
    if (!CI || !CI->onlyReadsMemory())
      return false;
    return true;
  }
};
}

namespace llvm {
template <> struct DenseMapInfo<CallValue> {
  static inline CallValue getEmptyKey() {
    return DenseMapInfo<Instruction *>::getEmptyKey();
  }
  static inline CallValue getTombstoneKey() {
    return DenseMapInfo<Instruction *>::getTombstoneKey();
  }
  static unsigned getHashValue(CallValue Val);
  static bool isEqual(CallValue LHS, CallValue RHS);
};
}

unsigned DenseMapInfo<CallValue>::getHashValue(CallValue Val) {
  Instruction *Inst = Val.Inst;
  // Hash all of the operands as pointers and mix in the opcode.
  return hash_combine(
      Inst->getOpcode(),
      hash_combine_range(Inst->value_op_begin(), Inst->value_op_end()));
}

bool DenseMapInfo<CallValue>::isEqual(CallValue LHS, CallValue RHS) {
  Instruction *LHSI = LHS.Inst, *RHSI = RHS.Inst;
  if (LHS.isSentinel() || RHS.isSentinel())
    return LHSI == RHSI;
  return LHSI->isIdenticalTo(RHSI);
}

//===----------------------------------------------------------------------===//
// EarlyCSE implementation
//===----------------------------------------------------------------------===//

namespace {
/// \brief A simple and fast domtree-based CSE pass.
///
/// This pass does a simple depth-first walk over the dominator tree,
/// eliminating trivially redundant instructions and using instsimplify to
/// canonicalize things as it goes. It is intended to be fast and catch obvious
/// cases so that instcombine and other passes are more effective. It is
/// expected that a later pass of GVN will catch the interesting/hard cases.
class EarlyCSE {
public:
  Function &F;
  const TargetLibraryInfo &TLI;
  const TargetTransformInfo &TTI;
  DominatorTree &DT;
  AssumptionCache &AC;
  typedef RecyclingAllocator<
      BumpPtrAllocator, ScopedHashTableVal<SimpleValue, Value *>> AllocatorTy;
  typedef ScopedHashTable<SimpleValue, Value *, DenseMapInfo<SimpleValue>,
                          AllocatorTy> ScopedHTType;

  /// \brief A scoped hash table of the current values of all of our simple
  /// scalar expressions.
  ///
  /// As we walk down the domtree, we look to see if instructions are in this:
  /// if so, we replace them with what we find, otherwise we insert them so
  /// that dominated values can succeed in their lookup.
  ScopedHTType AvailableValues;

  /// \brief A scoped hash table of the current values of loads.
  ///
  /// This allows us to get efficient access to dominating loads when we have
  /// a fully redundant load.  In addition to the most recent load, we keep
  /// track of a generation count of the read, which is compared against the
  /// current generation count.  The current generation count is incremented
  /// after every possibly writing memory operation, which ensures that we only
  /// CSE loads with other loads that have no intervening store.
  typedef RecyclingAllocator<
      BumpPtrAllocator,
      ScopedHashTableVal<Value *, std::pair<Value *, unsigned>>>
      LoadMapAllocator;
  typedef ScopedHashTable<Value *, std::pair<Value *, unsigned>,
                          DenseMapInfo<Value *>, LoadMapAllocator> LoadHTType;
  LoadHTType AvailableLoads;

  /// \brief A scoped hash table of the current values of read-only call
  /// values.
  ///
  /// It uses the same generation count as loads.
  typedef ScopedHashTable<CallValue, std::pair<Value *, unsigned>> CallHTType;
  CallHTType AvailableCalls;

  /// \brief This is the current generation of the memory value.
  unsigned CurrentGeneration;

  /// \brief Set up the EarlyCSE runner for a particular function.
  EarlyCSE(Function &F, const TargetLibraryInfo &TLI,
           const TargetTransformInfo &TTI, DominatorTree &DT,
           AssumptionCache &AC)
      : F(F), TLI(TLI), TTI(TTI), DT(DT), AC(AC), CurrentGeneration(0) {}

  bool run();

private:
  // Almost a POD, but needs to call the constructors for the scoped hash
  // tables so that a new scope gets pushed on. These are RAII so that the
  // scope gets popped when the NodeScope is destroyed.
  class NodeScope {
  public:
    NodeScope(ScopedHTType &AvailableValues, LoadHTType &AvailableLoads,
              CallHTType &AvailableCalls)
        : Scope(AvailableValues), LoadScope(AvailableLoads),
          CallScope(AvailableCalls) {}

  private:
    NodeScope(const NodeScope &) = delete;
    void operator=(const NodeScope &) = delete;

    ScopedHTType::ScopeTy Scope;
    LoadHTType::ScopeTy LoadScope;
    CallHTType::ScopeTy CallScope;
  };

  // Contains all the needed information to create a stack for doing a depth
  // first tranversal of the tree. This includes scopes for values, loads, and
  // calls as well as the generation. There is a child iterator so that the
  // children do not need to be store spearately.
  class StackNode {
  public:
    StackNode(ScopedHTType &AvailableValues, LoadHTType &AvailableLoads,
              CallHTType &AvailableCalls, unsigned cg, DomTreeNode *n,
              DomTreeNode::iterator child, DomTreeNode::iterator end)
        : CurrentGeneration(cg), ChildGeneration(cg), Node(n), ChildIter(child),
          EndIter(end), Scopes(AvailableValues, AvailableLoads, AvailableCalls),
          Processed(false) {}

    // Accessors.
    unsigned currentGeneration() { return CurrentGeneration; }
    unsigned childGeneration() { return ChildGeneration; }
    void childGeneration(unsigned generation) { ChildGeneration = generation; }
    DomTreeNode *node() { return Node; }
    DomTreeNode::iterator childIter() { return ChildIter; }
    DomTreeNode *nextChild() {
      DomTreeNode *child = *ChildIter;
      ++ChildIter;
      return child;
    }
    DomTreeNode::iterator end() { return EndIter; }
    bool isProcessed() { return Processed; }
    void process() { Processed = true; }

  private:
    StackNode(const StackNode &) = delete;
    void operator=(const StackNode &) = delete;

    // Members.
    unsigned CurrentGeneration;
    unsigned ChildGeneration;
    DomTreeNode *Node;
    DomTreeNode::iterator ChildIter;
    DomTreeNode::iterator EndIter;
    NodeScope Scopes;
    bool Processed;
  };

  /// \brief Wrapper class to handle memory instructions, including loads,
  /// stores and intrinsic loads and stores defined by the target.
  class ParseMemoryInst {
  public:
    ParseMemoryInst(Instruction *Inst, const TargetTransformInfo &TTI)
        : Load(false), Store(false), Vol(false), MayReadFromMemory(false),
          MayWriteToMemory(false), MatchingId(-1), Ptr(nullptr) {
      MayReadFromMemory = Inst->mayReadFromMemory();
      MayWriteToMemory = Inst->mayWriteToMemory();
      if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(Inst)) {
        MemIntrinsicInfo Info;
        if (!TTI.getTgtMemIntrinsic(II, Info))
          return;
        if (Info.NumMemRefs == 1) {
          Store = Info.WriteMem;
          Load = Info.ReadMem;
          MatchingId = Info.MatchingId;
          MayReadFromMemory = Info.ReadMem;
          MayWriteToMemory = Info.WriteMem;
          Vol = Info.Vol;
          Ptr = Info.PtrVal;
        }
      } else if (LoadInst *LI = dyn_cast<LoadInst>(Inst)) {
        Load = true;
        Vol = !LI->isSimple();
        Ptr = LI->getPointerOperand();
      } else if (StoreInst *SI = dyn_cast<StoreInst>(Inst)) {
        Store = true;
        Vol = !SI->isSimple();
        Ptr = SI->getPointerOperand();
      }
    }
    bool isLoad() { return Load; }
    bool isStore() { return Store; }
    bool isVolatile() { return Vol; }
    bool isMatchingMemLoc(const ParseMemoryInst &Inst) {
      return Ptr == Inst.Ptr && MatchingId == Inst.MatchingId;
    }
    bool isValid() { return Ptr != nullptr; }
    int getMatchingId() { return MatchingId; }
    Value *getPtr() { return Ptr; }
    bool mayReadFromMemory() { return MayReadFromMemory; }
    bool mayWriteToMemory() { return MayWriteToMemory; }

  private:
    bool Load;
    bool Store;
    bool Vol;
    bool MayReadFromMemory;
    bool MayWriteToMemory;
    // For regular (non-intrinsic) loads/stores, this is set to -1. For
    // intrinsic loads/stores, the id is retrieved from the corresponding
    // field in the MemIntrinsicInfo structure.  That field contains
    // non-negative values only.
    int MatchingId;
    Value *Ptr;
  };

  bool processNode(DomTreeNode *Node);

  Value *getOrCreateResult(Value *Inst, Type *ExpectedType) const {
    if (LoadInst *LI = dyn_cast<LoadInst>(Inst))
      return LI;
    else if (StoreInst *SI = dyn_cast<StoreInst>(Inst))
      return SI->getValueOperand();
    assert(isa<IntrinsicInst>(Inst) && "Instruction not supported");
    return TTI.getOrCreateResultFromMemIntrinsic(cast<IntrinsicInst>(Inst),
                                                 ExpectedType);
  }
};
}

bool EarlyCSE::processNode(DomTreeNode *Node) {
  BasicBlock *BB = Node->getBlock();

  // If this block has a single predecessor, then the predecessor is the parent
  // of the domtree node and all of the live out memory values are still current
  // in this block.  If this block has multiple predecessors, then they could
  // have invalidated the live-out memory values of our parent value.  For now,
  // just be conservative and invalidate memory if this block has multiple
  // predecessors.
  if (!BB->getSinglePredecessor())
    ++CurrentGeneration;

  // If this node has a single predecessor which ends in a conditional branch,
  // we can infer the value of the branch condition given that we took this
  // path.  We need the single predeccesor to ensure there's not another path
  // which reaches this block where the condition might hold a different
  // value.  Since we're adding this to the scoped hash table (like any other
  // def), it will have been popped if we encounter a future merge block.
  if (BasicBlock *Pred = BB->getSinglePredecessor())
    if (auto *BI = dyn_cast<BranchInst>(Pred->getTerminator()))
      if (BI->isConditional())
        if (auto *CondInst = dyn_cast<Instruction>(BI->getCondition()))
          if (SimpleValue::canHandle(CondInst)) {
            assert(BI->getSuccessor(0) == BB || BI->getSuccessor(1) == BB);
            auto *ConditionalConstant = (BI->getSuccessor(0) == BB) ?
              ConstantInt::getTrue(BB->getContext()) :
              ConstantInt::getFalse(BB->getContext());
            AvailableValues.insert(CondInst, ConditionalConstant);
            DEBUG(dbgs() << "EarlyCSE CVP: Add conditional value for '"
                  << CondInst->getName() << "' as " << *ConditionalConstant
                  << " in " << BB->getName() << "\n");
            // Replace all dominated uses with the known value
            replaceDominatedUsesWith(CondInst, ConditionalConstant, DT,
                                     BasicBlockEdge(Pred, BB));
          }

  /// LastStore - Keep track of the last non-volatile store that we saw... for
  /// as long as there in no instruction that reads memory.  If we see a store
  /// to the same location, we delete the dead store.  This zaps trivial dead
  /// stores which can occur in bitfield code among other things.
  Instruction *LastStore = nullptr;

  bool Changed = false;
  const DataLayout &DL = BB->getModule()->getDataLayout();

  // See if any instructions in the block can be eliminated.  If so, do it.  If
  // not, add them to AvailableValues.
  for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E;) {
    Instruction *Inst = I++;

    // Dead instructions should just be removed.
    if (isInstructionTriviallyDead(Inst, &TLI)) {
      DEBUG(dbgs() << "EarlyCSE DCE: " << *Inst << '\n');
      Inst->eraseFromParent();
      Changed = true;
      ++NumSimplify;
      continue;
    }

    // Skip assume intrinsics, they don't really have side effects (although
    // they're marked as such to ensure preservation of control dependencies),
    // and this pass will not disturb any of the assumption's control
    // dependencies.
    if (match(Inst, m_Intrinsic<Intrinsic::assume>())) {
      DEBUG(dbgs() << "EarlyCSE skipping assumption: " << *Inst << '\n');
      continue;
    }

    // If the instruction can be simplified (e.g. X+0 = X) then replace it with
    // its simpler value.
    if (Value *V = SimplifyInstruction(Inst, DL, &TLI, &DT, &AC)) {
      DEBUG(dbgs() << "EarlyCSE Simplify: " << *Inst << "  to: " << *V << '\n');
      Inst->replaceAllUsesWith(V);
      Inst->eraseFromParent();
      Changed = true;
      ++NumSimplify;
      continue;
    }

    // If this is a simple instruction that we can value number, process it.
    if (SimpleValue::canHandle(Inst)) {
      // See if the instruction has an available value.  If so, use it.
      if (Value *V = AvailableValues.lookup(Inst)) {
        DEBUG(dbgs() << "EarlyCSE CSE: " << *Inst << "  to: " << *V << '\n');
        Inst->replaceAllUsesWith(V);
        Inst->eraseFromParent();
        Changed = true;
        ++NumCSE;
        continue;
      }

      // Otherwise, just remember that this value is available.
      AvailableValues.insert(Inst, Inst);
      continue;
    }

    ParseMemoryInst MemInst(Inst, TTI);
    // If this is a non-volatile load, process it.
    if (MemInst.isValid() && MemInst.isLoad()) {
      // Ignore volatile loads.
      if (MemInst.isVolatile()) {
        LastStore = nullptr;
        // Don't CSE across synchronization boundaries.
        if (Inst->mayWriteToMemory())
          ++CurrentGeneration;
        continue;
      }

      // If we have an available version of this load, and if it is the right
      // generation, replace this instruction.
      std::pair<Value *, unsigned> InVal =
          AvailableLoads.lookup(MemInst.getPtr());
      if (InVal.first != nullptr && InVal.second == CurrentGeneration) {
        Value *Op = getOrCreateResult(InVal.first, Inst->getType());
        if (Op != nullptr) {
          DEBUG(dbgs() << "EarlyCSE CSE LOAD: " << *Inst
                       << "  to: " << *InVal.first << '\n');
          if (!Inst->use_empty())
            Inst->replaceAllUsesWith(Op);
          Inst->eraseFromParent();
          Changed = true;
          ++NumCSELoad;
          continue;
        }
      }

      // Otherwise, remember that we have this instruction.
      AvailableLoads.insert(MemInst.getPtr(), std::pair<Value *, unsigned>(
                                                  Inst, CurrentGeneration));
      LastStore = nullptr;
      continue;
    }

    // If this instruction may read from memory, forget LastStore.
    // Load/store intrinsics will indicate both a read and a write to
    // memory.  The target may override this (e.g. so that a store intrinsic
    // does not read  from memory, and thus will be treated the same as a
    // regular store for commoning purposes).
    if (Inst->mayReadFromMemory() &&
        !(MemInst.isValid() && !MemInst.mayReadFromMemory()))
      LastStore = nullptr;

    // If this is a read-only call, process it.
    if (CallValue::canHandle(Inst)) {
      // If we have an available version of this call, and if it is the right
      // generation, replace this instruction.
      std::pair<Value *, unsigned> InVal = AvailableCalls.lookup(Inst);
      if (InVal.first != nullptr && InVal.second == CurrentGeneration) {
        DEBUG(dbgs() << "EarlyCSE CSE CALL: " << *Inst
                     << "  to: " << *InVal.first << '\n');
        if (!Inst->use_empty())
          Inst->replaceAllUsesWith(InVal.first);
        Inst->eraseFromParent();
        Changed = true;
        ++NumCSECall;
        continue;
      }

      // Otherwise, remember that we have this instruction.
      AvailableCalls.insert(
          Inst, std::pair<Value *, unsigned>(Inst, CurrentGeneration));
      continue;
    }

    // Okay, this isn't something we can CSE at all.  Check to see if it is
    // something that could modify memory.  If so, our available memory values
    // cannot be used so bump the generation count.
    if (Inst->mayWriteToMemory()) {
      ++CurrentGeneration;

      if (MemInst.isValid() && MemInst.isStore()) {
        // We do a trivial form of DSE if there are two stores to the same
        // location with no intervening loads.  Delete the earlier store.
        if (LastStore) {
          ParseMemoryInst LastStoreMemInst(LastStore, TTI);
          if (LastStoreMemInst.isMatchingMemLoc(MemInst)) {
            DEBUG(dbgs() << "EarlyCSE DEAD STORE: " << *LastStore
                         << "  due to: " << *Inst << '\n');
            LastStore->eraseFromParent();
            Changed = true;
            ++NumDSE;
            LastStore = nullptr;
          }
          // fallthrough - we can exploit information about this store
        }

        // Okay, we just invalidated anything we knew about loaded values.  Try
        // to salvage *something* by remembering that the stored value is a live
        // version of the pointer.  It is safe to forward from volatile stores
        // to non-volatile loads, so we don't have to check for volatility of
        // the store.
        AvailableLoads.insert(MemInst.getPtr(), std::pair<Value *, unsigned>(
                                                    Inst, CurrentGeneration));

        // Remember that this was the last store we saw for DSE.
        if (!MemInst.isVolatile())
          LastStore = Inst;
      }
    }
  }

  return Changed;
}

bool EarlyCSE::run() {
  // Note, deque is being used here because there is significant performance
  // gains over vector when the container becomes very large due to the
  // specific access patterns. For more information see the mailing list
  // discussion on this:
  // http://lists.llvm.org/pipermail/llvm-commits/Week-of-Mon-20120116/135228.html
  std::deque<StackNode *> nodesToProcess;

  bool Changed = false;

  // Process the root node.
  nodesToProcess.push_back(new StackNode(
      AvailableValues, AvailableLoads, AvailableCalls, CurrentGeneration,
      DT.getRootNode(), DT.getRootNode()->begin(), DT.getRootNode()->end()));

  // Save the current generation.
  unsigned LiveOutGeneration = CurrentGeneration;

  // Process the stack.
  while (!nodesToProcess.empty()) {
    // Grab the first item off the stack. Set the current generation, remove
    // the node from the stack, and process it.
    StackNode *NodeToProcess = nodesToProcess.back();

    // Initialize class members.
    CurrentGeneration = NodeToProcess->currentGeneration();

    // Check if the node needs to be processed.
    if (!NodeToProcess->isProcessed()) {
      // Process the node.
      Changed |= processNode(NodeToProcess->node());
      NodeToProcess->childGeneration(CurrentGeneration);
      NodeToProcess->process();
    } else if (NodeToProcess->childIter() != NodeToProcess->end()) {
      // Push the next child onto the stack.
      DomTreeNode *child = NodeToProcess->nextChild();
      nodesToProcess.push_back(
          new StackNode(AvailableValues, AvailableLoads, AvailableCalls,
                        NodeToProcess->childGeneration(), child, child->begin(),
                        child->end()));
    } else {
      // It has been processed, and there are no more children to process,
      // so delete it and pop it off the stack.
      delete NodeToProcess;
      nodesToProcess.pop_back();
    }
  } // while (!nodes...)

  // Reset the current generation.
  CurrentGeneration = LiveOutGeneration;

  return Changed;
}

PreservedAnalyses EarlyCSEPass::run(Function &F,
                                    AnalysisManager<Function> *AM) {
  auto &TLI = AM->getResult<TargetLibraryAnalysis>(F);
  auto &TTI = AM->getResult<TargetIRAnalysis>(F);
  auto &DT = AM->getResult<DominatorTreeAnalysis>(F);
  auto &AC = AM->getResult<AssumptionAnalysis>(F);

  EarlyCSE CSE(F, TLI, TTI, DT, AC);

  if (!CSE.run())
    return PreservedAnalyses::all();

  // CSE preserves the dominator tree because it doesn't mutate the CFG.
  // FIXME: Bundle this with other CFG-preservation.
  PreservedAnalyses PA;
  PA.preserve<DominatorTreeAnalysis>();
  return PA;
}

namespace {
/// \brief A simple and fast domtree-based CSE pass.
///
/// This pass does a simple depth-first walk over the dominator tree,
/// eliminating trivially redundant instructions and using instsimplify to
/// canonicalize things as it goes. It is intended to be fast and catch obvious
/// cases so that instcombine and other passes are more effective. It is
/// expected that a later pass of GVN will catch the interesting/hard cases.
class EarlyCSELegacyPass : public FunctionPass {
public:
  static char ID;

  EarlyCSELegacyPass() : FunctionPass(ID) {
    initializeEarlyCSELegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    if (skipOptnoneFunction(F))
      return false;

    auto &TLI = getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();
    auto &TTI = getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
    auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    auto &AC = getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);

    EarlyCSE CSE(F, TLI, TTI, DT, AC);

    return CSE.run();
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.setPreservesCFG();
  }
};
}

char EarlyCSELegacyPass::ID = 0;

FunctionPass *llvm::createEarlyCSEPass() { return new EarlyCSELegacyPass(); }

INITIALIZE_PASS_BEGIN(EarlyCSELegacyPass, "early-cse", "Early CSE", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(EarlyCSELegacyPass, "early-cse", "Early CSE", false, false)
