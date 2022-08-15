//===-- StructurizeCFG.cpp ------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Analysis/RegionIterator.h"
#include "llvm/Analysis/RegionPass.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"

using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "structurizecfg"

namespace {

// Definition of the complex types used in this pass.

typedef std::pair<BasicBlock *, Value *> BBValuePair;

typedef SmallVector<RegionNode*, 8> RNVector;
typedef SmallVector<BasicBlock*, 8> BBVector;
typedef SmallVector<BranchInst*, 8> BranchVector;
typedef SmallVector<BBValuePair, 2> BBValueVector;

typedef SmallPtrSet<BasicBlock *, 8> BBSet;

typedef MapVector<PHINode *, BBValueVector> PhiMap;
typedef MapVector<BasicBlock *, BBVector> BB2BBVecMap;

typedef DenseMap<DomTreeNode *, unsigned> DTN2UnsignedMap;
typedef DenseMap<BasicBlock *, PhiMap> BBPhiMap;
typedef DenseMap<BasicBlock *, Value *> BBPredicates;
typedef DenseMap<BasicBlock *, BBPredicates> PredMap;
typedef DenseMap<BasicBlock *, BasicBlock*> BB2BBMap;

// The name for newly created blocks.

static const char *const FlowBlockName = "Flow";

/// @brief Find the nearest common dominator for multiple BasicBlocks
///
/// Helper class for StructurizeCFG
/// TODO: Maybe move into common code
class NearestCommonDominator {
  DominatorTree *DT;

  DTN2UnsignedMap IndexMap;

  BasicBlock *Result;
  unsigned ResultIndex;
  bool ExplicitMentioned;

public:
  /// \brief Start a new query
  NearestCommonDominator(DominatorTree *DomTree) {
    DT = DomTree;
    Result = nullptr;
  }

  /// \brief Add BB to the resulting dominator
  void addBlock(BasicBlock *BB, bool Remember = true) {
    DomTreeNode *Node = DT->getNode(BB);

    if (!Result) {
      unsigned Numbering = 0;
      for (;Node;Node = Node->getIDom())
        IndexMap[Node] = ++Numbering;
      Result = BB;
      ResultIndex = 1;
      ExplicitMentioned = Remember;
      return;
    }

    for (;Node;Node = Node->getIDom())
      if (IndexMap.count(Node))
        break;
      else
        IndexMap[Node] = 0;

    assert(Node && "Dominator tree invalid!");

    unsigned Numbering = IndexMap[Node];
    if (Numbering > ResultIndex) {
      Result = Node->getBlock();
      ResultIndex = Numbering;
      ExplicitMentioned = Remember && (Result == BB);
    } else if (Numbering == ResultIndex) {
      ExplicitMentioned |= Remember;
    }
  }

  /// \brief Is "Result" one of the BBs added with "Remember" = True?
  bool wasResultExplicitMentioned() {
    return ExplicitMentioned;
  }

  /// \brief Get the query result
  BasicBlock *getResult() {
    return Result;
  }
};

/// @brief Transforms the control flow graph on one single entry/exit region
/// at a time.
///
/// After the transform all "If"/"Then"/"Else" style control flow looks like
/// this:
///
/// \verbatim
/// 1
/// ||
/// | |
/// 2 |
/// | /
/// |/
/// 3
/// ||   Where:
/// | |  1 = "If" block, calculates the condition
/// 4 |  2 = "Then" subregion, runs if the condition is true
/// | /  3 = "Flow" blocks, newly inserted flow blocks, rejoins the flow
/// |/   4 = "Else" optional subregion, runs if the condition is false
/// 5    5 = "End" block, also rejoins the control flow
/// \endverbatim
///
/// Control flow is expressed as a branch where the true exit goes into the
/// "Then"/"Else" region, while the false exit skips the region
/// The condition for the optional "Else" region is expressed as a PHI node.
/// The incomming values of the PHI node are true for the "If" edge and false
/// for the "Then" edge.
///
/// Additionally to that even complicated loops look like this:
///
/// \verbatim
/// 1
/// ||
/// | |
/// 2 ^  Where:
/// | /  1 = "Entry" block
/// |/   2 = "Loop" optional subregion, with all exits at "Flow" block
/// 3    3 = "Flow" block, with back edge to entry block
/// |
/// \endverbatim
///
/// The back edge of the "Flow" block is always on the false side of the branch
/// while the true side continues the general flow. So the loop condition
/// consist of a network of PHI nodes where the true incoming values expresses
/// breaks and the false values expresses continue states.
class StructurizeCFG : public RegionPass {
  Type *Boolean;
  ConstantInt *BoolTrue;
  ConstantInt *BoolFalse;
  UndefValue *BoolUndef;

  Function *Func;
  Region *ParentRegion;

  DominatorTree *DT;
  LoopInfo *LI;

  RNVector Order;
  BBSet Visited;

  BBPhiMap DeletedPhis;
  BB2BBVecMap AddedPhis;

  PredMap Predicates;
  BranchVector Conditions;

  BB2BBMap Loops;
  PredMap LoopPreds;
  BranchVector LoopConds;

  RegionNode *PrevNode;

  void orderNodes();

  void analyzeLoops(RegionNode *N);

  Value *invert(Value *Condition);

  Value *buildCondition(BranchInst *Term, unsigned Idx, bool Invert);

  void gatherPredicates(RegionNode *N);

  void collectInfos();

  void insertConditions(bool Loops);

  void delPhiValues(BasicBlock *From, BasicBlock *To);

  void addPhiValues(BasicBlock *From, BasicBlock *To);

  void setPhiValues();

  void killTerminator(BasicBlock *BB);

  void changeExit(RegionNode *Node, BasicBlock *NewExit,
                  bool IncludeDominator);

  BasicBlock *getNextFlow(BasicBlock *Dominator);

  BasicBlock *needPrefix(bool NeedEmpty);

  BasicBlock *needPostfix(BasicBlock *Flow, bool ExitUseAllowed);

  void setPrevNode(BasicBlock *BB);

  bool dominatesPredicates(BasicBlock *BB, RegionNode *Node);

  bool isPredictableTrue(RegionNode *Node);

  void wireFlow(bool ExitUseAllowed, BasicBlock *LoopEnd);

  void handleLoops(bool ExitUseAllowed, BasicBlock *LoopEnd);

  void createFlow();

  void rebuildSSA();

public:
  static char ID;

  StructurizeCFG() :
    RegionPass(ID) {
    initializeStructurizeCFGPass(*PassRegistry::getPassRegistry());
  }

  using Pass::doInitialization;
  bool doInitialization(Region *R, RGPassManager &RGM) override;

  bool runOnRegion(Region *R, RGPassManager &RGM) override;

  StringRef getPassName() const override {
    return "Structurize control flow";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequiredID(LowerSwitchID);
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addPreserved<DominatorTreeWrapperPass>();
    RegionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

char StructurizeCFG::ID = 0;

INITIALIZE_PASS_BEGIN(StructurizeCFG, "structurizecfg", "Structurize the CFG",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(LowerSwitch)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(RegionInfoPass)
INITIALIZE_PASS_END(StructurizeCFG, "structurizecfg", "Structurize the CFG",
                    false, false)

/// \brief Initialize the types and constants used in the pass
bool StructurizeCFG::doInitialization(Region *R, RGPassManager &RGM) {
  LLVMContext &Context = R->getEntry()->getContext();

  Boolean = Type::getInt1Ty(Context);
  BoolTrue = ConstantInt::getTrue(Context);
  BoolFalse = ConstantInt::getFalse(Context);
  BoolUndef = UndefValue::get(Boolean);

  return false;
}

/// \brief Build up the general order of nodes
void StructurizeCFG::orderNodes() {
  RNVector TempOrder;
  ReversePostOrderTraversal<Region*> RPOT(ParentRegion);
  TempOrder.append(RPOT.begin(), RPOT.end());

  std::map<Loop*, unsigned> LoopBlocks;


  // The reverse post-order traversal of the list gives us an ordering close
  // to what we want.  The only problem with it is that sometimes backedges
  // for outer loops will be visited before backedges for inner loops.
  for (RegionNode *RN : TempOrder) {
    BasicBlock *BB = RN->getEntry();
    Loop *Loop = LI->getLoopFor(BB);
    if (!LoopBlocks.count(Loop)) {
      LoopBlocks[Loop] = 1;
      continue;
    }
    LoopBlocks[Loop]++;
  }

  unsigned CurrentLoopDepth = 0;
  Loop *CurrentLoop = nullptr;
  BBSet TempVisited;
  for (RNVector::iterator I = TempOrder.begin(), E = TempOrder.end(); I != E; ++I) {
    BasicBlock *BB = (*I)->getEntry();
    unsigned LoopDepth = LI->getLoopDepth(BB);

    if (std::find(Order.begin(), Order.end(), *I) != Order.end())
      continue;

    if (LoopDepth < CurrentLoopDepth) {
      // Make sure we have visited all blocks in this loop before moving back to
      // the outer loop.

      RNVector::iterator LoopI = I;
      while(LoopBlocks[CurrentLoop]) {
        LoopI++;
        BasicBlock *LoopBB = (*LoopI)->getEntry();
        if (LI->getLoopFor(LoopBB) == CurrentLoop) {
          LoopBlocks[CurrentLoop]--;
          Order.push_back(*LoopI);
        }
      }
    }

    CurrentLoop = LI->getLoopFor(BB);
    if (CurrentLoop) {
      LoopBlocks[CurrentLoop]--;
    }

    CurrentLoopDepth = LoopDepth;
    Order.push_back(*I);
  }

  // This pass originally used a post-order traversal and then operated on
  // the list in reverse. Now that we are using a reverse post-order traversal
  // rather than re-working the whole pass to operate on the list in order,
  // we just reverse the list and continue to operate on it in reverse.
  std::reverse(Order.begin(), Order.end());
}

/// \brief Determine the end of the loops
void StructurizeCFG::analyzeLoops(RegionNode *N) {
  if (N->isSubRegion()) {
    // Test for exit as back edge
    BasicBlock *Exit = N->getNodeAs<Region>()->getExit();
    if (Visited.count(Exit))
      Loops[Exit] = N->getEntry();

  } else {
    // Test for sucessors as back edge
    BasicBlock *BB = N->getNodeAs<BasicBlock>();
    BranchInst *Term = cast<BranchInst>(BB->getTerminator());

    for (unsigned i = 0, e = Term->getNumSuccessors(); i != e; ++i) {
      BasicBlock *Succ = Term->getSuccessor(i);

      if (Visited.count(Succ)) {
        Loops[Succ] = BB;
      }
    }
  }
}

/// \brief Invert the given condition
Value *StructurizeCFG::invert(Value *Condition) {
  // First: Check if it's a constant
  if (Condition == BoolTrue)
    return BoolFalse;

  if (Condition == BoolFalse)
    return BoolTrue;

  if (Condition == BoolUndef)
    return BoolUndef;

  // Second: If the condition is already inverted, return the original value
  if (match(Condition, m_Not(m_Value(Condition))))
    return Condition;

  if (Instruction *Inst = dyn_cast<Instruction>(Condition)) {
    // Third: Check all the users for an invert
    BasicBlock *Parent = Inst->getParent();
    for (User *U : Condition->users())
      if (Instruction *I = dyn_cast<Instruction>(U))
        if (I->getParent() == Parent && match(I, m_Not(m_Specific(Condition))))
          return I;

    // Last option: Create a new instruction
    return BinaryOperator::CreateNot(Condition, "", Parent->getTerminator());
  }

  if (Argument *Arg = dyn_cast<Argument>(Condition)) {
    BasicBlock &EntryBlock = Arg->getParent()->getEntryBlock();
    return BinaryOperator::CreateNot(Condition,
                                     Arg->getName() + ".inv",
                                     EntryBlock.getTerminator());
  }

  llvm_unreachable("Unhandled condition to invert");
}

/// \brief Build the condition for one edge
Value *StructurizeCFG::buildCondition(BranchInst *Term, unsigned Idx,
                                      bool Invert) {
  Value *Cond = Invert ? BoolFalse : BoolTrue;
  if (Term->isConditional()) {
    Cond = Term->getCondition();

    if (Idx != (unsigned)Invert)
      Cond = invert(Cond);
  }
  return Cond;
}

/// \brief Analyze the predecessors of each block and build up predicates
void StructurizeCFG::gatherPredicates(RegionNode *N) {
  RegionInfo *RI = ParentRegion->getRegionInfo();
  BasicBlock *BB = N->getEntry();
  BBPredicates &Pred = Predicates[BB];
  BBPredicates &LPred = LoopPreds[BB];

  for (pred_iterator PI = pred_begin(BB), PE = pred_end(BB);
       PI != PE; ++PI) {

    // Ignore it if it's a branch from outside into our region entry
    if (!ParentRegion->contains(*PI))
      continue;

    Region *R = RI->getRegionFor(*PI);
    if (R == ParentRegion) {

      // It's a top level block in our region
      BranchInst *Term = cast<BranchInst>((*PI)->getTerminator());
      for (unsigned i = 0, e = Term->getNumSuccessors(); i != e; ++i) {
        BasicBlock *Succ = Term->getSuccessor(i);
        if (Succ != BB)
          continue;

        if (Visited.count(*PI)) {
          // Normal forward edge
          if (Term->isConditional()) {
            // Try to treat it like an ELSE block
            BasicBlock *Other = Term->getSuccessor(!i);
            if (Visited.count(Other) && !Loops.count(Other) &&
                !Pred.count(Other) && !Pred.count(*PI)) {

              Pred[Other] = BoolFalse;
              Pred[*PI] = BoolTrue;
              continue;
            }
          }
          Pred[*PI] = buildCondition(Term, i, false);

        } else {
          // Back edge
          LPred[*PI] = buildCondition(Term, i, true);
        }
      }

    } else {

      // It's an exit from a sub region
      while (R->getParent() != ParentRegion)
        R = R->getParent();

      // Edge from inside a subregion to its entry, ignore it
      if (*R == *N)
        continue;

      BasicBlock *Entry = R->getEntry();
      if (Visited.count(Entry))
        Pred[Entry] = BoolTrue;
      else
        LPred[Entry] = BoolFalse;
    }
  }
}

/// \brief Collect various loop and predicate infos
void StructurizeCFG::collectInfos() {
  // Reset predicate
  Predicates.clear();

  // and loop infos
  Loops.clear();
  LoopPreds.clear();

  // Reset the visited nodes
  Visited.clear();

  for (RNVector::reverse_iterator OI = Order.rbegin(), OE = Order.rend();
       OI != OE; ++OI) {

    DEBUG(dbgs() << "Visiting: " <<
                    ((*OI)->isSubRegion() ? "SubRegion with entry: " : "") <<
                    (*OI)->getEntry()->getName() << " Loop Depth: " << LI->getLoopDepth((*OI)->getEntry()) << "\n");

    // Analyze all the conditions leading to a node
    gatherPredicates(*OI);

    // Remember that we've seen this node
    Visited.insert((*OI)->getEntry());

    // Find the last back edges
    analyzeLoops(*OI);
  }
}

/// \brief Insert the missing branch conditions
void StructurizeCFG::insertConditions(bool Loops) {
  BranchVector &Conds = Loops ? LoopConds : Conditions;
  Value *Default = Loops ? BoolTrue : BoolFalse;
  SSAUpdater PhiInserter;

  for (BranchInst *Term : Conds) {
    assert(Term->isConditional());

    BasicBlock *Parent = Term->getParent();
    BasicBlock *SuccTrue = Term->getSuccessor(0);
    BasicBlock *SuccFalse = Term->getSuccessor(1);

    PhiInserter.Initialize(Boolean, "");
    PhiInserter.AddAvailableValue(&Func->getEntryBlock(), Default);
    PhiInserter.AddAvailableValue(Loops ? SuccFalse : Parent, Default);

    BBPredicates &Preds = Loops ? LoopPreds[SuccFalse] : Predicates[SuccTrue];

    NearestCommonDominator Dominator(DT);
    Dominator.addBlock(Parent, false);

    Value *ParentValue = nullptr;
    for (BBPredicates::iterator PI = Preds.begin(), PE = Preds.end();
         PI != PE; ++PI) {

      if (PI->first == Parent) {
        ParentValue = PI->second;
        break;
      }
      PhiInserter.AddAvailableValue(PI->first, PI->second);
      Dominator.addBlock(PI->first);
    }

    if (ParentValue) {
      Term->setCondition(ParentValue);
    } else {
      if (!Dominator.wasResultExplicitMentioned())
        PhiInserter.AddAvailableValue(Dominator.getResult(), Default);

      Term->setCondition(PhiInserter.GetValueInMiddleOfBlock(Parent));
    }
  }
}

/// \brief Remove all PHI values coming from "From" into "To" and remember
/// them in DeletedPhis
void StructurizeCFG::delPhiValues(BasicBlock *From, BasicBlock *To) {
  PhiMap &Map = DeletedPhis[To];
  for (BasicBlock::iterator I = To->begin(), E = To->end();
       I != E && isa<PHINode>(*I);) {

    PHINode &Phi = cast<PHINode>(*I++);
    while (Phi.getBasicBlockIndex(From) != -1) {
      Value *Deleted = Phi.removeIncomingValue(From, false);
      Map[&Phi].push_back(std::make_pair(From, Deleted));
    }
  }
}

/// \brief Add a dummy PHI value as soon as we knew the new predecessor
void StructurizeCFG::addPhiValues(BasicBlock *From, BasicBlock *To) {
  for (BasicBlock::iterator I = To->begin(), E = To->end();
       I != E && isa<PHINode>(*I);) {

    PHINode &Phi = cast<PHINode>(*I++);
    Value *Undef = UndefValue::get(Phi.getType());
    Phi.addIncoming(Undef, From);
  }
  AddedPhis[To].push_back(From);
}

/// \brief Add the real PHI value as soon as everything is set up
void StructurizeCFG::setPhiValues() {
  SSAUpdater Updater;
  for (BB2BBVecMap::iterator AI = AddedPhis.begin(), AE = AddedPhis.end();
       AI != AE; ++AI) {

    BasicBlock *To = AI->first;
    BBVector &From = AI->second;

    if (!DeletedPhis.count(To))
      continue;

    PhiMap &Map = DeletedPhis[To];
    for (PhiMap::iterator PI = Map.begin(), PE = Map.end();
         PI != PE; ++PI) {

      PHINode *Phi = PI->first;
      Value *Undef = UndefValue::get(Phi->getType());
      Updater.Initialize(Phi->getType(), "");
      Updater.AddAvailableValue(&Func->getEntryBlock(), Undef);
      Updater.AddAvailableValue(To, Undef);

      NearestCommonDominator Dominator(DT);
      Dominator.addBlock(To, false);
      for (BBValueVector::iterator VI = PI->second.begin(),
           VE = PI->second.end(); VI != VE; ++VI) {

        Updater.AddAvailableValue(VI->first, VI->second);
        Dominator.addBlock(VI->first);
      }

      if (!Dominator.wasResultExplicitMentioned())
        Updater.AddAvailableValue(Dominator.getResult(), Undef);

      for (BBVector::iterator FI = From.begin(), FE = From.end();
           FI != FE; ++FI) {

        int Idx = Phi->getBasicBlockIndex(*FI);
        assert(Idx != -1);
        Phi->setIncomingValue(Idx, Updater.GetValueAtEndOfBlock(*FI));
      }
    }

    DeletedPhis.erase(To);
  }
  assert(DeletedPhis.empty());
}

/// \brief Remove phi values from all successors and then remove the terminator.
void StructurizeCFG::killTerminator(BasicBlock *BB) {
  TerminatorInst *Term = BB->getTerminator();
  if (!Term)
    return;

  for (succ_iterator SI = succ_begin(BB), SE = succ_end(BB);
       SI != SE; ++SI) {

    delPhiValues(BB, *SI);
  }

  Term->eraseFromParent();
}

/// \brief Let node exit(s) point to NewExit
void StructurizeCFG::changeExit(RegionNode *Node, BasicBlock *NewExit,
                                bool IncludeDominator) {
  if (Node->isSubRegion()) {
    Region *SubRegion = Node->getNodeAs<Region>();
    BasicBlock *OldExit = SubRegion->getExit();
    BasicBlock *Dominator = nullptr;

    // Find all the edges from the sub region to the exit
    for (pred_iterator I = pred_begin(OldExit), E = pred_end(OldExit);
         I != E;) {

      BasicBlock *BB = *I++;
      if (!SubRegion->contains(BB))
        continue;

      // Modify the edges to point to the new exit
      delPhiValues(BB, OldExit);
      BB->getTerminator()->replaceUsesOfWith(OldExit, NewExit);
      addPhiValues(BB, NewExit);

      // Find the new dominator (if requested)
      if (IncludeDominator) {
        if (!Dominator)
          Dominator = BB;
        else
          Dominator = DT->findNearestCommonDominator(Dominator, BB);
      }
    }

    // Change the dominator (if requested)
    if (Dominator)
      DT->changeImmediateDominator(NewExit, Dominator);

    // Update the region info
    SubRegion->replaceExit(NewExit);

  } else {
    BasicBlock *BB = Node->getNodeAs<BasicBlock>();
    killTerminator(BB);
    BranchInst::Create(NewExit, BB);
    addPhiValues(BB, NewExit);
    if (IncludeDominator)
      DT->changeImmediateDominator(NewExit, BB);
  }
}

/// \brief Create a new flow node and update dominator tree and region info
BasicBlock *StructurizeCFG::getNextFlow(BasicBlock *Dominator) {
  LLVMContext &Context = Func->getContext();
  BasicBlock *Insert = Order.empty() ? ParentRegion->getExit() :
                       Order.back()->getEntry();
  BasicBlock *Flow = BasicBlock::Create(Context, FlowBlockName,
                                        Func, Insert);
  DT->addNewBlock(Flow, Dominator);
  ParentRegion->getRegionInfo()->setRegionFor(Flow, ParentRegion);
  return Flow;
}

/// \brief Create a new or reuse the previous node as flow node
BasicBlock *StructurizeCFG::needPrefix(bool NeedEmpty) {
  BasicBlock *Entry = PrevNode->getEntry();

  if (!PrevNode->isSubRegion()) {
    killTerminator(Entry);
    if (!NeedEmpty || Entry->getFirstInsertionPt() == Entry->end())
      return Entry;

  }

  // create a new flow node
  BasicBlock *Flow = getNextFlow(Entry);

  // and wire it up
  changeExit(PrevNode, Flow, true);
  PrevNode = ParentRegion->getBBNode(Flow);
  return Flow;
}

/// \brief Returns the region exit if possible, otherwise just a new flow node
BasicBlock *StructurizeCFG::needPostfix(BasicBlock *Flow,
                                        bool ExitUseAllowed) {
  if (Order.empty() && ExitUseAllowed) {
    BasicBlock *Exit = ParentRegion->getExit();
    DT->changeImmediateDominator(Exit, Flow);
    addPhiValues(Flow, Exit);
    return Exit;
  }
  return getNextFlow(Flow);
}

/// \brief Set the previous node
void StructurizeCFG::setPrevNode(BasicBlock *BB) {
  PrevNode = ParentRegion->contains(BB) ? ParentRegion->getBBNode(BB)
                                        : nullptr;
}

/// \brief Does BB dominate all the predicates of Node ?
bool StructurizeCFG::dominatesPredicates(BasicBlock *BB, RegionNode *Node) {
  BBPredicates &Preds = Predicates[Node->getEntry()];
  for (BBPredicates::iterator PI = Preds.begin(), PE = Preds.end();
       PI != PE; ++PI) {

    if (!DT->dominates(BB, PI->first))
      return false;
  }
  return true;
}

/// \brief Can we predict that this node will always be called?
bool StructurizeCFG::isPredictableTrue(RegionNode *Node) {
  BBPredicates &Preds = Predicates[Node->getEntry()];
  bool Dominated = false;

  // Regionentry is always true
  if (!PrevNode)
    return true;

  for (BBPredicates::iterator I = Preds.begin(), E = Preds.end();
       I != E; ++I) {

    if (I->second != BoolTrue)
      return false;

    if (!Dominated && DT->dominates(I->first, PrevNode->getEntry()))
      Dominated = true;
  }

  // TODO: The dominator check is too strict
  return Dominated;
}

/// Take one node from the order vector and wire it up
void StructurizeCFG::wireFlow(bool ExitUseAllowed,
                              BasicBlock *LoopEnd) {
  RegionNode *Node = Order.pop_back_val();
  Visited.insert(Node->getEntry());

  if (isPredictableTrue(Node)) {
    // Just a linear flow
    if (PrevNode) {
      changeExit(PrevNode, Node->getEntry(), true);
    }
    PrevNode = Node;

  } else {
    // Insert extra prefix node (or reuse last one)
    BasicBlock *Flow = needPrefix(false);

    // Insert extra postfix node (or use exit instead)
    BasicBlock *Entry = Node->getEntry();
    BasicBlock *Next = needPostfix(Flow, ExitUseAllowed);

    // let it point to entry and next block
    Conditions.push_back(BranchInst::Create(Entry, Next, BoolUndef, Flow));
    addPhiValues(Flow, Entry);
    DT->changeImmediateDominator(Entry, Flow);

    PrevNode = Node;
    while (!Order.empty() && !Visited.count(LoopEnd) &&
           dominatesPredicates(Entry, Order.back())) {
      handleLoops(false, LoopEnd);
    }

    changeExit(PrevNode, Next, false);
    setPrevNode(Next);
  }
}

void StructurizeCFG::handleLoops(bool ExitUseAllowed,
                                 BasicBlock *LoopEnd) {
  RegionNode *Node = Order.back();
  BasicBlock *LoopStart = Node->getEntry();

  if (!Loops.count(LoopStart)) {
    wireFlow(ExitUseAllowed, LoopEnd);
    return;
  }

  if (!isPredictableTrue(Node))
    LoopStart = needPrefix(true);

  LoopEnd = Loops[Node->getEntry()];
  wireFlow(false, LoopEnd);
  while (!Visited.count(LoopEnd)) {
    handleLoops(false, LoopEnd);
  }

  // If the start of the loop is the entry block, we can't branch to it so
  // insert a new dummy entry block.
  Function *LoopFunc = LoopStart->getParent();
  if (LoopStart == &LoopFunc->getEntryBlock()) {
    LoopStart->setName("entry.orig");

    BasicBlock *NewEntry =
      BasicBlock::Create(LoopStart->getContext(),
                         "entry",
                         LoopFunc,
                         LoopStart);
    BranchInst::Create(LoopStart, NewEntry);
  }

  // Create an extra loop end node
  LoopEnd = needPrefix(false);
  BasicBlock *Next = needPostfix(LoopEnd, ExitUseAllowed);
  LoopConds.push_back(BranchInst::Create(Next, LoopStart,
                                         BoolUndef, LoopEnd));
  addPhiValues(LoopEnd, LoopStart);
  setPrevNode(Next);
}

/// After this function control flow looks like it should be, but
/// branches and PHI nodes only have undefined conditions.
void StructurizeCFG::createFlow() {
  BasicBlock *Exit = ParentRegion->getExit();
  bool EntryDominatesExit = DT->dominates(ParentRegion->getEntry(), Exit);

  DeletedPhis.clear();
  AddedPhis.clear();
  Conditions.clear();
  LoopConds.clear();

  PrevNode = nullptr;
  Visited.clear();

  while (!Order.empty()) {
    handleLoops(EntryDominatesExit, nullptr);
  }

  if (PrevNode)
    changeExit(PrevNode, Exit, EntryDominatesExit);
  else
    assert(EntryDominatesExit);
}

/// Handle a rare case where the disintegrated nodes instructions
/// no longer dominate all their uses. Not sure if this is really nessasary
void StructurizeCFG::rebuildSSA() {
  SSAUpdater Updater;
  for (auto *BB : ParentRegion->blocks())
    for (BasicBlock::iterator II = BB->begin(), IE = BB->end();
         II != IE; ++II) {

      bool Initialized = false;
      for (auto I = II->use_begin(), E = II->use_end(); I != E;) {
        Use &U = *I++;
        Instruction *User = cast<Instruction>(U.getUser());
        if (User->getParent() == BB) {
          continue;

        } else if (PHINode *UserPN = dyn_cast<PHINode>(User)) {
          if (UserPN->getIncomingBlock(U) == BB)
            continue;
        }

        if (DT->dominates(II, User))
          continue;

        if (!Initialized) {
          Value *Undef = UndefValue::get(II->getType());
          Updater.Initialize(II->getType(), "");
          Updater.AddAvailableValue(&Func->getEntryBlock(), Undef);
          Updater.AddAvailableValue(BB, II);
          Initialized = true;
        }
        Updater.RewriteUseAfterInsertions(U);
      }
    }
}

/// \brief Run the transformation for each region found
bool StructurizeCFG::runOnRegion(Region *R, RGPassManager &RGM) {
  if (R->isTopLevelRegion())
    return false;

  Func = R->getEntry()->getParent();
  ParentRegion = R;

  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();

  orderNodes();
  collectInfos();
  createFlow();
  insertConditions(false);
  insertConditions(true);
  setPhiValues();
  rebuildSSA();

  // Cleanup
  Order.clear();
  Visited.clear();
  DeletedPhis.clear();
  AddedPhis.clear();
  Predicates.clear();
  Conditions.clear();
  Loops.clear();
  LoopPreds.clear();
  LoopConds.clear();

  return true;
}

/// \brief Create the pass
Pass *llvm::createStructurizeCFGPass() {
  return new StructurizeCFG();
}
