//===-- LegalizeTypes.cpp - Common code for DAG type legalizer ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SelectionDAG::LegalizeTypes method.  It transforms
// an arbitrary well-formed SelectionDAG to only consist of legal types.  This
// is common code shared among the LegalizeTypes*.cpp files.
//
//===----------------------------------------------------------------------===//

#include "LegalizeTypes.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define DEBUG_TYPE "legalize-types"

static cl::opt<bool>
EnableExpensiveChecks("enable-legalize-types-checking", cl::Hidden);

/// PerformExpensiveChecks - Do extensive, expensive, sanity checking.
void DAGTypeLegalizer::PerformExpensiveChecks() {
  // If a node is not processed, then none of its values should be mapped by any
  // of PromotedIntegers, ExpandedIntegers, ..., ReplacedValues.

  // If a node is processed, then each value with an illegal type must be mapped
  // by exactly one of PromotedIntegers, ExpandedIntegers, ..., ReplacedValues.
  // Values with a legal type may be mapped by ReplacedValues, but not by any of
  // the other maps.

  // Note that these invariants may not hold momentarily when processing a node:
  // the node being processed may be put in a map before being marked Processed.

  // Note that it is possible to have nodes marked NewNode in the DAG.  This can
  // occur in two ways.  Firstly, a node may be created during legalization but
  // never passed to the legalization core.  This is usually due to the implicit
  // folding that occurs when using the DAG.getNode operators.  Secondly, a new
  // node may be passed to the legalization core, but when analyzed may morph
  // into a different node, leaving the original node as a NewNode in the DAG.
  // A node may morph if one of its operands changes during analysis.  Whether
  // it actually morphs or not depends on whether, after updating its operands,
  // it is equivalent to an existing node: if so, it morphs into that existing
  // node (CSE).  An operand can change during analysis if the operand is a new
  // node that morphs, or it is a processed value that was mapped to some other
  // value (as recorded in ReplacedValues) in which case the operand is turned
  // into that other value.  If a node morphs then the node it morphed into will
  // be used instead of it for legalization, however the original node continues
  // to live on in the DAG.
  // The conclusion is that though there may be nodes marked NewNode in the DAG,
  // all uses of such nodes are also marked NewNode: the result is a fungus of
  // NewNodes growing on top of the useful nodes, and perhaps using them, but
  // not used by them.

  // If a value is mapped by ReplacedValues, then it must have no uses, except
  // by nodes marked NewNode (see above).

  // The final node obtained by mapping by ReplacedValues is not marked NewNode.
  // Note that ReplacedValues should be applied iteratively.

  // Note that the ReplacedValues map may also map deleted nodes (by iterating
  // over the DAG we never dereference deleted nodes).  This means that it may
  // also map nodes marked NewNode if the deallocated memory was reallocated as
  // another node, and that new node was not seen by the LegalizeTypes machinery
  // (for example because it was created but not used).  In general, we cannot
  // distinguish between new nodes and deleted nodes.
  SmallVector<SDNode*, 16> NewNodes;
  for (SelectionDAG::allnodes_iterator I = DAG.allnodes_begin(),
       E = DAG.allnodes_end(); I != E; ++I) {
    // Remember nodes marked NewNode - they are subject to extra checking below.
    if (I->getNodeId() == NewNode)
      NewNodes.push_back(I);

    for (unsigned i = 0, e = I->getNumValues(); i != e; ++i) {
      SDValue Res(I, i);
      bool Failed = false;

      unsigned Mapped = 0;
      if (ReplacedValues.find(Res) != ReplacedValues.end()) {
        Mapped |= 1;
        // Check that remapped values are only used by nodes marked NewNode.
        for (SDNode::use_iterator UI = I->use_begin(), UE = I->use_end();
             UI != UE; ++UI)
          if (UI.getUse().getResNo() == i)
            assert(UI->getNodeId() == NewNode &&
                   "Remapped value has non-trivial use!");

        // Check that the final result of applying ReplacedValues is not
        // marked NewNode.
        SDValue NewVal = ReplacedValues[Res];
        DenseMap<SDValue, SDValue>::iterator I = ReplacedValues.find(NewVal);
        while (I != ReplacedValues.end()) {
          NewVal = I->second;
          I = ReplacedValues.find(NewVal);
        }
        assert(NewVal.getNode()->getNodeId() != NewNode &&
               "ReplacedValues maps to a new node!");
      }
      if (PromotedIntegers.find(Res) != PromotedIntegers.end())
        Mapped |= 2;
      if (SoftenedFloats.find(Res) != SoftenedFloats.end())
        Mapped |= 4;
      if (ScalarizedVectors.find(Res) != ScalarizedVectors.end())
        Mapped |= 8;
      if (ExpandedIntegers.find(Res) != ExpandedIntegers.end())
        Mapped |= 16;
      if (ExpandedFloats.find(Res) != ExpandedFloats.end())
        Mapped |= 32;
      if (SplitVectors.find(Res) != SplitVectors.end())
        Mapped |= 64;
      if (WidenedVectors.find(Res) != WidenedVectors.end())
        Mapped |= 128;

      if (I->getNodeId() != Processed) {
        // Since we allow ReplacedValues to map deleted nodes, it may map nodes
        // marked NewNode too, since a deleted node may have been reallocated as
        // another node that has not been seen by the LegalizeTypes machinery.
        if ((I->getNodeId() == NewNode && Mapped > 1) ||
            (I->getNodeId() != NewNode && Mapped != 0)) {
          dbgs() << "Unprocessed value in a map!";
          Failed = true;
        }
      } else if (isTypeLegal(Res.getValueType()) || IgnoreNodeResults(I)) {
        if (Mapped > 1) {
          dbgs() << "Value with legal type was transformed!";
          Failed = true;
        }
      } else {
        if (Mapped == 0) {
          dbgs() << "Processed value not in any map!";
          Failed = true;
        } else if (Mapped & (Mapped - 1)) {
          dbgs() << "Value in multiple maps!";
          Failed = true;
        }
      }

      if (Failed) {
        if (Mapped & 1)
          dbgs() << " ReplacedValues";
        if (Mapped & 2)
          dbgs() << " PromotedIntegers";
        if (Mapped & 4)
          dbgs() << " SoftenedFloats";
        if (Mapped & 8)
          dbgs() << " ScalarizedVectors";
        if (Mapped & 16)
          dbgs() << " ExpandedIntegers";
        if (Mapped & 32)
          dbgs() << " ExpandedFloats";
        if (Mapped & 64)
          dbgs() << " SplitVectors";
        if (Mapped & 128)
          dbgs() << " WidenedVectors";
        dbgs() << "\n";
        llvm_unreachable(nullptr);
      }
    }
  }

  // Checked that NewNodes are only used by other NewNodes.
  for (unsigned i = 0, e = NewNodes.size(); i != e; ++i) {
    SDNode *N = NewNodes[i];
    for (SDNode::use_iterator UI = N->use_begin(), UE = N->use_end();
         UI != UE; ++UI)
      assert(UI->getNodeId() == NewNode && "NewNode used by non-NewNode!");
  }
}

/// run - This is the main entry point for the type legalizer.  This does a
/// top-down traversal of the dag, legalizing types as it goes.  Returns "true"
/// if it made any changes.
bool DAGTypeLegalizer::run() {
  bool Changed = false;

  // Create a dummy node (which is not added to allnodes), that adds a reference
  // to the root node, preventing it from being deleted, and tracking any
  // changes of the root.
  HandleSDNode Dummy(DAG.getRoot());
  Dummy.setNodeId(Unanalyzed);

  // The root of the dag may dangle to deleted nodes until the type legalizer is
  // done.  Set it to null to avoid confusion.
  DAG.setRoot(SDValue());

  // Walk all nodes in the graph, assigning them a NodeId of 'ReadyToProcess'
  // (and remembering them) if they are leaves and assigning 'Unanalyzed' if
  // non-leaves.
  for (SelectionDAG::allnodes_iterator I = DAG.allnodes_begin(),
       E = DAG.allnodes_end(); I != E; ++I) {
    if (I->getNumOperands() == 0) {
      I->setNodeId(ReadyToProcess);
      Worklist.push_back(I);
    } else {
      I->setNodeId(Unanalyzed);
    }
  }

  // Now that we have a set of nodes to process, handle them all.
  while (!Worklist.empty()) {
#ifndef XDEBUG
    if (EnableExpensiveChecks)
#endif
      PerformExpensiveChecks();

    SDNode *N = Worklist.back();
    Worklist.pop_back();
    assert(N->getNodeId() == ReadyToProcess &&
           "Node should be ready if on worklist!");

    if (IgnoreNodeResults(N))
      goto ScanOperands;

    // Scan the values produced by the node, checking to see if any result
    // types are illegal.
    for (unsigned i = 0, NumResults = N->getNumValues(); i < NumResults; ++i) {
      EVT ResultVT = N->getValueType(i);
      switch (getTypeAction(ResultVT)) {
      case TargetLowering::TypeLegal:
        break;
      // The following calls must take care of *all* of the node's results,
      // not just the illegal result they were passed (this includes results
      // with a legal type).  Results can be remapped using ReplaceValueWith,
      // or their promoted/expanded/etc values registered in PromotedIntegers,
      // ExpandedIntegers etc.
      case TargetLowering::TypePromoteInteger:
        PromoteIntegerResult(N, i);
        Changed = true;
        goto NodeDone;
      case TargetLowering::TypeExpandInteger:
        ExpandIntegerResult(N, i);
        Changed = true;
        goto NodeDone;
      case TargetLowering::TypeSoftenFloat:
        SoftenFloatResult(N, i);
        Changed = true;
        goto NodeDone;
      case TargetLowering::TypeExpandFloat:
        ExpandFloatResult(N, i);
        Changed = true;
        goto NodeDone;
      case TargetLowering::TypeScalarizeVector:
        ScalarizeVectorResult(N, i);
        Changed = true;
        goto NodeDone;
      case TargetLowering::TypeSplitVector:
        SplitVectorResult(N, i);
        Changed = true;
        goto NodeDone;
      case TargetLowering::TypeWidenVector:
        WidenVectorResult(N, i);
        Changed = true;
        goto NodeDone;
      case TargetLowering::TypePromoteFloat:
        PromoteFloatResult(N, i);
        Changed = true;
        goto NodeDone;
      }
    }

ScanOperands:
    // Scan the operand list for the node, handling any nodes with operands that
    // are illegal.
    {
    unsigned NumOperands = N->getNumOperands();
    bool NeedsReanalyzing = false;
    unsigned i;
    for (i = 0; i != NumOperands; ++i) {
      if (IgnoreNodeResults(N->getOperand(i).getNode()))
        continue;

      EVT OpVT = N->getOperand(i).getValueType();
      switch (getTypeAction(OpVT)) {
      case TargetLowering::TypeLegal:
        continue;
      // The following calls must either replace all of the node's results
      // using ReplaceValueWith, and return "false"; or update the node's
      // operands in place, and return "true".
      case TargetLowering::TypePromoteInteger:
        NeedsReanalyzing = PromoteIntegerOperand(N, i);
        Changed = true;
        break;
      case TargetLowering::TypeExpandInteger:
        NeedsReanalyzing = ExpandIntegerOperand(N, i);
        Changed = true;
        break;
      case TargetLowering::TypeSoftenFloat:
        NeedsReanalyzing = SoftenFloatOperand(N, i);
        Changed = true;
        break;
      case TargetLowering::TypeExpandFloat:
        NeedsReanalyzing = ExpandFloatOperand(N, i);
        Changed = true;
        break;
      case TargetLowering::TypeScalarizeVector:
        NeedsReanalyzing = ScalarizeVectorOperand(N, i);
        Changed = true;
        break;
      case TargetLowering::TypeSplitVector:
        NeedsReanalyzing = SplitVectorOperand(N, i);
        Changed = true;
        break;
      case TargetLowering::TypeWidenVector:
        NeedsReanalyzing = WidenVectorOperand(N, i);
        Changed = true;
        break;
      case TargetLowering::TypePromoteFloat:
        NeedsReanalyzing = PromoteFloatOperand(N, i);
        Changed = true;
        break;
      }
      break;
    }

    // The sub-method updated N in place.  Check to see if any operands are new,
    // and if so, mark them.  If the node needs revisiting, don't add all users
    // to the worklist etc.
    if (NeedsReanalyzing) {
      assert(N->getNodeId() == ReadyToProcess && "Node ID recalculated?");
      N->setNodeId(NewNode);
      // Recompute the NodeId and correct processed operands, adding the node to
      // the worklist if ready.
      SDNode *M = AnalyzeNewNode(N);
      if (M == N)
        // The node didn't morph - nothing special to do, it will be revisited.
        continue;

      // The node morphed - this is equivalent to legalizing by replacing every
      // value of N with the corresponding value of M.  So do that now.
      assert(N->getNumValues() == M->getNumValues() &&
             "Node morphing changed the number of results!");
      for (unsigned i = 0, e = N->getNumValues(); i != e; ++i)
        // Replacing the value takes care of remapping the new value.
        ReplaceValueWith(SDValue(N, i), SDValue(M, i));
      assert(N->getNodeId() == NewNode && "Unexpected node state!");
      // The node continues to live on as part of the NewNode fungus that
      // grows on top of the useful nodes.  Nothing more needs to be done
      // with it - move on to the next node.
      continue;
    }

    if (i == NumOperands) {
      DEBUG(dbgs() << "Legally typed node: "; N->dump(&DAG); dbgs() << "\n");
    }
    }
NodeDone:

    // If we reach here, the node was processed, potentially creating new nodes.
    // Mark it as processed and add its users to the worklist as appropriate.
    assert(N->getNodeId() == ReadyToProcess && "Node ID recalculated?");
    N->setNodeId(Processed);

    for (SDNode::use_iterator UI = N->use_begin(), E = N->use_end();
         UI != E; ++UI) {
      SDNode *User = *UI;
      int NodeId = User->getNodeId();

      // This node has two options: it can either be a new node or its Node ID
      // may be a count of the number of operands it has that are not ready.
      if (NodeId > 0) {
        User->setNodeId(NodeId-1);

        // If this was the last use it was waiting on, add it to the ready list.
        if (NodeId-1 == ReadyToProcess)
          Worklist.push_back(User);
        continue;
      }

      // If this is an unreachable new node, then ignore it.  If it ever becomes
      // reachable by being used by a newly created node then it will be handled
      // by AnalyzeNewNode.
      if (NodeId == NewNode)
        continue;

      // Otherwise, this node is new: this is the first operand of it that
      // became ready.  Its new NodeId is the number of operands it has minus 1
      // (as this node is now processed).
      assert(NodeId == Unanalyzed && "Unknown node ID!");
      User->setNodeId(User->getNumOperands() - 1);

      // If the node only has a single operand, it is now ready.
      if (User->getNumOperands() == 1)
        Worklist.push_back(User);
    }
  }

#ifndef XDEBUG
  if (EnableExpensiveChecks)
#endif
    PerformExpensiveChecks();

  // If the root changed (e.g. it was a dead load) update the root.
  DAG.setRoot(Dummy.getValue());

  // Remove dead nodes.  This is important to do for cleanliness but also before
  // the checking loop below.  Implicit folding by the DAG.getNode operators and
  // node morphing can cause unreachable nodes to be around with their flags set
  // to new.
  DAG.RemoveDeadNodes();

  // In a debug build, scan all the nodes to make sure we found them all.  This
  // ensures that there are no cycles and that everything got processed.
#ifndef NDEBUG
  for (SelectionDAG::allnodes_iterator I = DAG.allnodes_begin(),
       E = DAG.allnodes_end(); I != E; ++I) {
    bool Failed = false;

    // Check that all result types are legal.
    if (!IgnoreNodeResults(I))
      for (unsigned i = 0, NumVals = I->getNumValues(); i < NumVals; ++i)
        if (!isTypeLegal(I->getValueType(i))) {
          dbgs() << "Result type " << i << " illegal!\n";
          Failed = true;
        }

    // Check that all operand types are legal.
    for (unsigned i = 0, NumOps = I->getNumOperands(); i < NumOps; ++i)
      if (!IgnoreNodeResults(I->getOperand(i).getNode()) &&
          !isTypeLegal(I->getOperand(i).getValueType())) {
        dbgs() << "Operand type " << i << " illegal!\n";
        Failed = true;
      }

    if (I->getNodeId() != Processed) {
       if (I->getNodeId() == NewNode)
         dbgs() << "New node not analyzed?\n";
       else if (I->getNodeId() == Unanalyzed)
         dbgs() << "Unanalyzed node not noticed?\n";
       else if (I->getNodeId() > 0)
         dbgs() << "Operand not processed?\n";
       else if (I->getNodeId() == ReadyToProcess)
         dbgs() << "Not added to worklist?\n";
       Failed = true;
    }

    if (Failed) {
      I->dump(&DAG); dbgs() << "\n";
      llvm_unreachable(nullptr);
    }
  }
#endif

  return Changed;
}

/// AnalyzeNewNode - The specified node is the root of a subtree of potentially
/// new nodes.  Correct any processed operands (this may change the node) and
/// calculate the NodeId.  If the node itself changes to a processed node, it
/// is not remapped - the caller needs to take care of this.
/// Returns the potentially changed node.
SDNode *DAGTypeLegalizer::AnalyzeNewNode(SDNode *N) {
  // If this was an existing node that is already done, we're done.
  if (N->getNodeId() != NewNode && N->getNodeId() != Unanalyzed)
    return N;

  // Remove any stale map entries.
  ExpungeNode(N);

  // Okay, we know that this node is new.  Recursively walk all of its operands
  // to see if they are new also.  The depth of this walk is bounded by the size
  // of the new tree that was constructed (usually 2-3 nodes), so we don't worry
  // about revisiting of nodes.
  //
  // As we walk the operands, keep track of the number of nodes that are
  // processed.  If non-zero, this will become the new nodeid of this node.
  // Operands may morph when they are analyzed.  If so, the node will be
  // updated after all operands have been analyzed.  Since this is rare,
  // the code tries to minimize overhead in the non-morphing case.

  SmallVector<SDValue, 8> NewOps;
  unsigned NumProcessed = 0;
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i) {
    SDValue OrigOp = N->getOperand(i);
    SDValue Op = OrigOp;

    AnalyzeNewValue(Op); // Op may morph.

    if (Op.getNode()->getNodeId() == Processed)
      ++NumProcessed;

    if (!NewOps.empty()) {
      // Some previous operand changed.  Add this one to the list.
      NewOps.push_back(Op);
    } else if (Op != OrigOp) {
      // This is the first operand to change - add all operands so far.
      NewOps.append(N->op_begin(), N->op_begin() + i);
      NewOps.push_back(Op);
    }
  }

  // Some operands changed - update the node.
  if (!NewOps.empty()) {
    SDNode *M = DAG.UpdateNodeOperands(N, NewOps);
    if (M != N) {
      // The node morphed into a different node.  Normally for this to happen
      // the original node would have to be marked NewNode.  However this can
      // in theory momentarily not be the case while ReplaceValueWith is doing
      // its stuff.  Mark the original node NewNode to help sanity checking.
      N->setNodeId(NewNode);
      if (M->getNodeId() != NewNode && M->getNodeId() != Unanalyzed)
        // It morphed into a previously analyzed node - nothing more to do.
        return M;

      // It morphed into a different new node.  Do the equivalent of passing
      // it to AnalyzeNewNode: expunge it and calculate the NodeId.  No need
      // to remap the operands, since they are the same as the operands we
      // remapped above.
      N = M;
      ExpungeNode(N);
    }
  }

  // Calculate the NodeId.
  N->setNodeId(N->getNumOperands() - NumProcessed);
  if (N->getNodeId() == ReadyToProcess)
    Worklist.push_back(N);

  return N;
}

/// AnalyzeNewValue - Call AnalyzeNewNode, updating the node in Val if needed.
/// If the node changes to a processed node, then remap it.
void DAGTypeLegalizer::AnalyzeNewValue(SDValue &Val) {
  Val.setNode(AnalyzeNewNode(Val.getNode()));
  if (Val.getNode()->getNodeId() == Processed)
    // We were passed a processed node, or it morphed into one - remap it.
    RemapValue(Val);
}

/// ExpungeNode - If N has a bogus mapping in ReplacedValues, eliminate it.
/// This can occur when a node is deleted then reallocated as a new node -
/// the mapping in ReplacedValues applies to the deleted node, not the new
/// one.
/// The only map that can have a deleted node as a source is ReplacedValues.
/// Other maps can have deleted nodes as targets, but since their looked-up
/// values are always immediately remapped using RemapValue, resulting in a
/// not-deleted node, this is harmless as long as ReplacedValues/RemapValue
/// always performs correct mappings.  In order to keep the mapping correct,
/// ExpungeNode should be called on any new nodes *before* adding them as
/// either source or target to ReplacedValues (which typically means calling
/// Expunge when a new node is first seen, since it may no longer be marked
/// NewNode by the time it is added to ReplacedValues).
void DAGTypeLegalizer::ExpungeNode(SDNode *N) {
  if (N->getNodeId() != NewNode)
    return;

  // If N is not remapped by ReplacedValues then there is nothing to do.
  unsigned i, e;
  for (i = 0, e = N->getNumValues(); i != e; ++i)
    if (ReplacedValues.find(SDValue(N, i)) != ReplacedValues.end())
      break;

  if (i == e)
    return;

  // Remove N from all maps - this is expensive but rare.

  for (DenseMap<SDValue, SDValue>::iterator I = PromotedIntegers.begin(),
       E = PromotedIntegers.end(); I != E; ++I) {
    assert(I->first.getNode() != N);
    RemapValue(I->second);
  }

  for (DenseMap<SDValue, SDValue>::iterator I = SoftenedFloats.begin(),
       E = SoftenedFloats.end(); I != E; ++I) {
    assert(I->first.getNode() != N);
    RemapValue(I->second);
  }

  for (DenseMap<SDValue, SDValue>::iterator I = ScalarizedVectors.begin(),
       E = ScalarizedVectors.end(); I != E; ++I) {
    assert(I->first.getNode() != N);
    RemapValue(I->second);
  }

  for (DenseMap<SDValue, SDValue>::iterator I = WidenedVectors.begin(),
       E = WidenedVectors.end(); I != E; ++I) {
    assert(I->first.getNode() != N);
    RemapValue(I->second);
  }

  for (DenseMap<SDValue, std::pair<SDValue, SDValue> >::iterator
       I = ExpandedIntegers.begin(), E = ExpandedIntegers.end(); I != E; ++I){
    assert(I->first.getNode() != N);
    RemapValue(I->second.first);
    RemapValue(I->second.second);
  }

  for (DenseMap<SDValue, std::pair<SDValue, SDValue> >::iterator
       I = ExpandedFloats.begin(), E = ExpandedFloats.end(); I != E; ++I) {
    assert(I->first.getNode() != N);
    RemapValue(I->second.first);
    RemapValue(I->second.second);
  }

  for (DenseMap<SDValue, std::pair<SDValue, SDValue> >::iterator
       I = SplitVectors.begin(), E = SplitVectors.end(); I != E; ++I) {
    assert(I->first.getNode() != N);
    RemapValue(I->second.first);
    RemapValue(I->second.second);
  }

  for (DenseMap<SDValue, SDValue>::iterator I = ReplacedValues.begin(),
       E = ReplacedValues.end(); I != E; ++I)
    RemapValue(I->second);

  for (unsigned i = 0, e = N->getNumValues(); i != e; ++i)
    ReplacedValues.erase(SDValue(N, i));
}

/// RemapValue - If the specified value was already legalized to another value,
/// replace it by that value.
void DAGTypeLegalizer::RemapValue(SDValue &N) {
  DenseMap<SDValue, SDValue>::iterator I = ReplacedValues.find(N);
  if (I != ReplacedValues.end()) {
    // Use path compression to speed up future lookups if values get multiply
    // replaced with other values.
    RemapValue(I->second);
    N = I->second;

    // Note that it is possible to have N.getNode()->getNodeId() == NewNode at
    // this point because it is possible for a node to be put in the map before
    // being processed.
  }
}

namespace {
  /// NodeUpdateListener - This class is a DAGUpdateListener that listens for
  /// updates to nodes and recomputes their ready state.
  class NodeUpdateListener : public SelectionDAG::DAGUpdateListener {
    DAGTypeLegalizer &DTL;
    SmallSetVector<SDNode*, 16> &NodesToAnalyze;
  public:
    explicit NodeUpdateListener(DAGTypeLegalizer &dtl,
                                SmallSetVector<SDNode*, 16> &nta)
      : SelectionDAG::DAGUpdateListener(dtl.getDAG()),
        DTL(dtl), NodesToAnalyze(nta) {}

    void NodeDeleted(SDNode *N, SDNode *E) override {
      assert(N->getNodeId() != DAGTypeLegalizer::ReadyToProcess &&
             N->getNodeId() != DAGTypeLegalizer::Processed &&
             "Invalid node ID for RAUW deletion!");
      // It is possible, though rare, for the deleted node N to occur as a
      // target in a map, so note the replacement N -> E in ReplacedValues.
      assert(E && "Node not replaced?");
      DTL.NoteDeletion(N, E);

      // In theory the deleted node could also have been scheduled for analysis.
      // So remove it from the set of nodes which will be analyzed.
      NodesToAnalyze.remove(N);

      // In general nothing needs to be done for E, since it didn't change but
      // only gained new uses.  However N -> E was just added to ReplacedValues,
      // and the result of a ReplacedValues mapping is not allowed to be marked
      // NewNode.  So if E is marked NewNode, then it needs to be analyzed.
      if (E->getNodeId() == DAGTypeLegalizer::NewNode)
        NodesToAnalyze.insert(E);
    }

    void NodeUpdated(SDNode *N) override {
      // Node updates can mean pretty much anything.  It is possible that an
      // operand was set to something already processed (f.e.) in which case
      // this node could become ready.  Recompute its flags.
      assert(N->getNodeId() != DAGTypeLegalizer::ReadyToProcess &&
             N->getNodeId() != DAGTypeLegalizer::Processed &&
             "Invalid node ID for RAUW deletion!");
      N->setNodeId(DAGTypeLegalizer::NewNode);
      NodesToAnalyze.insert(N);
    }
  };
}


/// ReplaceValueWith - The specified value was legalized to the specified other
/// value.  Update the DAG and NodeIds replacing any uses of From to use To
/// instead.
void DAGTypeLegalizer::ReplaceValueWith(SDValue From, SDValue To) {
  assert(From.getNode() != To.getNode() && "Potential legalization loop!");

  // If expansion produced new nodes, make sure they are properly marked.
  ExpungeNode(From.getNode());
  AnalyzeNewValue(To); // Expunges To.

  // Anything that used the old node should now use the new one.  Note that this
  // can potentially cause recursive merging.
  SmallSetVector<SDNode*, 16> NodesToAnalyze;
  NodeUpdateListener NUL(*this, NodesToAnalyze);
  do {
    DAG.ReplaceAllUsesOfValueWith(From, To);

    // The old node may still be present in a map like ExpandedIntegers or
    // PromotedIntegers.  Inform maps about the replacement.
    ReplacedValues[From] = To;

    // Process the list of nodes that need to be reanalyzed.
    while (!NodesToAnalyze.empty()) {
      SDNode *N = NodesToAnalyze.back();
      NodesToAnalyze.pop_back();
      if (N->getNodeId() != DAGTypeLegalizer::NewNode)
        // The node was analyzed while reanalyzing an earlier node - it is safe
        // to skip.  Note that this is not a morphing node - otherwise it would
        // still be marked NewNode.
        continue;

      // Analyze the node's operands and recalculate the node ID.
      SDNode *M = AnalyzeNewNode(N);
      if (M != N) {
        // The node morphed into a different node.  Make everyone use the new
        // node instead.
        assert(M->getNodeId() != NewNode && "Analysis resulted in NewNode!");
        assert(N->getNumValues() == M->getNumValues() &&
               "Node morphing changed the number of results!");
        for (unsigned i = 0, e = N->getNumValues(); i != e; ++i) {
          SDValue OldVal(N, i);
          SDValue NewVal(M, i);
          if (M->getNodeId() == Processed)
            RemapValue(NewVal);
          DAG.ReplaceAllUsesOfValueWith(OldVal, NewVal);
          // OldVal may be a target of the ReplacedValues map which was marked
          // NewNode to force reanalysis because it was updated.  Ensure that
          // anything that ReplacedValues mapped to OldVal will now be mapped
          // all the way to NewVal.
          ReplacedValues[OldVal] = NewVal;
        }
        // The original node continues to exist in the DAG, marked NewNode.
      }
    }
    // When recursively update nodes with new nodes, it is possible to have
    // new uses of From due to CSE. If this happens, replace the new uses of
    // From with To.
  } while (!From.use_empty());
}

void DAGTypeLegalizer::SetPromotedInteger(SDValue Op, SDValue Result) {
  assert(Result.getValueType() ==
         TLI.getTypeToTransformTo(*DAG.getContext(), Op.getValueType()) &&
         "Invalid type for promoted integer");
  AnalyzeNewValue(Result);

  SDValue &OpEntry = PromotedIntegers[Op];
  assert(!OpEntry.getNode() && "Node is already promoted!");
  OpEntry = Result;
}

void DAGTypeLegalizer::SetSoftenedFloat(SDValue Op, SDValue Result) {
  assert(Result.getValueType() ==
         TLI.getTypeToTransformTo(*DAG.getContext(), Op.getValueType()) &&
         "Invalid type for softened float");
  AnalyzeNewValue(Result);

  SDValue &OpEntry = SoftenedFloats[Op];
  assert(!OpEntry.getNode() && "Node is already converted to integer!");
  OpEntry = Result;
}

void DAGTypeLegalizer::SetPromotedFloat(SDValue Op, SDValue Result) {
  assert(Result.getValueType() ==
         TLI.getTypeToTransformTo(*DAG.getContext(), Op.getValueType()) &&
         "Invalid type for promoted float");
  AnalyzeNewValue(Result);

  SDValue &OpEntry = PromotedFloats[Op];
  assert(!OpEntry.getNode() && "Node is already promoted!");
  OpEntry = Result;
}

void DAGTypeLegalizer::SetScalarizedVector(SDValue Op, SDValue Result) {
  // Note that in some cases vector operation operands may be greater than
  // the vector element type. For example BUILD_VECTOR of type <1 x i1> with
  // a constant i8 operand.
  assert(Result.getValueType().getSizeInBits() >=
         Op.getValueType().getVectorElementType().getSizeInBits() &&
         "Invalid type for scalarized vector");
  AnalyzeNewValue(Result);

  SDValue &OpEntry = ScalarizedVectors[Op];
  assert(!OpEntry.getNode() && "Node is already scalarized!");
  OpEntry = Result;
}

void DAGTypeLegalizer::GetExpandedInteger(SDValue Op, SDValue &Lo,
                                          SDValue &Hi) {
  std::pair<SDValue, SDValue> &Entry = ExpandedIntegers[Op];
  RemapValue(Entry.first);
  RemapValue(Entry.second);
  assert(Entry.first.getNode() && "Operand isn't expanded");
  Lo = Entry.first;
  Hi = Entry.second;
}

void DAGTypeLegalizer::SetExpandedInteger(SDValue Op, SDValue Lo,
                                          SDValue Hi) {
  assert(Lo.getValueType() ==
         TLI.getTypeToTransformTo(*DAG.getContext(), Op.getValueType()) &&
         Hi.getValueType() == Lo.getValueType() &&
         "Invalid type for expanded integer");
  // Lo/Hi may have been newly allocated, if so, add nodeid's as relevant.
  AnalyzeNewValue(Lo);
  AnalyzeNewValue(Hi);

  // Remember that this is the result of the node.
  std::pair<SDValue, SDValue> &Entry = ExpandedIntegers[Op];
  assert(!Entry.first.getNode() && "Node already expanded");
  Entry.first = Lo;
  Entry.second = Hi;
}

void DAGTypeLegalizer::GetExpandedFloat(SDValue Op, SDValue &Lo,
                                        SDValue &Hi) {
  std::pair<SDValue, SDValue> &Entry = ExpandedFloats[Op];
  RemapValue(Entry.first);
  RemapValue(Entry.second);
  assert(Entry.first.getNode() && "Operand isn't expanded");
  Lo = Entry.first;
  Hi = Entry.second;
}

void DAGTypeLegalizer::SetExpandedFloat(SDValue Op, SDValue Lo,
                                        SDValue Hi) {
  assert(Lo.getValueType() ==
         TLI.getTypeToTransformTo(*DAG.getContext(), Op.getValueType()) &&
         Hi.getValueType() == Lo.getValueType() &&
         "Invalid type for expanded float");
  // Lo/Hi may have been newly allocated, if so, add nodeid's as relevant.
  AnalyzeNewValue(Lo);
  AnalyzeNewValue(Hi);

  // Remember that this is the result of the node.
  std::pair<SDValue, SDValue> &Entry = ExpandedFloats[Op];
  assert(!Entry.first.getNode() && "Node already expanded");
  Entry.first = Lo;
  Entry.second = Hi;
}

void DAGTypeLegalizer::GetSplitVector(SDValue Op, SDValue &Lo,
                                      SDValue &Hi) {
  std::pair<SDValue, SDValue> &Entry = SplitVectors[Op];
  RemapValue(Entry.first);
  RemapValue(Entry.second);
  assert(Entry.first.getNode() && "Operand isn't split");
  Lo = Entry.first;
  Hi = Entry.second;
}

void DAGTypeLegalizer::SetSplitVector(SDValue Op, SDValue Lo,
                                      SDValue Hi) {
  assert(Lo.getValueType().getVectorElementType() ==
         Op.getValueType().getVectorElementType() &&
         2*Lo.getValueType().getVectorNumElements() ==
         Op.getValueType().getVectorNumElements() &&
         Hi.getValueType() == Lo.getValueType() &&
         "Invalid type for split vector");
  // Lo/Hi may have been newly allocated, if so, add nodeid's as relevant.
  AnalyzeNewValue(Lo);
  AnalyzeNewValue(Hi);

  // Remember that this is the result of the node.
  std::pair<SDValue, SDValue> &Entry = SplitVectors[Op];
  assert(!Entry.first.getNode() && "Node already split");
  Entry.first = Lo;
  Entry.second = Hi;
}

void DAGTypeLegalizer::SetWidenedVector(SDValue Op, SDValue Result) {
  assert(Result.getValueType() ==
         TLI.getTypeToTransformTo(*DAG.getContext(), Op.getValueType()) &&
         "Invalid type for widened vector");
  AnalyzeNewValue(Result);

  SDValue &OpEntry = WidenedVectors[Op];
  assert(!OpEntry.getNode() && "Node already widened!");
  OpEntry = Result;
}


//===----------------------------------------------------------------------===//
// Utilities.
//===----------------------------------------------------------------------===//

/// BitConvertToInteger - Convert to an integer of the same size.
SDValue DAGTypeLegalizer::BitConvertToInteger(SDValue Op) {
  unsigned BitWidth = Op.getValueType().getSizeInBits();
  return DAG.getNode(ISD::BITCAST, SDLoc(Op),
                     EVT::getIntegerVT(*DAG.getContext(), BitWidth), Op);
}

/// BitConvertVectorToIntegerVector - Convert to a vector of integers of the
/// same size.
SDValue DAGTypeLegalizer::BitConvertVectorToIntegerVector(SDValue Op) {
  assert(Op.getValueType().isVector() && "Only applies to vectors!");
  unsigned EltWidth = Op.getValueType().getVectorElementType().getSizeInBits();
  EVT EltNVT = EVT::getIntegerVT(*DAG.getContext(), EltWidth);
  unsigned NumElts = Op.getValueType().getVectorNumElements();
  return DAG.getNode(ISD::BITCAST, SDLoc(Op),
                     EVT::getVectorVT(*DAG.getContext(), EltNVT, NumElts), Op);
}

SDValue DAGTypeLegalizer::CreateStackStoreLoad(SDValue Op,
                                               EVT DestVT) {
  SDLoc dl(Op);
  // Create the stack frame object.  Make sure it is aligned for both
  // the source and destination types.
  SDValue StackPtr = DAG.CreateStackTemporary(Op.getValueType(), DestVT);
  // Emit a store to the stack slot.
  SDValue Store = DAG.getStore(DAG.getEntryNode(), dl, Op, StackPtr,
                               MachinePointerInfo(), false, false, 0);
  // Result is a load from the stack slot.
  return DAG.getLoad(DestVT, dl, Store, StackPtr, MachinePointerInfo(),
                     false, false, false, 0);
}

/// CustomLowerNode - Replace the node's results with custom code provided
/// by the target and return "true", or do nothing and return "false".
/// The last parameter is FALSE if we are dealing with a node with legal
/// result types and illegal operand. The second parameter denotes the type of
/// illegal OperandNo in that case.
/// The last parameter being TRUE means we are dealing with a
/// node with illegal result types. The second parameter denotes the type of
/// illegal ResNo in that case.
bool DAGTypeLegalizer::CustomLowerNode(SDNode *N, EVT VT, bool LegalizeResult) {
  // See if the target wants to custom lower this node.
  if (TLI.getOperationAction(N->getOpcode(), VT) != TargetLowering::Custom)
    return false;

  SmallVector<SDValue, 8> Results;
  if (LegalizeResult)
    TLI.ReplaceNodeResults(N, Results, DAG);
  else
    TLI.LowerOperationWrapper(N, Results, DAG);

  if (Results.empty())
    // The target didn't want to custom lower it after all.
    return false;

  // When called from DAGTypeLegalizer::ExpandIntegerResult, we might need to
  // provide the same kind of custom splitting behavior.
  if (Results.size() == N->getNumValues() + 1 && LegalizeResult) {
    // We've legalized a return type by splitting it. If there is a chain,
    // replace that too.
    SetExpandedInteger(SDValue(N, 0), Results[0], Results[1]);
    if (N->getNumValues() > 1)
      ReplaceValueWith(SDValue(N, 1), Results[2]);
    return true;
  }

  // Make everything that once used N's values now use those in Results instead.
  assert(Results.size() == N->getNumValues() &&
         "Custom lowering returned the wrong number of results!");
  for (unsigned i = 0, e = Results.size(); i != e; ++i) {
    ReplaceValueWith(SDValue(N, i), Results[i]);
  }
  return true;
}


/// CustomWidenLowerNode - Widen the node's results with custom code provided
/// by the target and return "true", or do nothing and return "false".
bool DAGTypeLegalizer::CustomWidenLowerNode(SDNode *N, EVT VT) {
  // See if the target wants to custom lower this node.
  if (TLI.getOperationAction(N->getOpcode(), VT) != TargetLowering::Custom)
    return false;

  SmallVector<SDValue, 8> Results;
  TLI.ReplaceNodeResults(N, Results, DAG);

  if (Results.empty())
    // The target didn't want to custom widen lower its result  after all.
    return false;

  // Update the widening map.
  assert(Results.size() == N->getNumValues() &&
         "Custom lowering returned the wrong number of results!");
  for (unsigned i = 0, e = Results.size(); i != e; ++i)
    SetWidenedVector(SDValue(N, i), Results[i]);
  return true;
}

SDValue DAGTypeLegalizer::DisintegrateMERGE_VALUES(SDNode *N, unsigned ResNo) {
  for (unsigned i = 0, e = N->getNumValues(); i != e; ++i)
    if (i != ResNo)
      ReplaceValueWith(SDValue(N, i), SDValue(N->getOperand(i)));
  return SDValue(N->getOperand(ResNo));
}

/// GetPairElements - Use ISD::EXTRACT_ELEMENT nodes to extract the low and
/// high parts of the given value.
void DAGTypeLegalizer::GetPairElements(SDValue Pair,
                                       SDValue &Lo, SDValue &Hi) {
  SDLoc dl(Pair);
  EVT NVT = TLI.getTypeToTransformTo(*DAG.getContext(), Pair.getValueType());
  Lo = DAG.getNode(ISD::EXTRACT_ELEMENT, dl, NVT, Pair,
                   DAG.getIntPtrConstant(0, dl));
  Hi = DAG.getNode(ISD::EXTRACT_ELEMENT, dl, NVT, Pair,
                   DAG.getIntPtrConstant(1, dl));
}

SDValue DAGTypeLegalizer::GetVectorElementPointer(SDValue VecPtr, EVT EltVT,
                                                  SDValue Index) {
  SDLoc dl(Index);
  // Make sure the index type is big enough to compute in.
  Index = DAG.getZExtOrTrunc(Index, dl, TLI.getPointerTy(DAG.getDataLayout()));

  // Calculate the element offset and add it to the pointer.
  unsigned EltSize = EltVT.getSizeInBits() / 8; // FIXME: should be ABI size.

  Index = DAG.getNode(ISD::MUL, dl, Index.getValueType(), Index,
                      DAG.getConstant(EltSize, dl, Index.getValueType()));
  return DAG.getNode(ISD::ADD, dl, Index.getValueType(), Index, VecPtr);
}

/// JoinIntegers - Build an integer with low bits Lo and high bits Hi.
SDValue DAGTypeLegalizer::JoinIntegers(SDValue Lo, SDValue Hi) {
  // Arbitrarily use dlHi for result SDLoc
  SDLoc dlHi(Hi);
  SDLoc dlLo(Lo);
  EVT LVT = Lo.getValueType();
  EVT HVT = Hi.getValueType();
  EVT NVT = EVT::getIntegerVT(*DAG.getContext(),
                              LVT.getSizeInBits() + HVT.getSizeInBits());

  Lo = DAG.getNode(ISD::ZERO_EXTEND, dlLo, NVT, Lo);
  Hi = DAG.getNode(ISD::ANY_EXTEND, dlHi, NVT, Hi);
  Hi = DAG.getNode(ISD::SHL, dlHi, NVT, Hi,
                   DAG.getConstant(LVT.getSizeInBits(), dlHi,
                                   TLI.getPointerTy(DAG.getDataLayout())));
  return DAG.getNode(ISD::OR, dlHi, NVT, Lo, Hi);
}

/// LibCallify - Convert the node into a libcall with the same prototype.
SDValue DAGTypeLegalizer::LibCallify(RTLIB::Libcall LC, SDNode *N,
                                     bool isSigned) {
  unsigned NumOps = N->getNumOperands();
  SDLoc dl(N);
  if (NumOps == 0) {
    return TLI.makeLibCall(DAG, LC, N->getValueType(0), nullptr, 0, isSigned,
                           dl).first;
  } else if (NumOps == 1) {
    SDValue Op = N->getOperand(0);
    return TLI.makeLibCall(DAG, LC, N->getValueType(0), &Op, 1, isSigned,
                           dl).first;
  } else if (NumOps == 2) {
    SDValue Ops[2] = { N->getOperand(0), N->getOperand(1) };
    return TLI.makeLibCall(DAG, LC, N->getValueType(0), Ops, 2, isSigned,
                           dl).first;
  }
  SmallVector<SDValue, 8> Ops(NumOps);
  for (unsigned i = 0; i < NumOps; ++i)
    Ops[i] = N->getOperand(i);

  return TLI.makeLibCall(DAG, LC, N->getValueType(0),
                         &Ops[0], NumOps, isSigned, dl).first;
}

// ExpandChainLibCall - Expand a node into a call to a libcall. Similar to
// ExpandLibCall except that the first operand is the in-chain.
std::pair<SDValue, SDValue>
DAGTypeLegalizer::ExpandChainLibCall(RTLIB::Libcall LC,
                                         SDNode *Node,
                                         bool isSigned) {
  SDValue InChain = Node->getOperand(0);

  TargetLowering::ArgListTy Args;
  TargetLowering::ArgListEntry Entry;
  for (unsigned i = 1, e = Node->getNumOperands(); i != e; ++i) {
    EVT ArgVT = Node->getOperand(i).getValueType();
    Type *ArgTy = ArgVT.getTypeForEVT(*DAG.getContext());
    Entry.Node = Node->getOperand(i);
    Entry.Ty = ArgTy;
    Entry.isSExt = isSigned;
    Entry.isZExt = !isSigned;
    Args.push_back(Entry);
  }
  SDValue Callee = DAG.getExternalSymbol(TLI.getLibcallName(LC),
                                         TLI.getPointerTy(DAG.getDataLayout()));

  Type *RetTy = Node->getValueType(0).getTypeForEVT(*DAG.getContext());

  TargetLowering::CallLoweringInfo CLI(DAG);
  CLI.setDebugLoc(SDLoc(Node)).setChain(InChain)
    .setCallee(TLI.getLibcallCallingConv(LC), RetTy, Callee, std::move(Args), 0)
    .setSExtResult(isSigned).setZExtResult(!isSigned);

  std::pair<SDValue, SDValue> CallInfo = TLI.LowerCallTo(CLI);

  return CallInfo;
}

/// PromoteTargetBoolean - Promote the given target boolean to a target boolean
/// of the given type.  A target boolean is an integer value, not necessarily of
/// type i1, the bits of which conform to getBooleanContents.
///
/// ValVT is the type of values that produced the boolean.
SDValue DAGTypeLegalizer::PromoteTargetBoolean(SDValue Bool, EVT ValVT) {
  SDLoc dl(Bool);
  EVT BoolVT = getSetCCResultType(ValVT);
  ISD::NodeType ExtendCode =
      TargetLowering::getExtendForContent(TLI.getBooleanContents(ValVT));
  return DAG.getNode(ExtendCode, dl, BoolVT, Bool);
}

/// SplitInteger - Return the lower LoVT bits of Op in Lo and the upper HiVT
/// bits in Hi.
void DAGTypeLegalizer::SplitInteger(SDValue Op,
                                    EVT LoVT, EVT HiVT,
                                    SDValue &Lo, SDValue &Hi) {
  SDLoc dl(Op);
  assert(LoVT.getSizeInBits() + HiVT.getSizeInBits() ==
         Op.getValueType().getSizeInBits() && "Invalid integer splitting!");
  Lo = DAG.getNode(ISD::TRUNCATE, dl, LoVT, Op);
  Hi = DAG.getNode(ISD::SRL, dl, Op.getValueType(), Op,
                   DAG.getConstant(LoVT.getSizeInBits(), dl,
                                   TLI.getPointerTy(DAG.getDataLayout())));
  Hi = DAG.getNode(ISD::TRUNCATE, dl, HiVT, Hi);
}

/// SplitInteger - Return the lower and upper halves of Op's bits in a value
/// type half the size of Op's.
void DAGTypeLegalizer::SplitInteger(SDValue Op,
                                    SDValue &Lo, SDValue &Hi) {
  EVT HalfVT = EVT::getIntegerVT(*DAG.getContext(),
                                 Op.getValueType().getSizeInBits()/2);
  SplitInteger(Op, HalfVT, HalfVT, Lo, Hi);
}


//===----------------------------------------------------------------------===//
//  Entry Point
//===----------------------------------------------------------------------===//

/// LegalizeTypes - This transforms the SelectionDAG into a SelectionDAG that
/// only uses types natively supported by the target.  Returns "true" if it made
/// any changes.
///
/// Note that this is an involved process that may invalidate pointers into
/// the graph.
bool SelectionDAG::LegalizeTypes() {
  return DAGTypeLegalizer(*this).run();
}
