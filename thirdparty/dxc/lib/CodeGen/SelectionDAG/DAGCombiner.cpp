//===-- DAGCombiner.cpp - Implement a DAG node combiner -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass combines dag nodes to form fewer, simpler DAG nodes.  It can be run
// both before and after the DAG is legalized.
//
// This pass is not a substitute for the LLVM IR instcombine pass. This pass is
// primarily intended to handle simplification opportunities that are implicit
// in the LLVM IR and exposed by the various codegen lowering phases.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include <algorithm>
using namespace llvm;

#define DEBUG_TYPE "dagcombine"

STATISTIC(NodesCombined   , "Number of dag nodes combined");
STATISTIC(PreIndexedNodes , "Number of pre-indexed nodes created");
STATISTIC(PostIndexedNodes, "Number of post-indexed nodes created");
STATISTIC(OpsNarrowed     , "Number of load/op/store narrowed");
STATISTIC(LdStFP2Int      , "Number of fp load/store pairs transformed to int");
STATISTIC(SlicedLoads, "Number of load sliced");

namespace {
  static cl::opt<bool>
    CombinerAA("combiner-alias-analysis", cl::Hidden,
               cl::desc("Enable DAG combiner alias-analysis heuristics"));

  static cl::opt<bool>
    CombinerGlobalAA("combiner-global-alias-analysis", cl::Hidden,
               cl::desc("Enable DAG combiner's use of IR alias analysis"));

  static cl::opt<bool>
    UseTBAA("combiner-use-tbaa", cl::Hidden, cl::init(true),
               cl::desc("Enable DAG combiner's use of TBAA"));

#ifndef NDEBUG
  static cl::opt<std::string>
    CombinerAAOnlyFunc("combiner-aa-only-func", cl::Hidden,
               cl::desc("Only use DAG-combiner alias analysis in this"
                        " function"));
#endif

  /// Hidden option to stress test load slicing, i.e., when this option
  /// is enabled, load slicing bypasses most of its profitability guards.
  static cl::opt<bool>
  StressLoadSlicing("combiner-stress-load-slicing", cl::Hidden,
                    cl::desc("Bypass the profitability model of load "
                             "slicing"),
                    cl::init(false));

  static cl::opt<bool>
    MaySplitLoadIndex("combiner-split-load-index", cl::Hidden, cl::init(true),
                      cl::desc("DAG combiner may split indexing from loads"));

//------------------------------ DAGCombiner ---------------------------------//

  class DAGCombiner {
    SelectionDAG &DAG;
    const TargetLowering &TLI;
    CombineLevel Level;
    CodeGenOpt::Level OptLevel;
    bool LegalOperations;
    bool LegalTypes;
    bool ForCodeSize;

    /// \brief Worklist of all of the nodes that need to be simplified.
    ///
    /// This must behave as a stack -- new nodes to process are pushed onto the
    /// back and when processing we pop off of the back.
    ///
    /// The worklist will not contain duplicates but may contain null entries
    /// due to nodes being deleted from the underlying DAG.
    SmallVector<SDNode *, 64> Worklist;

    /// \brief Mapping from an SDNode to its position on the worklist.
    ///
    /// This is used to find and remove nodes from the worklist (by nulling
    /// them) when they are deleted from the underlying DAG. It relies on
    /// stable indices of nodes within the worklist.
    DenseMap<SDNode *, unsigned> WorklistMap;

    /// \brief Set of nodes which have been combined (at least once).
    ///
    /// This is used to allow us to reliably add any operands of a DAG node
    /// which have not yet been combined to the worklist.
    SmallPtrSet<SDNode *, 64> CombinedNodes;

    // AA - Used for DAG load/store alias analysis.
    AliasAnalysis &AA;

    /// When an instruction is simplified, add all users of the instruction to
    /// the work lists because they might get more simplified now.
    void AddUsersToWorklist(SDNode *N) {
      for (SDNode *Node : N->uses())
        AddToWorklist(Node);
    }

    /// Call the node-specific routine that folds each particular type of node.
    SDValue visit(SDNode *N);

  public:
    /// Add to the worklist making sure its instance is at the back (next to be
    /// processed.)
    void AddToWorklist(SDNode *N) {
      // Skip handle nodes as they can't usefully be combined and confuse the
      // zero-use deletion strategy.
      if (N->getOpcode() == ISD::HANDLENODE)
        return;

      if (WorklistMap.insert(std::make_pair(N, Worklist.size())).second)
        Worklist.push_back(N);
    }

    /// Remove all instances of N from the worklist.
    void removeFromWorklist(SDNode *N) {
      CombinedNodes.erase(N);

      auto It = WorklistMap.find(N);
      if (It == WorklistMap.end())
        return; // Not in the worklist.

      // Null out the entry rather than erasing it to avoid a linear operation.
      Worklist[It->second] = nullptr;
      WorklistMap.erase(It);
    }

    void deleteAndRecombine(SDNode *N);
    bool recursivelyDeleteUnusedNodes(SDNode *N);

    SDValue CombineTo(SDNode *N, const SDValue *To, unsigned NumTo,
                      bool AddTo = true);

    SDValue CombineTo(SDNode *N, SDValue Res, bool AddTo = true) {
      return CombineTo(N, &Res, 1, AddTo);
    }

    SDValue CombineTo(SDNode *N, SDValue Res0, SDValue Res1,
                      bool AddTo = true) {
      SDValue To[] = { Res0, Res1 };
      return CombineTo(N, To, 2, AddTo);
    }

    void CommitTargetLoweringOpt(const TargetLowering::TargetLoweringOpt &TLO);

  private:

    /// Check the specified integer node value to see if it can be simplified or
    /// if things it uses can be simplified by bit propagation.
    /// If so, return true.
    bool SimplifyDemandedBits(SDValue Op) {
      unsigned BitWidth = Op.getValueType().getScalarType().getSizeInBits();
      APInt Demanded = APInt::getAllOnesValue(BitWidth);
      return SimplifyDemandedBits(Op, Demanded);
    }

    bool SimplifyDemandedBits(SDValue Op, const APInt &Demanded);

    bool CombineToPreIndexedLoadStore(SDNode *N);
    bool CombineToPostIndexedLoadStore(SDNode *N);
    SDValue SplitIndexingFromLoad(LoadSDNode *LD);
    bool SliceUpLoad(SDNode *N);

    /// \brief Replace an ISD::EXTRACT_VECTOR_ELT of a load with a narrowed
    ///   load.
    ///
    /// \param EVE ISD::EXTRACT_VECTOR_ELT to be replaced.
    /// \param InVecVT type of the input vector to EVE with bitcasts resolved.
    /// \param EltNo index of the vector element to load.
    /// \param OriginalLoad load that EVE came from to be replaced.
    /// \returns EVE on success SDValue() on failure.
    SDValue ReplaceExtractVectorEltOfLoadWithNarrowedLoad(
        SDNode *EVE, EVT InVecVT, SDValue EltNo, LoadSDNode *OriginalLoad);
    void ReplaceLoadWithPromotedLoad(SDNode *Load, SDNode *ExtLoad);
    SDValue PromoteOperand(SDValue Op, EVT PVT, bool &Replace);
    SDValue SExtPromoteOperand(SDValue Op, EVT PVT);
    SDValue ZExtPromoteOperand(SDValue Op, EVT PVT);
    SDValue PromoteIntBinOp(SDValue Op);
    SDValue PromoteIntShiftOp(SDValue Op);
    SDValue PromoteExtend(SDValue Op);
    bool PromoteLoad(SDValue Op);

    void ExtendSetCCUses(const SmallVectorImpl<SDNode *> &SetCCs,
                         SDValue Trunc, SDValue ExtLoad, SDLoc DL,
                         ISD::NodeType ExtType);

    /// Call the node-specific routine that knows how to fold each
    /// particular type of node. If that doesn't do anything, try the
    /// target-specific DAG combines.
    SDValue combine(SDNode *N);

    // Visitation implementation - Implement dag node combining for different
    // node types.  The semantics are as follows:
    // Return Value:
    //   SDValue.getNode() == 0 - No change was made
    //   SDValue.getNode() == N - N was replaced, is dead and has been handled.
    //   otherwise              - N should be replaced by the returned Operand.
    //
    SDValue visitTokenFactor(SDNode *N);
    SDValue visitMERGE_VALUES(SDNode *N);
    SDValue visitADD(SDNode *N);
    SDValue visitSUB(SDNode *N);
    SDValue visitADDC(SDNode *N);
    SDValue visitSUBC(SDNode *N);
    SDValue visitADDE(SDNode *N);
    SDValue visitSUBE(SDNode *N);
    SDValue visitMUL(SDNode *N);
    SDValue visitSDIV(SDNode *N);
    SDValue visitUDIV(SDNode *N);
    SDValue visitSREM(SDNode *N);
    SDValue visitUREM(SDNode *N);
    SDValue visitMULHU(SDNode *N);
    SDValue visitMULHS(SDNode *N);
    SDValue visitSMUL_LOHI(SDNode *N);
    SDValue visitUMUL_LOHI(SDNode *N);
    SDValue visitSMULO(SDNode *N);
    SDValue visitUMULO(SDNode *N);
    SDValue visitSDIVREM(SDNode *N);
    SDValue visitUDIVREM(SDNode *N);
    SDValue visitAND(SDNode *N);
    SDValue visitANDLike(SDValue N0, SDValue N1, SDNode *LocReference);
    SDValue visitOR(SDNode *N);
    SDValue visitORLike(SDValue N0, SDValue N1, SDNode *LocReference);
    SDValue visitXOR(SDNode *N);
    SDValue SimplifyVBinOp(SDNode *N);
    SDValue visitSHL(SDNode *N);
    SDValue visitSRA(SDNode *N);
    SDValue visitSRL(SDNode *N);
    SDValue visitRotate(SDNode *N);
    SDValue visitBSWAP(SDNode *N);
    SDValue visitCTLZ(SDNode *N);
    SDValue visitCTLZ_ZERO_UNDEF(SDNode *N);
    SDValue visitCTTZ(SDNode *N);
    SDValue visitCTTZ_ZERO_UNDEF(SDNode *N);
    SDValue visitCTPOP(SDNode *N);
    SDValue visitSELECT(SDNode *N);
    SDValue visitVSELECT(SDNode *N);
    SDValue visitSELECT_CC(SDNode *N);
    SDValue visitSETCC(SDNode *N);
    SDValue visitSIGN_EXTEND(SDNode *N);
    SDValue visitZERO_EXTEND(SDNode *N);
    SDValue visitANY_EXTEND(SDNode *N);
    SDValue visitSIGN_EXTEND_INREG(SDNode *N);
    SDValue visitSIGN_EXTEND_VECTOR_INREG(SDNode *N);
    SDValue visitTRUNCATE(SDNode *N);
    SDValue visitBITCAST(SDNode *N);
    SDValue visitBUILD_PAIR(SDNode *N);
    SDValue visitFADD(SDNode *N);
    SDValue visitFSUB(SDNode *N);
    SDValue visitFMUL(SDNode *N);
    SDValue visitFMA(SDNode *N);
    SDValue visitFDIV(SDNode *N);
    SDValue visitFREM(SDNode *N);
    SDValue visitFSQRT(SDNode *N);
    SDValue visitFCOPYSIGN(SDNode *N);
    SDValue visitSINT_TO_FP(SDNode *N);
    SDValue visitUINT_TO_FP(SDNode *N);
    SDValue visitFP_TO_SINT(SDNode *N);
    SDValue visitFP_TO_UINT(SDNode *N);
    SDValue visitFP_ROUND(SDNode *N);
    SDValue visitFP_ROUND_INREG(SDNode *N);
    SDValue visitFP_EXTEND(SDNode *N);
    SDValue visitFNEG(SDNode *N);
    SDValue visitFABS(SDNode *N);
    SDValue visitFCEIL(SDNode *N);
    SDValue visitFTRUNC(SDNode *N);
    SDValue visitFFLOOR(SDNode *N);
    SDValue visitFMINNUM(SDNode *N);
    SDValue visitFMAXNUM(SDNode *N);
    SDValue visitBRCOND(SDNode *N);
    SDValue visitBR_CC(SDNode *N);
    SDValue visitLOAD(SDNode *N);
    SDValue visitSTORE(SDNode *N);
    SDValue visitINSERT_VECTOR_ELT(SDNode *N);
    SDValue visitEXTRACT_VECTOR_ELT(SDNode *N);
    SDValue visitBUILD_VECTOR(SDNode *N);
    SDValue visitCONCAT_VECTORS(SDNode *N);
    SDValue visitEXTRACT_SUBVECTOR(SDNode *N);
    SDValue visitVECTOR_SHUFFLE(SDNode *N);
    SDValue visitSCALAR_TO_VECTOR(SDNode *N);
    SDValue visitINSERT_SUBVECTOR(SDNode *N);
    SDValue visitMLOAD(SDNode *N);
    SDValue visitMSTORE(SDNode *N);
    SDValue visitMGATHER(SDNode *N);
    SDValue visitMSCATTER(SDNode *N);
    SDValue visitFP_TO_FP16(SDNode *N);

    SDValue visitFADDForFMACombine(SDNode *N);
    SDValue visitFSUBForFMACombine(SDNode *N);

    SDValue XformToShuffleWithZero(SDNode *N);
    SDValue ReassociateOps(unsigned Opc, SDLoc DL, SDValue LHS, SDValue RHS);

    SDValue visitShiftByConstant(SDNode *N, ConstantSDNode *Amt);

    bool SimplifySelectOps(SDNode *SELECT, SDValue LHS, SDValue RHS);
    SDValue SimplifyBinOpWithSameOpcodeHands(SDNode *N);
    SDValue SimplifySelect(SDLoc DL, SDValue N0, SDValue N1, SDValue N2);
    SDValue SimplifySelectCC(SDLoc DL, SDValue N0, SDValue N1, SDValue N2,
                             SDValue N3, ISD::CondCode CC,
                             bool NotExtCompare = false);
    SDValue SimplifySetCC(EVT VT, SDValue N0, SDValue N1, ISD::CondCode Cond,
                          SDLoc DL, bool foldBooleans = true);

    bool isSetCCEquivalent(SDValue N, SDValue &LHS, SDValue &RHS,
                           SDValue &CC) const;
    bool isOneUseSetCC(SDValue N) const;

    SDValue SimplifyNodeWithTwoResults(SDNode *N, unsigned LoOp,
                                         unsigned HiOp);
    SDValue CombineConsecutiveLoads(SDNode *N, EVT VT);
    SDValue CombineExtLoad(SDNode *N);
    SDValue ConstantFoldBITCASTofBUILD_VECTOR(SDNode *, EVT);
    SDValue BuildSDIV(SDNode *N);
    SDValue BuildSDIVPow2(SDNode *N);
    SDValue BuildUDIV(SDNode *N);
    SDValue BuildReciprocalEstimate(SDValue Op);
    SDValue BuildRsqrtEstimate(SDValue Op);
    SDValue BuildRsqrtNROneConst(SDValue Op, SDValue Est, unsigned Iterations);
    SDValue BuildRsqrtNRTwoConst(SDValue Op, SDValue Est, unsigned Iterations);
    SDValue MatchBSwapHWordLow(SDNode *N, SDValue N0, SDValue N1,
                               bool DemandHighBits = true);
    SDValue MatchBSwapHWord(SDNode *N, SDValue N0, SDValue N1);
    SDNode *MatchRotatePosNeg(SDValue Shifted, SDValue Pos, SDValue Neg,
                              SDValue InnerPos, SDValue InnerNeg,
                              unsigned PosOpcode, unsigned NegOpcode,
                              SDLoc DL);
    SDNode *MatchRotate(SDValue LHS, SDValue RHS, SDLoc DL);
    SDValue ReduceLoadWidth(SDNode *N);
    SDValue ReduceLoadOpStoreWidth(SDNode *N);
    SDValue TransformFPLoadStorePair(SDNode *N);
    SDValue reduceBuildVecExtToExtBuildVec(SDNode *N);
    SDValue reduceBuildVecConvertToConvertBuildVec(SDNode *N);

    SDValue GetDemandedBits(SDValue V, const APInt &Mask);

    /// Walk up chain skipping non-aliasing memory nodes,
    /// looking for aliasing nodes and adding them to the Aliases vector.
    void GatherAllAliases(SDNode *N, SDValue OriginalChain,
                          SmallVectorImpl<SDValue> &Aliases);

    /// Return true if there is any possibility that the two addresses overlap.
    bool isAlias(LSBaseSDNode *Op0, LSBaseSDNode *Op1) const;

    /// Walk up chain skipping non-aliasing memory nodes, looking for a better
    /// chain (aliasing node.)
    SDValue FindBetterChain(SDNode *N, SDValue Chain);

    /// Holds a pointer to an LSBaseSDNode as well as information on where it
    /// is located in a sequence of memory operations connected by a chain.
    struct MemOpLink {
      MemOpLink (LSBaseSDNode *N, int64_t Offset, unsigned Seq):
      MemNode(N), OffsetFromBase(Offset), SequenceNum(Seq) { }
      // Ptr to the mem node.
      LSBaseSDNode *MemNode;
      // Offset from the base ptr.
      int64_t OffsetFromBase;
      // What is the sequence number of this mem node.
      // Lowest mem operand in the DAG starts at zero.
      unsigned SequenceNum;
    };

    /// This is a helper function for MergeStoresOfConstantsOrVecElts. Returns a
    /// constant build_vector of the stored constant values in Stores.
    SDValue getMergedConstantVectorStore(SelectionDAG &DAG,
                                         SDLoc SL,
                                         ArrayRef<MemOpLink> Stores,
                                         EVT Ty) const;

    /// This is a helper function for MergeConsecutiveStores. When the source
    /// elements of the consecutive stores are all constants or all extracted
    /// vector elements, try to merge them into one larger store.
    /// \return True if a merged store was created.
    bool MergeStoresOfConstantsOrVecElts(SmallVectorImpl<MemOpLink> &StoreNodes,
                                         EVT MemVT, unsigned NumElem,
                                         bool IsConstantSrc, bool UseVector);

    /// This is a helper function for MergeConsecutiveStores.
    /// Stores that may be merged are placed in StoreNodes.
    /// Loads that may alias with those stores are placed in AliasLoadNodes.
    void getStoreMergeAndAliasCandidates(
        StoreSDNode* St, SmallVectorImpl<MemOpLink> &StoreNodes,
        SmallVectorImpl<LSBaseSDNode*> &AliasLoadNodes);
    
    /// Merge consecutive store operations into a wide store.
    /// This optimization uses wide integers or vectors when possible.
    /// \return True if some memory operations were changed.
    bool MergeConsecutiveStores(StoreSDNode *N);

    /// \brief Try to transform a truncation where C is a constant:
    ///     (trunc (and X, C)) -> (and (trunc X), (trunc C))
    ///
    /// \p N needs to be a truncation and its first operand an AND. Other
    /// requirements are checked by the function (e.g. that trunc is
    /// single-use) and if missed an empty SDValue is returned.
    SDValue distributeTruncateThroughAnd(SDNode *N);

  public:
    DAGCombiner(SelectionDAG &D, AliasAnalysis &A, CodeGenOpt::Level OL)
        : DAG(D), TLI(D.getTargetLoweringInfo()), Level(BeforeLegalizeTypes),
          OptLevel(OL), LegalOperations(false), LegalTypes(false), AA(A) {
      auto *F = DAG.getMachineFunction().getFunction();
      ForCodeSize = F->hasFnAttribute(Attribute::OptimizeForSize) ||
                    F->hasFnAttribute(Attribute::MinSize);
    }

    /// Runs the dag combiner on all nodes in the work list
    void Run(CombineLevel AtLevel);

    SelectionDAG &getDAG() const { return DAG; }

    /// Returns a type large enough to hold any valid shift amount - before type
    /// legalization these can be huge.
    EVT getShiftAmountTy(EVT LHSTy) {
      assert(LHSTy.isInteger() && "Shift amount is not an integer type!");
      if (LHSTy.isVector())
        return LHSTy;
      auto &DL = DAG.getDataLayout();
      return LegalTypes ? TLI.getScalarShiftAmountTy(DL, LHSTy)
                        : TLI.getPointerTy(DL);
    }

    /// This method returns true if we are running before type legalization or
    /// if the specified VT is legal.
    bool isTypeLegal(const EVT &VT) {
      if (!LegalTypes) return true;
      return TLI.isTypeLegal(VT);
    }

    /// Convenience wrapper around TargetLowering::getSetCCResultType
    EVT getSetCCResultType(EVT VT) const {
      return TLI.getSetCCResultType(DAG.getDataLayout(), *DAG.getContext(), VT);
    }
  };
}


namespace {
/// This class is a DAGUpdateListener that removes any deleted
/// nodes from the worklist.
class WorklistRemover : public SelectionDAG::DAGUpdateListener {
  DAGCombiner &DC;
public:
  explicit WorklistRemover(DAGCombiner &dc)
    : SelectionDAG::DAGUpdateListener(dc.getDAG()), DC(dc) {}

  void NodeDeleted(SDNode *N, SDNode *E) override {
    DC.removeFromWorklist(N);
  }
};
}

//===----------------------------------------------------------------------===//
//  TargetLowering::DAGCombinerInfo implementation
//===----------------------------------------------------------------------===//

void TargetLowering::DAGCombinerInfo::AddToWorklist(SDNode *N) {
  ((DAGCombiner*)DC)->AddToWorklist(N);
}

void TargetLowering::DAGCombinerInfo::RemoveFromWorklist(SDNode *N) {
  ((DAGCombiner*)DC)->removeFromWorklist(N);
}

SDValue TargetLowering::DAGCombinerInfo::
CombineTo(SDNode *N, ArrayRef<SDValue> To, bool AddTo) {
  return ((DAGCombiner*)DC)->CombineTo(N, &To[0], To.size(), AddTo);
}

SDValue TargetLowering::DAGCombinerInfo::
CombineTo(SDNode *N, SDValue Res, bool AddTo) {
  return ((DAGCombiner*)DC)->CombineTo(N, Res, AddTo);
}


SDValue TargetLowering::DAGCombinerInfo::
CombineTo(SDNode *N, SDValue Res0, SDValue Res1, bool AddTo) {
  return ((DAGCombiner*)DC)->CombineTo(N, Res0, Res1, AddTo);
}

void TargetLowering::DAGCombinerInfo::
CommitTargetLoweringOpt(const TargetLowering::TargetLoweringOpt &TLO) {
  return ((DAGCombiner*)DC)->CommitTargetLoweringOpt(TLO);
}

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

void DAGCombiner::deleteAndRecombine(SDNode *N) {
  removeFromWorklist(N);

  // If the operands of this node are only used by the node, they will now be
  // dead. Make sure to re-visit them and recursively delete dead nodes.
  for (const SDValue &Op : N->ops())
    // For an operand generating multiple values, one of the values may
    // become dead allowing further simplification (e.g. split index
    // arithmetic from an indexed load).
    if (Op->hasOneUse() || Op->getNumValues() > 1)
      AddToWorklist(Op.getNode());

  DAG.DeleteNode(N);
}

/// Return 1 if we can compute the negated form of the specified expression for
/// the same cost as the expression itself, or 2 if we can compute the negated
/// form more cheaply than the expression itself.
static char isNegatibleForFree(SDValue Op, bool LegalOperations,
                               const TargetLowering &TLI,
                               const TargetOptions *Options,
                               unsigned Depth = 0) {
  // fneg is removable even if it has multiple uses.
  if (Op.getOpcode() == ISD::FNEG) return 2;

  // Don't allow anything with multiple uses.
  if (!Op.hasOneUse()) return 0;

  // Don't recurse exponentially.
  if (Depth > 6) return 0;

  switch (Op.getOpcode()) {
  default: return false;
  case ISD::ConstantFP:
    // Don't invert constant FP values after legalize.  The negated constant
    // isn't necessarily legal.
    return LegalOperations ? 0 : 1;
  case ISD::FADD:
    // FIXME: determine better conditions for this xform.
    if (!Options->UnsafeFPMath) return 0;

    // After operation legalization, it might not be legal to create new FSUBs.
    if (LegalOperations &&
        !TLI.isOperationLegalOrCustom(ISD::FSUB,  Op.getValueType()))
      return 0;

    // fold (fneg (fadd A, B)) -> (fsub (fneg A), B)
    if (char V = isNegatibleForFree(Op.getOperand(0), LegalOperations, TLI,
                                    Options, Depth + 1))
      return V;
    // fold (fneg (fadd A, B)) -> (fsub (fneg B), A)
    return isNegatibleForFree(Op.getOperand(1), LegalOperations, TLI, Options,
                              Depth + 1);
  case ISD::FSUB:
    // We can't turn -(A-B) into B-A when we honor signed zeros.
    if (!Options->UnsafeFPMath) return 0;

    // fold (fneg (fsub A, B)) -> (fsub B, A)
    return 1;

  case ISD::FMUL:
  case ISD::FDIV:
    if (Options->HonorSignDependentRoundingFPMath()) return 0;

    // fold (fneg (fmul X, Y)) -> (fmul (fneg X), Y) or (fmul X, (fneg Y))
    if (char V = isNegatibleForFree(Op.getOperand(0), LegalOperations, TLI,
                                    Options, Depth + 1))
      return V;

    return isNegatibleForFree(Op.getOperand(1), LegalOperations, TLI, Options,
                              Depth + 1);

  case ISD::FP_EXTEND:
  case ISD::FP_ROUND:
  case ISD::FSIN:
    return isNegatibleForFree(Op.getOperand(0), LegalOperations, TLI, Options,
                              Depth + 1);
  }
}

/// If isNegatibleForFree returns true, return the newly negated expression.
static SDValue GetNegatedExpression(SDValue Op, SelectionDAG &DAG,
                                    bool LegalOperations, unsigned Depth = 0) {
  const TargetOptions &Options = DAG.getTarget().Options;
  // fneg is removable even if it has multiple uses.
  if (Op.getOpcode() == ISD::FNEG) return Op.getOperand(0);

  // Don't allow anything with multiple uses.
  assert(Op.hasOneUse() && "Unknown reuse!");

  assert(Depth <= 6 && "GetNegatedExpression doesn't match isNegatibleForFree");
  switch (Op.getOpcode()) {
  default: llvm_unreachable("Unknown code");
  case ISD::ConstantFP: {
    APFloat V = cast<ConstantFPSDNode>(Op)->getValueAPF();
    V.changeSign();
    return DAG.getConstantFP(V, SDLoc(Op), Op.getValueType());
  }
  case ISD::FADD:
    // FIXME: determine better conditions for this xform.
    assert(Options.UnsafeFPMath);

    // fold (fneg (fadd A, B)) -> (fsub (fneg A), B)
    if (isNegatibleForFree(Op.getOperand(0), LegalOperations,
                           DAG.getTargetLoweringInfo(), &Options, Depth+1))
      return DAG.getNode(ISD::FSUB, SDLoc(Op), Op.getValueType(),
                         GetNegatedExpression(Op.getOperand(0), DAG,
                                              LegalOperations, Depth+1),
                         Op.getOperand(1));
    // fold (fneg (fadd A, B)) -> (fsub (fneg B), A)
    return DAG.getNode(ISD::FSUB, SDLoc(Op), Op.getValueType(),
                       GetNegatedExpression(Op.getOperand(1), DAG,
                                            LegalOperations, Depth+1),
                       Op.getOperand(0));
  case ISD::FSUB:
    // We can't turn -(A-B) into B-A when we honor signed zeros.
    assert(Options.UnsafeFPMath);

    // fold (fneg (fsub 0, B)) -> B
    if (ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(Op.getOperand(0)))
      if (N0CFP->isZero())
        return Op.getOperand(1);

    // fold (fneg (fsub A, B)) -> (fsub B, A)
    return DAG.getNode(ISD::FSUB, SDLoc(Op), Op.getValueType(),
                       Op.getOperand(1), Op.getOperand(0));

  case ISD::FMUL:
  case ISD::FDIV:
    assert(!Options.HonorSignDependentRoundingFPMath());

    // fold (fneg (fmul X, Y)) -> (fmul (fneg X), Y)
    if (isNegatibleForFree(Op.getOperand(0), LegalOperations,
                           DAG.getTargetLoweringInfo(), &Options, Depth+1))
      return DAG.getNode(Op.getOpcode(), SDLoc(Op), Op.getValueType(),
                         GetNegatedExpression(Op.getOperand(0), DAG,
                                              LegalOperations, Depth+1),
                         Op.getOperand(1));

    // fold (fneg (fmul X, Y)) -> (fmul X, (fneg Y))
    return DAG.getNode(Op.getOpcode(), SDLoc(Op), Op.getValueType(),
                       Op.getOperand(0),
                       GetNegatedExpression(Op.getOperand(1), DAG,
                                            LegalOperations, Depth+1));

  case ISD::FP_EXTEND:
  case ISD::FSIN:
    return DAG.getNode(Op.getOpcode(), SDLoc(Op), Op.getValueType(),
                       GetNegatedExpression(Op.getOperand(0), DAG,
                                            LegalOperations, Depth+1));
  case ISD::FP_ROUND:
      return DAG.getNode(ISD::FP_ROUND, SDLoc(Op), Op.getValueType(),
                         GetNegatedExpression(Op.getOperand(0), DAG,
                                              LegalOperations, Depth+1),
                         Op.getOperand(1));
  }
}

// Return true if this node is a setcc, or is a select_cc
// that selects between the target values used for true and false, making it
// equivalent to a setcc. Also, set the incoming LHS, RHS, and CC references to
// the appropriate nodes based on the type of node we are checking. This
// simplifies life a bit for the callers.
bool DAGCombiner::isSetCCEquivalent(SDValue N, SDValue &LHS, SDValue &RHS,
                                    SDValue &CC) const {
  if (N.getOpcode() == ISD::SETCC) {
    LHS = N.getOperand(0);
    RHS = N.getOperand(1);
    CC  = N.getOperand(2);
    return true;
  }

  if (N.getOpcode() != ISD::SELECT_CC ||
      !TLI.isConstTrueVal(N.getOperand(2).getNode()) ||
      !TLI.isConstFalseVal(N.getOperand(3).getNode()))
    return false;

  if (TLI.getBooleanContents(N.getValueType()) ==
      TargetLowering::UndefinedBooleanContent)
    return false;

  LHS = N.getOperand(0);
  RHS = N.getOperand(1);
  CC  = N.getOperand(4);
  return true;
}

/// Return true if this is a SetCC-equivalent operation with only one use.
/// If this is true, it allows the users to invert the operation for free when
/// it is profitable to do so.
bool DAGCombiner::isOneUseSetCC(SDValue N) const {
  SDValue N0, N1, N2;
  if (isSetCCEquivalent(N, N0, N1, N2) && N.getNode()->hasOneUse())
    return true;
  return false;
}

/// Returns true if N is a BUILD_VECTOR node whose
/// elements are all the same constant or undefined.
static bool isConstantSplatVector(SDNode *N, APInt& SplatValue) {
  BuildVectorSDNode *C = dyn_cast<BuildVectorSDNode>(N);
  if (!C)
    return false;

  APInt SplatUndef;
  unsigned SplatBitSize;
  bool HasAnyUndefs;
  EVT EltVT = N->getValueType(0).getVectorElementType();
  return (C->isConstantSplat(SplatValue, SplatUndef, SplatBitSize,
                             HasAnyUndefs) &&
          EltVT.getSizeInBits() >= SplatBitSize);
}

// \brief Returns the SDNode if it is a constant integer BuildVector
// or constant integer.
static SDNode *isConstantIntBuildVectorOrConstantInt(SDValue N) {
  if (isa<ConstantSDNode>(N))
    return N.getNode();
  if (ISD::isBuildVectorOfConstantSDNodes(N.getNode()))
    return N.getNode();
  return nullptr;
}

// \brief Returns the SDNode if it is a constant float BuildVector
// or constant float.
static SDNode *isConstantFPBuildVectorOrConstantFP(SDValue N) {
  if (isa<ConstantFPSDNode>(N))
    return N.getNode();
  if (ISD::isBuildVectorOfConstantFPSDNodes(N.getNode()))
    return N.getNode();
  return nullptr;
}

// \brief Returns the SDNode if it is a constant splat BuildVector or constant
// int.
static ConstantSDNode *isConstOrConstSplat(SDValue N) {
  if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N))
    return CN;

  if (BuildVectorSDNode *BV = dyn_cast<BuildVectorSDNode>(N)) {
    BitVector UndefElements;
    ConstantSDNode *CN = BV->getConstantSplatNode(&UndefElements);

    // BuildVectors can truncate their operands. Ignore that case here.
    // FIXME: We blindly ignore splats which include undef which is overly
    // pessimistic.
    if (CN && UndefElements.none() &&
        CN->getValueType(0) == N.getValueType().getScalarType())
      return CN;
  }

  return nullptr;
}

// \brief Returns the SDNode if it is a constant splat BuildVector or constant
// float.
static ConstantFPSDNode *isConstOrConstSplatFP(SDValue N) {
  if (ConstantFPSDNode *CN = dyn_cast<ConstantFPSDNode>(N))
    return CN;

  if (BuildVectorSDNode *BV = dyn_cast<BuildVectorSDNode>(N)) {
    BitVector UndefElements;
    ConstantFPSDNode *CN = BV->getConstantFPSplatNode(&UndefElements);

    if (CN && UndefElements.none())
      return CN;
  }

  return nullptr;
}

SDValue DAGCombiner::ReassociateOps(unsigned Opc, SDLoc DL,
                                    SDValue N0, SDValue N1) {
  EVT VT = N0.getValueType();
  if (N0.getOpcode() == Opc) {
    if (SDNode *L = isConstantIntBuildVectorOrConstantInt(N0.getOperand(1))) {
      if (SDNode *R = isConstantIntBuildVectorOrConstantInt(N1)) {
        // reassoc. (op (op x, c1), c2) -> (op x, (op c1, c2))
        if (SDValue OpNode = DAG.FoldConstantArithmetic(Opc, DL, VT, L, R))
          return DAG.getNode(Opc, DL, VT, N0.getOperand(0), OpNode);
        return SDValue();
      }
      if (N0.hasOneUse()) {
        // reassoc. (op (op x, c1), y) -> (op (op x, y), c1) iff x+c1 has one
        // use
        SDValue OpNode = DAG.getNode(Opc, SDLoc(N0), VT, N0.getOperand(0), N1);
        if (!OpNode.getNode())
          return SDValue();
        AddToWorklist(OpNode.getNode());
        return DAG.getNode(Opc, DL, VT, OpNode, N0.getOperand(1));
      }
    }
  }

  if (N1.getOpcode() == Opc) {
    if (SDNode *R = isConstantIntBuildVectorOrConstantInt(N1.getOperand(1))) {
      if (SDNode *L = isConstantIntBuildVectorOrConstantInt(N0)) {
        // reassoc. (op c2, (op x, c1)) -> (op x, (op c1, c2))
        if (SDValue OpNode = DAG.FoldConstantArithmetic(Opc, DL, VT, R, L))
          return DAG.getNode(Opc, DL, VT, N1.getOperand(0), OpNode);
        return SDValue();
      }
      if (N1.hasOneUse()) {
        // reassoc. (op y, (op x, c1)) -> (op (op x, y), c1) iff x+c1 has one
        // use
        SDValue OpNode = DAG.getNode(Opc, SDLoc(N0), VT, N1.getOperand(0), N0);
        if (!OpNode.getNode())
          return SDValue();
        AddToWorklist(OpNode.getNode());
        return DAG.getNode(Opc, DL, VT, OpNode, N1.getOperand(1));
      }
    }
  }

  return SDValue();
}

SDValue DAGCombiner::CombineTo(SDNode *N, const SDValue *To, unsigned NumTo,
                               bool AddTo) {
  assert(N->getNumValues() == NumTo && "Broken CombineTo call!");
  ++NodesCombined;
  DEBUG(dbgs() << "\nReplacing.1 ";
        N->dump(&DAG);
        dbgs() << "\nWith: ";
        To[0].getNode()->dump(&DAG);
        dbgs() << " and " << NumTo-1 << " other values\n");
  for (unsigned i = 0, e = NumTo; i != e; ++i)
    assert((!To[i].getNode() ||
            N->getValueType(i) == To[i].getValueType()) &&
           "Cannot combine value to value of different type!");

  WorklistRemover DeadNodes(*this);
  DAG.ReplaceAllUsesWith(N, To);
  if (AddTo) {
    // Push the new nodes and any users onto the worklist
    for (unsigned i = 0, e = NumTo; i != e; ++i) {
      if (To[i].getNode()) {
        AddToWorklist(To[i].getNode());
        AddUsersToWorklist(To[i].getNode());
      }
    }
  }

  // Finally, if the node is now dead, remove it from the graph.  The node
  // may not be dead if the replacement process recursively simplified to
  // something else needing this node.
  if (N->use_empty())
    deleteAndRecombine(N);
  return SDValue(N, 0);
}

void DAGCombiner::
CommitTargetLoweringOpt(const TargetLowering::TargetLoweringOpt &TLO) {
  // Replace all uses.  If any nodes become isomorphic to other nodes and
  // are deleted, make sure to remove them from our worklist.
  WorklistRemover DeadNodes(*this);
  DAG.ReplaceAllUsesOfValueWith(TLO.Old, TLO.New);

  // Push the new node and any (possibly new) users onto the worklist.
  AddToWorklist(TLO.New.getNode());
  AddUsersToWorklist(TLO.New.getNode());

  // Finally, if the node is now dead, remove it from the graph.  The node
  // may not be dead if the replacement process recursively simplified to
  // something else needing this node.
  if (TLO.Old.getNode()->use_empty())
    deleteAndRecombine(TLO.Old.getNode());
}

/// Check the specified integer node value to see if it can be simplified or if
/// things it uses can be simplified by bit propagation. If so, return true.
bool DAGCombiner::SimplifyDemandedBits(SDValue Op, const APInt &Demanded) {
  TargetLowering::TargetLoweringOpt TLO(DAG, LegalTypes, LegalOperations);
  APInt KnownZero, KnownOne;
  if (!TLI.SimplifyDemandedBits(Op, Demanded, KnownZero, KnownOne, TLO))
    return false;

  // Revisit the node.
  AddToWorklist(Op.getNode());

  // Replace the old value with the new one.
  ++NodesCombined;
  DEBUG(dbgs() << "\nReplacing.2 ";
        TLO.Old.getNode()->dump(&DAG);
        dbgs() << "\nWith: ";
        TLO.New.getNode()->dump(&DAG);
        dbgs() << '\n');

  CommitTargetLoweringOpt(TLO);
  return true;
}

void DAGCombiner::ReplaceLoadWithPromotedLoad(SDNode *Load, SDNode *ExtLoad) {
  SDLoc dl(Load);
  EVT VT = Load->getValueType(0);
  SDValue Trunc = DAG.getNode(ISD::TRUNCATE, dl, VT, SDValue(ExtLoad, 0));

  DEBUG(dbgs() << "\nReplacing.9 ";
        Load->dump(&DAG);
        dbgs() << "\nWith: ";
        Trunc.getNode()->dump(&DAG);
        dbgs() << '\n');
  WorklistRemover DeadNodes(*this);
  DAG.ReplaceAllUsesOfValueWith(SDValue(Load, 0), Trunc);
  DAG.ReplaceAllUsesOfValueWith(SDValue(Load, 1), SDValue(ExtLoad, 1));
  deleteAndRecombine(Load);
  AddToWorklist(Trunc.getNode());
}

SDValue DAGCombiner::PromoteOperand(SDValue Op, EVT PVT, bool &Replace) {
  Replace = false;
  SDLoc dl(Op);
  if (LoadSDNode *LD = dyn_cast<LoadSDNode>(Op)) {
    EVT MemVT = LD->getMemoryVT();
    ISD::LoadExtType ExtType = ISD::isNON_EXTLoad(LD)
      ? (TLI.isLoadExtLegal(ISD::ZEXTLOAD, PVT, MemVT) ? ISD::ZEXTLOAD
                                                       : ISD::EXTLOAD)
      : LD->getExtensionType();
    Replace = true;
    return DAG.getExtLoad(ExtType, dl, PVT,
                          LD->getChain(), LD->getBasePtr(),
                          MemVT, LD->getMemOperand());
  }

  unsigned Opc = Op.getOpcode();
  switch (Opc) {
  default: break;
  case ISD::AssertSext:
    return DAG.getNode(ISD::AssertSext, dl, PVT,
                       SExtPromoteOperand(Op.getOperand(0), PVT),
                       Op.getOperand(1));
  case ISD::AssertZext:
    return DAG.getNode(ISD::AssertZext, dl, PVT,
                       ZExtPromoteOperand(Op.getOperand(0), PVT),
                       Op.getOperand(1));
  case ISD::Constant: {
    unsigned ExtOpc =
      Op.getValueType().isByteSized() ? ISD::SIGN_EXTEND : ISD::ZERO_EXTEND;
    return DAG.getNode(ExtOpc, dl, PVT, Op);
  }
  }

  if (!TLI.isOperationLegal(ISD::ANY_EXTEND, PVT))
    return SDValue();
  return DAG.getNode(ISD::ANY_EXTEND, dl, PVT, Op);
}

SDValue DAGCombiner::SExtPromoteOperand(SDValue Op, EVT PVT) {
  if (!TLI.isOperationLegal(ISD::SIGN_EXTEND_INREG, PVT))
    return SDValue();
  EVT OldVT = Op.getValueType();
  SDLoc dl(Op);
  bool Replace = false;
  SDValue NewOp = PromoteOperand(Op, PVT, Replace);
  if (!NewOp.getNode())
    return SDValue();
  AddToWorklist(NewOp.getNode());

  if (Replace)
    ReplaceLoadWithPromotedLoad(Op.getNode(), NewOp.getNode());
  return DAG.getNode(ISD::SIGN_EXTEND_INREG, dl, NewOp.getValueType(), NewOp,
                     DAG.getValueType(OldVT));
}

SDValue DAGCombiner::ZExtPromoteOperand(SDValue Op, EVT PVT) {
  EVT OldVT = Op.getValueType();
  SDLoc dl(Op);
  bool Replace = false;
  SDValue NewOp = PromoteOperand(Op, PVT, Replace);
  if (!NewOp.getNode())
    return SDValue();
  AddToWorklist(NewOp.getNode());

  if (Replace)
    ReplaceLoadWithPromotedLoad(Op.getNode(), NewOp.getNode());
  return DAG.getZeroExtendInReg(NewOp, dl, OldVT);
}

/// Promote the specified integer binary operation if the target indicates it is
/// beneficial. e.g. On x86, it's usually better to promote i16 operations to
/// i32 since i16 instructions are longer.
SDValue DAGCombiner::PromoteIntBinOp(SDValue Op) {
  if (!LegalOperations)
    return SDValue();

  EVT VT = Op.getValueType();
  if (VT.isVector() || !VT.isInteger())
    return SDValue();

  // If operation type is 'undesirable', e.g. i16 on x86, consider
  // promoting it.
  unsigned Opc = Op.getOpcode();
  if (TLI.isTypeDesirableForOp(Opc, VT))
    return SDValue();

  EVT PVT = VT;
  // Consult target whether it is a good idea to promote this operation and
  // what's the right type to promote it to.
  if (TLI.IsDesirableToPromoteOp(Op, PVT)) {
    assert(PVT != VT && "Don't know what type to promote to!");

    bool Replace0 = false;
    SDValue N0 = Op.getOperand(0);
    SDValue NN0 = PromoteOperand(N0, PVT, Replace0);
    if (!NN0.getNode())
      return SDValue();

    bool Replace1 = false;
    SDValue N1 = Op.getOperand(1);
    SDValue NN1;
    if (N0 == N1)
      NN1 = NN0;
    else {
      NN1 = PromoteOperand(N1, PVT, Replace1);
      if (!NN1.getNode())
        return SDValue();
    }

    AddToWorklist(NN0.getNode());
    if (NN1.getNode())
      AddToWorklist(NN1.getNode());

    if (Replace0)
      ReplaceLoadWithPromotedLoad(N0.getNode(), NN0.getNode());
    if (Replace1)
      ReplaceLoadWithPromotedLoad(N1.getNode(), NN1.getNode());

    DEBUG(dbgs() << "\nPromoting ";
          Op.getNode()->dump(&DAG));
    SDLoc dl(Op);
    return DAG.getNode(ISD::TRUNCATE, dl, VT,
                       DAG.getNode(Opc, dl, PVT, NN0, NN1));
  }
  return SDValue();
}

/// Promote the specified integer shift operation if the target indicates it is
/// beneficial. e.g. On x86, it's usually better to promote i16 operations to
/// i32 since i16 instructions are longer.
SDValue DAGCombiner::PromoteIntShiftOp(SDValue Op) {
  if (!LegalOperations)
    return SDValue();

  EVT VT = Op.getValueType();
  if (VT.isVector() || !VT.isInteger())
    return SDValue();

  // If operation type is 'undesirable', e.g. i16 on x86, consider
  // promoting it.
  unsigned Opc = Op.getOpcode();
  if (TLI.isTypeDesirableForOp(Opc, VT))
    return SDValue();

  EVT PVT = VT;
  // Consult target whether it is a good idea to promote this operation and
  // what's the right type to promote it to.
  if (TLI.IsDesirableToPromoteOp(Op, PVT)) {
    assert(PVT != VT && "Don't know what type to promote to!");

    bool Replace = false;
    SDValue N0 = Op.getOperand(0);
    if (Opc == ISD::SRA)
      N0 = SExtPromoteOperand(Op.getOperand(0), PVT);
    else if (Opc == ISD::SRL)
      N0 = ZExtPromoteOperand(Op.getOperand(0), PVT);
    else
      N0 = PromoteOperand(N0, PVT, Replace);
    if (!N0.getNode())
      return SDValue();

    AddToWorklist(N0.getNode());
    if (Replace)
      ReplaceLoadWithPromotedLoad(Op.getOperand(0).getNode(), N0.getNode());

    DEBUG(dbgs() << "\nPromoting ";
          Op.getNode()->dump(&DAG));
    SDLoc dl(Op);
    return DAG.getNode(ISD::TRUNCATE, dl, VT,
                       DAG.getNode(Opc, dl, PVT, N0, Op.getOperand(1)));
  }
  return SDValue();
}

SDValue DAGCombiner::PromoteExtend(SDValue Op) {
  if (!LegalOperations)
    return SDValue();

  EVT VT = Op.getValueType();
  if (VT.isVector() || !VT.isInteger())
    return SDValue();

  // If operation type is 'undesirable', e.g. i16 on x86, consider
  // promoting it.
  unsigned Opc = Op.getOpcode();
  if (TLI.isTypeDesirableForOp(Opc, VT))
    return SDValue();

  EVT PVT = VT;
  // Consult target whether it is a good idea to promote this operation and
  // what's the right type to promote it to.
  if (TLI.IsDesirableToPromoteOp(Op, PVT)) {
    assert(PVT != VT && "Don't know what type to promote to!");
    // fold (aext (aext x)) -> (aext x)
    // fold (aext (zext x)) -> (zext x)
    // fold (aext (sext x)) -> (sext x)
    DEBUG(dbgs() << "\nPromoting ";
          Op.getNode()->dump(&DAG));
    return DAG.getNode(Op.getOpcode(), SDLoc(Op), VT, Op.getOperand(0));
  }
  return SDValue();
}

bool DAGCombiner::PromoteLoad(SDValue Op) {
  if (!LegalOperations)
    return false;

  EVT VT = Op.getValueType();
  if (VT.isVector() || !VT.isInteger())
    return false;

  // If operation type is 'undesirable', e.g. i16 on x86, consider
  // promoting it.
  unsigned Opc = Op.getOpcode();
  if (TLI.isTypeDesirableForOp(Opc, VT))
    return false;

  EVT PVT = VT;
  // Consult target whether it is a good idea to promote this operation and
  // what's the right type to promote it to.
  if (TLI.IsDesirableToPromoteOp(Op, PVT)) {
    assert(PVT != VT && "Don't know what type to promote to!");

    SDLoc dl(Op);
    SDNode *N = Op.getNode();
    LoadSDNode *LD = cast<LoadSDNode>(N);
    EVT MemVT = LD->getMemoryVT();
    ISD::LoadExtType ExtType = ISD::isNON_EXTLoad(LD)
      ? (TLI.isLoadExtLegal(ISD::ZEXTLOAD, PVT, MemVT) ? ISD::ZEXTLOAD
                                                       : ISD::EXTLOAD)
      : LD->getExtensionType();
    SDValue NewLD = DAG.getExtLoad(ExtType, dl, PVT,
                                   LD->getChain(), LD->getBasePtr(),
                                   MemVT, LD->getMemOperand());
    SDValue Result = DAG.getNode(ISD::TRUNCATE, dl, VT, NewLD);

    DEBUG(dbgs() << "\nPromoting ";
          N->dump(&DAG);
          dbgs() << "\nTo: ";
          Result.getNode()->dump(&DAG);
          dbgs() << '\n');
    WorklistRemover DeadNodes(*this);
    DAG.ReplaceAllUsesOfValueWith(SDValue(N, 0), Result);
    DAG.ReplaceAllUsesOfValueWith(SDValue(N, 1), NewLD.getValue(1));
    deleteAndRecombine(N);
    AddToWorklist(Result.getNode());
    return true;
  }
  return false;
}

/// \brief Recursively delete a node which has no uses and any operands for
/// which it is the only use.
///
/// Note that this both deletes the nodes and removes them from the worklist.
/// It also adds any nodes who have had a user deleted to the worklist as they
/// may now have only one use and subject to other combines.
bool DAGCombiner::recursivelyDeleteUnusedNodes(SDNode *N) {
  if (!N->use_empty())
    return false;

  SmallSetVector<SDNode *, 16> Nodes;
  Nodes.insert(N);
  do {
    N = Nodes.pop_back_val();
    if (!N)
      continue;

    if (N->use_empty()) {
      for (const SDValue &ChildN : N->op_values())
        Nodes.insert(ChildN.getNode());

      removeFromWorklist(N);
      DAG.DeleteNode(N);
    } else {
      AddToWorklist(N);
    }
  } while (!Nodes.empty());
  return true;
}

//===----------------------------------------------------------------------===//
//  Main DAG Combiner implementation
//===----------------------------------------------------------------------===//

void DAGCombiner::Run(CombineLevel AtLevel) {
  // set the instance variables, so that the various visit routines may use it.
  Level = AtLevel;
  LegalOperations = Level >= AfterLegalizeVectorOps;
  LegalTypes = Level >= AfterLegalizeTypes;

  // Add all the dag nodes to the worklist.
  for (SelectionDAG::allnodes_iterator I = DAG.allnodes_begin(),
       E = DAG.allnodes_end(); I != E; ++I)
    AddToWorklist(I);

  // Create a dummy node (which is not added to allnodes), that adds a reference
  // to the root node, preventing it from being deleted, and tracking any
  // changes of the root.
  HandleSDNode Dummy(DAG.getRoot());

  // while the worklist isn't empty, find a node and
  // try and combine it.
  while (!WorklistMap.empty()) {
    SDNode *N;
    // The Worklist holds the SDNodes in order, but it may contain null entries.
    do {
      N = Worklist.pop_back_val();
    } while (!N);

    bool GoodWorklistEntry = WorklistMap.erase(N);
    (void)GoodWorklistEntry;
    assert(GoodWorklistEntry &&
           "Found a worklist entry without a corresponding map entry!");

    // If N has no uses, it is dead.  Make sure to revisit all N's operands once
    // N is deleted from the DAG, since they too may now be dead or may have a
    // reduced number of uses, allowing other xforms.
    if (recursivelyDeleteUnusedNodes(N))
      continue;

    WorklistRemover DeadNodes(*this);

    // If this combine is running after legalizing the DAG, re-legalize any
    // nodes pulled off the worklist.
    if (Level == AfterLegalizeDAG) {
      SmallSetVector<SDNode *, 16> UpdatedNodes;
      bool NIsValid = DAG.LegalizeOp(N, UpdatedNodes);

      for (SDNode *LN : UpdatedNodes) {
        AddToWorklist(LN);
        AddUsersToWorklist(LN);
      }
      if (!NIsValid)
        continue;
    }

    DEBUG(dbgs() << "\nCombining: "; N->dump(&DAG));

    // Add any operands of the new node which have not yet been combined to the
    // worklist as well. Because the worklist uniques things already, this
    // won't repeatedly process the same operand.
    CombinedNodes.insert(N);
    for (const SDValue &ChildN : N->op_values())
      if (!CombinedNodes.count(ChildN.getNode()))
        AddToWorklist(ChildN.getNode());

    SDValue RV = combine(N);

    if (!RV.getNode())
      continue;

    ++NodesCombined;

    // If we get back the same node we passed in, rather than a new node or
    // zero, we know that the node must have defined multiple values and
    // CombineTo was used.  Since CombineTo takes care of the worklist
    // mechanics for us, we have no work to do in this case.
    if (RV.getNode() == N)
      continue;

    assert(N->getOpcode() != ISD::DELETED_NODE &&
           RV.getNode()->getOpcode() != ISD::DELETED_NODE &&
           "Node was deleted but visit returned new node!");

    DEBUG(dbgs() << " ... into: ";
          RV.getNode()->dump(&DAG));

    // Transfer debug value.
    DAG.TransferDbgValues(SDValue(N, 0), RV);
    if (N->getNumValues() == RV.getNode()->getNumValues())
      DAG.ReplaceAllUsesWith(N, RV.getNode());
    else {
      assert(N->getValueType(0) == RV.getValueType() &&
             N->getNumValues() == 1 && "Type mismatch");
      SDValue OpV = RV;
      DAG.ReplaceAllUsesWith(N, &OpV);
    }

    // Push the new node and any users onto the worklist
    AddToWorklist(RV.getNode());
    AddUsersToWorklist(RV.getNode());

    // Finally, if the node is now dead, remove it from the graph.  The node
    // may not be dead if the replacement process recursively simplified to
    // something else needing this node. This will also take care of adding any
    // operands which have lost a user to the worklist.
    recursivelyDeleteUnusedNodes(N);
  }

  // If the root changed (e.g. it was a dead load, update the root).
  DAG.setRoot(Dummy.getValue());
  DAG.RemoveDeadNodes();
}

SDValue DAGCombiner::visit(SDNode *N) {
  switch (N->getOpcode()) {
  default: break;
  case ISD::TokenFactor:        return visitTokenFactor(N);
  case ISD::MERGE_VALUES:       return visitMERGE_VALUES(N);
  case ISD::ADD:                return visitADD(N);
  case ISD::SUB:                return visitSUB(N);
  case ISD::ADDC:               return visitADDC(N);
  case ISD::SUBC:               return visitSUBC(N);
  case ISD::ADDE:               return visitADDE(N);
  case ISD::SUBE:               return visitSUBE(N);
  case ISD::MUL:                return visitMUL(N);
  case ISD::SDIV:               return visitSDIV(N);
  case ISD::UDIV:               return visitUDIV(N);
  case ISD::SREM:               return visitSREM(N);
  case ISD::UREM:               return visitUREM(N);
  case ISD::MULHU:              return visitMULHU(N);
  case ISD::MULHS:              return visitMULHS(N);
  case ISD::SMUL_LOHI:          return visitSMUL_LOHI(N);
  case ISD::UMUL_LOHI:          return visitUMUL_LOHI(N);
  case ISD::SMULO:              return visitSMULO(N);
  case ISD::UMULO:              return visitUMULO(N);
  case ISD::SDIVREM:            return visitSDIVREM(N);
  case ISD::UDIVREM:            return visitUDIVREM(N);
  case ISD::AND:                return visitAND(N);
  case ISD::OR:                 return visitOR(N);
  case ISD::XOR:                return visitXOR(N);
  case ISD::SHL:                return visitSHL(N);
  case ISD::SRA:                return visitSRA(N);
  case ISD::SRL:                return visitSRL(N);
  case ISD::ROTR:
  case ISD::ROTL:               return visitRotate(N);
  case ISD::BSWAP:              return visitBSWAP(N);
  case ISD::CTLZ:               return visitCTLZ(N);
  case ISD::CTLZ_ZERO_UNDEF:    return visitCTLZ_ZERO_UNDEF(N);
  case ISD::CTTZ:               return visitCTTZ(N);
  case ISD::CTTZ_ZERO_UNDEF:    return visitCTTZ_ZERO_UNDEF(N);
  case ISD::CTPOP:              return visitCTPOP(N);
  case ISD::SELECT:             return visitSELECT(N);
  case ISD::VSELECT:            return visitVSELECT(N);
  case ISD::SELECT_CC:          return visitSELECT_CC(N);
  case ISD::SETCC:              return visitSETCC(N);
  case ISD::SIGN_EXTEND:        return visitSIGN_EXTEND(N);
  case ISD::ZERO_EXTEND:        return visitZERO_EXTEND(N);
  case ISD::ANY_EXTEND:         return visitANY_EXTEND(N);
  case ISD::SIGN_EXTEND_INREG:  return visitSIGN_EXTEND_INREG(N);
  case ISD::SIGN_EXTEND_VECTOR_INREG: return visitSIGN_EXTEND_VECTOR_INREG(N);
  case ISD::TRUNCATE:           return visitTRUNCATE(N);
  case ISD::BITCAST:            return visitBITCAST(N);
  case ISD::BUILD_PAIR:         return visitBUILD_PAIR(N);
  case ISD::FADD:               return visitFADD(N);
  case ISD::FSUB:               return visitFSUB(N);
  case ISD::FMUL:               return visitFMUL(N);
  case ISD::FMA:                return visitFMA(N);
  case ISD::FDIV:               return visitFDIV(N);
  case ISD::FREM:               return visitFREM(N);
  case ISD::FSQRT:              return visitFSQRT(N);
  case ISD::FCOPYSIGN:          return visitFCOPYSIGN(N);
  case ISD::SINT_TO_FP:         return visitSINT_TO_FP(N);
  case ISD::UINT_TO_FP:         return visitUINT_TO_FP(N);
  case ISD::FP_TO_SINT:         return visitFP_TO_SINT(N);
  case ISD::FP_TO_UINT:         return visitFP_TO_UINT(N);
  case ISD::FP_ROUND:           return visitFP_ROUND(N);
  case ISD::FP_ROUND_INREG:     return visitFP_ROUND_INREG(N);
  case ISD::FP_EXTEND:          return visitFP_EXTEND(N);
  case ISD::FNEG:               return visitFNEG(N);
  case ISD::FABS:               return visitFABS(N);
  case ISD::FFLOOR:             return visitFFLOOR(N);
  case ISD::FMINNUM:            return visitFMINNUM(N);
  case ISD::FMAXNUM:            return visitFMAXNUM(N);
  case ISD::FCEIL:              return visitFCEIL(N);
  case ISD::FTRUNC:             return visitFTRUNC(N);
  case ISD::BRCOND:             return visitBRCOND(N);
  case ISD::BR_CC:              return visitBR_CC(N);
  case ISD::LOAD:               return visitLOAD(N);
  case ISD::STORE:              return visitSTORE(N);
  case ISD::INSERT_VECTOR_ELT:  return visitINSERT_VECTOR_ELT(N);
  case ISD::EXTRACT_VECTOR_ELT: return visitEXTRACT_VECTOR_ELT(N);
  case ISD::BUILD_VECTOR:       return visitBUILD_VECTOR(N);
  case ISD::CONCAT_VECTORS:     return visitCONCAT_VECTORS(N);
  case ISD::EXTRACT_SUBVECTOR:  return visitEXTRACT_SUBVECTOR(N);
  case ISD::VECTOR_SHUFFLE:     return visitVECTOR_SHUFFLE(N);
  case ISD::SCALAR_TO_VECTOR:   return visitSCALAR_TO_VECTOR(N);
  case ISD::INSERT_SUBVECTOR:   return visitINSERT_SUBVECTOR(N);
  case ISD::MGATHER:            return visitMGATHER(N);
  case ISD::MLOAD:              return visitMLOAD(N);
  case ISD::MSCATTER:           return visitMSCATTER(N);
  case ISD::MSTORE:             return visitMSTORE(N);
  case ISD::FP_TO_FP16:         return visitFP_TO_FP16(N);
  }
  return SDValue();
}

SDValue DAGCombiner::combine(SDNode *N) {
  SDValue RV = visit(N);

  // If nothing happened, try a target-specific DAG combine.
  if (!RV.getNode()) {
    assert(N->getOpcode() != ISD::DELETED_NODE &&
           "Node was deleted but visit returned NULL!");

    if (N->getOpcode() >= ISD::BUILTIN_OP_END ||
        TLI.hasTargetDAGCombine((ISD::NodeType)N->getOpcode())) {

      // Expose the DAG combiner to the target combiner impls.
      TargetLowering::DAGCombinerInfo
        DagCombineInfo(DAG, Level, false, this);

      RV = TLI.PerformDAGCombine(N, DagCombineInfo);
    }
  }

  // If nothing happened still, try promoting the operation.
  if (!RV.getNode()) {
    switch (N->getOpcode()) {
    default: break;
    case ISD::ADD:
    case ISD::SUB:
    case ISD::MUL:
    case ISD::AND:
    case ISD::OR:
    case ISD::XOR:
      RV = PromoteIntBinOp(SDValue(N, 0));
      break;
    case ISD::SHL:
    case ISD::SRA:
    case ISD::SRL:
      RV = PromoteIntShiftOp(SDValue(N, 0));
      break;
    case ISD::SIGN_EXTEND:
    case ISD::ZERO_EXTEND:
    case ISD::ANY_EXTEND:
      RV = PromoteExtend(SDValue(N, 0));
      break;
    case ISD::LOAD:
      if (PromoteLoad(SDValue(N, 0)))
        RV = SDValue(N, 0);
      break;
    }
  }

  // If N is a commutative binary node, try commuting it to enable more
  // sdisel CSE.
  if (!RV.getNode() && SelectionDAG::isCommutativeBinOp(N->getOpcode()) &&
      N->getNumValues() == 1) {
    SDValue N0 = N->getOperand(0);
    SDValue N1 = N->getOperand(1);

    // Constant operands are canonicalized to RHS.
    if (isa<ConstantSDNode>(N0) || !isa<ConstantSDNode>(N1)) {
      SDValue Ops[] = {N1, N0};
      SDNode *CSENode;
      if (const auto *BinNode = dyn_cast<BinaryWithFlagsSDNode>(N)) {
        CSENode = DAG.getNodeIfExists(N->getOpcode(), N->getVTList(), Ops,
                                      &BinNode->Flags);
      } else {
        CSENode = DAG.getNodeIfExists(N->getOpcode(), N->getVTList(), Ops);
      }
      if (CSENode)
        return SDValue(CSENode, 0);
    }
  }

  return RV;
}

/// Given a node, return its input chain if it has one, otherwise return a null
/// sd operand.
static SDValue getInputChainForNode(SDNode *N) {
  if (unsigned NumOps = N->getNumOperands()) {
    if (N->getOperand(0).getValueType() == MVT::Other)
      return N->getOperand(0);
    if (N->getOperand(NumOps-1).getValueType() == MVT::Other)
      return N->getOperand(NumOps-1);
    for (unsigned i = 1; i < NumOps-1; ++i)
      if (N->getOperand(i).getValueType() == MVT::Other)
        return N->getOperand(i);
  }
  return SDValue();
}

SDValue DAGCombiner::visitTokenFactor(SDNode *N) {
  // If N has two operands, where one has an input chain equal to the other,
  // the 'other' chain is redundant.
  if (N->getNumOperands() == 2) {
    if (getInputChainForNode(N->getOperand(0).getNode()) == N->getOperand(1))
      return N->getOperand(0);
    if (getInputChainForNode(N->getOperand(1).getNode()) == N->getOperand(0))
      return N->getOperand(1);
  }

  SmallVector<SDNode *, 8> TFs;     // List of token factors to visit.
  SmallVector<SDValue, 8> Ops;    // Ops for replacing token factor.
  SmallPtrSet<SDNode*, 16> SeenOps;
  bool Changed = false;             // If we should replace this token factor.

  // Start out with this token factor.
  TFs.push_back(N);

  // Iterate through token factors.  The TFs grows when new token factors are
  // encountered.
  for (unsigned i = 0; i < TFs.size(); ++i) {
    SDNode *TF = TFs[i];

    // Check each of the operands.
    for (const SDValue &Op : TF->op_values()) {

      switch (Op.getOpcode()) {
      case ISD::EntryToken:
        // Entry tokens don't need to be added to the list. They are
        // redundant.
        Changed = true;
        break;

      case ISD::TokenFactor:
        if (Op.hasOneUse() &&
            std::find(TFs.begin(), TFs.end(), Op.getNode()) == TFs.end()) {
          // Queue up for processing.
          TFs.push_back(Op.getNode());
          // Clean up in case the token factor is removed.
          AddToWorklist(Op.getNode());
          Changed = true;
          break;
        }
        // Fall thru

      default:
        // Only add if it isn't already in the list.
        if (SeenOps.insert(Op.getNode()).second)
          Ops.push_back(Op);
        else
          Changed = true;
        break;
      }
    }
  }

  SDValue Result;

  // If we've changed things around then replace token factor.
  if (Changed) {
    if (Ops.empty()) {
      // The entry token is the only possible outcome.
      Result = DAG.getEntryNode();
    } else {
      // New and improved token factor.
      Result = DAG.getNode(ISD::TokenFactor, SDLoc(N), MVT::Other, Ops);
    }

    // Add users to worklist if AA is enabled, since it may introduce
    // a lot of new chained token factors while removing memory deps.
    bool UseAA = CombinerAA.getNumOccurrences() > 0 ? CombinerAA
      : DAG.getSubtarget().useAA();
    return CombineTo(N, Result, UseAA /*add to worklist*/);
  }

  return Result;
}

/// MERGE_VALUES can always be eliminated.
SDValue DAGCombiner::visitMERGE_VALUES(SDNode *N) {
  WorklistRemover DeadNodes(*this);
  // Replacing results may cause a different MERGE_VALUES to suddenly
  // be CSE'd with N, and carry its uses with it. Iterate until no
  // uses remain, to ensure that the node can be safely deleted.
  // First add the users of this node to the work list so that they
  // can be tried again once they have new operands.
  AddUsersToWorklist(N);
  do {
    for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i)
      DAG.ReplaceAllUsesOfValueWith(SDValue(N, i), N->getOperand(i));
  } while (!N->use_empty());
  deleteAndRecombine(N);
  return SDValue(N, 0);   // Return N so it doesn't get rechecked!
}

static bool isNullConstant(SDValue V) {
  ConstantSDNode *Const = dyn_cast<ConstantSDNode>(V);
  return Const != nullptr && Const->isNullValue();
}

static bool isNullFPConstant(SDValue V) {
  ConstantFPSDNode *Const = dyn_cast<ConstantFPSDNode>(V);
  return Const != nullptr && Const->isZero() && !Const->isNegative();
}

static bool isAllOnesConstant(SDValue V) {
  ConstantSDNode *Const = dyn_cast<ConstantSDNode>(V);
  return Const != nullptr && Const->isAllOnesValue();
}

static bool isOneConstant(SDValue V) {
  ConstantSDNode *Const = dyn_cast<ConstantSDNode>(V);
  return Const != nullptr && Const->isOne();
}

/// If \p N is a ContantSDNode with isOpaque() == false return it casted to a
/// ContantSDNode pointer else nullptr.
static ConstantSDNode *getAsNonOpaqueConstant(SDValue N) {
  ConstantSDNode *Const = dyn_cast<ConstantSDNode>(N);
  return Const != nullptr && !Const->isOpaque() ? Const : nullptr;
}

SDValue DAGCombiner::visitADD(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N0.getValueType();

  // fold vector ops
  if (VT.isVector()) {
    if (SDValue FoldedVOp = SimplifyVBinOp(N))
      return FoldedVOp;

    // fold (add x, 0) -> x, vector edition
    if (ISD::isBuildVectorAllZeros(N1.getNode()))
      return N0;
    if (ISD::isBuildVectorAllZeros(N0.getNode()))
      return N1;
  }

  // fold (add x, undef) -> undef
  if (N0.getOpcode() == ISD::UNDEF)
    return N0;
  if (N1.getOpcode() == ISD::UNDEF)
    return N1;
  // fold (add c1, c2) -> c1+c2
  ConstantSDNode *N0C = getAsNonOpaqueConstant(N0);
  ConstantSDNode *N1C = getAsNonOpaqueConstant(N1);
  if (N0C && N1C)
    return DAG.FoldConstantArithmetic(ISD::ADD, SDLoc(N), VT, N0C, N1C);
  // canonicalize constant to RHS
  if (isConstantIntBuildVectorOrConstantInt(N0) &&
     !isConstantIntBuildVectorOrConstantInt(N1))
    return DAG.getNode(ISD::ADD, SDLoc(N), VT, N1, N0);
  // fold (add x, 0) -> x
  if (isNullConstant(N1))
    return N0;
  // fold (add Sym, c) -> Sym+c
  if (GlobalAddressSDNode *GA = dyn_cast<GlobalAddressSDNode>(N0))
    if (!LegalOperations && TLI.isOffsetFoldingLegal(GA) && N1C &&
        GA->getOpcode() == ISD::GlobalAddress)
      return DAG.getGlobalAddress(GA->getGlobal(), SDLoc(N1C), VT,
                                  GA->getOffset() +
                                    (uint64_t)N1C->getSExtValue());
  // fold ((c1-A)+c2) -> (c1+c2)-A
  if (N1C && N0.getOpcode() == ISD::SUB)
    if (ConstantSDNode *N0C = getAsNonOpaqueConstant(N0.getOperand(0))) {
      SDLoc DL(N);
      return DAG.getNode(ISD::SUB, DL, VT,
                         DAG.getConstant(N1C->getAPIntValue()+
                                         N0C->getAPIntValue(), DL, VT),
                         N0.getOperand(1));
    }
  // reassociate add
  if (SDValue RADD = ReassociateOps(ISD::ADD, SDLoc(N), N0, N1))
    return RADD;
  // fold ((0-A) + B) -> B-A
  if (N0.getOpcode() == ISD::SUB && isNullConstant(N0.getOperand(0)))
    return DAG.getNode(ISD::SUB, SDLoc(N), VT, N1, N0.getOperand(1));
  // fold (A + (0-B)) -> A-B
  if (N1.getOpcode() == ISD::SUB && isNullConstant(N1.getOperand(0)))
    return DAG.getNode(ISD::SUB, SDLoc(N), VT, N0, N1.getOperand(1));
  // fold (A+(B-A)) -> B
  if (N1.getOpcode() == ISD::SUB && N0 == N1.getOperand(1))
    return N1.getOperand(0);
  // fold ((B-A)+A) -> B
  if (N0.getOpcode() == ISD::SUB && N1 == N0.getOperand(1))
    return N0.getOperand(0);
  // fold (A+(B-(A+C))) to (B-C)
  if (N1.getOpcode() == ISD::SUB && N1.getOperand(1).getOpcode() == ISD::ADD &&
      N0 == N1.getOperand(1).getOperand(0))
    return DAG.getNode(ISD::SUB, SDLoc(N), VT, N1.getOperand(0),
                       N1.getOperand(1).getOperand(1));
  // fold (A+(B-(C+A))) to (B-C)
  if (N1.getOpcode() == ISD::SUB && N1.getOperand(1).getOpcode() == ISD::ADD &&
      N0 == N1.getOperand(1).getOperand(1))
    return DAG.getNode(ISD::SUB, SDLoc(N), VT, N1.getOperand(0),
                       N1.getOperand(1).getOperand(0));
  // fold (A+((B-A)+or-C)) to (B+or-C)
  if ((N1.getOpcode() == ISD::SUB || N1.getOpcode() == ISD::ADD) &&
      N1.getOperand(0).getOpcode() == ISD::SUB &&
      N0 == N1.getOperand(0).getOperand(1))
    return DAG.getNode(N1.getOpcode(), SDLoc(N), VT,
                       N1.getOperand(0).getOperand(0), N1.getOperand(1));

  // fold (A-B)+(C-D) to (A+C)-(B+D) when A or C is constant
  if (N0.getOpcode() == ISD::SUB && N1.getOpcode() == ISD::SUB) {
    SDValue N00 = N0.getOperand(0);
    SDValue N01 = N0.getOperand(1);
    SDValue N10 = N1.getOperand(0);
    SDValue N11 = N1.getOperand(1);

    if (isa<ConstantSDNode>(N00) || isa<ConstantSDNode>(N10))
      return DAG.getNode(ISD::SUB, SDLoc(N), VT,
                         DAG.getNode(ISD::ADD, SDLoc(N0), VT, N00, N10),
                         DAG.getNode(ISD::ADD, SDLoc(N1), VT, N01, N11));
  }

  if (!VT.isVector() && SimplifyDemandedBits(SDValue(N, 0)))
    return SDValue(N, 0);

  // fold (a+b) -> (a|b) iff a and b share no bits.
  if (VT.isInteger() && !VT.isVector()) {
    APInt LHSZero, LHSOne;
    APInt RHSZero, RHSOne;
    DAG.computeKnownBits(N0, LHSZero, LHSOne);

    if (LHSZero.getBoolValue()) {
      DAG.computeKnownBits(N1, RHSZero, RHSOne);

      // If all possibly-set bits on the LHS are clear on the RHS, return an OR.
      // If all possibly-set bits on the RHS are clear on the LHS, return an OR.
      if ((RHSZero & ~LHSZero) == ~LHSZero || (LHSZero & ~RHSZero) == ~RHSZero){
        if (!LegalOperations || TLI.isOperationLegal(ISD::OR, VT))
          return DAG.getNode(ISD::OR, SDLoc(N), VT, N0, N1);
      }
    }
  }

  // fold (add x, shl(0 - y, n)) -> sub(x, shl(y, n))
  if (N1.getOpcode() == ISD::SHL && N1.getOperand(0).getOpcode() == ISD::SUB &&
      isNullConstant(N1.getOperand(0).getOperand(0)))
    return DAG.getNode(ISD::SUB, SDLoc(N), VT, N0,
                       DAG.getNode(ISD::SHL, SDLoc(N), VT,
                                   N1.getOperand(0).getOperand(1),
                                   N1.getOperand(1)));
  if (N0.getOpcode() == ISD::SHL && N0.getOperand(0).getOpcode() == ISD::SUB &&
      isNullConstant(N0.getOperand(0).getOperand(0)))
    return DAG.getNode(ISD::SUB, SDLoc(N), VT, N1,
                       DAG.getNode(ISD::SHL, SDLoc(N), VT,
                                   N0.getOperand(0).getOperand(1),
                                   N0.getOperand(1)));

  if (N1.getOpcode() == ISD::AND) {
    SDValue AndOp0 = N1.getOperand(0);
    unsigned NumSignBits = DAG.ComputeNumSignBits(AndOp0);
    unsigned DestBits = VT.getScalarType().getSizeInBits();

    // (add z, (and (sbbl x, x), 1)) -> (sub z, (sbbl x, x))
    // and similar xforms where the inner op is either ~0 or 0.
    if (NumSignBits == DestBits && isOneConstant(N1->getOperand(1))) {
      SDLoc DL(N);
      return DAG.getNode(ISD::SUB, DL, VT, N->getOperand(0), AndOp0);
    }
  }

  // add (sext i1), X -> sub X, (zext i1)
  if (N0.getOpcode() == ISD::SIGN_EXTEND &&
      N0.getOperand(0).getValueType() == MVT::i1 &&
      !TLI.isOperationLegal(ISD::SIGN_EXTEND, MVT::i1)) {
    SDLoc DL(N);
    SDValue ZExt = DAG.getNode(ISD::ZERO_EXTEND, DL, VT, N0.getOperand(0));
    return DAG.getNode(ISD::SUB, DL, VT, N1, ZExt);
  }

  // add X, (sextinreg Y i1) -> sub X, (and Y 1)
  if (N1.getOpcode() == ISD::SIGN_EXTEND_INREG) {
    VTSDNode *TN = cast<VTSDNode>(N1.getOperand(1));
    if (TN->getVT() == MVT::i1) {
      SDLoc DL(N);
      SDValue ZExt = DAG.getNode(ISD::AND, DL, VT, N1.getOperand(0),
                                 DAG.getConstant(1, DL, VT));
      return DAG.getNode(ISD::SUB, DL, VT, N0, ZExt);
    }
  }

  return SDValue();
}

SDValue DAGCombiner::visitADDC(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N0.getValueType();

  // If the flag result is dead, turn this into an ADD.
  if (!N->hasAnyUseOfValue(1))
    return CombineTo(N, DAG.getNode(ISD::ADD, SDLoc(N), VT, N0, N1),
                     DAG.getNode(ISD::CARRY_FALSE,
                                 SDLoc(N), MVT::Glue));

  // canonicalize constant to RHS.
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  if (N0C && !N1C)
    return DAG.getNode(ISD::ADDC, SDLoc(N), N->getVTList(), N1, N0);

  // fold (addc x, 0) -> x + no carry out
  if (isNullConstant(N1))
    return CombineTo(N, N0, DAG.getNode(ISD::CARRY_FALSE,
                                        SDLoc(N), MVT::Glue));

  // fold (addc a, b) -> (or a, b), CARRY_FALSE iff a and b share no bits.
  APInt LHSZero, LHSOne;
  APInt RHSZero, RHSOne;
  DAG.computeKnownBits(N0, LHSZero, LHSOne);

  if (LHSZero.getBoolValue()) {
    DAG.computeKnownBits(N1, RHSZero, RHSOne);

    // If all possibly-set bits on the LHS are clear on the RHS, return an OR.
    // If all possibly-set bits on the RHS are clear on the LHS, return an OR.
    if ((RHSZero & ~LHSZero) == ~LHSZero || (LHSZero & ~RHSZero) == ~RHSZero)
      return CombineTo(N, DAG.getNode(ISD::OR, SDLoc(N), VT, N0, N1),
                       DAG.getNode(ISD::CARRY_FALSE,
                                   SDLoc(N), MVT::Glue));
  }

  return SDValue();
}

SDValue DAGCombiner::visitADDE(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  SDValue CarryIn = N->getOperand(2);

  // canonicalize constant to RHS
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  if (N0C && !N1C)
    return DAG.getNode(ISD::ADDE, SDLoc(N), N->getVTList(),
                       N1, N0, CarryIn);

  // fold (adde x, y, false) -> (addc x, y)
  if (CarryIn.getOpcode() == ISD::CARRY_FALSE)
    return DAG.getNode(ISD::ADDC, SDLoc(N), N->getVTList(), N0, N1);

  return SDValue();
}

// Since it may not be valid to emit a fold to zero for vector initializers
// check if we can before folding.
static SDValue tryFoldToZero(SDLoc DL, const TargetLowering &TLI, EVT VT,
                             SelectionDAG &DAG,
                             bool LegalOperations, bool LegalTypes) {
  if (!VT.isVector())
    return DAG.getConstant(0, DL, VT);
  if (!LegalOperations || TLI.isOperationLegal(ISD::BUILD_VECTOR, VT))
    return DAG.getConstant(0, DL, VT);
  return SDValue();
}

SDValue DAGCombiner::visitSUB(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N0.getValueType();

  // fold vector ops
  if (VT.isVector()) {
    if (SDValue FoldedVOp = SimplifyVBinOp(N))
      return FoldedVOp;

    // fold (sub x, 0) -> x, vector edition
    if (ISD::isBuildVectorAllZeros(N1.getNode()))
      return N0;
  }

  // fold (sub x, x) -> 0
  // FIXME: Refactor this and xor and other similar operations together.
  if (N0 == N1)
    return tryFoldToZero(SDLoc(N), TLI, VT, DAG, LegalOperations, LegalTypes);
  // fold (sub c1, c2) -> c1-c2
  ConstantSDNode *N0C = getAsNonOpaqueConstant(N0);
  ConstantSDNode *N1C = getAsNonOpaqueConstant(N1);
  if (N0C && N1C)
    return DAG.FoldConstantArithmetic(ISD::SUB, SDLoc(N), VT, N0C, N1C);
  // fold (sub x, c) -> (add x, -c)
  if (N1C) {
    SDLoc DL(N);
    return DAG.getNode(ISD::ADD, DL, VT, N0,
                       DAG.getConstant(-N1C->getAPIntValue(), DL, VT));
  }
  // Canonicalize (sub -1, x) -> ~x, i.e. (xor x, -1)
  if (isAllOnesConstant(N0))
    return DAG.getNode(ISD::XOR, SDLoc(N), VT, N1, N0);
  // fold A-(A-B) -> B
  if (N1.getOpcode() == ISD::SUB && N0 == N1.getOperand(0))
    return N1.getOperand(1);
  // fold (A+B)-A -> B
  if (N0.getOpcode() == ISD::ADD && N0.getOperand(0) == N1)
    return N0.getOperand(1);
  // fold (A+B)-B -> A
  if (N0.getOpcode() == ISD::ADD && N0.getOperand(1) == N1)
    return N0.getOperand(0);
  // fold C2-(A+C1) -> (C2-C1)-A
  ConstantSDNode *N1C1 = N1.getOpcode() != ISD::ADD ? nullptr :
    dyn_cast<ConstantSDNode>(N1.getOperand(1).getNode());
  if (N1.getOpcode() == ISD::ADD && N0C && N1C1) {
    SDLoc DL(N);
    SDValue NewC = DAG.getConstant(N0C->getAPIntValue() - N1C1->getAPIntValue(),
                                   DL, VT);
    return DAG.getNode(ISD::SUB, DL, VT, NewC,
                       N1.getOperand(0));
  }
  // fold ((A+(B+or-C))-B) -> A+or-C
  if (N0.getOpcode() == ISD::ADD &&
      (N0.getOperand(1).getOpcode() == ISD::SUB ||
       N0.getOperand(1).getOpcode() == ISD::ADD) &&
      N0.getOperand(1).getOperand(0) == N1)
    return DAG.getNode(N0.getOperand(1).getOpcode(), SDLoc(N), VT,
                       N0.getOperand(0), N0.getOperand(1).getOperand(1));
  // fold ((A+(C+B))-B) -> A+C
  if (N0.getOpcode() == ISD::ADD &&
      N0.getOperand(1).getOpcode() == ISD::ADD &&
      N0.getOperand(1).getOperand(1) == N1)
    return DAG.getNode(ISD::ADD, SDLoc(N), VT,
                       N0.getOperand(0), N0.getOperand(1).getOperand(0));
  // fold ((A-(B-C))-C) -> A-B
  if (N0.getOpcode() == ISD::SUB &&
      N0.getOperand(1).getOpcode() == ISD::SUB &&
      N0.getOperand(1).getOperand(1) == N1)
    return DAG.getNode(ISD::SUB, SDLoc(N), VT,
                       N0.getOperand(0), N0.getOperand(1).getOperand(0));

  // If either operand of a sub is undef, the result is undef
  if (N0.getOpcode() == ISD::UNDEF)
    return N0;
  if (N1.getOpcode() == ISD::UNDEF)
    return N1;

  // If the relocation model supports it, consider symbol offsets.
  if (GlobalAddressSDNode *GA = dyn_cast<GlobalAddressSDNode>(N0))
    if (!LegalOperations && TLI.isOffsetFoldingLegal(GA)) {
      // fold (sub Sym, c) -> Sym-c
      if (N1C && GA->getOpcode() == ISD::GlobalAddress)
        return DAG.getGlobalAddress(GA->getGlobal(), SDLoc(N1C), VT,
                                    GA->getOffset() -
                                      (uint64_t)N1C->getSExtValue());
      // fold (sub Sym+c1, Sym+c2) -> c1-c2
      if (GlobalAddressSDNode *GB = dyn_cast<GlobalAddressSDNode>(N1))
        if (GA->getGlobal() == GB->getGlobal())
          return DAG.getConstant((uint64_t)GA->getOffset() - GB->getOffset(),
                                 SDLoc(N), VT);
    }

  // sub X, (sextinreg Y i1) -> add X, (and Y 1)
  if (N1.getOpcode() == ISD::SIGN_EXTEND_INREG) {
    VTSDNode *TN = cast<VTSDNode>(N1.getOperand(1));
    if (TN->getVT() == MVT::i1) {
      SDLoc DL(N);
      SDValue ZExt = DAG.getNode(ISD::AND, DL, VT, N1.getOperand(0),
                                 DAG.getConstant(1, DL, VT));
      return DAG.getNode(ISD::ADD, DL, VT, N0, ZExt);
    }
  }

  return SDValue();
}

SDValue DAGCombiner::visitSUBC(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N0.getValueType();

  // If the flag result is dead, turn this into an SUB.
  if (!N->hasAnyUseOfValue(1))
    return CombineTo(N, DAG.getNode(ISD::SUB, SDLoc(N), VT, N0, N1),
                     DAG.getNode(ISD::CARRY_FALSE, SDLoc(N),
                                 MVT::Glue));

  // fold (subc x, x) -> 0 + no borrow
  if (N0 == N1) {
    SDLoc DL(N);
    return CombineTo(N, DAG.getConstant(0, DL, VT),
                     DAG.getNode(ISD::CARRY_FALSE, DL,
                                 MVT::Glue));
  }

  // fold (subc x, 0) -> x + no borrow
  if (isNullConstant(N1))
    return CombineTo(N, N0, DAG.getNode(ISD::CARRY_FALSE, SDLoc(N),
                                        MVT::Glue));

  // Canonicalize (sub -1, x) -> ~x, i.e. (xor x, -1) + no borrow
  if (isAllOnesConstant(N0))
    return CombineTo(N, DAG.getNode(ISD::XOR, SDLoc(N), VT, N1, N0),
                     DAG.getNode(ISD::CARRY_FALSE, SDLoc(N),
                                 MVT::Glue));

  return SDValue();
}

SDValue DAGCombiner::visitSUBE(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  SDValue CarryIn = N->getOperand(2);

  // fold (sube x, y, false) -> (subc x, y)
  if (CarryIn.getOpcode() == ISD::CARRY_FALSE)
    return DAG.getNode(ISD::SUBC, SDLoc(N), N->getVTList(), N0, N1);

  return SDValue();
}

SDValue DAGCombiner::visitMUL(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N0.getValueType();

  // fold (mul x, undef) -> 0
  if (N0.getOpcode() == ISD::UNDEF || N1.getOpcode() == ISD::UNDEF)
    return DAG.getConstant(0, SDLoc(N), VT);

  bool N0IsConst = false;
  bool N1IsConst = false;
  bool N1IsOpaqueConst = false;
  bool N0IsOpaqueConst = false;
  APInt ConstValue0, ConstValue1;
  // fold vector ops
  if (VT.isVector()) {
    if (SDValue FoldedVOp = SimplifyVBinOp(N))
      return FoldedVOp;

    N0IsConst = isConstantSplatVector(N0.getNode(), ConstValue0);
    N1IsConst = isConstantSplatVector(N1.getNode(), ConstValue1);
  } else {
    N0IsConst = isa<ConstantSDNode>(N0);
    if (N0IsConst) {
      ConstValue0 = cast<ConstantSDNode>(N0)->getAPIntValue();
      N0IsOpaqueConst = cast<ConstantSDNode>(N0)->isOpaque();
    }
    N1IsConst = isa<ConstantSDNode>(N1);
    if (N1IsConst) {
      ConstValue1 = cast<ConstantSDNode>(N1)->getAPIntValue();
      N1IsOpaqueConst = cast<ConstantSDNode>(N1)->isOpaque();
    }
  }

  // fold (mul c1, c2) -> c1*c2
  if (N0IsConst && N1IsConst && !N0IsOpaqueConst && !N1IsOpaqueConst)
    return DAG.FoldConstantArithmetic(ISD::MUL, SDLoc(N), VT,
                                      N0.getNode(), N1.getNode());

  // canonicalize constant to RHS (vector doesn't have to splat)
  if (isConstantIntBuildVectorOrConstantInt(N0) &&
     !isConstantIntBuildVectorOrConstantInt(N1))
    return DAG.getNode(ISD::MUL, SDLoc(N), VT, N1, N0);
  // fold (mul x, 0) -> 0
  if (N1IsConst && ConstValue1 == 0)
    return N1;
  // We require a splat of the entire scalar bit width for non-contiguous
  // bit patterns.
  bool IsFullSplat =
    ConstValue1.getBitWidth() == VT.getScalarType().getSizeInBits();
  // fold (mul x, 1) -> x
  if (N1IsConst && ConstValue1 == 1 && IsFullSplat)
    return N0;
  // fold (mul x, -1) -> 0-x
  if (N1IsConst && ConstValue1.isAllOnesValue()) {
    SDLoc DL(N);
    return DAG.getNode(ISD::SUB, DL, VT,
                       DAG.getConstant(0, DL, VT), N0);
  }
  // fold (mul x, (1 << c)) -> x << c
  if (N1IsConst && !N1IsOpaqueConst && ConstValue1.isPowerOf2() &&
      IsFullSplat) {
    SDLoc DL(N);
    return DAG.getNode(ISD::SHL, DL, VT, N0,
                       DAG.getConstant(ConstValue1.logBase2(), DL,
                                       getShiftAmountTy(N0.getValueType())));
  }
  // fold (mul x, -(1 << c)) -> -(x << c) or (-x) << c
  if (N1IsConst && !N1IsOpaqueConst && (-ConstValue1).isPowerOf2() &&
      IsFullSplat) {
    unsigned Log2Val = (-ConstValue1).logBase2();
    SDLoc DL(N);
    // FIXME: If the input is something that is easily negated (e.g. a
    // single-use add), we should put the negate there.
    return DAG.getNode(ISD::SUB, DL, VT,
                       DAG.getConstant(0, DL, VT),
                       DAG.getNode(ISD::SHL, DL, VT, N0,
                            DAG.getConstant(Log2Val, DL,
                                      getShiftAmountTy(N0.getValueType()))));
  }

  APInt Val;
  // (mul (shl X, c1), c2) -> (mul X, c2 << c1)
  if (N1IsConst && N0.getOpcode() == ISD::SHL &&
      (isConstantSplatVector(N0.getOperand(1).getNode(), Val) ||
                     isa<ConstantSDNode>(N0.getOperand(1)))) {
    SDValue C3 = DAG.getNode(ISD::SHL, SDLoc(N), VT,
                             N1, N0.getOperand(1));
    AddToWorklist(C3.getNode());
    return DAG.getNode(ISD::MUL, SDLoc(N), VT,
                       N0.getOperand(0), C3);
  }

  // Change (mul (shl X, C), Y) -> (shl (mul X, Y), C) when the shift has one
  // use.
  {
    SDValue Sh(nullptr,0), Y(nullptr,0);
    // Check for both (mul (shl X, C), Y)  and  (mul Y, (shl X, C)).
    if (N0.getOpcode() == ISD::SHL &&
        (isConstantSplatVector(N0.getOperand(1).getNode(), Val) ||
                       isa<ConstantSDNode>(N0.getOperand(1))) &&
        N0.getNode()->hasOneUse()) {
      Sh = N0; Y = N1;
    } else if (N1.getOpcode() == ISD::SHL &&
               isa<ConstantSDNode>(N1.getOperand(1)) &&
               N1.getNode()->hasOneUse()) {
      Sh = N1; Y = N0;
    }

    if (Sh.getNode()) {
      SDValue Mul = DAG.getNode(ISD::MUL, SDLoc(N), VT,
                                Sh.getOperand(0), Y);
      return DAG.getNode(ISD::SHL, SDLoc(N), VT,
                         Mul, Sh.getOperand(1));
    }
  }

  // fold (mul (add x, c1), c2) -> (add (mul x, c2), c1*c2)
  if (N1IsConst && N0.getOpcode() == ISD::ADD && N0.getNode()->hasOneUse() &&
      (isConstantSplatVector(N0.getOperand(1).getNode(), Val) ||
                     isa<ConstantSDNode>(N0.getOperand(1))))
    return DAG.getNode(ISD::ADD, SDLoc(N), VT,
                       DAG.getNode(ISD::MUL, SDLoc(N0), VT,
                                   N0.getOperand(0), N1),
                       DAG.getNode(ISD::MUL, SDLoc(N1), VT,
                                   N0.getOperand(1), N1));

  // reassociate mul
  if (SDValue RMUL = ReassociateOps(ISD::MUL, SDLoc(N), N0, N1))
    return RMUL;

  return SDValue();
}

SDValue DAGCombiner::visitSDIV(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N->getValueType(0);

  // fold vector ops
  if (VT.isVector())
    if (SDValue FoldedVOp = SimplifyVBinOp(N))
      return FoldedVOp;

  // fold (sdiv c1, c2) -> c1/c2
  ConstantSDNode *N0C = isConstOrConstSplat(N0);
  ConstantSDNode *N1C = isConstOrConstSplat(N1);
  if (N0C && N1C && !N0C->isOpaque() && !N1C->isOpaque())
    return DAG.FoldConstantArithmetic(ISD::SDIV, SDLoc(N), VT, N0C, N1C);
  // fold (sdiv X, 1) -> X
  if (N1C && N1C->isOne())
    return N0;
  // fold (sdiv X, -1) -> 0-X
  if (N1C && N1C->isAllOnesValue()) {
    SDLoc DL(N);
    return DAG.getNode(ISD::SUB, DL, VT,
                       DAG.getConstant(0, DL, VT), N0);
  }
  // If we know the sign bits of both operands are zero, strength reduce to a
  // udiv instead.  Handles (X&15) /s 4 -> X&15 >> 2
  if (!VT.isVector()) {
    if (DAG.SignBitIsZero(N1) && DAG.SignBitIsZero(N0))
      return DAG.getNode(ISD::UDIV, SDLoc(N), N1.getValueType(),
                         N0, N1);
  }

  // fold (sdiv X, pow2) -> simple ops after legalize
  // FIXME: We check for the exact bit here because the generic lowering gives
  // better results in that case. The target-specific lowering should learn how
  // to handle exact sdivs efficiently.
  if (N1C && !N1C->isNullValue() && !N1C->isOpaque() &&
      !cast<BinaryWithFlagsSDNode>(N)->Flags.hasExact() &&
      (N1C->getAPIntValue().isPowerOf2() ||
       (-N1C->getAPIntValue()).isPowerOf2())) {
    // If dividing by powers of two is cheap, then don't perform the following
    // fold.
    if (TLI.isPow2SDivCheap())
      return SDValue();

    // Target-specific implementation of sdiv x, pow2.
    SDValue Res = BuildSDIVPow2(N);
    if (Res.getNode())
      return Res;

    unsigned lg2 = N1C->getAPIntValue().countTrailingZeros();
    SDLoc DL(N);

    // Splat the sign bit into the register
    SDValue SGN =
        DAG.getNode(ISD::SRA, DL, VT, N0,
                    DAG.getConstant(VT.getScalarSizeInBits() - 1, DL,
                                    getShiftAmountTy(N0.getValueType())));
    AddToWorklist(SGN.getNode());

    // Add (N0 < 0) ? abs2 - 1 : 0;
    SDValue SRL =
        DAG.getNode(ISD::SRL, DL, VT, SGN,
                    DAG.getConstant(VT.getScalarSizeInBits() - lg2, DL,
                                    getShiftAmountTy(SGN.getValueType())));
    SDValue ADD = DAG.getNode(ISD::ADD, DL, VT, N0, SRL);
    AddToWorklist(SRL.getNode());
    AddToWorklist(ADD.getNode());    // Divide by pow2
    SDValue SRA = DAG.getNode(ISD::SRA, DL, VT, ADD,
                  DAG.getConstant(lg2, DL,
                                  getShiftAmountTy(ADD.getValueType())));

    // If we're dividing by a positive value, we're done.  Otherwise, we must
    // negate the result.
    if (N1C->getAPIntValue().isNonNegative())
      return SRA;

    AddToWorklist(SRA.getNode());
    return DAG.getNode(ISD::SUB, DL, VT, DAG.getConstant(0, DL, VT), SRA);
  }

  // If integer divide is expensive and we satisfy the requirements, emit an
  // alternate sequence.
  if (N1C && !TLI.isIntDivCheap()) {
    SDValue Op = BuildSDIV(N);
    if (Op.getNode()) return Op;
  }

  // undef / X -> 0
  if (N0.getOpcode() == ISD::UNDEF)
    return DAG.getConstant(0, SDLoc(N), VT);
  // X / undef -> undef
  if (N1.getOpcode() == ISD::UNDEF)
    return N1;

  return SDValue();
}

SDValue DAGCombiner::visitUDIV(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N->getValueType(0);

  // fold vector ops
  if (VT.isVector())
    if (SDValue FoldedVOp = SimplifyVBinOp(N))
      return FoldedVOp;

  // fold (udiv c1, c2) -> c1/c2
  ConstantSDNode *N0C = isConstOrConstSplat(N0);
  ConstantSDNode *N1C = isConstOrConstSplat(N1);
  if (N0C && N1C)
    if (SDValue Folded = DAG.FoldConstantArithmetic(ISD::UDIV, SDLoc(N), VT,
                                                    N0C, N1C))
      return Folded;
  // fold (udiv x, (1 << c)) -> x >>u c
  if (N1C && !N1C->isOpaque() && N1C->getAPIntValue().isPowerOf2()) {
    SDLoc DL(N);
    return DAG.getNode(ISD::SRL, DL, VT, N0,
                       DAG.getConstant(N1C->getAPIntValue().logBase2(), DL,
                                       getShiftAmountTy(N0.getValueType())));
  }
  // fold (udiv x, (shl c, y)) -> x >>u (log2(c)+y) iff c is power of 2
  if (N1.getOpcode() == ISD::SHL) {
    if (ConstantSDNode *SHC = getAsNonOpaqueConstant(N1.getOperand(0))) {
      if (SHC->getAPIntValue().isPowerOf2()) {
        EVT ADDVT = N1.getOperand(1).getValueType();
        SDLoc DL(N);
        SDValue Add = DAG.getNode(ISD::ADD, DL, ADDVT,
                                  N1.getOperand(1),
                                  DAG.getConstant(SHC->getAPIntValue()
                                                                  .logBase2(),
                                                  DL, ADDVT));
        AddToWorklist(Add.getNode());
        return DAG.getNode(ISD::SRL, DL, VT, N0, Add);
      }
    }
  }
  // fold (udiv x, c) -> alternate
  if (N1C && !TLI.isIntDivCheap()) {
    SDValue Op = BuildUDIV(N);
    if (Op.getNode()) return Op;
  }

  // undef / X -> 0
  if (N0.getOpcode() == ISD::UNDEF)
    return DAG.getConstant(0, SDLoc(N), VT);
  // X / undef -> undef
  if (N1.getOpcode() == ISD::UNDEF)
    return N1;

  return SDValue();
}

SDValue DAGCombiner::visitSREM(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N->getValueType(0);

  // fold (srem c1, c2) -> c1%c2
  ConstantSDNode *N0C = isConstOrConstSplat(N0);
  ConstantSDNode *N1C = isConstOrConstSplat(N1);
  if (N0C && N1C)
    if (SDValue Folded = DAG.FoldConstantArithmetic(ISD::SREM, SDLoc(N), VT,
                                                    N0C, N1C))
      return Folded;
  // If we know the sign bits of both operands are zero, strength reduce to a
  // urem instead.  Handles (X & 0x0FFFFFFF) %s 16 -> X&15
  if (!VT.isVector()) {
    if (DAG.SignBitIsZero(N1) && DAG.SignBitIsZero(N0))
      return DAG.getNode(ISD::UREM, SDLoc(N), VT, N0, N1);
  }

  // If X/C can be simplified by the division-by-constant logic, lower
  // X%C to the equivalent of X-X/C*C.
  if (N1C && !N1C->isNullValue()) {
    SDValue Div = DAG.getNode(ISD::SDIV, SDLoc(N), VT, N0, N1);
    AddToWorklist(Div.getNode());
    SDValue OptimizedDiv = combine(Div.getNode());
    if (OptimizedDiv.getNode() && OptimizedDiv.getNode() != Div.getNode()) {
      SDValue Mul = DAG.getNode(ISD::MUL, SDLoc(N), VT,
                                OptimizedDiv, N1);
      SDValue Sub = DAG.getNode(ISD::SUB, SDLoc(N), VT, N0, Mul);
      AddToWorklist(Mul.getNode());
      return Sub;
    }
  }

  // undef % X -> 0
  if (N0.getOpcode() == ISD::UNDEF)
    return DAG.getConstant(0, SDLoc(N), VT);
  // X % undef -> undef
  if (N1.getOpcode() == ISD::UNDEF)
    return N1;

  return SDValue();
}

SDValue DAGCombiner::visitUREM(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N->getValueType(0);

  // fold (urem c1, c2) -> c1%c2
  ConstantSDNode *N0C = isConstOrConstSplat(N0);
  ConstantSDNode *N1C = isConstOrConstSplat(N1);
  if (N0C && N1C)
    if (SDValue Folded = DAG.FoldConstantArithmetic(ISD::UREM, SDLoc(N), VT,
                                                    N0C, N1C))
      return Folded;
  // fold (urem x, pow2) -> (and x, pow2-1)
  if (N1C && !N1C->isNullValue() && !N1C->isOpaque() &&
      N1C->getAPIntValue().isPowerOf2()) {
    SDLoc DL(N);
    return DAG.getNode(ISD::AND, DL, VT, N0,
                       DAG.getConstant(N1C->getAPIntValue() - 1, DL, VT));
  }
  // fold (urem x, (shl pow2, y)) -> (and x, (add (shl pow2, y), -1))
  if (N1.getOpcode() == ISD::SHL) {
    if (ConstantSDNode *SHC = getAsNonOpaqueConstant(N1.getOperand(0))) {
      if (SHC->getAPIntValue().isPowerOf2()) {
        SDLoc DL(N);
        SDValue Add =
          DAG.getNode(ISD::ADD, DL, VT, N1,
                 DAG.getConstant(APInt::getAllOnesValue(VT.getSizeInBits()), DL,
                                 VT));
        AddToWorklist(Add.getNode());
        return DAG.getNode(ISD::AND, DL, VT, N0, Add);
      }
    }
  }

  // If X/C can be simplified by the division-by-constant logic, lower
  // X%C to the equivalent of X-X/C*C.
  if (N1C && !N1C->isNullValue()) {
    SDValue Div = DAG.getNode(ISD::UDIV, SDLoc(N), VT, N0, N1);
    AddToWorklist(Div.getNode());
    SDValue OptimizedDiv = combine(Div.getNode());
    if (OptimizedDiv.getNode() && OptimizedDiv.getNode() != Div.getNode()) {
      SDValue Mul = DAG.getNode(ISD::MUL, SDLoc(N), VT,
                                OptimizedDiv, N1);
      SDValue Sub = DAG.getNode(ISD::SUB, SDLoc(N), VT, N0, Mul);
      AddToWorklist(Mul.getNode());
      return Sub;
    }
  }

  // undef % X -> 0
  if (N0.getOpcode() == ISD::UNDEF)
    return DAG.getConstant(0, SDLoc(N), VT);
  // X % undef -> undef
  if (N1.getOpcode() == ISD::UNDEF)
    return N1;

  return SDValue();
}

SDValue DAGCombiner::visitMULHS(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N->getValueType(0);
  SDLoc DL(N);

  // fold (mulhs x, 0) -> 0
  if (isNullConstant(N1))
    return N1;
  // fold (mulhs x, 1) -> (sra x, size(x)-1)
  if (isOneConstant(N1)) {
    SDLoc DL(N);
    return DAG.getNode(ISD::SRA, DL, N0.getValueType(), N0,
                       DAG.getConstant(N0.getValueType().getSizeInBits() - 1,
                                       DL,
                                       getShiftAmountTy(N0.getValueType())));
  }
  // fold (mulhs x, undef) -> 0
  if (N0.getOpcode() == ISD::UNDEF || N1.getOpcode() == ISD::UNDEF)
    return DAG.getConstant(0, SDLoc(N), VT);

  // If the type twice as wide is legal, transform the mulhs to a wider multiply
  // plus a shift.
  if (VT.isSimple() && !VT.isVector()) {
    MVT Simple = VT.getSimpleVT();
    unsigned SimpleSize = Simple.getSizeInBits();
    EVT NewVT = EVT::getIntegerVT(*DAG.getContext(), SimpleSize*2);
    if (TLI.isOperationLegal(ISD::MUL, NewVT)) {
      N0 = DAG.getNode(ISD::SIGN_EXTEND, DL, NewVT, N0);
      N1 = DAG.getNode(ISD::SIGN_EXTEND, DL, NewVT, N1);
      N1 = DAG.getNode(ISD::MUL, DL, NewVT, N0, N1);
      N1 = DAG.getNode(ISD::SRL, DL, NewVT, N1,
            DAG.getConstant(SimpleSize, DL,
                            getShiftAmountTy(N1.getValueType())));
      return DAG.getNode(ISD::TRUNCATE, DL, VT, N1);
    }
  }

  return SDValue();
}

SDValue DAGCombiner::visitMULHU(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N->getValueType(0);
  SDLoc DL(N);

  // fold (mulhu x, 0) -> 0
  if (isNullConstant(N1))
    return N1;
  // fold (mulhu x, 1) -> 0
  if (isOneConstant(N1))
    return DAG.getConstant(0, DL, N0.getValueType());
  // fold (mulhu x, undef) -> 0
  if (N0.getOpcode() == ISD::UNDEF || N1.getOpcode() == ISD::UNDEF)
    return DAG.getConstant(0, DL, VT);

  // If the type twice as wide is legal, transform the mulhu to a wider multiply
  // plus a shift.
  if (VT.isSimple() && !VT.isVector()) {
    MVT Simple = VT.getSimpleVT();
    unsigned SimpleSize = Simple.getSizeInBits();
    EVT NewVT = EVT::getIntegerVT(*DAG.getContext(), SimpleSize*2);
    if (TLI.isOperationLegal(ISD::MUL, NewVT)) {
      N0 = DAG.getNode(ISD::ZERO_EXTEND, DL, NewVT, N0);
      N1 = DAG.getNode(ISD::ZERO_EXTEND, DL, NewVT, N1);
      N1 = DAG.getNode(ISD::MUL, DL, NewVT, N0, N1);
      N1 = DAG.getNode(ISD::SRL, DL, NewVT, N1,
            DAG.getConstant(SimpleSize, DL,
                            getShiftAmountTy(N1.getValueType())));
      return DAG.getNode(ISD::TRUNCATE, DL, VT, N1);
    }
  }

  return SDValue();
}

/// Perform optimizations common to nodes that compute two values. LoOp and HiOp
/// give the opcodes for the two computations that are being performed. Return
/// true if a simplification was made.
SDValue DAGCombiner::SimplifyNodeWithTwoResults(SDNode *N, unsigned LoOp,
                                                unsigned HiOp) {
  // If the high half is not needed, just compute the low half.
  bool HiExists = N->hasAnyUseOfValue(1);
  if (!HiExists &&
      (!LegalOperations ||
       TLI.isOperationLegalOrCustom(LoOp, N->getValueType(0)))) {
    SDValue Res = DAG.getNode(LoOp, SDLoc(N), N->getValueType(0), N->ops());
    return CombineTo(N, Res, Res);
  }

  // If the low half is not needed, just compute the high half.
  bool LoExists = N->hasAnyUseOfValue(0);
  if (!LoExists &&
      (!LegalOperations ||
       TLI.isOperationLegal(HiOp, N->getValueType(1)))) {
    SDValue Res = DAG.getNode(HiOp, SDLoc(N), N->getValueType(1), N->ops());
    return CombineTo(N, Res, Res);
  }

  // If both halves are used, return as it is.
  if (LoExists && HiExists)
    return SDValue();

  // If the two computed results can be simplified separately, separate them.
  if (LoExists) {
    SDValue Lo = DAG.getNode(LoOp, SDLoc(N), N->getValueType(0), N->ops());
    AddToWorklist(Lo.getNode());
    SDValue LoOpt = combine(Lo.getNode());
    if (LoOpt.getNode() && LoOpt.getNode() != Lo.getNode() &&
        (!LegalOperations ||
         TLI.isOperationLegal(LoOpt.getOpcode(), LoOpt.getValueType())))
      return CombineTo(N, LoOpt, LoOpt);
  }

  if (HiExists) {
    SDValue Hi = DAG.getNode(HiOp, SDLoc(N), N->getValueType(1), N->ops());
    AddToWorklist(Hi.getNode());
    SDValue HiOpt = combine(Hi.getNode());
    if (HiOpt.getNode() && HiOpt != Hi &&
        (!LegalOperations ||
         TLI.isOperationLegal(HiOpt.getOpcode(), HiOpt.getValueType())))
      return CombineTo(N, HiOpt, HiOpt);
  }

  return SDValue();
}

SDValue DAGCombiner::visitSMUL_LOHI(SDNode *N) {
  SDValue Res = SimplifyNodeWithTwoResults(N, ISD::MUL, ISD::MULHS);
  if (Res.getNode()) return Res;

  EVT VT = N->getValueType(0);
  SDLoc DL(N);

  // If the type is twice as wide is legal, transform the mulhu to a wider
  // multiply plus a shift.
  if (VT.isSimple() && !VT.isVector()) {
    MVT Simple = VT.getSimpleVT();
    unsigned SimpleSize = Simple.getSizeInBits();
    EVT NewVT = EVT::getIntegerVT(*DAG.getContext(), SimpleSize*2);
    if (TLI.isOperationLegal(ISD::MUL, NewVT)) {
      SDValue Lo = DAG.getNode(ISD::SIGN_EXTEND, DL, NewVT, N->getOperand(0));
      SDValue Hi = DAG.getNode(ISD::SIGN_EXTEND, DL, NewVT, N->getOperand(1));
      Lo = DAG.getNode(ISD::MUL, DL, NewVT, Lo, Hi);
      // Compute the high part as N1.
      Hi = DAG.getNode(ISD::SRL, DL, NewVT, Lo,
            DAG.getConstant(SimpleSize, DL,
                            getShiftAmountTy(Lo.getValueType())));
      Hi = DAG.getNode(ISD::TRUNCATE, DL, VT, Hi);
      // Compute the low part as N0.
      Lo = DAG.getNode(ISD::TRUNCATE, DL, VT, Lo);
      return CombineTo(N, Lo, Hi);
    }
  }

  return SDValue();
}

SDValue DAGCombiner::visitUMUL_LOHI(SDNode *N) {
  SDValue Res = SimplifyNodeWithTwoResults(N, ISD::MUL, ISD::MULHU);
  if (Res.getNode()) return Res;

  EVT VT = N->getValueType(0);
  SDLoc DL(N);

  // If the type is twice as wide is legal, transform the mulhu to a wider
  // multiply plus a shift.
  if (VT.isSimple() && !VT.isVector()) {
    MVT Simple = VT.getSimpleVT();
    unsigned SimpleSize = Simple.getSizeInBits();
    EVT NewVT = EVT::getIntegerVT(*DAG.getContext(), SimpleSize*2);
    if (TLI.isOperationLegal(ISD::MUL, NewVT)) {
      SDValue Lo = DAG.getNode(ISD::ZERO_EXTEND, DL, NewVT, N->getOperand(0));
      SDValue Hi = DAG.getNode(ISD::ZERO_EXTEND, DL, NewVT, N->getOperand(1));
      Lo = DAG.getNode(ISD::MUL, DL, NewVT, Lo, Hi);
      // Compute the high part as N1.
      Hi = DAG.getNode(ISD::SRL, DL, NewVT, Lo,
            DAG.getConstant(SimpleSize, DL,
                            getShiftAmountTy(Lo.getValueType())));
      Hi = DAG.getNode(ISD::TRUNCATE, DL, VT, Hi);
      // Compute the low part as N0.
      Lo = DAG.getNode(ISD::TRUNCATE, DL, VT, Lo);
      return CombineTo(N, Lo, Hi);
    }
  }

  return SDValue();
}

SDValue DAGCombiner::visitSMULO(SDNode *N) {
  // (smulo x, 2) -> (saddo x, x)
  if (ConstantSDNode *C2 = dyn_cast<ConstantSDNode>(N->getOperand(1)))
    if (C2->getAPIntValue() == 2)
      return DAG.getNode(ISD::SADDO, SDLoc(N), N->getVTList(),
                         N->getOperand(0), N->getOperand(0));

  return SDValue();
}

SDValue DAGCombiner::visitUMULO(SDNode *N) {
  // (umulo x, 2) -> (uaddo x, x)
  if (ConstantSDNode *C2 = dyn_cast<ConstantSDNode>(N->getOperand(1)))
    if (C2->getAPIntValue() == 2)
      return DAG.getNode(ISD::UADDO, SDLoc(N), N->getVTList(),
                         N->getOperand(0), N->getOperand(0));

  return SDValue();
}

SDValue DAGCombiner::visitSDIVREM(SDNode *N) {
  SDValue Res = SimplifyNodeWithTwoResults(N, ISD::SDIV, ISD::SREM);
  if (Res.getNode()) return Res;

  return SDValue();
}

SDValue DAGCombiner::visitUDIVREM(SDNode *N) {
  SDValue Res = SimplifyNodeWithTwoResults(N, ISD::UDIV, ISD::UREM);
  if (Res.getNode()) return Res;

  return SDValue();
}

/// If this is a binary operator with two operands of the same opcode, try to
/// simplify it.
SDValue DAGCombiner::SimplifyBinOpWithSameOpcodeHands(SDNode *N) {
  SDValue N0 = N->getOperand(0), N1 = N->getOperand(1);
  EVT VT = N0.getValueType();
  assert(N0.getOpcode() == N1.getOpcode() && "Bad input!");

  // Bail early if none of these transforms apply.
  if (N0.getNode()->getNumOperands() == 0) return SDValue();

  // For each of OP in AND/OR/XOR:
  // fold (OP (zext x), (zext y)) -> (zext (OP x, y))
  // fold (OP (sext x), (sext y)) -> (sext (OP x, y))
  // fold (OP (aext x), (aext y)) -> (aext (OP x, y))
  // fold (OP (bswap x), (bswap y)) -> (bswap (OP x, y))
  // fold (OP (trunc x), (trunc y)) -> (trunc (OP x, y)) (if trunc isn't free)
  //
  // do not sink logical op inside of a vector extend, since it may combine
  // into a vsetcc.
  EVT Op0VT = N0.getOperand(0).getValueType();
  if ((N0.getOpcode() == ISD::ZERO_EXTEND ||
       N0.getOpcode() == ISD::SIGN_EXTEND ||
       N0.getOpcode() == ISD::BSWAP ||
       // Avoid infinite looping with PromoteIntBinOp.
       (N0.getOpcode() == ISD::ANY_EXTEND &&
        (!LegalTypes || TLI.isTypeDesirableForOp(N->getOpcode(), Op0VT))) ||
       (N0.getOpcode() == ISD::TRUNCATE &&
        (!TLI.isZExtFree(VT, Op0VT) ||
         !TLI.isTruncateFree(Op0VT, VT)) &&
        TLI.isTypeLegal(Op0VT))) &&
      !VT.isVector() &&
      Op0VT == N1.getOperand(0).getValueType() &&
      (!LegalOperations || TLI.isOperationLegal(N->getOpcode(), Op0VT))) {
    SDValue ORNode = DAG.getNode(N->getOpcode(), SDLoc(N0),
                                 N0.getOperand(0).getValueType(),
                                 N0.getOperand(0), N1.getOperand(0));
    AddToWorklist(ORNode.getNode());
    return DAG.getNode(N0.getOpcode(), SDLoc(N), VT, ORNode);
  }

  // For each of OP in SHL/SRL/SRA/AND...
  //   fold (and (OP x, z), (OP y, z)) -> (OP (and x, y), z)
  //   fold (or  (OP x, z), (OP y, z)) -> (OP (or  x, y), z)
  //   fold (xor (OP x, z), (OP y, z)) -> (OP (xor x, y), z)
  if ((N0.getOpcode() == ISD::SHL || N0.getOpcode() == ISD::SRL ||
       N0.getOpcode() == ISD::SRA || N0.getOpcode() == ISD::AND) &&
      N0.getOperand(1) == N1.getOperand(1)) {
    SDValue ORNode = DAG.getNode(N->getOpcode(), SDLoc(N0),
                                 N0.getOperand(0).getValueType(),
                                 N0.getOperand(0), N1.getOperand(0));
    AddToWorklist(ORNode.getNode());
    return DAG.getNode(N0.getOpcode(), SDLoc(N), VT,
                       ORNode, N0.getOperand(1));
  }

  // Simplify xor/and/or (bitcast(A), bitcast(B)) -> bitcast(op (A,B))
  // Only perform this optimization after type legalization and before
  // LegalizeVectorOprs. LegalizeVectorOprs promotes vector operations by
  // adding bitcasts. For example (xor v4i32) is promoted to (v2i64), and
  // we don't want to undo this promotion.
  // We also handle SCALAR_TO_VECTOR because xor/or/and operations are cheaper
  // on scalars.
  if ((N0.getOpcode() == ISD::BITCAST ||
       N0.getOpcode() == ISD::SCALAR_TO_VECTOR) &&
      Level == AfterLegalizeTypes) {
    SDValue In0 = N0.getOperand(0);
    SDValue In1 = N1.getOperand(0);
    EVT In0Ty = In0.getValueType();
    EVT In1Ty = In1.getValueType();
    SDLoc DL(N);
    // If both incoming values are integers, and the original types are the
    // same.
    if (In0Ty.isInteger() && In1Ty.isInteger() && In0Ty == In1Ty) {
      SDValue Op = DAG.getNode(N->getOpcode(), DL, In0Ty, In0, In1);
      SDValue BC = DAG.getNode(N0.getOpcode(), DL, VT, Op);
      AddToWorklist(Op.getNode());
      return BC;
    }
  }

  // Xor/and/or are indifferent to the swizzle operation (shuffle of one value).
  // Simplify xor/and/or (shuff(A), shuff(B)) -> shuff(op (A,B))
  // If both shuffles use the same mask, and both shuffle within a single
  // vector, then it is worthwhile to move the swizzle after the operation.
  // The type-legalizer generates this pattern when loading illegal
  // vector types from memory. In many cases this allows additional shuffle
  // optimizations.
  // There are other cases where moving the shuffle after the xor/and/or
  // is profitable even if shuffles don't perform a swizzle.
  // If both shuffles use the same mask, and both shuffles have the same first
  // or second operand, then it might still be profitable to move the shuffle
  // after the xor/and/or operation.
  if (N0.getOpcode() == ISD::VECTOR_SHUFFLE && Level < AfterLegalizeDAG) {
    ShuffleVectorSDNode *SVN0 = cast<ShuffleVectorSDNode>(N0);
    ShuffleVectorSDNode *SVN1 = cast<ShuffleVectorSDNode>(N1);

    assert(N0.getOperand(0).getValueType() == N1.getOperand(0).getValueType() &&
           "Inputs to shuffles are not the same type");

    // Check that both shuffles use the same mask. The masks are known to be of
    // the same length because the result vector type is the same.
    // Check also that shuffles have only one use to avoid introducing extra
    // instructions.
    if (SVN0->hasOneUse() && SVN1->hasOneUse() &&
        SVN0->getMask().equals(SVN1->getMask())) {
      SDValue ShOp = N0->getOperand(1);

      // Don't try to fold this node if it requires introducing a
      // build vector of all zeros that might be illegal at this stage.
      if (N->getOpcode() == ISD::XOR && ShOp.getOpcode() != ISD::UNDEF) {
        if (!LegalTypes)
          ShOp = DAG.getConstant(0, SDLoc(N), VT);
        else
          ShOp = SDValue();
      }

      // (AND (shuf (A, C), shuf (B, C)) -> shuf (AND (A, B), C)
      // (OR  (shuf (A, C), shuf (B, C)) -> shuf (OR  (A, B), C)
      // (XOR (shuf (A, C), shuf (B, C)) -> shuf (XOR (A, B), V_0)
      if (N0.getOperand(1) == N1.getOperand(1) && ShOp.getNode()) {
        SDValue NewNode = DAG.getNode(N->getOpcode(), SDLoc(N), VT,
                                      N0->getOperand(0), N1->getOperand(0));
        AddToWorklist(NewNode.getNode());
        return DAG.getVectorShuffle(VT, SDLoc(N), NewNode, ShOp,
                                    &SVN0->getMask()[0]);
      }

      // Don't try to fold this node if it requires introducing a
      // build vector of all zeros that might be illegal at this stage.
      ShOp = N0->getOperand(0);
      if (N->getOpcode() == ISD::XOR && ShOp.getOpcode() != ISD::UNDEF) {
        if (!LegalTypes)
          ShOp = DAG.getConstant(0, SDLoc(N), VT);
        else
          ShOp = SDValue();
      }

      // (AND (shuf (C, A), shuf (C, B)) -> shuf (C, AND (A, B))
      // (OR  (shuf (C, A), shuf (C, B)) -> shuf (C, OR  (A, B))
      // (XOR (shuf (C, A), shuf (C, B)) -> shuf (V_0, XOR (A, B))
      if (N0->getOperand(0) == N1->getOperand(0) && ShOp.getNode()) {
        SDValue NewNode = DAG.getNode(N->getOpcode(), SDLoc(N), VT,
                                      N0->getOperand(1), N1->getOperand(1));
        AddToWorklist(NewNode.getNode());
        return DAG.getVectorShuffle(VT, SDLoc(N), ShOp, NewNode,
                                    &SVN0->getMask()[0]);
      }
    }
  }

  return SDValue();
}

/// This contains all DAGCombine rules which reduce two values combined by
/// an And operation to a single value. This makes them reusable in the context
/// of visitSELECT(). Rules involving constants are not included as
/// visitSELECT() already handles those cases.
SDValue DAGCombiner::visitANDLike(SDValue N0, SDValue N1,
                                  SDNode *LocReference) {
  EVT VT = N1.getValueType();

  // fold (and x, undef) -> 0
  if (N0.getOpcode() == ISD::UNDEF || N1.getOpcode() == ISD::UNDEF)
    return DAG.getConstant(0, SDLoc(LocReference), VT);
  // fold (and (setcc x), (setcc y)) -> (setcc (and x, y))
  SDValue LL, LR, RL, RR, CC0, CC1;
  if (isSetCCEquivalent(N0, LL, LR, CC0) && isSetCCEquivalent(N1, RL, RR, CC1)){
    ISD::CondCode Op0 = cast<CondCodeSDNode>(CC0)->get();
    ISD::CondCode Op1 = cast<CondCodeSDNode>(CC1)->get();

    if (LR == RR && isa<ConstantSDNode>(LR) && Op0 == Op1 &&
        LL.getValueType().isInteger()) {
      // fold (and (seteq X, 0), (seteq Y, 0)) -> (seteq (or X, Y), 0)
      if (isNullConstant(LR) && Op1 == ISD::SETEQ) {
        SDValue ORNode = DAG.getNode(ISD::OR, SDLoc(N0),
                                     LR.getValueType(), LL, RL);
        AddToWorklist(ORNode.getNode());
        return DAG.getSetCC(SDLoc(LocReference), VT, ORNode, LR, Op1);
      }
      if (isAllOnesConstant(LR)) {
        // fold (and (seteq X, -1), (seteq Y, -1)) -> (seteq (and X, Y), -1)
        if (Op1 == ISD::SETEQ) {
          SDValue ANDNode = DAG.getNode(ISD::AND, SDLoc(N0),
                                        LR.getValueType(), LL, RL);
          AddToWorklist(ANDNode.getNode());
          return DAG.getSetCC(SDLoc(LocReference), VT, ANDNode, LR, Op1);
        }
        // fold (and (setgt X, -1), (setgt Y, -1)) -> (setgt (or X, Y), -1)
        if (Op1 == ISD::SETGT) {
          SDValue ORNode = DAG.getNode(ISD::OR, SDLoc(N0),
                                       LR.getValueType(), LL, RL);
          AddToWorklist(ORNode.getNode());
          return DAG.getSetCC(SDLoc(LocReference), VT, ORNode, LR, Op1);
        }
      }
    }
    // Simplify (and (setne X, 0), (setne X, -1)) -> (setuge (add X, 1), 2)
    if (LL == RL && isa<ConstantSDNode>(LR) && isa<ConstantSDNode>(RR) &&
        Op0 == Op1 && LL.getValueType().isInteger() &&
      Op0 == ISD::SETNE && ((isNullConstant(LR) && isAllOnesConstant(RR)) ||
                            (isAllOnesConstant(LR) && isNullConstant(RR)))) {
      SDLoc DL(N0);
      SDValue ADDNode = DAG.getNode(ISD::ADD, DL, LL.getValueType(),
                                    LL, DAG.getConstant(1, DL,
                                                        LL.getValueType()));
      AddToWorklist(ADDNode.getNode());
      return DAG.getSetCC(SDLoc(LocReference), VT, ADDNode,
                          DAG.getConstant(2, DL, LL.getValueType()),
                          ISD::SETUGE);
    }
    // canonicalize equivalent to ll == rl
    if (LL == RR && LR == RL) {
      Op1 = ISD::getSetCCSwappedOperands(Op1);
      std::swap(RL, RR);
    }
    if (LL == RL && LR == RR) {
      bool isInteger = LL.getValueType().isInteger();
      ISD::CondCode Result = ISD::getSetCCAndOperation(Op0, Op1, isInteger);
      if (Result != ISD::SETCC_INVALID &&
          (!LegalOperations ||
           (TLI.isCondCodeLegal(Result, LL.getSimpleValueType()) &&
            TLI.isOperationLegal(ISD::SETCC,
                            getSetCCResultType(N0.getSimpleValueType())))))
        return DAG.getSetCC(SDLoc(LocReference), N0.getValueType(),
                            LL, LR, Result);
    }
  }

  if (N0.getOpcode() == ISD::ADD && N1.getOpcode() == ISD::SRL &&
      VT.getSizeInBits() <= 64) {
    if (ConstantSDNode *ADDI = dyn_cast<ConstantSDNode>(N0.getOperand(1))) {
      APInt ADDC = ADDI->getAPIntValue();
      if (!TLI.isLegalAddImmediate(ADDC.getSExtValue())) {
        // Look for (and (add x, c1), (lshr y, c2)). If C1 wasn't a legal
        // immediate for an add, but it is legal if its top c2 bits are set,
        // transform the ADD so the immediate doesn't need to be materialized
        // in a register.
        if (ConstantSDNode *SRLI = dyn_cast<ConstantSDNode>(N1.getOperand(1))) {
          APInt Mask = APInt::getHighBitsSet(VT.getSizeInBits(),
                                             SRLI->getZExtValue());
          if (DAG.MaskedValueIsZero(N0.getOperand(1), Mask)) {
            ADDC |= Mask;
            if (TLI.isLegalAddImmediate(ADDC.getSExtValue())) {
              SDLoc DL(N0);
              SDValue NewAdd =
                DAG.getNode(ISD::ADD, DL, VT,
                            N0.getOperand(0), DAG.getConstant(ADDC, DL, VT));
              CombineTo(N0.getNode(), NewAdd);
              // Return N so it doesn't get rechecked!
              return SDValue(LocReference, 0);
            }
          }
        }
      }
    }
  }

  return SDValue();
}

SDValue DAGCombiner::visitAND(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N1.getValueType();

  // fold vector ops
  if (VT.isVector()) {
    if (SDValue FoldedVOp = SimplifyVBinOp(N))
      return FoldedVOp;

    // fold (and x, 0) -> 0, vector edition
    if (ISD::isBuildVectorAllZeros(N0.getNode()))
      // do not return N0, because undef node may exist in N0
      return DAG.getConstant(
          APInt::getNullValue(
              N0.getValueType().getScalarType().getSizeInBits()),
          SDLoc(N), N0.getValueType());
    if (ISD::isBuildVectorAllZeros(N1.getNode()))
      // do not return N1, because undef node may exist in N1
      return DAG.getConstant(
          APInt::getNullValue(
              N1.getValueType().getScalarType().getSizeInBits()),
          SDLoc(N), N1.getValueType());

    // fold (and x, -1) -> x, vector edition
    if (ISD::isBuildVectorAllOnes(N0.getNode()))
      return N1;
    if (ISD::isBuildVectorAllOnes(N1.getNode()))
      return N0;
  }

  // fold (and c1, c2) -> c1&c2
  ConstantSDNode *N0C = getAsNonOpaqueConstant(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  if (N0C && N1C && !N1C->isOpaque())
    return DAG.FoldConstantArithmetic(ISD::AND, SDLoc(N), VT, N0C, N1C);
  // canonicalize constant to RHS
  if (isConstantIntBuildVectorOrConstantInt(N0) &&
     !isConstantIntBuildVectorOrConstantInt(N1))
    return DAG.getNode(ISD::AND, SDLoc(N), VT, N1, N0);
  // fold (and x, -1) -> x
  if (isAllOnesConstant(N1))
    return N0;
  // if (and x, c) is known to be zero, return 0
  unsigned BitWidth = VT.getScalarType().getSizeInBits();
  if (N1C && DAG.MaskedValueIsZero(SDValue(N, 0),
                                   APInt::getAllOnesValue(BitWidth)))
    return DAG.getConstant(0, SDLoc(N), VT);
  // reassociate and
  if (SDValue RAND = ReassociateOps(ISD::AND, SDLoc(N), N0, N1))
    return RAND;
  // fold (and (or x, C), D) -> D if (C & D) == D
  if (N1C && N0.getOpcode() == ISD::OR)
    if (ConstantSDNode *ORI = dyn_cast<ConstantSDNode>(N0.getOperand(1)))
      if ((ORI->getAPIntValue() & N1C->getAPIntValue()) == N1C->getAPIntValue())
        return N1;
  // fold (and (any_ext V), c) -> (zero_ext V) if 'and' only clears top bits.
  if (N1C && N0.getOpcode() == ISD::ANY_EXTEND) {
    SDValue N0Op0 = N0.getOperand(0);
    APInt Mask = ~N1C->getAPIntValue();
    Mask = Mask.trunc(N0Op0.getValueSizeInBits());
    if (DAG.MaskedValueIsZero(N0Op0, Mask)) {
      SDValue Zext = DAG.getNode(ISD::ZERO_EXTEND, SDLoc(N),
                                 N0.getValueType(), N0Op0);

      // Replace uses of the AND with uses of the Zero extend node.
      CombineTo(N, Zext);

      // We actually want to replace all uses of the any_extend with the
      // zero_extend, to avoid duplicating things.  This will later cause this
      // AND to be folded.
      CombineTo(N0.getNode(), Zext);
      return SDValue(N, 0);   // Return N so it doesn't get rechecked!
    }
  }
  // similarly fold (and (X (load ([non_ext|any_ext|zero_ext] V))), c) ->
  // (X (load ([non_ext|zero_ext] V))) if 'and' only clears top bits which must
  // already be zero by virtue of the width of the base type of the load.
  //
  // the 'X' node here can either be nothing or an extract_vector_elt to catch
  // more cases.
  if ((N0.getOpcode() == ISD::EXTRACT_VECTOR_ELT &&
       N0.getOperand(0).getOpcode() == ISD::LOAD) ||
      N0.getOpcode() == ISD::LOAD) {
    LoadSDNode *Load = cast<LoadSDNode>( (N0.getOpcode() == ISD::LOAD) ?
                                         N0 : N0.getOperand(0) );

    // Get the constant (if applicable) the zero'th operand is being ANDed with.
    // This can be a pure constant or a vector splat, in which case we treat the
    // vector as a scalar and use the splat value.
    APInt Constant = APInt::getNullValue(1);
    if (const ConstantSDNode *C = dyn_cast<ConstantSDNode>(N1)) {
      Constant = C->getAPIntValue();
    } else if (BuildVectorSDNode *Vector = dyn_cast<BuildVectorSDNode>(N1)) {
      APInt SplatValue, SplatUndef;
      unsigned SplatBitSize;
      bool HasAnyUndefs;
      bool IsSplat = Vector->isConstantSplat(SplatValue, SplatUndef,
                                             SplatBitSize, HasAnyUndefs);
      if (IsSplat) {
        // Undef bits can contribute to a possible optimisation if set, so
        // set them.
        SplatValue |= SplatUndef;

        // The splat value may be something like "0x00FFFFFF", which means 0 for
        // the first vector value and FF for the rest, repeating. We need a mask
        // that will apply equally to all members of the vector, so AND all the
        // lanes of the constant together.
        EVT VT = Vector->getValueType(0);
        unsigned BitWidth = VT.getVectorElementType().getSizeInBits();

        // If the splat value has been compressed to a bitlength lower
        // than the size of the vector lane, we need to re-expand it to
        // the lane size.
        if (BitWidth > SplatBitSize)
          for (SplatValue = SplatValue.zextOrTrunc(BitWidth);
               SplatBitSize < BitWidth;
               SplatBitSize = SplatBitSize * 2)
            SplatValue |= SplatValue.shl(SplatBitSize);

        // Make sure that variable 'Constant' is only set if 'SplatBitSize' is a
        // multiple of 'BitWidth'. Otherwise, we could propagate a wrong value.
        if (SplatBitSize % BitWidth == 0) {
          Constant = APInt::getAllOnesValue(BitWidth);
          for (unsigned i = 0, n = SplatBitSize/BitWidth; i < n; ++i)
            Constant &= SplatValue.lshr(i*BitWidth).zextOrTrunc(BitWidth);
        }
      }
    }

    // If we want to change an EXTLOAD to a ZEXTLOAD, ensure a ZEXTLOAD is
    // actually legal and isn't going to get expanded, else this is a false
    // optimisation.
    bool CanZextLoadProfitably = TLI.isLoadExtLegal(ISD::ZEXTLOAD,
                                                    Load->getValueType(0),
                                                    Load->getMemoryVT());

    // Resize the constant to the same size as the original memory access before
    // extension. If it is still the AllOnesValue then this AND is completely
    // unneeded.
    Constant =
      Constant.zextOrTrunc(Load->getMemoryVT().getScalarType().getSizeInBits());

    bool B;
    switch (Load->getExtensionType()) {
    default: B = false; break;
    case ISD::EXTLOAD: B = CanZextLoadProfitably; break;
    case ISD::ZEXTLOAD:
    case ISD::NON_EXTLOAD: B = true; break;
    }

    if (B && Constant.isAllOnesValue()) {
      // If the load type was an EXTLOAD, convert to ZEXTLOAD in order to
      // preserve semantics once we get rid of the AND.
      SDValue NewLoad(Load, 0);
      if (Load->getExtensionType() == ISD::EXTLOAD) {
        NewLoad = DAG.getLoad(Load->getAddressingMode(), ISD::ZEXTLOAD,
                              Load->getValueType(0), SDLoc(Load),
                              Load->getChain(), Load->getBasePtr(),
                              Load->getOffset(), Load->getMemoryVT(),
                              Load->getMemOperand());
        // Replace uses of the EXTLOAD with the new ZEXTLOAD.
        if (Load->getNumValues() == 3) {
          // PRE/POST_INC loads have 3 values.
          SDValue To[] = { NewLoad.getValue(0), NewLoad.getValue(1),
                           NewLoad.getValue(2) };
          CombineTo(Load, To, 3, true);
        } else {
          CombineTo(Load, NewLoad.getValue(0), NewLoad.getValue(1));
        }
      }

      // Fold the AND away, taking care not to fold to the old load node if we
      // replaced it.
      CombineTo(N, (N0.getNode() == Load) ? NewLoad : N0);

      return SDValue(N, 0); // Return N so it doesn't get rechecked!
    }
  }

  // fold (and (load x), 255) -> (zextload x, i8)
  // fold (and (extload x, i16), 255) -> (zextload x, i8)
  // fold (and (any_ext (extload x, i16)), 255) -> (zextload x, i8)
  if (N1C && (N0.getOpcode() == ISD::LOAD ||
              (N0.getOpcode() == ISD::ANY_EXTEND &&
               N0.getOperand(0).getOpcode() == ISD::LOAD))) {
    bool HasAnyExt = N0.getOpcode() == ISD::ANY_EXTEND;
    LoadSDNode *LN0 = HasAnyExt
      ? cast<LoadSDNode>(N0.getOperand(0))
      : cast<LoadSDNode>(N0);
    if (LN0->getExtensionType() != ISD::SEXTLOAD &&
        LN0->isUnindexed() && N0.hasOneUse() && SDValue(LN0, 0).hasOneUse()) {
      uint32_t ActiveBits = N1C->getAPIntValue().getActiveBits();
      if (ActiveBits > 0 && APIntOps::isMask(ActiveBits, N1C->getAPIntValue())){
        EVT ExtVT = EVT::getIntegerVT(*DAG.getContext(), ActiveBits);
        EVT LoadedVT = LN0->getMemoryVT();
        EVT LoadResultTy = HasAnyExt ? LN0->getValueType(0) : VT;

        if (ExtVT == LoadedVT &&
            (!LegalOperations || TLI.isLoadExtLegal(ISD::ZEXTLOAD, LoadResultTy,
                                                    ExtVT))) {

          SDValue NewLoad =
            DAG.getExtLoad(ISD::ZEXTLOAD, SDLoc(LN0), LoadResultTy,
                           LN0->getChain(), LN0->getBasePtr(), ExtVT,
                           LN0->getMemOperand());
          AddToWorklist(N);
          CombineTo(LN0, NewLoad, NewLoad.getValue(1));
          return SDValue(N, 0);   // Return N so it doesn't get rechecked!
        }

        // Do not change the width of a volatile load.
        // Do not generate loads of non-round integer types since these can
        // be expensive (and would be wrong if the type is not byte sized).
        if (!LN0->isVolatile() && LoadedVT.bitsGT(ExtVT) && ExtVT.isRound() &&
            (!LegalOperations || TLI.isLoadExtLegal(ISD::ZEXTLOAD, LoadResultTy,
                                                    ExtVT))) {
          EVT PtrType = LN0->getOperand(1).getValueType();

          unsigned Alignment = LN0->getAlignment();
          SDValue NewPtr = LN0->getBasePtr();

          // For big endian targets, we need to add an offset to the pointer
          // to load the correct bytes.  For little endian systems, we merely
          // need to read fewer bytes from the same pointer.
          if (DAG.getDataLayout().isBigEndian()) {
            unsigned LVTStoreBytes = LoadedVT.getStoreSize();
            unsigned EVTStoreBytes = ExtVT.getStoreSize();
            unsigned PtrOff = LVTStoreBytes - EVTStoreBytes;
            SDLoc DL(LN0);
            NewPtr = DAG.getNode(ISD::ADD, DL, PtrType,
                                 NewPtr, DAG.getConstant(PtrOff, DL, PtrType));
            Alignment = MinAlign(Alignment, PtrOff);
          }

          AddToWorklist(NewPtr.getNode());

          SDValue Load =
            DAG.getExtLoad(ISD::ZEXTLOAD, SDLoc(LN0), LoadResultTy,
                           LN0->getChain(), NewPtr,
                           LN0->getPointerInfo(),
                           ExtVT, LN0->isVolatile(), LN0->isNonTemporal(),
                           LN0->isInvariant(), Alignment, LN0->getAAInfo());
          AddToWorklist(N);
          CombineTo(LN0, Load, Load.getValue(1));
          return SDValue(N, 0);   // Return N so it doesn't get rechecked!
        }
      }
    }
  }

  if (SDValue Combined = visitANDLike(N0, N1, N))
    return Combined;

  // Simplify: (and (op x...), (op y...))  -> (op (and x, y))
  if (N0.getOpcode() == N1.getOpcode()) {
    SDValue Tmp = SimplifyBinOpWithSameOpcodeHands(N);
    if (Tmp.getNode()) return Tmp;
  }

  // fold (and (sign_extend_inreg x, i16 to i32), 1) -> (and x, 1)
  // fold (and (sra)) -> (and (srl)) when possible.
  if (!VT.isVector() &&
      SimplifyDemandedBits(SDValue(N, 0)))
    return SDValue(N, 0);

  // fold (zext_inreg (extload x)) -> (zextload x)
  if (ISD::isEXTLoad(N0.getNode()) && ISD::isUNINDEXEDLoad(N0.getNode())) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);
    EVT MemVT = LN0->getMemoryVT();
    // If we zero all the possible extended bits, then we can turn this into
    // a zextload if we are running before legalize or the operation is legal.
    unsigned BitWidth = N1.getValueType().getScalarType().getSizeInBits();
    if (DAG.MaskedValueIsZero(N1, APInt::getHighBitsSet(BitWidth,
                           BitWidth - MemVT.getScalarType().getSizeInBits())) &&
        ((!LegalOperations && !LN0->isVolatile()) ||
         TLI.isLoadExtLegal(ISD::ZEXTLOAD, VT, MemVT))) {
      SDValue ExtLoad = DAG.getExtLoad(ISD::ZEXTLOAD, SDLoc(N0), VT,
                                       LN0->getChain(), LN0->getBasePtr(),
                                       MemVT, LN0->getMemOperand());
      AddToWorklist(N);
      CombineTo(N0.getNode(), ExtLoad, ExtLoad.getValue(1));
      return SDValue(N, 0);   // Return N so it doesn't get rechecked!
    }
  }
  // fold (zext_inreg (sextload x)) -> (zextload x) iff load has one use
  if (ISD::isSEXTLoad(N0.getNode()) && ISD::isUNINDEXEDLoad(N0.getNode()) &&
      N0.hasOneUse()) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);
    EVT MemVT = LN0->getMemoryVT();
    // If we zero all the possible extended bits, then we can turn this into
    // a zextload if we are running before legalize or the operation is legal.
    unsigned BitWidth = N1.getValueType().getScalarType().getSizeInBits();
    if (DAG.MaskedValueIsZero(N1, APInt::getHighBitsSet(BitWidth,
                           BitWidth - MemVT.getScalarType().getSizeInBits())) &&
        ((!LegalOperations && !LN0->isVolatile()) ||
         TLI.isLoadExtLegal(ISD::ZEXTLOAD, VT, MemVT))) {
      SDValue ExtLoad = DAG.getExtLoad(ISD::ZEXTLOAD, SDLoc(N0), VT,
                                       LN0->getChain(), LN0->getBasePtr(),
                                       MemVT, LN0->getMemOperand());
      AddToWorklist(N);
      CombineTo(N0.getNode(), ExtLoad, ExtLoad.getValue(1));
      return SDValue(N, 0);   // Return N so it doesn't get rechecked!
    }
  }
  // fold (and (or (srl N, 8), (shl N, 8)), 0xffff) -> (srl (bswap N), const)
  if (N1C && N1C->getAPIntValue() == 0xffff && N0.getOpcode() == ISD::OR) {
    SDValue BSwap = MatchBSwapHWordLow(N0.getNode(), N0.getOperand(0),
                                       N0.getOperand(1), false);
    if (BSwap.getNode())
      return BSwap;
  }

  return SDValue();
}

/// Match (a >> 8) | (a << 8) as (bswap a) >> 16.
SDValue DAGCombiner::MatchBSwapHWordLow(SDNode *N, SDValue N0, SDValue N1,
                                        bool DemandHighBits) {
  if (!LegalOperations)
    return SDValue();

  EVT VT = N->getValueType(0);
  if (VT != MVT::i64 && VT != MVT::i32 && VT != MVT::i16)
    return SDValue();
  if (!TLI.isOperationLegal(ISD::BSWAP, VT))
    return SDValue();

  // Recognize (and (shl a, 8), 0xff), (and (srl a, 8), 0xff00)
  bool LookPassAnd0 = false;
  bool LookPassAnd1 = false;
  if (N0.getOpcode() == ISD::AND && N0.getOperand(0).getOpcode() == ISD::SRL)
      std::swap(N0, N1);
  if (N1.getOpcode() == ISD::AND && N1.getOperand(0).getOpcode() == ISD::SHL)
      std::swap(N0, N1);
  if (N0.getOpcode() == ISD::AND) {
    if (!N0.getNode()->hasOneUse())
      return SDValue();
    ConstantSDNode *N01C = dyn_cast<ConstantSDNode>(N0.getOperand(1));
    if (!N01C || N01C->getZExtValue() != 0xFF00)
      return SDValue();
    N0 = N0.getOperand(0);
    LookPassAnd0 = true;
  }

  if (N1.getOpcode() == ISD::AND) {
    if (!N1.getNode()->hasOneUse())
      return SDValue();
    ConstantSDNode *N11C = dyn_cast<ConstantSDNode>(N1.getOperand(1));
    if (!N11C || N11C->getZExtValue() != 0xFF)
      return SDValue();
    N1 = N1.getOperand(0);
    LookPassAnd1 = true;
  }

  if (N0.getOpcode() == ISD::SRL && N1.getOpcode() == ISD::SHL)
    std::swap(N0, N1);
  if (N0.getOpcode() != ISD::SHL || N1.getOpcode() != ISD::SRL)
    return SDValue();
  if (!N0.getNode()->hasOneUse() ||
      !N1.getNode()->hasOneUse())
    return SDValue();

  ConstantSDNode *N01C = dyn_cast<ConstantSDNode>(N0.getOperand(1));
  ConstantSDNode *N11C = dyn_cast<ConstantSDNode>(N1.getOperand(1));
  if (!N01C || !N11C)
    return SDValue();
  if (N01C->getZExtValue() != 8 || N11C->getZExtValue() != 8)
    return SDValue();

  // Look for (shl (and a, 0xff), 8), (srl (and a, 0xff00), 8)
  SDValue N00 = N0->getOperand(0);
  if (!LookPassAnd0 && N00.getOpcode() == ISD::AND) {
    if (!N00.getNode()->hasOneUse())
      return SDValue();
    ConstantSDNode *N001C = dyn_cast<ConstantSDNode>(N00.getOperand(1));
    if (!N001C || N001C->getZExtValue() != 0xFF)
      return SDValue();
    N00 = N00.getOperand(0);
    LookPassAnd0 = true;
  }

  SDValue N10 = N1->getOperand(0);
  if (!LookPassAnd1 && N10.getOpcode() == ISD::AND) {
    if (!N10.getNode()->hasOneUse())
      return SDValue();
    ConstantSDNode *N101C = dyn_cast<ConstantSDNode>(N10.getOperand(1));
    if (!N101C || N101C->getZExtValue() != 0xFF00)
      return SDValue();
    N10 = N10.getOperand(0);
    LookPassAnd1 = true;
  }

  if (N00 != N10)
    return SDValue();

  // Make sure everything beyond the low halfword gets set to zero since the SRL
  // 16 will clear the top bits.
  unsigned OpSizeInBits = VT.getSizeInBits();
  if (DemandHighBits && OpSizeInBits > 16) {
    // If the left-shift isn't masked out then the only way this is a bswap is
    // if all bits beyond the low 8 are 0. In that case the entire pattern
    // reduces to a left shift anyway: leave it for other parts of the combiner.
    if (!LookPassAnd0)
      return SDValue();

    // However, if the right shift isn't masked out then it might be because
    // it's not needed. See if we can spot that too.
    if (!LookPassAnd1 &&
        !DAG.MaskedValueIsZero(
            N10, APInt::getHighBitsSet(OpSizeInBits, OpSizeInBits - 16)))
      return SDValue();
  }

  SDValue Res = DAG.getNode(ISD::BSWAP, SDLoc(N), VT, N00);
  if (OpSizeInBits > 16) {
    SDLoc DL(N);
    Res = DAG.getNode(ISD::SRL, DL, VT, Res,
                      DAG.getConstant(OpSizeInBits - 16, DL,
                                      getShiftAmountTy(VT)));
  }
  return Res;
}

/// Return true if the specified node is an element that makes up a 32-bit
/// packed halfword byteswap.
/// ((x & 0x000000ff) << 8) |
/// ((x & 0x0000ff00) >> 8) |
/// ((x & 0x00ff0000) << 8) |
/// ((x & 0xff000000) >> 8)
static bool isBSwapHWordElement(SDValue N, MutableArrayRef<SDNode *> Parts) {
  if (!N.getNode()->hasOneUse())
    return false;

  unsigned Opc = N.getOpcode();
  if (Opc != ISD::AND && Opc != ISD::SHL && Opc != ISD::SRL)
    return false;

  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N.getOperand(1));
  if (!N1C)
    return false;

  unsigned Num;
  switch (N1C->getZExtValue()) {
  default:
    return false;
  case 0xFF:       Num = 0; break;
  case 0xFF00:     Num = 1; break;
  case 0xFF0000:   Num = 2; break;
  case 0xFF000000: Num = 3; break;
  }

  // Look for (x & 0xff) << 8 as well as ((x << 8) & 0xff00).
  SDValue N0 = N.getOperand(0);
  if (Opc == ISD::AND) {
    if (Num == 0 || Num == 2) {
      // (x >> 8) & 0xff
      // (x >> 8) & 0xff0000
      if (N0.getOpcode() != ISD::SRL)
        return false;
      ConstantSDNode *C = dyn_cast<ConstantSDNode>(N0.getOperand(1));
      if (!C || C->getZExtValue() != 8)
        return false;
    } else {
      // (x << 8) & 0xff00
      // (x << 8) & 0xff000000
      if (N0.getOpcode() != ISD::SHL)
        return false;
      ConstantSDNode *C = dyn_cast<ConstantSDNode>(N0.getOperand(1));
      if (!C || C->getZExtValue() != 8)
        return false;
    }
  } else if (Opc == ISD::SHL) {
    // (x & 0xff) << 8
    // (x & 0xff0000) << 8
    if (Num != 0 && Num != 2)
      return false;
    ConstantSDNode *C = dyn_cast<ConstantSDNode>(N.getOperand(1));
    if (!C || C->getZExtValue() != 8)
      return false;
  } else { // Opc == ISD::SRL
    // (x & 0xff00) >> 8
    // (x & 0xff000000) >> 8
    if (Num != 1 && Num != 3)
      return false;
    ConstantSDNode *C = dyn_cast<ConstantSDNode>(N.getOperand(1));
    if (!C || C->getZExtValue() != 8)
      return false;
  }

  if (Parts[Num])
    return false;

  Parts[Num] = N0.getOperand(0).getNode();
  return true;
}

/// Match a 32-bit packed halfword bswap. That is
/// ((x & 0x000000ff) << 8) |
/// ((x & 0x0000ff00) >> 8) |
/// ((x & 0x00ff0000) << 8) |
/// ((x & 0xff000000) >> 8)
/// => (rotl (bswap x), 16)
SDValue DAGCombiner::MatchBSwapHWord(SDNode *N, SDValue N0, SDValue N1) {
  if (!LegalOperations)
    return SDValue();

  EVT VT = N->getValueType(0);
  if (VT != MVT::i32)
    return SDValue();
  if (!TLI.isOperationLegal(ISD::BSWAP, VT))
    return SDValue();

  // Look for either
  // (or (or (and), (and)), (or (and), (and)))
  // (or (or (or (and), (and)), (and)), (and))
  if (N0.getOpcode() != ISD::OR)
    return SDValue();
  SDValue N00 = N0.getOperand(0);
  SDValue N01 = N0.getOperand(1);
  SDNode *Parts[4] = {};

  if (N1.getOpcode() == ISD::OR &&
      N00.getNumOperands() == 2 && N01.getNumOperands() == 2) {
    // (or (or (and), (and)), (or (and), (and)))
    SDValue N000 = N00.getOperand(0);
    if (!isBSwapHWordElement(N000, Parts))
      return SDValue();

    SDValue N001 = N00.getOperand(1);
    if (!isBSwapHWordElement(N001, Parts))
      return SDValue();
    SDValue N010 = N01.getOperand(0);
    if (!isBSwapHWordElement(N010, Parts))
      return SDValue();
    SDValue N011 = N01.getOperand(1);
    if (!isBSwapHWordElement(N011, Parts))
      return SDValue();
  } else {
    // (or (or (or (and), (and)), (and)), (and))
    if (!isBSwapHWordElement(N1, Parts))
      return SDValue();
    if (!isBSwapHWordElement(N01, Parts))
      return SDValue();
    if (N00.getOpcode() != ISD::OR)
      return SDValue();
    SDValue N000 = N00.getOperand(0);
    if (!isBSwapHWordElement(N000, Parts))
      return SDValue();
    SDValue N001 = N00.getOperand(1);
    if (!isBSwapHWordElement(N001, Parts))
      return SDValue();
  }

  // Make sure the parts are all coming from the same node.
  if (Parts[0] != Parts[1] || Parts[0] != Parts[2] || Parts[0] != Parts[3])
    return SDValue();

  SDLoc DL(N);
  SDValue BSwap = DAG.getNode(ISD::BSWAP, DL, VT,
                              SDValue(Parts[0], 0));

  // Result of the bswap should be rotated by 16. If it's not legal, then
  // do  (x << 16) | (x >> 16).
  SDValue ShAmt = DAG.getConstant(16, DL, getShiftAmountTy(VT));
  if (TLI.isOperationLegalOrCustom(ISD::ROTL, VT))
    return DAG.getNode(ISD::ROTL, DL, VT, BSwap, ShAmt);
  if (TLI.isOperationLegalOrCustom(ISD::ROTR, VT))
    return DAG.getNode(ISD::ROTR, DL, VT, BSwap, ShAmt);
  return DAG.getNode(ISD::OR, DL, VT,
                     DAG.getNode(ISD::SHL, DL, VT, BSwap, ShAmt),
                     DAG.getNode(ISD::SRL, DL, VT, BSwap, ShAmt));
}

/// This contains all DAGCombine rules which reduce two values combined by
/// an Or operation to a single value \see visitANDLike().
SDValue DAGCombiner::visitORLike(SDValue N0, SDValue N1, SDNode *LocReference) {
  EVT VT = N1.getValueType();
  // fold (or x, undef) -> -1
  if (!LegalOperations &&
      (N0.getOpcode() == ISD::UNDEF || N1.getOpcode() == ISD::UNDEF)) {
    EVT EltVT = VT.isVector() ? VT.getVectorElementType() : VT;
    return DAG.getConstant(APInt::getAllOnesValue(EltVT.getSizeInBits()),
                           SDLoc(LocReference), VT);
  }
  // fold (or (setcc x), (setcc y)) -> (setcc (or x, y))
  SDValue LL, LR, RL, RR, CC0, CC1;
  if (isSetCCEquivalent(N0, LL, LR, CC0) && isSetCCEquivalent(N1, RL, RR, CC1)){
    ISD::CondCode Op0 = cast<CondCodeSDNode>(CC0)->get();
    ISD::CondCode Op1 = cast<CondCodeSDNode>(CC1)->get();

    if (LR == RR && Op0 == Op1 && LL.getValueType().isInteger()) {
      // fold (or (setne X, 0), (setne Y, 0)) -> (setne (or X, Y), 0)
      // fold (or (setlt X, 0), (setlt Y, 0)) -> (setne (or X, Y), 0)
      if (isNullConstant(LR) && (Op1 == ISD::SETNE || Op1 == ISD::SETLT)) {
        SDValue ORNode = DAG.getNode(ISD::OR, SDLoc(LR),
                                     LR.getValueType(), LL, RL);
        AddToWorklist(ORNode.getNode());
        return DAG.getSetCC(SDLoc(LocReference), VT, ORNode, LR, Op1);
      }
      // fold (or (setne X, -1), (setne Y, -1)) -> (setne (and X, Y), -1)
      // fold (or (setgt X, -1), (setgt Y  -1)) -> (setgt (and X, Y), -1)
      if (isAllOnesConstant(LR) && (Op1 == ISD::SETNE || Op1 == ISD::SETGT)) {
        SDValue ANDNode = DAG.getNode(ISD::AND, SDLoc(LR),
                                      LR.getValueType(), LL, RL);
        AddToWorklist(ANDNode.getNode());
        return DAG.getSetCC(SDLoc(LocReference), VT, ANDNode, LR, Op1);
      }
    }
    // canonicalize equivalent to ll == rl
    if (LL == RR && LR == RL) {
      Op1 = ISD::getSetCCSwappedOperands(Op1);
      std::swap(RL, RR);
    }
    if (LL == RL && LR == RR) {
      bool isInteger = LL.getValueType().isInteger();
      ISD::CondCode Result = ISD::getSetCCOrOperation(Op0, Op1, isInteger);
      if (Result != ISD::SETCC_INVALID &&
          (!LegalOperations ||
           (TLI.isCondCodeLegal(Result, LL.getSimpleValueType()) &&
            TLI.isOperationLegal(ISD::SETCC,
              getSetCCResultType(N0.getValueType())))))
        return DAG.getSetCC(SDLoc(LocReference), N0.getValueType(),
                            LL, LR, Result);
    }
  }

  // (or (and X, C1), (and Y, C2))  -> (and (or X, Y), C3) if possible.
  if (N0.getOpcode() == ISD::AND && N1.getOpcode() == ISD::AND &&
      // Don't increase # computations.
      (N0.getNode()->hasOneUse() || N1.getNode()->hasOneUse())) {
    // We can only do this xform if we know that bits from X that are set in C2
    // but not in C1 are already zero.  Likewise for Y.
    if (const ConstantSDNode *N0O1C =
        getAsNonOpaqueConstant(N0.getOperand(1))) {
      if (const ConstantSDNode *N1O1C =
          getAsNonOpaqueConstant(N1.getOperand(1))) {
        // We can only do this xform if we know that bits from X that are set in
        // C2 but not in C1 are already zero.  Likewise for Y.
        const APInt &LHSMask = N0O1C->getAPIntValue();
        const APInt &RHSMask = N1O1C->getAPIntValue();

        if (DAG.MaskedValueIsZero(N0.getOperand(0), RHSMask&~LHSMask) &&
            DAG.MaskedValueIsZero(N1.getOperand(0), LHSMask&~RHSMask)) {
          SDValue X = DAG.getNode(ISD::OR, SDLoc(N0), VT,
                                  N0.getOperand(0), N1.getOperand(0));
          SDLoc DL(LocReference);
          return DAG.getNode(ISD::AND, DL, VT, X,
                             DAG.getConstant(LHSMask | RHSMask, DL, VT));
        }
      }
    }
  }

  // (or (and X, M), (and X, N)) -> (and X, (or M, N))
  if (N0.getOpcode() == ISD::AND &&
      N1.getOpcode() == ISD::AND &&
      N0.getOperand(0) == N1.getOperand(0) &&
      // Don't increase # computations.
      (N0.getNode()->hasOneUse() || N1.getNode()->hasOneUse())) {
    SDValue X = DAG.getNode(ISD::OR, SDLoc(N0), VT,
                            N0.getOperand(1), N1.getOperand(1));
    return DAG.getNode(ISD::AND, SDLoc(LocReference), VT, N0.getOperand(0), X);
  }

  return SDValue();
}

SDValue DAGCombiner::visitOR(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N1.getValueType();

  // fold vector ops
  if (VT.isVector()) {
    if (SDValue FoldedVOp = SimplifyVBinOp(N))
      return FoldedVOp;

    // fold (or x, 0) -> x, vector edition
    if (ISD::isBuildVectorAllZeros(N0.getNode()))
      return N1;
    if (ISD::isBuildVectorAllZeros(N1.getNode()))
      return N0;

    // fold (or x, -1) -> -1, vector edition
    if (ISD::isBuildVectorAllOnes(N0.getNode()))
      // do not return N0, because undef node may exist in N0
      return DAG.getConstant(
          APInt::getAllOnesValue(
              N0.getValueType().getScalarType().getSizeInBits()),
          SDLoc(N), N0.getValueType());
    if (ISD::isBuildVectorAllOnes(N1.getNode()))
      // do not return N1, because undef node may exist in N1
      return DAG.getConstant(
          APInt::getAllOnesValue(
              N1.getValueType().getScalarType().getSizeInBits()),
          SDLoc(N), N1.getValueType());

    // fold (or (shuf A, V_0, MA), (shuf B, V_0, MB)) -> (shuf A, B, Mask1)
    // fold (or (shuf A, V_0, MA), (shuf B, V_0, MB)) -> (shuf B, A, Mask2)
    // Do this only if the resulting shuffle is legal.
    if (isa<ShuffleVectorSDNode>(N0) &&
        isa<ShuffleVectorSDNode>(N1) &&
        // Avoid folding a node with illegal type.
        TLI.isTypeLegal(VT) &&
        N0->getOperand(1) == N1->getOperand(1) &&
        ISD::isBuildVectorAllZeros(N0.getOperand(1).getNode())) {
      bool CanFold = true;
      unsigned NumElts = VT.getVectorNumElements();
      const ShuffleVectorSDNode *SV0 = cast<ShuffleVectorSDNode>(N0);
      const ShuffleVectorSDNode *SV1 = cast<ShuffleVectorSDNode>(N1);
      // We construct two shuffle masks:
      // - Mask1 is a shuffle mask for a shuffle with N0 as the first operand
      // and N1 as the second operand.
      // - Mask2 is a shuffle mask for a shuffle with N1 as the first operand
      // and N0 as the second operand.
      // We do this because OR is commutable and therefore there might be
      // two ways to fold this node into a shuffle.
      SmallVector<int,4> Mask1;
      SmallVector<int,4> Mask2;

      for (unsigned i = 0; i != NumElts && CanFold; ++i) {
        int M0 = SV0->getMaskElt(i);
        int M1 = SV1->getMaskElt(i);

        // Both shuffle indexes are undef. Propagate Undef.
        if (M0 < 0 && M1 < 0) {
          Mask1.push_back(M0);
          Mask2.push_back(M0);
          continue;
        }

        if (M0 < 0 || M1 < 0 ||
            (M0 < (int)NumElts && M1 < (int)NumElts) ||
            (M0 >= (int)NumElts && M1 >= (int)NumElts)) {
          CanFold = false;
          break;
        }

        Mask1.push_back(M0 < (int)NumElts ? M0 : M1 + NumElts);
        Mask2.push_back(M1 < (int)NumElts ? M1 : M0 + NumElts);
      }

      if (CanFold) {
        // Fold this sequence only if the resulting shuffle is 'legal'.
        if (TLI.isShuffleMaskLegal(Mask1, VT))
          return DAG.getVectorShuffle(VT, SDLoc(N), N0->getOperand(0),
                                      N1->getOperand(0), &Mask1[0]);
        if (TLI.isShuffleMaskLegal(Mask2, VT))
          return DAG.getVectorShuffle(VT, SDLoc(N), N1->getOperand(0),
                                      N0->getOperand(0), &Mask2[0]);
      }
    }
  }

  // fold (or c1, c2) -> c1|c2
  ConstantSDNode *N0C = getAsNonOpaqueConstant(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  if (N0C && N1C && !N1C->isOpaque())
    return DAG.FoldConstantArithmetic(ISD::OR, SDLoc(N), VT, N0C, N1C);
  // canonicalize constant to RHS
  if (isConstantIntBuildVectorOrConstantInt(N0) &&
     !isConstantIntBuildVectorOrConstantInt(N1))
    return DAG.getNode(ISD::OR, SDLoc(N), VT, N1, N0);
  // fold (or x, 0) -> x
  if (isNullConstant(N1))
    return N0;
  // fold (or x, -1) -> -1
  if (isAllOnesConstant(N1))
    return N1;
  // fold (or x, c) -> c iff (x & ~c) == 0
  if (N1C && DAG.MaskedValueIsZero(N0, ~N1C->getAPIntValue()))
    return N1;

  if (SDValue Combined = visitORLike(N0, N1, N))
    return Combined;

  // Recognize halfword bswaps as (bswap + rotl 16) or (bswap + shl 16)
  SDValue BSwap = MatchBSwapHWord(N, N0, N1);
  if (BSwap.getNode())
    return BSwap;
  BSwap = MatchBSwapHWordLow(N, N0, N1);
  if (BSwap.getNode())
    return BSwap;

  // reassociate or
  if (SDValue ROR = ReassociateOps(ISD::OR, SDLoc(N), N0, N1))
    return ROR;
  // Canonicalize (or (and X, c1), c2) -> (and (or X, c2), c1|c2)
  // iff (c1 & c2) == 0.
  if (N1C && N0.getOpcode() == ISD::AND && N0.getNode()->hasOneUse() &&
             isa<ConstantSDNode>(N0.getOperand(1))) {
    ConstantSDNode *C1 = cast<ConstantSDNode>(N0.getOperand(1));
    if ((C1->getAPIntValue() & N1C->getAPIntValue()) != 0) {
      if (SDValue COR = DAG.FoldConstantArithmetic(ISD::OR, SDLoc(N1), VT,
                                                   N1C, C1))
        return DAG.getNode(
            ISD::AND, SDLoc(N), VT,
            DAG.getNode(ISD::OR, SDLoc(N0), VT, N0.getOperand(0), N1), COR);
      return SDValue();
    }
  }
  // Simplify: (or (op x...), (op y...))  -> (op (or x, y))
  if (N0.getOpcode() == N1.getOpcode()) {
    SDValue Tmp = SimplifyBinOpWithSameOpcodeHands(N);
    if (Tmp.getNode()) return Tmp;
  }

  // See if this is some rotate idiom.
  if (SDNode *Rot = MatchRotate(N0, N1, SDLoc(N)))
    return SDValue(Rot, 0);

  // Simplify the operands using demanded-bits information.
  if (!VT.isVector() &&
      SimplifyDemandedBits(SDValue(N, 0)))
    return SDValue(N, 0);

  return SDValue();
}

/// Match "(X shl/srl V1) & V2" where V2 may not be present.
static bool MatchRotateHalf(SDValue Op, SDValue &Shift, SDValue &Mask) {
  if (Op.getOpcode() == ISD::AND) {
    if (isa<ConstantSDNode>(Op.getOperand(1))) {
      Mask = Op.getOperand(1);
      Op = Op.getOperand(0);
    } else {
      return false;
    }
  }

  if (Op.getOpcode() == ISD::SRL || Op.getOpcode() == ISD::SHL) {
    Shift = Op;
    return true;
  }

  return false;
}

// Return true if we can prove that, whenever Neg and Pos are both in the
// range [0, OpSize), Neg == (Pos == 0 ? 0 : OpSize - Pos).  This means that
// for two opposing shifts shift1 and shift2 and a value X with OpBits bits:
//
//     (or (shift1 X, Neg), (shift2 X, Pos))
//
// reduces to a rotate in direction shift2 by Pos or (equivalently) a rotate
// in direction shift1 by Neg.  The range [0, OpSize) means that we only need
// to consider shift amounts with defined behavior.
static bool matchRotateSub(SDValue Pos, SDValue Neg, unsigned OpSize) {
  // If OpSize is a power of 2 then:
  //
  //  (a) (Pos == 0 ? 0 : OpSize - Pos) == (OpSize - Pos) & (OpSize - 1)
  //  (b) Neg == Neg & (OpSize - 1) whenever Neg is in [0, OpSize).
  //
  // So if OpSize is a power of 2 and Neg is (and Neg', OpSize-1), we check
  // for the stronger condition:
  //
  //     Neg & (OpSize - 1) == (OpSize - Pos) & (OpSize - 1)    [A]
  //
  // for all Neg and Pos.  Since Neg & (OpSize - 1) == Neg' & (OpSize - 1)
  // we can just replace Neg with Neg' for the rest of the function.
  //
  // In other cases we check for the even stronger condition:
  //
  //     Neg == OpSize - Pos                                    [B]
  //
  // for all Neg and Pos.  Note that the (or ...) then invokes undefined
  // behavior if Pos == 0 (and consequently Neg == OpSize).
  //
  // We could actually use [A] whenever OpSize is a power of 2, but the
  // only extra cases that it would match are those uninteresting ones
  // where Neg and Pos are never in range at the same time.  E.g. for
  // OpSize == 32, using [A] would allow a Neg of the form (sub 64, Pos)
  // as well as (sub 32, Pos), but:
  //
  //     (or (shift1 X, (sub 64, Pos)), (shift2 X, Pos))
  //
  // always invokes undefined behavior for 32-bit X.
  //
  // Below, Mask == OpSize - 1 when using [A] and is all-ones otherwise.
  unsigned MaskLoBits = 0;
  if (Neg.getOpcode() == ISD::AND &&
      isPowerOf2_64(OpSize) &&
      Neg.getOperand(1).getOpcode() == ISD::Constant &&
      cast<ConstantSDNode>(Neg.getOperand(1))->getAPIntValue() == OpSize - 1) {
    Neg = Neg.getOperand(0);
    MaskLoBits = Log2_64(OpSize);
  }

  // Check whether Neg has the form (sub NegC, NegOp1) for some NegC and NegOp1.
  if (Neg.getOpcode() != ISD::SUB)
    return 0;
  ConstantSDNode *NegC = dyn_cast<ConstantSDNode>(Neg.getOperand(0));
  if (!NegC)
    return 0;
  SDValue NegOp1 = Neg.getOperand(1);

  // On the RHS of [A], if Pos is Pos' & (OpSize - 1), just replace Pos with
  // Pos'.  The truncation is redundant for the purpose of the equality.
  if (MaskLoBits &&
      Pos.getOpcode() == ISD::AND &&
      Pos.getOperand(1).getOpcode() == ISD::Constant &&
      cast<ConstantSDNode>(Pos.getOperand(1))->getAPIntValue() == OpSize - 1)
    Pos = Pos.getOperand(0);

  // The condition we need is now:
  //
  //     (NegC - NegOp1) & Mask == (OpSize - Pos) & Mask
  //
  // If NegOp1 == Pos then we need:
  //
  //              OpSize & Mask == NegC & Mask
  //
  // (because "x & Mask" is a truncation and distributes through subtraction).
  APInt Width;
  if (Pos == NegOp1)
    Width = NegC->getAPIntValue();
  // Check for cases where Pos has the form (add NegOp1, PosC) for some PosC.
  // Then the condition we want to prove becomes:
  //
  //     (NegC - NegOp1) & Mask == (OpSize - (NegOp1 + PosC)) & Mask
  //
  // which, again because "x & Mask" is a truncation, becomes:
  //
  //                NegC & Mask == (OpSize - PosC) & Mask
  //              OpSize & Mask == (NegC + PosC) & Mask
  else if (Pos.getOpcode() == ISD::ADD &&
           Pos.getOperand(0) == NegOp1 &&
           Pos.getOperand(1).getOpcode() == ISD::Constant)
    Width = (cast<ConstantSDNode>(Pos.getOperand(1))->getAPIntValue() +
             NegC->getAPIntValue());
  else
    return false;

  // Now we just need to check that OpSize & Mask == Width & Mask.
  if (MaskLoBits)
    // Opsize & Mask is 0 since Mask is Opsize - 1.
    return Width.getLoBits(MaskLoBits) == 0;
  return Width == OpSize;
}

// A subroutine of MatchRotate used once we have found an OR of two opposite
// shifts of Shifted.  If Neg == <operand size> - Pos then the OR reduces
// to both (PosOpcode Shifted, Pos) and (NegOpcode Shifted, Neg), with the
// former being preferred if supported.  InnerPos and InnerNeg are Pos and
// Neg with outer conversions stripped away.
SDNode *DAGCombiner::MatchRotatePosNeg(SDValue Shifted, SDValue Pos,
                                       SDValue Neg, SDValue InnerPos,
                                       SDValue InnerNeg, unsigned PosOpcode,
                                       unsigned NegOpcode, SDLoc DL) {
  // fold (or (shl x, (*ext y)),
  //          (srl x, (*ext (sub 32, y)))) ->
  //   (rotl x, y) or (rotr x, (sub 32, y))
  //
  // fold (or (shl x, (*ext (sub 32, y))),
  //          (srl x, (*ext y))) ->
  //   (rotr x, y) or (rotl x, (sub 32, y))
  EVT VT = Shifted.getValueType();
  if (matchRotateSub(InnerPos, InnerNeg, VT.getSizeInBits())) {
    bool HasPos = TLI.isOperationLegalOrCustom(PosOpcode, VT);
    return DAG.getNode(HasPos ? PosOpcode : NegOpcode, DL, VT, Shifted,
                       HasPos ? Pos : Neg).getNode();
  }

  return nullptr;
}

// MatchRotate - Handle an 'or' of two operands.  If this is one of the many
// idioms for rotate, and if the target supports rotation instructions, generate
// a rot[lr].
SDNode *DAGCombiner::MatchRotate(SDValue LHS, SDValue RHS, SDLoc DL) {
  // Must be a legal type.  Expanded 'n promoted things won't work with rotates.
  EVT VT = LHS.getValueType();
  if (!TLI.isTypeLegal(VT)) return nullptr;

  // The target must have at least one rotate flavor.
  bool HasROTL = TLI.isOperationLegalOrCustom(ISD::ROTL, VT);
  bool HasROTR = TLI.isOperationLegalOrCustom(ISD::ROTR, VT);
  if (!HasROTL && !HasROTR) return nullptr;

  // Match "(X shl/srl V1) & V2" where V2 may not be present.
  SDValue LHSShift;   // The shift.
  SDValue LHSMask;    // AND value if any.
  if (!MatchRotateHalf(LHS, LHSShift, LHSMask))
    return nullptr; // Not part of a rotate.

  SDValue RHSShift;   // The shift.
  SDValue RHSMask;    // AND value if any.
  if (!MatchRotateHalf(RHS, RHSShift, RHSMask))
    return nullptr; // Not part of a rotate.

  if (LHSShift.getOperand(0) != RHSShift.getOperand(0))
    return nullptr;   // Not shifting the same value.

  if (LHSShift.getOpcode() == RHSShift.getOpcode())
    return nullptr;   // Shifts must disagree.

  // Canonicalize shl to left side in a shl/srl pair.
  if (RHSShift.getOpcode() == ISD::SHL) {
    std::swap(LHS, RHS);
    std::swap(LHSShift, RHSShift);
    std::swap(LHSMask , RHSMask );
  }

  unsigned OpSizeInBits = VT.getSizeInBits();
  SDValue LHSShiftArg = LHSShift.getOperand(0);
  SDValue LHSShiftAmt = LHSShift.getOperand(1);
  SDValue RHSShiftArg = RHSShift.getOperand(0);
  SDValue RHSShiftAmt = RHSShift.getOperand(1);

  // fold (or (shl x, C1), (srl x, C2)) -> (rotl x, C1)
  // fold (or (shl x, C1), (srl x, C2)) -> (rotr x, C2)
  if (LHSShiftAmt.getOpcode() == ISD::Constant &&
      RHSShiftAmt.getOpcode() == ISD::Constant) {
    uint64_t LShVal = cast<ConstantSDNode>(LHSShiftAmt)->getZExtValue();
    uint64_t RShVal = cast<ConstantSDNode>(RHSShiftAmt)->getZExtValue();
    if ((LShVal + RShVal) != OpSizeInBits)
      return nullptr;

    SDValue Rot = DAG.getNode(HasROTL ? ISD::ROTL : ISD::ROTR, DL, VT,
                              LHSShiftArg, HasROTL ? LHSShiftAmt : RHSShiftAmt);

    // If there is an AND of either shifted operand, apply it to the result.
    if (LHSMask.getNode() || RHSMask.getNode()) {
      APInt Mask = APInt::getAllOnesValue(OpSizeInBits);

      if (LHSMask.getNode()) {
        APInt RHSBits = APInt::getLowBitsSet(OpSizeInBits, LShVal);
        Mask &= cast<ConstantSDNode>(LHSMask)->getAPIntValue() | RHSBits;
      }
      if (RHSMask.getNode()) {
        APInt LHSBits = APInt::getHighBitsSet(OpSizeInBits, RShVal);
        Mask &= cast<ConstantSDNode>(RHSMask)->getAPIntValue() | LHSBits;
      }

      Rot = DAG.getNode(ISD::AND, DL, VT, Rot, DAG.getConstant(Mask, DL, VT));
    }

    return Rot.getNode();
  }

  // If there is a mask here, and we have a variable shift, we can't be sure
  // that we're masking out the right stuff.
  if (LHSMask.getNode() || RHSMask.getNode())
    return nullptr;

  // If the shift amount is sign/zext/any-extended just peel it off.
  SDValue LExtOp0 = LHSShiftAmt;
  SDValue RExtOp0 = RHSShiftAmt;
  if ((LHSShiftAmt.getOpcode() == ISD::SIGN_EXTEND ||
       LHSShiftAmt.getOpcode() == ISD::ZERO_EXTEND ||
       LHSShiftAmt.getOpcode() == ISD::ANY_EXTEND ||
       LHSShiftAmt.getOpcode() == ISD::TRUNCATE) &&
      (RHSShiftAmt.getOpcode() == ISD::SIGN_EXTEND ||
       RHSShiftAmt.getOpcode() == ISD::ZERO_EXTEND ||
       RHSShiftAmt.getOpcode() == ISD::ANY_EXTEND ||
       RHSShiftAmt.getOpcode() == ISD::TRUNCATE)) {
    LExtOp0 = LHSShiftAmt.getOperand(0);
    RExtOp0 = RHSShiftAmt.getOperand(0);
  }

  SDNode *TryL = MatchRotatePosNeg(LHSShiftArg, LHSShiftAmt, RHSShiftAmt,
                                   LExtOp0, RExtOp0, ISD::ROTL, ISD::ROTR, DL);
  if (TryL)
    return TryL;

  SDNode *TryR = MatchRotatePosNeg(RHSShiftArg, RHSShiftAmt, LHSShiftAmt,
                                   RExtOp0, LExtOp0, ISD::ROTR, ISD::ROTL, DL);
  if (TryR)
    return TryR;

  return nullptr;
}

SDValue DAGCombiner::visitXOR(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N0.getValueType();

  // fold vector ops
  if (VT.isVector()) {
    if (SDValue FoldedVOp = SimplifyVBinOp(N))
      return FoldedVOp;

    // fold (xor x, 0) -> x, vector edition
    if (ISD::isBuildVectorAllZeros(N0.getNode()))
      return N1;
    if (ISD::isBuildVectorAllZeros(N1.getNode()))
      return N0;
  }

  // fold (xor undef, undef) -> 0. This is a common idiom (misuse).
  if (N0.getOpcode() == ISD::UNDEF && N1.getOpcode() == ISD::UNDEF)
    return DAG.getConstant(0, SDLoc(N), VT);
  // fold (xor x, undef) -> undef
  if (N0.getOpcode() == ISD::UNDEF)
    return N0;
  if (N1.getOpcode() == ISD::UNDEF)
    return N1;
  // fold (xor c1, c2) -> c1^c2
  ConstantSDNode *N0C = getAsNonOpaqueConstant(N0);
  ConstantSDNode *N1C = getAsNonOpaqueConstant(N1);
  if (N0C && N1C)
    return DAG.FoldConstantArithmetic(ISD::XOR, SDLoc(N), VT, N0C, N1C);
  // canonicalize constant to RHS
  if (isConstantIntBuildVectorOrConstantInt(N0) &&
     !isConstantIntBuildVectorOrConstantInt(N1))
    return DAG.getNode(ISD::XOR, SDLoc(N), VT, N1, N0);
  // fold (xor x, 0) -> x
  if (isNullConstant(N1))
    return N0;
  // reassociate xor
  if (SDValue RXOR = ReassociateOps(ISD::XOR, SDLoc(N), N0, N1))
    return RXOR;

  // fold !(x cc y) -> (x !cc y)
  SDValue LHS, RHS, CC;
  if (TLI.isConstTrueVal(N1.getNode()) && isSetCCEquivalent(N0, LHS, RHS, CC)) {
    bool isInt = LHS.getValueType().isInteger();
    ISD::CondCode NotCC = ISD::getSetCCInverse(cast<CondCodeSDNode>(CC)->get(),
                                               isInt);

    if (!LegalOperations ||
        TLI.isCondCodeLegal(NotCC, LHS.getSimpleValueType())) {
      switch (N0.getOpcode()) {
      default:
        llvm_unreachable("Unhandled SetCC Equivalent!");
      case ISD::SETCC:
        return DAG.getSetCC(SDLoc(N), VT, LHS, RHS, NotCC);
      case ISD::SELECT_CC:
        return DAG.getSelectCC(SDLoc(N), LHS, RHS, N0.getOperand(2),
                               N0.getOperand(3), NotCC);
      }
    }
  }

  // fold (not (zext (setcc x, y))) -> (zext (not (setcc x, y)))
  if (isOneConstant(N1) && N0.getOpcode() == ISD::ZERO_EXTEND &&
      N0.getNode()->hasOneUse() &&
      isSetCCEquivalent(N0.getOperand(0), LHS, RHS, CC)){
    SDValue V = N0.getOperand(0);
    SDLoc DL(N0);
    V = DAG.getNode(ISD::XOR, DL, V.getValueType(), V,
                    DAG.getConstant(1, DL, V.getValueType()));
    AddToWorklist(V.getNode());
    return DAG.getNode(ISD::ZERO_EXTEND, SDLoc(N), VT, V);
  }

  // fold (not (or x, y)) -> (and (not x), (not y)) iff x or y are setcc
  if (isOneConstant(N1) && VT == MVT::i1 &&
      (N0.getOpcode() == ISD::OR || N0.getOpcode() == ISD::AND)) {
    SDValue LHS = N0.getOperand(0), RHS = N0.getOperand(1);
    if (isOneUseSetCC(RHS) || isOneUseSetCC(LHS)) {
      unsigned NewOpcode = N0.getOpcode() == ISD::AND ? ISD::OR : ISD::AND;
      LHS = DAG.getNode(ISD::XOR, SDLoc(LHS), VT, LHS, N1); // LHS = ~LHS
      RHS = DAG.getNode(ISD::XOR, SDLoc(RHS), VT, RHS, N1); // RHS = ~RHS
      AddToWorklist(LHS.getNode()); AddToWorklist(RHS.getNode());
      return DAG.getNode(NewOpcode, SDLoc(N), VT, LHS, RHS);
    }
  }
  // fold (not (or x, y)) -> (and (not x), (not y)) iff x or y are constants
  if (isAllOnesConstant(N1) &&
      (N0.getOpcode() == ISD::OR || N0.getOpcode() == ISD::AND)) {
    SDValue LHS = N0.getOperand(0), RHS = N0.getOperand(1);
    if (isa<ConstantSDNode>(RHS) || isa<ConstantSDNode>(LHS)) {
      unsigned NewOpcode = N0.getOpcode() == ISD::AND ? ISD::OR : ISD::AND;
      LHS = DAG.getNode(ISD::XOR, SDLoc(LHS), VT, LHS, N1); // LHS = ~LHS
      RHS = DAG.getNode(ISD::XOR, SDLoc(RHS), VT, RHS, N1); // RHS = ~RHS
      AddToWorklist(LHS.getNode()); AddToWorklist(RHS.getNode());
      return DAG.getNode(NewOpcode, SDLoc(N), VT, LHS, RHS);
    }
  }
  // fold (xor (and x, y), y) -> (and (not x), y)
  if (N0.getOpcode() == ISD::AND && N0.getNode()->hasOneUse() &&
      N0->getOperand(1) == N1) {
    SDValue X = N0->getOperand(0);
    SDValue NotX = DAG.getNOT(SDLoc(X), X, VT);
    AddToWorklist(NotX.getNode());
    return DAG.getNode(ISD::AND, SDLoc(N), VT, NotX, N1);
  }
  // fold (xor (xor x, c1), c2) -> (xor x, (xor c1, c2))
  if (N1C && N0.getOpcode() == ISD::XOR) {
    if (const ConstantSDNode *N00C = getAsNonOpaqueConstant(N0.getOperand(0))) {
      SDLoc DL(N);
      return DAG.getNode(ISD::XOR, DL, VT, N0.getOperand(1),
                         DAG.getConstant(N1C->getAPIntValue() ^
                                         N00C->getAPIntValue(), DL, VT));
    }
    if (const ConstantSDNode *N01C = getAsNonOpaqueConstant(N0.getOperand(1))) {
      SDLoc DL(N);
      return DAG.getNode(ISD::XOR, DL, VT, N0.getOperand(0),
                         DAG.getConstant(N1C->getAPIntValue() ^
                                         N01C->getAPIntValue(), DL, VT));
    }
  }
  // fold (xor x, x) -> 0
  if (N0 == N1)
    return tryFoldToZero(SDLoc(N), TLI, VT, DAG, LegalOperations, LegalTypes);

  // fold (xor (shl 1, x), -1) -> (rotl ~1, x)
  // Here is a concrete example of this equivalence:
  // i16   x ==  14
  // i16 shl ==   1 << 14  == 16384 == 0b0100000000000000
  // i16 xor == ~(1 << 14) == 49151 == 0b1011111111111111
  //
  // =>
  //
  // i16     ~1      == 0b1111111111111110
  // i16 rol(~1, 14) == 0b1011111111111111
  //
  // Some additional tips to help conceptualize this transform:
  // - Try to see the operation as placing a single zero in a value of all ones.
  // - There exists no value for x which would allow the result to contain zero.
  // - Values of x larger than the bitwidth are undefined and do not require a
  //   consistent result.
  // - Pushing the zero left requires shifting one bits in from the right.
  // A rotate left of ~1 is a nice way of achieving the desired result.
  if (TLI.isOperationLegalOrCustom(ISD::ROTL, VT) && N0.getOpcode() == ISD::SHL
      && isAllOnesConstant(N1) && isOneConstant(N0.getOperand(0))) {
    SDLoc DL(N);
    return DAG.getNode(ISD::ROTL, DL, VT, DAG.getConstant(~1, DL, VT),
                       N0.getOperand(1));
  }

  // Simplify: xor (op x...), (op y...)  -> (op (xor x, y))
  if (N0.getOpcode() == N1.getOpcode()) {
    SDValue Tmp = SimplifyBinOpWithSameOpcodeHands(N);
    if (Tmp.getNode()) return Tmp;
  }

  // Simplify the expression using non-local knowledge.
  if (!VT.isVector() &&
      SimplifyDemandedBits(SDValue(N, 0)))
    return SDValue(N, 0);

  return SDValue();
}

/// Handle transforms common to the three shifts, when the shift amount is a
/// constant.
SDValue DAGCombiner::visitShiftByConstant(SDNode *N, ConstantSDNode *Amt) {
  SDNode *LHS = N->getOperand(0).getNode();
  if (!LHS->hasOneUse()) return SDValue();

  // We want to pull some binops through shifts, so that we have (and (shift))
  // instead of (shift (and)), likewise for add, or, xor, etc.  This sort of
  // thing happens with address calculations, so it's important to canonicalize
  // it.
  bool HighBitSet = false;  // Can we transform this if the high bit is set?

  switch (LHS->getOpcode()) {
  default: return SDValue();
  case ISD::OR:
  case ISD::XOR:
    HighBitSet = false; // We can only transform sra if the high bit is clear.
    break;
  case ISD::AND:
    HighBitSet = true;  // We can only transform sra if the high bit is set.
    break;
  case ISD::ADD:
    if (N->getOpcode() != ISD::SHL)
      return SDValue(); // only shl(add) not sr[al](add).
    HighBitSet = false; // We can only transform sra if the high bit is clear.
    break;
  }

  // We require the RHS of the binop to be a constant and not opaque as well.
  ConstantSDNode *BinOpCst = getAsNonOpaqueConstant(LHS->getOperand(1));
  if (!BinOpCst) return SDValue();

  // FIXME: disable this unless the input to the binop is a shift by a constant.
  // If it is not a shift, it pessimizes some common cases like:
  //
  //    void foo(int *X, int i) { X[i & 1235] = 1; }
  //    int bar(int *X, int i) { return X[i & 255]; }
  SDNode *BinOpLHSVal = LHS->getOperand(0).getNode();
  if ((BinOpLHSVal->getOpcode() != ISD::SHL &&
       BinOpLHSVal->getOpcode() != ISD::SRA &&
       BinOpLHSVal->getOpcode() != ISD::SRL) ||
      !isa<ConstantSDNode>(BinOpLHSVal->getOperand(1)))
    return SDValue();

  EVT VT = N->getValueType(0);

  // If this is a signed shift right, and the high bit is modified by the
  // logical operation, do not perform the transformation. The highBitSet
  // boolean indicates the value of the high bit of the constant which would
  // cause it to be modified for this operation.
  if (N->getOpcode() == ISD::SRA) {
    bool BinOpRHSSignSet = BinOpCst->getAPIntValue().isNegative();
    if (BinOpRHSSignSet != HighBitSet)
      return SDValue();
  }

  if (!TLI.isDesirableToCommuteWithShift(LHS))
    return SDValue();

  // Fold the constants, shifting the binop RHS by the shift amount.
  SDValue NewRHS = DAG.getNode(N->getOpcode(), SDLoc(LHS->getOperand(1)),
                               N->getValueType(0),
                               LHS->getOperand(1), N->getOperand(1));
  assert(isa<ConstantSDNode>(NewRHS) && "Folding was not successful!");

  // Create the new shift.
  SDValue NewShift = DAG.getNode(N->getOpcode(),
                                 SDLoc(LHS->getOperand(0)),
                                 VT, LHS->getOperand(0), N->getOperand(1));

  // Create the new binop.
  return DAG.getNode(LHS->getOpcode(), SDLoc(N), VT, NewShift, NewRHS);
}

SDValue DAGCombiner::distributeTruncateThroughAnd(SDNode *N) {
  assert(N->getOpcode() == ISD::TRUNCATE);
  assert(N->getOperand(0).getOpcode() == ISD::AND);

  // (truncate:TruncVT (and N00, N01C)) -> (and (truncate:TruncVT N00), TruncC)
  if (N->hasOneUse() && N->getOperand(0).hasOneUse()) {
    SDValue N01 = N->getOperand(0).getOperand(1);

    if (ConstantSDNode *N01C = isConstOrConstSplat(N01)) {
      if (!N01C->isOpaque()) {
        EVT TruncVT = N->getValueType(0);
        SDValue N00 = N->getOperand(0).getOperand(0);
        APInt TruncC = N01C->getAPIntValue();
        TruncC = TruncC.trunc(TruncVT.getScalarSizeInBits());
        SDLoc DL(N);

        return DAG.getNode(ISD::AND, DL, TruncVT,
                           DAG.getNode(ISD::TRUNCATE, DL, TruncVT, N00),
                           DAG.getConstant(TruncC, DL, TruncVT));
      }
    }
  }

  return SDValue();
}

SDValue DAGCombiner::visitRotate(SDNode *N) {
  // fold (rot* x, (trunc (and y, c))) -> (rot* x, (and (trunc y), (trunc c))).
  if (N->getOperand(1).getOpcode() == ISD::TRUNCATE &&
      N->getOperand(1).getOperand(0).getOpcode() == ISD::AND) {
    SDValue NewOp1 = distributeTruncateThroughAnd(N->getOperand(1).getNode());
    if (NewOp1.getNode())
      return DAG.getNode(N->getOpcode(), SDLoc(N), N->getValueType(0),
                         N->getOperand(0), NewOp1);
  }
  return SDValue();
}

SDValue DAGCombiner::visitSHL(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N0.getValueType();
  unsigned OpSizeInBits = VT.getScalarSizeInBits();

  // fold vector ops
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  if (VT.isVector()) {
    if (SDValue FoldedVOp = SimplifyVBinOp(N))
      return FoldedVOp;

    BuildVectorSDNode *N1CV = dyn_cast<BuildVectorSDNode>(N1);
    // If setcc produces all-one true value then:
    // (shl (and (setcc) N01CV) N1CV) -> (and (setcc) N01CV<<N1CV)
    if (N1CV && N1CV->isConstant()) {
      if (N0.getOpcode() == ISD::AND) {
        SDValue N00 = N0->getOperand(0);
        SDValue N01 = N0->getOperand(1);
        BuildVectorSDNode *N01CV = dyn_cast<BuildVectorSDNode>(N01);

        if (N01CV && N01CV->isConstant() && N00.getOpcode() == ISD::SETCC &&
            TLI.getBooleanContents(N00.getOperand(0).getValueType()) ==
                TargetLowering::ZeroOrNegativeOneBooleanContent) {
          if (SDValue C = DAG.FoldConstantArithmetic(ISD::SHL, SDLoc(N), VT,
                                                     N01CV, N1CV))
            return DAG.getNode(ISD::AND, SDLoc(N), VT, N00, C);
        }
      } else {
        N1C = isConstOrConstSplat(N1);
      }
    }
  }

  // fold (shl c1, c2) -> c1<<c2
  ConstantSDNode *N0C = getAsNonOpaqueConstant(N0);
  if (N0C && N1C && !N1C->isOpaque())
    return DAG.FoldConstantArithmetic(ISD::SHL, SDLoc(N), VT, N0C, N1C);
  // fold (shl 0, x) -> 0
  if (isNullConstant(N0))
    return N0;
  // fold (shl x, c >= size(x)) -> undef
  if (N1C && N1C->getAPIntValue().uge(OpSizeInBits))
    return DAG.getUNDEF(VT);
  // fold (shl x, 0) -> x
  if (N1C && N1C->isNullValue())
    return N0;
  // fold (shl undef, x) -> 0
  if (N0.getOpcode() == ISD::UNDEF)
    return DAG.getConstant(0, SDLoc(N), VT);
  // if (shl x, c) is known to be zero, return 0
  if (DAG.MaskedValueIsZero(SDValue(N, 0),
                            APInt::getAllOnesValue(OpSizeInBits)))
    return DAG.getConstant(0, SDLoc(N), VT);
  // fold (shl x, (trunc (and y, c))) -> (shl x, (and (trunc y), (trunc c))).
  if (N1.getOpcode() == ISD::TRUNCATE &&
      N1.getOperand(0).getOpcode() == ISD::AND) {
    SDValue NewOp1 = distributeTruncateThroughAnd(N1.getNode());
    if (NewOp1.getNode())
      return DAG.getNode(ISD::SHL, SDLoc(N), VT, N0, NewOp1);
  }

  if (N1C && SimplifyDemandedBits(SDValue(N, 0)))
    return SDValue(N, 0);

  // fold (shl (shl x, c1), c2) -> 0 or (shl x, (add c1, c2))
  if (N1C && N0.getOpcode() == ISD::SHL) {
    if (ConstantSDNode *N0C1 = isConstOrConstSplat(N0.getOperand(1))) {
      uint64_t c1 = N0C1->getZExtValue();
      uint64_t c2 = N1C->getZExtValue();
      SDLoc DL(N);
      if (c1 + c2 >= OpSizeInBits)
        return DAG.getConstant(0, DL, VT);
      return DAG.getNode(ISD::SHL, DL, VT, N0.getOperand(0),
                         DAG.getConstant(c1 + c2, DL, N1.getValueType()));
    }
  }

  // fold (shl (ext (shl x, c1)), c2) -> (ext (shl x, (add c1, c2)))
  // For this to be valid, the second form must not preserve any of the bits
  // that are shifted out by the inner shift in the first form.  This means
  // the outer shift size must be >= the number of bits added by the ext.
  // As a corollary, we don't care what kind of ext it is.
  if (N1C && (N0.getOpcode() == ISD::ZERO_EXTEND ||
              N0.getOpcode() == ISD::ANY_EXTEND ||
              N0.getOpcode() == ISD::SIGN_EXTEND) &&
      N0.getOperand(0).getOpcode() == ISD::SHL) {
    SDValue N0Op0 = N0.getOperand(0);
    if (ConstantSDNode *N0Op0C1 = isConstOrConstSplat(N0Op0.getOperand(1))) {
      uint64_t c1 = N0Op0C1->getZExtValue();
      uint64_t c2 = N1C->getZExtValue();
      EVT InnerShiftVT = N0Op0.getValueType();
      uint64_t InnerShiftSize = InnerShiftVT.getScalarSizeInBits();
      if (c2 >= OpSizeInBits - InnerShiftSize) {
        SDLoc DL(N0);
        if (c1 + c2 >= OpSizeInBits)
          return DAG.getConstant(0, DL, VT);
        return DAG.getNode(ISD::SHL, DL, VT,
                           DAG.getNode(N0.getOpcode(), DL, VT,
                                       N0Op0->getOperand(0)),
                           DAG.getConstant(c1 + c2, DL, N1.getValueType()));
      }
    }
  }

  // fold (shl (zext (srl x, C)), C) -> (zext (shl (srl x, C), C))
  // Only fold this if the inner zext has no other uses to avoid increasing
  // the total number of instructions.
  if (N1C && N0.getOpcode() == ISD::ZERO_EXTEND && N0.hasOneUse() &&
      N0.getOperand(0).getOpcode() == ISD::SRL) {
    SDValue N0Op0 = N0.getOperand(0);
    if (ConstantSDNode *N0Op0C1 = isConstOrConstSplat(N0Op0.getOperand(1))) {
      uint64_t c1 = N0Op0C1->getZExtValue();
      if (c1 < VT.getScalarSizeInBits()) {
        uint64_t c2 = N1C->getZExtValue();
        if (c1 == c2) {
          SDValue NewOp0 = N0.getOperand(0);
          EVT CountVT = NewOp0.getOperand(1).getValueType();
          SDLoc DL(N);
          SDValue NewSHL = DAG.getNode(ISD::SHL, DL, NewOp0.getValueType(),
                                       NewOp0,
                                       DAG.getConstant(c2, DL, CountVT));
          AddToWorklist(NewSHL.getNode());
          return DAG.getNode(ISD::ZERO_EXTEND, SDLoc(N0), VT, NewSHL);
        }
      }
    }
  }

  // fold (shl (sr[la] exact X,  C1), C2) -> (shl    X, (C2-C1)) if C1 <= C2
  // fold (shl (sr[la] exact X,  C1), C2) -> (sr[la] X, (C2-C1)) if C1  > C2
  if (N1C && (N0.getOpcode() == ISD::SRL || N0.getOpcode() == ISD::SRA) &&
      cast<BinaryWithFlagsSDNode>(N0)->Flags.hasExact()) {
    if (ConstantSDNode *N0C1 = isConstOrConstSplat(N0.getOperand(1))) {
      uint64_t C1 = N0C1->getZExtValue();
      uint64_t C2 = N1C->getZExtValue();
      SDLoc DL(N);
      if (C1 <= C2)
        return DAG.getNode(ISD::SHL, DL, VT, N0.getOperand(0),
                           DAG.getConstant(C2 - C1, DL, N1.getValueType()));
      return DAG.getNode(N0.getOpcode(), DL, VT, N0.getOperand(0),
                         DAG.getConstant(C1 - C2, DL, N1.getValueType()));
    }
  }

  // fold (shl (srl x, c1), c2) -> (and (shl x, (sub c2, c1), MASK) or
  //                               (and (srl x, (sub c1, c2), MASK)
  // Only fold this if the inner shift has no other uses -- if it does, folding
  // this will increase the total number of instructions.
  if (N1C && N0.getOpcode() == ISD::SRL && N0.hasOneUse()) {
    if (ConstantSDNode *N0C1 = isConstOrConstSplat(N0.getOperand(1))) {
      uint64_t c1 = N0C1->getZExtValue();
      if (c1 < OpSizeInBits) {
        uint64_t c2 = N1C->getZExtValue();
        APInt Mask = APInt::getHighBitsSet(OpSizeInBits, OpSizeInBits - c1);
        SDValue Shift;
        if (c2 > c1) {
          Mask = Mask.shl(c2 - c1);
          SDLoc DL(N);
          Shift = DAG.getNode(ISD::SHL, DL, VT, N0.getOperand(0),
                              DAG.getConstant(c2 - c1, DL, N1.getValueType()));
        } else {
          Mask = Mask.lshr(c1 - c2);
          SDLoc DL(N);
          Shift = DAG.getNode(ISD::SRL, DL, VT, N0.getOperand(0),
                              DAG.getConstant(c1 - c2, DL, N1.getValueType()));
        }
        SDLoc DL(N0);
        return DAG.getNode(ISD::AND, DL, VT, Shift,
                           DAG.getConstant(Mask, DL, VT));
      }
    }
  }
  // fold (shl (sra x, c1), c1) -> (and x, (shl -1, c1))
  if (N1C && N0.getOpcode() == ISD::SRA && N1 == N0.getOperand(1)) {
    unsigned BitSize = VT.getScalarSizeInBits();
    SDLoc DL(N);
    SDValue HiBitsMask =
      DAG.getConstant(APInt::getHighBitsSet(BitSize,
                                            BitSize - N1C->getZExtValue()),
                      DL, VT);
    return DAG.getNode(ISD::AND, DL, VT, N0.getOperand(0),
                       HiBitsMask);
  }

  // fold (shl (add x, c1), c2) -> (add (shl x, c2), c1 << c2)
  // Variant of version done on multiply, except mul by a power of 2 is turned
  // into a shift.
  APInt Val;
  if (N1C && N0.getOpcode() == ISD::ADD && N0.getNode()->hasOneUse() &&
      (isa<ConstantSDNode>(N0.getOperand(1)) ||
       isConstantSplatVector(N0.getOperand(1).getNode(), Val))) {
    SDValue Shl0 = DAG.getNode(ISD::SHL, SDLoc(N0), VT, N0.getOperand(0), N1);
    SDValue Shl1 = DAG.getNode(ISD::SHL, SDLoc(N1), VT, N0.getOperand(1), N1);
    return DAG.getNode(ISD::ADD, SDLoc(N), VT, Shl0, Shl1);
  }

  if (N1C && !N1C->isOpaque()) {
    SDValue NewSHL = visitShiftByConstant(N, N1C);
    if (NewSHL.getNode())
      return NewSHL;
  }

  return SDValue();
}

SDValue DAGCombiner::visitSRA(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N0.getValueType();
  unsigned OpSizeInBits = VT.getScalarType().getSizeInBits();

  // fold vector ops
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  if (VT.isVector()) {
    if (SDValue FoldedVOp = SimplifyVBinOp(N))
      return FoldedVOp;

    N1C = isConstOrConstSplat(N1);
  }

  // fold (sra c1, c2) -> (sra c1, c2)
  ConstantSDNode *N0C = getAsNonOpaqueConstant(N0);
  if (N0C && N1C && !N1C->isOpaque())
    return DAG.FoldConstantArithmetic(ISD::SRA, SDLoc(N), VT, N0C, N1C);
  // fold (sra 0, x) -> 0
  if (isNullConstant(N0))
    return N0;
  // fold (sra -1, x) -> -1
  if (isAllOnesConstant(N0))
    return N0;
  // fold (sra x, (setge c, size(x))) -> undef
  if (N1C && N1C->getZExtValue() >= OpSizeInBits)
    return DAG.getUNDEF(VT);
  // fold (sra x, 0) -> x
  if (N1C && N1C->isNullValue())
    return N0;
  // fold (sra (shl x, c1), c1) -> sext_inreg for some c1 and target supports
  // sext_inreg.
  if (N1C && N0.getOpcode() == ISD::SHL && N1 == N0.getOperand(1)) {
    unsigned LowBits = OpSizeInBits - (unsigned)N1C->getZExtValue();
    EVT ExtVT = EVT::getIntegerVT(*DAG.getContext(), LowBits);
    if (VT.isVector())
      ExtVT = EVT::getVectorVT(*DAG.getContext(),
                               ExtVT, VT.getVectorNumElements());
    if ((!LegalOperations ||
         TLI.isOperationLegal(ISD::SIGN_EXTEND_INREG, ExtVT)))
      return DAG.getNode(ISD::SIGN_EXTEND_INREG, SDLoc(N), VT,
                         N0.getOperand(0), DAG.getValueType(ExtVT));
  }

  // fold (sra (sra x, c1), c2) -> (sra x, (add c1, c2))
  if (N1C && N0.getOpcode() == ISD::SRA) {
    if (ConstantSDNode *C1 = isConstOrConstSplat(N0.getOperand(1))) {
      unsigned Sum = N1C->getZExtValue() + C1->getZExtValue();
      if (Sum >= OpSizeInBits)
        Sum = OpSizeInBits - 1;
      SDLoc DL(N);
      return DAG.getNode(ISD::SRA, DL, VT, N0.getOperand(0),
                         DAG.getConstant(Sum, DL, N1.getValueType()));
    }
  }

  // fold (sra (shl X, m), (sub result_size, n))
  // -> (sign_extend (trunc (shl X, (sub (sub result_size, n), m)))) for
  // result_size - n != m.
  // If truncate is free for the target sext(shl) is likely to result in better
  // code.
  if (N0.getOpcode() == ISD::SHL && N1C) {
    // Get the two constanst of the shifts, CN0 = m, CN = n.
    const ConstantSDNode *N01C = isConstOrConstSplat(N0.getOperand(1));
    if (N01C) {
      LLVMContext &Ctx = *DAG.getContext();
      // Determine what the truncate's result bitsize and type would be.
      EVT TruncVT = EVT::getIntegerVT(Ctx, OpSizeInBits - N1C->getZExtValue());

      if (VT.isVector())
        TruncVT = EVT::getVectorVT(Ctx, TruncVT, VT.getVectorNumElements());

      // Determine the residual right-shift amount.
      signed ShiftAmt = N1C->getZExtValue() - N01C->getZExtValue();

      // If the shift is not a no-op (in which case this should be just a sign
      // extend already), the truncated to type is legal, sign_extend is legal
      // on that type, and the truncate to that type is both legal and free,
      // perform the transform.
      if ((ShiftAmt > 0) &&
          TLI.isOperationLegalOrCustom(ISD::SIGN_EXTEND, TruncVT) &&
          TLI.isOperationLegalOrCustom(ISD::TRUNCATE, VT) &&
          TLI.isTruncateFree(VT, TruncVT)) {

        SDLoc DL(N);
        SDValue Amt = DAG.getConstant(ShiftAmt, DL,
            getShiftAmountTy(N0.getOperand(0).getValueType()));
        SDValue Shift = DAG.getNode(ISD::SRL, DL, VT,
                                    N0.getOperand(0), Amt);
        SDValue Trunc = DAG.getNode(ISD::TRUNCATE, DL, TruncVT,
                                    Shift);
        return DAG.getNode(ISD::SIGN_EXTEND, DL,
                           N->getValueType(0), Trunc);
      }
    }
  }

  // fold (sra x, (trunc (and y, c))) -> (sra x, (and (trunc y), (trunc c))).
  if (N1.getOpcode() == ISD::TRUNCATE &&
      N1.getOperand(0).getOpcode() == ISD::AND) {
    SDValue NewOp1 = distributeTruncateThroughAnd(N1.getNode());
    if (NewOp1.getNode())
      return DAG.getNode(ISD::SRA, SDLoc(N), VT, N0, NewOp1);
  }

  // fold (sra (trunc (srl x, c1)), c2) -> (trunc (sra x, c1 + c2))
  //      if c1 is equal to the number of bits the trunc removes
  if (N0.getOpcode() == ISD::TRUNCATE &&
      (N0.getOperand(0).getOpcode() == ISD::SRL ||
       N0.getOperand(0).getOpcode() == ISD::SRA) &&
      N0.getOperand(0).hasOneUse() &&
      N0.getOperand(0).getOperand(1).hasOneUse() &&
      N1C) {
    SDValue N0Op0 = N0.getOperand(0);
    if (ConstantSDNode *LargeShift = isConstOrConstSplat(N0Op0.getOperand(1))) {
      unsigned LargeShiftVal = LargeShift->getZExtValue();
      EVT LargeVT = N0Op0.getValueType();

      if (LargeVT.getScalarSizeInBits() - OpSizeInBits == LargeShiftVal) {
        SDLoc DL(N);
        SDValue Amt =
          DAG.getConstant(LargeShiftVal + N1C->getZExtValue(), DL,
                          getShiftAmountTy(N0Op0.getOperand(0).getValueType()));
        SDValue SRA = DAG.getNode(ISD::SRA, DL, LargeVT,
                                  N0Op0.getOperand(0), Amt);
        return DAG.getNode(ISD::TRUNCATE, DL, VT, SRA);
      }
    }
  }

  // Simplify, based on bits shifted out of the LHS.
  if (N1C && SimplifyDemandedBits(SDValue(N, 0)))
    return SDValue(N, 0);


  // If the sign bit is known to be zero, switch this to a SRL.
  if (DAG.SignBitIsZero(N0))
    return DAG.getNode(ISD::SRL, SDLoc(N), VT, N0, N1);

  if (N1C && !N1C->isOpaque()) {
    SDValue NewSRA = visitShiftByConstant(N, N1C);
    if (NewSRA.getNode())
      return NewSRA;
  }

  return SDValue();
}

SDValue DAGCombiner::visitSRL(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N0.getValueType();
  unsigned OpSizeInBits = VT.getScalarType().getSizeInBits();

  // fold vector ops
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  if (VT.isVector()) {
    if (SDValue FoldedVOp = SimplifyVBinOp(N))
      return FoldedVOp;

    N1C = isConstOrConstSplat(N1);
  }

  // fold (srl c1, c2) -> c1 >>u c2
  ConstantSDNode *N0C = getAsNonOpaqueConstant(N0);
  if (N0C && N1C && !N1C->isOpaque())
    return DAG.FoldConstantArithmetic(ISD::SRL, SDLoc(N), VT, N0C, N1C);
  // fold (srl 0, x) -> 0
  if (isNullConstant(N0))
    return N0;
  // fold (srl x, c >= size(x)) -> undef
  if (N1C && N1C->getZExtValue() >= OpSizeInBits)
    return DAG.getUNDEF(VT);
  // fold (srl x, 0) -> x
  if (N1C && N1C->isNullValue())
    return N0;
  // if (srl x, c) is known to be zero, return 0
  if (N1C && DAG.MaskedValueIsZero(SDValue(N, 0),
                                   APInt::getAllOnesValue(OpSizeInBits)))
    return DAG.getConstant(0, SDLoc(N), VT);

  // fold (srl (srl x, c1), c2) -> 0 or (srl x, (add c1, c2))
  if (N1C && N0.getOpcode() == ISD::SRL) {
    if (ConstantSDNode *N01C = isConstOrConstSplat(N0.getOperand(1))) {
      uint64_t c1 = N01C->getZExtValue();
      uint64_t c2 = N1C->getZExtValue();
      SDLoc DL(N);
      if (c1 + c2 >= OpSizeInBits)
        return DAG.getConstant(0, DL, VT);
      return DAG.getNode(ISD::SRL, DL, VT, N0.getOperand(0),
                         DAG.getConstant(c1 + c2, DL, N1.getValueType()));
    }
  }

  // fold (srl (trunc (srl x, c1)), c2) -> 0 or (trunc (srl x, (add c1, c2)))
  if (N1C && N0.getOpcode() == ISD::TRUNCATE &&
      N0.getOperand(0).getOpcode() == ISD::SRL &&
      isa<ConstantSDNode>(N0.getOperand(0)->getOperand(1))) {
    uint64_t c1 =
      cast<ConstantSDNode>(N0.getOperand(0)->getOperand(1))->getZExtValue();
    uint64_t c2 = N1C->getZExtValue();
    EVT InnerShiftVT = N0.getOperand(0).getValueType();
    EVT ShiftCountVT = N0.getOperand(0)->getOperand(1).getValueType();
    uint64_t InnerShiftSize = InnerShiftVT.getScalarType().getSizeInBits();
    // This is only valid if the OpSizeInBits + c1 = size of inner shift.
    if (c1 + OpSizeInBits == InnerShiftSize) {
      SDLoc DL(N0);
      if (c1 + c2 >= InnerShiftSize)
        return DAG.getConstant(0, DL, VT);
      return DAG.getNode(ISD::TRUNCATE, DL, VT,
                         DAG.getNode(ISD::SRL, DL, InnerShiftVT,
                                     N0.getOperand(0)->getOperand(0),
                                     DAG.getConstant(c1 + c2, DL,
                                                     ShiftCountVT)));
    }
  }

  // fold (srl (shl x, c), c) -> (and x, cst2)
  if (N1C && N0.getOpcode() == ISD::SHL && N0.getOperand(1) == N1) {
    unsigned BitSize = N0.getScalarValueSizeInBits();
    if (BitSize <= 64) {
      uint64_t ShAmt = N1C->getZExtValue() + 64 - BitSize;
      SDLoc DL(N);
      return DAG.getNode(ISD::AND, DL, VT, N0.getOperand(0),
                         DAG.getConstant(~0ULL >> ShAmt, DL, VT));
    }
  }

  // fold (srl (anyextend x), c) -> (and (anyextend (srl x, c)), mask)
  if (N1C && N0.getOpcode() == ISD::ANY_EXTEND) {
    // Shifting in all undef bits?
    EVT SmallVT = N0.getOperand(0).getValueType();
    unsigned BitSize = SmallVT.getScalarSizeInBits();
    if (N1C->getZExtValue() >= BitSize)
      return DAG.getUNDEF(VT);

    if (!LegalTypes || TLI.isTypeDesirableForOp(ISD::SRL, SmallVT)) {
      uint64_t ShiftAmt = N1C->getZExtValue();
      SDLoc DL0(N0);
      SDValue SmallShift = DAG.getNode(ISD::SRL, DL0, SmallVT,
                                       N0.getOperand(0),
                          DAG.getConstant(ShiftAmt, DL0,
                                          getShiftAmountTy(SmallVT)));
      AddToWorklist(SmallShift.getNode());
      APInt Mask = APInt::getAllOnesValue(OpSizeInBits).lshr(ShiftAmt);
      SDLoc DL(N);
      return DAG.getNode(ISD::AND, DL, VT,
                         DAG.getNode(ISD::ANY_EXTEND, DL, VT, SmallShift),
                         DAG.getConstant(Mask, DL, VT));
    }
  }

  // fold (srl (sra X, Y), 31) -> (srl X, 31).  This srl only looks at the sign
  // bit, which is unmodified by sra.
  if (N1C && N1C->getZExtValue() + 1 == OpSizeInBits) {
    if (N0.getOpcode() == ISD::SRA)
      return DAG.getNode(ISD::SRL, SDLoc(N), VT, N0.getOperand(0), N1);
  }

  // fold (srl (ctlz x), "5") -> x  iff x has one bit set (the low bit).
  if (N1C && N0.getOpcode() == ISD::CTLZ &&
      N1C->getAPIntValue() == Log2_32(OpSizeInBits)) {
    APInt KnownZero, KnownOne;
    DAG.computeKnownBits(N0.getOperand(0), KnownZero, KnownOne);

    // If any of the input bits are KnownOne, then the input couldn't be all
    // zeros, thus the result of the srl will always be zero.
    if (KnownOne.getBoolValue()) return DAG.getConstant(0, SDLoc(N0), VT);

    // If all of the bits input the to ctlz node are known to be zero, then
    // the result of the ctlz is "32" and the result of the shift is one.
    APInt UnknownBits = ~KnownZero;
    if (UnknownBits == 0) return DAG.getConstant(1, SDLoc(N0), VT);

    // Otherwise, check to see if there is exactly one bit input to the ctlz.
    if ((UnknownBits & (UnknownBits - 1)) == 0) {
      // Okay, we know that only that the single bit specified by UnknownBits
      // could be set on input to the CTLZ node. If this bit is set, the SRL
      // will return 0, if it is clear, it returns 1. Change the CTLZ/SRL pair
      // to an SRL/XOR pair, which is likely to simplify more.
      unsigned ShAmt = UnknownBits.countTrailingZeros();
      SDValue Op = N0.getOperand(0);

      if (ShAmt) {
        SDLoc DL(N0);
        Op = DAG.getNode(ISD::SRL, DL, VT, Op,
                  DAG.getConstant(ShAmt, DL,
                                  getShiftAmountTy(Op.getValueType())));
        AddToWorklist(Op.getNode());
      }

      SDLoc DL(N);
      return DAG.getNode(ISD::XOR, DL, VT,
                         Op, DAG.getConstant(1, DL, VT));
    }
  }

  // fold (srl x, (trunc (and y, c))) -> (srl x, (and (trunc y), (trunc c))).
  if (N1.getOpcode() == ISD::TRUNCATE &&
      N1.getOperand(0).getOpcode() == ISD::AND) {
    SDValue NewOp1 = distributeTruncateThroughAnd(N1.getNode());
    if (NewOp1.getNode())
      return DAG.getNode(ISD::SRL, SDLoc(N), VT, N0, NewOp1);
  }

  // fold operands of srl based on knowledge that the low bits are not
  // demanded.
  if (N1C && SimplifyDemandedBits(SDValue(N, 0)))
    return SDValue(N, 0);

  if (N1C && !N1C->isOpaque()) {
    SDValue NewSRL = visitShiftByConstant(N, N1C);
    if (NewSRL.getNode())
      return NewSRL;
  }

  // Attempt to convert a srl of a load into a narrower zero-extending load.
  SDValue NarrowLoad = ReduceLoadWidth(N);
  if (NarrowLoad.getNode())
    return NarrowLoad;

  // Here is a common situation. We want to optimize:
  //
  //   %a = ...
  //   %b = and i32 %a, 2
  //   %c = srl i32 %b, 1
  //   brcond i32 %c ...
  //
  // into
  //
  //   %a = ...
  //   %b = and %a, 2
  //   %c = setcc eq %b, 0
  //   brcond %c ...
  //
  // However when after the source operand of SRL is optimized into AND, the SRL
  // itself may not be optimized further. Look for it and add the BRCOND into
  // the worklist.
  if (N->hasOneUse()) {
    SDNode *Use = *N->use_begin();
    if (Use->getOpcode() == ISD::BRCOND)
      AddToWorklist(Use);
    else if (Use->getOpcode() == ISD::TRUNCATE && Use->hasOneUse()) {
      // Also look pass the truncate.
      Use = *Use->use_begin();
      if (Use->getOpcode() == ISD::BRCOND)
        AddToWorklist(Use);
    }
  }

  return SDValue();
}

SDValue DAGCombiner::visitBSWAP(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // fold (bswap c1) -> c2
  if (isConstantIntBuildVectorOrConstantInt(N0))
    return DAG.getNode(ISD::BSWAP, SDLoc(N), VT, N0);
  // fold (bswap (bswap x)) -> x
  if (N0.getOpcode() == ISD::BSWAP)
    return N0->getOperand(0);
  return SDValue();
}

SDValue DAGCombiner::visitCTLZ(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // fold (ctlz c1) -> c2
  if (isConstantIntBuildVectorOrConstantInt(N0))
    return DAG.getNode(ISD::CTLZ, SDLoc(N), VT, N0);
  return SDValue();
}

SDValue DAGCombiner::visitCTLZ_ZERO_UNDEF(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // fold (ctlz_zero_undef c1) -> c2
  if (isConstantIntBuildVectorOrConstantInt(N0))
    return DAG.getNode(ISD::CTLZ_ZERO_UNDEF, SDLoc(N), VT, N0);
  return SDValue();
}

SDValue DAGCombiner::visitCTTZ(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // fold (cttz c1) -> c2
  if (isConstantIntBuildVectorOrConstantInt(N0))
    return DAG.getNode(ISD::CTTZ, SDLoc(N), VT, N0);
  return SDValue();
}

SDValue DAGCombiner::visitCTTZ_ZERO_UNDEF(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // fold (cttz_zero_undef c1) -> c2
  if (isConstantIntBuildVectorOrConstantInt(N0))
    return DAG.getNode(ISD::CTTZ_ZERO_UNDEF, SDLoc(N), VT, N0);
  return SDValue();
}

SDValue DAGCombiner::visitCTPOP(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // fold (ctpop c1) -> c2
  if (isConstantIntBuildVectorOrConstantInt(N0))
    return DAG.getNode(ISD::CTPOP, SDLoc(N), VT, N0);
  return SDValue();
}


/// \brief Generate Min/Max node
static SDValue combineMinNumMaxNum(SDLoc DL, EVT VT, SDValue LHS, SDValue RHS,
                                   SDValue True, SDValue False,
                                   ISD::CondCode CC, const TargetLowering &TLI,
                                   SelectionDAG &DAG) {
  if (!(LHS == True && RHS == False) && !(LHS == False && RHS == True))
    return SDValue();

  switch (CC) {
  case ISD::SETOLT:
  case ISD::SETOLE:
  case ISD::SETLT:
  case ISD::SETLE:
  case ISD::SETULT:
  case ISD::SETULE: {
    unsigned Opcode = (LHS == True) ? ISD::FMINNUM : ISD::FMAXNUM;
    if (TLI.isOperationLegal(Opcode, VT))
      return DAG.getNode(Opcode, DL, VT, LHS, RHS);
    return SDValue();
  }
  case ISD::SETOGT:
  case ISD::SETOGE:
  case ISD::SETGT:
  case ISD::SETGE:
  case ISD::SETUGT:
  case ISD::SETUGE: {
    unsigned Opcode = (LHS == True) ? ISD::FMAXNUM : ISD::FMINNUM;
    if (TLI.isOperationLegal(Opcode, VT))
      return DAG.getNode(Opcode, DL, VT, LHS, RHS);
    return SDValue();
  }
  default:
    return SDValue();
  }
}

SDValue DAGCombiner::visitSELECT(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  SDValue N2 = N->getOperand(2);
  EVT VT = N->getValueType(0);
  EVT VT0 = N0.getValueType();

  // fold (select C, X, X) -> X
  if (N1 == N2)
    return N1;
  if (const ConstantSDNode *N0C = dyn_cast<const ConstantSDNode>(N0)) {
    // fold (select true, X, Y) -> X
    // fold (select false, X, Y) -> Y
    return !N0C->isNullValue() ? N1 : N2;
  }
  // fold (select C, 1, X) -> (or C, X)
  if (VT == MVT::i1 && isOneConstant(N1))
    return DAG.getNode(ISD::OR, SDLoc(N), VT, N0, N2);
  // fold (select C, 0, 1) -> (xor C, 1)
  // We can't do this reliably if integer based booleans have different contents
  // to floating point based booleans. This is because we can't tell whether we
  // have an integer-based boolean or a floating-point-based boolean unless we
  // can find the SETCC that produced it and inspect its operands. This is
  // fairly easy if C is the SETCC node, but it can potentially be
  // undiscoverable (or not reasonably discoverable). For example, it could be
  // in another basic block or it could require searching a complicated
  // expression.
  if (VT.isInteger() &&
      (VT0 == MVT::i1 || (VT0.isInteger() &&
                          TLI.getBooleanContents(false, false) ==
                              TLI.getBooleanContents(false, true) &&
                          TLI.getBooleanContents(false, false) ==
                              TargetLowering::ZeroOrOneBooleanContent)) &&
      isNullConstant(N1) && isOneConstant(N2)) {
    SDValue XORNode;
    if (VT == VT0) {
      SDLoc DL(N);
      return DAG.getNode(ISD::XOR, DL, VT0,
                         N0, DAG.getConstant(1, DL, VT0));
    }
    SDLoc DL0(N0);
    XORNode = DAG.getNode(ISD::XOR, DL0, VT0,
                          N0, DAG.getConstant(1, DL0, VT0));
    AddToWorklist(XORNode.getNode());
    if (VT.bitsGT(VT0))
      return DAG.getNode(ISD::ZERO_EXTEND, SDLoc(N), VT, XORNode);
    return DAG.getNode(ISD::TRUNCATE, SDLoc(N), VT, XORNode);
  }
  // fold (select C, 0, X) -> (and (not C), X)
  if (VT == VT0 && VT == MVT::i1 && isNullConstant(N1)) {
    SDValue NOTNode = DAG.getNOT(SDLoc(N0), N0, VT);
    AddToWorklist(NOTNode.getNode());
    return DAG.getNode(ISD::AND, SDLoc(N), VT, NOTNode, N2);
  }
  // fold (select C, X, 1) -> (or (not C), X)
  if (VT == VT0 && VT == MVT::i1 && isOneConstant(N2)) {
    SDValue NOTNode = DAG.getNOT(SDLoc(N0), N0, VT);
    AddToWorklist(NOTNode.getNode());
    return DAG.getNode(ISD::OR, SDLoc(N), VT, NOTNode, N1);
  }
  // fold (select C, X, 0) -> (and C, X)
  if (VT == MVT::i1 && isNullConstant(N2))
    return DAG.getNode(ISD::AND, SDLoc(N), VT, N0, N1);
  // fold (select X, X, Y) -> (or X, Y)
  // fold (select X, 1, Y) -> (or X, Y)
  if (VT == MVT::i1 && (N0 == N1 || isOneConstant(N1)))
    return DAG.getNode(ISD::OR, SDLoc(N), VT, N0, N2);
  // fold (select X, Y, X) -> (and X, Y)
  // fold (select X, Y, 0) -> (and X, Y)
  if (VT == MVT::i1 && (N0 == N2 || isNullConstant(N2)))
    return DAG.getNode(ISD::AND, SDLoc(N), VT, N0, N1);

  // If we can fold this based on the true/false value, do so.
  if (SimplifySelectOps(N, N1, N2))
    return SDValue(N, 0);  // Don't revisit N.

  // fold selects based on a setcc into other things, such as min/max/abs
  if (N0.getOpcode() == ISD::SETCC) {
    // select x, y (fcmp lt x, y) -> fminnum x, y
    // select x, y (fcmp gt x, y) -> fmaxnum x, y
    //
    // This is OK if we don't care about what happens if either operand is a
    // NaN.
    //

    // FIXME: Instead of testing for UnsafeFPMath, this should be checking for
    // no signed zeros as well as no nans.
    const TargetOptions &Options = DAG.getTarget().Options;
    if (Options.UnsafeFPMath &&
        VT.isFloatingPoint() && N0.hasOneUse() &&
        DAG.isKnownNeverNaN(N1) && DAG.isKnownNeverNaN(N2)) {
      ISD::CondCode CC = cast<CondCodeSDNode>(N0.getOperand(2))->get();

      SDValue FMinMax =
          combineMinNumMaxNum(SDLoc(N), VT, N0.getOperand(0), N0.getOperand(1),
                              N1, N2, CC, TLI, DAG);
      if (FMinMax)
        return FMinMax;
    }

    if ((!LegalOperations &&
         TLI.isOperationLegalOrCustom(ISD::SELECT_CC, VT)) ||
        TLI.isOperationLegal(ISD::SELECT_CC, VT))
      return DAG.getNode(ISD::SELECT_CC, SDLoc(N), VT,
                         N0.getOperand(0), N0.getOperand(1),
                         N1, N2, N0.getOperand(2));
    return SimplifySelect(SDLoc(N), N0, N1, N2);
  }

  if (VT0 == MVT::i1) {
    if (TLI.shouldNormalizeToSelectSequence(*DAG.getContext(), VT)) {
      // select (and Cond0, Cond1), X, Y
      //   -> select Cond0, (select Cond1, X, Y), Y
      if (N0->getOpcode() == ISD::AND && N0->hasOneUse()) {
        SDValue Cond0 = N0->getOperand(0);
        SDValue Cond1 = N0->getOperand(1);
        SDValue InnerSelect = DAG.getNode(ISD::SELECT, SDLoc(N),
                                          N1.getValueType(), Cond1, N1, N2);
        return DAG.getNode(ISD::SELECT, SDLoc(N), N1.getValueType(), Cond0,
                           InnerSelect, N2);
      }
      // select (or Cond0, Cond1), X, Y -> select Cond0, X, (select Cond1, X, Y)
      if (N0->getOpcode() == ISD::OR && N0->hasOneUse()) {
        SDValue Cond0 = N0->getOperand(0);
        SDValue Cond1 = N0->getOperand(1);
        SDValue InnerSelect = DAG.getNode(ISD::SELECT, SDLoc(N),
                                          N1.getValueType(), Cond1, N1, N2);
        return DAG.getNode(ISD::SELECT, SDLoc(N), N1.getValueType(), Cond0, N1,
                           InnerSelect);
      }
    }

    // select Cond0, (select Cond1, X, Y), Y -> select (and Cond0, Cond1), X, Y
    if (N1->getOpcode() == ISD::SELECT) {
      SDValue N1_0 = N1->getOperand(0);
      SDValue N1_1 = N1->getOperand(1);
      SDValue N1_2 = N1->getOperand(2);
      if (N1_2 == N2 && N0.getValueType() == N1_0.getValueType()) {
        // Create the actual and node if we can generate good code for it.
        if (!TLI.shouldNormalizeToSelectSequence(*DAG.getContext(), VT)) {
          SDValue And = DAG.getNode(ISD::AND, SDLoc(N), N0.getValueType(),
                                    N0, N1_0);
          return DAG.getNode(ISD::SELECT, SDLoc(N), N1.getValueType(), And,
                             N1_1, N2);
        }
        // Otherwise see if we can optimize the "and" to a better pattern.
        if (SDValue Combined = visitANDLike(N0, N1_0, N))
          return DAG.getNode(ISD::SELECT, SDLoc(N), N1.getValueType(), Combined,
                             N1_1, N2);
      }
    }
    // select Cond0, X, (select Cond1, X, Y) -> select (or Cond0, Cond1), X, Y
    if (N2->getOpcode() == ISD::SELECT) {
      SDValue N2_0 = N2->getOperand(0);
      SDValue N2_1 = N2->getOperand(1);
      SDValue N2_2 = N2->getOperand(2);
      if (N2_1 == N1 && N0.getValueType() == N2_0.getValueType()) {
        // Create the actual or node if we can generate good code for it.
        if (!TLI.shouldNormalizeToSelectSequence(*DAG.getContext(), VT)) {
          SDValue Or = DAG.getNode(ISD::OR, SDLoc(N), N0.getValueType(),
                                   N0, N2_0);
          return DAG.getNode(ISD::SELECT, SDLoc(N), N1.getValueType(), Or,
                             N1, N2_2);
        }
        // Otherwise see if we can optimize to a better pattern.
        if (SDValue Combined = visitORLike(N0, N2_0, N))
          return DAG.getNode(ISD::SELECT, SDLoc(N), N1.getValueType(), Combined,
                             N1, N2_2);
      }
    }
  }

  return SDValue();
}

static
std::pair<SDValue, SDValue> SplitVSETCC(const SDNode *N, SelectionDAG &DAG) {
  SDLoc DL(N);
  EVT LoVT, HiVT;
  std::tie(LoVT, HiVT) = DAG.GetSplitDestVTs(N->getValueType(0));

  // Split the inputs.
  SDValue Lo, Hi, LL, LH, RL, RH;
  std::tie(LL, LH) = DAG.SplitVectorOperand(N, 0);
  std::tie(RL, RH) = DAG.SplitVectorOperand(N, 1);

  Lo = DAG.getNode(N->getOpcode(), DL, LoVT, LL, RL, N->getOperand(2));
  Hi = DAG.getNode(N->getOpcode(), DL, HiVT, LH, RH, N->getOperand(2));

  return std::make_pair(Lo, Hi);
}

// This function assumes all the vselect's arguments are CONCAT_VECTOR
// nodes and that the condition is a BV of ConstantSDNodes (or undefs).
static SDValue ConvertSelectToConcatVector(SDNode *N, SelectionDAG &DAG) {
  SDLoc dl(N);
  SDValue Cond = N->getOperand(0);
  SDValue LHS = N->getOperand(1);
  SDValue RHS = N->getOperand(2);
  EVT VT = N->getValueType(0);
  int NumElems = VT.getVectorNumElements();
  assert(LHS.getOpcode() == ISD::CONCAT_VECTORS &&
         RHS.getOpcode() == ISD::CONCAT_VECTORS &&
         Cond.getOpcode() == ISD::BUILD_VECTOR);

  // CONCAT_VECTOR can take an arbitrary number of arguments. We only care about
  // binary ones here.
  if (LHS->getNumOperands() != 2 || RHS->getNumOperands() != 2)
    return SDValue();

  // We're sure we have an even number of elements due to the
  // concat_vectors we have as arguments to vselect.
  // Skip BV elements until we find one that's not an UNDEF
  // After we find an UNDEF element, keep looping until we get to half the
  // length of the BV and see if all the non-undef nodes are the same.
  ConstantSDNode *BottomHalf = nullptr;
  for (int i = 0; i < NumElems / 2; ++i) {
    if (Cond->getOperand(i)->getOpcode() == ISD::UNDEF)
      continue;

    if (BottomHalf == nullptr)
      BottomHalf = cast<ConstantSDNode>(Cond.getOperand(i));
    else if (Cond->getOperand(i).getNode() != BottomHalf)
      return SDValue();
  }

  // Do the same for the second half of the BuildVector
  ConstantSDNode *TopHalf = nullptr;
  for (int i = NumElems / 2; i < NumElems; ++i) {
    if (Cond->getOperand(i)->getOpcode() == ISD::UNDEF)
      continue;

    if (TopHalf == nullptr)
      TopHalf = cast<ConstantSDNode>(Cond.getOperand(i));
    else if (Cond->getOperand(i).getNode() != TopHalf)
      return SDValue();
  }

  assert(TopHalf && BottomHalf &&
         "One half of the selector was all UNDEFs and the other was all the "
         "same value. This should have been addressed before this function.");
  return DAG.getNode(
      ISD::CONCAT_VECTORS, dl, VT,
      BottomHalf->isNullValue() ? RHS->getOperand(0) : LHS->getOperand(0),
      TopHalf->isNullValue() ? RHS->getOperand(1) : LHS->getOperand(1));
}

SDValue DAGCombiner::visitMSCATTER(SDNode *N) {

  if (Level >= AfterLegalizeTypes)
    return SDValue();

  MaskedScatterSDNode *MSC = cast<MaskedScatterSDNode>(N);
  SDValue Mask = MSC->getMask();
  SDValue Data  = MSC->getValue();
  SDLoc DL(N);

  // If the MSCATTER data type requires splitting and the mask is provided by a
  // SETCC, then split both nodes and its operands before legalization. This
  // prevents the type legalizer from unrolling SETCC into scalar comparisons
  // and enables future optimizations (e.g. min/max pattern matching on X86).
  if (Mask.getOpcode() != ISD::SETCC)
    return SDValue();

  // Check if any splitting is required.
  if (TLI.getTypeAction(*DAG.getContext(), Data.getValueType()) !=
      TargetLowering::TypeSplitVector)
    return SDValue();
  SDValue MaskLo, MaskHi, Lo, Hi;
  std::tie(MaskLo, MaskHi) = SplitVSETCC(Mask.getNode(), DAG);

  EVT LoVT, HiVT;
  std::tie(LoVT, HiVT) = DAG.GetSplitDestVTs(MSC->getValueType(0));

  SDValue Chain = MSC->getChain();

  EVT MemoryVT = MSC->getMemoryVT();
  unsigned Alignment = MSC->getOriginalAlignment();

  EVT LoMemVT, HiMemVT;
  std::tie(LoMemVT, HiMemVT) = DAG.GetSplitDestVTs(MemoryVT);

  SDValue DataLo, DataHi;
  std::tie(DataLo, DataHi) = DAG.SplitVector(Data, DL);

  SDValue BasePtr = MSC->getBasePtr();
  SDValue IndexLo, IndexHi;
  std::tie(IndexLo, IndexHi) = DAG.SplitVector(MSC->getIndex(), DL);

  MachineMemOperand *MMO = DAG.getMachineFunction().
    getMachineMemOperand(MSC->getPointerInfo(),
                          MachineMemOperand::MOStore,  LoMemVT.getStoreSize(),
                          Alignment, MSC->getAAInfo(), MSC->getRanges());

  SDValue OpsLo[] = { Chain, DataLo, MaskLo, BasePtr, IndexLo };
  Lo = DAG.getMaskedScatter(DAG.getVTList(MVT::Other), DataLo.getValueType(),
                            DL, OpsLo, MMO);

  SDValue OpsHi[] = {Chain, DataHi, MaskHi, BasePtr, IndexHi};
  Hi = DAG.getMaskedScatter(DAG.getVTList(MVT::Other), DataHi.getValueType(),
                            DL, OpsHi, MMO);

  AddToWorklist(Lo.getNode());
  AddToWorklist(Hi.getNode());

  return DAG.getNode(ISD::TokenFactor, DL, MVT::Other, Lo, Hi);
}

SDValue DAGCombiner::visitMSTORE(SDNode *N) {

  if (Level >= AfterLegalizeTypes)
    return SDValue();

  MaskedStoreSDNode *MST = dyn_cast<MaskedStoreSDNode>(N);
  SDValue Mask = MST->getMask();
  SDValue Data  = MST->getValue();
  SDLoc DL(N);

  // If the MSTORE data type requires splitting and the mask is provided by a
  // SETCC, then split both nodes and its operands before legalization. This
  // prevents the type legalizer from unrolling SETCC into scalar comparisons
  // and enables future optimizations (e.g. min/max pattern matching on X86).
  if (Mask.getOpcode() == ISD::SETCC) {

    // Check if any splitting is required.
    if (TLI.getTypeAction(*DAG.getContext(), Data.getValueType()) !=
        TargetLowering::TypeSplitVector)
      return SDValue();

    SDValue MaskLo, MaskHi, Lo, Hi;
    std::tie(MaskLo, MaskHi) = SplitVSETCC(Mask.getNode(), DAG);

    EVT LoVT, HiVT;
    std::tie(LoVT, HiVT) = DAG.GetSplitDestVTs(MST->getValueType(0));

    SDValue Chain = MST->getChain();
    SDValue Ptr   = MST->getBasePtr();

    EVT MemoryVT = MST->getMemoryVT();
    unsigned Alignment = MST->getOriginalAlignment();

    // if Alignment is equal to the vector size,
    // take the half of it for the second part
    unsigned SecondHalfAlignment =
      (Alignment == Data->getValueType(0).getSizeInBits()/8) ?
         Alignment/2 : Alignment;

    EVT LoMemVT, HiMemVT;
    std::tie(LoMemVT, HiMemVT) = DAG.GetSplitDestVTs(MemoryVT);

    SDValue DataLo, DataHi;
    std::tie(DataLo, DataHi) = DAG.SplitVector(Data, DL);

    MachineMemOperand *MMO = DAG.getMachineFunction().
      getMachineMemOperand(MST->getPointerInfo(),
                           MachineMemOperand::MOStore,  LoMemVT.getStoreSize(),
                           Alignment, MST->getAAInfo(), MST->getRanges());

    Lo = DAG.getMaskedStore(Chain, DL, DataLo, Ptr, MaskLo, LoMemVT, MMO,
                            MST->isTruncatingStore());

    unsigned IncrementSize = LoMemVT.getSizeInBits()/8;
    Ptr = DAG.getNode(ISD::ADD, DL, Ptr.getValueType(), Ptr,
                      DAG.getConstant(IncrementSize, DL, Ptr.getValueType()));

    MMO = DAG.getMachineFunction().
      getMachineMemOperand(MST->getPointerInfo(),
                           MachineMemOperand::MOStore,  HiMemVT.getStoreSize(),
                           SecondHalfAlignment, MST->getAAInfo(),
                           MST->getRanges());

    Hi = DAG.getMaskedStore(Chain, DL, DataHi, Ptr, MaskHi, HiMemVT, MMO,
                            MST->isTruncatingStore());

    AddToWorklist(Lo.getNode());
    AddToWorklist(Hi.getNode());

    return DAG.getNode(ISD::TokenFactor, DL, MVT::Other, Lo, Hi);
  }
  return SDValue();
}

SDValue DAGCombiner::visitMGATHER(SDNode *N) {

  if (Level >= AfterLegalizeTypes)
    return SDValue();

  MaskedGatherSDNode *MGT = dyn_cast<MaskedGatherSDNode>(N);
  SDValue Mask = MGT->getMask();
  SDLoc DL(N);

  // If the MGATHER result requires splitting and the mask is provided by a
  // SETCC, then split both nodes and its operands before legalization. This
  // prevents the type legalizer from unrolling SETCC into scalar comparisons
  // and enables future optimizations (e.g. min/max pattern matching on X86).

  if (Mask.getOpcode() != ISD::SETCC)
    return SDValue();

  EVT VT = N->getValueType(0);

  // Check if any splitting is required.
  if (TLI.getTypeAction(*DAG.getContext(), VT) !=
      TargetLowering::TypeSplitVector)
    return SDValue();

  SDValue MaskLo, MaskHi, Lo, Hi;
  std::tie(MaskLo, MaskHi) = SplitVSETCC(Mask.getNode(), DAG);

  SDValue Src0 = MGT->getValue();
  SDValue Src0Lo, Src0Hi;
  std::tie(Src0Lo, Src0Hi) = DAG.SplitVector(Src0, DL);

  EVT LoVT, HiVT;
  std::tie(LoVT, HiVT) = DAG.GetSplitDestVTs(VT);

  SDValue Chain = MGT->getChain();
  EVT MemoryVT = MGT->getMemoryVT();
  unsigned Alignment = MGT->getOriginalAlignment();

  EVT LoMemVT, HiMemVT;
  std::tie(LoMemVT, HiMemVT) = DAG.GetSplitDestVTs(MemoryVT);

  SDValue BasePtr = MGT->getBasePtr();
  SDValue Index = MGT->getIndex();
  SDValue IndexLo, IndexHi;
  std::tie(IndexLo, IndexHi) = DAG.SplitVector(Index, DL);

  MachineMemOperand *MMO = DAG.getMachineFunction().
    getMachineMemOperand(MGT->getPointerInfo(),
                          MachineMemOperand::MOLoad,  LoMemVT.getStoreSize(),
                          Alignment, MGT->getAAInfo(), MGT->getRanges());

  SDValue OpsLo[] = { Chain, Src0Lo, MaskLo, BasePtr, IndexLo };
  Lo = DAG.getMaskedGather(DAG.getVTList(LoVT, MVT::Other), LoVT, DL, OpsLo,
                            MMO);

  SDValue OpsHi[] = {Chain, Src0Hi, MaskHi, BasePtr, IndexHi};
  Hi = DAG.getMaskedGather(DAG.getVTList(HiVT, MVT::Other), HiVT, DL, OpsHi,
                            MMO);

  AddToWorklist(Lo.getNode());
  AddToWorklist(Hi.getNode());

  // Build a factor node to remember that this load is independent of the
  // other one.
  Chain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other, Lo.getValue(1),
                      Hi.getValue(1));

  // Legalized the chain result - switch anything that used the old chain to
  // use the new one.
  DAG.ReplaceAllUsesOfValueWith(SDValue(MGT, 1), Chain);

  SDValue GatherRes = DAG.getNode(ISD::CONCAT_VECTORS, DL, VT, Lo, Hi);

  SDValue RetOps[] = { GatherRes, Chain };
  return DAG.getMergeValues(RetOps, DL);
}

SDValue DAGCombiner::visitMLOAD(SDNode *N) {

  if (Level >= AfterLegalizeTypes)
    return SDValue();

  MaskedLoadSDNode *MLD = dyn_cast<MaskedLoadSDNode>(N);
  SDValue Mask = MLD->getMask();
  SDLoc DL(N);

  // If the MLOAD result requires splitting and the mask is provided by a
  // SETCC, then split both nodes and its operands before legalization. This
  // prevents the type legalizer from unrolling SETCC into scalar comparisons
  // and enables future optimizations (e.g. min/max pattern matching on X86).

  if (Mask.getOpcode() == ISD::SETCC) {
    EVT VT = N->getValueType(0);

    // Check if any splitting is required.
    if (TLI.getTypeAction(*DAG.getContext(), VT) !=
        TargetLowering::TypeSplitVector)
      return SDValue();

    SDValue MaskLo, MaskHi, Lo, Hi;
    std::tie(MaskLo, MaskHi) = SplitVSETCC(Mask.getNode(), DAG);

    SDValue Src0 = MLD->getSrc0();
    SDValue Src0Lo, Src0Hi;
    std::tie(Src0Lo, Src0Hi) = DAG.SplitVector(Src0, DL);

    EVT LoVT, HiVT;
    std::tie(LoVT, HiVT) = DAG.GetSplitDestVTs(MLD->getValueType(0));

    SDValue Chain = MLD->getChain();
    SDValue Ptr   = MLD->getBasePtr();
    EVT MemoryVT = MLD->getMemoryVT();
    unsigned Alignment = MLD->getOriginalAlignment();

    // if Alignment is equal to the vector size,
    // take the half of it for the second part
    unsigned SecondHalfAlignment =
      (Alignment == MLD->getValueType(0).getSizeInBits()/8) ?
         Alignment/2 : Alignment;

    EVT LoMemVT, HiMemVT;
    std::tie(LoMemVT, HiMemVT) = DAG.GetSplitDestVTs(MemoryVT);

    MachineMemOperand *MMO = DAG.getMachineFunction().
    getMachineMemOperand(MLD->getPointerInfo(),
                         MachineMemOperand::MOLoad,  LoMemVT.getStoreSize(),
                         Alignment, MLD->getAAInfo(), MLD->getRanges());

    Lo = DAG.getMaskedLoad(LoVT, DL, Chain, Ptr, MaskLo, Src0Lo, LoMemVT, MMO,
                           ISD::NON_EXTLOAD);

    unsigned IncrementSize = LoMemVT.getSizeInBits()/8;
    Ptr = DAG.getNode(ISD::ADD, DL, Ptr.getValueType(), Ptr,
                      DAG.getConstant(IncrementSize, DL, Ptr.getValueType()));

    MMO = DAG.getMachineFunction().
    getMachineMemOperand(MLD->getPointerInfo(),
                         MachineMemOperand::MOLoad,  HiMemVT.getStoreSize(),
                         SecondHalfAlignment, MLD->getAAInfo(), MLD->getRanges());

    Hi = DAG.getMaskedLoad(HiVT, DL, Chain, Ptr, MaskHi, Src0Hi, HiMemVT, MMO,
                           ISD::NON_EXTLOAD);

    AddToWorklist(Lo.getNode());
    AddToWorklist(Hi.getNode());

    // Build a factor node to remember that this load is independent of the
    // other one.
    Chain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other, Lo.getValue(1),
                        Hi.getValue(1));

    // Legalized the chain result - switch anything that used the old chain to
    // use the new one.
    DAG.ReplaceAllUsesOfValueWith(SDValue(MLD, 1), Chain);

    SDValue LoadRes = DAG.getNode(ISD::CONCAT_VECTORS, DL, VT, Lo, Hi);

    SDValue RetOps[] = { LoadRes, Chain };
    return DAG.getMergeValues(RetOps, DL);
  }
  return SDValue();
}

SDValue DAGCombiner::visitVSELECT(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  SDValue N2 = N->getOperand(2);
  SDLoc DL(N);

  // Canonicalize integer abs.
  // vselect (setg[te] X,  0),  X, -X ->
  // vselect (setgt    X, -1),  X, -X ->
  // vselect (setl[te] X,  0), -X,  X ->
  // Y = sra (X, size(X)-1); xor (add (X, Y), Y)
  if (N0.getOpcode() == ISD::SETCC) {
    SDValue LHS = N0.getOperand(0), RHS = N0.getOperand(1);
    ISD::CondCode CC = cast<CondCodeSDNode>(N0.getOperand(2))->get();
    bool isAbs = false;
    bool RHSIsAllZeros = ISD::isBuildVectorAllZeros(RHS.getNode());

    if (((RHSIsAllZeros && (CC == ISD::SETGT || CC == ISD::SETGE)) ||
         (ISD::isBuildVectorAllOnes(RHS.getNode()) && CC == ISD::SETGT)) &&
        N1 == LHS && N2.getOpcode() == ISD::SUB && N1 == N2.getOperand(1))
      isAbs = ISD::isBuildVectorAllZeros(N2.getOperand(0).getNode());
    else if ((RHSIsAllZeros && (CC == ISD::SETLT || CC == ISD::SETLE)) &&
             N2 == LHS && N1.getOpcode() == ISD::SUB && N2 == N1.getOperand(1))
      isAbs = ISD::isBuildVectorAllZeros(N1.getOperand(0).getNode());

    if (isAbs) {
      EVT VT = LHS.getValueType();
      SDValue Shift = DAG.getNode(
          ISD::SRA, DL, VT, LHS,
          DAG.getConstant(VT.getScalarType().getSizeInBits() - 1, DL, VT));
      SDValue Add = DAG.getNode(ISD::ADD, DL, VT, LHS, Shift);
      AddToWorklist(Shift.getNode());
      AddToWorklist(Add.getNode());
      return DAG.getNode(ISD::XOR, DL, VT, Add, Shift);
    }
  }

  if (SimplifySelectOps(N, N1, N2))
    return SDValue(N, 0);  // Don't revisit N.

  // If the VSELECT result requires splitting and the mask is provided by a
  // SETCC, then split both nodes and its operands before legalization. This
  // prevents the type legalizer from unrolling SETCC into scalar comparisons
  // and enables future optimizations (e.g. min/max pattern matching on X86).
  if (N0.getOpcode() == ISD::SETCC) {
    EVT VT = N->getValueType(0);

    // Check if any splitting is required.
    if (TLI.getTypeAction(*DAG.getContext(), VT) !=
        TargetLowering::TypeSplitVector)
      return SDValue();

    SDValue Lo, Hi, CCLo, CCHi, LL, LH, RL, RH;
    std::tie(CCLo, CCHi) = SplitVSETCC(N0.getNode(), DAG);
    std::tie(LL, LH) = DAG.SplitVectorOperand(N, 1);
    std::tie(RL, RH) = DAG.SplitVectorOperand(N, 2);

    Lo = DAG.getNode(N->getOpcode(), DL, LL.getValueType(), CCLo, LL, RL);
    Hi = DAG.getNode(N->getOpcode(), DL, LH.getValueType(), CCHi, LH, RH);

    // Add the new VSELECT nodes to the work list in case they need to be split
    // again.
    AddToWorklist(Lo.getNode());
    AddToWorklist(Hi.getNode());

    return DAG.getNode(ISD::CONCAT_VECTORS, DL, VT, Lo, Hi);
  }

  // Fold (vselect (build_vector all_ones), N1, N2) -> N1
  if (ISD::isBuildVectorAllOnes(N0.getNode()))
    return N1;
  // Fold (vselect (build_vector all_zeros), N1, N2) -> N2
  if (ISD::isBuildVectorAllZeros(N0.getNode()))
    return N2;

  // The ConvertSelectToConcatVector function is assuming both the above
  // checks for (vselect (build_vector all{ones,zeros) ...) have been made
  // and addressed.
  if (N1.getOpcode() == ISD::CONCAT_VECTORS &&
      N2.getOpcode() == ISD::CONCAT_VECTORS &&
      ISD::isBuildVectorOfConstantSDNodes(N0.getNode())) {
    SDValue CV = ConvertSelectToConcatVector(N, DAG);
    if (CV.getNode())
      return CV;
  }

  return SDValue();
}

SDValue DAGCombiner::visitSELECT_CC(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  SDValue N2 = N->getOperand(2);
  SDValue N3 = N->getOperand(3);
  SDValue N4 = N->getOperand(4);
  ISD::CondCode CC = cast<CondCodeSDNode>(N4)->get();

  // fold select_cc lhs, rhs, x, x, cc -> x
  if (N2 == N3)
    return N2;

  // Determine if the condition we're dealing with is constant
  SDValue SCC = SimplifySetCC(getSetCCResultType(N0.getValueType()),
                              N0, N1, CC, SDLoc(N), false);
  if (SCC.getNode()) {
    AddToWorklist(SCC.getNode());

    if (ConstantSDNode *SCCC = dyn_cast<ConstantSDNode>(SCC.getNode())) {
      if (!SCCC->isNullValue())
        return N2;    // cond always true -> true val
      else
        return N3;    // cond always false -> false val
    } else if (SCC->getOpcode() == ISD::UNDEF) {
      // When the condition is UNDEF, just return the first operand. This is
      // coherent the DAG creation, no setcc node is created in this case
      return N2;
    } else if (SCC.getOpcode() == ISD::SETCC) {
      // Fold to a simpler select_cc
      return DAG.getNode(ISD::SELECT_CC, SDLoc(N), N2.getValueType(),
                         SCC.getOperand(0), SCC.getOperand(1), N2, N3,
                         SCC.getOperand(2));
    }
  }

  // If we can fold this based on the true/false value, do so.
  if (SimplifySelectOps(N, N2, N3))
    return SDValue(N, 0);  // Don't revisit N.

  // fold select_cc into other things, such as min/max/abs
  return SimplifySelectCC(SDLoc(N), N0, N1, N2, N3, CC);
}

SDValue DAGCombiner::visitSETCC(SDNode *N) {
  return SimplifySetCC(N->getValueType(0), N->getOperand(0), N->getOperand(1),
                       cast<CondCodeSDNode>(N->getOperand(2))->get(),
                       SDLoc(N));
}

/// Try to fold a sext/zext/aext dag node into a ConstantSDNode or 
/// a build_vector of constants.
/// This function is called by the DAGCombiner when visiting sext/zext/aext
/// dag nodes (see for example method DAGCombiner::visitSIGN_EXTEND).
/// Vector extends are not folded if operations are legal; this is to
/// avoid introducing illegal build_vector dag nodes.
static SDNode *tryToFoldExtendOfConstant(SDNode *N, const TargetLowering &TLI,
                                         SelectionDAG &DAG, bool LegalTypes,
                                         bool LegalOperations) {
  unsigned Opcode = N->getOpcode();
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  assert((Opcode == ISD::SIGN_EXTEND || Opcode == ISD::ZERO_EXTEND ||
         Opcode == ISD::ANY_EXTEND || Opcode == ISD::SIGN_EXTEND_VECTOR_INREG)
         && "Expected EXTEND dag node in input!");

  // fold (sext c1) -> c1
  // fold (zext c1) -> c1
  // fold (aext c1) -> c1
  if (isa<ConstantSDNode>(N0))
    return DAG.getNode(Opcode, SDLoc(N), VT, N0).getNode();

  // fold (sext (build_vector AllConstants) -> (build_vector AllConstants)
  // fold (zext (build_vector AllConstants) -> (build_vector AllConstants)
  // fold (aext (build_vector AllConstants) -> (build_vector AllConstants)
  EVT SVT = VT.getScalarType();
  if (!(VT.isVector() &&
      (!LegalTypes || (!LegalOperations && TLI.isTypeLegal(SVT))) &&
      ISD::isBuildVectorOfConstantSDNodes(N0.getNode())))
    return nullptr;

  // We can fold this node into a build_vector.
  unsigned VTBits = SVT.getSizeInBits();
  unsigned EVTBits = N0->getValueType(0).getScalarType().getSizeInBits();
  SmallVector<SDValue, 8> Elts;
  unsigned NumElts = VT.getVectorNumElements();
  SDLoc DL(N);

  for (unsigned i=0; i != NumElts; ++i) {
    SDValue Op = N0->getOperand(i);
    if (Op->getOpcode() == ISD::UNDEF) {
      Elts.push_back(DAG.getUNDEF(SVT));
      continue;
    }

    SDLoc DL(Op);
    // Get the constant value and if needed trunc it to the size of the type.
    // Nodes like build_vector might have constants wider than the scalar type.
    APInt C = cast<ConstantSDNode>(Op)->getAPIntValue().zextOrTrunc(EVTBits);
    if (Opcode == ISD::SIGN_EXTEND || Opcode == ISD::SIGN_EXTEND_VECTOR_INREG)
      Elts.push_back(DAG.getConstant(C.sext(VTBits), DL, SVT));
    else
      Elts.push_back(DAG.getConstant(C.zext(VTBits), DL, SVT));
  }

  return DAG.getNode(ISD::BUILD_VECTOR, DL, VT, Elts).getNode();
}

// ExtendUsesToFormExtLoad - Trying to extend uses of a load to enable this:
// "fold ({s|z|a}ext (load x)) -> ({s|z|a}ext (truncate ({s|z|a}extload x)))"
// transformation. Returns true if extension are possible and the above
// mentioned transformation is profitable.
static bool ExtendUsesToFormExtLoad(SDNode *N, SDValue N0,
                                    unsigned ExtOpc,
                                    SmallVectorImpl<SDNode *> &ExtendNodes,
                                    const TargetLowering &TLI) {
  bool HasCopyToRegUses = false;
  bool isTruncFree = TLI.isTruncateFree(N->getValueType(0), N0.getValueType());
  for (SDNode::use_iterator UI = N0.getNode()->use_begin(),
                            UE = N0.getNode()->use_end();
       UI != UE; ++UI) {
    SDNode *User = *UI;
    if (User == N)
      continue;
    if (UI.getUse().getResNo() != N0.getResNo())
      continue;
    // FIXME: Only extend SETCC N, N and SETCC N, c for now.
    if (ExtOpc != ISD::ANY_EXTEND && User->getOpcode() == ISD::SETCC) {
      ISD::CondCode CC = cast<CondCodeSDNode>(User->getOperand(2))->get();
      if (ExtOpc == ISD::ZERO_EXTEND && ISD::isSignedIntSetCC(CC))
        // Sign bits will be lost after a zext.
        return false;
      bool Add = false;
      for (unsigned i = 0; i != 2; ++i) {
        SDValue UseOp = User->getOperand(i);
        if (UseOp == N0)
          continue;
        if (!isa<ConstantSDNode>(UseOp))
          return false;
        Add = true;
      }
      if (Add)
        ExtendNodes.push_back(User);
      continue;
    }
    // If truncates aren't free and there are users we can't
    // extend, it isn't worthwhile.
    if (!isTruncFree)
      return false;
    // Remember if this value is live-out.
    if (User->getOpcode() == ISD::CopyToReg)
      HasCopyToRegUses = true;
  }

  if (HasCopyToRegUses) {
    bool BothLiveOut = false;
    for (SDNode::use_iterator UI = N->use_begin(), UE = N->use_end();
         UI != UE; ++UI) {
      SDUse &Use = UI.getUse();
      if (Use.getResNo() == 0 && Use.getUser()->getOpcode() == ISD::CopyToReg) {
        BothLiveOut = true;
        break;
      }
    }
    if (BothLiveOut)
      // Both unextended and extended values are live out. There had better be
      // a good reason for the transformation.
      return ExtendNodes.size();
  }
  return true;
}

void DAGCombiner::ExtendSetCCUses(const SmallVectorImpl<SDNode *> &SetCCs,
                                  SDValue Trunc, SDValue ExtLoad, SDLoc DL,
                                  ISD::NodeType ExtType) {
  // Extend SetCC uses if necessary.
  for (unsigned i = 0, e = SetCCs.size(); i != e; ++i) {
    SDNode *SetCC = SetCCs[i];
    SmallVector<SDValue, 4> Ops;

    for (unsigned j = 0; j != 2; ++j) {
      SDValue SOp = SetCC->getOperand(j);
      if (SOp == Trunc)
        Ops.push_back(ExtLoad);
      else
        Ops.push_back(DAG.getNode(ExtType, DL, ExtLoad->getValueType(0), SOp));
    }

    Ops.push_back(SetCC->getOperand(2));
    CombineTo(SetCC, DAG.getNode(ISD::SETCC, DL, SetCC->getValueType(0), Ops));
  }
}

// FIXME: Bring more similar combines here, common to sext/zext (maybe aext?).
SDValue DAGCombiner::CombineExtLoad(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT DstVT = N->getValueType(0);
  EVT SrcVT = N0.getValueType();

  assert((N->getOpcode() == ISD::SIGN_EXTEND ||
          N->getOpcode() == ISD::ZERO_EXTEND) &&
         "Unexpected node type (not an extend)!");

  // fold (sext (load x)) to multiple smaller sextloads; same for zext.
  // For example, on a target with legal v4i32, but illegal v8i32, turn:
  //   (v8i32 (sext (v8i16 (load x))))
  // into:
  //   (v8i32 (concat_vectors (v4i32 (sextload x)),
  //                          (v4i32 (sextload (x + 16)))))
  // Where uses of the original load, i.e.:
  //   (v8i16 (load x))
  // are replaced with:
  //   (v8i16 (truncate
  //     (v8i32 (concat_vectors (v4i32 (sextload x)),
  //                            (v4i32 (sextload (x + 16)))))))
  //
  // This combine is only applicable to illegal, but splittable, vectors.
  // All legal types, and illegal non-vector types, are handled elsewhere.
  // This combine is controlled by TargetLowering::isVectorLoadExtDesirable.
  //
  if (N0->getOpcode() != ISD::LOAD)
    return SDValue();

  LoadSDNode *LN0 = cast<LoadSDNode>(N0);

  if (!ISD::isNON_EXTLoad(LN0) || !ISD::isUNINDEXEDLoad(LN0) ||
      !N0.hasOneUse() || LN0->isVolatile() || !DstVT.isVector() ||
      !DstVT.isPow2VectorType() || !TLI.isVectorLoadExtDesirable(SDValue(N, 0)))
    return SDValue();

  SmallVector<SDNode *, 4> SetCCs;
  if (!ExtendUsesToFormExtLoad(N, N0, N->getOpcode(), SetCCs, TLI))
    return SDValue();

  ISD::LoadExtType ExtType =
      N->getOpcode() == ISD::SIGN_EXTEND ? ISD::SEXTLOAD : ISD::ZEXTLOAD;

  // Try to split the vector types to get down to legal types.
  EVT SplitSrcVT = SrcVT;
  EVT SplitDstVT = DstVT;
  while (!TLI.isLoadExtLegalOrCustom(ExtType, SplitDstVT, SplitSrcVT) &&
         SplitSrcVT.getVectorNumElements() > 1) {
    SplitDstVT = DAG.GetSplitDestVTs(SplitDstVT).first;
    SplitSrcVT = DAG.GetSplitDestVTs(SplitSrcVT).first;
  }

  if (!TLI.isLoadExtLegalOrCustom(ExtType, SplitDstVT, SplitSrcVT))
    return SDValue();

  SDLoc DL(N);
  const unsigned NumSplits =
      DstVT.getVectorNumElements() / SplitDstVT.getVectorNumElements();
  const unsigned Stride = SplitSrcVT.getStoreSize();
  SmallVector<SDValue, 4> Loads;
  SmallVector<SDValue, 4> Chains;

  SDValue BasePtr = LN0->getBasePtr();
  for (unsigned Idx = 0; Idx < NumSplits; Idx++) {
    const unsigned Offset = Idx * Stride;
    const unsigned Align = MinAlign(LN0->getAlignment(), Offset);

    SDValue SplitLoad = DAG.getExtLoad(
        ExtType, DL, SplitDstVT, LN0->getChain(), BasePtr,
        LN0->getPointerInfo().getWithOffset(Offset), SplitSrcVT,
        LN0->isVolatile(), LN0->isNonTemporal(), LN0->isInvariant(),
        Align, LN0->getAAInfo());

    BasePtr = DAG.getNode(ISD::ADD, DL, BasePtr.getValueType(), BasePtr,
                          DAG.getConstant(Stride, DL, BasePtr.getValueType()));

    Loads.push_back(SplitLoad.getValue(0));
    Chains.push_back(SplitLoad.getValue(1));
  }

  SDValue NewChain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other, Chains);
  SDValue NewValue = DAG.getNode(ISD::CONCAT_VECTORS, DL, DstVT, Loads);

  CombineTo(N, NewValue);

  // Replace uses of the original load (before extension)
  // with a truncate of the concatenated sextloaded vectors.
  SDValue Trunc =
      DAG.getNode(ISD::TRUNCATE, SDLoc(N0), N0.getValueType(), NewValue);
  CombineTo(N0.getNode(), Trunc, NewChain);
  ExtendSetCCUses(SetCCs, Trunc, NewValue, DL,
                  (ISD::NodeType)N->getOpcode());
  return SDValue(N, 0); // Return N so it doesn't get rechecked!
}

SDValue DAGCombiner::visitSIGN_EXTEND(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  if (SDNode *Res = tryToFoldExtendOfConstant(N, TLI, DAG, LegalTypes,
                                              LegalOperations))
    return SDValue(Res, 0);

  // fold (sext (sext x)) -> (sext x)
  // fold (sext (aext x)) -> (sext x)
  if (N0.getOpcode() == ISD::SIGN_EXTEND || N0.getOpcode() == ISD::ANY_EXTEND)
    return DAG.getNode(ISD::SIGN_EXTEND, SDLoc(N), VT,
                       N0.getOperand(0));

  if (N0.getOpcode() == ISD::TRUNCATE) {
    // fold (sext (truncate (load x))) -> (sext (smaller load x))
    // fold (sext (truncate (srl (load x), c))) -> (sext (smaller load (x+c/n)))
    SDValue NarrowLoad = ReduceLoadWidth(N0.getNode());
    if (NarrowLoad.getNode()) {
      SDNode* oye = N0.getNode()->getOperand(0).getNode();
      if (NarrowLoad.getNode() != N0.getNode()) {
        CombineTo(N0.getNode(), NarrowLoad);
        // CombineTo deleted the truncate, if needed, but not what's under it.
        AddToWorklist(oye);
      }
      return SDValue(N, 0);   // Return N so it doesn't get rechecked!
    }

    // See if the value being truncated is already sign extended.  If so, just
    // eliminate the trunc/sext pair.
    SDValue Op = N0.getOperand(0);
    unsigned OpBits   = Op.getValueType().getScalarType().getSizeInBits();
    unsigned MidBits  = N0.getValueType().getScalarType().getSizeInBits();
    unsigned DestBits = VT.getScalarType().getSizeInBits();
    unsigned NumSignBits = DAG.ComputeNumSignBits(Op);

    if (OpBits == DestBits) {
      // Op is i32, Mid is i8, and Dest is i32.  If Op has more than 24 sign
      // bits, it is already ready.
      if (NumSignBits > DestBits-MidBits)
        return Op;
    } else if (OpBits < DestBits) {
      // Op is i32, Mid is i8, and Dest is i64.  If Op has more than 24 sign
      // bits, just sext from i32.
      if (NumSignBits > OpBits-MidBits)
        return DAG.getNode(ISD::SIGN_EXTEND, SDLoc(N), VT, Op);
    } else {
      // Op is i64, Mid is i8, and Dest is i32.  If Op has more than 56 sign
      // bits, just truncate to i32.
      if (NumSignBits > OpBits-MidBits)
        return DAG.getNode(ISD::TRUNCATE, SDLoc(N), VT, Op);
    }

    // fold (sext (truncate x)) -> (sextinreg x).
    if (!LegalOperations || TLI.isOperationLegal(ISD::SIGN_EXTEND_INREG,
                                                 N0.getValueType())) {
      if (OpBits < DestBits)
        Op = DAG.getNode(ISD::ANY_EXTEND, SDLoc(N0), VT, Op);
      else if (OpBits > DestBits)
        Op = DAG.getNode(ISD::TRUNCATE, SDLoc(N0), VT, Op);
      return DAG.getNode(ISD::SIGN_EXTEND_INREG, SDLoc(N), VT, Op,
                         DAG.getValueType(N0.getValueType()));
    }
  }

  // fold (sext (load x)) -> (sext (truncate (sextload x)))
  // Only generate vector extloads when 1) they're legal, and 2) they are
  // deemed desirable by the target.
  if (ISD::isNON_EXTLoad(N0.getNode()) && ISD::isUNINDEXEDLoad(N0.getNode()) &&
      ((!LegalOperations && !VT.isVector() &&
        !cast<LoadSDNode>(N0)->isVolatile()) ||
       TLI.isLoadExtLegal(ISD::SEXTLOAD, VT, N0.getValueType()))) {
    bool DoXform = true;
    SmallVector<SDNode*, 4> SetCCs;
    if (!N0.hasOneUse())
      DoXform = ExtendUsesToFormExtLoad(N, N0, ISD::SIGN_EXTEND, SetCCs, TLI);
    if (VT.isVector())
      DoXform &= TLI.isVectorLoadExtDesirable(SDValue(N, 0));
    if (DoXform) {
      LoadSDNode *LN0 = cast<LoadSDNode>(N0);
      SDValue ExtLoad = DAG.getExtLoad(ISD::SEXTLOAD, SDLoc(N), VT,
                                       LN0->getChain(),
                                       LN0->getBasePtr(), N0.getValueType(),
                                       LN0->getMemOperand());
      CombineTo(N, ExtLoad);
      SDValue Trunc = DAG.getNode(ISD::TRUNCATE, SDLoc(N0),
                                  N0.getValueType(), ExtLoad);
      CombineTo(N0.getNode(), Trunc, ExtLoad.getValue(1));
      ExtendSetCCUses(SetCCs, Trunc, ExtLoad, SDLoc(N),
                      ISD::SIGN_EXTEND);
      return SDValue(N, 0);   // Return N so it doesn't get rechecked!
    }
  }

  // fold (sext (load x)) to multiple smaller sextloads.
  // Only on illegal but splittable vectors.
  if (SDValue ExtLoad = CombineExtLoad(N))
    return ExtLoad;

  // fold (sext (sextload x)) -> (sext (truncate (sextload x)))
  // fold (sext ( extload x)) -> (sext (truncate (sextload x)))
  if ((ISD::isSEXTLoad(N0.getNode()) || ISD::isEXTLoad(N0.getNode())) &&
      ISD::isUNINDEXEDLoad(N0.getNode()) && N0.hasOneUse()) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);
    EVT MemVT = LN0->getMemoryVT();
    if ((!LegalOperations && !LN0->isVolatile()) ||
        TLI.isLoadExtLegal(ISD::SEXTLOAD, VT, MemVT)) {
      SDValue ExtLoad = DAG.getExtLoad(ISD::SEXTLOAD, SDLoc(N), VT,
                                       LN0->getChain(),
                                       LN0->getBasePtr(), MemVT,
                                       LN0->getMemOperand());
      CombineTo(N, ExtLoad);
      CombineTo(N0.getNode(),
                DAG.getNode(ISD::TRUNCATE, SDLoc(N0),
                            N0.getValueType(), ExtLoad),
                ExtLoad.getValue(1));
      return SDValue(N, 0);   // Return N so it doesn't get rechecked!
    }
  }

  // fold (sext (and/or/xor (load x), cst)) ->
  //      (and/or/xor (sextload x), (sext cst))
  if ((N0.getOpcode() == ISD::AND || N0.getOpcode() == ISD::OR ||
       N0.getOpcode() == ISD::XOR) &&
      isa<LoadSDNode>(N0.getOperand(0)) &&
      N0.getOperand(1).getOpcode() == ISD::Constant &&
      TLI.isLoadExtLegal(ISD::SEXTLOAD, VT, N0.getValueType()) &&
      (!LegalOperations && TLI.isOperationLegal(N0.getOpcode(), VT))) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0.getOperand(0));
    if (LN0->getExtensionType() != ISD::ZEXTLOAD && LN0->isUnindexed()) {
      bool DoXform = true;
      SmallVector<SDNode*, 4> SetCCs;
      if (!N0.hasOneUse())
        DoXform = ExtendUsesToFormExtLoad(N, N0.getOperand(0), ISD::SIGN_EXTEND,
                                          SetCCs, TLI);
      if (DoXform) {
        SDValue ExtLoad = DAG.getExtLoad(ISD::SEXTLOAD, SDLoc(LN0), VT,
                                         LN0->getChain(), LN0->getBasePtr(),
                                         LN0->getMemoryVT(),
                                         LN0->getMemOperand());
        APInt Mask = cast<ConstantSDNode>(N0.getOperand(1))->getAPIntValue();
        Mask = Mask.sext(VT.getSizeInBits());
        SDLoc DL(N);
        SDValue And = DAG.getNode(N0.getOpcode(), DL, VT,
                                  ExtLoad, DAG.getConstant(Mask, DL, VT));
        SDValue Trunc = DAG.getNode(ISD::TRUNCATE,
                                    SDLoc(N0.getOperand(0)),
                                    N0.getOperand(0).getValueType(), ExtLoad);
        CombineTo(N, And);
        CombineTo(N0.getOperand(0).getNode(), Trunc, ExtLoad.getValue(1));
        ExtendSetCCUses(SetCCs, Trunc, ExtLoad, DL,
                        ISD::SIGN_EXTEND);
        return SDValue(N, 0);   // Return N so it doesn't get rechecked!
      }
    }
  }

  if (N0.getOpcode() == ISD::SETCC) {
    EVT N0VT = N0.getOperand(0).getValueType();
    // sext(setcc) -> sext_in_reg(vsetcc) for vectors.
    // Only do this before legalize for now.
    if (VT.isVector() && !LegalOperations &&
        TLI.getBooleanContents(N0VT) ==
            TargetLowering::ZeroOrNegativeOneBooleanContent) {
      // On some architectures (such as SSE/NEON/etc) the SETCC result type is
      // of the same size as the compared operands. Only optimize sext(setcc())
      // if this is the case.
      EVT SVT = getSetCCResultType(N0VT);

      // We know that the # elements of the results is the same as the
      // # elements of the compare (and the # elements of the compare result
      // for that matter).  Check to see that they are the same size.  If so,
      // we know that the element size of the sext'd result matches the
      // element size of the compare operands.
      if (VT.getSizeInBits() == SVT.getSizeInBits())
        return DAG.getSetCC(SDLoc(N), VT, N0.getOperand(0),
                             N0.getOperand(1),
                             cast<CondCodeSDNode>(N0.getOperand(2))->get());

      // If the desired elements are smaller or larger than the source
      // elements we can use a matching integer vector type and then
      // truncate/sign extend
      EVT MatchingVectorType = N0VT.changeVectorElementTypeToInteger();
      if (SVT == MatchingVectorType) {
        SDValue VsetCC = DAG.getSetCC(SDLoc(N), MatchingVectorType,
                               N0.getOperand(0), N0.getOperand(1),
                               cast<CondCodeSDNode>(N0.getOperand(2))->get());
        return DAG.getSExtOrTrunc(VsetCC, SDLoc(N), VT);
      }
    }

    // sext(setcc x, y, cc) -> (select (setcc x, y, cc), -1, 0)
    unsigned ElementWidth = VT.getScalarType().getSizeInBits();
    SDLoc DL(N);
    SDValue NegOne =
      DAG.getConstant(APInt::getAllOnesValue(ElementWidth), DL, VT);
    SDValue SCC =
      SimplifySelectCC(DL, N0.getOperand(0), N0.getOperand(1),
                       NegOne, DAG.getConstant(0, DL, VT),
                       cast<CondCodeSDNode>(N0.getOperand(2))->get(), true);
    if (SCC.getNode()) return SCC;

    if (!VT.isVector()) {
      EVT SetCCVT = getSetCCResultType(N0.getOperand(0).getValueType());
      if (!LegalOperations || TLI.isOperationLegal(ISD::SETCC, SetCCVT)) {
        SDLoc DL(N);
        ISD::CondCode CC = cast<CondCodeSDNode>(N0.getOperand(2))->get();
        SDValue SetCC = DAG.getSetCC(DL, SetCCVT,
                                     N0.getOperand(0), N0.getOperand(1), CC);
        return DAG.getSelect(DL, VT, SetCC,
                             NegOne, DAG.getConstant(0, DL, VT));
      }
    }
  }

  // fold (sext x) -> (zext x) if the sign bit is known zero.
  if ((!LegalOperations || TLI.isOperationLegal(ISD::ZERO_EXTEND, VT)) &&
      DAG.SignBitIsZero(N0))
    return DAG.getNode(ISD::ZERO_EXTEND, SDLoc(N), VT, N0);

  return SDValue();
}

// isTruncateOf - If N is a truncate of some other value, return true, record
// the value being truncated in Op and which of Op's bits are zero in KnownZero.
// This function computes KnownZero to avoid a duplicated call to
// computeKnownBits in the caller.
static bool isTruncateOf(SelectionDAG &DAG, SDValue N, SDValue &Op,
                         APInt &KnownZero) {
  APInt KnownOne;
  if (N->getOpcode() == ISD::TRUNCATE) {
    Op = N->getOperand(0);
    DAG.computeKnownBits(Op, KnownZero, KnownOne);
    return true;
  }

  if (N->getOpcode() != ISD::SETCC || N->getValueType(0) != MVT::i1 ||
      cast<CondCodeSDNode>(N->getOperand(2))->get() != ISD::SETNE)
    return false;

  SDValue Op0 = N->getOperand(0);
  SDValue Op1 = N->getOperand(1);
  assert(Op0.getValueType() == Op1.getValueType());

  if (isNullConstant(Op0))
    Op = Op1;
  else if (isNullConstant(Op1))
    Op = Op0;
  else
    return false;

  DAG.computeKnownBits(Op, KnownZero, KnownOne);

  if (!(KnownZero | APInt(Op.getValueSizeInBits(), 1)).isAllOnesValue())
    return false;

  return true;
}

SDValue DAGCombiner::visitZERO_EXTEND(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  if (SDNode *Res = tryToFoldExtendOfConstant(N, TLI, DAG, LegalTypes,
                                              LegalOperations))
    return SDValue(Res, 0);

  // fold (zext (zext x)) -> (zext x)
  // fold (zext (aext x)) -> (zext x)
  if (N0.getOpcode() == ISD::ZERO_EXTEND || N0.getOpcode() == ISD::ANY_EXTEND)
    return DAG.getNode(ISD::ZERO_EXTEND, SDLoc(N), VT,
                       N0.getOperand(0));

  // fold (zext (truncate x)) -> (zext x) or
  //      (zext (truncate x)) -> (truncate x)
  // This is valid when the truncated bits of x are already zero.
  // FIXME: We should extend this to work for vectors too.
  SDValue Op;
  APInt KnownZero;
  if (!VT.isVector() && isTruncateOf(DAG, N0, Op, KnownZero)) {
    APInt TruncatedBits =
      (Op.getValueSizeInBits() == N0.getValueSizeInBits()) ?
      APInt(Op.getValueSizeInBits(), 0) :
      APInt::getBitsSet(Op.getValueSizeInBits(),
                        N0.getValueSizeInBits(),
                        std::min(Op.getValueSizeInBits(),
                                 VT.getSizeInBits()));
    if (TruncatedBits == (KnownZero & TruncatedBits)) {
      if (VT.bitsGT(Op.getValueType()))
        return DAG.getNode(ISD::ZERO_EXTEND, SDLoc(N), VT, Op);
      if (VT.bitsLT(Op.getValueType()))
        return DAG.getNode(ISD::TRUNCATE, SDLoc(N), VT, Op);

      return Op;
    }
  }

  // fold (zext (truncate (load x))) -> (zext (smaller load x))
  // fold (zext (truncate (srl (load x), c))) -> (zext (small load (x+c/n)))
  if (N0.getOpcode() == ISD::TRUNCATE) {
    SDValue NarrowLoad = ReduceLoadWidth(N0.getNode());
    if (NarrowLoad.getNode()) {
      SDNode* oye = N0.getNode()->getOperand(0).getNode();
      if (NarrowLoad.getNode() != N0.getNode()) {
        CombineTo(N0.getNode(), NarrowLoad);
        // CombineTo deleted the truncate, if needed, but not what's under it.
        AddToWorklist(oye);
      }
      return SDValue(N, 0);   // Return N so it doesn't get rechecked!
    }
  }

  // fold (zext (truncate x)) -> (and x, mask)
  if (N0.getOpcode() == ISD::TRUNCATE &&
      (!LegalOperations || TLI.isOperationLegal(ISD::AND, VT))) {

    // fold (zext (truncate (load x))) -> (zext (smaller load x))
    // fold (zext (truncate (srl (load x), c))) -> (zext (smaller load (x+c/n)))
    SDValue NarrowLoad = ReduceLoadWidth(N0.getNode());
    if (NarrowLoad.getNode()) {
      SDNode* oye = N0.getNode()->getOperand(0).getNode();
      if (NarrowLoad.getNode() != N0.getNode()) {
        CombineTo(N0.getNode(), NarrowLoad);
        // CombineTo deleted the truncate, if needed, but not what's under it.
        AddToWorklist(oye);
      }
      return SDValue(N, 0);   // Return N so it doesn't get rechecked!
    }

    SDValue Op = N0.getOperand(0);
    if (Op.getValueType().bitsLT(VT)) {
      Op = DAG.getNode(ISD::ANY_EXTEND, SDLoc(N), VT, Op);
      AddToWorklist(Op.getNode());
    } else if (Op.getValueType().bitsGT(VT)) {
      Op = DAG.getNode(ISD::TRUNCATE, SDLoc(N), VT, Op);
      AddToWorklist(Op.getNode());
    }
    return DAG.getZeroExtendInReg(Op, SDLoc(N),
                                  N0.getValueType().getScalarType());
  }

  // Fold (zext (and (trunc x), cst)) -> (and x, cst),
  // if either of the casts is not free.
  if (N0.getOpcode() == ISD::AND &&
      N0.getOperand(0).getOpcode() == ISD::TRUNCATE &&
      N0.getOperand(1).getOpcode() == ISD::Constant &&
      (!TLI.isTruncateFree(N0.getOperand(0).getOperand(0).getValueType(),
                           N0.getValueType()) ||
       !TLI.isZExtFree(N0.getValueType(), VT))) {
    SDValue X = N0.getOperand(0).getOperand(0);
    if (X.getValueType().bitsLT(VT)) {
      X = DAG.getNode(ISD::ANY_EXTEND, SDLoc(X), VT, X);
    } else if (X.getValueType().bitsGT(VT)) {
      X = DAG.getNode(ISD::TRUNCATE, SDLoc(X), VT, X);
    }
    APInt Mask = cast<ConstantSDNode>(N0.getOperand(1))->getAPIntValue();
    Mask = Mask.zext(VT.getSizeInBits());
    SDLoc DL(N);
    return DAG.getNode(ISD::AND, DL, VT,
                       X, DAG.getConstant(Mask, DL, VT));
  }

  // fold (zext (load x)) -> (zext (truncate (zextload x)))
  // Only generate vector extloads when 1) they're legal, and 2) they are
  // deemed desirable by the target.
  if (ISD::isNON_EXTLoad(N0.getNode()) && ISD::isUNINDEXEDLoad(N0.getNode()) &&
      ((!LegalOperations && !VT.isVector() &&
        !cast<LoadSDNode>(N0)->isVolatile()) ||
       TLI.isLoadExtLegal(ISD::ZEXTLOAD, VT, N0.getValueType()))) {
    bool DoXform = true;
    SmallVector<SDNode*, 4> SetCCs;
    if (!N0.hasOneUse())
      DoXform = ExtendUsesToFormExtLoad(N, N0, ISD::ZERO_EXTEND, SetCCs, TLI);
    if (VT.isVector())
      DoXform &= TLI.isVectorLoadExtDesirable(SDValue(N, 0));
    if (DoXform) {
      LoadSDNode *LN0 = cast<LoadSDNode>(N0);
      SDValue ExtLoad = DAG.getExtLoad(ISD::ZEXTLOAD, SDLoc(N), VT,
                                       LN0->getChain(),
                                       LN0->getBasePtr(), N0.getValueType(),
                                       LN0->getMemOperand());
      CombineTo(N, ExtLoad);
      SDValue Trunc = DAG.getNode(ISD::TRUNCATE, SDLoc(N0),
                                  N0.getValueType(), ExtLoad);
      CombineTo(N0.getNode(), Trunc, ExtLoad.getValue(1));

      ExtendSetCCUses(SetCCs, Trunc, ExtLoad, SDLoc(N),
                      ISD::ZERO_EXTEND);
      return SDValue(N, 0);   // Return N so it doesn't get rechecked!
    }
  }

  // fold (zext (load x)) to multiple smaller zextloads.
  // Only on illegal but splittable vectors.
  if (SDValue ExtLoad = CombineExtLoad(N))
    return ExtLoad;

  // fold (zext (and/or/xor (load x), cst)) ->
  //      (and/or/xor (zextload x), (zext cst))
  if ((N0.getOpcode() == ISD::AND || N0.getOpcode() == ISD::OR ||
       N0.getOpcode() == ISD::XOR) &&
      isa<LoadSDNode>(N0.getOperand(0)) &&
      N0.getOperand(1).getOpcode() == ISD::Constant &&
      TLI.isLoadExtLegal(ISD::ZEXTLOAD, VT, N0.getValueType()) &&
      (!LegalOperations && TLI.isOperationLegal(N0.getOpcode(), VT))) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0.getOperand(0));
    if (LN0->getExtensionType() != ISD::SEXTLOAD && LN0->isUnindexed()) {
      bool DoXform = true;
      SmallVector<SDNode*, 4> SetCCs;
      if (!N0.hasOneUse())
        DoXform = ExtendUsesToFormExtLoad(N, N0.getOperand(0), ISD::ZERO_EXTEND,
                                          SetCCs, TLI);
      if (DoXform) {
        SDValue ExtLoad = DAG.getExtLoad(ISD::ZEXTLOAD, SDLoc(LN0), VT,
                                         LN0->getChain(), LN0->getBasePtr(),
                                         LN0->getMemoryVT(),
                                         LN0->getMemOperand());
        APInt Mask = cast<ConstantSDNode>(N0.getOperand(1))->getAPIntValue();
        Mask = Mask.zext(VT.getSizeInBits());
        SDLoc DL(N);
        SDValue And = DAG.getNode(N0.getOpcode(), DL, VT,
                                  ExtLoad, DAG.getConstant(Mask, DL, VT));
        SDValue Trunc = DAG.getNode(ISD::TRUNCATE,
                                    SDLoc(N0.getOperand(0)),
                                    N0.getOperand(0).getValueType(), ExtLoad);
        CombineTo(N, And);
        CombineTo(N0.getOperand(0).getNode(), Trunc, ExtLoad.getValue(1));
        ExtendSetCCUses(SetCCs, Trunc, ExtLoad, DL,
                        ISD::ZERO_EXTEND);
        return SDValue(N, 0);   // Return N so it doesn't get rechecked!
      }
    }
  }

  // fold (zext (zextload x)) -> (zext (truncate (zextload x)))
  // fold (zext ( extload x)) -> (zext (truncate (zextload x)))
  if ((ISD::isZEXTLoad(N0.getNode()) || ISD::isEXTLoad(N0.getNode())) &&
      ISD::isUNINDEXEDLoad(N0.getNode()) && N0.hasOneUse()) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);
    EVT MemVT = LN0->getMemoryVT();
    if ((!LegalOperations && !LN0->isVolatile()) ||
        TLI.isLoadExtLegal(ISD::ZEXTLOAD, VT, MemVT)) {
      SDValue ExtLoad = DAG.getExtLoad(ISD::ZEXTLOAD, SDLoc(N), VT,
                                       LN0->getChain(),
                                       LN0->getBasePtr(), MemVT,
                                       LN0->getMemOperand());
      CombineTo(N, ExtLoad);
      CombineTo(N0.getNode(),
                DAG.getNode(ISD::TRUNCATE, SDLoc(N0), N0.getValueType(),
                            ExtLoad),
                ExtLoad.getValue(1));
      return SDValue(N, 0);   // Return N so it doesn't get rechecked!
    }
  }

  if (N0.getOpcode() == ISD::SETCC) {
    if (!LegalOperations && VT.isVector() &&
        N0.getValueType().getVectorElementType() == MVT::i1) {
      EVT N0VT = N0.getOperand(0).getValueType();
      if (getSetCCResultType(N0VT) == N0.getValueType())
        return SDValue();

      // zext(setcc) -> (and (vsetcc), (1, 1, ...) for vectors.
      // Only do this before legalize for now.
      EVT EltVT = VT.getVectorElementType();
      SDLoc DL(N);
      SmallVector<SDValue,8> OneOps(VT.getVectorNumElements(),
                                    DAG.getConstant(1, DL, EltVT));
      if (VT.getSizeInBits() == N0VT.getSizeInBits())
        // We know that the # elements of the results is the same as the
        // # elements of the compare (and the # elements of the compare result
        // for that matter).  Check to see that they are the same size.  If so,
        // we know that the element size of the sext'd result matches the
        // element size of the compare operands.
        return DAG.getNode(ISD::AND, DL, VT,
                           DAG.getSetCC(DL, VT, N0.getOperand(0),
                                         N0.getOperand(1),
                                 cast<CondCodeSDNode>(N0.getOperand(2))->get()),
                           DAG.getNode(ISD::BUILD_VECTOR, DL, VT,
                                       OneOps));

      // If the desired elements are smaller or larger than the source
      // elements we can use a matching integer vector type and then
      // truncate/sign extend
      EVT MatchingElementType =
        EVT::getIntegerVT(*DAG.getContext(),
                          N0VT.getScalarType().getSizeInBits());
      EVT MatchingVectorType =
        EVT::getVectorVT(*DAG.getContext(), MatchingElementType,
                         N0VT.getVectorNumElements());
      SDValue VsetCC =
        DAG.getSetCC(DL, MatchingVectorType, N0.getOperand(0),
                      N0.getOperand(1),
                      cast<CondCodeSDNode>(N0.getOperand(2))->get());
      return DAG.getNode(ISD::AND, DL, VT,
                         DAG.getSExtOrTrunc(VsetCC, DL, VT),
                         DAG.getNode(ISD::BUILD_VECTOR, DL, VT, OneOps));
    }

    // zext(setcc x,y,cc) -> select_cc x, y, 1, 0, cc
    SDLoc DL(N);
    SDValue SCC =
      SimplifySelectCC(DL, N0.getOperand(0), N0.getOperand(1),
                       DAG.getConstant(1, DL, VT), DAG.getConstant(0, DL, VT),
                       cast<CondCodeSDNode>(N0.getOperand(2))->get(), true);
    if (SCC.getNode()) return SCC;
  }

  // (zext (shl (zext x), cst)) -> (shl (zext x), cst)
  if ((N0.getOpcode() == ISD::SHL || N0.getOpcode() == ISD::SRL) &&
      isa<ConstantSDNode>(N0.getOperand(1)) &&
      N0.getOperand(0).getOpcode() == ISD::ZERO_EXTEND &&
      N0.hasOneUse()) {
    SDValue ShAmt = N0.getOperand(1);
    unsigned ShAmtVal = cast<ConstantSDNode>(ShAmt)->getZExtValue();
    if (N0.getOpcode() == ISD::SHL) {
      SDValue InnerZExt = N0.getOperand(0);
      // If the original shl may be shifting out bits, do not perform this
      // transformation.
      unsigned KnownZeroBits = InnerZExt.getValueType().getSizeInBits() -
        InnerZExt.getOperand(0).getValueType().getSizeInBits();
      if (ShAmtVal > KnownZeroBits)
        return SDValue();
    }

    SDLoc DL(N);

    // Ensure that the shift amount is wide enough for the shifted value.
    if (VT.getSizeInBits() >= 256)
      ShAmt = DAG.getNode(ISD::ZERO_EXTEND, DL, MVT::i32, ShAmt);

    return DAG.getNode(N0.getOpcode(), DL, VT,
                       DAG.getNode(ISD::ZERO_EXTEND, DL, VT, N0.getOperand(0)),
                       ShAmt);
  }

  return SDValue();
}

SDValue DAGCombiner::visitANY_EXTEND(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  if (SDNode *Res = tryToFoldExtendOfConstant(N, TLI, DAG, LegalTypes,
                                              LegalOperations))
    return SDValue(Res, 0);

  // fold (aext (aext x)) -> (aext x)
  // fold (aext (zext x)) -> (zext x)
  // fold (aext (sext x)) -> (sext x)
  if (N0.getOpcode() == ISD::ANY_EXTEND  ||
      N0.getOpcode() == ISD::ZERO_EXTEND ||
      N0.getOpcode() == ISD::SIGN_EXTEND)
    return DAG.getNode(N0.getOpcode(), SDLoc(N), VT, N0.getOperand(0));

  // fold (aext (truncate (load x))) -> (aext (smaller load x))
  // fold (aext (truncate (srl (load x), c))) -> (aext (small load (x+c/n)))
  if (N0.getOpcode() == ISD::TRUNCATE) {
    SDValue NarrowLoad = ReduceLoadWidth(N0.getNode());
    if (NarrowLoad.getNode()) {
      SDNode* oye = N0.getNode()->getOperand(0).getNode();
      if (NarrowLoad.getNode() != N0.getNode()) {
        CombineTo(N0.getNode(), NarrowLoad);
        // CombineTo deleted the truncate, if needed, but not what's under it.
        AddToWorklist(oye);
      }
      return SDValue(N, 0);   // Return N so it doesn't get rechecked!
    }
  }

  // fold (aext (truncate x))
  if (N0.getOpcode() == ISD::TRUNCATE) {
    SDValue TruncOp = N0.getOperand(0);
    if (TruncOp.getValueType() == VT)
      return TruncOp; // x iff x size == zext size.
    if (TruncOp.getValueType().bitsGT(VT))
      return DAG.getNode(ISD::TRUNCATE, SDLoc(N), VT, TruncOp);
    return DAG.getNode(ISD::ANY_EXTEND, SDLoc(N), VT, TruncOp);
  }

  // Fold (aext (and (trunc x), cst)) -> (and x, cst)
  // if the trunc is not free.
  if (N0.getOpcode() == ISD::AND &&
      N0.getOperand(0).getOpcode() == ISD::TRUNCATE &&
      N0.getOperand(1).getOpcode() == ISD::Constant &&
      !TLI.isTruncateFree(N0.getOperand(0).getOperand(0).getValueType(),
                          N0.getValueType())) {
    SDValue X = N0.getOperand(0).getOperand(0);
    if (X.getValueType().bitsLT(VT)) {
      X = DAG.getNode(ISD::ANY_EXTEND, SDLoc(N), VT, X);
    } else if (X.getValueType().bitsGT(VT)) {
      X = DAG.getNode(ISD::TRUNCATE, SDLoc(N), VT, X);
    }
    APInt Mask = cast<ConstantSDNode>(N0.getOperand(1))->getAPIntValue();
    Mask = Mask.zext(VT.getSizeInBits());
    SDLoc DL(N);
    return DAG.getNode(ISD::AND, DL, VT,
                       X, DAG.getConstant(Mask, DL, VT));
  }

  // fold (aext (load x)) -> (aext (truncate (extload x)))
  // None of the supported targets knows how to perform load and any_ext
  // on vectors in one instruction.  We only perform this transformation on
  // scalars.
  if (ISD::isNON_EXTLoad(N0.getNode()) && !VT.isVector() &&
      ISD::isUNINDEXEDLoad(N0.getNode()) &&
      TLI.isLoadExtLegal(ISD::EXTLOAD, VT, N0.getValueType())) {
    bool DoXform = true;
    SmallVector<SDNode*, 4> SetCCs;
    if (!N0.hasOneUse())
      DoXform = ExtendUsesToFormExtLoad(N, N0, ISD::ANY_EXTEND, SetCCs, TLI);
    if (DoXform) {
      LoadSDNode *LN0 = cast<LoadSDNode>(N0);
      SDValue ExtLoad = DAG.getExtLoad(ISD::EXTLOAD, SDLoc(N), VT,
                                       LN0->getChain(),
                                       LN0->getBasePtr(), N0.getValueType(),
                                       LN0->getMemOperand());
      CombineTo(N, ExtLoad);
      SDValue Trunc = DAG.getNode(ISD::TRUNCATE, SDLoc(N0),
                                  N0.getValueType(), ExtLoad);
      CombineTo(N0.getNode(), Trunc, ExtLoad.getValue(1));
      ExtendSetCCUses(SetCCs, Trunc, ExtLoad, SDLoc(N),
                      ISD::ANY_EXTEND);
      return SDValue(N, 0);   // Return N so it doesn't get rechecked!
    }
  }

  // fold (aext (zextload x)) -> (aext (truncate (zextload x)))
  // fold (aext (sextload x)) -> (aext (truncate (sextload x)))
  // fold (aext ( extload x)) -> (aext (truncate (extload  x)))
  if (N0.getOpcode() == ISD::LOAD &&
      !ISD::isNON_EXTLoad(N0.getNode()) && ISD::isUNINDEXEDLoad(N0.getNode()) &&
      N0.hasOneUse()) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);
    ISD::LoadExtType ExtType = LN0->getExtensionType();
    EVT MemVT = LN0->getMemoryVT();
    if (!LegalOperations || TLI.isLoadExtLegal(ExtType, VT, MemVT)) {
      SDValue ExtLoad = DAG.getExtLoad(ExtType, SDLoc(N),
                                       VT, LN0->getChain(), LN0->getBasePtr(),
                                       MemVT, LN0->getMemOperand());
      CombineTo(N, ExtLoad);
      CombineTo(N0.getNode(),
                DAG.getNode(ISD::TRUNCATE, SDLoc(N0),
                            N0.getValueType(), ExtLoad),
                ExtLoad.getValue(1));
      return SDValue(N, 0);   // Return N so it doesn't get rechecked!
    }
  }

  if (N0.getOpcode() == ISD::SETCC) {
    // For vectors:
    // aext(setcc) -> vsetcc
    // aext(setcc) -> truncate(vsetcc)
    // aext(setcc) -> aext(vsetcc)
    // Only do this before legalize for now.
    if (VT.isVector() && !LegalOperations) {
      EVT N0VT = N0.getOperand(0).getValueType();
        // We know that the # elements of the results is the same as the
        // # elements of the compare (and the # elements of the compare result
        // for that matter).  Check to see that they are the same size.  If so,
        // we know that the element size of the sext'd result matches the
        // element size of the compare operands.
      if (VT.getSizeInBits() == N0VT.getSizeInBits())
        return DAG.getSetCC(SDLoc(N), VT, N0.getOperand(0),
                             N0.getOperand(1),
                             cast<CondCodeSDNode>(N0.getOperand(2))->get());
      // If the desired elements are smaller or larger than the source
      // elements we can use a matching integer vector type and then
      // truncate/any extend
      else {
        EVT MatchingVectorType = N0VT.changeVectorElementTypeToInteger();
        SDValue VsetCC =
          DAG.getSetCC(SDLoc(N), MatchingVectorType, N0.getOperand(0),
                        N0.getOperand(1),
                        cast<CondCodeSDNode>(N0.getOperand(2))->get());
        return DAG.getAnyExtOrTrunc(VsetCC, SDLoc(N), VT);
      }
    }

    // aext(setcc x,y,cc) -> select_cc x, y, 1, 0, cc
    SDLoc DL(N);
    SDValue SCC =
      SimplifySelectCC(DL, N0.getOperand(0), N0.getOperand(1),
                       DAG.getConstant(1, DL, VT), DAG.getConstant(0, DL, VT),
                       cast<CondCodeSDNode>(N0.getOperand(2))->get(), true);
    if (SCC.getNode())
      return SCC;
  }

  return SDValue();
}

/// See if the specified operand can be simplified with the knowledge that only
/// the bits specified by Mask are used.  If so, return the simpler operand,
/// otherwise return a null SDValue.
SDValue DAGCombiner::GetDemandedBits(SDValue V, const APInt &Mask) {
  switch (V.getOpcode()) {
  default: break;
  case ISD::Constant: {
    const ConstantSDNode *CV = cast<ConstantSDNode>(V.getNode());
    assert(CV && "Const value should be ConstSDNode.");
    const APInt &CVal = CV->getAPIntValue();
    APInt NewVal = CVal & Mask;
    if (NewVal != CVal)
      return DAG.getConstant(NewVal, SDLoc(V), V.getValueType());
    break;
  }
  case ISD::OR:
  case ISD::XOR:
    // If the LHS or RHS don't contribute bits to the or, drop them.
    if (DAG.MaskedValueIsZero(V.getOperand(0), Mask))
      return V.getOperand(1);
    if (DAG.MaskedValueIsZero(V.getOperand(1), Mask))
      return V.getOperand(0);
    break;
  case ISD::SRL:
    // Only look at single-use SRLs.
    if (!V.getNode()->hasOneUse())
      break;
    if (ConstantSDNode *RHSC = getAsNonOpaqueConstant(V.getOperand(1))) {
      // See if we can recursively simplify the LHS.
      unsigned Amt = RHSC->getZExtValue();

      // Watch out for shift count overflow though.
      if (Amt >= Mask.getBitWidth()) break;
      APInt NewMask = Mask << Amt;
      SDValue SimplifyLHS = GetDemandedBits(V.getOperand(0), NewMask);
      if (SimplifyLHS.getNode())
        return DAG.getNode(ISD::SRL, SDLoc(V), V.getValueType(),
                           SimplifyLHS, V.getOperand(1));
    }
  }
  return SDValue();
}

/// If the result of a wider load is shifted to right of N  bits and then
/// truncated to a narrower type and where N is a multiple of number of bits of
/// the narrower type, transform it to a narrower load from address + N / num of
/// bits of new type. If the result is to be extended, also fold the extension
/// to form a extending load.
SDValue DAGCombiner::ReduceLoadWidth(SDNode *N) {
  unsigned Opc = N->getOpcode();

  ISD::LoadExtType ExtType = ISD::NON_EXTLOAD;
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);
  EVT ExtVT = VT;

  // This transformation isn't valid for vector loads.
  if (VT.isVector())
    return SDValue();

  // Special case: SIGN_EXTEND_INREG is basically truncating to ExtVT then
  // extended to VT.
  if (Opc == ISD::SIGN_EXTEND_INREG) {
    ExtType = ISD::SEXTLOAD;
    ExtVT = cast<VTSDNode>(N->getOperand(1))->getVT();
  } else if (Opc == ISD::SRL) {
    // Another special-case: SRL is basically zero-extending a narrower value.
    ExtType = ISD::ZEXTLOAD;
    N0 = SDValue(N, 0);
    ConstantSDNode *N01 = dyn_cast<ConstantSDNode>(N0.getOperand(1));
    if (!N01) return SDValue();
    ExtVT = EVT::getIntegerVT(*DAG.getContext(),
                              VT.getSizeInBits() - N01->getZExtValue());
  }
  if (LegalOperations && !TLI.isLoadExtLegal(ExtType, VT, ExtVT))
    return SDValue();

  unsigned EVTBits = ExtVT.getSizeInBits();

  // Do not generate loads of non-round integer types since these can
  // be expensive (and would be wrong if the type is not byte sized).
  if (!ExtVT.isRound())
    return SDValue();

  unsigned ShAmt = 0;
  if (N0.getOpcode() == ISD::SRL && N0.hasOneUse()) {
    if (ConstantSDNode *N01 = dyn_cast<ConstantSDNode>(N0.getOperand(1))) {
      ShAmt = N01->getZExtValue();
      // Is the shift amount a multiple of size of VT?
      if ((ShAmt & (EVTBits-1)) == 0) {
        N0 = N0.getOperand(0);
        // Is the load width a multiple of size of VT?
        if ((N0.getValueType().getSizeInBits() & (EVTBits-1)) != 0)
          return SDValue();
      }

      // At this point, we must have a load or else we can't do the transform.
      if (!isa<LoadSDNode>(N0)) return SDValue();

      // Because a SRL must be assumed to *need* to zero-extend the high bits
      // (as opposed to anyext the high bits), we can't combine the zextload
      // lowering of SRL and an sextload.
      if (cast<LoadSDNode>(N0)->getExtensionType() == ISD::SEXTLOAD)
        return SDValue();

      // If the shift amount is larger than the input type then we're not
      // accessing any of the loaded bytes.  If the load was a zextload/extload
      // then the result of the shift+trunc is zero/undef (handled elsewhere).
      if (ShAmt >= cast<LoadSDNode>(N0)->getMemoryVT().getSizeInBits())
        return SDValue();
    }
  }

  // If the load is shifted left (and the result isn't shifted back right),
  // we can fold the truncate through the shift.
  unsigned ShLeftAmt = 0;
  if (ShAmt == 0 && N0.getOpcode() == ISD::SHL && N0.hasOneUse() &&
      ExtVT == VT && TLI.isNarrowingProfitable(N0.getValueType(), VT)) {
    if (ConstantSDNode *N01 = dyn_cast<ConstantSDNode>(N0.getOperand(1))) {
      ShLeftAmt = N01->getZExtValue();
      N0 = N0.getOperand(0);
    }
  }

  // If we haven't found a load, we can't narrow it.  Don't transform one with
  // multiple uses, this would require adding a new load.
  if (!isa<LoadSDNode>(N0) || !N0.hasOneUse())
    return SDValue();

  // Don't change the width of a volatile load.
  LoadSDNode *LN0 = cast<LoadSDNode>(N0);
  if (LN0->isVolatile())
    return SDValue();

  // Verify that we are actually reducing a load width here.
  if (LN0->getMemoryVT().getSizeInBits() < EVTBits)
    return SDValue();

  // For the transform to be legal, the load must produce only two values
  // (the value loaded and the chain).  Don't transform a pre-increment
  // load, for example, which produces an extra value.  Otherwise the
  // transformation is not equivalent, and the downstream logic to replace
  // uses gets things wrong.
  if (LN0->getNumValues() > 2)
    return SDValue();

  // If the load that we're shrinking is an extload and we're not just
  // discarding the extension we can't simply shrink the load. Bail.
  // TODO: It would be possible to merge the extensions in some cases.
  if (LN0->getExtensionType() != ISD::NON_EXTLOAD &&
      LN0->getMemoryVT().getSizeInBits() < ExtVT.getSizeInBits() + ShAmt)
    return SDValue();

  if (!TLI.shouldReduceLoadWidth(LN0, ExtType, ExtVT))
    return SDValue();

  EVT PtrType = N0.getOperand(1).getValueType();

  if (PtrType == MVT::Untyped || PtrType.isExtended())
    // It's not possible to generate a constant of extended or untyped type.
    return SDValue();

  // For big endian targets, we need to adjust the offset to the pointer to
  // load the correct bytes.
  if (DAG.getDataLayout().isBigEndian()) {
    unsigned LVTStoreBits = LN0->getMemoryVT().getStoreSizeInBits();
    unsigned EVTStoreBits = ExtVT.getStoreSizeInBits();
    ShAmt = LVTStoreBits - EVTStoreBits - ShAmt;
  }

  uint64_t PtrOff = ShAmt / 8;
  unsigned NewAlign = MinAlign(LN0->getAlignment(), PtrOff);
  SDLoc DL(LN0);
  SDValue NewPtr = DAG.getNode(ISD::ADD, DL,
                               PtrType, LN0->getBasePtr(),
                               DAG.getConstant(PtrOff, DL, PtrType));
  AddToWorklist(NewPtr.getNode());

  SDValue Load;
  if (ExtType == ISD::NON_EXTLOAD)
    Load =  DAG.getLoad(VT, SDLoc(N0), LN0->getChain(), NewPtr,
                        LN0->getPointerInfo().getWithOffset(PtrOff),
                        LN0->isVolatile(), LN0->isNonTemporal(),
                        LN0->isInvariant(), NewAlign, LN0->getAAInfo());
  else
    Load = DAG.getExtLoad(ExtType, SDLoc(N0), VT, LN0->getChain(),NewPtr,
                          LN0->getPointerInfo().getWithOffset(PtrOff),
                          ExtVT, LN0->isVolatile(), LN0->isNonTemporal(),
                          LN0->isInvariant(), NewAlign, LN0->getAAInfo());

  // Replace the old load's chain with the new load's chain.
  WorklistRemover DeadNodes(*this);
  DAG.ReplaceAllUsesOfValueWith(N0.getValue(1), Load.getValue(1));

  // Shift the result left, if we've swallowed a left shift.
  SDValue Result = Load;
  if (ShLeftAmt != 0) {
    EVT ShImmTy = getShiftAmountTy(Result.getValueType());
    if (!isUIntN(ShImmTy.getSizeInBits(), ShLeftAmt))
      ShImmTy = VT;
    // If the shift amount is as large as the result size (but, presumably,
    // no larger than the source) then the useful bits of the result are
    // zero; we can't simply return the shortened shift, because the result
    // of that operation is undefined.
    SDLoc DL(N0);
    if (ShLeftAmt >= VT.getSizeInBits())
      Result = DAG.getConstant(0, DL, VT);
    else
      Result = DAG.getNode(ISD::SHL, DL, VT,
                          Result, DAG.getConstant(ShLeftAmt, DL, ShImmTy));
  }

  // Return the new loaded value.
  return Result;
}

SDValue DAGCombiner::visitSIGN_EXTEND_INREG(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N->getValueType(0);
  EVT EVT = cast<VTSDNode>(N1)->getVT();
  unsigned VTBits = VT.getScalarType().getSizeInBits();
  unsigned EVTBits = EVT.getScalarType().getSizeInBits();

  // fold (sext_in_reg c1) -> c1
  if (isa<ConstantSDNode>(N0) || N0.getOpcode() == ISD::UNDEF)
    return DAG.getNode(ISD::SIGN_EXTEND_INREG, SDLoc(N), VT, N0, N1);

  // If the input is already sign extended, just drop the extension.
  if (DAG.ComputeNumSignBits(N0) >= VTBits-EVTBits+1)
    return N0;

  // fold (sext_in_reg (sext_in_reg x, VT2), VT1) -> (sext_in_reg x, minVT) pt2
  if (N0.getOpcode() == ISD::SIGN_EXTEND_INREG &&
      EVT.bitsLT(cast<VTSDNode>(N0.getOperand(1))->getVT()))
    return DAG.getNode(ISD::SIGN_EXTEND_INREG, SDLoc(N), VT,
                       N0.getOperand(0), N1);

  // fold (sext_in_reg (sext x)) -> (sext x)
  // fold (sext_in_reg (aext x)) -> (sext x)
  // if x is small enough.
  if (N0.getOpcode() == ISD::SIGN_EXTEND || N0.getOpcode() == ISD::ANY_EXTEND) {
    SDValue N00 = N0.getOperand(0);
    if (N00.getValueType().getScalarType().getSizeInBits() <= EVTBits &&
        (!LegalOperations || TLI.isOperationLegal(ISD::SIGN_EXTEND, VT)))
      return DAG.getNode(ISD::SIGN_EXTEND, SDLoc(N), VT, N00, N1);
  }

  // fold (sext_in_reg x) -> (zext_in_reg x) if the sign bit is known zero.
  if (DAG.MaskedValueIsZero(N0, APInt::getBitsSet(VTBits, EVTBits-1, EVTBits)))
    return DAG.getZeroExtendInReg(N0, SDLoc(N), EVT);

  // fold operands of sext_in_reg based on knowledge that the top bits are not
  // demanded.
  if (SimplifyDemandedBits(SDValue(N, 0)))
    return SDValue(N, 0);

  // fold (sext_in_reg (load x)) -> (smaller sextload x)
  // fold (sext_in_reg (srl (load x), c)) -> (smaller sextload (x+c/evtbits))
  SDValue NarrowLoad = ReduceLoadWidth(N);
  if (NarrowLoad.getNode())
    return NarrowLoad;

  // fold (sext_in_reg (srl X, 24), i8) -> (sra X, 24)
  // fold (sext_in_reg (srl X, 23), i8) -> (sra X, 23) iff possible.
  // We already fold "(sext_in_reg (srl X, 25), i8) -> srl X, 25" above.
  if (N0.getOpcode() == ISD::SRL) {
    if (ConstantSDNode *ShAmt = dyn_cast<ConstantSDNode>(N0.getOperand(1)))
      if (ShAmt->getZExtValue()+EVTBits <= VTBits) {
        // We can turn this into an SRA iff the input to the SRL is already sign
        // extended enough.
        unsigned InSignBits = DAG.ComputeNumSignBits(N0.getOperand(0));
        if (VTBits-(ShAmt->getZExtValue()+EVTBits) < InSignBits)
          return DAG.getNode(ISD::SRA, SDLoc(N), VT,
                             N0.getOperand(0), N0.getOperand(1));
      }
  }

  // fold (sext_inreg (extload x)) -> (sextload x)
  if (ISD::isEXTLoad(N0.getNode()) &&
      ISD::isUNINDEXEDLoad(N0.getNode()) &&
      EVT == cast<LoadSDNode>(N0)->getMemoryVT() &&
      ((!LegalOperations && !cast<LoadSDNode>(N0)->isVolatile()) ||
       TLI.isLoadExtLegal(ISD::SEXTLOAD, VT, EVT))) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);
    SDValue ExtLoad = DAG.getExtLoad(ISD::SEXTLOAD, SDLoc(N), VT,
                                     LN0->getChain(),
                                     LN0->getBasePtr(), EVT,
                                     LN0->getMemOperand());
    CombineTo(N, ExtLoad);
    CombineTo(N0.getNode(), ExtLoad, ExtLoad.getValue(1));
    AddToWorklist(ExtLoad.getNode());
    return SDValue(N, 0);   // Return N so it doesn't get rechecked!
  }
  // fold (sext_inreg (zextload x)) -> (sextload x) iff load has one use
  if (ISD::isZEXTLoad(N0.getNode()) && ISD::isUNINDEXEDLoad(N0.getNode()) &&
      N0.hasOneUse() &&
      EVT == cast<LoadSDNode>(N0)->getMemoryVT() &&
      ((!LegalOperations && !cast<LoadSDNode>(N0)->isVolatile()) ||
       TLI.isLoadExtLegal(ISD::SEXTLOAD, VT, EVT))) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);
    SDValue ExtLoad = DAG.getExtLoad(ISD::SEXTLOAD, SDLoc(N), VT,
                                     LN0->getChain(),
                                     LN0->getBasePtr(), EVT,
                                     LN0->getMemOperand());
    CombineTo(N, ExtLoad);
    CombineTo(N0.getNode(), ExtLoad, ExtLoad.getValue(1));
    return SDValue(N, 0);   // Return N so it doesn't get rechecked!
  }

  // Form (sext_inreg (bswap >> 16)) or (sext_inreg (rotl (bswap) 16))
  if (EVTBits <= 16 && N0.getOpcode() == ISD::OR) {
    SDValue BSwap = MatchBSwapHWordLow(N0.getNode(), N0.getOperand(0),
                                       N0.getOperand(1), false);
    if (BSwap.getNode())
      return DAG.getNode(ISD::SIGN_EXTEND_INREG, SDLoc(N), VT,
                         BSwap, N1);
  }

  // Fold a sext_inreg of a build_vector of ConstantSDNodes or undefs
  // into a build_vector.
  if (ISD::isBuildVectorOfConstantSDNodes(N0.getNode())) {
    SmallVector<SDValue, 8> Elts;
    unsigned NumElts = N0->getNumOperands();
    unsigned ShAmt = VTBits - EVTBits;

    for (unsigned i = 0; i != NumElts; ++i) {
      SDValue Op = N0->getOperand(i);
      if (Op->getOpcode() == ISD::UNDEF) {
        Elts.push_back(Op);
        continue;
      }

      ConstantSDNode *CurrentND = cast<ConstantSDNode>(Op);
      const APInt &C = APInt(VTBits, CurrentND->getAPIntValue().getZExtValue());
      Elts.push_back(DAG.getConstant(C.shl(ShAmt).ashr(ShAmt).getZExtValue(),
                                     SDLoc(Op), Op.getValueType()));
    }

    return DAG.getNode(ISD::BUILD_VECTOR, SDLoc(N), VT, Elts);
  }

  return SDValue();
}

SDValue DAGCombiner::visitSIGN_EXTEND_VECTOR_INREG(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  if (N0.getOpcode() == ISD::UNDEF)
    return DAG.getUNDEF(VT);

  if (SDNode *Res = tryToFoldExtendOfConstant(N, TLI, DAG, LegalTypes,
                                              LegalOperations))
    return SDValue(Res, 0);

  return SDValue();
}

SDValue DAGCombiner::visitTRUNCATE(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);
  bool isLE = DAG.getDataLayout().isLittleEndian();

  // noop truncate
  if (N0.getValueType() == N->getValueType(0))
    return N0;
  // fold (truncate c1) -> c1
  if (isConstantIntBuildVectorOrConstantInt(N0))
    return DAG.getNode(ISD::TRUNCATE, SDLoc(N), VT, N0);
  // fold (truncate (truncate x)) -> (truncate x)
  if (N0.getOpcode() == ISD::TRUNCATE)
    return DAG.getNode(ISD::TRUNCATE, SDLoc(N), VT, N0.getOperand(0));
  // fold (truncate (ext x)) -> (ext x) or (truncate x) or x
  if (N0.getOpcode() == ISD::ZERO_EXTEND ||
      N0.getOpcode() == ISD::SIGN_EXTEND ||
      N0.getOpcode() == ISD::ANY_EXTEND) {
    if (N0.getOperand(0).getValueType().bitsLT(VT))
      // if the source is smaller than the dest, we still need an extend
      return DAG.getNode(N0.getOpcode(), SDLoc(N), VT,
                         N0.getOperand(0));
    if (N0.getOperand(0).getValueType().bitsGT(VT))
      // if the source is larger than the dest, than we just need the truncate
      return DAG.getNode(ISD::TRUNCATE, SDLoc(N), VT, N0.getOperand(0));
    // if the source and dest are the same type, we can drop both the extend
    // and the truncate.
    return N0.getOperand(0);
  }

  // Fold extract-and-trunc into a narrow extract. For example:
  //   i64 x = EXTRACT_VECTOR_ELT(v2i64 val, i32 1)
  //   i32 y = TRUNCATE(i64 x)
  //        -- becomes --
  //   v16i8 b = BITCAST (v2i64 val)
  //   i8 x = EXTRACT_VECTOR_ELT(v16i8 b, i32 8)
  //
  // Note: We only run this optimization after type legalization (which often
  // creates this pattern) and before operation legalization after which
  // we need to be more careful about the vector instructions that we generate.
  if (N0.getOpcode() == ISD::EXTRACT_VECTOR_ELT &&
      LegalTypes && !LegalOperations && N0->hasOneUse() && VT != MVT::i1) {

    EVT VecTy = N0.getOperand(0).getValueType();
    EVT ExTy = N0.getValueType();
    EVT TrTy = N->getValueType(0);

    unsigned NumElem = VecTy.getVectorNumElements();
    unsigned SizeRatio = ExTy.getSizeInBits()/TrTy.getSizeInBits();

    EVT NVT = EVT::getVectorVT(*DAG.getContext(), TrTy, SizeRatio * NumElem);
    assert(NVT.getSizeInBits() == VecTy.getSizeInBits() && "Invalid Size");

    SDValue EltNo = N0->getOperand(1);
    if (isa<ConstantSDNode>(EltNo) && isTypeLegal(NVT)) {
      int Elt = cast<ConstantSDNode>(EltNo)->getZExtValue();
      EVT IndexTy = TLI.getVectorIdxTy(DAG.getDataLayout());
      int Index = isLE ? (Elt*SizeRatio) : (Elt*SizeRatio + (SizeRatio-1));

      SDValue V = DAG.getNode(ISD::BITCAST, SDLoc(N),
                              NVT, N0.getOperand(0));

      SDLoc DL(N);
      return DAG.getNode(ISD::EXTRACT_VECTOR_ELT,
                         DL, TrTy, V,
                         DAG.getConstant(Index, DL, IndexTy));
    }
  }

  // trunc (select c, a, b) -> select c, (trunc a), (trunc b)
  if (N0.getOpcode() == ISD::SELECT) {
    EVT SrcVT = N0.getValueType();
    if ((!LegalOperations || TLI.isOperationLegal(ISD::SELECT, SrcVT)) &&
        TLI.isTruncateFree(SrcVT, VT)) {
      SDLoc SL(N0);
      SDValue Cond = N0.getOperand(0);
      SDValue TruncOp0 = DAG.getNode(ISD::TRUNCATE, SL, VT, N0.getOperand(1));
      SDValue TruncOp1 = DAG.getNode(ISD::TRUNCATE, SL, VT, N0.getOperand(2));
      return DAG.getNode(ISD::SELECT, SDLoc(N), VT, Cond, TruncOp0, TruncOp1);
    }
  }

  // Fold a series of buildvector, bitcast, and truncate if possible.
  // For example fold
  //   (2xi32 trunc (bitcast ((4xi32)buildvector x, x, y, y) 2xi64)) to
  //   (2xi32 (buildvector x, y)).
  if (Level == AfterLegalizeVectorOps && VT.isVector() &&
      N0.getOpcode() == ISD::BITCAST && N0.hasOneUse() &&
      N0.getOperand(0).getOpcode() == ISD::BUILD_VECTOR &&
      N0.getOperand(0).hasOneUse()) {

    SDValue BuildVect = N0.getOperand(0);
    EVT BuildVectEltTy = BuildVect.getValueType().getVectorElementType();
    EVT TruncVecEltTy = VT.getVectorElementType();

    // Check that the element types match.
    if (BuildVectEltTy == TruncVecEltTy) {
      // Now we only need to compute the offset of the truncated elements.
      unsigned BuildVecNumElts =  BuildVect.getNumOperands();
      unsigned TruncVecNumElts = VT.getVectorNumElements();
      unsigned TruncEltOffset = BuildVecNumElts / TruncVecNumElts;

      assert((BuildVecNumElts % TruncVecNumElts) == 0 &&
             "Invalid number of elements");

      SmallVector<SDValue, 8> Opnds;
      for (unsigned i = 0, e = BuildVecNumElts; i != e; i += TruncEltOffset)
        Opnds.push_back(BuildVect.getOperand(i));

      return DAG.getNode(ISD::BUILD_VECTOR, SDLoc(N), VT, Opnds);
    }
  }

  // See if we can simplify the input to this truncate through knowledge that
  // only the low bits are being used.
  // For example "trunc (or (shl x, 8), y)" // -> trunc y
  // Currently we only perform this optimization on scalars because vectors
  // may have different active low bits.
  if (!VT.isVector()) {
    SDValue Shorter =
      GetDemandedBits(N0, APInt::getLowBitsSet(N0.getValueSizeInBits(),
                                               VT.getSizeInBits()));
    if (Shorter.getNode())
      return DAG.getNode(ISD::TRUNCATE, SDLoc(N), VT, Shorter);
  }
  // fold (truncate (load x)) -> (smaller load x)
  // fold (truncate (srl (load x), c)) -> (smaller load (x+c/evtbits))
  if (!LegalTypes || TLI.isTypeDesirableForOp(N0.getOpcode(), VT)) {
    SDValue Reduced = ReduceLoadWidth(N);
    if (Reduced.getNode())
      return Reduced;
    // Handle the case where the load remains an extending load even
    // after truncation.
    if (N0.hasOneUse() && ISD::isUNINDEXEDLoad(N0.getNode())) {
      LoadSDNode *LN0 = cast<LoadSDNode>(N0);
      if (!LN0->isVolatile() &&
          LN0->getMemoryVT().getStoreSizeInBits() < VT.getSizeInBits()) {
        SDValue NewLoad = DAG.getExtLoad(LN0->getExtensionType(), SDLoc(LN0),
                                         VT, LN0->getChain(), LN0->getBasePtr(),
                                         LN0->getMemoryVT(),
                                         LN0->getMemOperand());
        DAG.ReplaceAllUsesOfValueWith(N0.getValue(1), NewLoad.getValue(1));
        return NewLoad;
      }
    }
  }
  // fold (trunc (concat ... x ...)) -> (concat ..., (trunc x), ...)),
  // where ... are all 'undef'.
  if (N0.getOpcode() == ISD::CONCAT_VECTORS && !LegalTypes) {
    SmallVector<EVT, 8> VTs;
    SDValue V;
    unsigned Idx = 0;
    unsigned NumDefs = 0;

    for (unsigned i = 0, e = N0.getNumOperands(); i != e; ++i) {
      SDValue X = N0.getOperand(i);
      if (X.getOpcode() != ISD::UNDEF) {
        V = X;
        Idx = i;
        NumDefs++;
      }
      // Stop if more than one members are non-undef.
      if (NumDefs > 1)
        break;
      VTs.push_back(EVT::getVectorVT(*DAG.getContext(),
                                     VT.getVectorElementType(),
                                     X.getValueType().getVectorNumElements()));
    }

    if (NumDefs == 0)
      return DAG.getUNDEF(VT);

    if (NumDefs == 1) {
      assert(V.getNode() && "The single defined operand is empty!");
      SmallVector<SDValue, 8> Opnds;
      for (unsigned i = 0, e = VTs.size(); i != e; ++i) {
        if (i != Idx) {
          Opnds.push_back(DAG.getUNDEF(VTs[i]));
          continue;
        }
        SDValue NV = DAG.getNode(ISD::TRUNCATE, SDLoc(V), VTs[i], V);
        AddToWorklist(NV.getNode());
        Opnds.push_back(NV);
      }
      return DAG.getNode(ISD::CONCAT_VECTORS, SDLoc(N), VT, Opnds);
    }
  }

  // Simplify the operands using demanded-bits information.
  if (!VT.isVector() &&
      SimplifyDemandedBits(SDValue(N, 0)))
    return SDValue(N, 0);

  return SDValue();
}

static SDNode *getBuildPairElt(SDNode *N, unsigned i) {
  SDValue Elt = N->getOperand(i);
  if (Elt.getOpcode() != ISD::MERGE_VALUES)
    return Elt.getNode();
  return Elt.getOperand(Elt.getResNo()).getNode();
}

/// build_pair (load, load) -> load
/// if load locations are consecutive.
SDValue DAGCombiner::CombineConsecutiveLoads(SDNode *N, EVT VT) {
  assert(N->getOpcode() == ISD::BUILD_PAIR);

  LoadSDNode *LD1 = dyn_cast<LoadSDNode>(getBuildPairElt(N, 0));
  LoadSDNode *LD2 = dyn_cast<LoadSDNode>(getBuildPairElt(N, 1));
  if (!LD1 || !LD2 || !ISD::isNON_EXTLoad(LD1) || !LD1->hasOneUse() ||
      LD1->getAddressSpace() != LD2->getAddressSpace())
    return SDValue();
  EVT LD1VT = LD1->getValueType(0);

  if (ISD::isNON_EXTLoad(LD2) &&
      LD2->hasOneUse() &&
      // If both are volatile this would reduce the number of volatile loads.
      // If one is volatile it might be ok, but play conservative and bail out.
      !LD1->isVolatile() &&
      !LD2->isVolatile() &&
      DAG.isConsecutiveLoad(LD2, LD1, LD1VT.getSizeInBits()/8, 1)) {
    unsigned Align = LD1->getAlignment();
    unsigned NewAlign = DAG.getDataLayout().getABITypeAlignment(
        VT.getTypeForEVT(*DAG.getContext()));

    if (NewAlign <= Align &&
        (!LegalOperations || TLI.isOperationLegal(ISD::LOAD, VT)))
      return DAG.getLoad(VT, SDLoc(N), LD1->getChain(),
                         LD1->getBasePtr(), LD1->getPointerInfo(),
                         false, false, false, Align);
  }

  return SDValue();
}

SDValue DAGCombiner::visitBITCAST(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // If the input is a BUILD_VECTOR with all constant elements, fold this now.
  // Only do this before legalize, since afterward the target may be depending
  // on the bitconvert.
  // First check to see if this is all constant.
  if (!LegalTypes &&
      N0.getOpcode() == ISD::BUILD_VECTOR && N0.getNode()->hasOneUse() &&
      VT.isVector()) {
    bool isSimple = cast<BuildVectorSDNode>(N0)->isConstant();

    EVT DestEltVT = N->getValueType(0).getVectorElementType();
    assert(!DestEltVT.isVector() &&
           "Element type of vector ValueType must not be vector!");
    if (isSimple)
      return ConstantFoldBITCASTofBUILD_VECTOR(N0.getNode(), DestEltVT);
  }

  // If the input is a constant, let getNode fold it.
  if (isa<ConstantSDNode>(N0) || isa<ConstantFPSDNode>(N0)) {
    // If we can't allow illegal operations, we need to check that this is just
    // a fp -> int or int -> conversion and that the resulting operation will
    // be legal.
    if (!LegalOperations ||
        (isa<ConstantSDNode>(N0) && VT.isFloatingPoint() && !VT.isVector() &&
         TLI.isOperationLegal(ISD::ConstantFP, VT)) ||
        (isa<ConstantFPSDNode>(N0) && VT.isInteger() && !VT.isVector() &&
         TLI.isOperationLegal(ISD::Constant, VT)))
      return DAG.getNode(ISD::BITCAST, SDLoc(N), VT, N0);
  }

  // (conv (conv x, t1), t2) -> (conv x, t2)
  if (N0.getOpcode() == ISD::BITCAST)
    return DAG.getNode(ISD::BITCAST, SDLoc(N), VT,
                       N0.getOperand(0));

  // fold (conv (load x)) -> (load (conv*)x)
  // If the resultant load doesn't need a higher alignment than the original!
  if (ISD::isNormalLoad(N0.getNode()) && N0.hasOneUse() &&
      // Do not change the width of a volatile load.
      !cast<LoadSDNode>(N0)->isVolatile() &&
      // Do not remove the cast if the types differ in endian layout.
      TLI.hasBigEndianPartOrdering(N0.getValueType(), DAG.getDataLayout()) ==
          TLI.hasBigEndianPartOrdering(VT, DAG.getDataLayout()) &&
      (!LegalOperations || TLI.isOperationLegal(ISD::LOAD, VT)) &&
      TLI.isLoadBitCastBeneficial(N0.getValueType(), VT)) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);
    unsigned Align = DAG.getDataLayout().getABITypeAlignment(
        VT.getTypeForEVT(*DAG.getContext()));
    unsigned OrigAlign = LN0->getAlignment();

    if (Align <= OrigAlign) {
      SDValue Load = DAG.getLoad(VT, SDLoc(N), LN0->getChain(),
                                 LN0->getBasePtr(), LN0->getPointerInfo(),
                                 LN0->isVolatile(), LN0->isNonTemporal(),
                                 LN0->isInvariant(), OrigAlign,
                                 LN0->getAAInfo());
      DAG.ReplaceAllUsesOfValueWith(N0.getValue(1), Load.getValue(1));
      return Load;
    }
  }

  // fold (bitconvert (fneg x)) -> (xor (bitconvert x), signbit)
  // fold (bitconvert (fabs x)) -> (and (bitconvert x), (not signbit))
  // This often reduces constant pool loads.
  if (((N0.getOpcode() == ISD::FNEG && !TLI.isFNegFree(N0.getValueType())) ||
       (N0.getOpcode() == ISD::FABS && !TLI.isFAbsFree(N0.getValueType()))) &&
      N0.getNode()->hasOneUse() && VT.isInteger() &&
      !VT.isVector() && !N0.getValueType().isVector()) {
    SDValue NewConv = DAG.getNode(ISD::BITCAST, SDLoc(N0), VT,
                                  N0.getOperand(0));
    AddToWorklist(NewConv.getNode());

    SDLoc DL(N);
    APInt SignBit = APInt::getSignBit(VT.getSizeInBits());
    if (N0.getOpcode() == ISD::FNEG)
      return DAG.getNode(ISD::XOR, DL, VT,
                         NewConv, DAG.getConstant(SignBit, DL, VT));
    assert(N0.getOpcode() == ISD::FABS);
    return DAG.getNode(ISD::AND, DL, VT,
                       NewConv, DAG.getConstant(~SignBit, DL, VT));
  }

  // fold (bitconvert (fcopysign cst, x)) ->
  //         (or (and (bitconvert x), sign), (and cst, (not sign)))
  // Note that we don't handle (copysign x, cst) because this can always be
  // folded to an fneg or fabs.
  if (N0.getOpcode() == ISD::FCOPYSIGN && N0.getNode()->hasOneUse() &&
      isa<ConstantFPSDNode>(N0.getOperand(0)) &&
      VT.isInteger() && !VT.isVector()) {
    unsigned OrigXWidth = N0.getOperand(1).getValueType().getSizeInBits();
    EVT IntXVT = EVT::getIntegerVT(*DAG.getContext(), OrigXWidth);
    if (isTypeLegal(IntXVT)) {
      SDValue X = DAG.getNode(ISD::BITCAST, SDLoc(N0),
                              IntXVT, N0.getOperand(1));
      AddToWorklist(X.getNode());

      // If X has a different width than the result/lhs, sext it or truncate it.
      unsigned VTWidth = VT.getSizeInBits();
      if (OrigXWidth < VTWidth) {
        X = DAG.getNode(ISD::SIGN_EXTEND, SDLoc(N), VT, X);
        AddToWorklist(X.getNode());
      } else if (OrigXWidth > VTWidth) {
        // To get the sign bit in the right place, we have to shift it right
        // before truncating.
        SDLoc DL(X);
        X = DAG.getNode(ISD::SRL, DL,
                        X.getValueType(), X,
                        DAG.getConstant(OrigXWidth-VTWidth, DL,
                                        X.getValueType()));
        AddToWorklist(X.getNode());
        X = DAG.getNode(ISD::TRUNCATE, SDLoc(X), VT, X);
        AddToWorklist(X.getNode());
      }

      APInt SignBit = APInt::getSignBit(VT.getSizeInBits());
      X = DAG.getNode(ISD::AND, SDLoc(X), VT,
                      X, DAG.getConstant(SignBit, SDLoc(X), VT));
      AddToWorklist(X.getNode());

      SDValue Cst = DAG.getNode(ISD::BITCAST, SDLoc(N0),
                                VT, N0.getOperand(0));
      Cst = DAG.getNode(ISD::AND, SDLoc(Cst), VT,
                        Cst, DAG.getConstant(~SignBit, SDLoc(Cst), VT));
      AddToWorklist(Cst.getNode());

      return DAG.getNode(ISD::OR, SDLoc(N), VT, X, Cst);
    }
  }

  // bitconvert(build_pair(ld, ld)) -> ld iff load locations are consecutive.
  if (N0.getOpcode() == ISD::BUILD_PAIR) {
    SDValue CombineLD = CombineConsecutiveLoads(N0.getNode(), VT);
    if (CombineLD.getNode())
      return CombineLD;
  }

  // Remove double bitcasts from shuffles - this is often a legacy of
  // XformToShuffleWithZero being used to combine bitmaskings (of
  // float vectors bitcast to integer vectors) into shuffles.
  // bitcast(shuffle(bitcast(s0),bitcast(s1))) -> shuffle(s0,s1)
  if (Level < AfterLegalizeDAG && TLI.isTypeLegal(VT) && VT.isVector() &&
      N0->getOpcode() == ISD::VECTOR_SHUFFLE &&
      VT.getVectorNumElements() >= N0.getValueType().getVectorNumElements() &&
      !(VT.getVectorNumElements() % N0.getValueType().getVectorNumElements())) {
    ShuffleVectorSDNode *SVN = cast<ShuffleVectorSDNode>(N0);

    // If operands are a bitcast, peek through if it casts the original VT.
    // If operands are a UNDEF or constant, just bitcast back to original VT.
    auto PeekThroughBitcast = [&](SDValue Op) {
      if (Op.getOpcode() == ISD::BITCAST &&
          Op.getOperand(0)->getValueType(0) == VT)
        return SDValue(Op.getOperand(0));
      if (ISD::isBuildVectorOfConstantSDNodes(Op.getNode()) ||
          ISD::isBuildVectorOfConstantFPSDNodes(Op.getNode()))
        return DAG.getNode(ISD::BITCAST, SDLoc(N), VT, Op);
      return SDValue();
    };

    SDValue SV0 = PeekThroughBitcast(N0->getOperand(0));
    SDValue SV1 = PeekThroughBitcast(N0->getOperand(1));
    if (!(SV0 && SV1))
      return SDValue();

    int MaskScale =
        VT.getVectorNumElements() / N0.getValueType().getVectorNumElements();
    SmallVector<int, 8> NewMask;
    for (int M : SVN->getMask())
      for (int i = 0; i != MaskScale; ++i)
        NewMask.push_back(M < 0 ? -1 : M * MaskScale + i);

    bool LegalMask = TLI.isShuffleMaskLegal(NewMask, VT);
    if (!LegalMask) {
      std::swap(SV0, SV1);
      ShuffleVectorSDNode::commuteMask(NewMask);
      LegalMask = TLI.isShuffleMaskLegal(NewMask, VT);
    }

    if (LegalMask)
      return DAG.getVectorShuffle(VT, SDLoc(N), SV0, SV1, NewMask);
  }

  return SDValue();
}

SDValue DAGCombiner::visitBUILD_PAIR(SDNode *N) {
  EVT VT = N->getValueType(0);
  return CombineConsecutiveLoads(N, VT);
}

/// We know that BV is a build_vector node with Constant, ConstantFP or Undef
/// operands. DstEltVT indicates the destination element value type.
SDValue DAGCombiner::
ConstantFoldBITCASTofBUILD_VECTOR(SDNode *BV, EVT DstEltVT) {
  EVT SrcEltVT = BV->getValueType(0).getVectorElementType();

  // If this is already the right type, we're done.
  if (SrcEltVT == DstEltVT) return SDValue(BV, 0);

  unsigned SrcBitSize = SrcEltVT.getSizeInBits();
  unsigned DstBitSize = DstEltVT.getSizeInBits();

  // If this is a conversion of N elements of one type to N elements of another
  // type, convert each element.  This handles FP<->INT cases.
  if (SrcBitSize == DstBitSize) {
    EVT VT = EVT::getVectorVT(*DAG.getContext(), DstEltVT,
                              BV->getValueType(0).getVectorNumElements());

    // Due to the FP element handling below calling this routine recursively,
    // we can end up with a scalar-to-vector node here.
    if (BV->getOpcode() == ISD::SCALAR_TO_VECTOR)
      return DAG.getNode(ISD::SCALAR_TO_VECTOR, SDLoc(BV), VT,
                         DAG.getNode(ISD::BITCAST, SDLoc(BV),
                                     DstEltVT, BV->getOperand(0)));

    SmallVector<SDValue, 8> Ops;
    for (SDValue Op : BV->op_values()) {
      // If the vector element type is not legal, the BUILD_VECTOR operands
      // are promoted and implicitly truncated.  Make that explicit here.
      if (Op.getValueType() != SrcEltVT)
        Op = DAG.getNode(ISD::TRUNCATE, SDLoc(BV), SrcEltVT, Op);
      Ops.push_back(DAG.getNode(ISD::BITCAST, SDLoc(BV),
                                DstEltVT, Op));
      AddToWorklist(Ops.back().getNode());
    }
    return DAG.getNode(ISD::BUILD_VECTOR, SDLoc(BV), VT, Ops);
  }

  // Otherwise, we're growing or shrinking the elements.  To avoid having to
  // handle annoying details of growing/shrinking FP values, we convert them to
  // int first.
  if (SrcEltVT.isFloatingPoint()) {
    // Convert the input float vector to a int vector where the elements are the
    // same sizes.
    EVT IntVT = EVT::getIntegerVT(*DAG.getContext(), SrcEltVT.getSizeInBits());
    BV = ConstantFoldBITCASTofBUILD_VECTOR(BV, IntVT).getNode();
    SrcEltVT = IntVT;
  }

  // Now we know the input is an integer vector.  If the output is a FP type,
  // convert to integer first, then to FP of the right size.
  if (DstEltVT.isFloatingPoint()) {
    EVT TmpVT = EVT::getIntegerVT(*DAG.getContext(), DstEltVT.getSizeInBits());
    SDNode *Tmp = ConstantFoldBITCASTofBUILD_VECTOR(BV, TmpVT).getNode();

    // Next, convert to FP elements of the same size.
    return ConstantFoldBITCASTofBUILD_VECTOR(Tmp, DstEltVT);
  }

  SDLoc DL(BV);

  // Okay, we know the src/dst types are both integers of differing types.
  // Handling growing first.
  assert(SrcEltVT.isInteger() && DstEltVT.isInteger());
  if (SrcBitSize < DstBitSize) {
    unsigned NumInputsPerOutput = DstBitSize/SrcBitSize;

    SmallVector<SDValue, 8> Ops;
    for (unsigned i = 0, e = BV->getNumOperands(); i != e;
         i += NumInputsPerOutput) {
      bool isLE = DAG.getDataLayout().isLittleEndian();
      APInt NewBits = APInt(DstBitSize, 0);
      bool EltIsUndef = true;
      for (unsigned j = 0; j != NumInputsPerOutput; ++j) {
        // Shift the previously computed bits over.
        NewBits <<= SrcBitSize;
        SDValue Op = BV->getOperand(i+ (isLE ? (NumInputsPerOutput-j-1) : j));
        if (Op.getOpcode() == ISD::UNDEF) continue;
        EltIsUndef = false;

        NewBits |= cast<ConstantSDNode>(Op)->getAPIntValue().
                   zextOrTrunc(SrcBitSize).zext(DstBitSize);
      }

      if (EltIsUndef)
        Ops.push_back(DAG.getUNDEF(DstEltVT));
      else
        Ops.push_back(DAG.getConstant(NewBits, DL, DstEltVT));
    }

    EVT VT = EVT::getVectorVT(*DAG.getContext(), DstEltVT, Ops.size());
    return DAG.getNode(ISD::BUILD_VECTOR, DL, VT, Ops);
  }

  // Finally, this must be the case where we are shrinking elements: each input
  // turns into multiple outputs.
  unsigned NumOutputsPerInput = SrcBitSize/DstBitSize;
  EVT VT = EVT::getVectorVT(*DAG.getContext(), DstEltVT,
                            NumOutputsPerInput*BV->getNumOperands());
  SmallVector<SDValue, 8> Ops;

  for (const SDValue &Op : BV->op_values()) {
    if (Op.getOpcode() == ISD::UNDEF) {
      Ops.append(NumOutputsPerInput, DAG.getUNDEF(DstEltVT));
      continue;
    }

    APInt OpVal = cast<ConstantSDNode>(Op)->
                  getAPIntValue().zextOrTrunc(SrcBitSize);

    for (unsigned j = 0; j != NumOutputsPerInput; ++j) {
      APInt ThisVal = OpVal.trunc(DstBitSize);
      Ops.push_back(DAG.getConstant(ThisVal, DL, DstEltVT));
      OpVal = OpVal.lshr(DstBitSize);
    }

    // For big endian targets, swap the order of the pieces of each element.
    if (DAG.getDataLayout().isBigEndian())
      std::reverse(Ops.end()-NumOutputsPerInput, Ops.end());
  }

  return DAG.getNode(ISD::BUILD_VECTOR, DL, VT, Ops);
}

/// Try to perform FMA combining on a given FADD node.
SDValue DAGCombiner::visitFADDForFMACombine(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N->getValueType(0);
  SDLoc SL(N);

  const TargetOptions &Options = DAG.getTarget().Options;
  bool UnsafeFPMath = (Options.AllowFPOpFusion == FPOpFusion::Fast ||
                       Options.UnsafeFPMath);

  // Floating-point multiply-add with intermediate rounding.
  bool HasFMAD = (LegalOperations &&
                  TLI.isOperationLegal(ISD::FMAD, VT));

  // Floating-point multiply-add without intermediate rounding.
  bool HasFMA = ((!LegalOperations ||
                  TLI.isOperationLegalOrCustom(ISD::FMA, VT)) &&
                 TLI.isFMAFasterThanFMulAndFAdd(VT) &&
                 UnsafeFPMath);

  // No valid opcode, do not combine.
  if (!HasFMAD && !HasFMA)
    return SDValue();

  // Always prefer FMAD to FMA for precision.
  unsigned int PreferredFusedOpcode = HasFMAD ? ISD::FMAD : ISD::FMA;
  bool Aggressive = TLI.enableAggressiveFMAFusion(VT);
  bool LookThroughFPExt = TLI.isFPExtFree(VT);

  // fold (fadd (fmul x, y), z) -> (fma x, y, z)
  if (N0.getOpcode() == ISD::FMUL &&
      (Aggressive || N0->hasOneUse())) {
    return DAG.getNode(PreferredFusedOpcode, SL, VT,
                       N0.getOperand(0), N0.getOperand(1), N1);
  }

  // fold (fadd x, (fmul y, z)) -> (fma y, z, x)
  // Note: Commutes FADD operands.
  if (N1.getOpcode() == ISD::FMUL &&
      (Aggressive || N1->hasOneUse())) {
    return DAG.getNode(PreferredFusedOpcode, SL, VT,
                       N1.getOperand(0), N1.getOperand(1), N0);
  }

  // Look through FP_EXTEND nodes to do more combining.
  if (UnsafeFPMath && LookThroughFPExt) {
    // fold (fadd (fpext (fmul x, y)), z) -> (fma (fpext x), (fpext y), z)
    if (N0.getOpcode() == ISD::FP_EXTEND) {
      SDValue N00 = N0.getOperand(0);
      if (N00.getOpcode() == ISD::FMUL)
        return DAG.getNode(PreferredFusedOpcode, SL, VT,
                           DAG.getNode(ISD::FP_EXTEND, SL, VT,
                                       N00.getOperand(0)),
                           DAG.getNode(ISD::FP_EXTEND, SL, VT,
                                       N00.getOperand(1)), N1);
    }

    // fold (fadd x, (fpext (fmul y, z))) -> (fma (fpext y), (fpext z), x)
    // Note: Commutes FADD operands.
    if (N1.getOpcode() == ISD::FP_EXTEND) {
      SDValue N10 = N1.getOperand(0);
      if (N10.getOpcode() == ISD::FMUL)
        return DAG.getNode(PreferredFusedOpcode, SL, VT,
                           DAG.getNode(ISD::FP_EXTEND, SL, VT,
                                       N10.getOperand(0)),
                           DAG.getNode(ISD::FP_EXTEND, SL, VT,
                                       N10.getOperand(1)), N0);
    }
  }

  // More folding opportunities when target permits.
  if ((UnsafeFPMath || HasFMAD)  && Aggressive) {
    // fold (fadd (fma x, y, (fmul u, v)), z) -> (fma x, y (fma u, v, z))
    if (N0.getOpcode() == PreferredFusedOpcode &&
        N0.getOperand(2).getOpcode() == ISD::FMUL) {
      return DAG.getNode(PreferredFusedOpcode, SL, VT,
                         N0.getOperand(0), N0.getOperand(1),
                         DAG.getNode(PreferredFusedOpcode, SL, VT,
                                     N0.getOperand(2).getOperand(0),
                                     N0.getOperand(2).getOperand(1),
                                     N1));
    }

    // fold (fadd x, (fma y, z, (fmul u, v)) -> (fma y, z (fma u, v, x))
    if (N1->getOpcode() == PreferredFusedOpcode &&
        N1.getOperand(2).getOpcode() == ISD::FMUL) {
      return DAG.getNode(PreferredFusedOpcode, SL, VT,
                         N1.getOperand(0), N1.getOperand(1),
                         DAG.getNode(PreferredFusedOpcode, SL, VT,
                                     N1.getOperand(2).getOperand(0),
                                     N1.getOperand(2).getOperand(1),
                                     N0));
    }

    if (UnsafeFPMath && LookThroughFPExt) {
      // fold (fadd (fma x, y, (fpext (fmul u, v))), z)
      //   -> (fma x, y, (fma (fpext u), (fpext v), z))
      auto FoldFAddFMAFPExtFMul = [&] (
          SDValue X, SDValue Y, SDValue U, SDValue V, SDValue Z) {
        return DAG.getNode(PreferredFusedOpcode, SL, VT, X, Y,
                           DAG.getNode(PreferredFusedOpcode, SL, VT,
                                       DAG.getNode(ISD::FP_EXTEND, SL, VT, U),
                                       DAG.getNode(ISD::FP_EXTEND, SL, VT, V),
                                       Z));
      };
      if (N0.getOpcode() == PreferredFusedOpcode) {
        SDValue N02 = N0.getOperand(2);
        if (N02.getOpcode() == ISD::FP_EXTEND) {
          SDValue N020 = N02.getOperand(0);
          if (N020.getOpcode() == ISD::FMUL)
            return FoldFAddFMAFPExtFMul(N0.getOperand(0), N0.getOperand(1),
                                        N020.getOperand(0), N020.getOperand(1),
                                        N1);
        }
      }

      // fold (fadd (fpext (fma x, y, (fmul u, v))), z)
      //   -> (fma (fpext x), (fpext y), (fma (fpext u), (fpext v), z))
      // FIXME: This turns two single-precision and one double-precision
      // operation into two double-precision operations, which might not be
      // interesting for all targets, especially GPUs.
      auto FoldFAddFPExtFMAFMul = [&] (
          SDValue X, SDValue Y, SDValue U, SDValue V, SDValue Z) {
        return DAG.getNode(PreferredFusedOpcode, SL, VT,
                           DAG.getNode(ISD::FP_EXTEND, SL, VT, X),
                           DAG.getNode(ISD::FP_EXTEND, SL, VT, Y),
                           DAG.getNode(PreferredFusedOpcode, SL, VT,
                                       DAG.getNode(ISD::FP_EXTEND, SL, VT, U),
                                       DAG.getNode(ISD::FP_EXTEND, SL, VT, V),
                                       Z));
      };
      if (N0.getOpcode() == ISD::FP_EXTEND) {
        SDValue N00 = N0.getOperand(0);
        if (N00.getOpcode() == PreferredFusedOpcode) {
          SDValue N002 = N00.getOperand(2);
          if (N002.getOpcode() == ISD::FMUL)
            return FoldFAddFPExtFMAFMul(N00.getOperand(0), N00.getOperand(1),
                                        N002.getOperand(0), N002.getOperand(1),
                                        N1);
        }
      }

      // fold (fadd x, (fma y, z, (fpext (fmul u, v)))
      //   -> (fma y, z, (fma (fpext u), (fpext v), x))
      if (N1.getOpcode() == PreferredFusedOpcode) {
        SDValue N12 = N1.getOperand(2);
        if (N12.getOpcode() == ISD::FP_EXTEND) {
          SDValue N120 = N12.getOperand(0);
          if (N120.getOpcode() == ISD::FMUL)
            return FoldFAddFMAFPExtFMul(N1.getOperand(0), N1.getOperand(1),
                                        N120.getOperand(0), N120.getOperand(1),
                                        N0);
        }
      }

      // fold (fadd x, (fpext (fma y, z, (fmul u, v)))
      //   -> (fma (fpext y), (fpext z), (fma (fpext u), (fpext v), x))
      // FIXME: This turns two single-precision and one double-precision
      // operation into two double-precision operations, which might not be
      // interesting for all targets, especially GPUs.
      if (N1.getOpcode() == ISD::FP_EXTEND) {
        SDValue N10 = N1.getOperand(0);
        if (N10.getOpcode() == PreferredFusedOpcode) {
          SDValue N102 = N10.getOperand(2);
          if (N102.getOpcode() == ISD::FMUL)
            return FoldFAddFPExtFMAFMul(N10.getOperand(0), N10.getOperand(1),
                                        N102.getOperand(0), N102.getOperand(1),
                                        N0);
        }
      }
    }
  }

  return SDValue();
}

/// Try to perform FMA combining on a given FSUB node.
SDValue DAGCombiner::visitFSUBForFMACombine(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N->getValueType(0);
  SDLoc SL(N);

  const TargetOptions &Options = DAG.getTarget().Options;
  bool UnsafeFPMath = (Options.AllowFPOpFusion == FPOpFusion::Fast ||
                       Options.UnsafeFPMath);

  // Floating-point multiply-add with intermediate rounding.
  bool HasFMAD = (LegalOperations &&
                  TLI.isOperationLegal(ISD::FMAD, VT));

  // Floating-point multiply-add without intermediate rounding.
  bool HasFMA = ((!LegalOperations ||
                  TLI.isOperationLegalOrCustom(ISD::FMA, VT)) &&
                 TLI.isFMAFasterThanFMulAndFAdd(VT) &&
                 UnsafeFPMath);

  // No valid opcode, do not combine.
  if (!HasFMAD && !HasFMA)
    return SDValue();

  // Always prefer FMAD to FMA for precision.
  unsigned int PreferredFusedOpcode = HasFMAD ? ISD::FMAD : ISD::FMA;
  bool Aggressive = TLI.enableAggressiveFMAFusion(VT);
  bool LookThroughFPExt = TLI.isFPExtFree(VT);

  // fold (fsub (fmul x, y), z) -> (fma x, y, (fneg z))
  if (N0.getOpcode() == ISD::FMUL &&
      (Aggressive || N0->hasOneUse())) {
    return DAG.getNode(PreferredFusedOpcode, SL, VT,
                       N0.getOperand(0), N0.getOperand(1),
                       DAG.getNode(ISD::FNEG, SL, VT, N1));
  }

  // fold (fsub x, (fmul y, z)) -> (fma (fneg y), z, x)
  // Note: Commutes FSUB operands.
  if (N1.getOpcode() == ISD::FMUL &&
      (Aggressive || N1->hasOneUse()))
    return DAG.getNode(PreferredFusedOpcode, SL, VT,
                       DAG.getNode(ISD::FNEG, SL, VT,
                                   N1.getOperand(0)),
                       N1.getOperand(1), N0);

  // fold (fsub (fneg (fmul, x, y)), z) -> (fma (fneg x), y, (fneg z))
  if (N0.getOpcode() == ISD::FNEG &&
      N0.getOperand(0).getOpcode() == ISD::FMUL &&
      (Aggressive || (N0->hasOneUse() && N0.getOperand(0).hasOneUse()))) {
    SDValue N00 = N0.getOperand(0).getOperand(0);
    SDValue N01 = N0.getOperand(0).getOperand(1);
    return DAG.getNode(PreferredFusedOpcode, SL, VT,
                       DAG.getNode(ISD::FNEG, SL, VT, N00), N01,
                       DAG.getNode(ISD::FNEG, SL, VT, N1));
  }

  // Look through FP_EXTEND nodes to do more combining.
  if (UnsafeFPMath && LookThroughFPExt) {
    // fold (fsub (fpext (fmul x, y)), z)
    //   -> (fma (fpext x), (fpext y), (fneg z))
    if (N0.getOpcode() == ISD::FP_EXTEND) {
      SDValue N00 = N0.getOperand(0);
      if (N00.getOpcode() == ISD::FMUL)
        return DAG.getNode(PreferredFusedOpcode, SL, VT,
                           DAG.getNode(ISD::FP_EXTEND, SL, VT,
                                       N00.getOperand(0)),
                           DAG.getNode(ISD::FP_EXTEND, SL, VT,
                                       N00.getOperand(1)),
                           DAG.getNode(ISD::FNEG, SL, VT, N1));
    }

    // fold (fsub x, (fpext (fmul y, z)))
    //   -> (fma (fneg (fpext y)), (fpext z), x)
    // Note: Commutes FSUB operands.
    if (N1.getOpcode() == ISD::FP_EXTEND) {
      SDValue N10 = N1.getOperand(0);
      if (N10.getOpcode() == ISD::FMUL)
        return DAG.getNode(PreferredFusedOpcode, SL, VT,
                           DAG.getNode(ISD::FNEG, SL, VT,
                                       DAG.getNode(ISD::FP_EXTEND, SL, VT,
                                                   N10.getOperand(0))),
                           DAG.getNode(ISD::FP_EXTEND, SL, VT,
                                       N10.getOperand(1)),
                           N0);
    }

    // fold (fsub (fpext (fneg (fmul, x, y))), z)
    //   -> (fneg (fma (fpext x), (fpext y), z))
    // Note: This could be removed with appropriate canonicalization of the
    // input expression into (fneg (fadd (fpext (fmul, x, y)), z). However, the
    // orthogonal flags -fp-contract=fast and -enable-unsafe-fp-math prevent
    // from implementing the canonicalization in visitFSUB.
    if (N0.getOpcode() == ISD::FP_EXTEND) {
      SDValue N00 = N0.getOperand(0);
      if (N00.getOpcode() == ISD::FNEG) {
        SDValue N000 = N00.getOperand(0);
        if (N000.getOpcode() == ISD::FMUL) {
          return DAG.getNode(ISD::FNEG, SL, VT,
                             DAG.getNode(PreferredFusedOpcode, SL, VT,
                                         DAG.getNode(ISD::FP_EXTEND, SL, VT,
                                                     N000.getOperand(0)),
                                         DAG.getNode(ISD::FP_EXTEND, SL, VT,
                                                     N000.getOperand(1)),
                                         N1));
        }
      }
    }

    // fold (fsub (fneg (fpext (fmul, x, y))), z)
    //   -> (fneg (fma (fpext x)), (fpext y), z)
    // Note: This could be removed with appropriate canonicalization of the
    // input expression into (fneg (fadd (fpext (fmul, x, y)), z). However, the
    // orthogonal flags -fp-contract=fast and -enable-unsafe-fp-math prevent
    // from implementing the canonicalization in visitFSUB.
    if (N0.getOpcode() == ISD::FNEG) {
      SDValue N00 = N0.getOperand(0);
      if (N00.getOpcode() == ISD::FP_EXTEND) {
        SDValue N000 = N00.getOperand(0);
        if (N000.getOpcode() == ISD::FMUL) {
          return DAG.getNode(ISD::FNEG, SL, VT,
                             DAG.getNode(PreferredFusedOpcode, SL, VT,
                                         DAG.getNode(ISD::FP_EXTEND, SL, VT,
                                                     N000.getOperand(0)),
                                         DAG.getNode(ISD::FP_EXTEND, SL, VT,
                                                     N000.getOperand(1)),
                                         N1));
        }
      }
    }

  }

  // More folding opportunities when target permits.
  if ((UnsafeFPMath || HasFMAD) && Aggressive) {
    // fold (fsub (fma x, y, (fmul u, v)), z)
    //   -> (fma x, y (fma u, v, (fneg z)))
    if (N0.getOpcode() == PreferredFusedOpcode &&
        N0.getOperand(2).getOpcode() == ISD::FMUL) {
      return DAG.getNode(PreferredFusedOpcode, SL, VT,
                         N0.getOperand(0), N0.getOperand(1),
                         DAG.getNode(PreferredFusedOpcode, SL, VT,
                                     N0.getOperand(2).getOperand(0),
                                     N0.getOperand(2).getOperand(1),
                                     DAG.getNode(ISD::FNEG, SL, VT,
                                                 N1)));
    }

    // fold (fsub x, (fma y, z, (fmul u, v)))
    //   -> (fma (fneg y), z, (fma (fneg u), v, x))
    if (N1.getOpcode() == PreferredFusedOpcode &&
        N1.getOperand(2).getOpcode() == ISD::FMUL) {
      SDValue N20 = N1.getOperand(2).getOperand(0);
      SDValue N21 = N1.getOperand(2).getOperand(1);
      return DAG.getNode(PreferredFusedOpcode, SL, VT,
                         DAG.getNode(ISD::FNEG, SL, VT,
                                     N1.getOperand(0)),
                         N1.getOperand(1),
                         DAG.getNode(PreferredFusedOpcode, SL, VT,
                                     DAG.getNode(ISD::FNEG, SL, VT, N20),

                                     N21, N0));
    }

    if (UnsafeFPMath && LookThroughFPExt) {
      // fold (fsub (fma x, y, (fpext (fmul u, v))), z)
      //   -> (fma x, y (fma (fpext u), (fpext v), (fneg z)))
      if (N0.getOpcode() == PreferredFusedOpcode) {
        SDValue N02 = N0.getOperand(2);
        if (N02.getOpcode() == ISD::FP_EXTEND) {
          SDValue N020 = N02.getOperand(0);
          if (N020.getOpcode() == ISD::FMUL)
            return DAG.getNode(PreferredFusedOpcode, SL, VT,
                               N0.getOperand(0), N0.getOperand(1),
                               DAG.getNode(PreferredFusedOpcode, SL, VT,
                                           DAG.getNode(ISD::FP_EXTEND, SL, VT,
                                                       N020.getOperand(0)),
                                           DAG.getNode(ISD::FP_EXTEND, SL, VT,
                                                       N020.getOperand(1)),
                                           DAG.getNode(ISD::FNEG, SL, VT,
                                                       N1)));
        }
      }

      // fold (fsub (fpext (fma x, y, (fmul u, v))), z)
      //   -> (fma (fpext x), (fpext y),
      //           (fma (fpext u), (fpext v), (fneg z)))
      // FIXME: This turns two single-precision and one double-precision
      // operation into two double-precision operations, which might not be
      // interesting for all targets, especially GPUs.
      if (N0.getOpcode() == ISD::FP_EXTEND) {
        SDValue N00 = N0.getOperand(0);
        if (N00.getOpcode() == PreferredFusedOpcode) {
          SDValue N002 = N00.getOperand(2);
          if (N002.getOpcode() == ISD::FMUL)
            return DAG.getNode(PreferredFusedOpcode, SL, VT,
                               DAG.getNode(ISD::FP_EXTEND, SL, VT,
                                           N00.getOperand(0)),
                               DAG.getNode(ISD::FP_EXTEND, SL, VT,
                                           N00.getOperand(1)),
                               DAG.getNode(PreferredFusedOpcode, SL, VT,
                                           DAG.getNode(ISD::FP_EXTEND, SL, VT,
                                                       N002.getOperand(0)),
                                           DAG.getNode(ISD::FP_EXTEND, SL, VT,
                                                       N002.getOperand(1)),
                                           DAG.getNode(ISD::FNEG, SL, VT,
                                                       N1)));
        }
      }

      // fold (fsub x, (fma y, z, (fpext (fmul u, v))))
      //   -> (fma (fneg y), z, (fma (fneg (fpext u)), (fpext v), x))
      if (N1.getOpcode() == PreferredFusedOpcode &&
        N1.getOperand(2).getOpcode() == ISD::FP_EXTEND) {
        SDValue N120 = N1.getOperand(2).getOperand(0);
        if (N120.getOpcode() == ISD::FMUL) {
          SDValue N1200 = N120.getOperand(0);
          SDValue N1201 = N120.getOperand(1);
          return DAG.getNode(PreferredFusedOpcode, SL, VT,
                             DAG.getNode(ISD::FNEG, SL, VT, N1.getOperand(0)),
                             N1.getOperand(1),
                             DAG.getNode(PreferredFusedOpcode, SL, VT,
                                         DAG.getNode(ISD::FNEG, SL, VT,
                                             DAG.getNode(ISD::FP_EXTEND, SL,
                                                         VT, N1200)),
                                         DAG.getNode(ISD::FP_EXTEND, SL, VT,
                                                     N1201),
                                         N0));
        }
      }

      // fold (fsub x, (fpext (fma y, z, (fmul u, v))))
      //   -> (fma (fneg (fpext y)), (fpext z),
      //           (fma (fneg (fpext u)), (fpext v), x))
      // FIXME: This turns two single-precision and one double-precision
      // operation into two double-precision operations, which might not be
      // interesting for all targets, especially GPUs.
      if (N1.getOpcode() == ISD::FP_EXTEND &&
        N1.getOperand(0).getOpcode() == PreferredFusedOpcode) {
        SDValue N100 = N1.getOperand(0).getOperand(0);
        SDValue N101 = N1.getOperand(0).getOperand(1);
        SDValue N102 = N1.getOperand(0).getOperand(2);
        if (N102.getOpcode() == ISD::FMUL) {
          SDValue N1020 = N102.getOperand(0);
          SDValue N1021 = N102.getOperand(1);
          return DAG.getNode(PreferredFusedOpcode, SL, VT,
                             DAG.getNode(ISD::FNEG, SL, VT,
                                         DAG.getNode(ISD::FP_EXTEND, SL, VT,
                                                     N100)),
                             DAG.getNode(ISD::FP_EXTEND, SL, VT, N101),
                             DAG.getNode(PreferredFusedOpcode, SL, VT,
                                         DAG.getNode(ISD::FNEG, SL, VT,
                                             DAG.getNode(ISD::FP_EXTEND, SL,
                                                         VT, N1020)),
                                         DAG.getNode(ISD::FP_EXTEND, SL, VT,
                                                     N1021),
                                         N0));
        }
      }
    }
  }

  return SDValue();
}

SDValue DAGCombiner::visitFADD(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1);
  EVT VT = N->getValueType(0);
  SDLoc DL(N);
  const TargetOptions &Options = DAG.getTarget().Options;

  // fold vector ops
  if (VT.isVector())
    if (SDValue FoldedVOp = SimplifyVBinOp(N))
      return FoldedVOp;

  // fold (fadd c1, c2) -> c1 + c2
  if (N0CFP && N1CFP)
    return DAG.getNode(ISD::FADD, DL, VT, N0, N1);

  // canonicalize constant to RHS
  if (N0CFP && !N1CFP)
    return DAG.getNode(ISD::FADD, DL, VT, N1, N0);

  // fold (fadd A, (fneg B)) -> (fsub A, B)
  if ((!LegalOperations || TLI.isOperationLegalOrCustom(ISD::FSUB, VT)) &&
      isNegatibleForFree(N1, LegalOperations, TLI, &Options) == 2)
    return DAG.getNode(ISD::FSUB, DL, VT, N0,
                       GetNegatedExpression(N1, DAG, LegalOperations));

  // fold (fadd (fneg A), B) -> (fsub B, A)
  if ((!LegalOperations || TLI.isOperationLegalOrCustom(ISD::FSUB, VT)) &&
      isNegatibleForFree(N0, LegalOperations, TLI, &Options) == 2)
    return DAG.getNode(ISD::FSUB, DL, VT, N1,
                       GetNegatedExpression(N0, DAG, LegalOperations));

  // If 'unsafe math' is enabled, fold lots of things.
  if (Options.UnsafeFPMath) {
    // No FP constant should be created after legalization as Instruction
    // Selection pass has a hard time dealing with FP constants.
    bool AllowNewConst = (Level < AfterLegalizeDAG);

    // fold (fadd A, 0) -> A
    if (N1CFP && N1CFP->isZero())
      return N0;

    // fold (fadd (fadd x, c1), c2) -> (fadd x, (fadd c1, c2))
    if (N1CFP && N0.getOpcode() == ISD::FADD && N0.getNode()->hasOneUse() &&
        isa<ConstantFPSDNode>(N0.getOperand(1)))
      return DAG.getNode(ISD::FADD, DL, VT, N0.getOperand(0),
                         DAG.getNode(ISD::FADD, DL, VT, N0.getOperand(1), N1));

    // If allowed, fold (fadd (fneg x), x) -> 0.0
    if (AllowNewConst && N0.getOpcode() == ISD::FNEG && N0.getOperand(0) == N1)
      return DAG.getConstantFP(0.0, DL, VT);

    // If allowed, fold (fadd x, (fneg x)) -> 0.0
    if (AllowNewConst && N1.getOpcode() == ISD::FNEG && N1.getOperand(0) == N0)
      return DAG.getConstantFP(0.0, DL, VT);

    // We can fold chains of FADD's of the same value into multiplications.
    // This transform is not safe in general because we are reducing the number
    // of rounding steps.
    if (TLI.isOperationLegalOrCustom(ISD::FMUL, VT) && !N0CFP && !N1CFP) {
      if (N0.getOpcode() == ISD::FMUL) {
        ConstantFPSDNode *CFP00 = dyn_cast<ConstantFPSDNode>(N0.getOperand(0));
        ConstantFPSDNode *CFP01 = dyn_cast<ConstantFPSDNode>(N0.getOperand(1));

        // (fadd (fmul x, c), x) -> (fmul x, c+1)
        if (CFP01 && !CFP00 && N0.getOperand(0) == N1) {
          SDValue NewCFP = DAG.getNode(ISD::FADD, DL, VT, SDValue(CFP01, 0),
                                       DAG.getConstantFP(1.0, DL, VT));
          return DAG.getNode(ISD::FMUL, DL, VT, N1, NewCFP);
        }

        // (fadd (fmul x, c), (fadd x, x)) -> (fmul x, c+2)
        if (CFP01 && !CFP00 && N1.getOpcode() == ISD::FADD &&
            N1.getOperand(0) == N1.getOperand(1) &&
            N0.getOperand(0) == N1.getOperand(0)) {
          SDValue NewCFP = DAG.getNode(ISD::FADD, DL, VT, SDValue(CFP01, 0),
                                       DAG.getConstantFP(2.0, DL, VT));
          return DAG.getNode(ISD::FMUL, DL, VT, N0.getOperand(0), NewCFP);
        }
      }

      if (N1.getOpcode() == ISD::FMUL) {
        ConstantFPSDNode *CFP10 = dyn_cast<ConstantFPSDNode>(N1.getOperand(0));
        ConstantFPSDNode *CFP11 = dyn_cast<ConstantFPSDNode>(N1.getOperand(1));

        // (fadd x, (fmul x, c)) -> (fmul x, c+1)
        if (CFP11 && !CFP10 && N1.getOperand(0) == N0) {
          SDValue NewCFP = DAG.getNode(ISD::FADD, DL, VT, SDValue(CFP11, 0),
                                       DAG.getConstantFP(1.0, DL, VT));
          return DAG.getNode(ISD::FMUL, DL, VT, N0, NewCFP);
        }

        // (fadd (fadd x, x), (fmul x, c)) -> (fmul x, c+2)
        if (CFP11 && !CFP10 && N0.getOpcode() == ISD::FADD &&
            N0.getOperand(0) == N0.getOperand(1) &&
            N1.getOperand(0) == N0.getOperand(0)) {
          SDValue NewCFP = DAG.getNode(ISD::FADD, DL, VT, SDValue(CFP11, 0),
                                       DAG.getConstantFP(2.0, DL, VT));
          return DAG.getNode(ISD::FMUL, DL, VT, N1.getOperand(0), NewCFP);
        }
      }

      if (N0.getOpcode() == ISD::FADD && AllowNewConst) {
        ConstantFPSDNode *CFP = dyn_cast<ConstantFPSDNode>(N0.getOperand(0));
        // (fadd (fadd x, x), x) -> (fmul x, 3.0)
        if (!CFP && N0.getOperand(0) == N0.getOperand(1) &&
            (N0.getOperand(0) == N1)) {
          return DAG.getNode(ISD::FMUL, DL, VT,
                             N1, DAG.getConstantFP(3.0, DL, VT));
        }
      }

      if (N1.getOpcode() == ISD::FADD && AllowNewConst) {
        ConstantFPSDNode *CFP10 = dyn_cast<ConstantFPSDNode>(N1.getOperand(0));
        // (fadd x, (fadd x, x)) -> (fmul x, 3.0)
        if (!CFP10 && N1.getOperand(0) == N1.getOperand(1) &&
            N1.getOperand(0) == N0) {
          return DAG.getNode(ISD::FMUL, DL, VT,
                             N0, DAG.getConstantFP(3.0, DL, VT));
        }
      }

      // (fadd (fadd x, x), (fadd x, x)) -> (fmul x, 4.0)
      if (AllowNewConst &&
          N0.getOpcode() == ISD::FADD && N1.getOpcode() == ISD::FADD &&
          N0.getOperand(0) == N0.getOperand(1) &&
          N1.getOperand(0) == N1.getOperand(1) &&
          N0.getOperand(0) == N1.getOperand(0)) {
        return DAG.getNode(ISD::FMUL, DL, VT,
                           N0.getOperand(0), DAG.getConstantFP(4.0, DL, VT));
      }
    }
  } // enable-unsafe-fp-math

  // FADD -> FMA combines:
  SDValue Fused = visitFADDForFMACombine(N);
  if (Fused) {
    AddToWorklist(Fused.getNode());
    return Fused;
  }

  return SDValue();
}

SDValue DAGCombiner::visitFSUB(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantFPSDNode *N0CFP = isConstOrConstSplatFP(N0);
  ConstantFPSDNode *N1CFP = isConstOrConstSplatFP(N1);
  EVT VT = N->getValueType(0);
  SDLoc dl(N);
  const TargetOptions &Options = DAG.getTarget().Options;

  // fold vector ops
  if (VT.isVector())
    if (SDValue FoldedVOp = SimplifyVBinOp(N))
      return FoldedVOp;

  // fold (fsub c1, c2) -> c1-c2
  if (N0CFP && N1CFP)
    return DAG.getNode(ISD::FSUB, dl, VT, N0, N1);

  // fold (fsub A, (fneg B)) -> (fadd A, B)
  if (isNegatibleForFree(N1, LegalOperations, TLI, &Options))
    return DAG.getNode(ISD::FADD, dl, VT, N0,
                       GetNegatedExpression(N1, DAG, LegalOperations));

  // If 'unsafe math' is enabled, fold lots of things.
  if (Options.UnsafeFPMath) {
    // (fsub A, 0) -> A
    if (N1CFP && N1CFP->isZero())
      return N0;

    // (fsub 0, B) -> -B
    if (N0CFP && N0CFP->isZero()) {
      if (isNegatibleForFree(N1, LegalOperations, TLI, &Options))
        return GetNegatedExpression(N1, DAG, LegalOperations);
      if (!LegalOperations || TLI.isOperationLegal(ISD::FNEG, VT))
        return DAG.getNode(ISD::FNEG, dl, VT, N1);
    }

    // (fsub x, x) -> 0.0
    if (N0 == N1)
      return DAG.getConstantFP(0.0f, dl, VT);

    // (fsub x, (fadd x, y)) -> (fneg y)
    // (fsub x, (fadd y, x)) -> (fneg y)
    if (N1.getOpcode() == ISD::FADD) {
      SDValue N10 = N1->getOperand(0);
      SDValue N11 = N1->getOperand(1);

      if (N10 == N0 && isNegatibleForFree(N11, LegalOperations, TLI, &Options))
        return GetNegatedExpression(N11, DAG, LegalOperations);

      if (N11 == N0 && isNegatibleForFree(N10, LegalOperations, TLI, &Options))
        return GetNegatedExpression(N10, DAG, LegalOperations);
    }
  }

  // FSUB -> FMA combines:
  SDValue Fused = visitFSUBForFMACombine(N);
  if (Fused) {
    AddToWorklist(Fused.getNode());
    return Fused;
  }

  return SDValue();
}

SDValue DAGCombiner::visitFMUL(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantFPSDNode *N0CFP = isConstOrConstSplatFP(N0);
  ConstantFPSDNode *N1CFP = isConstOrConstSplatFP(N1);
  EVT VT = N->getValueType(0);
  SDLoc DL(N);
  const TargetOptions &Options = DAG.getTarget().Options;

  // fold vector ops
  if (VT.isVector()) {
    // This just handles C1 * C2 for vectors. Other vector folds are below.
    if (SDValue FoldedVOp = SimplifyVBinOp(N))
      return FoldedVOp;
  }

  // fold (fmul c1, c2) -> c1*c2
  if (N0CFP && N1CFP)
    return DAG.getNode(ISD::FMUL, DL, VT, N0, N1);

  // canonicalize constant to RHS
  if (isConstantFPBuildVectorOrConstantFP(N0) &&
     !isConstantFPBuildVectorOrConstantFP(N1))
    return DAG.getNode(ISD::FMUL, DL, VT, N1, N0);

  // fold (fmul A, 1.0) -> A
  if (N1CFP && N1CFP->isExactlyValue(1.0))
    return N0;

  if (Options.UnsafeFPMath) {
    // fold (fmul A, 0) -> 0
    if (N1CFP && N1CFP->isZero())
      return N1;

    // fold (fmul (fmul x, c1), c2) -> (fmul x, (fmul c1, c2))
    if (N0.getOpcode() == ISD::FMUL) {
      // Fold scalars or any vector constants (not just splats).
      // This fold is done in general by InstCombine, but extra fmul insts
      // may have been generated during lowering.
      SDValue N00 = N0.getOperand(0);
      SDValue N01 = N0.getOperand(1);
      auto *BV1 = dyn_cast<BuildVectorSDNode>(N1);
      auto *BV00 = dyn_cast<BuildVectorSDNode>(N00);
      auto *BV01 = dyn_cast<BuildVectorSDNode>(N01);

      // Check 1: Make sure that the first operand of the inner multiply is NOT
      // a constant. Otherwise, we may induce infinite looping.
      if (!(isConstOrConstSplatFP(N00) || (BV00 && BV00->isConstant()))) {
        // Check 2: Make sure that the second operand of the inner multiply and
        // the second operand of the outer multiply are constants.
        if ((N1CFP && isConstOrConstSplatFP(N01)) ||
            (BV1 && BV01 && BV1->isConstant() && BV01->isConstant())) {
          SDValue MulConsts = DAG.getNode(ISD::FMUL, DL, VT, N01, N1);
          return DAG.getNode(ISD::FMUL, DL, VT, N00, MulConsts);
        }
      }
    }

    // fold (fmul (fadd x, x), c) -> (fmul x, (fmul 2.0, c))
    // Undo the fmul 2.0, x -> fadd x, x transformation, since if it occurs
    // during an early run of DAGCombiner can prevent folding with fmuls
    // inserted during lowering.
    if (N0.getOpcode() == ISD::FADD && N0.getOperand(0) == N0.getOperand(1)) {
      const SDValue Two = DAG.getConstantFP(2.0, DL, VT);
      SDValue MulConsts = DAG.getNode(ISD::FMUL, DL, VT, Two, N1);
      return DAG.getNode(ISD::FMUL, DL, VT, N0.getOperand(0), MulConsts);
    }
  }

  // fold (fmul X, 2.0) -> (fadd X, X)
  if (N1CFP && N1CFP->isExactlyValue(+2.0))
    return DAG.getNode(ISD::FADD, DL, VT, N0, N0);

  // fold (fmul X, -1.0) -> (fneg X)
  if (N1CFP && N1CFP->isExactlyValue(-1.0))
    if (!LegalOperations || TLI.isOperationLegal(ISD::FNEG, VT))
      return DAG.getNode(ISD::FNEG, DL, VT, N0);

  // fold (fmul (fneg X), (fneg Y)) -> (fmul X, Y)
  if (char LHSNeg = isNegatibleForFree(N0, LegalOperations, TLI, &Options)) {
    if (char RHSNeg = isNegatibleForFree(N1, LegalOperations, TLI, &Options)) {
      // Both can be negated for free, check to see if at least one is cheaper
      // negated.
      if (LHSNeg == 2 || RHSNeg == 2)
        return DAG.getNode(ISD::FMUL, DL, VT,
                           GetNegatedExpression(N0, DAG, LegalOperations),
                           GetNegatedExpression(N1, DAG, LegalOperations));
    }
  }

  return SDValue();
}

SDValue DAGCombiner::visitFMA(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  SDValue N2 = N->getOperand(2);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1);
  EVT VT = N->getValueType(0);
  SDLoc dl(N);
  const TargetOptions &Options = DAG.getTarget().Options;

  // Constant fold FMA.
  if (isa<ConstantFPSDNode>(N0) &&
      isa<ConstantFPSDNode>(N1) &&
      isa<ConstantFPSDNode>(N2)) {
    return DAG.getNode(ISD::FMA, dl, VT, N0, N1, N2);
  }

  if (Options.UnsafeFPMath) {
    if (N0CFP && N0CFP->isZero())
      return N2;
    if (N1CFP && N1CFP->isZero())
      return N2;
  }
  if (N0CFP && N0CFP->isExactlyValue(1.0))
    return DAG.getNode(ISD::FADD, SDLoc(N), VT, N1, N2);
  if (N1CFP && N1CFP->isExactlyValue(1.0))
    return DAG.getNode(ISD::FADD, SDLoc(N), VT, N0, N2);

  // Canonicalize (fma c, x, y) -> (fma x, c, y)
  if (N0CFP && !N1CFP)
    return DAG.getNode(ISD::FMA, SDLoc(N), VT, N1, N0, N2);

  // (fma x, c1, (fmul x, c2)) -> (fmul x, c1+c2)
  if (Options.UnsafeFPMath && N1CFP &&
      N2.getOpcode() == ISD::FMUL &&
      N0 == N2.getOperand(0) &&
      N2.getOperand(1).getOpcode() == ISD::ConstantFP) {
    return DAG.getNode(ISD::FMUL, dl, VT, N0,
                       DAG.getNode(ISD::FADD, dl, VT, N1, N2.getOperand(1)));
  }


  // (fma (fmul x, c1), c2, y) -> (fma x, c1*c2, y)
  if (Options.UnsafeFPMath &&
      N0.getOpcode() == ISD::FMUL && N1CFP &&
      N0.getOperand(1).getOpcode() == ISD::ConstantFP) {
    return DAG.getNode(ISD::FMA, dl, VT,
                       N0.getOperand(0),
                       DAG.getNode(ISD::FMUL, dl, VT, N1, N0.getOperand(1)),
                       N2);
  }

  // (fma x, 1, y) -> (fadd x, y)
  // (fma x, -1, y) -> (fadd (fneg x), y)
  if (N1CFP) {
    if (N1CFP->isExactlyValue(1.0))
      return DAG.getNode(ISD::FADD, dl, VT, N0, N2);

    if (N1CFP->isExactlyValue(-1.0) &&
        (!LegalOperations || TLI.isOperationLegal(ISD::FNEG, VT))) {
      SDValue RHSNeg = DAG.getNode(ISD::FNEG, dl, VT, N0);
      AddToWorklist(RHSNeg.getNode());
      return DAG.getNode(ISD::FADD, dl, VT, N2, RHSNeg);
    }
  }

  // (fma x, c, x) -> (fmul x, (c+1))
  if (Options.UnsafeFPMath && N1CFP && N0 == N2)
    return DAG.getNode(ISD::FMUL, dl, VT, N0,
                       DAG.getNode(ISD::FADD, dl, VT,
                                   N1, DAG.getConstantFP(1.0, dl, VT)));

  // (fma x, c, (fneg x)) -> (fmul x, (c-1))
  if (Options.UnsafeFPMath && N1CFP &&
      N2.getOpcode() == ISD::FNEG && N2.getOperand(0) == N0)
    return DAG.getNode(ISD::FMUL, dl, VT, N0,
                       DAG.getNode(ISD::FADD, dl, VT,
                                   N1, DAG.getConstantFP(-1.0, dl, VT)));


  return SDValue();
}

SDValue DAGCombiner::visitFDIV(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1);
  EVT VT = N->getValueType(0);
  SDLoc DL(N);
  const TargetOptions &Options = DAG.getTarget().Options;

  // fold vector ops
  if (VT.isVector())
    if (SDValue FoldedVOp = SimplifyVBinOp(N))
      return FoldedVOp;

  // fold (fdiv c1, c2) -> c1/c2
  if (N0CFP && N1CFP)
    return DAG.getNode(ISD::FDIV, SDLoc(N), VT, N0, N1);

  if (Options.UnsafeFPMath) {
    // fold (fdiv X, c2) -> fmul X, 1/c2 if losing precision is acceptable.
    if (N1CFP) {
      // Compute the reciprocal 1.0 / c2.
      APFloat N1APF = N1CFP->getValueAPF();
      APFloat Recip(N1APF.getSemantics(), 1); // 1.0
      APFloat::opStatus st = Recip.divide(N1APF, APFloat::rmNearestTiesToEven);
      // Only do the transform if the reciprocal is a legal fp immediate that
      // isn't too nasty (eg NaN, denormal, ...).
      if ((st == APFloat::opOK || st == APFloat::opInexact) && // Not too nasty
          (!LegalOperations ||
           // FIXME: custom lowering of ConstantFP might fail (see e.g. ARM
           // backend)... we should handle this gracefully after Legalize.
           // TLI.isOperationLegalOrCustom(llvm::ISD::ConstantFP, VT) ||
           TLI.isOperationLegal(llvm::ISD::ConstantFP, VT) ||
           TLI.isFPImmLegal(Recip, VT)))
        return DAG.getNode(ISD::FMUL, DL, VT, N0,
                           DAG.getConstantFP(Recip, DL, VT));
    }

    // If this FDIV is part of a reciprocal square root, it may be folded
    // into a target-specific square root estimate instruction.
    if (N1.getOpcode() == ISD::FSQRT) {
      if (SDValue RV = BuildRsqrtEstimate(N1.getOperand(0))) {
        return DAG.getNode(ISD::FMUL, DL, VT, N0, RV);
      }
    } else if (N1.getOpcode() == ISD::FP_EXTEND &&
               N1.getOperand(0).getOpcode() == ISD::FSQRT) {
      if (SDValue RV = BuildRsqrtEstimate(N1.getOperand(0).getOperand(0))) {
        RV = DAG.getNode(ISD::FP_EXTEND, SDLoc(N1), VT, RV);
        AddToWorklist(RV.getNode());
        return DAG.getNode(ISD::FMUL, DL, VT, N0, RV);
      }
    } else if (N1.getOpcode() == ISD::FP_ROUND &&
               N1.getOperand(0).getOpcode() == ISD::FSQRT) {
      if (SDValue RV = BuildRsqrtEstimate(N1.getOperand(0).getOperand(0))) {
        RV = DAG.getNode(ISD::FP_ROUND, SDLoc(N1), VT, RV, N1.getOperand(1));
        AddToWorklist(RV.getNode());
        return DAG.getNode(ISD::FMUL, DL, VT, N0, RV);
      }
    } else if (N1.getOpcode() == ISD::FMUL) {
      // Look through an FMUL. Even though this won't remove the FDIV directly,
      // it's still worthwhile to get rid of the FSQRT if possible.
      SDValue SqrtOp;
      SDValue OtherOp;
      if (N1.getOperand(0).getOpcode() == ISD::FSQRT) {
        SqrtOp = N1.getOperand(0);
        OtherOp = N1.getOperand(1);
      } else if (N1.getOperand(1).getOpcode() == ISD::FSQRT) {
        SqrtOp = N1.getOperand(1);
        OtherOp = N1.getOperand(0);
      }
      if (SqrtOp.getNode()) {
        // We found a FSQRT, so try to make this fold:
        // x / (y * sqrt(z)) -> x * (rsqrt(z) / y)
        if (SDValue RV = BuildRsqrtEstimate(SqrtOp.getOperand(0))) {
          RV = DAG.getNode(ISD::FDIV, SDLoc(N1), VT, RV, OtherOp);
          AddToWorklist(RV.getNode());
          return DAG.getNode(ISD::FMUL, DL, VT, N0, RV);
        }
      }
    }

    // Fold into a reciprocal estimate and multiply instead of a real divide.
    if (SDValue RV = BuildReciprocalEstimate(N1)) {
      AddToWorklist(RV.getNode());
      return DAG.getNode(ISD::FMUL, DL, VT, N0, RV);
    }
  }

  // (fdiv (fneg X), (fneg Y)) -> (fdiv X, Y)
  if (char LHSNeg = isNegatibleForFree(N0, LegalOperations, TLI, &Options)) {
    if (char RHSNeg = isNegatibleForFree(N1, LegalOperations, TLI, &Options)) {
      // Both can be negated for free, check to see if at least one is cheaper
      // negated.
      if (LHSNeg == 2 || RHSNeg == 2)
        return DAG.getNode(ISD::FDIV, SDLoc(N), VT,
                           GetNegatedExpression(N0, DAG, LegalOperations),
                           GetNegatedExpression(N1, DAG, LegalOperations));
    }
  }

  // Combine multiple FDIVs with the same divisor into multiple FMULs by the
  // reciprocal.
  // E.g., (a / D; b / D;) -> (recip = 1.0 / D; a * recip; b * recip)
  // Notice that this is not always beneficial. One reason is different target
  // may have different costs for FDIV and FMUL, so sometimes the cost of two
  // FDIVs may be lower than the cost of one FDIV and two FMULs. Another reason
  // is the critical path is increased from "one FDIV" to "one FDIV + one FMUL".
  if (Options.UnsafeFPMath) {
    // Skip if current node is a reciprocal.
    if (N0CFP && N0CFP->isExactlyValue(1.0))
      return SDValue();

    // Find all FDIV users of the same divisor.
    // Use a set because duplicates may be present in the user list.
    SetVector<SDNode *> Users;
    for (auto *U : N1->uses())
      if (U->getOpcode() == ISD::FDIV && U->getOperand(1) == N1)
        Users.insert(U);

    if (TLI.combineRepeatedFPDivisors(Users.size())) {
      SDValue FPOne = DAG.getConstantFP(1.0, DL, VT);
      // FIXME: This optimization requires some level of fast-math, so the
      // created reciprocal node should at least have the 'allowReciprocal'
      // fast-math-flag set.
      SDValue Reciprocal = DAG.getNode(ISD::FDIV, DL, VT, FPOne, N1);

      // Dividend / Divisor -> Dividend * Reciprocal
      for (auto *U : Users) {
        SDValue Dividend = U->getOperand(0);
        if (Dividend != FPOne) {
          SDValue NewNode = DAG.getNode(ISD::FMUL, SDLoc(U), VT, Dividend,
                                        Reciprocal);
          CombineTo(U, NewNode);
        } else if (U != Reciprocal.getNode()) {
          // In the absence of fast-math-flags, this user node is always the
          // same node as Reciprocal, but with FMF they may be different nodes.
          CombineTo(U, Reciprocal);
        }
      }
      return SDValue(N, 0);  // N was replaced.
    }
  }

  return SDValue();
}

SDValue DAGCombiner::visitFREM(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1);
  EVT VT = N->getValueType(0);

  // fold (frem c1, c2) -> fmod(c1,c2)
  if (N0CFP && N1CFP)
    return DAG.getNode(ISD::FREM, SDLoc(N), VT, N0, N1);

  return SDValue();
}

SDValue DAGCombiner::visitFSQRT(SDNode *N) {
  if (!DAG.getTarget().Options.UnsafeFPMath || TLI.isFsqrtCheap())
    return SDValue();

  // Compute this as X * (1/sqrt(X)) = X * (X ** -0.5)
  SDValue RV = BuildRsqrtEstimate(N->getOperand(0));
  if (!RV)
    return SDValue();
  
  EVT VT = RV.getValueType();
  SDLoc DL(N);
  RV = DAG.getNode(ISD::FMUL, DL, VT, N->getOperand(0), RV);
  AddToWorklist(RV.getNode());

  // Unfortunately, RV is now NaN if the input was exactly 0.
  // Select out this case and force the answer to 0.
  SDValue Zero = DAG.getConstantFP(0.0, DL, VT);
  EVT CCVT = TLI.getSetCCResultType(DAG.getDataLayout(), *DAG.getContext(), VT);
  SDValue ZeroCmp = DAG.getSetCC(DL, CCVT, N->getOperand(0), Zero, ISD::SETEQ);
  AddToWorklist(ZeroCmp.getNode());
  AddToWorklist(RV.getNode());

  return DAG.getNode(VT.isVector() ? ISD::VSELECT : ISD::SELECT, DL, VT,
                     ZeroCmp, Zero, RV);
}

SDValue DAGCombiner::visitFCOPYSIGN(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1);
  EVT VT = N->getValueType(0);

  if (N0CFP && N1CFP)  // Constant fold
    return DAG.getNode(ISD::FCOPYSIGN, SDLoc(N), VT, N0, N1);

  if (N1CFP) {
    const APFloat& V = N1CFP->getValueAPF();
    // copysign(x, c1) -> fabs(x)       iff ispos(c1)
    // copysign(x, c1) -> fneg(fabs(x)) iff isneg(c1)
    if (!V.isNegative()) {
      if (!LegalOperations || TLI.isOperationLegal(ISD::FABS, VT))
        return DAG.getNode(ISD::FABS, SDLoc(N), VT, N0);
    } else {
      if (!LegalOperations || TLI.isOperationLegal(ISD::FNEG, VT))
        return DAG.getNode(ISD::FNEG, SDLoc(N), VT,
                           DAG.getNode(ISD::FABS, SDLoc(N0), VT, N0));
    }
  }

  // copysign(fabs(x), y) -> copysign(x, y)
  // copysign(fneg(x), y) -> copysign(x, y)
  // copysign(copysign(x,z), y) -> copysign(x, y)
  if (N0.getOpcode() == ISD::FABS || N0.getOpcode() == ISD::FNEG ||
      N0.getOpcode() == ISD::FCOPYSIGN)
    return DAG.getNode(ISD::FCOPYSIGN, SDLoc(N), VT,
                       N0.getOperand(0), N1);

  // copysign(x, abs(y)) -> abs(x)
  if (N1.getOpcode() == ISD::FABS)
    return DAG.getNode(ISD::FABS, SDLoc(N), VT, N0);

  // copysign(x, copysign(y,z)) -> copysign(x, z)
  if (N1.getOpcode() == ISD::FCOPYSIGN)
    return DAG.getNode(ISD::FCOPYSIGN, SDLoc(N), VT,
                       N0, N1.getOperand(1));

  // copysign(x, fp_extend(y)) -> copysign(x, y)
  // copysign(x, fp_round(y)) -> copysign(x, y)
  if (N1.getOpcode() == ISD::FP_EXTEND || N1.getOpcode() == ISD::FP_ROUND)
    return DAG.getNode(ISD::FCOPYSIGN, SDLoc(N), VT,
                       N0, N1.getOperand(0));

  return SDValue();
}

SDValue DAGCombiner::visitSINT_TO_FP(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);
  EVT OpVT = N0.getValueType();

  // fold (sint_to_fp c1) -> c1fp
  if (isConstantIntBuildVectorOrConstantInt(N0) &&
      // ...but only if the target supports immediate floating-point values
      (!LegalOperations ||
       TLI.isOperationLegalOrCustom(llvm::ISD::ConstantFP, VT)))
    return DAG.getNode(ISD::SINT_TO_FP, SDLoc(N), VT, N0);

  // If the input is a legal type, and SINT_TO_FP is not legal on this target,
  // but UINT_TO_FP is legal on this target, try to convert.
  if (!TLI.isOperationLegalOrCustom(ISD::SINT_TO_FP, OpVT) &&
      TLI.isOperationLegalOrCustom(ISD::UINT_TO_FP, OpVT)) {
    // If the sign bit is known to be zero, we can change this to UINT_TO_FP.
    if (DAG.SignBitIsZero(N0))
      return DAG.getNode(ISD::UINT_TO_FP, SDLoc(N), VT, N0);
  }

  // The next optimizations are desirable only if SELECT_CC can be lowered.
  if (TLI.isOperationLegalOrCustom(ISD::SELECT_CC, VT) || !LegalOperations) {
    // fold (sint_to_fp (setcc x, y, cc)) -> (select_cc x, y, -1.0, 0.0,, cc)
    if (N0.getOpcode() == ISD::SETCC && N0.getValueType() == MVT::i1 &&
        !VT.isVector() &&
        (!LegalOperations ||
         TLI.isOperationLegalOrCustom(llvm::ISD::ConstantFP, VT))) {
      SDLoc DL(N);
      SDValue Ops[] =
        { N0.getOperand(0), N0.getOperand(1),
          DAG.getConstantFP(-1.0, DL, VT), DAG.getConstantFP(0.0, DL, VT),
          N0.getOperand(2) };
      return DAG.getNode(ISD::SELECT_CC, DL, VT, Ops);
    }

    // fold (sint_to_fp (zext (setcc x, y, cc))) ->
    //      (select_cc x, y, 1.0, 0.0,, cc)
    if (N0.getOpcode() == ISD::ZERO_EXTEND &&
        N0.getOperand(0).getOpcode() == ISD::SETCC &&!VT.isVector() &&
        (!LegalOperations ||
         TLI.isOperationLegalOrCustom(llvm::ISD::ConstantFP, VT))) {
      SDLoc DL(N);
      SDValue Ops[] =
        { N0.getOperand(0).getOperand(0), N0.getOperand(0).getOperand(1),
          DAG.getConstantFP(1.0, DL, VT), DAG.getConstantFP(0.0, DL, VT),
          N0.getOperand(0).getOperand(2) };
      return DAG.getNode(ISD::SELECT_CC, DL, VT, Ops);
    }
  }

  return SDValue();
}

SDValue DAGCombiner::visitUINT_TO_FP(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);
  EVT OpVT = N0.getValueType();

  // fold (uint_to_fp c1) -> c1fp
  if (isConstantIntBuildVectorOrConstantInt(N0) &&
      // ...but only if the target supports immediate floating-point values
      (!LegalOperations ||
       TLI.isOperationLegalOrCustom(llvm::ISD::ConstantFP, VT)))
    return DAG.getNode(ISD::UINT_TO_FP, SDLoc(N), VT, N0);

  // If the input is a legal type, and UINT_TO_FP is not legal on this target,
  // but SINT_TO_FP is legal on this target, try to convert.
  if (!TLI.isOperationLegalOrCustom(ISD::UINT_TO_FP, OpVT) &&
      TLI.isOperationLegalOrCustom(ISD::SINT_TO_FP, OpVT)) {
    // If the sign bit is known to be zero, we can change this to SINT_TO_FP.
    if (DAG.SignBitIsZero(N0))
      return DAG.getNode(ISD::SINT_TO_FP, SDLoc(N), VT, N0);
  }

  // The next optimizations are desirable only if SELECT_CC can be lowered.
  if (TLI.isOperationLegalOrCustom(ISD::SELECT_CC, VT) || !LegalOperations) {
    // fold (uint_to_fp (setcc x, y, cc)) -> (select_cc x, y, -1.0, 0.0,, cc)

    if (N0.getOpcode() == ISD::SETCC && !VT.isVector() &&
        (!LegalOperations ||
         TLI.isOperationLegalOrCustom(llvm::ISD::ConstantFP, VT))) {
      SDLoc DL(N);
      SDValue Ops[] =
        { N0.getOperand(0), N0.getOperand(1),
          DAG.getConstantFP(1.0, DL, VT), DAG.getConstantFP(0.0, DL, VT),
          N0.getOperand(2) };
      return DAG.getNode(ISD::SELECT_CC, DL, VT, Ops);
    }
  }

  return SDValue();
}

// Fold (fp_to_{s/u}int ({s/u}int_to_fpx)) -> zext x, sext x, trunc x, or x
static SDValue FoldIntToFPToInt(SDNode *N, SelectionDAG &DAG) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  if (N0.getOpcode() != ISD::UINT_TO_FP && N0.getOpcode() != ISD::SINT_TO_FP)
    return SDValue();

  SDValue Src = N0.getOperand(0);
  EVT SrcVT = Src.getValueType();
  bool IsInputSigned = N0.getOpcode() == ISD::SINT_TO_FP;
  bool IsOutputSigned = N->getOpcode() == ISD::FP_TO_SINT;

  // We can safely assume the conversion won't overflow the output range,
  // because (for example) (uint8_t)18293.f is undefined behavior.

  // Since we can assume the conversion won't overflow, our decision as to
  // whether the input will fit in the float should depend on the minimum
  // of the input range and output range.

  // This means this is also safe for a signed input and unsigned output, since
  // a negative input would lead to undefined behavior.
  unsigned InputSize = (int)SrcVT.getScalarSizeInBits() - IsInputSigned;
  unsigned OutputSize = (int)VT.getScalarSizeInBits() - IsOutputSigned;
  unsigned ActualSize = std::min(InputSize, OutputSize);
  const fltSemantics &sem = DAG.EVTToAPFloatSemantics(N0.getValueType());

  // We can only fold away the float conversion if the input range can be
  // represented exactly in the float range.
  if (APFloat::semanticsPrecision(sem) >= ActualSize) {
    if (VT.getScalarSizeInBits() > SrcVT.getScalarSizeInBits()) {
      unsigned ExtOp = IsInputSigned && IsOutputSigned ? ISD::SIGN_EXTEND
                                                       : ISD::ZERO_EXTEND;
      return DAG.getNode(ExtOp, SDLoc(N), VT, Src);
    }
    if (VT.getScalarSizeInBits() < SrcVT.getScalarSizeInBits())
      return DAG.getNode(ISD::TRUNCATE, SDLoc(N), VT, Src);
    if (SrcVT == VT)
      return Src;
    return DAG.getNode(ISD::BITCAST, SDLoc(N), VT, Src);
  }
  return SDValue();
}

SDValue DAGCombiner::visitFP_TO_SINT(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // fold (fp_to_sint c1fp) -> c1
  if (isConstantFPBuildVectorOrConstantFP(N0))
    return DAG.getNode(ISD::FP_TO_SINT, SDLoc(N), VT, N0);

  return FoldIntToFPToInt(N, DAG);
}

SDValue DAGCombiner::visitFP_TO_UINT(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // fold (fp_to_uint c1fp) -> c1
  if (isConstantFPBuildVectorOrConstantFP(N0))
    return DAG.getNode(ISD::FP_TO_UINT, SDLoc(N), VT, N0);

  return FoldIntToFPToInt(N, DAG);
}

SDValue DAGCombiner::visitFP_ROUND(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  EVT VT = N->getValueType(0);

  // fold (fp_round c1fp) -> c1fp
  if (N0CFP)
    return DAG.getNode(ISD::FP_ROUND, SDLoc(N), VT, N0, N1);

  // fold (fp_round (fp_extend x)) -> x
  if (N0.getOpcode() == ISD::FP_EXTEND && VT == N0.getOperand(0).getValueType())
    return N0.getOperand(0);

  // fold (fp_round (fp_round x)) -> (fp_round x)
  if (N0.getOpcode() == ISD::FP_ROUND) {
    const bool NIsTrunc = N->getConstantOperandVal(1) == 1;
    const bool N0IsTrunc = N0.getNode()->getConstantOperandVal(1) == 1;
    // If the first fp_round isn't a value preserving truncation, it might
    // introduce a tie in the second fp_round, that wouldn't occur in the
    // single-step fp_round we want to fold to.
    // In other words, double rounding isn't the same as rounding.
    // Also, this is a value preserving truncation iff both fp_round's are.
    if (DAG.getTarget().Options.UnsafeFPMath || N0IsTrunc) {
      SDLoc DL(N);
      return DAG.getNode(ISD::FP_ROUND, DL, VT, N0.getOperand(0),
                         DAG.getIntPtrConstant(NIsTrunc && N0IsTrunc, DL));
    }
  }

  // fold (fp_round (copysign X, Y)) -> (copysign (fp_round X), Y)
  if (N0.getOpcode() == ISD::FCOPYSIGN && N0.getNode()->hasOneUse()) {
    SDValue Tmp = DAG.getNode(ISD::FP_ROUND, SDLoc(N0), VT,
                              N0.getOperand(0), N1);
    AddToWorklist(Tmp.getNode());
    return DAG.getNode(ISD::FCOPYSIGN, SDLoc(N), VT,
                       Tmp, N0.getOperand(1));
  }

  return SDValue();
}

SDValue DAGCombiner::visitFP_ROUND_INREG(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);
  EVT EVT = cast<VTSDNode>(N->getOperand(1))->getVT();
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);

  // fold (fp_round_inreg c1fp) -> c1fp
  if (N0CFP && isTypeLegal(EVT)) {
    SDLoc DL(N);
    SDValue Round = DAG.getConstantFP(*N0CFP->getConstantFPValue(), DL, EVT);
    return DAG.getNode(ISD::FP_EXTEND, DL, VT, Round);
  }

  return SDValue();
}

SDValue DAGCombiner::visitFP_EXTEND(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // If this is fp_round(fpextend), don't fold it, allow ourselves to be folded.
  if (N->hasOneUse() &&
      N->use_begin()->getOpcode() == ISD::FP_ROUND)
    return SDValue();

  // fold (fp_extend c1fp) -> c1fp
  if (isConstantFPBuildVectorOrConstantFP(N0))
    return DAG.getNode(ISD::FP_EXTEND, SDLoc(N), VT, N0);

  // fold (fp_extend (fp16_to_fp op)) -> (fp16_to_fp op)
  if (N0.getOpcode() == ISD::FP16_TO_FP &&
      TLI.getOperationAction(ISD::FP16_TO_FP, VT) == TargetLowering::Legal)
    return DAG.getNode(ISD::FP16_TO_FP, SDLoc(N), VT, N0.getOperand(0));

  // Turn fp_extend(fp_round(X, 1)) -> x since the fp_round doesn't affect the
  // value of X.
  if (N0.getOpcode() == ISD::FP_ROUND
      && N0.getNode()->getConstantOperandVal(1) == 1) {
    SDValue In = N0.getOperand(0);
    if (In.getValueType() == VT) return In;
    if (VT.bitsLT(In.getValueType()))
      return DAG.getNode(ISD::FP_ROUND, SDLoc(N), VT,
                         In, N0.getOperand(1));
    return DAG.getNode(ISD::FP_EXTEND, SDLoc(N), VT, In);
  }

  // fold (fpext (load x)) -> (fpext (fptrunc (extload x)))
  if (ISD::isNormalLoad(N0.getNode()) && N0.hasOneUse() &&
       TLI.isLoadExtLegal(ISD::EXTLOAD, VT, N0.getValueType())) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);
    SDValue ExtLoad = DAG.getExtLoad(ISD::EXTLOAD, SDLoc(N), VT,
                                     LN0->getChain(),
                                     LN0->getBasePtr(), N0.getValueType(),
                                     LN0->getMemOperand());
    CombineTo(N, ExtLoad);
    CombineTo(N0.getNode(),
              DAG.getNode(ISD::FP_ROUND, SDLoc(N0),
                          N0.getValueType(), ExtLoad,
                          DAG.getIntPtrConstant(1, SDLoc(N0))),
              ExtLoad.getValue(1));
    return SDValue(N, 0);   // Return N so it doesn't get rechecked!
  }

  return SDValue();
}

SDValue DAGCombiner::visitFCEIL(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // fold (fceil c1) -> fceil(c1)
  if (isConstantFPBuildVectorOrConstantFP(N0))
    return DAG.getNode(ISD::FCEIL, SDLoc(N), VT, N0);

  return SDValue();
}

SDValue DAGCombiner::visitFTRUNC(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // fold (ftrunc c1) -> ftrunc(c1)
  if (isConstantFPBuildVectorOrConstantFP(N0))
    return DAG.getNode(ISD::FTRUNC, SDLoc(N), VT, N0);

  return SDValue();
}

SDValue DAGCombiner::visitFFLOOR(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // fold (ffloor c1) -> ffloor(c1)
  if (isConstantFPBuildVectorOrConstantFP(N0))
    return DAG.getNode(ISD::FFLOOR, SDLoc(N), VT, N0);

  return SDValue();
}

// FIXME: FNEG and FABS have a lot in common; refactor.
SDValue DAGCombiner::visitFNEG(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // Constant fold FNEG.
  if (isConstantFPBuildVectorOrConstantFP(N0))
    return DAG.getNode(ISD::FNEG, SDLoc(N), VT, N0);

  if (isNegatibleForFree(N0, LegalOperations, DAG.getTargetLoweringInfo(),
                         &DAG.getTarget().Options))
    return GetNegatedExpression(N0, DAG, LegalOperations);

  // Transform fneg(bitconvert(x)) -> bitconvert(x ^ sign) to avoid loading
  // constant pool values.
  if (!TLI.isFNegFree(VT) &&
      N0.getOpcode() == ISD::BITCAST &&
      N0.getNode()->hasOneUse()) {
    SDValue Int = N0.getOperand(0);
    EVT IntVT = Int.getValueType();
    if (IntVT.isInteger() && !IntVT.isVector()) {
      APInt SignMask;
      if (N0.getValueType().isVector()) {
        // For a vector, get a mask such as 0x80... per scalar element
        // and splat it.
        SignMask = APInt::getSignBit(N0.getValueType().getScalarSizeInBits());
        SignMask = APInt::getSplat(IntVT.getSizeInBits(), SignMask);
      } else {
        // For a scalar, just generate 0x80...
        SignMask = APInt::getSignBit(IntVT.getSizeInBits());
      }
      SDLoc DL0(N0);
      Int = DAG.getNode(ISD::XOR, DL0, IntVT, Int,
                        DAG.getConstant(SignMask, DL0, IntVT));
      AddToWorklist(Int.getNode());
      return DAG.getNode(ISD::BITCAST, SDLoc(N), VT, Int);
    }
  }

  // (fneg (fmul c, x)) -> (fmul -c, x)
  if (N0.getOpcode() == ISD::FMUL &&
      (N0.getNode()->hasOneUse() || !TLI.isFNegFree(VT))) {
    ConstantFPSDNode *CFP1 = dyn_cast<ConstantFPSDNode>(N0.getOperand(1));
    if (CFP1) {
      APFloat CVal = CFP1->getValueAPF();
      CVal.changeSign();
      if (Level >= AfterLegalizeDAG &&
          (TLI.isFPImmLegal(CVal, N->getValueType(0)) ||
           TLI.isOperationLegal(ISD::ConstantFP, N->getValueType(0))))
        return DAG.getNode(
            ISD::FMUL, SDLoc(N), VT, N0.getOperand(0),
            DAG.getNode(ISD::FNEG, SDLoc(N), VT, N0.getOperand(1)));
    }
  }

  return SDValue();
}

SDValue DAGCombiner::visitFMINNUM(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  const ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  const ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1);

  if (N0CFP && N1CFP) {
    const APFloat &C0 = N0CFP->getValueAPF();
    const APFloat &C1 = N1CFP->getValueAPF();
    return DAG.getConstantFP(minnum(C0, C1), SDLoc(N), N->getValueType(0));
  }

  if (N0CFP) {
    EVT VT = N->getValueType(0);
    // Canonicalize to constant on RHS.
    return DAG.getNode(ISD::FMINNUM, SDLoc(N), VT, N1, N0);
  }

  return SDValue();
}

SDValue DAGCombiner::visitFMAXNUM(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  const ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  const ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1);

  if (N0CFP && N1CFP) {
    const APFloat &C0 = N0CFP->getValueAPF();
    const APFloat &C1 = N1CFP->getValueAPF();
    return DAG.getConstantFP(maxnum(C0, C1), SDLoc(N), N->getValueType(0));
  }

  if (N0CFP) {
    EVT VT = N->getValueType(0);
    // Canonicalize to constant on RHS.
    return DAG.getNode(ISD::FMAXNUM, SDLoc(N), VT, N1, N0);
  }

  return SDValue();
}

SDValue DAGCombiner::visitFABS(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // fold (fabs c1) -> fabs(c1)
  if (isConstantFPBuildVectorOrConstantFP(N0))
    return DAG.getNode(ISD::FABS, SDLoc(N), VT, N0);

  // fold (fabs (fabs x)) -> (fabs x)
  if (N0.getOpcode() == ISD::FABS)
    return N->getOperand(0);

  // fold (fabs (fneg x)) -> (fabs x)
  // fold (fabs (fcopysign x, y)) -> (fabs x)
  if (N0.getOpcode() == ISD::FNEG || N0.getOpcode() == ISD::FCOPYSIGN)
    return DAG.getNode(ISD::FABS, SDLoc(N), VT, N0.getOperand(0));

  // Transform fabs(bitconvert(x)) -> bitconvert(x & ~sign) to avoid loading
  // constant pool values.
  if (!TLI.isFAbsFree(VT) &&
      N0.getOpcode() == ISD::BITCAST &&
      N0.getNode()->hasOneUse()) {
    SDValue Int = N0.getOperand(0);
    EVT IntVT = Int.getValueType();
    if (IntVT.isInteger() && !IntVT.isVector()) {
      APInt SignMask;
      if (N0.getValueType().isVector()) {
        // For a vector, get a mask such as 0x7f... per scalar element
        // and splat it.
        SignMask = ~APInt::getSignBit(N0.getValueType().getScalarSizeInBits());
        SignMask = APInt::getSplat(IntVT.getSizeInBits(), SignMask);
      } else {
        // For a scalar, just generate 0x7f...
        SignMask = ~APInt::getSignBit(IntVT.getSizeInBits());
      }
      SDLoc DL(N0);
      Int = DAG.getNode(ISD::AND, DL, IntVT, Int,
                        DAG.getConstant(SignMask, DL, IntVT));
      AddToWorklist(Int.getNode());
      return DAG.getNode(ISD::BITCAST, SDLoc(N), N->getValueType(0), Int);
    }
  }

  return SDValue();
}

SDValue DAGCombiner::visitBRCOND(SDNode *N) {
  SDValue Chain = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  SDValue N2 = N->getOperand(2);

  // If N is a constant we could fold this into a fallthrough or unconditional
  // branch. However that doesn't happen very often in normal code, because
  // Instcombine/SimplifyCFG should have handled the available opportunities.
  // If we did this folding here, it would be necessary to update the
  // MachineBasicBlock CFG, which is awkward.

  // fold a brcond with a setcc condition into a BR_CC node if BR_CC is legal
  // on the target.
  if (N1.getOpcode() == ISD::SETCC &&
      TLI.isOperationLegalOrCustom(ISD::BR_CC,
                                   N1.getOperand(0).getValueType())) {
    return DAG.getNode(ISD::BR_CC, SDLoc(N), MVT::Other,
                       Chain, N1.getOperand(2),
                       N1.getOperand(0), N1.getOperand(1), N2);
  }

  if ((N1.hasOneUse() && N1.getOpcode() == ISD::SRL) ||
      ((N1.getOpcode() == ISD::TRUNCATE && N1.hasOneUse()) &&
       (N1.getOperand(0).hasOneUse() &&
        N1.getOperand(0).getOpcode() == ISD::SRL))) {
    SDNode *Trunc = nullptr;
    if (N1.getOpcode() == ISD::TRUNCATE) {
      // Look pass the truncate.
      Trunc = N1.getNode();
      N1 = N1.getOperand(0);
    }

    // Match this pattern so that we can generate simpler code:
    //
    //   %a = ...
    //   %b = and i32 %a, 2
    //   %c = srl i32 %b, 1
    //   brcond i32 %c ...
    //
    // into
    //
    //   %a = ...
    //   %b = and i32 %a, 2
    //   %c = setcc eq %b, 0
    //   brcond %c ...
    //
    // This applies only when the AND constant value has one bit set and the
    // SRL constant is equal to the log2 of the AND constant. The back-end is
    // smart enough to convert the result into a TEST/JMP sequence.
    SDValue Op0 = N1.getOperand(0);
    SDValue Op1 = N1.getOperand(1);

    if (Op0.getOpcode() == ISD::AND &&
        Op1.getOpcode() == ISD::Constant) {
      SDValue AndOp1 = Op0.getOperand(1);

      if (AndOp1.getOpcode() == ISD::Constant) {
        const APInt &AndConst = cast<ConstantSDNode>(AndOp1)->getAPIntValue();

        if (AndConst.isPowerOf2() &&
            cast<ConstantSDNode>(Op1)->getAPIntValue()==AndConst.logBase2()) {
          SDLoc DL(N);
          SDValue SetCC =
            DAG.getSetCC(DL,
                         getSetCCResultType(Op0.getValueType()),
                         Op0, DAG.getConstant(0, DL, Op0.getValueType()),
                         ISD::SETNE);

          SDValue NewBRCond = DAG.getNode(ISD::BRCOND, DL,
                                          MVT::Other, Chain, SetCC, N2);
          // Don't add the new BRCond into the worklist or else SimplifySelectCC
          // will convert it back to (X & C1) >> C2.
          CombineTo(N, NewBRCond, false);
          // Truncate is dead.
          if (Trunc)
            deleteAndRecombine(Trunc);
          // Replace the uses of SRL with SETCC
          WorklistRemover DeadNodes(*this);
          DAG.ReplaceAllUsesOfValueWith(N1, SetCC);
          deleteAndRecombine(N1.getNode());
          return SDValue(N, 0);   // Return N so it doesn't get rechecked!
        }
      }
    }

    if (Trunc)
      // Restore N1 if the above transformation doesn't match.
      N1 = N->getOperand(1);
  }

  // Transform br(xor(x, y)) -> br(x != y)
  // Transform br(xor(xor(x,y), 1)) -> br (x == y)
  if (N1.hasOneUse() && N1.getOpcode() == ISD::XOR) {
    SDNode *TheXor = N1.getNode();
    SDValue Op0 = TheXor->getOperand(0);
    SDValue Op1 = TheXor->getOperand(1);
    if (Op0.getOpcode() == Op1.getOpcode()) {
      // Avoid missing important xor optimizations.
      SDValue Tmp = visitXOR(TheXor);
      if (Tmp.getNode()) {
        if (Tmp.getNode() != TheXor) {
          DEBUG(dbgs() << "\nReplacing.8 ";
                TheXor->dump(&DAG);
                dbgs() << "\nWith: ";
                Tmp.getNode()->dump(&DAG);
                dbgs() << '\n');
          WorklistRemover DeadNodes(*this);
          DAG.ReplaceAllUsesOfValueWith(N1, Tmp);
          deleteAndRecombine(TheXor);
          return DAG.getNode(ISD::BRCOND, SDLoc(N),
                             MVT::Other, Chain, Tmp, N2);
        }

        // visitXOR has changed XOR's operands or replaced the XOR completely,
        // bail out.
        return SDValue(N, 0);
      }
    }

    if (Op0.getOpcode() != ISD::SETCC && Op1.getOpcode() != ISD::SETCC) {
      bool Equal = false;
      if (isOneConstant(Op0) && Op0.hasOneUse() &&
          Op0.getOpcode() == ISD::XOR) {
        TheXor = Op0.getNode();
        Equal = true;
      }

      EVT SetCCVT = N1.getValueType();
      if (LegalTypes)
        SetCCVT = getSetCCResultType(SetCCVT);
      SDValue SetCC = DAG.getSetCC(SDLoc(TheXor),
                                   SetCCVT,
                                   Op0, Op1,
                                   Equal ? ISD::SETEQ : ISD::SETNE);
      // Replace the uses of XOR with SETCC
      WorklistRemover DeadNodes(*this);
      DAG.ReplaceAllUsesOfValueWith(N1, SetCC);
      deleteAndRecombine(N1.getNode());
      return DAG.getNode(ISD::BRCOND, SDLoc(N),
                         MVT::Other, Chain, SetCC, N2);
    }
  }

  return SDValue();
}

// Operand List for BR_CC: Chain, CondCC, CondLHS, CondRHS, DestBB.
//
SDValue DAGCombiner::visitBR_CC(SDNode *N) {
  CondCodeSDNode *CC = cast<CondCodeSDNode>(N->getOperand(1));
  SDValue CondLHS = N->getOperand(2), CondRHS = N->getOperand(3);

  // If N is a constant we could fold this into a fallthrough or unconditional
  // branch. However that doesn't happen very often in normal code, because
  // Instcombine/SimplifyCFG should have handled the available opportunities.
  // If we did this folding here, it would be necessary to update the
  // MachineBasicBlock CFG, which is awkward.

  // Use SimplifySetCC to simplify SETCC's.
  SDValue Simp = SimplifySetCC(getSetCCResultType(CondLHS.getValueType()),
                               CondLHS, CondRHS, CC->get(), SDLoc(N),
                               false);
  if (Simp.getNode()) AddToWorklist(Simp.getNode());

  // fold to a simpler setcc
  if (Simp.getNode() && Simp.getOpcode() == ISD::SETCC)
    return DAG.getNode(ISD::BR_CC, SDLoc(N), MVT::Other,
                       N->getOperand(0), Simp.getOperand(2),
                       Simp.getOperand(0), Simp.getOperand(1),
                       N->getOperand(4));

  return SDValue();
}

/// Return true if 'Use' is a load or a store that uses N as its base pointer
/// and that N may be folded in the load / store addressing mode.
static bool canFoldInAddressingMode(SDNode *N, SDNode *Use,
                                    SelectionDAG &DAG,
                                    const TargetLowering &TLI) {
  EVT VT;
  unsigned AS;

  if (LoadSDNode *LD  = dyn_cast<LoadSDNode>(Use)) {
    if (LD->isIndexed() || LD->getBasePtr().getNode() != N)
      return false;
    VT = LD->getMemoryVT();
    AS = LD->getAddressSpace();
  } else if (StoreSDNode *ST  = dyn_cast<StoreSDNode>(Use)) {
    if (ST->isIndexed() || ST->getBasePtr().getNode() != N)
      return false;
    VT = ST->getMemoryVT();
    AS = ST->getAddressSpace();
  } else
    return false;

  TargetLowering::AddrMode AM;
  if (N->getOpcode() == ISD::ADD) {
    ConstantSDNode *Offset = dyn_cast<ConstantSDNode>(N->getOperand(1));
    if (Offset)
      // [reg +/- imm]
      AM.BaseOffs = Offset->getSExtValue();
    else
      // [reg +/- reg]
      AM.Scale = 1;
  } else if (N->getOpcode() == ISD::SUB) {
    ConstantSDNode *Offset = dyn_cast<ConstantSDNode>(N->getOperand(1));
    if (Offset)
      // [reg +/- imm]
      AM.BaseOffs = -Offset->getSExtValue();
    else
      // [reg +/- reg]
      AM.Scale = 1;
  } else
    return false;

  return TLI.isLegalAddressingMode(DAG.getDataLayout(), AM,
                                   VT.getTypeForEVT(*DAG.getContext()), AS);
}

/// Try turning a load/store into a pre-indexed load/store when the base
/// pointer is an add or subtract and it has other uses besides the load/store.
/// After the transformation, the new indexed load/store has effectively folded
/// the add/subtract in and all of its other uses are redirected to the
/// new load/store.
bool DAGCombiner::CombineToPreIndexedLoadStore(SDNode *N) {
  if (Level < AfterLegalizeDAG)
    return false;

  bool isLoad = true;
  SDValue Ptr;
  EVT VT;
  if (LoadSDNode *LD  = dyn_cast<LoadSDNode>(N)) {
    if (LD->isIndexed())
      return false;
    VT = LD->getMemoryVT();
    if (!TLI.isIndexedLoadLegal(ISD::PRE_INC, VT) &&
        !TLI.isIndexedLoadLegal(ISD::PRE_DEC, VT))
      return false;
    Ptr = LD->getBasePtr();
  } else if (StoreSDNode *ST  = dyn_cast<StoreSDNode>(N)) {
    if (ST->isIndexed())
      return false;
    VT = ST->getMemoryVT();
    if (!TLI.isIndexedStoreLegal(ISD::PRE_INC, VT) &&
        !TLI.isIndexedStoreLegal(ISD::PRE_DEC, VT))
      return false;
    Ptr = ST->getBasePtr();
    isLoad = false;
  } else {
    return false;
  }

  // If the pointer is not an add/sub, or if it doesn't have multiple uses, bail
  // out.  There is no reason to make this a preinc/predec.
  if ((Ptr.getOpcode() != ISD::ADD && Ptr.getOpcode() != ISD::SUB) ||
      Ptr.getNode()->hasOneUse())
    return false;

  // Ask the target to do addressing mode selection.
  SDValue BasePtr;
  SDValue Offset;
  ISD::MemIndexedMode AM = ISD::UNINDEXED;
  if (!TLI.getPreIndexedAddressParts(N, BasePtr, Offset, AM, DAG))
    return false;

  // Backends without true r+i pre-indexed forms may need to pass a
  // constant base with a variable offset so that constant coercion
  // will work with the patterns in canonical form.
  bool Swapped = false;
  if (isa<ConstantSDNode>(BasePtr)) {
    std::swap(BasePtr, Offset);
    Swapped = true;
  }

  // Don't create a indexed load / store with zero offset.
  if (isNullConstant(Offset))
    return false;

  // Try turning it into a pre-indexed load / store except when:
  // 1) The new base ptr is a frame index.
  // 2) If N is a store and the new base ptr is either the same as or is a
  //    predecessor of the value being stored.
  // 3) Another use of old base ptr is a predecessor of N. If ptr is folded
  //    that would create a cycle.
  // 4) All uses are load / store ops that use it as old base ptr.

  // Check #1.  Preinc'ing a frame index would require copying the stack pointer
  // (plus the implicit offset) to a register to preinc anyway.
  if (isa<FrameIndexSDNode>(BasePtr) || isa<RegisterSDNode>(BasePtr))
    return false;

  // Check #2.
  if (!isLoad) {
    SDValue Val = cast<StoreSDNode>(N)->getValue();
    if (Val == BasePtr || BasePtr.getNode()->isPredecessorOf(Val.getNode()))
      return false;
  }

  // If the offset is a constant, there may be other adds of constants that
  // can be folded with this one. We should do this to avoid having to keep
  // a copy of the original base pointer.
  SmallVector<SDNode *, 16> OtherUses;
  if (isa<ConstantSDNode>(Offset))
    for (SDNode::use_iterator UI = BasePtr.getNode()->use_begin(),
                              UE = BasePtr.getNode()->use_end();
         UI != UE; ++UI) {
      SDUse &Use = UI.getUse();
      // Skip the use that is Ptr and uses of other results from BasePtr's
      // node (important for nodes that return multiple results).
      if (Use.getUser() == Ptr.getNode() || Use != BasePtr)
        continue;

      if (Use.getUser()->isPredecessorOf(N))
        continue;

      if (Use.getUser()->getOpcode() != ISD::ADD &&
          Use.getUser()->getOpcode() != ISD::SUB) {
        OtherUses.clear();
        break;
      }

      SDValue Op1 = Use.getUser()->getOperand((UI.getOperandNo() + 1) & 1);
      if (!isa<ConstantSDNode>(Op1)) {
        OtherUses.clear();
        break;
      }

      // FIXME: In some cases, we can be smarter about this.
      if (Op1.getValueType() != Offset.getValueType()) {
        OtherUses.clear();
        break;
      }

      OtherUses.push_back(Use.getUser());
    }

  if (Swapped)
    std::swap(BasePtr, Offset);

  // Now check for #3 and #4.
  bool RealUse = false;

  // Caches for hasPredecessorHelper
  SmallPtrSet<const SDNode *, 32> Visited;
  SmallVector<const SDNode *, 16> Worklist;

  for (SDNode *Use : Ptr.getNode()->uses()) {
    if (Use == N)
      continue;
    if (N->hasPredecessorHelper(Use, Visited, Worklist))
      return false;

    // If Ptr may be folded in addressing mode of other use, then it's
    // not profitable to do this transformation.
    if (!canFoldInAddressingMode(Ptr.getNode(), Use, DAG, TLI))
      RealUse = true;
  }

  if (!RealUse)
    return false;

  SDValue Result;
  if (isLoad)
    Result = DAG.getIndexedLoad(SDValue(N,0), SDLoc(N),
                                BasePtr, Offset, AM);
  else
    Result = DAG.getIndexedStore(SDValue(N,0), SDLoc(N),
                                 BasePtr, Offset, AM);
  ++PreIndexedNodes;
  ++NodesCombined;
  DEBUG(dbgs() << "\nReplacing.4 ";
        N->dump(&DAG);
        dbgs() << "\nWith: ";
        Result.getNode()->dump(&DAG);
        dbgs() << '\n');
  WorklistRemover DeadNodes(*this);
  if (isLoad) {
    DAG.ReplaceAllUsesOfValueWith(SDValue(N, 0), Result.getValue(0));
    DAG.ReplaceAllUsesOfValueWith(SDValue(N, 1), Result.getValue(2));
  } else {
    DAG.ReplaceAllUsesOfValueWith(SDValue(N, 0), Result.getValue(1));
  }

  // Finally, since the node is now dead, remove it from the graph.
  deleteAndRecombine(N);

  if (Swapped)
    std::swap(BasePtr, Offset);

  // Replace other uses of BasePtr that can be updated to use Ptr
  for (unsigned i = 0, e = OtherUses.size(); i != e; ++i) {
    unsigned OffsetIdx = 1;
    if (OtherUses[i]->getOperand(OffsetIdx).getNode() == BasePtr.getNode())
      OffsetIdx = 0;
    assert(OtherUses[i]->getOperand(!OffsetIdx).getNode() ==
           BasePtr.getNode() && "Expected BasePtr operand");

    // We need to replace ptr0 in the following expression:
    //   x0 * offset0 + y0 * ptr0 = t0
    // knowing that
    //   x1 * offset1 + y1 * ptr0 = t1 (the indexed load/store)
    //
    // where x0, x1, y0 and y1 in {-1, 1} are given by the types of the
    // indexed load/store and the expresion that needs to be re-written.
    //
    // Therefore, we have:
    //   t0 = (x0 * offset0 - x1 * y0 * y1 *offset1) + (y0 * y1) * t1

    ConstantSDNode *CN =
      cast<ConstantSDNode>(OtherUses[i]->getOperand(OffsetIdx));
    int X0, X1, Y0, Y1;
    APInt Offset0 = CN->getAPIntValue();
    APInt Offset1 = cast<ConstantSDNode>(Offset)->getAPIntValue();

    X0 = (OtherUses[i]->getOpcode() == ISD::SUB && OffsetIdx == 1) ? -1 : 1;
    Y0 = (OtherUses[i]->getOpcode() == ISD::SUB && OffsetIdx == 0) ? -1 : 1;
    X1 = (AM == ISD::PRE_DEC && !Swapped) ? -1 : 1;
    Y1 = (AM == ISD::PRE_DEC && Swapped) ? -1 : 1;

    unsigned Opcode = (Y0 * Y1 < 0) ? ISD::SUB : ISD::ADD;

    APInt CNV = Offset0;
    if (X0 < 0) CNV = -CNV;
    if (X1 * Y0 * Y1 < 0) CNV = CNV + Offset1;
    else CNV = CNV - Offset1;

    SDLoc DL(OtherUses[i]);

    // We can now generate the new expression.
    SDValue NewOp1 = DAG.getConstant(CNV, DL, CN->getValueType(0));
    SDValue NewOp2 = Result.getValue(isLoad ? 1 : 0);

    SDValue NewUse = DAG.getNode(Opcode,
                                 DL,
                                 OtherUses[i]->getValueType(0), NewOp1, NewOp2);
    DAG.ReplaceAllUsesOfValueWith(SDValue(OtherUses[i], 0), NewUse);
    deleteAndRecombine(OtherUses[i]);
  }

  // Replace the uses of Ptr with uses of the updated base value.
  DAG.ReplaceAllUsesOfValueWith(Ptr, Result.getValue(isLoad ? 1 : 0));
  deleteAndRecombine(Ptr.getNode());

  return true;
}

/// Try to combine a load/store with a add/sub of the base pointer node into a
/// post-indexed load/store. The transformation folded the add/subtract into the
/// new indexed load/store effectively and all of its uses are redirected to the
/// new load/store.
bool DAGCombiner::CombineToPostIndexedLoadStore(SDNode *N) {
  if (Level < AfterLegalizeDAG)
    return false;

  bool isLoad = true;
  SDValue Ptr;
  EVT VT;
  if (LoadSDNode *LD  = dyn_cast<LoadSDNode>(N)) {
    if (LD->isIndexed())
      return false;
    VT = LD->getMemoryVT();
    if (!TLI.isIndexedLoadLegal(ISD::POST_INC, VT) &&
        !TLI.isIndexedLoadLegal(ISD::POST_DEC, VT))
      return false;
    Ptr = LD->getBasePtr();
  } else if (StoreSDNode *ST  = dyn_cast<StoreSDNode>(N)) {
    if (ST->isIndexed())
      return false;
    VT = ST->getMemoryVT();
    if (!TLI.isIndexedStoreLegal(ISD::POST_INC, VT) &&
        !TLI.isIndexedStoreLegal(ISD::POST_DEC, VT))
      return false;
    Ptr = ST->getBasePtr();
    isLoad = false;
  } else {
    return false;
  }

  if (Ptr.getNode()->hasOneUse())
    return false;

  for (SDNode *Op : Ptr.getNode()->uses()) {
    if (Op == N ||
        (Op->getOpcode() != ISD::ADD && Op->getOpcode() != ISD::SUB))
      continue;

    SDValue BasePtr;
    SDValue Offset;
    ISD::MemIndexedMode AM = ISD::UNINDEXED;
    if (TLI.getPostIndexedAddressParts(N, Op, BasePtr, Offset, AM, DAG)) {
      // Don't create a indexed load / store with zero offset.
      if (isNullConstant(Offset))
        continue;

      // Try turning it into a post-indexed load / store except when
      // 1) All uses are load / store ops that use it as base ptr (and
      //    it may be folded as addressing mmode).
      // 2) Op must be independent of N, i.e. Op is neither a predecessor
      //    nor a successor of N. Otherwise, if Op is folded that would
      //    create a cycle.

      if (isa<FrameIndexSDNode>(BasePtr) || isa<RegisterSDNode>(BasePtr))
        continue;

      // Check for #1.
      bool TryNext = false;
      for (SDNode *Use : BasePtr.getNode()->uses()) {
        if (Use == Ptr.getNode())
          continue;

        // If all the uses are load / store addresses, then don't do the
        // transformation.
        if (Use->getOpcode() == ISD::ADD || Use->getOpcode() == ISD::SUB){
          bool RealUse = false;
          for (SDNode *UseUse : Use->uses()) {
            if (!canFoldInAddressingMode(Use, UseUse, DAG, TLI))
              RealUse = true;
          }

          if (!RealUse) {
            TryNext = true;
            break;
          }
        }
      }

      if (TryNext)
        continue;

      // Check for #2
      if (!Op->isPredecessorOf(N) && !N->isPredecessorOf(Op)) {
        SDValue Result = isLoad
          ? DAG.getIndexedLoad(SDValue(N,0), SDLoc(N),
                               BasePtr, Offset, AM)
          : DAG.getIndexedStore(SDValue(N,0), SDLoc(N),
                                BasePtr, Offset, AM);
        ++PostIndexedNodes;
        ++NodesCombined;
        DEBUG(dbgs() << "\nReplacing.5 ";
              N->dump(&DAG);
              dbgs() << "\nWith: ";
              Result.getNode()->dump(&DAG);
              dbgs() << '\n');
        WorklistRemover DeadNodes(*this);
        if (isLoad) {
          DAG.ReplaceAllUsesOfValueWith(SDValue(N, 0), Result.getValue(0));
          DAG.ReplaceAllUsesOfValueWith(SDValue(N, 1), Result.getValue(2));
        } else {
          DAG.ReplaceAllUsesOfValueWith(SDValue(N, 0), Result.getValue(1));
        }

        // Finally, since the node is now dead, remove it from the graph.
        deleteAndRecombine(N);

        // Replace the uses of Use with uses of the updated base value.
        DAG.ReplaceAllUsesOfValueWith(SDValue(Op, 0),
                                      Result.getValue(isLoad ? 1 : 0));
        deleteAndRecombine(Op);
        return true;
      }
    }
  }

  return false;
}

/// \brief Return the base-pointer arithmetic from an indexed \p LD.
SDValue DAGCombiner::SplitIndexingFromLoad(LoadSDNode *LD) {
  ISD::MemIndexedMode AM = LD->getAddressingMode();
  assert(AM != ISD::UNINDEXED);
  SDValue BP = LD->getOperand(1);
  SDValue Inc = LD->getOperand(2);

  // Some backends use TargetConstants for load offsets, but don't expect
  // TargetConstants in general ADD nodes. We can convert these constants into
  // regular Constants (if the constant is not opaque).
  assert((Inc.getOpcode() != ISD::TargetConstant ||
          !cast<ConstantSDNode>(Inc)->isOpaque()) &&
         "Cannot split out indexing using opaque target constants");
  if (Inc.getOpcode() == ISD::TargetConstant) {
    ConstantSDNode *ConstInc = cast<ConstantSDNode>(Inc);
    Inc = DAG.getConstant(*ConstInc->getConstantIntValue(), SDLoc(Inc),
                          ConstInc->getValueType(0));
  }

  unsigned Opc =
      (AM == ISD::PRE_INC || AM == ISD::POST_INC ? ISD::ADD : ISD::SUB);
  return DAG.getNode(Opc, SDLoc(LD), BP.getSimpleValueType(), BP, Inc);
}

SDValue DAGCombiner::visitLOAD(SDNode *N) {
  LoadSDNode *LD  = cast<LoadSDNode>(N);
  SDValue Chain = LD->getChain();
  SDValue Ptr   = LD->getBasePtr();

  // If load is not volatile and there are no uses of the loaded value (and
  // the updated indexed value in case of indexed loads), change uses of the
  // chain value into uses of the chain input (i.e. delete the dead load).
  if (!LD->isVolatile()) {
    if (N->getValueType(1) == MVT::Other) {
      // Unindexed loads.
      if (!N->hasAnyUseOfValue(0)) {
        // It's not safe to use the two value CombineTo variant here. e.g.
        // v1, chain2 = load chain1, loc
        // v2, chain3 = load chain2, loc
        // v3         = add v2, c
        // Now we replace use of chain2 with chain1.  This makes the second load
        // isomorphic to the one we are deleting, and thus makes this load live.
        DEBUG(dbgs() << "\nReplacing.6 ";
              N->dump(&DAG);
              dbgs() << "\nWith chain: ";
              Chain.getNode()->dump(&DAG);
              dbgs() << "\n");
        WorklistRemover DeadNodes(*this);
        DAG.ReplaceAllUsesOfValueWith(SDValue(N, 1), Chain);

        if (N->use_empty())
          deleteAndRecombine(N);

        return SDValue(N, 0);   // Return N so it doesn't get rechecked!
      }
    } else {
      // Indexed loads.
      assert(N->getValueType(2) == MVT::Other && "Malformed indexed loads?");

      // If this load has an opaque TargetConstant offset, then we cannot split
      // the indexing into an add/sub directly (that TargetConstant may not be
      // valid for a different type of node, and we cannot convert an opaque
      // target constant into a regular constant).
      bool HasOTCInc = LD->getOperand(2).getOpcode() == ISD::TargetConstant &&
                       cast<ConstantSDNode>(LD->getOperand(2))->isOpaque();

      if (!N->hasAnyUseOfValue(0) &&
          ((MaySplitLoadIndex && !HasOTCInc) || !N->hasAnyUseOfValue(1))) {
        SDValue Undef = DAG.getUNDEF(N->getValueType(0));
        SDValue Index;
        if (N->hasAnyUseOfValue(1) && MaySplitLoadIndex && !HasOTCInc) {
          Index = SplitIndexingFromLoad(LD);
          // Try to fold the base pointer arithmetic into subsequent loads and
          // stores.
          AddUsersToWorklist(N);
        } else
          Index = DAG.getUNDEF(N->getValueType(1));
        DEBUG(dbgs() << "\nReplacing.7 ";
              N->dump(&DAG);
              dbgs() << "\nWith: ";
              Undef.getNode()->dump(&DAG);
              dbgs() << " and 2 other values\n");
        WorklistRemover DeadNodes(*this);
        DAG.ReplaceAllUsesOfValueWith(SDValue(N, 0), Undef);
        DAG.ReplaceAllUsesOfValueWith(SDValue(N, 1), Index);
        DAG.ReplaceAllUsesOfValueWith(SDValue(N, 2), Chain);
        deleteAndRecombine(N);
        return SDValue(N, 0);   // Return N so it doesn't get rechecked!
      }
    }
  }

  // If this load is directly stored, replace the load value with the stored
  // value.
  // TODO: Handle store large -> read small portion.
  // TODO: Handle TRUNCSTORE/LOADEXT
  if (ISD::isNormalLoad(N) && !LD->isVolatile()) {
    if (ISD::isNON_TRUNCStore(Chain.getNode())) {
      StoreSDNode *PrevST = cast<StoreSDNode>(Chain);
      if (PrevST->getBasePtr() == Ptr &&
          PrevST->getValue().getValueType() == N->getValueType(0))
      return CombineTo(N, Chain.getOperand(1), Chain);
    }
  }

  // Try to infer better alignment information than the load already has.
  if (OptLevel != CodeGenOpt::None && LD->isUnindexed()) {
    if (unsigned Align = DAG.InferPtrAlignment(Ptr)) {
      if (Align > LD->getMemOperand()->getBaseAlignment()) {
        SDValue NewLoad =
               DAG.getExtLoad(LD->getExtensionType(), SDLoc(N),
                              LD->getValueType(0),
                              Chain, Ptr, LD->getPointerInfo(),
                              LD->getMemoryVT(),
                              LD->isVolatile(), LD->isNonTemporal(),
                              LD->isInvariant(), Align, LD->getAAInfo());
        if (NewLoad.getNode() != N)
          return CombineTo(N, NewLoad, SDValue(NewLoad.getNode(), 1), true);
      }
    }
  }

  bool UseAA = CombinerAA.getNumOccurrences() > 0 ? CombinerAA
                                                  : DAG.getSubtarget().useAA();
#ifndef NDEBUG
  if (CombinerAAOnlyFunc.getNumOccurrences() &&
      CombinerAAOnlyFunc != DAG.getMachineFunction().getName())
    UseAA = false;
#endif
  if (UseAA && LD->isUnindexed()) {
    // Walk up chain skipping non-aliasing memory nodes.
    SDValue BetterChain = FindBetterChain(N, Chain);

    // If there is a better chain.
    if (Chain != BetterChain) {
      SDValue ReplLoad;

      // Replace the chain to void dependency.
      if (LD->getExtensionType() == ISD::NON_EXTLOAD) {
        ReplLoad = DAG.getLoad(N->getValueType(0), SDLoc(LD),
                               BetterChain, Ptr, LD->getMemOperand());
      } else {
        ReplLoad = DAG.getExtLoad(LD->getExtensionType(), SDLoc(LD),
                                  LD->getValueType(0),
                                  BetterChain, Ptr, LD->getMemoryVT(),
                                  LD->getMemOperand());
      }

      // Create token factor to keep old chain connected.
      SDValue Token = DAG.getNode(ISD::TokenFactor, SDLoc(N),
                                  MVT::Other, Chain, ReplLoad.getValue(1));

      // Make sure the new and old chains are cleaned up.
      AddToWorklist(Token.getNode());

      // Replace uses with load result and token factor. Don't add users
      // to work list.
      return CombineTo(N, ReplLoad.getValue(0), Token, false);
    }
  }

  // Try transforming N to an indexed load.
  if (CombineToPreIndexedLoadStore(N) || CombineToPostIndexedLoadStore(N))
    return SDValue(N, 0);

  // Try to slice up N to more direct loads if the slices are mapped to
  // different register banks or pairing can take place.
  if (SliceUpLoad(N))
    return SDValue(N, 0);

  return SDValue();
}

namespace {
/// \brief Helper structure used to slice a load in smaller loads.
/// Basically a slice is obtained from the following sequence:
/// Origin = load Ty1, Base
/// Shift = srl Ty1 Origin, CstTy Amount
/// Inst = trunc Shift to Ty2
///
/// Then, it will be rewriten into:
/// Slice = load SliceTy, Base + SliceOffset
/// [Inst = zext Slice to Ty2], only if SliceTy <> Ty2
///
/// SliceTy is deduced from the number of bits that are actually used to
/// build Inst.
struct LoadedSlice {
  /// \brief Helper structure used to compute the cost of a slice.
  struct Cost {
    /// Are we optimizing for code size.
    bool ForCodeSize;
    /// Various cost.
    unsigned Loads;
    unsigned Truncates;
    unsigned CrossRegisterBanksCopies;
    unsigned ZExts;
    unsigned Shift;

    Cost(bool ForCodeSize = false)
        : ForCodeSize(ForCodeSize), Loads(0), Truncates(0),
          CrossRegisterBanksCopies(0), ZExts(0), Shift(0) {}

    /// \brief Get the cost of one isolated slice.
    Cost(const LoadedSlice &LS, bool ForCodeSize = false)
        : ForCodeSize(ForCodeSize), Loads(1), Truncates(0),
          CrossRegisterBanksCopies(0), ZExts(0), Shift(0) {
      EVT TruncType = LS.Inst->getValueType(0);
      EVT LoadedType = LS.getLoadedType();
      if (TruncType != LoadedType &&
          !LS.DAG->getTargetLoweringInfo().isZExtFree(LoadedType, TruncType))
        ZExts = 1;
    }

    /// \brief Account for slicing gain in the current cost.
    /// Slicing provide a few gains like removing a shift or a
    /// truncate. This method allows to grow the cost of the original
    /// load with the gain from this slice.
    void addSliceGain(const LoadedSlice &LS) {
      // Each slice saves a truncate.
      const TargetLowering &TLI = LS.DAG->getTargetLoweringInfo();
      if (!TLI.isTruncateFree(LS.Inst->getValueType(0),
                              LS.Inst->getOperand(0).getValueType()))
        ++Truncates;
      // If there is a shift amount, this slice gets rid of it.
      if (LS.Shift)
        ++Shift;
      // If this slice can merge a cross register bank copy, account for it.
      if (LS.canMergeExpensiveCrossRegisterBankCopy())
        ++CrossRegisterBanksCopies;
    }

    Cost &operator+=(const Cost &RHS) {
      Loads += RHS.Loads;
      Truncates += RHS.Truncates;
      CrossRegisterBanksCopies += RHS.CrossRegisterBanksCopies;
      ZExts += RHS.ZExts;
      Shift += RHS.Shift;
      return *this;
    }

    bool operator==(const Cost &RHS) const {
      return Loads == RHS.Loads && Truncates == RHS.Truncates &&
             CrossRegisterBanksCopies == RHS.CrossRegisterBanksCopies &&
             ZExts == RHS.ZExts && Shift == RHS.Shift;
    }

    bool operator!=(const Cost &RHS) const { return !(*this == RHS); }

    bool operator<(const Cost &RHS) const {
      // Assume cross register banks copies are as expensive as loads.
      // FIXME: Do we want some more target hooks?
      unsigned ExpensiveOpsLHS = Loads + CrossRegisterBanksCopies;
      unsigned ExpensiveOpsRHS = RHS.Loads + RHS.CrossRegisterBanksCopies;
      // Unless we are optimizing for code size, consider the
      // expensive operation first.
      if (!ForCodeSize && ExpensiveOpsLHS != ExpensiveOpsRHS)
        return ExpensiveOpsLHS < ExpensiveOpsRHS;
      return (Truncates + ZExts + Shift + ExpensiveOpsLHS) <
             (RHS.Truncates + RHS.ZExts + RHS.Shift + ExpensiveOpsRHS);
    }

    bool operator>(const Cost &RHS) const { return RHS < *this; }

    bool operator<=(const Cost &RHS) const { return !(RHS < *this); }

    bool operator>=(const Cost &RHS) const { return !(*this < RHS); }
  };
  // The last instruction that represent the slice. This should be a
  // truncate instruction.
  SDNode *Inst;
  // The original load instruction.
  LoadSDNode *Origin;
  // The right shift amount in bits from the original load.
  unsigned Shift;
  // The DAG from which Origin came from.
  // This is used to get some contextual information about legal types, etc.
  SelectionDAG *DAG;

  LoadedSlice(SDNode *Inst = nullptr, LoadSDNode *Origin = nullptr,
              unsigned Shift = 0, SelectionDAG *DAG = nullptr)
      : Inst(Inst), Origin(Origin), Shift(Shift), DAG(DAG) {}

  /// \brief Get the bits used in a chunk of bits \p BitWidth large.
  /// \return Result is \p BitWidth and has used bits set to 1 and
  ///         not used bits set to 0.
  APInt getUsedBits() const {
    // Reproduce the trunc(lshr) sequence:
    // - Start from the truncated value.
    // - Zero extend to the desired bit width.
    // - Shift left.
    assert(Origin && "No original load to compare against.");
    unsigned BitWidth = Origin->getValueSizeInBits(0);
    assert(Inst && "This slice is not bound to an instruction");
    assert(Inst->getValueSizeInBits(0) <= BitWidth &&
           "Extracted slice is bigger than the whole type!");
    APInt UsedBits(Inst->getValueSizeInBits(0), 0);
    UsedBits.setAllBits();
    UsedBits = UsedBits.zext(BitWidth);
    UsedBits <<= Shift;
    return UsedBits;
  }

  /// \brief Get the size of the slice to be loaded in bytes.
  unsigned getLoadedSize() const {
    unsigned SliceSize = getUsedBits().countPopulation();
    assert(!(SliceSize & 0x7) && "Size is not a multiple of a byte.");
    return SliceSize / 8;
  }

  /// \brief Get the type that will be loaded for this slice.
  /// Note: This may not be the final type for the slice.
  EVT getLoadedType() const {
    assert(DAG && "Missing context");
    LLVMContext &Ctxt = *DAG->getContext();
    return EVT::getIntegerVT(Ctxt, getLoadedSize() * 8);
  }

  /// \brief Get the alignment of the load used for this slice.
  unsigned getAlignment() const {
    unsigned Alignment = Origin->getAlignment();
    unsigned Offset = getOffsetFromBase();
    if (Offset != 0)
      Alignment = MinAlign(Alignment, Alignment + Offset);
    return Alignment;
  }

  /// \brief Check if this slice can be rewritten with legal operations.
  bool isLegal() const {
    // An invalid slice is not legal.
    if (!Origin || !Inst || !DAG)
      return false;

    // Offsets are for indexed load only, we do not handle that.
    if (Origin->getOffset().getOpcode() != ISD::UNDEF)
      return false;

    const TargetLowering &TLI = DAG->getTargetLoweringInfo();

    // Check that the type is legal.
    EVT SliceType = getLoadedType();
    if (!TLI.isTypeLegal(SliceType))
      return false;

    // Check that the load is legal for this type.
    if (!TLI.isOperationLegal(ISD::LOAD, SliceType))
      return false;

    // Check that the offset can be computed.
    // 1. Check its type.
    EVT PtrType = Origin->getBasePtr().getValueType();
    if (PtrType == MVT::Untyped || PtrType.isExtended())
      return false;

    // 2. Check that it fits in the immediate.
    if (!TLI.isLegalAddImmediate(getOffsetFromBase()))
      return false;

    // 3. Check that the computation is legal.
    if (!TLI.isOperationLegal(ISD::ADD, PtrType))
      return false;

    // Check that the zext is legal if it needs one.
    EVT TruncateType = Inst->getValueType(0);
    if (TruncateType != SliceType &&
        !TLI.isOperationLegal(ISD::ZERO_EXTEND, TruncateType))
      return false;

    return true;
  }

  /// \brief Get the offset in bytes of this slice in the original chunk of
  /// bits.
  /// \pre DAG != nullptr.
  uint64_t getOffsetFromBase() const {
    assert(DAG && "Missing context.");
    bool IsBigEndian = DAG->getDataLayout().isBigEndian();
    assert(!(Shift & 0x7) && "Shifts not aligned on Bytes are not supported.");
    uint64_t Offset = Shift / 8;
    unsigned TySizeInBytes = Origin->getValueSizeInBits(0) / 8;
    assert(!(Origin->getValueSizeInBits(0) & 0x7) &&
           "The size of the original loaded type is not a multiple of a"
           " byte.");
    // If Offset is bigger than TySizeInBytes, it means we are loading all
    // zeros. This should have been optimized before in the process.
    assert(TySizeInBytes > Offset &&
           "Invalid shift amount for given loaded size");
    if (IsBigEndian)
      Offset = TySizeInBytes - Offset - getLoadedSize();
    return Offset;
  }

  /// \brief Generate the sequence of instructions to load the slice
  /// represented by this object and redirect the uses of this slice to
  /// this new sequence of instructions.
  /// \pre this->Inst && this->Origin are valid Instructions and this
  /// object passed the legal check: LoadedSlice::isLegal returned true.
  /// \return The last instruction of the sequence used to load the slice.
  SDValue loadSlice() const {
    assert(Inst && Origin && "Unable to replace a non-existing slice.");
    const SDValue &OldBaseAddr = Origin->getBasePtr();
    SDValue BaseAddr = OldBaseAddr;
    // Get the offset in that chunk of bytes w.r.t. the endianess.
    int64_t Offset = static_cast<int64_t>(getOffsetFromBase());
    assert(Offset >= 0 && "Offset too big to fit in int64_t!");
    if (Offset) {
      // BaseAddr = BaseAddr + Offset.
      EVT ArithType = BaseAddr.getValueType();
      SDLoc DL(Origin);
      BaseAddr = DAG->getNode(ISD::ADD, DL, ArithType, BaseAddr,
                              DAG->getConstant(Offset, DL, ArithType));
    }

    // Create the type of the loaded slice according to its size.
    EVT SliceType = getLoadedType();

    // Create the load for the slice.
    SDValue LastInst = DAG->getLoad(
        SliceType, SDLoc(Origin), Origin->getChain(), BaseAddr,
        Origin->getPointerInfo().getWithOffset(Offset), Origin->isVolatile(),
        Origin->isNonTemporal(), Origin->isInvariant(), getAlignment());
    // If the final type is not the same as the loaded type, this means that
    // we have to pad with zero. Create a zero extend for that.
    EVT FinalType = Inst->getValueType(0);
    if (SliceType != FinalType)
      LastInst =
          DAG->getNode(ISD::ZERO_EXTEND, SDLoc(LastInst), FinalType, LastInst);
    return LastInst;
  }

  /// \brief Check if this slice can be merged with an expensive cross register
  /// bank copy. E.g.,
  /// i = load i32
  /// f = bitcast i32 i to float
  bool canMergeExpensiveCrossRegisterBankCopy() const {
    if (!Inst || !Inst->hasOneUse())
      return false;
    SDNode *Use = *Inst->use_begin();
    if (Use->getOpcode() != ISD::BITCAST)
      return false;
    assert(DAG && "Missing context");
    const TargetLowering &TLI = DAG->getTargetLoweringInfo();
    EVT ResVT = Use->getValueType(0);
    const TargetRegisterClass *ResRC = TLI.getRegClassFor(ResVT.getSimpleVT());
    const TargetRegisterClass *ArgRC =
        TLI.getRegClassFor(Use->getOperand(0).getValueType().getSimpleVT());
    if (ArgRC == ResRC || !TLI.isOperationLegal(ISD::LOAD, ResVT))
      return false;

    // At this point, we know that we perform a cross-register-bank copy.
    // Check if it is expensive.
    const TargetRegisterInfo *TRI = DAG->getSubtarget().getRegisterInfo();
    // Assume bitcasts are cheap, unless both register classes do not
    // explicitly share a common sub class.
    if (!TRI || TRI->getCommonSubClass(ArgRC, ResRC))
      return false;

    // Check if it will be merged with the load.
    // 1. Check the alignment constraint.
    unsigned RequiredAlignment = DAG->getDataLayout().getABITypeAlignment(
        ResVT.getTypeForEVT(*DAG->getContext()));

    if (RequiredAlignment > getAlignment())
      return false;

    // 2. Check that the load is a legal operation for that type.
    if (!TLI.isOperationLegal(ISD::LOAD, ResVT))
      return false;

    // 3. Check that we do not have a zext in the way.
    if (Inst->getValueType(0) != getLoadedType())
      return false;

    return true;
  }
};
}

/// \brief Check that all bits set in \p UsedBits form a dense region, i.e.,
/// \p UsedBits looks like 0..0 1..1 0..0.
static bool areUsedBitsDense(const APInt &UsedBits) {
  // If all the bits are one, this is dense!
  if (UsedBits.isAllOnesValue())
    return true;

  // Get rid of the unused bits on the right.
  APInt NarrowedUsedBits = UsedBits.lshr(UsedBits.countTrailingZeros());
  // Get rid of the unused bits on the left.
  if (NarrowedUsedBits.countLeadingZeros())
    NarrowedUsedBits = NarrowedUsedBits.trunc(NarrowedUsedBits.getActiveBits());
  // Check that the chunk of bits is completely used.
  return NarrowedUsedBits.isAllOnesValue();
}

/// \brief Check whether or not \p First and \p Second are next to each other
/// in memory. This means that there is no hole between the bits loaded
/// by \p First and the bits loaded by \p Second.
static bool areSlicesNextToEachOther(const LoadedSlice &First,
                                     const LoadedSlice &Second) {
  assert(First.Origin == Second.Origin && First.Origin &&
         "Unable to match different memory origins.");
  APInt UsedBits = First.getUsedBits();
  assert((UsedBits & Second.getUsedBits()) == 0 &&
         "Slices are not supposed to overlap.");
  UsedBits |= Second.getUsedBits();
  return areUsedBitsDense(UsedBits);
}

/// \brief Adjust the \p GlobalLSCost according to the target
/// paring capabilities and the layout of the slices.
/// \pre \p GlobalLSCost should account for at least as many loads as
/// there is in the slices in \p LoadedSlices.
static void adjustCostForPairing(SmallVectorImpl<LoadedSlice> &LoadedSlices,
                                 LoadedSlice::Cost &GlobalLSCost) {
  unsigned NumberOfSlices = LoadedSlices.size();
  // If there is less than 2 elements, no pairing is possible.
  if (NumberOfSlices < 2)
    return;

  // Sort the slices so that elements that are likely to be next to each
  // other in memory are next to each other in the list.
  std::sort(LoadedSlices.begin(), LoadedSlices.end(),
            [](const LoadedSlice &LHS, const LoadedSlice &RHS) {
    assert(LHS.Origin == RHS.Origin && "Different bases not implemented.");
    return LHS.getOffsetFromBase() < RHS.getOffsetFromBase();
  });
  const TargetLowering &TLI = LoadedSlices[0].DAG->getTargetLoweringInfo();
  // First (resp. Second) is the first (resp. Second) potentially candidate
  // to be placed in a paired load.
  const LoadedSlice *First = nullptr;
  const LoadedSlice *Second = nullptr;
  for (unsigned CurrSlice = 0; CurrSlice < NumberOfSlices; ++CurrSlice,
                // Set the beginning of the pair.
                                                           First = Second) {

    Second = &LoadedSlices[CurrSlice];

    // If First is NULL, it means we start a new pair.
    // Get to the next slice.
    if (!First)
      continue;

    EVT LoadedType = First->getLoadedType();

    // If the types of the slices are different, we cannot pair them.
    if (LoadedType != Second->getLoadedType())
      continue;

    // Check if the target supplies paired loads for this type.
    unsigned RequiredAlignment = 0;
    if (!TLI.hasPairedLoad(LoadedType, RequiredAlignment)) {
      // move to the next pair, this type is hopeless.
      Second = nullptr;
      continue;
    }
    // Check if we meet the alignment requirement.
    if (RequiredAlignment > First->getAlignment())
      continue;

    // Check that both loads are next to each other in memory.
    if (!areSlicesNextToEachOther(*First, *Second))
      continue;

    assert(GlobalLSCost.Loads > 0 && "We save more loads than we created!");
    --GlobalLSCost.Loads;
    // Move to the next pair.
    Second = nullptr;
  }
}

/// \brief Check the profitability of all involved LoadedSlice.
/// Currently, it is considered profitable if there is exactly two
/// involved slices (1) which are (2) next to each other in memory, and
/// whose cost (\see LoadedSlice::Cost) is smaller than the original load (3).
///
/// Note: The order of the elements in \p LoadedSlices may be modified, but not
/// the elements themselves.
///
/// FIXME: When the cost model will be mature enough, we can relax
/// constraints (1) and (2).
static bool isSlicingProfitable(SmallVectorImpl<LoadedSlice> &LoadedSlices,
                                const APInt &UsedBits, bool ForCodeSize) {
  unsigned NumberOfSlices = LoadedSlices.size();
  if (StressLoadSlicing)
    return NumberOfSlices > 1;

  // Check (1).
  if (NumberOfSlices != 2)
    return false;

  // Check (2).
  if (!areUsedBitsDense(UsedBits))
    return false;

  // Check (3).
  LoadedSlice::Cost OrigCost(ForCodeSize), GlobalSlicingCost(ForCodeSize);
  // The original code has one big load.
  OrigCost.Loads = 1;
  for (unsigned CurrSlice = 0; CurrSlice < NumberOfSlices; ++CurrSlice) {
    const LoadedSlice &LS = LoadedSlices[CurrSlice];
    // Accumulate the cost of all the slices.
    LoadedSlice::Cost SliceCost(LS, ForCodeSize);
    GlobalSlicingCost += SliceCost;

    // Account as cost in the original configuration the gain obtained
    // with the current slices.
    OrigCost.addSliceGain(LS);
  }

  // If the target supports paired load, adjust the cost accordingly.
  adjustCostForPairing(LoadedSlices, GlobalSlicingCost);
  return OrigCost > GlobalSlicingCost;
}

/// \brief If the given load, \p LI, is used only by trunc or trunc(lshr)
/// operations, split it in the various pieces being extracted.
///
/// This sort of thing is introduced by SROA.
/// This slicing takes care not to insert overlapping loads.
/// \pre LI is a simple load (i.e., not an atomic or volatile load).
bool DAGCombiner::SliceUpLoad(SDNode *N) {
  if (Level < AfterLegalizeDAG)
    return false;

  LoadSDNode *LD = cast<LoadSDNode>(N);
  if (LD->isVolatile() || !ISD::isNormalLoad(LD) ||
      !LD->getValueType(0).isInteger())
    return false;

  // Keep track of already used bits to detect overlapping values.
  // In that case, we will just abort the transformation.
  APInt UsedBits(LD->getValueSizeInBits(0), 0);

  SmallVector<LoadedSlice, 4> LoadedSlices;

  // Check if this load is used as several smaller chunks of bits.
  // Basically, look for uses in trunc or trunc(lshr) and record a new chain
  // of computation for each trunc.
  for (SDNode::use_iterator UI = LD->use_begin(), UIEnd = LD->use_end();
       UI != UIEnd; ++UI) {
    // Skip the uses of the chain.
    if (UI.getUse().getResNo() != 0)
      continue;

    SDNode *User = *UI;
    unsigned Shift = 0;

    // Check if this is a trunc(lshr).
    if (User->getOpcode() == ISD::SRL && User->hasOneUse() &&
        isa<ConstantSDNode>(User->getOperand(1))) {
      Shift = cast<ConstantSDNode>(User->getOperand(1))->getZExtValue();
      User = *User->use_begin();
    }

    // At this point, User is a Truncate, iff we encountered, trunc or
    // trunc(lshr).
    if (User->getOpcode() != ISD::TRUNCATE)
      return false;

    // The width of the type must be a power of 2 and greater than 8-bits.
    // Otherwise the load cannot be represented in LLVM IR.
    // Moreover, if we shifted with a non-8-bits multiple, the slice
    // will be across several bytes. We do not support that.
    unsigned Width = User->getValueSizeInBits(0);
    if (Width < 8 || !isPowerOf2_32(Width) || (Shift & 0x7))
      return 0;

    // Build the slice for this chain of computations.
    LoadedSlice LS(User, LD, Shift, &DAG);
    APInt CurrentUsedBits = LS.getUsedBits();

    // Check if this slice overlaps with another.
    if ((CurrentUsedBits & UsedBits) != 0)
      return false;
    // Update the bits used globally.
    UsedBits |= CurrentUsedBits;

    // Check if the new slice would be legal.
    if (!LS.isLegal())
      return false;

    // Record the slice.
    LoadedSlices.push_back(LS);
  }

  // Abort slicing if it does not seem to be profitable.
  if (!isSlicingProfitable(LoadedSlices, UsedBits, ForCodeSize))
    return false;

  ++SlicedLoads;

  // Rewrite each chain to use an independent load.
  // By construction, each chain can be represented by a unique load.

  // Prepare the argument for the new token factor for all the slices.
  SmallVector<SDValue, 8> ArgChains;
  for (SmallVectorImpl<LoadedSlice>::const_iterator
           LSIt = LoadedSlices.begin(),
           LSItEnd = LoadedSlices.end();
       LSIt != LSItEnd; ++LSIt) {
    SDValue SliceInst = LSIt->loadSlice();
    CombineTo(LSIt->Inst, SliceInst, true);
    if (SliceInst.getNode()->getOpcode() != ISD::LOAD)
      SliceInst = SliceInst.getOperand(0);
    assert(SliceInst->getOpcode() == ISD::LOAD &&
           "It takes more than a zext to get to the loaded slice!!");
    ArgChains.push_back(SliceInst.getValue(1));
  }

  SDValue Chain = DAG.getNode(ISD::TokenFactor, SDLoc(LD), MVT::Other,
                              ArgChains);
  DAG.ReplaceAllUsesOfValueWith(SDValue(N, 1), Chain);
  return true;
}

/// Check to see if V is (and load (ptr), imm), where the load is having
/// specific bytes cleared out.  If so, return the byte size being masked out
/// and the shift amount.
static std::pair<unsigned, unsigned>
CheckForMaskedLoad(SDValue V, SDValue Ptr, SDValue Chain) {
  std::pair<unsigned, unsigned> Result(0, 0);

  // Check for the structure we're looking for.
  if (V->getOpcode() != ISD::AND ||
      !isa<ConstantSDNode>(V->getOperand(1)) ||
      !ISD::isNormalLoad(V->getOperand(0).getNode()))
    return Result;

  // Check the chain and pointer.
  LoadSDNode *LD = cast<LoadSDNode>(V->getOperand(0));
  if (LD->getBasePtr() != Ptr) return Result;  // Not from same pointer.

  // The store should be chained directly to the load or be an operand of a
  // tokenfactor.
  if (LD == Chain.getNode())
    ; // ok.
  else if (Chain->getOpcode() != ISD::TokenFactor)
    return Result; // Fail.
  else {
    bool isOk = false;
    for (const SDValue &ChainOp : Chain->op_values())
      if (ChainOp.getNode() == LD) {
        isOk = true;
        break;
      }
    if (!isOk) return Result;
  }

  // This only handles simple types.
  if (V.getValueType() != MVT::i16 &&
      V.getValueType() != MVT::i32 &&
      V.getValueType() != MVT::i64)
    return Result;

  // Check the constant mask.  Invert it so that the bits being masked out are
  // 0 and the bits being kept are 1.  Use getSExtValue so that leading bits
  // follow the sign bit for uniformity.
  uint64_t NotMask = ~cast<ConstantSDNode>(V->getOperand(1))->getSExtValue();
  unsigned NotMaskLZ = countLeadingZeros(NotMask);
  if (NotMaskLZ & 7) return Result;  // Must be multiple of a byte.
  unsigned NotMaskTZ = countTrailingZeros(NotMask);
  if (NotMaskTZ & 7) return Result;  // Must be multiple of a byte.
  if (NotMaskLZ == 64) return Result;  // All zero mask.

  // See if we have a continuous run of bits.  If so, we have 0*1+0*
  if (countTrailingOnes(NotMask >> NotMaskTZ) + NotMaskTZ + NotMaskLZ != 64)
    return Result;

  // Adjust NotMaskLZ down to be from the actual size of the int instead of i64.
  if (V.getValueType() != MVT::i64 && NotMaskLZ)
    NotMaskLZ -= 64-V.getValueSizeInBits();

  unsigned MaskedBytes = (V.getValueSizeInBits()-NotMaskLZ-NotMaskTZ)/8;
  switch (MaskedBytes) {
  case 1:
  case 2:
  case 4: break;
  default: return Result; // All one mask, or 5-byte mask.
  }

  // Verify that the first bit starts at a multiple of mask so that the access
  // is aligned the same as the access width.
  if (NotMaskTZ && NotMaskTZ/8 % MaskedBytes) return Result;

  Result.first = MaskedBytes;
  Result.second = NotMaskTZ/8;
  return Result;
}


/// Check to see if IVal is something that provides a value as specified by
/// MaskInfo. If so, replace the specified store with a narrower store of
/// truncated IVal.
static SDNode *
ShrinkLoadReplaceStoreWithStore(const std::pair<unsigned, unsigned> &MaskInfo,
                                SDValue IVal, StoreSDNode *St,
                                DAGCombiner *DC) {
  unsigned NumBytes = MaskInfo.first;
  unsigned ByteShift = MaskInfo.second;
  SelectionDAG &DAG = DC->getDAG();

  // Check to see if IVal is all zeros in the part being masked in by the 'or'
  // that uses this.  If not, this is not a replacement.
  APInt Mask = ~APInt::getBitsSet(IVal.getValueSizeInBits(),
                                  ByteShift*8, (ByteShift+NumBytes)*8);
  if (!DAG.MaskedValueIsZero(IVal, Mask)) return nullptr;

  // Check that it is legal on the target to do this.  It is legal if the new
  // VT we're shrinking to (i8/i16/i32) is legal or we're still before type
  // legalization.
  MVT VT = MVT::getIntegerVT(NumBytes*8);
  if (!DC->isTypeLegal(VT))
    return nullptr;

  // Okay, we can do this!  Replace the 'St' store with a store of IVal that is
  // shifted by ByteShift and truncated down to NumBytes.
  if (ByteShift) {
    SDLoc DL(IVal);
    IVal = DAG.getNode(ISD::SRL, DL, IVal.getValueType(), IVal,
                       DAG.getConstant(ByteShift*8, DL,
                                    DC->getShiftAmountTy(IVal.getValueType())));
  }

  // Figure out the offset for the store and the alignment of the access.
  unsigned StOffset;
  unsigned NewAlign = St->getAlignment();

  if (DAG.getDataLayout().isLittleEndian())
    StOffset = ByteShift;
  else
    StOffset = IVal.getValueType().getStoreSize() - ByteShift - NumBytes;

  SDValue Ptr = St->getBasePtr();
  if (StOffset) {
    SDLoc DL(IVal);
    Ptr = DAG.getNode(ISD::ADD, DL, Ptr.getValueType(),
                      Ptr, DAG.getConstant(StOffset, DL, Ptr.getValueType()));
    NewAlign = MinAlign(NewAlign, StOffset);
  }

  // Truncate down to the new size.
  IVal = DAG.getNode(ISD::TRUNCATE, SDLoc(IVal), VT, IVal);

  ++OpsNarrowed;
  return DAG.getStore(St->getChain(), SDLoc(St), IVal, Ptr,
                      St->getPointerInfo().getWithOffset(StOffset),
                      false, false, NewAlign).getNode();
}


/// Look for sequence of load / op / store where op is one of 'or', 'xor', and
/// 'and' of immediates. If 'op' is only touching some of the loaded bits, try
/// narrowing the load and store if it would end up being a win for performance
/// or code size.
SDValue DAGCombiner::ReduceLoadOpStoreWidth(SDNode *N) {
  StoreSDNode *ST  = cast<StoreSDNode>(N);
  if (ST->isVolatile())
    return SDValue();

  SDValue Chain = ST->getChain();
  SDValue Value = ST->getValue();
  SDValue Ptr   = ST->getBasePtr();
  EVT VT = Value.getValueType();

  if (ST->isTruncatingStore() || VT.isVector() || !Value.hasOneUse())
    return SDValue();

  unsigned Opc = Value.getOpcode();

  // If this is "store (or X, Y), P" and X is "(and (load P), cst)", where cst
  // is a byte mask indicating a consecutive number of bytes, check to see if
  // Y is known to provide just those bytes.  If so, we try to replace the
  // load + replace + store sequence with a single (narrower) store, which makes
  // the load dead.
  if (Opc == ISD::OR) {
    std::pair<unsigned, unsigned> MaskedLoad;
    MaskedLoad = CheckForMaskedLoad(Value.getOperand(0), Ptr, Chain);
    if (MaskedLoad.first)
      if (SDNode *NewST = ShrinkLoadReplaceStoreWithStore(MaskedLoad,
                                                  Value.getOperand(1), ST,this))
        return SDValue(NewST, 0);

    // Or is commutative, so try swapping X and Y.
    MaskedLoad = CheckForMaskedLoad(Value.getOperand(1), Ptr, Chain);
    if (MaskedLoad.first)
      if (SDNode *NewST = ShrinkLoadReplaceStoreWithStore(MaskedLoad,
                                                  Value.getOperand(0), ST,this))
        return SDValue(NewST, 0);
  }

  if ((Opc != ISD::OR && Opc != ISD::XOR && Opc != ISD::AND) ||
      Value.getOperand(1).getOpcode() != ISD::Constant)
    return SDValue();

  SDValue N0 = Value.getOperand(0);
  if (ISD::isNormalLoad(N0.getNode()) && N0.hasOneUse() &&
      Chain == SDValue(N0.getNode(), 1)) {
    LoadSDNode *LD = cast<LoadSDNode>(N0);
    if (LD->getBasePtr() != Ptr ||
        LD->getPointerInfo().getAddrSpace() !=
        ST->getPointerInfo().getAddrSpace())
      return SDValue();

    // Find the type to narrow it the load / op / store to.
    SDValue N1 = Value.getOperand(1);
    unsigned BitWidth = N1.getValueSizeInBits();
    APInt Imm = cast<ConstantSDNode>(N1)->getAPIntValue();
    if (Opc == ISD::AND)
      Imm ^= APInt::getAllOnesValue(BitWidth);
    if (Imm == 0 || Imm.isAllOnesValue())
      return SDValue();
    unsigned ShAmt = Imm.countTrailingZeros();
    unsigned MSB = BitWidth - Imm.countLeadingZeros() - 1;
    unsigned NewBW = NextPowerOf2(MSB - ShAmt);
    EVT NewVT = EVT::getIntegerVT(*DAG.getContext(), NewBW);
    // The narrowing should be profitable, the load/store operation should be
    // legal (or custom) and the store size should be equal to the NewVT width.
    while (NewBW < BitWidth &&
           (NewVT.getStoreSizeInBits() != NewBW ||
            !TLI.isOperationLegalOrCustom(Opc, NewVT) ||
            !TLI.isNarrowingProfitable(VT, NewVT))) {
      NewBW = NextPowerOf2(NewBW);
      NewVT = EVT::getIntegerVT(*DAG.getContext(), NewBW);
    }
    if (NewBW >= BitWidth)
      return SDValue();

    // If the lsb changed does not start at the type bitwidth boundary,
    // start at the previous one.
    if (ShAmt % NewBW)
      ShAmt = (((ShAmt + NewBW - 1) / NewBW) * NewBW) - NewBW;
    APInt Mask = APInt::getBitsSet(BitWidth, ShAmt,
                                   std::min(BitWidth, ShAmt + NewBW));
    if ((Imm & Mask) == Imm) {
      APInt NewImm = (Imm & Mask).lshr(ShAmt).trunc(NewBW);
      if (Opc == ISD::AND)
        NewImm ^= APInt::getAllOnesValue(NewBW);
      uint64_t PtrOff = ShAmt / 8;
      // For big endian targets, we need to adjust the offset to the pointer to
      // load the correct bytes.
      if (DAG.getDataLayout().isBigEndian())
        PtrOff = (BitWidth + 7 - NewBW) / 8 - PtrOff;

      unsigned NewAlign = MinAlign(LD->getAlignment(), PtrOff);
      Type *NewVTTy = NewVT.getTypeForEVT(*DAG.getContext());
      if (NewAlign < DAG.getDataLayout().getABITypeAlignment(NewVTTy))
        return SDValue();

      SDValue NewPtr = DAG.getNode(ISD::ADD, SDLoc(LD),
                                   Ptr.getValueType(), Ptr,
                                   DAG.getConstant(PtrOff, SDLoc(LD),
                                                   Ptr.getValueType()));
      SDValue NewLD = DAG.getLoad(NewVT, SDLoc(N0),
                                  LD->getChain(), NewPtr,
                                  LD->getPointerInfo().getWithOffset(PtrOff),
                                  LD->isVolatile(), LD->isNonTemporal(),
                                  LD->isInvariant(), NewAlign,
                                  LD->getAAInfo());
      SDValue NewVal = DAG.getNode(Opc, SDLoc(Value), NewVT, NewLD,
                                   DAG.getConstant(NewImm, SDLoc(Value),
                                                   NewVT));
      SDValue NewST = DAG.getStore(Chain, SDLoc(N),
                                   NewVal, NewPtr,
                                   ST->getPointerInfo().getWithOffset(PtrOff),
                                   false, false, NewAlign);

      AddToWorklist(NewPtr.getNode());
      AddToWorklist(NewLD.getNode());
      AddToWorklist(NewVal.getNode());
      WorklistRemover DeadNodes(*this);
      DAG.ReplaceAllUsesOfValueWith(N0.getValue(1), NewLD.getValue(1));
      ++OpsNarrowed;
      return NewST;
    }
  }

  return SDValue();
}

/// For a given floating point load / store pair, if the load value isn't used
/// by any other operations, then consider transforming the pair to integer
/// load / store operations if the target deems the transformation profitable.
SDValue DAGCombiner::TransformFPLoadStorePair(SDNode *N) {
  StoreSDNode *ST  = cast<StoreSDNode>(N);
  SDValue Chain = ST->getChain();
  SDValue Value = ST->getValue();
  if (ISD::isNormalStore(ST) && ISD::isNormalLoad(Value.getNode()) &&
      Value.hasOneUse() &&
      Chain == SDValue(Value.getNode(), 1)) {
    LoadSDNode *LD = cast<LoadSDNode>(Value);
    EVT VT = LD->getMemoryVT();
    if (!VT.isFloatingPoint() ||
        VT != ST->getMemoryVT() ||
        LD->isNonTemporal() ||
        ST->isNonTemporal() ||
        LD->getPointerInfo().getAddrSpace() != 0 ||
        ST->getPointerInfo().getAddrSpace() != 0)
      return SDValue();

    EVT IntVT = EVT::getIntegerVT(*DAG.getContext(), VT.getSizeInBits());
    if (!TLI.isOperationLegal(ISD::LOAD, IntVT) ||
        !TLI.isOperationLegal(ISD::STORE, IntVT) ||
        !TLI.isDesirableToTransformToIntegerOp(ISD::LOAD, VT) ||
        !TLI.isDesirableToTransformToIntegerOp(ISD::STORE, VT))
      return SDValue();

    unsigned LDAlign = LD->getAlignment();
    unsigned STAlign = ST->getAlignment();
    Type *IntVTTy = IntVT.getTypeForEVT(*DAG.getContext());
    unsigned ABIAlign = DAG.getDataLayout().getABITypeAlignment(IntVTTy);
    if (LDAlign < ABIAlign || STAlign < ABIAlign)
      return SDValue();

    SDValue NewLD = DAG.getLoad(IntVT, SDLoc(Value),
                                LD->getChain(), LD->getBasePtr(),
                                LD->getPointerInfo(),
                                false, false, false, LDAlign);

    SDValue NewST = DAG.getStore(NewLD.getValue(1), SDLoc(N),
                                 NewLD, ST->getBasePtr(),
                                 ST->getPointerInfo(),
                                 false, false, STAlign);

    AddToWorklist(NewLD.getNode());
    AddToWorklist(NewST.getNode());
    WorklistRemover DeadNodes(*this);
    DAG.ReplaceAllUsesOfValueWith(Value.getValue(1), NewLD.getValue(1));
    ++LdStFP2Int;
    return NewST;
  }

  return SDValue();
}

namespace {
/// Helper struct to parse and store a memory address as base + index + offset.
/// We ignore sign extensions when it is safe to do so.
/// The following two expressions are not equivalent. To differentiate we need
/// to store whether there was a sign extension involved in the index
/// computation.
///  (load (i64 add (i64 copyfromreg %c)
///                 (i64 signextend (add (i8 load %index)
///                                      (i8 1))))
/// vs
///
/// (load (i64 add (i64 copyfromreg %c)
///                (i64 signextend (i32 add (i32 signextend (i8 load %index))
///                                         (i32 1)))))
struct BaseIndexOffset {
  SDValue Base;
  SDValue Index;
  int64_t Offset;
  bool IsIndexSignExt;

  BaseIndexOffset() : Offset(0), IsIndexSignExt(false) {}

  BaseIndexOffset(SDValue Base, SDValue Index, int64_t Offset,
                  bool IsIndexSignExt) :
    Base(Base), Index(Index), Offset(Offset), IsIndexSignExt(IsIndexSignExt) {}

  bool equalBaseIndex(const BaseIndexOffset &Other) {
    return Other.Base == Base && Other.Index == Index &&
      Other.IsIndexSignExt == IsIndexSignExt;
  }

  /// Parses tree in Ptr for base, index, offset addresses.
  static BaseIndexOffset match(SDValue Ptr) {
    bool IsIndexSignExt = false;

    // We only can pattern match BASE + INDEX + OFFSET. If Ptr is not an ADD
    // instruction, then it could be just the BASE or everything else we don't
    // know how to handle. Just use Ptr as BASE and give up.
    if (Ptr->getOpcode() != ISD::ADD)
      return BaseIndexOffset(Ptr, SDValue(), 0, IsIndexSignExt);

    // We know that we have at least an ADD instruction. Try to pattern match
    // the simple case of BASE + OFFSET.
    if (isa<ConstantSDNode>(Ptr->getOperand(1))) {
      int64_t Offset = cast<ConstantSDNode>(Ptr->getOperand(1))->getSExtValue();
      return  BaseIndexOffset(Ptr->getOperand(0), SDValue(), Offset,
                              IsIndexSignExt);
    }

    // Inside a loop the current BASE pointer is calculated using an ADD and a
    // MUL instruction. In this case Ptr is the actual BASE pointer.
    // (i64 add (i64 %array_ptr)
    //          (i64 mul (i64 %induction_var)
    //                   (i64 %element_size)))
    if (Ptr->getOperand(1)->getOpcode() == ISD::MUL)
      return BaseIndexOffset(Ptr, SDValue(), 0, IsIndexSignExt);

    // Look at Base + Index + Offset cases.
    SDValue Base = Ptr->getOperand(0);
    SDValue IndexOffset = Ptr->getOperand(1);

    // Skip signextends.
    if (IndexOffset->getOpcode() == ISD::SIGN_EXTEND) {
      IndexOffset = IndexOffset->getOperand(0);
      IsIndexSignExt = true;
    }

    // Either the case of Base + Index (no offset) or something else.
    if (IndexOffset->getOpcode() != ISD::ADD)
      return BaseIndexOffset(Base, IndexOffset, 0, IsIndexSignExt);

    // Now we have the case of Base + Index + offset.
    SDValue Index = IndexOffset->getOperand(0);
    SDValue Offset = IndexOffset->getOperand(1);

    if (!isa<ConstantSDNode>(Offset))
      return BaseIndexOffset(Ptr, SDValue(), 0, IsIndexSignExt);

    // Ignore signextends.
    if (Index->getOpcode() == ISD::SIGN_EXTEND) {
      Index = Index->getOperand(0);
      IsIndexSignExt = true;
    } else IsIndexSignExt = false;

    int64_t Off = cast<ConstantSDNode>(Offset)->getSExtValue();
    return BaseIndexOffset(Base, Index, Off, IsIndexSignExt);
  }
};
} // namespace

SDValue DAGCombiner::getMergedConstantVectorStore(SelectionDAG &DAG,
                                                  SDLoc SL,
                                                  ArrayRef<MemOpLink> Stores,
                                                  EVT Ty) const {
  SmallVector<SDValue, 8> BuildVector;

  for (unsigned I = 0, E = Ty.getVectorNumElements(); I != E; ++I)
    BuildVector.push_back(cast<StoreSDNode>(Stores[I].MemNode)->getValue());

  return DAG.getNode(ISD::BUILD_VECTOR, SL, Ty, BuildVector);
}

bool DAGCombiner::MergeStoresOfConstantsOrVecElts(
                  SmallVectorImpl<MemOpLink> &StoreNodes, EVT MemVT,
                  unsigned NumElem, bool IsConstantSrc, bool UseVector) {
  // Make sure we have something to merge.
  if (NumElem < 2)
    return false;

  int64_t ElementSizeBytes = MemVT.getSizeInBits() / 8;
  LSBaseSDNode *FirstInChain = StoreNodes[0].MemNode;
  unsigned LatestNodeUsed = 0;

  for (unsigned i=0; i < NumElem; ++i) {
    // Find a chain for the new wide-store operand. Notice that some
    // of the store nodes that we found may not be selected for inclusion
    // in the wide store. The chain we use needs to be the chain of the
    // latest store node which is *used* and replaced by the wide store.
    if (StoreNodes[i].SequenceNum < StoreNodes[LatestNodeUsed].SequenceNum)
      LatestNodeUsed = i;
  }

  // The latest Node in the DAG.
  LSBaseSDNode *LatestOp = StoreNodes[LatestNodeUsed].MemNode;
  SDLoc DL(StoreNodes[0].MemNode);

  SDValue StoredVal;
  if (UseVector) {
    // Find a legal type for the vector store.
    EVT Ty = EVT::getVectorVT(*DAG.getContext(), MemVT, NumElem);
    assert(TLI.isTypeLegal(Ty) && "Illegal vector store");
    if (IsConstantSrc) {
      StoredVal = getMergedConstantVectorStore(DAG, DL, StoreNodes, Ty);
    } else {
      SmallVector<SDValue, 8> Ops;
      for (unsigned i = 0; i < NumElem ; ++i) {
        StoreSDNode *St = cast<StoreSDNode>(StoreNodes[i].MemNode);
        SDValue Val = St->getValue();
        // All of the operands of a BUILD_VECTOR must have the same type.
        if (Val.getValueType() != MemVT)
          return false;
        Ops.push_back(Val);
      }

      // Build the extracted vector elements back into a vector.
      StoredVal = DAG.getNode(ISD::BUILD_VECTOR, DL, Ty, Ops);
    }
  } else {
    // We should always use a vector store when merging extracted vector
    // elements, so this path implies a store of constants.
    assert(IsConstantSrc && "Merged vector elements should use vector store");

    unsigned SizeInBits = NumElem * ElementSizeBytes * 8;
    APInt StoreInt(SizeInBits, 0);

    // Construct a single integer constant which is made of the smaller
    // constant inputs.
    bool IsLE = DAG.getDataLayout().isLittleEndian();
    for (unsigned i = 0; i < NumElem ; ++i) {
      unsigned Idx = IsLE ? (NumElem - 1 - i) : i;
      StoreSDNode *St  = cast<StoreSDNode>(StoreNodes[Idx].MemNode);
      SDValue Val = St->getValue();
      StoreInt <<= ElementSizeBytes * 8;
      if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Val)) {
        StoreInt |= C->getAPIntValue().zext(SizeInBits);
      } else if (ConstantFPSDNode *C = dyn_cast<ConstantFPSDNode>(Val)) {
        StoreInt |= C->getValueAPF().bitcastToAPInt().zext(SizeInBits);
      } else {
        llvm_unreachable("Invalid constant element type");
      }
    }

    // Create the new Load and Store operations.
    EVT StoreTy = EVT::getIntegerVT(*DAG.getContext(), SizeInBits);
    StoredVal = DAG.getConstant(StoreInt, DL, StoreTy);
  }

  SDValue NewStore = DAG.getStore(LatestOp->getChain(), DL, StoredVal,
                                  FirstInChain->getBasePtr(),
                                  FirstInChain->getPointerInfo(),
                                  false, false,
                                  FirstInChain->getAlignment());

  // Replace the last store with the new store
  CombineTo(LatestOp, NewStore);
  // Erase all other stores.
  for (unsigned i = 0; i < NumElem ; ++i) {
    if (StoreNodes[i].MemNode == LatestOp)
      continue;
    StoreSDNode *St = cast<StoreSDNode>(StoreNodes[i].MemNode);
    // ReplaceAllUsesWith will replace all uses that existed when it was
    // called, but graph optimizations may cause new ones to appear. For
    // example, the case in pr14333 looks like
    //
    //  St's chain -> St -> another store -> X
    //
    // And the only difference from St to the other store is the chain.
    // When we change it's chain to be St's chain they become identical,
    // get CSEed and the net result is that X is now a use of St.
    // Since we know that St is redundant, just iterate.
    while (!St->use_empty())
      DAG.ReplaceAllUsesWith(SDValue(St, 0), St->getChain());
    deleteAndRecombine(St);
  }

  return true;
}

static bool allowableAlignment(const SelectionDAG &DAG,
                               const TargetLowering &TLI, EVT EVTTy,
                               unsigned AS, unsigned Align) {
  if (TLI.allowsMisalignedMemoryAccesses(EVTTy, AS, Align))
    return true;

  Type *Ty = EVTTy.getTypeForEVT(*DAG.getContext());
  unsigned ABIAlignment = DAG.getDataLayout().getPrefTypeAlignment(Ty);
  return (Align >= ABIAlignment);
}

void DAGCombiner::getStoreMergeAndAliasCandidates(
    StoreSDNode* St, SmallVectorImpl<MemOpLink> &StoreNodes,
    SmallVectorImpl<LSBaseSDNode*> &AliasLoadNodes) {
  // This holds the base pointer, index, and the offset in bytes from the base
  // pointer.
  BaseIndexOffset BasePtr = BaseIndexOffset::match(St->getBasePtr());

  // We must have a base and an offset.
  if (!BasePtr.Base.getNode())
    return;

  // Do not handle stores to undef base pointers.
  if (BasePtr.Base.getOpcode() == ISD::UNDEF)
    return;

  // Walk up the chain and look for nodes with offsets from the same
  // base pointer. Stop when reaching an instruction with a different kind
  // or instruction which has a different base pointer.
  EVT MemVT = St->getMemoryVT();
  unsigned Seq = 0;
  StoreSDNode *Index = St;
  while (Index) {
    // If the chain has more than one use, then we can't reorder the mem ops.
    if (Index != St && !SDValue(Index, 0)->hasOneUse())
      break;

    // Find the base pointer and offset for this memory node.
    BaseIndexOffset Ptr = BaseIndexOffset::match(Index->getBasePtr());

    // Check that the base pointer is the same as the original one.
    if (!Ptr.equalBaseIndex(BasePtr))
      break;

    // The memory operands must not be volatile.
    if (Index->isVolatile() || Index->isIndexed())
      break;

    // No truncation.
    if (StoreSDNode *St = dyn_cast<StoreSDNode>(Index))
      if (St->isTruncatingStore())
        break;

    // The stored memory type must be the same.
    if (Index->getMemoryVT() != MemVT)
      break;

    // We found a potential memory operand to merge.
    StoreNodes.push_back(MemOpLink(Index, Ptr.Offset, Seq++));

    // Find the next memory operand in the chain. If the next operand in the
    // chain is a store then move up and continue the scan with the next
    // memory operand. If the next operand is a load save it and use alias
    // information to check if it interferes with anything.
    SDNode *NextInChain = Index->getChain().getNode();
    while (1) {
      if (StoreSDNode *STn = dyn_cast<StoreSDNode>(NextInChain)) {
        // We found a store node. Use it for the next iteration.
        Index = STn;
        break;
      } else if (LoadSDNode *Ldn = dyn_cast<LoadSDNode>(NextInChain)) {
        if (Ldn->isVolatile()) {
          Index = nullptr;
          break;
        }

        // Save the load node for later. Continue the scan.
        AliasLoadNodes.push_back(Ldn);
        NextInChain = Ldn->getChain().getNode();
        continue;
      } else {
        Index = nullptr;
        break;
      }
    }
  }
}

bool DAGCombiner::MergeConsecutiveStores(StoreSDNode* St) {
  if (OptLevel == CodeGenOpt::None)
    return false;

  EVT MemVT = St->getMemoryVT();
  int64_t ElementSizeBytes = MemVT.getSizeInBits() / 8;
  bool NoVectors = DAG.getMachineFunction().getFunction()->hasFnAttribute(
      Attribute::NoImplicitFloat);

  // This function cannot currently deal with non-byte-sized memory sizes.
  if (ElementSizeBytes * 8 != MemVT.getSizeInBits())
    return false;

  // Don't merge vectors into wider inputs.
  if (MemVT.isVector() || !MemVT.isSimple())
    return false;

  // Perform an early exit check. Do not bother looking at stored values that
  // are not constants, loads, or extracted vector elements.
  SDValue StoredVal = St->getValue();
  bool IsLoadSrc = isa<LoadSDNode>(StoredVal);
  bool IsConstantSrc = isa<ConstantSDNode>(StoredVal) ||
                       isa<ConstantFPSDNode>(StoredVal);
  bool IsExtractVecEltSrc = (StoredVal.getOpcode() == ISD::EXTRACT_VECTOR_ELT);

  if (!IsConstantSrc && !IsLoadSrc && !IsExtractVecEltSrc)
    return false;

  // Only look at ends of store sequences.
  SDValue Chain = SDValue(St, 0);
  if (Chain->hasOneUse() && Chain->use_begin()->getOpcode() == ISD::STORE)
    return false;

  // Save the LoadSDNodes that we find in the chain.
  // We need to make sure that these nodes do not interfere with
  // any of the store nodes.
  SmallVector<LSBaseSDNode*, 8> AliasLoadNodes;
  
  // Save the StoreSDNodes that we find in the chain.
  SmallVector<MemOpLink, 8> StoreNodes;

  getStoreMergeAndAliasCandidates(St, StoreNodes, AliasLoadNodes);
  
  // Check if there is anything to merge.
  if (StoreNodes.size() < 2)
    return false;

  // Sort the memory operands according to their distance from the base pointer.
  std::sort(StoreNodes.begin(), StoreNodes.end(),
            [](MemOpLink LHS, MemOpLink RHS) {
    return LHS.OffsetFromBase < RHS.OffsetFromBase ||
           (LHS.OffsetFromBase == RHS.OffsetFromBase &&
            LHS.SequenceNum > RHS.SequenceNum);
  });

  // Scan the memory operations on the chain and find the first non-consecutive
  // store memory address.
  unsigned LastConsecutiveStore = 0;
  int64_t StartAddress = StoreNodes[0].OffsetFromBase;
  for (unsigned i = 0, e = StoreNodes.size(); i < e; ++i) {

    // Check that the addresses are consecutive starting from the second
    // element in the list of stores.
    if (i > 0) {
      int64_t CurrAddress = StoreNodes[i].OffsetFromBase;
      if (CurrAddress - StartAddress != (ElementSizeBytes * i))
        break;
    }

    bool Alias = false;
    // Check if this store interferes with any of the loads that we found.
    for (unsigned ld = 0, lde = AliasLoadNodes.size(); ld < lde; ++ld)
      if (isAlias(AliasLoadNodes[ld], StoreNodes[i].MemNode)) {
        Alias = true;
        break;
      }
    // We found a load that alias with this store. Stop the sequence.
    if (Alias)
      break;

    // Mark this node as useful.
    LastConsecutiveStore = i;
  }

  // The node with the lowest store address.
  LSBaseSDNode *FirstInChain = StoreNodes[0].MemNode;
  unsigned FirstStoreAS = FirstInChain->getAddressSpace();
  unsigned FirstStoreAlign = FirstInChain->getAlignment();

  // Store the constants into memory as one consecutive store.
  if (IsConstantSrc) {
    unsigned LastLegalType = 0;
    unsigned LastLegalVectorType = 0;
    bool NonZero = false;
    for (unsigned i=0; i<LastConsecutiveStore+1; ++i) {
      StoreSDNode *St  = cast<StoreSDNode>(StoreNodes[i].MemNode);
      SDValue StoredVal = St->getValue();

      if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(StoredVal)) {
        NonZero |= !C->isNullValue();
      } else if (ConstantFPSDNode *C = dyn_cast<ConstantFPSDNode>(StoredVal)) {
        NonZero |= !C->getConstantFPValue()->isNullValue();
      } else {
        // Non-constant.
        break;
      }

      // Find a legal type for the constant store.
      unsigned SizeInBits = (i+1) * ElementSizeBytes * 8;
      EVT StoreTy = EVT::getIntegerVT(*DAG.getContext(), SizeInBits);
      if (TLI.isTypeLegal(StoreTy) &&
          allowableAlignment(DAG, TLI, StoreTy, FirstStoreAS,
                             FirstStoreAlign)) {
        LastLegalType = i+1;
      // Or check whether a truncstore is legal.
      } else if (TLI.getTypeAction(*DAG.getContext(), StoreTy) ==
                 TargetLowering::TypePromoteInteger) {
        EVT LegalizedStoredValueTy =
          TLI.getTypeToTransformTo(*DAG.getContext(), StoredVal.getValueType());
        if (TLI.isTruncStoreLegal(LegalizedStoredValueTy, StoreTy) &&
            allowableAlignment(DAG, TLI, LegalizedStoredValueTy, FirstStoreAS,
                               FirstStoreAlign)) {
          LastLegalType = i + 1;
        }
      }

      // Find a legal type for the vector store.
      EVT Ty = EVT::getVectorVT(*DAG.getContext(), MemVT, i+1);
      if (TLI.isTypeLegal(Ty) &&
          allowableAlignment(DAG, TLI, Ty, FirstStoreAS, FirstStoreAlign)) {
        LastLegalVectorType = i + 1;
      }
    }


    // We only use vectors if the constant is known to be zero or the target
    // allows it and the function is not marked with the noimplicitfloat
    // attribute.
    if (NoVectors) {
      LastLegalVectorType = 0;
    } else if (NonZero && !TLI.storeOfVectorConstantIsCheap(MemVT,
                                                            LastLegalVectorType,
                                                            FirstStoreAS)) {
      LastLegalVectorType = 0;
    }

    // Check if we found a legal integer type to store.
    if (LastLegalType == 0 && LastLegalVectorType == 0)
      return false;

    bool UseVector = (LastLegalVectorType > LastLegalType) && !NoVectors;
    unsigned NumElem = UseVector ? LastLegalVectorType : LastLegalType;

    return MergeStoresOfConstantsOrVecElts(StoreNodes, MemVT, NumElem,
                                           true, UseVector);
  }

  // When extracting multiple vector elements, try to store them
  // in one vector store rather than a sequence of scalar stores.
  if (IsExtractVecEltSrc) {
    unsigned NumElem = 0;
    for (unsigned i = 0; i < LastConsecutiveStore + 1; ++i) {
      StoreSDNode *St  = cast<StoreSDNode>(StoreNodes[i].MemNode);
      SDValue StoredVal = St->getValue();
      // This restriction could be loosened.
      // Bail out if any stored values are not elements extracted from a vector.
      // It should be possible to handle mixed sources, but load sources need
      // more careful handling (see the block of code below that handles
      // consecutive loads).
      if (StoredVal.getOpcode() != ISD::EXTRACT_VECTOR_ELT)
        return false;

      // Find a legal type for the vector store.
      EVT Ty = EVT::getVectorVT(*DAG.getContext(), MemVT, i+1);
      if (TLI.isTypeLegal(Ty) &&
          allowableAlignment(DAG, TLI, Ty, FirstStoreAS, FirstStoreAlign))
        NumElem = i + 1;
    }

    return MergeStoresOfConstantsOrVecElts(StoreNodes, MemVT, NumElem,
                                           false, true);
  }

  // Below we handle the case of multiple consecutive stores that
  // come from multiple consecutive loads. We merge them into a single
  // wide load and a single wide store.

  // Look for load nodes which are used by the stored values.
  SmallVector<MemOpLink, 8> LoadNodes;

  // Find acceptable loads. Loads need to have the same chain (token factor),
  // must not be zext, volatile, indexed, and they must be consecutive.
  BaseIndexOffset LdBasePtr;
  for (unsigned i=0; i<LastConsecutiveStore+1; ++i) {
    StoreSDNode *St  = cast<StoreSDNode>(StoreNodes[i].MemNode);
    LoadSDNode *Ld = dyn_cast<LoadSDNode>(St->getValue());
    if (!Ld) break;

    // Loads must only have one use.
    if (!Ld->hasNUsesOfValue(1, 0))
      break;

    // The memory operands must not be volatile.
    if (Ld->isVolatile() || Ld->isIndexed())
      break;

    // We do not accept ext loads.
    if (Ld->getExtensionType() != ISD::NON_EXTLOAD)
      break;

    // The stored memory type must be the same.
    if (Ld->getMemoryVT() != MemVT)
      break;

    BaseIndexOffset LdPtr = BaseIndexOffset::match(Ld->getBasePtr());
    // If this is not the first ptr that we check.
    if (LdBasePtr.Base.getNode()) {
      // The base ptr must be the same.
      if (!LdPtr.equalBaseIndex(LdBasePtr))
        break;
    } else {
      // Check that all other base pointers are the same as this one.
      LdBasePtr = LdPtr;
    }

    // We found a potential memory operand to merge.
    LoadNodes.push_back(MemOpLink(Ld, LdPtr.Offset, 0));
  }

  if (LoadNodes.size() < 2)
    return false;

  // If we have load/store pair instructions and we only have two values,
  // don't bother.
  unsigned RequiredAlignment;
  if (LoadNodes.size() == 2 && TLI.hasPairedLoad(MemVT, RequiredAlignment) &&
      St->getAlignment() >= RequiredAlignment)
    return false;

  LoadSDNode *FirstLoad = cast<LoadSDNode>(LoadNodes[0].MemNode);
  unsigned FirstLoadAS = FirstLoad->getAddressSpace();
  unsigned FirstLoadAlign = FirstLoad->getAlignment();

  // Scan the memory operations on the chain and find the first non-consecutive
  // load memory address. These variables hold the index in the store node
  // array.
  unsigned LastConsecutiveLoad = 0;
  // This variable refers to the size and not index in the array.
  unsigned LastLegalVectorType = 0;
  unsigned LastLegalIntegerType = 0;
  StartAddress = LoadNodes[0].OffsetFromBase;
  SDValue FirstChain = FirstLoad->getChain();
  for (unsigned i = 1; i < LoadNodes.size(); ++i) {
    // All loads much share the same chain.
    if (LoadNodes[i].MemNode->getChain() != FirstChain)
      break;

    int64_t CurrAddress = LoadNodes[i].OffsetFromBase;
    if (CurrAddress - StartAddress != (ElementSizeBytes * i))
      break;
    LastConsecutiveLoad = i;

    // Find a legal type for the vector store.
    EVT StoreTy = EVT::getVectorVT(*DAG.getContext(), MemVT, i+1);
    if (TLI.isTypeLegal(StoreTy) &&
        allowableAlignment(DAG, TLI, StoreTy, FirstStoreAS, FirstStoreAlign) &&
        allowableAlignment(DAG, TLI, StoreTy, FirstLoadAS, FirstLoadAlign)) {
      LastLegalVectorType = i + 1;
    }

    // Find a legal type for the integer store.
    unsigned SizeInBits = (i+1) * ElementSizeBytes * 8;
    StoreTy = EVT::getIntegerVT(*DAG.getContext(), SizeInBits);
    if (TLI.isTypeLegal(StoreTy) &&
        allowableAlignment(DAG, TLI, StoreTy, FirstStoreAS, FirstStoreAlign) &&
        allowableAlignment(DAG, TLI, StoreTy, FirstLoadAS, FirstLoadAlign))
      LastLegalIntegerType = i + 1;
    // Or check whether a truncstore and extload is legal.
    else if (TLI.getTypeAction(*DAG.getContext(), StoreTy) ==
             TargetLowering::TypePromoteInteger) {
      EVT LegalizedStoredValueTy =
        TLI.getTypeToTransformTo(*DAG.getContext(), StoreTy);
      if (TLI.isTruncStoreLegal(LegalizedStoredValueTy, StoreTy) &&
          TLI.isLoadExtLegal(ISD::ZEXTLOAD, LegalizedStoredValueTy, StoreTy) &&
          TLI.isLoadExtLegal(ISD::SEXTLOAD, LegalizedStoredValueTy, StoreTy) &&
          TLI.isLoadExtLegal(ISD::EXTLOAD, LegalizedStoredValueTy, StoreTy) &&
          allowableAlignment(DAG, TLI, LegalizedStoredValueTy, FirstStoreAS,
                             FirstStoreAlign) &&
          allowableAlignment(DAG, TLI, LegalizedStoredValueTy, FirstLoadAS,
                             FirstLoadAlign))
        LastLegalIntegerType = i+1;
    }
  }

  // Only use vector types if the vector type is larger than the integer type.
  // If they are the same, use integers.
  bool UseVectorTy = LastLegalVectorType > LastLegalIntegerType && !NoVectors;
  unsigned LastLegalType = std::max(LastLegalVectorType, LastLegalIntegerType);

  // We add +1 here because the LastXXX variables refer to location while
  // the NumElem refers to array/index size.
  unsigned NumElem = std::min(LastConsecutiveStore, LastConsecutiveLoad) + 1;
  NumElem = std::min(LastLegalType, NumElem);

  if (NumElem < 2)
    return false;

  // The latest Node in the DAG.
  unsigned LatestNodeUsed = 0;
  for (unsigned i=1; i<NumElem; ++i) {
    // Find a chain for the new wide-store operand. Notice that some
    // of the store nodes that we found may not be selected for inclusion
    // in the wide store. The chain we use needs to be the chain of the
    // latest store node which is *used* and replaced by the wide store.
    if (StoreNodes[i].SequenceNum < StoreNodes[LatestNodeUsed].SequenceNum)
      LatestNodeUsed = i;
  }

  LSBaseSDNode *LatestOp = StoreNodes[LatestNodeUsed].MemNode;

  // Find if it is better to use vectors or integers to load and store
  // to memory.
  EVT JointMemOpVT;
  if (UseVectorTy) {
    JointMemOpVT = EVT::getVectorVT(*DAG.getContext(), MemVT, NumElem);
  } else {
    unsigned SizeInBits = NumElem * ElementSizeBytes * 8;
    JointMemOpVT = EVT::getIntegerVT(*DAG.getContext(), SizeInBits);
  }

  SDLoc LoadDL(LoadNodes[0].MemNode);
  SDLoc StoreDL(StoreNodes[0].MemNode);

  SDValue NewLoad = DAG.getLoad(
      JointMemOpVT, LoadDL, FirstLoad->getChain(), FirstLoad->getBasePtr(),
      FirstLoad->getPointerInfo(), false, false, false, FirstLoadAlign);

  SDValue NewStore = DAG.getStore(
      LatestOp->getChain(), StoreDL, NewLoad, FirstInChain->getBasePtr(),
      FirstInChain->getPointerInfo(), false, false, FirstStoreAlign);

  // Replace one of the loads with the new load.
  LoadSDNode *Ld = cast<LoadSDNode>(LoadNodes[0].MemNode);
  DAG.ReplaceAllUsesOfValueWith(SDValue(Ld, 1),
                                SDValue(NewLoad.getNode(), 1));

  // Remove the rest of the load chains.
  for (unsigned i = 1; i < NumElem ; ++i) {
    // Replace all chain users of the old load nodes with the chain of the new
    // load node.
    LoadSDNode *Ld = cast<LoadSDNode>(LoadNodes[i].MemNode);
    DAG.ReplaceAllUsesOfValueWith(SDValue(Ld, 1), Ld->getChain());
  }

  // Replace the last store with the new store.
  CombineTo(LatestOp, NewStore);
  // Erase all other stores.
  for (unsigned i = 0; i < NumElem ; ++i) {
    // Remove all Store nodes.
    if (StoreNodes[i].MemNode == LatestOp)
      continue;
    StoreSDNode *St = cast<StoreSDNode>(StoreNodes[i].MemNode);
    DAG.ReplaceAllUsesOfValueWith(SDValue(St, 0), St->getChain());
    deleteAndRecombine(St);
  }

  return true;
}

SDValue DAGCombiner::visitSTORE(SDNode *N) {
  StoreSDNode *ST  = cast<StoreSDNode>(N);
  SDValue Chain = ST->getChain();
  SDValue Value = ST->getValue();
  SDValue Ptr   = ST->getBasePtr();

  // If this is a store of a bit convert, store the input value if the
  // resultant store does not need a higher alignment than the original.
  if (Value.getOpcode() == ISD::BITCAST && !ST->isTruncatingStore() &&
      ST->isUnindexed()) {
    unsigned OrigAlign = ST->getAlignment();
    EVT SVT = Value.getOperand(0).getValueType();
    unsigned Align = DAG.getDataLayout().getABITypeAlignment(
        SVT.getTypeForEVT(*DAG.getContext()));
    if (Align <= OrigAlign &&
        ((!LegalOperations && !ST->isVolatile()) ||
         TLI.isOperationLegalOrCustom(ISD::STORE, SVT)))
      return DAG.getStore(Chain, SDLoc(N), Value.getOperand(0),
                          Ptr, ST->getPointerInfo(), ST->isVolatile(),
                          ST->isNonTemporal(), OrigAlign,
                          ST->getAAInfo());
  }

  // Turn 'store undef, Ptr' -> nothing.
  if (Value.getOpcode() == ISD::UNDEF && ST->isUnindexed())
    return Chain;

  // Turn 'store float 1.0, Ptr' -> 'store int 0x12345678, Ptr'
  if (ConstantFPSDNode *CFP = dyn_cast<ConstantFPSDNode>(Value)) {
    // NOTE: If the original store is volatile, this transform must not increase
    // the number of stores.  For example, on x86-32 an f64 can be stored in one
    // processor operation but an i64 (which is not legal) requires two.  So the
    // transform should not be done in this case.
    if (Value.getOpcode() != ISD::TargetConstantFP) {
      SDValue Tmp;
      switch (CFP->getSimpleValueType(0).SimpleTy) {
      default: llvm_unreachable("Unknown FP type");
      case MVT::f16:    // We don't do this for these yet.
      case MVT::f80:
      case MVT::f128:
      case MVT::ppcf128:
        break;
      case MVT::f32:
        if ((isTypeLegal(MVT::i32) && !LegalOperations && !ST->isVolatile()) ||
            TLI.isOperationLegalOrCustom(ISD::STORE, MVT::i32)) {
          ;
          Tmp = DAG.getConstant((uint32_t)CFP->getValueAPF().
                              bitcastToAPInt().getZExtValue(), SDLoc(CFP),
                              MVT::i32);
          return DAG.getStore(Chain, SDLoc(N), Tmp,
                              Ptr, ST->getMemOperand());
        }
        break;
      case MVT::f64:
        if ((TLI.isTypeLegal(MVT::i64) && !LegalOperations &&
             !ST->isVolatile()) ||
            TLI.isOperationLegalOrCustom(ISD::STORE, MVT::i64)) {
          ;
          Tmp = DAG.getConstant(CFP->getValueAPF().bitcastToAPInt().
                                getZExtValue(), SDLoc(CFP), MVT::i64);
          return DAG.getStore(Chain, SDLoc(N), Tmp,
                              Ptr, ST->getMemOperand());
        }

        if (!ST->isVolatile() &&
            TLI.isOperationLegalOrCustom(ISD::STORE, MVT::i32)) {
          // Many FP stores are not made apparent until after legalize, e.g. for
          // argument passing.  Since this is so common, custom legalize the
          // 64-bit integer store into two 32-bit stores.
          uint64_t Val = CFP->getValueAPF().bitcastToAPInt().getZExtValue();
          SDValue Lo = DAG.getConstant(Val & 0xFFFFFFFF, SDLoc(CFP), MVT::i32);
          SDValue Hi = DAG.getConstant(Val >> 32, SDLoc(CFP), MVT::i32);
          if (DAG.getDataLayout().isBigEndian())
            std::swap(Lo, Hi);

          unsigned Alignment = ST->getAlignment();
          bool isVolatile = ST->isVolatile();
          bool isNonTemporal = ST->isNonTemporal();
          AAMDNodes AAInfo = ST->getAAInfo();

          SDLoc DL(N);

          SDValue St0 = DAG.getStore(Chain, SDLoc(ST), Lo,
                                     Ptr, ST->getPointerInfo(),
                                     isVolatile, isNonTemporal,
                                     ST->getAlignment(), AAInfo);
          Ptr = DAG.getNode(ISD::ADD, DL, Ptr.getValueType(), Ptr,
                            DAG.getConstant(4, DL, Ptr.getValueType()));
          Alignment = MinAlign(Alignment, 4U);
          SDValue St1 = DAG.getStore(Chain, SDLoc(ST), Hi,
                                     Ptr, ST->getPointerInfo().getWithOffset(4),
                                     isVolatile, isNonTemporal,
                                     Alignment, AAInfo);
          return DAG.getNode(ISD::TokenFactor, DL, MVT::Other,
                             St0, St1);
        }

        break;
      }
    }
  }

  // Try to infer better alignment information than the store already has.
  if (OptLevel != CodeGenOpt::None && ST->isUnindexed()) {
    if (unsigned Align = DAG.InferPtrAlignment(Ptr)) {
      if (Align > ST->getAlignment()) {
        SDValue NewStore =
               DAG.getTruncStore(Chain, SDLoc(N), Value,
                                 Ptr, ST->getPointerInfo(), ST->getMemoryVT(),
                                 ST->isVolatile(), ST->isNonTemporal(), Align,
                                 ST->getAAInfo());
        if (NewStore.getNode() != N)
          return CombineTo(ST, NewStore, true);
      }
    }
  }

  // Try transforming a pair floating point load / store ops to integer
  // load / store ops.
  SDValue NewST = TransformFPLoadStorePair(N);
  if (NewST.getNode())
    return NewST;

  bool UseAA = CombinerAA.getNumOccurrences() > 0 ? CombinerAA
                                                  : DAG.getSubtarget().useAA();
#ifndef NDEBUG
  if (CombinerAAOnlyFunc.getNumOccurrences() &&
      CombinerAAOnlyFunc != DAG.getMachineFunction().getName())
    UseAA = false;
#endif
  if (UseAA && ST->isUnindexed()) {
    // Walk up chain skipping non-aliasing memory nodes.
    SDValue BetterChain = FindBetterChain(N, Chain);

    // If there is a better chain.
    if (Chain != BetterChain) {
      SDValue ReplStore;

      // Replace the chain to avoid dependency.
      if (ST->isTruncatingStore()) {
        ReplStore = DAG.getTruncStore(BetterChain, SDLoc(N), Value, Ptr,
                                      ST->getMemoryVT(), ST->getMemOperand());
      } else {
        ReplStore = DAG.getStore(BetterChain, SDLoc(N), Value, Ptr,
                                 ST->getMemOperand());
      }

      // Create token to keep both nodes around.
      SDValue Token = DAG.getNode(ISD::TokenFactor, SDLoc(N),
                                  MVT::Other, Chain, ReplStore);

      // Make sure the new and old chains are cleaned up.
      AddToWorklist(Token.getNode());

      // Don't add users to work list.
      return CombineTo(N, Token, false);
    }
  }

  // Try transforming N to an indexed store.
  if (CombineToPreIndexedLoadStore(N) || CombineToPostIndexedLoadStore(N))
    return SDValue(N, 0);

  // FIXME: is there such a thing as a truncating indexed store?
  if (ST->isTruncatingStore() && ST->isUnindexed() &&
      Value.getValueType().isInteger()) {
    // See if we can simplify the input to this truncstore with knowledge that
    // only the low bits are being used.  For example:
    // "truncstore (or (shl x, 8), y), i8"  -> "truncstore y, i8"
    SDValue Shorter =
      GetDemandedBits(Value,
                      APInt::getLowBitsSet(
                        Value.getValueType().getScalarType().getSizeInBits(),
                        ST->getMemoryVT().getScalarType().getSizeInBits()));
    AddToWorklist(Value.getNode());
    if (Shorter.getNode())
      return DAG.getTruncStore(Chain, SDLoc(N), Shorter,
                               Ptr, ST->getMemoryVT(), ST->getMemOperand());

    // Otherwise, see if we can simplify the operation with
    // SimplifyDemandedBits, which only works if the value has a single use.
    if (SimplifyDemandedBits(Value,
                        APInt::getLowBitsSet(
                          Value.getValueType().getScalarType().getSizeInBits(),
                          ST->getMemoryVT().getScalarType().getSizeInBits())))
      return SDValue(N, 0);
  }

  // If this is a load followed by a store to the same location, then the store
  // is dead/noop.
  if (LoadSDNode *Ld = dyn_cast<LoadSDNode>(Value)) {
    if (Ld->getBasePtr() == Ptr && ST->getMemoryVT() == Ld->getMemoryVT() &&
        ST->isUnindexed() && !ST->isVolatile() &&
        // There can't be any side effects between the load and store, such as
        // a call or store.
        Chain.reachesChainWithoutSideEffects(SDValue(Ld, 1))) {
      // The store is dead, remove it.
      return Chain;
    }
  }

  // If this is a store followed by a store with the same value to the same
  // location, then the store is dead/noop.
  if (StoreSDNode *ST1 = dyn_cast<StoreSDNode>(Chain)) {
    if (ST1->getBasePtr() == Ptr && ST->getMemoryVT() == ST1->getMemoryVT() &&
        ST1->getValue() == Value && ST->isUnindexed() && !ST->isVolatile() &&
        ST1->isUnindexed() && !ST1->isVolatile()) {
      // The store is dead, remove it.
      return Chain;
    }
  }

  // If this is an FP_ROUND or TRUNC followed by a store, fold this into a
  // truncating store.  We can do this even if this is already a truncstore.
  if ((Value.getOpcode() == ISD::FP_ROUND || Value.getOpcode() == ISD::TRUNCATE)
      && Value.getNode()->hasOneUse() && ST->isUnindexed() &&
      TLI.isTruncStoreLegal(Value.getOperand(0).getValueType(),
                            ST->getMemoryVT())) {
    return DAG.getTruncStore(Chain, SDLoc(N), Value.getOperand(0),
                             Ptr, ST->getMemoryVT(), ST->getMemOperand());
  }

  // Only perform this optimization before the types are legal, because we
  // don't want to perform this optimization on every DAGCombine invocation.
  if (!LegalTypes) {
    bool EverChanged = false;

    do {
      // There can be multiple store sequences on the same chain.
      // Keep trying to merge store sequences until we are unable to do so
      // or until we merge the last store on the chain.
      bool Changed = MergeConsecutiveStores(ST);
      EverChanged |= Changed;
      if (!Changed) break;
    } while (ST->getOpcode() != ISD::DELETED_NODE);

    if (EverChanged)
      return SDValue(N, 0);
  }

  return ReduceLoadOpStoreWidth(N);
}

SDValue DAGCombiner::visitINSERT_VECTOR_ELT(SDNode *N) {
  SDValue InVec = N->getOperand(0);
  SDValue InVal = N->getOperand(1);
  SDValue EltNo = N->getOperand(2);
  SDLoc dl(N);

  // If the inserted element is an UNDEF, just use the input vector.
  if (InVal.getOpcode() == ISD::UNDEF)
    return InVec;

  EVT VT = InVec.getValueType();

  // If we can't generate a legal BUILD_VECTOR, exit
  if (LegalOperations && !TLI.isOperationLegal(ISD::BUILD_VECTOR, VT))
    return SDValue();

  // Check that we know which element is being inserted
  if (!isa<ConstantSDNode>(EltNo))
    return SDValue();
  unsigned Elt = cast<ConstantSDNode>(EltNo)->getZExtValue();

  // Canonicalize insert_vector_elt dag nodes.
  // Example:
  // (insert_vector_elt (insert_vector_elt A, Idx0), Idx1)
  // -> (insert_vector_elt (insert_vector_elt A, Idx1), Idx0)
  //
  // Do this only if the child insert_vector node has one use; also
  // do this only if indices are both constants and Idx1 < Idx0.
  if (InVec.getOpcode() == ISD::INSERT_VECTOR_ELT && InVec.hasOneUse()
      && isa<ConstantSDNode>(InVec.getOperand(2))) {
    unsigned OtherElt =
      cast<ConstantSDNode>(InVec.getOperand(2))->getZExtValue();
    if (Elt < OtherElt) {
      // Swap nodes.
      SDValue NewOp = DAG.getNode(ISD::INSERT_VECTOR_ELT, SDLoc(N), VT,
                                  InVec.getOperand(0), InVal, EltNo);
      AddToWorklist(NewOp.getNode());
      return DAG.getNode(ISD::INSERT_VECTOR_ELT, SDLoc(InVec.getNode()),
                         VT, NewOp, InVec.getOperand(1), InVec.getOperand(2));
    }
  }

  // Check that the operand is a BUILD_VECTOR (or UNDEF, which can essentially
  // be converted to a BUILD_VECTOR).  Fill in the Ops vector with the
  // vector elements.
  SmallVector<SDValue, 8> Ops;
  // Do not combine these two vectors if the output vector will not replace
  // the input vector.
  if (InVec.getOpcode() == ISD::BUILD_VECTOR && InVec.hasOneUse()) {
    Ops.append(InVec.getNode()->op_begin(),
               InVec.getNode()->op_end());
  } else if (InVec.getOpcode() == ISD::UNDEF) {
    unsigned NElts = VT.getVectorNumElements();
    Ops.append(NElts, DAG.getUNDEF(InVal.getValueType()));
  } else {
    return SDValue();
  }

  // Insert the element
  if (Elt < Ops.size()) {
    // All the operands of BUILD_VECTOR must have the same type;
    // we enforce that here.
    EVT OpVT = Ops[0].getValueType();
    if (InVal.getValueType() != OpVT)
      InVal = OpVT.bitsGT(InVal.getValueType()) ?
                DAG.getNode(ISD::ANY_EXTEND, dl, OpVT, InVal) :
                DAG.getNode(ISD::TRUNCATE, dl, OpVT, InVal);
    Ops[Elt] = InVal;
  }

  // Return the new vector
  return DAG.getNode(ISD::BUILD_VECTOR, dl, VT, Ops);
}

SDValue DAGCombiner::ReplaceExtractVectorEltOfLoadWithNarrowedLoad(
    SDNode *EVE, EVT InVecVT, SDValue EltNo, LoadSDNode *OriginalLoad) {
  EVT ResultVT = EVE->getValueType(0);
  EVT VecEltVT = InVecVT.getVectorElementType();
  unsigned Align = OriginalLoad->getAlignment();
  unsigned NewAlign = DAG.getDataLayout().getABITypeAlignment(
      VecEltVT.getTypeForEVT(*DAG.getContext()));

  if (NewAlign > Align || !TLI.isOperationLegalOrCustom(ISD::LOAD, VecEltVT))
    return SDValue();

  Align = NewAlign;

  SDValue NewPtr = OriginalLoad->getBasePtr();
  SDValue Offset;
  EVT PtrType = NewPtr.getValueType();
  MachinePointerInfo MPI;
  SDLoc DL(EVE);
  if (auto *ConstEltNo = dyn_cast<ConstantSDNode>(EltNo)) {
    int Elt = ConstEltNo->getZExtValue();
    unsigned PtrOff = VecEltVT.getSizeInBits() * Elt / 8;
    Offset = DAG.getConstant(PtrOff, DL, PtrType);
    MPI = OriginalLoad->getPointerInfo().getWithOffset(PtrOff);
  } else {
    Offset = DAG.getZExtOrTrunc(EltNo, DL, PtrType);
    Offset = DAG.getNode(
        ISD::MUL, DL, PtrType, Offset,
        DAG.getConstant(VecEltVT.getStoreSize(), DL, PtrType));
    MPI = OriginalLoad->getPointerInfo();
  }
  NewPtr = DAG.getNode(ISD::ADD, DL, PtrType, NewPtr, Offset);

  // The replacement we need to do here is a little tricky: we need to
  // replace an extractelement of a load with a load.
  // Use ReplaceAllUsesOfValuesWith to do the replacement.
  // Note that this replacement assumes that the extractvalue is the only
  // use of the load; that's okay because we don't want to perform this
  // transformation in other cases anyway.
  SDValue Load;
  SDValue Chain;
  if (ResultVT.bitsGT(VecEltVT)) {
    // If the result type of vextract is wider than the load, then issue an
    // extending load instead.
    ISD::LoadExtType ExtType = TLI.isLoadExtLegal(ISD::ZEXTLOAD, ResultVT,
                                                  VecEltVT)
                                   ? ISD::ZEXTLOAD
                                   : ISD::EXTLOAD;
    Load = DAG.getExtLoad(
        ExtType, SDLoc(EVE), ResultVT, OriginalLoad->getChain(), NewPtr, MPI,
        VecEltVT, OriginalLoad->isVolatile(), OriginalLoad->isNonTemporal(),
        OriginalLoad->isInvariant(), Align, OriginalLoad->getAAInfo());
    Chain = Load.getValue(1);
  } else {
    Load = DAG.getLoad(
        VecEltVT, SDLoc(EVE), OriginalLoad->getChain(), NewPtr, MPI,
        OriginalLoad->isVolatile(), OriginalLoad->isNonTemporal(),
        OriginalLoad->isInvariant(), Align, OriginalLoad->getAAInfo());
    Chain = Load.getValue(1);
    if (ResultVT.bitsLT(VecEltVT))
      Load = DAG.getNode(ISD::TRUNCATE, SDLoc(EVE), ResultVT, Load);
    else
      Load = DAG.getNode(ISD::BITCAST, SDLoc(EVE), ResultVT, Load);
  }
  WorklistRemover DeadNodes(*this);
  SDValue From[] = { SDValue(EVE, 0), SDValue(OriginalLoad, 1) };
  SDValue To[] = { Load, Chain };
  DAG.ReplaceAllUsesOfValuesWith(From, To, 2);
  // Since we're explicitly calling ReplaceAllUses, add the new node to the
  // worklist explicitly as well.
  AddToWorklist(Load.getNode());
  AddUsersToWorklist(Load.getNode()); // Add users too
  // Make sure to revisit this node to clean it up; it will usually be dead.
  AddToWorklist(EVE);
  ++OpsNarrowed;
  return SDValue(EVE, 0);
}

SDValue DAGCombiner::visitEXTRACT_VECTOR_ELT(SDNode *N) {
  // (vextract (scalar_to_vector val, 0) -> val
  SDValue InVec = N->getOperand(0);
  EVT VT = InVec.getValueType();
  EVT NVT = N->getValueType(0);

  if (InVec.getOpcode() == ISD::SCALAR_TO_VECTOR) {
    // Check if the result type doesn't match the inserted element type. A
    // SCALAR_TO_VECTOR may truncate the inserted element and the
    // EXTRACT_VECTOR_ELT may widen the extracted vector.
    SDValue InOp = InVec.getOperand(0);
    if (InOp.getValueType() != NVT) {
      assert(InOp.getValueType().isInteger() && NVT.isInteger());
      return DAG.getSExtOrTrunc(InOp, SDLoc(InVec), NVT);
    }
    return InOp;
  }

  SDValue EltNo = N->getOperand(1);
  bool ConstEltNo = isa<ConstantSDNode>(EltNo);

  // Transform: (EXTRACT_VECTOR_ELT( VECTOR_SHUFFLE )) -> EXTRACT_VECTOR_ELT.
  // We only perform this optimization before the op legalization phase because
  // we may introduce new vector instructions which are not backed by TD
  // patterns. For example on AVX, extracting elements from a wide vector
  // without using extract_subvector. However, if we can find an underlying
  // scalar value, then we can always use that.
  if (InVec.getOpcode() == ISD::VECTOR_SHUFFLE
      && ConstEltNo) {
    int Elt = cast<ConstantSDNode>(EltNo)->getZExtValue();
    int NumElem = VT.getVectorNumElements();
    ShuffleVectorSDNode *SVOp = cast<ShuffleVectorSDNode>(InVec);
    // Find the new index to extract from.
    int OrigElt = SVOp->getMaskElt(Elt);

    // Extracting an undef index is undef.
    if (OrigElt == -1)
      return DAG.getUNDEF(NVT);

    // Select the right vector half to extract from.
    SDValue SVInVec;
    if (OrigElt < NumElem) {
      SVInVec = InVec->getOperand(0);
    } else {
      SVInVec = InVec->getOperand(1);
      OrigElt -= NumElem;
    }

    if (SVInVec.getOpcode() == ISD::BUILD_VECTOR) {
      SDValue InOp = SVInVec.getOperand(OrigElt);
      if (InOp.getValueType() != NVT) {
        assert(InOp.getValueType().isInteger() && NVT.isInteger());
        InOp = DAG.getSExtOrTrunc(InOp, SDLoc(SVInVec), NVT);
      }

      return InOp;
    }

    // FIXME: We should handle recursing on other vector shuffles and
    // scalar_to_vector here as well.

    if (!LegalOperations) {
      EVT IndexTy = TLI.getVectorIdxTy(DAG.getDataLayout());
      return DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SDLoc(N), NVT, SVInVec,
                         DAG.getConstant(OrigElt, SDLoc(SVOp), IndexTy));
    }
  }

  bool BCNumEltsChanged = false;
  EVT ExtVT = VT.getVectorElementType();
  EVT LVT = ExtVT;

  // If the result of load has to be truncated, then it's not necessarily
  // profitable.
  if (NVT.bitsLT(LVT) && !TLI.isTruncateFree(LVT, NVT))
    return SDValue();

  if (InVec.getOpcode() == ISD::BITCAST) {
    // Don't duplicate a load with other uses.
    if (!InVec.hasOneUse())
      return SDValue();

    EVT BCVT = InVec.getOperand(0).getValueType();
    if (!BCVT.isVector() || ExtVT.bitsGT(BCVT.getVectorElementType()))
      return SDValue();
    if (VT.getVectorNumElements() != BCVT.getVectorNumElements())
      BCNumEltsChanged = true;
    InVec = InVec.getOperand(0);
    ExtVT = BCVT.getVectorElementType();
  }

  // (vextract (vN[if]M load $addr), i) -> ([if]M load $addr + i * size)
  if (!LegalOperations && !ConstEltNo && InVec.hasOneUse() &&
      ISD::isNormalLoad(InVec.getNode()) &&
      !N->getOperand(1)->hasPredecessor(InVec.getNode())) {
    SDValue Index = N->getOperand(1);
    if (LoadSDNode *OrigLoad = dyn_cast<LoadSDNode>(InVec))
      return ReplaceExtractVectorEltOfLoadWithNarrowedLoad(N, VT, Index,
                                                           OrigLoad);
  }

  // Perform only after legalization to ensure build_vector / vector_shuffle
  // optimizations have already been done.
  if (!LegalOperations) return SDValue();

  // (vextract (v4f32 load $addr), c) -> (f32 load $addr+c*size)
  // (vextract (v4f32 s2v (f32 load $addr)), c) -> (f32 load $addr+c*size)
  // (vextract (v4f32 shuffle (load $addr), <1,u,u,u>), 0) -> (f32 load $addr)

  if (ConstEltNo) {
    int Elt = cast<ConstantSDNode>(EltNo)->getZExtValue();

    LoadSDNode *LN0 = nullptr;
    const ShuffleVectorSDNode *SVN = nullptr;
    if (ISD::isNormalLoad(InVec.getNode())) {
      LN0 = cast<LoadSDNode>(InVec);
    } else if (InVec.getOpcode() == ISD::SCALAR_TO_VECTOR &&
               InVec.getOperand(0).getValueType() == ExtVT &&
               ISD::isNormalLoad(InVec.getOperand(0).getNode())) {
      // Don't duplicate a load with other uses.
      if (!InVec.hasOneUse())
        return SDValue();

      LN0 = cast<LoadSDNode>(InVec.getOperand(0));
    } else if ((SVN = dyn_cast<ShuffleVectorSDNode>(InVec))) {
      // (vextract (vector_shuffle (load $addr), v2, <1, u, u, u>), 1)
      // =>
      // (load $addr+1*size)

      // Don't duplicate a load with other uses.
      if (!InVec.hasOneUse())
        return SDValue();

      // If the bit convert changed the number of elements, it is unsafe
      // to examine the mask.
      if (BCNumEltsChanged)
        return SDValue();

      // Select the input vector, guarding against out of range extract vector.
      unsigned NumElems = VT.getVectorNumElements();
      int Idx = (Elt > (int)NumElems) ? -1 : SVN->getMaskElt(Elt);
      InVec = (Idx < (int)NumElems) ? InVec.getOperand(0) : InVec.getOperand(1);

      if (InVec.getOpcode() == ISD::BITCAST) {
        // Don't duplicate a load with other uses.
        if (!InVec.hasOneUse())
          return SDValue();

        InVec = InVec.getOperand(0);
      }
      if (ISD::isNormalLoad(InVec.getNode())) {
        LN0 = cast<LoadSDNode>(InVec);
        Elt = (Idx < (int)NumElems) ? Idx : Idx - (int)NumElems;
        EltNo = DAG.getConstant(Elt, SDLoc(EltNo), EltNo.getValueType());
      }
    }

    // Make sure we found a non-volatile load and the extractelement is
    // the only use.
    if (!LN0 || !LN0->hasNUsesOfValue(1,0) || LN0->isVolatile())
      return SDValue();

    // If Idx was -1 above, Elt is going to be -1, so just return undef.
    if (Elt == -1)
      return DAG.getUNDEF(LVT);

    return ReplaceExtractVectorEltOfLoadWithNarrowedLoad(N, VT, EltNo, LN0);
  }

  return SDValue();
}

// Simplify (build_vec (ext )) to (bitcast (build_vec ))
SDValue DAGCombiner::reduceBuildVecExtToExtBuildVec(SDNode *N) {
  // We perform this optimization post type-legalization because
  // the type-legalizer often scalarizes integer-promoted vectors.
  // Performing this optimization before may create bit-casts which
  // will be type-legalized to complex code sequences.
  // We perform this optimization only before the operation legalizer because we
  // may introduce illegal operations.
  if (Level != AfterLegalizeVectorOps && Level != AfterLegalizeTypes)
    return SDValue();

  unsigned NumInScalars = N->getNumOperands();
  SDLoc dl(N);
  EVT VT = N->getValueType(0);

  // Check to see if this is a BUILD_VECTOR of a bunch of values
  // which come from any_extend or zero_extend nodes. If so, we can create
  // a new BUILD_VECTOR using bit-casts which may enable other BUILD_VECTOR
  // optimizations. We do not handle sign-extend because we can't fill the sign
  // using shuffles.
  EVT SourceType = MVT::Other;
  bool AllAnyExt = true;

  for (unsigned i = 0; i != NumInScalars; ++i) {
    SDValue In = N->getOperand(i);
    // Ignore undef inputs.
    if (In.getOpcode() == ISD::UNDEF) continue;

    bool AnyExt  = In.getOpcode() == ISD::ANY_EXTEND;
    bool ZeroExt = In.getOpcode() == ISD::ZERO_EXTEND;

    // Abort if the element is not an extension.
    if (!ZeroExt && !AnyExt) {
      SourceType = MVT::Other;
      break;
    }

    // The input is a ZeroExt or AnyExt. Check the original type.
    EVT InTy = In.getOperand(0).getValueType();

    // Check that all of the widened source types are the same.
    if (SourceType == MVT::Other)
      // First time.
      SourceType = InTy;
    else if (InTy != SourceType) {
      // Multiple income types. Abort.
      SourceType = MVT::Other;
      break;
    }

    // Check if all of the extends are ANY_EXTENDs.
    AllAnyExt &= AnyExt;
  }

  // In order to have valid types, all of the inputs must be extended from the
  // same source type and all of the inputs must be any or zero extend.
  // Scalar sizes must be a power of two.
  EVT OutScalarTy = VT.getScalarType();
  bool ValidTypes = SourceType != MVT::Other &&
                 isPowerOf2_32(OutScalarTy.getSizeInBits()) &&
                 isPowerOf2_32(SourceType.getSizeInBits());

  // Create a new simpler BUILD_VECTOR sequence which other optimizations can
  // turn into a single shuffle instruction.
  if (!ValidTypes)
    return SDValue();

  bool isLE = DAG.getDataLayout().isLittleEndian();
  unsigned ElemRatio = OutScalarTy.getSizeInBits()/SourceType.getSizeInBits();
  assert(ElemRatio > 1 && "Invalid element size ratio");
  SDValue Filler = AllAnyExt ? DAG.getUNDEF(SourceType):
                               DAG.getConstant(0, SDLoc(N), SourceType);

  unsigned NewBVElems = ElemRatio * VT.getVectorNumElements();
  SmallVector<SDValue, 8> Ops(NewBVElems, Filler);

  // Populate the new build_vector
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i) {
    SDValue Cast = N->getOperand(i);
    assert((Cast.getOpcode() == ISD::ANY_EXTEND ||
            Cast.getOpcode() == ISD::ZERO_EXTEND ||
            Cast.getOpcode() == ISD::UNDEF) && "Invalid cast opcode");
    SDValue In;
    if (Cast.getOpcode() == ISD::UNDEF)
      In = DAG.getUNDEF(SourceType);
    else
      In = Cast->getOperand(0);
    unsigned Index = isLE ? (i * ElemRatio) :
                            (i * ElemRatio + (ElemRatio - 1));

    assert(Index < Ops.size() && "Invalid index");
    Ops[Index] = In;
  }

  // The type of the new BUILD_VECTOR node.
  EVT VecVT = EVT::getVectorVT(*DAG.getContext(), SourceType, NewBVElems);
  assert(VecVT.getSizeInBits() == VT.getSizeInBits() &&
         "Invalid vector size");
  // Check if the new vector type is legal.
  if (!isTypeLegal(VecVT)) return SDValue();

  // Make the new BUILD_VECTOR.
  SDValue BV = DAG.getNode(ISD::BUILD_VECTOR, dl, VecVT, Ops);

  // The new BUILD_VECTOR node has the potential to be further optimized.
  AddToWorklist(BV.getNode());
  // Bitcast to the desired type.
  return DAG.getNode(ISD::BITCAST, dl, VT, BV);
}

SDValue DAGCombiner::reduceBuildVecConvertToConvertBuildVec(SDNode *N) {
  EVT VT = N->getValueType(0);

  unsigned NumInScalars = N->getNumOperands();
  SDLoc dl(N);

  EVT SrcVT = MVT::Other;
  unsigned Opcode = ISD::DELETED_NODE;
  unsigned NumDefs = 0;

  for (unsigned i = 0; i != NumInScalars; ++i) {
    SDValue In = N->getOperand(i);
    unsigned Opc = In.getOpcode();

    if (Opc == ISD::UNDEF)
      continue;

    // If all scalar values are floats and converted from integers.
    if (Opcode == ISD::DELETED_NODE &&
        (Opc == ISD::UINT_TO_FP || Opc == ISD::SINT_TO_FP)) {
      Opcode = Opc;
    }

    if (Opc != Opcode)
      return SDValue();

    EVT InVT = In.getOperand(0).getValueType();

    // If all scalar values are typed differently, bail out. It's chosen to
    // simplify BUILD_VECTOR of integer types.
    if (SrcVT == MVT::Other)
      SrcVT = InVT;
    if (SrcVT != InVT)
      return SDValue();
    NumDefs++;
  }

  // If the vector has just one element defined, it's not worth to fold it into
  // a vectorized one.
  if (NumDefs < 2)
    return SDValue();

  assert((Opcode == ISD::UINT_TO_FP || Opcode == ISD::SINT_TO_FP)
         && "Should only handle conversion from integer to float.");
  assert(SrcVT != MVT::Other && "Cannot determine source type!");

  EVT NVT = EVT::getVectorVT(*DAG.getContext(), SrcVT, NumInScalars);

  if (!TLI.isOperationLegalOrCustom(Opcode, NVT))
    return SDValue();

  // Just because the floating-point vector type is legal does not necessarily
  // mean that the corresponding integer vector type is.
  if (!isTypeLegal(NVT))
    return SDValue();

  SmallVector<SDValue, 8> Opnds;
  for (unsigned i = 0; i != NumInScalars; ++i) {
    SDValue In = N->getOperand(i);

    if (In.getOpcode() == ISD::UNDEF)
      Opnds.push_back(DAG.getUNDEF(SrcVT));
    else
      Opnds.push_back(In.getOperand(0));
  }
  SDValue BV = DAG.getNode(ISD::BUILD_VECTOR, dl, NVT, Opnds);
  AddToWorklist(BV.getNode());

  return DAG.getNode(Opcode, dl, VT, BV);
}

SDValue DAGCombiner::visitBUILD_VECTOR(SDNode *N) {
  unsigned NumInScalars = N->getNumOperands();
  SDLoc dl(N);
  EVT VT = N->getValueType(0);

  // A vector built entirely of undefs is undef.
  if (ISD::allOperandsUndef(N))
    return DAG.getUNDEF(VT);

  if (SDValue V = reduceBuildVecExtToExtBuildVec(N))
    return V;

  if (SDValue V = reduceBuildVecConvertToConvertBuildVec(N))
    return V;

  // Check to see if this is a BUILD_VECTOR of a bunch of EXTRACT_VECTOR_ELT
  // operations.  If so, and if the EXTRACT_VECTOR_ELT vector inputs come from
  // at most two distinct vectors, turn this into a shuffle node.

  // Only type-legal BUILD_VECTOR nodes are converted to shuffle nodes.
  if (!isTypeLegal(VT))
    return SDValue();

  // May only combine to shuffle after legalize if shuffle is legal.
  if (LegalOperations && !TLI.isOperationLegal(ISD::VECTOR_SHUFFLE, VT))
    return SDValue();

  SDValue VecIn1, VecIn2;
  bool UsesZeroVector = false;
  for (unsigned i = 0; i != NumInScalars; ++i) {
    SDValue Op = N->getOperand(i);
    // Ignore undef inputs.
    if (Op.getOpcode() == ISD::UNDEF) continue;

    // See if we can combine this build_vector into a blend with a zero vector.
    if (!VecIn2.getNode() && (isNullConstant(Op) || isNullFPConstant(Op))) {
      UsesZeroVector = true;
      continue;
    }

    // If this input is something other than a EXTRACT_VECTOR_ELT with a
    // constant index, bail out.
    if (Op.getOpcode() != ISD::EXTRACT_VECTOR_ELT ||
        !isa<ConstantSDNode>(Op.getOperand(1))) {
      VecIn1 = VecIn2 = SDValue(nullptr, 0);
      break;
    }

    // We allow up to two distinct input vectors.
    SDValue ExtractedFromVec = Op.getOperand(0);
    if (ExtractedFromVec == VecIn1 || ExtractedFromVec == VecIn2)
      continue;

    if (!VecIn1.getNode()) {
      VecIn1 = ExtractedFromVec;
    } else if (!VecIn2.getNode() && !UsesZeroVector) {
      VecIn2 = ExtractedFromVec;
    } else {
      // Too many inputs.
      VecIn1 = VecIn2 = SDValue(nullptr, 0);
      break;
    }
  }

  // If everything is good, we can make a shuffle operation.
  if (VecIn1.getNode()) {
    unsigned InNumElements = VecIn1.getValueType().getVectorNumElements();
    SmallVector<int, 8> Mask;
    for (unsigned i = 0; i != NumInScalars; ++i) {
      unsigned Opcode = N->getOperand(i).getOpcode();
      if (Opcode == ISD::UNDEF) {
        Mask.push_back(-1);
        continue;
      }

      // Operands can also be zero.
      if (Opcode != ISD::EXTRACT_VECTOR_ELT) {
        assert(UsesZeroVector &&
               (Opcode == ISD::Constant || Opcode == ISD::ConstantFP) &&
               "Unexpected node found!");
        Mask.push_back(NumInScalars+i);
        continue;
      }

      // If extracting from the first vector, just use the index directly.
      SDValue Extract = N->getOperand(i);
      SDValue ExtVal = Extract.getOperand(1);
      unsigned ExtIndex = cast<ConstantSDNode>(ExtVal)->getZExtValue();
      if (Extract.getOperand(0) == VecIn1) {
        Mask.push_back(ExtIndex);
        continue;
      }

      // Otherwise, use InIdx + InputVecSize
      Mask.push_back(InNumElements + ExtIndex);
    }

    // Avoid introducing illegal shuffles with zero.
    if (UsesZeroVector && !TLI.isVectorClearMaskLegal(Mask, VT))
      return SDValue();

    // We can't generate a shuffle node with mismatched input and output types.
    // Attempt to transform a single input vector to the correct type.
    if ((VT != VecIn1.getValueType())) {
      // If the input vector type has a different base type to the output
      // vector type, bail out.
      EVT VTElemType = VT.getVectorElementType();
      if ((VecIn1.getValueType().getVectorElementType() != VTElemType) ||
          (VecIn2.getNode() &&
           (VecIn2.getValueType().getVectorElementType() != VTElemType)))
        return SDValue();

      // If the input vector is too small, widen it.
      // We only support widening of vectors which are half the size of the
      // output registers. For example XMM->YMM widening on X86 with AVX.
      EVT VecInT = VecIn1.getValueType();
      if (VecInT.getSizeInBits() * 2 == VT.getSizeInBits()) {
        // If we only have one small input, widen it by adding undef values.
        if (!VecIn2.getNode())
          VecIn1 = DAG.getNode(ISD::CONCAT_VECTORS, dl, VT, VecIn1,
                               DAG.getUNDEF(VecIn1.getValueType()));
        else if (VecIn1.getValueType() == VecIn2.getValueType()) {
          // If we have two small inputs of the same type, try to concat them.
          VecIn1 = DAG.getNode(ISD::CONCAT_VECTORS, dl, VT, VecIn1, VecIn2);
          VecIn2 = SDValue(nullptr, 0);
        } else
          return SDValue();
      } else if (VecInT.getSizeInBits() == VT.getSizeInBits() * 2) {
        // If the input vector is too large, try to split it.
        // We don't support having two input vectors that are too large.
        // If the zero vector was used, we can not split the vector,
        // since we'd need 3 inputs.
        if (UsesZeroVector || VecIn2.getNode())
          return SDValue();

        if (!TLI.isExtractSubvectorCheap(VT, VT.getVectorNumElements()))
          return SDValue();

        // Try to replace VecIn1 with two extract_subvectors
        // No need to update the masks, they should still be correct.
        VecIn2 = DAG.getNode(
            ISD::EXTRACT_SUBVECTOR, dl, VT, VecIn1,
            DAG.getConstant(VT.getVectorNumElements(), dl,
                            TLI.getVectorIdxTy(DAG.getDataLayout())));
        VecIn1 = DAG.getNode(
            ISD::EXTRACT_SUBVECTOR, dl, VT, VecIn1,
            DAG.getConstant(0, dl, TLI.getVectorIdxTy(DAG.getDataLayout())));
      } else
        return SDValue();
    }

    if (UsesZeroVector)
      VecIn2 = VT.isInteger() ? DAG.getConstant(0, dl, VT) :
                                DAG.getConstantFP(0.0, dl, VT);
    else
      // If VecIn2 is unused then change it to undef.
      VecIn2 = VecIn2.getNode() ? VecIn2 : DAG.getUNDEF(VT);

    // Check that we were able to transform all incoming values to the same
    // type.
    if (VecIn2.getValueType() != VecIn1.getValueType() ||
        VecIn1.getValueType() != VT)
          return SDValue();

    // Return the new VECTOR_SHUFFLE node.
    SDValue Ops[2];
    Ops[0] = VecIn1;
    Ops[1] = VecIn2;
    return DAG.getVectorShuffle(VT, dl, Ops[0], Ops[1], &Mask[0]);
  }

  return SDValue();
}

static SDValue combineConcatVectorOfScalars(SDNode *N, SelectionDAG &DAG) {
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  EVT OpVT = N->getOperand(0).getValueType();

  // If the operands are legal vectors, leave them alone.
  if (TLI.isTypeLegal(OpVT))
    return SDValue();

  SDLoc DL(N);
  EVT VT = N->getValueType(0);
  SmallVector<SDValue, 8> Ops;

  EVT SVT = EVT::getIntegerVT(*DAG.getContext(), OpVT.getSizeInBits());
  SDValue ScalarUndef = DAG.getNode(ISD::UNDEF, DL, SVT);

  // Keep track of what we encounter.
  bool AnyInteger = false;
  bool AnyFP = false;
  for (const SDValue &Op : N->ops()) {
    if (ISD::BITCAST == Op.getOpcode() &&
        !Op.getOperand(0).getValueType().isVector())
      Ops.push_back(Op.getOperand(0));
    else if (ISD::UNDEF == Op.getOpcode())
      Ops.push_back(ScalarUndef);
    else
      return SDValue();

    // Note whether we encounter an integer or floating point scalar.
    // If it's neither, bail out, it could be something weird like x86mmx.
    EVT LastOpVT = Ops.back().getValueType();
    if (LastOpVT.isFloatingPoint())
      AnyFP = true;
    else if (LastOpVT.isInteger())
      AnyInteger = true;
    else
      return SDValue();
  }

  // If any of the operands is a floating point scalar bitcast to a vector,
  // use floating point types throughout, and bitcast everything.
  // Replace UNDEFs by another scalar UNDEF node, of the final desired type.
  if (AnyFP) {
    SVT = EVT::getFloatingPointVT(OpVT.getSizeInBits());
    ScalarUndef = DAG.getNode(ISD::UNDEF, DL, SVT);
    if (AnyInteger) {
      for (SDValue &Op : Ops) {
        if (Op.getValueType() == SVT)
          continue;
        if (Op.getOpcode() == ISD::UNDEF)
          Op = ScalarUndef;
        else
          Op = DAG.getNode(ISD::BITCAST, DL, SVT, Op);
      }
    }
  }

  EVT VecVT = EVT::getVectorVT(*DAG.getContext(), SVT,
                               VT.getSizeInBits() / SVT.getSizeInBits());
  return DAG.getNode(ISD::BITCAST, DL, VT,
                     DAG.getNode(ISD::BUILD_VECTOR, DL, VecVT, Ops));
}

SDValue DAGCombiner::visitCONCAT_VECTORS(SDNode *N) {
  // TODO: Check to see if this is a CONCAT_VECTORS of a bunch of
  // EXTRACT_SUBVECTOR operations.  If so, and if the EXTRACT_SUBVECTOR vector
  // inputs come from at most two distinct vectors, turn this into a shuffle
  // node.

  // If we only have one input vector, we don't need to do any concatenation.
  if (N->getNumOperands() == 1)
    return N->getOperand(0);

  // Check if all of the operands are undefs.
  EVT VT = N->getValueType(0);
  if (ISD::allOperandsUndef(N))
    return DAG.getUNDEF(VT);

  // Optimize concat_vectors where all but the first of the vectors are undef.
  if (std::all_of(std::next(N->op_begin()), N->op_end(), [](const SDValue &Op) {
        return Op.getOpcode() == ISD::UNDEF;
      })) {
    SDValue In = N->getOperand(0);
    assert(In.getValueType().isVector() && "Must concat vectors");

    // Transform: concat_vectors(scalar, undef) -> scalar_to_vector(sclr).
    if (In->getOpcode() == ISD::BITCAST &&
        !In->getOperand(0)->getValueType(0).isVector()) {
      SDValue Scalar = In->getOperand(0);

      // If the bitcast type isn't legal, it might be a trunc of a legal type;
      // look through the trunc so we can still do the transform:
      //   concat_vectors(trunc(scalar), undef) -> scalar_to_vector(scalar)
      if (Scalar->getOpcode() == ISD::TRUNCATE &&
          !TLI.isTypeLegal(Scalar.getValueType()) &&
          TLI.isTypeLegal(Scalar->getOperand(0).getValueType()))
        Scalar = Scalar->getOperand(0);

      EVT SclTy = Scalar->getValueType(0);

      if (!SclTy.isFloatingPoint() && !SclTy.isInteger())
        return SDValue();

      EVT NVT = EVT::getVectorVT(*DAG.getContext(), SclTy,
                                 VT.getSizeInBits() / SclTy.getSizeInBits());
      if (!TLI.isTypeLegal(NVT) || !TLI.isTypeLegal(Scalar.getValueType()))
        return SDValue();

      SDLoc dl = SDLoc(N);
      SDValue Res = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, NVT, Scalar);
      return DAG.getNode(ISD::BITCAST, dl, VT, Res);
    }
  }

  // Fold any combination of BUILD_VECTOR or UNDEF nodes into one BUILD_VECTOR.
  // We have already tested above for an UNDEF only concatenation.
  // fold (concat_vectors (BUILD_VECTOR A, B, ...), (BUILD_VECTOR C, D, ...))
  // -> (BUILD_VECTOR A, B, ..., C, D, ...)
  auto IsBuildVectorOrUndef = [](const SDValue &Op) {
    return ISD::UNDEF == Op.getOpcode() || ISD::BUILD_VECTOR == Op.getOpcode();
  };
  bool AllBuildVectorsOrUndefs =
      std::all_of(N->op_begin(), N->op_end(), IsBuildVectorOrUndef);
  if (AllBuildVectorsOrUndefs) {
    SmallVector<SDValue, 8> Opnds;
    EVT SVT = VT.getScalarType();

    EVT MinVT = SVT;
    if (!SVT.isFloatingPoint()) {
      // If BUILD_VECTOR are from built from integer, they may have different
      // operand types. Get the smallest type and truncate all operands to it.
      bool FoundMinVT = false;
      for (const SDValue &Op : N->ops())
        if (ISD::BUILD_VECTOR == Op.getOpcode()) {
          EVT OpSVT = Op.getOperand(0)->getValueType(0);
          MinVT = (!FoundMinVT || OpSVT.bitsLE(MinVT)) ? OpSVT : MinVT;
          FoundMinVT = true;
        }
      assert(FoundMinVT && "Concat vector type mismatch");
    }

    for (const SDValue &Op : N->ops()) {
      EVT OpVT = Op.getValueType();
      unsigned NumElts = OpVT.getVectorNumElements();

      if (ISD::UNDEF == Op.getOpcode())
        Opnds.append(NumElts, DAG.getUNDEF(MinVT));

      if (ISD::BUILD_VECTOR == Op.getOpcode()) {
        if (SVT.isFloatingPoint()) {
          assert(SVT == OpVT.getScalarType() && "Concat vector type mismatch");
          Opnds.append(Op->op_begin(), Op->op_begin() + NumElts);
        } else {
          for (unsigned i = 0; i != NumElts; ++i)
            Opnds.push_back(
                DAG.getNode(ISD::TRUNCATE, SDLoc(N), MinVT, Op.getOperand(i)));
        }
      }
    }

    assert(VT.getVectorNumElements() == Opnds.size() &&
           "Concat vector type mismatch");
    return DAG.getNode(ISD::BUILD_VECTOR, SDLoc(N), VT, Opnds);
  }

  // Fold CONCAT_VECTORS of only bitcast scalars (or undef) to BUILD_VECTOR.
  if (SDValue V = combineConcatVectorOfScalars(N, DAG))
    return V;

  // Type legalization of vectors and DAG canonicalization of SHUFFLE_VECTOR
  // nodes often generate nop CONCAT_VECTOR nodes.
  // Scan the CONCAT_VECTOR operands and look for a CONCAT operations that
  // place the incoming vectors at the exact same location.
  SDValue SingleSource = SDValue();
  unsigned PartNumElem = N->getOperand(0).getValueType().getVectorNumElements();

  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i) {
    SDValue Op = N->getOperand(i);

    if (Op.getOpcode() == ISD::UNDEF)
      continue;

    // Check if this is the identity extract:
    if (Op.getOpcode() != ISD::EXTRACT_SUBVECTOR)
      return SDValue();

    // Find the single incoming vector for the extract_subvector.
    if (SingleSource.getNode()) {
      if (Op.getOperand(0) != SingleSource)
        return SDValue();
    } else {
      SingleSource = Op.getOperand(0);

      // Check the source type is the same as the type of the result.
      // If not, this concat may extend the vector, so we can not
      // optimize it away.
      if (SingleSource.getValueType() != N->getValueType(0))
        return SDValue();
    }

    unsigned IdentityIndex = i * PartNumElem;
    ConstantSDNode *CS = dyn_cast<ConstantSDNode>(Op.getOperand(1));
    // The extract index must be constant.
    if (!CS)
      return SDValue();

    // Check that we are reading from the identity index.
    if (CS->getZExtValue() != IdentityIndex)
      return SDValue();
  }

  if (SingleSource.getNode())
    return SingleSource;

  return SDValue();
}

SDValue DAGCombiner::visitEXTRACT_SUBVECTOR(SDNode* N) {
  EVT NVT = N->getValueType(0);
  SDValue V = N->getOperand(0);

  if (V->getOpcode() == ISD::CONCAT_VECTORS) {
    // Combine:
    //    (extract_subvec (concat V1, V2, ...), i)
    // Into:
    //    Vi if possible
    // Only operand 0 is checked as 'concat' assumes all inputs of the same
    // type.
    if (V->getOperand(0).getValueType() != NVT)
      return SDValue();
    unsigned Idx = N->getConstantOperandVal(1);
    unsigned NumElems = NVT.getVectorNumElements();
    assert((Idx % NumElems) == 0 &&
           "IDX in concat is not a multiple of the result vector length.");
    return V->getOperand(Idx / NumElems);
  }

  // Skip bitcasting
  if (V->getOpcode() == ISD::BITCAST)
    V = V.getOperand(0);

  if (V->getOpcode() == ISD::INSERT_SUBVECTOR) {
    SDLoc dl(N);
    // Handle only simple case where vector being inserted and vector
    // being extracted are of same type, and are half size of larger vectors.
    EVT BigVT = V->getOperand(0).getValueType();
    EVT SmallVT = V->getOperand(1).getValueType();
    if (!NVT.bitsEq(SmallVT) || NVT.getSizeInBits()*2 != BigVT.getSizeInBits())
      return SDValue();

    // Only handle cases where both indexes are constants with the same type.
    ConstantSDNode *ExtIdx = dyn_cast<ConstantSDNode>(N->getOperand(1));
    ConstantSDNode *InsIdx = dyn_cast<ConstantSDNode>(V->getOperand(2));

    if (InsIdx && ExtIdx &&
        InsIdx->getValueType(0).getSizeInBits() <= 64 &&
        ExtIdx->getValueType(0).getSizeInBits() <= 64) {
      // Combine:
      //    (extract_subvec (insert_subvec V1, V2, InsIdx), ExtIdx)
      // Into:
      //    indices are equal or bit offsets are equal => V1
      //    otherwise => (extract_subvec V1, ExtIdx)
      if (InsIdx->getZExtValue() * SmallVT.getScalarType().getSizeInBits() ==
          ExtIdx->getZExtValue() * NVT.getScalarType().getSizeInBits())
        return DAG.getNode(ISD::BITCAST, dl, NVT, V->getOperand(1));
      return DAG.getNode(ISD::EXTRACT_SUBVECTOR, dl, NVT,
                         DAG.getNode(ISD::BITCAST, dl,
                                     N->getOperand(0).getValueType(),
                                     V->getOperand(0)), N->getOperand(1));
    }
  }

  return SDValue();
}

static SDValue simplifyShuffleOperandRecursively(SmallBitVector &UsedElements,
                                                 SDValue V, SelectionDAG &DAG) {
  SDLoc DL(V);
  EVT VT = V.getValueType();

  switch (V.getOpcode()) {
  default:
    return V;

  case ISD::CONCAT_VECTORS: {
    EVT OpVT = V->getOperand(0).getValueType();
    int OpSize = OpVT.getVectorNumElements();
    SmallBitVector OpUsedElements(OpSize, false);
    bool FoundSimplification = false;
    SmallVector<SDValue, 4> NewOps;
    NewOps.reserve(V->getNumOperands());
    for (int i = 0, NumOps = V->getNumOperands(); i < NumOps; ++i) {
      SDValue Op = V->getOperand(i);
      bool OpUsed = false;
      for (int j = 0; j < OpSize; ++j)
        if (UsedElements[i * OpSize + j]) {
          OpUsedElements[j] = true;
          OpUsed = true;
        }
      NewOps.push_back(
          OpUsed ? simplifyShuffleOperandRecursively(OpUsedElements, Op, DAG)
                 : DAG.getUNDEF(OpVT));
      FoundSimplification |= Op == NewOps.back();
      OpUsedElements.reset();
    }
    if (FoundSimplification)
      V = DAG.getNode(ISD::CONCAT_VECTORS, DL, VT, NewOps);
    return V;
  }

  case ISD::INSERT_SUBVECTOR: {
    SDValue BaseV = V->getOperand(0);
    SDValue SubV = V->getOperand(1);
    auto *IdxN = dyn_cast<ConstantSDNode>(V->getOperand(2));
    if (!IdxN)
      return V;

    int SubSize = SubV.getValueType().getVectorNumElements();
    int Idx = IdxN->getZExtValue();
    bool SubVectorUsed = false;
    SmallBitVector SubUsedElements(SubSize, false);
    for (int i = 0; i < SubSize; ++i)
      if (UsedElements[i + Idx]) {
        SubVectorUsed = true;
        SubUsedElements[i] = true;
        UsedElements[i + Idx] = false;
      }

    // Now recurse on both the base and sub vectors.
    SDValue SimplifiedSubV =
        SubVectorUsed
            ? simplifyShuffleOperandRecursively(SubUsedElements, SubV, DAG)
            : DAG.getUNDEF(SubV.getValueType());
    SDValue SimplifiedBaseV = simplifyShuffleOperandRecursively(UsedElements, BaseV, DAG);
    if (SimplifiedSubV != SubV || SimplifiedBaseV != BaseV)
      V = DAG.getNode(ISD::INSERT_SUBVECTOR, DL, VT,
                      SimplifiedBaseV, SimplifiedSubV, V->getOperand(2));
    return V;
  }
  }
}

static SDValue simplifyShuffleOperands(ShuffleVectorSDNode *SVN, SDValue N0,
                                       SDValue N1, SelectionDAG &DAG) {
  EVT VT = SVN->getValueType(0);
  int NumElts = VT.getVectorNumElements();
  SmallBitVector N0UsedElements(NumElts, false), N1UsedElements(NumElts, false);
  for (int M : SVN->getMask())
    if (M >= 0 && M < NumElts)
      N0UsedElements[M] = true;
    else if (M >= NumElts)
      N1UsedElements[M - NumElts] = true;

  SDValue S0 = simplifyShuffleOperandRecursively(N0UsedElements, N0, DAG);
  SDValue S1 = simplifyShuffleOperandRecursively(N1UsedElements, N1, DAG);
  if (S0 == N0 && S1 == N1)
    return SDValue();

  return DAG.getVectorShuffle(VT, SDLoc(SVN), S0, S1, SVN->getMask());
}

// Tries to turn a shuffle of two CONCAT_VECTORS into a single concat,
// or turn a shuffle of a single concat into simpler shuffle then concat.
static SDValue partitionShuffleOfConcats(SDNode *N, SelectionDAG &DAG) {
  EVT VT = N->getValueType(0);
  unsigned NumElts = VT.getVectorNumElements();

  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ShuffleVectorSDNode *SVN = cast<ShuffleVectorSDNode>(N);

  SmallVector<SDValue, 4> Ops;
  EVT ConcatVT = N0.getOperand(0).getValueType();
  unsigned NumElemsPerConcat = ConcatVT.getVectorNumElements();
  unsigned NumConcats = NumElts / NumElemsPerConcat;

  // Special case: shuffle(concat(A,B)) can be more efficiently represented
  // as concat(shuffle(A,B),UNDEF) if the shuffle doesn't set any of the high
  // half vector elements.
  if (NumElemsPerConcat * 2 == NumElts && N1.getOpcode() == ISD::UNDEF &&
      std::all_of(SVN->getMask().begin() + NumElemsPerConcat,
                  SVN->getMask().end(), [](int i) { return i == -1; })) {
    N0 = DAG.getVectorShuffle(ConcatVT, SDLoc(N), N0.getOperand(0), N0.getOperand(1),
                              ArrayRef<int>(SVN->getMask().begin(), NumElemsPerConcat));
    N1 = DAG.getUNDEF(ConcatVT);
    return DAG.getNode(ISD::CONCAT_VECTORS, SDLoc(N), VT, N0, N1);
  }

  // Look at every vector that's inserted. We're looking for exact
  // subvector-sized copies from a concatenated vector
  for (unsigned I = 0; I != NumConcats; ++I) {
    // Make sure we're dealing with a copy.
    unsigned Begin = I * NumElemsPerConcat;
    bool AllUndef = true, NoUndef = true;
    for (unsigned J = Begin; J != Begin + NumElemsPerConcat; ++J) {
      if (SVN->getMaskElt(J) >= 0)
        AllUndef = false;
      else
        NoUndef = false;
    }

    if (NoUndef) {
      if (SVN->getMaskElt(Begin) % NumElemsPerConcat != 0)
        return SDValue();

      for (unsigned J = 1; J != NumElemsPerConcat; ++J)
        if (SVN->getMaskElt(Begin + J - 1) + 1 != SVN->getMaskElt(Begin + J))
          return SDValue();

      unsigned FirstElt = SVN->getMaskElt(Begin) / NumElemsPerConcat;
      if (FirstElt < N0.getNumOperands())
        Ops.push_back(N0.getOperand(FirstElt));
      else
        Ops.push_back(N1.getOperand(FirstElt - N0.getNumOperands()));

    } else if (AllUndef) {
      Ops.push_back(DAG.getUNDEF(N0.getOperand(0).getValueType()));
    } else { // Mixed with general masks and undefs, can't do optimization.
      return SDValue();
    }
  }

  return DAG.getNode(ISD::CONCAT_VECTORS, SDLoc(N), VT, Ops);
}

SDValue DAGCombiner::visitVECTOR_SHUFFLE(SDNode *N) {
  EVT VT = N->getValueType(0);
  unsigned NumElts = VT.getVectorNumElements();

  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);

  assert(N0.getValueType() == VT && "Vector shuffle must be normalized in DAG");

  // Canonicalize shuffle undef, undef -> undef
  if (N0.getOpcode() == ISD::UNDEF && N1.getOpcode() == ISD::UNDEF)
    return DAG.getUNDEF(VT);

  ShuffleVectorSDNode *SVN = cast<ShuffleVectorSDNode>(N);

  // Canonicalize shuffle v, v -> v, undef
  if (N0 == N1) {
    SmallVector<int, 8> NewMask;
    for (unsigned i = 0; i != NumElts; ++i) {
      int Idx = SVN->getMaskElt(i);
      if (Idx >= (int)NumElts) Idx -= NumElts;
      NewMask.push_back(Idx);
    }
    return DAG.getVectorShuffle(VT, SDLoc(N), N0, DAG.getUNDEF(VT),
                                &NewMask[0]);
  }

  // Canonicalize shuffle undef, v -> v, undef.  Commute the shuffle mask.
  if (N0.getOpcode() == ISD::UNDEF) {
    SmallVector<int, 8> NewMask;
    for (unsigned i = 0; i != NumElts; ++i) {
      int Idx = SVN->getMaskElt(i);
      if (Idx >= 0) {
        if (Idx >= (int)NumElts)
          Idx -= NumElts;
        else
          Idx = -1; // remove reference to lhs
      }
      NewMask.push_back(Idx);
    }
    return DAG.getVectorShuffle(VT, SDLoc(N), N1, DAG.getUNDEF(VT),
                                &NewMask[0]);
  }

  // Remove references to rhs if it is undef
  if (N1.getOpcode() == ISD::UNDEF) {
    bool Changed = false;
    SmallVector<int, 8> NewMask;
    for (unsigned i = 0; i != NumElts; ++i) {
      int Idx = SVN->getMaskElt(i);
      if (Idx >= (int)NumElts) {
        Idx = -1;
        Changed = true;
      }
      NewMask.push_back(Idx);
    }
    if (Changed)
      return DAG.getVectorShuffle(VT, SDLoc(N), N0, N1, &NewMask[0]);
  }

  // If it is a splat, check if the argument vector is another splat or a
  // build_vector.
  if (SVN->isSplat() && SVN->getSplatIndex() < (int)NumElts) {
    SDNode *V = N0.getNode();

    // If this is a bit convert that changes the element type of the vector but
    // not the number of vector elements, look through it.  Be careful not to
    // look though conversions that change things like v4f32 to v2f64.
    if (V->getOpcode() == ISD::BITCAST) {
      SDValue ConvInput = V->getOperand(0);
      if (ConvInput.getValueType().isVector() &&
          ConvInput.getValueType().getVectorNumElements() == NumElts)
        V = ConvInput.getNode();
    }

    if (V->getOpcode() == ISD::BUILD_VECTOR) {
      assert(V->getNumOperands() == NumElts &&
             "BUILD_VECTOR has wrong number of operands");
      SDValue Base;
      bool AllSame = true;
      for (unsigned i = 0; i != NumElts; ++i) {
        if (V->getOperand(i).getOpcode() != ISD::UNDEF) {
          Base = V->getOperand(i);
          break;
        }
      }
      // Splat of <u, u, u, u>, return <u, u, u, u>
      if (!Base.getNode())
        return N0;
      for (unsigned i = 0; i != NumElts; ++i) {
        if (V->getOperand(i) != Base) {
          AllSame = false;
          break;
        }
      }
      // Splat of <x, x, x, x>, return <x, x, x, x>
      if (AllSame)
        return N0;

      // Canonicalize any other splat as a build_vector.
      const SDValue &Splatted = V->getOperand(SVN->getSplatIndex());
      SmallVector<SDValue, 8> Ops(NumElts, Splatted);
      SDValue NewBV = DAG.getNode(ISD::BUILD_VECTOR, SDLoc(N),
                                  V->getValueType(0), Ops);

      // We may have jumped through bitcasts, so the type of the
      // BUILD_VECTOR may not match the type of the shuffle.
      if (V->getValueType(0) != VT)
        NewBV = DAG.getNode(ISD::BITCAST, SDLoc(N), VT, NewBV);
      return NewBV;
    }
  }

  // There are various patterns used to build up a vector from smaller vectors,
  // subvectors, or elements. Scan chains of these and replace unused insertions
  // or components with undef.
  if (SDValue S = simplifyShuffleOperands(SVN, N0, N1, DAG))
    return S;

  if (N0.getOpcode() == ISD::CONCAT_VECTORS &&
      Level < AfterLegalizeVectorOps &&
      (N1.getOpcode() == ISD::UNDEF ||
      (N1.getOpcode() == ISD::CONCAT_VECTORS &&
       N0.getOperand(0).getValueType() == N1.getOperand(0).getValueType()))) {
    SDValue V = partitionShuffleOfConcats(N, DAG);

    if (V.getNode())
      return V;
  }

  // Attempt to combine a shuffle of 2 inputs of 'scalar sources' -
  // BUILD_VECTOR or SCALAR_TO_VECTOR into a single BUILD_VECTOR.
  if (Level < AfterLegalizeVectorOps && TLI.isTypeLegal(VT)) {
    SmallVector<SDValue, 8> Ops;
    for (int M : SVN->getMask()) {
      SDValue Op = DAG.getUNDEF(VT.getScalarType());
      if (M >= 0) {
        int Idx = M % NumElts;
        SDValue &S = (M < (int)NumElts ? N0 : N1);
        if (S.getOpcode() == ISD::BUILD_VECTOR && S.hasOneUse()) {
          Op = S.getOperand(Idx);
        } else if (S.getOpcode() == ISD::SCALAR_TO_VECTOR && S.hasOneUse()) {
          if (Idx == 0)
            Op = S.getOperand(0);
        } else {
          // Operand can't be combined - bail out.
          break;
        }
      }
      Ops.push_back(Op);
    }
    if (Ops.size() == VT.getVectorNumElements()) {
      // BUILD_VECTOR requires all inputs to be of the same type, find the
      // maximum type and extend them all.
      EVT SVT = VT.getScalarType();
      if (SVT.isInteger())
        for (SDValue &Op : Ops)
          SVT = (SVT.bitsLT(Op.getValueType()) ? Op.getValueType() : SVT);
      if (SVT != VT.getScalarType())
        for (SDValue &Op : Ops)
          Op = TLI.isZExtFree(Op.getValueType(), SVT)
                   ? DAG.getZExtOrTrunc(Op, SDLoc(N), SVT)
                   : DAG.getSExtOrTrunc(Op, SDLoc(N), SVT);
      return DAG.getNode(ISD::BUILD_VECTOR, SDLoc(N), VT, Ops);
    }
  }

  // If this shuffle only has a single input that is a bitcasted shuffle,
  // attempt to merge the 2 shuffles and suitably bitcast the inputs/output
  // back to their original types.
  if (N0.getOpcode() == ISD::BITCAST && N0.hasOneUse() &&
      N1.getOpcode() == ISD::UNDEF && Level < AfterLegalizeVectorOps &&
      TLI.isTypeLegal(VT)) {

    // Peek through the bitcast only if there is one user.
    SDValue BC0 = N0;
    while (BC0.getOpcode() == ISD::BITCAST) {
      if (!BC0.hasOneUse())
        break;
      BC0 = BC0.getOperand(0);
    }

    auto ScaleShuffleMask = [](ArrayRef<int> Mask, int Scale) {
      if (Scale == 1)
        return SmallVector<int, 8>(Mask.begin(), Mask.end());

      SmallVector<int, 8> NewMask;
      for (int M : Mask)
        for (int s = 0; s != Scale; ++s)
          NewMask.push_back(M < 0 ? -1 : Scale * M + s);
      return NewMask;
    };

    if (BC0.getOpcode() == ISD::VECTOR_SHUFFLE && BC0.hasOneUse()) {
      EVT SVT = VT.getScalarType();
      EVT InnerVT = BC0->getValueType(0);
      EVT InnerSVT = InnerVT.getScalarType();

      // Determine which shuffle works with the smaller scalar type.
      EVT ScaleVT = SVT.bitsLT(InnerSVT) ? VT : InnerVT;
      EVT ScaleSVT = ScaleVT.getScalarType();

      if (TLI.isTypeLegal(ScaleVT) &&
          0 == (InnerSVT.getSizeInBits() % ScaleSVT.getSizeInBits()) &&
          0 == (SVT.getSizeInBits() % ScaleSVT.getSizeInBits())) {

        int InnerScale = InnerSVT.getSizeInBits() / ScaleSVT.getSizeInBits();
        int OuterScale = SVT.getSizeInBits() / ScaleSVT.getSizeInBits();

        // Scale the shuffle masks to the smaller scalar type.
        ShuffleVectorSDNode *InnerSVN = cast<ShuffleVectorSDNode>(BC0);
        SmallVector<int, 8> InnerMask =
            ScaleShuffleMask(InnerSVN->getMask(), InnerScale);
        SmallVector<int, 8> OuterMask =
            ScaleShuffleMask(SVN->getMask(), OuterScale);

        // Merge the shuffle masks.
        SmallVector<int, 8> NewMask;
        for (int M : OuterMask)
          NewMask.push_back(M < 0 ? -1 : InnerMask[M]);

        // Test for shuffle mask legality over both commutations.
        SDValue SV0 = BC0->getOperand(0);
        SDValue SV1 = BC0->getOperand(1);
        bool LegalMask = TLI.isShuffleMaskLegal(NewMask, ScaleVT);
        if (!LegalMask) {
          std::swap(SV0, SV1);
          ShuffleVectorSDNode::commuteMask(NewMask);
          LegalMask = TLI.isShuffleMaskLegal(NewMask, ScaleVT);
        }

        if (LegalMask) {
          SV0 = DAG.getNode(ISD::BITCAST, SDLoc(N), ScaleVT, SV0);
          SV1 = DAG.getNode(ISD::BITCAST, SDLoc(N), ScaleVT, SV1);
          return DAG.getNode(
              ISD::BITCAST, SDLoc(N), VT,
              DAG.getVectorShuffle(ScaleVT, SDLoc(N), SV0, SV1, NewMask));
        }
      }
    }
  }

  // Canonicalize shuffles according to rules:
  //  shuffle(A, shuffle(A, B)) -> shuffle(shuffle(A,B), A)
  //  shuffle(B, shuffle(A, B)) -> shuffle(shuffle(A,B), B)
  //  shuffle(B, shuffle(A, Undef)) -> shuffle(shuffle(A, Undef), B)
  if (N1.getOpcode() == ISD::VECTOR_SHUFFLE &&
      N0.getOpcode() != ISD::VECTOR_SHUFFLE && Level < AfterLegalizeDAG &&
      TLI.isTypeLegal(VT)) {
    // The incoming shuffle must be of the same type as the result of the
    // current shuffle.
    assert(N1->getOperand(0).getValueType() == VT &&
           "Shuffle types don't match");

    SDValue SV0 = N1->getOperand(0);
    SDValue SV1 = N1->getOperand(1);
    bool HasSameOp0 = N0 == SV0;
    bool IsSV1Undef = SV1.getOpcode() == ISD::UNDEF;
    if (HasSameOp0 || IsSV1Undef || N0 == SV1)
      // Commute the operands of this shuffle so that next rule
      // will trigger.
      return DAG.getCommutedVectorShuffle(*SVN);
  }

  // Try to fold according to rules:
  //   shuffle(shuffle(A, B, M0), C, M1) -> shuffle(A, B, M2)
  //   shuffle(shuffle(A, B, M0), C, M1) -> shuffle(A, C, M2)
  //   shuffle(shuffle(A, B, M0), C, M1) -> shuffle(B, C, M2)
  // Don't try to fold shuffles with illegal type.
  // Only fold if this shuffle is the only user of the other shuffle.
  if (N0.getOpcode() == ISD::VECTOR_SHUFFLE && N->isOnlyUserOf(N0.getNode()) &&
      Level < AfterLegalizeDAG && TLI.isTypeLegal(VT)) {
    ShuffleVectorSDNode *OtherSV = cast<ShuffleVectorSDNode>(N0);

    // The incoming shuffle must be of the same type as the result of the
    // current shuffle.
    assert(OtherSV->getOperand(0).getValueType() == VT &&
           "Shuffle types don't match");

    SDValue SV0, SV1;
    SmallVector<int, 4> Mask;
    // Compute the combined shuffle mask for a shuffle with SV0 as the first
    // operand, and SV1 as the second operand.
    for (unsigned i = 0; i != NumElts; ++i) {
      int Idx = SVN->getMaskElt(i);
      if (Idx < 0) {
        // Propagate Undef.
        Mask.push_back(Idx);
        continue;
      }

      SDValue CurrentVec;
      if (Idx < (int)NumElts) {
        // This shuffle index refers to the inner shuffle N0. Lookup the inner
        // shuffle mask to identify which vector is actually referenced.
        Idx = OtherSV->getMaskElt(Idx);
        if (Idx < 0) {
          // Propagate Undef.
          Mask.push_back(Idx);
          continue;
        }

        CurrentVec = (Idx < (int) NumElts) ? OtherSV->getOperand(0)
                                           : OtherSV->getOperand(1);
      } else {
        // This shuffle index references an element within N1.
        CurrentVec = N1;
      }

      // Simple case where 'CurrentVec' is UNDEF.
      if (CurrentVec.getOpcode() == ISD::UNDEF) {
        Mask.push_back(-1);
        continue;
      }

      // Canonicalize the shuffle index. We don't know yet if CurrentVec
      // will be the first or second operand of the combined shuffle.
      Idx = Idx % NumElts;
      if (!SV0.getNode() || SV0 == CurrentVec) {
        // Ok. CurrentVec is the left hand side.
        // Update the mask accordingly.
        SV0 = CurrentVec;
        Mask.push_back(Idx);
        continue;
      }

      // Bail out if we cannot convert the shuffle pair into a single shuffle.
      if (SV1.getNode() && SV1 != CurrentVec)
        return SDValue();

      // Ok. CurrentVec is the right hand side.
      // Update the mask accordingly.
      SV1 = CurrentVec;
      Mask.push_back(Idx + NumElts);
    }

    // Check if all indices in Mask are Undef. In case, propagate Undef.
    bool isUndefMask = true;
    for (unsigned i = 0; i != NumElts && isUndefMask; ++i)
      isUndefMask &= Mask[i] < 0;

    if (isUndefMask)
      return DAG.getUNDEF(VT);

    if (!SV0.getNode())
      SV0 = DAG.getUNDEF(VT);
    if (!SV1.getNode())
      SV1 = DAG.getUNDEF(VT);

    // Avoid introducing shuffles with illegal mask.
    if (!TLI.isShuffleMaskLegal(Mask, VT)) {
      ShuffleVectorSDNode::commuteMask(Mask);

      if (!TLI.isShuffleMaskLegal(Mask, VT))
        return SDValue();

      //   shuffle(shuffle(A, B, M0), C, M1) -> shuffle(B, A, M2)
      //   shuffle(shuffle(A, B, M0), C, M1) -> shuffle(C, A, M2)
      //   shuffle(shuffle(A, B, M0), C, M1) -> shuffle(C, B, M2)
      std::swap(SV0, SV1);
    }

    //   shuffle(shuffle(A, B, M0), C, M1) -> shuffle(A, B, M2)
    //   shuffle(shuffle(A, B, M0), C, M1) -> shuffle(A, C, M2)
    //   shuffle(shuffle(A, B, M0), C, M1) -> shuffle(B, C, M2)
    return DAG.getVectorShuffle(VT, SDLoc(N), SV0, SV1, &Mask[0]);
  }

  return SDValue();
}

SDValue DAGCombiner::visitSCALAR_TO_VECTOR(SDNode *N) {
  SDValue InVal = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // Replace a SCALAR_TO_VECTOR(EXTRACT_VECTOR_ELT(V,C0)) pattern
  // with a VECTOR_SHUFFLE.
  if (InVal.getOpcode() == ISD::EXTRACT_VECTOR_ELT) {
    SDValue InVec = InVal->getOperand(0);
    SDValue EltNo = InVal->getOperand(1);

    // FIXME: We could support implicit truncation if the shuffle can be
    // scaled to a smaller vector scalar type.
    ConstantSDNode *C0 = dyn_cast<ConstantSDNode>(EltNo);
    if (C0 && VT == InVec.getValueType() &&
        VT.getScalarType() == InVal.getValueType()) {
      SmallVector<int, 8> NewMask(VT.getVectorNumElements(), -1);
      int Elt = C0->getZExtValue();
      NewMask[0] = Elt;

      if (TLI.isShuffleMaskLegal(NewMask, VT))
        return DAG.getVectorShuffle(VT, SDLoc(N), InVec, DAG.getUNDEF(VT),
                                    NewMask);
    }
  }

  return SDValue();
}

SDValue DAGCombiner::visitINSERT_SUBVECTOR(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N2 = N->getOperand(2);

  // If the input vector is a concatenation, and the insert replaces
  // one of the halves, we can optimize into a single concat_vectors.
  if (N0.getOpcode() == ISD::CONCAT_VECTORS &&
      N0->getNumOperands() == 2 && N2.getOpcode() == ISD::Constant) {
    APInt InsIdx = cast<ConstantSDNode>(N2)->getAPIntValue();
    EVT VT = N->getValueType(0);

    // Lower half: fold (insert_subvector (concat_vectors X, Y), Z) ->
    // (concat_vectors Z, Y)
    if (InsIdx == 0)
      return DAG.getNode(ISD::CONCAT_VECTORS, SDLoc(N), VT,
                         N->getOperand(1), N0.getOperand(1));

    // Upper half: fold (insert_subvector (concat_vectors X, Y), Z) ->
    // (concat_vectors X, Z)
    if (InsIdx == VT.getVectorNumElements()/2)
      return DAG.getNode(ISD::CONCAT_VECTORS, SDLoc(N), VT,
                         N0.getOperand(0), N->getOperand(1));
  }

  return SDValue();
}

SDValue DAGCombiner::visitFP_TO_FP16(SDNode *N) {
  SDValue N0 = N->getOperand(0);

  // fold (fp_to_fp16 (fp16_to_fp op)) -> op
  if (N0->getOpcode() == ISD::FP16_TO_FP)
    return N0->getOperand(0);

  return SDValue();
}

/// Returns a vector_shuffle if it able to transform an AND to a vector_shuffle
/// with the destination vector and a zero vector.
/// e.g. AND V, <0xffffffff, 0, 0xffffffff, 0>. ==>
///      vector_shuffle V, Zero, <0, 4, 2, 4>
SDValue DAGCombiner::XformToShuffleWithZero(SDNode *N) {
  EVT VT = N->getValueType(0);
  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);
  SDLoc dl(N);

  // Make sure we're not running after operation legalization where it
  // may have custom lowered the vector shuffles.
  if (LegalOperations)
    return SDValue();

  if (N->getOpcode() != ISD::AND)
    return SDValue();

  if (RHS.getOpcode() == ISD::BITCAST)
    RHS = RHS.getOperand(0);

  if (RHS.getOpcode() == ISD::BUILD_VECTOR) {
    SmallVector<int, 8> Indices;
    unsigned NumElts = RHS.getNumOperands();

    for (unsigned i = 0; i != NumElts; ++i) {
      SDValue Elt = RHS.getOperand(i);
      if (isAllOnesConstant(Elt))
        Indices.push_back(i);
      else if (isNullConstant(Elt))
        Indices.push_back(NumElts+i);
      else
        return SDValue();
    }

    // Let's see if the target supports this vector_shuffle.
    EVT RVT = RHS.getValueType();
    if (!TLI.isVectorClearMaskLegal(Indices, RVT))
      return SDValue();

    // Return the new VECTOR_SHUFFLE node.
    EVT EltVT = RVT.getVectorElementType();
    SmallVector<SDValue,8> ZeroOps(RVT.getVectorNumElements(),
                                   DAG.getConstant(0, dl, EltVT));
    SDValue Zero = DAG.getNode(ISD::BUILD_VECTOR, dl, RVT, ZeroOps);
    LHS = DAG.getNode(ISD::BITCAST, dl, RVT, LHS);
    SDValue Shuf = DAG.getVectorShuffle(RVT, dl, LHS, Zero, &Indices[0]);
    return DAG.getNode(ISD::BITCAST, dl, VT, Shuf);
  }

  return SDValue();
}

/// Visit a binary vector operation, like ADD.
SDValue DAGCombiner::SimplifyVBinOp(SDNode *N) {
  assert(N->getValueType(0).isVector() &&
         "SimplifyVBinOp only works on vectors!");

  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);

  if (SDValue Shuffle = XformToShuffleWithZero(N))
    return Shuffle;

  // If the LHS and RHS are BUILD_VECTOR nodes, see if we can constant fold
  // this operation.
  if (LHS.getOpcode() == ISD::BUILD_VECTOR &&
      RHS.getOpcode() == ISD::BUILD_VECTOR) {
    // Check if both vectors are constants. If not bail out.
    if (!(cast<BuildVectorSDNode>(LHS)->isConstant() &&
          cast<BuildVectorSDNode>(RHS)->isConstant()))
      return SDValue();

    SmallVector<SDValue, 8> Ops;
    for (unsigned i = 0, e = LHS.getNumOperands(); i != e; ++i) {
      SDValue LHSOp = LHS.getOperand(i);
      SDValue RHSOp = RHS.getOperand(i);

      // Can't fold divide by zero.
      if (N->getOpcode() == ISD::SDIV || N->getOpcode() == ISD::UDIV ||
          N->getOpcode() == ISD::FDIV) {
        if (isNullConstant(RHSOp) || (RHSOp.getOpcode() == ISD::ConstantFP &&
             cast<ConstantFPSDNode>(RHSOp.getNode())->isZero()))
          break;
      }

      EVT VT = LHSOp.getValueType();
      EVT RVT = RHSOp.getValueType();
      if (RVT != VT) {
        // Integer BUILD_VECTOR operands may have types larger than the element
        // size (e.g., when the element type is not legal).  Prior to type
        // legalization, the types may not match between the two BUILD_VECTORS.
        // Truncate one of the operands to make them match.
        if (RVT.getSizeInBits() > VT.getSizeInBits()) {
          RHSOp = DAG.getNode(ISD::TRUNCATE, SDLoc(N), VT, RHSOp);
        } else {
          LHSOp = DAG.getNode(ISD::TRUNCATE, SDLoc(N), RVT, LHSOp);
          VT = RVT;
        }
      }
      SDValue FoldOp = DAG.getNode(N->getOpcode(), SDLoc(LHS), VT,
                                   LHSOp, RHSOp);
      if (FoldOp.getOpcode() != ISD::UNDEF &&
          FoldOp.getOpcode() != ISD::Constant &&
          FoldOp.getOpcode() != ISD::ConstantFP)
        break;
      Ops.push_back(FoldOp);
      AddToWorklist(FoldOp.getNode());
    }

    if (Ops.size() == LHS.getNumOperands())
      return DAG.getNode(ISD::BUILD_VECTOR, SDLoc(N), LHS.getValueType(), Ops);
  }

  // Type legalization might introduce new shuffles in the DAG.
  // Fold (VBinOp (shuffle (A, Undef, Mask)), (shuffle (B, Undef, Mask)))
  //   -> (shuffle (VBinOp (A, B)), Undef, Mask).
  if (LegalTypes && isa<ShuffleVectorSDNode>(LHS) &&
      isa<ShuffleVectorSDNode>(RHS) && LHS.hasOneUse() && RHS.hasOneUse() &&
      LHS.getOperand(1).getOpcode() == ISD::UNDEF &&
      RHS.getOperand(1).getOpcode() == ISD::UNDEF) {
    ShuffleVectorSDNode *SVN0 = cast<ShuffleVectorSDNode>(LHS);
    ShuffleVectorSDNode *SVN1 = cast<ShuffleVectorSDNode>(RHS);

    if (SVN0->getMask().equals(SVN1->getMask())) {
      EVT VT = N->getValueType(0);
      SDValue UndefVector = LHS.getOperand(1);
      SDValue NewBinOp = DAG.getNode(N->getOpcode(), SDLoc(N), VT,
                                     LHS.getOperand(0), RHS.getOperand(0));
      AddUsersToWorklist(N);
      return DAG.getVectorShuffle(VT, SDLoc(N), NewBinOp, UndefVector,
                                  &SVN0->getMask()[0]);
    }
  }

  return SDValue();
}

SDValue DAGCombiner::SimplifySelect(SDLoc DL, SDValue N0,
                                    SDValue N1, SDValue N2){
  assert(N0.getOpcode() ==ISD::SETCC && "First argument must be a SetCC node!");

  SDValue SCC = SimplifySelectCC(DL, N0.getOperand(0), N0.getOperand(1), N1, N2,
                                 cast<CondCodeSDNode>(N0.getOperand(2))->get());

  // If we got a simplified select_cc node back from SimplifySelectCC, then
  // break it down into a new SETCC node, and a new SELECT node, and then return
  // the SELECT node, since we were called with a SELECT node.
  if (SCC.getNode()) {
    // Check to see if we got a select_cc back (to turn into setcc/select).
    // Otherwise, just return whatever node we got back, like fabs.
    if (SCC.getOpcode() == ISD::SELECT_CC) {
      SDValue SETCC = DAG.getNode(ISD::SETCC, SDLoc(N0),
                                  N0.getValueType(),
                                  SCC.getOperand(0), SCC.getOperand(1),
                                  SCC.getOperand(4));
      AddToWorklist(SETCC.getNode());
      return DAG.getSelect(SDLoc(SCC), SCC.getValueType(), SETCC,
                           SCC.getOperand(2), SCC.getOperand(3));
    }

    return SCC;
  }
  return SDValue();
}

/// Given a SELECT or a SELECT_CC node, where LHS and RHS are the two values
/// being selected between, see if we can simplify the select.  Callers of this
/// should assume that TheSelect is deleted if this returns true.  As such, they
/// should return the appropriate thing (e.g. the node) back to the top-level of
/// the DAG combiner loop to avoid it being looked at.
bool DAGCombiner::SimplifySelectOps(SDNode *TheSelect, SDValue LHS,
                                    SDValue RHS) {

  // fold (select (setcc x, -0.0, *lt), NaN, (fsqrt x))
  // The select + setcc is redundant, because fsqrt returns NaN for X < -0.
  if (const ConstantFPSDNode *NaN = isConstOrConstSplatFP(LHS)) {
    if (NaN->isNaN() && RHS.getOpcode() == ISD::FSQRT) {
      // We have: (select (setcc ?, ?, ?), NaN, (fsqrt ?))
      SDValue Sqrt = RHS;
      ISD::CondCode CC;
      SDValue CmpLHS;
      const ConstantFPSDNode *NegZero = nullptr;

      if (TheSelect->getOpcode() == ISD::SELECT_CC) {
        CC = dyn_cast<CondCodeSDNode>(TheSelect->getOperand(4))->get();
        CmpLHS = TheSelect->getOperand(0);
        NegZero = isConstOrConstSplatFP(TheSelect->getOperand(1));
      } else {
        // SELECT or VSELECT
        SDValue Cmp = TheSelect->getOperand(0);
        if (Cmp.getOpcode() == ISD::SETCC) {
          CC = dyn_cast<CondCodeSDNode>(Cmp.getOperand(2))->get();
          CmpLHS = Cmp.getOperand(0);
          NegZero = isConstOrConstSplatFP(Cmp.getOperand(1));
        }
      }
      if (NegZero && NegZero->isNegative() && NegZero->isZero() &&
          Sqrt.getOperand(0) == CmpLHS && (CC == ISD::SETOLT ||
          CC == ISD::SETULT || CC == ISD::SETLT)) {
        // We have: (select (setcc x, -0.0, *lt), NaN, (fsqrt x))
        CombineTo(TheSelect, Sqrt);
        return true;
      }
    }
  }
  // Cannot simplify select with vector condition
  if (TheSelect->getOperand(0).getValueType().isVector()) return false;

  // If this is a select from two identical things, try to pull the operation
  // through the select.
  if (LHS.getOpcode() != RHS.getOpcode() ||
      !LHS.hasOneUse() || !RHS.hasOneUse())
    return false;

  // If this is a load and the token chain is identical, replace the select
  // of two loads with a load through a select of the address to load from.
  // This triggers in things like "select bool X, 10.0, 123.0" after the FP
  // constants have been dropped into the constant pool.
  if (LHS.getOpcode() == ISD::LOAD) {
    LoadSDNode *LLD = cast<LoadSDNode>(LHS);
    LoadSDNode *RLD = cast<LoadSDNode>(RHS);

    // Token chains must be identical.
    if (LHS.getOperand(0) != RHS.getOperand(0) ||
        // Do not let this transformation reduce the number of volatile loads.
        LLD->isVolatile() || RLD->isVolatile() ||
        // FIXME: If either is a pre/post inc/dec load,
        // we'd need to split out the address adjustment.
        LLD->isIndexed() || RLD->isIndexed() ||
        // If this is an EXTLOAD, the VT's must match.
        LLD->getMemoryVT() != RLD->getMemoryVT() ||
        // If this is an EXTLOAD, the kind of extension must match.
        (LLD->getExtensionType() != RLD->getExtensionType() &&
         // The only exception is if one of the extensions is anyext.
         LLD->getExtensionType() != ISD::EXTLOAD &&
         RLD->getExtensionType() != ISD::EXTLOAD) ||
        // FIXME: this discards src value information.  This is
        // over-conservative. It would be beneficial to be able to remember
        // both potential memory locations.  Since we are discarding
        // src value info, don't do the transformation if the memory
        // locations are not in the default address space.
        LLD->getPointerInfo().getAddrSpace() != 0 ||
        RLD->getPointerInfo().getAddrSpace() != 0 ||
        !TLI.isOperationLegalOrCustom(TheSelect->getOpcode(),
                                      LLD->getBasePtr().getValueType()))
      return false;

    // Check that the select condition doesn't reach either load.  If so,
    // folding this will induce a cycle into the DAG.  If not, this is safe to
    // xform, so create a select of the addresses.
    SDValue Addr;
    if (TheSelect->getOpcode() == ISD::SELECT) {
      SDNode *CondNode = TheSelect->getOperand(0).getNode();
      if ((LLD->hasAnyUseOfValue(1) && LLD->isPredecessorOf(CondNode)) ||
          (RLD->hasAnyUseOfValue(1) && RLD->isPredecessorOf(CondNode)))
        return false;
      // The loads must not depend on one another.
      if (LLD->isPredecessorOf(RLD) ||
          RLD->isPredecessorOf(LLD))
        return false;
      Addr = DAG.getSelect(SDLoc(TheSelect),
                           LLD->getBasePtr().getValueType(),
                           TheSelect->getOperand(0), LLD->getBasePtr(),
                           RLD->getBasePtr());
    } else {  // Otherwise SELECT_CC
      SDNode *CondLHS = TheSelect->getOperand(0).getNode();
      SDNode *CondRHS = TheSelect->getOperand(1).getNode();

      if ((LLD->hasAnyUseOfValue(1) &&
           (LLD->isPredecessorOf(CondLHS) || LLD->isPredecessorOf(CondRHS))) ||
          (RLD->hasAnyUseOfValue(1) &&
           (RLD->isPredecessorOf(CondLHS) || RLD->isPredecessorOf(CondRHS))))
        return false;

      Addr = DAG.getNode(ISD::SELECT_CC, SDLoc(TheSelect),
                         LLD->getBasePtr().getValueType(),
                         TheSelect->getOperand(0),
                         TheSelect->getOperand(1),
                         LLD->getBasePtr(), RLD->getBasePtr(),
                         TheSelect->getOperand(4));
    }

    SDValue Load;
    // It is safe to replace the two loads if they have different alignments,
    // but the new load must be the minimum (most restrictive) alignment of the
    // inputs.
    bool isInvariant = LLD->isInvariant() & RLD->isInvariant();
    unsigned Alignment = std::min(LLD->getAlignment(), RLD->getAlignment());
    if (LLD->getExtensionType() == ISD::NON_EXTLOAD) {
      Load = DAG.getLoad(TheSelect->getValueType(0),
                         SDLoc(TheSelect),
                         // FIXME: Discards pointer and AA info.
                         LLD->getChain(), Addr, MachinePointerInfo(),
                         LLD->isVolatile(), LLD->isNonTemporal(),
                         isInvariant, Alignment);
    } else {
      Load = DAG.getExtLoad(LLD->getExtensionType() == ISD::EXTLOAD ?
                            RLD->getExtensionType() : LLD->getExtensionType(),
                            SDLoc(TheSelect),
                            TheSelect->getValueType(0),
                            // FIXME: Discards pointer and AA info.
                            LLD->getChain(), Addr, MachinePointerInfo(),
                            LLD->getMemoryVT(), LLD->isVolatile(),
                            LLD->isNonTemporal(), isInvariant, Alignment);
    }

    // Users of the select now use the result of the load.
    CombineTo(TheSelect, Load);

    // Users of the old loads now use the new load's chain.  We know the
    // old-load value is dead now.
    CombineTo(LHS.getNode(), Load.getValue(0), Load.getValue(1));
    CombineTo(RHS.getNode(), Load.getValue(0), Load.getValue(1));
    return true;
  }

  return false;
}

/// Simplify an expression of the form (N0 cond N1) ? N2 : N3
/// where 'cond' is the comparison specified by CC.
SDValue DAGCombiner::SimplifySelectCC(SDLoc DL, SDValue N0, SDValue N1,
                                      SDValue N2, SDValue N3,
                                      ISD::CondCode CC, bool NotExtCompare) {
  // (x ? y : y) -> y.
  if (N2 == N3) return N2;

  EVT VT = N2.getValueType();
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1.getNode());
  ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(N2.getNode());

  // Determine if the condition we're dealing with is constant
  SDValue SCC = SimplifySetCC(getSetCCResultType(N0.getValueType()),
                              N0, N1, CC, DL, false);
  if (SCC.getNode()) AddToWorklist(SCC.getNode());

  if (ConstantSDNode *SCCC = dyn_cast_or_null<ConstantSDNode>(SCC.getNode())) {
    // fold select_cc true, x, y -> x
    // fold select_cc false, x, y -> y
    return !SCCC->isNullValue() ? N2 : N3;
  }

  // Check to see if we can simplify the select into an fabs node
  if (ConstantFPSDNode *CFP = dyn_cast<ConstantFPSDNode>(N1)) {
    // Allow either -0.0 or 0.0
    if (CFP->isZero()) {
      // select (setg[te] X, +/-0.0), X, fneg(X) -> fabs
      if ((CC == ISD::SETGE || CC == ISD::SETGT) &&
          N0 == N2 && N3.getOpcode() == ISD::FNEG &&
          N2 == N3.getOperand(0))
        return DAG.getNode(ISD::FABS, DL, VT, N0);

      // select (setl[te] X, +/-0.0), fneg(X), X -> fabs
      if ((CC == ISD::SETLT || CC == ISD::SETLE) &&
          N0 == N3 && N2.getOpcode() == ISD::FNEG &&
          N2.getOperand(0) == N3)
        return DAG.getNode(ISD::FABS, DL, VT, N3);
    }
  }

  // Turn "(a cond b) ? 1.0f : 2.0f" into "load (tmp + ((a cond b) ? 0 : 4)"
  // where "tmp" is a constant pool entry containing an array with 1.0 and 2.0
  // in it.  This is a win when the constant is not otherwise available because
  // it replaces two constant pool loads with one.  We only do this if the FP
  // type is known to be legal, because if it isn't, then we are before legalize
  // types an we want the other legalization to happen first (e.g. to avoid
  // messing with soft float) and if the ConstantFP is not legal, because if
  // it is legal, we may not need to store the FP constant in a constant pool.
  if (ConstantFPSDNode *TV = dyn_cast<ConstantFPSDNode>(N2))
    if (ConstantFPSDNode *FV = dyn_cast<ConstantFPSDNode>(N3)) {
      if (TLI.isTypeLegal(N2.getValueType()) &&
          (TLI.getOperationAction(ISD::ConstantFP, N2.getValueType()) !=
               TargetLowering::Legal &&
           !TLI.isFPImmLegal(TV->getValueAPF(), TV->getValueType(0)) &&
           !TLI.isFPImmLegal(FV->getValueAPF(), FV->getValueType(0))) &&
          // If both constants have multiple uses, then we won't need to do an
          // extra load, they are likely around in registers for other users.
          (TV->hasOneUse() || FV->hasOneUse())) {
        Constant *Elts[] = {
          const_cast<ConstantFP*>(FV->getConstantFPValue()),
          const_cast<ConstantFP*>(TV->getConstantFPValue())
        };
        Type *FPTy = Elts[0]->getType();
        const DataLayout &TD = DAG.getDataLayout();

        // Create a ConstantArray of the two constants.
        Constant *CA = ConstantArray::get(ArrayType::get(FPTy, 2), Elts);
        SDValue CPIdx =
            DAG.getConstantPool(CA, TLI.getPointerTy(DAG.getDataLayout()),
                                TD.getPrefTypeAlignment(FPTy));
        unsigned Alignment = cast<ConstantPoolSDNode>(CPIdx)->getAlignment();

        // Get the offsets to the 0 and 1 element of the array so that we can
        // select between them.
        SDValue Zero = DAG.getIntPtrConstant(0, DL);
        unsigned EltSize = (unsigned)TD.getTypeAllocSize(Elts[0]->getType());
        SDValue One = DAG.getIntPtrConstant(EltSize, SDLoc(FV));

        SDValue Cond = DAG.getSetCC(DL,
                                    getSetCCResultType(N0.getValueType()),
                                    N0, N1, CC);
        AddToWorklist(Cond.getNode());
        SDValue CstOffset = DAG.getSelect(DL, Zero.getValueType(),
                                          Cond, One, Zero);
        AddToWorklist(CstOffset.getNode());
        CPIdx = DAG.getNode(ISD::ADD, DL, CPIdx.getValueType(), CPIdx,
                            CstOffset);
        AddToWorklist(CPIdx.getNode());
        return DAG.getLoad(TV->getValueType(0), DL, DAG.getEntryNode(), CPIdx,
                           MachinePointerInfo::getConstantPool(), false,
                           false, false, Alignment);
      }
    }

  // Check to see if we can perform the "gzip trick", transforming
  // (select_cc setlt X, 0, A, 0) -> (and (sra X, (sub size(X), 1), A)
  if (isNullConstant(N3) && CC == ISD::SETLT &&
      (isNullConstant(N1) ||                 // (a < 0) ? b : 0
       (isOneConstant(N1) && N0 == N2))) {   // (a < 1) ? a : 0
    EVT XType = N0.getValueType();
    EVT AType = N2.getValueType();
    if (XType.bitsGE(AType)) {
      // and (sra X, size(X)-1, A) -> "and (srl X, C2), A" iff A is a
      // single-bit constant.
      if (N2C && ((N2C->getAPIntValue() & (N2C->getAPIntValue() - 1)) == 0)) {
        unsigned ShCtV = N2C->getAPIntValue().logBase2();
        ShCtV = XType.getSizeInBits() - ShCtV - 1;
        SDValue ShCt = DAG.getConstant(ShCtV, SDLoc(N0),
                                       getShiftAmountTy(N0.getValueType()));
        SDValue Shift = DAG.getNode(ISD::SRL, SDLoc(N0),
                                    XType, N0, ShCt);
        AddToWorklist(Shift.getNode());

        if (XType.bitsGT(AType)) {
          Shift = DAG.getNode(ISD::TRUNCATE, DL, AType, Shift);
          AddToWorklist(Shift.getNode());
        }

        return DAG.getNode(ISD::AND, DL, AType, Shift, N2);
      }

      SDValue Shift = DAG.getNode(ISD::SRA, SDLoc(N0),
                                  XType, N0,
                                  DAG.getConstant(XType.getSizeInBits() - 1,
                                                  SDLoc(N0),
                                         getShiftAmountTy(N0.getValueType())));
      AddToWorklist(Shift.getNode());

      if (XType.bitsGT(AType)) {
        Shift = DAG.getNode(ISD::TRUNCATE, DL, AType, Shift);
        AddToWorklist(Shift.getNode());
      }

      return DAG.getNode(ISD::AND, DL, AType, Shift, N2);
    }
  }

  // fold (select_cc seteq (and x, y), 0, 0, A) -> (and (shr (shl x)) A)
  // where y is has a single bit set.
  // A plaintext description would be, we can turn the SELECT_CC into an AND
  // when the condition can be materialized as an all-ones register.  Any
  // single bit-test can be materialized as an all-ones register with
  // shift-left and shift-right-arith.
  if (CC == ISD::SETEQ && N0->getOpcode() == ISD::AND &&
      N0->getValueType(0) == VT && isNullConstant(N1) && isNullConstant(N2)) {
    SDValue AndLHS = N0->getOperand(0);
    ConstantSDNode *ConstAndRHS = dyn_cast<ConstantSDNode>(N0->getOperand(1));
    if (ConstAndRHS && ConstAndRHS->getAPIntValue().countPopulation() == 1) {
      // Shift the tested bit over the sign bit.
      APInt AndMask = ConstAndRHS->getAPIntValue();
      SDValue ShlAmt =
        DAG.getConstant(AndMask.countLeadingZeros(), SDLoc(AndLHS),
                        getShiftAmountTy(AndLHS.getValueType()));
      SDValue Shl = DAG.getNode(ISD::SHL, SDLoc(N0), VT, AndLHS, ShlAmt);

      // Now arithmetic right shift it all the way over, so the result is either
      // all-ones, or zero.
      SDValue ShrAmt =
        DAG.getConstant(AndMask.getBitWidth() - 1, SDLoc(Shl),
                        getShiftAmountTy(Shl.getValueType()));
      SDValue Shr = DAG.getNode(ISD::SRA, SDLoc(N0), VT, Shl, ShrAmt);

      return DAG.getNode(ISD::AND, DL, VT, Shr, N3);
    }
  }

  // fold select C, 16, 0 -> shl C, 4
  if (N2C && isNullConstant(N3) && N2C->getAPIntValue().isPowerOf2() &&
      TLI.getBooleanContents(N0.getValueType()) ==
          TargetLowering::ZeroOrOneBooleanContent) {

    // If the caller doesn't want us to simplify this into a zext of a compare,
    // don't do it.
    if (NotExtCompare && N2C->isOne())
      return SDValue();

    // Get a SetCC of the condition
    // NOTE: Don't create a SETCC if it's not legal on this target.
    if (!LegalOperations ||
        TLI.isOperationLegal(ISD::SETCC,
          LegalTypes ? getSetCCResultType(N0.getValueType()) : MVT::i1)) {
      SDValue Temp, SCC;
      // cast from setcc result type to select result type
      if (LegalTypes) {
        SCC  = DAG.getSetCC(DL, getSetCCResultType(N0.getValueType()),
                            N0, N1, CC);
        if (N2.getValueType().bitsLT(SCC.getValueType()))
          Temp = DAG.getZeroExtendInReg(SCC, SDLoc(N2),
                                        N2.getValueType());
        else
          Temp = DAG.getNode(ISD::ZERO_EXTEND, SDLoc(N2),
                             N2.getValueType(), SCC);
      } else {
        SCC  = DAG.getSetCC(SDLoc(N0), MVT::i1, N0, N1, CC);
        Temp = DAG.getNode(ISD::ZERO_EXTEND, SDLoc(N2),
                           N2.getValueType(), SCC);
      }

      AddToWorklist(SCC.getNode());
      AddToWorklist(Temp.getNode());

      if (N2C->isOne())
        return Temp;

      // shl setcc result by log2 n2c
      return DAG.getNode(
          ISD::SHL, DL, N2.getValueType(), Temp,
          DAG.getConstant(N2C->getAPIntValue().logBase2(), SDLoc(Temp),
                          getShiftAmountTy(Temp.getValueType())));
    }
  }

  // Check to see if this is the equivalent of setcc
  // FIXME: Turn all of these into setcc if setcc if setcc is legal
  // otherwise, go ahead with the folds.
  if (0 && isNullConstant(N3) && isOneConstant(N2)) {
    EVT XType = N0.getValueType();
    if (!LegalOperations ||
        TLI.isOperationLegal(ISD::SETCC, getSetCCResultType(XType))) {
      SDValue Res = DAG.getSetCC(DL, getSetCCResultType(XType), N0, N1, CC);
      if (Res.getValueType() != VT)
        Res = DAG.getNode(ISD::ZERO_EXTEND, DL, VT, Res);
      return Res;
    }

    // fold (seteq X, 0) -> (srl (ctlz X, log2(size(X))))
    if (isNullConstant(N1) && CC == ISD::SETEQ &&
        (!LegalOperations ||
         TLI.isOperationLegal(ISD::CTLZ, XType))) {
      SDValue Ctlz = DAG.getNode(ISD::CTLZ, SDLoc(N0), XType, N0);
      return DAG.getNode(ISD::SRL, DL, XType, Ctlz,
                         DAG.getConstant(Log2_32(XType.getSizeInBits()),
                                         SDLoc(Ctlz),
                                       getShiftAmountTy(Ctlz.getValueType())));
    }
    // fold (setgt X, 0) -> (srl (and (-X, ~X), size(X)-1))
    if (isNullConstant(N1) && CC == ISD::SETGT) {
      SDLoc DL(N0);
      SDValue NegN0 = DAG.getNode(ISD::SUB, DL,
                                  XType, DAG.getConstant(0, DL, XType), N0);
      SDValue NotN0 = DAG.getNOT(DL, N0, XType);
      return DAG.getNode(ISD::SRL, DL, XType,
                         DAG.getNode(ISD::AND, DL, XType, NegN0, NotN0),
                         DAG.getConstant(XType.getSizeInBits() - 1, DL,
                                         getShiftAmountTy(XType)));
    }
    // fold (setgt X, -1) -> (xor (srl (X, size(X)-1), 1))
    if (isAllOnesConstant(N1) && CC == ISD::SETGT) {
      SDLoc DL(N0);
      SDValue Sign = DAG.getNode(ISD::SRL, DL, XType, N0,
                                 DAG.getConstant(XType.getSizeInBits() - 1, DL,
                                         getShiftAmountTy(N0.getValueType())));
      return DAG.getNode(ISD::XOR, DL, XType, Sign, DAG.getConstant(1, DL,
                                                                    XType));
    }
  }

  // Check to see if this is an integer abs.
  // select_cc setg[te] X,  0,  X, -X ->
  // select_cc setgt    X, -1,  X, -X ->
  // select_cc setl[te] X,  0, -X,  X ->
  // select_cc setlt    X,  1, -X,  X ->
  // Y = sra (X, size(X)-1); xor (add (X, Y), Y)
  if (N1C) {
    ConstantSDNode *SubC = nullptr;
    if (((N1C->isNullValue() && (CC == ISD::SETGT || CC == ISD::SETGE)) ||
         (N1C->isAllOnesValue() && CC == ISD::SETGT)) &&
        N0 == N2 && N3.getOpcode() == ISD::SUB && N0 == N3.getOperand(1))
      SubC = dyn_cast<ConstantSDNode>(N3.getOperand(0));
    else if (((N1C->isNullValue() && (CC == ISD::SETLT || CC == ISD::SETLE)) ||
              (N1C->isOne() && CC == ISD::SETLT)) &&
             N0 == N3 && N2.getOpcode() == ISD::SUB && N0 == N2.getOperand(1))
      SubC = dyn_cast<ConstantSDNode>(N2.getOperand(0));

    EVT XType = N0.getValueType();
    if (SubC && SubC->isNullValue() && XType.isInteger()) {
      SDLoc DL(N0);
      SDValue Shift = DAG.getNode(ISD::SRA, DL, XType,
                                  N0,
                                  DAG.getConstant(XType.getSizeInBits() - 1, DL,
                                         getShiftAmountTy(N0.getValueType())));
      SDValue Add = DAG.getNode(ISD::ADD, DL,
                                XType, N0, Shift);
      AddToWorklist(Shift.getNode());
      AddToWorklist(Add.getNode());
      return DAG.getNode(ISD::XOR, DL, XType, Add, Shift);
    }
  }

  return SDValue();
}

/// This is a stub for TargetLowering::SimplifySetCC.
SDValue DAGCombiner::SimplifySetCC(EVT VT, SDValue N0,
                                   SDValue N1, ISD::CondCode Cond,
                                   SDLoc DL, bool foldBooleans) {
  TargetLowering::DAGCombinerInfo
    DagCombineInfo(DAG, Level, false, this);
  return TLI.SimplifySetCC(VT, N0, N1, Cond, foldBooleans, DagCombineInfo, DL);
}

/// Given an ISD::SDIV node expressing a divide by constant, return
/// a DAG expression to select that will generate the same value by multiplying
/// by a magic number.
/// Ref: "Hacker's Delight" or "The PowerPC Compiler Writer's Guide".
SDValue DAGCombiner::BuildSDIV(SDNode *N) {
  ConstantSDNode *C = isConstOrConstSplat(N->getOperand(1));
  if (!C)
    return SDValue();

  // Avoid division by zero.
  if (C->isNullValue())
    return SDValue();

  std::vector<SDNode*> Built;
  SDValue S =
      TLI.BuildSDIV(N, C->getAPIntValue(), DAG, LegalOperations, &Built);

  for (SDNode *N : Built)
    AddToWorklist(N);
  return S;
}

/// Given an ISD::SDIV node expressing a divide by constant power of 2, return a
/// DAG expression that will generate the same value by right shifting.
SDValue DAGCombiner::BuildSDIVPow2(SDNode *N) {
  ConstantSDNode *C = isConstOrConstSplat(N->getOperand(1));
  if (!C)
    return SDValue();

  // Avoid division by zero.
  if (C->isNullValue())
    return SDValue();

  std::vector<SDNode *> Built;
  SDValue S = TLI.BuildSDIVPow2(N, C->getAPIntValue(), DAG, &Built);

  for (SDNode *N : Built)
    AddToWorklist(N);
  return S;
}

/// Given an ISD::UDIV node expressing a divide by constant, return a DAG
/// expression that will generate the same value by multiplying by a magic
/// number.
/// Ref: "Hacker's Delight" or "The PowerPC Compiler Writer's Guide".
SDValue DAGCombiner::BuildUDIV(SDNode *N) {
  ConstantSDNode *C = isConstOrConstSplat(N->getOperand(1));
  if (!C)
    return SDValue();

  // Avoid division by zero.
  if (C->isNullValue())
    return SDValue();

  std::vector<SDNode*> Built;
  SDValue S =
      TLI.BuildUDIV(N, C->getAPIntValue(), DAG, LegalOperations, &Built);

  for (SDNode *N : Built)
    AddToWorklist(N);
  return S;
}

SDValue DAGCombiner::BuildReciprocalEstimate(SDValue Op) {
  if (Level >= AfterLegalizeDAG)
    return SDValue();

  // Expose the DAG combiner to the target combiner implementations.
  TargetLowering::DAGCombinerInfo DCI(DAG, Level, false, this);

  unsigned Iterations = 0;
  if (SDValue Est = TLI.getRecipEstimate(Op, DCI, Iterations)) {
    if (Iterations) {
      // Newton iteration for a function: F(X) is X_{i+1} = X_i - F(X_i)/F'(X_i)
      // For the reciprocal, we need to find the zero of the function:
      //   F(X) = A X - 1 [which has a zero at X = 1/A]
      //     =>
      //   X_{i+1} = X_i (2 - A X_i) = X_i + X_i (1 - A X_i) [this second form
      //     does not require additional intermediate precision]
      EVT VT = Op.getValueType();
      SDLoc DL(Op);
      SDValue FPOne = DAG.getConstantFP(1.0, DL, VT);

      AddToWorklist(Est.getNode());

      // Newton iterations: Est = Est + Est (1 - Arg * Est)
      for (unsigned i = 0; i < Iterations; ++i) {
        SDValue NewEst = DAG.getNode(ISD::FMUL, DL, VT, Op, Est);
        AddToWorklist(NewEst.getNode());

        NewEst = DAG.getNode(ISD::FSUB, DL, VT, FPOne, NewEst);
        AddToWorklist(NewEst.getNode());

        NewEst = DAG.getNode(ISD::FMUL, DL, VT, Est, NewEst);
        AddToWorklist(NewEst.getNode());

        Est = DAG.getNode(ISD::FADD, DL, VT, Est, NewEst);
        AddToWorklist(Est.getNode());
      }
    }
    return Est;
  }

  return SDValue();
}

/// Newton iteration for a function: F(X) is X_{i+1} = X_i - F(X_i)/F'(X_i)
/// For the reciprocal sqrt, we need to find the zero of the function:
///   F(X) = 1/X^2 - A [which has a zero at X = 1/sqrt(A)]
///     =>
///   X_{i+1} = X_i (1.5 - A X_i^2 / 2)
/// As a result, we precompute A/2 prior to the iteration loop.
SDValue DAGCombiner::BuildRsqrtNROneConst(SDValue Arg, SDValue Est,
                                          unsigned Iterations) {
  EVT VT = Arg.getValueType();
  SDLoc DL(Arg);
  SDValue ThreeHalves = DAG.getConstantFP(1.5, DL, VT);

  // We now need 0.5 * Arg which we can write as (1.5 * Arg - Arg) so that
  // this entire sequence requires only one FP constant.
  SDValue HalfArg = DAG.getNode(ISD::FMUL, DL, VT, ThreeHalves, Arg);
  AddToWorklist(HalfArg.getNode());

  HalfArg = DAG.getNode(ISD::FSUB, DL, VT, HalfArg, Arg);
  AddToWorklist(HalfArg.getNode());

  // Newton iterations: Est = Est * (1.5 - HalfArg * Est * Est)
  for (unsigned i = 0; i < Iterations; ++i) {
    SDValue NewEst = DAG.getNode(ISD::FMUL, DL, VT, Est, Est);
    AddToWorklist(NewEst.getNode());

    NewEst = DAG.getNode(ISD::FMUL, DL, VT, HalfArg, NewEst);
    AddToWorklist(NewEst.getNode());

    NewEst = DAG.getNode(ISD::FSUB, DL, VT, ThreeHalves, NewEst);
    AddToWorklist(NewEst.getNode());

    Est = DAG.getNode(ISD::FMUL, DL, VT, Est, NewEst);
    AddToWorklist(Est.getNode());
  }
  return Est;
}

/// Newton iteration for a function: F(X) is X_{i+1} = X_i - F(X_i)/F'(X_i)
/// For the reciprocal sqrt, we need to find the zero of the function:
///   F(X) = 1/X^2 - A [which has a zero at X = 1/sqrt(A)]
///     =>
///   X_{i+1} = (-0.5 * X_i) * (A * X_i * X_i + (-3.0))
SDValue DAGCombiner::BuildRsqrtNRTwoConst(SDValue Arg, SDValue Est,
                                          unsigned Iterations) {
  EVT VT = Arg.getValueType();
  SDLoc DL(Arg);
  SDValue MinusThree = DAG.getConstantFP(-3.0, DL, VT);
  SDValue MinusHalf = DAG.getConstantFP(-0.5, DL, VT);

  // Newton iterations: Est = -0.5 * Est * (-3.0 + Arg * Est * Est)
  for (unsigned i = 0; i < Iterations; ++i) {
    SDValue HalfEst = DAG.getNode(ISD::FMUL, DL, VT, Est, MinusHalf);
    AddToWorklist(HalfEst.getNode());

    Est = DAG.getNode(ISD::FMUL, DL, VT, Est, Est);
    AddToWorklist(Est.getNode());

    Est = DAG.getNode(ISD::FMUL, DL, VT, Est, Arg);
    AddToWorklist(Est.getNode());

    Est = DAG.getNode(ISD::FADD, DL, VT, Est, MinusThree);
    AddToWorklist(Est.getNode());

    Est = DAG.getNode(ISD::FMUL, DL, VT, Est, HalfEst);
    AddToWorklist(Est.getNode());
  }
  return Est;
}

SDValue DAGCombiner::BuildRsqrtEstimate(SDValue Op) {
  if (Level >= AfterLegalizeDAG)
    return SDValue();

  // Expose the DAG combiner to the target combiner implementations.
  TargetLowering::DAGCombinerInfo DCI(DAG, Level, false, this);
  unsigned Iterations = 0;
  bool UseOneConstNR = false;
  if (SDValue Est = TLI.getRsqrtEstimate(Op, DCI, Iterations, UseOneConstNR)) {
    AddToWorklist(Est.getNode());
    if (Iterations) {
      Est = UseOneConstNR ?
        BuildRsqrtNROneConst(Op, Est, Iterations) :
        BuildRsqrtNRTwoConst(Op, Est, Iterations);
    }
    return Est;
  }

  return SDValue();
}

/// Return true if base is a frame index, which is known not to alias with
/// anything but itself.  Provides base object and offset as results.
static bool FindBaseOffset(SDValue Ptr, SDValue &Base, int64_t &Offset,
                           const GlobalValue *&GV, const void *&CV) {
  // Assume it is a primitive operation.
  Base = Ptr; Offset = 0; GV = nullptr; CV = nullptr;

  // If it's an adding a simple constant then integrate the offset.
  if (Base.getOpcode() == ISD::ADD) {
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Base.getOperand(1))) {
      Base = Base.getOperand(0);
      Offset += C->getZExtValue();
    }
  }

  // Return the underlying GlobalValue, and update the Offset.  Return false
  // for GlobalAddressSDNode since the same GlobalAddress may be represented
  // by multiple nodes with different offsets.
  if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Base)) {
    GV = G->getGlobal();
    Offset += G->getOffset();
    return false;
  }

  // Return the underlying Constant value, and update the Offset.  Return false
  // for ConstantSDNodes since the same constant pool entry may be represented
  // by multiple nodes with different offsets.
  if (ConstantPoolSDNode *C = dyn_cast<ConstantPoolSDNode>(Base)) {
    CV = C->isMachineConstantPoolEntry() ? (const void *)C->getMachineCPVal()
                                         : (const void *)C->getConstVal();
    Offset += C->getOffset();
    return false;
  }
  // If it's any of the following then it can't alias with anything but itself.
  return isa<FrameIndexSDNode>(Base);
}

/// Return true if there is any possibility that the two addresses overlap.
bool DAGCombiner::isAlias(LSBaseSDNode *Op0, LSBaseSDNode *Op1) const {
  // If they are the same then they must be aliases.
  if (Op0->getBasePtr() == Op1->getBasePtr()) return true;

  // If they are both volatile then they cannot be reordered.
  if (Op0->isVolatile() && Op1->isVolatile()) return true;

  // If one operation reads from invariant memory, and the other may store, they
  // cannot alias. These should really be checking the equivalent of mayWrite,
  // but it only matters for memory nodes other than load /store.
  if (Op0->isInvariant() && Op1->writeMem())
    return false;

  if (Op1->isInvariant() && Op0->writeMem())
    return false;

  // Gather base node and offset information.
  SDValue Base1, Base2;
  int64_t Offset1, Offset2;
  const GlobalValue *GV1, *GV2;
  const void *CV1, *CV2;
  bool isFrameIndex1 = FindBaseOffset(Op0->getBasePtr(),
                                      Base1, Offset1, GV1, CV1);
  bool isFrameIndex2 = FindBaseOffset(Op1->getBasePtr(),
                                      Base2, Offset2, GV2, CV2);

  // If they have a same base address then check to see if they overlap.
  if (Base1 == Base2 || (GV1 && (GV1 == GV2)) || (CV1 && (CV1 == CV2)))
    return !((Offset1 + (Op0->getMemoryVT().getSizeInBits() >> 3)) <= Offset2 ||
             (Offset2 + (Op1->getMemoryVT().getSizeInBits() >> 3)) <= Offset1);

  // It is possible for different frame indices to alias each other, mostly
  // when tail call optimization reuses return address slots for arguments.
  // To catch this case, look up the actual index of frame indices to compute
  // the real alias relationship.
  if (isFrameIndex1 && isFrameIndex2) {
    MachineFrameInfo *MFI = DAG.getMachineFunction().getFrameInfo();
    Offset1 += MFI->getObjectOffset(cast<FrameIndexSDNode>(Base1)->getIndex());
    Offset2 += MFI->getObjectOffset(cast<FrameIndexSDNode>(Base2)->getIndex());
    return !((Offset1 + (Op0->getMemoryVT().getSizeInBits() >> 3)) <= Offset2 ||
             (Offset2 + (Op1->getMemoryVT().getSizeInBits() >> 3)) <= Offset1);
  }

  // Otherwise, if we know what the bases are, and they aren't identical, then
  // we know they cannot alias.
  if ((isFrameIndex1 || CV1 || GV1) && (isFrameIndex2 || CV2 || GV2))
    return false;

  // If we know required SrcValue1 and SrcValue2 have relatively large alignment
  // compared to the size and offset of the access, we may be able to prove they
  // do not alias.  This check is conservative for now to catch cases created by
  // splitting vector types.
  if ((Op0->getOriginalAlignment() == Op1->getOriginalAlignment()) &&
      (Op0->getSrcValueOffset() != Op1->getSrcValueOffset()) &&
      (Op0->getMemoryVT().getSizeInBits() >> 3 ==
       Op1->getMemoryVT().getSizeInBits() >> 3) &&
      (Op0->getOriginalAlignment() > Op0->getMemoryVT().getSizeInBits()) >> 3) {
    int64_t OffAlign1 = Op0->getSrcValueOffset() % Op0->getOriginalAlignment();
    int64_t OffAlign2 = Op1->getSrcValueOffset() % Op1->getOriginalAlignment();

    // There is no overlap between these relatively aligned accesses of similar
    // size, return no alias.
    if ((OffAlign1 + (Op0->getMemoryVT().getSizeInBits() >> 3)) <= OffAlign2 ||
        (OffAlign2 + (Op1->getMemoryVT().getSizeInBits() >> 3)) <= OffAlign1)
      return false;
  }

  bool UseAA = CombinerGlobalAA.getNumOccurrences() > 0
                   ? CombinerGlobalAA
                   : DAG.getSubtarget().useAA();
#ifndef NDEBUG
  if (CombinerAAOnlyFunc.getNumOccurrences() &&
      CombinerAAOnlyFunc != DAG.getMachineFunction().getName())
    UseAA = false;
#endif
  if (UseAA &&
      Op0->getMemOperand()->getValue() && Op1->getMemOperand()->getValue()) {
    // Use alias analysis information.
    int64_t MinOffset = std::min(Op0->getSrcValueOffset(),
                                 Op1->getSrcValueOffset());
    int64_t Overlap1 = (Op0->getMemoryVT().getSizeInBits() >> 3) +
        Op0->getSrcValueOffset() - MinOffset;
    int64_t Overlap2 = (Op1->getMemoryVT().getSizeInBits() >> 3) +
        Op1->getSrcValueOffset() - MinOffset;
    AliasResult AAResult =
        AA.alias(MemoryLocation(Op0->getMemOperand()->getValue(), Overlap1,
                                UseTBAA ? Op0->getAAInfo() : AAMDNodes()),
                 MemoryLocation(Op1->getMemOperand()->getValue(), Overlap2,
                                UseTBAA ? Op1->getAAInfo() : AAMDNodes()));
    if (AAResult == NoAlias)
      return false;
  }

  // Otherwise we have to assume they alias.
  return true;
}

/// Walk up chain skipping non-aliasing memory nodes,
/// looking for aliasing nodes and adding them to the Aliases vector.
void DAGCombiner::GatherAllAliases(SDNode *N, SDValue OriginalChain,
                                   SmallVectorImpl<SDValue> &Aliases) {
  SmallVector<SDValue, 8> Chains;     // List of chains to visit.
  SmallPtrSet<SDNode *, 16> Visited;  // Visited node set.

  // Get alias information for node.
  bool IsLoad = isa<LoadSDNode>(N) && !cast<LSBaseSDNode>(N)->isVolatile();

  // Starting off.
  Chains.push_back(OriginalChain);
  unsigned Depth = 0;

  // Look at each chain and determine if it is an alias.  If so, add it to the
  // aliases list.  If not, then continue up the chain looking for the next
  // candidate.
  while (!Chains.empty()) {
    SDValue Chain = Chains.pop_back_val();

    // For TokenFactor nodes, look at each operand and only continue up the
    // chain until we find two aliases.  If we've seen two aliases, assume we'll
    // find more and revert to original chain since the xform is unlikely to be
    // profitable.
    //
    // FIXME: The depth check could be made to return the last non-aliasing
    // chain we found before we hit a tokenfactor rather than the original
    // chain.
    if (Depth > 6 || Aliases.size() == 2) {
      Aliases.clear();
      Aliases.push_back(OriginalChain);
      return;
    }

    // Don't bother if we've been before.
    if (!Visited.insert(Chain.getNode()).second)
      continue;

    switch (Chain.getOpcode()) {
    case ISD::EntryToken:
      // Entry token is ideal chain operand, but handled in FindBetterChain.
      break;

    case ISD::LOAD:
    case ISD::STORE: {
      // Get alias information for Chain.
      bool IsOpLoad = isa<LoadSDNode>(Chain.getNode()) &&
          !cast<LSBaseSDNode>(Chain.getNode())->isVolatile();

      // If chain is alias then stop here.
      if (!(IsLoad && IsOpLoad) &&
          isAlias(cast<LSBaseSDNode>(N), cast<LSBaseSDNode>(Chain.getNode()))) {
        Aliases.push_back(Chain);
      } else {
        // Look further up the chain.
        Chains.push_back(Chain.getOperand(0));
        ++Depth;
      }
      break;
    }

    case ISD::TokenFactor:
      // We have to check each of the operands of the token factor for "small"
      // token factors, so we queue them up.  Adding the operands to the queue
      // (stack) in reverse order maintains the original order and increases the
      // likelihood that getNode will find a matching token factor (CSE.)
      if (Chain.getNumOperands() > 16) {
        Aliases.push_back(Chain);
        break;
      }
      for (unsigned n = Chain.getNumOperands(); n;)
        Chains.push_back(Chain.getOperand(--n));
      ++Depth;
      break;

    default:
      // For all other instructions we will just have to take what we can get.
      Aliases.push_back(Chain);
      break;
    }
  }

  // We need to be careful here to also search for aliases through the
  // value operand of a store, etc. Consider the following situation:
  //   Token1 = ...
  //   L1 = load Token1, %52
  //   S1 = store Token1, L1, %51
  //   L2 = load Token1, %52+8
  //   S2 = store Token1, L2, %51+8
  //   Token2 = Token(S1, S2)
  //   L3 = load Token2, %53
  //   S3 = store Token2, L3, %52
  //   L4 = load Token2, %53+8
  //   S4 = store Token2, L4, %52+8
  // If we search for aliases of S3 (which loads address %52), and we look
  // only through the chain, then we'll miss the trivial dependence on L1
  // (which also loads from %52). We then might change all loads and
  // stores to use Token1 as their chain operand, which could result in
  // copying %53 into %52 before copying %52 into %51 (which should
  // happen first).
  //
  // The problem is, however, that searching for such data dependencies
  // can become expensive, and the cost is not directly related to the
  // chain depth. Instead, we'll rule out such configurations here by
  // insisting that we've visited all chain users (except for users
  // of the original chain, which is not necessary). When doing this,
  // we need to look through nodes we don't care about (otherwise, things
  // like register copies will interfere with trivial cases).

  SmallVector<const SDNode *, 16> Worklist;
  for (const SDNode *N : Visited)
    if (N != OriginalChain.getNode())
      Worklist.push_back(N);

  while (!Worklist.empty()) {
    const SDNode *M = Worklist.pop_back_val();

    // We have already visited M, and want to make sure we've visited any uses
    // of M that we care about. For uses that we've not visisted, and don't
    // care about, queue them to the worklist.

    for (SDNode::use_iterator UI = M->use_begin(),
         UIE = M->use_end(); UI != UIE; ++UI)
      if (UI.getUse().getValueType() == MVT::Other &&
          Visited.insert(*UI).second) {
        if (isa<MemSDNode>(*UI)) {
          // We've not visited this use, and we care about it (it could have an
          // ordering dependency with the original node).
          Aliases.clear();
          Aliases.push_back(OriginalChain);
          return;
        }

        // We've not visited this use, but we don't care about it. Mark it as
        // visited and enqueue it to the worklist.
        Worklist.push_back(*UI);
      }
  }
}

/// Walk up chain skipping non-aliasing memory nodes, looking for a better chain
/// (aliasing node.)
SDValue DAGCombiner::FindBetterChain(SDNode *N, SDValue OldChain) {
  SmallVector<SDValue, 8> Aliases;  // Ops for replacing token factor.

  // Accumulate all the aliases to this node.
  GatherAllAliases(N, OldChain, Aliases);

  // If no operands then chain to entry token.
  if (Aliases.size() == 0)
    return DAG.getEntryNode();

  // If a single operand then chain to it.  We don't need to revisit it.
  if (Aliases.size() == 1)
    return Aliases[0];

  // Construct a custom tailored token factor.
  return DAG.getNode(ISD::TokenFactor, SDLoc(N), MVT::Other, Aliases);
}

/// This is the entry point for the file.
void SelectionDAG::Combine(CombineLevel Level, AliasAnalysis &AA,
                           CodeGenOpt::Level OptLevel) {
  /// This is the main entry point to this class.
  DAGCombiner(*this, AA, OptLevel).Run(Level);
}
