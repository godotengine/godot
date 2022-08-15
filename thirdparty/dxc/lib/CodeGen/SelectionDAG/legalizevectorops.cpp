//===-- LegalizeVectorOps.cpp - Implement SelectionDAG::LegalizeVectors ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SelectionDAG::LegalizeVectors method.
//
// The vector legalizer looks for vector operations which might need to be
// scalarized and legalizes them. This is a separate step from Legalize because
// scalarizing can introduce illegal types.  For example, suppose we have an
// ISD::SDIV of type v2i64 on x86-32.  The type is legal (for example, addition
// on a v2i64 is legal), but ISD::SDIV isn't legal, so we have to unroll the
// operation, which introduces nodes with the illegal type i64 which must be
// expanded.  Similarly, suppose we have an ISD::SRA of type v16i8 on PowerPC;
// the operation must be unrolled, which introduces nodes with the illegal
// type i8 which must be promoted.
//
// This does not legalize vector manipulations like ISD::BUILD_VECTOR,
// or operations that happen to take a vector which are custom-lowered;
// the legalization for such operations never produces nodes
// with illegal types, so it's okay to put off legalizing them until
// SelectionDAG::Legalize runs.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Target/TargetLowering.h"
using namespace llvm;

namespace {
class VectorLegalizer {
  SelectionDAG& DAG;
  const TargetLowering &TLI;
  bool Changed; // Keep track of whether anything changed

  /// For nodes that are of legal width, and that have more than one use, this
  /// map indicates what regularized operand to use.  This allows us to avoid
  /// legalizing the same thing more than once.
  SmallDenseMap<SDValue, SDValue, 64> LegalizedNodes;

  /// \brief Adds a node to the translation cache.
  void AddLegalizedOperand(SDValue From, SDValue To) {
    LegalizedNodes.insert(std::make_pair(From, To));
    // If someone requests legalization of the new node, return itself.
    if (From != To)
      LegalizedNodes.insert(std::make_pair(To, To));
  }

  /// \brief Legalizes the given node.
  SDValue LegalizeOp(SDValue Op);

  /// \brief Assuming the node is legal, "legalize" the results.
  SDValue TranslateLegalizeResults(SDValue Op, SDValue Result);

  /// \brief Implements unrolling a VSETCC.
  SDValue UnrollVSETCC(SDValue Op);

  /// \brief Implement expand-based legalization of vector operations.
  ///
  /// This is just a high-level routine to dispatch to specific code paths for
  /// operations to legalize them.
  SDValue Expand(SDValue Op);

  /// \brief Implements expansion for FNEG; falls back to UnrollVectorOp if
  /// FSUB isn't legal.
  ///
  /// Implements expansion for UINT_TO_FLOAT; falls back to UnrollVectorOp if
  /// SINT_TO_FLOAT and SHR on vectors isn't legal.
  SDValue ExpandUINT_TO_FLOAT(SDValue Op);

  /// \brief Implement expansion for SIGN_EXTEND_INREG using SRL and SRA.
  SDValue ExpandSEXTINREG(SDValue Op);

  /// \brief Implement expansion for ANY_EXTEND_VECTOR_INREG.
  ///
  /// Shuffles the low lanes of the operand into place and bitcasts to the proper
  /// type. The contents of the bits in the extended part of each element are
  /// undef.
  SDValue ExpandANY_EXTEND_VECTOR_INREG(SDValue Op);

  /// \brief Implement expansion for SIGN_EXTEND_VECTOR_INREG.
  ///
  /// Shuffles the low lanes of the operand into place, bitcasts to the proper
  /// type, then shifts left and arithmetic shifts right to introduce a sign
  /// extension.
  SDValue ExpandSIGN_EXTEND_VECTOR_INREG(SDValue Op);

  /// \brief Implement expansion for ZERO_EXTEND_VECTOR_INREG.
  ///
  /// Shuffles the low lanes of the operand into place and blends zeros into
  /// the remaining lanes, finally bitcasting to the proper type.
  SDValue ExpandZERO_EXTEND_VECTOR_INREG(SDValue Op);

  /// \brief Expand bswap of vectors into a shuffle if legal.
  SDValue ExpandBSWAP(SDValue Op);

  /// \brief Implement vselect in terms of XOR, AND, OR when blend is not
  /// supported by the target.
  SDValue ExpandVSELECT(SDValue Op);
  SDValue ExpandSELECT(SDValue Op);
  SDValue ExpandLoad(SDValue Op);
  SDValue ExpandStore(SDValue Op);
  SDValue ExpandFNEG(SDValue Op);

  /// \brief Implements vector promotion.
  ///
  /// This is essentially just bitcasting the operands to a different type and
  /// bitcasting the result back to the original type.
  SDValue Promote(SDValue Op);

  /// \brief Implements [SU]INT_TO_FP vector promotion.
  ///
  /// This is a [zs]ext of the input operand to the next size up.
  SDValue PromoteINT_TO_FP(SDValue Op);

  /// \brief Implements FP_TO_[SU]INT vector promotion of the result type.
  ///
  /// It is promoted to the next size up integer type.  The result is then
  /// truncated back to the original type.
  SDValue PromoteFP_TO_INT(SDValue Op, bool isSigned);

public:
  /// \brief Begin legalizer the vector operations in the DAG.
  bool Run();
  VectorLegalizer(SelectionDAG& dag) :
      DAG(dag), TLI(dag.getTargetLoweringInfo()), Changed(false) {}
};

bool VectorLegalizer::Run() {
  // Before we start legalizing vector nodes, check if there are any vectors.
  bool HasVectors = false;
  for (SelectionDAG::allnodes_iterator I = DAG.allnodes_begin(),
       E = std::prev(DAG.allnodes_end()); I != std::next(E); ++I) {
    // Check if the values of the nodes contain vectors. We don't need to check
    // the operands because we are going to check their values at some point.
    for (SDNode::value_iterator J = I->value_begin(), E = I->value_end();
         J != E; ++J)
      HasVectors |= J->isVector();

    // If we found a vector node we can start the legalization.
    if (HasVectors)
      break;
  }

  // If this basic block has no vectors then no need to legalize vectors.
  if (!HasVectors)
    return false;

  // The legalize process is inherently a bottom-up recursive process (users
  // legalize their uses before themselves).  Given infinite stack space, we
  // could just start legalizing on the root and traverse the whole graph.  In
  // practice however, this causes us to run out of stack space on large basic
  // blocks.  To avoid this problem, compute an ordering of the nodes where each
  // node is only legalized after all of its operands are legalized.
  DAG.AssignTopologicalOrder();
  for (SelectionDAG::allnodes_iterator I = DAG.allnodes_begin(),
       E = std::prev(DAG.allnodes_end()); I != std::next(E); ++I)
    LegalizeOp(SDValue(I, 0));

  // Finally, it's possible the root changed.  Get the new root.
  SDValue OldRoot = DAG.getRoot();
  assert(LegalizedNodes.count(OldRoot) && "Root didn't get legalized?");
  DAG.setRoot(LegalizedNodes[OldRoot]);

  LegalizedNodes.clear();

  // Remove dead nodes now.
  DAG.RemoveDeadNodes();

  return Changed;
}

SDValue VectorLegalizer::TranslateLegalizeResults(SDValue Op, SDValue Result) {
  // Generic legalization: just pass the operand through.
  for (unsigned i = 0, e = Op.getNode()->getNumValues(); i != e; ++i)
    AddLegalizedOperand(Op.getValue(i), Result.getValue(i));
  return Result.getValue(Op.getResNo());
}

SDValue VectorLegalizer::LegalizeOp(SDValue Op) {
  // Note that LegalizeOp may be reentered even from single-use nodes, which
  // means that we always must cache transformed nodes.
  DenseMap<SDValue, SDValue>::iterator I = LegalizedNodes.find(Op);
  if (I != LegalizedNodes.end()) return I->second;

  SDNode* Node = Op.getNode();

  // Legalize the operands
  SmallVector<SDValue, 8> Ops;
  for (const SDValue &Op : Node->op_values())
    Ops.push_back(LegalizeOp(Op));

  SDValue Result = SDValue(DAG.UpdateNodeOperands(Op.getNode(), Ops), 0);

  bool HasVectorValue = false;
  if (Op.getOpcode() == ISD::LOAD) {
    LoadSDNode *LD = cast<LoadSDNode>(Op.getNode());
    ISD::LoadExtType ExtType = LD->getExtensionType();
    if (LD->getMemoryVT().isVector() && ExtType != ISD::NON_EXTLOAD)
      switch (TLI.getLoadExtAction(LD->getExtensionType(), LD->getValueType(0),
                                   LD->getMemoryVT())) {
      default: llvm_unreachable("This action is not supported yet!");
      case TargetLowering::Legal:
        return TranslateLegalizeResults(Op, Result);
      case TargetLowering::Custom:
        if (SDValue Lowered = TLI.LowerOperation(Result, DAG)) {
          if (Lowered == Result)
            return TranslateLegalizeResults(Op, Lowered);
          Changed = true;
          if (Lowered->getNumValues() != Op->getNumValues()) {
            // This expanded to something other than the load. Assume the
            // lowering code took care of any chain values, and just handle the
            // returned value.
            assert(Result.getValue(1).use_empty() &&
                   "There are still live users of the old chain!");
            return LegalizeOp(Lowered);
          } else {
            return TranslateLegalizeResults(Op, Lowered);
          }
        }
      case TargetLowering::Expand:
        Changed = true;
        return LegalizeOp(ExpandLoad(Op));
      }
  } else if (Op.getOpcode() == ISD::STORE) {
    StoreSDNode *ST = cast<StoreSDNode>(Op.getNode());
    EVT StVT = ST->getMemoryVT();
    MVT ValVT = ST->getValue().getSimpleValueType();
    if (StVT.isVector() && ST->isTruncatingStore())
      switch (TLI.getTruncStoreAction(ValVT, StVT.getSimpleVT())) {
      default: llvm_unreachable("This action is not supported yet!");
      case TargetLowering::Legal:
        return TranslateLegalizeResults(Op, Result);
      case TargetLowering::Custom: {
        SDValue Lowered = TLI.LowerOperation(Result, DAG);
        Changed = Lowered != Result;
        return TranslateLegalizeResults(Op, Lowered);
      }
      case TargetLowering::Expand:
        Changed = true;
        return LegalizeOp(ExpandStore(Op));
      }
  } else if (Op.getOpcode() == ISD::MSCATTER)
    HasVectorValue = true;

  for (SDNode::value_iterator J = Node->value_begin(), E = Node->value_end();
       J != E;
       ++J)
    HasVectorValue |= J->isVector();
  if (!HasVectorValue)
    return TranslateLegalizeResults(Op, Result);

  EVT QueryType;
  switch (Op.getOpcode()) {
  default:
    return TranslateLegalizeResults(Op, Result);
  case ISD::ADD:
  case ISD::SUB:
  case ISD::MUL:
  case ISD::SDIV:
  case ISD::UDIV:
  case ISD::SREM:
  case ISD::UREM:
  case ISD::FADD:
  case ISD::FSUB:
  case ISD::FMUL:
  case ISD::FDIV:
  case ISD::FREM:
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:
  case ISD::SHL:
  case ISD::SRA:
  case ISD::SRL:
  case ISD::ROTL:
  case ISD::ROTR:
  case ISD::BSWAP:
  case ISD::CTLZ:
  case ISD::CTTZ:
  case ISD::CTLZ_ZERO_UNDEF:
  case ISD::CTTZ_ZERO_UNDEF:
  case ISD::CTPOP:
  case ISD::SELECT:
  case ISD::VSELECT:
  case ISD::SELECT_CC:
  case ISD::SETCC:
  case ISD::ZERO_EXTEND:
  case ISD::ANY_EXTEND:
  case ISD::TRUNCATE:
  case ISD::SIGN_EXTEND:
  case ISD::FP_TO_SINT:
  case ISD::FP_TO_UINT:
  case ISD::FNEG:
  case ISD::FABS:
  case ISD::FMINNUM:
  case ISD::FMAXNUM:
  case ISD::FCOPYSIGN:
  case ISD::FSQRT:
  case ISD::FSIN:
  case ISD::FCOS:
  case ISD::FPOWI:
  case ISD::FPOW:
  case ISD::FLOG:
  case ISD::FLOG2:
  case ISD::FLOG10:
  case ISD::FEXP:
  case ISD::FEXP2:
  case ISD::FCEIL:
  case ISD::FTRUNC:
  case ISD::FRINT:
  case ISD::FNEARBYINT:
  case ISD::FROUND:
  case ISD::FFLOOR:
  case ISD::FP_ROUND:
  case ISD::FP_EXTEND:
  case ISD::FMA:
  case ISD::SIGN_EXTEND_INREG:
  case ISD::ANY_EXTEND_VECTOR_INREG:
  case ISD::SIGN_EXTEND_VECTOR_INREG:
  case ISD::ZERO_EXTEND_VECTOR_INREG:
  case ISD::SMIN:
  case ISD::SMAX:
  case ISD::UMIN:
  case ISD::UMAX:
    QueryType = Node->getValueType(0);
    break;
  case ISD::FP_ROUND_INREG:
    QueryType = cast<VTSDNode>(Node->getOperand(1))->getVT();
    break;
  case ISD::SINT_TO_FP:
  case ISD::UINT_TO_FP:
    QueryType = Node->getOperand(0).getValueType();
    break;
  case ISD::MSCATTER:
    QueryType = cast<MaskedScatterSDNode>(Node)->getValue().getValueType();
    break;
  }

  switch (TLI.getOperationAction(Node->getOpcode(), QueryType)) {
  case TargetLowering::Promote:
    Result = Promote(Op);
    Changed = true;
    break;
  case TargetLowering::Legal:
    break;
  case TargetLowering::Custom: {
    SDValue Tmp1 = TLI.LowerOperation(Op, DAG);
    if (Tmp1.getNode()) {
      Result = Tmp1;
      break;
    }
    // FALL THROUGH
  }
  case TargetLowering::Expand:
    Result = Expand(Op);
  }

  // Make sure that the generated code is itself legal.
  if (Result != Op) {
    Result = LegalizeOp(Result);
    Changed = true;
  }

  // Note that LegalizeOp may be reentered even from single-use nodes, which
  // means that we always must cache transformed nodes.
  AddLegalizedOperand(Op, Result);
  return Result;
}

SDValue VectorLegalizer::Promote(SDValue Op) {
  // For a few operations there is a specific concept for promotion based on
  // the operand's type.
  switch (Op.getOpcode()) {
  case ISD::SINT_TO_FP:
  case ISD::UINT_TO_FP:
    // "Promote" the operation by extending the operand.
    return PromoteINT_TO_FP(Op);
  case ISD::FP_TO_UINT:
  case ISD::FP_TO_SINT:
    // Promote the operation by extending the operand.
    return PromoteFP_TO_INT(Op, Op->getOpcode() == ISD::FP_TO_SINT);
  }

  // There are currently two cases of vector promotion:
  // 1) Bitcasting a vector of integers to a different type to a vector of the
  //    same overall length. For example, x86 promotes ISD::AND v2i32 to v1i64.
  // 2) Extending a vector of floats to a vector of the same number of larger
  //    floats. For example, AArch64 promotes ISD::FADD on v4f16 to v4f32.
  MVT VT = Op.getSimpleValueType();
  assert(Op.getNode()->getNumValues() == 1 &&
         "Can't promote a vector with multiple results!");
  MVT NVT = TLI.getTypeToPromoteTo(Op.getOpcode(), VT);
  SDLoc dl(Op);
  SmallVector<SDValue, 4> Operands(Op.getNumOperands());

  for (unsigned j = 0; j != Op.getNumOperands(); ++j) {
    if (Op.getOperand(j).getValueType().isVector())
      if (Op.getOperand(j)
              .getValueType()
              .getVectorElementType()
              .isFloatingPoint() &&
          NVT.isVector() && NVT.getVectorElementType().isFloatingPoint())
        Operands[j] = DAG.getNode(ISD::FP_EXTEND, dl, NVT, Op.getOperand(j));
      else
        Operands[j] = DAG.getNode(ISD::BITCAST, dl, NVT, Op.getOperand(j));
    else
      Operands[j] = Op.getOperand(j);
  }

  Op = DAG.getNode(Op.getOpcode(), dl, NVT, Operands);
  if ((VT.isFloatingPoint() && NVT.isFloatingPoint()) ||
      (VT.isVector() && VT.getVectorElementType().isFloatingPoint() &&
       NVT.isVector() && NVT.getVectorElementType().isFloatingPoint()))
    return DAG.getNode(ISD::FP_ROUND, dl, VT, Op, DAG.getIntPtrConstant(0, dl));
  else
    return DAG.getNode(ISD::BITCAST, dl, VT, Op);
}

SDValue VectorLegalizer::PromoteINT_TO_FP(SDValue Op) {
  // INT_TO_FP operations may require the input operand be promoted even
  // when the type is otherwise legal.
  EVT VT = Op.getOperand(0).getValueType();
  assert(Op.getNode()->getNumValues() == 1 &&
         "Can't promote a vector with multiple results!");

  // Normal getTypeToPromoteTo() doesn't work here, as that will promote
  // by widening the vector w/ the same element width and twice the number
  // of elements. We want the other way around, the same number of elements,
  // each twice the width.
  //
  // Increase the bitwidth of the element to the next pow-of-two
  // (which is greater than 8 bits).

  EVT NVT = VT.widenIntegerVectorElementType(*DAG.getContext());
  assert(NVT.isSimple() && "Promoting to a non-simple vector type!");
  SDLoc dl(Op);
  SmallVector<SDValue, 4> Operands(Op.getNumOperands());

  unsigned Opc = Op.getOpcode() == ISD::UINT_TO_FP ? ISD::ZERO_EXTEND :
    ISD::SIGN_EXTEND;
  for (unsigned j = 0; j != Op.getNumOperands(); ++j) {
    if (Op.getOperand(j).getValueType().isVector())
      Operands[j] = DAG.getNode(Opc, dl, NVT, Op.getOperand(j));
    else
      Operands[j] = Op.getOperand(j);
  }

  return DAG.getNode(Op.getOpcode(), dl, Op.getValueType(), Operands);
}

// For FP_TO_INT we promote the result type to a vector type with wider
// elements and then truncate the result.  This is different from the default
// PromoteVector which uses bitcast to promote thus assumning that the
// promoted vector type has the same overall size.
SDValue VectorLegalizer::PromoteFP_TO_INT(SDValue Op, bool isSigned) {
  assert(Op.getNode()->getNumValues() == 1 &&
         "Can't promote a vector with multiple results!");
  EVT VT = Op.getValueType();

  EVT NewVT;
  unsigned NewOpc;
  while (1) {
    NewVT = VT.widenIntegerVectorElementType(*DAG.getContext());
    assert(NewVT.isSimple() && "Promoting to a non-simple vector type!");
    if (TLI.isOperationLegalOrCustom(ISD::FP_TO_SINT, NewVT)) {
      NewOpc = ISD::FP_TO_SINT;
      break;
    }
    if (!isSigned && TLI.isOperationLegalOrCustom(ISD::FP_TO_UINT, NewVT)) {
      NewOpc = ISD::FP_TO_UINT;
      break;
    }
  }

  SDLoc loc(Op);
  SDValue promoted  = DAG.getNode(NewOpc, SDLoc(Op), NewVT, Op.getOperand(0));
  return DAG.getNode(ISD::TRUNCATE, SDLoc(Op), VT, promoted);
}


SDValue VectorLegalizer::ExpandLoad(SDValue Op) {
  SDLoc dl(Op);
  LoadSDNode *LD = cast<LoadSDNode>(Op.getNode());
  SDValue Chain = LD->getChain();
  SDValue BasePTR = LD->getBasePtr();
  EVT SrcVT = LD->getMemoryVT();
  ISD::LoadExtType ExtType = LD->getExtensionType();

  SmallVector<SDValue, 8> Vals;
  SmallVector<SDValue, 8> LoadChains;
  unsigned NumElem = SrcVT.getVectorNumElements();

  EVT SrcEltVT = SrcVT.getScalarType();
  EVT DstEltVT = Op.getNode()->getValueType(0).getScalarType();

  if (SrcVT.getVectorNumElements() > 1 && !SrcEltVT.isByteSized()) {
    // When elements in a vector is not byte-addressable, we cannot directly
    // load each element by advancing pointer, which could only address bytes.
    // Instead, we load all significant words, mask bits off, and concatenate
    // them to form each element. Finally, they are extended to destination
    // scalar type to build the destination vector.
    EVT WideVT = TLI.getPointerTy(DAG.getDataLayout());

    assert(WideVT.isRound() &&
           "Could not handle the sophisticated case when the widest integer is"
           " not power of 2.");
    assert(WideVT.bitsGE(SrcEltVT) &&
           "Type is not legalized?");

    unsigned WideBytes = WideVT.getStoreSize();
    unsigned Offset = 0;
    unsigned RemainingBytes = SrcVT.getStoreSize();
    SmallVector<SDValue, 8> LoadVals;

    while (RemainingBytes > 0) {
      SDValue ScalarLoad;
      unsigned LoadBytes = WideBytes;

      if (RemainingBytes >= LoadBytes) {
        ScalarLoad = DAG.getLoad(WideVT, dl, Chain, BasePTR,
                                 LD->getPointerInfo().getWithOffset(Offset),
                                 LD->isVolatile(), LD->isNonTemporal(),
                                 LD->isInvariant(),
                                 MinAlign(LD->getAlignment(), Offset),
                                 LD->getAAInfo());
      } else {
        EVT LoadVT = WideVT;
        while (RemainingBytes < LoadBytes) {
          LoadBytes >>= 1; // Reduce the load size by half.
          LoadVT = EVT::getIntegerVT(*DAG.getContext(), LoadBytes << 3);
        }
        ScalarLoad = DAG.getExtLoad(ISD::EXTLOAD, dl, WideVT, Chain, BasePTR,
                                    LD->getPointerInfo().getWithOffset(Offset),
                                    LoadVT, LD->isVolatile(),
                                    LD->isNonTemporal(), LD->isInvariant(),
                                    MinAlign(LD->getAlignment(), Offset),
                                    LD->getAAInfo());
      }

      RemainingBytes -= LoadBytes;
      Offset += LoadBytes;
      BasePTR = DAG.getNode(ISD::ADD, dl, BasePTR.getValueType(), BasePTR,
                            DAG.getConstant(LoadBytes, dl,
                                            BasePTR.getValueType()));

      LoadVals.push_back(ScalarLoad.getValue(0));
      LoadChains.push_back(ScalarLoad.getValue(1));
    }

    // Extract bits, pack and extend/trunc them into destination type.
    unsigned SrcEltBits = SrcEltVT.getSizeInBits();
    SDValue SrcEltBitMask = DAG.getConstant((1U << SrcEltBits) - 1, dl, WideVT);

    unsigned BitOffset = 0;
    unsigned WideIdx = 0;
    unsigned WideBits = WideVT.getSizeInBits();

    for (unsigned Idx = 0; Idx != NumElem; ++Idx) {
      SDValue Lo, Hi, ShAmt;

      if (BitOffset < WideBits) {
        ShAmt = DAG.getConstant(
            BitOffset, dl, TLI.getShiftAmountTy(WideVT, DAG.getDataLayout()));
        Lo = DAG.getNode(ISD::SRL, dl, WideVT, LoadVals[WideIdx], ShAmt);
        Lo = DAG.getNode(ISD::AND, dl, WideVT, Lo, SrcEltBitMask);
      }

      BitOffset += SrcEltBits;
      if (BitOffset >= WideBits) {
        WideIdx++;
        BitOffset -= WideBits;
        if (BitOffset > 0) {
          ShAmt = DAG.getConstant(
              SrcEltBits - BitOffset, dl,
              TLI.getShiftAmountTy(WideVT, DAG.getDataLayout()));
          Hi = DAG.getNode(ISD::SHL, dl, WideVT, LoadVals[WideIdx], ShAmt);
          Hi = DAG.getNode(ISD::AND, dl, WideVT, Hi, SrcEltBitMask);
        }
      }

      if (Hi.getNode())
        Lo = DAG.getNode(ISD::OR, dl, WideVT, Lo, Hi);

      switch (ExtType) {
      default: llvm_unreachable("Unknown extended-load op!");
      case ISD::EXTLOAD:
        Lo = DAG.getAnyExtOrTrunc(Lo, dl, DstEltVT);
        break;
      case ISD::ZEXTLOAD:
        Lo = DAG.getZExtOrTrunc(Lo, dl, DstEltVT);
        break;
      case ISD::SEXTLOAD:
        ShAmt =
            DAG.getConstant(WideBits - SrcEltBits, dl,
                            TLI.getShiftAmountTy(WideVT, DAG.getDataLayout()));
        Lo = DAG.getNode(ISD::SHL, dl, WideVT, Lo, ShAmt);
        Lo = DAG.getNode(ISD::SRA, dl, WideVT, Lo, ShAmt);
        Lo = DAG.getSExtOrTrunc(Lo, dl, DstEltVT);
        break;
      }
      Vals.push_back(Lo);
    }
  } else {
    unsigned Stride = SrcVT.getScalarType().getSizeInBits()/8;

    for (unsigned Idx=0; Idx<NumElem; Idx++) {
      SDValue ScalarLoad = DAG.getExtLoad(ExtType, dl,
                Op.getNode()->getValueType(0).getScalarType(),
                Chain, BasePTR, LD->getPointerInfo().getWithOffset(Idx * Stride),
                SrcVT.getScalarType(),
                LD->isVolatile(), LD->isNonTemporal(), LD->isInvariant(),
                MinAlign(LD->getAlignment(), Idx * Stride), LD->getAAInfo());

      BasePTR = DAG.getNode(ISD::ADD, dl, BasePTR.getValueType(), BasePTR,
                         DAG.getConstant(Stride, dl, BasePTR.getValueType()));

      Vals.push_back(ScalarLoad.getValue(0));
      LoadChains.push_back(ScalarLoad.getValue(1));
    }
  }

  SDValue NewChain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, LoadChains);
  SDValue Value = DAG.getNode(ISD::BUILD_VECTOR, dl,
                              Op.getNode()->getValueType(0), Vals);

  AddLegalizedOperand(Op.getValue(0), Value);
  AddLegalizedOperand(Op.getValue(1), NewChain);

  return (Op.getResNo() ? NewChain : Value);
}

SDValue VectorLegalizer::ExpandStore(SDValue Op) {
  SDLoc dl(Op);
  StoreSDNode *ST = cast<StoreSDNode>(Op.getNode());
  SDValue Chain = ST->getChain();
  SDValue BasePTR = ST->getBasePtr();
  SDValue Value = ST->getValue();
  EVT StVT = ST->getMemoryVT();

  unsigned Alignment = ST->getAlignment();
  bool isVolatile = ST->isVolatile();
  bool isNonTemporal = ST->isNonTemporal();
  AAMDNodes AAInfo = ST->getAAInfo();

  unsigned NumElem = StVT.getVectorNumElements();
  // The type of the data we want to save
  EVT RegVT = Value.getValueType();
  EVT RegSclVT = RegVT.getScalarType();
  // The type of data as saved in memory.
  EVT MemSclVT = StVT.getScalarType();

  // Cast floats into integers
  unsigned ScalarSize = MemSclVT.getSizeInBits();

  // Round odd types to the next pow of two.
  if (!isPowerOf2_32(ScalarSize))
    ScalarSize = NextPowerOf2(ScalarSize);

  // Store Stride in bytes
  unsigned Stride = ScalarSize/8;
  // Extract each of the elements from the original vector
  // and save them into memory individually.
  SmallVector<SDValue, 8> Stores;
  for (unsigned Idx = 0; Idx < NumElem; Idx++) {
    SDValue Ex = DAG.getNode(
        ISD::EXTRACT_VECTOR_ELT, dl, RegSclVT, Value,
        DAG.getConstant(Idx, dl, TLI.getVectorIdxTy(DAG.getDataLayout())));

    // This scalar TruncStore may be illegal, but we legalize it later.
    SDValue Store = DAG.getTruncStore(Chain, dl, Ex, BasePTR,
               ST->getPointerInfo().getWithOffset(Idx*Stride), MemSclVT,
               isVolatile, isNonTemporal, MinAlign(Alignment, Idx*Stride),
               AAInfo);

    BasePTR = DAG.getNode(ISD::ADD, dl, BasePTR.getValueType(), BasePTR,
                          DAG.getConstant(Stride, dl, BasePTR.getValueType()));

    Stores.push_back(Store);
  }
  SDValue TF =  DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Stores);
  AddLegalizedOperand(Op, TF);
  return TF;
}

SDValue VectorLegalizer::Expand(SDValue Op) {
  switch (Op->getOpcode()) {
  case ISD::SIGN_EXTEND_INREG:
    return ExpandSEXTINREG(Op);
  case ISD::ANY_EXTEND_VECTOR_INREG:
    return ExpandANY_EXTEND_VECTOR_INREG(Op);
  case ISD::SIGN_EXTEND_VECTOR_INREG:
    return ExpandSIGN_EXTEND_VECTOR_INREG(Op);
  case ISD::ZERO_EXTEND_VECTOR_INREG:
    return ExpandZERO_EXTEND_VECTOR_INREG(Op);
  case ISD::BSWAP:
    return ExpandBSWAP(Op);
  case ISD::VSELECT:
    return ExpandVSELECT(Op);
  case ISD::SELECT:
    return ExpandSELECT(Op);
  case ISD::UINT_TO_FP:
    return ExpandUINT_TO_FLOAT(Op);
  case ISD::FNEG:
    return ExpandFNEG(Op);
  case ISD::SETCC:
    return UnrollVSETCC(Op);
  default:
    return DAG.UnrollVectorOp(Op.getNode());
  }
}

SDValue VectorLegalizer::ExpandSELECT(SDValue Op) {
  // Lower a select instruction where the condition is a scalar and the
  // operands are vectors. Lower this select to VSELECT and implement it
  // using XOR AND OR. The selector bit is broadcasted.
  EVT VT = Op.getValueType();
  SDLoc DL(Op);

  SDValue Mask = Op.getOperand(0);
  SDValue Op1 = Op.getOperand(1);
  SDValue Op2 = Op.getOperand(2);

  assert(VT.isVector() && !Mask.getValueType().isVector()
         && Op1.getValueType() == Op2.getValueType() && "Invalid type");

  unsigned NumElem = VT.getVectorNumElements();

  // If we can't even use the basic vector operations of
  // AND,OR,XOR, we will have to scalarize the op.
  // Notice that the operation may be 'promoted' which means that it is
  // 'bitcasted' to another type which is handled.
  // Also, we need to be able to construct a splat vector using BUILD_VECTOR.
  if (TLI.getOperationAction(ISD::AND, VT) == TargetLowering::Expand ||
      TLI.getOperationAction(ISD::XOR, VT) == TargetLowering::Expand ||
      TLI.getOperationAction(ISD::OR,  VT) == TargetLowering::Expand ||
      TLI.getOperationAction(ISD::BUILD_VECTOR,  VT) == TargetLowering::Expand)
    return DAG.UnrollVectorOp(Op.getNode());

  // Generate a mask operand.
  EVT MaskTy = VT.changeVectorElementTypeToInteger();

  // What is the size of each element in the vector mask.
  EVT BitTy = MaskTy.getScalarType();

  Mask = DAG.getSelect(DL, BitTy, Mask,
          DAG.getConstant(APInt::getAllOnesValue(BitTy.getSizeInBits()), DL,
                          BitTy),
          DAG.getConstant(0, DL, BitTy));

  // Broadcast the mask so that the entire vector is all-one or all zero.
  SmallVector<SDValue, 8> Ops(NumElem, Mask);
  Mask = DAG.getNode(ISD::BUILD_VECTOR, DL, MaskTy, Ops);

  // Bitcast the operands to be the same type as the mask.
  // This is needed when we select between FP types because
  // the mask is a vector of integers.
  Op1 = DAG.getNode(ISD::BITCAST, DL, MaskTy, Op1);
  Op2 = DAG.getNode(ISD::BITCAST, DL, MaskTy, Op2);

  SDValue AllOnes = DAG.getConstant(
            APInt::getAllOnesValue(BitTy.getSizeInBits()), DL, MaskTy);
  SDValue NotMask = DAG.getNode(ISD::XOR, DL, MaskTy, Mask, AllOnes);

  Op1 = DAG.getNode(ISD::AND, DL, MaskTy, Op1, Mask);
  Op2 = DAG.getNode(ISD::AND, DL, MaskTy, Op2, NotMask);
  SDValue Val = DAG.getNode(ISD::OR, DL, MaskTy, Op1, Op2);
  return DAG.getNode(ISD::BITCAST, DL, Op.getValueType(), Val);
}

SDValue VectorLegalizer::ExpandSEXTINREG(SDValue Op) {
  EVT VT = Op.getValueType();

  // Make sure that the SRA and SHL instructions are available.
  if (TLI.getOperationAction(ISD::SRA, VT) == TargetLowering::Expand ||
      TLI.getOperationAction(ISD::SHL, VT) == TargetLowering::Expand)
    return DAG.UnrollVectorOp(Op.getNode());

  SDLoc DL(Op);
  EVT OrigTy = cast<VTSDNode>(Op->getOperand(1))->getVT();

  unsigned BW = VT.getScalarType().getSizeInBits();
  unsigned OrigBW = OrigTy.getScalarType().getSizeInBits();
  SDValue ShiftSz = DAG.getConstant(BW - OrigBW, DL, VT);

  Op = Op.getOperand(0);
  Op =   DAG.getNode(ISD::SHL, DL, VT, Op, ShiftSz);
  return DAG.getNode(ISD::SRA, DL, VT, Op, ShiftSz);
}

// Generically expand a vector anyext in register to a shuffle of the relevant
// lanes into the appropriate locations, with other lanes left undef.
SDValue VectorLegalizer::ExpandANY_EXTEND_VECTOR_INREG(SDValue Op) {
  SDLoc DL(Op);
  EVT VT = Op.getValueType();
  int NumElements = VT.getVectorNumElements();
  SDValue Src = Op.getOperand(0);
  EVT SrcVT = Src.getValueType();
  int NumSrcElements = SrcVT.getVectorNumElements();

  // Build a base mask of undef shuffles.
  SmallVector<int, 16> ShuffleMask;
  ShuffleMask.resize(NumSrcElements, -1);

  // Place the extended lanes into the correct locations.
  int ExtLaneScale = NumSrcElements / NumElements;
  int EndianOffset = DAG.getDataLayout().isBigEndian() ? ExtLaneScale - 1 : 0;
  for (int i = 0; i < NumElements; ++i)
    ShuffleMask[i * ExtLaneScale + EndianOffset] = i;

  return DAG.getNode(
      ISD::BITCAST, DL, VT,
      DAG.getVectorShuffle(SrcVT, DL, Src, DAG.getUNDEF(SrcVT), ShuffleMask));
}

SDValue VectorLegalizer::ExpandSIGN_EXTEND_VECTOR_INREG(SDValue Op) {
  SDLoc DL(Op);
  EVT VT = Op.getValueType();
  SDValue Src = Op.getOperand(0);
  EVT SrcVT = Src.getValueType();

  // First build an any-extend node which can be legalized above when we
  // recurse through it.
  Op = DAG.getAnyExtendVectorInReg(Src, DL, VT);

  // Now we need sign extend. Do this by shifting the elements. Even if these
  // aren't legal operations, they have a better chance of being legalized
  // without full scalarization than the sign extension does.
  unsigned EltWidth = VT.getVectorElementType().getSizeInBits();
  unsigned SrcEltWidth = SrcVT.getVectorElementType().getSizeInBits();
  SDValue ShiftAmount = DAG.getConstant(EltWidth - SrcEltWidth, DL, VT);
  return DAG.getNode(ISD::SRA, DL, VT,
                     DAG.getNode(ISD::SHL, DL, VT, Op, ShiftAmount),
                     ShiftAmount);
}

// Generically expand a vector zext in register to a shuffle of the relevant
// lanes into the appropriate locations, a blend of zero into the high bits,
// and a bitcast to the wider element type.
SDValue VectorLegalizer::ExpandZERO_EXTEND_VECTOR_INREG(SDValue Op) {
  SDLoc DL(Op);
  EVT VT = Op.getValueType();
  int NumElements = VT.getVectorNumElements();
  SDValue Src = Op.getOperand(0);
  EVT SrcVT = Src.getValueType();
  int NumSrcElements = SrcVT.getVectorNumElements();

  // Build up a zero vector to blend into this one.
  EVT SrcScalarVT = SrcVT.getScalarType();
  SDValue ScalarZero = DAG.getTargetConstant(0, DL, SrcScalarVT);
  SmallVector<SDValue, 4> BuildVectorOperands(NumSrcElements, ScalarZero);
  SDValue Zero = DAG.getNode(ISD::BUILD_VECTOR, DL, SrcVT, BuildVectorOperands);

  // Shuffle the incoming lanes into the correct position, and pull all other
  // lanes from the zero vector.
  SmallVector<int, 16> ShuffleMask;
  ShuffleMask.reserve(NumSrcElements);
  for (int i = 0; i < NumSrcElements; ++i)
    ShuffleMask.push_back(i);

  int ExtLaneScale = NumSrcElements / NumElements;
  int EndianOffset = DAG.getDataLayout().isBigEndian() ? ExtLaneScale - 1 : 0;
  for (int i = 0; i < NumElements; ++i)
    ShuffleMask[i * ExtLaneScale + EndianOffset] = NumSrcElements + i;

  return DAG.getNode(ISD::BITCAST, DL, VT,
                     DAG.getVectorShuffle(SrcVT, DL, Zero, Src, ShuffleMask));
}

SDValue VectorLegalizer::ExpandBSWAP(SDValue Op) {
  EVT VT = Op.getValueType();

  // Generate a byte wise shuffle mask for the BSWAP.
  SmallVector<int, 16> ShuffleMask;
  int ScalarSizeInBytes = VT.getScalarSizeInBits() / 8;
  for (int I = 0, E = VT.getVectorNumElements(); I != E; ++I)
    for (int J = ScalarSizeInBytes - 1; J >= 0; --J)
      ShuffleMask.push_back((I * ScalarSizeInBytes) + J);

  EVT ByteVT = EVT::getVectorVT(*DAG.getContext(), MVT::i8, ShuffleMask.size());

  // Only emit a shuffle if the mask is legal.
  if (!TLI.isShuffleMaskLegal(ShuffleMask, ByteVT))
    return DAG.UnrollVectorOp(Op.getNode());

  SDLoc DL(Op);
  Op = DAG.getNode(ISD::BITCAST, DL, ByteVT, Op.getOperand(0));
  Op = DAG.getVectorShuffle(ByteVT, DL, Op, DAG.getUNDEF(ByteVT),
                            ShuffleMask.data());
  return DAG.getNode(ISD::BITCAST, DL, VT, Op);
}

SDValue VectorLegalizer::ExpandVSELECT(SDValue Op) {
  // Implement VSELECT in terms of XOR, AND, OR
  // on platforms which do not support blend natively.
  SDLoc DL(Op);

  SDValue Mask = Op.getOperand(0);
  SDValue Op1 = Op.getOperand(1);
  SDValue Op2 = Op.getOperand(2);

  EVT VT = Mask.getValueType();

  // If we can't even use the basic vector operations of
  // AND,OR,XOR, we will have to scalarize the op.
  // Notice that the operation may be 'promoted' which means that it is
  // 'bitcasted' to another type which is handled.
  // This operation also isn't safe with AND, OR, XOR when the boolean
  // type is 0/1 as we need an all ones vector constant to mask with.
  // FIXME: Sign extend 1 to all ones if thats legal on the target.
  if (TLI.getOperationAction(ISD::AND, VT) == TargetLowering::Expand ||
      TLI.getOperationAction(ISD::XOR, VT) == TargetLowering::Expand ||
      TLI.getOperationAction(ISD::OR, VT) == TargetLowering::Expand ||
      TLI.getBooleanContents(Op1.getValueType()) !=
          TargetLowering::ZeroOrNegativeOneBooleanContent)
    return DAG.UnrollVectorOp(Op.getNode());

  // If the mask and the type are different sizes, unroll the vector op. This
  // can occur when getSetCCResultType returns something that is different in
  // size from the operand types. For example, v4i8 = select v4i32, v4i8, v4i8.
  if (VT.getSizeInBits() != Op1.getValueType().getSizeInBits())
    return DAG.UnrollVectorOp(Op.getNode());

  // Bitcast the operands to be the same type as the mask.
  // This is needed when we select between FP types because
  // the mask is a vector of integers.
  Op1 = DAG.getNode(ISD::BITCAST, DL, VT, Op1);
  Op2 = DAG.getNode(ISD::BITCAST, DL, VT, Op2);

  SDValue AllOnes = DAG.getConstant(
    APInt::getAllOnesValue(VT.getScalarType().getSizeInBits()), DL, VT);
  SDValue NotMask = DAG.getNode(ISD::XOR, DL, VT, Mask, AllOnes);

  Op1 = DAG.getNode(ISD::AND, DL, VT, Op1, Mask);
  Op2 = DAG.getNode(ISD::AND, DL, VT, Op2, NotMask);
  SDValue Val = DAG.getNode(ISD::OR, DL, VT, Op1, Op2);
  return DAG.getNode(ISD::BITCAST, DL, Op.getValueType(), Val);
}

SDValue VectorLegalizer::ExpandUINT_TO_FLOAT(SDValue Op) {
  EVT VT = Op.getOperand(0).getValueType();
  SDLoc DL(Op);

  // Make sure that the SINT_TO_FP and SRL instructions are available.
  if (TLI.getOperationAction(ISD::SINT_TO_FP, VT) == TargetLowering::Expand ||
      TLI.getOperationAction(ISD::SRL,        VT) == TargetLowering::Expand)
    return DAG.UnrollVectorOp(Op.getNode());

 EVT SVT = VT.getScalarType();
  assert((SVT.getSizeInBits() == 64 || SVT.getSizeInBits() == 32) &&
      "Elements in vector-UINT_TO_FP must be 32 or 64 bits wide");

  unsigned BW = SVT.getSizeInBits();
  SDValue HalfWord = DAG.getConstant(BW/2, DL, VT);

  // Constants to clear the upper part of the word.
  // Notice that we can also use SHL+SHR, but using a constant is slightly
  // faster on x86.
  uint64_t HWMask = (SVT.getSizeInBits()==64)?0x00000000FFFFFFFF:0x0000FFFF;
  SDValue HalfWordMask = DAG.getConstant(HWMask, DL, VT);

  // Two to the power of half-word-size.
  SDValue TWOHW = DAG.getConstantFP((((uint64_t)1)<<(BW/2)), DL, Op.getValueType()); // HLSL Change: do the 64-bit conversion before shift not after

  // Clear upper part of LO, lower HI
  SDValue HI = DAG.getNode(ISD::SRL, DL, VT, Op.getOperand(0), HalfWord);
  SDValue LO = DAG.getNode(ISD::AND, DL, VT, Op.getOperand(0), HalfWordMask);

  // Convert hi and lo to floats
  // Convert the hi part back to the upper values
  SDValue fHI = DAG.getNode(ISD::SINT_TO_FP, DL, Op.getValueType(), HI);
          fHI = DAG.getNode(ISD::FMUL, DL, Op.getValueType(), fHI, TWOHW);
  SDValue fLO = DAG.getNode(ISD::SINT_TO_FP, DL, Op.getValueType(), LO);

  // Add the two halves
  return DAG.getNode(ISD::FADD, DL, Op.getValueType(), fHI, fLO);
}


SDValue VectorLegalizer::ExpandFNEG(SDValue Op) {
  if (TLI.isOperationLegalOrCustom(ISD::FSUB, Op.getValueType())) {
    SDLoc DL(Op);
    SDValue Zero = DAG.getConstantFP(-0.0, DL, Op.getValueType());
    return DAG.getNode(ISD::FSUB, DL, Op.getValueType(),
                       Zero, Op.getOperand(0));
  }
  return DAG.UnrollVectorOp(Op.getNode());
}

SDValue VectorLegalizer::UnrollVSETCC(SDValue Op) {
  EVT VT = Op.getValueType();
  unsigned NumElems = VT.getVectorNumElements();
  EVT EltVT = VT.getVectorElementType();
  SDValue LHS = Op.getOperand(0), RHS = Op.getOperand(1), CC = Op.getOperand(2);
  EVT TmpEltVT = LHS.getValueType().getVectorElementType();
  SDLoc dl(Op);
  SmallVector<SDValue, 8> Ops(NumElems);
  for (unsigned i = 0; i < NumElems; ++i) {
    SDValue LHSElem = DAG.getNode(
        ISD::EXTRACT_VECTOR_ELT, dl, TmpEltVT, LHS,
        DAG.getConstant(i, dl, TLI.getVectorIdxTy(DAG.getDataLayout())));
    SDValue RHSElem = DAG.getNode(
        ISD::EXTRACT_VECTOR_ELT, dl, TmpEltVT, RHS,
        DAG.getConstant(i, dl, TLI.getVectorIdxTy(DAG.getDataLayout())));
    Ops[i] = DAG.getNode(ISD::SETCC, dl,
                         TLI.getSetCCResultType(DAG.getDataLayout(),
                                                *DAG.getContext(), TmpEltVT),
                         LHSElem, RHSElem, CC);
    Ops[i] = DAG.getSelect(dl, EltVT, Ops[i],
                           DAG.getConstant(APInt::getAllOnesValue
                                           (EltVT.getSizeInBits()), dl, EltVT),
                           DAG.getConstant(0, dl, EltVT));
  }
  return DAG.getNode(ISD::BUILD_VECTOR, dl, VT, Ops);
}

}

bool SelectionDAG::LegalizeVectors() {
  return VectorLegalizer(*this).Run();
}
