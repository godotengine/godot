//===-- TargetLowering.cpp - Implement the TargetLowering class -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements the TargetLowering class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetLowering.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include <cctype>
using namespace llvm;

/// NOTE: The TargetMachine owns TLOF.
TargetLowering::TargetLowering(const TargetMachine &tm)
  : TargetLoweringBase(tm) {}

const char *TargetLowering::getTargetNodeName(unsigned Opcode) const {
  return nullptr;
}

/// Check whether a given call node is in tail position within its function. If
/// so, it sets Chain to the input chain of the tail call.
bool TargetLowering::isInTailCallPosition(SelectionDAG &DAG, SDNode *Node,
                                          SDValue &Chain) const {
  const Function *F = DAG.getMachineFunction().getFunction();

  // Conservatively require the attributes of the call to match those of
  // the return. Ignore noalias because it doesn't affect the call sequence.
  AttributeSet CallerAttrs = F->getAttributes();
  if (AttrBuilder(CallerAttrs, AttributeSet::ReturnIndex)
      .removeAttribute(Attribute::NoAlias).hasAttributes())
    return false;

  // It's not safe to eliminate the sign / zero extension of the return value.
  if (CallerAttrs.hasAttribute(AttributeSet::ReturnIndex, Attribute::ZExt) ||
      CallerAttrs.hasAttribute(AttributeSet::ReturnIndex, Attribute::SExt))
    return false;

  // Check if the only use is a function return node.
  return isUsedByReturnOnly(Node, Chain);
}

/// \brief Set CallLoweringInfo attribute flags based on a call instruction
/// and called function attributes.
void TargetLowering::ArgListEntry::setAttributes(ImmutableCallSite *CS,
                                                 unsigned AttrIdx) {
  isSExt     = CS->paramHasAttr(AttrIdx, Attribute::SExt);
  isZExt     = CS->paramHasAttr(AttrIdx, Attribute::ZExt);
  isInReg    = CS->paramHasAttr(AttrIdx, Attribute::InReg);
  isSRet     = CS->paramHasAttr(AttrIdx, Attribute::StructRet);
  isNest     = CS->paramHasAttr(AttrIdx, Attribute::Nest);
  isByVal    = CS->paramHasAttr(AttrIdx, Attribute::ByVal);
  isInAlloca = CS->paramHasAttr(AttrIdx, Attribute::InAlloca);
  isReturned = CS->paramHasAttr(AttrIdx, Attribute::Returned);
  Alignment  = CS->getParamAlignment(AttrIdx);
}

/// Generate a libcall taking the given operands as arguments and returning a
/// result of type RetVT.
std::pair<SDValue, SDValue>
TargetLowering::makeLibCall(SelectionDAG &DAG,
                            RTLIB::Libcall LC, EVT RetVT,
                            const SDValue *Ops, unsigned NumOps,
                            bool isSigned, SDLoc dl,
                            bool doesNotReturn,
                            bool isReturnValueUsed) const {
  TargetLowering::ArgListTy Args;
  Args.reserve(NumOps);

  TargetLowering::ArgListEntry Entry;
  for (unsigned i = 0; i != NumOps; ++i) {
    Entry.Node = Ops[i];
    Entry.Ty = Entry.Node.getValueType().getTypeForEVT(*DAG.getContext());
    Entry.isSExt = shouldSignExtendTypeInLibCall(Ops[i].getValueType(), isSigned);
    Entry.isZExt = !shouldSignExtendTypeInLibCall(Ops[i].getValueType(), isSigned);
    Args.push_back(Entry);
  }
  if (LC == RTLIB::UNKNOWN_LIBCALL)
    report_fatal_error("Unsupported library call operation!");
  SDValue Callee = DAG.getExternalSymbol(getLibcallName(LC),
                                         getPointerTy(DAG.getDataLayout()));

  Type *RetTy = RetVT.getTypeForEVT(*DAG.getContext());
  TargetLowering::CallLoweringInfo CLI(DAG);
  bool signExtend = shouldSignExtendTypeInLibCall(RetVT, isSigned);
  CLI.setDebugLoc(dl).setChain(DAG.getEntryNode())
    .setCallee(getLibcallCallingConv(LC), RetTy, Callee, std::move(Args), 0)
    .setNoReturn(doesNotReturn).setDiscardResult(!isReturnValueUsed)
    .setSExtResult(signExtend).setZExtResult(!signExtend);
  return LowerCallTo(CLI);
}


/// SoftenSetCCOperands - Soften the operands of a comparison.  This code is
/// shared among BR_CC, SELECT_CC, and SETCC handlers.
void TargetLowering::softenSetCCOperands(SelectionDAG &DAG, EVT VT,
                                         SDValue &NewLHS, SDValue &NewRHS,
                                         ISD::CondCode &CCCode,
                                         SDLoc dl) const {
  assert((VT == MVT::f32 || VT == MVT::f64 || VT == MVT::f128)
         && "Unsupported setcc type!");

  // Expand into one or more soft-fp libcall(s).
  RTLIB::Libcall LC1 = RTLIB::UNKNOWN_LIBCALL, LC2 = RTLIB::UNKNOWN_LIBCALL;
  switch (CCCode) {
  case ISD::SETEQ:
  case ISD::SETOEQ:
    LC1 = (VT == MVT::f32) ? RTLIB::OEQ_F32 :
          (VT == MVT::f64) ? RTLIB::OEQ_F64 : RTLIB::OEQ_F128;
    break;
  case ISD::SETNE:
  case ISD::SETUNE:
    LC1 = (VT == MVT::f32) ? RTLIB::UNE_F32 :
          (VT == MVT::f64) ? RTLIB::UNE_F64 : RTLIB::UNE_F128;
    break;
  case ISD::SETGE:
  case ISD::SETOGE:
    LC1 = (VT == MVT::f32) ? RTLIB::OGE_F32 :
          (VT == MVT::f64) ? RTLIB::OGE_F64 : RTLIB::OGE_F128;
    break;
  case ISD::SETLT:
  case ISD::SETOLT:
    LC1 = (VT == MVT::f32) ? RTLIB::OLT_F32 :
          (VT == MVT::f64) ? RTLIB::OLT_F64 : RTLIB::OLT_F128;
    break;
  case ISD::SETLE:
  case ISD::SETOLE:
    LC1 = (VT == MVT::f32) ? RTLIB::OLE_F32 :
          (VT == MVT::f64) ? RTLIB::OLE_F64 : RTLIB::OLE_F128;
    break;
  case ISD::SETGT:
  case ISD::SETOGT:
    LC1 = (VT == MVT::f32) ? RTLIB::OGT_F32 :
          (VT == MVT::f64) ? RTLIB::OGT_F64 : RTLIB::OGT_F128;
    break;
  case ISD::SETUO:
    LC1 = (VT == MVT::f32) ? RTLIB::UO_F32 :
          (VT == MVT::f64) ? RTLIB::UO_F64 : RTLIB::UO_F128;
    break;
  case ISD::SETO:
    LC1 = (VT == MVT::f32) ? RTLIB::O_F32 :
          (VT == MVT::f64) ? RTLIB::O_F64 : RTLIB::O_F128;
    break;
  default:
    LC1 = (VT == MVT::f32) ? RTLIB::UO_F32 :
          (VT == MVT::f64) ? RTLIB::UO_F64 : RTLIB::UO_F128;
    switch (CCCode) {
    case ISD::SETONE:
      // SETONE = SETOLT | SETOGT
      LC1 = (VT == MVT::f32) ? RTLIB::OLT_F32 :
            (VT == MVT::f64) ? RTLIB::OLT_F64 : RTLIB::OLT_F128;
      // Fallthrough
    case ISD::SETUGT:
      LC2 = (VT == MVT::f32) ? RTLIB::OGT_F32 :
            (VT == MVT::f64) ? RTLIB::OGT_F64 : RTLIB::OGT_F128;
      break;
    case ISD::SETUGE:
      LC2 = (VT == MVT::f32) ? RTLIB::OGE_F32 :
            (VT == MVT::f64) ? RTLIB::OGE_F64 : RTLIB::OGE_F128;
      break;
    case ISD::SETULT:
      LC2 = (VT == MVT::f32) ? RTLIB::OLT_F32 :
            (VT == MVT::f64) ? RTLIB::OLT_F64 : RTLIB::OLT_F128;
      break;
    case ISD::SETULE:
      LC2 = (VT == MVT::f32) ? RTLIB::OLE_F32 :
            (VT == MVT::f64) ? RTLIB::OLE_F64 : RTLIB::OLE_F128;
      break;
    case ISD::SETUEQ:
      LC2 = (VT == MVT::f32) ? RTLIB::OEQ_F32 :
            (VT == MVT::f64) ? RTLIB::OEQ_F64 : RTLIB::OEQ_F128;
      break;
    default: llvm_unreachable("Do not know how to soften this setcc!");
    }
  }

  // Use the target specific return value for comparions lib calls.
  EVT RetVT = getCmpLibcallReturnType();
  SDValue Ops[2] = { NewLHS, NewRHS };
  NewLHS = makeLibCall(DAG, LC1, RetVT, Ops, 2, false/*sign irrelevant*/,
                       dl).first;
  NewRHS = DAG.getConstant(0, dl, RetVT);
  CCCode = getCmpLibcallCC(LC1);
  if (LC2 != RTLIB::UNKNOWN_LIBCALL) {
    SDValue Tmp = DAG.getNode(
        ISD::SETCC, dl,
        getSetCCResultType(DAG.getDataLayout(), *DAG.getContext(), RetVT),
        NewLHS, NewRHS, DAG.getCondCode(CCCode));
    NewLHS = makeLibCall(DAG, LC2, RetVT, Ops, 2, false/*sign irrelevant*/,
                         dl).first;
    NewLHS = DAG.getNode(
        ISD::SETCC, dl,
        getSetCCResultType(DAG.getDataLayout(), *DAG.getContext(), RetVT),
        NewLHS, NewRHS, DAG.getCondCode(getCmpLibcallCC(LC2)));
    NewLHS = DAG.getNode(ISD::OR, dl, Tmp.getValueType(), Tmp, NewLHS);
    NewRHS = SDValue();
  }
}

/// getJumpTableEncoding - Return the entry encoding for a jump table in the
/// current function.  The returned value is a member of the
/// MachineJumpTableInfo::JTEntryKind enum.
unsigned TargetLowering::getJumpTableEncoding() const {
  // In non-pic modes, just use the address of a block.
  if (getTargetMachine().getRelocationModel() != Reloc::PIC_)
    return MachineJumpTableInfo::EK_BlockAddress;

  // In PIC mode, if the target supports a GPRel32 directive, use it.
  if (getTargetMachine().getMCAsmInfo()->getGPRel32Directive() != nullptr)
    return MachineJumpTableInfo::EK_GPRel32BlockAddress;

  // Otherwise, use a label difference.
  return MachineJumpTableInfo::EK_LabelDifference32;
}

SDValue TargetLowering::getPICJumpTableRelocBase(SDValue Table,
                                                 SelectionDAG &DAG) const {
  // If our PIC model is GP relative, use the global offset table as the base.
  unsigned JTEncoding = getJumpTableEncoding();

  if ((JTEncoding == MachineJumpTableInfo::EK_GPRel64BlockAddress) ||
      (JTEncoding == MachineJumpTableInfo::EK_GPRel32BlockAddress))
    return DAG.getGLOBAL_OFFSET_TABLE(getPointerTy(DAG.getDataLayout()));

  return Table;
}

/// getPICJumpTableRelocBaseExpr - This returns the relocation base for the
/// given PIC jumptable, the same as getPICJumpTableRelocBase, but as an
/// MCExpr.
const MCExpr *
TargetLowering::getPICJumpTableRelocBaseExpr(const MachineFunction *MF,
                                             unsigned JTI,MCContext &Ctx) const{
  // The normal PIC reloc base is the label at the start of the jump table.
  return MCSymbolRefExpr::create(MF->getJTISymbol(JTI, Ctx), Ctx);
}

bool
TargetLowering::isOffsetFoldingLegal(const GlobalAddressSDNode *GA) const {
  // Assume that everything is safe in static mode.
  if (getTargetMachine().getRelocationModel() == Reloc::Static)
    return true;

  // In dynamic-no-pic mode, assume that known defined values are safe.
  if (getTargetMachine().getRelocationModel() == Reloc::DynamicNoPIC &&
      GA && GA->getGlobal()->isStrongDefinitionForLinker())
    return true;

  // Otherwise assume nothing is safe.
  return false;
}

//===----------------------------------------------------------------------===//
//  Optimization Methods
//===----------------------------------------------------------------------===//

/// ShrinkDemandedConstant - Check to see if the specified operand of the
/// specified instruction is a constant integer.  If so, check to see if there
/// are any bits set in the constant that are not demanded.  If so, shrink the
/// constant and return true.
bool TargetLowering::TargetLoweringOpt::ShrinkDemandedConstant(SDValue Op,
                                                        const APInt &Demanded) {
  SDLoc dl(Op);

  // FIXME: ISD::SELECT, ISD::SELECT_CC
  switch (Op.getOpcode()) {
  default: break;
  case ISD::XOR:
  case ISD::AND:
  case ISD::OR: {
    ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op.getOperand(1));
    if (!C) return false;

    if (Op.getOpcode() == ISD::XOR &&
        (C->getAPIntValue() | (~Demanded)).isAllOnesValue())
      return false;

    // if we can expand it to have all bits set, do it
    if (C->getAPIntValue().intersects(~Demanded)) {
      EVT VT = Op.getValueType();
      SDValue New = DAG.getNode(Op.getOpcode(), dl, VT, Op.getOperand(0),
                                DAG.getConstant(Demanded &
                                                C->getAPIntValue(),
                                                dl, VT));
      return CombineTo(Op, New);
    }

    break;
  }
  }

  return false;
}

/// ShrinkDemandedOp - Convert x+y to (VT)((SmallVT)x+(SmallVT)y) if the
/// casts are free.  This uses isZExtFree and ZERO_EXTEND for the widening
/// cast, but it could be generalized for targets with other types of
/// implicit widening casts.
bool
TargetLowering::TargetLoweringOpt::ShrinkDemandedOp(SDValue Op,
                                                    unsigned BitWidth,
                                                    const APInt &Demanded,
                                                    SDLoc dl) {
  assert(Op.getNumOperands() == 2 &&
         "ShrinkDemandedOp only supports binary operators!");
  assert(Op.getNode()->getNumValues() == 1 &&
         "ShrinkDemandedOp only supports nodes with one result!");

  // Early return, as this function cannot handle vector types.
  if (Op.getValueType().isVector())
    return false;

  // Don't do this if the node has another user, which may require the
  // full value.
  if (!Op.getNode()->hasOneUse())
    return false;

  // Search for the smallest integer type with free casts to and from
  // Op's type. For expedience, just check power-of-2 integer types.
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  unsigned DemandedSize = BitWidth - Demanded.countLeadingZeros();
  unsigned SmallVTBits = DemandedSize;
  if (!isPowerOf2_32(SmallVTBits))
    SmallVTBits = NextPowerOf2(SmallVTBits);
  for (; SmallVTBits < BitWidth; SmallVTBits = NextPowerOf2(SmallVTBits)) {
    EVT SmallVT = EVT::getIntegerVT(*DAG.getContext(), SmallVTBits);
    if (TLI.isTruncateFree(Op.getValueType(), SmallVT) &&
        TLI.isZExtFree(SmallVT, Op.getValueType())) {
      // We found a type with free casts.
      SDValue X = DAG.getNode(Op.getOpcode(), dl, SmallVT,
                              DAG.getNode(ISD::TRUNCATE, dl, SmallVT,
                                          Op.getNode()->getOperand(0)),
                              DAG.getNode(ISD::TRUNCATE, dl, SmallVT,
                                          Op.getNode()->getOperand(1)));
      bool NeedZext = DemandedSize > SmallVTBits;
      SDValue Z = DAG.getNode(NeedZext ? ISD::ZERO_EXTEND : ISD::ANY_EXTEND,
                              dl, Op.getValueType(), X);
      return CombineTo(Op, Z);
    }
  }
  return false;
}

/// SimplifyDemandedBits - Look at Op.  At this point, we know that only the
/// DemandedMask bits of the result of Op are ever used downstream.  If we can
/// use this information to simplify Op, create a new simplified DAG node and
/// return true, returning the original and new nodes in Old and New. Otherwise,
/// analyze the expression and return a mask of KnownOne and KnownZero bits for
/// the expression (used to simplify the caller).  The KnownZero/One bits may
/// only be accurate for those bits in the DemandedMask.
bool TargetLowering::SimplifyDemandedBits(SDValue Op,
                                          const APInt &DemandedMask,
                                          APInt &KnownZero,
                                          APInt &KnownOne,
                                          TargetLoweringOpt &TLO,
                                          unsigned Depth) const {
  unsigned BitWidth = DemandedMask.getBitWidth();
  assert(Op.getValueType().getScalarType().getSizeInBits() == BitWidth &&
         "Mask size mismatches value type size!");
  APInt NewMask = DemandedMask;
  SDLoc dl(Op);
  auto &DL = TLO.DAG.getDataLayout();

  // Don't know anything.
  KnownZero = KnownOne = APInt(BitWidth, 0);

  // Other users may use these bits.
  if (!Op.getNode()->hasOneUse()) {
    if (Depth != 0) {
      // If not at the root, Just compute the KnownZero/KnownOne bits to
      // simplify things downstream.
      TLO.DAG.computeKnownBits(Op, KnownZero, KnownOne, Depth);
      return false;
    }
    // If this is the root being simplified, allow it to have multiple uses,
    // just set the NewMask to all bits.
    NewMask = APInt::getAllOnesValue(BitWidth);
  } else if (DemandedMask == 0) {
    // Not demanding any bits from Op.
    if (Op.getOpcode() != ISD::UNDEF)
      return TLO.CombineTo(Op, TLO.DAG.getUNDEF(Op.getValueType()));
    return false;
  } else if (Depth == 6) {        // Limit search depth.
    return false;
  }

  APInt KnownZero2, KnownOne2, KnownZeroOut, KnownOneOut;
  switch (Op.getOpcode()) {
  case ISD::Constant:
    // We know all of the bits for a constant!
    KnownOne = cast<ConstantSDNode>(Op)->getAPIntValue();
    KnownZero = ~KnownOne;
    return false;   // Don't fall through, will infinitely loop.
  case ISD::AND:
    // If the RHS is a constant, check to see if the LHS would be zero without
    // using the bits from the RHS.  Below, we use knowledge about the RHS to
    // simplify the LHS, here we're using information from the LHS to simplify
    // the RHS.
    if (ConstantSDNode *RHSC = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      APInt LHSZero, LHSOne;
      // Do not increment Depth here; that can cause an infinite loop.
      TLO.DAG.computeKnownBits(Op.getOperand(0), LHSZero, LHSOne, Depth);
      // If the LHS already has zeros where RHSC does, this and is dead.
      if ((LHSZero & NewMask) == (~RHSC->getAPIntValue() & NewMask))
        return TLO.CombineTo(Op, Op.getOperand(0));
      // If any of the set bits in the RHS are known zero on the LHS, shrink
      // the constant.
      if (TLO.ShrinkDemandedConstant(Op, ~LHSZero & NewMask))
        return true;
    }

    if (SimplifyDemandedBits(Op.getOperand(1), NewMask, KnownZero,
                             KnownOne, TLO, Depth+1))
      return true;
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?");
    if (SimplifyDemandedBits(Op.getOperand(0), ~KnownZero & NewMask,
                             KnownZero2, KnownOne2, TLO, Depth+1))
      return true;
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?");

    // If all of the demanded bits are known one on one side, return the other.
    // These bits cannot contribute to the result of the 'and'.
    if ((NewMask & ~KnownZero2 & KnownOne) == (~KnownZero2 & NewMask))
      return TLO.CombineTo(Op, Op.getOperand(0));
    if ((NewMask & ~KnownZero & KnownOne2) == (~KnownZero & NewMask))
      return TLO.CombineTo(Op, Op.getOperand(1));
    // If all of the demanded bits in the inputs are known zeros, return zero.
    if ((NewMask & (KnownZero|KnownZero2)) == NewMask)
      return TLO.CombineTo(Op, TLO.DAG.getConstant(0, dl, Op.getValueType()));
    // If the RHS is a constant, see if we can simplify it.
    if (TLO.ShrinkDemandedConstant(Op, ~KnownZero2 & NewMask))
      return true;
    // If the operation can be done in a smaller type, do so.
    if (TLO.ShrinkDemandedOp(Op, BitWidth, NewMask, dl))
      return true;

    // Output known-1 bits are only known if set in both the LHS & RHS.
    KnownOne &= KnownOne2;
    // Output known-0 are known to be clear if zero in either the LHS | RHS.
    KnownZero |= KnownZero2;
    break;
  case ISD::OR:
    if (SimplifyDemandedBits(Op.getOperand(1), NewMask, KnownZero,
                             KnownOne, TLO, Depth+1))
      return true;
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?");
    if (SimplifyDemandedBits(Op.getOperand(0), ~KnownOne & NewMask,
                             KnownZero2, KnownOne2, TLO, Depth+1))
      return true;
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?");

    // If all of the demanded bits are known zero on one side, return the other.
    // These bits cannot contribute to the result of the 'or'.
    if ((NewMask & ~KnownOne2 & KnownZero) == (~KnownOne2 & NewMask))
      return TLO.CombineTo(Op, Op.getOperand(0));
    if ((NewMask & ~KnownOne & KnownZero2) == (~KnownOne & NewMask))
      return TLO.CombineTo(Op, Op.getOperand(1));
    // If all of the potentially set bits on one side are known to be set on
    // the other side, just use the 'other' side.
    if ((NewMask & ~KnownZero & KnownOne2) == (~KnownZero & NewMask))
      return TLO.CombineTo(Op, Op.getOperand(0));
    if ((NewMask & ~KnownZero2 & KnownOne) == (~KnownZero2 & NewMask))
      return TLO.CombineTo(Op, Op.getOperand(1));
    // If the RHS is a constant, see if we can simplify it.
    if (TLO.ShrinkDemandedConstant(Op, NewMask))
      return true;
    // If the operation can be done in a smaller type, do so.
    if (TLO.ShrinkDemandedOp(Op, BitWidth, NewMask, dl))
      return true;

    // Output known-0 bits are only known if clear in both the LHS & RHS.
    KnownZero &= KnownZero2;
    // Output known-1 are known to be set if set in either the LHS | RHS.
    KnownOne |= KnownOne2;
    break;
  case ISD::XOR:
    if (SimplifyDemandedBits(Op.getOperand(1), NewMask, KnownZero,
                             KnownOne, TLO, Depth+1))
      return true;
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?");
    if (SimplifyDemandedBits(Op.getOperand(0), NewMask, KnownZero2,
                             KnownOne2, TLO, Depth+1))
      return true;
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?");

    // If all of the demanded bits are known zero on one side, return the other.
    // These bits cannot contribute to the result of the 'xor'.
    if ((KnownZero & NewMask) == NewMask)
      return TLO.CombineTo(Op, Op.getOperand(0));
    if ((KnownZero2 & NewMask) == NewMask)
      return TLO.CombineTo(Op, Op.getOperand(1));
    // If the operation can be done in a smaller type, do so.
    if (TLO.ShrinkDemandedOp(Op, BitWidth, NewMask, dl))
      return true;

    // If all of the unknown bits are known to be zero on one side or the other
    // (but not both) turn this into an *inclusive* or.
    //    e.g. (A & C1)^(B & C2) -> (A & C1)|(B & C2) iff C1&C2 == 0
    if ((NewMask & ~KnownZero & ~KnownZero2) == 0)
      return TLO.CombineTo(Op, TLO.DAG.getNode(ISD::OR, dl, Op.getValueType(),
                                               Op.getOperand(0),
                                               Op.getOperand(1)));

    // Output known-0 bits are known if clear or set in both the LHS & RHS.
    KnownZeroOut = (KnownZero & KnownZero2) | (KnownOne & KnownOne2);
    // Output known-1 are known to be set if set in only one of the LHS, RHS.
    KnownOneOut = (KnownZero & KnownOne2) | (KnownOne & KnownZero2);

    // If all of the demanded bits on one side are known, and all of the set
    // bits on that side are also known to be set on the other side, turn this
    // into an AND, as we know the bits will be cleared.
    //    e.g. (X | C1) ^ C2 --> (X | C1) & ~C2 iff (C1&C2) == C2
    // NB: it is okay if more bits are known than are requested
    if ((NewMask & (KnownZero|KnownOne)) == NewMask) { // all known on one side
      if (KnownOne == KnownOne2) { // set bits are the same on both sides
        EVT VT = Op.getValueType();
        SDValue ANDC = TLO.DAG.getConstant(~KnownOne & NewMask, dl, VT);
        return TLO.CombineTo(Op, TLO.DAG.getNode(ISD::AND, dl, VT,
                                                 Op.getOperand(0), ANDC));
      }
    }

    // If the RHS is a constant, see if we can simplify it.
    // for XOR, we prefer to force bits to 1 if they will make a -1.
    // if we can't force bits, try to shrink constant
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      APInt Expanded = C->getAPIntValue() | (~NewMask);
      // if we can expand it to have all bits set, do it
      if (Expanded.isAllOnesValue()) {
        if (Expanded != C->getAPIntValue()) {
          EVT VT = Op.getValueType();
          SDValue New = TLO.DAG.getNode(Op.getOpcode(), dl,VT, Op.getOperand(0),
                                        TLO.DAG.getConstant(Expanded, dl, VT));
          return TLO.CombineTo(Op, New);
        }
        // if it already has all the bits set, nothing to change
        // but don't shrink either!
      } else if (TLO.ShrinkDemandedConstant(Op, NewMask)) {
        return true;
      }
    }

    KnownZero = KnownZeroOut;
    KnownOne  = KnownOneOut;
    break;
  case ISD::SELECT:
    if (SimplifyDemandedBits(Op.getOperand(2), NewMask, KnownZero,
                             KnownOne, TLO, Depth+1))
      return true;
    if (SimplifyDemandedBits(Op.getOperand(1), NewMask, KnownZero2,
                             KnownOne2, TLO, Depth+1))
      return true;
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?");
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?");

    // If the operands are constants, see if we can simplify them.
    if (TLO.ShrinkDemandedConstant(Op, NewMask))
      return true;

    // Only known if known in both the LHS and RHS.
    KnownOne &= KnownOne2;
    KnownZero &= KnownZero2;
    break;
  case ISD::SELECT_CC:
    if (SimplifyDemandedBits(Op.getOperand(3), NewMask, KnownZero,
                             KnownOne, TLO, Depth+1))
      return true;
    if (SimplifyDemandedBits(Op.getOperand(2), NewMask, KnownZero2,
                             KnownOne2, TLO, Depth+1))
      return true;
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?");
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?");

    // If the operands are constants, see if we can simplify them.
    if (TLO.ShrinkDemandedConstant(Op, NewMask))
      return true;

    // Only known if known in both the LHS and RHS.
    KnownOne &= KnownOne2;
    KnownZero &= KnownZero2;
    break;
  case ISD::SHL:
    if (ConstantSDNode *SA = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      unsigned ShAmt = SA->getZExtValue();
      SDValue InOp = Op.getOperand(0);

      // If the shift count is an invalid immediate, don't do anything.
      if (ShAmt >= BitWidth)
        break;

      // If this is ((X >>u C1) << ShAmt), see if we can simplify this into a
      // single shift.  We can do this if the bottom bits (which are shifted
      // out) are never demanded.
      if (InOp.getOpcode() == ISD::SRL &&
          isa<ConstantSDNode>(InOp.getOperand(1))) {
        if (ShAmt && (NewMask & APInt::getLowBitsSet(BitWidth, ShAmt)) == 0) {
          unsigned C1= cast<ConstantSDNode>(InOp.getOperand(1))->getZExtValue();
          unsigned Opc = ISD::SHL;
          int Diff = ShAmt-C1;
          if (Diff < 0) {
            Diff = -Diff;
            Opc = ISD::SRL;
          }

          SDValue NewSA =
            TLO.DAG.getConstant(Diff, dl, Op.getOperand(1).getValueType());
          EVT VT = Op.getValueType();
          return TLO.CombineTo(Op, TLO.DAG.getNode(Opc, dl, VT,
                                                   InOp.getOperand(0), NewSA));
        }
      }

      if (SimplifyDemandedBits(InOp, NewMask.lshr(ShAmt),
                               KnownZero, KnownOne, TLO, Depth+1))
        return true;

      // Convert (shl (anyext x, c)) to (anyext (shl x, c)) if the high bits
      // are not demanded. This will likely allow the anyext to be folded away.
      if (InOp.getNode()->getOpcode() == ISD::ANY_EXTEND) {
        SDValue InnerOp = InOp.getNode()->getOperand(0);
        EVT InnerVT = InnerOp.getValueType();
        unsigned InnerBits = InnerVT.getSizeInBits();
        if (ShAmt < InnerBits && NewMask.lshr(InnerBits) == 0 &&
            isTypeDesirableForOp(ISD::SHL, InnerVT)) {
          EVT ShTy = getShiftAmountTy(InnerVT, DL);
          if (!APInt(BitWidth, ShAmt).isIntN(ShTy.getSizeInBits()))
            ShTy = InnerVT;
          SDValue NarrowShl =
            TLO.DAG.getNode(ISD::SHL, dl, InnerVT, InnerOp,
                            TLO.DAG.getConstant(ShAmt, dl, ShTy));
          return
            TLO.CombineTo(Op,
                          TLO.DAG.getNode(ISD::ANY_EXTEND, dl, Op.getValueType(),
                                          NarrowShl));
        }
        // Repeat the SHL optimization above in cases where an extension
        // intervenes: (shl (anyext (shr x, c1)), c2) to
        // (shl (anyext x), c2-c1).  This requires that the bottom c1 bits
        // aren't demanded (as above) and that the shifted upper c1 bits of
        // x aren't demanded.
        if (InOp.hasOneUse() &&
            InnerOp.getOpcode() == ISD::SRL &&
            InnerOp.hasOneUse() &&
            isa<ConstantSDNode>(InnerOp.getOperand(1))) {
          uint64_t InnerShAmt = cast<ConstantSDNode>(InnerOp.getOperand(1))
            ->getZExtValue();
          if (InnerShAmt < ShAmt &&
              InnerShAmt < InnerBits &&
              NewMask.lshr(InnerBits - InnerShAmt + ShAmt) == 0 &&
              NewMask.trunc(ShAmt) == 0) {
            SDValue NewSA =
              TLO.DAG.getConstant(ShAmt - InnerShAmt, dl,
                                  Op.getOperand(1).getValueType());
            EVT VT = Op.getValueType();
            SDValue NewExt = TLO.DAG.getNode(ISD::ANY_EXTEND, dl, VT,
                                             InnerOp.getOperand(0));
            return TLO.CombineTo(Op, TLO.DAG.getNode(ISD::SHL, dl, VT,
                                                     NewExt, NewSA));
          }
        }
      }

      KnownZero <<= SA->getZExtValue();
      KnownOne  <<= SA->getZExtValue();
      // low bits known zero.
      KnownZero |= APInt::getLowBitsSet(BitWidth, SA->getZExtValue());
    }
    break;
  case ISD::SRL:
    if (ConstantSDNode *SA = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      EVT VT = Op.getValueType();
      unsigned ShAmt = SA->getZExtValue();
      unsigned VTSize = VT.getSizeInBits();
      SDValue InOp = Op.getOperand(0);

      // If the shift count is an invalid immediate, don't do anything.
      if (ShAmt >= BitWidth)
        break;

      APInt InDemandedMask = (NewMask << ShAmt);

      // If the shift is exact, then it does demand the low bits (and knows that
      // they are zero).
      if (cast<BinaryWithFlagsSDNode>(Op)->Flags.hasExact())
        InDemandedMask |= APInt::getLowBitsSet(BitWidth, ShAmt);

      // If this is ((X << C1) >>u ShAmt), see if we can simplify this into a
      // single shift.  We can do this if the top bits (which are shifted out)
      // are never demanded.
      if (InOp.getOpcode() == ISD::SHL &&
          isa<ConstantSDNode>(InOp.getOperand(1))) {
        if (ShAmt && (NewMask & APInt::getHighBitsSet(VTSize, ShAmt)) == 0) {
          unsigned C1= cast<ConstantSDNode>(InOp.getOperand(1))->getZExtValue();
          unsigned Opc = ISD::SRL;
          int Diff = ShAmt-C1;
          if (Diff < 0) {
            Diff = -Diff;
            Opc = ISD::SHL;
          }

          SDValue NewSA =
            TLO.DAG.getConstant(Diff, dl, Op.getOperand(1).getValueType());
          return TLO.CombineTo(Op, TLO.DAG.getNode(Opc, dl, VT,
                                                   InOp.getOperand(0), NewSA));
        }
      }

      // Compute the new bits that are at the top now.
      if (SimplifyDemandedBits(InOp, InDemandedMask,
                               KnownZero, KnownOne, TLO, Depth+1))
        return true;
      assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?");
      KnownZero = KnownZero.lshr(ShAmt);
      KnownOne  = KnownOne.lshr(ShAmt);

      APInt HighBits = APInt::getHighBitsSet(BitWidth, ShAmt);
      KnownZero |= HighBits;  // High bits known zero.
    }
    break;
  case ISD::SRA:
    // If this is an arithmetic shift right and only the low-bit is set, we can
    // always convert this into a logical shr, even if the shift amount is
    // variable.  The low bit of the shift cannot be an input sign bit unless
    // the shift amount is >= the size of the datatype, which is undefined.
    if (NewMask == 1)
      return TLO.CombineTo(Op,
                           TLO.DAG.getNode(ISD::SRL, dl, Op.getValueType(),
                                           Op.getOperand(0), Op.getOperand(1)));

    if (ConstantSDNode *SA = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      EVT VT = Op.getValueType();
      unsigned ShAmt = SA->getZExtValue();

      // If the shift count is an invalid immediate, don't do anything.
      if (ShAmt >= BitWidth)
        break;

      APInt InDemandedMask = (NewMask << ShAmt);

      // If the shift is exact, then it does demand the low bits (and knows that
      // they are zero).
      if (cast<BinaryWithFlagsSDNode>(Op)->Flags.hasExact())
        InDemandedMask |= APInt::getLowBitsSet(BitWidth, ShAmt);

      // If any of the demanded bits are produced by the sign extension, we also
      // demand the input sign bit.
      APInt HighBits = APInt::getHighBitsSet(BitWidth, ShAmt);
      if (HighBits.intersects(NewMask))
        InDemandedMask |= APInt::getSignBit(VT.getScalarType().getSizeInBits());

      if (SimplifyDemandedBits(Op.getOperand(0), InDemandedMask,
                               KnownZero, KnownOne, TLO, Depth+1))
        return true;
      assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?");
      KnownZero = KnownZero.lshr(ShAmt);
      KnownOne  = KnownOne.lshr(ShAmt);

      // Handle the sign bit, adjusted to where it is now in the mask.
      APInt SignBit = APInt::getSignBit(BitWidth).lshr(ShAmt);

      // If the input sign bit is known to be zero, or if none of the top bits
      // are demanded, turn this into an unsigned shift right.
      if (KnownZero.intersects(SignBit) || (HighBits & ~NewMask) == HighBits) {
        SDNodeFlags Flags;
        Flags.setExact(cast<BinaryWithFlagsSDNode>(Op)->Flags.hasExact());
        return TLO.CombineTo(Op,
                             TLO.DAG.getNode(ISD::SRL, dl, VT, Op.getOperand(0),
                                             Op.getOperand(1), &Flags));
      }

      int Log2 = NewMask.exactLogBase2();
      if (Log2 >= 0) {
        // The bit must come from the sign.
        SDValue NewSA =
          TLO.DAG.getConstant(BitWidth - 1 - Log2, dl,
                              Op.getOperand(1).getValueType());
        return TLO.CombineTo(Op, TLO.DAG.getNode(ISD::SRL, dl, VT,
                                                 Op.getOperand(0), NewSA));
      }

      if (KnownOne.intersects(SignBit))
        // New bits are known one.
        KnownOne |= HighBits;
    }
    break;
  case ISD::SIGN_EXTEND_INREG: {
    EVT ExVT = cast<VTSDNode>(Op.getOperand(1))->getVT();

    APInt MsbMask = APInt::getHighBitsSet(BitWidth, 1);
    // If we only care about the highest bit, don't bother shifting right.
    if (MsbMask == NewMask) {
      unsigned ShAmt = ExVT.getScalarType().getSizeInBits();
      SDValue InOp = Op.getOperand(0);
      unsigned VTBits = Op->getValueType(0).getScalarType().getSizeInBits();
      bool AlreadySignExtended =
        TLO.DAG.ComputeNumSignBits(InOp) >= VTBits-ShAmt+1;
      // However if the input is already sign extended we expect the sign
      // extension to be dropped altogether later and do not simplify.
      if (!AlreadySignExtended) {
        // Compute the correct shift amount type, which must be getShiftAmountTy
        // for scalar types after legalization.
        EVT ShiftAmtTy = Op.getValueType();
        if (TLO.LegalTypes() && !ShiftAmtTy.isVector())
          ShiftAmtTy = getShiftAmountTy(ShiftAmtTy, DL);

        SDValue ShiftAmt = TLO.DAG.getConstant(BitWidth - ShAmt, dl,
                                               ShiftAmtTy);
        return TLO.CombineTo(Op, TLO.DAG.getNode(ISD::SHL, dl,
                                                 Op.getValueType(), InOp,
                                                 ShiftAmt));
      }
    }

    // Sign extension.  Compute the demanded bits in the result that are not
    // present in the input.
    APInt NewBits =
      APInt::getHighBitsSet(BitWidth,
                            BitWidth - ExVT.getScalarType().getSizeInBits());

    // If none of the extended bits are demanded, eliminate the sextinreg.
    if ((NewBits & NewMask) == 0)
      return TLO.CombineTo(Op, Op.getOperand(0));

    APInt InSignBit =
      APInt::getSignBit(ExVT.getScalarType().getSizeInBits()).zext(BitWidth);
    APInt InputDemandedBits =
      APInt::getLowBitsSet(BitWidth,
                           ExVT.getScalarType().getSizeInBits()) &
      NewMask;

    // Since the sign extended bits are demanded, we know that the sign
    // bit is demanded.
    InputDemandedBits |= InSignBit;

    if (SimplifyDemandedBits(Op.getOperand(0), InputDemandedBits,
                             KnownZero, KnownOne, TLO, Depth+1))
      return true;
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?");

    // If the sign bit of the input is known set or clear, then we know the
    // top bits of the result.

    // If the input sign bit is known zero, convert this into a zero extension.
    if (KnownZero.intersects(InSignBit))
      return TLO.CombineTo(Op,
                          TLO.DAG.getZeroExtendInReg(Op.getOperand(0),dl,ExVT));

    if (KnownOne.intersects(InSignBit)) {    // Input sign bit known set
      KnownOne |= NewBits;
      KnownZero &= ~NewBits;
    } else {                       // Input sign bit unknown
      KnownZero &= ~NewBits;
      KnownOne &= ~NewBits;
    }
    break;
  }
  case ISD::BUILD_PAIR: {
    EVT HalfVT = Op.getOperand(0).getValueType();
    unsigned HalfBitWidth = HalfVT.getScalarSizeInBits();

    APInt MaskLo = NewMask.getLoBits(HalfBitWidth).trunc(HalfBitWidth);
    APInt MaskHi = NewMask.getHiBits(HalfBitWidth).trunc(HalfBitWidth);

    APInt KnownZeroLo, KnownOneLo;
    APInt KnownZeroHi, KnownOneHi;

    if (SimplifyDemandedBits(Op.getOperand(0), MaskLo, KnownZeroLo,
                             KnownOneLo, TLO, Depth + 1))
      return true;

    if (SimplifyDemandedBits(Op.getOperand(1), MaskHi, KnownZeroHi,
                             KnownOneHi, TLO, Depth + 1))
      return true;

    KnownZero = KnownZeroLo.zext(BitWidth) |
                KnownZeroHi.zext(BitWidth).shl(HalfBitWidth);

    KnownOne = KnownOneLo.zext(BitWidth) |
               KnownOneHi.zext(BitWidth).shl(HalfBitWidth);
    break;
  }
  case ISD::ZERO_EXTEND: {
    unsigned OperandBitWidth =
      Op.getOperand(0).getValueType().getScalarType().getSizeInBits();
    APInt InMask = NewMask.trunc(OperandBitWidth);

    // If none of the top bits are demanded, convert this into an any_extend.
    APInt NewBits =
      APInt::getHighBitsSet(BitWidth, BitWidth - OperandBitWidth) & NewMask;
    if (!NewBits.intersects(NewMask))
      return TLO.CombineTo(Op, TLO.DAG.getNode(ISD::ANY_EXTEND, dl,
                                               Op.getValueType(),
                                               Op.getOperand(0)));

    if (SimplifyDemandedBits(Op.getOperand(0), InMask,
                             KnownZero, KnownOne, TLO, Depth+1))
      return true;
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?");
    KnownZero = KnownZero.zext(BitWidth);
    KnownOne = KnownOne.zext(BitWidth);
    KnownZero |= NewBits;
    break;
  }
  case ISD::SIGN_EXTEND: {
    EVT InVT = Op.getOperand(0).getValueType();
    unsigned InBits = InVT.getScalarType().getSizeInBits();
    APInt InMask    = APInt::getLowBitsSet(BitWidth, InBits);
    APInt InSignBit = APInt::getBitsSet(BitWidth, InBits - 1, InBits);
    APInt NewBits   = ~InMask & NewMask;

    // If none of the top bits are demanded, convert this into an any_extend.
    if (NewBits == 0)
      return TLO.CombineTo(Op,TLO.DAG.getNode(ISD::ANY_EXTEND, dl,
                                              Op.getValueType(),
                                              Op.getOperand(0)));

    // Since some of the sign extended bits are demanded, we know that the sign
    // bit is demanded.
    APInt InDemandedBits = InMask & NewMask;
    InDemandedBits |= InSignBit;
    InDemandedBits = InDemandedBits.trunc(InBits);

    if (SimplifyDemandedBits(Op.getOperand(0), InDemandedBits, KnownZero,
                             KnownOne, TLO, Depth+1))
      return true;
    KnownZero = KnownZero.zext(BitWidth);
    KnownOne = KnownOne.zext(BitWidth);

    // If the sign bit is known zero, convert this to a zero extend.
    if (KnownZero.intersects(InSignBit))
      return TLO.CombineTo(Op, TLO.DAG.getNode(ISD::ZERO_EXTEND, dl,
                                               Op.getValueType(),
                                               Op.getOperand(0)));

    // If the sign bit is known one, the top bits match.
    if (KnownOne.intersects(InSignBit)) {
      KnownOne |= NewBits;
      assert((KnownZero & NewBits) == 0);
    } else {   // Otherwise, top bits aren't known.
      assert((KnownOne & NewBits) == 0);
      assert((KnownZero & NewBits) == 0);
    }
    break;
  }
  case ISD::ANY_EXTEND: {
    unsigned OperandBitWidth =
      Op.getOperand(0).getValueType().getScalarType().getSizeInBits();
    APInt InMask = NewMask.trunc(OperandBitWidth);
    if (SimplifyDemandedBits(Op.getOperand(0), InMask,
                             KnownZero, KnownOne, TLO, Depth+1))
      return true;
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?");
    KnownZero = KnownZero.zext(BitWidth);
    KnownOne = KnownOne.zext(BitWidth);
    break;
  }
  case ISD::TRUNCATE: {
    // Simplify the input, using demanded bit information, and compute the known
    // zero/one bits live out.
    unsigned OperandBitWidth =
      Op.getOperand(0).getValueType().getScalarType().getSizeInBits();
    APInt TruncMask = NewMask.zext(OperandBitWidth);
    if (SimplifyDemandedBits(Op.getOperand(0), TruncMask,
                             KnownZero, KnownOne, TLO, Depth+1))
      return true;
    KnownZero = KnownZero.trunc(BitWidth);
    KnownOne = KnownOne.trunc(BitWidth);

    // If the input is only used by this truncate, see if we can shrink it based
    // on the known demanded bits.
    if (Op.getOperand(0).getNode()->hasOneUse()) {
      SDValue In = Op.getOperand(0);
      switch (In.getOpcode()) {
      default: break;
      case ISD::SRL:
        // Shrink SRL by a constant if none of the high bits shifted in are
        // demanded.
        if (TLO.LegalTypes() &&
            !isTypeDesirableForOp(ISD::SRL, Op.getValueType()))
          // Do not turn (vt1 truncate (vt2 srl)) into (vt1 srl) if vt1 is
          // undesirable.
          break;
        ConstantSDNode *ShAmt = dyn_cast<ConstantSDNode>(In.getOperand(1));
        if (!ShAmt)
          break;
        SDValue Shift = In.getOperand(1);
        if (TLO.LegalTypes()) {
          uint64_t ShVal = ShAmt->getZExtValue();
          Shift = TLO.DAG.getConstant(ShVal, dl,
                                      getShiftAmountTy(Op.getValueType(), DL));
        }

        APInt HighBits = APInt::getHighBitsSet(OperandBitWidth,
                                               OperandBitWidth - BitWidth);
        HighBits = HighBits.lshr(ShAmt->getZExtValue()).trunc(BitWidth);

        if (ShAmt->getZExtValue() < BitWidth && !(HighBits & NewMask)) {
          // None of the shifted in bits are needed.  Add a truncate of the
          // shift input, then shift it.
          SDValue NewTrunc = TLO.DAG.getNode(ISD::TRUNCATE, dl,
                                             Op.getValueType(),
                                             In.getOperand(0));
          return TLO.CombineTo(Op, TLO.DAG.getNode(ISD::SRL, dl,
                                                   Op.getValueType(),
                                                   NewTrunc,
                                                   Shift));
        }
        break;
      }
    }

    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?");
    break;
  }
  case ISD::AssertZext: {
    // AssertZext demands all of the high bits, plus any of the low bits
    // demanded by its users.
    EVT VT = cast<VTSDNode>(Op.getOperand(1))->getVT();
    APInt InMask = APInt::getLowBitsSet(BitWidth,
                                        VT.getSizeInBits());
    if (SimplifyDemandedBits(Op.getOperand(0), ~InMask | NewMask,
                             KnownZero, KnownOne, TLO, Depth+1))
      return true;
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?");

    KnownZero |= ~InMask & NewMask;
    break;
  }
  case ISD::BITCAST:
    // If this is an FP->Int bitcast and if the sign bit is the only
    // thing demanded, turn this into a FGETSIGN.
    if (!TLO.LegalOperations() &&
        !Op.getValueType().isVector() &&
        !Op.getOperand(0).getValueType().isVector() &&
        NewMask == APInt::getSignBit(Op.getValueType().getSizeInBits()) &&
        Op.getOperand(0).getValueType().isFloatingPoint()) {
      bool OpVTLegal = isOperationLegalOrCustom(ISD::FGETSIGN, Op.getValueType());
      bool i32Legal  = isOperationLegalOrCustom(ISD::FGETSIGN, MVT::i32);
      if ((OpVTLegal || i32Legal) && Op.getValueType().isSimple()) {
        EVT Ty = OpVTLegal ? Op.getValueType() : MVT::i32;
        // Make a FGETSIGN + SHL to move the sign bit into the appropriate
        // place.  We expect the SHL to be eliminated by other optimizations.
        SDValue Sign = TLO.DAG.getNode(ISD::FGETSIGN, dl, Ty, Op.getOperand(0));
        unsigned OpVTSizeInBits = Op.getValueType().getSizeInBits();
        if (!OpVTLegal && OpVTSizeInBits > 32)
          Sign = TLO.DAG.getNode(ISD::ZERO_EXTEND, dl, Op.getValueType(), Sign);
        unsigned ShVal = Op.getValueType().getSizeInBits()-1;
        SDValue ShAmt = TLO.DAG.getConstant(ShVal, dl, Op.getValueType());
        return TLO.CombineTo(Op, TLO.DAG.getNode(ISD::SHL, dl,
                                                 Op.getValueType(),
                                                 Sign, ShAmt));
      }
    }
    break;
  case ISD::ADD:
  case ISD::MUL:
  case ISD::SUB: {
    // Add, Sub, and Mul don't demand any bits in positions beyond that
    // of the highest bit demanded of them.
    APInt LoMask = APInt::getLowBitsSet(BitWidth,
                                        BitWidth - NewMask.countLeadingZeros());
    if (SimplifyDemandedBits(Op.getOperand(0), LoMask, KnownZero2,
                             KnownOne2, TLO, Depth+1))
      return true;
    if (SimplifyDemandedBits(Op.getOperand(1), LoMask, KnownZero2,
                             KnownOne2, TLO, Depth+1))
      return true;
    // See if the operation should be performed at a smaller bit width.
    if (TLO.ShrinkDemandedOp(Op, BitWidth, NewMask, dl))
      return true;
  }
  // FALL THROUGH
  default:
    // Just use computeKnownBits to compute output bits.
    TLO.DAG.computeKnownBits(Op, KnownZero, KnownOne, Depth);
    break;
  }

  // If we know the value of all of the demanded bits, return this as a
  // constant.
  if ((NewMask & (KnownZero|KnownOne)) == NewMask) {
    // Avoid folding to a constant if any OpaqueConstant is involved.
    const SDNode *N = Op.getNode();
    for (SDNodeIterator I = SDNodeIterator::begin(N),
         E = SDNodeIterator::end(N); I != E; ++I) {
      SDNode *Op = *I;
      if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op))
        if (C->isOpaque())
          return false;
    }
    return TLO.CombineTo(Op,
                         TLO.DAG.getConstant(KnownOne, dl, Op.getValueType()));
  }

  return false;
}

/// computeKnownBitsForTargetNode - Determine which of the bits specified
/// in Mask are known to be either zero or one and return them in the
/// KnownZero/KnownOne bitsets.
void TargetLowering::computeKnownBitsForTargetNode(const SDValue Op,
                                                   APInt &KnownZero,
                                                   APInt &KnownOne,
                                                   const SelectionDAG &DAG,
                                                   unsigned Depth) const {
  assert((Op.getOpcode() >= ISD::BUILTIN_OP_END ||
          Op.getOpcode() == ISD::INTRINSIC_WO_CHAIN ||
          Op.getOpcode() == ISD::INTRINSIC_W_CHAIN ||
          Op.getOpcode() == ISD::INTRINSIC_VOID) &&
         "Should use MaskedValueIsZero if you don't know whether Op"
         " is a target node!");
  KnownZero = KnownOne = APInt(KnownOne.getBitWidth(), 0);
}

/// ComputeNumSignBitsForTargetNode - This method can be implemented by
/// targets that want to expose additional information about sign bits to the
/// DAG Combiner.
unsigned TargetLowering::ComputeNumSignBitsForTargetNode(SDValue Op,
                                                         const SelectionDAG &,
                                                         unsigned Depth) const {
  assert((Op.getOpcode() >= ISD::BUILTIN_OP_END ||
          Op.getOpcode() == ISD::INTRINSIC_WO_CHAIN ||
          Op.getOpcode() == ISD::INTRINSIC_W_CHAIN ||
          Op.getOpcode() == ISD::INTRINSIC_VOID) &&
         "Should use ComputeNumSignBits if you don't know whether Op"
         " is a target node!");
  return 1;
}

/// ValueHasExactlyOneBitSet - Test if the given value is known to have exactly
/// one bit set. This differs from computeKnownBits in that it doesn't need to
/// determine which bit is set.
///
static bool ValueHasExactlyOneBitSet(SDValue Val, const SelectionDAG &DAG) {
  // A left-shift of a constant one will have exactly one bit set, because
  // shifting the bit off the end is undefined.
  if (Val.getOpcode() == ISD::SHL)
    if (ConstantSDNode *C =
         dyn_cast<ConstantSDNode>(Val.getNode()->getOperand(0)))
      if (C->getAPIntValue() == 1)
        return true;

  // Similarly, a right-shift of a constant sign-bit will have exactly
  // one bit set.
  if (Val.getOpcode() == ISD::SRL)
    if (ConstantSDNode *C =
         dyn_cast<ConstantSDNode>(Val.getNode()->getOperand(0)))
      if (C->getAPIntValue().isSignBit())
        return true;

  // More could be done here, though the above checks are enough
  // to handle some common cases.

  // Fall back to computeKnownBits to catch other known cases.
  EVT OpVT = Val.getValueType();
  unsigned BitWidth = OpVT.getScalarType().getSizeInBits();
  APInt KnownZero, KnownOne;
  DAG.computeKnownBits(Val, KnownZero, KnownOne);
  return (KnownZero.countPopulation() == BitWidth - 1) &&
         (KnownOne.countPopulation() == 1);
}

bool TargetLowering::isConstTrueVal(const SDNode *N) const {
  if (!N)
    return false;

  const ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N);
  if (!CN) {
    const BuildVectorSDNode *BV = dyn_cast<BuildVectorSDNode>(N);
    if (!BV)
      return false;

    BitVector UndefElements;
    CN = BV->getConstantSplatNode(&UndefElements);
    // Only interested in constant splats, and we don't try to handle undef
    // elements in identifying boolean constants.
    if (!CN || UndefElements.none())
      return false;
  }

  switch (getBooleanContents(N->getValueType(0))) {
  case UndefinedBooleanContent:
    return CN->getAPIntValue()[0];
  case ZeroOrOneBooleanContent:
    return CN->isOne();
  case ZeroOrNegativeOneBooleanContent:
    return CN->isAllOnesValue();
  }

  llvm_unreachable("Invalid boolean contents");
}

bool TargetLowering::isConstFalseVal(const SDNode *N) const {
  if (!N)
    return false;

  const ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N);
  if (!CN) {
    const BuildVectorSDNode *BV = dyn_cast<BuildVectorSDNode>(N);
    if (!BV)
      return false;

    BitVector UndefElements;
    CN = BV->getConstantSplatNode(&UndefElements);
    // Only interested in constant splats, and we don't try to handle undef
    // elements in identifying boolean constants.
    if (!CN || UndefElements.none())
      return false;
  }

  if (getBooleanContents(N->getValueType(0)) == UndefinedBooleanContent)
    return !CN->getAPIntValue()[0];

  return CN->isNullValue();
}

/// SimplifySetCC - Try to simplify a setcc built with the specified operands
/// and cc. If it is unable to simplify it, return a null SDValue.
SDValue
TargetLowering::SimplifySetCC(EVT VT, SDValue N0, SDValue N1,
                              ISD::CondCode Cond, bool foldBooleans,
                              DAGCombinerInfo &DCI, SDLoc dl) const {
  SelectionDAG &DAG = DCI.DAG;

  // These setcc operations always fold.
  switch (Cond) {
  default: break;
  case ISD::SETFALSE:
  case ISD::SETFALSE2: return DAG.getConstant(0, dl, VT);
  case ISD::SETTRUE:
  case ISD::SETTRUE2: {
    TargetLowering::BooleanContent Cnt =
        getBooleanContents(N0->getValueType(0));
    return DAG.getConstant(
        Cnt == TargetLowering::ZeroOrNegativeOneBooleanContent ? -1ULL : 1, dl,
        VT);
  }
  }

  // Ensure that the constant occurs on the RHS, and fold constant
  // comparisons.
  ISD::CondCode SwappedCC = ISD::getSetCCSwappedOperands(Cond);
  if (isa<ConstantSDNode>(N0.getNode()) &&
      (DCI.isBeforeLegalizeOps() ||
       isCondCodeLegal(SwappedCC, N0.getSimpleValueType())))
    return DAG.getSetCC(dl, VT, N1, N0, SwappedCC);

  if (ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1.getNode())) {
    const APInt &C1 = N1C->getAPIntValue();

    // If the LHS is '(srl (ctlz x), 5)', the RHS is 0/1, and this is an
    // equality comparison, then we're just comparing whether X itself is
    // zero.
    if (N0.getOpcode() == ISD::SRL && (C1 == 0 || C1 == 1) &&
        N0.getOperand(0).getOpcode() == ISD::CTLZ &&
        N0.getOperand(1).getOpcode() == ISD::Constant) {
      const APInt &ShAmt
        = cast<ConstantSDNode>(N0.getOperand(1))->getAPIntValue();
      if ((Cond == ISD::SETEQ || Cond == ISD::SETNE) &&
          ShAmt == Log2_32(N0.getValueType().getSizeInBits())) {
        if ((C1 == 0) == (Cond == ISD::SETEQ)) {
          // (srl (ctlz x), 5) == 0  -> X != 0
          // (srl (ctlz x), 5) != 1  -> X != 0
          Cond = ISD::SETNE;
        } else {
          // (srl (ctlz x), 5) != 0  -> X == 0
          // (srl (ctlz x), 5) == 1  -> X == 0
          Cond = ISD::SETEQ;
        }
        SDValue Zero = DAG.getConstant(0, dl, N0.getValueType());
        return DAG.getSetCC(dl, VT, N0.getOperand(0).getOperand(0),
                            Zero, Cond);
      }
    }

    SDValue CTPOP = N0;
    // Look through truncs that don't change the value of a ctpop.
    if (N0.hasOneUse() && N0.getOpcode() == ISD::TRUNCATE)
      CTPOP = N0.getOperand(0);

    if (CTPOP.hasOneUse() && CTPOP.getOpcode() == ISD::CTPOP &&
        (N0 == CTPOP || N0.getValueType().getSizeInBits() >
                        Log2_32_Ceil(CTPOP.getValueType().getSizeInBits()))) {
      EVT CTVT = CTPOP.getValueType();
      SDValue CTOp = CTPOP.getOperand(0);

      // (ctpop x) u< 2 -> (x & x-1) == 0
      // (ctpop x) u> 1 -> (x & x-1) != 0
      if ((Cond == ISD::SETULT && C1 == 2) || (Cond == ISD::SETUGT && C1 == 1)){
        SDValue Sub = DAG.getNode(ISD::SUB, dl, CTVT, CTOp,
                                  DAG.getConstant(1, dl, CTVT));
        SDValue And = DAG.getNode(ISD::AND, dl, CTVT, CTOp, Sub);
        ISD::CondCode CC = Cond == ISD::SETULT ? ISD::SETEQ : ISD::SETNE;
        return DAG.getSetCC(dl, VT, And, DAG.getConstant(0, dl, CTVT), CC);
      }

      // TODO: (ctpop x) == 1 -> x && (x & x-1) == 0 iff ctpop is illegal.
    }

    // (zext x) == C --> x == (trunc C)
    // (sext x) == C --> x == (trunc C)
    if ((Cond == ISD::SETEQ || Cond == ISD::SETNE) &&
        DCI.isBeforeLegalize() && N0->hasOneUse()) {
      unsigned MinBits = N0.getValueSizeInBits();
      SDValue PreExt;
      bool Signed = false;
      if (N0->getOpcode() == ISD::ZERO_EXTEND) {
        // ZExt
        MinBits = N0->getOperand(0).getValueSizeInBits();
        PreExt = N0->getOperand(0);
      } else if (N0->getOpcode() == ISD::AND) {
        // DAGCombine turns costly ZExts into ANDs
        if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(N0->getOperand(1)))
          if ((C->getAPIntValue()+1).isPowerOf2()) {
            MinBits = C->getAPIntValue().countTrailingOnes();
            PreExt = N0->getOperand(0);
          }
      } else if (N0->getOpcode() == ISD::SIGN_EXTEND) {
        // SExt
        MinBits = N0->getOperand(0).getValueSizeInBits();
        PreExt = N0->getOperand(0);
        Signed = true;
      } else if (LoadSDNode *LN0 = dyn_cast<LoadSDNode>(N0)) {
        // ZEXTLOAD / SEXTLOAD
        if (LN0->getExtensionType() == ISD::ZEXTLOAD) {
          MinBits = LN0->getMemoryVT().getSizeInBits();
          PreExt = N0;
        } else if (LN0->getExtensionType() == ISD::SEXTLOAD) {
          Signed = true;
          MinBits = LN0->getMemoryVT().getSizeInBits();
          PreExt = N0;
        }
      }

      // Figure out how many bits we need to preserve this constant.
      unsigned ReqdBits = Signed ?
        C1.getBitWidth() - C1.getNumSignBits() + 1 :
        C1.getActiveBits();

      // Make sure we're not losing bits from the constant.
      if (MinBits > 0 &&
          MinBits < C1.getBitWidth() &&
          MinBits >= ReqdBits) {
        EVT MinVT = EVT::getIntegerVT(*DAG.getContext(), MinBits);
        if (isTypeDesirableForOp(ISD::SETCC, MinVT)) {
          // Will get folded away.
          SDValue Trunc = DAG.getNode(ISD::TRUNCATE, dl, MinVT, PreExt);
          SDValue C = DAG.getConstant(C1.trunc(MinBits), dl, MinVT);
          return DAG.getSetCC(dl, VT, Trunc, C, Cond);
        }
      }
    }

    // If the LHS is '(and load, const)', the RHS is 0,
    // the test is for equality or unsigned, and all 1 bits of the const are
    // in the same partial word, see if we can shorten the load.
    if (DCI.isBeforeLegalize() &&
        !ISD::isSignedIntSetCC(Cond) &&
        N0.getOpcode() == ISD::AND && C1 == 0 &&
        N0.getNode()->hasOneUse() &&
        isa<LoadSDNode>(N0.getOperand(0)) &&
        N0.getOperand(0).getNode()->hasOneUse() &&
        isa<ConstantSDNode>(N0.getOperand(1))) {
      LoadSDNode *Lod = cast<LoadSDNode>(N0.getOperand(0));
      APInt bestMask;
      unsigned bestWidth = 0, bestOffset = 0;
      if (!Lod->isVolatile() && Lod->isUnindexed()) {
        unsigned origWidth = N0.getValueType().getSizeInBits();
        unsigned maskWidth = origWidth;
        // We can narrow (e.g.) 16-bit extending loads on 32-bit target to
        // 8 bits, but have to be careful...
        if (Lod->getExtensionType() != ISD::NON_EXTLOAD)
          origWidth = Lod->getMemoryVT().getSizeInBits();
        const APInt &Mask =
          cast<ConstantSDNode>(N0.getOperand(1))->getAPIntValue();
        for (unsigned width = origWidth / 2; width>=8; width /= 2) {
          APInt newMask = APInt::getLowBitsSet(maskWidth, width);
          for (unsigned offset=0; offset<origWidth/width; offset++) {
            if ((newMask & Mask) == Mask) {
              if (!DAG.getDataLayout().isLittleEndian())
                bestOffset = (origWidth/width - offset - 1) * (width/8);
              else
                bestOffset = (uint64_t)offset * (width/8);
              bestMask = Mask.lshr(offset * (width/8) * 8);
              bestWidth = width;
              break;
            }
            newMask = newMask << width;
          }
        }
      }
      if (bestWidth) {
        EVT newVT = EVT::getIntegerVT(*DAG.getContext(), bestWidth);
        if (newVT.isRound()) {
          EVT PtrType = Lod->getOperand(1).getValueType();
          SDValue Ptr = Lod->getBasePtr();
          if (bestOffset != 0)
            Ptr = DAG.getNode(ISD::ADD, dl, PtrType, Lod->getBasePtr(),
                              DAG.getConstant(bestOffset, dl, PtrType));
          unsigned NewAlign = MinAlign(Lod->getAlignment(), bestOffset);
          SDValue NewLoad = DAG.getLoad(newVT, dl, Lod->getChain(), Ptr,
                                Lod->getPointerInfo().getWithOffset(bestOffset),
                                        false, false, false, NewAlign);
          return DAG.getSetCC(dl, VT,
                              DAG.getNode(ISD::AND, dl, newVT, NewLoad,
                                      DAG.getConstant(bestMask.trunc(bestWidth),
                                                      dl, newVT)),
                              DAG.getConstant(0LL, dl, newVT), Cond);
        }
      }
    }

    // If the LHS is a ZERO_EXTEND, perform the comparison on the input.
    if (N0.getOpcode() == ISD::ZERO_EXTEND) {
      unsigned InSize = N0.getOperand(0).getValueType().getSizeInBits();

      // If the comparison constant has bits in the upper part, the
      // zero-extended value could never match.
      if (C1.intersects(APInt::getHighBitsSet(C1.getBitWidth(),
                                              C1.getBitWidth() - InSize))) {
        switch (Cond) {
        case ISD::SETUGT:
        case ISD::SETUGE:
        case ISD::SETEQ: return DAG.getConstant(0, dl, VT);
        case ISD::SETULT:
        case ISD::SETULE:
        case ISD::SETNE: return DAG.getConstant(1, dl, VT);
        case ISD::SETGT:
        case ISD::SETGE:
          // True if the sign bit of C1 is set.
          return DAG.getConstant(C1.isNegative(), dl, VT);
        case ISD::SETLT:
        case ISD::SETLE:
          // True if the sign bit of C1 isn't set.
          return DAG.getConstant(C1.isNonNegative(), dl, VT);
        default:
          break;
        }
      }

      // Otherwise, we can perform the comparison with the low bits.
      switch (Cond) {
      case ISD::SETEQ:
      case ISD::SETNE:
      case ISD::SETUGT:
      case ISD::SETUGE:
      case ISD::SETULT:
      case ISD::SETULE: {
        EVT newVT = N0.getOperand(0).getValueType();
        if (DCI.isBeforeLegalizeOps() ||
            (isOperationLegal(ISD::SETCC, newVT) &&
             getCondCodeAction(Cond, newVT.getSimpleVT()) == Legal)) {
          EVT NewSetCCVT =
              getSetCCResultType(DAG.getDataLayout(), *DAG.getContext(), newVT);
          SDValue NewConst = DAG.getConstant(C1.trunc(InSize), dl, newVT);

          SDValue NewSetCC = DAG.getSetCC(dl, NewSetCCVT, N0.getOperand(0),
                                          NewConst, Cond);
          return DAG.getBoolExtOrTrunc(NewSetCC, dl, VT, N0.getValueType());
        }
        break;
      }
      default:
        break;   // todo, be more careful with signed comparisons
      }
    } else if (N0.getOpcode() == ISD::SIGN_EXTEND_INREG &&
               (Cond == ISD::SETEQ || Cond == ISD::SETNE)) {
      EVT ExtSrcTy = cast<VTSDNode>(N0.getOperand(1))->getVT();
      unsigned ExtSrcTyBits = ExtSrcTy.getSizeInBits();
      EVT ExtDstTy = N0.getValueType();
      unsigned ExtDstTyBits = ExtDstTy.getSizeInBits();

      // If the constant doesn't fit into the number of bits for the source of
      // the sign extension, it is impossible for both sides to be equal.
      if (C1.getMinSignedBits() > ExtSrcTyBits)
        return DAG.getConstant(Cond == ISD::SETNE, dl, VT);

      SDValue ZextOp;
      EVT Op0Ty = N0.getOperand(0).getValueType();
      if (Op0Ty == ExtSrcTy) {
        ZextOp = N0.getOperand(0);
      } else {
        APInt Imm = APInt::getLowBitsSet(ExtDstTyBits, ExtSrcTyBits);
        ZextOp = DAG.getNode(ISD::AND, dl, Op0Ty, N0.getOperand(0),
                              DAG.getConstant(Imm, dl, Op0Ty));
      }
      if (!DCI.isCalledByLegalizer())
        DCI.AddToWorklist(ZextOp.getNode());
      // Otherwise, make this a use of a zext.
      return DAG.getSetCC(dl, VT, ZextOp,
                          DAG.getConstant(C1 & APInt::getLowBitsSet(
                                                              ExtDstTyBits,
                                                              ExtSrcTyBits),
                                          dl, ExtDstTy),
                          Cond);
    } else if ((N1C->isNullValue() || N1C->getAPIntValue() == 1) &&
                (Cond == ISD::SETEQ || Cond == ISD::SETNE)) {
      // SETCC (SETCC), [0|1], [EQ|NE]  -> SETCC
      if (N0.getOpcode() == ISD::SETCC &&
          isTypeLegal(VT) && VT.bitsLE(N0.getValueType())) {
        bool TrueWhenTrue = (Cond == ISD::SETEQ) ^ (N1C->getAPIntValue() != 1);
        if (TrueWhenTrue)
          return DAG.getNode(ISD::TRUNCATE, dl, VT, N0);
        // Invert the condition.
        ISD::CondCode CC = cast<CondCodeSDNode>(N0.getOperand(2))->get();
        CC = ISD::getSetCCInverse(CC,
                                  N0.getOperand(0).getValueType().isInteger());
        if (DCI.isBeforeLegalizeOps() ||
            isCondCodeLegal(CC, N0.getOperand(0).getSimpleValueType()))
          return DAG.getSetCC(dl, VT, N0.getOperand(0), N0.getOperand(1), CC);
      }

      if ((N0.getOpcode() == ISD::XOR ||
           (N0.getOpcode() == ISD::AND &&
            N0.getOperand(0).getOpcode() == ISD::XOR &&
            N0.getOperand(1) == N0.getOperand(0).getOperand(1))) &&
          isa<ConstantSDNode>(N0.getOperand(1)) &&
          cast<ConstantSDNode>(N0.getOperand(1))->getAPIntValue() == 1) {
        // If this is (X^1) == 0/1, swap the RHS and eliminate the xor.  We
        // can only do this if the top bits are known zero.
        unsigned BitWidth = N0.getValueSizeInBits();
        if (DAG.MaskedValueIsZero(N0,
                                  APInt::getHighBitsSet(BitWidth,
                                                        BitWidth-1))) {
          // Okay, get the un-inverted input value.
          SDValue Val;
          if (N0.getOpcode() == ISD::XOR)
            Val = N0.getOperand(0);
          else {
            assert(N0.getOpcode() == ISD::AND &&
                    N0.getOperand(0).getOpcode() == ISD::XOR);
            // ((X^1)&1)^1 -> X & 1
            Val = DAG.getNode(ISD::AND, dl, N0.getValueType(),
                              N0.getOperand(0).getOperand(0),
                              N0.getOperand(1));
          }

          return DAG.getSetCC(dl, VT, Val, N1,
                              Cond == ISD::SETEQ ? ISD::SETNE : ISD::SETEQ);
        }
      } else if (N1C->getAPIntValue() == 1 &&
                 (VT == MVT::i1 ||
                  getBooleanContents(N0->getValueType(0)) ==
                      ZeroOrOneBooleanContent)) {
        SDValue Op0 = N0;
        if (Op0.getOpcode() == ISD::TRUNCATE)
          Op0 = Op0.getOperand(0);

        if ((Op0.getOpcode() == ISD::XOR) &&
            Op0.getOperand(0).getOpcode() == ISD::SETCC &&
            Op0.getOperand(1).getOpcode() == ISD::SETCC) {
          // (xor (setcc), (setcc)) == / != 1 -> (setcc) != / == (setcc)
          Cond = (Cond == ISD::SETEQ) ? ISD::SETNE : ISD::SETEQ;
          return DAG.getSetCC(dl, VT, Op0.getOperand(0), Op0.getOperand(1),
                              Cond);
        }
        if (Op0.getOpcode() == ISD::AND &&
            isa<ConstantSDNode>(Op0.getOperand(1)) &&
            cast<ConstantSDNode>(Op0.getOperand(1))->getAPIntValue() == 1) {
          // If this is (X&1) == / != 1, normalize it to (X&1) != / == 0.
          if (Op0.getValueType().bitsGT(VT))
            Op0 = DAG.getNode(ISD::AND, dl, VT,
                          DAG.getNode(ISD::TRUNCATE, dl, VT, Op0.getOperand(0)),
                          DAG.getConstant(1, dl, VT));
          else if (Op0.getValueType().bitsLT(VT))
            Op0 = DAG.getNode(ISD::AND, dl, VT,
                        DAG.getNode(ISD::ANY_EXTEND, dl, VT, Op0.getOperand(0)),
                        DAG.getConstant(1, dl, VT));

          return DAG.getSetCC(dl, VT, Op0,
                              DAG.getConstant(0, dl, Op0.getValueType()),
                              Cond == ISD::SETEQ ? ISD::SETNE : ISD::SETEQ);
        }
        if (Op0.getOpcode() == ISD::AssertZext &&
            cast<VTSDNode>(Op0.getOperand(1))->getVT() == MVT::i1)
          return DAG.getSetCC(dl, VT, Op0,
                              DAG.getConstant(0, dl, Op0.getValueType()),
                              Cond == ISD::SETEQ ? ISD::SETNE : ISD::SETEQ);
      }
    }

    APInt MinVal, MaxVal;
    unsigned OperandBitSize = N1C->getValueType(0).getSizeInBits();
    if (ISD::isSignedIntSetCC(Cond)) {
      MinVal = APInt::getSignedMinValue(OperandBitSize);
      MaxVal = APInt::getSignedMaxValue(OperandBitSize);
    } else {
      MinVal = APInt::getMinValue(OperandBitSize);
      MaxVal = APInt::getMaxValue(OperandBitSize);
    }

    // Canonicalize GE/LE comparisons to use GT/LT comparisons.
    if (Cond == ISD::SETGE || Cond == ISD::SETUGE) {
      if (C1 == MinVal) return DAG.getConstant(1, dl, VT);  // X >= MIN --> true
      // X >= C0 --> X > (C0 - 1)
      APInt C = C1 - 1;
      ISD::CondCode NewCC = (Cond == ISD::SETGE) ? ISD::SETGT : ISD::SETUGT;
      if ((DCI.isBeforeLegalizeOps() ||
           isCondCodeLegal(NewCC, VT.getSimpleVT())) &&
          (!N1C->isOpaque() || (N1C->isOpaque() && C.getBitWidth() <= 64 &&
                                isLegalICmpImmediate(C.getSExtValue())))) {
        return DAG.getSetCC(dl, VT, N0,
                            DAG.getConstant(C, dl, N1.getValueType()),
                            NewCC);
      }
    }

    if (Cond == ISD::SETLE || Cond == ISD::SETULE) {
      if (C1 == MaxVal) return DAG.getConstant(1, dl, VT);  // X <= MAX --> true
      // X <= C0 --> X < (C0 + 1)
      APInt C = C1 + 1;
      ISD::CondCode NewCC = (Cond == ISD::SETLE) ? ISD::SETLT : ISD::SETULT;
      if ((DCI.isBeforeLegalizeOps() ||
           isCondCodeLegal(NewCC, VT.getSimpleVT())) &&
          (!N1C->isOpaque() || (N1C->isOpaque() && C.getBitWidth() <= 64 &&
                                isLegalICmpImmediate(C.getSExtValue())))) {
        return DAG.getSetCC(dl, VT, N0,
                            DAG.getConstant(C, dl, N1.getValueType()),
                            NewCC);
      }
    }

    if ((Cond == ISD::SETLT || Cond == ISD::SETULT) && C1 == MinVal)
      return DAG.getConstant(0, dl, VT);      // X < MIN --> false
    if ((Cond == ISD::SETGE || Cond == ISD::SETUGE) && C1 == MinVal)
      return DAG.getConstant(1, dl, VT);      // X >= MIN --> true
    if ((Cond == ISD::SETGT || Cond == ISD::SETUGT) && C1 == MaxVal)
      return DAG.getConstant(0, dl, VT);      // X > MAX --> false
    if ((Cond == ISD::SETLE || Cond == ISD::SETULE) && C1 == MaxVal)
      return DAG.getConstant(1, dl, VT);      // X <= MAX --> true

    // Canonicalize setgt X, Min --> setne X, Min
    if ((Cond == ISD::SETGT || Cond == ISD::SETUGT) && C1 == MinVal)
      return DAG.getSetCC(dl, VT, N0, N1, ISD::SETNE);
    // Canonicalize setlt X, Max --> setne X, Max
    if ((Cond == ISD::SETLT || Cond == ISD::SETULT) && C1 == MaxVal)
      return DAG.getSetCC(dl, VT, N0, N1, ISD::SETNE);

    // If we have setult X, 1, turn it into seteq X, 0
    if ((Cond == ISD::SETLT || Cond == ISD::SETULT) && C1 == MinVal+1)
      return DAG.getSetCC(dl, VT, N0,
                          DAG.getConstant(MinVal, dl, N0.getValueType()),
                          ISD::SETEQ);
    // If we have setugt X, Max-1, turn it into seteq X, Max
    if ((Cond == ISD::SETGT || Cond == ISD::SETUGT) && C1 == MaxVal-1)
      return DAG.getSetCC(dl, VT, N0,
                          DAG.getConstant(MaxVal, dl, N0.getValueType()),
                          ISD::SETEQ);

    // If we have "setcc X, C0", check to see if we can shrink the immediate
    // by changing cc.

    // SETUGT X, SINTMAX  -> SETLT X, 0
    if (Cond == ISD::SETUGT &&
        C1 == APInt::getSignedMaxValue(OperandBitSize))
      return DAG.getSetCC(dl, VT, N0,
                          DAG.getConstant(0, dl, N1.getValueType()),
                          ISD::SETLT);

    // SETULT X, SINTMIN  -> SETGT X, -1
    if (Cond == ISD::SETULT &&
        C1 == APInt::getSignedMinValue(OperandBitSize)) {
      SDValue ConstMinusOne =
          DAG.getConstant(APInt::getAllOnesValue(OperandBitSize), dl,
                          N1.getValueType());
      return DAG.getSetCC(dl, VT, N0, ConstMinusOne, ISD::SETGT);
    }

    // Fold bit comparisons when we can.
    if ((Cond == ISD::SETEQ || Cond == ISD::SETNE) &&
        (VT == N0.getValueType() ||
         (isTypeLegal(VT) && VT.bitsLE(N0.getValueType()))) &&
        N0.getOpcode() == ISD::AND) {
      auto &DL = DAG.getDataLayout();
      if (ConstantSDNode *AndRHS =
                  dyn_cast<ConstantSDNode>(N0.getOperand(1))) {
        EVT ShiftTy = DCI.isBeforeLegalize()
                          ? getPointerTy(DL)
                          : getShiftAmountTy(N0.getValueType(), DL);
        if (Cond == ISD::SETNE && C1 == 0) {// (X & 8) != 0  -->  (X & 8) >> 3
          // Perform the xform if the AND RHS is a single bit.
          if (AndRHS->getAPIntValue().isPowerOf2()) {
            return DAG.getNode(ISD::TRUNCATE, dl, VT,
                              DAG.getNode(ISD::SRL, dl, N0.getValueType(), N0,
                   DAG.getConstant(AndRHS->getAPIntValue().logBase2(), dl,
                                   ShiftTy)));
          }
        } else if (Cond == ISD::SETEQ && C1 == AndRHS->getAPIntValue()) {
          // (X & 8) == 8  -->  (X & 8) >> 3
          // Perform the xform if C1 is a single bit.
          if (C1.isPowerOf2()) {
            return DAG.getNode(ISD::TRUNCATE, dl, VT,
                               DAG.getNode(ISD::SRL, dl, N0.getValueType(), N0,
                                      DAG.getConstant(C1.logBase2(), dl,
                                                      ShiftTy)));
          }
        }
      }
    }

    if (C1.getMinSignedBits() <= 64 &&
        !isLegalICmpImmediate(C1.getSExtValue())) {
      // (X & -256) == 256 -> (X >> 8) == 1
      if ((Cond == ISD::SETEQ || Cond == ISD::SETNE) &&
          N0.getOpcode() == ISD::AND && N0.hasOneUse()) {
        if (ConstantSDNode *AndRHS =
            dyn_cast<ConstantSDNode>(N0.getOperand(1))) {
          const APInt &AndRHSC = AndRHS->getAPIntValue();
          if ((-AndRHSC).isPowerOf2() && (AndRHSC & C1) == C1) {
            unsigned ShiftBits = AndRHSC.countTrailingZeros();
            auto &DL = DAG.getDataLayout();
            EVT ShiftTy = DCI.isBeforeLegalize()
                              ? getPointerTy(DL)
                              : getShiftAmountTy(N0.getValueType(), DL);
            EVT CmpTy = N0.getValueType();
            SDValue Shift = DAG.getNode(ISD::SRL, dl, CmpTy, N0.getOperand(0),
                                        DAG.getConstant(ShiftBits, dl,
                                                        ShiftTy));
            SDValue CmpRHS = DAG.getConstant(C1.lshr(ShiftBits), dl, CmpTy);
            return DAG.getSetCC(dl, VT, Shift, CmpRHS, Cond);
          }
        }
      } else if (Cond == ISD::SETULT || Cond == ISD::SETUGE ||
                 Cond == ISD::SETULE || Cond == ISD::SETUGT) {
        bool AdjOne = (Cond == ISD::SETULE || Cond == ISD::SETUGT);
        // X <  0x100000000 -> (X >> 32) <  1
        // X >= 0x100000000 -> (X >> 32) >= 1
        // X <= 0x0ffffffff -> (X >> 32) <  1
        // X >  0x0ffffffff -> (X >> 32) >= 1
        unsigned ShiftBits;
        APInt NewC = C1;
        ISD::CondCode NewCond = Cond;
        if (AdjOne) {
          ShiftBits = C1.countTrailingOnes();
          NewC = NewC + 1;
          NewCond = (Cond == ISD::SETULE) ? ISD::SETULT : ISD::SETUGE;
        } else {
          ShiftBits = C1.countTrailingZeros();
        }
        NewC = NewC.lshr(ShiftBits);
        if (ShiftBits && NewC.getMinSignedBits() <= 64 &&
          isLegalICmpImmediate(NewC.getSExtValue())) {
          auto &DL = DAG.getDataLayout();
          EVT ShiftTy = DCI.isBeforeLegalize()
                            ? getPointerTy(DL)
                            : getShiftAmountTy(N0.getValueType(), DL);
          EVT CmpTy = N0.getValueType();
          SDValue Shift = DAG.getNode(ISD::SRL, dl, CmpTy, N0,
                                      DAG.getConstant(ShiftBits, dl, ShiftTy));
          SDValue CmpRHS = DAG.getConstant(NewC, dl, CmpTy);
          return DAG.getSetCC(dl, VT, Shift, CmpRHS, NewCond);
        }
      }
    }
  }

  if (isa<ConstantFPSDNode>(N0.getNode())) {
    // Constant fold or commute setcc.
    SDValue O = DAG.FoldSetCC(VT, N0, N1, Cond, dl);
    if (O.getNode()) return O;
  } else if (ConstantFPSDNode *CFP = dyn_cast<ConstantFPSDNode>(N1.getNode())) {
    // If the RHS of an FP comparison is a constant, simplify it away in
    // some cases.
    if (CFP->getValueAPF().isNaN()) {
      // If an operand is known to be a nan, we can fold it.
      switch (ISD::getUnorderedFlavor(Cond)) {
      default: llvm_unreachable("Unknown flavor!");
      case 0:  // Known false.
        return DAG.getConstant(0, dl, VT);
      case 1:  // Known true.
        return DAG.getConstant(1, dl, VT);
      case 2:  // Undefined.
        return DAG.getUNDEF(VT);
      }
    }

    // Otherwise, we know the RHS is not a NaN.  Simplify the node to drop the
    // constant if knowing that the operand is non-nan is enough.  We prefer to
    // have SETO(x,x) instead of SETO(x, 0.0) because this avoids having to
    // materialize 0.0.
    if (Cond == ISD::SETO || Cond == ISD::SETUO)
      return DAG.getSetCC(dl, VT, N0, N0, Cond);

    // If the condition is not legal, see if we can find an equivalent one
    // which is legal.
    if (!isCondCodeLegal(Cond, N0.getSimpleValueType())) {
      // If the comparison was an awkward floating-point == or != and one of
      // the comparison operands is infinity or negative infinity, convert the
      // condition to a less-awkward <= or >=.
      if (CFP->getValueAPF().isInfinity()) {
        if (CFP->getValueAPF().isNegative()) {
          if (Cond == ISD::SETOEQ &&
              isCondCodeLegal(ISD::SETOLE, N0.getSimpleValueType()))
            return DAG.getSetCC(dl, VT, N0, N1, ISD::SETOLE);
          if (Cond == ISD::SETUEQ &&
              isCondCodeLegal(ISD::SETOLE, N0.getSimpleValueType()))
            return DAG.getSetCC(dl, VT, N0, N1, ISD::SETULE);
          if (Cond == ISD::SETUNE &&
              isCondCodeLegal(ISD::SETUGT, N0.getSimpleValueType()))
            return DAG.getSetCC(dl, VT, N0, N1, ISD::SETUGT);
          if (Cond == ISD::SETONE &&
              isCondCodeLegal(ISD::SETUGT, N0.getSimpleValueType()))
            return DAG.getSetCC(dl, VT, N0, N1, ISD::SETOGT);
        } else {
          if (Cond == ISD::SETOEQ &&
              isCondCodeLegal(ISD::SETOGE, N0.getSimpleValueType()))
            return DAG.getSetCC(dl, VT, N0, N1, ISD::SETOGE);
          if (Cond == ISD::SETUEQ &&
              isCondCodeLegal(ISD::SETOGE, N0.getSimpleValueType()))
            return DAG.getSetCC(dl, VT, N0, N1, ISD::SETUGE);
          if (Cond == ISD::SETUNE &&
              isCondCodeLegal(ISD::SETULT, N0.getSimpleValueType()))
            return DAG.getSetCC(dl, VT, N0, N1, ISD::SETULT);
          if (Cond == ISD::SETONE &&
              isCondCodeLegal(ISD::SETULT, N0.getSimpleValueType()))
            return DAG.getSetCC(dl, VT, N0, N1, ISD::SETOLT);
        }
      }
    }
  }

  if (N0 == N1) {
    // The sext(setcc()) => setcc() optimization relies on the appropriate
    // constant being emitted.
    uint64_t EqVal = 0;
    switch (getBooleanContents(N0.getValueType())) {
    case UndefinedBooleanContent:
    case ZeroOrOneBooleanContent:
      EqVal = ISD::isTrueWhenEqual(Cond);
      break;
    case ZeroOrNegativeOneBooleanContent:
      EqVal = ISD::isTrueWhenEqual(Cond) ? -1 : 0;
      break;
    }

    // We can always fold X == X for integer setcc's.
    if (N0.getValueType().isInteger()) {
      return DAG.getConstant(EqVal, dl, VT);
    }
    unsigned UOF = ISD::getUnorderedFlavor(Cond);
    if (UOF == 2)   // FP operators that are undefined on NaNs.
      return DAG.getConstant(EqVal, dl, VT);
    if (UOF == unsigned(ISD::isTrueWhenEqual(Cond)))
      return DAG.getConstant(EqVal, dl, VT);
    // Otherwise, we can't fold it.  However, we can simplify it to SETUO/SETO
    // if it is not already.
    ISD::CondCode NewCond = UOF == 0 ? ISD::SETO : ISD::SETUO;
    if (NewCond != Cond && (DCI.isBeforeLegalizeOps() ||
          getCondCodeAction(NewCond, N0.getSimpleValueType()) == Legal))
      return DAG.getSetCC(dl, VT, N0, N1, NewCond);
  }

  if ((Cond == ISD::SETEQ || Cond == ISD::SETNE) &&
      N0.getValueType().isInteger()) {
    if (N0.getOpcode() == ISD::ADD || N0.getOpcode() == ISD::SUB ||
        N0.getOpcode() == ISD::XOR) {
      // Simplify (X+Y) == (X+Z) -->  Y == Z
      if (N0.getOpcode() == N1.getOpcode()) {
        if (N0.getOperand(0) == N1.getOperand(0))
          return DAG.getSetCC(dl, VT, N0.getOperand(1), N1.getOperand(1), Cond);
        if (N0.getOperand(1) == N1.getOperand(1))
          return DAG.getSetCC(dl, VT, N0.getOperand(0), N1.getOperand(0), Cond);
        if (DAG.isCommutativeBinOp(N0.getOpcode())) {
          // If X op Y == Y op X, try other combinations.
          if (N0.getOperand(0) == N1.getOperand(1))
            return DAG.getSetCC(dl, VT, N0.getOperand(1), N1.getOperand(0),
                                Cond);
          if (N0.getOperand(1) == N1.getOperand(0))
            return DAG.getSetCC(dl, VT, N0.getOperand(0), N1.getOperand(1),
                                Cond);
        }
      }

      // If RHS is a legal immediate value for a compare instruction, we need
      // to be careful about increasing register pressure needlessly.
      bool LegalRHSImm = false;

      if (ConstantSDNode *RHSC = dyn_cast<ConstantSDNode>(N1)) {
        if (ConstantSDNode *LHSR = dyn_cast<ConstantSDNode>(N0.getOperand(1))) {
          // Turn (X+C1) == C2 --> X == C2-C1
          if (N0.getOpcode() == ISD::ADD && N0.getNode()->hasOneUse()) {
            return DAG.getSetCC(dl, VT, N0.getOperand(0),
                                DAG.getConstant(RHSC->getAPIntValue()-
                                                LHSR->getAPIntValue(),
                                dl, N0.getValueType()), Cond);
          }

          // Turn (X^C1) == C2 into X == C1^C2 iff X&~C1 = 0.
          if (N0.getOpcode() == ISD::XOR)
            // If we know that all of the inverted bits are zero, don't bother
            // performing the inversion.
            if (DAG.MaskedValueIsZero(N0.getOperand(0), ~LHSR->getAPIntValue()))
              return
                DAG.getSetCC(dl, VT, N0.getOperand(0),
                             DAG.getConstant(LHSR->getAPIntValue() ^
                                               RHSC->getAPIntValue(),
                                             dl, N0.getValueType()),
                             Cond);
        }

        // Turn (C1-X) == C2 --> X == C1-C2
        if (ConstantSDNode *SUBC = dyn_cast<ConstantSDNode>(N0.getOperand(0))) {
          if (N0.getOpcode() == ISD::SUB && N0.getNode()->hasOneUse()) {
            return
              DAG.getSetCC(dl, VT, N0.getOperand(1),
                           DAG.getConstant(SUBC->getAPIntValue() -
                                             RHSC->getAPIntValue(),
                                           dl, N0.getValueType()),
                           Cond);
          }
        }

        // Could RHSC fold directly into a compare?
        if (RHSC->getValueType(0).getSizeInBits() <= 64)
          LegalRHSImm = isLegalICmpImmediate(RHSC->getSExtValue());
      }

      // Simplify (X+Z) == X -->  Z == 0
      // Don't do this if X is an immediate that can fold into a cmp
      // instruction and X+Z has other uses. It could be an induction variable
      // chain, and the transform would increase register pressure.
      if (!LegalRHSImm || N0.getNode()->hasOneUse()) {
        if (N0.getOperand(0) == N1)
          return DAG.getSetCC(dl, VT, N0.getOperand(1),
                              DAG.getConstant(0, dl, N0.getValueType()), Cond);
        if (N0.getOperand(1) == N1) {
          if (DAG.isCommutativeBinOp(N0.getOpcode()))
            return DAG.getSetCC(dl, VT, N0.getOperand(0),
                                DAG.getConstant(0, dl, N0.getValueType()),
                                Cond);
          if (N0.getNode()->hasOneUse()) {
            assert(N0.getOpcode() == ISD::SUB && "Unexpected operation!");
            auto &DL = DAG.getDataLayout();
            // (Z-X) == X  --> Z == X<<1
            SDValue SH = DAG.getNode(
                ISD::SHL, dl, N1.getValueType(), N1,
                DAG.getConstant(1, dl,
                                getShiftAmountTy(N1.getValueType(), DL)));
            if (!DCI.isCalledByLegalizer())
              DCI.AddToWorklist(SH.getNode());
            return DAG.getSetCC(dl, VT, N0.getOperand(0), SH, Cond);
          }
        }
      }
    }

    if (N1.getOpcode() == ISD::ADD || N1.getOpcode() == ISD::SUB ||
        N1.getOpcode() == ISD::XOR) {
      // Simplify  X == (X+Z) -->  Z == 0
      if (N1.getOperand(0) == N0)
        return DAG.getSetCC(dl, VT, N1.getOperand(1),
                        DAG.getConstant(0, dl, N1.getValueType()), Cond);
      if (N1.getOperand(1) == N0) {
        if (DAG.isCommutativeBinOp(N1.getOpcode()))
          return DAG.getSetCC(dl, VT, N1.getOperand(0),
                          DAG.getConstant(0, dl, N1.getValueType()), Cond);
        if (N1.getNode()->hasOneUse()) {
          assert(N1.getOpcode() == ISD::SUB && "Unexpected operation!");
          auto &DL = DAG.getDataLayout();
          // X == (Z-X)  --> X<<1 == Z
          SDValue SH = DAG.getNode(
              ISD::SHL, dl, N1.getValueType(), N0,
              DAG.getConstant(1, dl, getShiftAmountTy(N0.getValueType(), DL)));
          if (!DCI.isCalledByLegalizer())
            DCI.AddToWorklist(SH.getNode());
          return DAG.getSetCC(dl, VT, SH, N1.getOperand(0), Cond);
        }
      }
    }

    // Simplify x&y == y to x&y != 0 if y has exactly one bit set.
    // Note that where y is variable and is known to have at most
    // one bit set (for example, if it is z&1) we cannot do this;
    // the expressions are not equivalent when y==0.
    if (N0.getOpcode() == ISD::AND)
      if (N0.getOperand(0) == N1 || N0.getOperand(1) == N1) {
        if (ValueHasExactlyOneBitSet(N1, DAG)) {
          Cond = ISD::getSetCCInverse(Cond, /*isInteger=*/true);
          if (DCI.isBeforeLegalizeOps() ||
              isCondCodeLegal(Cond, N0.getSimpleValueType())) {
            SDValue Zero = DAG.getConstant(0, dl, N1.getValueType());
            return DAG.getSetCC(dl, VT, N0, Zero, Cond);
          }
        }
      }
    if (N1.getOpcode() == ISD::AND)
      if (N1.getOperand(0) == N0 || N1.getOperand(1) == N0) {
        if (ValueHasExactlyOneBitSet(N0, DAG)) {
          Cond = ISD::getSetCCInverse(Cond, /*isInteger=*/true);
          if (DCI.isBeforeLegalizeOps() ||
              isCondCodeLegal(Cond, N1.getSimpleValueType())) {
            SDValue Zero = DAG.getConstant(0, dl, N0.getValueType());
            return DAG.getSetCC(dl, VT, N1, Zero, Cond);
          }
        }
      }
  }

  // Fold away ALL boolean setcc's.
  SDValue Temp;
  if (N0.getValueType() == MVT::i1 && foldBooleans) {
    switch (Cond) {
    default: llvm_unreachable("Unknown integer setcc!");
    case ISD::SETEQ:  // X == Y  -> ~(X^Y)
      Temp = DAG.getNode(ISD::XOR, dl, MVT::i1, N0, N1);
      N0 = DAG.getNOT(dl, Temp, MVT::i1);
      if (!DCI.isCalledByLegalizer())
        DCI.AddToWorklist(Temp.getNode());
      break;
    case ISD::SETNE:  // X != Y   -->  (X^Y)
      N0 = DAG.getNode(ISD::XOR, dl, MVT::i1, N0, N1);
      break;
    case ISD::SETGT:  // X >s Y   -->  X == 0 & Y == 1  -->  ~X & Y
    case ISD::SETULT: // X <u Y   -->  X == 0 & Y == 1  -->  ~X & Y
      Temp = DAG.getNOT(dl, N0, MVT::i1);
      N0 = DAG.getNode(ISD::AND, dl, MVT::i1, N1, Temp);
      if (!DCI.isCalledByLegalizer())
        DCI.AddToWorklist(Temp.getNode());
      break;
    case ISD::SETLT:  // X <s Y   --> X == 1 & Y == 0  -->  ~Y & X
    case ISD::SETUGT: // X >u Y   --> X == 1 & Y == 0  -->  ~Y & X
      Temp = DAG.getNOT(dl, N1, MVT::i1);
      N0 = DAG.getNode(ISD::AND, dl, MVT::i1, N0, Temp);
      if (!DCI.isCalledByLegalizer())
        DCI.AddToWorklist(Temp.getNode());
      break;
    case ISD::SETULE: // X <=u Y  --> X == 0 | Y == 1  -->  ~X | Y
    case ISD::SETGE:  // X >=s Y  --> X == 0 | Y == 1  -->  ~X | Y
      Temp = DAG.getNOT(dl, N0, MVT::i1);
      N0 = DAG.getNode(ISD::OR, dl, MVT::i1, N1, Temp);
      if (!DCI.isCalledByLegalizer())
        DCI.AddToWorklist(Temp.getNode());
      break;
    case ISD::SETUGE: // X >=u Y  --> X == 1 | Y == 0  -->  ~Y | X
    case ISD::SETLE:  // X <=s Y  --> X == 1 | Y == 0  -->  ~Y | X
      Temp = DAG.getNOT(dl, N1, MVT::i1);
      N0 = DAG.getNode(ISD::OR, dl, MVT::i1, N0, Temp);
      break;
    }
    if (VT != MVT::i1) {
      if (!DCI.isCalledByLegalizer())
        DCI.AddToWorklist(N0.getNode());
      // FIXME: If running after legalize, we probably can't do this.
      N0 = DAG.getNode(ISD::ZERO_EXTEND, dl, VT, N0);
    }
    return N0;
  }

  // Could not fold it.
  return SDValue();
}

/// isGAPlusOffset - Returns true (and the GlobalValue and the offset) if the
/// node is a GlobalAddress + offset.
bool TargetLowering::isGAPlusOffset(SDNode *N, const GlobalValue *&GA,
                                    int64_t &Offset) const {
  if (isa<GlobalAddressSDNode>(N)) {
    GlobalAddressSDNode *GASD = cast<GlobalAddressSDNode>(N);
    GA = GASD->getGlobal();
    Offset += GASD->getOffset();
    return true;
  }

  if (N->getOpcode() == ISD::ADD) {
    SDValue N1 = N->getOperand(0);
    SDValue N2 = N->getOperand(1);
    if (isGAPlusOffset(N1.getNode(), GA, Offset)) {
      ConstantSDNode *V = dyn_cast<ConstantSDNode>(N2);
      if (V) {
        Offset += V->getSExtValue();
        return true;
      }
    } else if (isGAPlusOffset(N2.getNode(), GA, Offset)) {
      ConstantSDNode *V = dyn_cast<ConstantSDNode>(N1);
      if (V) {
        Offset += V->getSExtValue();
        return true;
      }
    }
  }

  return false;
}


SDValue TargetLowering::
PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI) const {
  // Default implementation: no optimization.
  return SDValue();
}

//===----------------------------------------------------------------------===//
//  Inline Assembler Implementation Methods
//===----------------------------------------------------------------------===//

TargetLowering::ConstraintType
TargetLowering::getConstraintType(StringRef Constraint) const {
  unsigned S = Constraint.size();

  if (S == 1) {
    switch (Constraint[0]) {
    default: break;
    case 'r': return C_RegisterClass;
    case 'm':    // memory
    case 'o':    // offsetable
    case 'V':    // not offsetable
      return C_Memory;
    case 'i':    // Simple Integer or Relocatable Constant
    case 'n':    // Simple Integer
    case 'E':    // Floating Point Constant
    case 'F':    // Floating Point Constant
    case 's':    // Relocatable Constant
    case 'p':    // Address.
    case 'X':    // Allow ANY value.
    case 'I':    // Target registers.
    case 'J':
    case 'K':
    case 'L':
    case 'M':
    case 'N':
    case 'O':
    case 'P':
    case '<':
    case '>':
      return C_Other;
    }
  }

  if (S > 1 && Constraint[0] == '{' && Constraint[S-1] == '}') {
    if (S == 8 && Constraint.substr(1, 6) == "memory") // "{memory}"
      return C_Memory;
    return C_Register;
  }
  return C_Unknown;
}

/// LowerXConstraint - try to replace an X constraint, which matches anything,
/// with another that has more specific requirements based on the type of the
/// corresponding operand.
const char *TargetLowering::LowerXConstraint(EVT ConstraintVT) const{
  if (ConstraintVT.isInteger())
    return "r";
  if (ConstraintVT.isFloatingPoint())
    return "f";      // works for many targets
  return nullptr;
}

/// LowerAsmOperandForConstraint - Lower the specified operand into the Ops
/// vector.  If it is invalid, don't add anything to Ops.
void TargetLowering::LowerAsmOperandForConstraint(SDValue Op,
                                                  std::string &Constraint,
                                                  std::vector<SDValue> &Ops,
                                                  SelectionDAG &DAG) const {

  if (Constraint.length() > 1) return;

  char ConstraintLetter = Constraint[0];
  switch (ConstraintLetter) {
  default: break;
  case 'X':     // Allows any operand; labels (basic block) use this.
    if (Op.getOpcode() == ISD::BasicBlock) {
      Ops.push_back(Op);
      return;
    }
    // fall through
  case 'i':    // Simple Integer or Relocatable Constant
  case 'n':    // Simple Integer
  case 's': {  // Relocatable Constant
    // These operands are interested in values of the form (GV+C), where C may
    // be folded in as an offset of GV, or it may be explicitly added.  Also, it
    // is possible and fine if either GV or C are missing.
    ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op);
    GlobalAddressSDNode *GA = dyn_cast<GlobalAddressSDNode>(Op);

    // If we have "(add GV, C)", pull out GV/C
    if (Op.getOpcode() == ISD::ADD) {
      C = dyn_cast<ConstantSDNode>(Op.getOperand(1));
      GA = dyn_cast<GlobalAddressSDNode>(Op.getOperand(0));
      if (!C || !GA) {
        C = dyn_cast<ConstantSDNode>(Op.getOperand(0));
        GA = dyn_cast<GlobalAddressSDNode>(Op.getOperand(1));
      }
      if (!C || !GA)
        C = nullptr, GA = nullptr;
    }

    // If we find a valid operand, map to the TargetXXX version so that the
    // value itself doesn't get selected.
    if (GA) {   // Either &GV   or   &GV+C
      if (ConstraintLetter != 'n') {
        int64_t Offs = GA->getOffset();
        if (C) Offs += C->getZExtValue();
        Ops.push_back(DAG.getTargetGlobalAddress(GA->getGlobal(),
                                                 C ? SDLoc(C) : SDLoc(),
                                                 Op.getValueType(), Offs));
      }
      return;
    }
    if (C) {   // just C, no GV.
      // Simple constants are not allowed for 's'.
      if (ConstraintLetter != 's') {
        // gcc prints these as sign extended.  Sign extend value to 64 bits
        // now; without this it would get ZExt'd later in
        // ScheduleDAGSDNodes::EmitNode, which is very generic.
        Ops.push_back(DAG.getTargetConstant(C->getAPIntValue().getSExtValue(),
                                            SDLoc(C), MVT::i64));
      }
      return;
    }
    break;
  }
  }
}

std::pair<unsigned, const TargetRegisterClass *>
TargetLowering::getRegForInlineAsmConstraint(const TargetRegisterInfo *RI,
                                             StringRef Constraint,
                                             MVT VT) const {
  if (Constraint.empty() || Constraint[0] != '{')
    return std::make_pair(0u, static_cast<TargetRegisterClass*>(nullptr));
  assert(*(Constraint.end()-1) == '}' && "Not a brace enclosed constraint?");

  // Remove the braces from around the name.
  StringRef RegName(Constraint.data()+1, Constraint.size()-2);

  std::pair<unsigned, const TargetRegisterClass*> R =
    std::make_pair(0u, static_cast<const TargetRegisterClass*>(nullptr));

  // Figure out which register class contains this reg.
  for (TargetRegisterInfo::regclass_iterator RCI = RI->regclass_begin(),
       E = RI->regclass_end(); RCI != E; ++RCI) {
    const TargetRegisterClass *RC = *RCI;

    // If none of the value types for this register class are valid, we
    // can't use it.  For example, 64-bit reg classes on 32-bit targets.
    if (!isLegalRC(RC))
      continue;

    for (TargetRegisterClass::iterator I = RC->begin(), E = RC->end();
         I != E; ++I) {
      if (RegName.equals_lower(RI->getName(*I))) {
        std::pair<unsigned, const TargetRegisterClass*> S =
          std::make_pair(*I, RC);

        // If this register class has the requested value type, return it,
        // otherwise keep searching and return the first class found
        // if no other is found which explicitly has the requested type.
        if (RC->hasType(VT))
          return S;
        else if (!R.second)
          R = S;
      }
    }
  }

  return R;
}

//===----------------------------------------------------------------------===//
// Constraint Selection.

/// isMatchingInputConstraint - Return true of this is an input operand that is
/// a matching constraint like "4".
bool TargetLowering::AsmOperandInfo::isMatchingInputConstraint() const {
  assert(!ConstraintCode.empty() && "No known constraint!");
  return isdigit(static_cast<unsigned char>(ConstraintCode[0]));
}

/// getMatchedOperand - If this is an input matching constraint, this method
/// returns the output operand it matches.
unsigned TargetLowering::AsmOperandInfo::getMatchedOperand() const {
  assert(!ConstraintCode.empty() && "No known constraint!");
  return atoi(ConstraintCode.c_str());
}


/// ParseConstraints - Split up the constraint string from the inline
/// assembly value into the specific constraints and their prefixes,
/// and also tie in the associated operand values.
/// If this returns an empty vector, and if the constraint string itself
/// isn't empty, there was an error parsing.
TargetLowering::AsmOperandInfoVector
TargetLowering::ParseConstraints(const DataLayout &DL,
                                 const TargetRegisterInfo *TRI,
                                 ImmutableCallSite CS) const {
  /// ConstraintOperands - Information about all of the constraints.
  AsmOperandInfoVector ConstraintOperands;
  const InlineAsm *IA = cast<InlineAsm>(CS.getCalledValue());
  unsigned maCount = 0; // Largest number of multiple alternative constraints.

  // Do a prepass over the constraints, canonicalizing them, and building up the
  // ConstraintOperands list.
  unsigned ArgNo = 0;   // ArgNo - The argument of the CallInst.
  unsigned ResNo = 0;   // ResNo - The result number of the next output.

  for (InlineAsm::ConstraintInfo &CI : IA->ParseConstraints()) {
    ConstraintOperands.emplace_back(std::move(CI));
    AsmOperandInfo &OpInfo = ConstraintOperands.back();

    // Update multiple alternative constraint count.
    if (OpInfo.multipleAlternatives.size() > maCount)
      maCount = OpInfo.multipleAlternatives.size();

    OpInfo.ConstraintVT = MVT::Other;

    // Compute the value type for each operand.
    switch (OpInfo.Type) {
    case InlineAsm::isOutput:
      // Indirect outputs just consume an argument.
      if (OpInfo.isIndirect) {
        OpInfo.CallOperandVal = const_cast<Value *>(CS.getArgument(ArgNo++));
        break;
      }

      // The return value of the call is this value.  As such, there is no
      // corresponding argument.
      assert(!CS.getType()->isVoidTy() &&
             "Bad inline asm!");
      if (StructType *STy = dyn_cast<StructType>(CS.getType())) {
        OpInfo.ConstraintVT =
            getSimpleValueType(DL, STy->getElementType(ResNo));
      } else {
        assert(ResNo == 0 && "Asm only has one result!");
        OpInfo.ConstraintVT = getSimpleValueType(DL, CS.getType());
      }
      ++ResNo;
      break;
    case InlineAsm::isInput:
      OpInfo.CallOperandVal = const_cast<Value *>(CS.getArgument(ArgNo++));
      break;
    case InlineAsm::isClobber:
      // Nothing to do.
      break;
    }

    if (OpInfo.CallOperandVal) {
      llvm::Type *OpTy = OpInfo.CallOperandVal->getType();
      if (OpInfo.isIndirect) {
        llvm::PointerType *PtrTy = dyn_cast<PointerType>(OpTy);
        if (!PtrTy)
          report_fatal_error("Indirect operand for inline asm not a pointer!");
        OpTy = PtrTy->getElementType();
      }

      // Look for vector wrapped in a struct. e.g. { <16 x i8> }.
      if (StructType *STy = dyn_cast<StructType>(OpTy))
        if (STy->getNumElements() == 1)
          OpTy = STy->getElementType(0);

      // If OpTy is not a single value, it may be a struct/union that we
      // can tile with integers.
      if (!OpTy->isSingleValueType() && OpTy->isSized()) {
        unsigned BitSize = DL.getTypeSizeInBits(OpTy);
        switch (BitSize) {
        default: break;
        case 1:
        case 8:
        case 16:
        case 32:
        case 64:
        case 128:
          OpInfo.ConstraintVT =
            MVT::getVT(IntegerType::get(OpTy->getContext(), BitSize), true);
          break;
        }
      } else if (PointerType *PT = dyn_cast<PointerType>(OpTy)) {
        unsigned PtrSize = DL.getPointerSizeInBits(PT->getAddressSpace());
        OpInfo.ConstraintVT = MVT::getIntegerVT(PtrSize);
      } else {
        OpInfo.ConstraintVT = MVT::getVT(OpTy, true);
      }
    }
  }

  // If we have multiple alternative constraints, select the best alternative.
  if (!ConstraintOperands.empty()) {
    if (maCount) {
      unsigned bestMAIndex = 0;
      int bestWeight = -1;
      // weight:  -1 = invalid match, and 0 = so-so match to 5 = good match.
      int weight = -1;
      unsigned maIndex;
      // Compute the sums of the weights for each alternative, keeping track
      // of the best (highest weight) one so far.
      for (maIndex = 0; maIndex < maCount; ++maIndex) {
        int weightSum = 0;
        for (unsigned cIndex = 0, eIndex = ConstraintOperands.size();
            cIndex != eIndex; ++cIndex) {
          AsmOperandInfo& OpInfo = ConstraintOperands[cIndex];
          if (OpInfo.Type == InlineAsm::isClobber)
            continue;

          // If this is an output operand with a matching input operand,
          // look up the matching input. If their types mismatch, e.g. one
          // is an integer, the other is floating point, or their sizes are
          // different, flag it as an maCantMatch.
          if (OpInfo.hasMatchingInput()) {
            AsmOperandInfo &Input = ConstraintOperands[OpInfo.MatchingInput];
            if (OpInfo.ConstraintVT != Input.ConstraintVT) {
              if ((OpInfo.ConstraintVT.isInteger() !=
                   Input.ConstraintVT.isInteger()) ||
                  (OpInfo.ConstraintVT.getSizeInBits() !=
                   Input.ConstraintVT.getSizeInBits())) {
                weightSum = -1;  // Can't match.
                break;
              }
            }
          }
          weight = getMultipleConstraintMatchWeight(OpInfo, maIndex);
          if (weight == -1) {
            weightSum = -1;
            break;
          }
          weightSum += weight;
        }
        // Update best.
        if (weightSum > bestWeight) {
          bestWeight = weightSum;
          bestMAIndex = maIndex;
        }
      }

      // Now select chosen alternative in each constraint.
      for (unsigned cIndex = 0, eIndex = ConstraintOperands.size();
          cIndex != eIndex; ++cIndex) {
        AsmOperandInfo& cInfo = ConstraintOperands[cIndex];
        if (cInfo.Type == InlineAsm::isClobber)
          continue;
        cInfo.selectAlternative(bestMAIndex);
      }
    }
  }

  // Check and hook up tied operands, choose constraint code to use.
  for (unsigned cIndex = 0, eIndex = ConstraintOperands.size();
      cIndex != eIndex; ++cIndex) {
    AsmOperandInfo& OpInfo = ConstraintOperands[cIndex];

    // If this is an output operand with a matching input operand, look up the
    // matching input. If their types mismatch, e.g. one is an integer, the
    // other is floating point, or their sizes are different, flag it as an
    // error.
    if (OpInfo.hasMatchingInput()) {
      AsmOperandInfo &Input = ConstraintOperands[OpInfo.MatchingInput];

      if (OpInfo.ConstraintVT != Input.ConstraintVT) {
        std::pair<unsigned, const TargetRegisterClass *> MatchRC =
            getRegForInlineAsmConstraint(TRI, OpInfo.ConstraintCode,
                                         OpInfo.ConstraintVT);
        std::pair<unsigned, const TargetRegisterClass *> InputRC =
            getRegForInlineAsmConstraint(TRI, Input.ConstraintCode,
                                         Input.ConstraintVT);
        if ((OpInfo.ConstraintVT.isInteger() !=
             Input.ConstraintVT.isInteger()) ||
            (MatchRC.second != InputRC.second)) {
          report_fatal_error("Unsupported asm: input constraint"
                             " with a matching output constraint of"
                             " incompatible type!");
        }
      }

    }
  }

  return ConstraintOperands;
}


/// getConstraintGenerality - Return an integer indicating how general CT
/// is.
static unsigned getConstraintGenerality(TargetLowering::ConstraintType CT) {
  switch (CT) {
  case TargetLowering::C_Other:
  case TargetLowering::C_Unknown:
    return 0;
  case TargetLowering::C_Register:
    return 1;
  case TargetLowering::C_RegisterClass:
    return 2;
  case TargetLowering::C_Memory:
    return 3;
  }
  llvm_unreachable("Invalid constraint type");
}

/// Examine constraint type and operand type and determine a weight value.
/// This object must already have been set up with the operand type
/// and the current alternative constraint selected.
TargetLowering::ConstraintWeight
  TargetLowering::getMultipleConstraintMatchWeight(
    AsmOperandInfo &info, int maIndex) const {
  InlineAsm::ConstraintCodeVector *rCodes;
  if (maIndex >= (int)info.multipleAlternatives.size())
    rCodes = &info.Codes;
  else
    rCodes = &info.multipleAlternatives[maIndex].Codes;
  ConstraintWeight BestWeight = CW_Invalid;

  // Loop over the options, keeping track of the most general one.
  for (unsigned i = 0, e = rCodes->size(); i != e; ++i) {
    ConstraintWeight weight =
      getSingleConstraintMatchWeight(info, (*rCodes)[i].c_str());
    if (weight > BestWeight)
      BestWeight = weight;
  }

  return BestWeight;
}

/// Examine constraint type and operand type and determine a weight value.
/// This object must already have been set up with the operand type
/// and the current alternative constraint selected.
TargetLowering::ConstraintWeight
  TargetLowering::getSingleConstraintMatchWeight(
    AsmOperandInfo &info, const char *constraint) const {
  ConstraintWeight weight = CW_Invalid;
  Value *CallOperandVal = info.CallOperandVal;
    // If we don't have a value, we can't do a match,
    // but allow it at the lowest weight.
  if (!CallOperandVal)
    return CW_Default;
  // Look at the constraint type.
  switch (*constraint) {
    case 'i': // immediate integer.
    case 'n': // immediate integer with a known value.
      if (isa<ConstantInt>(CallOperandVal))
        weight = CW_Constant;
      break;
    case 's': // non-explicit intregal immediate.
      if (isa<GlobalValue>(CallOperandVal))
        weight = CW_Constant;
      break;
    case 'E': // immediate float if host format.
    case 'F': // immediate float.
      if (isa<ConstantFP>(CallOperandVal))
        weight = CW_Constant;
      break;
    case '<': // memory operand with autodecrement.
    case '>': // memory operand with autoincrement.
    case 'm': // memory operand.
    case 'o': // offsettable memory operand
    case 'V': // non-offsettable memory operand
      weight = CW_Memory;
      break;
    case 'r': // general register.
    case 'g': // general register, memory operand or immediate integer.
              // note: Clang converts "g" to "imr".
      if (CallOperandVal->getType()->isIntegerTy())
        weight = CW_Register;
      break;
    case 'X': // any operand.
    default:
      weight = CW_Default;
      break;
  }
  return weight;
}

/// ChooseConstraint - If there are multiple different constraints that we
/// could pick for this operand (e.g. "imr") try to pick the 'best' one.
/// This is somewhat tricky: constraints fall into four classes:
///    Other         -> immediates and magic values
///    Register      -> one specific register
///    RegisterClass -> a group of regs
///    Memory        -> memory
/// Ideally, we would pick the most specific constraint possible: if we have
/// something that fits into a register, we would pick it.  The problem here
/// is that if we have something that could either be in a register or in
/// memory that use of the register could cause selection of *other*
/// operands to fail: they might only succeed if we pick memory.  Because of
/// this the heuristic we use is:
///
///  1) If there is an 'other' constraint, and if the operand is valid for
///     that constraint, use it.  This makes us take advantage of 'i'
///     constraints when available.
///  2) Otherwise, pick the most general constraint present.  This prefers
///     'm' over 'r', for example.
///
static void ChooseConstraint(TargetLowering::AsmOperandInfo &OpInfo,
                             const TargetLowering &TLI,
                             SDValue Op, SelectionDAG *DAG) {
  assert(OpInfo.Codes.size() > 1 && "Doesn't have multiple constraint options");
  unsigned BestIdx = 0;
  TargetLowering::ConstraintType BestType = TargetLowering::C_Unknown;
  int BestGenerality = -1;

  // Loop over the options, keeping track of the most general one.
  for (unsigned i = 0, e = OpInfo.Codes.size(); i != e; ++i) {
    TargetLowering::ConstraintType CType =
      TLI.getConstraintType(OpInfo.Codes[i]);

    // If this is an 'other' constraint, see if the operand is valid for it.
    // For example, on X86 we might have an 'rI' constraint.  If the operand
    // is an integer in the range [0..31] we want to use I (saving a load
    // of a register), otherwise we must use 'r'.
    if (CType == TargetLowering::C_Other && Op.getNode()) {
      assert(OpInfo.Codes[i].size() == 1 &&
             "Unhandled multi-letter 'other' constraint");
      std::vector<SDValue> ResultOps;
      TLI.LowerAsmOperandForConstraint(Op, OpInfo.Codes[i],
                                       ResultOps, *DAG);
      if (!ResultOps.empty()) {
        BestType = CType;
        BestIdx = i;
        break;
      }
    }

    // Things with matching constraints can only be registers, per gcc
    // documentation.  This mainly affects "g" constraints.
    if (CType == TargetLowering::C_Memory && OpInfo.hasMatchingInput())
      continue;

    // This constraint letter is more general than the previous one, use it.
    int Generality = getConstraintGenerality(CType);
    if (Generality > BestGenerality) {
      BestType = CType;
      BestIdx = i;
      BestGenerality = Generality;
    }
  }

  OpInfo.ConstraintCode = OpInfo.Codes[BestIdx];
  OpInfo.ConstraintType = BestType;
}

/// ComputeConstraintToUse - Determines the constraint code and constraint
/// type to use for the specific AsmOperandInfo, setting
/// OpInfo.ConstraintCode and OpInfo.ConstraintType.
void TargetLowering::ComputeConstraintToUse(AsmOperandInfo &OpInfo,
                                            SDValue Op,
                                            SelectionDAG *DAG) const {
  assert(!OpInfo.Codes.empty() && "Must have at least one constraint");

  // Single-letter constraints ('r') are very common.
  if (OpInfo.Codes.size() == 1) {
    OpInfo.ConstraintCode = OpInfo.Codes[0];
    OpInfo.ConstraintType = getConstraintType(OpInfo.ConstraintCode);
  } else {
    ChooseConstraint(OpInfo, *this, Op, DAG);
  }

  // 'X' matches anything.
  if (OpInfo.ConstraintCode == "X" && OpInfo.CallOperandVal) {
    // Labels and constants are handled elsewhere ('X' is the only thing
    // that matches labels).  For Functions, the type here is the type of
    // the result, which is not what we want to look at; leave them alone.
    Value *v = OpInfo.CallOperandVal;
    if (isa<BasicBlock>(v) || isa<ConstantInt>(v) || isa<Function>(v)) {
      OpInfo.CallOperandVal = v;
      return;
    }

    // Otherwise, try to resolve it to something we know about by looking at
    // the actual operand type.
    if (const char *Repl = LowerXConstraint(OpInfo.ConstraintVT)) {
      OpInfo.ConstraintCode = Repl;
      OpInfo.ConstraintType = getConstraintType(OpInfo.ConstraintCode);
    }
  }
}

/// \brief Given an exact SDIV by a constant, create a multiplication
/// with the multiplicative inverse of the constant.
static SDValue BuildExactSDIV(const TargetLowering &TLI, SDValue Op1, APInt d,
                              SDLoc dl, SelectionDAG &DAG,
                              std::vector<SDNode *> &Created) {
  assert(d != 0 && "Division by zero!");

  // Shift the value upfront if it is even, so the LSB is one.
  unsigned ShAmt = d.countTrailingZeros();
  if (ShAmt) {
    // TODO: For UDIV use SRL instead of SRA.
    SDValue Amt =
        DAG.getConstant(ShAmt, dl, TLI.getShiftAmountTy(Op1.getValueType(),
                                                        DAG.getDataLayout()));
    SDNodeFlags Flags;
    Flags.setExact(true);
    Op1 = DAG.getNode(ISD::SRA, dl, Op1.getValueType(), Op1, Amt, &Flags);
    Created.push_back(Op1.getNode());
    d = d.ashr(ShAmt);
  }

  // Calculate the multiplicative inverse, using Newton's method.
  APInt t, xn = d;
  while ((t = d*xn) != 1)
    xn *= APInt(d.getBitWidth(), 2) - t;

  SDValue Op2 = DAG.getConstant(xn, dl, Op1.getValueType());
  SDValue Mul = DAG.getNode(ISD::MUL, dl, Op1.getValueType(), Op1, Op2);
  Created.push_back(Mul.getNode());
  return Mul;
}

/// \brief Given an ISD::SDIV node expressing a divide by constant,
/// return a DAG expression to select that will generate the same value by
/// multiplying by a magic number.
/// Ref: "Hacker's Delight" or "The PowerPC Compiler Writer's Guide".
SDValue TargetLowering::BuildSDIV(SDNode *N, const APInt &Divisor,
                                  SelectionDAG &DAG, bool IsAfterLegalization,
                                  std::vector<SDNode *> *Created) const {
  assert(Created && "No vector to hold sdiv ops.");

  EVT VT = N->getValueType(0);
  SDLoc dl(N);

  // Check to see if we can do this.
  // FIXME: We should be more aggressive here.
  if (!isTypeLegal(VT))
    return SDValue();

  // If the sdiv has an 'exact' bit we can use a simpler lowering.
  if (cast<BinaryWithFlagsSDNode>(N)->Flags.hasExact())
    return BuildExactSDIV(*this, N->getOperand(0), Divisor, dl, DAG, *Created);

  APInt::ms magics = Divisor.magic();

  // Multiply the numerator (operand 0) by the magic value
  // FIXME: We should support doing a MUL in a wider type
  SDValue Q;
  if (IsAfterLegalization ? isOperationLegal(ISD::MULHS, VT) :
                            isOperationLegalOrCustom(ISD::MULHS, VT))
    Q = DAG.getNode(ISD::MULHS, dl, VT, N->getOperand(0),
                    DAG.getConstant(magics.m, dl, VT));
  else if (IsAfterLegalization ? isOperationLegal(ISD::SMUL_LOHI, VT) :
                                 isOperationLegalOrCustom(ISD::SMUL_LOHI, VT))
    Q = SDValue(DAG.getNode(ISD::SMUL_LOHI, dl, DAG.getVTList(VT, VT),
                              N->getOperand(0),
                              DAG.getConstant(magics.m, dl, VT)).getNode(), 1);
  else
    return SDValue();       // No mulhs or equvialent
  // If d > 0 and m < 0, add the numerator
  if (Divisor.isStrictlyPositive() && magics.m.isNegative()) {
    Q = DAG.getNode(ISD::ADD, dl, VT, Q, N->getOperand(0));
    Created->push_back(Q.getNode());
  }
  // If d < 0 and m > 0, subtract the numerator.
  if (Divisor.isNegative() && magics.m.isStrictlyPositive()) {
    Q = DAG.getNode(ISD::SUB, dl, VT, Q, N->getOperand(0));
    Created->push_back(Q.getNode());
  }
  auto &DL = DAG.getDataLayout();
  // Shift right algebraic if shift value is nonzero
  if (magics.s > 0) {
    Q = DAG.getNode(
        ISD::SRA, dl, VT, Q,
        DAG.getConstant(magics.s, dl, getShiftAmountTy(Q.getValueType(), DL)));
    Created->push_back(Q.getNode());
  }
  // Extract the sign bit and add it to the quotient
  SDValue T =
      DAG.getNode(ISD::SRL, dl, VT, Q,
                  DAG.getConstant(VT.getScalarSizeInBits() - 1, dl,
                                  getShiftAmountTy(Q.getValueType(), DL)));
  Created->push_back(T.getNode());
  return DAG.getNode(ISD::ADD, dl, VT, Q, T);
}

/// \brief Given an ISD::UDIV node expressing a divide by constant,
/// return a DAG expression to select that will generate the same value by
/// multiplying by a magic number.
/// Ref: "Hacker's Delight" or "The PowerPC Compiler Writer's Guide".
SDValue TargetLowering::BuildUDIV(SDNode *N, const APInt &Divisor,
                                  SelectionDAG &DAG, bool IsAfterLegalization,
                                  std::vector<SDNode *> *Created) const {
  assert(Created && "No vector to hold udiv ops.");

  EVT VT = N->getValueType(0);
  SDLoc dl(N);
  auto &DL = DAG.getDataLayout();

  // Check to see if we can do this.
  // FIXME: We should be more aggressive here.
  if (!isTypeLegal(VT))
    return SDValue();

  // FIXME: We should use a narrower constant when the upper
  // bits are known to be zero.
  APInt::mu magics = Divisor.magicu();

  SDValue Q = N->getOperand(0);

  // If the divisor is even, we can avoid using the expensive fixup by shifting
  // the divided value upfront.
  if (magics.a != 0 && !Divisor[0]) {
    unsigned Shift = Divisor.countTrailingZeros();
    Q = DAG.getNode(
        ISD::SRL, dl, VT, Q,
        DAG.getConstant(Shift, dl, getShiftAmountTy(Q.getValueType(), DL)));
    Created->push_back(Q.getNode());

    // Get magic number for the shifted divisor.
    magics = Divisor.lshr(Shift).magicu(Shift);
    assert(magics.a == 0 && "Should use cheap fixup now");
  }

  // Multiply the numerator (operand 0) by the magic value
  // FIXME: We should support doing a MUL in a wider type
  if (IsAfterLegalization ? isOperationLegal(ISD::MULHU, VT) :
                            isOperationLegalOrCustom(ISD::MULHU, VT))
    Q = DAG.getNode(ISD::MULHU, dl, VT, Q, DAG.getConstant(magics.m, dl, VT));
  else if (IsAfterLegalization ? isOperationLegal(ISD::UMUL_LOHI, VT) :
                                 isOperationLegalOrCustom(ISD::UMUL_LOHI, VT))
    Q = SDValue(DAG.getNode(ISD::UMUL_LOHI, dl, DAG.getVTList(VT, VT), Q,
                            DAG.getConstant(magics.m, dl, VT)).getNode(), 1);
  else
    return SDValue();       // No mulhu or equvialent

  Created->push_back(Q.getNode());

  if (magics.a == 0) {
    assert(magics.s < Divisor.getBitWidth() &&
           "We shouldn't generate an undefined shift!");
    return DAG.getNode(
        ISD::SRL, dl, VT, Q,
        DAG.getConstant(magics.s, dl, getShiftAmountTy(Q.getValueType(), DL)));
  } else {
    SDValue NPQ = DAG.getNode(ISD::SUB, dl, VT, N->getOperand(0), Q);
    Created->push_back(NPQ.getNode());
    NPQ = DAG.getNode(
        ISD::SRL, dl, VT, NPQ,
        DAG.getConstant(1, dl, getShiftAmountTy(NPQ.getValueType(), DL)));
    Created->push_back(NPQ.getNode());
    NPQ = DAG.getNode(ISD::ADD, dl, VT, NPQ, Q);
    Created->push_back(NPQ.getNode());
    return DAG.getNode(
        ISD::SRL, dl, VT, NPQ,
        DAG.getConstant(magics.s - 1, dl,
                        getShiftAmountTy(NPQ.getValueType(), DL)));
  }
}

bool TargetLowering::
verifyReturnAddressArgumentIsConstant(SDValue Op, SelectionDAG &DAG) const {
  if (!isa<ConstantSDNode>(Op.getOperand(0))) {
    DAG.getContext()->emitError("argument to '__builtin_return_address' must "
                                "be a constant integer");
    return true;
  }

  return false;
}

//===----------------------------------------------------------------------===//
// Legalization Utilities
//===----------------------------------------------------------------------===//

bool TargetLowering::expandMUL(SDNode *N, SDValue &Lo, SDValue &Hi, EVT HiLoVT,
                               SelectionDAG &DAG, SDValue LL, SDValue LH,
                               SDValue RL, SDValue RH) const {
  EVT VT = N->getValueType(0);
  SDLoc dl(N);

  bool HasMULHS = isOperationLegalOrCustom(ISD::MULHS, HiLoVT);
  bool HasMULHU = isOperationLegalOrCustom(ISD::MULHU, HiLoVT);
  bool HasSMUL_LOHI = isOperationLegalOrCustom(ISD::SMUL_LOHI, HiLoVT);
  bool HasUMUL_LOHI = isOperationLegalOrCustom(ISD::UMUL_LOHI, HiLoVT);
  if (HasMULHU || HasMULHS || HasUMUL_LOHI || HasSMUL_LOHI) {
    unsigned OuterBitSize = VT.getSizeInBits();
    unsigned InnerBitSize = HiLoVT.getSizeInBits();
    unsigned LHSSB = DAG.ComputeNumSignBits(N->getOperand(0));
    unsigned RHSSB = DAG.ComputeNumSignBits(N->getOperand(1));

    // LL, LH, RL, and RH must be either all NULL or all set to a value.
    assert((LL.getNode() && LH.getNode() && RL.getNode() && RH.getNode()) ||
           (!LL.getNode() && !LH.getNode() && !RL.getNode() && !RH.getNode()));

    if (!LL.getNode() && !RL.getNode() &&
        isOperationLegalOrCustom(ISD::TRUNCATE, HiLoVT)) {
      LL = DAG.getNode(ISD::TRUNCATE, dl, HiLoVT, N->getOperand(0));
      RL = DAG.getNode(ISD::TRUNCATE, dl, HiLoVT, N->getOperand(1));
    }

    if (!LL.getNode())
      return false;

    APInt HighMask = APInt::getHighBitsSet(OuterBitSize, InnerBitSize);
    if (DAG.MaskedValueIsZero(N->getOperand(0), HighMask) &&
        DAG.MaskedValueIsZero(N->getOperand(1), HighMask)) {
      // The inputs are both zero-extended.
      if (HasUMUL_LOHI) {
        // We can emit a umul_lohi.
        Lo = DAG.getNode(ISD::UMUL_LOHI, dl, DAG.getVTList(HiLoVT, HiLoVT), LL,
                         RL);
        Hi = SDValue(Lo.getNode(), 1);
        return true;
      }
      if (HasMULHU) {
        // We can emit a mulhu+mul.
        Lo = DAG.getNode(ISD::MUL, dl, HiLoVT, LL, RL);
        Hi = DAG.getNode(ISD::MULHU, dl, HiLoVT, LL, RL);
        return true;
      }
    }
    if (LHSSB > InnerBitSize && RHSSB > InnerBitSize) {
      // The input values are both sign-extended.
      if (HasSMUL_LOHI) {
        // We can emit a smul_lohi.
        Lo = DAG.getNode(ISD::SMUL_LOHI, dl, DAG.getVTList(HiLoVT, HiLoVT), LL,
                         RL);
        Hi = SDValue(Lo.getNode(), 1);
        return true;
      }
      if (HasMULHS) {
        // We can emit a mulhs+mul.
        Lo = DAG.getNode(ISD::MUL, dl, HiLoVT, LL, RL);
        Hi = DAG.getNode(ISD::MULHS, dl, HiLoVT, LL, RL);
        return true;
      }
    }

    if (!LH.getNode() && !RH.getNode() &&
        isOperationLegalOrCustom(ISD::SRL, VT) &&
        isOperationLegalOrCustom(ISD::TRUNCATE, HiLoVT)) {
      auto &DL = DAG.getDataLayout();
      unsigned ShiftAmt = VT.getSizeInBits() - HiLoVT.getSizeInBits();
      SDValue Shift = DAG.getConstant(ShiftAmt, dl, getShiftAmountTy(VT, DL));
      LH = DAG.getNode(ISD::SRL, dl, VT, N->getOperand(0), Shift);
      LH = DAG.getNode(ISD::TRUNCATE, dl, HiLoVT, LH);
      RH = DAG.getNode(ISD::SRL, dl, VT, N->getOperand(1), Shift);
      RH = DAG.getNode(ISD::TRUNCATE, dl, HiLoVT, RH);
    }

    if (!LH.getNode())
      return false;

    if (HasUMUL_LOHI) {
      // Lo,Hi = umul LHS, RHS.
      SDValue UMulLOHI = DAG.getNode(ISD::UMUL_LOHI, dl,
                                     DAG.getVTList(HiLoVT, HiLoVT), LL, RL);
      Lo = UMulLOHI;
      Hi = UMulLOHI.getValue(1);
      RH = DAG.getNode(ISD::MUL, dl, HiLoVT, LL, RH);
      LH = DAG.getNode(ISD::MUL, dl, HiLoVT, LH, RL);
      Hi = DAG.getNode(ISD::ADD, dl, HiLoVT, Hi, RH);
      Hi = DAG.getNode(ISD::ADD, dl, HiLoVT, Hi, LH);
      return true;
    }
    if (HasMULHU) {
      Lo = DAG.getNode(ISD::MUL, dl, HiLoVT, LL, RL);
      Hi = DAG.getNode(ISD::MULHU, dl, HiLoVT, LL, RL);
      RH = DAG.getNode(ISD::MUL, dl, HiLoVT, LL, RH);
      LH = DAG.getNode(ISD::MUL, dl, HiLoVT, LH, RL);
      Hi = DAG.getNode(ISD::ADD, dl, HiLoVT, Hi, RH);
      Hi = DAG.getNode(ISD::ADD, dl, HiLoVT, Hi, LH);
      return true;
    }
  }
  return false;
}

bool TargetLowering::expandFP_TO_SINT(SDNode *Node, SDValue &Result,
                               SelectionDAG &DAG) const {
  EVT VT = Node->getOperand(0).getValueType();
  EVT NVT = Node->getValueType(0);
  SDLoc dl(SDValue(Node, 0));

  // FIXME: Only f32 to i64 conversions are supported.
  if (VT != MVT::f32 || NVT != MVT::i64)
    return false;

  // Expand f32 -> i64 conversion
  // This algorithm comes from compiler-rt's implementation of fixsfdi:
  // https://github.com/llvm-mirror/compiler-rt/blob/master/lib/builtins/fixsfdi.c
  EVT IntVT = EVT::getIntegerVT(*DAG.getContext(),
                                VT.getSizeInBits());
  SDValue ExponentMask = DAG.getConstant(0x7F800000, dl, IntVT);
  SDValue ExponentLoBit = DAG.getConstant(23, dl, IntVT);
  SDValue Bias = DAG.getConstant(127, dl, IntVT);
  SDValue SignMask = DAG.getConstant(APInt::getSignBit(VT.getSizeInBits()), dl,
                                     IntVT);
  SDValue SignLowBit = DAG.getConstant(VT.getSizeInBits() - 1, dl, IntVT);
  SDValue MantissaMask = DAG.getConstant(0x007FFFFF, dl, IntVT);

  SDValue Bits = DAG.getNode(ISD::BITCAST, dl, IntVT, Node->getOperand(0));

  auto &DL = DAG.getDataLayout();
  SDValue ExponentBits = DAG.getNode(
      ISD::SRL, dl, IntVT, DAG.getNode(ISD::AND, dl, IntVT, Bits, ExponentMask),
      DAG.getZExtOrTrunc(ExponentLoBit, dl, getShiftAmountTy(IntVT, DL)));
  SDValue Exponent = DAG.getNode(ISD::SUB, dl, IntVT, ExponentBits, Bias);

  SDValue Sign = DAG.getNode(
      ISD::SRA, dl, IntVT, DAG.getNode(ISD::AND, dl, IntVT, Bits, SignMask),
      DAG.getZExtOrTrunc(SignLowBit, dl, getShiftAmountTy(IntVT, DL)));
  Sign = DAG.getSExtOrTrunc(Sign, dl, NVT);

  SDValue R = DAG.getNode(ISD::OR, dl, IntVT,
      DAG.getNode(ISD::AND, dl, IntVT, Bits, MantissaMask),
      DAG.getConstant(0x00800000, dl, IntVT));

  R = DAG.getZExtOrTrunc(R, dl, NVT);

  R = DAG.getSelectCC(
      dl, Exponent, ExponentLoBit,
      DAG.getNode(ISD::SHL, dl, NVT, R,
                  DAG.getZExtOrTrunc(
                      DAG.getNode(ISD::SUB, dl, IntVT, Exponent, ExponentLoBit),
                      dl, getShiftAmountTy(IntVT, DL))),
      DAG.getNode(ISD::SRL, dl, NVT, R,
                  DAG.getZExtOrTrunc(
                      DAG.getNode(ISD::SUB, dl, IntVT, ExponentLoBit, Exponent),
                      dl, getShiftAmountTy(IntVT, DL))),
      ISD::SETGT);

  SDValue Ret = DAG.getNode(ISD::SUB, dl, NVT,
      DAG.getNode(ISD::XOR, dl, NVT, R, Sign),
      Sign);

  Result = DAG.getSelectCC(dl, Exponent, DAG.getConstant(0, dl, IntVT),
      DAG.getConstant(0, dl, NVT), Ret, ISD::SETLT);
  return true;
}
