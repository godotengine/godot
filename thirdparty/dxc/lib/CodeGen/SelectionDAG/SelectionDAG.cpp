//===-- SelectionDAG.cpp - Implement the SelectionDAG data structures -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements the SelectionDAG class.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/SelectionDAG.h"
#include "SDNodeDbgValue.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetIntrinsicInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSelectionDAGInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include <algorithm>
#include <cmath>
#include <utility>

using namespace llvm;

/// makeVTList - Return an instance of the SDVTList struct initialized with the
/// specified members.
static SDVTList makeVTList(const EVT *VTs, unsigned NumVTs) {
  SDVTList Res = {VTs, NumVTs};
  return Res;
}

// Default null implementations of the callbacks.
void SelectionDAG::DAGUpdateListener::NodeDeleted(SDNode*, SDNode*) {}
void SelectionDAG::DAGUpdateListener::NodeUpdated(SDNode*) {}

//===----------------------------------------------------------------------===//
//                              ConstantFPSDNode Class
//===----------------------------------------------------------------------===//

/// isExactlyValue - We don't rely on operator== working on double values, as
/// it returns true for things that are clearly not equal, like -0.0 and 0.0.
/// As such, this method can be used to do an exact bit-for-bit comparison of
/// two floating point values.
bool ConstantFPSDNode::isExactlyValue(const APFloat& V) const {
  return getValueAPF().bitwiseIsEqual(V);
}

bool ConstantFPSDNode::isValueValidForType(EVT VT,
                                           const APFloat& Val) {
  assert(VT.isFloatingPoint() && "Can only convert between FP types");

  // convert modifies in place, so make a copy.
  APFloat Val2 = APFloat(Val);
  bool losesInfo;
  (void) Val2.convert(SelectionDAG::EVTToAPFloatSemantics(VT),
                      APFloat::rmNearestTiesToEven,
                      &losesInfo);
  return !losesInfo;
}

//===----------------------------------------------------------------------===//
//                              ISD Namespace
//===----------------------------------------------------------------------===//

/// isBuildVectorAllOnes - Return true if the specified node is a
/// BUILD_VECTOR where all of the elements are ~0 or undef.
bool ISD::isBuildVectorAllOnes(const SDNode *N) {
  // Look through a bit convert.
  while (N->getOpcode() == ISD::BITCAST)
    N = N->getOperand(0).getNode();

  if (N->getOpcode() != ISD::BUILD_VECTOR) return false;

  unsigned i = 0, e = N->getNumOperands();

  // Skip over all of the undef values.
  while (i != e && N->getOperand(i).getOpcode() == ISD::UNDEF)
    ++i;

  // Do not accept an all-undef vector.
  if (i == e) return false;

  // Do not accept build_vectors that aren't all constants or which have non-~0
  // elements. We have to be a bit careful here, as the type of the constant
  // may not be the same as the type of the vector elements due to type
  // legalization (the elements are promoted to a legal type for the target and
  // a vector of a type may be legal when the base element type is not).
  // We only want to check enough bits to cover the vector elements, because
  // we care if the resultant vector is all ones, not whether the individual
  // constants are.
  SDValue NotZero = N->getOperand(i);
  unsigned EltSize = N->getValueType(0).getVectorElementType().getSizeInBits();
  if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(NotZero)) {
    if (CN->getAPIntValue().countTrailingOnes() < EltSize)
      return false;
  } else if (ConstantFPSDNode *CFPN = dyn_cast<ConstantFPSDNode>(NotZero)) {
    if (CFPN->getValueAPF().bitcastToAPInt().countTrailingOnes() < EltSize)
      return false;
  } else
    return false;

  // Okay, we have at least one ~0 value, check to see if the rest match or are
  // undefs. Even with the above element type twiddling, this should be OK, as
  // the same type legalization should have applied to all the elements.
  for (++i; i != e; ++i)
    if (N->getOperand(i) != NotZero &&
        N->getOperand(i).getOpcode() != ISD::UNDEF)
      return false;
  return true;
}


/// isBuildVectorAllZeros - Return true if the specified node is a
/// BUILD_VECTOR where all of the elements are 0 or undef.
bool ISD::isBuildVectorAllZeros(const SDNode *N) {
  // Look through a bit convert.
  while (N->getOpcode() == ISD::BITCAST)
    N = N->getOperand(0).getNode();

  if (N->getOpcode() != ISD::BUILD_VECTOR) return false;

  bool IsAllUndef = true;
  for (const SDValue &Op : N->op_values()) {
    if (Op.getOpcode() == ISD::UNDEF)
      continue;
    IsAllUndef = false;
    // Do not accept build_vectors that aren't all constants or which have non-0
    // elements. We have to be a bit careful here, as the type of the constant
    // may not be the same as the type of the vector elements due to type
    // legalization (the elements are promoted to a legal type for the target
    // and a vector of a type may be legal when the base element type is not).
    // We only want to check enough bits to cover the vector elements, because
    // we care if the resultant vector is all zeros, not whether the individual
    // constants are.
    unsigned EltSize = N->getValueType(0).getVectorElementType().getSizeInBits();
    if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Op)) {
      if (CN->getAPIntValue().countTrailingZeros() < EltSize)
        return false;
    } else if (ConstantFPSDNode *CFPN = dyn_cast<ConstantFPSDNode>(Op)) {
      if (CFPN->getValueAPF().bitcastToAPInt().countTrailingZeros() < EltSize)
        return false;
    } else
      return false;
  }

  // Do not accept an all-undef vector.
  if (IsAllUndef)
    return false;
  return true;
}

/// \brief Return true if the specified node is a BUILD_VECTOR node of
/// all ConstantSDNode or undef.
bool ISD::isBuildVectorOfConstantSDNodes(const SDNode *N) {
  if (N->getOpcode() != ISD::BUILD_VECTOR)
    return false;

  for (const SDValue &Op : N->op_values()) {
    if (Op.getOpcode() == ISD::UNDEF)
      continue;
    if (!isa<ConstantSDNode>(Op))
      return false;
  }
  return true;
}

/// \brief Return true if the specified node is a BUILD_VECTOR node of
/// all ConstantFPSDNode or undef.
bool ISD::isBuildVectorOfConstantFPSDNodes(const SDNode *N) {
  if (N->getOpcode() != ISD::BUILD_VECTOR)
    return false;

  for (const SDValue &Op : N->op_values()) {
    if (Op.getOpcode() == ISD::UNDEF)
      continue;
    if (!isa<ConstantFPSDNode>(Op))
      return false;
  }
  return true;
}

/// isScalarToVector - Return true if the specified node is a
/// ISD::SCALAR_TO_VECTOR node or a BUILD_VECTOR node where only the low
/// element is not an undef.
bool ISD::isScalarToVector(const SDNode *N) {
  if (N->getOpcode() == ISD::SCALAR_TO_VECTOR)
    return true;

  if (N->getOpcode() != ISD::BUILD_VECTOR)
    return false;
  if (N->getOperand(0).getOpcode() == ISD::UNDEF)
    return false;
  unsigned NumElems = N->getNumOperands();
  if (NumElems == 1)
    return false;
  for (unsigned i = 1; i < NumElems; ++i) {
    SDValue V = N->getOperand(i);
    if (V.getOpcode() != ISD::UNDEF)
      return false;
  }
  return true;
}

/// allOperandsUndef - Return true if the node has at least one operand
/// and all operands of the specified node are ISD::UNDEF.
bool ISD::allOperandsUndef(const SDNode *N) {
  // Return false if the node has no operands.
  // This is "logically inconsistent" with the definition of "all" but
  // is probably the desired behavior.
  if (N->getNumOperands() == 0)
    return false;

  for (const SDValue &Op : N->op_values())
    if (Op.getOpcode() != ISD::UNDEF)
      return false;

  return true;
}

ISD::NodeType ISD::getExtForLoadExtType(bool IsFP, ISD::LoadExtType ExtType) {
  switch (ExtType) {
  case ISD::EXTLOAD:
    return IsFP ? ISD::FP_EXTEND : ISD::ANY_EXTEND;
  case ISD::SEXTLOAD:
    return ISD::SIGN_EXTEND;
  case ISD::ZEXTLOAD:
    return ISD::ZERO_EXTEND;
  default:
    break;
  }

  llvm_unreachable("Invalid LoadExtType");
}

/// getSetCCSwappedOperands - Return the operation corresponding to (Y op X)
/// when given the operation for (X op Y).
ISD::CondCode ISD::getSetCCSwappedOperands(ISD::CondCode Operation) {
  // To perform this operation, we just need to swap the L and G bits of the
  // operation.
  unsigned OldL = (Operation >> 2) & 1;
  unsigned OldG = (Operation >> 1) & 1;
  return ISD::CondCode((Operation & ~6) |  // Keep the N, U, E bits
                       (OldL << 1) |       // New G bit
                       (OldG << 2));       // New L bit.
}

/// getSetCCInverse - Return the operation corresponding to !(X op Y), where
/// 'op' is a valid SetCC operation.
ISD::CondCode ISD::getSetCCInverse(ISD::CondCode Op, bool isInteger) {
  unsigned Operation = Op;
  if (isInteger)
    Operation ^= 7;   // Flip L, G, E bits, but not U.
  else
    Operation ^= 15;  // Flip all of the condition bits.

  if (Operation > ISD::SETTRUE2)
    Operation &= ~8;  // Don't let N and U bits get set.

  return ISD::CondCode(Operation);
}


/// isSignedOp - For an integer comparison, return 1 if the comparison is a
/// signed operation and 2 if the result is an unsigned comparison.  Return zero
/// if the operation does not depend on the sign of the input (setne and seteq).
static int isSignedOp(ISD::CondCode Opcode) {
  switch (Opcode) {
  default: llvm_unreachable("Illegal integer setcc operation!");
  case ISD::SETEQ:
  case ISD::SETNE: return 0;
  case ISD::SETLT:
  case ISD::SETLE:
  case ISD::SETGT:
  case ISD::SETGE: return 1;
  case ISD::SETULT:
  case ISD::SETULE:
  case ISD::SETUGT:
  case ISD::SETUGE: return 2;
  }
}

/// getSetCCOrOperation - Return the result of a logical OR between different
/// comparisons of identical values: ((X op1 Y) | (X op2 Y)).  This function
/// returns SETCC_INVALID if it is not possible to represent the resultant
/// comparison.
ISD::CondCode ISD::getSetCCOrOperation(ISD::CondCode Op1, ISD::CondCode Op2,
                                       bool isInteger) {
  if (isInteger && (isSignedOp(Op1) | isSignedOp(Op2)) == 3)
    // Cannot fold a signed integer setcc with an unsigned integer setcc.
    return ISD::SETCC_INVALID;

  unsigned Op = Op1 | Op2;  // Combine all of the condition bits.

  // If the N and U bits get set then the resultant comparison DOES suddenly
  // care about orderedness, and is true when ordered.
  if (Op > ISD::SETTRUE2)
    Op &= ~16;     // Clear the U bit if the N bit is set.

  // Canonicalize illegal integer setcc's.
  if (isInteger && Op == ISD::SETUNE)  // e.g. SETUGT | SETULT
    Op = ISD::SETNE;

  return ISD::CondCode(Op);
}

/// getSetCCAndOperation - Return the result of a logical AND between different
/// comparisons of identical values: ((X op1 Y) & (X op2 Y)).  This
/// function returns zero if it is not possible to represent the resultant
/// comparison.
ISD::CondCode ISD::getSetCCAndOperation(ISD::CondCode Op1, ISD::CondCode Op2,
                                        bool isInteger) {
  if (isInteger && (isSignedOp(Op1) | isSignedOp(Op2)) == 3)
    // Cannot fold a signed setcc with an unsigned setcc.
    return ISD::SETCC_INVALID;

  // Combine all of the condition bits.
  ISD::CondCode Result = ISD::CondCode(Op1 & Op2);

  // Canonicalize illegal integer setcc's.
  if (isInteger) {
    switch (Result) {
    default: break;
    case ISD::SETUO : Result = ISD::SETFALSE; break;  // SETUGT & SETULT
    case ISD::SETOEQ:                                 // SETEQ  & SETU[LG]E
    case ISD::SETUEQ: Result = ISD::SETEQ   ; break;  // SETUGE & SETULE
    case ISD::SETOLT: Result = ISD::SETULT  ; break;  // SETULT & SETNE
    case ISD::SETOGT: Result = ISD::SETUGT  ; break;  // SETUGT & SETNE
    }
  }

  return Result;
}

//===----------------------------------------------------------------------===//
//                           SDNode Profile Support
//===----------------------------------------------------------------------===//

/// AddNodeIDOpcode - Add the node opcode to the NodeID data.
///
static void AddNodeIDOpcode(FoldingSetNodeID &ID, unsigned OpC)  {
  ID.AddInteger(OpC);
}

/// AddNodeIDValueTypes - Value type lists are intern'd so we can represent them
/// solely with their pointer.
static void AddNodeIDValueTypes(FoldingSetNodeID &ID, SDVTList VTList) {
  ID.AddPointer(VTList.VTs);
}

/// AddNodeIDOperands - Various routines for adding operands to the NodeID data.
///
static void AddNodeIDOperands(FoldingSetNodeID &ID,
                              ArrayRef<SDValue> Ops) {
  for (auto& Op : Ops) {
    ID.AddPointer(Op.getNode());
    ID.AddInteger(Op.getResNo());
  }
}

/// AddNodeIDOperands - Various routines for adding operands to the NodeID data.
///
static void AddNodeIDOperands(FoldingSetNodeID &ID,
                              ArrayRef<SDUse> Ops) {
  for (auto& Op : Ops) {
    ID.AddPointer(Op.getNode());
    ID.AddInteger(Op.getResNo());
  }
}
/// Add logical or fast math flag values to FoldingSetNodeID value.
static void AddNodeIDFlags(FoldingSetNodeID &ID, unsigned Opcode,
                           const SDNodeFlags *Flags) {
  if (!Flags || !isBinOpWithFlags(Opcode))
    return;

  unsigned RawFlags = Flags->getRawFlags();
  // If no flags are set, do not alter the ID. We must match the ID of nodes
  // that were created without explicitly specifying flags. This also saves time
  // and allows a gradual increase in API usage of the optional optimization
  // flags.
  if (RawFlags != 0)
    ID.AddInteger(RawFlags);
}

static void AddNodeIDFlags(FoldingSetNodeID &ID, const SDNode *N) {
  if (auto *Node = dyn_cast<BinaryWithFlagsSDNode>(N))
    AddNodeIDFlags(ID, Node->getOpcode(), &Node->Flags);
}

static void AddNodeIDNode(FoldingSetNodeID &ID, unsigned short OpC,
                          SDVTList VTList, ArrayRef<SDValue> OpList) {
  AddNodeIDOpcode(ID, OpC);
  AddNodeIDValueTypes(ID, VTList);
  AddNodeIDOperands(ID, OpList);
}

/// If this is an SDNode with special info, add this info to the NodeID data.
static void AddNodeIDCustom(FoldingSetNodeID &ID, const SDNode *N) {
  switch (N->getOpcode()) {
  case ISD::TargetExternalSymbol:
  case ISD::ExternalSymbol:
  case ISD::MCSymbol:
    llvm_unreachable("Should only be used on nodes with operands");
  default: break;  // Normal nodes don't need extra info.
  case ISD::TargetConstant:
  case ISD::Constant: {
    const ConstantSDNode *C = cast<ConstantSDNode>(N);
    ID.AddPointer(C->getConstantIntValue());
    ID.AddBoolean(C->isOpaque());
    break;
  }
  case ISD::TargetConstantFP:
  case ISD::ConstantFP: {
    ID.AddPointer(cast<ConstantFPSDNode>(N)->getConstantFPValue());
    break;
  }
  case ISD::TargetGlobalAddress:
  case ISD::GlobalAddress:
  case ISD::TargetGlobalTLSAddress:
  case ISD::GlobalTLSAddress: {
    const GlobalAddressSDNode *GA = cast<GlobalAddressSDNode>(N);
    ID.AddPointer(GA->getGlobal());
    ID.AddInteger(GA->getOffset());
    ID.AddInteger(GA->getTargetFlags());
    ID.AddInteger(GA->getAddressSpace());
    break;
  }
  case ISD::BasicBlock:
    ID.AddPointer(cast<BasicBlockSDNode>(N)->getBasicBlock());
    break;
  case ISD::Register:
    ID.AddInteger(cast<RegisterSDNode>(N)->getReg());
    break;
  case ISD::RegisterMask:
    ID.AddPointer(cast<RegisterMaskSDNode>(N)->getRegMask());
    break;
  case ISD::SRCVALUE:
    ID.AddPointer(cast<SrcValueSDNode>(N)->getValue());
    break;
  case ISD::FrameIndex:
  case ISD::TargetFrameIndex:
    ID.AddInteger(cast<FrameIndexSDNode>(N)->getIndex());
    break;
  case ISD::JumpTable:
  case ISD::TargetJumpTable:
    ID.AddInteger(cast<JumpTableSDNode>(N)->getIndex());
    ID.AddInteger(cast<JumpTableSDNode>(N)->getTargetFlags());
    break;
  case ISD::ConstantPool:
  case ISD::TargetConstantPool: {
    const ConstantPoolSDNode *CP = cast<ConstantPoolSDNode>(N);
    ID.AddInteger(CP->getAlignment());
    ID.AddInteger(CP->getOffset());
    if (CP->isMachineConstantPoolEntry())
      CP->getMachineCPVal()->addSelectionDAGCSEId(ID);
    else
      ID.AddPointer(CP->getConstVal());
    ID.AddInteger(CP->getTargetFlags());
    break;
  }
  case ISD::TargetIndex: {
    const TargetIndexSDNode *TI = cast<TargetIndexSDNode>(N);
    ID.AddInteger(TI->getIndex());
    ID.AddInteger(TI->getOffset());
    ID.AddInteger(TI->getTargetFlags());
    break;
  }
  case ISD::LOAD: {
    const LoadSDNode *LD = cast<LoadSDNode>(N);
    ID.AddInteger(LD->getMemoryVT().getRawBits());
    ID.AddInteger(LD->getRawSubclassData());
    ID.AddInteger(LD->getPointerInfo().getAddrSpace());
    break;
  }
  case ISD::STORE: {
    const StoreSDNode *ST = cast<StoreSDNode>(N);
    ID.AddInteger(ST->getMemoryVT().getRawBits());
    ID.AddInteger(ST->getRawSubclassData());
    ID.AddInteger(ST->getPointerInfo().getAddrSpace());
    break;
  }
  case ISD::ATOMIC_CMP_SWAP:
  case ISD::ATOMIC_CMP_SWAP_WITH_SUCCESS:
  case ISD::ATOMIC_SWAP:
  case ISD::ATOMIC_LOAD_ADD:
  case ISD::ATOMIC_LOAD_SUB:
  case ISD::ATOMIC_LOAD_AND:
  case ISD::ATOMIC_LOAD_OR:
  case ISD::ATOMIC_LOAD_XOR:
  case ISD::ATOMIC_LOAD_NAND:
  case ISD::ATOMIC_LOAD_MIN:
  case ISD::ATOMIC_LOAD_MAX:
  case ISD::ATOMIC_LOAD_UMIN:
  case ISD::ATOMIC_LOAD_UMAX:
  case ISD::ATOMIC_LOAD:
  case ISD::ATOMIC_STORE: {
    const AtomicSDNode *AT = cast<AtomicSDNode>(N);
    ID.AddInteger(AT->getMemoryVT().getRawBits());
    ID.AddInteger(AT->getRawSubclassData());
    ID.AddInteger(AT->getPointerInfo().getAddrSpace());
    break;
  }
  case ISD::PREFETCH: {
    const MemSDNode *PF = cast<MemSDNode>(N);
    ID.AddInteger(PF->getPointerInfo().getAddrSpace());
    break;
  }
  case ISD::VECTOR_SHUFFLE: {
    const ShuffleVectorSDNode *SVN = cast<ShuffleVectorSDNode>(N);
    for (unsigned i = 0, e = N->getValueType(0).getVectorNumElements();
         i != e; ++i)
      ID.AddInteger(SVN->getMaskElt(i));
    break;
  }
  case ISD::TargetBlockAddress:
  case ISD::BlockAddress: {
    const BlockAddressSDNode *BA = cast<BlockAddressSDNode>(N);
    ID.AddPointer(BA->getBlockAddress());
    ID.AddInteger(BA->getOffset());
    ID.AddInteger(BA->getTargetFlags());
    break;
  }
  } // end switch (N->getOpcode())

  AddNodeIDFlags(ID, N);

  // Target specific memory nodes could also have address spaces to check.
  if (N->isTargetMemoryOpcode())
    ID.AddInteger(cast<MemSDNode>(N)->getPointerInfo().getAddrSpace());
}

/// AddNodeIDNode - Generic routine for adding a nodes info to the NodeID
/// data.
static void AddNodeIDNode(FoldingSetNodeID &ID, const SDNode *N) {
  AddNodeIDOpcode(ID, N->getOpcode());
  // Add the return value info.
  AddNodeIDValueTypes(ID, N->getVTList());
  // Add the operand info.
  AddNodeIDOperands(ID, N->ops());

  // Handle SDNode leafs with special info.
  AddNodeIDCustom(ID, N);
}

/// encodeMemSDNodeFlags - Generic routine for computing a value for use in
/// the CSE map that carries volatility, temporalness, indexing mode, and
/// extension/truncation information.
///
static inline unsigned
encodeMemSDNodeFlags(int ConvType, ISD::MemIndexedMode AM, bool isVolatile,
                     bool isNonTemporal, bool isInvariant) {
  assert((ConvType & 3) == ConvType &&
         "ConvType may not require more than 2 bits!");
  assert((AM & 7) == AM &&
         "AM may not require more than 3 bits!");
  return ConvType |
         (AM << 2) |
         (isVolatile << 5) |
         (isNonTemporal << 6) |
         (isInvariant << 7);
}

//===----------------------------------------------------------------------===//
//                              SelectionDAG Class
//===----------------------------------------------------------------------===//

/// doNotCSE - Return true if CSE should not be performed for this node.
static bool doNotCSE(SDNode *N) {
  if (N->getValueType(0) == MVT::Glue)
    return true; // Never CSE anything that produces a flag.

  switch (N->getOpcode()) {
  default: break;
  case ISD::HANDLENODE:
  case ISD::EH_LABEL:
    return true;   // Never CSE these nodes.
  }

  // Check that remaining values produced are not flags.
  for (unsigned i = 1, e = N->getNumValues(); i != e; ++i)
    if (N->getValueType(i) == MVT::Glue)
      return true; // Never CSE anything that produces a flag.

  return false;
}

/// RemoveDeadNodes - This method deletes all unreachable nodes in the
/// SelectionDAG.
void SelectionDAG::RemoveDeadNodes() {
  // Create a dummy node (which is not added to allnodes), that adds a reference
  // to the root node, preventing it from being deleted.
  HandleSDNode Dummy(getRoot());

  SmallVector<SDNode*, 128> DeadNodes;

  // Add all obviously-dead nodes to the DeadNodes worklist.
  for (allnodes_iterator I = allnodes_begin(), E = allnodes_end(); I != E; ++I)
    if (I->use_empty())
      DeadNodes.push_back(I);

  RemoveDeadNodes(DeadNodes);

  // If the root changed (e.g. it was a dead load, update the root).
  setRoot(Dummy.getValue());
}

/// RemoveDeadNodes - This method deletes the unreachable nodes in the
/// given list, and any nodes that become unreachable as a result.
void SelectionDAG::RemoveDeadNodes(SmallVectorImpl<SDNode *> &DeadNodes) {

  // Process the worklist, deleting the nodes and adding their uses to the
  // worklist.
  while (!DeadNodes.empty()) {
    SDNode *N = DeadNodes.pop_back_val();

    for (DAGUpdateListener *DUL = UpdateListeners; DUL; DUL = DUL->Next)
      DUL->NodeDeleted(N, nullptr);

    // Take the node out of the appropriate CSE map.
    RemoveNodeFromCSEMaps(N);

    // Next, brutally remove the operand list.  This is safe to do, as there are
    // no cycles in the graph.
    for (SDNode::op_iterator I = N->op_begin(), E = N->op_end(); I != E; ) {
      SDUse &Use = *I++;
      SDNode *Operand = Use.getNode();
      Use.set(SDValue());

      // Now that we removed this operand, see if there are no uses of it left.
      if (Operand->use_empty())
        DeadNodes.push_back(Operand);
    }

    DeallocateNode(N);
  }
}

void SelectionDAG::RemoveDeadNode(SDNode *N){
  SmallVector<SDNode*, 16> DeadNodes(1, N);

  // Create a dummy node that adds a reference to the root node, preventing
  // it from being deleted.  (This matters if the root is an operand of the
  // dead node.)
  HandleSDNode Dummy(getRoot());

  RemoveDeadNodes(DeadNodes);
}

void SelectionDAG::DeleteNode(SDNode *N) {
  // First take this out of the appropriate CSE map.
  RemoveNodeFromCSEMaps(N);

  // Finally, remove uses due to operands of this node, remove from the
  // AllNodes list, and delete the node.
  DeleteNodeNotInCSEMaps(N);
}

void SelectionDAG::DeleteNodeNotInCSEMaps(SDNode *N) {
  assert(N != AllNodes.begin() && "Cannot delete the entry node!");
  assert(N->use_empty() && "Cannot delete a node that is not dead!");

  // Drop all of the operands and decrement used node's use counts.
  N->DropOperands();

  DeallocateNode(N);
}

void SDDbgInfo::erase(const SDNode *Node) {
  DbgValMapType::iterator I = DbgValMap.find(Node);
  if (I == DbgValMap.end())
    return;
  for (auto &Val: I->second)
    Val->setIsInvalidated();
  DbgValMap.erase(I);
}

void SelectionDAG::DeallocateNode(SDNode *N) {
  if (N->OperandsNeedDelete)
    delete[] N->OperandList;

  // Set the opcode to DELETED_NODE to help catch bugs when node
  // memory is reallocated.
  N->NodeType = ISD::DELETED_NODE;

  NodeAllocator.Deallocate(AllNodes.remove(N));

  // If any of the SDDbgValue nodes refer to this SDNode, invalidate
  // them and forget about that node.
  DbgInfo->erase(N);
}

#ifndef NDEBUG
/// VerifySDNode - Sanity check the given SDNode.  Aborts if it is invalid.
static void VerifySDNode(SDNode *N) {
  switch (N->getOpcode()) {
  default:
    break;
  case ISD::BUILD_PAIR: {
    EVT VT = N->getValueType(0);
    assert(N->getNumValues() == 1 && "Too many results!");
    assert(!VT.isVector() && (VT.isInteger() || VT.isFloatingPoint()) &&
           "Wrong return type!");
    assert(N->getNumOperands() == 2 && "Wrong number of operands!");
    assert(N->getOperand(0).getValueType() == N->getOperand(1).getValueType() &&
           "Mismatched operand types!");
    assert(N->getOperand(0).getValueType().isInteger() == VT.isInteger() &&
           "Wrong operand type!");
    assert(VT.getSizeInBits() == 2 * N->getOperand(0).getValueSizeInBits() &&
           "Wrong return type size");
    break;
  }
  case ISD::BUILD_VECTOR: {
    assert(N->getNumValues() == 1 && "Too many results!");
    assert(N->getValueType(0).isVector() && "Wrong return type!");
    assert(N->getNumOperands() == N->getValueType(0).getVectorNumElements() &&
           "Wrong number of operands!");
    EVT EltVT = N->getValueType(0).getVectorElementType();
    for (SDNode::op_iterator I = N->op_begin(), E = N->op_end(); I != E; ++I) {
      assert((I->getValueType() == EltVT ||
             (EltVT.isInteger() && I->getValueType().isInteger() &&
              EltVT.bitsLE(I->getValueType()))) &&
            "Wrong operand type!");
      assert(I->getValueType() == N->getOperand(0).getValueType() &&
             "Operands must all have the same type");
    }
    break;
  }
  }
}
#endif // NDEBUG

/// \brief Insert a newly allocated node into the DAG.
///
/// Handles insertion into the all nodes list and CSE map, as well as
/// verification and other common operations when a new node is allocated.
void SelectionDAG::InsertNode(SDNode *N) {
  AllNodes.push_back(N);
#ifndef NDEBUG
  VerifySDNode(N);
#endif
}

/// RemoveNodeFromCSEMaps - Take the specified node out of the CSE map that
/// correspond to it.  This is useful when we're about to delete or repurpose
/// the node.  We don't want future request for structurally identical nodes
/// to return N anymore.
bool SelectionDAG::RemoveNodeFromCSEMaps(SDNode *N) {
  bool Erased = false;
  switch (N->getOpcode()) {
  case ISD::HANDLENODE: return false;  // noop.
  case ISD::CONDCODE:
    assert(CondCodeNodes[cast<CondCodeSDNode>(N)->get()] &&
           "Cond code doesn't exist!");
    Erased = CondCodeNodes[cast<CondCodeSDNode>(N)->get()] != nullptr;
    CondCodeNodes[cast<CondCodeSDNode>(N)->get()] = nullptr;
    break;
  case ISD::ExternalSymbol:
    Erased = ExternalSymbols.erase(cast<ExternalSymbolSDNode>(N)->getSymbol());
    break;
  case ISD::TargetExternalSymbol: {
    ExternalSymbolSDNode *ESN = cast<ExternalSymbolSDNode>(N);
    Erased = TargetExternalSymbols.erase(
               std::pair<std::string,unsigned char>(ESN->getSymbol(),
                                                    ESN->getTargetFlags()));
    break;
  }
  case ISD::MCSymbol: {
    auto *MCSN = cast<MCSymbolSDNode>(N);
    Erased = MCSymbols.erase(MCSN->getMCSymbol());
    break;
  }
  case ISD::VALUETYPE: {
    EVT VT = cast<VTSDNode>(N)->getVT();
    if (VT.isExtended()) {
      Erased = ExtendedValueTypeNodes.erase(VT);
    } else {
      Erased = ValueTypeNodes[VT.getSimpleVT().SimpleTy] != nullptr;
      ValueTypeNodes[VT.getSimpleVT().SimpleTy] = nullptr;
    }
    break;
  }
  default:
    // Remove it from the CSE Map.
    assert(N->getOpcode() != ISD::DELETED_NODE && "DELETED_NODE in CSEMap!");
    assert(N->getOpcode() != ISD::EntryToken && "EntryToken in CSEMap!");
    Erased = CSEMap.RemoveNode(N);
    break;
  }
#ifndef NDEBUG
  // Verify that the node was actually in one of the CSE maps, unless it has a
  // flag result (which cannot be CSE'd) or is one of the special cases that are
  // not subject to CSE.
  if (!Erased && N->getValueType(N->getNumValues()-1) != MVT::Glue &&
      !N->isMachineOpcode() && !doNotCSE(N)) {
    N->dump(this);
    dbgs() << "\n";
    llvm_unreachable("Node is not in map!");
  }
#endif
  return Erased;
}

/// AddModifiedNodeToCSEMaps - The specified node has been removed from the CSE
/// maps and modified in place. Add it back to the CSE maps, unless an identical
/// node already exists, in which case transfer all its users to the existing
/// node. This transfer can potentially trigger recursive merging.
///
void
SelectionDAG::AddModifiedNodeToCSEMaps(SDNode *N) {
  // For node types that aren't CSE'd, just act as if no identical node
  // already exists.
  if (!doNotCSE(N)) {
    SDNode *Existing = CSEMap.GetOrInsertNode(N);
    if (Existing != N) {
      // If there was already an existing matching node, use ReplaceAllUsesWith
      // to replace the dead one with the existing one.  This can cause
      // recursive merging of other unrelated nodes down the line.
      ReplaceAllUsesWith(N, Existing);

      // N is now dead. Inform the listeners and delete it.
      for (DAGUpdateListener *DUL = UpdateListeners; DUL; DUL = DUL->Next)
        DUL->NodeDeleted(N, Existing);
      DeleteNodeNotInCSEMaps(N);
      return;
    }
  }

  // If the node doesn't already exist, we updated it.  Inform listeners.
  for (DAGUpdateListener *DUL = UpdateListeners; DUL; DUL = DUL->Next)
    DUL->NodeUpdated(N);
}

/// FindModifiedNodeSlot - Find a slot for the specified node if its operands
/// were replaced with those specified.  If this node is never memoized,
/// return null, otherwise return a pointer to the slot it would take.  If a
/// node already exists with these operands, the slot will be non-null.
SDNode *SelectionDAG::FindModifiedNodeSlot(SDNode *N, SDValue Op,
                                           void *&InsertPos) {
  if (doNotCSE(N))
    return nullptr;

  SDValue Ops[] = { Op };
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, N->getOpcode(), N->getVTList(), Ops);
  AddNodeIDCustom(ID, N);
  SDNode *Node = FindNodeOrInsertPos(ID, N->getDebugLoc(), InsertPos);
  return Node;
}

/// FindModifiedNodeSlot - Find a slot for the specified node if its operands
/// were replaced with those specified.  If this node is never memoized,
/// return null, otherwise return a pointer to the slot it would take.  If a
/// node already exists with these operands, the slot will be non-null.
SDNode *SelectionDAG::FindModifiedNodeSlot(SDNode *N,
                                           SDValue Op1, SDValue Op2,
                                           void *&InsertPos) {
  if (doNotCSE(N))
    return nullptr;

  SDValue Ops[] = { Op1, Op2 };
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, N->getOpcode(), N->getVTList(), Ops);
  AddNodeIDCustom(ID, N);
  SDNode *Node = FindNodeOrInsertPos(ID, N->getDebugLoc(), InsertPos);
  return Node;
}


/// FindModifiedNodeSlot - Find a slot for the specified node if its operands
/// were replaced with those specified.  If this node is never memoized,
/// return null, otherwise return a pointer to the slot it would take.  If a
/// node already exists with these operands, the slot will be non-null.
SDNode *SelectionDAG::FindModifiedNodeSlot(SDNode *N, ArrayRef<SDValue> Ops,
                                           void *&InsertPos) {
  if (doNotCSE(N))
    return nullptr;

  FoldingSetNodeID ID;
  AddNodeIDNode(ID, N->getOpcode(), N->getVTList(), Ops);
  AddNodeIDCustom(ID, N);
  SDNode *Node = FindNodeOrInsertPos(ID, N->getDebugLoc(), InsertPos);
  return Node;
}

/// getEVTAlignment - Compute the default alignment value for the
/// given type.
///
unsigned SelectionDAG::getEVTAlignment(EVT VT) const {
  Type *Ty = VT == MVT::iPTR ?
                   PointerType::get(Type::getInt8Ty(*getContext()), 0) :
                   VT.getTypeForEVT(*getContext());

  return getDataLayout().getABITypeAlignment(Ty);
}

// EntryNode could meaningfully have debug info if we can find it...
SelectionDAG::SelectionDAG(const TargetMachine &tm, CodeGenOpt::Level OL)
    : TM(tm), TSI(nullptr), TLI(nullptr), OptLevel(OL),
      EntryNode(ISD::EntryToken, 0, DebugLoc(), getVTList(MVT::Other)),
      Root(getEntryNode()), NewNodesMustHaveLegalTypes(false),
      UpdateListeners(nullptr) {
  AllNodes.push_back(&EntryNode);
  DbgInfo = new SDDbgInfo();
}

void SelectionDAG::init(MachineFunction &mf) {
  MF = &mf;
  TLI = getSubtarget().getTargetLowering();
  TSI = getSubtarget().getSelectionDAGInfo();
  Context = &mf.getFunction()->getContext();
}

SelectionDAG::~SelectionDAG() {
  assert(!UpdateListeners && "Dangling registered DAGUpdateListeners");
  allnodes_clear();
  delete DbgInfo;
}

void SelectionDAG::allnodes_clear() {
  assert(&*AllNodes.begin() == &EntryNode);
  AllNodes.remove(AllNodes.begin());
  while (!AllNodes.empty())
    DeallocateNode(AllNodes.begin());
}

BinarySDNode *SelectionDAG::GetBinarySDNode(unsigned Opcode, SDLoc DL,
                                            SDVTList VTs, SDValue N1,
                                            SDValue N2,
                                            const SDNodeFlags *Flags) {
  if (isBinOpWithFlags(Opcode)) {
    // If no flags were passed in, use a default flags object.
    SDNodeFlags F;
    if (Flags == nullptr)
      Flags = &F;

    BinaryWithFlagsSDNode *FN = new (NodeAllocator) BinaryWithFlagsSDNode(
        Opcode, DL.getIROrder(), DL.getDebugLoc(), VTs, N1, N2, *Flags);

    return FN;
  }

  BinarySDNode *N = new (NodeAllocator)
      BinarySDNode(Opcode, DL.getIROrder(), DL.getDebugLoc(), VTs, N1, N2);
  return N;
}

SDNode *SelectionDAG::FindNodeOrInsertPos(const FoldingSetNodeID &ID,
                                          void *&InsertPos) {
  SDNode *N = CSEMap.FindNodeOrInsertPos(ID, InsertPos);
  if (N) {
    switch (N->getOpcode()) {
    default: break;
    case ISD::Constant:
    case ISD::ConstantFP:
      llvm_unreachable("Querying for Constant and ConstantFP nodes requires "
                       "debug location.  Use another overload.");
    }
  }
  return N;
}

SDNode *SelectionDAG::FindNodeOrInsertPos(const FoldingSetNodeID &ID,
                                          DebugLoc DL, void *&InsertPos) {
  SDNode *N = CSEMap.FindNodeOrInsertPos(ID, InsertPos);
  if (N) {
    switch (N->getOpcode()) {
    default: break; // Process only regular (non-target) constant nodes.
    case ISD::Constant:
    case ISD::ConstantFP:
      // Erase debug location from the node if the node is used at several
      // different places to do not propagate one location to all uses as it
      // leads to incorrect debug info.
      if (N->getDebugLoc() != DL)
        N->setDebugLoc(DebugLoc());
      break;
    }
  }
  return N;
}

void SelectionDAG::clear() {
  allnodes_clear();
  OperandAllocator.Reset();
  CSEMap.clear();

  ExtendedValueTypeNodes.clear();
  ExternalSymbols.clear();
  TargetExternalSymbols.clear();
  MCSymbols.clear();
  std::fill(CondCodeNodes.begin(), CondCodeNodes.end(),
            static_cast<CondCodeSDNode*>(nullptr));
  std::fill(ValueTypeNodes.begin(), ValueTypeNodes.end(),
            static_cast<SDNode*>(nullptr));

  EntryNode.UseList = nullptr;
  AllNodes.push_back(&EntryNode);
  Root = getEntryNode();
  DbgInfo->clear();
}

SDValue SelectionDAG::getAnyExtOrTrunc(SDValue Op, SDLoc DL, EVT VT) {
  return VT.bitsGT(Op.getValueType()) ?
    getNode(ISD::ANY_EXTEND, DL, VT, Op) :
    getNode(ISD::TRUNCATE, DL, VT, Op);
}

SDValue SelectionDAG::getSExtOrTrunc(SDValue Op, SDLoc DL, EVT VT) {
  return VT.bitsGT(Op.getValueType()) ?
    getNode(ISD::SIGN_EXTEND, DL, VT, Op) :
    getNode(ISD::TRUNCATE, DL, VT, Op);
}

SDValue SelectionDAG::getZExtOrTrunc(SDValue Op, SDLoc DL, EVT VT) {
  return VT.bitsGT(Op.getValueType()) ?
    getNode(ISD::ZERO_EXTEND, DL, VT, Op) :
    getNode(ISD::TRUNCATE, DL, VT, Op);
}

SDValue SelectionDAG::getBoolExtOrTrunc(SDValue Op, SDLoc SL, EVT VT,
                                        EVT OpVT) {
  if (VT.bitsLE(Op.getValueType()))
    return getNode(ISD::TRUNCATE, SL, VT, Op);

  TargetLowering::BooleanContent BType = TLI->getBooleanContents(OpVT);
  return getNode(TLI->getExtendForContent(BType), SL, VT, Op);
}

SDValue SelectionDAG::getZeroExtendInReg(SDValue Op, SDLoc DL, EVT VT) {
  assert(!VT.isVector() &&
         "getZeroExtendInReg should use the vector element type instead of "
         "the vector type!");
  if (Op.getValueType() == VT) return Op;
  unsigned BitWidth = Op.getValueType().getScalarType().getSizeInBits();
  APInt Imm = APInt::getLowBitsSet(BitWidth,
                                   VT.getSizeInBits());
  return getNode(ISD::AND, DL, Op.getValueType(), Op,
                 getConstant(Imm, DL, Op.getValueType()));
}

SDValue SelectionDAG::getAnyExtendVectorInReg(SDValue Op, SDLoc DL, EVT VT) {
  assert(VT.isVector() && "This DAG node is restricted to vector types.");
  assert(VT.getSizeInBits() == Op.getValueType().getSizeInBits() &&
         "The sizes of the input and result must match in order to perform the "
         "extend in-register.");
  assert(VT.getVectorNumElements() < Op.getValueType().getVectorNumElements() &&
         "The destination vector type must have fewer lanes than the input.");
  return getNode(ISD::ANY_EXTEND_VECTOR_INREG, DL, VT, Op);
}

SDValue SelectionDAG::getSignExtendVectorInReg(SDValue Op, SDLoc DL, EVT VT) {
  assert(VT.isVector() && "This DAG node is restricted to vector types.");
  assert(VT.getSizeInBits() == Op.getValueType().getSizeInBits() &&
         "The sizes of the input and result must match in order to perform the "
         "extend in-register.");
  assert(VT.getVectorNumElements() < Op.getValueType().getVectorNumElements() &&
         "The destination vector type must have fewer lanes than the input.");
  return getNode(ISD::SIGN_EXTEND_VECTOR_INREG, DL, VT, Op);
}

SDValue SelectionDAG::getZeroExtendVectorInReg(SDValue Op, SDLoc DL, EVT VT) {
  assert(VT.isVector() && "This DAG node is restricted to vector types.");
  assert(VT.getSizeInBits() == Op.getValueType().getSizeInBits() &&
         "The sizes of the input and result must match in order to perform the "
         "extend in-register.");
  assert(VT.getVectorNumElements() < Op.getValueType().getVectorNumElements() &&
         "The destination vector type must have fewer lanes than the input.");
  return getNode(ISD::ZERO_EXTEND_VECTOR_INREG, DL, VT, Op);
}

/// getNOT - Create a bitwise NOT operation as (XOR Val, -1).
///
SDValue SelectionDAG::getNOT(SDLoc DL, SDValue Val, EVT VT) {
  EVT EltVT = VT.getScalarType();
  SDValue NegOne =
    getConstant(APInt::getAllOnesValue(EltVT.getSizeInBits()), DL, VT);
  return getNode(ISD::XOR, DL, VT, Val, NegOne);
}

SDValue SelectionDAG::getLogicalNOT(SDLoc DL, SDValue Val, EVT VT) {
  EVT EltVT = VT.getScalarType();
  SDValue TrueValue;
  switch (TLI->getBooleanContents(VT)) {
    case TargetLowering::ZeroOrOneBooleanContent:
    case TargetLowering::UndefinedBooleanContent:
      TrueValue = getConstant(1, DL, VT);
      break;
    case TargetLowering::ZeroOrNegativeOneBooleanContent:
      TrueValue = getConstant(APInt::getAllOnesValue(EltVT.getSizeInBits()), DL,
                              VT);
      break;
  }
  return getNode(ISD::XOR, DL, VT, Val, TrueValue);
}

SDValue SelectionDAG::getConstant(uint64_t Val, SDLoc DL, EVT VT, bool isT,
                                  bool isO) {
  EVT EltVT = VT.getScalarType();
  assert((EltVT.getSizeInBits() >= 64 ||
         (uint64_t)((int64_t)Val >> EltVT.getSizeInBits()) + 1 < 2) &&
         "getConstant with a uint64_t value that doesn't fit in the type!");
  return getConstant(APInt(EltVT.getSizeInBits(), Val), DL, VT, isT, isO);
}

SDValue SelectionDAG::getConstant(const APInt &Val, SDLoc DL, EVT VT, bool isT,
                                  bool isO)
{
  return getConstant(*ConstantInt::get(*Context, Val), DL, VT, isT, isO);
}

SDValue SelectionDAG::getConstant(const ConstantInt &Val, SDLoc DL, EVT VT,
                                  bool isT, bool isO) {
  assert(VT.isInteger() && "Cannot create FP integer constant!");

  EVT EltVT = VT.getScalarType();
  const ConstantInt *Elt = &Val;

  // In some cases the vector type is legal but the element type is illegal and
  // needs to be promoted, for example v8i8 on ARM.  In this case, promote the
  // inserted value (the type does not need to match the vector element type).
  // Any extra bits introduced will be truncated away.
  if (VT.isVector() && TLI->getTypeAction(*getContext(), EltVT) ==
      TargetLowering::TypePromoteInteger) {
   EltVT = TLI->getTypeToTransformTo(*getContext(), EltVT);
   APInt NewVal = Elt->getValue().zext(EltVT.getSizeInBits());
   Elt = ConstantInt::get(*getContext(), NewVal);
  }
  // In other cases the element type is illegal and needs to be expanded, for
  // example v2i64 on MIPS32. In this case, find the nearest legal type, split
  // the value into n parts and use a vector type with n-times the elements.
  // Then bitcast to the type requested.
  // Legalizing constants too early makes the DAGCombiner's job harder so we
  // only legalize if the DAG tells us we must produce legal types.
  else if (NewNodesMustHaveLegalTypes && VT.isVector() &&
           TLI->getTypeAction(*getContext(), EltVT) ==
           TargetLowering::TypeExpandInteger) {
    APInt NewVal = Elt->getValue();
    EVT ViaEltVT = TLI->getTypeToTransformTo(*getContext(), EltVT);
    unsigned ViaEltSizeInBits = ViaEltVT.getSizeInBits();
    unsigned ViaVecNumElts = VT.getSizeInBits() / ViaEltSizeInBits;
    EVT ViaVecVT = EVT::getVectorVT(*getContext(), ViaEltVT, ViaVecNumElts);

    // Check the temporary vector is the correct size. If this fails then
    // getTypeToTransformTo() probably returned a type whose size (in bits)
    // isn't a power-of-2 factor of the requested type size.
    assert(ViaVecVT.getSizeInBits() == VT.getSizeInBits());

    SmallVector<SDValue, 2> EltParts;
    for (unsigned i = 0; i < ViaVecNumElts / VT.getVectorNumElements(); ++i) {
      EltParts.push_back(getConstant(NewVal.lshr(i * ViaEltSizeInBits)
                                           .trunc(ViaEltSizeInBits), DL,
                                     ViaEltVT, isT, isO));
    }

    // EltParts is currently in little endian order. If we actually want
    // big-endian order then reverse it now.
    if (getDataLayout().isBigEndian())
      std::reverse(EltParts.begin(), EltParts.end());

    // The elements must be reversed when the element order is different
    // to the endianness of the elements (because the BITCAST is itself a
    // vector shuffle in this situation). However, we do not need any code to
    // perform this reversal because getConstant() is producing a vector
    // splat.
    // This situation occurs in MIPS MSA.

    SmallVector<SDValue, 8> Ops;
    for (unsigned i = 0; i < VT.getVectorNumElements(); ++i)
      Ops.insert(Ops.end(), EltParts.begin(), EltParts.end());

    SDValue Result = getNode(ISD::BITCAST, SDLoc(), VT,
                             getNode(ISD::BUILD_VECTOR, SDLoc(), ViaVecVT,
                                     Ops));
    return Result;
  }

  assert(Elt->getBitWidth() == EltVT.getSizeInBits() &&
         "APInt size does not match type size!");
  unsigned Opc = isT ? ISD::TargetConstant : ISD::Constant;
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, Opc, getVTList(EltVT), None);
  ID.AddPointer(Elt);
  ID.AddBoolean(isO);
  void *IP = nullptr;
  SDNode *N = nullptr;
  if ((N = FindNodeOrInsertPos(ID, DL.getDebugLoc(), IP)))
    if (!VT.isVector())
      return SDValue(N, 0);

  if (!N) {
    N = new (NodeAllocator) ConstantSDNode(isT, isO, Elt, DL.getDebugLoc(),
                                           EltVT);
    CSEMap.InsertNode(N, IP);
    InsertNode(N);
  }

  SDValue Result(N, 0);
  if (VT.isVector()) {
    SmallVector<SDValue, 8> Ops;
    Ops.assign(VT.getVectorNumElements(), Result);
    Result = getNode(ISD::BUILD_VECTOR, SDLoc(), VT, Ops);
  }
  return Result;
}

SDValue SelectionDAG::getIntPtrConstant(uint64_t Val, SDLoc DL, bool isTarget) {
  return getConstant(Val, DL, TLI->getPointerTy(getDataLayout()), isTarget);
}

SDValue SelectionDAG::getConstantFP(const APFloat& V, SDLoc DL, EVT VT,
                                    bool isTarget) {
  return getConstantFP(*ConstantFP::get(*getContext(), V), DL, VT, isTarget);
}

SDValue SelectionDAG::getConstantFP(const ConstantFP& V, SDLoc DL, EVT VT,
                                    bool isTarget){
  assert(VT.isFloatingPoint() && "Cannot create integer FP constant!");

  EVT EltVT = VT.getScalarType();

  // Do the map lookup using the actual bit pattern for the floating point
  // value, so that we don't have problems with 0.0 comparing equal to -0.0, and
  // we don't have issues with SNANs.
  unsigned Opc = isTarget ? ISD::TargetConstantFP : ISD::ConstantFP;
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, Opc, getVTList(EltVT), None);
  ID.AddPointer(&V);
  void *IP = nullptr;
  SDNode *N = nullptr;
  if ((N = FindNodeOrInsertPos(ID, DL.getDebugLoc(), IP)))
    if (!VT.isVector())
      return SDValue(N, 0);

  if (!N) {
    N = new (NodeAllocator) ConstantFPSDNode(isTarget, &V, DL.getDebugLoc(),
                                             EltVT);
    CSEMap.InsertNode(N, IP);
    InsertNode(N);
  }

  SDValue Result(N, 0);
  if (VT.isVector()) {
    SmallVector<SDValue, 8> Ops;
    Ops.assign(VT.getVectorNumElements(), Result);
    Result = getNode(ISD::BUILD_VECTOR, SDLoc(), VT, Ops);
  }
  return Result;
}

SDValue SelectionDAG::getConstantFP(double Val, SDLoc DL, EVT VT,
                                    bool isTarget) {
  EVT EltVT = VT.getScalarType();
  if (EltVT==MVT::f32)
    return getConstantFP(APFloat((float)Val), DL, VT, isTarget);
  else if (EltVT==MVT::f64)
    return getConstantFP(APFloat(Val), DL, VT, isTarget);
  else if (EltVT==MVT::f80 || EltVT==MVT::f128 || EltVT==MVT::ppcf128 ||
           EltVT==MVT::f16) {
    bool ignored;
    APFloat apf = APFloat(Val);
    apf.convert(EVTToAPFloatSemantics(EltVT), APFloat::rmNearestTiesToEven,
                &ignored);
    return getConstantFP(apf, DL, VT, isTarget);
  } else
    llvm_unreachable("Unsupported type in getConstantFP");
}

SDValue SelectionDAG::getGlobalAddress(const GlobalValue *GV, SDLoc DL,
                                       EVT VT, int64_t Offset,
                                       bool isTargetGA,
                                       unsigned char TargetFlags) {
  assert((TargetFlags == 0 || isTargetGA) &&
         "Cannot set target flags on target-independent globals");

  // Truncate (with sign-extension) the offset value to the pointer size.
  unsigned BitWidth = getDataLayout().getPointerTypeSizeInBits(GV->getType());
  if (BitWidth < 64)
    Offset = SignExtend64(Offset, BitWidth);

  unsigned Opc;
  if (GV->isThreadLocal())
    Opc = isTargetGA ? ISD::TargetGlobalTLSAddress : ISD::GlobalTLSAddress;
  else
    Opc = isTargetGA ? ISD::TargetGlobalAddress : ISD::GlobalAddress;

  FoldingSetNodeID ID;
  AddNodeIDNode(ID, Opc, getVTList(VT), None);
  ID.AddPointer(GV);
  ID.AddInteger(Offset);
  ID.AddInteger(TargetFlags);
  ID.AddInteger(GV->getType()->getAddressSpace());
  void *IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, DL.getDebugLoc(), IP))
    return SDValue(E, 0);

  SDNode *N = new (NodeAllocator) GlobalAddressSDNode(Opc, DL.getIROrder(),
                                                      DL.getDebugLoc(), GV, VT,
                                                      Offset, TargetFlags);
  CSEMap.InsertNode(N, IP);
    InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getFrameIndex(int FI, EVT VT, bool isTarget) {
  unsigned Opc = isTarget ? ISD::TargetFrameIndex : ISD::FrameIndex;
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, Opc, getVTList(VT), None);
  ID.AddInteger(FI);
  void *IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, IP))
    return SDValue(E, 0);

  SDNode *N = new (NodeAllocator) FrameIndexSDNode(FI, VT, isTarget);
  CSEMap.InsertNode(N, IP);
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getJumpTable(int JTI, EVT VT, bool isTarget,
                                   unsigned char TargetFlags) {
  assert((TargetFlags == 0 || isTarget) &&
         "Cannot set target flags on target-independent jump tables");
  unsigned Opc = isTarget ? ISD::TargetJumpTable : ISD::JumpTable;
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, Opc, getVTList(VT), None);
  ID.AddInteger(JTI);
  ID.AddInteger(TargetFlags);
  void *IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, IP))
    return SDValue(E, 0);

  SDNode *N = new (NodeAllocator) JumpTableSDNode(JTI, VT, isTarget,
                                                  TargetFlags);
  CSEMap.InsertNode(N, IP);
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getConstantPool(const Constant *C, EVT VT,
                                      unsigned Alignment, int Offset,
                                      bool isTarget,
                                      unsigned char TargetFlags) {
  assert((TargetFlags == 0 || isTarget) &&
         "Cannot set target flags on target-independent globals");
  if (Alignment == 0)
    Alignment = getDataLayout().getPrefTypeAlignment(C->getType());
  unsigned Opc = isTarget ? ISD::TargetConstantPool : ISD::ConstantPool;
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, Opc, getVTList(VT), None);
  ID.AddInteger(Alignment);
  ID.AddInteger(Offset);
  ID.AddPointer(C);
  ID.AddInteger(TargetFlags);
  void *IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, IP))
    return SDValue(E, 0);

  SDNode *N = new (NodeAllocator) ConstantPoolSDNode(isTarget, C, VT, Offset,
                                                     Alignment, TargetFlags);
  CSEMap.InsertNode(N, IP);
  InsertNode(N);
  return SDValue(N, 0);
}


SDValue SelectionDAG::getConstantPool(MachineConstantPoolValue *C, EVT VT,
                                      unsigned Alignment, int Offset,
                                      bool isTarget,
                                      unsigned char TargetFlags) {
  assert((TargetFlags == 0 || isTarget) &&
         "Cannot set target flags on target-independent globals");
  if (Alignment == 0)
    Alignment = getDataLayout().getPrefTypeAlignment(C->getType());
  unsigned Opc = isTarget ? ISD::TargetConstantPool : ISD::ConstantPool;
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, Opc, getVTList(VT), None);
  ID.AddInteger(Alignment);
  ID.AddInteger(Offset);
  C->addSelectionDAGCSEId(ID);
  ID.AddInteger(TargetFlags);
  void *IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, IP))
    return SDValue(E, 0);

  SDNode *N = new (NodeAllocator) ConstantPoolSDNode(isTarget, C, VT, Offset,
                                                     Alignment, TargetFlags);
  CSEMap.InsertNode(N, IP);
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getTargetIndex(int Index, EVT VT, int64_t Offset,
                                     unsigned char TargetFlags) {
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::TargetIndex, getVTList(VT), None);
  ID.AddInteger(Index);
  ID.AddInteger(Offset);
  ID.AddInteger(TargetFlags);
  void *IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, IP))
    return SDValue(E, 0);

  SDNode *N = new (NodeAllocator) TargetIndexSDNode(Index, VT, Offset,
                                                    TargetFlags);
  CSEMap.InsertNode(N, IP);
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getBasicBlock(MachineBasicBlock *MBB) {
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::BasicBlock, getVTList(MVT::Other), None);
  ID.AddPointer(MBB);
  void *IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, IP))
    return SDValue(E, 0);

  SDNode *N = new (NodeAllocator) BasicBlockSDNode(MBB);
  CSEMap.InsertNode(N, IP);
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getValueType(EVT VT) {
  if (VT.isSimple() && (unsigned)VT.getSimpleVT().SimpleTy >=
      ValueTypeNodes.size())
    ValueTypeNodes.resize(VT.getSimpleVT().SimpleTy+1);

  SDNode *&N = VT.isExtended() ?
    ExtendedValueTypeNodes[VT] : ValueTypeNodes[VT.getSimpleVT().SimpleTy];

  if (N) return SDValue(N, 0);
  N = new (NodeAllocator) VTSDNode(VT);
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getExternalSymbol(const char *Sym, EVT VT) {
  SDNode *&N = ExternalSymbols[Sym];
  if (N) return SDValue(N, 0);
  N = new (NodeAllocator) ExternalSymbolSDNode(false, Sym, 0, VT);
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getMCSymbol(MCSymbol *Sym, EVT VT) {
  SDNode *&N = MCSymbols[Sym];
  if (N)
    return SDValue(N, 0);
  N = new (NodeAllocator) MCSymbolSDNode(Sym, VT);
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getTargetExternalSymbol(const char *Sym, EVT VT,
                                              unsigned char TargetFlags) {
  SDNode *&N =
    TargetExternalSymbols[std::pair<std::string,unsigned char>(Sym,
                                                               TargetFlags)];
  if (N) return SDValue(N, 0);
  N = new (NodeAllocator) ExternalSymbolSDNode(true, Sym, TargetFlags, VT);
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getCondCode(ISD::CondCode Cond) {
  if ((unsigned)Cond >= CondCodeNodes.size())
    CondCodeNodes.resize(Cond+1);

  if (!CondCodeNodes[Cond]) {
    CondCodeSDNode *N = new (NodeAllocator) CondCodeSDNode(Cond);
    CondCodeNodes[Cond] = N;
    InsertNode(N);
  }

  return SDValue(CondCodeNodes[Cond], 0);
}

// commuteShuffle - swaps the values of N1 and N2, and swaps all indices in
// the shuffle mask M that point at N1 to point at N2, and indices that point
// N2 to point at N1.
static void commuteShuffle(SDValue &N1, SDValue &N2, SmallVectorImpl<int> &M) {
  std::swap(N1, N2);
  ShuffleVectorSDNode::commuteMask(M);
}

SDValue SelectionDAG::getVectorShuffle(EVT VT, SDLoc dl, SDValue N1,
                                       SDValue N2, const int *Mask) {
  assert(VT == N1.getValueType() && VT == N2.getValueType() &&
         "Invalid VECTOR_SHUFFLE");

  // Canonicalize shuffle undef, undef -> undef
  if (N1.getOpcode() == ISD::UNDEF && N2.getOpcode() == ISD::UNDEF)
    return getUNDEF(VT);

  // Validate that all indices in Mask are within the range of the elements
  // input to the shuffle.
  unsigned NElts = VT.getVectorNumElements();
  SmallVector<int, 8> MaskVec;
  for (unsigned i = 0; i != NElts; ++i) {
    assert(Mask[i] < (int)(NElts * 2) && "Index out of range");
    MaskVec.push_back(Mask[i]);
  }

  // Canonicalize shuffle v, v -> v, undef
  if (N1 == N2) {
    N2 = getUNDEF(VT);
    for (unsigned i = 0; i != NElts; ++i)
      if (MaskVec[i] >= (int)NElts) MaskVec[i] -= NElts;
  }

  // Canonicalize shuffle undef, v -> v, undef.  Commute the shuffle mask.
  if (N1.getOpcode() == ISD::UNDEF)
    commuteShuffle(N1, N2, MaskVec);

  // If shuffling a splat, try to blend the splat instead. We do this here so
  // that even when this arises during lowering we don't have to re-handle it.
  auto BlendSplat = [&](BuildVectorSDNode *BV, int Offset) {
    BitVector UndefElements;
    SDValue Splat = BV->getSplatValue(&UndefElements);
    if (!Splat)
      return;

    for (int i = 0; i < (int)NElts; ++i) {
      if (MaskVec[i] < Offset || MaskVec[i] >= (Offset + (int)NElts))
        continue;

      // If this input comes from undef, mark it as such.
      if (UndefElements[MaskVec[i] - Offset]) {
        MaskVec[i] = -1;
        continue;
      }

      // If we can blend a non-undef lane, use that instead.
      if (!UndefElements[i])
        MaskVec[i] = i + Offset;
    }
  };
  if (auto *N1BV = dyn_cast<BuildVectorSDNode>(N1))
    BlendSplat(N1BV, 0);
  if (auto *N2BV = dyn_cast<BuildVectorSDNode>(N2))
    BlendSplat(N2BV, NElts);

  // Canonicalize all index into lhs, -> shuffle lhs, undef
  // Canonicalize all index into rhs, -> shuffle rhs, undef
  bool AllLHS = true, AllRHS = true;
  bool N2Undef = N2.getOpcode() == ISD::UNDEF;
  for (unsigned i = 0; i != NElts; ++i) {
    if (MaskVec[i] >= (int)NElts) {
      if (N2Undef)
        MaskVec[i] = -1;
      else
        AllLHS = false;
    } else if (MaskVec[i] >= 0) {
      AllRHS = false;
    }
  }
  if (AllLHS && AllRHS)
    return getUNDEF(VT);
  if (AllLHS && !N2Undef)
    N2 = getUNDEF(VT);
  if (AllRHS) {
    N1 = getUNDEF(VT);
    commuteShuffle(N1, N2, MaskVec);
  }
  // Reset our undef status after accounting for the mask.
  N2Undef = N2.getOpcode() == ISD::UNDEF;
  // Re-check whether both sides ended up undef.
  if (N1.getOpcode() == ISD::UNDEF && N2Undef)
    return getUNDEF(VT);

  // If Identity shuffle return that node.
  bool Identity = true, AllSame = true;
  for (unsigned i = 0; i != NElts; ++i) {
    if (MaskVec[i] >= 0 && MaskVec[i] != (int)i) Identity = false;
    if (MaskVec[i] != MaskVec[0]) AllSame = false;
  }
  if (Identity && NElts)
    return N1;

  // Shuffling a constant splat doesn't change the result.
  if (N2Undef) {
    SDValue V = N1;

    // Look through any bitcasts. We check that these don't change the number
    // (and size) of elements and just changes their types.
    while (V.getOpcode() == ISD::BITCAST)
      V = V->getOperand(0);

    // A splat should always show up as a build vector node.
    if (auto *BV = dyn_cast<BuildVectorSDNode>(V)) {
      BitVector UndefElements;
      SDValue Splat = BV->getSplatValue(&UndefElements);
      // If this is a splat of an undef, shuffling it is also undef.
      if (Splat && Splat.getOpcode() == ISD::UNDEF)
        return getUNDEF(VT);

      bool SameNumElts =
          V.getValueType().getVectorNumElements() == VT.getVectorNumElements();

      // We only have a splat which can skip shuffles if there is a splatted
      // value and no undef lanes rearranged by the shuffle.
      if (Splat && UndefElements.none()) {
        // Splat of <x, x, ..., x>, return <x, x, ..., x>, provided that the
        // number of elements match or the value splatted is a zero constant.
        if (SameNumElts)
          return N1;
        if (auto *C = dyn_cast<ConstantSDNode>(Splat))
          if (C->isNullValue())
            return N1;
      }

      // If the shuffle itself creates a splat, build the vector directly.
      if (AllSame && SameNumElts) {
        const SDValue &Splatted = BV->getOperand(MaskVec[0]);
        SmallVector<SDValue, 8> Ops(NElts, Splatted);

        EVT BuildVT = BV->getValueType(0);
        SDValue NewBV = getNode(ISD::BUILD_VECTOR, dl, BuildVT, Ops);

        // We may have jumped through bitcasts, so the type of the
        // BUILD_VECTOR may not match the type of the shuffle.
        if (BuildVT != VT)
          NewBV = getNode(ISD::BITCAST, dl, VT, NewBV);
        return NewBV;
      }
    }
  }

  FoldingSetNodeID ID;
  SDValue Ops[2] = { N1, N2 };
  AddNodeIDNode(ID, ISD::VECTOR_SHUFFLE, getVTList(VT), Ops);
  for (unsigned i = 0; i != NElts; ++i)
    ID.AddInteger(MaskVec[i]);

  void* IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, dl.getDebugLoc(), IP))
    return SDValue(E, 0);

  // Allocate the mask array for the node out of the BumpPtrAllocator, since
  // SDNode doesn't have access to it.  This memory will be "leaked" when
  // the node is deallocated, but recovered when the NodeAllocator is released.
  int *MaskAlloc = OperandAllocator.Allocate<int>(NElts);
  memcpy(MaskAlloc, &MaskVec[0], NElts * sizeof(int));

  ShuffleVectorSDNode *N =
    new (NodeAllocator) ShuffleVectorSDNode(VT, dl.getIROrder(),
                                            dl.getDebugLoc(), N1, N2,
                                            MaskAlloc);
  CSEMap.InsertNode(N, IP);
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getCommutedVectorShuffle(const ShuffleVectorSDNode &SV) {
  MVT VT = SV.getSimpleValueType(0);
  SmallVector<int, 8> MaskVec(SV.getMask().begin(), SV.getMask().end());
  ShuffleVectorSDNode::commuteMask(MaskVec);

  SDValue Op0 = SV.getOperand(0);
  SDValue Op1 = SV.getOperand(1);
  return getVectorShuffle(VT, SDLoc(&SV), Op1, Op0, &MaskVec[0]);
}

SDValue SelectionDAG::getConvertRndSat(EVT VT, SDLoc dl,
                                       SDValue Val, SDValue DTy,
                                       SDValue STy, SDValue Rnd, SDValue Sat,
                                       ISD::CvtCode Code) {
  // If the src and dest types are the same and the conversion is between
  // integer types of the same sign or two floats, no conversion is necessary.
  if (DTy == STy &&
      (Code == ISD::CVT_UU || Code == ISD::CVT_SS || Code == ISD::CVT_FF))
    return Val;

  FoldingSetNodeID ID;
  SDValue Ops[] = { Val, DTy, STy, Rnd, Sat };
  AddNodeIDNode(ID, ISD::CONVERT_RNDSAT, getVTList(VT), Ops);
  void* IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, dl.getDebugLoc(), IP))
    return SDValue(E, 0);

  CvtRndSatSDNode *N = new (NodeAllocator) CvtRndSatSDNode(VT, dl.getIROrder(),
                                                           dl.getDebugLoc(),
                                                           Ops, Code);
  CSEMap.InsertNode(N, IP);
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getRegister(unsigned RegNo, EVT VT) {
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::Register, getVTList(VT), None);
  ID.AddInteger(RegNo);
  void *IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, IP))
    return SDValue(E, 0);

  SDNode *N = new (NodeAllocator) RegisterSDNode(RegNo, VT);
  CSEMap.InsertNode(N, IP);
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getRegisterMask(const uint32_t *RegMask) {
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::RegisterMask, getVTList(MVT::Untyped), None);
  ID.AddPointer(RegMask);
  void *IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, IP))
    return SDValue(E, 0);

  SDNode *N = new (NodeAllocator) RegisterMaskSDNode(RegMask);
  CSEMap.InsertNode(N, IP);
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getEHLabel(SDLoc dl, SDValue Root, MCSymbol *Label) {
  FoldingSetNodeID ID;
  SDValue Ops[] = { Root };
  AddNodeIDNode(ID, ISD::EH_LABEL, getVTList(MVT::Other), Ops);
  ID.AddPointer(Label);
  void *IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, IP))
    return SDValue(E, 0);

  SDNode *N = new (NodeAllocator) EHLabelSDNode(dl.getIROrder(),
                                                dl.getDebugLoc(), Root, Label);
  CSEMap.InsertNode(N, IP);
  InsertNode(N);
  return SDValue(N, 0);
}


SDValue SelectionDAG::getBlockAddress(const BlockAddress *BA, EVT VT,
                                      int64_t Offset,
                                      bool isTarget,
                                      unsigned char TargetFlags) {
  unsigned Opc = isTarget ? ISD::TargetBlockAddress : ISD::BlockAddress;

  FoldingSetNodeID ID;
  AddNodeIDNode(ID, Opc, getVTList(VT), None);
  ID.AddPointer(BA);
  ID.AddInteger(Offset);
  ID.AddInteger(TargetFlags);
  void *IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, IP))
    return SDValue(E, 0);

  SDNode *N = new (NodeAllocator) BlockAddressSDNode(Opc, VT, BA, Offset,
                                                     TargetFlags);
  CSEMap.InsertNode(N, IP);
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getSrcValue(const Value *V) {
  assert((!V || V->getType()->isPointerTy()) &&
         "SrcValue is not a pointer?");

  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::SRCVALUE, getVTList(MVT::Other), None);
  ID.AddPointer(V);

  void *IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, IP))
    return SDValue(E, 0);

  SDNode *N = new (NodeAllocator) SrcValueSDNode(V);
  CSEMap.InsertNode(N, IP);
  InsertNode(N);
  return SDValue(N, 0);
}

/// getMDNode - Return an MDNodeSDNode which holds an MDNode.
SDValue SelectionDAG::getMDNode(const MDNode *MD) {
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::MDNODE_SDNODE, getVTList(MVT::Other), None);
  ID.AddPointer(MD);

  void *IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, IP))
    return SDValue(E, 0);

  SDNode *N = new (NodeAllocator) MDNodeSDNode(MD);
  CSEMap.InsertNode(N, IP);
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getBitcast(EVT VT, SDValue V) {
  if (VT == V.getValueType())
    return V;

  return getNode(ISD::BITCAST, SDLoc(V), VT, V);
}

/// getAddrSpaceCast - Return an AddrSpaceCastSDNode.
SDValue SelectionDAG::getAddrSpaceCast(SDLoc dl, EVT VT, SDValue Ptr,
                                       unsigned SrcAS, unsigned DestAS) {
  SDValue Ops[] = {Ptr};
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::ADDRSPACECAST, getVTList(VT), Ops);
  ID.AddInteger(SrcAS);
  ID.AddInteger(DestAS);

  void *IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, dl.getDebugLoc(), IP))
    return SDValue(E, 0);

  SDNode *N = new (NodeAllocator) AddrSpaceCastSDNode(dl.getIROrder(),
                                                      dl.getDebugLoc(),
                                                      VT, Ptr, SrcAS, DestAS);
  CSEMap.InsertNode(N, IP);
  InsertNode(N);
  return SDValue(N, 0);
}

/// getShiftAmountOperand - Return the specified value casted to
/// the target's desired shift amount type.
SDValue SelectionDAG::getShiftAmountOperand(EVT LHSTy, SDValue Op) {
  EVT OpTy = Op.getValueType();
  EVT ShTy = TLI->getShiftAmountTy(LHSTy, getDataLayout());
  if (OpTy == ShTy || OpTy.isVector()) return Op;

  ISD::NodeType Opcode = OpTy.bitsGT(ShTy) ?  ISD::TRUNCATE : ISD::ZERO_EXTEND;
  return getNode(Opcode, SDLoc(Op), ShTy, Op);
}

/// CreateStackTemporary - Create a stack temporary, suitable for holding the
/// specified value type.
SDValue SelectionDAG::CreateStackTemporary(EVT VT, unsigned minAlign) {
  MachineFrameInfo *FrameInfo = getMachineFunction().getFrameInfo();
  unsigned ByteSize = VT.getStoreSize();
  Type *Ty = VT.getTypeForEVT(*getContext());
  unsigned StackAlign =
      std::max((unsigned)getDataLayout().getPrefTypeAlignment(Ty), minAlign);

  int FrameIdx = FrameInfo->CreateStackObject(ByteSize, StackAlign, false);
  return getFrameIndex(FrameIdx, TLI->getPointerTy(getDataLayout()));
}

/// CreateStackTemporary - Create a stack temporary suitable for holding
/// either of the specified value types.
SDValue SelectionDAG::CreateStackTemporary(EVT VT1, EVT VT2) {
  unsigned Bytes = std::max(VT1.getStoreSizeInBits(),
                            VT2.getStoreSizeInBits())/8;
  Type *Ty1 = VT1.getTypeForEVT(*getContext());
  Type *Ty2 = VT2.getTypeForEVT(*getContext());
  const DataLayout &DL = getDataLayout();
  unsigned Align =
      std::max(DL.getPrefTypeAlignment(Ty1), DL.getPrefTypeAlignment(Ty2));

  MachineFrameInfo *FrameInfo = getMachineFunction().getFrameInfo();
  int FrameIdx = FrameInfo->CreateStackObject(Bytes, Align, false);
  return getFrameIndex(FrameIdx, TLI->getPointerTy(getDataLayout()));
}

SDValue SelectionDAG::FoldSetCC(EVT VT, SDValue N1,
                                SDValue N2, ISD::CondCode Cond, SDLoc dl) {
  // These setcc operations always fold.
  switch (Cond) {
  default: break;
  case ISD::SETFALSE:
  case ISD::SETFALSE2: return getConstant(0, dl, VT);
  case ISD::SETTRUE:
  case ISD::SETTRUE2: {
    TargetLowering::BooleanContent Cnt =
        TLI->getBooleanContents(N1->getValueType(0));
    return getConstant(
        Cnt == TargetLowering::ZeroOrNegativeOneBooleanContent ? -1ULL : 1, dl,
        VT);
  }

  case ISD::SETOEQ:
  case ISD::SETOGT:
  case ISD::SETOGE:
  case ISD::SETOLT:
  case ISD::SETOLE:
  case ISD::SETONE:
  case ISD::SETO:
  case ISD::SETUO:
  case ISD::SETUEQ:
  case ISD::SETUNE:
    assert(!N1.getValueType().isInteger() && "Illegal setcc for integer!");
    break;
  }

  if (ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(N2)) {
    const APInt &C2 = N2C->getAPIntValue();
    if (ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1)) {
      const APInt &C1 = N1C->getAPIntValue();

      switch (Cond) {
      default: llvm_unreachable("Unknown integer setcc!");
      case ISD::SETEQ:  return getConstant(C1 == C2, dl, VT);
      case ISD::SETNE:  return getConstant(C1 != C2, dl, VT);
      case ISD::SETULT: return getConstant(C1.ult(C2), dl, VT);
      case ISD::SETUGT: return getConstant(C1.ugt(C2), dl, VT);
      case ISD::SETULE: return getConstant(C1.ule(C2), dl, VT);
      case ISD::SETUGE: return getConstant(C1.uge(C2), dl, VT);
      case ISD::SETLT:  return getConstant(C1.slt(C2), dl, VT);
      case ISD::SETGT:  return getConstant(C1.sgt(C2), dl, VT);
      case ISD::SETLE:  return getConstant(C1.sle(C2), dl, VT);
      case ISD::SETGE:  return getConstant(C1.sge(C2), dl, VT);
      }
    }
  }
  if (ConstantFPSDNode *N1C = dyn_cast<ConstantFPSDNode>(N1)) {
    if (ConstantFPSDNode *N2C = dyn_cast<ConstantFPSDNode>(N2)) {
      APFloat::cmpResult R = N1C->getValueAPF().compare(N2C->getValueAPF());
      switch (Cond) {
      default: break;
      case ISD::SETEQ:  if (R==APFloat::cmpUnordered)
                          return getUNDEF(VT);
                        // fall through
      case ISD::SETOEQ: return getConstant(R==APFloat::cmpEqual, dl, VT);
      case ISD::SETNE:  if (R==APFloat::cmpUnordered)
                          return getUNDEF(VT);
                        // fall through
      case ISD::SETONE: return getConstant(R==APFloat::cmpGreaterThan ||
                                           R==APFloat::cmpLessThan, dl, VT);
      case ISD::SETLT:  if (R==APFloat::cmpUnordered)
                          return getUNDEF(VT);
                        // fall through
      case ISD::SETOLT: return getConstant(R==APFloat::cmpLessThan, dl, VT);
      case ISD::SETGT:  if (R==APFloat::cmpUnordered)
                          return getUNDEF(VT);
                        // fall through
      case ISD::SETOGT: return getConstant(R==APFloat::cmpGreaterThan, dl, VT);
      case ISD::SETLE:  if (R==APFloat::cmpUnordered)
                          return getUNDEF(VT);
                        // fall through
      case ISD::SETOLE: return getConstant(R==APFloat::cmpLessThan ||
                                           R==APFloat::cmpEqual, dl, VT);
      case ISD::SETGE:  if (R==APFloat::cmpUnordered)
                          return getUNDEF(VT);
                        // fall through
      case ISD::SETOGE: return getConstant(R==APFloat::cmpGreaterThan ||
                                           R==APFloat::cmpEqual, dl, VT);
      case ISD::SETO:   return getConstant(R!=APFloat::cmpUnordered, dl, VT);
      case ISD::SETUO:  return getConstant(R==APFloat::cmpUnordered, dl, VT);
      case ISD::SETUEQ: return getConstant(R==APFloat::cmpUnordered ||
                                           R==APFloat::cmpEqual, dl, VT);
      case ISD::SETUNE: return getConstant(R!=APFloat::cmpEqual, dl, VT);
      case ISD::SETULT: return getConstant(R==APFloat::cmpUnordered ||
                                           R==APFloat::cmpLessThan, dl, VT);
      case ISD::SETUGT: return getConstant(R==APFloat::cmpGreaterThan ||
                                           R==APFloat::cmpUnordered, dl, VT);
      case ISD::SETULE: return getConstant(R!=APFloat::cmpGreaterThan, dl, VT);
      case ISD::SETUGE: return getConstant(R!=APFloat::cmpLessThan, dl, VT);
      }
    } else {
      // Ensure that the constant occurs on the RHS.
      ISD::CondCode SwappedCond = ISD::getSetCCSwappedOperands(Cond);
      MVT CompVT = N1.getValueType().getSimpleVT();
      if (!TLI->isCondCodeLegal(SwappedCond, CompVT))
        return SDValue();

      return getSetCC(dl, VT, N2, N1, SwappedCond);
    }
  }

  // Could not fold it.
  return SDValue();
}

/// SignBitIsZero - Return true if the sign bit of Op is known to be zero.  We
/// use this predicate to simplify operations downstream.
bool SelectionDAG::SignBitIsZero(SDValue Op, unsigned Depth) const {
  // This predicate is not safe for vector operations.
  if (Op.getValueType().isVector())
    return false;

  unsigned BitWidth = Op.getValueType().getScalarType().getSizeInBits();
  return MaskedValueIsZero(Op, APInt::getSignBit(BitWidth), Depth);
}

/// MaskedValueIsZero - Return true if 'V & Mask' is known to be zero.  We use
/// this predicate to simplify operations downstream.  Mask is known to be zero
/// for bits that V cannot have.
bool SelectionDAG::MaskedValueIsZero(SDValue Op, const APInt &Mask,
                                     unsigned Depth) const {
  APInt KnownZero, KnownOne;
  computeKnownBits(Op, KnownZero, KnownOne, Depth);
  return (KnownZero & Mask) == Mask;
}

/// Determine which bits of Op are known to be either zero or one and return
/// them in the KnownZero/KnownOne bitsets.
void SelectionDAG::computeKnownBits(SDValue Op, APInt &KnownZero,
                                    APInt &KnownOne, unsigned Depth) const {
  unsigned BitWidth = Op.getValueType().getScalarType().getSizeInBits();

  KnownZero = KnownOne = APInt(BitWidth, 0);   // Don't know anything.
  if (Depth == 6)
    return;  // Limit search depth.

  APInt KnownZero2, KnownOne2;

  switch (Op.getOpcode()) {
  case ISD::Constant:
    // We know all of the bits for a constant!
    KnownOne = cast<ConstantSDNode>(Op)->getAPIntValue();
    KnownZero = ~KnownOne;
    break;
  case ISD::AND:
    // If either the LHS or the RHS are Zero, the result is zero.
    computeKnownBits(Op.getOperand(1), KnownZero, KnownOne, Depth+1);
    computeKnownBits(Op.getOperand(0), KnownZero2, KnownOne2, Depth+1);

    // Output known-1 bits are only known if set in both the LHS & RHS.
    KnownOne &= KnownOne2;
    // Output known-0 are known to be clear if zero in either the LHS | RHS.
    KnownZero |= KnownZero2;
    break;
  case ISD::OR:
    computeKnownBits(Op.getOperand(1), KnownZero, KnownOne, Depth+1);
    computeKnownBits(Op.getOperand(0), KnownZero2, KnownOne2, Depth+1);

    // Output known-0 bits are only known if clear in both the LHS & RHS.
    KnownZero &= KnownZero2;
    // Output known-1 are known to be set if set in either the LHS | RHS.
    KnownOne |= KnownOne2;
    break;
  case ISD::XOR: {
    computeKnownBits(Op.getOperand(1), KnownZero, KnownOne, Depth+1);
    computeKnownBits(Op.getOperand(0), KnownZero2, KnownOne2, Depth+1);

    // Output known-0 bits are known if clear or set in both the LHS & RHS.
    APInt KnownZeroOut = (KnownZero & KnownZero2) | (KnownOne & KnownOne2);
    // Output known-1 are known to be set if set in only one of the LHS, RHS.
    KnownOne = (KnownZero & KnownOne2) | (KnownOne & KnownZero2);
    KnownZero = KnownZeroOut;
    break;
  }
  case ISD::MUL: {
    computeKnownBits(Op.getOperand(1), KnownZero, KnownOne, Depth+1);
    computeKnownBits(Op.getOperand(0), KnownZero2, KnownOne2, Depth+1);

    // If low bits are zero in either operand, output low known-0 bits.
    // Also compute a conserative estimate for high known-0 bits.
    // More trickiness is possible, but this is sufficient for the
    // interesting case of alignment computation.
    KnownOne.clearAllBits();
    unsigned TrailZ = KnownZero.countTrailingOnes() +
                      KnownZero2.countTrailingOnes();
    unsigned LeadZ =  std::max(KnownZero.countLeadingOnes() +
                               KnownZero2.countLeadingOnes(),
                               BitWidth) - BitWidth;

    TrailZ = std::min(TrailZ, BitWidth);
    LeadZ = std::min(LeadZ, BitWidth);
    KnownZero = APInt::getLowBitsSet(BitWidth, TrailZ) |
                APInt::getHighBitsSet(BitWidth, LeadZ);
    break;
  }
  case ISD::UDIV: {
    // For the purposes of computing leading zeros we can conservatively
    // treat a udiv as a logical right shift by the power of 2 known to
    // be less than the denominator.
    computeKnownBits(Op.getOperand(0), KnownZero2, KnownOne2, Depth+1);
    unsigned LeadZ = KnownZero2.countLeadingOnes();

    KnownOne2.clearAllBits();
    KnownZero2.clearAllBits();
    computeKnownBits(Op.getOperand(1), KnownZero2, KnownOne2, Depth+1);
    unsigned RHSUnknownLeadingOnes = KnownOne2.countLeadingZeros();
    if (RHSUnknownLeadingOnes != BitWidth)
      LeadZ = std::min(BitWidth,
                       LeadZ + BitWidth - RHSUnknownLeadingOnes - 1);

    KnownZero = APInt::getHighBitsSet(BitWidth, LeadZ);
    break;
  }
  case ISD::SELECT:
    computeKnownBits(Op.getOperand(2), KnownZero, KnownOne, Depth+1);
    computeKnownBits(Op.getOperand(1), KnownZero2, KnownOne2, Depth+1);

    // Only known if known in both the LHS and RHS.
    KnownOne &= KnownOne2;
    KnownZero &= KnownZero2;
    break;
  case ISD::SELECT_CC:
    computeKnownBits(Op.getOperand(3), KnownZero, KnownOne, Depth+1);
    computeKnownBits(Op.getOperand(2), KnownZero2, KnownOne2, Depth+1);

    // Only known if known in both the LHS and RHS.
    KnownOne &= KnownOne2;
    KnownZero &= KnownZero2;
    break;
  case ISD::SADDO:
  case ISD::UADDO:
  case ISD::SSUBO:
  case ISD::USUBO:
  case ISD::SMULO:
  case ISD::UMULO:
    if (Op.getResNo() != 1)
      break;
    // The boolean result conforms to getBooleanContents.
    // If we know the result of a setcc has the top bits zero, use this info.
    // We know that we have an integer-based boolean since these operations
    // are only available for integer.
    if (TLI->getBooleanContents(Op.getValueType().isVector(), false) ==
            TargetLowering::ZeroOrOneBooleanContent &&
        BitWidth > 1)
      KnownZero |= APInt::getHighBitsSet(BitWidth, BitWidth - 1);
    break;
  case ISD::SETCC:
    // If we know the result of a setcc has the top bits zero, use this info.
    if (TLI->getBooleanContents(Op.getOperand(0).getValueType()) ==
            TargetLowering::ZeroOrOneBooleanContent &&
        BitWidth > 1)
      KnownZero |= APInt::getHighBitsSet(BitWidth, BitWidth - 1);
    break;
  case ISD::SHL:
    // (shl X, C1) & C2 == 0   iff   (X & C2 >>u C1) == 0
    if (ConstantSDNode *SA = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      unsigned ShAmt = SA->getZExtValue();

      // If the shift count is an invalid immediate, don't do anything.
      if (ShAmt >= BitWidth)
        break;

      computeKnownBits(Op.getOperand(0), KnownZero, KnownOne, Depth+1);
      KnownZero <<= ShAmt;
      KnownOne  <<= ShAmt;
      // low bits known zero.
      KnownZero |= APInt::getLowBitsSet(BitWidth, ShAmt);
    }
    break;
  case ISD::SRL:
    // (ushr X, C1) & C2 == 0   iff  (-1 >> C1) & C2 == 0
    if (ConstantSDNode *SA = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      unsigned ShAmt = SA->getZExtValue();

      // If the shift count is an invalid immediate, don't do anything.
      if (ShAmt >= BitWidth)
        break;

      computeKnownBits(Op.getOperand(0), KnownZero, KnownOne, Depth+1);
      KnownZero = KnownZero.lshr(ShAmt);
      KnownOne  = KnownOne.lshr(ShAmt);

      APInt HighBits = APInt::getHighBitsSet(BitWidth, ShAmt);
      KnownZero |= HighBits;  // High bits known zero.
    }
    break;
  case ISD::SRA:
    if (ConstantSDNode *SA = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      unsigned ShAmt = SA->getZExtValue();

      // If the shift count is an invalid immediate, don't do anything.
      if (ShAmt >= BitWidth)
        break;

      // If any of the demanded bits are produced by the sign extension, we also
      // demand the input sign bit.
      APInt HighBits = APInt::getHighBitsSet(BitWidth, ShAmt);

      computeKnownBits(Op.getOperand(0), KnownZero, KnownOne, Depth+1);
      KnownZero = KnownZero.lshr(ShAmt);
      KnownOne  = KnownOne.lshr(ShAmt);

      // Handle the sign bits.
      APInt SignBit = APInt::getSignBit(BitWidth);
      SignBit = SignBit.lshr(ShAmt);  // Adjust to where it is now in the mask.

      if (KnownZero.intersects(SignBit)) {
        KnownZero |= HighBits;  // New bits are known zero.
      } else if (KnownOne.intersects(SignBit)) {
        KnownOne  |= HighBits;  // New bits are known one.
      }
    }
    break;
  case ISD::SIGN_EXTEND_INREG: {
    EVT EVT = cast<VTSDNode>(Op.getOperand(1))->getVT();
    unsigned EBits = EVT.getScalarType().getSizeInBits();

    // Sign extension.  Compute the demanded bits in the result that are not
    // present in the input.
    APInt NewBits = APInt::getHighBitsSet(BitWidth, BitWidth - EBits);

    APInt InSignBit = APInt::getSignBit(EBits);
    APInt InputDemandedBits = APInt::getLowBitsSet(BitWidth, EBits);

    // If the sign extended bits are demanded, we know that the sign
    // bit is demanded.
    InSignBit = InSignBit.zext(BitWidth);
    if (NewBits.getBoolValue())
      InputDemandedBits |= InSignBit;

    computeKnownBits(Op.getOperand(0), KnownZero, KnownOne, Depth+1);
    KnownOne &= InputDemandedBits;
    KnownZero &= InputDemandedBits;

    // If the sign bit of the input is known set or clear, then we know the
    // top bits of the result.
    if (KnownZero.intersects(InSignBit)) {         // Input sign bit known clear
      KnownZero |= NewBits;
      KnownOne  &= ~NewBits;
    } else if (KnownOne.intersects(InSignBit)) {   // Input sign bit known set
      KnownOne  |= NewBits;
      KnownZero &= ~NewBits;
    } else {                              // Input sign bit unknown
      KnownZero &= ~NewBits;
      KnownOne  &= ~NewBits;
    }
    break;
  }
  case ISD::CTTZ:
  case ISD::CTTZ_ZERO_UNDEF:
  case ISD::CTLZ:
  case ISD::CTLZ_ZERO_UNDEF:
  case ISD::CTPOP: {
    unsigned LowBits = Log2_32(BitWidth)+1;
    KnownZero = APInt::getHighBitsSet(BitWidth, BitWidth - LowBits);
    KnownOne.clearAllBits();
    break;
  }
  case ISD::LOAD: {
    LoadSDNode *LD = cast<LoadSDNode>(Op);
    // If this is a ZEXTLoad and we are looking at the loaded value.
    if (ISD::isZEXTLoad(Op.getNode()) && Op.getResNo() == 0) {
      EVT VT = LD->getMemoryVT();
      unsigned MemBits = VT.getScalarType().getSizeInBits();
      KnownZero |= APInt::getHighBitsSet(BitWidth, BitWidth - MemBits);
    } else if (const MDNode *Ranges = LD->getRanges()) {
      computeKnownBitsFromRangeMetadata(*Ranges, KnownZero);
    }
    break;
  }
  case ISD::ZERO_EXTEND: {
    EVT InVT = Op.getOperand(0).getValueType();
    unsigned InBits = InVT.getScalarType().getSizeInBits();
    APInt NewBits   = APInt::getHighBitsSet(BitWidth, BitWidth - InBits);
    KnownZero = KnownZero.trunc(InBits);
    KnownOne = KnownOne.trunc(InBits);
    computeKnownBits(Op.getOperand(0), KnownZero, KnownOne, Depth+1);
    KnownZero = KnownZero.zext(BitWidth);
    KnownOne = KnownOne.zext(BitWidth);
    KnownZero |= NewBits;
    break;
  }
  case ISD::SIGN_EXTEND: {
    EVT InVT = Op.getOperand(0).getValueType();
    unsigned InBits = InVT.getScalarType().getSizeInBits();
    APInt NewBits   = APInt::getHighBitsSet(BitWidth, BitWidth - InBits);

    KnownZero = KnownZero.trunc(InBits);
    KnownOne = KnownOne.trunc(InBits);
    computeKnownBits(Op.getOperand(0), KnownZero, KnownOne, Depth+1);

    // Note if the sign bit is known to be zero or one.
    bool SignBitKnownZero = KnownZero.isNegative();
    bool SignBitKnownOne  = KnownOne.isNegative();

    KnownZero = KnownZero.zext(BitWidth);
    KnownOne = KnownOne.zext(BitWidth);

    // If the sign bit is known zero or one, the top bits match.
    if (SignBitKnownZero)
      KnownZero |= NewBits;
    else if (SignBitKnownOne)
      KnownOne  |= NewBits;
    break;
  }
  case ISD::ANY_EXTEND: {
    EVT InVT = Op.getOperand(0).getValueType();
    unsigned InBits = InVT.getScalarType().getSizeInBits();
    KnownZero = KnownZero.trunc(InBits);
    KnownOne = KnownOne.trunc(InBits);
    computeKnownBits(Op.getOperand(0), KnownZero, KnownOne, Depth+1);
    KnownZero = KnownZero.zext(BitWidth);
    KnownOne = KnownOne.zext(BitWidth);
    break;
  }
  case ISD::TRUNCATE: {
    EVT InVT = Op.getOperand(0).getValueType();
    unsigned InBits = InVT.getScalarType().getSizeInBits();
    KnownZero = KnownZero.zext(InBits);
    KnownOne = KnownOne.zext(InBits);
    computeKnownBits(Op.getOperand(0), KnownZero, KnownOne, Depth+1);
    KnownZero = KnownZero.trunc(BitWidth);
    KnownOne = KnownOne.trunc(BitWidth);
    break;
  }
  case ISD::AssertZext: {
    EVT VT = cast<VTSDNode>(Op.getOperand(1))->getVT();
    APInt InMask = APInt::getLowBitsSet(BitWidth, VT.getSizeInBits());
    computeKnownBits(Op.getOperand(0), KnownZero, KnownOne, Depth+1);
    KnownZero |= (~InMask);
    KnownOne  &= (~KnownZero);
    break;
  }
  case ISD::FGETSIGN:
    // All bits are zero except the low bit.
    KnownZero = APInt::getHighBitsSet(BitWidth, BitWidth - 1);
    break;

  case ISD::SUB: {
    if (ConstantSDNode *CLHS = dyn_cast<ConstantSDNode>(Op.getOperand(0))) {
      // We know that the top bits of C-X are clear if X contains less bits
      // than C (i.e. no wrap-around can happen).  For example, 20-X is
      // positive if we can prove that X is >= 0 and < 16.
      if (CLHS->getAPIntValue().isNonNegative()) {
        unsigned NLZ = (CLHS->getAPIntValue()+1).countLeadingZeros();
        // NLZ can't be BitWidth with no sign bit
        APInt MaskV = APInt::getHighBitsSet(BitWidth, NLZ+1);
        computeKnownBits(Op.getOperand(1), KnownZero2, KnownOne2, Depth+1);

        // If all of the MaskV bits are known to be zero, then we know the
        // output top bits are zero, because we now know that the output is
        // from [0-C].
        if ((KnownZero2 & MaskV) == MaskV) {
          unsigned NLZ2 = CLHS->getAPIntValue().countLeadingZeros();
          // Top bits known zero.
          KnownZero = APInt::getHighBitsSet(BitWidth, NLZ2);
        }
      }
    }
  }
  // fall through
  case ISD::ADD:
  case ISD::ADDE: {
    // Output known-0 bits are known if clear or set in both the low clear bits
    // common to both LHS & RHS.  For example, 8+(X<<3) is known to have the
    // low 3 bits clear.
    // Output known-0 bits are also known if the top bits of each input are
    // known to be clear. For example, if one input has the top 10 bits clear
    // and the other has the top 8 bits clear, we know the top 7 bits of the
    // output must be clear.
    computeKnownBits(Op.getOperand(0), KnownZero2, KnownOne2, Depth+1);
    unsigned KnownZeroHigh = KnownZero2.countLeadingOnes();
    unsigned KnownZeroLow = KnownZero2.countTrailingOnes();

    computeKnownBits(Op.getOperand(1), KnownZero2, KnownOne2, Depth+1);
    KnownZeroHigh = std::min(KnownZeroHigh,
                             KnownZero2.countLeadingOnes());
    KnownZeroLow = std::min(KnownZeroLow,
                            KnownZero2.countTrailingOnes());

    if (Op.getOpcode() == ISD::ADD) {
      KnownZero |= APInt::getLowBitsSet(BitWidth, KnownZeroLow);
      if (KnownZeroHigh > 1)
        KnownZero |= APInt::getHighBitsSet(BitWidth, KnownZeroHigh - 1);
      break;
    }

    // With ADDE, a carry bit may be added in, so we can only use this
    // information if we know (at least) that the low two bits are clear.  We
    // then return to the caller that the low bit is unknown but that other bits
    // are known zero.
    if (KnownZeroLow >= 2) // ADDE
      KnownZero |= APInt::getBitsSet(BitWidth, 1, KnownZeroLow);
    break;
  }
  case ISD::SREM:
    if (ConstantSDNode *Rem = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      const APInt &RA = Rem->getAPIntValue().abs();
      if (RA.isPowerOf2()) {
        APInt LowBits = RA - 1;
        computeKnownBits(Op.getOperand(0), KnownZero2,KnownOne2,Depth+1);

        // The low bits of the first operand are unchanged by the srem.
        KnownZero = KnownZero2 & LowBits;
        KnownOne = KnownOne2 & LowBits;

        // If the first operand is non-negative or has all low bits zero, then
        // the upper bits are all zero.
        if (KnownZero2[BitWidth-1] || ((KnownZero2 & LowBits) == LowBits))
          KnownZero |= ~LowBits;

        // If the first operand is negative and not all low bits are zero, then
        // the upper bits are all one.
        if (KnownOne2[BitWidth-1] && ((KnownOne2 & LowBits) != 0))
          KnownOne |= ~LowBits;
        assert((KnownZero & KnownOne) == 0&&"Bits known to be one AND zero?");
      }
    }
    break;
  case ISD::UREM: {
    if (ConstantSDNode *Rem = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      const APInt &RA = Rem->getAPIntValue();
      if (RA.isPowerOf2()) {
        APInt LowBits = (RA - 1);
        computeKnownBits(Op.getOperand(0), KnownZero2, KnownOne2, Depth + 1);

        // The upper bits are all zero, the lower ones are unchanged.
        KnownZero = KnownZero2 | ~LowBits;
        KnownOne = KnownOne2 & LowBits;
        break;
      }
    }

    // Since the result is less than or equal to either operand, any leading
    // zero bits in either operand must also exist in the result.
    computeKnownBits(Op.getOperand(0), KnownZero, KnownOne, Depth+1);
    computeKnownBits(Op.getOperand(1), KnownZero2, KnownOne2, Depth+1);

    uint32_t Leaders = std::max(KnownZero.countLeadingOnes(),
                                KnownZero2.countLeadingOnes());
    KnownOne.clearAllBits();
    KnownZero = APInt::getHighBitsSet(BitWidth, Leaders);
    break;
  }
  case ISD::EXTRACT_ELEMENT: {
    computeKnownBits(Op.getOperand(0), KnownZero, KnownOne, Depth+1);
    const unsigned Index =
      cast<ConstantSDNode>(Op.getOperand(1))->getZExtValue();
    const unsigned BitWidth = Op.getValueType().getSizeInBits();

    // Remove low part of known bits mask
    KnownZero = KnownZero.getHiBits(KnownZero.getBitWidth() - Index * BitWidth);
    KnownOne = KnownOne.getHiBits(KnownOne.getBitWidth() - Index * BitWidth);

    // Remove high part of known bit mask
    KnownZero = KnownZero.trunc(BitWidth);
    KnownOne = KnownOne.trunc(BitWidth);
    break;
  }
  case ISD::SMIN:
  case ISD::SMAX:
  case ISD::UMIN:
  case ISD::UMAX: {
    APInt Op0Zero, Op0One;
    APInt Op1Zero, Op1One;
    computeKnownBits(Op.getOperand(0), Op0Zero, Op0One, Depth);
    computeKnownBits(Op.getOperand(1), Op1Zero, Op1One, Depth);

    KnownZero = Op0Zero & Op1Zero;
    KnownOne = Op0One & Op1One;
    break;
  }
  case ISD::FrameIndex:
  case ISD::TargetFrameIndex:
    if (unsigned Align = InferPtrAlignment(Op)) {
      // The low bits are known zero if the pointer is aligned.
      KnownZero = APInt::getLowBitsSet(BitWidth, Log2_32(Align));
      break;
    }
    break;

  default:
    if (Op.getOpcode() < ISD::BUILTIN_OP_END)
      break;
    // Fallthrough
  case ISD::INTRINSIC_WO_CHAIN:
  case ISD::INTRINSIC_W_CHAIN:
  case ISD::INTRINSIC_VOID:
    // Allow the target to implement this method for its nodes.
    TLI->computeKnownBitsForTargetNode(Op, KnownZero, KnownOne, *this, Depth);
    break;
  }

  assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?");
}

/// ComputeNumSignBits - Return the number of times the sign bit of the
/// register is replicated into the other bits.  We know that at least 1 bit
/// is always equal to the sign bit (itself), but other cases can give us
/// information.  For example, immediately after an "SRA X, 2", we know that
/// the top 3 bits are all equal to each other, so we return 3.
unsigned SelectionDAG::ComputeNumSignBits(SDValue Op, unsigned Depth) const{
  EVT VT = Op.getValueType();
  assert(VT.isInteger() && "Invalid VT!");
  unsigned VTBits = VT.getScalarType().getSizeInBits();
  unsigned Tmp, Tmp2;
  unsigned FirstAnswer = 1;

  if (Depth == 6)
    return 1;  // Limit search depth.

  switch (Op.getOpcode()) {
  default: break;
  case ISD::AssertSext:
    Tmp = cast<VTSDNode>(Op.getOperand(1))->getVT().getSizeInBits();
    return VTBits-Tmp+1;
  case ISD::AssertZext:
    Tmp = cast<VTSDNode>(Op.getOperand(1))->getVT().getSizeInBits();
    return VTBits-Tmp;

  case ISD::Constant: {
    const APInt &Val = cast<ConstantSDNode>(Op)->getAPIntValue();
    return Val.getNumSignBits();
  }

  case ISD::SIGN_EXTEND:
    Tmp =
        VTBits-Op.getOperand(0).getValueType().getScalarType().getSizeInBits();
    return ComputeNumSignBits(Op.getOperand(0), Depth+1) + Tmp;

  case ISD::SIGN_EXTEND_INREG:
    // Max of the input and what this extends.
    Tmp =
      cast<VTSDNode>(Op.getOperand(1))->getVT().getScalarType().getSizeInBits();
    Tmp = VTBits-Tmp+1;

    Tmp2 = ComputeNumSignBits(Op.getOperand(0), Depth+1);
    return std::max(Tmp, Tmp2);

  case ISD::SRA:
    Tmp = ComputeNumSignBits(Op.getOperand(0), Depth+1);
    // SRA X, C   -> adds C sign bits.
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      Tmp += C->getZExtValue();
      if (Tmp > VTBits) Tmp = VTBits;
    }
    return Tmp;
  case ISD::SHL:
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      // shl destroys sign bits.
      Tmp = ComputeNumSignBits(Op.getOperand(0), Depth+1);
      if (C->getZExtValue() >= VTBits ||      // Bad shift.
          C->getZExtValue() >= Tmp) break;    // Shifted all sign bits out.
      return Tmp - C->getZExtValue();
    }
    break;
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:    // NOT is handled here.
    // Logical binary ops preserve the number of sign bits at the worst.
    Tmp = ComputeNumSignBits(Op.getOperand(0), Depth+1);
    if (Tmp != 1) {
      Tmp2 = ComputeNumSignBits(Op.getOperand(1), Depth+1);
      FirstAnswer = std::min(Tmp, Tmp2);
      // We computed what we know about the sign bits as our first
      // answer. Now proceed to the generic code that uses
      // computeKnownBits, and pick whichever answer is better.
    }
    break;

  case ISD::SELECT:
    Tmp = ComputeNumSignBits(Op.getOperand(1), Depth+1);
    if (Tmp == 1) return 1;  // Early out.
    Tmp2 = ComputeNumSignBits(Op.getOperand(2), Depth+1);
    return std::min(Tmp, Tmp2);
  case ISD::SMIN:
  case ISD::SMAX:
  case ISD::UMIN:
  case ISD::UMAX:
    Tmp = ComputeNumSignBits(Op.getOperand(0), Depth + 1);
    if (Tmp == 1)
      return 1;  // Early out.
    Tmp2 = ComputeNumSignBits(Op.getOperand(1), Depth + 1);
    return std::min(Tmp, Tmp2);
  case ISD::SADDO:
  case ISD::UADDO:
  case ISD::SSUBO:
  case ISD::USUBO:
  case ISD::SMULO:
  case ISD::UMULO:
    if (Op.getResNo() != 1)
      break;
    // The boolean result conforms to getBooleanContents.  Fall through.
    // If setcc returns 0/-1, all bits are sign bits.
    // We know that we have an integer-based boolean since these operations
    // are only available for integer.
    if (TLI->getBooleanContents(Op.getValueType().isVector(), false) ==
        TargetLowering::ZeroOrNegativeOneBooleanContent)
      return VTBits;
    break;
  case ISD::SETCC:
    // If setcc returns 0/-1, all bits are sign bits.
    if (TLI->getBooleanContents(Op.getOperand(0).getValueType()) ==
        TargetLowering::ZeroOrNegativeOneBooleanContent)
      return VTBits;
    break;
  case ISD::ROTL:
  case ISD::ROTR:
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      unsigned RotAmt = C->getZExtValue() & (VTBits-1);

      // Handle rotate right by N like a rotate left by 32-N.
      if (Op.getOpcode() == ISD::ROTR)
        RotAmt = (VTBits-RotAmt) & (VTBits-1);

      // If we aren't rotating out all of the known-in sign bits, return the
      // number that are left.  This handles rotl(sext(x), 1) for example.
      Tmp = ComputeNumSignBits(Op.getOperand(0), Depth+1);
      if (Tmp > RotAmt+1) return Tmp-RotAmt;
    }
    break;
  case ISD::ADD:
    // Add can have at most one carry bit.  Thus we know that the output
    // is, at worst, one more bit than the inputs.
    Tmp = ComputeNumSignBits(Op.getOperand(0), Depth+1);
    if (Tmp == 1) return 1;  // Early out.

    // Special case decrementing a value (ADD X, -1):
    if (ConstantSDNode *CRHS = dyn_cast<ConstantSDNode>(Op.getOperand(1)))
      if (CRHS->isAllOnesValue()) {
        APInt KnownZero, KnownOne;
        computeKnownBits(Op.getOperand(0), KnownZero, KnownOne, Depth+1);

        // If the input is known to be 0 or 1, the output is 0/-1, which is all
        // sign bits set.
        if ((KnownZero | APInt(VTBits, 1)).isAllOnesValue())
          return VTBits;

        // If we are subtracting one from a positive number, there is no carry
        // out of the result.
        if (KnownZero.isNegative())
          return Tmp;
      }

    Tmp2 = ComputeNumSignBits(Op.getOperand(1), Depth+1);
    if (Tmp2 == 1) return 1;
    return std::min(Tmp, Tmp2)-1;

  case ISD::SUB:
    Tmp2 = ComputeNumSignBits(Op.getOperand(1), Depth+1);
    if (Tmp2 == 1) return 1;

    // Handle NEG.
    if (ConstantSDNode *CLHS = dyn_cast<ConstantSDNode>(Op.getOperand(0)))
      if (CLHS->isNullValue()) {
        APInt KnownZero, KnownOne;
        computeKnownBits(Op.getOperand(1), KnownZero, KnownOne, Depth+1);
        // If the input is known to be 0 or 1, the output is 0/-1, which is all
        // sign bits set.
        if ((KnownZero | APInt(VTBits, 1)).isAllOnesValue())
          return VTBits;

        // If the input is known to be positive (the sign bit is known clear),
        // the output of the NEG has the same number of sign bits as the input.
        if (KnownZero.isNegative())
          return Tmp2;

        // Otherwise, we treat this like a SUB.
      }

    // Sub can have at most one carry bit.  Thus we know that the output
    // is, at worst, one more bit than the inputs.
    Tmp = ComputeNumSignBits(Op.getOperand(0), Depth+1);
    if (Tmp == 1) return 1;  // Early out.
    return std::min(Tmp, Tmp2)-1;
  case ISD::TRUNCATE:
    // FIXME: it's tricky to do anything useful for this, but it is an important
    // case for targets like X86.
    break;
  case ISD::EXTRACT_ELEMENT: {
    const int KnownSign = ComputeNumSignBits(Op.getOperand(0), Depth+1);
    const int BitWidth = Op.getValueType().getSizeInBits();
    const int Items =
      Op.getOperand(0).getValueType().getSizeInBits() / BitWidth;

    // Get reverse index (starting from 1), Op1 value indexes elements from
    // little end. Sign starts at big end.
    const int rIndex = Items - 1 -
      cast<ConstantSDNode>(Op.getOperand(1))->getZExtValue();

    // If the sign portion ends in our element the substraction gives correct
    // result. Otherwise it gives either negative or > bitwidth result
    return std::max(std::min(KnownSign - rIndex * BitWidth, BitWidth), 0);
  }
  }

  // If we are looking at the loaded value of the SDNode.
  if (Op.getResNo() == 0) {
    // Handle LOADX separately here. EXTLOAD case will fallthrough.
    if (LoadSDNode *LD = dyn_cast<LoadSDNode>(Op)) {
      unsigned ExtType = LD->getExtensionType();
      switch (ExtType) {
        default: break;
        case ISD::SEXTLOAD:    // '17' bits known
          Tmp = LD->getMemoryVT().getScalarType().getSizeInBits();
          return VTBits-Tmp+1;
        case ISD::ZEXTLOAD:    // '16' bits known
          Tmp = LD->getMemoryVT().getScalarType().getSizeInBits();
          return VTBits-Tmp;
      }
    }
  }

  // Allow the target to implement this method for its nodes.
  if (Op.getOpcode() >= ISD::BUILTIN_OP_END ||
      Op.getOpcode() == ISD::INTRINSIC_WO_CHAIN ||
      Op.getOpcode() == ISD::INTRINSIC_W_CHAIN ||
      Op.getOpcode() == ISD::INTRINSIC_VOID) {
    unsigned NumBits = TLI->ComputeNumSignBitsForTargetNode(Op, *this, Depth);
    if (NumBits > 1) FirstAnswer = std::max(FirstAnswer, NumBits);
  }

  // Finally, if we can prove that the top bits of the result are 0's or 1's,
  // use this information.
  APInt KnownZero, KnownOne;
  computeKnownBits(Op, KnownZero, KnownOne, Depth);

  APInt Mask;
  if (KnownZero.isNegative()) {        // sign bit is 0
    Mask = KnownZero;
  } else if (KnownOne.isNegative()) {  // sign bit is 1;
    Mask = KnownOne;
  } else {
    // Nothing known.
    return FirstAnswer;
  }

  // Okay, we know that the sign bit in Mask is set.  Use CLZ to determine
  // the number of identical bits in the top of the input value.
  Mask = ~Mask;
  Mask <<= Mask.getBitWidth()-VTBits;
  // Return # leading zeros.  We use 'min' here in case Val was zero before
  // shifting.  We don't want to return '64' as for an i32 "0".
  return std::max(FirstAnswer, std::min(VTBits, Mask.countLeadingZeros()));
}

/// isBaseWithConstantOffset - Return true if the specified operand is an
/// ISD::ADD with a ConstantSDNode on the right-hand side, or if it is an
/// ISD::OR with a ConstantSDNode that is guaranteed to have the same
/// semantics as an ADD.  This handles the equivalence:
///     X|Cst == X+Cst iff X&Cst = 0.
bool SelectionDAG::isBaseWithConstantOffset(SDValue Op) const {
  if ((Op.getOpcode() != ISD::ADD && Op.getOpcode() != ISD::OR) ||
      !isa<ConstantSDNode>(Op.getOperand(1)))
    return false;

  if (Op.getOpcode() == ISD::OR &&
      !MaskedValueIsZero(Op.getOperand(0),
                     cast<ConstantSDNode>(Op.getOperand(1))->getAPIntValue()))
    return false;

  return true;
}


bool SelectionDAG::isKnownNeverNaN(SDValue Op) const {
  // If we're told that NaNs won't happen, assume they won't.
  if (getTarget().Options.NoNaNsFPMath)
    return true;

  // If the value is a constant, we can obviously see if it is a NaN or not.
  if (const ConstantFPSDNode *C = dyn_cast<ConstantFPSDNode>(Op))
    return !C->getValueAPF().isNaN();

  // TODO: Recognize more cases here.

  return false;
}

bool SelectionDAG::isKnownNeverZero(SDValue Op) const {
  // If the value is a constant, we can obviously see if it is a zero or not.
  if (const ConstantFPSDNode *C = dyn_cast<ConstantFPSDNode>(Op))
    return !C->isZero();

  // TODO: Recognize more cases here.
  switch (Op.getOpcode()) {
  default: break;
  case ISD::OR:
    if (const ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op.getOperand(1)))
      return !C->isNullValue();
    break;
  }

  return false;
}

bool SelectionDAG::isEqualTo(SDValue A, SDValue B) const {
  // Check the obvious case.
  if (A == B) return true;

  // For for negative and positive zero.
  if (const ConstantFPSDNode *CA = dyn_cast<ConstantFPSDNode>(A))
    if (const ConstantFPSDNode *CB = dyn_cast<ConstantFPSDNode>(B))
      if (CA->isZero() && CB->isZero()) return true;

  // Otherwise they may not be equal.
  return false;
}

/// getNode - Gets or creates the specified node.
///
SDValue SelectionDAG::getNode(unsigned Opcode, SDLoc DL, EVT VT) {
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, Opcode, getVTList(VT), None);
  void *IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, DL.getDebugLoc(), IP))
    return SDValue(E, 0);

  SDNode *N = new (NodeAllocator) SDNode(Opcode, DL.getIROrder(),
                                         DL.getDebugLoc(), getVTList(VT));
  CSEMap.InsertNode(N, IP);

  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getNode(unsigned Opcode, SDLoc DL,
                              EVT VT, SDValue Operand) {
  // Constant fold unary operations with an integer constant operand. Even
  // opaque constant will be folded, because the folding of unary operations
  // doesn't create new constants with different values. Nevertheless, the
  // opaque flag is preserved during folding to prevent future folding with
  // other constants.
  if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Operand)) {
    const APInt &Val = C->getAPIntValue();
    switch (Opcode) {
    default: break;
    case ISD::SIGN_EXTEND:
      return getConstant(Val.sextOrTrunc(VT.getSizeInBits()), DL, VT,
                         C->isTargetOpcode(), C->isOpaque());
    case ISD::ANY_EXTEND:
    case ISD::ZERO_EXTEND:
    case ISD::TRUNCATE:
      return getConstant(Val.zextOrTrunc(VT.getSizeInBits()), DL, VT,
                         C->isTargetOpcode(), C->isOpaque());
    case ISD::UINT_TO_FP:
    case ISD::SINT_TO_FP: {
      APFloat apf(EVTToAPFloatSemantics(VT),
                  APInt::getNullValue(VT.getSizeInBits()));
      (void)apf.convertFromAPInt(Val,
                                 Opcode==ISD::SINT_TO_FP,
                                 APFloat::rmNearestTiesToEven);
      return getConstantFP(apf, DL, VT);
    }
    case ISD::BITCAST:
      if (VT == MVT::f16 && C->getValueType(0) == MVT::i16)
        return getConstantFP(APFloat(APFloat::IEEEhalf, Val), DL, VT);
      if (VT == MVT::f32 && C->getValueType(0) == MVT::i32)
        return getConstantFP(APFloat(APFloat::IEEEsingle, Val), DL, VT);
      else if (VT == MVT::f64 && C->getValueType(0) == MVT::i64)
        return getConstantFP(APFloat(APFloat::IEEEdouble, Val), DL, VT);
      break;
    case ISD::BSWAP:
      return getConstant(Val.byteSwap(), DL, VT, C->isTargetOpcode(),
                         C->isOpaque());
    case ISD::CTPOP:
      return getConstant(Val.countPopulation(), DL, VT, C->isTargetOpcode(),
                         C->isOpaque());
    case ISD::CTLZ:
    case ISD::CTLZ_ZERO_UNDEF:
      return getConstant(Val.countLeadingZeros(), DL, VT, C->isTargetOpcode(),
                         C->isOpaque());
    case ISD::CTTZ:
    case ISD::CTTZ_ZERO_UNDEF:
      return getConstant(Val.countTrailingZeros(), DL, VT, C->isTargetOpcode(),
                         C->isOpaque());
    }
  }

  // Constant fold unary operations with a floating point constant operand.
  if (ConstantFPSDNode *C = dyn_cast<ConstantFPSDNode>(Operand)) {
    APFloat V = C->getValueAPF();    // make copy
    switch (Opcode) {
    case ISD::FNEG:
      V.changeSign();
      return getConstantFP(V, DL, VT);
    case ISD::FABS:
      V.clearSign();
      return getConstantFP(V, DL, VT);
    case ISD::FCEIL: {
      APFloat::opStatus fs = V.roundToIntegral(APFloat::rmTowardPositive);
      if (fs == APFloat::opOK || fs == APFloat::opInexact)
        return getConstantFP(V, DL, VT);
      break;
    }
    case ISD::FTRUNC: {
      APFloat::opStatus fs = V.roundToIntegral(APFloat::rmTowardZero);
      if (fs == APFloat::opOK || fs == APFloat::opInexact)
        return getConstantFP(V, DL, VT);
      break;
    }
    case ISD::FFLOOR: {
      APFloat::opStatus fs = V.roundToIntegral(APFloat::rmTowardNegative);
      if (fs == APFloat::opOK || fs == APFloat::opInexact)
        return getConstantFP(V, DL, VT);
      break;
    }
    case ISD::FP_EXTEND: {
      bool ignored;
      // This can return overflow, underflow, or inexact; we don't care.
      // FIXME need to be more flexible about rounding mode.
      (void)V.convert(EVTToAPFloatSemantics(VT),
                      APFloat::rmNearestTiesToEven, &ignored);
      return getConstantFP(V, DL, VT);
    }
    case ISD::FP_TO_SINT:
    case ISD::FP_TO_UINT: {
      integerPart x[2];
      bool ignored;
      static_assert(integerPartWidth >= 64, "APFloat parts too small!");
      // FIXME need to be more flexible about rounding mode.
      APFloat::opStatus s = V.convertToInteger(x, VT.getSizeInBits(),
                            Opcode==ISD::FP_TO_SINT,
                            APFloat::rmTowardZero, &ignored);
      if (s==APFloat::opInvalidOp)     // inexact is OK, in fact usual
        break;
      APInt api(VT.getSizeInBits(), x);
      return getConstant(api, DL, VT);
    }
    case ISD::BITCAST:
      if (VT == MVT::i16 && C->getValueType(0) == MVT::f16)
        return getConstant((uint16_t)V.bitcastToAPInt().getZExtValue(), DL, VT);
      else if (VT == MVT::i32 && C->getValueType(0) == MVT::f32)
        return getConstant((uint32_t)V.bitcastToAPInt().getZExtValue(), DL, VT);
      else if (VT == MVT::i64 && C->getValueType(0) == MVT::f64)
        return getConstant(V.bitcastToAPInt().getZExtValue(), DL, VT);
      break;
    }
  }

  // Constant fold unary operations with a vector integer or float operand.
  if (BuildVectorSDNode *BV = dyn_cast<BuildVectorSDNode>(Operand)) {
    if (BV->isConstant()) {
      switch (Opcode) {
      default:
        // FIXME: Entirely reasonable to perform folding of other unary
        // operations here as the need arises.
        break;
      case ISD::FNEG:
      case ISD::FABS:
      case ISD::FCEIL:
      case ISD::FTRUNC:
      case ISD::FFLOOR:
      case ISD::FP_EXTEND:
      case ISD::FP_TO_SINT:
      case ISD::FP_TO_UINT:
      case ISD::TRUNCATE:
      case ISD::UINT_TO_FP:
      case ISD::SINT_TO_FP:
      case ISD::BSWAP:
      case ISD::CTLZ:
      case ISD::CTLZ_ZERO_UNDEF:
      case ISD::CTTZ:
      case ISD::CTTZ_ZERO_UNDEF:
      case ISD::CTPOP: {
        EVT SVT = VT.getScalarType();
        EVT InVT = BV->getValueType(0);
        EVT InSVT = InVT.getScalarType();

        // Find legal integer scalar type for constant promotion and
        // ensure that its scalar size is at least as large as source.
        EVT LegalSVT = SVT;
        if (SVT.isInteger()) {
          LegalSVT = TLI->getTypeToTransformTo(*getContext(), SVT);
          if (LegalSVT.bitsLT(SVT)) break;
        }

        // Let the above scalar folding handle the folding of each element.
        SmallVector<SDValue, 8> Ops;
        for (int i = 0, e = VT.getVectorNumElements(); i != e; ++i) {
          SDValue OpN = BV->getOperand(i);
          EVT OpVT = OpN.getValueType();

          // Build vector (integer) scalar operands may need implicit
          // truncation - do this before constant folding.
          if (OpVT.isInteger() && OpVT.bitsGT(InSVT))
            OpN = getNode(ISD::TRUNCATE, DL, InSVT, OpN);

          OpN = getNode(Opcode, DL, SVT, OpN);

          // Legalize the (integer) scalar constant if necessary.
          if (LegalSVT != SVT)
            OpN = getNode(ISD::ANY_EXTEND, DL, LegalSVT, OpN);

          if (OpN.getOpcode() != ISD::UNDEF &&
              OpN.getOpcode() != ISD::Constant &&
              OpN.getOpcode() != ISD::ConstantFP)
            break;
          Ops.push_back(OpN);
        }
        if (Ops.size() == VT.getVectorNumElements())
          return getNode(ISD::BUILD_VECTOR, DL, VT, Ops);
        break;
      }
      }
    }
  }

  unsigned OpOpcode = Operand.getNode()->getOpcode();
  switch (Opcode) {
  case ISD::TokenFactor:
  case ISD::MERGE_VALUES:
  case ISD::CONCAT_VECTORS:
    return Operand;         // Factor, merge or concat of one node?  No need.
  case ISD::FP_ROUND: llvm_unreachable("Invalid method to make FP_ROUND node");
  case ISD::FP_EXTEND:
    assert(VT.isFloatingPoint() &&
           Operand.getValueType().isFloatingPoint() && "Invalid FP cast!");
    if (Operand.getValueType() == VT) return Operand;  // noop conversion.
    assert((!VT.isVector() ||
            VT.getVectorNumElements() ==
            Operand.getValueType().getVectorNumElements()) &&
           "Vector element count mismatch!");
    if (Operand.getOpcode() == ISD::UNDEF)
      return getUNDEF(VT);
    break;
  case ISD::SIGN_EXTEND:
    assert(VT.isInteger() && Operand.getValueType().isInteger() &&
           "Invalid SIGN_EXTEND!");
    if (Operand.getValueType() == VT) return Operand;   // noop extension
    assert(Operand.getValueType().getScalarType().bitsLT(VT.getScalarType()) &&
           "Invalid sext node, dst < src!");
    assert((!VT.isVector() ||
            VT.getVectorNumElements() ==
            Operand.getValueType().getVectorNumElements()) &&
           "Vector element count mismatch!");
    if (OpOpcode == ISD::SIGN_EXTEND || OpOpcode == ISD::ZERO_EXTEND)
      return getNode(OpOpcode, DL, VT, Operand.getNode()->getOperand(0));
    else if (OpOpcode == ISD::UNDEF)
      // sext(undef) = 0, because the top bits will all be the same.
      return getConstant(0, DL, VT);
    break;
  case ISD::ZERO_EXTEND:
    assert(VT.isInteger() && Operand.getValueType().isInteger() &&
           "Invalid ZERO_EXTEND!");
    if (Operand.getValueType() == VT) return Operand;   // noop extension
    assert(Operand.getValueType().getScalarType().bitsLT(VT.getScalarType()) &&
           "Invalid zext node, dst < src!");
    assert((!VT.isVector() ||
            VT.getVectorNumElements() ==
            Operand.getValueType().getVectorNumElements()) &&
           "Vector element count mismatch!");
    if (OpOpcode == ISD::ZERO_EXTEND)   // (zext (zext x)) -> (zext x)
      return getNode(ISD::ZERO_EXTEND, DL, VT,
                     Operand.getNode()->getOperand(0));
    else if (OpOpcode == ISD::UNDEF)
      // zext(undef) = 0, because the top bits will be zero.
      return getConstant(0, DL, VT);
    break;
  case ISD::ANY_EXTEND:
    assert(VT.isInteger() && Operand.getValueType().isInteger() &&
           "Invalid ANY_EXTEND!");
    if (Operand.getValueType() == VT) return Operand;   // noop extension
    assert(Operand.getValueType().getScalarType().bitsLT(VT.getScalarType()) &&
           "Invalid anyext node, dst < src!");
    assert((!VT.isVector() ||
            VT.getVectorNumElements() ==
            Operand.getValueType().getVectorNumElements()) &&
           "Vector element count mismatch!");

    if (OpOpcode == ISD::ZERO_EXTEND || OpOpcode == ISD::SIGN_EXTEND ||
        OpOpcode == ISD::ANY_EXTEND)
      // (ext (zext x)) -> (zext x)  and  (ext (sext x)) -> (sext x)
      return getNode(OpOpcode, DL, VT, Operand.getNode()->getOperand(0));
    else if (OpOpcode == ISD::UNDEF)
      return getUNDEF(VT);

    // (ext (trunx x)) -> x
    if (OpOpcode == ISD::TRUNCATE) {
      SDValue OpOp = Operand.getNode()->getOperand(0);
      if (OpOp.getValueType() == VT)
        return OpOp;
    }
    break;
  case ISD::TRUNCATE:
    assert(VT.isInteger() && Operand.getValueType().isInteger() &&
           "Invalid TRUNCATE!");
    if (Operand.getValueType() == VT) return Operand;   // noop truncate
    assert(Operand.getValueType().getScalarType().bitsGT(VT.getScalarType()) &&
           "Invalid truncate node, src < dst!");
    assert((!VT.isVector() ||
            VT.getVectorNumElements() ==
            Operand.getValueType().getVectorNumElements()) &&
           "Vector element count mismatch!");
    if (OpOpcode == ISD::TRUNCATE)
      return getNode(ISD::TRUNCATE, DL, VT, Operand.getNode()->getOperand(0));
    if (OpOpcode == ISD::ZERO_EXTEND || OpOpcode == ISD::SIGN_EXTEND ||
        OpOpcode == ISD::ANY_EXTEND) {
      // If the source is smaller than the dest, we still need an extend.
      if (Operand.getNode()->getOperand(0).getValueType().getScalarType()
            .bitsLT(VT.getScalarType()))
        return getNode(OpOpcode, DL, VT, Operand.getNode()->getOperand(0));
      if (Operand.getNode()->getOperand(0).getValueType().bitsGT(VT))
        return getNode(ISD::TRUNCATE, DL, VT, Operand.getNode()->getOperand(0));
      return Operand.getNode()->getOperand(0);
    }
    if (OpOpcode == ISD::UNDEF)
      return getUNDEF(VT);
    break;
  case ISD::BSWAP:
    assert(VT.isInteger() && VT == Operand.getValueType() &&
           "Invalid BSWAP!");
    assert((VT.getScalarSizeInBits() % 16 == 0) &&
           "BSWAP types must be a multiple of 16 bits!");
    if (OpOpcode == ISD::UNDEF)
      return getUNDEF(VT);
    break;
  case ISD::BITCAST:
    // Basic sanity checking.
    assert(VT.getSizeInBits() == Operand.getValueType().getSizeInBits()
           && "Cannot BITCAST between types of different sizes!");
    if (VT == Operand.getValueType()) return Operand;  // noop conversion.
    if (OpOpcode == ISD::BITCAST)  // bitconv(bitconv(x)) -> bitconv(x)
      return getNode(ISD::BITCAST, DL, VT, Operand.getOperand(0));
    if (OpOpcode == ISD::UNDEF)
      return getUNDEF(VT);
    break;
  case ISD::SCALAR_TO_VECTOR:
    assert(VT.isVector() && !Operand.getValueType().isVector() &&
           (VT.getVectorElementType() == Operand.getValueType() ||
            (VT.getVectorElementType().isInteger() &&
             Operand.getValueType().isInteger() &&
             VT.getVectorElementType().bitsLE(Operand.getValueType()))) &&
           "Illegal SCALAR_TO_VECTOR node!");
    if (OpOpcode == ISD::UNDEF)
      return getUNDEF(VT);
    // scalar_to_vector(extract_vector_elt V, 0) -> V, top bits are undefined.
    if (OpOpcode == ISD::EXTRACT_VECTOR_ELT &&
        isa<ConstantSDNode>(Operand.getOperand(1)) &&
        Operand.getConstantOperandVal(1) == 0 &&
        Operand.getOperand(0).getValueType() == VT)
      return Operand.getOperand(0);
    break;
  case ISD::FNEG:
    // -(X-Y) -> (Y-X) is unsafe because when X==Y, -0.0 != +0.0
    if (getTarget().Options.UnsafeFPMath && OpOpcode == ISD::FSUB)
      return getNode(ISD::FSUB, DL, VT, Operand.getNode()->getOperand(1),
                     Operand.getNode()->getOperand(0));
    if (OpOpcode == ISD::FNEG)  // --X -> X
      return Operand.getNode()->getOperand(0);
    break;
  case ISD::FABS:
    if (OpOpcode == ISD::FNEG)  // abs(-X) -> abs(X)
      return getNode(ISD::FABS, DL, VT, Operand.getNode()->getOperand(0));
    break;
  }

  SDNode *N;
  SDVTList VTs = getVTList(VT);
  if (VT != MVT::Glue) { // Don't CSE flag producing nodes
    FoldingSetNodeID ID;
    SDValue Ops[1] = { Operand };
    AddNodeIDNode(ID, Opcode, VTs, Ops);
    void *IP = nullptr;
    if (SDNode *E = FindNodeOrInsertPos(ID, DL.getDebugLoc(), IP))
      return SDValue(E, 0);

    N = new (NodeAllocator) UnarySDNode(Opcode, DL.getIROrder(),
                                        DL.getDebugLoc(), VTs, Operand);
    CSEMap.InsertNode(N, IP);
  } else {
    N = new (NodeAllocator) UnarySDNode(Opcode, DL.getIROrder(),
                                        DL.getDebugLoc(), VTs, Operand);
  }

  InsertNode(N);
  return SDValue(N, 0);
}

static std::pair<APInt, bool> FoldValue(unsigned Opcode, const APInt &C1,
                                        const APInt &C2) {
  switch (Opcode) {
  case ISD::ADD:  return std::make_pair(C1 + C2, true);
  case ISD::SUB:  return std::make_pair(C1 - C2, true);
  case ISD::MUL:  return std::make_pair(C1 * C2, true);
  case ISD::AND:  return std::make_pair(C1 & C2, true);
  case ISD::OR:   return std::make_pair(C1 | C2, true);
  case ISD::XOR:  return std::make_pair(C1 ^ C2, true);
  case ISD::SHL:  return std::make_pair(C1 << C2, true);
  case ISD::SRL:  return std::make_pair(C1.lshr(C2), true);
  case ISD::SRA:  return std::make_pair(C1.ashr(C2), true);
  case ISD::ROTL: return std::make_pair(C1.rotl(C2), true);
  case ISD::ROTR: return std::make_pair(C1.rotr(C2), true);
  case ISD::UDIV:
    if (!C2.getBoolValue())
      break;
    return std::make_pair(C1.udiv(C2), true);
  case ISD::UREM:
    if (!C2.getBoolValue())
      break;
    return std::make_pair(C1.urem(C2), true);
  case ISD::SDIV:
    if (!C2.getBoolValue())
      break;
    return std::make_pair(C1.sdiv(C2), true);
  case ISD::SREM:
    if (!C2.getBoolValue())
      break;
    return std::make_pair(C1.srem(C2), true);
  }
  return std::make_pair(APInt(1, 0), false);
}

SDValue SelectionDAG::FoldConstantArithmetic(unsigned Opcode, SDLoc DL, EVT VT,
                                             const ConstantSDNode *Cst1,
                                             const ConstantSDNode *Cst2) {
  if (Cst1->isOpaque() || Cst2->isOpaque())
    return SDValue();

  std::pair<APInt, bool> Folded = FoldValue(Opcode, Cst1->getAPIntValue(),
                                            Cst2->getAPIntValue());
  if (!Folded.second)
    return SDValue();
  return getConstant(Folded.first, DL, VT);
}

SDValue SelectionDAG::FoldConstantArithmetic(unsigned Opcode, SDLoc DL, EVT VT,
                                             SDNode *Cst1, SDNode *Cst2) {
  // If the opcode is a target-specific ISD node, there's nothing we can
  // do here and the operand rules may not line up with the below, so
  // bail early.
  if (Opcode >= ISD::BUILTIN_OP_END)
    return SDValue();

  // Handle the case of two scalars.
  if (const ConstantSDNode *Scalar1 = dyn_cast<ConstantSDNode>(Cst1)) {
    if (const ConstantSDNode *Scalar2 = dyn_cast<ConstantSDNode>(Cst2)) {
      if (SDValue Folded =
          FoldConstantArithmetic(Opcode, DL, VT, Scalar1, Scalar2)) {
        if (!VT.isVector())
          return Folded;
        SmallVector<SDValue, 4> Outputs;
        // We may have a vector type but a scalar result. Create a splat.
        Outputs.resize(VT.getVectorNumElements(), Outputs.back());
        // Build a big vector out of the scalar elements we generated.
        return getNode(ISD::BUILD_VECTOR, SDLoc(), VT, Outputs);
      } else {
        return SDValue();
      }
    }
  }

  // For vectors extract each constant element into Inputs so we can constant
  // fold them individually.
  BuildVectorSDNode *BV1 = dyn_cast<BuildVectorSDNode>(Cst1);
  BuildVectorSDNode *BV2 = dyn_cast<BuildVectorSDNode>(Cst2);
  if (!BV1 || !BV2)
    return SDValue();

  assert(BV1->getNumOperands() == BV2->getNumOperands() && "Out of sync!");

  EVT SVT = VT.getScalarType();
  SmallVector<SDValue, 4> Outputs;
  for (unsigned I = 0, E = BV1->getNumOperands(); I != E; ++I) {
    ConstantSDNode *V1 = dyn_cast<ConstantSDNode>(BV1->getOperand(I));
    ConstantSDNode *V2 = dyn_cast<ConstantSDNode>(BV2->getOperand(I));
    if (!V1 || !V2) // Not a constant, bail.
      return SDValue();

    if (V1->isOpaque() || V2->isOpaque())
      return SDValue();

    // Avoid BUILD_VECTOR nodes that perform implicit truncation.
    // FIXME: This is valid and could be handled by truncating the APInts.
    if (V1->getValueType(0) != SVT || V2->getValueType(0) != SVT)
      return SDValue();

    // Fold one vector element.
    std::pair<APInt, bool> Folded = FoldValue(Opcode, V1->getAPIntValue(),
                                              V2->getAPIntValue());
    if (!Folded.second)
      return SDValue();
    Outputs.push_back(getConstant(Folded.first, DL, SVT));
  }

  assert(VT.getVectorNumElements() == Outputs.size() &&
         "Vector size mismatch!");

  // We may have a vector type but a scalar result. Create a splat.
  Outputs.resize(VT.getVectorNumElements(), Outputs.back());

  // Build a big vector out of the scalar elements we generated.
  return getNode(ISD::BUILD_VECTOR, SDLoc(), VT, Outputs);
}

SDValue SelectionDAG::getNode(unsigned Opcode, SDLoc DL, EVT VT, SDValue N1,
                              SDValue N2, const SDNodeFlags *Flags) {
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(N2);
  switch (Opcode) {
  default: break;
  case ISD::TokenFactor:
    assert(VT == MVT::Other && N1.getValueType() == MVT::Other &&
           N2.getValueType() == MVT::Other && "Invalid token factor!");
    // Fold trivial token factors.
    if (N1.getOpcode() == ISD::EntryToken) return N2;
    if (N2.getOpcode() == ISD::EntryToken) return N1;
    if (N1 == N2) return N1;
    break;
  case ISD::CONCAT_VECTORS:
    // Concat of UNDEFs is UNDEF.
    if (N1.getOpcode() == ISD::UNDEF &&
        N2.getOpcode() == ISD::UNDEF)
      return getUNDEF(VT);

    // A CONCAT_VECTOR with all operands BUILD_VECTOR can be simplified to
    // one big BUILD_VECTOR.
    if (N1.getOpcode() == ISD::BUILD_VECTOR &&
        N2.getOpcode() == ISD::BUILD_VECTOR) {
      SmallVector<SDValue, 16> Elts(N1.getNode()->op_begin(),
                                    N1.getNode()->op_end());
      Elts.append(N2.getNode()->op_begin(), N2.getNode()->op_end());

      // BUILD_VECTOR requires all inputs to be of the same type, find the
      // maximum type and extend them all.
      EVT SVT = VT.getScalarType();
      for (SDValue Op : Elts)
        SVT = (SVT.bitsLT(Op.getValueType()) ? Op.getValueType() : SVT);
      if (SVT.bitsGT(VT.getScalarType()))
        for (SDValue &Op : Elts)
          Op = TLI->isZExtFree(Op.getValueType(), SVT)
             ? getZExtOrTrunc(Op, DL, SVT)
             : getSExtOrTrunc(Op, DL, SVT);

      return getNode(ISD::BUILD_VECTOR, DL, VT, Elts);
    }
    break;
  case ISD::AND:
    assert(VT.isInteger() && "This operator does not apply to FP types!");
    assert(N1.getValueType() == N2.getValueType() &&
           N1.getValueType() == VT && "Binary operator types must match!");
    // (X & 0) -> 0.  This commonly occurs when legalizing i64 values, so it's
    // worth handling here.
    if (N2C && N2C->isNullValue())
      return N2;
    if (N2C && N2C->isAllOnesValue())  // X & -1 -> X
      return N1;
    break;
  case ISD::OR:
  case ISD::XOR:
  case ISD::ADD:
  case ISD::SUB:
    assert(VT.isInteger() && "This operator does not apply to FP types!");
    assert(N1.getValueType() == N2.getValueType() &&
           N1.getValueType() == VT && "Binary operator types must match!");
    // (X ^|+- 0) -> X.  This commonly occurs when legalizing i64 values, so
    // it's worth handling here.
    if (N2C && N2C->isNullValue())
      return N1;
    break;
  case ISD::UDIV:
  case ISD::UREM:
  case ISD::MULHU:
  case ISD::MULHS:
  case ISD::MUL:
  case ISD::SDIV:
  case ISD::SREM:
    assert(VT.isInteger() && "This operator does not apply to FP types!");
    assert(N1.getValueType() == N2.getValueType() &&
           N1.getValueType() == VT && "Binary operator types must match!");
    break;
  case ISD::FADD:
  case ISD::FSUB:
  case ISD::FMUL:
  case ISD::FDIV:
  case ISD::FREM:
    if (getTarget().Options.UnsafeFPMath) {
      if (Opcode == ISD::FADD) {
        // 0+x --> x
        if (ConstantFPSDNode *CFP = dyn_cast<ConstantFPSDNode>(N1))
          if (CFP->getValueAPF().isZero())
            return N2;
        // x+0 --> x
        if (ConstantFPSDNode *CFP = dyn_cast<ConstantFPSDNode>(N2))
          if (CFP->getValueAPF().isZero())
            return N1;
      } else if (Opcode == ISD::FSUB) {
        // x-0 --> x
        if (ConstantFPSDNode *CFP = dyn_cast<ConstantFPSDNode>(N2))
          if (CFP->getValueAPF().isZero())
            return N1;
      } else if (Opcode == ISD::FMUL) {
        ConstantFPSDNode *CFP = dyn_cast<ConstantFPSDNode>(N1);
        SDValue V = N2;

        // If the first operand isn't the constant, try the second
        if (!CFP) {
          CFP = dyn_cast<ConstantFPSDNode>(N2);
          V = N1;
        }

        if (CFP) {
          // 0*x --> 0
          if (CFP->isZero())
            return SDValue(CFP,0);
          // 1*x --> x
          if (CFP->isExactlyValue(1.0))
            return V;
        }
      }
    }
    assert(VT.isFloatingPoint() && "This operator only applies to FP types!");
    assert(N1.getValueType() == N2.getValueType() &&
           N1.getValueType() == VT && "Binary operator types must match!");
    break;
  case ISD::FCOPYSIGN:   // N1 and result must match.  N1/N2 need not match.
    assert(N1.getValueType() == VT &&
           N1.getValueType().isFloatingPoint() &&
           N2.getValueType().isFloatingPoint() &&
           "Invalid FCOPYSIGN!");
    break;
  case ISD::SHL:
  case ISD::SRA:
  case ISD::SRL:
  case ISD::ROTL:
  case ISD::ROTR:
    assert(VT == N1.getValueType() &&
           "Shift operators return type must be the same as their first arg");
    assert(VT.isInteger() && N2.getValueType().isInteger() &&
           "Shifts only work on integers");
    assert((!VT.isVector() || VT == N2.getValueType()) &&
           "Vector shift amounts must be in the same as their first arg");
    // Verify that the shift amount VT is bit enough to hold valid shift
    // amounts.  This catches things like trying to shift an i1024 value by an
    // i8, which is easy to fall into in generic code that uses
    // TLI.getShiftAmount().
    assert(N2.getValueType().getSizeInBits() >=
                   Log2_32_Ceil(N1.getValueType().getSizeInBits()) &&
           "Invalid use of small shift amount with oversized value!");

    // Always fold shifts of i1 values so the code generator doesn't need to
    // handle them.  Since we know the size of the shift has to be less than the
    // size of the value, the shift/rotate count is guaranteed to be zero.
    if (VT == MVT::i1)
      return N1;
    if (N2C && N2C->isNullValue())
      return N1;
    break;
  case ISD::FP_ROUND_INREG: {
    EVT EVT = cast<VTSDNode>(N2)->getVT();
    assert(VT == N1.getValueType() && "Not an inreg round!");
    assert(VT.isFloatingPoint() && EVT.isFloatingPoint() &&
           "Cannot FP_ROUND_INREG integer types");
    assert(EVT.isVector() == VT.isVector() &&
           "FP_ROUND_INREG type should be vector iff the operand "
           "type is vector!");
    assert((!EVT.isVector() ||
            EVT.getVectorNumElements() == VT.getVectorNumElements()) &&
           "Vector element counts must match in FP_ROUND_INREG");
    assert(EVT.bitsLE(VT) && "Not rounding down!");
    (void)EVT;
    if (cast<VTSDNode>(N2)->getVT() == VT) return N1;  // Not actually rounding.
    break;
  }
  case ISD::FP_ROUND:
    assert(VT.isFloatingPoint() &&
           N1.getValueType().isFloatingPoint() &&
           VT.bitsLE(N1.getValueType()) &&
           isa<ConstantSDNode>(N2) && "Invalid FP_ROUND!");
    if (N1.getValueType() == VT) return N1;  // noop conversion.
    break;
  case ISD::AssertSext:
  case ISD::AssertZext: {
    EVT EVT = cast<VTSDNode>(N2)->getVT();
    assert(VT == N1.getValueType() && "Not an inreg extend!");
    assert(VT.isInteger() && EVT.isInteger() &&
           "Cannot *_EXTEND_INREG FP types");
    assert(!EVT.isVector() &&
           "AssertSExt/AssertZExt type should be the vector element type "
           "rather than the vector type!");
    assert(EVT.bitsLE(VT) && "Not extending!");
    if (VT == EVT) return N1; // noop assertion.
    break;
  }
  case ISD::SIGN_EXTEND_INREG: {
    EVT EVT = cast<VTSDNode>(N2)->getVT();
    assert(VT == N1.getValueType() && "Not an inreg extend!");
    assert(VT.isInteger() && EVT.isInteger() &&
           "Cannot *_EXTEND_INREG FP types");
    assert(EVT.isVector() == VT.isVector() &&
           "SIGN_EXTEND_INREG type should be vector iff the operand "
           "type is vector!");
    assert((!EVT.isVector() ||
            EVT.getVectorNumElements() == VT.getVectorNumElements()) &&
           "Vector element counts must match in SIGN_EXTEND_INREG");
    assert(EVT.bitsLE(VT) && "Not extending!");
    if (EVT == VT) return N1;  // Not actually extending

    auto SignExtendInReg = [&](APInt Val) {
      unsigned FromBits = EVT.getScalarType().getSizeInBits();
      Val <<= Val.getBitWidth() - FromBits;
      Val = Val.ashr(Val.getBitWidth() - FromBits);
      return getConstant(Val, DL, VT.getScalarType());
    };

    if (N1C) {
      APInt Val = N1C->getAPIntValue();
      return SignExtendInReg(Val);
    }
    if (ISD::isBuildVectorOfConstantSDNodes(N1.getNode())) {
      SmallVector<SDValue, 8> Ops;
      for (int i = 0, e = VT.getVectorNumElements(); i != e; ++i) {
        SDValue Op = N1.getOperand(i);
        if (Op.getValueType() != VT.getScalarType()) break;
        if (Op.getOpcode() == ISD::UNDEF) {
          Ops.push_back(Op);
          continue;
        }
        if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op)) {
          APInt Val = C->getAPIntValue();
          Ops.push_back(SignExtendInReg(Val));
          continue;
        }
        break;
      }
      if (Ops.size() == VT.getVectorNumElements())
        return getNode(ISD::BUILD_VECTOR, DL, VT, Ops);
    }
    break;
  }
  case ISD::EXTRACT_VECTOR_ELT:
    // EXTRACT_VECTOR_ELT of an UNDEF is an UNDEF.
    if (N1.getOpcode() == ISD::UNDEF)
      return getUNDEF(VT);

    // EXTRACT_VECTOR_ELT of out-of-bounds element is an UNDEF
    if (N2C && N2C->getZExtValue() >= N1.getValueType().getVectorNumElements())
      return getUNDEF(VT);

    // EXTRACT_VECTOR_ELT of CONCAT_VECTORS is often formed while lowering is
    // expanding copies of large vectors from registers.
    if (N2C &&
        N1.getOpcode() == ISD::CONCAT_VECTORS &&
        N1.getNumOperands() > 0) {
      unsigned Factor =
        N1.getOperand(0).getValueType().getVectorNumElements();
      return getNode(ISD::EXTRACT_VECTOR_ELT, DL, VT,
                     N1.getOperand(N2C->getZExtValue() / Factor),
                     getConstant(N2C->getZExtValue() % Factor, DL,
                                 N2.getValueType()));
    }

    // EXTRACT_VECTOR_ELT of BUILD_VECTOR is often formed while lowering is
    // expanding large vector constants.
    if (N2C && N1.getOpcode() == ISD::BUILD_VECTOR) {
      SDValue Elt = N1.getOperand(N2C->getZExtValue());

      if (VT != Elt.getValueType())
        // If the vector element type is not legal, the BUILD_VECTOR operands
        // are promoted and implicitly truncated, and the result implicitly
        // extended. Make that explicit here.
        Elt = getAnyExtOrTrunc(Elt, DL, VT);

      return Elt;
    }

    // EXTRACT_VECTOR_ELT of INSERT_VECTOR_ELT is often formed when vector
    // operations are lowered to scalars.
    if (N1.getOpcode() == ISD::INSERT_VECTOR_ELT) {
      // If the indices are the same, return the inserted element else
      // if the indices are known different, extract the element from
      // the original vector.
      SDValue N1Op2 = N1.getOperand(2);
      ConstantSDNode *N1Op2C = dyn_cast<ConstantSDNode>(N1Op2);

      if (N1Op2C && N2C) {
        if (N1Op2C->getZExtValue() == N2C->getZExtValue()) {
          if (VT == N1.getOperand(1).getValueType())
            return N1.getOperand(1);
          else
            return getSExtOrTrunc(N1.getOperand(1), DL, VT);
        }

        return getNode(ISD::EXTRACT_VECTOR_ELT, DL, VT, N1.getOperand(0), N2);
      }
    }
    break;
  case ISD::EXTRACT_ELEMENT:
    assert(N2C && (unsigned)N2C->getZExtValue() < 2 && "Bad EXTRACT_ELEMENT!");
    assert(!N1.getValueType().isVector() && !VT.isVector() &&
           (N1.getValueType().isInteger() == VT.isInteger()) &&
           N1.getValueType() != VT &&
           "Wrong types for EXTRACT_ELEMENT!");

    // EXTRACT_ELEMENT of BUILD_PAIR is often formed while legalize is expanding
    // 64-bit integers into 32-bit parts.  Instead of building the extract of
    // the BUILD_PAIR, only to have legalize rip it apart, just do it now.
    if (N1.getOpcode() == ISD::BUILD_PAIR)
      return N1.getOperand(N2C->getZExtValue());

    // EXTRACT_ELEMENT of a constant int is also very common.
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(N1)) {
      unsigned ElementSize = VT.getSizeInBits();
      unsigned Shift = ElementSize * N2C->getZExtValue();
      APInt ShiftedVal = C->getAPIntValue().lshr(Shift);
      return getConstant(ShiftedVal.trunc(ElementSize), DL, VT);
    }
    break;
  case ISD::EXTRACT_SUBVECTOR: {
    SDValue Index = N2;
    if (VT.isSimple() && N1.getValueType().isSimple()) {
      assert(VT.isVector() && N1.getValueType().isVector() &&
             "Extract subvector VTs must be a vectors!");
      assert(VT.getVectorElementType() ==
             N1.getValueType().getVectorElementType() &&
             "Extract subvector VTs must have the same element type!");
      assert(VT.getSimpleVT() <= N1.getSimpleValueType() &&
             "Extract subvector must be from larger vector to smaller vector!");

      if (isa<ConstantSDNode>(Index)) {
        assert((VT.getVectorNumElements() +
                cast<ConstantSDNode>(Index)->getZExtValue()
                <= N1.getValueType().getVectorNumElements())
               && "Extract subvector overflow!");
      }

      // Trivial extraction.
      if (VT.getSimpleVT() == N1.getSimpleValueType())
        return N1;
    }
    break;
  }
  }

  // Perform trivial constant folding.
  if (SDValue SV =
          FoldConstantArithmetic(Opcode, DL, VT, N1.getNode(), N2.getNode()))
    return SV;

  // Canonicalize constant to RHS if commutative.
  if (N1C && !N2C && isCommutativeBinOp(Opcode)) {
    std::swap(N1C, N2C);
    std::swap(N1, N2);
  }

  // Constant fold FP operations.
  bool HasFPExceptions = TLI->hasFloatingPointExceptions();
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1);
  ConstantFPSDNode *N2CFP = dyn_cast<ConstantFPSDNode>(N2);
  if (N1CFP) {
    if (!N2CFP && isCommutativeBinOp(Opcode)) {
      // Canonicalize constant to RHS if commutative.
      std::swap(N1CFP, N2CFP);
      std::swap(N1, N2);
    } else if (N2CFP) {
      APFloat V1 = N1CFP->getValueAPF(), V2 = N2CFP->getValueAPF();
      APFloat::opStatus s;
      switch (Opcode) {
      case ISD::FADD:
        s = V1.add(V2, APFloat::rmNearestTiesToEven);
        if (!HasFPExceptions || s != APFloat::opInvalidOp)
          return getConstantFP(V1, DL, VT);
        break;
      case ISD::FSUB:
        s = V1.subtract(V2, APFloat::rmNearestTiesToEven);
        if (!HasFPExceptions || s!=APFloat::opInvalidOp)
          return getConstantFP(V1, DL, VT);
        break;
      case ISD::FMUL:
        s = V1.multiply(V2, APFloat::rmNearestTiesToEven);
        if (!HasFPExceptions || s!=APFloat::opInvalidOp)
          return getConstantFP(V1, DL, VT);
        break;
      case ISD::FDIV:
        s = V1.divide(V2, APFloat::rmNearestTiesToEven);
        if (!HasFPExceptions || (s!=APFloat::opInvalidOp &&
                                 s!=APFloat::opDivByZero)) {
          return getConstantFP(V1, DL, VT);
        }
        break;
      case ISD::FREM :
        s = V1.mod(V2, APFloat::rmNearestTiesToEven);
        if (!HasFPExceptions || (s!=APFloat::opInvalidOp &&
                                 s!=APFloat::opDivByZero)) {
          return getConstantFP(V1, DL, VT);
        }
        break;
      case ISD::FCOPYSIGN:
        V1.copySign(V2);
        return getConstantFP(V1, DL, VT);
      default: break;
      }
    }

    if (Opcode == ISD::FP_ROUND) {
      APFloat V = N1CFP->getValueAPF();    // make copy
      bool ignored;
      // This can return overflow, underflow, or inexact; we don't care.
      // FIXME need to be more flexible about rounding mode.
      (void)V.convert(EVTToAPFloatSemantics(VT),
                      APFloat::rmNearestTiesToEven, &ignored);
      return getConstantFP(V, DL, VT);
    }
  }

  // Canonicalize an UNDEF to the RHS, even over a constant.
  if (N1.getOpcode() == ISD::UNDEF) {
    if (isCommutativeBinOp(Opcode)) {
      std::swap(N1, N2);
    } else {
      switch (Opcode) {
      case ISD::FP_ROUND_INREG:
      case ISD::SIGN_EXTEND_INREG:
      case ISD::SUB:
      case ISD::FSUB:
      case ISD::FDIV:
      case ISD::FREM:
      case ISD::SRA:
        return N1;     // fold op(undef, arg2) -> undef
      case ISD::UDIV:
      case ISD::SDIV:
      case ISD::UREM:
      case ISD::SREM:
      case ISD::SRL:
      case ISD::SHL:
        if (!VT.isVector())
          return getConstant(0, DL, VT);    // fold op(undef, arg2) -> 0
        // For vectors, we can't easily build an all zero vector, just return
        // the LHS.
        return N2;
      }
    }
  }

  // Fold a bunch of operators when the RHS is undef.
  if (N2.getOpcode() == ISD::UNDEF) {
    switch (Opcode) {
    case ISD::XOR:
      if (N1.getOpcode() == ISD::UNDEF)
        // Handle undef ^ undef -> 0 special case. This is a common
        // idiom (misuse).
        return getConstant(0, DL, VT);
      // fallthrough
    case ISD::ADD:
    case ISD::ADDC:
    case ISD::ADDE:
    case ISD::SUB:
    case ISD::UDIV:
    case ISD::SDIV:
    case ISD::UREM:
    case ISD::SREM:
      return N2;       // fold op(arg1, undef) -> undef
    case ISD::FADD:
    case ISD::FSUB:
    case ISD::FMUL:
    case ISD::FDIV:
    case ISD::FREM:
      if (getTarget().Options.UnsafeFPMath)
        return N2;
      break;
    case ISD::MUL:
    case ISD::AND:
    case ISD::SRL:
    case ISD::SHL:
      if (!VT.isVector())
        return getConstant(0, DL, VT);  // fold op(arg1, undef) -> 0
      // For vectors, we can't easily build an all zero vector, just return
      // the LHS.
      return N1;
    case ISD::OR:
      if (!VT.isVector())
        return getConstant(APInt::getAllOnesValue(VT.getSizeInBits()), DL, VT);
      // For vectors, we can't easily build an all one vector, just return
      // the LHS.
      return N1;
    case ISD::SRA:
      return N1;
    }
  }

  // Memoize this node if possible.
  BinarySDNode *N;
  SDVTList VTs = getVTList(VT);
  if (VT != MVT::Glue) {
    SDValue Ops[] = {N1, N2};
    FoldingSetNodeID ID;
    AddNodeIDNode(ID, Opcode, VTs, Ops);
    AddNodeIDFlags(ID, Opcode, Flags);
    void *IP = nullptr;
    if (SDNode *E = FindNodeOrInsertPos(ID, DL.getDebugLoc(), IP))
      return SDValue(E, 0);

    N = GetBinarySDNode(Opcode, DL, VTs, N1, N2, Flags);

    CSEMap.InsertNode(N, IP);
  } else {
    N = GetBinarySDNode(Opcode, DL, VTs, N1, N2, Flags);
  }

  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getNode(unsigned Opcode, SDLoc DL, EVT VT,
                              SDValue N1, SDValue N2, SDValue N3) {
  // Perform various simplifications.
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  switch (Opcode) {
  case ISD::FMA: {
    ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1);
    ConstantFPSDNode *N2CFP = dyn_cast<ConstantFPSDNode>(N2);
    ConstantFPSDNode *N3CFP = dyn_cast<ConstantFPSDNode>(N3);
    if (N1CFP && N2CFP && N3CFP) {
      APFloat  V1 = N1CFP->getValueAPF();
      const APFloat &V2 = N2CFP->getValueAPF();
      const APFloat &V3 = N3CFP->getValueAPF();
      APFloat::opStatus s =
        V1.fusedMultiplyAdd(V2, V3, APFloat::rmNearestTiesToEven);
      if (!TLI->hasFloatingPointExceptions() || s != APFloat::opInvalidOp)
        return getConstantFP(V1, DL, VT);
    }
    break;
  }
  case ISD::CONCAT_VECTORS:
    // A CONCAT_VECTOR with all operands BUILD_VECTOR can be simplified to
    // one big BUILD_VECTOR.
    if (N1.getOpcode() == ISD::BUILD_VECTOR &&
        N2.getOpcode() == ISD::BUILD_VECTOR &&
        N3.getOpcode() == ISD::BUILD_VECTOR) {
      SmallVector<SDValue, 16> Elts(N1.getNode()->op_begin(),
                                    N1.getNode()->op_end());
      Elts.append(N2.getNode()->op_begin(), N2.getNode()->op_end());
      Elts.append(N3.getNode()->op_begin(), N3.getNode()->op_end());
      return getNode(ISD::BUILD_VECTOR, DL, VT, Elts);
    }
    break;
  case ISD::SETCC: {
    // Use FoldSetCC to simplify SETCC's.
    SDValue Simp = FoldSetCC(VT, N1, N2, cast<CondCodeSDNode>(N3)->get(), DL);
    if (Simp.getNode()) return Simp;
    break;
  }
  case ISD::SELECT:
    if (N1C) {
     if (N1C->getZExtValue())
       return N2;             // select true, X, Y -> X
     return N3;             // select false, X, Y -> Y
    }

    if (N2 == N3) return N2;   // select C, X, X -> X
    break;
  case ISD::VECTOR_SHUFFLE:
    llvm_unreachable("should use getVectorShuffle constructor!");
  case ISD::INSERT_SUBVECTOR: {
    SDValue Index = N3;
    if (VT.isSimple() && N1.getValueType().isSimple()
        && N2.getValueType().isSimple()) {
      assert(VT.isVector() && N1.getValueType().isVector() &&
             N2.getValueType().isVector() &&
             "Insert subvector VTs must be a vectors");
      assert(VT == N1.getValueType() &&
             "Dest and insert subvector source types must match!");
      assert(N2.getSimpleValueType() <= N1.getSimpleValueType() &&
             "Insert subvector must be from smaller vector to larger vector!");
      if (isa<ConstantSDNode>(Index)) {
        assert((N2.getValueType().getVectorNumElements() +
                cast<ConstantSDNode>(Index)->getZExtValue()
                <= VT.getVectorNumElements())
               && "Insert subvector overflow!");
      }

      // Trivial insertion.
      if (VT.getSimpleVT() == N2.getSimpleValueType())
        return N2;
    }
    break;
  }
  case ISD::BITCAST:
    // Fold bit_convert nodes from a type to themselves.
    if (N1.getValueType() == VT)
      return N1;
    break;
  }

  // Memoize node if it doesn't produce a flag.
  SDNode *N;
  SDVTList VTs = getVTList(VT);
  if (VT != MVT::Glue) {
    SDValue Ops[] = { N1, N2, N3 };
    FoldingSetNodeID ID;
    AddNodeIDNode(ID, Opcode, VTs, Ops);
    void *IP = nullptr;
    if (SDNode *E = FindNodeOrInsertPos(ID, DL.getDebugLoc(), IP))
      return SDValue(E, 0);

    N = new (NodeAllocator) TernarySDNode(Opcode, DL.getIROrder(),
                                          DL.getDebugLoc(), VTs, N1, N2, N3);
    CSEMap.InsertNode(N, IP);
  } else {
    N = new (NodeAllocator) TernarySDNode(Opcode, DL.getIROrder(),
                                          DL.getDebugLoc(), VTs, N1, N2, N3);
  }

  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getNode(unsigned Opcode, SDLoc DL, EVT VT,
                              SDValue N1, SDValue N2, SDValue N3,
                              SDValue N4) {
  SDValue Ops[] = { N1, N2, N3, N4 };
  return getNode(Opcode, DL, VT, Ops);
}

SDValue SelectionDAG::getNode(unsigned Opcode, SDLoc DL, EVT VT,
                              SDValue N1, SDValue N2, SDValue N3,
                              SDValue N4, SDValue N5) {
  SDValue Ops[] = { N1, N2, N3, N4, N5 };
  return getNode(Opcode, DL, VT, Ops);
}

/// getStackArgumentTokenFactor - Compute a TokenFactor to force all
/// the incoming stack arguments to be loaded from the stack.
SDValue SelectionDAG::getStackArgumentTokenFactor(SDValue Chain) {
  SmallVector<SDValue, 8> ArgChains;

  // Include the original chain at the beginning of the list. When this is
  // used by target LowerCall hooks, this helps legalize find the
  // CALLSEQ_BEGIN node.
  ArgChains.push_back(Chain);

  // Add a chain value for each stack argument.
  for (SDNode::use_iterator U = getEntryNode().getNode()->use_begin(),
       UE = getEntryNode().getNode()->use_end(); U != UE; ++U)
    if (LoadSDNode *L = dyn_cast<LoadSDNode>(*U))
      if (FrameIndexSDNode *FI = dyn_cast<FrameIndexSDNode>(L->getBasePtr()))
        if (FI->getIndex() < 0)
          ArgChains.push_back(SDValue(L, 1));

  // Build a tokenfactor for all the chains.
  return getNode(ISD::TokenFactor, SDLoc(Chain), MVT::Other, ArgChains);
}

/// getMemsetValue - Vectorized representation of the memset value
/// operand.
static SDValue getMemsetValue(SDValue Value, EVT VT, SelectionDAG &DAG,
                              SDLoc dl) {
  assert(Value.getOpcode() != ISD::UNDEF);

  unsigned NumBits = VT.getScalarType().getSizeInBits();
  if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Value)) {
    assert(C->getAPIntValue().getBitWidth() == 8);
    APInt Val = APInt::getSplat(NumBits, C->getAPIntValue());
    if (VT.isInteger())
      return DAG.getConstant(Val, dl, VT);
    return DAG.getConstantFP(APFloat(DAG.EVTToAPFloatSemantics(VT), Val), dl,
                             VT);
  }

  assert(Value.getValueType() == MVT::i8 && "memset with non-byte fill value?");
  EVT IntVT = VT.getScalarType();
  if (!IntVT.isInteger())
    IntVT = EVT::getIntegerVT(*DAG.getContext(), IntVT.getSizeInBits());

  Value = DAG.getNode(ISD::ZERO_EXTEND, dl, IntVT, Value);
  if (NumBits > 8) {
    // Use a multiplication with 0x010101... to extend the input to the
    // required length.
    APInt Magic = APInt::getSplat(NumBits, APInt(8, 0x01));
    Value = DAG.getNode(ISD::MUL, dl, IntVT, Value,
                        DAG.getConstant(Magic, dl, IntVT));
  }

  if (VT != Value.getValueType() && !VT.isInteger())
    Value = DAG.getNode(ISD::BITCAST, dl, VT.getScalarType(), Value);
  if (VT != Value.getValueType()) {
    assert(VT.getVectorElementType() == Value.getValueType() &&
           "value type should be one vector element here");
    SmallVector<SDValue, 8> BVOps(VT.getVectorNumElements(), Value);
    Value = DAG.getNode(ISD::BUILD_VECTOR, dl, VT, BVOps);
  }

  return Value;
}

/// getMemsetStringVal - Similar to getMemsetValue. Except this is only
/// used when a memcpy is turned into a memset when the source is a constant
/// string ptr.
static SDValue getMemsetStringVal(EVT VT, SDLoc dl, SelectionDAG &DAG,
                                  const TargetLowering &TLI, StringRef Str) {
  // Handle vector with all elements zero.
  if (Str.empty()) {
    if (VT.isInteger())
      return DAG.getConstant(0, dl, VT);
    else if (VT == MVT::f32 || VT == MVT::f64 || VT == MVT::f128)
      return DAG.getConstantFP(0.0, dl, VT);
    else if (VT.isVector()) {
      unsigned NumElts = VT.getVectorNumElements();
      MVT EltVT = (VT.getVectorElementType() == MVT::f32) ? MVT::i32 : MVT::i64;
      return DAG.getNode(ISD::BITCAST, dl, VT,
                         DAG.getConstant(0, dl,
                                         EVT::getVectorVT(*DAG.getContext(),
                                                          EltVT, NumElts)));
    } else
      llvm_unreachable("Expected type!");
  }

  assert(!VT.isVector() && "Can't handle vector type here!");
  unsigned NumVTBits = VT.getSizeInBits();
  unsigned NumVTBytes = NumVTBits / 8;
  unsigned NumBytes = std::min(NumVTBytes, unsigned(Str.size()));

  APInt Val(NumVTBits, 0);
  if (DAG.getDataLayout().isLittleEndian()) {
    for (unsigned i = 0; i != NumBytes; ++i)
      Val |= (uint64_t)(unsigned char)Str[i] << i*8;
  } else {
    for (unsigned i = 0; i != NumBytes; ++i)
      Val |= (uint64_t)(unsigned char)Str[i] << (NumVTBytes-i-1)*8;
  }

  // If the "cost" of materializing the integer immediate is less than the cost
  // of a load, then it is cost effective to turn the load into the immediate.
  Type *Ty = VT.getTypeForEVT(*DAG.getContext());
  if (TLI.shouldConvertConstantLoadToIntImm(Val, Ty))
    return DAG.getConstant(Val, dl, VT);
  return SDValue(nullptr, 0);
}

/// getMemBasePlusOffset - Returns base and offset node for the
///
static SDValue getMemBasePlusOffset(SDValue Base, unsigned Offset, SDLoc dl,
                                      SelectionDAG &DAG) {
  EVT VT = Base.getValueType();
  return DAG.getNode(ISD::ADD, dl,
                     VT, Base, DAG.getConstant(Offset, dl, VT));
}

/// isMemSrcFromString - Returns true if memcpy source is a string constant.
///
static bool isMemSrcFromString(SDValue Src, StringRef &Str) {
  unsigned SrcDelta = 0;
  GlobalAddressSDNode *G = nullptr;
  if (Src.getOpcode() == ISD::GlobalAddress)
    G = cast<GlobalAddressSDNode>(Src);
  else if (Src.getOpcode() == ISD::ADD &&
           Src.getOperand(0).getOpcode() == ISD::GlobalAddress &&
           Src.getOperand(1).getOpcode() == ISD::Constant) {
    G = cast<GlobalAddressSDNode>(Src.getOperand(0));
    SrcDelta = cast<ConstantSDNode>(Src.getOperand(1))->getZExtValue();
  }
  if (!G)
    return false;

  return getConstantStringInfo(G->getGlobal(), Str, SrcDelta, false);
}

/// Determines the optimal series of memory ops to replace the memset / memcpy.
/// Return true if the number of memory ops is below the threshold (Limit).
/// It returns the types of the sequence of memory ops to perform
/// memset / memcpy by reference.
static bool FindOptimalMemOpLowering(std::vector<EVT> &MemOps,
                                     unsigned Limit, uint64_t Size,
                                     unsigned DstAlign, unsigned SrcAlign,
                                     bool IsMemset,
                                     bool ZeroMemset,
                                     bool MemcpyStrSrc,
                                     bool AllowOverlap,
                                     SelectionDAG &DAG,
                                     const TargetLowering &TLI) {
  assert((SrcAlign == 0 || SrcAlign >= DstAlign) &&
         "Expecting memcpy / memset source to meet alignment requirement!");
  // If 'SrcAlign' is zero, that means the memory operation does not need to
  // load the value, i.e. memset or memcpy from constant string. Otherwise,
  // it's the inferred alignment of the source. 'DstAlign', on the other hand,
  // is the specified alignment of the memory operation. If it is zero, that
  // means it's possible to change the alignment of the destination.
  // 'MemcpyStrSrc' indicates whether the memcpy source is constant so it does
  // not need to be loaded.
  EVT VT = TLI.getOptimalMemOpType(Size, DstAlign, SrcAlign,
                                   IsMemset, ZeroMemset, MemcpyStrSrc,
                                   DAG.getMachineFunction());

  if (VT == MVT::Other) {
    unsigned AS = 0;
    if (DstAlign >= DAG.getDataLayout().getPointerPrefAlignment(AS) ||
        TLI.allowsMisalignedMemoryAccesses(VT, AS, DstAlign)) {
      VT = TLI.getPointerTy(DAG.getDataLayout());
    } else {
      switch (DstAlign & 7) {
      case 0:  VT = MVT::i64; break;
      case 4:  VT = MVT::i32; break;
      case 2:  VT = MVT::i16; break;
      default: VT = MVT::i8;  break;
      }
    }

    MVT LVT = MVT::i64;
    while (!TLI.isTypeLegal(LVT))
      LVT = (MVT::SimpleValueType)(LVT.SimpleTy - 1);
    assert(LVT.isInteger());

    if (VT.bitsGT(LVT))
      VT = LVT;
  }

  unsigned NumMemOps = 0;
  while (Size != 0) {
    unsigned VTSize = VT.getSizeInBits() / 8;
    while (VTSize > Size) {
      // For now, only use non-vector load / store's for the left-over pieces.
      EVT NewVT = VT;
      unsigned NewVTSize;

      bool Found = false;
      if (VT.isVector() || VT.isFloatingPoint()) {
        NewVT = (VT.getSizeInBits() > 64) ? MVT::i64 : MVT::i32;
        if (TLI.isOperationLegalOrCustom(ISD::STORE, NewVT) &&
            TLI.isSafeMemOpType(NewVT.getSimpleVT()))
          Found = true;
        else if (NewVT == MVT::i64 &&
                 TLI.isOperationLegalOrCustom(ISD::STORE, MVT::f64) &&
                 TLI.isSafeMemOpType(MVT::f64)) {
          // i64 is usually not legal on 32-bit targets, but f64 may be.
          NewVT = MVT::f64;
          Found = true;
        }
      }

      if (!Found) {
        do {
          NewVT = (MVT::SimpleValueType)(NewVT.getSimpleVT().SimpleTy - 1);
          if (NewVT == MVT::i8)
            break;
        } while (!TLI.isSafeMemOpType(NewVT.getSimpleVT()));
      }
      NewVTSize = NewVT.getSizeInBits() / 8;

      // If the new VT cannot cover all of the remaining bits, then consider
      // issuing a (or a pair of) unaligned and overlapping load / store.
      // FIXME: Only does this for 64-bit or more since we don't have proper
      // cost model for unaligned load / store.
      bool Fast;
      unsigned AS = 0;
      if (NumMemOps && AllowOverlap &&
          VTSize >= 8 && NewVTSize < Size &&
          TLI.allowsMisalignedMemoryAccesses(VT, AS, DstAlign, &Fast) && Fast)
        VTSize = Size;
      else {
        VT = NewVT;
        VTSize = NewVTSize;
      }
    }

    if (++NumMemOps > Limit)
      return false;

    MemOps.push_back(VT);
    Size -= VTSize;
  }

  return true;
}

static SDValue getMemcpyLoadsAndStores(SelectionDAG &DAG, SDLoc dl,
                                       SDValue Chain, SDValue Dst,
                                       SDValue Src, uint64_t Size,
                                       unsigned Align, bool isVol,
                                       bool AlwaysInline,
                                       MachinePointerInfo DstPtrInfo,
                                       MachinePointerInfo SrcPtrInfo) {
  // Turn a memcpy of undef to nop.
  if (Src.getOpcode() == ISD::UNDEF)
    return Chain;

  // Expand memcpy to a series of load and store ops if the size operand falls
  // below a certain threshold.
  // TODO: In the AlwaysInline case, if the size is big then generate a loop
  // rather than maybe a humongous number of loads and stores.
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  std::vector<EVT> MemOps;
  bool DstAlignCanChange = false;
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  bool OptSize = MF.getFunction()->hasFnAttribute(Attribute::OptimizeForSize);
  FrameIndexSDNode *FI = dyn_cast<FrameIndexSDNode>(Dst);
  if (FI && !MFI->isFixedObjectIndex(FI->getIndex()))
    DstAlignCanChange = true;
  unsigned SrcAlign = DAG.InferPtrAlignment(Src);
  if (Align > SrcAlign)
    SrcAlign = Align;
  StringRef Str;
  bool CopyFromStr = isMemSrcFromString(Src, Str);
  bool isZeroStr = CopyFromStr && Str.empty();
  unsigned Limit = AlwaysInline ? ~0U : TLI.getMaxStoresPerMemcpy(OptSize);

  if (!FindOptimalMemOpLowering(MemOps, Limit, Size,
                                (DstAlignCanChange ? 0 : Align),
                                (isZeroStr ? 0 : SrcAlign),
                                false, false, CopyFromStr, true, DAG, TLI))
    return SDValue();

  if (DstAlignCanChange) {
    Type *Ty = MemOps[0].getTypeForEVT(*DAG.getContext());
    unsigned NewAlign = (unsigned)DAG.getDataLayout().getABITypeAlignment(Ty);

    // Don't promote to an alignment that would require dynamic stack
    // realignment.
    const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
    if (!TRI->needsStackRealignment(MF))
      while (NewAlign > Align &&
             DAG.getDataLayout().exceedsNaturalStackAlignment(NewAlign))
          NewAlign /= 2;

    if (NewAlign > Align) {
      // Give the stack frame object a larger alignment if needed.
      if (MFI->getObjectAlignment(FI->getIndex()) < NewAlign)
        MFI->setObjectAlignment(FI->getIndex(), NewAlign);
      Align = NewAlign;
    }
  }

  SmallVector<SDValue, 8> OutChains;
  unsigned NumMemOps = MemOps.size();
  uint64_t SrcOff = 0, DstOff = 0;
  for (unsigned i = 0; i != NumMemOps; ++i) {
    EVT VT = MemOps[i];
    unsigned VTSize = VT.getSizeInBits() / 8;
    SDValue Value, Store;

    if (VTSize > Size) {
      // Issuing an unaligned load / store pair  that overlaps with the previous
      // pair. Adjust the offset accordingly.
      assert(i == NumMemOps-1 && i != 0);
      SrcOff -= VTSize - Size;
      DstOff -= VTSize - Size;
    }

    if (CopyFromStr &&
        (isZeroStr || (VT.isInteger() && !VT.isVector()))) {
      // It's unlikely a store of a vector immediate can be done in a single
      // instruction. It would require a load from a constantpool first.
      // We only handle zero vectors here.
      // FIXME: Handle other cases where store of vector immediate is done in
      // a single instruction.
      Value = getMemsetStringVal(VT, dl, DAG, TLI, Str.substr(SrcOff));
      if (Value.getNode())
        Store = DAG.getStore(Chain, dl, Value,
                             getMemBasePlusOffset(Dst, DstOff, dl, DAG),
                             DstPtrInfo.getWithOffset(DstOff), isVol,
                             false, Align);
    }

    if (!Store.getNode()) {
      // The type might not be legal for the target.  This should only happen
      // if the type is smaller than a legal type, as on PPC, so the right
      // thing to do is generate a LoadExt/StoreTrunc pair.  These simplify
      // to Load/Store if NVT==VT.
      // FIXME does the case above also need this?
      EVT NVT = TLI.getTypeToTransformTo(*DAG.getContext(), VT);
      assert(NVT.bitsGE(VT));
      Value = DAG.getExtLoad(ISD::EXTLOAD, dl, NVT, Chain,
                             getMemBasePlusOffset(Src, SrcOff, dl, DAG),
                             SrcPtrInfo.getWithOffset(SrcOff), VT, isVol, false,
                             false, MinAlign(SrcAlign, SrcOff));
      Store = DAG.getTruncStore(Chain, dl, Value,
                                getMemBasePlusOffset(Dst, DstOff, dl, DAG),
                                DstPtrInfo.getWithOffset(DstOff), VT, isVol,
                                false, Align);
    }
    OutChains.push_back(Store);
    SrcOff += VTSize;
    DstOff += VTSize;
    Size -= VTSize;
  }

  return DAG.getNode(ISD::TokenFactor, dl, MVT::Other, OutChains);
}

static SDValue getMemmoveLoadsAndStores(SelectionDAG &DAG, SDLoc dl,
                                        SDValue Chain, SDValue Dst,
                                        SDValue Src, uint64_t Size,
                                        unsigned Align,  bool isVol,
                                        bool AlwaysInline,
                                        MachinePointerInfo DstPtrInfo,
                                        MachinePointerInfo SrcPtrInfo) {
  // Turn a memmove of undef to nop.
  if (Src.getOpcode() == ISD::UNDEF)
    return Chain;

  // Expand memmove to a series of load and store ops if the size operand falls
  // below a certain threshold.
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  std::vector<EVT> MemOps;
  bool DstAlignCanChange = false;
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  bool OptSize = MF.getFunction()->hasFnAttribute(Attribute::OptimizeForSize);
  FrameIndexSDNode *FI = dyn_cast<FrameIndexSDNode>(Dst);
  if (FI && !MFI->isFixedObjectIndex(FI->getIndex()))
    DstAlignCanChange = true;
  unsigned SrcAlign = DAG.InferPtrAlignment(Src);
  if (Align > SrcAlign)
    SrcAlign = Align;
  unsigned Limit = AlwaysInline ? ~0U : TLI.getMaxStoresPerMemmove(OptSize);

  if (!FindOptimalMemOpLowering(MemOps, Limit, Size,
                                (DstAlignCanChange ? 0 : Align), SrcAlign,
                                false, false, false, false, DAG, TLI))
    return SDValue();

  if (DstAlignCanChange) {
    Type *Ty = MemOps[0].getTypeForEVT(*DAG.getContext());
    unsigned NewAlign = (unsigned)DAG.getDataLayout().getABITypeAlignment(Ty);
    if (NewAlign > Align) {
      // Give the stack frame object a larger alignment if needed.
      if (MFI->getObjectAlignment(FI->getIndex()) < NewAlign)
        MFI->setObjectAlignment(FI->getIndex(), NewAlign);
      Align = NewAlign;
    }
  }

  uint64_t SrcOff = 0, DstOff = 0;
  SmallVector<SDValue, 8> LoadValues;
  SmallVector<SDValue, 8> LoadChains;
  SmallVector<SDValue, 8> OutChains;
  unsigned NumMemOps = MemOps.size();
  for (unsigned i = 0; i < NumMemOps; i++) {
    EVT VT = MemOps[i];
    unsigned VTSize = VT.getSizeInBits() / 8;
    SDValue Value;

    Value = DAG.getLoad(VT, dl, Chain,
                        getMemBasePlusOffset(Src, SrcOff, dl, DAG),
                        SrcPtrInfo.getWithOffset(SrcOff), isVol,
                        false, false, SrcAlign);
    LoadValues.push_back(Value);
    LoadChains.push_back(Value.getValue(1));
    SrcOff += VTSize;
  }
  Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, LoadChains);
  OutChains.clear();
  for (unsigned i = 0; i < NumMemOps; i++) {
    EVT VT = MemOps[i];
    unsigned VTSize = VT.getSizeInBits() / 8;
    SDValue Store;

    Store = DAG.getStore(Chain, dl, LoadValues[i],
                         getMemBasePlusOffset(Dst, DstOff, dl, DAG),
                         DstPtrInfo.getWithOffset(DstOff), isVol, false, Align);
    OutChains.push_back(Store);
    DstOff += VTSize;
  }

  return DAG.getNode(ISD::TokenFactor, dl, MVT::Other, OutChains);
}

/// \brief Lower the call to 'memset' intrinsic function into a series of store
/// operations.
///
/// \param DAG Selection DAG where lowered code is placed.
/// \param dl Link to corresponding IR location.
/// \param Chain Control flow dependency.
/// \param Dst Pointer to destination memory location.
/// \param Src Value of byte to write into the memory.
/// \param Size Number of bytes to write.
/// \param Align Alignment of the destination in bytes.
/// \param isVol True if destination is volatile.
/// \param DstPtrInfo IR information on the memory pointer.
/// \returns New head in the control flow, if lowering was successful, empty
/// SDValue otherwise.
///
/// The function tries to replace 'llvm.memset' intrinsic with several store
/// operations and value calculation code. This is usually profitable for small
/// memory size.
static SDValue getMemsetStores(SelectionDAG &DAG, SDLoc dl,
                               SDValue Chain, SDValue Dst,
                               SDValue Src, uint64_t Size,
                               unsigned Align, bool isVol,
                               MachinePointerInfo DstPtrInfo) {
  // Turn a memset of undef to nop.
  if (Src.getOpcode() == ISD::UNDEF)
    return Chain;

  // Expand memset to a series of load/store ops if the size operand
  // falls below a certain threshold.
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  std::vector<EVT> MemOps;
  bool DstAlignCanChange = false;
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  bool OptSize = MF.getFunction()->hasFnAttribute(Attribute::OptimizeForSize);
  FrameIndexSDNode *FI = dyn_cast<FrameIndexSDNode>(Dst);
  if (FI && !MFI->isFixedObjectIndex(FI->getIndex()))
    DstAlignCanChange = true;
  bool IsZeroVal =
    isa<ConstantSDNode>(Src) && cast<ConstantSDNode>(Src)->isNullValue();
  if (!FindOptimalMemOpLowering(MemOps, TLI.getMaxStoresPerMemset(OptSize),
                                Size, (DstAlignCanChange ? 0 : Align), 0,
                                true, IsZeroVal, false, true, DAG, TLI))
    return SDValue();

  if (DstAlignCanChange) {
    Type *Ty = MemOps[0].getTypeForEVT(*DAG.getContext());
    unsigned NewAlign = (unsigned)DAG.getDataLayout().getABITypeAlignment(Ty);
    if (NewAlign > Align) {
      // Give the stack frame object a larger alignment if needed.
      if (MFI->getObjectAlignment(FI->getIndex()) < NewAlign)
        MFI->setObjectAlignment(FI->getIndex(), NewAlign);
      Align = NewAlign;
    }
  }

  SmallVector<SDValue, 8> OutChains;
  uint64_t DstOff = 0;
  unsigned NumMemOps = MemOps.size();

  // Find the largest store and generate the bit pattern for it.
  EVT LargestVT = MemOps[0];
  for (unsigned i = 1; i < NumMemOps; i++)
    if (MemOps[i].bitsGT(LargestVT))
      LargestVT = MemOps[i];
  SDValue MemSetValue = getMemsetValue(Src, LargestVT, DAG, dl);

  for (unsigned i = 0; i < NumMemOps; i++) {
    EVT VT = MemOps[i];
    unsigned VTSize = VT.getSizeInBits() / 8;
    if (VTSize > Size) {
      // Issuing an unaligned load / store pair  that overlaps with the previous
      // pair. Adjust the offset accordingly.
      assert(i == NumMemOps-1 && i != 0);
      DstOff -= VTSize - Size;
    }

    // If this store is smaller than the largest store see whether we can get
    // the smaller value for free with a truncate.
    SDValue Value = MemSetValue;
    if (VT.bitsLT(LargestVT)) {
      if (!LargestVT.isVector() && !VT.isVector() &&
          TLI.isTruncateFree(LargestVT, VT))
        Value = DAG.getNode(ISD::TRUNCATE, dl, VT, MemSetValue);
      else
        Value = getMemsetValue(Src, VT, DAG, dl);
    }
    assert(Value.getValueType() == VT && "Value with wrong type.");
    SDValue Store = DAG.getStore(Chain, dl, Value,
                                 getMemBasePlusOffset(Dst, DstOff, dl, DAG),
                                 DstPtrInfo.getWithOffset(DstOff),
                                 isVol, false, Align);
    OutChains.push_back(Store);
    DstOff += VT.getSizeInBits() / 8;
    Size -= VTSize;
  }

  return DAG.getNode(ISD::TokenFactor, dl, MVT::Other, OutChains);
}

SDValue SelectionDAG::getMemcpy(SDValue Chain, SDLoc dl, SDValue Dst,
                                SDValue Src, SDValue Size,
                                unsigned Align, bool isVol, bool AlwaysInline,
                                bool isTailCall, MachinePointerInfo DstPtrInfo,
                                MachinePointerInfo SrcPtrInfo) {
  assert(Align && "The SDAG layer expects explicit alignment and reserves 0");

  // Check to see if we should lower the memcpy to loads and stores first.
  // For cases within the target-specified limits, this is the best choice.
  ConstantSDNode *ConstantSize = dyn_cast<ConstantSDNode>(Size);
  if (ConstantSize) {
    // Memcpy with size zero? Just return the original chain.
    if (ConstantSize->isNullValue())
      return Chain;

    SDValue Result = getMemcpyLoadsAndStores(*this, dl, Chain, Dst, Src,
                                             ConstantSize->getZExtValue(),Align,
                                isVol, false, DstPtrInfo, SrcPtrInfo);
    if (Result.getNode())
      return Result;
  }

  // Then check to see if we should lower the memcpy with target-specific
  // code. If the target chooses to do this, this is the next best.
  if (TSI) {
    SDValue Result = TSI->EmitTargetCodeForMemcpy(
        *this, dl, Chain, Dst, Src, Size, Align, isVol, AlwaysInline,
        DstPtrInfo, SrcPtrInfo);
    if (Result.getNode())
      return Result;
  }

  // If we really need inline code and the target declined to provide it,
  // use a (potentially long) sequence of loads and stores.
  if (AlwaysInline) {
    assert(ConstantSize && "AlwaysInline requires a constant size!");
    return getMemcpyLoadsAndStores(*this, dl, Chain, Dst, Src,
                                   ConstantSize->getZExtValue(), Align, isVol,
                                   true, DstPtrInfo, SrcPtrInfo);
  }

  // FIXME: If the memcpy is volatile (isVol), lowering it to a plain libc
  // memcpy is not guaranteed to be safe. libc memcpys aren't required to
  // respect volatile, so they may do things like read or write memory
  // beyond the given memory regions. But fixing this isn't easy, and most
  // people don't care.

  // Emit a library call.
  TargetLowering::ArgListTy Args;
  TargetLowering::ArgListEntry Entry;
  Entry.Ty = getDataLayout().getIntPtrType(*getContext());
  Entry.Node = Dst; Args.push_back(Entry);
  Entry.Node = Src; Args.push_back(Entry);
  Entry.Node = Size; Args.push_back(Entry);
  // FIXME: pass in SDLoc
  TargetLowering::CallLoweringInfo CLI(*this);
  CLI.setDebugLoc(dl)
      .setChain(Chain)
      .setCallee(TLI->getLibcallCallingConv(RTLIB::MEMCPY),
                 Type::getVoidTy(*getContext()),
                 getExternalSymbol(TLI->getLibcallName(RTLIB::MEMCPY),
                                   TLI->getPointerTy(getDataLayout())),
                 std::move(Args), 0)
      .setDiscardResult()
      .setTailCall(isTailCall);

  std::pair<SDValue,SDValue> CallResult = TLI->LowerCallTo(CLI);
  return CallResult.second;
}

SDValue SelectionDAG::getMemmove(SDValue Chain, SDLoc dl, SDValue Dst,
                                 SDValue Src, SDValue Size,
                                 unsigned Align, bool isVol, bool isTailCall,
                                 MachinePointerInfo DstPtrInfo,
                                 MachinePointerInfo SrcPtrInfo) {
  assert(Align && "The SDAG layer expects explicit alignment and reserves 0");

  // Check to see if we should lower the memmove to loads and stores first.
  // For cases within the target-specified limits, this is the best choice.
  ConstantSDNode *ConstantSize = dyn_cast<ConstantSDNode>(Size);
  if (ConstantSize) {
    // Memmove with size zero? Just return the original chain.
    if (ConstantSize->isNullValue())
      return Chain;

    SDValue Result =
      getMemmoveLoadsAndStores(*this, dl, Chain, Dst, Src,
                               ConstantSize->getZExtValue(), Align, isVol,
                               false, DstPtrInfo, SrcPtrInfo);
    if (Result.getNode())
      return Result;
  }

  // Then check to see if we should lower the memmove with target-specific
  // code. If the target chooses to do this, this is the next best.
  if (TSI) {
    SDValue Result = TSI->EmitTargetCodeForMemmove(
        *this, dl, Chain, Dst, Src, Size, Align, isVol, DstPtrInfo, SrcPtrInfo);
    if (Result.getNode())
      return Result;
  }

  // FIXME: If the memmove is volatile, lowering it to plain libc memmove may
  // not be safe.  See memcpy above for more details.

  // Emit a library call.
  TargetLowering::ArgListTy Args;
  TargetLowering::ArgListEntry Entry;
  Entry.Ty = getDataLayout().getIntPtrType(*getContext());
  Entry.Node = Dst; Args.push_back(Entry);
  Entry.Node = Src; Args.push_back(Entry);
  Entry.Node = Size; Args.push_back(Entry);
  // FIXME:  pass in SDLoc
  TargetLowering::CallLoweringInfo CLI(*this);
  CLI.setDebugLoc(dl)
      .setChain(Chain)
      .setCallee(TLI->getLibcallCallingConv(RTLIB::MEMMOVE),
                 Type::getVoidTy(*getContext()),
                 getExternalSymbol(TLI->getLibcallName(RTLIB::MEMMOVE),
                                   TLI->getPointerTy(getDataLayout())),
                 std::move(Args), 0)
      .setDiscardResult()
      .setTailCall(isTailCall);

  std::pair<SDValue,SDValue> CallResult = TLI->LowerCallTo(CLI);
  return CallResult.second;
}

SDValue SelectionDAG::getMemset(SDValue Chain, SDLoc dl, SDValue Dst,
                                SDValue Src, SDValue Size,
                                unsigned Align, bool isVol, bool isTailCall,
                                MachinePointerInfo DstPtrInfo) {
  assert(Align && "The SDAG layer expects explicit alignment and reserves 0");

  // Check to see if we should lower the memset to stores first.
  // For cases within the target-specified limits, this is the best choice.
  ConstantSDNode *ConstantSize = dyn_cast<ConstantSDNode>(Size);
  if (ConstantSize) {
    // Memset with size zero? Just return the original chain.
    if (ConstantSize->isNullValue())
      return Chain;

    SDValue Result =
      getMemsetStores(*this, dl, Chain, Dst, Src, ConstantSize->getZExtValue(),
                      Align, isVol, DstPtrInfo);

    if (Result.getNode())
      return Result;
  }

  // Then check to see if we should lower the memset with target-specific
  // code. If the target chooses to do this, this is the next best.
  if (TSI) {
    SDValue Result = TSI->EmitTargetCodeForMemset(
        *this, dl, Chain, Dst, Src, Size, Align, isVol, DstPtrInfo);
    if (Result.getNode())
      return Result;
  }

  // Emit a library call.
  Type *IntPtrTy = getDataLayout().getIntPtrType(*getContext());
  TargetLowering::ArgListTy Args;
  TargetLowering::ArgListEntry Entry;
  Entry.Node = Dst; Entry.Ty = IntPtrTy;
  Args.push_back(Entry);
  Entry.Node = Src;
  Entry.Ty = Src.getValueType().getTypeForEVT(*getContext());
  Args.push_back(Entry);
  Entry.Node = Size;
  Entry.Ty = IntPtrTy;
  Args.push_back(Entry);

  // FIXME: pass in SDLoc
  TargetLowering::CallLoweringInfo CLI(*this);
  CLI.setDebugLoc(dl)
      .setChain(Chain)
      .setCallee(TLI->getLibcallCallingConv(RTLIB::MEMSET),
                 Type::getVoidTy(*getContext()),
                 getExternalSymbol(TLI->getLibcallName(RTLIB::MEMSET),
                                   TLI->getPointerTy(getDataLayout())),
                 std::move(Args), 0)
      .setDiscardResult()
      .setTailCall(isTailCall);

  std::pair<SDValue,SDValue> CallResult = TLI->LowerCallTo(CLI);
  return CallResult.second;
}

SDValue SelectionDAG::getAtomic(unsigned Opcode, SDLoc dl, EVT MemVT,
                                SDVTList VTList, ArrayRef<SDValue> Ops,
                                MachineMemOperand *MMO,
                                AtomicOrdering SuccessOrdering,
                                AtomicOrdering FailureOrdering,
                                SynchronizationScope SynchScope) {
  FoldingSetNodeID ID;
  ID.AddInteger(MemVT.getRawBits());
  AddNodeIDNode(ID, Opcode, VTList, Ops);
  ID.AddInteger(MMO->getPointerInfo().getAddrSpace());
  void* IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, dl.getDebugLoc(), IP)) {
    cast<AtomicSDNode>(E)->refineAlignment(MMO);
    return SDValue(E, 0);
  }

  // Allocate the operands array for the node out of the BumpPtrAllocator, since
  // SDNode doesn't have access to it.  This memory will be "leaked" when
  // the node is deallocated, but recovered when the allocator is released.
  // If the number of operands is less than 5 we use AtomicSDNode's internal
  // storage.
  unsigned NumOps = Ops.size();
  SDUse *DynOps = NumOps > 4 ? OperandAllocator.Allocate<SDUse>(NumOps)
                             : nullptr;

  SDNode *N = new (NodeAllocator) AtomicSDNode(Opcode, dl.getIROrder(),
                                               dl.getDebugLoc(), VTList, MemVT,
                                               Ops.data(), DynOps, NumOps, MMO,
                                               SuccessOrdering, FailureOrdering,
                                               SynchScope);
  CSEMap.InsertNode(N, IP);
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getAtomic(unsigned Opcode, SDLoc dl, EVT MemVT,
                                SDVTList VTList, ArrayRef<SDValue> Ops,
                                MachineMemOperand *MMO,
                                AtomicOrdering Ordering,
                                SynchronizationScope SynchScope) {
  return getAtomic(Opcode, dl, MemVT, VTList, Ops, MMO, Ordering,
                   Ordering, SynchScope);
}

SDValue SelectionDAG::getAtomicCmpSwap(
    unsigned Opcode, SDLoc dl, EVT MemVT, SDVTList VTs, SDValue Chain,
    SDValue Ptr, SDValue Cmp, SDValue Swp, MachinePointerInfo PtrInfo,
    unsigned Alignment, AtomicOrdering SuccessOrdering,
    AtomicOrdering FailureOrdering, SynchronizationScope SynchScope) {
  assert(Opcode == ISD::ATOMIC_CMP_SWAP ||
         Opcode == ISD::ATOMIC_CMP_SWAP_WITH_SUCCESS);
  assert(Cmp.getValueType() == Swp.getValueType() && "Invalid Atomic Op Types");

  if (Alignment == 0)  // Ensure that codegen never sees alignment 0
    Alignment = getEVTAlignment(MemVT);

  MachineFunction &MF = getMachineFunction();

  // FIXME: Volatile isn't really correct; we should keep track of atomic
  // orderings in the memoperand.
  unsigned Flags = MachineMemOperand::MOVolatile;
  Flags |= MachineMemOperand::MOLoad;
  Flags |= MachineMemOperand::MOStore;

  MachineMemOperand *MMO =
    MF.getMachineMemOperand(PtrInfo, Flags, MemVT.getStoreSize(), Alignment);

  return getAtomicCmpSwap(Opcode, dl, MemVT, VTs, Chain, Ptr, Cmp, Swp, MMO,
                          SuccessOrdering, FailureOrdering, SynchScope);
}

SDValue SelectionDAG::getAtomicCmpSwap(unsigned Opcode, SDLoc dl, EVT MemVT,
                                       SDVTList VTs, SDValue Chain, SDValue Ptr,
                                       SDValue Cmp, SDValue Swp,
                                       MachineMemOperand *MMO,
                                       AtomicOrdering SuccessOrdering,
                                       AtomicOrdering FailureOrdering,
                                       SynchronizationScope SynchScope) {
  assert(Opcode == ISD::ATOMIC_CMP_SWAP ||
         Opcode == ISD::ATOMIC_CMP_SWAP_WITH_SUCCESS);
  assert(Cmp.getValueType() == Swp.getValueType() && "Invalid Atomic Op Types");

  SDValue Ops[] = {Chain, Ptr, Cmp, Swp};
  return getAtomic(Opcode, dl, MemVT, VTs, Ops, MMO,
                   SuccessOrdering, FailureOrdering, SynchScope);
}

SDValue SelectionDAG::getAtomic(unsigned Opcode, SDLoc dl, EVT MemVT,
                                SDValue Chain,
                                SDValue Ptr, SDValue Val,
                                const Value* PtrVal,
                                unsigned Alignment,
                                AtomicOrdering Ordering,
                                SynchronizationScope SynchScope) {
  if (Alignment == 0)  // Ensure that codegen never sees alignment 0
    Alignment = getEVTAlignment(MemVT);

  MachineFunction &MF = getMachineFunction();
  // An atomic store does not load. An atomic load does not store.
  // (An atomicrmw obviously both loads and stores.)
  // For now, atomics are considered to be volatile always, and they are
  // chained as such.
  // FIXME: Volatile isn't really correct; we should keep track of atomic
  // orderings in the memoperand.
  unsigned Flags = MachineMemOperand::MOVolatile;
  if (Opcode != ISD::ATOMIC_STORE)
    Flags |= MachineMemOperand::MOLoad;
  if (Opcode != ISD::ATOMIC_LOAD)
    Flags |= MachineMemOperand::MOStore;

  MachineMemOperand *MMO =
    MF.getMachineMemOperand(MachinePointerInfo(PtrVal), Flags,
                            MemVT.getStoreSize(), Alignment);

  return getAtomic(Opcode, dl, MemVT, Chain, Ptr, Val, MMO,
                   Ordering, SynchScope);
}

SDValue SelectionDAG::getAtomic(unsigned Opcode, SDLoc dl, EVT MemVT,
                                SDValue Chain,
                                SDValue Ptr, SDValue Val,
                                MachineMemOperand *MMO,
                                AtomicOrdering Ordering,
                                SynchronizationScope SynchScope) {
  assert((Opcode == ISD::ATOMIC_LOAD_ADD ||
          Opcode == ISD::ATOMIC_LOAD_SUB ||
          Opcode == ISD::ATOMIC_LOAD_AND ||
          Opcode == ISD::ATOMIC_LOAD_OR ||
          Opcode == ISD::ATOMIC_LOAD_XOR ||
          Opcode == ISD::ATOMIC_LOAD_NAND ||
          Opcode == ISD::ATOMIC_LOAD_MIN ||
          Opcode == ISD::ATOMIC_LOAD_MAX ||
          Opcode == ISD::ATOMIC_LOAD_UMIN ||
          Opcode == ISD::ATOMIC_LOAD_UMAX ||
          Opcode == ISD::ATOMIC_SWAP ||
          Opcode == ISD::ATOMIC_STORE) &&
         "Invalid Atomic Op");

  EVT VT = Val.getValueType();

  SDVTList VTs = Opcode == ISD::ATOMIC_STORE ? getVTList(MVT::Other) :
                                               getVTList(VT, MVT::Other);
  SDValue Ops[] = {Chain, Ptr, Val};
  return getAtomic(Opcode, dl, MemVT, VTs, Ops, MMO, Ordering, SynchScope);
}

SDValue SelectionDAG::getAtomic(unsigned Opcode, SDLoc dl, EVT MemVT,
                                EVT VT, SDValue Chain,
                                SDValue Ptr,
                                MachineMemOperand *MMO,
                                AtomicOrdering Ordering,
                                SynchronizationScope SynchScope) {
  assert(Opcode == ISD::ATOMIC_LOAD && "Invalid Atomic Op");

  SDVTList VTs = getVTList(VT, MVT::Other);
  SDValue Ops[] = {Chain, Ptr};
  return getAtomic(Opcode, dl, MemVT, VTs, Ops, MMO, Ordering, SynchScope);
}

/// getMergeValues - Create a MERGE_VALUES node from the given operands.
SDValue SelectionDAG::getMergeValues(ArrayRef<SDValue> Ops, SDLoc dl) {
  if (Ops.size() == 1)
    return Ops[0];

  SmallVector<EVT, 4> VTs;
  VTs.reserve(Ops.size());
  for (unsigned i = 0; i < Ops.size(); ++i)
    VTs.push_back(Ops[i].getValueType());
  return getNode(ISD::MERGE_VALUES, dl, getVTList(VTs), Ops);
}

SDValue
SelectionDAG::getMemIntrinsicNode(unsigned Opcode, SDLoc dl, SDVTList VTList,
                                  ArrayRef<SDValue> Ops,
                                  EVT MemVT, MachinePointerInfo PtrInfo,
                                  unsigned Align, bool Vol,
                                  bool ReadMem, bool WriteMem, unsigned Size) {
  if (Align == 0)  // Ensure that codegen never sees alignment 0
    Align = getEVTAlignment(MemVT);

  MachineFunction &MF = getMachineFunction();
  unsigned Flags = 0;
  if (WriteMem)
    Flags |= MachineMemOperand::MOStore;
  if (ReadMem)
    Flags |= MachineMemOperand::MOLoad;
  if (Vol)
    Flags |= MachineMemOperand::MOVolatile;
  if (!Size)
    Size = MemVT.getStoreSize();
  MachineMemOperand *MMO =
    MF.getMachineMemOperand(PtrInfo, Flags, Size, Align);

  return getMemIntrinsicNode(Opcode, dl, VTList, Ops, MemVT, MMO);
}

SDValue
SelectionDAG::getMemIntrinsicNode(unsigned Opcode, SDLoc dl, SDVTList VTList,
                                  ArrayRef<SDValue> Ops, EVT MemVT,
                                  MachineMemOperand *MMO) {
  assert((Opcode == ISD::INTRINSIC_VOID ||
          Opcode == ISD::INTRINSIC_W_CHAIN ||
          Opcode == ISD::PREFETCH ||
          Opcode == ISD::LIFETIME_START ||
          Opcode == ISD::LIFETIME_END ||
          (Opcode <= INT_MAX &&
           (int)Opcode >= ISD::FIRST_TARGET_MEMORY_OPCODE)) &&
         "Opcode is not a memory-accessing opcode!");

  // Memoize the node unless it returns a flag.
  MemIntrinsicSDNode *N;
  if (VTList.VTs[VTList.NumVTs-1] != MVT::Glue) {
    FoldingSetNodeID ID;
    AddNodeIDNode(ID, Opcode, VTList, Ops);
    ID.AddInteger(MMO->getPointerInfo().getAddrSpace());
    void *IP = nullptr;
    if (SDNode *E = FindNodeOrInsertPos(ID, dl.getDebugLoc(), IP)) {
      cast<MemIntrinsicSDNode>(E)->refineAlignment(MMO);
      return SDValue(E, 0);
    }

    N = new (NodeAllocator) MemIntrinsicSDNode(Opcode, dl.getIROrder(),
                                               dl.getDebugLoc(), VTList, Ops,
                                               MemVT, MMO);
    CSEMap.InsertNode(N, IP);
  } else {
    N = new (NodeAllocator) MemIntrinsicSDNode(Opcode, dl.getIROrder(),
                                               dl.getDebugLoc(), VTList, Ops,
                                               MemVT, MMO);
  }
  InsertNode(N);
  return SDValue(N, 0);
}

/// InferPointerInfo - If the specified ptr/offset is a frame index, infer a
/// MachinePointerInfo record from it.  This is particularly useful because the
/// code generator has many cases where it doesn't bother passing in a
/// MachinePointerInfo to getLoad or getStore when it has "FI+Cst".
static MachinePointerInfo InferPointerInfo(SDValue Ptr, int64_t Offset = 0) {
  // If this is FI+Offset, we can model it.
  if (const FrameIndexSDNode *FI = dyn_cast<FrameIndexSDNode>(Ptr))
    return MachinePointerInfo::getFixedStack(FI->getIndex(), Offset);

  // If this is (FI+Offset1)+Offset2, we can model it.
  if (Ptr.getOpcode() != ISD::ADD ||
      !isa<ConstantSDNode>(Ptr.getOperand(1)) ||
      !isa<FrameIndexSDNode>(Ptr.getOperand(0)))
    return MachinePointerInfo();

  int FI = cast<FrameIndexSDNode>(Ptr.getOperand(0))->getIndex();
  return MachinePointerInfo::getFixedStack(FI, Offset+
                       cast<ConstantSDNode>(Ptr.getOperand(1))->getSExtValue());
}

/// InferPointerInfo - If the specified ptr/offset is a frame index, infer a
/// MachinePointerInfo record from it.  This is particularly useful because the
/// code generator has many cases where it doesn't bother passing in a
/// MachinePointerInfo to getLoad or getStore when it has "FI+Cst".
static MachinePointerInfo InferPointerInfo(SDValue Ptr, SDValue OffsetOp) {
  // If the 'Offset' value isn't a constant, we can't handle this.
  if (ConstantSDNode *OffsetNode = dyn_cast<ConstantSDNode>(OffsetOp))
    return InferPointerInfo(Ptr, OffsetNode->getSExtValue());
  if (OffsetOp.getOpcode() == ISD::UNDEF)
    return InferPointerInfo(Ptr);
  return MachinePointerInfo();
}


SDValue
SelectionDAG::getLoad(ISD::MemIndexedMode AM, ISD::LoadExtType ExtType,
                      EVT VT, SDLoc dl, SDValue Chain,
                      SDValue Ptr, SDValue Offset,
                      MachinePointerInfo PtrInfo, EVT MemVT,
                      bool isVolatile, bool isNonTemporal, bool isInvariant,
                      unsigned Alignment, const AAMDNodes &AAInfo,
                      const MDNode *Ranges) {
  assert(Chain.getValueType() == MVT::Other &&
        "Invalid chain type");
  if (Alignment == 0)  // Ensure that codegen never sees alignment 0
    Alignment = getEVTAlignment(VT);

  unsigned Flags = MachineMemOperand::MOLoad;
  if (isVolatile)
    Flags |= MachineMemOperand::MOVolatile;
  if (isNonTemporal)
    Flags |= MachineMemOperand::MONonTemporal;
  if (isInvariant)
    Flags |= MachineMemOperand::MOInvariant;

  // If we don't have a PtrInfo, infer the trivial frame index case to simplify
  // clients.
  if (PtrInfo.V.isNull())
    PtrInfo = InferPointerInfo(Ptr, Offset);

  MachineFunction &MF = getMachineFunction();
  MachineMemOperand *MMO =
    MF.getMachineMemOperand(PtrInfo, Flags, MemVT.getStoreSize(), Alignment,
                            AAInfo, Ranges);
  return getLoad(AM, ExtType, VT, dl, Chain, Ptr, Offset, MemVT, MMO);
}

SDValue
SelectionDAG::getLoad(ISD::MemIndexedMode AM, ISD::LoadExtType ExtType,
                      EVT VT, SDLoc dl, SDValue Chain,
                      SDValue Ptr, SDValue Offset, EVT MemVT,
                      MachineMemOperand *MMO) {
  if (VT == MemVT) {
    ExtType = ISD::NON_EXTLOAD;
  } else if (ExtType == ISD::NON_EXTLOAD) {
    assert(VT == MemVT && "Non-extending load from different memory type!");
  } else {
    // Extending load.
    assert(MemVT.getScalarType().bitsLT(VT.getScalarType()) &&
           "Should only be an extending load, not truncating!");
    assert(VT.isInteger() == MemVT.isInteger() &&
           "Cannot convert from FP to Int or Int -> FP!");
    assert(VT.isVector() == MemVT.isVector() &&
           "Cannot use an ext load to convert to or from a vector!");
    assert((!VT.isVector() ||
            VT.getVectorNumElements() == MemVT.getVectorNumElements()) &&
           "Cannot use an ext load to change the number of vector elements!");
  }

  bool Indexed = AM != ISD::UNINDEXED;
  assert((Indexed || Offset.getOpcode() == ISD::UNDEF) &&
         "Unindexed load with an offset!");

  SDVTList VTs = Indexed ?
    getVTList(VT, Ptr.getValueType(), MVT::Other) : getVTList(VT, MVT::Other);
  SDValue Ops[] = { Chain, Ptr, Offset };
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::LOAD, VTs, Ops);
  ID.AddInteger(MemVT.getRawBits());
  ID.AddInteger(encodeMemSDNodeFlags(ExtType, AM, MMO->isVolatile(),
                                     MMO->isNonTemporal(),
                                     MMO->isInvariant()));
  ID.AddInteger(MMO->getPointerInfo().getAddrSpace());
  void *IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, dl.getDebugLoc(), IP)) {
    cast<LoadSDNode>(E)->refineAlignment(MMO);
    return SDValue(E, 0);
  }
  SDNode *N = new (NodeAllocator) LoadSDNode(Ops, dl.getIROrder(),
                                             dl.getDebugLoc(), VTs, AM, ExtType,
                                             MemVT, MMO);
  CSEMap.InsertNode(N, IP);
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getLoad(EVT VT, SDLoc dl,
                              SDValue Chain, SDValue Ptr,
                              MachinePointerInfo PtrInfo,
                              bool isVolatile, bool isNonTemporal,
                              bool isInvariant, unsigned Alignment,
                              const AAMDNodes &AAInfo,
                              const MDNode *Ranges) {
  SDValue Undef = getUNDEF(Ptr.getValueType());
  return getLoad(ISD::UNINDEXED, ISD::NON_EXTLOAD, VT, dl, Chain, Ptr, Undef,
                 PtrInfo, VT, isVolatile, isNonTemporal, isInvariant, Alignment,
                 AAInfo, Ranges);
}

SDValue SelectionDAG::getLoad(EVT VT, SDLoc dl,
                              SDValue Chain, SDValue Ptr,
                              MachineMemOperand *MMO) {
  SDValue Undef = getUNDEF(Ptr.getValueType());
  return getLoad(ISD::UNINDEXED, ISD::NON_EXTLOAD, VT, dl, Chain, Ptr, Undef,
                 VT, MMO);
}

SDValue SelectionDAG::getExtLoad(ISD::LoadExtType ExtType, SDLoc dl, EVT VT,
                                 SDValue Chain, SDValue Ptr,
                                 MachinePointerInfo PtrInfo, EVT MemVT,
                                 bool isVolatile, bool isNonTemporal,
                                 bool isInvariant, unsigned Alignment,
                                 const AAMDNodes &AAInfo) {
  SDValue Undef = getUNDEF(Ptr.getValueType());
  return getLoad(ISD::UNINDEXED, ExtType, VT, dl, Chain, Ptr, Undef,
                 PtrInfo, MemVT, isVolatile, isNonTemporal, isInvariant,
                 Alignment, AAInfo);
}


SDValue SelectionDAG::getExtLoad(ISD::LoadExtType ExtType, SDLoc dl, EVT VT,
                                 SDValue Chain, SDValue Ptr, EVT MemVT,
                                 MachineMemOperand *MMO) {
  SDValue Undef = getUNDEF(Ptr.getValueType());
  return getLoad(ISD::UNINDEXED, ExtType, VT, dl, Chain, Ptr, Undef,
                 MemVT, MMO);
}

SDValue
SelectionDAG::getIndexedLoad(SDValue OrigLoad, SDLoc dl, SDValue Base,
                             SDValue Offset, ISD::MemIndexedMode AM) {
  LoadSDNode *LD = cast<LoadSDNode>(OrigLoad);
  assert(LD->getOffset().getOpcode() == ISD::UNDEF &&
         "Load is already a indexed load!");
  return getLoad(AM, LD->getExtensionType(), OrigLoad.getValueType(), dl,
                 LD->getChain(), Base, Offset, LD->getPointerInfo(),
                 LD->getMemoryVT(), LD->isVolatile(), LD->isNonTemporal(),
                 false, LD->getAlignment());
}

SDValue SelectionDAG::getStore(SDValue Chain, SDLoc dl, SDValue Val,
                               SDValue Ptr, MachinePointerInfo PtrInfo,
                               bool isVolatile, bool isNonTemporal,
                               unsigned Alignment, const AAMDNodes &AAInfo) {
  assert(Chain.getValueType() == MVT::Other &&
        "Invalid chain type");
  if (Alignment == 0)  // Ensure that codegen never sees alignment 0
    Alignment = getEVTAlignment(Val.getValueType());

  unsigned Flags = MachineMemOperand::MOStore;
  if (isVolatile)
    Flags |= MachineMemOperand::MOVolatile;
  if (isNonTemporal)
    Flags |= MachineMemOperand::MONonTemporal;

  if (PtrInfo.V.isNull())
    PtrInfo = InferPointerInfo(Ptr);

  MachineFunction &MF = getMachineFunction();
  MachineMemOperand *MMO =
    MF.getMachineMemOperand(PtrInfo, Flags,
                            Val.getValueType().getStoreSize(), Alignment,
                            AAInfo);

  return getStore(Chain, dl, Val, Ptr, MMO);
}

SDValue SelectionDAG::getStore(SDValue Chain, SDLoc dl, SDValue Val,
                               SDValue Ptr, MachineMemOperand *MMO) {
  assert(Chain.getValueType() == MVT::Other &&
        "Invalid chain type");
  EVT VT = Val.getValueType();
  SDVTList VTs = getVTList(MVT::Other);
  SDValue Undef = getUNDEF(Ptr.getValueType());
  SDValue Ops[] = { Chain, Val, Ptr, Undef };
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::STORE, VTs, Ops);
  ID.AddInteger(VT.getRawBits());
  ID.AddInteger(encodeMemSDNodeFlags(false, ISD::UNINDEXED, MMO->isVolatile(),
                                     MMO->isNonTemporal(), MMO->isInvariant()));
  ID.AddInteger(MMO->getPointerInfo().getAddrSpace());
  void *IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, dl.getDebugLoc(), IP)) {
    cast<StoreSDNode>(E)->refineAlignment(MMO);
    return SDValue(E, 0);
  }
  SDNode *N = new (NodeAllocator) StoreSDNode(Ops, dl.getIROrder(),
                                              dl.getDebugLoc(), VTs,
                                              ISD::UNINDEXED, false, VT, MMO);
  CSEMap.InsertNode(N, IP);
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getTruncStore(SDValue Chain, SDLoc dl, SDValue Val,
                                    SDValue Ptr, MachinePointerInfo PtrInfo,
                                    EVT SVT,bool isVolatile, bool isNonTemporal,
                                    unsigned Alignment,
                                    const AAMDNodes &AAInfo) {
  assert(Chain.getValueType() == MVT::Other &&
        "Invalid chain type");
  if (Alignment == 0)  // Ensure that codegen never sees alignment 0
    Alignment = getEVTAlignment(SVT);

  unsigned Flags = MachineMemOperand::MOStore;
  if (isVolatile)
    Flags |= MachineMemOperand::MOVolatile;
  if (isNonTemporal)
    Flags |= MachineMemOperand::MONonTemporal;

  if (PtrInfo.V.isNull())
    PtrInfo = InferPointerInfo(Ptr);

  MachineFunction &MF = getMachineFunction();
  MachineMemOperand *MMO =
    MF.getMachineMemOperand(PtrInfo, Flags, SVT.getStoreSize(), Alignment,
                            AAInfo);

  return getTruncStore(Chain, dl, Val, Ptr, SVT, MMO);
}

SDValue SelectionDAG::getTruncStore(SDValue Chain, SDLoc dl, SDValue Val,
                                    SDValue Ptr, EVT SVT,
                                    MachineMemOperand *MMO) {
  EVT VT = Val.getValueType();

  assert(Chain.getValueType() == MVT::Other &&
        "Invalid chain type");
  if (VT == SVT)
    return getStore(Chain, dl, Val, Ptr, MMO);

  assert(SVT.getScalarType().bitsLT(VT.getScalarType()) &&
         "Should only be a truncating store, not extending!");
  assert(VT.isInteger() == SVT.isInteger() &&
         "Can't do FP-INT conversion!");
  assert(VT.isVector() == SVT.isVector() &&
         "Cannot use trunc store to convert to or from a vector!");
  assert((!VT.isVector() ||
          VT.getVectorNumElements() == SVT.getVectorNumElements()) &&
         "Cannot use trunc store to change the number of vector elements!");

  SDVTList VTs = getVTList(MVT::Other);
  SDValue Undef = getUNDEF(Ptr.getValueType());
  SDValue Ops[] = { Chain, Val, Ptr, Undef };
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::STORE, VTs, Ops);
  ID.AddInteger(SVT.getRawBits());
  ID.AddInteger(encodeMemSDNodeFlags(true, ISD::UNINDEXED, MMO->isVolatile(),
                                     MMO->isNonTemporal(), MMO->isInvariant()));
  ID.AddInteger(MMO->getPointerInfo().getAddrSpace());
  void *IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, dl.getDebugLoc(), IP)) {
    cast<StoreSDNode>(E)->refineAlignment(MMO);
    return SDValue(E, 0);
  }
  SDNode *N = new (NodeAllocator) StoreSDNode(Ops, dl.getIROrder(),
                                              dl.getDebugLoc(), VTs,
                                              ISD::UNINDEXED, true, SVT, MMO);
  CSEMap.InsertNode(N, IP);
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue
SelectionDAG::getIndexedStore(SDValue OrigStore, SDLoc dl, SDValue Base,
                              SDValue Offset, ISD::MemIndexedMode AM) {
  StoreSDNode *ST = cast<StoreSDNode>(OrigStore);
  assert(ST->getOffset().getOpcode() == ISD::UNDEF &&
         "Store is already a indexed store!");
  SDVTList VTs = getVTList(Base.getValueType(), MVT::Other);
  SDValue Ops[] = { ST->getChain(), ST->getValue(), Base, Offset };
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::STORE, VTs, Ops);
  ID.AddInteger(ST->getMemoryVT().getRawBits());
  ID.AddInteger(ST->getRawSubclassData());
  ID.AddInteger(ST->getPointerInfo().getAddrSpace());
  void *IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, dl.getDebugLoc(), IP))
    return SDValue(E, 0);

  SDNode *N = new (NodeAllocator) StoreSDNode(Ops, dl.getIROrder(),
                                              dl.getDebugLoc(), VTs, AM,
                                              ST->isTruncatingStore(),
                                              ST->getMemoryVT(),
                                              ST->getMemOperand());
  CSEMap.InsertNode(N, IP);
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue
SelectionDAG::getMaskedLoad(EVT VT, SDLoc dl, SDValue Chain,
                            SDValue Ptr, SDValue Mask, SDValue Src0, EVT MemVT,
                            MachineMemOperand *MMO, ISD::LoadExtType ExtTy) {

  SDVTList VTs = getVTList(VT, MVT::Other);
  SDValue Ops[] = { Chain, Ptr, Mask, Src0 };
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::MLOAD, VTs, Ops);
  ID.AddInteger(VT.getRawBits());
  ID.AddInteger(encodeMemSDNodeFlags(ExtTy, ISD::UNINDEXED,
                                     MMO->isVolatile(),
                                     MMO->isNonTemporal(),
                                     MMO->isInvariant()));
  ID.AddInteger(MMO->getPointerInfo().getAddrSpace());
  void *IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, dl.getDebugLoc(), IP)) {
    cast<MaskedLoadSDNode>(E)->refineAlignment(MMO);
    return SDValue(E, 0);
  }
  SDNode *N = new (NodeAllocator) MaskedLoadSDNode(dl.getIROrder(),
                                             dl.getDebugLoc(), Ops, 4, VTs,
                                             ExtTy, MemVT, MMO);
  CSEMap.InsertNode(N, IP);
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getMaskedStore(SDValue Chain, SDLoc dl, SDValue Val,
                                     SDValue Ptr, SDValue Mask, EVT MemVT,
                                     MachineMemOperand *MMO, bool isTrunc) {
  assert(Chain.getValueType() == MVT::Other &&
        "Invalid chain type");
  EVT VT = Val.getValueType();
  SDVTList VTs = getVTList(MVT::Other);
  SDValue Ops[] = { Chain, Ptr, Mask, Val };
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::MSTORE, VTs, Ops);
  ID.AddInteger(VT.getRawBits());
  ID.AddInteger(encodeMemSDNodeFlags(false, ISD::UNINDEXED, MMO->isVolatile(),
                                     MMO->isNonTemporal(), MMO->isInvariant()));
  ID.AddInteger(MMO->getPointerInfo().getAddrSpace());
  void *IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, dl.getDebugLoc(), IP)) {
    cast<MaskedStoreSDNode>(E)->refineAlignment(MMO);
    return SDValue(E, 0);
  }
  SDNode *N = new (NodeAllocator) MaskedStoreSDNode(dl.getIROrder(),
                                                    dl.getDebugLoc(), Ops, 4,
                                                    VTs, isTrunc, MemVT, MMO);
  CSEMap.InsertNode(N, IP);
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue
SelectionDAG::getMaskedGather(SDVTList VTs, EVT VT, SDLoc dl,
                              ArrayRef<SDValue> Ops,
                              MachineMemOperand *MMO) {

  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::MGATHER, VTs, Ops);
  ID.AddInteger(VT.getRawBits());
  ID.AddInteger(encodeMemSDNodeFlags(ISD::NON_EXTLOAD, ISD::UNINDEXED,
                                     MMO->isVolatile(),
                                     MMO->isNonTemporal(),
                                     MMO->isInvariant()));
  ID.AddInteger(MMO->getPointerInfo().getAddrSpace());
  void *IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, dl.getDebugLoc(), IP)) {
    cast<MaskedGatherSDNode>(E)->refineAlignment(MMO);
    return SDValue(E, 0);
  }
  MaskedGatherSDNode *N = 
    new (NodeAllocator) MaskedGatherSDNode(dl.getIROrder(), dl.getDebugLoc(),
                                           Ops, VTs, VT, MMO);
  CSEMap.InsertNode(N, IP);
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getMaskedScatter(SDVTList VTs, EVT VT, SDLoc dl,
                                       ArrayRef<SDValue> Ops,
                                       MachineMemOperand *MMO) {
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::MSCATTER, VTs, Ops);
  ID.AddInteger(VT.getRawBits());
  ID.AddInteger(encodeMemSDNodeFlags(false, ISD::UNINDEXED, MMO->isVolatile(),
                                     MMO->isNonTemporal(),
                                     MMO->isInvariant()));
  ID.AddInteger(MMO->getPointerInfo().getAddrSpace());
  void *IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, dl.getDebugLoc(), IP)) {
    cast<MaskedScatterSDNode>(E)->refineAlignment(MMO);
    return SDValue(E, 0);
  }
  SDNode *N =
    new (NodeAllocator) MaskedScatterSDNode(dl.getIROrder(), dl.getDebugLoc(),
                                            Ops, VTs, VT, MMO);
  CSEMap.InsertNode(N, IP);
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getVAArg(EVT VT, SDLoc dl,
                               SDValue Chain, SDValue Ptr,
                               SDValue SV,
                               unsigned Align) {
  SDValue Ops[] = { Chain, Ptr, SV, getTargetConstant(Align, dl, MVT::i32) };
  return getNode(ISD::VAARG, dl, getVTList(VT, MVT::Other), Ops);
}

SDValue SelectionDAG::getNode(unsigned Opcode, SDLoc DL, EVT VT,
                              ArrayRef<SDUse> Ops) {
  switch (Ops.size()) {
  case 0: return getNode(Opcode, DL, VT);
  case 1: return getNode(Opcode, DL, VT, static_cast<const SDValue>(Ops[0]));
  case 2: return getNode(Opcode, DL, VT, Ops[0], Ops[1]);
  case 3: return getNode(Opcode, DL, VT, Ops[0], Ops[1], Ops[2]);
  default: break;
  }

  // Copy from an SDUse array into an SDValue array for use with
  // the regular getNode logic.
  SmallVector<SDValue, 8> NewOps(Ops.begin(), Ops.end());
  return getNode(Opcode, DL, VT, NewOps);
}

SDValue SelectionDAG::getNode(unsigned Opcode, SDLoc DL, EVT VT,
                              ArrayRef<SDValue> Ops) {
  unsigned NumOps = Ops.size();
  switch (NumOps) {
  case 0: return getNode(Opcode, DL, VT);
  case 1: return getNode(Opcode, DL, VT, Ops[0]);
  case 2: return getNode(Opcode, DL, VT, Ops[0], Ops[1]);
  case 3: return getNode(Opcode, DL, VT, Ops[0], Ops[1], Ops[2]);
  default: break;
  }

  switch (Opcode) {
  default: break;
  case ISD::SELECT_CC: {
    assert(NumOps == 5 && "SELECT_CC takes 5 operands!");
    assert(Ops[0].getValueType() == Ops[1].getValueType() &&
           "LHS and RHS of condition must have same type!");
    assert(Ops[2].getValueType() == Ops[3].getValueType() &&
           "True and False arms of SelectCC must have same type!");
    assert(Ops[2].getValueType() == VT &&
           "select_cc node must be of same type as true and false value!");
    break;
  }
  case ISD::BR_CC: {
    assert(NumOps == 5 && "BR_CC takes 5 operands!");
    assert(Ops[2].getValueType() == Ops[3].getValueType() &&
           "LHS/RHS of comparison should match types!");
    break;
  }
  }

  // Memoize nodes.
  SDNode *N;
  SDVTList VTs = getVTList(VT);

  if (VT != MVT::Glue) {
    FoldingSetNodeID ID;
    AddNodeIDNode(ID, Opcode, VTs, Ops);
    void *IP = nullptr;

    if (SDNode *E = FindNodeOrInsertPos(ID, DL.getDebugLoc(), IP))
      return SDValue(E, 0);

    N = new (NodeAllocator) SDNode(Opcode, DL.getIROrder(), DL.getDebugLoc(),
                                   VTs, Ops);
    CSEMap.InsertNode(N, IP);
  } else {
    N = new (NodeAllocator) SDNode(Opcode, DL.getIROrder(), DL.getDebugLoc(),
                                   VTs, Ops);
  }

  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getNode(unsigned Opcode, SDLoc DL,
                              ArrayRef<EVT> ResultTys, ArrayRef<SDValue> Ops) {
  return getNode(Opcode, DL, getVTList(ResultTys), Ops);
}

SDValue SelectionDAG::getNode(unsigned Opcode, SDLoc DL, SDVTList VTList,
                              ArrayRef<SDValue> Ops) {
  if (VTList.NumVTs == 1)
    return getNode(Opcode, DL, VTList.VTs[0], Ops);

#if 0
  switch (Opcode) {
  // FIXME: figure out how to safely handle things like
  // int foo(int x) { return 1 << (x & 255); }
  // int bar() { return foo(256); }
  case ISD::SRA_PARTS:
  case ISD::SRL_PARTS:
  case ISD::SHL_PARTS:
    if (N3.getOpcode() == ISD::SIGN_EXTEND_INREG &&
        cast<VTSDNode>(N3.getOperand(1))->getVT() != MVT::i1)
      return getNode(Opcode, DL, VT, N1, N2, N3.getOperand(0));
    else if (N3.getOpcode() == ISD::AND)
      if (ConstantSDNode *AndRHS = dyn_cast<ConstantSDNode>(N3.getOperand(1))) {
        // If the and is only masking out bits that cannot effect the shift,
        // eliminate the and.
        unsigned NumBits = VT.getScalarType().getSizeInBits()*2;
        if ((AndRHS->getValue() & (NumBits-1)) == NumBits-1)
          return getNode(Opcode, DL, VT, N1, N2, N3.getOperand(0));
      }
    break;
  }
#endif

  // Memoize the node unless it returns a flag.
  SDNode *N;
  unsigned NumOps = Ops.size();
  if (VTList.VTs[VTList.NumVTs-1] != MVT::Glue) {
    FoldingSetNodeID ID;
    AddNodeIDNode(ID, Opcode, VTList, Ops);
    void *IP = nullptr;
    if (SDNode *E = FindNodeOrInsertPos(ID, DL.getDebugLoc(), IP))
      return SDValue(E, 0);

    if (NumOps == 1) {
      N = new (NodeAllocator) UnarySDNode(Opcode, DL.getIROrder(),
                                          DL.getDebugLoc(), VTList, Ops[0]);
    } else if (NumOps == 2) {
      N = new (NodeAllocator) BinarySDNode(Opcode, DL.getIROrder(),
                                           DL.getDebugLoc(), VTList, Ops[0],
                                           Ops[1]);
    } else if (NumOps == 3) {
      N = new (NodeAllocator) TernarySDNode(Opcode, DL.getIROrder(),
                                            DL.getDebugLoc(), VTList, Ops[0],
                                            Ops[1], Ops[2]);
    } else {
      N = new (NodeAllocator) SDNode(Opcode, DL.getIROrder(), DL.getDebugLoc(),
                                     VTList, Ops);
    }
    CSEMap.InsertNode(N, IP);
  } else {
    if (NumOps == 1) {
      N = new (NodeAllocator) UnarySDNode(Opcode, DL.getIROrder(),
                                          DL.getDebugLoc(), VTList, Ops[0]);
    } else if (NumOps == 2) {
      N = new (NodeAllocator) BinarySDNode(Opcode, DL.getIROrder(),
                                           DL.getDebugLoc(), VTList, Ops[0],
                                           Ops[1]);
    } else if (NumOps == 3) {
      N = new (NodeAllocator) TernarySDNode(Opcode, DL.getIROrder(),
                                            DL.getDebugLoc(), VTList, Ops[0],
                                            Ops[1], Ops[2]);
    } else {
      N = new (NodeAllocator) SDNode(Opcode, DL.getIROrder(), DL.getDebugLoc(),
                                     VTList, Ops);
    }
  }
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getNode(unsigned Opcode, SDLoc DL, SDVTList VTList) {
  return getNode(Opcode, DL, VTList, None);
}

SDValue SelectionDAG::getNode(unsigned Opcode, SDLoc DL, SDVTList VTList,
                              SDValue N1) {
  SDValue Ops[] = { N1 };
  return getNode(Opcode, DL, VTList, Ops);
}

SDValue SelectionDAG::getNode(unsigned Opcode, SDLoc DL, SDVTList VTList,
                              SDValue N1, SDValue N2) {
  SDValue Ops[] = { N1, N2 };
  return getNode(Opcode, DL, VTList, Ops);
}

SDValue SelectionDAG::getNode(unsigned Opcode, SDLoc DL, SDVTList VTList,
                              SDValue N1, SDValue N2, SDValue N3) {
  SDValue Ops[] = { N1, N2, N3 };
  return getNode(Opcode, DL, VTList, Ops);
}

SDValue SelectionDAG::getNode(unsigned Opcode, SDLoc DL, SDVTList VTList,
                              SDValue N1, SDValue N2, SDValue N3,
                              SDValue N4) {
  SDValue Ops[] = { N1, N2, N3, N4 };
  return getNode(Opcode, DL, VTList, Ops);
}

SDValue SelectionDAG::getNode(unsigned Opcode, SDLoc DL, SDVTList VTList,
                              SDValue N1, SDValue N2, SDValue N3,
                              SDValue N4, SDValue N5) {
  SDValue Ops[] = { N1, N2, N3, N4, N5 };
  return getNode(Opcode, DL, VTList, Ops);
}

SDVTList SelectionDAG::getVTList(EVT VT) {
  return makeVTList(SDNode::getValueTypeList(VT), 1);
}

SDVTList SelectionDAG::getVTList(EVT VT1, EVT VT2) {
  FoldingSetNodeID ID;
  ID.AddInteger(2U);
  ID.AddInteger(VT1.getRawBits());
  ID.AddInteger(VT2.getRawBits());

  void *IP = nullptr;
  SDVTListNode *Result = VTListMap.FindNodeOrInsertPos(ID, IP);
  if (!Result) {
    EVT *Array = Allocator.Allocate<EVT>(2);
    Array[0] = VT1;
    Array[1] = VT2;
    Result = new (Allocator) SDVTListNode(ID.Intern(Allocator), Array, 2);
    VTListMap.InsertNode(Result, IP);
  }
  return Result->getSDVTList();
}

SDVTList SelectionDAG::getVTList(EVT VT1, EVT VT2, EVT VT3) {
  FoldingSetNodeID ID;
  ID.AddInteger(3U);
  ID.AddInteger(VT1.getRawBits());
  ID.AddInteger(VT2.getRawBits());
  ID.AddInteger(VT3.getRawBits());

  void *IP = nullptr;
  SDVTListNode *Result = VTListMap.FindNodeOrInsertPos(ID, IP);
  if (!Result) {
    EVT *Array = Allocator.Allocate<EVT>(3);
    Array[0] = VT1;
    Array[1] = VT2;
    Array[2] = VT3;
    Result = new (Allocator) SDVTListNode(ID.Intern(Allocator), Array, 3);
    VTListMap.InsertNode(Result, IP);
  }
  return Result->getSDVTList();
}

SDVTList SelectionDAG::getVTList(EVT VT1, EVT VT2, EVT VT3, EVT VT4) {
  FoldingSetNodeID ID;
  ID.AddInteger(4U);
  ID.AddInteger(VT1.getRawBits());
  ID.AddInteger(VT2.getRawBits());
  ID.AddInteger(VT3.getRawBits());
  ID.AddInteger(VT4.getRawBits());

  void *IP = nullptr;
  SDVTListNode *Result = VTListMap.FindNodeOrInsertPos(ID, IP);
  if (!Result) {
    EVT *Array = Allocator.Allocate<EVT>(4);
    Array[0] = VT1;
    Array[1] = VT2;
    Array[2] = VT3;
    Array[3] = VT4;
    Result = new (Allocator) SDVTListNode(ID.Intern(Allocator), Array, 4);
    VTListMap.InsertNode(Result, IP);
  }
  return Result->getSDVTList();
}

SDVTList SelectionDAG::getVTList(ArrayRef<EVT> VTs) {
  unsigned NumVTs = VTs.size();
  FoldingSetNodeID ID;
  ID.AddInteger(NumVTs);
  for (unsigned index = 0; index < NumVTs; index++) {
    ID.AddInteger(VTs[index].getRawBits());
  }

  void *IP = nullptr;
  SDVTListNode *Result = VTListMap.FindNodeOrInsertPos(ID, IP);
  if (!Result) {
    EVT *Array = Allocator.Allocate<EVT>(NumVTs);
    std::copy(VTs.begin(), VTs.end(), Array);
    Result = new (Allocator) SDVTListNode(ID.Intern(Allocator), Array, NumVTs);
    VTListMap.InsertNode(Result, IP);
  }
  return Result->getSDVTList();
}


/// UpdateNodeOperands - *Mutate* the specified node in-place to have the
/// specified operands.  If the resultant node already exists in the DAG,
/// this does not modify the specified node, instead it returns the node that
/// already exists.  If the resultant node does not exist in the DAG, the
/// input node is returned.  As a degenerate case, if you specify the same
/// input operands as the node already has, the input node is returned.
SDNode *SelectionDAG::UpdateNodeOperands(SDNode *N, SDValue Op) {
  assert(N->getNumOperands() == 1 && "Update with wrong number of operands");

  // Check to see if there is no change.
  if (Op == N->getOperand(0)) return N;

  // See if the modified node already exists.
  void *InsertPos = nullptr;
  if (SDNode *Existing = FindModifiedNodeSlot(N, Op, InsertPos))
    return Existing;

  // Nope it doesn't.  Remove the node from its current place in the maps.
  if (InsertPos)
    if (!RemoveNodeFromCSEMaps(N))
      InsertPos = nullptr;

  // Now we update the operands.
  N->OperandList[0].set(Op);

  // If this gets put into a CSE map, add it.
  if (InsertPos) CSEMap.InsertNode(N, InsertPos);
  return N;
}

SDNode *SelectionDAG::UpdateNodeOperands(SDNode *N, SDValue Op1, SDValue Op2) {
  assert(N->getNumOperands() == 2 && "Update with wrong number of operands");

  // Check to see if there is no change.
  if (Op1 == N->getOperand(0) && Op2 == N->getOperand(1))
    return N;   // No operands changed, just return the input node.

  // See if the modified node already exists.
  void *InsertPos = nullptr;
  if (SDNode *Existing = FindModifiedNodeSlot(N, Op1, Op2, InsertPos))
    return Existing;

  // Nope it doesn't.  Remove the node from its current place in the maps.
  if (InsertPos)
    if (!RemoveNodeFromCSEMaps(N))
      InsertPos = nullptr;

  // Now we update the operands.
  if (N->OperandList[0] != Op1)
    N->OperandList[0].set(Op1);
  if (N->OperandList[1] != Op2)
    N->OperandList[1].set(Op2);

  // If this gets put into a CSE map, add it.
  if (InsertPos) CSEMap.InsertNode(N, InsertPos);
  return N;
}

SDNode *SelectionDAG::
UpdateNodeOperands(SDNode *N, SDValue Op1, SDValue Op2, SDValue Op3) {
  SDValue Ops[] = { Op1, Op2, Op3 };
  return UpdateNodeOperands(N, Ops);
}

SDNode *SelectionDAG::
UpdateNodeOperands(SDNode *N, SDValue Op1, SDValue Op2,
                   SDValue Op3, SDValue Op4) {
  SDValue Ops[] = { Op1, Op2, Op3, Op4 };
  return UpdateNodeOperands(N, Ops);
}

SDNode *SelectionDAG::
UpdateNodeOperands(SDNode *N, SDValue Op1, SDValue Op2,
                   SDValue Op3, SDValue Op4, SDValue Op5) {
  SDValue Ops[] = { Op1, Op2, Op3, Op4, Op5 };
  return UpdateNodeOperands(N, Ops);
}

SDNode *SelectionDAG::
UpdateNodeOperands(SDNode *N, ArrayRef<SDValue> Ops) {
  unsigned NumOps = Ops.size();
  assert(N->getNumOperands() == NumOps &&
         "Update with wrong number of operands");

  // If no operands changed just return the input node.
  if (Ops.empty() || std::equal(Ops.begin(), Ops.end(), N->op_begin()))
    return N;

  // See if the modified node already exists.
  void *InsertPos = nullptr;
  if (SDNode *Existing = FindModifiedNodeSlot(N, Ops, InsertPos))
    return Existing;

  // Nope it doesn't.  Remove the node from its current place in the maps.
  if (InsertPos)
    if (!RemoveNodeFromCSEMaps(N))
      InsertPos = nullptr;

  // Now we update the operands.
  for (unsigned i = 0; i != NumOps; ++i)
    if (N->OperandList[i] != Ops[i])
      N->OperandList[i].set(Ops[i]);

  // If this gets put into a CSE map, add it.
  if (InsertPos) CSEMap.InsertNode(N, InsertPos);
  return N;
}

/// DropOperands - Release the operands and set this node to have
/// zero operands.
void SDNode::DropOperands() {
  // Unlike the code in MorphNodeTo that does this, we don't need to
  // watch for dead nodes here.
  for (op_iterator I = op_begin(), E = op_end(); I != E; ) {
    SDUse &Use = *I++;
    Use.set(SDValue());
  }
}

/// SelectNodeTo - These are wrappers around MorphNodeTo that accept a
/// machine opcode.
///
SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned MachineOpc,
                                   EVT VT) {
  SDVTList VTs = getVTList(VT);
  return SelectNodeTo(N, MachineOpc, VTs, None);
}

SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned MachineOpc,
                                   EVT VT, SDValue Op1) {
  SDVTList VTs = getVTList(VT);
  SDValue Ops[] = { Op1 };
  return SelectNodeTo(N, MachineOpc, VTs, Ops);
}

SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned MachineOpc,
                                   EVT VT, SDValue Op1,
                                   SDValue Op2) {
  SDVTList VTs = getVTList(VT);
  SDValue Ops[] = { Op1, Op2 };
  return SelectNodeTo(N, MachineOpc, VTs, Ops);
}

SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned MachineOpc,
                                   EVT VT, SDValue Op1,
                                   SDValue Op2, SDValue Op3) {
  SDVTList VTs = getVTList(VT);
  SDValue Ops[] = { Op1, Op2, Op3 };
  return SelectNodeTo(N, MachineOpc, VTs, Ops);
}

SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned MachineOpc,
                                   EVT VT, ArrayRef<SDValue> Ops) {
  SDVTList VTs = getVTList(VT);
  return SelectNodeTo(N, MachineOpc, VTs, Ops);
}

SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned MachineOpc,
                                   EVT VT1, EVT VT2, ArrayRef<SDValue> Ops) {
  SDVTList VTs = getVTList(VT1, VT2);
  return SelectNodeTo(N, MachineOpc, VTs, Ops);
}

SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned MachineOpc,
                                   EVT VT1, EVT VT2) {
  SDVTList VTs = getVTList(VT1, VT2);
  return SelectNodeTo(N, MachineOpc, VTs, None);
}

SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned MachineOpc,
                                   EVT VT1, EVT VT2, EVT VT3,
                                   ArrayRef<SDValue> Ops) {
  SDVTList VTs = getVTList(VT1, VT2, VT3);
  return SelectNodeTo(N, MachineOpc, VTs, Ops);
}

SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned MachineOpc,
                                   EVT VT1, EVT VT2, EVT VT3, EVT VT4,
                                   ArrayRef<SDValue> Ops) {
  SDVTList VTs = getVTList(VT1, VT2, VT3, VT4);
  return SelectNodeTo(N, MachineOpc, VTs, Ops);
}

SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned MachineOpc,
                                   EVT VT1, EVT VT2,
                                   SDValue Op1) {
  SDVTList VTs = getVTList(VT1, VT2);
  SDValue Ops[] = { Op1 };
  return SelectNodeTo(N, MachineOpc, VTs, Ops);
}

SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned MachineOpc,
                                   EVT VT1, EVT VT2,
                                   SDValue Op1, SDValue Op2) {
  SDVTList VTs = getVTList(VT1, VT2);
  SDValue Ops[] = { Op1, Op2 };
  return SelectNodeTo(N, MachineOpc, VTs, Ops);
}

SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned MachineOpc,
                                   EVT VT1, EVT VT2,
                                   SDValue Op1, SDValue Op2,
                                   SDValue Op3) {
  SDVTList VTs = getVTList(VT1, VT2);
  SDValue Ops[] = { Op1, Op2, Op3 };
  return SelectNodeTo(N, MachineOpc, VTs, Ops);
}

SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned MachineOpc,
                                   EVT VT1, EVT VT2, EVT VT3,
                                   SDValue Op1, SDValue Op2,
                                   SDValue Op3) {
  SDVTList VTs = getVTList(VT1, VT2, VT3);
  SDValue Ops[] = { Op1, Op2, Op3 };
  return SelectNodeTo(N, MachineOpc, VTs, Ops);
}

SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned MachineOpc,
                                   SDVTList VTs,ArrayRef<SDValue> Ops) {
  N = MorphNodeTo(N, ~MachineOpc, VTs, Ops);
  // Reset the NodeID to -1.
  N->setNodeId(-1);
  return N;
}

/// UpdadeSDLocOnMergedSDNode - If the opt level is -O0 then it throws away
/// the line number information on the merged node since it is not possible to
/// preserve the information that operation is associated with multiple lines.
/// This will make the debugger working better at -O0, were there is a higher
/// probability having other instructions associated with that line.
///
/// For IROrder, we keep the smaller of the two
SDNode *SelectionDAG::UpdadeSDLocOnMergedSDNode(SDNode *N, SDLoc OLoc) {
  DebugLoc NLoc = N->getDebugLoc();
  if (NLoc && OptLevel == CodeGenOpt::None && OLoc.getDebugLoc() != NLoc) {
    N->setDebugLoc(DebugLoc());
  }
  unsigned Order = std::min(N->getIROrder(), OLoc.getIROrder());
  N->setIROrder(Order);
  return N;
}

/// MorphNodeTo - This *mutates* the specified node to have the specified
/// return type, opcode, and operands.
///
/// Note that MorphNodeTo returns the resultant node.  If there is already a
/// node of the specified opcode and operands, it returns that node instead of
/// the current one.  Note that the SDLoc need not be the same.
///
/// Using MorphNodeTo is faster than creating a new node and swapping it in
/// with ReplaceAllUsesWith both because it often avoids allocating a new
/// node, and because it doesn't require CSE recalculation for any of
/// the node's users.
///
/// However, note that MorphNodeTo recursively deletes dead nodes from the DAG.
/// As a consequence it isn't appropriate to use from within the DAG combiner or
/// the legalizer which maintain worklists that would need to be updated when
/// deleting things.
SDNode *SelectionDAG::MorphNodeTo(SDNode *N, unsigned Opc,
                                  SDVTList VTs, ArrayRef<SDValue> Ops) {
  unsigned NumOps = Ops.size();
  // If an identical node already exists, use it.
  void *IP = nullptr;
  if (VTs.VTs[VTs.NumVTs-1] != MVT::Glue) {
    FoldingSetNodeID ID;
    AddNodeIDNode(ID, Opc, VTs, Ops);
    if (SDNode *ON = FindNodeOrInsertPos(ID, N->getDebugLoc(), IP))
      return UpdadeSDLocOnMergedSDNode(ON, SDLoc(N));
  }

  if (!RemoveNodeFromCSEMaps(N))
    IP = nullptr;

  // Start the morphing.
  N->NodeType = Opc;
  N->ValueList = VTs.VTs;
  N->NumValues = VTs.NumVTs;

  // Clear the operands list, updating used nodes to remove this from their
  // use list.  Keep track of any operands that become dead as a result.
  SmallPtrSet<SDNode*, 16> DeadNodeSet;
  for (SDNode::op_iterator I = N->op_begin(), E = N->op_end(); I != E; ) {
    SDUse &Use = *I++;
    SDNode *Used = Use.getNode();
    Use.set(SDValue());
    if (Used->use_empty())
      DeadNodeSet.insert(Used);
  }

  if (MachineSDNode *MN = dyn_cast<MachineSDNode>(N)) {
    // Initialize the memory references information.
    MN->setMemRefs(nullptr, nullptr);
    // If NumOps is larger than the # of operands we can have in a
    // MachineSDNode, reallocate the operand list.
    if (NumOps > MN->NumOperands || !MN->OperandsNeedDelete) {
      if (MN->OperandsNeedDelete)
        delete[] MN->OperandList;
      if (NumOps > array_lengthof(MN->LocalOperands))
        // We're creating a final node that will live unmorphed for the
        // remainder of the current SelectionDAG iteration, so we can allocate
        // the operands directly out of a pool with no recycling metadata.
        MN->InitOperands(OperandAllocator.Allocate<SDUse>(NumOps),
                         Ops.data(), NumOps);
      else
        MN->InitOperands(MN->LocalOperands, Ops.data(), NumOps);
      MN->OperandsNeedDelete = false;
    } else
      MN->InitOperands(MN->OperandList, Ops.data(), NumOps);
  } else {
    // If NumOps is larger than the # of operands we currently have, reallocate
    // the operand list.
    if (NumOps > N->NumOperands) {
      if (N->OperandsNeedDelete)
        delete[] N->OperandList;
      N->InitOperands(new SDUse[NumOps], Ops.data(), NumOps);
      N->OperandsNeedDelete = true;
    } else
      N->InitOperands(N->OperandList, Ops.data(), NumOps);
  }

  // Delete any nodes that are still dead after adding the uses for the
  // new operands.
  if (!DeadNodeSet.empty()) {
    SmallVector<SDNode *, 16> DeadNodes;
    for (SDNode *N : DeadNodeSet)
      if (N->use_empty())
        DeadNodes.push_back(N);
    RemoveDeadNodes(DeadNodes);
  }

  if (IP)
    CSEMap.InsertNode(N, IP);   // Memoize the new node.
  return N;
}


/// getMachineNode - These are used for target selectors to create a new node
/// with specified return type(s), MachineInstr opcode, and operands.
///
/// Note that getMachineNode returns the resultant node.  If there is already a
/// node of the specified opcode and operands, it returns that node instead of
/// the current one.
MachineSDNode *
SelectionDAG::getMachineNode(unsigned Opcode, SDLoc dl, EVT VT) {
  SDVTList VTs = getVTList(VT);
  return getMachineNode(Opcode, dl, VTs, None);
}

MachineSDNode *
SelectionDAG::getMachineNode(unsigned Opcode, SDLoc dl, EVT VT, SDValue Op1) {
  SDVTList VTs = getVTList(VT);
  SDValue Ops[] = { Op1 };
  return getMachineNode(Opcode, dl, VTs, Ops);
}

MachineSDNode *
SelectionDAG::getMachineNode(unsigned Opcode, SDLoc dl, EVT VT,
                             SDValue Op1, SDValue Op2) {
  SDVTList VTs = getVTList(VT);
  SDValue Ops[] = { Op1, Op2 };
  return getMachineNode(Opcode, dl, VTs, Ops);
}

MachineSDNode *
SelectionDAG::getMachineNode(unsigned Opcode, SDLoc dl, EVT VT,
                             SDValue Op1, SDValue Op2, SDValue Op3) {
  SDVTList VTs = getVTList(VT);
  SDValue Ops[] = { Op1, Op2, Op3 };
  return getMachineNode(Opcode, dl, VTs, Ops);
}

MachineSDNode *
SelectionDAG::getMachineNode(unsigned Opcode, SDLoc dl, EVT VT,
                             ArrayRef<SDValue> Ops) {
  SDVTList VTs = getVTList(VT);
  return getMachineNode(Opcode, dl, VTs, Ops);
}

MachineSDNode *
SelectionDAG::getMachineNode(unsigned Opcode, SDLoc dl, EVT VT1, EVT VT2) {
  SDVTList VTs = getVTList(VT1, VT2);
  return getMachineNode(Opcode, dl, VTs, None);
}

MachineSDNode *
SelectionDAG::getMachineNode(unsigned Opcode, SDLoc dl,
                             EVT VT1, EVT VT2, SDValue Op1) {
  SDVTList VTs = getVTList(VT1, VT2);
  SDValue Ops[] = { Op1 };
  return getMachineNode(Opcode, dl, VTs, Ops);
}

MachineSDNode *
SelectionDAG::getMachineNode(unsigned Opcode, SDLoc dl,
                             EVT VT1, EVT VT2, SDValue Op1, SDValue Op2) {
  SDVTList VTs = getVTList(VT1, VT2);
  SDValue Ops[] = { Op1, Op2 };
  return getMachineNode(Opcode, dl, VTs, Ops);
}

MachineSDNode *
SelectionDAG::getMachineNode(unsigned Opcode, SDLoc dl,
                             EVT VT1, EVT VT2, SDValue Op1,
                             SDValue Op2, SDValue Op3) {
  SDVTList VTs = getVTList(VT1, VT2);
  SDValue Ops[] = { Op1, Op2, Op3 };
  return getMachineNode(Opcode, dl, VTs, Ops);
}

MachineSDNode *
SelectionDAG::getMachineNode(unsigned Opcode, SDLoc dl,
                             EVT VT1, EVT VT2,
                             ArrayRef<SDValue> Ops) {
  SDVTList VTs = getVTList(VT1, VT2);
  return getMachineNode(Opcode, dl, VTs, Ops);
}

MachineSDNode *
SelectionDAG::getMachineNode(unsigned Opcode, SDLoc dl,
                             EVT VT1, EVT VT2, EVT VT3,
                             SDValue Op1, SDValue Op2) {
  SDVTList VTs = getVTList(VT1, VT2, VT3);
  SDValue Ops[] = { Op1, Op2 };
  return getMachineNode(Opcode, dl, VTs, Ops);
}

MachineSDNode *
SelectionDAG::getMachineNode(unsigned Opcode, SDLoc dl,
                             EVT VT1, EVT VT2, EVT VT3,
                             SDValue Op1, SDValue Op2, SDValue Op3) {
  SDVTList VTs = getVTList(VT1, VT2, VT3);
  SDValue Ops[] = { Op1, Op2, Op3 };
  return getMachineNode(Opcode, dl, VTs, Ops);
}

MachineSDNode *
SelectionDAG::getMachineNode(unsigned Opcode, SDLoc dl,
                             EVT VT1, EVT VT2, EVT VT3,
                             ArrayRef<SDValue> Ops) {
  SDVTList VTs = getVTList(VT1, VT2, VT3);
  return getMachineNode(Opcode, dl, VTs, Ops);
}

MachineSDNode *
SelectionDAG::getMachineNode(unsigned Opcode, SDLoc dl, EVT VT1,
                             EVT VT2, EVT VT3, EVT VT4,
                             ArrayRef<SDValue> Ops) {
  SDVTList VTs = getVTList(VT1, VT2, VT3, VT4);
  return getMachineNode(Opcode, dl, VTs, Ops);
}

MachineSDNode *
SelectionDAG::getMachineNode(unsigned Opcode, SDLoc dl,
                             ArrayRef<EVT> ResultTys,
                             ArrayRef<SDValue> Ops) {
  SDVTList VTs = getVTList(ResultTys);
  return getMachineNode(Opcode, dl, VTs, Ops);
}

MachineSDNode *
SelectionDAG::getMachineNode(unsigned Opcode, SDLoc DL, SDVTList VTs,
                             ArrayRef<SDValue> OpsArray) {
  bool DoCSE = VTs.VTs[VTs.NumVTs-1] != MVT::Glue;
  MachineSDNode *N;
  void *IP = nullptr;
  const SDValue *Ops = OpsArray.data();
  unsigned NumOps = OpsArray.size();

  if (DoCSE) {
    FoldingSetNodeID ID;
    AddNodeIDNode(ID, ~Opcode, VTs, OpsArray);
    IP = nullptr;
    if (SDNode *E = FindNodeOrInsertPos(ID, DL.getDebugLoc(), IP)) {
      return cast<MachineSDNode>(UpdadeSDLocOnMergedSDNode(E, DL));
    }
  }

  // Allocate a new MachineSDNode.
  N = new (NodeAllocator) MachineSDNode(~Opcode, DL.getIROrder(),
                                        DL.getDebugLoc(), VTs);

  // Initialize the operands list.
  if (NumOps > array_lengthof(N->LocalOperands))
    // We're creating a final node that will live unmorphed for the
    // remainder of the current SelectionDAG iteration, so we can allocate
    // the operands directly out of a pool with no recycling metadata.
    N->InitOperands(OperandAllocator.Allocate<SDUse>(NumOps),
                    Ops, NumOps);
  else
    N->InitOperands(N->LocalOperands, Ops, NumOps);
  N->OperandsNeedDelete = false;

  if (DoCSE)
    CSEMap.InsertNode(N, IP);

  InsertNode(N);
  return N;
}

/// getTargetExtractSubreg - A convenience function for creating
/// TargetOpcode::EXTRACT_SUBREG nodes.
SDValue
SelectionDAG::getTargetExtractSubreg(int SRIdx, SDLoc DL, EVT VT,
                                     SDValue Operand) {
  SDValue SRIdxVal = getTargetConstant(SRIdx, DL, MVT::i32);
  SDNode *Subreg = getMachineNode(TargetOpcode::EXTRACT_SUBREG, DL,
                                  VT, Operand, SRIdxVal);
  return SDValue(Subreg, 0);
}

/// getTargetInsertSubreg - A convenience function for creating
/// TargetOpcode::INSERT_SUBREG nodes.
SDValue
SelectionDAG::getTargetInsertSubreg(int SRIdx, SDLoc DL, EVT VT,
                                    SDValue Operand, SDValue Subreg) {
  SDValue SRIdxVal = getTargetConstant(SRIdx, DL, MVT::i32);
  SDNode *Result = getMachineNode(TargetOpcode::INSERT_SUBREG, DL,
                                  VT, Operand, Subreg, SRIdxVal);
  return SDValue(Result, 0);
}

/// getNodeIfExists - Get the specified node if it's already available, or
/// else return NULL.
SDNode *SelectionDAG::getNodeIfExists(unsigned Opcode, SDVTList VTList,
                                      ArrayRef<SDValue> Ops,
                                      const SDNodeFlags *Flags) {
  if (VTList.VTs[VTList.NumVTs - 1] != MVT::Glue) {
    FoldingSetNodeID ID;
    AddNodeIDNode(ID, Opcode, VTList, Ops);
    AddNodeIDFlags(ID, Opcode, Flags);
    void *IP = nullptr;
    if (SDNode *E = FindNodeOrInsertPos(ID, DebugLoc(), IP))
      return E;
  }
  return nullptr;
}

/// getDbgValue - Creates a SDDbgValue node.
///
/// SDNode
SDDbgValue *SelectionDAG::getDbgValue(MDNode *Var, MDNode *Expr, SDNode *N,
                                      unsigned R, bool IsIndirect, uint64_t Off,
                                      DebugLoc DL, unsigned O) {
  assert(cast<DILocalVariable>(Var)->isValidLocationForIntrinsic(DL) &&
         "Expected inlined-at fields to agree");
  return new (DbgInfo->getAlloc())
      SDDbgValue(Var, Expr, N, R, IsIndirect, Off, DL, O);
}

/// Constant
SDDbgValue *SelectionDAG::getConstantDbgValue(MDNode *Var, MDNode *Expr,
                                              const Value *C, uint64_t Off,
                                              DebugLoc DL, unsigned O) {
  assert(cast<DILocalVariable>(Var)->isValidLocationForIntrinsic(DL) &&
         "Expected inlined-at fields to agree");
  return new (DbgInfo->getAlloc()) SDDbgValue(Var, Expr, C, Off, DL, O);
}

/// FrameIndex
SDDbgValue *SelectionDAG::getFrameIndexDbgValue(MDNode *Var, MDNode *Expr,
                                                unsigned FI, uint64_t Off,
                                                DebugLoc DL, unsigned O) {
  assert(cast<DILocalVariable>(Var)->isValidLocationForIntrinsic(DL) &&
         "Expected inlined-at fields to agree");
  return new (DbgInfo->getAlloc()) SDDbgValue(Var, Expr, FI, Off, DL, O);
}

namespace {

/// RAUWUpdateListener - Helper for ReplaceAllUsesWith - When the node
/// pointed to by a use iterator is deleted, increment the use iterator
/// so that it doesn't dangle.
///
class RAUWUpdateListener : public SelectionDAG::DAGUpdateListener {
  SDNode::use_iterator &UI;
  SDNode::use_iterator &UE;

  void NodeDeleted(SDNode *N, SDNode *E) override {
    // Increment the iterator as needed.
    while (UI != UE && N == *UI)
      ++UI;
  }

public:
  RAUWUpdateListener(SelectionDAG &d,
                     SDNode::use_iterator &ui,
                     SDNode::use_iterator &ue)
    : SelectionDAG::DAGUpdateListener(d), UI(ui), UE(ue) {}
};

}

/// ReplaceAllUsesWith - Modify anything using 'From' to use 'To' instead.
/// This can cause recursive merging of nodes in the DAG.
///
/// This version assumes From has a single result value.
///
void SelectionDAG::ReplaceAllUsesWith(SDValue FromN, SDValue To) {
  SDNode *From = FromN.getNode();
  assert(From->getNumValues() == 1 && FromN.getResNo() == 0 &&
         "Cannot replace with this method!");
  assert(From != To.getNode() && "Cannot replace uses of with self");

  // Iterate over all the existing uses of From. New uses will be added
  // to the beginning of the use list, which we avoid visiting.
  // This specifically avoids visiting uses of From that arise while the
  // replacement is happening, because any such uses would be the result
  // of CSE: If an existing node looks like From after one of its operands
  // is replaced by To, we don't want to replace of all its users with To
  // too. See PR3018 for more info.
  SDNode::use_iterator UI = From->use_begin(), UE = From->use_end();
  RAUWUpdateListener Listener(*this, UI, UE);
  while (UI != UE) {
    SDNode *User = *UI;

    // This node is about to morph, remove its old self from the CSE maps.
    RemoveNodeFromCSEMaps(User);

    // A user can appear in a use list multiple times, and when this
    // happens the uses are usually next to each other in the list.
    // To help reduce the number of CSE recomputations, process all
    // the uses of this user that we can find this way.
    do {
      SDUse &Use = UI.getUse();
      ++UI;
      Use.set(To);
    } while (UI != UE && *UI == User);

    // Now that we have modified User, add it back to the CSE maps.  If it
    // already exists there, recursively merge the results together.
    AddModifiedNodeToCSEMaps(User);
  }

  // If we just RAUW'd the root, take note.
  if (FromN == getRoot())
    setRoot(To);
}

/// ReplaceAllUsesWith - Modify anything using 'From' to use 'To' instead.
/// This can cause recursive merging of nodes in the DAG.
///
/// This version assumes that for each value of From, there is a
/// corresponding value in To in the same position with the same type.
///
void SelectionDAG::ReplaceAllUsesWith(SDNode *From, SDNode *To) {
#ifndef NDEBUG
  for (unsigned i = 0, e = From->getNumValues(); i != e; ++i)
    assert((!From->hasAnyUseOfValue(i) ||
            From->getValueType(i) == To->getValueType(i)) &&
           "Cannot use this version of ReplaceAllUsesWith!");
#endif

  // Handle the trivial case.
  if (From == To)
    return;

  // Iterate over just the existing users of From. See the comments in
  // the ReplaceAllUsesWith above.
  SDNode::use_iterator UI = From->use_begin(), UE = From->use_end();
  RAUWUpdateListener Listener(*this, UI, UE);
  while (UI != UE) {
    SDNode *User = *UI;

    // This node is about to morph, remove its old self from the CSE maps.
    RemoveNodeFromCSEMaps(User);

    // A user can appear in a use list multiple times, and when this
    // happens the uses are usually next to each other in the list.
    // To help reduce the number of CSE recomputations, process all
    // the uses of this user that we can find this way.
    do {
      SDUse &Use = UI.getUse();
      ++UI;
      Use.setNode(To);
    } while (UI != UE && *UI == User);

    // Now that we have modified User, add it back to the CSE maps.  If it
    // already exists there, recursively merge the results together.
    AddModifiedNodeToCSEMaps(User);
  }

  // If we just RAUW'd the root, take note.
  if (From == getRoot().getNode())
    setRoot(SDValue(To, getRoot().getResNo()));
}

/// ReplaceAllUsesWith - Modify anything using 'From' to use 'To' instead.
/// This can cause recursive merging of nodes in the DAG.
///
/// This version can replace From with any result values.  To must match the
/// number and types of values returned by From.
void SelectionDAG::ReplaceAllUsesWith(SDNode *From, const SDValue *To) {
  if (From->getNumValues() == 1)  // Handle the simple case efficiently.
    return ReplaceAllUsesWith(SDValue(From, 0), To[0]);

  // Iterate over just the existing users of From. See the comments in
  // the ReplaceAllUsesWith above.
  SDNode::use_iterator UI = From->use_begin(), UE = From->use_end();
  RAUWUpdateListener Listener(*this, UI, UE);
  while (UI != UE) {
    SDNode *User = *UI;

    // This node is about to morph, remove its old self from the CSE maps.
    RemoveNodeFromCSEMaps(User);

    // A user can appear in a use list multiple times, and when this
    // happens the uses are usually next to each other in the list.
    // To help reduce the number of CSE recomputations, process all
    // the uses of this user that we can find this way.
    do {
      SDUse &Use = UI.getUse();
      const SDValue &ToOp = To[Use.getResNo()];
      ++UI;
      Use.set(ToOp);
    } while (UI != UE && *UI == User);

    // Now that we have modified User, add it back to the CSE maps.  If it
    // already exists there, recursively merge the results together.
    AddModifiedNodeToCSEMaps(User);
  }

  // If we just RAUW'd the root, take note.
  if (From == getRoot().getNode())
    setRoot(SDValue(To[getRoot().getResNo()]));
}

/// ReplaceAllUsesOfValueWith - Replace any uses of From with To, leaving
/// uses of other values produced by From.getNode() alone.  The Deleted
/// vector is handled the same way as for ReplaceAllUsesWith.
void SelectionDAG::ReplaceAllUsesOfValueWith(SDValue From, SDValue To){
  // Handle the really simple, really trivial case efficiently.
  if (From == To) return;

  // Handle the simple, trivial, case efficiently.
  if (From.getNode()->getNumValues() == 1) {
    ReplaceAllUsesWith(From, To);
    return;
  }

  // Iterate over just the existing users of From. See the comments in
  // the ReplaceAllUsesWith above.
  SDNode::use_iterator UI = From.getNode()->use_begin(),
                       UE = From.getNode()->use_end();
  RAUWUpdateListener Listener(*this, UI, UE);
  while (UI != UE) {
    SDNode *User = *UI;
    bool UserRemovedFromCSEMaps = false;

    // A user can appear in a use list multiple times, and when this
    // happens the uses are usually next to each other in the list.
    // To help reduce the number of CSE recomputations, process all
    // the uses of this user that we can find this way.
    do {
      SDUse &Use = UI.getUse();

      // Skip uses of different values from the same node.
      if (Use.getResNo() != From.getResNo()) {
        ++UI;
        continue;
      }

      // If this node hasn't been modified yet, it's still in the CSE maps,
      // so remove its old self from the CSE maps.
      if (!UserRemovedFromCSEMaps) {
        RemoveNodeFromCSEMaps(User);
        UserRemovedFromCSEMaps = true;
      }

      ++UI;
      Use.set(To);
    } while (UI != UE && *UI == User);

    // We are iterating over all uses of the From node, so if a use
    // doesn't use the specific value, no changes are made.
    if (!UserRemovedFromCSEMaps)
      continue;

    // Now that we have modified User, add it back to the CSE maps.  If it
    // already exists there, recursively merge the results together.
    AddModifiedNodeToCSEMaps(User);
  }

  // If we just RAUW'd the root, take note.
  if (From == getRoot())
    setRoot(To);
}

namespace {
  /// UseMemo - This class is used by SelectionDAG::ReplaceAllUsesOfValuesWith
  /// to record information about a use.
  struct UseMemo {
    SDNode *User;
    unsigned Index;
    SDUse *Use;
  };

  /// operator< - Sort Memos by User.
  bool operator<(const UseMemo &L, const UseMemo &R) {
    return (intptr_t)L.User < (intptr_t)R.User;
  }
}

/// ReplaceAllUsesOfValuesWith - Replace any uses of From with To, leaving
/// uses of other values produced by From.getNode() alone.  The same value
/// may appear in both the From and To list.  The Deleted vector is
/// handled the same way as for ReplaceAllUsesWith.
void SelectionDAG::ReplaceAllUsesOfValuesWith(const SDValue *From,
                                              const SDValue *To,
                                              unsigned Num){
  // Handle the simple, trivial case efficiently.
  if (Num == 1)
    return ReplaceAllUsesOfValueWith(*From, *To);

  // Read up all the uses and make records of them. This helps
  // processing new uses that are introduced during the
  // replacement process.
  SmallVector<UseMemo, 4> Uses;
  for (unsigned i = 0; i != Num; ++i) {
    unsigned FromResNo = From[i].getResNo();
    SDNode *FromNode = From[i].getNode();
    for (SDNode::use_iterator UI = FromNode->use_begin(),
         E = FromNode->use_end(); UI != E; ++UI) {
      SDUse &Use = UI.getUse();
      if (Use.getResNo() == FromResNo) {
        UseMemo Memo = { *UI, i, &Use };
        Uses.push_back(Memo);
      }
    }
  }

  // Sort the uses, so that all the uses from a given User are together.
  std::sort(Uses.begin(), Uses.end());

  for (unsigned UseIndex = 0, UseIndexEnd = Uses.size();
       UseIndex != UseIndexEnd; ) {
    // We know that this user uses some value of From.  If it is the right
    // value, update it.
    SDNode *User = Uses[UseIndex].User;

    // This node is about to morph, remove its old self from the CSE maps.
    RemoveNodeFromCSEMaps(User);

    // The Uses array is sorted, so all the uses for a given User
    // are next to each other in the list.
    // To help reduce the number of CSE recomputations, process all
    // the uses of this user that we can find this way.
    do {
      unsigned i = Uses[UseIndex].Index;
      SDUse &Use = *Uses[UseIndex].Use;
      ++UseIndex;

      Use.set(To[i]);
    } while (UseIndex != UseIndexEnd && Uses[UseIndex].User == User);

    // Now that we have modified User, add it back to the CSE maps.  If it
    // already exists there, recursively merge the results together.
    AddModifiedNodeToCSEMaps(User);
  }
}

/// AssignTopologicalOrder - Assign a unique node id for each node in the DAG
/// based on their topological order. It returns the maximum id and a vector
/// of the SDNodes* in assigned order by reference.
unsigned SelectionDAG::AssignTopologicalOrder() {

  unsigned DAGSize = 0;

  // SortedPos tracks the progress of the algorithm. Nodes before it are
  // sorted, nodes after it are unsorted. When the algorithm completes
  // it is at the end of the list.
  allnodes_iterator SortedPos = allnodes_begin();

  // Visit all the nodes. Move nodes with no operands to the front of
  // the list immediately. Annotate nodes that do have operands with their
  // operand count. Before we do this, the Node Id fields of the nodes
  // may contain arbitrary values. After, the Node Id fields for nodes
  // before SortedPos will contain the topological sort index, and the
  // Node Id fields for nodes At SortedPos and after will contain the
  // count of outstanding operands.
  for (allnodes_iterator I = allnodes_begin(),E = allnodes_end(); I != E; ) {
    SDNode *N = I++;
    checkForCycles(N, this);
    unsigned Degree = N->getNumOperands();
    if (Degree == 0) {
      // A node with no uses, add it to the result array immediately.
      N->setNodeId(DAGSize++);
      allnodes_iterator Q = N;
      if (Q != SortedPos)
        SortedPos = AllNodes.insert(SortedPos, AllNodes.remove(Q));
      assert(SortedPos != AllNodes.end() && "Overran node list");
      ++SortedPos;
    } else {
      // Temporarily use the Node Id as scratch space for the degree count.
      N->setNodeId(Degree);
    }
  }

  // Visit all the nodes. As we iterate, move nodes into sorted order,
  // such that by the time the end is reached all nodes will be sorted.
  for (allnodes_iterator I = allnodes_begin(),E = allnodes_end(); I != E; ++I) {
    SDNode *N = I;
    checkForCycles(N, this);
    // N is in sorted position, so all its uses have one less operand
    // that needs to be sorted.
    for (SDNode::use_iterator UI = N->use_begin(), UE = N->use_end();
         UI != UE; ++UI) {
      SDNode *P = *UI;
      unsigned Degree = P->getNodeId();
      assert(Degree != 0 && "Invalid node degree");
      --Degree;
      if (Degree == 0) {
        // All of P's operands are sorted, so P may sorted now.
        P->setNodeId(DAGSize++);
        if (P != SortedPos)
          SortedPos = AllNodes.insert(SortedPos, AllNodes.remove(P));
        assert(SortedPos != AllNodes.end() && "Overran node list");
        ++SortedPos;
      } else {
        // Update P's outstanding operand count.
        P->setNodeId(Degree);
      }
    }
    if (I == SortedPos) {
#ifndef NDEBUG
      SDNode *S = ++I;
      dbgs() << "Overran sorted position:\n";
      S->dumprFull(this); dbgs() << "\n";
      dbgs() << "Checking if this is due to cycles\n";
      checkForCycles(this, true);
#endif
      llvm_unreachable(nullptr);
    }
  }

  assert(SortedPos == AllNodes.end() &&
         "Topological sort incomplete!");
  assert(AllNodes.front().getOpcode() == ISD::EntryToken &&
         "First node in topological sort is not the entry token!");
  assert(AllNodes.front().getNodeId() == 0 &&
         "First node in topological sort has non-zero id!");
  assert(AllNodes.front().getNumOperands() == 0 &&
         "First node in topological sort has operands!");
  assert(AllNodes.back().getNodeId() == (int)DAGSize-1 &&
         "Last node in topologic sort has unexpected id!");
  assert(AllNodes.back().use_empty() &&
         "Last node in topologic sort has users!");
  assert(DAGSize == allnodes_size() && "Node count mismatch!");
  return DAGSize;
}

/// AddDbgValue - Add a dbg_value SDNode. If SD is non-null that means the
/// value is produced by SD.
void SelectionDAG::AddDbgValue(SDDbgValue *DB, SDNode *SD, bool isParameter) {
  if (SD) {
    assert(DbgInfo->getSDDbgValues(SD).empty() || SD->getHasDebugValue());
    SD->setHasDebugValue(true);
  }
  DbgInfo->add(DB, SD, isParameter);
}

/// TransferDbgValues - Transfer SDDbgValues.
void SelectionDAG::TransferDbgValues(SDValue From, SDValue To) {
  if (From == To || !From.getNode()->getHasDebugValue())
    return;
  SDNode *FromNode = From.getNode();
  SDNode *ToNode = To.getNode();
  ArrayRef<SDDbgValue *> DVs = GetDbgValues(FromNode);
  SmallVector<SDDbgValue *, 2> ClonedDVs;
  for (ArrayRef<SDDbgValue *>::iterator I = DVs.begin(), E = DVs.end();
       I != E; ++I) {
    SDDbgValue *Dbg = *I;
    if (Dbg->getKind() == SDDbgValue::SDNODE) {
      SDDbgValue *Clone =
          getDbgValue(Dbg->getVariable(), Dbg->getExpression(), ToNode,
                      To.getResNo(), Dbg->isIndirect(), Dbg->getOffset(),
                      Dbg->getDebugLoc(), Dbg->getOrder());
      ClonedDVs.push_back(Clone);
    }
  }
  for (SmallVectorImpl<SDDbgValue *>::iterator I = ClonedDVs.begin(),
         E = ClonedDVs.end(); I != E; ++I)
    AddDbgValue(*I, ToNode, false);
}

//===----------------------------------------------------------------------===//
//                              SDNode Class
//===----------------------------------------------------------------------===//

HandleSDNode::~HandleSDNode() {
  DropOperands();
}

GlobalAddressSDNode::GlobalAddressSDNode(unsigned Opc, unsigned Order,
                                         DebugLoc DL, const GlobalValue *GA,
                                         EVT VT, int64_t o, unsigned char TF)
  : SDNode(Opc, Order, DL, getSDVTList(VT)), Offset(o), TargetFlags(TF) {
  TheGlobal = GA;
}

AddrSpaceCastSDNode::AddrSpaceCastSDNode(unsigned Order, DebugLoc dl, EVT VT,
                                         SDValue X, unsigned SrcAS,
                                         unsigned DestAS)
 : UnarySDNode(ISD::ADDRSPACECAST, Order, dl, getSDVTList(VT), X),
   SrcAddrSpace(SrcAS), DestAddrSpace(DestAS) {}

MemSDNode::MemSDNode(unsigned Opc, unsigned Order, DebugLoc dl, SDVTList VTs,
                     EVT memvt, MachineMemOperand *mmo)
 : SDNode(Opc, Order, dl, VTs), MemoryVT(memvt), MMO(mmo) {
  SubclassData = encodeMemSDNodeFlags(0, ISD::UNINDEXED, MMO->isVolatile(),
                                      MMO->isNonTemporal(), MMO->isInvariant());
  assert(isVolatile() == MMO->isVolatile() && "Volatile encoding error!");
  assert(isNonTemporal() == MMO->isNonTemporal() &&
         "Non-temporal encoding error!");
  // We check here that the size of the memory operand fits within the size of
  // the MMO. This is because the MMO might indicate only a possible address
  // range instead of specifying the affected memory addresses precisely.
  assert(memvt.getStoreSize() <= MMO->getSize() && "Size mismatch!");
}

MemSDNode::MemSDNode(unsigned Opc, unsigned Order, DebugLoc dl, SDVTList VTs,
                     ArrayRef<SDValue> Ops, EVT memvt, MachineMemOperand *mmo)
   : SDNode(Opc, Order, dl, VTs, Ops),
     MemoryVT(memvt), MMO(mmo) {
  SubclassData = encodeMemSDNodeFlags(0, ISD::UNINDEXED, MMO->isVolatile(),
                                      MMO->isNonTemporal(), MMO->isInvariant());
  assert(isVolatile() == MMO->isVolatile() && "Volatile encoding error!");
  assert(memvt.getStoreSize() <= MMO->getSize() && "Size mismatch!");
}

/// Profile - Gather unique data for the node.
///
void SDNode::Profile(FoldingSetNodeID &ID) const {
  AddNodeIDNode(ID, this);
}

namespace {
  struct EVTArray {
    std::vector<EVT> VTs;

    EVTArray() {
      VTs.reserve(MVT::LAST_VALUETYPE);
      for (unsigned i = 0; i < MVT::LAST_VALUETYPE; ++i)
        VTs.push_back(MVT((MVT::SimpleValueType)i));
    }
  };
}

static ManagedStatic<std::set<EVT, EVT::compareRawBits> > EVTs;
static ManagedStatic<EVTArray> SimpleVTArray;
static ManagedStatic<sys::SmartMutex<true> > VTMutex;

/// getValueTypeList - Return a pointer to the specified value type.
///
const EVT *SDNode::getValueTypeList(EVT VT) {
  if (VT.isExtended()) {
    sys::SmartScopedLock<true> Lock(*VTMutex);
    return &(*EVTs->insert(VT).first);
  } else {
    assert(VT.getSimpleVT() < MVT::LAST_VALUETYPE &&
           "Value type out of range!");
    return &SimpleVTArray->VTs[VT.getSimpleVT().SimpleTy];
  }
}

/// hasNUsesOfValue - Return true if there are exactly NUSES uses of the
/// indicated value.  This method ignores uses of other values defined by this
/// operation.
bool SDNode::hasNUsesOfValue(unsigned NUses, unsigned Value) const {
  assert(Value < getNumValues() && "Bad value!");

  // TODO: Only iterate over uses of a given value of the node
  for (SDNode::use_iterator UI = use_begin(), E = use_end(); UI != E; ++UI) {
    if (UI.getUse().getResNo() == Value) {
      if (NUses == 0)
        return false;
      --NUses;
    }
  }

  // Found exactly the right number of uses?
  return NUses == 0;
}


/// hasAnyUseOfValue - Return true if there are any use of the indicated
/// value. This method ignores uses of other values defined by this operation.
bool SDNode::hasAnyUseOfValue(unsigned Value) const {
  assert(Value < getNumValues() && "Bad value!");

  for (SDNode::use_iterator UI = use_begin(), E = use_end(); UI != E; ++UI)
    if (UI.getUse().getResNo() == Value)
      return true;

  return false;
}


/// isOnlyUserOf - Return true if this node is the only use of N.
///
bool SDNode::isOnlyUserOf(const SDNode *N) const {
  bool Seen = false;
  for (SDNode::use_iterator I = N->use_begin(), E = N->use_end(); I != E; ++I) {
    SDNode *User = *I;
    if (User == this)
      Seen = true;
    else
      return false;
  }

  return Seen;
}

/// isOperand - Return true if this node is an operand of N.
///
bool SDValue::isOperandOf(const SDNode *N) const {
  for (const SDValue &Op : N->op_values())
    if (*this == Op)
      return true;
  return false;
}

bool SDNode::isOperandOf(const SDNode *N) const {
  for (const SDValue &Op : N->op_values())
    if (this == Op.getNode())
      return true;
  return false;
}

/// reachesChainWithoutSideEffects - Return true if this operand (which must
/// be a chain) reaches the specified operand without crossing any
/// side-effecting instructions on any chain path.  In practice, this looks
/// through token factors and non-volatile loads.  In order to remain efficient,
/// this only looks a couple of nodes in, it does not do an exhaustive search.
bool SDValue::reachesChainWithoutSideEffects(SDValue Dest,
                                               unsigned Depth) const {
  if (*this == Dest) return true;

  // Don't search too deeply, we just want to be able to see through
  // TokenFactor's etc.
  if (Depth == 0) return false;

  // If this is a token factor, all inputs to the TF happen in parallel.  If any
  // of the operands of the TF does not reach dest, then we cannot do the xform.
  if (getOpcode() == ISD::TokenFactor) {
    for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
      if (!getOperand(i).reachesChainWithoutSideEffects(Dest, Depth-1))
        return false;
    return true;
  }

  // Loads don't have side effects, look through them.
  if (LoadSDNode *Ld = dyn_cast<LoadSDNode>(*this)) {
    if (!Ld->isVolatile())
      return Ld->getChain().reachesChainWithoutSideEffects(Dest, Depth-1);
  }
  return false;
}

/// hasPredecessor - Return true if N is a predecessor of this node.
/// N is either an operand of this node, or can be reached by recursively
/// traversing up the operands.
/// NOTE: This is an expensive method. Use it carefully.
bool SDNode::hasPredecessor(const SDNode *N) const {
  SmallPtrSet<const SDNode *, 32> Visited;
  SmallVector<const SDNode *, 16> Worklist;
  return hasPredecessorHelper(N, Visited, Worklist);
}

bool
SDNode::hasPredecessorHelper(const SDNode *N,
                             SmallPtrSetImpl<const SDNode *> &Visited,
                             SmallVectorImpl<const SDNode *> &Worklist) const {
  if (Visited.empty()) {
    Worklist.push_back(this);
  } else {
    // Take a look in the visited set. If we've already encountered this node
    // we needn't search further.
    if (Visited.count(N))
      return true;
  }

  // Haven't visited N yet. Continue the search.
  while (!Worklist.empty()) {
    const SDNode *M = Worklist.pop_back_val();
    for (const SDValue &OpV : M->op_values()) {
      SDNode *Op = OpV.getNode();
      if (Visited.insert(Op).second)
        Worklist.push_back(Op);
      if (Op == N)
        return true;
    }
  }

  return false;
}

uint64_t SDNode::getConstantOperandVal(unsigned Num) const {
  assert(Num < NumOperands && "Invalid child # of SDNode!");
  return cast<ConstantSDNode>(OperandList[Num])->getZExtValue();
}

SDValue SelectionDAG::UnrollVectorOp(SDNode *N, unsigned ResNE) {
  assert(N->getNumValues() == 1 &&
         "Can't unroll a vector with multiple results!");

  EVT VT = N->getValueType(0);
  unsigned NE = VT.getVectorNumElements();
  EVT EltVT = VT.getVectorElementType();
  SDLoc dl(N);

  SmallVector<SDValue, 8> Scalars;
  SmallVector<SDValue, 4> Operands(N->getNumOperands());

  // If ResNE is 0, fully unroll the vector op.
  if (ResNE == 0)
    ResNE = NE;
  else if (NE > ResNE)
    NE = ResNE;

  unsigned i;
  for (i= 0; i != NE; ++i) {
    for (unsigned j = 0, e = N->getNumOperands(); j != e; ++j) {
      SDValue Operand = N->getOperand(j);
      EVT OperandVT = Operand.getValueType();
      if (OperandVT.isVector()) {
        // A vector operand; extract a single element.
        EVT OperandEltVT = OperandVT.getVectorElementType();
        Operands[j] =
            getNode(ISD::EXTRACT_VECTOR_ELT, dl, OperandEltVT, Operand,
                    getConstant(i, dl, TLI->getVectorIdxTy(getDataLayout())));
      } else {
        // A scalar operand; just use it as is.
        Operands[j] = Operand;
      }
    }

    switch (N->getOpcode()) {
    default:
      Scalars.push_back(getNode(N->getOpcode(), dl, EltVT, Operands));
      break;
    case ISD::VSELECT:
      Scalars.push_back(getNode(ISD::SELECT, dl, EltVT, Operands));
      break;
    case ISD::SHL:
    case ISD::SRA:
    case ISD::SRL:
    case ISD::ROTL:
    case ISD::ROTR:
      Scalars.push_back(getNode(N->getOpcode(), dl, EltVT, Operands[0],
                               getShiftAmountOperand(Operands[0].getValueType(),
                                                     Operands[1])));
      break;
    case ISD::SIGN_EXTEND_INREG:
    case ISD::FP_ROUND_INREG: {
      EVT ExtVT = cast<VTSDNode>(Operands[1])->getVT().getVectorElementType();
      Scalars.push_back(getNode(N->getOpcode(), dl, EltVT,
                                Operands[0],
                                getValueType(ExtVT)));
    }
    }
  }

  for (; i < ResNE; ++i)
    Scalars.push_back(getUNDEF(EltVT));

  return getNode(ISD::BUILD_VECTOR, dl,
                 EVT::getVectorVT(*getContext(), EltVT, ResNE), Scalars);
}


/// isConsecutiveLoad - Return true if LD is loading 'Bytes' bytes from a
/// location that is 'Dist' units away from the location that the 'Base' load
/// is loading from.
bool SelectionDAG::isConsecutiveLoad(LoadSDNode *LD, LoadSDNode *Base,
                                     unsigned Bytes, int Dist) const {
  if (LD->getChain() != Base->getChain())
    return false;
  EVT VT = LD->getValueType(0);
  if (VT.getSizeInBits() / 8 != Bytes)
    return false;

  SDValue Loc = LD->getOperand(1);
  SDValue BaseLoc = Base->getOperand(1);
  if (Loc.getOpcode() == ISD::FrameIndex) {
    if (BaseLoc.getOpcode() != ISD::FrameIndex)
      return false;
    const MachineFrameInfo *MFI = getMachineFunction().getFrameInfo();
    int FI  = cast<FrameIndexSDNode>(Loc)->getIndex();
    int BFI = cast<FrameIndexSDNode>(BaseLoc)->getIndex();
    int FS  = MFI->getObjectSize(FI);
    int BFS = MFI->getObjectSize(BFI);
    if (FS != BFS || FS != (int)Bytes) return false;
    return MFI->getObjectOffset(FI) == (MFI->getObjectOffset(BFI) + Dist*Bytes);
  }

  // Handle X + C.
  if (isBaseWithConstantOffset(Loc)) {
    int64_t LocOffset = cast<ConstantSDNode>(Loc.getOperand(1))->getSExtValue();
    if (Loc.getOperand(0) == BaseLoc) {
      // If the base location is a simple address with no offset itself, then
      // the second load's first add operand should be the base address.
      if (LocOffset == Dist * (int)Bytes)
        return true;
    } else if (isBaseWithConstantOffset(BaseLoc)) {
      // The base location itself has an offset, so subtract that value from the
      // second load's offset before comparing to distance * size.
      int64_t BOffset =
        cast<ConstantSDNode>(BaseLoc.getOperand(1))->getSExtValue();
      if (Loc.getOperand(0) == BaseLoc.getOperand(0)) {
        if ((LocOffset - BOffset) == Dist * (int)Bytes)
          return true;
      }
    }
  }
  const GlobalValue *GV1 = nullptr;
  const GlobalValue *GV2 = nullptr;
  int64_t Offset1 = 0;
  int64_t Offset2 = 0;
  bool isGA1 = TLI->isGAPlusOffset(Loc.getNode(), GV1, Offset1);
  bool isGA2 = TLI->isGAPlusOffset(BaseLoc.getNode(), GV2, Offset2);
  if (isGA1 && isGA2 && GV1 == GV2)
    return Offset1 == (Offset2 + Dist*Bytes);
  return false;
}


/// InferPtrAlignment - Infer alignment of a load / store address. Return 0 if
/// it cannot be inferred.
unsigned SelectionDAG::InferPtrAlignment(SDValue Ptr) const {
  // If this is a GlobalAddress + cst, return the alignment.
  const GlobalValue *GV;
  int64_t GVOffset = 0;
  if (TLI->isGAPlusOffset(Ptr.getNode(), GV, GVOffset)) {
    unsigned PtrWidth = getDataLayout().getPointerTypeSizeInBits(GV->getType());
    APInt KnownZero(PtrWidth, 0), KnownOne(PtrWidth, 0);
    llvm::computeKnownBits(const_cast<GlobalValue *>(GV), KnownZero, KnownOne,
                           getDataLayout());
    unsigned AlignBits = KnownZero.countTrailingOnes();
    unsigned Align = AlignBits ? 1 << std::min(31U, AlignBits) : 0;
    if (Align)
      return MinAlign(Align, GVOffset);
  }

  // If this is a direct reference to a stack slot, use information about the
  // stack slot's alignment.
  int FrameIdx = 1 << 31;
  int64_t FrameOffset = 0;
  if (FrameIndexSDNode *FI = dyn_cast<FrameIndexSDNode>(Ptr)) {
    FrameIdx = FI->getIndex();
  } else if (isBaseWithConstantOffset(Ptr) &&
             isa<FrameIndexSDNode>(Ptr.getOperand(0))) {
    // Handle FI+Cst
    FrameIdx = cast<FrameIndexSDNode>(Ptr.getOperand(0))->getIndex();
    FrameOffset = Ptr.getConstantOperandVal(1);
  }

  if (FrameIdx != (1 << 31)) {
    const MachineFrameInfo &MFI = *getMachineFunction().getFrameInfo();
    unsigned FIInfoAlign = MinAlign(MFI.getObjectAlignment(FrameIdx),
                                    FrameOffset);
    return FIInfoAlign;
  }

  return 0;
}

/// GetSplitDestVTs - Compute the VTs needed for the low/hi parts of a type
/// which is split (or expanded) into two not necessarily identical pieces.
std::pair<EVT, EVT> SelectionDAG::GetSplitDestVTs(const EVT &VT) const {
  // Currently all types are split in half.
  EVT LoVT, HiVT;
  if (!VT.isVector()) {
    LoVT = HiVT = TLI->getTypeToTransformTo(*getContext(), VT);
  } else {
    unsigned NumElements = VT.getVectorNumElements();
    assert(!(NumElements & 1) && "Splitting vector, but not in half!");
    LoVT = HiVT = EVT::getVectorVT(*getContext(), VT.getVectorElementType(),
                                   NumElements/2);
  }
  return std::make_pair(LoVT, HiVT);
}

/// SplitVector - Split the vector with EXTRACT_SUBVECTOR and return the
/// low/high part.
std::pair<SDValue, SDValue>
SelectionDAG::SplitVector(const SDValue &N, const SDLoc &DL, const EVT &LoVT,
                          const EVT &HiVT) {
  assert(LoVT.getVectorNumElements() + HiVT.getVectorNumElements() <=
         N.getValueType().getVectorNumElements() &&
         "More vector elements requested than available!");
  SDValue Lo, Hi;
  Lo = getNode(ISD::EXTRACT_SUBVECTOR, DL, LoVT, N,
               getConstant(0, DL, TLI->getVectorIdxTy(getDataLayout())));
  Hi = getNode(ISD::EXTRACT_SUBVECTOR, DL, HiVT, N,
               getConstant(LoVT.getVectorNumElements(), DL,
                           TLI->getVectorIdxTy(getDataLayout())));
  return std::make_pair(Lo, Hi);
}

void SelectionDAG::ExtractVectorElements(SDValue Op,
                                         SmallVectorImpl<SDValue> &Args,
                                         unsigned Start, unsigned Count) {
  EVT VT = Op.getValueType();
  if (Count == 0)
    Count = VT.getVectorNumElements();

  EVT EltVT = VT.getVectorElementType();
  EVT IdxTy = TLI->getVectorIdxTy(getDataLayout());
  SDLoc SL(Op);
  for (unsigned i = Start, e = Start + Count; i != e; ++i) {
    Args.push_back(getNode(ISD::EXTRACT_VECTOR_ELT, SL, EltVT,
                           Op, getConstant(i, SL, IdxTy)));
  }
}

// getAddressSpace - Return the address space this GlobalAddress belongs to.
unsigned GlobalAddressSDNode::getAddressSpace() const {
  return getGlobal()->getType()->getAddressSpace();
}


Type *ConstantPoolSDNode::getType() const {
  if (isMachineConstantPoolEntry())
    return Val.MachineCPVal->getType();
  return Val.ConstVal->getType();
}

bool BuildVectorSDNode::isConstantSplat(APInt &SplatValue,
                                        APInt &SplatUndef,
                                        unsigned &SplatBitSize,
                                        bool &HasAnyUndefs,
                                        unsigned MinSplatBits,
                                        bool isBigEndian) const {
  EVT VT = getValueType(0);
  assert(VT.isVector() && "Expected a vector type");
  unsigned sz = VT.getSizeInBits();
  if (MinSplatBits > sz)
    return false;

  SplatValue = APInt(sz, 0);
  SplatUndef = APInt(sz, 0);

  // Get the bits.  Bits with undefined values (when the corresponding element
  // of the vector is an ISD::UNDEF value) are set in SplatUndef and cleared
  // in SplatValue.  If any of the values are not constant, give up and return
  // false.
  unsigned int nOps = getNumOperands();
  assert(nOps > 0 && "isConstantSplat has 0-size build vector");
  unsigned EltBitSize = VT.getVectorElementType().getSizeInBits();

  for (unsigned j = 0; j < nOps; ++j) {
    unsigned i = isBigEndian ? nOps-1-j : j;
    SDValue OpVal = getOperand(i);
    unsigned BitPos = j * EltBitSize;

    if (OpVal.getOpcode() == ISD::UNDEF)
      SplatUndef |= APInt::getBitsSet(sz, BitPos, BitPos + EltBitSize);
    else if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(OpVal))
      SplatValue |= CN->getAPIntValue().zextOrTrunc(EltBitSize).
                    zextOrTrunc(sz) << BitPos;
    else if (ConstantFPSDNode *CN = dyn_cast<ConstantFPSDNode>(OpVal))
      SplatValue |= CN->getValueAPF().bitcastToAPInt().zextOrTrunc(sz) <<BitPos;
     else
      return false;
  }

  // The build_vector is all constants or undefs.  Find the smallest element
  // size that splats the vector.

  HasAnyUndefs = (SplatUndef != 0);
  while (sz > 8) {

    unsigned HalfSize = sz / 2;
    APInt HighValue = SplatValue.lshr(HalfSize).trunc(HalfSize);
    APInt LowValue = SplatValue.trunc(HalfSize);
    APInt HighUndef = SplatUndef.lshr(HalfSize).trunc(HalfSize);
    APInt LowUndef = SplatUndef.trunc(HalfSize);

    // If the two halves do not match (ignoring undef bits), stop here.
    if ((HighValue & ~LowUndef) != (LowValue & ~HighUndef) ||
        MinSplatBits > HalfSize)
      break;

    SplatValue = HighValue | LowValue;
    SplatUndef = HighUndef & LowUndef;

    sz = HalfSize;
  }

  SplatBitSize = sz;
  return true;
}

SDValue BuildVectorSDNode::getSplatValue(BitVector *UndefElements) const {
  if (UndefElements) {
    UndefElements->clear();
    UndefElements->resize(getNumOperands());
  }
  SDValue Splatted;
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i) {
    SDValue Op = getOperand(i);
    if (Op.getOpcode() == ISD::UNDEF) {
      if (UndefElements)
        (*UndefElements)[i] = true;
    } else if (!Splatted) {
      Splatted = Op;
    } else if (Splatted != Op) {
      return SDValue();
    }
  }

  if (!Splatted) {
    assert(getOperand(0).getOpcode() == ISD::UNDEF &&
           "Can only have a splat without a constant for all undefs.");
    return getOperand(0);
  }

  return Splatted;
}

ConstantSDNode *
BuildVectorSDNode::getConstantSplatNode(BitVector *UndefElements) const {
  return dyn_cast_or_null<ConstantSDNode>(getSplatValue(UndefElements));
}

ConstantFPSDNode *
BuildVectorSDNode::getConstantFPSplatNode(BitVector *UndefElements) const {
  return dyn_cast_or_null<ConstantFPSDNode>(getSplatValue(UndefElements));
}

bool BuildVectorSDNode::isConstant() const {
  for (const SDValue &Op : op_values()) {
    unsigned Opc = Op.getOpcode();
    if (Opc != ISD::UNDEF && Opc != ISD::Constant && Opc != ISD::ConstantFP)
      return false;
  }
  return true;
}

bool ShuffleVectorSDNode::isSplatMask(const int *Mask, EVT VT) {
  // Find the first non-undef value in the shuffle mask.
  unsigned i, e;
  for (i = 0, e = VT.getVectorNumElements(); i != e && Mask[i] < 0; ++i)
    /* search */;

  assert(i != e && "VECTOR_SHUFFLE node with all undef indices!");

  // Make sure all remaining elements are either undef or the same as the first
  // non-undef value.
  for (int Idx = Mask[i]; i != e; ++i)
    if (Mask[i] >= 0 && Mask[i] != Idx)
      return false;
  return true;
}

#ifndef NDEBUG
static void checkForCyclesHelper(const SDNode *N,
                                 SmallPtrSetImpl<const SDNode*> &Visited,
                                 SmallPtrSetImpl<const SDNode*> &Checked,
                                 const llvm::SelectionDAG *DAG) {
  // If this node has already been checked, don't check it again.
  if (Checked.count(N))
    return;

  // If a node has already been visited on this depth-first walk, reject it as
  // a cycle.
  if (!Visited.insert(N).second) {
    errs() << "Detected cycle in SelectionDAG\n";
    dbgs() << "Offending node:\n";
    N->dumprFull(DAG); dbgs() << "\n";
    abort();
  }

  for (const SDValue &Op : N->op_values())
    checkForCyclesHelper(Op.getNode(), Visited, Checked, DAG);

  Checked.insert(N);
  Visited.erase(N);
}
#endif

void llvm::checkForCycles(const llvm::SDNode *N,
                          const llvm::SelectionDAG *DAG,
                          bool force) {
#ifndef NDEBUG
  bool check = force;
#ifdef XDEBUG
  check = true;
#endif  // XDEBUG
  if (check) {
    assert(N && "Checking nonexistent SDNode");
    SmallPtrSet<const SDNode*, 32> visited;
    SmallPtrSet<const SDNode*, 32> checked;
    checkForCyclesHelper(N, visited, checked, DAG);
  }
#endif  // !NDEBUG
}

void llvm::checkForCycles(const llvm::SelectionDAG *DAG, bool force) {
  checkForCycles(DAG->getRoot().getNode(), DAG, force);
}
