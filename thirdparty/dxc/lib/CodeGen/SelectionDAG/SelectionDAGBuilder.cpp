//===-- SelectionDAGBuilder.cpp - Selection-DAG building ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements routines for translating from LLVM IR into SelectionDAG IR.
//
//===----------------------------------------------------------------------===//

#include "SelectionDAGBuilder.h"
#include "SDNodeDbgValue.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/FastISel.h"
#include "llvm/CodeGen/FunctionLoweringInfo.h"
#include "llvm/CodeGen/GCMetadata.h"
#include "llvm/CodeGen/GCStrategy.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/StackMaps.h"
#include "llvm/CodeGen/WinEHFuncInfo.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Statepoint.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetIntrinsicInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetSelectionDAGInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include <algorithm>
using namespace llvm;

#define DEBUG_TYPE "isel"

/// LimitFloatPrecision - Generate low-precision inline sequences for
/// some float libcalls (6, 8 or 12 bits).
static unsigned LimitFloatPrecision;

static cl::opt<unsigned, true>
LimitFPPrecision("limit-float-precision",
                 cl::desc("Generate low-precision inline sequences "
                          "for some float libcalls"),
                 cl::location(LimitFloatPrecision),
                 cl::init(0));

static cl::opt<bool>
EnableFMFInDAG("enable-fmf-dag", cl::init(false), cl::Hidden,
                cl::desc("Enable fast-math-flags for DAG nodes"));

// Limit the width of DAG chains. This is important in general to prevent
// DAG-based analysis from blowing up. For example, alias analysis and
// load clustering may not complete in reasonable time. It is difficult to
// recognize and avoid this situation within each individual analysis, and
// future analyses are likely to have the same behavior. Limiting DAG width is
// the safe approach and will be especially important with global DAGs.
//
// MaxParallelChains default is arbitrarily high to avoid affecting
// optimization, but could be lowered to improve compile time. Any ld-ld-st-st
// sequence over this should have been converted to llvm.memcpy by the
// frontend. It easy to induce this behavior with .ll code such as:
// %buffer = alloca [4096 x i8]
// %data = load [4096 x i8]* %argPtr
// store [4096 x i8] %data, [4096 x i8]* %buffer
static const unsigned MaxParallelChains = 64;

static SDValue getCopyFromPartsVector(SelectionDAG &DAG, SDLoc DL,
                                      const SDValue *Parts, unsigned NumParts,
                                      MVT PartVT, EVT ValueVT, const Value *V);

/// getCopyFromParts - Create a value that contains the specified legal parts
/// combined into the value they represent.  If the parts combine to a type
/// larger then ValueVT then AssertOp can be used to specify whether the extra
/// bits are known to be zero (ISD::AssertZext) or sign extended from ValueVT
/// (ISD::AssertSext).
static SDValue getCopyFromParts(SelectionDAG &DAG, SDLoc DL,
                                const SDValue *Parts,
                                unsigned NumParts, MVT PartVT, EVT ValueVT,
                                const Value *V,
                                ISD::NodeType AssertOp = ISD::DELETED_NODE) {
  if (ValueVT.isVector())
    return getCopyFromPartsVector(DAG, DL, Parts, NumParts,
                                  PartVT, ValueVT, V);

  assert(NumParts > 0 && "No parts to assemble!");
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  SDValue Val = Parts[0];

  if (NumParts > 1) {
    // Assemble the value from multiple parts.
    if (ValueVT.isInteger()) {
      unsigned PartBits = PartVT.getSizeInBits();
      unsigned ValueBits = ValueVT.getSizeInBits();

      // Assemble the power of 2 part.
      unsigned RoundParts = NumParts & (NumParts - 1) ?
        1 << Log2_32(NumParts) : NumParts;
      unsigned RoundBits = PartBits * RoundParts;
      EVT RoundVT = RoundBits == ValueBits ?
        ValueVT : EVT::getIntegerVT(*DAG.getContext(), RoundBits);
      SDValue Lo, Hi;

      EVT HalfVT = EVT::getIntegerVT(*DAG.getContext(), RoundBits/2);

      if (RoundParts > 2) {
        Lo = getCopyFromParts(DAG, DL, Parts, RoundParts / 2,
                              PartVT, HalfVT, V);
        Hi = getCopyFromParts(DAG, DL, Parts + RoundParts / 2,
                              RoundParts / 2, PartVT, HalfVT, V);
      } else {
        Lo = DAG.getNode(ISD::BITCAST, DL, HalfVT, Parts[0]);
        Hi = DAG.getNode(ISD::BITCAST, DL, HalfVT, Parts[1]);
      }

      if (DAG.getDataLayout().isBigEndian())
        std::swap(Lo, Hi);

      Val = DAG.getNode(ISD::BUILD_PAIR, DL, RoundVT, Lo, Hi);

      if (RoundParts < NumParts) {
        // Assemble the trailing non-power-of-2 part.
        unsigned OddParts = NumParts - RoundParts;
        EVT OddVT = EVT::getIntegerVT(*DAG.getContext(), OddParts * PartBits);
        Hi = getCopyFromParts(DAG, DL,
                              Parts + RoundParts, OddParts, PartVT, OddVT, V);

        // Combine the round and odd parts.
        Lo = Val;
        if (DAG.getDataLayout().isBigEndian())
          std::swap(Lo, Hi);
        EVT TotalVT = EVT::getIntegerVT(*DAG.getContext(), NumParts * PartBits);
        Hi = DAG.getNode(ISD::ANY_EXTEND, DL, TotalVT, Hi);
        Hi =
            DAG.getNode(ISD::SHL, DL, TotalVT, Hi,
                        DAG.getConstant(Lo.getValueType().getSizeInBits(), DL,
                                        TLI.getPointerTy(DAG.getDataLayout())));
        Lo = DAG.getNode(ISD::ZERO_EXTEND, DL, TotalVT, Lo);
        Val = DAG.getNode(ISD::OR, DL, TotalVT, Lo, Hi);
      }
    } else if (PartVT.isFloatingPoint()) {
      // FP split into multiple FP parts (for ppcf128)
      assert(ValueVT == EVT(MVT::ppcf128) && PartVT == MVT::f64 &&
             "Unexpected split");
      SDValue Lo, Hi;
      Lo = DAG.getNode(ISD::BITCAST, DL, EVT(MVT::f64), Parts[0]);
      Hi = DAG.getNode(ISD::BITCAST, DL, EVT(MVT::f64), Parts[1]);
      if (TLI.hasBigEndianPartOrdering(ValueVT, DAG.getDataLayout()))
        std::swap(Lo, Hi);
      Val = DAG.getNode(ISD::BUILD_PAIR, DL, ValueVT, Lo, Hi);
    } else {
      // FP split into integer parts (soft fp)
      assert(ValueVT.isFloatingPoint() && PartVT.isInteger() &&
             !PartVT.isVector() && "Unexpected split");
      EVT IntVT = EVT::getIntegerVT(*DAG.getContext(), ValueVT.getSizeInBits());
      Val = getCopyFromParts(DAG, DL, Parts, NumParts, PartVT, IntVT, V);
    }
  }

  // There is now one part, held in Val.  Correct it to match ValueVT.
  EVT PartEVT = Val.getValueType();

  if (PartEVT == ValueVT)
    return Val;

  if (PartEVT.isInteger() && ValueVT.isInteger()) {
    if (ValueVT.bitsLT(PartEVT)) {
      // For a truncate, see if we have any information to
      // indicate whether the truncated bits will always be
      // zero or sign-extension.
      if (AssertOp != ISD::DELETED_NODE)
        Val = DAG.getNode(AssertOp, DL, PartEVT, Val,
                          DAG.getValueType(ValueVT));
      return DAG.getNode(ISD::TRUNCATE, DL, ValueVT, Val);
    }
    return DAG.getNode(ISD::ANY_EXTEND, DL, ValueVT, Val);
  }

  if (PartEVT.isFloatingPoint() && ValueVT.isFloatingPoint()) {
    // FP_ROUND's are always exact here.
    if (ValueVT.bitsLT(Val.getValueType()))
      return DAG.getNode(
          ISD::FP_ROUND, DL, ValueVT, Val,
          DAG.getTargetConstant(1, DL, TLI.getPointerTy(DAG.getDataLayout())));

    return DAG.getNode(ISD::FP_EXTEND, DL, ValueVT, Val);
  }

  if (PartEVT.getSizeInBits() == ValueVT.getSizeInBits())
    return DAG.getNode(ISD::BITCAST, DL, ValueVT, Val);

  llvm_unreachable("Unknown mismatch!");
}

static void diagnosePossiblyInvalidConstraint(LLVMContext &Ctx, const Value *V,
                                              const Twine &ErrMsg) {
  const Instruction *I = dyn_cast_or_null<Instruction>(V);
  if (!V)
    return Ctx.emitError(ErrMsg);

  const char *AsmError = ", possible invalid constraint for vector type";
  if (const CallInst *CI = dyn_cast<CallInst>(I))
    if (isa<InlineAsm>(CI->getCalledValue()))
      return Ctx.emitError(I, ErrMsg + AsmError);

  return Ctx.emitError(I, ErrMsg);
}

/// getCopyFromPartsVector - Create a value that contains the specified legal
/// parts combined into the value they represent.  If the parts combine to a
/// type larger then ValueVT then AssertOp can be used to specify whether the
/// extra bits are known to be zero (ISD::AssertZext) or sign extended from
/// ValueVT (ISD::AssertSext).
static SDValue getCopyFromPartsVector(SelectionDAG &DAG, SDLoc DL,
                                      const SDValue *Parts, unsigned NumParts,
                                      MVT PartVT, EVT ValueVT, const Value *V) {
  assert(ValueVT.isVector() && "Not a vector value");
  assert(NumParts > 0 && "No parts to assemble!");
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  SDValue Val = Parts[0];

  // Handle a multi-element vector.
  if (NumParts > 1) {
    EVT IntermediateVT;
    MVT RegisterVT;
    unsigned NumIntermediates;
    unsigned NumRegs =
    TLI.getVectorTypeBreakdown(*DAG.getContext(), ValueVT, IntermediateVT,
                               NumIntermediates, RegisterVT);
    assert(NumRegs == NumParts && "Part count doesn't match vector breakdown!");
    NumParts = NumRegs; // Silence a compiler warning.
    assert(RegisterVT == PartVT && "Part type doesn't match vector breakdown!");
    assert(RegisterVT.getSizeInBits() ==
           Parts[0].getSimpleValueType().getSizeInBits() &&
           "Part type sizes don't match!");

    // Assemble the parts into intermediate operands.
    SmallVector<SDValue, 8> Ops(NumIntermediates);
    if (NumIntermediates == NumParts) {
      // If the register was not expanded, truncate or copy the value,
      // as appropriate.
      for (unsigned i = 0; i != NumParts; ++i)
        Ops[i] = getCopyFromParts(DAG, DL, &Parts[i], 1,
                                  PartVT, IntermediateVT, V);
    } else if (NumParts > 0) {
      // If the intermediate type was expanded, build the intermediate
      // operands from the parts.
      assert(NumParts % NumIntermediates == 0 &&
             "Must expand into a divisible number of parts!");
      unsigned Factor = NumParts / NumIntermediates;
      for (unsigned i = 0; i != NumIntermediates; ++i)
        Ops[i] = getCopyFromParts(DAG, DL, &Parts[i * Factor], Factor,
                                  PartVT, IntermediateVT, V);
    }

    // Build a vector with BUILD_VECTOR or CONCAT_VECTORS from the
    // intermediate operands.
    Val = DAG.getNode(IntermediateVT.isVector() ? ISD::CONCAT_VECTORS
                                                : ISD::BUILD_VECTOR,
                      DL, ValueVT, Ops);
  }

  // There is now one part, held in Val.  Correct it to match ValueVT.
  EVT PartEVT = Val.getValueType();

  if (PartEVT == ValueVT)
    return Val;

  if (PartEVT.isVector()) {
    // If the element type of the source/dest vectors are the same, but the
    // parts vector has more elements than the value vector, then we have a
    // vector widening case (e.g. <2 x float> -> <4 x float>).  Extract the
    // elements we want.
    if (PartEVT.getVectorElementType() == ValueVT.getVectorElementType()) {
      assert(PartEVT.getVectorNumElements() > ValueVT.getVectorNumElements() &&
             "Cannot narrow, it would be a lossy transformation");
      return DAG.getNode(
          ISD::EXTRACT_SUBVECTOR, DL, ValueVT, Val,
          DAG.getConstant(0, DL, TLI.getVectorIdxTy(DAG.getDataLayout())));
    }

    // Vector/Vector bitcast.
    if (ValueVT.getSizeInBits() == PartEVT.getSizeInBits())
      return DAG.getNode(ISD::BITCAST, DL, ValueVT, Val);

    assert(PartEVT.getVectorNumElements() == ValueVT.getVectorNumElements() &&
      "Cannot handle this kind of promotion");
    // Promoted vector extract
    bool Smaller = ValueVT.bitsLE(PartEVT);
    return DAG.getNode((Smaller ? ISD::TRUNCATE : ISD::ANY_EXTEND),
                       DL, ValueVT, Val);

  }

  // Trivial bitcast if the types are the same size and the destination
  // vector type is legal.
  if (PartEVT.getSizeInBits() == ValueVT.getSizeInBits() &&
      TLI.isTypeLegal(ValueVT))
    return DAG.getNode(ISD::BITCAST, DL, ValueVT, Val);

  // Handle cases such as i8 -> <1 x i1>
  if (ValueVT.getVectorNumElements() != 1) {
    diagnosePossiblyInvalidConstraint(*DAG.getContext(), V,
                                      "non-trivial scalar-to-vector conversion");
    return DAG.getUNDEF(ValueVT);
  }

  if (ValueVT.getVectorNumElements() == 1 &&
      ValueVT.getVectorElementType() != PartEVT) {
    bool Smaller = ValueVT.bitsLE(PartEVT);
    Val = DAG.getNode((Smaller ? ISD::TRUNCATE : ISD::ANY_EXTEND),
                       DL, ValueVT.getScalarType(), Val);
  }

  return DAG.getNode(ISD::BUILD_VECTOR, DL, ValueVT, Val);
}

static void getCopyToPartsVector(SelectionDAG &DAG, SDLoc dl,
                                 SDValue Val, SDValue *Parts, unsigned NumParts,
                                 MVT PartVT, const Value *V);

/// getCopyToParts - Create a series of nodes that contain the specified value
/// split into legal parts.  If the parts contain more bits than Val, then, for
/// integers, ExtendKind can be used to specify how to generate the extra bits.
static void getCopyToParts(SelectionDAG &DAG, SDLoc DL,
                           SDValue Val, SDValue *Parts, unsigned NumParts,
                           MVT PartVT, const Value *V,
                           ISD::NodeType ExtendKind = ISD::ANY_EXTEND) {
  EVT ValueVT = Val.getValueType();

  // Handle the vector case separately.
  if (ValueVT.isVector())
    return getCopyToPartsVector(DAG, DL, Val, Parts, NumParts, PartVT, V);

  unsigned PartBits = PartVT.getSizeInBits();
  unsigned OrigNumParts = NumParts;
  assert(DAG.getTargetLoweringInfo().isTypeLegal(PartVT) &&
         "Copying to an illegal type!");

  if (NumParts == 0)
    return;

  assert(!ValueVT.isVector() && "Vector case handled elsewhere");
  EVT PartEVT = PartVT;
  if (PartEVT == ValueVT) {
    assert(NumParts == 1 && "No-op copy with multiple parts!");
    Parts[0] = Val;
    return;
  }

  if (NumParts * PartBits > ValueVT.getSizeInBits()) {
    // If the parts cover more bits than the value has, promote the value.
    if (PartVT.isFloatingPoint() && ValueVT.isFloatingPoint()) {
      assert(NumParts == 1 && "Do not know what to promote to!");
      Val = DAG.getNode(ISD::FP_EXTEND, DL, PartVT, Val);
    } else {
      assert((PartVT.isInteger() || PartVT == MVT::x86mmx) &&
             ValueVT.isInteger() &&
             "Unknown mismatch!");
      ValueVT = EVT::getIntegerVT(*DAG.getContext(), NumParts * PartBits);
      Val = DAG.getNode(ExtendKind, DL, ValueVT, Val);
      if (PartVT == MVT::x86mmx)
        Val = DAG.getNode(ISD::BITCAST, DL, PartVT, Val);
    }
  } else if (PartBits == ValueVT.getSizeInBits()) {
    // Different types of the same size.
    assert(NumParts == 1 && PartEVT != ValueVT);
    Val = DAG.getNode(ISD::BITCAST, DL, PartVT, Val);
  } else if (NumParts * PartBits < ValueVT.getSizeInBits()) {
    // If the parts cover less bits than value has, truncate the value.
    assert((PartVT.isInteger() || PartVT == MVT::x86mmx) &&
           ValueVT.isInteger() &&
           "Unknown mismatch!");
    ValueVT = EVT::getIntegerVT(*DAG.getContext(), NumParts * PartBits);
    Val = DAG.getNode(ISD::TRUNCATE, DL, ValueVT, Val);
    if (PartVT == MVT::x86mmx)
      Val = DAG.getNode(ISD::BITCAST, DL, PartVT, Val);
  }

  // The value may have changed - recompute ValueVT.
  ValueVT = Val.getValueType();
  assert(NumParts * PartBits == ValueVT.getSizeInBits() &&
         "Failed to tile the value with PartVT!");

  if (NumParts == 1) {
    if (PartEVT != ValueVT)
      diagnosePossiblyInvalidConstraint(*DAG.getContext(), V,
                                        "scalar-to-vector conversion failed");

    Parts[0] = Val;
    return;
  }

  // Expand the value into multiple parts.
  if (NumParts & (NumParts - 1)) {
    // The number of parts is not a power of 2.  Split off and copy the tail.
    assert(PartVT.isInteger() && ValueVT.isInteger() &&
           "Do not know what to expand to!");
    unsigned RoundParts = 1 << Log2_32(NumParts);
    unsigned RoundBits = RoundParts * PartBits;
    unsigned OddParts = NumParts - RoundParts;
    SDValue OddVal = DAG.getNode(ISD::SRL, DL, ValueVT, Val,
                                 DAG.getIntPtrConstant(RoundBits, DL));
    getCopyToParts(DAG, DL, OddVal, Parts + RoundParts, OddParts, PartVT, V);

    if (DAG.getDataLayout().isBigEndian())
      // The odd parts were reversed by getCopyToParts - unreverse them.
      std::reverse(Parts + RoundParts, Parts + NumParts);

    NumParts = RoundParts;
    ValueVT = EVT::getIntegerVT(*DAG.getContext(), NumParts * PartBits);
    Val = DAG.getNode(ISD::TRUNCATE, DL, ValueVT, Val);
  }

  // The number of parts is a power of 2.  Repeatedly bisect the value using
  // EXTRACT_ELEMENT.
  Parts[0] = DAG.getNode(ISD::BITCAST, DL,
                         EVT::getIntegerVT(*DAG.getContext(),
                                           ValueVT.getSizeInBits()),
                         Val);

  for (unsigned StepSize = NumParts; StepSize > 1; StepSize /= 2) {
    for (unsigned i = 0; i < NumParts; i += StepSize) {
      unsigned ThisBits = StepSize * PartBits / 2;
      EVT ThisVT = EVT::getIntegerVT(*DAG.getContext(), ThisBits);
      SDValue &Part0 = Parts[i];
      SDValue &Part1 = Parts[i+StepSize/2];

      Part1 = DAG.getNode(ISD::EXTRACT_ELEMENT, DL,
                          ThisVT, Part0, DAG.getIntPtrConstant(1, DL));
      Part0 = DAG.getNode(ISD::EXTRACT_ELEMENT, DL,
                          ThisVT, Part0, DAG.getIntPtrConstant(0, DL));

      if (ThisBits == PartBits && ThisVT != PartVT) {
        Part0 = DAG.getNode(ISD::BITCAST, DL, PartVT, Part0);
        Part1 = DAG.getNode(ISD::BITCAST, DL, PartVT, Part1);
      }
    }
  }

  if (DAG.getDataLayout().isBigEndian())
    std::reverse(Parts, Parts + OrigNumParts);
}


/// getCopyToPartsVector - Create a series of nodes that contain the specified
/// value split into legal parts.
static void getCopyToPartsVector(SelectionDAG &DAG, SDLoc DL,
                                 SDValue Val, SDValue *Parts, unsigned NumParts,
                                 MVT PartVT, const Value *V) {
  EVT ValueVT = Val.getValueType();
  assert(ValueVT.isVector() && "Not a vector");
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();

  if (NumParts == 1) {
    EVT PartEVT = PartVT;
    if (PartEVT == ValueVT) {
      // Nothing to do.
    } else if (PartVT.getSizeInBits() == ValueVT.getSizeInBits()) {
      // Bitconvert vector->vector case.
      Val = DAG.getNode(ISD::BITCAST, DL, PartVT, Val);
    } else if (PartVT.isVector() &&
               PartEVT.getVectorElementType() == ValueVT.getVectorElementType() &&
               PartEVT.getVectorNumElements() > ValueVT.getVectorNumElements()) {
      EVT ElementVT = PartVT.getVectorElementType();
      // Vector widening case, e.g. <2 x float> -> <4 x float>.  Shuffle in
      // undef elements.
      SmallVector<SDValue, 16> Ops;
      for (unsigned i = 0, e = ValueVT.getVectorNumElements(); i != e; ++i)
        Ops.push_back(DAG.getNode(
            ISD::EXTRACT_VECTOR_ELT, DL, ElementVT, Val,
            DAG.getConstant(i, DL, TLI.getVectorIdxTy(DAG.getDataLayout()))));

      for (unsigned i = ValueVT.getVectorNumElements(),
           e = PartVT.getVectorNumElements(); i != e; ++i)
        Ops.push_back(DAG.getUNDEF(ElementVT));

      Val = DAG.getNode(ISD::BUILD_VECTOR, DL, PartVT, Ops);

      // FIXME: Use CONCAT for 2x -> 4x.

      //SDValue UndefElts = DAG.getUNDEF(VectorTy);
      //Val = DAG.getNode(ISD::CONCAT_VECTORS, DL, PartVT, Val, UndefElts);
    } else if (PartVT.isVector() &&
               PartEVT.getVectorElementType().bitsGE(
                 ValueVT.getVectorElementType()) &&
               PartEVT.getVectorNumElements() == ValueVT.getVectorNumElements()) {

      // Promoted vector extract
      bool Smaller = PartEVT.bitsLE(ValueVT);
      Val = DAG.getNode((Smaller ? ISD::TRUNCATE : ISD::ANY_EXTEND),
                        DL, PartVT, Val);
    } else{
      // Vector -> scalar conversion.
      assert(ValueVT.getVectorNumElements() == 1 &&
             "Only trivial vector-to-scalar conversions should get here!");
      Val = DAG.getNode(
          ISD::EXTRACT_VECTOR_ELT, DL, PartVT, Val,
          DAG.getConstant(0, DL, TLI.getVectorIdxTy(DAG.getDataLayout())));

      bool Smaller = ValueVT.bitsLE(PartVT);
      Val = DAG.getNode((Smaller ? ISD::TRUNCATE : ISD::ANY_EXTEND),
                         DL, PartVT, Val);
    }

    Parts[0] = Val;
    return;
  }

  // Handle a multi-element vector.
  EVT IntermediateVT;
  MVT RegisterVT;
  unsigned NumIntermediates;
  unsigned NumRegs = TLI.getVectorTypeBreakdown(*DAG.getContext(), ValueVT,
                                                IntermediateVT,
                                                NumIntermediates, RegisterVT);
  unsigned NumElements = ValueVT.getVectorNumElements();

  assert(NumRegs == NumParts && "Part count doesn't match vector breakdown!");
  NumParts = NumRegs; // Silence a compiler warning.
  assert(RegisterVT == PartVT && "Part type doesn't match vector breakdown!");

  // Split the vector into intermediate operands.
  SmallVector<SDValue, 8> Ops(NumIntermediates);
  for (unsigned i = 0; i != NumIntermediates; ++i) {
    if (IntermediateVT.isVector())
      Ops[i] =
          DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, IntermediateVT, Val,
                      DAG.getConstant(i * (NumElements / NumIntermediates), DL,
                                      TLI.getVectorIdxTy(DAG.getDataLayout())));
    else
      Ops[i] = DAG.getNode(
          ISD::EXTRACT_VECTOR_ELT, DL, IntermediateVT, Val,
          DAG.getConstant(i, DL, TLI.getVectorIdxTy(DAG.getDataLayout())));
  }

  // Split the intermediate operands into legal parts.
  if (NumParts == NumIntermediates) {
    // If the register was not expanded, promote or copy the value,
    // as appropriate.
    for (unsigned i = 0; i != NumParts; ++i)
      getCopyToParts(DAG, DL, Ops[i], &Parts[i], 1, PartVT, V);
  } else if (NumParts > 0) {
    // If the intermediate type was expanded, split each the value into
    // legal parts.
    assert(NumIntermediates != 0 && "division by zero");
    assert(NumParts % NumIntermediates == 0 &&
           "Must expand into a divisible number of parts!");
    unsigned Factor = NumParts / NumIntermediates;
    for (unsigned i = 0; i != NumIntermediates; ++i)
      getCopyToParts(DAG, DL, Ops[i], &Parts[i*Factor], Factor, PartVT, V);
  }
}

RegsForValue::RegsForValue() {}

RegsForValue::RegsForValue(const SmallVector<unsigned, 4> &regs, MVT regvt,
                           EVT valuevt)
    : ValueVTs(1, valuevt), RegVTs(1, regvt), Regs(regs) {}

RegsForValue::RegsForValue(LLVMContext &Context, const TargetLowering &TLI,
                           const DataLayout &DL, unsigned Reg, Type *Ty) {
  ComputeValueVTs(TLI, DL, Ty, ValueVTs);

  for (unsigned Value = 0, e = ValueVTs.size(); Value != e; ++Value) {
    EVT ValueVT = ValueVTs[Value];
    unsigned NumRegs = TLI.getNumRegisters(Context, ValueVT);
    MVT RegisterVT = TLI.getRegisterType(Context, ValueVT);
    for (unsigned i = 0; i != NumRegs; ++i)
      Regs.push_back(Reg + i);
    RegVTs.push_back(RegisterVT);
    Reg += NumRegs;
  }
}

/// getCopyFromRegs - Emit a series of CopyFromReg nodes that copies from
/// this value and returns the result as a ValueVT value.  This uses
/// Chain/Flag as the input and updates them for the output Chain/Flag.
/// If the Flag pointer is NULL, no flag is used.
SDValue RegsForValue::getCopyFromRegs(SelectionDAG &DAG,
                                      FunctionLoweringInfo &FuncInfo,
                                      SDLoc dl,
                                      SDValue &Chain, SDValue *Flag,
                                      const Value *V) const {
  // A Value with type {} or [0 x %t] needs no registers.
  if (ValueVTs.empty())
    return SDValue();

  const TargetLowering &TLI = DAG.getTargetLoweringInfo();

  // Assemble the legal parts into the final values.
  SmallVector<SDValue, 4> Values(ValueVTs.size());
  SmallVector<SDValue, 8> Parts;
  for (unsigned Value = 0, Part = 0, e = ValueVTs.size(); Value != e; ++Value) {
    // Copy the legal parts from the registers.
    EVT ValueVT = ValueVTs[Value];
    unsigned NumRegs = TLI.getNumRegisters(*DAG.getContext(), ValueVT);
    MVT RegisterVT = RegVTs[Value];

    Parts.resize(NumRegs);
    for (unsigned i = 0; i != NumRegs; ++i) {
      SDValue P;
      if (!Flag) {
        P = DAG.getCopyFromReg(Chain, dl, Regs[Part+i], RegisterVT);
      } else {
        P = DAG.getCopyFromReg(Chain, dl, Regs[Part+i], RegisterVT, *Flag);
        *Flag = P.getValue(2);
      }

      Chain = P.getValue(1);
      Parts[i] = P;

      // If the source register was virtual and if we know something about it,
      // add an assert node.
      if (!TargetRegisterInfo::isVirtualRegister(Regs[Part+i]) ||
          !RegisterVT.isInteger() || RegisterVT.isVector())
        continue;

      const FunctionLoweringInfo::LiveOutInfo *LOI =
        FuncInfo.GetLiveOutRegInfo(Regs[Part+i]);
      if (!LOI)
        continue;

      unsigned RegSize = RegisterVT.getSizeInBits();
      unsigned NumSignBits = LOI->NumSignBits;
      unsigned NumZeroBits = LOI->KnownZero.countLeadingOnes();

      if (NumZeroBits == RegSize) {
        // The current value is a zero.
        // Explicitly express that as it would be easier for
        // optimizations to kick in.
        Parts[i] = DAG.getConstant(0, dl, RegisterVT);
        continue;
      }

      // FIXME: We capture more information than the dag can represent.  For
      // now, just use the tightest assertzext/assertsext possible.
      bool isSExt = true;
      EVT FromVT(MVT::Other);
      if (NumSignBits == RegSize)
        isSExt = true, FromVT = MVT::i1;   // ASSERT SEXT 1
      else if (NumZeroBits >= RegSize-1)
        isSExt = false, FromVT = MVT::i1;  // ASSERT ZEXT 1
      else if (NumSignBits > RegSize-8)
        isSExt = true, FromVT = MVT::i8;   // ASSERT SEXT 8
      else if (NumZeroBits >= RegSize-8)
        isSExt = false, FromVT = MVT::i8;  // ASSERT ZEXT 8
      else if (NumSignBits > RegSize-16)
        isSExt = true, FromVT = MVT::i16;  // ASSERT SEXT 16
      else if (NumZeroBits >= RegSize-16)
        isSExt = false, FromVT = MVT::i16; // ASSERT ZEXT 16
      else if (NumSignBits > RegSize-32)
        isSExt = true, FromVT = MVT::i32;  // ASSERT SEXT 32
      else if (NumZeroBits >= RegSize-32)
        isSExt = false, FromVT = MVT::i32; // ASSERT ZEXT 32
      else
        continue;

      // Add an assertion node.
      assert(FromVT != MVT::Other);
      Parts[i] = DAG.getNode(isSExt ? ISD::AssertSext : ISD::AssertZext, dl,
                             RegisterVT, P, DAG.getValueType(FromVT));
    }

    Values[Value] = getCopyFromParts(DAG, dl, Parts.begin(),
                                     NumRegs, RegisterVT, ValueVT, V);
    Part += NumRegs;
    Parts.clear();
  }

  return DAG.getNode(ISD::MERGE_VALUES, dl, DAG.getVTList(ValueVTs), Values);
}

/// getCopyToRegs - Emit a series of CopyToReg nodes that copies the
/// specified value into the registers specified by this object.  This uses
/// Chain/Flag as the input and updates them for the output Chain/Flag.
/// If the Flag pointer is NULL, no flag is used.
void RegsForValue::getCopyToRegs(SDValue Val, SelectionDAG &DAG, SDLoc dl,
                                 SDValue &Chain, SDValue *Flag, const Value *V,
                                 ISD::NodeType PreferredExtendType) const {
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  ISD::NodeType ExtendKind = PreferredExtendType;

  // Get the list of the values's legal parts.
  unsigned NumRegs = Regs.size();
  SmallVector<SDValue, 8> Parts(NumRegs);
  for (unsigned Value = 0, Part = 0, e = ValueVTs.size(); Value != e; ++Value) {
    EVT ValueVT = ValueVTs[Value];
    unsigned NumParts = TLI.getNumRegisters(*DAG.getContext(), ValueVT);
    MVT RegisterVT = RegVTs[Value];

    if (ExtendKind == ISD::ANY_EXTEND && TLI.isZExtFree(Val, RegisterVT))
      ExtendKind = ISD::ZERO_EXTEND;

    getCopyToParts(DAG, dl, Val.getValue(Val.getResNo() + Value),
                   &Parts[Part], NumParts, RegisterVT, V, ExtendKind);
    Part += NumParts;
  }

  // Copy the parts into the registers.
  SmallVector<SDValue, 8> Chains(NumRegs);
  for (unsigned i = 0; i != NumRegs; ++i) {
    SDValue Part;
    if (!Flag) {
      Part = DAG.getCopyToReg(Chain, dl, Regs[i], Parts[i]);
    } else {
      Part = DAG.getCopyToReg(Chain, dl, Regs[i], Parts[i], *Flag);
      *Flag = Part.getValue(1);
    }

    Chains[i] = Part.getValue(0);
  }

  if (NumRegs == 1 || Flag)
    // If NumRegs > 1 && Flag is used then the use of the last CopyToReg is
    // flagged to it. That is the CopyToReg nodes and the user are considered
    // a single scheduling unit. If we create a TokenFactor and return it as
    // chain, then the TokenFactor is both a predecessor (operand) of the
    // user as well as a successor (the TF operands are flagged to the user).
    // c1, f1 = CopyToReg
    // c2, f2 = CopyToReg
    // c3     = TokenFactor c1, c2
    // ...
    //        = op c3, ..., f2
    Chain = Chains[NumRegs-1];
  else
    Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Chains);
}

/// AddInlineAsmOperands - Add this value to the specified inlineasm node
/// operand list.  This adds the code marker and includes the number of
/// values added into it.
void RegsForValue::AddInlineAsmOperands(unsigned Code, bool HasMatching,
                                        unsigned MatchingIdx, SDLoc dl,
                                        SelectionDAG &DAG,
                                        std::vector<SDValue> &Ops) const {
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();

  unsigned Flag = InlineAsm::getFlagWord(Code, Regs.size());
  if (HasMatching)
    Flag = InlineAsm::getFlagWordForMatchingOp(Flag, MatchingIdx);
  else if (!Regs.empty() &&
           TargetRegisterInfo::isVirtualRegister(Regs.front())) {
    // Put the register class of the virtual registers in the flag word.  That
    // way, later passes can recompute register class constraints for inline
    // assembly as well as normal instructions.
    // Don't do this for tied operands that can use the regclass information
    // from the def.
    const MachineRegisterInfo &MRI = DAG.getMachineFunction().getRegInfo();
    const TargetRegisterClass *RC = MRI.getRegClass(Regs.front());
    Flag = InlineAsm::getFlagWordForRegClass(Flag, RC->getID());
  }

  SDValue Res = DAG.getTargetConstant(Flag, dl, MVT::i32);
  Ops.push_back(Res);

  unsigned SP = TLI.getStackPointerRegisterToSaveRestore();
  for (unsigned Value = 0, Reg = 0, e = ValueVTs.size(); Value != e; ++Value) {
    unsigned NumRegs = TLI.getNumRegisters(*DAG.getContext(), ValueVTs[Value]);
    MVT RegisterVT = RegVTs[Value];
    for (unsigned i = 0; i != NumRegs; ++i) {
      assert(Reg < Regs.size() && "Mismatch in # registers expected");
      unsigned TheReg = Regs[Reg++];
      Ops.push_back(DAG.getRegister(TheReg, RegisterVT));

      if (TheReg == SP && Code == InlineAsm::Kind_Clobber) {
        // If we clobbered the stack pointer, MFI should know about it.
        assert(DAG.getMachineFunction().getFrameInfo()->
            hasOpaqueSPAdjustment());
      }
    }
  }
}

void SelectionDAGBuilder::init(GCFunctionInfo *gfi, AliasAnalysis &aa,
                               const TargetLibraryInfo *li) {
  AA = &aa;
  GFI = gfi;
  LibInfo = li;
  DL = &DAG.getDataLayout();
  Context = DAG.getContext();
  LPadToCallSiteMap.clear();
}

/// clear - Clear out the current SelectionDAG and the associated
/// state and prepare this SelectionDAGBuilder object to be used
/// for a new block. This doesn't clear out information about
/// additional blocks that are needed to complete switch lowering
/// or PHI node updating; that information is cleared out as it is
/// consumed.
void SelectionDAGBuilder::clear() {
  NodeMap.clear();
  UnusedArgNodeMap.clear();
  PendingLoads.clear();
  PendingExports.clear();
  CurInst = nullptr;
  HasTailCall = false;
  SDNodeOrder = LowestSDNodeOrder;
  StatepointLowering.clear();
}

/// clearDanglingDebugInfo - Clear the dangling debug information
/// map. This function is separated from the clear so that debug
/// information that is dangling in a basic block can be properly
/// resolved in a different basic block. This allows the
/// SelectionDAG to resolve dangling debug information attached
/// to PHI nodes.
void SelectionDAGBuilder::clearDanglingDebugInfo() {
  DanglingDebugInfoMap.clear();
}

/// getRoot - Return the current virtual root of the Selection DAG,
/// flushing any PendingLoad items. This must be done before emitting
/// a store or any other node that may need to be ordered after any
/// prior load instructions.
///
SDValue SelectionDAGBuilder::getRoot() {
  if (PendingLoads.empty())
    return DAG.getRoot();

  if (PendingLoads.size() == 1) {
    SDValue Root = PendingLoads[0];
    DAG.setRoot(Root);
    PendingLoads.clear();
    return Root;
  }

  // Otherwise, we have to make a token factor node.
  SDValue Root = DAG.getNode(ISD::TokenFactor, getCurSDLoc(), MVT::Other,
                             PendingLoads);
  PendingLoads.clear();
  DAG.setRoot(Root);
  return Root;
}

/// getControlRoot - Similar to getRoot, but instead of flushing all the
/// PendingLoad items, flush all the PendingExports items. It is necessary
/// to do this before emitting a terminator instruction.
///
SDValue SelectionDAGBuilder::getControlRoot() {
  SDValue Root = DAG.getRoot();

  if (PendingExports.empty())
    return Root;

  // Turn all of the CopyToReg chains into one factored node.
  if (Root.getOpcode() != ISD::EntryToken) {
    unsigned i = 0, e = PendingExports.size();
    for (; i != e; ++i) {
      assert(PendingExports[i].getNode()->getNumOperands() > 1);
      if (PendingExports[i].getNode()->getOperand(0) == Root)
        break;  // Don't add the root if we already indirectly depend on it.
    }

    if (i == e)
      PendingExports.push_back(Root);
  }

  Root = DAG.getNode(ISD::TokenFactor, getCurSDLoc(), MVT::Other,
                     PendingExports);
  PendingExports.clear();
  DAG.setRoot(Root);
  return Root;
}

void SelectionDAGBuilder::visit(const Instruction &I) {
  // Set up outgoing PHI node register values before emitting the terminator.
  if (isa<TerminatorInst>(&I))
    HandlePHINodesInSuccessorBlocks(I.getParent());

  ++SDNodeOrder;

  CurInst = &I;

  visit(I.getOpcode(), I);

  if (!isa<TerminatorInst>(&I) && !HasTailCall)
    CopyToExportRegsIfNeeded(&I);

  CurInst = nullptr;
}

void SelectionDAGBuilder::visitPHI(const PHINode &) {
  llvm_unreachable("SelectionDAGBuilder shouldn't visit PHI nodes!");
}

void SelectionDAGBuilder::visit(unsigned Opcode, const User &I) {
  // Note: this doesn't use InstVisitor, because it has to work with
  // ConstantExpr's in addition to instructions.
  switch (Opcode) {
  default: llvm_unreachable("Unknown instruction type encountered!");
    // Build the switch statement using the Instruction.def file.
#define HANDLE_INST(NUM, OPCODE, CLASS) \
    case Instruction::OPCODE: visit##OPCODE((const CLASS&)I); break;
#include "llvm/IR/Instruction.def"
  }
}

// resolveDanglingDebugInfo - if we saw an earlier dbg_value referring to V,
// generate the debug data structures now that we've seen its definition.
void SelectionDAGBuilder::resolveDanglingDebugInfo(const Value *V,
                                                   SDValue Val) {
  DanglingDebugInfo &DDI = DanglingDebugInfoMap[V];
  if (DDI.getDI()) {
    const DbgValueInst *DI = DDI.getDI();
    DebugLoc dl = DDI.getdl();
    unsigned DbgSDNodeOrder = DDI.getSDNodeOrder();
    DILocalVariable *Variable = DI->getVariable();
    DIExpression *Expr = DI->getExpression();
    assert(Variable->isValidLocationForIntrinsic(dl) &&
           "Expected inlined-at fields to agree");
    uint64_t Offset = DI->getOffset();
    // A dbg.value for an alloca is always indirect.
    bool IsIndirect = isa<AllocaInst>(V) || Offset != 0;
    SDDbgValue *SDV;
    if (Val.getNode()) {
      if (!EmitFuncArgumentDbgValue(V, Variable, Expr, dl, Offset, IsIndirect,
                                    Val)) {
        SDV = DAG.getDbgValue(Variable, Expr, Val.getNode(), Val.getResNo(),
                              IsIndirect, Offset, dl, DbgSDNodeOrder);
        DAG.AddDbgValue(SDV, Val.getNode(), false);
      }
    } else
      DEBUG(dbgs() << "Dropping debug info for " << *DI << "\n");
    DanglingDebugInfoMap[V] = DanglingDebugInfo();
  }
}

/// getCopyFromRegs - If there was virtual register allocated for the value V
/// emit CopyFromReg of the specified type Ty. Return empty SDValue() otherwise.
SDValue SelectionDAGBuilder::getCopyFromRegs(const Value *V, Type *Ty) {
  DenseMap<const Value *, unsigned>::iterator It = FuncInfo.ValueMap.find(V);
  SDValue Result;

  if (It != FuncInfo.ValueMap.end()) {
    unsigned InReg = It->second;
    RegsForValue RFV(*DAG.getContext(), DAG.getTargetLoweringInfo(),
                     DAG.getDataLayout(), InReg, Ty);
    SDValue Chain = DAG.getEntryNode();
    Result = RFV.getCopyFromRegs(DAG, FuncInfo, getCurSDLoc(), Chain, nullptr, V);
    resolveDanglingDebugInfo(V, Result);
  }

  return Result;
}

/// getValue - Return an SDValue for the given Value.
SDValue SelectionDAGBuilder::getValue(const Value *V) {
  // If we already have an SDValue for this value, use it. It's important
  // to do this first, so that we don't create a CopyFromReg if we already
  // have a regular SDValue.
  SDValue &N = NodeMap[V];
  if (N.getNode()) return N;

  // If there's a virtual register allocated and initialized for this
  // value, use it.
  SDValue copyFromReg = getCopyFromRegs(V, V->getType());
  if (copyFromReg.getNode()) {
    return copyFromReg;
  }

  // Otherwise create a new SDValue and remember it.
  SDValue Val = getValueImpl(V);
  NodeMap[V] = Val;
  resolveDanglingDebugInfo(V, Val);
  return Val;
}

// Return true if SDValue exists for the given Value
bool SelectionDAGBuilder::findValue(const Value *V) const {
  return (NodeMap.find(V) != NodeMap.end()) ||
    (FuncInfo.ValueMap.find(V) != FuncInfo.ValueMap.end());
}

/// getNonRegisterValue - Return an SDValue for the given Value, but
/// don't look in FuncInfo.ValueMap for a virtual register.
SDValue SelectionDAGBuilder::getNonRegisterValue(const Value *V) {
  // If we already have an SDValue for this value, use it.
  SDValue &N = NodeMap[V];
  if (N.getNode()) {
    if (isa<ConstantSDNode>(N) || isa<ConstantFPSDNode>(N)) {
      // Remove the debug location from the node as the node is about to be used
      // in a location which may differ from the original debug location.  This
      // is relevant to Constant and ConstantFP nodes because they can appear
      // as constant expressions inside PHI nodes.
      N->setDebugLoc(DebugLoc());
    }
    return N;
  }

  // Otherwise create a new SDValue and remember it.
  SDValue Val = getValueImpl(V);
  NodeMap[V] = Val;
  resolveDanglingDebugInfo(V, Val);
  return Val;
}

/// getValueImpl - Helper function for getValue and getNonRegisterValue.
/// Create an SDValue for the given value.
SDValue SelectionDAGBuilder::getValueImpl(const Value *V) {
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();

  if (const Constant *C = dyn_cast<Constant>(V)) {
    EVT VT = TLI.getValueType(DAG.getDataLayout(), V->getType(), true);

    if (const ConstantInt *CI = dyn_cast<ConstantInt>(C))
      return DAG.getConstant(*CI, getCurSDLoc(), VT);

    if (const GlobalValue *GV = dyn_cast<GlobalValue>(C))
      return DAG.getGlobalAddress(GV, getCurSDLoc(), VT);

    if (isa<ConstantPointerNull>(C)) {
      unsigned AS = V->getType()->getPointerAddressSpace();
      return DAG.getConstant(0, getCurSDLoc(),
                             TLI.getPointerTy(DAG.getDataLayout(), AS));
    }

    if (const ConstantFP *CFP = dyn_cast<ConstantFP>(C))
      return DAG.getConstantFP(*CFP, getCurSDLoc(), VT);

    if (isa<UndefValue>(C) && !V->getType()->isAggregateType())
      return DAG.getUNDEF(VT);

    if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
      visit(CE->getOpcode(), *CE);
      SDValue N1 = NodeMap[V];
      assert(N1.getNode() && "visit didn't populate the NodeMap!");
      return N1;
    }

    if (isa<ConstantStruct>(C) || isa<ConstantArray>(C)) {
      SmallVector<SDValue, 4> Constants;
      for (User::const_op_iterator OI = C->op_begin(), OE = C->op_end();
           OI != OE; ++OI) {
        SDNode *Val = getValue(*OI).getNode();
        // If the operand is an empty aggregate, there are no values.
        if (!Val) continue;
        // Add each leaf value from the operand to the Constants list
        // to form a flattened list of all the values.
        for (unsigned i = 0, e = Val->getNumValues(); i != e; ++i)
          Constants.push_back(SDValue(Val, i));
      }

      return DAG.getMergeValues(Constants, getCurSDLoc());
    }

    if (const ConstantDataSequential *CDS =
          dyn_cast<ConstantDataSequential>(C)) {
      SmallVector<SDValue, 4> Ops;
      for (unsigned i = 0, e = CDS->getNumElements(); i != e; ++i) {
        SDNode *Val = getValue(CDS->getElementAsConstant(i)).getNode();
        // Add each leaf value from the operand to the Constants list
        // to form a flattened list of all the values.
        for (unsigned i = 0, e = Val->getNumValues(); i != e; ++i)
          Ops.push_back(SDValue(Val, i));
      }

      if (isa<ArrayType>(CDS->getType()))
        return DAG.getMergeValues(Ops, getCurSDLoc());
      return NodeMap[V] = DAG.getNode(ISD::BUILD_VECTOR, getCurSDLoc(),
                                      VT, Ops);
    }

    if (C->getType()->isStructTy() || C->getType()->isArrayTy()) {
      assert((isa<ConstantAggregateZero>(C) || isa<UndefValue>(C)) &&
             "Unknown struct or array constant!");

      SmallVector<EVT, 4> ValueVTs;
      ComputeValueVTs(TLI, DAG.getDataLayout(), C->getType(), ValueVTs);
      unsigned NumElts = ValueVTs.size();
      if (NumElts == 0)
        return SDValue(); // empty struct
      SmallVector<SDValue, 4> Constants(NumElts);
      for (unsigned i = 0; i != NumElts; ++i) {
        EVT EltVT = ValueVTs[i];
        if (isa<UndefValue>(C))
          Constants[i] = DAG.getUNDEF(EltVT);
        else if (EltVT.isFloatingPoint())
          Constants[i] = DAG.getConstantFP(0, getCurSDLoc(), EltVT);
        else
          Constants[i] = DAG.getConstant(0, getCurSDLoc(), EltVT);
      }

      return DAG.getMergeValues(Constants, getCurSDLoc());
    }

    if (const BlockAddress *BA = dyn_cast<BlockAddress>(C))
      return DAG.getBlockAddress(BA, VT);

    VectorType *VecTy = cast<VectorType>(V->getType());
    unsigned NumElements = VecTy->getNumElements();

    // Now that we know the number and type of the elements, get that number of
    // elements into the Ops array based on what kind of constant it is.
    SmallVector<SDValue, 16> Ops;
    if (const ConstantVector *CV = dyn_cast<ConstantVector>(C)) {
      for (unsigned i = 0; i != NumElements; ++i)
        Ops.push_back(getValue(CV->getOperand(i)));
    } else {
      assert(isa<ConstantAggregateZero>(C) && "Unknown vector constant!");
      EVT EltVT =
          TLI.getValueType(DAG.getDataLayout(), VecTy->getElementType());

      SDValue Op;
      if (EltVT.isFloatingPoint())
        Op = DAG.getConstantFP(0, getCurSDLoc(), EltVT);
      else
        Op = DAG.getConstant(0, getCurSDLoc(), EltVT);
      Ops.assign(NumElements, Op);
    }

    // Create a BUILD_VECTOR node.
    return NodeMap[V] = DAG.getNode(ISD::BUILD_VECTOR, getCurSDLoc(), VT, Ops);
  }

  // If this is a static alloca, generate it as the frameindex instead of
  // computation.
  if (const AllocaInst *AI = dyn_cast<AllocaInst>(V)) {
    DenseMap<const AllocaInst*, int>::iterator SI =
      FuncInfo.StaticAllocaMap.find(AI);
    if (SI != FuncInfo.StaticAllocaMap.end())
      return DAG.getFrameIndex(SI->second,
                               TLI.getPointerTy(DAG.getDataLayout()));
  }

  // If this is an instruction which fast-isel has deferred, select it now.
  if (const Instruction *Inst = dyn_cast<Instruction>(V)) {
    unsigned InReg = FuncInfo.InitializeRegForValue(Inst);
    RegsForValue RFV(*DAG.getContext(), TLI, DAG.getDataLayout(), InReg,
                     Inst->getType());
    SDValue Chain = DAG.getEntryNode();
    return RFV.getCopyFromRegs(DAG, FuncInfo, getCurSDLoc(), Chain, nullptr, V);
  }

  llvm_unreachable("Can't get register for value!");
}

void SelectionDAGBuilder::visitRet(const ReturnInst &I) {
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  auto &DL = DAG.getDataLayout();
  SDValue Chain = getControlRoot();
  SmallVector<ISD::OutputArg, 8> Outs;
  SmallVector<SDValue, 8> OutVals;

  if (!FuncInfo.CanLowerReturn) {
    unsigned DemoteReg = FuncInfo.DemoteRegister;
    const Function *F = I.getParent()->getParent();

    // Emit a store of the return value through the virtual register.
    // Leave Outs empty so that LowerReturn won't try to load return
    // registers the usual way.
    SmallVector<EVT, 1> PtrValueVTs;
    ComputeValueVTs(TLI, DL, PointerType::getUnqual(F->getReturnType()),
                    PtrValueVTs);

    SDValue RetPtr = DAG.getRegister(DemoteReg, PtrValueVTs[0]);
    SDValue RetOp = getValue(I.getOperand(0));

    SmallVector<EVT, 4> ValueVTs;
    SmallVector<uint64_t, 4> Offsets;
    ComputeValueVTs(TLI, DL, I.getOperand(0)->getType(), ValueVTs, &Offsets);
    unsigned NumValues = ValueVTs.size();

    SmallVector<SDValue, 4> Chains(NumValues);
    for (unsigned i = 0; i != NumValues; ++i) {
      SDValue Add = DAG.getNode(ISD::ADD, getCurSDLoc(),
                                RetPtr.getValueType(), RetPtr,
                                DAG.getIntPtrConstant(Offsets[i],
                                                      getCurSDLoc()));
      Chains[i] =
        DAG.getStore(Chain, getCurSDLoc(),
                     SDValue(RetOp.getNode(), RetOp.getResNo() + i),
                     // FIXME: better loc info would be nice.
                     Add, MachinePointerInfo(), false, false, 0);
    }

    Chain = DAG.getNode(ISD::TokenFactor, getCurSDLoc(),
                        MVT::Other, Chains);
  } else if (I.getNumOperands() != 0) {
    SmallVector<EVT, 4> ValueVTs;
    ComputeValueVTs(TLI, DL, I.getOperand(0)->getType(), ValueVTs);
    unsigned NumValues = ValueVTs.size();
    if (NumValues) {
      SDValue RetOp = getValue(I.getOperand(0));

      const Function *F = I.getParent()->getParent();

      ISD::NodeType ExtendKind = ISD::ANY_EXTEND;
      if (F->getAttributes().hasAttribute(AttributeSet::ReturnIndex,
                                          Attribute::SExt))
        ExtendKind = ISD::SIGN_EXTEND;
      else if (F->getAttributes().hasAttribute(AttributeSet::ReturnIndex,
                                               Attribute::ZExt))
        ExtendKind = ISD::ZERO_EXTEND;

      LLVMContext &Context = F->getContext();
      bool RetInReg = F->getAttributes().hasAttribute(AttributeSet::ReturnIndex,
                                                      Attribute::InReg);

      for (unsigned j = 0; j != NumValues; ++j) {
        EVT VT = ValueVTs[j];

        if (ExtendKind != ISD::ANY_EXTEND && VT.isInteger())
          VT = TLI.getTypeForExtArgOrReturn(Context, VT, ExtendKind);

        unsigned NumParts = TLI.getNumRegisters(Context, VT);
        MVT PartVT = TLI.getRegisterType(Context, VT);
        SmallVector<SDValue, 4> Parts(NumParts);
        getCopyToParts(DAG, getCurSDLoc(),
                       SDValue(RetOp.getNode(), RetOp.getResNo() + j),
                       &Parts[0], NumParts, PartVT, &I, ExtendKind);

        // 'inreg' on function refers to return value
        ISD::ArgFlagsTy Flags = ISD::ArgFlagsTy();
        if (RetInReg)
          Flags.setInReg();

        // Propagate extension type if any
        if (ExtendKind == ISD::SIGN_EXTEND)
          Flags.setSExt();
        else if (ExtendKind == ISD::ZERO_EXTEND)
          Flags.setZExt();

        for (unsigned i = 0; i < NumParts; ++i) {
          Outs.push_back(ISD::OutputArg(Flags, Parts[i].getValueType(),
                                        VT, /*isfixed=*/true, 0, 0));
          OutVals.push_back(Parts[i]);
        }
      }
    }
  }

  bool isVarArg = DAG.getMachineFunction().getFunction()->isVarArg();
  CallingConv::ID CallConv =
    DAG.getMachineFunction().getFunction()->getCallingConv();
  Chain = DAG.getTargetLoweringInfo().LowerReturn(
      Chain, CallConv, isVarArg, Outs, OutVals, getCurSDLoc(), DAG);

  // Verify that the target's LowerReturn behaved as expected.
  assert(Chain.getNode() && Chain.getValueType() == MVT::Other &&
         "LowerReturn didn't return a valid chain!");

  // Update the DAG with the new chain value resulting from return lowering.
  DAG.setRoot(Chain);
}

/// CopyToExportRegsIfNeeded - If the given value has virtual registers
/// created for it, emit nodes to copy the value into the virtual
/// registers.
void SelectionDAGBuilder::CopyToExportRegsIfNeeded(const Value *V) {
  // Skip empty types
  if (V->getType()->isEmptyTy())
    return;

  DenseMap<const Value *, unsigned>::iterator VMI = FuncInfo.ValueMap.find(V);
  if (VMI != FuncInfo.ValueMap.end()) {
    assert(!V->use_empty() && "Unused value assigned virtual registers!");
    CopyValueToVirtualRegister(V, VMI->second);
  }
}

/// ExportFromCurrentBlock - If this condition isn't known to be exported from
/// the current basic block, add it to ValueMap now so that we'll get a
/// CopyTo/FromReg.
void SelectionDAGBuilder::ExportFromCurrentBlock(const Value *V) {
  // No need to export constants.
  if (!isa<Instruction>(V) && !isa<Argument>(V)) return;

  // Already exported?
  if (FuncInfo.isExportedInst(V)) return;

  unsigned Reg = FuncInfo.InitializeRegForValue(V);
  CopyValueToVirtualRegister(V, Reg);
}

bool SelectionDAGBuilder::isExportableFromCurrentBlock(const Value *V,
                                                     const BasicBlock *FromBB) {
  // The operands of the setcc have to be in this block.  We don't know
  // how to export them from some other block.
  if (const Instruction *VI = dyn_cast<Instruction>(V)) {
    // Can export from current BB.
    if (VI->getParent() == FromBB)
      return true;

    // Is already exported, noop.
    return FuncInfo.isExportedInst(V);
  }

  // If this is an argument, we can export it if the BB is the entry block or
  // if it is already exported.
  if (isa<Argument>(V)) {
    if (FromBB == &FromBB->getParent()->getEntryBlock())
      return true;

    // Otherwise, can only export this if it is already exported.
    return FuncInfo.isExportedInst(V);
  }

  // Otherwise, constants can always be exported.
  return true;
}

/// Return branch probability calculated by BranchProbabilityInfo for IR blocks.
uint32_t SelectionDAGBuilder::getEdgeWeight(const MachineBasicBlock *Src,
                                            const MachineBasicBlock *Dst) const {
  BranchProbabilityInfo *BPI = FuncInfo.BPI;
  if (!BPI)
    return 0;
  const BasicBlock *SrcBB = Src->getBasicBlock();
  const BasicBlock *DstBB = Dst->getBasicBlock();
  return BPI->getEdgeWeight(SrcBB, DstBB);
}

void SelectionDAGBuilder::
addSuccessorWithWeight(MachineBasicBlock *Src, MachineBasicBlock *Dst,
                       uint32_t Weight /* = 0 */) {
  if (!Weight)
    Weight = getEdgeWeight(Src, Dst);
  Src->addSuccessor(Dst, Weight);
}


static bool InBlock(const Value *V, const BasicBlock *BB) {
  if (const Instruction *I = dyn_cast<Instruction>(V))
    return I->getParent() == BB;
  return true;
}

/// EmitBranchForMergedCondition - Helper method for FindMergedConditions.
/// This function emits a branch and is used at the leaves of an OR or an
/// AND operator tree.
///
void
SelectionDAGBuilder::EmitBranchForMergedCondition(const Value *Cond,
                                                  MachineBasicBlock *TBB,
                                                  MachineBasicBlock *FBB,
                                                  MachineBasicBlock *CurBB,
                                                  MachineBasicBlock *SwitchBB,
                                                  uint32_t TWeight,
                                                  uint32_t FWeight) {
  const BasicBlock *BB = CurBB->getBasicBlock();

  // If the leaf of the tree is a comparison, merge the condition into
  // the caseblock.
  if (const CmpInst *BOp = dyn_cast<CmpInst>(Cond)) {
    // The operands of the cmp have to be in this block.  We don't know
    // how to export them from some other block.  If this is the first block
    // of the sequence, no exporting is needed.
    if (CurBB == SwitchBB ||
        (isExportableFromCurrentBlock(BOp->getOperand(0), BB) &&
         isExportableFromCurrentBlock(BOp->getOperand(1), BB))) {
      ISD::CondCode Condition;
      if (const ICmpInst *IC = dyn_cast<ICmpInst>(Cond)) {
        Condition = getICmpCondCode(IC->getPredicate());
      } else if (const FCmpInst *FC = dyn_cast<FCmpInst>(Cond)) {
        Condition = getFCmpCondCode(FC->getPredicate());
        if (TM.Options.NoNaNsFPMath)
          Condition = getFCmpCodeWithoutNaN(Condition);
      } else {
        (void)Condition; // silence warning.
        llvm_unreachable("Unknown compare instruction");
      }

      CaseBlock CB(Condition, BOp->getOperand(0), BOp->getOperand(1), nullptr,
                   TBB, FBB, CurBB, TWeight, FWeight);
      SwitchCases.push_back(CB);
      return;
    }
  }

  // Create a CaseBlock record representing this branch.
  CaseBlock CB(ISD::SETEQ, Cond, ConstantInt::getTrue(*DAG.getContext()),
               nullptr, TBB, FBB, CurBB, TWeight, FWeight);
  SwitchCases.push_back(CB);
}

/// Scale down both weights to fit into uint32_t.
static void ScaleWeights(uint64_t &NewTrue, uint64_t &NewFalse) {
  uint64_t NewMax = (NewTrue > NewFalse) ? NewTrue : NewFalse;
  uint32_t Scale = (NewMax / UINT32_MAX) + 1;
  NewTrue = NewTrue / Scale;
  NewFalse = NewFalse / Scale;
}

/// FindMergedConditions - If Cond is an expression like
void SelectionDAGBuilder::FindMergedConditions(const Value *Cond,
                                               MachineBasicBlock *TBB,
                                               MachineBasicBlock *FBB,
                                               MachineBasicBlock *CurBB,
                                               MachineBasicBlock *SwitchBB,
                                               unsigned Opc, uint32_t TWeight,
                                               uint32_t FWeight) {
  // If this node is not part of the or/and tree, emit it as a branch.
  const Instruction *BOp = dyn_cast<Instruction>(Cond);
  if (!BOp || !(isa<BinaryOperator>(BOp) || isa<CmpInst>(BOp)) ||
      (unsigned)BOp->getOpcode() != Opc || !BOp->hasOneUse() ||
      BOp->getParent() != CurBB->getBasicBlock() ||
      !InBlock(BOp->getOperand(0), CurBB->getBasicBlock()) ||
      !InBlock(BOp->getOperand(1), CurBB->getBasicBlock())) {
    EmitBranchForMergedCondition(Cond, TBB, FBB, CurBB, SwitchBB,
                                 TWeight, FWeight);
    return;
  }

  //  Create TmpBB after CurBB.
  MachineFunction::iterator BBI = CurBB;
  MachineFunction &MF = DAG.getMachineFunction();
  MachineBasicBlock *TmpBB = MF.CreateMachineBasicBlock(CurBB->getBasicBlock());
  CurBB->getParent()->insert(++BBI, TmpBB);

  if (Opc == Instruction::Or) {
    // Codegen X | Y as:
    // BB1:
    //   jmp_if_X TBB
    //   jmp TmpBB
    // TmpBB:
    //   jmp_if_Y TBB
    //   jmp FBB
    //

    // We have flexibility in setting Prob for BB1 and Prob for TmpBB.
    // The requirement is that
    //   TrueProb for BB1 + (FalseProb for BB1 * TrueProb for TmpBB)
    //     = TrueProb for original BB.
    // Assuming the original weights are A and B, one choice is to set BB1's
    // weights to A and A+2B, and set TmpBB's weights to A and 2B. This choice
    // assumes that
    //   TrueProb for BB1 == FalseProb for BB1 * TrueProb for TmpBB.
    // Another choice is to assume TrueProb for BB1 equals to TrueProb for
    // TmpBB, but the math is more complicated.

    uint64_t NewTrueWeight = TWeight;
    uint64_t NewFalseWeight = (uint64_t)TWeight + 2 * (uint64_t)FWeight;
    ScaleWeights(NewTrueWeight, NewFalseWeight);
    // Emit the LHS condition.
    FindMergedConditions(BOp->getOperand(0), TBB, TmpBB, CurBB, SwitchBB, Opc,
                         NewTrueWeight, NewFalseWeight);

    NewTrueWeight = TWeight;
    NewFalseWeight = 2 * (uint64_t)FWeight;
    ScaleWeights(NewTrueWeight, NewFalseWeight);
    // Emit the RHS condition into TmpBB.
    FindMergedConditions(BOp->getOperand(1), TBB, FBB, TmpBB, SwitchBB, Opc,
                         NewTrueWeight, NewFalseWeight);
  } else {
    assert(Opc == Instruction::And && "Unknown merge op!");
    // Codegen X & Y as:
    // BB1:
    //   jmp_if_X TmpBB
    //   jmp FBB
    // TmpBB:
    //   jmp_if_Y TBB
    //   jmp FBB
    //
    //  This requires creation of TmpBB after CurBB.

    // We have flexibility in setting Prob for BB1 and Prob for TmpBB.
    // The requirement is that
    //   FalseProb for BB1 + (TrueProb for BB1 * FalseProb for TmpBB)
    //     = FalseProb for original BB.
    // Assuming the original weights are A and B, one choice is to set BB1's
    // weights to 2A+B and B, and set TmpBB's weights to 2A and B. This choice
    // assumes that
    //   FalseProb for BB1 == TrueProb for BB1 * FalseProb for TmpBB.

    uint64_t NewTrueWeight = 2 * (uint64_t)TWeight + (uint64_t)FWeight;
    uint64_t NewFalseWeight = FWeight;
    ScaleWeights(NewTrueWeight, NewFalseWeight);
    // Emit the LHS condition.
    FindMergedConditions(BOp->getOperand(0), TmpBB, FBB, CurBB, SwitchBB, Opc,
                         NewTrueWeight, NewFalseWeight);

    NewTrueWeight = 2 * (uint64_t)TWeight;
    NewFalseWeight = FWeight;
    ScaleWeights(NewTrueWeight, NewFalseWeight);
    // Emit the RHS condition into TmpBB.
    FindMergedConditions(BOp->getOperand(1), TBB, FBB, TmpBB, SwitchBB, Opc,
                         NewTrueWeight, NewFalseWeight);
  }
}

/// If the set of cases should be emitted as a series of branches, return true.
/// If we should emit this as a bunch of and/or'd together conditions, return
/// false.
bool
SelectionDAGBuilder::ShouldEmitAsBranches(const std::vector<CaseBlock> &Cases) {
  if (Cases.size() != 2) return true;

  // If this is two comparisons of the same values or'd or and'd together, they
  // will get folded into a single comparison, so don't emit two blocks.
  if ((Cases[0].CmpLHS == Cases[1].CmpLHS &&
       Cases[0].CmpRHS == Cases[1].CmpRHS) ||
      (Cases[0].CmpRHS == Cases[1].CmpLHS &&
       Cases[0].CmpLHS == Cases[1].CmpRHS)) {
    return false;
  }

  // Handle: (X != null) | (Y != null) --> (X|Y) != 0
  // Handle: (X == null) & (Y == null) --> (X|Y) == 0
  if (Cases[0].CmpRHS == Cases[1].CmpRHS &&
      Cases[0].CC == Cases[1].CC &&
      isa<Constant>(Cases[0].CmpRHS) &&
      cast<Constant>(Cases[0].CmpRHS)->isNullValue()) {
    if (Cases[0].CC == ISD::SETEQ && Cases[0].TrueBB == Cases[1].ThisBB)
      return false;
    if (Cases[0].CC == ISD::SETNE && Cases[0].FalseBB == Cases[1].ThisBB)
      return false;
  }

  return true;
}

void SelectionDAGBuilder::visitBr(const BranchInst &I) {
  MachineBasicBlock *BrMBB = FuncInfo.MBB;

  // Update machine-CFG edges.
  MachineBasicBlock *Succ0MBB = FuncInfo.MBBMap[I.getSuccessor(0)];

  if (I.isUnconditional()) {
    // Update machine-CFG edges.
    BrMBB->addSuccessor(Succ0MBB);

    // If this is not a fall-through branch or optimizations are switched off,
    // emit the branch.
    if (Succ0MBB != NextBlock(BrMBB) || TM.getOptLevel() == CodeGenOpt::None)
      DAG.setRoot(DAG.getNode(ISD::BR, getCurSDLoc(),
                              MVT::Other, getControlRoot(),
                              DAG.getBasicBlock(Succ0MBB)));

    return;
  }

  // If this condition is one of the special cases we handle, do special stuff
  // now.
  const Value *CondVal = I.getCondition();
  MachineBasicBlock *Succ1MBB = FuncInfo.MBBMap[I.getSuccessor(1)];

  // If this is a series of conditions that are or'd or and'd together, emit
  // this as a sequence of branches instead of setcc's with and/or operations.
  // As long as jumps are not expensive, this should improve performance.
  // For example, instead of something like:
  //     cmp A, B
  //     C = seteq
  //     cmp D, E
  //     F = setle
  //     or C, F
  //     jnz foo
  // Emit:
  //     cmp A, B
  //     je foo
  //     cmp D, E
  //     jle foo
  //
  if (const BinaryOperator *BOp = dyn_cast<BinaryOperator>(CondVal)) {
    if (!DAG.getTargetLoweringInfo().isJumpExpensive() &&
        BOp->hasOneUse() && (BOp->getOpcode() == Instruction::And ||
                             BOp->getOpcode() == Instruction::Or)) {
      FindMergedConditions(BOp, Succ0MBB, Succ1MBB, BrMBB, BrMBB,
                           BOp->getOpcode(), getEdgeWeight(BrMBB, Succ0MBB),
                           getEdgeWeight(BrMBB, Succ1MBB));
      // If the compares in later blocks need to use values not currently
      // exported from this block, export them now.  This block should always
      // be the first entry.
      assert(SwitchCases[0].ThisBB == BrMBB && "Unexpected lowering!");

      // Allow some cases to be rejected.
      if (ShouldEmitAsBranches(SwitchCases)) {
        for (unsigned i = 1, e = SwitchCases.size(); i != e; ++i) {
          ExportFromCurrentBlock(SwitchCases[i].CmpLHS);
          ExportFromCurrentBlock(SwitchCases[i].CmpRHS);
        }

        // Emit the branch for this block.
        visitSwitchCase(SwitchCases[0], BrMBB);
        SwitchCases.erase(SwitchCases.begin());
        return;
      }

      // Okay, we decided not to do this, remove any inserted MBB's and clear
      // SwitchCases.
      for (unsigned i = 1, e = SwitchCases.size(); i != e; ++i)
        FuncInfo.MF->erase(SwitchCases[i].ThisBB);

      SwitchCases.clear();
    }
  }

  // Create a CaseBlock record representing this branch.
  CaseBlock CB(ISD::SETEQ, CondVal, ConstantInt::getTrue(*DAG.getContext()),
               nullptr, Succ0MBB, Succ1MBB, BrMBB);

  // Use visitSwitchCase to actually insert the fast branch sequence for this
  // cond branch.
  visitSwitchCase(CB, BrMBB);
}

/// visitSwitchCase - Emits the necessary code to represent a single node in
/// the binary search tree resulting from lowering a switch instruction.
void SelectionDAGBuilder::visitSwitchCase(CaseBlock &CB,
                                          MachineBasicBlock *SwitchBB) {
  SDValue Cond;
  SDValue CondLHS = getValue(CB.CmpLHS);
  SDLoc dl = getCurSDLoc();

  // Build the setcc now.
  if (!CB.CmpMHS) {
    // Fold "(X == true)" to X and "(X == false)" to !X to
    // handle common cases produced by branch lowering.
    if (CB.CmpRHS == ConstantInt::getTrue(*DAG.getContext()) &&
        CB.CC == ISD::SETEQ)
      Cond = CondLHS;
    else if (CB.CmpRHS == ConstantInt::getFalse(*DAG.getContext()) &&
             CB.CC == ISD::SETEQ) {
      SDValue True = DAG.getConstant(1, dl, CondLHS.getValueType());
      Cond = DAG.getNode(ISD::XOR, dl, CondLHS.getValueType(), CondLHS, True);
    } else
      Cond = DAG.getSetCC(dl, MVT::i1, CondLHS, getValue(CB.CmpRHS), CB.CC);
  } else {
    assert(CB.CC == ISD::SETLE && "Can handle only LE ranges now");

    const APInt& Low = cast<ConstantInt>(CB.CmpLHS)->getValue();
    const APInt& High = cast<ConstantInt>(CB.CmpRHS)->getValue();

    SDValue CmpOp = getValue(CB.CmpMHS);
    EVT VT = CmpOp.getValueType();

    if (cast<ConstantInt>(CB.CmpLHS)->isMinValue(true)) {
      Cond = DAG.getSetCC(dl, MVT::i1, CmpOp, DAG.getConstant(High, dl, VT),
                          ISD::SETLE);
    } else {
      SDValue SUB = DAG.getNode(ISD::SUB, dl,
                                VT, CmpOp, DAG.getConstant(Low, dl, VT));
      Cond = DAG.getSetCC(dl, MVT::i1, SUB,
                          DAG.getConstant(High-Low, dl, VT), ISD::SETULE);
    }
  }

  // Update successor info
  addSuccessorWithWeight(SwitchBB, CB.TrueBB, CB.TrueWeight);
  // TrueBB and FalseBB are always different unless the incoming IR is
  // degenerate. This only happens when running llc on weird IR.
  if (CB.TrueBB != CB.FalseBB)
    addSuccessorWithWeight(SwitchBB, CB.FalseBB, CB.FalseWeight);

  // If the lhs block is the next block, invert the condition so that we can
  // fall through to the lhs instead of the rhs block.
  if (CB.TrueBB == NextBlock(SwitchBB)) {
    std::swap(CB.TrueBB, CB.FalseBB);
    SDValue True = DAG.getConstant(1, dl, Cond.getValueType());
    Cond = DAG.getNode(ISD::XOR, dl, Cond.getValueType(), Cond, True);
  }

  SDValue BrCond = DAG.getNode(ISD::BRCOND, dl,
                               MVT::Other, getControlRoot(), Cond,
                               DAG.getBasicBlock(CB.TrueBB));

  // Insert the false branch. Do this even if it's a fall through branch,
  // this makes it easier to do DAG optimizations which require inverting
  // the branch condition.
  BrCond = DAG.getNode(ISD::BR, dl, MVT::Other, BrCond,
                       DAG.getBasicBlock(CB.FalseBB));

  DAG.setRoot(BrCond);
}

/// visitJumpTable - Emit JumpTable node in the current MBB
void SelectionDAGBuilder::visitJumpTable(JumpTable &JT) {
  // Emit the code for the jump table
  assert(JT.Reg != -1U && "Should lower JT Header first!");
  EVT PTy = DAG.getTargetLoweringInfo().getPointerTy(DAG.getDataLayout());
  SDValue Index = DAG.getCopyFromReg(getControlRoot(), getCurSDLoc(),
                                     JT.Reg, PTy);
  SDValue Table = DAG.getJumpTable(JT.JTI, PTy);
  SDValue BrJumpTable = DAG.getNode(ISD::BR_JT, getCurSDLoc(),
                                    MVT::Other, Index.getValue(1),
                                    Table, Index);
  DAG.setRoot(BrJumpTable);
}

/// visitJumpTableHeader - This function emits necessary code to produce index
/// in the JumpTable from switch case.
void SelectionDAGBuilder::visitJumpTableHeader(JumpTable &JT,
                                               JumpTableHeader &JTH,
                                               MachineBasicBlock *SwitchBB) {
  SDLoc dl = getCurSDLoc();

  // Subtract the lowest switch case value from the value being switched on and
  // conditional branch to default mbb if the result is greater than the
  // difference between smallest and largest cases.
  SDValue SwitchOp = getValue(JTH.SValue);
  EVT VT = SwitchOp.getValueType();
  SDValue Sub = DAG.getNode(ISD::SUB, dl, VT, SwitchOp,
                            DAG.getConstant(JTH.First, dl, VT));

  // The SDNode we just created, which holds the value being switched on minus
  // the smallest case value, needs to be copied to a virtual register so it
  // can be used as an index into the jump table in a subsequent basic block.
  // This value may be smaller or larger than the target's pointer type, and
  // therefore require extension or truncating.
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  SwitchOp = DAG.getZExtOrTrunc(Sub, dl, TLI.getPointerTy(DAG.getDataLayout()));

  unsigned JumpTableReg =
      FuncInfo.CreateReg(TLI.getPointerTy(DAG.getDataLayout()));
  SDValue CopyTo = DAG.getCopyToReg(getControlRoot(), dl,
                                    JumpTableReg, SwitchOp);
  JT.Reg = JumpTableReg;

  // Emit the range check for the jump table, and branch to the default block
  // for the switch statement if the value being switched on exceeds the largest
  // case in the switch.
  SDValue CMP = DAG.getSetCC(
      dl, TLI.getSetCCResultType(DAG.getDataLayout(), *DAG.getContext(),
                                 Sub.getValueType()),
      Sub, DAG.getConstant(JTH.Last - JTH.First, dl, VT), ISD::SETUGT);

  SDValue BrCond = DAG.getNode(ISD::BRCOND, dl,
                               MVT::Other, CopyTo, CMP,
                               DAG.getBasicBlock(JT.Default));

  // Avoid emitting unnecessary branches to the next block.
  if (JT.MBB != NextBlock(SwitchBB))
    BrCond = DAG.getNode(ISD::BR, dl, MVT::Other, BrCond,
                         DAG.getBasicBlock(JT.MBB));

  DAG.setRoot(BrCond);
}

/// Codegen a new tail for a stack protector check ParentMBB which has had its
/// tail spliced into a stack protector check success bb.
///
/// For a high level explanation of how this fits into the stack protector
/// generation see the comment on the declaration of class
/// StackProtectorDescriptor.
void SelectionDAGBuilder::visitSPDescriptorParent(StackProtectorDescriptor &SPD,
                                                  MachineBasicBlock *ParentBB) {

  // First create the loads to the guard/stack slot for the comparison.
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  EVT PtrTy = TLI.getPointerTy(DAG.getDataLayout());

  MachineFrameInfo *MFI = ParentBB->getParent()->getFrameInfo();
  int FI = MFI->getStackProtectorIndex();

  const Value *IRGuard = SPD.getGuard();
  SDValue GuardPtr = getValue(IRGuard);
  SDValue StackSlotPtr = DAG.getFrameIndex(FI, PtrTy);

  unsigned Align = DL->getPrefTypeAlignment(IRGuard->getType());

  SDValue Guard;
  SDLoc dl = getCurSDLoc();

  // If GuardReg is set and useLoadStackGuardNode returns true, retrieve the
  // guard value from the virtual register holding the value. Otherwise, emit a
  // volatile load to retrieve the stack guard value.
  unsigned GuardReg = SPD.getGuardReg();

  if (GuardReg && TLI.useLoadStackGuardNode())
    Guard = DAG.getCopyFromReg(DAG.getEntryNode(), dl, GuardReg,
                               PtrTy);
  else
    Guard = DAG.getLoad(PtrTy, dl, DAG.getEntryNode(),
                        GuardPtr, MachinePointerInfo(IRGuard, 0),
                        true, false, false, Align);

  SDValue StackSlot = DAG.getLoad(PtrTy, dl, DAG.getEntryNode(),
                                  StackSlotPtr,
                                  MachinePointerInfo::getFixedStack(FI),
                                  true, false, false, Align);

  // Perform the comparison via a subtract/getsetcc.
  EVT VT = Guard.getValueType();
  SDValue Sub = DAG.getNode(ISD::SUB, dl, VT, Guard, StackSlot);

  SDValue Cmp = DAG.getSetCC(dl, TLI.getSetCCResultType(DAG.getDataLayout(),
                                                        *DAG.getContext(),
                                                        Sub.getValueType()),
                             Sub, DAG.getConstant(0, dl, VT), ISD::SETNE);

  // If the sub is not 0, then we know the guard/stackslot do not equal, so
  // branch to failure MBB.
  SDValue BrCond = DAG.getNode(ISD::BRCOND, dl,
                               MVT::Other, StackSlot.getOperand(0),
                               Cmp, DAG.getBasicBlock(SPD.getFailureMBB()));
  // Otherwise branch to success MBB.
  SDValue Br = DAG.getNode(ISD::BR, dl,
                           MVT::Other, BrCond,
                           DAG.getBasicBlock(SPD.getSuccessMBB()));

  DAG.setRoot(Br);
}

/// Codegen the failure basic block for a stack protector check.
///
/// A failure stack protector machine basic block consists simply of a call to
/// __stack_chk_fail().
///
/// For a high level explanation of how this fits into the stack protector
/// generation see the comment on the declaration of class
/// StackProtectorDescriptor.
void
SelectionDAGBuilder::visitSPDescriptorFailure(StackProtectorDescriptor &SPD) {
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  SDValue Chain =
      TLI.makeLibCall(DAG, RTLIB::STACKPROTECTOR_CHECK_FAIL, MVT::isVoid,
                      nullptr, 0, false, getCurSDLoc(), false, false).second;
  DAG.setRoot(Chain);
}

/// visitBitTestHeader - This function emits necessary code to produce value
/// suitable for "bit tests"
void SelectionDAGBuilder::visitBitTestHeader(BitTestBlock &B,
                                             MachineBasicBlock *SwitchBB) {
  SDLoc dl = getCurSDLoc();

  // Subtract the minimum value
  SDValue SwitchOp = getValue(B.SValue);
  EVT VT = SwitchOp.getValueType();
  SDValue Sub = DAG.getNode(ISD::SUB, dl, VT, SwitchOp,
                            DAG.getConstant(B.First, dl, VT));

  // Check range
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  SDValue RangeCmp = DAG.getSetCC(
      dl, TLI.getSetCCResultType(DAG.getDataLayout(), *DAG.getContext(),
                                 Sub.getValueType()),
      Sub, DAG.getConstant(B.Range, dl, VT), ISD::SETUGT);

  // Determine the type of the test operands.
  bool UsePtrType = false;
  if (!TLI.isTypeLegal(VT))
    UsePtrType = true;
  else {
    for (unsigned i = 0, e = B.Cases.size(); i != e; ++i)
      if (!isUIntN(VT.getSizeInBits(), B.Cases[i].Mask)) {
        // Switch table case range are encoded into series of masks.
        // Just use pointer type, it's guaranteed to fit.
        UsePtrType = true;
        break;
      }
  }
  if (UsePtrType) {
    VT = TLI.getPointerTy(DAG.getDataLayout());
    Sub = DAG.getZExtOrTrunc(Sub, dl, VT);
  }

  B.RegVT = VT.getSimpleVT();
  B.Reg = FuncInfo.CreateReg(B.RegVT);
  SDValue CopyTo = DAG.getCopyToReg(getControlRoot(), dl, B.Reg, Sub);

  MachineBasicBlock* MBB = B.Cases[0].ThisBB;

  addSuccessorWithWeight(SwitchBB, B.Default);
  addSuccessorWithWeight(SwitchBB, MBB);

  SDValue BrRange = DAG.getNode(ISD::BRCOND, dl,
                                MVT::Other, CopyTo, RangeCmp,
                                DAG.getBasicBlock(B.Default));

  // Avoid emitting unnecessary branches to the next block.
  if (MBB != NextBlock(SwitchBB))
    BrRange = DAG.getNode(ISD::BR, dl, MVT::Other, BrRange,
                          DAG.getBasicBlock(MBB));

  DAG.setRoot(BrRange);
}

/// visitBitTestCase - this function produces one "bit test"
void SelectionDAGBuilder::visitBitTestCase(BitTestBlock &BB,
                                           MachineBasicBlock* NextMBB,
                                           uint32_t BranchWeightToNext,
                                           unsigned Reg,
                                           BitTestCase &B,
                                           MachineBasicBlock *SwitchBB) {
  SDLoc dl = getCurSDLoc();
  MVT VT = BB.RegVT;
  SDValue ShiftOp = DAG.getCopyFromReg(getControlRoot(), dl, Reg, VT);
  SDValue Cmp;
  unsigned PopCount = countPopulation(B.Mask);
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  if (PopCount == 1) {
    // Testing for a single bit; just compare the shift count with what it
    // would need to be to shift a 1 bit in that position.
    Cmp = DAG.getSetCC(
        dl, TLI.getSetCCResultType(DAG.getDataLayout(), *DAG.getContext(), VT),
        ShiftOp, DAG.getConstant(countTrailingZeros(B.Mask), dl, VT),
        ISD::SETEQ);
  } else if (PopCount == BB.Range) {
    // There is only one zero bit in the range, test for it directly.
    Cmp = DAG.getSetCC(
        dl, TLI.getSetCCResultType(DAG.getDataLayout(), *DAG.getContext(), VT),
        ShiftOp, DAG.getConstant(countTrailingOnes(B.Mask), dl, VT),
        ISD::SETNE);
  } else {
    // Make desired shift
    SDValue SwitchVal = DAG.getNode(ISD::SHL, dl, VT,
                                    DAG.getConstant(1, dl, VT), ShiftOp);

    // Emit bit tests and jumps
    SDValue AndOp = DAG.getNode(ISD::AND, dl,
                                VT, SwitchVal, DAG.getConstant(B.Mask, dl, VT));
    Cmp = DAG.getSetCC(
        dl, TLI.getSetCCResultType(DAG.getDataLayout(), *DAG.getContext(), VT),
        AndOp, DAG.getConstant(0, dl, VT), ISD::SETNE);
  }

  // The branch weight from SwitchBB to B.TargetBB is B.ExtraWeight.
  addSuccessorWithWeight(SwitchBB, B.TargetBB, B.ExtraWeight);
  // The branch weight from SwitchBB to NextMBB is BranchWeightToNext.
  addSuccessorWithWeight(SwitchBB, NextMBB, BranchWeightToNext);

  SDValue BrAnd = DAG.getNode(ISD::BRCOND, dl,
                              MVT::Other, getControlRoot(),
                              Cmp, DAG.getBasicBlock(B.TargetBB));

  // Avoid emitting unnecessary branches to the next block.
  if (NextMBB != NextBlock(SwitchBB))
    BrAnd = DAG.getNode(ISD::BR, dl, MVT::Other, BrAnd,
                        DAG.getBasicBlock(NextMBB));

  DAG.setRoot(BrAnd);
}

void SelectionDAGBuilder::visitInvoke(const InvokeInst &I) {
  MachineBasicBlock *InvokeMBB = FuncInfo.MBB;

  // Retrieve successors.
  MachineBasicBlock *Return = FuncInfo.MBBMap[I.getSuccessor(0)];
  MachineBasicBlock *LandingPad = FuncInfo.MBBMap[I.getSuccessor(1)];

  const Value *Callee(I.getCalledValue());
  const Function *Fn = dyn_cast<Function>(Callee);
  if (isa<InlineAsm>(Callee))
    visitInlineAsm(&I);
  else if (Fn && Fn->isIntrinsic()) {
    switch (Fn->getIntrinsicID()) {
    default:
      llvm_unreachable("Cannot invoke this intrinsic");
    case Intrinsic::donothing:
      // Ignore invokes to @llvm.donothing: jump directly to the next BB.
      break;
    case Intrinsic::experimental_patchpoint_void:
    case Intrinsic::experimental_patchpoint_i64:
      visitPatchpoint(&I, LandingPad);
      break;
    case Intrinsic::experimental_gc_statepoint:
      LowerStatepoint(ImmutableStatepoint(&I), LandingPad);
      break;
    }
  } else
    LowerCallTo(&I, getValue(Callee), false, LandingPad);

  // If the value of the invoke is used outside of its defining block, make it
  // available as a virtual register.
  // We already took care of the exported value for the statepoint instruction
  // during call to the LowerStatepoint.
  if (!isStatepoint(I)) {
    CopyToExportRegsIfNeeded(&I);
  }

  // Update successor info
  addSuccessorWithWeight(InvokeMBB, Return);
  addSuccessorWithWeight(InvokeMBB, LandingPad);

  // Drop into normal successor.
  DAG.setRoot(DAG.getNode(ISD::BR, getCurSDLoc(),
                          MVT::Other, getControlRoot(),
                          DAG.getBasicBlock(Return)));
}

void SelectionDAGBuilder::visitResume(const ResumeInst &RI) {
  llvm_unreachable("SelectionDAGBuilder shouldn't visit resume instructions!");
}

void SelectionDAGBuilder::visitLandingPad(const LandingPadInst &LP) {
  assert(FuncInfo.MBB->isLandingPad() &&
         "Call to landingpad not in landing pad!");

  MachineBasicBlock *MBB = FuncInfo.MBB;
  MachineModuleInfo &MMI = DAG.getMachineFunction().getMMI();
  AddLandingPadInfo(LP, MMI, MBB);

  // If there aren't registers to copy the values into (e.g., during SjLj
  // exceptions), then don't bother to create these DAG nodes.
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  if (TLI.getExceptionPointerRegister() == 0 &&
      TLI.getExceptionSelectorRegister() == 0)
    return;

  SmallVector<EVT, 2> ValueVTs;
  SDLoc dl = getCurSDLoc();
  ComputeValueVTs(TLI, DAG.getDataLayout(), LP.getType(), ValueVTs);
  assert(ValueVTs.size() == 2 && "Only two-valued landingpads are supported");

  // Get the two live-in registers as SDValues. The physregs have already been
  // copied into virtual registers.
  SDValue Ops[2];
  if (FuncInfo.ExceptionPointerVirtReg) {
    Ops[0] = DAG.getZExtOrTrunc(
        DAG.getCopyFromReg(DAG.getEntryNode(), dl,
                           FuncInfo.ExceptionPointerVirtReg,
                           TLI.getPointerTy(DAG.getDataLayout())),
        dl, ValueVTs[0]);
  } else {
    Ops[0] = DAG.getConstant(0, dl, TLI.getPointerTy(DAG.getDataLayout()));
  }
  Ops[1] = DAG.getZExtOrTrunc(
      DAG.getCopyFromReg(DAG.getEntryNode(), dl,
                         FuncInfo.ExceptionSelectorVirtReg,
                         TLI.getPointerTy(DAG.getDataLayout())),
      dl, ValueVTs[1]);

  // Merge into one.
  SDValue Res = DAG.getNode(ISD::MERGE_VALUES, dl,
                            DAG.getVTList(ValueVTs), Ops);
  setValue(&LP, Res);
}

void SelectionDAGBuilder::sortAndRangeify(CaseClusterVector &Clusters) {
#ifndef NDEBUG
  for (const CaseCluster &CC : Clusters)
    assert(CC.Low == CC.High && "Input clusters must be single-case");
#endif

  std::sort(Clusters.begin(), Clusters.end(),
            [](const CaseCluster &a, const CaseCluster &b) {
    return a.Low->getValue().slt(b.Low->getValue());
  });

  // Merge adjacent clusters with the same destination.
  const unsigned N = Clusters.size();
  unsigned DstIndex = 0;
  for (unsigned SrcIndex = 0; SrcIndex < N; ++SrcIndex) {
    CaseCluster &CC = Clusters[SrcIndex];
    const ConstantInt *CaseVal = CC.Low;
    MachineBasicBlock *Succ = CC.MBB;

    if (DstIndex != 0 && Clusters[DstIndex - 1].MBB == Succ &&
        (CaseVal->getValue() - Clusters[DstIndex - 1].High->getValue()) == 1) {
      // If this case has the same successor and is a neighbour, merge it into
      // the previous cluster.
      Clusters[DstIndex - 1].High = CaseVal;
      Clusters[DstIndex - 1].Weight += CC.Weight;
      assert(Clusters[DstIndex - 1].Weight >= CC.Weight && "Weight overflow!");
    } else {
      std::memmove(&Clusters[DstIndex++], &Clusters[SrcIndex],
                   sizeof(Clusters[SrcIndex]));
    }
  }
  Clusters.resize(DstIndex);
}

void SelectionDAGBuilder::UpdateSplitBlock(MachineBasicBlock *First,
                                           MachineBasicBlock *Last) {
  // Update JTCases.
  for (unsigned i = 0, e = JTCases.size(); i != e; ++i)
    if (JTCases[i].first.HeaderBB == First)
      JTCases[i].first.HeaderBB = Last;

  // Update BitTestCases.
  for (unsigned i = 0, e = BitTestCases.size(); i != e; ++i)
    if (BitTestCases[i].Parent == First)
      BitTestCases[i].Parent = Last;
}

void SelectionDAGBuilder::visitIndirectBr(const IndirectBrInst &I) {
  MachineBasicBlock *IndirectBrMBB = FuncInfo.MBB;

  // Update machine-CFG edges with unique successors.
  SmallSet<BasicBlock*, 32> Done;
  for (unsigned i = 0, e = I.getNumSuccessors(); i != e; ++i) {
    BasicBlock *BB = I.getSuccessor(i);
    bool Inserted = Done.insert(BB).second;
    if (!Inserted)
        continue;

    MachineBasicBlock *Succ = FuncInfo.MBBMap[BB];
    addSuccessorWithWeight(IndirectBrMBB, Succ);
  }

  DAG.setRoot(DAG.getNode(ISD::BRIND, getCurSDLoc(),
                          MVT::Other, getControlRoot(),
                          getValue(I.getAddress())));
}

void SelectionDAGBuilder::visitUnreachable(const UnreachableInst &I) {
  if (DAG.getTarget().Options.TrapUnreachable)
    DAG.setRoot(DAG.getNode(ISD::TRAP, getCurSDLoc(), MVT::Other, DAG.getRoot()));
}

void SelectionDAGBuilder::visitFSub(const User &I) {
  // -0.0 - X --> fneg
  Type *Ty = I.getType();
  if (isa<Constant>(I.getOperand(0)) &&
      I.getOperand(0) == ConstantFP::getZeroValueForNegation(Ty)) {
    SDValue Op2 = getValue(I.getOperand(1));
    setValue(&I, DAG.getNode(ISD::FNEG, getCurSDLoc(),
                             Op2.getValueType(), Op2));
    return;
  }

  visitBinary(I, ISD::FSUB);
}

void SelectionDAGBuilder::visitBinary(const User &I, unsigned OpCode) {
  SDValue Op1 = getValue(I.getOperand(0));
  SDValue Op2 = getValue(I.getOperand(1));

  bool nuw = false;
  bool nsw = false;
  bool exact = false;
  FastMathFlags FMF;

  if (const OverflowingBinaryOperator *OFBinOp =
          dyn_cast<const OverflowingBinaryOperator>(&I)) {
    nuw = OFBinOp->hasNoUnsignedWrap();
    nsw = OFBinOp->hasNoSignedWrap();
  }
  if (const PossiblyExactOperator *ExactOp =
          dyn_cast<const PossiblyExactOperator>(&I))
    exact = ExactOp->isExact();
  if (const FPMathOperator *FPOp = dyn_cast<const FPMathOperator>(&I))
    FMF = FPOp->getFastMathFlags();

  SDNodeFlags Flags;
  Flags.setExact(exact);
  Flags.setNoSignedWrap(nsw);
  Flags.setNoUnsignedWrap(nuw);
  if (EnableFMFInDAG) {
    Flags.setAllowReciprocal(FMF.allowReciprocal());
    Flags.setNoInfs(FMF.noInfs());
    Flags.setNoNaNs(FMF.noNaNs());
    Flags.setNoSignedZeros(FMF.noSignedZeros());
    Flags.setUnsafeAlgebra(FMF.unsafeAlgebra());
  }
  SDValue BinNodeValue = DAG.getNode(OpCode, getCurSDLoc(), Op1.getValueType(),
                                     Op1, Op2, &Flags);
  setValue(&I, BinNodeValue);
}

void SelectionDAGBuilder::visitShift(const User &I, unsigned Opcode) {
  SDValue Op1 = getValue(I.getOperand(0));
  SDValue Op2 = getValue(I.getOperand(1));

  EVT ShiftTy = DAG.getTargetLoweringInfo().getShiftAmountTy(
      Op2.getValueType(), DAG.getDataLayout());

  // Coerce the shift amount to the right type if we can.
  if (!I.getType()->isVectorTy() && Op2.getValueType() != ShiftTy) {
    unsigned ShiftSize = ShiftTy.getSizeInBits();
    unsigned Op2Size = Op2.getValueType().getSizeInBits();
    SDLoc DL = getCurSDLoc();

    // If the operand is smaller than the shift count type, promote it.
    if (ShiftSize > Op2Size)
      Op2 = DAG.getNode(ISD::ZERO_EXTEND, DL, ShiftTy, Op2);

    // If the operand is larger than the shift count type but the shift
    // count type has enough bits to represent any shift value, truncate
    // it now. This is a common case and it exposes the truncate to
    // optimization early.
    else if (ShiftSize >= Log2_32_Ceil(Op2.getValueType().getSizeInBits()))
      Op2 = DAG.getNode(ISD::TRUNCATE, DL, ShiftTy, Op2);
    // Otherwise we'll need to temporarily settle for some other convenient
    // type.  Type legalization will make adjustments once the shiftee is split.
    else
      Op2 = DAG.getZExtOrTrunc(Op2, DL, MVT::i32);
  }

  bool nuw = false;
  bool nsw = false;
  bool exact = false;

  if (Opcode == ISD::SRL || Opcode == ISD::SRA || Opcode == ISD::SHL) {

    if (const OverflowingBinaryOperator *OFBinOp =
            dyn_cast<const OverflowingBinaryOperator>(&I)) {
      nuw = OFBinOp->hasNoUnsignedWrap();
      nsw = OFBinOp->hasNoSignedWrap();
    }
    if (const PossiblyExactOperator *ExactOp =
            dyn_cast<const PossiblyExactOperator>(&I))
      exact = ExactOp->isExact();
  }
  SDNodeFlags Flags;
  Flags.setExact(exact);
  Flags.setNoSignedWrap(nsw);
  Flags.setNoUnsignedWrap(nuw);
  SDValue Res = DAG.getNode(Opcode, getCurSDLoc(), Op1.getValueType(), Op1, Op2,
                            &Flags);
  setValue(&I, Res);
}

void SelectionDAGBuilder::visitSDiv(const User &I) {
  SDValue Op1 = getValue(I.getOperand(0));
  SDValue Op2 = getValue(I.getOperand(1));

  SDNodeFlags Flags;
  Flags.setExact(isa<PossiblyExactOperator>(&I) &&
                 cast<PossiblyExactOperator>(&I)->isExact());
  setValue(&I, DAG.getNode(ISD::SDIV, getCurSDLoc(), Op1.getValueType(), Op1,
                           Op2, &Flags));
}

void SelectionDAGBuilder::visitICmp(const User &I) {
  ICmpInst::Predicate predicate = ICmpInst::BAD_ICMP_PREDICATE;
  if (const ICmpInst *IC = dyn_cast<ICmpInst>(&I))
    predicate = IC->getPredicate();
  else if (const ConstantExpr *IC = dyn_cast<ConstantExpr>(&I))
    predicate = ICmpInst::Predicate(IC->getPredicate());
  SDValue Op1 = getValue(I.getOperand(0));
  SDValue Op2 = getValue(I.getOperand(1));
  ISD::CondCode Opcode = getICmpCondCode(predicate);

  EVT DestVT = DAG.getTargetLoweringInfo().getValueType(DAG.getDataLayout(),
                                                        I.getType());
  setValue(&I, DAG.getSetCC(getCurSDLoc(), DestVT, Op1, Op2, Opcode));
}

void SelectionDAGBuilder::visitFCmp(const User &I) {
  FCmpInst::Predicate predicate = FCmpInst::BAD_FCMP_PREDICATE;
  if (const FCmpInst *FC = dyn_cast<FCmpInst>(&I))
    predicate = FC->getPredicate();
  else if (const ConstantExpr *FC = dyn_cast<ConstantExpr>(&I))
    predicate = FCmpInst::Predicate(FC->getPredicate());
  SDValue Op1 = getValue(I.getOperand(0));
  SDValue Op2 = getValue(I.getOperand(1));
  ISD::CondCode Condition = getFCmpCondCode(predicate);
  if (TM.Options.NoNaNsFPMath)
    Condition = getFCmpCodeWithoutNaN(Condition);
  EVT DestVT = DAG.getTargetLoweringInfo().getValueType(DAG.getDataLayout(),
                                                        I.getType());
  setValue(&I, DAG.getSetCC(getCurSDLoc(), DestVT, Op1, Op2, Condition));
}

void SelectionDAGBuilder::visitSelect(const User &I) {
  SmallVector<EVT, 4> ValueVTs;
  ComputeValueVTs(DAG.getTargetLoweringInfo(), DAG.getDataLayout(), I.getType(),
                  ValueVTs);
  unsigned NumValues = ValueVTs.size();
  if (NumValues == 0) return;

  SmallVector<SDValue, 4> Values(NumValues);
  SDValue Cond     = getValue(I.getOperand(0));
  SDValue LHSVal   = getValue(I.getOperand(1));
  SDValue RHSVal   = getValue(I.getOperand(2));
  auto BaseOps = {Cond};
  ISD::NodeType OpCode = Cond.getValueType().isVector() ?
    ISD::VSELECT : ISD::SELECT;

  // Min/max matching is only viable if all output VTs are the same.
  if (std::equal(ValueVTs.begin(), ValueVTs.end(), ValueVTs.begin())) {
    Value *LHS, *RHS;
    SelectPatternFlavor SPF = matchSelectPattern(const_cast<User*>(&I), LHS, RHS);
    ISD::NodeType Opc = ISD::DELETED_NODE;
    switch (SPF) {
    case SPF_UMAX: Opc = ISD::UMAX; break;
    case SPF_UMIN: Opc = ISD::UMIN; break;
    case SPF_SMAX: Opc = ISD::SMAX; break;
    case SPF_SMIN: Opc = ISD::SMIN; break;
    default: break;
    }

    EVT VT = ValueVTs[0];
    LLVMContext &Ctx = *DAG.getContext();
    auto &TLI = DAG.getTargetLoweringInfo();
    while (TLI.getTypeAction(Ctx, VT) == TargetLoweringBase::TypeSplitVector)
      VT = TLI.getTypeToTransformTo(Ctx, VT);

    if (Opc != ISD::DELETED_NODE && TLI.isOperationLegalOrCustom(Opc, VT) &&
        // If the underlying comparison instruction is used by any other instruction,
        // the consumed instructions won't be destroyed, so it is not profitable
        // to convert to a min/max.
        cast<SelectInst>(&I)->getCondition()->hasOneUse()) {
      OpCode = Opc;
      LHSVal = getValue(LHS);
      RHSVal = getValue(RHS);
      BaseOps = {};
    }
  }

  for (unsigned i = 0; i != NumValues; ++i) {
    SmallVector<SDValue, 3> Ops(BaseOps.begin(), BaseOps.end());
    Ops.push_back(SDValue(LHSVal.getNode(), LHSVal.getResNo() + i));
    Ops.push_back(SDValue(RHSVal.getNode(), RHSVal.getResNo() + i));
    Values[i] = DAG.getNode(OpCode, getCurSDLoc(),
                            LHSVal.getNode()->getValueType(LHSVal.getResNo()+i),
                            Ops);
  }

  setValue(&I, DAG.getNode(ISD::MERGE_VALUES, getCurSDLoc(),
                           DAG.getVTList(ValueVTs), Values));
}

void SelectionDAGBuilder::visitTrunc(const User &I) {
  // TruncInst cannot be a no-op cast because sizeof(src) > sizeof(dest).
  SDValue N = getValue(I.getOperand(0));
  EVT DestVT = DAG.getTargetLoweringInfo().getValueType(DAG.getDataLayout(),
                                                        I.getType());
  setValue(&I, DAG.getNode(ISD::TRUNCATE, getCurSDLoc(), DestVT, N));
}

void SelectionDAGBuilder::visitZExt(const User &I) {
  // ZExt cannot be a no-op cast because sizeof(src) < sizeof(dest).
  // ZExt also can't be a cast to bool for same reason. So, nothing much to do
  SDValue N = getValue(I.getOperand(0));
  EVT DestVT = DAG.getTargetLoweringInfo().getValueType(DAG.getDataLayout(),
                                                        I.getType());
  setValue(&I, DAG.getNode(ISD::ZERO_EXTEND, getCurSDLoc(), DestVT, N));
}

void SelectionDAGBuilder::visitSExt(const User &I) {
  // SExt cannot be a no-op cast because sizeof(src) < sizeof(dest).
  // SExt also can't be a cast to bool for same reason. So, nothing much to do
  SDValue N = getValue(I.getOperand(0));
  EVT DestVT = DAG.getTargetLoweringInfo().getValueType(DAG.getDataLayout(),
                                                        I.getType());
  setValue(&I, DAG.getNode(ISD::SIGN_EXTEND, getCurSDLoc(), DestVT, N));
}

void SelectionDAGBuilder::visitFPTrunc(const User &I) {
  // FPTrunc is never a no-op cast, no need to check
  SDValue N = getValue(I.getOperand(0));
  SDLoc dl = getCurSDLoc();
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  EVT DestVT = TLI.getValueType(DAG.getDataLayout(), I.getType());
  setValue(&I, DAG.getNode(ISD::FP_ROUND, dl, DestVT, N,
                           DAG.getTargetConstant(
                               0, dl, TLI.getPointerTy(DAG.getDataLayout()))));
}

void SelectionDAGBuilder::visitFPExt(const User &I) {
  // FPExt is never a no-op cast, no need to check
  SDValue N = getValue(I.getOperand(0));
  EVT DestVT = DAG.getTargetLoweringInfo().getValueType(DAG.getDataLayout(),
                                                        I.getType());
  setValue(&I, DAG.getNode(ISD::FP_EXTEND, getCurSDLoc(), DestVT, N));
}

void SelectionDAGBuilder::visitFPToUI(const User &I) {
  // FPToUI is never a no-op cast, no need to check
  SDValue N = getValue(I.getOperand(0));
  EVT DestVT = DAG.getTargetLoweringInfo().getValueType(DAG.getDataLayout(),
                                                        I.getType());
  setValue(&I, DAG.getNode(ISD::FP_TO_UINT, getCurSDLoc(), DestVT, N));
}

void SelectionDAGBuilder::visitFPToSI(const User &I) {
  // FPToSI is never a no-op cast, no need to check
  SDValue N = getValue(I.getOperand(0));
  EVT DestVT = DAG.getTargetLoweringInfo().getValueType(DAG.getDataLayout(),
                                                        I.getType());
  setValue(&I, DAG.getNode(ISD::FP_TO_SINT, getCurSDLoc(), DestVT, N));
}

void SelectionDAGBuilder::visitUIToFP(const User &I) {
  // UIToFP is never a no-op cast, no need to check
  SDValue N = getValue(I.getOperand(0));
  EVT DestVT = DAG.getTargetLoweringInfo().getValueType(DAG.getDataLayout(),
                                                        I.getType());
  setValue(&I, DAG.getNode(ISD::UINT_TO_FP, getCurSDLoc(), DestVT, N));
}

void SelectionDAGBuilder::visitSIToFP(const User &I) {
  // SIToFP is never a no-op cast, no need to check
  SDValue N = getValue(I.getOperand(0));
  EVT DestVT = DAG.getTargetLoweringInfo().getValueType(DAG.getDataLayout(),
                                                        I.getType());
  setValue(&I, DAG.getNode(ISD::SINT_TO_FP, getCurSDLoc(), DestVT, N));
}

void SelectionDAGBuilder::visitPtrToInt(const User &I) {
  // What to do depends on the size of the integer and the size of the pointer.
  // We can either truncate, zero extend, or no-op, accordingly.
  SDValue N = getValue(I.getOperand(0));
  EVT DestVT = DAG.getTargetLoweringInfo().getValueType(DAG.getDataLayout(),
                                                        I.getType());
  setValue(&I, DAG.getZExtOrTrunc(N, getCurSDLoc(), DestVT));
}

void SelectionDAGBuilder::visitIntToPtr(const User &I) {
  // What to do depends on the size of the integer and the size of the pointer.
  // We can either truncate, zero extend, or no-op, accordingly.
  SDValue N = getValue(I.getOperand(0));
  EVT DestVT = DAG.getTargetLoweringInfo().getValueType(DAG.getDataLayout(),
                                                        I.getType());
  setValue(&I, DAG.getZExtOrTrunc(N, getCurSDLoc(), DestVT));
}

void SelectionDAGBuilder::visitBitCast(const User &I) {
  SDValue N = getValue(I.getOperand(0));
  SDLoc dl = getCurSDLoc();
  EVT DestVT = DAG.getTargetLoweringInfo().getValueType(DAG.getDataLayout(),
                                                        I.getType());

  // BitCast assures us that source and destination are the same size so this is
  // either a BITCAST or a no-op.
  if (DestVT != N.getValueType())
    setValue(&I, DAG.getNode(ISD::BITCAST, dl,
                             DestVT, N)); // convert types.
  // Check if the original LLVM IR Operand was a ConstantInt, because getValue()
  // might fold any kind of constant expression to an integer constant and that
  // is not what we are looking for. Only regcognize a bitcast of a genuine
  // constant integer as an opaque constant.
  else if(ConstantInt *C = dyn_cast<ConstantInt>(I.getOperand(0)))
    setValue(&I, DAG.getConstant(C->getValue(), dl, DestVT, /*isTarget=*/false,
                                 /*isOpaque*/true));
  else
    setValue(&I, N);            // noop cast.
}

void SelectionDAGBuilder::visitAddrSpaceCast(const User &I) {
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  const Value *SV = I.getOperand(0);
  SDValue N = getValue(SV);
  EVT DestVT = TLI.getValueType(DAG.getDataLayout(), I.getType());

  unsigned SrcAS = SV->getType()->getPointerAddressSpace();
  unsigned DestAS = I.getType()->getPointerAddressSpace();

  if (!TLI.isNoopAddrSpaceCast(SrcAS, DestAS))
    N = DAG.getAddrSpaceCast(getCurSDLoc(), DestVT, N, SrcAS, DestAS);

  setValue(&I, N);
}

void SelectionDAGBuilder::visitInsertElement(const User &I) {
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  SDValue InVec = getValue(I.getOperand(0));
  SDValue InVal = getValue(I.getOperand(1));
  SDValue InIdx = DAG.getSExtOrTrunc(getValue(I.getOperand(2)), getCurSDLoc(),
                                     TLI.getVectorIdxTy(DAG.getDataLayout()));
  setValue(&I, DAG.getNode(ISD::INSERT_VECTOR_ELT, getCurSDLoc(),
                           TLI.getValueType(DAG.getDataLayout(), I.getType()),
                           InVec, InVal, InIdx));
}

void SelectionDAGBuilder::visitExtractElement(const User &I) {
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  SDValue InVec = getValue(I.getOperand(0));
  SDValue InIdx = DAG.getSExtOrTrunc(getValue(I.getOperand(1)), getCurSDLoc(),
                                     TLI.getVectorIdxTy(DAG.getDataLayout()));
  setValue(&I, DAG.getNode(ISD::EXTRACT_VECTOR_ELT, getCurSDLoc(),
                           TLI.getValueType(DAG.getDataLayout(), I.getType()),
                           InVec, InIdx));
}

// Utility for visitShuffleVector - Return true if every element in Mask,
// beginning from position Pos and ending in Pos+Size, falls within the
// specified sequential range [L, L+Pos). or is undef.
static bool isSequentialInRange(const SmallVectorImpl<int> &Mask,
                                unsigned Pos, unsigned Size, int Low) {
  for (unsigned i = Pos, e = Pos+Size; i != e; ++i, ++Low)
    if (Mask[i] >= 0 && Mask[i] != Low)
      return false;
  return true;
}

void SelectionDAGBuilder::visitShuffleVector(const User &I) {
  SDValue Src1 = getValue(I.getOperand(0));
  SDValue Src2 = getValue(I.getOperand(1));

  SmallVector<int, 8> Mask;
  ShuffleVectorInst::getShuffleMask(cast<Constant>(I.getOperand(2)), Mask);
  unsigned MaskNumElts = Mask.size();

  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  EVT VT = TLI.getValueType(DAG.getDataLayout(), I.getType());
  EVT SrcVT = Src1.getValueType();
  unsigned SrcNumElts = SrcVT.getVectorNumElements();

  if (SrcNumElts == MaskNumElts) {
    setValue(&I, DAG.getVectorShuffle(VT, getCurSDLoc(), Src1, Src2,
                                      &Mask[0]));
    return;
  }

  // Normalize the shuffle vector since mask and vector length don't match.
  if (SrcNumElts < MaskNumElts && MaskNumElts % SrcNumElts == 0) {
    // Mask is longer than the source vectors and is a multiple of the source
    // vectors.  We can use concatenate vector to make the mask and vectors
    // lengths match.
    if (SrcNumElts*2 == MaskNumElts) {
      // First check for Src1 in low and Src2 in high
      if (isSequentialInRange(Mask, 0, SrcNumElts, 0) &&
          isSequentialInRange(Mask, SrcNumElts, SrcNumElts, SrcNumElts)) {
        // The shuffle is concatenating two vectors together.
        setValue(&I, DAG.getNode(ISD::CONCAT_VECTORS, getCurSDLoc(),
                                 VT, Src1, Src2));
        return;
      }
      // Then check for Src2 in low and Src1 in high
      if (isSequentialInRange(Mask, 0, SrcNumElts, SrcNumElts) &&
          isSequentialInRange(Mask, SrcNumElts, SrcNumElts, 0)) {
        // The shuffle is concatenating two vectors together.
        setValue(&I, DAG.getNode(ISD::CONCAT_VECTORS, getCurSDLoc(),
                                 VT, Src2, Src1));
        return;
      }
    }

    // Pad both vectors with undefs to make them the same length as the mask.
    unsigned NumConcat = MaskNumElts / SrcNumElts;
    bool Src1U = Src1.getOpcode() == ISD::UNDEF;
    bool Src2U = Src2.getOpcode() == ISD::UNDEF;
    SDValue UndefVal = DAG.getUNDEF(SrcVT);

    SmallVector<SDValue, 8> MOps1(NumConcat, UndefVal);
    SmallVector<SDValue, 8> MOps2(NumConcat, UndefVal);
    MOps1[0] = Src1;
    MOps2[0] = Src2;

    Src1 = Src1U ? DAG.getUNDEF(VT) : DAG.getNode(ISD::CONCAT_VECTORS,
                                                  getCurSDLoc(), VT, MOps1);
    Src2 = Src2U ? DAG.getUNDEF(VT) : DAG.getNode(ISD::CONCAT_VECTORS,
                                                  getCurSDLoc(), VT, MOps2);

    // Readjust mask for new input vector length.
    SmallVector<int, 8> MappedOps;
    for (unsigned i = 0; i != MaskNumElts; ++i) {
      int Idx = Mask[i];
      if (Idx >= (int)SrcNumElts)
        Idx -= SrcNumElts - MaskNumElts;
      MappedOps.push_back(Idx);
    }

    setValue(&I, DAG.getVectorShuffle(VT, getCurSDLoc(), Src1, Src2,
                                      &MappedOps[0]));
    return;
  }

  if (SrcNumElts > MaskNumElts) {
    // Analyze the access pattern of the vector to see if we can extract
    // two subvectors and do the shuffle. The analysis is done by calculating
    // the range of elements the mask access on both vectors.
    int MinRange[2] = { static_cast<int>(SrcNumElts),
                        static_cast<int>(SrcNumElts)};
    int MaxRange[2] = {-1, -1};

    for (unsigned i = 0; i != MaskNumElts; ++i) {
      int Idx = Mask[i];
      unsigned Input = 0;
      if (Idx < 0)
        continue;

      if (Idx >= (int)SrcNumElts) {
        Input = 1;
        Idx -= SrcNumElts;
      }
      if (Idx > MaxRange[Input])
        MaxRange[Input] = Idx;
      if (Idx < MinRange[Input])
        MinRange[Input] = Idx;
    }

    // Check if the access is smaller than the vector size and can we find
    // a reasonable extract index.
    int RangeUse[2] = { -1, -1 };  // 0 = Unused, 1 = Extract, -1 = Can not
                                   // Extract.
    int StartIdx[2];  // StartIdx to extract from
    for (unsigned Input = 0; Input < 2; ++Input) {
      if (MinRange[Input] >= (int)SrcNumElts && MaxRange[Input] < 0) {
        RangeUse[Input] = 0; // Unused
        StartIdx[Input] = 0;
        continue;
      }

      // Find a good start index that is a multiple of the mask length. Then
      // see if the rest of the elements are in range.
      StartIdx[Input] = (MinRange[Input]/MaskNumElts)*MaskNumElts;
      if (MaxRange[Input] - StartIdx[Input] < (int)MaskNumElts &&
          StartIdx[Input] + MaskNumElts <= SrcNumElts)
        RangeUse[Input] = 1; // Extract from a multiple of the mask length.
    }

    if (RangeUse[0] == 0 && RangeUse[1] == 0) {
      setValue(&I, DAG.getUNDEF(VT)); // Vectors are not used.
      return;
    }
    if (RangeUse[0] >= 0 && RangeUse[1] >= 0) {
      // Extract appropriate subvector and generate a vector shuffle
      for (unsigned Input = 0; Input < 2; ++Input) {
        SDValue &Src = Input == 0 ? Src1 : Src2;
        if (RangeUse[Input] == 0)
          Src = DAG.getUNDEF(VT);
        else {
          SDLoc dl = getCurSDLoc();
          Src = DAG.getNode(
              ISD::EXTRACT_SUBVECTOR, dl, VT, Src,
              DAG.getConstant(StartIdx[Input], dl,
                              TLI.getVectorIdxTy(DAG.getDataLayout())));
        }
      }

      // Calculate new mask.
      SmallVector<int, 8> MappedOps;
      for (unsigned i = 0; i != MaskNumElts; ++i) {
        int Idx = Mask[i];
        if (Idx >= 0) {
          if (Idx < (int)SrcNumElts)
            Idx -= StartIdx[0];
          else
            Idx -= SrcNumElts + StartIdx[1] - MaskNumElts;
        }
        MappedOps.push_back(Idx);
      }

      setValue(&I, DAG.getVectorShuffle(VT, getCurSDLoc(), Src1, Src2,
                                        &MappedOps[0]));
      return;
    }
  }

  // We can't use either concat vectors or extract subvectors so fall back to
  // replacing the shuffle with extract and build vector.
  // to insert and build vector.
  EVT EltVT = VT.getVectorElementType();
  EVT IdxVT = TLI.getVectorIdxTy(DAG.getDataLayout());
  SDLoc dl = getCurSDLoc();
  SmallVector<SDValue,8> Ops;
  for (unsigned i = 0; i != MaskNumElts; ++i) {
    int Idx = Mask[i];
    SDValue Res;

    if (Idx < 0) {
      Res = DAG.getUNDEF(EltVT);
    } else {
      SDValue &Src = Idx < (int)SrcNumElts ? Src1 : Src2;
      if (Idx >= (int)SrcNumElts) Idx -= SrcNumElts;

      Res = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl,
                        EltVT, Src, DAG.getConstant(Idx, dl, IdxVT));
    }

    Ops.push_back(Res);
  }

  setValue(&I, DAG.getNode(ISD::BUILD_VECTOR, dl, VT, Ops));
}

void SelectionDAGBuilder::visitInsertValue(const InsertValueInst &I) {
  const Value *Op0 = I.getOperand(0);
  const Value *Op1 = I.getOperand(1);
  Type *AggTy = I.getType();
  Type *ValTy = Op1->getType();
  bool IntoUndef = isa<UndefValue>(Op0);
  bool FromUndef = isa<UndefValue>(Op1);

  unsigned LinearIndex = ComputeLinearIndex(AggTy, I.getIndices());

  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  SmallVector<EVT, 4> AggValueVTs;
  ComputeValueVTs(TLI, DAG.getDataLayout(), AggTy, AggValueVTs);
  SmallVector<EVT, 4> ValValueVTs;
  ComputeValueVTs(TLI, DAG.getDataLayout(), ValTy, ValValueVTs);

  unsigned NumAggValues = AggValueVTs.size();
  unsigned NumValValues = ValValueVTs.size();
  SmallVector<SDValue, 4> Values(NumAggValues);

  // Ignore an insertvalue that produces an empty object
  if (!NumAggValues) {
    setValue(&I, DAG.getUNDEF(MVT(MVT::Other)));
    return;
  }

  SDValue Agg = getValue(Op0);
  unsigned i = 0;
  // Copy the beginning value(s) from the original aggregate.
  for (; i != LinearIndex; ++i)
    Values[i] = IntoUndef ? DAG.getUNDEF(AggValueVTs[i]) :
                SDValue(Agg.getNode(), Agg.getResNo() + i);
  // Copy values from the inserted value(s).
  if (NumValValues) {
    SDValue Val = getValue(Op1);
    for (; i != LinearIndex + NumValValues; ++i)
      Values[i] = FromUndef ? DAG.getUNDEF(AggValueVTs[i]) :
                  SDValue(Val.getNode(), Val.getResNo() + i - LinearIndex);
  }
  // Copy remaining value(s) from the original aggregate.
  for (; i != NumAggValues; ++i)
    Values[i] = IntoUndef ? DAG.getUNDEF(AggValueVTs[i]) :
                SDValue(Agg.getNode(), Agg.getResNo() + i);

  setValue(&I, DAG.getNode(ISD::MERGE_VALUES, getCurSDLoc(),
                           DAG.getVTList(AggValueVTs), Values));
}

void SelectionDAGBuilder::visitExtractValue(const ExtractValueInst &I) {
  const Value *Op0 = I.getOperand(0);
  Type *AggTy = Op0->getType();
  Type *ValTy = I.getType();
  bool OutOfUndef = isa<UndefValue>(Op0);

  unsigned LinearIndex = ComputeLinearIndex(AggTy, I.getIndices());

  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  SmallVector<EVT, 4> ValValueVTs;
  ComputeValueVTs(TLI, DAG.getDataLayout(), ValTy, ValValueVTs);

  unsigned NumValValues = ValValueVTs.size();

  // Ignore a extractvalue that produces an empty object
  if (!NumValValues) {
    setValue(&I, DAG.getUNDEF(MVT(MVT::Other)));
    return;
  }

  SmallVector<SDValue, 4> Values(NumValValues);

  SDValue Agg = getValue(Op0);
  // Copy out the selected value(s).
  for (unsigned i = LinearIndex; i != LinearIndex + NumValValues; ++i)
    Values[i - LinearIndex] =
      OutOfUndef ?
        DAG.getUNDEF(Agg.getNode()->getValueType(Agg.getResNo() + i)) :
        SDValue(Agg.getNode(), Agg.getResNo() + i);

  setValue(&I, DAG.getNode(ISD::MERGE_VALUES, getCurSDLoc(),
                           DAG.getVTList(ValValueVTs), Values));
}

void SelectionDAGBuilder::visitGetElementPtr(const User &I) {
  Value *Op0 = I.getOperand(0);
  // Note that the pointer operand may be a vector of pointers. Take the scalar
  // element which holds a pointer.
  Type *Ty = Op0->getType()->getScalarType();
  unsigned AS = Ty->getPointerAddressSpace();
  SDValue N = getValue(Op0);
  SDLoc dl = getCurSDLoc();

  // Normalize Vector GEP - all scalar operands should be converted to the
  // splat vector.
  unsigned VectorWidth = I.getType()->isVectorTy() ?
    cast<VectorType>(I.getType())->getVectorNumElements() : 0;

  if (VectorWidth && !N.getValueType().isVector()) {
    MVT VT = MVT::getVectorVT(N.getValueType().getSimpleVT(), VectorWidth);
    SmallVector<SDValue, 16> Ops(VectorWidth, N);
    N = DAG.getNode(ISD::BUILD_VECTOR, dl, VT, Ops);
  }
  for (GetElementPtrInst::const_op_iterator OI = I.op_begin()+1, E = I.op_end();
       OI != E; ++OI) {
    const Value *Idx = *OI;
    if (StructType *StTy = dyn_cast<StructType>(Ty)) {
      unsigned Field = cast<Constant>(Idx)->getUniqueInteger().getZExtValue();
      if (Field) {
        // N = N + Offset
        uint64_t Offset = DL->getStructLayout(StTy)->getElementOffset(Field);
        N = DAG.getNode(ISD::ADD, dl, N.getValueType(), N,
                        DAG.getConstant(Offset, dl, N.getValueType()));
      }

      Ty = StTy->getElementType(Field);
    } else {
      Ty = cast<SequentialType>(Ty)->getElementType();
      MVT PtrTy =
          DAG.getTargetLoweringInfo().getPointerTy(DAG.getDataLayout(), AS);
      unsigned PtrSize = PtrTy.getSizeInBits();
      APInt ElementSize(PtrSize, DL->getTypeAllocSize(Ty));

      // If this is a scalar constant or a splat vector of constants,
      // handle it quickly.
      const auto *CI = dyn_cast<ConstantInt>(Idx);
      if (!CI && isa<ConstantDataVector>(Idx) &&
          cast<ConstantDataVector>(Idx)->getSplatValue())
        CI = cast<ConstantInt>(cast<ConstantDataVector>(Idx)->getSplatValue());

      if (CI) {
        if (CI->isZero())
          continue;
        APInt Offs = ElementSize * CI->getValue().sextOrTrunc(PtrSize);
        SDValue OffsVal = VectorWidth ?
          DAG.getConstant(Offs, dl, MVT::getVectorVT(PtrTy, VectorWidth)) :
          DAG.getConstant(Offs, dl, PtrTy);
        N = DAG.getNode(ISD::ADD, dl, N.getValueType(), N, OffsVal);
        continue;
      }

      // N = N + Idx * ElementSize;
      SDValue IdxN = getValue(Idx);

      if (!IdxN.getValueType().isVector() && VectorWidth) {
        MVT VT = MVT::getVectorVT(IdxN.getValueType().getSimpleVT(), VectorWidth);
        SmallVector<SDValue, 16> Ops(VectorWidth, IdxN);
        IdxN = DAG.getNode(ISD::BUILD_VECTOR, dl, VT, Ops);      
      }
      // If the index is smaller or larger than intptr_t, truncate or extend
      // it.
      IdxN = DAG.getSExtOrTrunc(IdxN, dl, N.getValueType());

      // If this is a multiply by a power of two, turn it into a shl
      // immediately.  This is a very common case.
      if (ElementSize != 1) {
        if (ElementSize.isPowerOf2()) {
          unsigned Amt = ElementSize.logBase2();
          IdxN = DAG.getNode(ISD::SHL, dl,
                             N.getValueType(), IdxN,
                             DAG.getConstant(Amt, dl, IdxN.getValueType()));
        } else {
          SDValue Scale = DAG.getConstant(ElementSize, dl, IdxN.getValueType());
          IdxN = DAG.getNode(ISD::MUL, dl,
                             N.getValueType(), IdxN, Scale);
        }
      }

      N = DAG.getNode(ISD::ADD, dl,
                      N.getValueType(), N, IdxN);
    }
  }

  setValue(&I, N);
}

void SelectionDAGBuilder::visitAlloca(const AllocaInst &I) {
  // If this is a fixed sized alloca in the entry block of the function,
  // allocate it statically on the stack.
  if (FuncInfo.StaticAllocaMap.count(&I))
    return;   // getValue will auto-populate this.

  SDLoc dl = getCurSDLoc();
  Type *Ty = I.getAllocatedType();
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  auto &DL = DAG.getDataLayout();
  uint64_t TySize = DL.getTypeAllocSize(Ty);
  unsigned Align =
      std::max((unsigned)DL.getPrefTypeAlignment(Ty), I.getAlignment());

  SDValue AllocSize = getValue(I.getArraySize());

  EVT IntPtr = TLI.getPointerTy(DAG.getDataLayout());
  if (AllocSize.getValueType() != IntPtr)
    AllocSize = DAG.getZExtOrTrunc(AllocSize, dl, IntPtr);

  AllocSize = DAG.getNode(ISD::MUL, dl, IntPtr,
                          AllocSize,
                          DAG.getConstant(TySize, dl, IntPtr));

  // Handle alignment.  If the requested alignment is less than or equal to
  // the stack alignment, ignore it.  If the size is greater than or equal to
  // the stack alignment, we note this in the DYNAMIC_STACKALLOC node.
  unsigned StackAlign =
      DAG.getSubtarget().getFrameLowering()->getStackAlignment();
  if (Align <= StackAlign)
    Align = 0;

  // Round the size of the allocation up to the stack alignment size
  // by add SA-1 to the size.
  AllocSize = DAG.getNode(ISD::ADD, dl,
                          AllocSize.getValueType(), AllocSize,
                          DAG.getIntPtrConstant(StackAlign - 1, dl));

  // Mask out the low bits for alignment purposes.
  AllocSize = DAG.getNode(ISD::AND, dl,
                          AllocSize.getValueType(), AllocSize,
                          DAG.getIntPtrConstant(~(uint64_t)(StackAlign - 1),
                                                dl));

  SDValue Ops[] = { getRoot(), AllocSize, DAG.getIntPtrConstant(Align, dl) };
  SDVTList VTs = DAG.getVTList(AllocSize.getValueType(), MVT::Other);
  SDValue DSA = DAG.getNode(ISD::DYNAMIC_STACKALLOC, dl, VTs, Ops);
  setValue(&I, DSA);
  DAG.setRoot(DSA.getValue(1));

  assert(FuncInfo.MF->getFrameInfo()->hasVarSizedObjects());
}

void SelectionDAGBuilder::visitLoad(const LoadInst &I) {
  if (I.isAtomic())
    return visitAtomicLoad(I);

  const Value *SV = I.getOperand(0);
  SDValue Ptr = getValue(SV);

  Type *Ty = I.getType();

  bool isVolatile = I.isVolatile();
  bool isNonTemporal = I.getMetadata(LLVMContext::MD_nontemporal) != nullptr;

  // The IR notion of invariant_load only guarantees that all *non-faulting*
  // invariant loads result in the same value.  The MI notion of invariant load
  // guarantees that the load can be legally moved to any location within its
  // containing function.  The MI notion of invariant_load is stronger than the
  // IR notion of invariant_load -- an MI invariant_load is an IR invariant_load
  // with a guarantee that the location being loaded from is dereferenceable
  // throughout the function's lifetime.

  bool isInvariant = I.getMetadata(LLVMContext::MD_invariant_load) != nullptr &&
    isDereferenceablePointer(SV, *DAG.getTarget().getDataLayout());
  unsigned Alignment = I.getAlignment();

  AAMDNodes AAInfo;
  I.getAAMetadata(AAInfo);
  const MDNode *Ranges = I.getMetadata(LLVMContext::MD_range);

  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  SmallVector<EVT, 4> ValueVTs;
  SmallVector<uint64_t, 4> Offsets;
  ComputeValueVTs(TLI, DAG.getDataLayout(), Ty, ValueVTs, &Offsets);
  unsigned NumValues = ValueVTs.size();
  if (NumValues == 0)
    return;

  SDValue Root;
  bool ConstantMemory = false;
  if (isVolatile || NumValues > MaxParallelChains)
    // Serialize volatile loads with other side effects.
    Root = getRoot();
  else if (AA->pointsToConstantMemory(
               MemoryLocation(SV, AA->getTypeStoreSize(Ty), AAInfo))) {
    // Do not serialize (non-volatile) loads of constant memory with anything.
    Root = DAG.getEntryNode();
    ConstantMemory = true;
  } else {
    // Do not serialize non-volatile loads against each other.
    Root = DAG.getRoot();
  }

  SDLoc dl = getCurSDLoc();

  if (isVolatile)
    Root = TLI.prepareVolatileOrAtomicLoad(Root, dl, DAG);

  SmallVector<SDValue, 4> Values(NumValues);
  SmallVector<SDValue, 4> Chains(std::min(MaxParallelChains, NumValues));
  EVT PtrVT = Ptr.getValueType();
  unsigned ChainI = 0;
  for (unsigned i = 0; i != NumValues; ++i, ++ChainI) {
    // Serializing loads here may result in excessive register pressure, and
    // TokenFactor places arbitrary choke points on the scheduler. SD scheduling
    // could recover a bit by hoisting nodes upward in the chain by recognizing
    // they are side-effect free or do not alias. The optimizer should really
    // avoid this case by converting large object/array copies to llvm.memcpy
    // (MaxParallelChains should always remain as failsafe).
    if (ChainI == MaxParallelChains) {
      assert(PendingLoads.empty() && "PendingLoads must be serialized first");
      SDValue Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other,
                                  makeArrayRef(Chains.data(), ChainI));
      Root = Chain;
      ChainI = 0;
    }
    SDValue A = DAG.getNode(ISD::ADD, dl,
                            PtrVT, Ptr,
                            DAG.getConstant(Offsets[i], dl, PtrVT));
    SDValue L = DAG.getLoad(ValueVTs[i], dl, Root,
                            A, MachinePointerInfo(SV, Offsets[i]), isVolatile,
                            isNonTemporal, isInvariant, Alignment, AAInfo,
                            Ranges);

    Values[i] = L;
    Chains[ChainI] = L.getValue(1);
  }

  if (!ConstantMemory) {
    SDValue Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other,
                                makeArrayRef(Chains.data(), ChainI));
    if (isVolatile)
      DAG.setRoot(Chain);
    else
      PendingLoads.push_back(Chain);
  }

  setValue(&I, DAG.getNode(ISD::MERGE_VALUES, dl,
                           DAG.getVTList(ValueVTs), Values));
}

void SelectionDAGBuilder::visitStore(const StoreInst &I) {
  if (I.isAtomic())
    return visitAtomicStore(I);

  const Value *SrcV = I.getOperand(0);
  const Value *PtrV = I.getOperand(1);

  SmallVector<EVT, 4> ValueVTs;
  SmallVector<uint64_t, 4> Offsets;
  ComputeValueVTs(DAG.getTargetLoweringInfo(), DAG.getDataLayout(),
                  SrcV->getType(), ValueVTs, &Offsets);
  unsigned NumValues = ValueVTs.size();
  if (NumValues == 0)
    return;

  // Get the lowered operands. Note that we do this after
  // checking if NumResults is zero, because with zero results
  // the operands won't have values in the map.
  SDValue Src = getValue(SrcV);
  SDValue Ptr = getValue(PtrV);

  SDValue Root = getRoot();
  SmallVector<SDValue, 4> Chains(std::min(MaxParallelChains, NumValues));
  EVT PtrVT = Ptr.getValueType();
  bool isVolatile = I.isVolatile();
  bool isNonTemporal = I.getMetadata(LLVMContext::MD_nontemporal) != nullptr;
  unsigned Alignment = I.getAlignment();
  SDLoc dl = getCurSDLoc();

  AAMDNodes AAInfo;
  I.getAAMetadata(AAInfo);

  unsigned ChainI = 0;
  for (unsigned i = 0; i != NumValues; ++i, ++ChainI) {
    // See visitLoad comments.
    if (ChainI == MaxParallelChains) {
      SDValue Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other,
                                  makeArrayRef(Chains.data(), ChainI));
      Root = Chain;
      ChainI = 0;
    }
    SDValue Add = DAG.getNode(ISD::ADD, dl, PtrVT, Ptr,
                              DAG.getConstant(Offsets[i], dl, PtrVT));
    SDValue St = DAG.getStore(Root, dl,
                              SDValue(Src.getNode(), Src.getResNo() + i),
                              Add, MachinePointerInfo(PtrV, Offsets[i]),
                              isVolatile, isNonTemporal, Alignment, AAInfo);
    Chains[ChainI] = St;
  }

  SDValue StoreNode = DAG.getNode(ISD::TokenFactor, dl, MVT::Other,
                                  makeArrayRef(Chains.data(), ChainI));
  DAG.setRoot(StoreNode);
}

void SelectionDAGBuilder::visitMaskedStore(const CallInst &I) {
  SDLoc sdl = getCurSDLoc();

  // llvm.masked.store.*(Src0, Ptr, alignemt, Mask)
  Value  *PtrOperand = I.getArgOperand(1);
  SDValue Ptr = getValue(PtrOperand);
  SDValue Src0 = getValue(I.getArgOperand(0));
  SDValue Mask = getValue(I.getArgOperand(3));
  EVT VT = Src0.getValueType();
  unsigned Alignment = (cast<ConstantInt>(I.getArgOperand(2)))->getZExtValue();
  if (!Alignment)
    Alignment = DAG.getEVTAlignment(VT);

  AAMDNodes AAInfo;
  I.getAAMetadata(AAInfo);

  MachineMemOperand *MMO =
    DAG.getMachineFunction().
    getMachineMemOperand(MachinePointerInfo(PtrOperand),
                          MachineMemOperand::MOStore,  VT.getStoreSize(),
                          Alignment, AAInfo);
  SDValue StoreNode = DAG.getMaskedStore(getRoot(), sdl, Src0, Ptr, Mask, VT,
                                         MMO, false);
  DAG.setRoot(StoreNode);
  setValue(&I, StoreNode);
}

// Gather/scatter receive a vector of pointers.
// This vector of pointers may be represented as a base pointer + vector of 
// indices, it depends on GEP and instruction preceeding GEP
// that calculates indices
static bool getUniformBase(Value *& Ptr, SDValue& Base, SDValue& Index,
                           SelectionDAGBuilder* SDB) {

  assert (Ptr->getType()->isVectorTy() && "Uexpected pointer type");
  GetElementPtrInst *Gep = dyn_cast<GetElementPtrInst>(Ptr);
  if (!Gep || Gep->getNumOperands() > 2)
    return false;
  ShuffleVectorInst *ShuffleInst = 
    dyn_cast<ShuffleVectorInst>(Gep->getPointerOperand());
  if (!ShuffleInst || !ShuffleInst->getMask()->isNullValue() ||
      cast<Instruction>(ShuffleInst->getOperand(0))->getOpcode() !=
      Instruction::InsertElement)
    return false;

  Ptr = cast<InsertElementInst>(ShuffleInst->getOperand(0))->getOperand(1);

  SelectionDAG& DAG = SDB->DAG;
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  // Check is the Ptr is inside current basic block
  // If not, look for the shuffle instruction
  if (SDB->findValue(Ptr))
    Base = SDB->getValue(Ptr);
  else if (SDB->findValue(ShuffleInst)) {
    SDValue ShuffleNode = SDB->getValue(ShuffleInst);
    SDLoc sdl = ShuffleNode;
    Base = DAG.getNode(
        ISD::EXTRACT_VECTOR_ELT, sdl,
        ShuffleNode.getValueType().getScalarType(), ShuffleNode,
        DAG.getConstant(0, sdl, TLI.getVectorIdxTy(DAG.getDataLayout())));
    SDB->setValue(Ptr, Base);
  }
  else
    return false;

  Value *IndexVal = Gep->getOperand(1);
  if (SDB->findValue(IndexVal)) {
    Index = SDB->getValue(IndexVal);

    if (SExtInst* Sext = dyn_cast<SExtInst>(IndexVal)) {
      IndexVal = Sext->getOperand(0);
      if (SDB->findValue(IndexVal))
        Index = SDB->getValue(IndexVal);
    }
    return true;
  }
  return false;
}

void SelectionDAGBuilder::visitMaskedScatter(const CallInst &I) {
  SDLoc sdl = getCurSDLoc();

  // llvm.masked.scatter.*(Src0, Ptrs, alignemt, Mask)
  Value  *Ptr = I.getArgOperand(1);
  SDValue Src0 = getValue(I.getArgOperand(0));
  SDValue Mask = getValue(I.getArgOperand(3));
  EVT VT = Src0.getValueType();
  unsigned Alignment = (cast<ConstantInt>(I.getArgOperand(2)))->getZExtValue();
  if (!Alignment)
    Alignment = DAG.getEVTAlignment(VT);
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();

  AAMDNodes AAInfo;
  I.getAAMetadata(AAInfo);

  SDValue Base;
  SDValue Index;
  Value *BasePtr = Ptr;
  bool UniformBase = getUniformBase(BasePtr, Base, Index, this);

  Value *MemOpBasePtr = UniformBase ? BasePtr : nullptr;
  MachineMemOperand *MMO = DAG.getMachineFunction().
    getMachineMemOperand(MachinePointerInfo(MemOpBasePtr),
                         MachineMemOperand::MOStore,  VT.getStoreSize(),
                         Alignment, AAInfo);
  if (!UniformBase) {
    Base = DAG.getTargetConstant(0, sdl, TLI.getPointerTy(DAG.getDataLayout()));
    Index = getValue(Ptr);
  }
  SDValue Ops[] = { getRoot(), Src0, Mask, Base, Index };
  SDValue Scatter = DAG.getMaskedScatter(DAG.getVTList(MVT::Other), VT, sdl,
                                         Ops, MMO);
  DAG.setRoot(Scatter);
  setValue(&I, Scatter);
}

void SelectionDAGBuilder::visitMaskedLoad(const CallInst &I) {
  SDLoc sdl = getCurSDLoc();

  // @llvm.masked.load.*(Ptr, alignment, Mask, Src0)
  Value  *PtrOperand = I.getArgOperand(0);
  SDValue Ptr = getValue(PtrOperand);
  SDValue Src0 = getValue(I.getArgOperand(3));
  SDValue Mask = getValue(I.getArgOperand(2));

  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  EVT VT = TLI.getValueType(DAG.getDataLayout(), I.getType());
  unsigned Alignment = (cast<ConstantInt>(I.getArgOperand(1)))->getZExtValue();
  if (!Alignment)
    Alignment = DAG.getEVTAlignment(VT);

  AAMDNodes AAInfo;
  I.getAAMetadata(AAInfo);
  const MDNode *Ranges = I.getMetadata(LLVMContext::MD_range);

  SDValue InChain = DAG.getRoot();
  if (AA->pointsToConstantMemory(MemoryLocation(
          PtrOperand, AA->getTypeStoreSize(I.getType()), AAInfo))) {
    // Do not serialize (non-volatile) loads of constant memory with anything.
    InChain = DAG.getEntryNode();
  }

  MachineMemOperand *MMO =
    DAG.getMachineFunction().
    getMachineMemOperand(MachinePointerInfo(PtrOperand),
                          MachineMemOperand::MOLoad,  VT.getStoreSize(),
                          Alignment, AAInfo, Ranges);

  SDValue Load = DAG.getMaskedLoad(VT, sdl, InChain, Ptr, Mask, Src0, VT, MMO,
                                   ISD::NON_EXTLOAD);
  SDValue OutChain = Load.getValue(1);
  DAG.setRoot(OutChain);
  setValue(&I, Load);
}

void SelectionDAGBuilder::visitMaskedGather(const CallInst &I) {
  SDLoc sdl = getCurSDLoc();

  // @llvm.masked.gather.*(Ptrs, alignment, Mask, Src0)
  Value  *Ptr = I.getArgOperand(0);
  SDValue Src0 = getValue(I.getArgOperand(3));
  SDValue Mask = getValue(I.getArgOperand(2));

  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  EVT VT = TLI.getValueType(DAG.getDataLayout(), I.getType());
  unsigned Alignment = (cast<ConstantInt>(I.getArgOperand(1)))->getZExtValue();
  if (!Alignment)
    Alignment = DAG.getEVTAlignment(VT);

  AAMDNodes AAInfo;
  I.getAAMetadata(AAInfo);
  const MDNode *Ranges = I.getMetadata(LLVMContext::MD_range);

  SDValue Root = DAG.getRoot();
  SDValue Base;
  SDValue Index;
  Value *BasePtr = Ptr;
  bool UniformBase = getUniformBase(BasePtr, Base, Index, this);
  bool ConstantMemory = false;
  if (UniformBase &&
      AA->pointsToConstantMemory(
          MemoryLocation(BasePtr, AA->getTypeStoreSize(I.getType()), AAInfo))) {
    // Do not serialize (non-volatile) loads of constant memory with anything.
    Root = DAG.getEntryNode();
    ConstantMemory = true;
  }

  MachineMemOperand *MMO =
    DAG.getMachineFunction().
    getMachineMemOperand(MachinePointerInfo(UniformBase ? BasePtr : nullptr),
                         MachineMemOperand::MOLoad,  VT.getStoreSize(),
                         Alignment, AAInfo, Ranges);

  if (!UniformBase) {
    Base = DAG.getTargetConstant(0, sdl, TLI.getPointerTy(DAG.getDataLayout()));
    Index = getValue(Ptr);
  }
  SDValue Ops[] = { Root, Src0, Mask, Base, Index };
  SDValue Gather = DAG.getMaskedGather(DAG.getVTList(VT, MVT::Other), VT, sdl,
                                       Ops, MMO);

  SDValue OutChain = Gather.getValue(1);
  if (!ConstantMemory)
    PendingLoads.push_back(OutChain);
  setValue(&I, Gather);
}

void SelectionDAGBuilder::visitAtomicCmpXchg(const AtomicCmpXchgInst &I) {
  SDLoc dl = getCurSDLoc();
  AtomicOrdering SuccessOrder = I.getSuccessOrdering();
  AtomicOrdering FailureOrder = I.getFailureOrdering();
  SynchronizationScope Scope = I.getSynchScope();

  SDValue InChain = getRoot();

  MVT MemVT = getValue(I.getCompareOperand()).getSimpleValueType();
  SDVTList VTs = DAG.getVTList(MemVT, MVT::i1, MVT::Other);
  SDValue L = DAG.getAtomicCmpSwap(
      ISD::ATOMIC_CMP_SWAP_WITH_SUCCESS, dl, MemVT, VTs, InChain,
      getValue(I.getPointerOperand()), getValue(I.getCompareOperand()),
      getValue(I.getNewValOperand()), MachinePointerInfo(I.getPointerOperand()),
      /*Alignment=*/ 0, SuccessOrder, FailureOrder, Scope);

  SDValue OutChain = L.getValue(2);

  setValue(&I, L);
  DAG.setRoot(OutChain);
}

void SelectionDAGBuilder::visitAtomicRMW(const AtomicRMWInst &I) {
  SDLoc dl = getCurSDLoc();
  ISD::NodeType NT;
  switch (I.getOperation()) {
  default: llvm_unreachable("Unknown atomicrmw operation");
  case AtomicRMWInst::Xchg: NT = ISD::ATOMIC_SWAP; break;
  case AtomicRMWInst::Add:  NT = ISD::ATOMIC_LOAD_ADD; break;
  case AtomicRMWInst::Sub:  NT = ISD::ATOMIC_LOAD_SUB; break;
  case AtomicRMWInst::And:  NT = ISD::ATOMIC_LOAD_AND; break;
  case AtomicRMWInst::Nand: NT = ISD::ATOMIC_LOAD_NAND; break;
  case AtomicRMWInst::Or:   NT = ISD::ATOMIC_LOAD_OR; break;
  case AtomicRMWInst::Xor:  NT = ISD::ATOMIC_LOAD_XOR; break;
  case AtomicRMWInst::Max:  NT = ISD::ATOMIC_LOAD_MAX; break;
  case AtomicRMWInst::Min:  NT = ISD::ATOMIC_LOAD_MIN; break;
  case AtomicRMWInst::UMax: NT = ISD::ATOMIC_LOAD_UMAX; break;
  case AtomicRMWInst::UMin: NT = ISD::ATOMIC_LOAD_UMIN; break;
  }
  AtomicOrdering Order = I.getOrdering();
  SynchronizationScope Scope = I.getSynchScope();

  SDValue InChain = getRoot();

  SDValue L =
    DAG.getAtomic(NT, dl,
                  getValue(I.getValOperand()).getSimpleValueType(),
                  InChain,
                  getValue(I.getPointerOperand()),
                  getValue(I.getValOperand()),
                  I.getPointerOperand(),
                  /* Alignment=*/ 0, Order, Scope);

  SDValue OutChain = L.getValue(1);

  setValue(&I, L);
  DAG.setRoot(OutChain);
}

void SelectionDAGBuilder::visitFence(const FenceInst &I) {
  SDLoc dl = getCurSDLoc();
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  SDValue Ops[3];
  Ops[0] = getRoot();
  Ops[1] = DAG.getConstant(I.getOrdering(), dl,
                           TLI.getPointerTy(DAG.getDataLayout()));
  Ops[2] = DAG.getConstant(I.getSynchScope(), dl,
                           TLI.getPointerTy(DAG.getDataLayout()));
  DAG.setRoot(DAG.getNode(ISD::ATOMIC_FENCE, dl, MVT::Other, Ops));
}

void SelectionDAGBuilder::visitAtomicLoad(const LoadInst &I) {
  SDLoc dl = getCurSDLoc();
  AtomicOrdering Order = I.getOrdering();
  SynchronizationScope Scope = I.getSynchScope();

  SDValue InChain = getRoot();

  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  EVT VT = TLI.getValueType(DAG.getDataLayout(), I.getType());

  if (I.getAlignment() < VT.getSizeInBits() / 8)
    report_fatal_error("Cannot generate unaligned atomic load");

  MachineMemOperand *MMO =
      DAG.getMachineFunction().
      getMachineMemOperand(MachinePointerInfo(I.getPointerOperand()),
                           MachineMemOperand::MOVolatile |
                           MachineMemOperand::MOLoad,
                           VT.getStoreSize(),
                           I.getAlignment() ? I.getAlignment() :
                                              DAG.getEVTAlignment(VT));

  InChain = TLI.prepareVolatileOrAtomicLoad(InChain, dl, DAG);
  SDValue L =
      DAG.getAtomic(ISD::ATOMIC_LOAD, dl, VT, VT, InChain,
                    getValue(I.getPointerOperand()), MMO,
                    Order, Scope);

  SDValue OutChain = L.getValue(1);

  setValue(&I, L);
  DAG.setRoot(OutChain);
}

void SelectionDAGBuilder::visitAtomicStore(const StoreInst &I) {
  SDLoc dl = getCurSDLoc();

  AtomicOrdering Order = I.getOrdering();
  SynchronizationScope Scope = I.getSynchScope();

  SDValue InChain = getRoot();

  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  EVT VT =
      TLI.getValueType(DAG.getDataLayout(), I.getValueOperand()->getType());

  if (I.getAlignment() < VT.getSizeInBits() / 8)
    report_fatal_error("Cannot generate unaligned atomic store");

  SDValue OutChain =
    DAG.getAtomic(ISD::ATOMIC_STORE, dl, VT,
                  InChain,
                  getValue(I.getPointerOperand()),
                  getValue(I.getValueOperand()),
                  I.getPointerOperand(), I.getAlignment(),
                  Order, Scope);

  DAG.setRoot(OutChain);
}

/// visitTargetIntrinsic - Lower a call of a target intrinsic to an INTRINSIC
/// node.
void SelectionDAGBuilder::visitTargetIntrinsic(const CallInst &I,
                                               unsigned Intrinsic) {
  bool HasChain = !I.doesNotAccessMemory();
  bool OnlyLoad = HasChain && I.onlyReadsMemory();

  // Build the operand list.
  SmallVector<SDValue, 8> Ops;
  if (HasChain) {  // If this intrinsic has side-effects, chainify it.
    if (OnlyLoad) {
      // We don't need to serialize loads against other loads.
      Ops.push_back(DAG.getRoot());
    } else {
      Ops.push_back(getRoot());
    }
  }

  // Info is set by getTgtMemInstrinsic
  TargetLowering::IntrinsicInfo Info;
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  bool IsTgtIntrinsic = TLI.getTgtMemIntrinsic(Info, I, Intrinsic);

  // Add the intrinsic ID as an integer operand if it's not a target intrinsic.
  if (!IsTgtIntrinsic || Info.opc == ISD::INTRINSIC_VOID ||
      Info.opc == ISD::INTRINSIC_W_CHAIN)
    Ops.push_back(DAG.getTargetConstant(Intrinsic, getCurSDLoc(),
                                        TLI.getPointerTy(DAG.getDataLayout())));

  // Add all operands of the call to the operand list.
  for (unsigned i = 0, e = I.getNumArgOperands(); i != e; ++i) {
    SDValue Op = getValue(I.getArgOperand(i));
    Ops.push_back(Op);
  }

  SmallVector<EVT, 4> ValueVTs;
  ComputeValueVTs(TLI, DAG.getDataLayout(), I.getType(), ValueVTs);

  if (HasChain)
    ValueVTs.push_back(MVT::Other);

  SDVTList VTs = DAG.getVTList(ValueVTs);

  // Create the node.
  SDValue Result;
  if (IsTgtIntrinsic) {
    // This is target intrinsic that touches memory
    Result = DAG.getMemIntrinsicNode(Info.opc, getCurSDLoc(),
                                     VTs, Ops, Info.memVT,
                                   MachinePointerInfo(Info.ptrVal, Info.offset),
                                     Info.align, Info.vol,
                                     Info.readMem, Info.writeMem, Info.size);
  } else if (!HasChain) {
    Result = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, getCurSDLoc(), VTs, Ops);
  } else if (!I.getType()->isVoidTy()) {
    Result = DAG.getNode(ISD::INTRINSIC_W_CHAIN, getCurSDLoc(), VTs, Ops);
  } else {
    Result = DAG.getNode(ISD::INTRINSIC_VOID, getCurSDLoc(), VTs, Ops);
  }

  if (HasChain) {
    SDValue Chain = Result.getValue(Result.getNode()->getNumValues()-1);
    if (OnlyLoad)
      PendingLoads.push_back(Chain);
    else
      DAG.setRoot(Chain);
  }

  if (!I.getType()->isVoidTy()) {
    if (VectorType *PTy = dyn_cast<VectorType>(I.getType())) {
      EVT VT = TLI.getValueType(DAG.getDataLayout(), PTy);
      Result = DAG.getNode(ISD::BITCAST, getCurSDLoc(), VT, Result);
    }

    setValue(&I, Result);
  }
}

/// GetSignificand - Get the significand and build it into a floating-point
/// number with exponent of 1:
///
///   Op = (Op & 0x007fffff) | 0x3f800000;
///
/// where Op is the hexadecimal representation of floating point value.
static SDValue
GetSignificand(SelectionDAG &DAG, SDValue Op, SDLoc dl) {
  SDValue t1 = DAG.getNode(ISD::AND, dl, MVT::i32, Op,
                           DAG.getConstant(0x007fffff, dl, MVT::i32));
  SDValue t2 = DAG.getNode(ISD::OR, dl, MVT::i32, t1,
                           DAG.getConstant(0x3f800000, dl, MVT::i32));
  return DAG.getNode(ISD::BITCAST, dl, MVT::f32, t2);
}

/// GetExponent - Get the exponent:
///
///   (float)(int)(((Op & 0x7f800000) >> 23) - 127);
///
/// where Op is the hexadecimal representation of floating point value.
static SDValue
GetExponent(SelectionDAG &DAG, SDValue Op, const TargetLowering &TLI,
            SDLoc dl) {
  SDValue t0 = DAG.getNode(ISD::AND, dl, MVT::i32, Op,
                           DAG.getConstant(0x7f800000, dl, MVT::i32));
  SDValue t1 = DAG.getNode(
      ISD::SRL, dl, MVT::i32, t0,
      DAG.getConstant(23, dl, TLI.getPointerTy(DAG.getDataLayout())));
  SDValue t2 = DAG.getNode(ISD::SUB, dl, MVT::i32, t1,
                           DAG.getConstant(127, dl, MVT::i32));
  return DAG.getNode(ISD::SINT_TO_FP, dl, MVT::f32, t2);
}

/// getF32Constant - Get 32-bit floating point constant.
static SDValue
getF32Constant(SelectionDAG &DAG, unsigned Flt, SDLoc dl) {
  return DAG.getConstantFP(APFloat(APFloat::IEEEsingle, APInt(32, Flt)), dl,
                           MVT::f32);
}

static SDValue getLimitedPrecisionExp2(SDValue t0, SDLoc dl,
                                       SelectionDAG &DAG) {
  //   IntegerPartOfX = ((int32_t)(t0);
  SDValue IntegerPartOfX = DAG.getNode(ISD::FP_TO_SINT, dl, MVT::i32, t0);

  //   FractionalPartOfX = t0 - (float)IntegerPartOfX;
  SDValue t1 = DAG.getNode(ISD::SINT_TO_FP, dl, MVT::f32, IntegerPartOfX);
  SDValue X = DAG.getNode(ISD::FSUB, dl, MVT::f32, t0, t1);

  //   IntegerPartOfX <<= 23;
  IntegerPartOfX = DAG.getNode(
      ISD::SHL, dl, MVT::i32, IntegerPartOfX,
      DAG.getConstant(23, dl, DAG.getTargetLoweringInfo().getPointerTy(
                                  DAG.getDataLayout())));

  SDValue TwoToFractionalPartOfX;
  if (LimitFloatPrecision <= 6) {
    // For floating-point precision of 6:
    //
    //   TwoToFractionalPartOfX =
    //     0.997535578f +
    //       (0.735607626f + 0.252464424f * x) * x;
    //
    // error 0.0144103317, which is 6 bits
    SDValue t2 = DAG.getNode(ISD::FMUL, dl, MVT::f32, X,
                             getF32Constant(DAG, 0x3e814304, dl));
    SDValue t3 = DAG.getNode(ISD::FADD, dl, MVT::f32, t2,
                             getF32Constant(DAG, 0x3f3c50c8, dl));
    SDValue t4 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t3, X);
    TwoToFractionalPartOfX = DAG.getNode(ISD::FADD, dl, MVT::f32, t4,
                                         getF32Constant(DAG, 0x3f7f5e7e, dl));
  } else if (LimitFloatPrecision <= 12) {
    // For floating-point precision of 12:
    //
    //   TwoToFractionalPartOfX =
    //     0.999892986f +
    //       (0.696457318f +
    //         (0.224338339f + 0.792043434e-1f * x) * x) * x;
    //
    // error 0.000107046256, which is 13 to 14 bits
    SDValue t2 = DAG.getNode(ISD::FMUL, dl, MVT::f32, X,
                             getF32Constant(DAG, 0x3da235e3, dl));
    SDValue t3 = DAG.getNode(ISD::FADD, dl, MVT::f32, t2,
                             getF32Constant(DAG, 0x3e65b8f3, dl));
    SDValue t4 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t3, X);
    SDValue t5 = DAG.getNode(ISD::FADD, dl, MVT::f32, t4,
                             getF32Constant(DAG, 0x3f324b07, dl));
    SDValue t6 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t5, X);
    TwoToFractionalPartOfX = DAG.getNode(ISD::FADD, dl, MVT::f32, t6,
                                         getF32Constant(DAG, 0x3f7ff8fd, dl));
  } else { // LimitFloatPrecision <= 18
    // For floating-point precision of 18:
    //
    //   TwoToFractionalPartOfX =
    //     0.999999982f +
    //       (0.693148872f +
    //         (0.240227044f +
    //           (0.554906021e-1f +
    //             (0.961591928e-2f +
    //               (0.136028312e-2f + 0.157059148e-3f *x)*x)*x)*x)*x)*x;
    // error 2.47208000*10^(-7), which is better than 18 bits
    SDValue t2 = DAG.getNode(ISD::FMUL, dl, MVT::f32, X,
                             getF32Constant(DAG, 0x3924b03e, dl));
    SDValue t3 = DAG.getNode(ISD::FADD, dl, MVT::f32, t2,
                             getF32Constant(DAG, 0x3ab24b87, dl));
    SDValue t4 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t3, X);
    SDValue t5 = DAG.getNode(ISD::FADD, dl, MVT::f32, t4,
                             getF32Constant(DAG, 0x3c1d8c17, dl));
    SDValue t6 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t5, X);
    SDValue t7 = DAG.getNode(ISD::FADD, dl, MVT::f32, t6,
                             getF32Constant(DAG, 0x3d634a1d, dl));
    SDValue t8 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t7, X);
    SDValue t9 = DAG.getNode(ISD::FADD, dl, MVT::f32, t8,
                             getF32Constant(DAG, 0x3e75fe14, dl));
    SDValue t10 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t9, X);
    SDValue t11 = DAG.getNode(ISD::FADD, dl, MVT::f32, t10,
                              getF32Constant(DAG, 0x3f317234, dl));
    SDValue t12 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t11, X);
    TwoToFractionalPartOfX = DAG.getNode(ISD::FADD, dl, MVT::f32, t12,
                                         getF32Constant(DAG, 0x3f800000, dl));
  }

  // Add the exponent into the result in integer domain.
  SDValue t13 = DAG.getNode(ISD::BITCAST, dl, MVT::i32, TwoToFractionalPartOfX);
  return DAG.getNode(ISD::BITCAST, dl, MVT::f32,
                     DAG.getNode(ISD::ADD, dl, MVT::i32, t13, IntegerPartOfX));
}

/// expandExp - Lower an exp intrinsic. Handles the special sequences for
/// limited-precision mode.
static SDValue expandExp(SDLoc dl, SDValue Op, SelectionDAG &DAG,
                         const TargetLowering &TLI) {
  if (Op.getValueType() == MVT::f32 &&
      LimitFloatPrecision > 0 && LimitFloatPrecision <= 18) {

    // Put the exponent in the right bit position for later addition to the
    // final result:
    //
    //   #define LOG2OFe 1.4426950f
    //   t0 = Op * LOG2OFe
    SDValue t0 = DAG.getNode(ISD::FMUL, dl, MVT::f32, Op,
                             getF32Constant(DAG, 0x3fb8aa3b, dl));
    return getLimitedPrecisionExp2(t0, dl, DAG);
  }

  // No special expansion.
  return DAG.getNode(ISD::FEXP, dl, Op.getValueType(), Op);
}

/// expandLog - Lower a log intrinsic. Handles the special sequences for
/// limited-precision mode.
static SDValue expandLog(SDLoc dl, SDValue Op, SelectionDAG &DAG,
                         const TargetLowering &TLI) {
  if (Op.getValueType() == MVT::f32 &&
      LimitFloatPrecision > 0 && LimitFloatPrecision <= 18) {
    SDValue Op1 = DAG.getNode(ISD::BITCAST, dl, MVT::i32, Op);

    // Scale the exponent by log(2) [0.69314718f].
    SDValue Exp = GetExponent(DAG, Op1, TLI, dl);
    SDValue LogOfExponent = DAG.getNode(ISD::FMUL, dl, MVT::f32, Exp,
                                        getF32Constant(DAG, 0x3f317218, dl));

    // Get the significand and build it into a floating-point number with
    // exponent of 1.
    SDValue X = GetSignificand(DAG, Op1, dl);

    SDValue LogOfMantissa;
    if (LimitFloatPrecision <= 6) {
      // For floating-point precision of 6:
      //
      //   LogofMantissa =
      //     -1.1609546f +
      //       (1.4034025f - 0.23903021f * x) * x;
      //
      // error 0.0034276066, which is better than 8 bits
      SDValue t0 = DAG.getNode(ISD::FMUL, dl, MVT::f32, X,
                               getF32Constant(DAG, 0xbe74c456, dl));
      SDValue t1 = DAG.getNode(ISD::FADD, dl, MVT::f32, t0,
                               getF32Constant(DAG, 0x3fb3a2b1, dl));
      SDValue t2 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t1, X);
      LogOfMantissa = DAG.getNode(ISD::FSUB, dl, MVT::f32, t2,
                                  getF32Constant(DAG, 0x3f949a29, dl));
    } else if (LimitFloatPrecision <= 12) {
      // For floating-point precision of 12:
      //
      //   LogOfMantissa =
      //     -1.7417939f +
      //       (2.8212026f +
      //         (-1.4699568f +
      //           (0.44717955f - 0.56570851e-1f * x) * x) * x) * x;
      //
      // error 0.000061011436, which is 14 bits
      SDValue t0 = DAG.getNode(ISD::FMUL, dl, MVT::f32, X,
                               getF32Constant(DAG, 0xbd67b6d6, dl));
      SDValue t1 = DAG.getNode(ISD::FADD, dl, MVT::f32, t0,
                               getF32Constant(DAG, 0x3ee4f4b8, dl));
      SDValue t2 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t1, X);
      SDValue t3 = DAG.getNode(ISD::FSUB, dl, MVT::f32, t2,
                               getF32Constant(DAG, 0x3fbc278b, dl));
      SDValue t4 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t3, X);
      SDValue t5 = DAG.getNode(ISD::FADD, dl, MVT::f32, t4,
                               getF32Constant(DAG, 0x40348e95, dl));
      SDValue t6 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t5, X);
      LogOfMantissa = DAG.getNode(ISD::FSUB, dl, MVT::f32, t6,
                                  getF32Constant(DAG, 0x3fdef31a, dl));
    } else { // LimitFloatPrecision <= 18
      // For floating-point precision of 18:
      //
      //   LogOfMantissa =
      //     -2.1072184f +
      //       (4.2372794f +
      //         (-3.7029485f +
      //           (2.2781945f +
      //             (-0.87823314f +
      //               (0.19073739f - 0.17809712e-1f * x) * x) * x) * x) * x)*x;
      //
      // error 0.0000023660568, which is better than 18 bits
      SDValue t0 = DAG.getNode(ISD::FMUL, dl, MVT::f32, X,
                               getF32Constant(DAG, 0xbc91e5ac, dl));
      SDValue t1 = DAG.getNode(ISD::FADD, dl, MVT::f32, t0,
                               getF32Constant(DAG, 0x3e4350aa, dl));
      SDValue t2 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t1, X);
      SDValue t3 = DAG.getNode(ISD::FSUB, dl, MVT::f32, t2,
                               getF32Constant(DAG, 0x3f60d3e3, dl));
      SDValue t4 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t3, X);
      SDValue t5 = DAG.getNode(ISD::FADD, dl, MVT::f32, t4,
                               getF32Constant(DAG, 0x4011cdf0, dl));
      SDValue t6 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t5, X);
      SDValue t7 = DAG.getNode(ISD::FSUB, dl, MVT::f32, t6,
                               getF32Constant(DAG, 0x406cfd1c, dl));
      SDValue t8 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t7, X);
      SDValue t9 = DAG.getNode(ISD::FADD, dl, MVT::f32, t8,
                               getF32Constant(DAG, 0x408797cb, dl));
      SDValue t10 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t9, X);
      LogOfMantissa = DAG.getNode(ISD::FSUB, dl, MVT::f32, t10,
                                  getF32Constant(DAG, 0x4006dcab, dl));
    }

    return DAG.getNode(ISD::FADD, dl, MVT::f32, LogOfExponent, LogOfMantissa);
  }

  // No special expansion.
  return DAG.getNode(ISD::FLOG, dl, Op.getValueType(), Op);
}

/// expandLog2 - Lower a log2 intrinsic. Handles the special sequences for
/// limited-precision mode.
static SDValue expandLog2(SDLoc dl, SDValue Op, SelectionDAG &DAG,
                          const TargetLowering &TLI) {
  if (Op.getValueType() == MVT::f32 &&
      LimitFloatPrecision > 0 && LimitFloatPrecision <= 18) {
    SDValue Op1 = DAG.getNode(ISD::BITCAST, dl, MVT::i32, Op);

    // Get the exponent.
    SDValue LogOfExponent = GetExponent(DAG, Op1, TLI, dl);

    // Get the significand and build it into a floating-point number with
    // exponent of 1.
    SDValue X = GetSignificand(DAG, Op1, dl);

    // Different possible minimax approximations of significand in
    // floating-point for various degrees of accuracy over [1,2].
    SDValue Log2ofMantissa;
    if (LimitFloatPrecision <= 6) {
      // For floating-point precision of 6:
      //
      //   Log2ofMantissa = -1.6749035f + (2.0246817f - .34484768f * x) * x;
      //
      // error 0.0049451742, which is more than 7 bits
      SDValue t0 = DAG.getNode(ISD::FMUL, dl, MVT::f32, X,
                               getF32Constant(DAG, 0xbeb08fe0, dl));
      SDValue t1 = DAG.getNode(ISD::FADD, dl, MVT::f32, t0,
                               getF32Constant(DAG, 0x40019463, dl));
      SDValue t2 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t1, X);
      Log2ofMantissa = DAG.getNode(ISD::FSUB, dl, MVT::f32, t2,
                                   getF32Constant(DAG, 0x3fd6633d, dl));
    } else if (LimitFloatPrecision <= 12) {
      // For floating-point precision of 12:
      //
      //   Log2ofMantissa =
      //     -2.51285454f +
      //       (4.07009056f +
      //         (-2.12067489f +
      //           (.645142248f - 0.816157886e-1f * x) * x) * x) * x;
      //
      // error 0.0000876136000, which is better than 13 bits
      SDValue t0 = DAG.getNode(ISD::FMUL, dl, MVT::f32, X,
                               getF32Constant(DAG, 0xbda7262e, dl));
      SDValue t1 = DAG.getNode(ISD::FADD, dl, MVT::f32, t0,
                               getF32Constant(DAG, 0x3f25280b, dl));
      SDValue t2 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t1, X);
      SDValue t3 = DAG.getNode(ISD::FSUB, dl, MVT::f32, t2,
                               getF32Constant(DAG, 0x4007b923, dl));
      SDValue t4 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t3, X);
      SDValue t5 = DAG.getNode(ISD::FADD, dl, MVT::f32, t4,
                               getF32Constant(DAG, 0x40823e2f, dl));
      SDValue t6 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t5, X);
      Log2ofMantissa = DAG.getNode(ISD::FSUB, dl, MVT::f32, t6,
                                   getF32Constant(DAG, 0x4020d29c, dl));
    } else { // LimitFloatPrecision <= 18
      // For floating-point precision of 18:
      //
      //   Log2ofMantissa =
      //     -3.0400495f +
      //       (6.1129976f +
      //         (-5.3420409f +
      //           (3.2865683f +
      //             (-1.2669343f +
      //               (0.27515199f -
      //                 0.25691327e-1f * x) * x) * x) * x) * x) * x;
      //
      // error 0.0000018516, which is better than 18 bits
      SDValue t0 = DAG.getNode(ISD::FMUL, dl, MVT::f32, X,
                               getF32Constant(DAG, 0xbcd2769e, dl));
      SDValue t1 = DAG.getNode(ISD::FADD, dl, MVT::f32, t0,
                               getF32Constant(DAG, 0x3e8ce0b9, dl));
      SDValue t2 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t1, X);
      SDValue t3 = DAG.getNode(ISD::FSUB, dl, MVT::f32, t2,
                               getF32Constant(DAG, 0x3fa22ae7, dl));
      SDValue t4 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t3, X);
      SDValue t5 = DAG.getNode(ISD::FADD, dl, MVT::f32, t4,
                               getF32Constant(DAG, 0x40525723, dl));
      SDValue t6 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t5, X);
      SDValue t7 = DAG.getNode(ISD::FSUB, dl, MVT::f32, t6,
                               getF32Constant(DAG, 0x40aaf200, dl));
      SDValue t8 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t7, X);
      SDValue t9 = DAG.getNode(ISD::FADD, dl, MVT::f32, t8,
                               getF32Constant(DAG, 0x40c39dad, dl));
      SDValue t10 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t9, X);
      Log2ofMantissa = DAG.getNode(ISD::FSUB, dl, MVT::f32, t10,
                                   getF32Constant(DAG, 0x4042902c, dl));
    }

    return DAG.getNode(ISD::FADD, dl, MVT::f32, LogOfExponent, Log2ofMantissa);
  }

  // No special expansion.
  return DAG.getNode(ISD::FLOG2, dl, Op.getValueType(), Op);
}

/// expandLog10 - Lower a log10 intrinsic. Handles the special sequences for
/// limited-precision mode.
static SDValue expandLog10(SDLoc dl, SDValue Op, SelectionDAG &DAG,
                           const TargetLowering &TLI) {
  if (Op.getValueType() == MVT::f32 &&
      LimitFloatPrecision > 0 && LimitFloatPrecision <= 18) {
    SDValue Op1 = DAG.getNode(ISD::BITCAST, dl, MVT::i32, Op);

    // Scale the exponent by log10(2) [0.30102999f].
    SDValue Exp = GetExponent(DAG, Op1, TLI, dl);
    SDValue LogOfExponent = DAG.getNode(ISD::FMUL, dl, MVT::f32, Exp,
                                        getF32Constant(DAG, 0x3e9a209a, dl));

    // Get the significand and build it into a floating-point number with
    // exponent of 1.
    SDValue X = GetSignificand(DAG, Op1, dl);

    SDValue Log10ofMantissa;
    if (LimitFloatPrecision <= 6) {
      // For floating-point precision of 6:
      //
      //   Log10ofMantissa =
      //     -0.50419619f +
      //       (0.60948995f - 0.10380950f * x) * x;
      //
      // error 0.0014886165, which is 6 bits
      SDValue t0 = DAG.getNode(ISD::FMUL, dl, MVT::f32, X,
                               getF32Constant(DAG, 0xbdd49a13, dl));
      SDValue t1 = DAG.getNode(ISD::FADD, dl, MVT::f32, t0,
                               getF32Constant(DAG, 0x3f1c0789, dl));
      SDValue t2 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t1, X);
      Log10ofMantissa = DAG.getNode(ISD::FSUB, dl, MVT::f32, t2,
                                    getF32Constant(DAG, 0x3f011300, dl));
    } else if (LimitFloatPrecision <= 12) {
      // For floating-point precision of 12:
      //
      //   Log10ofMantissa =
      //     -0.64831180f +
      //       (0.91751397f +
      //         (-0.31664806f + 0.47637168e-1f * x) * x) * x;
      //
      // error 0.00019228036, which is better than 12 bits
      SDValue t0 = DAG.getNode(ISD::FMUL, dl, MVT::f32, X,
                               getF32Constant(DAG, 0x3d431f31, dl));
      SDValue t1 = DAG.getNode(ISD::FSUB, dl, MVT::f32, t0,
                               getF32Constant(DAG, 0x3ea21fb2, dl));
      SDValue t2 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t1, X);
      SDValue t3 = DAG.getNode(ISD::FADD, dl, MVT::f32, t2,
                               getF32Constant(DAG, 0x3f6ae232, dl));
      SDValue t4 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t3, X);
      Log10ofMantissa = DAG.getNode(ISD::FSUB, dl, MVT::f32, t4,
                                    getF32Constant(DAG, 0x3f25f7c3, dl));
    } else { // LimitFloatPrecision <= 18
      // For floating-point precision of 18:
      //
      //   Log10ofMantissa =
      //     -0.84299375f +
      //       (1.5327582f +
      //         (-1.0688956f +
      //           (0.49102474f +
      //             (-0.12539807f + 0.13508273e-1f * x) * x) * x) * x) * x;
      //
      // error 0.0000037995730, which is better than 18 bits
      SDValue t0 = DAG.getNode(ISD::FMUL, dl, MVT::f32, X,
                               getF32Constant(DAG, 0x3c5d51ce, dl));
      SDValue t1 = DAG.getNode(ISD::FSUB, dl, MVT::f32, t0,
                               getF32Constant(DAG, 0x3e00685a, dl));
      SDValue t2 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t1, X);
      SDValue t3 = DAG.getNode(ISD::FADD, dl, MVT::f32, t2,
                               getF32Constant(DAG, 0x3efb6798, dl));
      SDValue t4 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t3, X);
      SDValue t5 = DAG.getNode(ISD::FSUB, dl, MVT::f32, t4,
                               getF32Constant(DAG, 0x3f88d192, dl));
      SDValue t6 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t5, X);
      SDValue t7 = DAG.getNode(ISD::FADD, dl, MVT::f32, t6,
                               getF32Constant(DAG, 0x3fc4316c, dl));
      SDValue t8 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t7, X);
      Log10ofMantissa = DAG.getNode(ISD::FSUB, dl, MVT::f32, t8,
                                    getF32Constant(DAG, 0x3f57ce70, dl));
    }

    return DAG.getNode(ISD::FADD, dl, MVT::f32, LogOfExponent, Log10ofMantissa);
  }

  // No special expansion.
  return DAG.getNode(ISD::FLOG10, dl, Op.getValueType(), Op);
}

/// expandExp2 - Lower an exp2 intrinsic. Handles the special sequences for
/// limited-precision mode.
static SDValue expandExp2(SDLoc dl, SDValue Op, SelectionDAG &DAG,
                          const TargetLowering &TLI) {
  if (Op.getValueType() == MVT::f32 &&
      LimitFloatPrecision > 0 && LimitFloatPrecision <= 18)
    return getLimitedPrecisionExp2(Op, dl, DAG);

  // No special expansion.
  return DAG.getNode(ISD::FEXP2, dl, Op.getValueType(), Op);
}

/// visitPow - Lower a pow intrinsic. Handles the special sequences for
/// limited-precision mode with x == 10.0f.
static SDValue expandPow(SDLoc dl, SDValue LHS, SDValue RHS,
                         SelectionDAG &DAG, const TargetLowering &TLI) {
  bool IsExp10 = false;
  if (LHS.getValueType() == MVT::f32 && RHS.getValueType() == MVT::f32 &&
      LimitFloatPrecision > 0 && LimitFloatPrecision <= 18) {
    if (ConstantFPSDNode *LHSC = dyn_cast<ConstantFPSDNode>(LHS)) {
      APFloat Ten(10.0f);
      IsExp10 = LHSC->isExactlyValue(Ten);
    }
  }

  if (IsExp10) {
    // Put the exponent in the right bit position for later addition to the
    // final result:
    //
    //   #define LOG2OF10 3.3219281f
    //   t0 = Op * LOG2OF10;
    SDValue t0 = DAG.getNode(ISD::FMUL, dl, MVT::f32, RHS,
                             getF32Constant(DAG, 0x40549a78, dl));
    return getLimitedPrecisionExp2(t0, dl, DAG);
  }

  // No special expansion.
  return DAG.getNode(ISD::FPOW, dl, LHS.getValueType(), LHS, RHS);
}


/// ExpandPowI - Expand a llvm.powi intrinsic.
static SDValue ExpandPowI(SDLoc DL, SDValue LHS, SDValue RHS,
                          SelectionDAG &DAG) {
  // If RHS is a constant, we can expand this out to a multiplication tree,
  // otherwise we end up lowering to a call to __powidf2 (for example).  When
  // optimizing for size, we only want to do this if the expansion would produce
  // a small number of multiplies, otherwise we do the full expansion.
  if (ConstantSDNode *RHSC = dyn_cast<ConstantSDNode>(RHS)) {
    // Get the exponent as a positive value.
    unsigned Val = RHSC->getSExtValue();
    if ((int)Val < 0) Val = -Val;

    // powi(x, 0) -> 1.0
    if (Val == 0)
      return DAG.getConstantFP(1.0, DL, LHS.getValueType());

    const Function *F = DAG.getMachineFunction().getFunction();
    if (!F->hasFnAttribute(Attribute::OptimizeForSize) ||
        // If optimizing for size, don't insert too many multiplies.  This
        // inserts up to 5 multiplies.
        countPopulation(Val) + Log2_32(Val) < 7) {
      // We use the simple binary decomposition method to generate the multiply
      // sequence.  There are more optimal ways to do this (for example,
      // powi(x,15) generates one more multiply than it should), but this has
      // the benefit of being both really simple and much better than a libcall.
      SDValue Res;  // Logically starts equal to 1.0
      SDValue CurSquare = LHS;
      while (Val) {
        if (Val & 1) {
          if (Res.getNode())
            Res = DAG.getNode(ISD::FMUL, DL,Res.getValueType(), Res, CurSquare);
          else
            Res = CurSquare;  // 1.0*CurSquare.
        }

        CurSquare = DAG.getNode(ISD::FMUL, DL, CurSquare.getValueType(),
                                CurSquare, CurSquare);
        Val >>= 1;
      }

      // If the original was negative, invert the result, producing 1/(x*x*x).
      if (RHSC->getSExtValue() < 0)
        Res = DAG.getNode(ISD::FDIV, DL, LHS.getValueType(),
                          DAG.getConstantFP(1.0, DL, LHS.getValueType()), Res);
      return Res;
    }
  }

  // Otherwise, expand to a libcall.
  return DAG.getNode(ISD::FPOWI, DL, LHS.getValueType(), LHS, RHS);
}

// getTruncatedArgReg - Find underlying register used for an truncated
// argument.
static unsigned getTruncatedArgReg(const SDValue &N) {
  if (N.getOpcode() != ISD::TRUNCATE)
    return 0;

  const SDValue &Ext = N.getOperand(0);
  if (Ext.getOpcode() == ISD::AssertZext ||
      Ext.getOpcode() == ISD::AssertSext) {
    const SDValue &CFR = Ext.getOperand(0);
    if (CFR.getOpcode() == ISD::CopyFromReg)
      return cast<RegisterSDNode>(CFR.getOperand(1))->getReg();
    if (CFR.getOpcode() == ISD::TRUNCATE)
      return getTruncatedArgReg(CFR);
  }
  return 0;
}

/// EmitFuncArgumentDbgValue - If the DbgValueInst is a dbg_value of a function
/// argument, create the corresponding DBG_VALUE machine instruction for it now.
/// At the end of instruction selection, they will be inserted to the entry BB.
bool SelectionDAGBuilder::EmitFuncArgumentDbgValue(
    const Value *V, DILocalVariable *Variable, DIExpression *Expr,
    DILocation *DL, int64_t Offset, bool IsIndirect, const SDValue &N) {
  const Argument *Arg = dyn_cast<Argument>(V);
  if (!Arg)
    return false;

  MachineFunction &MF = DAG.getMachineFunction();
  const TargetInstrInfo *TII = DAG.getSubtarget().getInstrInfo();

  // Ignore inlined function arguments here.
  //
  // FIXME: Should we be checking DL->inlinedAt() to determine this?
  if (!Variable->getScope()->getSubprogram()->describes(MF.getFunction()))
    return false;

  Optional<MachineOperand> Op;
  // Some arguments' frame index is recorded during argument lowering.
  if (int FI = FuncInfo.getArgumentFrameIndex(Arg))
    Op = MachineOperand::CreateFI(FI);

  if (!Op && N.getNode()) {
    unsigned Reg;
    if (N.getOpcode() == ISD::CopyFromReg)
      Reg = cast<RegisterSDNode>(N.getOperand(1))->getReg();
    else
      Reg = getTruncatedArgReg(N);
    if (Reg && TargetRegisterInfo::isVirtualRegister(Reg)) {
      MachineRegisterInfo &RegInfo = MF.getRegInfo();
      unsigned PR = RegInfo.getLiveInPhysReg(Reg);
      if (PR)
        Reg = PR;
    }
    if (Reg)
      Op = MachineOperand::CreateReg(Reg, false);
  }

  if (!Op) {
    // Check if ValueMap has reg number.
    DenseMap<const Value *, unsigned>::iterator VMI = FuncInfo.ValueMap.find(V);
    if (VMI != FuncInfo.ValueMap.end())
      Op = MachineOperand::CreateReg(VMI->second, false);
  }

  if (!Op && N.getNode())
    // Check if frame index is available.
    if (LoadSDNode *LNode = dyn_cast<LoadSDNode>(N.getNode()))
      if (FrameIndexSDNode *FINode =
          dyn_cast<FrameIndexSDNode>(LNode->getBasePtr().getNode()))
        Op = MachineOperand::CreateFI(FINode->getIndex());

  if (!Op)
    return false;

  assert(Variable->isValidLocationForIntrinsic(DL) &&
         "Expected inlined-at fields to agree");
  if (Op->isReg())
    FuncInfo.ArgDbgValues.push_back(
        BuildMI(MF, DL, TII->get(TargetOpcode::DBG_VALUE), IsIndirect,
                Op->getReg(), Offset, Variable, Expr));
  else
    FuncInfo.ArgDbgValues.push_back(
        BuildMI(MF, DL, TII->get(TargetOpcode::DBG_VALUE))
            .addOperand(*Op)
            .addImm(Offset)
            .addMetadata(Variable)
            .addMetadata(Expr));

  return true;
}

// VisualStudio defines setjmp as _setjmp
#if defined(_MSC_VER) && defined(setjmp) && \
                         !defined(setjmp_undefined_for_msvc)
#  pragma push_macro("setjmp")
#  undef setjmp
#  define setjmp_undefined_for_msvc
#endif

/// visitIntrinsicCall - Lower the call to the specified intrinsic function.  If
/// we want to emit this as a call to a named external function, return the name
/// otherwise lower it and return null.
const char *
SelectionDAGBuilder::visitIntrinsicCall(const CallInst &I, unsigned Intrinsic) {
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  SDLoc sdl = getCurSDLoc();
  DebugLoc dl = getCurDebugLoc();
  SDValue Res;

  switch (Intrinsic) {
  default:
    // By default, turn this into a target intrinsic node.
    visitTargetIntrinsic(I, Intrinsic);
    return nullptr;
  case Intrinsic::vastart:  visitVAStart(I); return nullptr;
  case Intrinsic::vaend:    visitVAEnd(I); return nullptr;
  case Intrinsic::vacopy:   visitVACopy(I); return nullptr;
  case Intrinsic::returnaddress:
    setValue(&I, DAG.getNode(ISD::RETURNADDR, sdl,
                             TLI.getPointerTy(DAG.getDataLayout()),
                             getValue(I.getArgOperand(0))));
    return nullptr;
  case Intrinsic::frameaddress:
    setValue(&I, DAG.getNode(ISD::FRAMEADDR, sdl,
                             TLI.getPointerTy(DAG.getDataLayout()),
                             getValue(I.getArgOperand(0))));
    return nullptr;
  case Intrinsic::read_register: {
    Value *Reg = I.getArgOperand(0);
    SDValue Chain = getRoot();
    SDValue RegName =
        DAG.getMDNode(cast<MDNode>(cast<MetadataAsValue>(Reg)->getMetadata()));
    EVT VT = TLI.getValueType(DAG.getDataLayout(), I.getType());
    Res = DAG.getNode(ISD::READ_REGISTER, sdl,
      DAG.getVTList(VT, MVT::Other), Chain, RegName);
    setValue(&I, Res);
    DAG.setRoot(Res.getValue(1));
    return nullptr;
  }
  case Intrinsic::write_register: {
    Value *Reg = I.getArgOperand(0);
    Value *RegValue = I.getArgOperand(1);
    SDValue Chain = getRoot();
    SDValue RegName =
        DAG.getMDNode(cast<MDNode>(cast<MetadataAsValue>(Reg)->getMetadata()));
    DAG.setRoot(DAG.getNode(ISD::WRITE_REGISTER, sdl, MVT::Other, Chain,
                            RegName, getValue(RegValue)));
    return nullptr;
  }
  case Intrinsic::setjmp:
    return &"_setjmp"[!TLI.usesUnderscoreSetJmp()];
  case Intrinsic::longjmp:
    return &"_longjmp"[!TLI.usesUnderscoreLongJmp()];
  case Intrinsic::memcpy: {
    // FIXME: this definition of "user defined address space" is x86-specific
    // Assert for address < 256 since we support only user defined address
    // spaces.
    assert(cast<PointerType>(I.getArgOperand(0)->getType())->getAddressSpace()
           < 256 &&
           cast<PointerType>(I.getArgOperand(1)->getType())->getAddressSpace()
           < 256 &&
           "Unknown address space");
    SDValue Op1 = getValue(I.getArgOperand(0));
    SDValue Op2 = getValue(I.getArgOperand(1));
    SDValue Op3 = getValue(I.getArgOperand(2));
    unsigned Align = cast<ConstantInt>(I.getArgOperand(3))->getZExtValue();
    if (!Align)
      Align = 1; // @llvm.memcpy defines 0 and 1 to both mean no alignment.
    bool isVol = cast<ConstantInt>(I.getArgOperand(4))->getZExtValue();
    bool isTC = I.isTailCall() && isInTailCallPosition(&I, DAG.getTarget());
    SDValue MC = DAG.getMemcpy(getRoot(), sdl, Op1, Op2, Op3, Align, isVol,
                               false, isTC,
                               MachinePointerInfo(I.getArgOperand(0)),
                               MachinePointerInfo(I.getArgOperand(1)));
    updateDAGForMaybeTailCall(MC);
    return nullptr;
  }
  case Intrinsic::memset: {
    // FIXME: this definition of "user defined address space" is x86-specific
    // Assert for address < 256 since we support only user defined address
    // spaces.
    assert(cast<PointerType>(I.getArgOperand(0)->getType())->getAddressSpace()
           < 256 &&
           "Unknown address space");
    SDValue Op1 = getValue(I.getArgOperand(0));
    SDValue Op2 = getValue(I.getArgOperand(1));
    SDValue Op3 = getValue(I.getArgOperand(2));
    unsigned Align = cast<ConstantInt>(I.getArgOperand(3))->getZExtValue();
    if (!Align)
      Align = 1; // @llvm.memset defines 0 and 1 to both mean no alignment.
    bool isVol = cast<ConstantInt>(I.getArgOperand(4))->getZExtValue();
    bool isTC = I.isTailCall() && isInTailCallPosition(&I, DAG.getTarget());
    SDValue MS = DAG.getMemset(getRoot(), sdl, Op1, Op2, Op3, Align, isVol,
                               isTC, MachinePointerInfo(I.getArgOperand(0)));
    updateDAGForMaybeTailCall(MS);
    return nullptr;
  }
  case Intrinsic::memmove: {
    // FIXME: this definition of "user defined address space" is x86-specific
    // Assert for address < 256 since we support only user defined address
    // spaces.
    assert(cast<PointerType>(I.getArgOperand(0)->getType())->getAddressSpace()
           < 256 &&
           cast<PointerType>(I.getArgOperand(1)->getType())->getAddressSpace()
           < 256 &&
           "Unknown address space");
    SDValue Op1 = getValue(I.getArgOperand(0));
    SDValue Op2 = getValue(I.getArgOperand(1));
    SDValue Op3 = getValue(I.getArgOperand(2));
    unsigned Align = cast<ConstantInt>(I.getArgOperand(3))->getZExtValue();
    if (!Align)
      Align = 1; // @llvm.memmove defines 0 and 1 to both mean no alignment.
    bool isVol = cast<ConstantInt>(I.getArgOperand(4))->getZExtValue();
    bool isTC = I.isTailCall() && isInTailCallPosition(&I, DAG.getTarget());
    SDValue MM = DAG.getMemmove(getRoot(), sdl, Op1, Op2, Op3, Align, isVol,
                                isTC, MachinePointerInfo(I.getArgOperand(0)),
                                MachinePointerInfo(I.getArgOperand(1)));
    updateDAGForMaybeTailCall(MM);
    return nullptr;
  }
  case Intrinsic::dbg_declare: {
    const DbgDeclareInst &DI = cast<DbgDeclareInst>(I);
    DILocalVariable *Variable = DI.getVariable();
    DIExpression *Expression = DI.getExpression();
    const Value *Address = DI.getAddress();
    assert(Variable && "Missing variable");
    if (!Address) {
      DEBUG(dbgs() << "Dropping debug info for " << DI << "\n");
      return nullptr;
    }

    // Check if address has undef value.
    if (isa<UndefValue>(Address) ||
        (Address->use_empty() && !isa<Argument>(Address))) {
      DEBUG(dbgs() << "Dropping debug info for " << DI << "\n");
      return nullptr;
    }

    SDValue &N = NodeMap[Address];
    if (!N.getNode() && isa<Argument>(Address))
      // Check unused arguments map.
      N = UnusedArgNodeMap[Address];
    SDDbgValue *SDV;
    if (N.getNode()) {
      if (const BitCastInst *BCI = dyn_cast<BitCastInst>(Address))
        Address = BCI->getOperand(0);
      // Parameters are handled specially.
      bool isParameter = Variable->getTag() == dwarf::DW_TAG_arg_variable ||
                         isa<Argument>(Address);

      const AllocaInst *AI = dyn_cast<AllocaInst>(Address);

      if (isParameter && !AI) {
        FrameIndexSDNode *FINode = dyn_cast<FrameIndexSDNode>(N.getNode());
        if (FINode)
          // Byval parameter.  We have a frame index at this point.
          SDV = DAG.getFrameIndexDbgValue(
              Variable, Expression, FINode->getIndex(), 0, dl, SDNodeOrder);
        else {
          // Address is an argument, so try to emit its dbg value using
          // virtual register info from the FuncInfo.ValueMap.
          EmitFuncArgumentDbgValue(Address, Variable, Expression, dl, 0, false,
                                   N);
          return nullptr;
        }
      } else if (AI)
        SDV = DAG.getDbgValue(Variable, Expression, N.getNode(), N.getResNo(),
                              true, 0, dl, SDNodeOrder);
      else {
        // Can't do anything with other non-AI cases yet.
        DEBUG(dbgs() << "Dropping debug info for " << DI << "\n");
        DEBUG(dbgs() << "non-AllocaInst issue for Address: \n\t");
        DEBUG(Address->dump());
        return nullptr;
      }
      DAG.AddDbgValue(SDV, N.getNode(), isParameter);
    } else {
      // If Address is an argument then try to emit its dbg value using
      // virtual register info from the FuncInfo.ValueMap.
      if (!EmitFuncArgumentDbgValue(Address, Variable, Expression, dl, 0, false,
                                    N)) {
        // If variable is pinned by a alloca in dominating bb then
        // use StaticAllocaMap.
        if (const AllocaInst *AI = dyn_cast<AllocaInst>(Address)) {
          if (AI->getParent() != DI.getParent()) {
            DenseMap<const AllocaInst*, int>::iterator SI =
              FuncInfo.StaticAllocaMap.find(AI);
            if (SI != FuncInfo.StaticAllocaMap.end()) {
              SDV = DAG.getFrameIndexDbgValue(Variable, Expression, SI->second,
                                              0, dl, SDNodeOrder);
              DAG.AddDbgValue(SDV, nullptr, false);
              return nullptr;
            }
          }
        }
        DEBUG(dbgs() << "Dropping debug info for " << DI << "\n");
      }
    }
    return nullptr;
  }
  case Intrinsic::dbg_value: {
    const DbgValueInst &DI = cast<DbgValueInst>(I);
    assert(DI.getVariable() && "Missing variable");

    DILocalVariable *Variable = DI.getVariable();
    DIExpression *Expression = DI.getExpression();
    uint64_t Offset = DI.getOffset();
    const Value *V = DI.getValue();
    if (!V)
      return nullptr;

    SDDbgValue *SDV;
    if (isa<ConstantInt>(V) || isa<ConstantFP>(V) || isa<UndefValue>(V)) {
      SDV = DAG.getConstantDbgValue(Variable, Expression, V, Offset, dl,
                                    SDNodeOrder);
      DAG.AddDbgValue(SDV, nullptr, false);
    } else {
      // Do not use getValue() in here; we don't want to generate code at
      // this point if it hasn't been done yet.
      SDValue N = NodeMap[V];
      if (!N.getNode() && isa<Argument>(V))
        // Check unused arguments map.
        N = UnusedArgNodeMap[V];
      if (N.getNode()) {
        // A dbg.value for an alloca is always indirect.
        bool IsIndirect = isa<AllocaInst>(V) || Offset != 0;
        if (!EmitFuncArgumentDbgValue(V, Variable, Expression, dl, Offset,
                                      IsIndirect, N)) {
          SDV = DAG.getDbgValue(Variable, Expression, N.getNode(), N.getResNo(),
                                IsIndirect, Offset, dl, SDNodeOrder);
          DAG.AddDbgValue(SDV, N.getNode(), false);
        }
      } else if (!V->use_empty() ) {
        // Do not call getValue(V) yet, as we don't want to generate code.
        // Remember it for later.
        DanglingDebugInfo DDI(&DI, dl, SDNodeOrder);
        DanglingDebugInfoMap[V] = DDI;
      } else {
        // We may expand this to cover more cases.  One case where we have no
        // data available is an unreferenced parameter.
        DEBUG(dbgs() << "Dropping debug info for " << DI << "\n");
      }
    }

    // Build a debug info table entry.
    if (const BitCastInst *BCI = dyn_cast<BitCastInst>(V))
      V = BCI->getOperand(0);
    const AllocaInst *AI = dyn_cast<AllocaInst>(V);
    // Don't handle byval struct arguments or VLAs, for example.
    if (!AI) {
      DEBUG(dbgs() << "Dropping debug location info for:\n  " << DI << "\n");
      DEBUG(dbgs() << "  Last seen at:\n    " << *V << "\n");
      return nullptr;
    }
    DenseMap<const AllocaInst*, int>::iterator SI =
      FuncInfo.StaticAllocaMap.find(AI);
    if (SI == FuncInfo.StaticAllocaMap.end())
      return nullptr; // VLAs.
    return nullptr;
  }

  case Intrinsic::eh_typeid_for: {
    // Find the type id for the given typeinfo.
    GlobalValue *GV = ExtractTypeInfo(I.getArgOperand(0));
    unsigned TypeID = DAG.getMachineFunction().getMMI().getTypeIDFor(GV);
    Res = DAG.getConstant(TypeID, sdl, MVT::i32);
    setValue(&I, Res);
    return nullptr;
  }

  case Intrinsic::eh_return_i32:
  case Intrinsic::eh_return_i64:
    DAG.getMachineFunction().getMMI().setCallsEHReturn(true);
    DAG.setRoot(DAG.getNode(ISD::EH_RETURN, sdl,
                            MVT::Other,
                            getControlRoot(),
                            getValue(I.getArgOperand(0)),
                            getValue(I.getArgOperand(1))));
    return nullptr;
  case Intrinsic::eh_unwind_init:
    DAG.getMachineFunction().getMMI().setCallsUnwindInit(true);
    return nullptr;
  case Intrinsic::eh_dwarf_cfa: {
    SDValue CfaArg = DAG.getSExtOrTrunc(getValue(I.getArgOperand(0)), sdl,
                                        TLI.getPointerTy(DAG.getDataLayout()));
    SDValue Offset = DAG.getNode(ISD::ADD, sdl,
                                 CfaArg.getValueType(),
                                 DAG.getNode(ISD::FRAME_TO_ARGS_OFFSET, sdl,
                                             CfaArg.getValueType()),
                                 CfaArg);
    SDValue FA = DAG.getNode(
        ISD::FRAMEADDR, sdl, TLI.getPointerTy(DAG.getDataLayout()),
        DAG.getConstant(0, sdl, TLI.getPointerTy(DAG.getDataLayout())));
    setValue(&I, DAG.getNode(ISD::ADD, sdl, FA.getValueType(),
                             FA, Offset));
    return nullptr;
  }
  case Intrinsic::eh_sjlj_callsite: {
    MachineModuleInfo &MMI = DAG.getMachineFunction().getMMI();
    ConstantInt *CI = dyn_cast<ConstantInt>(I.getArgOperand(0));
    assert(CI && "Non-constant call site value in eh.sjlj.callsite!");
    assert(MMI.getCurrentCallSite() == 0 && "Overlapping call sites!");

    MMI.setCurrentCallSite(CI->getZExtValue());
    return nullptr;
  }
  case Intrinsic::eh_sjlj_functioncontext: {
    // Get and store the index of the function context.
    MachineFrameInfo *MFI = DAG.getMachineFunction().getFrameInfo();
    AllocaInst *FnCtx =
      cast<AllocaInst>(I.getArgOperand(0)->stripPointerCasts());
    int FI = FuncInfo.StaticAllocaMap[FnCtx];
    MFI->setFunctionContextIndex(FI);
    return nullptr;
  }
  case Intrinsic::eh_sjlj_setjmp: {
    SDValue Ops[2];
    Ops[0] = getRoot();
    Ops[1] = getValue(I.getArgOperand(0));
    SDValue Op = DAG.getNode(ISD::EH_SJLJ_SETJMP, sdl,
                             DAG.getVTList(MVT::i32, MVT::Other), Ops);
    setValue(&I, Op.getValue(0));
    DAG.setRoot(Op.getValue(1));
    return nullptr;
  }
  case Intrinsic::eh_sjlj_longjmp: {
    DAG.setRoot(DAG.getNode(ISD::EH_SJLJ_LONGJMP, sdl, MVT::Other,
                            getRoot(), getValue(I.getArgOperand(0))));
    return nullptr;
  }

  case Intrinsic::masked_gather:
    visitMaskedGather(I);
    return nullptr;
  case Intrinsic::masked_load:
    visitMaskedLoad(I);
    return nullptr;
  case Intrinsic::masked_scatter:
    visitMaskedScatter(I);
    return nullptr;
  case Intrinsic::masked_store:
    visitMaskedStore(I);
    return nullptr;
#if 0 // HLSL Change - remove platform intrinsics
  case Intrinsic::x86_mmx_pslli_w:
  case Intrinsic::x86_mmx_pslli_d:
  case Intrinsic::x86_mmx_pslli_q:
  case Intrinsic::x86_mmx_psrli_w:
  case Intrinsic::x86_mmx_psrli_d:
  case Intrinsic::x86_mmx_psrli_q:
  case Intrinsic::x86_mmx_psrai_w:
  case Intrinsic::x86_mmx_psrai_d: {
    SDValue ShAmt = getValue(I.getArgOperand(1));
    if (isa<ConstantSDNode>(ShAmt)) {
      visitTargetIntrinsic(I, Intrinsic);
      return nullptr;
    }
    unsigned NewIntrinsic = 0;
    EVT ShAmtVT = MVT::v2i32;
    switch (Intrinsic) {
    case Intrinsic::x86_mmx_pslli_w:
      NewIntrinsic = Intrinsic::x86_mmx_psll_w;
      break;
    case Intrinsic::x86_mmx_pslli_d:
      NewIntrinsic = Intrinsic::x86_mmx_psll_d;
      break;
    case Intrinsic::x86_mmx_pslli_q:
      NewIntrinsic = Intrinsic::x86_mmx_psll_q;
      break;
    case Intrinsic::x86_mmx_psrli_w:
      NewIntrinsic = Intrinsic::x86_mmx_psrl_w;
      break;
    case Intrinsic::x86_mmx_psrli_d:
      NewIntrinsic = Intrinsic::x86_mmx_psrl_d;
      break;
    case Intrinsic::x86_mmx_psrli_q:
      NewIntrinsic = Intrinsic::x86_mmx_psrl_q;
      break;
    case Intrinsic::x86_mmx_psrai_w:
      NewIntrinsic = Intrinsic::x86_mmx_psra_w;
      break;
    case Intrinsic::x86_mmx_psrai_d:
      NewIntrinsic = Intrinsic::x86_mmx_psra_d;
      break;
    default: llvm_unreachable("Impossible intrinsic");  // Can't reach here.
    }

    // The vector shift intrinsics with scalars uses 32b shift amounts but
    // the sse2/mmx shift instructions reads 64 bits. Set the upper 32 bits
    // to be zero.
    // We must do this early because v2i32 is not a legal type.
    SDValue ShOps[2];
    ShOps[0] = ShAmt;
    ShOps[1] = DAG.getConstant(0, sdl, MVT::i32);
    ShAmt =  DAG.getNode(ISD::BUILD_VECTOR, sdl, ShAmtVT, ShOps);
    EVT DestVT = TLI.getValueType(DAG.getDataLayout(), I.getType());
    ShAmt = DAG.getNode(ISD::BITCAST, sdl, DestVT, ShAmt);
    Res = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, sdl, DestVT,
                       DAG.getConstant(NewIntrinsic, sdl, MVT::i32),
                       getValue(I.getArgOperand(0)), ShAmt);
    setValue(&I, Res);
    return nullptr;
  }
#endif // HLSL Change - remove platform intrinsics
  case Intrinsic::convertff:
  case Intrinsic::convertfsi:
  case Intrinsic::convertfui:
  case Intrinsic::convertsif:
  case Intrinsic::convertuif:
  case Intrinsic::convertss:
  case Intrinsic::convertsu:
  case Intrinsic::convertus:
  case Intrinsic::convertuu: {
    ISD::CvtCode Code = ISD::CVT_INVALID;
    switch (Intrinsic) {
    default: llvm_unreachable("Impossible intrinsic");  // Can't reach here.
    case Intrinsic::convertff:  Code = ISD::CVT_FF; break;
    case Intrinsic::convertfsi: Code = ISD::CVT_FS; break;
    case Intrinsic::convertfui: Code = ISD::CVT_FU; break;
    case Intrinsic::convertsif: Code = ISD::CVT_SF; break;
    case Intrinsic::convertuif: Code = ISD::CVT_UF; break;
    case Intrinsic::convertss:  Code = ISD::CVT_SS; break;
    case Intrinsic::convertsu:  Code = ISD::CVT_SU; break;
    case Intrinsic::convertus:  Code = ISD::CVT_US; break;
    case Intrinsic::convertuu:  Code = ISD::CVT_UU; break;
    }
    EVT DestVT = TLI.getValueType(DAG.getDataLayout(), I.getType());
    const Value *Op1 = I.getArgOperand(0);
    Res = DAG.getConvertRndSat(DestVT, sdl, getValue(Op1),
                               DAG.getValueType(DestVT),
                               DAG.getValueType(getValue(Op1).getValueType()),
                               getValue(I.getArgOperand(1)),
                               getValue(I.getArgOperand(2)),
                               Code);
    setValue(&I, Res);
    return nullptr;
  }
  case Intrinsic::powi:
    setValue(&I, ExpandPowI(sdl, getValue(I.getArgOperand(0)),
                            getValue(I.getArgOperand(1)), DAG));
    return nullptr;
  case Intrinsic::log:
    setValue(&I, expandLog(sdl, getValue(I.getArgOperand(0)), DAG, TLI));
    return nullptr;
  case Intrinsic::log2:
    setValue(&I, expandLog2(sdl, getValue(I.getArgOperand(0)), DAG, TLI));
    return nullptr;
  case Intrinsic::log10:
    setValue(&I, expandLog10(sdl, getValue(I.getArgOperand(0)), DAG, TLI));
    return nullptr;
  case Intrinsic::exp:
    setValue(&I, expandExp(sdl, getValue(I.getArgOperand(0)), DAG, TLI));
    return nullptr;
  case Intrinsic::exp2:
    setValue(&I, expandExp2(sdl, getValue(I.getArgOperand(0)), DAG, TLI));
    return nullptr;
  case Intrinsic::pow:
    setValue(&I, expandPow(sdl, getValue(I.getArgOperand(0)),
                           getValue(I.getArgOperand(1)), DAG, TLI));
    return nullptr;
  case Intrinsic::sqrt:
  case Intrinsic::fabs:
  case Intrinsic::sin:
  case Intrinsic::cos:
  case Intrinsic::floor:
  case Intrinsic::ceil:
  case Intrinsic::trunc:
  case Intrinsic::rint:
  case Intrinsic::nearbyint:
  case Intrinsic::round: {
    unsigned Opcode;
    switch (Intrinsic) {
    default: llvm_unreachable("Impossible intrinsic");  // Can't reach here.
    case Intrinsic::sqrt:      Opcode = ISD::FSQRT;      break;
    case Intrinsic::fabs:      Opcode = ISD::FABS;       break;
    case Intrinsic::sin:       Opcode = ISD::FSIN;       break;
    case Intrinsic::cos:       Opcode = ISD::FCOS;       break;
    case Intrinsic::floor:     Opcode = ISD::FFLOOR;     break;
    case Intrinsic::ceil:      Opcode = ISD::FCEIL;      break;
    case Intrinsic::trunc:     Opcode = ISD::FTRUNC;     break;
    case Intrinsic::rint:      Opcode = ISD::FRINT;      break;
    case Intrinsic::nearbyint: Opcode = ISD::FNEARBYINT; break;
    case Intrinsic::round:     Opcode = ISD::FROUND;     break;
    }

    setValue(&I, DAG.getNode(Opcode, sdl,
                             getValue(I.getArgOperand(0)).getValueType(),
                             getValue(I.getArgOperand(0))));
    return nullptr;
  }
  case Intrinsic::minnum:
    setValue(&I, DAG.getNode(ISD::FMINNUM, sdl,
                             getValue(I.getArgOperand(0)).getValueType(),
                             getValue(I.getArgOperand(0)),
                             getValue(I.getArgOperand(1))));
    return nullptr;
  case Intrinsic::maxnum:
    setValue(&I, DAG.getNode(ISD::FMAXNUM, sdl,
                             getValue(I.getArgOperand(0)).getValueType(),
                             getValue(I.getArgOperand(0)),
                             getValue(I.getArgOperand(1))));
    return nullptr;
  case Intrinsic::copysign:
    setValue(&I, DAG.getNode(ISD::FCOPYSIGN, sdl,
                             getValue(I.getArgOperand(0)).getValueType(),
                             getValue(I.getArgOperand(0)),
                             getValue(I.getArgOperand(1))));
    return nullptr;
  case Intrinsic::fma:
    setValue(&I, DAG.getNode(ISD::FMA, sdl,
                             getValue(I.getArgOperand(0)).getValueType(),
                             getValue(I.getArgOperand(0)),
                             getValue(I.getArgOperand(1)),
                             getValue(I.getArgOperand(2))));
    return nullptr;
  case Intrinsic::fmuladd: {
    EVT VT = TLI.getValueType(DAG.getDataLayout(), I.getType());
    if (TM.Options.AllowFPOpFusion != FPOpFusion::Strict &&
        TLI.isFMAFasterThanFMulAndFAdd(VT)) {
      setValue(&I, DAG.getNode(ISD::FMA, sdl,
                               getValue(I.getArgOperand(0)).getValueType(),
                               getValue(I.getArgOperand(0)),
                               getValue(I.getArgOperand(1)),
                               getValue(I.getArgOperand(2))));
    } else {
      SDValue Mul = DAG.getNode(ISD::FMUL, sdl,
                                getValue(I.getArgOperand(0)).getValueType(),
                                getValue(I.getArgOperand(0)),
                                getValue(I.getArgOperand(1)));
      SDValue Add = DAG.getNode(ISD::FADD, sdl,
                                getValue(I.getArgOperand(0)).getValueType(),
                                Mul,
                                getValue(I.getArgOperand(2)));
      setValue(&I, Add);
    }
    return nullptr;
  }
  case Intrinsic::convert_to_fp16:
    setValue(&I, DAG.getNode(ISD::BITCAST, sdl, MVT::i16,
                             DAG.getNode(ISD::FP_ROUND, sdl, MVT::f16,
                                         getValue(I.getArgOperand(0)),
                                         DAG.getTargetConstant(0, sdl,
                                                               MVT::i32))));
    return nullptr;
  case Intrinsic::convert_from_fp16:
    setValue(&I, DAG.getNode(ISD::FP_EXTEND, sdl,
                             TLI.getValueType(DAG.getDataLayout(), I.getType()),
                             DAG.getNode(ISD::BITCAST, sdl, MVT::f16,
                                         getValue(I.getArgOperand(0)))));
    return nullptr;
  case Intrinsic::pcmarker: {
    SDValue Tmp = getValue(I.getArgOperand(0));
    DAG.setRoot(DAG.getNode(ISD::PCMARKER, sdl, MVT::Other, getRoot(), Tmp));
    return nullptr;
  }
  case Intrinsic::readcyclecounter: {
    SDValue Op = getRoot();
    Res = DAG.getNode(ISD::READCYCLECOUNTER, sdl,
                      DAG.getVTList(MVT::i64, MVT::Other), Op);
    setValue(&I, Res);
    DAG.setRoot(Res.getValue(1));
    return nullptr;
  }
  case Intrinsic::bswap:
    setValue(&I, DAG.getNode(ISD::BSWAP, sdl,
                             getValue(I.getArgOperand(0)).getValueType(),
                             getValue(I.getArgOperand(0))));
    return nullptr;
  case Intrinsic::cttz: {
    SDValue Arg = getValue(I.getArgOperand(0));
    ConstantInt *CI = cast<ConstantInt>(I.getArgOperand(1));
    EVT Ty = Arg.getValueType();
    setValue(&I, DAG.getNode(CI->isZero() ? ISD::CTTZ : ISD::CTTZ_ZERO_UNDEF,
                             sdl, Ty, Arg));
    return nullptr;
  }
  case Intrinsic::ctlz: {
    SDValue Arg = getValue(I.getArgOperand(0));
    ConstantInt *CI = cast<ConstantInt>(I.getArgOperand(1));
    EVT Ty = Arg.getValueType();
    setValue(&I, DAG.getNode(CI->isZero() ? ISD::CTLZ : ISD::CTLZ_ZERO_UNDEF,
                             sdl, Ty, Arg));
    return nullptr;
  }
  case Intrinsic::ctpop: {
    SDValue Arg = getValue(I.getArgOperand(0));
    EVT Ty = Arg.getValueType();
    setValue(&I, DAG.getNode(ISD::CTPOP, sdl, Ty, Arg));
    return nullptr;
  }
  case Intrinsic::stacksave: {
    SDValue Op = getRoot();
    Res = DAG.getNode(
        ISD::STACKSAVE, sdl,
        DAG.getVTList(TLI.getPointerTy(DAG.getDataLayout()), MVT::Other), Op);
    setValue(&I, Res);
    DAG.setRoot(Res.getValue(1));
    return nullptr;
  }
  case Intrinsic::stackrestore: {
    Res = getValue(I.getArgOperand(0));
    DAG.setRoot(DAG.getNode(ISD::STACKRESTORE, sdl, MVT::Other, getRoot(), Res));
    return nullptr;
  }
  case Intrinsic::stackprotector: {
    // Emit code into the DAG to store the stack guard onto the stack.
    MachineFunction &MF = DAG.getMachineFunction();
    MachineFrameInfo *MFI = MF.getFrameInfo();
    EVT PtrTy = TLI.getPointerTy(DAG.getDataLayout());
    SDValue Src, Chain = getRoot();
    const Value *Ptr = cast<LoadInst>(I.getArgOperand(0))->getPointerOperand();
    const GlobalVariable *GV = dyn_cast<GlobalVariable>(Ptr);

    // See if Ptr is a bitcast. If it is, look through it and see if we can get
    // global variable __stack_chk_guard.
    if (!GV)
      if (const Operator *BC = dyn_cast<Operator>(Ptr))
        if (BC->getOpcode() == Instruction::BitCast)
          GV = dyn_cast<GlobalVariable>(BC->getOperand(0));

    if (GV && TLI.useLoadStackGuardNode()) {
      // Emit a LOAD_STACK_GUARD node.
      MachineSDNode *Node = DAG.getMachineNode(TargetOpcode::LOAD_STACK_GUARD,
                                               sdl, PtrTy, Chain);
      MachinePointerInfo MPInfo(GV);
      MachineInstr::mmo_iterator MemRefs = MF.allocateMemRefsArray(1);
      unsigned Flags = MachineMemOperand::MOLoad |
                       MachineMemOperand::MOInvariant;
      *MemRefs = MF.getMachineMemOperand(MPInfo, Flags,
                                         PtrTy.getSizeInBits() / 8,
                                         DAG.getEVTAlignment(PtrTy));
      Node->setMemRefs(MemRefs, MemRefs + 1);

      // Copy the guard value to a virtual register so that it can be
      // retrieved in the epilogue.
      Src = SDValue(Node, 0);
      const TargetRegisterClass *RC =
          TLI.getRegClassFor(Src.getSimpleValueType());
      unsigned Reg = MF.getRegInfo().createVirtualRegister(RC);

      SPDescriptor.setGuardReg(Reg);
      Chain = DAG.getCopyToReg(Chain, sdl, Reg, Src);
    } else {
      Src = getValue(I.getArgOperand(0));   // The guard's value.
    }

    AllocaInst *Slot = cast<AllocaInst>(I.getArgOperand(1));

    int FI = FuncInfo.StaticAllocaMap[Slot];
    MFI->setStackProtectorIndex(FI);

    SDValue FIN = DAG.getFrameIndex(FI, PtrTy);

    // Store the stack protector onto the stack.
    Res = DAG.getStore(Chain, sdl, Src, FIN,
                       MachinePointerInfo::getFixedStack(FI),
                       true, false, 0);
    setValue(&I, Res);
    DAG.setRoot(Res);
    return nullptr;
  }
  case Intrinsic::objectsize: {
    // If we don't know by now, we're never going to know.
    ConstantInt *CI = dyn_cast<ConstantInt>(I.getArgOperand(1));

    assert(CI && "Non-constant type in __builtin_object_size?");

    SDValue Arg = getValue(I.getCalledValue());
    EVT Ty = Arg.getValueType();

    if (CI->isZero())
      Res = DAG.getConstant(-1ULL, sdl, Ty);
    else
      Res = DAG.getConstant(0, sdl, Ty);

    setValue(&I, Res);
    return nullptr;
  }
  case Intrinsic::annotation:
  case Intrinsic::ptr_annotation:
    // Drop the intrinsic, but forward the value
    setValue(&I, getValue(I.getOperand(0)));
    return nullptr;
  case Intrinsic::assume:
  case Intrinsic::var_annotation:
    // Discard annotate attributes and assumptions
    return nullptr;

  case Intrinsic::init_trampoline: {
    const Function *F = cast<Function>(I.getArgOperand(1)->stripPointerCasts());

    SDValue Ops[6];
    Ops[0] = getRoot();
    Ops[1] = getValue(I.getArgOperand(0));
    Ops[2] = getValue(I.getArgOperand(1));
    Ops[3] = getValue(I.getArgOperand(2));
    Ops[4] = DAG.getSrcValue(I.getArgOperand(0));
    Ops[5] = DAG.getSrcValue(F);

    Res = DAG.getNode(ISD::INIT_TRAMPOLINE, sdl, MVT::Other, Ops);

    DAG.setRoot(Res);
    return nullptr;
  }
  case Intrinsic::adjust_trampoline: {
    setValue(&I, DAG.getNode(ISD::ADJUST_TRAMPOLINE, sdl,
                             TLI.getPointerTy(DAG.getDataLayout()),
                             getValue(I.getArgOperand(0))));
    return nullptr;
  }
  case Intrinsic::gcroot:
    if (GFI) {
      const Value *Alloca = I.getArgOperand(0)->stripPointerCasts();
      const Constant *TypeMap = cast<Constant>(I.getArgOperand(1));

      FrameIndexSDNode *FI = cast<FrameIndexSDNode>(getValue(Alloca).getNode());
      GFI->addStackRoot(FI->getIndex(), TypeMap);
    }
    return nullptr;
  case Intrinsic::gcread:
  case Intrinsic::gcwrite:
    llvm_unreachable("GC failed to lower gcread/gcwrite intrinsics!");
  case Intrinsic::flt_rounds:
    setValue(&I, DAG.getNode(ISD::FLT_ROUNDS_, sdl, MVT::i32));
    return nullptr;

  case Intrinsic::expect: {
    // Just replace __builtin_expect(exp, c) with EXP.
    setValue(&I, getValue(I.getArgOperand(0)));
    return nullptr;
  }

  case Intrinsic::debugtrap:
  case Intrinsic::trap: {
    StringRef TrapFuncName =
        I.getAttributes()
            .getAttribute(AttributeSet::FunctionIndex, "trap-func-name")
            .getValueAsString();
    if (TrapFuncName.empty()) {
      ISD::NodeType Op = (Intrinsic == Intrinsic::trap) ?
        ISD::TRAP : ISD::DEBUGTRAP;
      DAG.setRoot(DAG.getNode(Op, sdl,MVT::Other, getRoot()));
      return nullptr;
    }
    TargetLowering::ArgListTy Args;

    TargetLowering::CallLoweringInfo CLI(DAG);
    CLI.setDebugLoc(sdl).setChain(getRoot()).setCallee(
        CallingConv::C, I.getType(),
        DAG.getExternalSymbol(TrapFuncName.data(),
                              TLI.getPointerTy(DAG.getDataLayout())),
        std::move(Args), 0);

    std::pair<SDValue, SDValue> Result = TLI.LowerCallTo(CLI);
    DAG.setRoot(Result.second);
    return nullptr;
  }

  case Intrinsic::uadd_with_overflow:
  case Intrinsic::sadd_with_overflow:
  case Intrinsic::usub_with_overflow:
  case Intrinsic::ssub_with_overflow:
  case Intrinsic::umul_with_overflow:
  case Intrinsic::smul_with_overflow: {
    ISD::NodeType Op;
    switch (Intrinsic) {
    default: llvm_unreachable("Impossible intrinsic");  // Can't reach here.
    case Intrinsic::uadd_with_overflow: Op = ISD::UADDO; break;
    case Intrinsic::sadd_with_overflow: Op = ISD::SADDO; break;
    case Intrinsic::usub_with_overflow: Op = ISD::USUBO; break;
    case Intrinsic::ssub_with_overflow: Op = ISD::SSUBO; break;
    case Intrinsic::umul_with_overflow: Op = ISD::UMULO; break;
    case Intrinsic::smul_with_overflow: Op = ISD::SMULO; break;
    }
    SDValue Op1 = getValue(I.getArgOperand(0));
    SDValue Op2 = getValue(I.getArgOperand(1));

    SDVTList VTs = DAG.getVTList(Op1.getValueType(), MVT::i1);
    setValue(&I, DAG.getNode(Op, sdl, VTs, Op1, Op2));
    return nullptr;
  }
  case Intrinsic::prefetch: {
    SDValue Ops[5];
    unsigned rw = cast<ConstantInt>(I.getArgOperand(1))->getZExtValue();
    Ops[0] = getRoot();
    Ops[1] = getValue(I.getArgOperand(0));
    Ops[2] = getValue(I.getArgOperand(1));
    Ops[3] = getValue(I.getArgOperand(2));
    Ops[4] = getValue(I.getArgOperand(3));
    DAG.setRoot(DAG.getMemIntrinsicNode(ISD::PREFETCH, sdl,
                                        DAG.getVTList(MVT::Other), Ops,
                                        EVT::getIntegerVT(*Context, 8),
                                        MachinePointerInfo(I.getArgOperand(0)),
                                        0, /* align */
                                        false, /* volatile */
                                        rw==0, /* read */
                                        rw==1)); /* write */
    return nullptr;
  }
  case Intrinsic::lifetime_start:
  case Intrinsic::lifetime_end: {
    bool IsStart = (Intrinsic == Intrinsic::lifetime_start);
    // Stack coloring is not enabled in O0, discard region information.
    if (TM.getOptLevel() == CodeGenOpt::None)
      return nullptr;

    SmallVector<Value *, 4> Allocas;
    GetUnderlyingObjects(I.getArgOperand(1), Allocas, *DL);

    for (SmallVectorImpl<Value*>::iterator Object = Allocas.begin(),
           E = Allocas.end(); Object != E; ++Object) {
      AllocaInst *LifetimeObject = dyn_cast_or_null<AllocaInst>(*Object);

      // Could not find an Alloca.
      if (!LifetimeObject)
        continue;

      // First check that the Alloca is static, otherwise it won't have a
      // valid frame index.
      auto SI = FuncInfo.StaticAllocaMap.find(LifetimeObject);
      if (SI == FuncInfo.StaticAllocaMap.end())
        return nullptr;

      int FI = SI->second;

      SDValue Ops[2];
      Ops[0] = getRoot();
      Ops[1] =
          DAG.getFrameIndex(FI, TLI.getPointerTy(DAG.getDataLayout()), true);
      unsigned Opcode = (IsStart ? ISD::LIFETIME_START : ISD::LIFETIME_END);

      Res = DAG.getNode(Opcode, sdl, MVT::Other, Ops);
      DAG.setRoot(Res);
    }
    return nullptr;
  }
  case Intrinsic::invariant_start:
    // Discard region information.
    setValue(&I, DAG.getUNDEF(TLI.getPointerTy(DAG.getDataLayout())));
    return nullptr;
  case Intrinsic::invariant_end:
    // Discard region information.
    return nullptr;
  case Intrinsic::stackprotectorcheck: {
    // Do not actually emit anything for this basic block. Instead we initialize
    // the stack protector descriptor and export the guard variable so we can
    // access it in FinishBasicBlock.
    const BasicBlock *BB = I.getParent();
    SPDescriptor.initialize(BB, FuncInfo.MBBMap[BB], I);
    ExportFromCurrentBlock(SPDescriptor.getGuard());

    // Flush our exports since we are going to process a terminator.
    (void)getControlRoot();
    return nullptr;
  }
  case Intrinsic::clear_cache:
    return TLI.getClearCacheBuiltinName();
  case Intrinsic::eh_actions:
    setValue(&I, DAG.getUNDEF(TLI.getPointerTy(DAG.getDataLayout())));
    return nullptr;
  case Intrinsic::donothing:
    // ignore
    return nullptr;
  case Intrinsic::experimental_stackmap: {
    visitStackmap(I);
    return nullptr;
  }
  case Intrinsic::experimental_patchpoint_void:
  case Intrinsic::experimental_patchpoint_i64: {
    visitPatchpoint(&I);
    return nullptr;
  }
  case Intrinsic::experimental_gc_statepoint: {
    visitStatepoint(I);
    return nullptr;
  }
  case Intrinsic::experimental_gc_result_int:
  case Intrinsic::experimental_gc_result_float:
  case Intrinsic::experimental_gc_result_ptr:
  case Intrinsic::experimental_gc_result: {
    visitGCResult(I);
    return nullptr;
  }
  case Intrinsic::experimental_gc_relocate: {
    visitGCRelocate(I);
    return nullptr;
  }
  case Intrinsic::instrprof_increment:
    llvm_unreachable("instrprof failed to lower an increment");

  case Intrinsic::localescape: {
    MachineFunction &MF = DAG.getMachineFunction();
    const TargetInstrInfo *TII = DAG.getSubtarget().getInstrInfo();

    // Directly emit some LOCAL_ESCAPE machine instrs. Label assignment emission
    // is the same on all targets.
    for (unsigned Idx = 0, E = I.getNumArgOperands(); Idx < E; ++Idx) {
      Value *Arg = I.getArgOperand(Idx)->stripPointerCasts();
      if (isa<ConstantPointerNull>(Arg))
        continue; // Skip null pointers. They represent a hole in index space.
      AllocaInst *Slot = cast<AllocaInst>(Arg);
      assert(FuncInfo.StaticAllocaMap.count(Slot) &&
             "can only escape static allocas");
      int FI = FuncInfo.StaticAllocaMap[Slot];
      MCSymbol *FrameAllocSym =
          MF.getMMI().getContext().getOrCreateFrameAllocSymbol(
              GlobalValue::getRealLinkageName(MF.getName()), Idx);
      BuildMI(*FuncInfo.MBB, FuncInfo.InsertPt, dl,
              TII->get(TargetOpcode::LOCAL_ESCAPE))
          .addSym(FrameAllocSym)
          .addFrameIndex(FI);
    }

    return nullptr;
  }

  case Intrinsic::localrecover: {
    // i8* @llvm.localrecover(i8* %fn, i8* %fp, i32 %idx)
    MachineFunction &MF = DAG.getMachineFunction();
    MVT PtrVT = TLI.getPointerTy(DAG.getDataLayout(), 0);

    // Get the symbol that defines the frame offset.
    auto *Fn = cast<Function>(I.getArgOperand(0)->stripPointerCasts());
    auto *Idx = cast<ConstantInt>(I.getArgOperand(2));
    unsigned IdxVal = unsigned(Idx->getLimitedValue(INT_MAX));
    MCSymbol *FrameAllocSym =
        MF.getMMI().getContext().getOrCreateFrameAllocSymbol(
            GlobalValue::getRealLinkageName(Fn->getName()), IdxVal);

    // Create a MCSymbol for the label to avoid any target lowering
    // that would make this PC relative.
    SDValue OffsetSym = DAG.getMCSymbol(FrameAllocSym, PtrVT);
    SDValue OffsetVal =
        DAG.getNode(ISD::LOCAL_RECOVER, sdl, PtrVT, OffsetSym);

    // Add the offset to the FP.
    Value *FP = I.getArgOperand(1);
    SDValue FPVal = getValue(FP);
    SDValue Add = DAG.getNode(ISD::ADD, sdl, PtrVT, FPVal, OffsetVal);
    setValue(&I, Add);

    return nullptr;
  }
  case Intrinsic::eh_begincatch:
  case Intrinsic::eh_endcatch:
    llvm_unreachable("begin/end catch intrinsics not lowered in codegen");
  case Intrinsic::eh_exceptioncode: {
    unsigned Reg = TLI.getExceptionPointerRegister();
    assert(Reg && "cannot get exception code on this platform");
    MVT PtrVT = TLI.getPointerTy(DAG.getDataLayout());
    const TargetRegisterClass *PtrRC = TLI.getRegClassFor(PtrVT);
    assert(FuncInfo.MBB->isLandingPad() && "eh.exceptioncode in non-lpad");
    unsigned VReg = FuncInfo.MBB->addLiveIn(Reg, PtrRC);
    SDValue N =
        DAG.getCopyFromReg(DAG.getEntryNode(), getCurSDLoc(), VReg, PtrVT);
    N = DAG.getZExtOrTrunc(N, getCurSDLoc(), MVT::i32);
    setValue(&I, N);
    return nullptr;
  }
  }
}

std::pair<SDValue, SDValue>
SelectionDAGBuilder::lowerInvokable(TargetLowering::CallLoweringInfo &CLI,
                                    MachineBasicBlock *LandingPad) {
  MachineModuleInfo &MMI = DAG.getMachineFunction().getMMI();
  MCSymbol *BeginLabel = nullptr;

  if (LandingPad) {
    // Insert a label before the invoke call to mark the try range.  This can be
    // used to detect deletion of the invoke via the MachineModuleInfo.
    BeginLabel = MMI.getContext().createTempSymbol();

    // For SjLj, keep track of which landing pads go with which invokes
    // so as to maintain the ordering of pads in the LSDA.
    unsigned CallSiteIndex = MMI.getCurrentCallSite();
    if (CallSiteIndex) {
      MMI.setCallSiteBeginLabel(BeginLabel, CallSiteIndex);
      LPadToCallSiteMap[LandingPad].push_back(CallSiteIndex);

      // Now that the call site is handled, stop tracking it.
      MMI.setCurrentCallSite(0);
    }

    // Both PendingLoads and PendingExports must be flushed here;
    // this call might not return.
    (void)getRoot();
    DAG.setRoot(DAG.getEHLabel(getCurSDLoc(), getControlRoot(), BeginLabel));

    CLI.setChain(getRoot());
  }
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  std::pair<SDValue, SDValue> Result = TLI.LowerCallTo(CLI);

  assert((CLI.IsTailCall || Result.second.getNode()) &&
         "Non-null chain expected with non-tail call!");
  assert((Result.second.getNode() || !Result.first.getNode()) &&
         "Null value expected with tail call!");

  if (!Result.second.getNode()) {
    // As a special case, a null chain means that a tail call has been emitted
    // and the DAG root is already updated.
    HasTailCall = true;

    // Since there's no actual continuation from this block, nothing can be
    // relying on us setting vregs for them.
    PendingExports.clear();
  } else {
    DAG.setRoot(Result.second);
  }

  if (LandingPad) {
    // Insert a label at the end of the invoke call to mark the try range.  This
    // can be used to detect deletion of the invoke via the MachineModuleInfo.
    MCSymbol *EndLabel = MMI.getContext().createTempSymbol();
    DAG.setRoot(DAG.getEHLabel(getCurSDLoc(), getRoot(), EndLabel));

    // Inform MachineModuleInfo of range.
    MMI.addInvoke(LandingPad, BeginLabel, EndLabel);
  }

  return Result;
}

void SelectionDAGBuilder::LowerCallTo(ImmutableCallSite CS, SDValue Callee,
                                      bool isTailCall,
                                      MachineBasicBlock *LandingPad) {
  PointerType *PT = cast<PointerType>(CS.getCalledValue()->getType());
  FunctionType *FTy = cast<FunctionType>(PT->getElementType());
  Type *RetTy = FTy->getReturnType();

  TargetLowering::ArgListTy Args;
  TargetLowering::ArgListEntry Entry;
  Args.reserve(CS.arg_size());

  for (ImmutableCallSite::arg_iterator i = CS.arg_begin(), e = CS.arg_end();
       i != e; ++i) {
    const Value *V = *i;

    // Skip empty types
    if (V->getType()->isEmptyTy())
      continue;

    SDValue ArgNode = getValue(V);
    Entry.Node = ArgNode; Entry.Ty = V->getType();

    // Skip the first return-type Attribute to get to params.
    Entry.setAttributes(&CS, i - CS.arg_begin() + 1);
    Args.push_back(Entry);

    // If we have an explicit sret argument that is an Instruction, (i.e., it
    // might point to function-local memory), we can't meaningfully tail-call.
    if (Entry.isSRet && isa<Instruction>(V))
      isTailCall = false;
  }

  // Check if target-independent constraints permit a tail call here.
  // Target-dependent constraints are checked within TLI->LowerCallTo.
  if (isTailCall && !isInTailCallPosition(CS, DAG.getTarget()))
    isTailCall = false;

  TargetLowering::CallLoweringInfo CLI(DAG);
  CLI.setDebugLoc(getCurSDLoc()).setChain(getRoot())
    .setCallee(RetTy, FTy, Callee, std::move(Args), CS)
    .setTailCall(isTailCall);
  std::pair<SDValue,SDValue> Result = lowerInvokable(CLI, LandingPad);

  if (Result.first.getNode())
    setValue(CS.getInstruction(), Result.first);
}

/// IsOnlyUsedInZeroEqualityComparison - Return true if it only matters that the
/// value is equal or not-equal to zero.
static bool IsOnlyUsedInZeroEqualityComparison(const Value *V) {
  for (const User *U : V->users()) {
    if (const ICmpInst *IC = dyn_cast<ICmpInst>(U))
      if (IC->isEquality())
        if (const Constant *C = dyn_cast<Constant>(IC->getOperand(1)))
          if (C->isNullValue())
            continue;
    // Unknown instruction.
    return false;
  }
  return true;
}

static SDValue getMemCmpLoad(const Value *PtrVal, MVT LoadVT,
                             Type *LoadTy,
                             SelectionDAGBuilder &Builder) {

  // Check to see if this load can be trivially constant folded, e.g. if the
  // input is from a string literal.
  if (const Constant *LoadInput = dyn_cast<Constant>(PtrVal)) {
    // Cast pointer to the type we really want to load.
    LoadInput = ConstantExpr::getBitCast(const_cast<Constant *>(LoadInput),
                                         PointerType::getUnqual(LoadTy));

    if (const Constant *LoadCst = ConstantFoldLoadFromConstPtr(
            const_cast<Constant *>(LoadInput), *Builder.DL))
      return Builder.getValue(LoadCst);
  }

  // Otherwise, we have to emit the load.  If the pointer is to unfoldable but
  // still constant memory, the input chain can be the entry node.
  SDValue Root;
  bool ConstantMemory = false;

  // Do not serialize (non-volatile) loads of constant memory with anything.
  if (Builder.AA->pointsToConstantMemory(PtrVal)) {
    Root = Builder.DAG.getEntryNode();
    ConstantMemory = true;
  } else {
    // Do not serialize non-volatile loads against each other.
    Root = Builder.DAG.getRoot();
  }

  SDValue Ptr = Builder.getValue(PtrVal);
  SDValue LoadVal = Builder.DAG.getLoad(LoadVT, Builder.getCurSDLoc(), Root,
                                        Ptr, MachinePointerInfo(PtrVal),
                                        false /*volatile*/,
                                        false /*nontemporal*/,
                                        false /*isinvariant*/, 1 /* align=1 */);

  if (!ConstantMemory)
    Builder.PendingLoads.push_back(LoadVal.getValue(1));
  return LoadVal;
}

/// processIntegerCallValue - Record the value for an instruction that
/// produces an integer result, converting the type where necessary.
void SelectionDAGBuilder::processIntegerCallValue(const Instruction &I,
                                                  SDValue Value,
                                                  bool IsSigned) {
  EVT VT = DAG.getTargetLoweringInfo().getValueType(DAG.getDataLayout(),
                                                    I.getType(), true);
  if (IsSigned)
    Value = DAG.getSExtOrTrunc(Value, getCurSDLoc(), VT);
  else
    Value = DAG.getZExtOrTrunc(Value, getCurSDLoc(), VT);
  setValue(&I, Value);
}

/// visitMemCmpCall - See if we can lower a call to memcmp in an optimized form.
/// If so, return true and lower it, otherwise return false and it will be
/// lowered like a normal call.
bool SelectionDAGBuilder::visitMemCmpCall(const CallInst &I) {
  // Verify that the prototype makes sense.  int memcmp(void*,void*,size_t)
  if (I.getNumArgOperands() != 3)
    return false;

  const Value *LHS = I.getArgOperand(0), *RHS = I.getArgOperand(1);
  if (!LHS->getType()->isPointerTy() || !RHS->getType()->isPointerTy() ||
      !I.getArgOperand(2)->getType()->isIntegerTy() ||
      !I.getType()->isIntegerTy())
    return false;

  const Value *Size = I.getArgOperand(2);
  const ConstantInt *CSize = dyn_cast<ConstantInt>(Size);
  if (CSize && CSize->getZExtValue() == 0) {
    EVT CallVT = DAG.getTargetLoweringInfo().getValueType(DAG.getDataLayout(),
                                                          I.getType(), true);
    setValue(&I, DAG.getConstant(0, getCurSDLoc(), CallVT));
    return true;
  }

  const TargetSelectionDAGInfo &TSI = DAG.getSelectionDAGInfo();
  std::pair<SDValue, SDValue> Res =
    TSI.EmitTargetCodeForMemcmp(DAG, getCurSDLoc(), DAG.getRoot(),
                                getValue(LHS), getValue(RHS), getValue(Size),
                                MachinePointerInfo(LHS),
                                MachinePointerInfo(RHS));
  if (Res.first.getNode()) {
    processIntegerCallValue(I, Res.first, true);
    PendingLoads.push_back(Res.second);
    return true;
  }

  // memcmp(S1,S2,2) != 0 -> (*(short*)LHS != *(short*)RHS)  != 0
  // memcmp(S1,S2,4) != 0 -> (*(int*)LHS != *(int*)RHS)  != 0
  if (CSize && IsOnlyUsedInZeroEqualityComparison(&I)) {
    bool ActuallyDoIt = true;
    MVT LoadVT;
    Type *LoadTy;
    switch (CSize->getZExtValue()) {
    default:
      LoadVT = MVT::Other;
      LoadTy = nullptr;
      ActuallyDoIt = false;
      break;
    case 2:
      LoadVT = MVT::i16;
      LoadTy = Type::getInt16Ty(CSize->getContext());
      break;
    case 4:
      LoadVT = MVT::i32;
      LoadTy = Type::getInt32Ty(CSize->getContext());
      break;
    case 8:
      LoadVT = MVT::i64;
      LoadTy = Type::getInt64Ty(CSize->getContext());
      break;
        /*
    case 16:
      LoadVT = MVT::v4i32;
      LoadTy = Type::getInt32Ty(CSize->getContext());
      LoadTy = VectorType::get(LoadTy, 4);
      break;
         */
    }

    // This turns into unaligned loads.  We only do this if the target natively
    // supports the MVT we'll be loading or if it is small enough (<= 4) that
    // we'll only produce a small number of byte loads.

    // Require that we can find a legal MVT, and only do this if the target
    // supports unaligned loads of that type.  Expanding into byte loads would
    // bloat the code.
    const TargetLowering &TLI = DAG.getTargetLoweringInfo();
    if (ActuallyDoIt && CSize->getZExtValue() > 4) {
      unsigned DstAS = LHS->getType()->getPointerAddressSpace();
      unsigned SrcAS = RHS->getType()->getPointerAddressSpace();
      // TODO: Handle 5 byte compare as 4-byte + 1 byte.
      // TODO: Handle 8 byte compare on x86-32 as two 32-bit loads.
      // TODO: Check alignment of src and dest ptrs.
      if (!TLI.isTypeLegal(LoadVT) ||
          !TLI.allowsMisalignedMemoryAccesses(LoadVT, SrcAS) ||
          !TLI.allowsMisalignedMemoryAccesses(LoadVT, DstAS))
        ActuallyDoIt = false;
    }

    if (ActuallyDoIt) {
      SDValue LHSVal = getMemCmpLoad(LHS, LoadVT, LoadTy, *this);
      SDValue RHSVal = getMemCmpLoad(RHS, LoadVT, LoadTy, *this);

      SDValue Res = DAG.getSetCC(getCurSDLoc(), MVT::i1, LHSVal, RHSVal,
                                 ISD::SETNE);
      processIntegerCallValue(I, Res, false);
      return true;
    }
  }


  return false;
}

/// visitMemChrCall -- See if we can lower a memchr call into an optimized
/// form.  If so, return true and lower it, otherwise return false and it
/// will be lowered like a normal call.
bool SelectionDAGBuilder::visitMemChrCall(const CallInst &I) {
  // Verify that the prototype makes sense.  void *memchr(void *, int, size_t)
  if (I.getNumArgOperands() != 3)
    return false;

  const Value *Src = I.getArgOperand(0);
  const Value *Char = I.getArgOperand(1);
  const Value *Length = I.getArgOperand(2);
  if (!Src->getType()->isPointerTy() ||
      !Char->getType()->isIntegerTy() ||
      !Length->getType()->isIntegerTy() ||
      !I.getType()->isPointerTy())
    return false;

  const TargetSelectionDAGInfo &TSI = DAG.getSelectionDAGInfo();
  std::pair<SDValue, SDValue> Res =
    TSI.EmitTargetCodeForMemchr(DAG, getCurSDLoc(), DAG.getRoot(),
                                getValue(Src), getValue(Char), getValue(Length),
                                MachinePointerInfo(Src));
  if (Res.first.getNode()) {
    setValue(&I, Res.first);
    PendingLoads.push_back(Res.second);
    return true;
  }

  return false;
}

/// visitStrCpyCall -- See if we can lower a strcpy or stpcpy call into an
/// optimized form.  If so, return true and lower it, otherwise return false
/// and it will be lowered like a normal call.
bool SelectionDAGBuilder::visitStrCpyCall(const CallInst &I, bool isStpcpy) {
  // Verify that the prototype makes sense.  char *strcpy(char *, char *)
  if (I.getNumArgOperands() != 2)
    return false;

  const Value *Arg0 = I.getArgOperand(0), *Arg1 = I.getArgOperand(1);
  if (!Arg0->getType()->isPointerTy() ||
      !Arg1->getType()->isPointerTy() ||
      !I.getType()->isPointerTy())
    return false;

  const TargetSelectionDAGInfo &TSI = DAG.getSelectionDAGInfo();
  std::pair<SDValue, SDValue> Res =
    TSI.EmitTargetCodeForStrcpy(DAG, getCurSDLoc(), getRoot(),
                                getValue(Arg0), getValue(Arg1),
                                MachinePointerInfo(Arg0),
                                MachinePointerInfo(Arg1), isStpcpy);
  if (Res.first.getNode()) {
    setValue(&I, Res.first);
    DAG.setRoot(Res.second);
    return true;
  }

  return false;
}

/// visitStrCmpCall - See if we can lower a call to strcmp in an optimized form.
/// If so, return true and lower it, otherwise return false and it will be
/// lowered like a normal call.
bool SelectionDAGBuilder::visitStrCmpCall(const CallInst &I) {
  // Verify that the prototype makes sense.  int strcmp(void*,void*)
  if (I.getNumArgOperands() != 2)
    return false;

  const Value *Arg0 = I.getArgOperand(0), *Arg1 = I.getArgOperand(1);
  if (!Arg0->getType()->isPointerTy() ||
      !Arg1->getType()->isPointerTy() ||
      !I.getType()->isIntegerTy())
    return false;

  const TargetSelectionDAGInfo &TSI = DAG.getSelectionDAGInfo();
  std::pair<SDValue, SDValue> Res =
    TSI.EmitTargetCodeForStrcmp(DAG, getCurSDLoc(), DAG.getRoot(),
                                getValue(Arg0), getValue(Arg1),
                                MachinePointerInfo(Arg0),
                                MachinePointerInfo(Arg1));
  if (Res.first.getNode()) {
    processIntegerCallValue(I, Res.first, true);
    PendingLoads.push_back(Res.second);
    return true;
  }

  return false;
}

/// visitStrLenCall -- See if we can lower a strlen call into an optimized
/// form.  If so, return true and lower it, otherwise return false and it
/// will be lowered like a normal call.
bool SelectionDAGBuilder::visitStrLenCall(const CallInst &I) {
  // Verify that the prototype makes sense.  size_t strlen(char *)
  if (I.getNumArgOperands() != 1)
    return false;

  const Value *Arg0 = I.getArgOperand(0);
  if (!Arg0->getType()->isPointerTy() || !I.getType()->isIntegerTy())
    return false;

  const TargetSelectionDAGInfo &TSI = DAG.getSelectionDAGInfo();
  std::pair<SDValue, SDValue> Res =
    TSI.EmitTargetCodeForStrlen(DAG, getCurSDLoc(), DAG.getRoot(),
                                getValue(Arg0), MachinePointerInfo(Arg0));
  if (Res.first.getNode()) {
    processIntegerCallValue(I, Res.first, false);
    PendingLoads.push_back(Res.second);
    return true;
  }

  return false;
}

/// visitStrNLenCall -- See if we can lower a strnlen call into an optimized
/// form.  If so, return true and lower it, otherwise return false and it
/// will be lowered like a normal call.
bool SelectionDAGBuilder::visitStrNLenCall(const CallInst &I) {
  // Verify that the prototype makes sense.  size_t strnlen(char *, size_t)
  if (I.getNumArgOperands() != 2)
    return false;

  const Value *Arg0 = I.getArgOperand(0), *Arg1 = I.getArgOperand(1);
  if (!Arg0->getType()->isPointerTy() ||
      !Arg1->getType()->isIntegerTy() ||
      !I.getType()->isIntegerTy())
    return false;

  const TargetSelectionDAGInfo &TSI = DAG.getSelectionDAGInfo();
  std::pair<SDValue, SDValue> Res =
    TSI.EmitTargetCodeForStrnlen(DAG, getCurSDLoc(), DAG.getRoot(),
                                 getValue(Arg0), getValue(Arg1),
                                 MachinePointerInfo(Arg0));
  if (Res.first.getNode()) {
    processIntegerCallValue(I, Res.first, false);
    PendingLoads.push_back(Res.second);
    return true;
  }

  return false;
}

/// visitUnaryFloatCall - If a call instruction is a unary floating-point
/// operation (as expected), translate it to an SDNode with the specified opcode
/// and return true.
bool SelectionDAGBuilder::visitUnaryFloatCall(const CallInst &I,
                                              unsigned Opcode) {
  // Sanity check that it really is a unary floating-point call.
  if (I.getNumArgOperands() != 1 ||
      !I.getArgOperand(0)->getType()->isFloatingPointTy() ||
      I.getType() != I.getArgOperand(0)->getType() ||
      !I.onlyReadsMemory())
    return false;

  SDValue Tmp = getValue(I.getArgOperand(0));
  setValue(&I, DAG.getNode(Opcode, getCurSDLoc(), Tmp.getValueType(), Tmp));
  return true;
}

/// visitBinaryFloatCall - If a call instruction is a binary floating-point
/// operation (as expected), translate it to an SDNode with the specified opcode
/// and return true.
bool SelectionDAGBuilder::visitBinaryFloatCall(const CallInst &I,
                                               unsigned Opcode) {
  // Sanity check that it really is a binary floating-point call.
  if (I.getNumArgOperands() != 2 ||
      !I.getArgOperand(0)->getType()->isFloatingPointTy() ||
      I.getType() != I.getArgOperand(0)->getType() ||
      I.getType() != I.getArgOperand(1)->getType() ||
      !I.onlyReadsMemory())
    return false;

  SDValue Tmp0 = getValue(I.getArgOperand(0));
  SDValue Tmp1 = getValue(I.getArgOperand(1));
  EVT VT = Tmp0.getValueType();
  setValue(&I, DAG.getNode(Opcode, getCurSDLoc(), VT, Tmp0, Tmp1));
  return true;
}

void SelectionDAGBuilder::visitCall(const CallInst &I) {
  // Handle inline assembly differently.
  if (isa<InlineAsm>(I.getCalledValue())) {
    visitInlineAsm(&I);
    return;
  }

  MachineModuleInfo &MMI = DAG.getMachineFunction().getMMI();
  ComputeUsesVAFloatArgument(I, &MMI);

  const char *RenameFn = nullptr;
  if (Function *F = I.getCalledFunction()) {
    if (F->isDeclaration()) {
      if (const TargetIntrinsicInfo *II = TM.getIntrinsicInfo()) {
        if (unsigned IID = II->getIntrinsicID(F)) {
          RenameFn = visitIntrinsicCall(I, IID);
          if (!RenameFn)
            return;
        }
      }
      if (Intrinsic::ID IID = F->getIntrinsicID()) {
        RenameFn = visitIntrinsicCall(I, IID);
        if (!RenameFn)
          return;
      }
    }

    // Check for well-known libc/libm calls.  If the function is internal, it
    // can't be a library call.
    LibFunc::Func Func;
    if (!F->hasLocalLinkage() && F->hasName() &&
        LibInfo->getLibFunc(F->getName(), Func) &&
        LibInfo->hasOptimizedCodeGen(Func)) {
      switch (Func) {
      default: break;
      case LibFunc::copysign:
      case LibFunc::copysignf:
      case LibFunc::copysignl:
        if (I.getNumArgOperands() == 2 &&   // Basic sanity checks.
            I.getArgOperand(0)->getType()->isFloatingPointTy() &&
            I.getType() == I.getArgOperand(0)->getType() &&
            I.getType() == I.getArgOperand(1)->getType() &&
            I.onlyReadsMemory()) {
          SDValue LHS = getValue(I.getArgOperand(0));
          SDValue RHS = getValue(I.getArgOperand(1));
          setValue(&I, DAG.getNode(ISD::FCOPYSIGN, getCurSDLoc(),
                                   LHS.getValueType(), LHS, RHS));
          return;
        }
        break;
      case LibFunc::fabs:
      case LibFunc::fabsf:
      case LibFunc::fabsl:
        if (visitUnaryFloatCall(I, ISD::FABS))
          return;
        break;
      case LibFunc::fmin:
      case LibFunc::fminf:
      case LibFunc::fminl:
        if (visitBinaryFloatCall(I, ISD::FMINNUM))
          return;
        break;
      case LibFunc::fmax:
      case LibFunc::fmaxf:
      case LibFunc::fmaxl:
        if (visitBinaryFloatCall(I, ISD::FMAXNUM))
          return;
        break;
      case LibFunc::sin:
      case LibFunc::sinf:
      case LibFunc::sinl:
        if (visitUnaryFloatCall(I, ISD::FSIN))
          return;
        break;
      case LibFunc::cos:
      case LibFunc::cosf:
      case LibFunc::cosl:
        if (visitUnaryFloatCall(I, ISD::FCOS))
          return;
        break;
      case LibFunc::sqrt:
      case LibFunc::sqrtf:
      case LibFunc::sqrtl:
      case LibFunc::sqrt_finite:
      case LibFunc::sqrtf_finite:
      case LibFunc::sqrtl_finite:
        if (visitUnaryFloatCall(I, ISD::FSQRT))
          return;
        break;
      case LibFunc::floor:
      case LibFunc::floorf:
      case LibFunc::floorl:
        if (visitUnaryFloatCall(I, ISD::FFLOOR))
          return;
        break;
      case LibFunc::nearbyint:
      case LibFunc::nearbyintf:
      case LibFunc::nearbyintl:
        if (visitUnaryFloatCall(I, ISD::FNEARBYINT))
          return;
        break;
      case LibFunc::ceil:
      case LibFunc::ceilf:
      case LibFunc::ceill:
        if (visitUnaryFloatCall(I, ISD::FCEIL))
          return;
        break;
      case LibFunc::rint:
      case LibFunc::rintf:
      case LibFunc::rintl:
        if (visitUnaryFloatCall(I, ISD::FRINT))
          return;
        break;
      case LibFunc::round:
      case LibFunc::roundf:
      case LibFunc::roundl:
        if (visitUnaryFloatCall(I, ISD::FROUND))
          return;
        break;
      case LibFunc::trunc:
      case LibFunc::truncf:
      case LibFunc::truncl:
        if (visitUnaryFloatCall(I, ISD::FTRUNC))
          return;
        break;
      case LibFunc::log2:
      case LibFunc::log2f:
      case LibFunc::log2l:
        if (visitUnaryFloatCall(I, ISD::FLOG2))
          return;
        break;
      case LibFunc::exp2:
      case LibFunc::exp2f:
      case LibFunc::exp2l:
        if (visitUnaryFloatCall(I, ISD::FEXP2))
          return;
        break;
      case LibFunc::memcmp:
        if (visitMemCmpCall(I))
          return;
        break;
      case LibFunc::memchr:
        if (visitMemChrCall(I))
          return;
        break;
      case LibFunc::strcpy:
        if (visitStrCpyCall(I, false))
          return;
        break;
      case LibFunc::stpcpy:
        if (visitStrCpyCall(I, true))
          return;
        break;
      case LibFunc::strcmp:
        if (visitStrCmpCall(I))
          return;
        break;
      case LibFunc::strlen:
        if (visitStrLenCall(I))
          return;
        break;
      case LibFunc::strnlen:
        if (visitStrNLenCall(I))
          return;
        break;
      }
    }
  }

  SDValue Callee;
  if (!RenameFn)
    Callee = getValue(I.getCalledValue());
  else
    Callee = DAG.getExternalSymbol(
        RenameFn,
        DAG.getTargetLoweringInfo().getPointerTy(DAG.getDataLayout()));

  // Check if we can potentially perform a tail call. More detailed checking is
  // be done within LowerCallTo, after more information about the call is known.
  LowerCallTo(&I, Callee, I.isTailCall());
}

namespace {

/// AsmOperandInfo - This contains information for each constraint that we are
/// lowering.
class SDISelAsmOperandInfo : public TargetLowering::AsmOperandInfo {
public:
  /// CallOperand - If this is the result output operand or a clobber
  /// this is null, otherwise it is the incoming operand to the CallInst.
  /// This gets modified as the asm is processed.
  SDValue CallOperand;

  /// AssignedRegs - If this is a register or register class operand, this
  /// contains the set of register corresponding to the operand.
  RegsForValue AssignedRegs;

  explicit SDISelAsmOperandInfo(const TargetLowering::AsmOperandInfo &info)
    : TargetLowering::AsmOperandInfo(info), CallOperand(nullptr,0) {
  }

  /// getCallOperandValEVT - Return the EVT of the Value* that this operand
  /// corresponds to.  If there is no Value* for this operand, it returns
  /// MVT::Other.
  EVT getCallOperandValEVT(LLVMContext &Context, const TargetLowering &TLI,
                           const DataLayout &DL) const {
    if (!CallOperandVal) return MVT::Other;

    if (isa<BasicBlock>(CallOperandVal))
      return TLI.getPointerTy(DL);

    llvm::Type *OpTy = CallOperandVal->getType();

    // FIXME: code duplicated from TargetLowering::ParseConstraints().
    // If this is an indirect operand, the operand is a pointer to the
    // accessed type.
    if (isIndirect) {
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
        OpTy = IntegerType::get(Context, BitSize);
        break;
      }
    }

    return TLI.getValueType(DL, OpTy, true);
  }
};

typedef SmallVector<SDISelAsmOperandInfo,16> SDISelAsmOperandInfoVector;

} // end anonymous namespace

/// GetRegistersForValue - Assign registers (virtual or physical) for the
/// specified operand.  We prefer to assign virtual registers, to allow the
/// register allocator to handle the assignment process.  However, if the asm
/// uses features that we can't model on machineinstrs, we have SDISel do the
/// allocation.  This produces generally horrible, but correct, code.
///
///   OpInfo describes the operand.
///
static void GetRegistersForValue(SelectionDAG &DAG,
                                 const TargetLowering &TLI,
                                 SDLoc DL,
                                 SDISelAsmOperandInfo &OpInfo) {
  LLVMContext &Context = *DAG.getContext();

  MachineFunction &MF = DAG.getMachineFunction();
  SmallVector<unsigned, 4> Regs;

  // If this is a constraint for a single physreg, or a constraint for a
  // register class, find it.
  std::pair<unsigned, const TargetRegisterClass *> PhysReg =
      TLI.getRegForInlineAsmConstraint(MF.getSubtarget().getRegisterInfo(),
                                       OpInfo.ConstraintCode,
                                       OpInfo.ConstraintVT);

  unsigned NumRegs = 1;
  if (OpInfo.ConstraintVT != MVT::Other) {
    // If this is a FP input in an integer register (or visa versa) insert a bit
    // cast of the input value.  More generally, handle any case where the input
    // value disagrees with the register class we plan to stick this in.
    if (OpInfo.Type == InlineAsm::isInput &&
        PhysReg.second && !PhysReg.second->hasType(OpInfo.ConstraintVT)) {
      // Try to convert to the first EVT that the reg class contains.  If the
      // types are identical size, use a bitcast to convert (e.g. two differing
      // vector types).
      MVT RegVT = *PhysReg.second->vt_begin();
      if (RegVT.getSizeInBits() == OpInfo.CallOperand.getValueSizeInBits()) {
        OpInfo.CallOperand = DAG.getNode(ISD::BITCAST, DL,
                                         RegVT, OpInfo.CallOperand);
        OpInfo.ConstraintVT = RegVT;
      } else if (RegVT.isInteger() && OpInfo.ConstraintVT.isFloatingPoint()) {
        // If the input is a FP value and we want it in FP registers, do a
        // bitcast to the corresponding integer type.  This turns an f64 value
        // into i64, which can be passed with two i32 values on a 32-bit
        // machine.
        RegVT = MVT::getIntegerVT(OpInfo.ConstraintVT.getSizeInBits());
        OpInfo.CallOperand = DAG.getNode(ISD::BITCAST, DL,
                                         RegVT, OpInfo.CallOperand);
        OpInfo.ConstraintVT = RegVT;
      }
    }

    NumRegs = TLI.getNumRegisters(Context, OpInfo.ConstraintVT);
  }

  MVT RegVT;
  EVT ValueVT = OpInfo.ConstraintVT;

  // If this is a constraint for a specific physical register, like {r17},
  // assign it now.
  if (unsigned AssignedReg = PhysReg.first) {
    const TargetRegisterClass *RC = PhysReg.second;
    if (OpInfo.ConstraintVT == MVT::Other)
      ValueVT = *RC->vt_begin();

    // Get the actual register value type.  This is important, because the user
    // may have asked for (e.g.) the AX register in i32 type.  We need to
    // remember that AX is actually i16 to get the right extension.
    RegVT = *RC->vt_begin();

    // This is a explicit reference to a physical register.
    Regs.push_back(AssignedReg);

    // If this is an expanded reference, add the rest of the regs to Regs.
    if (NumRegs != 1) {
      TargetRegisterClass::iterator I = RC->begin();
      for (; *I != AssignedReg; ++I)
        assert(I != RC->end() && "Didn't find reg!");

      // Already added the first reg.
      --NumRegs; ++I;
      for (; NumRegs; --NumRegs, ++I) {
        assert(I != RC->end() && "Ran out of registers to allocate!");
        Regs.push_back(*I);
      }
    }

    OpInfo.AssignedRegs = RegsForValue(Regs, RegVT, ValueVT);
    return;
  }

  // Otherwise, if this was a reference to an LLVM register class, create vregs
  // for this reference.
  if (const TargetRegisterClass *RC = PhysReg.second) {
    RegVT = *RC->vt_begin();
    if (OpInfo.ConstraintVT == MVT::Other)
      ValueVT = RegVT;

    // Create the appropriate number of virtual registers.
    MachineRegisterInfo &RegInfo = MF.getRegInfo();
    for (; NumRegs; --NumRegs)
      Regs.push_back(RegInfo.createVirtualRegister(RC));

    OpInfo.AssignedRegs = RegsForValue(Regs, RegVT, ValueVT);
    return;
  }

  // Otherwise, we couldn't allocate enough registers for this.
}

/// visitInlineAsm - Handle a call to an InlineAsm object.
///
void SelectionDAGBuilder::visitInlineAsm(ImmutableCallSite CS) {
  const InlineAsm *IA = cast<InlineAsm>(CS.getCalledValue());

  /// ConstraintOperands - Information about all of the constraints.
  SDISelAsmOperandInfoVector ConstraintOperands;

  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  TargetLowering::AsmOperandInfoVector TargetConstraints = TLI.ParseConstraints(
      DAG.getDataLayout(), DAG.getSubtarget().getRegisterInfo(), CS);

  bool hasMemory = false;

  unsigned ArgNo = 0;   // ArgNo - The argument of the CallInst.
  unsigned ResNo = 0;   // ResNo - The result number of the next output.
  for (unsigned i = 0, e = TargetConstraints.size(); i != e; ++i) {
    ConstraintOperands.push_back(SDISelAsmOperandInfo(TargetConstraints[i]));
    SDISelAsmOperandInfo &OpInfo = ConstraintOperands.back();

    MVT OpVT = MVT::Other;

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
      assert(!CS.getType()->isVoidTy() && "Bad inline asm!");
      if (StructType *STy = dyn_cast<StructType>(CS.getType())) {
        OpVT = TLI.getSimpleValueType(DAG.getDataLayout(),
                                      STy->getElementType(ResNo));
      } else {
        assert(ResNo == 0 && "Asm only has one result!");
        OpVT = TLI.getSimpleValueType(DAG.getDataLayout(), CS.getType());
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

    // If this is an input or an indirect output, process the call argument.
    // BasicBlocks are labels, currently appearing only in asm's.
    if (OpInfo.CallOperandVal) {
      if (const BasicBlock *BB = dyn_cast<BasicBlock>(OpInfo.CallOperandVal)) {
        OpInfo.CallOperand = DAG.getBasicBlock(FuncInfo.MBBMap[BB]);
      } else {
        OpInfo.CallOperand = getValue(OpInfo.CallOperandVal);
      }

      OpVT = OpInfo.getCallOperandValEVT(*DAG.getContext(), TLI,
                                         DAG.getDataLayout()).getSimpleVT();
    }

    OpInfo.ConstraintVT = OpVT;

    // Indirect operand accesses access memory.
    if (OpInfo.isIndirect)
      hasMemory = true;
    else {
      for (unsigned j = 0, ee = OpInfo.Codes.size(); j != ee; ++j) {
        TargetLowering::ConstraintType
          CType = TLI.getConstraintType(OpInfo.Codes[j]);
        if (CType == TargetLowering::C_Memory) {
          hasMemory = true;
          break;
        }
      }
    }
  }

  SDValue Chain, Flag;

  // We won't need to flush pending loads if this asm doesn't touch
  // memory and is nonvolatile.
  if (hasMemory || IA->hasSideEffects())
    Chain = getRoot();
  else
    Chain = DAG.getRoot();

  // Second pass over the constraints: compute which constraint option to use
  // and assign registers to constraints that want a specific physreg.
  for (unsigned i = 0, e = ConstraintOperands.size(); i != e; ++i) {
    SDISelAsmOperandInfo &OpInfo = ConstraintOperands[i];

    // If this is an output operand with a matching input operand, look up the
    // matching input. If their types mismatch, e.g. one is an integer, the
    // other is floating point, or their sizes are different, flag it as an
    // error.
    if (OpInfo.hasMatchingInput()) {
      SDISelAsmOperandInfo &Input = ConstraintOperands[OpInfo.MatchingInput];

      if (OpInfo.ConstraintVT != Input.ConstraintVT) {
	const TargetRegisterInfo *TRI = DAG.getSubtarget().getRegisterInfo();
        std::pair<unsigned, const TargetRegisterClass *> MatchRC =
            TLI.getRegForInlineAsmConstraint(TRI, OpInfo.ConstraintCode,
                                             OpInfo.ConstraintVT);
        std::pair<unsigned, const TargetRegisterClass *> InputRC =
            TLI.getRegForInlineAsmConstraint(TRI, Input.ConstraintCode,
                                             Input.ConstraintVT);
        if ((OpInfo.ConstraintVT.isInteger() !=
             Input.ConstraintVT.isInteger()) ||
            (MatchRC.second != InputRC.second)) {
          report_fatal_error("Unsupported asm: input constraint"
                             " with a matching output constraint of"
                             " incompatible type!");
        }
        Input.ConstraintVT = OpInfo.ConstraintVT;
      }
    }

    // Compute the constraint code and ConstraintType to use.
    TLI.ComputeConstraintToUse(OpInfo, OpInfo.CallOperand, &DAG);

    if (OpInfo.ConstraintType == TargetLowering::C_Memory &&
        OpInfo.Type == InlineAsm::isClobber)
      continue;

    // If this is a memory input, and if the operand is not indirect, do what we
    // need to to provide an address for the memory input.
    if (OpInfo.ConstraintType == TargetLowering::C_Memory &&
        !OpInfo.isIndirect) {
      assert((OpInfo.isMultipleAlternative ||
              (OpInfo.Type == InlineAsm::isInput)) &&
             "Can only indirectify direct input operands!");

      // Memory operands really want the address of the value.  If we don't have
      // an indirect input, put it in the constpool if we can, otherwise spill
      // it to a stack slot.
      // TODO: This isn't quite right. We need to handle these according to
      // the addressing mode that the constraint wants. Also, this may take
      // an additional register for the computation and we don't want that
      // either.

      // If the operand is a float, integer, or vector constant, spill to a
      // constant pool entry to get its address.
      const Value *OpVal = OpInfo.CallOperandVal;
      if (isa<ConstantFP>(OpVal) || isa<ConstantInt>(OpVal) ||
          isa<ConstantVector>(OpVal) || isa<ConstantDataVector>(OpVal)) {
        OpInfo.CallOperand = DAG.getConstantPool(
            cast<Constant>(OpVal), TLI.getPointerTy(DAG.getDataLayout()));
      } else {
        // Otherwise, create a stack slot and emit a store to it before the
        // asm.
        Type *Ty = OpVal->getType();
        auto &DL = DAG.getDataLayout();
        uint64_t TySize = DL.getTypeAllocSize(Ty);
        unsigned Align = DL.getPrefTypeAlignment(Ty);
        MachineFunction &MF = DAG.getMachineFunction();
        int SSFI = MF.getFrameInfo()->CreateStackObject(TySize, Align, false);
        SDValue StackSlot =
            DAG.getFrameIndex(SSFI, TLI.getPointerTy(DAG.getDataLayout()));
        Chain = DAG.getStore(Chain, getCurSDLoc(),
                             OpInfo.CallOperand, StackSlot,
                             MachinePointerInfo::getFixedStack(SSFI),
                             false, false, 0);
        OpInfo.CallOperand = StackSlot;
      }

      // There is no longer a Value* corresponding to this operand.
      OpInfo.CallOperandVal = nullptr;

      // It is now an indirect operand.
      OpInfo.isIndirect = true;
    }

    // If this constraint is for a specific register, allocate it before
    // anything else.
    if (OpInfo.ConstraintType == TargetLowering::C_Register)
      GetRegistersForValue(DAG, TLI, getCurSDLoc(), OpInfo);
  }

  // Second pass - Loop over all of the operands, assigning virtual or physregs
  // to register class operands.
  for (unsigned i = 0, e = ConstraintOperands.size(); i != e; ++i) {
    SDISelAsmOperandInfo &OpInfo = ConstraintOperands[i];

    // C_Register operands have already been allocated, Other/Memory don't need
    // to be.
    if (OpInfo.ConstraintType == TargetLowering::C_RegisterClass)
      GetRegistersForValue(DAG, TLI, getCurSDLoc(), OpInfo);
  }

  // AsmNodeOperands - The operands for the ISD::INLINEASM node.
  std::vector<SDValue> AsmNodeOperands;
  AsmNodeOperands.push_back(SDValue());  // reserve space for input chain
  AsmNodeOperands.push_back(DAG.getTargetExternalSymbol(
      IA->getAsmString().c_str(), TLI.getPointerTy(DAG.getDataLayout())));

  // If we have a !srcloc metadata node associated with it, we want to attach
  // this to the ultimately generated inline asm machineinstr.  To do this, we
  // pass in the third operand as this (potentially null) inline asm MDNode.
  const MDNode *SrcLoc = CS.getInstruction()->getMetadata("srcloc");
  AsmNodeOperands.push_back(DAG.getMDNode(SrcLoc));

  // Remember the HasSideEffect, AlignStack, AsmDialect, MayLoad and MayStore
  // bits as operand 3.
  unsigned ExtraInfo = 0;
  if (IA->hasSideEffects())
    ExtraInfo |= InlineAsm::Extra_HasSideEffects;
  if (IA->isAlignStack())
    ExtraInfo |= InlineAsm::Extra_IsAlignStack;
  // Set the asm dialect.
  ExtraInfo |= IA->getDialect() * InlineAsm::Extra_AsmDialect;

  // Determine if this InlineAsm MayLoad or MayStore based on the constraints.
  for (unsigned i = 0, e = TargetConstraints.size(); i != e; ++i) {
    TargetLowering::AsmOperandInfo &OpInfo = TargetConstraints[i];

    // Compute the constraint code and ConstraintType to use.
    TLI.ComputeConstraintToUse(OpInfo, SDValue());

    // Ideally, we would only check against memory constraints.  However, the
    // meaning of an other constraint can be target-specific and we can't easily
    // reason about it.  Therefore, be conservative and set MayLoad/MayStore
    // for other constriants as well.
    if (OpInfo.ConstraintType == TargetLowering::C_Memory ||
        OpInfo.ConstraintType == TargetLowering::C_Other) {
      if (OpInfo.Type == InlineAsm::isInput)
        ExtraInfo |= InlineAsm::Extra_MayLoad;
      else if (OpInfo.Type == InlineAsm::isOutput)
        ExtraInfo |= InlineAsm::Extra_MayStore;
      else if (OpInfo.Type == InlineAsm::isClobber)
        ExtraInfo |= (InlineAsm::Extra_MayLoad | InlineAsm::Extra_MayStore);
    }
  }

  AsmNodeOperands.push_back(DAG.getTargetConstant(
      ExtraInfo, getCurSDLoc(), TLI.getPointerTy(DAG.getDataLayout())));

  // Loop over all of the inputs, copying the operand values into the
  // appropriate registers and processing the output regs.
  RegsForValue RetValRegs;

  // IndirectStoresToEmit - The set of stores to emit after the inline asm node.
  std::vector<std::pair<RegsForValue, Value*> > IndirectStoresToEmit;

  for (unsigned i = 0, e = ConstraintOperands.size(); i != e; ++i) {
    SDISelAsmOperandInfo &OpInfo = ConstraintOperands[i];

    switch (OpInfo.Type) {
    case InlineAsm::isOutput: {
      if (OpInfo.ConstraintType != TargetLowering::C_RegisterClass &&
          OpInfo.ConstraintType != TargetLowering::C_Register) {
        // Memory output, or 'other' output (e.g. 'X' constraint).
        assert(OpInfo.isIndirect && "Memory output must be indirect operand");

        unsigned ConstraintID =
            TLI.getInlineAsmMemConstraint(OpInfo.ConstraintCode);
        assert(ConstraintID != InlineAsm::Constraint_Unknown &&
               "Failed to convert memory constraint code to constraint id.");

        // Add information to the INLINEASM node to know about this output.
        unsigned OpFlags = InlineAsm::getFlagWord(InlineAsm::Kind_Mem, 1);
        OpFlags = InlineAsm::getFlagWordForMem(OpFlags, ConstraintID);
        AsmNodeOperands.push_back(DAG.getTargetConstant(OpFlags, getCurSDLoc(),
                                                        MVT::i32));
        AsmNodeOperands.push_back(OpInfo.CallOperand);
        break;
      }

      // Otherwise, this is a register or register class output.

      // Copy the output from the appropriate register.  Find a register that
      // we can use.
      if (OpInfo.AssignedRegs.Regs.empty()) {
        LLVMContext &Ctx = *DAG.getContext();
        Ctx.emitError(CS.getInstruction(),
                      "couldn't allocate output register for constraint '" +
                          Twine(OpInfo.ConstraintCode) + "'");
        return;
      }

      // If this is an indirect operand, store through the pointer after the
      // asm.
      if (OpInfo.isIndirect) {
        IndirectStoresToEmit.push_back(std::make_pair(OpInfo.AssignedRegs,
                                                      OpInfo.CallOperandVal));
      } else {
        // This is the result value of the call.
        assert(!CS.getType()->isVoidTy() && "Bad inline asm!");
        // Concatenate this output onto the outputs list.
        RetValRegs.append(OpInfo.AssignedRegs);
      }

      // Add information to the INLINEASM node to know that this register is
      // set.
      OpInfo.AssignedRegs
          .AddInlineAsmOperands(OpInfo.isEarlyClobber
                                    ? InlineAsm::Kind_RegDefEarlyClobber
                                    : InlineAsm::Kind_RegDef,
                                false, 0, getCurSDLoc(), DAG, AsmNodeOperands);
      break;
    }
    case InlineAsm::isInput: {
      SDValue InOperandVal = OpInfo.CallOperand;

      if (OpInfo.isMatchingInputConstraint()) {   // Matching constraint?
        // If this is required to match an output register we have already set,
        // just use its register.
        unsigned OperandNo = OpInfo.getMatchedOperand();

        // Scan until we find the definition we already emitted of this operand.
        // When we find it, create a RegsForValue operand.
        unsigned CurOp = InlineAsm::Op_FirstOperand;
        for (; OperandNo; --OperandNo) {
          // Advance to the next operand.
          unsigned OpFlag =
            cast<ConstantSDNode>(AsmNodeOperands[CurOp])->getZExtValue();
          assert((InlineAsm::isRegDefKind(OpFlag) ||
                  InlineAsm::isRegDefEarlyClobberKind(OpFlag) ||
                  InlineAsm::isMemKind(OpFlag)) && "Skipped past definitions?");
          CurOp += InlineAsm::getNumOperandRegisters(OpFlag)+1;
        }

        unsigned OpFlag =
          cast<ConstantSDNode>(AsmNodeOperands[CurOp])->getZExtValue();
        if (InlineAsm::isRegDefKind(OpFlag) ||
            InlineAsm::isRegDefEarlyClobberKind(OpFlag)) {
          // Add (OpFlag&0xffff)>>3 registers to MatchedRegs.
          if (OpInfo.isIndirect) {
            // This happens on gcc/testsuite/gcc.dg/pr8788-1.c
            LLVMContext &Ctx = *DAG.getContext();
            Ctx.emitError(CS.getInstruction(), "inline asm not supported yet:"
                                               " don't know how to handle tied "
                                               "indirect register inputs");
            return;
          }

          RegsForValue MatchedRegs;
          MatchedRegs.ValueVTs.push_back(InOperandVal.getValueType());
          MVT RegVT = AsmNodeOperands[CurOp+1].getSimpleValueType();
          MatchedRegs.RegVTs.push_back(RegVT);
          MachineRegisterInfo &RegInfo = DAG.getMachineFunction().getRegInfo();
          for (unsigned i = 0, e = InlineAsm::getNumOperandRegisters(OpFlag);
               i != e; ++i) {
            if (const TargetRegisterClass *RC = TLI.getRegClassFor(RegVT))
              MatchedRegs.Regs.push_back(RegInfo.createVirtualRegister(RC));
            else {
              LLVMContext &Ctx = *DAG.getContext();
              Ctx.emitError(CS.getInstruction(),
                            "inline asm error: This value"
                            " type register class is not natively supported!");
              return;
            }
          }
          SDLoc dl = getCurSDLoc();
          // Use the produced MatchedRegs object to
          MatchedRegs.getCopyToRegs(InOperandVal, DAG, dl,
                                    Chain, &Flag, CS.getInstruction());
          MatchedRegs.AddInlineAsmOperands(InlineAsm::Kind_RegUse,
                                           true, OpInfo.getMatchedOperand(), dl,
                                           DAG, AsmNodeOperands);
          break;
        }

        assert(InlineAsm::isMemKind(OpFlag) && "Unknown matching constraint!");
        assert(InlineAsm::getNumOperandRegisters(OpFlag) == 1 &&
               "Unexpected number of operands");
        // Add information to the INLINEASM node to know about this input.
        // See InlineAsm.h isUseOperandTiedToDef.
        OpFlag = InlineAsm::convertMemFlagWordToMatchingFlagWord(OpFlag);
        OpFlag = InlineAsm::getFlagWordForMatchingOp(OpFlag,
                                                    OpInfo.getMatchedOperand());
        AsmNodeOperands.push_back(DAG.getTargetConstant(
            OpFlag, getCurSDLoc(), TLI.getPointerTy(DAG.getDataLayout())));
        AsmNodeOperands.push_back(AsmNodeOperands[CurOp+1]);
        break;
      }

      // Treat indirect 'X' constraint as memory.
      if (OpInfo.ConstraintType == TargetLowering::C_Other &&
          OpInfo.isIndirect)
        OpInfo.ConstraintType = TargetLowering::C_Memory;

      if (OpInfo.ConstraintType == TargetLowering::C_Other) {
        std::vector<SDValue> Ops;
        TLI.LowerAsmOperandForConstraint(InOperandVal, OpInfo.ConstraintCode,
                                          Ops, DAG);
        if (Ops.empty()) {
          LLVMContext &Ctx = *DAG.getContext();
          Ctx.emitError(CS.getInstruction(),
                        "invalid operand for inline asm constraint '" +
                            Twine(OpInfo.ConstraintCode) + "'");
          return;
        }

        // Add information to the INLINEASM node to know about this input.
        unsigned ResOpType =
          InlineAsm::getFlagWord(InlineAsm::Kind_Imm, Ops.size());
        AsmNodeOperands.push_back(DAG.getTargetConstant(
            ResOpType, getCurSDLoc(), TLI.getPointerTy(DAG.getDataLayout())));
        AsmNodeOperands.insert(AsmNodeOperands.end(), Ops.begin(), Ops.end());
        break;
      }

      if (OpInfo.ConstraintType == TargetLowering::C_Memory) {
        assert(OpInfo.isIndirect && "Operand must be indirect to be a mem!");
        assert(InOperandVal.getValueType() ==
                   TLI.getPointerTy(DAG.getDataLayout()) &&
               "Memory operands expect pointer values");

        unsigned ConstraintID =
            TLI.getInlineAsmMemConstraint(OpInfo.ConstraintCode);
        assert(ConstraintID != InlineAsm::Constraint_Unknown &&
               "Failed to convert memory constraint code to constraint id.");

        // Add information to the INLINEASM node to know about this input.
        unsigned ResOpType = InlineAsm::getFlagWord(InlineAsm::Kind_Mem, 1);
        ResOpType = InlineAsm::getFlagWordForMem(ResOpType, ConstraintID);
        AsmNodeOperands.push_back(DAG.getTargetConstant(ResOpType,
                                                        getCurSDLoc(),
                                                        MVT::i32));
        AsmNodeOperands.push_back(InOperandVal);
        break;
      }

      assert((OpInfo.ConstraintType == TargetLowering::C_RegisterClass ||
              OpInfo.ConstraintType == TargetLowering::C_Register) &&
             "Unknown constraint type!");

      // TODO: Support this.
      if (OpInfo.isIndirect) {
        LLVMContext &Ctx = *DAG.getContext();
        Ctx.emitError(CS.getInstruction(),
                      "Don't know how to handle indirect register inputs yet "
                      "for constraint '" +
                          Twine(OpInfo.ConstraintCode) + "'");
        return;
      }

      // Copy the input into the appropriate registers.
      if (OpInfo.AssignedRegs.Regs.empty()) {
        LLVMContext &Ctx = *DAG.getContext();
        Ctx.emitError(CS.getInstruction(),
                      "couldn't allocate input reg for constraint '" +
                          Twine(OpInfo.ConstraintCode) + "'");
        return;
      }

      SDLoc dl = getCurSDLoc();

      OpInfo.AssignedRegs.getCopyToRegs(InOperandVal, DAG, dl,
                                        Chain, &Flag, CS.getInstruction());

      OpInfo.AssignedRegs.AddInlineAsmOperands(InlineAsm::Kind_RegUse, false, 0,
                                               dl, DAG, AsmNodeOperands);
      break;
    }
    case InlineAsm::isClobber: {
      // Add the clobbered value to the operand list, so that the register
      // allocator is aware that the physreg got clobbered.
      if (!OpInfo.AssignedRegs.Regs.empty())
        OpInfo.AssignedRegs.AddInlineAsmOperands(InlineAsm::Kind_Clobber,
                                                 false, 0, getCurSDLoc(), DAG,
                                                 AsmNodeOperands);
      break;
    }
    }
  }

  // Finish up input operands.  Set the input chain and add the flag last.
  AsmNodeOperands[InlineAsm::Op_InputChain] = Chain;
  if (Flag.getNode()) AsmNodeOperands.push_back(Flag);

  Chain = DAG.getNode(ISD::INLINEASM, getCurSDLoc(),
                      DAG.getVTList(MVT::Other, MVT::Glue), AsmNodeOperands);
  Flag = Chain.getValue(1);

  // If this asm returns a register value, copy the result from that register
  // and set it as the value of the call.
  if (!RetValRegs.Regs.empty()) {
    SDValue Val = RetValRegs.getCopyFromRegs(DAG, FuncInfo, getCurSDLoc(),
                                             Chain, &Flag, CS.getInstruction());

    // FIXME: Why don't we do this for inline asms with MRVs?
    if (CS.getType()->isSingleValueType() && CS.getType()->isSized()) {
      EVT ResultType = TLI.getValueType(DAG.getDataLayout(), CS.getType());

      // If any of the results of the inline asm is a vector, it may have the
      // wrong width/num elts.  This can happen for register classes that can
      // contain multiple different value types.  The preg or vreg allocated may
      // not have the same VT as was expected.  Convert it to the right type
      // with bit_convert.
      if (ResultType != Val.getValueType() && Val.getValueType().isVector()) {
        Val = DAG.getNode(ISD::BITCAST, getCurSDLoc(),
                          ResultType, Val);

      } else if (ResultType != Val.getValueType() &&
                 ResultType.isInteger() && Val.getValueType().isInteger()) {
        // If a result value was tied to an input value, the computed result may
        // have a wider width than the expected result.  Extract the relevant
        // portion.
        Val = DAG.getNode(ISD::TRUNCATE, getCurSDLoc(), ResultType, Val);
      }

      assert(ResultType == Val.getValueType() && "Asm result value mismatch!");
    }

    setValue(CS.getInstruction(), Val);
    // Don't need to use this as a chain in this case.
    if (!IA->hasSideEffects() && !hasMemory && IndirectStoresToEmit.empty())
      return;
  }

  std::vector<std::pair<SDValue, const Value *> > StoresToEmit;

  // Process indirect outputs, first output all of the flagged copies out of
  // physregs.
  for (unsigned i = 0, e = IndirectStoresToEmit.size(); i != e; ++i) {
    RegsForValue &OutRegs = IndirectStoresToEmit[i].first;
    const Value *Ptr = IndirectStoresToEmit[i].second;
    SDValue OutVal = OutRegs.getCopyFromRegs(DAG, FuncInfo, getCurSDLoc(),
                                             Chain, &Flag, IA);
    StoresToEmit.push_back(std::make_pair(OutVal, Ptr));
  }

  // Emit the non-flagged stores from the physregs.
  SmallVector<SDValue, 8> OutChains;
  for (unsigned i = 0, e = StoresToEmit.size(); i != e; ++i) {
    SDValue Val = DAG.getStore(Chain, getCurSDLoc(),
                               StoresToEmit[i].first,
                               getValue(StoresToEmit[i].second),
                               MachinePointerInfo(StoresToEmit[i].second),
                               false, false, 0);
    OutChains.push_back(Val);
  }

  if (!OutChains.empty())
    Chain = DAG.getNode(ISD::TokenFactor, getCurSDLoc(), MVT::Other, OutChains);

  DAG.setRoot(Chain);
}

void SelectionDAGBuilder::visitVAStart(const CallInst &I) {
  DAG.setRoot(DAG.getNode(ISD::VASTART, getCurSDLoc(),
                          MVT::Other, getRoot(),
                          getValue(I.getArgOperand(0)),
                          DAG.getSrcValue(I.getArgOperand(0))));
}

void SelectionDAGBuilder::visitVAArg(const VAArgInst &I) {
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  const DataLayout &DL = DAG.getDataLayout();
  SDValue V = DAG.getVAArg(TLI.getValueType(DAG.getDataLayout(), I.getType()),
                           getCurSDLoc(), getRoot(), getValue(I.getOperand(0)),
                           DAG.getSrcValue(I.getOperand(0)),
                           DL.getABITypeAlignment(I.getType()));
  setValue(&I, V);
  DAG.setRoot(V.getValue(1));
}

void SelectionDAGBuilder::visitVAEnd(const CallInst &I) {
  DAG.setRoot(DAG.getNode(ISD::VAEND, getCurSDLoc(),
                          MVT::Other, getRoot(),
                          getValue(I.getArgOperand(0)),
                          DAG.getSrcValue(I.getArgOperand(0))));
}

void SelectionDAGBuilder::visitVACopy(const CallInst &I) {
  DAG.setRoot(DAG.getNode(ISD::VACOPY, getCurSDLoc(),
                          MVT::Other, getRoot(),
                          getValue(I.getArgOperand(0)),
                          getValue(I.getArgOperand(1)),
                          DAG.getSrcValue(I.getArgOperand(0)),
                          DAG.getSrcValue(I.getArgOperand(1))));
}

/// \brief Lower an argument list according to the target calling convention.
///
/// \return A tuple of <return-value, token-chain>
///
/// This is a helper for lowering intrinsics that follow a target calling
/// convention or require stack pointer adjustment. Only a subset of the
/// intrinsic's operands need to participate in the calling convention.
std::pair<SDValue, SDValue>
SelectionDAGBuilder::lowerCallOperands(ImmutableCallSite CS, unsigned ArgIdx,
                                       unsigned NumArgs, SDValue Callee,
                                       Type *ReturnTy,
                                       MachineBasicBlock *LandingPad,
                                       bool IsPatchPoint) {
  TargetLowering::ArgListTy Args;
  Args.reserve(NumArgs);

  // Populate the argument list.
  // Attributes for args start at offset 1, after the return attribute.
  for (unsigned ArgI = ArgIdx, ArgE = ArgIdx + NumArgs, AttrI = ArgIdx + 1;
       ArgI != ArgE; ++ArgI) {
    const Value *V = CS->getOperand(ArgI);

    assert(!V->getType()->isEmptyTy() && "Empty type passed to intrinsic.");

    TargetLowering::ArgListEntry Entry;
    Entry.Node = getValue(V);
    Entry.Ty = V->getType();
    Entry.setAttributes(&CS, AttrI);
    Args.push_back(Entry);
  }

  TargetLowering::CallLoweringInfo CLI(DAG);
  CLI.setDebugLoc(getCurSDLoc()).setChain(getRoot())
    .setCallee(CS.getCallingConv(), ReturnTy, Callee, std::move(Args), NumArgs)
    .setDiscardResult(CS->use_empty()).setIsPatchPoint(IsPatchPoint);

  return lowerInvokable(CLI, LandingPad);
}

/// \brief Add a stack map intrinsic call's live variable operands to a stackmap
/// or patchpoint target node's operand list.
///
/// Constants are converted to TargetConstants purely as an optimization to
/// avoid constant materialization and register allocation.
///
/// FrameIndex operands are converted to TargetFrameIndex so that ISEL does not
/// generate addess computation nodes, and so ExpandISelPseudo can convert the
/// TargetFrameIndex into a DirectMemRefOp StackMap location. This avoids
/// address materialization and register allocation, but may also be required
/// for correctness. If a StackMap (or PatchPoint) intrinsic directly uses an
/// alloca in the entry block, then the runtime may assume that the alloca's
/// StackMap location can be read immediately after compilation and that the
/// location is valid at any point during execution (this is similar to the
/// assumption made by the llvm.gcroot intrinsic). If the alloca's location were
/// only available in a register, then the runtime would need to trap when
/// execution reaches the StackMap in order to read the alloca's location.
static void addStackMapLiveVars(ImmutableCallSite CS, unsigned StartIdx,
                                SDLoc DL, SmallVectorImpl<SDValue> &Ops,
                                SelectionDAGBuilder &Builder) {
  for (unsigned i = StartIdx, e = CS.arg_size(); i != e; ++i) {
    SDValue OpVal = Builder.getValue(CS.getArgument(i));
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(OpVal)) {
      Ops.push_back(
        Builder.DAG.getTargetConstant(StackMaps::ConstantOp, DL, MVT::i64));
      Ops.push_back(
        Builder.DAG.getTargetConstant(C->getSExtValue(), DL, MVT::i64));
    } else if (FrameIndexSDNode *FI = dyn_cast<FrameIndexSDNode>(OpVal)) {
      const TargetLowering &TLI = Builder.DAG.getTargetLoweringInfo();
      Ops.push_back(Builder.DAG.getTargetFrameIndex(
          FI->getIndex(), TLI.getPointerTy(Builder.DAG.getDataLayout())));
    } else
      Ops.push_back(OpVal);
  }
}

/// \brief Lower llvm.experimental.stackmap directly to its target opcode.
void SelectionDAGBuilder::visitStackmap(const CallInst &CI) {
  // void @llvm.experimental.stackmap(i32 <id>, i32 <numShadowBytes>,
  //                                  [live variables...])

  assert(CI.getType()->isVoidTy() && "Stackmap cannot return a value.");

  SDValue Chain, InFlag, Callee, NullPtr;
  SmallVector<SDValue, 32> Ops;

  SDLoc DL = getCurSDLoc();
  Callee = getValue(CI.getCalledValue());
  NullPtr = DAG.getIntPtrConstant(0, DL, true);

  // The stackmap intrinsic only records the live variables (the arguemnts
  // passed to it) and emits NOPS (if requested). Unlike the patchpoint
  // intrinsic, this won't be lowered to a function call. This means we don't
  // have to worry about calling conventions and target specific lowering code.
  // Instead we perform the call lowering right here.
  //
  // chain, flag = CALLSEQ_START(chain, 0)
  // chain, flag = STACKMAP(id, nbytes, ..., chain, flag)
  // chain, flag = CALLSEQ_END(chain, 0, 0, flag)
  //
  Chain = DAG.getCALLSEQ_START(getRoot(), NullPtr, DL);
  InFlag = Chain.getValue(1);

  // Add the <id> and <numBytes> constants.
  SDValue IDVal = getValue(CI.getOperand(PatchPointOpers::IDPos));
  Ops.push_back(DAG.getTargetConstant(
                  cast<ConstantSDNode>(IDVal)->getZExtValue(), DL, MVT::i64));
  SDValue NBytesVal = getValue(CI.getOperand(PatchPointOpers::NBytesPos));
  Ops.push_back(DAG.getTargetConstant(
                  cast<ConstantSDNode>(NBytesVal)->getZExtValue(), DL,
                  MVT::i32));

  // Push live variables for the stack map.
  addStackMapLiveVars(&CI, 2, DL, Ops, *this);

  // We are not pushing any register mask info here on the operands list,
  // because the stackmap doesn't clobber anything.

  // Push the chain and the glue flag.
  Ops.push_back(Chain);
  Ops.push_back(InFlag);

  // Create the STACKMAP node.
  SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Glue);
  SDNode *SM = DAG.getMachineNode(TargetOpcode::STACKMAP, DL, NodeTys, Ops);
  Chain = SDValue(SM, 0);
  InFlag = Chain.getValue(1);

  Chain = DAG.getCALLSEQ_END(Chain, NullPtr, NullPtr, InFlag, DL);

  // Stackmaps don't generate values, so nothing goes into the NodeMap.

  // Set the root to the target-lowered call chain.
  DAG.setRoot(Chain);

  // Inform the Frame Information that we have a stackmap in this function.
  FuncInfo.MF->getFrameInfo()->setHasStackMap();
}

/// \brief Lower llvm.experimental.patchpoint directly to its target opcode.
void SelectionDAGBuilder::visitPatchpoint(ImmutableCallSite CS,
                                          MachineBasicBlock *LandingPad) {
  // void|i64 @llvm.experimental.patchpoint.void|i64(i64 <id>,
  //                                                 i32 <numBytes>,
  //                                                 i8* <target>,
  //                                                 i32 <numArgs>,
  //                                                 [Args...],
  //                                                 [live variables...])

  CallingConv::ID CC = CS.getCallingConv();
  bool IsAnyRegCC = CC == CallingConv::AnyReg;
  bool HasDef = !CS->getType()->isVoidTy();
  SDLoc dl = getCurSDLoc();
  SDValue Callee = getValue(CS->getOperand(PatchPointOpers::TargetPos));

  // Handle immediate and symbolic callees.
  if (auto* ConstCallee = dyn_cast<ConstantSDNode>(Callee))
    Callee = DAG.getIntPtrConstant(ConstCallee->getZExtValue(), dl,
                                   /*isTarget=*/true);
  else if (auto* SymbolicCallee = dyn_cast<GlobalAddressSDNode>(Callee))
    Callee =  DAG.getTargetGlobalAddress(SymbolicCallee->getGlobal(),
                                         SDLoc(SymbolicCallee),
                                         SymbolicCallee->getValueType(0));

  // Get the real number of arguments participating in the call <numArgs>
  SDValue NArgVal = getValue(CS.getArgument(PatchPointOpers::NArgPos));
  unsigned NumArgs = cast<ConstantSDNode>(NArgVal)->getZExtValue();

  // Skip the four meta args: <id>, <numNopBytes>, <target>, <numArgs>
  // Intrinsics include all meta-operands up to but not including CC.
  unsigned NumMetaOpers = PatchPointOpers::CCPos;
  assert(CS.arg_size() >= NumMetaOpers + NumArgs &&
         "Not enough arguments provided to the patchpoint intrinsic");

  // For AnyRegCC the arguments are lowered later on manually.
  unsigned NumCallArgs = IsAnyRegCC ? 0 : NumArgs;
  Type *ReturnTy =
    IsAnyRegCC ? Type::getVoidTy(*DAG.getContext()) : CS->getType();
  std::pair<SDValue, SDValue> Result =
    lowerCallOperands(CS, NumMetaOpers, NumCallArgs, Callee, ReturnTy,
                      LandingPad, true);

  SDNode *CallEnd = Result.second.getNode();
  if (HasDef && (CallEnd->getOpcode() == ISD::CopyFromReg))
    CallEnd = CallEnd->getOperand(0).getNode();

  /// Get a call instruction from the call sequence chain.
  /// Tail calls are not allowed.
  assert(CallEnd->getOpcode() == ISD::CALLSEQ_END &&
         "Expected a callseq node.");
  SDNode *Call = CallEnd->getOperand(0).getNode();
  bool HasGlue = Call->getGluedNode();

  // Replace the target specific call node with the patchable intrinsic.
  SmallVector<SDValue, 8> Ops;

  // Add the <id> and <numBytes> constants.
  SDValue IDVal = getValue(CS->getOperand(PatchPointOpers::IDPos));
  Ops.push_back(DAG.getTargetConstant(
                  cast<ConstantSDNode>(IDVal)->getZExtValue(), dl, MVT::i64));
  SDValue NBytesVal = getValue(CS->getOperand(PatchPointOpers::NBytesPos));
  Ops.push_back(DAG.getTargetConstant(
                  cast<ConstantSDNode>(NBytesVal)->getZExtValue(), dl,
                  MVT::i32));

  // Add the callee.
  Ops.push_back(Callee);

  // Adjust <numArgs> to account for any arguments that have been passed on the
  // stack instead.
  // Call Node: Chain, Target, {Args}, RegMask, [Glue]
  unsigned NumCallRegArgs = Call->getNumOperands() - (HasGlue ? 4 : 3);
  NumCallRegArgs = IsAnyRegCC ? NumArgs : NumCallRegArgs;
  Ops.push_back(DAG.getTargetConstant(NumCallRegArgs, dl, MVT::i32));

  // Add the calling convention
  Ops.push_back(DAG.getTargetConstant((unsigned)CC, dl, MVT::i32));

  // Add the arguments we omitted previously. The register allocator should
  // place these in any free register.
  if (IsAnyRegCC)
    for (unsigned i = NumMetaOpers, e = NumMetaOpers + NumArgs; i != e; ++i)
      Ops.push_back(getValue(CS.getArgument(i)));

  // Push the arguments from the call instruction up to the register mask.
  SDNode::op_iterator e = HasGlue ? Call->op_end()-2 : Call->op_end()-1;
  Ops.append(Call->op_begin() + 2, e);

  // Push live variables for the stack map.
  addStackMapLiveVars(CS, NumMetaOpers + NumArgs, dl, Ops, *this);

  // Push the register mask info.
  if (HasGlue)
    Ops.push_back(*(Call->op_end()-2));
  else
    Ops.push_back(*(Call->op_end()-1));

  // Push the chain (this is originally the first operand of the call, but
  // becomes now the last or second to last operand).
  Ops.push_back(*(Call->op_begin()));

  // Push the glue flag (last operand).
  if (HasGlue)
    Ops.push_back(*(Call->op_end()-1));

  SDVTList NodeTys;
  if (IsAnyRegCC && HasDef) {
    // Create the return types based on the intrinsic definition
    const TargetLowering &TLI = DAG.getTargetLoweringInfo();
    SmallVector<EVT, 3> ValueVTs;
    ComputeValueVTs(TLI, DAG.getDataLayout(), CS->getType(), ValueVTs);
    assert(ValueVTs.size() == 1 && "Expected only one return value type.");

    // There is always a chain and a glue type at the end
    ValueVTs.push_back(MVT::Other);
    ValueVTs.push_back(MVT::Glue);
    NodeTys = DAG.getVTList(ValueVTs);
  } else
    NodeTys = DAG.getVTList(MVT::Other, MVT::Glue);

  // Replace the target specific call node with a PATCHPOINT node.
  MachineSDNode *MN = DAG.getMachineNode(TargetOpcode::PATCHPOINT,
                                         dl, NodeTys, Ops);

  // Update the NodeMap.
  if (HasDef) {
    if (IsAnyRegCC)
      setValue(CS.getInstruction(), SDValue(MN, 0));
    else
      setValue(CS.getInstruction(), Result.first);
  }

  // Fixup the consumers of the intrinsic. The chain and glue may be used in the
  // call sequence. Furthermore the location of the chain and glue can change
  // when the AnyReg calling convention is used and the intrinsic returns a
  // value.
  if (IsAnyRegCC && HasDef) {
    SDValue From[] = {SDValue(Call, 0), SDValue(Call, 1)};
    SDValue To[] = {SDValue(MN, 1), SDValue(MN, 2)};
    DAG.ReplaceAllUsesOfValuesWith(From, To, 2);
  } else
    DAG.ReplaceAllUsesWith(Call, MN);
  DAG.DeleteNode(Call);

  // Inform the Frame Information that we have a patchpoint in this function.
  FuncInfo.MF->getFrameInfo()->setHasPatchPoint();
}

/// Returns an AttributeSet representing the attributes applied to the return
/// value of the given call.
static AttributeSet getReturnAttrs(TargetLowering::CallLoweringInfo &CLI) {
  SmallVector<Attribute::AttrKind, 2> Attrs;
  if (CLI.RetSExt)
    Attrs.push_back(Attribute::SExt);
  if (CLI.RetZExt)
    Attrs.push_back(Attribute::ZExt);
  if (CLI.IsInReg)
    Attrs.push_back(Attribute::InReg);

  return AttributeSet::get(CLI.RetTy->getContext(), AttributeSet::ReturnIndex,
                           Attrs);
}

/// TargetLowering::LowerCallTo - This is the default LowerCallTo
/// implementation, which just calls LowerCall.
/// FIXME: When all targets are
/// migrated to using LowerCall, this hook should be integrated into SDISel.
std::pair<SDValue, SDValue>
TargetLowering::LowerCallTo(TargetLowering::CallLoweringInfo &CLI) const {
  // Handle the incoming return values from the call.
  CLI.Ins.clear();
  Type *OrigRetTy = CLI.RetTy;
  SmallVector<EVT, 4> RetTys;
  SmallVector<uint64_t, 4> Offsets;
  auto &DL = CLI.DAG.getDataLayout();
  ComputeValueVTs(*this, DL, CLI.RetTy, RetTys, &Offsets);

  SmallVector<ISD::OutputArg, 4> Outs;
  GetReturnInfo(CLI.RetTy, getReturnAttrs(CLI), Outs, *this, DL);

  bool CanLowerReturn =
      this->CanLowerReturn(CLI.CallConv, CLI.DAG.getMachineFunction(),
                           CLI.IsVarArg, Outs, CLI.RetTy->getContext());

  SDValue DemoteStackSlot;
  int DemoteStackIdx = -100;
  if (!CanLowerReturn) {
    // FIXME: equivalent assert?
    // assert(!CS.hasInAllocaArgument() &&
    //        "sret demotion is incompatible with inalloca");
    uint64_t TySize = DL.getTypeAllocSize(CLI.RetTy);
    unsigned Align = DL.getPrefTypeAlignment(CLI.RetTy);
    MachineFunction &MF = CLI.DAG.getMachineFunction();
    DemoteStackIdx = MF.getFrameInfo()->CreateStackObject(TySize, Align, false);
    Type *StackSlotPtrType = PointerType::getUnqual(CLI.RetTy);

    DemoteStackSlot = CLI.DAG.getFrameIndex(DemoteStackIdx, getPointerTy(DL));
    ArgListEntry Entry;
    Entry.Node = DemoteStackSlot;
    Entry.Ty = StackSlotPtrType;
    Entry.isSExt = false;
    Entry.isZExt = false;
    Entry.isInReg = false;
    Entry.isSRet = true;
    Entry.isNest = false;
    Entry.isByVal = false;
    Entry.isReturned = false;
    Entry.Alignment = Align;
    CLI.getArgs().insert(CLI.getArgs().begin(), Entry);
    CLI.RetTy = Type::getVoidTy(CLI.RetTy->getContext());

    // sret demotion isn't compatible with tail-calls, since the sret argument
    // points into the callers stack frame.
    CLI.IsTailCall = false;
  } else {
    for (unsigned I = 0, E = RetTys.size(); I != E; ++I) {
      EVT VT = RetTys[I];
      MVT RegisterVT = getRegisterType(CLI.RetTy->getContext(), VT);
      unsigned NumRegs = getNumRegisters(CLI.RetTy->getContext(), VT);
      for (unsigned i = 0; i != NumRegs; ++i) {
        ISD::InputArg MyFlags;
        MyFlags.VT = RegisterVT;
        MyFlags.ArgVT = VT;
        MyFlags.Used = CLI.IsReturnValueUsed;
        if (CLI.RetSExt)
          MyFlags.Flags.setSExt();
        if (CLI.RetZExt)
          MyFlags.Flags.setZExt();
        if (CLI.IsInReg)
          MyFlags.Flags.setInReg();
        CLI.Ins.push_back(MyFlags);
      }
    }
  }

  // Handle all of the outgoing arguments.
  CLI.Outs.clear();
  CLI.OutVals.clear();
  ArgListTy &Args = CLI.getArgs();
  for (unsigned i = 0, e = Args.size(); i != e; ++i) {
    SmallVector<EVT, 4> ValueVTs;
    ComputeValueVTs(*this, DL, Args[i].Ty, ValueVTs);
    Type *FinalType = Args[i].Ty;
    if (Args[i].isByVal)
      FinalType = cast<PointerType>(Args[i].Ty)->getElementType();
    bool NeedsRegBlock = functionArgumentNeedsConsecutiveRegisters(
        FinalType, CLI.CallConv, CLI.IsVarArg);
    for (unsigned Value = 0, NumValues = ValueVTs.size(); Value != NumValues;
         ++Value) {
      EVT VT = ValueVTs[Value];
      Type *ArgTy = VT.getTypeForEVT(CLI.RetTy->getContext());
      SDValue Op = SDValue(Args[i].Node.getNode(),
                           Args[i].Node.getResNo() + Value);
      ISD::ArgFlagsTy Flags;
      unsigned OriginalAlignment = DL.getABITypeAlignment(ArgTy);

      if (Args[i].isZExt)
        Flags.setZExt();
      if (Args[i].isSExt)
        Flags.setSExt();
      if (Args[i].isInReg)
        Flags.setInReg();
      if (Args[i].isSRet)
        Flags.setSRet();
      if (Args[i].isByVal)
        Flags.setByVal();
      if (Args[i].isInAlloca) {
        Flags.setInAlloca();
        // Set the byval flag for CCAssignFn callbacks that don't know about
        // inalloca.  This way we can know how many bytes we should've allocated
        // and how many bytes a callee cleanup function will pop.  If we port
        // inalloca to more targets, we'll have to add custom inalloca handling
        // in the various CC lowering callbacks.
        Flags.setByVal();
      }
      if (Args[i].isByVal || Args[i].isInAlloca) {
        PointerType *Ty = cast<PointerType>(Args[i].Ty);
        Type *ElementTy = Ty->getElementType();
        Flags.setByValSize(DL.getTypeAllocSize(ElementTy));
        // For ByVal, alignment should come from FE.  BE will guess if this
        // info is not there but there are cases it cannot get right.
        unsigned FrameAlign;
        if (Args[i].Alignment)
          FrameAlign = Args[i].Alignment;
        else
          FrameAlign = getByValTypeAlignment(ElementTy, DL);
        Flags.setByValAlign(FrameAlign);
      }
      if (Args[i].isNest)
        Flags.setNest();
      if (NeedsRegBlock)
        Flags.setInConsecutiveRegs();
      Flags.setOrigAlign(OriginalAlignment);

      MVT PartVT = getRegisterType(CLI.RetTy->getContext(), VT);
      unsigned NumParts = getNumRegisters(CLI.RetTy->getContext(), VT);
      SmallVector<SDValue, 4> Parts(NumParts);
      ISD::NodeType ExtendKind = ISD::ANY_EXTEND;

      if (Args[i].isSExt)
        ExtendKind = ISD::SIGN_EXTEND;
      else if (Args[i].isZExt)
        ExtendKind = ISD::ZERO_EXTEND;

      // Conservatively only handle 'returned' on non-vectors for now
      if (Args[i].isReturned && !Op.getValueType().isVector()) {
        assert(CLI.RetTy == Args[i].Ty && RetTys.size() == NumValues &&
               "unexpected use of 'returned'");
        // Before passing 'returned' to the target lowering code, ensure that
        // either the register MVT and the actual EVT are the same size or that
        // the return value and argument are extended in the same way; in these
        // cases it's safe to pass the argument register value unchanged as the
        // return register value (although it's at the target's option whether
        // to do so)
        // TODO: allow code generation to take advantage of partially preserved
        // registers rather than clobbering the entire register when the
        // parameter extension method is not compatible with the return
        // extension method
        if ((NumParts * PartVT.getSizeInBits() == VT.getSizeInBits()) ||
            (ExtendKind != ISD::ANY_EXTEND &&
             CLI.RetSExt == Args[i].isSExt && CLI.RetZExt == Args[i].isZExt))
        Flags.setReturned();
      }

      getCopyToParts(CLI.DAG, CLI.DL, Op, &Parts[0], NumParts, PartVT,
                     CLI.CS ? CLI.CS->getInstruction() : nullptr, ExtendKind);

      for (unsigned j = 0; j != NumParts; ++j) {
        // if it isn't first piece, alignment must be 1
        ISD::OutputArg MyFlags(Flags, Parts[j].getValueType(), VT,
                               i < CLI.NumFixedArgs,
                               i, j*Parts[j].getValueType().getStoreSize());
        if (NumParts > 1 && j == 0)
          MyFlags.Flags.setSplit();
        else if (j != 0)
          MyFlags.Flags.setOrigAlign(1);

        CLI.Outs.push_back(MyFlags);
        CLI.OutVals.push_back(Parts[j]);
      }

      if (NeedsRegBlock && Value == NumValues - 1)
        CLI.Outs[CLI.Outs.size() - 1].Flags.setInConsecutiveRegsLast();
    }
  }

  SmallVector<SDValue, 4> InVals;
  CLI.Chain = LowerCall(CLI, InVals);

  // Verify that the target's LowerCall behaved as expected.
  assert(CLI.Chain.getNode() && CLI.Chain.getValueType() == MVT::Other &&
         "LowerCall didn't return a valid chain!");
  assert((!CLI.IsTailCall || InVals.empty()) &&
         "LowerCall emitted a return value for a tail call!");
  assert((CLI.IsTailCall || InVals.size() == CLI.Ins.size()) &&
         "LowerCall didn't emit the correct number of values!");

  // For a tail call, the return value is merely live-out and there aren't
  // any nodes in the DAG representing it. Return a special value to
  // indicate that a tail call has been emitted and no more Instructions
  // should be processed in the current block.
  if (CLI.IsTailCall) {
    CLI.DAG.setRoot(CLI.Chain);
    return std::make_pair(SDValue(), SDValue());
  }

  DEBUG(for (unsigned i = 0, e = CLI.Ins.size(); i != e; ++i) {
          assert(InVals[i].getNode() &&
                 "LowerCall emitted a null value!");
          assert(EVT(CLI.Ins[i].VT) == InVals[i].getValueType() &&
                 "LowerCall emitted a value with the wrong type!");
        });

  SmallVector<SDValue, 4> ReturnValues;
  if (!CanLowerReturn) {
    // The instruction result is the result of loading from the
    // hidden sret parameter.
    SmallVector<EVT, 1> PVTs;
    Type *PtrRetTy = PointerType::getUnqual(OrigRetTy);

    ComputeValueVTs(*this, DL, PtrRetTy, PVTs);
    assert(PVTs.size() == 1 && "Pointers should fit in one register");
    EVT PtrVT = PVTs[0];

    unsigned NumValues = RetTys.size();
    ReturnValues.resize(NumValues);
    SmallVector<SDValue, 4> Chains(NumValues);

    for (unsigned i = 0; i < NumValues; ++i) {
      SDValue Add = CLI.DAG.getNode(ISD::ADD, CLI.DL, PtrVT, DemoteStackSlot,
                                    CLI.DAG.getConstant(Offsets[i], CLI.DL,
                                                        PtrVT));
      SDValue L = CLI.DAG.getLoad(
          RetTys[i], CLI.DL, CLI.Chain, Add,
          MachinePointerInfo::getFixedStack(DemoteStackIdx, Offsets[i]), false,
          false, false, 1);
      ReturnValues[i] = L;
      Chains[i] = L.getValue(1);
    }

    CLI.Chain = CLI.DAG.getNode(ISD::TokenFactor, CLI.DL, MVT::Other, Chains);
  } else {
    // Collect the legal value parts into potentially illegal values
    // that correspond to the original function's return values.
    ISD::NodeType AssertOp = ISD::DELETED_NODE;
    if (CLI.RetSExt)
      AssertOp = ISD::AssertSext;
    else if (CLI.RetZExt)
      AssertOp = ISD::AssertZext;
    unsigned CurReg = 0;
    for (unsigned I = 0, E = RetTys.size(); I != E; ++I) {
      EVT VT = RetTys[I];
      MVT RegisterVT = getRegisterType(CLI.RetTy->getContext(), VT);
      unsigned NumRegs = getNumRegisters(CLI.RetTy->getContext(), VT);

      ReturnValues.push_back(getCopyFromParts(CLI.DAG, CLI.DL, &InVals[CurReg],
                                              NumRegs, RegisterVT, VT, nullptr,
                                              AssertOp));
      CurReg += NumRegs;
    }

    // For a function returning void, there is no return value. We can't create
    // such a node, so we just return a null return value in that case. In
    // that case, nothing will actually look at the value.
    if (ReturnValues.empty())
      return std::make_pair(SDValue(), CLI.Chain);
  }

  SDValue Res = CLI.DAG.getNode(ISD::MERGE_VALUES, CLI.DL,
                                CLI.DAG.getVTList(RetTys), ReturnValues);
  return std::make_pair(Res, CLI.Chain);
}

void TargetLowering::LowerOperationWrapper(SDNode *N,
                                           SmallVectorImpl<SDValue> &Results,
                                           SelectionDAG &DAG) const {
  SDValue Res = LowerOperation(SDValue(N, 0), DAG);
  if (Res.getNode())
    Results.push_back(Res);
}

SDValue TargetLowering::LowerOperation(SDValue Op, SelectionDAG &DAG) const {
  llvm_unreachable("LowerOperation not implemented for this target!");
}

void
SelectionDAGBuilder::CopyValueToVirtualRegister(const Value *V, unsigned Reg) {
  SDValue Op = getNonRegisterValue(V);
  assert((Op.getOpcode() != ISD::CopyFromReg ||
          cast<RegisterSDNode>(Op.getOperand(1))->getReg() != Reg) &&
         "Copy from a reg to the same reg!");
  assert(!TargetRegisterInfo::isPhysicalRegister(Reg) && "Is a physreg");

  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  RegsForValue RFV(V->getContext(), TLI, DAG.getDataLayout(), Reg,
                   V->getType());
  SDValue Chain = DAG.getEntryNode();

  ISD::NodeType ExtendType = (FuncInfo.PreferredExtendType.find(V) ==
                              FuncInfo.PreferredExtendType.end())
                                 ? ISD::ANY_EXTEND
                                 : FuncInfo.PreferredExtendType[V];
  RFV.getCopyToRegs(Op, DAG, getCurSDLoc(), Chain, nullptr, V, ExtendType);
  PendingExports.push_back(Chain);
}

#include "llvm/CodeGen/SelectionDAGISel.h"

/// isOnlyUsedInEntryBlock - If the specified argument is only used in the
/// entry block, return true.  This includes arguments used by switches, since
/// the switch may expand into multiple basic blocks.
static bool isOnlyUsedInEntryBlock(const Argument *A, bool FastISel) {
  // With FastISel active, we may be splitting blocks, so force creation
  // of virtual registers for all non-dead arguments.
  if (FastISel)
    return A->use_empty();

  const BasicBlock *Entry = A->getParent()->begin();
  for (const User *U : A->users())
    if (cast<Instruction>(U)->getParent() != Entry || isa<SwitchInst>(U))
      return false;  // Use not in entry block.

  return true;
}

void SelectionDAGISel::LowerArguments(const Function &F) {
  SelectionDAG &DAG = SDB->DAG;
  SDLoc dl = SDB->getCurSDLoc();
  const DataLayout &DL = DAG.getDataLayout();
  SmallVector<ISD::InputArg, 16> Ins;

  if (!FuncInfo->CanLowerReturn) {
    // Put in an sret pointer parameter before all the other parameters.
    SmallVector<EVT, 1> ValueVTs;
    ComputeValueVTs(*TLI, DAG.getDataLayout(),
                    PointerType::getUnqual(F.getReturnType()), ValueVTs);

    // NOTE: Assuming that a pointer will never break down to more than one VT
    // or one register.
    ISD::ArgFlagsTy Flags;
    Flags.setSRet();
    MVT RegisterVT = TLI->getRegisterType(*DAG.getContext(), ValueVTs[0]);
    ISD::InputArg RetArg(Flags, RegisterVT, ValueVTs[0], true,
                         ISD::InputArg::NoArgIndex, 0);
    Ins.push_back(RetArg);
  }

  // Set up the incoming argument description vector.
  unsigned Idx = 1;
  for (Function::const_arg_iterator I = F.arg_begin(), E = F.arg_end();
       I != E; ++I, ++Idx) {
    SmallVector<EVT, 4> ValueVTs;
    ComputeValueVTs(*TLI, DAG.getDataLayout(), I->getType(), ValueVTs);
    bool isArgValueUsed = !I->use_empty();
    unsigned PartBase = 0;
    Type *FinalType = I->getType();
    if (F.getAttributes().hasAttribute(Idx, Attribute::ByVal))
      FinalType = cast<PointerType>(FinalType)->getElementType();
    bool NeedsRegBlock = TLI->functionArgumentNeedsConsecutiveRegisters(
        FinalType, F.getCallingConv(), F.isVarArg());
    for (unsigned Value = 0, NumValues = ValueVTs.size();
         Value != NumValues; ++Value) {
      EVT VT = ValueVTs[Value];
      Type *ArgTy = VT.getTypeForEVT(*DAG.getContext());
      ISD::ArgFlagsTy Flags;
      unsigned OriginalAlignment = DL.getABITypeAlignment(ArgTy);

      if (F.getAttributes().hasAttribute(Idx, Attribute::ZExt))
        Flags.setZExt();
      if (F.getAttributes().hasAttribute(Idx, Attribute::SExt))
        Flags.setSExt();
      if (F.getAttributes().hasAttribute(Idx, Attribute::InReg))
        Flags.setInReg();
      if (F.getAttributes().hasAttribute(Idx, Attribute::StructRet))
        Flags.setSRet();
      if (F.getAttributes().hasAttribute(Idx, Attribute::ByVal))
        Flags.setByVal();
      if (F.getAttributes().hasAttribute(Idx, Attribute::InAlloca)) {
        Flags.setInAlloca();
        // Set the byval flag for CCAssignFn callbacks that don't know about
        // inalloca.  This way we can know how many bytes we should've allocated
        // and how many bytes a callee cleanup function will pop.  If we port
        // inalloca to more targets, we'll have to add custom inalloca handling
        // in the various CC lowering callbacks.
        Flags.setByVal();
      }
      if (Flags.isByVal() || Flags.isInAlloca()) {
        PointerType *Ty = cast<PointerType>(I->getType());
        Type *ElementTy = Ty->getElementType();
        Flags.setByValSize(DL.getTypeAllocSize(ElementTy));
        // For ByVal, alignment should be passed from FE.  BE will guess if
        // this info is not there but there are cases it cannot get right.
        unsigned FrameAlign;
        if (F.getParamAlignment(Idx))
          FrameAlign = F.getParamAlignment(Idx);
        else
          FrameAlign = TLI->getByValTypeAlignment(ElementTy, DL);
        Flags.setByValAlign(FrameAlign);
      }
      if (F.getAttributes().hasAttribute(Idx, Attribute::Nest))
        Flags.setNest();
      if (NeedsRegBlock)
        Flags.setInConsecutiveRegs();
      Flags.setOrigAlign(OriginalAlignment);

      MVT RegisterVT = TLI->getRegisterType(*CurDAG->getContext(), VT);
      unsigned NumRegs = TLI->getNumRegisters(*CurDAG->getContext(), VT);
      for (unsigned i = 0; i != NumRegs; ++i) {
        ISD::InputArg MyFlags(Flags, RegisterVT, VT, isArgValueUsed,
                              Idx-1, PartBase+i*RegisterVT.getStoreSize());
        if (NumRegs > 1 && i == 0)
          MyFlags.Flags.setSplit();
        // if it isn't first piece, alignment must be 1
        else if (i > 0)
          MyFlags.Flags.setOrigAlign(1);
        Ins.push_back(MyFlags);
      }
      if (NeedsRegBlock && Value == NumValues - 1)
        Ins[Ins.size() - 1].Flags.setInConsecutiveRegsLast();
      PartBase += VT.getStoreSize();
    }
  }

  // Call the target to set up the argument values.
  SmallVector<SDValue, 8> InVals;
  SDValue NewRoot = TLI->LowerFormalArguments(
      DAG.getRoot(), F.getCallingConv(), F.isVarArg(), Ins, dl, DAG, InVals);

  // Verify that the target's LowerFormalArguments behaved as expected.
  assert(NewRoot.getNode() && NewRoot.getValueType() == MVT::Other &&
         "LowerFormalArguments didn't return a valid chain!");
  assert(InVals.size() == Ins.size() &&
         "LowerFormalArguments didn't emit the correct number of values!");
  DEBUG({
      for (unsigned i = 0, e = Ins.size(); i != e; ++i) {
        assert(InVals[i].getNode() &&
               "LowerFormalArguments emitted a null value!");
        assert(EVT(Ins[i].VT) == InVals[i].getValueType() &&
               "LowerFormalArguments emitted a value with the wrong type!");
      }
    });

  // Update the DAG with the new chain value resulting from argument lowering.
  DAG.setRoot(NewRoot);

  // Set up the argument values.
  unsigned i = 0;
  Idx = 1;
  if (!FuncInfo->CanLowerReturn) {
    // Create a virtual register for the sret pointer, and put in a copy
    // from the sret argument into it.
    SmallVector<EVT, 1> ValueVTs;
    ComputeValueVTs(*TLI, DAG.getDataLayout(),
                    PointerType::getUnqual(F.getReturnType()), ValueVTs);
    MVT VT = ValueVTs[0].getSimpleVT();
    MVT RegVT = TLI->getRegisterType(*CurDAG->getContext(), VT);
    ISD::NodeType AssertOp = ISD::DELETED_NODE;
    SDValue ArgValue = getCopyFromParts(DAG, dl, &InVals[0], 1,
                                        RegVT, VT, nullptr, AssertOp);

    MachineFunction& MF = SDB->DAG.getMachineFunction();
    MachineRegisterInfo& RegInfo = MF.getRegInfo();
    unsigned SRetReg = RegInfo.createVirtualRegister(TLI->getRegClassFor(RegVT));
    FuncInfo->DemoteRegister = SRetReg;
    NewRoot =
        SDB->DAG.getCopyToReg(NewRoot, SDB->getCurSDLoc(), SRetReg, ArgValue);
    DAG.setRoot(NewRoot);

    // i indexes lowered arguments.  Bump it past the hidden sret argument.
    // Idx indexes LLVM arguments.  Don't touch it.
    ++i;
  }

  for (Function::const_arg_iterator I = F.arg_begin(), E = F.arg_end(); I != E;
      ++I, ++Idx) {
    SmallVector<SDValue, 4> ArgValues;
    SmallVector<EVT, 4> ValueVTs;
    ComputeValueVTs(*TLI, DAG.getDataLayout(), I->getType(), ValueVTs);
    unsigned NumValues = ValueVTs.size();

    // If this argument is unused then remember its value. It is used to generate
    // debugging information.
    if (I->use_empty() && NumValues) {
      SDB->setUnusedArgValue(I, InVals[i]);

      // Also remember any frame index for use in FastISel.
      if (FrameIndexSDNode *FI =
          dyn_cast<FrameIndexSDNode>(InVals[i].getNode()))
        FuncInfo->setArgumentFrameIndex(I, FI->getIndex());
    }

    for (unsigned Val = 0; Val != NumValues; ++Val) {
      EVT VT = ValueVTs[Val];
      MVT PartVT = TLI->getRegisterType(*CurDAG->getContext(), VT);
      unsigned NumParts = TLI->getNumRegisters(*CurDAG->getContext(), VT);

      if (!I->use_empty()) {
        ISD::NodeType AssertOp = ISD::DELETED_NODE;
        if (F.getAttributes().hasAttribute(Idx, Attribute::SExt))
          AssertOp = ISD::AssertSext;
        else if (F.getAttributes().hasAttribute(Idx, Attribute::ZExt))
          AssertOp = ISD::AssertZext;

        ArgValues.push_back(getCopyFromParts(DAG, dl, &InVals[i],
                                             NumParts, PartVT, VT,
                                             nullptr, AssertOp));
      }

      i += NumParts;
    }

    // We don't need to do anything else for unused arguments.
    if (ArgValues.empty())
      continue;

    // Note down frame index.
    if (FrameIndexSDNode *FI =
        dyn_cast<FrameIndexSDNode>(ArgValues[0].getNode()))
      FuncInfo->setArgumentFrameIndex(I, FI->getIndex());

    SDValue Res = DAG.getMergeValues(makeArrayRef(ArgValues.data(), NumValues),
                                     SDB->getCurSDLoc());

    SDB->setValue(I, Res);
    if (!TM.Options.EnableFastISel && Res.getOpcode() == ISD::BUILD_PAIR) {
      if (LoadSDNode *LNode =
          dyn_cast<LoadSDNode>(Res.getOperand(0).getNode()))
        if (FrameIndexSDNode *FI =
            dyn_cast<FrameIndexSDNode>(LNode->getBasePtr().getNode()))
        FuncInfo->setArgumentFrameIndex(I, FI->getIndex());
    }

    // If this argument is live outside of the entry block, insert a copy from
    // wherever we got it to the vreg that other BB's will reference it as.
    if (!TM.Options.EnableFastISel && Res.getOpcode() == ISD::CopyFromReg) {
      // If we can, though, try to skip creating an unnecessary vreg.
      // FIXME: This isn't very clean... it would be nice to make this more
      // general.  It's also subtly incompatible with the hacks FastISel
      // uses with vregs.
      unsigned Reg = cast<RegisterSDNode>(Res.getOperand(1))->getReg();
      if (TargetRegisterInfo::isVirtualRegister(Reg)) {
        FuncInfo->ValueMap[I] = Reg;
        continue;
      }
    }
    if (!isOnlyUsedInEntryBlock(I, TM.Options.EnableFastISel)) {
      FuncInfo->InitializeRegForValue(I);
      SDB->CopyToExportRegsIfNeeded(I);
    }
  }

  assert(i == InVals.size() && "Argument register count mismatch!");

  // Finally, if the target has anything special to do, allow it to do so.
  EmitFunctionEntryCode();
}

/// Handle PHI nodes in successor blocks.  Emit code into the SelectionDAG to
/// ensure constants are generated when needed.  Remember the virtual registers
/// that need to be added to the Machine PHI nodes as input.  We cannot just
/// directly add them, because expansion might result in multiple MBB's for one
/// BB.  As such, the start of the BB might correspond to a different MBB than
/// the end.
///
void
SelectionDAGBuilder::HandlePHINodesInSuccessorBlocks(const BasicBlock *LLVMBB) {
  const TerminatorInst *TI = LLVMBB->getTerminator();

  SmallPtrSet<MachineBasicBlock *, 4> SuccsHandled;

  // Check PHI nodes in successors that expect a value to be available from this
  // block.
  for (unsigned succ = 0, e = TI->getNumSuccessors(); succ != e; ++succ) {
    const BasicBlock *SuccBB = TI->getSuccessor(succ);
    if (!isa<PHINode>(SuccBB->begin())) continue;
    MachineBasicBlock *SuccMBB = FuncInfo.MBBMap[SuccBB];

    // If this terminator has multiple identical successors (common for
    // switches), only handle each succ once.
    if (!SuccsHandled.insert(SuccMBB).second)
      continue;

    MachineBasicBlock::iterator MBBI = SuccMBB->begin();

    // At this point we know that there is a 1-1 correspondence between LLVM PHI
    // nodes and Machine PHI nodes, but the incoming operands have not been
    // emitted yet.
    for (BasicBlock::const_iterator I = SuccBB->begin();
         const PHINode *PN = dyn_cast<PHINode>(I); ++I) {
      // Ignore dead phi's.
      if (PN->use_empty()) continue;

      // Skip empty types
      if (PN->getType()->isEmptyTy())
        continue;

      unsigned Reg;
      const Value *PHIOp = PN->getIncomingValueForBlock(LLVMBB);

      if (const Constant *C = dyn_cast<Constant>(PHIOp)) {
        unsigned &RegOut = ConstantsOut[C];
        if (RegOut == 0) {
          RegOut = FuncInfo.CreateRegs(C->getType());
          CopyValueToVirtualRegister(C, RegOut);
        }
        Reg = RegOut;
      } else {
        DenseMap<const Value *, unsigned>::iterator I =
          FuncInfo.ValueMap.find(PHIOp);
        if (I != FuncInfo.ValueMap.end())
          Reg = I->second;
        else {
          assert(isa<AllocaInst>(PHIOp) &&
                 FuncInfo.StaticAllocaMap.count(cast<AllocaInst>(PHIOp)) &&
                 "Didn't codegen value into a register!??");
          Reg = FuncInfo.CreateRegs(PHIOp->getType());
          CopyValueToVirtualRegister(PHIOp, Reg);
        }
      }

      // Remember that this register needs to added to the machine PHI node as
      // the input for this MBB.
      SmallVector<EVT, 4> ValueVTs;
      const TargetLowering &TLI = DAG.getTargetLoweringInfo();
      ComputeValueVTs(TLI, DAG.getDataLayout(), PN->getType(), ValueVTs);
      for (unsigned vti = 0, vte = ValueVTs.size(); vti != vte; ++vti) {
        EVT VT = ValueVTs[vti];
        unsigned NumRegisters = TLI.getNumRegisters(*DAG.getContext(), VT);
        for (unsigned i = 0, e = NumRegisters; i != e; ++i)
          FuncInfo.PHINodesToUpdate.push_back(std::make_pair(MBBI++, Reg+i));
        Reg += NumRegisters;
      }
    }
  }

  ConstantsOut.clear();
}

/// Add a successor MBB to ParentMBB< creating a new MachineBB for BB if SuccMBB
/// is 0.
MachineBasicBlock *
SelectionDAGBuilder::StackProtectorDescriptor::
AddSuccessorMBB(const BasicBlock *BB,
                MachineBasicBlock *ParentMBB,
                bool IsLikely,
                MachineBasicBlock *SuccMBB) {
  // If SuccBB has not been created yet, create it.
  if (!SuccMBB) {
    MachineFunction *MF = ParentMBB->getParent();
    MachineFunction::iterator BBI = ParentMBB;
    SuccMBB = MF->CreateMachineBasicBlock(BB);
    MF->insert(++BBI, SuccMBB);
  }
  // Add it as a successor of ParentMBB.
  ParentMBB->addSuccessor(
      SuccMBB, BranchProbabilityInfo::getBranchWeightStackProtector(IsLikely));
  return SuccMBB;
}

MachineBasicBlock *SelectionDAGBuilder::NextBlock(MachineBasicBlock *MBB) {
  MachineFunction::iterator I = MBB;
  if (++I == FuncInfo.MF->end())
    return nullptr;
  return I;
}

/// During lowering new call nodes can be created (such as memset, etc.).
/// Those will become new roots of the current DAG, but complications arise
/// when they are tail calls. In such cases, the call lowering will update
/// the root, but the builder still needs to know that a tail call has been
/// lowered in order to avoid generating an additional return.
void SelectionDAGBuilder::updateDAGForMaybeTailCall(SDValue MaybeTC) {
  // If the node is null, we do have a tail call.
  if (MaybeTC.getNode() != nullptr)
    DAG.setRoot(MaybeTC);
  else
    HasTailCall = true;
}

bool SelectionDAGBuilder::isDense(const CaseClusterVector &Clusters,
                                  unsigned *TotalCases, unsigned First,
                                  unsigned Last) {
  assert(Last >= First);
  assert(TotalCases[Last] >= TotalCases[First]);

  APInt LowCase = Clusters[First].Low->getValue();
  APInt HighCase = Clusters[Last].High->getValue();
  assert(LowCase.getBitWidth() == HighCase.getBitWidth());

  // FIXME: A range of consecutive cases has 100% density, but only requires one
  // comparison to lower. We should discriminate against such consecutive ranges
  // in jump tables.

  uint64_t Diff = (HighCase - LowCase).getLimitedValue((UINT64_MAX - 1) / 100);
  uint64_t Range = Diff + 1;

  uint64_t NumCases =
      TotalCases[Last] - (First == 0 ? 0 : TotalCases[First - 1]);

  assert(NumCases < UINT64_MAX / 100);
  assert(Range >= NumCases);

  return NumCases * 100 >= Range * MinJumpTableDensity;
}

static inline bool areJTsAllowed(const TargetLowering &TLI) {
  return TLI.isOperationLegalOrCustom(ISD::BR_JT, MVT::Other) ||
         TLI.isOperationLegalOrCustom(ISD::BRIND, MVT::Other);
}

bool SelectionDAGBuilder::buildJumpTable(CaseClusterVector &Clusters,
                                         unsigned First, unsigned Last,
                                         const SwitchInst *SI,
                                         MachineBasicBlock *DefaultMBB,
                                         CaseCluster &JTCluster) {
  assert(First <= Last);

  uint32_t Weight = 0;
  unsigned NumCmps = 0;
  std::vector<MachineBasicBlock*> Table;
  DenseMap<MachineBasicBlock*, uint32_t> JTWeights;
  for (unsigned I = First; I <= Last; ++I) {
    assert(Clusters[I].Kind == CC_Range);
    Weight += Clusters[I].Weight;
    assert(Weight >= Clusters[I].Weight && "Weight overflow!");
    APInt Low = Clusters[I].Low->getValue();
    APInt High = Clusters[I].High->getValue();
    NumCmps += (Low == High) ? 1 : 2;
    if (I != First) {
      // Fill the gap between this and the previous cluster.
      APInt PreviousHigh = Clusters[I - 1].High->getValue();
      assert(PreviousHigh.slt(Low));
      uint64_t Gap = (Low - PreviousHigh).getLimitedValue() - 1;
      for (uint64_t J = 0; J < Gap; J++)
        Table.push_back(DefaultMBB);
    }
    uint64_t ClusterSize = (High - Low).getLimitedValue() + 1;
    for (uint64_t J = 0; J < ClusterSize; ++J)
      Table.push_back(Clusters[I].MBB);
    JTWeights[Clusters[I].MBB] += Clusters[I].Weight;
  }

  unsigned NumDests = JTWeights.size();
  if (isSuitableForBitTests(NumDests, NumCmps,
                            Clusters[First].Low->getValue(),
                            Clusters[Last].High->getValue())) {
    // Clusters[First..Last] should be lowered as bit tests instead.
    return false;
  }

  // Create the MBB that will load from and jump through the table.
  // Note: We create it here, but it's not inserted into the function yet.
  MachineFunction *CurMF = FuncInfo.MF;
  MachineBasicBlock *JumpTableMBB =
      CurMF->CreateMachineBasicBlock(SI->getParent());

  // Add successors. Note: use table order for determinism.
  SmallPtrSet<MachineBasicBlock *, 8> Done;
  for (MachineBasicBlock *Succ : Table) {
    if (Done.count(Succ))
      continue;
    addSuccessorWithWeight(JumpTableMBB, Succ, JTWeights[Succ]);
    Done.insert(Succ);
  }

  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  unsigned JTI = CurMF->getOrCreateJumpTableInfo(TLI.getJumpTableEncoding())
                     ->createJumpTableIndex(Table);

  // Set up the jump table info.
  JumpTable JT(-1U, JTI, JumpTableMBB, nullptr);
  JumpTableHeader JTH(Clusters[First].Low->getValue(),
                      Clusters[Last].High->getValue(), SI->getCondition(),
                      nullptr, false);
  JTCases.emplace_back(std::move(JTH), std::move(JT));

  JTCluster = CaseCluster::jumpTable(Clusters[First].Low, Clusters[Last].High,
                                     JTCases.size() - 1, Weight);
  return true;
}

void SelectionDAGBuilder::findJumpTables(CaseClusterVector &Clusters,
                                         const SwitchInst *SI,
                                         MachineBasicBlock *DefaultMBB) {
#ifndef NDEBUG
  // Clusters must be non-empty, sorted, and only contain Range clusters.
  assert(!Clusters.empty());
  for (CaseCluster &C : Clusters)
    assert(C.Kind == CC_Range);
  for (unsigned i = 1, e = Clusters.size(); i < e; ++i)
    assert(Clusters[i - 1].High->getValue().slt(Clusters[i].Low->getValue()));
#endif

  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  if (!areJTsAllowed(TLI))
    return;

  const int64_t N = Clusters.size();
  const unsigned MinJumpTableSize = TLI.getMinimumJumpTableEntries();

  // TotalCases[i]: Total nbr of cases in Clusters[0..i].
  SmallVector<unsigned, 8> TotalCases(N);

  for (unsigned i = 0; i < N; ++i) {
    APInt Hi = Clusters[i].High->getValue();
    APInt Lo = Clusters[i].Low->getValue();
    TotalCases[i] = (Hi - Lo).getLimitedValue() + 1;
    if (i != 0)
      TotalCases[i] += TotalCases[i - 1];
  }

  if (N >= MinJumpTableSize && isDense(Clusters, &TotalCases[0], 0, N - 1)) {
    // Cheap case: the whole range might be suitable for jump table.
    CaseCluster JTCluster;
    if (buildJumpTable(Clusters, 0, N - 1, SI, DefaultMBB, JTCluster)) {
      Clusters[0] = JTCluster;
      Clusters.resize(1);
      return;
    }
  }

  // The algorithm below is not suitable for -O0.
  if (TM.getOptLevel() == CodeGenOpt::None)
    return;

  // Split Clusters into minimum number of dense partitions. The algorithm uses
  // the same idea as Kannan & Proebsting "Correction to 'Producing Good Code
  // for the Case Statement'" (1994), but builds the MinPartitions array in
  // reverse order to make it easier to reconstruct the partitions in ascending
  // order. In the choice between two optimal partitionings, it picks the one
  // which yields more jump tables.

  // MinPartitions[i] is the minimum nbr of partitions of Clusters[i..N-1].
  SmallVector<unsigned, 8> MinPartitions(N);
  // LastElement[i] is the last element of the partition starting at i.
  SmallVector<unsigned, 8> LastElement(N);
  // NumTables[i]: nbr of >= MinJumpTableSize partitions from Clusters[i..N-1].
  SmallVector<unsigned, 8> NumTables(N);

  // Base case: There is only one way to partition Clusters[N-1].
  MinPartitions[N - 1] = 1;
  LastElement[N - 1] = N - 1;
  assert(MinJumpTableSize > 1);
  NumTables[N - 1] = 0;

  // Note: loop indexes are signed to avoid underflow.
  for (int64_t i = N - 2; i >= 0; i--) {
    // Find optimal partitioning of Clusters[i..N-1].
    // Baseline: Put Clusters[i] into a partition on its own.
    MinPartitions[i] = MinPartitions[i + 1] + 1;
    LastElement[i] = i;
    NumTables[i] = NumTables[i + 1];

    // Search for a solution that results in fewer partitions.
    for (int64_t j = N - 1; j > i; j--) {
      // Try building a partition from Clusters[i..j].
      if (isDense(Clusters, &TotalCases[0], i, j)) {
        unsigned NumPartitions = 1 + (j == N - 1 ? 0 : MinPartitions[j + 1]);
        bool IsTable = j - i + 1 >= MinJumpTableSize;
        unsigned Tables = IsTable + (j == N - 1 ? 0 : NumTables[j + 1]);

        // If this j leads to fewer partitions, or same number of partitions
        // with more lookup tables, it is a better partitioning.
        if (NumPartitions < MinPartitions[i] ||
            (NumPartitions == MinPartitions[i] && Tables > NumTables[i])) {
          MinPartitions[i] = NumPartitions;
          LastElement[i] = j;
          NumTables[i] = Tables;
        }
      }
    }
  }

  // Iterate over the partitions, replacing some with jump tables in-place.
  unsigned DstIndex = 0;
  for (unsigned First = 0, Last; First < N; First = Last + 1) {
    Last = LastElement[First];
    assert(Last >= First);
    assert(DstIndex <= First);
    unsigned NumClusters = Last - First + 1;

    CaseCluster JTCluster;
    if (NumClusters >= MinJumpTableSize &&
        buildJumpTable(Clusters, First, Last, SI, DefaultMBB, JTCluster)) {
      Clusters[DstIndex++] = JTCluster;
    } else {
      for (unsigned I = First; I <= Last; ++I)
        std::memmove(&Clusters[DstIndex++], &Clusters[I], sizeof(Clusters[I]));
    }
  }
  Clusters.resize(DstIndex);
}

bool SelectionDAGBuilder::rangeFitsInWord(const APInt &Low, const APInt &High) {
  // FIXME: Using the pointer type doesn't seem ideal.
  uint64_t BW = DAG.getDataLayout().getPointerSizeInBits();
  uint64_t Range = (High - Low).getLimitedValue(UINT64_MAX - 1) + 1;
  return Range <= BW;
}

bool SelectionDAGBuilder::isSuitableForBitTests(unsigned NumDests,
                                                unsigned NumCmps,
                                                const APInt &Low,
                                                const APInt &High) {
  // FIXME: I don't think NumCmps is the correct metric: a single case and a
  // range of cases both require only one branch to lower. Just looking at the
  // number of clusters and destinations should be enough to decide whether to
  // build bit tests.

  // To lower a range with bit tests, the range must fit the bitwidth of a
  // machine word.
  if (!rangeFitsInWord(Low, High))
    return false;

  // Decide whether it's profitable to lower this range with bit tests. Each
  // destination requires a bit test and branch, and there is an overall range
  // check branch. For a small number of clusters, separate comparisons might be
  // cheaper, and for many destinations, splitting the range might be better.
  return (NumDests == 1 && NumCmps >= 3) ||
         (NumDests == 2 && NumCmps >= 5) ||
         (NumDests == 3 && NumCmps >= 6);
}

bool SelectionDAGBuilder::buildBitTests(CaseClusterVector &Clusters,
                                        unsigned First, unsigned Last,
                                        const SwitchInst *SI,
                                        CaseCluster &BTCluster) {
  assert(First <= Last);
  if (First == Last)
    return false;

  BitVector Dests(FuncInfo.MF->getNumBlockIDs());
  unsigned NumCmps = 0;
  for (int64_t I = First; I <= Last; ++I) {
    assert(Clusters[I].Kind == CC_Range);
    Dests.set(Clusters[I].MBB->getNumber());
    NumCmps += (Clusters[I].Low == Clusters[I].High) ? 1 : 2;
  }
  unsigned NumDests = Dests.count();

  APInt Low = Clusters[First].Low->getValue();
  APInt High = Clusters[Last].High->getValue();
  assert(Low.slt(High));

  if (!isSuitableForBitTests(NumDests, NumCmps, Low, High))
    return false;

  APInt LowBound;
  APInt CmpRange;

  const int BitWidth = DAG.getTargetLoweringInfo()
                           .getPointerTy(DAG.getDataLayout())
                           .getSizeInBits();
  assert(rangeFitsInWord(Low, High) && "Case range must fit in bit mask!");

  if (Low.isNonNegative() && High.slt(BitWidth)) {
    // Optimize the case where all the case values fit in a
    // word without having to subtract minValue. In this case,
    // we can optimize away the subtraction.
    LowBound = APInt::getNullValue(Low.getBitWidth());
    CmpRange = High;
  } else {
    LowBound = Low;
    CmpRange = High - Low;
  }

  CaseBitsVector CBV;
  uint32_t TotalWeight = 0;
  for (unsigned i = First; i <= Last; ++i) {
    // Find the CaseBits for this destination.
    unsigned j;
    for (j = 0; j < CBV.size(); ++j)
      if (CBV[j].BB == Clusters[i].MBB)
        break;
    if (j == CBV.size())
      CBV.push_back(CaseBits(0, Clusters[i].MBB, 0, 0));
    CaseBits *CB = &CBV[j];

    // Update Mask, Bits and ExtraWeight.
    uint64_t Lo = (Clusters[i].Low->getValue() - LowBound).getZExtValue();
    uint64_t Hi = (Clusters[i].High->getValue() - LowBound).getZExtValue();
    assert(Hi >= Lo && Hi < 64 && "Invalid bit case!");
    CB->Mask |= (-1ULL >> (63 - (Hi - Lo))) << Lo;
    CB->Bits += Hi - Lo + 1;
    CB->ExtraWeight += Clusters[i].Weight;
    TotalWeight += Clusters[i].Weight;
    assert(TotalWeight >= Clusters[i].Weight && "Weight overflow!");
  }

  BitTestInfo BTI;
  std::sort(CBV.begin(), CBV.end(), [](const CaseBits &a, const CaseBits &b) {
    // Sort by weight first, number of bits second.
    if (a.ExtraWeight != b.ExtraWeight)
      return a.ExtraWeight > b.ExtraWeight;
    return a.Bits > b.Bits;
  });

  for (auto &CB : CBV) {
    MachineBasicBlock *BitTestBB =
        FuncInfo.MF->CreateMachineBasicBlock(SI->getParent());
    BTI.push_back(BitTestCase(CB.Mask, BitTestBB, CB.BB, CB.ExtraWeight));
  }
  BitTestCases.emplace_back(std::move(LowBound), std::move(CmpRange),
                            SI->getCondition(), -1U, MVT::Other, false, nullptr,
                            nullptr, std::move(BTI));

  BTCluster = CaseCluster::bitTests(Clusters[First].Low, Clusters[Last].High,
                                    BitTestCases.size() - 1, TotalWeight);
  return true;
}

void SelectionDAGBuilder::findBitTestClusters(CaseClusterVector &Clusters,
                                              const SwitchInst *SI) {
// Partition Clusters into as few subsets as possible, where each subset has a
// range that fits in a machine word and has <= 3 unique destinations.

#ifndef NDEBUG
  // Clusters must be sorted and contain Range or JumpTable clusters.
  assert(!Clusters.empty());
  assert(Clusters[0].Kind == CC_Range || Clusters[0].Kind == CC_JumpTable);
  for (const CaseCluster &C : Clusters)
    assert(C.Kind == CC_Range || C.Kind == CC_JumpTable);
  for (unsigned i = 1; i < Clusters.size(); ++i)
    assert(Clusters[i-1].High->getValue().slt(Clusters[i].Low->getValue()));
#endif

  // The algorithm below is not suitable for -O0.
  if (TM.getOptLevel() == CodeGenOpt::None)
    return;

  // If target does not have legal shift left, do not emit bit tests at all.
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  EVT PTy = TLI.getPointerTy(DAG.getDataLayout());
  if (!TLI.isOperationLegal(ISD::SHL, PTy))
    return;

  int BitWidth = PTy.getSizeInBits();
  const int64_t N = Clusters.size();

  // MinPartitions[i] is the minimum nbr of partitions of Clusters[i..N-1].
  SmallVector<unsigned, 8> MinPartitions(N);
  // LastElement[i] is the last element of the partition starting at i.
  SmallVector<unsigned, 8> LastElement(N);

  // FIXME: This might not be the best algorithm for finding bit test clusters.

  // Base case: There is only one way to partition Clusters[N-1].
  MinPartitions[N - 1] = 1;
  LastElement[N - 1] = N - 1;

  // Note: loop indexes are signed to avoid underflow.
  for (int64_t i = N - 2; i >= 0; --i) {
    // Find optimal partitioning of Clusters[i..N-1].
    // Baseline: Put Clusters[i] into a partition on its own.
    MinPartitions[i] = MinPartitions[i + 1] + 1;
    LastElement[i] = i;

    // Search for a solution that results in fewer partitions.
    // Note: the search is limited by BitWidth, reducing time complexity.
    for (int64_t j = std::min(N - 1, i + BitWidth - 1); j > i; --j) {
      // Try building a partition from Clusters[i..j].

      // Check the range.
      if (!rangeFitsInWord(Clusters[i].Low->getValue(),
                           Clusters[j].High->getValue()))
        continue;

      // Check nbr of destinations and cluster types.
      // FIXME: This works, but doesn't seem very efficient.
      bool RangesOnly = true;
      BitVector Dests(FuncInfo.MF->getNumBlockIDs());
      for (int64_t k = i; k <= j; k++) {
        if (Clusters[k].Kind != CC_Range) {
          RangesOnly = false;
          break;
        }
        Dests.set(Clusters[k].MBB->getNumber());
      }
      if (!RangesOnly || Dests.count() > 3)
        break;

      // Check if it's a better partition.
      unsigned NumPartitions = 1 + (j == N - 1 ? 0 : MinPartitions[j + 1]);
      if (NumPartitions < MinPartitions[i]) {
        // Found a better partition.
        MinPartitions[i] = NumPartitions;
        LastElement[i] = j;
      }
    }
  }

  // Iterate over the partitions, replacing with bit-test clusters in-place.
  unsigned DstIndex = 0;
  for (unsigned First = 0, Last; First < N; First = Last + 1) {
    Last = LastElement[First];
    assert(First <= Last);
    assert(DstIndex <= First);

    CaseCluster BitTestCluster;
    if (buildBitTests(Clusters, First, Last, SI, BitTestCluster)) {
      Clusters[DstIndex++] = BitTestCluster;
    } else {
      size_t NumClusters = Last - First + 1;
      std::memmove(&Clusters[DstIndex], &Clusters[First],
                   sizeof(Clusters[0]) * NumClusters);
      DstIndex += NumClusters;
    }
  }
  Clusters.resize(DstIndex);
}

void SelectionDAGBuilder::lowerWorkItem(SwitchWorkListItem W, Value *Cond,
                                        MachineBasicBlock *SwitchMBB,
                                        MachineBasicBlock *DefaultMBB) {
  MachineFunction *CurMF = FuncInfo.MF;
  MachineBasicBlock *NextMBB = nullptr;
  MachineFunction::iterator BBI = W.MBB;
  if (++BBI != FuncInfo.MF->end())
    NextMBB = BBI;

  unsigned Size = W.LastCluster - W.FirstCluster + 1;

  BranchProbabilityInfo *BPI = FuncInfo.BPI;

  if (Size == 2 && W.MBB == SwitchMBB) {
    // If any two of the cases has the same destination, and if one value
    // is the same as the other, but has one bit unset that the other has set,
    // use bit manipulation to do two compares at once.  For example:
    // "if (X == 6 || X == 4)" -> "if ((X|2) == 6)"
    // TODO: This could be extended to merge any 2 cases in switches with 3
    // cases.
    // TODO: Handle cases where W.CaseBB != SwitchBB.
    CaseCluster &Small = *W.FirstCluster;
    CaseCluster &Big = *W.LastCluster;

    if (Small.Low == Small.High && Big.Low == Big.High &&
        Small.MBB == Big.MBB) {
      const APInt &SmallValue = Small.Low->getValue();
      const APInt &BigValue = Big.Low->getValue();

      // Check that there is only one bit different.
      APInt CommonBit = BigValue ^ SmallValue;
      if (CommonBit.isPowerOf2()) {
        SDValue CondLHS = getValue(Cond);
        EVT VT = CondLHS.getValueType();
        SDLoc DL = getCurSDLoc();

        SDValue Or = DAG.getNode(ISD::OR, DL, VT, CondLHS,
                                 DAG.getConstant(CommonBit, DL, VT));
        SDValue Cond = DAG.getSetCC(
            DL, MVT::i1, Or, DAG.getConstant(BigValue | SmallValue, DL, VT),
            ISD::SETEQ);

        // Update successor info.
        // Both Small and Big will jump to Small.BB, so we sum up the weights.
        addSuccessorWithWeight(SwitchMBB, Small.MBB, Small.Weight + Big.Weight);
        addSuccessorWithWeight(
            SwitchMBB, DefaultMBB,
            // The default destination is the first successor in IR.
            BPI ? BPI->getEdgeWeight(SwitchMBB->getBasicBlock(), (unsigned)0)
                : 0);

        // Insert the true branch.
        SDValue BrCond =
            DAG.getNode(ISD::BRCOND, DL, MVT::Other, getControlRoot(), Cond,
                        DAG.getBasicBlock(Small.MBB));
        // Insert the false branch.
        BrCond = DAG.getNode(ISD::BR, DL, MVT::Other, BrCond,
                             DAG.getBasicBlock(DefaultMBB));

        DAG.setRoot(BrCond);
        return;
      }
    }
  }

  if (TM.getOptLevel() != CodeGenOpt::None) {
    // Order cases by weight so the most likely case will be checked first.
    std::sort(W.FirstCluster, W.LastCluster + 1,
              [](const CaseCluster &a, const CaseCluster &b) {
      return a.Weight > b.Weight;
    });

    // Rearrange the case blocks so that the last one falls through if possible
    // without without changing the order of weights.
    for (CaseClusterIt I = W.LastCluster; I > W.FirstCluster; ) {
      --I;
      if (I->Weight > W.LastCluster->Weight)
        break;
      if (I->Kind == CC_Range && I->MBB == NextMBB) {
        std::swap(*I, *W.LastCluster);
        break;
      }
    }
  }

  // Compute total weight.
  uint32_t UnhandledWeights = 0;
  for (CaseClusterIt I = W.FirstCluster; I <= W.LastCluster; ++I) {
    UnhandledWeights += I->Weight;
    assert(UnhandledWeights >= I->Weight && "Weight overflow!");
  }

  MachineBasicBlock *CurMBB = W.MBB;
  for (CaseClusterIt I = W.FirstCluster, E = W.LastCluster; I <= E; ++I) {
    MachineBasicBlock *Fallthrough;
    if (I == W.LastCluster) {
      // For the last cluster, fall through to the default destination.
      Fallthrough = DefaultMBB;
    } else {
      Fallthrough = CurMF->CreateMachineBasicBlock(CurMBB->getBasicBlock());
      CurMF->insert(BBI, Fallthrough);
      // Put Cond in a virtual register to make it available from the new blocks.
      ExportFromCurrentBlock(Cond);
    }

    switch (I->Kind) {
      case CC_JumpTable: {
        // FIXME: Optimize away range check based on pivot comparisons.
        JumpTableHeader *JTH = &JTCases[I->JTCasesIndex].first;
        JumpTable *JT = &JTCases[I->JTCasesIndex].second;

        // The jump block hasn't been inserted yet; insert it here.
        MachineBasicBlock *JumpMBB = JT->MBB;
        CurMF->insert(BBI, JumpMBB);
        addSuccessorWithWeight(CurMBB, Fallthrough);
        addSuccessorWithWeight(CurMBB, JumpMBB);

        // The jump table header will be inserted in our current block, do the
        // range check, and fall through to our fallthrough block.
        JTH->HeaderBB = CurMBB;
        JT->Default = Fallthrough; // FIXME: Move Default to JumpTableHeader.

        // If we're in the right place, emit the jump table header right now.
        if (CurMBB == SwitchMBB) {
          visitJumpTableHeader(*JT, *JTH, SwitchMBB);
          JTH->Emitted = true;
        }
        break;
      }
      case CC_BitTests: {
        // FIXME: Optimize away range check based on pivot comparisons.
        BitTestBlock *BTB = &BitTestCases[I->BTCasesIndex];

        // The bit test blocks haven't been inserted yet; insert them here.
        for (BitTestCase &BTC : BTB->Cases)
          CurMF->insert(BBI, BTC.ThisBB);

        // Fill in fields of the BitTestBlock.
        BTB->Parent = CurMBB;
        BTB->Default = Fallthrough;

        // If we're in the right place, emit the bit test header header right now.
        if (CurMBB ==SwitchMBB) {
          visitBitTestHeader(*BTB, SwitchMBB);
          BTB->Emitted = true;
        }
        break;
      }
      case CC_Range: {
        const Value *RHS, *LHS, *MHS;
        ISD::CondCode CC;
        if (I->Low == I->High) {
          // Check Cond == I->Low.
          CC = ISD::SETEQ;
          LHS = Cond;
          RHS=I->Low;
          MHS = nullptr;
        } else {
          // Check I->Low <= Cond <= I->High.
          CC = ISD::SETLE;
          LHS = I->Low;
          MHS = Cond;
          RHS = I->High;
        }

        // The false weight is the sum of all unhandled cases.
        UnhandledWeights -= I->Weight;
        CaseBlock CB(CC, LHS, RHS, MHS, I->MBB, Fallthrough, CurMBB, I->Weight,
                     UnhandledWeights);

        if (CurMBB == SwitchMBB)
          visitSwitchCase(CB, SwitchMBB);
        else
          SwitchCases.push_back(CB);

        break;
      }
    }
    CurMBB = Fallthrough;
  }
}

unsigned SelectionDAGBuilder::caseClusterRank(const CaseCluster &CC,
                                              CaseClusterIt First,
                                              CaseClusterIt Last) {
  return std::count_if(First, Last + 1, [&](const CaseCluster &X) {
    if (X.Weight != CC.Weight)
      return X.Weight > CC.Weight;

    // Ties are broken by comparing the case value.
    return X.Low->getValue().slt(CC.Low->getValue());
  });
}

void SelectionDAGBuilder::splitWorkItem(SwitchWorkList &WorkList,
                                        const SwitchWorkListItem &W,
                                        Value *Cond,
                                        MachineBasicBlock *SwitchMBB) {
  assert(W.FirstCluster->Low->getValue().slt(W.LastCluster->Low->getValue()) &&
         "Clusters not sorted?");

  assert(W.LastCluster - W.FirstCluster + 1 >= 2 && "Too small to split!");

  // Balance the tree based on branch weights to create a near-optimal (in terms
  // of search time given key frequency) binary search tree. See e.g. Kurt
  // Mehlhorn "Nearly Optimal Binary Search Trees" (1975).
  CaseClusterIt LastLeft = W.FirstCluster;
  CaseClusterIt FirstRight = W.LastCluster;
  uint32_t LeftWeight = LastLeft->Weight;
  uint32_t RightWeight = FirstRight->Weight;

  // Move LastLeft and FirstRight towards each other from opposite directions to
  // find a partitioning of the clusters which balances the weight on both
  // sides. If LeftWeight and RightWeight are equal, alternate which side is
  // taken to ensure 0-weight nodes are distributed evenly.
  unsigned I = 0;
  while (LastLeft + 1 < FirstRight) {
    if (LeftWeight < RightWeight || (LeftWeight == RightWeight && (I & 1)))
      LeftWeight += (++LastLeft)->Weight;
    else
      RightWeight += (--FirstRight)->Weight;
    I++;
  }

  for (;;) {
    // Our binary search tree differs from a typical BST in that ours can have up
    // to three values in each leaf. The pivot selection above doesn't take that
    // into account, which means the tree might require more nodes and be less
    // efficient. We compensate for this here.

    unsigned NumLeft = LastLeft - W.FirstCluster + 1;
    unsigned NumRight = W.LastCluster - FirstRight + 1;

    if (std::min(NumLeft, NumRight) < 3 && std::max(NumLeft, NumRight) > 3) {
      // If one side has less than 3 clusters, and the other has more than 3,
      // consider taking a cluster from the other side.

      if (NumLeft < NumRight) {
        // Consider moving the first cluster on the right to the left side.
        CaseCluster &CC = *FirstRight;
        unsigned RightSideRank = caseClusterRank(CC, FirstRight, W.LastCluster);
        unsigned LeftSideRank = caseClusterRank(CC, W.FirstCluster, LastLeft);
        if (LeftSideRank <= RightSideRank) {
          // Moving the cluster to the left does not demote it.
          ++LastLeft;
          ++FirstRight;
          continue;
        }
      } else {
        assert(NumRight < NumLeft);
        // Consider moving the last element on the left to the right side.
        CaseCluster &CC = *LastLeft;
        unsigned LeftSideRank = caseClusterRank(CC, W.FirstCluster, LastLeft);
        unsigned RightSideRank = caseClusterRank(CC, FirstRight, W.LastCluster);
        if (RightSideRank <= LeftSideRank) {
          // Moving the cluster to the right does not demot it.
          --LastLeft;
          --FirstRight;
          continue;
        }
      }
    }
    break;
  }

  assert(LastLeft + 1 == FirstRight);
  assert(LastLeft >= W.FirstCluster);
  assert(FirstRight <= W.LastCluster);

  // Use the first element on the right as pivot since we will make less-than
  // comparisons against it.
  CaseClusterIt PivotCluster = FirstRight;
  assert(PivotCluster > W.FirstCluster);
  assert(PivotCluster <= W.LastCluster);

  CaseClusterIt FirstLeft = W.FirstCluster;
  CaseClusterIt LastRight = W.LastCluster;

  const ConstantInt *Pivot = PivotCluster->Low;

  // New blocks will be inserted immediately after the current one.
  MachineFunction::iterator BBI = W.MBB;
  ++BBI;

  // We will branch to the LHS if Value < Pivot. If LHS is a single cluster,
  // we can branch to its destination directly if it's squeezed exactly in
  // between the known lower bound and Pivot - 1.
  MachineBasicBlock *LeftMBB;
  if (FirstLeft == LastLeft && FirstLeft->Kind == CC_Range &&
      FirstLeft->Low == W.GE &&
      (FirstLeft->High->getValue() + 1LL) == Pivot->getValue()) {
    LeftMBB = FirstLeft->MBB;
  } else {
    LeftMBB = FuncInfo.MF->CreateMachineBasicBlock(W.MBB->getBasicBlock());
    FuncInfo.MF->insert(BBI, LeftMBB);
    WorkList.push_back({LeftMBB, FirstLeft, LastLeft, W.GE, Pivot});
    // Put Cond in a virtual register to make it available from the new blocks.
    ExportFromCurrentBlock(Cond);
  }

  // Similarly, we will branch to the RHS if Value >= Pivot. If RHS is a
  // single cluster, RHS.Low == Pivot, and we can branch to its destination
  // directly if RHS.High equals the current upper bound.
  MachineBasicBlock *RightMBB;
  if (FirstRight == LastRight && FirstRight->Kind == CC_Range &&
      W.LT && (FirstRight->High->getValue() + 1ULL) == W.LT->getValue()) {
    RightMBB = FirstRight->MBB;
  } else {
    RightMBB = FuncInfo.MF->CreateMachineBasicBlock(W.MBB->getBasicBlock());
    FuncInfo.MF->insert(BBI, RightMBB);
    WorkList.push_back({RightMBB, FirstRight, LastRight, Pivot, W.LT});
    // Put Cond in a virtual register to make it available from the new blocks.
    ExportFromCurrentBlock(Cond);
  }

  // Create the CaseBlock record that will be used to lower the branch.
  CaseBlock CB(ISD::SETLT, Cond, Pivot, nullptr, LeftMBB, RightMBB, W.MBB,
               LeftWeight, RightWeight);

  if (W.MBB == SwitchMBB)
    visitSwitchCase(CB, SwitchMBB);
  else
    SwitchCases.push_back(CB);
}

void SelectionDAGBuilder::visitSwitch(const SwitchInst &SI) {
  // Extract cases from the switch.
  BranchProbabilityInfo *BPI = FuncInfo.BPI;
  CaseClusterVector Clusters;
  Clusters.reserve(SI.getNumCases());
  for (auto I : SI.cases()) {
    MachineBasicBlock *Succ = FuncInfo.MBBMap[I.getCaseSuccessor()];
    const ConstantInt *CaseVal = I.getCaseValue();
    uint32_t Weight =
        BPI ? BPI->getEdgeWeight(SI.getParent(), I.getSuccessorIndex()) : 0;
    Clusters.push_back(CaseCluster::range(CaseVal, CaseVal, Succ, Weight));
  }

  MachineBasicBlock *DefaultMBB = FuncInfo.MBBMap[SI.getDefaultDest()];

  // Cluster adjacent cases with the same destination. We do this at all
  // optimization levels because it's cheap to do and will make codegen faster
  // if there are many clusters.
  sortAndRangeify(Clusters);

  if (TM.getOptLevel() != CodeGenOpt::None) {
    // Replace an unreachable default with the most popular destination.
    // FIXME: Exploit unreachable default more aggressively.
    bool UnreachableDefault =
        isa<UnreachableInst>(SI.getDefaultDest()->getFirstNonPHIOrDbg());
    if (UnreachableDefault && !Clusters.empty()) {
      DenseMap<const BasicBlock *, unsigned> Popularity;
      unsigned MaxPop = 0;
      const BasicBlock *MaxBB = nullptr;
      for (auto I : SI.cases()) {
        const BasicBlock *BB = I.getCaseSuccessor();
        if (++Popularity[BB] > MaxPop) {
          MaxPop = Popularity[BB];
          MaxBB = BB;
        }
      }
      // Set new default.
      assert(MaxPop > 0 && MaxBB);
      DefaultMBB = FuncInfo.MBBMap[MaxBB];

      // Remove cases that were pointing to the destination that is now the
      // default.
      CaseClusterVector New;
      New.reserve(Clusters.size());
      for (CaseCluster &CC : Clusters) {
        if (CC.MBB != DefaultMBB)
          New.push_back(CC);
      }
      Clusters = std::move(New);
    }
  }

  // If there is only the default destination, jump there directly.
  MachineBasicBlock *SwitchMBB = FuncInfo.MBB;
  if (Clusters.empty()) {
    SwitchMBB->addSuccessor(DefaultMBB);
    if (DefaultMBB != NextBlock(SwitchMBB)) {
      DAG.setRoot(DAG.getNode(ISD::BR, getCurSDLoc(), MVT::Other,
                              getControlRoot(), DAG.getBasicBlock(DefaultMBB)));
    }
    return;
  }

  findJumpTables(Clusters, &SI, DefaultMBB);
  findBitTestClusters(Clusters, &SI);

  DEBUG({
    dbgs() << "Case clusters: ";
    for (const CaseCluster &C : Clusters) {
      if (C.Kind == CC_JumpTable) dbgs() << "JT:";
      if (C.Kind == CC_BitTests) dbgs() << "BT:";

      C.Low->getValue().print(dbgs(), true);
      if (C.Low != C.High) {
        dbgs() << '-';
        C.High->getValue().print(dbgs(), true);
      }
      dbgs() << ' ';
    }
    dbgs() << '\n';
  });

  assert(!Clusters.empty());
  SwitchWorkList WorkList;
  CaseClusterIt First = Clusters.begin();
  CaseClusterIt Last = Clusters.end() - 1;
  WorkList.push_back({SwitchMBB, First, Last, nullptr, nullptr});

  while (!WorkList.empty()) {
    SwitchWorkListItem W = WorkList.back();
    WorkList.pop_back();
    unsigned NumClusters = W.LastCluster - W.FirstCluster + 1;

    if (NumClusters > 3 && TM.getOptLevel() != CodeGenOpt::None) {
      // For optimized builds, lower large range as a balanced binary tree.
      splitWorkItem(WorkList, W, SI.getCondition(), SwitchMBB);
      continue;
    }

    lowerWorkItem(W, SI.getCondition(), SwitchMBB, DefaultMBB);
  }
}
