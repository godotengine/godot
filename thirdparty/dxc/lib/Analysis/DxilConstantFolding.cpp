//===-- DxilConstantFolding.cpp - Fold dxil intrinsics into constants -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// Copyright (C) Microsoft Corporation. All rights reserved.
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//
#include "llvm/Analysis/DxilConstantFolding.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Config/config.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include <cerrno>
#include <cmath>
#include <algorithm>
#include <functional>

#include "dxc/DXIL/DXIL.h"
#include "dxc/HLSL/DxilConvergentName.h"
using namespace llvm;
using namespace hlsl;

namespace {

bool IsConvergentMarker(const Function *F) {
  return F->getName().startswith(kConvergentFunctionPrefix);
}

bool IsConvergentMarker(const char *Name) {
  StringRef RName = Name;
  return RName.startswith(kConvergentFunctionPrefix);
}

} // namespace

// Check if the given function is a dxil intrinsic and if so extract the
// opcode for the instrinsic being called.
static bool GetDxilOpcode(StringRef Name, ArrayRef<Constant *> Operands, OP::OpCode &out) {
  if (!OP::IsDxilOpFuncName(Name))
    return false;
  if (!Operands.size())
    return false;
  if (ConstantInt *ci = dyn_cast<ConstantInt>(Operands[0])) {
    uint64_t opcode = ci->getLimitedValue();
    if (opcode < static_cast<uint64_t>(OP::OpCode::NumOpCodes)) {
      out = static_cast<OP::OpCode>(opcode);
      return true;
    }
  }

  return false;
}

// Typedefs for passing function pointers to evaluate float constants.
typedef double(__cdecl *NativeFPUnaryOp)(double);
typedef std::function<APFloat::opStatus(APFloat&)> APFloatUnaryOp;

/// Currently APFloat versions of these functions do not exist, so we use
/// the host native double versions.  Float versions are not called
/// directly but for all these it is true (float)(f((double)arg)) ==
/// f(arg).  Long double not supported yet.
///
/// Calls out to the llvm constant folding function to do the real work.
static Constant *DxilConstantFoldFP(NativeFPUnaryOp NativeFP, ConstantFP *C, Type *Ty) {
  double V = llvm::getValueAsDouble(C);
  return llvm::ConstantFoldFP(NativeFP, V, Ty);
}

// Constant fold using the provided function on APFloats.
static Constant *HLSLConstantFoldAPFloat(APFloatUnaryOp NativeFP, ConstantFP *C, Type *Ty) {
  APFloat APF = C->getValueAPF();

  if (NativeFP(APF) != APFloat::opStatus::opOK)
    return nullptr;

  return ConstantFP::get(Ty->getContext(), APF);
}

// Constant fold a round dxil intrinsic.
static Constant *HLSLConstantFoldRound(APFloat::roundingMode roundingMode, ConstantFP *C, Type *Ty) {
  APFloatUnaryOp f = [roundingMode](APFloat &x) { return x.roundToIntegral(roundingMode); };
  return HLSLConstantFoldAPFloat(f, C, Ty);
}

namespace {
// Wrapper for call operands that "shifts past" the hlsl intrinsic opcode.
// Also provides accessors that dyn_cast the operand to a constant type.
class DxilIntrinsicOperands {
public:
  DxilIntrinsicOperands(ArrayRef<Constant *> RawCallOperands) : m_RawCallOperands(RawCallOperands) {}
  Constant * const &operator[](size_t index) const {
    return m_RawCallOperands[index + 1];
  }

  ConstantInt *GetConstantInt(size_t index) const {
    return dyn_cast<ConstantInt>(this->operator[](index));
  }
  
  ConstantFP *GetConstantFloat(size_t index) const {
    return dyn_cast<ConstantFP>(this->operator[](index));
  }

  size_t Size() const {
    return m_RawCallOperands.size() - 1;
  }
private:
  ArrayRef<Constant *> m_RawCallOperands;
};
}

/// We only fold functions with finite arguments. Folding NaN and inf is
/// likely to be aborted with an exception anyway, and some host libms
/// have known errors raising exceptions.
static bool IsFinite(ConstantFP *C) {
  if (C->getValueAPF().isNaN() || C->getValueAPF().isInfinity())
    return false;

  return true;
}

// Check that the op is non-null and finite.
static bool IsValidOp(ConstantFP *C) {
  if (!C || !IsFinite(C))
    return false;

  return true;
}

// Check that all ops are valid.
static bool AllValidOps(ArrayRef<ConstantFP *> Ops) {
  return std::all_of(Ops.begin(), Ops.end(), IsValidOp);
}

// Constant fold unary floating point intrinsics.
static Constant *ConstantFoldUnaryFPIntrinsic(OP::OpCode opcode, Type *Ty, ConstantFP *Op) {
  switch (opcode) {
  default: break;
  case OP::OpCode::FAbs: return DxilConstantFoldFP(fabs, Op, Ty);
  case OP::OpCode::Saturate: {
    NativeFPUnaryOp f = [](double x) { return std::max(std::min(x, 1.0), 0.0); };
    return DxilConstantFoldFP(f, Op, Ty);
  }
  case OP::OpCode::Cos:  return DxilConstantFoldFP(cos, Op, Ty);
  case OP::OpCode::Sin:  return DxilConstantFoldFP(sin, Op, Ty);
  case OP::OpCode::Tan:  return DxilConstantFoldFP(tan, Op, Ty);
  case OP::OpCode::Acos: return DxilConstantFoldFP(acos, Op, Ty);
  case OP::OpCode::Asin: return DxilConstantFoldFP(asin, Op, Ty);
  case OP::OpCode::Atan: return DxilConstantFoldFP(atan, Op, Ty);
  case OP::OpCode::Hcos: return DxilConstantFoldFP(cosh, Op, Ty);
  case OP::OpCode::Hsin: return DxilConstantFoldFP(sinh, Op, Ty);
  case OP::OpCode::Htan: return DxilConstantFoldFP(tanh, Op, Ty);
  case OP::OpCode::Exp:  return DxilConstantFoldFP(exp2, Op, Ty);
  case OP::OpCode::Frc: {
    NativeFPUnaryOp f = [](double x) { double unused; return fabs(modf(x, &unused)); };
    return DxilConstantFoldFP(f, Op, Ty);
  }
  case OP::OpCode::Log: return DxilConstantFoldFP(log2, Op, Ty);
  case OP::OpCode::Sqrt: return DxilConstantFoldFP(sqrt, Op, Ty);
  case OP::OpCode::Rsqrt: {
    NativeFPUnaryOp f = [](double x) { return 1.0 / sqrt(x); };
    return DxilConstantFoldFP(f, Op, Ty);
  }
  case OP::OpCode::Round_ne: return HLSLConstantFoldRound(APFloat::roundingMode::rmNearestTiesToEven, Op, Ty);
  case OP::OpCode::Round_ni: return HLSLConstantFoldRound(APFloat::roundingMode::rmTowardNegative, Op, Ty);
  case OP::OpCode::Round_pi: return HLSLConstantFoldRound(APFloat::roundingMode::rmTowardPositive, Op, Ty);
  case OP::OpCode::Round_z: return HLSLConstantFoldRound(APFloat::roundingMode::rmTowardZero, Op, Ty);
  }
  
  return nullptr;
}

// Constant fold binary floating point intrinsics.
static Constant *ConstantFoldBinaryFPIntrinsic(OP::OpCode opcode, Type *Ty, ConstantFP *Op1, ConstantFP *Op2) {
  const APFloat &C1 = Op1->getValueAPF();
  const APFloat &C2 = Op2->getValueAPF();
  switch (opcode) {
  default: break;
  case OP::OpCode::FMax: return ConstantFP::get(Ty->getContext(), maxnum(C1, C2));
  case OP::OpCode::FMin: return ConstantFP::get(Ty->getContext(), minnum(C1, C2));
  }

  return nullptr;
}

// Constant fold ternary floating point intrinsics.
static Constant *ConstantFoldTernaryFPIntrinsic(OP::OpCode opcode, Type *Ty, ConstantFP *Op1, ConstantFP *Op2, ConstantFP *Op3) {
  const APFloat &C1 = Op1->getValueAPF();
  const APFloat &C2 = Op2->getValueAPF();
  const APFloat &C3 = Op3->getValueAPF();
  APFloat::roundingMode roundingMode = APFloat::rmNearestTiesToEven;
  switch (opcode) {
  default: break;
  case OP::OpCode::FMad: {
    APFloat result(C1);
    result.multiply(C2, roundingMode);
    result.add(C3, roundingMode);
    return ConstantFP::get(Ty->getContext(), result);
  }
  case OP::OpCode::Fma: {
    APFloat result(C1);
    result.fusedMultiplyAdd(C2, C3, roundingMode);
    return ConstantFP::get(Ty->getContext(), result);
  }
  }
  return nullptr;
}

// Compute dot product for arbitrary sized vectors.
static Constant *ComputeDot(Type *Ty, ArrayRef<ConstantFP *> A, ArrayRef<ConstantFP *> B) {
  if (A.size() != B.size() || !A.size()) {
    assert(false && "invalid call to compute dot");
    return nullptr;
  }

  if (!AllValidOps(A) || !AllValidOps(B))
    return nullptr;
  
  APFloat::roundingMode roundingMode = APFloat::roundingMode::rmNearestTiesToEven;
  APFloat sum = APFloat::getZero(A[0]->getValueAPF().getSemantics());
  for (int i = 0, e = A.size(); i != e; ++i) {
    APFloat  val(A[i]->getValueAPF());
    val.multiply(B[i]->getValueAPF(), roundingMode);
    sum.add(val, roundingMode);
  }

  return ConstantFP::get(Ty->getContext(), sum);

}

// Constant folding for dot2, dot3, and dot4.
static Constant *ConstantFoldDot(OP::OpCode opcode, Type *Ty, const DxilIntrinsicOperands &operands) {
  switch (opcode) {
  default: break;
  case OP::OpCode::Dot2: {
    ConstantFP *Ax = operands.GetConstantFloat(0);
    ConstantFP *Ay = operands.GetConstantFloat(1);
    ConstantFP *Bx = operands.GetConstantFloat(2);
    ConstantFP *By = operands.GetConstantFloat(3);
    return ComputeDot(Ty, { Ax, Ay }, { Bx, By });
  }
  case OP::OpCode::Dot3: {
    ConstantFP *Ax = operands.GetConstantFloat(0);
    ConstantFP *Ay = operands.GetConstantFloat(1);
    ConstantFP *Az = operands.GetConstantFloat(2);
    ConstantFP *Bx = operands.GetConstantFloat(3);
    ConstantFP *By = operands.GetConstantFloat(4);
    ConstantFP *Bz = operands.GetConstantFloat(5);
    return ComputeDot(Ty, { Ax, Ay, Az }, { Bx, By, Bz });
  }
  case OP::OpCode::Dot4: {
    ConstantFP *Ax = operands.GetConstantFloat(0);
    ConstantFP *Ay = operands.GetConstantFloat(1);
    ConstantFP *Az = operands.GetConstantFloat(2);
    ConstantFP *Aw = operands.GetConstantFloat(3);
    ConstantFP *Bx = operands.GetConstantFloat(4);
    ConstantFP *By = operands.GetConstantFloat(5);
    ConstantFP *Bz = operands.GetConstantFloat(6);
    ConstantFP *Bw = operands.GetConstantFloat(7);
    return ComputeDot(Ty, { Ax, Ay, Az, Aw }, { Bx, By, Bz, Bw });
  }
  }

  return nullptr;
}

// Constant fold a Bfrev dxil intrinsic.
static Constant *HLSLConstantFoldBfrev(ConstantInt *C, Type *Ty) {
  APInt API = C->getValue();

  uint64_t result = 0;
  if (Ty == Type::getInt32Ty(Ty->getContext())) {
    uint32_t val = static_cast<uint32_t>(API.getLimitedValue());
    result = llvm::reverseBits(val);
  }
  else if (Ty == Type::getInt16Ty(Ty->getContext())) {
    uint16_t val = static_cast<uint16_t>(API.getLimitedValue());
    result = llvm::reverseBits(val);
  }
  else if (Ty == Type::getInt64Ty(Ty->getContext())) {
    uint64_t val = static_cast<uint64_t>(API.getLimitedValue());
    result = llvm::reverseBits(val);
  }
  else {
    return nullptr;
  }
  return ConstantInt::get(Ty, result);
}

// Handle special case for findfirst* bit functions.
// When the position is equal to the bitwidth the value was not found
// and we need to return a result of -1.
static Constant *HLSLConstantFoldFindBit(Type *Ty, unsigned position, unsigned bitwidth) {
  if (position == bitwidth)
    return ConstantInt::get(Ty, APInt::getAllOnesValue(Ty->getScalarSizeInBits()));

  return ConstantInt::get(Ty, position);
}

// Constant fold unary integer intrinsics.
static Constant *ConstantFoldUnaryIntIntrinsic(OP::OpCode opcode, Type *Ty, ConstantInt *Op) {
  APInt API = Op->getValue();
  switch (opcode) {
  default: break;
  case OP::OpCode::Bfrev:      return HLSLConstantFoldBfrev(Op, Ty);
  case OP::OpCode::Countbits:  return ConstantInt::get(Ty, API.countPopulation());
  case OP::OpCode::FirstbitLo: return HLSLConstantFoldFindBit(Ty, API.countTrailingZeros(), API.getBitWidth());
  case OP::OpCode::FirstbitHi: return HLSLConstantFoldFindBit(Ty, API.countLeadingZeros(), API.getBitWidth());
  case OP::OpCode::FirstbitSHi: {
    if (API.isNegative())
      return HLSLConstantFoldFindBit(Ty, API.countLeadingOnes(), API.getBitWidth());
    else
      return HLSLConstantFoldFindBit(Ty, API.countLeadingZeros(), API.getBitWidth());
  }
  }
  
  return nullptr;
}

// Constant fold binary integer intrinsics.
static Constant *ConstantFoldBinaryIntIntrinsic(OP::OpCode opcode, Type *Ty, ConstantInt *Op1, ConstantInt *Op2) {
  APInt C1 = Op1->getValue();
  APInt C2 = Op2->getValue();
  switch (opcode) {
  default: break;
  case OP::OpCode::IMin: {
    APInt minVal = C1.slt(C2) ? C1 : C2;
    return ConstantInt::get(Ty, minVal);
  }
  case OP::OpCode::IMax: {
    APInt maxVal = C1.sgt(C2) ? C1 : C2;
    return ConstantInt::get(Ty, maxVal);
  }
  case OP::OpCode::UMin: {
    APInt minVal = C1.ult(C2) ? C1 : C2;
    return ConstantInt::get(Ty, minVal);
  }
  case OP::OpCode::UMax: {
    APInt maxVal = C1.ugt(C2) ? C1 : C2;
    return ConstantInt::get(Ty, maxVal);
  }
  }

  return nullptr;
}

// Constant fold MakeDouble
static Constant *ConstantFoldMakeDouble(Type *Ty, const DxilIntrinsicOperands &IntrinsicOperands) {
  assert(IntrinsicOperands.Size() == 2);
  ConstantInt *Op1 = IntrinsicOperands.GetConstantInt(0);
  ConstantInt *Op2 = IntrinsicOperands.GetConstantInt(1);
  if (!Op1 || !Op2)
    return nullptr;
  uint64_t C1 = Op1->getZExtValue();
  uint64_t C2 = Op2->getZExtValue();
  uint64_t dbits = C2 << 32 | C1;
  double dval = *(double*)&dbits;
  return ConstantFP::get(Ty, dval);
}

// Compute bit field extract for ibfe and ubfe.
// The comptuation for ibfe and ubfe is the same except for the right shift,
// which is an arithemetic shift for ibfe and logical shift for ubfe.
// ubfe: https://msdn.microsoft.com/en-us/library/windows/desktop/hh447243(v=vs.85).aspx
// ibfe: https://msdn.microsoft.com/en-us/library/windows/desktop/hh447243(v=vs.85).aspx
static Constant *ComputeBFE(Type *Ty, APInt width, APInt offset, APInt val, std::function<APInt(APInt, APInt)> shr) {
    const APInt bitwidth(width.getBitWidth(), width.getBitWidth());
	// Limit width and offset to the bitwidth of the value.
    width  = width.And(bitwidth-1); 
    offset = offset.And(bitwidth-1);
    
    if (width == 0) {
      return ConstantInt::get(Ty, 0);
    }
    else if ((width + offset).ult(bitwidth)) {
      APInt dest = val.shl(bitwidth - (width + offset));
      dest = shr(dest, bitwidth - width);
      return ConstantInt::get(Ty, dest);
    }
    else {
      APInt dest = shr(val, offset);
      return ConstantInt::get(Ty, dest);
    }
}

// Constant fold ternary integer intrinsic.
static Constant *ConstantFoldTernaryIntIntrinsic(OP::OpCode opcode, Type *Ty, ConstantInt *Op1, ConstantInt *Op2, ConstantInt *Op3) {
  APInt C1 = Op1->getValue();
  APInt C2 = Op2->getValue();
  APInt C3 = Op3->getValue();
  switch (opcode) {
  default: break;
  case OP::OpCode::IMad:
  case OP::OpCode::UMad: {
    // Result is same for signed/unsigned since this is twos complement and we only
    // keep the lower half of the multiply.
    APInt result = C1 * C2 + C3;
    return ConstantInt::get(Ty, result);
  }
  case OP::OpCode::Ubfe: return ComputeBFE(Ty, C1, C2, C3, [](APInt val, APInt amt) {return val.lshr(amt); });
  case OP::OpCode::Ibfe: return ComputeBFE(Ty, C1, C2, C3, [](APInt val, APInt amt) {return val.ashr(amt); });
  }

  return nullptr;
}

// Constant fold quaternary integer intrinsic.
//
// Currently we only have one quaternary intrinsic: Bfi.
// The Bfi computaion is described here:
// https://msdn.microsoft.com/en-us/library/windows/desktop/hh446837(v=vs.85).aspx
static Constant *ConstantFoldQuaternaryIntInstrinsic(OP::OpCode opcode, Type *Ty, ConstantInt *Op1, ConstantInt *Op2, ConstantInt *Op3, ConstantInt *Op4) {
  if (opcode != OP::OpCode::Bfi)
    return nullptr;

  APInt bitwidth(Op1->getValue().getBitWidth(), Op1->getValue().getBitWidth());
  APInt width  = Op1->getValue().And(bitwidth-1);
  APInt offset = Op2->getValue().And(bitwidth-1);
  APInt src = Op3->getValue();
  APInt dst = Op4->getValue();
  APInt one(bitwidth.getBitWidth(), 1);
  APInt allOnes = APInt::getAllOnesValue(bitwidth.getBitWidth());

  // bitmask = (((1 << width)-1) << offset) & 0xffffffff
  // dest = ((src2 << offset) & bitmask) | (src3 & ~bitmask)
  APInt bitmask = (one.shl(width) - 1).shl(offset).And(allOnes);
  APInt result = (src.shl(offset).And(bitmask)).Or(dst.And(~bitmask));

  return ConstantInt::get(Ty, result);
}

// Top level function to constant fold floating point intrinsics.
static Constant *ConstantFoldFPIntrinsic(OP::OpCode opcode, Type *Ty, const DxilIntrinsicOperands &IntrinsicOperands) {
  if (!Ty->isHalfTy() && !Ty->isFloatTy() && !Ty->isDoubleTy())
    return nullptr;

  OP::OpCodeClass opClass = OP::GetOpCodeClass(opcode);

  switch (opClass) {
  default: break;
  case OP::OpCodeClass::Unary: {
    assert(IntrinsicOperands.Size() == 1);
    ConstantFP *Op = IntrinsicOperands.GetConstantFloat(0);

    if (!IsValidOp(Op))
      return nullptr;

    return ConstantFoldUnaryFPIntrinsic(opcode, Ty, Op);
  }
  case OP::OpCodeClass::Binary: {
    assert(IntrinsicOperands.Size() == 2);
    ConstantFP *Op1 = IntrinsicOperands.GetConstantFloat(0);
    ConstantFP *Op2 = IntrinsicOperands.GetConstantFloat(1);

    if (!IsValidOp(Op1) || !IsValidOp(Op2))
      return nullptr;

    return ConstantFoldBinaryFPIntrinsic(opcode, Ty, Op1, Op2);
  }
  case OP::OpCodeClass::Tertiary: {
    assert(IntrinsicOperands.Size() == 3);
    ConstantFP *Op1 = IntrinsicOperands.GetConstantFloat(0);
    ConstantFP *Op2 = IntrinsicOperands.GetConstantFloat(1);
    ConstantFP *Op3 = IntrinsicOperands.GetConstantFloat(2);

    if (!IsValidOp(Op1) || !IsValidOp(Op2) || !IsValidOp(Op3))
      return nullptr;

    return ConstantFoldTernaryFPIntrinsic(opcode, Ty, Op1, Op2, Op3);
  }
  case OP::OpCodeClass::Dot2:
  case OP::OpCodeClass::Dot3:
  case OP::OpCodeClass::Dot4:
    return ConstantFoldDot(opcode, Ty, IntrinsicOperands);
  case OP::OpCodeClass::MakeDouble:
    return ConstantFoldMakeDouble(Ty, IntrinsicOperands);
  }

  return nullptr;
}

// Top level function to constant fold integer intrinsics.
static Constant *ConstantFoldIntIntrinsic(OP::OpCode opcode, Type *Ty, const DxilIntrinsicOperands &IntrinsicOperands) {
  if (Ty->getScalarSizeInBits() > (sizeof(int64_t) * CHAR_BIT))
    return nullptr;

  OP::OpCodeClass opClass = OP::GetOpCodeClass(opcode);

  switch (opClass) {
  default: break;
  case OP::OpCodeClass::Unary:
  case OP::OpCodeClass::UnaryBits: {
    assert(IntrinsicOperands.Size() == 1);
    ConstantInt *Op = IntrinsicOperands.GetConstantInt(0);
    if (!Op)
      return nullptr;

    return ConstantFoldUnaryIntIntrinsic(opcode, Ty, Op);
  }
  case OP::OpCodeClass::Binary: {
    assert(IntrinsicOperands.Size() == 2);
    ConstantInt *Op1 = IntrinsicOperands.GetConstantInt(0);
    ConstantInt *Op2 = IntrinsicOperands.GetConstantInt(1);
    if (!Op1 || !Op2)
      return nullptr;
    
    return ConstantFoldBinaryIntIntrinsic(opcode, Ty, Op1, Op2);
  }
  case OP::OpCodeClass::Tertiary: {
    assert(IntrinsicOperands.Size() == 3);
    ConstantInt *Op1 = IntrinsicOperands.GetConstantInt(0);
    ConstantInt *Op2 = IntrinsicOperands.GetConstantInt(1);
    ConstantInt *Op3 = IntrinsicOperands.GetConstantInt(2);
    if (!Op1 || !Op2 || !Op3)
      return nullptr;
    
    return ConstantFoldTernaryIntIntrinsic(opcode, Ty, Op1, Op2, Op3);
  }
  case OP::OpCodeClass::Quaternary: {
    assert(IntrinsicOperands.Size() == 4);
    ConstantInt *Op1 = IntrinsicOperands.GetConstantInt(0);
    ConstantInt *Op2 = IntrinsicOperands.GetConstantInt(1);
    ConstantInt *Op3 = IntrinsicOperands.GetConstantInt(2);
    ConstantInt *Op4 = IntrinsicOperands.GetConstantInt(3);
    if (!Op1 || !Op2 || !Op3 || !Op4)
      return nullptr;

    return ConstantFoldQuaternaryIntInstrinsic(opcode, Ty, Op1, Op2, Op3, Op4);
  }
  case OP::OpCodeClass::IsHelperLane:
    return ConstantInt::get(Ty, (uint64_t)0);
  }

  return nullptr;
}

// External entry point to constant fold dxil intrinsics.
// Called from the llvm constant folding routine.
Constant *hlsl::ConstantFoldScalarCall(StringRef Name, Type *Ty, ArrayRef<Constant *> RawOperands) {
  OP::OpCode opcode;
  if (GetDxilOpcode(Name, RawOperands, opcode)) {
    DxilIntrinsicOperands IntrinsicOperands(RawOperands);

    if (Ty->isFloatingPointTy()) {
      return ConstantFoldFPIntrinsic(opcode, Ty, IntrinsicOperands);
    }
    else if (Ty->isIntegerTy()) {
      return ConstantFoldIntIntrinsic(opcode, Ty, IntrinsicOperands);
    }
  } else if (IsConvergentMarker(Name.data())) {
    assert(RawOperands.size() == 1);
    if (ConstantInt *C = dyn_cast<ConstantInt>(RawOperands[0]))
      return C;
    if (ConstantFP *C = dyn_cast<ConstantFP>(RawOperands[0]))
      return C;
  }

  return hlsl::ConstantFoldScalarCallExt(Name, Ty, RawOperands);
}

// External entry point to determine if we can constant fold calls to
// the given function. We have to overestimate the set of functions because
// we only have the function value here instead of the call. We need the
// actual call to get the opcode for the intrinsic.
bool hlsl::CanConstantFoldCallTo(const Function *F) {
  // Only constant fold dxil functions when we have a valid dxil module.
  if (!F->getParent()->HasDxilModule()) {
    assert(!OP::IsDxilOpFunc(F) && "dx.op function with no dxil module?");
    return false;
  }
  if (IsConvergentMarker(F))
    return true;
  // Lookup opcode class in dxil module. Set default value to invalid class.
  OP::OpCodeClass opClass = OP::OpCodeClass::NumOpClasses;
  const bool found = F->getParent()->GetDxilModule().GetOP()->GetOpCodeClass(F, opClass);

  // Return true for those dxil operation classes we can constant fold.
  if (found) {
    switch (opClass) {
    default: break;
    case OP::OpCodeClass::Unary:
    case OP::OpCodeClass::UnaryBits:
    case OP::OpCodeClass::Binary:
    case OP::OpCodeClass::Tertiary:
    case OP::OpCodeClass::Quaternary:
    case OP::OpCodeClass::Dot2:
    case OP::OpCodeClass::Dot3:
    case OP::OpCodeClass::Dot4:
    case OP::OpCodeClass::MakeDouble:
      return true;
    case OP::OpCodeClass::IsHelperLane: {
      const hlsl::ShaderModel *pSM =
          F->getParent()->GetDxilModule().GetShaderModel();
      return !pSM->IsPS() && !pSM->IsLib();
    }
    }
  }

  return hlsl::CanConstantFoldCallToExt(F);
}
