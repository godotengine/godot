///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilExpandTrigIntrinsics.cpp                                              //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Expand trigonmetric intrinsics to a sequence of dxil instructions.        //
// ========================================================================= //
//
// We provide expansions to approximate several trigonmetric functions that
// typically do not have native instructions in hardware. The details of each
// expansion is given below, but typically the exansion occurs in three steps
// 
//     1. Perform range reduction (if necessary) to reduce input range
//        to a value that works with the approximation.
//     2. Compute an approximation to the function (typically by evaluating 
//        a polynomial).
//     3. Perform range expansion (if necessary) to map the result back to
//        the original range.
// 
// For example, say we are expanding f(x) using an approximation to f, call it
// f*(x). And assume that f* only works for positive inputs, but we know that
// f(-x) = -f(x).Then the expansion would be
// 
//     1. a = abs(x)
//     2. v = f*(a)
//     3. e = x < 0 ? -v : v
// 
// where e contains the final expanded result.
// 
// References
// ---------------------------------------------------------------------------
// [HMF] Handbook of Mathematical Formulas by Abramowitz and Stegun, 1964
// [ADC] Approximations for Digital Computers by Hastings, 1955
// [WIK] Wikipedia, 2017
// 
// The approximation functions mostly come from [ADC]. The approximations
// are also referenced in [HMF], but they give original credit to [ADC].
// 
///////////////////////////////////////////////////////////////////////////////

#include "dxc/HLSL/DxilGenerationPass.h"
#include "dxc/DXIL/DxilOperations.h"
#include "dxc/DXIL/DxilSignatureElement.h"
#include "dxc/DXIL/DxilModule.h"
#include "dxc/Support/Global.h"
#include "dxc/DXIL/DxilInstructions.h"

#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/ADT/MapVector.h"

#include <cmath>
#include <utility>

using namespace llvm;
using namespace hlsl;

namespace {
class DxilExpandTrigIntrinsics : public FunctionPass {
private:

public:
  static char ID; // Pass identification, replacement for typeid
  explicit DxilExpandTrigIntrinsics() : FunctionPass(ID) {}

  StringRef getPassName() const override {
    return "DXIL expand trig intrinsics";
  }
  
  bool runOnFunction(Function &F) override;
  

private:
  typedef std::vector<CallInst *> IntrinsicList;
  IntrinsicList findTrigFunctionsToExpand(Function &F);
  CallInst *isExpandableTrigIntrinsicCall(Instruction *I);
  bool expandTrigIntrinsics(DxilModule &DM, const IntrinsicList &worklist);
  FastMathFlags getFastMathFlagsForIntrinsic(CallInst *intrinsic);
  void prepareBuilderToExpandIntrinsic(IRBuilder<> &builder, CallInst *intrinsic);

  // Expansion implementations.
  Value *expandACos(IRBuilder<> &builder, DxilInst_Acos acos, DxilModule &DM);
  Value *expandASin(IRBuilder<> &builder, DxilInst_Asin asin, DxilModule &DM);
  Value *expandATan(IRBuilder<> &builder, DxilInst_Atan atan, DxilModule &DM);
  Value *expandHCos(IRBuilder<> &builder, DxilInst_Hcos hcos, DxilModule &DM);
  Value *expandHSin(IRBuilder<> &builder, DxilInst_Hsin hsin, DxilModule &DM);
  Value *expandHTan(IRBuilder<> &builder, DxilInst_Htan htan, DxilModule &DM);
  Value *expandTan(IRBuilder<> &builder, DxilInst_Tan tan, DxilModule &DM);
};

// Math constants.
// Values taken from https://msdn.microsoft.com/en-us/library/4hwaceh6.aspx.
// Replicated here because they are not part of standard C++.
namespace math {
  constexpr double PI    = 3.14159265358979323846;
  constexpr double PI_2  = 1.57079632679489661923;
  constexpr double LOG2E = 1.44269504088896340736;
}

}


bool DxilExpandTrigIntrinsics::runOnFunction(Function &F) {
  DxilModule &DM = F.getParent()->GetOrCreateDxilModule(); 
  IntrinsicList intrinsics = findTrigFunctionsToExpand(F);
  const bool changed = expandTrigIntrinsics(DM, intrinsics);
  return changed;
}

CallInst *DxilExpandTrigIntrinsics::isExpandableTrigIntrinsicCall(Instruction *I) {
    if (OP::IsDxilOpFuncCallInst(I)) {
      switch (OP::GetDxilOpFuncCallInst(I)) {
      case OP::OpCode::Acos:
      case OP::OpCode::Asin:
      case OP::OpCode::Atan:
      case OP::OpCode::Hcos:
      case OP::OpCode::Hsin:
      case OP::OpCode::Htan:
      case OP::OpCode::Tan:
        return cast<CallInst>(I);
      default: break;
      }
    }
    return nullptr;
}

DxilExpandTrigIntrinsics::IntrinsicList DxilExpandTrigIntrinsics::findTrigFunctionsToExpand(Function &F) {
  IntrinsicList worklist;
  for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I)
    if (CallInst *call = isExpandableTrigIntrinsicCall(&*I))
      worklist.push_back(call);

  return worklist;
}

static bool isPreciseBuilder(IRBuilder<> &builder) {
  return !builder.getFastMathFlags().any();
}

static void setPreciseBuilder(IRBuilder<> &builder, bool precise) {
  FastMathFlags flags;
  if (precise)
    flags.clear();
  else
    flags.setUnsafeAlgebra();
  builder.SetFastMathFlags(flags);
}

void DxilExpandTrigIntrinsics::prepareBuilderToExpandIntrinsic(IRBuilder<> &builder, CallInst *intrinsic) {
  DxilModule &DM = intrinsic->getModule()->GetOrCreateDxilModule();
  builder.SetInsertPoint(intrinsic);
  setPreciseBuilder(builder, DM.IsPrecise(intrinsic));
}
  
bool DxilExpandTrigIntrinsics::expandTrigIntrinsics(DxilModule &DM, const IntrinsicList &worklist) {
  IRBuilder<> builder(DM.GetCtx());
  for (CallInst *intrinsic: worklist) {
    Value *expansion = nullptr;
    prepareBuilderToExpandIntrinsic(builder, intrinsic);
    
    OP::OpCode opcode = OP::GetDxilOpFuncCallInst(intrinsic);
    switch (opcode) {
    case OP::OpCode::Acos: expansion = expandACos(builder, intrinsic, DM); break;
    case OP::OpCode::Asin: expansion = expandASin(builder, intrinsic, DM); break;
    case OP::OpCode::Atan: expansion = expandATan(builder, intrinsic, DM); break;
    case OP::OpCode::Hcos: expansion = expandHCos(builder, intrinsic, DM); break;
    case OP::OpCode::Hsin: expansion = expandHSin(builder, intrinsic, DM); break;
    case OP::OpCode::Htan: expansion = expandHTan(builder, intrinsic, DM); break;
    case OP::OpCode::Tan: expansion = expandTan(builder, intrinsic, DM); break;
    default:
      assert(false && "unexpected intrinsic");
      break;
    }

    assert(expansion);
    intrinsic->replaceAllUsesWith(expansion);
    intrinsic->eraseFromParent();
  }

  return !worklist.empty();
}

// Helper
// return dx.op.UnaryFloat(X)
//
static Value *emitUnaryFloat(IRBuilder<> &builder, Value *X, OP *dxOp, OP::OpCode opcode, StringRef name) {
  Function *F = dxOp->GetOpFunc(opcode, X->getType());
  Value *Args[] = { dxOp->GetI32Const(static_cast<int>(opcode)), X };
  CallInst *Call = builder.CreateCall(F, Args, name);

  if (isPreciseBuilder(builder))
    DxilMDHelper::MarkPrecise(Call);
  return Call;
}

// Helper
// return dx.op.Fabs(X)
//
static Value *emitFAbs(IRBuilder<> &builder, Value *X, OP *dxOp, StringRef name) {
  return emitUnaryFloat(builder, X, dxOp, OP::OpCode::FAbs, name);
}

// Helper
// return dx.op.Sqrt(X)
//
static Value *emitSqrt(IRBuilder<> &builder, Value *X, OP *dxOp, StringRef name) {
  return emitUnaryFloat(builder, X, dxOp, OP::OpCode::Sqrt, name);
}

// Helper
// return sqrt(1 - X) * psi*(X)
//
// We compute the polynomial using Horners method to evaluate it efficently.
//
// psi*(X) = a0 + a1x + a2x^2 + a3x^3
//         = a0 + x(a1 + a2x + a3x^2)
//         = a0 + x(a1 + x(a2 + a3x))
//
static Value *emitSqrt1mXtimesPsiX(IRBuilder<> &builder, Value *X, OP *dxOp, StringRef name) {
  Value *One = ConstantFP::get(X->getType(), 1.0);
  Value *a0 = ConstantFP::get(X->getType(),  1.5707288);
  Value *a1 = ConstantFP::get(X->getType(), -0.2121144);
  Value *a2 = ConstantFP::get(X->getType(),  0.0742610);
  Value *a3 = ConstantFP::get(X->getType(), -0.0187293);


  // sqrt(1-x)
  Value *r1 = builder.CreateFSub(One, X, name);
  Value *r2 = emitSqrt(builder, r1, dxOp, name);

  // psi*(x)
  Value *r3 = builder.CreateFMul(X,  a3, name);
         r3 = builder.CreateFAdd(r3, a2, name);
         r3 = builder.CreateFMul(X,  r3, name);
         r3 = builder.CreateFAdd(r3, a1, name);
         r3 = builder.CreateFMul(X,  r3, name);
         r3 = builder.CreateFAdd(r3, a0, name);

  // sqrt(1-x) * psi*(x)
  Value *r4 = builder.CreateFMul(r2, r3,  name);
  return r4;
}

// Helper
// return e^x, e^-x
//
// We can use the dxil Exp function to compute the exponential. The only slight
// wrinkle is that in dxil Exp(x) = 2^x and we need e^x. Luckily we can easily
// change the base of the exponent using the following identity [HFM(p69)]
//
//  e^x = 2^{x * log_2(e)}
//
static std::pair<Value *, Value *> emitExEmx(IRBuilder<> &builder, Value *X, OP *dxOp, StringRef name) {
  Value *Zero  = ConstantFP::get(X->getType(), 0.0);
  Value *Log2e = ConstantFP::get(X->getType(), math::LOG2E);

  Value *r0 = builder.CreateFMul(X, Log2e, name);
  Value *r1 = emitUnaryFloat(builder, r0, dxOp, OP::OpCode::Exp, name);
  Value *r2 = builder.CreateFSub(Zero, r0, name);
  Value *r3 = emitUnaryFloat(builder, r2, dxOp, OP::OpCode::Exp, name);

  return std::make_pair(r1, r3);
}

// Asin
// ----------------------------------------------------------------------------
// Function
//    arcsin X = pi/2  - sqrt(1 - X) * psi(X)
//
// Range
//    0 <= X <= 1
//
// Approximation
//    Psi*(X) = a0 + a1x + a2x^2 + a3x^3
//      a0 =  1.5707288
//      a1 = -0.2121144
//      a2 =  0.0742610
//      a3 = -0.0187293
// 
// The domain of the approximation is 0 <=x <= 1, but the domain of asin is
// -1 <= x <= 1. So we need to perform a range reduction to [0,1] before
// computing the approximation. 
// 
// We use the following identity from [HMF(p80),WIK] for range reduction
// 
// 	asin(-x) = -asin(x)
// 
// We take the absolute value of x, compute asin(x) using the approximation
// and then negate the value if x < 0.
//
// In [HMF] the authors claim an error, e, of |e| <= 5e-5, but the error graph
// in [ADC] looks like the error can be larger that that for some inputs.
// 
Value *DxilExpandTrigIntrinsics::expandASin(IRBuilder<> &builder, DxilInst_Asin asin, DxilModule &DM) {
  assert(asin);
  StringRef name = "asin.x";
  Value *X = asin.get_value();
  Value *PI_2 = ConstantFP::get(X->getType(), math::PI_2);
  Value *Zero = ConstantFP::get(X->getType(), 0.0);
  
  // Range reduction to [0, 1]
  Value *absX = emitFAbs(builder, X, DM.GetOP(), name);

  // Approximation
  Value *psiX = emitSqrt1mXtimesPsiX(builder, absX, DM.GetOP(), name);
  Value *asinX = builder.CreateFSub(PI_2, psiX, name);
  Value *asinmX = builder.CreateFSub(Zero, asinX, name);

  // Range expansion to [-1, 1]
  Value *lt0 = builder.CreateFCmp(CmpInst::FCMP_ULT, X, Zero, name);
  Value *r = builder.CreateSelect(lt0, asinmX, asinX, name);

  return r;
}


// Acos
// ----------------------------------------------------------------------------
// The acos expansion uses the following identity [WIK]. So that we can use the
// same approximation psi*(x) that we use for asin.
// 
// 	acos(x) = pi/2 - asin(x)
// 
// Substituting the equation for asin(x) we get
// 
// 	acos(x) = pi/2 - asin(x)
// 	        = pi/2 - (pi/2 - sqrt(1-x)*psi(x))
// 	        = sqrt(1-x)*psi(x)
// 
// We use the following identity from [HMF(p80),WIK] for range reduction
// 
// 	acos(-x) = pi - acos(x)
//               = pi - sqrt(1-x)*psi(x)
//
// We take the absolute value of x, compute acos(x) using the approximation
// and then subtract from pi if x < 0.
//
Value *DxilExpandTrigIntrinsics::expandACos(IRBuilder<> &builder, DxilInst_Acos acos, DxilModule &DM) {
  assert(acos);
  StringRef name = "acos.x";
  Value *X = acos.get_value();
  Value *PI = ConstantFP::get(X->getType(), math::PI);
  Value *Zero = ConstantFP::get(X->getType(), 0.0);
  
  // Range reduction to [0, 1]
  Value *absX = emitFAbs(builder, X, DM.GetOP(), name);

  // Approximation
  Value *acosX = emitSqrt1mXtimesPsiX(builder, absX, DM.GetOP(), name);
  Value *acosmX = builder.CreateFSub(PI, acosX, name);

  // Range expansion to [-1, 1]
  Value *lt0 = builder.CreateFCmp(CmpInst::FCMP_ULT, X, Zero, name);
  Value *r = builder.CreateSelect(lt0, acosmX, acosX, name);

  return r;
}

// Atan
// ----------------------------------------------------------------------------
// Function
//    arctan X
//
// Range
//    -1 <= X <= 1
//
// Approximation
//    arctan*(x) = c1x + c3x^3 + c5x^5 + c7x^7 + c9x^9
//      c1 =  0.9998660
//      c3 = -0.3302995
//      c5 =  0.1801410
//      c7 = -0.0851330
//      c9 =  0.0208351
// 	
// The polynomial is evaluated using Horner's method to efficiently compute the
// value
// 
// 	  c1x + c3x^3 + c5x^5 + c7x^7 + c9x^9 
// 	= x(c1 + c3x^2 + c5x^4 + c7x^6 + c9x^8)
// 	= x(c1 + x^2(c3 + c5x^2 + c7x^4 + c9x^6))
// 	= x(c1 + x^2(c3 + x^2(c5 + c7x^2 + c9x^4)))
// 	= x(c1 + x^2(c3 + x^2(c5 + x^2(c7 + c9x^2))))
// 	
// The range reduction is a little more compilicated for atan because the
// domain of atan is [-inf, inf], but the domain of the approximation is only
// [-1, 1]. We use the following identities for range reduction from
// [HMF(p80),WIK]
// 	
// 	arctan(-x) = -arctan(x)
//      arctan(x)   = pi/2 - arctan(1/x) if x > 0
// 
// The first identity allows us to only work with positive numbers. The second
// identity allows us to reduce the range to [0,1]. We first convert the value
// to positive by taking abs(x). Then if x > 1 we compute arctan(1/x).
// 
// To expand the range we check if x > 1 then subtracted the computed value from
// pi/2 and if x is negative then negate the final value.
//
Value *DxilExpandTrigIntrinsics::expandATan(IRBuilder<> &builder, DxilInst_Atan atan, DxilModule &DM) {
  assert(atan);
  StringRef name  = "atan.x";
  Value *X = atan.get_value();
  Value *PI_2 = ConstantFP::get(X->getType(), math::PI_2);
  Value *One  = ConstantFP::get(X->getType(), 1.0);
  Value *Zero = ConstantFP::get(X->getType(), 0.0);
  Value *c1 = ConstantFP::get(X->getType(),  0.9998660);
  Value *c3 = ConstantFP::get(X->getType(), -0.3302995);
  Value *c5 = ConstantFP::get(X->getType(),  0.1801410);
  Value *c7 = ConstantFP::get(X->getType(), -0.0851330);
  Value *c9 = ConstantFP::get(X->getType(),  0.0208351);

  // Range reduction to [0, inf]
  Value *absX = emitFAbs(builder, X, DM.GetOP(), name);

  // Range reduction to [0, 1]
  Value *gt1 = builder.CreateFCmp(CmpInst::FCMP_UGT, absX, One, name);
  Value *r1 = builder.CreateFDiv(One, absX, name);
  Value *r2 = builder.CreateSelect(gt1, r1, absX, name);

  // Approximate
  Value *r3 = builder.CreateFMul(r2, r2, name);
  Value *r4 = builder.CreateFMul(r3, c9, name);
         r4 = builder.CreateFAdd(r4, c7, name);
         r4 = builder.CreateFMul(r4, r3, name);
         r4 = builder.CreateFAdd(r4, c5, name);
         r4 = builder.CreateFMul(r4, r3, name);
         r4 = builder.CreateFAdd(r4, c3, name);
         r4 = builder.CreateFMul(r4, r3, name);
         r4 = builder.CreateFAdd(r4, c1, name);
         r4 = builder.CreateFMul(r2, r4, name);

  // Range Expansion to [0, inf]
  Value *r5 = builder.CreateFSub(PI_2, r4, name);
  Value *r6 = builder.CreateSelect(gt1, r5, r4, name);

  // Range Expansion to [-inf, inf]
  Value *r7 = builder.CreateFSub(Zero, r6, name);
  Value *lt0 = builder.CreateFCmp(CmpInst::FCMP_ULT, X, Zero, name);
  Value *r = builder.CreateSelect(lt0, r7, r6, name);

  return r;
}

// Hcos
// ----------------------------------------------------------------------------
// We use the following identity for computing hcos(x) from [HMF(p83)]
// 	
//    cosh(x) = (e^x + e^-x) / 2
// 
// No range reduction is needed.
//
Value *DxilExpandTrigIntrinsics::expandHCos(IRBuilder<> &builder, DxilInst_Hcos hcos, DxilModule &DM) {
  assert(hcos);
  StringRef name = "hcos.x";
  Value *eX, *emX;
  Value *X = hcos.get_value();
  Value *Two = ConstantFP::get(X->getType(), 2.0);

  std::tie(eX, emX) = emitExEmx(builder, X, DM.GetOP(), name);
  Value *r4 = builder.CreateFAdd(eX, emX, name);
  Value *r  = builder.CreateFDiv(r4, Two, name);

  return r;
}

// Hsin
// ----------------------------------------------------------------------------
// We use the following identity for computing hsin(x) from[HMF(p83)]
//
//    sinh(x) = (e^x - e^-x) / 2
//
// No range reduction is needed.
//
Value *DxilExpandTrigIntrinsics::expandHSin(IRBuilder<> &builder, DxilInst_Hsin hsin, DxilModule &DM) {
  assert(hsin);
  StringRef name = "hsin.x";
  Value *eX, *emX;
  Value *X = hsin.get_value();
  Value *Two = ConstantFP::get(X->getType(), 2.0);

  std::tie(eX, emX) = emitExEmx(builder, X, DM.GetOP(), name);
  Value *r4 = builder.CreateFSub(eX, emX, name);
  Value *r  = builder.CreateFDiv(r4, Two, name);

  return r;
}

// Htan
// ----------------------------------------------------------------------------
// We use the following identity for computing hsin(x) from[HMF(p83)]
//
//    tanh(x) = (e^x - e^-x) / (e^x + e^-x)
//
// No range reduction is needed.
//
Value *DxilExpandTrigIntrinsics::expandHTan(IRBuilder<> &builder, DxilInst_Htan htan, DxilModule &DM) {
  assert(htan);
  StringRef name = "htan.x";
  Value *eX, *emX;
  Value *X = htan.get_value();

  std::tie(eX, emX) = emitExEmx(builder, X, DM.GetOP(), name);
  Value *r4 = builder.CreateFSub(eX, emX, name);
  Value *r5 = builder.CreateFAdd(eX, emX, name);
  Value *r  = builder.CreateFDiv(r4, r5, name);

  return r;
}

// Tan
// ----------------------------------------------------------------------------
// We use the following identity for computing tan(x)
//
//    tan(x) = sin(x) / cos(x)
//
// No range reduction is needed.
//
Value *DxilExpandTrigIntrinsics::expandTan(IRBuilder<> &builder,
                                           DxilInst_Tan tan, DxilModule &DM) {
  assert(tan);
  StringRef name = "tan.x";
  Value *X = tan.get_value();
  OP *dxOp = DM.GetOP();
  Value *sin = emitUnaryFloat(builder, X, dxOp, OP::OpCode::Sin, name);
  Value *cos = emitUnaryFloat(builder, X, dxOp, OP::OpCode::Cos, name);
  Value *r = builder.CreateFDiv(sin, cos, name);

  return r;
}

char DxilExpandTrigIntrinsics::ID = 0;

FunctionPass *llvm::createDxilExpandTrigIntrinsicsPass() {
  return new DxilExpandTrigIntrinsics();
}

INITIALIZE_PASS(DxilExpandTrigIntrinsics,
                "hlsl-dxil-expand-trig-intrinsics",
                "DXIL expand trig intrinsics", false, false)
