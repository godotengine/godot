//===-- DxilSimplify.cpp - Fold dxil intrinsics into constants -----===//
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

// simplify dxil op like mad 0, a, b->b.

#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/IRBuilder.h"

#include "dxc/DXIL/DxilModule.h"
#include "dxc/DXIL/DxilOperations.h"
#include "llvm/Analysis/DxilConstantFolding.h"
#include "llvm/Analysis/DxilSimplify.h"

using namespace llvm;
using namespace hlsl;

namespace {
DXIL::OpCode GetOpcode(Value *opArg) {
  if (ConstantInt *ci = dyn_cast<ConstantInt>(opArg)) {
    uint64_t opcode = ci->getLimitedValue();
    if (opcode < static_cast<uint64_t>(OP::OpCode::NumOpCodes)) {
      return static_cast<OP::OpCode>(opcode);
    }
  }
  return DXIL::OpCode::NumOpCodes;
}
} // namespace

namespace hlsl {
bool CanSimplify(const llvm::Function *F) {
  // Only simplify dxil functions when we have a valid dxil module.
  if (!F->getParent()->HasDxilModule()) {
    assert(!OP::IsDxilOpFunc(F) && "dx.op function with no dxil module?");
    return false;
  }

  if (CanConstantFoldCallTo(F))
    return true;

  // Lookup opcode class in dxil module. Set default value to invalid class.
  OP::OpCodeClass opClass = OP::OpCodeClass::NumOpClasses;
  const bool found =
      F->getParent()->GetDxilModule().GetOP()->GetOpCodeClass(F, opClass);

  // Return true for those dxil operation classes we can simplify.
  if (found) {
    switch (opClass) {
    default:
      break;
    case OP::OpCodeClass::Tertiary:
      return true;
    }
  }

  return false;
}

/// \brief Given a function and set of arguments, see if we can fold the
/// result as dxil operation.
///
/// If this call could not be simplified returns null.
Value *SimplifyDxilCall(llvm::Function *F, ArrayRef<Value *> Args,
                        llvm::Instruction *I,
                        bool MayInsert)
{
  if (!F->getParent()->HasDxilModule()) {
    assert(!OP::IsDxilOpFunc(F) && "dx.op function with no dxil module?");
    return nullptr;
  }

  DxilModule &DM = F->getParent()->GetDxilModule();
  // Skip precise.
  if (DM.IsPrecise(I))
    return nullptr;

  // Lookup opcode class in dxil module. Set default value to invalid class.
  OP::OpCodeClass opClass = OP::OpCodeClass::NumOpClasses;
  const bool found = DM.GetOP()->GetOpCodeClass(F, opClass);
  if (!found)
    return nullptr;

  DXIL::OpCode opcode = GetOpcode(Args[0]);
  if (opcode == DXIL::OpCode::NumOpCodes)
    return nullptr;

  if (CanConstantFoldCallTo(F)) {
    bool bAllConstant = true;
    SmallVector<Constant *, 4> ConstantArgs;
    ConstantArgs.reserve(Args.size());
    for (Value *V : Args) {
      Constant *C = dyn_cast<Constant>(V);
      if (!C) {
        bAllConstant = false;
        break;
      }
      ConstantArgs.push_back(C);
    }

    if (bAllConstant)
      return hlsl::ConstantFoldScalarCall(F->getName(), F->getReturnType(),
                                          ConstantArgs);
  }

  switch (opcode) {
  default:
    return nullptr;
  case DXIL::OpCode::FMad: {
    Value *op0 = Args[DXIL::OperandIndex::kTrinarySrc0OpIdx];
    Value *op2 = Args[DXIL::OperandIndex::kTrinarySrc2OpIdx];
    Constant *zero = ConstantFP::get(op0->getType(), 0);
    if (op0 == zero)
      return op2;
    Value *op1 = Args[DXIL::OperandIndex::kTrinarySrc1OpIdx];
    if (op1 == zero)
      return op2;

    if (MayInsert) {
      Constant *one = ConstantFP::get(op0->getType(), 1);
      if (op0 == one) {
        IRBuilder<> Builder(I);
        llvm::FastMathFlags FMF;
        FMF.setUnsafeAlgebraHLSL();
        Builder.SetFastMathFlags(FMF);
        return Builder.CreateFAdd(op1, op2);
      }
      if (op1 == one) {
        IRBuilder<> Builder(I);
        llvm::FastMathFlags FMF;
        FMF.setUnsafeAlgebraHLSL();
        Builder.SetFastMathFlags(FMF);

        return Builder.CreateFAdd(op0, op2);
      }
    }
    return nullptr;
  } break;
  case DXIL::OpCode::IMad:
  case DXIL::OpCode::UMad: {
    Value *op0 = Args[DXIL::OperandIndex::kTrinarySrc0OpIdx];
    Value *op2 = Args[DXIL::OperandIndex::kTrinarySrc2OpIdx];
    Constant *zero = ConstantInt::get(op0->getType(), 0);
    if (op0 == zero)
      return op2;
    Value *op1 = Args[DXIL::OperandIndex::kTrinarySrc1OpIdx];
    if (op1 == zero)
      return op2;

    if (MayInsert) {
      Constant *one = ConstantInt::get(op0->getType(), 1);
      if (op0 == one) {
        IRBuilder<> Builder(I);
        return Builder.CreateAdd(op1, op2);
      }
      if (op1 == one) {
        IRBuilder<> Builder(I);
        return Builder.CreateAdd(op0, op2);
      }
    }
    return nullptr;
  } break;
  case DXIL::OpCode::UMax: {
    Value *op0 = Args[DXIL::OperandIndex::kBinarySrc0OpIdx];
    Value *op1 = Args[DXIL::OperandIndex::kBinarySrc1OpIdx];
    Constant *zero = ConstantInt::get(op0->getType(), 0);
    if (op0 == zero)
      return op1;
    if (op1 == zero)
      return op0;
    return nullptr;
  } break;
  }
}

} // namespace hlsl
