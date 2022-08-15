///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// HLExpandStoreIntrinsics.cpp                                               //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/Support/Global.h"
#include "dxc/HLSL/HLOperations.h"
#include "dxc/HLSL/HLMatrixType.h"
#include "dxc/HLSL/HLModule.h"
#include "dxc/DXIL/DxilTypeSystem.h"
#include "dxc/HlslIntrinsicOp.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Transforms/Scalar.h"

using namespace hlsl;
using namespace llvm;

namespace {

// Expands buffer stores of aggregate value types
// into stores of its individual elements, 
// before SROA happens and we lose the layout information.
class HLExpandStoreIntrinsics : public FunctionPass {
public:
  static char ID;
  explicit HLExpandStoreIntrinsics() : FunctionPass(ID) {}

  StringRef getPassName() const override {
    return "Expand HLSL store intrinsics";
  }

  bool runOnFunction(Function& Func) override;

private:
  DxilTypeSystem *m_typeSys;
  bool expand(CallInst *StoreCall);
  void emitElementStores(CallInst &OriginalCall,
    SmallVectorImpl<Value*>& GEPIndicesStack, Type *StackTopTy,
    unsigned OffsetFromBase, DxilFieldAnnotation *fieldAnnotation);
};

char HLExpandStoreIntrinsics::ID = 0;

bool HLExpandStoreIntrinsics::runOnFunction(Function& Func) {
  bool changed = false;

  m_typeSys = &(Func.getParent()->GetHLModule().GetTypeSystem());
  for (auto InstIt = inst_begin(Func), InstEnd = inst_end(Func); InstIt != InstEnd;) {
    CallInst *Call = dyn_cast<CallInst>(&*(InstIt++));
    if (Call == nullptr
      || GetHLOpcodeGroup(Call->getCalledFunction()) != HLOpcodeGroup::HLIntrinsic
      || static_cast<IntrinsicOp>(GetHLOpcode(Call)) != IntrinsicOp::MOP_Store) {
      continue;
    }

    changed |= expand(Call);
  }
  return changed;
}

bool HLExpandStoreIntrinsics::expand(CallInst* StoreCall) {
  Value *OldStoreValueArg = StoreCall->getArgOperand(HLOperandIndex::kStoreValOpIdx);
  Type *OldStoreValueArgTy = OldStoreValueArg->getType();
  // Only expand if the value argument is by pointer, which means it's an aggregate.
  if (!OldStoreValueArgTy->isPointerTy()) return false;
  
  IRBuilder<> Builder(StoreCall);
  SmallVector<Value*, 4> GEPIndicesStack;
  GEPIndicesStack.emplace_back(Builder.getInt32(0));
  emitElementStores(*StoreCall, GEPIndicesStack, OldStoreValueArgTy->getPointerElementType(), /* OffsetFromBase */ 0, nullptr);
  DXASSERT(StoreCall->getType()->isVoidTy() && StoreCall->use_empty(),
    "Buffer store intrinsic is expected to return void and hence not have uses.");
  StoreCall->eraseFromParent();
  return true;
}

void HLExpandStoreIntrinsics::emitElementStores(CallInst &OriginalCall,
    SmallVectorImpl<Value*>& GEPIndicesStack, Type *StackTopTy,
    unsigned OffsetFromBase, DxilFieldAnnotation* fieldAnnotation) {
  llvm::Module &Module = *OriginalCall.getModule();
  IRBuilder<> Builder(&OriginalCall);

  StructType* StructTy = dyn_cast<StructType>(StackTopTy);
  if (StructTy != nullptr && !HLMatrixType::isa(StructTy)) {
    const StructLayout* Layout = Module.getDataLayout().getStructLayout(StructTy);
    DxilStructAnnotation *SA = m_typeSys->GetStructAnnotation(StructTy);
    for (unsigned i = 0; i < StructTy->getNumElements(); ++i) {
      Type *ElemTy = StructTy->getElementType(i);
      unsigned ElemOffsetFromBase = OffsetFromBase + Layout->getElementOffset(i);
      GEPIndicesStack.emplace_back(Builder.getInt32(i));
      DxilFieldAnnotation* FA = SA != nullptr ? &(SA->GetFieldAnnotation(i)) : nullptr;
      emitElementStores(OriginalCall, GEPIndicesStack, ElemTy, ElemOffsetFromBase, FA);
      GEPIndicesStack.pop_back();
    }
  }
  else if (ArrayType *ArrayTy = dyn_cast<ArrayType>(StackTopTy)) {
    unsigned ElemSize = (unsigned)Module.getDataLayout().getTypeAllocSize(ArrayTy->getElementType());
    for (int i = 0; i < (int)ArrayTy->getNumElements(); ++i) {
      unsigned ElemOffsetFromBase = OffsetFromBase + ElemSize * i;
      GEPIndicesStack.emplace_back(Builder.getInt32(i));
      emitElementStores(OriginalCall, GEPIndicesStack, ArrayTy->getElementType(), ElemOffsetFromBase, fieldAnnotation);
      GEPIndicesStack.pop_back();
    }
  }
  else {
    // Scalar or vector
    Value* OpcodeVal = OriginalCall.getArgOperand(HLOperandIndex::kOpcodeIdx);

    Value* BufHandle = OriginalCall.getArgOperand(HLOperandIndex::kHandleOpIdx);

    Value* OffsetVal = OriginalCall.getArgOperand(HLOperandIndex::kStoreOffsetOpIdx);
    if (OffsetFromBase > 0)
      OffsetVal = Builder.CreateAdd(OffsetVal, Builder.getInt32(OffsetFromBase));

    Value* AggPtr = OriginalCall.getArgOperand(HLOperandIndex::kStoreValOpIdx);
    Value *ElemPtr = Builder.CreateGEP(AggPtr, GEPIndicesStack);
    Value* ElemVal = nullptr;

    if (HLMatrixType::isa(StackTopTy) && fieldAnnotation &&
        fieldAnnotation->HasMatrixAnnotation()) {

      // For matrix load, we generate HL intrinsic matldst.colLoad/matldst.rowLoad
      // instead of LLVM LoadInst to ensure that it gets lowered properly later
      // in HLMatrixLowerPass
      bool isRowMajor = fieldAnnotation->GetMatrixAnnotation().Orientation ==
                        hlsl::MatrixOrientation::RowMajor;
      unsigned matLdOpcode =
          isRowMajor ? static_cast<unsigned>(HLMatLoadStoreOpcode::RowMatLoad)
                     : static_cast<unsigned>(HLMatLoadStoreOpcode::ColMatLoad);
      // Generate matrix load
      FunctionType *MatLdFnType = FunctionType::get(
          StackTopTy, {Builder.getInt32Ty(), ElemPtr->getType()},
          /* isVarArg */ false);

      Function *MatLdFn = GetOrCreateHLFunction(
          Module, MatLdFnType, HLOpcodeGroup::HLMatLoadStore, matLdOpcode);
      Value *MatLdOpCode = ConstantInt::get(Builder.getInt32Ty(), matLdOpcode);
      ElemVal = Builder.CreateCall(MatLdFn, {MatLdOpCode, ElemPtr});
    } else {
      ElemVal = Builder.CreateLoad(ElemPtr); // We go from memory to memory so no special bool handling needed
    }

    FunctionType *NewCalleeType = FunctionType::get(Builder.getVoidTy(),
      { OpcodeVal->getType(), BufHandle->getType(), OffsetVal->getType(), ElemVal->getType() },
      /* isVarArg */ false);
    Function *NewCallee = GetOrCreateHLFunction(Module, NewCalleeType,
      HLOpcodeGroup::HLIntrinsic, (unsigned)IntrinsicOp::MOP_Store);
    Builder.CreateCall(NewCallee, { OpcodeVal, BufHandle, OffsetVal, ElemVal });
  }
}

} // namespace

FunctionPass *llvm::createHLExpandStoreIntrinsicsPass() { return new HLExpandStoreIntrinsics(); }

INITIALIZE_PASS(HLExpandStoreIntrinsics, "hl-expand-store-intrinsics",
                "Expand HLSL store intrinsics", false, false)
