///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// HLOperationLower.cpp                                                      //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Lower functions to lower HL operations to DXIL operations.                //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#define _USE_MATH_DEFINES
#include <array>
#include <cmath>
#include <unordered_set>
#include <functional>

#include "dxc/DXIL/DxilModule.h"
#include "dxc/DXIL/DxilOperations.h"
#include "dxc/HLSL/HLMatrixLowerHelper.h"
#include "dxc/HLSL/HLMatrixType.h"
#include "dxc/HLSL/HLModule.h"
#include "dxc/DXIL/DxilUtil.h"
#include "dxc/HLSL/HLOperationLower.h"
#include "dxc/HLSL/HLOperationLowerExtension.h"
#include "dxc/HLSL/HLOperations.h"
#include "dxc/HlslIntrinsicOp.h"
#include "dxc/DXIL/DxilResourceProperties.h"

#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/ADT/APSInt.h"

using namespace llvm;
using namespace hlsl;

struct HLOperationLowerHelper {
  HLModule &M;
  OP &hlslOP;
  Type *voidTy;
  Type *f32Ty;
  Type *i32Ty;
  Type *i16Ty;
  llvm::Type *i1Ty;
  Type *i8Ty;
  DxilTypeSystem &dxilTypeSys;
  DxilFunctionProps *functionProps;
  bool bLegacyCBufferLoad;
  DataLayout dataLayout;
  HLOperationLowerHelper(HLModule &HLM);
};

HLOperationLowerHelper::HLOperationLowerHelper(HLModule &HLM)
    : M(HLM), hlslOP(*HLM.GetOP()), dxilTypeSys(HLM.GetTypeSystem()),
      dataLayout(DataLayout(HLM.GetHLOptions().bUseMinPrecision
                                  ? hlsl::DXIL::kLegacyLayoutString
                                  : hlsl::DXIL::kNewLayoutString)) {
  llvm::LLVMContext &Ctx = HLM.GetCtx();
  voidTy = Type::getVoidTy(Ctx);
  f32Ty = Type::getFloatTy(Ctx);
  i32Ty = Type::getInt32Ty(Ctx);
  i16Ty = Type::getInt16Ty(Ctx);
  i1Ty = Type::getInt1Ty(Ctx);
  i8Ty = Type::getInt8Ty(Ctx);
  Function *EntryFunc = HLM.GetEntryFunction();
  functionProps = nullptr;
  if (HLM.HasDxilFunctionProps(EntryFunc))
    functionProps = &HLM.GetDxilFunctionProps(EntryFunc);
  bLegacyCBufferLoad = HLM.GetHLOptions().bLegacyCBufferLoad;
}

struct HLObjectOperationLowerHelper {
private:
  // For object intrinsics.
  HLModule &HLM;
  struct ResAttribute {
    DXIL::ResourceClass RC;
    DXIL::ResourceKind RK;
    Type *ResourceType;
  };
  std::unordered_map<Value *, ResAttribute> HandleMetaMap;
  std::unordered_set<Instruction *> &UpdateCounterSet;
  // Map from pointer of cbuffer to pointer of resource.
  // For cbuffer like this:
  //   cbuffer A {
  //     Texture2D T;
  //   };
  // A global resource Texture2D T2 will be created for Texture2D T.
  // CBPtrToResourceMap[T] will return T2.
  std::unordered_map<Value *, Value *> CBPtrToResourceMap;

public:
  HLObjectOperationLowerHelper(HLModule &HLM,
                               std::unordered_set<Instruction *> &UpdateCounter)
      : HLM(HLM), UpdateCounterSet(UpdateCounter) {}
  DXIL::ResourceClass GetRC(Value *Handle) {
    ResAttribute &Res = FindCreateHandleResourceBase(Handle);
    return Res.RC;
  }
  DXIL::ResourceKind GetRK(Value *Handle) {
    ResAttribute &Res = FindCreateHandleResourceBase(Handle);
    return Res.RK;
  }
  Type *GetResourceType(Value *Handle) {
    ResAttribute &Res = FindCreateHandleResourceBase(Handle);
    return Res.ResourceType;
  }

  void MarkHasCounter(Value *handle, Type *i8Ty) {
    CallInst *CIHandle = cast<CallInst>(handle);
    DXASSERT(hlsl::GetHLOpcodeGroup(CIHandle->getCalledFunction()) == HLOpcodeGroup::HLAnnotateHandle, "else invalid handle");
    // Mark has counter for the input handle.
    Value *counterHandle =
        CIHandle->getArgOperand(HLOperandIndex::kAnnotateHandleHandleOpIdx);
    // Change kind into StructurBufferWithCounter.
    Constant *Props = cast<Constant>(CIHandle->getArgOperand(
        HLOperandIndex::kAnnotateHandleResourcePropertiesOpIdx));
    DxilResourceProperties RP = resource_helper::loadPropsFromConstant(*Props);
    RP.Basic.SamplerCmpOrHasCounter = true;

    CIHandle->setArgOperand(
        HLOperandIndex::kAnnotateHandleResourcePropertiesOpIdx,
        resource_helper::getAsConstant(
            RP,
            HLM.GetOP()->GetResourcePropertiesType(),
            *HLM.GetShaderModel()));

    DXIL::ResourceClass RC = GetRC(handle);
    DXASSERT_LOCALVAR(RC, RC == DXIL::ResourceClass::UAV,
                      "must UAV for counter");
    std::unordered_set<Value *> resSet;
    MarkHasCounterOnCreateHandle(counterHandle, resSet);
  }

  Value *GetOrCreateResourceForCbPtr(GetElementPtrInst *CbPtr,
                                     GlobalVariable *CbGV,
                                     DxilResourceProperties &RP) {
    // Change array idx to 0 to make sure all array ptr share same key.
    Value *Key = UniformCbPtr(CbPtr, CbGV);
    if (CBPtrToResourceMap.count(Key))
      return CBPtrToResourceMap[Key];
    Value *Resource = CreateResourceForCbPtr(CbPtr, CbGV, RP);
    CBPtrToResourceMap[Key] = Resource;
    return Resource;
  }

  Value *LowerCbResourcePtr(GetElementPtrInst *CbPtr, Value *ResPtr) {
    // Simple case.
    if (ResPtr->getType() == CbPtr->getType())
      return ResPtr;

    // Array case.
    DXASSERT_NOMSG(ResPtr->getType()->getPointerElementType()->isArrayTy());

    IRBuilder<> Builder(CbPtr);
    gep_type_iterator GEPIt = gep_type_begin(CbPtr), E = gep_type_end(CbPtr);

    Value *arrayIdx = GEPIt.getOperand();

    // Only calc array idx and size.
    // Ignore struct type part.
    for (; GEPIt != E; ++GEPIt) {
      if (GEPIt->isArrayTy()) {
        arrayIdx = Builder.CreateMul(
            arrayIdx, Builder.getInt32(GEPIt->getArrayNumElements()));
        arrayIdx = Builder.CreateAdd(arrayIdx, GEPIt.getOperand());
      }
    }

    return Builder.CreateGEP(ResPtr, {Builder.getInt32(0), arrayIdx});
  }

  DxilResourceProperties GetResPropsFromAnnotateHandle(CallInst *Anno) {
    Constant *Props = cast<Constant>(Anno->getArgOperand(
        HLOperandIndex::kAnnotateHandleResourcePropertiesOpIdx));
    DxilResourceProperties RP = resource_helper::loadPropsFromConstant(
        *Props);
    return RP;
  }

private:
  ResAttribute &FindCreateHandleResourceBase(Value *Handle) {
    if (HandleMetaMap.count(Handle))
      return HandleMetaMap[Handle];

    // Add invalid first to avoid dead loop.
    HandleMetaMap[Handle] = {DXIL::ResourceClass::Invalid,
                             DXIL::ResourceKind::Invalid,
                             StructType::get(Type::getVoidTy(HLM.GetCtx()), nullptr)};
    if (CallInst *CI = dyn_cast<CallInst>(Handle)) {
      hlsl::HLOpcodeGroup group =
          hlsl::GetHLOpcodeGroupByName(CI->getCalledFunction());
      if (group == HLOpcodeGroup::HLAnnotateHandle) {
        Constant *Props = cast<Constant>(CI->getArgOperand(
            HLOperandIndex::kAnnotateHandleResourcePropertiesOpIdx));
        DxilResourceProperties RP =
            resource_helper::loadPropsFromConstant(*Props);
        Type *ResTy =
            CI->getArgOperand(HLOperandIndex::kAnnotateHandleResourceTypeOpIdx)
                ->getType();

        ResAttribute Attrib = {RP.getResourceClass(), RP.getResourceKind(),
                               ResTy};

        HandleMetaMap[Handle] = Attrib;
        return HandleMetaMap[Handle];
      }
    }
    Handle->getContext().emitError("cannot map resource to handle");

    return HandleMetaMap[Handle];
  }
  CallInst *FindCreateHandle(Value *handle,
                             std::unordered_set<Value *> &resSet) {
    // Already checked.
    if (resSet.count(handle))
      return nullptr;
    resSet.insert(handle);

    if (CallInst *CI = dyn_cast<CallInst>(handle))
      return CI;
    if (SelectInst *Sel = dyn_cast<SelectInst>(handle)) {
      if (CallInst *CI = FindCreateHandle(Sel->getTrueValue(), resSet))
        return CI;
      if (CallInst *CI = FindCreateHandle(Sel->getFalseValue(), resSet))
        return CI;
      return nullptr;
    }
    if (PHINode *Phi = dyn_cast<PHINode>(handle)) {
      for (unsigned i = 0; i < Phi->getNumOperands(); i++) {
        if (CallInst *CI = FindCreateHandle(Phi->getOperand(i), resSet))
          return CI;
      }
      return nullptr;
    }

    return nullptr;
  }
  void MarkHasCounterOnCreateHandle(Value *handle,
                                    std::unordered_set<Value *> &resSet) {
    // Already checked.
    if (resSet.count(handle))
      return;
    resSet.insert(handle);

    if (CallInst *CI = dyn_cast<CallInst>(handle)) {
      Value *Res =
          CI->getArgOperand(HLOperandIndex::kCreateHandleResourceOpIdx);
      LoadInst *LdRes = dyn_cast<LoadInst>(Res);
      if (LdRes) {
        UpdateCounterSet.insert(LdRes);
        return;
      }
      if (CallInst *CallRes = dyn_cast<CallInst>(Res)) {
        hlsl::HLOpcodeGroup group =
            hlsl::GetHLOpcodeGroup(CallRes->getCalledFunction());
        if (group == HLOpcodeGroup::HLCast) {
          HLCastOpcode opcode =
              static_cast<HLCastOpcode>(hlsl::GetHLOpcode(CallRes));
          if (opcode == HLCastOpcode::HandleToResCast) {
            if (Instruction *Hdl = dyn_cast<Instruction>(
                    CallRes->getArgOperand(HLOperandIndex::kUnaryOpSrc0Idx)))
              UpdateCounterSet.insert(Hdl);
            return;
          }
        }
      }
      dxilutil::EmitErrorOnInstruction(CI, "cannot map resource to handle.");
      return;
    }
    if (SelectInst *Sel = dyn_cast<SelectInst>(handle)) {
      MarkHasCounterOnCreateHandle(Sel->getTrueValue(), resSet);
      MarkHasCounterOnCreateHandle(Sel->getFalseValue(), resSet);
    }
    if (PHINode *Phi = dyn_cast<PHINode>(handle)) {
      for (unsigned i = 0; i < Phi->getNumOperands(); i++) {
        MarkHasCounterOnCreateHandle(Phi->getOperand(i), resSet);
      }
    }
  }

  Value *UniformCbPtr(GetElementPtrInst *CbPtr, GlobalVariable *CbGV) {
    gep_type_iterator GEPIt = gep_type_begin(CbPtr), E = gep_type_end(CbPtr);
    std::vector<Value *> idxList(CbPtr->idx_begin(), CbPtr->idx_end());
    unsigned i = 0;
    IRBuilder<> Builder(HLM.GetCtx());
    Value *zero = Builder.getInt32(0);
    for (; GEPIt != E; ++GEPIt, ++i) {
      ConstantInt *ImmIdx = dyn_cast<ConstantInt>(GEPIt.getOperand());
      if (!ImmIdx) {
        // Remove dynamic indexing to avoid crash.
        idxList[i] = zero;
      }
    }

    Value *Key = Builder.CreateInBoundsGEP(CbGV, idxList);
    return Key;
  }

  Value *CreateResourceForCbPtr(GetElementPtrInst *CbPtr, GlobalVariable *CbGV,
                                DxilResourceProperties &RP) {
    Type *CbTy = CbPtr->getPointerOperandType();
    DXASSERT_LOCALVAR(CbTy, CbTy == CbGV->getType(), "else arg not point to var");

    gep_type_iterator GEPIt = gep_type_begin(CbPtr), E = gep_type_end(CbPtr);
    unsigned i = 0;
    IRBuilder<> Builder(HLM.GetCtx());
    unsigned arraySize = 1;
    DxilTypeSystem &typeSys = HLM.GetTypeSystem();

    std::string Name;
    for (; GEPIt != E; ++GEPIt, ++i) {
      if (GEPIt->isArrayTy()) {
        arraySize *= GEPIt->getArrayNumElements();
        if (!Name.empty())
          Name += ".";
        if (ConstantInt *ImmIdx = dyn_cast<ConstantInt>(GEPIt.getOperand())) {
          unsigned idx = ImmIdx->getLimitedValue();
          Name += std::to_string(idx);
        }
      } else if (GEPIt->isStructTy()) {
        DxilStructAnnotation *typeAnnot =
            typeSys.GetStructAnnotation(cast<StructType>(*GEPIt));
        DXASSERT_NOMSG(typeAnnot);
        unsigned idx = cast<ConstantInt>(GEPIt.getOperand())->getLimitedValue();
        DXASSERT_NOMSG(typeAnnot->GetNumFields() > idx);
        DxilFieldAnnotation &fieldAnnot = typeAnnot->GetFieldAnnotation(idx);
        if (!Name.empty())
          Name += ".";
        Name += fieldAnnot.GetFieldName();
      }
    }

    Type *Ty = CbPtr->getResultElementType();
    // Not support resource array in cbuffer.
    unsigned ResBinding = HLM.GetBindingForResourceInCB(CbPtr, CbGV, RP.getResourceClass());
    return CreateResourceGV(Ty, Name, RP, ResBinding);
  }

  Value *CreateResourceGV(Type *Ty, StringRef Name, DxilResourceProperties &RP,
                          unsigned ResBinding) {
    Module &M = *HLM.GetModule();
    Constant *GV = M.getOrInsertGlobal(Name, Ty);
    // Create resource and set GV as globalSym.
    DxilResourceBase *Res = HLM.AddResourceWithGlobalVariableAndProps(GV, RP);
    DXASSERT(Res, "fail to create resource for global variable in cbuffer");
    Res->SetLowerBound(ResBinding);
    return GV;
  }
};

// Helper for lowering resource extension methods.
struct HLObjectExtensionLowerHelper : public hlsl::HLResourceLookup {
    explicit HLObjectExtensionLowerHelper(HLObjectOperationLowerHelper &ObjHelper)
        : m_ObjHelper(ObjHelper)
    { }

    virtual bool GetResourceKindName(Value *HLHandle, const char **ppName)
    {
        DXIL::ResourceKind K = m_ObjHelper.GetRK(HLHandle);
        bool Success = K != DXIL::ResourceKind::Invalid;
        if (Success)
        {
            *ppName = hlsl::GetResourceKindName(K);
        }
        return Success;
    }

private:
    HLObjectOperationLowerHelper &m_ObjHelper;
};

using IntrinsicLowerFuncTy = Value *(CallInst *CI, IntrinsicOp IOP,
                                     DXIL::OpCode opcode,
                                     HLOperationLowerHelper &helper, HLObjectOperationLowerHelper *pObjHelper, bool &Translated);

struct IntrinsicLower {
  // Intrinsic opcode.
  IntrinsicOp IntriOpcode;
  // Lower function.
  IntrinsicLowerFuncTy &LowerFunc;
  // DXIL opcode if can direct map.
  DXIL::OpCode DxilOpcode;
};

// IOP intrinsics.
namespace {

Value *TrivialDxilOperation(Function *dxilFunc, OP::OpCode opcode, ArrayRef<Value *> refArgs,
                            Type *Ty, Type *RetTy, OP *hlslOP,
                            IRBuilder<> &Builder) {
  unsigned argNum = refArgs.size();

  std::vector<Value *> args = refArgs;

  if (Ty->isVectorTy()) {
    Value *retVal = llvm::UndefValue::get(RetTy);
    unsigned vecSize = Ty->getVectorNumElements();
    for (unsigned i = 0; i < vecSize; i++) {
      // Update vector args, skip known opcode arg.
      for (unsigned argIdx = HLOperandIndex::kUnaryOpSrc0Idx; argIdx < argNum;
           argIdx++) {
        if (refArgs[argIdx]->getType()->isVectorTy()) {
          Value *arg = refArgs[argIdx];
          args[argIdx] = Builder.CreateExtractElement(arg, i);
        }
      }
      Value *EltOP =
          Builder.CreateCall(dxilFunc, args, hlslOP->GetOpCodeName(opcode));
      retVal = Builder.CreateInsertElement(retVal, EltOP, i);
    }
    return retVal;
  } else {
    if (!RetTy->isVoidTy()) {
      Value *retVal =
          Builder.CreateCall(dxilFunc, args, hlslOP->GetOpCodeName(opcode));
      return retVal;
    } else {
      // Cannot add name to void.
      return Builder.CreateCall(dxilFunc, args);
    }
  }
}
// Generates a DXIL operation over an overloaded type (Ty), returning a
// RetTy value; when Ty is a vector, it will replicate per-element operations
// into RetTy to rebuild it.
Value *TrivialDxilOperation(OP::OpCode opcode, ArrayRef<Value *> refArgs,
                            Type *Ty, Type *RetTy, OP *hlslOP,
                            IRBuilder<> &Builder) {
  Type *EltTy = Ty->getScalarType();
  Function *dxilFunc = hlslOP->GetOpFunc(opcode, EltTy);

  return TrivialDxilOperation(dxilFunc, opcode, refArgs, Ty, RetTy, hlslOP, Builder);
}

Value *TrivialDxilOperation(OP::OpCode opcode, ArrayRef<Value *> refArgs,
                            Type *Ty, Instruction *Inst, OP *hlslOP) {
  DXASSERT(refArgs.size() > 0, "else opcode isn't in signature");
  DXASSERT(refArgs[0] == nullptr,
           "else caller has already filled the value in");
  IRBuilder<> B(Inst);
  Constant *opArg = hlslOP->GetU32Const((unsigned)opcode);
  const_cast<llvm::Value **>(refArgs.data())[0] =
      opArg; // actually stack memory from caller
  return TrivialDxilOperation(opcode, refArgs, Ty, Inst->getType(), hlslOP, B);
}

Value *TrivialDxilUnaryOperationRet(OP::OpCode opcode, Value *src, Type *RetTy,
                                    hlsl::OP *hlslOP, IRBuilder<> &Builder) {
  Type *Ty = src->getType();

  Constant *opArg = hlslOP->GetU32Const((unsigned)opcode);
  Value *args[] = {opArg, src};

  return TrivialDxilOperation(opcode, args, Ty, RetTy, hlslOP, Builder);
}

Value *TrivialDxilUnaryOperation(OP::OpCode opcode, Value *src,
                                 hlsl::OP *hlslOP, IRBuilder<> &Builder) {
  return TrivialDxilUnaryOperationRet(opcode, src, src->getType(), hlslOP,
                                      Builder);
}

Value *TrivialDxilBinaryOperation(OP::OpCode opcode, Value *src0, Value *src1,
                                  hlsl::OP *hlslOP, IRBuilder<> &Builder) {
  Type *Ty = src0->getType();

  Constant *opArg = hlslOP->GetU32Const((unsigned)opcode);
  Value *args[] = {opArg, src0, src1};

  return TrivialDxilOperation(opcode, args, Ty, Ty, hlslOP, Builder);
}

Value *TrivialDxilTrinaryOperation(OP::OpCode opcode, Value *src0, Value *src1,
                                   Value *src2, hlsl::OP *hlslOP,
                                   IRBuilder<> &Builder) {
  Type *Ty = src0->getType();

  Constant *opArg = hlslOP->GetU32Const((unsigned)opcode);
  Value *args[] = {opArg, src0, src1, src2};

  return TrivialDxilOperation(opcode, args, Ty, Ty, hlslOP, Builder);
}

Value *TrivialUnaryOperation(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                             HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  Value *src0 = CI->getArgOperand(HLOperandIndex::kUnaryOpSrc0Idx);
  IRBuilder<> Builder(CI);
  hlsl::OP *hlslOP = &helper.hlslOP;
  Value *retVal = TrivialDxilUnaryOperationRet(opcode, src0, CI->getType(), hlslOP, Builder);
  return retVal;
}

Value *TrivialBinaryOperation(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                              HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  Value *src0 = CI->getArgOperand(HLOperandIndex::kBinaryOpSrc0Idx);
  Value *src1 = CI->getArgOperand(HLOperandIndex::kBinaryOpSrc1Idx);
  IRBuilder<> Builder(CI);

  Value *binOp =
      TrivialDxilBinaryOperation(opcode, src0, src1, hlslOP, Builder);
  return binOp;
}

Value *TrivialTrinaryOperation(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                               HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  Value *src0 = CI->getArgOperand(HLOperandIndex::kTrinaryOpSrc0Idx);
  Value *src1 = CI->getArgOperand(HLOperandIndex::kTrinaryOpSrc1Idx);
  Value *src2 = CI->getArgOperand(HLOperandIndex::kTrinaryOpSrc2Idx);
  IRBuilder<> Builder(CI);

  Value *triOp =
      TrivialDxilTrinaryOperation(opcode, src0, src1, src2, hlslOP, Builder);
  return triOp;
}

Value *TrivialIsSpecialFloat(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                             HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  Value *src = CI->getArgOperand(HLOperandIndex::kUnaryOpSrc0Idx);
  IRBuilder<> Builder(CI);

  Type *Ty = src->getType();
  Type *RetTy = Type::getInt1Ty(CI->getContext());
  if (Ty->isVectorTy())
    RetTy = VectorType::get(RetTy, Ty->getVectorNumElements());

  Constant *opArg = hlslOP->GetU32Const((unsigned)opcode);
  Value *args[] = {opArg, src};

  return TrivialDxilOperation(opcode, args, Ty, RetTy, hlslOP, Builder);
}

bool IsResourceGEP(GetElementPtrInst *I) {
  Type *Ty = I->getType()->getPointerElementType();
  Ty = dxilutil::GetArrayEltTy(Ty);
  // Only mark on GEP which point to resource.
  return dxilutil::IsHLSLResourceType(Ty);
}

Value *TranslateNonUniformResourceIndex(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                      HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  Value *V = CI->getArgOperand(HLOperandIndex::kUnaryOpSrc0Idx);
  Type *hdlTy = helper.hlslOP.GetHandleType();
  for (User *U : CI->users()) {
    if (GetElementPtrInst *I = dyn_cast<GetElementPtrInst>(U)) {
      // Only mark on GEP which point to resource.
      if (IsResourceGEP(I))
        DxilMDHelper::MarkNonUniform(I);
    } else if (CastInst *castI = dyn_cast<CastInst>(U)) {
      for (User *castU : castI->users()) {
        if (GetElementPtrInst *I = dyn_cast<GetElementPtrInst>(castU)) {
          // Only mark on GEP which point to resource.
          if (IsResourceGEP(I))
            DxilMDHelper::MarkNonUniform(I);
        } else if (CallInst *CI = dyn_cast<CallInst>(castU)) {
          if (CI->getType() == hdlTy)
            DxilMDHelper::MarkNonUniform(CI);
        }
      }
    } else if (CallInst *CI = dyn_cast<CallInst>(U)) {
      if (CI->getType() == hdlTy)
        DxilMDHelper::MarkNonUniform(CI);
    }
  }
  CI->replaceAllUsesWith(V);
  return nullptr;
}

Value *TrivialBarrier(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                      HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *OP = &helper.hlslOP;
  Function *dxilFunc = OP->GetOpFunc(OP::OpCode::Barrier, CI->getType());
  Constant *opArg = OP->GetU32Const((unsigned)OP::OpCode::Barrier);

  unsigned uglobal = static_cast<unsigned>(DXIL::BarrierMode::UAVFenceGlobal);
  unsigned g = static_cast<unsigned>(DXIL::BarrierMode::TGSMFence);
  unsigned t = static_cast<unsigned>(DXIL::BarrierMode::SyncThreadGroup);
  // unsigned ut = static_cast<unsigned>(DXIL::BarrierMode::UAVFenceThreadGroup);

  unsigned barrierMode = 0;
  switch (IOP) {
  case IntrinsicOp::IOP_AllMemoryBarrier:
    barrierMode = uglobal | g;
    break;
  case IntrinsicOp::IOP_AllMemoryBarrierWithGroupSync:
    barrierMode = uglobal | g | t;
    break;
  case IntrinsicOp::IOP_GroupMemoryBarrier:
    barrierMode = g;
    break;
  case IntrinsicOp::IOP_GroupMemoryBarrierWithGroupSync:
    barrierMode = g | t;
    break;
  case IntrinsicOp::IOP_DeviceMemoryBarrier:
    barrierMode = uglobal;
    break;
  case IntrinsicOp::IOP_DeviceMemoryBarrierWithGroupSync:
    barrierMode = uglobal | t;
    break;
  default:
    DXASSERT(0, "invalid opcode for barrier");
    break;
  }
  Value *src0 = OP->GetU32Const(static_cast<unsigned>(barrierMode));

  Value *args[] = {opArg, src0};

  IRBuilder<> Builder(CI);
  Builder.CreateCall(dxilFunc, args);
  return nullptr;
}

Value *TranslateD3DColorToUByte4(CallInst *CI, IntrinsicOp IOP,
                                 OP::OpCode opcode,
                                 HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  IRBuilder<> Builder(CI);
  Value *val = CI->getArgOperand(HLOperandIndex::kUnaryOpSrc0Idx);
  Type *Ty = val->getType();

  // Use the same scaling factor used by FXC (i.e., 255.001953)
  // Excerpt from stackoverflow discussion:
  // "Built-in rounding, necessary because of truncation. 0.001953 * 256 = 0.5"
  Constant *toByteConst = ConstantFP::get(Ty->getScalarType(), 255.001953);

  if (Ty->isVectorTy()) {
    static constexpr int supportedVecElemCount = 4;
    if (Ty->getVectorNumElements() == supportedVecElemCount) {
      toByteConst = ConstantVector::getSplat(supportedVecElemCount, toByteConst);
      // Swizzle the input val -> val.zyxw
      std::vector<int> mask { 2, 1, 0, 3 };
      val = Builder.CreateShuffleVector(val, val, mask);
    } else {
      dxilutil::EmitErrorOnInstruction(CI, "Unsupported input type for intrinsic D3DColorToUByte4.");
      return UndefValue::get(CI->getType());
    }
  }

  Value *byte4 = Builder.CreateFMul(toByteConst, val);
  return Builder.CreateCast(Instruction::CastOps::FPToSI, byte4, CI->getType());
}

// Returns true if pow can be implemented using Fxc's mul-only code gen pattern.
// Fxc uses the below rules when choosing mul-only code gen pattern to implement pow function.
// Rule 1: Applicable only to power values in the range [INT32_MIN, INT32_MAX]
// Rule 2: The maximum number of mul ops needed shouldn't exceed (2n+1) or (n+1) based on whether the power
//         is a positive or a negative value. Here "n" is the number of scalar elements in power.
// Rule 3: Power must be an exact value.
// +----------+---------------------+------------------+
// | BaseType | IsExponentPositive  | MaxMulOpsAllowed |
// +----------+---------------------+------------------+
// | float4x4 | True                |               33 |
// | float4x4 | False               |               17 |
// | float4x2 | True                |               17 |
// | float4x2 | False               |                9 |
// | float2x4 | True                |               17 |
// | float2x4 | False               |                9 |
// | float4   | True                |                9 |
// | float4   | False               |                5 |
// | float2   | True                |                5 |
// | float2   | False               |                3 |
// | float    | True                |                3 |
// | float    | False               |                2 |
// +----------+---------------------+------------------+

bool CanUseFxcMulOnlyPatternForPow(IRBuilder<>& Builder, Value *x, Value *pow, int32_t& powI) {
  // Applicable only when power is a literal.
  if (!isa<ConstantDataVector>(pow) && !isa<ConstantFP>(pow)) {
    return false;
  }

  // Only apply this code gen on splat values.
  if (ConstantDataVector *cdv = dyn_cast<ConstantDataVector>(pow)) {
    if (!hlsl::dxilutil::IsSplat(cdv)) {
      return false;
    }
  }

  APFloat powAPF = isa<ConstantDataVector>(pow) ?
    cast<ConstantDataVector>(pow)->getElementAsAPFloat(0) : // should be a splat value
    cast<ConstantFP>(pow)->getValueAPF();
  APSInt powAPS(32, false);
  bool isExact = false;
  // Try converting float value of power to integer and also check if the float value is exact.
  APFloat::opStatus status = powAPF.convertToInteger(powAPS, APFloat::rmTowardZero, &isExact);
  if (status == APFloat::opStatus::opOK && isExact) {
    powI = powAPS.getExtValue();
    uint32_t powU = abs(powI);
    int setBitCount = 0;
    int maxBitSetPos = -1;
    for (int i = 0; i < 32; i++) {
      if ((powU >> i) & 1) {
        setBitCount++;
        maxBitSetPos = i;
      }
    }

    DXASSERT(maxBitSetPos <= 30, "msb should always be zero.");
    unsigned numElem = isa<ConstantDataVector>(pow) ? x->getType()->getVectorNumElements() : 1;
    int mulOpThreshold = powI < 0 ? numElem + 1 : 2 * numElem + 1;
    int mulOpNeeded = maxBitSetPos + setBitCount - 1;
    return mulOpNeeded <= mulOpThreshold;
  }

  return false;
}

Value *TranslatePowUsingFxcMulOnlyPattern(IRBuilder<>& Builder, Value *x, const int32_t y) {
  uint32_t absY = abs(y);
  // If y is zero then always return 1.
  if (absY == 0) {
    return ConstantFP::get(x->getType(), 1);
  }

  int lastSetPos = -1;
  Value *result = nullptr;
  Value *mul = nullptr;
  for (int i = 0; i < 32; i++) {
    if ((absY >> i) & 1) {
      for (int j = i; j > lastSetPos; j--) {
        if (!mul) {
          mul = x;
        }
        else {
          mul = Builder.CreateFMul(mul, mul);
        }
      }

      result = (result == nullptr) ? mul : Builder.CreateFMul(result, mul);
      lastSetPos = i;
    }
  }

  // Compute reciprocal for negative power values.
  if (y < 0) {
    Value* constOne = ConstantFP::get(x->getType(), 1);
    result = Builder.CreateFDiv(constOne, result);
  }

  return result;
}

Value *TranslatePowImpl(hlsl::OP *hlslOP, IRBuilder<>& Builder, Value *x, Value *y, bool isFXCCompatMode = false) {
  // As applicable implement pow using only mul ops as done by Fxc.
  int32_t p = 0;
  if (CanUseFxcMulOnlyPatternForPow(Builder, x, y, p)) {
    if (isFXCCompatMode) {
    return TranslatePowUsingFxcMulOnlyPattern(Builder, x, p);
    } else if (p == 2) {
      // Only take care 2 for it will not affect register pressure.
      return Builder.CreateFMul(x, x);
    }
  }

  // Default to log-mul-exp pattern if previous scenarios don't apply.
  // t = log(x);
  Value *logX =
    TrivialDxilUnaryOperation(DXIL::OpCode::Log, x, hlslOP, Builder);
  // t = y * t;
  Value *mulY = Builder.CreateFMul(logX, y);
  // pow = exp(t);
  return TrivialDxilUnaryOperation(DXIL::OpCode::Exp, mulY, hlslOP, Builder);
}

Value *TranslateAddUint64(CallInst *CI, IntrinsicOp IOP,
                                 OP::OpCode opcode,
                                 HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  IRBuilder<> Builder(CI);
  Value *val = CI->getArgOperand(HLOperandIndex::kUnaryOpSrc0Idx);
  Type *Ty = val->getType();
  VectorType *VT = dyn_cast<VectorType>(Ty);
  if (!VT) {
    dxilutil::EmitErrorOnInstruction(
        CI, "AddUint64 can only be applied to uint2 and uint4 operands.");
    return UndefValue::get(Ty);
  }

  unsigned size = VT->getNumElements();
  if (size != 2 && size != 4) {
    dxilutil::EmitErrorOnInstruction(
        CI, "AddUint64 can only be applied to uint2 and uint4 operands.");
    return UndefValue::get(Ty);
  }
  Value *op0 = CI->getArgOperand(HLOperandIndex::kBinaryOpSrc0Idx);
  Value *op1 = CI->getArgOperand(HLOperandIndex::kBinaryOpSrc1Idx);

  Value *RetVal = UndefValue::get(Ty);

  Function *AddC = hlslOP->GetOpFunc(DXIL::OpCode::UAddc, helper.i32Ty);
  Value *opArg = Builder.getInt32(static_cast<unsigned>(DXIL::OpCode::UAddc));
  for (unsigned i=0; i<size; i+=2) {
    Value *low0 = Builder.CreateExtractElement(op0, i);
    Value *low1 = Builder.CreateExtractElement(op1, i);
    Value *lowWithC = Builder.CreateCall(AddC, { opArg, low0, low1});
    Value *low = Builder.CreateExtractValue(lowWithC, 0);
    RetVal = Builder.CreateInsertElement(RetVal, low, i);

    Value *carry = Builder.CreateExtractValue(lowWithC, 1);
    // Ext i1 to i32
    carry = Builder.CreateZExt(carry, helper.i32Ty);

    Value *hi0 = Builder.CreateExtractElement(op0, i+1);
    Value *hi1 = Builder.CreateExtractElement(op1, i+1);
    Value *hi = Builder.CreateAdd(hi0, hi1);
    hi = Builder.CreateAdd(hi, carry);
    RetVal = Builder.CreateInsertElement(RetVal, hi, i+1);
  }
  return RetVal;
}


bool IsValidLoadInput(Value *V) {
  // Must be load input.
  // TODO: report this error on front-end
  if (!V || !isa<CallInst>(V)) {
    return false;
  }
  CallInst *CI = cast<CallInst>(V);
  // Must be immediate.
  ConstantInt *opArg =
      cast<ConstantInt>(CI->getArgOperand(DXIL::OperandIndex::kOpcodeIdx));
  DXIL::OpCode op = static_cast<DXIL::OpCode>(opArg->getLimitedValue());
  if (op != DXIL::OpCode::LoadInput) {
    return false;
  }
  return true;
}

// Tunnel through insert/extract element and shuffle to find original source
// of scalar value, or specified element (vecIdx) of vector value.
Value *FindScalarSource(Value *src, unsigned vecIdx = 0) {
  Type *srcTy = src->getType()->getScalarType();
  while (src && !isa<UndefValue>(src)) {
    if (src->getType()->isVectorTy()) {
      if (InsertElementInst *IE = dyn_cast<InsertElementInst>(src)) {
        unsigned curIdx = (unsigned)cast<ConstantInt>(IE->getOperand(2))
          ->getUniqueInteger().getLimitedValue();
        src = IE->getOperand( (curIdx == vecIdx) ? 1 : 0 );
      } else if (ShuffleVectorInst *SV = dyn_cast<ShuffleVectorInst>(src)) {
        int newIdx = SV->getMaskValue(vecIdx);
        if (newIdx < 0)
          return UndefValue::get(srcTy);
        vecIdx = (unsigned)newIdx;
        src = SV->getOperand(0);
        unsigned numElt = src->getType()->getVectorNumElements();
        if (numElt <= vecIdx) {
          vecIdx -= numElt;
          src = SV->getOperand(1);
        }
      } else {
        return UndefValue::get(srcTy);  // Didn't find it.
      }
    } else {
      if (ExtractElementInst *EE = dyn_cast<ExtractElementInst>(src)) {
        vecIdx = (unsigned)cast<ConstantInt>(EE->getIndexOperand())
          ->getUniqueInteger().getLimitedValue();
        src = EE->getVectorOperand();
      } else if (hlsl::dxilutil::IsConvergentMarker(src)) {
        src = hlsl::dxilutil::GetConvergentSource(src);
      } else {
        break;  // Found it.
      }
    }
  }
  return src;
}

// Finds corresponding inputs, calls translation for each, and returns
// resulting vector or scalar.
// Uses functor that takes (inputElemID, rowIdx, colIdx), and returns
// translation for one input scalar.
Value *TranslateEvalHelper(CallInst *CI, Value *val, IRBuilder<> &Builder,
    std::function<Value*(Value*, Value*, Value*)> fnTranslateScalarInput) {
  Type *Ty = CI->getType();
  Value *result = UndefValue::get(Ty);
  if (Ty->isVectorTy()) {
    for (unsigned i = 0; i < Ty->getVectorNumElements(); ++i) {
      Value *InputEl = FindScalarSource(val, i);
      if (!IsValidLoadInput(InputEl)) {
        dxilutil::EmitErrorOnInstruction(CI, "attribute evaluation can only be done "
                                             "on values taken directly from inputs.");
        return result;
      }
      CallInst *loadInput = cast<CallInst>(InputEl);
      Value *inputElemID = loadInput->getArgOperand(DXIL::OperandIndex::kLoadInputIDOpIdx);
      Value *rowIdx = loadInput->getArgOperand(DXIL::OperandIndex::kLoadInputRowOpIdx);
      Value *colIdx = loadInput->getArgOperand(DXIL::OperandIndex::kLoadInputColOpIdx);
      Value *Elt = fnTranslateScalarInput(inputElemID, rowIdx, colIdx);
      result = Builder.CreateInsertElement(result, Elt, i);
    }
  }
  else {
    Value *InputEl = FindScalarSource(val);
    if (!IsValidLoadInput(InputEl)) {
      dxilutil::EmitErrorOnInstruction(CI, "attribute evaluation can only be done "
                                           "on values taken directly from inputs.");
      return result;
    }
    CallInst *loadInput = cast<CallInst>(InputEl);
    Value *inputElemID = loadInput->getArgOperand(DXIL::OperandIndex::kLoadInputIDOpIdx);
    Value *rowIdx = loadInput->getArgOperand(DXIL::OperandIndex::kLoadInputRowOpIdx);
    Value *colIdx = loadInput->getArgOperand(DXIL::OperandIndex::kLoadInputColOpIdx);
    result = fnTranslateScalarInput(inputElemID, rowIdx, colIdx);
  }
  return result;
}

Value *TranslateEvalSample(CallInst *CI, IntrinsicOp IOP, OP::OpCode op,
                           HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  Value *val = CI->getArgOperand(HLOperandIndex::kBinaryOpSrc0Idx);
  Value *sampleIdx = CI->getArgOperand(HLOperandIndex::kBinaryOpSrc1Idx);
  IRBuilder<> Builder(CI);
  OP::OpCode opcode = OP::OpCode::EvalSampleIndex; 
  Value *opArg = hlslOP->GetU32Const((unsigned)opcode);
  Function *evalFunc = hlslOP->GetOpFunc(opcode, CI->getType()->getScalarType());

  return TranslateEvalHelper(CI, val, Builder,
    [&](Value *inputElemID, Value *rowIdx, Value *colIdx) -> Value* {
      return Builder.CreateCall(evalFunc, { opArg, inputElemID, rowIdx, colIdx, sampleIdx });
    }
  );
}

Value *TranslateEvalSnapped(CallInst *CI, IntrinsicOp IOP, OP::OpCode op,
                            HLOperationLowerHelper &helper,
                            HLObjectOperationLowerHelper *pObjHelper,
                            bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  Value *val = CI->getArgOperand(HLOperandIndex::kBinaryOpSrc0Idx);
  Value *offset = CI->getArgOperand(HLOperandIndex::kBinaryOpSrc1Idx);
  IRBuilder<> Builder(CI);
  Value *offsetX = Builder.CreateExtractElement(offset, (uint64_t)0);
  Value *offsetY = Builder.CreateExtractElement(offset, 1);
  OP::OpCode opcode = OP::OpCode::EvalSnapped; 
  Value *opArg = hlslOP->GetU32Const((unsigned)opcode);
  Function *evalFunc = hlslOP->GetOpFunc(opcode, CI->getType()->getScalarType());

  return TranslateEvalHelper(CI, val, Builder,
    [&](Value *inputElemID, Value *rowIdx, Value *colIdx) -> Value* {
      return Builder.CreateCall(evalFunc, { opArg, inputElemID, rowIdx, colIdx, offsetX, offsetY });
    }
  );
}

Value *TranslateEvalCentroid(CallInst *CI, IntrinsicOp IOP, OP::OpCode op,
                             HLOperationLowerHelper &helper,
                             HLObjectOperationLowerHelper *pObjHelper,
                             bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  Value *val = CI->getArgOperand(DXIL::OperandIndex::kUnarySrc0OpIdx);
  IRBuilder<> Builder(CI);
  OP::OpCode opcode = OP::OpCode::EvalCentroid; 
  Value *opArg = hlslOP->GetU32Const((unsigned)opcode);
  Function *evalFunc = hlslOP->GetOpFunc(opcode, CI->getType()->getScalarType());

  return TranslateEvalHelper(CI, val, Builder,
    [&](Value *inputElemID, Value *rowIdx, Value *colIdx) -> Value* {
      return Builder.CreateCall(evalFunc, { opArg, inputElemID, rowIdx, colIdx });
    }
  );
}

Value *TranslateGetAttributeAtVertex(CallInst *CI, IntrinsicOp IOP, OP::OpCode op,
                                     HLOperationLowerHelper &helper,
                                     HLObjectOperationLowerHelper *pObjHelper,
                                     bool &Translated) {
  DXASSERT(op == OP::OpCode::AttributeAtVertex, "Wrong opcode to translate");
  hlsl::OP *hlslOP = &helper.hlslOP;
  IRBuilder<> Builder(CI);
  Value *val = CI->getArgOperand(DXIL::OperandIndex::kBinarySrc0OpIdx);
  Value *vertexIdx = CI->getArgOperand(DXIL::OperandIndex::kBinarySrc1OpIdx);
  Value *vertexI8Idx = Builder.CreateTrunc(vertexIdx, Type::getInt8Ty(CI->getContext()));
  Value *opArg = hlslOP->GetU32Const((unsigned)op);
  Function *evalFunc = hlslOP->GetOpFunc(op, val->getType()->getScalarType());

  return TranslateEvalHelper(CI, val, Builder,
    [&](Value *inputElemID, Value *rowIdx, Value *colIdx) -> Value* {
      return Builder.CreateCall(evalFunc, { opArg, inputElemID, rowIdx, colIdx,  vertexI8Idx });
    }
  );
}

Value *TrivialNoArgOperation(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                             HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  Type *Ty = Type::getVoidTy(CI->getContext());

  Constant *opArg = hlslOP->GetU32Const((unsigned)opcode);
  Value *args[] = {opArg};
  IRBuilder<> Builder(CI);
  Value *dxilOp = TrivialDxilOperation(opcode, args, Ty, Ty, hlslOP, Builder);

  return dxilOp;
}

Value *TrivialNoArgWithRetOperation(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                             HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  Type *Ty = CI->getType();

  Constant *opArg = hlslOP->GetU32Const((unsigned)opcode);
  Value *args[] = {opArg};
  IRBuilder<> Builder(CI);
  Value *dxilOp = TrivialDxilOperation(opcode, args, Ty, Ty, hlslOP, Builder);

  return dxilOp;
}

Value *TranslateGetRTSamplePos(CallInst *CI, IntrinsicOp IOP, OP::OpCode op,
                               HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  OP::OpCode opcode = OP::OpCode::RenderTargetGetSamplePosition;
  IRBuilder<> Builder(CI);

  Type *Ty = Type::getVoidTy(CI->getContext());
  Value *val = CI->getArgOperand(HLOperandIndex::kUnaryOpSrc0Idx);

  Constant *opArg = hlslOP->GetU32Const((unsigned)opcode);
  Value *args[] = {opArg, val};

  Value *samplePos =
      TrivialDxilOperation(opcode, args, Ty, Ty, hlslOP, Builder);

  Value *result = UndefValue::get(CI->getType());
  Value *samplePosX = Builder.CreateExtractValue(samplePos, 0);
  Value *samplePosY = Builder.CreateExtractValue(samplePos, 1);
  result = Builder.CreateInsertElement(result, samplePosX, (uint64_t)0);
  result = Builder.CreateInsertElement(result, samplePosY, 1);
  return result;
}

// val QuadReadLaneAt(val, uint);
Value *TranslateQuadReadLaneAt(CallInst *CI, IntrinsicOp IOP,
                                  OP::OpCode opcode,
                                  HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  Value *refArgs[] = {nullptr, CI->getOperand(1), CI->getOperand(2)};
  return TrivialDxilOperation(DXIL::OpCode::QuadReadLaneAt, refArgs,
                              CI->getOperand(1)->getType(), CI, hlslOP);
}

// Quad intrinsics of the form fn(val,QuadOpKind)->val
Value *TranslateQuadAnyAll(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                           HLOperationLowerHelper &helper,
                           HLObjectOperationLowerHelper *pObjHelper,
                           bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  DXIL::QuadVoteOpKind opKind;
  switch (IOP) {
  case IntrinsicOp::IOP_QuadAll:
    opKind = DXIL::QuadVoteOpKind::All;
    break;
  case IntrinsicOp::IOP_QuadAny:
    opKind = DXIL::QuadVoteOpKind::Any;
    break;
  default:
    llvm_unreachable(
        "QuadAny/QuadAll translation called with wrong isntruction");
  }
  Constant *OpArg = hlslOP->GetI8Const((unsigned)opKind);
  Value *refArgs[] = {nullptr, CI->getOperand(1), OpArg};
  return TrivialDxilOperation(DXIL::OpCode::QuadVote, refArgs,
                              CI->getOperand(1)->getType(), CI, hlslOP);
}

// Wave intrinsics of the form fn(val,QuadOpKind)->val
Value *TranslateQuadReadAcross(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                               HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  DXIL::QuadOpKind opKind;
  switch (IOP) {
  case IntrinsicOp::IOP_QuadReadAcrossX: opKind = DXIL::QuadOpKind::ReadAcrossX; break;
  case IntrinsicOp::IOP_QuadReadAcrossY: opKind = DXIL::QuadOpKind::ReadAcrossY; break;
  default: DXASSERT_NOMSG(IOP == IntrinsicOp::IOP_QuadReadAcrossDiagonal);
  case IntrinsicOp::IOP_QuadReadAcrossDiagonal: opKind = DXIL::QuadOpKind::ReadAcrossDiagonal; break;
  }
  Constant *OpArg = hlslOP->GetI8Const((unsigned)opKind);
  Value *refArgs[] = {nullptr, CI->getOperand(1), OpArg};
  return TrivialDxilOperation(DXIL::OpCode::QuadOp, refArgs,
                              CI->getOperand(1)->getType(), CI, hlslOP);
}

// WaveAllEqual(val<n>)->bool<n>
Value *TranslateWaveAllEqual(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                             HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  Value *src = CI->getArgOperand(HLOperandIndex::kWaveAllEqualValueOpIdx);
  IRBuilder<> Builder(CI);

  Type *Ty = src->getType();
  Type *RetTy = Type::getInt1Ty(CI->getContext());
  if (Ty->isVectorTy())
    RetTy = VectorType::get(RetTy, Ty->getVectorNumElements());

  Constant *opArg = hlslOP->GetU32Const((unsigned)DXIL::OpCode::WaveActiveAllEqual);
  Value *args[] = {opArg, src};

  return TrivialDxilOperation(DXIL::OpCode::WaveActiveAllEqual, args, Ty, RetTy,
                              hlslOP, Builder);
}

// WaveMatch(val<n>)->uint4
Value *TranslateWaveMatch(CallInst *CI, IntrinsicOp IOP, OP::OpCode Opc,
                          HLOperationLowerHelper &Helper,
                          HLObjectOperationLowerHelper *ObjHelper,
                          bool &Translated) {
  hlsl::OP *Op = &Helper.hlslOP;
  IRBuilder<> Builder(CI);

  // Generate a dx.op.waveMatch call for each scalar in the input, and perform
  // a bitwise AND between each result to derive the final bitmask in the case
  // of vector inputs.

  // (1) Collect the list of all scalar inputs (e.g. decompose vectors)
  SmallVector<Value *, 4> ScalarInputs;

  Value *Val = CI->getArgOperand(1);
  Type *ValTy = Val->getType();
  Type *EltTy = ValTy->getScalarType();

  if (ValTy->isVectorTy()) {
    for (uint64_t i = 0, e = ValTy->getVectorNumElements(); i != e; ++i) {
      Value *Elt = Builder.CreateExtractElement(Val, i);
      ScalarInputs.push_back(Elt);
    }
  } else {
    ScalarInputs.push_back(Val);
  }

  Value *Res = nullptr;
  Constant *OpcArg = Op->GetU32Const((unsigned)DXIL::OpCode::WaveMatch);
  Value *Fn = Op->GetOpFunc(OP::OpCode::WaveMatch, EltTy);

  // (2) For each scalar, emit a call to dx.op.waveMatch. If this is not the
  // first scalar, then AND the result with the accumulator.
  for (unsigned i = 0, e = ScalarInputs.size(); i != e; ++i) {
    Value *Args[] = { OpcArg, ScalarInputs[i] };
    Value *Call = Builder.CreateCall(Fn, Args);

    if (Res) {
      // Generate bitwise AND of the components
      for (unsigned j = 0; j != 4; ++j) {
        Value *ResVal = Builder.CreateExtractValue(Res, j);
        Value *CallVal = Builder.CreateExtractValue(Call, j);
        Value *And = Builder.CreateAnd(ResVal, CallVal);
        Res = Builder.CreateInsertValue(Res, And, j);
      }
    } else {
      Res = Call;
    }
  }

  // (3) Convert the final aggregate into a vector to make the types match
  Value *ResVec = UndefValue::get(CI->getType());
  for (unsigned i = 0; i != 4; ++i) {
    Value *Elt = Builder.CreateExtractValue(Res, i);
    ResVec = Builder.CreateInsertElement(ResVec, Elt, i);
  }

  return ResVec;
}

// Wave intrinsics of the form fn(valA)->valB, where no overloading takes place
Value *TranslateWaveA2B(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                        HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  Value *refArgs[] = {nullptr, CI->getOperand(1)};
  return TrivialDxilOperation(opcode, refArgs, helper.voidTy, CI, hlslOP);
}
// Wave ballot intrinsic.
Value *TranslateWaveBallot(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
  HLOperationLowerHelper &helper, HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  // The high-level operation is uint4 ballot(i1).
  // The DXIL operation is struct.u4 ballot(i1).
  // To avoid updating users with more than a simple replace, we translate into
  // a call into struct.u4, then reassemble the vector.
  // Scalarization and constant propagation take care of cleanup.
  IRBuilder<> B(CI);

  // Make the DXIL call itself.
  hlsl::OP *hlslOP = &helper.hlslOP;
  Constant *opArg = hlslOP->GetU32Const((unsigned)opcode);
  Value *refArgs[] = { opArg, CI->getOperand(1) };
  Function *dxilFunc = hlslOP->GetOpFunc(opcode, Type::getVoidTy(CI->getContext()));
  Value *dxilVal = B.CreateCall(dxilFunc, refArgs, hlslOP->GetOpCodeName(opcode));

  // Assign from the call results into a vector.
  Type *ResTy = CI->getType();
  DXASSERT_NOMSG(ResTy->isVectorTy() && ResTy->getVectorNumElements() == 4);
  DXASSERT_NOMSG(dxilVal->getType()->isStructTy() &&
                 dxilVal->getType()->getNumContainedTypes() == 4);

  // 'x' component is the first vector element, highest bits.
  Value *ResVal = llvm::UndefValue::get(ResTy);
  for (unsigned Idx = 0; Idx < 4; ++Idx) {
    ResVal = B.CreateInsertElement(
        ResVal, B.CreateExtractValue(dxilVal, ArrayRef<unsigned>(Idx)), Idx);
  }

  return ResVal;
}

static bool WaveIntrinsicNeedsSign(OP::OpCode opcode) {
  return opcode == OP::OpCode::WaveActiveOp ||
         opcode == OP::OpCode::WavePrefixOp;
}

static unsigned WaveIntrinsicToSignedOpKind(IntrinsicOp IOP) {
  if (IOP == IntrinsicOp::IOP_WaveActiveUMax ||
      IOP == IntrinsicOp::IOP_WaveActiveUMin ||
      IOP == IntrinsicOp::IOP_WaveActiveUSum ||
      IOP == IntrinsicOp::IOP_WaveActiveUProduct ||
      IOP == IntrinsicOp::IOP_WaveMultiPrefixUProduct ||
      IOP == IntrinsicOp::IOP_WaveMultiPrefixUSum ||
      IOP == IntrinsicOp::IOP_WavePrefixUSum ||
      IOP == IntrinsicOp::IOP_WavePrefixUProduct)
    return (unsigned)DXIL::SignedOpKind::Unsigned;
  return (unsigned)DXIL::SignedOpKind::Signed;
}

static unsigned WaveIntrinsicToOpKind(IntrinsicOp IOP) {
  switch (IOP) {
  // Bit operations.
  case IntrinsicOp::IOP_WaveActiveBitOr:
    return (unsigned)DXIL::WaveBitOpKind::Or;
  case IntrinsicOp::IOP_WaveActiveBitAnd:
    return (unsigned)DXIL::WaveBitOpKind::And;
  case IntrinsicOp::IOP_WaveActiveBitXor:
    return (unsigned)DXIL::WaveBitOpKind::Xor;
  // Prefix operations.
  case IntrinsicOp::IOP_WavePrefixSum:
  case IntrinsicOp::IOP_WavePrefixUSum:
    return (unsigned)DXIL::WaveOpKind::Sum;
  case IntrinsicOp::IOP_WavePrefixProduct:
  case IntrinsicOp::IOP_WavePrefixUProduct:
    return (unsigned)DXIL::WaveOpKind::Product;
    // Numeric operations.
  case IntrinsicOp::IOP_WaveActiveMax:
  case IntrinsicOp::IOP_WaveActiveUMax:
    return (unsigned)DXIL::WaveOpKind::Max;
  case IntrinsicOp::IOP_WaveActiveMin:
  case IntrinsicOp::IOP_WaveActiveUMin:
    return (unsigned)DXIL::WaveOpKind::Min;
  case IntrinsicOp::IOP_WaveActiveSum:
  case IntrinsicOp::IOP_WaveActiveUSum:
    return (unsigned)DXIL::WaveOpKind::Sum;
  case IntrinsicOp::IOP_WaveActiveProduct:
  case IntrinsicOp::IOP_WaveActiveUProduct:
  // MultiPrefix operations
  case IntrinsicOp::IOP_WaveMultiPrefixBitAnd:
    return (unsigned)DXIL::WaveMultiPrefixOpKind::And;
  case IntrinsicOp::IOP_WaveMultiPrefixBitOr:
    return (unsigned)DXIL::WaveMultiPrefixOpKind::Or;
  case IntrinsicOp::IOP_WaveMultiPrefixBitXor:
    return (unsigned)DXIL::WaveMultiPrefixOpKind::Xor;
  case IntrinsicOp::IOP_WaveMultiPrefixProduct:
  case IntrinsicOp::IOP_WaveMultiPrefixUProduct:
    return (unsigned)DXIL::WaveMultiPrefixOpKind::Product;
  case IntrinsicOp::IOP_WaveMultiPrefixSum:
  case IntrinsicOp::IOP_WaveMultiPrefixUSum:
    return (unsigned)DXIL::WaveMultiPrefixOpKind::Sum;
  default:
    DXASSERT(IOP == IntrinsicOp::IOP_WaveActiveProduct ||
             IOP == IntrinsicOp::IOP_WaveActiveUProduct,
             "else caller passed incorrect value");
    return (unsigned)DXIL::WaveOpKind::Product;
  }
}

// Wave intrinsics of the form fn(valA)->valA
Value *TranslateWaveA2A(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                        HLOperationLowerHelper &helper,
                        HLObjectOperationLowerHelper *pObjHelper,
                        bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;

  Constant *kindValInt = hlslOP->GetI8Const(WaveIntrinsicToOpKind(IOP));
  Constant *signValInt = hlslOP->GetI8Const(WaveIntrinsicToSignedOpKind(IOP));
  Value *refArgs[] = {nullptr, CI->getOperand(1), kindValInt, signValInt};
  unsigned refArgCount = _countof(refArgs);
  if (!WaveIntrinsicNeedsSign(opcode))
    refArgCount--;
  return TrivialDxilOperation(opcode,
                              llvm::ArrayRef<Value *>(refArgs, refArgCount),
                              CI->getOperand(1)->getType(), CI, hlslOP);
}

// WaveMultiPrefixOP(val<n>, mask) -> val<n>
Value *TranslateWaveMultiPrefix(CallInst *CI, IntrinsicOp IOP, OP::OpCode Opc,
                                HLOperationLowerHelper &Helper,
                                HLObjectOperationLowerHelper *ObjHelper,
                                bool &Translated) {
  hlsl::OP *Op = &Helper.hlslOP;

  Constant *KindValInt = Op->GetI8Const(WaveIntrinsicToOpKind(IOP));
  Constant *SignValInt = Op->GetI8Const(WaveIntrinsicToSignedOpKind(IOP));

  // Decompose mask into scalars
  IRBuilder<> Builder(CI);
  Value *Mask = CI->getArgOperand(2);
  Value *Mask0 = Builder.CreateExtractElement(Mask, (uint64_t)0);
  Value *Mask1 = Builder.CreateExtractElement(Mask, (uint64_t)1);
  Value *Mask2 = Builder.CreateExtractElement(Mask, (uint64_t)2);
  Value *Mask3 = Builder.CreateExtractElement(Mask, (uint64_t)3);

  Value *Args[] = { nullptr, CI->getOperand(1),
                    Mask0, Mask1, Mask2, Mask3, KindValInt, SignValInt };

  return TrivialDxilOperation(Opc, Args, CI->getOperand(1)->getType(), CI, Op);
}

// WaveMultiPrefixBitCount(i1, mask) -> i32
Value *TranslateWaveMultiPrefixBitCount(CallInst *CI, IntrinsicOp IOP,
                                        OP::OpCode Opc,
                                        HLOperationLowerHelper &Helper,
                                        HLObjectOperationLowerHelper *ObjHelper,
                                        bool &Translated) {
  hlsl::OP *Op = &Helper.hlslOP;

  // Decompose mask into scalars
  IRBuilder<> Builder(CI);
  Value *Mask = CI->getArgOperand(2);
  Value *Mask0 = Builder.CreateExtractElement(Mask, (uint64_t)0);
  Value *Mask1 = Builder.CreateExtractElement(Mask, (uint64_t)1);
  Value *Mask2 = Builder.CreateExtractElement(Mask, (uint64_t)2);
  Value *Mask3 = Builder.CreateExtractElement(Mask, (uint64_t)3);

  Value *Args[] = { nullptr, CI->getOperand(1), Mask0, Mask1, Mask2, Mask3 };

  return TrivialDxilOperation(Opc, Args, Helper.voidTy, CI, Op);
}

// Wave intrinsics of the form fn()->val
Value *TranslateWaveToVal(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                          HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  Value *refArgs[] = {nullptr};
  return TrivialDxilOperation(opcode, refArgs, helper.voidTy, CI, hlslOP);
}

// Wave intrinsics of the form fn(val,lane)->val
Value *TranslateWaveReadLaneAt(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                               HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  Value *refArgs[] = {nullptr, CI->getOperand(1), CI->getOperand(2)};
  return TrivialDxilOperation(DXIL::OpCode::WaveReadLaneAt, refArgs,
                              CI->getOperand(1)->getType(), CI, hlslOP);
}

// Wave intrinsics of the form fn(val)->val
Value *TranslateWaveReadLaneFirst(CallInst *CI, IntrinsicOp IOP,
                                  OP::OpCode opcode,
                                  HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  Value *refArgs[] = {nullptr, CI->getOperand(1)};
  return TrivialDxilOperation(DXIL::OpCode::WaveReadLaneFirst, refArgs,
                              CI->getOperand(1)->getType(), CI, hlslOP);
}

Value *TranslateAbs(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                    HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  Type *pOverloadTy = CI->getType()->getScalarType();
  if (pOverloadTy->isFloatingPointTy()) {
    Value *refArgs[] = {nullptr, CI->getOperand(1)};
    return TrivialDxilOperation(DXIL::OpCode::FAbs, refArgs, CI->getType(), CI,
                                hlslOP);
  } else {
    Value *src = CI->getArgOperand(HLOperandIndex::kUnaryOpSrc0Idx);
    IRBuilder<> Builder(CI);
    Value *neg = Builder.CreateNeg(src);
    return TrivialDxilBinaryOperation(DXIL::OpCode::IMax, src, neg, hlslOP,
                                      Builder);
  }
}

Value *TranslateUAbs(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
  HLOperationLowerHelper &helper, HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  return CI->getOperand(HLOperandIndex::kUnaryOpSrc0Idx); // No-op
}

Value *GenerateCmpNEZero(Value *val, IRBuilder<> Builder) {
  Type *Ty = val->getType();
  Type *EltTy = Ty->getScalarType();

  Constant *zero = nullptr;
  if (EltTy->isFloatingPointTy())
    zero = ConstantFP::get(EltTy, 0);
  else
    zero = ConstantInt::get(EltTy, 0);

  if (Ty != EltTy) {
    zero = ConstantVector::getSplat(Ty->getVectorNumElements(), zero);
  }

  if (EltTy->isFloatingPointTy())
    return Builder.CreateFCmpUNE(val, zero);
  else
    return Builder.CreateICmpNE(val, zero);
}

Value *TranslateAllForValue(Value *val, IRBuilder<> &Builder) {
  Value *cond = GenerateCmpNEZero(val, Builder);

  Type *Ty = val->getType();
  Type *EltTy = Ty->getScalarType();

  if (Ty != EltTy) {
    Value *Result = Builder.CreateExtractElement(cond, (uint64_t)0);
    for (unsigned i = 1; i < Ty->getVectorNumElements(); i++) {
      Value *Elt = Builder.CreateExtractElement(cond, i);
      Result = Builder.CreateAnd(Result, Elt);
    }
    return Result;
  } else
    return cond;
}

Value *TranslateAll(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                    HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  Value *val = CI->getArgOperand(HLOperandIndex::kUnaryOpSrc0Idx);
  IRBuilder<> Builder(CI);
  return TranslateAllForValue(val, Builder);
}

Value *TranslateAny(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                    HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  Value *val = CI->getArgOperand(HLOperandIndex::kUnaryOpSrc0Idx);

  IRBuilder<> Builder(CI);

  Value *cond = GenerateCmpNEZero(val, Builder);

  Type *Ty = val->getType();
  Type *EltTy = Ty->getScalarType();

  if (Ty != EltTy) {
    Value *Result = Builder.CreateExtractElement(cond, (uint64_t)0);
    for (unsigned i = 1; i < Ty->getVectorNumElements(); i++) {
      Value *Elt = Builder.CreateExtractElement(cond, i);
      Result = Builder.CreateOr(Result, Elt);
    }
    return Result;
  } else
    return cond;
}

Value *TranslateBitcast(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                        HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  Type *Ty = CI->getType();
  Value *op = CI->getArgOperand(HLOperandIndex::kUnaryOpSrc0Idx);
  IRBuilder<> Builder(CI);
  return Builder.CreateBitCast(op, Ty);
}

Value *TranslateDoubleAsUint(Value *x, Value *lo, Value *hi,
                             IRBuilder<> &Builder, hlsl::OP *hlslOP) {
  Type *Ty = x->getType();
  Type *outTy = lo->getType()->getPointerElementType();
  DXIL::OpCode opcode = DXIL::OpCode::SplitDouble;

  Function *dxilFunc = hlslOP->GetOpFunc(opcode, Ty->getScalarType());
  Value *opArg = hlslOP->GetU32Const(static_cast<unsigned>(opcode));


  if (Ty->isVectorTy()) {
    Value *retValLo = llvm::UndefValue::get(outTy);
    Value *retValHi = llvm::UndefValue::get(outTy);
    unsigned vecSize = Ty->getVectorNumElements();

    for (unsigned i = 0; i < vecSize; i++) {
      Value *Elt = Builder.CreateExtractElement(x, i);
      Value *EltOP = Builder.CreateCall(dxilFunc, {opArg, Elt},
                                        hlslOP->GetOpCodeName(opcode));
      Value *EltLo = Builder.CreateExtractValue(EltOP, 0);
      retValLo = Builder.CreateInsertElement(retValLo, EltLo, i);
      Value *EltHi = Builder.CreateExtractValue(EltOP, 1);
      retValHi = Builder.CreateInsertElement(retValHi, EltHi, i);
    }
    Builder.CreateStore(retValLo, lo);
    Builder.CreateStore(retValHi, hi);
  } else {
    Value *retVal =
        Builder.CreateCall(dxilFunc, {opArg, x}, hlslOP->GetOpCodeName(opcode));
    Value *retValLo = Builder.CreateExtractValue(retVal, 0);
    Value *retValHi = Builder.CreateExtractValue(retVal, 1);
    Builder.CreateStore(retValLo, lo);
    Builder.CreateStore(retValHi, hi);
  }

  return nullptr;
}

Value *TranslateAsUint(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                       HLOperationLowerHelper &helper,
                       HLObjectOperationLowerHelper *pObjHelper,
                       bool &Translated) {
  if (CI->getNumArgOperands() == 2) {
    return TranslateBitcast(CI, IOP, opcode, helper, pObjHelper, Translated);
  } else {
    DXASSERT_NOMSG(CI->getNumArgOperands() == 4);
    hlsl::OP *hlslOP = &helper.hlslOP;
    Value *x = CI->getArgOperand(HLOperandIndex::kTrinaryOpSrc0Idx);
    DXASSERT_NOMSG(x->getType()->getScalarType()->isDoubleTy());
    Value *lo = CI->getArgOperand(HLOperandIndex::kTrinaryOpSrc1Idx);
    Value *hi = CI->getArgOperand(HLOperandIndex::kTrinaryOpSrc2Idx);
    IRBuilder<> Builder(CI);
    return TranslateDoubleAsUint(x, lo, hi, Builder, hlslOP);
  }
}

Value *TranslateAsDouble(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                        HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  Value *x = CI->getArgOperand(HLOperandIndex::kBinaryOpSrc0Idx);
  Value *y = CI->getArgOperand(HLOperandIndex::kBinaryOpSrc1Idx);

  Value *opArg = hlslOP->GetU32Const(static_cast<unsigned>(opcode));
  IRBuilder<> Builder(CI);
  return TrivialDxilOperation(opcode, { opArg, x, y }, CI->getType(), CI->getType(), hlslOP, Builder);
}

Value *TranslateAtan2(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                      HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  Value *y = CI->getArgOperand(HLOperandIndex::kBinaryOpSrc0Idx);
  Value *x = CI->getArgOperand(HLOperandIndex::kBinaryOpSrc1Idx);

  IRBuilder<> Builder(CI);
  Value *tan = Builder.CreateFDiv(y, x);

  Value *atan =
      TrivialDxilUnaryOperation(OP::OpCode::Atan, tan, hlslOP, Builder);
  // Modify atan result based on https://en.wikipedia.org/wiki/Atan2.
  Type *Ty = x->getType();
  Constant *pi = ConstantFP::get(Ty->getScalarType(), M_PI);
  Constant *halfPi = ConstantFP::get(Ty->getScalarType(), M_PI / 2);
  Constant *negHalfPi = ConstantFP::get(Ty->getScalarType(), -M_PI / 2);
  Constant *zero = ConstantFP::get(Ty->getScalarType(), 0);
  if (Ty->isVectorTy()) {
    unsigned vecSize = Ty->getVectorNumElements();
    pi = ConstantVector::getSplat(vecSize, pi);
    halfPi = ConstantVector::getSplat(vecSize, halfPi);
    negHalfPi = ConstantVector::getSplat(vecSize, negHalfPi);
    zero = ConstantVector::getSplat(vecSize, zero);
  }
  Value *atanAddPi = Builder.CreateFAdd(atan, pi);
  Value *atanSubPi = Builder.CreateFSub(atan, pi);

  // x > 0 -> atan.
  Value *result = atan;
  Value *xLt0 = Builder.CreateFCmpOLT(x, zero);
  Value *xEq0 = Builder.CreateFCmpOEQ(x, zero);

  Value *yGe0 = Builder.CreateFCmpOGE(y, zero);
  Value *yLt0 = Builder.CreateFCmpOLT(y, zero);
  // x < 0, y >= 0 -> atan + pi.
  Value *xLt0AndyGe0 = Builder.CreateAnd(xLt0, yGe0);
  result = Builder.CreateSelect(xLt0AndyGe0, atanAddPi, result);

  // x < 0, y < 0 -> atan - pi.
  Value *xLt0AndYLt0 = Builder.CreateAnd(xLt0, yLt0);
  result = Builder.CreateSelect(xLt0AndYLt0, atanSubPi, result);

  // x == 0, y < 0 -> -pi/2
  Value *xEq0AndYLt0 = Builder.CreateAnd(xEq0, yLt0);
  result = Builder.CreateSelect(xEq0AndYLt0, negHalfPi, result);
  // x == 0, y > 0 -> pi/2
  Value *xEq0AndYGe0 = Builder.CreateAnd(xEq0, yGe0);
  result = Builder.CreateSelect(xEq0AndYGe0, halfPi, result);

  return result;
}

Value *TranslateClamp(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                      HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  Type *Ty = CI->getType();
  Type *EltTy = Ty->getScalarType();
  DXIL::OpCode maxOp = DXIL::OpCode::FMax;
  DXIL::OpCode minOp = DXIL::OpCode::FMin;
  if (IOP == IntrinsicOp::IOP_uclamp) {
    maxOp = DXIL::OpCode::UMax;
    minOp = DXIL::OpCode::UMin;
  } else if (EltTy->isIntegerTy()) {
    maxOp = DXIL::OpCode::IMax;
    minOp = DXIL::OpCode::IMin;
  }

  Value *x = CI->getArgOperand(HLOperandIndex::kClampOpXIdx);
  Value *maxVal = CI->getArgOperand(HLOperandIndex::kClampOpMaxIdx);
  Value *minVal = CI->getArgOperand(HLOperandIndex::kClampOpMinIdx);

  IRBuilder<> Builder(CI);
  // min(max(x, minVal), maxVal).
  Value *maxXMinVal =
      TrivialDxilBinaryOperation(maxOp, x, minVal, hlslOP, Builder);
  return TrivialDxilBinaryOperation(minOp, maxXMinVal, maxVal, hlslOP, Builder);
}

Value *TranslateClip(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                     HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  Function *discard =
      hlslOP->GetOpFunc(OP::OpCode::Discard, Type::getVoidTy(CI->getContext()));
  IRBuilder<> Builder(CI);
  Value *cond = nullptr;
  Value *arg = CI->getArgOperand(HLOperandIndex::kUnaryOpSrc0Idx);
  if (VectorType *VT = dyn_cast<VectorType>(arg->getType())) {
    Value *elt = Builder.CreateExtractElement(arg, (uint64_t)0);
    cond = Builder.CreateFCmpOLT(elt, hlslOP->GetFloatConst(0));
    for (unsigned i = 1; i < VT->getNumElements(); i++) {
      Value *elt = Builder.CreateExtractElement(arg, i);
      Value *eltCond = Builder.CreateFCmpOLT(elt, hlslOP->GetFloatConst(0));
      cond = Builder.CreateOr(cond, eltCond);
    }
  } else
    cond = Builder.CreateFCmpOLT(arg, hlslOP->GetFloatConst(0));

  /*If discard condition evaluates to false at compile-time, then
  don't emit the discard instruction.*/
  if (ConstantInt *constCond = dyn_cast<ConstantInt>(cond))
    if (!constCond->getLimitedValue())
      return nullptr;

  Constant *opArg = hlslOP->GetU32Const((unsigned)OP::OpCode::Discard);
  Builder.CreateCall(discard, {opArg, cond});
  return nullptr;
}

Value *TranslateCross(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                      HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  VectorType *VT = cast<VectorType>(CI->getType());
  DXASSERT_NOMSG(VT->getNumElements() == 3);

  Value *op0 = CI->getArgOperand(HLOperandIndex::kBinaryOpSrc0Idx);
  Value *op1 = CI->getArgOperand(HLOperandIndex::kBinaryOpSrc1Idx);

  IRBuilder<> Builder(CI);
  Value *op0_x = Builder.CreateExtractElement(op0, (uint64_t)0);
  Value *op0_y = Builder.CreateExtractElement(op0, 1);
  Value *op0_z = Builder.CreateExtractElement(op0, 2);

  Value *op1_x = Builder.CreateExtractElement(op1, (uint64_t)0);
  Value *op1_y = Builder.CreateExtractElement(op1, 1);
  Value *op1_z = Builder.CreateExtractElement(op1, 2);

  auto MulSub = [&](Value *x0, Value *y0, Value *x1, Value *y1) -> Value * {
    Value *xy = Builder.CreateFMul(x0, y1);
    Value *yx = Builder.CreateFMul(y0, x1);
    return Builder.CreateFSub(xy, yx);
  };

  Value *yz_zy = MulSub(op0_y, op0_z, op1_y, op1_z);
  Value *zx_xz = MulSub(op0_z, op0_x, op1_z, op1_x);
  Value *xy_yx = MulSub(op0_x, op0_y, op1_x, op1_y);

  Value *cross = UndefValue::get(VT);
  cross = Builder.CreateInsertElement(cross, yz_zy, (uint64_t)0);
  cross = Builder.CreateInsertElement(cross, zx_xz, 1);
  cross = Builder.CreateInsertElement(cross, xy_yx, 2);
  return cross;
}

Value *TranslateDegrees(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                        HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  IRBuilder<> Builder(CI);
  Type *Ty = CI->getType();
  Value *val = CI->getArgOperand(HLOperandIndex::kUnaryOpSrc0Idx);
  // 180/pi.
  Constant *toDegreeConst = ConstantFP::get(Ty->getScalarType(), 180 / M_PI);
  if (Ty != Ty->getScalarType()) {
    toDegreeConst =
        ConstantVector::getSplat(Ty->getVectorNumElements(), toDegreeConst);
  }
  return Builder.CreateFMul(toDegreeConst, val);
}

Value *TranslateDst(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                    HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  Value *src0 = CI->getArgOperand(HLOperandIndex::kBinaryOpSrc0Idx);
  Value *src1 = CI->getArgOperand(HLOperandIndex::kBinaryOpSrc1Idx);
  Type *Ty = src1->getType();
  IRBuilder<> Builder(CI);
  Value *Result = UndefValue::get(Ty);
  Constant *oneConst = ConstantFP::get(Ty->getScalarType(), 1);
  // dest.x = 1;
  Result = Builder.CreateInsertElement(Result, oneConst, (uint64_t)0);
  // dest.y = src0.y * src1.y;
  Value *src0_y = Builder.CreateExtractElement(src0, 1);
  Value *src1_y = Builder.CreateExtractElement(src1, 1);
  Value *yMuly = Builder.CreateFMul(src0_y, src1_y);
  Result = Builder.CreateInsertElement(Result, yMuly, 1);
  // dest.z = src0.z;
  Value *src0_z = Builder.CreateExtractElement(src0, 2);
  Result = Builder.CreateInsertElement(Result, src0_z, 2);
  // dest.w = src1.w;
  Value *src1_w = Builder.CreateExtractElement(src1, 3);
  Result = Builder.CreateInsertElement(Result, src1_w, 3);
  return Result;
}

Value *TranslateFirstbitHi(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                           HLOperationLowerHelper &helper,
                           HLObjectOperationLowerHelper *pObjHelper,
                           bool &Translated) {
  Value *firstbitHi =
      TrivialUnaryOperation(CI, IOP, opcode, helper, pObjHelper, Translated);
  // firstbitHi == -1? -1 : (bitWidth-1 -firstbitHi);
  IRBuilder<> Builder(CI);
  Constant *neg1 = Builder.getInt32(-1);
  Value *src = CI->getArgOperand(HLOperandIndex::kUnaryOpSrc0Idx);

  Type *Ty = src->getType();
  IntegerType *EltTy = cast<IntegerType>(Ty->getScalarType());
  Constant *bitWidth = Builder.getInt32(EltTy->getBitWidth()-1);

  if (Ty == Ty->getScalarType()) {
    Value *sub = Builder.CreateSub(bitWidth, firstbitHi);
    Value *cond = Builder.CreateICmpEQ(neg1, firstbitHi);
    return Builder.CreateSelect(cond, neg1, sub);
  } else {
    Value *result = UndefValue::get(CI->getType());
    unsigned vecSize = Ty->getVectorNumElements();
    for (unsigned i = 0; i < vecSize; i++) {
      Value *EltFirstBit = Builder.CreateExtractElement(firstbitHi, i);
      Value *sub = Builder.CreateSub(bitWidth, EltFirstBit);
      Value *cond = Builder.CreateICmpEQ(neg1, EltFirstBit);
      Value *Elt = Builder.CreateSelect(cond, neg1, sub);
      result = Builder.CreateInsertElement(result, Elt, i);
    }
    return result;
  }
}

Value *TranslateFirstbitLo(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                           HLOperationLowerHelper &helper,
                           HLObjectOperationLowerHelper *pObjHelper,
                           bool &Translated) {
  Value *firstbitLo =
      TrivialUnaryOperation(CI, IOP, opcode, helper, pObjHelper, Translated);
  return firstbitLo;
}

Value *TranslateLit(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                    HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  Value *n_dot_l = CI->getArgOperand(HLOperandIndex::kTrinaryOpSrc0Idx);
  Value *n_dot_h = CI->getArgOperand(HLOperandIndex::kTrinaryOpSrc1Idx);
  Value *m = CI->getArgOperand(HLOperandIndex::kTrinaryOpSrc2Idx);
  IRBuilder<> Builder(CI);

  Type *Ty = m->getType();
  Value *Result = UndefValue::get(VectorType::get(Ty, 4));
  // Result = (ambient, diffuse, specular, 1)
  // ambient = 1.
  Constant *oneConst = ConstantFP::get(Ty, 1);
  Result = Builder.CreateInsertElement(Result, oneConst, (uint64_t)0);
  // Result.w = 1.
  Result = Builder.CreateInsertElement(Result, oneConst, 3);
  // diffuse = (n_dot_l < 0) ? 0 : n_dot_l.
  Constant *zeroConst = ConstantFP::get(Ty, 0);
  Value *nlCmp = Builder.CreateFCmpOLT(n_dot_l, zeroConst);
  Value *diffuse = Builder.CreateSelect(nlCmp, zeroConst, n_dot_l);
  Result = Builder.CreateInsertElement(Result, diffuse, 1);
  // specular = ((n_dot_l < 0) || (n_dot_h < 0)) ? 0: (n_dot_h ^ m).
  Value *nhCmp = Builder.CreateFCmpOLT(n_dot_h, zeroConst);
  Value *specCond = Builder.CreateOr(nlCmp, nhCmp);
  bool isFXCCompatMode = CI->getModule()->GetHLModule().GetHLOptions().bFXCCompatMode;
  Value *nhPowM = TranslatePowImpl(&helper.hlslOP, Builder, n_dot_h, m, isFXCCompatMode);
  Value *spec = Builder.CreateSelect(specCond, zeroConst, nhPowM);
  Result = Builder.CreateInsertElement(Result, spec, 2);
  return Result;
}

Value *TranslateRadians(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                        HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  IRBuilder<> Builder(CI);
  Type *Ty = CI->getType();
  Value *val = CI->getArgOperand(HLOperandIndex::kUnaryOpSrc0Idx);
  // pi/180.
  Constant *toRadianConst = ConstantFP::get(Ty->getScalarType(), M_PI / 180);
  if (Ty != Ty->getScalarType()) {
    toRadianConst =
        ConstantVector::getSplat(Ty->getVectorNumElements(), toRadianConst);
  }
  return Builder.CreateFMul(toRadianConst, val);
}

Value *TranslateF16ToF32(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                         HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  IRBuilder<> Builder(CI);

  Value *x = CI->getArgOperand(HLOperandIndex::kUnaryOpSrc0Idx);
  Type *Ty = CI->getType();

  Function *f16tof32 =
      helper.hlslOP.GetOpFunc(opcode, helper.voidTy);
  return TrivialDxilOperation(
      f16tof32, opcode, {Builder.getInt32(static_cast<unsigned>(opcode)), x},
      x->getType(), Ty, &helper.hlslOP, Builder);
}

Value *TranslateF32ToF16(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                         HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  IRBuilder<> Builder(CI);

  Value *x = CI->getArgOperand(HLOperandIndex::kUnaryOpSrc0Idx);
  Type *Ty = CI->getType();

  Function *f32tof16 =
      helper.hlslOP.GetOpFunc(opcode, helper.voidTy);
  return TrivialDxilOperation(
      f32tof16, opcode, {Builder.getInt32(static_cast<unsigned>(opcode)), x},
      x->getType(), Ty, &helper.hlslOP, Builder);
}

Value *TranslateLength(CallInst *CI, Value *val, hlsl::OP *hlslOP) {
  IRBuilder<> Builder(CI);
  if (VectorType *VT = dyn_cast<VectorType>(val->getType())) {
    Value *Elt = Builder.CreateExtractElement(val, (uint64_t)0);
    unsigned size = VT->getNumElements();
    if (size > 1) {
      Value *Sum = Builder.CreateFMul(Elt, Elt);
      for (unsigned i = 1; i < size; i++) {
        Elt = Builder.CreateExtractElement(val, i);
        Value *Mul = Builder.CreateFMul(Elt, Elt);
        Sum = Builder.CreateFAdd(Sum, Mul);
      }
      DXIL::OpCode sqrt = DXIL::OpCode::Sqrt;
      Function *dxilSqrt = hlslOP->GetOpFunc(sqrt, VT->getElementType());
      Value *opArg = hlslOP->GetI32Const((unsigned)sqrt);
      return Builder.CreateCall(dxilSqrt, {opArg, Sum},
                                hlslOP->GetOpCodeName(sqrt));
    } else {
      val = Elt;
    }
  }
  DXIL::OpCode fabs = DXIL::OpCode::FAbs;
  Function *dxilFAbs = hlslOP->GetOpFunc(fabs, val->getType());
  Value *opArg = hlslOP->GetI32Const((unsigned)fabs);
  return Builder.CreateCall(dxilFAbs, {opArg, val},
                            hlslOP->GetOpCodeName(fabs));
}

Value *TranslateLength(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                       HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  Value *val = CI->getArgOperand(HLOperandIndex::kUnaryOpSrc0Idx);
  return TranslateLength(CI, val, hlslOP);
}

Value *TranslateModF(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                     HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  Value *val = CI->getArgOperand(HLOperandIndex::kBinaryOpSrc0Idx);
  Value *outIntPtr = CI->getArgOperand(HLOperandIndex::kBinaryOpSrc1Idx);
  IRBuilder<> Builder(CI);
  Value *intP =
      TrivialDxilUnaryOperation(OP::OpCode::Round_z, val, hlslOP, Builder);
  Value *fracP = Builder.CreateFSub(val, intP);
  Builder.CreateStore(intP, outIntPtr);
  return fracP;
}

Value *TranslateDistance(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                         HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  Value *src0 = CI->getArgOperand(HLOperandIndex::kBinaryOpSrc0Idx);
  Value *src1 = CI->getArgOperand(HLOperandIndex::kBinaryOpSrc1Idx);
  IRBuilder<> Builder(CI);
  Value *sub = Builder.CreateFSub(src0, src1);
  return TranslateLength(CI, sub, hlslOP);
}

Value *TranslateExp(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                    HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  IRBuilder<> Builder(CI);
  Type *Ty = CI->getType();
  Value *val = CI->getArgOperand(HLOperandIndex::kUnaryOpSrc0Idx);
  Constant *log2eConst = ConstantFP::get(Ty->getScalarType(), M_LOG2E);
  if (Ty != Ty->getScalarType()) {
    log2eConst =
        ConstantVector::getSplat(Ty->getVectorNumElements(), log2eConst);
  }
  val = Builder.CreateFMul(log2eConst, val);
  Value *exp = TrivialDxilUnaryOperation(OP::OpCode::Exp, val, hlslOP, Builder);
  return exp;
}

Value *TranslateLog(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                    HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  IRBuilder<> Builder(CI);
  Type *Ty = CI->getType();
  Value *val = CI->getArgOperand(HLOperandIndex::kUnaryOpSrc0Idx);
  Constant *ln2Const = ConstantFP::get(Ty->getScalarType(), M_LN2);
  if (Ty != Ty->getScalarType()) {
    ln2Const = ConstantVector::getSplat(Ty->getVectorNumElements(), ln2Const);
  }
  Value *log = TrivialDxilUnaryOperation(OP::OpCode::Log, val, hlslOP, Builder);

  return Builder.CreateFMul(ln2Const, log);
}

Value *TranslateLog10(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                      HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  IRBuilder<> Builder(CI);
  Type *Ty = CI->getType();
  Value *val = CI->getArgOperand(HLOperandIndex::kUnaryOpSrc0Idx);
  Constant *log2_10Const = ConstantFP::get(Ty->getScalarType(), M_LN2 / M_LN10);
  if (Ty != Ty->getScalarType()) {
    log2_10Const =
        ConstantVector::getSplat(Ty->getVectorNumElements(), log2_10Const);
  }
  Value *log = TrivialDxilUnaryOperation(OP::OpCode::Log, val, hlslOP, Builder);

  return Builder.CreateFMul(log2_10Const, log);
}

Value *TranslateFMod(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                     HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  Value *src0 = CI->getArgOperand(HLOperandIndex::kBinaryOpSrc0Idx);
  Value *src1 = CI->getArgOperand(HLOperandIndex::kBinaryOpSrc1Idx);
  IRBuilder<> Builder(CI);
  Value *div = Builder.CreateFDiv(src0, src1);
  Value *negDiv = Builder.CreateFNeg(div);
  Value *ge = Builder.CreateFCmpOGE(div, negDiv);
  Value *absDiv =
      TrivialDxilUnaryOperation(OP::OpCode::FAbs, div, hlslOP, Builder);
  Value *frc =
      TrivialDxilUnaryOperation(OP::OpCode::Frc, absDiv, hlslOP, Builder);
  Value *negFrc = Builder.CreateFNeg(frc);
  Value *realFrc = Builder.CreateSelect(ge, frc, negFrc);
  return Builder.CreateFMul(realFrc, src1);
}

Value *TranslateFUIBinary(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                          HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  bool isFloat = CI->getType()->getScalarType()->isFloatingPointTy();
  if (isFloat) {
    switch (IOP) {
    case IntrinsicOp::IOP_max:
      opcode = OP::OpCode::FMax;
      break;
    case IntrinsicOp::IOP_min:
    default:
      DXASSERT_NOMSG(IOP == IntrinsicOp::IOP_min);
      opcode = OP::OpCode::FMin;
      break;
    }
  }
  return TrivialBinaryOperation(CI, IOP, opcode, helper, pObjHelper, Translated);
}

Value *TranslateFUITrinary(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                           HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  bool isFloat = CI->getType()->getScalarType()->isFloatingPointTy();
  if (isFloat) {
    switch (IOP) {
    case IntrinsicOp::IOP_mad:
    default:
      DXASSERT_NOMSG(IOP == IntrinsicOp::IOP_mad);
      opcode = OP::OpCode::FMad;
      break;
    }
  }
  return TrivialTrinaryOperation(CI, IOP, opcode, helper, pObjHelper, Translated);
}

Value *TranslateFrexp(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                      HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  Value *val = CI->getArgOperand(HLOperandIndex::kBinaryOpSrc0Idx);
  Value *expPtr = CI->getArgOperand(HLOperandIndex::kBinaryOpSrc1Idx);
  IRBuilder<> Builder(CI);
  Type *i32Ty = Type::getInt32Ty(CI->getContext());
  Constant *exponentMaskConst = ConstantInt::get(i32Ty, 0x7f800000);
  Constant *mantisaMaskConst = ConstantInt::get(i32Ty, 0x007fffff);
  Constant *exponentShiftConst = ConstantInt::get(i32Ty, 23);
  Constant *mantisaOrConst = ConstantInt::get(i32Ty, 0x3f000000);
  Constant *exponentBiasConst = ConstantInt::get(i32Ty, -(int)0x3f000000);
  Constant *zeroVal = hlslOP->GetFloatConst(0);
  // int iVal = asint(val);
  Type *dstTy = i32Ty;
  Type *Ty = val->getType();
  if (Ty->isVectorTy()) {
    unsigned vecSize = Ty->getVectorNumElements();
    dstTy = VectorType::get(i32Ty, vecSize);
    exponentMaskConst = ConstantVector::getSplat(vecSize, exponentMaskConst);
    mantisaMaskConst = ConstantVector::getSplat(vecSize, mantisaMaskConst);
    exponentShiftConst = ConstantVector::getSplat(vecSize, exponentShiftConst);
    mantisaOrConst = ConstantVector::getSplat(vecSize, mantisaOrConst);
    exponentBiasConst = ConstantVector::getSplat(vecSize, exponentBiasConst);
    zeroVal = ConstantVector::getSplat(vecSize, zeroVal);
  }

  // bool ne = val != 0;
  Value *notZero = Builder.CreateFCmpUNE(val, zeroVal);
  notZero = Builder.CreateSExt(notZero, dstTy);

  Value *intVal = Builder.CreateBitCast(val, dstTy);
  // temp = intVal & exponentMask;
  Value *temp = Builder.CreateAnd(intVal, exponentMaskConst);
  // temp = temp + exponentBias;
  temp = Builder.CreateAdd(temp, exponentBiasConst);
  // temp = temp & ne;
  temp = Builder.CreateAnd(temp, notZero);
  // temp = temp >> exponentShift;
  temp = Builder.CreateAShr(temp, exponentShiftConst);
  // exp = float(temp);
  Value *exp = Builder.CreateSIToFP(temp, Ty);
  Builder.CreateStore(exp, expPtr);
  // temp = iVal & mantisaMask;
  temp = Builder.CreateAnd(intVal, mantisaMaskConst);
  // temp = temp | mantisaOr;
  temp = Builder.CreateOr(temp, mantisaOrConst);
  // mantisa = temp & ne;
  Value *mantisa = Builder.CreateAnd(temp, notZero);
  return Builder.CreateBitCast(mantisa, Ty);
}

Value *TranslateLdExp(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                      HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  Value *src0 = CI->getArgOperand(HLOperandIndex::kBinaryOpSrc0Idx);
  Value *src1 = CI->getArgOperand(HLOperandIndex::kBinaryOpSrc1Idx);
  IRBuilder<> Builder(CI);
  Value *exp =
      TrivialDxilUnaryOperation(OP::OpCode::Exp, src1, hlslOP, Builder);
  return Builder.CreateFMul(exp, src0);
}

Value *TranslateFWidth(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                       HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  Value *src = CI->getArgOperand(HLOperandIndex::kUnaryOpSrc0Idx);
  IRBuilder<> Builder(CI);
  Value *ddx =
      TrivialDxilUnaryOperation(OP::OpCode::DerivCoarseX, src, hlslOP, Builder);
  Value *absDdx =
      TrivialDxilUnaryOperation(OP::OpCode::FAbs, ddx, hlslOP, Builder);
  Value *ddy =
      TrivialDxilUnaryOperation(OP::OpCode::DerivCoarseY, src, hlslOP, Builder);
  Value *absDdy =
      TrivialDxilUnaryOperation(OP::OpCode::FAbs, ddy, hlslOP, Builder);
  return Builder.CreateFAdd(absDdx, absDdy);
}

Value *TranslateLerp(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                     HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  // x + s(y-x)
  Value *x = CI->getArgOperand(HLOperandIndex::kLerpOpXIdx);
  Value *y = CI->getArgOperand(HLOperandIndex::kLerpOpYIdx);
  IRBuilder<> Builder(CI);
  Value *ySubx = Builder.CreateFSub(y, x);
  Value *s = CI->getArgOperand(HLOperandIndex::kLerpOpSIdx);
  Value *sMulSub = Builder.CreateFMul(s, ySubx);
  return Builder.CreateFAdd(x, sMulSub);
}

Value *TrivialDotOperation(OP::OpCode opcode, Value *src0,
                           Value *src1, hlsl::OP *hlslOP,
                           IRBuilder<> &Builder) {
  Type *Ty = src0->getType()->getScalarType();
  Function *dxilFunc = hlslOP->GetOpFunc(opcode, Ty);
  Constant *opArg = hlslOP->GetU32Const((unsigned)opcode);

  SmallVector<Value *, 9> args;
  args.emplace_back(opArg);

  unsigned vecSize = src0->getType()->getVectorNumElements();
  for (unsigned i = 0; i < vecSize; i++)
    args.emplace_back(Builder.CreateExtractElement(src0, i));

  for (unsigned i = 0; i < vecSize; i++)
    args.emplace_back(Builder.CreateExtractElement(src1, i));
  Value *dotOP = Builder.CreateCall(dxilFunc, args);

  return dotOP;
}

Value *TranslateIDot(Value *arg0, Value *arg1, unsigned vecSize, hlsl::OP *hlslOP, IRBuilder<> &Builder, bool Unsigned = false) {
  auto madOpCode = Unsigned ? DXIL::OpCode::UMad : DXIL::OpCode::IMad;
  Value *Elt0 = Builder.CreateExtractElement(arg0, (uint64_t)0);
  Value *Elt1 = Builder.CreateExtractElement(arg1, (uint64_t)0);
  Value *Result = Builder.CreateMul(Elt0, Elt1);
  for (unsigned iVecElt = 1; iVecElt < vecSize; ++iVecElt) {
    Elt0 = Builder.CreateExtractElement(arg0, iVecElt);
    Elt1 = Builder.CreateExtractElement(arg1, iVecElt);
    Result = TrivialDxilTrinaryOperation(madOpCode, Elt0, Elt1, Result, hlslOP, Builder);
  }

  return Result;
}

Value *TranslateFDot(Value *arg0, Value *arg1, unsigned vecSize,
                     hlsl::OP *hlslOP, IRBuilder<> &Builder) {
  switch (vecSize) {
  case 2:
    return TrivialDotOperation(OP::OpCode::Dot2, arg0, arg1, hlslOP, Builder);
    break;
  case 3:
    return TrivialDotOperation(OP::OpCode::Dot3, arg0, arg1, hlslOP, Builder);
    break;
  case 4:
    return TrivialDotOperation(OP::OpCode::Dot4, arg0, arg1, hlslOP, Builder);
    break;
  default:
    DXASSERT(vecSize == 1, "wrong vector size");
    {
      Value *vecMul = Builder.CreateFMul(arg0, arg1);
      return Builder.CreateExtractElement(vecMul, (uint64_t)0);
    }
    break;
  }
}

Value *TranslateDot(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                    HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  Value *arg0 = CI->getArgOperand(HLOperandIndex::kBinaryOpSrc0Idx);
  Type *Ty = arg0->getType();
  unsigned vecSize = Ty->getVectorNumElements();
  Value *arg1 = CI->getArgOperand(HLOperandIndex::kBinaryOpSrc1Idx);
  IRBuilder<> Builder(CI);
  if (Ty->getScalarType()->isFloatingPointTy()) {
    return TranslateFDot(arg0, arg1, vecSize, hlslOP, Builder);
  } else {
    return TranslateIDot(arg0, arg1, vecSize, hlslOP, Builder);
  }
}

Value *TranslateNormalize(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                          HLOperationLowerHelper &helper,
                          HLObjectOperationLowerHelper *pObjHelper,
                          bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  Type *Ty = CI->getType();
  Value *op = CI->getArgOperand(HLOperandIndex::kUnaryOpSrc0Idx);
  VectorType *VT = cast<VectorType>(Ty);
  unsigned vecSize = VT->getNumElements();

  IRBuilder<> Builder(CI);
  Value *dot = TranslateFDot(op, op, vecSize, hlslOP, Builder);
  DXIL::OpCode rsqrtOp = DXIL::OpCode::Rsqrt;
  Function *dxilRsqrt = hlslOP->GetOpFunc(rsqrtOp, VT->getElementType());
  Value *rsqrt = Builder.CreateCall(
      dxilRsqrt, {hlslOP->GetI32Const((unsigned)rsqrtOp), dot},
      hlslOP->GetOpCodeName(rsqrtOp));
  Value *vecRsqrt = UndefValue::get(VT);
  for (unsigned i = 0; i < VT->getNumElements(); i++)
    vecRsqrt = Builder.CreateInsertElement(vecRsqrt, rsqrt, i);

  return Builder.CreateFMul(op, vecRsqrt);
}

Value *TranslateReflect(CallInst *CI, IntrinsicOp IOP, OP::OpCode op,
                        HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  //  v = i - 2 * n * dot(i, n).
  IRBuilder<> Builder(CI);
  Value *i = CI->getArgOperand(HLOperandIndex::kReflectOpIIdx);
  Value *n = CI->getArgOperand(HLOperandIndex::kReflectOpNIdx);

  VectorType *VT = cast<VectorType>(i->getType());
  unsigned vecSize = VT->getNumElements();
  Value *dot = TranslateFDot(i, n, vecSize, hlslOP, Builder);
  // 2 * dot (i, n).
  dot = Builder.CreateFMul(ConstantFP::get(dot->getType(), 2.0), dot);
  // 2 * n * dot(i, n).
  Value *vecDot = Builder.CreateVectorSplat(vecSize, dot);
  Value *nMulDot = Builder.CreateFMul(vecDot, n);
  // i - 2 * n * dot(i, n).
  return Builder.CreateFSub(i, nMulDot);
}

Value *TranslateRefract(CallInst *CI, IntrinsicOp IOP, OP::OpCode op,
                        HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  //  d = dot(i, n);
  //  t = 1 - eta * eta * ( 1 - d*d);
  //  cond = t >= 1;
  //  r = eta * i - (eta * d + sqrt(t)) * n;
  //  return cond ? r : 0;
  IRBuilder<> Builder(CI);
  Value *i = CI->getArgOperand(HLOperandIndex::kRefractOpIIdx);
  Value *n = CI->getArgOperand(HLOperandIndex::kRefractOpNIdx);
  Value *eta = CI->getArgOperand(HLOperandIndex::kRefractOpEtaIdx);

  VectorType *VT = cast<VectorType>(i->getType());
  unsigned vecSize = VT->getNumElements();
  Value *dot = TranslateFDot(i, n, vecSize, hlslOP, Builder);
  // eta * eta;
  Value *eta2 = Builder.CreateFMul(eta, eta);
  // d*d;
  Value *dot2 = Builder.CreateFMul(dot, dot);
  Constant *one = ConstantFP::get(eta->getType(), 1);
  Constant *zero = ConstantFP::get(eta->getType(), 0);
  // 1- d*d;
  dot2 = Builder.CreateFSub(one, dot2);
  // eta * eta * (1-d*d);
  eta2 = Builder.CreateFMul(dot2, eta2);
  // t = 1 - eta * eta * ( 1 - d*d);
  Value *t = Builder.CreateFSub(one, eta2);
  // cond = t >= 0;
  Value *cond = Builder.CreateFCmpOGE(t, zero);
  // eta * i;
  Value *vecEta = UndefValue::get(VT);
  for (unsigned i = 0; i < vecSize; i++)
    vecEta = Builder.CreateInsertElement(vecEta, eta, i);
  Value *etaMulI = Builder.CreateFMul(i, vecEta);
  // sqrt(t);
  Value *sqrt = TrivialDxilUnaryOperation(OP::OpCode::Sqrt, t, hlslOP, Builder);
  // eta * d;
  Value *etaMulD = Builder.CreateFMul(eta, dot);
  // eta * d + sqrt(t);
  Value *etaSqrt = Builder.CreateFAdd(etaMulD, sqrt);
  // (eta * d + sqrt(t)) * n;
  Value *vecEtaSqrt = Builder.CreateVectorSplat(vecSize, etaSqrt);
  Value *r = Builder.CreateFMul(vecEtaSqrt, n);
  // r = eta * i - (eta * d + sqrt(t)) * n;
  r = Builder.CreateFSub(etaMulI, r);
  Value *refract =
      Builder.CreateSelect(cond, r, ConstantVector::getSplat(vecSize, zero));
  return refract;
}

Value *TranslateSmoothStep(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                           HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  // s = saturate((x-min)/(max-min)).
  IRBuilder<> Builder(CI);
  Value *minVal = CI->getArgOperand(HLOperandIndex::kSmoothStepOpMinIdx);
  Value *maxVal = CI->getArgOperand(HLOperandIndex::kSmoothStepOpMaxIdx);
  Value *maxSubMin = Builder.CreateFSub(maxVal, minVal);
  Value *x = CI->getArgOperand(HLOperandIndex::kSmoothStepOpXIdx);
  Value *xSubMin = Builder.CreateFSub(x, minVal);
  Value *satVal = Builder.CreateFDiv(xSubMin, maxSubMin);

  Value *s = TrivialDxilUnaryOperation(DXIL::OpCode::Saturate, satVal, hlslOP,
                                       Builder);
  // return s * s *(3-2*s).
  Constant *c2 = ConstantFP::get(CI->getType(),2);
  Constant *c3 = ConstantFP::get(CI->getType(),3);

  Value *sMul2 = Builder.CreateFMul(s, c2);
  Value *result = Builder.CreateFSub(c3, sMul2);
  result = Builder.CreateFMul(s, result);
  result = Builder.CreateFMul(s, result);
  return result;
}

Value *TranslateMSad4(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                      HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  Value *ref = CI->getArgOperand(HLOperandIndex::kTrinaryOpSrc0Idx);
  Value *src = CI->getArgOperand(HLOperandIndex::kTrinaryOpSrc1Idx);
  Value *accum = CI->getArgOperand(HLOperandIndex::kTrinaryOpSrc2Idx);
  Type *Ty = CI->getType();
  IRBuilder<> Builder(CI);
  Value *vecRef = UndefValue::get(Ty);
  for (unsigned i = 0; i < 4; i++)
    vecRef = Builder.CreateInsertElement(vecRef, ref, i);

  Value *srcX = Builder.CreateExtractElement(src, (uint64_t)0);
  Value *srcY = Builder.CreateExtractElement(src, 1);

  Value *byteSrc = UndefValue::get(Ty);
  byteSrc = Builder.CreateInsertElement(byteSrc, srcX, (uint64_t)0);

  // ushr r0.yzw, srcX, l(0, 8, 16, 24)
  // bfi r1.yzw, l(0, 8, 16, 24), l(0, 24, 16, 8), srcX, r0.yyzw
  Value *bfiOpArg =
      hlslOP->GetU32Const(static_cast<unsigned>(DXIL::OpCode::Bfi));

  Value *imm8 = hlslOP->GetU32Const(8);
  Value *imm16 = hlslOP->GetU32Const(16);
  Value *imm24 = hlslOP->GetU32Const(24);

  Ty = ref->getType();
  // Get x[31:8].
  Value *srcXShift = Builder.CreateLShr(srcX, imm8);
  // y[0~7] x[31:8].
  Value *byteSrcElt = TrivialDxilOperation(
      DXIL::OpCode::Bfi, {bfiOpArg, imm8, imm24, srcY, srcXShift}, Ty, Ty,
      hlslOP, Builder);
  byteSrc = Builder.CreateInsertElement(byteSrc, byteSrcElt, 1);
  // Get x[31:16].
  srcXShift = Builder.CreateLShr(srcXShift, imm8);
  // y[0~15] x[31:16].
  byteSrcElt = TrivialDxilOperation(DXIL::OpCode::Bfi,
                                    {bfiOpArg, imm16, imm16, srcY, srcXShift},
                                    Ty, Ty, hlslOP, Builder);
  byteSrc = Builder.CreateInsertElement(byteSrc, byteSrcElt, 2);
  // Get x[31:24].
  srcXShift = Builder.CreateLShr(srcXShift, imm8);
  // y[0~23] x[31:24].
  byteSrcElt = TrivialDxilOperation(DXIL::OpCode::Bfi,
                                    {bfiOpArg, imm24, imm8, srcY, srcXShift},
                                    Ty, Ty, hlslOP, Builder);
  byteSrc = Builder.CreateInsertElement(byteSrc, byteSrcElt, 3);

  // Msad on vecref and byteSrc.
  return TrivialDxilTrinaryOperation(DXIL::OpCode::Msad, vecRef, byteSrc, accum,
                                     hlslOP, Builder);
}

Value *TranslateRCP(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                    HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  Type *Ty = CI->getType();
  Value *op = CI->getArgOperand(HLOperandIndex::kUnaryOpSrc0Idx);
  IRBuilder<> Builder(CI);
  Constant *one = ConstantFP::get(Ty->getScalarType(), 1.0);
  if (Ty != Ty->getScalarType()) {
    one = ConstantVector::getSplat(Ty->getVectorNumElements(), one);
  }
  return Builder.CreateFDiv(one, op);
}

Value *TranslateSign(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                     HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  Value *val = CI->getArgOperand(HLOperandIndex::kUnaryOpSrc0Idx);
  Type *Ty = val->getType();
  bool IsInt = Ty->getScalarType()->isIntegerTy();

  IRBuilder<> Builder(CI);
  Constant *zero = Constant::getNullValue(Ty);
  Value *zeroLtVal = IsInt ? Builder.CreateICmpSLT(zero, val) : Builder.CreateFCmpOLT(zero, val);
  Value *valLtZero = IsInt ? Builder.CreateICmpSLT(val, zero) : Builder.CreateFCmpOLT(val, zero);
  zeroLtVal = Builder.CreateZExt(zeroLtVal, CI->getType());
  valLtZero = Builder.CreateZExt(valLtZero, CI->getType());
  return Builder.CreateSub(zeroLtVal, valLtZero);
}

Value *TranslateUSign(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
  HLOperationLowerHelper &helper, HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  Value *val = CI->getArgOperand(HLOperandIndex::kUnaryOpSrc0Idx);
  Type *Ty = val->getType();

  IRBuilder<> Builder(CI);
  Constant *zero = Constant::getNullValue(Ty);
  Value *nonZero = Builder.CreateICmpNE(val, zero);
  return Builder.CreateZExt(nonZero, CI->getType());
}

Value *TranslateStep(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                     HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  Value *edge = CI->getArgOperand(HLOperandIndex::kBinaryOpSrc0Idx);
  Value *x = CI->getArgOperand(HLOperandIndex::kBinaryOpSrc1Idx);
  Type *Ty = CI->getType();
  IRBuilder<> Builder(CI);

  Constant *one = ConstantFP::get(Ty->getScalarType(), 1.0);
  Constant *zero = ConstantFP::get(Ty->getScalarType(), 0);
  Value *cond = Builder.CreateFCmpOLT(x, edge);

  if (Ty != Ty->getScalarType()) {
    one = ConstantVector::getSplat(Ty->getVectorNumElements(), one);
    zero = ConstantVector::getSplat(Ty->getVectorNumElements(), zero);
  }

  return Builder.CreateSelect(cond, zero, one);
}

Value *TranslatePow(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                    HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  Value *x = CI->getArgOperand(HLOperandIndex::kBinaryOpSrc0Idx);
  Value *y = CI->getArgOperand(HLOperandIndex::kBinaryOpSrc1Idx);
  bool isFXCCompatMode = CI->getModule()->GetHLModule().GetHLOptions().bFXCCompatMode;
  IRBuilder<> Builder(CI);
  return TranslatePowImpl(hlslOP,Builder,x,y,isFXCCompatMode);
}

Value *TranslatePrintf(CallInst *CI, IntrinsicOp IOP, DXIL::OpCode opcode,
                       HLOperationLowerHelper &helper,
                       HLObjectOperationLowerHelper *pObjHelper,
                       bool &Translated) {
  Translated = false;
  CI->getContext().emitError(CI, "use of undeclared identifier 'printf'");
  return nullptr;
}

Value *TranslateFaceforward(CallInst *CI, IntrinsicOp IOP, OP::OpCode op,
                            HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  Type *Ty = CI->getType();

  Value *n = CI->getArgOperand(HLOperandIndex::kTrinaryOpSrc0Idx);
  Value *i = CI->getArgOperand(HLOperandIndex::kTrinaryOpSrc1Idx);
  Value *ng = CI->getArgOperand(HLOperandIndex::kTrinaryOpSrc2Idx);
  IRBuilder<> Builder(CI);

  unsigned vecSize = Ty->getVectorNumElements();
  // -n x sign(dot(i, ng)).
  Value *dotOp = TranslateFDot(i, ng, vecSize, hlslOP, Builder);

  Constant *zero = ConstantFP::get(Ty->getScalarType(), 0);
  Value *dotLtZero = Builder.CreateFCmpOLT(dotOp, zero);

  Value *negN = Builder.CreateFNeg(n);
  Value *faceforward = Builder.CreateSelect(dotLtZero, n, negN);
  return faceforward;
}

Value *TrivialSetMeshOutputCounts(CallInst *CI, IntrinsicOp IOP, OP::OpCode op,
  HLOperationLowerHelper &helper, HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  Value *src0 = CI->getArgOperand(HLOperandIndex::kBinaryOpSrc0Idx);
  Value *src1 = CI->getArgOperand(HLOperandIndex::kBinaryOpSrc1Idx);
  IRBuilder<> Builder(CI);
  Constant *opArg = hlslOP->GetU32Const((unsigned)op);
  Value *args[] = { opArg, src0, src1 };
  Function *dxilFunc = hlslOP->GetOpFunc(op, Type::getVoidTy(CI->getContext()));

  Builder.CreateCall(dxilFunc, args);
  return nullptr;
}

Value *TrivialDispatchMesh(CallInst *CI, IntrinsicOp IOP, OP::OpCode op,
  HLOperationLowerHelper &helper, HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  Value *src0 = CI->getArgOperand(HLOperandIndex::kDispatchMeshOpThreadX);
  Value *src1 = CI->getArgOperand(HLOperandIndex::kDispatchMeshOpThreadY);
  Value *src2 = CI->getArgOperand(HLOperandIndex::kDispatchMeshOpThreadZ);
  Value *src3 = CI->getArgOperand(HLOperandIndex::kDispatchMeshOpPayload);
  IRBuilder<> Builder(CI);
  Constant *opArg = hlslOP->GetU32Const((unsigned)op);
  Value *args[] = { opArg, src0, src1, src2, src3 };
  Function *dxilFunc = hlslOP->GetOpFunc(op, src3->getType());

  Builder.CreateCall(dxilFunc, args);
  return nullptr;
}
}

// MOP intrinsics
namespace {

Value *TranslateGetSamplePosition(CallInst *CI, IntrinsicOp IOP, OP::OpCode op,
                                  HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  Value *handle = CI->getArgOperand(HLOperandIndex::kHandleOpIdx);

  IRBuilder<> Builder(CI);
  Value *sampleIdx =
      CI->getArgOperand(HLOperandIndex::kGetSamplePositionSampleIdxOpIndex);

  OP::OpCode opcode = OP::OpCode::Texture2DMSGetSamplePosition;
  llvm::Constant *opArg = hlslOP->GetU32Const((unsigned)opcode);
  Function *dxilFunc =
      hlslOP->GetOpFunc(opcode, Type::getVoidTy(CI->getContext()));

  Value *args[] = {opArg, handle, sampleIdx};
  Value *samplePos = Builder.CreateCall(dxilFunc, args);

  Value *result = UndefValue::get(CI->getType());
  Value *samplePosX = Builder.CreateExtractValue(samplePos, 0);
  Value *samplePosY = Builder.CreateExtractValue(samplePos, 1);
  result = Builder.CreateInsertElement(result, samplePosX, (uint64_t)0);
  result = Builder.CreateInsertElement(result, samplePosY, 1);
  return result;
}

Value *TranslateGetDimensions(CallInst *CI, IntrinsicOp IOP, OP::OpCode op,
                              HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;

  Value *handle = CI->getArgOperand(HLOperandIndex::kHandleOpIdx);
  DxilResource::Kind RK = pObjHelper->GetRK(handle);

  IRBuilder<> Builder(CI);
  OP::OpCode opcode = OP::OpCode::GetDimensions;
  llvm::Constant *opArg = hlslOP->GetU32Const((unsigned)opcode);
  Function *dxilFunc =
      hlslOP->GetOpFunc(opcode, Type::getVoidTy(CI->getContext()));

  Type *i32Ty = Type::getInt32Ty(CI->getContext());
  Value *mipLevel = UndefValue::get(i32Ty);
  unsigned widthOpIdx = HLOperandIndex::kGetDimensionsMipWidthOpIndex;
  switch (RK) {
  case DxilResource::Kind::Texture1D:
  case DxilResource::Kind::Texture1DArray:
  case DxilResource::Kind::Texture2D:
  case DxilResource::Kind::Texture2DArray:
  case DxilResource::Kind::TextureCube:
  case DxilResource::Kind::TextureCubeArray:
  case DxilResource::Kind::Texture3D: {
    Value *opMipLevel =
        CI->getArgOperand(HLOperandIndex::kGetDimensionsMipLevelOpIndex);
    // mipLevel is in parameter, should not be pointer.
    if (!opMipLevel->getType()->isPointerTy())
      mipLevel = opMipLevel;
    else {
      // No mip level.
      widthOpIdx = HLOperandIndex::kGetDimensionsNoMipWidthOpIndex;
      mipLevel = ConstantInt::get(i32Ty, 0);
    }
  } break;
  default:
    widthOpIdx = HLOperandIndex::kGetDimensionsNoMipWidthOpIndex;
    break;
  }
  Value *args[] = {opArg, handle, mipLevel};
  Value *dims = Builder.CreateCall(dxilFunc, args);

  unsigned dimensionIdx = 0;

  Value *width = Builder.CreateExtractValue(dims, dimensionIdx++);
  Value *widthPtr = CI->getArgOperand(widthOpIdx);
  if (widthPtr->getType()->getPointerElementType()->isFloatingPointTy())
    width = Builder.CreateSIToFP(width,
                                 widthPtr->getType()->getPointerElementType());

  Builder.CreateStore(width, widthPtr);

  if (DXIL::IsStructuredBuffer(RK)) {
    // Set stride.
    Value *stridePtr = CI->getArgOperand(widthOpIdx + 1);
    const DataLayout &DL = helper.dataLayout;
    Value *handle = CI->getArgOperand(HLOperandIndex::kHandleOpIdx);
    Type *bufTy = pObjHelper->GetResourceType(handle);
    Type *bufRetTy = bufTy->getStructElementType(0);
    unsigned stride = DL.getTypeAllocSize(bufRetTy);
    Builder.CreateStore(hlslOP->GetU32Const(stride), stridePtr);
  } else {
    if (widthOpIdx == HLOperandIndex::kGetDimensionsMipWidthOpIndex ||
        // Samples is in w channel too.
        RK == DXIL::ResourceKind::Texture2DMS) {
      // Has mip.
      for (unsigned argIdx = widthOpIdx + 1;
           argIdx < CI->getNumArgOperands() - 1; argIdx++) {
        Value *dim = Builder.CreateExtractValue(dims, dimensionIdx++);
        Value *ptr = CI->getArgOperand(argIdx);
        if (ptr->getType()->getPointerElementType()->isFloatingPointTy())
          dim = Builder.CreateSIToFP(dim,
                                     ptr->getType()->getPointerElementType());
        Builder.CreateStore(dim, ptr);
      }
      // NumOfLevel is in w channel.
      dimensionIdx = 3;
      Value *dim = Builder.CreateExtractValue(dims, dimensionIdx);
      Value *ptr = CI->getArgOperand(CI->getNumArgOperands() - 1);
      if (ptr->getType()->getPointerElementType()->isFloatingPointTy())
        dim =
            Builder.CreateSIToFP(dim, ptr->getType()->getPointerElementType());
      Builder.CreateStore(dim, ptr);
    } else {
      for (unsigned argIdx = widthOpIdx + 1; argIdx < CI->getNumArgOperands();
           argIdx++) {
        Value *dim = Builder.CreateExtractValue(dims, dimensionIdx++);
        Value *ptr = CI->getArgOperand(argIdx);
        if (ptr->getType()->getPointerElementType()->isFloatingPointTy())
          dim = Builder.CreateSIToFP(dim,
                                     ptr->getType()->getPointerElementType());
        Builder.CreateStore(dim, ptr);
      }
    }
  }
  return nullptr;
}

Value *GenerateUpdateCounter(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                             HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  Value *handle = CI->getArgOperand(HLOperandIndex::kHandleOpIdx);

  pObjHelper->MarkHasCounter(handle, helper.i8Ty);

  bool bInc = IOP == IntrinsicOp::MOP_IncrementCounter;
  IRBuilder<> Builder(CI);

  OP::OpCode OpCode = OP::OpCode::BufferUpdateCounter;
  Value *OpCodeArg = hlslOP->GetU32Const((unsigned)OpCode);
  Value *IncVal = hlslOP->GetI8Const(bInc ? 1 : -1);
  // Create BufferUpdateCounter call.
  Value *Args[] = {OpCodeArg, handle, IncVal};

  Function *F =
      hlslOP->GetOpFunc(OpCode, Type::getVoidTy(handle->getContext()));
  return Builder.CreateCall(F, Args);
}

static Value *ScalarizeResRet(Type *RetTy, Value *ResRet, IRBuilder<> &Builder) {
  // Extract value part.
  Value *retVal = llvm::UndefValue::get(RetTy);
  if (RetTy->isVectorTy()) {
    for (unsigned i = 0; i < RetTy->getVectorNumElements(); i++) {
      Value *retComp = Builder.CreateExtractValue(ResRet, i);
      retVal = Builder.CreateInsertElement(retVal, retComp, i);
    }
  } else {
    retVal = Builder.CreateExtractValue(ResRet, 0);
  }
  return retVal;
}

static Value *ScalarizeElements(Type *RetTy, ArrayRef<Value*> Elts, IRBuilder<> &Builder) {
  // Extract value part.
  Value *retVal = llvm::UndefValue::get(RetTy);
  if (RetTy->isVectorTy()) {
    unsigned vecSize = RetTy->getVectorNumElements();
    DXASSERT(vecSize <= Elts.size(), "vector size mismatch");
    for (unsigned i = 0; i < vecSize; i++) {
      Value *retComp = Elts[i];
      retVal = Builder.CreateInsertElement(retVal, retComp, i);
    }
  } else {
    retVal = Elts[0];
  }
  return retVal;
}

void UpdateStatus(Value *ResRet, Value *status, IRBuilder<> &Builder,
                  hlsl::OP *hlslOp) {
  if (status && !isa<UndefValue>(status)) {
    Value *statusVal = Builder.CreateExtractValue(ResRet, DXIL::kResRetStatusIndex);
    Value *checkAccessOp = hlslOp->GetI32Const(
        static_cast<unsigned>(DXIL::OpCode::CheckAccessFullyMapped));
    Function *checkAccessFn = hlslOp->GetOpFunc(
        DXIL::OpCode::CheckAccessFullyMapped, statusVal->getType());
    // CheckAccess on status.
    Value *bStatus =
        Builder.CreateCall(checkAccessFn, {checkAccessOp, statusVal});
    Value *extStatus =
        Builder.CreateZExt(bStatus, Type::getInt32Ty(status->getContext()));
    Builder.CreateStore(extStatus, status);
  }
}

Value *SplatToVector(Value *Elt, Type *DstTy, IRBuilder<> &Builder) {
  Value *Result = UndefValue::get(DstTy);
  for (unsigned i = 0; i < DstTy->getVectorNumElements(); i++)
    Result = Builder.CreateInsertElement(Result, Elt, i);
  return Result;
}

Value *TranslateMul(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
  HLOperationLowerHelper &helper, HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {

  hlsl::OP *hlslOP = &helper.hlslOP;
  Value *arg0 = CI->getArgOperand(HLOperandIndex::kBinaryOpSrc0Idx);
  Value *arg1 = CI->getArgOperand(HLOperandIndex::kBinaryOpSrc1Idx);
  Type *arg0Ty = arg0->getType();
  Type *arg1Ty = arg1->getType();
  IRBuilder<> Builder(CI);

  if (arg0Ty->isVectorTy()) {
    if (arg1Ty->isVectorTy()) {
      // mul(vector, vector) == dot(vector, vector)
      unsigned vecSize = arg0Ty->getVectorNumElements();
      if (arg0Ty->getScalarType()->isFloatingPointTy()) {
        return TranslateFDot(arg0, arg1, vecSize, hlslOP, Builder);
      }
      else {
        return TranslateIDot(arg0, arg1, vecSize, hlslOP, Builder, IOP == IntrinsicOp::IOP_umul);
      }
    }
    else {
      // mul(vector, scalar) == vector * scalar-splat
      arg1 = SplatToVector(arg1, arg0Ty, Builder);
    }
  }
  else {
    if (arg1Ty->isVectorTy()) {
      // mul(scalar, vector) == scalar-splat * vector
      arg0 = SplatToVector(arg0, arg1Ty, Builder);
    }
    // else mul(scalar, scalar) == scalar * scalar;
  }

  // create fmul/mul for the pair of vectors or scalars
  if (arg0Ty->getScalarType()->isFloatingPointTy()) {
    return Builder.CreateFMul(arg0, arg1);
  }
  else {
    return Builder.CreateMul(arg0, arg1);
  }
}

// Sample intrinsics.
struct SampleHelper {
  SampleHelper(CallInst *CI, OP::OpCode op, HLObjectOperationLowerHelper *pObjHelper);

  OP::OpCode opcode = OP::OpCode::NumOpCodes;
  DXIL::ResourceKind resourceKind = DXIL::ResourceKind::Invalid;
  Value *sampledTexHandle = nullptr;
  Value *texHandle = nullptr;
  Value *samplerHandle = nullptr;
  static const unsigned kMaxCoordDimensions = 4;
  unsigned coordDimensions = 0;
  Value *coord[kMaxCoordDimensions];
  Value *compareValue = nullptr;
  Value *bias = nullptr;
  Value *lod = nullptr;
  // SampleGrad only.
  static const unsigned kMaxDDXYDimensions = 3;
  Value *ddx[kMaxDDXYDimensions];
  Value *ddy[kMaxDDXYDimensions];
  // Optional.
  static const unsigned kMaxOffsetDimensions = 3;
  unsigned offsetDimensions = 0;
  Value *offset[kMaxOffsetDimensions];
  Value *clamp = nullptr;
  Value *status = nullptr;
  unsigned maxHLOperandRead = 0;
  Value *ReadHLOperand(CallInst *CI, unsigned opIdx) {
    if (CI->getNumArgOperands() > opIdx) {
      maxHLOperandRead = std::max(maxHLOperandRead, opIdx);
      return CI->getArgOperand(opIdx);
    }
    return nullptr;
  }
  void TranslateCoord(CallInst *CI, unsigned coordIdx) {
    Value *coordArg = ReadHLOperand(CI, coordIdx);
    DXASSERT_NOMSG(coordArg);
    DXASSERT(coordArg->getType()->getVectorNumElements() == coordDimensions,
             "otherwise, HL coordinate dimensions mismatch");
    IRBuilder<> Builder(CI);
    for (unsigned i = 0; i < coordDimensions; i++)
      coord[i] = Builder.CreateExtractElement(coordArg, i);
    Value *undefF = UndefValue::get(Type::getFloatTy(CI->getContext()));
    for (unsigned i = coordDimensions; i < kMaxCoordDimensions; i++)
      coord[i] = undefF;
  }
  void TranslateOffset(CallInst *CI, unsigned offsetIdx) {
    IntegerType *i32Ty = Type::getInt32Ty(CI->getContext());
    if (Value *offsetArg = ReadHLOperand(CI, offsetIdx)) {
      DXASSERT(offsetArg->getType()->getVectorNumElements() == offsetDimensions,
               "otherwise, HL coordinate dimensions mismatch");
      IRBuilder<> Builder(CI);
      for (unsigned i = 0; i < offsetDimensions; i++)
        offset[i] = Builder.CreateExtractElement(offsetArg, i);
    } else {
      // Use zeros for offsets when not specified, not undef.
      Value *zero = ConstantInt::get(i32Ty, (uint64_t)0);
      for (unsigned i = 0; i < offsetDimensions; i++)
        offset[i] = zero;
    }
    // Use undef for components that should not be used for this resource dim.
    Value *undefI = UndefValue::get(i32Ty);
    for (unsigned i = offsetDimensions; i < kMaxOffsetDimensions; i++)
      offset[i] = undefI;
  }
  void SetBias(CallInst *CI, unsigned biasIdx) {
    // Clamp bias for immediate.
    bias = ReadHLOperand(CI, biasIdx);
    DXASSERT_NOMSG(bias);
    if (ConstantFP *FP = dyn_cast<ConstantFP>(bias)) {
      float v = FP->getValueAPF().convertToFloat();
      if (v > DXIL::kMaxMipLodBias)
        bias = ConstantFP::get(FP->getType(), DXIL::kMaxMipLodBias);
      if (v < DXIL::kMinMipLodBias)
        bias = ConstantFP::get(FP->getType(), DXIL::kMinMipLodBias);
    }
  }
  void SetLOD(CallInst *CI, unsigned lodIdx) {
    lod = ReadHLOperand(CI, lodIdx);
    DXASSERT_NOMSG(lod);
  }
  void SetCompareValue(CallInst *CI, unsigned cmpIdx) {
    compareValue = ReadHLOperand(CI, cmpIdx);
    DXASSERT_NOMSG(compareValue);
  }
  void SetClamp(CallInst *CI, unsigned clampIdx) {
    if ((clamp = ReadHLOperand(CI, clampIdx))) {
      if (clamp->getType()->isVectorTy()) {
        IRBuilder<> Builder(CI);
        clamp = Builder.CreateExtractElement(clamp, (uint64_t)0);
      }
    } else
      clamp = UndefValue::get(Type::getFloatTy(CI->getContext()));
  }
  void SetStatus(CallInst *CI, unsigned statusIdx) {
    status = ReadHLOperand(CI, statusIdx);
  }
  void SetDDX(CallInst *CI, unsigned ddxIdx) {
    SetDDXY(CI, ddx, ReadHLOperand(CI, ddxIdx));
  }
  void SetDDY(CallInst *CI, unsigned ddyIdx) {
    SetDDXY(CI, ddy, ReadHLOperand(CI, ddyIdx));
  }
  void SetDDXY(CallInst *CI, MutableArrayRef<Value *> ddxy, Value *ddxyArg) {
    DXASSERT_NOMSG(ddxyArg);
    IRBuilder<> Builder(CI);
    unsigned ddxySize = ddxyArg->getType()->getVectorNumElements();
    for (unsigned i = 0; i < ddxySize; i++)
      ddxy[i] = Builder.CreateExtractElement(ddxyArg, i);
    Value *undefF = UndefValue::get(Type::getFloatTy(CI->getContext()));
    for (unsigned i = ddxySize; i < kMaxDDXYDimensions; i++)
      ddxy[i] = undefF;
  }
};

SampleHelper::SampleHelper(
    CallInst *CI, OP::OpCode op, HLObjectOperationLowerHelper *pObjHelper)
    : opcode(op) {

  texHandle = CI->getArgOperand(HLOperandIndex::kHandleOpIdx);
  resourceKind = pObjHelper->GetRK(texHandle);
  if (resourceKind == DXIL::ResourceKind::Invalid) {
    opcode = DXIL::OpCode::NumOpCodes;
    return;
  }

  coordDimensions = opcode == DXIL::OpCode::CalculateLOD ? DxilResource::GetNumDimensionsForCalcLOD(resourceKind)
                                                         : DxilResource::GetNumCoords(resourceKind);
  offsetDimensions = DxilResource::GetNumOffsets(resourceKind);

  const bool bFeedbackOp = hlsl::OP::IsDxilOpFeedback(op);
  sampledTexHandle = bFeedbackOp ? CI->getArgOperand(HLOperandIndex::kWriteSamplerFeedbackSampledArgIndex)
                                 : nullptr;
  const unsigned kSamplerArgIndex = bFeedbackOp ? HLOperandIndex::kWriteSamplerFeedbackSamplerArgIndex
                                                : HLOperandIndex::kSampleSamplerArgIndex;
  samplerHandle = CI->getArgOperand(kSamplerArgIndex);

  const unsigned kCoordArgIdx = bFeedbackOp ? HLOperandIndex::kWriteSamplerFeedbackCoordArgIndex
                                            : HLOperandIndex::kSampleCoordArgIndex;
  TranslateCoord(CI, kCoordArgIdx);

  // TextureCube does not support offsets, shifting each subsequent arg index down by 1
  unsigned cube = (resourceKind == DXIL::ResourceKind::TextureCube ||
                   resourceKind == DXIL::ResourceKind::TextureCubeArray)
                  ? 1 : 0;

  switch (op) {
  case OP::OpCode::Sample:
    TranslateOffset(CI, cube ? HLOperandIndex::kInvalidIdx : HLOperandIndex::kSampleOffsetArgIndex);
    SetClamp(CI, HLOperandIndex::kSampleClampArgIndex - cube);
    SetStatus(CI, HLOperandIndex::kSampleStatusArgIndex - cube);
    break;
  case OP::OpCode::SampleLevel:
    SetLOD(CI, HLOperandIndex::kSampleLLevelArgIndex);
    TranslateOffset(CI, cube ? HLOperandIndex::kInvalidIdx : HLOperandIndex::kSampleLOffsetArgIndex);
    SetStatus(CI, HLOperandIndex::kSampleLStatusArgIndex - cube);
    break;
  case OP::OpCode::SampleBias:
    SetBias(CI, HLOperandIndex::kSampleBBiasArgIndex);
    TranslateOffset(CI, cube ? HLOperandIndex::kInvalidIdx : HLOperandIndex::kSampleBOffsetArgIndex);
    SetClamp(CI, HLOperandIndex::kSampleBClampArgIndex - cube);
    SetStatus(CI, HLOperandIndex::kSampleBStatusArgIndex - cube);
    break;
  case OP::OpCode::SampleCmp:
    SetCompareValue(CI, HLOperandIndex::kSampleCmpCmpValArgIndex);
    TranslateOffset(CI, cube ? HLOperandIndex::kInvalidIdx : HLOperandIndex::kSampleCmpOffsetArgIndex);
    SetClamp(CI, HLOperandIndex::kSampleCmpClampArgIndex - cube);
    SetStatus(CI, HLOperandIndex::kSampleCmpStatusArgIndex - cube);
    break;
  case OP::OpCode::SampleCmpLevel:
    SetCompareValue(CI, HLOperandIndex::kSampleCmpCmpValArgIndex);
    TranslateOffset(CI, cube ? HLOperandIndex::kInvalidIdx : HLOperandIndex::kSampleCmpLOffsetArgIndex);
    SetLOD(CI, HLOperandIndex::kSampleCmpLLevelArgIndex);
    SetStatus(CI, HLOperandIndex::kSampleCmpStatusArgIndex - cube);
    break;
  case OP::OpCode::SampleCmpLevelZero:
    SetCompareValue(CI, HLOperandIndex::kSampleCmpLZCmpValArgIndex);
    TranslateOffset(CI, cube ? HLOperandIndex::kInvalidIdx : HLOperandIndex::kSampleCmpLZOffsetArgIndex);
    SetStatus(CI, HLOperandIndex::kSampleCmpLZStatusArgIndex - cube);
    break;
  case OP::OpCode::SampleGrad:
    SetDDX(CI, HLOperandIndex::kSampleGDDXArgIndex);
    SetDDY(CI, HLOperandIndex::kSampleGDDYArgIndex);
    TranslateOffset(CI, cube ? HLOperandIndex::kInvalidIdx : HLOperandIndex::kSampleGOffsetArgIndex);
    SetClamp(CI, HLOperandIndex::kSampleGClampArgIndex - cube);
    SetStatus(CI, HLOperandIndex::kSampleGStatusArgIndex - cube);
    break;
  case OP::OpCode::CalculateLOD:
    // Only need coord for LOD calculation.
    break;
  case OP::OpCode::WriteSamplerFeedback:
    SetClamp(CI, HLOperandIndex::kWriteSamplerFeedback_ClampArgIndex);
    break;
  case OP::OpCode::WriteSamplerFeedbackBias:
    SetBias(CI, HLOperandIndex::kWriteSamplerFeedbackBias_BiasArgIndex);
    SetClamp(CI, HLOperandIndex::kWriteSamplerFeedbackBias_ClampArgIndex);
    break;
  case OP::OpCode::WriteSamplerFeedbackGrad:
    SetDDX(CI, HLOperandIndex::kWriteSamplerFeedbackGrad_DdxArgIndex);
    SetDDY(CI, HLOperandIndex::kWriteSamplerFeedbackGrad_DdyArgIndex);
    SetClamp(CI, HLOperandIndex::kWriteSamplerFeedbackGrad_ClampArgIndex);
    break;
  case OP::OpCode::WriteSamplerFeedbackLevel:
    SetLOD(CI, HLOperandIndex::kWriteSamplerFeedbackLevel_LodArgIndex);
    break;
  default:
    DXASSERT(0, "invalid opcode for Sample");
    break;
  }
  DXASSERT(maxHLOperandRead == CI->getNumArgOperands() - 1,
           "otherwise, unused HL arguments for Sample op");
}

Value *TranslateCalculateLOD(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                             HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  SampleHelper sampleHelper(CI, OP::OpCode::CalculateLOD, pObjHelper);
  if (sampleHelper.opcode == DXIL::OpCode::NumOpCodes) {
    Translated = false;
    return nullptr;
  }

  bool bClamped = IOP == IntrinsicOp::MOP_CalculateLevelOfDetail;
  IRBuilder<> Builder(CI);
  Value *opArg =
      hlslOP->GetU32Const(static_cast<unsigned>(OP::OpCode::CalculateLOD));
  Value *clamped = hlslOP->GetI1Const(bClamped);

  Value *args[] = {opArg,
                   sampleHelper.texHandle,
                   sampleHelper.samplerHandle,
                   sampleHelper.coord[0],
                   sampleHelper.coord[1],
                   sampleHelper.coord[2],
                   clamped};
  Function *dxilFunc = hlslOP->GetOpFunc(OP::OpCode::CalculateLOD,
                                         Type::getFloatTy(opArg->getContext()));
  Value *LOD = Builder.CreateCall(dxilFunc, args);
  return LOD;
}

Value *TranslateCheckAccess(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                            HLOperationLowerHelper &helper,
                            HLObjectOperationLowerHelper *pObjHelper,
                            bool &Translated) {
  // Translate CheckAccess into uint->bool, later optimization should remove it.
  // Real checkaccess is generated in UpdateStatus.
  IRBuilder<> Builder(CI);
  Value *V = CI->getArgOperand(HLOperandIndex::kUnaryOpSrc0Idx);
  return Builder.CreateTrunc(V, helper.i1Ty);
}

void GenerateDxilSample(CallInst *CI, Function *F, ArrayRef<Value *> sampleArgs,
                        Value *status, hlsl::OP *hlslOp) {
  IRBuilder<> Builder(CI);

  CallInst *call = Builder.CreateCall(F, sampleArgs);

  dxilutil::MigrateDebugValue(CI, call);

  // extract value part
  Value *retVal = ScalarizeResRet(CI->getType(), call, Builder);

  // Replace ret val.
  CI->replaceAllUsesWith(retVal);

  // get status
  if (status) {
    UpdateStatus(call, status, Builder, hlslOp);
  }
}

Value *TranslateSample(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                       HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  SampleHelper sampleHelper(CI, opcode, pObjHelper);

  if (sampleHelper.opcode == DXIL::OpCode::NumOpCodes) {
    Translated = false;
    return nullptr;
  }
  Type *Ty = CI->getType();

  Function *F = hlslOP->GetOpFunc(opcode, Ty->getScalarType());

  Constant *opArg = hlslOP->GetU32Const((unsigned)opcode);

  switch (opcode) {
  case OP::OpCode::Sample: {
    Value *sampleArgs[] = {
        opArg, sampleHelper.texHandle, sampleHelper.samplerHandle,
        // Coord.
        sampleHelper.coord[0], sampleHelper.coord[1], sampleHelper.coord[2],
        sampleHelper.coord[3],
        // Offset.
        sampleHelper.offset[0], sampleHelper.offset[1], sampleHelper.offset[2],
        // Clamp.
        sampleHelper.clamp};
    GenerateDxilSample(CI, F, sampleArgs, sampleHelper.status, hlslOP);
  } break;
  case OP::OpCode::SampleLevel: {
    Value *sampleArgs[] = {
        opArg, sampleHelper.texHandle, sampleHelper.samplerHandle,
        // Coord.
        sampleHelper.coord[0], sampleHelper.coord[1], sampleHelper.coord[2],
        sampleHelper.coord[3],
        // Offset.
        sampleHelper.offset[0], sampleHelper.offset[1], sampleHelper.offset[2],
        // LOD.
        sampleHelper.lod};
    GenerateDxilSample(CI, F, sampleArgs, sampleHelper.status, hlslOP);
  } break;
  case OP::OpCode::SampleGrad: {
    Value *sampleArgs[] = {
        opArg, sampleHelper.texHandle, sampleHelper.samplerHandle,
        // Coord.
        sampleHelper.coord[0], sampleHelper.coord[1], sampleHelper.coord[2],
        sampleHelper.coord[3],
        // Offset.
        sampleHelper.offset[0], sampleHelper.offset[1], sampleHelper.offset[2],
        // Ddx.
        sampleHelper.ddx[0], sampleHelper.ddx[1], sampleHelper.ddx[2],
        // Ddy.
        sampleHelper.ddy[0], sampleHelper.ddy[1], sampleHelper.ddy[2],
        // Clamp.
        sampleHelper.clamp};
    GenerateDxilSample(CI, F, sampleArgs, sampleHelper.status, hlslOP);
  } break;
  case OP::OpCode::SampleBias: {
    Value *sampleArgs[] = {
        opArg, sampleHelper.texHandle, sampleHelper.samplerHandle,
        // Coord.
        sampleHelper.coord[0], sampleHelper.coord[1], sampleHelper.coord[2],
        sampleHelper.coord[3],
        // Offset.
        sampleHelper.offset[0], sampleHelper.offset[1], sampleHelper.offset[2],
        // Bias.
        sampleHelper.bias,
        // Clamp.
        sampleHelper.clamp};
    GenerateDxilSample(CI, F, sampleArgs, sampleHelper.status, hlslOP);
  } break;
  case OP::OpCode::SampleCmp: {
    Value *sampleArgs[] = {
        opArg, sampleHelper.texHandle, sampleHelper.samplerHandle,
        // Coord.
        sampleHelper.coord[0], sampleHelper.coord[1], sampleHelper.coord[2],
        sampleHelper.coord[3],
        // Offset.
        sampleHelper.offset[0], sampleHelper.offset[1], sampleHelper.offset[2],
        // CmpVal.
        sampleHelper.compareValue,
        // Clamp.
        sampleHelper.clamp};
    GenerateDxilSample(CI, F, sampleArgs, sampleHelper.status, hlslOP);
  } break;
  case OP::OpCode::SampleCmpLevel: {
    Value *sampleArgs[] = {
        opArg, sampleHelper.texHandle, sampleHelper.samplerHandle,
        // Coord.
        sampleHelper.coord[0], sampleHelper.coord[1], sampleHelper.coord[2],
        sampleHelper.coord[3],
        // Offset.
        sampleHelper.offset[0], sampleHelper.offset[1], sampleHelper.offset[2],
        // CmpVal.
        sampleHelper.compareValue,
        // LOD.
        sampleHelper.lod};
    GenerateDxilSample(CI, F, sampleArgs, sampleHelper.status, hlslOP);
  } break;
  case OP::OpCode::SampleCmpLevelZero:
  default: {
    DXASSERT(opcode == OP::OpCode::SampleCmpLevelZero, "invalid sample opcode");
    Value *sampleArgs[] = {
        opArg, sampleHelper.texHandle, sampleHelper.samplerHandle,
        // Coord.
        sampleHelper.coord[0], sampleHelper.coord[1], sampleHelper.coord[2],
        sampleHelper.coord[3],
        // Offset.
        sampleHelper.offset[0], sampleHelper.offset[1], sampleHelper.offset[2],
        // CmpVal.
        sampleHelper.compareValue};
    GenerateDxilSample(CI, F, sampleArgs, sampleHelper.status, hlslOP);
  } break;
  }
  // CI is replaced in GenerateDxilSample.
  return nullptr;
}

// Gather intrinsics.
struct GatherHelper {
  enum class GatherChannel {
    GatherAll,
    GatherRed,
    GatherGreen,
    GatherBlue,
    GatherAlpha,
  };

  GatherHelper(CallInst *CI, OP::OpCode op, HLObjectOperationLowerHelper *pObjHelper,
               GatherHelper::GatherChannel ch);

  OP::OpCode opcode;
  Value *texHandle;
  Value *samplerHandle;
  static const unsigned kMaxCoordDimensions = 4;
  Value *coord[kMaxCoordDimensions];
  unsigned channel;
  Value *special; // For CompareValue, Bias, LOD.
  // Optional.
  static const unsigned kMaxOffsetDimensions = 2;
  Value *offset[kMaxOffsetDimensions];
  // For the overload send different offset for each sample.
  // Only save 3 sampleOffsets because use offset for normal overload as first
  // sample offset.
  static const unsigned kSampleOffsetDimensions = 3;
  Value *sampleOffsets[kSampleOffsetDimensions][kMaxOffsetDimensions];
  Value *status;

  bool hasSampleOffsets;

  unsigned maxHLOperandRead = 0;
  Value *ReadHLOperand(CallInst *CI, unsigned opIdx) {
    if (CI->getNumArgOperands() > opIdx) {
      maxHLOperandRead = std::max(maxHLOperandRead, opIdx);
      return CI->getArgOperand(opIdx);
    }
    return nullptr;
  }
  void TranslateCoord(CallInst *CI, unsigned coordIdx,
                      unsigned coordDimensions) {
    Value *coordArg = ReadHLOperand(CI, coordIdx);
    DXASSERT_NOMSG(coordArg);
    DXASSERT(coordArg->getType()->getVectorNumElements() == coordDimensions,
             "otherwise, HL coordinate dimensions mismatch");
    IRBuilder<> Builder(CI);
    for (unsigned i = 0; i < coordDimensions; i++)
      coord[i] = Builder.CreateExtractElement(coordArg, i);
    Value *undefF = UndefValue::get(Type::getFloatTy(CI->getContext()));
    for (unsigned i = coordDimensions; i < kMaxCoordDimensions; i++)
      coord[i] = undefF;
  }
  void SetStatus(CallInst *CI, unsigned statusIdx) {
    status = ReadHLOperand(CI, statusIdx);
  }
  void TranslateOffset(CallInst *CI, unsigned offsetIdx,
                       unsigned offsetDimensions) {
    IntegerType *i32Ty = Type::getInt32Ty(CI->getContext());
    if (Value *offsetArg = ReadHLOperand(CI, offsetIdx)) {
      DXASSERT(offsetArg->getType()->getVectorNumElements() == offsetDimensions,
               "otherwise, HL coordinate dimensions mismatch");
      IRBuilder<> Builder(CI);
      for (unsigned i = 0; i < offsetDimensions; i++)
        offset[i] = Builder.CreateExtractElement(offsetArg, i);
    } else {
      // Use zeros for offsets when not specified, not undef.
      Value *zero = ConstantInt::get(i32Ty, (uint64_t)0);
      for (unsigned i = 0; i < offsetDimensions; i++)
        offset[i] = zero;
    }
    // Use undef for components that should not be used for this resource dim.
    Value *undefI = UndefValue::get(i32Ty);
    for (unsigned i = offsetDimensions; i < kMaxOffsetDimensions; i++)
      offset[i] = undefI;
  }
  void TranslateSampleOffset(CallInst *CI, unsigned offsetIdx,
                             unsigned offsetDimensions) {
    Value *undefI = UndefValue::get(Type::getInt32Ty(CI->getContext()));
    if (CI->getNumArgOperands() >= (offsetIdx + kSampleOffsetDimensions)) {
      hasSampleOffsets = true;
      IRBuilder<> Builder(CI);
      for (unsigned ch = 0; ch < kSampleOffsetDimensions; ch++) {
        Value *offsetArg = ReadHLOperand(CI, offsetIdx + ch);
        for (unsigned i = 0; i < offsetDimensions; i++)
          sampleOffsets[ch][i] = Builder.CreateExtractElement(offsetArg, i);
        for (unsigned i = offsetDimensions; i < kMaxOffsetDimensions; i++)
          sampleOffsets[ch][i] = undefI;
      }
    }
  }
  // Update the offset args for gather with sample offset at sampleIdx.
  void UpdateOffsetInGatherArgs(MutableArrayRef<Value *> gatherArgs,
                                unsigned sampleIdx) {
    unsigned offsetBase = DXIL::OperandIndex::kTextureGatherOffset0OpIdx;
    for (unsigned i = 0; i < kMaxOffsetDimensions; i++)
      // -1 because offset for sample 0 is in GatherHelper::offset.
      gatherArgs[offsetBase + i] = sampleOffsets[sampleIdx - 1][i];
  }
};

GatherHelper::GatherHelper(
    CallInst *CI, OP::OpCode op, HLObjectOperationLowerHelper *pObjHelper,
    GatherHelper::GatherChannel ch)
    : opcode(op), special(nullptr), hasSampleOffsets(false) {

  switch (ch) {
  case GatherChannel::GatherAll:
    channel = 0;
    break;
  case GatherChannel::GatherRed:
    channel = 0;
    break;
  case GatherChannel::GatherGreen:
    channel = 1;
    break;
  case GatherChannel::GatherBlue:
    channel = 2;
    break;
  case GatherChannel::GatherAlpha:
    channel = 3;
    break;
  }

  IRBuilder<> Builder(CI);
  texHandle = CI->getArgOperand(HLOperandIndex::kHandleOpIdx);
  samplerHandle = CI->getArgOperand(HLOperandIndex::kSampleSamplerArgIndex);

  DXIL::ResourceKind RK = pObjHelper->GetRK(texHandle);
  if (RK == DXIL::ResourceKind::Invalid) {
    opcode = DXIL::OpCode::NumOpCodes;
    return;
  }
  unsigned coordSize = DxilResource::GetNumCoords(RK);
  unsigned offsetSize = DxilResource::GetNumOffsets(RK);
  bool cube = RK == DXIL::ResourceKind::TextureCube ||
              RK == DXIL::ResourceKind::TextureCubeArray;

  const unsigned kCoordArgIdx = HLOperandIndex::kSampleCoordArgIndex;
  TranslateCoord(CI, kCoordArgIdx, coordSize);

  switch (op) {
  case OP::OpCode::TextureGather: {
    unsigned statusIdx;
    if (cube) {
      TranslateOffset(CI, HLOperandIndex::kInvalidIdx, offsetSize);
      statusIdx = HLOperandIndex::kGatherCubeStatusArgIndex;
    } else {
      TranslateOffset(CI, HLOperandIndex::kGatherOffsetArgIndex, offsetSize);
      // Gather all don't have sample offset version overload.
      if (ch != GatherChannel::GatherAll)
        TranslateSampleOffset(CI, HLOperandIndex::kGatherSampleOffsetArgIndex,
                              offsetSize);
      statusIdx =
          hasSampleOffsets ? HLOperandIndex::kGatherStatusWithSampleOffsetArgIndex
                           : HLOperandIndex::kGatherStatusArgIndex;
    }
    SetStatus(CI, statusIdx);
  } break;
  case OP::OpCode::TextureGatherCmp: {
    special = ReadHLOperand(CI, HLOperandIndex::kGatherCmpCmpValArgIndex);
    unsigned statusIdx;
    if (cube) {
      TranslateOffset(CI, HLOperandIndex::kInvalidIdx, offsetSize);
      statusIdx = HLOperandIndex::kGatherCmpCubeStatusArgIndex;
    } else {
      TranslateOffset(CI, HLOperandIndex::kGatherCmpOffsetArgIndex, offsetSize);
      // Gather all don't have sample offset version overload.
      if (ch != GatherChannel::GatherAll)
        TranslateSampleOffset(CI, HLOperandIndex::kGatherCmpSampleOffsetArgIndex,
                              offsetSize);
      statusIdx =
          hasSampleOffsets
              ? HLOperandIndex::kGatherCmpStatusWithSampleOffsetArgIndex
              : HLOperandIndex::kGatherCmpStatusArgIndex;
    }
    SetStatus(CI, statusIdx);
  } break;
  case OP::OpCode::TextureGatherRaw: {
    unsigned statusIdx;
    TranslateOffset(CI, HLOperandIndex::kGatherOffsetArgIndex, offsetSize);
    // Gather all don't have sample offset version overload.
    DXASSERT(ch == GatherChannel::GatherAll, "Raw gather must use all channels");
    DXASSERT(!cube, "Raw gather can't be used with cube textures");
    DXASSERT(!hasSampleOffsets, "Raw gather doesn't support individual offsets");
    statusIdx = HLOperandIndex::kGatherStatusArgIndex;
    SetStatus(CI, statusIdx);
  } break;
  default:
    DXASSERT(0, "invalid opcode for Gather");
    break;
  }
  DXASSERT(maxHLOperandRead == CI->getNumArgOperands() - 1,
           "otherwise, unused HL arguments for Sample op");
}

void GenerateDxilGather(CallInst *CI, Function *F,
                        MutableArrayRef<Value *> gatherArgs,
                        GatherHelper &helper, hlsl::OP *hlslOp) {
  IRBuilder<> Builder(CI);

  CallInst *call = Builder.CreateCall(F, gatherArgs);

  dxilutil::MigrateDebugValue(CI, call);

  Value *retVal;
  if (!helper.hasSampleOffsets) {
    // extract value part
    retVal = ScalarizeResRet(CI->getType(), call, Builder);
  } else {
    retVal = UndefValue::get(CI->getType());
    Value *elt = Builder.CreateExtractValue(call, (uint64_t)0);
    retVal = Builder.CreateInsertElement(retVal, elt, (uint64_t)0);

    helper.UpdateOffsetInGatherArgs(gatherArgs, /*sampleIdx*/ 1);
    CallInst *callY = Builder.CreateCall(F, gatherArgs);
    elt = Builder.CreateExtractValue(callY, (uint64_t)1);
    retVal = Builder.CreateInsertElement(retVal, elt, 1);

    helper.UpdateOffsetInGatherArgs(gatherArgs, /*sampleIdx*/ 2);
    CallInst *callZ = Builder.CreateCall(F, gatherArgs);
    elt = Builder.CreateExtractValue(callZ, (uint64_t)2);
    retVal = Builder.CreateInsertElement(retVal, elt, 2);

    helper.UpdateOffsetInGatherArgs(gatherArgs, /*sampleIdx*/ 3);
    CallInst *callW = Builder.CreateCall(F, gatherArgs);
    elt = Builder.CreateExtractValue(callW, (uint64_t)3);
    retVal = Builder.CreateInsertElement(retVal, elt, 3);

    // TODO: UpdateStatus for each gather call.
  }

  // Replace ret val.
  CI->replaceAllUsesWith(retVal);

  // Get status
  if (helper.status) {
    UpdateStatus(call, helper.status, Builder, hlslOp);
  }
}

Value *TranslateGather(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                       HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  GatherHelper::GatherChannel ch = GatherHelper::GatherChannel::GatherAll;
  switch (IOP) {
  case IntrinsicOp::MOP_Gather:
  case IntrinsicOp::MOP_GatherCmp:
  case IntrinsicOp::MOP_GatherRaw:
    ch = GatherHelper::GatherChannel::GatherAll;
    break;
  case IntrinsicOp::MOP_GatherRed:
  case IntrinsicOp::MOP_GatherCmpRed:
    ch = GatherHelper::GatherChannel::GatherRed;
    break;
  case IntrinsicOp::MOP_GatherGreen:
  case IntrinsicOp::MOP_GatherCmpGreen:
    ch = GatherHelper::GatherChannel::GatherGreen;
    break;
  case IntrinsicOp::MOP_GatherBlue:
  case IntrinsicOp::MOP_GatherCmpBlue:
    ch = GatherHelper::GatherChannel::GatherBlue;
    break;
  case IntrinsicOp::MOP_GatherAlpha:
  case IntrinsicOp::MOP_GatherCmpAlpha:
    ch = GatherHelper::GatherChannel::GatherAlpha;
    break;
  default:
    DXASSERT(0, "invalid gather intrinsic");
    break;
  }

  GatherHelper gatherHelper(CI, opcode, pObjHelper, ch);

  if (gatherHelper.opcode == DXIL::OpCode::NumOpCodes) {
    Translated = false;
    return nullptr;
  }
  Type *Ty = CI->getType();

  Function *F = hlslOP->GetOpFunc(gatherHelper.opcode, Ty->getScalarType());

  Constant *opArg = hlslOP->GetU32Const((unsigned)gatherHelper.opcode);
  Value *channelArg = hlslOP->GetU32Const(gatherHelper.channel);

  switch (opcode) {
  case OP::OpCode::TextureGather: {
    Value *gatherArgs[] = {
        opArg, gatherHelper.texHandle, gatherHelper.samplerHandle,
        // Coord.
        gatherHelper.coord[0], gatherHelper.coord[1], gatherHelper.coord[2],
        gatherHelper.coord[3],
        // Offset.
        gatherHelper.offset[0], gatherHelper.offset[1],
        // Channel.
        channelArg};
    GenerateDxilGather(CI, F, gatherArgs, gatherHelper, hlslOP);
  } break;
  case OP::OpCode::TextureGatherCmp: {
    Value *gatherArgs[] = {
        opArg, gatherHelper.texHandle, gatherHelper.samplerHandle,
        // Coord.
        gatherHelper.coord[0], gatherHelper.coord[1], gatherHelper.coord[2],
        gatherHelper.coord[3],
        // Offset.
        gatherHelper.offset[0], gatherHelper.offset[1],
        // Channel.
        channelArg,
        // CmpVal.
        gatherHelper.special};
    GenerateDxilGather(CI, F, gatherArgs, gatherHelper, hlslOP);
  } break;
  case OP::OpCode::TextureGatherRaw: {
    Value *gatherArgs[] = {
        opArg, gatherHelper.texHandle, gatherHelper.samplerHandle,
        // Coord.
        gatherHelper.coord[0], gatherHelper.coord[1], gatherHelper.coord[2],
        gatherHelper.coord[3],
        // Offset.
        gatherHelper.offset[0], gatherHelper.offset[1]};
    GenerateDxilGather(CI, F, gatherArgs, gatherHelper, hlslOP);
    break;
  }
  default:
    DXASSERT(0, "invalid opcode for Gather");
    break;
  }
  // CI is replaced in GenerateDxilGather.
  return nullptr;
}

static Value* TranslateWriteSamplerFeedback(CallInst* CI, IntrinsicOp IOP, OP::OpCode opcode,
                                            HLOperationLowerHelper& helper,
                                            HLObjectOperationLowerHelper* pObjHelper,
                                            bool& Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  SampleHelper sampleHelper(CI, opcode, pObjHelper);

  if (sampleHelper.opcode == DXIL::OpCode::NumOpCodes) {
    Translated = false;
    return nullptr;
  }
  Type *Ty = CI->getType();

  Function *F = hlslOP->GetOpFunc(opcode, Ty->getScalarType());

  Constant *opArg = hlslOP->GetU32Const((unsigned)opcode);

  IRBuilder<> Builder(CI);

  switch (opcode) {
  case OP::OpCode::WriteSamplerFeedback: {
    Value *samplerFeedbackArgs[] = {
        opArg, sampleHelper.texHandle, sampleHelper.sampledTexHandle, sampleHelper.samplerHandle,
        // Coord.
        sampleHelper.coord[0], sampleHelper.coord[1], sampleHelper.coord[2],
        sampleHelper.coord[3],
        // Clamp.
        sampleHelper.clamp};
    return Builder.CreateCall(F, samplerFeedbackArgs);
  } break;
  case OP::OpCode::WriteSamplerFeedbackBias: {
    Value *samplerFeedbackArgs[] = {
        opArg, sampleHelper.texHandle, sampleHelper.sampledTexHandle, sampleHelper.samplerHandle,
        // Coord.
        sampleHelper.coord[0], sampleHelper.coord[1], sampleHelper.coord[2],
        sampleHelper.coord[3],
        // Bias.
        sampleHelper.bias,
        // Clamp.
        sampleHelper.clamp};
    return Builder.CreateCall(F, samplerFeedbackArgs);
  } break;
  case OP::OpCode::WriteSamplerFeedbackGrad: {
    Value *samplerFeedbackArgs[] = {
        opArg, sampleHelper.texHandle, sampleHelper.sampledTexHandle, sampleHelper.samplerHandle,
        // Coord.
        sampleHelper.coord[0], sampleHelper.coord[1], sampleHelper.coord[2],
        sampleHelper.coord[3],
        // Ddx.
        sampleHelper.ddx[0], sampleHelper.ddx[1], sampleHelper.ddx[2],
        // Ddy.
        sampleHelper.ddy[0], sampleHelper.ddy[1], sampleHelper.ddy[2],
        // Clamp.
        sampleHelper.clamp};
    return Builder.CreateCall(F, samplerFeedbackArgs);
  } break;
  case OP::OpCode::WriteSamplerFeedbackLevel: {
    Value *samplerFeedbackArgs[] = {
        opArg, sampleHelper.texHandle, sampleHelper.sampledTexHandle, sampleHelper.samplerHandle,
        // Coord.
        sampleHelper.coord[0], sampleHelper.coord[1], sampleHelper.coord[2],
        sampleHelper.coord[3],
        // LOD.
        sampleHelper.lod};
    return Builder.CreateCall(F, samplerFeedbackArgs);
  } break;
  default:
    DXASSERT(false, "otherwise, unknown SamplerFeedback Op");
    break;
  }
  return nullptr;
}

// Load/Store intrinsics.
struct ResLoadHelper {
  ResLoadHelper(CallInst *CI, DxilResource::Kind RK, DxilResourceBase::Class RC,
                Value *h, IntrinsicOp IOP, bool bForSubscript=false);
  // For double subscript.
  ResLoadHelper(Instruction *ldInst, Value *h, Value *idx, Value *mip)
      : opcode(OP::OpCode::TextureLoad),
        intrinsicOpCode(IntrinsicOp::Num_Intrinsics), handle(h), retVal(ldInst),
        addr(idx), offset(nullptr), status(nullptr), mipLevel(mip) {}
  OP::OpCode opcode;
  IntrinsicOp intrinsicOpCode;
  unsigned dxilMajor;
  unsigned dxilMinor;
  Value *handle;
  Value *retVal;
  Value *addr;
  Value *offset;
  Value *status;
  Value *mipLevel;
};

ResLoadHelper::ResLoadHelper(CallInst *CI, DxilResource::Kind RK,
                             DxilResourceBase::Class RC, Value *hdl, IntrinsicOp IOP, bool bForSubscript)
    : intrinsicOpCode(IOP), handle(hdl), offset(nullptr), status(nullptr) {
  switch (RK) {
  case DxilResource::Kind::RawBuffer:
  case DxilResource::Kind::StructuredBuffer:
    opcode = OP::OpCode::RawBufferLoad;
    break;
  case DxilResource::Kind::TypedBuffer:
    opcode = OP::OpCode::BufferLoad;
    break;
  case DxilResource::Kind::Invalid:
    DXASSERT(0, "invalid resource kind");
    break;
  default:
    opcode = OP::OpCode::TextureLoad;
    break;
  }
  retVal = CI;
  const unsigned kAddrIdx = HLOperandIndex::kBufLoadAddrOpIdx;
  addr = CI->getArgOperand(kAddrIdx);
  unsigned argc = CI->getNumArgOperands();

  if (opcode == OP::OpCode::TextureLoad) {
    // mip at last channel
    unsigned coordSize = DxilResource::GetNumCoords(RK);

    if (RC == DxilResourceBase::Class::SRV) {
      if (bForSubscript) {
        // Use 0 when access by [].
        mipLevel = IRBuilder<>(CI).getInt32(0);
      } else {
        if (coordSize == 1 && !addr->getType()->isVectorTy()) {
          // Use addr when access by Load.
          mipLevel = addr;
        } else {
          mipLevel = IRBuilder<>(CI).CreateExtractElement(addr, coordSize);
        }
      }
    } else {
      // Set mip level to undef for UAV.
      mipLevel = UndefValue::get(Type::getInt32Ty(addr->getContext()));
    }

    if (RC == DxilResourceBase::Class::SRV) {
      unsigned offsetIdx = HLOperandIndex::kTexLoadOffsetOpIdx;
      unsigned statusIdx = HLOperandIndex::kTexLoadStatusOpIdx;
      if (RK == DxilResource::Kind::Texture2DMS ||
          RK == DxilResource::Kind::Texture2DMSArray) {
        offsetIdx = HLOperandIndex::kTex2DMSLoadOffsetOpIdx;
        statusIdx = HLOperandIndex::kTex2DMSLoadStatusOpIdx;
        mipLevel =
            CI->getArgOperand(HLOperandIndex::kTex2DMSLoadSampleIdxOpIdx);
      }

      if (argc > offsetIdx)
        offset = CI->getArgOperand(offsetIdx);

      if (argc > statusIdx)
        status = CI->getArgOperand(statusIdx);
    } else if (RC == DxilResourceBase::Class::UAV &&
               (RK == DxilResource::Kind::Texture2DMS ||
                RK == DxilResource::Kind::Texture2DMSArray)) {
      unsigned statusIdx = HLOperandIndex::kTex2DMSLoadStatusOpIdx;
      mipLevel =
        CI->getArgOperand(HLOperandIndex::kTex2DMSLoadSampleIdxOpIdx);

      if (argc > statusIdx)
        status = CI->getArgOperand(statusIdx);

    } else {
      const unsigned kStatusIdx = HLOperandIndex::kRWTexLoadStatusOpIdx;

      if (argc > kStatusIdx)
        status = CI->getArgOperand(kStatusIdx);
    }
  } else {
    const unsigned kStatusIdx = HLOperandIndex::kBufLoadStatusOpIdx;
    if (argc > kStatusIdx)
      status = CI->getArgOperand(kStatusIdx);
  }
}


void TranslateStructBufSubscript(CallInst *CI, Value *handle, Value *status,
                                 hlsl::OP *OP, HLResource::Kind RK, const DataLayout &DL);

// Create { v0, v1 } from { v0.lo, v0.hi, v1.lo, v1.hi }
void Make64bitResultForLoad(Type *EltTy, ArrayRef<Value *> resultElts32,
                            unsigned size, MutableArrayRef<Value *> resultElts,
                            hlsl::OP *hlslOP, IRBuilder<> &Builder) {
  Type *i64Ty = Builder.getInt64Ty();
  Type *doubleTy = Builder.getDoubleTy();
  if (EltTy == doubleTy) {
    Function *makeDouble =
        hlslOP->GetOpFunc(DXIL::OpCode::MakeDouble, doubleTy);
    Value *makeDoubleOpArg =
        Builder.getInt32((unsigned)DXIL::OpCode::MakeDouble);
    for (unsigned i = 0; i < size; i++) {
      Value *lo = resultElts32[2 * i];
      Value *hi = resultElts32[2 * i + 1];
      Value *V = Builder.CreateCall(makeDouble, {makeDoubleOpArg, lo, hi});
      resultElts[i] = V;
    }
  } else {
    for (unsigned i = 0; i < size; i++) {
      Value *lo = resultElts32[2 * i];
      Value *hi = resultElts32[2 * i + 1];
      lo = Builder.CreateZExt(lo, i64Ty);
      hi = Builder.CreateZExt(hi, i64Ty);
      hi = Builder.CreateShl(hi, 32);
      resultElts[i] = Builder.CreateOr(lo, hi);
    }
  }
}

static Constant *GetRawBufferMaskForETy(Type *Ty, unsigned NumComponents, hlsl::OP *OP) {
  unsigned mask = 0;

  switch (NumComponents) {
  case 0:
    break;
  case 1:
    mask = DXIL::kCompMask_X;
    break;
  case 2:
    mask = DXIL::kCompMask_X | DXIL::kCompMask_Y;
    break;
  case 3:
    mask = DXIL::kCompMask_X | DXIL::kCompMask_Y | DXIL::kCompMask_Z;
    break;
  case 4:
    mask = DXIL::kCompMask_All;
    break;
  default:
    DXASSERT(false, "Cannot load more than 2 components for 64bit types.");
  }
  return OP->GetI8Const(mask);
}

Value *GenerateRawBufLd(Value *handle, Value *bufIdx, Value *offset,
  Value *status, Type *EltTy,
  MutableArrayRef<Value *> resultElts, hlsl::OP *OP,
  IRBuilder<> &Builder, unsigned NumComponents, Constant *alignment);

static Value* TranslateRawBufVecLd(Type* VecEltTy, unsigned VecElemCount,
  IRBuilder<>& Builder, Value* handle, hlsl::OP* OP, Value* status,
  Value* bufIdx, Value* baseOffset, const DataLayout& DL,
  std::vector<Value*>& bufLds, unsigned baseAlign, bool isScalarTy = false);

void TranslateLoad(ResLoadHelper &helper, HLResource::Kind RK,
                   IRBuilder<> &Builder, hlsl::OP *OP, const DataLayout &DL) {

  Type *Ty = helper.retVal->getType();
  if (Ty->isPointerTy()) {
    DXASSERT(!DxilResource::IsAnyTexture(RK), "Textures should not be treated as structured buffers.");
    TranslateStructBufSubscript(cast<CallInst>(helper.retVal), helper.handle,
                                helper.status, OP, RK, DL);
    return;
  }

  OP::OpCode opcode = helper.opcode;

  Type *i32Ty = Builder.getInt32Ty();
  Type *i64Ty = Builder.getInt64Ty();
  Type *doubleTy = Builder.getDoubleTy();
  Type *EltTy = Ty->getScalarType();
  unsigned numComponents = 1;
  if (Ty->isVectorTy()) {
    numComponents = Ty->getVectorNumElements();
  }

  if (DXIL::IsStructuredBuffer(RK) || DXIL::IsRawBuffer(RK)) {
    std::vector<Value*> bufLds;
    const bool isBool = EltTy->isIntegerTy(1);

    // Bool are represented as i32 in memory
    Type* MemReprTy = isBool ? Builder.getInt32Ty() : EltTy;
    bool isScalarTy = !Ty->isVectorTy();

    Value* retValNew = nullptr;
    if (DXIL::IsStructuredBuffer(RK)) {
      retValNew = TranslateRawBufVecLd(MemReprTy, numComponents, Builder, helper.handle, OP, helper.status,
        helper.addr, OP->GetU32Const(0), DL, bufLds, /*baseAlign (in bytes)*/ 8, isScalarTy);
    } else {
      retValNew = TranslateRawBufVecLd(MemReprTy, numComponents, Builder, helper.handle, OP, helper.status,
        nullptr, helper.addr, DL, bufLds, /*baseAlign (in bytes)*/ 4, isScalarTy);
    }

    DXASSERT_NOMSG(!bufLds.empty());
    dxilutil::MigrateDebugValue(helper.retVal, bufLds.front());

    if (isBool) {
      // Convert result back to register representation.
      retValNew = Builder.CreateICmpNE(retValNew, Constant::getNullValue(retValNew->getType()));
    }

    helper.retVal->replaceAllUsesWith(retValNew);
    helper.retVal = retValNew;
    return;
  }

  bool isTyped = opcode == OP::OpCode::TextureLoad ||
                 RK == DxilResource::Kind::TypedBuffer;
  bool is64 = EltTy == i64Ty || EltTy == doubleTy;
  if (is64 && isTyped) {
    EltTy = i32Ty;
  }
  bool isBool = EltTy->isIntegerTy(1);
  if (isBool) {
    // Value will be loaded in its memory representation.
    EltTy = i32Ty;
    if (Ty->isVectorTy()) Ty = VectorType::get(EltTy, numComponents);
  }

  Function *F = OP->GetOpFunc(opcode, EltTy);
  llvm::Constant *opArg = OP->GetU32Const((unsigned)opcode);

  llvm::Value *undefI = llvm::UndefValue::get(i32Ty);

  SmallVector<Value *, 12> loadArgs;
  loadArgs.emplace_back(opArg);         // opcode
  loadArgs.emplace_back(helper.handle); // resource handle

  if (opcode == OP::OpCode::TextureLoad) {
    // set mip level
    loadArgs.emplace_back(helper.mipLevel);
  }

  if (opcode == OP::OpCode::TextureLoad) {
    // texture coord
    unsigned coordSize = DxilResource::GetNumCoords(RK);
    bool isVectorAddr = helper.addr->getType()->isVectorTy();
    for (unsigned i = 0; i < 3; i++) {
      if (i < coordSize) {
        loadArgs.emplace_back(
          isVectorAddr ? Builder.CreateExtractElement(helper.addr, i) : helper.addr);
      }
      else
        loadArgs.emplace_back(undefI);
    }
  } else {
    if (helper.addr->getType()->isVectorTy()) {
      Value *scalarOffset =
          Builder.CreateExtractElement(helper.addr, (uint64_t)0);

      // TODO: calculate the real address based on opcode

      loadArgs.emplace_back(scalarOffset); // offset
    } else {
      // TODO: calculate the real address based on opcode

      loadArgs.emplace_back(helper.addr); // offset
    }
  }
  // offset 0
  if (opcode == OP::OpCode::TextureLoad) {
    if (helper.offset && !isa<llvm::UndefValue>(helper.offset)) {
      unsigned offsetSize = DxilResource::GetNumOffsets(RK);
      for (unsigned i = 0; i < 3; i++) {
        if (i < offsetSize)
          loadArgs.emplace_back(Builder.CreateExtractElement(helper.offset, i));
        else
          loadArgs.emplace_back(undefI);
      }
    } else {
      loadArgs.emplace_back(undefI);
      loadArgs.emplace_back(undefI);
      loadArgs.emplace_back(undefI);
    }
  }

  // Offset 1
  if (RK == DxilResource::Kind::TypedBuffer) {
    loadArgs.emplace_back(undefI);
  }

  Value *ResRet =
      Builder.CreateCall(F, loadArgs, OP->GetOpCodeName(opcode));
  dxilutil::MigrateDebugValue(helper.retVal, ResRet);

  Value *retValNew = nullptr;
  if (!is64 || !isTyped) {
    retValNew = ScalarizeResRet(Ty, ResRet, Builder);
  } else {
    unsigned size = numComponents;
    DXASSERT(size <= 2, "typed buffer only allow 4 dwords");
    EltTy = Ty->getScalarType();
    Value *Elts[2];

    Make64bitResultForLoad(Ty->getScalarType(),
                           {
                               Builder.CreateExtractValue(ResRet, 0),
                               Builder.CreateExtractValue(ResRet, 1),
                               Builder.CreateExtractValue(ResRet, 2),
                               Builder.CreateExtractValue(ResRet, 3),
                           },
                           size, Elts, OP, Builder);

    retValNew = ScalarizeElements(Ty, Elts, Builder);
  }

  if (isBool) {
    // Convert result back to register representation.
    retValNew = Builder.CreateICmpNE(retValNew, Constant::getNullValue(retValNew->getType()));
  }

  // replace
  helper.retVal->replaceAllUsesWith(retValNew);
  // Save new ret val.
  helper.retVal = retValNew;
  // get status
  UpdateStatus(ResRet, helper.status, Builder, OP);
}

Value *TranslateResourceLoad(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                             HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  Value *handle = CI->getArgOperand(HLOperandIndex::kHandleOpIdx);
  IRBuilder<> Builder(CI);

  DXIL::ResourceClass RC = pObjHelper->GetRC(handle);
  DXIL::ResourceKind RK = pObjHelper->GetRK(handle);

  ResLoadHelper loadHelper(CI, RK, RC, handle, IOP);
  TranslateLoad(loadHelper, RK, Builder, hlslOP, helper.dataLayout);
  // CI is replaced in TranslateLoad.
  return nullptr;
}

// Split { v0, v1 } to { v0.lo, v0.hi, v1.lo, v1.hi }
void Split64bitValForStore(Type *EltTy, ArrayRef<Value *> vals, unsigned size,
                           MutableArrayRef<Value *> vals32, hlsl::OP *hlslOP,
                           IRBuilder<> &Builder) {
  Type *i32Ty = Builder.getInt32Ty();
  Type *doubleTy = Builder.getDoubleTy();
  Value *undefI32 = UndefValue::get(i32Ty);

  if (EltTy == doubleTy) {
    Function *dToU = hlslOP->GetOpFunc(DXIL::OpCode::SplitDouble, doubleTy);
    Value *dToUOpArg = Builder.getInt32((unsigned)DXIL::OpCode::SplitDouble);
    for (unsigned i = 0; i < size; i++) {
      if (isa<UndefValue>(vals[i])) {
        vals32[2 * i] = undefI32;
        vals32[2 * i + 1] = undefI32;
      } else {
        Value *retVal = Builder.CreateCall(dToU, {dToUOpArg, vals[i]});
        Value *lo = Builder.CreateExtractValue(retVal, 0);
        Value *hi = Builder.CreateExtractValue(retVal, 1);
        vals32[2 * i] = lo;
        vals32[2 * i + 1] = hi;
      }
    }
  } else {
    for (unsigned i = 0; i < size; i++) {
      if (isa<UndefValue>(vals[i])) {
        vals32[2 * i] = undefI32;
        vals32[2 * i + 1] = undefI32;
      } else {
        Value *lo = Builder.CreateTrunc(vals[i], i32Ty);
        Value *hi = Builder.CreateLShr(vals[i], 32);
        hi = Builder.CreateTrunc(hi, i32Ty);
        vals32[2 * i] = lo;
        vals32[2 * i + 1] = hi;
      }
    }
  }
}

void TranslateStore(DxilResource::Kind RK, Value *handle, Value *val,
                    Value *offset, IRBuilder<> &Builder, hlsl::OP *OP,
                    Value *sampIdx = nullptr) {
  Type *Ty = val->getType();

  // This function is no longer used for lowering stores to a
  // structured buffer.
  DXASSERT_NOMSG(RK != DxilResource::Kind::StructuredBuffer);

  OP::OpCode opcode = OP::OpCode::NumOpCodes;
  switch (RK) {
  case DxilResource::Kind::RawBuffer:
  case DxilResource::Kind::StructuredBuffer:
    opcode = OP::OpCode::RawBufferStore;
    break;
  case DxilResource::Kind::TypedBuffer:
    opcode = OP::OpCode::BufferStore;
    break;
  case DxilResource::Kind::Invalid:
    DXASSERT(0, "invalid resource kind");
    break;
  case DxilResource::Kind::Texture2DMS:
  case DxilResource::Kind::Texture2DMSArray:
    opcode = OP::OpCode::TextureStoreSample;
    break;
  default:
    opcode = OP::OpCode::TextureStore;
    break;
  }

  bool isTyped = opcode == OP::OpCode::TextureStore ||
                 opcode == OP::OpCode::TextureStoreSample ||
                 RK == DxilResource::Kind::TypedBuffer;

  Type *i32Ty = Builder.getInt32Ty();
  Type *i64Ty = Builder.getInt64Ty();
  Type *doubleTy = Builder.getDoubleTy();
  Type *EltTy = Ty->getScalarType();
  if (EltTy->isIntegerTy(1)) {
    // Since we're going to memory, convert bools to their memory representation.
    EltTy = i32Ty;
    if (Ty->isVectorTy()) Ty = VectorType::get(EltTy, Ty->getVectorNumElements());
    else Ty = EltTy;
    val = Builder.CreateZExt(val, Ty);
  }

  // If RawBuffer store of 64-bit value, don't set alignment to 8,
  // since buffer alignment isn't known to be anything over 4.
  unsigned alignValue = OP->GetAllocSizeForType(EltTy);
  if (RK == HLResource::Kind::RawBuffer && alignValue > 4)
    alignValue = 4;
  Constant *Alignment = OP->GetI32Const(alignValue);
  bool is64 = EltTy == i64Ty || EltTy == doubleTy;
  if (is64 && isTyped) {
    EltTy = i32Ty;
  }

  Function *F = OP->GetOpFunc(opcode, EltTy);
  llvm::Constant *opArg = OP->GetU32Const((unsigned)opcode);

  llvm::Value *undefI =
      llvm::UndefValue::get(llvm::Type::getInt32Ty(Ty->getContext()));

  llvm::Value *undefVal = llvm::UndefValue::get(Ty->getScalarType());

  SmallVector<Value *, 13> storeArgs;
  storeArgs.emplace_back(opArg);  // opcode
  storeArgs.emplace_back(handle); // resource handle

  unsigned offset0Idx = 0;
  if (RK == DxilResource::Kind::RawBuffer ||
      RK == DxilResource::Kind::TypedBuffer) {
    // Offset 0
    if (offset->getType()->isVectorTy()) {
      Value *scalarOffset = Builder.CreateExtractElement(offset, (uint64_t)0);
      storeArgs.emplace_back(scalarOffset); // offset
    } else {
      storeArgs.emplace_back(offset); // offset
    }

    // Store offset0 for later use
    offset0Idx = storeArgs.size() - 1;

    // Offset 1
    storeArgs.emplace_back(undefI);
  } else {
    // texture store
    unsigned coordSize = DxilResource::GetNumCoords(RK);

    // Set x first.
    if (offset->getType()->isVectorTy())
      storeArgs.emplace_back(Builder.CreateExtractElement(offset, (uint64_t)0));
    else
      storeArgs.emplace_back(offset);

    // Store offset0 for later use
    offset0Idx = storeArgs.size() - 1;

    for (unsigned i = 1; i < 3; i++) {
      if (i < coordSize)
        storeArgs.emplace_back(Builder.CreateExtractElement(offset, i));
      else
        storeArgs.emplace_back(undefI);
    }
    // TODO: support mip for texture ST
  }

  constexpr unsigned MaxStoreElemCount = 4;
  const unsigned CompCount = Ty->isVectorTy() ? Ty->getVectorNumElements() : 1;
  const unsigned StoreInstCount = (CompCount / MaxStoreElemCount) + (CompCount % MaxStoreElemCount != 0);
  SmallVector<decltype(storeArgs), 4> storeArgsList;

  // Max number of element to store should be 16 (for a 4x4 matrix)
  DXASSERT_NOMSG(StoreInstCount >= 1 && StoreInstCount <= 4);
  
  // If number of elements to store exceeds the maximum number of elements
  // that can be stored in a single store call,  make sure to generate enough 
  // store calls to store all elements
  for (unsigned j = 0; j < StoreInstCount; j++) {
    decltype(storeArgs) newStoreArgs;
    for (Value* storeArg : storeArgs)
      newStoreArgs.emplace_back(storeArg);
    storeArgsList.emplace_back(newStoreArgs);
  }

  for (unsigned j = 0; j < storeArgsList.size(); j++) {

    // For second and subsequent store calls, increment the offset0 (i.e. store index)
    if (j > 0) {
      // Greater than four-components store is not allowed for
      // TypedBuffer and Textures. So greater than four elements
      // scenario should only get hit here for RawBuffer.
      DXASSERT_NOMSG(RK == DxilResource::Kind::RawBuffer);
      unsigned EltSize = OP->GetAllocSizeForType(EltTy);
      unsigned newOffset = EltSize * MaxStoreElemCount * j;
      Value* newOffsetVal = ConstantInt::get(Builder.getInt32Ty(), newOffset);
      newOffsetVal = Builder.CreateAdd(storeArgsList[0][offset0Idx], newOffsetVal);
      storeArgsList[j][offset0Idx] = newOffsetVal;
    }

    // values
    uint8_t mask = 0;
    if (Ty->isVectorTy()) {
      unsigned vecSize = std::min((j + 1) * MaxStoreElemCount, Ty->getVectorNumElements()) - (j * MaxStoreElemCount);
      Value* emptyVal = undefVal;
      if (isTyped) {
        mask = DXIL::kCompMask_All;
        emptyVal = Builder.CreateExtractElement(val, (uint64_t)0);
      }

      for (unsigned i = 0; i < MaxStoreElemCount; i++) {
        if (i < vecSize) {
          storeArgsList[j].emplace_back(Builder.CreateExtractElement(val, (j * MaxStoreElemCount) + i));
          mask |= (1 << i);
        }
        else {
          storeArgsList[j].emplace_back(emptyVal);
        }
      }

    }
    else {
      if (isTyped) {
        mask = DXIL::kCompMask_All;
        storeArgsList[j].emplace_back(val);
        storeArgsList[j].emplace_back(val);
        storeArgsList[j].emplace_back(val);
        storeArgsList[j].emplace_back(val);
      }
      else {
        storeArgsList[j].emplace_back(val);
        storeArgsList[j].emplace_back(undefVal);
        storeArgsList[j].emplace_back(undefVal);
        storeArgsList[j].emplace_back(undefVal);
        mask = DXIL::kCompMask_X;
      }
    }

    if (is64 && isTyped) {
      unsigned size = 1;
      if (Ty->isVectorTy()) {
        size = std::min((j + 1) * MaxStoreElemCount, Ty->getVectorNumElements()) - (j * MaxStoreElemCount);
      }
      DXASSERT(size <= 2, "raw/typed buffer only allow 4 dwords");
      unsigned val0OpIdx = opcode == DXIL::OpCode::TextureStore || opcode == DXIL::OpCode::TextureStoreSample
        ? DXIL::OperandIndex::kTextureStoreVal0OpIdx
        : DXIL::OperandIndex::kBufferStoreVal0OpIdx;
      Value* V0 = storeArgsList[j][val0OpIdx];
      Value* V1 = storeArgsList[j][val0OpIdx + 1];

      Value* vals32[4];
      EltTy = Ty->getScalarType();
      Split64bitValForStore(EltTy, { V0, V1 }, size, vals32, OP, Builder);
      // Fill the uninit vals.
      if (size == 1) {
        vals32[2] = vals32[0];
        vals32[3] = vals32[1];
      }
      // Change valOp to 32 version.
      for (unsigned i = 0; i < 4; i++) {
        storeArgsList[j][val0OpIdx + i] = vals32[i];
      }
      // change mask for double
      if (opcode == DXIL::OpCode::RawBufferStore) {
        mask = size == 1 ?
          DXIL::kCompMask_X | DXIL::kCompMask_Y : DXIL::kCompMask_All;
      }
    }

    storeArgsList[j].emplace_back(OP->GetU8Const(mask)); // mask
    if (opcode == DXIL::OpCode::RawBufferStore)
      storeArgsList[j].emplace_back(Alignment); // alignment only for raw buffer
    else if (opcode == DXIL::OpCode::TextureStoreSample) {
      storeArgsList[j].emplace_back(sampIdx?sampIdx:Builder.getInt32(0)); // sample idx only for MS textures
    }
    Builder.CreateCall(F, storeArgsList[j]);
  }
}

Value *TranslateResourceStore(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                              HLOperationLowerHelper &helper, 
                              HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  Value *handle = CI->getArgOperand(HLOperandIndex::kHandleOpIdx);
  IRBuilder<> Builder(CI);
  DXIL::ResourceKind RK = pObjHelper->GetRK(handle);

  Value *val = CI->getArgOperand(HLOperandIndex::kStoreValOpIdx);
  Value *offset = CI->getArgOperand(HLOperandIndex::kStoreOffsetOpIdx);
  TranslateStore(RK, handle, val, offset, Builder, hlslOP);

  return nullptr;
}
}

// Atomic intrinsics.
namespace {
// Atomic intrinsics.
struct AtomicHelper {
  AtomicHelper(CallInst *CI, OP::OpCode op, Value *h, Type *opType=nullptr);
  AtomicHelper(CallInst *CI, OP::OpCode op, Value *h, Value *bufIdx,
               Value *baseOffset, Type *opType=nullptr);
  OP::OpCode opcode;
  Value *handle;
  Value *addr;
  Value *offset; // Offset for structrued buffer.
  Value *value;
  Value *originalValue;
  Value *compareValue;
  Type *operationType;
};

// For MOP version of Interlocked*.
AtomicHelper::AtomicHelper(CallInst *CI, OP::OpCode op, Value *h, Type *opType)
    : opcode(op), handle(h), offset(nullptr), originalValue(nullptr),
      operationType(opType) {
  addr = CI->getArgOperand(HLOperandIndex::kObjectInterlockedDestOpIndex);
  if (op == OP::OpCode::AtomicCompareExchange) {
    compareValue = CI->getArgOperand(
        HLOperandIndex::kObjectInterlockedCmpCompareValueOpIndex);
    value =
        CI->getArgOperand(HLOperandIndex::kObjectInterlockedCmpValueOpIndex);
    if (CI->getNumArgOperands() ==
        (HLOperandIndex::kObjectInterlockedCmpOriginalValueOpIndex + 1))
      originalValue = CI->getArgOperand(
          HLOperandIndex::kObjectInterlockedCmpOriginalValueOpIndex);
  } else {
    value = CI->getArgOperand(HLOperandIndex::kObjectInterlockedValueOpIndex);
    if (CI->getNumArgOperands() ==
        (HLOperandIndex::kObjectInterlockedOriginalValueOpIndex + 1))
      originalValue = CI->getArgOperand(
          HLOperandIndex::kObjectInterlockedOriginalValueOpIndex);
  }
  if (nullptr == operationType)
    operationType = value->getType();
}
// For IOP version of Interlocked*.
AtomicHelper::AtomicHelper(CallInst *CI, OP::OpCode op, Value *h, Value *bufIdx,
                           Value *baseOffset, Type *opType)
    : opcode(op), handle(h), addr(bufIdx),
      offset(baseOffset), originalValue(nullptr),
      operationType(opType) {
  if (op == OP::OpCode::AtomicCompareExchange) {
    compareValue =
        CI->getArgOperand(HLOperandIndex::kInterlockedCmpCompareValueOpIndex);
    value = CI->getArgOperand(HLOperandIndex::kInterlockedCmpValueOpIndex);
    if (CI->getNumArgOperands() ==
        (HLOperandIndex::kInterlockedCmpOriginalValueOpIndex + 1))
      originalValue = CI->getArgOperand(
          HLOperandIndex::kInterlockedCmpOriginalValueOpIndex);
  } else {
    value = CI->getArgOperand(HLOperandIndex::kInterlockedValueOpIndex);
    if (CI->getNumArgOperands() ==
        (HLOperandIndex::kInterlockedOriginalValueOpIndex + 1))
      originalValue =
          CI->getArgOperand(HLOperandIndex::kInterlockedOriginalValueOpIndex);
  }
  if (nullptr == operationType)
    operationType = value->getType();
}

void TranslateAtomicBinaryOperation(AtomicHelper &helper,
                                    DXIL::AtomicBinOpCode atomicOp,
                                    IRBuilder<> &Builder, hlsl::OP *hlslOP) {
  Value *handle = helper.handle;
  Value *addr = helper.addr;
  Value *val = helper.value;
  Type *Ty = helper.operationType;
  Type *valTy = val->getType();

  Value *undefI = UndefValue::get(Type::getInt32Ty(Ty->getContext()));

  Function *dxilAtomic = hlslOP->GetOpFunc(helper.opcode, Ty->getScalarType());
  Value *opArg = hlslOP->GetU32Const(static_cast<unsigned>(helper.opcode));
  Value *atomicOpArg = hlslOP->GetU32Const(static_cast<unsigned>(atomicOp));

  if (Ty != valTy)
    val = Builder.CreateBitCast(val, Ty);

  Value *args[] = {opArg,  handle, atomicOpArg,
                   undefI, undefI, undefI, // coordinates
                   val};

  // Setup coordinates.
  if (addr->getType()->isVectorTy()) {
    unsigned vectorNumElements = addr->getType()->getVectorNumElements();
    DXASSERT(vectorNumElements <= 3, "up to 3 elements for atomic binary op");
    _Analysis_assume_(vectorNumElements <= 3);
    for (unsigned i = 0; i < vectorNumElements; i++) {
      Value *Elt = Builder.CreateExtractElement(addr, i);
      args[DXIL::OperandIndex::kAtomicBinOpCoord0OpIdx + i] = Elt;
    }
  } else
    args[DXIL::OperandIndex::kAtomicBinOpCoord0OpIdx] = addr;

  // Set offset for structured buffer.
  if (helper.offset)
    args[DXIL::OperandIndex::kAtomicBinOpCoord1OpIdx] = helper.offset;

  Value *origVal =
      Builder.CreateCall(dxilAtomic, args, hlslOP->GetAtomicOpName(atomicOp));
  if (helper.originalValue) {
    if (Ty != valTy)
      origVal = Builder.CreateBitCast(origVal, valTy);
    Builder.CreateStore(origVal, helper.originalValue);
  }
}

Value *TranslateMopAtomicBinaryOperation(CallInst *CI, IntrinsicOp IOP,
                                         OP::OpCode opcode,
                                         HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;

  Value *handle = CI->getArgOperand(HLOperandIndex::kHandleOpIdx);
  IRBuilder<> Builder(CI);

  switch (IOP) {
  case IntrinsicOp::MOP_InterlockedAdd:
  case IntrinsicOp::MOP_InterlockedAdd64: {
    AtomicHelper helper(CI, DXIL::OpCode::AtomicBinOp, handle);
    TranslateAtomicBinaryOperation(helper, DXIL::AtomicBinOpCode::Add, Builder,
                                   hlslOP);
  } break;
  case IntrinsicOp::MOP_InterlockedAnd:
  case IntrinsicOp::MOP_InterlockedAnd64: {
    AtomicHelper helper(CI, DXIL::OpCode::AtomicBinOp, handle);
    TranslateAtomicBinaryOperation(helper, DXIL::AtomicBinOpCode::And, Builder,
                                   hlslOP);
  } break;
  case IntrinsicOp::MOP_InterlockedExchange:
  case IntrinsicOp::MOP_InterlockedExchange64: {
    AtomicHelper helper(CI, DXIL::OpCode::AtomicBinOp, handle);
    TranslateAtomicBinaryOperation(helper, DXIL::AtomicBinOpCode::Exchange,
                                   Builder, hlslOP);
  } break;
  case IntrinsicOp::MOP_InterlockedExchangeFloat: {
    AtomicHelper helper(CI, DXIL::OpCode::AtomicBinOp, handle, Type::getInt32Ty(CI->getContext()));
    TranslateAtomicBinaryOperation(helper, DXIL::AtomicBinOpCode::Exchange,
                                   Builder, hlslOP);
  } break;
  case IntrinsicOp::MOP_InterlockedMax:
  case IntrinsicOp::MOP_InterlockedMax64: {
    AtomicHelper helper(CI, DXIL::OpCode::AtomicBinOp, handle);
    TranslateAtomicBinaryOperation(helper, DXIL::AtomicBinOpCode::IMax, Builder,
                                   hlslOP);
  } break;
  case IntrinsicOp::MOP_InterlockedMin:
  case IntrinsicOp::MOP_InterlockedMin64: {
    AtomicHelper helper(CI, DXIL::OpCode::AtomicBinOp, handle);
    TranslateAtomicBinaryOperation(helper, DXIL::AtomicBinOpCode::IMin, Builder,
                                   hlslOP);
  } break;
  case IntrinsicOp::MOP_InterlockedUMax: {
    AtomicHelper helper(CI, DXIL::OpCode::AtomicBinOp, handle);
    TranslateAtomicBinaryOperation(helper, DXIL::AtomicBinOpCode::UMax, Builder,
                                   hlslOP);
  } break;
  case IntrinsicOp::MOP_InterlockedUMin: {
    AtomicHelper helper(CI, DXIL::OpCode::AtomicBinOp, handle);
    TranslateAtomicBinaryOperation(helper, DXIL::AtomicBinOpCode::UMin, Builder,
                                   hlslOP);
  } break;
  case IntrinsicOp::MOP_InterlockedOr:
  case IntrinsicOp::MOP_InterlockedOr64: {
    AtomicHelper helper(CI, DXIL::OpCode::AtomicBinOp, handle);
    TranslateAtomicBinaryOperation(helper, DXIL::AtomicBinOpCode::Or, Builder,
                                   hlslOP);
  } break;
  case IntrinsicOp::MOP_InterlockedXor:
  case IntrinsicOp::MOP_InterlockedXor64:
  default: {
    DXASSERT(IOP == IntrinsicOp::MOP_InterlockedXor || IOP == IntrinsicOp::MOP_InterlockedXor64,
             "invalid MOP atomic intrinsic");
    AtomicHelper helper(CI, DXIL::OpCode::AtomicBinOp, handle);
    TranslateAtomicBinaryOperation(helper, DXIL::AtomicBinOpCode::Xor, Builder,
                                   hlslOP);
  } break;
  }

  return nullptr;
}
void TranslateAtomicCmpXChg(AtomicHelper &helper, IRBuilder<> &Builder,
                            hlsl::OP *hlslOP) {
  Value *handle = helper.handle;
  Value *addr = helper.addr;
  Value *val = helper.value;
  Value *cmpVal = helper.compareValue;

  Type *Ty = helper.operationType;
  Type *valTy = val->getType();

  Value *undefI = UndefValue::get(Type::getInt32Ty(Ty->getContext()));

  Function *dxilAtomic = hlslOP->GetOpFunc(helper.opcode, Ty->getScalarType());
  Value *opArg = hlslOP->GetU32Const(static_cast<unsigned>(helper.opcode));

  if (Ty != valTy) {
    val = Builder.CreateBitCast(val, Ty);
    if (cmpVal)
      cmpVal = Builder.CreateBitCast(cmpVal, Ty);
  }

  Value *args[] = {opArg,  handle, undefI, undefI, undefI, // coordinates
                   cmpVal, val};

  // Setup coordinates.
  if (addr->getType()->isVectorTy()) {
    unsigned vectorNumElements = addr->getType()->getVectorNumElements();
    DXASSERT(vectorNumElements <= 3, "up to 3 elements in atomic op");
    _Analysis_assume_(vectorNumElements <= 3);
    for (unsigned i = 0; i < vectorNumElements; i++) {
      Value *Elt = Builder.CreateExtractElement(addr, i);
      args[DXIL::OperandIndex::kAtomicCmpExchangeCoord0OpIdx + i] = Elt;
    }
  } else
    args[DXIL::OperandIndex::kAtomicCmpExchangeCoord0OpIdx] = addr;

  // Set offset for structured buffer.
  if (helper.offset)
    args[DXIL::OperandIndex::kAtomicCmpExchangeCoord1OpIdx] = helper.offset;

  Value *origVal = Builder.CreateCall(dxilAtomic, args);
  if (helper.originalValue) {
    if (Ty != valTy)
      origVal = Builder.CreateBitCast(origVal, valTy);
    Builder.CreateStore(origVal, helper.originalValue);
  }
}

Value *TranslateMopAtomicCmpXChg(CallInst *CI, IntrinsicOp IOP,
                                 OP::OpCode opcode,
                                 HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;

  Value *handle = CI->getArgOperand(HLOperandIndex::kHandleOpIdx);
  IRBuilder<> Builder(CI);
  Type *opType = nullptr;
  if (IOP == IntrinsicOp::MOP_InterlockedCompareStoreFloatBitwise ||
      IOP == IntrinsicOp::MOP_InterlockedCompareExchangeFloatBitwise)
    opType = Type::getInt32Ty(CI->getContext());
  AtomicHelper atomicHelper(CI, OP::OpCode::AtomicCompareExchange, handle, opType);
  TranslateAtomicCmpXChg(atomicHelper, Builder, hlslOP);
  return nullptr;
}

void TranslateSharedMemAtomicBinOp(CallInst *CI, IntrinsicOp IOP, Value *addr) {
  AtomicRMWInst::BinOp Op;
  IRBuilder<> Builder(CI);
  Value *val = CI->getArgOperand(HLOperandIndex::kInterlockedValueOpIndex);
  PointerType *ptrType = dyn_cast<PointerType>(
               CI->getArgOperand(HLOperandIndex::kInterlockedDestOpIndex)->getType());
  bool needCast = ptrType && ptrType->getElementType()->isFloatTy();
  switch (IOP) {
  case IntrinsicOp::IOP_InterlockedAdd:
    Op = AtomicRMWInst::BinOp::Add;
    break;
  case IntrinsicOp::IOP_InterlockedAnd:
    Op = AtomicRMWInst::BinOp::And;
    break;
  case IntrinsicOp::IOP_InterlockedExchange:
    if (needCast) {
      val = Builder.CreateBitCast(val, Type::getInt32Ty(CI->getContext()));
      addr = Builder.CreateBitCast(addr, Type::getInt32PtrTy(CI->getContext(), DXIL::kTGSMAddrSpace));
    }
    Op = AtomicRMWInst::BinOp::Xchg;
    break;
  case IntrinsicOp::IOP_InterlockedMax:
    Op = AtomicRMWInst::BinOp::Max;
    break;
  case IntrinsicOp::IOP_InterlockedUMax:
    Op = AtomicRMWInst::BinOp::UMax;
    break;
  case IntrinsicOp::IOP_InterlockedMin:
    Op = AtomicRMWInst::BinOp::Min;
    break;
  case IntrinsicOp::IOP_InterlockedUMin:
    Op = AtomicRMWInst::BinOp::UMin;
    break;
  case IntrinsicOp::IOP_InterlockedOr:
    Op = AtomicRMWInst::BinOp::Or;
    break;
  case IntrinsicOp::IOP_InterlockedXor:
  default:
    DXASSERT(IOP == IntrinsicOp::IOP_InterlockedXor, "Invalid Intrinsic");
    Op = AtomicRMWInst::BinOp::Xor;
    break;
  }

  Value *Result = Builder.CreateAtomicRMW(
      Op, addr, val, AtomicOrdering::SequentiallyConsistent);
  if (CI->getNumArgOperands() >
      HLOperandIndex::kInterlockedOriginalValueOpIndex) {
    if (needCast)
      Result = Builder.CreateBitCast(Result, Type::getFloatTy(CI->getContext()));
    Builder.CreateStore(
        Result,
        CI->getArgOperand(HLOperandIndex::kInterlockedOriginalValueOpIndex));
  }
}

static Value* SkipAddrSpaceCast(Value* Ptr) {
  if (AddrSpaceCastInst *CastInst = dyn_cast<AddrSpaceCastInst>(Ptr))
    return CastInst->getOperand(0);
  else if (ConstantExpr *ConstExpr = dyn_cast<ConstantExpr>(Ptr)) {
    if (ConstExpr->getOpcode() == Instruction::AddrSpaceCast) {
      return ConstExpr->getOperand(0);
    }
  }
  return Ptr;
}

// For known non-groupshared, verify that the destination param is valid
void ValidateAtomicDestination(CallInst *CI, HLObjectOperationLowerHelper *pObjHelper) {
  Value *dest = CI->getArgOperand(HLOperandIndex::kInterlockedDestOpIndex);
  // If we encounter a gep, we may provide a more specific error message
  bool hasGep = isa<GetElementPtrInst>(dest);

  // Confirm that dest is a properly-used UAV

  // Drill through subscripts and geps, anything else indicates a misuse
  while(true) {
    if (GetElementPtrInst *gep = dyn_cast<GetElementPtrInst>(dest)) {
      dest = gep->getPointerOperand();
      continue;
    }
    if (CallInst *handle = dyn_cast<CallInst>(dest)) {
      hlsl::HLOpcodeGroup group = hlsl::GetHLOpcodeGroup(handle->getCalledFunction());
      if (group != HLOpcodeGroup::HLSubscript)
        break;
      dest = handle->getArgOperand(HLOperandIndex::kSubscriptObjectOpIdx);
      continue;
    }
    break;
  }

  if(pObjHelper->GetRC(dest) == DXIL::ResourceClass::UAV) {
    DXIL::ResourceKind RK = pObjHelper->GetRK(dest);
    if (DXIL::IsStructuredBuffer(RK))
      return; // no errors
    if (DXIL::IsTyped(RK)) {
      if (hasGep)
        dxilutil::EmitErrorOnInstruction(CI, "Typed resources used in atomic operations must have a scalar element type.");
      return; // error emitted or else no errors
    }
  }

  dxilutil::EmitErrorOnInstruction(CI, "Atomic operation targets must be groupshared or UAV.");
}

Value *TranslateIopAtomicBinaryOperation(CallInst *CI, IntrinsicOp IOP,
                                         DXIL::OpCode opcode,
                                         HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  Value *addr = CI->getArgOperand(HLOperandIndex::kInterlockedDestOpIndex);
  addr = SkipAddrSpaceCast(addr);

  unsigned addressSpace = addr->getType()->getPointerAddressSpace();
  if (addressSpace == DXIL::kTGSMAddrSpace)
    TranslateSharedMemAtomicBinOp(CI, IOP, addr);
  else {
    // If not groupshared, we either have an error case or will translate
    // the atomic op in the process of translating users of the subscript operator
    // Mark not translated and validate dest param
    Translated = false;
    ValidateAtomicDestination(CI, pObjHelper);
  }

  return nullptr;
}

void TranslateSharedMemAtomicCmpXChg(CallInst *CI, Value *addr) {
  Value *val = CI->getArgOperand(HLOperandIndex::kInterlockedCmpValueOpIndex);
  Value *cmpVal =
      CI->getArgOperand(HLOperandIndex::kInterlockedCmpCompareValueOpIndex);
  IRBuilder<> Builder(CI);

  PointerType *ptrType = dyn_cast<PointerType>(
               CI->getArgOperand(HLOperandIndex::kInterlockedDestOpIndex)->getType());
  bool needCast = false;
  if (ptrType && ptrType->getElementType()->isFloatTy()) {
    needCast = true;
    val = Builder.CreateBitCast(val, Type::getInt32Ty(CI->getContext()));
    cmpVal = Builder.CreateBitCast(cmpVal, Type::getInt32Ty(CI->getContext()));
    addr = Builder.CreateBitCast(addr, Type::getInt32PtrTy(CI->getContext(), DXIL::kTGSMAddrSpace));
  }

  Value *Result = Builder.CreateAtomicCmpXchg(
      addr, cmpVal, val, AtomicOrdering::SequentiallyConsistent,
      AtomicOrdering::SequentiallyConsistent);

  if (CI->getNumArgOperands() >
      HLOperandIndex::kInterlockedCmpOriginalValueOpIndex) {
    Value *originVal = Builder.CreateExtractValue(Result, 0);
    if (needCast)
      originVal = Builder.CreateBitCast(originVal, Type::getFloatTy(CI->getContext()));
    Builder.CreateStore(
        originVal,
        CI->getArgOperand(HLOperandIndex::kInterlockedCmpOriginalValueOpIndex));
  }
}

Value *TranslateIopAtomicCmpXChg(CallInst *CI, IntrinsicOp IOP,
                                 DXIL::OpCode opcode,
                                 HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  Value *addr = CI->getArgOperand(HLOperandIndex::kInterlockedDestOpIndex);
  addr = SkipAddrSpaceCast(addr);

  unsigned addressSpace = addr->getType()->getPointerAddressSpace();
  if (addressSpace == DXIL::kTGSMAddrSpace)
    TranslateSharedMemAtomicCmpXChg(CI, addr);
  else {
    // If not groupshared, we either have an error case or will translate
    // the atomic op in the process of translating users of the subscript operator
    // Mark not translated and validate dest param
    Translated = false;
    ValidateAtomicDestination(CI, pObjHelper);
  }

  return nullptr;
}
}

// Process Tess Factor.
namespace {

// Clamp to [0.0f..1.0f], NaN->0.0f.
Value *CleanupTessFactorScale(Value *input, hlsl::OP *hlslOP, IRBuilder<> &Builder) {
  float fMin = 0;
  float fMax = 1;
  Type *f32Ty = input->getType()->getScalarType();
  Value *minFactor = ConstantFP::get(f32Ty, fMin);
  Value *maxFactor = ConstantFP::get(f32Ty, fMax);
  Type *Ty = input->getType();
  if (Ty->isVectorTy())
    minFactor = SplatToVector(minFactor, input->getType(), Builder);
  Value *temp = TrivialDxilBinaryOperation(DXIL::OpCode::FMax, input, minFactor, hlslOP, Builder);
  if (Ty->isVectorTy())
    maxFactor = SplatToVector(maxFactor, input->getType(), Builder);
  return TrivialDxilBinaryOperation(DXIL::OpCode::FMin, temp, maxFactor, hlslOP, Builder);
}

// Clamp to [1.0f..Inf], NaN->1.0f.
Value *CleanupTessFactor(Value *input, hlsl::OP *hlslOP, IRBuilder<> &Builder)
{
  float fMin = 1.0;  
  Type *f32Ty = input->getType()->getScalarType();
  Value *minFactor = ConstantFP::get(f32Ty, fMin);
  minFactor = SplatToVector(minFactor, input->getType(), Builder);
  return TrivialDxilBinaryOperation(DXIL::OpCode::FMax, input, minFactor, hlslOP, Builder);
}

// Do partitioning-specific clamping.
Value *ClampTessFactor(Value *input, DXIL::TessellatorPartitioning partitionMode, 
    hlsl::OP *hlslOP, IRBuilder<> &Builder) {
  const unsigned kTESSELLATOR_MAX_EVEN_TESSELLATION_FACTOR = 64;
  const unsigned kTESSELLATOR_MAX_ODD_TESSELLATION_FACTOR = 63;

  const unsigned kTESSELLATOR_MIN_EVEN_TESSELLATION_FACTOR = 2;
  const unsigned kTESSELLATOR_MIN_ODD_TESSELLATION_FACTOR = 1;

  const unsigned kTESSELLATOR_MAX_TESSELLATION_FACTOR = 64;

  float fMin;
  float fMax;
  switch (partitionMode) {
  case DXIL::TessellatorPartitioning::Integer:
    fMin = kTESSELLATOR_MIN_ODD_TESSELLATION_FACTOR;
    fMax = kTESSELLATOR_MAX_TESSELLATION_FACTOR;
    break;
  case DXIL::TessellatorPartitioning::Pow2:
    fMin = kTESSELLATOR_MIN_ODD_TESSELLATION_FACTOR;
    fMax = kTESSELLATOR_MAX_EVEN_TESSELLATION_FACTOR;
    break;
  case DXIL::TessellatorPartitioning::FractionalOdd:
    fMin = kTESSELLATOR_MIN_ODD_TESSELLATION_FACTOR;
    fMax = kTESSELLATOR_MAX_ODD_TESSELLATION_FACTOR;
    break;
  case DXIL::TessellatorPartitioning::FractionalEven:
  default:
    DXASSERT(partitionMode == DXIL::TessellatorPartitioning::FractionalEven,
        "invalid partition mode");
    fMin = kTESSELLATOR_MIN_EVEN_TESSELLATION_FACTOR;
    fMax = kTESSELLATOR_MAX_EVEN_TESSELLATION_FACTOR;
    break;
  }
  Type *f32Ty = input->getType()->getScalarType();
  Value *minFactor = ConstantFP::get(f32Ty, fMin);
  Value *maxFactor = ConstantFP::get(f32Ty, fMax);
  Type *Ty = input->getType();
  if (Ty->isVectorTy())
    minFactor = SplatToVector(minFactor, input->getType(), Builder);
  Value *temp = TrivialDxilBinaryOperation(DXIL::OpCode::FMax, input, minFactor, hlslOP, Builder);
  if (Ty->isVectorTy())
    maxFactor = SplatToVector(maxFactor, input->getType(), Builder);
  return TrivialDxilBinaryOperation(DXIL::OpCode::FMin, temp, maxFactor, hlslOP, Builder);
}

// round up for integer/pow2 partitioning
// note that this code assumes the inputs should be in the range [1, inf),
// which should be enforced by the clamp above.
Value *RoundUpTessFactor(Value *input, DXIL::TessellatorPartitioning partitionMode,
    hlsl::OP *hlslOP, IRBuilder<> &Builder) {
  switch (partitionMode) {
  case DXIL::TessellatorPartitioning::Integer:
    return TrivialDxilUnaryOperation(DXIL::OpCode::Round_pi, input, hlslOP, Builder);
  case DXIL::TessellatorPartitioning::Pow2: {
    const unsigned kExponentMask = 0x7f800000;
    const unsigned kExponentLSB = 0x00800000;
    const unsigned kMantissaMask = 0x007fffff;
    Type *Ty = input->getType();
    // (val = (asuint(val) & mantissamask) ?
    //      (asuint(val) & exponentmask) + exponentbump :
    //      asuint(val) & exponentmask;
    Type *uintTy = Type::getInt32Ty(Ty->getContext());
    if (Ty->isVectorTy())
      uintTy = VectorType::get(uintTy, Ty->getVectorNumElements());
    Value *uintVal = Builder.CreateCast(Instruction::CastOps::FPToUI, input, uintTy);

    Value *mantMask = ConstantInt::get(uintTy->getScalarType(), kMantissaMask);
    mantMask = SplatToVector(mantMask, uintTy, Builder);
    Value *manVal = Builder.CreateAnd(uintVal, mantMask);
    
    Value *expMask = ConstantInt::get(uintTy->getScalarType(), kExponentMask);
    expMask = SplatToVector(expMask, uintTy, Builder);
    Value *expVal = Builder.CreateAnd(uintVal, expMask);
    
    Value *expLSB = ConstantInt::get(uintTy->getScalarType(), kExponentLSB);
    expLSB = SplatToVector(expLSB, uintTy, Builder);
    Value *newExpVal = Builder.CreateAdd(expVal, expLSB);

    Value *manValNotZero = Builder.CreateICmpEQ(manVal, ConstantAggregateZero::get(uintTy));
    Value *factors = Builder.CreateSelect(manValNotZero, newExpVal, expVal);
    return Builder.CreateUIToFP(factors, Ty);
  } break;
  case DXIL::TessellatorPartitioning::FractionalEven:
  case DXIL::TessellatorPartitioning::FractionalOdd:
    return input;
  default:
    DXASSERT(0, "invalid partition mode");
    return nullptr;
  }
}

Value *TranslateProcessIsolineTessFactors(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                              HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  // Get partition mode 
  DXASSERT_NOMSG(helper.functionProps);
  DXASSERT(helper.functionProps->shaderKind == ShaderModel::Kind::Hull, "must be hull shader");
  DXIL::TessellatorPartitioning partition = helper.functionProps->ShaderProps.HS.partition;
  
  IRBuilder<> Builder(CI);

  Value *rawDetailFactor = CI->getArgOperand(HLOperandIndex::kProcessTessFactorRawDetailFactor);
  rawDetailFactor = Builder.CreateExtractElement(rawDetailFactor, (uint64_t)0);

  Value *rawDensityFactor = CI->getArgOperand(HLOperandIndex::kProcessTessFactorRawDensityFactor);
  rawDensityFactor = Builder.CreateExtractElement(rawDensityFactor, (uint64_t)0);

  Value *init = UndefValue::get(VectorType::get(helper.f32Ty, 2));
  init = Builder.CreateInsertElement(init, rawDetailFactor, (uint64_t)0);
  init = Builder.CreateInsertElement(init, rawDetailFactor, (uint64_t)1);

  Value *clamped = ClampTessFactor(init, partition, hlslOP, Builder);
  Value *rounded = RoundUpTessFactor(clamped, partition, hlslOP, Builder);

  Value *roundedDetailFactor = CI->getArgOperand(HLOperandIndex::kProcessTessFactorRoundedDetailFactor);
  Value *temp = UndefValue::get(VectorType::get(helper.f32Ty, 1));
  Value *roundedX = Builder.CreateExtractElement(rounded, (uint64_t)0);
  temp = Builder.CreateInsertElement(temp, roundedX, (uint64_t)0);
  Builder.CreateStore(temp, roundedDetailFactor);

  Value *roundedDensityFactor = CI->getArgOperand(HLOperandIndex::kProcessTessFactorRoundedDensityFactor);
  Value *roundedY = Builder.CreateExtractElement(rounded, 1);
  temp = Builder.CreateInsertElement(temp, roundedY, (uint64_t)0);
  Builder.CreateStore(temp, roundedDensityFactor);
  return nullptr;
}

// 3 inputs, 1 result
Value *ApplyTriTessFactorOp(Value *input, DXIL::OpCode opcode, hlsl::OP *hlslOP,
                            IRBuilder<> &Builder) {
  Value *input0 = Builder.CreateExtractElement(input, (uint64_t)0);
  Value *input1 = Builder.CreateExtractElement(input, 1);
  Value *input2 = Builder.CreateExtractElement(input, 2);

  if (opcode == DXIL::OpCode::FMax || opcode == DXIL::OpCode::FMin) {
    Value *temp =
        TrivialDxilBinaryOperation(opcode, input0, input1, hlslOP, Builder);
    Value *combined =
        TrivialDxilBinaryOperation(opcode, temp, input2, hlslOP, Builder);
    return combined;
  } else {
    // Avg.
    Value *temp = Builder.CreateFAdd(input0, input1);
    Value *combined = Builder.CreateFAdd(temp, input2);
    Value *rcp = ConstantFP::get(input0->getType(), 1.0 / 3.0);
    combined = Builder.CreateFMul(combined, rcp);
    return combined;
  }
}

// 4 inputs, 1 result
Value *ApplyQuadTessFactorOp(Value *input, DXIL::OpCode opcode,
                             hlsl::OP *hlslOP, IRBuilder<> &Builder) {
  Value *input0 = Builder.CreateExtractElement(input, (uint64_t)0);
  Value *input1 = Builder.CreateExtractElement(input, 1);
  Value *input2 = Builder.CreateExtractElement(input, 2);
  Value *input3 = Builder.CreateExtractElement(input, 3);

  if (opcode == DXIL::OpCode::FMax || opcode == DXIL::OpCode::FMin) {
    Value *temp0 =
        TrivialDxilBinaryOperation(opcode, input0, input1, hlslOP, Builder);
    Value *temp1 =
        TrivialDxilBinaryOperation(opcode, input2, input3, hlslOP, Builder);
    Value *combined =
        TrivialDxilBinaryOperation(opcode, temp0, temp1, hlslOP, Builder);
    return combined;
  } else {
    // Avg.
    Value *temp0 = Builder.CreateFAdd(input0, input1);
    Value *temp1 = Builder.CreateFAdd(input2, input3);
    Value *combined = Builder.CreateFAdd(temp0, temp1);
    Value *rcp = ConstantFP::get(input0->getType(), 0.25);
    combined = Builder.CreateFMul(combined, rcp);
    return combined;
  }
}

// 4 inputs, 2 result
Value *Apply2DQuadTessFactorOp(Value *input, DXIL::OpCode opcode,
                               hlsl::OP *hlslOP, IRBuilder<> &Builder) {
  Value *input0 = Builder.CreateExtractElement(input, (uint64_t)0);
  Value *input1 = Builder.CreateExtractElement(input, 1);
  Value *input2 = Builder.CreateExtractElement(input, 2);
  Value *input3 = Builder.CreateExtractElement(input, 3);

  if (opcode == DXIL::OpCode::FMax || opcode == DXIL::OpCode::FMin) {
    Value *temp0 =
        TrivialDxilBinaryOperation(opcode, input0, input1, hlslOP, Builder);
    Value *temp1 =
        TrivialDxilBinaryOperation(opcode, input2, input3, hlslOP, Builder);
    Value *combined = UndefValue::get(VectorType::get(input0->getType(), 2));
    combined = Builder.CreateInsertElement(combined, temp0, (uint64_t)0);
    combined = Builder.CreateInsertElement(combined, temp1, 1);
    return combined;
  } else {
    // Avg.
    Value *temp0 = Builder.CreateFAdd(input0, input1);
    Value *temp1 = Builder.CreateFAdd(input2, input3);
    Value *combined = UndefValue::get(VectorType::get(input0->getType(), 2));
    combined = Builder.CreateInsertElement(combined, temp0, (uint64_t)0);
    combined = Builder.CreateInsertElement(combined, temp1, 1);
    Constant *rcp = ConstantFP::get(input0->getType(), 0.5);
    rcp = ConstantVector::getSplat(2, rcp);
    combined = Builder.CreateFMul(combined, rcp);
    return combined;
  }
}

Value *ResolveSmallValue(Value  **pClampedResult, Value *rounded, Value *averageUnscaled,
    float cutoffVal, DXIL::TessellatorPartitioning partitionMode, hlsl::OP *hlslOP, IRBuilder<> &Builder) {
  Value  *clampedResult = *pClampedResult;
  Value *clampedVal = clampedResult;
  Value *roundedVal = rounded;
  // Do partitioning-specific clamping.
  Value *clampedAvg = ClampTessFactor(averageUnscaled, partitionMode, hlslOP, Builder);
  Constant *cutoffVals = ConstantFP::get(Type::getFloatTy(rounded->getContext()), cutoffVal);
  if (clampedAvg->getType()->isVectorTy())
    cutoffVals = ConstantVector::getSplat(clampedAvg->getType()->getVectorNumElements(), cutoffVals);
  // Limit the value.
  clampedAvg = TrivialDxilBinaryOperation(DXIL::OpCode::FMin, clampedAvg, cutoffVals, hlslOP, Builder);
  // Round up for integer/pow2 partitioning.
  Value *roundedAvg = RoundUpTessFactor(clampedAvg, partitionMode, hlslOP, Builder);

  if (rounded->getType() != cutoffVals->getType())
    cutoffVals = ConstantVector::getSplat(rounded->getType()->getVectorNumElements(), cutoffVals);
  // If the scaled value is less than three, then take the unscaled average.
  Value *lt = Builder.CreateFCmpOLT(rounded, cutoffVals);
  if (clampedAvg->getType() != clampedVal->getType())
    clampedAvg = SplatToVector(clampedAvg, clampedVal->getType(), Builder);
  *pClampedResult = Builder.CreateSelect(lt, clampedAvg, clampedVal);

  if (roundedAvg->getType() != roundedVal->getType())
    roundedAvg = SplatToVector(roundedAvg, roundedVal->getType(), Builder);
  Value *result = Builder.CreateSelect(lt, roundedAvg, roundedVal);
  return result;
}

void ResolveQuadAxes( Value  **pFinalResult, Value **pClampedResult,
    float cutoffVal, DXIL::TessellatorPartitioning partitionMode, hlsl::OP *hlslOP, IRBuilder<> &Builder) {
  Value  *finalResult = *pFinalResult;
  Value *clampedResult = *pClampedResult;

  Value *clampR = clampedResult;
  Value *finalR = finalResult;
  Type *f32Ty = Type::getFloatTy(finalR->getContext());
  Constant *cutoffVals = ConstantFP::get(f32Ty, cutoffVal);

  Value *minValsX = cutoffVals;
  Value *minValsY = RoundUpTessFactor(cutoffVals, partitionMode, hlslOP, Builder);

  Value *clampRX = Builder.CreateExtractElement(clampR, (uint64_t)0);
  Value *clampRY = Builder.CreateExtractElement(clampR, 1);
  Value *maxValsX = TrivialDxilBinaryOperation(DXIL::OpCode::FMax, clampRX, clampRY, hlslOP, Builder);

  Value *finalRX = Builder.CreateExtractElement(finalR, (uint64_t)0);
  Value *finalRY = Builder.CreateExtractElement(finalR, 1);
  Value *maxValsY = TrivialDxilBinaryOperation(DXIL::OpCode::FMax, finalRX, finalRY, hlslOP, Builder);

  // Don't go over our threshold ("final" one is rounded).
  Value * optionX = TrivialDxilBinaryOperation(DXIL::OpCode::FMin, maxValsX, minValsX, hlslOP, Builder);
  Value * optionY = TrivialDxilBinaryOperation(DXIL::OpCode::FMin, maxValsY, minValsY, hlslOP, Builder);

  Value *clampL = SplatToVector(optionX, clampR->getType(), Builder);
  Value *finalL = SplatToVector(optionY, finalR->getType(), Builder);

  cutoffVals = ConstantVector::getSplat(2, cutoffVals);
  Value *lt = Builder.CreateFCmpOLT(clampedResult, cutoffVals);
  *pClampedResult = Builder.CreateSelect(lt, clampL, clampR);
  *pFinalResult = Builder.CreateSelect(lt, finalL, finalR);
}

Value *TranslateProcessTessFactors(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                              HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  // Get partition mode 
  DXASSERT_NOMSG(helper.functionProps);
  DXASSERT(helper.functionProps->shaderKind == ShaderModel::Kind::Hull, "must be hull shader");
  DXIL::TessellatorPartitioning partition = helper.functionProps->ShaderProps.HS.partition;
  
  IRBuilder<> Builder(CI);

  DXIL::OpCode tessFactorOp = DXIL::OpCode::NumOpCodes;
  switch (IOP) {
  case IntrinsicOp::IOP_Process2DQuadTessFactorsMax:
  case IntrinsicOp::IOP_ProcessQuadTessFactorsMax:
  case IntrinsicOp::IOP_ProcessTriTessFactorsMax:
    tessFactorOp = DXIL::OpCode::FMax;
    break;
  case IntrinsicOp::IOP_Process2DQuadTessFactorsMin:
  case IntrinsicOp::IOP_ProcessQuadTessFactorsMin:
  case IntrinsicOp::IOP_ProcessTriTessFactorsMin:
    tessFactorOp = DXIL::OpCode::FMin;
    break;  
  default:
    // Default is Avg.
    break;
  }

  Value *rawEdgeFactor = CI->getArgOperand(HLOperandIndex::kProcessTessFactorRawEdgeFactor);

  Value *insideScale = CI->getArgOperand(HLOperandIndex::kProcessTessFactorInsideScale);
  // Clamp to [0.0f..1.0f], NaN->0.0f.
  Value *scales = CleanupTessFactorScale(insideScale, hlslOP, Builder);
  // Do partitioning-specific clamping.
  Value *clamped = ClampTessFactor(rawEdgeFactor, partition, hlslOP, Builder);
  // Round up for integer/pow2 partitioning.
  Value *rounded = RoundUpTessFactor(clamped, partition, hlslOP, Builder);
  // Store the output.
  Value *roundedEdgeFactor = CI->getArgOperand(HLOperandIndex::kProcessTessFactorRoundedEdgeFactor);
  Builder.CreateStore(rounded, roundedEdgeFactor);

  // Clamp to [1.0f..Inf], NaN->1.0f.
  bool isQuad = false;
  Value *clean = CleanupTessFactor(rawEdgeFactor, hlslOP, Builder);
  Value *factors = nullptr;
  switch (IOP) {
  case IntrinsicOp::IOP_Process2DQuadTessFactorsAvg:
  case IntrinsicOp::IOP_Process2DQuadTessFactorsMax:
  case IntrinsicOp::IOP_Process2DQuadTessFactorsMin:
    factors = Apply2DQuadTessFactorOp(clean, tessFactorOp, hlslOP, Builder);
    break;
  case IntrinsicOp::IOP_ProcessQuadTessFactorsAvg:
  case IntrinsicOp::IOP_ProcessQuadTessFactorsMax:
  case IntrinsicOp::IOP_ProcessQuadTessFactorsMin:
    factors = ApplyQuadTessFactorOp(clean, tessFactorOp, hlslOP, Builder);
    isQuad = true;
    break;
  case IntrinsicOp::IOP_ProcessTriTessFactorsAvg:
  case IntrinsicOp::IOP_ProcessTriTessFactorsMax:
  case IntrinsicOp::IOP_ProcessTriTessFactorsMin:
    factors = ApplyTriTessFactorOp(clean, tessFactorOp, hlslOP, Builder);
    break;
  default:
    DXASSERT(0, "invalid opcode for ProcessTessFactor");
    break;
  }

  Value *scaledI = nullptr;
  if (scales->getType() == factors->getType())
    scaledI = Builder.CreateFMul(factors, scales);
  else {
    Value *vecFactors = SplatToVector(factors, scales->getType(), Builder);
    scaledI = Builder.CreateFMul(vecFactors, scales);
  }

  // Do partitioning-specific clamping.
  Value *clampedI = ClampTessFactor(scaledI, partition, hlslOP, Builder);
  
  // Round up for integer/pow2 partitioning.
  Value *roundedI = RoundUpTessFactor(clampedI, partition, hlslOP, Builder);

  Value *finalI = roundedI;

  if (partition == DXIL::TessellatorPartitioning::FractionalOdd) {
    // If not max, set to AVG.
    if (tessFactorOp != DXIL::OpCode::FMax)
      tessFactorOp = DXIL::OpCode::NumOpCodes;

    bool b2D = false;
    Value *avgFactorsI = nullptr;
    switch (IOP) {
    case IntrinsicOp::IOP_Process2DQuadTessFactorsAvg:
    case IntrinsicOp::IOP_Process2DQuadTessFactorsMax:
    case IntrinsicOp::IOP_Process2DQuadTessFactorsMin:
      avgFactorsI = Apply2DQuadTessFactorOp(clean, tessFactorOp, hlslOP, Builder);
      b2D = true;
      break;
    case IntrinsicOp::IOP_ProcessQuadTessFactorsAvg:
    case IntrinsicOp::IOP_ProcessQuadTessFactorsMax:
    case IntrinsicOp::IOP_ProcessQuadTessFactorsMin:
      avgFactorsI = ApplyQuadTessFactorOp(clean, tessFactorOp, hlslOP, Builder);
      break;
    case IntrinsicOp::IOP_ProcessTriTessFactorsAvg:
    case IntrinsicOp::IOP_ProcessTriTessFactorsMax:
    case IntrinsicOp::IOP_ProcessTriTessFactorsMin:
      avgFactorsI = ApplyTriTessFactorOp(clean, tessFactorOp, hlslOP, Builder);
      break;
    default:
      DXASSERT(0, "invalid opcode for ProcessTessFactor");
      break;
    }

    finalI =
        ResolveSmallValue(/*inout*/&clampedI, roundedI, avgFactorsI, /*cufoff*/ 3.0,
                          partition, hlslOP, Builder);

    if (b2D)
      ResolveQuadAxes(/*inout*/&finalI, /*inout*/&clampedI, /*cutoff*/3.0, partition, hlslOP, Builder);
  }

  Value *unroundedInsideFactor = CI->getArgOperand(HLOperandIndex::kProcessTessFactorUnRoundedInsideFactor);
  Type *outFactorTy = unroundedInsideFactor->getType()->getPointerElementType();
  if (outFactorTy != clampedI->getType()) {
    DXASSERT(isQuad, "quad only write one channel of out factor");
    (void)isQuad;
    clampedI = Builder.CreateExtractElement(clampedI, (uint64_t)0);
    // Splat clampedI to float2.
    clampedI = SplatToVector(clampedI, outFactorTy, Builder);
  }
  Builder.CreateStore(clampedI, unroundedInsideFactor);

  Value *roundedInsideFactor = CI->getArgOperand(HLOperandIndex::kProcessTessFactorRoundedInsideFactor);  
  if (outFactorTy != finalI->getType()) {
    DXASSERT(isQuad, "quad only write one channel of out factor");
    finalI = Builder.CreateExtractElement(finalI, (uint64_t)0);
    // Splat finalI to float2.
    finalI = SplatToVector(finalI, outFactorTy, Builder);
  }
  Builder.CreateStore(finalI, roundedInsideFactor);
  return nullptr;
}


}

// Ray Tracing.
namespace {
Value *TranslateReportIntersection(CallInst *CI, IntrinsicOp IOP,
                                   OP::OpCode opcode,
                                   HLOperationLowerHelper &helper,
                                   HLObjectOperationLowerHelper *pObjHelper,
                                   bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  Value *THit = CI->getArgOperand(HLOperandIndex::kTrinaryOpSrc0Idx);
  Value *HitKind = CI->getArgOperand(HLOperandIndex::kTrinaryOpSrc1Idx);
  Value *Attr = CI->getArgOperand(HLOperandIndex::kTrinaryOpSrc2Idx);
  Value *opArg = hlslOP->GetU32Const(static_cast<unsigned>(opcode));

  Type *Ty = Attr->getType();
  Function *F = hlslOP->GetOpFunc(opcode, Ty);

  IRBuilder<> Builder(CI);
  return Builder.CreateCall(F, {opArg, THit, HitKind, Attr});
}

Value *TranslateCallShader(CallInst *CI, IntrinsicOp IOP,
                                   OP::OpCode opcode,
                                   HLOperationLowerHelper &helper,
                                   HLObjectOperationLowerHelper *pObjHelper,
                                   bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  Value *ShaderIndex = CI->getArgOperand(HLOperandIndex::kBinaryOpSrc0Idx);
  Value *Parameter = CI->getArgOperand(HLOperandIndex::kBinaryOpSrc1Idx);
  Value *opArg = hlslOP->GetU32Const(static_cast<unsigned>(opcode));

  Type *Ty = Parameter->getType();
  Function *F = hlslOP->GetOpFunc(opcode, Ty);

  IRBuilder<> Builder(CI);
  return Builder.CreateCall(F, {opArg, ShaderIndex, Parameter});
}

Value *TranslateTraceRay(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                         HLOperationLowerHelper &helper,
                         HLObjectOperationLowerHelper *pObjHelper,
                         bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;

  Value *rayDesc = CI->getArgOperand(HLOperandIndex::kTraceRayRayDescOpIdx);
  Value *payLoad = CI->getArgOperand(HLOperandIndex::kTraceRayPayLoadOpIdx);

  Value *opArg = hlslOP->GetU32Const(static_cast<unsigned>(opcode));

  Value *Args[DXIL::OperandIndex::kTraceRayNumOp];
  Args[0] = opArg;
  for (unsigned i = 1; i < HLOperandIndex::kTraceRayRayDescOpIdx; i++) {
    Args[i] = CI->getArgOperand(i);
  }
  IRBuilder<> Builder(CI);
  // struct RayDesc
  //{
  //    float3 Origin;
  //    float  TMin;
  //    float3 Direction;
  //    float  TMax;
  //};
  Value *zeroIdx = hlslOP->GetU32Const(0);
  Value *origin = Builder.CreateGEP(rayDesc, {zeroIdx, zeroIdx});
  origin = Builder.CreateLoad(origin);
  unsigned index = DXIL::OperandIndex::kTraceRayRayDescOpIdx;
  Args[index++] = Builder.CreateExtractElement(origin, (uint64_t)0);
  Args[index++] = Builder.CreateExtractElement(origin, 1);
  Args[index++] = Builder.CreateExtractElement(origin, 2);

  Value *tmin = Builder.CreateGEP(rayDesc, {zeroIdx, hlslOP->GetU32Const(1)});
  tmin = Builder.CreateLoad(tmin);
  Args[index++] = tmin;

  Value *direction = Builder.CreateGEP(rayDesc, {zeroIdx, hlslOP->GetU32Const(2)});
  direction = Builder.CreateLoad(direction);

  Args[index++] = Builder.CreateExtractElement(direction, (uint64_t)0);
  Args[index++] = Builder.CreateExtractElement(direction, 1);
  Args[index++] = Builder.CreateExtractElement(direction, 2);

  Value *tmax = Builder.CreateGEP(rayDesc, {zeroIdx, hlslOP->GetU32Const(3)});
  tmax = Builder.CreateLoad(tmax);
  Args[index++] = tmax;

  Args[DXIL::OperandIndex::kTraceRayPayloadOpIdx] = payLoad;

  Type *Ty = payLoad->getType();
  Function *F = hlslOP->GetOpFunc(opcode, Ty);


  return Builder.CreateCall(F, Args);
}

// RayQuery methods

Value *TranslateAllocateRayQuery(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                                 HLOperationLowerHelper &helper,
                                 HLObjectOperationLowerHelper *pObjHelper,
                                 bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  Value *refArgs[] = {nullptr, CI->getOperand(1)};
  return TrivialDxilOperation(opcode, refArgs, helper.voidTy, CI, hlslOP);
}

Value *TranslateTraceRayInline(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                         HLOperationLowerHelper &helper,
                         HLObjectOperationLowerHelper *pObjHelper,
                         bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;

  Value *opArg = hlslOP->GetU32Const(static_cast<unsigned>(opcode));

  Value *Args[DXIL::OperandIndex::kTraceRayInlineNumOp];
  Args[0] = opArg;
  for (unsigned i = 1; i < HLOperandIndex::kTraceRayInlineRayDescOpIdx; i++) {
    Args[i] = CI->getArgOperand(i);
  }

  IRBuilder<> Builder(CI);
  unsigned hlIndex = HLOperandIndex::kTraceRayInlineRayDescOpIdx;
  unsigned index = DXIL::OperandIndex::kTraceRayInlineRayDescOpIdx;

  // struct RayDesc
  //{
  //    float3 Origin;
  Value *origin = CI->getArgOperand(hlIndex++);
  Args[index++] = Builder.CreateExtractElement(origin, (uint64_t)0);
  Args[index++] = Builder.CreateExtractElement(origin, 1);
  Args[index++] = Builder.CreateExtractElement(origin, 2);
  //    float  TMin;
  Args[index++] = CI->getArgOperand(hlIndex++);
  //    float3 Direction;
  Value *direction = CI->getArgOperand(hlIndex++);
  Args[index++] = Builder.CreateExtractElement(direction, (uint64_t)0);
  Args[index++] = Builder.CreateExtractElement(direction, 1);
  Args[index++] = Builder.CreateExtractElement(direction, 2);
  //    float  TMax;
  Args[index++] = CI->getArgOperand(hlIndex++);
  //};

  DXASSERT_NOMSG(index == DXIL::OperandIndex::kTraceRayInlineNumOp);

  Function *F = hlslOP->GetOpFunc(opcode, Builder.getVoidTy());

  return Builder.CreateCall(F, Args);
}

Value *TranslateCommitProceduralPrimitiveHit(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                         HLOperationLowerHelper &helper,
                         HLObjectOperationLowerHelper *pObjHelper,
                         bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  Value *THit = CI->getArgOperand(HLOperandIndex::kBinaryOpSrc1Idx);
  Value *opArg = hlslOP->GetU32Const(static_cast<unsigned>(opcode));
  Value *handle = CI->getArgOperand(HLOperandIndex::kHandleOpIdx);

  Value *Args[] = {opArg, handle, THit};

  IRBuilder<> Builder(CI);
  Function *F = hlslOP->GetOpFunc(opcode, Builder.getVoidTy());

  return Builder.CreateCall(F, Args);
}

Value *TranslateGenericRayQueryMethod(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                         HLOperationLowerHelper &helper,
                         HLObjectOperationLowerHelper *pObjHelper,
                         bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;

  Value *opArg = hlslOP->GetU32Const(static_cast<unsigned>(opcode));
  Value *handle = CI->getArgOperand(HLOperandIndex::kHandleOpIdx);

  IRBuilder<> Builder(CI);
  Function *F = hlslOP->GetOpFunc(opcode, CI->getType());

  return Builder.CreateCall(F, {opArg, handle});
}

Value *TranslateRayQueryMatrix3x4Operation(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                         HLOperationLowerHelper &helper,
                         HLObjectOperationLowerHelper *pObjHelper,
                         bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  VectorType *Ty = cast<VectorType>(CI->getType());
  Value *handle = CI->getArgOperand(HLOperandIndex::kHandleOpIdx);
  uint32_t rVals[] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
  Constant *rows = ConstantDataVector::get(CI->getContext(), rVals);
  uint8_t cVals[] = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
  Constant *cols = ConstantDataVector::get(CI->getContext(), cVals);
  Value *retVal =
      TrivialDxilOperation(opcode, {nullptr, handle, rows, cols}, Ty, CI, hlslOP);
  return retVal;
}

Value *TranslateRayQueryTransposedMatrix3x4Operation(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                                                  HLOperationLowerHelper &helper,
                                                  HLObjectOperationLowerHelper *pObjHelper,
                                                  bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP; 
  VectorType *Ty = cast<VectorType>(CI->getType());
  Value *handle = CI->getArgOperand(HLOperandIndex::kHandleOpIdx);
  uint32_t rVals[] = { 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2 };
  Constant *rows = ConstantDataVector::get(CI->getContext(), rVals);
  uint8_t cVals[] = { 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3 };
  Constant *cols = ConstantDataVector::get(CI->getContext(), cVals);
  Value *retVal =
      TrivialDxilOperation(opcode, {nullptr, handle, rows, cols}, Ty, CI, hlslOP);
  return retVal;
}

Value *TranslateRayQueryFloat2Getter(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                         HLOperationLowerHelper &helper,
                         HLObjectOperationLowerHelper *pObjHelper,
                         bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  VectorType *Ty = cast<VectorType>(CI->getType());
  Value *handle = CI->getArgOperand(HLOperandIndex::kHandleOpIdx);
  uint8_t elementVals[] = {0, 1};
  Constant *element = ConstantDataVector::get(CI->getContext(), elementVals);
  Value *retVal =
      TrivialDxilOperation(opcode, {nullptr, handle, element}, Ty, CI, hlslOP);
  return retVal;
}

Value *TranslateRayQueryFloat3Getter(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                         HLOperationLowerHelper &helper,
                         HLObjectOperationLowerHelper *pObjHelper,
                         bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  VectorType *Ty = cast<VectorType>(CI->getType());
  Value *handle = CI->getArgOperand(HLOperandIndex::kHandleOpIdx);
  uint8_t elementVals[] = {0, 1, 2};
  Constant *element = ConstantDataVector::get(CI->getContext(), elementVals);
  Value *retVal =
      TrivialDxilOperation(opcode, {nullptr, handle, element}, Ty, CI, hlslOP);
  return retVal;
}

Value *TranslateNoArgVectorOperation(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                         HLOperationLowerHelper &helper,
                         HLObjectOperationLowerHelper *pObjHelper,
                         bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  VectorType *Ty = cast<VectorType>(CI->getType());
  uint8_t vals[] = {0,1,2,3};
  Constant *src = ConstantDataVector::get(CI->getContext(), vals);
  Value *retVal = TrivialDxilOperation(opcode, {nullptr, src}, Ty, CI, hlslOP);
  return retVal;
}

Value *TranslateNoArgMatrix3x4Operation(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                         HLOperationLowerHelper &helper,
                         HLObjectOperationLowerHelper *pObjHelper,
                         bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  VectorType *Ty = cast<VectorType>(CI->getType());
  uint32_t rVals[] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
  Constant *rows = ConstantDataVector::get(CI->getContext(), rVals);
  uint8_t cVals[] = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
  Constant *cols = ConstantDataVector::get(CI->getContext(), cVals);
  Value *retVal =
      TrivialDxilOperation(opcode, {nullptr, rows, cols}, Ty, CI, hlslOP);
  return retVal;
}

Value *TranslateNoArgTransposedMatrix3x4Operation(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                                                  HLOperationLowerHelper &helper,
                                                  HLObjectOperationLowerHelper *pObjHelper,
                                                  bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  VectorType *Ty = cast<VectorType>(CI->getType());
  uint32_t rVals[] = { 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2 };
  Constant *rows = ConstantDataVector::get(CI->getContext(), rVals);
  uint8_t cVals[] = { 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3 };
  Constant *cols = ConstantDataVector::get(CI->getContext(), cVals);
  Value *retVal =
    TrivialDxilOperation(opcode, { nullptr, rows, cols }, Ty, CI, hlslOP);
  return retVal;
}

Value *TranslateNoArgNoReturnPreserveOutput(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
  HLOperationLowerHelper &helper, HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  Instruction *pResult = cast<Instruction>(
    TrivialNoArgOperation(CI, IOP, opcode, helper, pObjHelper, Translated));
  // HL intrinsic must have had a return injected just after the call.
  // SROA_Parameter_HLSL will copy from alloca to output just before each return.
  // Now move call after the copy and just before the return.
  if (isa<ReturnInst>(pResult->getNextNode()))
    return pResult;
  ReturnInst *RetI = cast<ReturnInst>(pResult->getParent()->getTerminator());
  pResult->removeFromParent();
  pResult->insertBefore(RetI);
  return pResult;
}

// Special half dot2 with accumulate to float
Value *TranslateDot2Add(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                        HLOperationLowerHelper &helper,
                        HLObjectOperationLowerHelper *pObjHelper,
                        bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  Value *src0 = CI->getArgOperand(HLOperandIndex::kTrinaryOpSrc0Idx);
  const unsigned vecSize = 2;
  DXASSERT(src0->getType()->isVectorTy() &&
               vecSize == src0->getType()->getVectorNumElements() &&
               src0->getType()->getScalarType()->isHalfTy(),
           "otherwise, unexpected input dimension or component type");

  Value *src1 = CI->getArgOperand(HLOperandIndex::kTrinaryOpSrc1Idx);
  DXASSERT(src0->getType() == src1->getType(),
           "otherwise, mismatched argument types");
  Value *accArg = CI->getArgOperand(HLOperandIndex::kTrinaryOpSrc2Idx);
  Type *accTy = accArg->getType();
  DXASSERT(!accTy->isVectorTy() && accTy->isFloatTy(),
           "otherwise, unexpected accumulator type");
  IRBuilder<> Builder(CI);

  Function *dxilFunc = hlslOP->GetOpFunc(opcode, accTy);
  Constant *opArg = hlslOP->GetU32Const((unsigned)opcode);

  SmallVector<Value *, 6> args;
  args.emplace_back(opArg);
  args.emplace_back(accArg);
  for (unsigned i = 0; i < vecSize; i++)
    args.emplace_back(Builder.CreateExtractElement(src0, i));
  for (unsigned i = 0; i < vecSize; i++)
    args.emplace_back(Builder.CreateExtractElement(src1, i));
  return Builder.CreateCall(dxilFunc, args);
}

Value *TranslateDot4AddPacked(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                              HLOperationLowerHelper &helper,
                              HLObjectOperationLowerHelper *pObjHelper,
                              bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  Value *src0 = CI->getArgOperand(HLOperandIndex::kTrinaryOpSrc0Idx);
  DXASSERT(
      !src0->getType()->isVectorTy() && src0->getType()->isIntegerTy(32),
      "otherwise, unexpected vector support in high level intrinsic template");
  Value *src1 = CI->getArgOperand(HLOperandIndex::kTrinaryOpSrc1Idx);
  DXASSERT(src0->getType() == src1->getType(), "otherwise, mismatched argument types");
  Value *accArg = CI->getArgOperand(HLOperandIndex::kTrinaryOpSrc2Idx);
  Type *accTy = accArg->getType();
  DXASSERT(!accTy->isVectorTy() && accTy->isIntegerTy(32),
    "otherwise, unexpected vector support in high level intrinsic template");
  IRBuilder<> Builder(CI);

  Function *dxilFunc = hlslOP->GetOpFunc(opcode, accTy);
  Constant *opArg = hlslOP->GetU32Const((unsigned)opcode);
  return Builder.CreateCall(dxilFunc, { opArg, accArg, src0, src1 });
}

Value *TranslatePack(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                     HLOperationLowerHelper &helper, 
                     HLObjectOperationLowerHelper *pObjHelper,
                     bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;

  Value *val = CI->getArgOperand(HLOperandIndex::kUnaryOpSrc0Idx);
  Type *valTy = val->getType();
  Type *eltTy = valTy->getScalarType();

  DXASSERT(valTy->isVectorTy() && valTy->getVectorNumElements() == 4 && eltTy->isIntegerTy() &&
    (eltTy->getIntegerBitWidth() == 32 || eltTy->getIntegerBitWidth() == 16),
    "otherwise, unexpected input dimension or component type");

  DXIL::PackMode packMode = DXIL::PackMode::Trunc;
  switch (IOP) {
    case hlsl::IntrinsicOp::IOP_pack_clamp_s8: 
      packMode = DXIL::PackMode::SClamp;
      break;
    case hlsl::IntrinsicOp::IOP_pack_clamp_u8:
      packMode = DXIL::PackMode::UClamp;
      break;
    case hlsl::IntrinsicOp::IOP_pack_s8:
    case hlsl::IntrinsicOp::IOP_pack_u8:
      packMode = DXIL::PackMode::Trunc;
      break;
    default:
      DXASSERT(false, "unexpected opcode");
      break;
  }

  IRBuilder<> Builder(CI);
  Function *dxilFunc = hlslOP->GetOpFunc(opcode, eltTy);
  Constant *opArg = hlslOP->GetU32Const((unsigned)opcode);
  Constant *packModeArg = hlslOP->GetU8Const((unsigned)packMode);

  Value *elt0 = Builder.CreateExtractElement(val, (uint64_t)0);
  Value *elt1 = Builder.CreateExtractElement(val, (uint64_t)1);
  Value *elt2 = Builder.CreateExtractElement(val, (uint64_t)2);
  Value *elt3 = Builder.CreateExtractElement(val, (uint64_t)3);
  return Builder.CreateCall(dxilFunc, { opArg, packModeArg, elt0, elt1, elt2, elt3 });
}

Value *TranslateUnpack(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                     HLOperationLowerHelper &helper, 
                     HLObjectOperationLowerHelper *pObjHelper,
                     bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;

  Value *packedVal = CI->getArgOperand(HLOperandIndex::kUnaryOpSrc0Idx);
  DXASSERT(!packedVal->getType()->isVectorTy() && packedVal->getType()->isIntegerTy(32),
    "otherwise, unexpected vector support in high level intrinsic template");

  Type *overloadType = nullptr;
  DXIL::UnpackMode unpackMode = DXIL::UnpackMode::Unsigned;
  switch (IOP) {
    case hlsl::IntrinsicOp::IOP_unpack_s8s32:
      unpackMode = DXIL::UnpackMode::Signed;
      overloadType = helper.i32Ty;
      break;
    case hlsl::IntrinsicOp::IOP_unpack_u8u32:
      unpackMode = DXIL::UnpackMode::Unsigned;
      overloadType = helper.i32Ty;
      break;
    case hlsl::IntrinsicOp::IOP_unpack_s8s16:
      unpackMode = DXIL::UnpackMode::Signed;
      overloadType = helper.i16Ty;
      break;
    case hlsl::IntrinsicOp::IOP_unpack_u8u16:
      unpackMode = DXIL::UnpackMode::Unsigned;
      overloadType = helper.i16Ty;
      break;
    default:
      DXASSERT(false, "unexpected opcode");
      break;
  }
  
  IRBuilder<> Builder(CI);
  Function *dxilFunc = hlslOP->GetOpFunc(opcode, overloadType);
  Constant *opArg = hlslOP->GetU32Const((unsigned)opcode);
  Constant *unpackModeArg = hlslOP->GetU8Const((unsigned)unpackMode);
  Value *Res = Builder.CreateCall(dxilFunc, { opArg, unpackModeArg , packedVal });

  // Convert the final aggregate into a vector to make the types match
  const unsigned vecSize = 4;
  Value *ResVec = UndefValue::get(CI->getType());
  for (unsigned i = 0; i < vecSize; ++i) {
    Value *Elt = Builder.CreateExtractValue(Res, i);
    ResVec = Builder.CreateInsertElement(ResVec, Elt, i);
  }
  return ResVec;
}

} // namespace

// Resource Handle.
namespace {
Value *TranslateGetHandleFromHeap(CallInst *CI, IntrinsicOp IOP,
                                  DXIL::OpCode opcode,
                                  HLOperationLowerHelper &helper,
                                  HLObjectOperationLowerHelper *pObjHelper,
                                  bool &Translated) {
  hlsl::OP &hlslOP = helper.hlslOP;
  Function *dxilFunc = hlslOP.GetOpFunc(opcode, helper.voidTy);
  IRBuilder<> Builder(CI);
  Value *opArg = ConstantInt::get(helper.i32Ty, (unsigned)opcode);
  return Builder.CreateCall(
      dxilFunc, {opArg, CI->getArgOperand(HLOperandIndex::kBinaryOpSrc0Idx),
                 CI->getArgOperand(HLOperandIndex::kBinaryOpSrc1Idx),
                 // TODO: update nonUniformIndex later.
                 Builder.getInt1(false)});
}
}

// Translate and/or/select intrinsics
namespace {
Value *TranslateAnd(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                     HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  Value *x = CI->getArgOperand(HLOperandIndex::kBinaryOpSrc0Idx);
  Value *y = CI->getArgOperand(HLOperandIndex::kBinaryOpSrc1Idx);
  Type *Ty = CI->getType();
  Type *EltTy = Ty->getScalarType();
  IRBuilder<> Builder(CI);

  if (Ty != EltTy) {
    Value *Result = UndefValue::get(Ty);
    for (unsigned i = 0; i < Ty->getVectorNumElements(); i++) {
      Value *EltX = Builder.CreateExtractElement(x, i);
      Value *EltY = Builder.CreateExtractElement(y, i);
      Value *tmp = Builder.CreateAnd(EltX, EltY);
      Result = Builder.CreateInsertElement(Result, tmp, i);
    }
    return Result;
  }
  return Builder.CreateAnd(x, y);
}
Value *TranslateOr(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                     HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  Value *x = CI->getArgOperand(HLOperandIndex::kBinaryOpSrc0Idx);
  Value *y = CI->getArgOperand(HLOperandIndex::kBinaryOpSrc1Idx);
  Type *Ty = CI->getType();
  Type *EltTy = Ty->getScalarType();
  IRBuilder<> Builder(CI);

  if (Ty != EltTy) {
    Value *Result = UndefValue::get(Ty);
    for (unsigned i = 0; i < Ty->getVectorNumElements(); i++) {
      Value *EltX = Builder.CreateExtractElement(x, i);
      Value *EltY = Builder.CreateExtractElement(y, i);
      Value *tmp = Builder.CreateOr(EltX, EltY);
      Result = Builder.CreateInsertElement(Result, tmp, i);
    }
    return Result;
  }
  return Builder.CreateOr(x, y);
}
Value *TranslateSelect(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                     HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  Value *cond = CI->getArgOperand(HLOperandIndex::kTrinaryOpSrc0Idx);
  Value *t = CI->getArgOperand(HLOperandIndex::kTrinaryOpSrc1Idx);
  Value *f = CI->getArgOperand(HLOperandIndex::kTrinaryOpSrc2Idx);
  Type *Ty = CI->getType();
  Type *EltTy = Ty->getScalarType();
  IRBuilder<> Builder(CI);

  if (Ty != EltTy) {
    Value *Result = UndefValue::get(Ty);
    for (unsigned i = 0; i < Ty->getVectorNumElements(); i++) {
      Value *EltCond = Builder.CreateExtractElement(cond, i);
      Value *EltTrue = Builder.CreateExtractElement(t, i);
      Value *EltFalse = Builder.CreateExtractElement(f, i);
      Value *tmp = Builder.CreateSelect(EltCond, EltTrue, EltFalse);
      Result = Builder.CreateInsertElement(Result, tmp, i);
    }
    return Result;
  }
  return Builder.CreateSelect(cond, t, f);
}
}

// Lower table.
namespace {

Value *EmptyLower(CallInst *CI, IntrinsicOp IOP, DXIL::OpCode opcode,
                  HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  Translated = false;
  dxilutil::EmitErrorOnInstruction(CI, "Unsupported intrinsic.");
  return nullptr;
}

// SPIRV change starts
#ifdef ENABLE_SPIRV_CODEGEN
Value *UnsupportedVulkanIntrinsic(CallInst *CI, IntrinsicOp IOP,
                                  DXIL::OpCode opcode,
                                  HLOperationLowerHelper &helper,
                                  HLObjectOperationLowerHelper *pObjHelper,
                                  bool &Translated) {
  Translated = false;
  dxilutil::EmitErrorOnInstruction(CI, "Unsupported Vulkan intrinsic.");
  return nullptr;
}
#endif // ENABLE_SPIRV_CODEGEN
// SPIRV change ends

Value *StreamOutputLower(CallInst *CI, IntrinsicOp IOP, DXIL::OpCode opcode,
                         HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  // Translated in DxilGenerationPass::GenerateStreamOutputOperation.
  // Do nothing here.
  // Mark not translated.
  Translated = false;
  return nullptr;
}

// This table has to match IntrinsicOp orders
IntrinsicLower gLowerTable[] = {
    {IntrinsicOp::IOP_AcceptHitAndEndSearch, TranslateNoArgNoReturnPreserveOutput, DXIL::OpCode::AcceptHitAndEndSearch},
    {IntrinsicOp::IOP_AddUint64,  TranslateAddUint64,  DXIL::OpCode::UAddc},
    {IntrinsicOp::IOP_AllMemoryBarrier, TrivialBarrier, DXIL::OpCode::Barrier},
    {IntrinsicOp::IOP_AllMemoryBarrierWithGroupSync, TrivialBarrier, DXIL::OpCode::Barrier},
    {IntrinsicOp::IOP_AllocateRayQuery, TranslateAllocateRayQuery, DXIL::OpCode::AllocateRayQuery},
    {IntrinsicOp::IOP_CallShader, TranslateCallShader, DXIL::OpCode::CallShader},
    {IntrinsicOp::IOP_CheckAccessFullyMapped, TranslateCheckAccess, DXIL::OpCode::CheckAccessFullyMapped},
    {IntrinsicOp::IOP_CreateResourceFromHeap, TranslateGetHandleFromHeap, DXIL::OpCode::CreateHandleFromHeap},
    {IntrinsicOp::IOP_D3DCOLORtoUBYTE4, TranslateD3DColorToUByte4, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_DeviceMemoryBarrier, TrivialBarrier, DXIL::OpCode::Barrier},
    {IntrinsicOp::IOP_DeviceMemoryBarrierWithGroupSync, TrivialBarrier, DXIL::OpCode::Barrier},
    {IntrinsicOp::IOP_DispatchMesh, TrivialDispatchMesh, DXIL::OpCode::DispatchMesh },
    {IntrinsicOp::IOP_DispatchRaysDimensions, TranslateNoArgVectorOperation, DXIL::OpCode::DispatchRaysDimensions},
    {IntrinsicOp::IOP_DispatchRaysIndex, TranslateNoArgVectorOperation, DXIL::OpCode::DispatchRaysIndex},
    {IntrinsicOp::IOP_EvaluateAttributeAtSample, TranslateEvalSample, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_EvaluateAttributeCentroid, TranslateEvalCentroid, DXIL::OpCode::EvalCentroid},
    {IntrinsicOp::IOP_EvaluateAttributeSnapped, TranslateEvalSnapped, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_GeometryIndex, TrivialNoArgWithRetOperation, DXIL::OpCode::GeometryIndex},
    {IntrinsicOp::IOP_GetAttributeAtVertex, TranslateGetAttributeAtVertex, DXIL::OpCode::AttributeAtVertex},
    {IntrinsicOp::IOP_GetRenderTargetSampleCount, TrivialNoArgOperation, DXIL::OpCode::RenderTargetGetSampleCount},
    {IntrinsicOp::IOP_GetRenderTargetSamplePosition, TranslateGetRTSamplePos, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_GroupMemoryBarrier, TrivialBarrier, DXIL::OpCode::Barrier},
    {IntrinsicOp::IOP_GroupMemoryBarrierWithGroupSync, TrivialBarrier, DXIL::OpCode::Barrier},
    {IntrinsicOp::IOP_HitKind, TrivialNoArgWithRetOperation, DXIL::OpCode::HitKind},
    {IntrinsicOp::IOP_IgnoreHit, TranslateNoArgNoReturnPreserveOutput, DXIL::OpCode::IgnoreHit},
    {IntrinsicOp::IOP_InstanceID, TrivialNoArgWithRetOperation, DXIL::OpCode::InstanceID},
    {IntrinsicOp::IOP_InstanceIndex, TrivialNoArgWithRetOperation, DXIL::OpCode::InstanceIndex},
    {IntrinsicOp::IOP_InterlockedAdd, TranslateIopAtomicBinaryOperation, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_InterlockedAnd, TranslateIopAtomicBinaryOperation, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_InterlockedCompareExchange, TranslateIopAtomicCmpXChg, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_InterlockedCompareExchangeFloatBitwise, TranslateIopAtomicCmpXChg, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_InterlockedCompareStore, TranslateIopAtomicCmpXChg, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_InterlockedCompareStoreFloatBitwise, TranslateIopAtomicCmpXChg, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_InterlockedExchange, TranslateIopAtomicBinaryOperation, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_InterlockedMax, TranslateIopAtomicBinaryOperation, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_InterlockedMin, TranslateIopAtomicBinaryOperation, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_InterlockedOr, TranslateIopAtomicBinaryOperation, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_InterlockedXor, TranslateIopAtomicBinaryOperation, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_IsHelperLane, TrivialNoArgWithRetOperation, DXIL::OpCode::IsHelperLane},
    {IntrinsicOp::IOP_NonUniformResourceIndex, TranslateNonUniformResourceIndex, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_ObjectRayDirection, TranslateNoArgVectorOperation, DXIL::OpCode::ObjectRayDirection},
    {IntrinsicOp::IOP_ObjectRayOrigin, TranslateNoArgVectorOperation, DXIL::OpCode::ObjectRayOrigin},
    {IntrinsicOp::IOP_ObjectToWorld, TranslateNoArgMatrix3x4Operation, DXIL::OpCode::ObjectToWorld},
    {IntrinsicOp::IOP_ObjectToWorld3x4, TranslateNoArgMatrix3x4Operation, DXIL::OpCode::ObjectToWorld},
    {IntrinsicOp::IOP_ObjectToWorld4x3, TranslateNoArgTransposedMatrix3x4Operation, DXIL::OpCode::ObjectToWorld},
    {IntrinsicOp::IOP_PrimitiveIndex, TrivialNoArgWithRetOperation, DXIL::OpCode::PrimitiveIndex},
    {IntrinsicOp::IOP_Process2DQuadTessFactorsAvg, TranslateProcessTessFactors, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_Process2DQuadTessFactorsMax, TranslateProcessTessFactors, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_Process2DQuadTessFactorsMin, TranslateProcessTessFactors, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_ProcessIsolineTessFactors, TranslateProcessIsolineTessFactors, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_ProcessQuadTessFactorsAvg, TranslateProcessTessFactors, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_ProcessQuadTessFactorsMax, TranslateProcessTessFactors, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_ProcessQuadTessFactorsMin, TranslateProcessTessFactors, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_ProcessTriTessFactorsAvg, TranslateProcessTessFactors, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_ProcessTriTessFactorsMax, TranslateProcessTessFactors, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_ProcessTriTessFactorsMin, TranslateProcessTessFactors, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_QuadAll, TranslateQuadAnyAll, DXIL::OpCode::QuadVote},
    {IntrinsicOp::IOP_QuadAny, TranslateQuadAnyAll, DXIL::OpCode::QuadVote},
    {IntrinsicOp::IOP_QuadReadAcrossDiagonal, TranslateQuadReadAcross, DXIL::OpCode::QuadOp},
    {IntrinsicOp::IOP_QuadReadAcrossX, TranslateQuadReadAcross, DXIL::OpCode::QuadOp},
    {IntrinsicOp::IOP_QuadReadAcrossY, TranslateQuadReadAcross, DXIL::OpCode::QuadOp},
    {IntrinsicOp::IOP_QuadReadLaneAt,  TranslateQuadReadLaneAt, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_RayFlags, TrivialNoArgWithRetOperation, DXIL::OpCode::RayFlags},
    {IntrinsicOp::IOP_RayTCurrent, TrivialNoArgWithRetOperation, DXIL::OpCode::RayTCurrent},
    {IntrinsicOp::IOP_RayTMin, TrivialNoArgWithRetOperation, DXIL::OpCode::RayTMin},
    {IntrinsicOp::IOP_ReportHit, TranslateReportIntersection, DXIL::OpCode::ReportHit},
    {IntrinsicOp::IOP_SetMeshOutputCounts, TrivialSetMeshOutputCounts, DXIL::OpCode::SetMeshOutputCounts},
    {IntrinsicOp::IOP_TraceRay, TranslateTraceRay, DXIL::OpCode::TraceRay},
    {IntrinsicOp::IOP_WaveActiveAllEqual, TranslateWaveAllEqual, DXIL::OpCode::WaveActiveAllEqual},
    {IntrinsicOp::IOP_WaveActiveAllTrue, TranslateWaveA2B, DXIL::OpCode::WaveAllTrue},
    {IntrinsicOp::IOP_WaveActiveAnyTrue, TranslateWaveA2B, DXIL::OpCode::WaveAnyTrue},
    {IntrinsicOp::IOP_WaveActiveBallot, TranslateWaveBallot, DXIL::OpCode::WaveActiveBallot},
    {IntrinsicOp::IOP_WaveActiveBitAnd, TranslateWaveA2A, DXIL::OpCode::WaveActiveBit},
    {IntrinsicOp::IOP_WaveActiveBitOr, TranslateWaveA2A, DXIL::OpCode::WaveActiveBit},
    {IntrinsicOp::IOP_WaveActiveBitXor, TranslateWaveA2A, DXIL::OpCode::WaveActiveBit},
    {IntrinsicOp::IOP_WaveActiveCountBits, TranslateWaveA2B, DXIL::OpCode::WaveAllBitCount},
    {IntrinsicOp::IOP_WaveActiveMax, TranslateWaveA2A, DXIL::OpCode::WaveActiveOp},
    {IntrinsicOp::IOP_WaveActiveMin, TranslateWaveA2A, DXIL::OpCode::WaveActiveOp},
    {IntrinsicOp::IOP_WaveActiveProduct, TranslateWaveA2A, DXIL::OpCode::WaveActiveOp},
    {IntrinsicOp::IOP_WaveActiveSum, TranslateWaveA2A, DXIL::OpCode::WaveActiveOp},
    {IntrinsicOp::IOP_WaveGetLaneCount, TranslateWaveToVal, DXIL::OpCode::WaveGetLaneCount},
    {IntrinsicOp::IOP_WaveGetLaneIndex, TranslateWaveToVal, DXIL::OpCode::WaveGetLaneIndex},
    {IntrinsicOp::IOP_WaveIsFirstLane, TranslateWaveToVal, DXIL::OpCode::WaveIsFirstLane},
    {IntrinsicOp::IOP_WaveMatch, TranslateWaveMatch, DXIL::OpCode::WaveMatch},
    {IntrinsicOp::IOP_WaveMultiPrefixBitAnd, TranslateWaveMultiPrefix, DXIL::OpCode::WaveMultiPrefixOp},
    {IntrinsicOp::IOP_WaveMultiPrefixBitOr, TranslateWaveMultiPrefix, DXIL::OpCode::WaveMultiPrefixOp},
    {IntrinsicOp::IOP_WaveMultiPrefixBitXor, TranslateWaveMultiPrefix, DXIL::OpCode::WaveMultiPrefixOp},
    {IntrinsicOp::IOP_WaveMultiPrefixCountBits, TranslateWaveMultiPrefixBitCount, DXIL::OpCode::WaveMultiPrefixBitCount},
    {IntrinsicOp::IOP_WaveMultiPrefixProduct, TranslateWaveMultiPrefix, DXIL::OpCode::WaveMultiPrefixOp},
    {IntrinsicOp::IOP_WaveMultiPrefixSum, TranslateWaveMultiPrefix, DXIL::OpCode::WaveMultiPrefixOp},
    {IntrinsicOp::IOP_WavePrefixCountBits, TranslateWaveA2B, DXIL::OpCode::WavePrefixBitCount},
    {IntrinsicOp::IOP_WavePrefixProduct, TranslateWaveA2A, DXIL::OpCode::WavePrefixOp},
    {IntrinsicOp::IOP_WavePrefixSum, TranslateWaveA2A, DXIL::OpCode::WavePrefixOp},
    {IntrinsicOp::IOP_WaveReadLaneAt, TranslateWaveReadLaneAt, DXIL::OpCode::WaveReadLaneAt},
    {IntrinsicOp::IOP_WaveReadLaneFirst, TranslateWaveReadLaneFirst, DXIL::OpCode::WaveReadLaneFirst},
    {IntrinsicOp::IOP_WorldRayDirection, TranslateNoArgVectorOperation, DXIL::OpCode::WorldRayDirection},
    {IntrinsicOp::IOP_WorldRayOrigin, TranslateNoArgVectorOperation, DXIL::OpCode::WorldRayOrigin},
    {IntrinsicOp::IOP_WorldToObject, TranslateNoArgMatrix3x4Operation, DXIL::OpCode::WorldToObject},
    {IntrinsicOp::IOP_WorldToObject3x4, TranslateNoArgMatrix3x4Operation, DXIL::OpCode::WorldToObject},
    {IntrinsicOp::IOP_WorldToObject4x3, TranslateNoArgTransposedMatrix3x4Operation, DXIL::OpCode::WorldToObject},
    {IntrinsicOp::IOP_abort, EmptyLower, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_abs, TranslateAbs, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_acos, TrivialUnaryOperation, DXIL::OpCode::Acos},
    {IntrinsicOp::IOP_all, TranslateAll, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_and, TranslateAnd, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_any, TranslateAny, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_asdouble, TranslateAsDouble, DXIL::OpCode::MakeDouble},
    {IntrinsicOp::IOP_asfloat, TranslateBitcast, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_asfloat16, TranslateBitcast, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_asin, TrivialUnaryOperation, DXIL::OpCode::Asin},
    {IntrinsicOp::IOP_asint, TranslateBitcast, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_asint16, TranslateBitcast, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_asuint, TranslateAsUint, DXIL::OpCode::SplitDouble},
    {IntrinsicOp::IOP_asuint16, TranslateAsUint, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_atan, TrivialUnaryOperation, DXIL::OpCode::Atan},
    {IntrinsicOp::IOP_atan2, TranslateAtan2, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_ceil, TrivialUnaryOperation, DXIL::OpCode::Round_pi},
    {IntrinsicOp::IOP_clamp, TranslateClamp, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_clip, TranslateClip, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_cos, TrivialUnaryOperation, DXIL::OpCode::Cos},
    {IntrinsicOp::IOP_cosh, TrivialUnaryOperation, DXIL::OpCode::Hcos},
    {IntrinsicOp::IOP_countbits, TrivialUnaryOperation, DXIL::OpCode::Countbits},
    {IntrinsicOp::IOP_cross, TranslateCross, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_ddx, TrivialUnaryOperation, DXIL::OpCode::DerivCoarseX},
    {IntrinsicOp::IOP_ddx_coarse, TrivialUnaryOperation, DXIL::OpCode::DerivCoarseX},
    {IntrinsicOp::IOP_ddx_fine, TrivialUnaryOperation, DXIL::OpCode::DerivFineX},
    {IntrinsicOp::IOP_ddy, TrivialUnaryOperation, DXIL::OpCode::DerivCoarseY},
    {IntrinsicOp::IOP_ddy_coarse, TrivialUnaryOperation, DXIL::OpCode::DerivCoarseY},
    {IntrinsicOp::IOP_ddy_fine, TrivialUnaryOperation, DXIL::OpCode::DerivFineY},
    {IntrinsicOp::IOP_degrees, TranslateDegrees, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_determinant, EmptyLower, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_distance, TranslateDistance, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_dot, TranslateDot, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_dot2add, TranslateDot2Add, DXIL::OpCode::Dot2AddHalf},
    {IntrinsicOp::IOP_dot4add_i8packed, TranslateDot4AddPacked, DXIL::OpCode::Dot4AddI8Packed},
    {IntrinsicOp::IOP_dot4add_u8packed, TranslateDot4AddPacked, DXIL::OpCode::Dot4AddU8Packed},
    {IntrinsicOp::IOP_dst, TranslateDst, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_exp, TranslateExp, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_exp2, TrivialUnaryOperation, DXIL::OpCode::Exp},
    {IntrinsicOp::IOP_f16tof32, TranslateF16ToF32, DXIL::OpCode::LegacyF16ToF32},
    {IntrinsicOp::IOP_f32tof16, TranslateF32ToF16, DXIL::OpCode::LegacyF32ToF16},
    {IntrinsicOp::IOP_faceforward, TranslateFaceforward, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_firstbithigh, TranslateFirstbitHi, DXIL::OpCode::FirstbitSHi},
    {IntrinsicOp::IOP_firstbitlow, TranslateFirstbitLo, DXIL::OpCode::FirstbitLo},
    {IntrinsicOp::IOP_floor, TrivialUnaryOperation, DXIL::OpCode::Round_ni},
    {IntrinsicOp::IOP_fma, TrivialTrinaryOperation, DXIL::OpCode::Fma},
    {IntrinsicOp::IOP_fmod, TranslateFMod, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_frac, TrivialUnaryOperation, DXIL::OpCode::Frc},
    {IntrinsicOp::IOP_frexp, TranslateFrexp, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_fwidth, TranslateFWidth, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_isfinite, TrivialIsSpecialFloat, DXIL::OpCode::IsFinite},
    {IntrinsicOp::IOP_isinf, TrivialIsSpecialFloat, DXIL::OpCode::IsInf},
    {IntrinsicOp::IOP_isnan, TrivialIsSpecialFloat, DXIL::OpCode::IsNaN},
    {IntrinsicOp::IOP_ldexp, TranslateLdExp, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_length, TranslateLength, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_lerp, TranslateLerp, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_lit, TranslateLit, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_log, TranslateLog, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_log10, TranslateLog10, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_log2, TrivialUnaryOperation, DXIL::OpCode::Log},
    {IntrinsicOp::IOP_mad, TranslateFUITrinary, DXIL::OpCode::IMad},
    {IntrinsicOp::IOP_max, TranslateFUIBinary, DXIL::OpCode::IMax},
    {IntrinsicOp::IOP_min, TranslateFUIBinary, DXIL::OpCode::IMin},
    {IntrinsicOp::IOP_modf, TranslateModF, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_msad4, TranslateMSad4, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_mul, TranslateMul, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_normalize, TranslateNormalize, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_or, TranslateOr, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_pack_clamp_s8, TranslatePack, DXIL::OpCode::Pack4x8 },
    {IntrinsicOp::IOP_pack_clamp_u8, TranslatePack, DXIL::OpCode::Pack4x8 },
    {IntrinsicOp::IOP_pack_s8, TranslatePack, DXIL::OpCode::Pack4x8 },
    {IntrinsicOp::IOP_pack_u8, TranslatePack, DXIL::OpCode::Pack4x8 },
    {IntrinsicOp::IOP_pow, TranslatePow, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_printf, TranslatePrintf, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_radians, TranslateRadians, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_rcp, TranslateRCP, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_reflect, TranslateReflect, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_refract, TranslateRefract, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_reversebits, TrivialUnaryOperation, DXIL::OpCode::Bfrev},
    {IntrinsicOp::IOP_round, TrivialUnaryOperation, DXIL::OpCode::Round_ne},
    {IntrinsicOp::IOP_rsqrt, TrivialUnaryOperation, DXIL::OpCode::Rsqrt},
    {IntrinsicOp::IOP_saturate, TrivialUnaryOperation, DXIL::OpCode::Saturate},
    {IntrinsicOp::IOP_select, TranslateSelect, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_sign, TranslateSign, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_sin, TrivialUnaryOperation, DXIL::OpCode::Sin},
    {IntrinsicOp::IOP_sincos, EmptyLower, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_sinh, TrivialUnaryOperation, DXIL::OpCode::Hsin},
    {IntrinsicOp::IOP_smoothstep, TranslateSmoothStep, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_source_mark, EmptyLower, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_sqrt, TrivialUnaryOperation, DXIL::OpCode::Sqrt},
    {IntrinsicOp::IOP_step, TranslateStep, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_tan, TrivialUnaryOperation, DXIL::OpCode::Tan},
    {IntrinsicOp::IOP_tanh, TrivialUnaryOperation, DXIL::OpCode::Htan},
    {IntrinsicOp::IOP_tex1D, EmptyLower, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_tex1Dbias, EmptyLower, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_tex1Dgrad, EmptyLower, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_tex1Dlod, EmptyLower, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_tex1Dproj, EmptyLower, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_tex2D, EmptyLower, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_tex2Dbias, EmptyLower, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_tex2Dgrad, EmptyLower, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_tex2Dlod, EmptyLower, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_tex2Dproj, EmptyLower, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_tex3D, EmptyLower, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_tex3Dbias, EmptyLower, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_tex3Dgrad, EmptyLower, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_tex3Dlod, EmptyLower, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_tex3Dproj, EmptyLower, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_texCUBE, EmptyLower, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_texCUBEbias, EmptyLower, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_texCUBEgrad, EmptyLower, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_texCUBElod, EmptyLower, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_texCUBEproj, EmptyLower, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_transpose, EmptyLower, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::IOP_trunc, TrivialUnaryOperation, DXIL::OpCode::Round_z},
    {IntrinsicOp::IOP_unpack_s8s16, TranslateUnpack, DXIL::OpCode::Unpack4x8},
    {IntrinsicOp::IOP_unpack_s8s32, TranslateUnpack, DXIL::OpCode::Unpack4x8},
    {IntrinsicOp::IOP_unpack_u8u16, TranslateUnpack, DXIL::OpCode::Unpack4x8},
    {IntrinsicOp::IOP_unpack_u8u32, TranslateUnpack, DXIL::OpCode::Unpack4x8},
#ifdef ENABLE_SPIRV_CODEGEN
    { IntrinsicOp::IOP_VkRawBufferLoad, UnsupportedVulkanIntrinsic, DXIL::OpCode::NumOpCodes },
    { IntrinsicOp::IOP_VkReadClock, UnsupportedVulkanIntrinsic, DXIL::OpCode::NumOpCodes },
    { IntrinsicOp::IOP_Vkext_execution_mode, UnsupportedVulkanIntrinsic, DXIL::OpCode::NumOpCodes },
    { IntrinsicOp::IOP_Vkext_execution_mode_id, UnsupportedVulkanIntrinsic, DXIL::OpCode::NumOpCodes },
#endif // ENABLE_SPIRV_CODEGEN
    {IntrinsicOp::MOP_Append, StreamOutputLower, DXIL::OpCode::EmitStream},
    {IntrinsicOp::MOP_RestartStrip, StreamOutputLower, DXIL::OpCode::CutStream},
    {IntrinsicOp::MOP_CalculateLevelOfDetail, TranslateCalculateLOD, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::MOP_CalculateLevelOfDetailUnclamped, TranslateCalculateLOD, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::MOP_GetDimensions, TranslateGetDimensions, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::MOP_Load, TranslateResourceLoad, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::MOP_Sample, TranslateSample, DXIL::OpCode::Sample},
    {IntrinsicOp::MOP_SampleBias, TranslateSample, DXIL::OpCode::SampleBias},
    {IntrinsicOp::MOP_SampleCmp, TranslateSample, DXIL::OpCode::SampleCmp},
    {IntrinsicOp::MOP_SampleCmpLevel, TranslateSample, DXIL::OpCode::SampleCmpLevel},
    {IntrinsicOp::MOP_SampleCmpLevelZero, TranslateSample, DXIL::OpCode::SampleCmpLevelZero},
    {IntrinsicOp::MOP_SampleGrad, TranslateSample, DXIL::OpCode::SampleGrad},
    {IntrinsicOp::MOP_SampleLevel, TranslateSample, DXIL::OpCode::SampleLevel},
    {IntrinsicOp::MOP_Gather, TranslateGather, DXIL::OpCode::TextureGather},
    {IntrinsicOp::MOP_GatherAlpha, TranslateGather, DXIL::OpCode::TextureGather},
    {IntrinsicOp::MOP_GatherBlue, TranslateGather, DXIL::OpCode::TextureGather},
    {IntrinsicOp::MOP_GatherCmp, TranslateGather, DXIL::OpCode::TextureGatherCmp},
    {IntrinsicOp::MOP_GatherCmpAlpha, TranslateGather, DXIL::OpCode::TextureGatherCmp},
    {IntrinsicOp::MOP_GatherCmpBlue, TranslateGather, DXIL::OpCode::TextureGatherCmp},
    {IntrinsicOp::MOP_GatherCmpGreen, TranslateGather, DXIL::OpCode::TextureGatherCmp},
    {IntrinsicOp::MOP_GatherCmpRed, TranslateGather, DXIL::OpCode::TextureGatherCmp},
    {IntrinsicOp::MOP_GatherGreen, TranslateGather, DXIL::OpCode::TextureGather},
    {IntrinsicOp::MOP_GatherRaw, TranslateGather, DXIL::OpCode::TextureGatherRaw},
    {IntrinsicOp::MOP_GatherRed, TranslateGather, DXIL::OpCode::TextureGather},
    {IntrinsicOp::MOP_GetSamplePosition, TranslateGetSamplePosition, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::MOP_Load2, TranslateResourceLoad, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::MOP_Load3, TranslateResourceLoad, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::MOP_Load4, TranslateResourceLoad, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::MOP_InterlockedAdd, TranslateMopAtomicBinaryOperation, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::MOP_InterlockedAdd64, TranslateMopAtomicBinaryOperation, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::MOP_InterlockedAnd, TranslateMopAtomicBinaryOperation, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::MOP_InterlockedAnd64, TranslateMopAtomicBinaryOperation, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::MOP_InterlockedCompareExchange, TranslateMopAtomicCmpXChg, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::MOP_InterlockedCompareExchange64, TranslateMopAtomicCmpXChg, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::MOP_InterlockedCompareExchangeFloatBitwise, TranslateMopAtomicCmpXChg, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::MOP_InterlockedCompareStore, TranslateMopAtomicCmpXChg, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::MOP_InterlockedCompareStore64, TranslateMopAtomicCmpXChg, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::MOP_InterlockedCompareStoreFloatBitwise, TranslateMopAtomicCmpXChg, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::MOP_InterlockedExchange, TranslateMopAtomicBinaryOperation, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::MOP_InterlockedExchange64, TranslateMopAtomicBinaryOperation, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::MOP_InterlockedExchangeFloat, TranslateMopAtomicBinaryOperation, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::MOP_InterlockedMax, TranslateMopAtomicBinaryOperation, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::MOP_InterlockedMax64, TranslateMopAtomicBinaryOperation, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::MOP_InterlockedMin, TranslateMopAtomicBinaryOperation, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::MOP_InterlockedMin64, TranslateMopAtomicBinaryOperation, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::MOP_InterlockedOr, TranslateMopAtomicBinaryOperation, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::MOP_InterlockedOr64, TranslateMopAtomicBinaryOperation, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::MOP_InterlockedXor, TranslateMopAtomicBinaryOperation, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::MOP_InterlockedXor64, TranslateMopAtomicBinaryOperation, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::MOP_Store, TranslateResourceStore, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::MOP_Store2, TranslateResourceStore, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::MOP_Store3, TranslateResourceStore, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::MOP_Store4, TranslateResourceStore, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::MOP_DecrementCounter, GenerateUpdateCounter, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::MOP_IncrementCounter, GenerateUpdateCounter, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::MOP_Consume, EmptyLower, DXIL::OpCode::NumOpCodes},
    {IntrinsicOp::MOP_WriteSamplerFeedback, TranslateWriteSamplerFeedback, DXIL::OpCode::WriteSamplerFeedback},
    {IntrinsicOp::MOP_WriteSamplerFeedbackBias, TranslateWriteSamplerFeedback, DXIL::OpCode::WriteSamplerFeedbackBias},
    {IntrinsicOp::MOP_WriteSamplerFeedbackGrad, TranslateWriteSamplerFeedback, DXIL::OpCode::WriteSamplerFeedbackGrad},
    {IntrinsicOp::MOP_WriteSamplerFeedbackLevel, TranslateWriteSamplerFeedback, DXIL::OpCode::WriteSamplerFeedbackLevel},

    {IntrinsicOp::MOP_Abort, TranslateGenericRayQueryMethod, DXIL::OpCode::RayQuery_Abort},
    {IntrinsicOp::MOP_CandidateGeometryIndex, TranslateGenericRayQueryMethod, DXIL::OpCode::RayQuery_CandidateGeometryIndex},
    {IntrinsicOp::MOP_CandidateInstanceContributionToHitGroupIndex, TranslateGenericRayQueryMethod, DXIL::OpCode::RayQuery_CandidateInstanceContributionToHitGroupIndex},
    {IntrinsicOp::MOP_CandidateInstanceID, TranslateGenericRayQueryMethod, DXIL::OpCode::RayQuery_CandidateInstanceID},
    {IntrinsicOp::MOP_CandidateInstanceIndex, TranslateGenericRayQueryMethod, DXIL::OpCode::RayQuery_CandidateInstanceIndex},
    {IntrinsicOp::MOP_CandidateObjectRayDirection, TranslateRayQueryFloat3Getter, DXIL::OpCode::RayQuery_CandidateObjectRayDirection},
    {IntrinsicOp::MOP_CandidateObjectRayOrigin, TranslateRayQueryFloat3Getter, DXIL::OpCode::RayQuery_CandidateObjectRayOrigin},
    {IntrinsicOp::MOP_CandidateObjectToWorld3x4, TranslateRayQueryMatrix3x4Operation, DXIL::OpCode::RayQuery_CandidateObjectToWorld3x4},
    {IntrinsicOp::MOP_CandidateObjectToWorld4x3, TranslateRayQueryTransposedMatrix3x4Operation, DXIL::OpCode::RayQuery_CandidateObjectToWorld3x4},
    {IntrinsicOp::MOP_CandidatePrimitiveIndex, TranslateGenericRayQueryMethod, DXIL::OpCode::RayQuery_CandidatePrimitiveIndex},
    {IntrinsicOp::MOP_CandidateProceduralPrimitiveNonOpaque, TranslateGenericRayQueryMethod, DXIL::OpCode::RayQuery_CandidateProceduralPrimitiveNonOpaque},
    {IntrinsicOp::MOP_CandidateTriangleBarycentrics, TranslateRayQueryFloat2Getter, DXIL::OpCode::RayQuery_CandidateTriangleBarycentrics},
    {IntrinsicOp::MOP_CandidateTriangleFrontFace, TranslateGenericRayQueryMethod, DXIL::OpCode::RayQuery_CandidateTriangleFrontFace},
    {IntrinsicOp::MOP_CandidateTriangleRayT, TranslateGenericRayQueryMethod, DXIL::OpCode::RayQuery_CandidateTriangleRayT},
    {IntrinsicOp::MOP_CandidateType, TranslateGenericRayQueryMethod, DXIL::OpCode::RayQuery_CandidateType},
    {IntrinsicOp::MOP_CandidateWorldToObject3x4, TranslateRayQueryMatrix3x4Operation, DXIL::OpCode::RayQuery_CandidateWorldToObject3x4},
    {IntrinsicOp::MOP_CandidateWorldToObject4x3, TranslateRayQueryTransposedMatrix3x4Operation, DXIL::OpCode::RayQuery_CandidateWorldToObject3x4},
    {IntrinsicOp::MOP_CommitNonOpaqueTriangleHit, TranslateGenericRayQueryMethod, DXIL::OpCode::RayQuery_CommitNonOpaqueTriangleHit},
    {IntrinsicOp::MOP_CommitProceduralPrimitiveHit, TranslateCommitProceduralPrimitiveHit, DXIL::OpCode::RayQuery_CommitProceduralPrimitiveHit},
    {IntrinsicOp::MOP_CommittedGeometryIndex, TranslateGenericRayQueryMethod, DXIL::OpCode::RayQuery_CommittedGeometryIndex},
    {IntrinsicOp::MOP_CommittedInstanceContributionToHitGroupIndex, TranslateGenericRayQueryMethod, DXIL::OpCode::RayQuery_CommittedInstanceContributionToHitGroupIndex},
    {IntrinsicOp::MOP_CommittedInstanceID, TranslateGenericRayQueryMethod, DXIL::OpCode::RayQuery_CommittedInstanceID},
    {IntrinsicOp::MOP_CommittedInstanceIndex, TranslateGenericRayQueryMethod, DXIL::OpCode::RayQuery_CommittedInstanceIndex},
    {IntrinsicOp::MOP_CommittedObjectRayDirection, TranslateRayQueryFloat3Getter, DXIL::OpCode::RayQuery_CommittedObjectRayDirection},
    {IntrinsicOp::MOP_CommittedObjectRayOrigin, TranslateRayQueryFloat3Getter, DXIL::OpCode::RayQuery_CommittedObjectRayOrigin},
    {IntrinsicOp::MOP_CommittedObjectToWorld3x4, TranslateRayQueryMatrix3x4Operation, DXIL::OpCode::RayQuery_CommittedObjectToWorld3x4},
    {IntrinsicOp::MOP_CommittedObjectToWorld4x3, TranslateRayQueryTransposedMatrix3x4Operation, DXIL::OpCode::RayQuery_CommittedObjectToWorld3x4},
    {IntrinsicOp::MOP_CommittedPrimitiveIndex, TranslateGenericRayQueryMethod, DXIL::OpCode::RayQuery_CommittedPrimitiveIndex},
    {IntrinsicOp::MOP_CommittedRayT, TranslateGenericRayQueryMethod, DXIL::OpCode::RayQuery_CommittedRayT},
    {IntrinsicOp::MOP_CommittedStatus, TranslateGenericRayQueryMethod, DXIL::OpCode::RayQuery_CommittedStatus},
    {IntrinsicOp::MOP_CommittedTriangleBarycentrics, TranslateRayQueryFloat2Getter, DXIL::OpCode::RayQuery_CommittedTriangleBarycentrics},
    {IntrinsicOp::MOP_CommittedTriangleFrontFace, TranslateGenericRayQueryMethod, DXIL::OpCode::RayQuery_CommittedTriangleFrontFace},
    {IntrinsicOp::MOP_CommittedWorldToObject3x4, TranslateRayQueryMatrix3x4Operation, DXIL::OpCode::RayQuery_CommittedWorldToObject3x4},
    {IntrinsicOp::MOP_CommittedWorldToObject4x3, TranslateRayQueryTransposedMatrix3x4Operation, DXIL::OpCode::RayQuery_CommittedWorldToObject3x4},
    {IntrinsicOp::MOP_Proceed, TranslateGenericRayQueryMethod, DXIL::OpCode::RayQuery_Proceed},
    {IntrinsicOp::MOP_RayFlags, TranslateGenericRayQueryMethod, DXIL::OpCode::RayQuery_RayFlags},
    {IntrinsicOp::MOP_RayTMin, TranslateGenericRayQueryMethod, DXIL::OpCode::RayQuery_RayTMin},
    {IntrinsicOp::MOP_TraceRayInline,  TranslateTraceRayInline,  DXIL::OpCode::RayQuery_TraceRayInline},
    {IntrinsicOp::MOP_WorldRayDirection, TranslateRayQueryFloat3Getter, DXIL::OpCode::RayQuery_WorldRayDirection},
    {IntrinsicOp::MOP_WorldRayOrigin, TranslateRayQueryFloat3Getter, DXIL::OpCode::RayQuery_WorldRayOrigin},

    // SPIRV change starts
#ifdef ENABLE_SPIRV_CODEGEN
    {IntrinsicOp::MOP_SubpassLoad, UnsupportedVulkanIntrinsic, DXIL::OpCode::NumOpCodes},
#endif // ENABLE_SPIRV_CODEGEN
    // SPIRV change ends

    // Manully added part.
    { IntrinsicOp::IOP_InterlockedUMax, TranslateIopAtomicBinaryOperation, DXIL::OpCode::NumOpCodes },
    { IntrinsicOp::IOP_InterlockedUMin, TranslateIopAtomicBinaryOperation, DXIL::OpCode::NumOpCodes },
    { IntrinsicOp::IOP_WaveActiveUMax, TranslateWaveA2A, DXIL::OpCode::WaveActiveOp },
    { IntrinsicOp::IOP_WaveActiveUMin, TranslateWaveA2A, DXIL::OpCode::WaveActiveOp },
    { IntrinsicOp::IOP_WaveActiveUProduct, TranslateWaveA2A, DXIL::OpCode::WaveActiveOp },
    { IntrinsicOp::IOP_WaveActiveUSum, TranslateWaveA2A, DXIL::OpCode::WaveActiveOp },
    { IntrinsicOp::IOP_WaveMultiPrefixUProduct, TranslateWaveMultiPrefix, DXIL::OpCode::WaveMultiPrefixOp },
    { IntrinsicOp::IOP_WaveMultiPrefixUSum, TranslateWaveMultiPrefix, DXIL::OpCode::WaveMultiPrefixOp },
    { IntrinsicOp::IOP_WavePrefixUProduct, TranslateWaveA2A, DXIL::OpCode::WavePrefixOp },
    { IntrinsicOp::IOP_WavePrefixUSum, TranslateWaveA2A, DXIL::OpCode::WavePrefixOp },
    { IntrinsicOp::IOP_uabs, TranslateUAbs, DXIL::OpCode::NumOpCodes },
    { IntrinsicOp::IOP_uclamp, TranslateClamp, DXIL::OpCode::NumOpCodes },
    { IntrinsicOp::IOP_ufirstbithigh, TranslateFirstbitHi, DXIL::OpCode::FirstbitHi },
    { IntrinsicOp::IOP_umad, TranslateFUITrinary, DXIL::OpCode::UMad},
    { IntrinsicOp::IOP_umax, TranslateFUIBinary, DXIL::OpCode::UMax},
    { IntrinsicOp::IOP_umin, TranslateFUIBinary, DXIL::OpCode::UMin },
    { IntrinsicOp::IOP_umul, TranslateMul, DXIL::OpCode::UMul },
    { IntrinsicOp::IOP_usign, TranslateUSign, DXIL::OpCode::UMax },
    { IntrinsicOp::MOP_InterlockedUMax, TranslateMopAtomicBinaryOperation, DXIL::OpCode::NumOpCodes },
    { IntrinsicOp::MOP_InterlockedUMin, TranslateMopAtomicBinaryOperation, DXIL::OpCode::NumOpCodes },
};
}
static_assert(sizeof(gLowerTable) / sizeof(gLowerTable[0]) == static_cast<size_t>(IntrinsicOp::Num_Intrinsics),
  "Intrinsic lowering table must be updated to account for new intrinsics.");

static void TranslateBuiltinIntrinsic(CallInst *CI,
                                      HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  unsigned opcode = hlsl::GetHLOpcode(CI);
  const IntrinsicLower &lower = gLowerTable[opcode];
  Value *Result =
      lower.LowerFunc(CI, lower.IntriOpcode, lower.DxilOpcode, helper, pObjHelper, Translated);
  if (Result)
    CI->replaceAllUsesWith(Result);
}

// SharedMem.
namespace {

bool IsSharedMemPtr(Value *Ptr) {
  return Ptr->getType()->getPointerAddressSpace() == DXIL::kTGSMAddrSpace;
}

bool IsLocalVariablePtr(Value *Ptr) {
  while (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Ptr)) {
    Ptr = GEP->getPointerOperand();
  }
  bool isAlloca = isa<AllocaInst>(Ptr);
  if (isAlloca) return true;

  GlobalVariable *GV = dyn_cast<GlobalVariable>(Ptr);
  if (!GV) return false;

  return GV->getLinkage() == GlobalValue::LinkageTypes::InternalLinkage;
}

}

// Constant buffer.
namespace {
unsigned GetEltTypeByteSizeForConstBuf(Type *EltType, const DataLayout &DL) {
  DXASSERT(EltType->isIntegerTy() || EltType->isFloatingPointTy(),
           "not an element type");
  // TODO: Use real size after change constant buffer into linear layout.
  if (DL.getTypeSizeInBits(EltType) <= 32) {
    // Constant buffer is 4 bytes align.
    return 4;
  } else
    return 8;
}

Value *GenerateCBLoad(Value *handle, Value *offset, Type *EltTy, OP *hlslOP,
                      IRBuilder<> &Builder) {
  Constant *OpArg = hlslOP->GetU32Const((unsigned)OP::OpCode::CBufferLoad);

  DXASSERT(!EltTy->isIntegerTy(1), "Bools should not be loaded as their register representation.");

  // Align to 8 bytes for now.
  Constant *align = hlslOP->GetU32Const(8);
  Function *CBLoad = hlslOP->GetOpFunc(OP::OpCode::CBufferLoad, EltTy);
  return Builder.CreateCall(CBLoad, {OpArg, handle, offset, align});
}

Value *TranslateConstBufMatLd(Type *matType, Value *handle, Value *offset,
                              bool colMajor, OP *OP, const DataLayout &DL,
                              IRBuilder<> &Builder) {
  HLMatrixType MatTy = HLMatrixType::cast(matType);
  Type *EltTy = MatTy.getElementTypeForMem();
  unsigned matSize = MatTy.getNumElements();
  std::vector<Value *> elts(matSize);
  Value *EltByteSize = ConstantInt::get(
      offset->getType(), GetEltTypeByteSizeForConstBuf(EltTy, DL));

  // TODO: use real size after change constant buffer into linear layout.
  Value *baseOffset = offset;
  for (unsigned i = 0; i < matSize; i++) {
    elts[i] = GenerateCBLoad(handle, baseOffset, EltTy, OP, Builder);
    baseOffset = Builder.CreateAdd(baseOffset, EltByteSize);
  }

  Value* Vec = HLMatrixLower::BuildVector(EltTy, elts, Builder);
  Vec = MatTy.emitLoweredMemToReg(Vec, Builder);
  return Vec;
}

void TranslateCBGep(GetElementPtrInst *GEP, Value *handle, Value *baseOffset,
                    hlsl::OP *hlslOP, IRBuilder<> &Builder,
                    DxilFieldAnnotation *prevFieldAnnotation,
                    const DataLayout &DL, DxilTypeSystem &dxilTypeSys,
                    HLObjectOperationLowerHelper *pObjHelper);

Value *GenerateVecEltFromGEP(Value *ldData, GetElementPtrInst *GEP,
                             IRBuilder<> &Builder, bool bInsertLdNextToGEP) {
  DXASSERT(GEP->getNumIndices() == 2, "must have 2 level");
  Value *baseIdx = (GEP->idx_begin())->get();
  Value *zeroIdx = Builder.getInt32(0);
  DXASSERT_LOCALVAR(baseIdx && zeroIdx, baseIdx == zeroIdx,
                    "base index must be 0");
  Value *idx = (GEP->idx_begin() + 1)->get();
  if (dyn_cast<ConstantInt>(idx)) {
    return Builder.CreateExtractElement(ldData, idx);
  } else {
    // Dynamic indexing.
    // Copy vec to array.
    Type *Ty = ldData->getType();
    Type *EltTy = Ty->getVectorElementType();
    unsigned vecSize = Ty->getVectorNumElements();
    ArrayType *AT = ArrayType::get(EltTy, vecSize);
    IRBuilder<> AllocaBuilder(
        GEP->getParent()->getParent()->getEntryBlock().getFirstInsertionPt());
    Value *tempArray = AllocaBuilder.CreateAlloca(AT);
    Value *zero = Builder.getInt32(0);
    for (unsigned int i = 0; i < vecSize; i++) {
      Value *Elt = Builder.CreateExtractElement(ldData, Builder.getInt32(i));
      Value *Ptr =
          Builder.CreateInBoundsGEP(tempArray, {zero, Builder.getInt32(i)});
      Builder.CreateStore(Elt, Ptr);
    }
    // Load from temp array.
    if (bInsertLdNextToGEP) {
      // Insert the new GEP just before the old and to-be-deleted GEP
      Builder.SetInsertPoint(GEP);
    }
    Value *EltGEP = Builder.CreateInBoundsGEP(tempArray, {zero, idx});
    return Builder.CreateLoad(EltGEP);
  }
}

void TranslateResourceInCB(LoadInst *LI,
                           HLObjectOperationLowerHelper *pObjHelper,
                           GlobalVariable *CbGV) {
  if (LI->user_empty()) {
    LI->eraseFromParent();
    return;
  }

  GetElementPtrInst *Ptr = cast<GetElementPtrInst>(LI->getPointerOperand());
  CallInst *CI = cast<CallInst>(LI->user_back());
  CallInst *Anno = cast<CallInst>(CI->user_back());
  DxilResourceProperties RP = pObjHelper->GetResPropsFromAnnotateHandle(Anno);
  Value *ResPtr = pObjHelper->GetOrCreateResourceForCbPtr(Ptr, CbGV, RP);

  // Lower Ptr to GV base Ptr.
  Value *GvPtr = pObjHelper->LowerCbResourcePtr(Ptr, ResPtr);
  IRBuilder<> Builder(LI);
  Value *GvLd = Builder.CreateLoad(GvPtr);
  LI->replaceAllUsesWith(GvLd);
  LI->eraseFromParent();
}

void TranslateCBAddressUser(Instruction *user, Value *handle, Value *baseOffset,
                            hlsl::OP *hlslOP,
                            DxilFieldAnnotation *prevFieldAnnotation,
                            DxilTypeSystem &dxilTypeSys, const DataLayout &DL,
                            HLObjectOperationLowerHelper *pObjHelper) {
  IRBuilder<> Builder(user);
  if (CallInst *CI = dyn_cast<CallInst>(user)) {
    HLOpcodeGroup group = GetHLOpcodeGroupByName(CI->getCalledFunction());
    unsigned opcode = GetHLOpcode(CI);
    if (group == HLOpcodeGroup::HLMatLoadStore) {
      HLMatLoadStoreOpcode matOp = static_cast<HLMatLoadStoreOpcode>(opcode);
      bool colMajor = matOp == HLMatLoadStoreOpcode::ColMatLoad;
      DXASSERT(matOp == HLMatLoadStoreOpcode::ColMatLoad ||
                   matOp == HLMatLoadStoreOpcode::RowMatLoad,
               "No store on cbuffer");
      Type *matType = CI->getArgOperand(HLOperandIndex::kMatLoadPtrOpIdx)
                          ->getType()
                          ->getPointerElementType();
      Value *newLd = TranslateConstBufMatLd(matType, handle, baseOffset,
                                            colMajor, hlslOP, DL, Builder);
      CI->replaceAllUsesWith(newLd);
      CI->eraseFromParent();
    } else if (group == HLOpcodeGroup::HLSubscript) {
      HLSubscriptOpcode subOp = static_cast<HLSubscriptOpcode>(opcode);
      Value *basePtr = CI->getArgOperand(HLOperandIndex::kMatSubscriptMatOpIdx);
      HLMatrixType MatTy = HLMatrixType::cast(basePtr->getType()->getPointerElementType());
      Type *EltTy = MatTy.getElementTypeForReg();

      Value *EltByteSize = ConstantInt::get(
          baseOffset->getType(), GetEltTypeByteSizeForConstBuf(EltTy, DL));

      Value *idx = CI->getArgOperand(HLOperandIndex::kMatSubscriptSubOpIdx);

      Type *resultType = CI->getType()->getPointerElementType();
      unsigned resultSize = 1;
      if (resultType->isVectorTy())
        resultSize = resultType->getVectorNumElements();
      DXASSERT(resultSize <= 16, "up to 4x4 elements in vector or matrix");
      _Analysis_assume_(resultSize <= 16);
      Value *idxList[16];

      switch (subOp) {
      case HLSubscriptOpcode::ColMatSubscript:
      case HLSubscriptOpcode::RowMatSubscript: {
        for (unsigned i = 0; i < resultSize; i++) {
          Value *idx =
              CI->getArgOperand(HLOperandIndex::kMatSubscriptSubOpIdx + i);
          Value *offset = Builder.CreateMul(idx, EltByteSize);
          idxList[i] = Builder.CreateAdd(baseOffset, offset);
        }

      } break;
      case HLSubscriptOpcode::RowMatElement:
      case HLSubscriptOpcode::ColMatElement: {
        Constant *EltIdxs = cast<Constant>(idx);
        for (unsigned i = 0; i < resultSize; i++) {
          Value *offset =
              Builder.CreateMul(EltIdxs->getAggregateElement(i), EltByteSize);
          idxList[i] = Builder.CreateAdd(baseOffset, offset);
        }
      } break;
      default:
        DXASSERT(0, "invalid operation on const buffer");
        break;
      }

      Value *ldData = UndefValue::get(resultType);
      if (resultType->isVectorTy()) {
        for (unsigned i = 0; i < resultSize; i++) {
          Value *eltData =
              GenerateCBLoad(handle, idxList[i], EltTy, hlslOP, Builder);
          ldData = Builder.CreateInsertElement(ldData, eltData, i);
        }
      } else {
        ldData = GenerateCBLoad(handle, idxList[0], EltTy, hlslOP, Builder);
      }

      for (auto U = CI->user_begin(); U != CI->user_end();) {
        Value *subsUser = *(U++);
        if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(subsUser)) {
          Value *subData = GenerateVecEltFromGEP(ldData, GEP, Builder,
                                                 /*bInsertLdNextToGEP*/ true);

          for (auto gepU = GEP->user_begin(); gepU != GEP->user_end();) {
            Value *gepUser = *(gepU++);
            // Must be load here;
            LoadInst *ldUser = cast<LoadInst>(gepUser);
            ldUser->replaceAllUsesWith(subData);
            ldUser->eraseFromParent();
          }
          GEP->eraseFromParent();
        } else {
          // Must be load here.
          LoadInst *ldUser = cast<LoadInst>(subsUser);
          ldUser->replaceAllUsesWith(ldData);
          ldUser->eraseFromParent();
        }
      }

      CI->eraseFromParent();
    } else {
      DXASSERT(0, "not implemented yet");
    }
  } else if (LoadInst *ldInst = dyn_cast<LoadInst>(user)) {
    Type *Ty = ldInst->getType();
    Type *EltTy = Ty->getScalarType();
    // Resource inside cbuffer is lowered after GenerateDxilOperations.
    if (dxilutil::IsHLSLObjectType(Ty)) {
      CallInst *CI = cast<CallInst>(handle);
      // CI should be annotate handle.
      // Need createHandle here.
      if (GetHLOpcodeGroup(CI->getCalledFunction()) == HLOpcodeGroup::HLAnnotateHandle)
        CI = cast<CallInst>(CI->getArgOperand(HLOperandIndex::kAnnotateHandleHandleOpIdx));
      GlobalVariable *CbGV = cast<GlobalVariable>(
          CI->getArgOperand(HLOperandIndex::kCreateHandleResourceOpIdx));
      TranslateResourceInCB(ldInst, pObjHelper, CbGV);
      return;
    }
    DXASSERT(!Ty->isAggregateType(), "should be flat in previous pass");

    unsigned EltByteSize = GetEltTypeByteSizeForConstBuf(EltTy, DL);

    Value *newLd = GenerateCBLoad(handle, baseOffset, EltTy, hlslOP, Builder);
    if (Ty->isVectorTy()) {
      Value *result = UndefValue::get(Ty);
      result = Builder.CreateInsertElement(result, newLd, (uint64_t)0);
      // Update offset by 4 bytes.
      Value *offset =
          Builder.CreateAdd(baseOffset, hlslOP->GetU32Const(EltByteSize));
      for (unsigned i = 1; i < Ty->getVectorNumElements(); i++) {
        Value *elt = GenerateCBLoad(handle, offset, EltTy, hlslOP, Builder);
        result = Builder.CreateInsertElement(result, elt, i);
        // Update offset by 4 bytes.
        offset = Builder.CreateAdd(offset, hlslOP->GetU32Const(EltByteSize));
      }
      newLd = result;
    }

    ldInst->replaceAllUsesWith(newLd);
    ldInst->eraseFromParent();
  } else {
    // Must be GEP here
    GetElementPtrInst *GEP = cast<GetElementPtrInst>(user);
    TranslateCBGep(GEP, handle, baseOffset, hlslOP, Builder,
                   prevFieldAnnotation, DL, dxilTypeSys, pObjHelper);
    GEP->eraseFromParent();
  }
}

void TranslateCBGep(GetElementPtrInst *GEP, Value *handle, Value *baseOffset,
                    hlsl::OP *hlslOP, IRBuilder<> &Builder,
                    DxilFieldAnnotation *prevFieldAnnotation,
                    const DataLayout &DL, DxilTypeSystem &dxilTypeSys,
                    HLObjectOperationLowerHelper *pObjHelper) {
  SmallVector<Value *, 8> Indices(GEP->idx_begin(), GEP->idx_end());

  Value *offset = baseOffset;
  // update offset
  DxilFieldAnnotation *fieldAnnotation = prevFieldAnnotation;

  gep_type_iterator GEPIt = gep_type_begin(GEP), E = gep_type_end(GEP);

  for (; GEPIt != E; GEPIt++) {
    Value *idx = GEPIt.getOperand();
    unsigned immIdx = 0;
    bool bImmIdx = false;
    if (Constant *constIdx = dyn_cast<Constant>(idx)) {
      immIdx = constIdx->getUniqueInteger().getLimitedValue();
      bImmIdx = true;
    }

    if (GEPIt->isPointerTy()) {
      Type *EltTy = GEPIt->getPointerElementType();
      unsigned size = 0;
      if (StructType *ST = dyn_cast<StructType>(EltTy)) {
        DxilStructAnnotation *annotation = dxilTypeSys.GetStructAnnotation(ST);
        size = annotation->GetCBufferSize();
      } else {
        DXASSERT(fieldAnnotation, "must be a field");
        if (ArrayType *AT = dyn_cast<ArrayType>(EltTy)) {
          unsigned EltSize = dxilutil::GetLegacyCBufferFieldElementSize(
              *fieldAnnotation, EltTy, dxilTypeSys);

          // Decide the nested array size.
          unsigned nestedArraySize = 1;

          Type *EltTy = AT->getArrayElementType();
          // support multi level of array
          while (EltTy->isArrayTy()) {
            ArrayType *EltAT = cast<ArrayType>(EltTy);
            nestedArraySize *= EltAT->getNumElements();
            EltTy = EltAT->getElementType();
          }
          // Align to 4 * 4 bytes.
          unsigned alignedSize = (EltSize + 15) & 0xfffffff0;
          size = nestedArraySize * alignedSize;
        } else {
          size = DL.getTypeAllocSize(EltTy);
        }
      }
      // Align to 4 * 4 bytes.
      size = (size + 15) & 0xfffffff0;
      if (bImmIdx) {
        unsigned tempOffset = size * immIdx;
        offset = Builder.CreateAdd(offset, hlslOP->GetU32Const(tempOffset));
      } else {
        Value *tempOffset = Builder.CreateMul(idx, hlslOP->GetU32Const(size));
        offset = Builder.CreateAdd(offset, tempOffset);
      }
    } else if (GEPIt->isStructTy()) {
      StructType *ST = cast<StructType>(*GEPIt);
      DxilStructAnnotation *annotation = dxilTypeSys.GetStructAnnotation(ST);
      fieldAnnotation = &annotation->GetFieldAnnotation(immIdx);
      unsigned structOffset = fieldAnnotation->GetCBufferOffset();
      offset = Builder.CreateAdd(offset, hlslOP->GetU32Const(structOffset));
    } else if (GEPIt->isArrayTy()) {
      DXASSERT(fieldAnnotation != nullptr, "must a field");
      unsigned EltSize = dxilutil::GetLegacyCBufferFieldElementSize(
              *fieldAnnotation, *GEPIt, dxilTypeSys);
      // Decide the nested array size.
      unsigned nestedArraySize = 1;

      Type *EltTy = GEPIt->getArrayElementType();
      // support multi level of array
      while (EltTy->isArrayTy()) {
        ArrayType *EltAT = cast<ArrayType>(EltTy);
        nestedArraySize *= EltAT->getNumElements();
        EltTy = EltAT->getElementType();
      }
      // Align to 4 * 4 bytes.
      unsigned alignedSize = (EltSize + 15) & 0xfffffff0;
      unsigned size = nestedArraySize * alignedSize;
      if (bImmIdx) {
        unsigned tempOffset = size * immIdx;
        offset = Builder.CreateAdd(offset, hlslOP->GetU32Const(tempOffset));
      } else {
        Value *tempOffset = Builder.CreateMul(idx, hlslOP->GetU32Const(size));
        offset = Builder.CreateAdd(offset, tempOffset);
      }
    } else if (GEPIt->isVectorTy()) {
      unsigned size = DL.getTypeAllocSize(GEPIt->getVectorElementType());
      if (bImmIdx) {
        unsigned tempOffset = size * immIdx;
        offset = Builder.CreateAdd(offset, hlslOP->GetU32Const(tempOffset));
      } else {
        Value *tempOffset = Builder.CreateMul(idx, hlslOP->GetU32Const(size));
        offset = Builder.CreateAdd(offset, tempOffset);
      }
    } else {
      gep_type_iterator temp = GEPIt;
      temp++;
      DXASSERT(temp == E, "scalar type must be the last");
    }
  }

  for (auto U = GEP->user_begin(); U != GEP->user_end();) {
    Instruction *user = cast<Instruction>(*(U++));

    TranslateCBAddressUser(user, handle, offset, hlslOP, fieldAnnotation,
                           dxilTypeSys, DL, pObjHelper);
  }
}

void TranslateCBOperations(Value *handle, Value *ptr, Value *offset, OP *hlslOP,
                           DxilTypeSystem &dxilTypeSys, const DataLayout &DL,
                           HLObjectOperationLowerHelper *pObjHelper) {
  auto User = ptr->user_begin();
  auto UserE = ptr->user_end();
  for (; User != UserE;) {
    // Must be Instruction.
    Instruction *I = cast<Instruction>(*(User++));
    TranslateCBAddressUser(I, handle, offset, hlslOP,
                           /*prevFieldAnnotation*/ nullptr, dxilTypeSys, DL,
                           pObjHelper);
  }
}

Value *GenerateCBLoadLegacy(Value *handle, Value *legacyIdx,
                            unsigned channelOffset, Type *EltTy, OP *hlslOP,
                            IRBuilder<> &Builder) {
  Constant *OpArg = hlslOP->GetU32Const((unsigned)OP::OpCode::CBufferLoadLegacy);

  DXASSERT(!EltTy->isIntegerTy(1), "Bools should not be loaded as their register representation.");

  Type *doubleTy = Type::getDoubleTy(EltTy->getContext());
  Type *halfTy = Type::getHalfTy(EltTy->getContext());
  Type *i64Ty = Type::getInt64Ty(EltTy->getContext());
  Type *i16Ty = Type::getInt16Ty(EltTy->getContext());

  bool is64 = (EltTy == doubleTy) | (EltTy == i64Ty);
  bool is16 = (EltTy == halfTy || EltTy == i16Ty) && !hlslOP->UseMinPrecision();
  DXASSERT_LOCALVAR(is16, (is16 && channelOffset < 8) || channelOffset < 4,
           "legacy cbuffer don't across 16 bytes register.");
  if (is64) {
    Function *CBLoad = hlslOP->GetOpFunc(OP::OpCode::CBufferLoadLegacy, EltTy);
    Value *loadLegacy = Builder.CreateCall(CBLoad, {OpArg, handle, legacyIdx});
    DXASSERT((channelOffset&1)==0,"channel offset must be even for double");
    unsigned eltIdx = channelOffset>>1;
    Value *Result = Builder.CreateExtractValue(loadLegacy, eltIdx);
    return Result;
  } else {
    Function *CBLoad = hlslOP->GetOpFunc(OP::OpCode::CBufferLoadLegacy, EltTy);
    Value *loadLegacy = Builder.CreateCall(CBLoad, { OpArg, handle, legacyIdx });
    return Builder.CreateExtractValue(loadLegacy, channelOffset);
  }
}

Value *GenerateCBLoadLegacy(Value *handle, Value *legacyIdx,
                            unsigned channelOffset, Type *EltTy,
                            unsigned vecSize, OP *hlslOP,
                            IRBuilder<> &Builder) {
  Constant *OpArg = hlslOP->GetU32Const((unsigned)OP::OpCode::CBufferLoadLegacy);

  DXASSERT(!EltTy->isIntegerTy(1), "Bools should not be loaded as their register representation.");

  Type *doubleTy = Type::getDoubleTy(EltTy->getContext());
  Type *i64Ty = Type::getInt64Ty(EltTy->getContext());
  Type *halfTy = Type::getHalfTy(EltTy->getContext());
  Type *shortTy = Type::getInt16Ty(EltTy->getContext());

  bool is64 = (EltTy == doubleTy) | (EltTy == i64Ty);
  bool is16 = (EltTy == shortTy || EltTy == halfTy) && !hlslOP->UseMinPrecision();
  DXASSERT((is16 && channelOffset + vecSize <= 8) ||
               (channelOffset + vecSize) <= 4,
           "legacy cbuffer don't across 16 bytes register.");
  if (is16) {
    Function *CBLoad = hlslOP->GetOpFunc(OP::OpCode::CBufferLoadLegacy, EltTy);
    Value *loadLegacy = Builder.CreateCall(CBLoad, {OpArg, handle, legacyIdx});
    Value *Result = UndefValue::get(VectorType::get(EltTy, vecSize));
    for (unsigned i = 0; i < vecSize; ++i) {
      Value *NewElt = Builder.CreateExtractValue(loadLegacy, channelOffset + i);
      Result = Builder.CreateInsertElement(Result, NewElt, i);
    }
    return Result;
  } else if (is64) {
    Function *CBLoad = hlslOP->GetOpFunc(OP::OpCode::CBufferLoadLegacy, EltTy);
    Value *loadLegacy = Builder.CreateCall(CBLoad, { OpArg, handle, legacyIdx });
    Value *Result = UndefValue::get(VectorType::get(EltTy, vecSize));
    unsigned smallVecSize = 2;
    if (vecSize < smallVecSize)
      smallVecSize = vecSize;
    for (unsigned i = 0; i < smallVecSize; ++i) {
      Value *NewElt = Builder.CreateExtractValue(loadLegacy, channelOffset+i);
      Result = Builder.CreateInsertElement(Result, NewElt, i);
    }
    if (vecSize > 2) {
      // Got to next cb register.
      legacyIdx = Builder.CreateAdd(legacyIdx, hlslOP->GetU32Const(1));
      Value *loadLegacy = Builder.CreateCall(CBLoad, {OpArg, handle, legacyIdx});
      for (unsigned i = 2; i < vecSize; ++i) {
        Value *NewElt =
            Builder.CreateExtractValue(loadLegacy, i-2);
        Result = Builder.CreateInsertElement(Result, NewElt, i);
      }
    }
    return Result;
  } else {
    Function *CBLoad = hlslOP->GetOpFunc(OP::OpCode::CBufferLoadLegacy, EltTy);
    Value *loadLegacy = Builder.CreateCall(CBLoad, { OpArg, handle, legacyIdx });
    Value *Result = UndefValue::get(VectorType::get(EltTy, vecSize));
    for (unsigned i = 0; i < vecSize; ++i) {
      Value *NewElt = Builder.CreateExtractValue(loadLegacy, channelOffset + i);
      Result = Builder.CreateInsertElement(Result, NewElt, i);
    }
    return Result;
  }
}

Value *TranslateConstBufMatLdLegacy(HLMatrixType MatTy, Value *handle,
                                    Value *legacyIdx, bool colMajor, OP *OP,
                                    bool memElemRepr, const DataLayout &DL,
                                    IRBuilder<> &Builder) {
  Type *EltTy = MatTy.getElementTypeForMem();

  unsigned matSize = MatTy.getNumElements();
  std::vector<Value *> elts(matSize);
  unsigned EltByteSize = GetEltTypeByteSizeForConstBuf(EltTy, DL);
  if (colMajor) {
    unsigned colByteSize = 4 * EltByteSize;
    unsigned colRegSize = (colByteSize + 15) >> 4;
    for (unsigned c = 0; c < MatTy.getNumColumns(); c++) {
      Value *col = GenerateCBLoadLegacy(handle, legacyIdx, /*channelOffset*/ 0,
                                        EltTy, MatTy.getNumRows(), OP, Builder);

      for (unsigned r = 0; r < MatTy.getNumRows(); r++) {
        unsigned matIdx = MatTy.getColumnMajorIndex(r, c);
        elts[matIdx] = Builder.CreateExtractElement(col, r);
      }
      // Update offset for a column.
      legacyIdx = Builder.CreateAdd(legacyIdx, OP->GetU32Const(colRegSize));
    }
  } else {
    unsigned rowByteSize = 4 * EltByteSize;
    unsigned rowRegSize = (rowByteSize + 15) >> 4;
    for (unsigned r = 0; r < MatTy.getNumRows(); r++) {
      Value *row = GenerateCBLoadLegacy(handle, legacyIdx, /*channelOffset*/ 0,
                                        EltTy, MatTy.getNumColumns(), OP, Builder);
      for (unsigned c = 0; c < MatTy.getNumColumns(); c++) {
        unsigned matIdx = MatTy.getRowMajorIndex(r, c);
        elts[matIdx] = Builder.CreateExtractElement(row, c);
      }
      // Update offset for a row.
      legacyIdx = Builder.CreateAdd(legacyIdx, OP->GetU32Const(rowRegSize));
    }
  }

  Value *Vec = HLMatrixLower::BuildVector(EltTy, elts, Builder);
  if (!memElemRepr)
    Vec = MatTy.emitLoweredMemToReg(Vec, Builder);
  return Vec;
}

void TranslateCBGepLegacy(GetElementPtrInst *GEP, Value *handle,
                          Value *legacyIdx, unsigned channelOffset,
                          hlsl::OP *hlslOP, IRBuilder<> &Builder,
                          DxilFieldAnnotation *prevFieldAnnotation,
                          const DataLayout &DL, DxilTypeSystem &dxilTypeSys,
                          HLObjectOperationLowerHelper *pObjHelper);

void TranslateCBAddressUserLegacy(Instruction *user, Value *handle,
                                  Value *legacyIdx, unsigned channelOffset,
                                  hlsl::OP *hlslOP,
                                  DxilFieldAnnotation *prevFieldAnnotation,
                                  DxilTypeSystem &dxilTypeSys,
                                  const DataLayout &DL,
                                  HLObjectOperationLowerHelper *pObjHelper) {
  IRBuilder<> Builder(user);
  if (CallInst *CI = dyn_cast<CallInst>(user)) {
    HLOpcodeGroup group = GetHLOpcodeGroupByName(CI->getCalledFunction());
    if (group == HLOpcodeGroup::HLMatLoadStore) {
      unsigned opcode = GetHLOpcode(CI);
      HLMatLoadStoreOpcode matOp = static_cast<HLMatLoadStoreOpcode>(opcode);
      bool colMajor = matOp == HLMatLoadStoreOpcode::ColMatLoad;
      DXASSERT(matOp == HLMatLoadStoreOpcode::ColMatLoad ||
                   matOp == HLMatLoadStoreOpcode::RowMatLoad,
               "No store on cbuffer");
      HLMatrixType MatTy = HLMatrixType::cast(
        CI->getArgOperand(HLOperandIndex::kMatLoadPtrOpIdx)
          ->getType()->getPointerElementType());
      // This will replace a call, so we should use the register representation of elements
      Value *newLd = TranslateConstBufMatLdLegacy(
        MatTy, handle, legacyIdx, colMajor, hlslOP, /*memElemRepr*/false, DL, Builder);
      CI->replaceAllUsesWith(newLd);
      dxilutil::TryScatterDebugValueToVectorElements(newLd);
      CI->eraseFromParent();
    } else if (group == HLOpcodeGroup::HLSubscript) {
      unsigned opcode = GetHLOpcode(CI);
      HLSubscriptOpcode subOp = static_cast<HLSubscriptOpcode>(opcode);
      Value *basePtr = CI->getArgOperand(HLOperandIndex::kMatSubscriptMatOpIdx);
      HLMatrixType MatTy = HLMatrixType::cast(basePtr->getType()->getPointerElementType());
      Type *EltTy = MatTy.getElementTypeForReg();

      Value *idx = CI->getArgOperand(HLOperandIndex::kMatSubscriptSubOpIdx);

      Type *resultType = CI->getType()->getPointerElementType();
      unsigned resultSize = 1;
      if (resultType->isVectorTy())
        resultSize = resultType->getVectorNumElements();
      DXASSERT(resultSize <= 16, "up to 4x4 elements in vector or matrix");
      _Analysis_assume_(resultSize <= 16);
      Value *idxList[16];
      bool colMajor = subOp == HLSubscriptOpcode::ColMatSubscript ||
                      subOp == HLSubscriptOpcode::ColMatElement;
      bool dynamicIndexing = !isa<ConstantInt>(idx) &&
                             !isa<ConstantAggregateZero>(idx) &&
                             !isa<ConstantDataSequential>(idx);

      Value *ldData = UndefValue::get(resultType);
      if (!dynamicIndexing) {
        // This will replace a load or GEP, so we should use the memory representation of elements
        Value *matLd = TranslateConstBufMatLdLegacy(
          MatTy, handle, legacyIdx, colMajor, hlslOP, /*memElemRepr*/true, DL, Builder);
        // The matLd is keep original layout, just use the idx calc in
        // EmitHLSLMatrixElement and EmitHLSLMatrixSubscript.
        switch (subOp) {
        case HLSubscriptOpcode::RowMatSubscript:
        case HLSubscriptOpcode::ColMatSubscript: {
          for (unsigned i = 0; i < resultSize; i++) {
            idxList[i] =
                CI->getArgOperand(HLOperandIndex::kMatSubscriptSubOpIdx + i);
          }
        } break;
        case HLSubscriptOpcode::RowMatElement:
        case HLSubscriptOpcode::ColMatElement: {
          Constant *EltIdxs = cast<Constant>(idx);
          for (unsigned i = 0; i < resultSize; i++) {
            idxList[i] = EltIdxs->getAggregateElement(i);
          }
        } break;
        default:
          DXASSERT(0, "invalid operation on const buffer");
          break;
        }

        if (resultType->isVectorTy()) {
          for (unsigned i = 0; i < resultSize; i++) {
            Value *eltData = Builder.CreateExtractElement(matLd, idxList[i]);
            ldData = Builder.CreateInsertElement(ldData, eltData, i);
          }
        } else {
          Value *eltData = Builder.CreateExtractElement(matLd, idxList[0]);
          ldData = eltData;
        }
      } else {
        // Must be matSub here.
        Value *idx = CI->getArgOperand(HLOperandIndex::kMatSubscriptSubOpIdx);

        if (colMajor) {
          // idx is c * row + r.
          // For first col, c is 0, so idx is r.
          Value *one = Builder.getInt32(1);
          // row.x = c[0].[idx]
          // row.y = c[1].[idx]
          // row.z = c[2].[idx]
          // row.w = c[3].[idx]
          Value *Elts[4];
          ArrayType *AT = ArrayType::get(EltTy, MatTy.getNumColumns());

          IRBuilder<> AllocaBuilder(user->getParent()
                                        ->getParent()
                                        ->getEntryBlock()
                                        .getFirstInsertionPt());

          Value *tempArray = AllocaBuilder.CreateAlloca(AT);
          Value *zero = AllocaBuilder.getInt32(0);
          Value *cbufIdx = legacyIdx;
          for (unsigned int c = 0; c < MatTy.getNumColumns(); c++) {
            Value *ColVal =
                GenerateCBLoadLegacy(handle, cbufIdx, /*channelOffset*/ 0,
                                     EltTy, MatTy.getNumRows(), hlslOP, Builder);
            // Convert ColVal to array for indexing.
            for (unsigned int r = 0; r < MatTy.getNumRows(); r++) {
              Value *Elt =
                  Builder.CreateExtractElement(ColVal, Builder.getInt32(r));
              Value *Ptr = Builder.CreateInBoundsGEP(
                  tempArray, {zero, Builder.getInt32(r)});
              Builder.CreateStore(Elt, Ptr);
            }

            Value *Ptr = Builder.CreateInBoundsGEP(tempArray, {zero, idx});
            Elts[c] = Builder.CreateLoad(Ptr);
            // Update cbufIdx.
            cbufIdx = Builder.CreateAdd(cbufIdx, one);
          }
          if (resultType->isVectorTy()) {
            for (unsigned int c = 0; c < MatTy.getNumColumns(); c++) {
              ldData = Builder.CreateInsertElement(ldData, Elts[c], c);
            }
          } else {
            ldData = Elts[0];
          }
        } else {
          // idx is r * col + c;
          // r = idx / col;
          Value *cCol = ConstantInt::get(idx->getType(), MatTy.getNumColumns());
          idx = Builder.CreateUDiv(idx, cCol);
          idx = Builder.CreateAdd(idx, legacyIdx);
          // Just return a row; 'col' is the number of columns in the row.
          ldData = GenerateCBLoadLegacy(handle, idx, /*channelOffset*/ 0, EltTy,
            MatTy.getNumColumns(), hlslOP, Builder);
        }
        if (!resultType->isVectorTy()) {
          ldData = Builder.CreateExtractElement(ldData, Builder.getInt32(0));
        }
      }

      for (auto U = CI->user_begin(); U != CI->user_end();) {
        Value *subsUser = *(U++);
        if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(subsUser)) {
          Value *subData = GenerateVecEltFromGEP(ldData, GEP, Builder,
                                                 /*bInsertLdNextToGEP*/ true);
          for (auto gepU = GEP->user_begin(); gepU != GEP->user_end();) {
            Value *gepUser = *(gepU++);
            // Must be load here;
            LoadInst *ldUser = cast<LoadInst>(gepUser);
            ldUser->replaceAllUsesWith(subData);
            ldUser->eraseFromParent();
          }
          GEP->eraseFromParent();
        } else {
          // Must be load here.
          LoadInst *ldUser = cast<LoadInst>(subsUser);
          ldUser->replaceAllUsesWith(ldData);
          ldUser->eraseFromParent();
        }
      }

      CI->eraseFromParent();
    } else if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(user)) {
      if( II->getIntrinsicID() == Intrinsic::lifetime_start ||
          II->getIntrinsicID() == Intrinsic::lifetime_end ) {
        DXASSERT(II->use_empty(), "lifetime intrinsic can't have uses");
        II->eraseFromParent();
      } else {
        DXASSERT(0, "not implemented yet");
      }
    } else {
      DXASSERT(0, "not implemented yet");
    }
  } else if (LoadInst *ldInst = dyn_cast<LoadInst>(user)) {
    Type *Ty = ldInst->getType();
    Type *EltTy = Ty->getScalarType();
    // Resource inside cbuffer is lowered after GenerateDxilOperations.
    if (dxilutil::IsHLSLObjectType(Ty)) {
      CallInst *CI = cast<CallInst>(handle);
      // CI should be annotate handle.
      // Need createHandle here.
      if (GetHLOpcodeGroup(CI->getCalledFunction()) == HLOpcodeGroup::HLAnnotateHandle)
        CI = cast<CallInst>(CI->getArgOperand(HLOperandIndex::kAnnotateHandleHandleOpIdx));

      GlobalVariable *CbGV = cast<GlobalVariable>(
          CI->getArgOperand(HLOperandIndex::kCreateHandleResourceOpIdx));
      TranslateResourceInCB(ldInst, pObjHelper, CbGV);
      return;
    }
    DXASSERT(!Ty->isAggregateType(), "should be flat in previous pass");

    Value *newLd = nullptr;

    if (Ty->isVectorTy())
      newLd = GenerateCBLoadLegacy(handle, legacyIdx, channelOffset, EltTy,
                                   Ty->getVectorNumElements(), hlslOP, Builder);
    else
      newLd = GenerateCBLoadLegacy(handle, legacyIdx, channelOffset, EltTy,
                                   hlslOP, Builder);

    ldInst->replaceAllUsesWith(newLd);
    dxilutil::TryScatterDebugValueToVectorElements(newLd);
    ldInst->eraseFromParent();
  } else if (BitCastInst *BCI = dyn_cast<BitCastInst>(user)) {
    for (auto it = BCI->user_begin(); it != BCI->user_end(); ) {
      Instruction *I = cast<Instruction>(*it++);
      TranslateCBAddressUserLegacy(I,
                                   handle, legacyIdx, channelOffset, hlslOP,
                                   prevFieldAnnotation, dxilTypeSys,
                                   DL, pObjHelper);
    }
    BCI->eraseFromParent();
  } else {
    // Must be GEP here
    GetElementPtrInst *GEP = cast<GetElementPtrInst>(user);
    TranslateCBGepLegacy(GEP, handle, legacyIdx, channelOffset, hlslOP, Builder,
                         prevFieldAnnotation, DL, dxilTypeSys, pObjHelper);
    GEP->eraseFromParent();
  }
}

void TranslateCBGepLegacy(GetElementPtrInst *GEP, Value *handle,
                          Value *legacyIndex, unsigned channel,
                          hlsl::OP *hlslOP, IRBuilder<> &Builder,
                          DxilFieldAnnotation *prevFieldAnnotation,
                          const DataLayout &DL, DxilTypeSystem &dxilTypeSys,
                          HLObjectOperationLowerHelper *pObjHelper) {
  SmallVector<Value *, 8> Indices(GEP->idx_begin(), GEP->idx_end());

  // update offset
  DxilFieldAnnotation *fieldAnnotation = prevFieldAnnotation;

  gep_type_iterator GEPIt = gep_type_begin(GEP), E = gep_type_end(GEP);

  for (; GEPIt != E; GEPIt++) {
    Value *idx = GEPIt.getOperand();
    unsigned immIdx = 0;
    bool bImmIdx = false;
    if (Constant *constIdx = dyn_cast<Constant>(idx)) {
      immIdx = constIdx->getUniqueInteger().getLimitedValue();
      bImmIdx = true;
    }

    if (GEPIt->isPointerTy()) {
      Type *EltTy = GEPIt->getPointerElementType();
      unsigned size = 0;
      if (StructType *ST = dyn_cast<StructType>(EltTy)) {
        DxilStructAnnotation *annotation = dxilTypeSys.GetStructAnnotation(ST);
        size = annotation->GetCBufferSize();
      } else {
        DXASSERT(fieldAnnotation, "must be a field");
        if (ArrayType *AT = dyn_cast<ArrayType>(EltTy)) {
          unsigned EltSize = dxilutil::GetLegacyCBufferFieldElementSize(
              *fieldAnnotation, EltTy, dxilTypeSys);

          // Decide the nested array size.
          unsigned nestedArraySize = 1;

          Type *EltTy = AT->getArrayElementType();
          // support multi level of array
          while (EltTy->isArrayTy()) {
            ArrayType *EltAT = cast<ArrayType>(EltTy);
            nestedArraySize *= EltAT->getNumElements();
            EltTy = EltAT->getElementType();
          }
          // Align to 4 * 4 bytes.
          unsigned alignedSize = (EltSize + 15) & 0xfffffff0;
          size = nestedArraySize * alignedSize;
        } else {
          size = DL.getTypeAllocSize(EltTy);
        }
      }
      // Skip 0 idx.
      if (bImmIdx && immIdx == 0)
        continue;
      // Align to 4 * 4 bytes.
      size = (size + 15) & 0xfffffff0;

      // Take this as array idxing.
      if (bImmIdx) {
        unsigned tempOffset = size * immIdx;
        unsigned idxInc = tempOffset >> 4;
        legacyIndex = Builder.CreateAdd(legacyIndex, hlslOP->GetU32Const(idxInc));
      } else {
        Value *idxInc = Builder.CreateMul(idx, hlslOP->GetU32Const(size>>4));
        legacyIndex = Builder.CreateAdd(legacyIndex, idxInc);
      }

      // Array always start from x channel.
      channel = 0;
    } else if (GEPIt->isStructTy()) {
      StructType *ST = cast<StructType>(*GEPIt);
      DxilStructAnnotation *annotation = dxilTypeSys.GetStructAnnotation(ST);
      fieldAnnotation = &annotation->GetFieldAnnotation(immIdx);

      unsigned idxInc = 0;
      unsigned structOffset = 0;
      if (fieldAnnotation->GetCompType().Is16Bit() &&
          !hlslOP->UseMinPrecision()) {
        structOffset = fieldAnnotation->GetCBufferOffset() >> 1;
        channel += structOffset;
        idxInc = channel >> 3;
        channel = channel & 0x7;
      }
      else {
        structOffset = fieldAnnotation->GetCBufferOffset() >> 2;
        channel += structOffset;
        idxInc = channel >> 2;
        channel = channel & 0x3;
      }
      if (idxInc) 
        legacyIndex = Builder.CreateAdd(legacyIndex, hlslOP->GetU32Const(idxInc));
    } else if (GEPIt->isArrayTy()) {
      DXASSERT(fieldAnnotation != nullptr, "must a field");
      unsigned EltSize = dxilutil::GetLegacyCBufferFieldElementSize(
              *fieldAnnotation, *GEPIt, dxilTypeSys);
      // Decide the nested array size.
      unsigned nestedArraySize = 1;

      Type *EltTy = GEPIt->getArrayElementType();
      // support multi level of array
      while (EltTy->isArrayTy()) {
        ArrayType *EltAT = cast<ArrayType>(EltTy);
        nestedArraySize *= EltAT->getNumElements();
        EltTy = EltAT->getElementType();
      }
      // Align to 4 * 4 bytes.
      unsigned alignedSize = (EltSize + 15) & 0xfffffff0;
      unsigned size = nestedArraySize * alignedSize;
      if (bImmIdx) {
        unsigned tempOffset = size * immIdx;
        unsigned idxInc = tempOffset >> 4;
        legacyIndex = Builder.CreateAdd(legacyIndex, hlslOP->GetU32Const(idxInc));
      } else {
        Value *idxInc = Builder.CreateMul(idx, hlslOP->GetU32Const(size>>4));
        legacyIndex = Builder.CreateAdd(legacyIndex, idxInc);
      }

      // Array always start from x channel.
      channel = 0;
    } else if (GEPIt->isVectorTy()) {
      unsigned size = DL.getTypeAllocSize(GEPIt->getVectorElementType());
      // Indexing on vector.
      if (bImmIdx) {
        unsigned tempOffset = size * immIdx;
        if (size == 2) { // 16-bit types
          unsigned channelInc = tempOffset >> 1;
          DXASSERT((channel + channelInc) <= 8, "vector should not cross cb register (8x16bit)");
          channel += channelInc;
          if (channel == 8) {
            // Get to another row.
            // Update index and channel.
            channel = 0;
            legacyIndex = Builder.CreateAdd(legacyIndex, Builder.getInt32(1));
          }
        }
        else {
          unsigned channelInc = tempOffset >> 2;
          DXASSERT((channel + channelInc) <= 4, "vector should not cross cb register (8x32bit)");
          channel += channelInc;
          if (channel == 4) {
            // Get to another row.
            // Update index and channel.
            channel = 0;
            legacyIndex = Builder.CreateAdd(legacyIndex, Builder.getInt32(1));
          }
        }
      } else {
        Type *EltTy = GEPIt->getVectorElementType();
        unsigned vecSize = GEPIt->getVectorNumElements();

        // Load the whole register.
        Value *newLd = GenerateCBLoadLegacy(handle, legacyIndex,
                                     /*channelOffset*/ channel, EltTy,
                                     /*vecSize*/ vecSize, hlslOP, Builder);
        // Copy to array.
        IRBuilder<> AllocaBuilder(GEP->getParent()->getParent()->getEntryBlock().getFirstInsertionPt());
        Value *tempArray = AllocaBuilder.CreateAlloca(ArrayType::get(EltTy, vecSize));
        Value *zeroIdx = hlslOP->GetU32Const(0);
        for (unsigned i = 0; i < vecSize; i++) {
          Value *Elt = Builder.CreateExtractElement(newLd, i);
          Value *EltGEP = Builder.CreateInBoundsGEP(tempArray, {zeroIdx, hlslOP->GetU32Const(i)});
          Builder.CreateStore(Elt, EltGEP);
        }
        // Make sure this is the end of GEP.
        gep_type_iterator temp = GEPIt;
        temp++;
        DXASSERT(temp == E, "scalar type must be the last");

        // Replace the GEP with array GEP.
        Value *ArrayGEP = Builder.CreateInBoundsGEP(tempArray, {zeroIdx, idx});
        GEP->replaceAllUsesWith(ArrayGEP);
        return;
      }
    } else {
      gep_type_iterator temp = GEPIt;
      temp++;
      DXASSERT(temp == E, "scalar type must be the last");
    }
  }

  for (auto U = GEP->user_begin(); U != GEP->user_end();) {
    Instruction *user = cast<Instruction>(*(U++));

    TranslateCBAddressUserLegacy(user, handle, legacyIndex, channel, hlslOP, fieldAnnotation,
                           dxilTypeSys, DL, pObjHelper);
  }
}

void TranslateCBOperationsLegacy(Value *handle, Value *ptr, OP *hlslOP,
                                 DxilTypeSystem &dxilTypeSys,
                                 const DataLayout &DL,
                                 HLObjectOperationLowerHelper *pObjHelper) {
  auto User = ptr->user_begin();
  auto UserE = ptr->user_end();
  Value *zeroIdx = hlslOP->GetU32Const(0);
  for (; User != UserE;) {
    // Must be Instruction.
    Instruction *I = cast<Instruction>(*(User++));
    TranslateCBAddressUserLegacy(
        I, handle, zeroIdx, /*channelOffset*/ 0, hlslOP,
        /*prevFieldAnnotation*/ nullptr, dxilTypeSys, DL, pObjHelper);
  }
}

}

// Structured buffer.
namespace {
// Calculate offset.
Value *GEPIdxToOffset(GetElementPtrInst *GEP, IRBuilder<> &Builder,
                      hlsl::OP *OP, const DataLayout &DL) {
  SmallVector<Value *, 8> Indices(GEP->idx_begin(), GEP->idx_end());
  Value *addr = nullptr;
  // update offset
  if (GEP->hasAllConstantIndices()) {
    unsigned gepOffset =
        DL.getIndexedOffset(GEP->getPointerOperandType(), Indices);
    addr = OP->GetU32Const(gepOffset);
  } else {
    Value *offset = OP->GetU32Const(0);
    gep_type_iterator GEPIt = gep_type_begin(GEP), E = gep_type_end(GEP);
    for (; GEPIt != E; GEPIt++) {
      Value *idx = GEPIt.getOperand();
      unsigned immIdx = 0;
      if (llvm::Constant *constIdx = dyn_cast<llvm::Constant>(idx)) {
        immIdx = constIdx->getUniqueInteger().getLimitedValue();
        if (immIdx == 0) {
          continue;
        }
      }
      if (GEPIt->isPointerTy() || GEPIt->isArrayTy() || GEPIt->isVectorTy()) {
        unsigned size = DL.getTypeAllocSize(GEPIt->getSequentialElementType());
        if (immIdx) {
          unsigned tempOffset = size * immIdx;
          offset = Builder.CreateAdd(offset, OP->GetU32Const(tempOffset));
        } else {
          Value *tempOffset = Builder.CreateMul(idx, OP->GetU32Const(size));
          offset = Builder.CreateAdd(offset, tempOffset);
        }
      } else if (GEPIt->isStructTy()) {
        const StructLayout *Layout = DL.getStructLayout(cast<StructType>(*GEPIt));
        unsigned structOffset = Layout->getElementOffset(immIdx);
        offset = Builder.CreateAdd(offset, OP->GetU32Const(structOffset));
      } else {
        gep_type_iterator temp = GEPIt;
        temp++;
        DXASSERT(temp == E, "scalar type must be the last");
      }
    };
    addr = offset;
  }
  // TODO: x4 for byte address
  return addr;
}

// Load a value from a typedef buffer with an offset.
// Typed buffer do not directly support reading at offsets
// because the whole value (e.g. float4) must be read at once.
// If we are provided a non-zero offset, we need to simulate it
// by returning the correct elements.
using ResRetValueArray = std::array<Value*, 4>;
static ResRetValueArray GenerateTypedBufferLoad(
  Value *Handle, Type *BufferElemTy, Value *ElemIdx, Value *StatusPtr,
  OP* HlslOP, IRBuilder<> &Builder) {

  OP::OpCode OpCode = OP::OpCode::BufferLoad;
  Value* LoadArgs[] = { HlslOP->GetU32Const((unsigned)OpCode), Handle, ElemIdx, UndefValue::get(Builder.getInt32Ty()) };
  Function* LoadFunc = HlslOP->GetOpFunc(OpCode, BufferElemTy);
  Value* Load = Builder.CreateCall(LoadFunc, LoadArgs, OP::GetOpCodeName(OpCode));

  ResRetValueArray ResultValues;
  for (unsigned i = 0; i < ResultValues.size(); ++i) {
    ResultValues[i] = cast<ExtractValueInst>(Builder.CreateExtractValue(Load, { i }));
  }

  UpdateStatus(Load, StatusPtr, Builder, HlslOP);
  return ResultValues;
}

static AllocaInst* SpillValuesToArrayAlloca(ArrayRef<Value*> Values, IRBuilder<>& Builder) {
  DXASSERT_NOMSG(!Values.empty());
  IRBuilder<> AllocaBuilder(dxilutil::FindAllocaInsertionPt(Builder.GetInsertPoint()));
  AllocaInst* ArrayAlloca = AllocaBuilder.CreateAlloca(ArrayType::get(Values[0]->getType(), Values.size()));
  for (unsigned i = 0; i < Values.size(); ++i) {
    Value* ArrayElemPtr = Builder.CreateGEP(ArrayAlloca, { Builder.getInt32(0), Builder.getInt32(i) });
    Builder.CreateStore(Values[i], ArrayElemPtr);
  }
  return ArrayAlloca;
}

static Value* ExtractFromTypedBufferLoad(const ResRetValueArray& ResRet,
    Type* ResultTy, Value* Offset, IRBuilder<>& Builder) {
  unsigned ElemCount = ResultTy->isVectorTy() ? ResultTy->getVectorNumElements() : 1;
  DXASSERT_NOMSG(ElemCount < ResRet.size());
  unsigned ElemSizeInBytes = ResRet[0]->getType()->getScalarSizeInBits() / 8;

  SmallVector<Value*, 4> Elems;
  if (ConstantInt *OffsetAsConstantInt = dyn_cast<ConstantInt>(Offset)) {
    // Get all elements to be returned
    uint64_t FirstElemOffset = OffsetAsConstantInt->getLimitedValue();
    DXASSERT_NOMSG(FirstElemOffset % ElemSizeInBytes == 0);
    uint64_t FirstElemIdx = FirstElemOffset / ElemSizeInBytes;
    DXASSERT_NOMSG(FirstElemIdx <= ResRet.size() - ElemCount);
    for (unsigned ElemIdx = 0; ElemIdx < ElemCount; ++ElemIdx) {
      Elems.emplace_back(ResRet[std::min<size_t>(FirstElemIdx + ElemIdx, ResRet.size() - 1)]);
    }
  }
  else {
    Value* ArrayAlloca = SpillValuesToArrayAlloca(
      ArrayRef<Value*>(ResRet.data(), ResRet.size()), Builder);

    // Get all elements to be returned through dynamic indices
    Value *FirstElemIdx = Builder.CreateUDiv(Offset, Builder.getInt32(ElemSizeInBytes));
    for (unsigned i = 0; i < ElemCount; ++i) {
      Value *ElemIdx = Builder.CreateAdd(FirstElemIdx, Builder.getInt32(i));
      Value* ElemPtr = Builder.CreateGEP(ArrayAlloca, { Builder.getInt32(0), ElemIdx });
      Elems.emplace_back(Builder.CreateLoad(ElemPtr));
    }
  }

  return ScalarizeElements(ResultTy, Elems, Builder);
}

Value *GenerateRawBufLd(Value *handle, Value *bufIdx, Value *offset,
                         Value *status, Type *EltTy,
                         MutableArrayRef<Value *> resultElts, hlsl::OP *OP,
                         IRBuilder<> &Builder, unsigned NumComponents, Constant *alignment) {
  OP::OpCode opcode = OP::OpCode::RawBufferLoad;

  DXASSERT(resultElts.size() <= 4,
           "buffer load cannot load more than 4 values");

  if (bufIdx == nullptr) {
    // This is actually a byte address buffer load with a struct template type.
    // The call takes only one coordinates for the offset.
    bufIdx = offset;
    offset = UndefValue::get(offset->getType());
  }

  Function *dxilF = OP->GetOpFunc(opcode, EltTy);
  Constant *mask = GetRawBufferMaskForETy(EltTy, NumComponents, OP);
  Value *Args[] = {OP->GetU32Const((unsigned)opcode),
                   handle,
                   bufIdx,
                   offset,
                   mask,
                   alignment};
  Value *Ld = Builder.CreateCall(dxilF, Args, OP::GetOpCodeName(opcode));

  for (unsigned i = 0; i < resultElts.size(); i++) {
    resultElts[i] = Builder.CreateExtractValue(Ld, i);
  }

  // status
  UpdateStatus(Ld, status, Builder, OP);
  return Ld;
}

void GenerateStructBufSt(Value *handle, Value *bufIdx, Value *offset,
                         Type *EltTy, hlsl::OP *OP, IRBuilder<> &Builder,
                         ArrayRef<Value *> vals, uint8_t mask, Constant *alignment) {
  OP::OpCode opcode = OP::OpCode::RawBufferStore;
  DXASSERT(vals.size() == 4, "buffer store need 4 values");

  Value *Args[] = {OP->GetU32Const((unsigned)opcode),
                   handle,
                   bufIdx,
                   offset,
                   vals[0],
                   vals[1],
                   vals[2],
                   vals[3],
                   OP->GetU8Const(mask),
                   alignment};
  Function *dxilF = OP->GetOpFunc(opcode, EltTy);
  Builder.CreateCall(dxilF, Args);
}


static Value* TranslateRawBufVecLd(Type* VecEltTy, unsigned ElemCount,
  IRBuilder<>& Builder, Value* handle, hlsl::OP* OP, Value* status,
  Value* bufIdx, Value* baseOffset, const DataLayout& DL,
  std::vector<Value*> &bufLds, unsigned baseAlign, bool isScalarTy) {

  unsigned  EltSize = DL.getTypeAllocSize(VecEltTy);
  unsigned alignment = std::min(baseAlign, EltSize);
  Constant* alignmentVal = OP->GetI32Const(alignment);

  if (baseOffset == nullptr) {
    baseOffset = OP->GetU32Const(0);
  }

  std::vector<Value*> elts(ElemCount);
  unsigned rest = (ElemCount % 4);
  for (unsigned i = 0; i < ElemCount - rest; i += 4) {
    Value* ResultElts[4];
    Value* bufLd = GenerateRawBufLd(handle, bufIdx, baseOffset, status, VecEltTy, ResultElts, OP, Builder, 4, alignmentVal);
    bufLds.emplace_back(bufLd);
    elts[i] = ResultElts[0];
    elts[i + 1] = ResultElts[1];
    elts[i + 2] = ResultElts[2];
    elts[i + 3] = ResultElts[3];

    baseOffset = Builder.CreateAdd(baseOffset, OP->GetU32Const(4 * EltSize));
  }

  if (rest) {
    Value* ResultElts[4];
    Value* bufLd = GenerateRawBufLd(handle, bufIdx, baseOffset, status, VecEltTy, ResultElts, OP, Builder, rest, alignmentVal);
    bufLds.emplace_back(bufLd);
    for (unsigned i = 0; i < rest; i++)
      elts[ElemCount - rest + i] = ResultElts[i];
  }

  // If the expected return type is scalar then skip building a vector
  if (isScalarTy) {
    return elts[0];
  }

  Value* Vec = HLMatrixLower::BuildVector(VecEltTy, elts, Builder);
  return Vec;
}

Value *TranslateStructBufMatLd(Type *matType, IRBuilder<> &Builder,
                               Value *handle, hlsl::OP *OP, Value *status,
                               Value *bufIdx, Value *baseOffset,
                               const DataLayout &DL) {
  HLMatrixType MatTy = HLMatrixType::cast(matType);
  Type *EltTy = MatTy.getElementTypeForMem();
  unsigned matSize = MatTy.getNumElements();
  std::vector<Value*> bufLds;
  Value* Vec = TranslateRawBufVecLd(EltTy, matSize, Builder, handle, OP, status, bufIdx,
    baseOffset, DL, bufLds, /*baseAlign (in bytes)*/ 8);
  Vec = MatTy.emitLoweredMemToReg(Vec, Builder);
  return Vec;
}

void TranslateStructBufMatSt(Type *matType, IRBuilder<> &Builder, Value *handle,
                             hlsl::OP *OP, Value *bufIdx, Value *baseOffset,
                             Value *val, const DataLayout &DL) {
  HLMatrixType MatTy = HLMatrixType::cast(matType);
  Type *EltTy = MatTy.getElementTypeForMem();

  val = MatTy.emitLoweredRegToMem(val, Builder);

  unsigned EltSize = DL.getTypeAllocSize(EltTy);
  Constant *Alignment = OP->GetI32Const(EltSize);
  Value *offset = baseOffset;
  if (baseOffset == nullptr)
    offset = OP->GetU32Const(0);

  unsigned matSize = MatTy.getNumElements();
  Value *undefElt = UndefValue::get(EltTy);

  unsigned storeSize = matSize;
  if (matSize % 4) {
    storeSize = matSize + 4 - (matSize & 3);
  }
  std::vector<Value *> elts(storeSize, undefElt);
  for (unsigned i = 0; i < matSize; i++)
    elts[i] = Builder.CreateExtractElement(val, i);

  for (unsigned i = 0; i < matSize; i += 4) { 
    uint8_t mask = 0;
    for (unsigned j = 0; j < 4 && (i+j) < matSize; j++) {
      if (elts[i+j] != undefElt)
        mask |= (1<<j);
    }
    GenerateStructBufSt(handle, bufIdx, offset, EltTy, OP, Builder,
                        {elts[i], elts[i + 1], elts[i + 2], elts[i + 3]}, mask,
                        Alignment);
    // Update offset by 4*4bytes.
    offset = Builder.CreateAdd(offset, OP->GetU32Const(4 * EltSize));
  }
}

void TranslateStructBufMatLdSt(CallInst *CI, Value *handle, hlsl::OP *OP,
                               Value *status, Value *bufIdx,
                               Value *baseOffset, const DataLayout &DL) {
  IRBuilder<> Builder(CI);
  HLOpcodeGroup group = hlsl::GetHLOpcodeGroupByName(CI->getCalledFunction());
  unsigned opcode = GetHLOpcode(CI);
  DXASSERT_LOCALVAR(group, group == HLOpcodeGroup::HLMatLoadStore,
                    "only translate matrix loadStore here.");
  HLMatLoadStoreOpcode matOp = static_cast<HLMatLoadStoreOpcode>(opcode);
  // Due to the current way the initial codegen generates matrix
  // orientation casts, the in-register vector matrix has already been
  // reordered based on the destination's row or column-major packing orientation.
  switch (matOp) {
  case HLMatLoadStoreOpcode::RowMatLoad:
  case HLMatLoadStoreOpcode::ColMatLoad: {
    Value *ptr = CI->getArgOperand(HLOperandIndex::kMatLoadPtrOpIdx);
    Value *NewLd = TranslateStructBufMatLd(
        ptr->getType()->getPointerElementType(), Builder, handle, OP, status,
        bufIdx, baseOffset, DL);
    CI->replaceAllUsesWith(NewLd);
  } break;
  case HLMatLoadStoreOpcode::RowMatStore:
  case HLMatLoadStoreOpcode::ColMatStore: {
    Value *ptr = CI->getArgOperand(HLOperandIndex::kMatStoreDstPtrOpIdx);
    Value *val = CI->getArgOperand(HLOperandIndex::kMatStoreValOpIdx);
    TranslateStructBufMatSt(ptr->getType()->getPointerElementType(), Builder,
                            handle, OP, bufIdx, baseOffset, val,
                            DL);
  } break;
  }

  CI->eraseFromParent();
}

void TranslateStructBufSubscriptUser(Instruction *user,
  Value *handle, HLResource::Kind ResKind,
  Value *bufIdx, Value *baseOffset, Value *status, 
  hlsl::OP *OP, const DataLayout &DL);

// For case like mat[i][j].
// IdxList is [i][0], [i][1], [i][2],[i][3].
// Idx is j.
// return [i][j] not mat[i][j] because resource ptr and temp ptr need different
// code gen.
static Value *LowerGEPOnMatIndexListToIndex(
  llvm::GetElementPtrInst *GEP, ArrayRef<Value *> IdxList) {
  IRBuilder<> Builder(GEP);
  Value *zero = Builder.getInt32(0);
  DXASSERT(GEP->getNumIndices() == 2, "must have 2 level");
  Value *baseIdx = (GEP->idx_begin())->get();
  DXASSERT_LOCALVAR(baseIdx, baseIdx == zero, "base index must be 0");
  Value *Idx = (GEP->idx_begin() + 1)->get();

  if (ConstantInt *immIdx = dyn_cast<ConstantInt>(Idx)) {
    return IdxList[immIdx->getSExtValue()];
  }
  else {
    IRBuilder<> AllocaBuilder(
      GEP->getParent()->getParent()->getEntryBlock().getFirstInsertionPt());
    unsigned size = IdxList.size();
    // Store idxList to temp array.
    ArrayType *AT = ArrayType::get(IdxList[0]->getType(), size);
    Value *tempArray = AllocaBuilder.CreateAlloca(AT);

    for (unsigned i = 0; i < size; i++) {
      Value *EltPtr = Builder.CreateGEP(tempArray, { zero, Builder.getInt32(i) });
      Builder.CreateStore(IdxList[i], EltPtr);
    }
    // Load the idx.
    Value *GEPOffset = Builder.CreateGEP(tempArray, { zero, Idx });
    return Builder.CreateLoad(GEPOffset);
  }
}

// subscript operator for matrix of struct element.
void TranslateStructBufMatSubscript(CallInst *CI,
  Value *handle, HLResource::Kind ResKind,
  Value *bufIdx, Value *baseOffset, Value *status,
  hlsl::OP* hlslOP, const DataLayout &DL) {
  unsigned opcode = GetHLOpcode(CI);
  IRBuilder<> subBuilder(CI);
  HLSubscriptOpcode subOp = static_cast<HLSubscriptOpcode>(opcode);
  Value *basePtr = CI->getArgOperand(HLOperandIndex::kMatSubscriptMatOpIdx);
  HLMatrixType MatTy = HLMatrixType::cast(basePtr->getType()->getPointerElementType());
  Type *EltTy = MatTy.getElementTypeForReg();
  Constant *alignment = hlslOP->GetI32Const(DL.getTypeAllocSize(EltTy));

  Value *EltByteSize = ConstantInt::get(
      baseOffset->getType(), GetEltTypeByteSizeForConstBuf(EltTy, DL));

  Value *idx = CI->getArgOperand(HLOperandIndex::kMatSubscriptSubOpIdx);

  Type *resultType = CI->getType()->getPointerElementType();
  unsigned resultSize = 1;
  if (resultType->isVectorTy())
    resultSize = resultType->getVectorNumElements();
  DXASSERT(resultSize <= 16, "up to 4x4 elements in vector or matrix");
  _Analysis_assume_(resultSize <= 16);
  std::vector<Value *> idxList(resultSize);

  switch (subOp) {
  case HLSubscriptOpcode::ColMatSubscript:
  case HLSubscriptOpcode::RowMatSubscript: {
    for (unsigned i = 0; i < resultSize; i++) {
      Value *offset =
          CI->getArgOperand(HLOperandIndex::kMatSubscriptSubOpIdx + i);
      offset = subBuilder.CreateMul(offset, EltByteSize);
      idxList[i] = subBuilder.CreateAdd(baseOffset, offset);
    }
  } break;
  case HLSubscriptOpcode::RowMatElement:
  case HLSubscriptOpcode::ColMatElement: {
    Constant *EltIdxs = cast<Constant>(idx);
    for (unsigned i = 0; i < resultSize; i++) {
      Value *offset =
          subBuilder.CreateMul(EltIdxs->getAggregateElement(i), EltByteSize);
      idxList[i] = subBuilder.CreateAdd(baseOffset, offset);
    }
  } break;
  default:
    DXASSERT(0, "invalid operation on const buffer");
    break;
  }

  Value *undefElt = UndefValue::get(EltTy);

  for (auto U = CI->user_begin(); U != CI->user_end();) {
    Value *subsUser = *(U++);
    if (resultSize == 1) {
      TranslateStructBufSubscriptUser(cast<Instruction>(subsUser),
        handle, ResKind, bufIdx, idxList[0], status, hlslOP, DL);
      continue;
    }
    if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(subsUser)) {
      Value *GEPOffset = LowerGEPOnMatIndexListToIndex(GEP, idxList);

      for (auto gepU = GEP->user_begin(); gepU != GEP->user_end();) {
        Instruction *gepUserInst = cast<Instruction>(*(gepU++));
        TranslateStructBufSubscriptUser(gepUserInst,
          handle, ResKind, bufIdx, GEPOffset, status, hlslOP, DL);
      }

      GEP->eraseFromParent();
    } else if (StoreInst *stUser = dyn_cast<StoreInst>(subsUser)) {
      IRBuilder<> stBuilder(stUser);
      Value *Val = stUser->getValueOperand();
      if (Val->getType()->isVectorTy()) {
        for (unsigned i = 0; i < resultSize; i++) {
          Value *EltVal = stBuilder.CreateExtractElement(Val, i);
          uint8_t mask = DXIL::kCompMask_X;
          GenerateStructBufSt(handle, bufIdx, idxList[i], EltTy, hlslOP,
                              stBuilder, {EltVal, undefElt, undefElt, undefElt},
                              mask, alignment);
        }
      } else {
        uint8_t mask = DXIL::kCompMask_X;
        GenerateStructBufSt(handle, bufIdx, idxList[0], EltTy, hlslOP,
                            stBuilder, {Val, undefElt, undefElt, undefElt},
                            mask, alignment);
      }

      stUser->eraseFromParent();
    } else {
      // Must be load here.
      LoadInst *ldUser = cast<LoadInst>(subsUser);
      IRBuilder<> ldBuilder(ldUser);
      Value *ldData = UndefValue::get(resultType);
      if (resultType->isVectorTy()) {
        for (unsigned i = 0; i < resultSize; i++) {
          Value *ResultElt;
          // TODO: This can be inefficient for row major matrix load
          GenerateRawBufLd(handle, bufIdx, idxList[i],
                              /*status*/ nullptr, EltTy, ResultElt, hlslOP,
                              ldBuilder, 1, alignment);
          ldData = ldBuilder.CreateInsertElement(ldData, ResultElt, i);
        }
      } else {
        GenerateRawBufLd(handle, bufIdx, idxList[0], /*status*/ nullptr,
                            EltTy, ldData, hlslOP, ldBuilder, 4, alignment);
      }
      ldUser->replaceAllUsesWith(ldData);
      ldUser->eraseFromParent();
    }
  }

  CI->eraseFromParent();
}

void TranslateStructBufSubscriptUser(
    Instruction *user, Value *handle, HLResource::Kind ResKind,
    Value *bufIdx, Value *baseOffset, Value *status,
    hlsl::OP *OP, const DataLayout &DL) {
  IRBuilder<> Builder(user);
  if (CallInst *userCall = dyn_cast<CallInst>(user)) {
    HLOpcodeGroup group = // user call?
        hlsl::GetHLOpcodeGroupByName(userCall->getCalledFunction());
    unsigned opcode = GetHLOpcode(userCall);
    // For case element type of structure buffer is not structure type.
    if (baseOffset == nullptr)
      baseOffset = OP->GetU32Const(0);
    if (group == HLOpcodeGroup::HLIntrinsic) {
      IntrinsicOp IOP = static_cast<IntrinsicOp>(opcode);
      switch (IOP) {
      case IntrinsicOp::MOP_Load: {
        if (userCall->getType()->isPointerTy()) {
          // Struct will return pointers which like []

        } else {
          // Use builtin types on structuredBuffer.
        }
        DXASSERT(0, "not implement yet");
      } break;
      case IntrinsicOp::IOP_InterlockedAdd: {
        AtomicHelper helper(userCall, DXIL::OpCode::AtomicBinOp, handle, bufIdx,
                            baseOffset);
        TranslateAtomicBinaryOperation(helper, DXIL::AtomicBinOpCode::Add,
                                       Builder, OP);
      } break;
      case IntrinsicOp::IOP_InterlockedAnd: {
        AtomicHelper helper(userCall, DXIL::OpCode::AtomicBinOp, handle, bufIdx,
                            baseOffset);
        TranslateAtomicBinaryOperation(helper, DXIL::AtomicBinOpCode::And,
                                       Builder, OP);
      } break;
      case IntrinsicOp::IOP_InterlockedExchange: {
        Type *opType = nullptr;
        PointerType *ptrType = dyn_cast<PointerType>(
                      userCall->getArgOperand(HLOperandIndex::kInterlockedDestOpIndex)->getType());
        if (ptrType && ptrType->getElementType()->isFloatTy())
          opType = Type::getInt32Ty(userCall->getContext());
        AtomicHelper helper(userCall, DXIL::OpCode::AtomicBinOp, handle, bufIdx,
                            baseOffset, opType);
        TranslateAtomicBinaryOperation(helper, DXIL::AtomicBinOpCode::Exchange,
                                       Builder, OP);
      } break;
      case IntrinsicOp::IOP_InterlockedMax: {
        AtomicHelper helper(userCall, DXIL::OpCode::AtomicBinOp, handle, bufIdx,
                            baseOffset);
        TranslateAtomicBinaryOperation(helper, DXIL::AtomicBinOpCode::IMax,
                                       Builder, OP);
      } break;
      case IntrinsicOp::IOP_InterlockedMin: {
        AtomicHelper helper(userCall, DXIL::OpCode::AtomicBinOp, handle, bufIdx,
                            baseOffset);
        TranslateAtomicBinaryOperation(helper, DXIL::AtomicBinOpCode::IMin,
                                       Builder, OP);
      } break;
      case IntrinsicOp::IOP_InterlockedUMax: {
        AtomicHelper helper(userCall, DXIL::OpCode::AtomicBinOp, handle, bufIdx,
                            baseOffset);
        TranslateAtomicBinaryOperation(helper, DXIL::AtomicBinOpCode::UMax,
                                       Builder, OP);
      } break;
      case IntrinsicOp::IOP_InterlockedUMin: {
        AtomicHelper helper(userCall, DXIL::OpCode::AtomicBinOp, handle, bufIdx,
                            baseOffset);
        TranslateAtomicBinaryOperation(helper, DXIL::AtomicBinOpCode::UMin,
                                       Builder, OP);
      } break;
      case IntrinsicOp::IOP_InterlockedOr: {
        AtomicHelper helper(userCall, DXIL::OpCode::AtomicBinOp, handle, bufIdx,
                            baseOffset);
        TranslateAtomicBinaryOperation(helper, DXIL::AtomicBinOpCode::Or,
                                       Builder, OP);
      } break;
      case IntrinsicOp::IOP_InterlockedXor: {
        AtomicHelper helper(userCall, DXIL::OpCode::AtomicBinOp, handle, bufIdx,
                            baseOffset);
        TranslateAtomicBinaryOperation(helper, DXIL::AtomicBinOpCode::Xor,
                                       Builder, OP);
      } break;
      case IntrinsicOp::IOP_InterlockedCompareStore:
      case IntrinsicOp::IOP_InterlockedCompareExchange: {
        AtomicHelper helper(userCall, DXIL::OpCode::AtomicCompareExchange,
                            handle, bufIdx, baseOffset);
        TranslateAtomicCmpXChg(helper, Builder, OP);
      } break;
      case IntrinsicOp::IOP_InterlockedCompareStoreFloatBitwise:
      case IntrinsicOp::IOP_InterlockedCompareExchangeFloatBitwise: {
        Type *i32Ty = Type::getInt32Ty(userCall->getContext());
        AtomicHelper helper(userCall, DXIL::OpCode::AtomicCompareExchange,
                            handle, bufIdx, baseOffset, i32Ty);
        TranslateAtomicCmpXChg(helper, Builder, OP);
      } break;
      default:
        DXASSERT(0, "invalid opcode");
        break;
      }
      userCall->eraseFromParent();
    } else if (group == HLOpcodeGroup::HLMatLoadStore)
      TranslateStructBufMatLdSt(userCall, handle, OP, status, bufIdx,
                                baseOffset, DL);
    else if (group == HLOpcodeGroup::HLSubscript) {
      TranslateStructBufMatSubscript(userCall,
        handle, ResKind, bufIdx, baseOffset, status, OP, DL);
    }
  } else if (isa<LoadInst>(user) || isa<StoreInst>(user)) {
    LoadInst *ldInst = dyn_cast<LoadInst>(user);
    StoreInst *stInst = dyn_cast<StoreInst>(user);

    Type *Ty = isa<LoadInst>(user) ? ldInst->getType()
                                   : stInst->getValueOperand()->getType();
    Type *pOverloadTy = Ty->getScalarType();
    Value *offset = baseOffset;
    unsigned arraySize = 1;
    Value *eltSize = nullptr;

    if (pOverloadTy->isArrayTy()) {
      arraySize = pOverloadTy->getArrayNumElements();
      eltSize = OP->GetU32Const(
          DL.getTypeAllocSize(pOverloadTy->getArrayElementType()));

      pOverloadTy = pOverloadTy->getArrayElementType()->getScalarType();
    }

    if (ldInst) {
      auto LdElement = [=](Value *offset, IRBuilder<> &Builder) -> Value * {
        unsigned numComponents = 0;
        if (VectorType *VTy = dyn_cast<VectorType>(Ty)) {
          numComponents = VTy->getNumElements();
        }
        else {
          numComponents = 1;
        }
        Constant *alignment =
            OP->GetI32Const(DL.getTypeAllocSize(Ty->getScalarType()));
        if (ResKind == HLResource::Kind::TypedBuffer) {
          // Typed buffer cannot have offsets, they must be loaded all at once
          ResRetValueArray ResRet = GenerateTypedBufferLoad(
            handle, pOverloadTy, bufIdx, status, OP, Builder);

          return ExtractFromTypedBufferLoad(ResRet, Ty, offset, Builder);
        }
        else {
          Value* ResultElts[4];
          GenerateRawBufLd(handle, bufIdx, offset, status, pOverloadTy,
                              ResultElts, OP, Builder, numComponents, alignment);
          return ScalarizeElements(Ty, ResultElts, Builder);
        }
      };

      Value *newLd = LdElement(offset, Builder);
      if (arraySize > 1) {
        newLd =
            Builder.CreateInsertValue(UndefValue::get(Ty), newLd, (uint64_t)0);

        for (unsigned i = 1; i < arraySize; i++) {
          offset = Builder.CreateAdd(offset, eltSize);
          Value *eltLd = LdElement(offset, Builder);
          newLd = Builder.CreateInsertValue(newLd, eltLd, i);
        }
      }
      ldInst->replaceAllUsesWith(newLd);
    } else {
      Value *val = stInst->getValueOperand();
      auto StElement = [&](Value *offset, Value *val, IRBuilder<> &Builder) {
        Value *undefVal = llvm::UndefValue::get(pOverloadTy);
        Value *vals[] = {undefVal, undefVal, undefVal, undefVal};
        uint8_t mask = 0;
        if (Ty->isVectorTy()) {
          unsigned vectorNumElements = Ty->getVectorNumElements();
          DXASSERT(vectorNumElements <= 4, "up to 4 elements in vector");
          _Analysis_assume_(vectorNumElements <= 4);
          for (unsigned i = 0; i < vectorNumElements; i++) {
            vals[i] = Builder.CreateExtractElement(val, i);
            mask |= (1<<i);
          }
        } else {
          vals[0] = val;
          mask = DXIL::kCompMask_X;
        }
        Constant *alignment =
          OP->GetI32Const(DL.getTypeAllocSize(Ty->getScalarType()));
        GenerateStructBufSt(handle, bufIdx, offset, pOverloadTy, OP, Builder,
                            vals, mask, alignment);
      };
      if (arraySize > 1)
        val = Builder.CreateExtractValue(val, 0);

      StElement(offset, val, Builder);
      if (arraySize > 1) {
        val = stInst->getValueOperand();

        for (unsigned i = 1; i < arraySize; i++) {
          offset = Builder.CreateAdd(offset, eltSize);
          Value *eltVal = Builder.CreateExtractValue(val, i);
          StElement(offset, eltVal, Builder);
        }
      }
    }
    user->eraseFromParent();
  } else if (BitCastInst *BCI = dyn_cast<BitCastInst>(user)) {
    // Recurse users
    for (auto U = BCI->user_begin(); U != BCI->user_end();) {
      Value *BCIUser = *(U++);
      TranslateStructBufSubscriptUser(cast<Instruction>(BCIUser),
        handle, ResKind, bufIdx, baseOffset, status, OP, DL);
    }
    BCI->eraseFromParent();
  } else {
    // should only used by GEP
    GetElementPtrInst *GEP = cast<GetElementPtrInst>(user);
    Type *Ty = GEP->getType()->getPointerElementType();

    Value *offset = GEPIdxToOffset(GEP, Builder, OP, DL);
    DXASSERT_LOCALVAR(Ty, offset->getType() == Type::getInt32Ty(Ty->getContext()),
             "else bitness is wrong");
    offset = Builder.CreateAdd(offset, baseOffset);

    for (auto U = GEP->user_begin(); U != GEP->user_end();) {
      Value *GEPUser = *(U++);

      TranslateStructBufSubscriptUser(cast<Instruction>(GEPUser),
        handle, ResKind, bufIdx, offset, status, OP, DL);
    }
    // delete the inst
    GEP->eraseFromParent();
  }
}

void TranslateStructBufSubscript(CallInst *CI, Value *handle, Value *status,
                                 hlsl::OP *OP, HLResource::Kind ResKind, const DataLayout &DL) {
  Value *subscriptIndex = CI->getArgOperand(HLOperandIndex::kSubscriptIndexOpIdx);
  Value* bufIdx = nullptr;
  Value *offset = nullptr;
  if (ResKind == HLResource::Kind::RawBuffer) {
    offset = subscriptIndex;
  }
  else {
    // StructuredBuffer, TypedBuffer, etc.
    bufIdx = subscriptIndex;
    offset = OP->GetU32Const(0);
  }

  for (auto U = CI->user_begin(); U != CI->user_end();) {
    Value *user = *(U++);

    TranslateStructBufSubscriptUser(cast<Instruction>(user),
      handle, ResKind, bufIdx, offset, status, OP, DL);
  }
}
}

// HLSubscript.
namespace {

Value *TranslateTypedBufLoad(CallInst *CI, DXIL::ResourceKind RK,
                             DXIL::ResourceClass RC, Value *handle,
                             LoadInst *ldInst, IRBuilder<> &Builder,
                             hlsl::OP *hlslOP, const DataLayout &DL) {
  ResLoadHelper ldHelper(CI, RK, RC, handle, IntrinsicOp::MOP_Load, /*bForSubscript*/ true);
  // Default sampleIdx for 2DMS textures.
  if (RK == DxilResource::Kind::Texture2DMS ||
      RK == DxilResource::Kind::Texture2DMSArray)
    ldHelper.mipLevel = hlslOP->GetU32Const(0);
  // use ldInst as retVal
  ldHelper.retVal = ldInst;
  TranslateLoad(ldHelper, RK, Builder, hlslOP, DL);
  // delete the ld
  ldInst->eraseFromParent();
  return ldHelper.retVal;
}

Value *UpdateVectorElt(Value *VecVal, Value *EltVal, Value *EltIdx,
                       unsigned vectorSize, Instruction *InsertPt) {
  IRBuilder<> Builder(InsertPt);
  if (ConstantInt *CEltIdx = dyn_cast<ConstantInt>(EltIdx)) {
    VecVal =
        Builder.CreateInsertElement(VecVal, EltVal, CEltIdx->getLimitedValue());
  } else {
    BasicBlock *BB = InsertPt->getParent();
    BasicBlock *EndBB = BB->splitBasicBlock(InsertPt);

    TerminatorInst *TI = BB->getTerminator();
    IRBuilder<> SwitchBuilder(TI);
    LLVMContext &Ctx = InsertPt->getContext();

    SwitchInst *Switch = SwitchBuilder.CreateSwitch(EltIdx, EndBB, vectorSize);
    TI->eraseFromParent();

    Function *F = EndBB->getParent();
    IRBuilder<> endSwitchBuilder(EndBB->begin());
    Type *Ty = VecVal->getType();
    PHINode *VecPhi = endSwitchBuilder.CreatePHI(Ty, vectorSize + 1);

    for (unsigned i = 0; i < vectorSize; i++) {
      BasicBlock *CaseBB = BasicBlock::Create(Ctx, "case", F, EndBB);
      Switch->addCase(SwitchBuilder.getInt32(i), CaseBB);
      IRBuilder<> CaseBuilder(CaseBB);

      Value *CaseVal = CaseBuilder.CreateInsertElement(VecVal, EltVal, i);
      VecPhi->addIncoming(CaseVal, CaseBB);
      CaseBuilder.CreateBr(EndBB);
    }
    VecPhi->addIncoming(VecVal, BB);
    VecVal = VecPhi;
  }
  return VecVal;
}

void TranslateDefaultSubscript(CallInst *CI, HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  Value *ptr = CI->getArgOperand(HLOperandIndex::kSubscriptObjectOpIdx);

  hlsl::OP *hlslOP = &helper.hlslOP;
  // Resource ptr.
  Value *handle = ptr;
  DXIL::ResourceClass RC = pObjHelper->GetRC(handle);
  DXIL::ResourceKind RK = pObjHelper->GetRK(handle);

  Type *Ty = CI->getType()->getPointerElementType();

  for (auto It = CI->user_begin(); It != CI->user_end(); ) {
    User *user = *(It++);
    Instruction *I = cast<Instruction>(user);
    IRBuilder<> Builder(I);
    if (LoadInst *ldInst = dyn_cast<LoadInst>(user)) {
      TranslateTypedBufLoad(CI, RK, RC, handle, ldInst, Builder, hlslOP, helper.dataLayout);
    } else if (StoreInst *stInst = dyn_cast<StoreInst>(user)) {
      Value *val = stInst->getValueOperand();
      TranslateStore(RK, handle, val,
                     CI->getArgOperand(HLOperandIndex::kStoreOffsetOpIdx),
                     Builder, hlslOP);
      // delete the st
      stInst->eraseFromParent();
    } else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(user)) {
      // Must be vector type here.
      unsigned vectorSize = Ty->getVectorNumElements();
      DXASSERT_NOMSG(GEP->getNumIndices() == 2);
      Use *GEPIdx = GEP->idx_begin();
      GEPIdx++;
      Value *EltIdx = *GEPIdx;
      for (auto GEPIt = GEP->user_begin(); GEPIt != GEP->user_end();) {
        User *GEPUser = *(GEPIt++);
        if (StoreInst *SI = dyn_cast<StoreInst>(GEPUser)) {
          IRBuilder<> StBuilder(SI);
          // Generate Ld.
          LoadInst *tmpLd = StBuilder.CreateLoad(CI);

          Value *ldVal = TranslateTypedBufLoad(CI, RK, RC, handle, tmpLd, StBuilder,
                                          hlslOP, helper.dataLayout);
          // Update vector.
          ldVal = UpdateVectorElt(ldVal, SI->getValueOperand(), EltIdx,
                                  vectorSize, SI);
          // Generate St.
          // Reset insert point, UpdateVectorElt may move SI to different block.
          StBuilder.SetInsertPoint(SI);
          TranslateStore(RK, handle, ldVal,
                         CI->getArgOperand(HLOperandIndex::kStoreOffsetOpIdx),
                         StBuilder, hlslOP);
          SI->eraseFromParent();
          continue;
        }
        if (LoadInst *LI = dyn_cast<LoadInst>(GEPUser)) {
          IRBuilder<> LdBuilder(LI);

          // Generate tmp vector load with vector type & translate it
          LoadInst *tmpLd = LdBuilder.CreateLoad(CI);

          Value *ldVal = TranslateTypedBufLoad(CI, RK, RC, handle, tmpLd, LdBuilder,
            hlslOP, helper.dataLayout);

          // get the single element
          ldVal = GenerateVecEltFromGEP(ldVal, GEP, LdBuilder,
                                        /*bInsertLdNextToGEP*/ false);

          LI->replaceAllUsesWith(ldVal);
          LI->eraseFromParent();
          continue;
        }
        // Invalid operations.
        Translated = false;
        dxilutil::EmitErrorOnInstruction(GEP, "Invalid operation on typed buffer.");
        return;
      }
      GEP->eraseFromParent();
    } else {
      CallInst *userCall = cast<CallInst>(user);
      HLOpcodeGroup group =
          hlsl::GetHLOpcodeGroupByName(userCall->getCalledFunction());
      unsigned opcode = hlsl::GetHLOpcode(userCall);
      if (group == HLOpcodeGroup::HLIntrinsic) {
        IntrinsicOp IOP = static_cast<IntrinsicOp>(opcode);
        if (RC == DXIL::ResourceClass::SRV) {
          // Invalid operations.
          Translated = false;
          dxilutil::EmitErrorOnInstruction(userCall, "Invalid operation on SRV.");
          return;
        }
        switch (IOP) {
        case IntrinsicOp::IOP_InterlockedAdd: {
          ResLoadHelper helper(CI, RK, RC, handle, IntrinsicOp::IOP_InterlockedAdd);
          AtomicHelper atomHelper(userCall, DXIL::OpCode::AtomicBinOp, handle,
                                  helper.addr, /*offset*/ nullptr);
          TranslateAtomicBinaryOperation(atomHelper, DXIL::AtomicBinOpCode::Add,
                                         Builder, hlslOP);
        } break;
        case IntrinsicOp::IOP_InterlockedAnd: {
          ResLoadHelper helper(CI, RK, RC, handle, IntrinsicOp::IOP_InterlockedAnd);
          AtomicHelper atomHelper(userCall, DXIL::OpCode::AtomicBinOp, handle,
                                  helper.addr, /*offset*/ nullptr);
          TranslateAtomicBinaryOperation(atomHelper, DXIL::AtomicBinOpCode::And,
                                         Builder, hlslOP);
        } break;
        case IntrinsicOp::IOP_InterlockedExchange: {
          ResLoadHelper helper(CI, RK, RC, handle, IntrinsicOp::IOP_InterlockedExchange);
          Type *opType = nullptr;
          PointerType *ptrType = dyn_cast<PointerType>(
                       userCall->getArgOperand(HLOperandIndex::kInterlockedDestOpIndex)->getType());
          if (ptrType && ptrType->getElementType()->isFloatTy())
            opType = Type::getInt32Ty(userCall->getContext());
          AtomicHelper atomHelper(userCall, DXIL::OpCode::AtomicBinOp, handle,
                                  helper.addr, /*offset*/ nullptr, opType);
          TranslateAtomicBinaryOperation(
              atomHelper, DXIL::AtomicBinOpCode::Exchange, Builder, hlslOP);
        } break;
        case IntrinsicOp::IOP_InterlockedMax: {
          ResLoadHelper helper(CI, RK, RC, handle, IntrinsicOp::IOP_InterlockedMax);
          AtomicHelper atomHelper(userCall, DXIL::OpCode::AtomicBinOp, handle,
                                  helper.addr, /*offset*/ nullptr);
          TranslateAtomicBinaryOperation(
              atomHelper, DXIL::AtomicBinOpCode::IMax, Builder, hlslOP);
        } break;
        case IntrinsicOp::IOP_InterlockedMin: {
          ResLoadHelper helper(CI, RK, RC, handle, IntrinsicOp::IOP_InterlockedMin);
          AtomicHelper atomHelper(userCall, DXIL::OpCode::AtomicBinOp, handle,
                                  helper.addr, /*offset*/ nullptr);
          TranslateAtomicBinaryOperation(
              atomHelper, DXIL::AtomicBinOpCode::IMin, Builder, hlslOP);
        } break;
        case IntrinsicOp::IOP_InterlockedUMax: {
          ResLoadHelper helper(CI, RK, RC, handle, IntrinsicOp::IOP_InterlockedUMax);
          AtomicHelper atomHelper(userCall, DXIL::OpCode::AtomicBinOp, handle,
                                  helper.addr, /*offset*/ nullptr);
          TranslateAtomicBinaryOperation(
              atomHelper, DXIL::AtomicBinOpCode::UMax, Builder, hlslOP);
        } break;
        case IntrinsicOp::IOP_InterlockedUMin: {
          ResLoadHelper helper(CI, RK, RC, handle, IntrinsicOp::IOP_InterlockedUMin);
          AtomicHelper atomHelper(userCall, DXIL::OpCode::AtomicBinOp, handle,
                                  helper.addr, /*offset*/ nullptr);
          TranslateAtomicBinaryOperation(
              atomHelper, DXIL::AtomicBinOpCode::UMin, Builder, hlslOP);
        } break;
        case IntrinsicOp::IOP_InterlockedOr: {
          ResLoadHelper helper(CI, RK, RC, handle, IntrinsicOp::IOP_InterlockedOr);
          AtomicHelper atomHelper(userCall, DXIL::OpCode::AtomicBinOp, handle,
                                  helper.addr, /*offset*/ nullptr);
          TranslateAtomicBinaryOperation(atomHelper, DXIL::AtomicBinOpCode::Or,
                                         Builder, hlslOP);
        } break;
        case IntrinsicOp::IOP_InterlockedXor: {
          ResLoadHelper helper(CI, RK, RC, handle, IntrinsicOp::IOP_InterlockedXor);
          AtomicHelper atomHelper(userCall, DXIL::OpCode::AtomicBinOp, handle,
                                  helper.addr, /*offset*/ nullptr);
          TranslateAtomicBinaryOperation(atomHelper, DXIL::AtomicBinOpCode::Xor,
                                         Builder, hlslOP);
        } break;
        case IntrinsicOp::IOP_InterlockedCompareStore:
        case IntrinsicOp::IOP_InterlockedCompareExchange: {
          ResLoadHelper helper(CI, RK, RC, handle, IntrinsicOp::IOP_InterlockedCompareExchange);
          AtomicHelper atomHelper(userCall, DXIL::OpCode::AtomicCompareExchange,
                                  handle, helper.addr, /*offset*/ nullptr);
          TranslateAtomicCmpXChg(atomHelper, Builder, hlslOP);
        } break;
        case IntrinsicOp::IOP_InterlockedCompareStoreFloatBitwise:
        case IntrinsicOp::IOP_InterlockedCompareExchangeFloatBitwise: {
          Type *i32Ty = Type::getInt32Ty(userCall->getContext());
          ResLoadHelper helper(CI, RK, RC, handle, IntrinsicOp::IOP_InterlockedCompareExchange);
          AtomicHelper atomHelper(userCall, DXIL::OpCode::AtomicCompareExchange,
                                  handle, helper.addr, /*offset*/ nullptr, i32Ty);
          TranslateAtomicCmpXChg(atomHelper, Builder, hlslOP);
        } break;
        default:
          DXASSERT(0, "invalid opcode");
          break;
        }
      } else {
        DXASSERT(0, "invalid group");
      }
      userCall->eraseFromParent();
    }
  }
}

void TranslateHLSubscript(CallInst *CI, HLSubscriptOpcode opcode,
                          HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  if (CI->user_empty()) {
    Translated = true;
    return;
  }
  hlsl::OP *hlslOP = &helper.hlslOP;

  Value *ptr = CI->getArgOperand(HLOperandIndex::kSubscriptObjectOpIdx);
  if (opcode == HLSubscriptOpcode::CBufferSubscript) {
    dxilutil::MergeGepUse(CI);
    // Resource ptr.
    Value *handle = CI->getArgOperand(HLOperandIndex::kSubscriptObjectOpIdx);
    if (helper.bLegacyCBufferLoad)
      TranslateCBOperationsLegacy(handle, CI, hlslOP, helper.dxilTypeSys,
                                  helper.dataLayout, pObjHelper);
    else {
      TranslateCBOperations(handle, CI, /*offset*/ hlslOP->GetU32Const(0),
                            hlslOP, helper.dxilTypeSys,
                            CI->getModule()->getDataLayout(), pObjHelper);
    }
    Translated = true;
    return;
  } else if (opcode == HLSubscriptOpcode::DoubleSubscript) {
    // Resource ptr.
    Value *handle = ptr;
    DXIL::ResourceKind RK = pObjHelper->GetRK(handle);
    Value *coord = CI->getArgOperand(HLOperandIndex::kSubscriptIndexOpIdx);
    Value *mipLevel =
        CI->getArgOperand(HLOperandIndex::kDoubleSubscriptMipLevelOpIdx);

    auto U = CI->user_begin();
    DXASSERT(CI->hasOneUse(), "subscript should only have one use");
    IRBuilder<> Builder(CI);
    if (LoadInst *ldInst = dyn_cast<LoadInst>(*U)) {
      ResLoadHelper ldHelper(ldInst, handle, coord, mipLevel);
      TranslateLoad(ldHelper, RK, Builder, hlslOP, helper.dataLayout);
      ldInst->eraseFromParent();
    } else {
      StoreInst *stInst = cast<StoreInst>(*U);
      Value *val = stInst->getValueOperand();
      TranslateStore(RK, handle, val,
                     CI->getArgOperand(HLOperandIndex::kStoreOffsetOpIdx),
                     Builder, hlslOP, mipLevel);
      stInst->eraseFromParent();
    }
    Translated = true;
    return;
  } else {
    Type *HandleTy = hlslOP->GetHandleType();
    if (ptr->getType() == HandleTy) {
      // Resource ptr.
      Value *handle = ptr;
      DXIL::ResourceKind RK = pObjHelper->GetRK(handle);
      if (RK == DxilResource::Kind::Invalid) {
        Translated = false;
        return;
      }
      Translated = true;
      Type *ObjTy = pObjHelper->GetResourceType(handle);
      Type *RetTy = ObjTy->getStructElementType(0);
      if (DXIL::IsStructuredBuffer(RK)) {
        TranslateStructBufSubscript(CI, handle, /*status*/ nullptr, hlslOP, RK,
                                    helper.dataLayout);
      } else if (RetTy->isAggregateType() &&
                 RK == DxilResource::Kind::TypedBuffer) {
        TranslateStructBufSubscript(CI, handle, /*status*/ nullptr, hlslOP, RK,
                                    helper.dataLayout);
        // Clear offset for typed buf.
        for (auto User = handle->user_begin(); User != handle->user_end(); ) {
          CallInst *CI = cast<CallInst>(*(User++));
          // Skip not lowered HL functions.
          if (hlsl::GetHLOpcodeGroupByName(CI->getCalledFunction()) != HLOpcodeGroup::NotHL)
            continue;
          switch (hlslOP->GetDxilOpFuncCallInst(CI)) {
          case DXIL::OpCode::BufferLoad: {
            CI->setArgOperand(DXIL::OperandIndex::kBufferLoadCoord1OpIdx,
                              UndefValue::get(helper.i32Ty));
          } break;
          case DXIL::OpCode::BufferStore: {
            CI->setArgOperand(DXIL::OperandIndex::kBufferStoreCoord1OpIdx,
                              UndefValue::get(helper.i32Ty));
          } break;
          case DXIL::OpCode::AtomicBinOp: {
            CI->setArgOperand(DXIL::OperandIndex::kAtomicBinOpCoord1OpIdx,
                              UndefValue::get(helper.i32Ty));
          } break;
          case DXIL::OpCode::AtomicCompareExchange: {
            CI->setArgOperand(DXIL::OperandIndex::kAtomicCmpExchangeCoord1OpIdx,
                              UndefValue::get(helper.i32Ty));
          } break;
          case DXIL::OpCode::RawBufferLoad: {
            // Structured buffer inside a typed buffer must be converted to typed buffer load.
            // Typed buffer load is equivalent to raw buffer load, except there is no mask.
            StructType *STy = cast<StructType>(CI->getFunctionType()->getReturnType());
            Type *ETy = STy->getElementType(0);
            SmallVector<Value *, 4> Args;
            Args.emplace_back(hlslOP->GetI32Const((unsigned)DXIL::OpCode::BufferLoad));
            Args.emplace_back(CI->getArgOperand(1)); // handle
            Args.emplace_back(CI->getArgOperand(2)); // index
            Args.emplace_back(UndefValue::get(helper.i32Ty)); // offset
            IRBuilder<> builder(CI);
            Function *newFunction = hlslOP->GetOpFunc(DXIL::OpCode::BufferLoad, ETy);
            CallInst *newCall = builder.CreateCall(newFunction, Args);
            CI->replaceAllUsesWith(newCall);
            CI->eraseFromParent();
          } break;
          default:
            DXASSERT(0, "Invalid operation on resource handle");
            break;
          }
        }
      } else {
        TranslateDefaultSubscript(CI, helper, pObjHelper, Translated);
      }
      return;
    }
  }

  Value *basePtr = CI->getArgOperand(HLOperandIndex::kMatSubscriptMatOpIdx);
  if (IsLocalVariablePtr(basePtr) || IsSharedMemPtr(basePtr)) {
    // Translate matrix into vector of array for share memory or local
    // variable should be done in HLMatrixLowerPass
    DXASSERT_NOMSG(0);
    Translated = true;
    return;
  }
  // Other case should be take care in TranslateStructBufSubscript or
  // TranslateCBOperations.
  Translated = false;
  return;
}

}

void TranslateSubscriptOperation(Function *F, HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper) {
  for (auto U = F->user_begin(); U != F->user_end();) {
    Value *user = *(U++);
    if (!isa<Instruction>(user))
      continue;
    // must be call inst
    CallInst *CI = cast<CallInst>(user);
    unsigned opcode = GetHLOpcode(CI);
    bool Translated = true;
    TranslateHLSubscript(
        CI, static_cast<HLSubscriptOpcode>(opcode), helper, pObjHelper, Translated);
    if (Translated) {
      // delete the call
      DXASSERT(CI->use_empty(),
               "else TranslateHLSubscript didn't replace/erase uses");
      CI->eraseFromParent();
    }
  }
}

// Create BitCast if ptr, otherwise, create alloca of new type, write to bitcast of alloca, and return load from alloca
// If bOrigAllocaTy is true: create alloca of old type instead, write to alloca, and return load from bitcast of alloca
static Instruction *BitCastValueOrPtr(Value* V, Instruction *Insert, Type *Ty, bool bOrigAllocaTy = false, const Twine &Name = "") {
  IRBuilder<> Builder(Insert);
  if (Ty->isPointerTy()) {
    // If pointer, we can bitcast directly
    return cast<Instruction>(Builder.CreateBitCast(V, Ty, Name));
  } else {
    // If value, we have to alloca, store to bitcast ptr, and load
    IRBuilder<> AllocaBuilder(dxilutil::FindAllocaInsertionPt(Insert));
    Type *allocaTy = bOrigAllocaTy ? V->getType() : Ty;
    Type *otherTy = bOrigAllocaTy ? Ty : V->getType();
    Instruction *allocaInst = AllocaBuilder.CreateAlloca(allocaTy);
    Instruction *bitCast = cast<Instruction>(Builder.CreateBitCast(allocaInst, otherTy->getPointerTo()));
    Builder.CreateStore(V, bOrigAllocaTy ? allocaInst : bitCast);
    return Builder.CreateLoad(bOrigAllocaTy ? bitCast : allocaInst, Name);
  }
}

static Instruction *CreateTransposeShuffle(IRBuilder<> &Builder, Value *vecVal, unsigned toRows, unsigned toCols) {
  SmallVector<int, 16> castMask(toCols * toRows);
  unsigned idx = 0;
  for (unsigned r = 0; r < toRows; r++)
    for (unsigned c = 0; c < toCols; c++)
      castMask[idx++] = c * toRows + r;
  return cast<Instruction>(
    Builder.CreateShuffleVector(vecVal, vecVal, castMask));
}


void TranslateHLBuiltinOperation(Function *F, HLOperationLowerHelper &helper,
                               hlsl::HLOpcodeGroup group, HLObjectOperationLowerHelper *pObjHelper) {
  if (group == HLOpcodeGroup::HLIntrinsic) {
    // map to dxil operations
    for (auto U = F->user_begin(); U != F->user_end();) {
      Value *User = *(U++);
      if (!isa<Instruction>(User))
        continue;
      // must be call inst
      CallInst *CI = cast<CallInst>(User);

      // Keep the instruction to lower by other function.
      bool Translated = true;

      TranslateBuiltinIntrinsic(CI, helper, pObjHelper, Translated);

      if (Translated) {
        // delete the call
        DXASSERT(CI->use_empty(),
                 "else TranslateBuiltinIntrinsic didn't replace/erase uses");
        CI->eraseFromParent();
      }
    }
  } else {
    if (group == HLOpcodeGroup::HLMatLoadStore) {
      // Both ld/st use arg1 for the pointer.
      Type *PtrTy =
          F->getFunctionType()->getParamType(HLOperandIndex::kMatLoadPtrOpIdx);

      if (PtrTy->getPointerAddressSpace() == DXIL::kTGSMAddrSpace) {
        // Translate matrix into vector of array for shared memory
        // variable should be done in HLMatrixLowerPass.
        if (!F->user_empty())
          F->getContext().emitError("Fail to lower matrix load/store.");
      } else if (PtrTy->getPointerAddressSpace() == DXIL::kDefaultAddrSpace) {
        // Default address space may be function argument in lib target
        if (!F->user_empty()) {
          for (auto U = F->user_begin(); U != F->user_end();) {
            Value *User = *(U++);
            if (!isa<Instruction>(User))
              continue;
            // must be call inst
            CallInst *CI = cast<CallInst>(User);
            IRBuilder<> Builder(CI);
            HLMatLoadStoreOpcode opcode = static_cast<HLMatLoadStoreOpcode>(hlsl::GetHLOpcode(CI));
            switch (opcode) {
            case HLMatLoadStoreOpcode::ColMatStore:
            case HLMatLoadStoreOpcode::RowMatStore: {
              Value *vecVal = CI->getArgOperand(HLOperandIndex::kMatStoreValOpIdx);
              Value *matPtr = CI->getArgOperand(HLOperandIndex::kMatStoreDstPtrOpIdx);
              Value *castPtr = Builder.CreateBitCast(matPtr, vecVal->getType()->getPointerTo());
              Builder.CreateStore(vecVal, castPtr);
              CI->eraseFromParent();
            } break;
            case HLMatLoadStoreOpcode::ColMatLoad:
            case HLMatLoadStoreOpcode::RowMatLoad: {
              Value *matPtr = CI->getArgOperand(HLOperandIndex::kMatLoadPtrOpIdx);
              Value *castPtr = Builder.CreateBitCast(matPtr, CI->getType()->getPointerTo());
              Value *vecVal = Builder.CreateLoad(castPtr);
              CI->replaceAllUsesWith(vecVal);
              CI->eraseFromParent();
            } break;
            }
          }
        }
      }
    } else if (group == HLOpcodeGroup::HLCast) {
      // HLCast may be used on matrix value function argument in lib target
      if (!F->user_empty()) {
        for (auto U = F->user_begin(); U != F->user_end();) {
          Value *User = *(U++);
          if (!isa<Instruction>(User))
            continue;
          // must be call inst
          CallInst *CI = cast<CallInst>(User);
          IRBuilder<> Builder(CI);
          HLCastOpcode opcode = static_cast<HLCastOpcode>(hlsl::GetHLOpcode(CI));
          bool bTranspose = false;
          bool bColDest = false;
          switch (opcode) {
          case HLCastOpcode::RowMatrixToColMatrix:
            bColDest = true;
          case HLCastOpcode::ColMatrixToRowMatrix:
            bTranspose = true;
          case HLCastOpcode::ColMatrixToVecCast:
          case HLCastOpcode::RowMatrixToVecCast: {
            Value *matVal = CI->getArgOperand(HLOperandIndex::kInitFirstArgOpIdx);
            Value *vecVal = BitCastValueOrPtr(matVal, CI, CI->getType(),
              /*bOrigAllocaTy*/false,
              matVal->getName());
            if (bTranspose) {
              HLMatrixType MatTy = HLMatrixType::cast(matVal->getType());
              unsigned row = MatTy.getNumRows();
              unsigned col = MatTy.getNumColumns();
              if (bColDest) std::swap(row, col);
              vecVal = CreateTransposeShuffle(Builder, vecVal, row, col);
            }
            CI->replaceAllUsesWith(vecVal);
            CI->eraseFromParent();
          } break;
          }
        }
      }
    } else if (group == HLOpcodeGroup::HLSubscript) {
      TranslateSubscriptOperation(F, helper, pObjHelper);
    }
    // map to math function or llvm ir
  }
}

typedef std::unordered_map<llvm::Instruction *, llvm::Value *> HandleMap;
static void TranslateHLExtension(Function *F,
                                 HLSLExtensionsCodegenHelper *helper,
                                 OP& hlslOp,
                                 HLObjectOperationLowerHelper &objHelper) {
  // Find all calls to the function F.
  // Store the calls in a vector for now to be replaced the loop below.
  // We use a two step "find then replace" to avoid removing uses while
  // iterating.
  SmallVector<CallInst *, 8> CallsToReplace;
  for (User *U : F->users()) {
    if (CallInst *CI = dyn_cast<CallInst>(U)) {
      CallsToReplace.push_back(CI);
    }
  }

  // Get the lowering strategy to use for this intrinsic.
  llvm::StringRef LowerStrategy = GetHLLowerStrategy(F);
  HLObjectExtensionLowerHelper extObjHelper(objHelper);
  ExtensionLowering lower(LowerStrategy, helper, hlslOp, extObjHelper);

  // Replace all calls that were successfully translated.
  for (CallInst *CI : CallsToReplace) {
      Value *Result = lower.Translate(CI);
      if (Result && Result != CI) {
        CI->replaceAllUsesWith(Result);
        CI->eraseFromParent();
      }
  }
}

namespace hlsl {

void TranslateBuiltinOperations(
    HLModule &HLM, HLSLExtensionsCodegenHelper *extCodegenHelper,
    std::unordered_set<Instruction *> &UpdateCounterSet) {
  HLOperationLowerHelper helper(HLM);

  HLObjectOperationLowerHelper objHelper = {HLM, UpdateCounterSet};

  Module *M = HLM.GetModule();

  SmallVector<Function *, 4> NonUniformResourceIndexIntrinsics;

  // generate dxil operation
  for (iplist<Function>::iterator F : M->getFunctionList()) {
    if (F->user_empty())
      continue;
    if (!F->isDeclaration()) {
      continue;
    }
    hlsl::HLOpcodeGroup group = hlsl::GetHLOpcodeGroup(F);
    if (group == HLOpcodeGroup::NotHL) {
      // Nothing to do.
      continue;
    }
    if (group == HLOpcodeGroup::HLExtIntrinsic) {
      TranslateHLExtension(F, extCodegenHelper, helper.hlslOP, objHelper);
      continue;
    }
    if (group == HLOpcodeGroup::HLIntrinsic) {
      CallInst *CI = cast<CallInst>(*F->user_begin()); // must be call inst
      unsigned opcode = hlsl::GetHLOpcode(CI);
      if (opcode == (unsigned)IntrinsicOp::IOP_NonUniformResourceIndex) {
        NonUniformResourceIndexIntrinsics.push_back(F);
        continue;
      }
    }
    TranslateHLBuiltinOperation(F, helper, group, &objHelper);
  }

  // Translate last so value placed in NonUniformSet is still valid.
  if (!NonUniformResourceIndexIntrinsics.empty()) {
    for (auto F : NonUniformResourceIndexIntrinsics) {
      TranslateHLBuiltinOperation(F, helper, HLOpcodeGroup::HLIntrinsic, &objHelper);
    }
  }
}

}
