///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// HLLegalizeParameter.cpp                                                   //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Legalize in parameter has write and out parameter has read.               //
// Must be call before inline pass.                                          //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/HLSL/HLModule.h"
#include "dxc/DXIL/DxilOperations.h"
#include "dxc/DXIL/DxilUtil.h"
#include "dxc/HLSL/DxilGenerationPass.h"
#include "dxc/HLSL/HLUtil.h"
#include "dxc/DXIL/DxilTypeSystem.h"

#include "llvm/IR/IntrinsicInst.h"

#include "dxc/Support/Global.h"
#include "llvm/Pass.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"

#include <vector>

using namespace llvm;
using namespace hlsl;

// For parameter need to legalize, create alloca to replace all uses of it, and copy between the alloca and the parameter.

namespace {

class HLLegalizeParameter : public ModulePass {
public:
  static char ID;
  explicit HLLegalizeParameter() : ModulePass(ID) {}
  bool runOnModule(Module &M) override;

private:
  void patchWriteOnInParam(Function &F, Argument &Arg, const DataLayout &DL);
  void patchReadOnOutParam(Function &F, Argument &Arg, const DataLayout &DL);
};

AllocaInst *createAllocaForPatch(Function &F, Type *Ty) {
  IRBuilder<> Builder(F.getEntryBlock().getFirstInsertionPt());
  return Builder.CreateAlloca(Ty);
}

void copyIn(AllocaInst *temp, Value *arg, CallInst *CI, unsigned size) {
  if (size == 0)
    return;
  // Copy arg to temp before CI.
  IRBuilder<> Builder(CI);
  Builder.CreateMemCpy(temp, arg, size, 1);
}

void copyOut(AllocaInst *temp, Value *arg, CallInst *CI, unsigned size) {
  if (size == 0)
    return;
  // Copy temp to arg after CI.
  IRBuilder<> Builder(CI->getNextNode());
  Builder.CreateMemCpy(arg, temp, size, 1);
}

bool isPointerNeedToLower(Value *V, Type *HandleTy) {
  // CBuffer, Buffer, Texture....
  // Anything related to dxil op.
  // hl.subscript.
  // Got to root of GEP.
  while (GEPOperator *GEP = dyn_cast<GEPOperator>(V)) {
    V = GEP->getPointerOperand();
  }
  CallInst *CI = dyn_cast<CallInst>(V);
  if (!CI) {
    // If array of vector, we need a copy to handle vector to array in LowerTypePasses.
    Type *Ty = V->getType();
    if (Ty->isPointerTy())
      Ty = Ty->getPointerElementType();
    if (!Ty->isArrayTy())
      return false;
    while (Ty->isArrayTy()) {
      Ty = Ty->getArrayElementType();
    }
    return Ty->isVectorTy();
  }
  HLOpcodeGroup group = GetHLOpcodeGroup(CI->getCalledFunction());
  if (group != HLOpcodeGroup::HLSubscript)
    return false;
  Value *Ptr = CI->getArgOperand(HLOperandIndex::kSubscriptObjectOpIdx);

  // Ptr from resource handle.
  if (Ptr->getType() == HandleTy)
    return true;
  unsigned Opcode = GetHLOpcode(CI);
  // Ptr from cbuffer.
  if (Opcode == (unsigned)HLSubscriptOpcode::CBufferSubscript)
    return true;

  return isPointerNeedToLower(Ptr, HandleTy);
}

bool mayAliasWithGlobal(Value *V, CallInst *CallSite, std::vector<GlobalVariable *> &staticGVs) {
  // The unsafe case need copy-in copy-out will be global variable alias with
  // parameter. Then global variable is updated in the function, the parameter
  // will be updated silently.

  // Currently add copy for all non-const static global in
  // CGMSHLSLRuntime::EmitHLSLOutParamConversionInit.
  //So here just return false and do nothing.
  // For case like
  // struct T {
  //  float4 a[10];
  //};
  // static T g;
  // void foo(inout T t) {
  //  // modify g
  //}
  // void bar() {
  //  T t = g;
  //  // Not copy because t is local.
  //  // But optimizations will change t to g later.
  //  foo(t);
  //}
  // Optimizations which remove the copy should not replace foo(t) into foo(g)
  // when g could be modified.
  // TODO: remove copy for global in
  // CGMSHLSLRuntime::EmitHLSLOutParamConversionInit, do analysis to check alias
  // only generate copy when there's alias.
  return false;
}

struct CopyData {
  CallInst *CallSite;
  Value *Arg;
  bool bCopyIn;
  bool bCopyOut;
};

void ParameterCopyInCopyOut(hlsl::HLModule &HLM) {
  Module &M = *HLM.GetModule();
  Type *HandleTy = HLM.GetOP()->GetHandleType();
  const DataLayout &DL = M.getDataLayout();

  std::vector<GlobalVariable *> staticGVs;
  for (GlobalVariable &GV : M.globals()) {
    if (dxilutil::IsStaticGlobal(&GV) && !GV.isConstant()) {
      staticGVs.emplace_back(&GV);
    }
  }

  SmallVector<CopyData, 4> WorkList;
  for (Function &F : M) {
    if (F.user_empty())
      continue;
    DxilFunctionAnnotation *Annot = HLM.GetFunctionAnnotation(&F);
    // Skip functions don't have annotation, include llvm intrinsic and HLOp
    // functions.
    if (!Annot)
      continue;

    bool bNoInline = F.hasFnAttribute(llvm::Attribute::NoInline) || F.isDeclaration();

    for (User *U : F.users()) {
      CallInst *CI = dyn_cast<CallInst>(U);
      if (!CI)
        continue;
      for (unsigned i = 0; i < CI->getNumArgOperands(); i++) {
        Value *arg = CI->getArgOperand(i);

        if (!arg->getType()->isPointerTy())
          continue;

        DxilParameterAnnotation &ParamAnnot = Annot->GetParameterAnnotation(i);
        bool bCopyIn = false;
        bool bCopyOut = false;
        switch (ParamAnnot.GetParamInputQual()) {
        default:
          break;
        case DxilParamInputQual::In: {
          bCopyIn = true;
        } break;
        case DxilParamInputQual::Out: {
          bCopyOut = true;
        } break;
        case DxilParamInputQual::Inout: {
          bCopyIn = true;
          bCopyOut = true;
        } break;
        }

        if (!bCopyIn && !bCopyOut)
          continue;

        // When use ptr from cbuffer/buffer, need copy to avoid lower on user
        // function.
        bool bNeedCopy = mayAliasWithGlobal(arg, CI, staticGVs);
        if (bNoInline)
          bNeedCopy |= isPointerNeedToLower(arg, HandleTy);

        if (!bNeedCopy)
          continue;

        CopyData data = {CI, arg, bCopyIn, bCopyOut};
        WorkList.emplace_back(data);
      }
    }
  }

  for (CopyData &data : WorkList) {
    CallInst *CI = data.CallSite;
    Value *arg = data.Arg;
    Type *Ty = arg->getType()->getPointerElementType();
    Type *EltTy = dxilutil::GetArrayEltTy(Ty);
    // Skip on object type and resource type.
    if (dxilutil::IsHLSLObjectType(EltTy) ||
        dxilutil::IsHLSLResourceType(EltTy))
      continue;
    unsigned size = DL.getTypeAllocSize(Ty);
    AllocaInst *temp = createAllocaForPatch(*CI->getParent()->getParent(), Ty);
    // TODO: Adding lifetime intrinsics isn't easy here, have to analyze uses.
    if (data.bCopyIn)
      copyIn(temp, arg, CI, size);
    if (data.bCopyOut)
      copyOut(temp, arg, CI, size);
    CI->replaceUsesOfWith(arg, temp);
  }
}

} // namespace

bool HLLegalizeParameter::runOnModule(Module &M) {
  HLModule &HLM = M.GetOrCreateHLModule();

  auto &typeSys = HLM.GetTypeSystem();
  const DataLayout &DL = M.getDataLayout();

  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    DxilFunctionAnnotation *Annot = HLM.GetFunctionAnnotation(&F);
    if (!Annot)
      continue;

    for (Argument &Arg : F.args()) {
      if (!Arg.getType()->isPointerTy())
        continue;
      Type *EltTy = dxilutil::GetArrayEltTy(Arg.getType());
      if (dxilutil::IsHLSLObjectType(EltTy) ||
          dxilutil::IsHLSLResourceType(EltTy))
        continue;

      DxilParameterAnnotation &ParamAnnot =
          Annot->GetParameterAnnotation(Arg.getArgNo());
      switch (ParamAnnot.GetParamInputQual()) {
      default:
        break;
      case DxilParamInputQual::In: {
        hlutil::PointerStatus PS(&Arg, 0, /*bLdStOnly*/ true);
        PS.analyze(typeSys, /*bStructElt*/ false);
        if (PS.HasStored()) {
          patchWriteOnInParam(F, Arg, DL);
        }
      } break;
      case DxilParamInputQual::Out: {
        hlutil::PointerStatus PS(&Arg, 0, /*bLdStOnly*/ true);
        PS.analyze(typeSys, /*bStructElt*/false);
        if (PS.HasLoaded()) {
          patchReadOnOutParam(F, Arg, DL);
        }
      }
      }
    }
  }

  // Copy-in copy-out for ptr arg when need.
  ParameterCopyInCopyOut(HLM);

  return true;
}

void HLLegalizeParameter::patchWriteOnInParam(Function &F, Argument &Arg,
                                              const DataLayout &DL) {
  // TODO: Adding lifetime intrinsics isn't easy here, have to analyze uses.
  Type *Ty = Arg.getType()->getPointerElementType();
  AllocaInst *temp = createAllocaForPatch(F, Ty);
  Arg.replaceAllUsesWith(temp);
  IRBuilder<> Builder(temp->getNextNode());
  unsigned size = DL.getTypeAllocSize(Ty);
  // copy arg to temp at beginning of function.
  Builder.CreateMemCpy(temp, &Arg, size, 1);
}

void HLLegalizeParameter::patchReadOnOutParam(Function &F, Argument &Arg,
                                              const DataLayout &DL) {
  // TODO: Adding lifetime intrinsics isn't easy here, have to analyze uses.
  Type *Ty = Arg.getType()->getPointerElementType();
  AllocaInst *temp = createAllocaForPatch(F, Ty);
  Arg.replaceAllUsesWith(temp);

  unsigned size = DL.getTypeAllocSize(Ty);
  for (auto &BB : F.getBasicBlockList()) {
    // copy temp to arg before every return.
    if (ReturnInst *RI = dyn_cast<ReturnInst>(BB.getTerminator())) {
      IRBuilder<> RetBuilder(RI);
      RetBuilder.CreateMemCpy(&Arg, temp, size, 1);
    }
  }
}

char HLLegalizeParameter::ID = 0;
ModulePass *llvm::createHLLegalizeParameter() {
  return new HLLegalizeParameter();
}

INITIALIZE_PASS(HLLegalizeParameter, "hl-legalize-parameter",
                "Legalize parameter", false, false)
