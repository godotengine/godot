///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilPromoteResourcePasses.cpp                                             //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/DXIL/DxilUtil.h"
#include "dxc/HLSL/DxilGenerationPass.h"
#include "dxc/HLSL/HLModule.h"
#include "dxc/DXIL/DxilResourceBase.h"
#include "dxc/DXIL/DxilResource.h"
#include "dxc/DXIL/DxilCBuffer.h"
#include "dxc/DXIL/DxilOperations.h"
#include "dxc/DXIL/DxilModule.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Operator.h"

#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"
#include <unordered_set>
#include <vector>

using namespace llvm;
using namespace hlsl;

// Legalize resource use.
// Map local or static global resource to global resource.
// Require inline for static global resource.

namespace {

static const StringRef kStaticResourceLibErrorMsg = "non const static global resource use is disallowed in library exports.";

class DxilPromoteStaticResources : public ModulePass {
public:
  static char ID; // Pass identification, replacement for typeid
  explicit DxilPromoteStaticResources()
      : ModulePass(ID) {}

  StringRef getPassName() const override {
    return "DXIL Legalize Static Resource Use";
  }

  bool runOnModule(Module &M) override {
    // Promote static global variables.
    return PromoteStaticGlobalResources(M);
  }

private:
  bool PromoteStaticGlobalResources(Module &M);
};

char DxilPromoteStaticResources::ID = 0;

class DxilPromoteLocalResources : public FunctionPass {
  void getAnalysisUsage(AnalysisUsage &AU) const override;

public:
  static char ID; // Pass identification, replacement for typeid
  explicit DxilPromoteLocalResources()
      : FunctionPass(ID) {}

  StringRef getPassName() const override {
    return "DXIL Legalize Resource Use";
  }

  bool runOnFunction(Function &F) override {
    // Promote local resource first.
    return PromoteLocalResource(F);
  }

private:
  bool PromoteLocalResource(Function &F);
};

char DxilPromoteLocalResources::ID = 0;

}

void DxilPromoteLocalResources::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<AssumptionCacheTracker>();
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.setPreservesAll();
}

bool DxilPromoteLocalResources::PromoteLocalResource(Function &F) {
  bool bModified = false;
  std::vector<AllocaInst *> Allocas;
  DominatorTree *DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  AssumptionCache &AC =
      getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);

  BasicBlock &BB = F.getEntryBlock();
  unsigned allocaSize = 0;
  while (1) {
    Allocas.clear();

    // Find allocas that are safe to promote, by looking at all instructions in
    // the entry node
    for (BasicBlock::iterator I = BB.begin(), E = --BB.end(); I != E; ++I)
      if (AllocaInst *AI = dyn_cast<AllocaInst>(I)) { // Is it an alloca?
        if (dxilutil::IsHLSLObjectType(dxilutil::GetArrayEltTy(AI->getAllocatedType()))) {
          if (isAllocaPromotable(AI))
            Allocas.push_back(AI);
        }
      }
    if (Allocas.empty())
      break;

    // No update.
    // Report error and break.
    if (allocaSize == Allocas.size()) {
      F.getContext().emitError(dxilutil::kResourceMapErrorMsg);
      break;
    }
    allocaSize = Allocas.size();

    PromoteMemToReg(Allocas, *DT, nullptr, &AC);
    bModified = true;
  }

  return bModified;
}

FunctionPass *llvm::createDxilPromoteLocalResources() {
  return new DxilPromoteLocalResources();
}

INITIALIZE_PASS_BEGIN(DxilPromoteLocalResources,
                      "hlsl-dxil-promote-local-resources",
                      "DXIL promote local resource use", false, true)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(DxilPromoteLocalResources,
                    "hlsl-dxil-promote-local-resources",
                    "DXIL promote local resource use", false, true)

bool DxilPromoteStaticResources::PromoteStaticGlobalResources(
    Module &M) {
  if (M.GetOrCreateHLModule().GetShaderModel()->IsLib()) {
    // Read/write to global static resource is disallowed for libraries:
    // Resource use needs to be resolved to a single real global resource,
    // but it may not be possible since any external function call may re-enter
    // at any other library export, which could modify the global static
    // between write and read.
    // While it could work for certain cases, describing the boundary at
    // the HLSL level is difficult, so at this point it's better to disallow.
    // example of what could work:
    //  After inlining, exported functions must have writes to static globals
    //  before reads, and must not have any external function calls between
    //  writes and subsequent reads, such that the static global may be
    //  optimized away for the exported function.
    for (auto &GV : M.globals()) {
      if (GV.getLinkage() == GlobalVariable::LinkageTypes::InternalLinkage &&
		!GV.isConstant() && 
        dxilutil::IsHLSLObjectType(dxilutil::GetArrayEltTy(GV.getType()))) {
        if (!GV.user_empty()) {
          if (Instruction *I = dyn_cast<Instruction>(*GV.user_begin())) {
            dxilutil::EmitErrorOnInstruction(I, kStaticResourceLibErrorMsg);
            break;
          }
        }
      }
    }
    return false;
  }

  bool bModified = false;
  std::set<GlobalVariable *> staticResources;
  for (auto &GV : M.globals()) {
    if (GV.getLinkage() == GlobalVariable::LinkageTypes::InternalLinkage &&
        dxilutil::IsHLSLObjectType(dxilutil::GetArrayEltTy(GV.getType()))) {
      staticResources.insert(&GV);
    }
  }
  SSAUpdater SSA;
  SmallVector<Instruction *, 4> Insts;
  // Make sure every resource load has mapped to global variable.
  while (!staticResources.empty()) {
    bool bUpdated = false;
    for (auto it = staticResources.begin(); it != staticResources.end();) {
      GlobalVariable *GV = *(it++);
      // Build list of instructions to promote.
      for (User *U : GV->users()) {
        if (isa<LoadInst>(U) || isa<StoreInst>(U)) {
          Insts.emplace_back(cast<Instruction>(U));
        } else if (GEPOperator *GEP = dyn_cast<GEPOperator>(U)) {
          for (User *gepU : GEP->users()) {
            DXASSERT_NOMSG(isa<LoadInst>(gepU) || isa<StoreInst>(gepU));
            if (isa<LoadInst>(gepU) || isa<StoreInst>(gepU))
              Insts.emplace_back(cast<Instruction>(gepU));
          }
        } else {
          DXASSERT(false, "Unhandled user of resource static global");
        }
      }

      LoadAndStorePromoter(Insts, SSA).run(Insts);
      GV->removeDeadConstantUsers();
      if (GV->user_empty()) {
        bUpdated = true;
        staticResources.erase(GV);
      }

      Insts.clear();
    }
    if (!bUpdated) {
      M.getContext().emitError(dxilutil::kResourceMapErrorMsg);
      break;
    }
    bModified = true;
  }
  return bModified;
}

ModulePass *llvm::createDxilPromoteStaticResources() {
  return new DxilPromoteStaticResources();
}

INITIALIZE_PASS(DxilPromoteStaticResources,
                "hlsl-dxil-promote-static-resources",
                "DXIL promote static resource use", false, false)

// Mutate high-level resource type into handle.
// This is used for SM 6.6+, on libraries only, where
// CreateHandleForLib is eliminated, and high-level resource
// types are only preserved in metadata for reflection purposes.
namespace {
// Overview
// 1. collectCandidates - collect to MutateValSet
//    Start from resource global variable, function parameter/ret, alloca.
//    Propagate to all insts, GEP/ld/st/phi/select/called functions.
// 2. mutateCandidates
//    Mutate all non-function value types.
//    Mutate functions by creating new function with new type, then
//    splice original function blocks into new function, and
//    replace old argument uses with new function's arguments.
class DxilMutateResourceToHandle : public ModulePass {
public:
  static char ID; // Pass identification, replacement for typeid
  explicit DxilMutateResourceToHandle() : ModulePass(ID) {}

  StringRef getPassName() const override {
    return "DXIL Mutate resource to handle";
  }

  bool runOnModule(Module &M) override {
    if (M.HasHLModule()) {
      auto &HLM = M.GetHLModule();
      if (!HLM.GetShaderModel()->IsSM66Plus())
        return false;

      hlslOP = HLM.GetOP();
      pTypeSys = &HLM.GetTypeSystem();
    } else if (M.HasDxilModule()) {
      auto &DM = M.GetDxilModule();
      if (!DM.GetShaderModel()->IsSM66Plus())
        return false;

      hlslOP = DM.GetOP();
      pTypeSys = &DM.GetTypeSystem();
    } else {
      return false;
    }

    hdlTy = hlslOP->GetHandleType();
    if (hlslOP->IsDxilOpUsed(DXIL::OpCode::CreateHandleForLib)) {
      createHandleForLibOnHandle =
          hlslOP->GetOpFunc(DXIL::OpCode::CreateHandleForLib, hdlTy);
    }

    collectCandidates(M);
    mutateCandidates(M);
    // Remove cast to handle.
    return !MutateValSet.empty();
  }

private:
  Type *mutateToHandleTy(Type *Ty, bool bResType = false);
  bool mutateTypesToHandleTy(SmallVector<Type *, 4> &Tys);

  void collectGlobalResource(DxilResourceBase *Res,
                             SmallVector<Value *, 8> &WorkList);
  void collectAlloca(Function &F, SmallVector<Value *, 8> &WorkList);

  SmallVector<Value *, 8> collectHlslObjects(Module &M);

  void collectCandidates(Module &M);
  void mutateCandidates(Module &M);

  Type *hdlTy = nullptr;
  hlsl::OP *hlslOP = nullptr;
  Function *createHandleForLibOnHandle = nullptr;
  DxilTypeSystem *pTypeSys;
  DenseSet<Value *> MutateValSet;
  DenseMap<Type *, Type *> MutateTypeMap;
};

char DxilMutateResourceToHandle::ID = 0;

Type *DxilMutateResourceToHandle::mutateToHandleTy(Type *Ty, bool bResType) {
  auto it = MutateTypeMap.find(Ty);
  if (it != MutateTypeMap.end())
    return it->second;

  Type *ResultTy = nullptr;
  if (ArrayType *AT = dyn_cast<ArrayType>(Ty)) {
    SmallVector<unsigned, 2> nestedSize;
    Type *EltTy = Ty;
    while (ArrayType *NestAT = dyn_cast<ArrayType>(EltTy)) {
      nestedSize.emplace_back(NestAT->getNumElements());
      EltTy = NestAT->getElementType();
    }
    Type *mutatedTy = mutateToHandleTy(EltTy, bResType);
    if (mutatedTy == EltTy) {
      ResultTy = Ty;
    } else {
      Type *newAT = mutatedTy;
      for (auto it = nestedSize.rbegin(), E = nestedSize.rend(); it != E; ++it)
        newAT = ArrayType::get(newAT, *it);
      ResultTy = newAT;
    }
  } else if (PointerType *PT = dyn_cast<PointerType>(Ty)) {
    Type *EltTy = PT->getElementType();
    Type *mutatedTy = mutateToHandleTy(EltTy, bResType);
    if (mutatedTy == EltTy)
      ResultTy = Ty;
    else
      ResultTy = mutatedTy->getPointerTo(PT->getAddressSpace());
  } else if (dxilutil::IsHLSLResourceType(Ty)) {
    ResultTy = hdlTy;
  } else if (StructType *ST = dyn_cast<StructType>(Ty)) {
    if (bResType) {
      // For top-level resource GV type, the first struct type is the resource
      // type to be changed into handle.
      ResultTy = hdlTy;
    } else if (!ST->isOpaque()) {
      SmallVector<Type *, 4> Elts(ST->element_begin(), ST->element_end());
      if (!mutateTypesToHandleTy(Elts)) {
        ResultTy = Ty;
      } else {
        ResultTy = StructType::create(Elts, ST->getName().str() + ".hdl");
      }
    } else {

      // FIXME: Opaque type "ConstantBuffer" is only used for an empty cbuffer.
      // We should never use an empty cbuffer, and we should try to get rid of
      // the need for this type in the first place, like using nullptr for the
      // DxilResourceBase::m_pSymbol, since this resource should get deleted
      // before final dxil in all cases.

      if (ST->getName() == "ConstantBuffer")
        ResultTy = hdlTy;
      else
        ResultTy = Ty;
    }
  } else if (FunctionType *FT = dyn_cast<FunctionType>(Ty)) {
    Type *RetTy = FT->getReturnType();
    SmallVector<Type *, 4> Args(FT->param_begin(), FT->param_end());
    Type *mutatedRetTy = mutateToHandleTy(RetTy);
    if (!mutateTypesToHandleTy(Args) && RetTy == mutatedRetTy) {
      ResultTy = Ty;
    } else {
      ResultTy = FunctionType::get(mutatedRetTy, Args, FT->isVarArg());
    }
  } else {
    ResultTy = Ty;
  }
  MutateTypeMap[Ty] = ResultTy;
  return ResultTy;
}

bool DxilMutateResourceToHandle::mutateTypesToHandleTy(
    SmallVector<Type *, 4> &Tys) {
  bool bMutated = false;
  for (size_t i = 0; i < Tys.size(); i++) {
    Type *Ty = Tys[i];
    Type *mutatedTy = mutateToHandleTy(Ty);
    if (Ty != mutatedTy) {
      Tys[i] = mutatedTy;
      bMutated = true;
    }
  }
  return bMutated;
}

void DxilMutateResourceToHandle::collectGlobalResource(
    DxilResourceBase *Res, SmallVector<Value *, 8> &WorkList) {
  Value *GV = Res->GetGlobalSymbol();
  // If already handle, don't overwrite HLSL type.
  // It's still posible that load users have a wrong type (invalid IR) due to
  // linking mixed targets.  But in that case, we need to start at the
  // non-handle overloads of CreateHandleForLib and mutate/rewrite from there.
  // That's because we may have an already translated GV, but some load and
  // CreateHandleForLib calls use the wrong type from linked code.
  Type *MTy = mutateToHandleTy(GV->getType(), /*bResType*/true);
  if (GV->getType() != MTy) {
    // Save hlsl type before mutate to handle.
    Res->SetHLSLType(GV->getType());
    WorkList.emplace_back(GV);
  }
}
void DxilMutateResourceToHandle::collectAlloca(
    Function &F, SmallVector<Value *, 8> &WorkList) {
  if (F.isDeclaration())
    return;
  for (Instruction &I : F.getEntryBlock()) {
    AllocaInst *AI = dyn_cast<AllocaInst>(&I);
    if (!AI)
      continue;
    Type *Ty = AI->getType();
    Type *MTy = mutateToHandleTy(Ty);
    if (Ty == MTy)
      continue;
    WorkList.emplace_back(AI);
  }
}

}

SmallVector<Value *, 8>
DxilMutateResourceToHandle::collectHlslObjects(Module &M) {
  // Add all global/function/argument/alloca has resource type.
  SmallVector<Value *, 8> WorkList;

  // Global resources.
  if (M.HasHLModule()) {
    auto &HLM = M.GetHLModule();
    for (auto &Res : HLM.GetCBuffers()) {
      collectGlobalResource(Res.get(), WorkList);
    }
    for (auto &Res : HLM.GetSRVs()) {
      collectGlobalResource(Res.get(), WorkList);
    }
    for (auto &Res : HLM.GetUAVs()) {
      collectGlobalResource(Res.get(), WorkList);
    }
    for (auto &Res : HLM.GetSamplers()) {
      collectGlobalResource(Res.get(), WorkList);
    }
  } else {
    auto &DM = M.GetDxilModule();
    for (auto &Res : DM.GetCBuffers()) {
      collectGlobalResource(Res.get(), WorkList);
    }
    for (auto &Res : DM.GetSRVs()) {
      collectGlobalResource(Res.get(), WorkList);
    }
    for (auto &Res : DM.GetUAVs()) {
      collectGlobalResource(Res.get(), WorkList);
    }
    for (auto &Res : DM.GetSamplers()) {
      collectGlobalResource(Res.get(), WorkList);
    }
  }

  // Assume this is after SROA so no struct for global/alloca.

  // Functions.
  for (Function &F : M) {
    if (hlslOP && hlslOP->IsDxilOpFunc(&F)) {
      DXIL::OpCodeClass OpcodeClass;
      if (hlslOP->GetOpCodeClass(&F, OpcodeClass)) {
        if (OpcodeClass == DXIL::OpCodeClass::CreateHandleForLib &&
            &F != createHandleForLibOnHandle) {
          WorkList.emplace_back(&F);
          MutateTypeMap[F.getFunctionType()->getFunctionParamType(1)] = hdlTy;
          continue;
        }
      }
    }

    collectAlloca(F, WorkList);
    FunctionType *FT = F.getFunctionType();
    FunctionType *MFT = cast<FunctionType>(mutateToHandleTy(FT));
    if (FT == MFT)
      continue;
    WorkList.emplace_back(&F);
    // Check args.
    for (Argument &Arg : F.args()) {
      Type *Ty = Arg.getType();
      Type *MTy = mutateToHandleTy(Ty);
      if (Ty == MTy)
        continue;
      WorkList.emplace_back(&Arg);
    }
  }

  // Static globals.
  for (GlobalVariable &GV : M.globals()) {
    if (!dxilutil::IsStaticGlobal(&GV))
      continue;
    Type *Ty = dxilutil::GetArrayEltTy(GV.getValueType());
    if (!dxilutil::IsHLSLObjectType(Ty))
      continue;
    WorkList.emplace_back(&GV);
  }

  return WorkList;
}

void DxilMutateResourceToHandle::collectCandidates(Module &M) {
  SmallVector<Value *, 8> WorkList = collectHlslObjects(M);

  // Propagate candidates.
  while (!WorkList.empty()) {
    Value *V = WorkList.pop_back_val();
    MutateValSet.insert(V);

    for (User *U : V->users()) {
      // collect in a user.
      SmallVector<Value *, 2> newCandidates;
      // Should only used by ld/st/sel/phi/gep/call.
      if (LoadInst *LI = dyn_cast<LoadInst>(U)) {
        newCandidates.emplace_back(LI);
      } else if (StoreInst *SI = dyn_cast<StoreInst>(U)) {
        Value *Ptr = SI->getPointerOperand();
        Value *Val = SI->getValueOperand();
        if (V == Ptr)
          newCandidates.emplace_back(Val);
        else
          newCandidates.emplace_back(Ptr);
      } else if (GEPOperator *GEP = dyn_cast<GEPOperator>(U)) {
        // If result type of GEP not related to resource type, skip.
        Type *Ty = GEP->getType();
        Type *MTy = mutateToHandleTy(Ty);
        if (MTy == Ty)
          continue;
        newCandidates.emplace_back(GEP);
      } else if (PHINode *Phi = dyn_cast<PHINode>(U)) {
        // Propagate all operands.
        newCandidates.emplace_back(Phi);
        for (Use &PhiOp : Phi->incoming_values()) {
          if (V == PhiOp)
            continue;
          newCandidates.emplace_back(PhiOp);
        }
      } else if (SelectInst *Sel = dyn_cast<SelectInst>(U)) {
        // Propagate other result.
        newCandidates.emplace_back(Sel);
        Value *TrueV = Sel->getTrueValue();
        Value *FalseV = Sel->getFalseValue();
        if (TrueV == V)
          newCandidates.emplace_back(FalseV);
        else
          newCandidates.emplace_back(TrueV);
      } else if (BitCastOperator *BCO = dyn_cast<BitCastOperator>(U)) {
        // Make sure only used for lifetime intrinsic.
        for (User *BCUser : BCO->users()) {
          if (ConstantArray *CA = dyn_cast<ConstantArray>(BCUser)) {
            // For llvm.used.
            if (CA->hasOneUse()) {
              Value *CAUser = CA->user_back();
              if (GlobalVariable *GV = dyn_cast<GlobalVariable>(CAUser)) {
                if (GV->getName() == "llvm.used")
                  continue;
              }
            } else if (CA->user_empty()) {
              continue;
            }
          }
          CallInst *CI = cast<CallInst>(BCUser);
          Function *F = CI->getCalledFunction();
          Intrinsic::ID ID = F->getIntrinsicID();
          if (ID != Intrinsic::lifetime_start &&
              ID != Intrinsic::lifetime_end) {
            DXASSERT(false, "unexpected resource object user");
          }
        }
      } else {
        CallInst *CI = cast<CallInst>(U);
        Type *Ty = CI->getType();
        Type *MTy = mutateToHandleTy(Ty);
        if (Ty != MTy)
          newCandidates.emplace_back(CI);

        SmallVector<Value *, 4> Args(CI->arg_operands().begin(),
                                     CI->arg_operands().end());
        for (Value *Arg : Args) {
          if (Arg == V)
            continue;
          Type *Ty = Arg->getType();
          Type *MTy = mutateToHandleTy(Ty);
          if (Ty == MTy)
            continue;
          newCandidates.emplace_back(Arg);
        }
      }

      for (Value *Val : newCandidates) {
        // New candidate find.
        if (MutateValSet.insert(Val).second) {
          WorkList.emplace_back(Val);
        }
      }
    }
  }
}

void DxilMutateResourceToHandle::mutateCandidates(Module &M) {
  SmallVector<Function *, 2> CandidateFns;
  for (Value *V : MutateValSet) {
    if (Function *F = dyn_cast<Function>(V)) {
      CandidateFns.emplace_back(F);
      continue;
    }
    Type *Ty = V->getType();
    Type *MTy = mutateToHandleTy(Ty);
    if (AllocaInst *AI = dyn_cast<AllocaInst>(V)) {
      AI->setAllocatedType(MTy->getPointerElementType());
    } else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(V)) {
      Type *MResultEltTy = mutateToHandleTy(GEP->getResultElementType());
      GEP->setResultElementType(MResultEltTy);
      Type *MSrcEltTy = mutateToHandleTy(GEP->getSourceElementType());
      GEP->setSourceElementType(MSrcEltTy);
    } else if (GEPOperator *GEPO = dyn_cast<GEPOperator>(V)) {
      // GEP operator not support setSourceElementType.
      // Create a new GEP here.
      Constant *C = cast<Constant>(GEPO->getPointerOperand());
      IRBuilder<> B(C->getContext());
      // Make sure C is mutated so the GEP get correct sourceElementType.
      C->mutateType(mutateToHandleTy(C->getType()));

      // Collect user of GEPs, then replace all use with undef.
      SmallVector<Use *, 2> Uses;
      for (Use &U : GEPO->uses()) {
        Uses.emplace_back(&U);
      }

      SmallVector<Value *, 2> idxList(GEPO->idx_begin(), GEPO->idx_end());
      Type *Ty = GEPO->getType();
      GEPO->replaceAllUsesWith(UndefValue::get(Ty));
      StringRef Name = GEPO->getName();

      // GO and newGO will be same constant except has different
      // sourceElementType. ConstantMap think they're the same constant. Have to
      // remove GO first before create newGO.
      C->removeDeadConstantUsers();

      Value *newGO = B.CreateGEP(C, idxList, Name);
      // update uses.
      for (Use *U : Uses) {
        U->set(newGO);
      }
      continue;
    }
    V->mutateType(MTy);
  }

  // Mutate functions.
  for (Function *F : CandidateFns) {
    Function *MF = nullptr;
    if (hlslOP) {
      if (hlslOP->IsDxilOpFunc(F)) {
        DXIL::OpCodeClass OpcodeClass;
        if (hlslOP->GetOpCodeClass(F, OpcodeClass)) {
          if (OpcodeClass == DXIL::OpCodeClass::CreateHandleForLib) {
            MF = createHandleForLibOnHandle;
          }
        }
      }
    }

    if (hlsl::GetHLOpcodeGroup(F) == HLOpcodeGroup::HLCast) {
      // Eliminate pass-through cast
      for (auto it = F->user_begin(); it != F->user_end();) {
        CallInst *CI = cast<CallInst>(*(it++));
        CI->replaceAllUsesWith(CI->getArgOperand(1));
        CI->eraseFromParent();
      }
      continue;
    }

    if (!MF) {
      FunctionType *FT = F->getFunctionType();
      FunctionType *MFT = cast<FunctionType>(MutateTypeMap[FT]);

      MF = Function::Create(MFT, F->getLinkage(), "", &M);
      MF->takeName(F);

      // Copy calling conv.
      MF->setCallingConv(F->getCallingConv());
      // Copy attributes.
      AttributeSet AS = F->getAttributes();
      MF->setAttributes(AS);
      // Annotation.
      if (DxilFunctionAnnotation *FnAnnot =
              pTypeSys->GetFunctionAnnotation(F)) {
        DxilFunctionAnnotation *newFnAnnot =
            pTypeSys->AddFunctionAnnotation(MF);
        DxilParameterAnnotation &RetAnnot = newFnAnnot->GetRetTypeAnnotation();
        RetAnnot = FnAnnot->GetRetTypeAnnotation();
        for (unsigned i = 0; i < FnAnnot->GetNumParameters(); i++) {
          newFnAnnot->GetParameterAnnotation(i) =
              FnAnnot->GetParameterAnnotation(i);
        }
      }
      // Update function debug info.
      if (DISubprogram *funcDI = getDISubprogram(F))
        funcDI->replaceFunction(MF);
    }

    for (auto it = F->user_begin(); it != F->user_end();) {
      CallInst *CI = cast<CallInst>(*(it++));
      CI->setCalledFunction(MF);
    }

    if (F->isDeclaration()) {
      F->eraseFromParent();
      continue;
    }
    // Take body of F.
    // Splice the body of the old function right into the new function.
    MF->getBasicBlockList().splice(MF->begin(), F->getBasicBlockList());
    // Replace use of arg.
    auto argIt = F->arg_begin();
    for (auto MArgIt = MF->arg_begin(); MArgIt != MF->arg_end();) {
      Argument *Arg = (argIt++);
      Argument *MArg = (MArgIt++);
      Arg->replaceAllUsesWith(MArg);
    }
  }
}

ModulePass *llvm::createDxilMutateResourceToHandlePass() {
  return new DxilMutateResourceToHandle();
}

INITIALIZE_PASS(DxilMutateResourceToHandle,
                "hlsl-dxil-resources-to-handle",
                "Mutate resource to handle", false, false)
