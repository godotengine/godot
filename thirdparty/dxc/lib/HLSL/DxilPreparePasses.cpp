///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilPreparePasses.cpp                                                     //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Passes to prepare DxilModule.                                             //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/HLSL/DxilGenerationPass.h"
#include "dxc/DXIL/DxilOperations.h"
#include "dxc/HLSL/HLOperations.h"
#include "dxc/DXIL/DxilModule.h"
#include "dxc/Support/Global.h"
#include "dxc/DXIL/DxilTypeSystem.h"
#include "dxc/DXIL/DxilUtil.h"
#include "dxc/DXIL/DxilEntryProps.h"
#include "dxc/DXIL/DxilFunctionProps.h"
#include "dxc/DXIL/DxilInstructions.h"
#include "dxc/DXIL/DxilConstants.h"
#include "dxc/HlslIntrinsicOp.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/PassManager.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/DxilValueCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include <memory>
#include <unordered_set>

using namespace llvm;
using namespace hlsl;

namespace {
class InvalidateUndefResources : public ModulePass {
public:
  static char ID;

  explicit InvalidateUndefResources() : ModulePass(ID) {
    initializeScalarizerPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override { return "Invalidate undef resources"; }

  bool runOnModule(Module &M) override;
};
}

char InvalidateUndefResources::ID = 0;

ModulePass *llvm::createInvalidateUndefResourcesPass() { return new InvalidateUndefResources(); }

INITIALIZE_PASS(InvalidateUndefResources, "invalidate-undef-resource", "Invalidate undef resources", false, false)

bool InvalidateUndefResources::runOnModule(Module &M) {
  // Undef resources typically indicate uninitialized locals being used
  // in some code path, which we should catch and report. However, some
  // code patterns in large shaders cause dead undef resources to momentarily,
  // which is not an error. We must wait until cleanup passes
  // have run to know whether we must produce an error.
  // However, we can't leave the undef values in because they could eliminated,
  // such as by reading from resources seen in a code path that was not taken.
  // We avoid the problem by replacing undef values by another invalid
  // value that we can identify later.
  for (auto &F : M.functions()) {
    if (GetHLOpcodeGroupByName(&F) == HLOpcodeGroup::HLCreateHandle) {
      Type *ResTy = F.getFunctionType()->getParamType(
        HLOperandIndex::kCreateHandleResourceOpIdx);
      UndefValue *UndefRes = UndefValue::get(ResTy);
      if (!UndefRes->use_empty()) {
        Constant *InvalidRes = ConstantAggregateZero::get(ResTy);
        UndefRes->replaceAllUsesWith(InvalidRes);
      }
    }
  }
  return false;
}

///////////////////////////////////////////////////////////////////////////////

namespace {
class SimplifyInst : public FunctionPass {
public:
  static char ID;

  SimplifyInst() : FunctionPass(ID) {
    initializeScalarizerPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override;

private:
};
}

char SimplifyInst::ID = 0;

FunctionPass *llvm::createSimplifyInstPass() { return new SimplifyInst(); }

INITIALIZE_PASS(SimplifyInst, "simplify-inst", "Simplify Instructions", false, false)

bool SimplifyInst::runOnFunction(Function &F) {
  for (Function::iterator BBI = F.begin(), BBE = F.end(); BBI != BBE; ++BBI) {
    BasicBlock *BB = BBI;
    llvm::SimplifyInstructionsInBlock(BB, nullptr);
  }
  return true;
}

///////////////////////////////////////////////////////////////////////////////

namespace {
class DxilDeadFunctionElimination : public ModulePass {
public:
  static char ID; // Pass identification, replacement for typeid
  explicit DxilDeadFunctionElimination () : ModulePass(ID) {}

  StringRef getPassName() const override { return "Remove all unused function except entry from DxilModule"; }

  bool runOnModule(Module &M) override {
    if (M.HasDxilModule()) {
      DxilModule &DM = M.GetDxilModule();

      bool IsLib = DM.GetShaderModel()->IsLib();
      // Remove unused functions except entry and patch constant func.
      // For library profile, only remove unused external functions.
      Function *EntryFunc = DM.GetEntryFunction();
      Function *PatchConstantFunc = DM.GetPatchConstantFunction();

      return dxilutil::RemoveUnusedFunctions(M, EntryFunc, PatchConstantFunc,
                                             IsLib);
    }

    return false;
  }
};
}

char DxilDeadFunctionElimination::ID = 0;

ModulePass *llvm::createDxilDeadFunctionEliminationPass() {
  return new DxilDeadFunctionElimination();
}

INITIALIZE_PASS(DxilDeadFunctionElimination, "dxil-dfe", "Remove all unused function except entry from DxilModule", false, false)

///////////////////////////////////////////////////////////////////////////////

bool CleanupSharedMemoryAddrSpaceCast(Module &M);

namespace {

static void TransferEntryFunctionAttributes(Function *F, Function *NewFunc) {
  // Keep necessary function attributes
  AttributeSet attributeSet = F->getAttributes();
  StringRef attrKind, attrValue;
  if (attributeSet.hasAttribute(AttributeSet::FunctionIndex, DXIL::kFP32DenormKindString)) {
    Attribute attribute = attributeSet.getAttribute(AttributeSet::FunctionIndex, DXIL::kFP32DenormKindString);
    DXASSERT(attribute.isStringAttribute(), "otherwise we have wrong fp-denorm-mode attribute.");
    attrKind = attribute.getKindAsString();
    attrValue = attribute.getValueAsString();
  }
  bool helperLane = attributeSet.hasAttribute(AttributeSet::FunctionIndex, DXIL::kWaveOpsIncludeHelperLanesString);
  if (F == NewFunc) {
    NewFunc->removeAttributes(AttributeSet::FunctionIndex, attributeSet);
  }
  if (!attrKind.empty() && !attrValue.empty())
    NewFunc->addFnAttr(attrKind, attrValue);
  if (helperLane)
    NewFunc->addFnAttr(DXIL::kWaveOpsIncludeHelperLanesString);
}

// If this returns non-null, the old function F has been stripped and can be deleted.
static Function *StripFunctionParameter(Function *F, DxilModule &DM,
    DenseMap<const Function *, DISubprogram *> &FunctionDIs) {
  if (F->arg_empty() && F->getReturnType()->isVoidTy()) {
    // This will strip non-entry function attributes
    TransferEntryFunctionAttributes(F, F);
    return nullptr;
  }

  Module &M = *DM.GetModule();
  Type *VoidTy = Type::getVoidTy(M.getContext());
  FunctionType *FT = FunctionType::get(VoidTy, false);
  for (auto &arg : F->args()) {
    if (!arg.user_empty())
      return nullptr;
    DbgDeclareInst *DDI = llvm::FindAllocaDbgDeclare(&arg);
    if (DDI) {
      DDI->eraseFromParent();
    }
  }

  Function *NewFunc = Function::Create(FT, F->getLinkage());
  M.getFunctionList().insert(F, NewFunc);
  // Splice the body of the old function right into the new function.
  NewFunc->getBasicBlockList().splice(NewFunc->begin(), F->getBasicBlockList());

  TransferEntryFunctionAttributes(F, NewFunc);

  // Patch the pointer to LLVM function in debug info descriptor.
  auto DI = FunctionDIs.find(F);
  if (DI != FunctionDIs.end()) {
    DISubprogram *SP = DI->second;
    SP->replaceFunction(NewFunc);
    // Ensure the map is updated so it can be reused on subsequent argument
    // promotions of the same function.
    FunctionDIs.erase(DI);
    FunctionDIs[NewFunc] = SP;
  }
  NewFunc->takeName(F);
  if (DM.HasDxilFunctionProps(F)) {
    DM.ReplaceDxilEntryProps(F, NewFunc);
  }
  DM.GetTypeSystem().EraseFunctionAnnotation(F);
  DM.GetTypeSystem().AddFunctionAnnotation(NewFunc);
  return NewFunc;
}

void CheckInBoundForTGSM(GlobalVariable &GV, const DataLayout &DL) {
  for (User *U : GV.users()) {
    if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(U)) {
      bool allImmIndex = true;
      for (auto Idx = GEP->idx_begin(), E = GEP->idx_end(); Idx != E; Idx++) {
        if (!isa<ConstantInt>(Idx)) {
          allImmIndex = false;
          break;
        }
      }
      if (!allImmIndex)
        GEP->setIsInBounds(false);
      else {
        Value *Ptr = GEP->getPointerOperand();
        unsigned size =
            DL.getTypeAllocSize(Ptr->getType()->getPointerElementType());
        unsigned valSize =
            DL.getTypeAllocSize(GEP->getType()->getPointerElementType());
        SmallVector<Value *, 8> Indices(GEP->idx_begin(), GEP->idx_end());
        unsigned offset =
            DL.getIndexedOffset(GEP->getPointerOperandType(), Indices);
        if ((offset + valSize) > size)
          GEP->setIsInBounds(false);
      }
    }
  }
}

static bool GetUnsignedVal(Value *V, uint32_t *pValue) {
  ConstantInt *CI = dyn_cast<ConstantInt>(V);
  if (!CI) return false;
  uint64_t u = CI->getZExtValue();
  if (u > UINT32_MAX) return false;
  *pValue = (uint32_t)u;
  return true;
}

static void MarkUsedSignatureElements(Function *F, DxilModule &DM) {
  DXASSERT_NOMSG(F != nullptr);
  // For every loadInput/storeOutput, update the corresponding ReadWriteMask.
  // F is a pointer to a Function instance
  for (llvm::inst_iterator I = llvm::inst_begin(F), E = llvm::inst_end(F); I != E; ++I) {
    DxilInst_LoadInput LI(&*I);
    DxilInst_StoreOutput SO(&*I);
    DxilInst_LoadPatchConstant LPC(&*I);
    DxilInst_StorePatchConstant SPC(&*I);
    DxilInst_StoreVertexOutput SVO(&*I);
    DxilInst_StorePrimitiveOutput SPO(&*I);
    DxilSignature *pSig;
    uint32_t col, row, sigId;
    bool bDynIdx = false;
    if (LI) {
      if (!GetUnsignedVal(LI.get_inputSigId(), &sigId)) continue;
      if (!GetUnsignedVal(LI.get_colIndex(), &col)) continue;
      if (!GetUnsignedVal(LI.get_rowIndex(), &row)) bDynIdx = true;
      pSig = &DM.GetInputSignature();
    }
    else if (SO) {
      if (!GetUnsignedVal(SO.get_outputSigId(), &sigId)) continue;
      if (!GetUnsignedVal(SO.get_colIndex(), &col)) continue;
      if (!GetUnsignedVal(SO.get_rowIndex(), &row)) bDynIdx = true;
      pSig = &DM.GetOutputSignature();
    }
    else if (SPC) {
      if (!GetUnsignedVal(SPC.get_outputSigID(), &sigId)) continue;
      if (!GetUnsignedVal(SPC.get_col(), &col)) continue;
      if (!GetUnsignedVal(SPC.get_row(), &row)) bDynIdx = true;
      pSig = &DM.GetPatchConstOrPrimSignature();
    }
    else if (LPC) {
      if (!GetUnsignedVal(LPC.get_inputSigId(), &sigId)) continue;
      if (!GetUnsignedVal(LPC.get_col(), &col)) continue;
      if (!GetUnsignedVal(LPC.get_row(), &row)) bDynIdx = true;
      pSig = &DM.GetPatchConstOrPrimSignature();
    }
    else if (SVO) {
      if (!GetUnsignedVal(SVO.get_outputSigId(), &sigId)) continue;
      if (!GetUnsignedVal(SVO.get_colIndex(), &col)) continue;
      if (!GetUnsignedVal(SVO.get_rowIndex(), &row)) bDynIdx = true;
      pSig = &DM.GetOutputSignature();
    }
    else if (SPO) {
      if (!GetUnsignedVal(SPO.get_outputSigId(), &sigId)) continue;
      if (!GetUnsignedVal(SPO.get_colIndex(), &col)) continue;
      if (!GetUnsignedVal(SPO.get_rowIndex(), &row)) bDynIdx = true;
      pSig = &DM.GetPatchConstOrPrimSignature();
    }
    else {
      continue;
    }

    // Consider being more fine-grained about masks.
    // We report sometimes-read on input as always-read.
    auto &El = pSig->GetElement(sigId);
    unsigned UsageMask = El.GetUsageMask();
    unsigned colBit = 1 << col;
    if (!(colBit & UsageMask)) {
      El.SetUsageMask(UsageMask | colBit);
    }
    if (bDynIdx && (El.GetDynIdxCompMask() & colBit) == 0) {
      El.SetDynIdxCompMask(El.GetDynIdxCompMask() | colBit);
    }
  }
}

class DxilFinalizeModule : public ModulePass {
public:
  static char ID; // Pass identification, replacement for typeid
  explicit DxilFinalizeModule() : ModulePass(ID) {}

  StringRef getPassName() const override { return "HLSL DXIL Finalize Module"; }

  void patchValidation_1_1(Module &M) {
    for (iplist<Function>::iterator F : M.getFunctionList()) {
      for (Function::iterator BBI = F->begin(), BBE = F->end(); BBI != BBE;
           ++BBI) {
        BasicBlock *BB = BBI;
        for (BasicBlock::iterator II = BB->begin(), IE = BB->end(); II != IE;
             ++II) {
          Instruction *I = II;
          if (I->hasMetadataOtherThanDebugLoc()) {
            SmallVector<std::pair<unsigned, MDNode*>, 2> MDs;
            I->getAllMetadataOtherThanDebugLoc(MDs);
            for (auto &MD : MDs) {
              unsigned kind = MD.first;
              // Remove Metadata which validation_1_0 not allowed.
              bool bNeedPatch = kind == LLVMContext::MD_tbaa ||
                  kind == LLVMContext::MD_prof ||
                  (kind > LLVMContext::MD_fpmath &&
                  kind <= LLVMContext::MD_dereferenceable_or_null);
              if (bNeedPatch)
                I->setMetadata(kind, nullptr);
            }
          }
        }
      }
    }
  }

  void RemoveAnnotateHandle(hlsl::OP *hlslOP) {
    for (auto it : hlslOP->GetOpFuncList(DXIL::OpCode::AnnotateHandle)) {
      Function *F = it.second;
      if (!F)
        continue;
      for (auto uit = F->user_begin(); uit != F->user_end();) {
        CallInst *CI = cast<CallInst>(*(uit++));
        DxilInst_AnnotateHandle annoteHdl(CI);
        Value *hdl = annoteHdl.get_res();
        CI->replaceAllUsesWith(hdl);
        CI->eraseFromParent();
      }
    }
  }

  ///////////////////////////////////////////////////
  // IsHelperLane() lowering for SM < 6.6

  // Identify pattern icmp_eq(0, dx.coverage())
  bool IsCmpZOfCoverage(Value *V, hlsl::OP *hlslOP) {
    if (ICmpInst *IC = dyn_cast<ICmpInst>(V)) {
      if (IC->getPredicate() == ICmpInst::ICMP_EQ) {
        Value *V0 = IC->getOperand(0);
        Value *V1 = IC->getOperand(1);
        if (!isa<ConstantInt>(V0))
          std::swap(V0, V1);
        if (ConstantInt *C = dyn_cast<ConstantInt>(V0)) {
          if (CallInst *CI = dyn_cast<CallInst>(V1)) {
            // compare dx.op.coverage with zero
            if (C->isZero() &&
                hlslOP->IsDxilOpFuncCallInst(CI, DXIL::OpCode::Coverage)) {
              return true;
            }
          }
        }
      }
    }
    return false;
  }

  // Identify init as use in entry block that either:
  //  - non-PS: store i32 0
  //  - PS: store zext(icmp_eq(0, dx.coverage()))
  bool IsInitOfIsHelperGV(User *U, hlsl::OP *hlslOP) {
    if (StoreInst *SI = dyn_cast<StoreInst>(U)) {
      BasicBlock *BB = SI->getParent();
      if (BB == &BB->getParent()->getEntryBlock()) {
        Value *V = SI->getValueOperand();
        if (ConstantInt *C = dyn_cast<ConstantInt>(V)) {
          if (C->isZero()) {
            return true;
          }
        } else if (ZExtInst *ZEI = dyn_cast<ZExtInst>(V)) {
          if (IsCmpZOfCoverage(ZEI->getOperand(0), hlslOP)) {
            return true;
          }
        }
      }
    }
    return false;
  }

  void RemoveFnIfIsHelperInit(User *U, hlsl::OP *hlslOP,
                              SmallSetVector<Function *, 4> &psEntries) {
    if (Instruction *I = dyn_cast<Instruction>(U)) {
      // Early out: only check if in function still in set
      Function *F = I->getParent()->getParent();
      if (!psEntries.count(F))
        return;
      if (IsInitOfIsHelperGV(I, hlslOP)) {
        psEntries.remove(F);
      }
    }
  }

  // Init IsHelper GV to zext(!dx.op.coverage()) in PS entry points
  void InitIsHelperGV(Module &M) {
    GlobalVariable *GV =
        M.getGlobalVariable(DXIL::kDxIsHelperGlobalName, /*AllowLocal*/ true);
    if (!GV)
      return;

    DxilModule &DM = M.GetDxilModule();
    hlsl::OP *hlslOP = DM.GetOP();
    const ShaderModel *pSM = DM.GetShaderModel();

    // If PS, and GV is ExternalLinkage, change to InternalLinkage
    // This can happen after link to final PS.
    if (pSM->IsPS() && GV->getLinkage() == GlobalValue::ExternalLinkage) {
      GV->setLinkage(GlobalValue::InternalLinkage);
    }

    // add PS entry points to set
    SmallSetVector<Function*, 4> psEntries;
    if (pSM->IsPS()) {
      psEntries.insert(DM.GetEntryFunction());
    } else if (pSM->IsLib()) {
      for (auto &F : M.functions()) {
        if (DM.HasDxilEntryProps(&F)) {
          if (DM.GetDxilEntryProps(&F).props.IsPS()) {
            psEntries.insert(&F);
          }
        }
      }
    }

    // iterate users of GV to skip entries that already init GV
    for (auto &U : GV->uses()) {
      RemoveFnIfIsHelperInit(U.getUser(), DM.GetOP(), psEntries);
    }

    // store zext(!dx.op.coverage())
    Type *I32Ty = Type::getInt32Ty(hlslOP->GetCtx());
    Constant *C0 = hlslOP->GetI32Const(0);
    Constant *OpArg = hlslOP->GetI32Const((int)DXIL::OpCode::Coverage);
    Function *CoverageF = nullptr;
    for (auto *F : psEntries) {
      if (!CoverageF)
        CoverageF = hlslOP->GetOpFunc(DXIL::OpCode::Coverage, I32Ty);
      IRBuilder<> Builder(F->getEntryBlock().getFirstInsertionPt());
      Value *V = Builder.CreateCall(CoverageF, {OpArg});
      V = Builder.CreateICmpEQ(C0, V);
      V = Builder.CreateZExt(V, I32Ty);
      Builder.CreateStore(V, GV);
    }
  }

  GlobalVariable *GetIsHelperGV(Module &M) {
    return M.getGlobalVariable(DXIL::kDxIsHelperGlobalName, /*AllowLocal*/ true);
  }
  GlobalVariable *GetOrCreateIsHelperGV(Module &M, hlsl::OP *hlslOP) {
    GlobalVariable *GV = GetIsHelperGV(M);
    if (GV)
      return GV;
    DxilModule &DM = M.GetDxilModule();
    const ShaderModel *pSM = DM.GetShaderModel();
    GV = new GlobalVariable(M, IntegerType::get(M.getContext(), 32),
                            /*constant*/ false,
                            pSM->IsLib() ? GlobalValue::ExternalLinkage
                                         : GlobalValue::InternalLinkage,
                            /*Initializer*/ hlslOP->GetI32Const(0),
                            DXIL::kDxIsHelperGlobalName);
    return GV;
  }

  // Replace IsHelperLane() with false (for non-lib, non-PS SM)
  void ReplaceIsHelperWithConstFalse(hlsl::OP *hlslOP) {
    Constant *False = hlslOP->GetI1Const(0);
    bool bDone = false;
    while (!bDone) {
      bDone = true;
      for (auto it : hlslOP->GetOpFuncList(DXIL::OpCode::IsHelperLane)) {
        Function *F = it.second;
        if (!F)
          continue;
        for (auto uit = F->user_begin(); uit != F->user_end();) {
          CallInst *CI = dyn_cast<CallInst>(*(uit++));
          CI->replaceAllUsesWith(False);
          CI->eraseFromParent();
        }
        hlslOP->RemoveFunction(F);
        F->eraseFromParent();
        bDone = false;
        break;
      }
    }
  }

  void ConvertIsHelperToLoadGV(hlsl::OP *hlslOP) {
    GlobalVariable *GV = nullptr;
    Type *I1Ty = Type::getInt1Ty(hlslOP->GetCtx());
    bool bDone = false;
    while (!bDone) {
      bDone = true;
      for (auto it : hlslOP->GetOpFuncList(DXIL::OpCode::IsHelperLane)) {
        Function *F = it.second;
        if (!F)
          continue;
        for (auto uit = F->user_begin(); uit != F->user_end();) {
          CallInst *CI = cast<CallInst>(*(uit++));
          if (!GV)
            GV = GetOrCreateIsHelperGV(*F->getParent(), hlslOP);
          IRBuilder<> Builder(CI);
          Value *V = Builder.CreateLoad(GV);
          V = Builder.CreateTrunc(V, I1Ty);
          CI->replaceAllUsesWith(V);
          CI->eraseFromParent();
        }
        hlslOP->RemoveFunction(F);
        F->eraseFromParent();
        bDone = false;
        break;
      }
    }
  }

  void ConvertDiscardToStoreGV(hlsl::OP *hlslOP) {
    GlobalVariable *GV = nullptr;
    Type *I32Ty = Type::getInt32Ty(hlslOP->GetCtx());
    for (auto it : hlslOP->GetOpFuncList(DXIL::OpCode::Discard)) {
      Function *F = it.second;
      if (!F)
        continue;
      for (auto uit = F->user_begin(); uit != F->user_end();) {
        CallInst *CI = cast<CallInst>(*(uit++));
        if (!GV)
          GV = GetIsHelperGV(*F->getParent());
        // If we don't already have a global for this,
        // we didn't have any IsHelper() calls, so no need to add one now.
        if (!GV)
          return;
        IRBuilder<> Builder(CI);
        Value *Cond =
            Builder.CreateZExt(DxilInst_Discard(CI).get_condition(), I32Ty);
        Builder.CreateStore(Cond, GV);
      }
    }
  }
  ///////////////////////////////////////////////////

  void patchDxil_1_6(Module &M, hlsl::OP *hlslOP, unsigned ValMajor, unsigned ValMinor) {
    RemoveAnnotateHandle(hlslOP);

    // Convert IsHelperLane() on down-level targets
    const ShaderModel *pSM = M.GetDxilModule().GetShaderModel();
    if (pSM->IsLib() || pSM->IsPS()) {
      ConvertIsHelperToLoadGV(hlslOP);
      ConvertDiscardToStoreGV(hlslOP);
      InitIsHelperGV(M);

      // Set linkage of dx.ishelper to internal for validator version < 1.6
      // This means IsHelperLane() fallback code will not return correct result
      // in an exported function linked to a PS in another library in this case.
      // But it won't pass validation otherwise.
      if (pSM->IsLib() && DXIL::CompareVersions(ValMajor, ValMinor, 1, 6) < 1) {
        if (GlobalVariable *GV = GetIsHelperGV(M)) {
          GV->setLinkage(GlobalValue::InternalLinkage);
        }
      }
    } else {
      ReplaceIsHelperWithConstFalse(hlslOP);
    }
  }

  void convertQuadVote(Module &M, hlsl::OP *hlslOP) {
    for (auto FnIt : hlslOP->GetOpFuncList(DXIL::OpCode::QuadVote)) {
      Function *F = FnIt.second;
      if (!F)
        continue;
      for (auto UserIt = F->user_begin(); UserIt != F->user_end();) {
        CallInst *CI = cast<CallInst>(*(UserIt++));

        IRBuilder<> B(CI);
        DXASSERT_NOMSG(CI->getOperand(1)->getType() ==
                       Type::getInt1Ty(M.getContext()));

        Type *i32Ty = Type::getInt32Ty(M.getContext());
        Value *Cond = B.CreateSExt(CI->getOperand(1), i32Ty);

        Function *QuadOpFn = hlslOP->GetOpFunc(DXIL::OpCode::QuadOp, i32Ty);
        const std::string &OpName = hlslOP->GetOpCodeName(DXIL::OpCode::QuadOp);

        Value *refArgs[] = {hlslOP->GetU32Const((unsigned)DXIL::OpCode::QuadOp),
                            Cond, nullptr};
        refArgs[2] =
            hlslOP->GetI8Const((unsigned)DXIL::QuadOpKind::ReadAcrossX);
        Value *X = B.CreateCall(QuadOpFn, refArgs, OpName);
        refArgs[2] =
            hlslOP->GetI8Const((unsigned)DXIL::QuadOpKind::ReadAcrossY);
        Value *Y = B.CreateCall(QuadOpFn, refArgs, OpName);
        refArgs[2] =
            hlslOP->GetI8Const((unsigned)DXIL::QuadOpKind::ReadAcrossDiagonal);
        Value *Z = B.CreateCall(QuadOpFn, refArgs, OpName);
        Value *Result = nullptr;

        uint64_t OpKind = cast<ConstantInt>(CI->getOperand(2))->getZExtValue();

        if (OpKind == (uint64_t)DXIL::QuadVoteOpKind::All) {
          Value *XY = B.CreateAnd(X, Y);
          Value *XYZ = B.CreateAnd(XY, Z);
          Result = B.CreateAnd(XYZ, Cond);
        } else {
          DXASSERT_NOMSG(OpKind == (uint64_t)DXIL::QuadVoteOpKind::Any);
          Value *XY = B.CreateOr(X, Y);
          Value *XYZ = B.CreateOr(XY, Z);
          Result = B.CreateOr(XYZ, Cond);
        }
        Value *Res = B.CreateTrunc(Result, Type::getInt1Ty(M.getContext()));
        CI->replaceAllUsesWith(Res);
        CI->eraseFromParent();
      }
    }
  }

  // Replace llvm.lifetime.start/.end intrinsics with undef or zeroinitializer
  // stores (for earlier validator versions) unless the pointer is a global
  // that has an initializer.
  // This works around losing scoping information in earlier shader models
  // that do not support the intrinsics natively.
  void patchLifetimeIntrinsics(Module &M, unsigned ValMajor, unsigned ValMinor, bool forceZeroStoreLifetimes) {
    // Get the declarations. This may introduce them if there were none before.
    Value *StartDecl = Intrinsic::getDeclaration(&M, Intrinsic::lifetime_start);
    Value *EndDecl   = Intrinsic::getDeclaration(&M, Intrinsic::lifetime_end);

    // Collect all calls to both intrinsics.
    std::vector<CallInst*> intrinsicCalls;
    for (Use &U : StartDecl->uses()) {
      // All users must be call instructions.
      CallInst *CI = dyn_cast<CallInst>(U.getUser());
      DXASSERT(CI,
               "Expected user of lifetime.start intrinsic to be a CallInst");
      intrinsicCalls.push_back(CI);
    }
    for (Use &U : EndDecl->uses()) {
      // All users must be call instructions.
      CallInst *CI = dyn_cast<CallInst>(U.getUser());
      DXASSERT(CI, "Expected user of lifetime.end intrinsic to be a CallInst");
      intrinsicCalls.push_back(CI);
    }

    // Replace each intrinsic with an undef store.
    for (CallInst *CI : intrinsicCalls) {
      // Find the corresponding pointer (bitcast from alloca, global value, an
      // argument, ...).
      Value *voidPtr = CI->getArgOperand(1);
      DXASSERT(voidPtr->getType()->isPointerTy() &&
               voidPtr->getType()->getPointerElementType()->isIntegerTy(8),
               "Expected operand of lifetime intrinsic to be of type i8*" );

      Value *ptr = nullptr;
      if (ConstantExpr *CE = dyn_cast<ConstantExpr>(voidPtr)) {
        // This can happen if a local variable/array is promoted to a constant
        // global. In this case we must not introduce a store, since that would
        // overwrite the constant values in the initializer. Thus, we simply
        // remove the intrinsic.
        DXASSERT(CE->getOpcode() == Instruction::BitCast,
                 "expected operand of lifetime intrinsic to be a bitcast");
      } else {
        // Otherwise, it must be a normal bitcast.
        DXASSERT(isa<BitCastInst>(voidPtr),
                 "Expected operand of lifetime intrinsic to be a bitcast");
        BitCastInst *BC = cast<BitCastInst>(voidPtr);
        ptr = BC->getOperand(0);

        // If the original pointer is a global with initializer, do not replace
        // the intrinsic with a store.
        if (GlobalVariable *GV = dyn_cast<GlobalVariable>(ptr))
          if (GV->hasInitializer() || GV->isExternallyInitialized())
            ptr = nullptr;
      }

      if (ptr) {
        // Determine the type to use when storing undef.
        DXASSERT(ptr->getType()->isPointerTy(),
                 "Expected type of operand of lifetime intrinsic bitcast operand to be a pointer");
        Type *T = ptr->getType()->getPointerElementType();

        // Store undef at the location of the start/end intrinsic.
        // If we are targeting validator version < 6.6 we cannot store undef
        // since it causes a validation error. As a workaround we store 0, which
        // achieves mostly the same as storing undef but can cause overhead in
        // some situations.
        // We also allow to force zeroinitializer through a flag.
        if (forceZeroStoreLifetimes || ValMajor < 1 || (ValMajor == 1 && ValMinor < 6))
          IRBuilder<>(CI).CreateStore(Constant::getNullValue(T), ptr);
        else
          IRBuilder<>(CI).CreateStore(UndefValue::get(T), ptr);
      }

      // Erase the intrinsic call and, if it has no uses anymore, the bitcast as
      // well.
      DXASSERT_NOMSG(CI->use_empty());
      CI->eraseFromParent();

      // Erase the bitcast inst if it is not a ConstantExpr.
      if (BitCastInst *BC = dyn_cast<BitCastInst>(voidPtr))
        if (BC->use_empty())
          BC->eraseFromParent();
    }

    // Erase the intrinsic declarations.
    DXASSERT_NOMSG(StartDecl->use_empty());
    DXASSERT_NOMSG(EndDecl->use_empty());
    cast<Function>(StartDecl)->eraseFromParent();
    cast<Function>(EndDecl)->eraseFromParent();
  }

  bool runOnModule(Module &M) override {
    if (M.HasDxilModule()) {
      DxilModule &DM = M.GetDxilModule();
      unsigned ValMajor = 0;
      unsigned ValMinor = 0;
      DM.GetValidatorVersion(ValMajor, ValMinor);
      unsigned DxilMajor = 0;
      unsigned DxilMinor = 0;
      DM.GetDxilVersion(DxilMajor, DxilMinor);

      bool IsLib = DM.GetShaderModel()->IsLib();
      // Skip validation patch for lib.
      if (!IsLib) {
        if (DXIL::CompareVersions(ValMajor, ValMinor, 1, 1) <= 0) {
          patchValidation_1_1(M);
        }
      }

      // Replace lifetime intrinsics if requested or necessary.
      const bool forceZeroStoreLifetimes = DM.GetForceZeroStoreLifetimes();
      if (forceZeroStoreLifetimes ||
          DXIL::CompareVersions(DxilMajor, DxilMinor, 1, 6) < 0) {
        patchLifetimeIntrinsics(M, ValMajor, ValMinor, forceZeroStoreLifetimes);
      }

      hlsl::OP *hlslOP = DM.GetOP();
      // Basic down-conversions for Dxil < 1.6
      if (DXIL::CompareVersions(DxilMajor, DxilMinor, 1, 6) < 0) {
        patchDxil_1_6(M, hlslOP, ValMajor, ValMinor);
      }

      // Convert quad vote
      if (DXIL::CompareVersions(DxilMajor, DxilMinor, 1, 7) < 0) {
        convertQuadVote(M, DM.GetOP());
      }

      // Remove store undef output.
      RemoveStoreUndefOutput(M, hlslOP);

      if (!IsLib) {
        // Set used masks for signature elements
        MarkUsedSignatureElements(DM.GetEntryFunction(), DM);
        if (DM.GetShaderModel()->IsHS())
          MarkUsedSignatureElements(DM.GetPatchConstantFunction(), DM);
      }

      // Adding warning for pixel shader with unassigned target
      if (DM.GetShaderModel()->IsPS()) {
        DxilSignature &sig = DM.GetOutputSignature();
        for (auto &Elt : sig.GetElements()) {
          if (Elt->GetKind() == Semantic::Kind::Target &&
              Elt->GetUsageMask() != Elt->GetColsAsMask()) {
            dxilutil::EmitWarningOnContext(
                M.getContext(),
                "Declared output " + llvm::Twine(Elt->GetName()) +
                    llvm::Twine(Elt->GetSemanticStartIndex()) +
                    " not fully written in shader.");
          }
        }
      }

      // Turn dx.break() conditional into global
      LowerDxBreak(M);

      RemoveUnusedStaticGlobal(M);

      // Remove unnecessary address space casts.
      CleanupSharedMemoryAddrSpaceCast(M);

      // Clear inbound for GEP which has none-const index.
      LegalizeSharedMemoryGEPInbound(M);

      // Strip parameters of entry function.
      StripEntryParameters(M, DM, IsLib);

      // Update flags to reflect any changes.
      DM.CollectShaderFlagsForModule();

      // Update Validator Version
      DM.UpgradeToMinValidatorVersion();

      // Clear intermediate options that shouldn't be in the final DXIL
      DM.ClearIntermediateOptions();

      // Remove unused AllocateRayQuery calls
      RemoveUnusedRayQuery(M);

      if (IsLib && DXIL::CompareVersions(ValMajor, ValMinor, 1, 4) <= 0) {
        // 1.4 validator requires function annotations for all functions
        AddFunctionAnnotationForInitializers(M, DM);
      }

      // Fix DIExpression fragments that cover whole variables
      LegalizeDbgFragments(M);

      return true;
    }

    return false;
  }

private:
  void RemoveUnusedStaticGlobal(Module &M) {
    // Remove unused internal global.
    std::vector<GlobalVariable *> staticGVs;
    for (GlobalVariable &GV : M.globals()) {
      if (dxilutil::IsStaticGlobal(&GV) ||
          dxilutil::IsSharedMemoryGlobal(&GV)) {
        staticGVs.emplace_back(&GV);
      }
    }

    for (GlobalVariable *GV : staticGVs) {
      bool onlyStoreUse = true;
      for (User *user : GV->users()) {
        if (isa<StoreInst>(user))
          continue;
        if (isa<ConstantExpr>(user) && user->user_empty())
          continue;
        onlyStoreUse = false;
        break;
      }
      if (onlyStoreUse) {
        for (auto UserIt = GV->user_begin(); UserIt != GV->user_end();) {
          Value *User = *(UserIt++);
          if (Instruction *I = dyn_cast<Instruction>(User)) {
            I->eraseFromParent();
          } else {
            ConstantExpr *CE = cast<ConstantExpr>(User);
            CE->dropAllReferences();
          }
        }
        GV->eraseFromParent();
      }
    }
  }

  static bool BitPieceCoversEntireVar(DIExpression *expr, DILocalVariable *var, DITypeIdentifierMap &TypeIdentifierMap) {
    if (expr->isBitPiece()) {
      DIType *ty = var->getType().resolve(TypeIdentifierMap);
      return expr->getBitPieceOffset() == 0 && expr->getBitPieceSize() == ty->getSizeInBits();
    }
    return false;
  }

  static void LegalizeDbgFragmentsForDbgIntrinsic(Function *f, DITypeIdentifierMap &TypeIdentifierMap) {
    Intrinsic::ID intrinsic = f->getIntrinsicID();

    DIBuilder dib(*f->getParent());
    if (intrinsic == Intrinsic::dbg_value) {
      for (auto it = f->user_begin(), end = f->user_end(); it != end;) {
        User *u = *(it++);
        DbgValueInst *di = cast<DbgValueInst>(u);
        Value *value = di->getValue();
        if (!value) {
          di->eraseFromParent();
          continue;
        }
        DIExpression *expr = di->getExpression();
        DILocalVariable *var = di->getVariable();
        if (BitPieceCoversEntireVar(expr, var, TypeIdentifierMap)) {
          dib.insertDbgValueIntrinsic(value, 0, var, DIExpression::get(di->getContext(), {}), di->getDebugLoc(), di);
          di->eraseFromParent();
        }
      }
    }
    else if (intrinsic == Intrinsic::dbg_declare) {
      for (auto it = f->user_begin(), end = f->user_end(); it != end;) {
        User *u = *(it++);
        DbgDeclareInst *di = cast<DbgDeclareInst>(u);
        Value *addr = di->getAddress();
        if (!addr) {
          di->eraseFromParent();
          continue;
        }
        DIExpression *expr = di->getExpression();
        DILocalVariable *var = di->getVariable();
        if (BitPieceCoversEntireVar(expr, var, TypeIdentifierMap)) {
          dib.insertDeclare(addr, var, DIExpression::get(di->getContext(), {}), di->getDebugLoc(), di);
          di->eraseFromParent();
        }
      }
    }
  }

  static void LegalizeDbgFragments(Module &M) {
    DITypeIdentifierMap TypeIdentifierMap;

    if (Function *f = M.getFunction(Intrinsic::getName(Intrinsic::dbg_value))) {
      LegalizeDbgFragmentsForDbgIntrinsic(f, TypeIdentifierMap);
    }
    if (Function *f = M.getFunction(Intrinsic::getName(Intrinsic::dbg_declare))) {
      LegalizeDbgFragmentsForDbgIntrinsic(f, TypeIdentifierMap);
    }
  }

  void RemoveStoreUndefOutput(Module &M, hlsl::OP *hlslOP) {
    for (iplist<Function>::iterator F : M.getFunctionList()) {
      if (!hlslOP->IsDxilOpFunc(F))
        continue;
      DXIL::OpCodeClass opClass;
      bool bHasOpClass = hlslOP->GetOpCodeClass(F, opClass);
      DXASSERT_LOCALVAR(bHasOpClass, bHasOpClass, "else not a dxil op func");
      if (opClass != DXIL::OpCodeClass::StoreOutput)
        continue;

      for (auto it = F->user_begin(); it != F->user_end();) {
        CallInst *CI = dyn_cast<CallInst>(*(it++));
        if (!CI)
          continue;

        Value *V = CI->getArgOperand(DXIL::OperandIndex::kStoreOutputValOpIdx);
        // Remove the store of undef.
        if (isa<UndefValue>(V))
          CI->eraseFromParent();
      }
    }
  }

  void LegalizeSharedMemoryGEPInbound(Module &M) {
    const DataLayout &DL = M.getDataLayout();
    // Clear inbound for GEP which has none-const index.
    for (GlobalVariable &GV : M.globals()) {
      if (dxilutil::IsSharedMemoryGlobal(&GV)) {
        CheckInBoundForTGSM(GV, DL);
      }
    }
  }

  void StripEntryParameters(Module &M, DxilModule &DM, bool IsLib) {
    DenseMap<const Function *, DISubprogram *> FunctionDIs =
        makeSubprogramMap(M);
    // Strip parameters of entry function.
    if (!IsLib) {
      if (Function *OldPatchConstantFunc = DM.GetPatchConstantFunction()) {
        Function *NewPatchConstantFunc =
            StripFunctionParameter(OldPatchConstantFunc, DM, FunctionDIs);
        if (NewPatchConstantFunc) {
          DM.SetPatchConstantFunction(NewPatchConstantFunc);

          // Erase once the DxilModule doesn't track the old function anymore
          DXASSERT(DM.IsPatchConstantShader(NewPatchConstantFunc) && !DM.IsPatchConstantShader(OldPatchConstantFunc),
            "Error while migrating to parameter-stripped patch constant function.");
          OldPatchConstantFunc->eraseFromParent();
        }
      }

      if (Function *OldEntryFunc = DM.GetEntryFunction()) {
        StringRef Name = DM.GetEntryFunctionName();
        OldEntryFunc->setName(Name);
        Function *NewEntryFunc = StripFunctionParameter(OldEntryFunc, DM, FunctionDIs);
        if (NewEntryFunc) {
          DM.SetEntryFunction(NewEntryFunc);
          OldEntryFunc->eraseFromParent();
        }
      }
    } else {
      std::vector<Function *> entries;
      // Handle when multiple hull shaders point to the same patch constant function
      MapVector<Function*, llvm::SmallVector<Function*, 2>> PatchConstantFuncUsers;
      for (iplist<Function>::iterator F : M.getFunctionList()) {
        if (DM.IsEntryThatUsesSignatures(F)) {
          auto *FT = F->getFunctionType();
          // Only do this when has parameters.
          if (FT->getNumParams() > 0 || !FT->getReturnType()->isVoidTy()) {
            entries.emplace_back(F);
          }

          DxilFunctionProps& props = DM.GetDxilFunctionProps(F);
          if (props.IsHS() && props.ShaderProps.HS.patchConstantFunc) {
            FunctionType* PatchConstantFuncTy = props.ShaderProps.HS.patchConstantFunc->getFunctionType();
            if (PatchConstantFuncTy->getNumParams() > 0 || !PatchConstantFuncTy->getReturnType()->isVoidTy()) {
              // Accumulate all hull shaders using a given patch constant function,
              // so we can update it once and fix all hull shaders, without having an intermediary
              // state where some hull shaders point to a destroyed patch constant function.
              PatchConstantFuncUsers[props.ShaderProps.HS.patchConstantFunc].emplace_back(F);
            }
          }
        }
      }

      // Strip patch constant functions first
      for (auto &PatchConstantFuncEntry : PatchConstantFuncUsers) {
        Function* OldPatchConstantFunc = PatchConstantFuncEntry.first;
        Function* NewPatchConstantFunc = StripFunctionParameter(OldPatchConstantFunc, DM, FunctionDIs);
        if (NewPatchConstantFunc) {
          // Update all user hull shaders
          for (Function *HullShaderFunc : PatchConstantFuncEntry.second)
            DM.SetPatchConstantFunctionForHS(HullShaderFunc, NewPatchConstantFunc);

          // Erase once the DxilModule doesn't track the old function anymore
          DXASSERT(DM.IsPatchConstantShader(NewPatchConstantFunc) && !DM.IsPatchConstantShader(OldPatchConstantFunc),
            "Error while migrating to parameter-stripped patch constant function.");
          OldPatchConstantFunc->eraseFromParent();
        }
      }

      for (Function *OldEntry : entries) {
        Function *NewEntry = StripFunctionParameter(OldEntry, DM, FunctionDIs);
        if (NewEntry) OldEntry->eraseFromParent();
      }
    }
  }

  void AddFunctionAnnotationForInitializers(Module &M, DxilModule &DM) {
    if (GlobalVariable *GV = M.getGlobalVariable("llvm.global_ctors")) {
      if (isa<ConstantAggregateZero>(GV->getInitializer())) {
        DXASSERT_NOMSG(GV->user_empty());
        GV->eraseFromParent();
        return;
      }
      ConstantArray *init = cast<ConstantArray>(GV->getInitializer());
      for (auto V : init->operand_values()) {
        if (isa<ConstantAggregateZero>(V))
          continue;
        ConstantStruct *CS = cast<ConstantStruct>(V);
        if (isa<ConstantPointerNull>(CS->getOperand(1)))
          continue;
        Function *F = cast<Function>(CS->getOperand(1));
        if (DM.GetTypeSystem().GetFunctionAnnotation(F) == nullptr)
          DM.GetTypeSystem().AddFunctionAnnotation(F);
      }
    }
  }

  void RemoveUnusedRayQuery(Module &M) {
    hlsl::OP *hlslOP = M.GetDxilModule().GetOP();
    llvm::Function *AllocFn = hlslOP->GetOpFunc(
      DXIL::OpCode::AllocateRayQuery, Type::getVoidTy(M.getContext()));
    SmallVector<CallInst*, 4> DeadInsts;
    for (auto U : AllocFn->users()) {
      if (CallInst *CI = dyn_cast<CallInst>(U)) {
        if (CI->user_empty()) {
          DeadInsts.emplace_back(CI);
        }
      }
    }
    for (auto CI : DeadInsts) {
      CI->eraseFromParent();
    }
    if (AllocFn->user_empty()) {
      AllocFn->eraseFromParent();
    }
  }

  // Convert all uses of dx.break() into per-function load/cmp of dx.break.cond global constant
  void LowerDxBreak(Module &M) {
    if (Function *BreakFunc = M.getFunction(DXIL::kDxBreakFuncName)) {
      if (!BreakFunc->use_empty()) {
        llvm::Type *i32Ty = llvm::Type::getInt32Ty(M.getContext());
        Type *i32ArrayTy = ArrayType::get(i32Ty, 1);
        unsigned int Values[1] = { 0 };
        Constant *InitialValue = ConstantDataArray::get(M.getContext(), Values);
        Constant *GV = new GlobalVariable(M, i32ArrayTy, true,
                                          GlobalValue::InternalLinkage,
                                          InitialValue, DXIL::kDxBreakCondName);

        Constant *Indices[] = { ConstantInt::get(i32Ty, 0), ConstantInt::get(i32Ty, 0) };
        Constant *Gep = ConstantExpr::getGetElementPtr(nullptr, GV, Indices);
        SmallDenseMap<llvm::Function*, llvm::ICmpInst*, 16> DxBreakCmpMap;
        // Replace all uses of dx.break with references to the constant global
        for (auto I = BreakFunc->user_begin(), E = BreakFunc->user_end(); I != E;) {
          User *U = *I++;
          CallInst *CI = cast<CallInst>(U);
          Function *F = CI->getParent()->getParent();
          ICmpInst *Cmp = DxBreakCmpMap.lookup(F);
          if (!Cmp) {
            Instruction *IP = dxilutil::FindAllocaInsertionPt(F);
            LoadInst *LI = new LoadInst(Gep, nullptr, false, IP);
            Cmp = new ICmpInst(IP, ICmpInst::ICMP_EQ, LI, llvm::ConstantInt::get(i32Ty,0));
            DxBreakCmpMap[F] = Cmp;
          }
          CI->replaceAllUsesWith(Cmp);
          CI->eraseFromParent();
        }
      }
      BreakFunc->eraseFromParent();
    }

    for (Function &F : M) {
      for (BasicBlock &BB : F) {
        if (BranchInst *BI = dyn_cast<BranchInst>(BB.getTerminator())) {
          BI->setMetadata(DXIL::kDxBreakMDName, nullptr);
        }
      }
    }
  }
};
}

char DxilFinalizeModule::ID = 0;

ModulePass *llvm::createDxilFinalizeModulePass() {
  return new DxilFinalizeModule();
}

INITIALIZE_PASS(DxilFinalizeModule, "hlsl-dxilfinalize", "HLSL DXIL Finalize Module", false, false)


///////////////////////////////////////////////////////////////////////////////

namespace {
typedef MapVector< PHINode*, SmallVector<Value*,8> > PHIReplacementMap;
bool RemoveAddrSpaceCasts(Value *Val, Value *NewVal,
                          PHIReplacementMap &phiReplacements,
                          DenseMap<Value*, Value*> &valueMap) {
  bool bChanged = false;
  for (auto itU = Val->use_begin(), itEnd = Val->use_end(); itU != itEnd; ) {
    Use &use = *(itU++);
    User *user = use.getUser();
    Value *userReplacement = user;
    bool bConstructReplacement = false;
    bool bCleanupInst = false;
    auto valueMapIter = valueMap.find(user);
    if (valueMapIter != valueMap.end())
      userReplacement = valueMapIter->second;
    else if (Val != NewVal)
      bConstructReplacement = true;
    if (ConstantExpr* CE = dyn_cast<ConstantExpr>(user)) {
      if (CE->getOpcode() == Instruction::BitCast) {
        if (bConstructReplacement) {
          // Replicate bitcast in target address space
          Type* NewTy = PointerType::get(
            CE->getType()->getPointerElementType(),
            NewVal->getType()->getPointerAddressSpace());
          userReplacement = ConstantExpr::getBitCast(cast<Constant>(NewVal), NewTy);
        }
      } else if (CE->getOpcode() == Instruction::GetElementPtr) {
        if (bConstructReplacement) {
          // Replicate GEP in target address space
          GEPOperator *GEP = cast<GEPOperator>(CE);
          SmallVector<Value*, 8> idxList(GEP->idx_begin(), GEP->idx_end());
          userReplacement = ConstantExpr::getGetElementPtr(
            nullptr, cast<Constant>(NewVal), idxList, GEP->isInBounds());
        }
      } else if (CE->getOpcode() == Instruction::AddrSpaceCast) {
        userReplacement = NewVal;
        bConstructReplacement = false;
      } else {
        DXASSERT(false, "RemoveAddrSpaceCasts: unhandled pointer ConstantExpr");
      }
    } else if (Instruction *I = dyn_cast<Instruction>(user)) {
      if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(user)) {
        if (bConstructReplacement) {
          IRBuilder<> Builder(GEP);
          SmallVector<Value*, 8> idxList(GEP->idx_begin(), GEP->idx_end());
          if (GEP->isInBounds())
            userReplacement = Builder.CreateInBoundsGEP(NewVal, idxList, GEP->getName());
          else
            userReplacement = Builder.CreateGEP(NewVal, idxList, GEP->getName());
        }
      } else if (BitCastInst *BC = dyn_cast<BitCastInst>(user)) {
        if (bConstructReplacement) {
          IRBuilder<> Builder(BC);
          Type* NewTy = PointerType::get(
            BC->getType()->getPointerElementType(),
            NewVal->getType()->getPointerAddressSpace());
          userReplacement = Builder.CreateBitCast(NewVal, NewTy);
        }
      } else if (PHINode *PHI = dyn_cast<PHINode>(user)) {
        // set replacement phi values for PHI pass
        unsigned numValues = PHI->getNumIncomingValues();
        auto &phiValues = phiReplacements[PHI];
        if (phiValues.empty())
          phiValues.resize(numValues, nullptr);
        for (unsigned idx = 0; idx < numValues; ++idx) {
          if (phiValues[idx] == nullptr &&
              PHI->getIncomingValue(idx) == Val) {
            phiValues[idx] = NewVal;
            bChanged = true;
          }
        }
        continue;
      } else if (isa<AddrSpaceCastInst>(user)) {
        userReplacement = NewVal;
        bConstructReplacement = false;
        bCleanupInst = true;
      } else if (isa<CallInst>(user)) {
        continue;
      } else {
        if (Val != NewVal) {
          use.set(NewVal);
          bChanged = true;
        }
        continue;
      }
    }
    if (bConstructReplacement && user != userReplacement)
      valueMap[user] = userReplacement;
    bChanged |= RemoveAddrSpaceCasts(user, userReplacement, phiReplacements,
                                      valueMap);
    if (bCleanupInst && user->use_empty()) {
      // Clean up old instruction if it's now unused.
      // Safe during this use iteration when only one use of V in instruction.
      if (Instruction *I = dyn_cast<Instruction>(user))
        I->eraseFromParent();
      bChanged = true;
    }
  }
  return bChanged;
}
}

bool CleanupSharedMemoryAddrSpaceCast(Module &M) {
  bool bChanged = false;
  // Eliminate address space casts if possible
  // Collect phi nodes so we can replace iteratively after pass over GVs
  PHIReplacementMap phiReplacements;
  DenseMap<Value*, Value*> valueMap;
  for (GlobalVariable &GV : M.globals()) {
    if (dxilutil::IsSharedMemoryGlobal(&GV)) {
      bChanged |= RemoveAddrSpaceCasts(&GV, &GV, phiReplacements,
                                       valueMap);
    }
  }
  bool bConverged = false;
  while (!phiReplacements.empty() && !bConverged) {
    bConverged = true;
    for (auto &phiReplacement : phiReplacements) {
      PHINode *PHI = phiReplacement.first;
      unsigned origAddrSpace = PHI->getType()->getPointerAddressSpace();
      unsigned incomingAddrSpace = UINT_MAX;
      bool bReplacePHI = true;
      bool bRemovePHI = false;
      for (auto V : phiReplacement.second) {
        if (nullptr == V) {
          // cannot replace phi (yet)
          bReplacePHI = false;
          break;
        }
        unsigned addrSpace = V->getType()->getPointerAddressSpace();
        if (incomingAddrSpace == UINT_MAX) {
          incomingAddrSpace = addrSpace;
        } else if (addrSpace != incomingAddrSpace) {
          bRemovePHI = true;
          break;
        }
      }
      if (origAddrSpace == incomingAddrSpace)
        bRemovePHI = true;
      if (bRemovePHI) {
        // Cannot replace phi.  Remove it and restart.
        phiReplacements.erase(PHI);
        bConverged = false;
        break;
      }
      if (!bReplacePHI)
        continue;
      auto &NewVal = valueMap[PHI];
      PHINode *NewPHI = nullptr;
      if (NewVal) {
        NewPHI = cast<PHINode>(NewVal);
      } else {
        IRBuilder<> Builder(PHI);
        NewPHI = Builder.CreatePHI(
          PointerType::get(PHI->getType()->getPointerElementType(),
                           incomingAddrSpace),
          PHI->getNumIncomingValues(),
          PHI->getName());
        NewVal = NewPHI;
        for (unsigned idx = 0; idx < PHI->getNumIncomingValues(); idx++) {
          NewPHI->addIncoming(phiReplacement.second[idx],
                              PHI->getIncomingBlock(idx));
        }
      }
      if (RemoveAddrSpaceCasts(PHI, NewPHI, phiReplacements,
                               valueMap)) {
        bConverged = false;
        bChanged = true;
        break;
      }
      if (PHI->use_empty()) {
        phiReplacements.erase(PHI);
        bConverged = false;
        bChanged = true;
        break;
      }
    }
  }

  // Cleanup unused replacement instructions
  SmallVector<WeakVH, 8> cleanupInsts;
  for (auto it : valueMap) {
    if (isa<Instruction>(it.first))
      cleanupInsts.push_back(it.first);
    if (isa<Instruction>(it.second))
      cleanupInsts.push_back(it.second);
  }
  for (auto V : cleanupInsts) {
    if (!V)
      continue;
    if (PHINode *PHI = dyn_cast<PHINode>(V))
      RecursivelyDeleteDeadPHINode(PHI);
    else if (Instruction *I = dyn_cast<Instruction>(V))
      RecursivelyDeleteTriviallyDeadInstructions(I);
  }

  return bChanged;
}

class DxilCleanupAddrSpaceCast : public ModulePass {
public:
  static char ID; // Pass identification, replacement for typeid
  explicit DxilCleanupAddrSpaceCast() : ModulePass(ID) {}

  StringRef getPassName() const override { return "HLSL DXIL Cleanup Address Space Cast"; }

  bool runOnModule(Module &M) override {
    return CleanupSharedMemoryAddrSpaceCast(M);
  }
};

char DxilCleanupAddrSpaceCast::ID = 0;

ModulePass *llvm::createDxilCleanupAddrSpaceCastPass() {
  return new DxilCleanupAddrSpaceCast();
}

INITIALIZE_PASS(DxilCleanupAddrSpaceCast, "hlsl-dxil-cleanup-addrspacecast", "HLSL DXIL Cleanup Address Space Cast", false, false)

///////////////////////////////////////////////////////////////////////////////

namespace {

class DxilEmitMetadata : public ModulePass {
public:
  static char ID; // Pass identification, replacement for typeid
  explicit DxilEmitMetadata() : ModulePass(ID) {}

  StringRef getPassName() const override { return "HLSL DXIL Metadata Emit"; }

  bool runOnModule(Module &M) override {
    if (M.HasDxilModule()) {
      DxilModule::ClearDxilMetadata(M);
      patchIsFrontfaceTy(M);
      M.GetDxilModule().EmitDxilMetadata();
      return true;
    }

    return false;
  }
private:
  void patchIsFrontfaceTy(Module &M);
};

void patchIsFrontface(DxilSignatureElement &Elt, bool bForceUint) {
  // If force to uint, change i1 to u32.
  // If not force to uint, change u32 to i1.
  if (bForceUint && Elt.GetCompType() == CompType::Kind::I1)
    Elt.SetCompType(CompType::Kind::U32);
  else if (!bForceUint && Elt.GetCompType() == CompType::Kind::U32)
    Elt.SetCompType(CompType::Kind::I1);
}

void patchIsFrontface(DxilSignature &sig, bool bForceUint) {
  for (auto &Elt : sig.GetElements()) {
    if (Elt->GetSemantic()->GetKind() == Semantic::Kind::IsFrontFace) {
      patchIsFrontface(*Elt, bForceUint);
    }
  }
}

void DxilEmitMetadata::patchIsFrontfaceTy(Module &M) {
  DxilModule &DM = M.GetDxilModule();
  const ShaderModel *pSM = DM.GetShaderModel();
  if (!pSM->IsGS() && !pSM->IsPS())
    return;
  unsigned ValMajor, ValMinor;
  DM.GetValidatorVersion(ValMajor, ValMinor);
  bool bForceUint = ValMajor == 0 || (ValMajor >= 1 && ValMinor >= 2);
  if (pSM->IsPS()) {
    patchIsFrontface(DM.GetInputSignature(), bForceUint);
  } else if (pSM->IsGS()) {
    patchIsFrontface(DM.GetOutputSignature(), bForceUint);
  }
}

}

char DxilEmitMetadata::ID = 0;

ModulePass *llvm::createDxilEmitMetadataPass() {
  return new DxilEmitMetadata();
}

INITIALIZE_PASS(DxilEmitMetadata, "hlsl-dxilemit", "HLSL DXIL Metadata Emit", false, false)

///////////////////////////////////////////////////////////////////////////////

namespace {

const StringRef UniNoWaveSensitiveGradientErrMsg =
    "Gradient operations are not affected by wave-sensitive data or control "
    "flow.";

class DxilValidateWaveSensitivity : public ModulePass {
public:
  static char ID; // Pass identification, replacement for typeid
  explicit DxilValidateWaveSensitivity() : ModulePass(ID) {}

  StringRef getPassName() const override {
    return "HLSL DXIL wave sensitiveity validation";
  }

  bool runOnModule(Module &M) override {
    // Only check ps and lib profile.
    DxilModule &DM = M.GetDxilModule();
    const ShaderModel *pSM = DM.GetShaderModel();
    if (!pSM->IsPS() && !pSM->IsLib())
      return false;

    SmallVector<CallInst *, 16> gradientOps;
    SmallVector<CallInst *, 16> barriers;
    SmallVector<CallInst *, 16> waveOps;

    for (auto &F : M) {
      if (!F.isDeclaration())
        continue;

      for (User *U : F.users()) {
        CallInst *CI = dyn_cast<CallInst>(U);
        if (!CI)
          continue;
        Function *FCalled = CI->getCalledFunction();
        if (!FCalled || !FCalled->isDeclaration())
          continue;

        if (!hlsl::OP::IsDxilOpFunc(FCalled))
          continue;

        DXIL::OpCode dxilOpcode = hlsl::OP::GetDxilOpFuncCallInst(CI);

        if (OP::IsDxilOpWave(dxilOpcode)) {
          waveOps.emplace_back(CI);
        }

        if (OP::IsDxilOpGradient(dxilOpcode)) {
          gradientOps.push_back(CI);
        }

        if (dxilOpcode == DXIL::OpCode::Barrier) {
          barriers.push_back(CI);
        }
      }
    }

    // Skip if not have wave op.
    if (waveOps.empty())
      return false;

    // Skip if no gradient op.
    if (gradientOps.empty())
      return false;

    for (auto &F : M) {
      if (F.isDeclaration())
        continue;

      SetVector<Instruction *> localGradientArgs;
      for (CallInst *CI : gradientOps) {
        if (CI->getParent()->getParent() == &F) {
          for (Value *V : CI->arg_operands()) {
            // TODO: only check operand which used for gradient calculation.
            Instruction *vI = dyn_cast<Instruction>(V);
            if (!vI)
              continue;
            localGradientArgs.insert(vI);
          }
        }
      }

      if (localGradientArgs.empty())
        continue;

      PostDominatorTree PDT;
      PDT.runOnFunction(F);
      std::unique_ptr<WaveSensitivityAnalysis> WaveVal(
          WaveSensitivityAnalysis::create(PDT));

      WaveVal->Analyze(&F);
      for (Instruction *gradArg : localGradientArgs) {
        // Check operand of gradient ops, not gradientOps itself.
        if (WaveVal->IsWaveSensitive(gradArg)) {
          dxilutil::EmitWarningOnInstruction(gradArg,
                                             UniNoWaveSensitiveGradientErrMsg);
        }
      }
    }
    return false;
  }
};

}

char DxilValidateWaveSensitivity::ID = 0;

ModulePass *llvm::createDxilValidateWaveSensitivityPass() {
  return new DxilValidateWaveSensitivity();
}

INITIALIZE_PASS(DxilValidateWaveSensitivity, "hlsl-validate-wave-sensitivity", "HLSL DXIL wave sensitiveity validation", false, false)


namespace {

// Cull blocks from BreakBBs that containing instructions that are sensitive to the wave-sensitive Inst
// Sensitivity entails being an eventual user of the Inst and also belonging to a block with
// a break conditional on dx.break that breaks out of a loop that contains WaveCI
// LInfo is needed to determine loop contents. Visited is needed to prevent infinite looping.
static void CullSensitiveBlocks(LoopInfo *LInfo, Loop *WaveLoop, BasicBlock *LastBB, Instruction *Inst,
                                std::unordered_set<Instruction *> &Visited,
                                SmallDenseMap<BasicBlock *, Instruction *, 16> &BreakBBs) {
  BasicBlock *BB = Inst->getParent();
  Loop *BreakLoop = LInfo->getLoopFor(BB);
  // If this instruction isn't in a loop, there is no need to track its sensitivity further
  if (!BreakLoop || BreakBBs.empty())
    return;

  // To prevent infinite looping, only visit each instruction once
  if (!Visited.insert(Inst).second)
    return;

  // If this BB wasn't already just processed, handle it now
  if (LastBB != BB) {
    // Determine if the instruction's block has an artificially-conditional break
    // and breaks out of a loop that contains the waveCI
    BranchInst *BI = dyn_cast<BranchInst>(BB->getTerminator());
    if (BI && BI->isConditional() && BreakLoop->contains(WaveLoop))
      BreakBBs.erase(BB);
  }

  // Recurse on the users
  for (User *U : Inst->users()) {
    Instruction *I = cast<Instruction>(U);
    CullSensitiveBlocks(LInfo, WaveLoop, BB, I, Visited, BreakBBs);
  }
}

// Collect blocks that end in a dx.break dependent branch by tracing the descendants of BreakFunc
// that are found in ThisFunc and store the block and call instruction in BreakBBs
static void CollectBreakBlocks(Function *BreakFunc, Function *ThisFunc,
                               SmallDenseMap<BasicBlock *, Instruction *, 16> &BreakBBs) {
  for (User *U : BreakFunc->users()) {
    SmallVector<User *, 16> WorkList;
    Instruction *CI = cast<Instruction>(U);
    // If this user doesn't pertain to the current function, skip it.
    if (CI->getParent()->getParent() != ThisFunc)
      continue;
    WorkList.append(CI->user_begin(), CI->user_end());
    while (!WorkList.empty()) {
      Instruction *I = dyn_cast<Instruction>(WorkList.pop_back_val());
      // When we find a Branch that depends on dx.break, save it and stop
      // This should almost always be the first user of the Call Inst
      // If not, iterate on the users
      if (BranchInst *BI = dyn_cast<BranchInst>(I))
        BreakBBs[BI->getParent()] = CI;
      else
        WorkList.append(I->user_begin(), I->user_end());
    }
  }
}


// A pass to remove conditions from breaks that do not contain instructions that
// depend on wave operations that are in the loop that the break leaves.
class CleanupDxBreak : public FunctionPass {
public:
  static char ID; // Pass identification, replacement for typeid
  explicit CleanupDxBreak() : FunctionPass(ID) {}
  StringRef getPassName() const override { return "HLSL Remove unnecessary dx.break conditions"; }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LoopInfoWrapperPass>();
  }

  LoopInfo *LInfo;

  bool runOnFunction(Function &F) override {
    if (F.isDeclaration())
      return false;
    Module *M = F.getEntryBlock().getModule();

    Function *BreakFunc = M->getFunction(DXIL::kDxBreakFuncName);
    if (!BreakFunc)
      return false;

    LInfo = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    // Collect the blocks that depend on dx.break and the instructions that call dx.break()
    SmallDenseMap<BasicBlock *, Instruction *, 16> BreakBBs;
    CollectBreakBlocks(BreakFunc, &F, BreakBBs);

    if (BreakBBs.empty())
      return false;



    // Collect all wave calls in this function and group by loop
    SmallDenseMap<Loop *, SmallVector<CallInst *, 8>, 16> WaveCalls;

    for (Function &IF : M->functions()) {
      HLOpcodeGroup opgroup = hlsl::GetHLOpcodeGroup(&IF);
      // Only consider wave-sensitive intrinsics or extintrinsics
      if (IF.isDeclaration() && IsHLWaveSensitive(&IF) && !BreakBBs.empty() &&
          (opgroup == HLOpcodeGroup::HLIntrinsic || opgroup == HLOpcodeGroup::HLExtIntrinsic)) {
        // For each user of the function, trace all its users to remove the blocks
        for (User *U : IF.users()) {
          CallInst *CI = cast<CallInst>(U);
          if (CI->getParent()->getParent() == &F) {
            Loop *WaveLoop = LInfo->getLoopFor(CI->getParent());
            WaveCalls[WaveLoop].emplace_back(CI);
          }
        }
      }
    }

    // For each wave operation, remove all the dx.break blocks that are sensitive to it
    for (DenseMap<Loop*, SmallVector<CallInst *, 8>>::iterator I =
           WaveCalls.begin(), E = WaveCalls.end();
         I != E; ++I) {
      Loop *loop = I->first;
      std::unordered_set<Instruction *> Visited;
      for (CallInst *CI : I->second) {
        CullSensitiveBlocks(LInfo, loop, nullptr, CI, Visited, BreakBBs);
      }
    }

    bool Changed = false;
    // Revert artificially conditional breaks in non-wave-sensitive blocks that remain in BreakBBs
    Constant *C = ConstantInt::get(Type::getInt1Ty(M->getContext()), 1);
    for (auto &BB : BreakBBs) {
      // Replace the call instruction with a constant boolen
      BB.second->replaceAllUsesWith(C);
      BB.second->eraseFromParent();
      Changed = true;
    }
    return Changed;
  }
};

}

char CleanupDxBreak::ID = 0;

INITIALIZE_PASS_BEGIN(CleanupDxBreak, "hlsl-cleanup-dxbreak", "HLSL Remove unnecessary dx.break conditions", false, false)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_END(CleanupDxBreak, "hlsl-cleanup-dxbreak", "HLSL Remove unnecessary dx.break conditions", false, false)

FunctionPass *llvm::createCleanupDxBreakPass() {
  return new CleanupDxBreak();
}
