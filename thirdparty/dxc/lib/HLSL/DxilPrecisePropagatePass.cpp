///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilPrecisePropagatePass.cpp                                              //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/DXIL/DxilModule.h"
#include "dxc/HLSL/DxilGenerationPass.h"
#include "dxc/HLSL/HLModule.h"
#include "dxc/HLSL/HLOperations.h"
#include "dxc/HLSL/ControlDependence.h"
#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"
#include <unordered_set>
#include <vector>

using namespace llvm;
using namespace hlsl;

namespace {

typedef std::unordered_set<Value *> ValueSet;

struct FuncInfo {
  ControlDependence CtrlDep;
  std::unique_ptr<llvm::DominatorTreeBase<llvm::BasicBlock>> pPostDom;
  void Init(Function *F);
  void Clear();
};
typedef std::unordered_map<llvm::Function *, std::unique_ptr<FuncInfo>> FuncInfoMap;

class DxilPrecisePropagatePass : public ModulePass {
public:
  static char ID; // Pass identification, replacement for typeid
  explicit DxilPrecisePropagatePass() : ModulePass(ID) {}

  StringRef getPassName() const override { return "DXIL Precise Propagate"; }

  bool runOnModule(Module &M) override {
    m_pDM = &(M.GetOrCreateDxilModule());
    std::vector<Function*> deadList;
    for (Function &F : M.functions()) {
      if (HLModule::HasPreciseAttribute(&F)) {
        PropagatePreciseOnFunctionUser(F);
        deadList.emplace_back(&F);
      }
    }
    for (Function *F : deadList)
      F->eraseFromParent();
    return true;
  }

private:
  void PropagatePreciseOnFunctionUser(Function &F);

  void AddToWorkList(Value *V);
  void ProcessWorkList();

  void Propagate(Instruction *I);
  void PropagateOnPointer(Value *Ptr);
  void PropagateOnPointerUsers(Value *Ptr);
  void PropagateThroughGEPs(Value *Ptr, ArrayRef<Value *> idxList,
                            ValueSet &processedGEPs);
  void PropagateOnPointerUsedInCall(Value *Ptr, CallInst *CI);

  void PropagateCtrlDep(FuncInfo &FI, BasicBlock *BB);
  void PropagateCtrlDep(BasicBlock *BB);
  void PropagateCtrlDep(Instruction *I);

  // Add to m_ProcessedSet, return true if already in set.
  bool Processed(Value *V) {
    return !m_ProcessedSet.insert(V).second;
  }

  FuncInfo &GetFuncInfo(Function *F);

  DxilModule *m_pDM;
  std::vector<Value*> m_WorkList;
  ValueSet m_ProcessedSet;
  FuncInfoMap m_FuncInfo;
};

char DxilPrecisePropagatePass::ID = 0;

}

void DxilPrecisePropagatePass::PropagatePreciseOnFunctionUser(Function &F) {
  for (auto U = F.user_begin(), E = F.user_end(); U != E;) {
    CallInst *CI = cast<CallInst>(*(U++));
    Value *V = CI->getArgOperand(0);
    AddToWorkList(V);
    ProcessWorkList();
    CI->eraseFromParent();
  }
}

void DxilPrecisePropagatePass::AddToWorkList(Value *V) {
  // Skip values already marked.
  if (Processed(V))
    return;

  m_WorkList.emplace_back(V);
}

void DxilPrecisePropagatePass::ProcessWorkList() {
  while (!m_WorkList.empty()) {
    Value *V = m_WorkList.back();
    m_WorkList.pop_back();

    if (V->getType()->isPointerTy()) {
      PropagateOnPointer(V);
    }

    Instruction *I = dyn_cast<Instruction>(V);
    if (!I)
      continue;

    // Set precise fast math on those instructions that support it.
    if (DxilModule::PreservesFastMathFlags(I))
      DxilModule::SetPreciseFastMathFlags(I);

    // Fast math not work on call, use metadata.
    if (isa<FPMathOperator>(I) && isa<CallInst>(I))
      HLModule::MarkPreciseAttributeWithMetadata(cast<CallInst>(I));

    Propagate(I);
    PropagateCtrlDep(I);
  }
}

void DxilPrecisePropagatePass::Propagate(Instruction *I) {
  if (CallInst *CI = dyn_cast<CallInst>(I)) {
    for (unsigned i = 0; i < CI->getNumArgOperands(); i++)
      AddToWorkList(CI->getArgOperand(i));
  } else {
    for (Value *src : I->operands())
      AddToWorkList(src);
  }

  if (PHINode *Phi = dyn_cast<PHINode>(I)) {
    // Use pred for control dependence when constant (for now)
    FuncInfo &FI = GetFuncInfo(I->getParent()->getParent());
    for (unsigned i = 0; i < Phi->getNumIncomingValues(); i++) {
      if (isa<Constant>(Phi->getIncomingValue(i)))
        PropagateCtrlDep(FI, Phi->getIncomingBlock(i));
    }
  }
}

// TODO: This could be a util function
// TODO: Should this tunnel through addrspace cast?
//       And how could bitcast be handled?
static Value *GetRootAndIndicesForGEP(
    GEPOperator *GEP, SmallVectorImpl<Value*> &idxList) {
  Value *Ptr = GEP;
  SmallVector<GEPOperator*, 4> GEPs;
  GEPs.emplace_back(GEP);
  while ((GEP = dyn_cast<GEPOperator>(Ptr = GEP->getPointerOperand())))
    GEPs.emplace_back(GEP);
  while (!GEPs.empty()) {
    GEP = GEPs.back();
    GEPs.pop_back();
    auto idx = GEP->idx_begin();
    idx++;
    while (idx != GEP->idx_end())
      idxList.emplace_back(*(idx++));
  }
  return Ptr;
}

void DxilPrecisePropagatePass::PropagateOnPointer(Value *Ptr) {

  PropagateOnPointerUsers(Ptr);

  // GetElementPointer gets special treatment since different GEPs may be used
  // at different points on the same root pointer to load or store data.  We
  // need to find any stores that could have written data to the pointer we are
  // marking, so we need to search through all GEPs from the root pointer for
  // ones that may write to the same location.
  //
  // In addition, there may be multiple GEPs between the root pointer and loads
  // or stores, so we need to accumulate all the indices between the root and
  // the leaf pointer we are marking.
  //
  // Starting at the root pointer, we follow users, looking for GEPs with
  // indices that could "match", or calls that may write to the pointer along
  // the way. A "match" to the reference index is one that matches with constant
  // values, or if either index is non-constant, since the compiler doesn't know
  // what index may be read or written in that case.
  //
  // This still doesn't handle addrspace cast or bitcast, so propagation through
  // groupshared aggregates will not work, as one example.

  if (GEPOperator *GEP = dyn_cast<GEPOperator>(Ptr)) {
    // Get root Ptr, gather index list, and mark matching stores
    SmallVector<Value*, 8> idxList;
    Ptr = GetRootAndIndicesForGEP(GEP, idxList);
    ValueSet processedGEPs;
    PropagateThroughGEPs(Ptr, idxList, processedGEPs);
  }
}

void DxilPrecisePropagatePass::PropagateOnPointerUsers(Value *Ptr) {
  // Find all store and propagate on the val operand of store.
  // For CallInst, if Ptr is used as out parameter, mark it.
  for (User *U : Ptr->users()) {
    if (StoreInst *stInst = dyn_cast<StoreInst>(U)) {
      Value *val = stInst->getValueOperand();
      AddToWorkList(val);
    } else if (CallInst *CI = dyn_cast<CallInst>(U)) {
      if (Function *F = CI->getCalledFunction()) {
        // Skip llvm intrinsics (debug/lifetime intrinsics)
        if (!F->isIntrinsic())
          PropagateOnPointerUsedInCall(Ptr, CI);
      }
    } else if (isa<GEPOperator>(U) || isa<BitCastOperator>(U)) {
      PropagateOnPointerUsers(U);
    }
  }
}

void DxilPrecisePropagatePass::PropagateThroughGEPs(
    Value *Ptr, ArrayRef<Value*> idxList, ValueSet &processedGEPs) {
  // recurse to matching GEP users
  for (User *U : Ptr->users()) {
    if (GEPOperator *GEP = dyn_cast<GEPOperator>(U)) {
      // skip visited GEPs
      // These are separate from processedSet because while we don't need to
      // visit an intermediate GEP multiple times while marking a single value
      // precise, we are not necessarily marking every value reachable from
      // the GEP as precise, so we may need to revisit when marking a different
      // value as precise.
      if (!processedGEPs.insert(GEP).second)
        continue;

      // Mismatch if both constant and unequal, otherwise be conservative.
      bool bMismatch = false;
      auto idx = GEP->idx_begin();
      idx++;
      unsigned i = 0;
      // FIXME: When i points outside idxList, it's an indication that this GEP
      // is deeper than the one we are matching. This can happen with vector
      // components or aggregates when marking the aggregate precise, such as
      // when propagating through call with aggregate argument. This solution
      // only prevents OOB memory access, it does not fix the underlying
      // problems that lead to it, which will likely require significant work -
      // perhaps even a rewrite using alias analysis or some other more accurate
      // mechanism.
      while (idx != GEP->idx_end() && i < idxList.size()) {
        if (ConstantInt *C = dyn_cast<ConstantInt>(*idx)) {
          if (ConstantInt *CRef = dyn_cast<ConstantInt>(idxList[i])) {
            if (CRef->getLimitedValue() != C->getLimitedValue()) {
              bMismatch = true;
              break;
            }
          }
        }
        idx++;
        i++;
      }
      if (bMismatch)
        continue;

      if ((unsigned)idxList.size() == i) {
        // Mark leaf users
        if (Processed(GEP))
          continue;
        PropagateOnPointerUsers(GEP);
      } else {
        // Recurse GEP users
        PropagateThroughGEPs(
            GEP, ArrayRef<Value*>(idxList.data() + i, idxList.end()),
            processedGEPs);
      }
    } else if (CallInst *CI = dyn_cast<CallInst>(U)) {
      // Root pointer or intermediate GEP used in call.
      // If it may write to the pointer, we must mark the call and recurse
      // arguments.
      // This also widens the precise propagation to the entire aggregate
      // pointed to by the root ptr or intermediate GEP.
      PropagateOnPointerUsedInCall(Ptr, CI);
    }
  }
}

void DxilPrecisePropagatePass::PropagateOnPointerUsedInCall(
    Value *Ptr, CallInst *CI) {
  bool bReadOnly = true;

  Function *F = CI->getCalledFunction();

  // skip starting points (dx.attribute.precise calls)
  if (HLModule::HasPreciseAttribute(F))
    return;

  const DxilFunctionAnnotation *funcAnnotation =
      m_pDM->GetTypeSystem().GetFunctionAnnotation(F);

  if (funcAnnotation) {
    for (unsigned i = 0; i < CI->getNumArgOperands(); ++i) {
      if (Ptr != CI->getArgOperand(i))
        continue;

      const DxilParameterAnnotation &paramAnnotation =
          funcAnnotation->GetParameterAnnotation(i);
      // OutputPatch and OutputStream will be checked after scalar repl.
      // Here only check out/inout
      if (paramAnnotation.GetParamInputQual() == DxilParamInputQual::Out ||
          paramAnnotation.GetParamInputQual() == DxilParamInputQual::Inout) {
        bReadOnly = false;
        break;
      }
    }
  } else {
    bReadOnly = false;
  }

  if (!bReadOnly) {
    AddToWorkList(CI);
  }
}

void FuncInfo::Init(Function *F) {
  if (!pPostDom) {
    pPostDom = make_unique<DominatorTreeBase<BasicBlock> >(true);
    pPostDom->recalculate(*F);
    CtrlDep.Compute(F, *pPostDom);
  }
}
void FuncInfo::Clear() {
  CtrlDep.Clear();
  pPostDom.reset();
}
FuncInfo &DxilPrecisePropagatePass::GetFuncInfo(Function *F) {
  auto &FI = m_FuncInfo[F];
  if (!FI) {
    FI = make_unique<FuncInfo>();
    FI->Init(F);
  }
  return *FI.get();
}

void DxilPrecisePropagatePass::PropagateCtrlDep(FuncInfo &FI, BasicBlock *BB) {
  if (Processed(BB))
    return;
  const BasicBlockSet &CtrlDepSet = FI.CtrlDep.GetCDBlocks(BB);
  for (BasicBlock *B : CtrlDepSet) {
    AddToWorkList(B->getTerminator());
  }
}

void DxilPrecisePropagatePass::PropagateCtrlDep(BasicBlock *BB) {
  FuncInfo &FI = GetFuncInfo(BB->getParent());
  PropagateCtrlDep(FI, BB);
}

void DxilPrecisePropagatePass::PropagateCtrlDep(Instruction *I) {
  PropagateCtrlDep(I->getParent());
}

ModulePass *llvm::createDxilPrecisePropagatePass() {
  return new DxilPrecisePropagatePass();
}

INITIALIZE_PASS(DxilPrecisePropagatePass, "hlsl-dxil-precise", "DXIL precise attribute propagate", false, false)
