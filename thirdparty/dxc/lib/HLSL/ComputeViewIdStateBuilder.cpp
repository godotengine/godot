///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// ComputeViewIdStateBuilder.cpp                                             //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/HlslIntrinsicOp.h"
#include "dxc/HLSL/ComputeViewIdState.h"
#include "dxc/HLSL/HLOperations.h"
#include "dxc/Support/Global.h"
#include "dxc/DXIL/DxilModule.h"
#include "dxc/DXIL/DxilOperations.h"
#include "dxc/DXIL/DxilInstructions.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Operator.h"
#include "llvm/Pass.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/Debug.h"
#include "llvm/IR/CFG.h"
#include "llvm/Analysis/CallGraph.h"

#include <algorithm>

using namespace llvm;
using namespace llvm::legacy;
using namespace hlsl;
using llvm::legacy::PassManager;
using llvm::legacy::FunctionPassManager;
using std::vector;
using std::unordered_set;
using std::unordered_map;

#define DXILVIEWID_DBG   0

#define DEBUG_TYPE "viewid_builder"

namespace {
class DxilViewIdStateBuilder {
  static const unsigned kNumComps = 4;
  static const unsigned kMaxSigScalars = 32 * 4;

public:
  using OutputsDependentOnViewIdType = DxilViewIdStateData::OutputsDependentOnViewIdType;
  using InputsContributingToOutputType = DxilViewIdStateData::InputsContributingToOutputType;

  DxilViewIdStateBuilder(DxilViewIdStateData &state, DxilModule *pDxilModule)
      : m_pModule(pDxilModule),
        m_NumInputSigScalars(state.m_NumInputSigScalars),
        m_NumOutputSigScalars(state.m_NumOutputSigScalars,
                              DxilViewIdStateData::kNumStreams),
        m_NumPCOrPrimSigScalars(state.m_NumPCOrPrimSigScalars),
        m_OutputsDependentOnViewId(state.m_OutputsDependentOnViewId,
                                   DxilViewIdStateData::kNumStreams),
        m_PCOrPrimOutputsDependentOnViewId(state.m_PCOrPrimOutputsDependentOnViewId),
        m_InputsContributingToOutputs(state.m_InputsContributingToOutputs,
                                      DxilViewIdStateData::kNumStreams),
        m_InputsContributingToPCOrPrimOutputs(state.m_InputsContributingToPCOrPrimOutputs),
        m_PCInputsContributingToOutputs(state.m_PCInputsContributingToOutputs),
        m_bUsesViewId(state.m_bUsesViewId) {}

  void Compute();

private:
  static const unsigned kNumStreams = 4;

  DxilModule *m_pModule;

  unsigned &m_NumInputSigScalars;
  MutableArrayRef<unsigned> m_NumOutputSigScalars;
  unsigned &m_NumPCOrPrimSigScalars;

  // Set of scalar outputs dependent on ViewID.
  MutableArrayRef<OutputsDependentOnViewIdType> m_OutputsDependentOnViewId;
  OutputsDependentOnViewIdType &m_PCOrPrimOutputsDependentOnViewId;

  // Set of scalar inputs contributing to computation of scalar outputs.
  MutableArrayRef<InputsContributingToOutputType> m_InputsContributingToOutputs;
  InputsContributingToOutputType &m_InputsContributingToPCOrPrimOutputs; // HS PC and MS Prim only.
  InputsContributingToOutputType &m_PCInputsContributingToOutputs; // DS only.

  bool &m_bUsesViewId;

  // Members for build ViewIdState.

  // Dynamically indexed components of signature elements.
  using DynamicallyIndexedElemsType = std::unordered_map<unsigned, unsigned>;
  DynamicallyIndexedElemsType m_InpSigDynIdxElems;
  DynamicallyIndexedElemsType m_OutSigDynIdxElems;
  DynamicallyIndexedElemsType m_PCSigDynIdxElems;

  // Information per entry point.
  using FunctionSetType = std::unordered_set<llvm::Function *>;
  using InstructionSetType = std::unordered_set<llvm::Instruction *>;
  struct EntryInfo {
    llvm::Function *pEntryFunc = nullptr;
    // Sets of functions that may be reachable from an entry.
    FunctionSetType Functions;
    // Outputs to analyze.
    InstructionSetType Outputs;
    // Contributing instructions per output.
    std::unordered_map<unsigned, InstructionSetType>
        ContributingInstructions[kNumStreams];

    void Clear();
  };

  EntryInfo m_Entry;
  EntryInfo m_PCEntry;

  // Information per function.
  using FunctionReturnSet = std::unordered_set<llvm::ReturnInst *>;
  struct FuncInfo {
    FunctionReturnSet Returns;
    ControlDependence CtrlDep;
    std::unique_ptr<llvm::DominatorTreeBase<llvm::BasicBlock>> pDomTree;
    void Clear();
  };

  std::unordered_map<llvm::Function *, std::unique_ptr<FuncInfo>> m_FuncInfo;

  // Cache of decls (global/alloca) reaching a pointer value.
  using ValueSetType = std::unordered_set<llvm::Value *>;
  std::unordered_map<llvm::Value *, ValueSetType> m_ReachingDeclsCache;
  // Cache of stores for each decl.
  std::unordered_map<llvm::Value *, ValueSetType> m_StoresPerDeclCache;


  void Clear();
  void DetermineMaxPackedLocation(DxilSignature &DxilSig, unsigned *pMaxSigLoc,
                                  unsigned NumStreams);
  void ComputeReachableFunctionsRec(llvm::CallGraph &CG,
                                    llvm::CallGraphNode *pNode,
                                    FunctionSetType &FuncSet);
  void AnalyzeFunctions(EntryInfo &Entry);
  void CollectValuesContributingToOutputs(EntryInfo &Entry);
  void CollectValuesContributingToOutputRec(
      EntryInfo &Entry, llvm::Value *pContributingValue,
      InstructionSetType &ContributingInstructions);
  void CollectPhiCFValuesContributingToOutputRec(
      llvm::PHINode *pPhi, EntryInfo &Entry,
      InstructionSetType &ContributingInstructions);
  const ValueSetType &CollectReachingDecls(llvm::Value *pValue);
  void CollectReachingDeclsRec(llvm::Value *pValue, ValueSetType &ReachingDecls,
                               ValueSetType &Visited);
  const ValueSetType &CollectStores(llvm::Value *pValue);
  void CollectStoresRec(llvm::Value *pValue, ValueSetType &Stores,
                        ValueSetType &Visited);
  void UpdateDynamicIndexUsageState() const;
  void
  CreateViewIdSets(const std::unordered_map<unsigned, InstructionSetType>
                       &ContributingInstructions,
                   OutputsDependentOnViewIdType &OutputsDependentOnViewId,
                   InputsContributingToOutputType &InputsContributingToOutputs,
                   bool bPC);

  void UpdateDynamicIndexUsageStateForSig(
      DxilSignature &Sig, const DynamicallyIndexedElemsType &DynIdxElems) const;
  unsigned GetLinearIndex(DxilSignatureElement &SigElem, int row,
                          unsigned col) const;

};
} // namespace

void DxilViewIdStateBuilder::Compute() {
  Clear();

  const ShaderModel *pSM = m_pModule->GetShaderModel();
  m_bUsesViewId = m_pModule->m_ShaderFlags.GetViewID();

  // 1. Traverse signature MD to determine max packed location.
  DetermineMaxPackedLocation(m_pModule->GetInputSignature(), &m_NumInputSigScalars, 1);
  DetermineMaxPackedLocation(m_pModule->GetOutputSignature(), &m_NumOutputSigScalars[0], pSM->IsGS() ? kNumStreams : 1);
  DetermineMaxPackedLocation(m_pModule->GetPatchConstOrPrimSignature(), &m_NumPCOrPrimSigScalars, 1);

  // 2. Collect sets of functions reachable from main and pc entries.
  CallGraphAnalysis CGA;
  CallGraph CG = CGA.run(m_pModule->GetModule());
  m_Entry.pEntryFunc = m_pModule->GetEntryFunction();
  m_PCEntry.pEntryFunc = m_pModule->GetPatchConstantFunction();
  ComputeReachableFunctionsRec(CG, CG[m_Entry.pEntryFunc], m_Entry.Functions);
  if (m_PCEntry.pEntryFunc) {
    DXASSERT_NOMSG(pSM->IsHS());
    ComputeReachableFunctionsRec(CG, CG[m_PCEntry.pEntryFunc], m_PCEntry.Functions);
  }

  // 3. Determine shape components that are dynamically accesses and collect all sig outputs.
  AnalyzeFunctions(m_Entry);
  if (m_PCEntry.pEntryFunc) {
    AnalyzeFunctions(m_PCEntry);
  }

  // 4. Collect sets of values contributing to outputs.
  CollectValuesContributingToOutputs(m_Entry);
  if (m_PCEntry.pEntryFunc) {
    CollectValuesContributingToOutputs(m_PCEntry);
  }

  // 5. Construct dependency sets.
  for (unsigned StreamId = 0; StreamId < (pSM->IsGS() ? kNumStreams : 1u); StreamId++) {
    CreateViewIdSets(m_Entry.ContributingInstructions[StreamId],
                     m_OutputsDependentOnViewId[StreamId],
                     m_InputsContributingToOutputs[StreamId], false);
  }
  if (pSM->IsHS() || pSM->IsMS()) {
    CreateViewIdSets(m_PCEntry.ContributingInstructions[0],
                     m_PCOrPrimOutputsDependentOnViewId,
                     m_InputsContributingToPCOrPrimOutputs, true);
  } else if (pSM->IsDS()) {
    OutputsDependentOnViewIdType OutputsDependentOnViewId;
    CreateViewIdSets(m_Entry.ContributingInstructions[0],
                     OutputsDependentOnViewId,
                     m_PCInputsContributingToOutputs, true);
    DXASSERT_NOMSG(OutputsDependentOnViewId == m_OutputsDependentOnViewId[0]);
  }

  // 6. Update dynamically indexed input/output component masks.
  UpdateDynamicIndexUsageState();

#if DXILVIEWID_DBG
  PrintSets(dbgs());
#endif
}

void DxilViewIdStateBuilder::Clear() {
  m_bUsesViewId = false;
  m_NumInputSigScalars  = 0;
  for (unsigned i = 0; i < kNumStreams; i++) {
    m_NumOutputSigScalars[i] = 0;
    m_OutputsDependentOnViewId[i].reset();
    m_InputsContributingToOutputs[i].clear();
  }
  m_NumPCOrPrimSigScalars     = 0;
  m_InpSigDynIdxElems.clear();
  m_OutSigDynIdxElems.clear();
  m_PCSigDynIdxElems.clear();
  m_PCOrPrimOutputsDependentOnViewId.reset();
  m_InputsContributingToPCOrPrimOutputs.clear();
  m_PCInputsContributingToOutputs.clear();
  m_Entry.Clear();
  m_PCEntry.Clear();
  m_FuncInfo.clear();
  m_ReachingDeclsCache.clear();
}

void DxilViewIdStateBuilder::EntryInfo::Clear() {
  pEntryFunc = nullptr;
  Functions.clear();
  Outputs.clear();
  for (unsigned i = 0; i < kNumStreams; i++)
    ContributingInstructions[i].clear();
}

void DxilViewIdStateBuilder::FuncInfo::Clear() {
  Returns.clear();
  CtrlDep.Clear();
  pDomTree.reset();
}

void DxilViewIdStateBuilder::DetermineMaxPackedLocation(DxilSignature &DxilSig,
                                                 unsigned *pMaxSigLoc,
                                                 unsigned NumStreams) {
  DXASSERT_NOMSG(NumStreams == 1 || NumStreams == kNumStreams);

  for (unsigned i = 0; i < NumStreams; i++) {
    pMaxSigLoc[i] = 0;
  }

  for (auto &E : DxilSig.GetElements()) {
    if (E->GetStartRow() == Semantic::kUndefinedRow) continue;

    unsigned StreamId = E->GetOutputStream();
    unsigned endLoc = GetLinearIndex(*E, E->GetRows() - 1, E->GetCols() - 1);
    pMaxSigLoc[StreamId] = std::max(pMaxSigLoc[StreamId], endLoc + 1);
    E->GetCols();
  }
}

void DxilViewIdStateBuilder::ComputeReachableFunctionsRec(CallGraph &CG, CallGraphNode *pNode, FunctionSetType &FuncSet) {
  Function *F = pNode->getFunction();
  // Accumulate only functions with bodies.
  if (F->empty()) return;
  auto itIns = FuncSet.emplace(F);
  DXASSERT_NOMSG(itIns.second);
  (void)itIns;
  for (auto it = pNode->begin(), itEnd = pNode->end(); it != itEnd; ++it) {
    CallGraphNode *pSuccNode = it->second;
    ComputeReachableFunctionsRec(CG, pSuccNode, FuncSet);
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

void DxilViewIdStateBuilder::AnalyzeFunctions(EntryInfo &Entry) {
  for (auto *F : Entry.Functions) {
    DXASSERT_NOMSG(!F->empty());

    auto itFI = m_FuncInfo.find(F);
    FuncInfo *pFuncInfo = nullptr;
    if (itFI != m_FuncInfo.end()) {
      pFuncInfo = itFI->second.get();
    } else {
      m_FuncInfo[F] = make_unique<FuncInfo>();
      pFuncInfo = m_FuncInfo[F].get();
    }

    for (auto itBB = F->begin(), endBB = F->end(); itBB != endBB; ++itBB) {
      BasicBlock *BB = itBB;

      for (auto itInst = BB->begin(), endInst = BB->end(); itInst != endInst; ++itInst) {
        if (ReturnInst *RI = dyn_cast<ReturnInst>(itInst)) {
          pFuncInfo->Returns.emplace(RI);
          continue;
        }

        CallInst *CI = dyn_cast<CallInst>(itInst);
        if (!CI) continue;

        DynamicallyIndexedElemsType *pDynIdxElems = nullptr;
        int row = Semantic::kUndefinedRow;
        unsigned id, col;
        if (DxilInst_LoadInput LI = DxilInst_LoadInput(CI)) {
          pDynIdxElems = &m_InpSigDynIdxElems;
          IFTBOOL(GetUnsignedVal(LI.get_inputSigId(), &id), DXC_E_GENERAL_INTERNAL_ERROR);
          GetUnsignedVal(LI.get_rowIndex(), (uint32_t*)&row);
          IFTBOOL(GetUnsignedVal(LI.get_colIndex(), &col), DXC_E_GENERAL_INTERNAL_ERROR);
        } else if (DxilInst_StoreOutput SO = DxilInst_StoreOutput(CI)) {
          pDynIdxElems = &m_OutSigDynIdxElems;
          IFTBOOL(GetUnsignedVal(SO.get_outputSigId(), &id), DXC_E_GENERAL_INTERNAL_ERROR);
          GetUnsignedVal(SO.get_rowIndex(), (uint32_t*)&row);
          IFTBOOL(GetUnsignedVal(SO.get_colIndex(), &col), DXC_E_GENERAL_INTERNAL_ERROR);
          Entry.Outputs.emplace(CI);
        } else if (DxilInst_StoreVertexOutput SVO = DxilInst_StoreVertexOutput(CI)) {
          pDynIdxElems = &m_OutSigDynIdxElems;
          IFTBOOL(GetUnsignedVal(SVO.get_outputSigId(), &id), DXC_E_GENERAL_INTERNAL_ERROR);
          GetUnsignedVal(SVO.get_rowIndex(), (uint32_t*)&row);
          IFTBOOL(GetUnsignedVal(SVO.get_colIndex(), &col), DXC_E_GENERAL_INTERNAL_ERROR);
          Entry.Outputs.emplace(CI);
        } else if (DxilInst_StorePrimitiveOutput SPO = DxilInst_StorePrimitiveOutput(CI)) {
          pDynIdxElems = &m_PCSigDynIdxElems;
          IFTBOOL(GetUnsignedVal(SPO.get_outputSigId(), &id), DXC_E_GENERAL_INTERNAL_ERROR);
          GetUnsignedVal(SPO.get_rowIndex(), (uint32_t*)&row);
          IFTBOOL(GetUnsignedVal(SPO.get_colIndex(), &col), DXC_E_GENERAL_INTERNAL_ERROR);
          Entry.Outputs.emplace(CI);
        } else if (DxilInst_LoadPatchConstant LPC = DxilInst_LoadPatchConstant(CI)) {
          if (m_pModule->GetShaderModel()->IsDS()) {
            pDynIdxElems = &m_PCSigDynIdxElems;
            IFTBOOL(GetUnsignedVal(LPC.get_inputSigId(), &id), DXC_E_GENERAL_INTERNAL_ERROR);
            GetUnsignedVal(LPC.get_row(), (uint32_t*)&row);
            IFTBOOL(GetUnsignedVal(LPC.get_col(), &col), DXC_E_GENERAL_INTERNAL_ERROR);
          } else {
            // Do nothing. This is an internal helper function for DXBC-2-DXIL converter.
            DXASSERT_NOMSG(m_pModule->GetShaderModel()->IsHS());
          }
        } else if (DxilInst_StorePatchConstant SPC = DxilInst_StorePatchConstant(CI)) {
          pDynIdxElems = &m_PCSigDynIdxElems;
          IFTBOOL(GetUnsignedVal(SPC.get_outputSigID(), &id), DXC_E_GENERAL_INTERNAL_ERROR);
          GetUnsignedVal(SPC.get_row(), (uint32_t*)&row);
          IFTBOOL(GetUnsignedVal(SPC.get_col(), &col), DXC_E_GENERAL_INTERNAL_ERROR);
          Entry.Outputs.emplace(CI);
        } else if (DxilInst_LoadOutputControlPoint LOCP = DxilInst_LoadOutputControlPoint(CI)) {
          if (m_pModule->GetShaderModel()->IsDS()) {
            pDynIdxElems = &m_InpSigDynIdxElems;
            IFTBOOL(GetUnsignedVal(LOCP.get_inputSigId(), &id), DXC_E_GENERAL_INTERNAL_ERROR);
            GetUnsignedVal(LOCP.get_row(), (uint32_t*)&row);
            IFTBOOL(GetUnsignedVal(LOCP.get_col(), &col), DXC_E_GENERAL_INTERNAL_ERROR);
          } else if (m_pModule->GetShaderModel()->IsHS()) {
            // Do nothings, as the information has been captured by the output signature of CP entry.
          } else {
            DXASSERT_NOMSG(false);
          }
        } else {
          continue;
        }

        // Record dynamic index usage.
        if (pDynIdxElems && row == Semantic::kUndefinedRow) {
          (*pDynIdxElems)[id] |= (1 << col);
        }
      }
    }

    // Compute dominator relation.
    pFuncInfo->pDomTree = make_unique<DominatorTreeBase<BasicBlock> >(false);
    pFuncInfo->pDomTree->recalculate(*F);
#if DXILVIEWID_DBG
    pFuncInfo->pDomTree->print(dbgs());
#endif

    // Compute postdominator relation.
    DominatorTreeBase<BasicBlock> PDR(true);
    PDR.recalculate(*F);
#if DXILVIEWID_DBG
    PDR.print(dbgs());
#endif
    // Compute control dependence.
    pFuncInfo->CtrlDep.Compute(F, PDR);
#if DXILVIEWID_DBG
    pFuncInfo->CtrlDep.print(dbgs());
#endif
  }
}

void DxilViewIdStateBuilder::CollectValuesContributingToOutputs(EntryInfo &Entry) {
  for (auto *CI : Entry.Outputs) {  // CI = call instruction
    DxilSignature *pDxilSig = nullptr;
    Value *pContributingValue = nullptr;
    unsigned id = (unsigned)-1;
    int startRow = Semantic::kUndefinedRow, endRow = Semantic::kUndefinedRow;
    unsigned col = (unsigned)-1;
    if (DxilInst_StoreOutput SO = DxilInst_StoreOutput(CI)) {
      pDxilSig = &m_pModule->GetOutputSignature();
      pContributingValue = SO.get_value();
      GetUnsignedVal(SO.get_outputSigId(), &id);
      GetUnsignedVal(SO.get_colIndex(), &col);
      GetUnsignedVal(SO.get_rowIndex(), (uint32_t*)&startRow);
    } else if (DxilInst_StoreVertexOutput SVO = DxilInst_StoreVertexOutput(CI)) {
      pDxilSig = &m_pModule->GetOutputSignature();
      pContributingValue = SVO.get_value();
      GetUnsignedVal(SVO.get_outputSigId(), &id);
      GetUnsignedVal(SVO.get_colIndex(), &col);
      GetUnsignedVal(SVO.get_rowIndex(), (uint32_t*)&startRow);
    } else if (DxilInst_StorePrimitiveOutput SPO = DxilInst_StorePrimitiveOutput(CI)) {
      pDxilSig = &m_pModule->GetPatchConstOrPrimSignature();
      pContributingValue = SPO.get_value();
      GetUnsignedVal(SPO.get_outputSigId(), &id);
      GetUnsignedVal(SPO.get_colIndex(), &col);
      GetUnsignedVal(SPO.get_rowIndex(), (uint32_t*)&startRow);
    } else if (DxilInst_StorePatchConstant SPC = DxilInst_StorePatchConstant(CI)) {
      pDxilSig = &m_pModule->GetPatchConstOrPrimSignature();
      pContributingValue = SPC.get_value();
      GetUnsignedVal(SPC.get_outputSigID(), &id);
      GetUnsignedVal(SPC.get_row(), (uint32_t*)&startRow);
      GetUnsignedVal(SPC.get_col(), &col);
    } else {
      IFT(DXC_E_GENERAL_INTERNAL_ERROR);
    }

    DxilSignatureElement &SigElem = pDxilSig->GetElement(id);
    if (!SigElem.IsAllocated())
      continue;

    unsigned StreamId = SigElem.GetOutputStream();

    if (startRow != Semantic::kUndefinedRow) {
      endRow = startRow;
    } else {
      // The entire column is affected by value.
      DXASSERT_NOMSG(SigElem.GetID() == id && SigElem.GetStartRow() != Semantic::kUndefinedRow);
      startRow = 0;
      endRow = SigElem.GetRows() - 1;
    }

    InstructionSetType ContributingInstructionsAllRows;
    InstructionSetType *pContributingInstructions = &ContributingInstructionsAllRows;
    if (startRow == endRow) {
      // Scalar or indexable with known index.
      unsigned index = GetLinearIndex(SigElem, startRow, col);
      pContributingInstructions = &Entry.ContributingInstructions[StreamId][index];
    }

    CollectValuesContributingToOutputRec(Entry, pContributingValue, *pContributingInstructions);

    // Handle control dependence of this instruction BB.
    BasicBlock *pBB = CI->getParent();
    Function *F = pBB->getParent();
    FuncInfo *pFuncInfo = m_FuncInfo[F].get();
    const BasicBlockSet &CtrlDepSet = pFuncInfo->CtrlDep.GetCDBlocks(pBB);
    for (BasicBlock *B : CtrlDepSet) {
      CollectValuesContributingToOutputRec(Entry, B->getTerminator(), *pContributingInstructions);
    }

    if (pContributingInstructions == &ContributingInstructionsAllRows) {
      // Write dynamically indexed output contributions to all rows.
      for (int row = startRow; row <= endRow; row++) {
        unsigned index = GetLinearIndex(SigElem, row, col);
        Entry.ContributingInstructions[StreamId][index].insert(ContributingInstructionsAllRows.begin(), ContributingInstructionsAllRows.end());
      }
    }
  }
}

void DxilViewIdStateBuilder::CollectValuesContributingToOutputRec(EntryInfo &Entry,
                                                           Value *pContributingValue,
                                                           InstructionSetType &ContributingInstructions) {
  if (dyn_cast<Argument>(pContributingValue)) {
    // This must be a leftover signature argument of an entry function.
    DXASSERT_NOMSG(Entry.pEntryFunc == m_pModule->GetEntryFunction() ||
                   Entry.pEntryFunc == m_pModule->GetPatchConstantFunction());
    return;
  }

  Instruction *pContributingInst = dyn_cast<Instruction>(pContributingValue);
  if (pContributingInst == nullptr) {
    // Can be literal constant, global decl, branch target.
    DXASSERT_NOMSG(isa<Constant>(pContributingValue) || isa<BasicBlock>(pContributingValue));
    return;
  }

  BasicBlock *pBB = pContributingInst->getParent();
  Function *F = pBB->getParent();
  auto FuncInfoIt = m_FuncInfo.find(F);
  DXASSERT_NOMSG(FuncInfoIt != m_FuncInfo.end());
  if (FuncInfoIt == m_FuncInfo.end()) {
    return;
  }

  auto itInst = ContributingInstructions.emplace(pContributingInst);
  // Already visited instruction.
  if (!itInst.second) return;

  // Handle special cases.
  if (PHINode *phi = dyn_cast<PHINode>(pContributingInst)) {
    CollectPhiCFValuesContributingToOutputRec(phi, Entry, ContributingInstructions);
  } else if (isa<LoadInst>(pContributingInst) || 
             isa<AtomicCmpXchgInst>(pContributingInst) ||
             isa<AtomicRMWInst>(pContributingInst)) {
    Value *pPtrValue = pContributingInst->getOperand(0);
    DXASSERT_NOMSG(pPtrValue->getType()->isPointerTy());
    const ValueSetType &ReachingDecls = CollectReachingDecls(pPtrValue);
    DXASSERT_NOMSG(ReachingDecls.size() > 0);
    for (Value *pDeclValue : ReachingDecls) {
      const ValueSetType &Stores = CollectStores(pDeclValue);
      for (Value *V : Stores) {
        CollectValuesContributingToOutputRec(Entry, V, ContributingInstructions);
      }
    }
  } else if (CallInst *CI = dyn_cast<CallInst>(pContributingInst)) {
    if (!hlsl::OP::IsDxilOpFuncCallInst(CI)) {
      Function *F = CI->getCalledFunction();
      if (!F->empty()) {
        // Return value of a user function.
        if (Entry.Functions.find(F) != Entry.Functions.end()) {
          const FuncInfo &FI = *m_FuncInfo[F];
          for (ReturnInst *pRetInst : FI.Returns) {
            CollectValuesContributingToOutputRec(Entry, pRetInst, ContributingInstructions);
          }
        }
      }
    }
  }

  // Handle instruction inputs.
  unsigned NumOps = pContributingInst->getNumOperands();
  for (unsigned i = 0; i < NumOps; i++) {
    Value *O = pContributingInst->getOperand(i);
    CollectValuesContributingToOutputRec(Entry, O, ContributingInstructions);
  }

  // Handle control dependence of this instruction BB.
  FuncInfo *pFuncInfo = FuncInfoIt->second.get();
  const BasicBlockSet &CtrlDepSet = pFuncInfo->CtrlDep.GetCDBlocks(pBB);
  for (BasicBlock *B : CtrlDepSet) {
    CollectValuesContributingToOutputRec(Entry, B->getTerminator(), ContributingInstructions);
  }
}

// Only process control-dependent basic blocks for constant operands of the phi-function.
// An obvious "definition" point for a constant operand is the predecessor along corresponding edge.
// However, this may be too conservative and, as such, pick up extra control dependent BBs.
// A better "definition" point is the highest dominator where it is still legal to "insert" constant assignment.
// In this context, "legal" means that only one value "leaves" the dominator and reaches Phi.
void DxilViewIdStateBuilder::CollectPhiCFValuesContributingToOutputRec(PHINode *pPhi,
                                                                EntryInfo &Entry,
                                                                InstructionSetType &ContributingInstructions) {
  Function *F = pPhi->getParent()->getParent();
  FuncInfo *pFuncInfo = m_FuncInfo[F].get();
  unordered_map<DomTreeNodeBase<BasicBlock> *, Value *> DomTreeMarkers;

  // Mark predecessors of each value, so that there is a legal "definition" point.
  for (unsigned i = 0; i < pPhi->getNumOperands(); i++) {
    Value *pValue = pPhi->getIncomingValue(i);
    BasicBlock *pBB = pPhi->getIncomingBlock(i);
    DomTreeNodeBase<BasicBlock> *pDomNode = pFuncInfo->pDomTree->getNode(pBB);
    auto it = DomTreeMarkers.emplace(pDomNode, pValue);
    DXASSERT_NOMSG(it.second || it.first->second == pValue); (void)it;
  }
  // Mark the dominator tree with "definition" values, walking up to the parent.
  for (unsigned i = 0; i < pPhi->getNumOperands(); i++) {
    Value *pValue = pPhi->getIncomingValue(i);
    BasicBlock *pDefBB = &F->getEntryBlock();
    if (Instruction *pDefInst = dyn_cast<Instruction>(pValue)) {
      pDefBB = pDefInst->getParent();
    }

    BasicBlock *pBB = pPhi->getIncomingBlock(i);
    if (pBB == pDefBB) {
      continue; // we already handled the predecessor.
    }
    DomTreeNodeBase<BasicBlock> *pDomNode = pFuncInfo->pDomTree->getNode(pBB);
    pDomNode = pDomNode->getIDom();
    while (pDomNode) {
      auto it = DomTreeMarkers.emplace(pDomNode, pValue);
      if (!it.second) {
        if (it.first->second != pValue && it.first->second != nullptr) {
          if (!isa<Constant>(it.first->second) || !isa<Constant>(pValue)) {
            // Unless both are different constants, mark the "definition" point as illegal.
            it.first->second = nullptr;
            // If both are constants, leave the marker of the first one.
          }
        }
        break;
      }

      // Do not go higher than a legal definition point.
      pBB = pDomNode->getBlock();
      if (pBB == pDefBB)
        break;

      pDomNode = pDomNode->getIDom();
    }
  }

  // Handle control dependence for Constant arguments of Phi.
  for (unsigned i = 0; i < pPhi->getNumOperands(); i++) {
    Value *pValue = pPhi->getIncomingValue(i);
    if (!isa<Constant>(pValue))
      continue;

    // Determine the higher legal "definition" point.
    BasicBlock *pBB = pPhi->getIncomingBlock(i);
    DomTreeNodeBase<BasicBlock> *pDomNode = pFuncInfo->pDomTree->getNode(pBB);
    DomTreeNodeBase<BasicBlock> *pDefDomNode = pDomNode;
    while (pDomNode) {
      auto it = DomTreeMarkers.find(pDomNode);
      DXASSERT_NOMSG(it != DomTreeMarkers.end());
      if (it->second != pValue) {
        DXASSERT_NOMSG(it->second == nullptr || isa<Constant>(it->second));
        break;
      }

      pDefDomNode = pDomNode;
      pDomNode = pDomNode->getIDom();
    }

    // Handle control dependence of this constant argument highest legal "definition" point.
    pBB = pDefDomNode->getBlock();
    const BasicBlockSet &CtrlDepSet = pFuncInfo->CtrlDep.GetCDBlocks(pBB);
    for (BasicBlock *B : CtrlDepSet) {
      CollectValuesContributingToOutputRec(Entry, B->getTerminator(), ContributingInstructions);
    }
  }
}

const DxilViewIdStateBuilder::ValueSetType &DxilViewIdStateBuilder::CollectReachingDecls(Value *pValue) {
  auto it = m_ReachingDeclsCache.emplace(pValue, ValueSetType());
  if (it.second) {
    // We have not seen this value before.
    ValueSetType Visited;
    CollectReachingDeclsRec(pValue, it.first->second, Visited);
  }
  return it.first->second;
}

void DxilViewIdStateBuilder::CollectReachingDeclsRec(Value *pValue, ValueSetType &ReachingDecls, ValueSetType &Visited) {
  if (Visited.find(pValue) != Visited.end())
    return;

  bool bInitialValue = Visited.size() == 0;
  Visited.emplace(pValue);

  if (!bInitialValue) {
    auto it = m_ReachingDeclsCache.find(pValue);
    if (it != m_ReachingDeclsCache.end()) {
      ReachingDecls.insert(it->second.begin(), it->second.end());
      return;
    }
  }

  if (dyn_cast<GlobalVariable>(pValue)) {
    ReachingDecls.emplace(pValue);
    return;
  }

  if (GetElementPtrInst *pGepInst = dyn_cast<GetElementPtrInst>(pValue)) {
    Value *pPtrValue = pGepInst->getPointerOperand();
    CollectReachingDeclsRec(pPtrValue, ReachingDecls, Visited);
  } else if (GEPOperator *pGepOp = dyn_cast<GEPOperator>(pValue)) {
    Value *pPtrValue = pGepOp->getPointerOperand();
    CollectReachingDeclsRec(pPtrValue, ReachingDecls, Visited);
  } else if (isa<ConstantExpr>(pValue) && cast<ConstantExpr>(pValue)->getOpcode() == Instruction::AddrSpaceCast) {
    CollectReachingDeclsRec(cast<ConstantExpr>(pValue)->getOperand(0), ReachingDecls, Visited);
  } else if (AddrSpaceCastInst *pCI = dyn_cast<AddrSpaceCastInst>(pValue)) {
    CollectReachingDeclsRec(pCI->getOperand(0), ReachingDecls, Visited);
  } else if (BitCastInst *pCI = dyn_cast<BitCastInst>(pValue)) {
    CollectReachingDeclsRec(pCI->getOperand(0), ReachingDecls, Visited);
  } else if (dyn_cast<AllocaInst>(pValue)) {
    ReachingDecls.emplace(pValue);
  } else if (PHINode *phi = dyn_cast<PHINode>(pValue)) {
    for (Value *pPtrValue : phi->operands()) {
      CollectReachingDeclsRec(pPtrValue, ReachingDecls, Visited);
    }
  } else if (SelectInst *SelI = dyn_cast<SelectInst>(pValue)) {
    CollectReachingDeclsRec(SelI->getTrueValue(), ReachingDecls, Visited);
    CollectReachingDeclsRec(SelI->getFalseValue(), ReachingDecls, Visited);
  } else if (dyn_cast<Argument>(pValue)) {
    ReachingDecls.emplace(pValue);
  } else if (CallInst *call = dyn_cast<CallInst>(pValue)) {
    DXASSERT(OP::GetDxilOpFuncCallInst(call) == DXIL::OpCode::GetMeshPayload,
             "the function must be @dx.op.getMeshPayload here.");
    ReachingDecls.emplace(pValue);
  } else {
    IFT(DXC_E_GENERAL_INTERNAL_ERROR);
  }
}

const DxilViewIdStateBuilder::ValueSetType &DxilViewIdStateBuilder::CollectStores(llvm::Value *pValue) {
  auto it = m_StoresPerDeclCache.emplace(pValue, ValueSetType());
  if (it.second) {
    // We have not seen this value before.
    ValueSetType Visited;
    CollectStoresRec(pValue, it.first->second, Visited);
  }
  return it.first->second;
}

void DxilViewIdStateBuilder::CollectStoresRec(llvm::Value *pValue, ValueSetType &Stores, ValueSetType &Visited) {
  if (Visited.find(pValue) != Visited.end())
    return;

  bool bInitialValue = Visited.size() == 0;
  Visited.emplace(pValue);

  if (!bInitialValue) {
    auto it = m_StoresPerDeclCache.find(pValue);
    if (it != m_StoresPerDeclCache.end()) {
      Stores.insert(it->second.begin(), it->second.end());
      return;
    }
  }

  if (isa<LoadInst>(pValue)) {
    return;
  } else if (isa<StoreInst>(pValue) ||
             isa<AtomicCmpXchgInst>(pValue) ||
             isa<AtomicRMWInst>(pValue)) {
    Stores.emplace(pValue);
    return;
  }

  for (auto *U : pValue->users()) {
    CollectStoresRec(U, Stores, Visited);
  }
}

void DxilViewIdStateBuilder::CreateViewIdSets(const std::unordered_map<unsigned, InstructionSetType> &ContributingInstructions, 
                                       OutputsDependentOnViewIdType &OutputsDependentOnViewId,
                                       InputsContributingToOutputType &InputsContributingToOutputs,
                                       bool bPC) {
  const ShaderModel *pSM = m_pModule->GetShaderModel();

  for (auto &itOut : ContributingInstructions) {
    unsigned outIdx = itOut.first;
    for (Instruction *pInst : itOut.second) {
      // Set output dependence on ViewId.
      if (DxilInst_ViewID VID = DxilInst_ViewID(pInst)) {
        DXASSERT(m_bUsesViewId, "otherwise, DxilModule flag not set properly");
        OutputsDependentOnViewId[outIdx] = true;
        continue;
      }

      // Start setting output dependence on inputs.
      DxilSignatureElement *pSigElem = nullptr;
      bool bLoadOutputCPInHS = false;
      unsigned inpId = (unsigned)-1;
      int startRow = Semantic::kUndefinedRow, endRow = Semantic::kUndefinedRow;
      unsigned col = (unsigned)-1;
      if (DxilInst_LoadInput LI = DxilInst_LoadInput(pInst)) {
        GetUnsignedVal(LI.get_inputSigId(), &inpId);
        GetUnsignedVal(LI.get_colIndex(), &col);
        GetUnsignedVal(LI.get_rowIndex(), (uint32_t*)&startRow);
        pSigElem = &m_pModule->GetInputSignature().GetElement(inpId);
        if (pSM->IsDS() && bPC) {
          pSigElem = nullptr;
        }
      } else if (DxilInst_LoadOutputControlPoint LOCP = DxilInst_LoadOutputControlPoint(pInst)) {
        GetUnsignedVal(LOCP.get_inputSigId(), &inpId);
        GetUnsignedVal(LOCP.get_col(), &col);
        GetUnsignedVal(LOCP.get_row(), (uint32_t*)&startRow);
        if (pSM->IsHS()) {
          pSigElem = &m_pModule->GetOutputSignature().GetElement(inpId);
          bLoadOutputCPInHS = true;
        } else if (pSM->IsDS()) {
          if (!bPC) {
            pSigElem = &m_pModule->GetInputSignature().GetElement(inpId);
          }
        } else {
          DXASSERT_NOMSG(false);
        }
      } else if (DxilInst_LoadPatchConstant LPC = DxilInst_LoadPatchConstant(pInst)) {
        if (pSM->IsDS() && bPC) {
          GetUnsignedVal(LPC.get_inputSigId(), &inpId);
          GetUnsignedVal(LPC.get_col(), &col);
          GetUnsignedVal(LPC.get_row(), (uint32_t*)&startRow);
          pSigElem = &m_pModule->GetPatchConstOrPrimSignature().GetElement(inpId);
        }
      } else {
        continue;
      }

      // Finalize setting output dependence on inputs.
      if (pSigElem && pSigElem->IsAllocated()) {
        if (startRow != Semantic::kUndefinedRow) {
          endRow = startRow;
        } else {
          // The entire column contributes to output.
          startRow = 0;
          endRow = pSigElem->GetRows() - 1;
        }

        auto &ContributingInputs = InputsContributingToOutputs[outIdx];
        for (int row = startRow; row <= endRow; row++) {
          unsigned index = GetLinearIndex(*pSigElem, row, col);
          if (!bLoadOutputCPInHS) {
            ContributingInputs.emplace(index);
          } else {
            // This HS patch-constant output depends on an input value of LoadOutputControlPoint
            // that is the output value of the HS main (control-point) function.
            // Transitively update this (patch-constant) output dependence on main (control-point) output.
            DXASSERT_NOMSG(&OutputsDependentOnViewId == &m_PCOrPrimOutputsDependentOnViewId);
            OutputsDependentOnViewId[outIdx] = OutputsDependentOnViewId[outIdx] || m_OutputsDependentOnViewId[0][index];

            const auto it = m_InputsContributingToOutputs[0].find(index);
            if (it != m_InputsContributingToOutputs[0].end()) {
              const std::set<unsigned> &LoadOutputCPInputsContributingToOutputs = it->second;
              ContributingInputs.insert(LoadOutputCPInputsContributingToOutputs.begin(),
                                        LoadOutputCPInputsContributingToOutputs.end());
            }
          }
        }
      }
    }
  }
}

unsigned DxilViewIdStateBuilder::GetLinearIndex(DxilSignatureElement &SigElem, int row, unsigned col) const {
  DXASSERT_NOMSG(row >= 0 && col < kNumComps && SigElem.GetStartRow() != Semantic::kUndefinedRow);
  unsigned idx = (((unsigned)row) + SigElem.GetStartRow())*kNumComps + col + SigElem.GetStartCol();
  DXASSERT_NOMSG(idx < kMaxSigScalars); (void)kMaxSigScalars;
  return idx;
}

void DxilViewIdStateBuilder::UpdateDynamicIndexUsageState() const {
  UpdateDynamicIndexUsageStateForSig(m_pModule->GetInputSignature(), m_InpSigDynIdxElems);
  UpdateDynamicIndexUsageStateForSig(m_pModule->GetOutputSignature(), m_OutSigDynIdxElems);
  UpdateDynamicIndexUsageStateForSig(m_pModule->GetPatchConstOrPrimSignature(), m_PCSigDynIdxElems);
}

void DxilViewIdStateBuilder::UpdateDynamicIndexUsageStateForSig(DxilSignature &Sig,
                                                         const DynamicallyIndexedElemsType &DynIdxElems) const {
  for (auto it : DynIdxElems) {
    unsigned id = it.first;
    unsigned mask = it.second;
    DxilSignatureElement &E = Sig.GetElement(id);
    E.SetDynIdxCompMask(mask);
  }
}



namespace {
class ComputeViewIdState : public ModulePass {
public:
  static char ID; // Pass ID, replacement for typeid

  ComputeViewIdState();

  bool runOnModule(Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;
};
} // namespace

char ComputeViewIdState::ID = 0;

INITIALIZE_PASS_BEGIN(ComputeViewIdState, "viewid-state",
                "Compute information related to ViewID", true, true)
INITIALIZE_PASS_END(ComputeViewIdState, "viewid-state",
                "Compute information related to ViewID", true, true)

ComputeViewIdState::ComputeViewIdState() : ModulePass(ID) {
}

bool ComputeViewIdState::runOnModule(Module &M) {
  DxilModule &DxilModule = M.GetOrCreateDxilModule();
  const ShaderModel *pSM = DxilModule.GetShaderModel();
  if (!pSM->IsCS() && !pSM->IsLib()) {
    DxilViewIdState ViewIdState(&DxilModule);
    DxilViewIdStateBuilder Builder(ViewIdState, &DxilModule);
    Builder.Compute();
    // Serialize viewidstate.
    ViewIdState.Serialize();
    auto &TmpSerialized = ViewIdState.GetSerialized();
    // Copy serilized viewidstate.
    auto &SerializedViewIdState = DxilModule.GetSerializedViewIdState();
    SerializedViewIdState.clear();
    SerializedViewIdState.resize(TmpSerialized.size());
    SerializedViewIdState.assign(TmpSerialized.begin(), TmpSerialized.end());
    return true;
  }
  return false;
}

void ComputeViewIdState::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
}


namespace llvm {

ModulePass *createComputeViewIdStatePass() {
  return new ComputeViewIdState();
}

} // end of namespace llvm
