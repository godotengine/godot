///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilCondenseResources.cpp                                                 //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Provides a pass to make resource IDs zero-based and dense.                //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/HLSL/DxilGenerationPass.h"
#include "dxc/DXIL/DxilOperations.h"
#include "dxc/DXIL/DxilSignatureElement.h"
#include "dxc/DXIL/DxilModule.h"
#include "dxc/Support/Global.h"
#include "dxc/DXIL/DxilTypeSystem.h"
#include "dxc/DXIL/DxilInstructions.h"
#include "dxc/DXIL/DxilResourceBinding.h"
#include "dxc/HLSL/DxilSpanAllocator.h"
#include "dxc/HLSL/HLMatrixType.h"
#include "dxc/DXIL/DxilUtil.h"
#include "dxc/HLSL/HLMatrixType.h"
#include "dxc/HLSL/HLModule.h"
#include "dxc/DxcBindingTable/DxcBindingTable.h"
#include "llvm/Analysis/DxilValueCache.h"
#include "dxc/DXIL/DxilMetadataHelper.h"

#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/Local.h"
#include <memory>
#include <unordered_set>

using namespace llvm;
using namespace hlsl;

// Resource rangeID remap.
namespace {
struct ResourceID {
  DXIL::ResourceClass Class; // Resource class.
  unsigned ID;               // Resource ID, as specified on entry.

  bool operator<(const ResourceID &other) const {
    if (Class < other.Class)
      return true;
    if (Class > other.Class)
      return false;
    if (ID < other.ID)
      return true;
    return false;
  }
};

struct RemapEntry {
  ResourceID ResID;           // Resource identity, as specified on entry.
  DxilResourceBase *Resource; // In-memory resource representation.
  unsigned Index; // Index in resource vector - new ID for the resource.
};

typedef std::map<ResourceID, RemapEntry> RemapEntryCollection;

template <typename TResource>
void BuildRewrites(const std::vector<std::unique_ptr<TResource>> &Rs,
                   RemapEntryCollection &C) {
  const unsigned s = (unsigned)Rs.size();
  for (unsigned i = 0; i < s; ++i) {
    const std::unique_ptr<TResource> &R = Rs[i];
    if (R->GetID() != i) {
      ResourceID RId = {R->GetClass(), R->GetID()};
      RemapEntry RE = {RId, R.get(), i};
      C[RId] = RE;
    }
  }
}

// Build m_rewrites, returns 'true' if any rewrites are needed.
bool BuildRewriteMap(RemapEntryCollection &rewrites, DxilModule &DM) {
  BuildRewrites(DM.GetCBuffers(), rewrites);
  BuildRewrites(DM.GetSRVs(), rewrites);
  BuildRewrites(DM.GetUAVs(), rewrites);
  BuildRewrites(DM.GetSamplers(), rewrites);

  return !rewrites.empty();
}

} // namespace

class DxilResourceRegisterAllocator {
private:
  SpacesAllocator<unsigned, hlsl::DxilCBuffer> m_reservedCBufferRegisters;
  SpacesAllocator<unsigned, hlsl::DxilSampler> m_reservedSamplerRegisters;
  SpacesAllocator<unsigned, hlsl::DxilResource> m_reservedUAVRegisters;
  SpacesAllocator<unsigned, hlsl::DxilResource> m_reservedSRVRegisters;

  template<typename T>
  static void GatherReservedRegisters(
    const std::vector<std::unique_ptr<T>> &ResourceList,
    SpacesAllocator<unsigned, T> &SAlloc) {
    for (auto &res : ResourceList) {
      if (res->IsAllocated()) {
        typename SpacesAllocator<unsigned, T>::Allocator &Alloc = SAlloc.Get(res->GetSpaceID());
        Alloc.ForceInsertAndClobber(res.get(), res->GetLowerBound(), res->GetUpperBound());
        if (res->IsUnbounded())
          Alloc.SetUnbounded(res.get());
      }
    }
  }

  template <typename T>
  static bool
  AllocateRegisters(LLVMContext &Ctx, const std::vector<std::unique_ptr<T>> &resourceList,
    SpacesAllocator<unsigned, T> &ReservedRegisters,
    unsigned AutoBindingSpace) {
    bool bChanged = false;
    SpacesAllocator<unsigned, T> SAlloc;

    // Reserve explicitly allocated resources
    for (auto &res : resourceList) {
      const unsigned space = res->GetSpaceID();
      typename SpacesAllocator<unsigned, T>::Allocator &alloc = SAlloc.Get(space);
      typename SpacesAllocator<unsigned, T>::Allocator &reservedAlloc = ReservedRegisters.Get(space);

      if (res->IsAllocated()) {
        const unsigned reg = res->GetLowerBound();
        const T *conflict = nullptr;
        if (res->IsUnbounded()) {
          const T *unbounded = alloc.GetUnbounded();
          if (unbounded) {
            dxilutil::EmitErrorOnGlobalVariable(Ctx, dyn_cast<GlobalVariable>(res->GetGlobalSymbol()),
                                                Twine("more than one unbounded resource (") +
                                                unbounded->GetGlobalName() + (" and ") +
                                                res->GetGlobalName() + (") in space ") + Twine(space));
          }
          else {
            conflict = alloc.Insert(res.get(), reg, res->GetUpperBound());
            if (!conflict) {
              alloc.SetUnbounded(res.get());
              reservedAlloc.SetUnbounded(res.get());
            }
          }
        }
        else {
          conflict = alloc.Insert(res.get(), reg, res->GetUpperBound());
        }
        if (conflict) {
          dxilutil::EmitErrorOnGlobalVariable(Ctx, dyn_cast<GlobalVariable>(res->GetGlobalSymbol()), 
                                              ((res->IsUnbounded()) ? Twine("unbounded ") : Twine("")) +
                                              Twine("resource ") + res->GetGlobalName() +
                                              Twine(" at register ") + Twine(reg) +
                                              Twine(" overlaps with resource ") +
                                              conflict->GetGlobalName() + Twine(" at register ") +
                                              Twine(conflict->GetLowerBound()) + Twine(", space ") +
                                              Twine(space));
        }
        else {
          // Also add this to the reserved (unallocatable) range, if it wasn't already there.
          reservedAlloc.ForceInsertAndClobber(res.get(), res->GetLowerBound(), res->GetUpperBound());
        }
      }
    }

    // Allocate unallocated resources
    for (auto &res : resourceList) {
      if (res->IsAllocated())
        continue;

      unsigned space = res->GetSpaceID();
      if (space == UINT_MAX) space = AutoBindingSpace;
      typename SpacesAllocator<unsigned, T>::Allocator& alloc = SAlloc.Get(space);
      typename SpacesAllocator<unsigned, T>::Allocator& reservedAlloc = ReservedRegisters.Get(space);

      unsigned reg = 0;
      unsigned end = 0;
      bool allocateSpaceFound = false;
      if (res->IsUnbounded()) {
        if (alloc.GetUnbounded() != nullptr) {
          const T *unbounded = alloc.GetUnbounded();
          dxilutil::EmitErrorOnGlobalVariable(Ctx, dyn_cast<GlobalVariable>(res->GetGlobalSymbol()),
                                              Twine("more than one unbounded resource (") +
                                              unbounded->GetGlobalName() + Twine(" and ") +
                                              res->GetGlobalName() + Twine(") in space ") +
                                              Twine(space));
          continue;
        }

        if (reservedAlloc.FindForUnbounded(reg)) {
          end = UINT_MAX;
          allocateSpaceFound = true;
        }
      }
      else if (reservedAlloc.Find(res->GetRangeSize(), reg)) {
        end = reg + res->GetRangeSize() - 1;
        allocateSpaceFound = true;
      }

      if (allocateSpaceFound) {
        bool success = reservedAlloc.Insert(res.get(), reg, end) == nullptr;
        DXASSERT_NOMSG(success);

        success = alloc.Insert(res.get(), reg, end) == nullptr;
        DXASSERT_NOMSG(success);

        if (res->IsUnbounded()) {
          alloc.SetUnbounded(res.get());
          reservedAlloc.SetUnbounded(res.get());
        }

        res->SetLowerBound(reg);
        res->SetSpaceID(space);
        bChanged = true;
      } else {
        dxilutil::EmitErrorOnGlobalVariable(Ctx, dyn_cast<GlobalVariable>(res->GetGlobalSymbol()),
                                            ((res->IsUnbounded()) ? Twine("unbounded ") : Twine("")) +
                                            Twine("resource ") + res->GetGlobalName() +
                                            Twine(" could not be allocated"));
      }
    }

    return bChanged;
  }

public:
  void GatherReservedRegisters(DxilModule &DM) {
    // For backcompat with FXC, shader models 5.0 and below will not auto-allocate
    // resources at a register explicitely assigned to even an unused resource.
    if (DM.GetLegacyResourceReservation()) {
      GatherReservedRegisters(DM.GetCBuffers(), m_reservedCBufferRegisters);
      GatherReservedRegisters(DM.GetSamplers(), m_reservedSamplerRegisters);
      GatherReservedRegisters(DM.GetUAVs(), m_reservedUAVRegisters);
      GatherReservedRegisters(DM.GetSRVs(), m_reservedSRVRegisters);
    }
  }

  bool AllocateRegisters(DxilModule &DM) {
    uint32_t AutoBindingSpace = DM.GetAutoBindingSpace();
    if (AutoBindingSpace == UINT_MAX) {
      // For libraries, we don't allocate unless AutoBindingSpace is set.
      if (DM.GetShaderModel()->IsLib())
        return false;
      // For shaders, we allocate in space 0 by default.
      AutoBindingSpace = 0;
    }

    bool bChanged = false;
    bChanged |= AllocateRegisters(DM.GetCtx(), DM.GetCBuffers(), m_reservedCBufferRegisters, AutoBindingSpace);
    bChanged |= AllocateRegisters(DM.GetCtx(), DM.GetSamplers(), m_reservedSamplerRegisters, AutoBindingSpace);
    bChanged |= AllocateRegisters(DM.GetCtx(), DM.GetUAVs(), m_reservedUAVRegisters, AutoBindingSpace);
    bChanged |= AllocateRegisters(DM.GetCtx(), DM.GetSRVs(), m_reservedSRVRegisters, AutoBindingSpace);
    return bChanged;
  }
};

bool llvm::AreDxilResourcesDense(llvm::Module *M, hlsl::DxilResourceBase **ppNonDense) {
  DxilModule &DM = M->GetOrCreateDxilModule();
  RemapEntryCollection rewrites;
  if (BuildRewriteMap(rewrites, DM)) {
    *ppNonDense = rewrites.begin()->second.Resource;
    return false;
  }
  else {
    *ppNonDense = nullptr;
    return true;
  }
}

static bool GetConstantLegalGepForSplitAlloca(GetElementPtrInst *gep, DxilValueCache *DVC, int64_t *ret) {
  if (gep->getNumIndices() != 2) {
    return false;
  }

  if (ConstantInt *Index0 = dyn_cast<ConstantInt>(gep->getOperand(1))) {
    if (Index0->getLimitedValue() != 0) {
      return false;
    }
  }
  else {
    return false;
  }

  if (ConstantInt *C = DVC->GetConstInt(gep->getOperand(2))) {
    int64_t index = C->getSExtValue();
    *ret = index;
    return true;
  }

  return false;
}

static bool LegalizeResourceArrays(Module &M, DxilValueCache *DVC) {
  SmallVector<AllocaInst *,16> Allocas;

  bool Changed = false;

  // Find all allocas
  for (Function &F : M) {
    if (F.empty())
      continue;

    BasicBlock &BB = F.getEntryBlock();
    for (Instruction &I : BB) {
      if (AllocaInst *AI = dyn_cast<AllocaInst>(&I)) {
        Type *ty = AI->getAllocatedType();
        // Only handle single dimentional array. Since this pass runs after MultiDimArrayToOneDimArray,
        // it should handle all arrays.
        if (ty->isArrayTy() && hlsl::dxilutil::IsHLSLResourceType(ty->getArrayElementType()))
          Allocas.push_back(AI);
      }
    }
  }

  SmallVector<AllocaInst *,16> ScalarAllocas;
  std::unordered_map<GetElementPtrInst *, int64_t> ConstIndices;

  for (AllocaInst *AI : Allocas) {
    Type *ty = AI->getAllocatedType();
    Type *resType = ty->getArrayElementType();

    ScalarAllocas.clear();
    ConstIndices.clear();

    bool SplitAlloca = true;

    for (User *U : AI->users()) {
      if (GetElementPtrInst *gep = dyn_cast<GetElementPtrInst>(U)) {
        int64_t index = 0;
        if (!GetConstantLegalGepForSplitAlloca(gep, DVC, &index)) {
          SplitAlloca = false;
          break;
        }

        // Out of bounds. Out of bounds GEP's will trigger and error later.
        if (index < 0 || index >= (int64_t)ty->getArrayNumElements()) {
          SplitAlloca = false;
          Changed = true;
          dxilutil::EmitErrorOnInstruction(gep, "Accessing resource array with out-out-bounds index.");
        }
        ConstIndices[gep] = index;
      }
      else {
        SplitAlloca = false;
        break;
      }
    }

    if (SplitAlloca) {

      IRBuilder<> B(AI);
      ScalarAllocas.resize(ty->getArrayNumElements());

      for (auto it = AI->user_begin(),end = AI->user_end(); it != end;) {
        GetElementPtrInst *gep = cast<GetElementPtrInst>(*(it++));
        assert(ConstIndices.count(gep));
        int64_t idx = ConstIndices[gep];

        AllocaInst *ScalarAI = ScalarAllocas[idx];
        if (!ScalarAI) {
          ScalarAI = B.CreateAlloca(resType);
          ScalarAllocas[idx] = ScalarAI;
        }

        gep->replaceAllUsesWith(ScalarAI);
        gep->eraseFromParent();
      }

      AI->eraseFromParent();

      Changed = true;
    }
  }

  return Changed;
}

typedef std::unordered_map<std::string, DxilResourceBase *> ResourceMap;
template<typename T>
static inline void GatherResources(const std::vector<std::unique_ptr<T> > &List, ResourceMap *Map) {
  for (const std::unique_ptr<T> &ptr : List) {
    (*Map)[ptr->GetGlobalName()] = ptr.get();
  }
}

static bool LegalizeResources(Module &M, DxilValueCache *DVC) {

  bool Changed = false;

  Changed |= LegalizeResourceArrays(M, DVC);

  // Simple pass to collect resource PHI's
  SmallVector<PHINode *, 8> PHIs;

  for (Function &F : M) {
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (PHINode *PN = dyn_cast<PHINode>(&I)) {
          if (hlsl::dxilutil::IsHLSLResourceType(PN->getType())) {
            PHIs.push_back(PN);
          }
        }
        else {
          break;
        }
      }
    }
  }


  SmallVector<Instruction *, 8> DCEWorklist;

  // Try to simplify those PHI's with DVC and collect them in DCEWorklist
  for (unsigned Attempt = 0, MaxAttempt = PHIs.size(); Attempt < MaxAttempt; Attempt++) {
    bool LocalChanged = false;
    for (unsigned i = 0; i < PHIs.size(); i++) {
      PHINode *PN = PHIs[i];
      if (Value *V = DVC->GetValue(PN)) {
        PN->replaceAllUsesWith(V);
        LocalChanged = true;
        DCEWorklist.push_back(PN);
        PHIs.erase(PHIs.begin() + i);
      }
      else {
        i++;
      }
    }

    Changed |= LocalChanged;
    if (!LocalChanged)
      break;
  }

  // Collect Resource GV loads
  for (GlobalVariable &GV : M.globals()) {
    Type *Ty = GV.getType()->getPointerElementType();
    while (Ty->isArrayTy())
      Ty = Ty->getArrayElementType();
    if (!hlsl::dxilutil::IsHLSLResourceType(Ty))
      continue;

    SmallVector<User *, 4> WorkList(GV.user_begin(), GV.user_end());
    while (WorkList.size()) {
      User *U = WorkList.pop_back_val();
      if (LoadInst *Load = dyn_cast<LoadInst>(U)) {
        DCEWorklist.push_back(Load);
      }
      else if (GEPOperator *GEP = dyn_cast<GEPOperator>(U)) {
        for (User *GepU : GEP->users())
          WorkList.push_back(GepU);
      } 
    }
  }

  // Simple DCE
  while (DCEWorklist.size()) {
    Instruction *I = DCEWorklist.back();
    DCEWorklist.pop_back();
    if (llvm::isInstructionTriviallyDead(I)) {
      for (Use &Op : I->operands())
        if (Instruction *OpI = dyn_cast<Instruction>(Op.get()))
          DCEWorklist.push_back(OpI);
      I->eraseFromParent();
      // Remove the instruction from the worklist if it still exists in it.
      DCEWorklist.erase(std::remove(DCEWorklist.begin(), DCEWorklist.end(), I),
                     DCEWorklist.end());
      Changed = true;
    }
  }

  return Changed;
}

namespace {
class DxilLowerCreateHandleForLib : public ModulePass {
private:
  RemapEntryCollection m_rewrites;
  DxilModule *m_DM;
  bool m_HasDbgInfo;
  bool m_bIsLib;
  bool m_bLegalizationFailed;
public:
  static char ID; // Pass identification, replacement for typeid
  explicit DxilLowerCreateHandleForLib() : ModulePass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DxilValueCache>();
  }

  StringRef getPassName() const override {
    return "DXIL Lower createHandleForLib";
  }

  bool runOnModule(Module &M) override {
    DxilModule &DM = M.GetOrCreateDxilModule();
    m_DM = &DM;
    // Clear llvm used to remove unused resource.
    m_DM->ClearLLVMUsed();
    m_bIsLib = DM.GetShaderModel()->IsLib();
    m_bLegalizationFailed = false;

    FailOnPoisonResources();

    bool bChanged = false;
    if (DM.GetShaderModel()->IsSM66Plus()) {
      bChanged = PatchDynamicTBuffers(DM);
      SetNonUniformIndexForDynamicResource(DM);
    }

    unsigned numResources = DM.GetCBuffers().size() + DM.GetUAVs().size() +
                            DM.GetSRVs().size() + DM.GetSamplers().size();

    if (!numResources) {
      // Remove createHandleFromHandle when not a lib
      if (!m_bIsLib)
        RemoveCreateHandleFromHandle(DM);
      return false;
    }
    // Switch tbuffers to SRVs, as they have been treated as cbuffers up to this
    // point.
    if (DM.GetCBuffers().size())
      bChanged |= PatchTBuffers(DM);

    // Assign resource binding overrides.
    hlsl::ApplyBindingTableFromMetadata(DM);

    // Gather reserved resource registers while we still have
    // unused resources that might have explicit register assignments.
    DxilResourceRegisterAllocator ResourceRegisterAllocator;
    ResourceRegisterAllocator.GatherReservedRegisters(DM);

    // Remove unused resources.
    DM.RemoveResourcesWithUnusedSymbols();

    unsigned newResources = DM.GetCBuffers().size() + DM.GetUAVs().size() +
                            DM.GetSRVs().size() + DM.GetSamplers().size();
    bChanged = bChanged || (numResources != newResources);

    if (0 == newResources)
      return bChanged;

    {
      DxilValueCache *DVC = &getAnalysis<DxilValueCache>();
      bool bLocalChanged = LegalizeResources(M, DVC);
      if (bLocalChanged) {
        // Remove unused resources.
        DM.RemoveResourcesWithUnusedSymbols();
      }
      bChanged |= bLocalChanged;
    }

    bChanged |= ResourceRegisterAllocator.AllocateRegisters(DM);

    // Fill in top-level CBuffer variable usage bit
    UpdateCBufferUsage();

    if (m_bIsLib && DM.GetShaderModel()->GetMinor() == ShaderModel::kOfflineMinor)
      return bChanged;

    // Make sure no select on resource.
    bChanged |= RemovePhiOnResource();

    if (m_bLegalizationFailed)
      return bChanged;

    if (m_bIsLib) {
      if (DM.GetOP()->UseMinPrecision())
        bChanged |= UpdateStructTypeForLegacyLayout();

      return bChanged;
    }

    bChanged = true;

    // Load up debug information, to cross-reference values and the instructions
    // used to load them.
    m_HasDbgInfo = llvm::getDebugMetadataVersionFromModule(M) != 0;

    GenerateDxilResourceHandles();

    if (DM.GetOP()->UseMinPrecision())
      UpdateStructTypeForLegacyLayout();

    // Change resource symbol into undef.
    UpdateResourceSymbols();

    // Remove createHandleFromHandle when not a lib.
    RemoveCreateHandleFromHandle(DM);

    // Remove unused createHandleForLib functions.
    dxilutil::RemoveUnusedFunctions(M, DM.GetEntryFunction(),
                                    DM.GetPatchConstantFunction(), m_bIsLib);

    // Erase type annotations for structures no longer used
    DM.GetTypeSystem().EraseUnusedStructAnnotations();

    return bChanged;
  }

private:
  void FailOnPoisonResources();
  bool RemovePhiOnResource();
  void UpdateResourceSymbols();
  void ReplaceResourceUserWithHandle(DxilResource &res, LoadInst *load,
                                     Instruction *handle);
  void TranslateDxilResourceUses(DxilResourceBase &res);
  void GenerateDxilResourceHandles();
  bool UpdateStructTypeForLegacyLayout();
  // Switch CBuffer for SRV for TBuffers.
  bool PatchDynamicTBuffers(DxilModule &DM);
  bool PatchTBuffers(DxilModule &DM);
  void PatchTBufferUse(Value *V, DxilModule &DM, DenseSet<Value *> &patchedSet);
  void UpdateCBufferUsage();
  void SetNonUniformIndexForDynamicResource(DxilModule &DM);
  void RemoveCreateHandleFromHandle(DxilModule &DM);
};

} // namespace

// Phi on resource.
namespace {

typedef std::unordered_map<Value*, Value*> ValueToValueMap;
typedef llvm::SetVector<Value*> ValueSetVector;
typedef llvm::SmallVector<Value*, 4> IndexVector;
typedef std::unordered_map<Value*, IndexVector> ValueToIdxMap;

//#define SUPPORT_SELECT_ON_ALLOCA

// Errors:
class ResourceUseErrors
{
  bool m_bErrorsReported;
public:
  ResourceUseErrors() : m_bErrorsReported(false) {}

  enum ErrorCode {
    // Collision between use of one resource GV and another.
    // All uses must be guaranteed to resolve to only one GV.
    // Additionally, when writing resource to alloca, all uses
    // of that alloca are considered resolving to a single GV.
    GVConflicts,

    // static global resources are disallowed for libraries at this time.
    // for non-library targets, they should have been eliminated already.
    StaticGVUsed,

    // user function calls with resource params or return type are
    // are currently disallowed for libraries.
    UserCallsWithResources,

    // When searching up from store pointer looking for alloca,
    // we encountered an unexpted value type
    UnexpectedValuesFromStorePointer,

    // When remapping values to be replaced, we add them to RemappedValues
    // so we don't use dead values stored in other sets/maps.  Circular
    // remaps that should not happen are aadded to RemappingCyclesDetected.
    RemappingCyclesDetected,

    // Without SUPPORT_SELECT_ON_ALLOCA, phi/select on alloca based
    // pointer is disallowed, since this scenario is still untested.
    // This error also covers any other unknown alloca pointer uses.
    // Supported:
    // alloca (-> gep)? -> load -> ...
    // alloca (-> gep)? -> store.
    // Unsupported without SUPPORT_SELECT_ON_ALLOCA:
    // alloca (-> gep)? -> phi/select -> ...
    AllocaUserDisallowed,
    MismatchHandleAnnotation,
    MixDynamicResourceWithBindingResource,
    MismatchIsSampler,
#ifdef SUPPORT_SELECT_ON_ALLOCA
    // Conflict in select/phi between GV pointer and alloca pointer.  This
    // algorithm can't handle this case.
    AllocaSelectConflict,
#endif

    ErrorCodeCount
  };

  const StringRef ErrorText[ErrorCodeCount] = {
    "local resource not guaranteed to map to unique global resource.",
    "static global resource use is disallowed for library functions.",
    "exported library functions cannot have resource parameters or return value.",
    "internal error: unexpected instruction type when looking for alloca from store.",
    "internal error: cycles detected in value remapping.",
    "phi/select disallowed on pointers to local resources.",
    "mismatch handle annotation",
    "possible mixing dynamic resource and binding resource",
    "merging sampler handle and resource handle",
#ifdef SUPPORT_SELECT_ON_ALLOCA
    ,"unable to resolve merge of global and local resource pointers."
#endif
  };

  ValueSetVector ErrorSets[ErrorCodeCount];

  // Ulitimately, the goal of ErrorUsers is to mark all create handles
  // so we don't try to report errors on them again later.
  std::unordered_set<Value*> ErrorUsers;  // users of error values
  bool AddErrorUsers(Value* V) {
    auto it = ErrorUsers.insert(V);
    if (!it.second)
      return false;   // already there
    if (isa<GEPOperator>(V) ||
        isa<LoadInst>(V) ||
        isa<PHINode>(V) ||
        isa<SelectInst>(V) ||
        isa<AllocaInst>(V)) {
      for (auto U : V->users()) {
        AddErrorUsers(U);
      }
    } else if(isa<StoreInst>(V)) {
      AddErrorUsers(cast<StoreInst>(V)->getPointerOperand());
    }
    // create handle will be marked, but users not followed
    return true;
  }
  void ReportError(ErrorCode ec, Value* V) {
    DXASSERT_NOMSG(ec < ErrorCodeCount);
    if (!ErrorSets[ec].insert(V))
      return;   // Error already reported
    AddErrorUsers(V);
    m_bErrorsReported = true;
    if (Instruction *I = dyn_cast<Instruction>(V)) {
      dxilutil::EmitErrorOnInstruction(I, ErrorText[ec]);
    } else {
      StringRef Name = V->getName();
      std::string escName;
      if (isa<Function>(V)) {
        llvm::raw_string_ostream os(escName);
        dxilutil::PrintEscapedString(Name, os);
        os.flush();
        Name = escName;
      }
      V->getContext().emitError(Twine(ErrorText[ec]) + " Value: " + Name);
    }
  }

  bool ErrorsReported() {
    return m_bErrorsReported;
  }
};

unsigned CountArrayDimensions(Type* Ty,
    // Optionally collect dimensions
    SmallVector<unsigned, 4> *dims = nullptr) {
  if (Ty->isPointerTy())
    Ty = Ty->getPointerElementType();
  unsigned dim = 0;
  if (dims)
    dims->clear();
  while (Ty->isArrayTy()) {
    if (dims)
      dims->push_back(Ty->getArrayNumElements());
    dim++;
    Ty = Ty->getArrayElementType();
  }
  return dim;
}

// Delete unused CleanupInsts, restarting when changed
// Return true if something was deleted
bool CleanupUnusedValues(std::unordered_set<Instruction *> &CleanupInsts) {
  //  - delete unused CleanupInsts, restarting when changed
  bool bAnyChanges = false;
  bool bChanged = false;
  do {
    bChanged = false;
    for (auto it = CleanupInsts.begin(); it != CleanupInsts.end();) {
      Instruction *I = *(it++);
      if (I->user_empty()) {
        // Add instructions operands CleanupInsts
        for (unsigned iOp = 0; iOp < I->getNumOperands(); iOp++) {
          if (Instruction *opI = dyn_cast<Instruction>(I->getOperand(iOp)))
            CleanupInsts.insert(opI);
        }
        I->eraseFromParent();
        CleanupInsts.erase(I);
        bChanged = true;
      }
    }
    if (bChanged)
      bAnyChanges = true;
  } while (bChanged);
  return bAnyChanges;
}

// Helper class for legalizing resource use
// Convert select/phi on resources to select/phi on index to GEP on GV.
// Convert resource alloca to index alloca.
// Assumes createHandleForLib has no select/phi
class LegalizeResourceUseHelper {
  // Change:
  //  gep1 = GEP gRes, i1
  //  res1 = load gep1
  //  gep2 = GEP gRes, i2
  //  gep3 = GEP gRes, i3
  //  gep4 = phi gep2, gep3           <-- handle select/phi on GEP
  //  res4 = load gep4
  //  res5 = phi res1, res4
  //  res6 = load GEP gRes, 23        <-- handle constant GepExpression
  //  res = select cnd2, res5, res6
  //  handle = createHandleForLib(res)
  // To:
  //  i4 = phi i2, i3
  //  i5 = phi i1, i4
  //  i6 = select cnd, i5, 23
  //  gep = GEP gRes, i6
  //  res = load gep
  //  handle = createHandleForLib(res)

  // Also handles alloca
  //  resArray = alloca [2 x Resource]
  //  gep1 = GEP gRes, i1
  //  res1 = load gep1
  //  gep2 = GEP gRes, i2
  //  gep3 = GEP gRes, i3
  //  phi4 = phi gep2, gep3
  //  res4 = load phi4
  //  gep5 = GEP resArray, 0
  //  gep6 = GEP resArray, 1
  //  store gep5, res1
  //  store gep6, res4
  //  gep7 = GEP resArray, i7   <-- dynamically index array
  //  res = load gep7
  //  handle = createHandleForLib(res)
  // Desired result:
  //  idxArray = alloca [2 x i32]
  //  phi4 = phi i2, i3
  //  gep5 = GEP idxArray, 0
  //  gep6 = GEP idxArray, 1
  //  store gep5, i1
  //  store gep6, phi4
  //  gep7 = GEP idxArray, i7
  //  gep8 = GEP gRes, gep7
  //  res = load gep8
  //  handle = createHandleForLib(res)

  // Also handles multi-dim resource index and multi-dim resource array allocas

  // Basic algorithm:
  // - recursively mark each GV user with GV (ValueToResourceGV)
  //  - verify only one GV used for any given value
  // - handle allocas by searching up from store for alloca
  //  - then recursively mark alloca users
  // - ResToIdxReplacement keeps track of vector of indices that
  //   will be used to replace a given resource value or pointer
  // - Next, create selects/phis for indices corresponding to
  //   selects/phis on resource pointers or values.
  //  - leave incoming index values undef for now
  // - Create index allocas to replace resource allocas
  // - Create GEPs on index allocas to replace GEPs on resource allocas
  // - Create index loads on index allocas to replace loads on resource alloca GEP
  // - Fill in replacements for GEPs on resource GVs
  //  - copy replacement index vectors to corresponding loads
  // - Create index stores to replace resource stores to alloca/GEPs
  // - Update selects/phis incoming index values
  // - SimplifyMerges: replace index phis/selects on same value with that value
  //  - RemappedValues[phi/select] set to replacement value
  //  - use LookupValue from now on when reading from ResToIdxReplacement
  // - Update handles by replacing load/GEP chains that go through select/phi
  //   with direct GV GEP + load, with select/phi on GEP indices instead.

public:
  ResourceUseErrors m_Errors;

  ValueToValueMap ValueToResourceGV;
  ValueToIdxMap ResToIdxReplacement;
  // Value sets we can use to iterate
  ValueSetVector Selects, GEPs, Stores, Handles;
  ValueSetVector Allocas, AllocaGEPs, AllocaLoads;
#ifdef SUPPORT_SELECT_ON_ALLOCA
  ValueSetVector AllocaSelects;
#endif

  std::unordered_set<Value *> NonUniformSet;

  // New index selects created by pass, so we can try simplifying later
  ValueSetVector NewSelects;

  // Values that have been replaced with other values need remapping
  ValueToValueMap RemappedValues;

  // Things to clean up if no users:
  std::unordered_set<Instruction*> CleanupInsts;

  GlobalVariable *LookupResourceGV(Value *V) {
    auto itGV = ValueToResourceGV.find(V);
    if (itGV == ValueToResourceGV.end())
      return nullptr;
    return cast<GlobalVariable>(itGV->second);
  }

  // Follow RemappedValues, return input if not remapped
  Value *LookupValue(Value *V) {
    auto it = RemappedValues.find(V);
    SmallPtrSet<Value*, 4> visited;
    while (it != RemappedValues.end()) {
      // Cycles should not happen, but are bad if they do.
      if (visited.count(it->second)) {
        DXASSERT(false, "otherwise, circular remapping");
        m_Errors.ReportError(ResourceUseErrors::RemappingCyclesDetected, V);
        break;
      }
      V = it->second;
      it = RemappedValues.find(V);
      if (it != RemappedValues.end())
        visited.insert(V);
    }
    return V;
  }

  bool AreLoadUsersTrivial(LoadInst *LI) {
    for (auto U : LI->users()) {
      if (CallInst *CI = dyn_cast<CallInst>(U)) {
        Function *F = CI->getCalledFunction();
        DxilModule &DM = F->getParent()->GetDxilModule();
        hlsl::OP *hlslOP = DM.GetOP();
        if (hlslOP->IsDxilOpFunc(F)) {
          hlsl::OP::OpCodeClass opClass;
          if (hlslOP->GetOpCodeClass(F, opClass) &&
            opClass == DXIL::OpCodeClass::CreateHandleForLib) {
            continue;
          }
        }
      }
      return false;
    }
    return true;
  }

  // This is used to quickly skip the common case where no work is needed
  bool AreGEPUsersTrivial(GEPOperator *GEP) {
    if (GlobalVariable *GV = LookupResourceGV(GEP)) {
      if (GEP->getPointerOperand() != LookupResourceGV(GEP))
        return false;
    }
    for (auto U : GEP->users()) {
      if (LoadInst *LI = dyn_cast<LoadInst>(U)) {
        if (AreLoadUsersTrivial(LI))
          continue;
      }
      return false;
    }
    return true;
  }

  // AssignResourceGVFromStore is used on pointer being stored to.
  // Follow GEP/Phi/Select up to Alloca, then CollectResourceGVUsers on Alloca
  void AssignResourceGVFromStore(GlobalVariable *GV, Value *V,
                                 SmallPtrSet<Value*, 4> &visited,
                                 bool bNonUniform) {
    // Prevent cycles as we search up
    if (visited.count(V) != 0)
      return;
    // Verify and skip if already processed
    auto it = ValueToResourceGV.find(V);
    if (it != ValueToResourceGV.end()) {
      if (it->second != GV) {
        m_Errors.ReportError(ResourceUseErrors::GVConflicts, V);
      }
      return;
    }
    if (AllocaInst *AI = dyn_cast<AllocaInst>(V)) {
      CollectResourceGVUsers(GV, AI, /*bAlloca*/true, bNonUniform);
      return;
    } else if (GEPOperator *GEP = dyn_cast<GEPOperator>(V)) {
      // follow the pointer up
      AssignResourceGVFromStore(GV, GEP->getPointerOperand(), visited, bNonUniform);
      return;
    } else if (PHINode *Phi = dyn_cast<PHINode>(V)) {
#ifdef SUPPORT_SELECT_ON_ALLOCA
      // follow all incoming values
      for (auto it : Phi->operand_values())
        AssignResourceGVFromStore(GV, it, visited, bNonUniform);
#else
      m_Errors.ReportError(ResourceUseErrors::AllocaUserDisallowed, V);
#endif
      return;
    } else if (SelectInst *Sel = dyn_cast<SelectInst>(V)) {
#ifdef SUPPORT_SELECT_ON_ALLOCA
      // follow all incoming values
      AssignResourceGVFromStore(GV, Sel->getTrueValue(), visited, bNonUniform);
      AssignResourceGVFromStore(GV, Sel->getFalseValue(), visited, bNonUniform);
#else
      m_Errors.ReportError(ResourceUseErrors::AllocaUserDisallowed, V);
#endif
      return;
    } else if (isa<GlobalVariable>(V) &&
               cast<GlobalVariable>(V)->getLinkage() ==
                    GlobalVariable::LinkageTypes::InternalLinkage) {
      // this is writing to global static, which is disallowed at this point.
      m_Errors.ReportError(ResourceUseErrors::StaticGVUsed, V);
      return;
    } else {
      // Most likely storing to output parameter
      m_Errors.ReportError(ResourceUseErrors::UserCallsWithResources, V);
      return;
    }
    return;
  }

  // Recursively mark values with GV, following users.
  // Starting value V should be GV itself.
  // Returns true if value/uses reference no other GV in map.
  void CollectResourceGVUsers(GlobalVariable *GV, Value *V, bool bAlloca = false, bool bNonUniform = false) {
    // Recursively tag value V and its users as using GV.
    auto it = ValueToResourceGV.find(V);
    if (it != ValueToResourceGV.end()) {
      if (it->second != GV) {
        m_Errors.ReportError(ResourceUseErrors::GVConflicts, V);
#ifdef SUPPORT_SELECT_ON_ALLOCA
      } else {
        // if select/phi, make sure bAlloca is consistent
        if (isa<PHINode>(V) || isa<SelectInst>(V))
          if ((bAlloca && AllocaSelects.count(V) == 0) ||
              (!bAlloca && Selects.count(V) == 0))
            m_Errors.ReportError(ResourceUseErrors::AllocaSelectConflict, V);
#endif
      }
      return;
    }
    ValueToResourceGV[V] = GV;
    if (GV == V) {
      // Just add and recurse users
      // make sure bAlloca is clear for users
      bAlloca = false;
    } else if (GEPOperator *GEP = dyn_cast<GEPOperator>(V)) {
      if (bAlloca)
        AllocaGEPs.insert(GEP);
      else if (!AreGEPUsersTrivial(GEP))
        GEPs.insert(GEP);
      else
        return; // Optimization: skip trivial GV->GEP->load->createHandle
      if (GetElementPtrInst *GEPInst = dyn_cast<GetElementPtrInst>(GEP)) {
        if (DxilMDHelper::IsMarkedNonUniform(GEPInst))
          bNonUniform = true;
      }
    } else if (LoadInst *LI = dyn_cast<LoadInst>(V)) {
      if (bAlloca)
        AllocaLoads.insert(LI);
      // clear bAlloca for users
      bAlloca = false;
      if (bNonUniform)
        NonUniformSet.insert(LI);
    } else if (StoreInst *SI = dyn_cast<StoreInst>(V)) {
      Stores.insert(SI);
      if (!bAlloca) {
        // Find and mark allocas this store could be storing to
        SmallPtrSet<Value*, 4> visited;
        AssignResourceGVFromStore(GV, SI->getPointerOperand(), visited, bNonUniform);
      }
      return;
    } else if (PHINode *Phi = dyn_cast<PHINode>(V)) {
      if (bAlloca) {
#ifdef SUPPORT_SELECT_ON_ALLOCA
        AllocaSelects.insert(Phi);
#else
        m_Errors.ReportError(ResourceUseErrors::AllocaUserDisallowed, V);
#endif
      } else {
        Selects.insert(Phi);
      }
    } else if (SelectInst *Sel = dyn_cast<SelectInst>(V)) {
      if (bAlloca) {
#ifdef SUPPORT_SELECT_ON_ALLOCA
        AllocaSelects.insert(Sel);
#else
        m_Errors.ReportError(ResourceUseErrors::AllocaUserDisallowed, V);
#endif
      } else {
        Selects.insert(Sel);
      }
    } else if (AllocaInst *AI = dyn_cast<AllocaInst>(V)) {
      Allocas.insert(AI);
      // set bAlloca for users
      bAlloca = true;
    } else if (Constant *C = dyn_cast<Constant>(V)) {
      // skip @llvm.used entry
      return;
    } else if (BitCastInst *BCI = dyn_cast<BitCastInst>(V)) {
      DXASSERT(onlyUsedByLifetimeMarkers(BCI),
               "expected bitcast to only be used by lifetime intrinsics");
      return;
    } else if (bAlloca) {
      m_Errors.ReportError(ResourceUseErrors::AllocaUserDisallowed, V);
    } else {
      // Must be createHandleForLib or user function call.
      CallInst *CI = cast<CallInst>(V);
      Function *F = CI->getCalledFunction();
      DxilModule &DM = GV->getParent()->GetDxilModule();
      hlsl::OP *hlslOP = DM.GetOP();
      if (hlslOP->IsDxilOpFunc(F)) {
        hlsl::OP::OpCodeClass opClass;
        if (hlslOP->GetOpCodeClass(F, opClass) &&
            (opClass == DXIL::OpCodeClass::CreateHandleForLib)) {
          Handles.insert(CI);
          if (bNonUniform)
            NonUniformSet.insert(CI);
          return;
        }
      }
      // This could be user call with resource param, which is disallowed for lib_6_3
      m_Errors.ReportError(ResourceUseErrors::UserCallsWithResources, V);
      return;
    }

    // Recurse users
    for (auto U : V->users())
      CollectResourceGVUsers(GV, U, bAlloca, bNonUniform);
    return;
  }

  // Remove conflicting values from sets before
  // transforming the remainder.
  void RemoveConflictingValue(Value* V) {
    bool bRemoved = false;
    if (isa<GEPOperator>(V)) {
      bRemoved = GEPs.remove(V) || AllocaGEPs.remove(V);
    } else if (isa<LoadInst>(V)) {
      bRemoved = AllocaLoads.remove(V);
    } else if (isa<StoreInst>(V)) {
      bRemoved = Stores.remove(V);
    } else if (isa<PHINode>(V) || isa<SelectInst>(V)) {
      bRemoved = Selects.remove(V);
#ifdef SUPPORT_SELECT_ON_ALLOCA
      bRemoved |= AllocaSelects.remove(V);
#endif
    } else if (isa<AllocaInst>(V)) {
      bRemoved = Allocas.remove(V);
    } else if (isa<CallInst>(V)) {
      bRemoved = Handles.remove(V);
      return; // don't recurse
    }
    if (bRemoved) {
      // Recurse users
      for (auto U : V->users())
        RemoveConflictingValue(U);
    }
  }
  void RemoveConflicts() {
    for (auto V : m_Errors.ErrorSets[ResourceUseErrors::GVConflicts]) {
      RemoveConflictingValue(V);
      ValueToResourceGV.erase(V);
    }
  }

  void CreateSelects() {
    if (Selects.empty()
#ifdef SUPPORT_SELECT_ON_ALLOCA
        && AllocaSelects.empty()
#endif
        )
      return;
    LLVMContext &Ctx =
#ifdef SUPPORT_SELECT_ON_ALLOCA
      Selects.empty() ? AllocaSelects[0]->getContext() :
#endif
      Selects[0]->getContext();
    Type *i32Ty = IntegerType::getInt32Ty(Ctx);
#ifdef SUPPORT_SELECT_ON_ALLOCA
    for (auto &SelectSet : {Selects, AllocaSelects}) {
      bool bAlloca = !(&SelectSet == &Selects);
#else
    for (auto &SelectSet : { Selects }) {
#endif
      for (auto pValue : SelectSet) {
        Type *SelectTy = i32Ty;
#ifdef SUPPORT_SELECT_ON_ALLOCA
        // For alloca case, type needs to match dimensionality of incoming value
        if (bAlloca) {
          // TODO: Not sure if this case will actually work
          //      (or whether it can even be generated from HLSL)
          Type *Ty = pValue->getType();
          SmallVector<unsigned, 4> dims;
          unsigned dim = CountArrayDimensions(Ty, &dims);
          for (unsigned i = 0; i < dim; i++)
            SelectTy = ArrayType::get(SelectTy, (uint64_t)dims[dim - i - 1]);
          if (Ty->isPointerTy())
            SelectTy = PointerType::get(SelectTy, 0);
        }
#endif
        Value *UndefValue = UndefValue::get(SelectTy);
        if (PHINode *Phi = dyn_cast<PHINode>(pValue)) {
          GlobalVariable *GV = LookupResourceGV(Phi);
          if (!GV)
            continue; // skip value removed due to conflict
          IRBuilder<> PhiBuilder(Phi);
          unsigned gvDim = CountArrayDimensions(GV->getType());
          IndexVector &idxVector = ResToIdxReplacement[Phi];
          idxVector.resize(gvDim, nullptr);
          unsigned numIncoming = Phi->getNumIncomingValues();
          for (unsigned i = 0; i < gvDim; i++) {
            PHINode *newPhi = PhiBuilder.CreatePHI(SelectTy, numIncoming);
            NewSelects.insert(newPhi);
            idxVector[i] = newPhi;
            for (unsigned j = 0; j < numIncoming; j++) {
              // Set incoming values to undef until next pass
              newPhi->addIncoming(UndefValue, Phi->getIncomingBlock(j));
            }
          }
        } else if (SelectInst *Sel = dyn_cast<SelectInst>(pValue)) {
          GlobalVariable *GV = LookupResourceGV(Sel);
          if (!GV)
            continue; // skip value removed due to conflict
          IRBuilder<> Builder(Sel);
          unsigned gvDim = CountArrayDimensions(GV->getType());
          IndexVector &idxVector = ResToIdxReplacement[Sel];
          idxVector.resize(gvDim, nullptr);
          for (unsigned i = 0; i < gvDim; i++) {
            Value *newSel = Builder.CreateSelect(Sel->getCondition(), UndefValue, UndefValue);
            NewSelects.insert(newSel);
            idxVector[i] = newSel;
          }
        } else {
          DXASSERT(false, "otherwise, non-select/phi in Selects set");
        }
      }
    }
  }

  // Create index allocas to replace resource allocas
  void CreateIndexAllocas() {
    if (Allocas.empty())
      return;
    Type *i32Ty = IntegerType::getInt32Ty(Allocas[0]->getContext());
    for (auto pValue : Allocas) {
      AllocaInst *pAlloca = cast<AllocaInst>(pValue);
      GlobalVariable *GV = LookupResourceGV(pAlloca);
      if (!GV)
        continue; // skip value removed due to conflict
      IRBuilder<> AllocaBuilder(pAlloca);
      unsigned gvDim = CountArrayDimensions(GV->getType());
      SmallVector<unsigned, 4> dimVector;
      unsigned allocaTyDim = CountArrayDimensions(pAlloca->getType(), &dimVector);
      Type *pIndexType = i32Ty;
      for (unsigned i = 0; i < allocaTyDim; i++) {
        pIndexType = ArrayType::get(pIndexType, dimVector[allocaTyDim - i - 1]);
      }
      Value *arraySize = pAlloca->getArraySize();
      IndexVector &idxVector = ResToIdxReplacement[pAlloca];
      idxVector.resize(gvDim, nullptr);
      for (unsigned i = 0; i < gvDim; i++) {
        AllocaInst *pAlloca = AllocaBuilder.CreateAlloca(pIndexType, arraySize);
        pAlloca->setAlignment(4);
        idxVector[i] = pAlloca;
      }
    }
  }

  // Add corresponding GEPs for index allocas
  IndexVector &ReplaceAllocaGEP(GetElementPtrInst *GEP) {
    IndexVector &idxVector = ResToIdxReplacement[GEP];
    if (!idxVector.empty())
      return idxVector;

    Value *Ptr = GEP->getPointerOperand();

    // Recurse for partial GEPs
    IndexVector &ptrIndices = isa<GetElementPtrInst>(Ptr) ?
      ReplaceAllocaGEP(cast<GetElementPtrInst>(Ptr)) : ResToIdxReplacement[Ptr];

    IRBuilder<> Builder(GEP);
    SmallVector<Value*, 4> gepIndices;
    for (auto it = GEP->idx_begin(), idxEnd = GEP->idx_end(); it != idxEnd; it++)
      gepIndices.push_back(*it);
    idxVector.resize(ptrIndices.size(), nullptr);
    for (unsigned i = 0; i < ptrIndices.size(); i++) {
      idxVector[i] = Builder.CreateInBoundsGEP(ptrIndices[i], gepIndices);
    }
    return idxVector;
  }

  void ReplaceAllocaGEPs() {
    for (auto V : AllocaGEPs) {
      ReplaceAllocaGEP(cast<GetElementPtrInst>(V));
    }
  }

  void ReplaceAllocaLoads() {
    for (auto V : AllocaLoads) {
      LoadInst *LI = cast<LoadInst>(V);
      Value *Ptr = LI->getPointerOperand();
      IRBuilder<> Builder(LI);
      IndexVector &idxVector = ResToIdxReplacement[V];
      IndexVector &ptrIndices = ResToIdxReplacement[Ptr];
      idxVector.resize(ptrIndices.size(), nullptr);
      for (unsigned i = 0; i < ptrIndices.size(); i++) {
        idxVector[i] = Builder.CreateLoad(ptrIndices[i]);
      }
    }
  }

  // Add GEP to ResToIdxReplacement with indices from incoming + GEP
  IndexVector &ReplaceGVGEPs(GEPOperator *GEP) {
    IndexVector &idxVector = ResToIdxReplacement[GEP];
    // Skip if already done
    // (we recurse into partial GEP and iterate all GEPs)
    if (!idxVector.empty())
      return idxVector;

    Type *i32Ty = IntegerType::getInt32Ty(GEP->getContext());
    Constant *Zero = Constant::getIntegerValue(i32Ty, APInt(32, 0));

    Value *Ptr = GEP->getPointerOperand();

    unsigned idx = 0;
    if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Ptr)) {
      unsigned gvDim = CountArrayDimensions(GV->getType());
      idxVector.resize(gvDim, Zero);
    } else if (isa<GEPOperator>(Ptr) || isa<PHINode>(Ptr) || isa<SelectInst>(Ptr)) {
      // Recurse for partial GEPs
      IndexVector &ptrIndices = isa<GEPOperator>(Ptr) ?
        ReplaceGVGEPs(cast<GEPOperator>(Ptr)) : ResToIdxReplacement[Ptr];
      unsigned ptrDim = CountArrayDimensions(Ptr->getType());
      unsigned gvDim = ptrIndices.size();
      DXASSERT(ptrDim <= gvDim, "otherwise incoming pointer has more dimensions than associated GV");
      unsigned gepStart = gvDim - ptrDim;
      // Copy indices and add ours
      idxVector.resize(ptrIndices.size(), Zero);
      for (; idx < gepStart; idx++)
        idxVector[idx] = ptrIndices[idx];
    }
    if (GEP->hasIndices()) {
      auto itIdx = GEP->idx_begin();
      ++itIdx;  // Always skip leading zero (we don't support GV+n pointer arith)
      while (itIdx != GEP->idx_end())
        idxVector[idx++] = *itIdx++;
    }
    return idxVector;
  }

  // Add GEPs to ResToIdxReplacement and update loads
  void ReplaceGVGEPs() {
    if (GEPs.empty())
      return;
    for (auto V : GEPs) {
      GEPOperator *GEP = cast<GEPOperator>(V);
      IndexVector &gepVector = ReplaceGVGEPs(GEP);
      for (auto U : GEP->users()) {
        if (LoadInst *LI = dyn_cast<LoadInst>(U)) {
          // Just copy incoming indices
          ResToIdxReplacement[LI] = gepVector;
        }
      }
    }
  }

  // Create new index stores for incoming indices
  void ReplaceStores() {
    // generate stores of incoming indices to corresponding index pointers
    if (Stores.empty())
      return;
    Type *i32Ty = IntegerType::getInt32Ty(Stores[0]->getContext());
    for (auto V : Stores) {
      StoreInst *SI = cast<StoreInst>(V);
      IRBuilder<> Builder(SI);
      IndexVector &idxVector = ResToIdxReplacement[SI];
      Value *Ptr = SI->getPointerOperand();
      Value *Val = SI->getValueOperand();
      IndexVector &ptrIndices = ResToIdxReplacement[Ptr];
      IndexVector &valIndices = ResToIdxReplacement[Val];
      // If Val is not found, it is treated as an undef value that will translate
      // to an undef index, which may still be valid if it's never used.
      Value *UndefIndex = valIndices.size() > 0 ? nullptr : UndefValue::get(i32Ty);
      DXASSERT_NOMSG(valIndices.size() == 0 || ptrIndices.size() == valIndices.size());
      idxVector.resize(ptrIndices.size(), nullptr);
      for (unsigned i = 0; i < idxVector.size(); i++) {
        idxVector[i] = Builder.CreateStore(
          UndefIndex ? UndefIndex : valIndices[i],
          ptrIndices[i]);
      }
    }
  }

  // For each Phi/Select: update matching incoming values for new phis
  void UpdateSelects() {
    if (Selects.empty())
      return;
    Type *i32Ty = IntegerType::getInt32Ty(Selects[0]->getContext());
    for (auto V : Selects) {
      // update incoming index values corresponding to incoming resource values
      IndexVector &idxVector = ResToIdxReplacement[V];
      Instruction *I = cast<Instruction>(V);
      unsigned numOperands = I->getNumOperands();
      unsigned startOp = isa<PHINode>(V) ? 0 : 1;
      for (unsigned iOp = startOp; iOp < numOperands; iOp++) {
        Value *Val = I->getOperand(iOp);
        IndexVector &incomingIndices = ResToIdxReplacement[Val];
        // If Val is not found, it is treated as an undef value that will translate
        // to an undef index, which may still be valid if it's never used.
        Value *UndefIndex = incomingIndices.size() > 0 ? nullptr : UndefValue::get(i32Ty);
        DXASSERT_NOMSG(incomingIndices.size() == 0 || idxVector.size() == incomingIndices.size());
        for (unsigned i = 0; i < idxVector.size(); i++) {
          // must be instruction (phi/select)
          Instruction *indexI = cast<Instruction>(idxVector[i]);
          indexI->setOperand(iOp,
            UndefIndex ? UndefIndex : incomingIndices[i]);
        }

        // Now clear incoming operand (adding to cleanup) to break cycles
        if (Instruction *OpI = dyn_cast<Instruction>(I->getOperand(iOp)))
          CleanupInsts.insert(OpI);
        I->setOperand(iOp, UndefValue::get(I->getType()));
      }
    }
  }

  // ReplaceHandles
  //  - iterate handles
  //    - insert GEP using new indices associated with resource value
  //    - load resource from new GEP
  //    - replace resource use in createHandleForLib with new load
  // Assumes: no users of handle are phi/select or store
  void ReplaceHandles() {
    if (Handles.empty())
      return;
    Type *i32Ty = IntegerType::getInt32Ty(Handles[0]->getContext());
    Constant *Zero = Constant::getIntegerValue(i32Ty, APInt(32, 0));
    for (auto V : Handles) {
      CallInst *CI = cast<CallInst>(V);
      DxilInst_CreateHandleForLib createHandle(CI);
      Value *res = createHandle.get_Resource();
      // Skip extra work if nothing between load and create handle
      if (LoadInst *LI = dyn_cast<LoadInst>(res)) {
        Value *Ptr = LI->getPointerOperand();
        if (GEPOperator *GEP = dyn_cast<GEPOperator>(Ptr))
          Ptr = GEP->getPointerOperand();
        if (isa<GlobalVariable>(Ptr))
          continue;
      }
      GlobalVariable *GV = LookupResourceGV(res);
      if (!GV)
        continue; // skip value removed due to conflict
      IRBuilder<> Builder(CI);
      IndexVector &idxVector = ResToIdxReplacement[res];
      DXASSERT(idxVector.size() == CountArrayDimensions(GV->getType()), "replacements empty or invalid");
      SmallVector<Value*, 4> gepIndices;
      gepIndices.push_back(Zero);
      for (auto idxVal : idxVector)
        gepIndices.push_back(LookupValue(idxVal));
      Value *GEP = Builder.CreateInBoundsGEP(GV, gepIndices);
      // Mark new GEP instruction non-uniform if necessary
      if (NonUniformSet.count(res) != 0 || NonUniformSet.count(CI) != 0)
        if (GetElementPtrInst *GEPInst = dyn_cast<GetElementPtrInst>(GEP))
          DxilMDHelper::MarkNonUniform(GEPInst);
      LoadInst *LI = Builder.CreateLoad(GEP);
      createHandle.set_Resource(LI);
      if (Instruction *resI = dyn_cast<Instruction>(res))
        CleanupInsts.insert(resI);
    }
  }

  void SimplifyMerges() {
    // Loop if changed
    bool bChanged = false;
    do {
      bChanged = false;
      for (auto V : NewSelects) {
        if (LookupValue(V) != V)
          continue;
        Instruction *I = cast<Instruction>(V);
        unsigned startOp = isa<PHINode>(I) ? 0 : 1;
        Value *newV = dxilutil::MergeSelectOnSameValue(
          cast<Instruction>(V), startOp, I->getNumOperands());
        if (newV) {
          RemappedValues[V] = newV;
          bChanged = true;
        }
      }
    } while (bChanged);
  }

  void CleanupDeadInsts() {
    // Assuming everything was successful:
    // delete stores to allocas to remove cycles
    for (auto V : Stores) {
      StoreInst *SI = cast<StoreInst>(V);
      if (Instruction *I = dyn_cast<Instruction>(SI->getValueOperand()))
        CleanupInsts.insert(I);
      if (Instruction *I = dyn_cast<Instruction>(SI->getPointerOperand()))
        CleanupInsts.insert(I);
      SI->eraseFromParent();
    }
    CleanupUnusedValues(CleanupInsts);
  }

  void VerifyComplete(DxilModule &DM) {
    // Check that all handles now resolve to a global variable, otherwise,
    // they are likely loading from resource function parameter, which
    // is disallowed.
    hlsl::OP *hlslOP = DM.GetOP();
    for (Function &F : DM.GetModule()->functions()) {
      if (hlslOP->IsDxilOpFunc(&F)) {
        hlsl::OP::OpCodeClass opClass;
        if (hlslOP->GetOpCodeClass(&F, opClass) &&
          opClass == DXIL::OpCodeClass::CreateHandleForLib) {
          for (auto U : F.users()) {
            CallInst *CI = cast<CallInst>(U);
            if (m_Errors.ErrorUsers.count(CI))
              continue;   // Error already reported
            DxilInst_CreateHandleForLib createHandle(CI);
            Value *res = createHandle.get_Resource();
            LoadInst *LI = dyn_cast<LoadInst>(res);
            if (LI) {
              Value *Ptr = LI->getPointerOperand();
              if (GEPOperator *GEP = dyn_cast<GEPOperator>(Ptr))
                Ptr = GEP->getPointerOperand();
              if (isa<GlobalVariable>(Ptr))
                continue;
            }
            // handle wasn't processed
            // Right now, the most likely cause is user call with resources, but
            // this should be updated if there are other reasons for this to happen.
            m_Errors.ReportError(ResourceUseErrors::UserCallsWithResources, U);
          }
        }
      }
    }
  }

  // Fix resource global variable properties to external constant
  bool SetExternalConstant(GlobalVariable *GV) {
    if (GV->hasInitializer() || !GV->isConstant() ||
        GV->getLinkage() != GlobalVariable::LinkageTypes::ExternalLinkage) {
      GV->setInitializer(nullptr);
      GV->setConstant(true);
      GV->setLinkage(GlobalVariable::LinkageTypes::ExternalLinkage);
      return true;
    }
    return false;
  }

  bool CollectResources(DxilModule &DM) {
    bool bChanged = false;
    for (const auto &res : DM.GetCBuffers()) {
      if (GlobalVariable *GV = dyn_cast<GlobalVariable>(res->GetGlobalSymbol())) {
        bChanged |= SetExternalConstant(GV);
        CollectResourceGVUsers(GV, GV);
      }
    }
    for (const auto &res : DM.GetSRVs()) {
      if (GlobalVariable *GV = dyn_cast<GlobalVariable>(res->GetGlobalSymbol())) {
        bChanged |= SetExternalConstant(GV);
        CollectResourceGVUsers(GV, GV);
      }
    }
    for (const auto &res : DM.GetUAVs()) {
      if (GlobalVariable *GV = dyn_cast<GlobalVariable>(res->GetGlobalSymbol())) {
        bChanged |= SetExternalConstant(GV);
        CollectResourceGVUsers(GV, GV);
      }
    }
    for (const auto &res : DM.GetSamplers()) {
      if (GlobalVariable *GV = dyn_cast<GlobalVariable>(res->GetGlobalSymbol())) {
        bChanged |= SetExternalConstant(GV);
        CollectResourceGVUsers(GV, GV);
      }
    }
    return bChanged;
  }

  void DoTransform(hlsl::OP *hlslOP) {
    RemoveConflicts();
    CreateSelects();
    CreateIndexAllocas();
    ReplaceAllocaGEPs();
    ReplaceAllocaLoads();
    ReplaceGVGEPs();
    ReplaceStores();
    UpdateSelects();
    SimplifyMerges();
    ReplaceHandles();
    if (!m_Errors.ErrorsReported())
      CleanupDeadInsts();
  }

  bool ErrorsReported() {
    return m_Errors.ErrorsReported();
  }

  bool runOnModule(llvm::Module &M) {
    DxilModule &DM = M.GetOrCreateDxilModule();
    hlsl::OP *hlslOP = DM.GetOP();

    bool bChanged = CollectResources(DM);

    // If no selects or allocas are involved, there isn't anything to do
    if (Selects.empty() && Allocas.empty())
      return bChanged;

    DoTransform(hlslOP);
    VerifyComplete(DM);

    return true;
  }
};

class DxilLegalizeResources : public ModulePass {
public:
  static char ID; // Pass identification, replacement for typeid
  explicit DxilLegalizeResources()
    : ModulePass(ID) {}

  StringRef getPassName() const override {
    return "DXIL Legalize Resource Use";
  }

  bool runOnModule(Module &M) override {
    LegalizeResourceUseHelper helper;
    return helper.runOnModule(M);
  }

private:
};

} // namespace

char DxilLegalizeResources::ID = 0;

ModulePass *llvm::createDxilLegalizeResources() {
  return new DxilLegalizeResources();
}

INITIALIZE_PASS(DxilLegalizeResources,
  "hlsl-dxil-legalize-resources",
  "DXIL legalize resource use", false, false)


bool DxilLowerCreateHandleForLib::RemovePhiOnResource() {
  LegalizeResourceUseHelper helper;
  bool bChanged = helper.runOnModule(*m_DM->GetModule());
  if (helper.ErrorsReported())
    m_bLegalizationFailed = true;
  return bChanged;
}


// LegacyLayout.
namespace {

StructType *UpdateStructTypeForLegacyLayout(StructType *ST,
                                            DxilTypeSystem &TypeSys, Module &M,
                                            bool includeTopLevelResource=false);

Type *UpdateFieldTypeForLegacyLayout(Type *Ty,
                                     DxilFieldAnnotation &annotation,
                                     DxilTypeSystem &TypeSys, Module &M) {
  DXASSERT(!Ty->isPointerTy(), "struct field should not be a pointer");

  if (Ty->isArrayTy()) {
    Type *EltTy = Ty->getArrayElementType();
    Type *UpdatedTy =
        UpdateFieldTypeForLegacyLayout(EltTy, annotation, TypeSys, M);
    if (EltTy == UpdatedTy)
      return Ty;
    else if (UpdatedTy)
      return ArrayType::get(UpdatedTy, Ty->getArrayNumElements());
    else
      return nullptr;
  } else if (hlsl::HLMatrixType::isa(Ty)) {
    DXASSERT(annotation.HasMatrixAnnotation(), "must a matrix");
    HLMatrixType MatTy = HLMatrixType::cast(Ty);
    unsigned rows = MatTy.getNumRows();
    unsigned cols = MatTy.getNumColumns();
    Type *EltTy = MatTy.getElementTypeForReg();

    // Get cols and rows from annotation.
    const DxilMatrixAnnotation &matrix = annotation.GetMatrixAnnotation();
    if (matrix.Orientation == MatrixOrientation::RowMajor) {
      rows = matrix.Rows;
      cols = matrix.Cols;
    } else {
      DXASSERT_NOMSG(matrix.Orientation == MatrixOrientation::ColumnMajor);
      cols = matrix.Rows;
      rows = matrix.Cols;
    }

    EltTy =
        UpdateFieldTypeForLegacyLayout(EltTy, annotation, TypeSys, M);
    Type *rowTy = VectorType::get(EltTy, cols);

    // Matrix should be aligned like array if rows > 1,
    // otherwise, it's just like a vector.
    if (rows > 1)
      return ArrayType::get(rowTy, rows);
    else
      return rowTy;
  } else if (StructType *ST = dyn_cast<StructType>(Ty)) {
    return UpdateStructTypeForLegacyLayout(ST, TypeSys, M);
  } else if (FixedVectorType *VT = dyn_cast<FixedVectorType>(Ty)) {
    Type *EltTy = VT->getElementType();
    Type *UpdatedTy =
        UpdateFieldTypeForLegacyLayout(EltTy, annotation, TypeSys, M);
    if (EltTy == UpdatedTy)
      return Ty;
    else
      return VectorType::get(UpdatedTy, VT->getNumElements());
  } else {
    Type *i32Ty = Type::getInt32Ty(Ty->getContext());
    // Basic types.
    if (Ty->isHalfTy()) {
      return Type::getFloatTy(Ty->getContext());
    } else if (IntegerType *ITy = dyn_cast<IntegerType>(Ty)) {
      if (ITy->getBitWidth() < 32)
        return i32Ty;
      else
        return Ty;
    } else
      return Ty;
  }
}

StructType *UpdateStructTypeForLegacyLayout(StructType *ST,
                                            DxilTypeSystem &TypeSys,
                                            Module &M,
                                            bool includeTopLevelResource) {
  bool bUpdated = false;
  unsigned fieldsCount = ST->getNumElements();
  std::vector<Type *> fieldTypes;
  fieldTypes.reserve(fieldsCount);
  DxilStructAnnotation *SA = TypeSys.GetStructAnnotation(ST);

  if (!includeTopLevelResource && dxilutil::IsHLSLResourceType(ST))
    return nullptr;

  // After reflection is stripped from library, this will be null if no update is required.
  if (!SA) {
    return ST;
  }

  if (SA->IsEmptyStruct()) {
    return ST;
  }

  // Resource fields must be deleted, since they don't actually
  // show up in the structure layout.
  // fieldMap maps from new field index to old field index for porting annotations
  std::vector<unsigned> fieldMap;
  fieldMap.reserve(fieldsCount);

  for (unsigned i = 0; i < fieldsCount; i++) {
    Type *EltTy = ST->getElementType(i);
    Type *UpdatedTy = UpdateFieldTypeForLegacyLayout(
        EltTy, SA->GetFieldAnnotation(i), TypeSys, M);
    if (UpdatedTy != nullptr) {
      fieldMap.push_back(i);
      fieldTypes.push_back(UpdatedTy);
    }
    if (EltTy != UpdatedTy)
      bUpdated = true;
  }

  if (!bUpdated) {
    return ST;
  } else {
    std::string legacyName = std::string(DXIL::kHostLayoutTypePrefix) + ST->getName().str();
    if (StructType *legacyST = M.getTypeByName(legacyName))
      return legacyST;

    StructType *NewST =
        StructType::create(ST->getContext(), fieldTypes, legacyName);

    // Only add annotation if struct is not empty.
    if (NewST->getNumElements() > 0) {
      DxilStructAnnotation *NewSA = TypeSys.AddStructAnnotation(NewST);

      // Clone annotation.
      NewSA->SetCBufferSize(SA->GetCBufferSize());
      NewSA->SetNumTemplateArgs(SA->GetNumTemplateArgs());
      for (unsigned i = 0; i < SA->GetNumTemplateArgs(); i++) {
        NewSA->GetTemplateArgAnnotation(i) = SA->GetTemplateArgAnnotation(i);
      }
      // Remap with deleted resource fields
      for (unsigned i = 0; i < NewSA->GetNumFields(); i++) {
        NewSA->GetFieldAnnotation(i) = SA->GetFieldAnnotation(fieldMap[i]);
      }
      TypeSys.FinishStructAnnotation(*NewSA);
    }

    return NewST;
  }
}

bool UpdateStructTypeForLegacyLayout(DxilResourceBase &Res,
                                     DxilTypeSystem &TypeSys, DxilModule &DM) {
  Module &M = *DM.GetModule();
  Constant *Symbol = Res.GetGlobalSymbol();
  Type *ElemTy = Res.GetHLSLType()->getPointerElementType();
  // Support Array of ConstantBuffer/StructuredBuffer.
  llvm::SmallVector<unsigned, 4> arrayDims;
  ElemTy = dxilutil::StripArrayTypes(ElemTy, &arrayDims);
  StructType *ST = cast<StructType>(ElemTy);
  if (ST->isOpaque()) {
    DXASSERT(Res.GetClass() == DxilResourceBase::Class::CBuffer,
             "Only cbuffer can have opaque struct.");
    return false;
  }

  Type *UpdatedST =
      UpdateStructTypeForLegacyLayout(ST, TypeSys, M,
        Res.GetKind() == DXIL::ResourceKind::StructuredBuffer);
  if (ST != UpdatedST) {
    // Support Array of ConstantBuffer/StructuredBuffer.
    Type *UpdatedTy = dxilutil::WrapInArrayTypes(UpdatedST, arrayDims);
    GlobalVariable *NewGV = cast<GlobalVariable>(
        M.getOrInsertGlobal(Symbol->getName().str() + "_legacy", UpdatedTy));
    Res.SetGlobalSymbol(NewGV);
    Res.SetHLSLType(NewGV->getType());
    OP *hlslOP = DM.GetOP();

    if (DM.GetShaderModel()->IsLib()) {
      TypeSys.EraseStructAnnotation(ST);
      // If it's a library, we need to replace the GV which involves a few replacements
      Function *NF = hlslOP->GetOpFunc(hlsl::OP::OpCode::CreateHandleForLib, UpdatedST);
      Value *opArg =
          hlslOP->GetI32Const((unsigned)hlsl::OP::OpCode::CreateHandleForLib);
      auto replaceResLd = [&NF,&opArg](LoadInst *ldInst, Value *NewPtr) {
        if (!ldInst->user_empty()) {
          IRBuilder<> Builder = IRBuilder<>(ldInst);
          LoadInst *newLoad = Builder.CreateLoad(NewPtr);
          Value *args[] = {opArg, newLoad};

          for (auto user = ldInst->user_begin(), E = ldInst->user_end();
               user != E;) {
            CallInst *CI = cast<CallInst>(*(user++));
            CallInst *newCI = CallInst::Create(NF, args, "", CI);
            CI->replaceAllUsesWith(newCI);
            CI->eraseFromParent();
          }
        }
        ldInst->eraseFromParent();
      };
      // Merge GEP to simplify replace old GV.
      if (!arrayDims.empty())
        dxilutil::MergeGepUse(Symbol);
      // Replace old GV.
      for (auto UserIt = Symbol->user_begin(), userEnd = Symbol->user_end(); UserIt != userEnd;) {
        Value *User = *(UserIt++);

        if (LoadInst *ldInst = dyn_cast<LoadInst>(User)) {
          replaceResLd(ldInst, NewGV);
        } else if (GEPOperator *GEP = dyn_cast<GEPOperator>(User)) {
          IRBuilder<> Builder(GEP->getContext());
          StringRef Name = "";
          if (Instruction *I = dyn_cast<Instruction>(GEP)) {
            Builder.SetInsertPoint(I);
            Name = GEP->getName();
          }
          SmallVector<Value *, 8> Indices(GEP->idx_begin(), GEP->idx_end());
          Value *NewPtr = Builder.CreateGEP(NewGV, Indices);
          for (auto GEPUserIt = GEP->user_begin(), GEPuserEnd = GEP->user_end();
               GEPUserIt != GEPuserEnd;) {
            Value *User = *(GEPUserIt++);
            if (LoadInst *ldInst = dyn_cast<LoadInst>(User)) {
              replaceResLd(ldInst, NewPtr);
            } else {
              User->dump();
              DXASSERT(0, "unsupported user when update resouce type");
            }
          }
          if (Instruction *I = dyn_cast<Instruction>(GEP))
            I->eraseFromParent();
        } else {
          User->dump();
          DXASSERT(0,"unsupported user when update resouce type");
        }
      }
    } else {
      // If not a library, the GV should be deleted
      for (auto UserIt = Symbol->user_begin(); UserIt != Symbol->user_end();) {
        Value *User = *(UserIt++);

        if (Instruction *I = dyn_cast<Instruction>(User)) {
          if (!User->user_empty())
            I->replaceAllUsesWith(UndefValue::get(I->getType()));

          I->eraseFromParent();
        } else {
          ConstantExpr *CE = cast<ConstantExpr>(User);
          if (!CE->user_empty())
            CE->replaceAllUsesWith(UndefValue::get(CE->getType()));
        }
      }
    }
    Symbol->removeDeadConstantUsers();

    if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Symbol))
      GV->eraseFromParent();

    return true;
  }

  return false;
}

bool UpdateStructTypeForLegacyLayoutOnDM(DxilModule &DM) {
  DxilTypeSystem &TypeSys = DM.GetTypeSystem();
  bool bChanged = false;
  for (auto &CBuf : DM.GetCBuffers()) {
    bChanged |= UpdateStructTypeForLegacyLayout(*CBuf.get(), TypeSys, DM);
  }

  for (auto &UAV : DM.GetUAVs()) {
    if (DXIL::IsStructuredBuffer(UAV->GetKind()))
      bChanged |= UpdateStructTypeForLegacyLayout(*UAV.get(), TypeSys, DM);
  }

  for (auto &SRV : DM.GetSRVs()) {
    if (SRV->IsStructuredBuffer() || SRV->IsTBuffer())
      bChanged |= UpdateStructTypeForLegacyLayout(*SRV.get(), TypeSys, DM);
  }

  return bChanged;
}

} // namespace

void DxilLowerCreateHandleForLib::FailOnPoisonResources() {
  // A previous pass replaced all undef resources with constant zero resources.
  // If those made it here, the program is malformed.
  for (Function &Func : this->m_DM->GetModule()->functions()) {
    hlsl::OP::OpCodeClass OpcodeClass;
    if (m_DM->GetOP()->GetOpCodeClass(&Func, OpcodeClass)
      && OpcodeClass == OP::OpCodeClass::CreateHandleForLib) {
      Type *ResTy = Func.getFunctionType()->getParamType(
        DXIL::OperandIndex::kCreateHandleForLibResOpIdx);
      Constant *PoisonRes = ConstantAggregateZero::get(ResTy);
      for (User *PoisonUser : PoisonRes->users())
        if (Instruction *PoisonUserInst = dyn_cast<Instruction>(PoisonUser))
          dxilutil::EmitResMappingError(PoisonUserInst);
    }
  }
}

bool DxilLowerCreateHandleForLib::UpdateStructTypeForLegacyLayout() {
  return UpdateStructTypeForLegacyLayoutOnDM(*m_DM);
}

// Change ResourceSymbol to undef if don't need.
void DxilLowerCreateHandleForLib::UpdateResourceSymbols() {
  auto UpdateResourceSymbol = [](DxilResourceBase *res) {
    if (GlobalVariable *GV = dyn_cast<GlobalVariable>(res->GetGlobalSymbol())) {
      GV->removeDeadConstantUsers();
      DXASSERT(GV->user_empty(), "else resource not lowered");
      res->SetGlobalSymbol(UndefValue::get(GV->getType()));
      if (GV->user_empty())
        GV->eraseFromParent();
    }
  };

  for (auto &&C : m_DM->GetCBuffers()) {
    UpdateResourceSymbol(C.get());
  }
  for (auto &&Srv : m_DM->GetSRVs()) {
    UpdateResourceSymbol(Srv.get());
  }
  for (auto &&Uav : m_DM->GetUAVs()) {
    UpdateResourceSymbol(Uav.get());
  }
  for (auto &&S : m_DM->GetSamplers()) {
    UpdateResourceSymbol(S.get());
  }
}

// Lower createHandleForLib
namespace {

Value *flattenGepIdx(GEPOperator *GEP) {
  Value *idx = nullptr;
  if (GEP->getNumIndices() == 2) {
    // one dim array of resource
    idx = (GEP->idx_begin() + 1)->get();
  } else {
    gep_type_iterator GEPIt = gep_type_begin(GEP), E = gep_type_end(GEP);
    // Must be instruction for multi dim array.
    std::unique_ptr<IRBuilder<>> Builder;
    if (GetElementPtrInst *GEPInst = dyn_cast<GetElementPtrInst>(GEP)) {
      Builder = llvm::make_unique<IRBuilder<>>(GEPInst);
    } else {
      Builder = llvm::make_unique<IRBuilder<>>(GEP->getContext());
    }
    for (; GEPIt != E; ++GEPIt) {
      if (GEPIt->isArrayTy()) {
        unsigned arraySize = GEPIt->getArrayNumElements();
        Value *tmpIdx = GEPIt.getOperand();
        if (idx == nullptr)
          idx = tmpIdx;
        else {
          idx = Builder->CreateMul(idx, Builder->getInt32(arraySize));
          idx = Builder->CreateAdd(idx, tmpIdx);
        }
      }
    }
  }
  return idx;
}

} // namespace


void DxilLowerCreateHandleForLib::ReplaceResourceUserWithHandle(
    DxilResource &res, LoadInst *load, Instruction *handle) {
  for (auto resUser = load->user_begin(), E = load->user_end(); resUser != E;) {
    Value *V = *(resUser++);
    CallInst *CI = dyn_cast<CallInst>(V);
    DxilInst_CreateHandleForLib createHandle(CI);
    DXASSERT(createHandle, "must be createHandle");
    CI->replaceAllUsesWith(handle);
    CI->eraseFromParent();
  }

  if (res.GetClass() == DXIL::ResourceClass::UAV) {
    // Before this pass, the global resources might not have been mapped with
    // all the uses. Now we're 100% sure who uses what resources (otherwise the
    // compilation would have failed), so we do a round on marking UAV's as
    // having counter.
    static auto IsDxilOp = [](Value *V, hlsl::OP::OpCode Op) -> bool {
      Instruction *I = dyn_cast<Instruction>(V);
      if (!I)
        return false;
      return hlsl::OP::IsDxilOpFuncCallInst(I, Op);
    };

    // Search all users for update counter
    bool updateAnnotateHandle = res.IsGloballyCoherent();
    if (!res.HasCounter()) {
      for (User *U : handle->users()) {
        if (IsDxilOp(U, hlsl::OP::OpCode::BufferUpdateCounter)) {
          res.SetHasCounter(true);
          break;
        } else if (IsDxilOp(U, hlsl::OP::OpCode::AnnotateHandle)) {
          for (User *UU : U->users()) {
            if (IsDxilOp(UU, hlsl::OP::OpCode::BufferUpdateCounter)) {
              res.SetHasCounter(true);
              updateAnnotateHandle = true;
              break;
            }
          }
          if (updateAnnotateHandle)
            break;
        }
      }
    }
    if (updateAnnotateHandle) {
      // Update resource props with counter flag
      DxilResourceProperties RP =
          resource_helper::loadPropsFromResourceBase(&res);
      // Require ShaderModule to reconstruct resource property constant
      const ShaderModel *pSM = m_DM->GetShaderModel();

      SmallVector<Instruction *, 4> annotHandles;
      for (User *U : handle->users()) {
        DxilInst_AnnotateHandle annotateHandle(cast<Instruction>(U));
        if (annotateHandle) {
          annotHandles.emplace_back(cast<Instruction>(U));
        }
      }
      if (!annotHandles.empty()) {
        Instruction *firstAnnot = annotHandles.pop_back_val();
        DxilInst_AnnotateHandle annotateHandle(firstAnnot);
        // Update props.
        Constant *propsConst = resource_helper::getAsConstant(
            RP, annotateHandle.get_props()->getType(), *pSM);
        annotateHandle.set_props(propsConst);
        if (!annotHandles.empty()) {
          // Move firstAnnot after handle.
          firstAnnot->removeFromParent();
          firstAnnot->insertAfter(handle);
          // Remove redundant annotate handles.
          for (auto *annotHdl : annotHandles) {
            annotHdl->replaceAllUsesWith(firstAnnot);
            annotHdl->eraseFromParent();
          }
        }
      }
    }
  }

  load->eraseFromParent();
}

void DxilLowerCreateHandleForLib::TranslateDxilResourceUses(
    DxilResourceBase &res) {
  OP *hlslOP = m_DM->GetOP();
  // Generate createHandleFromBinding for sm66 and later.
  bool bCreateFromBinding = m_DM->GetShaderModel()->IsSM66Plus();
  OP::OpCode createOp = bCreateFromBinding ? OP::OpCode::CreateHandleFromBinding
                                           : OP::OpCode::CreateHandle;
  Function *createHandle = hlslOP->GetOpFunc(
      createOp, llvm::Type::getVoidTy(m_DM->GetCtx()));
  Value *opArg = hlslOP->GetU32Const((unsigned)createOp);

  bool isViewResource = res.GetClass() == DXIL::ResourceClass::SRV ||
                        res.GetClass() == DXIL::ResourceClass::UAV;
  bool isROV = isViewResource && static_cast<DxilResource &>(res).IsROV();
  std::string handleName =
      (res.GetGlobalName() + Twine("_") + Twine(res.GetResClassName())).str();
  if (isViewResource)
    handleName += (Twine("_") + Twine(res.GetResDimName())).str();
  if (isROV)
    handleName += "_ROV";

  Value *resClassArg = hlslOP->GetU8Const(
      static_cast<std::underlying_type<DxilResourceBase::Class>::type>(
          res.GetClass()));
  Value *resIDArg = hlslOP->GetU32Const(res.GetID());
  // resLowerBound will be added after allocation in DxilCondenseResources.
  Value *resLowerBound = hlslOP->GetU32Const(res.GetLowerBound());

  Value *isUniformRes = hlslOP->GetI1Const(0);

  Value *GV = res.GetGlobalSymbol();
  DXASSERT(isa<GlobalValue>(GV), "DxilLowerCreateHandleForLib cannot deal with unused resources.");

  Module *pM = m_DM->GetModule();
  // TODO: add debug info to create handle.
  DIVariable *DIV = nullptr;
  DILocation *DL = nullptr;
  if (m_HasDbgInfo) {
    DebugInfoFinder &Finder = m_DM->GetOrCreateDebugInfoFinder();
    DIV = dxilutil::FindGlobalVariableDebugInfo(cast<GlobalVariable>(GV), Finder);
    if (DIV)
      // TODO: how to get col?
      DL =
          DILocation::get(pM->getContext(), DIV->getLine(), 1, DIV->getScope());
  }

  bool isResArray = res.GetRangeSize() > 1;
  std::unordered_map<Function *, Instruction *> handleMapOnFunction;

  Value *createHandleArgs[] = {opArg, resClassArg, resIDArg, resLowerBound,
                               isUniformRes};

  DxilResourceBinding binding = resource_helper::loadBindingFromResourceBase(&res);
  Value *bindingV = resource_helper::getAsConstant(
      binding, hlslOP->GetResourceBindingType(), *m_DM->GetShaderModel());

  Value *createHandleFromBindingArgs[] = {opArg, bindingV, resLowerBound, isUniformRes};

  MutableArrayRef<Value *> Args(bCreateFromBinding ? createHandleFromBindingArgs
                                                   : createHandleArgs,
                                bCreateFromBinding ? 4 : 5);

  const unsigned resIdxOpIdx = bCreateFromBinding
                                   ? DxilInst_CreateHandleFromBinding::arg_index
                                   : DxilInst_CreateHandle::arg_index;
  const unsigned nonUniformOpIdx = bCreateFromBinding
                                   ? DxilInst_CreateHandleFromBinding::arg_nonUniformIndex
                                   : DxilInst_CreateHandle::arg_nonUniformIndex;




  for (iplist<Function>::iterator F : pM->getFunctionList()) {
    if (!F->isDeclaration()) {
      if (!isResArray) {
        IRBuilder<> Builder(dxilutil::FindAllocaInsertionPt(F));
        if (m_HasDbgInfo) {
          // TODO: set debug info.
          // Builder.SetCurrentDebugLocation(DL);
        }
        handleMapOnFunction[F] =
            Builder.CreateCall(createHandle, Args, handleName);
      }
    }
  }

  for (auto U = GV->user_begin(), E = GV->user_end(); U != E;) {
    User *user = *(U++);
    // Skip unused user.
    if (user->user_empty())
      continue;

    if (LoadInst *ldInst = dyn_cast<LoadInst>(user)) {
      Function *userF = ldInst->getParent()->getParent();
      DXASSERT(handleMapOnFunction.count(userF), "must exist");
      Instruction *handle = handleMapOnFunction[userF];
      ReplaceResourceUserWithHandle(static_cast<DxilResource &>(res), ldInst, handle);
    } else if (GEPOperator *GEP = dyn_cast<GEPOperator>(user)) {
      Value *idx = flattenGepIdx(GEP);

      Args[resIdxOpIdx] = idx;

      Args[nonUniformOpIdx] =
          isUniformRes;

      Instruction *handle = nullptr;
      if (GetElementPtrInst *GEPInst = dyn_cast<GetElementPtrInst>(GEP)) {
        IRBuilder<> Builder = IRBuilder<>(GEPInst);
        if (DxilMDHelper::IsMarkedNonUniform(GEPInst)) {
          // Mark nonUniform.
          Args[nonUniformOpIdx] =
              hlslOP->GetI1Const(1);
          // Clear nonUniform on GEP.
          GEPInst->setMetadata(DxilMDHelper::kDxilNonUniformAttributeMDName, nullptr);
        }
        Args[resIdxOpIdx] = Builder.CreateAdd(idx, resLowerBound);
        handle = Builder.CreateCall(createHandle, Args, handleName);
      }

      for (auto GEPU = GEP->user_begin(), GEPE = GEP->user_end();
           GEPU != GEPE;) {
        // Must be load inst.
        LoadInst *ldInst = cast<LoadInst>(*(GEPU++));
        if (handle) {
          ReplaceResourceUserWithHandle(static_cast<DxilResource &>(res), ldInst, handle);
        } else {
          IRBuilder<> Builder = IRBuilder<>(ldInst);
          Args[resIdxOpIdx] = Builder.CreateAdd(idx, resLowerBound);
          Instruction *localHandle =
              Builder.CreateCall(createHandle, Args, handleName);
          ReplaceResourceUserWithHandle(static_cast<DxilResource &>(res), ldInst, localHandle);
        }
      }

      if (Instruction *I = dyn_cast<Instruction>(GEP)) {
        I->eraseFromParent();
      }
    } else if (BitCastInst *BCI = dyn_cast<BitCastInst>(user)) {
      DXASSERT(onlyUsedByLifetimeMarkers(BCI),
               "expected bitcast to only be used by lifetime intrinsics");
      for (auto BCIU = BCI->user_begin(), BCIE = BCI->user_end(); BCIU != BCIE;) {
        IntrinsicInst *II = cast<IntrinsicInst>(*(BCIU++));
        II->eraseFromParent();
      }
      BCI->eraseFromParent();
    } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(user)) {
      // A GEPOperator can also be a ConstantExpr, so it must be checked before
      // this code.
      DXASSERT(CE->getOpcode() == Instruction::BitCast, "expected bitcast");
      DXASSERT(onlyUsedByLifetimeMarkers(CE),
               "expected ConstantExpr to only be used by lifetime intrinsics");
      for (auto CEU = CE->user_begin(), CEE = CE->user_end(); CEU != CEE;) {
        IntrinsicInst *II = cast<IntrinsicInst>(*(CEU++));
        II->eraseFromParent();
      }
    } else {
      DXASSERT(false,
               "AddOpcodeParamForIntrinsic in CodeGen did not patch uses "
               "to only have ld/st refer to temp object");
    }
  }
  // Erase unused handle.
  for (auto It : handleMapOnFunction) {
    Instruction *I = It.second;
    if (I->user_empty())
      I->eraseFromParent();
  }
}

void DxilLowerCreateHandleForLib::GenerateDxilResourceHandles() {
  for (size_t i = 0; i < m_DM->GetCBuffers().size(); i++) {
    DxilCBuffer &C = m_DM->GetCBuffer(i);
    TranslateDxilResourceUses(C);
  }
  // Create sampler handle first, may be used by SRV operations.
  for (size_t i = 0; i < m_DM->GetSamplers().size(); i++) {
    DxilSampler &S = m_DM->GetSampler(i);
    TranslateDxilResourceUses(S);
  }

  for (size_t i = 0; i < m_DM->GetSRVs().size(); i++) {
    DxilResource &SRV = m_DM->GetSRV(i);
    TranslateDxilResourceUses(SRV);
  }

  for (size_t i = 0; i < m_DM->GetUAVs().size(); i++) {
    DxilResource &UAV = m_DM->GetUAV(i);
    TranslateDxilResourceUses(UAV);
  }
}

// TBuffer.
namespace {
void InitTBuffer(const DxilCBuffer *pSource, DxilResource *pDest) {
  pDest->SetKind(pSource->GetKind());
  pDest->SetCompType(DXIL::ComponentType::U32);
  pDest->SetSampleCount(0);
  pDest->SetElementStride(0);
  pDest->SetGloballyCoherent(false);
  pDest->SetHasCounter(false);
  pDest->SetRW(false);
  pDest->SetROV(false);
  pDest->SetID(pSource->GetID());
  pDest->SetSpaceID(pSource->GetSpaceID());
  pDest->SetLowerBound(pSource->GetLowerBound());
  pDest->SetRangeSize(pSource->GetRangeSize());
  pDest->SetGlobalSymbol(pSource->GetGlobalSymbol());
  pDest->SetGlobalName(pSource->GetGlobalName());
  pDest->SetHandle(pSource->GetHandle());
  pDest->SetHLSLType(pSource->GetHLSLType());
}

void PatchTBufferLoad(CallInst *handle, DxilModule &DM,
                      DenseSet<Value *> &patchedSet) {
  if (patchedSet.count(handle))
    return;
  patchedSet.insert(handle);
  hlsl::OP *hlslOP = DM.GetOP();
  llvm::LLVMContext &Ctx = DM.GetCtx();
  Type *doubleTy = Type::getDoubleTy(Ctx);
  Type *i64Ty = Type::getInt64Ty(Ctx);

  // Replace corresponding cbuffer loads with typed buffer loads
  for (auto U = handle->user_begin(); U != handle->user_end();) {
    User *user = *(U++);
    CallInst *I = dyn_cast<CallInst>(user);
    // Could also be store for out arg in lib.
    DXASSERT(isa<StoreInst>(user) || (I && OP::IsDxilOpFuncCallInst(I)),
             "otherwise unexpected user of CreateHandle value");
    if (!I)
      continue;
    DXIL::OpCode opcode = OP::GetDxilOpFuncCallInst(I);
    if (opcode == DXIL::OpCode::CBufferLoadLegacy) {
      DxilInst_CBufferLoadLegacy cbLoad(I);

      StructType *cbRetTy = cast<StructType>(I->getType());
      // elements will be 4, or 8 for native 16-bit types, which require special handling.
      bool cbRet8Elt = cbRetTy->getNumElements() > 4;

      // Replace with appropriate buffer load instruction
      IRBuilder<> Builder(I);
      opcode = OP::OpCode::BufferLoad;
      Type *Ty = Type::getInt32Ty(Ctx);
      Function *BufLoad = hlslOP->GetOpFunc(opcode, Ty);
      Constant *opArg = hlslOP->GetU32Const((unsigned)opcode);
      Value *undefI = UndefValue::get(Type::getInt32Ty(Ctx));
      Value *offset = cbLoad.get_regIndex();
      CallInst *load =
          Builder.CreateCall(BufLoad, {opArg, handle, offset, undefI});

      // Find extractelement uses of cbuffer load and replace + generate bitcast
      // as necessary
      for (auto LU = I->user_begin(); LU != I->user_end();) {
        ExtractValueInst *evInst = dyn_cast<ExtractValueInst>(*(LU++));
        DXASSERT(evInst && evInst->getNumIndices() == 1,
                 "user of cbuffer load result should be extractvalue");
        uint64_t idx = evInst->getIndices()[0];
        Type *EltTy = evInst->getType();
        IRBuilder<> EEBuilder(evInst);
        Value *result = nullptr;
        if (EltTy != Ty) {
          // extract two values and DXIL::OpCode::MakeDouble or construct i64
          if ((EltTy == doubleTy) || (EltTy == i64Ty)) {
            DXASSERT(idx < 2, "64-bit component index out of range");

            // This assumes big endian order in tbuffer elements (is this
            // correct?)
            Value *low = EEBuilder.CreateExtractValue(load, idx * 2);
            Value *high = EEBuilder.CreateExtractValue(load, idx * 2 + 1);
            if (EltTy == doubleTy) {
              opcode = OP::OpCode::MakeDouble;
              Function *MakeDouble = hlslOP->GetOpFunc(opcode, doubleTy);
              Constant *opArg = hlslOP->GetU32Const((unsigned)opcode);
              result = EEBuilder.CreateCall(MakeDouble, {opArg, low, high});
            } else {
              high = EEBuilder.CreateZExt(high, i64Ty);
              low = EEBuilder.CreateZExt(low, i64Ty);
              high = EEBuilder.CreateShl(high, hlslOP->GetU64Const(32));
              result = EEBuilder.CreateOr(high, low);
            }
          } else {
            if (cbRet8Elt) {
              DXASSERT_NOMSG(cbRetTy->getNumElements() == 8);
              DXASSERT_NOMSG(EltTy->getScalarSizeInBits() == 16);
              // Translate extract from 16bit x 8 to extract and translate from i32 by 4
              result = EEBuilder.CreateExtractValue(load, idx >> 1);
              if (idx & 1)
                result = EEBuilder.CreateLShr(result, 16);
              result = EEBuilder.CreateTrunc(result, Type::getInt16Ty(Ctx));
              if (EltTy->isHalfTy())
                result = EEBuilder.CreateBitCast(result, EltTy);
            } else {
              result = EEBuilder.CreateExtractValue(load, idx);
              if (Ty->getScalarSizeInBits() > EltTy->getScalarSizeInBits()) {
                if (EltTy->isIntegerTy()) {
                  result = EEBuilder.CreateTrunc(result, EltTy);
                } else {
                  result = EEBuilder.CreateBitCast(result, Type::getFloatTy(Ctx));
                  result = EEBuilder.CreateFPTrunc(result, EltTy);
                }
              } else {
                result = EEBuilder.CreateBitCast(result, EltTy);
              }
            }
          }
        } else {
          result = EEBuilder.CreateExtractValue(load, idx);
        }

        evInst->replaceAllUsesWith(result);
        evInst->eraseFromParent();
      }
    } else if (opcode == DXIL::OpCode::CBufferLoad) {
      // TODO: Handle this, or prevent this for tbuffer
      DXASSERT(false, "otherwise CBufferLoad used for tbuffer rather than "
                      "CBufferLoadLegacy");
    } else if (opcode == DXIL::OpCode::AnnotateHandle) {
      PatchTBufferLoad(cast<CallInst>(I), DM,
                       patchedSet);
      continue;
    } else if (opcode == DXIL::OpCode::BufferLoad) {
      // Already translated, skip.
      continue;
    } else {
      DXASSERT(false, "otherwise unexpected user of CreateHandle value");
    }
    I->eraseFromParent();
  }
}

} // namespace

void DxilLowerCreateHandleForLib::PatchTBufferUse(
    Value *V, DxilModule &DM, DenseSet<Value *> &patchedSet) {
  for (User *U : V->users()) {
    if (CallInst *CI = dyn_cast<CallInst>(U)) {
      // Patch dxil call.
      if (hlsl::OP::IsDxilOpFuncCallInst(CI))
        PatchTBufferLoad(CI, DM, patchedSet);
    } else {
      PatchTBufferUse(U, DM, patchedSet);
    }
  }
}

bool DxilLowerCreateHandleForLib::PatchDynamicTBuffers(DxilModule &DM) {
  hlsl::OP *hlslOP = DM.GetOP();
  Function *AnnotHandleFn = hlslOP->GetOpFunc(DXIL::OpCode::AnnotateHandle,
                                              Type::getVoidTy(DM.GetCtx()));
  if (AnnotHandleFn->user_empty()) {
    AnnotHandleFn->eraseFromParent();
    return false;
  }
  bool bUpdated = false;
  for (User *U : AnnotHandleFn->users()) {
    CallInst *CI = cast<CallInst>(U);
    DxilInst_AnnotateHandle annot(CI);
    DxilResourceProperties RP = resource_helper::loadPropsFromAnnotateHandle(
        annot, *DM.GetShaderModel());

    if (RP.getResourceKind() != DXIL::ResourceKind::TBuffer)
      continue;
    // Skip handle from createHandleForLib which take care in PatchTBuffers.
    if (CallInst *HdlCI = dyn_cast<CallInst>(annot.get_res())) {
      if (hlslOP->IsDxilOpFuncCallInst(HdlCI)) {
        if (hlslOP->GetDxilOpFuncCallInst(HdlCI) == DXIL::OpCode::CreateHandleForLib)
          continue;
      }
    }

    DenseSet<Value *> patchedSet;
    PatchTBufferLoad(CI, DM, patchedSet);
    bUpdated = true;
  }
  return bUpdated;
}

bool DxilLowerCreateHandleForLib::PatchTBuffers(DxilModule &DM) {
  bool bChanged = false;
  // move tbuffer resources to SRVs
  Module &M = *DM.GetModule();
  const ShaderModel &SM = *DM.GetShaderModel();
  DenseSet<Value*> patchedSet;

  // First, patch users of AnnotateHandle calls if we have them.
  // This will pick up uses in lib_6_x functions that otherwise
  // would be missed.
  if (SM.IsSM66Plus()) {
    for (auto it : DM.GetOP()->GetOpFuncList(DXIL::OpCode::AnnotateHandle)) {
      Function *F = it.second;
      for (auto U = F->user_begin(); U != F->user_end(); ) {
        User *user = *(U++);
        if (CallInst *CI = dyn_cast<CallInst>(user)) {
          DxilInst_AnnotateHandle AH(CI);
          if (AH) {
            DxilResourceProperties RP = resource_helper::loadPropsFromAnnotateHandle(AH, SM);
            if (RP.getResourceKind() == DXIL::ResourceKind::TBuffer)
              PatchTBufferLoad(CI, DM, patchedSet);
          }
        }
      }
    }
  }

  unsigned offset = DM.GetSRVs().size();
  for (auto it = DM.GetCBuffers().begin(); it != DM.GetCBuffers().end(); it++) {
    DxilCBuffer *CB = it->get();
    if (CB->GetKind() == DXIL::ResourceKind::TBuffer) {
      auto srv = make_unique<DxilResource>();
      InitTBuffer(CB, srv.get());
      srv->SetID(offset++);
      DM.AddSRV(std::move(srv));
      GlobalVariable *GV = dyn_cast<GlobalVariable>(CB->GetGlobalSymbol());
      if (GV == nullptr)
        continue;
      PatchTBufferUse(GV, DM, patchedSet);
      // Set global symbol for cbuffer to an unused value so it can be removed
      // in RemoveUnusedResourceSymbols.
      Type *Ty = GV->getType()->getElementType();
      GlobalVariable *NewGV = new GlobalVariable(
          M, Ty, GV->isConstant(), GV->getLinkage(), /*Initializer*/ nullptr,
          GV->getName(),
          /*InsertBefore*/ nullptr, GV->getThreadLocalMode(),
          GV->getType()->getAddressSpace(), GV->isExternallyInitialized());
      CB->SetGlobalSymbol(NewGV);
      bChanged = true;
    }
  }
  return bChanged;
}

typedef DenseMap<Value*, unsigned> OffsetForValueMap;

// Find the imm offset part from a value.
// It must exist unless offset is 0.
static unsigned GetCBOffset(Value *V, OffsetForValueMap &visited) {
  auto it = visited.find(V);
  if (it != visited.end())
    return it->second;
  visited[V] = 0;
  unsigned result = 0;
  if (ConstantInt *Imm = dyn_cast<ConstantInt>(V)) {
    result = Imm->getLimitedValue();
  } else if (BinaryOperator *BO = dyn_cast<BinaryOperator>(V)) {
    switch (BO->getOpcode()) {
    case Instruction::Add: {
      unsigned left = GetCBOffset(BO->getOperand(0), visited);
      unsigned right = GetCBOffset(BO->getOperand(1), visited);
      result = left + right;
    } break;
    case Instruction::Or: {
      unsigned left = GetCBOffset(BO->getOperand(0), visited);
      unsigned right = GetCBOffset(BO->getOperand(1), visited);
      result = left | right;
    } break;
    default:
      break;
    }
  } else if (SelectInst *SI = dyn_cast<SelectInst>(V)) {
    result = std::min(GetCBOffset(SI->getOperand(1), visited),
                      GetCBOffset(SI->getOperand(2), visited));
  } else if (PHINode *PN = dyn_cast<PHINode>(V)) {
    result = UINT_MAX;
    for (unsigned i = 0, ops = PN->getNumIncomingValues(); i < ops; ++i) {
      result = std::min(result, GetCBOffset(PN->getIncomingValue(i), visited));
    }
  }
  visited[V] = result;
  return result;
}

typedef std::map<unsigned, DxilFieldAnnotation*> FieldAnnotationByOffsetMap;

// Returns size in bits of the field if it's a basic type, otherwise 0.
static unsigned MarkCBUse(unsigned offset, FieldAnnotationByOffsetMap &fieldMap) {
  auto it = fieldMap.upper_bound(offset);
  it--;
  if (it != fieldMap.end()) {
    it->second->SetCBVarUsed(true);
    return it->second->GetCompType().GetSizeInBits();
  }
  return 0;
}

// Detect patterns of lshr v,16 or trunc to 16-bits and return low and high
// word usage.
static const unsigned kLowWordUsed = 1;
static const unsigned kHighWordUsed = 2;
static const unsigned kLowHighWordMask = kLowWordUsed | kHighWordUsed;
static unsigned DetectLowAndHighWordUsage(ExtractValueInst *EV) {
  unsigned result = 0;
  if (EV->getType()->getScalarSizeInBits() == 32) {
    for (auto U : EV->users()) {
      Instruction *I = cast<Instruction>(U);
      if (I->getOpcode() == Instruction::LShr) {
        ConstantInt *CShift = dyn_cast<ConstantInt>(I->getOperand(1));
        if (CShift && CShift->getLimitedValue() == 16)
          result |= kHighWordUsed;
      } else if (I->getOpcode() == Instruction::Trunc &&
               I->getType()->getPrimitiveSizeInBits() == 16) {
        result |= kLowWordUsed;
      } else {
        // Assume whole dword is used, return 0
        return 0;
      }
      if ((result & kLowHighWordMask) == kLowHighWordMask)
        break;
    }
  }
  return result;
}

static unsigned GetOffsetForCBExtractValue(ExtractValueInst *EV, bool bMinPrecision,
                                           unsigned &lowHighWordUsage) {
  DXASSERT(EV->getNumIndices() == 1, "otherwise, unexpected indices/type for extractvalue");
  unsigned typeSize = 4;
  unsigned bits = EV->getType()->getScalarSizeInBits();
  if (bits == 64)
    typeSize = 8;
  else if (bits == 16 && !bMinPrecision)
    typeSize = 2;
  lowHighWordUsage = DetectLowAndHighWordUsage(EV);
  return (EV->getIndices().front() * typeSize);
}

// Marks up to two CB uses for the case where only 16-bit type(s)
// are being used from lower or upper word of a tbuffer load,
// which is always 4 x 32 instead of 8 x 16, like cbuffer.
static void MarkCBUsesForExtractElement(
    unsigned offset, FieldAnnotationByOffsetMap &fieldMap,
    ExtractValueInst *EV, bool bMinPrecision) {
  unsigned lowHighWordUsage = 0;
  unsigned evOffset = GetOffsetForCBExtractValue(EV, bMinPrecision, lowHighWordUsage);

  // For tbuffer, where value extracted is always 32-bits:
  // If lowHighWordUsage is 0, it means 32-bits used.
  // If field marked is < 32 bits, we still need to mark the high 16-bits as
  // used, in case there is another 16-bit field.
  // Since MarkCBUse could return 0 on non-basic type field, look for 16
  // when determining whether we still need to mark high word as used.
  bool highUnmarked = EV->getType()->getScalarSizeInBits() == 32;
  if (!lowHighWordUsage || 0 != (lowHighWordUsage & kLowWordUsed))
    highUnmarked &= MarkCBUse(offset + evOffset, fieldMap) == 16;
  if (highUnmarked && (!lowHighWordUsage || 0 != (lowHighWordUsage & kHighWordUsed)))
    MarkCBUse(offset + evOffset + 2, fieldMap);
}

static void CollectInPhiChain(PHINode *cbUser, unsigned offset,
                              std::unordered_set<Value *> &userSet,
                              FieldAnnotationByOffsetMap &fieldMap,
                              bool bMinPrecision) {
  if (userSet.count(cbUser) > 0)
    return;

  userSet.insert(cbUser);
  for (User *cbU : cbUser->users()) {
    if (ExtractValueInst *EV = dyn_cast<ExtractValueInst>(cbU)) {
      MarkCBUsesForExtractElement(offset, fieldMap, EV, bMinPrecision);
    } else {
      PHINode *phi = cast<PHINode>(cbU);
      CollectInPhiChain(phi, offset, userSet, fieldMap, bMinPrecision);
    }
  }
}

static void CollectCBufferMemberUsage(Value *V,
                                      FieldAnnotationByOffsetMap &legacyFieldMap,
                                      FieldAnnotationByOffsetMap &newFieldMap,
                                      hlsl::OP *hlslOP, bool bMinPrecision,
                                      OffsetForValueMap &visited) {
  for (auto U : V->users()) {
    if (Constant *C = dyn_cast<Constant>(U)) {
      CollectCBufferMemberUsage(C, legacyFieldMap, newFieldMap, hlslOP, bMinPrecision, visited);
    } else if (LoadInst *LI = dyn_cast<LoadInst>(U)) {
      CollectCBufferMemberUsage(U, legacyFieldMap, newFieldMap, hlslOP, bMinPrecision, visited);
    } else if (CallInst *CI = dyn_cast<CallInst>(U)) {
      if (hlslOP->IsDxilOpFuncCallInst(CI)) {
        hlsl::OP::OpCode op = hlslOP->GetDxilOpFuncCallInst(CI);
        if (op == DXIL::OpCode::CreateHandleForLib) {
          CollectCBufferMemberUsage(U, legacyFieldMap, newFieldMap, hlslOP, bMinPrecision, visited);
        } else if (op == DXIL::OpCode::AnnotateHandle) {
          CollectCBufferMemberUsage(U, legacyFieldMap, newFieldMap, hlslOP,
                                    bMinPrecision, visited);
        } else if (op == DXIL::OpCode::CBufferLoadLegacy ||
                   op == DXIL::OpCode::BufferLoad) {
          Value *resIndex = (op == DXIL::OpCode::CBufferLoadLegacy) ?
            DxilInst_CBufferLoadLegacy(CI).get_regIndex() :
            DxilInst_BufferLoad(CI).get_index();
          unsigned offset = GetCBOffset(resIndex, visited) << 4;
          for (User *cbU : U->users()) {
            if (ExtractValueInst *EV = dyn_cast<ExtractValueInst>(cbU)) {
              MarkCBUsesForExtractElement(offset, legacyFieldMap, EV, bMinPrecision);
            } else {
              PHINode *phi = cast<PHINode>(cbU);
              std::unordered_set<Value *> userSet;
              CollectInPhiChain(phi, offset, userSet, legacyFieldMap, bMinPrecision);
            }
          }
        } else if (op == DXIL::OpCode::CBufferLoad) {
          DxilInst_CBufferLoad cbload(CI);
          Value *byteOffset = cbload.get_byteOffset();
          unsigned offset = GetCBOffset(byteOffset, visited);
          MarkCBUse(offset, newFieldMap);
        }
      }
    }
  }
}

void DxilLowerCreateHandleForLib::UpdateCBufferUsage() {
  DxilTypeSystem &TypeSys = m_DM->GetTypeSystem();
  hlsl::OP *hlslOP = m_DM->GetOP();
  const DataLayout &DL = m_DM->GetModule()->getDataLayout();
  const auto &CBuffers = m_DM->GetCBuffers();
  OffsetForValueMap visited;

  SmallVector<std::pair<GlobalVariable*, Type*>, 4> CBufferVars;

  // Collect cbuffers
  for (auto it = CBuffers.begin(); it != CBuffers.end(); it++) {
    DxilCBuffer *CB = it->get();
    GlobalVariable *GV = dyn_cast<GlobalVariable>(CB->GetGlobalSymbol());
    if (GV == nullptr)
      continue;
    CBufferVars.emplace_back(GV, CB->GetHLSLType());
  }

  // Collect tbuffers
  for (auto &it : m_DM->GetSRVs()) {
    if (it->GetKind() != DXIL::ResourceKind::TBuffer)
      continue;
    GlobalVariable *GV = dyn_cast<GlobalVariable>(it->GetGlobalSymbol());
    if (GV == nullptr)
      continue;
    CBufferVars.emplace_back(GV, it->GetHLSLType());
  }

  for (auto GV_Ty : CBufferVars) {
    auto GV = GV_Ty.first;
    Type *ElemTy = GV_Ty.second->getPointerElementType();
    ElemTy = dxilutil::StripArrayTypes(ElemTy, nullptr);
    StructType *ST = cast<StructType>(ElemTy);
    DxilStructAnnotation *SA = TypeSys.GetStructAnnotation(ST);
    if (SA == nullptr)
      continue;
    // If elements < 2, it's used if it exists.
    // Only old-style cbuffer { ... } will have more than one member, and
    // old-style cbuffers are the only ones that report usage per member.
    if (ST->getStructNumElements() < 2) {
      continue;
    }

    // Create offset maps for legacy layout and new compact layout, while resetting usage flags
    const StructLayout *SL = DL.getStructLayout(ST);
    FieldAnnotationByOffsetMap legacyFieldMap, newFieldMap;
    for (unsigned i = 0; i < SA->GetNumFields(); ++i) {
      DxilFieldAnnotation &FA = SA->GetFieldAnnotation(i);
      FA.SetCBVarUsed(false);
      legacyFieldMap[FA.GetCBufferOffset()] = &FA;
      newFieldMap[(unsigned)SL->getElementOffset(i)] = &FA;
    }
    CollectCBufferMemberUsage(GV, legacyFieldMap, newFieldMap, hlslOP, m_DM->GetUseMinPrecision(), visited);
 }
}

void DxilLowerCreateHandleForLib::SetNonUniformIndexForDynamicResource(
    DxilModule &DM) {
  hlsl::OP *hlslOP = DM.GetOP();
  Value *TrueVal = hlslOP->GetI1Const(true);
  for (auto it : hlslOP->GetOpFuncList(DXIL::OpCode::CreateHandleFromHeap)) {
    Function *F = it.second;
    if (!F)
      continue;
    for (User *U : F->users()) {
      CallInst *CI = cast<CallInst>(U);
      if (!DxilMDHelper::IsMarkedNonUniform(CI))
        continue;
      // Set NonUniform to be true.
      CI->setOperand(DxilInst_CreateHandleFromHeap::arg_nonUniformIndex,
                     TrueVal);
      // Clear nonUniform metadata.
      CI->setMetadata(DxilMDHelper::kDxilNonUniformAttributeMDName, nullptr);
    }
  }
}

// Remove createHandleFromHandle when not a lib
void DxilLowerCreateHandleForLib::RemoveCreateHandleFromHandle(DxilModule &DM) {
  hlsl::OP *hlslOP = DM.GetOP();
  Type *HdlTy = hlslOP->GetHandleType();
  for (auto it : hlslOP->GetOpFuncList(DXIL::OpCode::CreateHandleForLib)) {
    Function *F = it.second;
    if (!F)
      continue;
    if (it.first != HdlTy)
      continue;
    for (auto it = F->users().begin(); it != F->users().end();) {
      User *U = *(it++);
      CallInst *CI = cast<CallInst>(U);
      DxilInst_CreateHandleForLib Hdl(CI);
      Value *Res = Hdl.get_Resource();
      CI->replaceAllUsesWith(Res);
      CI->eraseFromParent();
    }
    break;
  }
}

char DxilLowerCreateHandleForLib::ID = 0;

ModulePass *llvm::createDxilLowerCreateHandleForLibPass() {
  return new DxilLowerCreateHandleForLib();
}

INITIALIZE_PASS_BEGIN(DxilLowerCreateHandleForLib, "hlsl-dxil-lower-handle-for-lib", "DXIL Lower createHandleForLib", false, false)
INITIALIZE_PASS_DEPENDENCY(DxilValueCache)
INITIALIZE_PASS_END(DxilLowerCreateHandleForLib, "hlsl-dxil-lower-handle-for-lib", "DXIL Lower createHandleForLib", false, false)


class DxilAllocateResourcesForLib : public ModulePass {
public:
  static char ID; // Pass identification, replacement for typeid
  explicit DxilAllocateResourcesForLib() : ModulePass(ID), m_AutoBindingSpace(UINT_MAX) {}

  void applyOptions(PassOptions O) override {
    GetPassOptionUInt32(O, "auto-binding-space", &m_AutoBindingSpace, UINT_MAX);
  }
  StringRef getPassName() const override { return "DXIL Allocate Resources For Library"; }

  bool runOnModule(Module &M) override {
    DxilModule &DM = M.GetOrCreateDxilModule();
    // Must specify a default space, and must apply to library.
    // Use DxilCondenseResources instead for shaders.
    if ((m_AutoBindingSpace == UINT_MAX) || !DM.GetShaderModel()->IsLib())
      return false;

    bool hasResource = DM.GetCBuffers().size() ||
      DM.GetUAVs().size() || DM.GetSRVs().size() || DM.GetSamplers().size();

    if (hasResource) {
      DM.SetAutoBindingSpace(m_AutoBindingSpace);

      DxilResourceRegisterAllocator ResourceRegisterAllocator;
      ResourceRegisterAllocator.AllocateRegisters(DM);
    }
    return true;
  }

private:
  uint32_t m_AutoBindingSpace;
};

char DxilAllocateResourcesForLib::ID = 0;

ModulePass *llvm::createDxilAllocateResourcesForLibPass() {
  return new DxilAllocateResourcesForLib();
}

INITIALIZE_PASS(DxilAllocateResourcesForLib, "hlsl-dxil-allocate-resources-for-lib", "DXIL Allocate Resources For Library", false, false)


namespace {
struct CreateHandleFromHeapArgs {
  Value *Index;
  bool isSampler;
  bool isNonUniform;
  // All incoming handle args are confirmed.
  // If not resolved, some of the incoming handle is not from createHandleFromHeap.
  // Might be resolved after linking for lib.
  bool isResolved;
  void merge(CreateHandleFromHeapArgs &args, ResourceUseErrors &Errors,
             Value *mergeHdl) {
    if (args.isSampler != isSampler) {
      // Report error.
      Errors.ReportError(ResourceUseErrors::ErrorCode::MismatchIsSampler,
                         mergeHdl);
    }
    args.isNonUniform |= isNonUniform;
  }
};

} // namespace

// Helper class for legalizing dynamic resource use
// Convert select/phi on resources to select/phi on index.
// TODO: support case when save dynamic resource as local array element.
// TODO: share code with LegalizeResourceUseHelper.
class LegalizeDynamicResourceUseHelper {

public:
  ResourceUseErrors m_Errors;
  DenseMap<Value *, CreateHandleFromHeapArgs> HandleToArgs;
  // Value sets we can use to iterate
  ValueSetVector HandleSelects;
  ResourceUseErrors Errors;
  std::unordered_set<Instruction *> CleanupInsts;

  void mergeHeapArgs(Value *SelHdl, Value *SelIdx, User::op_range Hdls) {
    CreateHandleFromHeapArgs args = {nullptr, false, false, true};

    for (Value *V : Hdls) {
      auto it = HandleToArgs.find(V);
      // keep invalid when V is not createHandleFromHeap.
      if (it == HandleToArgs.end()) {
        args.isResolved = false;
        continue;
      }
      CreateHandleFromHeapArgs &itArgs = it->second;
      if (!itArgs.isResolved) {
        args.isResolved = false;
        continue;
      }

      if (args.Index != nullptr) {
        args.merge(itArgs, Errors, SelHdl);
      } else {
        args.Index = SelIdx;
        args.isNonUniform = itArgs.isNonUniform;
        args.isSampler = itArgs.isSampler;
      }
    }
    // set Index when all incoming Hdls cannot be resolved.
    if (args.Index == nullptr)
      args.Index = SelIdx;
    HandleToArgs[SelHdl] = args;
  }

  void CreateSelectsForHandleSelects() {
    if (HandleSelects.empty())
      return;

    LLVMContext &Ctx = HandleSelects[0]->getContext();
    Type *i32Ty = Type::getInt32Ty(Ctx);
    Value *UndefValue = UndefValue::get(i32Ty);
    // Create select for each HandleSelects.
    for (auto &Select : HandleSelects) {
      if (PHINode *Phi = dyn_cast<PHINode>(Select)) {
        IRBuilder<> B(Phi);
        unsigned numIncoming = Phi->getNumIncomingValues();
        PHINode *newPhi = B.CreatePHI(i32Ty, numIncoming);
        for (unsigned j = 0; j < numIncoming; j++) {
          // Set incoming values to undef until next pass
          newPhi->addIncoming(UndefValue, Phi->getIncomingBlock(j));
        }
        mergeHeapArgs(Phi, newPhi, Phi->incoming_values());
      } else if (SelectInst *Sel = dyn_cast<SelectInst>(Select)) {
        IRBuilder<> B(Sel);
        Value *newSel =
            B.CreateSelect(Sel->getCondition(), UndefValue, UndefValue);
        User::op_range range = User::op_range(Sel->getOperandList() + 1,
                                       Sel->getOperandList() + 3);
        mergeHeapArgs(
            Sel, newSel,
            range);
      } else {
        DXASSERT(false, "otherwise, non-select/phi in Selects set");
      }
    }
  }
  // propagate CreateHandleFromHeapArgs for HandleSel which all operands are
  // other HandleSel.
  void PropagateHeapArgs() {
    SmallVector<Value *, 4> Candidates;
    for (auto &Select : HandleSelects) {
      CreateHandleFromHeapArgs &args = HandleToArgs[Select];
      if (args.isResolved)
        continue;
      Candidates.emplace_back(Select);
    }

    while (1) {
      SmallVector<Value *, 4> NextPass;
      for (auto &Select : Candidates) {
        CreateHandleFromHeapArgs &args = HandleToArgs[Select];
        if (PHINode *Phi = dyn_cast<PHINode>(Select)) {
          mergeHeapArgs(Phi, args.Index, Phi->incoming_values());
        } else if (SelectInst *Sel = dyn_cast<SelectInst>(Select)) {
          User::op_range range = User::op_range(Sel->getOperandList() + 1,
                                                Sel->getOperandList() + 3);
          mergeHeapArgs(
              Sel, args.Index,
              range);
        } else {
          DXASSERT(false, "otherwise, non-select/phi in Selects set");
        }

        if (args.isResolved)
          continue;
        NextPass.emplace_back(Select);
      }
      // Some node cannot be reached.
      if (NextPass.size() == Candidates.size())
        return;
      Candidates = NextPass;
    }
  }

  void UpdateSelectsForHandleSelect(hlsl::OP *hlslOP) {
    if (HandleSelects.empty())
      return;
    LLVMContext &Ctx = HandleSelects[0]->getContext();
    Type *pVoidTy = Type::getVoidTy(Ctx);
    // NOTE: phi of createHandleFromHeap and createHandleFromBinding
    // is not supported.
    Function *createHdlFromHeap =
        hlslOP->GetOpFunc(DXIL::OpCode::CreateHandleFromHeap, pVoidTy);
    Value *hdlFromHeapOP = hlslOP->GetI32Const(
        static_cast<unsigned>(DXIL::OpCode::CreateHandleFromHeap));
    for (auto &Select : HandleSelects) {
      if (PHINode *Phi = dyn_cast<PHINode>(Select)) {
        unsigned numIncoming = Phi->getNumIncomingValues();
        CreateHandleFromHeapArgs &args = HandleToArgs[Phi];
        PHINode *newPhi = cast<PHINode>(args.Index);
        if (args.isResolved) {
          for (unsigned j = 0; j < numIncoming; j++) {
            Value *V = Phi->getIncomingValue(j);
            auto it = HandleToArgs.find(V);
            DXASSERT(it != HandleToArgs.end(),
                     "args.isResolved should be false");
            CreateHandleFromHeapArgs &itArgs = it->second;
            newPhi->setIncomingValue(j, itArgs.Index);
          }

          IRBuilder<> B(Phi->getParent()->getFirstNonPHI());
          B.SetCurrentDebugLocation(Phi->getDebugLoc());
          Value *isSampler = hlslOP->GetI1Const(args.isSampler);
          // TODO: or args.IsNonUniform with !isUniform(Phi) with uniform
          // analysis.
          Value *isNonUniform = hlslOP->GetI1Const(args.isNonUniform);
          CallInst *newCI =
              B.CreateCall(createHdlFromHeap,
                           {hdlFromHeapOP, newPhi, isSampler, isNonUniform});
          Phi->replaceAllUsesWith(newCI);
          CleanupInsts.insert(Phi);
          // put newCI in HandleToArgs.
          HandleToArgs[newCI] = args;
        } else {
          newPhi->eraseFromParent();
        }
      } else if (SelectInst *Sel = dyn_cast<SelectInst>(Select)) {
        CreateHandleFromHeapArgs &args = HandleToArgs[Sel];
        SelectInst *newSel = cast<SelectInst>(args.Index);
        if (args.isResolved) {
          for (unsigned j = 1; j < 3; ++j) {
            Value *V = Sel->getOperand(j);
            auto it = HandleToArgs.find(V);
            DXASSERT(it != HandleToArgs.end(),
                     "args.isResolved should be false");
            CreateHandleFromHeapArgs &itArgs = it->second;
            newSel->setOperand(j, itArgs.Index);
          }

          IRBuilder<> B(newSel->getNextNode());
          B.SetCurrentDebugLocation(newSel->getDebugLoc());
          Value *isSampler = hlslOP->GetI1Const(args.isSampler);
          // TODO: or args.IsNonUniform with !isUniform(Phi).
          Value *isNonUniform = hlslOP->GetI1Const(args.isNonUniform);
          CallInst *newCI =
              B.CreateCall(createHdlFromHeap,
                           {hdlFromHeapOP, newSel, isSampler, isNonUniform});
          Sel->replaceAllUsesWith(newCI);
          CleanupInsts.insert(Sel);
          // put newCI in HandleToArgs.
          HandleToArgs[newCI] = args;
        } else {
          newSel->eraseFromParent();
        }
      } else {
        DXASSERT(false, "otherwise, non-select/phi in HandleSelects set");
      }
    }
  }

  void CollectResources(DxilModule &DM) {
    ValueSetVector tmpHandleSelects;
    hlsl::OP *hlslOP = DM.GetOP();
    if (hlslOP->IsDxilOpUsed(DXIL::OpCode::CreateHandleFromHeap)) {
      Function *F = hlslOP->GetOpFunc(DXIL::OpCode::CreateHandleFromHeap,
                                      Type::getVoidTy(DM.GetCtx()));
      for (User *U : F->users()) {
        DxilInst_CreateHandleFromHeap Hdl(cast<CallInst>(U));
        HandleToArgs[U] = {Hdl.get_index(), Hdl.get_samplerHeap_val(),
                           Hdl.get_nonUniformIndex_val(), true};
        for (User *HdlU : U->users()) {
          if (isa<PHINode>(HdlU) || isa<SelectInst>(HdlU)) {
            tmpHandleSelects.insert(HdlU);
          }
        }
      }
    }
    // Collect phi/sel of other phi/sel selected handles.
    while (!tmpHandleSelects.empty()) {
      HandleSelects.insert(tmpHandleSelects.begin(), tmpHandleSelects.end());
      ValueSetVector newHandleSelects;
      for (Value *Hdl : tmpHandleSelects) {
        for (User *HdlU : Hdl->users()) {
          if (HandleSelects.count(HdlU))
            continue;
          if (isa<PHINode>(HdlU) || isa<SelectInst>(HdlU)) {
            newHandleSelects.insert(HdlU);
          }
        }
      }
      tmpHandleSelects = newHandleSelects;
    }
  }

  void DoTransform(hlsl::OP *hlslOP) {
    CreateSelectsForHandleSelects();
    PropagateHeapArgs();
    UpdateSelectsForHandleSelect(hlslOP);
    CleanupUnusedValues(CleanupInsts);
  }

  bool runOnModule(llvm::Module &M) {
    DxilModule &DM = M.GetOrCreateDxilModule();
    hlsl::OP *hlslOP = DM.GetOP();

    CollectResources(DM);

    // If no selects or allocas are involved, there isn't anything to do
    if (HandleSelects.empty())
      return false;

    DoTransform(hlslOP);

    return true;
  }
};

namespace {
// Make sure no phi/sel on annotateHandle.
bool sinkAnnotateHandleAfterSelect(DxilModule &DM, Module &M) {
  // Collect AnnotateHandle calls.
  SmallVector<CallInst *, 4> annotHdls;
  hlsl::OP *op = DM.GetOP();
  LLVMContext &Ctx = M.getContext();
  Type *pVoidTy = Type::getVoidTy(Ctx);
  Function *annotHdlFn = op->GetOpFunc(DXIL::OpCode::AnnotateHandle, pVoidTy);
  for (auto it : op->GetOpFuncList(OP::OpCode::AnnotateHandle)) {
    Function *F = it.second;
    if (F == nullptr)
      continue;
    for (auto U = F->user_begin(); U != F->user_end();) {
      CallInst *CI = dyn_cast<CallInst>(*(U++));
      annotHdls.emplace_back(CI);
    }
  }
  if (annotHdls.empty())
    return false;

  SetVector<Instruction *> selectAnnotHdls;
  for (CallInst *CI : annotHdls) {
    for (User *U : CI->users()) {
      if (isa<PHINode>(U) || isa<SelectInst>(U))
        selectAnnotHdls.insert(cast<Instruction>(U));
    }
  }
  const ShaderModel *pSM = DM.GetShaderModel();
  Type *propsTy = op->GetResourcePropertiesType();
  Value *OpArg =
      op->GetI32Const(static_cast<unsigned>(DXIL::OpCode::AnnotateHandle));
  ResourceUseErrors Errors;
  Value *undefHdl = UndefValue::get(op->GetHandleType());
  // Sink annotateHandle after phi.
  for (Instruction *Hdl : selectAnnotHdls) {
    if (PHINode *phi = dyn_cast<PHINode>(Hdl)) {
      Value *props = nullptr;
      for (unsigned i = 0; i < phi->getNumIncomingValues(); ++i) {
        Value *V = phi->getIncomingValue(i);
        if (CallInst *CI = dyn_cast<CallInst>(V)) {
          DxilInst_AnnotateHandle annot(CI);
          if (annot) {
            if (props == nullptr) {
              props = annot.get_props();
            }
            else if (props != annot.get_props()) {
              props = resource_helper::tryMergeProps(
                  cast<Constant>(props), cast<Constant>(annot.get_props()),
                  propsTy, *pSM);
              if (props == nullptr) {
                Errors.ReportError(
                    ResourceUseErrors::ErrorCode::MismatchHandleAnnotation,
                    phi);
                props = annot.get_props();
              }
            }

            Value *res = annot.get_res();
            phi->setIncomingValue(i, res);
          }
        }
      }
      // Insert after phi.
      IRBuilder<> B(phi->getParent()->getFirstNonPHI());
      CallInst *annotCI = B.CreateCall(annotHdlFn, {OpArg, undefHdl, props});
      phi->replaceAllUsesWith(annotCI);
      annotCI->setArgOperand(DxilInst_AnnotateHandle::arg_res, phi);
    } else {
      SelectInst *sel = dyn_cast<SelectInst>(Hdl);
      Value *TVal = sel->getTrueValue();
      Value *FVal = sel->getFalseValue();
      Value *props = nullptr;
      if (CallInst *CI = dyn_cast<CallInst>(TVal)) {
        DxilInst_AnnotateHandle annot(CI);
        if (annot) {
          props = annot.get_props();
          Value *res = annot.get_res();
          sel->setOperand(1, res);
        }
      }
      if (CallInst *CI = dyn_cast<CallInst>(FVal)) {
        DxilInst_AnnotateHandle annot(CI);
        if (annot) {
          if (props == nullptr) {
            props = annot.get_props();
          } else if (props != annot.get_props()) {
            props = resource_helper::tryMergeProps(
                cast<Constant>(props), cast<Constant>(annot.get_props()),
                propsTy, *pSM);
            if (props == nullptr) {
              Errors.ReportError(
                  ResourceUseErrors::ErrorCode::MismatchHandleAnnotation, sel);
              props = annot.get_props();
            }
          }

          Value *res = annot.get_res();
          sel->setOperand(2, res);
        }
      }

      // Insert after sel.
      IRBuilder<> B(sel->getNextNode());
      CallInst *annotCI = B.CreateCall(annotHdlFn, {OpArg, undefHdl, props});
      sel->replaceAllUsesWith(annotCI);
      annotCI->setArgOperand(DxilInst_AnnotateHandle::arg_res, sel);
    }
  }
  return true;
}
} // namespace

// Remove redudant annotateHandle.
// Legalize phi on createHandleFromHeap.
class DxilCleanupDynamicResourceHandle : public ModulePass {
public:
  static char ID; // Pass identification, replacement for typeid
  explicit DxilCleanupDynamicResourceHandle() : ModulePass(ID) {}

  StringRef getPassName() const override { return "DXIL Cleanup dynamic resource handle calls"; }

  bool runOnModule(Module &M) override {
    DxilModule &DM = M.GetOrCreateDxilModule();

    // Nothing to do if Dxil ver < 1.6
    unsigned dxilMajor, dxilMinor;
    DM.GetShaderModel()->GetDxilVersion(dxilMajor, dxilMinor);
    if (DXIL::CompareVersions(dxilMajor, dxilMinor, 1, 6) < 0)
      return false;

    bool bChanged = sinkAnnotateHandleAfterSelect(DM, M);

    // Legalize phi on createHandleFromHeap.
    LegalizeDynamicResourceUseHelper helper;
    bChanged |= helper.runOnModule(M);

    hlsl::OP *op = DM.GetOP();
    const ShaderModel *pSM = DM.GetShaderModel();
    Type *propsTy = op->GetResourcePropertiesType();
    // Iterate AnnotateHandle calls and eliminate redundant annotate handle call
    // chains.
    for (auto it : op->GetOpFuncList(OP::OpCode::AnnotateHandle)) {
      Function *F = it.second;
      if (F == nullptr)
        continue;
      for (auto U = F->user_begin(); U != F->user_end();) {
        CallInst *CI = dyn_cast<CallInst>(*(U++));
        if (CI) {
          DxilInst_AnnotateHandle AH(CI);
          if (AH) {
            Value *Res = AH.get_res();
            // Skip handle from load global res.
            if (isa<LoadInst>(Res))
              continue;
            CallInst *CRes = dyn_cast<CallInst>(Res);
            if (!CRes)
              continue;
            DxilInst_AnnotateHandle PrevAH(CRes);
            if (PrevAH) {
              Value *mergedProps = resource_helper::tryMergeProps(
                  cast<Constant>(AH.get_props()), cast<Constant>(PrevAH.get_props()), propsTy, *pSM);
              if (mergedProps == nullptr) {
                ResourceUseErrors Errors;
                Errors.ReportError(
                    ResourceUseErrors::ErrorCode::MismatchHandleAnnotation, CI);
              } else if (mergedProps != PrevAH.get_props()) {
                PrevAH.set_props(mergedProps);
              }
              CI->replaceAllUsesWith(Res);
              CI->eraseFromParent();
              bChanged = true;
            }
          }
        }
      }
    }

    return bChanged;
  }

private:
};

char DxilCleanupDynamicResourceHandle::ID = 0;

ModulePass *llvm::createDxilCleanupDynamicResourceHandlePass() {
  return new DxilCleanupDynamicResourceHandle();
}

INITIALIZE_PASS(DxilCleanupDynamicResourceHandle,
                "hlsl-dxil-cleanup-dynamic-resource-handle",
                "DXIL Cleanup dynamic resource handle calls", false, false)
