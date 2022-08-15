///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilLinker.cpp                                                           //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/HLSL/DxilLinker.h"
#include "dxc/DXIL/DxilCBuffer.h"
#include "dxc/DXIL/DxilFunctionProps.h"
#include "dxc/DXIL/DxilEntryProps.h"
#include "dxc/DXIL/DxilModule.h"
#include "dxc/DXIL/DxilOperations.h"
#include "dxc/DXIL/DxilResource.h"
#include "dxc/DXIL/DxilSampler.h"
#include "dxc/DXIL/DxilUtil.h"
#include "dxc/Support/Global.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <vector>

#include "dxc/DxilContainer/DxilContainer.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/DebugInfo.h"

#include "dxc/HLSL/DxilGenerationPass.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar.h"

#include "dxc/HLSL/DxilExportMap.h"
#include "dxc/HLSL/ComputeViewIdState.h"

using namespace llvm;
using namespace hlsl;

namespace {

void CollectUsedFunctions(Constant *C,
                          llvm::SetVector<Function *> &funcSet) {
  for (User *U : C->users()) {
    if (Instruction *I = dyn_cast<Instruction>(U)) {
      funcSet.insert(I->getParent()->getParent());
    } else {
      Constant *CU = cast<Constant>(U);
      CollectUsedFunctions(CU, funcSet);
    }
  }
}

template <class T>
void AddResourceMap(
    const std::vector<std::unique_ptr<T>> &resTab, DXIL::ResourceClass resClass,
    llvm::MapVector<const llvm::Constant *, DxilResourceBase *> &resMap,
    DxilModule &DM) {
  for (auto &Res : resTab) {
    resMap[Res->GetGlobalSymbol()] = Res.get();
  }
}

void CloneFunction(Function *F, Function *NewF, ValueToValueMapTy &vmap,
                   hlsl::DxilTypeSystem *TypeSys = nullptr,
                   hlsl::DxilTypeSystem *SrcTypeSys = nullptr) {
  SmallVector<ReturnInst *, 2> Returns;
  // Map params.
  auto paramIt = NewF->arg_begin();
  for (Argument &param : F->args()) {
    vmap[&param] = (paramIt++);
  }

  llvm::CloneFunctionInto(NewF, F, vmap, /*ModuleLevelChanges*/ true, Returns);
  if (TypeSys) {
    if (SrcTypeSys == nullptr)
      SrcTypeSys = TypeSys;
    TypeSys->CopyFunctionAnnotation(NewF, F, *SrcTypeSys);
  }

  // Remove params from vmap.
  for (Argument &param : F->args()) {
    vmap.erase(&param);
  }
}

} // namespace

namespace {

struct DxilFunctionLinkInfo {
  DxilFunctionLinkInfo(llvm::Function *F);
  llvm::Function *func;
  // SetVectors for deterministic iteration
  llvm::SetVector<llvm::Function *> usedFunctions;
  llvm::SetVector<llvm::GlobalVariable *> usedGVs;
};

// Library to link.
class DxilLib {

public:
  DxilLib(std::unique_ptr<llvm::Module> pModule);
  virtual ~DxilLib() {}
  bool HasFunction(std::string &name);
  llvm::StringMap<std::unique_ptr<DxilFunctionLinkInfo>> &GetFunctionTable() {
    return m_functionNameMap;
  }
  bool IsInitFunc(llvm::Function *F);
  bool IsEntry(llvm::Function *F);
  bool IsResourceGlobal(const llvm::Constant *GV);
  DxilResourceBase *GetResource(const llvm::Constant *GV);

  DxilModule &GetDxilModule() { return m_DM; }
  void LazyLoadFunction(Function *F);
  void BuildGlobalUsage();
  void CollectUsedInitFunctions(SetVector<StringRef> &addedFunctionSet,
                                SmallVector<StringRef, 4> &workList);

  void FixIntrinsicOverloads();

private:
  std::unique_ptr<llvm::Module> m_pModule;
  DxilModule &m_DM;
  // Map from name to Link info for extern functions.
  llvm::StringMap<std::unique_ptr<DxilFunctionLinkInfo>> m_functionNameMap;
  llvm::SmallPtrSet<llvm::Function*,4>  m_entrySet;
  // Map from resource link global to resource. MapVector for deterministic iteration.
  llvm::MapVector<const llvm::Constant *, DxilResourceBase *> m_resourceMap;
  // Set of initialize functions for global variable. SetVector for deterministic iteration.
  llvm::SetVector<llvm::Function *> m_initFuncSet;
};

struct DxilLinkJob;

class DxilLinkerImpl : public hlsl::DxilLinker {
public:
  DxilLinkerImpl(LLVMContext &Ctx, unsigned valMajor, unsigned valMinor) : DxilLinker(Ctx, valMajor, valMinor) {}
  virtual ~DxilLinkerImpl() {}
  bool HasLibNameRegistered(StringRef name) override;
  bool RegisterLib(StringRef name, std::unique_ptr<llvm::Module> pModule,
                   std::unique_ptr<llvm::Module> pDebugModule) override;
  bool AttachLib(StringRef name) override;
  bool DetachLib(StringRef name) override;
  void DetachAll() override;

  std::unique_ptr<llvm::Module>
  Link(StringRef entry, StringRef profile, dxilutil::ExportMap &exportMap) override;

private:
  bool AttachLib(DxilLib *lib);
  bool DetachLib(DxilLib *lib);
  bool AddFunctions(SmallVector<StringRef, 4> &workList,
                    SetVector<DxilLib *> &libSet, SetVector<StringRef> &addedFunctionSet,
                    DxilLinkJob &linkJob, bool bLazyLoadDone,
                    bool bAllowFuncionDecls);
  // Attached libs to link.
  std::unordered_set<DxilLib *> m_attachedLibs;
  // Owner of all DxilLib.
  StringMap<std::unique_ptr<DxilLib>> m_LibMap;
  llvm::StringMap<std::pair<DxilFunctionLinkInfo *, DxilLib *>>
      m_functionNameMap;
};

} // namespace

//------------------------------------------------------------------------------
//
// DxilFunctionLinkInfo methods.
//
DxilFunctionLinkInfo::DxilFunctionLinkInfo(Function *F) : func(F) {
  DXASSERT_NOMSG(F);
}

//------------------------------------------------------------------------------
//
// DxilLib methods.
//

DxilLib::DxilLib(std::unique_ptr<llvm::Module> pModule)
    : m_pModule(std::move(pModule)), m_DM(m_pModule->GetOrCreateDxilModule()) {
  Module &M = *m_pModule;
  const std::string MID = (Twine(M.getModuleIdentifier()) + ".").str();

  // Collect function defines.
  for (Function &F : M.functions()) {
    if (F.isDeclaration())
      continue;
    if (F.getLinkage() == GlobalValue::LinkageTypes::InternalLinkage) {
      // Add prefix to internal function.
      F.setName(MID + F.getName());
    }
    m_functionNameMap[F.getName()] =
        llvm::make_unique<DxilFunctionLinkInfo>(&F);
    if (m_DM.IsEntry(&F))
      m_entrySet.insert(&F);
  }

  // Update internal global name.
  for (GlobalVariable &GV : M.globals()) {
    if (GV.getLinkage() == GlobalValue::LinkageTypes::InternalLinkage) {
      // Add prefix to internal global.
      GV.setName(MID + GV.getName());
    }
  }

}

void DxilLib::FixIntrinsicOverloads() {
  // Fix DXIL overload name collisions that may be caused by name
  // collisions between dxil ops with different overload types,
  // when those types may have had the same name in the original
  // modules.
  m_DM.GetOP()->FixOverloadNames();
}

void DxilLib::LazyLoadFunction(Function *F) {
  DXASSERT(m_functionNameMap.count(F->getName()), "else invalid Function");
  DxilFunctionLinkInfo *linkInfo = m_functionNameMap[F->getName()].get();
  std::error_code EC = F->materialize();
  DXASSERT_LOCALVAR(EC, !EC, "else fail to materialize");

  // Build used functions for F.
  for (auto &BB : F->getBasicBlockList()) {
    for (auto &I : BB.getInstList()) {
      if (CallInst *CI = dyn_cast<CallInst>(&I)) {
        linkInfo->usedFunctions.insert(CI->getCalledFunction());
      }
    }
  }

  if (m_DM.HasDxilFunctionProps(F)) {
    DxilFunctionProps &props = m_DM.GetDxilFunctionProps(F);
    if (props.IsHS()) {
      // Add patch constant function to usedFunctions of entry.
      Function *patchConstantFunc = props.ShaderProps.HS.patchConstantFunc;
      linkInfo->usedFunctions.insert(patchConstantFunc);
    }
  }
  // Used globals will be build before link.
}

void DxilLib::BuildGlobalUsage() {
  Module &M = *m_pModule;

  // Collect init functions for static globals.
  if (GlobalVariable *Ctors = M.getGlobalVariable("llvm.global_ctors")) {
    if (ConstantArray *CA = dyn_cast<ConstantArray>(Ctors->getInitializer())) {
      for (User::op_iterator i = CA->op_begin(), e = CA->op_end(); i != e;
           ++i) {
        if (isa<ConstantAggregateZero>(*i))
          continue;
        ConstantStruct *CS = cast<ConstantStruct>(*i);
        if (isa<ConstantPointerNull>(CS->getOperand(1)))
          continue;

        // Must have a function or null ptr.
        if (!isa<Function>(CS->getOperand(1)))
          continue;
        Function *Ctor = cast<Function>(CS->getOperand(1));
        assert(Ctor->getReturnType()->isVoidTy() && Ctor->arg_size() == 0 &&
               "function type must be void (void)");
        // Add Ctor.
        m_initFuncSet.insert(Ctor);
        LazyLoadFunction(Ctor);
      }
    }
  }

  // Build used globals.
  for (GlobalVariable &GV : M.globals()) {
    llvm::SetVector<Function *> funcSet;
    CollectUsedFunctions(&GV, funcSet);
    for (Function *F : funcSet) {
      DXASSERT(m_functionNameMap.count(F->getName()), "must exist in table");
      DxilFunctionLinkInfo *linkInfo = m_functionNameMap[F->getName()].get();
      linkInfo->usedGVs.insert(&GV);
    }
  }

  // Build resource map.
  AddResourceMap(m_DM.GetUAVs(), DXIL::ResourceClass::UAV, m_resourceMap, m_DM);
  AddResourceMap(m_DM.GetSRVs(), DXIL::ResourceClass::SRV, m_resourceMap, m_DM);
  AddResourceMap(m_DM.GetCBuffers(), DXIL::ResourceClass::CBuffer,
                 m_resourceMap, m_DM);
  AddResourceMap(m_DM.GetSamplers(), DXIL::ResourceClass::Sampler,
                 m_resourceMap, m_DM);
}

void DxilLib::CollectUsedInitFunctions(SetVector<StringRef> &addedFunctionSet,
                                       SmallVector<StringRef, 4> &workList) {
  // Add init functions to used functions.
  for (Function *Ctor : m_initFuncSet) {
    DXASSERT(m_functionNameMap.count(Ctor->getName()),
             "must exist in internal table");
    DxilFunctionLinkInfo *linkInfo = m_functionNameMap[Ctor->getName()].get();
    // If function other than Ctor used GV of Ctor.
    // Add Ctor to usedFunctions for it.
    for (GlobalVariable *GV : linkInfo->usedGVs) {
      llvm::SetVector<Function *> funcSet;
      CollectUsedFunctions(GV, funcSet);
      bool bAdded = false;
      for (Function *F : funcSet) {
        if (F == Ctor)
          continue;
        // If F is added for link, add init func to workList.
        if (addedFunctionSet.count(F->getName())) {
          workList.emplace_back(Ctor->getName());
          bAdded = true;
          break;
        }
      }
      if (bAdded)
        break;
    }
  }
}

bool DxilLib::HasFunction(std::string &name) {
  return m_functionNameMap.count(name);
}

bool DxilLib::IsEntry(llvm::Function *F) { return m_entrySet.count(F); }
bool DxilLib::IsInitFunc(llvm::Function *F) { return m_initFuncSet.count(F); }
bool DxilLib::IsResourceGlobal(const llvm::Constant *GV) {
  return m_resourceMap.count(GV);
}
DxilResourceBase *DxilLib::GetResource(const llvm::Constant *GV) {
  if (IsResourceGlobal(GV))
    return m_resourceMap[GV];
  else
    return nullptr;
}


namespace {
// Create module from link defines.
struct DxilLinkJob {
  DxilLinkJob(LLVMContext &Ctx, dxilutil::ExportMap &exportMap,
              unsigned valMajor, unsigned valMinor)
      : m_ctx(Ctx), m_exportMap(exportMap), m_valMajor(valMajor),
        m_valMinor(valMinor) {}
  std::unique_ptr<llvm::Module>
  Link(std::pair<DxilFunctionLinkInfo *, DxilLib *> &entryLinkPair,
       const ShaderModel *pSM);
  std::unique_ptr<llvm::Module> LinkToLib(const ShaderModel *pSM);
  void StripDeadDebugInfo(llvm::Module &M);
  // Fix issues when link to different shader model.
  void FixShaderModelMismatch(llvm::Module &M);
  void RunPreparePass(llvm::Module &M);
  void AddFunction(std::pair<DxilFunctionLinkInfo *, DxilLib *> &linkPair);
  void AddFunction(llvm::Function *F);

private:
  void LinkNamedMDNodes(Module *pM, ValueToValueMapTy &vmap);
  void AddFunctionDecls(Module *pM);
  bool AddGlobals(DxilModule &DM, ValueToValueMapTy &vmap);
  void EmitCtorListForLib(Module *pM);
  void CloneFunctions(ValueToValueMapTy &vmap);
  void AddFunctions(DxilModule &DM, ValueToValueMapTy &vmap);
  bool AddResource(DxilResourceBase *res, llvm::GlobalVariable *GV);
  void AddResourceToDM(DxilModule &DM);
  llvm::MapVector<DxilFunctionLinkInfo *, DxilLib *> m_functionDefs;

  // Function decls, in order added.
  llvm::MapVector<llvm::StringRef,
                  std::pair<llvm::SmallPtrSet<llvm::FunctionType *, 2>,
                            llvm::SmallVector<llvm::Function *, 2>>>
    m_functionDecls;

  // New created functions, in order added.
  llvm::MapVector<llvm::StringRef, llvm::Function *> m_newFunctions;

  // New created globals, in order added.
  llvm::MapVector<llvm::StringRef, llvm::GlobalVariable *> m_newGlobals;

  // Map for resource, ordered by name.
  std::map<llvm::StringRef,
           std::pair<DxilResourceBase *, llvm::GlobalVariable *>>
    m_resourceMap;

  LLVMContext &m_ctx;
  dxilutil::ExportMap &m_exportMap;
  unsigned m_valMajor, m_valMinor;
};
} // namespace

namespace {
const char kUndefFunction[] = "Cannot find definition of function ";
const char kRedefineFunction[] = "Definition already exists for function ";
const char kRedefineGlobal[] = "Definition already exists for global variable ";
const char kInvalidProfile[] = " is invalid profile to link";
const char kExportOnlyForLib[] = "export map is only for library";
const char kShaderKindMismatch[] =
    "Profile mismatch between entry function and target profile:";
const char kNoEntryProps[] =
    "Cannot find function property for entry function ";
const char kRedefineResource[] =
    "Resource already exists as ";
const char kInvalidValidatorVersion[] = "Validator version does not support target profile ";
const char kExportNameCollision[] = "Export name collides with another export: ";
const char kExportFunctionMissing[] = "Could not find target for export: ";
const char kNoFunctionsToExport[] = "Library has no functions to export";
} // namespace
//------------------------------------------------------------------------------
//
// DxilLinkJob methods.
//

namespace {
// Helper function to check type match.
bool IsMatchedType(Type *Ty0, Type *Ty);

StringRef RemoveNameSuffix(StringRef Name) {
  size_t DotPos = Name.rfind('.');
  if (DotPos != StringRef::npos && Name.back() != '.' &&
      isdigit(static_cast<unsigned char>(Name[DotPos + 1])))
    Name = Name.substr(0, DotPos);
  return Name;
}

bool IsMatchedStructType(StructType *ST0, StructType *ST) {
  StringRef Name0 = RemoveNameSuffix(ST0->getName());
  StringRef Name = RemoveNameSuffix(ST->getName());

  if (Name0 != Name)
    return false;

  if (ST0->getNumElements() != ST->getNumElements())
    return false;

  if (ST0->isLayoutIdentical(ST))
    return true;

  for (unsigned i = 0; i < ST->getNumElements(); i++) {
    Type *Ty = ST->getElementType(i);
    Type *Ty0 = ST0->getElementType(i);
    if (!IsMatchedType(Ty, Ty0))
      return false;
  }
  return true;
}

bool IsMatchedArrayType(ArrayType *AT0, ArrayType *AT) {
  if (AT0->getNumElements() != AT->getNumElements())
    return false;
  return IsMatchedType(AT0->getElementType(), AT->getElementType());
}

bool IsMatchedType(Type *Ty0, Type *Ty) {
  if (Ty0->isStructTy() && Ty->isStructTy()) {
    StructType *ST0 = cast<StructType>(Ty0);
    StructType *ST = cast<StructType>(Ty);
    return IsMatchedStructType(ST0, ST);
  }

  if (Ty0->isArrayTy() && Ty->isArrayTy()) {
    ArrayType *AT0 = cast<ArrayType>(Ty0);
    ArrayType *AT = cast<ArrayType>(Ty);
    return IsMatchedArrayType(AT0, AT);
  }

  if (Ty0->isPointerTy() && Ty->isPointerTy()) {
    if (Ty0->getPointerAddressSpace() != Ty->getPointerAddressSpace())
      return false;

    return IsMatchedType(Ty0->getPointerElementType(),
                         Ty->getPointerElementType());
  }

  return Ty0 == Ty;
}
} // namespace

bool DxilLinkJob::AddResource(DxilResourceBase *res, llvm::GlobalVariable *GV) {
  if (m_resourceMap.count(res->GetGlobalName())) {
    DxilResourceBase *res0 = m_resourceMap[res->GetGlobalName()].first;
    Type *Ty0 = res0->GetHLSLType()->getPointerElementType();
    Type *Ty = res->GetHLSLType()->getPointerElementType();
    // Make sure res0 match res.
    bool bMatch = IsMatchedType(Ty0, Ty);
    if (!bMatch) {
      // Report error.
      dxilutil::EmitErrorOnGlobalVariable(m_ctx, dyn_cast<GlobalVariable>(res->GetGlobalSymbol()),
                                          Twine(kRedefineResource) + res->GetResClassName() + " for " +
                                          res->GetGlobalName());
      return false;
    }
  } else {
    m_resourceMap[res->GetGlobalName()] = std::make_pair(res, GV);
  }
  return true;
}

void DxilLinkJob::AddResourceToDM(DxilModule &DM) {
  for (auto &it : m_resourceMap) {
    DxilResourceBase *res = it.second.first;
    GlobalVariable *GV = it.second.second;
    unsigned ID = 0;
    DxilResourceBase *basePtr = nullptr;
    switch (res->GetClass()) {
    case DXIL::ResourceClass::UAV: {
      std::unique_ptr<DxilResource> pUAV = llvm::make_unique<DxilResource>();
      DxilResource *ptr = pUAV.get();
      // Copy the content.
      *ptr = *(static_cast<DxilResource *>(res));
      ID = DM.AddUAV(std::move(pUAV));
      basePtr = &DM.GetUAV(ID);
    } break;
    case DXIL::ResourceClass::SRV: {
      std::unique_ptr<DxilResource> pSRV = llvm::make_unique<DxilResource>();
      DxilResource *ptr = pSRV.get();
      // Copy the content.
      *ptr = *(static_cast<DxilResource *>(res));
      ID = DM.AddSRV(std::move(pSRV));
      basePtr = &DM.GetSRV(ID);
    } break;
    case DXIL::ResourceClass::CBuffer: {
      std::unique_ptr<DxilCBuffer> pCBuf = llvm::make_unique<DxilCBuffer>();
      DxilCBuffer *ptr = pCBuf.get();
      // Copy the content.
      *ptr = *(static_cast<DxilCBuffer *>(res));
      ID = DM.AddCBuffer(std::move(pCBuf));
      basePtr = &DM.GetCBuffer(ID);
    } break;
    case DXIL::ResourceClass::Sampler: {
      std::unique_ptr<DxilSampler> pSampler = llvm::make_unique<DxilSampler>();
      DxilSampler *ptr = pSampler.get();
      // Copy the content.
      *ptr = *(static_cast<DxilSampler *>(res));
      ID = DM.AddSampler(std::move(pSampler));
      basePtr = &DM.GetSampler(ID);
    }
    default:
      DXASSERT(res->GetClass() == DXIL::ResourceClass::Sampler,
               "else invalid resource");
      break;
    }
    // Update ID.
    basePtr->SetID(ID);

    basePtr->SetGlobalSymbol(GV);
    DM.GetLLVMUsed().push_back(GV);
  }
  // Prevent global vars used for resources from being deleted through optimizations
  // while we still have hidden uses (pointers in resource vectors).
  DM.EmitLLVMUsed();
}

void DxilLinkJob::LinkNamedMDNodes(Module *pM, ValueToValueMapTy &vmap) {
  SetVector<Module *> moduleSet;
  for (auto &it : m_functionDefs) {
    DxilLib *pLib = it.second;
    moduleSet.insert(pLib->GetDxilModule().GetModule());
  }
  // Link normal NamedMDNode.
  // TODO: skip duplicate operands.
  for (Module *pSrcM : moduleSet) {
    const NamedMDNode *pSrcModFlags = pSrcM->getModuleFlagsMetadata();
    for (const NamedMDNode &NMD : pSrcM->named_metadata()) {
      // Don't link module flags here. Do them separately.
      if (&NMD == pSrcModFlags)
        continue;
      // Skip dxil metadata which will be regenerated.
      if (DxilMDHelper::IsKnownNamedMetaData(NMD))
        continue;
      NamedMDNode *DestNMD = pM->getOrInsertNamedMetadata(NMD.getName());
      // Add Src elements into Dest node.
      for (const MDNode *op : NMD.operands())
        DestNMD->addOperand(MapMetadata(op, vmap, RF_None, /*TypeMap*/ nullptr,
                                        /*ValMaterializer*/ nullptr));
    }
  }
  // Link mod flags.
  SetVector<MDNode *> flagSet;
  for (Module *pSrcM : moduleSet) {
    NamedMDNode *pSrcModFlags = pSrcM->getModuleFlagsMetadata();
    if (pSrcModFlags) {
      for (MDNode *flag : pSrcModFlags->operands()) {
        flagSet.insert(flag);
      }
    }
  }
  // TODO: check conflict in flags.
  if (!flagSet.empty()) {
    NamedMDNode *ModFlags = pM->getOrInsertModuleFlagsMetadata();
    for (MDNode *flag : flagSet) {
      ModFlags->addOperand(flag);
    }
  }
}

void DxilLinkJob::AddFunctionDecls(Module *pM) {
  for (auto &it : m_functionDecls) {
    for (auto F : it.second.second) {
      Function *NewF = pM->getFunction(F->getName());
      if (!NewF || F->getFunctionType() != NewF->getFunctionType()) {
        NewF = Function::Create(F->getFunctionType(), F->getLinkage(),
                                          F->getName(), pM);
        NewF->setAttributes(F->getAttributes());
      }
      m_newFunctions[F->getName()] = NewF;
    }
  }
}

bool DxilLinkJob::AddGlobals(DxilModule &DM, ValueToValueMapTy &vmap) {
  DxilTypeSystem &typeSys = DM.GetTypeSystem();
  Module *pM = DM.GetModule();
  bool bSuccess = true;
  for (auto &it : m_functionDefs) {
    DxilFunctionLinkInfo *linkInfo = it.first;
    DxilLib *pLib = it.second;
    DxilModule &tmpDM = pLib->GetDxilModule();
    DxilTypeSystem &tmpTypeSys = tmpDM.GetTypeSystem();
    for (GlobalVariable *GV : linkInfo->usedGVs) {
      // Skip added globals.
      if (m_newGlobals.count(GV->getName())) {
        if (vmap.find(GV) == vmap.end()) {
          if (DxilResourceBase *res = pLib->GetResource(GV)) {
            // For resource of same name, if class and type match, just map to
            // same NewGV.
            GlobalVariable *NewGV = m_newGlobals[GV->getName()];
            if (AddResource(res, NewGV)) {
              vmap[GV] = NewGV;
            } else {
              bSuccess = false;
            }
            continue;
          }

          // Redefine of global.
          dxilutil::EmitErrorOnGlobalVariable(m_ctx, GV, Twine(kRedefineGlobal) + GV->getName());
          bSuccess = false;
        }
        continue;
      }
      Constant *Initializer = nullptr;
      if (GV->hasInitializer())
        Initializer = GV->getInitializer();

      Type *Ty = GV->getType()->getElementType();
      GlobalVariable *NewGV = new GlobalVariable(
          *pM, Ty, GV->isConstant(), GV->getLinkage(), Initializer,
          GV->getName(),
          /*InsertBefore*/ nullptr, GV->getThreadLocalMode(),
          GV->getType()->getAddressSpace(), GV->isExternallyInitialized());

      m_newGlobals[GV->getName()] = NewGV;

      vmap[GV] = NewGV;

      typeSys.CopyTypeAnnotation(Ty, tmpTypeSys);

      if (DxilResourceBase *res = pLib->GetResource(GV)) {
        bSuccess &= AddResource(res, NewGV);
      }
    }
  }
  return bSuccess;
}

void DxilLinkJob::CloneFunctions(ValueToValueMapTy &vmap) {
  for (auto &it : m_functionDefs) {
    DxilFunctionLinkInfo *linkInfo = it.first;

    Function *F = linkInfo->func;
    Function *NewF = m_newFunctions[F->getName()];

    // Add dxil functions to vmap.
    for (Function *UsedF : linkInfo->usedFunctions) {
      if (!vmap.count(UsedF)) {
        // Extern function need match by name
        DXASSERT(m_newFunctions.count(UsedF->getName()),
                 "Must have new function.");
        vmap[UsedF] = m_newFunctions[UsedF->getName()];
      }
    }

    CloneFunction(F, NewF, vmap);
  }
}

void DxilLinkJob::AddFunctions(DxilModule &DM, ValueToValueMapTy &vmap) {
  DxilTypeSystem &typeSys = DM.GetTypeSystem();
  Module *pM = DM.GetModule();
  for (auto &it : m_functionDefs) {
    DxilFunctionLinkInfo *linkInfo = it.first;
    DxilLib *pLib = it.second;
    DxilModule &tmpDM = pLib->GetDxilModule();
    DxilTypeSystem &tmpTypeSys = tmpDM.GetTypeSystem();

    Function *F = linkInfo->func;
    Function *NewF = Function::Create(F->getFunctionType(), F->getLinkage(),
                                      F->getName(), pM);
    NewF->setAttributes(F->getAttributes());

    if (!NewF->hasFnAttribute(llvm::Attribute::NoInline))
      NewF->addFnAttr(llvm::Attribute::AlwaysInline);

    if (DxilFunctionAnnotation *funcAnnotation =
            tmpTypeSys.GetFunctionAnnotation(F)) {
      // Clone funcAnnotation to typeSys.
      typeSys.CopyFunctionAnnotation(NewF, F, tmpTypeSys);
    }

    // Add to function map.
    m_newFunctions[NewF->getName()] = NewF;

    vmap[F] = NewF;
  }
}

std::unique_ptr<Module>
DxilLinkJob::Link(std::pair<DxilFunctionLinkInfo *, DxilLib *> &entryLinkPair,
                  const ShaderModel *pSM) {
  Function *entryFunc = entryLinkPair.first->func;
  DxilModule &entryDM = entryLinkPair.second->GetDxilModule();
  if (!entryDM.HasDxilFunctionProps(entryFunc)) {
    // Cannot get function props.
    dxilutil::EmitErrorOnFunction(m_ctx, entryFunc, Twine(kNoEntryProps) + entryFunc->getName());
    return nullptr;
  }

  DxilFunctionProps props = entryDM.GetDxilFunctionProps(entryFunc);

  if (pSM->GetKind() != props.shaderKind) {
    // Shader kind mismatch.
    dxilutil::EmitErrorOnFunction(m_ctx, entryFunc, Twine(kShaderKindMismatch) +
                                  ShaderModel::GetKindName(pSM->GetKind()) + " and " +
                                  ShaderModel::GetKindName(props.shaderKind));
    return nullptr;
  }

  // Create new module.
  std::unique_ptr<Module> pM =
      llvm::make_unique<Module>(entryFunc->getName(), entryDM.GetCtx());
  // Set target.
  pM->setTargetTriple(entryDM.GetModule()->getTargetTriple());
  // Add dxil operation functions before create DxilModule.
  AddFunctionDecls(pM.get());

  // Create DxilModule.
  const bool bSkipInit = true;
  DxilModule &DM = pM->GetOrCreateDxilModule(bSkipInit);
  DM.SetShaderModel(pSM, entryDM.GetUseMinPrecision());

  // Set Validator version.
  DM.SetValidatorVersion(m_valMajor, m_valMinor);

  ValueToValueMapTy vmap;

  // Add function
  AddFunctions(DM, vmap);

  // Set Entry
  Function *NewEntryFunc = m_newFunctions[entryFunc->getName()];
  DM.SetEntryFunction(NewEntryFunc);
  DM.SetEntryFunctionName(entryFunc->getName());

  DxilEntryPropsMap EntryPropMap;
  std::unique_ptr<DxilEntryProps> pProps =
      llvm::make_unique<DxilEntryProps>(entryDM.GetDxilEntryProps(entryFunc));
  EntryPropMap[NewEntryFunc] = std::move(pProps);
  DM.ResetEntryPropsMap(std::move(EntryPropMap));


  if (NewEntryFunc->hasFnAttribute(llvm::Attribute::AlwaysInline))
    NewEntryFunc->removeFnAttr(llvm::Attribute::AlwaysInline);
  if (props.IsHS()) {
    Function *patchConstantFunc = props.ShaderProps.HS.patchConstantFunc;
    Function *newPatchConstantFunc =
        m_newFunctions[patchConstantFunc->getName()];
    props.ShaderProps.HS.patchConstantFunc = newPatchConstantFunc;

    if (newPatchConstantFunc->hasFnAttribute(llvm::Attribute::AlwaysInline))
      newPatchConstantFunc->removeFnAttr(llvm::Attribute::AlwaysInline);
  }

  // Set root sig if exist.
  if (!props.serializedRootSignature.empty()) {
    DM.ResetSerializedRootSignature(props.serializedRootSignature);
    props.serializedRootSignature.clear();
  }
  // Set EntryProps
  DM.SetShaderProperties(&props);

  // Add global
  bool bSuccess = AddGlobals(DM, vmap);
  if (!bSuccess)
    return nullptr;

  // Clone functions.
  CloneFunctions(vmap);

  // Call global constrctor.
  IRBuilder<> Builder(dxilutil::FindAllocaInsertionPt(DM.GetEntryFunction()));
  for (auto &it : m_functionDefs) {
    DxilFunctionLinkInfo *linkInfo = it.first;
    DxilLib *pLib = it.second;
    // Skip constructor in entry lib which is already called for entries inside
    // entry lib.
    if (pLib == entryLinkPair.second)
      continue;
    Function *F = linkInfo->func;
    if (pLib->IsInitFunc(F)) {
      Function *NewF = m_newFunctions[F->getName()];
      Builder.CreateCall(NewF);
    }
  }

  // Refresh intrinsic cache.
  DM.GetOP()->RefreshCache();

  // Add resource to DM.
  // This should be after functions cloned.
  AddResourceToDM(DM);

  // Link metadata like debug info.
  LinkNamedMDNodes(pM.get(), vmap);

  RunPreparePass(*pM);

  return pM;
}

// Based on CodeGenModule::EmitCtorList.
void DxilLinkJob::EmitCtorListForLib(Module *pM) {
  LLVMContext &Ctx = pM->getContext();

  Type *VoidTy = Type::getVoidTy(Ctx);
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  Type *VoidPtrTy = Type::getInt8PtrTy(Ctx);
  // Ctor function type is void()*.
  llvm::FunctionType *CtorFTy = llvm::FunctionType::get(VoidTy, false);
  llvm::Type *CtorPFTy = llvm::PointerType::getUnqual(CtorFTy);

  // Get the type of a ctor entry, { i32, void ()*, i8* }.
  llvm::StructType *CtorStructTy = llvm::StructType::get(
      Int32Ty, llvm::PointerType::getUnqual(CtorFTy), VoidPtrTy, nullptr);

  // Construct the constructor and destructor arrays.
  SmallVector<llvm::Constant *, 8> Ctors;

  for (auto &it : m_functionDefs) {
    DxilFunctionLinkInfo *linkInfo = it.first;
    DxilLib *pLib = it.second;

    Function *F = linkInfo->func;
    if (pLib->IsInitFunc(F)) {
      Function *NewF = m_newFunctions[F->getName()];

      llvm::Constant *S[] = {
          llvm::ConstantInt::get(Int32Ty, 65535, false),
          llvm::ConstantExpr::getBitCast(NewF, CtorPFTy),
          (llvm::Constant::getNullValue(VoidPtrTy))};
      Ctors.push_back(llvm::ConstantStruct::get(CtorStructTy, S));
    }
  }

  if (!Ctors.empty()) {
    const StringRef GlobalName = "llvm.global_ctors";
    llvm::ArrayType *AT = llvm::ArrayType::get(CtorStructTy, Ctors.size());
    new llvm::GlobalVariable(*pM, AT, false,
                             llvm::GlobalValue::AppendingLinkage,
                             llvm::ConstantArray::get(AT, Ctors), GlobalName);
  }
}

std::unique_ptr<Module>
DxilLinkJob::LinkToLib(const ShaderModel *pSM) {
  if (m_functionDefs.empty()) {
    m_ctx.emitError(Twine(kNoFunctionsToExport));
    return nullptr;
  }
  DxilLib *pLib = m_functionDefs.begin()->second;
  DxilModule &tmpDM = pLib->GetDxilModule();
  // Create new module.
  std::unique_ptr<Module> pM =
      llvm::make_unique<Module>("merged_lib", tmpDM.GetCtx());
  // Set target.
  pM->setTargetTriple(tmpDM.GetModule()->getTargetTriple());
  // Add dxil operation functions and external decls before create DxilModule.
  AddFunctionDecls(pM.get());

  // Create DxilModule.
  const bool bSkipInit = true;
  DxilModule &DM = pM->GetOrCreateDxilModule(bSkipInit);
  DM.SetShaderModel(pSM, tmpDM.GetUseMinPrecision());

  // Set Validator version.
  DM.SetValidatorVersion(m_valMajor, m_valMinor);

  ValueToValueMapTy vmap;

  // Add function
  AddFunctions(DM, vmap);

  // Set DxilFunctionProps.
  DxilEntryPropsMap EntryPropMap;
  for (auto &it : m_functionDefs) {
    DxilFunctionLinkInfo *linkInfo = it.first;
    DxilLib *pLib = it.second;
    DxilModule &tmpDM = pLib->GetDxilModule();

    Function *F = linkInfo->func;
    if (tmpDM.HasDxilEntryProps(F)) {
      Function *NewF = m_newFunctions[F->getName()];
      DxilEntryProps &props = tmpDM.GetDxilEntryProps(F);
      std::unique_ptr<DxilEntryProps> pProps =
          llvm::make_unique<DxilEntryProps>(props);
      EntryPropMap[NewF] = std::move(pProps);
    }
  }
  DM.ResetEntryPropsMap(std::move(EntryPropMap));

  // Add global
  bool bSuccess = AddGlobals(DM, vmap);
  if (!bSuccess)
    return nullptr;

  // Clone functions.
  CloneFunctions(vmap);

  // Refresh intrinsic cache.
  DM.GetOP()->RefreshCache();

  // Add resource to DM.
  // This should be after functions cloned.
  AddResourceToDM(DM);

  // Link metadata like debug info.
  LinkNamedMDNodes(pM.get(), vmap);

  // Build global.ctors.
  EmitCtorListForLib(pM.get());

  RunPreparePass(*pM);

  if (!m_exportMap.empty()) {
    m_exportMap.BeginProcessing();

    DM.ClearDxilMetadata(*pM);
    for (auto it = pM->begin(); it != pM->end();) {
      Function *F = it++;
      if (F->isDeclaration())
        continue;
      if (!m_exportMap.ProcessFunction(F, true)) {
        // Remove Function not in exportMap.
        DM.RemoveFunction(F);
        F->eraseFromParent();
      }
    }

    if(!m_exportMap.EndProcessing()) {
      for (auto &name : m_exportMap.GetNameCollisions()) {
        std::string escaped;
        llvm::raw_string_ostream os(escaped);
        dxilutil::PrintEscapedString(name, os);
        m_ctx.emitError(Twine(kExportNameCollision) + os.str());
      }
      for (auto &name : m_exportMap.GetUnusedExports()) {
        std::string escaped;
        llvm::raw_string_ostream os(escaped);
        dxilutil::PrintEscapedString(name, os);
        m_ctx.emitError(Twine(kExportFunctionMissing) + os.str());
      }
      return nullptr;
    }

    // Rename the original, if necessary, then clone the rest
    for (auto &it : m_exportMap.GetFunctionRenames()) {
      Function *F = it.first;
      auto &renames = it.second;

      if (renames.empty())
        continue;

      auto itName = renames.begin();

      // Rename the original, if necessary, then clone the rest
      if (renames.find(F->getName()) == renames.end())
        F->setName(*(itName++));

      while (itName != renames.end()) {
        if (F->getName() != *itName) {
          Function *NewF = Function::Create(F->getFunctionType(),
            GlobalValue::LinkageTypes::ExternalLinkage,
            *itName, DM.GetModule());
          ValueToValueMapTy vmap;
          CloneFunction(F, NewF, vmap, &DM.GetTypeSystem());
          // add DxilFunctionProps if entry
          if (DM.HasDxilFunctionProps(F)) {
            DM.CloneDxilEntryProps(F, NewF);
          }
        }
        itName++;
      }
    }

    DM.EmitDxilMetadata();
  }

  return pM;
}

void DxilLinkJob::AddFunction(
    std::pair<DxilFunctionLinkInfo *, DxilLib *> &linkPair) {
  m_functionDefs[linkPair.first] = linkPair.second;
}

void DxilLinkJob::AddFunction(llvm::Function *F) {
  // Rarely, DXIL op overloads could collide, due to different types with same name.
  // Later, we will rename these functions, but for now, we need to prevent clobbering
  // an existing entry.
  auto &entry = m_functionDecls[F->getName()];
  if (entry.first.insert(F->getFunctionType()).second)
    entry.second.push_back(F);
}

// Clone of StripDeadDebugInfo::runOnModule.
// Also remove function which not not in current Module.
void DxilLinkJob::StripDeadDebugInfo(Module &M) {
  LLVMContext &C = M.getContext();
  // Find all debug info in F. This is actually overkill in terms of what we
  // want to do, but we want to try and be as resilient as possible in the face
  // of potential debug info changes by using the formal interfaces given to us
  // as much as possible.
  DebugInfoFinder F;
  F.processModule(M);

  // For each compile unit, find the live set of global variables/functions and
  // replace the current list of potentially dead global variables/functions
  // with the live list.
  SmallVector<Metadata *, 64> LiveGlobalVariables;
  SmallVector<Metadata *, 64> LiveSubprograms;
  DenseSet<const MDNode *> VisitedSet;

  for (DICompileUnit *DIC : F.compile_units()) {
    // Create our live subprogram list.
    bool SubprogramChange = false;
    for (DISubprogram *DISP : DIC->getSubprograms()) {
      // Make sure we visit each subprogram only once.
      if (!VisitedSet.insert(DISP).second)
        continue;

      // If the function referenced by DISP is not null, the function is live.
      if (Function *Func = DISP->getFunction()) {
        LiveSubprograms.push_back(DISP);
        if (Func->getParent() != &M)
          DISP->replaceFunction(nullptr);
      } else {
        // Copy it in anyway even if there's no function. When function is inlined
        // the function reference is gone, but the subprogram is still valid as
        // scope.
        LiveSubprograms.push_back(DISP);
      }
    }

    // Create our live global variable list.
    bool GlobalVariableChange = false;
    for (DIGlobalVariable *DIG : DIC->getGlobalVariables()) {
      // Make sure we only visit each global variable only once.
      if (!VisitedSet.insert(DIG).second)
        continue;

      // If the global variable referenced by DIG is not null, the global
      // variable is live.
      if (Constant *CV = DIG->getVariable()) {
        if (GlobalVariable *GV = dyn_cast<GlobalVariable>(CV)) {
          if (GV->getParent() == &M) {
            LiveGlobalVariables.push_back(DIG);
          } else {
            GlobalVariableChange = true;
          }
        } else {
          LiveGlobalVariables.push_back(DIG);
        }
      } else {
        GlobalVariableChange = true;
      }
    }

    // If we found dead subprograms or global variables, replace the current
    // subprogram list/global variable list with our new live subprogram/global
    // variable list.
    if (SubprogramChange) {
      DIC->replaceSubprograms(MDTuple::get(C, LiveSubprograms));
    }

    if (GlobalVariableChange) {
      DIC->replaceGlobalVariables(MDTuple::get(C, LiveGlobalVariables));
    }

    // Reset lists for the next iteration.
    LiveSubprograms.clear();
    LiveGlobalVariables.clear();
  }
}

// TODO: move FixShaderModelMismatch to separate file.
#include "dxc/DXIL/DxilInstructions.h"
namespace {
bool onlyUsedByAnnotateHandle(Value *V) {
  bool bResult = true;
  for (User *U : V->users()) {
    CallInst *CI = dyn_cast<CallInst>(U);
    if (!CI) {
      bResult = false;
      break;
    }
    DxilInst_AnnotateHandle Hdl(CI);
    if (!Hdl) {
      bResult = false;
      break;
    }
  }
  return bResult;
}

DxilResourceBase *
findResourceFromPtr(Value *Ptr, DxilModule &DM,
                    DenseMap<Value *, DxilResourceBase *> &PtrResMap) {
  auto it = PtrResMap.find(Ptr);
  if (Ptr)
    return it->second;
  DxilResourceBase *Res = nullptr;
  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Ptr)) {
    DXASSERT(false, "global resource should already in map");
  } else {
    // Not support allocaInst of resource when missing annotateHandle.
    GEPOperator *GEP = cast<GEPOperator>(Ptr);
    Res = findResourceFromPtr(GEP->getPointerOperand(), DM, PtrResMap);
  }
  PtrResMap[Ptr] = Res;
  return Res;
}

template <typename T>
void addGVFromResTable(T &Tab,
                       DenseMap<Value *, DxilResourceBase *> &PtrResMap) {
  for (auto &it : Tab) {
    PtrResMap[it->GetGlobalSymbol()] = it.get();
  }
}

// Make sure createHandleForLib is annotated before use.
bool addAnnotHandle(Module &M, DxilModule &DM) {
  hlsl::OP *hlslOP = DM.GetOP();
  auto *pSM = DM.GetShaderModel();
  if (!pSM->IsSM66Plus())
    return false;
  // If no createHandleForLib, do nothing.
  if (!hlslOP->IsDxilOpUsed(DXIL::OpCode::CreateHandleForLib))
    return false;

  Type *pVoidTy = Type::getVoidTy(M.getContext());
  SmallVector<CallInst *, 4> Candidates;
  for (Function &F : M) {
    if (!F.isDeclaration())
      continue;
    if (!hlslOP->IsDxilOpFunc(&F))
      continue;
    DXIL::OpCodeClass opClass;
    if (!hlslOP->GetOpCodeClass(&F, opClass))
      continue;
    if (opClass != DXIL::OpCodeClass::CreateHandleForLib)
      continue;
    for (User *U : F.users()) {
      CallInst *CI = cast<CallInst>(U);
      // Check user is annotateHandle.
      if (onlyUsedByAnnotateHandle(CI))
        continue;
      Candidates.emplace_back(CI);
    }
  }

  if (Candidates.empty())
    return false;

  DenseMap<Value *, DxilResourceBase *> PtrResMap;
  // Add GV from resTable first.
  addGVFromResTable(DM.GetCBuffers(), PtrResMap);
  addGVFromResTable(DM.GetSRVs(), PtrResMap);
  addGVFromResTable(DM.GetUAVs(), PtrResMap);
  addGVFromResTable(DM.GetSamplers(), PtrResMap);

  Function *annotHandleFn =
      hlslOP->GetOpFunc(DXIL::OpCode::AnnotateHandle, pVoidTy);
  Value *annotHandleArg =
      hlslOP->GetI32Const((unsigned)DXIL::OpCode::AnnotateHandle);
  // Replace createHandle with annotateHandle and createHandleFromBinding.
  Type *resPropertyTy = hlslOP->GetResourcePropertiesType();
  for (CallInst *CI : Candidates) {
    DxilInst_CreateHandleForLib Hdl(CI);
    LoadInst *Ld = cast<LoadInst>(Hdl.get_Resource());
    Value *Ptr = Ld->getPointerOperand();
    DxilResourceBase *Res = findResourceFromPtr(Ptr, DM, PtrResMap);
    DXASSERT(Res, "fail to find resource when missing annotateHandle");

    DxilResourceProperties RP = resource_helper::loadPropsFromResourceBase(Res);
    Value *propertiesV =
        resource_helper::getAsConstant(RP, resPropertyTy, *DM.GetShaderModel());
    IRBuilder<> B(CI->getNextNode());
    CallInst *annotHdl =
        B.CreateCall(annotHandleFn, {annotHandleArg, CI, propertiesV});
    CI->replaceAllUsesWith(annotHdl);
    annotHdl->setArgOperand(DxilInst_AnnotateHandle::arg_res, CI);
  }
  return true;
}
} // namespace

void DxilLinkJob::FixShaderModelMismatch(llvm::Module &M) {
  // TODO: fix more issues.
  addAnnotHandle(M, M.GetDxilModule());
}

void DxilLinkJob::RunPreparePass(Module &M) {
  StripDeadDebugInfo(M);
  FixShaderModelMismatch(M);

  DxilModule &DM = M.GetDxilModule();
  const ShaderModel *pSM = DM.GetShaderModel();

  legacy::PassManager PM;
  PM.add(createAlwaysInlinerPass(/*InsertLifeTime*/ false));

  // Remove unused functions.
  PM.add(createDxilDeadFunctionEliminationPass());

  // SROA
  PM.add(createSROAPass(/*RequiresDomTree*/false, /*SkipHLSLMat*/false));
  // For static global handle.
  PM.add(createLowerStaticGlobalIntoAlloca());

  // Remove MultiDimArray from function call arg.
  PM.add(createMultiDimArrayToOneDimArrayPass());

  // Lower matrix bitcast.
  PM.add(createMatrixBitcastLowerPass());

  // mem2reg.
  PM.add(createPromoteMemoryToRegisterPass());

  // Clean up vectors, and run mem2reg again
  PM.add(createScalarizerPass());
  PM.add(createPromoteMemoryToRegisterPass());

  PM.add(createSimplifyInstPass());
  PM.add(createCFGSimplificationPass());

  PM.add(createDeadCodeEliminationPass());
  PM.add(createGlobalDCEPass());

  if (pSM->IsSM66Plus() && pSM->IsLib())
    PM.add(createDxilMutateResourceToHandlePass());
  PM.add(createDxilCleanupDynamicResourceHandlePass());
  PM.add(createDxilLowerCreateHandleForLibPass());
  PM.add(createDxilTranslateRawBuffer());
  PM.add(createDxilFinalizeModulePass());
  PM.add(createComputeViewIdStatePass());
  PM.add(createDxilDeadFunctionEliminationPass());
  PM.add(createNoPausePassesPass());
  PM.add(createDxilEmitMetadataPass());

  PM.run(M);
}

//------------------------------------------------------------------------------
//
// DxilLinkerImpl methods.
//

bool DxilLinkerImpl::HasLibNameRegistered(StringRef name) {
  return m_LibMap.count(name);
}

bool DxilLinkerImpl::RegisterLib(StringRef name,
                                 std::unique_ptr<llvm::Module> pModule,
                                 std::unique_ptr<llvm::Module> pDebugModule) {
  if (m_LibMap.count(name))
    return false;

  std::unique_ptr<llvm::Module> pM =
      pDebugModule ? std::move(pDebugModule) : std::move(pModule);

  if (!pM)
    return false;

  pM->setModuleIdentifier(name);
  std::unique_ptr<DxilLib> pLib =
      llvm::make_unique<DxilLib>(std::move(pM));
  m_LibMap[name] = std::move(pLib);
  return true;
}

bool DxilLinkerImpl::AttachLib(StringRef name) {
  auto iter = m_LibMap.find(name);
  if (iter == m_LibMap.end()) {
    return false;
  }

  return AttachLib(iter->second.get());
}
bool DxilLinkerImpl::DetachLib(StringRef name) {
  auto iter = m_LibMap.find(name);
  if (iter == m_LibMap.end()) {
    return false;
  }
  return DetachLib(iter->second.get());
}

void DxilLinkerImpl::DetachAll() {
  m_functionNameMap.clear();
  m_attachedLibs.clear();
}

bool DxilLinkerImpl::AttachLib(DxilLib *lib) {
  if (!lib) {
    // Invalid arg.
    return false;
  }

  if (m_attachedLibs.count(lib))
    return false;

  StringMap<std::unique_ptr<DxilFunctionLinkInfo>> &funcTable =
      lib->GetFunctionTable();
  bool bSuccess = true;
  for (auto it = funcTable.begin(), e = funcTable.end(); it != e; it++) {
    StringRef name = it->getKey();
    if (m_functionNameMap.count(name)) {
      // Redefine of function.
      const DxilFunctionLinkInfo *DFLI = it->getValue().get();
      dxilutil::EmitErrorOnFunction(m_ctx, DFLI->func, Twine(kRedefineFunction) + name);
      bSuccess = false;
      continue;
    }
    m_functionNameMap[name] = std::make_pair(it->second.get(), lib);
  }

  if (bSuccess) {
    m_attachedLibs.insert(lib);
  } else {
    for (auto it = funcTable.begin(), e = funcTable.end(); it != e; it++) {
      StringRef name = it->getKey();
      auto iter = m_functionNameMap.find(name);

      if (iter == m_functionNameMap.end())
        continue;

      // Remove functions of lib.
      if (m_functionNameMap[name].second == lib)
        m_functionNameMap.erase(name);
    }
  }

  return bSuccess;
}

bool DxilLinkerImpl::DetachLib(DxilLib *lib) {
  if (!lib) {
    // Invalid arg.
    return false;
  }

  if (!m_attachedLibs.count(lib))
    return false;

  m_attachedLibs.erase(lib);

  // Remove functions from lib.
  StringMap<std::unique_ptr<DxilFunctionLinkInfo>> &funcTable =
      lib->GetFunctionTable();
  for (auto it = funcTable.begin(), e = funcTable.end(); it != e; it++) {
    StringRef name = it->getKey();
    m_functionNameMap.erase(name);
  }
  return true;
}

bool DxilLinkerImpl::AddFunctions(SmallVector<StringRef, 4> &workList,
                                  SetVector<DxilLib *> &libSet,
                                  SetVector<StringRef> &addedFunctionSet,
                                  DxilLinkJob &linkJob, bool bLazyLoadDone,
                                  bool bAllowFuncionDecls) {
  while (!workList.empty()) {
    StringRef name = workList.pop_back_val();
    // Ignore added function.
    if (addedFunctionSet.count(name))
      continue;
    if (!m_functionNameMap.count(name)) {
      // Cannot find function, report error.
      m_ctx.emitError(Twine(kUndefFunction) + name);
      return false;
    }

    std::pair<DxilFunctionLinkInfo *, DxilLib *> &linkPair =
        m_functionNameMap[name];
    linkJob.AddFunction(linkPair);

    DxilLib *pLib = linkPair.second;
    libSet.insert(pLib);
    if (!bLazyLoadDone) {
      Function *F = linkPair.first->func;
      pLib->LazyLoadFunction(F);
    }
    for (Function *F : linkPair.first->usedFunctions) {
      if (hlsl::OP::IsDxilOpFunc(F) || F->isIntrinsic()) {
        // Add dxil operations directly.
        linkJob.AddFunction(F);
      } else if (addedFunctionSet.count(F->getName()) == 0) {
        if (bAllowFuncionDecls && F->isDeclaration() && !m_functionNameMap.count(F->getName())) {
          // When linking to lib, use of undefined function is allowed; add directly.
          linkJob.AddFunction(F);
        } else {
          // Push function name to work list.
          workList.emplace_back(F->getName());
        }
      }
    }

    addedFunctionSet.insert(name);
  }
  return true;
}

std::unique_ptr<llvm::Module>
DxilLinkerImpl::Link(StringRef entry, StringRef profile, dxilutil::ExportMap &exportMap) {
  const ShaderModel *pSM = ShaderModel::GetByName(profile.data());
  DXIL::ShaderKind kind = pSM->GetKind();
  if (kind == DXIL::ShaderKind::Invalid ||
      (kind >= DXIL::ShaderKind::RayGeneration &&
       kind <= DXIL::ShaderKind::Callable)) {
    m_ctx.emitError(profile + Twine(kInvalidProfile));
    // Invalid profile.
    return nullptr;
  }

  if (!exportMap.empty() && kind != DXIL::ShaderKind::Library) {
    m_ctx.emitError(Twine(kExportOnlyForLib));
    return nullptr;
  }

  // Verifying validator version supports the requested profile
  unsigned minValMajor, minValMinor;
  pSM->GetMinValidatorVersion(minValMajor, minValMinor);
  if (minValMajor > m_valMajor ||
      (minValMajor == m_valMajor && minValMinor > m_valMinor)) {
    m_ctx.emitError(Twine(kInvalidValidatorVersion) + profile);
    return nullptr;
  }

  DxilLinkJob linkJob(m_ctx, exportMap, m_valMajor, m_valMinor);

  SetVector<DxilLib *> libSet;
  SetVector<StringRef> addedFunctionSet;

  bool bIsLib = pSM->IsLib();
  if (!bIsLib) {
    SmallVector<StringRef, 4> workList;
    workList.emplace_back(entry);

    if (!AddFunctions(workList, libSet, addedFunctionSet, linkJob,
                      /*bLazyLoadDone*/ false,
                      /*bAllowFuncionDecls*/ false))
      return nullptr;

  } else {
    if (exportMap.empty() && !exportMap.isExportShadersOnly()) {
      // Add every function for lib profile.
      for (auto &it : m_functionNameMap) {
        StringRef name = it.getKey();
        std::pair<DxilFunctionLinkInfo *, DxilLib *> &linkPair = it.second;
        DxilFunctionLinkInfo *linkInfo = linkPair.first;
        DxilLib *pLib = linkPair.second;

        Function *F = linkInfo->func;
        pLib->LazyLoadFunction(F);

        linkJob.AddFunction(linkPair);

        libSet.insert(pLib);

        addedFunctionSet.insert(name);
      }
      // Add every dxil function and llvm intrinsic.
      for (auto *pLib : libSet) {
        auto &DM = pLib->GetDxilModule();
        DM.GetOP();
        auto *pM = DM.GetModule();
        for (Function &F : pM->functions()) {
          if (hlsl::OP::IsDxilOpFunc(&F) || F.isIntrinsic() ||
            (F.isDeclaration() && m_functionNameMap.count(F.getName()) == 0)) {
            // Add intrinsics and function decls still not defined in any lib
            linkJob.AddFunction(&F);
          }
        }
      }
    } else if (exportMap.isExportShadersOnly()) {
      SmallVector<StringRef, 4> workList;
      for (auto *pLib : m_attachedLibs) {
        auto &DM = pLib->GetDxilModule();
        auto *pM = DM.GetModule();
        for (Function &F : pM->functions()) {
          if (!pLib->IsEntry(&F)) {
            if (!F.isDeclaration()) {
              // Set none entry to be internal so they could be removed.
              F.setLinkage(GlobalValue::LinkageTypes::InternalLinkage);
            }
            continue;
          }
          workList.emplace_back(F.getName());
        }
        libSet.insert(pLib);
      }

      if (!AddFunctions(workList, libSet, addedFunctionSet, linkJob,
                        /*bLazyLoadDone*/ false,
                        /*bAllowFuncionDecls*/ false))
        return nullptr;
    } else {
      SmallVector<StringRef, 4> workList;

      // Only add exported functions.
      for (auto &it : m_functionNameMap) {
        StringRef name = it.getKey();
        // Only add names exist in exportMap.
        if (exportMap.IsExported(name))
          workList.emplace_back(name);
      }

      if (!AddFunctions(workList, libSet, addedFunctionSet, linkJob,
                        /*bLazyLoadDone*/ false,
                        /*bAllowFuncionDecls*/ true))
        return nullptr;
    }
  }

  // Save global users.
  for (auto &pLib : libSet) {
    pLib->BuildGlobalUsage();
  }

  SmallVector<StringRef, 4> workList;
  // Save global ctor users.
  for (auto &pLib : libSet) {
    pLib->CollectUsedInitFunctions(addedFunctionSet, workList);
  }

  for (auto &pLib : libSet) {
    pLib->FixIntrinsicOverloads();
  }

  // Add init functions if used.
  // All init function already loaded in BuildGlobalUsage,
  // so set bLazyLoadDone to true here.
  // Decls should have been added to addedFunctionSet if lib,
  // so set bAllowFuncionDecls is false here.
  if (!AddFunctions(workList, libSet, addedFunctionSet, linkJob,
                    /*bLazyLoadDone*/ true,
                    /*bAllowFuncionDecls*/ false))
    return nullptr;

  if (!bIsLib) {
    std::pair<DxilFunctionLinkInfo *, DxilLib *> &entryLinkPair =
        m_functionNameMap[entry];

    return linkJob.Link(entryLinkPair, pSM);
  } else {
    return linkJob.LinkToLib(pSM);
  }
}

namespace hlsl {

DxilLinker *DxilLinker::CreateLinker(LLVMContext &Ctx, unsigned valMajor, unsigned valMinor) {
  return new DxilLinkerImpl(Ctx, valMajor, valMinor);
}
} // namespace hlsl
