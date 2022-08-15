///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// HLModule.cpp                                                              //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// HighLevel DX IR module.                                                   //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/DXIL/DxilOperations.h"
#include "dxc/DXIL/DxilShaderModel.h"
#include "dxc/DXIL/DxilCBuffer.h"
#include "dxc/HLSL/HLModule.h"
#include "dxc/DXIL/DxilTypeSystem.h"
#include "dxc/DXIL/DxilUtil.h"
#include "dxc/Support/WinAdapter.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/Analysis/ValueTracking.h"

using namespace llvm;
using std::string;
using std::vector;
using std::unique_ptr;

namespace hlsl {

// Avoid dependency on HLModule from llvm::Module using this:
void HLModule_RemoveGlobal(llvm::Module* M, llvm::GlobalObject* G) {
  if (M && G && M->HasHLModule()) {
    if (llvm::GlobalVariable *GV = dyn_cast<llvm::GlobalVariable>(G))
      M->GetHLModule().RemoveGlobal(GV);
    else if (llvm::Function *F = dyn_cast<llvm::Function>(G))
      M->GetHLModule().RemoveFunction(F);
  }
}
void HLModule_ResetModule(llvm::Module* M) {
  if (M && M->HasHLModule())
    delete &M->GetHLModule();
  M->SetHLModule(nullptr);
}

//------------------------------------------------------------------------------
//
//  HLModule methods.
//
HLModule::HLModule(Module *pModule)
    : m_Ctx(pModule->getContext())
    , m_pModule(pModule)
    , m_pEntryFunc(nullptr)
    , m_EntryName("")
    , m_pMDHelper(llvm::make_unique<DxilMDHelper>(
          pModule, llvm::make_unique<HLExtraPropertyHelper>(pModule)))
    , m_pDebugInfoFinder(nullptr)
    , m_pSM(nullptr)
    , m_DxilMajor(DXIL::kDxilMajor)
    , m_DxilMinor(DXIL::kDxilMinor)
    , m_ValMajor(0)
    , m_ValMinor(0)
    , m_Float32DenormMode(DXIL::Float32DenormMode::Any)
    , m_pOP(llvm::make_unique<OP>(pModule->getContext(), pModule))
    , m_AutoBindingSpace(UINT_MAX)
    , m_DefaultLinkage(DXIL::DefaultLinkage::Default)
    , m_pTypeSystem(llvm::make_unique<DxilTypeSystem>(pModule)) {
  DXASSERT_NOMSG(m_pModule != nullptr);
  m_pModule->pfnRemoveGlobal = &HLModule_RemoveGlobal;
  m_pModule->pfnResetHLModule = &HLModule_ResetModule;

  // Pin LLVM dump methods. TODO: make debug-only.
  void (__thiscall Module::*pfnModuleDump)() const = &Module::dump;
  void (__thiscall Type::*pfnTypeDump)() const = &Type::dump;
  m_pUnused = (char *)&pfnModuleDump - (char *)&pfnTypeDump;
}

HLModule::~HLModule() {
  if (m_pModule->pfnRemoveGlobal == &HLModule_RemoveGlobal)
    m_pModule->pfnRemoveGlobal = nullptr;
}

LLVMContext &HLModule::GetCtx() const { return m_Ctx; }
Module *HLModule::GetModule() const { return m_pModule; }
OP *HLModule::GetOP() const { return m_pOP.get(); }

void HLModule::SetValidatorVersion(unsigned ValMajor, unsigned ValMinor) {
  m_ValMajor = ValMajor;
  m_ValMinor = ValMinor;
}

void HLModule::GetValidatorVersion(unsigned &ValMajor, unsigned &ValMinor) const {
  ValMajor = m_ValMajor;
  ValMinor = m_ValMinor;
}

void HLModule::SetShaderModel(const ShaderModel *pSM) {
  DXASSERT(m_pSM == nullptr, "shader model must not change for the module");
  DXASSERT(pSM != nullptr && pSM->IsValidForDxil(), "shader model must be valid");
  m_pSM = pSM;
  m_pSM->GetDxilVersion(m_DxilMajor, m_DxilMinor);
  m_pMDHelper->SetShaderModel(m_pSM);
  m_SerializedRootSignature.clear();
}

const ShaderModel *HLModule::GetShaderModel() const {
  return m_pSM;
}

uint32_t HLOptions::GetHLOptionsRaw() const {
  union Cast {
    Cast(const HLOptions &options) {
      hlOptions = options;
    }
    HLOptions hlOptions;
    uint32_t  rawData;
  };
  static_assert(sizeof(uint32_t) == sizeof(HLOptions),
                "size must match to make sure no undefined bits when cast");
  Cast rawCast(*this);
  return rawCast.rawData;
}
void HLOptions::SetHLOptionsRaw(uint32_t data) {
  union Cast {
    Cast(uint32_t data) {
      rawData = data;
    }
    HLOptions hlOptions;
    uint64_t  rawData;
  };

  Cast rawCast(data);
  *this = rawCast.hlOptions;
}

void HLModule::SetHLOptions(HLOptions &opts) {
  m_Options = opts;
}

const HLOptions &HLModule::GetHLOptions() const {
  return m_Options;
}

void HLModule::SetAutoBindingSpace(uint32_t Space) {
  m_AutoBindingSpace = Space;
}
uint32_t HLModule::GetAutoBindingSpace() const {
  return m_AutoBindingSpace;
}

Function *HLModule::GetEntryFunction() const {
  return m_pEntryFunc;
}

Function *HLModule::GetPatchConstantFunction() {
  if (!m_pSM->IsHS())
    return nullptr;
  if (!m_pEntryFunc)
    return nullptr;
  DxilFunctionProps &funcProps = GetDxilFunctionProps(m_pEntryFunc);
  return funcProps.ShaderProps.HS.patchConstantFunc;
}

void HLModule::SetEntryFunction(Function *pEntryFunc) {
  m_pEntryFunc = pEntryFunc;
}

const string &HLModule::GetEntryFunctionName() const { return m_EntryName; }
void HLModule::SetEntryFunctionName(const string &name) { m_EntryName = name; }

template<typename T> unsigned 
HLModule::AddResource(vector<unique_ptr<T> > &Vec, unique_ptr<T> pRes) {
  DXASSERT_NOMSG((unsigned)Vec.size() < UINT_MAX);
  unsigned Id = (unsigned)Vec.size();
  Vec.emplace_back(std::move(pRes));
  return Id;
}

unsigned HLModule::AddCBuffer(unique_ptr<DxilCBuffer> pCBuffer) {
  return AddResource<DxilCBuffer>(m_CBuffers, std::move(pCBuffer));
}

DxilCBuffer &HLModule::GetCBuffer(unsigned idx) {
  return *m_CBuffers[idx];
}

const DxilCBuffer &HLModule::GetCBuffer(unsigned idx) const {
  return *m_CBuffers[idx];
}

const vector<unique_ptr<DxilCBuffer> > &HLModule::GetCBuffers() const {
  return m_CBuffers;
}

unsigned HLModule::AddSampler(unique_ptr<DxilSampler> pSampler) {
  return AddResource<DxilSampler>(m_Samplers, std::move(pSampler));
}

DxilSampler &HLModule::GetSampler(unsigned idx) {
  return *m_Samplers[idx];
}

const DxilSampler &HLModule::GetSampler(unsigned idx) const {
  return *m_Samplers[idx];
}

const vector<unique_ptr<DxilSampler> > &HLModule::GetSamplers() const {
  return m_Samplers;
}

unsigned HLModule::AddSRV(unique_ptr<HLResource> pSRV) {
  return AddResource<HLResource>(m_SRVs, std::move(pSRV));
}

HLResource &HLModule::GetSRV(unsigned idx) {
  return *m_SRVs[idx];
}

const HLResource &HLModule::GetSRV(unsigned idx) const {
  return *m_SRVs[idx];
}

const vector<unique_ptr<HLResource> > &HLModule::GetSRVs() const {
  return m_SRVs;
}

unsigned HLModule::AddUAV(unique_ptr<HLResource> pUAV) {
  return AddResource<HLResource>(m_UAVs, std::move(pUAV));
}

HLResource &HLModule::GetUAV(unsigned idx) {
  return *m_UAVs[idx];
}

const HLResource &HLModule::GetUAV(unsigned idx) const {
  return *m_UAVs[idx];
}

const vector<unique_ptr<HLResource> > &HLModule::GetUAVs() const {
  return m_UAVs;
}

void HLModule::RemoveFunction(llvm::Function *F) {
  DXASSERT_NOMSG(F != nullptr);
  m_DxilFunctionPropsMap.erase(F);
  if (m_pTypeSystem.get()->GetFunctionAnnotation(F))
    m_pTypeSystem.get()->EraseFunctionAnnotation(F);
  m_pOP->RemoveFunction(F);
}

namespace {
  template <typename TResource>
  bool RemoveResource(std::vector<std::unique_ptr<TResource>> &vec,
    GlobalVariable *pVariable, bool keepAllocated) {
    for (auto p = vec.begin(), e = vec.end(); p != e; ++p) {
      if ((*p)->GetGlobalSymbol() != pVariable)
        continue;

      if (keepAllocated && (*p)->IsAllocated()) {
        // Keep the resource, but it has no more symbol.
        (*p)->SetGlobalSymbol(UndefValue::get(pVariable->getType()));
      } else {
        // Erase the resource alltogether and update IDs of subsequent ones
        p = vec.erase(p);
        for (e = vec.end(); p != e; ++p) {
          unsigned ID = (*p)->GetID() - 1;
          (*p)->SetID(ID);
        }
      }

      return true;
    }
    return false;
  }
}

void HLModule::RemoveGlobal(llvm::GlobalVariable *GV) {
  DXASSERT_NOMSG(GV != nullptr);

  // With legacy resource reservation, we must keep unused resources around
  // when they have a register allocation because they prevent that
  // register range from being allocated to other resources.
  bool keepAllocated = GetHLOptions().bLegacyResourceReservation;

  // This could be considerably faster - check variable type to see which
  // resource type this is rather than scanning all lists, and look for
  // usage and removal patterns.
  if (RemoveResource(m_CBuffers, GV, keepAllocated))
    return;
  if (RemoveResource(m_SRVs, GV, keepAllocated))
    return;
  if (RemoveResource(m_UAVs, GV, keepAllocated))
    return;
  if (RemoveResource(m_Samplers, GV, keepAllocated))
    return;
  // TODO: do m_TGSMVariables and m_StreamOutputs need maintenance?
}

HLModule::tgsm_iterator HLModule::tgsm_begin() {
  return m_TGSMVariables.begin();
}

HLModule::tgsm_iterator HLModule::tgsm_end() {
  return m_TGSMVariables.end();
}

void HLModule::AddGroupSharedVariable(GlobalVariable *GV) {
  m_TGSMVariables.emplace_back(GV);
}

std::vector<uint8_t> &HLModule::GetSerializedRootSignature() {
  return m_SerializedRootSignature;
}

void HLModule::SetSerializedRootSignature(const uint8_t *pData, unsigned size) {
  m_SerializedRootSignature.assign(pData, pData+size);
}

DxilTypeSystem &HLModule::GetTypeSystem() {
  return *m_pTypeSystem;
}

DxilTypeSystem *HLModule::ReleaseTypeSystem() {
  return m_pTypeSystem.release();
}

hlsl::OP *HLModule::ReleaseOP() {
  return m_pOP.release();
}

DxilFunctionPropsMap &&HLModule::ReleaseFunctionPropsMap() {
  return std::move(m_DxilFunctionPropsMap);
}

void HLModule::EmitLLVMUsed() {
  if (m_LLVMUsed.empty())
    return;

  vector<llvm::Constant*> GVs;
  GVs.resize(m_LLVMUsed.size());
  for (size_t i = 0, e = m_LLVMUsed.size(); i != e; i++) {
    GVs[i] = ConstantExpr::getAddrSpaceCast(cast<llvm::Constant>(&*m_LLVMUsed[i]), Type::getInt8PtrTy(m_Ctx));
  }

  ArrayType *pATy = ArrayType::get(Type::getInt8PtrTy(m_Ctx), GVs.size());

  GlobalVariable *pGV = new GlobalVariable(*m_pModule, pATy, false,
                                           GlobalValue::AppendingLinkage,
                                           ConstantArray::get(pATy, GVs),
                                           "llvm.used");

  pGV->setSection("llvm.metadata");
}

vector<GlobalVariable* > &HLModule::GetLLVMUsed() {
  return m_LLVMUsed;
}

bool HLModule::HasDxilFunctionProps(llvm::Function *F) {
  return m_DxilFunctionPropsMap.find(F) != m_DxilFunctionPropsMap.end();
}
DxilFunctionProps &HLModule::GetDxilFunctionProps(llvm::Function *F)  {
  DXASSERT(m_DxilFunctionPropsMap.count(F) != 0, "cannot find F in map");
  return *m_DxilFunctionPropsMap[F];
}
void HLModule::AddDxilFunctionProps(llvm::Function *F, std::unique_ptr<DxilFunctionProps> &info) {
  DXASSERT(m_DxilFunctionPropsMap.count(F) == 0, "F already in map, info will be overwritten");
  DXASSERT_NOMSG(info->shaderKind != DXIL::ShaderKind::Invalid);
  m_DxilFunctionPropsMap[F] = std::move(info);
}
void HLModule::SetPatchConstantFunctionForHS(llvm::Function *hullShaderFunc, llvm::Function *patchConstantFunc) {
  auto propIter = m_DxilFunctionPropsMap.find(hullShaderFunc);
  DXASSERT(propIter != m_DxilFunctionPropsMap.end(), "else Hull Shader missing function props");
  DxilFunctionProps &props = *(propIter->second);
  DXASSERT(props.IsHS(), "else hullShaderFunc is not a Hull Shader");
  if (props.ShaderProps.HS.patchConstantFunc)
    m_PatchConstantFunctions.erase(props.ShaderProps.HS.patchConstantFunc);
  props.ShaderProps.HS.patchConstantFunc = patchConstantFunc;
  if (patchConstantFunc)
    m_PatchConstantFunctions.insert(patchConstantFunc);
}
bool HLModule::IsGraphicsShader(llvm::Function *F) {
  return HasDxilFunctionProps(F) && GetDxilFunctionProps(F).IsGraphics();
}
bool HLModule::IsPatchConstantShader(llvm::Function *F) {
  return m_PatchConstantFunctions.count(F) != 0;
}
bool HLModule::IsComputeShader(llvm::Function *F) {
  return HasDxilFunctionProps(F) && GetDxilFunctionProps(F).IsCS();
}
bool HLModule::IsEntryThatUsesSignatures(llvm::Function *F) {
  auto propIter = m_DxilFunctionPropsMap.find(F);
  if (propIter != m_DxilFunctionPropsMap.end()) {
    DxilFunctionProps &props = *(propIter->second);
    return props.IsGraphics() || props.IsCS();
  }
  // Otherwise, return true if patch constant function
  return IsPatchConstantShader(F);
}
bool HLModule::IsEntry(llvm::Function *F) {
  auto propIter = m_DxilFunctionPropsMap.find(F);
  if (propIter != m_DxilFunctionPropsMap.end()) {
    DXASSERT(propIter->second->shaderKind != DXIL::ShaderKind::Invalid,
             "invalid entry props");
    return true;
  }
  // Otherwise, return true if patch constant function
  return IsPatchConstantShader(F);
}

DxilFunctionAnnotation *HLModule::GetFunctionAnnotation(llvm::Function *F) {
  return m_pTypeSystem->GetFunctionAnnotation(F);
}
DxilFunctionAnnotation *HLModule::AddFunctionAnnotation(llvm::Function *F) {
  DXASSERT(m_pTypeSystem->GetFunctionAnnotation(F)==nullptr, "function annotation already exist");
  return m_pTypeSystem->AddFunctionAnnotation(F);
}

DXIL::Float32DenormMode HLModule::GetFloat32DenormMode() const {
  return m_Float32DenormMode;
}

void HLModule::SetFloat32DenormMode(const DXIL::Float32DenormMode mode) {
  m_Float32DenormMode = mode;
}

DXIL::DefaultLinkage HLModule::GetDefaultLinkage() const {
  return m_DefaultLinkage;
}

void HLModule::SetDefaultLinkage(const DXIL::DefaultLinkage linkage) {
  m_DefaultLinkage = linkage;
}

static const StringRef kHLDxilFunctionPropertiesMDName           = "dx.fnprops";
static const StringRef kHLDxilOptionsMDName                      = "dx.options";

// DXIL metadata serialization/deserialization.
void HLModule::EmitHLMetadata() {
  m_pMDHelper->EmitDxilVersion(m_DxilMajor, m_DxilMinor);
  m_pMDHelper->EmitValidatorVersion(m_ValMajor, m_ValMinor);
  m_pMDHelper->EmitDxilShaderModel(m_pSM);

  MDTuple *pMDResources = EmitHLResources();
  MDTuple *pMDProperties = EmitHLShaderProperties();

  m_pMDHelper->EmitDxilTypeSystem(GetTypeSystem(), m_LLVMUsed);
  EmitLLVMUsed();
  MDTuple *const pNullMDSig = nullptr;
  MDTuple *pEntry = m_pMDHelper->EmitDxilEntryPointTuple(GetEntryFunction(), m_EntryName, pNullMDSig, pMDResources, pMDProperties);
  vector<MDNode *> Entries;
  Entries.emplace_back(pEntry);
  m_pMDHelper->EmitDxilEntryPoints(Entries);

  {
    NamedMDNode * fnProps = m_pModule->getOrInsertNamedMetadata(kHLDxilFunctionPropertiesMDName);
    for (auto && pair : m_DxilFunctionPropsMap) {
      const hlsl::DxilFunctionProps * props = pair.second.get();
      MDTuple *pProps = m_pMDHelper->EmitDxilFunctionProps(props, pair.first);
      fnProps->addOperand(pProps);
    }

    NamedMDNode * options = m_pModule->getOrInsertNamedMetadata(kHLDxilOptionsMDName);
    uint32_t hlOptions = m_Options.GetHLOptionsRaw();
    options->addOperand(MDNode::get(m_Ctx, m_pMDHelper->Uint32ToConstMD(hlOptions)));
    options->addOperand(MDNode::get(m_Ctx, m_pMDHelper->Uint32ToConstMD(GetAutoBindingSpace())));
  }

  if (!m_SerializedRootSignature.empty()) {
    m_pMDHelper->EmitRootSignature(m_SerializedRootSignature);
  }

  // Save Subobjects
  if (GetSubobjects()) {
    m_pMDHelper->EmitSubobjects(*GetSubobjects());
  }
}

void HLModule::LoadHLMetadata() {
  m_pMDHelper->LoadDxilVersion(m_DxilMajor, m_DxilMinor);
  m_pMDHelper->LoadValidatorVersion(m_ValMajor, m_ValMinor);
  m_pMDHelper->LoadDxilShaderModel(m_pSM);
  m_SerializedRootSignature.clear();

  const llvm::NamedMDNode *pEntries = m_pMDHelper->GetDxilEntryPoints();

  Function *pEntryFunc;
  string EntryName;
  const llvm::MDOperand *pSignatures, *pResources, *pProperties;
  m_pMDHelper->GetDxilEntryPoint(pEntries->getOperand(0), pEntryFunc, EntryName, pSignatures, pResources, pProperties);

  SetEntryFunction(pEntryFunc);
  SetEntryFunctionName(EntryName);

  LoadHLResources(*pResources);
  LoadHLShaderProperties(*pProperties);

  m_pMDHelper->LoadDxilTypeSystem(*m_pTypeSystem.get());

  {
    NamedMDNode * fnProps = m_pModule->getNamedMetadata(kHLDxilFunctionPropertiesMDName);
    size_t propIdx = 0;
    while (propIdx < fnProps->getNumOperands()) {
      MDTuple *pProps = dyn_cast<MDTuple>(fnProps->getOperand(propIdx++));

      std::unique_ptr<hlsl::DxilFunctionProps> props =
          llvm::make_unique<hlsl::DxilFunctionProps>();

      const Function *F = m_pMDHelper->LoadDxilFunctionProps(pProps, props.get());

      if (props->IsHS() && props->ShaderProps.HS.patchConstantFunc) {
        // Add patch constant function to m_PatchConstantFunctions
        m_PatchConstantFunctions.insert(props->ShaderProps.HS.patchConstantFunc);
      }

      m_DxilFunctionPropsMap[F] = std::move(props);
    }

    const NamedMDNode * options = m_pModule->getOrInsertNamedMetadata(kHLDxilOptionsMDName);
    const MDNode *MDOptions = options->getOperand(0);
    m_Options.SetHLOptionsRaw(DxilMDHelper::ConstMDToUint32(MDOptions->getOperand(0)));
    if (options->getNumOperands() > 1)
      SetAutoBindingSpace(DxilMDHelper::ConstMDToUint32(options->getOperand(1)->getOperand(0)));
  }

  m_pMDHelper->LoadRootSignature(m_SerializedRootSignature);

  // Load Subobjects
  std::unique_ptr<DxilSubobjects> pSubobjects(new DxilSubobjects());
  m_pMDHelper->LoadSubobjects(*pSubobjects);
  if (pSubobjects->GetSubobjects().size()) {
    ResetSubobjects(pSubobjects.release());
  }
}

void HLModule::ClearHLMetadata(llvm::Module &M) {
  Module::named_metadata_iterator
    b = M.named_metadata_begin(),
    e = M.named_metadata_end();
  SmallVector<NamedMDNode*, 8> nodes;
  for (; b != e; ++b) {
    StringRef name = b->getName();
    if (name == DxilMDHelper::kDxilVersionMDName ||
        name == DxilMDHelper::kDxilShaderModelMDName ||
        name == DxilMDHelper::kDxilEntryPointsMDName ||
        name == DxilMDHelper::kDxilRootSignatureMDName ||
        name == DxilMDHelper::kDxilResourcesMDName ||
        name == DxilMDHelper::kDxilTypeSystemMDName ||
        name == DxilMDHelper::kDxilValidatorVersionMDName ||
        name == kHLDxilFunctionPropertiesMDName || // TODO: adjust to proper name
        name == kHLDxilOptionsMDName ||
        name.startswith(DxilMDHelper::kDxilTypeSystemHelperVariablePrefix)) {
      nodes.push_back(b);
    }
  }
  for (size_t i = 0; i < nodes.size(); ++i) {
    M.eraseNamedMetadata(nodes[i]);
  }
}

MDTuple *HLModule::EmitHLResources() {
  // Emit SRV records.
  MDTuple *pTupleSRVs = nullptr;
  if (!m_SRVs.empty()) {
    vector<Metadata *> MDVals;
    for (size_t i = 0; i < m_SRVs.size(); i++) {
      MDVals.emplace_back(m_pMDHelper->EmitDxilSRV(*m_SRVs[i]));
    }
    pTupleSRVs = MDNode::get(m_Ctx, MDVals);
  }

  // Emit UAV records.
  MDTuple *pTupleUAVs = nullptr;
  if (!m_UAVs.empty()) {
    vector<Metadata *> MDVals;
    for (size_t i = 0; i < m_UAVs.size(); i++) {
      MDVals.emplace_back(m_pMDHelper->EmitDxilUAV(*m_UAVs[i]));
    }
    pTupleUAVs = MDNode::get(m_Ctx, MDVals);
  }

  // Emit CBuffer records.
  MDTuple *pTupleCBuffers = nullptr;
  if (!m_CBuffers.empty()) {
    vector<Metadata *> MDVals;
    for (size_t i = 0; i < m_CBuffers.size(); i++) {
      MDVals.emplace_back(m_pMDHelper->EmitDxilCBuffer(*m_CBuffers[i]));
    }
    pTupleCBuffers = MDNode::get(m_Ctx, MDVals);
  }

  // Emit Sampler records.
  MDTuple *pTupleSamplers = nullptr;
  if (!m_Samplers.empty()) {
    vector<Metadata *> MDVals;
    for (size_t i = 0; i < m_Samplers.size(); i++) {
      MDVals.emplace_back(m_pMDHelper->EmitDxilSampler(*m_Samplers[i]));
    }
    pTupleSamplers = MDNode::get(m_Ctx, MDVals);
  }

  if (pTupleSRVs != nullptr || pTupleUAVs != nullptr || pTupleCBuffers != nullptr || pTupleSamplers != nullptr) {
    return m_pMDHelper->EmitDxilResourceTuple(pTupleSRVs, pTupleUAVs, pTupleCBuffers, pTupleSamplers);
  } else {
    return nullptr;
  }
}

void HLModule::LoadHLResources(const llvm::MDOperand &MDO) {
  const llvm::MDTuple *pSRVs, *pUAVs, *pCBuffers, *pSamplers;
  m_pMDHelper->GetDxilResources(MDO, pSRVs, pUAVs, pCBuffers, pSamplers);

  // Load SRV records.
  if (pSRVs != nullptr) {
    for (unsigned i = 0; i < pSRVs->getNumOperands(); i++) {
      unique_ptr<HLResource> pSRV(new HLResource);
      m_pMDHelper->LoadDxilSRV(pSRVs->getOperand(i), *pSRV);
      AddSRV(std::move(pSRV));
    }
  }

  // Load UAV records.
  if (pUAVs != nullptr) {
    for (unsigned i = 0; i < pUAVs->getNumOperands(); i++) {
      unique_ptr<HLResource> pUAV(new HLResource);
      m_pMDHelper->LoadDxilUAV(pUAVs->getOperand(i), *pUAV);
      AddUAV(std::move(pUAV));
    }
  }

  // Load CBuffer records.
  if (pCBuffers != nullptr) {
    for (unsigned i = 0; i < pCBuffers->getNumOperands(); i++) {
      unique_ptr<DxilCBuffer> pCB = llvm::make_unique<DxilCBuffer>();
      m_pMDHelper->LoadDxilCBuffer(pCBuffers->getOperand(i), *pCB);
      AddCBuffer(std::move(pCB));
    }
  }

  // Load Sampler records.
  if (pSamplers != nullptr) {
    for (unsigned i = 0; i < pSamplers->getNumOperands(); i++) {
      unique_ptr<DxilSampler> pSampler(new DxilSampler);
      m_pMDHelper->LoadDxilSampler(pSamplers->getOperand(i), *pSampler);
      AddSampler(std::move(pSampler));
    }
  }
}

MDTuple *HLModule::EmitHLShaderProperties() {
  return nullptr;
}

void HLModule::LoadHLShaderProperties(const MDOperand &MDO) {
  return;
}

MDNode *HLModule::DxilSamplerToMDNode(const DxilSampler &S) {
  MDNode *MD = m_pMDHelper->EmitDxilSampler(S);
  ValueAsMetadata *ResClass =
      m_pMDHelper->Uint32ToConstMD((unsigned)DXIL::ResourceClass::Sampler);

  return MDNode::get(m_Ctx, {ResClass, MD});
}
MDNode *HLModule::DxilSRVToMDNode(const DxilResource &SRV) {
  MDNode *MD = m_pMDHelper->EmitDxilSRV(SRV);
  ValueAsMetadata *ResClass =
      m_pMDHelper->Uint32ToConstMD((unsigned)DXIL::ResourceClass::SRV);

  return MDNode::get(m_Ctx, {ResClass, MD});
}
MDNode *HLModule::DxilUAVToMDNode(const DxilResource &UAV) {
  MDNode *MD = m_pMDHelper->EmitDxilUAV(UAV);
  ValueAsMetadata *ResClass =
      m_pMDHelper->Uint32ToConstMD((unsigned)DXIL::ResourceClass::UAV);

  return MDNode::get(m_Ctx, {ResClass, MD});
}
MDNode *HLModule::DxilCBufferToMDNode(const DxilCBuffer &CB) {
  MDNode *MD = m_pMDHelper->EmitDxilCBuffer(CB);
  ValueAsMetadata *ResClass =
      m_pMDHelper->Uint32ToConstMD((unsigned)DXIL::ResourceClass::CBuffer);

  return MDNode::get(m_Ctx, {ResClass, MD});
}

void HLModule::LoadDxilResourceBaseFromMDNode(MDNode *MD, DxilResourceBase &R) {
  return m_pMDHelper->LoadDxilResourceBaseFromMDNode(MD, R);
}
void HLModule::LoadDxilResourceFromMDNode(llvm::MDNode *MD, DxilResource &R) {
  return m_pMDHelper->LoadDxilResourceFromMDNode(MD, R);
}
void HLModule::LoadDxilSamplerFromMDNode(llvm::MDNode *MD, DxilSampler &S) {
  return m_pMDHelper->LoadDxilSamplerFromMDNode(MD, S);
}

DxilResourceBase *
HLModule::AddResourceWithGlobalVariableAndProps(llvm::Constant *GV,
                                                 DxilResourceProperties &RP) {
  DxilResource::Class RC = RP.getResourceClass();
  DxilResource::Kind RK = RP.getResourceKind();
  unsigned rangeSize = 1;
  Type *Ty = GV->getType()->getPointerElementType();
  if (ArrayType *AT = dyn_cast<ArrayType>(Ty))
    rangeSize = AT->getNumElements();
  DxilResourceBase *R = nullptr;
  switch (RC) {
  case DxilResource::Class::Sampler: {
    std::unique_ptr<DxilSampler> S = llvm::make_unique<DxilSampler>();
    if (RP.Basic.SamplerCmpOrHasCounter)
      S->SetSamplerKind(DxilSampler::SamplerKind::Comparison);
    else
      S->SetSamplerKind(DxilSampler::SamplerKind::Default);
    S->SetKind(RK);
    S->SetGlobalSymbol(GV);
    S->SetGlobalName(GV->getName());
    S->SetRangeSize(rangeSize);
    R = S.get();
    AddSampler(std::move(S));
  } break;
  case DxilResource::Class::SRV: {
    std::unique_ptr<HLResource> Res = llvm::make_unique<HLResource>();
    if (DXIL::IsTyped(RP.getResourceKind())) {
      Res->SetCompType(RP.Typed.CompType);
    } else if (DXIL::IsStructuredBuffer(RK)) {
      Res->SetElementStride(RP.StructStrideInBytes);
    }
    Res->SetRW(false);
    Res->SetKind(RK);
    Res->SetGlobalSymbol(GV);
    Res->SetGlobalName(GV->getName());
    Res->SetRangeSize(rangeSize);
    R = Res.get();
    AddSRV(std::move(Res));
  } break;
  case DxilResource::Class::UAV: {
    std::unique_ptr<HLResource> Res = llvm::make_unique<HLResource>();
    if (DXIL::IsTyped(RK)) {
      Res->SetCompType(RP.Typed.CompType);
    } else if (DXIL::IsStructuredBuffer(RK)) {
      Res->SetElementStride(RP.StructStrideInBytes);
    }

    Res->SetRW(true);
    Res->SetROV(RP.Basic.IsROV);
    Res->SetGloballyCoherent(RP.Basic.IsGloballyCoherent);
    Res->SetHasCounter(RP.Basic.SamplerCmpOrHasCounter);
    Res->SetKind(RK);
    Res->SetGlobalSymbol(GV);
    Res->SetGlobalName(GV->getName());
    Res->SetRangeSize(rangeSize);
    R = Res.get();
    AddUAV(std::move(Res));
  } break;
  default:
    DXASSERT(0, "Invalid metadata for AddResourceWithGlobalVariableAndMDNode");
  }
  return R;
}

static uint64_t getRegBindingKey(unsigned CbID, unsigned ConstantIdx) {
  return (uint64_t)(CbID) << 32 | ConstantIdx;
}

void HLModule::AddRegBinding(unsigned CbID, unsigned ConstantIdx, unsigned Srv, unsigned Uav,
                             unsigned Sampler) {
  uint64_t Key = getRegBindingKey(CbID, ConstantIdx);
  m_SrvBindingInCB[Key] = Srv;
  m_UavBindingInCB[Key] = Uav;
  m_SamplerBindingInCB[Key] = Sampler;
}

// Helper functions for resource in cbuffer.
namespace {

DXIL::ResourceClass GetRCFromType(StructType *ST, Module &M) {
  for (Function &F : M.functions()) {
    if (F.user_empty())
      continue;
    hlsl::HLOpcodeGroup group = hlsl::GetHLOpcodeGroup(&F);
    if (group != HLOpcodeGroup::HLAnnotateHandle)
      continue;
    Type *Ty = F.getFunctionType()->getParamType(
        HLOperandIndex::kAnnotateHandleResourceTypeOpIdx);
    if (Ty != ST)
      continue;
    CallInst *CI = cast<CallInst>(F.user_back());
    Constant *Props = cast<Constant>(CI->getArgOperand(
        HLOperandIndex::kAnnotateHandleResourcePropertiesOpIdx));
    DxilResourceProperties RP = resource_helper::loadPropsFromConstant(*Props);
    return RP.getResourceClass();
  }
  return DXIL::ResourceClass::Invalid;
}

unsigned CountResNum(Module &M, Type *Ty, DXIL::ResourceClass RC) {
  // Count num of RCs.
  unsigned ArraySize = 1;
  while (ArrayType *AT = dyn_cast<ArrayType>(Ty)) {
    ArraySize *= AT->getNumElements();
    Ty = AT->getElementType();
  }

  if (!Ty->isAggregateType())
    return 0;

  StructType *ST = dyn_cast<StructType>(Ty);
  DXIL::ResourceClass TmpRC = GetRCFromType(ST, M);
  if (TmpRC == RC)
    return ArraySize;

  unsigned Size = 0;
  for (Type *EltTy : ST->elements()) {
    Size += CountResNum(M, EltTy, RC);
  }

  return Size * ArraySize;
}
// Note: the rule for register binding on struct array is like this:
// struct X {
//   Texture2D x;
//   SamplerState s ;
//   Texture2D y;
// };
// X x[2] : register(t3) : register(s3);
// x[0].x t3
// x[0].s s3
// x[0].y t4
// x[1].x t5
// x[1].s s4
// x[1].y t6
// So x[0].x and x[1].x not in an array.
unsigned CalcRegBinding(gep_type_iterator GEPIt, gep_type_iterator E,
                               Module &M, DXIL::ResourceClass RC) {
  unsigned NumRC = 0;
  // Count GEP offset when only count RC size.
  for (; GEPIt != E; GEPIt++) {
    Type *Ty = *GEPIt;
    Value *idx = GEPIt.getOperand();
    Constant *constIdx = dyn_cast<Constant>(idx);
    unsigned immIdx = constIdx->getUniqueInteger().getLimitedValue();
    // Not support dynamic indexing.
    // Array should be just 1d res array as global res.
    if (ArrayType *AT = dyn_cast<ArrayType>(Ty)) {
      NumRC += immIdx * CountResNum(M, AT->getElementType(), RC);
    } else if (StructType *ST = dyn_cast<StructType>(Ty)) {
      for (unsigned i=0;i<immIdx;i++) {
        NumRC += CountResNum(M, ST->getElementType(i), RC);
      }
    }
  }
  return NumRC;
}
} // namespace

unsigned HLModule::GetBindingForResourceInCB(GetElementPtrInst *CbPtr,
                                             GlobalVariable *CbGV,
                                             DXIL::ResourceClass RC) {
  if (!CbPtr->hasAllConstantIndices()) {
    // Not support dynmaic indexing resource array inside cb.
    string ErrorMsg("Index for resource array inside cbuffer must be a literal expression");
    dxilutil::EmitErrorOnInstruction(
        CbPtr,
        ErrorMsg);
    return UINT_MAX;
  }
  Module &M = *m_pModule;

  unsigned RegBinding = UINT_MAX;
  for (auto &CB : m_CBuffers) {
    if (CbGV != CB->GetGlobalSymbol())
      continue;

    gep_type_iterator GEPIt = gep_type_begin(CbPtr), E = gep_type_end(CbPtr);
    // The pointer index.
    GEPIt++;
    unsigned ID = CB->GetID();
    unsigned idx = cast<ConstantInt>(GEPIt.getOperand())->getLimitedValue();
    // The first level index to get current constant.
    GEPIt++;

    uint64_t Key = getRegBindingKey(ID, idx);
    switch (RC) {
    default:
      break;
    case DXIL::ResourceClass::SRV:
      if (m_SrvBindingInCB.count(Key))
        RegBinding = m_SrvBindingInCB[Key];
      break;
    case DXIL::ResourceClass::UAV:
      if (m_UavBindingInCB.count(Key))
        RegBinding = m_UavBindingInCB[Key];
      break;
    case DXIL::ResourceClass::Sampler:
      if (m_SamplerBindingInCB.count(Key))
        RegBinding = m_SamplerBindingInCB[Key];
      break;
    }
    if (RegBinding == UINT_MAX)
      break;

    // Calc RegBinding.
    RegBinding += CalcRegBinding(GEPIt, E, M, RC);

    break;
  }
  return RegBinding;
}

// TODO: Don't check names.
bool HLModule::IsStreamOutputType(llvm::Type *Ty) {
  if (StructType *ST = dyn_cast<StructType>(Ty)) {
    StringRef name = ST->getName();
    if (name.startswith("class.PointStream"))
      return true;
    if (name.startswith("class.LineStream"))
      return true;
    if (name.startswith("class.TriangleStream"))
      return true;
  }
  return false;
}

bool HLModule::IsStreamOutputPtrType(llvm::Type *Ty) {
  if (!Ty->isPointerTy())
    return false;
  Ty = Ty->getPointerElementType();
  return IsStreamOutputType(Ty);
}

void HLModule::GetParameterRowsAndCols(Type *Ty, unsigned &rows, unsigned &cols,
                                       DxilParameterAnnotation &paramAnnotation) {
  if (Ty->isPointerTy())
    Ty = Ty->getPointerElementType();
  // For array input of HS, DS, GS,
  // we need to skip the first level which size is based on primitive type.
  DxilParamInputQual inputQual = paramAnnotation.GetParamInputQual();
  bool skipOneLevelArray = inputQual == DxilParamInputQual::InputPatch;
  skipOneLevelArray |= inputQual == DxilParamInputQual::OutputPatch;
  skipOneLevelArray |= inputQual == DxilParamInputQual::InputPrimitive;
  skipOneLevelArray |= inputQual == DxilParamInputQual::OutVertices;
  skipOneLevelArray |= inputQual == DxilParamInputQual::OutPrimitives;

  if (skipOneLevelArray) {
    if (Ty->isArrayTy())
      Ty = Ty->getArrayElementType();
  }

  unsigned arraySize = 1;
  while (Ty->isArrayTy()) {
    arraySize *= Ty->getArrayNumElements();
    Ty = Ty->getArrayElementType();
  }

  rows = 1;
  cols = 1;

  if (paramAnnotation.HasMatrixAnnotation()) {
    const DxilMatrixAnnotation &matrix = paramAnnotation.GetMatrixAnnotation();
    if (matrix.Orientation == MatrixOrientation::RowMajor) {
      rows = matrix.Rows;
      cols = matrix.Cols;
    } else {
      DXASSERT_NOMSG(matrix.Orientation == MatrixOrientation::ColumnMajor);
      cols = matrix.Rows;
      rows = matrix.Cols;
    }
  } else if (FixedVectorType *VT = dyn_cast<FixedVectorType>(Ty))
    cols = VT->getNumElements();

  rows *= arraySize;
}

llvm::Function *HLModule::GetHLOperationFunction(
    HLOpcodeGroup group, unsigned opcode,
                                       llvm::Type *RetType,
                                       llvm::ArrayRef<llvm::Value *> paramList,
                                       llvm::Module &M) {
  SmallVector<llvm::Type *, 4> paramTyList;
  // Add the opcode param
  llvm::Type *opcodeTy = llvm::Type::getInt32Ty(M.getContext());
  paramTyList.emplace_back(opcodeTy);
  for (Value *param : paramList) {
    paramTyList.emplace_back(param->getType());
  }

  llvm::FunctionType *funcTy =
      llvm::FunctionType::get(RetType, paramTyList, false);

  Function *opFunc = GetOrCreateHLFunction(M, funcTy, group, opcode);
  return opFunc;
}

template CallInst *HLModule::EmitHLOperationCall(IRBuilder<> &Builder,
                                                 HLOpcodeGroup group,
                                                 unsigned opcode, Type *RetType,
                                                 ArrayRef<Value *> paramList,
                                                 llvm::Module &M);

template<typename BuilderTy>
CallInst *HLModule::EmitHLOperationCall(BuilderTy &Builder,
                                           HLOpcodeGroup group, unsigned opcode,
                                           Type *RetType,
                                           ArrayRef<Value *> paramList,
                                           llvm::Module &M) {
  // Add the opcode param
  llvm::Type *opcodeTy = llvm::Type::getInt32Ty(M.getContext());

  Function *opFunc =
      GetHLOperationFunction(group, opcode, RetType, paramList, M);

  SmallVector<Value *, 4> opcodeParamList;
  Value *opcodeConst = Constant::getIntegerValue(opcodeTy, APInt(32, opcode));
  opcodeParamList.emplace_back(opcodeConst);
  opcodeParamList.append(paramList.begin(), paramList.end());

  return Builder.CreateCall(opFunc, opcodeParamList);
}

unsigned HLModule::GetNumericCastOp(
  llvm::Type *SrcTy, bool SrcIsUnsigned, llvm::Type *DstTy, bool DstIsUnsigned) {
  DXASSERT(SrcTy != DstTy, "No-op conversions are not casts and should have been handled by the callee.");
  uint32_t SrcBitSize = SrcTy->getScalarSizeInBits();
  uint32_t DstBitSize = DstTy->getScalarSizeInBits();
  bool SrcIsInt = SrcTy->isIntOrIntVectorTy();
  bool DstIsInt = DstTy->isIntOrIntVectorTy();

  DXASSERT(DstBitSize != 1, "Conversions to bool are not a cast and should have been handled by the callee.");

  // Conversions from bools are like unsigned integer widening
  if (SrcBitSize == 1) SrcIsUnsigned = true;

  if (SrcIsInt) {
    if (DstIsInt) { // int to int
      if (SrcBitSize > DstBitSize) return Instruction::Trunc;
      // unsigned to unsigned: zext
      // unsigned to signed: zext (fully representable)
      // signed to signed: sext
      // signed to unsigned: sext (like C++)
      return SrcIsUnsigned ? Instruction::ZExt : Instruction::SExt;
    }
    else { // int to float
      return SrcIsUnsigned ? Instruction::UIToFP : Instruction::SIToFP;
    }
  }
  else {
    if (DstIsInt) { // float to int
      return DstIsUnsigned ? Instruction::FPToUI : Instruction::FPToSI;
    }
    else { // float to float
      return SrcBitSize > DstBitSize ? Instruction::FPTrunc : Instruction::FPExt;
    }
  }
}

bool HLModule::HasPreciseAttributeWithMetadata(Instruction *I) {
  return DxilMDHelper::IsMarkedPrecise(I);
}

void HLModule::MarkPreciseAttributeWithMetadata(Instruction *I) {
  return DxilMDHelper::MarkPrecise(I);
}

void HLModule::ClearPreciseAttributeWithMetadata(Instruction *I) {
  I->setMetadata(DxilMDHelper::kDxilPreciseAttributeMDName, nullptr);
}

static void MarkPreciseAttribute(Function *F) {
  LLVMContext &Ctx = F->getContext();
  MDNode *preciseNode = MDNode::get(
      Ctx, {MDString::get(Ctx, DxilMDHelper::kDxilPreciseAttributeMDName)});

  F->setMetadata(DxilMDHelper::kDxilPreciseAttributeMDName, preciseNode);
}

template<typename BuilderTy>
void HLModule::MarkPreciseAttributeOnValWithFunctionCall(
    llvm::Value *V, BuilderTy &Builder, llvm::Module &M) {
  Type *Ty = V->getType();
  Type *EltTy = Ty->getScalarType();

  // TODO: Only do this on basic types.
  
  FunctionType *preciseFuncTy =
      FunctionType::get(Type::getVoidTy(M.getContext()), {EltTy}, false);
  // The function will be deleted after precise propagate.
  std::string preciseFuncName = "dx.attribute.precise.";
  raw_string_ostream mangledNameStr(preciseFuncName);
  EltTy->print(mangledNameStr);
  mangledNameStr.flush();

  Function *preciseFunc =
      cast<Function>(M.getOrInsertFunction(preciseFuncName, preciseFuncTy));
  if (!HLModule::HasPreciseAttribute(preciseFunc))
    MarkPreciseAttribute(preciseFunc);
  if (FixedVectorType *VT = dyn_cast<FixedVectorType>(Ty)) {
    for (unsigned i = 0; i < VT->getNumElements(); i++) {
      Value *Elt = Builder.CreateExtractElement(V, i);
      Builder.CreateCall(preciseFunc, {Elt});
    }
  } else
    Builder.CreateCall(preciseFunc, {V});
}

void HLModule::MarkPreciseAttributeOnPtrWithFunctionCall(llvm::Value *Ptr,
                                               llvm::Module &M) {
  for (User *U : Ptr->users()) {
    // Skip load inst.
    if (dyn_cast<LoadInst>(U))
      continue;
    if (StoreInst *SI = dyn_cast<StoreInst>(U)) {
      Value *V = SI->getValueOperand();
      if (isa<Instruction>(V)) {
        // Mark the Value with function call.
        IRBuilder<> Builder(SI);
        MarkPreciseAttributeOnValWithFunctionCall(V, Builder, M);
      }
    } else if (CallInst *CI = dyn_cast<CallInst>(U)) {
      if (CI->getType()->isVoidTy()) {
        IRBuilder<> Builder(CI);
        // For void type, cannot use as function arg.
        // Mark all arg for it?
        for (auto &arg : CI->arg_operands()) {
          MarkPreciseAttributeOnValWithFunctionCall(arg, Builder, M);
        }
      } else {
        if (CI->getType()->isPointerTy()) {
          // For instance, matrix subscript...
          MarkPreciseAttributeOnPtrWithFunctionCall(CI, M);
        } else {
          IRBuilder<> Builder(CI->getNextNode());
          MarkPreciseAttributeOnValWithFunctionCall(CI, Builder, M);
        }
      }
    } else if (BitCastInst *BCI = dyn_cast<BitCastInst>(U)) {
      // Do not mark bitcasts. We only expect them here due to lifetime intrinsics.
      DXASSERT(onlyUsedByLifetimeMarkers(BCI),
               "expected bitcast to only be used by lifetime intrinsics");
    } else {
      // Must be GEP here.
      GetElementPtrInst *GEP = cast<GetElementPtrInst>(U);
      MarkPreciseAttributeOnPtrWithFunctionCall(GEP, M);
    }
  }
}

bool HLModule::HasPreciseAttribute(Function *F) {
  MDNode *preciseNode =
      F->getMetadata(DxilMDHelper::kDxilPreciseAttributeMDName);
  return preciseNode != nullptr;
}

static void AddDIGlobalVariable(DIBuilder &Builder, DIGlobalVariable *LocDIGV,
                                StringRef Name, DIType *DITy,
                                GlobalVariable *GV, DebugInfoFinder &DbgInfoFinder, bool removeLocDIGV) {
  DIGlobalVariable *EltDIGV = Builder.createGlobalVariable(
      LocDIGV->getScope(), Name, GV->getName(), LocDIGV->getFile(),
      LocDIGV->getLine(), DITy, false, GV);

  DICompileUnit *DICU = nullptr;
  std::vector<Metadata *> AllGVs;
  std::vector<Metadata *>::iterator locIt;
  for (auto itDICU : DbgInfoFinder.compile_units()) {
    MDTuple *GTuple = cast_or_null<MDTuple>(itDICU->getRawGlobalVariables());
    if (!GTuple)
      continue;
    AllGVs.assign(GTuple->operands().begin(), GTuple->operands().end());
    locIt = std::find(AllGVs.begin(), AllGVs.end(), LocDIGV);
    if (locIt == AllGVs.end())
      continue;
    DICU = itDICU;
    break;
  }
  DXASSERT_NOMSG(DICU);
  if (!DICU)
    return;

  // Add global to CU.
  if (removeLocDIGV) {
    AllGVs.erase(locIt);
  }
  AllGVs.emplace_back(EltDIGV);
  DICU->replaceGlobalVariables(MDTuple::get(GV->getContext(), AllGVs));
  DXVERIFY_NOMSG(DbgInfoFinder.appendGlobalVariable(EltDIGV));
}

static unsigned GetCompositeTypeSize(DIType *Ty) {
  DICompositeType *StructTy = nullptr;
  DITypeIdentifierMap EmptyMap;

  if (DIDerivedType *DerivedTy = dyn_cast<DIDerivedType>(Ty)) {
    DXASSERT_NOMSG(DerivedTy->getTag() == dwarf::DW_TAG_const_type || DerivedTy->getTag() == dwarf::DW_TAG_typedef);
    DIType *BaseTy = DerivedTy->getBaseType().resolve(EmptyMap);
    return GetCompositeTypeSize(BaseTy);
  }
  else {
    StructTy = cast<DICompositeType>(Ty);
  }

  return StructTy->getSizeInBits();
}

void HLModule::CreateElementGlobalVariableDebugInfo(
    GlobalVariable *GV, DebugInfoFinder &DbgInfoFinder, GlobalVariable *EltGV,
    unsigned sizeInBits, unsigned alignInBits, unsigned offsetInBits,
    StringRef eltName) {
  DIGlobalVariable *DIGV = dxilutil::FindGlobalVariableDebugInfo(GV, DbgInfoFinder);
  if (!DIGV) {
    DXASSERT(DIGV, "DIGV Parameter must be non-null");
    return;
  }
  DIBuilder Builder(*GV->getParent());
  DITypeIdentifierMap EmptyMap;

  DIType *DITy = DIGV->getType().resolve(EmptyMap);
  DIScope *DITyScope = DITy->getScope().resolve(EmptyMap);

  // If element size is greater than base size make sure we're dealing with an empty struct.
  unsigned compositeSize = GetCompositeTypeSize(DITy);
  if (sizeInBits > compositeSize) {
    DXASSERT_NOMSG(offsetInBits == 0 && compositeSize == 8);
    sizeInBits = compositeSize;
  }

  // Create Elt type.
  DIType *EltDITy =
      Builder.createMemberType(DITyScope, DITy->getName().str() + eltName.str(),
                               DITy->getFile(), DITy->getLine(), sizeInBits,
                               alignInBits, offsetInBits, /*Flags*/ 0, DITy);

  AddDIGlobalVariable(Builder, DIGV, DIGV->getName().str() + eltName.str(),
                      EltDITy, EltGV, DbgInfoFinder, /*removeDIGV*/false);
}

void HLModule::UpdateGlobalVariableDebugInfo(
    llvm::GlobalVariable *GV, llvm::DebugInfoFinder &DbgInfoFinder,
    llvm::GlobalVariable *NewGV) {
  DIGlobalVariable *DIGV = dxilutil::FindGlobalVariableDebugInfo(GV, DbgInfoFinder);
  if (!DIGV) {
    DXASSERT(DIGV, "DIGV Parameter must be non-null");
    return;
  }
  DIBuilder Builder(*GV->getParent());
  DITypeIdentifierMap EmptyMap;
  DIType *DITy = DIGV->getType().resolve(EmptyMap);

  AddDIGlobalVariable(Builder, DIGV, DIGV->getName(), DITy, NewGV,
                      DbgInfoFinder,/*removeDIGV*/true);
}

DebugInfoFinder &HLModule::GetOrCreateDebugInfoFinder() {
  if (m_pDebugInfoFinder == nullptr) {
    m_pDebugInfoFinder = llvm::make_unique<llvm::DebugInfoFinder>();
    m_pDebugInfoFinder->processModule(*m_pModule);
  }
  return *m_pDebugInfoFinder;
}

//------------------------------------------------------------------------------
//
// Subobject methods.
//
DxilSubobjects *HLModule::GetSubobjects() {
  return m_pSubobjects.get();
}
const DxilSubobjects *HLModule::GetSubobjects() const {
  return m_pSubobjects.get();
}
DxilSubobjects *HLModule::ReleaseSubobjects() {
  return m_pSubobjects.release();
}
void HLModule::ResetSubobjects(DxilSubobjects *subobjects) {
  m_pSubobjects.reset(subobjects);
}

//------------------------------------------------------------------------------
//
// Signature methods.
//

HLExtraPropertyHelper::HLExtraPropertyHelper(llvm::Module *pModule)
: DxilExtraPropertyHelper(pModule) {
}

void HLExtraPropertyHelper::EmitSignatureElementProperties(const DxilSignatureElement &SE, 
                                                              vector<Metadata *> &MDVals) {
}

void HLExtraPropertyHelper::LoadSignatureElementProperties(const MDOperand &MDO, 
                                                           DxilSignatureElement &SE) {
  if (MDO.get() == nullptr)
    return;
}

} // namespace hlsl

namespace llvm {
hlsl::HLModule &Module::GetOrCreateHLModule(bool skipInit) {
  std::unique_ptr<hlsl::HLModule> M;
  if (!HasHLModule()) {
    M = llvm::make_unique<hlsl::HLModule>(this);
    if (!skipInit) {
      M->LoadHLMetadata();
    }
    SetHLModule(M.release());
  }
  return GetHLModule();
}

}
