///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilModule.cpp                                                            //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/Support/Global.h"
#include "dxc/DXIL/DxilOperations.h"
#include "dxc/DXIL/DxilModule.h"
#include "dxc/DXIL/DxilConstants.h"
#include "dxc/DXIL/DxilShaderModel.h"
#include "dxc/DXIL/DxilSignatureElement.h"
#include "dxc/DXIL/DxilFunctionProps.h"
#include "dxc/Support/WinAdapter.h"
#include "dxc/DXIL/DxilEntryProps.h"
#include "dxc/DXIL/DxilSubobject.h"
#include "dxc/DXIL/DxilInstructions.h"
#include "dxc/DXIL/DxilCounters.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SetVector.h"
#include <unordered_set>

using std::make_unique;

using namespace llvm;
using std::string;
using std::vector;
using std::unique_ptr;


namespace {
class DxilErrorDiagnosticInfo : public DiagnosticInfo {
private:
  const char *m_message;
public:
  DxilErrorDiagnosticInfo(const char *str)
    : DiagnosticInfo(DK_FirstPluginKind, DiagnosticSeverity::DS_Error),
    m_message(str) { }

  void print(DiagnosticPrinter &DP) const override {
    DP << m_message;
  }
};
} // anon namespace

namespace hlsl {

namespace DXIL {
// Define constant variables exposed in DxilConstants.h
// TODO: revisit data layout descriptions for the following:
//      - x64 pointers?
//      - Keep elf manging(m:e)?

// For legacy data layout, everything less than 32 align to 32.
const char* kLegacyLayoutString = "e-m:e-p:32:32-i1:32-i8:32-i16:32-i32:32-i64:64-f16:32-f32:32-f64:64-n8:16:32:64";

// New data layout with native low precision types
const char* kNewLayoutString = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64";

// Function Attributes
// TODO: consider generating attributes from hctdb
const char* kFP32DenormKindString          = "fp32-denorm-mode";
const char* kFP32DenormValueAnyString      = "any";
const char* kFP32DenormValuePreserveString = "preserve";
const char* kFP32DenormValueFtzString      = "ftz";

const char *kDxBreakFuncName = "dx.break";
const char *kDxBreakCondName = "dx.break.cond";
const char *kDxBreakMDName = "dx.break.br";
const char *kDxIsHelperGlobalName = "dx.ishelper";

const char *kHostLayoutTypePrefix = "hostlayout.";

const char* kWaveOpsIncludeHelperLanesString = "waveops-include-helper-lanes";
}

void SetDxilHook(Module &M);
void ClearDxilHook(Module &M);

//------------------------------------------------------------------------------
//
//  DxilModule methods.
//
DxilModule::DxilModule(Module *pModule)
: m_StreamPrimitiveTopology(DXIL::PrimitiveTopology::Undefined)
, m_ActiveStreamMask(0)
, m_Ctx(pModule->getContext())
, m_pModule(pModule)
, m_pEntryFunc(nullptr)
, m_EntryName("")
, m_pMDHelper(make_unique<DxilMDHelper>(pModule, make_unique<DxilExtraPropertyHelper>(pModule)))
, m_pDebugInfoFinder(nullptr)
, m_pSM(nullptr)
, m_DxilMajor(DXIL::kDxilMajor)
, m_DxilMinor(DXIL::kDxilMinor)
, m_ValMajor(1)
, m_ValMinor(0)
, m_ForceZeroStoreLifetimes(false)
, m_pOP(make_unique<OP>(pModule->getContext(), pModule))
, m_pTypeSystem(make_unique<DxilTypeSystem>(pModule))
, m_bDisableOptimizations(false)
, m_bUseMinPrecision(true) // use min precision by default
, m_bAllResourcesBound(false)
, m_IntermediateFlags(0)
, m_AutoBindingSpace(UINT_MAX)
, m_pSubobjects(nullptr)
, m_bMetadataErrors(false)
{

  DXASSERT_NOMSG(m_pModule != nullptr);
  SetDxilHook(*m_pModule);

#ifndef NDEBUG
  // Pin LLVM dump methods.
  void (__thiscall Module::*pfnModuleDump)() const = &Module::dump;
  void (__thiscall Type::*pfnTypeDump)() const = &Type::dump;
  void (__thiscall Function::*pfnViewCFGOnly)() const = &Function::viewCFGOnly;
  m_pUnused = (char *)&pfnModuleDump - (char *)&pfnTypeDump;
  m_pUnused -= (size_t)&pfnViewCFGOnly;
#endif
}

DxilModule::~DxilModule() { ClearDxilHook(*m_pModule); }

LLVMContext &DxilModule::GetCtx() const { return m_Ctx; }
Module *DxilModule::GetModule() const { return m_pModule; }
OP *DxilModule::GetOP() const { return m_pOP.get(); }

void DxilModule::SetShaderModel(const ShaderModel *pSM, bool bUseMinPrecision) {
  DXASSERT(m_pSM == nullptr || (pSM != nullptr && *m_pSM == *pSM), "shader model must not change for the module");
  DXASSERT(pSM != nullptr && pSM->IsValidForDxil(), "shader model must be valid");
  DXASSERT(pSM->IsValidForModule(), "shader model must be valid for top-level module use");
  m_pSM = pSM;
  m_pSM->GetDxilVersion(m_DxilMajor, m_DxilMinor);
  m_pMDHelper->SetShaderModel(m_pSM);
  m_bUseMinPrecision = bUseMinPrecision;
  m_pOP->SetMinPrecision(m_bUseMinPrecision);
  m_pTypeSystem->SetMinPrecision(m_bUseMinPrecision);

  if (!m_pSM->IsLib()) {
    // Always have valid entry props for non-lib case from this point on.
    DxilFunctionProps props;
    props.shaderKind = m_pSM->GetKind();
    m_DxilEntryPropsMap[nullptr] =
      make_unique<DxilEntryProps>(props, m_bUseMinPrecision);
  }
  m_SerializedRootSignature.clear();
}

const ShaderModel *DxilModule::GetShaderModel() const {
  return m_pSM;
}

void DxilModule::GetDxilVersion(unsigned &DxilMajor, unsigned &DxilMinor) const {
  DxilMajor = m_DxilMajor;
  DxilMinor = m_DxilMinor;
}

void DxilModule::SetValidatorVersion(unsigned ValMajor, unsigned ValMinor) {
  m_ValMajor = ValMajor;
  m_ValMinor = ValMinor;
}

void DxilModule::SetForceZeroStoreLifetimes(bool ForceZeroStoreLifetimes) {
  m_ForceZeroStoreLifetimes = ForceZeroStoreLifetimes;
}

bool DxilModule::UpgradeValidatorVersion(unsigned ValMajor, unsigned ValMinor) {
  // Don't upgrade if validation was disabled.
  if (m_ValMajor == 0 && m_ValMinor == 0) {
    return false;
  }
  if (ValMajor > m_ValMajor || (ValMajor == m_ValMajor && ValMinor > m_ValMinor)) {
    // Module requires higher validator version than previously set
    SetValidatorVersion(ValMajor, ValMinor);
    return true;
  }
  return false;
}

void DxilModule::GetValidatorVersion(unsigned &ValMajor, unsigned &ValMinor) const {
  ValMajor = m_ValMajor;
  ValMinor = m_ValMinor;
}

bool DxilModule::GetForceZeroStoreLifetimes() const {
  return m_ForceZeroStoreLifetimes;
}

bool DxilModule::GetMinValidatorVersion(unsigned &ValMajor, unsigned &ValMinor) const {
  if (!m_pSM)
    return false;
  m_pSM->GetMinValidatorVersion(ValMajor, ValMinor);
  if (DXIL::CompareVersions(ValMajor, ValMinor, 1, 5) < 0 &&
      m_ShaderFlags.GetRaytracingTier1_1())
    ValMinor = 5;
  else if (DXIL::CompareVersions(ValMajor, ValMinor, 1, 4) < 0 &&
           GetSubobjects() && !GetSubobjects()->GetSubobjects().empty())
    ValMinor = 4;
  else if (DXIL::CompareVersions(ValMajor, ValMinor, 1, 1) < 0 &&
      (m_ShaderFlags.GetFeatureInfo() & hlsl::DXIL::ShaderFeatureInfo_ViewID))
    ValMinor = 1;
  return true;
}

bool DxilModule::UpgradeToMinValidatorVersion() {
  unsigned ValMajor = 1, ValMinor = 0;
  if (GetMinValidatorVersion(ValMajor, ValMinor)) {
    return UpgradeValidatorVersion(ValMajor, ValMinor);
  }
  return false;
}

Function *DxilModule::GetEntryFunction() {
  return m_pEntryFunc;
}

const Function *DxilModule::GetEntryFunction() const {
  return m_pEntryFunc;
}

llvm::SmallVector<llvm::Function *, 64> DxilModule::GetExportedFunctions() {
    llvm::SmallVector<llvm::Function *, 64> ret;
    for (auto const& fn : m_DxilEntryPropsMap) {
      if (fn.first != nullptr) {
        ret.push_back(const_cast<llvm::Function*>(fn.first));
      }
    }
    if (ret.empty()) {
      auto *entryFunction = m_pEntryFunc;
      if (entryFunction == nullptr) {
        entryFunction = GetPatchConstantFunction();
      }
      ret.push_back(entryFunction);
    }
    return ret;
}

void DxilModule::SetEntryFunction(Function *pEntryFunc) {
  if (m_pSM->IsLib()) {
    DXASSERT(pEntryFunc == nullptr,
             "Otherwise, trying to set an entry function on library");
    m_pEntryFunc = nullptr;
    return;
  }
  DXASSERT(m_DxilEntryPropsMap.size() == 1, "should have one entry prop");
  m_pEntryFunc = pEntryFunc;
  // Move entry props to new function in order to preserve them.
  std::unique_ptr<DxilEntryProps> Props = std::move(m_DxilEntryPropsMap.begin()->second);
  m_DxilEntryPropsMap.clear();
  m_DxilEntryPropsMap[m_pEntryFunc] = std::move(Props);
}

const string &DxilModule::GetEntryFunctionName() const {
  return m_EntryName;
}

void DxilModule::SetEntryFunctionName(const string &name) {
  m_EntryName = name;
}

llvm::Function *DxilModule::GetPatchConstantFunction() {
  if (!m_pSM->IsHS())
    return nullptr;
  DXASSERT(m_DxilEntryPropsMap.size() == 1, "should have one entry prop");
  DxilFunctionProps &props = m_DxilEntryPropsMap.begin()->second->props;
  DXASSERT(props.IsHS(), "Must be HS profile");
  return props.ShaderProps.HS.patchConstantFunc;
}

const llvm::Function *DxilModule::GetPatchConstantFunction() const {
  if (!m_pSM->IsHS())
    return nullptr;
  DXASSERT(m_DxilEntryPropsMap.size() == 1, "should have one entry prop");
  const DxilFunctionProps &props = m_DxilEntryPropsMap.begin()->second->props;
  DXASSERT(props.IsHS(), "Must be HS profile");
  return props.ShaderProps.HS.patchConstantFunc;
}

void DxilModule::SetPatchConstantFunction(llvm::Function *patchConstantFunc) {
  if (!m_pSM->IsHS())
    return;
  DXASSERT(m_DxilEntryPropsMap.size() == 1, "should have one entry prop");
  DxilFunctionProps &props = m_DxilEntryPropsMap.begin()->second->props;
  DXASSERT(props.IsHS(), "Must be HS profile");
  auto &HS = props.ShaderProps.HS;
  if (HS.patchConstantFunc != patchConstantFunc) {
    if (HS.patchConstantFunc)
      m_PatchConstantFunctions.erase(HS.patchConstantFunc);
    HS.patchConstantFunc = patchConstantFunc;
    if (patchConstantFunc)
      m_PatchConstantFunctions.insert(patchConstantFunc);
  }
}

bool DxilModule::IsEntryOrPatchConstantFunction(const llvm::Function* pFunc) const {
  return pFunc == GetEntryFunction() || pFunc == GetPatchConstantFunction();
}

unsigned DxilModule::GetGlobalFlags() const {
  unsigned Flags = m_ShaderFlags.GetGlobalFlags();
  return Flags;
}

void DxilModule::CollectShaderFlagsForModule(ShaderFlags &Flags) {
  for (Function &F : GetModule()->functions()) {
    ShaderFlags funcFlags = ShaderFlags::CollectShaderFlags(&F, this);
    Flags.CombineShaderFlags(funcFlags);
  };

  const ShaderModel *SM = GetShaderModel();

  unsigned NumUAVs = 0;
  const unsigned kSmallUAVCount = 8;

  bool hasRawAndStructuredBuffer = false;

  for (auto &UAV : m_UAVs) {
    unsigned uavSize = UAV->GetRangeSize();
    NumUAVs += uavSize > 8U? 9U: uavSize; // avoid overflow
    if (UAV->IsROV())
      Flags.SetROVs(true);
    switch (UAV->GetKind()) {
    case DXIL::ResourceKind::RawBuffer:
    case DXIL::ResourceKind::StructuredBuffer:
      hasRawAndStructuredBuffer = true;
      break;
    default:
      // Not raw/structured.
      break;
    }
  }
  // Maintain earlier erroneous counting of UAVs for compatibility
  if (DXIL::CompareVersions(m_ValMajor, m_ValMinor, 1, 6) < 0)
    Flags.Set64UAVs(m_UAVs.size() > kSmallUAVCount);
  else
    Flags.Set64UAVs(NumUAVs > kSmallUAVCount);

  if (NumUAVs && !(SM->IsCS() || SM->IsPS()))
    Flags.SetUAVsAtEveryStage(true);


  for (auto &SRV : m_SRVs) {
    switch (SRV->GetKind()) {
    case DXIL::ResourceKind::RawBuffer:
    case DXIL::ResourceKind::StructuredBuffer:
      hasRawAndStructuredBuffer = true;
      break;
    default:
      // Not raw/structured.
      break;
    }
  }
  
  Flags.SetEnableRawAndStructuredBuffers(hasRawAndStructuredBuffer);

  bool hasCSRawAndStructuredViaShader4X =
      hasRawAndStructuredBuffer && m_pSM->GetMajor() == 4 && m_pSM->IsCS();
  Flags.SetCSRawAndStructuredViaShader4X(hasCSRawAndStructuredViaShader4X);
}

void DxilModule::CollectShaderFlagsForModule() {
  CollectShaderFlagsForModule(m_ShaderFlags);

  // This is also where we record the size of the mesh payload for amplification shader output
  for (Function &F : GetModule()->functions()) {
    if (HasDxilEntryProps(&F)) {
      DxilFunctionProps &props = GetDxilFunctionProps(&F);
      if (props.shaderKind == DXIL::ShaderKind::Amplification) {
        if (props.ShaderProps.AS.payloadSizeInBytes != 0)
          continue;
        for (const BasicBlock &BB : F.getBasicBlockList()) {
          for (const Instruction &I : BB.getInstList()) {
            const DxilInst_DispatchMesh dispatch(const_cast<Instruction*>(&I));
            if (dispatch) {
              Type *payloadTy = dispatch.get_payload()->getType()->getPointerElementType();
              const DataLayout &DL = m_pModule->getDataLayout();
              props.ShaderProps.AS.payloadSizeInBytes = DL.getTypeAllocSize(payloadTy);
            }
          }
        }
      }
    }
  }
}

void DxilModule::SetNumThreads(unsigned x, unsigned y, unsigned z) {
  DXASSERT(m_DxilEntryPropsMap.size() == 1 &&
           (m_pSM->IsCS() || m_pSM->IsMS() || m_pSM->IsAS()),
           "only works for CS/MS/AS profiles");
  DxilFunctionProps &props = m_DxilEntryPropsMap.begin()->second->props;
  DXASSERT_NOMSG(m_pSM->GetKind() == props.shaderKind);
  unsigned *numThreads = props.IsCS() ? props.ShaderProps.CS.numThreads :
    props.IsMS() ? props.ShaderProps.MS.numThreads : props.ShaderProps.AS.numThreads;
  numThreads[0] = x;
  numThreads[1] = y;
  numThreads[2] = z;
}
unsigned DxilModule::GetNumThreads(unsigned idx) const {
  DXASSERT(m_DxilEntryPropsMap.size() == 1 &&
           (m_pSM->IsCS() || m_pSM->IsMS() || m_pSM->IsAS()),
           "only works for CS/MS/AS profiles");
  DXASSERT(idx < 3, "Thread dimension index must be 0-2");
  __analysis_assume(idx < 3);
  if (!(m_pSM->IsCS() || m_pSM->IsMS() || m_pSM->IsAS()))
    return 0;
  const DxilFunctionProps &props = m_DxilEntryPropsMap.begin()->second->props;
  DXASSERT_NOMSG(m_pSM->GetKind() == props.shaderKind);
  const unsigned *numThreads = props.IsCS() ? props.ShaderProps.CS.numThreads :
    props.IsMS() ? props.ShaderProps.MS.numThreads : props.ShaderProps.AS.numThreads;
  return numThreads[idx];
}

void DxilModule::SetWaveSize(unsigned size) {
  DXASSERT(m_DxilEntryPropsMap.size() == 1 && m_pSM->IsCS(),
    "only works for CS profile");
  DxilFunctionProps &props = m_DxilEntryPropsMap.begin()->second->props;
  DXASSERT_NOMSG(m_pSM->GetKind() == props.shaderKind);
  props.waveSize = size;
}

unsigned DxilModule::GetWaveSize() const {
  DXASSERT(m_DxilEntryPropsMap.size() == 1 && m_pSM->IsCS(),
    "only works for CS profiles");
  if (!m_pSM->IsCS())
    return 0;
  const DxilFunctionProps &props = m_DxilEntryPropsMap.begin()->second->props;
  DXASSERT_NOMSG(m_pSM->GetKind() == props.shaderKind);
  return props.waveSize;
}

DXIL::InputPrimitive DxilModule::GetInputPrimitive() const {
  if (!m_pSM->IsGS())
    return DXIL::InputPrimitive::Undefined;
  DXASSERT(m_DxilEntryPropsMap.size() == 1, "should have one entry prop");
  DxilFunctionProps &props = m_DxilEntryPropsMap.begin()->second->props;
  DXASSERT(props.IsGS(), "Must be GS profile");
  return props.ShaderProps.GS.inputPrimitive;
}

void DxilModule::SetInputPrimitive(DXIL::InputPrimitive IP) {
  DXASSERT(m_DxilEntryPropsMap.size() == 1 && m_pSM->IsGS(),
           "only works for GS profile");
  DxilFunctionProps &props = m_DxilEntryPropsMap.begin()->second->props;
  DXASSERT(props.IsGS(), "Must be GS profile");
  auto &GS = props.ShaderProps.GS;
  DXASSERT_NOMSG(DXIL::InputPrimitive::Undefined < IP && IP < DXIL::InputPrimitive::LastEntry);
  GS.inputPrimitive = IP;
}

unsigned DxilModule::GetMaxVertexCount() const {
  if (!m_pSM->IsGS())
    return 0;
  DXASSERT(m_DxilEntryPropsMap.size() == 1, "should have one entry prop");
  DxilFunctionProps &props = m_DxilEntryPropsMap.begin()->second->props;
  DXASSERT(props.IsGS(), "Must be GS profile");
  auto &GS = props.ShaderProps.GS;
  DXASSERT_NOMSG(GS.maxVertexCount != 0);
  return GS.maxVertexCount;
}

void DxilModule::SetMaxVertexCount(unsigned Count) {
  DXASSERT(m_DxilEntryPropsMap.size() == 1 && m_pSM->IsGS(),
           "only works for GS profile");
  DxilFunctionProps &props = m_DxilEntryPropsMap.begin()->second->props;
  DXASSERT(props.IsGS(), "Must be GS profile");
  auto &GS = props.ShaderProps.GS;
  GS.maxVertexCount = Count;
}

DXIL::PrimitiveTopology DxilModule::GetStreamPrimitiveTopology() const {
  return m_StreamPrimitiveTopology;
}

void DxilModule::SetStreamPrimitiveTopology(DXIL::PrimitiveTopology Topology) {
  m_StreamPrimitiveTopology = Topology;
  SetActiveStreamMask(m_ActiveStreamMask);  // Update props
}

bool DxilModule::HasMultipleOutputStreams() const {
  if (!m_pSM->IsGS()) {
    return false;
  } else {
    unsigned NumStreams = (m_ActiveStreamMask & 0x1) + 
                          ((m_ActiveStreamMask & 0x2) >> 1) + 
                          ((m_ActiveStreamMask & 0x4) >> 2) + 
                          ((m_ActiveStreamMask & 0x8) >> 3);
    DXASSERT_NOMSG(NumStreams <= DXIL::kNumOutputStreams);
    return NumStreams > 1;
  }
}

unsigned DxilModule::GetOutputStream() const {
  if (!m_pSM->IsGS()) {
    return 0;
  } else {
    DXASSERT_NOMSG(!HasMultipleOutputStreams());
    switch (m_ActiveStreamMask) {
    case 0x1: return 0;
    case 0x2: return 1;
    case 0x4: return 2;
    case 0x8: return 3;
    default: DXASSERT_NOMSG(false);
    }
    return (unsigned)(-1);
  }
}

unsigned DxilModule::GetGSInstanceCount() const {
  if (!m_pSM->IsGS())
    return 0;
  DXASSERT(m_DxilEntryPropsMap.size() == 1, "should have one entry prop");
  DxilFunctionProps &props = m_DxilEntryPropsMap.begin()->second->props;
  DXASSERT(props.IsGS(), "Must be GS profile");
  return props.ShaderProps.GS.instanceCount;
}

void DxilModule::SetGSInstanceCount(unsigned Count) {
  DXASSERT(m_DxilEntryPropsMap.size() == 1 && m_pSM->IsGS(),
           "only works for GS profile");
  DxilFunctionProps &props = m_DxilEntryPropsMap.begin()->second->props;
  DXASSERT(props.IsGS(), "Must be GS profile");
  props.ShaderProps.GS.instanceCount = Count;
}

bool DxilModule::IsStreamActive(unsigned Stream) const {
  return (m_ActiveStreamMask & (1<<Stream)) != 0;
}

void DxilModule::SetStreamActive(unsigned Stream, bool bActive) {
  if (bActive) {
    m_ActiveStreamMask |= (1<<Stream);
  } else {
    m_ActiveStreamMask &= ~(1<<Stream);
  }
  SetActiveStreamMask(m_ActiveStreamMask);
}

void DxilModule::SetActiveStreamMask(unsigned Mask) {
  m_ActiveStreamMask = Mask;
  DXASSERT(m_DxilEntryPropsMap.size() == 1 && m_pSM->IsGS(),
           "only works for GS profile");
  DxilFunctionProps &props = m_DxilEntryPropsMap.begin()->second->props;
  DXASSERT(props.IsGS(), "Must be GS profile");
  for (unsigned i = 0; i < 4; i++) {
    if (IsStreamActive(i))
      props.ShaderProps.GS.streamPrimitiveTopologies[i] = m_StreamPrimitiveTopology;
    else
      props.ShaderProps.GS.streamPrimitiveTopologies[i] = DXIL::PrimitiveTopology::Undefined;
  }
}

unsigned DxilModule::GetActiveStreamMask() const {
  return m_ActiveStreamMask;
}

bool DxilModule::GetUseMinPrecision() const {
  return m_bUseMinPrecision;
}

void DxilModule::SetDisableOptimization(bool DisableOptimization) {
  m_bDisableOptimizations = DisableOptimization;
}

bool DxilModule::GetDisableOptimization() const {
  return m_bDisableOptimizations;
}

void DxilModule::SetAllResourcesBound(bool ResourcesBound) {
  m_bAllResourcesBound = ResourcesBound;
}

bool DxilModule::GetAllResourcesBound() const {
  return m_bAllResourcesBound;
}

void DxilModule::SetResMayAlias(bool resMayAlias) {
  m_bResMayAlias = resMayAlias;
}

bool DxilModule::GetResMayAlias() const {
  return m_bResMayAlias;
}

void DxilModule::SetLegacyResourceReservation(bool legacyResourceReservation) {
  m_IntermediateFlags &= ~LegacyResourceReservation;
  if (legacyResourceReservation) m_IntermediateFlags |= LegacyResourceReservation;
}

bool DxilModule::GetLegacyResourceReservation() const {
  return (m_IntermediateFlags & LegacyResourceReservation) != 0;
}

void DxilModule::ClearIntermediateOptions() {
  m_IntermediateFlags = 0;
}

unsigned DxilModule::GetInputControlPointCount() const {
  if (!(m_pSM->IsHS() || m_pSM->IsDS()))
    return 0;
  DXASSERT(m_DxilEntryPropsMap.size() == 1, "should have one entry prop");
  DxilFunctionProps &props = m_DxilEntryPropsMap.begin()->second->props;
  DXASSERT(props.IsHS() || props.IsDS(), "Must be HS or DS profile");
  if (props.IsHS())
    return props.ShaderProps.HS.inputControlPoints;
  else
    return props.ShaderProps.DS.inputControlPoints;
}

void DxilModule::SetInputControlPointCount(unsigned NumICPs) {
  DXASSERT(m_DxilEntryPropsMap.size() == 1
           && (m_pSM->IsHS() || m_pSM->IsDS()),
           "only works for non-lib profile");
  DxilFunctionProps &props = m_DxilEntryPropsMap.begin()->second->props;
  DXASSERT(props.IsHS() || props.IsDS(), "Must be HS or DS profile");
  if (props.IsHS())
    props.ShaderProps.HS.inputControlPoints = NumICPs;
  else
    props.ShaderProps.DS.inputControlPoints = NumICPs;
}

DXIL::TessellatorDomain DxilModule::GetTessellatorDomain() const {
  if (!(m_pSM->IsHS() || m_pSM->IsDS()))
    return DXIL::TessellatorDomain::Undefined;
  DXASSERT_NOMSG(m_DxilEntryPropsMap.size() == 1);
  DxilFunctionProps &props = m_DxilEntryPropsMap.begin()->second->props;
  if (props.IsHS())
    return props.ShaderProps.HS.domain;
  else
    return props.ShaderProps.DS.domain;
}

void DxilModule::SetTessellatorDomain(DXIL::TessellatorDomain TessDomain) {
  DXASSERT(m_DxilEntryPropsMap.size() == 1
           && (m_pSM->IsHS() || m_pSM->IsDS()),
           "only works for HS or DS profile");
  DxilFunctionProps &props = m_DxilEntryPropsMap.begin()->second->props;
  DXASSERT(props.IsHS() || props.IsDS(), "Must be HS or DS profile");
  if (props.IsHS())
    props.ShaderProps.HS.domain = TessDomain;
  else
    props.ShaderProps.DS.domain = TessDomain;
}

unsigned DxilModule::GetOutputControlPointCount() const {
  if (!m_pSM->IsHS())
    return 0;
  DXASSERT(m_DxilEntryPropsMap.size() == 1, "should have one entry prop");
  DxilFunctionProps &props = m_DxilEntryPropsMap.begin()->second->props;
  DXASSERT(props.IsHS(), "Must be HS profile");
  return props.ShaderProps.HS.outputControlPoints;
}

void DxilModule::SetOutputControlPointCount(unsigned NumOCPs) {
  DXASSERT(m_DxilEntryPropsMap.size() == 1 && m_pSM->IsHS(),
           "only works for HS profile");
  DxilFunctionProps &props = m_DxilEntryPropsMap.begin()->second->props;
  DXASSERT(props.IsHS(), "Must be HS profile");
  props.ShaderProps.HS.outputControlPoints = NumOCPs;
}

DXIL::TessellatorPartitioning DxilModule::GetTessellatorPartitioning() const {
  if (!m_pSM->IsHS())
    return DXIL::TessellatorPartitioning::Undefined;
  DXASSERT(m_DxilEntryPropsMap.size() == 1, "should have one entry prop");
  DxilFunctionProps &props = m_DxilEntryPropsMap.begin()->second->props;
  DXASSERT(props.IsHS(), "Must be HS profile");
  return props.ShaderProps.HS.partition;
}

void DxilModule::SetTessellatorPartitioning(DXIL::TessellatorPartitioning TessPartitioning) {
  DXASSERT(m_DxilEntryPropsMap.size() == 1 && m_pSM->IsHS(),
           "only works for HS profile");
  DxilFunctionProps &props = m_DxilEntryPropsMap.begin()->second->props;
  DXASSERT(props.IsHS(), "Must be HS profile");
  props.ShaderProps.HS.partition = TessPartitioning;
}

DXIL::TessellatorOutputPrimitive DxilModule::GetTessellatorOutputPrimitive() const {
  if (!m_pSM->IsHS())
    return DXIL::TessellatorOutputPrimitive::Undefined;
  DXASSERT(m_DxilEntryPropsMap.size() == 1, "should have one entry prop");
  DxilFunctionProps &props = m_DxilEntryPropsMap.begin()->second->props;
  DXASSERT(props.IsHS(), "Must be HS profile");
  return props.ShaderProps.HS.outputPrimitive;
}

void DxilModule::SetTessellatorOutputPrimitive(DXIL::TessellatorOutputPrimitive TessOutputPrimitive) {
  DXASSERT(m_DxilEntryPropsMap.size() == 1 && m_pSM->IsHS(),
           "only works for HS profile");
  DxilFunctionProps &props = m_DxilEntryPropsMap.begin()->second->props;
  DXASSERT(props.IsHS(), "Must be HS profile");
  props.ShaderProps.HS.outputPrimitive = TessOutputPrimitive;
}

float DxilModule::GetMaxTessellationFactor() const {
  if (!m_pSM->IsHS())
    return 0.0F;
  DXASSERT(m_DxilEntryPropsMap.size() == 1, "should have one entry prop");
  DxilFunctionProps &props = m_DxilEntryPropsMap.begin()->second->props;
  DXASSERT(props.IsHS(), "Must be HS profile");
  return props.ShaderProps.HS.maxTessFactor;
}

void DxilModule::SetMaxTessellationFactor(float MaxTessellationFactor) {
  DXASSERT(m_DxilEntryPropsMap.size() == 1 && m_pSM->IsHS(),
           "only works for HS profile");
  DxilFunctionProps &props = m_DxilEntryPropsMap.begin()->second->props;
  DXASSERT(props.IsHS(), "Must be HS profile");
  props.ShaderProps.HS.maxTessFactor = MaxTessellationFactor;
}

unsigned DxilModule::GetMaxOutputVertices() const {
  if (!m_pSM->IsMS())
    return 0;
  DXASSERT(m_DxilEntryPropsMap.size() == 1, "should have one entry prop");
  DxilFunctionProps &props = m_DxilEntryPropsMap.begin()->second->props;
  DXASSERT(props.IsMS(), "Must be MS profile");
  return props.ShaderProps.MS.maxVertexCount;
}

void DxilModule::SetMaxOutputVertices(unsigned NumOVs) {
  DXASSERT(m_DxilEntryPropsMap.size() == 1 && m_pSM->IsMS(),
           "only works for MS profile");
  DxilFunctionProps &props = m_DxilEntryPropsMap.begin()->second->props;
  DXASSERT(props.IsMS(), "Must be MS profile");
  props.ShaderProps.MS.maxVertexCount = NumOVs;
}

unsigned DxilModule::GetMaxOutputPrimitives() const {
  if (!m_pSM->IsMS())
    return 0;
  DXASSERT(m_DxilEntryPropsMap.size() == 1, "should have one entry prop");
  DxilFunctionProps &props = m_DxilEntryPropsMap.begin()->second->props;
  DXASSERT(props.IsMS(), "Must be MS profile");
  return props.ShaderProps.MS.maxPrimitiveCount;
}

void DxilModule::SetMaxOutputPrimitives(unsigned NumOPs) {
  DXASSERT(m_DxilEntryPropsMap.size() == 1 && m_pSM->IsMS(),
           "only works for MS profile");
  DxilFunctionProps &props = m_DxilEntryPropsMap.begin()->second->props;
  DXASSERT(props.IsMS(), "Must be MS profile");
  props.ShaderProps.MS.maxPrimitiveCount = NumOPs;
}

DXIL::MeshOutputTopology DxilModule::GetMeshOutputTopology() const {
  if (!m_pSM->IsMS())
    return DXIL::MeshOutputTopology::Undefined;
  DXASSERT(m_DxilEntryPropsMap.size() == 1, "should have one entry prop");
  DxilFunctionProps &props = m_DxilEntryPropsMap.begin()->second->props;
  DXASSERT(props.IsMS(), "Must be MS profile");
  return props.ShaderProps.MS.outputTopology;
}

void DxilModule::SetMeshOutputTopology(DXIL::MeshOutputTopology MeshOutputTopology) {
  DXASSERT(m_DxilEntryPropsMap.size() == 1 && m_pSM->IsMS(),
           "only works for MS profile");
  DxilFunctionProps &props = m_DxilEntryPropsMap.begin()->second->props;
  DXASSERT(props.IsMS(), "Must be MS profile");
  props.ShaderProps.MS.outputTopology = MeshOutputTopology;
}

unsigned DxilModule::GetPayloadSizeInBytes() const {
  if (m_pSM->IsMS())
  {
    DXASSERT(m_DxilEntryPropsMap.size() == 1, "should have one entry prop");
    DxilFunctionProps &props = m_DxilEntryPropsMap.begin()->second->props;
    DXASSERT(props.IsMS(), "Must be MS profile");
    return props.ShaderProps.MS.payloadSizeInBytes;
  }
  else if(m_pSM->IsAS())
  {
    DXASSERT(m_DxilEntryPropsMap.size() == 1, "should have one entry prop");
    DxilFunctionProps &props = m_DxilEntryPropsMap.begin()->second->props;
    DXASSERT(props.IsAS(), "Must be AS profile");
    return props.ShaderProps.AS.payloadSizeInBytes;
  }
  else
  {
    return 0;
  }
}

void DxilModule::SetPayloadSizeInBytes(unsigned Size) {
  DXASSERT(m_DxilEntryPropsMap.size() == 1 && (m_pSM->IsMS() || m_pSM->IsAS()),
           "only works for MS or AS profile");
  if (m_pSM->IsMS())
  {
    DxilFunctionProps &props = m_DxilEntryPropsMap.begin()->second->props;
    DXASSERT(props.IsMS(), "Must be MS profile");
    props.ShaderProps.MS.payloadSizeInBytes = Size;
  } 
  else if (m_pSM->IsAS())
  {
    DxilFunctionProps &props = m_DxilEntryPropsMap.begin()->second->props;
    DXASSERT(props.IsAS(), "Must be AS profile");
    props.ShaderProps.AS.payloadSizeInBytes = Size;
  }
}

void DxilModule::SetAutoBindingSpace(uint32_t Space) {
  m_AutoBindingSpace = Space;
}
uint32_t DxilModule::GetAutoBindingSpace() const {
  return m_AutoBindingSpace;
}

void DxilModule::SetShaderProperties(DxilFunctionProps *props) {
  if (!props)
    return;
  DxilFunctionProps &ourProps = GetDxilFunctionProps(GetEntryFunction());
  if (props != &ourProps) {
    ourProps.shaderKind = props->shaderKind;
    ourProps.ShaderProps = props->ShaderProps;
  }
  switch (props->shaderKind) {
  case DXIL::ShaderKind::Pixel: {
    auto &PS = props->ShaderProps.PS;
    m_ShaderFlags.SetForceEarlyDepthStencil(PS.EarlyDepthStencil);
  } break;
  case DXIL::ShaderKind::Compute:
  case DXIL::ShaderKind::Domain:
  case DXIL::ShaderKind::Hull:
  case DXIL::ShaderKind::Vertex:
  case DXIL::ShaderKind::Mesh:
  case DXIL::ShaderKind::Amplification:
    break;
  default: {
    DXASSERT(props->shaderKind == DXIL::ShaderKind::Geometry,
             "else invalid shader kind");
    auto &GS = props->ShaderProps.GS;
    m_ActiveStreamMask = 0;
    for (size_t i = 0; i < _countof(GS.streamPrimitiveTopologies); ++i) {
      if (GS.streamPrimitiveTopologies[i] !=
          DXIL::PrimitiveTopology::Undefined) {
        m_ActiveStreamMask |= (1 << i);
        DXASSERT_NOMSG(m_StreamPrimitiveTopology ==
                           DXIL::PrimitiveTopology::Undefined ||
                       m_StreamPrimitiveTopology ==
                           GS.streamPrimitiveTopologies[i]);
        m_StreamPrimitiveTopology = GS.streamPrimitiveTopologies[i];
      }
    }
    // Refresh props:
    SetActiveStreamMask(m_ActiveStreamMask);
  } break;
  }
}

template<typename T> unsigned 
DxilModule::AddResource(vector<unique_ptr<T> > &Vec, unique_ptr<T> pRes) {
  DXASSERT_NOMSG((unsigned)Vec.size() < UINT_MAX);
  unsigned Id = (unsigned)Vec.size();
  Vec.emplace_back(std::move(pRes));
  return Id;
}

unsigned DxilModule::AddCBuffer(unique_ptr<DxilCBuffer> pCB) {
  return AddResource<DxilCBuffer>(m_CBuffers, std::move(pCB));
}

DxilCBuffer &DxilModule::GetCBuffer(unsigned idx) {
  return *m_CBuffers[idx];
}

const DxilCBuffer &DxilModule::GetCBuffer(unsigned idx) const {
  return *m_CBuffers[idx];
}

const vector<unique_ptr<DxilCBuffer> > &DxilModule::GetCBuffers() const {
  return m_CBuffers;
}

unsigned DxilModule::AddSampler(unique_ptr<DxilSampler> pSampler) {
  return AddResource<DxilSampler>(m_Samplers, std::move(pSampler));
}

DxilSampler &DxilModule::GetSampler(unsigned idx) {
  return *m_Samplers[idx];
}

const DxilSampler &DxilModule::GetSampler(unsigned idx) const {
  return *m_Samplers[idx];
}

const vector<unique_ptr<DxilSampler> > &DxilModule::GetSamplers() const {
  return m_Samplers;
}

unsigned DxilModule::AddSRV(unique_ptr<DxilResource> pSRV) {
  return AddResource<DxilResource>(m_SRVs, std::move(pSRV));
}

DxilResource &DxilModule::GetSRV(unsigned idx) {
  return *m_SRVs[idx];
}

const DxilResource &DxilModule::GetSRV(unsigned idx) const {
  return *m_SRVs[idx];
}

const vector<unique_ptr<DxilResource> > &DxilModule::GetSRVs() const {
  return m_SRVs;
}

unsigned DxilModule::AddUAV(unique_ptr<DxilResource> pUAV) {
  return AddResource<DxilResource>(m_UAVs, std::move(pUAV));
}

DxilResource &DxilModule::GetUAV(unsigned idx) {
  return *m_UAVs[idx];
}

const DxilResource &DxilModule::GetUAV(unsigned idx) const {
  return *m_UAVs[idx];
}

const vector<unique_ptr<DxilResource> > &DxilModule::GetUAVs() const {
  return m_UAVs;
}

void DxilModule::LoadDxilResourceBaseFromMDNode(MDNode *MD, DxilResourceBase &R) {
  return m_pMDHelper->LoadDxilResourceBaseFromMDNode(MD, R);
}
void DxilModule::LoadDxilResourceFromMDNode(llvm::MDNode *MD, DxilResource &R) {
  return m_pMDHelper->LoadDxilResourceFromMDNode(MD, R);
}
void DxilModule::LoadDxilSamplerFromMDNode(llvm::MDNode *MD, DxilSampler &S) {
  return m_pMDHelper->LoadDxilSamplerFromMDNode(MD, S);
}

template <typename TResource>
static void RemoveResources(std::vector<std::unique_ptr<TResource>> &vec,
                    std::unordered_set<unsigned> &immResID) {
  for (auto p = vec.begin(); p != vec.end();) {
    auto c = p++;
    if (immResID.count((*c)->GetID()) == 0) {
      p = vec.erase(c);
    }
  }
}

static void CollectUsedResource(Value *resID,
                                std::unordered_set<Value *> &usedResID) {
  if (usedResID.count(resID) > 0)
    return;

  usedResID.insert(resID);
  if (dyn_cast<ConstantInt>(resID)) {
    // Do nothing
  } else if (ZExtInst *ZEI = dyn_cast<ZExtInst>(resID)) {
    if (ZEI->getSrcTy()->isIntegerTy()) {
      IntegerType *ITy = cast<IntegerType>(ZEI->getSrcTy());
      if (ITy->getBitWidth() == 1) {
        usedResID.insert(ConstantInt::get(ZEI->getDestTy(), 0));
        usedResID.insert(ConstantInt::get(ZEI->getDestTy(), 1));
      }
    }
  } else if (SelectInst *SI = dyn_cast<SelectInst>(resID)) {
    CollectUsedResource(SI->getTrueValue(), usedResID);
    CollectUsedResource(SI->getFalseValue(), usedResID);
  } else if (PHINode *Phi = dyn_cast<PHINode>(resID)) {
    for (Use &U : Phi->incoming_values()) {
      CollectUsedResource(U.get(), usedResID);
    }
  }
  // TODO: resID could be other types of instructions depending on the compiler optimization.
}

static void ConvertUsedResource(std::unordered_set<unsigned> &immResID,
                                std::unordered_set<Value *> &usedResID) {
  for (Value *V : usedResID) {
    if (ConstantInt *cResID = dyn_cast<ConstantInt>(V)) {
      immResID.insert(cResID->getLimitedValue());
    }
  }
}

void DxilModule::RemoveFunction(llvm::Function *F) {
  DXASSERT_NOMSG(F != nullptr);
  m_DxilEntryPropsMap.erase(F);
  if (m_pTypeSystem.get()->GetFunctionAnnotation(F))
    m_pTypeSystem.get()->EraseFunctionAnnotation(F);
  m_pOP->RemoveFunction(F);
}

void DxilModule::RemoveUnusedResources() {
  DXASSERT(!m_pSM->IsLib(), "this function does not work on libraries");
  hlsl::OP *hlslOP = GetOP();
  Function *createHandleFunc = hlslOP->GetOpFunc(DXIL::OpCode::CreateHandle, Type::getVoidTy(GetCtx()));
  if (createHandleFunc->user_empty()) {
    m_CBuffers.clear();
    m_UAVs.clear();
    m_SRVs.clear();
    m_Samplers.clear();
    createHandleFunc->eraseFromParent();
    return;
  }

  std::unordered_set<Value *> usedUAVID;
  std::unordered_set<Value *> usedSRVID;
  std::unordered_set<Value *> usedSamplerID;
  std::unordered_set<Value *> usedCBufID;
  // Collect used ID.
  for (User *U : createHandleFunc->users()) {
    CallInst *CI = cast<CallInst>(U);
    Value *vResClass =
        CI->getArgOperand(DXIL::OperandIndex::kCreateHandleResClassOpIdx);
    ConstantInt *cResClass = cast<ConstantInt>(vResClass);
    DXIL::ResourceClass resClass =
        static_cast<DXIL::ResourceClass>(cResClass->getLimitedValue());
    // Skip unused resource handle.
    if (CI->user_empty())
      continue;

    Value *resID =
        CI->getArgOperand(DXIL::OperandIndex::kCreateHandleResIDOpIdx);
    switch (resClass) {
    case DXIL::ResourceClass::CBuffer:
      CollectUsedResource(resID, usedCBufID);
      break;
    case DXIL::ResourceClass::Sampler:
      CollectUsedResource(resID, usedSamplerID);
      break;
    case DXIL::ResourceClass::SRV:
      CollectUsedResource(resID, usedSRVID);
      break;
    case DXIL::ResourceClass::UAV:
      CollectUsedResource(resID, usedUAVID);
      break;
    default:
      DXASSERT(0, "invalid res class");
      break;
    }
  }

  std::unordered_set<unsigned> immUAVID;
  std::unordered_set<unsigned> immSRVID;
  std::unordered_set<unsigned> immSamplerID;
  std::unordered_set<unsigned> immCBufID;
  ConvertUsedResource(immUAVID, usedUAVID);
  ConvertUsedResource(immSRVID, usedSRVID);
  ConvertUsedResource(immSamplerID, usedSamplerID);
  ConvertUsedResource(immCBufID, usedCBufID);

  RemoveResources(m_UAVs, immUAVID);
  RemoveResources(m_SRVs, immSRVID);
  RemoveResources(m_Samplers, immSamplerID);
  RemoveResources(m_CBuffers, immCBufID);
}

namespace {
template <typename TResource>
static void RemoveResourcesWithUnusedSymbolsHelper(std::vector<std::unique_ptr<TResource>> &vec) {
  unsigned resID = 0;
  std::unordered_set<GlobalVariable *> eraseList; // Need in case of duplicate defs of lib resources
  for (auto p = vec.begin(); p != vec.end();) {
    auto c = p++;
    Constant *symbol = (*c)->GetGlobalSymbol();
    symbol->removeDeadConstantUsers();
    if (symbol->user_empty()) {
      p = vec.erase(c);
      if (GlobalVariable *GV = dyn_cast<GlobalVariable>(symbol))
        eraseList.insert(GV);
      continue;
    }
    if ((*c)->GetID() != resID) {
      (*c)->SetID(resID);
    }
    resID++;
  }
  for (auto gv : eraseList) {
    gv->eraseFromParent();
  }
}
}

void DxilModule::RemoveResourcesWithUnusedSymbols() {
  RemoveResourcesWithUnusedSymbolsHelper(m_SRVs);
  RemoveResourcesWithUnusedSymbolsHelper(m_UAVs);
  RemoveResourcesWithUnusedSymbolsHelper(m_CBuffers);
  RemoveResourcesWithUnusedSymbolsHelper(m_Samplers);
}

namespace {
template <typename TResource>
static bool RenameResources(std::vector<std::unique_ptr<TResource>> &vec, const std::string &prefix) {
  bool bChanged = false;
  for (auto &res : vec) {
    res->SetGlobalName(prefix + res->GetGlobalName());
    if (GlobalVariable *GV = dyn_cast<GlobalVariable>(res->GetGlobalSymbol())) {
      GV->setName(prefix + GV->getName());
    }
    bChanged = true;
  }
  return bChanged;
}
}

bool DxilModule::RenameResourcesWithPrefix(const std::string &prefix) {
  bool bChanged = false;
  bChanged |= RenameResources(m_SRVs, prefix);
  bChanged |= RenameResources(m_UAVs, prefix);
  bChanged |= RenameResources(m_CBuffers, prefix);
  bChanged |= RenameResources(m_Samplers, prefix);
  return bChanged;
}

namespace {
template <typename TResource>
static bool RenameGlobalsWithBinding(std::vector<std::unique_ptr<TResource>> &vec, llvm::StringRef prefix, bool bKeepName) {
  bool bChanged = false;
  for (auto &res : vec) {
    if (res->IsAllocated()) {
      std::string newName;
      if (bKeepName)
        newName = (Twine(res->GetGlobalName()) + "." + Twine(prefix) + Twine(res->GetLowerBound()) + "." + Twine(res->GetSpaceID())).str();
      else
        newName = (Twine(prefix) + Twine(res->GetLowerBound()) + "." + Twine(res->GetSpaceID())).str();

      res->SetGlobalName(newName);
      if (GlobalVariable *GV = dyn_cast<GlobalVariable>(res->GetGlobalSymbol())) {
        GV->setName(newName);
      }
      bChanged = true;
    }
  }
  return bChanged;
}
}

bool DxilModule::RenameResourceGlobalsWithBinding(bool bKeepName) {
  bool bChanged = false;
  bChanged |= RenameGlobalsWithBinding(m_SRVs, "t", bKeepName);
  bChanged |= RenameGlobalsWithBinding(m_UAVs, "u", bKeepName);
  bChanged |= RenameGlobalsWithBinding(m_CBuffers, "b", bKeepName);
  bChanged |= RenameGlobalsWithBinding(m_Samplers, "s", bKeepName);
  return bChanged;
}

DxilSignature &DxilModule::GetInputSignature() {
  DXASSERT(m_DxilEntryPropsMap.size() == 1 && !m_pSM->IsLib(),
           "only works for non-lib profile");
  return m_DxilEntryPropsMap.begin()->second->sig.InputSignature;
}

const DxilSignature &DxilModule::GetInputSignature() const {
  DXASSERT(m_DxilEntryPropsMap.size() == 1 && !m_pSM->IsLib(),
           "only works for non-lib profile");
  return m_DxilEntryPropsMap.begin()->second->sig.InputSignature;
}

DxilSignature &DxilModule::GetOutputSignature() {
  DXASSERT(m_DxilEntryPropsMap.size() == 1 && !m_pSM->IsLib(),
           "only works for non-lib profile");
  return m_DxilEntryPropsMap.begin()->second->sig.OutputSignature;
}

const DxilSignature &DxilModule::GetOutputSignature() const {
  DXASSERT(m_DxilEntryPropsMap.size() == 1 && !m_pSM->IsLib(),
           "only works for non-lib profile");
  return m_DxilEntryPropsMap.begin()->second->sig.OutputSignature;
}

DxilSignature &DxilModule::GetPatchConstOrPrimSignature() {
  DXASSERT(m_DxilEntryPropsMap.size() == 1 && !m_pSM->IsLib(),
           "only works for non-lib profile");
  return m_DxilEntryPropsMap.begin()->second->sig.PatchConstOrPrimSignature;
}

const DxilSignature &DxilModule::GetPatchConstOrPrimSignature() const {
  DXASSERT(m_DxilEntryPropsMap.size() == 1 && !m_pSM->IsLib(),
           "only works for non-lib profile");
  return m_DxilEntryPropsMap.begin()->second->sig.PatchConstOrPrimSignature;
}

const std::vector<uint8_t> &DxilModule::GetSerializedRootSignature() const {
  return m_SerializedRootSignature;
}

std::vector<uint8_t> &DxilModule::GetSerializedRootSignature() {
  return m_SerializedRootSignature;
}

// Entry props.
bool DxilModule::HasDxilEntrySignature(const llvm::Function *F) const {
  return m_DxilEntryPropsMap.find(F) != m_DxilEntryPropsMap.end();
}
DxilEntrySignature &DxilModule::GetDxilEntrySignature(const llvm::Function *F) {
  DXASSERT(m_DxilEntryPropsMap.count(F) != 0, "cannot find F in map");
  return m_DxilEntryPropsMap[F].get()->sig;
}
void DxilModule::ReplaceDxilEntryProps(llvm::Function *F,
                                       llvm::Function *NewF) {
  DXASSERT(m_DxilEntryPropsMap.count(F) != 0, "cannot find F in map");
  std::unique_ptr<DxilEntryProps> Props = std::move(m_DxilEntryPropsMap[F]);
  m_DxilEntryPropsMap.erase(F);
  m_DxilEntryPropsMap[NewF] = std::move(Props);
}
void DxilModule::CloneDxilEntryProps(llvm::Function *F, llvm::Function *NewF) {
  DXASSERT(m_DxilEntryPropsMap.count(F) != 0, "cannot find F in map");
  std::unique_ptr<DxilEntryProps> Props =
      make_unique<DxilEntryProps>(*m_DxilEntryPropsMap[F]);
  m_DxilEntryPropsMap[NewF] = std::move(Props);
}

bool DxilModule::HasDxilEntryProps(const llvm::Function *F) const {
  return m_DxilEntryPropsMap.find(F) != m_DxilEntryPropsMap.end();
}
DxilEntryProps &DxilModule::GetDxilEntryProps(const llvm::Function *F) {
  DXASSERT(m_DxilEntryPropsMap.count(F) != 0, "cannot find F in map");
  return *m_DxilEntryPropsMap.find(F)->second.get();
}
const DxilEntryProps &DxilModule::GetDxilEntryProps(const llvm::Function *F) const {
  DXASSERT(m_DxilEntryPropsMap.count(F) != 0, "cannot find F in map");
  return *m_DxilEntryPropsMap.find(F)->second.get();
}

bool DxilModule::HasDxilFunctionProps(const llvm::Function *F) const {
  return m_DxilEntryPropsMap.find(F) != m_DxilEntryPropsMap.end();
}
DxilFunctionProps &DxilModule::GetDxilFunctionProps(const llvm::Function *F) {
  return const_cast<DxilFunctionProps &>(
      static_cast<const DxilModule *>(this)->GetDxilFunctionProps(F));
}

const DxilFunctionProps &
DxilModule::GetDxilFunctionProps(const llvm::Function *F) const {
  DXASSERT(m_DxilEntryPropsMap.count(F) != 0, "cannot find F in map");
  return m_DxilEntryPropsMap.find(F)->second.get()->props;
}

void DxilModule::SetPatchConstantFunctionForHS(llvm::Function *hullShaderFunc, llvm::Function *patchConstantFunc) {
  auto propIter = m_DxilEntryPropsMap.find(hullShaderFunc);
  DXASSERT(propIter != m_DxilEntryPropsMap.end(),
           "Hull shader must already have function props!");
  DxilFunctionProps &props = propIter->second->props;
  DXASSERT(props.IsHS(), "else hullShaderFunc is not a Hull Shader");
  auto &HS = props.ShaderProps.HS;
  if (HS.patchConstantFunc != patchConstantFunc) {
    if (HS.patchConstantFunc)
      m_PatchConstantFunctions.erase(HS.patchConstantFunc);
    HS.patchConstantFunc = patchConstantFunc;
    if (patchConstantFunc)
      m_PatchConstantFunctions.insert(patchConstantFunc);
  }
}
bool DxilModule::IsGraphicsShader(const llvm::Function *F) const {
  return HasDxilFunctionProps(F) && GetDxilFunctionProps(F).IsGraphics();
}
bool DxilModule::IsPatchConstantShader(const llvm::Function *F) const {
  return m_PatchConstantFunctions.count(F) != 0;
}
bool DxilModule::IsComputeShader(const llvm::Function *F) const {
  return HasDxilFunctionProps(F) && GetDxilFunctionProps(F).IsCS();
}
bool DxilModule::IsEntryThatUsesSignatures(const llvm::Function *F) const {
  auto propIter = m_DxilEntryPropsMap.find(F);
  if (propIter != m_DxilEntryPropsMap.end()) {
    DxilFunctionProps &props = propIter->second->props;
    return props.IsGraphics() || props.IsCS();
  }
  // Otherwise, return true if patch constant function
  return IsPatchConstantShader(F);
}
bool DxilModule::IsEntry(const llvm::Function *F) const {
  auto propIter = m_DxilEntryPropsMap.find(F);
  if (propIter != m_DxilEntryPropsMap.end()) {
    DXASSERT(propIter->second->props.shaderKind != DXIL::ShaderKind::Invalid,
             "invalid entry props");
    return true;
  }
  // Otherwise, return true if patch constant function
  return IsPatchConstantShader(F);
}

bool DxilModule::StripRootSignatureFromMetadata() {
  NamedMDNode *pRootSignatureNamedMD = GetModule()->getNamedMetadata(DxilMDHelper::kDxilRootSignatureMDName);
  if (pRootSignatureNamedMD) {
    GetModule()->eraseNamedMetadata(pRootSignatureNamedMD);
    return true;
  }
  return false;
}

DxilSubobjects *DxilModule::GetSubobjects() {
  return m_pSubobjects.get();
}
const DxilSubobjects *DxilModule::GetSubobjects() const {
  return m_pSubobjects.get();
}
DxilSubobjects *DxilModule::ReleaseSubobjects() {
  return m_pSubobjects.release();
}
void DxilModule::ResetSubobjects(DxilSubobjects *subobjects) {
  m_pSubobjects.reset(subobjects);
}

bool DxilModule::StripSubobjectsFromMetadata() {
  NamedMDNode *pSubobjectsNamedMD = GetModule()->getNamedMetadata(DxilMDHelper::kDxilSubobjectsMDName);
  if (pSubobjectsNamedMD) {
    GetModule()->eraseNamedMetadata(pSubobjectsNamedMD);
    return true;
  }
  return false;
}

void DxilModule::UpdateValidatorVersionMetadata() {
  m_pMDHelper->EmitValidatorVersion(m_ValMajor, m_ValMinor);
}

void DxilModule::ResetSerializedRootSignature(std::vector<uint8_t> &Value) {
  m_SerializedRootSignature.assign(Value.begin(), Value.end());
}

DxilTypeSystem &DxilModule::GetTypeSystem() {
  return *m_pTypeSystem;
}

const DxilTypeSystem &DxilModule::GetTypeSystem() const {
  return *m_pTypeSystem;
}

std::vector<unsigned> &DxilModule::GetSerializedViewIdState() {
  return m_SerializedState;
}
const std::vector<unsigned> &DxilModule::GetSerializedViewIdState() const {
  return m_SerializedState;
}

void DxilModule::ResetTypeSystem(DxilTypeSystem *pValue) {
  m_pTypeSystem.reset(pValue);
}

void DxilModule::ResetOP(hlsl::OP *hlslOP) { m_pOP.reset(hlslOP); }

void DxilModule::ResetEntryPropsMap(DxilEntryPropsMap &&PropMap) {
  m_DxilEntryPropsMap.clear();
  std::move(PropMap.begin(), PropMap.end(),
            inserter(m_DxilEntryPropsMap, m_DxilEntryPropsMap.begin()));
}

static const StringRef llvmUsedName = "llvm.used";

void DxilModule::EmitLLVMUsed() {
  if (GlobalVariable *oldGV = m_pModule->getGlobalVariable(llvmUsedName)) {
    oldGV->eraseFromParent();
  }
  if (m_LLVMUsed.empty())
    return;

  vector<llvm::Constant *> GVs;
  Type *pI8PtrType = Type::getInt8PtrTy(m_Ctx, DXIL::kDefaultAddrSpace);

  GVs.resize(m_LLVMUsed.size());
  for (size_t i = 0, e = m_LLVMUsed.size(); i != e; i++) {
    Constant *pConst = cast<Constant>(&*m_LLVMUsed[i]);
    PointerType *pPtrType = dyn_cast<PointerType>(pConst->getType());
    if (pPtrType->getPointerAddressSpace() != DXIL::kDefaultAddrSpace) {
      // Cast pointer to addrspace 0, as LLVMUsed elements must have the same
      // type.
      GVs[i] = ConstantExpr::getAddrSpaceCast(pConst, pI8PtrType);
    } else {
      GVs[i] = ConstantExpr::getPointerCast(pConst, pI8PtrType);
    }
  }

  ArrayType *pATy = ArrayType::get(pI8PtrType, GVs.size());

  GlobalVariable *pGV =
      new GlobalVariable(*m_pModule, pATy, false, GlobalValue::AppendingLinkage,
                         ConstantArray::get(pATy, GVs), llvmUsedName);

  pGV->setSection("llvm.metadata");
}

void DxilModule::ClearLLVMUsed() {
  if (GlobalVariable *oldGV = m_pModule->getGlobalVariable(llvmUsedName)) {
    oldGV->eraseFromParent();
  }
  if (m_LLVMUsed.empty())
    return;

  for (size_t i = 0, e = m_LLVMUsed.size(); i != e; i++) {
    Constant *pConst = cast<Constant>(&*m_LLVMUsed[i]);
    pConst->removeDeadConstantUsers();
  }
  m_LLVMUsed.clear();
}

vector<GlobalVariable* > &DxilModule::GetLLVMUsed() {
  return m_LLVMUsed;
}

// DXIL metadata serialization/deserialization.
void DxilModule::ClearDxilMetadata(Module &M) {
  // Delete: DXIL version, validator version, DXIL shader model,
  // entry point tuples (shader properties, signatures, resources)
  // type system, view ID state, LLVM used, entry point tuples,
  // root signature, function properties.
  // Other cases for libs pending.
  // LLVM used is a global variable - handle separately.
  SmallVector<NamedMDNode*, 8> nodes;
  for (NamedMDNode &b : M.named_metadata()) {
    StringRef name = b.getName();
    if (name == DxilMDHelper::kDxilVersionMDName ||
      name == DxilMDHelper::kDxilValidatorVersionMDName ||
      name == DxilMDHelper::kDxilShaderModelMDName ||
      name == DxilMDHelper::kDxilEntryPointsMDName ||
      name == DxilMDHelper::kDxilRootSignatureMDName ||
      name == DxilMDHelper::kDxilIntermediateOptionsMDName ||
      name == DxilMDHelper::kDxilResourcesMDName ||
      name == DxilMDHelper::kDxilTypeSystemMDName ||
      name == DxilMDHelper::kDxilViewIdStateMDName ||
      name == DxilMDHelper::kDxilSubobjectsMDName ||
      name == DxilMDHelper::kDxilCountersMDName ||
      name.startswith(DxilMDHelper::kDxilTypeSystemHelperVariablePrefix)) {
      nodes.push_back(&b);
    }
  }
  for (size_t i = 0; i < nodes.size(); ++i) {
    M.eraseNamedMetadata(nodes[i]);
  }
}

void DxilModule::EmitDxilMetadata() {
  m_pMDHelper->EmitDxilVersion(m_DxilMajor, m_DxilMinor);
  m_pMDHelper->EmitValidatorVersion(m_ValMajor, m_ValMinor);
  m_pMDHelper->EmitDxilShaderModel(m_pSM);
  m_pMDHelper->EmitDxilIntermediateOptions(m_IntermediateFlags);

  MDTuple *pMDProperties = nullptr;
  uint64_t flag = m_ShaderFlags.GetShaderFlagsRaw();
  if (m_pSM->IsLib()) {
    DxilFunctionProps props;
    props.shaderKind = DXIL::ShaderKind::Library;
    pMDProperties = m_pMDHelper->EmitDxilEntryProperties(flag, props,
                                                         GetAutoBindingSpace());
  } else {
    pMDProperties = m_pMDHelper->EmitDxilEntryProperties(
        flag, m_DxilEntryPropsMap.begin()->second->props,
        GetAutoBindingSpace());
  }

  MDTuple *pMDSignatures = nullptr;
  if (!m_pSM->IsLib()) {
    pMDSignatures = m_pMDHelper->EmitDxilSignatures(
        m_DxilEntryPropsMap.begin()->second->sig);
  }
  MDTuple *pMDResources = EmitDxilResources();
  if (pMDResources)
    m_pMDHelper->EmitDxilResources(pMDResources);
  m_pMDHelper->EmitDxilTypeSystem(GetTypeSystem(), m_LLVMUsed);
  if (!m_pSM->IsLib() && !m_pSM->IsCS() &&
      ((m_ValMajor == 0 &&  m_ValMinor == 0) ||
       (m_ValMajor > 1 || (m_ValMajor == 1 && m_ValMinor >= 1)))) {
    m_pMDHelper->EmitDxilViewIdState(m_SerializedState);
  }

  // Emit the DXR Payload Annotations only for library Dxil 1.6 and above.
  if (m_pSM->IsLib()) {
    if (DXIL::CompareVersions(m_DxilMajor, m_DxilMinor, 1, 6) >= 0) {
      m_pMDHelper->EmitDxrPayloadAnnotations(GetTypeSystem());
    }
  }

  EmitLLVMUsed();
  MDTuple *pEntry = m_pMDHelper->EmitDxilEntryPointTuple(GetEntryFunction(), m_EntryName, pMDSignatures, pMDResources, pMDProperties);
  vector<MDNode *> Entries;
  Entries.emplace_back(pEntry);

  if (m_pSM->IsLib()) {
    // Sort functions by name to keep metadata deterministic
    vector<const Function *> funcOrder;
    funcOrder.reserve(m_DxilEntryPropsMap.size());

    std::transform( m_DxilEntryPropsMap.begin(),
                    m_DxilEntryPropsMap.end(),
                    std::back_inserter(funcOrder),
                    [](const std::pair<const llvm::Function * const, std::unique_ptr<DxilEntryProps>> &p) -> const Function* { return p.first; } );
    std::sort(funcOrder.begin(), funcOrder.end(), [](const Function *F1, const Function *F2) {
      return F1->getName() < F2->getName();
    });

    for (auto F : funcOrder) {
      auto &entryProps = m_DxilEntryPropsMap[F];
      MDTuple *pProps = m_pMDHelper->EmitDxilEntryProperties(0, entryProps->props, 0);
      MDTuple *pSig = m_pMDHelper->EmitDxilSignatures(entryProps->sig);

      MDTuple *pSubEntry = m_pMDHelper->EmitDxilEntryPointTuple(
          const_cast<Function *>(F), F->getName().str(), pSig, nullptr, pProps);

      Entries.emplace_back(pSubEntry);
    }
    funcOrder.clear();

    // Save Subobjects
    if (GetSubobjects()) {
      m_pMDHelper->EmitSubobjects(*GetSubobjects());
    }
  }
  m_pMDHelper->EmitDxilEntryPoints(Entries);

  if (!m_SerializedRootSignature.empty()) {
    m_pMDHelper->EmitRootSignature(m_SerializedRootSignature);
  }
}

bool DxilModule::IsKnownNamedMetaData(llvm::NamedMDNode &Node) {
  return DxilMDHelper::IsKnownNamedMetaData(Node);
}

bool DxilModule::HasMetadataErrors() {
  return m_bMetadataErrors;
}

void DxilModule::LoadDxilMetadata() {
  m_bMetadataErrors = false;
  m_pMDHelper->LoadValidatorVersion(m_ValMajor, m_ValMinor);
  const ShaderModel *loadedSM;
  m_pMDHelper->LoadDxilShaderModel(loadedSM);
  m_pMDHelper->LoadDxilIntermediateOptions(m_IntermediateFlags);

  // This must be set before LoadDxilEntryProperties
  m_pMDHelper->SetShaderModel(loadedSM);

  // Setting module shader model requires UseMinPrecision flag,
  // which requires loading m_ShaderFlags,
  // which requires global entry properties,
  // so load entry properties first, then set the shader model

  const llvm::NamedMDNode *pEntries = m_pMDHelper->GetDxilEntryPoints();
  if (!loadedSM->IsLib()) {
    IFTBOOL(pEntries->getNumOperands() == 1, DXC_E_INCORRECT_DXIL_METADATA);
  }
  Function *pEntryFunc;
  string EntryName;
  const llvm::MDOperand *pEntrySignatures, *pEntryResources, *pEntryProperties;
  m_pMDHelper->GetDxilEntryPoint(pEntries->getOperand(0),
                                 pEntryFunc, EntryName,
                                 pEntrySignatures, pEntryResources,
                                 pEntryProperties);

  uint64_t rawShaderFlags = 0;
  DxilFunctionProps entryFuncProps;
  entryFuncProps.shaderKind = loadedSM->GetKind();
  m_pMDHelper->LoadDxilEntryProperties(*pEntryProperties, rawShaderFlags,
                                       entryFuncProps, m_AutoBindingSpace);

  m_bUseMinPrecision = true;
  if (rawShaderFlags) {
    m_ShaderFlags.SetShaderFlagsRaw(rawShaderFlags);
    m_bUseMinPrecision = !m_ShaderFlags.GetUseNativeLowPrecision();
    m_bDisableOptimizations = m_ShaderFlags.GetDisableOptimizations();
    m_bAllResourcesBound = m_ShaderFlags.GetAllResourcesBound();
    m_bResMayAlias = !m_ShaderFlags.GetResMayNotAlias();
  }

  // Now that we have the UseMinPrecision flag, set shader model:
  SetShaderModel(loadedSM, m_bUseMinPrecision);
  // SetShaderModel will initialize m_DxilMajor/m_DxilMinor to min for SM,
  // so, load here after shader model so it matches the metadata.
  m_pMDHelper->LoadDxilVersion(m_DxilMajor, m_DxilMinor);

  if (loadedSM->IsLib()) {
    for (unsigned i = 1; i < pEntries->getNumOperands(); i++) {
      Function *pFunc;
      string Name;
      const llvm::MDOperand *pSignatures, *pResources, *pProperties;
      m_pMDHelper->GetDxilEntryPoint(pEntries->getOperand(i), pFunc, Name,
                                     pSignatures, pResources, pProperties);
      DxilFunctionProps props;

      uint64_t rawShaderFlags = 0;
      unsigned autoBindingSpace = 0;
      m_pMDHelper->LoadDxilEntryProperties(
          *pProperties, rawShaderFlags, props, autoBindingSpace);
      if (props.IsHS() && props.ShaderProps.HS.patchConstantFunc) {
        // Add patch constant function to m_PatchConstantFunctions
        m_PatchConstantFunctions.insert(props.ShaderProps.HS.patchConstantFunc);
      }

      std::unique_ptr<DxilEntryProps> pEntryProps =
          make_unique<DxilEntryProps>(props, m_bUseMinPrecision);
      m_pMDHelper->LoadDxilSignatures(*pSignatures, pEntryProps->sig);

      m_DxilEntryPropsMap[pFunc] = std::move(pEntryProps);
    }

    // Load Subobjects
    std::unique_ptr<DxilSubobjects> pSubobjects(new DxilSubobjects());
    m_pMDHelper->LoadSubobjects(*pSubobjects);
    if (pSubobjects->GetSubobjects().size()) {
      ResetSubobjects(pSubobjects.release());
    }
  } else {
    std::unique_ptr<DxilEntryProps> pEntryProps =
        make_unique<DxilEntryProps>(entryFuncProps, m_bUseMinPrecision);
    DxilFunctionProps *pFuncProps = &pEntryProps->props;
    m_pMDHelper->LoadDxilSignatures(*pEntrySignatures, pEntryProps->sig);

    m_DxilEntryPropsMap.clear();
    m_DxilEntryPropsMap[pEntryFunc] = std::move(pEntryProps);

    SetEntryFunction(pEntryFunc);
    SetEntryFunctionName(EntryName);
    SetShaderProperties(pFuncProps);
  }

  LoadDxilResources(*pEntryResources);

  // Type system is not required for consumption of dxil.
  try {
    m_pMDHelper->LoadDxilTypeSystem(*m_pTypeSystem.get());
  } catch (hlsl::Exception &) {
    m_bMetadataErrors = true;
#ifndef NDEBUG
    throw;
#endif
    m_pTypeSystem->GetStructAnnotationMap().clear();
    m_pTypeSystem->GetFunctionAnnotationMap().clear();
  }

  // Payload annotations not required for consumption of dxil.
  try {
    m_pMDHelper->LoadDxrPayloadAnnotations(*m_pTypeSystem.get());
  } catch (hlsl::Exception &) {
    m_bMetadataErrors = true;
#ifndef NDEBUG
    throw;
#endif
    m_pTypeSystem->GetPayloadAnnotationMap().clear();
  }

  m_pMDHelper->LoadRootSignature(m_SerializedRootSignature);

  m_pMDHelper->LoadDxilViewIdState(m_SerializedState);

  m_bMetadataErrors |= m_pMDHelper->HasExtraMetadata();
}

MDTuple *DxilModule::EmitDxilResources() {
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

void DxilModule::ReEmitDxilResources() {
  ClearDxilMetadata(*m_pModule);
  EmitDxilMetadata();
}

void DxilModule::EmitDxilCounters() {
  DxilCounters counters = {};
  hlsl::CountInstructions(*m_pModule, counters);
  m_pMDHelper->EmitDxilCounters(counters);
}
void DxilModule::LoadDxilCounters(DxilCounters &counters) const {
  m_pMDHelper->LoadDxilCounters(counters);
}


template <typename TResource>
static bool
StripResourcesReflection(std::vector<std::unique_ptr<TResource>> &vec) {
  bool bChanged = false;
  for (auto &p : vec) {
    p->SetGlobalName("");
    // Cannot remove global symbol which used by validation.
    bChanged = true;
  }
  return bChanged;
}

bool isSequentialType(Type *Ty) {
  return isa<ArrayType>(Ty) || isa<VectorType>(Ty) || isa<PointerType>(Ty);
}

// Return true if any members or components of struct <Ty> contain
// scalars of less than 32 bits or are matrices, in which case translation is required
typedef llvm::SmallSetVector<const StructType*, 4> SmallStructSetVector;
static bool ResourceTypeRequiresTranslation(const StructType * Ty, SmallStructSetVector & containedStructs) {
  if (Ty->getName().startswith("class.matrix."))
    return true;
  bool bResult = false;
  containedStructs.insert(Ty);
  for (auto eTy : Ty->elements()) {
    // Skip past all levels of sequential types to test their elements
    while ((isSequentialType(eTy))) {
      eTy = eTy->getContainedType(0);
    }
    // Recursively call this function again to process internal structs
    if (StructType *structTy = dyn_cast<StructType>(eTy)) {
      if (ResourceTypeRequiresTranslation(structTy, containedStructs))
        bResult = true;
    } else if (eTy->getScalarSizeInBits() < 32) { // test scalar sizes
      bResult = true;
    }
  }
  return bResult;
}

bool DxilModule::StripReflection() {
  bool bChanged = false;
  bool bIsLib = GetShaderModel()->IsLib();

  // Remove names.
  for (Function &F : m_pModule->functions()) {
    for (BasicBlock &BB : F) {
      if (BB.hasName()) {
        BB.setName("");
        bChanged = true;
      }
      for (Instruction &I : BB) {
        if (I.hasName()) {
          I.setName("");
          bChanged = true;
        }
      }
    }
  }

  if (bIsLib && GetUseMinPrecision())
  {
    // We must preserve struct annotations for resources containing min-precision types,
    // since they have not yet been converted for legacy layout.
    // Keep all structs contained in any we must keep.
    SmallStructSetVector structsToKeep;
      SmallStructSetVector containedStructs;
    for (auto &CBuf : GetCBuffers())
      if (StructType *ST = dyn_cast<StructType>(CBuf->GetHLSLType()))
        if (ResourceTypeRequiresTranslation(ST, containedStructs))
          structsToKeep.insert(containedStructs.begin(), containedStructs.end());

    for (auto &UAV : GetUAVs()) {
      if (DXIL::IsStructuredBuffer(UAV->GetKind()))
        if (StructType *ST = dyn_cast<StructType>(UAV->GetHLSLType()))
          if (ResourceTypeRequiresTranslation(ST, containedStructs))
            structsToKeep.insert(containedStructs.begin(), containedStructs.end());
    }

    for (auto &SRV : GetSRVs()) {
      if (SRV->IsStructuredBuffer() || SRV->IsTBuffer())
        if (StructType *ST = dyn_cast<StructType>(SRV->GetHLSLType()))
          if (ResourceTypeRequiresTranslation(ST, containedStructs))
            structsToKeep.insert(containedStructs.begin(), containedStructs.end());
    }

    m_pTypeSystem->GetStructAnnotationMap().remove_if([structsToKeep](
      const std::pair<const StructType *, std::unique_ptr<DxilStructAnnotation>>
          &I) { return !structsToKeep.count(I.first); });
  } else {
    // Remove struct annotations.
    if (!m_pTypeSystem->GetStructAnnotationMap().empty()) {
      m_pTypeSystem->GetStructAnnotationMap().clear();
      bChanged = true;
    }
    if (DXIL::CompareVersions(m_ValMajor, m_ValMinor, 1, 5) >= 0) {
      // Remove function annotations.
      if (!m_pTypeSystem->GetFunctionAnnotationMap().empty()) {
        m_pTypeSystem->GetFunctionAnnotationMap().clear();
        bChanged = true;
      }
    }
  }

  // Resource
  if (!bIsLib) {
    bChanged |= StripResourcesReflection(m_CBuffers);
    bChanged |= StripResourcesReflection(m_UAVs);
    bChanged |= StripResourcesReflection(m_SRVs);
    bChanged |= StripResourcesReflection(m_Samplers);
  }

  // Unused global.
  SmallVector<GlobalVariable *,2> UnusedGlobals;
  for (GlobalVariable &GV : m_pModule->globals()) {
    if (GV.use_empty()) {
      // Need to preserve this global, otherwise we drop constructors
      // for static globals.
      if (!bIsLib || GV.getName().compare("llvm.global_ctors") != 0)
        UnusedGlobals.emplace_back(&GV);
    }
  }
  bChanged |= !UnusedGlobals.empty();

  for (GlobalVariable *GV : UnusedGlobals) {
    GV->eraseFromParent();
  }

  // ReEmit meta.
  if (bChanged)
    ReEmitDxilResources();

  return bChanged;
}

void DxilModule::LoadDxilResources(const llvm::MDOperand &MDO) {
  if (MDO.get() == nullptr)
    return;

  const llvm::MDTuple *pSRVs, *pUAVs, *pCBuffers, *pSamplers;
  m_pMDHelper->GetDxilResources(MDO, pSRVs, pUAVs, pCBuffers, pSamplers);

  // Load SRV records.
  if (pSRVs != nullptr) {
    for (unsigned i = 0; i < pSRVs->getNumOperands(); i++) {
      unique_ptr<DxilResource> pSRV(new DxilResource);
      m_pMDHelper->LoadDxilSRV(pSRVs->getOperand(i), *pSRV);
      AddSRV(std::move(pSRV));
    }
  }

  // Load UAV records.
  if (pUAVs != nullptr) {
    for (unsigned i = 0; i < pUAVs->getNumOperands(); i++) {
      unique_ptr<DxilResource> pUAV(new DxilResource);
      m_pMDHelper->LoadDxilUAV(pUAVs->getOperand(i), *pUAV);
      AddUAV(std::move(pUAV));
    }
  }

  // Load CBuffer records.
  if (pCBuffers != nullptr) {
    for (unsigned i = 0; i < pCBuffers->getNumOperands(); i++) {
      unique_ptr<DxilCBuffer> pCB(new DxilCBuffer);
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

void DxilModule::StripShaderSourcesAndCompileOptions(bool bReplaceWithDummyData) {
  // Remove dx.source metadata.
  if (NamedMDNode *contents = m_pModule->getNamedMetadata(
          DxilMDHelper::kDxilSourceContentsMDName)) {
    contents->eraseFromParent();
    if (bReplaceWithDummyData) {
      // Insert an empty source and content
      llvm::LLVMContext &context = m_pModule->getContext();
      llvm::NamedMDNode *newNamedMD = m_pModule->getOrInsertNamedMetadata(DxilMDHelper::kDxilSourceContentsMDName);
      llvm::Metadata *operands[2] = { llvm::MDString::get(context, ""), llvm::MDString::get(context, "") };
      newNamedMD->addOperand(llvm::MDTuple::get(context, operands));
    }
  }
  if (NamedMDNode *defines =
          m_pModule->getNamedMetadata(DxilMDHelper::kDxilSourceDefinesMDName)) {
    defines->eraseFromParent();
    if (bReplaceWithDummyData) {
      llvm::LLVMContext &context = m_pModule->getContext();
      llvm::NamedMDNode *newNamedMD = m_pModule->getOrInsertNamedMetadata(DxilMDHelper::kDxilSourceDefinesMDName);
      newNamedMD->addOperand(llvm::MDTuple::get(context, llvm::ArrayRef<llvm::Metadata *>()));
    }
  }
  if (NamedMDNode *mainFileName = m_pModule->getNamedMetadata(
          DxilMDHelper::kDxilSourceMainFileNameMDName)) {
    mainFileName->eraseFromParent();
    if (bReplaceWithDummyData) {
      // Insert an empty file name
      llvm::LLVMContext &context = m_pModule->getContext();
      llvm::NamedMDNode *newNamedMD = m_pModule->getOrInsertNamedMetadata(DxilMDHelper::kDxilSourceMainFileNameMDName);
      llvm::Metadata *operands[1] = { llvm::MDString::get(context, "") };
      newNamedMD->addOperand(llvm::MDTuple::get(context, operands));
    }
  }
  if (NamedMDNode *arguments =
          m_pModule->getNamedMetadata(DxilMDHelper::kDxilSourceArgsMDName)) {
    arguments->eraseFromParent();
    if (bReplaceWithDummyData) {
      llvm::LLVMContext &context = m_pModule->getContext();
      llvm::NamedMDNode *newNamedMD = m_pModule->getOrInsertNamedMetadata(DxilMDHelper::kDxilSourceArgsMDName);
      newNamedMD->addOperand(llvm::MDTuple::get(context, llvm::ArrayRef<llvm::Metadata *>()));
    }
  }
  if (NamedMDNode *binding = m_pModule->getNamedMetadata(
          DxilMDHelper::kDxilDxcBindingTableMDName)) {
    binding->eraseFromParent();
  }
}

void DxilModule::StripDebugRelatedCode() {
  StripShaderSourcesAndCompileOptions();
  if (NamedMDNode *flags = m_pModule->getModuleFlagsMetadata()) {
    SmallVector<llvm::Module::ModuleFlagEntry, 4> flagEntries;
    m_pModule->getModuleFlagsMetadata(flagEntries);
    flags->eraseFromParent();

    for (unsigned i = 0; i < flagEntries.size(); i++) {
      llvm::Module::ModuleFlagEntry &entry = flagEntries[i];
      if (entry.Key->getString() == "Dwarf Version" || entry.Key->getString() == "Debug Info Version") {
        continue;
      }
      m_pModule->addModuleFlag(
        entry.Behavior, entry.Key->getString(),
        cast<ConstantAsMetadata>(entry.Val)->getValue());
    }
  }
}
DebugInfoFinder &DxilModule::GetOrCreateDebugInfoFinder() {
  if (m_pDebugInfoFinder == nullptr) {
    m_pDebugInfoFinder = make_unique<llvm::DebugInfoFinder>();
    m_pDebugInfoFinder->processModule(*m_pModule);
  }
  return *m_pDebugInfoFinder;
}

// Check if the instruction has fast math flags configured to indicate
// the instruction is precise.
// Precise fast math flags means none of the fast math flags are set.
bool DxilModule::HasPreciseFastMathFlags(const Instruction *inst) {
  return isa<FPMathOperator>(inst) && !inst->getFastMathFlags().any();
}

// Set fast math flags configured to indicate the instruction is precise.
void DxilModule::SetPreciseFastMathFlags(llvm::Instruction *inst) {
  assert(isa<FPMathOperator>(inst));
  inst->copyFastMathFlags(FastMathFlags());
}

// True if fast math flags are preserved across serialization/deserialization
// of the dxil module.
//
// We need to check for this when querying fast math flags for preciseness
// otherwise we will be overly conservative by reporting instructions precise
// because their fast math flags were not preserved.
//
// Currently we restrict it to the instruction types that have fast math
// preserved in the bitcode. We can expand this by converting fast math
// flags to dx.precise metadata during serialization and back to fast
// math flags during deserialization.
bool DxilModule::PreservesFastMathFlags(const llvm::Instruction *inst) {
  return
    isa<FPMathOperator>(inst) && (isa<BinaryOperator>(inst) || isa<FCmpInst>(inst));
}

bool DxilModule::IsPrecise(const Instruction *inst) const {
  if (m_ShaderFlags.GetDisableMathRefactoring())
    return true;
  else if (DxilMDHelper::IsMarkedPrecise(inst))
    return true;
  else if (PreservesFastMathFlags(inst))
    return HasPreciseFastMathFlags(inst);
  else
    return false;
}

} // namespace hlsl
