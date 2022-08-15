///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilMetadataHelper.cpp                                                    //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/Support/Global.h"
#include "dxc/DXIL/DxilMetadataHelper.h"
#include "dxc/DXIL/DxilShaderModel.h"
#include "dxc/DXIL/DxilCBuffer.h"
#include "dxc/DXIL/DxilCounters.h"
#include "dxc/DXIL/DxilResource.h"
#include "dxc/DXIL/DxilSampler.h"
#include "dxc/DXIL/DxilSignatureElement.h"
#include "dxc/DXIL/DxilSignature.h"
#include "dxc/DXIL/DxilTypeSystem.h"
#include "dxc/DXIL/DxilFunctionProps.h"
#include "dxc/DXIL/DxilShaderFlags.h"
#include "dxc/DXIL/DxilSubobject.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/DenseSet.h"
#include <array>
#include <algorithm>

#include "dxc/Support/WinIncludes.h"
#include "dxc/Support/WinFunctions.h"

using namespace llvm;
using std::string;
using std::vector;
using std::unique_ptr;

namespace {
void LoadSerializedRootSignature(MDNode *pNode,
                                 std::vector<uint8_t> &SerializedRootSignature,
                                 LLVMContext &Ctx) {
  IFTBOOL(pNode->getNumOperands() == 1, DXC_E_INCORRECT_DXIL_METADATA);
  const MDOperand &MDO = pNode->getOperand(0);

  const ConstantAsMetadata *pMetaData = dyn_cast<ConstantAsMetadata>(MDO.get());
  IFTBOOL(pMetaData != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  const ConstantDataArray *pData =
      dyn_cast<ConstantDataArray>(pMetaData->getValue());
  IFTBOOL(pData != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  IFTBOOL(pData->getElementType() == Type::getInt8Ty(Ctx),
          DXC_E_INCORRECT_DXIL_METADATA);

  SerializedRootSignature.assign(pData->getRawDataValues().begin(),
                                 pData->getRawDataValues().end());
}

MDNode *
EmitSerializedRootSignature(const std::vector<uint8_t> &SerializedRootSignature,
                            LLVMContext &Ctx) {
  if (SerializedRootSignature.empty())
    return nullptr;
  Constant *V = llvm::ConstantDataArray::get(
      Ctx, llvm::ArrayRef<uint8_t>(SerializedRootSignature.data(),
                                   SerializedRootSignature.size()));
  return MDNode::get(Ctx, {ConstantAsMetadata::get(V)});
}

} // namespace

namespace hlsl {

const char DxilMDHelper::kDxilVersionMDName[]                         = "dx.version";
const char DxilMDHelper::kDxilShaderModelMDName[]                     = "dx.shaderModel";
const char DxilMDHelper::kDxilEntryPointsMDName[]                     = "dx.entryPoints";
const char DxilMDHelper::kDxilResourcesMDName[]                       = "dx.resources";
const char DxilMDHelper::kDxilTypeSystemMDName[]                      = "dx.typeAnnotations";
const char DxilMDHelper::kDxilTypeSystemHelperVariablePrefix[]        = "dx.typevar.";
const char DxilMDHelper::kDxilControlFlowHintMDName[]                 = "dx.controlflow.hints";
const char DxilMDHelper::kDxilPreciseAttributeMDName[]                = "dx.precise";
const char DxilMDHelper::kDxilVariableDebugLayoutMDName[]             = "dx.dbg.varlayout";
const char DxilMDHelper::kDxilTempAllocaMDName[]                      = "dx.temp";
const char DxilMDHelper::kDxilNonUniformAttributeMDName[]             = "dx.nonuniform";
const char DxilMDHelper::kHLDxilResourceAttributeMDName[]             = "dx.hl.resource.attribute";
const char DxilMDHelper::kDxilValidatorVersionMDName[]                = "dx.valver";
const char DxilMDHelper::kDxilDxrPayloadAnnotationsMDName[]           = "dx.dxrPayloadAnnotations";

// This named metadata is not valid in final module (should be moved to DxilContainer)
const char DxilMDHelper::kDxilRootSignatureMDName[]                   = "dx.rootSignature";
const char DxilMDHelper::kDxilIntermediateOptionsMDName[]             = "dx.intermediateOptions";
const char DxilMDHelper::kDxilViewIdStateMDName[]                     = "dx.viewIdState";
const char DxilMDHelper::kDxilSubobjectsMDName[]                      = "dx.subobjects";

const char DxilMDHelper::kDxilSourceContentsMDName[]                  = "dx.source.contents";
const char DxilMDHelper::kDxilSourceDefinesMDName[]                   = "dx.source.defines";
const char DxilMDHelper::kDxilSourceMainFileNameMDName[]              = "dx.source.mainFileName";
const char DxilMDHelper::kDxilSourceArgsMDName[]                      = "dx.source.args";

const char DxilMDHelper::kDxilDxcBindingTableMDName[]                 = "dx.binding.table";

const char DxilMDHelper::kDxilSourceContentsOldMDName[]               = "llvm.dbg.contents";
const char DxilMDHelper::kDxilSourceDefinesOldMDName[]                = "llvm.dbg.defines";
const char DxilMDHelper::kDxilSourceMainFileNameOldMDName[]           = "llvm.dbg.mainFileName";
const char DxilMDHelper::kDxilSourceArgsOldMDName[]                   = "llvm.dbg.args";

// This is reflection-only metadata
const char DxilMDHelper::kDxilCountersMDName[]                        = "dx.counters";

static std::array<const char *, 8> DxilMDNames = { {
  DxilMDHelper::kDxilVersionMDName,
  DxilMDHelper::kDxilShaderModelMDName,
  DxilMDHelper::kDxilEntryPointsMDName,
  DxilMDHelper::kDxilResourcesMDName,
  DxilMDHelper::kDxilTypeSystemMDName,
  DxilMDHelper::kDxilValidatorVersionMDName,
  DxilMDHelper::kDxilViewIdStateMDName,
  DxilMDHelper::kDxilDxrPayloadAnnotationsMDName,
}};

DxilMDHelper::DxilMDHelper(Module *pModule, std::unique_ptr<ExtraPropertyHelper> EPH)
: m_Ctx(pModule->getContext())
, m_pModule(pModule)
, m_pSM(nullptr)
, m_ExtraPropertyHelper(std::move(EPH))
, m_ValMajor(1)
, m_ValMinor(0)
, m_MinValMajor(1)
, m_MinValMinor(0)
, m_bExtraMetadata(false)
{
}

DxilMDHelper::~DxilMDHelper() {
}

void DxilMDHelper::SetShaderModel(const ShaderModel *pSM) {
  m_pSM = pSM;
  m_pSM->GetMinValidatorVersion(m_MinValMajor, m_MinValMinor);
  if (DXIL::CompareVersions(m_ValMajor, m_ValMinor, m_MinValMajor, m_MinValMinor) < 0) {
    m_ValMajor = m_MinValMajor;
    m_ValMinor = m_MinValMinor;
  }
  // Validator version 0.0 is not meant for validation or retail driver consumption,
  // and is used for separate reflection.
  // MinVal version drives metadata decisions for compatilbility, so snap this to
  // latest for reflection/no validation case.
  if (DXIL::CompareVersions(m_ValMajor, m_ValMinor, 0, 0) == 0) {
    m_MinValMajor = 0;
    m_MinValMinor = 0;
  }
  if (m_ExtraPropertyHelper) {
    m_ExtraPropertyHelper->m_ValMajor = m_ValMajor;
    m_ExtraPropertyHelper->m_ValMinor = m_ValMinor;
    m_ExtraPropertyHelper->m_MinValMajor = m_MinValMajor;
    m_ExtraPropertyHelper->m_MinValMinor = m_MinValMinor;
  }
}

const ShaderModel *DxilMDHelper::GetShaderModel() const {
  return m_pSM;
}

//
// DXIL version.
//
void DxilMDHelper::EmitDxilVersion(unsigned Major, unsigned Minor) {
  NamedMDNode *pDxilVersionMD = m_pModule->getNamedMetadata(kDxilVersionMDName);
  IFTBOOL(pDxilVersionMD == nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  pDxilVersionMD = m_pModule->getOrInsertNamedMetadata(kDxilVersionMDName);

  Metadata *MDVals[kDxilVersionNumFields];
  MDVals[kDxilVersionMajorIdx] = Uint32ToConstMD(Major);
  MDVals[kDxilVersionMinorIdx] = Uint32ToConstMD(Minor);

  pDxilVersionMD->addOperand(MDNode::get(m_Ctx, MDVals));
}

void DxilMDHelper::LoadDxilVersion(unsigned &Major, unsigned &Minor) {
  NamedMDNode *pDxilVersionMD = m_pModule->getNamedMetadata(kDxilVersionMDName);
  IFTBOOL(pDxilVersionMD != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  IFTBOOL(pDxilVersionMD->getNumOperands() == 1, DXC_E_INCORRECT_DXIL_METADATA);

  MDNode *pVersionMD = pDxilVersionMD->getOperand(0);
  IFTBOOL(pVersionMD->getNumOperands() == kDxilVersionNumFields, DXC_E_INCORRECT_DXIL_METADATA);

  Major = ConstMDToUint32(pVersionMD->getOperand(kDxilVersionMajorIdx));
  Minor = ConstMDToUint32(pVersionMD->getOperand(kDxilVersionMinorIdx));
}

//
// Validator version.
//
void DxilMDHelper::EmitValidatorVersion(unsigned Major, unsigned Minor) {
  NamedMDNode *pDxilValidatorVersionMD = m_pModule->getNamedMetadata(kDxilValidatorVersionMDName);

  // Allow re-writing the validator version, since this can be changed at later points.
  if (pDxilValidatorVersionMD)
    m_pModule->eraseNamedMetadata(pDxilValidatorVersionMD);

  pDxilValidatorVersionMD = m_pModule->getOrInsertNamedMetadata(kDxilValidatorVersionMDName);

  Metadata *MDVals[kDxilVersionNumFields];
  MDVals[kDxilVersionMajorIdx] = Uint32ToConstMD(Major);
  MDVals[kDxilVersionMinorIdx] = Uint32ToConstMD(Minor);

  pDxilValidatorVersionMD->addOperand(MDNode::get(m_Ctx, MDVals));

  m_ValMajor = Major; m_ValMinor = Minor; // Keep these for later use
}

void DxilMDHelper::LoadValidatorVersion(unsigned &Major, unsigned &Minor) {
  NamedMDNode *pDxilValidatorVersionMD = m_pModule->getNamedMetadata(kDxilValidatorVersionMDName);

  if (pDxilValidatorVersionMD == nullptr) {
    // If no validator version metadata, assume 1.0
    Major = 1;
    Minor = 0;
    m_ValMajor = Major; m_ValMinor = Minor; // Keep these for later use
    return;
  }

  IFTBOOL(pDxilValidatorVersionMD->getNumOperands() == 1, DXC_E_INCORRECT_DXIL_METADATA);

  MDNode *pVersionMD = pDxilValidatorVersionMD->getOperand(0);
  IFTBOOL(pVersionMD->getNumOperands() == kDxilVersionNumFields, DXC_E_INCORRECT_DXIL_METADATA);

  Major = ConstMDToUint32(pVersionMD->getOperand(kDxilVersionMajorIdx));
  Minor = ConstMDToUint32(pVersionMD->getOperand(kDxilVersionMinorIdx));
  m_ValMajor = Major; m_ValMinor = Minor; // Keep these for later use
}

//
// DXIL shader model.
//
void DxilMDHelper::EmitDxilShaderModel(const ShaderModel *pSM) {
  NamedMDNode *pShaderModelNamedMD = m_pModule->getNamedMetadata(kDxilShaderModelMDName);
  IFTBOOL(pShaderModelNamedMD == nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  pShaderModelNamedMD = m_pModule->getOrInsertNamedMetadata(kDxilShaderModelMDName);

  Metadata *MDVals[kDxilShaderModelNumFields];
  MDVals[kDxilShaderModelTypeIdx ] = MDString::get(m_Ctx, pSM->GetKindName());
  MDVals[kDxilShaderModelMajorIdx] = Uint32ToConstMD(pSM->GetMajor());
  MDVals[kDxilShaderModelMinorIdx] = Uint32ToConstMD(pSM->GetMinor());

  pShaderModelNamedMD->addOperand(MDNode::get(m_Ctx, MDVals));

  SetShaderModel(pSM);
}

void DxilMDHelper::LoadDxilShaderModel(const ShaderModel *&pSM) {
  NamedMDNode *pShaderModelNamedMD = m_pModule->getNamedMetadata(kDxilShaderModelMDName);
  IFTBOOL(pShaderModelNamedMD != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  IFTBOOL(pShaderModelNamedMD->getNumOperands() == 1, DXC_E_INCORRECT_DXIL_METADATA);

  MDNode *pShaderModelMD = pShaderModelNamedMD->getOperand(0);
  IFTBOOL(pShaderModelMD->getNumOperands() == kDxilShaderModelNumFields, DXC_E_INCORRECT_DXIL_METADATA);

  MDString *pShaderTypeMD = dyn_cast<MDString>(pShaderModelMD->getOperand(kDxilShaderModelTypeIdx));
  IFTBOOL(pShaderTypeMD != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  unsigned Major = ConstMDToUint32(pShaderModelMD->getOperand(kDxilShaderModelMajorIdx));
  unsigned Minor = ConstMDToUint32(pShaderModelMD->getOperand(kDxilShaderModelMinorIdx));
  string ShaderModelName = pShaderTypeMD->getString().str();
  ShaderModelName += "_" + std::to_string(Major) + "_" +
    (Minor == ShaderModel::kOfflineMinor ? "x" : std::to_string(Minor));
  pSM = ShaderModel::GetByName(ShaderModelName.c_str());
  if (!pSM->IsValidForDxil()) {
    char ErrorMsgTxt[40];
    StringCchPrintfA(ErrorMsgTxt, _countof(ErrorMsgTxt),
                     "Unknown shader model '%s'", ShaderModelName.c_str());
    string ErrorMsg(ErrorMsgTxt);
    throw hlsl::Exception(DXC_E_INCORRECT_DXIL_METADATA, ErrorMsg);
  }
  SetShaderModel(pSM);
}

//
// intermediate options.
//
void DxilMDHelper::EmitDxilIntermediateOptions(uint32_t flags) {
  if (flags == 0) return;

  NamedMDNode *pIntermediateOptionsNamedMD = m_pModule->getNamedMetadata(kDxilIntermediateOptionsMDName);
  IFTBOOL(pIntermediateOptionsNamedMD == nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  pIntermediateOptionsNamedMD = m_pModule->getOrInsertNamedMetadata(kDxilIntermediateOptionsMDName);

  pIntermediateOptionsNamedMD->addOperand(
    MDNode::get(m_Ctx, { Uint32ToConstMD(kDxilIntermediateOptionsFlags), Uint32ToConstMD(flags) }));
}

void DxilMDHelper::LoadDxilIntermediateOptions(uint32_t &flags) {
  flags = 0;

  NamedMDNode *pIntermediateOptionsNamedMD = m_pModule->getNamedMetadata(kDxilIntermediateOptionsMDName);
  if (pIntermediateOptionsNamedMD == nullptr) return;

  for (unsigned i = 0; i < pIntermediateOptionsNamedMD->getNumOperands(); i++) {
    MDTuple *pEntry = dyn_cast<MDTuple>(pIntermediateOptionsNamedMD->getOperand(i));
    IFTBOOL(pEntry != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
    IFTBOOL(pEntry->getNumOperands() >= 1, DXC_E_INCORRECT_DXIL_METADATA);

    uint32_t id = ConstMDToUint32(pEntry->getOperand(0));
    switch (id) {
    case kDxilIntermediateOptionsFlags:
      IFTBOOL(pEntry->getNumOperands() == 2, DXC_E_INCORRECT_DXIL_METADATA);
      flags = ConstMDToUint32(pEntry->getOperand(1));
      break;

    default: throw hlsl::Exception(DXC_E_INCORRECT_DXIL_METADATA, "Unrecognized intermediate options metadata");
    }
  }
}

//
// Entry points.
//
void DxilMDHelper::EmitDxilEntryPoints(vector<MDNode *> &MDEntries) {
  DXASSERT(MDEntries.size() == 1 || GetShaderModel()->IsLib(),
           "only one entry point is supported for now");
  NamedMDNode *pEntryPointsNamedMD = m_pModule->getNamedMetadata(kDxilEntryPointsMDName);
  IFTBOOL(pEntryPointsNamedMD == nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  pEntryPointsNamedMD = m_pModule->getOrInsertNamedMetadata(kDxilEntryPointsMDName);

  for (size_t i = 0; i < MDEntries.size(); i++) {
    pEntryPointsNamedMD->addOperand(MDEntries[i]);
  }
}

void DxilMDHelper::UpdateDxilEntryPoints(vector<MDNode *> &MDEntries) {
  DXASSERT(MDEntries.size() == 1, "only one entry point is supported for now");
  NamedMDNode *pEntryPointsNamedMD =
      m_pModule->getNamedMetadata(kDxilEntryPointsMDName);
  IFTBOOL(pEntryPointsNamedMD != nullptr, DXC_E_INCORRECT_DXIL_METADATA);

  for (size_t i = 0; i < MDEntries.size(); i++) {
    pEntryPointsNamedMD->setOperand(i, MDEntries[i]);
  }
}

const NamedMDNode *DxilMDHelper::GetDxilEntryPoints() {
  NamedMDNode *pEntryPointsNamedMD = m_pModule->getNamedMetadata(kDxilEntryPointsMDName);
  IFTBOOL(pEntryPointsNamedMD != nullptr, DXC_E_INCORRECT_DXIL_METADATA);

  return pEntryPointsNamedMD;
}

MDTuple *DxilMDHelper::EmitDxilEntryPointTuple(Function *pFunc, const string &Name, 
                                               MDTuple *pSignatures, MDTuple *pResources,
                                               MDTuple *pProperties) {
  Metadata *MDVals[kDxilEntryPointNumFields];
  MDVals[kDxilEntryPointFunction  ] = pFunc ? ValueAsMetadata::get(pFunc) : nullptr;
  MDVals[kDxilEntryPointName      ] = MDString::get(m_Ctx, Name.c_str());
  MDVals[kDxilEntryPointSignatures] = pSignatures;
  MDVals[kDxilEntryPointResources ] = pResources;
  MDVals[kDxilEntryPointProperties] = pProperties;
  return MDNode::get(m_Ctx, MDVals);
}

void DxilMDHelper::GetDxilEntryPoint(const MDNode *MDO, Function *&pFunc, string &Name,
                                     const MDOperand *&pSignatures, const MDOperand *&pResources,
                                     const MDOperand *&pProperties) {
  IFTBOOL(MDO != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  const MDTuple *pTupleMD = dyn_cast<MDTuple>(MDO);
  IFTBOOL(pTupleMD != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  IFTBOOL(pTupleMD->getNumOperands() == kDxilEntryPointNumFields, DXC_E_INCORRECT_DXIL_METADATA);

  // Retrieve entry function symbol.
  const MDOperand &MDOFunc = pTupleMD->getOperand(kDxilEntryPointFunction);
  if (MDOFunc.get() != nullptr) {
    ValueAsMetadata *pValueFunc = dyn_cast<ValueAsMetadata>(MDOFunc.get());
    IFTBOOL(pValueFunc != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
    pFunc = dyn_cast<Function>(pValueFunc->getValue());
    IFTBOOL(pFunc != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  } else {
    pFunc = nullptr;  // pass-through CP.
  }

  // Retrieve entry function name.
  const MDOperand &MDOName = pTupleMD->getOperand(kDxilEntryPointName);
  IFTBOOL(MDOName.get() != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  MDString *pMDName = dyn_cast<MDString>(MDOName);
  IFTBOOL(pMDName != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  Name = pMDName->getString().str();

  pSignatures = &pTupleMD->getOperand(kDxilEntryPointSignatures);
  pResources  = &pTupleMD->getOperand(kDxilEntryPointResources );
  pProperties = &pTupleMD->getOperand(kDxilEntryPointProperties);
}

//
// Signatures.
//
MDTuple *DxilMDHelper::EmitDxilSignatures(const DxilEntrySignature &EntrySig) {
  MDTuple *pSignatureTupleMD = nullptr;

  const DxilSignature &InputSig = EntrySig.InputSignature;
  const DxilSignature &OutputSig = EntrySig.OutputSignature;
  const DxilSignature &PCPSig = EntrySig.PatchConstOrPrimSignature;

  if (!InputSig.GetElements().empty() || !OutputSig.GetElements().empty() || !PCPSig.GetElements().empty()) {
    Metadata *MDVals[kDxilNumSignatureFields];
    MDVals[kDxilInputSignature]         = EmitSignatureMetadata(InputSig);
    MDVals[kDxilOutputSignature]        = EmitSignatureMetadata(OutputSig);
    MDVals[kDxilPatchConstantSignature] = EmitSignatureMetadata(PCPSig);

    pSignatureTupleMD = MDNode::get(m_Ctx, MDVals);
  }

  return pSignatureTupleMD;
}

void DxilMDHelper::EmitRootSignature(
    std::vector<uint8_t> &SerializedRootSignature) {
  if (SerializedRootSignature.empty()) {
    return;
  }

  MDNode *Node = EmitSerializedRootSignature(SerializedRootSignature, m_Ctx);

  NamedMDNode *pRootSignatureNamedMD = m_pModule->getNamedMetadata(kDxilRootSignatureMDName);
  IFTBOOL(pRootSignatureNamedMD == nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  pRootSignatureNamedMD = m_pModule->getOrInsertNamedMetadata(kDxilRootSignatureMDName);
  pRootSignatureNamedMD->addOperand(Node);
  return ;
}

void DxilMDHelper::LoadDxilSignatures(const MDOperand &MDO, DxilEntrySignature &EntrySig) {
  if (MDO.get() == nullptr)
    return;
  DxilSignature &InputSig = EntrySig.InputSignature;
  DxilSignature &OutputSig = EntrySig.OutputSignature;
  DxilSignature &PCPSig = EntrySig.PatchConstOrPrimSignature;
  const MDTuple *pTupleMD = dyn_cast<MDTuple>(MDO.get());
  IFTBOOL(pTupleMD != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  IFTBOOL(pTupleMD->getNumOperands() == kDxilNumSignatureFields, DXC_E_INCORRECT_DXIL_METADATA);

  LoadSignatureMetadata(pTupleMD->getOperand(kDxilInputSignature),         InputSig);
  LoadSignatureMetadata(pTupleMD->getOperand(kDxilOutputSignature),        OutputSig);
  LoadSignatureMetadata(pTupleMD->getOperand(kDxilPatchConstantSignature), PCPSig);
}

MDTuple *DxilMDHelper::EmitSignatureMetadata(const DxilSignature &Sig) {
  auto &Elements = Sig.GetElements();
  if (Elements.empty())
    return nullptr;

  vector<Metadata *> MDVals;
  for (size_t i = 0; i < Elements.size(); i++) {
    MDVals.emplace_back(EmitSignatureElement(*Elements[i]));
  }

  return MDNode::get(m_Ctx, MDVals);
}

void DxilMDHelper::LoadSignatureMetadata(const MDOperand &MDO, DxilSignature &Sig) {
  if (MDO.get() == nullptr)
    return;

  const MDTuple *pTupleMD = dyn_cast<MDTuple>(MDO.get());
  IFTBOOL(pTupleMD != nullptr, DXC_E_INCORRECT_DXIL_METADATA);

  for (unsigned i = 0; i < pTupleMD->getNumOperands(); i++) {
    unique_ptr<DxilSignatureElement> pSE(Sig.CreateElement());
    LoadSignatureElement(pTupleMD->getOperand(i), *pSE.get());
    Sig.AppendElement(std::move(pSE));
  }
}

void DxilMDHelper::LoadRootSignature(std::vector<uint8_t> &SerializedRootSignature) {
  NamedMDNode *pRootSignatureNamedMD = m_pModule->getNamedMetadata(kDxilRootSignatureMDName);
  if(!pRootSignatureNamedMD)
    return;

  IFTBOOL(pRootSignatureNamedMD->getNumOperands() == 1, DXC_E_INCORRECT_DXIL_METADATA);

  MDNode *pNode = pRootSignatureNamedMD->getOperand(0);
  LoadSerializedRootSignature(pNode, SerializedRootSignature, m_Ctx);
}

static const MDTuple *CastToTupleOrNull(const MDOperand &MDO) {
  if (MDO.get() == nullptr)
    return nullptr;

  const MDTuple *pTupleMD = dyn_cast<MDTuple>(MDO.get());
  IFTBOOL(pTupleMD != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  return pTupleMD;
}

MDTuple *DxilMDHelper::EmitSignatureElement(const DxilSignatureElement &SE) {
  Metadata *MDVals[kDxilSignatureElementNumFields];

  MDVals[kDxilSignatureElementID            ] = Uint32ToConstMD(SE.GetID());
  MDVals[kDxilSignatureElementName          ] = MDString::get(m_Ctx, SE.GetName());
  MDVals[kDxilSignatureElementType          ] = Uint8ToConstMD((uint8_t)SE.GetCompType().GetKind());
  MDVals[kDxilSignatureElementSystemValue   ] = Uint8ToConstMD((uint8_t)SE.GetKind());
  MDVals[kDxilSignatureElementIndexVector   ] = Uint32VectorToConstMDTuple(SE.GetSemanticIndexVec());
  MDVals[kDxilSignatureElementInterpMode    ] = Uint8ToConstMD((uint8_t)SE.GetInterpolationMode()->GetKind());
  MDVals[kDxilSignatureElementRows          ] = Uint32ToConstMD(SE.GetRows());
  MDVals[kDxilSignatureElementCols          ] = Uint8ToConstMD((uint8_t)SE.GetCols());
  MDVals[kDxilSignatureElementStartRow      ] = Int32ToConstMD(SE.GetStartRow());
  MDVals[kDxilSignatureElementStartCol      ] = Int8ToConstMD((int8_t)SE.GetStartCol());

  // Name-value list of extended properties.
  MDVals[kDxilSignatureElementNameValueList] = nullptr;
  vector<Metadata *> MDExtraVals;
  m_ExtraPropertyHelper->EmitSignatureElementProperties(SE, MDExtraVals);
  if (!MDExtraVals.empty()) {
    MDVals[kDxilSignatureElementNameValueList] = MDNode::get(m_Ctx, MDExtraVals);
  }

  // NOTE: when extra properties for signature elements are needed, extend ExtraPropertyHelper.

  return MDNode::get(m_Ctx, MDVals);
}

void DxilMDHelper::LoadSignatureElement(const MDOperand &MDO, DxilSignatureElement &SE) {
  IFTBOOL(MDO.get() != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  const MDTuple *pTupleMD = dyn_cast<MDTuple>(MDO.get());
  IFTBOOL(pTupleMD != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  IFTBOOL(pTupleMD->getNumOperands() == kDxilSignatureElementNumFields, DXC_E_INCORRECT_DXIL_METADATA);

  unsigned ID = ConstMDToUint32(                      pTupleMD->getOperand(kDxilSignatureElementID));
  MDString *pName = dyn_cast<MDString>(               pTupleMD->getOperand(kDxilSignatureElementName));
  CompType CT = CompType(ConstMDToUint8(              pTupleMD->getOperand(kDxilSignatureElementType)));
  DXIL::SemanticKind SemKind = 
    (DXIL::SemanticKind)ConstMDToUint8(               pTupleMD->getOperand(kDxilSignatureElementSystemValue));
  MDTuple *pSemanticIndexVectorMD = dyn_cast<MDTuple>(pTupleMD->getOperand(kDxilSignatureElementIndexVector));
  InterpolationMode IM(ConstMDToUint8(                pTupleMD->getOperand(kDxilSignatureElementInterpMode)));
  unsigned NumRows = ConstMDToUint32(                 pTupleMD->getOperand(kDxilSignatureElementRows));
  uint8_t NumCols = ConstMDToUint8(                   pTupleMD->getOperand(kDxilSignatureElementCols));
  int32_t StartRow = ConstMDToInt32(                  pTupleMD->getOperand(kDxilSignatureElementStartRow));
  int8_t StartCol = ConstMDToInt8(                    pTupleMD->getOperand(kDxilSignatureElementStartCol));

  IFTBOOL(pName != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  IFTBOOL(pSemanticIndexVectorMD != nullptr, DXC_E_INCORRECT_DXIL_METADATA);

  vector<unsigned> SemanticIndexVector;
  ConstMDTupleToUint32Vector(pSemanticIndexVectorMD, SemanticIndexVector);

  SE.Initialize(pName->getString(), CT, IM, NumRows, NumCols, StartRow, StartCol, ID, SemanticIndexVector);
  SE.SetKind(SemKind);

  // For case a system semantic don't have index, add 0 for it.
  if (SemanticIndexVector.empty() && !SE.IsArbitrary()) {
    SE.SetSemanticIndexVec({0});
  }
  // Name-value list of extended properties.
  m_ExtraPropertyHelper->m_bExtraMetadata = false;
  m_ExtraPropertyHelper->LoadSignatureElementProperties(pTupleMD->getOperand(kDxilSignatureElementNameValueList), SE);
  m_bExtraMetadata |= m_ExtraPropertyHelper->m_bExtraMetadata;
}

//
// Resources.
//
MDTuple *DxilMDHelper::EmitDxilResourceTuple(MDTuple *pSRVs, MDTuple *pUAVs, 
                                             MDTuple *pCBuffers, MDTuple *pSamplers) {
  DXASSERT(pSRVs != nullptr || pUAVs != nullptr || pCBuffers != nullptr || pSamplers != nullptr, "resource tuple should not be emitted if there are no resources");
  Metadata *MDVals[kDxilNumResourceFields];
  MDVals[kDxilResourceSRVs    ] = pSRVs;
  MDVals[kDxilResourceUAVs    ] = pUAVs;
  MDVals[kDxilResourceCBuffers] = pCBuffers;
  MDVals[kDxilResourceSamplers] = pSamplers;
  MDTuple *pTupleMD = MDNode::get(m_Ctx, MDVals);

  return pTupleMD;
}

void DxilMDHelper::EmitDxilResources(llvm::MDTuple *pDxilResourceTuple) {
  NamedMDNode *pResourcesNamedMD = m_pModule->getNamedMetadata(kDxilResourcesMDName);
  IFTBOOL(pResourcesNamedMD == nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  pResourcesNamedMD = m_pModule->getOrInsertNamedMetadata(kDxilResourcesMDName);
  pResourcesNamedMD->addOperand(pDxilResourceTuple);
}

void DxilMDHelper::UpdateDxilResources(llvm::MDTuple *pDxilResourceTuple) {
  NamedMDNode *pResourcesNamedMD =
      m_pModule->getNamedMetadata(kDxilResourcesMDName);
  if (!pResourcesNamedMD) {
    pResourcesNamedMD =
        m_pModule->getOrInsertNamedMetadata(kDxilResourcesMDName);
  }
  if (pDxilResourceTuple) {
    if (pResourcesNamedMD->getNumOperands() != 0) {
      pResourcesNamedMD->setOperand(0, pDxilResourceTuple);
    }
    else {
      pResourcesNamedMD->addOperand(pDxilResourceTuple);
    }

  } else {
    m_pModule->eraseNamedMetadata(pResourcesNamedMD);
  }
}

void DxilMDHelper::GetDxilResources(const MDOperand &MDO, const MDTuple *&pSRVs,
                                    const MDTuple *&pUAVs, const MDTuple *&pCBuffers,
                                    const MDTuple *&pSamplers) {
  IFTBOOL(MDO.get() != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  const MDTuple *pTupleMD = dyn_cast<MDTuple>(MDO.get());
  IFTBOOL(pTupleMD != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  IFTBOOL(pTupleMD->getNumOperands() == kDxilNumResourceFields, DXC_E_INCORRECT_DXIL_METADATA);

  pSRVs     = CastToTupleOrNull(pTupleMD->getOperand(kDxilResourceSRVs    ));
  pUAVs     = CastToTupleOrNull(pTupleMD->getOperand(kDxilResourceUAVs    ));
  pCBuffers = CastToTupleOrNull(pTupleMD->getOperand(kDxilResourceCBuffers));
  pSamplers = CastToTupleOrNull(pTupleMD->getOperand(kDxilResourceSamplers));
}

void DxilMDHelper::EmitDxilResourceBase(const DxilResourceBase &R, Metadata *ppMDVals[]) {
  ppMDVals[kDxilResourceBaseID        ] = Uint32ToConstMD(R.GetID());
  Constant *GlobalSymbol = R.GetGlobalSymbol();
  // For sm66+, global symbol will be mutated into handle type.
  // Save hlsl type by generate bitcast on global symbol.
  if (m_pSM->IsSM66Plus()) {
    Type *HLSLTy = R.GetHLSLType();
    if (HLSLTy && HLSLTy != GlobalSymbol->getType())
      GlobalSymbol = cast<Constant>(
          ConstantExpr::getCast(Instruction::BitCast, GlobalSymbol, HLSLTy));
  }
  ppMDVals[kDxilResourceBaseVariable  ] = ValueAsMetadata::get(GlobalSymbol);
  ppMDVals[kDxilResourceBaseName      ] = MDString::get(m_Ctx, R.GetGlobalName());
  ppMDVals[kDxilResourceBaseSpaceID   ] = Uint32ToConstMD(R.GetSpaceID());
  ppMDVals[kDxilResourceBaseLowerBound] = Uint32ToConstMD(R.GetLowerBound());
  ppMDVals[kDxilResourceBaseRangeSize ] = Uint32ToConstMD(R.GetRangeSize());
}

void DxilMDHelper::LoadDxilResourceBase(const MDOperand &MDO, DxilResourceBase &R) {
  IFTBOOL(MDO.get() != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  const MDTuple *pTupleMD = dyn_cast<MDTuple>(MDO.get());
  IFTBOOL(pTupleMD != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  IFTBOOL(pTupleMD->getNumOperands() >= kDxilResourceBaseNumFields, DXC_E_INCORRECT_DXIL_METADATA);

  R.SetID(ConstMDToUint32(pTupleMD->getOperand(kDxilResourceBaseID)));
  Constant *GlobalSymbol = dyn_cast<Constant>(ValueMDToValue(pTupleMD->getOperand(kDxilResourceBaseVariable)));
  // For sm66+, global symbol will be mutated into handle type.
  // Read hlsl type and global symbol from bitcast.
  if (m_pSM->IsSM66Plus()) {
    // Before mutate, there's no bitcast. After GlobalSymbol changed into undef,
    // there's no bitcast too. Bitcast will only exist when global symbol is
    // mutated into handle and not changed into undef for lib linking.
    if (BitCastOperator *BCO = dyn_cast<BitCastOperator>(GlobalSymbol)) {
      GlobalSymbol = cast<Constant>(BCO->getOperand(0));
      R.SetHLSLType(BCO->getType());
    }
  }
  R.SetGlobalSymbol(GlobalSymbol);

  R.SetGlobalName(StringMDToString(pTupleMD->getOperand(kDxilResourceBaseName)));
  R.SetSpaceID(ConstMDToUint32(pTupleMD->getOperand(kDxilResourceBaseSpaceID)));
  R.SetLowerBound(ConstMDToUint32(pTupleMD->getOperand(kDxilResourceBaseLowerBound)));
  R.SetRangeSize(ConstMDToUint32(pTupleMD->getOperand(kDxilResourceBaseRangeSize)));
}

MDTuple *DxilMDHelper::EmitDxilSRV(const DxilResource &SRV) {
  Metadata *MDVals[kDxilSRVNumFields];

  EmitDxilResourceBase(SRV, &MDVals[0]);

  // SRV-specific fields.
  MDVals[kDxilSRVShape        ] = Uint32ToConstMD((unsigned)SRV.GetKind());
  MDVals[kDxilSRVSampleCount  ] = Uint32ToConstMD(SRV.GetSampleCount());

  // Name-value list of extended properties.
  MDVals[kDxilSRVNameValueList] = nullptr;
  vector<Metadata *> MDExtraVals;
  m_ExtraPropertyHelper->EmitSRVProperties(SRV, MDExtraVals);
  if (!MDExtraVals.empty()) {
    MDVals[kDxilSRVNameValueList] = MDNode::get(m_Ctx, MDExtraVals);
  }

  return MDNode::get(m_Ctx, MDVals);
}

void DxilMDHelper::LoadDxilSRV(const MDOperand &MDO, DxilResource &SRV) {
  IFTBOOL(MDO.get() != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  const MDTuple *pTupleMD = dyn_cast<MDTuple>(MDO.get());
  IFTBOOL(pTupleMD != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  IFTBOOL(pTupleMD->getNumOperands() == kDxilSRVNumFields, DXC_E_INCORRECT_DXIL_METADATA);

  SRV.SetRW(false);

  LoadDxilResourceBase(MDO, SRV);

  // SRV-specific fields.
  SRV.SetKind((DxilResource::Kind)ConstMDToUint32(pTupleMD->getOperand(kDxilSRVShape)));
  SRV.SetSampleCount(ConstMDToUint32(pTupleMD->getOperand(kDxilSRVSampleCount)));

  // Name-value list of extended properties.
  m_ExtraPropertyHelper->m_bExtraMetadata = false;
  m_ExtraPropertyHelper->LoadSRVProperties(pTupleMD->getOperand(kDxilSRVNameValueList), SRV);
  m_bExtraMetadata |= m_ExtraPropertyHelper->m_bExtraMetadata;
}

MDTuple *DxilMDHelper::EmitDxilUAV(const DxilResource &UAV) {
  Metadata *MDVals[kDxilUAVNumFields];

  EmitDxilResourceBase(UAV, &MDVals[0]);

  // UAV-specific fields.
  MDVals[kDxilUAVShape                ] = Uint32ToConstMD((unsigned)UAV.GetKind());
  MDVals[kDxilUAVGloballyCoherent     ] = BoolToConstMD(UAV.IsGloballyCoherent());
  MDVals[kDxilUAVCounter              ] = BoolToConstMD(UAV.HasCounter());
  MDVals[kDxilUAVRasterizerOrderedView] = BoolToConstMD(UAV.IsROV());

  // Name-value list of extended properties.
  MDVals[kDxilUAVNameValueList        ] = nullptr;
  vector<Metadata *> MDExtraVals;
  m_ExtraPropertyHelper->EmitUAVProperties(UAV, MDExtraVals);
  if (!MDExtraVals.empty()) {
    MDVals[kDxilUAVNameValueList] = MDNode::get(m_Ctx, MDExtraVals);
  }

  return MDNode::get(m_Ctx, MDVals);
}

void DxilMDHelper::LoadDxilUAV(const MDOperand &MDO, DxilResource &UAV) {
  IFTBOOL(MDO.get() != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  const MDTuple *pTupleMD = dyn_cast<MDTuple>(MDO.get());
  IFTBOOL(pTupleMD != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  IFTBOOL(pTupleMD->getNumOperands() == kDxilUAVNumFields, DXC_E_INCORRECT_DXIL_METADATA);

  UAV.SetRW(true);

  LoadDxilResourceBase(MDO, UAV);

  // UAV-specific fields.
  UAV.SetKind((DxilResource::Kind)ConstMDToUint32(pTupleMD->getOperand(kDxilUAVShape)));
  UAV.SetGloballyCoherent(ConstMDToBool(pTupleMD->getOperand(kDxilUAVGloballyCoherent)));
  UAV.SetHasCounter(ConstMDToBool(pTupleMD->getOperand(kDxilUAVCounter)));
  UAV.SetROV(ConstMDToBool(pTupleMD->getOperand(kDxilUAVRasterizerOrderedView)));

  // Name-value list of extended properties.
  m_ExtraPropertyHelper->m_bExtraMetadata = false;
  m_ExtraPropertyHelper->LoadUAVProperties(pTupleMD->getOperand(kDxilUAVNameValueList), UAV);
  m_bExtraMetadata |= m_ExtraPropertyHelper->m_bExtraMetadata;
}

MDTuple *DxilMDHelper::EmitDxilCBuffer(const DxilCBuffer &CB) {
  Metadata *MDVals[kDxilCBufferNumFields];

  EmitDxilResourceBase(CB, &MDVals[0]);

  // CBuffer-specific fields.
  // CBuffer size in bytes.
  MDVals[kDxilCBufferSizeInBytes  ] = Uint32ToConstMD(CB.GetSize());

  // Name-value list of extended properties.
  MDVals[kDxilCBufferNameValueList] = nullptr;
  vector<Metadata *> MDExtraVals;
  m_ExtraPropertyHelper->EmitCBufferProperties(CB, MDExtraVals);
  if (!MDExtraVals.empty()) {
    MDVals[kDxilCBufferNameValueList] = MDNode::get(m_Ctx, MDExtraVals);
  }

  return MDNode::get(m_Ctx, MDVals);
}

void DxilMDHelper::LoadDxilCBuffer(const MDOperand &MDO, DxilCBuffer &CB) {
  IFTBOOL(MDO.get() != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  const MDTuple *pTupleMD = dyn_cast<MDTuple>(MDO.get());
  IFTBOOL(pTupleMD != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  IFTBOOL(pTupleMD->getNumOperands() == kDxilCBufferNumFields, DXC_E_INCORRECT_DXIL_METADATA);

  LoadDxilResourceBase(MDO, CB);

  // CBuffer-specific fields.
  CB.SetSize(ConstMDToUint32(pTupleMD->getOperand(kDxilCBufferSizeInBytes)));

  // Name-value list of extended properties.
  m_ExtraPropertyHelper->m_bExtraMetadata = false;
  m_ExtraPropertyHelper->LoadCBufferProperties(pTupleMD->getOperand(kDxilCBufferNameValueList), CB);
  m_bExtraMetadata |= m_ExtraPropertyHelper->m_bExtraMetadata;
}

void DxilMDHelper::EmitDxilTypeSystem(DxilTypeSystem &TypeSystem, vector<GlobalVariable*> &LLVMUsed) {
  auto &TypeMap = TypeSystem.GetStructAnnotationMap();
  vector<Metadata *> MDVals;
  MDVals.emplace_back(Uint32ToConstMD(kDxilTypeSystemStructTag)); // Tag
  unsigned GVIdx = 0;
  for (auto it = TypeMap.begin(); it != TypeMap.end(); ++it, GVIdx++) {
    StructType *pStructType = const_cast<StructType *>(it->first);
    DxilStructAnnotation *pA = it->second.get();
    // Don't emit type annotation for empty struct.
    if (pA->IsEmptyStruct())
      continue;
    // Emit struct type field annotations.
    Metadata *pMD = EmitDxilStructAnnotation(*pA);

    MDVals.push_back(ValueAsMetadata::get(UndefValue::get(pStructType)));
    MDVals.push_back(pMD);
  }

  auto &FuncMap = TypeSystem.GetFunctionAnnotationMap();
  vector<Metadata *> MDFuncVals;
  MDFuncVals.emplace_back(Uint32ToConstMD(kDxilTypeSystemFunctionTag)); // Tag
  for (auto it = FuncMap.begin(); it != FuncMap.end(); ++it) {
    DxilFunctionAnnotation *pA = it->second.get();
    MDFuncVals.push_back(ValueAsMetadata::get(const_cast<Function*>(pA->GetFunction())));
    // Emit function annotations.

   Metadata *pMD;
    pMD = EmitDxilFunctionAnnotation(*pA);
    MDFuncVals.push_back(pMD);
  }

  NamedMDNode *pDxilTypeAnnotationsMD = m_pModule->getNamedMetadata(kDxilTypeSystemMDName);
  if (pDxilTypeAnnotationsMD != nullptr) {
    m_pModule->eraseNamedMetadata(pDxilTypeAnnotationsMD);
  }

  if (MDVals.size() > 1) {
    pDxilTypeAnnotationsMD = m_pModule->getOrInsertNamedMetadata(kDxilTypeSystemMDName);

    pDxilTypeAnnotationsMD->addOperand(MDNode::get(m_Ctx, MDVals));
  }
  if (MDFuncVals.size() > 1) {
    NamedMDNode *pDxilTypeAnnotationsMD = m_pModule->getNamedMetadata(kDxilTypeSystemMDName);
    if (pDxilTypeAnnotationsMD == nullptr)
      pDxilTypeAnnotationsMD = m_pModule->getOrInsertNamedMetadata(kDxilTypeSystemMDName);

    pDxilTypeAnnotationsMD->addOperand(MDNode::get(m_Ctx, MDFuncVals));
  }
}

void DxilMDHelper::LoadDxilTypeSystemNode(const llvm::MDTuple &MDT,
                                          DxilTypeSystem &TypeSystem) {

  unsigned Tag = ConstMDToUint32(MDT.getOperand(0));
  if (Tag == kDxilTypeSystemStructTag) {
    IFTBOOL((MDT.getNumOperands() & 0x1) == 1, DXC_E_INCORRECT_DXIL_METADATA);

    for (unsigned i = 1; i < MDT.getNumOperands(); i += 2) {
      Constant *pGV =
          dyn_cast<Constant>(ValueMDToValue(MDT.getOperand(i)));
      IFTBOOL(pGV != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
      StructType *pGVType =
          dyn_cast<StructType>(pGV->getType());
      IFTBOOL(pGVType != nullptr, DXC_E_INCORRECT_DXIL_METADATA);

      DxilStructAnnotation *pSA = TypeSystem.AddStructAnnotation(pGVType);
      LoadDxilStructAnnotation(MDT.getOperand(i + 1), *pSA);
      TypeSystem.FinishStructAnnotation(*pSA);
    }
  } else {
    IFTBOOL((MDT.getNumOperands() & 0x1) == 1, DXC_E_INCORRECT_DXIL_METADATA);
    for (unsigned i = 1; i < MDT.getNumOperands(); i += 2) {
      Function *F = dyn_cast<Function>(ValueMDToValue(MDT.getOperand(i)));
      DxilFunctionAnnotation *pFA = TypeSystem.AddFunctionAnnotation(F);
      LoadDxilFunctionAnnotation(MDT.getOperand(i + 1), *pFA);
      TypeSystem.FinishFunctionAnnotation(*pFA);
    }
  }
}

void DxilMDHelper::LoadDxilTypeSystem(DxilTypeSystem &TypeSystem) {
  NamedMDNode *pDxilTypeAnnotationsMD = m_pModule->getNamedMetadata(kDxilTypeSystemMDName);
  if (pDxilTypeAnnotationsMD == nullptr)
    return;

  IFTBOOL(pDxilTypeAnnotationsMD->getNumOperands() <= 2, DXC_E_INCORRECT_DXIL_METADATA);
  for (unsigned i = 0; i < pDxilTypeAnnotationsMD->getNumOperands(); i++) {
    const MDTuple *pTupleMD = dyn_cast<MDTuple>(pDxilTypeAnnotationsMD->getOperand(i));
    IFTBOOL(pTupleMD != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
    LoadDxilTypeSystemNode(*pTupleMD, TypeSystem);
  }
}

void DxilMDHelper::EmitDxrPayloadAnnotations(DxilTypeSystem &TypeSystem) {
  auto &TypeMap = TypeSystem.GetPayloadAnnotationMap();
  vector<Metadata *> MDVals;
  MDVals.emplace_back(Uint32ToConstMD(kDxilPayloadAnnotationStructTag)); // Tag
  unsigned GVIdx = 0;
  for (auto it = TypeMap.begin(); it != TypeMap.end(); ++it, GVIdx++) {
    StructType *pStructType = const_cast<StructType *>(it->first);
    DxilPayloadAnnotation *pA = it->second.get();
    // Emit struct type field annotations.
    Metadata *pMD = EmitDxrPayloadStructAnnotation(*pA);

    MDVals.push_back(ValueAsMetadata::get(UndefValue::get(pStructType)));
    MDVals.push_back(pMD);
  }

  NamedMDNode *pDxrPayloadAnnotationsMD = m_pModule->getNamedMetadata(kDxilDxrPayloadAnnotationsMDName);
  if (pDxrPayloadAnnotationsMD != nullptr) {
    m_pModule->eraseNamedMetadata(pDxrPayloadAnnotationsMD);
  }

  if (MDVals.size() > 1) {
    pDxrPayloadAnnotationsMD = m_pModule->getOrInsertNamedMetadata(kDxilDxrPayloadAnnotationsMDName);
    pDxrPayloadAnnotationsMD->addOperand(MDNode::get(m_Ctx, MDVals));
  }
}

Metadata *
DxilMDHelper::EmitDxrPayloadStructAnnotation(const DxilPayloadAnnotation &SA) {
  vector<Metadata *> MDVals;
  MDVals.reserve(SA.GetNumFields());
  MDVals.resize(SA.GetNumFields());

  const StructType* STy = SA.GetStructType();
  for (unsigned i = 0; i < SA.GetNumFields(); i++) {
    MDVals[i] = EmitDxrPayloadFieldAnnotation(SA.GetFieldAnnotation(i), STy->getElementType(i));
  }

  return MDNode::get(m_Ctx, MDVals);
}

void DxilMDHelper::LoadDxrPayloadAccessQualifiers(const MDOperand &MDO,
                                               DxilPayloadFieldAnnotation &FA) {
  unsigned fieldBitmask = ConstMDToInt32(MDO);
  if (fieldBitmask & ~DXIL::PayloadAccessQualifierValidMask) {
    DXASSERT(false, "Unknown payload access qualifier bits set");
    m_bExtraMetadata = true;
  }
  fieldBitmask &= DXIL::PayloadAccessQualifierValidMask;
  FA.SetPayloadFieldQualifierMask(fieldBitmask);
}

void DxilMDHelper::LoadDxrPayloadFieldAnnoation(
    const MDOperand &MDO, DxilPayloadFieldAnnotation &FA) {
  IFTBOOL(MDO.get() != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  const MDTuple *pTupleMD = dyn_cast<MDTuple>(MDO.get()); // Tag-Value list.
  IFTBOOL(pTupleMD != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  IFTBOOL((pTupleMD->getNumOperands() & 0x1) == 0, DXC_E_INCORRECT_DXIL_METADATA);

  for (unsigned i = 0; i < pTupleMD->getNumOperands(); i += 2) {
    unsigned Tag = ConstMDToUint32(pTupleMD->getOperand(i));
    const MDOperand &MDO = pTupleMD->getOperand(i + 1);
    IFTBOOL(MDO.get() != nullptr, DXC_E_INCORRECT_DXIL_METADATA);

    switch (Tag) {
    case kDxilPayloadFieldAnnotationAccessTag:
      LoadDxrPayloadAccessQualifiers(MDO, FA);
      break;
    default:
      DXASSERT(false, "Unknown payload field annotation tag");
      m_bExtraMetadata = true;
      break;
    }
  }
}

void DxilMDHelper::LoadDxrPayloadFieldAnnoations(const MDOperand &MDO,
                                                DxilPayloadAnnotation &SA) {
  IFTBOOL(MDO.get() != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  const MDTuple *pTupleMD = dyn_cast<MDTuple>(MDO.get());
  IFTBOOL(pTupleMD != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  IFTBOOL(pTupleMD->getNumOperands() == SA.GetNumFields(),
          DXC_E_INCORRECT_DXIL_METADATA);
  for (unsigned i = 0; i < SA.GetNumFields(); ++i) {
    LoadDxrPayloadFieldAnnoation(pTupleMD->getOperand(i), SA.GetFieldAnnotation(i));
  }
}

void DxilMDHelper::LoadDxrPayloadAnnotationNode(const llvm::MDTuple &MDT,
                                                DxilTypeSystem &TypeSystem) {
  unsigned Tag = ConstMDToUint32(MDT.getOperand(0));
  IFTBOOL(Tag == kDxilPayloadAnnotationStructTag, DXC_E_INCORRECT_DXIL_METADATA)
  IFTBOOL((MDT.getNumOperands() & 0x1) == 1, DXC_E_INCORRECT_DXIL_METADATA);

  Constant *pGV = dyn_cast<Constant>(ValueMDToValue(MDT.getOperand(1)));
  IFTBOOL(pGV != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  StructType *pGVType = dyn_cast<StructType>(pGV->getType());
  IFTBOOL(pGVType != nullptr, DXC_E_INCORRECT_DXIL_METADATA);

  // Check if this struct is already part of the DXIL Type System
  DxilPayloadAnnotation *pPA = TypeSystem.AddPayloadAnnotation(pGVType);

  LoadDxrPayloadFieldAnnoations(MDT.getOperand(2), *pPA);
}

void DxilMDHelper::LoadDxrPayloadAnnotations(DxilTypeSystem &TypeSystem) {
  NamedMDNode *pDxilPayloadAnnotationsMD =
      m_pModule->getNamedMetadata(kDxilDxrPayloadAnnotationsMDName);
  if (pDxilPayloadAnnotationsMD == nullptr)
    return;

  if (DXIL::CompareVersions(m_MinValMajor, m_MinValMinor, 1, 6) < 0) {
    DXASSERT(false, "payload access qualifier emitted for dxil version < 1.6");
    m_bExtraMetadata = true;
  }
  DXASSERT(pDxilPayloadAnnotationsMD->getNumOperands() != 0, "empty metadata node?");

  for (unsigned i = 0; i < pDxilPayloadAnnotationsMD->getNumOperands(); i++) {
    const MDTuple *pTupleMD =
        dyn_cast<MDTuple>(pDxilPayloadAnnotationsMD->getOperand(i));
    IFTBOOL(pTupleMD != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
    LoadDxrPayloadAnnotationNode(*pTupleMD, TypeSystem);
  }
}

Metadata *DxilMDHelper::EmitDxilTemplateArgAnnotation(const DxilTemplateArgAnnotation &annotation) {
  SmallVector<Metadata *, 2> MDVals;
  if (annotation.IsType()) {
    MDVals.emplace_back(Uint32ToConstMD(DxilMDHelper::kDxilTemplateArgTypeTag));
    MDVals.emplace_back(ValueAsMetadata::get(UndefValue::get(const_cast<Type*>(annotation.GetType()))));
  } else if (annotation.IsIntegral()) {
    MDVals.emplace_back(Uint32ToConstMD(DxilMDHelper::kDxilTemplateArgIntegralTag));
    MDVals.emplace_back(Uint64ToConstMD((uint64_t)annotation.GetIntegral()));
  }
  return MDNode::get(m_Ctx, MDVals);
}
void DxilMDHelper::LoadDxilTemplateArgAnnotation(const llvm::MDOperand &MDO, DxilTemplateArgAnnotation &annotation) {
  IFTBOOL(MDO.get() != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  const MDTuple *pTupleMD = dyn_cast<MDTuple>(MDO.get());
  IFTBOOL(pTupleMD != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  IFTBOOL(pTupleMD->getNumOperands() >= 1, DXC_E_INCORRECT_DXIL_METADATA);
  unsigned Tag = ConstMDToUint32(pTupleMD->getOperand(0));
  switch (Tag) {
  case kDxilTemplateArgTypeTag: {
    IFTBOOL(pTupleMD->getNumOperands() == 2, DXC_E_INCORRECT_DXIL_METADATA);
    Constant *C = dyn_cast<Constant>(ValueMDToValue(pTupleMD->getOperand(kDxilTemplateArgValue)));
    IFTBOOL(C != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
    annotation.SetType(C->getType());
  } break;
  case kDxilTemplateArgIntegralTag:
    IFTBOOL(pTupleMD->getNumOperands() == 2, DXC_E_INCORRECT_DXIL_METADATA);
    annotation.SetIntegral((int64_t)ConstMDToUint64(pTupleMD->getOperand(kDxilTemplateArgValue)));
    break;
  default:
    DXASSERT(false, "Unknown template argument type tag.");
    m_bExtraMetadata = true;
    break;
  }
}

Metadata *DxilMDHelper::EmitDxilStructAnnotation(const DxilStructAnnotation &SA) {
  bool bSupportExtended = DXIL::CompareVersions(m_MinValMajor, m_MinValMinor, 1, 5) >= 0;

  vector<Metadata *> MDVals;
  MDVals.reserve(SA.GetNumFields() + 2);  // In case of extended 1.5 property list
  MDVals.resize(SA.GetNumFields() + 1);

  MDVals[0] = Uint32ToConstMD(SA.GetCBufferSize());
  for (unsigned i = 0; i < SA.GetNumFields(); i++) {
    MDVals[i+1] = EmitDxilFieldAnnotation(SA.GetFieldAnnotation(i));
  }

  // Only add template args if shader target requires validator version that supports them.
  if (bSupportExtended && SA.GetNumTemplateArgs()) {
    vector<Metadata *> MDTemplateArgs(SA.GetNumTemplateArgs());
    for (unsigned i = 0; i < SA.GetNumTemplateArgs(); ++i) {
      MDTemplateArgs[i] = EmitDxilTemplateArgAnnotation(SA.GetTemplateArgAnnotation(i));
    }
    SmallVector<Metadata *, 2> MDExtraVals;
    MDExtraVals.emplace_back(Uint32ToConstMD(DxilMDHelper::kDxilTemplateArgumentsTag));
    MDExtraVals.emplace_back(MDNode::get(m_Ctx, MDTemplateArgs));
    MDVals.emplace_back(MDNode::get(m_Ctx, MDExtraVals));
  }

  return MDNode::get(m_Ctx, MDVals);
}

void DxilMDHelper::LoadDxilStructAnnotation(const MDOperand &MDO, DxilStructAnnotation &SA) {
  IFTBOOL(MDO.get() != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  const MDTuple *pTupleMD = dyn_cast<MDTuple>(MDO.get());
  IFTBOOL(pTupleMD != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  if (pTupleMD->getNumOperands() == 1) {
    SA.MarkEmptyStruct();
  }
  if (pTupleMD->getNumOperands() == SA.GetNumFields()+2) {
    DXASSERT(DXIL::CompareVersions(m_MinValMajor, m_MinValMinor, 1, 5) >= 0,
      "otherwise, template annotation emitted for dxil version < 1.5");
    // Load template args from extended operand
    const MDOperand &MDOExtra = pTupleMD->getOperand(SA.GetNumFields()+1);
    const MDTuple *pTupleMDExtra = dyn_cast_or_null<MDTuple>(MDOExtra.get());
    if(pTupleMDExtra) {
      for (unsigned i = 0; i < pTupleMDExtra->getNumOperands(); i += 2) {
        unsigned Tag = ConstMDToUint32(pTupleMDExtra->getOperand(i));
        const MDOperand &MDO = pTupleMDExtra->getOperand(i + 1);
        IFTBOOL(MDO.get() != nullptr, DXC_E_INCORRECT_DXIL_METADATA);

        switch (Tag) {
        case kDxilTemplateArgumentsTag: {
          const MDTuple *pTupleTemplateArgs = dyn_cast_or_null<MDTuple>(pTupleMDExtra->getOperand(1).get());
          IFTBOOL(pTupleTemplateArgs, DXC_E_INCORRECT_DXIL_METADATA);
          SA.SetNumTemplateArgs(pTupleTemplateArgs->getNumOperands());
          for (unsigned i = 0; i < pTupleTemplateArgs->getNumOperands(); ++i) {
            LoadDxilTemplateArgAnnotation(pTupleTemplateArgs->getOperand(i), SA.GetTemplateArgAnnotation(i));
          }
        } break;
        default:
          DXASSERT(false, "unknown extended tag for struct annotation.");
          m_bExtraMetadata = true;
          break;
        }
      }
    }
  } else {
    IFTBOOL(pTupleMD->getNumOperands() == SA.GetNumFields()+1, DXC_E_INCORRECT_DXIL_METADATA);
  }

  SA.SetCBufferSize(ConstMDToUint32(pTupleMD->getOperand(0)));
  for (unsigned i = 0; i < SA.GetNumFields(); i++) {
    const MDOperand &MDO = pTupleMD->getOperand(i+1);
    DxilFieldAnnotation &FA = SA.GetFieldAnnotation(i);
    LoadDxilFieldAnnotation(MDO, FA);
  }
}

Metadata *
DxilMDHelper::EmitDxilFunctionAnnotation(const DxilFunctionAnnotation &FA) {
  return EmitDxilParamAnnotations(FA);
}

void DxilMDHelper::LoadDxilFunctionAnnotation(const MDOperand &MDO,
                                              DxilFunctionAnnotation &FA) {
  LoadDxilParamAnnotations(MDO, FA);
}

llvm::Metadata *
DxilMDHelper::EmitDxilParamAnnotations(const DxilFunctionAnnotation &FA) {
  vector<Metadata *> MDParamAnnotations(FA.GetNumParameters() + 1);
  MDParamAnnotations[0] = EmitDxilParamAnnotation(FA.GetRetTypeAnnotation());
  for (unsigned i = 0; i < FA.GetNumParameters(); i++) {
    MDParamAnnotations[i + 1] =
        EmitDxilParamAnnotation(FA.GetParameterAnnotation(i));
  }
  return MDNode::get(m_Ctx, MDParamAnnotations);
}

void DxilMDHelper::LoadDxilParamAnnotations(const llvm::MDOperand &MDO,
                                            DxilFunctionAnnotation &FA) {
  IFTBOOL(MDO.get() != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  const MDTuple *pTupleMD = dyn_cast<MDTuple>(MDO.get());
  IFTBOOL(pTupleMD->getNumOperands() == FA.GetNumParameters() + 1,
          DXC_E_INCORRECT_DXIL_METADATA);
  DxilParameterAnnotation &retTyAnnotation = FA.GetRetTypeAnnotation();
  LoadDxilParamAnnotation(pTupleMD->getOperand(0), retTyAnnotation);
  for (unsigned i = 0; i < FA.GetNumParameters(); i++) {
    const MDOperand &MDO = pTupleMD->getOperand(i + 1);
    DxilParameterAnnotation &PA = FA.GetParameterAnnotation(i);
    LoadDxilParamAnnotation(MDO, PA);
  }
}

Metadata *
DxilMDHelper::EmitDxilParamAnnotation(const DxilParameterAnnotation &PA) {
  vector<Metadata *> MDVals(3);
  MDVals[0] = Uint32ToConstMD(static_cast<unsigned>(PA.GetParamInputQual()));
  MDVals[1] = EmitDxilFieldAnnotation(PA);
  MDVals[2] = Uint32VectorToConstMDTuple(PA.GetSemanticIndexVec());

  return MDNode::get(m_Ctx, MDVals);
}
void DxilMDHelper::LoadDxilParamAnnotation(const MDOperand &MDO,
                                           DxilParameterAnnotation &PA) {
  IFTBOOL(MDO.get() != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  const MDTuple *pTupleMD = dyn_cast<MDTuple>(MDO.get());
  IFTBOOL(pTupleMD != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  IFTBOOL(pTupleMD->getNumOperands() == 3, DXC_E_INCORRECT_DXIL_METADATA);
  PA.SetParamInputQual(static_cast<DxilParamInputQual>(
      ConstMDToUint32(pTupleMD->getOperand(0))));
  LoadDxilFieldAnnotation(pTupleMD->getOperand(1), PA);
  MDTuple *pSemanticIndexVectorMD = dyn_cast<MDTuple>(pTupleMD->getOperand(2));
  vector<unsigned> SemanticIndexVector;
  ConstMDTupleToUint32Vector(pSemanticIndexVectorMD, SemanticIndexVector);
  PA.SetSemanticIndexVec(SemanticIndexVector);
}

Metadata *DxilMDHelper::EmitDxilFieldAnnotation(const DxilFieldAnnotation &FA) {
  vector<Metadata *> MDVals;  // Tag-Value list.

  if (FA.HasFieldName()) {
    MDVals.emplace_back(Uint32ToConstMD(kDxilFieldAnnotationFieldNameTag));
    MDVals.emplace_back(MDString::get(m_Ctx, FA.GetFieldName()));
  }
  if (FA.IsPrecise()) {
    MDVals.emplace_back(Uint32ToConstMD(kDxilFieldAnnotationPreciseTag)); // Tag
    MDVals.emplace_back(BoolToConstMD(true));                             // Value
  }
  if (FA.HasMatrixAnnotation()) {
    const DxilMatrixAnnotation &MA = FA.GetMatrixAnnotation();
    Metadata *MatrixMD[3];
    MatrixMD[0] = Uint32ToConstMD(MA.Rows);
    MatrixMD[1] = Uint32ToConstMD(MA.Cols);
    MatrixMD[2] = Uint32ToConstMD((unsigned)MA.Orientation);

    MDVals.emplace_back(Uint32ToConstMD(kDxilFieldAnnotationMatrixTag));
    MDVals.emplace_back(MDNode::get(m_Ctx, MatrixMD));
  }
  if (FA.HasCBufferOffset()) {
    MDVals.emplace_back(Uint32ToConstMD(kDxilFieldAnnotationCBufferOffsetTag));
    MDVals.emplace_back(Uint32ToConstMD(FA.GetCBufferOffset()));
  }
  if (FA.HasSemanticString()) {
    MDVals.emplace_back(Uint32ToConstMD(kDxilFieldAnnotationSemanticStringTag));
    MDVals.emplace_back(MDString::get(m_Ctx, FA.GetSemanticString()));
  }
  if (FA.HasInterpolationMode()) {
    MDVals.emplace_back(Uint32ToConstMD(kDxilFieldAnnotationInterpolationModeTag));
    MDVals.emplace_back(Uint32ToConstMD((unsigned)FA.GetInterpolationMode().GetKind()));
  }
  if (FA.HasCompType()) {
    MDVals.emplace_back(Uint32ToConstMD(kDxilFieldAnnotationCompTypeTag));
    MDVals.emplace_back(Uint32ToConstMD((unsigned)FA.GetCompType().GetKind()));
  }
  if (FA.IsCBVarUsed() &&
      DXIL::CompareVersions(m_MinValMajor, m_MinValMinor, 1, 5) >= 0) {
    MDVals.emplace_back(Uint32ToConstMD(kDxilFieldAnnotationCBUsedTag));
    MDVals.emplace_back(BoolToConstMD(true));
  }

  return MDNode::get(m_Ctx, MDVals);
}


void DxilMDHelper::LoadDxilFieldAnnotation(const MDOperand &MDO, DxilFieldAnnotation &FA) {
  IFTBOOL(MDO.get() != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  const MDTuple *pTupleMD = dyn_cast<MDTuple>(MDO.get());
  IFTBOOL(pTupleMD != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  IFTBOOL((pTupleMD->getNumOperands() & 0x1) == 0, DXC_E_INCORRECT_DXIL_METADATA);

  for (unsigned i = 0; i < pTupleMD->getNumOperands(); i += 2) {
    unsigned Tag = ConstMDToUint32(pTupleMD->getOperand(i));
    const MDOperand &MDO = pTupleMD->getOperand(i + 1);
    IFTBOOL(MDO.get() != nullptr, DXC_E_INCORRECT_DXIL_METADATA);

    switch (Tag) {
    case kDxilFieldAnnotationPreciseTag:
      FA.SetPrecise(ConstMDToBool(MDO));
      break;
    case kDxilFieldAnnotationMatrixTag: {
      DxilMatrixAnnotation MA;
      const MDTuple *pMATupleMD = dyn_cast<MDTuple>(MDO.get());
      IFTBOOL(pMATupleMD != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
      IFTBOOL(pMATupleMD->getNumOperands() == 3, DXC_E_INCORRECT_DXIL_METADATA);
      MA.Rows = ConstMDToUint32(pMATupleMD->getOperand(0));
      MA.Cols = ConstMDToUint32(pMATupleMD->getOperand(1));
      MA.Orientation = (MatrixOrientation)ConstMDToUint32(pMATupleMD->getOperand(2));
      FA.SetMatrixAnnotation(MA);
    } break;
    case kDxilFieldAnnotationCBufferOffsetTag:
      FA.SetCBufferOffset(ConstMDToUint32(MDO));
      break;
    case kDxilFieldAnnotationSemanticStringTag:
      FA.SetSemanticString(StringMDToString(MDO));
      break;
    case kDxilFieldAnnotationInterpolationModeTag:
      FA.SetInterpolationMode(InterpolationMode((InterpolationMode::Kind)ConstMDToUint32(MDO)));
      break;
    case kDxilFieldAnnotationFieldNameTag:
      FA.SetFieldName(StringMDToString(MDO));
      break;
    case kDxilFieldAnnotationCompTypeTag:
      FA.SetCompType((CompType::Kind)ConstMDToUint32(MDO));
      break;
    case kDxilFieldAnnotationCBUsedTag:
      FA.SetCBVarUsed(ConstMDToBool(MDO));
      break;
    default:
      DXASSERT(false, "Unknown extended shader properties tag");
      m_bExtraMetadata = true;
      break;
    }
  }
}

Metadata *
DxilMDHelper::EmitDxrPayloadFieldAnnotation(const DxilPayloadFieldAnnotation &FA, Type* fieldType) {
  vector<Metadata *> MDVals; // Tag-Value list.
  MDVals.emplace_back(Uint32ToConstMD(kDxilPayloadFieldAnnotationAccessTag));

  auto mask = FA.GetPayloadFieldQualifierMask();
  MDVals.emplace_back(Uint32ToConstMD(mask));

  return MDNode::get(m_Ctx, MDVals);
}

const Function *DxilMDHelper::LoadDxilFunctionProps(const MDTuple *pProps,
                                              hlsl::DxilFunctionProps *props) {
  unsigned idx = 0;
  const Function *F = dyn_cast<Function>(
      dyn_cast<ValueAsMetadata>(pProps->getOperand(idx++))->getValue());
  DXIL::ShaderKind shaderKind =
      static_cast<DXIL::ShaderKind>(ConstMDToUint32(pProps->getOperand(idx++)));

  bool bRayAttributes = false;
  props->shaderKind = shaderKind;
  switch (shaderKind) {
  case DXIL::ShaderKind::Compute:
    props->ShaderProps.CS.numThreads[0] =
        ConstMDToUint32(pProps->getOperand(idx++));
    props->ShaderProps.CS.numThreads[1] =
        ConstMDToUint32(pProps->getOperand(idx++));
    props->ShaderProps.CS.numThreads[2] =
        ConstMDToUint32(pProps->getOperand(idx++));
    break;
  case DXIL::ShaderKind::Geometry:
    props->ShaderProps.GS.inputPrimitive =
        (DXIL::InputPrimitive)ConstMDToUint32(pProps->getOperand(idx++));
    props->ShaderProps.GS.maxVertexCount =
        ConstMDToUint32(pProps->getOperand(idx++));
    props->ShaderProps.GS.instanceCount =
        ConstMDToUint32(pProps->getOperand(idx++));
    for (size_t i = 0;
         i < _countof(props->ShaderProps.GS.streamPrimitiveTopologies); ++i)
      props->ShaderProps.GS.streamPrimitiveTopologies[i] =
          (DXIL::PrimitiveTopology)ConstMDToUint32(pProps->getOperand(idx++));
    break;
  case DXIL::ShaderKind::Hull:
    props->ShaderProps.HS.patchConstantFunc = dyn_cast<Function>(
        dyn_cast<ValueAsMetadata>(pProps->getOperand(idx++))->getValue());
    props->ShaderProps.HS.domain =
        (DXIL::TessellatorDomain)ConstMDToUint32(pProps->getOperand(idx++));
    props->ShaderProps.HS.partition =
        (DXIL::TessellatorPartitioning)ConstMDToUint32(
            pProps->getOperand(idx++));
    props->ShaderProps.HS.outputPrimitive =
        (DXIL::TessellatorOutputPrimitive)ConstMDToUint32(
            pProps->getOperand(idx++));
    props->ShaderProps.HS.inputControlPoints =
        ConstMDToUint32(pProps->getOperand(idx++));
    props->ShaderProps.HS.outputControlPoints =
        ConstMDToUint32(pProps->getOperand(idx++));
    props->ShaderProps.HS.maxTessFactor =
        ConstMDToFloat(pProps->getOperand(idx++));
    break;
  case DXIL::ShaderKind::Domain:
    props->ShaderProps.DS.domain =
        (DXIL::TessellatorDomain)ConstMDToUint32(pProps->getOperand(idx++));
    props->ShaderProps.DS.inputControlPoints =
        ConstMDToUint32(pProps->getOperand(idx++));
    break;
  case DXIL::ShaderKind::Pixel:
    props->ShaderProps.PS.EarlyDepthStencil =
        ConstMDToUint32(pProps->getOperand(idx++));
    break;
  case DXIL::ShaderKind::AnyHit:
  case DXIL::ShaderKind::ClosestHit:
    bRayAttributes = true;
  case DXIL::ShaderKind::Miss:
  case DXIL::ShaderKind::Callable:
    // payload/params unioned and first:
    props->ShaderProps.Ray.payloadSizeInBytes =
      ConstMDToUint32(pProps->getOperand(idx++));
    if (bRayAttributes)
      props->ShaderProps.Ray.attributeSizeInBytes =
        ConstMDToUint32(pProps->getOperand(idx++));
    break;
  case DXIL::ShaderKind::Mesh:
    props->ShaderProps.MS.numThreads[0] =
      ConstMDToUint32(pProps->getOperand(idx++));
    props->ShaderProps.MS.numThreads[1] =
      ConstMDToUint32(pProps->getOperand(idx++));
    props->ShaderProps.MS.numThreads[2] =
      ConstMDToUint32(pProps->getOperand(idx++));
    props->ShaderProps.MS.maxVertexCount =
      ConstMDToUint32(pProps->getOperand(idx++));
    props->ShaderProps.MS.maxPrimitiveCount =
      ConstMDToUint32(pProps->getOperand(idx++));
    props->ShaderProps.MS.outputTopology =
      (DXIL::MeshOutputTopology)ConstMDToUint32(pProps->getOperand(idx++));
    props->ShaderProps.MS.payloadSizeInBytes =
      ConstMDToUint32(pProps->getOperand(idx++));
    break;
  case DXIL::ShaderKind::Amplification:
    props->ShaderProps.AS.numThreads[0] =
      ConstMDToUint32(pProps->getOperand(idx++));
    props->ShaderProps.AS.numThreads[1] =
      ConstMDToUint32(pProps->getOperand(idx++));
    props->ShaderProps.AS.numThreads[2] =
      ConstMDToUint32(pProps->getOperand(idx++));
    props->ShaderProps.AS.payloadSizeInBytes =
      ConstMDToUint32(pProps->getOperand(idx++));
    break;
  default:
    break;
  }
  return F;
}

MDTuple *DxilMDHelper::EmitDxilEntryProperties(uint64_t rawShaderFlag,
                                                const DxilFunctionProps &props,
                                                unsigned autoBindingSpace) {
  vector<Metadata *> MDVals;

  // DXIL shader flags.
  if (props.IsPS()) {
    if (props.ShaderProps.PS.EarlyDepthStencil) {
      ShaderFlags flags;
      flags.SetShaderFlagsRaw(rawShaderFlag);
      flags.SetForceEarlyDepthStencil(true);
      rawShaderFlag = flags.GetShaderFlagsRaw();
    }
  }
  if (rawShaderFlag != 0) {
    MDVals.emplace_back(Uint32ToConstMD(kDxilShaderFlagsTag));
    MDVals.emplace_back(Uint64ToConstMD(rawShaderFlag));
  }

  // Add shader kind for lib entrys.
  if (m_pSM->IsLib() && props.shaderKind != DXIL::ShaderKind::Library) {
    MDVals.emplace_back(Uint32ToConstMD(kDxilShaderKindTag));
    MDVals.emplace_back(
        Uint32ToConstMD(static_cast<unsigned>(props.shaderKind)));
  }

  switch (props.shaderKind) {
  // Compute shader.
  case DXIL::ShaderKind::Compute: {
    auto &CS = props.ShaderProps.CS;
    MDVals.emplace_back(Uint32ToConstMD(DxilMDHelper::kDxilNumThreadsTag));
    vector<Metadata *> NumThreadVals;
    NumThreadVals.emplace_back(Uint32ToConstMD(CS.numThreads[0]));
    NumThreadVals.emplace_back(Uint32ToConstMD(CS.numThreads[1]));
    NumThreadVals.emplace_back(Uint32ToConstMD(CS.numThreads[2]));
    MDVals.emplace_back(MDNode::get(m_Ctx, NumThreadVals));

    if (props.waveSize != 0) {
      MDVals.emplace_back(Uint32ToConstMD(DxilMDHelper::kDxilWaveSizeTag));
      vector<Metadata *> WaveSizeVal;
      WaveSizeVal.emplace_back(Uint32ToConstMD(props.waveSize));
      MDVals.emplace_back(MDNode::get(m_Ctx, WaveSizeVal));
    }
  } break;
  // Geometry shader.
  case DXIL::ShaderKind::Geometry: {
    MDVals.emplace_back(Uint32ToConstMD(DxilMDHelper::kDxilGSStateTag));
    DXIL::PrimitiveTopology topo = DXIL::PrimitiveTopology::Undefined;
    unsigned activeStreamMask = 0;
    for (size_t i = 0;
         i < _countof(props.ShaderProps.GS.streamPrimitiveTopologies); ++i) {
      if (props.ShaderProps.GS.streamPrimitiveTopologies[i] !=
          DXIL::PrimitiveTopology::Undefined) {
        activeStreamMask |= 1 << i;
        DXASSERT_NOMSG(topo == DXIL::PrimitiveTopology::Undefined ||
                       topo ==
                           props.ShaderProps.GS.streamPrimitiveTopologies[i]);
        topo = props.ShaderProps.GS.streamPrimitiveTopologies[i];
      }
    }
    MDTuple *pMDTuple =
        EmitDxilGSState(props.ShaderProps.GS.inputPrimitive,
                        props.ShaderProps.GS.maxVertexCount, activeStreamMask,
                        topo, props.ShaderProps.GS.instanceCount);
    MDVals.emplace_back(pMDTuple);
  } break;
  // Domain shader.
  case DXIL::ShaderKind::Domain: {
    auto &DS = props.ShaderProps.DS;
    MDVals.emplace_back(Uint32ToConstMD(DxilMDHelper::kDxilDSStateTag));
    MDTuple *pMDTuple = EmitDxilDSState(DS.domain, DS.inputControlPoints);
    MDVals.emplace_back(pMDTuple);
  } break;
  // Hull shader.
  case DXIL::ShaderKind::Hull: {
    auto &HS = props.ShaderProps.HS;
    MDVals.emplace_back(Uint32ToConstMD(DxilMDHelper::kDxilHSStateTag));
    MDTuple *pMDTuple = EmitDxilHSState(
        HS.patchConstantFunc, HS.inputControlPoints, HS.outputControlPoints,
        HS.domain, HS.partition, HS.outputPrimitive, HS.maxTessFactor);
    MDVals.emplace_back(pMDTuple);
  } break;
  // Raytracing.
  case DXIL::ShaderKind::AnyHit:
  case DXIL::ShaderKind::ClosestHit: {
    MDVals.emplace_back(Uint32ToConstMD(kDxilRayPayloadSizeTag));
    MDVals.emplace_back(
        Uint32ToConstMD(props.ShaderProps.Ray.payloadSizeInBytes));

    MDVals.emplace_back(Uint32ToConstMD(kDxilRayAttribSizeTag));
    MDVals.emplace_back(
        Uint32ToConstMD(props.ShaderProps.Ray.attributeSizeInBytes));
  } break;
  case DXIL::ShaderKind::Miss:
  case DXIL::ShaderKind::Callable: {
    MDVals.emplace_back(Uint32ToConstMD(kDxilRayPayloadSizeTag));

    MDVals.emplace_back(
        Uint32ToConstMD(props.ShaderProps.Ray.payloadSizeInBytes));
  } break;
  case DXIL::ShaderKind::Mesh: {
    auto &MS = props.ShaderProps.MS;
    MDVals.emplace_back(Uint32ToConstMD(DxilMDHelper::kDxilMSStateTag));
    MDTuple *pMDTuple = EmitDxilMSState(MS.numThreads,
                                        MS.maxVertexCount,
                                        MS.maxPrimitiveCount,
                                        MS.outputTopology,
                                        MS.payloadSizeInBytes);
    MDVals.emplace_back(pMDTuple);
  } break;
  case DXIL::ShaderKind::Amplification: {
    auto &AS = props.ShaderProps.AS;
    MDVals.emplace_back(Uint32ToConstMD(DxilMDHelper::kDxilASStateTag));
    MDTuple *pMDTuple = EmitDxilASState(AS.numThreads, AS.payloadSizeInBytes);
    MDVals.emplace_back(pMDTuple);
  } break;
  default:
    break;
  }

  if (autoBindingSpace != UINT_MAX && m_pSM->IsSMAtLeast(6, 3)) {
    MDVals.emplace_back(Uint32ToConstMD(kDxilAutoBindingSpaceTag));
    MDVals.emplace_back(
        MDNode::get(m_Ctx, {Uint32ToConstMD(autoBindingSpace)}));
  }

  if (!props.serializedRootSignature.empty() &&
      DXIL::CompareVersions(m_MinValMajor, m_MinValMinor, 1, 6) > 0) {
    MDVals.emplace_back(Uint32ToConstMD(DxilMDHelper::kDxilEntryRootSigTag));
    MDVals.emplace_back(
        EmitSerializedRootSignature(props.serializedRootSignature, m_Ctx));
  }

  if (!MDVals.empty())
    return MDNode::get(m_Ctx, MDVals);
  else
    return nullptr;
}

void DxilMDHelper::LoadDxilEntryProperties(const MDOperand &MDO,
                                            uint64_t &rawShaderFlag,
                                            DxilFunctionProps &props,
                                            uint32_t &autoBindingSpace) {
  if (MDO.get() == nullptr)
    return;

  const MDTuple *pTupleMD = dyn_cast<MDTuple>(MDO.get());
  IFTBOOL(pTupleMD != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  IFTBOOL((pTupleMD->getNumOperands() & 0x1) == 0,
          DXC_E_INCORRECT_DXIL_METADATA);
  bool bEarlyDepth = false;

  if (!m_pSM->IsLib()) {
    props.shaderKind = m_pSM->GetKind();
  } else {
    props.shaderKind = DXIL::ShaderKind::Library;
  }

  for (unsigned iNode = 0; iNode < pTupleMD->getNumOperands(); iNode += 2) {
    unsigned Tag = DxilMDHelper::ConstMDToUint32(pTupleMD->getOperand(iNode));
    const MDOperand &MDO = pTupleMD->getOperand(iNode + 1);
    IFTBOOL(MDO.get() != nullptr, DXC_E_INCORRECT_DXIL_METADATA);

    switch (Tag) {
    case DxilMDHelper::kDxilShaderFlagsTag: {
      rawShaderFlag = ConstMDToUint64(MDO);
      ShaderFlags flags;
      flags.SetShaderFlagsRaw(rawShaderFlag);
      bEarlyDepth = flags.GetForceEarlyDepthStencil();
    } break;

    case DxilMDHelper::kDxilNumThreadsTag: {
      DXASSERT(props.IsCS(), "else invalid shader kind");
      auto &CS = props.ShaderProps.CS;
      MDNode *pNode = cast<MDNode>(MDO.get());
      CS.numThreads[0] = ConstMDToUint32(pNode->getOperand(0));
      CS.numThreads[1] = ConstMDToUint32(pNode->getOperand(1));
      CS.numThreads[2] = ConstMDToUint32(pNode->getOperand(2));
    } break;

    case DxilMDHelper::kDxilGSStateTag: {
      DXASSERT(props.IsGS(), "else invalid shader kind");
      auto &GS = props.ShaderProps.GS;
      DXIL::PrimitiveTopology topo = DXIL::PrimitiveTopology::Undefined;
      unsigned activeStreamMask;
      LoadDxilGSState(MDO, GS.inputPrimitive, GS.maxVertexCount,
                      activeStreamMask, topo, GS.instanceCount);
      if (topo != DXIL::PrimitiveTopology::Undefined) {
        for (size_t i = 0; i < _countof(GS.streamPrimitiveTopologies); ++i) {
          unsigned mask = 1 << i;
          if (activeStreamMask & mask) {
            GS.streamPrimitiveTopologies[i] = topo;
          } else {
            GS.streamPrimitiveTopologies[i] =
                DXIL::PrimitiveTopology::Undefined;
          }
        }
      }
    } break;

    case DxilMDHelper::kDxilDSStateTag: {
      DXASSERT(props.IsDS(), "else invalid shader kind");
      auto &DS = props.ShaderProps.DS;
      LoadDxilDSState(MDO, DS.domain, DS.inputControlPoints);
    } break;

    case DxilMDHelper::kDxilHSStateTag: {
      DXASSERT(props.IsHS(), "else invalid shader kind");
      auto &HS = props.ShaderProps.HS;
      LoadDxilHSState(MDO, HS.patchConstantFunc, HS.inputControlPoints,
                      HS.outputControlPoints, HS.domain, HS.partition,
                      HS.outputPrimitive, HS.maxTessFactor);
    } break;

    case DxilMDHelper::kDxilAutoBindingSpaceTag: {
      MDNode *pNode = cast<MDNode>(MDO.get());
      autoBindingSpace = ConstMDToUint32(pNode->getOperand(0));
      break;
    }
    case DxilMDHelper::kDxilRayPayloadSizeTag: {
      DXASSERT(props.IsAnyHit() || props.IsClosestHit() || props.IsMiss() ||
                   props.IsCallable(),
               "else invalid shader kind");
      props.ShaderProps.Ray.payloadSizeInBytes =
          ConstMDToUint32(MDO);
    } break;
    case DxilMDHelper::kDxilRayAttribSizeTag: {
      DXASSERT(props.IsAnyHit() || props.IsClosestHit(),
               "else invalid shader kind");
      props.ShaderProps.Ray.attributeSizeInBytes =
          ConstMDToUint32(MDO);
    } break;
    case DxilMDHelper::kDxilShaderKindTag: {
      DXIL::ShaderKind kind =
          static_cast<DXIL::ShaderKind>(ConstMDToUint32(MDO));
      DXASSERT(props.shaderKind == DXIL::ShaderKind::Library,
               "else invalid shader kind");
      props.shaderKind = kind;
    } break;
    case DxilMDHelper::kDxilMSStateTag: {
      DXASSERT(props.IsMS(), "else invalid shader kind");
      auto &MS = props.ShaderProps.MS;
      LoadDxilMSState(MDO, MS.numThreads, MS.maxVertexCount,
                      MS.maxPrimitiveCount, MS.outputTopology,
                      MS.payloadSizeInBytes);
    } break;
    case DxilMDHelper::kDxilASStateTag: {
      DXASSERT(props.IsAS(), "else invalid shader kind");
      auto &AS = props.ShaderProps.AS;
      LoadDxilASState(MDO, AS.numThreads, AS.payloadSizeInBytes);
    } break;
    case DxilMDHelper::kDxilWaveSizeTag: {
      DXASSERT(props.IsCS(), "else invalid shader kind");
      MDNode *pNode = cast<MDNode>(MDO.get());
      props.waveSize = ConstMDToUint32(pNode->getOperand(0));
    } break;
    case DxilMDHelper::kDxilEntryRootSigTag: {
      MDNode *pNode = cast<MDNode>(MDO.get());
      LoadSerializedRootSignature(pNode, props.serializedRootSignature, m_Ctx);
    } break;
    default:
      DXASSERT(false, "Unknown extended shader properties tag");
      m_bExtraMetadata = true;
      break;
    }
  }

  if (bEarlyDepth) {
    DXASSERT(props.IsPS(), "else invalid shader kind");
    props.ShaderProps.PS.EarlyDepthStencil = true;
  }
}

MDTuple *
DxilMDHelper::EmitDxilFunctionProps(const hlsl::DxilFunctionProps *props,
                                   const Function *F) {
  bool bRayAttributes = false;
  Metadata *MDVals[30];
  std::fill(MDVals, MDVals + _countof(MDVals), nullptr);
  unsigned valIdx = 0;
  MDVals[valIdx++] = ValueAsMetadata::get(const_cast<Function*>(F));
  MDVals[valIdx++] = Uint32ToConstMD(static_cast<unsigned>(props->shaderKind));
  switch (props->shaderKind) {
  case DXIL::ShaderKind::Compute:
    MDVals[valIdx++] = Uint32ToConstMD(props->ShaderProps.CS.numThreads[0]);
    MDVals[valIdx++] = Uint32ToConstMD(props->ShaderProps.CS.numThreads[1]);
    MDVals[valIdx++] = Uint32ToConstMD(props->ShaderProps.CS.numThreads[2]);
    break;
  case DXIL::ShaderKind::Geometry:
    MDVals[valIdx++] =
        Uint8ToConstMD((uint8_t)props->ShaderProps.GS.inputPrimitive);
    MDVals[valIdx++] = Uint32ToConstMD(props->ShaderProps.GS.maxVertexCount);
    MDVals[valIdx++] = Uint32ToConstMD(props->ShaderProps.GS.instanceCount);
    for (size_t i = 0;
         i < _countof(props->ShaderProps.GS.streamPrimitiveTopologies); ++i)
      MDVals[valIdx++] = Uint8ToConstMD(
          (uint8_t)props->ShaderProps.GS.streamPrimitiveTopologies[i]);
    break;
  case DXIL::ShaderKind::Hull:
    MDVals[valIdx++] =
        ValueAsMetadata::get(props->ShaderProps.HS.patchConstantFunc);
    MDVals[valIdx++] = Uint8ToConstMD((uint8_t)props->ShaderProps.HS.domain);
    MDVals[valIdx++] = Uint8ToConstMD((uint8_t)props->ShaderProps.HS.partition);
    MDVals[valIdx++] =
        Uint8ToConstMD((uint8_t)props->ShaderProps.HS.outputPrimitive);
    MDVals[valIdx++] =
        Uint32ToConstMD(props->ShaderProps.HS.inputControlPoints);
    MDVals[valIdx++] =
        Uint32ToConstMD(props->ShaderProps.HS.outputControlPoints);
    MDVals[valIdx++] = FloatToConstMD(props->ShaderProps.HS.maxTessFactor);
    break;
  case DXIL::ShaderKind::Domain:
    MDVals[valIdx++] = Uint8ToConstMD((uint8_t)props->ShaderProps.DS.domain);
    MDVals[valIdx++] =
        Uint32ToConstMD(props->ShaderProps.DS.inputControlPoints);
    break;
  case DXIL::ShaderKind::Pixel:
    MDVals[valIdx++] = BoolToConstMD(props->ShaderProps.PS.EarlyDepthStencil);
    break;
  case DXIL::ShaderKind::AnyHit:
  case DXIL::ShaderKind::ClosestHit:
    bRayAttributes = true;
  case DXIL::ShaderKind::Miss:
  case DXIL::ShaderKind::Callable:
    // payload/params unioned and first:
    MDVals[valIdx++] = Uint32ToConstMD(props->ShaderProps.Ray.payloadSizeInBytes);
    if (bRayAttributes)
      MDVals[valIdx++] = Uint32ToConstMD(props->ShaderProps.Ray.attributeSizeInBytes);
    break;
  case DXIL::ShaderKind::Mesh:
    MDVals[valIdx++] = Uint32ToConstMD(props->ShaderProps.MS.numThreads[0]);
    MDVals[valIdx++] = Uint32ToConstMD(props->ShaderProps.MS.numThreads[1]);
    MDVals[valIdx++] = Uint32ToConstMD(props->ShaderProps.MS.numThreads[2]);
    MDVals[valIdx++] = Uint32ToConstMD(props->ShaderProps.MS.maxVertexCount);
    MDVals[valIdx++] = Uint32ToConstMD(props->ShaderProps.MS.maxPrimitiveCount);
    MDVals[valIdx++] = Uint8ToConstMD((uint8_t)props->ShaderProps.MS.outputTopology);
    MDVals[valIdx++] = Uint32ToConstMD(props->ShaderProps.MS.payloadSizeInBytes);
    break;
  case DXIL::ShaderKind::Amplification:
    MDVals[valIdx++] = Uint32ToConstMD(props->ShaderProps.AS.numThreads[0]);
    MDVals[valIdx++] = Uint32ToConstMD(props->ShaderProps.AS.numThreads[1]);
    MDVals[valIdx++] = Uint32ToConstMD(props->ShaderProps.AS.numThreads[2]);
    MDVals[valIdx++] = Uint32ToConstMD(props->ShaderProps.AS.payloadSizeInBytes);
    break;
  default:
    break;
  }
  return MDTuple::get(m_Ctx, ArrayRef<llvm::Metadata *>(MDVals, valIdx));
}

void DxilMDHelper::EmitDxilViewIdState(std::vector<unsigned> &SerializedState) {
  const vector<unsigned> &Data = SerializedState;
  // If all UINTs are zero, do not emit ViewIdState.
  if (!std::any_of(Data.begin(), Data.end(), [](unsigned e){return e!=0;}))
    return;

  Constant *V = ConstantDataArray::get(m_Ctx, ArrayRef<uint32_t>(Data));
  NamedMDNode *pViewIdNamedMD = m_pModule->getNamedMetadata(kDxilViewIdStateMDName);
  IFTBOOL(pViewIdNamedMD == nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  pViewIdNamedMD = m_pModule->getOrInsertNamedMetadata(kDxilViewIdStateMDName);
  pViewIdNamedMD->addOperand(MDNode::get(m_Ctx, {ConstantAsMetadata::get(V)}));
}

void DxilMDHelper::LoadDxilViewIdState(std::vector<unsigned> &SerializedState) {
  NamedMDNode *pViewIdStateNamedMD = m_pModule->getNamedMetadata(kDxilViewIdStateMDName);
  if(!pViewIdStateNamedMD)
    return;

  IFTBOOL(pViewIdStateNamedMD->getNumOperands() == 1, DXC_E_INCORRECT_DXIL_METADATA);

  MDNode *pNode = pViewIdStateNamedMD->getOperand(0);
  IFTBOOL(pNode->getNumOperands() == 1, DXC_E_INCORRECT_DXIL_METADATA);
  const MDOperand &MDO = pNode->getOperand(0);

  const ConstantAsMetadata *pMetaData = dyn_cast<ConstantAsMetadata>(MDO.get());
  IFTBOOL(pMetaData != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  if (isa<ConstantAggregateZero>(pMetaData->getValue()))
    return;
  const ConstantDataArray *pData = dyn_cast<ConstantDataArray>(pMetaData->getValue());
  IFTBOOL(pData != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  IFTBOOL(pData->getElementType() == Type::getInt32Ty(m_Ctx), DXC_E_INCORRECT_DXIL_METADATA);
  IFTBOOL(pData->getRawDataValues().size() < UINT_MAX && 
          (pData->getRawDataValues().size() & 3) == 0, DXC_E_INCORRECT_DXIL_METADATA);

  SerializedState.clear();
  unsigned size = (unsigned)pData->getRawDataValues().size() / 4;
  SerializedState.resize(size);
  const unsigned *Ptr = (const unsigned *)pData->getRawDataValues().begin();
  memcpy(SerializedState.data(), Ptr, size * sizeof(unsigned));
}

MDNode *DxilMDHelper::EmitControlFlowHints(llvm::LLVMContext &Ctx, std::vector<DXIL::ControlFlowHint> &hints) {
  SmallVector<Metadata *, 4> Args;
  // Reserve operand 0 for self reference.
  auto TempNode = MDNode::getTemporary(Ctx, None);
  Args.emplace_back(TempNode.get());
  Args.emplace_back(MDString::get(Ctx, kDxilControlFlowHintMDName));
  for (DXIL::ControlFlowHint &hint : hints)
    Args.emplace_back(Uint32ToConstMD(static_cast<unsigned>(hint), Ctx));

  MDNode *hintsNode = MDNode::get(Ctx, Args);
  // Set the first operand to itself.
  hintsNode->replaceOperandWith(0, hintsNode);
  return hintsNode;
}

unsigned DxilMDHelper::GetControlFlowHintMask(const Instruction *I) {
  // Check that there are control hint to use
  // branch.
  MDNode *MD = I->getMetadata(hlsl::DxilMDHelper::kDxilControlFlowHintMDName);
  if (!MD)
    return 0;

  if (MD->getNumOperands() < 3)
    return 0;
  unsigned mask = 0;
  for (unsigned i = 2; i < MD->getNumOperands(); i++) {
    Metadata *Op = MD->getOperand(2).get();
    auto ConstOp = cast<ConstantAsMetadata>(Op);
    unsigned hint = ConstOp->getValue()->getUniqueInteger().getLimitedValue();
    mask |= 1 << hint;
  }
  return mask;
}

bool DxilMDHelper::HasControlFlowHintToPreventFlatten(
    const llvm::Instruction *I) {
  unsigned mask = GetControlFlowHintMask(I);
  const unsigned BranchMask =
      1 << (unsigned)(DXIL::ControlFlowHint::Branch) |
      1 << (unsigned)(DXIL::ControlFlowHint::Call) |
      1 << (unsigned)(DXIL::ControlFlowHint::ForceCase);
  return mask & BranchMask;
}

void DxilMDHelper::EmitSubobjects(const DxilSubobjects &Subobjects) {
  NamedMDNode *pSubobjectsNamedMD = m_pModule->getNamedMetadata(kDxilSubobjectsMDName);
  IFTBOOL(pSubobjectsNamedMD == nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  pSubobjectsNamedMD = m_pModule->getOrInsertNamedMetadata(kDxilSubobjectsMDName);

  const auto &objMap = Subobjects.GetSubobjects();
  for (auto &it : objMap)
    pSubobjectsNamedMD->addOperand(cast<MDNode>(EmitSubobject(*it.second)));
}

void DxilMDHelper::LoadSubobjects(DxilSubobjects &Subobjects) {
  NamedMDNode *pSubobjectsNamedMD = m_pModule->getNamedMetadata(kDxilSubobjectsMDName);
  if (!pSubobjectsNamedMD)
    return;

  for (unsigned i = 0; i < pSubobjectsNamedMD->getNumOperands(); ++i)
    LoadSubobject(*pSubobjectsNamedMD->getOperand(i), Subobjects);
}

Metadata *DxilMDHelper::EmitSubobject(const DxilSubobject &obj) {
  SmallVector<Metadata *, 6> Args;
  Args.emplace_back(MDString::get(m_Ctx, obj.GetName()));
  Args.emplace_back(Uint32ToConstMD((unsigned)obj.GetKind()));
  bool bLocalRS = false;
  IFTBOOL(DXIL::IsValidSubobjectKind(obj.GetKind()), DXC_E_INCORRECT_DXIL_METADATA);
  switch (obj.GetKind()) {
  case DXIL::SubobjectKind::StateObjectConfig: {
    uint32_t Flags;
    IFTBOOL(obj.GetStateObjectConfig(Flags),
      DXC_E_INCORRECT_DXIL_METADATA);
    Args.emplace_back(Uint32ToConstMD((unsigned)Flags));
    break;
  }
  case DXIL::SubobjectKind::LocalRootSignature:
    bLocalRS = true;
    __fallthrough;
  case DXIL::SubobjectKind::GlobalRootSignature: {
    const char * Text;
    const void * Data;
    uint32_t Size;
    IFTBOOL(obj.GetRootSignature(bLocalRS, Data, Size, &Text),
      DXC_E_INCORRECT_DXIL_METADATA);
    Constant *V = ConstantDataArray::get(m_Ctx,
      ArrayRef<uint8_t>((const uint8_t *)Data, Size));
    Args.emplace_back(MDNode::get(m_Ctx, { ConstantAsMetadata::get(V) }));
    Args.emplace_back(MDString::get(m_Ctx, Text));
    break;
  }
  case DXIL::SubobjectKind::SubobjectToExportsAssociation: {
    StringRef Subobj;
    const char * const * Exports;
    uint32_t NumExports;
    IFTBOOL(obj.GetSubobjectToExportsAssociation(Subobj, Exports, NumExports),
      DXC_E_INCORRECT_DXIL_METADATA);
    SmallVector<Metadata *, 4> strArgs;
    for (unsigned i = 0; i < NumExports; ++i) {
      strArgs.emplace_back(MDString::get(m_Ctx, Exports[i]));
    }
    Args.emplace_back(MDString::get(m_Ctx, Subobj));
    Args.emplace_back(MDNode::get(m_Ctx, strArgs));
    break;
  }
  case DXIL::SubobjectKind::RaytracingShaderConfig: {
    uint32_t MaxPayloadSizeInBytes;
    uint32_t MaxAttributeSizeInBytes;
    IFTBOOL(obj.GetRaytracingShaderConfig(MaxPayloadSizeInBytes,
                                          MaxAttributeSizeInBytes),
      DXC_E_INCORRECT_DXIL_METADATA);
    Args.emplace_back(Uint32ToConstMD(MaxPayloadSizeInBytes));
    Args.emplace_back(Uint32ToConstMD(MaxAttributeSizeInBytes));
    break;
  }
  case DXIL::SubobjectKind::RaytracingPipelineConfig: {
    uint32_t MaxTraceRecursionDepth;
    IFTBOOL(obj.GetRaytracingPipelineConfig(MaxTraceRecursionDepth),
      DXC_E_INCORRECT_DXIL_METADATA);
    Args.emplace_back(Uint32ToConstMD(MaxTraceRecursionDepth));
    break;
  }
  case DXIL::SubobjectKind::HitGroup: {
    llvm::StringRef Intersection, AnyHit, ClosestHit;
    DXIL::HitGroupType hgType;
    IFTBOOL(obj.GetHitGroup(hgType, Intersection, AnyHit, ClosestHit),
      DXC_E_INCORRECT_DXIL_METADATA);
    Args.emplace_back(Uint32ToConstMD((uint32_t)hgType));
    Args.emplace_back(MDString::get(m_Ctx, Intersection));
    Args.emplace_back(MDString::get(m_Ctx, AnyHit));
    Args.emplace_back(MDString::get(m_Ctx, ClosestHit));
    break;
  }
  case DXIL::SubobjectKind::RaytracingPipelineConfig1: {
    uint32_t MaxTraceRecursionDepth;
    uint32_t Flags;
    IFTBOOL(obj.GetRaytracingPipelineConfig1(MaxTraceRecursionDepth, Flags),
            DXC_E_INCORRECT_DXIL_METADATA);
    Args.emplace_back(Uint32ToConstMD(MaxTraceRecursionDepth));
    Args.emplace_back(Uint32ToConstMD(Flags));
    break;
  }
  default:
    DXASSERT(false, "otherwise, we didn't handle a valid subobject kind");
    m_bExtraMetadata = true;
    break;
  }
  return MDNode::get(m_Ctx, Args);
}
void DxilMDHelper::LoadSubobject(const llvm::MDNode &MD, DxilSubobjects &Subobjects) {
  IFTBOOL(MD.getNumOperands() >= 2, DXC_E_INCORRECT_DXIL_METADATA);
  unsigned i = 0;
  StringRef name(StringMDToStringRef(MD.getOperand(i++)));
  DXIL::SubobjectKind kind = (DXIL::SubobjectKind)ConstMDToUint32(MD.getOperand(i++));
  IFTBOOL(DXIL::IsValidSubobjectKind(kind), DXC_E_INCORRECT_DXIL_METADATA);
  bool bLocalRS = false;
  switch (kind) {
  case DXIL::SubobjectKind::StateObjectConfig: {
    uint32_t Flags = ConstMDToUint32(MD.getOperand(i++));
    IFTBOOL(0 == ((~(uint32_t)DXIL::StateObjectFlags::ValidMask) & Flags),
            DXC_E_INCORRECT_DXIL_METADATA);
    Subobjects.CreateStateObjectConfig(name, Flags);
    break;
  }
  case DXIL::SubobjectKind::LocalRootSignature:
    bLocalRS = true;
    __fallthrough;
  case DXIL::SubobjectKind::GlobalRootSignature: {
    const MDNode *pDataMDWrapper = dyn_cast<MDNode>(MD.getOperand(i++));
    IFTBOOL(pDataMDWrapper != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
    IFTBOOL(pDataMDWrapper->getNumOperands() == 1, DXC_E_INCORRECT_DXIL_METADATA);
    const ConstantAsMetadata *pDataMD = dyn_cast<ConstantAsMetadata>(pDataMDWrapper->getOperand(0));
    const ConstantDataArray *pData = dyn_cast<ConstantDataArray>(pDataMD->getValue());
    IFTBOOL(pData != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
    IFTBOOL(pData->getElementType() == Type::getInt8Ty(m_Ctx), DXC_E_INCORRECT_DXIL_METADATA);
    IFTBOOL(pData->getRawDataValues().size() < UINT_MAX &&
            (pData->getRawDataValues().size() & 3) == 0, DXC_E_INCORRECT_DXIL_METADATA);
    const void *Data = pData->getRawDataValues().begin();
    uint32_t Size = pData->getRawDataValues().size();
    StringRef Text(StringMDToStringRef(MD.getOperand(i++)));
    Subobjects.CreateRootSignature(name, bLocalRS, Data, Size, Text.size() ? &Text : nullptr);
    break;
  }
  case DXIL::SubobjectKind::SubobjectToExportsAssociation: {
    StringRef Subobj(StringMDToStringRef(MD.getOperand(i++)));
    const MDNode *exportMD = dyn_cast<MDNode>(MD.getOperand(i++));
    SmallVector<StringRef, 4> Exports;
    for (unsigned iExport = 0; iExport < exportMD->getNumOperands(); iExport++) {
      Exports.push_back(StringMDToStringRef(exportMD->getOperand(iExport)));
    }
    Subobjects.CreateSubobjectToExportsAssociation(name, Subobj, Exports.data(), Exports.size());
    break;
  }
  case DXIL::SubobjectKind::RaytracingShaderConfig: {
    uint32_t MaxPayloadSizeInBytes = ConstMDToUint32(MD.getOperand(i++));;
    uint32_t MaxAttributeSizeInBytes = ConstMDToUint32(MD.getOperand(i++));;
    Subobjects.CreateRaytracingShaderConfig(name, MaxPayloadSizeInBytes, MaxAttributeSizeInBytes);
    break;
  }
  case DXIL::SubobjectKind::RaytracingPipelineConfig: {
    uint32_t MaxTraceRecursionDepth = ConstMDToUint32(MD.getOperand(i++));;
    Subobjects.CreateRaytracingPipelineConfig(name, MaxTraceRecursionDepth);
    break;
  }
  case DXIL::SubobjectKind::HitGroup: {
    uint32_t hgType = ConstMDToUint32(MD.getOperand(i++));
    StringRef Intersection(StringMDToStringRef(MD.getOperand(i++)));
    StringRef AnyHit(StringMDToStringRef(MD.getOperand(i++)));
    StringRef ClosestHit(StringMDToStringRef(MD.getOperand(i++)));
    Subobjects.CreateHitGroup(name, (DXIL::HitGroupType)hgType, AnyHit, ClosestHit, Intersection);
    break;
  }
  case DXIL::SubobjectKind::RaytracingPipelineConfig1: {
    uint32_t MaxTraceRecursionDepth = ConstMDToUint32(MD.getOperand(i++));
    uint32_t Flags = ConstMDToUint32(MD.getOperand(i++));
    IFTBOOL(0 ==
                ((~(uint32_t)DXIL::RaytracingPipelineFlags::ValidMask) & Flags),
            DXC_E_INCORRECT_DXIL_METADATA);
    Subobjects.CreateRaytracingPipelineConfig1(name, MaxTraceRecursionDepth,
                                               Flags);
    break;
  }
  default:
    DXASSERT(false, "otherwise, we didn't handle a valid subobject kind");
    m_bExtraMetadata = true;
    break;
  }
}

MDTuple *DxilMDHelper::EmitDxilSampler(const DxilSampler &S) {
  Metadata *MDVals[kDxilSamplerNumFields];

  EmitDxilResourceBase(S, &MDVals[0]);

  // Sampler-specific fields.
  MDVals[kDxilSamplerType         ] = Uint32ToConstMD((unsigned)S.GetSamplerKind());

  // Name-value list of extended properties.
  MDVals[kDxilSamplerNameValueList] = nullptr;
  vector<Metadata *> MDExtraVals;
  m_ExtraPropertyHelper->EmitSamplerProperties(S, MDExtraVals);
  if (!MDExtraVals.empty()) {
    MDVals[kDxilSamplerNameValueList] = MDNode::get(m_Ctx, MDExtraVals);
  }

  return MDNode::get(m_Ctx, MDVals);
}

void DxilMDHelper::LoadDxilSampler(const MDOperand &MDO, DxilSampler &S) {
  IFTBOOL(MDO.get() != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  const MDTuple *pTupleMD = dyn_cast<MDTuple>(MDO.get());
  IFTBOOL(pTupleMD != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  IFTBOOL(pTupleMD->getNumOperands() == kDxilSamplerNumFields, DXC_E_INCORRECT_DXIL_METADATA);

  LoadDxilResourceBase(MDO, S);

  // Sampler-specific fields.
  S.SetSamplerKind((DxilSampler::SamplerKind)ConstMDToUint32(pTupleMD->getOperand(kDxilSamplerType)));

  // Name-value list of extended properties.
  m_ExtraPropertyHelper->m_bExtraMetadata = false;
  m_ExtraPropertyHelper->LoadSamplerProperties(pTupleMD->getOperand(kDxilSamplerNameValueList), S);
  m_bExtraMetadata |= m_ExtraPropertyHelper->m_bExtraMetadata;
}

const MDOperand &DxilMDHelper::GetResourceClass(llvm::MDNode *MD,
                                                DXIL::ResourceClass &RC) {
  IFTBOOL(MD->getNumOperands() >=
              DxilMDHelper::kHLDxilResourceAttributeNumFields,
          DXC_E_INCORRECT_DXIL_METADATA);
  RC = static_cast<DxilResource::Class>(ConstMDToUint32(
      MD->getOperand(DxilMDHelper::kHLDxilResourceAttributeClass)));
  return MD->getOperand(DxilMDHelper::kHLDxilResourceAttributeMeta);
}

void DxilMDHelper::LoadDxilResourceBaseFromMDNode(llvm::MDNode *MD,
                                                  DxilResourceBase &R) {
  DxilResource::Class RC = DxilResource::Class::Invalid;
  const MDOperand &Meta = GetResourceClass(MD, RC);

  switch (RC) {
  case DxilResource::Class::CBuffer: {
    DxilCBuffer CB;
    LoadDxilCBuffer(Meta, CB);
    R = CB;
  } break;
  case DxilResource::Class::Sampler: {
    DxilSampler S;
    LoadDxilSampler(Meta, S);
    R = S;
  } break;
  case DxilResource::Class::SRV: {
    DxilResource Res;
    LoadDxilSRV(Meta, Res);
    R = Res;
  } break;
  case DxilResource::Class::UAV: {
    DxilResource Res;
    LoadDxilUAV(Meta, Res);
    R = Res;
  } break;
  default:
    DXASSERT(0, "Invalid metadata");
  }
}

void DxilMDHelper::LoadDxilResourceFromMDNode(llvm::MDNode *MD,
                                              DxilResource &R) {
  DxilResource::Class RC = DxilResource::Class::Invalid;
  const MDOperand &Meta = GetResourceClass(MD, RC);

  switch (RC) {
  case DxilResource::Class::SRV: {
    LoadDxilSRV(Meta, R);
  } break;
  case DxilResource::Class::UAV: {
    LoadDxilUAV(Meta, R);
  } break;
  default:
    DXASSERT(0, "Invalid metadata");
  }
}

void DxilMDHelper::LoadDxilSamplerFromMDNode(llvm::MDNode *MD, DxilSampler &S) {
  DxilResource::Class RC = DxilResource::Class::Invalid;
  const MDOperand &Meta = GetResourceClass(MD, RC);

  switch (RC) {
  case DxilResource::Class::Sampler: {
    LoadDxilSampler(Meta, S);
  } break;
  default:
    DXASSERT(0, "Invalid metadata");
  }
}

//
// DxilExtraPropertyHelper shader-specific methods.
//
MDTuple *DxilMDHelper::EmitDxilGSState(DXIL::InputPrimitive Primitive, 
                                       unsigned MaxVertexCount, 
                                       unsigned ActiveStreamMask, 
                                       DXIL::PrimitiveTopology StreamPrimitiveTopology, 
                                       unsigned GSInstanceCount) {
  Metadata *MDVals[kDxilGSStateNumFields];

  MDVals[kDxilGSStateInputPrimitive      ] = Uint32ToConstMD((unsigned)Primitive);
  MDVals[kDxilGSStateMaxVertexCount      ] = Uint32ToConstMD(MaxVertexCount);
  MDVals[kDxilGSStateActiveStreamMask    ] = Uint32ToConstMD(ActiveStreamMask);
  MDVals[kDxilGSStateOutputStreamTopology] = Uint32ToConstMD((unsigned)StreamPrimitiveTopology);
  MDVals[kDxilGSStateGSInstanceCount     ] = Uint32ToConstMD(GSInstanceCount);

  return MDNode::get(m_Ctx, MDVals);
}

void DxilMDHelper::LoadDxilGSState(const MDOperand &MDO, 
                                   DXIL::InputPrimitive &Primitive,
                                   unsigned &MaxVertexCount, 
                                   unsigned &ActiveStreamMask,
                                   DXIL::PrimitiveTopology &StreamPrimitiveTopology,
                                   unsigned &GSInstanceCount) {
  IFTBOOL(MDO.get() != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  const MDTuple *pTupleMD = dyn_cast<MDTuple>(MDO.get());
  IFTBOOL(pTupleMD != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  IFTBOOL(pTupleMD->getNumOperands() == kDxilGSStateNumFields, DXC_E_INCORRECT_DXIL_METADATA);

  Primitive = (DXIL::InputPrimitive)ConstMDToUint32(pTupleMD->getOperand(kDxilGSStateInputPrimitive));
  MaxVertexCount = ConstMDToUint32(pTupleMD->getOperand(kDxilGSStateMaxVertexCount));
  ActiveStreamMask = ConstMDToUint32(pTupleMD->getOperand(kDxilGSStateActiveStreamMask));
  StreamPrimitiveTopology = (DXIL::PrimitiveTopology)ConstMDToUint32(pTupleMD->getOperand(kDxilGSStateOutputStreamTopology));
  GSInstanceCount = ConstMDToUint32(pTupleMD->getOperand(kDxilGSStateGSInstanceCount));
}

MDTuple *DxilMDHelper::EmitDxilDSState(DXIL::TessellatorDomain Domain, unsigned InputControlPointCount) {
  Metadata *MDVals[kDxilDSStateNumFields];

  MDVals[kDxilDSStateTessellatorDomain     ] = Uint32ToConstMD((unsigned)Domain);
  MDVals[kDxilDSStateInputControlPointCount] = Uint32ToConstMD(InputControlPointCount);

  return MDNode::get(m_Ctx, MDVals);
}

void DxilMDHelper::LoadDxilDSState(const MDOperand &MDO,
                                   DXIL::TessellatorDomain &Domain,
                                   unsigned &InputControlPointCount) {
  IFTBOOL(MDO.get() != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  const MDTuple *pTupleMD = dyn_cast<MDTuple>(MDO.get());
  IFTBOOL(pTupleMD != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  IFTBOOL(pTupleMD->getNumOperands() == kDxilDSStateNumFields, DXC_E_INCORRECT_DXIL_METADATA);

  Domain = (DXIL::TessellatorDomain)ConstMDToUint32(pTupleMD->getOperand(kDxilDSStateTessellatorDomain));
  InputControlPointCount = ConstMDToUint32(pTupleMD->getOperand(kDxilDSStateInputControlPointCount));
}

MDTuple *DxilMDHelper::EmitDxilHSState(Function *pPatchConstantFunction,
                                       unsigned InputControlPointCount,
                                       unsigned OutputControlPointCount,
                                       DXIL::TessellatorDomain TessDomain,
                                       DXIL::TessellatorPartitioning TessPartitioning,
                                       DXIL::TessellatorOutputPrimitive TessOutputPrimitive,
                                       float MaxTessFactor) {
  Metadata *MDVals[kDxilHSStateNumFields];

  MDVals[kDxilHSStatePatchConstantFunction     ] = ValueAsMetadata::get(pPatchConstantFunction);
  MDVals[kDxilHSStateInputControlPointCount    ] = Uint32ToConstMD(InputControlPointCount);
  MDVals[kDxilHSStateOutputControlPointCount   ] = Uint32ToConstMD(OutputControlPointCount);
  MDVals[kDxilHSStateTessellatorDomain         ] = Uint32ToConstMD((unsigned)TessDomain);
  MDVals[kDxilHSStateTessellatorPartitioning   ] = Uint32ToConstMD((unsigned)TessPartitioning);
  MDVals[kDxilHSStateTessellatorOutputPrimitive] = Uint32ToConstMD((unsigned)TessOutputPrimitive);
  MDVals[kDxilHSStateMaxTessellationFactor     ] = FloatToConstMD(MaxTessFactor);

  return MDNode::get(m_Ctx, MDVals);
}

void DxilMDHelper::LoadDxilHSState(const MDOperand &MDO,
                                   Function *&pPatchConstantFunction,
                                   unsigned &InputControlPointCount,
                                   unsigned &OutputControlPointCount,
                                   DXIL::TessellatorDomain &TessDomain,
                                   DXIL::TessellatorPartitioning &TessPartitioning,
                                   DXIL::TessellatorOutputPrimitive &TessOutputPrimitive,
                                   float &MaxTessFactor) {
  IFTBOOL(MDO.get() != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  const MDTuple *pTupleMD = dyn_cast<MDTuple>(MDO.get());
  IFTBOOL(pTupleMD != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  IFTBOOL(pTupleMD->getNumOperands() == kDxilHSStateNumFields, DXC_E_INCORRECT_DXIL_METADATA);

  pPatchConstantFunction  = dyn_cast<Function>(ValueMDToValue(pTupleMD->getOperand(kDxilHSStatePatchConstantFunction)));
  InputControlPointCount  = ConstMDToUint32(pTupleMD->getOperand(kDxilHSStateInputControlPointCount));
  OutputControlPointCount = ConstMDToUint32(pTupleMD->getOperand(kDxilHSStateOutputControlPointCount));
  TessDomain              = (DXIL::TessellatorDomain)ConstMDToUint32(pTupleMD->getOperand(kDxilHSStateTessellatorDomain));
  TessPartitioning        = (DXIL::TessellatorPartitioning)ConstMDToUint32(pTupleMD->getOperand(kDxilHSStateTessellatorPartitioning));
  TessOutputPrimitive     = (DXIL::TessellatorOutputPrimitive)ConstMDToUint32(pTupleMD->getOperand(kDxilHSStateTessellatorOutputPrimitive));
  MaxTessFactor           = ConstMDToFloat(pTupleMD->getOperand(kDxilHSStateMaxTessellationFactor));
}

MDTuple *DxilMDHelper::EmitDxilMSState(const unsigned *NumThreads,
                                       unsigned MaxVertexCount,
                                       unsigned MaxPrimitiveCount,
                                       DXIL::MeshOutputTopology OutputTopology,
                                       unsigned payloadSizeInBytes) {
  Metadata *MDVals[kDxilMSStateNumFields];
  vector<Metadata *> NumThreadVals;

  NumThreadVals.emplace_back(Uint32ToConstMD(NumThreads[0]));
  NumThreadVals.emplace_back(Uint32ToConstMD(NumThreads[1]));
  NumThreadVals.emplace_back(Uint32ToConstMD(NumThreads[2]));
  MDVals[kDxilMSStateNumThreads] = MDNode::get(m_Ctx, NumThreadVals);
  MDVals[kDxilMSStateMaxVertexCount] = Uint32ToConstMD(MaxVertexCount);
  MDVals[kDxilMSStateMaxPrimitiveCount] = Uint32ToConstMD(MaxPrimitiveCount);
  MDVals[kDxilMSStateOutputTopology] = Uint32ToConstMD((unsigned)OutputTopology);
  MDVals[kDxilMSStatePayloadSizeInBytes] = Uint32ToConstMD(payloadSizeInBytes);

  return MDNode::get(m_Ctx, MDVals);
}

void DxilMDHelper::LoadDxilMSState(const MDOperand &MDO,
                                   unsigned *NumThreads,
                                   unsigned &MaxVertexCount,
                                   unsigned &MaxPrimitiveCount,
                                   DXIL::MeshOutputTopology &OutputTopology,
                                   unsigned &payloadSizeInBytes) {
  IFTBOOL(MDO.get() != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  const MDTuple *pTupleMD = dyn_cast<MDTuple>(MDO.get());
  IFTBOOL(pTupleMD != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  IFTBOOL(pTupleMD->getNumOperands() == kDxilMSStateNumFields, DXC_E_INCORRECT_DXIL_METADATA);

  MDNode *pNode = cast<MDNode>(pTupleMD->getOperand(kDxilMSStateNumThreads));
  NumThreads[0] = ConstMDToUint32(pNode->getOperand(0));
  NumThreads[1] = ConstMDToUint32(pNode->getOperand(1));
  NumThreads[2] = ConstMDToUint32(pNode->getOperand(2));
  MaxVertexCount = ConstMDToUint32(pTupleMD->getOperand(kDxilMSStateMaxVertexCount));
  MaxPrimitiveCount = ConstMDToUint32(pTupleMD->getOperand(kDxilMSStateMaxPrimitiveCount));
  OutputTopology = (DXIL::MeshOutputTopology)ConstMDToUint32(pTupleMD->getOperand(kDxilMSStateOutputTopology));
  payloadSizeInBytes = ConstMDToUint32(pTupleMD->getOperand(kDxilMSStatePayloadSizeInBytes));
}

MDTuple *DxilMDHelper::EmitDxilASState(const unsigned *NumThreads, unsigned payloadSizeInBytes) {
  Metadata *MDVals[kDxilASStateNumFields];
  vector<Metadata *> NumThreadVals;

  NumThreadVals.emplace_back(Uint32ToConstMD(NumThreads[0]));
  NumThreadVals.emplace_back(Uint32ToConstMD(NumThreads[1]));
  NumThreadVals.emplace_back(Uint32ToConstMD(NumThreads[2]));
  MDVals[kDxilASStateNumThreads] = MDNode::get(m_Ctx, NumThreadVals);
  MDVals[kDxilASStatePayloadSizeInBytes] = Uint32ToConstMD(payloadSizeInBytes);

  return MDNode::get(m_Ctx, MDVals);
}

void DxilMDHelper::LoadDxilASState(const MDOperand &MDO, unsigned *NumThreads, unsigned &payloadSizeInBytes) {
  IFTBOOL(MDO.get() != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  const MDTuple *pTupleMD = dyn_cast<MDTuple>(MDO.get());
  IFTBOOL(pTupleMD != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  IFTBOOL(pTupleMD->getNumOperands() == kDxilASStateNumFields, DXC_E_INCORRECT_DXIL_METADATA);

  MDNode *pNode = cast<MDNode>(pTupleMD->getOperand(kDxilASStateNumThreads));
  NumThreads[0] = ConstMDToUint32(pNode->getOperand(0));
  NumThreads[1] = ConstMDToUint32(pNode->getOperand(1));
  NumThreads[2] = ConstMDToUint32(pNode->getOperand(2));
  payloadSizeInBytes = ConstMDToUint32(pTupleMD->getOperand(kDxilASStatePayloadSizeInBytes));
}

void DxilMDHelper::AddCounterIfNonZero(uint32_t value, StringRef name, vector<Metadata*> &MDVals) {
  if (value) {
    MDVals.emplace_back(MDString::get(m_Ctx, name));
    MDVals.emplace_back(Uint32ToConstMD(value));
  }
}

void DxilMDHelper::EmitDxilCounters(const DxilCounters &counters) {
  NamedMDNode *pDxilCountersMD = m_pModule->getNamedMetadata(kDxilCountersMDName);
  if (pDxilCountersMD)
    m_pModule->eraseNamedMetadata(pDxilCountersMD);

  vector<Metadata*> MDVals;
  // <py::lines('OPCODE-COUNTERS')>['AddCounterIfNonZero(counters.%s, "%s", MDVals);' % (c,c) for c in hctdb_instrhelp.get_counters()]</py>
  // OPCODE-COUNTERS:BEGIN
  AddCounterIfNonZero(counters.array_local_bytes, "array_local_bytes", MDVals);
  AddCounterIfNonZero(counters.array_local_ldst, "array_local_ldst", MDVals);
  AddCounterIfNonZero(counters.array_static_bytes, "array_static_bytes", MDVals);
  AddCounterIfNonZero(counters.array_static_ldst, "array_static_ldst", MDVals);
  AddCounterIfNonZero(counters.array_tgsm_bytes, "array_tgsm_bytes", MDVals);
  AddCounterIfNonZero(counters.array_tgsm_ldst, "array_tgsm_ldst", MDVals);
  AddCounterIfNonZero(counters.atomic, "atomic", MDVals);
  AddCounterIfNonZero(counters.barrier, "barrier", MDVals);
  AddCounterIfNonZero(counters.branches, "branches", MDVals);
  AddCounterIfNonZero(counters.fence, "fence", MDVals);
  AddCounterIfNonZero(counters.floats, "floats", MDVals);
  AddCounterIfNonZero(counters.gs_cut, "gs_cut", MDVals);
  AddCounterIfNonZero(counters.gs_emit, "gs_emit", MDVals);
  AddCounterIfNonZero(counters.insts, "insts", MDVals);
  AddCounterIfNonZero(counters.ints, "ints", MDVals);
  AddCounterIfNonZero(counters.sig_ld, "sig_ld", MDVals);
  AddCounterIfNonZero(counters.sig_st, "sig_st", MDVals);
  AddCounterIfNonZero(counters.tex_bias, "tex_bias", MDVals);
  AddCounterIfNonZero(counters.tex_cmp, "tex_cmp", MDVals);
  AddCounterIfNonZero(counters.tex_grad, "tex_grad", MDVals);
  AddCounterIfNonZero(counters.tex_load, "tex_load", MDVals);
  AddCounterIfNonZero(counters.tex_norm, "tex_norm", MDVals);
  AddCounterIfNonZero(counters.tex_store, "tex_store", MDVals);
  AddCounterIfNonZero(counters.uints, "uints", MDVals);
  // OPCODE-COUNTERS:END

  if (MDVals.size()) {
    pDxilCountersMD = m_pModule->getOrInsertNamedMetadata(kDxilCountersMDName);
    pDxilCountersMD->addOperand(MDNode::get(m_Ctx, MDVals));
  }
}

void DxilMDHelper::LoadCounterMD(const MDOperand &MDName, const MDOperand &MDValue, DxilCounters &counters) const {
  StringRef name = StringMDToStringRef(MDName);
  uint32_t value = ConstMDToUint32(MDValue);
  uint32_t *counter = LookupByName(name, counters);
  if (counter)
    *counter = value;
}

void DxilMDHelper::LoadDxilCounters(DxilCounters &counters) const {
  ZeroMemory(&counters, sizeof(counters));
  if (NamedMDNode *pDxilCountersMD = m_pModule->getNamedMetadata(kDxilCountersMDName)) {
    MDNode *pMDCounters = pDxilCountersMD->getOperand(0);
    for (unsigned i = 0; i < pMDCounters->getNumOperands(); i += 2) {
      LoadCounterMD(pMDCounters->getOperand(i), pMDCounters->getOperand(i+1), counters);
    }
  }
}


//
// DxilExtraPropertyHelper methods.
//
DxilMDHelper::ExtraPropertyHelper::ExtraPropertyHelper(Module *pModule)
: m_Ctx(pModule->getContext())
, m_pModule(pModule)
, m_bExtraMetadata(false) {
}

DxilExtraPropertyHelper::DxilExtraPropertyHelper(Module *pModule)
: ExtraPropertyHelper(pModule) {
}

void DxilExtraPropertyHelper::EmitSRVProperties(const DxilResource &SRV, std::vector<Metadata *> &MDVals) {
  // Element type for typed resource.
  if (!SRV.IsStructuredBuffer() && !SRV.IsRawBuffer()) {
    MDVals.emplace_back(DxilMDHelper::Uint32ToConstMD(DxilMDHelper::kDxilTypedBufferElementTypeTag, m_Ctx));
    MDVals.emplace_back(DxilMDHelper::Uint32ToConstMD((unsigned)SRV.GetCompType().GetKind(), m_Ctx));
  }
  // Element stride for structured buffer.
  if (SRV.IsStructuredBuffer()) {
    MDVals.emplace_back(DxilMDHelper::Uint32ToConstMD(DxilMDHelper::kDxilStructuredBufferElementStrideTag, m_Ctx));
    MDVals.emplace_back(DxilMDHelper::Uint32ToConstMD(SRV.GetElementStride(), m_Ctx));
  }
}

void DxilExtraPropertyHelper::LoadSRVProperties(const MDOperand &MDO, DxilResource &SRV) {
  SRV.SetElementStride(SRV.IsRawBuffer() ? 1 : 4);
  SRV.SetCompType(CompType());

  if (MDO.get() == nullptr) {
    return;
  }

  const MDTuple *pTupleMD = dyn_cast<MDTuple>(MDO.get());
  IFTBOOL(pTupleMD != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  IFTBOOL((pTupleMD->getNumOperands() & 0x1) == 0, DXC_E_INCORRECT_DXIL_METADATA);

  for (unsigned i = 0; i < pTupleMD->getNumOperands(); i += 2) {
    unsigned Tag = DxilMDHelper::ConstMDToUint32(pTupleMD->getOperand(i));
    const MDOperand &MDO = pTupleMD->getOperand(i + 1);

    switch (Tag) {
    case DxilMDHelper::kDxilTypedBufferElementTypeTag:
      DXASSERT_NOMSG(!SRV.IsStructuredBuffer() && !SRV.IsRawBuffer());
      SRV.SetCompType(CompType(DxilMDHelper::ConstMDToUint32(MDO)));
      break;
    case DxilMDHelper::kDxilStructuredBufferElementStrideTag:
      DXASSERT_NOMSG(SRV.IsStructuredBuffer());
      SRV.SetElementStride(DxilMDHelper::ConstMDToUint32(MDO));
      break;
    default:
      DXASSERT(false, "Unknown resource record tag");
      m_bExtraMetadata = true;
      break;
    }
  }
}

void DxilExtraPropertyHelper::EmitUAVProperties(const DxilResource &UAV, std::vector<Metadata *> &MDVals) {
  // Element type for typed RW resource.
  if (!UAV.IsStructuredBuffer() && !UAV.IsRawBuffer() && !UAV.GetCompType().IsInvalid()) {
    MDVals.emplace_back(DxilMDHelper::Uint32ToConstMD(DxilMDHelper::kDxilTypedBufferElementTypeTag, m_Ctx));
    MDVals.emplace_back(DxilMDHelper::Uint32ToConstMD((unsigned)UAV.GetCompType().GetKind(), m_Ctx));
  }
  // Element stride for structured RW buffer.
  if (UAV.IsStructuredBuffer()) {
    MDVals.emplace_back(DxilMDHelper::Uint32ToConstMD(DxilMDHelper::kDxilStructuredBufferElementStrideTag, m_Ctx));
    MDVals.emplace_back(DxilMDHelper::Uint32ToConstMD(UAV.GetElementStride(), m_Ctx));
  }
  // Sampler feedback kind
  if (UAV.IsFeedbackTexture()) {
    MDVals.emplace_back(DxilMDHelper::Uint32ToConstMD(DxilMDHelper::kDxilSamplerFeedbackKindTag, m_Ctx));
    MDVals.emplace_back(DxilMDHelper::Uint32ToConstMD((unsigned)UAV.GetSamplerFeedbackType(), m_Ctx));
  }
  // Whether resource is used for 64-bit atomic op
  if (DXIL::CompareVersions(m_ValMajor, m_ValMinor, 1, 6) >= 0 && UAV.HasAtomic64Use()) {
    MDVals.emplace_back(DxilMDHelper::Uint32ToConstMD(DxilMDHelper::kDxilAtomic64UseTag, m_Ctx));
    MDVals.emplace_back(DxilMDHelper::Uint32ToConstMD((unsigned)true, m_Ctx));
  }
}

void DxilExtraPropertyHelper::LoadUAVProperties(const MDOperand &MDO, DxilResource &UAV) {
  UAV.SetElementStride(UAV.IsRawBuffer() ? 1 : 4);
  UAV.SetCompType(CompType());

  if (MDO.get() == nullptr) {
    return;
  }

  const MDTuple *pTupleMD = dyn_cast<MDTuple>(MDO.get());
  IFTBOOL(pTupleMD != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  IFTBOOL((pTupleMD->getNumOperands() & 0x1) == 0, DXC_E_INCORRECT_DXIL_METADATA);

  for (unsigned i = 0; i < pTupleMD->getNumOperands(); i += 2) {
    unsigned Tag = DxilMDHelper::ConstMDToUint32(pTupleMD->getOperand(i));
    const MDOperand &MDO = pTupleMD->getOperand(i + 1);

    switch (Tag) {
    case DxilMDHelper::kDxilTypedBufferElementTypeTag:
      DXASSERT_NOMSG(!UAV.IsStructuredBuffer() && !UAV.IsRawBuffer());
      UAV.SetCompType(CompType(DxilMDHelper::ConstMDToUint32(MDO)));
      break;
    case DxilMDHelper::kDxilStructuredBufferElementStrideTag:
      DXASSERT_NOMSG(UAV.IsStructuredBuffer());
      UAV.SetElementStride(DxilMDHelper::ConstMDToUint32(MDO));
      break;
    case DxilMDHelper::kDxilSamplerFeedbackKindTag:
      DXASSERT_NOMSG(UAV.IsFeedbackTexture());
      UAV.SetSamplerFeedbackType((DXIL::SamplerFeedbackType)DxilMDHelper::ConstMDToUint32(MDO));
      break;
    case DxilMDHelper::kDxilAtomic64UseTag:
      UAV.SetHasAtomic64Use(DxilMDHelper::ConstMDToBool(MDO));
      break;
    default:
      DXASSERT(false, "Unknown resource record tag");
      m_bExtraMetadata = true;
      break;
    }
  }
}

void DxilExtraPropertyHelper::EmitCBufferProperties(const DxilCBuffer &CB, vector<Metadata *> &MDVals) {
  // Emit property to preserve tbuffer kind
  if (CB.GetKind() == DXIL::ResourceKind::TBuffer) {
    MDVals.emplace_back(DxilMDHelper::Uint32ToConstMD(DxilMDHelper::kHLCBufferIsTBufferTag, m_Ctx));
    MDVals.emplace_back(DxilMDHelper::BoolToConstMD(true, m_Ctx));
  }
}

void DxilExtraPropertyHelper::LoadCBufferProperties(const MDOperand &MDO, DxilCBuffer &CB) {
  if (MDO.get() == nullptr)
    return;

  const MDTuple *pTupleMD = dyn_cast<MDTuple>(MDO.get());
  IFTBOOL(pTupleMD != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  IFTBOOL((pTupleMD->getNumOperands() & 0x1) == 0, DXC_E_INCORRECT_DXIL_METADATA);

  // Override kind for tbuffer that has not yet been converted to SRV.
  CB.SetKind(DXIL::ResourceKind::CBuffer);
  for (unsigned i = 0; i < pTupleMD->getNumOperands(); i += 2) {
    unsigned Tag = DxilMDHelper::ConstMDToUint32(pTupleMD->getOperand(i));
    const MDOperand &MDO = pTupleMD->getOperand(i + 1);

    switch (Tag) {
    case DxilMDHelper::kHLCBufferIsTBufferTag:
      if (DxilMDHelper::ConstMDToBool(MDO)) {
        CB.SetKind(DXIL::ResourceKind::TBuffer);
      }
      break;
    default:
      DXASSERT(false, "Unknown cbuffer tag");
      m_bExtraMetadata = true;
      break;
    }
  }
}

void DxilExtraPropertyHelper::EmitSamplerProperties(const DxilSampler &S, std::vector<Metadata *> &MDVals) {
  // Nothing yet.
}

void DxilExtraPropertyHelper::LoadSamplerProperties(const MDOperand &MDO, DxilSampler &S) {
  // Nothing yet.
}

void DxilExtraPropertyHelper::EmitSignatureElementProperties(const DxilSignatureElement &SE, 
                                                             vector<Metadata *> &MDVals) {
  // Output stream, if non-zero.
  if (SE.GetOutputStream() != 0) {
    MDVals.emplace_back(DxilMDHelper::Uint32ToConstMD(DxilMDHelper::kDxilSignatureElementOutputStreamTag, m_Ctx));
    MDVals.emplace_back(DxilMDHelper::Uint32ToConstMD(SE.GetOutputStream(), m_Ctx));
  }

  // Mask of Dynamically indexed components.
  if (SE.GetDynIdxCompMask() != 0) {
    MDVals.emplace_back(DxilMDHelper::Uint32ToConstMD(DxilMDHelper::kDxilSignatureElementDynIdxCompMaskTag, m_Ctx));
    MDVals.emplace_back(DxilMDHelper::Uint32ToConstMD(SE.GetDynIdxCompMask(), m_Ctx));
  }

  if (SE.GetUsageMask() != 0 &&
      DXIL::CompareVersions(m_ValMajor, m_ValMinor, 1, 5) >= 0) {
    // Emitting this will not hurt old reatil loader (only asserts),
    // and is required for signatures to match in validation.
    MDVals.emplace_back(DxilMDHelper::Uint32ToConstMD(DxilMDHelper::kDxilSignatureElementUsageCompMaskTag, m_Ctx));
    MDVals.emplace_back(DxilMDHelper::Uint32ToConstMD(SE.GetUsageMask(), m_Ctx));
  }
}

void DxilExtraPropertyHelper::LoadSignatureElementProperties(const MDOperand &MDO, DxilSignatureElement &SE) {
  if (MDO.get() == nullptr)
    return;

  const MDTuple *pTupleMD = dyn_cast<MDTuple>(MDO.get());
  IFTBOOL(pTupleMD != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  IFTBOOL((pTupleMD->getNumOperands() & 0x1) == 0, DXC_E_INCORRECT_DXIL_METADATA);

  // Stream.
  for (unsigned i = 0; i < pTupleMD->getNumOperands(); i += 2) {
    unsigned Tag = DxilMDHelper::ConstMDToUint32(pTupleMD->getOperand(i));
    const MDOperand &MDO = pTupleMD->getOperand(i + 1);

    switch (Tag) {
    case DxilMDHelper::kDxilSignatureElementOutputStreamTag:
      SE.SetOutputStream(DxilMDHelper::ConstMDToUint32(MDO));
      break;
    case DxilMDHelper::kHLSignatureElementGlobalSymbolTag:
      break;
    case DxilMDHelper::kDxilSignatureElementDynIdxCompMaskTag:
      SE.SetDynIdxCompMask(DxilMDHelper::ConstMDToUint32(MDO));
      break;
    case DxilMDHelper::kDxilSignatureElementUsageCompMaskTag:
      SE.SetUsageMask(DxilMDHelper::ConstMDToUint32(MDO));
      break;
    default:
      DXASSERT(false, "Unknown signature element tag");
      m_bExtraMetadata = true;
      break;
    }
  }
}

//
// Utilities.
//
bool DxilMDHelper::IsKnownNamedMetaData(const llvm::NamedMDNode &Node) {
  StringRef name = Node.getName();
  for (unsigned i = 0; i < DxilMDNames.size(); i++) {
    if (name == DxilMDNames[i]) {
      return true;
    }
  }
  return false;
}

bool DxilMDHelper::IsKnownMetadataID(LLVMContext &Ctx, unsigned ID)
{
    SmallVector<unsigned, 2> IDs;
    GetKnownMetadataIDs(Ctx, &IDs);
    return std::find(IDs.begin(), IDs.end(), ID) != IDs.end();
}

void DxilMDHelper::GetKnownMetadataIDs(LLVMContext &Ctx, SmallVectorImpl<unsigned> *pIDs)
{
  SmallVector<StringRef, 4> Names;
  Ctx.getMDKindNames(Names);
  for (auto Name : Names) {
    if (Name == hlsl::DxilMDHelper::kDxilPreciseAttributeMDName ||
        Name == hlsl::DxilMDHelper::kDxilNonUniformAttributeMDName) {
      pIDs->push_back(Ctx.getMDKindID(Name));
    }
  }
}

void DxilMDHelper::combineDxilMetadata(llvm::Instruction *K,
                                       const llvm::Instruction *J) {
  if (IsMarkedNonUniform(J))
    MarkNonUniform(K);
  if (IsMarkedPrecise(J))
    MarkPrecise(K);
}

ConstantAsMetadata *DxilMDHelper::Int32ToConstMD(int32_t v, LLVMContext &Ctx) {
  return ConstantAsMetadata::get(Constant::getIntegerValue(IntegerType::get(Ctx, 32), APInt(32, v)));
}

ConstantAsMetadata *DxilMDHelper::Int32ToConstMD(int32_t v) {
  return DxilMDHelper::Int32ToConstMD(v, m_Ctx);
}

ConstantAsMetadata *DxilMDHelper::Uint32ToConstMD(unsigned v, LLVMContext &Ctx) {
  return ConstantAsMetadata::get(Constant::getIntegerValue(IntegerType::get(Ctx, 32), APInt(32, v)));
}

ConstantAsMetadata *DxilMDHelper::Uint32ToConstMD(unsigned v) {
  return DxilMDHelper::Uint32ToConstMD(v, m_Ctx);
}

ConstantAsMetadata *DxilMDHelper::Uint64ToConstMD(uint64_t v, LLVMContext &Ctx) {
  return ConstantAsMetadata::get(Constant::getIntegerValue(IntegerType::get(Ctx, 64), APInt(64, v)));
}
ConstantAsMetadata *DxilMDHelper::Uint64ToConstMD(uint64_t v) {
  return DxilMDHelper::Uint64ToConstMD(v, m_Ctx);
}
ConstantAsMetadata *DxilMDHelper::Int8ToConstMD(int8_t v) {
  return ConstantAsMetadata::get(Constant::getIntegerValue(IntegerType::get(m_Ctx, 8), APInt(8, v)));
}
ConstantAsMetadata *DxilMDHelper::Uint8ToConstMD(uint8_t v) {
  return ConstantAsMetadata::get(Constant::getIntegerValue(IntegerType::get(m_Ctx, 8), APInt(8, v)));
}

ConstantAsMetadata *DxilMDHelper::BoolToConstMD(bool v, LLVMContext &Ctx) {
  return ConstantAsMetadata::get(Constant::getIntegerValue(IntegerType::get(Ctx, 1), APInt(1, v ? 1 : 0)));
}
ConstantAsMetadata *DxilMDHelper::BoolToConstMD(bool v) {
  return DxilMDHelper::BoolToConstMD(v, m_Ctx);
}

ConstantAsMetadata *DxilMDHelper::FloatToConstMD(float v) {
  return ConstantAsMetadata::get(ConstantFP::get(m_Ctx, APFloat(v)));
}

int32_t DxilMDHelper::ConstMDToInt32(const MDOperand &MDO) {
  ConstantInt *pConst = mdconst::extract<ConstantInt>(MDO);
  IFTBOOL(pConst != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  return (int32_t)pConst->getZExtValue();
}

unsigned DxilMDHelper::ConstMDToUint32(const MDOperand &MDO) {
  ConstantInt *pConst = mdconst::extract<ConstantInt>(MDO);
  IFTBOOL(pConst != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  return (unsigned)pConst->getZExtValue();
}

uint64_t DxilMDHelper::ConstMDToUint64(const MDOperand &MDO) {
  ConstantInt *pConst = mdconst::extract<ConstantInt>(MDO);
  IFTBOOL(pConst != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  return pConst->getZExtValue();
}

int8_t DxilMDHelper::ConstMDToInt8(const MDOperand &MDO) {
  ConstantInt *pConst = mdconst::extract<ConstantInt>(MDO);
  IFTBOOL(pConst != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  return (int8_t)pConst->getZExtValue();
}

uint8_t DxilMDHelper::ConstMDToUint8(const MDOperand &MDO) {
  ConstantInt *pConst = mdconst::extract<ConstantInt>(MDO);
  IFTBOOL(pConst != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  return (uint8_t)pConst->getZExtValue();
}

bool DxilMDHelper::ConstMDToBool(const MDOperand &MDO) {
  ConstantInt *pConst = mdconst::extract<ConstantInt>(MDO);
  IFTBOOL(pConst != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  return pConst->getZExtValue() != 0;
}

float DxilMDHelper::ConstMDToFloat(const MDOperand &MDO) {
  ConstantFP *pConst = mdconst::extract<ConstantFP>(MDO);
  IFTBOOL(pConst != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  return pConst->getValueAPF().convertToFloat();
}

string DxilMDHelper::StringMDToString(const MDOperand &MDO) {
  MDString *pMDString = dyn_cast<MDString>(MDO.get());
  IFTBOOL(pMDString != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  return pMDString->getString().str();
}

StringRef DxilMDHelper::StringMDToStringRef(const MDOperand &MDO) {
  MDString *pMDString = dyn_cast<MDString>(MDO.get());
  IFTBOOL(pMDString != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  return pMDString->getString();
}

Value *DxilMDHelper::ValueMDToValue(const MDOperand &MDO) {
  IFTBOOL(MDO.get() != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  ValueAsMetadata *pValAsMD = dyn_cast<ValueAsMetadata>(MDO.get());
  IFTBOOL(pValAsMD != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  Value *pValue = pValAsMD->getValue();
  IFTBOOL(pValue != nullptr, DXC_E_INCORRECT_DXIL_METADATA);
  return pValue;
}

MDTuple *DxilMDHelper::Uint32VectorToConstMDTuple(const std::vector<unsigned> &Vec) {
  vector<Metadata *> MDVals;

  MDVals.resize(Vec.size());
  for (size_t i = 0; i < Vec.size(); i++) {
    MDVals[i] = Uint32ToConstMD(Vec[i]);
  }
  return MDNode::get(m_Ctx, MDVals);
}

void DxilMDHelper::ConstMDTupleToUint32Vector(MDTuple *pTupleMD, std::vector<unsigned> &Vec) {
  IFTBOOL(pTupleMD != nullptr, DXC_E_INCORRECT_DXIL_METADATA);

  Vec.resize(pTupleMD->getNumOperands());
  for (size_t i = 0; i < pTupleMD->getNumOperands(); i++) {
    Vec[i] = ConstMDToUint32(pTupleMD->getOperand(i));
  }
}

bool DxilMDHelper::IsMarkedPrecise(const Instruction *inst) {
  int32_t val = 0;
  if (MDNode *precise = inst->getMetadata(kDxilPreciseAttributeMDName)) {
    assert(precise->getNumOperands() == 1);
    val = ConstMDToInt32(precise->getOperand(0));
  }
  return val;
}

void DxilMDHelper::MarkPrecise(Instruction *I) {
  LLVMContext &Ctx = I->getContext();
  MDNode *preciseNode = MDNode::get(
    Ctx,
    { ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(Ctx), 1)) });

  I->setMetadata(DxilMDHelper::kDxilPreciseAttributeMDName, preciseNode);
}

bool DxilMDHelper::IsMarkedNonUniform(const Instruction *inst) {
  int32_t val = 0;
  if (MDNode *precise = inst->getMetadata(kDxilNonUniformAttributeMDName)) {
    assert(precise->getNumOperands() == 1);
    val = ConstMDToInt32(precise->getOperand(0));
  }
  return val;
}

void DxilMDHelper::MarkNonUniform(Instruction *I) {
  LLVMContext &Ctx = I->getContext();
  MDNode *preciseNode = MDNode::get(
    Ctx,
    { ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(Ctx), 1)) });

  I->setMetadata(DxilMDHelper::kDxilNonUniformAttributeMDName, preciseNode);
}

bool DxilMDHelper::GetVariableDebugLayout(llvm::DbgDeclareInst *inst,
    unsigned &StartOffsetInBits, std::vector<DxilDIArrayDim> &ArrayDims) {
  llvm::MDTuple *Tuple = dyn_cast_or_null<MDTuple>(inst->getMetadata(DxilMDHelper::kDxilVariableDebugLayoutMDName));
  if (Tuple == nullptr) return false;

  IFTBOOL(Tuple->getNumOperands() % 2 == 1, DXC_E_INCORRECT_DXIL_METADATA);

  StartOffsetInBits = ConstMDToUint32(Tuple->getOperand(0));

  for (unsigned Idx = 1; Idx < Tuple->getNumOperands(); Idx += 2) {
    DxilDIArrayDim ArrayDim = {};
    ArrayDim.StrideInBits = ConstMDToUint32(Tuple->getOperand(Idx + 0));
    ArrayDim.NumElements = ConstMDToUint32(Tuple->getOperand(Idx + 1));
    ArrayDims.emplace_back(ArrayDim);
  }

  return true;
}

void DxilMDHelper::SetVariableDebugLayout(llvm::DbgDeclareInst *inst,
    unsigned StartOffsetInBits, const std::vector<DxilDIArrayDim> &ArrayDims) {
  LLVMContext &Ctx = inst->getContext();

  std::vector<Metadata*> MDVals;
  MDVals.reserve(ArrayDims.size() + 1);
  MDVals.emplace_back(Uint32ToConstMD(StartOffsetInBits, Ctx));
  for (const DxilDIArrayDim &ArrayDim : ArrayDims) {
    MDVals.emplace_back(Uint32ToConstMD(ArrayDim.StrideInBits, Ctx));
    MDVals.emplace_back(Uint32ToConstMD(ArrayDim.NumElements, Ctx));
  }

  inst->setMetadata(DxilMDHelper::kDxilVariableDebugLayoutMDName, MDNode::get(Ctx, MDVals));
}

void DxilMDHelper::CopyMetadata(Instruction &I, Instruction &SrcInst, ArrayRef<unsigned> WL) {
  if (!SrcInst.hasMetadata())
    return;

  DenseSet<unsigned> WLS;
  for (unsigned M : WL)
    WLS.insert(M);

  // Otherwise, enumerate and copy over metadata from the old instruction to the
  // new one.
  SmallVector<std::pair<unsigned, MDNode *>, 4> TheMDs;
  SrcInst.getAllMetadataOtherThanDebugLoc(TheMDs);
  for (const auto &MD : TheMDs) {
    if (WL.empty() || WLS.count(MD.first))
      I.setMetadata(MD.first, MD.second);
  }
  if (WL.empty() || WLS.count(LLVMContext::MD_dbg))
    I.setDebugLoc(SrcInst.getDebugLoc());
}

} // namespace hlsl
