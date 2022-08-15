///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilTypeSystem.cpp                                                        //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/DXIL/DxilTypeSystem.h"
#include "dxc/DXIL/DxilModule.h"
#include "dxc/DXIL/DxilUtil.h"
#include "dxc/Support/Global.h"
#include "dxc/Support/WinFunctions.h"

#include "llvm/IR/Module.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using std::unique_ptr;
using std::string;
using std::vector;
using std::map;


namespace hlsl {

//------------------------------------------------------------------------------
//
// DxilMatrixAnnotation class methods.
//
DxilMatrixAnnotation::DxilMatrixAnnotation()
: Rows(0)
, Cols(0)
, Orientation(MatrixOrientation::Undefined) {
}


//------------------------------------------------------------------------------
//
// DxilFieldAnnotation class methods.
//
DxilFieldAnnotation::DxilFieldAnnotation()
: m_bPrecise(false)
, m_ResourceAttribute(nullptr)
, m_CBufferOffset(UINT_MAX)
, m_bCBufferVarUsed(false)
{}

bool DxilFieldAnnotation::IsPrecise() const { return m_bPrecise; }
void DxilFieldAnnotation::SetPrecise(bool b) { m_bPrecise = b; }
bool DxilFieldAnnotation::HasMatrixAnnotation() const { return m_Matrix.Cols != 0; }
const DxilMatrixAnnotation &DxilFieldAnnotation::GetMatrixAnnotation() const { return m_Matrix; }
void DxilFieldAnnotation::SetMatrixAnnotation(const DxilMatrixAnnotation &MA) { m_Matrix = MA; }
bool DxilFieldAnnotation::HasResourceAttribute() const {
  return m_ResourceAttribute;
}
llvm::MDNode *DxilFieldAnnotation::GetResourceAttribute() const {
  return m_ResourceAttribute;
}
void DxilFieldAnnotation::SetResourceAttribute(llvm::MDNode *MD) {
  m_ResourceAttribute = MD;
}
bool DxilFieldAnnotation::HasCBufferOffset() const { return m_CBufferOffset != UINT_MAX; }
unsigned DxilFieldAnnotation::GetCBufferOffset() const { return m_CBufferOffset; }
void DxilFieldAnnotation::SetCBufferOffset(unsigned Offset) { m_CBufferOffset = Offset; }
bool DxilFieldAnnotation::HasCompType() const { return m_CompType.GetKind() != CompType::Kind::Invalid; }
const CompType &DxilFieldAnnotation::GetCompType() const { return m_CompType; }
void DxilFieldAnnotation::SetCompType(CompType::Kind kind) { m_CompType = CompType(kind); }
bool DxilFieldAnnotation::HasSemanticString() const { return !m_Semantic.empty(); }
const std::string &DxilFieldAnnotation::GetSemanticString() const { return m_Semantic; }
llvm::StringRef DxilFieldAnnotation::GetSemanticStringRef() const { return llvm::StringRef(m_Semantic); }
void DxilFieldAnnotation::SetSemanticString(const std::string &SemString) { m_Semantic = SemString; }
bool DxilFieldAnnotation::HasInterpolationMode() const { return !m_InterpMode.IsUndefined(); }
const InterpolationMode &DxilFieldAnnotation::GetInterpolationMode() const { return m_InterpMode; }
void DxilFieldAnnotation::SetInterpolationMode(const InterpolationMode &IM) { m_InterpMode = IM; }
bool DxilFieldAnnotation::HasFieldName() const { return !m_FieldName.empty(); }
const std::string &DxilFieldAnnotation::GetFieldName() const { return m_FieldName; }
void DxilFieldAnnotation::SetFieldName(const std::string &FieldName) { m_FieldName = FieldName; }
bool DxilFieldAnnotation::IsCBVarUsed() const { return m_bCBufferVarUsed; }
void DxilFieldAnnotation::SetCBVarUsed(bool used) { m_bCBufferVarUsed = used; }

//------------------------------------------------------------------------------
//
// DxilPayloadFieldAnnotation class methods.
//
bool DxilPayloadFieldAnnotation::HasCompType() const { return m_CompType.GetKind() != CompType::Kind::Invalid; }
const CompType &DxilPayloadFieldAnnotation::GetCompType() const { return m_CompType; }
void DxilPayloadFieldAnnotation::SetCompType(CompType::Kind kind) { m_CompType = CompType(kind); }
uint32_t DxilPayloadFieldAnnotation::GetPayloadFieldQualifierMask() const {
  return m_bitmask;
}

unsigned DxilPayloadFieldAnnotation::GetBitOffsetForShaderStage(DXIL::PayloadAccessShaderStage shaderStage ) {
  unsigned bitOffset = static_cast<unsigned>(shaderStage) *
                       DXIL::PayloadAccessQualifierBitsPerStage;
  return bitOffset;
}

void DxilPayloadFieldAnnotation::SetPayloadFieldQualifierMask(uint32_t fieldBitmask) {
  DXASSERT((fieldBitmask & ~DXIL::PayloadAccessQualifierValidMask) == 0,
           "Unknown payload access qualifier bits set");
  m_bitmask = fieldBitmask & DXIL::PayloadAccessQualifierValidMask;
}

void DxilPayloadFieldAnnotation::AddPayloadFieldQualifier(
    DXIL::PayloadAccessShaderStage shaderStage, DXIL::PayloadAccessQualifier qualifier) {
  unsigned accessBits = static_cast<unsigned>(qualifier);
  DXASSERT((accessBits & ~DXIL::PayloadAccessQualifierValidMaskPerStage) == 0,
           "Unknown payload access qualifier bits set");
  accessBits &= DXIL::PayloadAccessQualifierValidMaskPerStage;

  accessBits <<= GetBitOffsetForShaderStage(shaderStage);
  m_bitmask |= accessBits;
}

DXIL::PayloadAccessQualifier DxilPayloadFieldAnnotation::GetPayloadFieldQualifier(
    DXIL::PayloadAccessShaderStage shaderStage) const {

  int bitOffset = GetBitOffsetForShaderStage(shaderStage);

  // default type is always ReadWrite
  DXIL::PayloadAccessQualifier accessType = DXIL::PayloadAccessQualifier::ReadWrite;

  const unsigned readBit = static_cast<unsigned>(DXIL::PayloadAccessQualifier::Read);
  const unsigned writeBit = static_cast<unsigned>(DXIL::PayloadAccessQualifier::Write);

  unsigned accessBits = m_bitmask >> bitOffset;
  if (accessBits & readBit) {
    // set Read if the first bit is set
    accessType = DXIL::PayloadAccessQualifier::Read;
  }
  if (accessBits & writeBit) {

    // set Write only if the second bit set, if both are set set to ReadWrite
    accessType = accessType == DXIL::PayloadAccessQualifier::ReadWrite
                     ? DXIL::PayloadAccessQualifier::Write
                     : DXIL::PayloadAccessQualifier::ReadWrite;
  }
  return accessType;
}

bool DxilPayloadFieldAnnotation::HasAnnotations() const {
  return m_bitmask != 0;
}

//------------------------------------------------------------------------------
//
// DxilStructAnnotation class methods.
//
DxilTemplateArgAnnotation::DxilTemplateArgAnnotation()
    : m_Type(nullptr), m_Integral(0)
{}

bool DxilTemplateArgAnnotation::IsType() const { return m_Type != nullptr; }
const llvm::Type *DxilTemplateArgAnnotation::GetType() const { return m_Type; }
void DxilTemplateArgAnnotation::SetType(const llvm::Type *pType) { m_Type = pType; }

bool DxilTemplateArgAnnotation::IsIntegral() const { return m_Type == nullptr; }
int64_t DxilTemplateArgAnnotation::GetIntegral() const { return m_Integral; }
void DxilTemplateArgAnnotation::SetIntegral(int64_t i64) { m_Type = nullptr; m_Integral = i64; }

unsigned DxilStructAnnotation::GetNumFields() const {
  return (unsigned)m_FieldAnnotations.size();
}

DxilFieldAnnotation &DxilStructAnnotation::GetFieldAnnotation(unsigned FieldIdx) {
  return m_FieldAnnotations[FieldIdx];
}

const DxilFieldAnnotation &DxilStructAnnotation::GetFieldAnnotation(unsigned FieldIdx) const {
  return m_FieldAnnotations[FieldIdx];
}

const StructType *DxilStructAnnotation::GetStructType() const {
  return m_pStructType;
}
void DxilStructAnnotation::SetStructType(const llvm::StructType *Ty) {
  m_pStructType = Ty;
}


unsigned DxilStructAnnotation::GetCBufferSize() const { return m_CBufferSize; }
void DxilStructAnnotation::SetCBufferSize(unsigned size) { m_CBufferSize = size; }
void DxilStructAnnotation::MarkEmptyStruct() {
  if (m_ResourcesContained == HasResources::True)
    m_ResourcesContained = HasResources::Only;
  else
    m_FieldAnnotations.clear();
}
bool DxilStructAnnotation::IsEmptyStruct() {
  return m_FieldAnnotations.empty();
}
bool DxilStructAnnotation::IsEmptyBesidesResources() {
  return m_ResourcesContained == HasResources::Only ||
         m_FieldAnnotations.empty();
}

// ContainsResources is for codegen only, not meant for metadata
void DxilStructAnnotation::SetContainsResources() {
  if (m_ResourcesContained == HasResources::False)
    m_ResourcesContained = HasResources::True;
}
bool DxilStructAnnotation::ContainsResources() const { return m_ResourcesContained != HasResources::False; }

// For template args, GetNumTemplateArgs() will return 0 if not a template
unsigned DxilStructAnnotation::GetNumTemplateArgs() const {
  return (unsigned)m_TemplateAnnotations.size();
}
void DxilStructAnnotation::SetNumTemplateArgs(unsigned count) {
  DXASSERT(m_TemplateAnnotations.empty(), "template args already initialized");
  m_TemplateAnnotations.resize(count);
}
DxilTemplateArgAnnotation &DxilStructAnnotation::GetTemplateArgAnnotation(unsigned argIdx) {
  return m_TemplateAnnotations[argIdx];
}
const DxilTemplateArgAnnotation &DxilStructAnnotation::GetTemplateArgAnnotation(unsigned argIdx) const {
  return m_TemplateAnnotations[argIdx];
}

//------------------------------------------------------------------------------
//
// DxilParameterAnnotation class methods.
//
DxilParameterAnnotation::DxilParameterAnnotation()
: DxilFieldAnnotation(), m_inputQual(DxilParamInputQual::In) {
}

DxilParamInputQual DxilParameterAnnotation::GetParamInputQual() const {
  return m_inputQual;
}
void DxilParameterAnnotation::SetParamInputQual(DxilParamInputQual qual) {
  m_inputQual = qual;
}

const std::vector<unsigned> &DxilParameterAnnotation::GetSemanticIndexVec() const {
  return m_semanticIndex;
}

void DxilParameterAnnotation::SetSemanticIndexVec(const std::vector<unsigned> &Vec) {
  m_semanticIndex = Vec;
}

void DxilParameterAnnotation::AppendSemanticIndex(unsigned SemIdx) {
  m_semanticIndex.emplace_back(SemIdx);
}

//------------------------------------------------------------------------------
//
// DxilFunctionAnnotation class methods.
//
unsigned DxilFunctionAnnotation::GetNumParameters() const {
  return (unsigned)m_parameterAnnotations.size();
}

DxilParameterAnnotation &DxilFunctionAnnotation::GetParameterAnnotation(unsigned ParamIdx) {
  return m_parameterAnnotations[ParamIdx];
}

const DxilParameterAnnotation &DxilFunctionAnnotation::GetParameterAnnotation(unsigned ParamIdx) const {
  return m_parameterAnnotations[ParamIdx];
}

DxilParameterAnnotation &DxilFunctionAnnotation::GetRetTypeAnnotation() {
  return m_retTypeAnnotation;
}

const DxilParameterAnnotation &DxilFunctionAnnotation::GetRetTypeAnnotation() const {
  return m_retTypeAnnotation;
}

const Function *DxilFunctionAnnotation::GetFunction() const {
  return m_pFunction;
}

//------------------------------------------------------------------------------
//
// DxilPayloadAnnotation class methods.
//
unsigned DxilPayloadAnnotation::GetNumFields() const {
  return (unsigned)m_FieldAnnotations.size();
}

DxilPayloadFieldAnnotation &DxilPayloadAnnotation::GetFieldAnnotation(unsigned FieldIdx) {
  return m_FieldAnnotations[FieldIdx];
}

const DxilPayloadFieldAnnotation &DxilPayloadAnnotation::GetFieldAnnotation(unsigned FieldIdx) const {
  return m_FieldAnnotations[FieldIdx];
}

const StructType *DxilPayloadAnnotation::GetStructType() const {
  return m_pStructType;
}
void DxilPayloadAnnotation::SetStructType(const llvm::StructType *Ty) {
  m_pStructType = Ty;
}

//------------------------------------------------------------------------------
//
// DxilTypeSystem class methods.
//
DxilTypeSystem::DxilTypeSystem(Module *pModule)
    : m_pModule(pModule),
      m_LowPrecisionMode(DXIL::LowPrecisionMode::Undefined) {}

DxilStructAnnotation *DxilTypeSystem::AddStructAnnotation(const StructType *pStructType, unsigned numTemplateArgs) {
  DXASSERT_NOMSG(m_StructAnnotations.find(pStructType) == m_StructAnnotations.end());
  DxilStructAnnotation *pA = new DxilStructAnnotation();
  m_StructAnnotations[pStructType] = unique_ptr<DxilStructAnnotation>(pA);
  pA->m_pStructType = pStructType;
  pA->m_FieldAnnotations.resize(pStructType->getNumElements());
  pA->SetNumTemplateArgs(numTemplateArgs);
  return pA;
}

void DxilTypeSystem::FinishStructAnnotation(DxilStructAnnotation &SA) {
  const llvm::StructType *ST = SA.GetStructType();
  DXASSERT(SA.GetNumFields() == ST->getNumElements(), "otherwise, mismatched field count.");

  // Update resource containment
  for (unsigned i = 0; i < SA.GetNumFields() && !SA.ContainsResources(); i++) {
    if (IsResourceContained(ST->getElementType(i)))
      SA.SetContainsResources();
  }

  // Mark if empty
  if (SA.GetCBufferSize() == 0)
    SA.MarkEmptyStruct();
}

DxilStructAnnotation *DxilTypeSystem::GetStructAnnotation(const StructType *pStructType) {
  auto it = m_StructAnnotations.find(pStructType);
  if (it != m_StructAnnotations.end()) {
    return it->second.get();
  } else {
    return nullptr;
  }
}

const DxilStructAnnotation *
DxilTypeSystem::GetStructAnnotation(const StructType *pStructType) const {
  auto it = m_StructAnnotations.find(pStructType);
  if (it != m_StructAnnotations.end()) {
    return it->second.get();
  } else {
    return nullptr;
  }
}

void DxilTypeSystem::EraseStructAnnotation(const StructType *pStructType) {
  DXASSERT_NOMSG(m_StructAnnotations.count(pStructType));
  m_StructAnnotations.remove_if([pStructType](
      const std::pair<const StructType *, std::unique_ptr<DxilStructAnnotation>>
          &I) { return pStructType == I.first; });
}

// Recurse type, removing any found StructType from the set
static void RemoveUsedStructsFromSet(Type *Ty, std::unordered_set<const llvm::StructType*> &unused_structs) {
  if (Ty->isPointerTy())
    RemoveUsedStructsFromSet(Ty->getPointerElementType(), unused_structs);
  else if (Ty->isArrayTy())
    RemoveUsedStructsFromSet(Ty->getArrayElementType(), unused_structs);
  else if (Ty->isStructTy()) {
    StructType *ST = cast<StructType>(Ty);
    // Only recurse first time into this struct
    if (unused_structs.erase(ST)) {
      for (auto &ET : ST->elements()) {
        RemoveUsedStructsFromSet(ET, unused_structs);
      }
    }
  }
}

void DxilTypeSystem::EraseUnusedStructAnnotations() {
  // Add all structures with annotations to a set
  // Iterate globals, resource types, and functions, recursing used structures to
  // remove matching struct annotations from set
  std::unordered_set<const llvm::StructType*> unused_structs;
  for (auto &it : m_StructAnnotations) {
    unused_structs.insert(it.first);
  }
  for (auto &GV : m_pModule->globals()) {
    RemoveUsedStructsFromSet(GV.getType(), unused_structs);
  }
  DxilModule &DM = m_pModule->GetDxilModule();
  for (auto &&C : DM.GetCBuffers()) {
    RemoveUsedStructsFromSet(C->GetHLSLType(), unused_structs);
  }
  for (auto &&Srv : DM.GetSRVs()) {
    RemoveUsedStructsFromSet(Srv->GetHLSLType(), unused_structs);
  }
  for (auto &&Uav : DM.GetUAVs()) {
    RemoveUsedStructsFromSet(Uav->GetHLSLType(), unused_structs);
  }
  for (auto &F : m_pModule->functions()) {
    FunctionType *FT = F.getFunctionType();
    RemoveUsedStructsFromSet(FT->getReturnType(), unused_structs);
    for (auto &argTy : FT->params()) {
      RemoveUsedStructsFromSet(argTy, unused_structs);
    }
  }
  // erase remaining structures in set
  for (auto *ST : unused_structs) {
    EraseStructAnnotation(ST);
  }
}

DxilTypeSystem::StructAnnotationMap &DxilTypeSystem::GetStructAnnotationMap() {
  return m_StructAnnotations;
}

const DxilTypeSystem::StructAnnotationMap &DxilTypeSystem::GetStructAnnotationMap() const{
  return m_StructAnnotations;
}

DxilPayloadAnnotation *DxilTypeSystem::AddPayloadAnnotation(const StructType *pStructType) {
  DXASSERT_NOMSG(m_PayloadAnnotations.find(pStructType) == m_PayloadAnnotations.end());
  DxilPayloadAnnotation *pA = new DxilPayloadAnnotation();
  m_PayloadAnnotations[pStructType] = unique_ptr<DxilPayloadAnnotation>(pA);
  pA->m_pStructType = pStructType;
  pA->m_FieldAnnotations.resize(pStructType->getNumElements());
  return pA;
}

DxilPayloadAnnotation *DxilTypeSystem::GetPayloadAnnotation(const StructType *pStructType) {
  auto it = m_PayloadAnnotations.find(pStructType);
  if (it != m_PayloadAnnotations.end()) {
    return it->second.get();
  } else {
    return nullptr;
  }
}

const DxilPayloadAnnotation *
DxilTypeSystem::GetPayloadAnnotation(const StructType *pStructType) const {
  auto it = m_PayloadAnnotations.find(pStructType);
  if (it != m_PayloadAnnotations.end()) {
    return it->second.get();
  } else {
    return nullptr;
  }
}

void DxilTypeSystem::ErasePayloadAnnotation(const StructType *pStructType) {
  DXASSERT_NOMSG(m_StructAnnotations.count(pStructType));
  m_PayloadAnnotations.remove_if([pStructType](
      const std::pair<const StructType *, std::unique_ptr<DxilPayloadAnnotation>>
          &I) { return pStructType == I.first; });
}

DxilTypeSystem::PayloadAnnotationMap &DxilTypeSystem::GetPayloadAnnotationMap() {
  return m_PayloadAnnotations;
}

const DxilTypeSystem::PayloadAnnotationMap &DxilTypeSystem::GetPayloadAnnotationMap() const{
  return m_PayloadAnnotations;
}

DxilFunctionAnnotation *DxilTypeSystem::AddFunctionAnnotation(const Function *pFunction) {
  DXASSERT_NOMSG(m_FunctionAnnotations.find(pFunction) == m_FunctionAnnotations.end());
  DxilFunctionAnnotation *pA = new DxilFunctionAnnotation();
  m_FunctionAnnotations[pFunction] = unique_ptr<DxilFunctionAnnotation>(pA);
  pA->m_pFunction = pFunction;
  pA->m_parameterAnnotations.resize(pFunction->getFunctionType()->getNumParams());
  return pA;
}

void DxilTypeSystem::FinishFunctionAnnotation(DxilFunctionAnnotation &FA) {
  auto FT = FA.GetFunction()->getFunctionType();

  // Update resource containment
  if (IsResourceContained(FT->getReturnType()))
    FA.SetContainsResourceArgs();
  for (unsigned i = 0; i < FT->getNumParams() && !FA.ContainsResourceArgs(); i++) {
    if (IsResourceContained(FT->getParamType(i)))
      FA.SetContainsResourceArgs();
  }
}

DxilFunctionAnnotation *DxilTypeSystem::GetFunctionAnnotation(const Function *pFunction) {
  auto it = m_FunctionAnnotations.find(pFunction);
  if (it != m_FunctionAnnotations.end()) {
    return it->second.get();
  } else {
    return nullptr;
  }
}

const DxilFunctionAnnotation *
DxilTypeSystem::GetFunctionAnnotation(const Function *pFunction) const {
  auto it = m_FunctionAnnotations.find(pFunction);
  if (it != m_FunctionAnnotations.end()) {
    return it->second.get();
  } else {
    return nullptr;
  }
}

void DxilTypeSystem::EraseFunctionAnnotation(const Function *pFunction) {
  DXASSERT_NOMSG(m_FunctionAnnotations.count(pFunction));
  m_FunctionAnnotations.remove_if([pFunction](
      const std::pair<const Function *, std::unique_ptr<DxilFunctionAnnotation>>
          &I) { return pFunction == I.first; });
}

DxilTypeSystem::FunctionAnnotationMap &DxilTypeSystem::GetFunctionAnnotationMap() {
  return m_FunctionAnnotations;
}

StructType *DxilTypeSystem::GetSNormF32Type(unsigned NumComps) {
  return GetNormFloatType(CompType::getSNormF32(), NumComps);
}

StructType *DxilTypeSystem::GetUNormF32Type(unsigned NumComps) {
  return GetNormFloatType(CompType::getUNormF32(), NumComps);
}

StructType *DxilTypeSystem::GetNormFloatType(CompType CT, unsigned NumComps) {
  Type *pCompType = CT.GetLLVMType(m_pModule->getContext());
  DXASSERT_NOMSG(pCompType->isFloatTy());
  Type *pFieldType = pCompType;
  string TypeName;
  raw_string_ostream NameStream(TypeName);
  if (NumComps > 1) {
    (NameStream << "dx.types." << NumComps << "x" << CT.GetName()).flush();
    pFieldType = FixedVectorType::get(pFieldType, NumComps);
  } else {
    (NameStream << "dx.types." << CT.GetName()).flush();
  }
  StructType *pStructType = m_pModule->getTypeByName(TypeName);
  if (pStructType == nullptr) {
    pStructType = StructType::create(m_pModule->getContext(), pFieldType, TypeName);
    DxilStructAnnotation &TA = *AddStructAnnotation(pStructType);
    DxilFieldAnnotation &FA = TA.GetFieldAnnotation(0);
    FA.SetCompType(CT.GetKind());
    DXASSERT_NOMSG(CT.IsSNorm() || CT.IsUNorm());
  }
  return pStructType;
}

void DxilTypeSystem::CopyTypeAnnotation(const llvm::Type *Ty,
                                        const DxilTypeSystem &src) {
  if (isa<PointerType>(Ty))
    Ty = Ty->getPointerElementType();

  while (isa<ArrayType>(Ty))
    Ty = Ty->getArrayElementType();

  // Only struct type has annotation.
  if (!isa<StructType>(Ty))
    return;

  const StructType *ST = cast<StructType>(Ty);
  // Already exist.
  if (GetStructAnnotation(ST))
    return;

  if (const DxilStructAnnotation *annot = src.GetStructAnnotation(ST)) {
    DxilStructAnnotation *dstAnnot = AddStructAnnotation(ST);
    // Copy the annotation.
    *dstAnnot = *annot;
    // Copy field type annotations.
    for (Type *Ty : ST->elements()) {
      CopyTypeAnnotation(Ty, src);
    }
  }
}

void DxilTypeSystem::CopyFunctionAnnotation(const llvm::Function *pDstFunction,
                                            const llvm::Function *pSrcFunction,
                                            const DxilTypeSystem &src) {
  const DxilFunctionAnnotation *annot = src.GetFunctionAnnotation(pSrcFunction);
  // Don't have annotation.
  if (!annot)
    return;
  // Already exist.
  if (GetFunctionAnnotation(pDstFunction))
    return;

  DxilFunctionAnnotation *dstAnnot = AddFunctionAnnotation(pDstFunction);

  // Copy the annotation.
  *dstAnnot = *annot;
  dstAnnot->m_pFunction = pDstFunction;
  // Clone ret type annotation.
  CopyTypeAnnotation(pDstFunction->getReturnType(), src);
  // Clone param type annotations.
  for (const Argument &arg : pDstFunction->args()) {
    CopyTypeAnnotation(arg.getType(), src);
  }
}

DXIL::SigPointKind SigPointFromInputQual(DxilParamInputQual Q, DXIL::ShaderKind SK, bool isPC) {
  DXASSERT(Q != DxilParamInputQual::Inout, "Inout not expected for SigPointFromInputQual");
  switch (SK) {
  case DXIL::ShaderKind::Vertex:
    switch (Q) {
    case DxilParamInputQual::In:
      return DXIL::SigPointKind::VSIn;
    case DxilParamInputQual::Out:
      return DXIL::SigPointKind::VSOut;
    default:
      break;
    }
    break;
  case DXIL::ShaderKind::Hull:
    switch (Q) {
    case DxilParamInputQual::In:
      if (isPC)
        return DXIL::SigPointKind::PCIn;
      else
        return DXIL::SigPointKind::HSIn;
    case DxilParamInputQual::Out:
      if (isPC)
        return DXIL::SigPointKind::PCOut;
      else
        return DXIL::SigPointKind::HSCPOut;
    case DxilParamInputQual::InputPatch:
      return DXIL::SigPointKind::HSCPIn;
    case DxilParamInputQual::OutputPatch:
      return DXIL::SigPointKind::HSCPOut;
    default:
      break;
    }
    break;
  case DXIL::ShaderKind::Domain:
    switch (Q) {
    case DxilParamInputQual::In:
      return DXIL::SigPointKind::DSIn;
    case DxilParamInputQual::Out:
      return DXIL::SigPointKind::DSOut;
    case DxilParamInputQual::InputPatch:
    case DxilParamInputQual::OutputPatch:
      return DXIL::SigPointKind::DSCPIn;
    default:
      break;
    }
    break;
  case DXIL::ShaderKind::Geometry:
    switch (Q) {
    case DxilParamInputQual::In:
      return DXIL::SigPointKind::GSIn;
    case DxilParamInputQual::InputPrimitive:
      return DXIL::SigPointKind::GSVIn;
    case DxilParamInputQual::OutStream0:
    case DxilParamInputQual::OutStream1:
    case DxilParamInputQual::OutStream2:
    case DxilParamInputQual::OutStream3:
      return DXIL::SigPointKind::GSOut;
    default:
      break;
    }
    break;
  case DXIL::ShaderKind::Pixel:
    switch (Q) {
    case DxilParamInputQual::In:
      return DXIL::SigPointKind::PSIn;
    case DxilParamInputQual::Out:
      return DXIL::SigPointKind::PSOut;
    default:
      break;
    }
    break;
  case DXIL::ShaderKind::Compute:
    switch (Q) {
    case DxilParamInputQual::In:
      return DXIL::SigPointKind::CSIn;
    default:
      break;
    }
    break;
  case DXIL::ShaderKind::Mesh:
    switch (Q) {
    case DxilParamInputQual::In:
    case DxilParamInputQual::InPayload:
      return DXIL::SigPointKind::MSIn;
    case DxilParamInputQual::OutIndices:
    case DxilParamInputQual::OutVertices:
      return DXIL::SigPointKind::MSOut;
    case DxilParamInputQual::OutPrimitives:
      return DXIL::SigPointKind::MSPOut;
    default:
      break;
    }
    break;
  case DXIL::ShaderKind::Amplification:
    switch (Q) {
    case DxilParamInputQual::In:
      return DXIL::SigPointKind::ASIn;
    default:
      break;
    }
    break;
  default:
    break;
  }
  return DXIL::SigPointKind::Invalid;
}

void RemapSemantic(llvm::StringRef &oldSemName, llvm::StringRef &oldSemFullName, const char *newSemName,
  DxilParameterAnnotation &paramInfo, llvm::LLVMContext &Context) {
  // format deprecation warning
  dxilutil::EmitWarningOnContext(
      Context, Twine("DX9-style semantic \"") + oldSemName +
                   Twine("\" mapped to DX10 system semantic \"") + newSemName +
                   Twine("\" due to -Gec flag. This functionality is "
                         "deprecated in newer language versions."));

  // create new semantic name with the same index
  std::string newSemNameStr(newSemName);
  unsigned indexLen = oldSemFullName.size() - oldSemName.size();
  if (indexLen > 0) {
    newSemNameStr = newSemNameStr.append(oldSemFullName.data() + oldSemName.size(), indexLen);
  }

  paramInfo.SetSemanticString(newSemNameStr);
}

void RemapObsoleteSemantic(DxilParameterAnnotation &paramInfo, DXIL::SigPointKind sigPoint, llvm::LLVMContext &Context) {
  DXASSERT(paramInfo.HasSemanticString(), "expected paramInfo with semantic");
  //*ppWarningMsg = nullptr;

  llvm::StringRef semFullName = paramInfo.GetSemanticStringRef();
  llvm::StringRef semName;
  unsigned semIndex;
  Semantic::DecomposeNameAndIndex(semFullName, &semName, &semIndex);

  if (sigPoint == DXIL::SigPointKind::PSOut) {
    if (semName.size() == 5) {
      if (_strnicmp(semName.data(), "COLOR", 5) == 0) {
        RemapSemantic(semName, semFullName, "SV_Target", paramInfo, Context);
      }
      else if (_strnicmp(semName.data(), "DEPTH", 5) == 0) {
        RemapSemantic(semName, semFullName, "SV_Depth", paramInfo, Context);
      }
    }
  }
  else if ((sigPoint == DXIL::SigPointKind::VSOut && semName.size() == 8 && _strnicmp(semName.data(), "POSITION", 8) == 0) ||
           (sigPoint == DXIL::SigPointKind::PSIn  && semName.size() == 4 && _strnicmp(semName.data(), "VPOS", 4) == 0)) {
    RemapSemantic(semName, semFullName, "SV_Position", paramInfo, Context);
  }
}

bool DxilTypeSystem::UseMinPrecision() {
  return m_LowPrecisionMode == DXIL::LowPrecisionMode::UseMinPrecision;
}

void DxilTypeSystem::SetMinPrecision(bool bMinPrecision) {
  DXIL::LowPrecisionMode mode =
      bMinPrecision ? DXIL::LowPrecisionMode::UseMinPrecision
                    : DXIL::LowPrecisionMode::UseNativeLowPrecision;
  DXASSERT((mode == m_LowPrecisionMode ||
            m_LowPrecisionMode == DXIL::LowPrecisionMode::Undefined),
           "LowPrecisionMode should only be set once.");

  m_LowPrecisionMode = mode;
}

bool DxilTypeSystem::IsResourceContained(llvm::Type *Ty) {
  // strip pointer/array
  if (Ty->isPointerTy())
    Ty = Ty->getPointerElementType();
  if (Ty->isArrayTy())
    Ty = Ty->getArrayElementType();

  if (auto ST = dyn_cast<StructType>(Ty)) {
    if (dxilutil::IsHLSLResourceType(Ty)) {
      return true;
    } else if (auto SA = GetStructAnnotation(ST)) {
      if (SA->ContainsResources())
        return true;
    }
  }
  return false;
}

DxilStructTypeIterator::DxilStructTypeIterator(llvm::StructType *sTy, DxilStructAnnotation *sAnnotation,
  unsigned idx)
  : STy(sTy), SAnnotation(sAnnotation), index(idx) {
  DXASSERT(
    sTy->getNumElements() == sAnnotation->GetNumFields(),
    "Otherwise the pairing of annotation and struct type does not match.");
}

// prefix
DxilStructTypeIterator &DxilStructTypeIterator::operator++() {
  index++;
  return *this;
}
// postfix
DxilStructTypeIterator DxilStructTypeIterator::operator++(int) {
  DxilStructTypeIterator iter(STy, SAnnotation, index);
  index++;
  return iter;
}

bool DxilStructTypeIterator::operator==(DxilStructTypeIterator iter) {
  return iter.STy == STy && iter.SAnnotation == SAnnotation &&
    iter.index == index;
}

bool DxilStructTypeIterator::operator!=(DxilStructTypeIterator iter) { return !(operator==(iter)); }

std::pair<llvm::Type *, DxilFieldAnnotation *> DxilStructTypeIterator::operator*() {
  return std::pair<llvm::Type *, DxilFieldAnnotation *>(
    STy->getElementType(index), &SAnnotation->GetFieldAnnotation(index));
}

DxilStructTypeIterator begin(llvm::StructType *STy, DxilStructAnnotation *SAnno) {
  return { STy, SAnno, 0 };
}

DxilStructTypeIterator end(llvm::StructType *STy, DxilStructAnnotation *SAnno) {
  return { STy, SAnno, STy->getNumElements() };
}

} // namespace hlsl
