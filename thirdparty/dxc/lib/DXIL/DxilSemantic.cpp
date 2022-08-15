///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilSemantic.cpp                                                          //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/DXIL/DxilSigPoint.h"
#include "dxc/DXIL/DxilSemantic.h"
#include "dxc/DXIL/DxilSignature.h"
#include "dxc/DXIL/DxilShaderModel.h"
#include "dxc/Support/Global.h"

#include <string>

using std::string;

namespace hlsl {

//------------------------------------------------------------------------------
//
// Semantic class methods.
//
Semantic::Semantic(Kind Kind,
                   const char *pszName)
: m_Kind(Kind)
, m_pszName(pszName)
{
}

const Semantic *Semantic::GetByName(llvm::StringRef name) {
  if (!HasSVPrefix(name))
    return GetArbitrary();

  // The search is a simple linear scan as it is fairly infrequent operation and the list is short.
  // The search can be improved if linear traversal has inadequate performance.
  for (unsigned i = (unsigned)Kind::Arbitrary + 1; i < (unsigned)Kind::Invalid; i++) {
    if (name.compare_lower(ms_SemanticTable[i].m_pszName) == 0)
      return &ms_SemanticTable[i];
  }

  return GetInvalid();
}

const Semantic *Semantic::GetByName(llvm::StringRef Name, DXIL::SigPointKind sigPointKind,
                                    unsigned MajorVersion, unsigned MinorVersion) {
  return Get(GetByName(Name)->GetKind(), sigPointKind, MajorVersion, MinorVersion);
}

const Semantic *Semantic::Get(Kind kind) {
  if (kind < Kind::Invalid)
    return &Semantic::ms_SemanticTable[(unsigned)kind];
  return GetInvalid();
}

const Semantic *Semantic::Get(Kind kind, DXIL::SigPointKind sigPointKind,
                              unsigned MajorVersion, unsigned MinorVersion) {
  if (sigPointKind == DXIL::SigPointKind::Invalid)
    return GetInvalid();
  const Semantic* pSemantic = Get(kind);
  DXIL::SemanticInterpretationKind SI = SigPoint::GetInterpretation(pSemantic->GetKind(), sigPointKind, MajorVersion, MinorVersion);
  if(SI == DXIL::SemanticInterpretationKind::NA)
    return GetInvalid();
  if(SI == DXIL::SemanticInterpretationKind::Arb)
    return GetArbitrary();
  return pSemantic;
}

const Semantic *Semantic::GetInvalid() {
  return &Semantic::ms_SemanticTable[(unsigned)Kind::Invalid];
}
const Semantic *Semantic::GetArbitrary() {
  return &Semantic::ms_SemanticTable[(unsigned)Kind::Arbitrary];
}

bool Semantic::HasSVPrefix(llvm::StringRef Name) {
  return Name.size() >= 3 && (Name[0] == 'S' || Name[0] == 's') &&
      (Name[1] == 'V' || Name[1] == 'v') && Name[2] == '_';
}

void Semantic::DecomposeNameAndIndex(llvm::StringRef FullName, llvm::StringRef *pName, unsigned *pIndex) {
  unsigned L = FullName.size(), i;

  for (i = L; i > 0; i--) {
    char d = FullName[i - 1];
    if ('0' > d || d > '9')
      break;
  }

  *pName = FullName.substr(0, i);

  if (i < L)
    *pIndex = atoi(FullName.data() + i);
  else
    *pIndex = 0;
}

Semantic::Kind Semantic::GetKind() const {
  return m_Kind;
}

const char *Semantic::GetName() const {
  return m_pszName;
}

bool Semantic::IsArbitrary() const {
  return GetKind() == Kind::Arbitrary;
}

bool Semantic::IsInvalid() const {
  return m_Kind == Kind::Invalid;
}

typedef Semantic SP;
const Semantic Semantic::ms_SemanticTable[kNumSemanticRecords] = {
  // Kind                         Name
  SP(Kind::Arbitrary,             nullptr),
  SP(Kind::VertexID,              "SV_VertexID"),
  SP(Kind::InstanceID,            "SV_InstanceID"),
  SP(Kind::Position,              "SV_Position"),
  SP(Kind::RenderTargetArrayIndex,"SV_RenderTargetArrayIndex"),
  SP(Kind::ViewPortArrayIndex,    "SV_ViewportArrayIndex"),
  SP(Kind::ClipDistance,          "SV_ClipDistance"),
  SP(Kind::CullDistance,          "SV_CullDistance"),
  SP(Kind::OutputControlPointID,  "SV_OutputControlPointID"),
  SP(Kind::DomainLocation,        "SV_DomainLocation"),
  SP(Kind::PrimitiveID,           "SV_PrimitiveID"),
  SP(Kind::GSInstanceID,          "SV_GSInstanceID"),
  SP(Kind::SampleIndex,           "SV_SampleIndex"),
  SP(Kind::IsFrontFace,           "SV_IsFrontFace"),
  SP(Kind::Coverage,              "SV_Coverage"),
  SP(Kind::InnerCoverage,         "SV_InnerCoverage"),
  SP(Kind::Target,                "SV_Target"),
  SP(Kind::Depth,                 "SV_Depth"),
  SP(Kind::DepthLessEqual,        "SV_DepthLessEqual"),
  SP(Kind::DepthGreaterEqual,     "SV_DepthGreaterEqual"),
  SP(Kind::StencilRef,            "SV_StencilRef"),
  SP(Kind::DispatchThreadID,      "SV_DispatchThreadID"),
  SP(Kind::GroupID,               "SV_GroupID"),
  SP(Kind::GroupIndex,            "SV_GroupIndex"),
  SP(Kind::GroupThreadID,         "SV_GroupThreadID"),
  SP(Kind::TessFactor,            "SV_TessFactor"),
  SP(Kind::InsideTessFactor,      "SV_InsideTessFactor"),
  SP(Kind::ViewID,                "SV_ViewID"),
  SP(Kind::Barycentrics,          "SV_Barycentrics"),
  SP(Kind::ShadingRate,           "SV_ShadingRate"),
  SP(Kind::CullPrimitive,         "SV_CullPrimitive"),
  SP(Kind::Invalid,               nullptr),
};

} // namespace hlsl
