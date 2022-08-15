///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilSignature.cpp                                                         //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/Support/Global.h"
#include "dxc/DXIL/DxilSignature.h"
#include "dxc/DXIL/DxilSigPoint.h"
#include "llvm/ADT/STLExtras.h"

using std::vector;
using std::unique_ptr;


namespace hlsl {

//------------------------------------------------------------------------------
//
// Singnature methods.
//
DxilSignature::DxilSignature(DXIL::ShaderKind shaderKind,
                             DXIL::SignatureKind sigKind,
                             bool useMinPrecision)
    : m_sigPointKind(SigPoint::GetKind(shaderKind, sigKind,
                                       /*isPatchConstantFunction*/ false,
                                       /*isSpecialInput*/ false)),
      m_UseMinPrecision(useMinPrecision) {}

DxilSignature::DxilSignature(DXIL::SigPointKind sigPointKind,
                             bool useMinPrecision)
    : m_sigPointKind(sigPointKind),
      m_UseMinPrecision(useMinPrecision) {}

DxilSignature::DxilSignature(const DxilSignature &src)
    : m_sigPointKind(src.m_sigPointKind),
      m_UseMinPrecision(src.m_UseMinPrecision) {
  const bool bSetID = false;
  for (auto &Elt : src.GetElements()) {
    std::unique_ptr<DxilSignatureElement> newElt = CreateElement();
    newElt->Initialize(Elt->GetName(), Elt->GetCompType(),
                       Elt->GetInterpolationMode()->GetKind(), Elt->GetRows(),
                       Elt->GetCols(), Elt->GetStartRow(), Elt->GetStartCol(),
                       Elt->GetID(), Elt->GetSemanticIndexVec());
    AppendElement(std::move(newElt), bSetID);
  }
}

DxilSignature::~DxilSignature() {
}

bool DxilSignature::IsInput() const {
  return SigPoint::GetSigPoint(m_sigPointKind)->IsInput();
}

bool DxilSignature::IsOutput() const {
  return SigPoint::GetSigPoint(m_sigPointKind)->IsOutput();
}

unique_ptr<DxilSignatureElement> DxilSignature::CreateElement() {
  return std::make_unique<DxilSignatureElement>(m_sigPointKind);
}

unsigned DxilSignature::AppendElement(std::unique_ptr<DxilSignatureElement> pSE, bool bSetID) {
  DXASSERT_NOMSG((unsigned)m_Elements.size() < UINT_MAX);
  unsigned Id = (unsigned)m_Elements.size();
  if (bSetID) {
    pSE->SetID(Id);
  }
  m_Elements.emplace_back(std::move(pSE));
  return Id;
}

DxilSignatureElement &DxilSignature::GetElement(unsigned idx) {
  return *m_Elements[idx];
}

const DxilSignatureElement &DxilSignature::GetElement(unsigned idx) const {
  return *m_Elements[idx];
}

const std::vector<std::unique_ptr<DxilSignatureElement> > &DxilSignature::GetElements() const {
  return m_Elements;
}

bool DxilSignature::ShouldBeAllocated(DXIL::SemanticInterpretationKind Kind) {
  switch (Kind) {
  case DXIL::SemanticInterpretationKind::NA:
  case DXIL::SemanticInterpretationKind::NotInSig:
  case DXIL::SemanticInterpretationKind::NotPacked:
  case DXIL::SemanticInterpretationKind::Shadow:
    return false;
  default:
    break;
  }
  return true;
}

bool DxilSignature::IsFullyAllocated() const {
  for (auto &SE : m_Elements) {
    if (!ShouldBeAllocated(SE.get()->GetInterpretation()))
      continue;
    if (!SE->IsAllocated())
      return false;
  }
  return true;
}

unsigned DxilSignature::NumVectorsUsed(unsigned streamIndex) const {
  unsigned NumVectors = 0;
  for (auto &SE : m_Elements) {
    if (SE->IsAllocated() && SE->GetOutputStream() == streamIndex)
      NumVectors = std::max(NumVectors, (unsigned)SE->GetStartRow() + SE->GetRows());
  }
  return NumVectors;
}

unsigned DxilSignature::GetRowCount() const {
  unsigned maxRow = 0;
  for (auto &E : GetElements()) {
    unsigned endRow = E->GetStartRow() + E->GetRows();
    if (maxRow < endRow) {
      maxRow = endRow;
    }
  }
  return maxRow;
}

//------------------------------------------------------------------------------
//
// EntrySingnature methods.
//
DxilEntrySignature::DxilEntrySignature(const DxilEntrySignature &src)
    : InputSignature(src.InputSignature), OutputSignature(src.OutputSignature),
      PatchConstOrPrimSignature(src.PatchConstOrPrimSignature) {}

} // namespace hlsl

#include "dxc/DXIL/DxilSigPoint.inl"
