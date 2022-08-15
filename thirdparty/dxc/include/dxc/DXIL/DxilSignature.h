///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilSignature.h                                                           //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Representation of HLSL shader signature.                                  //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "dxc/DXIL/DxilSignatureElement.h"
#include <memory>
#include <string>
#include <vector>


namespace hlsl {

/// Use this class to represent HLSL signature.
class DxilSignature {
public:
  using Kind = DXIL::SignatureKind;

  DxilSignature(DXIL::ShaderKind shaderKind, DXIL::SignatureKind sigKind, bool useMinPrecision);
  DxilSignature(DXIL::SigPointKind sigPointKind, bool useMinPrecision);
  DxilSignature(const DxilSignature &src);
  virtual ~DxilSignature();

  bool IsInput() const;
  bool IsOutput() const;

  virtual std::unique_ptr<DxilSignatureElement> CreateElement();

  unsigned AppendElement(std::unique_ptr<DxilSignatureElement> pSE, bool bSetID = true);

  DxilSignatureElement &GetElement(unsigned idx);
  const DxilSignatureElement &GetElement(unsigned idx) const;
  const std::vector<std::unique_ptr<DxilSignatureElement> > &GetElements() const;

  // Returns true if all signature elements that should be allocated are allocated
  bool IsFullyAllocated() const;

  // Returns the number of allocated vectors used to contain signature
  unsigned NumVectorsUsed(unsigned streamIndex =  0) const;

  bool UseMinPrecision() const { return m_UseMinPrecision; }

  DXIL::SigPointKind GetSigPointKind() const { return m_sigPointKind; }

  static bool ShouldBeAllocated(DXIL::SemanticInterpretationKind);

  unsigned GetRowCount() const;

private:
  DXIL::SigPointKind m_sigPointKind;
  std::vector<std::unique_ptr<DxilSignatureElement> > m_Elements;
  bool m_UseMinPrecision;
};

struct DxilEntrySignature {
  DxilEntrySignature(DXIL::ShaderKind shaderKind, bool useMinPrecision)
      : InputSignature(shaderKind, DxilSignature::Kind::Input, useMinPrecision),
        OutputSignature(shaderKind, DxilSignature::Kind::Output, useMinPrecision),
        PatchConstOrPrimSignature(shaderKind, DxilSignature::Kind::PatchConstOrPrim, useMinPrecision) {
  }
  DxilEntrySignature(const DxilEntrySignature &src);
  DxilSignature InputSignature;
  DxilSignature OutputSignature;
  DxilSignature PatchConstOrPrimSignature;
};

} // namespace hlsl
