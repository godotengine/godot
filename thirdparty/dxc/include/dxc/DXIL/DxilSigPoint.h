///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilSigPoint.h                                                            //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Representation of HLSL signature points.                                  //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "DxilConstants.h"

namespace hlsl {

struct VersionedSemanticInterpretation {
  VersionedSemanticInterpretation(DXIL::SemanticInterpretationKind k, unsigned MajorVersion=0, unsigned MinorVersion=0) :
    Kind(k), Major((unsigned short)MajorVersion), Minor((unsigned short)MinorVersion)
  {}
  DXIL::SemanticInterpretationKind Kind;
  unsigned short Major, Minor;
};

/// Use this class to describe an HLSL signature point.
/// A signature point is a set of signature parameters at a particular shader stage,
/// grouped by input/output/patch constant and value frequency.
class SigPoint {
public:
  using Kind = DXIL::SigPointKind;

  SigPoint(DXIL::SigPointKind spk, const char *name, DXIL::SigPointKind rspk, DXIL::ShaderKind shk, DXIL::SignatureKind sigk, DXIL::PackingKind pk);

  bool IsInput() const { return m_SignatureKind == DXIL::SignatureKind::Input; }
  bool IsOutput() const { return m_SignatureKind == DXIL::SignatureKind::Output; }
  bool IsPatchConstOrPrim() const { return m_SignatureKind == DXIL::SignatureKind::PatchConstOrPrim; }

  Kind GetKind() const { return m_Kind; }
  const char *GetName() const { return m_pszName; }
  DXIL::ShaderKind GetShaderKind() const { return m_ShaderKind; }
  Kind GetRelatedKind() const { return m_RelatedKind; }
  DXIL::SignatureKind GetSignatureKind() const { return m_SignatureKind; }
  DXIL::SignatureKind GetSignatureKindWithFallback() const;
  DXIL::PackingKind GetPackingKind() const { return m_PackingKind; }
  bool NeedsInterpMode() const { return m_PackingKind == DXIL::PackingKind::Vertex; }

  static const SigPoint* GetSigPoint(Kind K);

  // isSpecialInput selects a signature point outside the normal input/output/patch constant signatures.
  // These are used for a few system values that should not be included as part of the regular input
  // structure because they do not have the same dimensionality as other inputs, such as
  // SV_PrimitiveID for Geometry, Hull, and Patch Constant Functions.
  static DXIL::SigPointKind GetKind(DXIL::ShaderKind shaderKind, DXIL::SignatureKind sigKind, bool isPatchConstantFunction, bool isSpecialInput);

  // Interpretations are how system values are intrepeted at a particular signature point.
  static DXIL::SemanticInterpretationKind GetInterpretation(DXIL::SemanticKind SK, Kind K, unsigned MajorVersion, unsigned MinorVersion);

  // For Shadow elements, recover original SigPointKind
  static Kind RecoverKind(DXIL::SemanticKind SK, Kind K);

private:
  static const unsigned kNumSigPointRecords = (unsigned)Kind::Invalid + 1;
  static const SigPoint ms_SigPoints[kNumSigPointRecords];
  static const VersionedSemanticInterpretation ms_SemanticInterpretationTable[(unsigned)DXIL::SemanticKind::Invalid][(unsigned)Kind::Invalid];

  Kind m_Kind;
  Kind m_RelatedKind;
  DXIL::ShaderKind m_ShaderKind;
  DXIL::SignatureKind m_SignatureKind;
  const char *m_pszName;
  DXIL::PackingKind m_PackingKind;
};

} // namespace hlsl
