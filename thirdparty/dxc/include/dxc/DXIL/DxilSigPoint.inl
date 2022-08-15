///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilSigPoint.inl                                                          //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

/* <py>
import hctdb_instrhelp
</py> */

namespace hlsl {

// Related points to a SigPoint that would contain the signature element for a "Shadow" element.
// A "Shadow" element isn't actually accessed through that signature's Load/Store Input/Output.
// Instead, it uses a dedicated intrinsic, but still requires that an entry exist in the signature
// for compatibility purposes.
// <py::lines('SIGPOINT-TABLE')>hctdb_instrhelp.get_sigpoint_table()</py>
// SIGPOINT-TABLE:BEGIN
//   SigPoint, Related, ShaderKind,    PackingKind,    SignatureKind
#define DO_SIGPOINTS(ROW) \
  ROW(VSIn,     Invalid, Vertex,        InputAssembler, Input) \
  ROW(VSOut,    Invalid, Vertex,        Vertex,         Output) \
  ROW(PCIn,     HSCPIn,  Hull,          None,           Invalid) \
  ROW(HSIn,     HSCPIn,  Hull,          None,           Invalid) \
  ROW(HSCPIn,   Invalid, Hull,          Vertex,         Input) \
  ROW(HSCPOut,  Invalid, Hull,          Vertex,         Output) \
  ROW(PCOut,    Invalid, Hull,          PatchConstant,  PatchConstOrPrim) \
  ROW(DSIn,     Invalid, Domain,        PatchConstant,  PatchConstOrPrim) \
  ROW(DSCPIn,   Invalid, Domain,        Vertex,         Input) \
  ROW(DSOut,    Invalid, Domain,        Vertex,         Output) \
  ROW(GSVIn,    Invalid, Geometry,      Vertex,         Input) \
  ROW(GSIn,     GSVIn,   Geometry,      None,           Invalid) \
  ROW(GSOut,    Invalid, Geometry,      Vertex,         Output) \
  ROW(PSIn,     Invalid, Pixel,         Vertex,         Input) \
  ROW(PSOut,    Invalid, Pixel,         Target,         Output) \
  ROW(CSIn,     Invalid, Compute,       None,           Invalid) \
  ROW(MSIn,     Invalid, Mesh,          None,           Invalid) \
  ROW(MSOut,    Invalid, Mesh,          Vertex,         Output) \
  ROW(MSPOut,   Invalid, Mesh,          Vertex,         PatchConstOrPrim) \
  ROW(ASIn,     Invalid, Amplification, None,           Invalid) \
  ROW(Invalid,  Invalid, Invalid,       Invalid,        Invalid)
// SIGPOINT-TABLE:END

const SigPoint SigPoint::ms_SigPoints[kNumSigPointRecords] = {
#define DEF_SIGPOINT(spk, rspk, shk, pk, sigk) \
  SigPoint(DXIL::SigPointKind::spk, #spk, DXIL::SigPointKind::rspk, DXIL::ShaderKind::shk, DXIL::SignatureKind::sigk, DXIL::PackingKind::pk),
  DO_SIGPOINTS(DEF_SIGPOINT)
#undef DEF_SIGPOINT
};

// <py::lines('INTERPRETATION-TABLE')>hctdb_instrhelp.get_interpretation_table()</py>
// INTERPRETATION-TABLE:BEGIN
//   Semantic,               VSIn,         VSOut,    PCIn,         HSIn,         HSCPIn,   HSCPOut,  PCOut,      DSIn,         DSCPIn,   DSOut,    GSVIn,    GSIn,         GSOut,    PSIn,          PSOut,         CSIn,     MSIn,     MSOut,    MSPOut,    ASIn
#define DO_INTERPRETATION_TABLE(ROW) \
  ROW(Arbitrary,              Arb,          Arb,      NA,           NA,           Arb,      Arb,      Arb,        Arb,          Arb,      Arb,      Arb,      NA,           Arb,      Arb,           NA,            NA,       NA,       Arb,      Arb,       NA) \
  ROW(VertexID,               SV,           NA,       NA,           NA,           NA,       NA,       NA,         NA,           NA,       NA,       NA,       NA,           NA,       NA,            NA,            NA,       NA,       NA,       NA,        NA) \
  ROW(InstanceID,             SV,           Arb,      NA,           NA,           Arb,      Arb,      NA,         NA,           Arb,      Arb,      Arb,      NA,           Arb,      Arb,           NA,            NA,       NA,       NA,       NA,        NA) \
  ROW(Position,               Arb,          SV,       NA,           NA,           SV,       SV,       Arb,        Arb,          SV,       SV,       SV,       NA,           SV,       SV,            NA,            NA,       NA,       SV,       NA,        NA) \
  ROW(RenderTargetArrayIndex, Arb,          SV,       NA,           NA,           SV,       SV,       Arb,        Arb,          SV,       SV,       SV,       NA,           SV,       SV,            NA,            NA,       NA,       NA,       SV,        NA) \
  ROW(ViewPortArrayIndex,     Arb,          SV,       NA,           NA,           SV,       SV,       Arb,        Arb,          SV,       SV,       SV,       NA,           SV,       SV,            NA,            NA,       NA,       NA,       SV,        NA) \
  ROW(ClipDistance,           Arb,          ClipCull, NA,           NA,           ClipCull, ClipCull, Arb,        Arb,          ClipCull, ClipCull, ClipCull, NA,           ClipCull, ClipCull,      NA,            NA,       NA,       ClipCull, NA,        NA) \
  ROW(CullDistance,           Arb,          ClipCull, NA,           NA,           ClipCull, ClipCull, Arb,        Arb,          ClipCull, ClipCull, ClipCull, NA,           ClipCull, ClipCull,      NA,            NA,       NA,       ClipCull, NA,        NA) \
  ROW(OutputControlPointID,   NA,           NA,       NA,           NotInSig,     NA,       NA,       NA,         NA,           NA,       NA,       NA,       NA,           NA,       NA,            NA,            NA,       NA,       NA,       NA,        NA) \
  ROW(DomainLocation,         NA,           NA,       NA,           NA,           NA,       NA,       NA,         NotInSig,     NA,       NA,       NA,       NA,           NA,       NA,            NA,            NA,       NA,       NA,       NA,        NA) \
  ROW(PrimitiveID,            NA,           NA,       NotInSig,     NotInSig,     NA,       NA,       NA,         NotInSig,     NA,       NA,       NA,       Shadow,       SGV,      SGV,           NA,            NA,       NA,       NA,       SV,        NA) \
  ROW(GSInstanceID,           NA,           NA,       NA,           NA,           NA,       NA,       NA,         NA,           NA,       NA,       NA,       NotInSig,     NA,       NA,            NA,            NA,       NA,       NA,       NA,        NA) \
  ROW(SampleIndex,            NA,           NA,       NA,           NA,           NA,       NA,       NA,         NA,           NA,       NA,       NA,       NA,           NA,       Shadow _41,    NA,            NA,       NA,       NA,       NA,        NA) \
  ROW(IsFrontFace,            NA,           NA,       NA,           NA,           NA,       NA,       NA,         NA,           NA,       NA,       NA,       NA,           SGV,      SGV,           NA,            NA,       NA,       NA,       NA,        NA) \
  ROW(Coverage,               NA,           NA,       NA,           NA,           NA,       NA,       NA,         NA,           NA,       NA,       NA,       NA,           NA,       NotInSig _50,  NotPacked _41, NA,       NA,       NA,       NA,        NA) \
  ROW(InnerCoverage,          NA,           NA,       NA,           NA,           NA,       NA,       NA,         NA,           NA,       NA,       NA,       NA,           NA,       NotInSig _50,  NA,            NA,       NA,       NA,       NA,        NA) \
  ROW(Target,                 NA,           NA,       NA,           NA,           NA,       NA,       NA,         NA,           NA,       NA,       NA,       NA,           NA,       NA,            Target,        NA,       NA,       NA,       NA,        NA) \
  ROW(Depth,                  NA,           NA,       NA,           NA,           NA,       NA,       NA,         NA,           NA,       NA,       NA,       NA,           NA,       NA,            NotPacked,     NA,       NA,       NA,       NA,        NA) \
  ROW(DepthLessEqual,         NA,           NA,       NA,           NA,           NA,       NA,       NA,         NA,           NA,       NA,       NA,       NA,           NA,       NA,            NotPacked _50, NA,       NA,       NA,       NA,        NA) \
  ROW(DepthGreaterEqual,      NA,           NA,       NA,           NA,           NA,       NA,       NA,         NA,           NA,       NA,       NA,       NA,           NA,       NA,            NotPacked _50, NA,       NA,       NA,       NA,        NA) \
  ROW(StencilRef,             NA,           NA,       NA,           NA,           NA,       NA,       NA,         NA,           NA,       NA,       NA,       NA,           NA,       NA,            NotPacked _50, NA,       NA,       NA,       NA,        NA) \
  ROW(DispatchThreadID,       NA,           NA,       NA,           NA,           NA,       NA,       NA,         NA,           NA,       NA,       NA,       NA,           NA,       NA,            NA,            NotInSig, NotInSig, NA,       NA,        NotInSig) \
  ROW(GroupID,                NA,           NA,       NA,           NA,           NA,       NA,       NA,         NA,           NA,       NA,       NA,       NA,           NA,       NA,            NA,            NotInSig, NotInSig, NA,       NA,        NotInSig) \
  ROW(GroupIndex,             NA,           NA,       NA,           NA,           NA,       NA,       NA,         NA,           NA,       NA,       NA,       NA,           NA,       NA,            NA,            NotInSig, NotInSig, NA,       NA,        NotInSig) \
  ROW(GroupThreadID,          NA,           NA,       NA,           NA,           NA,       NA,       NA,         NA,           NA,       NA,       NA,       NA,           NA,       NA,            NA,            NotInSig, NotInSig, NA,       NA,        NotInSig) \
  ROW(TessFactor,             NA,           NA,       NA,           NA,           NA,       NA,       TessFactor, TessFactor,   NA,       NA,       NA,       NA,           NA,       NA,            NA,            NA,       NA,       NA,       NA,        NA) \
  ROW(InsideTessFactor,       NA,           NA,       NA,           NA,           NA,       NA,       TessFactor, TessFactor,   NA,       NA,       NA,       NA,           NA,       NA,            NA,            NA,       NA,       NA,       NA,        NA) \
  ROW(ViewID,                 NotInSig _61, NA,       NotInSig _61, NotInSig _61, NA,       NA,       NA,         NotInSig _61, NA,       NA,       NA,       NotInSig _61, NA,       NotInSig _61,  NA,            NA,       NotInSig, NA,       NA,        NA) \
  ROW(Barycentrics,           NA,           NA,       NA,           NA,           NA,       NA,       NA,         NA,           NA,       NA,       NA,       NA,           NA,       NotPacked _61, NA,            NA,       NA,       NA,       NA,        NA) \
  ROW(ShadingRate,            NA,           SV _64,   NA,           NA,           SV _64,   SV _64,   NA,         NA,           SV _64,   SV _64,   SV _64,   NA,           SV _64,   SV _64,        NA,            NA,       NA,       NA,       SV,        NA) \
  ROW(CullPrimitive,          NA,           NA,       NA,           NA,           NA,       NA,       NA,         NA,           NA,       NA,       NA,       NA,           NA,       NotInSig,      NA,            NA,       NA,       NA,       NotPacked, NA)
// INTERPRETATION-TABLE:END

const VersionedSemanticInterpretation SigPoint::ms_SemanticInterpretationTable[(unsigned)DXIL::SemanticKind::Invalid][(unsigned)SigPoint::Kind::Invalid] = {
#define _41 ,4,1
#define _50 ,5,0
#define _61 ,6,1
#define _64 ,6,4
#define _65 ,6,5
#define DO_ROW(SEM, VSIn, VSOut, PCIn, HSIn, HSCPIn, HSCPOut, PCOut, DSIn, DSCPIn, DSOut, GSVIn, GSIn, GSOut, PSIn, PSOut, CSIn, MSIn, MSOut, MSPOut, ASIn) \
  { VersionedSemanticInterpretation(DXIL::SemanticInterpretationKind::VSIn), \
    VersionedSemanticInterpretation(DXIL::SemanticInterpretationKind::VSOut), \
    VersionedSemanticInterpretation(DXIL::SemanticInterpretationKind::PCIn), \
    VersionedSemanticInterpretation(DXIL::SemanticInterpretationKind::HSIn), \
    VersionedSemanticInterpretation(DXIL::SemanticInterpretationKind::HSCPIn), \
    VersionedSemanticInterpretation(DXIL::SemanticInterpretationKind::HSCPOut), \
    VersionedSemanticInterpretation(DXIL::SemanticInterpretationKind::PCOut), \
    VersionedSemanticInterpretation(DXIL::SemanticInterpretationKind::DSIn), \
    VersionedSemanticInterpretation(DXIL::SemanticInterpretationKind::DSCPIn), \
    VersionedSemanticInterpretation(DXIL::SemanticInterpretationKind::DSOut), \
    VersionedSemanticInterpretation(DXIL::SemanticInterpretationKind::GSVIn), \
    VersionedSemanticInterpretation(DXIL::SemanticInterpretationKind::GSIn), \
    VersionedSemanticInterpretation(DXIL::SemanticInterpretationKind::GSOut), \
    VersionedSemanticInterpretation(DXIL::SemanticInterpretationKind::PSIn), \
    VersionedSemanticInterpretation(DXIL::SemanticInterpretationKind::PSOut), \
    VersionedSemanticInterpretation(DXIL::SemanticInterpretationKind::CSIn), \
    VersionedSemanticInterpretation(DXIL::SemanticInterpretationKind::MSIn), \
    VersionedSemanticInterpretation(DXIL::SemanticInterpretationKind::MSOut), \
    VersionedSemanticInterpretation(DXIL::SemanticInterpretationKind::MSPOut), \
    VersionedSemanticInterpretation(DXIL::SemanticInterpretationKind::ASIn), \
  },
  DO_INTERPRETATION_TABLE(DO_ROW)
#undef DO_ROW
};

// -----------------------
// SigPoint Implementation

SigPoint::SigPoint(DXIL::SigPointKind spk, const char *name, DXIL::SigPointKind rspk, DXIL::ShaderKind shk, DXIL::SignatureKind sigk, DXIL::PackingKind pk) :
  m_Kind(spk), m_RelatedKind(rspk), m_ShaderKind(shk), m_SignatureKind(sigk), m_pszName(name), m_PackingKind(pk)
{}

DXIL::SignatureKind SigPoint::GetSignatureKindWithFallback() const {
  DXIL::SignatureKind sigKind = GetSignatureKind();
  if (sigKind == DXIL::SignatureKind::Invalid) {
    DXIL::SigPointKind RK = GetRelatedKind();
    if (RK != DXIL::SigPointKind::Invalid)
      sigKind = SigPoint::GetSigPoint(RK)->GetSignatureKind();
  }
  return sigKind;
}

DXIL::SemanticInterpretationKind SigPoint::GetInterpretation(DXIL::SemanticKind SK, Kind K, unsigned MajorVersion, unsigned MinorVersion) {
  if (SK < DXIL::SemanticKind::Invalid && K < Kind::Invalid) {
    const VersionedSemanticInterpretation& VSI = ms_SemanticInterpretationTable[(unsigned)SK][(unsigned)K];
    if (VSI.Kind != DXIL::SemanticInterpretationKind::NA) {
      if (MajorVersion > (unsigned)VSI.Major ||
          (MajorVersion == (unsigned)VSI.Major && MinorVersion >= (unsigned)VSI.Minor))
        return VSI.Kind;
    }
  }
  return DXIL::SemanticInterpretationKind::NA;
}

SigPoint::Kind SigPoint::RecoverKind(DXIL::SemanticKind SK, Kind K) {
  if (SK == DXIL::SemanticKind::PrimitiveID && K == Kind::GSVIn)
    return Kind::GSIn;
  return K;
}

// --------------
// Static methods

const SigPoint* SigPoint::GetSigPoint(Kind K) {
  return ((unsigned)K < kNumSigPointRecords) ? &ms_SigPoints[(unsigned)K] : &ms_SigPoints[(unsigned)Kind::Invalid];
}

DXIL::SigPointKind SigPoint::GetKind(DXIL::ShaderKind shaderKind, DXIL::SignatureKind sigKind, bool isPatchConstantFunction, bool isSpecialInput) {
  if (isSpecialInput) {
    switch (shaderKind) {
    case DXIL::ShaderKind::Hull:
    if (sigKind == DXIL::SignatureKind::Input)
      return isPatchConstantFunction ? DXIL::SigPointKind::PCIn : DXIL::SigPointKind::HSIn;
    case DXIL::ShaderKind::Geometry:
      if (sigKind == DXIL::SignatureKind::Input)
        return DXIL::SigPointKind::GSIn;
    default:
      break;
    }
  }

  switch (shaderKind) {
  case DXIL::ShaderKind::Vertex:
    switch (sigKind) {
    case DXIL::SignatureKind::Input: return DXIL::SigPointKind::VSIn;
    case DXIL::SignatureKind::Output: return DXIL::SigPointKind::VSOut;
    default:
      break;
    }
    break;
  case DXIL::ShaderKind::Hull:
    switch (sigKind) {
    case DXIL::SignatureKind::Input: return DXIL::SigPointKind::HSCPIn;
    case DXIL::SignatureKind::Output: return DXIL::SigPointKind::HSCPOut;
    case DXIL::SignatureKind::PatchConstOrPrim: return DXIL::SigPointKind::PCOut;
    default:
      break;
    }
    break;
  case DXIL::ShaderKind::Domain:
    switch (sigKind) {
    case DXIL::SignatureKind::Input: return DXIL::SigPointKind::DSCPIn;
    case DXIL::SignatureKind::Output: return DXIL::SigPointKind::DSOut;
    case DXIL::SignatureKind::PatchConstOrPrim: return DXIL::SigPointKind::DSIn;
    default:
      break;
    }
    break;
  case DXIL::ShaderKind::Geometry:
    switch (sigKind) {
    case DXIL::SignatureKind::Input: return DXIL::SigPointKind::GSVIn;
    case DXIL::SignatureKind::Output: return DXIL::SigPointKind::GSOut;
    default:
      break;
    }
    break;
  case DXIL::ShaderKind::Pixel:
    switch (sigKind) {
    case DXIL::SignatureKind::Input: return DXIL::SigPointKind::PSIn;
    case DXIL::SignatureKind::Output: return DXIL::SigPointKind::PSOut;
    default:
      break;
    }
    break;
  case DXIL::ShaderKind::Compute:
    switch (sigKind) {
    case DXIL::SignatureKind::Input: return DXIL::SigPointKind::CSIn;
    default:
      break;
    }
    break;
  case DXIL::ShaderKind::Mesh:
    switch (sigKind) {
    case DXIL::SignatureKind::Input: return DXIL::SigPointKind::MSIn;
    case DXIL::SignatureKind::Output: return DXIL::SigPointKind::MSOut;
    case DXIL::SignatureKind::PatchConstOrPrim: return DXIL::SigPointKind::MSPOut;
    default:
      break;
    }
    break;
  case DXIL::ShaderKind::Amplification:
    switch (sigKind) {
    case DXIL::SignatureKind::Input: return DXIL::SigPointKind::ASIn;
    default:
      break;
    }
    break;
  default:
    break;
  }

  return DXIL::SigPointKind::Invalid;
}

} // namespace hlsl
