///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilContainerAssembler.cpp                                                //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Provides support for serializing a module into DXIL container structures. //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Support/MD5.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "dxc/DxilContainer/DxilContainer.h"
#include "dxc/DXIL/DxilModule.h"
#include "dxc/DXIL/DxilShaderModel.h"
#include "dxc/DxilRootSignature/DxilRootSignature.h"
#include "dxc/DxilContainer/DxilContainerAssembler.h"
#include "dxc/DXIL/DxilUtil.h"
#include "dxc/DXIL/DxilFunctionProps.h"
#include "dxc/DXIL/DxilEntryProps.h"
#include "dxc/DXIL/DxilOperations.h"
#include "dxc/DXIL/DxilInstructions.h"
#include "dxc/Support/Global.h"
#include "dxc/Support/Unicode.h"
#include "dxc/Support/WinIncludes.h"
#include "dxc/Support/FileIOHelper.h"
#include "dxc/Support/dxcapi.impl.h"
#include <assert.h> // Needed for DxilPipelineStateValidation.h
#include "dxc/DxilContainer/DxilPipelineStateValidation.h"
#include "dxc/DxilContainer/DxilRuntimeReflection.h"
#include "dxc/DXIL/DxilCounters.h"
#include <algorithm>
#include <functional>

using namespace llvm;
using namespace hlsl;
using namespace hlsl::RDAT;

static_assert((unsigned)PSVShaderKind::Invalid == (unsigned)DXIL::ShaderKind::Invalid,
              "otherwise, PSVShaderKind enum out of sync.");

static DxilProgramSigSemantic KindToSystemValue(Semantic::Kind kind, DXIL::TessellatorDomain domain) {
  switch (kind) {
  case Semantic::Kind::Arbitrary: return DxilProgramSigSemantic::Undefined;
  case Semantic::Kind::VertexID: return DxilProgramSigSemantic::VertexID;
  case Semantic::Kind::InstanceID: return DxilProgramSigSemantic::InstanceID;
  case Semantic::Kind::Position: return DxilProgramSigSemantic::Position;
  case Semantic::Kind::Coverage: return DxilProgramSigSemantic::Coverage;
  case Semantic::Kind::InnerCoverage: return DxilProgramSigSemantic::InnerCoverage;
  case Semantic::Kind::PrimitiveID: return DxilProgramSigSemantic::PrimitiveID;
  case Semantic::Kind::SampleIndex: return DxilProgramSigSemantic::SampleIndex;
  case Semantic::Kind::IsFrontFace: return DxilProgramSigSemantic::IsFrontFace;
  case Semantic::Kind::RenderTargetArrayIndex: return DxilProgramSigSemantic::RenderTargetArrayIndex;
  case Semantic::Kind::ViewPortArrayIndex: return DxilProgramSigSemantic::ViewPortArrayIndex;
  case Semantic::Kind::ClipDistance: return DxilProgramSigSemantic::ClipDistance;
  case Semantic::Kind::CullDistance: return DxilProgramSigSemantic::CullDistance;
  case Semantic::Kind::Barycentrics: return DxilProgramSigSemantic::Barycentrics;
  case Semantic::Kind::ShadingRate: return DxilProgramSigSemantic::ShadingRate;
  case Semantic::Kind::CullPrimitive: return DxilProgramSigSemantic::CullPrimitive;
  case Semantic::Kind::TessFactor: {
    switch (domain) {
    case DXIL::TessellatorDomain::IsoLine:
      // Will bu updated to DetailTessFactor in next row.
      return DxilProgramSigSemantic::FinalLineDensityTessfactor;
    case DXIL::TessellatorDomain::Tri:
      return DxilProgramSigSemantic::FinalTriEdgeTessfactor;
    case DXIL::TessellatorDomain::Quad:
      return DxilProgramSigSemantic::FinalQuadEdgeTessfactor;
    default:
      // No other valid TesselatorDomain options.
      return DxilProgramSigSemantic::Undefined;
    }
  }
  case Semantic::Kind::InsideTessFactor: {
    switch (domain) {
    case DXIL::TessellatorDomain::IsoLine:
      DXASSERT(0, "invalid semantic");
      return DxilProgramSigSemantic::Undefined;
    case DXIL::TessellatorDomain::Tri:
      return DxilProgramSigSemantic::FinalTriInsideTessfactor;
    case DXIL::TessellatorDomain::Quad:
      return DxilProgramSigSemantic::FinalQuadInsideTessfactor;
    default:
      // No other valid DxilProgramSigSemantic options.
      return DxilProgramSigSemantic::Undefined;
    }
  }
  case Semantic::Kind::Invalid:
    return DxilProgramSigSemantic::Undefined;
  case Semantic::Kind::Target: return DxilProgramSigSemantic::Target;
  case Semantic::Kind::Depth: return DxilProgramSigSemantic::Depth;
  case Semantic::Kind::DepthLessEqual: return DxilProgramSigSemantic::DepthLE;
  case Semantic::Kind::DepthGreaterEqual: return DxilProgramSigSemantic::DepthGE;
  case Semantic::Kind::StencilRef:
    __fallthrough;
  default:
    DXASSERT(kind == Semantic::Kind::StencilRef, "else Invalid or switch is missing a case");
    return DxilProgramSigSemantic::StencilRef;
  }
  // TODO: Final_* values need mappings
}

static DxilProgramSigCompType CompTypeToSigCompType(hlsl::CompType value, bool i1ToUnknownCompat) {
  switch (value.GetKind()) {
  case CompType::Kind::I32: return DxilProgramSigCompType::SInt32;

  case CompType::Kind::I1:
    // Validator 1.4 and below returned Unknown for i1
    if (i1ToUnknownCompat)  return DxilProgramSigCompType::Unknown;
    else                    return DxilProgramSigCompType::UInt32;

  case CompType::Kind::U32: return DxilProgramSigCompType::UInt32;
  case CompType::Kind::F32: return DxilProgramSigCompType::Float32;
  case CompType::Kind::I16: return DxilProgramSigCompType::SInt16;
  case CompType::Kind::I64: return DxilProgramSigCompType::SInt64;
  case CompType::Kind::U16: return DxilProgramSigCompType::UInt16;
  case CompType::Kind::U64: return DxilProgramSigCompType::UInt64;
  case CompType::Kind::F16: return DxilProgramSigCompType::Float16;
  case CompType::Kind::F64: return DxilProgramSigCompType::Float64;
  case CompType::Kind::Invalid: __fallthrough;
  default:
    return DxilProgramSigCompType::Unknown;
  }
}

static DxilProgramSigMinPrecision CompTypeToSigMinPrecision(hlsl::CompType value) {
  switch (value.GetKind()) {
  case CompType::Kind::I32: return DxilProgramSigMinPrecision::Default;
  case CompType::Kind::U32: return DxilProgramSigMinPrecision::Default;
  case CompType::Kind::F32: return DxilProgramSigMinPrecision::Default;
  case CompType::Kind::I1: return DxilProgramSigMinPrecision::Default;
  case CompType::Kind::U64: __fallthrough;
  case CompType::Kind::I64: __fallthrough;
  case CompType::Kind::F64: return DxilProgramSigMinPrecision::Default;
  case CompType::Kind::I16: return DxilProgramSigMinPrecision::SInt16;
  case CompType::Kind::U16: return DxilProgramSigMinPrecision::UInt16;
  case CompType::Kind::F16: return DxilProgramSigMinPrecision::Float16; // Float2_8 is not supported in DXIL.
  case CompType::Kind::Invalid: __fallthrough;
  default:
    return DxilProgramSigMinPrecision::Default;
  }
}

template <typename T>
struct sort_second {
  bool operator()(const T &a, const T &b) {
    return std::less<decltype(a.second)>()(a.second, b.second);
  }
};

struct sort_sig {
  bool operator()(const DxilProgramSignatureElement &a,
                  const DxilProgramSignatureElement &b) {
    return (a.Stream < b.Stream) ||
           ((a.Stream == b.Stream) && (a.Register < b.Register)) ||
           ((a.Stream == b.Stream) && (a.Register == b.Register) &&
            (a.SemanticName < b.SemanticName));
  }
};

static uint8_t NegMask(uint8_t V) {
  V ^= 0xF;
  return V & 0xF;
}

class DxilProgramSignatureWriter : public DxilPartWriter {
private:
  const DxilSignature &m_signature;
  DXIL::TessellatorDomain m_domain;
  bool   m_isInput;
  bool   m_useMinPrecision;
  bool m_bCompat_1_4;
  bool m_bCompat_1_6; // unaligned size, no dedup for < 1.7
  size_t m_fixedSize;
  typedef std::pair<const char *, uint32_t> NameOffsetPair_nodedup;
  typedef llvm::SmallMapVector<const char *, uint32_t, 8> NameOffsetMap_nodedup;
  typedef std::pair<llvm::StringRef, uint32_t> NameOffsetPair;
  typedef llvm::SmallMapVector<llvm::StringRef, uint32_t, 8> NameOffsetMap;
  uint32_t m_lastOffset;
  NameOffsetMap_nodedup m_semanticNameOffsets_nodedup;
  NameOffsetMap m_semanticNameOffsets;
  unsigned m_paramCount;

  const char *GetSemanticName(const hlsl::DxilSignatureElement *pElement) {
    DXASSERT_NOMSG(pElement != nullptr);
    DXASSERT(pElement->GetName() != nullptr, "else sig is malformed");
    return pElement->GetName();
  }

  uint32_t GetSemanticOffset_nodedup(const hlsl::DxilSignatureElement *pElement) {
    const char *pName = GetSemanticName(pElement);
    NameOffsetMap_nodedup::iterator nameOffset = m_semanticNameOffsets_nodedup.find(pName);
    uint32_t result;
    if (nameOffset == m_semanticNameOffsets_nodedup.end()) {
      result = m_lastOffset;
      m_semanticNameOffsets_nodedup.insert(NameOffsetPair_nodedup(pName, result));
      m_lastOffset += strlen(pName) + 1;
    }
    else {
      result = nameOffset->second;
    }
    return result;
  }
  uint32_t GetSemanticOffset(const hlsl::DxilSignatureElement *pElement) {
    if (m_bCompat_1_6)
      return GetSemanticOffset_nodedup(pElement);

    StringRef name = GetSemanticName(pElement);
    NameOffsetMap::iterator nameOffset = m_semanticNameOffsets.find(name);
    uint32_t result;
    if (nameOffset == m_semanticNameOffsets.end()) {
      result = m_lastOffset;
      m_semanticNameOffsets.insert(NameOffsetPair(name, result));
      m_lastOffset += name.size() + 1;
    }
    else {
      result = nameOffset->second;
    }
    return result;
  }

  void write(std::vector<DxilProgramSignatureElement> &orderedSig,
             const hlsl::DxilSignatureElement *pElement) {
    const std::vector<unsigned> &indexVec = pElement->GetSemanticIndexVec();
    unsigned eltCount = pElement->GetSemanticIndexVec().size();
    unsigned eltRows = 1;
    if (eltCount)
      eltRows = pElement->GetRows() / eltCount;
    DXASSERT_NOMSG(eltRows == 1);

    DxilProgramSignatureElement sig;
    memset(&sig, 0, sizeof(DxilProgramSignatureElement));
    sig.Stream = pElement->GetOutputStream();
    sig.SemanticName = GetSemanticOffset(pElement);
    sig.SystemValue = KindToSystemValue(pElement->GetKind(), m_domain);
    sig.CompType = CompTypeToSigCompType(pElement->GetCompType(), m_bCompat_1_4);
    sig.Register = pElement->GetStartRow();

    sig.Mask = pElement->GetColsAsMask();
    if (m_bCompat_1_4) {
      // Match what validator 1.4 and below expects
      // Only mark exist channel write for output.
      // All channel not used for input.
      if (!m_isInput)
        sig.NeverWrites_Mask = ~sig.Mask;
      else
        sig.AlwaysReads_Mask = 0;
    } else {
      unsigned UsageMask = pElement->GetUsageMask();
      if (pElement->IsAllocated())
        UsageMask <<= pElement->GetStartCol();
      if (!m_isInput)
        sig.NeverWrites_Mask = NegMask(UsageMask);
      else
        sig.AlwaysReads_Mask = UsageMask;
    }

    sig.MinPrecision = m_useMinPrecision
                           ? CompTypeToSigMinPrecision(pElement->GetCompType())
                           : DxilProgramSigMinPrecision::Default;

    for (unsigned i = 0; i < eltCount; ++i) {
      sig.SemanticIndex = indexVec[i];
      orderedSig.emplace_back(sig);
      if (pElement->IsAllocated())
        sig.Register += eltRows;
      if (sig.SystemValue == DxilProgramSigSemantic::FinalLineDensityTessfactor)
        sig.SystemValue = DxilProgramSigSemantic::FinalLineDetailTessfactor;
    }
  }

  void calcSizes() {
    // Calculate size for signature elements.
    const std::vector<std::unique_ptr<hlsl::DxilSignatureElement>> &elements = m_signature.GetElements();
    uint32_t result = sizeof(DxilProgramSignature);
    m_paramCount = 0;
    for (size_t i = 0; i < elements.size(); ++i) {
      DXIL::SemanticInterpretationKind I = elements[i]->GetInterpretation();
      if (I == DXIL::SemanticInterpretationKind::NA || I == DXIL::SemanticInterpretationKind::NotInSig)
        continue;
      unsigned semanticCount = elements[i]->GetSemanticIndexVec().size();
      result += semanticCount * sizeof(DxilProgramSignatureElement);
      m_paramCount += semanticCount;
    }
    m_fixedSize = result;
    m_lastOffset = m_fixedSize;

    // Calculate size for semantic strings.
    for (size_t i = 0; i < elements.size(); ++i) {
      GetSemanticOffset(elements[i].get());
    }
  }

public:
  DxilProgramSignatureWriter(const DxilSignature &signature,
                             DXIL::TessellatorDomain domain,
                             bool isInput, bool UseMinPrecision,
                             bool bCompat_1_4,
                             bool bCompat_1_6)
      : m_signature(signature), m_domain(domain),
        m_isInput(isInput), m_useMinPrecision(UseMinPrecision),
        m_bCompat_1_4(bCompat_1_4),
        m_bCompat_1_6(bCompat_1_6) {
    calcSizes();
  }

  uint32_t size() const override {
    if (m_bCompat_1_6)
      return m_lastOffset;
    else
      return PSVALIGN4(m_lastOffset);
  }

  void write(AbstractMemoryStream *pStream) override {
    UINT64 startPos = pStream->GetPosition();
    const std::vector<std::unique_ptr<hlsl::DxilSignatureElement>> &elements = m_signature.GetElements();

    DxilProgramSignature programSig;
    programSig.ParamCount = m_paramCount;
    programSig.ParamOffset = sizeof(DxilProgramSignature);
    IFT(WriteStreamValue(pStream, programSig));

    // Write structures in register order.
    std::vector<DxilProgramSignatureElement> orderedSig;
    for (size_t i = 0; i < elements.size(); ++i) {
      DXIL::SemanticInterpretationKind I = elements[i]->GetInterpretation();
      if (I == DXIL::SemanticInterpretationKind::NA || I == DXIL::SemanticInterpretationKind::NotInSig)
        continue;
      write(orderedSig, elements[i].get());
    }
    std::sort(orderedSig.begin(), orderedSig.end(), sort_sig());
    for (size_t i = 0; i < orderedSig.size(); ++i) {
      DxilProgramSignatureElement &sigElt = orderedSig[i];
      IFT(WriteStreamValue(pStream, sigElt));
    }

    // Write strings in the offset order.
    std::vector<NameOffsetPair> ordered;
    if (m_bCompat_1_6) {
      ordered.assign(m_semanticNameOffsets_nodedup.begin(), m_semanticNameOffsets_nodedup.end());
    } else {
      ordered.assign(m_semanticNameOffsets.begin(), m_semanticNameOffsets.end());
    }
    std::sort(ordered.begin(), ordered.end(), sort_second<NameOffsetPair>());
    for (size_t i = 0; i < ordered.size(); ++i) {
      StringRef name = ordered[i].first;
      ULONG cbWritten;
      UINT64 offsetPos = pStream->GetPosition();
      DXASSERT_LOCALVAR(offsetPos, offsetPos - startPos == ordered[i].second, "else str offset is incorrect");
      IFT(pStream->Write(name.data(), name.size() + 1, &cbWritten));
    }

    // Align, and verify we wrote the same number of bytes we though we would.
    UINT64 bytesWritten = pStream->GetPosition() - startPos;
    if (!m_bCompat_1_6 && (bytesWritten % 4 != 0)) {
      unsigned paddingToAdd = 4 - (bytesWritten % 4);
      char padding[4] = {0};
      ULONG cbWritten = 0;
      IFT(pStream->Write(padding, paddingToAdd, &cbWritten));
      bytesWritten += cbWritten;
    }
    DXASSERT(bytesWritten == size(), "else size is incorrect");
  }
};

DxilPartWriter *hlsl::NewProgramSignatureWriter(const DxilModule &M, DXIL::SignatureKind Kind) {
  DXIL::TessellatorDomain domain = DXIL::TessellatorDomain::Undefined;
  if (M.GetShaderModel()->IsHS() || M.GetShaderModel()->IsDS())
    domain = M.GetTessellatorDomain();
  unsigned ValMajor, ValMinor;
  M.GetValidatorVersion(ValMajor, ValMinor);
  bool bCompat_1_4 = DXIL::CompareVersions(ValMajor, ValMinor, 1, 5) < 0;
  bool bCompat_1_6 = DXIL::CompareVersions(ValMajor, ValMinor, 1, 7) < 0;
  switch (Kind) {
  case DXIL::SignatureKind::Input:
    return new DxilProgramSignatureWriter(
        M.GetInputSignature(), domain, true,
        M.GetUseMinPrecision(),
        bCompat_1_4, bCompat_1_6);
  case DXIL::SignatureKind::Output:
    return new DxilProgramSignatureWriter(
        M.GetOutputSignature(), domain, false,
        M.GetUseMinPrecision(),
        bCompat_1_4, bCompat_1_6);
  case DXIL::SignatureKind::PatchConstOrPrim:
    return new DxilProgramSignatureWriter(
        M.GetPatchConstOrPrimSignature(), domain,
        /*IsInput*/ M.GetShaderModel()->IsDS(),
        /*UseMinPrecision*/M.GetUseMinPrecision(),
        bCompat_1_4, bCompat_1_6);
  case DXIL::SignatureKind::Invalid:
    return nullptr;
  }
  return nullptr;
}

class DxilProgramRootSignatureWriter : public DxilPartWriter {
private:
  const RootSignatureHandle &m_Sig;
public:
  DxilProgramRootSignatureWriter(const RootSignatureHandle &S) : m_Sig(S) {}
  uint32_t size() const {
    return m_Sig.GetSerializedSize();
  }
  void write(AbstractMemoryStream *pStream) {
    ULONG cbWritten;
    IFT(pStream->Write(m_Sig.GetSerializedBytes(), size(), &cbWritten));
  }
};

DxilPartWriter *hlsl::NewRootSignatureWriter(const RootSignatureHandle &S) {
  return new DxilProgramRootSignatureWriter(S);
}

class DxilFeatureInfoWriter : public DxilPartWriter  {
private:
  // Only save the shader properties after create class for it.
  DxilShaderFeatureInfo featureInfo;
public:
  DxilFeatureInfoWriter(const DxilModule &M) {
    featureInfo.FeatureFlags = M.m_ShaderFlags.GetFeatureInfo();
  }
  uint32_t size() const override {
    return sizeof(DxilShaderFeatureInfo);
  }
  void write(AbstractMemoryStream *pStream) override {
    IFT(WriteStreamValue(pStream, featureInfo.FeatureFlags));
  }
};

DxilPartWriter *hlsl::NewFeatureInfoWriter(const DxilModule &M) {
  return new DxilFeatureInfoWriter(M);
}

class DxilPSVWriter : public DxilPartWriter  {
private:
  const DxilModule &m_Module;
  unsigned m_ValMajor, m_ValMinor;
  PSVInitInfo m_PSVInitInfo;
  DxilPipelineStateValidation m_PSV;
  uint32_t m_PSVBufferSize;
  SmallVector<char, 512> m_PSVBuffer;
  SmallVector<char, 256> m_StringBuffer;
  SmallVector<uint32_t, 8> m_SemanticIndexBuffer;
  std::vector<PSVSignatureElement0> m_SigInputElements;
  std::vector<PSVSignatureElement0> m_SigOutputElements;
  std::vector<PSVSignatureElement0> m_SigPatchConstOrPrimElements;

  void SetPSVSigElement(PSVSignatureElement0 &E, const DxilSignatureElement &SE) {
    memset(&E, 0, sizeof(PSVSignatureElement0));
    if (SE.GetKind() == DXIL::SemanticKind::Arbitrary && strlen(SE.GetName()) > 0) {
      E.SemanticName = (uint32_t)m_StringBuffer.size();
      StringRef Name(SE.GetName());
      m_StringBuffer.append(Name.size()+1, '\0');
      memcpy(m_StringBuffer.data() + E.SemanticName, Name.data(), Name.size());
    } else {
      // m_StringBuffer always starts with '\0' so offset 0 is empty string:
      E.SemanticName = 0;
    }
    // Search index buffer for matching semantic index sequence
    DXASSERT_NOMSG(SE.GetRows() == SE.GetSemanticIndexVec().size());
    auto &SemIdx = SE.GetSemanticIndexVec();
    bool match = false;
    for (uint32_t offset = 0; offset + SE.GetRows() - 1 < m_SemanticIndexBuffer.size(); offset++) {
      match = true;
      for (uint32_t row = 0; row < SE.GetRows(); row++) {
        if ((uint32_t)SemIdx[row] != m_SemanticIndexBuffer[offset + row]) {
          match = false;
          break;
        }
      }
      if (match) {
        E.SemanticIndexes = offset;
        break;
      }
    }
    if (!match) {
      E.SemanticIndexes = m_SemanticIndexBuffer.size();
      for (uint32_t row = 0; row < SemIdx.size(); row++) {
        m_SemanticIndexBuffer.push_back((uint32_t)SemIdx[row]);
      }
    }
    DXASSERT_NOMSG(SE.GetRows() <= 32);
    E.Rows = (uint8_t)SE.GetRows();
    DXASSERT_NOMSG(SE.GetCols() <= 4);
    E.ColsAndStart = (uint8_t)SE.GetCols() & 0xF;
    if (SE.IsAllocated()) {
      DXASSERT_NOMSG(SE.GetStartCol() < 4);
      DXASSERT_NOMSG(SE.GetStartRow() < 32);
      E.ColsAndStart |= 0x40 | (SE.GetStartCol() << 4);
      E.StartRow = (uint8_t)SE.GetStartRow();
    }
    E.SemanticKind = (uint8_t)SE.GetKind();
    E.ComponentType = (uint8_t)CompTypeToSigCompType(SE.GetCompType(),
      /*i1ToUnknownCompat*/DXIL::CompareVersions(m_ValMajor, m_ValMinor, 1, 5) < 0);
    E.InterpolationMode = (uint8_t)SE.GetInterpolationMode()->GetKind();
    DXASSERT_NOMSG(SE.GetOutputStream() < 4);
    E.DynamicMaskAndStream = (uint8_t)((SE.GetOutputStream() & 0x3) << 4);
    E.DynamicMaskAndStream |= (SE.GetDynIdxCompMask()) & 0xF;
  }

  const uint32_t *CopyViewIDState(const uint32_t *pSrc, uint32_t InputScalars, uint32_t OutputScalars, PSVComponentMask ViewIDMask, PSVDependencyTable IOTable) {
    unsigned MaskDwords = PSVComputeMaskDwordsFromVectors(PSVALIGN4(OutputScalars) / 4);
    if (ViewIDMask.IsValid()) {
      DXASSERT_NOMSG(!IOTable.Table || ViewIDMask.NumVectors == IOTable.OutputVectors);
      memcpy(ViewIDMask.Mask, pSrc, 4 * MaskDwords);
      pSrc += MaskDwords;
    }
    if (IOTable.IsValid() && IOTable.InputVectors && IOTable.OutputVectors) {
      DXASSERT_NOMSG((InputScalars <= IOTable.InputVectors * 4) && (IOTable.InputVectors * 4 - InputScalars < 4));
      DXASSERT_NOMSG((OutputScalars <= IOTable.OutputVectors * 4) && (IOTable.OutputVectors * 4 - OutputScalars < 4));
      memcpy(IOTable.Table, pSrc, 4 * MaskDwords * InputScalars);
      pSrc += MaskDwords * InputScalars;
    }
    return pSrc;
  }

public:
  DxilPSVWriter(const DxilModule &mod, uint32_t PSVVersion = UINT_MAX)
  : m_Module(mod),
    m_PSVInitInfo(PSVVersion)
  {
    m_Module.GetValidatorVersion(m_ValMajor, m_ValMinor);
    // Constraint PSVVersion based on validator version
    if (PSVVersion > 0 && DXIL::CompareVersions(m_ValMajor, m_ValMinor, 1, 1) < 0)
      m_PSVInitInfo.PSVVersion = 0;
    else if (PSVVersion > 1 && DXIL::CompareVersions(m_ValMajor, m_ValMinor, 1, 6) < 0)
      m_PSVInitInfo.PSVVersion = 1;
    else if (PSVVersion > MAX_PSV_VERSION)
      m_PSVInitInfo.PSVVersion = MAX_PSV_VERSION;

    const ShaderModel *SM = m_Module.GetShaderModel();
    UINT uCBuffers = m_Module.GetCBuffers().size();
    UINT uSamplers = m_Module.GetSamplers().size();
    UINT uSRVs = m_Module.GetSRVs().size();
    UINT uUAVs = m_Module.GetUAVs().size();
    m_PSVInitInfo.ResourceCount = uCBuffers + uSamplers + uSRVs + uUAVs;
    // TODO: for >= 6.2 version, create more efficient structure
    if (m_PSVInitInfo.PSVVersion > 0) {
      m_PSVInitInfo.ShaderStage = (PSVShaderKind)SM->GetKind();
      // Copy Dxil Signatures
      m_StringBuffer.push_back('\0'); // For empty semantic name (system value)
      m_PSVInitInfo.SigInputElements = m_Module.GetInputSignature().GetElements().size();
      m_SigInputElements.resize(m_PSVInitInfo.SigInputElements);
      m_PSVInitInfo.SigOutputElements = m_Module.GetOutputSignature().GetElements().size();
      m_SigOutputElements.resize(m_PSVInitInfo.SigOutputElements);
      m_PSVInitInfo.SigPatchConstOrPrimElements = m_Module.GetPatchConstOrPrimSignature().GetElements().size();
      m_SigPatchConstOrPrimElements.resize(m_PSVInitInfo.SigPatchConstOrPrimElements);
      uint32_t i = 0;
      for (auto &SE : m_Module.GetInputSignature().GetElements()) {
        SetPSVSigElement(m_SigInputElements[i++], *(SE.get()));
      }
      i = 0;
      for (auto &SE : m_Module.GetOutputSignature().GetElements()) {
        SetPSVSigElement(m_SigOutputElements[i++], *(SE.get()));
      }
      i = 0;
      for (auto &SE : m_Module.GetPatchConstOrPrimSignature().GetElements()) {
        SetPSVSigElement(m_SigPatchConstOrPrimElements[i++], *(SE.get()));
      }
      // Set String and SemanticInput Tables
      m_PSVInitInfo.StringTable.Table = m_StringBuffer.data();
      m_PSVInitInfo.StringTable.Size = m_StringBuffer.size();
      m_PSVInitInfo.SemanticIndexTable.Table = m_SemanticIndexBuffer.data();
      m_PSVInitInfo.SemanticIndexTable.Entries = m_SemanticIndexBuffer.size();
      // Set up ViewID and signature dependency info
      m_PSVInitInfo.UsesViewID = m_Module.m_ShaderFlags.GetViewID() ? true : false;
      m_PSVInitInfo.SigInputVectors = m_Module.GetInputSignature().NumVectorsUsed(0);
      for (unsigned streamIndex = 0; streamIndex < 4; streamIndex++) {
        m_PSVInitInfo.SigOutputVectors[streamIndex] = m_Module.GetOutputSignature().NumVectorsUsed(streamIndex);
      }
      m_PSVInitInfo.SigPatchConstOrPrimVectors = 0;
      if (SM->IsHS() || SM->IsDS() || SM->IsMS()) {
        m_PSVInitInfo.SigPatchConstOrPrimVectors = m_Module.GetPatchConstOrPrimSignature().NumVectorsUsed(0);
      }
    }
    if (!m_PSV.InitNew(m_PSVInitInfo, nullptr, &m_PSVBufferSize)) {
      DXASSERT(false, "PSV InitNew failed computing size!");
    }
  }
  uint32_t size() const override {
    return m_PSVBufferSize;
  }

  void write(AbstractMemoryStream *pStream) override {
    m_PSVBuffer.resize(m_PSVBufferSize);
    if (!m_PSV.InitNew(m_PSVInitInfo, m_PSVBuffer.data(), &m_PSVBufferSize)) {
      DXASSERT(false, "PSV InitNew failed!");
    }
    DXASSERT_NOMSG(m_PSVBuffer.size() == m_PSVBufferSize);

    // Set DxilRuntimeInfo
    PSVRuntimeInfo0* pInfo = m_PSV.GetPSVRuntimeInfo0();
    PSVRuntimeInfo1* pInfo1 = m_PSV.GetPSVRuntimeInfo1();
    PSVRuntimeInfo2* pInfo2 = m_PSV.GetPSVRuntimeInfo2();
    const ShaderModel* SM = m_Module.GetShaderModel();
    pInfo->MinimumExpectedWaveLaneCount = 0;
    pInfo->MaximumExpectedWaveLaneCount = (UINT)-1;

    switch (SM->GetKind()) {
      case ShaderModel::Kind::Vertex: {
        pInfo->VS.OutputPositionPresent = 0;
        const DxilSignature &S = m_Module.GetOutputSignature();
        for (auto &&E : S.GetElements()) {
          if (E->GetKind() == Semantic::Kind::Position) {
            // Ideally, we might check never writes mask here,
            // but this is not yet part of the signature element in Dxil
            pInfo->VS.OutputPositionPresent = 1;
            break;
          }
        }
        break;
      }
      case ShaderModel::Kind::Hull: {
        pInfo->HS.InputControlPointCount = (UINT)m_Module.GetInputControlPointCount();
        pInfo->HS.OutputControlPointCount = (UINT)m_Module.GetOutputControlPointCount();
        pInfo->HS.TessellatorDomain = (UINT)m_Module.GetTessellatorDomain();
        pInfo->HS.TessellatorOutputPrimitive = (UINT)m_Module.GetTessellatorOutputPrimitive();
        break;
      }
      case ShaderModel::Kind::Domain: {
        pInfo->DS.InputControlPointCount = (UINT)m_Module.GetInputControlPointCount();
        pInfo->DS.OutputPositionPresent = 0;
        const DxilSignature &S = m_Module.GetOutputSignature();
        for (auto &&E : S.GetElements()) {
          if (E->GetKind() == Semantic::Kind::Position) {
            // Ideally, we might check never writes mask here,
            // but this is not yet part of the signature element in Dxil
            pInfo->DS.OutputPositionPresent = 1;
            break;
          }
        }
        pInfo->DS.TessellatorDomain = (UINT)m_Module.GetTessellatorDomain();
        break;
      }
      case ShaderModel::Kind::Geometry: {
        pInfo->GS.InputPrimitive = (UINT)m_Module.GetInputPrimitive();
        // NOTE: For OutputTopology, pick one from a used stream, or if none
        // are used, use stream 0, and set OutputStreamMask to 1.
        pInfo->GS.OutputTopology = (UINT)m_Module.GetStreamPrimitiveTopology();
        pInfo->GS.OutputStreamMask = m_Module.GetActiveStreamMask();
        if (pInfo->GS.OutputStreamMask == 0) {
          pInfo->GS.OutputStreamMask = 1; // This is what runtime expects.
        }
        pInfo->GS.OutputPositionPresent = 0;
        const DxilSignature &S = m_Module.GetOutputSignature();
        for (auto &&E : S.GetElements()) {
          if (E->GetKind() == Semantic::Kind::Position) {
            // Ideally, we might check never writes mask here,
            // but this is not yet part of the signature element in Dxil
            pInfo->GS.OutputPositionPresent = 1;
            break;
          }
        }
        break;
      }
      case ShaderModel::Kind::Pixel: {
        pInfo->PS.DepthOutput = 0;
        pInfo->PS.SampleFrequency = 0;
        {
          const DxilSignature &S = m_Module.GetInputSignature();
          for (auto &&E : S.GetElements()) {
            if (E->GetInterpolationMode()->IsAnySample() ||
                E->GetKind() == Semantic::Kind::SampleIndex) {
              pInfo->PS.SampleFrequency = 1;
            }
          }
        }
        {
          const DxilSignature &S = m_Module.GetOutputSignature();
          for (auto &&E : S.GetElements()) {
            if (E->IsAnyDepth()) {
              pInfo->PS.DepthOutput = 1;
              break;
            }
          }
        }
        break;
      }
      case ShaderModel::Kind::Compute: {
        UINT waveSize = (UINT)m_Module.GetWaveSize();
        if (waveSize != 0) {
          pInfo->MinimumExpectedWaveLaneCount = waveSize;
          pInfo->MaximumExpectedWaveLaneCount = waveSize;
        }
        break;
      }
      case ShaderModel::Kind::Library:
      case ShaderModel::Kind::Invalid:
        // Library and Invalid not relevant to PSVRuntimeInfo0
        break;
      case ShaderModel::Kind::Mesh: {
        pInfo->MS.MaxOutputVertices = (UINT)m_Module.GetMaxOutputVertices();
        pInfo->MS.MaxOutputPrimitives = (UINT)m_Module.GetMaxOutputPrimitives();
        pInfo1->MS1.MeshOutputTopology = (UINT)m_Module.GetMeshOutputTopology();
        Module *mod = m_Module.GetModule();
        const DataLayout &DL = mod->getDataLayout();
        unsigned totalByteSize = 0;
        for (GlobalVariable &GV : mod->globals()) {
          PointerType *gvPtrType = cast<PointerType>(GV.getType());
          if (gvPtrType->getAddressSpace() == hlsl::DXIL::kTGSMAddrSpace) {
            Type *gvType = gvPtrType->getPointerElementType();
            unsigned byteSize = DL.getTypeAllocSize(gvType);
            totalByteSize += byteSize;
          }
        }
        pInfo->MS.GroupSharedBytesUsed = totalByteSize;
        pInfo->MS.PayloadSizeInBytes = m_Module.GetPayloadSizeInBytes();
        break;
      }
      case ShaderModel::Kind::Amplification: {
        pInfo->AS.PayloadSizeInBytes = m_Module.GetPayloadSizeInBytes();
        break;
      }
    }
    if (pInfo2) {
      switch (SM->GetKind()) {
      case ShaderModel::Kind::Compute:
      case ShaderModel::Kind::Mesh:
      case ShaderModel::Kind::Amplification:
        pInfo2->NumThreadsX = m_Module.GetNumThreads(0);
        pInfo2->NumThreadsY = m_Module.GetNumThreads(1);
        pInfo2->NumThreadsZ = m_Module.GetNumThreads(2);
        break;
      }
    }

    // Set resource binding information
    UINT uResIndex = 0;
    for (auto &&R : m_Module.GetCBuffers()) {
      DXASSERT_NOMSG(uResIndex < m_PSVInitInfo.ResourceCount);
      PSVResourceBindInfo0* pBindInfo = m_PSV.GetPSVResourceBindInfo0(uResIndex);
      PSVResourceBindInfo1* pBindInfo1 = m_PSV.GetPSVResourceBindInfo1(uResIndex);
      DXASSERT_NOMSG(pBindInfo);
      pBindInfo->ResType = (UINT)PSVResourceType::CBV;
      pBindInfo->Space = R->GetSpaceID();
      pBindInfo->LowerBound = R->GetLowerBound();
      pBindInfo->UpperBound = R->GetUpperBound();
      if (pBindInfo1) {
        pBindInfo1->ResKind = (UINT)R->GetKind();
      }
      uResIndex++;
    }
    for (auto &&R : m_Module.GetSamplers()) {
      DXASSERT_NOMSG(uResIndex < m_PSVInitInfo.ResourceCount);
      PSVResourceBindInfo0* pBindInfo = m_PSV.GetPSVResourceBindInfo0(uResIndex);
      PSVResourceBindInfo1* pBindInfo1 = m_PSV.GetPSVResourceBindInfo1(uResIndex);
      DXASSERT_NOMSG(pBindInfo);
      pBindInfo->ResType = (UINT)PSVResourceType::Sampler;
      pBindInfo->Space = R->GetSpaceID();
      pBindInfo->LowerBound = R->GetLowerBound();
      pBindInfo->UpperBound = R->GetUpperBound();
      if (pBindInfo1) {
        pBindInfo1->ResKind = (UINT)R->GetKind();
      }
      uResIndex++;
    }
    for (auto &&R : m_Module.GetSRVs()) {
      DXASSERT_NOMSG(uResIndex < m_PSVInitInfo.ResourceCount);
      PSVResourceBindInfo0* pBindInfo = m_PSV.GetPSVResourceBindInfo0(uResIndex);
      PSVResourceBindInfo1* pBindInfo1 = m_PSV.GetPSVResourceBindInfo1(uResIndex);
      DXASSERT_NOMSG(pBindInfo);
      if (R->IsStructuredBuffer()) {
        pBindInfo->ResType = (UINT)PSVResourceType::SRVStructured;
      } else if (R->IsRawBuffer() || (R->GetKind() == DxilResourceBase::Kind::RTAccelerationStructure)) {
        pBindInfo->ResType = (UINT)PSVResourceType::SRVRaw;
      } else {
        pBindInfo->ResType = (UINT)PSVResourceType::SRVTyped;
      }
      pBindInfo->Space = R->GetSpaceID();
      pBindInfo->LowerBound = R->GetLowerBound();
      pBindInfo->UpperBound = R->GetUpperBound();
      if (pBindInfo1) {
        pBindInfo1->ResKind = (UINT)R->GetKind();
      }
      uResIndex++;
    }
    for (auto &&R : m_Module.GetUAVs()) {
      DXASSERT_NOMSG(uResIndex < m_PSVInitInfo.ResourceCount);
      PSVResourceBindInfo0* pBindInfo = m_PSV.GetPSVResourceBindInfo0(uResIndex);
      PSVResourceBindInfo1* pBindInfo1 = m_PSV.GetPSVResourceBindInfo1(uResIndex);
      DXASSERT_NOMSG(pBindInfo);
      if (R->IsStructuredBuffer()) {
        if (R->HasCounter())
          pBindInfo->ResType = (UINT)PSVResourceType::UAVStructuredWithCounter;
        else
          pBindInfo->ResType = (UINT)PSVResourceType::UAVStructured;
      } else if (R->IsRawBuffer()) {
        pBindInfo->ResType = (UINT)PSVResourceType::UAVRaw;
      } else {
        pBindInfo->ResType = (UINT)PSVResourceType::UAVTyped;
      }
      pBindInfo->Space = R->GetSpaceID();
      pBindInfo->LowerBound = R->GetLowerBound();
      pBindInfo->UpperBound = R->GetUpperBound();
      if (pBindInfo1) {
        pBindInfo1->ResKind = (UINT)R->GetKind();
        pBindInfo1->ResFlags |= R->HasAtomic64Use()? (UINT)PSVResourceFlag::UsedByAtomic64 : 0;
      }
      uResIndex++;
    }
    DXASSERT_NOMSG(uResIndex == m_PSVInitInfo.ResourceCount);

    if (m_PSVInitInfo.PSVVersion > 0) {
      DXASSERT_NOMSG(pInfo1);

      // Write MaxVertexCount
      if (SM->IsGS()) {
        DXASSERT_NOMSG(m_Module.GetMaxVertexCount() <= 1024);
        pInfo1->MaxVertexCount = (uint16_t)m_Module.GetMaxVertexCount();
      }

      // Write Dxil Signature Elements
      for (unsigned i = 0; i < m_PSV.GetSigInputElements(); i++) {
        PSVSignatureElement0 *pInputElement = m_PSV.GetInputElement0(i);
        DXASSERT_NOMSG(pInputElement);
        memcpy(pInputElement, &m_SigInputElements[i], sizeof(PSVSignatureElement0));
      }
      for (unsigned i = 0; i < m_PSV.GetSigOutputElements(); i++) {
        PSVSignatureElement0 *pOutputElement = m_PSV.GetOutputElement0(i);
        DXASSERT_NOMSG(pOutputElement);
        memcpy(pOutputElement, &m_SigOutputElements[i], sizeof(PSVSignatureElement0));
      }
      for (unsigned i = 0; i < m_PSV.GetSigPatchConstOrPrimElements(); i++) {
        PSVSignatureElement0 *pPatchConstOrPrimElement = m_PSV.GetPatchConstOrPrimElement0(i);
        DXASSERT_NOMSG(pPatchConstOrPrimElement);
        memcpy(pPatchConstOrPrimElement, &m_SigPatchConstOrPrimElements[i], sizeof(PSVSignatureElement0));
      }

      // Gather ViewID dependency information
      auto &viewState = m_Module.GetSerializedViewIdState();
      if (!viewState.empty()) {
        const uint32_t *pSrc = viewState.data();
        const uint32_t InputScalars = *(pSrc++);
        uint32_t OutputScalars[4];
        for (unsigned streamIndex = 0; streamIndex < 4; streamIndex++) {
          OutputScalars[streamIndex] = *(pSrc++);
          pSrc = CopyViewIDState(pSrc, InputScalars, OutputScalars[streamIndex], m_PSV.GetViewIDOutputMask(streamIndex), m_PSV.GetInputToOutputTable(streamIndex));
          if (!SM->IsGS())
            break;
        }
        if (SM->IsHS() || SM->IsMS()) {
          const uint32_t PCScalars = *(pSrc++);
          pSrc = CopyViewIDState(pSrc, InputScalars, PCScalars, m_PSV.GetViewIDPCOutputMask(), m_PSV.GetInputToPCOutputTable());
        } else if (SM->IsDS()) {
          const uint32_t PCScalars = *(pSrc++);
          pSrc = CopyViewIDState(pSrc, PCScalars, OutputScalars[0], PSVComponentMask(), m_PSV.GetPCInputToOutputTable());
        }
        DXASSERT_NOMSG(viewState.data() + viewState.size() == pSrc);
      }
    }

    ULONG cbWritten;
    IFT(pStream->Write(m_PSVBuffer.data(), m_PSVBufferSize, &cbWritten));
    DXASSERT_NOMSG(cbWritten == m_PSVBufferSize);
  }
};

// Size-checked writer
//  on overrun: throw buffer_overrun{};
//  on overlap: throw buffer_overlap{};
class CheckedWriter {
  char *Ptr;
  size_t Size;
  size_t Offset;

public:
  class exception : public std::exception {};
  class buffer_overrun : public exception {
  public:
    buffer_overrun() noexcept {}
    virtual const char * what() const noexcept override {
      return ("buffer_overrun");
    }
  };
  class buffer_overlap : public exception {
  public:
    buffer_overlap() noexcept {}
    virtual const char * what() const noexcept override {
      return ("buffer_overlap");
    }
  };

  CheckedWriter(void *ptr, size_t size) :
    Ptr(reinterpret_cast<char*>(ptr)), Size(size), Offset(0) {}

  size_t GetOffset() const { return Offset; }
  void Reset(size_t offset = 0) {
    if (offset >= Size) throw buffer_overrun{};
    Offset = offset;
  }
  // offset is absolute, ensure offset is >= current offset
  void Advance(size_t offset = 0) {
    if (offset < Offset) throw buffer_overlap{};
    if (offset >= Size) throw buffer_overrun{};
    Offset = offset;
  }
  void CheckBounds(size_t size) const {
    assert(Offset <= Size && "otherwise, offset larger than size");
    if (size > Size - Offset)
      throw buffer_overrun{};
  }
  template <typename T>
  T *Cast(size_t size = 0) {
    if (0 == size) size = sizeof(T);
    CheckBounds(size);
    return reinterpret_cast<T*>(Ptr + Offset);
  }

  // Map and Write advance Offset:
  template <typename T>
  T &Map() {
    const size_t size = sizeof(T);
    T * p = Cast<T>(size);
    Offset += size;
    return *p;
  }
  template <typename T>
  T *MapArray(size_t count = 1) {
    const size_t size = sizeof(T) * count;
    T *p = Cast<T>(size);
    Offset += size;
    return p;
  }
  template <typename T>
  void Write(const T &obj) {
    const size_t size = sizeof(T);
    *Cast<T>(size) = obj;
    Offset += size;
  }
  template <typename T>
  void WriteArray(const T *pArray, size_t count = 1) {
    const size_t size = sizeof(T) * count;
    memcpy(Cast<T>(size), pArray, size);
    Offset += size;
  }
};

// Like DXIL container, RDAT itself is a mini container that contains multiple RDAT parts
class RDATPart {
public:
  virtual uint32_t GetPartSize() const { return 0; }
  virtual void Write(void *ptr) {}
  virtual RuntimeDataPartType GetType() const { return RuntimeDataPartType::Invalid; }
  virtual ~RDATPart() {}
};

// Most RDAT parts are tables each containing a list of structures of same type.
// Exceptions are string table and index table because each string or list of
// indicies can be of different sizes.
class RDATTable : public RDATPart {
protected:
  // m_map is map of records to their index.
  // Used to alias identical records.
  std::unordered_map<std::string, uint32_t> m_map;
  std::vector<StringRef> m_rows;
  size_t m_RecordStride = 0;
  bool m_bDeduplicationEnabled = false;
  RuntimeDataPartType m_Type = RuntimeDataPartType::Invalid;
public:
  virtual ~RDATTable() {}

  void SetType(RuntimeDataPartType type) { m_Type = type; }
  RuntimeDataPartType GetType() const { return m_Type; }
  void SetRecordStride(size_t RecordStride) {
    DXASSERT(m_rows.empty(), "record stride is fixed for the entire table");
    m_RecordStride = RecordStride;
  }
  size_t GetRecordStride() const { return m_RecordStride; }
  void SetDeduplication(bool bEnabled = true) { m_bDeduplicationEnabled = bEnabled; }

  uint32_t Count() {
    size_t count = m_rows.size();
    return (count < UINT32_MAX) ? count : 0;
  }

  template<typename RecordType>
  uint32_t Insert(const RecordType &data) {
    IFTBOOL(m_RecordStride <= sizeof(RecordType), DXC_E_GENERAL_INTERNAL_ERROR);
    size_t count = m_rows.size();
    if (count < (UINT32_MAX - 1)) {
      const char *pData = (const char*)(const void *)&data;
      auto result = m_map.insert(std::make_pair(std::string(pData, pData + m_RecordStride), count));
      if (!m_bDeduplicationEnabled || result.second) {
        m_rows.emplace_back(result.first->first);
        return count;
      } else {
        return result.first->second;
      }
    }
    return RDAT_NULL_REF;
  }

  void Write(void *ptr) {
    char *pCur = (char*)ptr;
    RuntimeDataTableHeader &header = *reinterpret_cast<RuntimeDataTableHeader*>(pCur);
    header.RecordCount = m_rows.size();
    header.RecordStride = m_RecordStride;
    pCur += sizeof(RuntimeDataTableHeader);
    for (auto &record : m_rows) {
      DXASSERT_NOMSG(record.size() == m_RecordStride);
      memcpy(pCur, record.data(), m_RecordStride);
      pCur += m_RecordStride;
    }
  };

  uint32_t GetPartSize() const {
    if (m_rows.empty())
      return 0;
    return sizeof(RuntimeDataTableHeader) + m_rows.size() * m_RecordStride;
  }
};

class StringBufferPart : public RDATPart {
private:
  StringMap<uint32_t> m_StringMap;
  SmallVector<char, 256> m_StringBuffer;
public:
  StringBufferPart() : m_StringMap(), m_StringBuffer() {
    // Always start string table with null so empty/null strings have offset of zero
    m_StringBuffer.push_back('\0');
  }
  // returns the offset of the name inserted
  uint32_t Insert(StringRef name) {
    if (name.empty())
      return 0;

    // Don't add duplicate strings
    auto found = m_StringMap.find(name);
    if (found != m_StringMap.end())
      return found->second;

    uint32_t prevIndex = (uint32_t)m_StringBuffer.size();
    m_StringMap[name] = prevIndex;
    m_StringBuffer.reserve(m_StringBuffer.size() + name.size() + 1);
    m_StringBuffer.append(name.begin(), name.end());
    m_StringBuffer.push_back('\0');
    return prevIndex;
  }
  RuntimeDataPartType GetType() const { return RuntimeDataPartType::StringBuffer; }
  uint32_t GetPartSize() const { return m_StringBuffer.size(); }
  void Write(void *ptr) { memcpy(ptr, m_StringBuffer.data(), m_StringBuffer.size()); }
};

struct IndexArraysPart : public RDATPart {
private:
  std::vector<uint32_t> m_IndexBuffer;

  // Use m_IndexSet with CmpIndices to avoid duplicate index arrays
  struct CmpIndices {
    const IndexArraysPart &Table;
    CmpIndices(const IndexArraysPart &table) : Table(table) {}
    bool operator()(uint32_t left, uint32_t right) const {
      const uint32_t *pLeft = Table.m_IndexBuffer.data() + left;
      const uint32_t *pRight = Table.m_IndexBuffer.data() + right;
      if (*pLeft != *pRight)
        return (*pLeft < *pRight);
      uint32_t count = *pLeft;
      for (unsigned i = 0; i < count; i++) {
        ++pLeft; ++pRight;
        if (*pLeft != *pRight)
          return (*pLeft < *pRight);
      }
      return false;
    }
  };
  std::set<uint32_t, CmpIndices> m_IndexSet;

public:
  IndexArraysPart() : m_IndexBuffer(), m_IndexSet(*this) {}
  template <class iterator>
  uint32_t AddIndex(iterator begin, iterator end) {
    uint32_t newOffset = m_IndexBuffer.size();
    m_IndexBuffer.push_back(0); // Size: update after insertion
    m_IndexBuffer.insert(m_IndexBuffer.end(), begin, end);
    m_IndexBuffer[newOffset] = (m_IndexBuffer.size() - newOffset) - 1;
    // Check for duplicate, return new offset if not duplicate
    auto insertResult = m_IndexSet.insert(newOffset);
    if (insertResult.second)
      return newOffset;
    // Otherwise it was a duplicate, so chop off the size and return the original
    m_IndexBuffer.resize(newOffset);
    return *insertResult.first;
  }

  RuntimeDataPartType GetType() const { return RuntimeDataPartType::IndexArrays; }
  uint32_t GetPartSize() const {
    return sizeof(uint32_t) * m_IndexBuffer.size();
  }

  void Write(void *ptr) {
    memcpy(ptr, m_IndexBuffer.data(), m_IndexBuffer.size() * sizeof(uint32_t));
  }
};

class RawBytesPart : public RDATPart {
private:
  std::unordered_map<const void *, uint32_t> m_PtrMap;
  std::vector<char> m_DataBuffer;
public:
  RawBytesPart() : m_DataBuffer() {}
  uint32_t Insert(const void *pData, size_t dataSize) {
    auto it = m_PtrMap.find(pData);
    if (it != m_PtrMap.end())
      return it->second;

    if (dataSize + m_DataBuffer.size() > UINT_MAX)
      return UINT_MAX;
    uint32_t offset = (uint32_t)m_DataBuffer.size();
    m_DataBuffer.reserve(m_DataBuffer.size() + dataSize);
    m_DataBuffer.insert(m_DataBuffer.end(),
      (const char*)pData, (const char*)pData + dataSize);
    return offset;
  }
  RuntimeDataPartType GetType() const { return RuntimeDataPartType::RawBytes; }
  uint32_t GetPartSize() const { return m_DataBuffer.size(); }
  void Write(void *ptr) { memcpy(ptr, m_DataBuffer.data(), m_DataBuffer.size()); }
};

using namespace DXIL;

class DxilRDATWriter : public DxilPartWriter {
private:
  SmallVector<char, 1024> m_RDATBuffer;

  std::vector<std::unique_ptr<RDATPart>> m_Parts;
  typedef llvm::SmallSetVector<uint32_t, 8> Indices;
  typedef std::unordered_map<const llvm::Function *, Indices> FunctionIndexMap;
  FunctionIndexMap m_FuncToResNameOffset; // list of resources used
  FunctionIndexMap m_FuncToDependencies;  // list of unresolved functions used

  unsigned m_ValMajor, m_ValMinor;

  struct ShaderCompatInfo {
    ShaderCompatInfo()
      : minMajor(6), minMinor(0),
        mask(((unsigned)1 << (unsigned)DXIL::ShaderKind::Invalid) - 1)
      {}
    unsigned minMajor, minMinor, mask;
  };
  typedef std::unordered_map<const llvm::Function*, ShaderCompatInfo> FunctionShaderCompatMap;
  FunctionShaderCompatMap m_FuncToShaderCompat;

  void UpdateFunctionToShaderCompat(const llvm::Function* dxilFunc) {
#define SFLAG(stage) ((unsigned)1 << (unsigned)DXIL::ShaderKind::stage)
    for (const llvm::User *user : dxilFunc->users()) {
      if (const llvm::CallInst *CI = dyn_cast<const llvm::CallInst>(user)) {
        // Find calling function
        const llvm::Function *F = cast<const llvm::Function>(CI->getParent()->getParent());
        // Insert or lookup info
        ShaderCompatInfo &info = m_FuncToShaderCompat[F];
        unsigned major, minor, mask;
        // bWithTranslation = true for library modules
        OP::GetMinShaderModelAndMask(CI, /*bWithTranslation*/true,
                                     m_ValMajor, m_ValMinor,
                                     major, minor, mask);
        if (major > info.minMajor) {
          info.minMajor = major;
          info.minMinor = minor;
        } else if (major == info.minMajor && minor > info.minMinor) {
          info.minMinor = minor;
        }
        info.mask &= mask;
      } else if (const llvm::LoadInst *LI = dyn_cast<LoadInst>(user)) {
        // If loading a groupshared variable, limit to CS/AS/MS
        if (LI->getPointerAddressSpace() == DXIL::kTGSMAddrSpace) {
          const llvm::Function *F = cast<const llvm::Function>(LI->getParent()->getParent());
          ShaderCompatInfo &info = m_FuncToShaderCompat[F];
          info.mask &= (SFLAG(Compute) | SFLAG(Mesh) | SFLAG(Amplification));
        }
      } else if (const llvm::StoreInst *SI = dyn_cast<StoreInst>(user)) {
        // If storing to a groupshared variable, limit to CS/AS/MS
        if (SI->getPointerAddressSpace() == DXIL::kTGSMAddrSpace) {
          const llvm::Function *F = cast<const llvm::Function>(SI->getParent()->getParent());
          ShaderCompatInfo &info = m_FuncToShaderCompat[F];
          info.mask &= (SFLAG(Compute) | SFLAG(Mesh) | SFLAG(Amplification));
        }
      }

    }
#undef SFLAG
  }

  void
  FindUsingFunctions(const llvm::Value *User,
                    llvm::SmallVectorImpl<const llvm::Function *> &functions) {
    if (const llvm::Instruction *I = dyn_cast<const llvm::Instruction>(User)) {
      // Instruction should be inside a basic block, which is in a function
      functions.push_back(cast<const llvm::Function>(I->getParent()->getParent()));
      return;
    }
    // User can be either instruction, constant, or operator. But User is an
    // operator only if constant is a scalar value, not resource pointer.
    const llvm::Constant *CU = cast<const llvm::Constant>(User);
    for (auto U : CU->users())
      FindUsingFunctions(U, functions);
  }

  void UpdateFunctionToResourceInfo(const DxilResourceBase *resource,
                                    uint32_t offset) {
    Constant *var = resource->GetGlobalSymbol();
    if (var) {
      for (auto user : var->users()) {
        // Find the function(s).
        llvm::SmallVector<const llvm::Function*, 8> functions;
        FindUsingFunctions(user, functions);
        for (const llvm::Function *F : functions) {
          if (m_FuncToResNameOffset.find(F) == m_FuncToResNameOffset.end()) {
            m_FuncToResNameOffset[F] = Indices();
          }
          m_FuncToResNameOffset[F].insert(offset);
        }
      }
    }
  }

  void InsertToResourceTable(DxilResourceBase &resource,
                             ResourceClass resourceClass,
                             uint32_t &resourceIndex) {
    uint32_t stringIndex = m_pStringBufferPart->Insert(resource.GetGlobalName());
    UpdateFunctionToResourceInfo(&resource, resourceIndex++);
    RuntimeDataResourceInfo info = {};
    info.ID = resource.GetID();
    info.Class = static_cast<uint32_t>(resourceClass);
    info.Kind = static_cast<uint32_t>(resource.GetKind());
    info.Space = resource.GetSpaceID();
    info.LowerBound = resource.GetLowerBound();
    info.UpperBound = resource.GetUpperBound();
    info.Name = stringIndex;
    info.Flags = 0;
    if (ResourceClass::UAV == resourceClass) {
      DxilResource *pRes = static_cast<DxilResource*>(&resource);
      if (pRes->HasCounter())
        info.Flags |= static_cast<uint32_t>(DxilResourceFlag::UAVCounter);
      if (pRes->IsGloballyCoherent())
        info.Flags |= static_cast<uint32_t>(DxilResourceFlag::UAVGloballyCoherent);
      if (pRes->IsROV())
        info.Flags |= static_cast<uint32_t>(DxilResourceFlag::UAVRasterizerOrderedView);
      if (pRes->HasAtomic64Use())
        info.Flags |= static_cast<uint32_t>(DxilResourceFlag::Atomics64Use);
      // TODO: add dynamic index flag
    }
    m_pResourceTable->Insert(info);
  }

  void UpdateResourceInfo(const DxilModule &DM) {
    // Try to allocate string table for resources. String table is a sequence
    // of strings delimited by \0
    uint32_t resourceIndex = 0;
    for (auto &resource : DM.GetCBuffers()) {
      InsertToResourceTable(*resource.get(), ResourceClass::CBuffer, resourceIndex);

    }
    for (auto &resource : DM.GetSamplers()) {
      InsertToResourceTable(*resource.get(), ResourceClass::Sampler, resourceIndex);
    }
    for (auto &resource : DM.GetSRVs()) {
      InsertToResourceTable(*resource.get(), ResourceClass::SRV, resourceIndex);
    }
    for (auto &resource : DM.GetUAVs()) {
      InsertToResourceTable(*resource.get(), ResourceClass::UAV, resourceIndex);
    }
  }

  void UpdateFunctionDependency(llvm::Function *F) {
    for (const llvm::User *user : F->users()) {
      llvm::SmallVector<const llvm::Function*, 8> functions;
      FindUsingFunctions(user, functions);
      for (const llvm::Function *userFunction : functions) {
        uint32_t index = m_pStringBufferPart->Insert(F->getName());
        if (m_FuncToDependencies.find(userFunction) ==
            m_FuncToDependencies.end()) {
          m_FuncToDependencies[userFunction] =
              Indices();
        }
        m_FuncToDependencies[userFunction].insert(index);
      }
    }
  }

  void UpdateFunctionInfo(const DxilModule &DM) {
    llvm::Module *M = DM.GetModule();
    // We must select the appropriate shader mask for the validator version,
    // so we don't set any bits the validator doesn't recognize.
    unsigned ValidShaderMask = (1 << ((unsigned)DXIL::ShaderKind::Amplification + 1)) - 1;
    if (DXIL::CompareVersions(m_ValMajor, m_ValMinor, 1, 5) < 0) {
      ValidShaderMask = (1 << ((unsigned)DXIL::ShaderKind::Callable + 1)) - 1;
    }
    for (auto &function : M->getFunctionList()) {
      if (function.isDeclaration() && !function.isIntrinsic() &&
          function.getLinkage() == llvm::GlobalValue::LinkageTypes::ExternalLinkage) {
        if (OP::IsDxilOpFunc(&function)) {
          // update min shader model and shader stage mask per function
          UpdateFunctionToShaderCompat(&function);
        } else {
          // collect unresolved dependencies per function
          UpdateFunctionDependency(&function);
        }
      }
    }


    for (auto &function : M->getFunctionList()) {
      if (!function.isDeclaration()) {
        StringRef mangled = function.getName();
        StringRef unmangled = hlsl::dxilutil::DemangleFunctionName(function.getName());
        uint32_t mangledIndex = m_pStringBufferPart->Insert(mangled);
        uint32_t unmangledIndex = m_pStringBufferPart->Insert(unmangled);
        // Update resource Index
        uint32_t resourceIndex = RDAT_NULL_REF;
        uint32_t functionDependencies = RDAT_NULL_REF;
        uint32_t payloadSizeInBytes = 0;
        uint32_t attrSizeInBytes = 0;
        uint32_t shaderKind = static_cast<uint32_t>(DXIL::ShaderKind::Library);

        if (m_FuncToResNameOffset.find(&function) != m_FuncToResNameOffset.end())
          resourceIndex =
          m_pIndexArraysPart->AddIndex(m_FuncToResNameOffset[&function].begin(),
                                  m_FuncToResNameOffset[&function].end());
        if (m_FuncToDependencies.find(&function) != m_FuncToDependencies.end())
          functionDependencies =
              m_pIndexArraysPart->AddIndex(m_FuncToDependencies[&function].begin(),
                                  m_FuncToDependencies[&function].end());
        RuntimeDataFunctionInfo info = {};
        ShaderFlags flags = ShaderFlags::CollectShaderFlags(&function, &DM);
        if (DM.HasDxilFunctionProps(&function)) {
          const auto &props = DM.GetDxilFunctionProps(&function);
          if (props.IsClosestHit() || props.IsAnyHit()) {
            payloadSizeInBytes = props.ShaderProps.Ray.payloadSizeInBytes;
            attrSizeInBytes = props.ShaderProps.Ray.attributeSizeInBytes;
          }
          else if (props.IsMiss()) {
            payloadSizeInBytes = props.ShaderProps.Ray.payloadSizeInBytes;
          }
          else if (props.IsCallable()) {
            payloadSizeInBytes = props.ShaderProps.Ray.paramSizeInBytes;
          }
          shaderKind = (uint32_t)props.shaderKind;
        }
        info.Name = mangledIndex;
        info.UnmangledName = unmangledIndex;
        info.ShaderKind = shaderKind;
        info.Resources = resourceIndex;
        info.FunctionDependencies = functionDependencies;
        info.PayloadSizeInBytes = payloadSizeInBytes;
        info.AttributeSizeInBytes = attrSizeInBytes;
        info.SetFeatureFlags(flags.GetFeatureInfo());
        // Init min target 6.0
        unsigned minMajor = 6, minMinor = 0;
        // Increase min target based on feature flags:
        if (flags.GetUseNativeLowPrecision() && flags.GetLowPrecisionPresent()) {
          minMinor = 2;
        } else if (flags.GetBarycentrics() || flags.GetViewID()) {
          minMinor = 1;
        }
        if ((DXIL::ShaderKind)shaderKind == DXIL::ShaderKind::Library) {
          // Init mask to all kinds for library functions
          info.ShaderStageFlag = ValidShaderMask;
        } else {
          // Init mask to current kind for shader functions
          info.ShaderStageFlag = (unsigned)1 << shaderKind;
        }
        auto it = m_FuncToShaderCompat.find(&function);
        if (it != m_FuncToShaderCompat.end()) {
          auto &compatInfo = it->second;
          if (compatInfo.minMajor > minMajor) {
            minMajor = compatInfo.minMajor;
            minMinor = compatInfo.minMinor;
          } else if (compatInfo.minMinor > minMinor) {
            minMinor = compatInfo.minMinor;
          }
          info.ShaderStageFlag &= compatInfo.mask;
        }
        info.MinShaderTarget = EncodeVersion((DXIL::ShaderKind)shaderKind, minMajor, minMinor);
        m_pFunctionTable->Insert(info);
      }
    }
  }

  void UpdateSubobjectInfo(const DxilModule &DM) {
    if (!DM.GetSubobjects())
      return;
    for (auto &it : DM.GetSubobjects()->GetSubobjects()) {
      auto &obj = *it.second;
      RuntimeDataSubobjectInfo info = {};
      info.Name = m_pStringBufferPart->Insert(obj.GetName());
      info.Kind = (uint32_t)obj.GetKind();
      bool bLocalRS = false;
      switch (obj.GetKind()) {
      case DXIL::SubobjectKind::StateObjectConfig:
        obj.GetStateObjectConfig(info.StateObjectConfig.Flags);
        break;
      case DXIL::SubobjectKind::LocalRootSignature:
        bLocalRS = true;
        __fallthrough;
      case DXIL::SubobjectKind::GlobalRootSignature: {
        const void *Data;
        obj.GetRootSignature(bLocalRS, Data, info.RootSignature.Data.Size);
        info.RootSignature.Data.Offset =
          m_pRawBytesPart->Insert(Data, info.RootSignature.Data.Size);
        break;
      }
      case DXIL::SubobjectKind::SubobjectToExportsAssociation: {
        llvm::StringRef Subobject;
        const char * const * Exports;
        uint32_t NumExports;
        std::vector<uint32_t> ExportIndices;
        obj.GetSubobjectToExportsAssociation(Subobject, Exports, NumExports);
        info.SubobjectToExportsAssociation.Subobject =
          m_pStringBufferPart->Insert(Subobject);
        ExportIndices.resize(NumExports);
        for (unsigned i = 0; i < NumExports; ++i) {
          ExportIndices[i] = m_pStringBufferPart->Insert(Exports[i]);
        }
        info.SubobjectToExportsAssociation.Exports =
          m_pIndexArraysPart->AddIndex(
            ExportIndices.begin(), ExportIndices.end());
        break;
      }
      case DXIL::SubobjectKind::RaytracingShaderConfig:
        obj.GetRaytracingShaderConfig(
          info.RaytracingShaderConfig.MaxPayloadSizeInBytes,
          info.RaytracingShaderConfig.MaxAttributeSizeInBytes);
        break;
      case DXIL::SubobjectKind::RaytracingPipelineConfig:
        obj.GetRaytracingPipelineConfig(
          info.RaytracingPipelineConfig.MaxTraceRecursionDepth);
        break;
      case DXIL::SubobjectKind::HitGroup:
      {
        HitGroupType hgType;
        StringRef AnyHit;
        StringRef ClosestHit;
        StringRef Intersection;
        obj.GetHitGroup(hgType, AnyHit, ClosestHit, Intersection);
        info.HitGroup.Type = (uint32_t)hgType;
        info.HitGroup.AnyHit = m_pStringBufferPart->Insert(AnyHit);
        info.HitGroup.ClosestHit = m_pStringBufferPart->Insert(ClosestHit);
        info.HitGroup.Intersection = m_pStringBufferPart->Insert(Intersection);
        break;
      }
      case DXIL::SubobjectKind::RaytracingPipelineConfig1:
        obj.GetRaytracingPipelineConfig1(
            info.RaytracingPipelineConfig1.MaxTraceRecursionDepth,
            info.RaytracingPipelineConfig1.Flags);
        break;
      }
      m_pSubobjectTable->Insert(info);
    }
  }

  void CreateParts() {
    bool bRecordDeduplicationEnabled = DXIL::CompareVersions(m_ValMajor, m_ValMinor, 1, 7) >= 0;
    RuntimeDataPartType maxPartID = RDAT::MaxPartTypeForValVer(m_ValMajor, m_ValMinor);

#define ADD_PART(type)                                                         \
    m_Parts.emplace_back(llvm::make_unique<type>());                           \
    m_p##type = reinterpret_cast<type *>(m_Parts.back().get());

#define ADD_TABLE(type, table)                                                 \
    if (RuntimeDataPartType::table <= maxPartID) {                             \
      m_Parts.emplace_back(llvm::make_unique<RDATTable>());                    \
      m_p##table = reinterpret_cast<RDATTable *>(m_Parts.back().get());        \
      m_p##table->SetRecordStride(sizeof(type));                               \
      m_p##table->SetType(RuntimeDataPartType::table);                         \
      m_p##table->SetDeduplication(bRecordDeduplicationEnabled);               \
    }

    ADD_PART(StringBufferPart);
    ADD_TABLE(RuntimeDataResourceInfo, ResourceTable);
    ADD_TABLE(RuntimeDataFunctionInfo, FunctionTable);
    ADD_PART(IndexArraysPart);
    ADD_PART(RawBytesPart);
    ADD_TABLE(RuntimeDataSubobjectInfo, SubobjectTable);

#undef ADD_PART
#undef ADD_TABLE
  }

  StringBufferPart *m_pStringBufferPart;
  IndexArraysPart *m_pIndexArraysPart;
  RawBytesPart *m_pRawBytesPart;

// Once per table.
#define RDAT_STRUCT_TABLE(type, table) RDATTable *m_p##table = nullptr;
#define DEF_RDAT_TYPES DEF_RDAT_DEFAULTS
#include "dxc/DxilContainer/RDAT_Macros.inl"

public:
  DxilRDATWriter(const DxilModule &mod)
      : m_RDATBuffer(), m_Parts(), m_FuncToResNameOffset() {
    // Keep track of validator version so we can make a compatible RDAT
    mod.GetValidatorVersion(m_ValMajor, m_ValMinor);

    CreateParts();
    UpdateResourceInfo(mod);
    UpdateFunctionInfo(mod);
    if (m_pSubobjectTable)
      UpdateSubobjectInfo(mod);

    // Delete any empty parts:
    std::vector<std::unique_ptr<RDATPart>>::iterator it = m_Parts.begin();
    while (it != m_Parts.end()) {
      if (it->get()->GetPartSize() == 0) {
        it = m_Parts.erase(it);
      }
      else
        it++;
    }
  }

  uint32_t size() const override {
    // header + offset array
    uint32_t total = sizeof(RuntimeDataHeader) + m_Parts.size() * sizeof(uint32_t);
    // For each part: part header + part size
    for (auto &part : m_Parts)
      total += sizeof(RuntimeDataPartHeader) + PSVALIGN4(part->GetPartSize());
    return total;
  }

  void write(AbstractMemoryStream *pStream) override {
    try {
      m_RDATBuffer.resize(size(), 0);
      CheckedWriter W(m_RDATBuffer.data(), m_RDATBuffer.size());
      // write RDAT header
      RuntimeDataHeader &header = W.Map<RuntimeDataHeader>();
      header.Version = RDAT_Version_10;
      header.PartCount = m_Parts.size();
      // map offsets
      uint32_t *offsets = W.MapArray<uint32_t>(header.PartCount);
      // write parts
      unsigned i = 0;
      for (auto &part : m_Parts) {
        offsets[i++] = W.GetOffset();
        RuntimeDataPartHeader &partHeader = W.Map<RuntimeDataPartHeader>();
        partHeader.Type = part->GetType();
        partHeader.Size = PSVALIGN4(part->GetPartSize());
        DXASSERT(partHeader.Size, "otherwise, failed to remove empty part");
        char *bytes = W.MapArray<char>(partHeader.Size);
        part->Write(bytes);
      }
    }
    catch (CheckedWriter::exception e) {
      throw hlsl::Exception(DXC_E_GENERAL_INTERNAL_ERROR, e.what());
    }

    ULONG cbWritten;
    IFT(pStream->Write(m_RDATBuffer.data(), m_RDATBuffer.size(), &cbWritten));
    DXASSERT_NOMSG(cbWritten == m_RDATBuffer.size());
  }
};

DxilPartWriter *hlsl::NewPSVWriter(const DxilModule &M, uint32_t PSVVersion) {
  return new DxilPSVWriter(M, PSVVersion);
}

DxilPartWriter *hlsl::NewRDATWriter(const DxilModule &M) {
  return new DxilRDATWriter(M);
}

class DxilContainerWriter_impl : public DxilContainerWriter  {
private:
  class DxilPart {
  public:
    DxilPartHeader Header;
    WriteFn Write;
    DxilPart(uint32_t fourCC, uint32_t size, WriteFn write) : Write(write) {
      Header.PartFourCC = fourCC;
      Header.PartSize = size;
    }
  };

  llvm::SmallVector<DxilPart, 8> m_Parts;
  bool m_bUnaligned;
  bool m_bHasPrivateData;

public:
  DxilContainerWriter_impl(bool bUnaligned) : m_bUnaligned(bUnaligned), m_bHasPrivateData(false) {}

  void AddPart(uint32_t FourCC, uint32_t Size, WriteFn Write) override {
    // Alignment required for all parts except private data, which must be last.
    IFTBOOL(!m_bHasPrivateData && "private data must be last, and cannot be added twice.", DXC_E_CONTAINER_INVALID);
    if (FourCC == DFCC_PrivateData) {
      m_bHasPrivateData = true;
    } else if (!m_bUnaligned) {
      IFTBOOL((Size % sizeof(uint32_t)) == 0, DXC_E_CONTAINER_INVALID);
    }
    m_Parts.emplace_back(FourCC, Size, Write);
  }

  uint32_t size() const override {
    uint32_t partSize = 0;
    for (auto &part : m_Parts) {
      partSize += part.Header.PartSize;
    }
    return (uint32_t)GetDxilContainerSizeFromParts((uint32_t)m_Parts.size(), partSize);
  }

  void write(AbstractMemoryStream *pStream) override {
    DxilContainerHeader header;
    const uint32_t PartCount = (uint32_t)m_Parts.size();
    uint32_t containerSizeInBytes = size();
    InitDxilContainer(&header, PartCount, containerSizeInBytes);
    IFT(pStream->Reserve(header.ContainerSizeInBytes));
    IFT(WriteStreamValue(pStream, header));
    uint32_t offset = sizeof(header) + (uint32_t)GetOffsetTableSize(PartCount);
    for (auto &&part : m_Parts) {
      IFT(WriteStreamValue(pStream, offset));
      offset += sizeof(DxilPartHeader) + part.Header.PartSize;
    }
    for (auto &&part : m_Parts) {
      IFT(WriteStreamValue(pStream, part.Header));
      size_t start = pStream->GetPosition();
      part.Write(pStream);
      DXASSERT_LOCALVAR(start, pStream->GetPosition() - start == (size_t)part.Header.PartSize, "out of bound");
    }
    DXASSERT(containerSizeInBytes == (uint32_t)pStream->GetPosition(), "else stream size is incorrect");
  }
};

DxilContainerWriter *hlsl::NewDxilContainerWriter(bool bUnaligned) {
  return new DxilContainerWriter_impl(bUnaligned);
}

static bool HasDebugInfoOrLineNumbers(const Module &M) {
  return
    llvm::getDebugMetadataVersionFromModule(M) != 0 ||
    llvm::hasDebugInfo(M);
}

static void GetPaddedProgramPartSize(AbstractMemoryStream *pStream,
                                     uint32_t &bitcodeInUInt32,
                                     uint32_t &bitcodePaddingBytes) {
  bitcodeInUInt32 = pStream->GetPtrSize();
  bitcodePaddingBytes = (bitcodeInUInt32 % 4);
  bitcodeInUInt32 = (bitcodeInUInt32 / 4) + (bitcodePaddingBytes ? 1 : 0);
}

void hlsl::WriteProgramPart(const ShaderModel *pModel,
                             AbstractMemoryStream *pModuleBitcode,
                             IStream *pStream) {
  DXASSERT(pModel != nullptr, "else generation should have failed");
  DxilProgramHeader programHeader;
  uint32_t shaderVersion =
      EncodeVersion(pModel->GetKind(), pModel->GetMajor(), pModel->GetMinor());
  unsigned dxilMajor, dxilMinor;
  pModel->GetDxilVersion(dxilMajor, dxilMinor);
  uint32_t dxilVersion = DXIL::MakeDxilVersion(dxilMajor, dxilMinor);
  InitProgramHeader(programHeader, shaderVersion, dxilVersion, pModuleBitcode->GetPtrSize());

  uint32_t programInUInt32, programPaddingBytes;
  GetPaddedProgramPartSize(pModuleBitcode, programInUInt32,
                           programPaddingBytes);

  ULONG cbWritten;
  IFT(WriteStreamValue(pStream, programHeader));
  IFT(pStream->Write(pModuleBitcode->GetPtr(), pModuleBitcode->GetPtrSize(),
                     &cbWritten));
  if (programPaddingBytes) {
    uint32_t paddingValue = 0;
    IFT(pStream->Write(&paddingValue, programPaddingBytes, &cbWritten));
  }
}

namespace {

class RootSignatureWriter : public DxilPartWriter {
private:
  std::vector<uint8_t> m_Sig;

public:
  RootSignatureWriter(std::vector<uint8_t> &&S) : m_Sig(std::move(S)) {}
  uint32_t size() const { return m_Sig.size(); }
  void write(AbstractMemoryStream *pStream) {
    ULONG cbWritten;
    IFT(pStream->Write(m_Sig.data(), size(), &cbWritten));
  }
};

} // namespace


void hlsl::ReEmitLatestReflectionData(llvm::Module *pM) {
  // Retain usage information in metadata for reflection by:
  // Upgrade validator version, re-emit metadata
  // 0,0 = Not meant to be validated, support latest

  DxilModule &DM = pM->GetOrCreateDxilModule();

  DM.SetValidatorVersion(0, 0);
  DM.ReEmitDxilResources();
  DM.EmitDxilCounters();
}

static std::unique_ptr<Module> CloneModuleForReflection(Module *pM) {
  DxilModule &DM = pM->GetOrCreateDxilModule();

  unsigned ValMajor = 0, ValMinor = 0;
  DM.GetValidatorVersion(ValMajor, ValMinor);

  // Emit the latest reflection metadata
  hlsl::ReEmitLatestReflectionData(pM);

  // Clone module
  std::unique_ptr<Module> reflectionModule( llvm::CloneModule(pM) );

  // Now restore validator version on main module and re-emit metadata.
  DM.SetValidatorVersion(ValMajor, ValMinor);
  DM.ReEmitDxilResources();

  return reflectionModule;
}

void hlsl::StripAndCreateReflectionStream(Module *pReflectionM, uint32_t *pReflectionPartSizeInBytes, AbstractMemoryStream **ppReflectionStreamOut) {
  for (Function &F : pReflectionM->functions()) {
    if (!F.isDeclaration()) {
      F.deleteBody();
    }
  }

  uint32_t reflectPartSizeInBytes = 0;
  CComPtr<AbstractMemoryStream> pReflectionBitcodeStream;

  IFT(CreateMemoryStream(DxcGetThreadMallocNoRef(), &pReflectionBitcodeStream));
  raw_stream_ostream outStream(pReflectionBitcodeStream.p);
  WriteBitcodeToFile(pReflectionM, outStream, false);
  outStream.flush();
  uint32_t reflectInUInt32 = 0, reflectPaddingBytes = 0;
  GetPaddedProgramPartSize(pReflectionBitcodeStream, reflectInUInt32, reflectPaddingBytes);
  reflectPartSizeInBytes = reflectInUInt32 * sizeof(uint32_t) + sizeof(DxilProgramHeader);

  *pReflectionPartSizeInBytes = reflectPartSizeInBytes;
  *ppReflectionStreamOut = pReflectionBitcodeStream.Detach();
}

void hlsl::SerializeDxilContainerForModule(
    DxilModule *pModule, AbstractMemoryStream *pModuleBitcode,
    AbstractMemoryStream *pFinalStream, llvm::StringRef DebugName,
    SerializeDxilFlags Flags, DxilShaderHash *pShaderHashOut,
    AbstractMemoryStream *pReflectionStreamOut,
    AbstractMemoryStream *pRootSigStreamOut,
    void *pPrivateData,
    size_t PrivateDataSize) {
  // TODO: add a flag to update the module and remove information that is not part
  // of DXIL proper and is used only to assemble the container.

  DXASSERT_NOMSG(pModule != nullptr);
  DXASSERT_NOMSG(pModuleBitcode != nullptr);
  DXASSERT_NOMSG(pFinalStream != nullptr);

  unsigned ValMajor, ValMinor;
  pModule->GetValidatorVersion(ValMajor, ValMinor);
  if (DXIL::CompareVersions(ValMajor, ValMinor, 1, 1) < 0)
    Flags &= ~SerializeDxilFlags::IncludeDebugNamePart;
  bool bSupportsShaderHash = DXIL::CompareVersions(ValMajor, ValMinor, 1, 5) >= 0;
  bool bCompat_1_4 = DXIL::CompareVersions(ValMajor, ValMinor, 1, 5) < 0;
  bool bUnaligned = DXIL::CompareVersions(ValMajor, ValMinor, 1, 7) < 0;
  bool bEmitReflection = Flags & SerializeDxilFlags::IncludeReflectionPart ||
                         pReflectionStreamOut;

  DxilContainerWriter_impl writer(bUnaligned);

  // Write the feature part.
  DxilFeatureInfoWriter featureInfoWriter(*pModule);
  writer.AddPart(DFCC_FeatureInfo, featureInfoWriter.size(), [&](AbstractMemoryStream *pStream) {
    featureInfoWriter.write(pStream);
  });

  std::unique_ptr<DxilProgramSignatureWriter> pInputSigWriter = nullptr;
  std::unique_ptr<DxilProgramSignatureWriter> pOutputSigWriter = nullptr;
  std::unique_ptr<DxilProgramSignatureWriter> pPatchConstOrPrimSigWriter = nullptr;
  if (!pModule->GetShaderModel()->IsLib()) {
    DXIL::TessellatorDomain domain = DXIL::TessellatorDomain::Undefined;
    if (pModule->GetShaderModel()->IsHS() || pModule->GetShaderModel()->IsDS())
      domain = pModule->GetTessellatorDomain();
    pInputSigWriter = llvm::make_unique<DxilProgramSignatureWriter>(
        pModule->GetInputSignature(), domain,
        /*IsInput*/ true,
        /*UseMinPrecision*/ pModule->GetUseMinPrecision(),
        bCompat_1_4, bUnaligned);
    pOutputSigWriter = llvm::make_unique<DxilProgramSignatureWriter>(
        pModule->GetOutputSignature(), domain,
        /*IsInput*/ false,
        /*UseMinPrecision*/ pModule->GetUseMinPrecision(),
        bCompat_1_4, bUnaligned);
    // Write the input and output signature parts.
    writer.AddPart(DFCC_InputSignature, pInputSigWriter->size(),
                   [&](AbstractMemoryStream *pStream) {
                     pInputSigWriter->write(pStream);
                   });
    writer.AddPart(DFCC_OutputSignature, pOutputSigWriter->size(),
                   [&](AbstractMemoryStream *pStream) {
                     pOutputSigWriter->write(pStream);
                   });

    pPatchConstOrPrimSigWriter = llvm::make_unique<DxilProgramSignatureWriter>(
        pModule->GetPatchConstOrPrimSignature(), domain,
        /*IsInput*/ pModule->GetShaderModel()->IsDS(),
        /*UseMinPrecision*/ pModule->GetUseMinPrecision(),
        bCompat_1_4, bUnaligned);
    if (pModule->GetPatchConstOrPrimSignature().GetElements().size()) {
      writer.AddPart(DFCC_PatchConstantSignature,
                     pPatchConstOrPrimSigWriter->size(),
                     [&](AbstractMemoryStream *pStream) {
                       pPatchConstOrPrimSigWriter->write(pStream);
                     });
    }
  }
  std::unique_ptr<DxilRDATWriter> pRDATWriter = nullptr;
  std::unique_ptr<DxilPSVWriter> pPSVWriter = nullptr;
  unsigned int major, minor;
  pModule->GetDxilVersion(major, minor);
  RootSignatureWriter rootSigWriter(std::move(pModule->GetSerializedRootSignature())); // Grab RS here
  DXASSERT_NOMSG(pModule->GetSerializedRootSignature().empty());

  bool bMetadataStripped = false;
  if (pModule->GetShaderModel()->IsLib()) {
    DXASSERT(pModule->GetSerializedRootSignature().empty(),
             "otherwise, library has root signature outside subobject definitions");
    // Write the DxilRuntimeData (RDAT) part.
    pRDATWriter = llvm::make_unique<DxilRDATWriter>(*pModule);
    writer.AddPart(
        DFCC_RuntimeData, pRDATWriter->size(),
        [&](AbstractMemoryStream *pStream) { pRDATWriter->write(pStream); });
    bMetadataStripped |= pModule->StripSubobjectsFromMetadata();
    pModule->ResetSubobjects(nullptr);
  } else {
    // Write the DxilPipelineStateValidation (PSV0) part.
    pPSVWriter = llvm::make_unique<DxilPSVWriter>(*pModule);
    writer.AddPart(
        DFCC_PipelineStateValidation, pPSVWriter->size(),
        [&](AbstractMemoryStream *pStream) { pPSVWriter->write(pStream); });

    // Write the root signature (RTS0) part.
    if (rootSigWriter.size()) {
      if (pRootSigStreamOut) {
        // Write root signature wrapped in container for separate output
        // Root signature container should never be unaligned.
        DxilContainerWriter_impl rootSigContainerWriter(false);
        rootSigContainerWriter.AddPart(
          DFCC_RootSignature, rootSigWriter.size(),
          [&](AbstractMemoryStream *pStream) { rootSigWriter.write(pStream); });
        rootSigContainerWriter.write(pRootSigStreamOut);
      }
      if ((Flags & SerializeDxilFlags::StripRootSignature) == 0) {
        // Write embedded root signature
        writer.AddPart(
          DFCC_RootSignature, rootSigWriter.size(),
          [&](AbstractMemoryStream *pStream) { rootSigWriter.write(pStream); });
      }
      bMetadataStripped |= pModule->StripRootSignatureFromMetadata();
    }
  }

  // If metadata was stripped, re-serialize the input module.
  CComPtr<AbstractMemoryStream> pInputProgramStream = pModuleBitcode;
  if (bMetadataStripped) {
    pInputProgramStream.Release();
    IFT(CreateMemoryStream(DxcGetThreadMallocNoRef(), &pInputProgramStream));
    raw_stream_ostream outStream(pInputProgramStream.p);
    WriteBitcodeToFile(pModule->GetModule(), outStream, true);
  }

  // If we have debug information present, serialize it to a debug part, then use the stripped version as the canonical program version.
  CComPtr<AbstractMemoryStream> pProgramStream = pInputProgramStream;
  bool bModuleStripped = false;
  if (HasDebugInfoOrLineNumbers(*pModule->GetModule())) {
    uint32_t debugInUInt32, debugPaddingBytes;
    GetPaddedProgramPartSize(pInputProgramStream, debugInUInt32, debugPaddingBytes);
    if (Flags & SerializeDxilFlags::IncludeDebugInfoPart) {
      writer.AddPart(DFCC_ShaderDebugInfoDXIL, debugInUInt32 * sizeof(uint32_t) + sizeof(DxilProgramHeader), [&](AbstractMemoryStream *pStream) {
        hlsl::WriteProgramPart(pModule->GetShaderModel(), pInputProgramStream, pStream);
      });
    }

    llvm::StripDebugInfo(*pModule->GetModule());
    pModule->StripDebugRelatedCode();
    bModuleStripped = true;
  } else {
    // If no debug info, clear DebugNameDependOnSource
    // (it's default, and this scenario can happen)
    Flags &= ~SerializeDxilFlags::DebugNameDependOnSource;
  }

  uint32_t reflectPartSizeInBytes = 0;
  CComPtr<AbstractMemoryStream> pReflectionBitcodeStream;

  if (bEmitReflection) {
    // Clone module for reflection
    std::unique_ptr<Module> reflectionModule = CloneModuleForReflection(pModule->GetModule());
    hlsl::StripAndCreateReflectionStream(reflectionModule.get(), &reflectPartSizeInBytes, &pReflectionBitcodeStream);
  }

  if (pReflectionStreamOut) {
    DxilPartHeader partSTAT;
    partSTAT.PartFourCC = DFCC_ShaderStatistics;
    partSTAT.PartSize = reflectPartSizeInBytes;
    IFT(WriteStreamValue(pReflectionStreamOut, partSTAT));
    WriteProgramPart(pModule->GetShaderModel(), pReflectionBitcodeStream, pReflectionStreamOut);

    // If library, we need RDAT part as well.  For now, we just append it
    if (pModule->GetShaderModel()->IsLib()) {
      DxilPartHeader partRDAT;
      partRDAT.PartFourCC = DFCC_RuntimeData;
      partRDAT.PartSize = pRDATWriter->size();
      IFT(WriteStreamValue(pReflectionStreamOut, partRDAT));
      pRDATWriter->write(pReflectionStreamOut);
    }
  }

  if (Flags & SerializeDxilFlags::IncludeReflectionPart) {
    writer.AddPart(DFCC_ShaderStatistics, reflectPartSizeInBytes,
      [pModule, pReflectionBitcodeStream](AbstractMemoryStream *pStream) {
        WriteProgramPart(pModule->GetShaderModel(), pReflectionBitcodeStream, pStream);
      });
  }

  if (Flags & SerializeDxilFlags::StripReflectionFromDxilPart) {
    bModuleStripped |= pModule->StripReflection();
  }

  // If debug info or reflection was stripped, re-serialize the module.
  if (bModuleStripped) {
    pProgramStream.Release();
    IFT(CreateMemoryStream(DxcGetThreadMallocNoRef(), &pProgramStream));
    raw_stream_ostream outStream(pProgramStream.p);
    WriteBitcodeToFile(pModule->GetModule(), outStream, false);
  }

  // Compute hash if needed.
  DxilShaderHash HashContent;
  SmallString<32> HashStr;
  if (bSupportsShaderHash || pShaderHashOut ||
      (Flags & SerializeDxilFlags::IncludeDebugNamePart &&
        DebugName.empty()))
  {
    // If the debug name should be specific to the sources, base the name on the debug
    // bitcode, which will include the source references, line numbers, etc. Otherwise,
    // do it exclusively on the target shader bitcode.
    llvm::MD5 md5;
    if (Flags & SerializeDxilFlags::DebugNameDependOnSource) {
      md5.update(ArrayRef<uint8_t>(pModuleBitcode->GetPtr(), pModuleBitcode->GetPtrSize()));
      HashContent.Flags = (uint32_t)DxilShaderHashFlags::IncludesSource;
    } else {
      md5.update(ArrayRef<uint8_t>(pProgramStream->GetPtr(), pProgramStream->GetPtrSize()));
      HashContent.Flags = (uint32_t)DxilShaderHashFlags::None;
    }
    md5.final(HashContent.Digest);
    md5.stringifyResult(HashContent.Digest, HashStr);
  }

  // Serialize debug name if requested.
  std::string DebugNameStr; // Used if constructing name based on hash
  if (Flags & SerializeDxilFlags::IncludeDebugNamePart) {
    if (DebugName.empty()) {
      DebugNameStr += HashStr;
      DebugNameStr += ".pdb";
      DebugName = DebugNameStr;
    }

    // Calculate the size of the blob part.
    const uint32_t DebugInfoContentLen = PSVALIGN4(
        sizeof(DxilShaderDebugName) + DebugName.size() + 1); // 1 for null

    writer.AddPart(DFCC_ShaderDebugName, DebugInfoContentLen,
      [DebugName]
      (AbstractMemoryStream *pStream)
    {
      DxilShaderDebugName NameContent;
      NameContent.Flags = 0;
      NameContent.NameLength = DebugName.size();
      IFT(WriteStreamValue(pStream, NameContent));

      ULONG cbWritten;
      IFT(pStream->Write(DebugName.begin(), DebugName.size(), &cbWritten));
      const char Pad[] = { '\0','\0','\0','\0' };
      // Always writes at least one null to align size
      unsigned padLen = (4 - ((sizeof(DxilShaderDebugName) + cbWritten) & 0x3));
      IFT(pStream->Write(Pad, padLen, &cbWritten));
    });
  }

  // Add hash to container if supported by validator version.
  if (bSupportsShaderHash) {
    writer.AddPart(DFCC_ShaderHash, sizeof(HashContent),
      [HashContent]
      (AbstractMemoryStream *pStream)
    {
      IFT(WriteStreamValue(pStream, HashContent));
    });
  }

  // Write hash to separate output if requested.
  if (pShaderHashOut) {
    memcpy(pShaderHashOut, &HashContent, sizeof(DxilShaderHash));
  }

  // Compute padded bitcode size.
  uint32_t programInUInt32, programPaddingBytes;
  GetPaddedProgramPartSize(pProgramStream, programInUInt32, programPaddingBytes);

  // Write the program part.
  writer.AddPart(DFCC_DXIL, programInUInt32 * sizeof(uint32_t) + sizeof(DxilProgramHeader), [&](AbstractMemoryStream *pStream) {
    WriteProgramPart(pModule->GetShaderModel(), pProgramStream, pStream);
  });

  // Private data part should be added last when assembling the container becasue there is no garuntee of aligned size
  if (pPrivateData) {
    writer.AddPart(
        hlsl::DFCC_PrivateData, PrivateDataSize,
        [&](AbstractMemoryStream *pStream) {
          ULONG cbWritten;
          IFT(pStream->Write(pPrivateData, PrivateDataSize, &cbWritten));
        });
  }

  writer.write(pFinalStream);
}

void hlsl::SerializeDxilContainerForRootSignature(hlsl::RootSignatureHandle *pRootSigHandle,
                                     AbstractMemoryStream *pFinalStream) {
  DXASSERT_NOMSG(pRootSigHandle != nullptr);
  DXASSERT_NOMSG(pFinalStream != nullptr);
  // Root signature container should never be unaligned.
  DxilContainerWriter_impl writer(false);
  // Write the root signature (RTS0) part.
  DxilProgramRootSignatureWriter rootSigWriter(*pRootSigHandle);
  if (!pRootSigHandle->IsEmpty()) {
    writer.AddPart(
        DFCC_RootSignature, rootSigWriter.size(),
        [&](AbstractMemoryStream *pStream) { rootSigWriter.write(pStream); });
  }
  writer.write(pFinalStream);
}
