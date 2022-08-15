///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilRootSignature.cpp                                                     //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Provides support for manipulating root signature structures.              //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/DXIL/DxilConstants.h"
#include "dxc/DxilRootSignature/DxilRootSignature.h"
#include "dxc/Support/Global.h"
#include "dxc/Support/WinIncludes.h"
#include "dxc/Support/WinFunctions.h"
#include "dxc/Support/FileIOHelper.h"
#include "dxc/dxcapi.h"

#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/DiagnosticPrinter.h"

#include <string>
#include <algorithm>
#include <utility>
#include <vector>
#include <set>
#include <ios>

#include <assert.h> // Needed for DxilPipelineStateValidation.h
#include "dxc/DxilContainer/DxilPipelineStateValidation.h"

#include "DxilRootSignatureHelper.h"

using namespace llvm;
using std::string;

namespace hlsl {

using namespace root_sig_helper;

//////////////////////////////////////////////////////////////////////////////
// Interval helper.

template <typename T>
class CIntervalCollection {
private:
  std::set<T> m_set;
public:
  const T* FindIntersectingInterval(const T &I) {
    auto it = m_set.find(I);
    if (it != m_set.end())
      return &*it;
    return nullptr;
  }
  void Insert(const T& value) {
    auto result = m_set.insert(value);
    UNREFERENCED_PARAMETER(result);
#if DBG
    DXASSERT(result.second, "otherwise interval collides with existing in collection");
#endif
  }
};

//////////////////////////////////////////////////////////////////////////////
// Verifier classes.

class DescriptorTableVerifier {
public:
  void Verify(const DxilDescriptorRange1 *pRanges, unsigned NumRanges,
              unsigned iRTS, DiagnosticPrinter &DiagPrinter);
};

class StaticSamplerVerifier {
public:
  void Verify(const DxilStaticSamplerDesc *pDesc, DiagnosticPrinter &DiagPrinter);
};

class RootSignatureVerifier {
public:
  RootSignatureVerifier();
  ~RootSignatureVerifier();

  void AllowReservedRegisterSpace(bool bAllow);

  // Call this before calling VerifyShader, as it accumulates root signature state.
  void VerifyRootSignature(const DxilVersionedRootSignatureDesc *pRootSignature,
                           DiagnosticPrinter &DiagPrinter);

  void VerifyShader(DxilShaderVisibility VisType,
                    const void *pPSVData,
                    uint32_t PSVSize,
                    DiagnosticPrinter &DiagPrinter);

  typedef enum NODE_TYPE {
    DESCRIPTOR_TABLE_ENTRY,
    ROOT_DESCRIPTOR,
    ROOT_CONSTANT,
    STATIC_SAMPLER
  } NODE_TYPE;

private:
  static const unsigned kMinVisType = (unsigned)DxilShaderVisibility::All;
  static const unsigned kMaxVisType = (unsigned)DxilShaderVisibility::MaxValue;
  static const unsigned kMinDescType = (unsigned)DxilDescriptorRangeType::SRV;
  static const unsigned kMaxDescType = (unsigned)DxilDescriptorRangeType::MaxValue;

  struct RegisterRange {
    NODE_TYPE nt;
    unsigned space;
    unsigned lb;    // inclusive lower bound
    unsigned ub;    // inclusive upper bound
    unsigned iRP;
    unsigned iDTS;
    // Sort by space, then lower bound.
    bool operator<(const RegisterRange& other) const {
      return space < other.space ||
        (space == other.space && ub < other.lb);
    }
    // Like a regular -1,0,1 comparison, but 0 indicates overlap.
    int overlap(const RegisterRange& other) const {
      if (space < other.space) return -1;
      if (space > other.space) return 1;
      if (ub < other.lb) return -1;
      if (lb > other.ub) return 1;
      return 0;
    }
    // Check containment.
    bool contains(const RegisterRange& other) const {
      return (space == other.space) && (lb <= other.lb && other.ub <= ub);
    }
  };
  typedef CIntervalCollection<RegisterRange> RegisterRanges;

  void AddRegisterRange(unsigned iRTS, NODE_TYPE nt, unsigned iDTS,
                        DxilDescriptorRangeType DescType,
                        DxilShaderVisibility VisType,
                        unsigned NumRegisters, unsigned BaseRegister,
                        unsigned RegisterSpace, DiagnosticPrinter &DiagPrinter);

  const RegisterRange *FindCoveringInterval(DxilDescriptorRangeType RangeType,
                                            DxilShaderVisibility VisType,
                                            unsigned Num,
                                            unsigned LB,
                                            unsigned Space);

  RegisterRanges &
  GetRanges(DxilShaderVisibility VisType, DxilDescriptorRangeType DescType) {
    return RangeKinds[(unsigned)VisType][(unsigned)DescType];
  }

  RegisterRanges RangeKinds[kMaxVisType + 1][kMaxDescType + 1];
  bool m_bAllowReservedRegisterSpace;
  DxilRootSignatureFlags m_RootSignatureFlags;
};

void DescriptorTableVerifier::Verify(const DxilDescriptorRange1 *pRanges,
                                     uint32_t NumRanges, uint32_t iRP,
                                     DiagnosticPrinter &DiagPrinter) {
  bool bHasSamplers = false;
  bool bHasResources = false;

  uint64_t iAppendStartSlot = 0;
  for (unsigned iDTS = 0; iDTS < NumRanges; iDTS++) {
    const DxilDescriptorRange1 *pRange = &pRanges[iDTS];

    switch (pRange->RangeType) {
    case DxilDescriptorRangeType::SRV:
    case DxilDescriptorRangeType::UAV:
    case DxilDescriptorRangeType::CBV:
      bHasResources = true;
      break;
    case DxilDescriptorRangeType::Sampler:
      bHasSamplers = true;
      break;
    default:
      static_assert(DxilDescriptorRangeType::Sampler == DxilDescriptorRangeType::MaxValue,
                    "otherwise, need to update cases here");
      EAT(DiagPrinter << "Unsupported RangeType value " << (uint32_t)pRange->RangeType
                      << " (descriptor table slot [" << iDTS << "], root parameter [" << iRP << "]).\n");
    }

    // Samplers cannot be mixed with other resources.
    if (bHasResources && bHasSamplers) {
      EAT(DiagPrinter << "Samplers cannot be mixed with other "
                      << "resource types in a descriptor table (root "
                      << "parameter [" << iRP << "]).\n");
    }

    // NumDescriptors is not 0.
    if (pRange->NumDescriptors == 0) {
      EAT(DiagPrinter << "NumDescriptors cannot be 0 (descriptor "
                      << "table slot [" << iDTS << "], root parameter [" << iRP << "]).\n");
    }

    // Range start.
    uint64_t iStartSlot = iAppendStartSlot;
    if (pRange->OffsetInDescriptorsFromTableStart != DxilDescriptorRangeOffsetAppend) {
      iStartSlot = pRange->OffsetInDescriptorsFromTableStart;
    }
    if (iStartSlot > UINT_MAX) {
      EAT(DiagPrinter << "Cannot append range with implicit lower "
                      << "bound after an unbounded range (descriptor "
                      << "table slot [" << iDTS << "], root parameter [" << iRP << "]).\n");
    }

    // Descriptor range and shader register range overlow.
    if (pRange->NumDescriptors != UINT_MAX) {
      // Bounded range.
      uint64_t ub1 = (uint64_t)pRange->BaseShaderRegister +
                     (uint64_t)pRange->NumDescriptors - 1ull;
      if (ub1 > UINT_MAX) {
        EAT(DiagPrinter << "Overflow for shader register range: "
                        << "BaseShaderRegister=" << pRange->BaseShaderRegister
                        << ", NumDescriptor=" << pRange->NumDescriptors
                        << "; (descriptor table slot [" << iDTS
                        << "], root parameter [" << iRP << "]).\n");
      }

      uint64_t ub2 = (uint64_t)iStartSlot + (uint64_t)pRange->NumDescriptors - 1ull;
      if (ub2 > UINT_MAX) {
        EAT(DiagPrinter << "Overflow for descriptor range (descriptor "
                        << "table slot [" << iDTS << "], root parameter [" << iRP << "])\n");
      }

      iAppendStartSlot = iStartSlot + (uint64_t)pRange->NumDescriptors;
    } else {
      // Unbounded range.
      iAppendStartSlot = 1ull + (uint64_t)UINT_MAX;
    }
  }
}

RootSignatureVerifier::RootSignatureVerifier() {
  m_RootSignatureFlags = DxilRootSignatureFlags::None;
  m_bAllowReservedRegisterSpace = false;
}

RootSignatureVerifier::~RootSignatureVerifier() {}

void RootSignatureVerifier::AllowReservedRegisterSpace(bool bAllow) {
  m_bAllowReservedRegisterSpace = bAllow;
}

const char* RangeTypeString(DxilDescriptorRangeType rt)
{
  static const char *RangeType[] = {"SRV", "UAV", "CBV", "SAMPLER"};
  static_assert(_countof(RangeType) == ((unsigned)DxilDescriptorRangeType::MaxValue + 1),
                "otherwise, need to update name array");
  return (rt <= DxilDescriptorRangeType::MaxValue) ? RangeType[(unsigned)rt]
                                                   : "unknown";
}

const char *VisTypeString(DxilShaderVisibility vis) {
  static const char *Vis[] = {"ALL",    "VERTEX",   "HULL",
                              "DOMAIN", "GEOMETRY", "PIXEL",
                              "AMPLIFICATION", "MESH"};
  static_assert(_countof(Vis) == ((unsigned)DxilShaderVisibility::MaxValue + 1),
                "otherwise, need to update name array");
  unsigned idx = (unsigned)vis;
  return vis <= DxilShaderVisibility::MaxValue ? Vis[idx] : "unknown";
}

static bool IsDxilShaderVisibility(DxilShaderVisibility v) {
  return v <= DxilShaderVisibility::MaxValue;
}

void RootSignatureVerifier::AddRegisterRange(unsigned iRP,
                                             NODE_TYPE nt,
                                             unsigned iDTS,
                                             DxilDescriptorRangeType DescType,
                                             DxilShaderVisibility VisType,
                                             unsigned NumRegisters,
                                             unsigned BaseRegister,
                                             unsigned RegisterSpace,
                                             DiagnosticPrinter &DiagPrinter) {
  RegisterRange interval;
  interval.space = RegisterSpace;
  interval.lb = BaseRegister;
  interval.ub = (NumRegisters != UINT_MAX) ? BaseRegister + NumRegisters - 1 : UINT_MAX;
  interval.nt = nt;
  interval.iDTS = iDTS;
  interval.iRP = iRP;

  if (!m_bAllowReservedRegisterSpace &&
       (RegisterSpace >= DxilSystemReservedRegisterSpaceValuesStart) &&
       (RegisterSpace <= DxilSystemReservedRegisterSpaceValuesEnd)) {
    if (nt == DESCRIPTOR_TABLE_ENTRY) {
      EAT(DiagPrinter << "Root parameter [" << iRP << "] descriptor table entry [" << iDTS
                      << "] specifies RegisterSpace=" << std::hex << RegisterSpace
                      << ", which is invalid since RegisterSpace values in the range "
                      << "[" << std::hex << DxilSystemReservedRegisterSpaceValuesStart
                      << "," << std::hex << DxilSystemReservedRegisterSpaceValuesEnd
                      << "] are reserved for system use.\n");
    }
    else {
      EAT(DiagPrinter << "Root parameter [" << iRP
                      << "] specifies RegisterSpace=" << std::hex << RegisterSpace
                      << ", which is invalid since RegisterSpace values in the range "
                      << "[" << std::hex << DxilSystemReservedRegisterSpaceValuesStart
                      << "," << std::hex << DxilSystemReservedRegisterSpaceValuesEnd
                      << "] are reserved for system use.\n");
    }
  }

  const RegisterRange *pNode = nullptr;
  DxilShaderVisibility NodeVis = VisType;
  if (VisType == DxilShaderVisibility::All) {
    // Check for overlap with each visibility type.
    for (unsigned iVT = kMinVisType; iVT <= kMaxVisType; iVT++) {
      pNode = GetRanges((DxilShaderVisibility)iVT, DescType).FindIntersectingInterval(interval);
      if (pNode != nullptr)
        break;
    }
  } else {
    // Check for overlap with the same visibility.
    pNode = GetRanges(VisType, DescType).FindIntersectingInterval(interval);

    // Check for overlap with ALL visibility.
    if (pNode == nullptr) {
      pNode = GetRanges(DxilShaderVisibility::All, DescType).FindIntersectingInterval(interval);
      NodeVis = DxilShaderVisibility::All;
    }
  }

  if (pNode != nullptr) {
    const int strSize = 132;
    char testString[strSize];
    char nodeString[strSize];
    switch (nt) {
    case DESCRIPTOR_TABLE_ENTRY:
      StringCchPrintfA(testString, strSize, "(root parameter [%u], visibility %s, descriptor table slot [%u])",
        iRP, VisTypeString(VisType), iDTS);
      break;
    case ROOT_DESCRIPTOR:
    case ROOT_CONSTANT:
      StringCchPrintfA(testString, strSize, "(root parameter [%u], visibility %s)",
        iRP, VisTypeString(VisType));
      break;
    case STATIC_SAMPLER:
      StringCchPrintfA(testString, strSize, "(static sampler [%u], visibility %s)",
        iRP, VisTypeString(VisType));
      break;
    default:
      DXASSERT_NOMSG(false);
      break;
    }

    switch (pNode->nt)
    {
    case DESCRIPTOR_TABLE_ENTRY:
      StringCchPrintfA(nodeString, strSize, "(root parameter[%u], visibility %s, descriptor table slot [%u])",
        pNode->iRP, VisTypeString(NodeVis), pNode->iDTS);
      break;
    case ROOT_DESCRIPTOR:
    case ROOT_CONSTANT:
      StringCchPrintfA(nodeString, strSize, "(root parameter [%u], visibility %s)",
        pNode->iRP, VisTypeString(NodeVis));
      break;
    case STATIC_SAMPLER:
      StringCchPrintfA(nodeString, strSize, "(static sampler [%u], visibility %s)",
        pNode->iRP, VisTypeString(NodeVis));
      break;
    default:
      DXASSERT_NOMSG(false);
      break;
    }
    EAT(DiagPrinter << "Shader register range of type " << RangeTypeString(DescType)
                    << " " << testString << " overlaps with another "
                    << "shader register range " << nodeString << ".\n");
  }

  // Insert node.
  GetRanges(VisType, DescType).Insert(interval);
}

const RootSignatureVerifier::RegisterRange *
RootSignatureVerifier::FindCoveringInterval(DxilDescriptorRangeType RangeType,
                                            DxilShaderVisibility VisType,
                                            unsigned Num,
                                            unsigned LB,
                                            unsigned Space) {
  RegisterRange RR;
  RR.space = Space;
  RR.lb = LB;
  RR.ub = LB + Num - 1;
  const RootSignatureVerifier::RegisterRange *pRange = GetRanges(DxilShaderVisibility::All, RangeType).FindIntersectingInterval(RR);
  if (!pRange && VisType != DxilShaderVisibility::All) {
    pRange = GetRanges(VisType, RangeType).FindIntersectingInterval(RR);
  }
  if (pRange && !pRange->contains(RR)) {
    pRange = nullptr;
  }
  return pRange;
}

static DxilDescriptorRangeType GetRangeType(DxilRootParameterType RPT) {
  switch (RPT) {
  case DxilRootParameterType::CBV: return DxilDescriptorRangeType::CBV;
  case DxilRootParameterType::SRV: return DxilDescriptorRangeType::SRV;
  case DxilRootParameterType::UAV: return DxilDescriptorRangeType::UAV;
  default:
    static_assert(DxilRootParameterType::UAV == DxilRootParameterType::MaxValue,
                  "otherwise, need to add cases here.");
    break;
  }

  DXASSERT_NOMSG(false);
  return DxilDescriptorRangeType::SRV;
}

void RootSignatureVerifier::VerifyRootSignature(
                                const DxilVersionedRootSignatureDesc *pVersionedRootSignature,
                                DiagnosticPrinter &DiagPrinter) {
  const DxilVersionedRootSignatureDesc *pUpconvertedRS = nullptr;

  // Up-convert root signature to the latest RS version.
  ConvertRootSignature(pVersionedRootSignature, DxilRootSignatureVersion::Version_1_1, &pUpconvertedRS);
  DXASSERT_NOMSG(pUpconvertedRS->Version == DxilRootSignatureVersion::Version_1_1);

  // Ensure this gets deleted as necessary.
  struct SigGuard {
    const DxilVersionedRootSignatureDesc *Orig, *Guard;
    SigGuard(const DxilVersionedRootSignatureDesc *pOrig, const DxilVersionedRootSignatureDesc *pGuard)
      : Orig(pOrig), Guard(pGuard) { }
    ~SigGuard() {
      if (Orig != Guard) {
        DeleteRootSignature(Guard);
      }
    }
  };
  SigGuard S(pVersionedRootSignature, pUpconvertedRS);

  const DxilRootSignatureDesc1 *pRootSignature = &pUpconvertedRS->Desc_1_1;

  // Flags (assume they are bits that can be combined with OR).
  if ((pRootSignature->Flags & ~DxilRootSignatureFlags::ValidFlags) != DxilRootSignatureFlags::None) {
    EAT(DiagPrinter << "Unsupported bit-flag set (root signature flags " 
                    << std::hex << (uint32_t)pRootSignature->Flags << ").\n");
  }

  m_RootSignatureFlags = pRootSignature->Flags;

  for (unsigned iRP = 0; iRP < pRootSignature->NumParameters; iRP++) {
    const DxilRootParameter1 *pSlot = &pRootSignature->pParameters[iRP];
    // Shader visibility.
    DxilShaderVisibility Visibility = pSlot->ShaderVisibility;
    if (!IsDxilShaderVisibility(Visibility)) {
      EAT(DiagPrinter << "Unsupported ShaderVisibility value " << (uint32_t)Visibility
                      << " (root parameter [" << iRP << "]).\n");
    }

    DxilRootParameterType ParameterType = pSlot->ParameterType;
    switch (ParameterType) {
    case DxilRootParameterType::DescriptorTable: {
      DescriptorTableVerifier DTV;
      DTV.Verify(pSlot->DescriptorTable.pDescriptorRanges,
                 pSlot->DescriptorTable.NumDescriptorRanges, iRP, DiagPrinter);

      for (unsigned iDTS = 0; iDTS < pSlot->DescriptorTable.NumDescriptorRanges; iDTS++) {
        const DxilDescriptorRange1 *pRange = &pSlot->DescriptorTable.pDescriptorRanges[iDTS];
        unsigned RangeFlags = (unsigned)pRange->Flags;

        // Verify range flags.
        if (RangeFlags & ~(unsigned)DxilDescriptorRangeFlags::ValidFlags) {
          EAT(DiagPrinter << "Unsupported bit-flag set (descriptor range flags "
                          << (uint32_t)pRange->Flags << ").\n");
        }
        switch (pRange->RangeType) {
        case DxilDescriptorRangeType::Sampler: {
          if (RangeFlags & (unsigned)(DxilDescriptorRangeFlags::DataVolatile |
                                      DxilDescriptorRangeFlags::DataStatic |
                                      DxilDescriptorRangeFlags::DataStaticWhileSetAtExecute)) {
            EAT(DiagPrinter << "Sampler descriptor ranges can't specify DATA_* flags "
                            << "since there is no data pointed to by samplers "
                            << "(descriptor range flags " << (uint32_t)pRange->Flags << ").\n");
          }
          break;
        }
        default: {
          unsigned NumDataFlags = 0;
          if (RangeFlags & (unsigned)DxilDescriptorRangeFlags::DataVolatile) { NumDataFlags++; }
          if (RangeFlags & (unsigned)DxilDescriptorRangeFlags::DataStatic) { NumDataFlags++; }
          if (RangeFlags & (unsigned)DxilDescriptorRangeFlags::DataStaticWhileSetAtExecute) { NumDataFlags++; }
          if (NumDataFlags > 1) {
            EAT(DiagPrinter << "Descriptor range flags cannot specify more than one DATA_* flag "
                            << "at a time (descriptor range flags " << (uint32_t)pRange->Flags << ").\n");
          }
          if ((RangeFlags & (unsigned)DxilDescriptorRangeFlags::DataStatic) && 
              (RangeFlags & (unsigned)DxilDescriptorRangeFlags::DescriptorsVolatile)) {
            EAT(DiagPrinter << "Descriptor range flags cannot specify DESCRIPTORS_VOLATILE with the DATA_STATIC flag at the same time (descriptor range flags " << (uint32_t)pRange->Flags << "). "
                            << "DATA_STATIC_WHILE_SET_AT_EXECUTE is fine to combine with DESCRIPTORS_VOLATILE, since DESCRIPTORS_VOLATILE still requires descriptors don't change during execution. \n");
          }
          break;
        }
        }

        AddRegisterRange(iRP,
                         DESCRIPTOR_TABLE_ENTRY,
                         iDTS,
                         pRange->RangeType,
                         Visibility,
                         pRange->NumDescriptors,
                         pRange->BaseShaderRegister,
                         pRange->RegisterSpace,
                         DiagPrinter);
      }
      break;
    }

    case DxilRootParameterType::Constants32Bit:
      AddRegisterRange(iRP,
                       ROOT_CONSTANT,
                       (unsigned)-1,
                       DxilDescriptorRangeType::CBV,
                       Visibility,
                       1,
                       pSlot->Constants.ShaderRegister,
                       pSlot->Constants.RegisterSpace,
                       DiagPrinter);
      break;

    case DxilRootParameterType::CBV:
    case DxilRootParameterType::SRV:
    case DxilRootParameterType::UAV: {
      // Verify root descriptor flags.
      unsigned Flags = (unsigned)pSlot->Descriptor.Flags;
      if (Flags & ~(unsigned)DxilRootDescriptorFlags::ValidFlags) {
        EAT(DiagPrinter << "Unsupported bit-flag set (root descriptor flags " << std::hex << Flags << ").\n");
      }

      unsigned NumDataFlags = 0;
      if (Flags & (unsigned)DxilRootDescriptorFlags::DataVolatile) { NumDataFlags++; }
      if (Flags & (unsigned)DxilRootDescriptorFlags::DataStatic) { NumDataFlags++; }
      if (Flags & (unsigned)DxilRootDescriptorFlags::DataStaticWhileSetAtExecute) { NumDataFlags++; }
      if (NumDataFlags > 1) {
        EAT(DiagPrinter << "Root descriptor flags cannot specify more "
                        << "than one DATA_* flag at a time (root "
                        << "descriptor flags " << NumDataFlags << ").\n");
      }

      AddRegisterRange(iRP, ROOT_DESCRIPTOR, (unsigned)-1,
                       GetRangeType(ParameterType), Visibility, 1,
                       pSlot->Descriptor.ShaderRegister,
                       pSlot->Descriptor.RegisterSpace, DiagPrinter);
      break;
    }

    default:
      static_assert(DxilRootParameterType::UAV == DxilRootParameterType::MaxValue,
                    "otherwise, need to add cases here.");
      EAT(DiagPrinter << "Unsupported ParameterType value " << (uint32_t)ParameterType
                      << " (root parameter " << iRP << ")\n");
    }
  }

  for (unsigned iSS = 0; iSS < pRootSignature->NumStaticSamplers; iSS++) {
    const DxilStaticSamplerDesc *pSS = &pRootSignature->pStaticSamplers[iSS];
    // Shader visibility.
    DxilShaderVisibility Visibility = pSS->ShaderVisibility;
    if (!IsDxilShaderVisibility(Visibility)) {
      EAT(DiagPrinter << "Unsupported ShaderVisibility value " << (uint32_t)Visibility
                      << " (static sampler [" << iSS << "]).\n");
    }

    StaticSamplerVerifier SSV;
    SSV.Verify(pSS, DiagPrinter);
    AddRegisterRange(iSS, STATIC_SAMPLER, (unsigned)-1,
                     DxilDescriptorRangeType::Sampler, Visibility, 1,
                     pSS->ShaderRegister, pSS->RegisterSpace, DiagPrinter);
  }
}

void RootSignatureVerifier::VerifyShader(DxilShaderVisibility VisType,
                                         const void *pPSVData,
                                         uint32_t PSVSize,
                                         DiagnosticPrinter &DiagPrinter) {
  DxilPipelineStateValidation PSV;
  IFTBOOL(PSV.InitFromPSV0(pPSVData, PSVSize), E_INVALIDARG);

  bool bShaderDeniedByRootSig = false;
  switch (VisType) {
  case DxilShaderVisibility::Vertex:
    if ((m_RootSignatureFlags & DxilRootSignatureFlags::DenyVertexShaderRootAccess) != DxilRootSignatureFlags::None) {
      bShaderDeniedByRootSig = true;
    }
    break;
  case DxilShaderVisibility::Hull:
    if ((m_RootSignatureFlags & DxilRootSignatureFlags::DenyHullShaderRootAccess) != DxilRootSignatureFlags::None) {
      bShaderDeniedByRootSig = true;
    }
    break;
  case DxilShaderVisibility::Domain:
    if ((m_RootSignatureFlags & DxilRootSignatureFlags::DenyDomainShaderRootAccess) != DxilRootSignatureFlags::None) {
      bShaderDeniedByRootSig = true;
    }
    break;
  case DxilShaderVisibility::Geometry:
    if ((m_RootSignatureFlags & DxilRootSignatureFlags::DenyGeometryShaderRootAccess) != DxilRootSignatureFlags::None) {
      bShaderDeniedByRootSig = true;
    }
    break;
  case DxilShaderVisibility::Pixel:
    if ((m_RootSignatureFlags & DxilRootSignatureFlags::DenyPixelShaderRootAccess) != DxilRootSignatureFlags::None) {
      bShaderDeniedByRootSig = true;
    }
    break;
  case DxilShaderVisibility::Amplification:
    if ((m_RootSignatureFlags & DxilRootSignatureFlags::DenyAmplificationShaderRootAccess) != DxilRootSignatureFlags::None) {
      bShaderDeniedByRootSig = true;
    }
    break;
  case DxilShaderVisibility::Mesh:
    if ((m_RootSignatureFlags & DxilRootSignatureFlags::DenyMeshShaderRootAccess) != DxilRootSignatureFlags::None) {
      bShaderDeniedByRootSig = true;
    }
    break;
  default:
    break;
  }

  bool bShaderHasRootBindings = false;

  for (unsigned iResource = 0; iResource < PSV.GetBindCount(); iResource++) {
    const PSVResourceBindInfo0 *pBindInfo0 = PSV.GetPSVResourceBindInfo0(iResource);
    DXASSERT_NOMSG(pBindInfo0);

    unsigned Space = pBindInfo0->Space;
    unsigned LB = pBindInfo0->LowerBound;
    unsigned UB = pBindInfo0->UpperBound;
    unsigned Num = (UB != UINT_MAX) ? (UB - LB + 1) : 1;
    PSVResourceType ResType = (PSVResourceType)pBindInfo0->ResType;

    switch(ResType) {
    case PSVResourceType::Sampler: {
      bShaderHasRootBindings = true;
      auto pCoveringRange = FindCoveringInterval(DxilDescriptorRangeType::Sampler, VisType, Num, LB, Space);
      if(!pCoveringRange) {
        EAT(DiagPrinter << "Shader sampler descriptor range (RegisterSpace=" << Space 
                        << ", NumDescriptors=" << Num << ", BaseShaderRegister=" << LB 
                        << ") is not fully bound in root signature.\n");
      }
      break;
    }

    case PSVResourceType::SRVTyped:
    case PSVResourceType::SRVRaw:
    case PSVResourceType::SRVStructured: {
      bShaderHasRootBindings = true;
      auto pCoveringRange = FindCoveringInterval(DxilDescriptorRangeType::SRV, VisType, Num, LB, Space);
      if (pCoveringRange) {
        if(pCoveringRange->nt == ROOT_DESCRIPTOR && ResType == PSVResourceType::SRVTyped) {
          EAT(DiagPrinter << "A Shader is declaring a resource object as a texture using "
                          << "a register mapped to a root descriptor SRV (RegisterSpace=" << Space
                          << ", ShaderRegister=" << LB << ").  "
                          << "SRV or UAV root descriptors can only be Raw or Structured buffers.\n");
        }
      }
      else {
        EAT(DiagPrinter << "Shader SRV descriptor range (RegisterSpace=" << Space
                        << ", NumDescriptors=" << Num << ", BaseShaderRegister=" << LB
                        << ") is not fully bound in root signature.\n");
      }
      break;
    }

    case PSVResourceType::UAVTyped:
    case PSVResourceType::UAVRaw:
    case PSVResourceType::UAVStructured:
    case PSVResourceType::UAVStructuredWithCounter: {
      bShaderHasRootBindings = true;
      auto pCoveringRange = FindCoveringInterval(DxilDescriptorRangeType::UAV, VisType, Num, LB, Space);
      if (pCoveringRange) {
        if (pCoveringRange->nt == ROOT_DESCRIPTOR) {
          if (ResType == PSVResourceType::UAVTyped) {
            EAT(DiagPrinter << "A shader is declaring a typed UAV using a register mapped "
                            << "to a root descriptor UAV (RegisterSpace=" << Space 
                            << ", ShaderRegister=" << LB << ").  "
                            << "SRV or UAV root descriptors can only be Raw or Structured buffers.\n");
          }
          if (ResType == PSVResourceType::UAVStructuredWithCounter) {
            EAT(DiagPrinter << "A Shader is declaring a structured UAV with counter using "
                            << "a register mapped to a root descriptor UAV (RegisterSpace=" << Space
                            << ", ShaderRegister=" << LB << ").  "
                            << "SRV or UAV root descriptors can only be Raw or Structured buffers.\n");
          }
        }
      }
      else {
        EAT(DiagPrinter << "Shader UAV descriptor range (RegisterSpace=" << Space
                        << ", NumDescriptors=" << Num << ", BaseShaderRegister=" << LB
                        << ") is not fully bound in root signature.\n");
      }
      break;
    }

    case PSVResourceType::CBV: {
      bShaderHasRootBindings = true;
      auto pCoveringRange = FindCoveringInterval(DxilDescriptorRangeType::CBV, VisType, Num, LB, Space);
      if (!pCoveringRange) {
        EAT(DiagPrinter << "Shader CBV descriptor range (RegisterSpace=" << Space
                        << ", NumDescriptors=" << Num << ", BaseShaderRegister=" << LB
                        << ") is not fully bound in root signature.\n");
      }
      break;
    }

    default:
      break;
    }
  }

  if (bShaderHasRootBindings && bShaderDeniedByRootSig) {
    EAT(DiagPrinter << "Shader has root bindings but root signature uses a DENY flag "
                    << "to disallow root binding access to the shader stage.\n");
  }
}

BOOL isNaN(const float &a) {
  static const unsigned exponentMask = 0x7f800000;
  static const unsigned mantissaMask = 0x007fffff;
  unsigned u = *(const unsigned *)&a;
  return (((u & exponentMask) == exponentMask) && (u & mantissaMask)); // NaN
}

static bool IsDxilTextureAddressMode(DxilTextureAddressMode v) {
  return DxilTextureAddressMode::Wrap <= v &&
         v <= DxilTextureAddressMode::MirrorOnce;
}
static bool IsDxilComparisonFunc(DxilComparisonFunc v) {
  return DxilComparisonFunc::Never <= v && v <= DxilComparisonFunc::Always;
}

// This validation closely mirrors CCreateSamplerStateValidator's checks
void StaticSamplerVerifier::Verify(const DxilStaticSamplerDesc* pDesc,
                                   DiagnosticPrinter &DiagPrinter) {
  if (!pDesc) {
    EAT(DiagPrinter << "Static sampler: A nullptr pSamplerDesc was specified.\n");
  }

  bool bIsComparison = false;
  switch (pDesc->Filter) {
  case DxilFilter::MINIMUM_MIN_MAG_MIP_POINT:
  case DxilFilter::MINIMUM_MIN_MAG_POINT_MIP_LINEAR:
  case DxilFilter::MINIMUM_MIN_POINT_MAG_LINEAR_MIP_POINT:
  case DxilFilter::MINIMUM_MIN_POINT_MAG_MIP_LINEAR:
  case DxilFilter::MINIMUM_MIN_LINEAR_MAG_MIP_POINT:
  case DxilFilter::MINIMUM_MIN_LINEAR_MAG_POINT_MIP_LINEAR:
  case DxilFilter::MINIMUM_MIN_MAG_LINEAR_MIP_POINT:
  case DxilFilter::MINIMUM_MIN_MAG_MIP_LINEAR:
  case DxilFilter::MINIMUM_ANISOTROPIC:
  case DxilFilter::MAXIMUM_MIN_MAG_MIP_POINT:
  case DxilFilter::MAXIMUM_MIN_MAG_POINT_MIP_LINEAR:
  case DxilFilter::MAXIMUM_MIN_POINT_MAG_LINEAR_MIP_POINT:
  case DxilFilter::MAXIMUM_MIN_POINT_MAG_MIP_LINEAR:
  case DxilFilter::MAXIMUM_MIN_LINEAR_MAG_MIP_POINT:
  case DxilFilter::MAXIMUM_MIN_LINEAR_MAG_POINT_MIP_LINEAR:
  case DxilFilter::MAXIMUM_MIN_MAG_LINEAR_MIP_POINT:
  case DxilFilter::MAXIMUM_MIN_MAG_MIP_LINEAR:
  case DxilFilter::MAXIMUM_ANISOTROPIC:
    break;
  case DxilFilter::MIN_MAG_MIP_POINT:
  case DxilFilter::MIN_MAG_POINT_MIP_LINEAR:
  case DxilFilter::MIN_POINT_MAG_LINEAR_MIP_POINT:
  case DxilFilter::MIN_POINT_MAG_MIP_LINEAR:
  case DxilFilter::MIN_LINEAR_MAG_MIP_POINT:
  case DxilFilter::MIN_LINEAR_MAG_POINT_MIP_LINEAR:
  case DxilFilter::MIN_MAG_LINEAR_MIP_POINT:
  case DxilFilter::MIN_MAG_MIP_LINEAR:
  case DxilFilter::ANISOTROPIC:
    break;
  case DxilFilter::COMPARISON_MIN_MAG_MIP_POINT:
  case DxilFilter::COMPARISON_MIN_MAG_POINT_MIP_LINEAR:
  case DxilFilter::COMPARISON_MIN_POINT_MAG_LINEAR_MIP_POINT:
  case DxilFilter::COMPARISON_MIN_POINT_MAG_MIP_LINEAR:
  case DxilFilter::COMPARISON_MIN_LINEAR_MAG_MIP_POINT:
  case DxilFilter::COMPARISON_MIN_LINEAR_MAG_POINT_MIP_LINEAR:
  case DxilFilter::COMPARISON_MIN_MAG_LINEAR_MIP_POINT:
  case DxilFilter::COMPARISON_MIN_MAG_MIP_LINEAR:
  case DxilFilter::COMPARISON_ANISOTROPIC:
    bIsComparison = true;
    break;
  default:
    EAT(DiagPrinter << "Static sampler: Filter unrecognized.\n");
  }

  if (!IsDxilTextureAddressMode(pDesc->AddressU)) {
    EAT(DiagPrinter << "Static sampler: AddressU unrecognized.\n");
  }
  if (!IsDxilTextureAddressMode(pDesc->AddressV)) {
    EAT(DiagPrinter << "Static sampler: AddressV unrecognized.\n");
  }
  if (!IsDxilTextureAddressMode(pDesc->AddressW)) {
    EAT(DiagPrinter << "Static sampler: AddressW unrecognized.\n");
  }

  if (isNaN(pDesc->MipLODBias) || (pDesc->MipLODBias < DxilMipLodBiaxMin) ||
      (pDesc->MipLODBias > DxilMipLodBiaxMax)) {
    EAT(DiagPrinter << "Static sampler: MipLODBias must be in the "
                    << "range [" << DxilMipLodBiaxMin << " to " << DxilMipLodBiaxMax
                    <<"].  " << pDesc->MipLODBias << "specified.\n");
  }

  if (pDesc->MaxAnisotropy > DxilMapAnisotropy) {
    EAT(DiagPrinter << "Static sampler: MaxAnisotropy must be in "
                    << "the range [0 to " << DxilMapAnisotropy << "].  "
                    << pDesc->MaxAnisotropy << " specified.\n");
  }

  if (bIsComparison && !IsDxilComparisonFunc(pDesc->ComparisonFunc)) {
    EAT(DiagPrinter << "Static sampler: ComparisonFunc unrecognized.");
  }

  if (isNaN(pDesc->MinLOD)) {
    EAT(DiagPrinter << "Static sampler: MinLOD be in the range [-INF to +INF].  "
                    << pDesc->MinLOD << " specified.\n");
  }

  if (isNaN(pDesc->MaxLOD)) {
    EAT(DiagPrinter << "Static sampler: MaxLOD be in the range [-INF to +INF].  "
                    << pDesc->MaxLOD << " specified.\n");
  }
}

static DxilShaderVisibility GetVisibilityType(DXIL::ShaderKind ShaderKind) {
  switch(ShaderKind) {
  case DXIL::ShaderKind::Pixel:         return DxilShaderVisibility::Pixel;
  case DXIL::ShaderKind::Vertex:        return DxilShaderVisibility::Vertex;
  case DXIL::ShaderKind::Geometry:      return DxilShaderVisibility::Geometry;
  case DXIL::ShaderKind::Hull:          return DxilShaderVisibility::Hull;
  case DXIL::ShaderKind::Domain:        return DxilShaderVisibility::Domain;
  case DXIL::ShaderKind::Amplification: return DxilShaderVisibility::Amplification;
  case DXIL::ShaderKind::Mesh:          return DxilShaderVisibility::Mesh;
  default:                              return DxilShaderVisibility::All;
  }
}

_Use_decl_annotations_
bool VerifyRootSignatureWithShaderPSV(const DxilVersionedRootSignatureDesc *pDesc,
                                      DXIL::ShaderKind ShaderKind,
                                      const void *pPSVData,
                                      uint32_t PSVSize,
                                      llvm::raw_ostream &DiagStream) {
  try {
    RootSignatureVerifier RSV;
    DiagnosticPrinterRawOStream DiagPrinter(DiagStream);
    RSV.VerifyRootSignature(pDesc, DiagPrinter);
    RSV.VerifyShader(GetVisibilityType(ShaderKind), pPSVData, PSVSize, DiagPrinter);
  } catch (...) {
    return false;
  }

  return true;
}

bool VerifyRootSignature(_In_ const DxilVersionedRootSignatureDesc *pDesc,
                         _In_ llvm::raw_ostream &DiagStream,
                         _In_ bool bAllowReservedRegisterSpace) {
  try {
    RootSignatureVerifier RSV;
    RSV.AllowReservedRegisterSpace(bAllowReservedRegisterSpace);
    DiagnosticPrinterRawOStream DiagPrinter(DiagStream);
    RSV.VerifyRootSignature(pDesc, DiagPrinter);
  } catch (...) {
    return false;
  }

  return true;
}

} // namespace hlsl
