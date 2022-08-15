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

#include "DxilRootSignatureHelper.h"

using namespace llvm;
using std::string;

namespace hlsl {

//////////////////////////////////////////////////////////////////////////////
// Root signature handler.

RootSignatureHandle::RootSignatureHandle(RootSignatureHandle&& other) {
  m_pDesc = nullptr;
  m_pSerialized = nullptr;
  std::swap(m_pDesc, other.m_pDesc);
  std::swap(m_pSerialized, other.m_pSerialized);
}

void RootSignatureHandle::Assign(const DxilVersionedRootSignatureDesc *pDesc,
                                 IDxcBlob *pSerialized) {
  Clear();
  m_pDesc = pDesc;
  m_pSerialized = pSerialized;
  if (m_pSerialized)
    m_pSerialized->AddRef();
}

void RootSignatureHandle::Clear() {
  hlsl::DeleteRootSignature(m_pDesc);
  m_pDesc = nullptr;
  if (m_pSerialized != nullptr) {
    m_pSerialized->Release();
    m_pSerialized = nullptr;
  }
}

const uint8_t *RootSignatureHandle::GetSerializedBytes() const {
  DXASSERT_NOMSG(m_pSerialized != nullptr);
  return (uint8_t *)m_pSerialized->GetBufferPointer();
}

unsigned RootSignatureHandle::GetSerializedSize() const {
  DXASSERT_NOMSG(m_pSerialized != nullptr);
  return m_pSerialized->GetBufferSize();
}

void RootSignatureHandle::EnsureSerializedAvailable() {
  DXASSERT_NOMSG(!IsEmpty());
  if (m_pSerialized == nullptr) {
    CComPtr<IDxcBlob> pResult;
    hlsl::SerializeRootSignature(m_pDesc, &pResult, nullptr, false);
    IFTBOOL(pResult != nullptr, E_FAIL);
    m_pSerialized = pResult.Detach();
  }
}

void RootSignatureHandle::Deserialize() {
  DXASSERT_NOMSG(m_pSerialized && !m_pDesc);
  DeserializeRootSignature((uint8_t*)m_pSerialized->GetBufferPointer(), (uint32_t)m_pSerialized->GetBufferSize(), &m_pDesc);
}

void RootSignatureHandle::LoadSerialized(const uint8_t *pData,
                                         unsigned length) {
  DXASSERT_NOMSG(IsEmpty());
  IDxcBlob *pCreated;
  IFT(DxcCreateBlobOnHeapCopy(pData, length, &pCreated));
  m_pSerialized = pCreated;
}

//////////////////////////////////////////////////////////////////////////////

namespace root_sig_helper {
// GetFlags/SetFlags overloads.
DxilRootDescriptorFlags GetFlags(const DxilRootDescriptor &) {
  // Upconvert root parameter flags to be volatile.
  return DxilRootDescriptorFlags::DataVolatile;
}
void SetFlags(DxilRootDescriptor &, DxilRootDescriptorFlags) {
  // Drop the flags; none existed in rs_1_0.
}
DxilRootDescriptorFlags GetFlags(const DxilRootDescriptor1 &D) {
  return D.Flags;
}
void SetFlags(DxilRootDescriptor1 &D, DxilRootDescriptorFlags Flags) {
  D.Flags = Flags;
}
void SetFlags(DxilContainerRootDescriptor1 &D, DxilRootDescriptorFlags Flags) {
  D.Flags = (uint32_t)Flags;
}
DxilDescriptorRangeFlags GetFlags(const DxilDescriptorRange &D) {
  // Upconvert range flags to be volatile.
  DxilDescriptorRangeFlags Flags =
      DxilDescriptorRangeFlags::DescriptorsVolatile;

  // Sampler does not have data.
  if (D.RangeType != DxilDescriptorRangeType::Sampler)
    Flags = (DxilDescriptorRangeFlags)(
        (unsigned)Flags | (unsigned)DxilDescriptorRangeFlags::DataVolatile);

  return Flags;
}
void SetFlags(DxilDescriptorRange &, DxilDescriptorRangeFlags) {}
DxilDescriptorRangeFlags GetFlags(const DxilContainerDescriptorRange &D) {
  // Upconvert range flags to be volatile.
  DxilDescriptorRangeFlags Flags =
      DxilDescriptorRangeFlags::DescriptorsVolatile;

  // Sampler does not have data.
  if (D.RangeType != (uint32_t)DxilDescriptorRangeType::Sampler)
    Flags |= DxilDescriptorRangeFlags::DataVolatile;

  return Flags;
}
void SetFlags(DxilContainerDescriptorRange &, DxilDescriptorRangeFlags) {}
DxilDescriptorRangeFlags GetFlags(const DxilDescriptorRange1 &D) {
  return D.Flags;
}
void SetFlags(DxilDescriptorRange1 &D, DxilDescriptorRangeFlags Flags) {
  D.Flags = Flags;
}
DxilDescriptorRangeFlags GetFlags(const DxilContainerDescriptorRange1 &D) {
  return (DxilDescriptorRangeFlags)D.Flags;
}
void SetFlags(DxilContainerDescriptorRange1 &D,
              DxilDescriptorRangeFlags Flags) {
  D.Flags = (uint32_t)Flags;
}

} // namespace root_sig_helper

//////////////////////////////////////////////////////////////////////////////

template <typename T>
void DeleteRootSignatureTemplate(const T &RS) {
  for (unsigned i = 0; i < RS.NumParameters; i++) {
    const auto &P = RS.pParameters[i];
    if (P.ParameterType == DxilRootParameterType::DescriptorTable) {
      delete[] P.DescriptorTable.pDescriptorRanges;
    }
  }

  delete[] RS.pParameters;
  delete[] RS.pStaticSamplers;
}

void DeleteRootSignature(const DxilVersionedRootSignatureDesc * pRootSignature)
{
  if (pRootSignature == nullptr)
    return;

  switch (pRootSignature->Version)
  {
  case DxilRootSignatureVersion::Version_1_0:
    DeleteRootSignatureTemplate<DxilRootSignatureDesc>(pRootSignature->Desc_1_0);
    break;
  case DxilRootSignatureVersion::Version_1_1:
  default:
    DXASSERT(pRootSignature->Version == DxilRootSignatureVersion::Version_1_1, "else version is incorrect");
    DeleteRootSignatureTemplate<DxilRootSignatureDesc1>(pRootSignature->Desc_1_1);
    break;
  }

  delete pRootSignature;
}

namespace {
// Dump root sig.

void printRootSigFlags(DxilRootSignatureFlags Flags, raw_ostream &os) {
  if (Flags == DxilRootSignatureFlags::None)
    return;
  unsigned UFlags = (unsigned)Flags;

  std::pair<unsigned, std::string> FlagTable[] = {
      {unsigned(DxilRootSignatureFlags::AllowInputAssemblerInputLayout),
       "ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT"},
      {unsigned(DxilRootSignatureFlags::DenyVertexShaderRootAccess),
       "DenyVertexShaderRootAccess"},
      {unsigned(DxilRootSignatureFlags::DenyHullShaderRootAccess),
       "DenyHullShaderRootAccess"},
      {unsigned(DxilRootSignatureFlags::DenyDomainShaderRootAccess),
       "DenyDomainShaderRootAccess"},
      {unsigned(DxilRootSignatureFlags::DenyGeometryShaderRootAccess),
       "DenyGeometryShaderRootAccess"},
      {unsigned(DxilRootSignatureFlags::DenyPixelShaderRootAccess),
       "DenyPixelShaderRootAccess"},
      {unsigned(DxilRootSignatureFlags::AllowStreamOutput),
       "AllowStreamOutput"},
      {unsigned(DxilRootSignatureFlags::LocalRootSignature),
       "LocalRootSignature"},
      {unsigned(DxilRootSignatureFlags::DenyAmplificationShaderRootAccess),
       "DenyAmplificationShaderRootAccess"},
      {unsigned(DxilRootSignatureFlags::DenyMeshShaderRootAccess),
       "DenyMeshShaderRootAccess"},
      {unsigned(DxilRootSignatureFlags::CBVSRVUAVHeapDirectlyIndexed),
       "CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED"},
      {unsigned(DxilRootSignatureFlags::SamplerHeapDirectlyIndexed),
       "SAMPLER_HEAP_DIRECTLY_INDEXED"},
      {unsigned(DxilRootSignatureFlags::AllowLowTierReservedHwCbLimit),
       "AllowLowTierReservedHwCbLimit"},
  };
  os << "RootFlags(";
  SmallVector<std::string, 4> FlagStrs;
  for (auto &f : FlagTable) {
    if (UFlags & f.first)
      FlagStrs.emplace_back(f.second);
  }
  auto it = FlagStrs.begin();
  os << *(it++);
  for (; it != FlagStrs.end(); it++) {
    os << "|" << *it;
  }

  os << "),";
}

void printDesc(unsigned Reg, unsigned Space, unsigned Size, raw_ostream &os) {
  os << Reg;
  if (Space)
    os << ", space=" << Space;
  if (Size && Size != 1)
    os << ", numDescriptors =" << Size;
}

void printDescType(DxilDescriptorRangeType Ty, raw_ostream &os) {
  switch (Ty) {
  case DxilDescriptorRangeType::CBV: {
    os << "CBV(b";
  } break;
  case DxilDescriptorRangeType::Sampler: {
    os << "Sampler(s";
  } break;
  case DxilDescriptorRangeType::UAV: {
    os << "UAV(u";
  } break;
  case DxilDescriptorRangeType::SRV: {
    os << "SRV(t";
  } break;
  }
}

template <typename RangeTy> void printDescRange(RangeTy &R, raw_ostream &os) {
  printDescType(R.RangeType, os);
  printDesc(R.BaseShaderRegister, R.RegisterSpace, R.NumDescriptors, os);
  os << ")";
}

template <typename TableTy> void printDescTable(TableTy &Tab, raw_ostream &os) {
  for (unsigned i = 0; i < Tab.NumDescriptorRanges; i++) {
    auto *pRange = Tab.pDescriptorRanges + i;
    printDescRange(*pRange, os);
    os << ",";
  }
}

void printVisibility(DxilShaderVisibility v, raw_ostream &os) {
  switch (v) {
  default:
    break;
  case DxilShaderVisibility::Amplification:
    os << ",visibility=SHADER_VISIBILITY_AMPLIFICATION";
    break;
  case DxilShaderVisibility::Domain:
    os << ",visibility=SHADER_VISIBILITY_DOMAIN";
    break;
  case DxilShaderVisibility::Geometry:
    os << ",visibility=SHADER_VISIBILITY_GEOMETRY";
    break;
  case DxilShaderVisibility::Hull:
    os << ",visibility=SHADER_VISIBILITY_HULL";
    break;
  case DxilShaderVisibility::Mesh:
    os << ",visibility=SHADER_VISIBILITY_MESH";
    break;
  case DxilShaderVisibility::Pixel:
    os << ",visibility=SHADER_VISIBILITY_PIXEL";
    break;
  case DxilShaderVisibility::Vertex:
    os << ",visibility=SHADER_VISIBILITY_VERTEX";
    break;
  }
}

template <typename ParamTy>
void printRootParam(ParamTy &Param, raw_ostream &os) {
  switch (Param.ParameterType) {
  case DxilRootParameterType::CBV:
    printDescType(DxilDescriptorRangeType::CBV, os);
    printDesc(Param.Descriptor.ShaderRegister, Param.Descriptor.RegisterSpace, 0,
             os);

    break;
  case DxilRootParameterType::SRV:
    printDescType(DxilDescriptorRangeType::SRV, os);
    printDesc(Param.Descriptor.ShaderRegister, Param.Descriptor.RegisterSpace, 0,
             os);
    break;
  case DxilRootParameterType::UAV:
    printDescType(DxilDescriptorRangeType::UAV, os);
    printDesc(Param.Descriptor.ShaderRegister, Param.Descriptor.RegisterSpace, 0,
             os);
    break;
  case DxilRootParameterType::Constants32Bit:
    os << "RootConstants(num32BitConstants=" << Param.Constants.Num32BitValues
       << "b";
    printDesc(Param.Constants.ShaderRegister, Param.Constants.RegisterSpace, 0,
             os);
    break;
  case DxilRootParameterType::DescriptorTable:
    os << "DescriptorTable(";
    printDescTable(Param.DescriptorTable, os);
    break;
  }

  printVisibility(Param.ShaderVisibility, os);
  os << ")";
}

void printSampler(DxilStaticSamplerDesc &Sampler, raw_ostream &os) {
  // StaticSampler(s4, filter=FILTER_MIN_MAG_MIP_LINEAR)
  os << "StaticSampler(s" << Sampler.ShaderRegister
     << ", space=" << Sampler.RegisterSpace;
  // TODO: set the fileds.
  printVisibility(Sampler.ShaderVisibility, os);
  os << ")";
}

template <typename DescTy> void printRootSig(DescTy &RS, raw_ostream &os) {
  printRootSigFlags(RS.Flags, os);
  for (unsigned i = 0; i < RS.NumParameters; i++) {
    auto *pParam = RS.pParameters + i;
    printRootParam(*pParam, os);
    os << ",";
  }
  for (unsigned i = 0; i < RS.NumStaticSamplers; i++) {
    auto *pSampler = RS.pStaticSamplers + i;
    printSampler(*pSampler, os);
    os << ",";
  }
}
} // namespace

void printRootSignature(const DxilVersionedRootSignatureDesc &RS, raw_ostream &os) {
  switch (RS.Version) {
  case DxilRootSignatureVersion::Version_1_0:
    printRootSig(RS.Desc_1_0, os);
    break;
  case DxilRootSignatureVersion::Version_1_1:
  default:
    DXASSERT(RS.Version == DxilRootSignatureVersion::Version_1_1,
             "else version is incorrect");
    printRootSig(RS.Desc_1_1, os);
    break;
  }
  os.flush();
}


} // namespace hlsl
