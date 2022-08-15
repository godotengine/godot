///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilRootSignature.h                                                       //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// HLSL root signature parsing.                                              //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#ifndef __DXC_ROOTSIGNATURE__
#define __DXC_ROOTSIGNATURE__

#include <stdint.h>

#include "dxc/Support/WinAdapter.h"

struct IDxcBlob;
struct IDxcBlobEncoding;

namespace llvm {
  class raw_ostream;
}

namespace hlsl {

// Forward declarations.
struct DxilDescriptorRange;
struct DxilDescriptorRange1;
struct DxilRootConstants;
struct DxilRootDescriptor;
struct DxilRootDescriptor1;
struct DxilRootDescriptorTable;
struct DxilRootDescriptorTable1;
struct DxilRootParameter;
struct DxilRootParameter1;
struct DxilRootSignatureDesc;
struct DxilRootSignatureDesc1;
struct DxilStaticSamplerDesc;
struct DxilVersionedRootSignatureDesc;

// Constant values.
static const uint32_t DxilDescriptorRangeOffsetAppend = 0xffffffff;
static const uint32_t DxilSystemReservedRegisterSpaceValuesStart = 0xfffffff0;
static const uint32_t DxilSystemReservedRegisterSpaceValuesEnd = 0xffffffff;
#define DxilMipLodBiaxMax ( 15.99f )
#define DxilMipLodBiaxMin ( -16.0f )
#define DxilFloat32Max ( 3.402823466e+38f )
static const uint32_t DxilMipLodFractionalBitCount = 8;
static const uint32_t DxilMapAnisotropy = 16;

// Enumerations and flags.
enum class DxilComparisonFunc : unsigned{
  Never = 1,
  Less = 2,
  Equal = 3,
  LessEqual = 4,
  Greater = 5,
  NotEqual = 6,
  GreaterEqual = 7,
  Always = 8
};
enum class DxilDescriptorRangeFlags : unsigned {
  None = 0,
  DescriptorsVolatile = 0x1,
  DataVolatile = 0x2,
  DataStaticWhileSetAtExecute = 0x4,
  DataStatic = 0x8,
  DescriptorsStaticKeepingBufferBoundsChecks = 0x10000,
  ValidFlags = 0x1000f,
  ValidSamplerFlags = DescriptorsVolatile
};
enum class DxilDescriptorRangeType : unsigned {
  SRV = 0,
  UAV = 1,
  CBV = 2,
  Sampler = 3,
  MaxValue = 3
};
enum class DxilRootDescriptorFlags : unsigned {
  None = 0,
  DataVolatile = 0x2,
  DataStaticWhileSetAtExecute = 0x4,
  DataStatic = 0x8,
  ValidFlags = 0xe
};
enum class DxilRootSignatureVersion {
  Version_1 = 1,
  Version_1_0 = 1,
  Version_1_1 = 2
};
enum class DxilRootSignatureCompilationFlags {
  None = 0x0,
  LocalRootSignature = 0x1,
  GlobalRootSignature = 0x2,
};
enum class DxilRootSignatureFlags : uint32_t {
  None = 0,
  AllowInputAssemblerInputLayout = 0x1,
  DenyVertexShaderRootAccess = 0x2,
  DenyHullShaderRootAccess = 0x4,
  DenyDomainShaderRootAccess = 0x8,
  DenyGeometryShaderRootAccess = 0x10,
  DenyPixelShaderRootAccess = 0x20,
  AllowStreamOutput = 0x40,
  LocalRootSignature = 0x80,
  DenyAmplificationShaderRootAccess = 0x100,
  DenyMeshShaderRootAccess = 0x200,
  CBVSRVUAVHeapDirectlyIndexed = 0x400,
  SamplerHeapDirectlyIndexed = 0x800,
  AllowLowTierReservedHwCbLimit = 0x80000000,
  ValidFlags = 0x80000fff
};
enum class DxilRootParameterType {
  DescriptorTable = 0,
  Constants32Bit = 1,
  CBV = 2,
  SRV = 3,
  UAV = 4,
  MaxValue = 4
};
enum class DxilFilter {
  // TODO: make these names consistent with code convention
  MIN_MAG_MIP_POINT = 0,
  MIN_MAG_POINT_MIP_LINEAR = 0x1,
  MIN_POINT_MAG_LINEAR_MIP_POINT = 0x4,
  MIN_POINT_MAG_MIP_LINEAR = 0x5,
  MIN_LINEAR_MAG_MIP_POINT = 0x10,
  MIN_LINEAR_MAG_POINT_MIP_LINEAR = 0x11,
  MIN_MAG_LINEAR_MIP_POINT = 0x14,
  MIN_MAG_MIP_LINEAR = 0x15,
  ANISOTROPIC = 0x55,
  COMPARISON_MIN_MAG_MIP_POINT = 0x80,
  COMPARISON_MIN_MAG_POINT_MIP_LINEAR = 0x81,
  COMPARISON_MIN_POINT_MAG_LINEAR_MIP_POINT = 0x84,
  COMPARISON_MIN_POINT_MAG_MIP_LINEAR = 0x85,
  COMPARISON_MIN_LINEAR_MAG_MIP_POINT = 0x90,
  COMPARISON_MIN_LINEAR_MAG_POINT_MIP_LINEAR = 0x91,
  COMPARISON_MIN_MAG_LINEAR_MIP_POINT = 0x94,
  COMPARISON_MIN_MAG_MIP_LINEAR = 0x95,
  COMPARISON_ANISOTROPIC = 0xd5,
  MINIMUM_MIN_MAG_MIP_POINT = 0x100,
  MINIMUM_MIN_MAG_POINT_MIP_LINEAR = 0x101,
  MINIMUM_MIN_POINT_MAG_LINEAR_MIP_POINT = 0x104,
  MINIMUM_MIN_POINT_MAG_MIP_LINEAR = 0x105,
  MINIMUM_MIN_LINEAR_MAG_MIP_POINT = 0x110,
  MINIMUM_MIN_LINEAR_MAG_POINT_MIP_LINEAR = 0x111,
  MINIMUM_MIN_MAG_LINEAR_MIP_POINT = 0x114,
  MINIMUM_MIN_MAG_MIP_LINEAR = 0x115,
  MINIMUM_ANISOTROPIC = 0x155,
  MAXIMUM_MIN_MAG_MIP_POINT = 0x180,
  MAXIMUM_MIN_MAG_POINT_MIP_LINEAR = 0x181,
  MAXIMUM_MIN_POINT_MAG_LINEAR_MIP_POINT = 0x184,
  MAXIMUM_MIN_POINT_MAG_MIP_LINEAR = 0x185,
  MAXIMUM_MIN_LINEAR_MAG_MIP_POINT = 0x190,
  MAXIMUM_MIN_LINEAR_MAG_POINT_MIP_LINEAR = 0x191,
  MAXIMUM_MIN_MAG_LINEAR_MIP_POINT = 0x194,
  MAXIMUM_MIN_MAG_MIP_LINEAR = 0x195,
  MAXIMUM_ANISOTROPIC = 0x1d5
};
enum class DxilShaderVisibility {
  All = 0,
  Vertex = 1,
  Hull = 2,
  Domain = 3,
  Geometry = 4,
  Pixel = 5,
  Amplification = 6,
  Mesh = 7,
  MaxValue = 7
};
enum class DxilStaticBorderColor {
  TransparentBlack = 0,
  OpaqueBlack = 1,
  OpaqueWhite = 2,
  OpaqueBlackUint = 3,
  OpaqueWhiteUint = 4
};
enum class DxilTextureAddressMode {
  Wrap = 1,
  Mirror = 2,
  Clamp = 3,
  Border = 4,
  MirrorOnce = 5
};

// Structure definitions for serialized structures.
#pragma pack(push, 1)
struct DxilContainerRootDescriptor1
{
  uint32_t ShaderRegister;
  uint32_t RegisterSpace;
  uint32_t Flags;
};
struct DxilContainerDescriptorRange
{
  uint32_t RangeType;
  uint32_t NumDescriptors;
  uint32_t BaseShaderRegister;
  uint32_t RegisterSpace;
  uint32_t OffsetInDescriptorsFromTableStart;
};
struct DxilContainerDescriptorRange1
{
  uint32_t RangeType;
  uint32_t NumDescriptors;
  uint32_t BaseShaderRegister;
  uint32_t RegisterSpace;
  uint32_t Flags;
  uint32_t OffsetInDescriptorsFromTableStart;
};
struct DxilContainerRootDescriptorTable
{
  uint32_t NumDescriptorRanges;
  uint32_t DescriptorRangesOffset;
};
struct DxilContainerRootParameter
{
  uint32_t ParameterType;
  uint32_t ShaderVisibility;
  uint32_t PayloadOffset;
};
struct DxilContainerRootSignatureDesc
{
  uint32_t Version;
  uint32_t NumParameters;
  uint32_t RootParametersOffset;
  uint32_t NumStaticSamplers;
  uint32_t StaticSamplersOffset;
  uint32_t Flags;
};
#pragma pack(pop)

// Structure definitions for in-memory structures.
struct DxilDescriptorRange {
  DxilDescriptorRangeType RangeType;
  uint32_t NumDescriptors;
  uint32_t BaseShaderRegister;
  uint32_t RegisterSpace;
  uint32_t OffsetInDescriptorsFromTableStart;
};
struct DxilRootDescriptorTable {
  uint32_t NumDescriptorRanges;
  _Field_size_full_(NumDescriptorRanges)  DxilDescriptorRange *pDescriptorRanges;
};
struct DxilRootConstants {
  uint32_t ShaderRegister;
  uint32_t RegisterSpace;
  uint32_t Num32BitValues;
};
struct DxilRootDescriptor {
  uint32_t ShaderRegister;
  uint32_t RegisterSpace;
};
struct DxilRootDescriptor1 {
  uint32_t ShaderRegister;
  uint32_t RegisterSpace;
  DxilRootDescriptorFlags Flags;
};
struct DxilRootParameter {
  DxilRootParameterType ParameterType;
  union {
    DxilRootDescriptorTable DescriptorTable;
    DxilRootConstants Constants;
    DxilRootDescriptor Descriptor;
  };
  DxilShaderVisibility ShaderVisibility;
};
struct DxilDescriptorRange1 {
  DxilDescriptorRangeType RangeType;
  uint32_t NumDescriptors;
  uint32_t BaseShaderRegister;
  uint32_t RegisterSpace;
  DxilDescriptorRangeFlags Flags;
  uint32_t OffsetInDescriptorsFromTableStart;
};
struct DxilRootDescriptorTable1 {
  uint32_t NumDescriptorRanges;
  _Field_size_full_(NumDescriptorRanges)  DxilDescriptorRange1 *pDescriptorRanges;
};
struct DxilRootParameter1 {
  DxilRootParameterType ParameterType;
  union {
    DxilRootDescriptorTable1 DescriptorTable;
    DxilRootConstants Constants;
    DxilRootDescriptor1 Descriptor;
  };
  DxilShaderVisibility ShaderVisibility;
};
struct DxilRootSignatureDesc {
  uint32_t NumParameters;
  _Field_size_full_(NumParameters) DxilRootParameter *pParameters;
  uint32_t NumStaticSamplers;
  _Field_size_full_(NumStaticSamplers) DxilStaticSamplerDesc *pStaticSamplers;
  DxilRootSignatureFlags Flags;
};
struct DxilStaticSamplerDesc {
  DxilFilter Filter;
  DxilTextureAddressMode AddressU;
  DxilTextureAddressMode AddressV;
  DxilTextureAddressMode AddressW;
  float MipLODBias;
  uint32_t MaxAnisotropy;
  DxilComparisonFunc ComparisonFunc;
  DxilStaticBorderColor BorderColor;
  float MinLOD;
  float MaxLOD;
  uint32_t ShaderRegister;
  uint32_t RegisterSpace;
  DxilShaderVisibility ShaderVisibility;
};
struct DxilRootSignatureDesc1 {
  uint32_t NumParameters;
  _Field_size_full_(NumParameters) DxilRootParameter1 *pParameters;
  uint32_t NumStaticSamplers;
  _Field_size_full_(NumStaticSamplers) DxilStaticSamplerDesc *pStaticSamplers;
  DxilRootSignatureFlags Flags;
};
struct DxilVersionedRootSignatureDesc {
  DxilRootSignatureVersion Version;
  union {
    DxilRootSignatureDesc Desc_1_0;
    DxilRootSignatureDesc1 Desc_1_1;
  };
};

void printRootSignature(const DxilVersionedRootSignatureDesc &RS, llvm::raw_ostream &os);

// Use this class to represent a root signature that may be in memory or serialized.
// There is just enough API surface to help callers not take a dependency on Windows headers.
class RootSignatureHandle {
private:
  const DxilVersionedRootSignatureDesc *m_pDesc;
  IDxcBlob *m_pSerialized;
public:
  RootSignatureHandle() : m_pDesc(nullptr), m_pSerialized(nullptr) {}
  RootSignatureHandle(const RootSignatureHandle &) = delete;
  RootSignatureHandle(RootSignatureHandle &&other);
  ~RootSignatureHandle() { Clear(); }

  bool IsEmpty() const {
    return m_pDesc == nullptr && m_pSerialized == nullptr;
  }
  IDxcBlob *GetSerialized() const { return m_pSerialized; }
  const uint8_t *GetSerializedBytes() const;
  unsigned GetSerializedSize() const;

  void Assign(const DxilVersionedRootSignatureDesc *pDesc, IDxcBlob *pSerialized);
  void Clear();
  void LoadSerialized(const uint8_t *pData, uint32_t length);
  void EnsureSerializedAvailable();
  void Deserialize();

  const DxilVersionedRootSignatureDesc *GetDesc() const { return m_pDesc; }
};

void DeleteRootSignature(const DxilVersionedRootSignatureDesc *pRootSignature);  

// Careful to delete: returns the original root signature, if conversion is not required.  
void ConvertRootSignature(const DxilVersionedRootSignatureDesc* pRootSignatureIn,
                          DxilRootSignatureVersion RootSignatureVersionOut,  
                          const DxilVersionedRootSignatureDesc ** ppRootSignatureOut);

void SerializeRootSignature(const DxilVersionedRootSignatureDesc *pRootSignature,
                            _Outptr_ IDxcBlob **ppBlob, _Outptr_ IDxcBlobEncoding **ppErrorBlob,
                            bool bAllowReservedRegisterSpace);

void DeserializeRootSignature(_In_reads_bytes_(SrcDataSizeInBytes) const void *pSrcData,
                              _In_ uint32_t SrcDataSizeInBytes,
                              _Out_ const DxilVersionedRootSignatureDesc **ppRootSignature);

// Takes PSV - pipeline state validation data, not shader container.
bool VerifyRootSignatureWithShaderPSV(_In_ const DxilVersionedRootSignatureDesc *pDesc,
                                      _In_ DXIL::ShaderKind ShaderKind,
                                      _In_reads_bytes_(PSVSize) const void *pPSVData,
                                      _In_ uint32_t PSVSize,
                                      _In_ llvm::raw_ostream &DiagStream);

// standalone verification
bool VerifyRootSignature(_In_ const DxilVersionedRootSignatureDesc *pDesc,
                         _In_ llvm::raw_ostream &DiagStream,
                         _In_ bool bAllowReservedRegisterSpace);

} // namespace hlsl

#endif // __DXC_ROOTSIGNATURE__
