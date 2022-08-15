#pragma once

#define FallbackLayerRegisterSpace 214743647

// SRVs
#define FallbackLayerHitGroupRecordByteAddressBufferRegister 0
#define FallbackLayerMissShaderRecordByteAddressBufferRegister 1
#define FallbackLayerRayGenShaderRecordByteAddressBufferRegister 2
#define FallbackLayerCallableShaderRecordByteAddressBufferRegister 3

// SRV & UAV
#define FallbackLayerDescriptorHeapTable 0

// There's a driver issue on some hardware that has issues
// starting a bindless table on any register but 0, so
// make sure each bindless table has it's own register space
#define FallbackLayerDescriptorHeapSpaceOffset 1
#define FallbackLayerNumDescriptorHeapSpacesPerView 10

// CBVs
#define FallbackLayerDispatchConstantsRegister 0
#define FallbackLayerAccelerationStructureList 1

#ifndef HLSL
struct ViewKey {
  unsigned int ViewType;
  union
  {
    unsigned int StructuredStride; // When ViewType == StructuredBuffer
    unsigned int SRVComponentType; // When ViewType != StructuredBuffer &&  ViewType != RawBuffer
  };
};

struct ShaderInfo {
  const wchar_t *ExportName;
  unsigned int SamplerDescriptorSizeInBytes;
  unsigned int SrvCbvUavDescriptorSizeInBytes;
  unsigned int ShaderRecordIdentifierSizeInBytes;
  const void *pRootSignatureDesc;

  ViewKey *pSRVRegisterSpaceArray;
  unsigned int *pNumSRVSpaces;

  ViewKey *pUAVRegisterSpaceArray;
  unsigned int *pNumUAVSpaces;
};

struct DispatchRaysConstants {
  uint32_t RayDispatchDimensionsWidth;
  uint32_t RayDispatchDimensionsHeight;
  uint32_t HitGroupShaderRecordStride;
  uint32_t MissShaderRecordStride;

  // 64-bit values
  uint64_t SamplerDescriptorHeapStart;
  uint64_t SrvCbvUavDescriptorHeapStart;
};

enum DescriptorRangeTypes { SRV = 0, CBV, UAV, Sampler, NumRangeTypes };

enum RootSignatureParameterOffset {
  HitGroupRecord = 0,
  MissShaderRecord,
  RayGenShaderRecord,
  CallableShaderRecord,
  DispatchConstants,
  CbvSrvUavDescriptorHeapAliasedTables,
  SamplerDescriptorHeapAliasedTables,
  AccelerationStructuresList,
#if ENABLE_UAV_LOG
  DebugUAVLog,
#endif
  NumParameters
};
#endif
