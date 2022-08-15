///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilPipelineStateValidation.h                                             //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Defines data used by the D3D runtime for PSO validation.                  //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#ifndef __DXIL_PIPELINE_STATE_VALIDATION__H__
#define __DXIL_PIPELINE_STATE_VALIDATION__H__

#include <stdint.h>
#include <string.h>

// Don't include assert.h here.
// Since this header is included from multiple environments,
// it is necessary to define assert before this header is included.
// #include <assert.h>

#ifndef UINT_MAX
#define UINT_MAX 0xffffffff
#endif
// How many dwords are required for mask with one bit per component, 4 components per vector
inline uint32_t PSVComputeMaskDwordsFromVectors(uint32_t Vectors) { return (Vectors + 7) >> 3; }
inline uint32_t PSVComputeInputOutputTableDwords(uint32_t InputVectors, uint32_t OutputVectors) {
  return PSVComputeMaskDwordsFromVectors(OutputVectors) * InputVectors * 4;
}
#define PSVALIGN(ptr, alignbits) (((ptr) + ((1 << (alignbits))-1)) & ~((1 << (alignbits))-1))
#define PSVALIGN4(ptr) (((ptr) + 3) & ~3)

#ifndef NDEBUG
#define PSV_RETB(exp) do { if(!(exp)) { assert(false && #exp); return false; } } while(0)
#else   // NDEBUG
#define PSV_RETB(exp) do { if(!(exp)) { return false; } } while(0)
#endif  // NDEBUG

struct VSInfo {
  char OutputPositionPresent;
};
struct HSInfo {
  uint32_t InputControlPointCount;      // max control points == 32
  uint32_t OutputControlPointCount;     // max control points == 32
  uint32_t TessellatorDomain;           // hlsl::DXIL::TessellatorDomain/D3D11_SB_TESSELLATOR_DOMAIN
  uint32_t TessellatorOutputPrimitive;  // hlsl::DXIL::TessellatorOutputPrimitive/D3D11_SB_TESSELLATOR_OUTPUT_PRIMITIVE
};
struct DSInfo {
  uint32_t InputControlPointCount;      // max control points == 32
  char OutputPositionPresent;
  uint32_t TessellatorDomain;           // hlsl::DXIL::TessellatorDomain/D3D11_SB_TESSELLATOR_DOMAIN
};
struct GSInfo {
  uint32_t InputPrimitive;              // hlsl::DXIL::InputPrimitive/D3D10_SB_PRIMITIVE
  uint32_t OutputTopology;              // hlsl::DXIL::PrimitiveTopology/D3D10_SB_PRIMITIVE_TOPOLOGY
  uint32_t OutputStreamMask;            // max streams == 4
  char OutputPositionPresent;
};
struct PSInfo {
  char DepthOutput;
  char SampleFrequency;
};
struct MSInfo {
  uint32_t GroupSharedBytesUsed;
  uint32_t GroupSharedBytesDependentOnViewID;
  uint32_t PayloadSizeInBytes;
  uint16_t MaxOutputVertices;
  uint16_t MaxOutputPrimitives;
};
struct ASInfo {
  uint32_t PayloadSizeInBytes;
};
struct MSInfo1 {
  uint8_t SigPrimVectors;     // Primitive output for MS
  uint8_t MeshOutputTopology;
};

// Versioning is additive and based on size
struct PSVRuntimeInfo0
{
  union {
    VSInfo VS;
    HSInfo HS;
    DSInfo DS;
    GSInfo GS;
    PSInfo PS;
    MSInfo MS;
    ASInfo AS;
  };
  uint32_t MinimumExpectedWaveLaneCount;  // minimum lane count required, 0 if unused
  uint32_t MaximumExpectedWaveLaneCount;  // maximum lane count required, 0xffffffff if unused
};

enum class PSVShaderKind : uint8_t    // DXIL::ShaderKind
{
  Pixel = 0,
  Vertex,
  Geometry,
  Hull,
  Domain,
  Compute,
  Library,
  RayGeneration,
  Intersection,
  AnyHit,
  ClosestHit,
  Miss,
  Callable,
  Mesh,
  Amplification,
  Invalid,
};

struct PSVRuntimeInfo1 : public PSVRuntimeInfo0
{
  uint8_t ShaderStage;              // PSVShaderKind
  uint8_t UsesViewID;
  union {
    uint16_t MaxVertexCount;          // MaxVertexCount for GS only (max 1024)
    uint8_t SigPatchConstOrPrimVectors;  // Output for HS; Input for DS; Primitive output for MS (overlaps MS1::SigPrimVectors)
    struct MSInfo1 MS1;
  };

  // PSVSignatureElement counts
  uint8_t SigInputElements;
  uint8_t SigOutputElements;
  uint8_t SigPatchConstOrPrimElements;

  // Number of packed vectors per signature
  uint8_t SigInputVectors;
  uint8_t SigOutputVectors[4];      // Array for GS Stream Out Index
};

struct PSVRuntimeInfo2 : public PSVRuntimeInfo1
{
  uint32_t NumThreadsX;
  uint32_t NumThreadsY;
  uint32_t NumThreadsZ;
};

enum class PSVResourceType
{
  Invalid = 0,

  Sampler,
  CBV,
  SRVTyped,
  SRVRaw,
  SRVStructured,
  UAVTyped,
  UAVRaw,
  UAVStructured,
  UAVStructuredWithCounter,
  NumEntries
};

enum class PSVResourceKind
{
  Invalid = 0,
  Texture1D,
  Texture2D,
  Texture2DMS,
  Texture3D,
  TextureCube,
  Texture1DArray,
  Texture2DArray,
  Texture2DMSArray,
  TextureCubeArray,
  TypedBuffer,
  RawBuffer,
  StructuredBuffer,
  CBuffer,
  Sampler,
  TBuffer,
  RTAccelerationStructure,
  FeedbackTexture2D,
  FeedbackTexture2DArray,
  NumEntries
};

enum class PSVResourceFlag
{
  None           = 0x00000000,
  UsedByAtomic64 = 0x00000001,
};

// Table of null-terminated strings, overall size aligned to dword boundary, last byte must be null
struct PSVStringTable {
  const char *Table;
  uint32_t Size;
  PSVStringTable() : Table(nullptr), Size(0) {}
  PSVStringTable(const char *table, uint32_t size) : Table(table), Size(size) {}
  const char *Get(uint32_t offset) const {
    _Analysis_assume_(offset < Size && Table && Table[Size-1] == '\0');
    return Table + offset;
  }
};

// Versioning is additive and based on size
struct PSVResourceBindInfo0
{
  uint32_t ResType;     // PSVResourceType
  uint32_t Space;
  uint32_t LowerBound;
  uint32_t UpperBound;
};

struct PSVResourceBindInfo1 : public PSVResourceBindInfo0
{
  uint32_t ResKind;     // PSVResourceKind
  uint32_t ResFlags;    // special characteristics of the resource
};

// Helpers for output dependencies (ViewID and Input-Output tables)
struct PSVComponentMask {
  uint32_t *Mask;
  uint32_t NumVectors;
  PSVComponentMask() : Mask(nullptr), NumVectors(0) {}
  PSVComponentMask(const PSVComponentMask &other) : Mask(other.Mask), NumVectors(other.NumVectors) {}
  PSVComponentMask(uint32_t *pMask, uint32_t outputVectors)
  : Mask(pMask),
    NumVectors(outputVectors)
  {}
  const PSVComponentMask &operator|=(const PSVComponentMask &other) {
    uint32_t dwords = PSVComputeMaskDwordsFromVectors(NumVectors < other.NumVectors ? NumVectors : other.NumVectors);
    for (uint32_t i = 0; i < dwords; ++i) {
      Mask[i] |= other.Mask[i];
    }
    return *this;
  }
  bool Get(uint32_t ComponentIndex) const {
    if(ComponentIndex < NumVectors * 4)
      return (bool)(Mask[ComponentIndex >> 5] & (1 << (ComponentIndex & 0x1F)));
    return false;
  }
  void Set(uint32_t ComponentIndex) {
    if (ComponentIndex < NumVectors * 4)
      Mask[ComponentIndex >> 5] |= (1 << (ComponentIndex & 0x1F));
  }
  void Clear(uint32_t ComponentIndex) {
    if (ComponentIndex < NumVectors * 4)
      Mask[ComponentIndex >> 5] &= ~(1 << (ComponentIndex & 0x1F));
  }
  bool IsValid() { return Mask != nullptr; }
};

struct PSVDependencyTable {
  uint32_t *Table;
  uint32_t InputVectors;
  uint32_t OutputVectors;
  PSVDependencyTable() : Table(nullptr), InputVectors(0), OutputVectors(0) {}
  PSVDependencyTable(const PSVDependencyTable &other) : Table(other.Table), InputVectors(other.InputVectors), OutputVectors(other.OutputVectors) {}
  PSVDependencyTable(uint32_t *pTable, uint32_t inputVectors, uint32_t outputVectors)
  : Table(pTable),
    InputVectors(inputVectors),
    OutputVectors(outputVectors)
  {}
  PSVComponentMask GetMaskForInput(uint32_t inputComponentIndex) {
    if (!Table || !InputVectors || !OutputVectors)
      return PSVComponentMask();
    return PSVComponentMask(Table + (PSVComputeMaskDwordsFromVectors(OutputVectors) * inputComponentIndex), OutputVectors);
  }
  bool IsValid() { return Table != nullptr; }
};

struct PSVString {
  uint32_t Offset;
  PSVString() : Offset(0) {}
  PSVString(uint32_t offset) : Offset(offset) {}
  const char *Get(const PSVStringTable &table) const { return table.Get(Offset); }
};

struct PSVSemanticIndexTable {
  const uint32_t *Table;
  uint32_t Entries;
  PSVSemanticIndexTable() : Table(nullptr), Entries(0) {}
  PSVSemanticIndexTable(const uint32_t *table, uint32_t entries) : Table(table), Entries(entries) {}
  const uint32_t *Get(uint32_t offset) const {
    _Analysis_assume_(offset < Entries && Table);
    return Table + offset;
  }
};

struct PSVSemanticIndexes {
  uint32_t Offset;
  PSVSemanticIndexes() : Offset(0) {}
  PSVSemanticIndexes(uint32_t offset) : Offset(offset) {}
  const uint32_t *Get(const PSVSemanticIndexTable &table) const { return table.Get(Offset); }
};

enum class PSVSemanticKind : uint8_t    // DXIL::SemanticKind
{
  Arbitrary,
  VertexID,
  InstanceID,
  Position,
  RenderTargetArrayIndex,
  ViewPortArrayIndex,
  ClipDistance,
  CullDistance,
  OutputControlPointID,
  DomainLocation,
  PrimitiveID,
  GSInstanceID,
  SampleIndex,
  IsFrontFace,
  Coverage,
  InnerCoverage,
  Target,
  Depth,
  DepthLessEqual,
  DepthGreaterEqual,
  StencilRef,
  DispatchThreadID,
  GroupID,
  GroupIndex,
  GroupThreadID,
  TessFactor,
  InsideTessFactor,
  ViewID,
  Barycentrics,
  ShadingRate,
  CullPrimitive,
  Invalid,
};

struct PSVSignatureElement0
{
  uint32_t SemanticName;          // Offset into StringTable
  uint32_t SemanticIndexes;       // Offset into PSVSemanticIndexTable, count == Rows
  uint8_t Rows;                   // Number of rows this element occupies
  uint8_t StartRow;               // Starting row of packing location if allocated
  uint8_t ColsAndStart;           // 0:4 = Cols, 4:6 = StartCol, 6:7 == Allocated
  uint8_t SemanticKind;           // PSVSemanticKind
  uint8_t ComponentType;          // DxilProgramSigCompType
  uint8_t InterpolationMode;      // DXIL::InterpolationMode or D3D10_SB_INTERPOLATION_MODE
  uint8_t DynamicMaskAndStream;   // 0:4 = DynamicIndexMask, 4:6 = OutputStream (0-3)
  uint8_t Reserved;
};

// Provides convenient access to packed PSVSignatureElementN structure
class PSVSignatureElement
{
private:
  const PSVStringTable &m_StringTable;
  const PSVSemanticIndexTable &m_SemanticIndexTable;
  const PSVSignatureElement0 *m_pElement0;
public:
  PSVSignatureElement(const PSVStringTable &stringTable, const PSVSemanticIndexTable &semanticIndexTable, const PSVSignatureElement0 *pElement0)
  : m_StringTable(stringTable), m_SemanticIndexTable(semanticIndexTable), m_pElement0(pElement0) {}
  const char *GetSemanticName() const { return !m_pElement0 ? "" : m_StringTable.Get(m_pElement0->SemanticName); }
  const uint32_t *GetSemanticIndexes() const { return !m_pElement0 ? nullptr: m_SemanticIndexTable.Get(m_pElement0->SemanticIndexes); }
  uint32_t GetRows() const { return !m_pElement0 ? 0 : ((uint32_t)m_pElement0->Rows); }
  uint32_t GetCols() const { return !m_pElement0 ? 0 : ((uint32_t)m_pElement0->ColsAndStart & 0xF); }
  bool IsAllocated() const { return !m_pElement0 ? false : !!(m_pElement0->ColsAndStart & 0x40); }
  int32_t GetStartRow() const { return !m_pElement0 ? 0 : !IsAllocated() ? -1 : (int32_t)m_pElement0->StartRow; }
  int32_t GetStartCol() const { return !m_pElement0 ? 0 : !IsAllocated() ? -1 : (int32_t)((m_pElement0->ColsAndStart >> 4) & 0x3); }
  PSVSemanticKind GetSemanticKind() const { return !m_pElement0 ? (PSVSemanticKind)0 : (PSVSemanticKind)m_pElement0->SemanticKind; }
  uint32_t GetComponentType() const { return !m_pElement0 ? 0 : (uint32_t)m_pElement0->ComponentType; }
  uint32_t GetInterpolationMode() const { return !m_pElement0 ? 0 : (uint32_t)m_pElement0->InterpolationMode; }
  uint32_t GetOutputStream() const { return !m_pElement0 ? 0 : (uint32_t)(m_pElement0->DynamicMaskAndStream >> 4) & 0x3; }
  uint32_t GetDynamicIndexMask() const { return !m_pElement0 ? 0 : (uint32_t)m_pElement0->DynamicMaskAndStream & 0xF; }
};

#define MAX_PSV_VERSION 2

struct PSVInitInfo
{
  PSVInitInfo(uint32_t psvVersion)
    : PSVVersion(psvVersion)
  {}
  uint32_t PSVVersion = 0;
  uint32_t ResourceCount = 0;
  PSVShaderKind ShaderStage = PSVShaderKind::Invalid;
  PSVStringTable StringTable;
  PSVSemanticIndexTable SemanticIndexTable;
  uint8_t UsesViewID = 0;
  uint8_t SigInputElements = 0;
  uint8_t SigOutputElements = 0;
  uint8_t SigPatchConstOrPrimElements = 0;
  uint8_t SigInputVectors = 0;
  uint8_t SigPatchConstOrPrimVectors = 0;
  uint8_t SigOutputVectors[4] = {0, 0, 0, 0};

  static_assert(MAX_PSV_VERSION == 2, "otherwise this needs updating.");
  uint32_t RuntimeInfoSize() const {
    switch (PSVVersion) {
    case 0: return sizeof(PSVRuntimeInfo0);
    case 1: return sizeof(PSVRuntimeInfo1);
    default: break;
    }
   return sizeof(PSVRuntimeInfo2);
  }
  uint32_t ResourceBindInfoSize() const {
    if (PSVVersion < 2)
      return sizeof(PSVResourceBindInfo0);
    return sizeof(PSVResourceBindInfo1);
  }
  uint32_t SignatureElementSize() const {
    return sizeof(PSVSignatureElement0);
  }
};

class DxilPipelineStateValidation
{
  uint32_t m_uPSVRuntimeInfoSize = 0;
  PSVRuntimeInfo0* m_pPSVRuntimeInfo0 = nullptr;
  PSVRuntimeInfo1 *m_pPSVRuntimeInfo1 = nullptr;
  PSVRuntimeInfo2 *m_pPSVRuntimeInfo2 = nullptr;
  uint32_t m_uResourceCount = 0;
  uint32_t m_uPSVResourceBindInfoSize = 0;
  void *m_pPSVResourceBindInfo = nullptr;
  PSVStringTable m_StringTable;
  PSVSemanticIndexTable m_SemanticIndexTable;
  uint32_t m_uPSVSignatureElementSize = 0;
  void *m_pSigInputElements = nullptr;
  void *m_pSigOutputElements = nullptr;
  void *m_pSigPatchConstOrPrimElements = nullptr;
  uint32_t *m_pViewIDOutputMask = nullptr;
  uint32_t *m_pViewIDPCOrPrimOutputMask = nullptr;
  uint32_t *m_pInputToOutputTable = nullptr;
  uint32_t *m_pInputToPCOutputTable = nullptr;
  uint32_t *m_pPCInputToOutputTable = nullptr;

public:
  DxilPipelineStateValidation() {}

  enum class RWMode {
    Read,
    CalcSize,
    Write,
  };

  class CheckedReaderWriter {
  private:
    char *Ptr;
    uint32_t Size;
    uint32_t Offset;
    RWMode Mode;

  public:
    CheckedReaderWriter(const void *ptr, uint32_t size, RWMode mode)
      : Ptr(reinterpret_cast<char*>(const_cast<void*>(ptr))),
        Size(mode == RWMode::CalcSize ? 0 : size), Offset(0), Mode(mode) {}

    uint32_t GetSize() { return Size; }
    RWMode GetMode() { return Mode; }

    // Return true if size fits in remaing buffer.
    bool CheckBounds(size_t size);
    // Increment Offset by size. Return false on error.
    bool IncrementPos(size_t size);

    // Cast and return true if it fits.
    template <typename _T> bool Cast(_T **ppPtr, size_t size);
    template <typename _T> bool Cast(_T **ppPtr);

    // Map* methods increment Offset.

    // Assign pointer, increment Offset, and return true if size fits.
    template<typename _T> bool MapPtr(_T **ppPtr, size_t size = 0);
    // Read value, increment Offset, and return true if value fits.
    template <typename _T> bool MapValue(_T *pValue, const _T init = {});
    // Assign pointer, increment Offset, and return true if array fits.
    template <typename _T>
    bool MapArray(_T **ppPtr, size_t count, size_t eltSize);
    //template <> bool MapArray<void>(void **ppPtr, size_t count, size_t eltSize);
    // Assign pointer, increment Offset, and return true if array fits.
    template <typename _T> bool MapArray(_T **ppPtr, size_t count = 1);

    void Clear();
  };

  // Assigned derived ptr to base if size large enough.
  template<typename _Base, typename _T>
  bool AssignDerived(_T** ppDerived, _Base* pBase, uint32_t size) {
    if ((size_t)size < sizeof(_T))
      return false;   // size not large enough for type
    *ppDerived = reinterpret_cast<_T *>(pBase);
    return true;
  }

private:
  bool ReadOrWrite(const void *pBits, uint32_t *pSize, RWMode mode,
                   const PSVInitInfo &initInfo = PSVInitInfo(MAX_PSV_VERSION));

public:
  bool InitFromPSV0(const void* pBits, uint32_t size) {
    return ReadOrWrite(pBits, &size, RWMode::Read);
  }

  // Initialize a new buffer
  // call with null pBuffer to get required size

  bool InitNew(uint32_t ResourceCount, void *pBuffer, uint32_t *pSize) {
    PSVInitInfo initInfo(0);
    initInfo.ResourceCount = ResourceCount;
    return InitNew(initInfo, pBuffer, pSize);
  }

  bool InitNew(const PSVInitInfo &initInfo, void *pBuffer, uint32_t *pSize) {
    RWMode Mode = nullptr != pBuffer ? RWMode::Write : RWMode::CalcSize;
    return ReadOrWrite(pBuffer, pSize, Mode, initInfo);
  }

  PSVRuntimeInfo0* GetPSVRuntimeInfo0() const {
    return m_pPSVRuntimeInfo0;
  }

  PSVRuntimeInfo1* GetPSVRuntimeInfo1() const {
    return m_pPSVRuntimeInfo1;
  }

  PSVRuntimeInfo2* GetPSVRuntimeInfo2() const {
    return m_pPSVRuntimeInfo2;
  }

  uint32_t GetBindCount() const {
    return m_uResourceCount;
  }

  template <typename _T>
  _T *GetRecord(void *pRecords, uint32_t recordSize, uint32_t numRecords,
                uint32_t index) const {
    if (pRecords && index < numRecords && sizeof(_T) <= recordSize) {
      __analysis_assume((size_t)index * (size_t)recordSize <= UINT_MAX);
      return reinterpret_cast<_T *>(reinterpret_cast<uint8_t *>(pRecords) +
                                    (index * recordSize));
    }
    return nullptr;
  }

  PSVResourceBindInfo0* GetPSVResourceBindInfo0(uint32_t index) const {
    return GetRecord<PSVResourceBindInfo0>(m_pPSVResourceBindInfo,
                                           m_uPSVResourceBindInfoSize,
                                           m_uResourceCount, index);
  }

  PSVResourceBindInfo1* GetPSVResourceBindInfo1(uint32_t index) const {
    return GetRecord<PSVResourceBindInfo1>(m_pPSVResourceBindInfo,
                                           m_uPSVResourceBindInfoSize,
                                           m_uResourceCount, index);
  }

  const PSVStringTable &GetStringTable() const { return m_StringTable; }
  const PSVSemanticIndexTable &GetSemanticIndexTable() const { return m_SemanticIndexTable; }

  // Signature element access
  uint32_t GetSigInputElements() const {
    if (m_pPSVRuntimeInfo1)
      return m_pPSVRuntimeInfo1->SigInputElements;
    return 0;
  }
  uint32_t GetSigOutputElements() const {
    if (m_pPSVRuntimeInfo1)
      return m_pPSVRuntimeInfo1->SigOutputElements;
    return 0;
  }
  uint32_t GetSigPatchConstOrPrimElements() const {
    if (m_pPSVRuntimeInfo1)
      return m_pPSVRuntimeInfo1->SigPatchConstOrPrimElements;
    return 0;
  }
  PSVSignatureElement0* GetInputElement0(uint32_t index) const {
    return GetRecord<PSVSignatureElement0>(
        m_pSigInputElements, m_uPSVSignatureElementSize,
        m_pPSVRuntimeInfo1 ? m_pPSVRuntimeInfo1->SigInputElements : 0, index);
  }
  PSVSignatureElement0* GetOutputElement0(uint32_t index) const {
    return GetRecord<PSVSignatureElement0>(
        m_pSigOutputElements, m_uPSVSignatureElementSize,
        m_pPSVRuntimeInfo1 ? m_pPSVRuntimeInfo1->SigOutputElements : 0, index);
  }
  PSVSignatureElement0* GetPatchConstOrPrimElement0(uint32_t index) const {
    return GetRecord<PSVSignatureElement0>(
        m_pSigPatchConstOrPrimElements, m_uPSVSignatureElementSize,
        m_pPSVRuntimeInfo1 ? m_pPSVRuntimeInfo1->SigPatchConstOrPrimElements : 0, index);
  }
  // More convenient wrapper:
  PSVSignatureElement GetSignatureElement(PSVSignatureElement0* pElement0) const {
    return PSVSignatureElement(m_StringTable, m_SemanticIndexTable, pElement0);
  }

  PSVShaderKind GetShaderKind() const {
    if (m_pPSVRuntimeInfo1 && m_pPSVRuntimeInfo1->ShaderStage < (uint8_t)PSVShaderKind::Invalid)
      return (PSVShaderKind)m_pPSVRuntimeInfo1->ShaderStage;
    return PSVShaderKind::Invalid;
  }
  bool IsVS() const { return GetShaderKind() == PSVShaderKind::Vertex; }
  bool IsHS() const { return GetShaderKind() == PSVShaderKind::Hull; }
  bool IsDS() const { return GetShaderKind() == PSVShaderKind::Domain; }
  bool IsGS() const { return GetShaderKind() == PSVShaderKind::Geometry; }
  bool IsPS() const { return GetShaderKind() == PSVShaderKind::Pixel; }
  bool IsCS() const { return GetShaderKind() == PSVShaderKind::Compute; }
  bool IsMS() const { return GetShaderKind() == PSVShaderKind::Mesh; }
  bool IsAS() const { return GetShaderKind() == PSVShaderKind::Amplification; }

  // ViewID dependencies
  PSVComponentMask GetViewIDOutputMask(unsigned streamIndex = 0) const {
    if (!m_pViewIDOutputMask || !m_pPSVRuntimeInfo1 || !m_pPSVRuntimeInfo1->SigOutputVectors[streamIndex])
      return PSVComponentMask();
    return PSVComponentMask(m_pViewIDOutputMask, m_pPSVRuntimeInfo1->SigOutputVectors[streamIndex]);
  }
  PSVComponentMask GetViewIDPCOutputMask() const {
    if ((!IsHS() && !IsMS()) || !m_pViewIDPCOrPrimOutputMask || !m_pPSVRuntimeInfo1 || !m_pPSVRuntimeInfo1->SigPatchConstOrPrimVectors)
      return PSVComponentMask();
    return PSVComponentMask(m_pViewIDPCOrPrimOutputMask, m_pPSVRuntimeInfo1->SigPatchConstOrPrimVectors);
  }

  // Input to Output dependencies
  PSVDependencyTable GetInputToOutputTable(unsigned streamIndex = 0) const {
    if (m_pInputToOutputTable && m_pPSVRuntimeInfo1) {
      return PSVDependencyTable(m_pInputToOutputTable, m_pPSVRuntimeInfo1->SigInputVectors, m_pPSVRuntimeInfo1->SigOutputVectors[streamIndex]);
    }
    return PSVDependencyTable();
  }
  PSVDependencyTable GetInputToPCOutputTable() const {
    if (IsHS() && m_pInputToPCOutputTable && m_pPSVRuntimeInfo1) {
      return PSVDependencyTable(m_pInputToPCOutputTable, m_pPSVRuntimeInfo1->SigInputVectors, m_pPSVRuntimeInfo1->SigPatchConstOrPrimVectors);
    }
    return PSVDependencyTable();
  }
  PSVDependencyTable GetPCInputToOutputTable() const {
    if (IsDS() && m_pPCInputToOutputTable && m_pPSVRuntimeInfo1) {
      return PSVDependencyTable(m_pPCInputToOutputTable, m_pPSVRuntimeInfo1->SigPatchConstOrPrimVectors, m_pPSVRuntimeInfo1->SigOutputVectors[0]);
    }
    return PSVDependencyTable();
  }

  bool GetNumThreads(uint32_t *pNumThreadsX, uint32_t *pNumThreadsY, uint32_t *pNumThreadsZ) {
    if (m_pPSVRuntimeInfo2) {
      if (pNumThreadsX) *pNumThreadsX = m_pPSVRuntimeInfo2->NumThreadsX;
      if (pNumThreadsY) *pNumThreadsY = m_pPSVRuntimeInfo2->NumThreadsY;
      if (pNumThreadsZ) *pNumThreadsZ = m_pPSVRuntimeInfo2->NumThreadsZ;
      return true;
    }
    return false;
  }
};

// Return true if size fits in remaing buffer.
inline bool
DxilPipelineStateValidation::CheckedReaderWriter::CheckBounds(size_t size) {
  if (Mode != RWMode::CalcSize) {
    PSV_RETB(size <= UINT_MAX);
    PSV_RETB(Offset <= Size);
    return (uint32_t)size <= Size - Offset;
  }
  return true;
}
// Increment Offset by size. Return false on error.
inline bool
DxilPipelineStateValidation::CheckedReaderWriter::IncrementPos(size_t size) {
  PSV_RETB(size <= UINT_MAX);
  uint32_t uSize = (uint32_t)size;
  if (Mode == RWMode::CalcSize) {
    PSV_RETB(uSize <= Size + uSize);
    Size += uSize;
  }
  PSV_RETB(CheckBounds(size));
  Offset += uSize;
  return true;
}

// Cast and return true if it fits.
template <typename _T> inline bool
DxilPipelineStateValidation::CheckedReaderWriter::Cast(_T **ppPtr, size_t size) {
  PSV_RETB(CheckBounds(size));
  *ppPtr = reinterpret_cast<_T*>(Ptr + Offset);
  return true;
}
template <typename _T>
inline bool DxilPipelineStateValidation::CheckedReaderWriter::Cast(_T **ppPtr) {
  return Cast(ppPtr, sizeof(_T));
}

// Map* methods increment Offset.

// Assign pointer, increment Offset, and return true if size fits.
template<typename _T>
inline bool
DxilPipelineStateValidation::CheckedReaderWriter::MapPtr(_T **ppPtr, size_t size) {
  PSV_RETB(Cast(ppPtr, size));
  PSV_RETB(IncrementPos(size));
  return true;
}
// Read value, increment Offset, and return true if value fits.
template <typename _T>
inline bool
DxilPipelineStateValidation::CheckedReaderWriter::MapValue(_T *pValue, const _T init) {
  _T *pPtr = nullptr;
  PSV_RETB(MapPtr(&pPtr, sizeof(_T)));
  switch (Mode) {
  case RWMode::Read: *pValue = *pPtr; break;
  case RWMode::CalcSize: *pValue = init; break;
  case RWMode::Write: *pPtr = *pValue = init; break;
  }
  return true;
}
// Assign pointer, increment Offset, and return true if array fits.
template <typename _T>
inline bool DxilPipelineStateValidation::CheckedReaderWriter::MapArray(
    _T **ppPtr, size_t count, size_t eltSize) {
  PSV_RETB(count <= UINT_MAX && eltSize <= UINT_MAX && eltSize >= sizeof(_T));
  return count ? MapPtr(ppPtr, eltSize * count) : true;
}
//template <> bool MapArray<void>(void **ppPtr, size_t count, size_t eltSize);
template <>
inline bool DxilPipelineStateValidation::CheckedReaderWriter::MapArray<void>(
    void **ppPtr, size_t count, size_t eltSize) {
  PSV_RETB(count <= UINT_MAX && eltSize <= UINT_MAX && eltSize > 0);
  return count ? MapPtr(ppPtr, eltSize * count) : true;
}

// Assign pointer, increment Offset, and return true if array fits.
template <typename _T>
inline bool
DxilPipelineStateValidation::CheckedReaderWriter::MapArray(_T **ppPtr, size_t count) {
  return count ? MapArray(ppPtr, count, sizeof(_T)) : true;
}

inline void DxilPipelineStateValidation::CheckedReaderWriter::Clear() {
  if (Mode == RWMode::Write) {
    memset(Ptr, 0, Size);
  }
}

// PSV0 blob part looks like:
// uint32_t PSVRuntimeInfo_size
// { PSVRuntimeInfoN structure }
// uint32_t ResourceCount
// If ResourceCount > 0:
//    uint32_t PSVResourceBindInfo_size
//    { PSVResourceBindInfoN structure } * ResourceCount
// If PSVRuntimeInfo1:
//    uint32_t StringTableSize (dword aligned)
//    If StringTableSize:
//      { StringTableSize bytes, null terminated }
//    uint32_t SemanticIndexTableEntries (number of dwords)
//    If SemanticIndexTableEntries:
//      { semantic index } * SemanticIndexTableEntries
//    If SigInputElements || SigOutputElements || SigPatchConstOrPrimElements:
//      uint32_t PSVSignatureElement_size
//      { PSVSignatureElementN structure } * SigInputElements
//      { PSVSignatureElementN structure } * SigOutputElements
//      { PSVSignatureElementN structure } * SigPatchConstOrPrimElements
//    If (UsesViewID):
//      For (i : each stream index 0-3):
//        If (SigOutputVectors[i] non-zero):
//          { uint32_t * PSVComputeMaskDwordsFromVectors(SigOutputVectors[i]) }
//            - Outputs affected by ViewID as a bitmask
//      If (HS and SigPatchConstOrPrimVectors non-zero):
//        { uint32_t * PSVComputeMaskDwordsFromVectors(SigPatchConstOrPrimVectors) }
//          - PCOutputs affected by ViewID as a bitmask
//    For (i : each stream index 0-3):
//      If (SigInputVectors and SigOutputVectors[i] non-zero):
//        { uint32_t * PSVComputeInputOutputTableDwords(SigInputVectors, SigOutputVectors[i]) }
//          - Outputs affected by inputs as a table of bitmasks
//    If (HS and SigPatchConstOrPrimVectors and SigInputVectors non-zero):
//      { uint32_t * PSVComputeInputOutputTableDwords(SigInputVectors, SigPatchConstOrPrimVectors) }
//        - Patch constant outputs affected by inputs as a table of bitmasks
//    If (DS and SigOutputVectors[0] and SigPatchConstOrPrimVectors non-zero):
//      { uint32_t * PSVComputeInputOutputTableDwords(SigPatchConstOrPrimVectors, SigOutputVectors[0]) }
//        - Outputs affected by patch constant inputs as a table of bitmasks
// returns true if no errors occurred.
inline bool DxilPipelineStateValidation::ReadOrWrite(
    const void *pBits, uint32_t *pSize, RWMode mode,
    const PSVInitInfo &initInfo) {
  PSV_RETB(pSize != nullptr);
  PSV_RETB(pBits != nullptr || mode == RWMode::CalcSize);
  PSV_RETB(initInfo.PSVVersion <= MAX_PSV_VERSION);

  CheckedReaderWriter rw(pBits, *pSize, mode);
  rw.Clear();

  PSV_RETB(rw.MapValue(&m_uPSVRuntimeInfoSize, initInfo.RuntimeInfoSize()));
  PSV_RETB(rw.MapArray(&m_pPSVRuntimeInfo0, 1, m_uPSVRuntimeInfoSize));
  AssignDerived(&m_pPSVRuntimeInfo1, m_pPSVRuntimeInfo0, m_uPSVRuntimeInfoSize); // failure ok
  AssignDerived(&m_pPSVRuntimeInfo2, m_pPSVRuntimeInfo0, m_uPSVRuntimeInfoSize); // failure ok

  // In RWMode::CalcSize, use temp runtime info to hold needed values from initInfo
  PSVRuntimeInfo1 tempRuntimeInfo = {};
  if (mode == RWMode::CalcSize && initInfo.PSVVersion > 0) {
    m_pPSVRuntimeInfo1 = &tempRuntimeInfo;
  }

  PSV_RETB(rw.MapValue(&m_uResourceCount, initInfo.ResourceCount));

  if (m_uResourceCount > 0) {
    PSV_RETB(rw.MapValue(&m_uPSVResourceBindInfoSize, initInfo.ResourceBindInfoSize()));
    PSV_RETB(sizeof(PSVResourceBindInfo0) <= m_uPSVResourceBindInfoSize);
    PSV_RETB(rw.MapArray(&m_pPSVResourceBindInfo, m_uResourceCount, m_uPSVResourceBindInfoSize));
  }

  if (m_pPSVRuntimeInfo1) {
    if (mode != RWMode::Read) {
      m_pPSVRuntimeInfo1->ShaderStage = (uint8_t)initInfo.ShaderStage;
      m_pPSVRuntimeInfo1->SigInputElements = initInfo.SigInputElements;
      m_pPSVRuntimeInfo1->SigOutputElements = initInfo.SigOutputElements;
      m_pPSVRuntimeInfo1->SigPatchConstOrPrimElements = initInfo.SigPatchConstOrPrimElements;
      m_pPSVRuntimeInfo1->UsesViewID = initInfo.UsesViewID;
      for (unsigned i = 0; i < 4; i++) {
        m_pPSVRuntimeInfo1->SigOutputVectors[i] = initInfo.SigOutputVectors[i];
      }
      if (IsHS() || IsDS() || IsMS()) {
        m_pPSVRuntimeInfo1->SigPatchConstOrPrimVectors = initInfo.SigPatchConstOrPrimVectors;
      }
      m_pPSVRuntimeInfo1->SigInputVectors = initInfo.SigInputVectors;
    }

    PSV_RETB(rw.MapValue(&m_StringTable.Size, PSVALIGN4(initInfo.StringTable.Size)));
    PSV_RETB(PSVALIGN4(m_StringTable.Size) == m_StringTable.Size);
    if (m_StringTable.Size) {
      PSV_RETB(rw.MapArray(&m_StringTable.Table, m_StringTable.Size));
      if (mode == RWMode::Write) {
        memcpy(const_cast<char*>(m_StringTable.Table), initInfo.StringTable.Table, initInfo.StringTable.Size);
      }
    }

    PSV_RETB(rw.MapValue(&m_SemanticIndexTable.Entries, initInfo.SemanticIndexTable.Entries));
    if (m_SemanticIndexTable.Entries) {
      PSV_RETB(rw.MapArray(&m_SemanticIndexTable.Table, m_SemanticIndexTable.Entries));
      if (mode == RWMode::Write) {
        memcpy(const_cast<uint32_t*>(m_SemanticIndexTable.Table), initInfo.SemanticIndexTable.Table, sizeof(uint32_t) * initInfo.SemanticIndexTable.Entries);
      }
    }

    // Dxil Signature Elements
    if (m_pPSVRuntimeInfo1->SigInputElements || m_pPSVRuntimeInfo1->SigOutputElements || m_pPSVRuntimeInfo1->SigPatchConstOrPrimElements) {
      PSV_RETB(rw.MapValue(&m_uPSVSignatureElementSize, initInfo.SignatureElementSize()));
      PSV_RETB(sizeof(PSVSignatureElement0) <= m_uPSVSignatureElementSize);
      if (m_pPSVRuntimeInfo1->SigInputElements) {
        PSV_RETB(rw.MapArray(&m_pSigInputElements, m_pPSVRuntimeInfo1->SigInputElements, m_uPSVSignatureElementSize));
      }
      if (m_pPSVRuntimeInfo1->SigOutputElements) {
        PSV_RETB(rw.MapArray(&m_pSigOutputElements, m_pPSVRuntimeInfo1->SigOutputElements, m_uPSVSignatureElementSize));
      }
      if (m_pPSVRuntimeInfo1->SigPatchConstOrPrimElements) {
        PSV_RETB(rw.MapArray(&m_pSigPatchConstOrPrimElements, m_pPSVRuntimeInfo1->SigPatchConstOrPrimElements, m_uPSVSignatureElementSize));
      }
    }

    // ViewID dependencies
    if (m_pPSVRuntimeInfo1->UsesViewID) {
      for (unsigned i = 0; i < 4; i++) {
        if (m_pPSVRuntimeInfo1->SigOutputVectors[i]) {
          PSV_RETB(rw.MapArray(&m_pViewIDOutputMask,
            PSVComputeMaskDwordsFromVectors(m_pPSVRuntimeInfo1->SigOutputVectors[i])));
        }
        if (!IsGS())
          break;
      }
      if ((IsHS() || IsMS()) && m_pPSVRuntimeInfo1->SigPatchConstOrPrimVectors) {
        PSV_RETB(rw.MapArray(&m_pViewIDPCOrPrimOutputMask,
          PSVComputeMaskDwordsFromVectors(m_pPSVRuntimeInfo1->SigPatchConstOrPrimVectors)));
      }
    }

    // Input to Output dependencies
    for (unsigned i = 0; i < 4; i++) {
      if (!IsMS() && m_pPSVRuntimeInfo1->SigOutputVectors[i] > 0 && m_pPSVRuntimeInfo1->SigInputVectors > 0) {
        PSV_RETB(rw.MapArray(&m_pInputToOutputTable,
          PSVComputeInputOutputTableDwords(m_pPSVRuntimeInfo1->SigInputVectors, m_pPSVRuntimeInfo1->SigOutputVectors[i])));
      }
      if (!IsGS())
        break;
    }
    if (IsHS() && m_pPSVRuntimeInfo1->SigPatchConstOrPrimVectors > 0 && m_pPSVRuntimeInfo1->SigInputVectors > 0) {
      PSV_RETB(rw.MapArray(&m_pInputToPCOutputTable,
        PSVComputeInputOutputTableDwords(m_pPSVRuntimeInfo1->SigInputVectors, m_pPSVRuntimeInfo1->SigPatchConstOrPrimVectors)));
    }
    if (IsDS() && m_pPSVRuntimeInfo1->SigOutputVectors[0] > 0 && m_pPSVRuntimeInfo1->SigPatchConstOrPrimVectors > 0) {
      PSV_RETB(rw.MapArray(&m_pPCInputToOutputTable,
        PSVComputeInputOutputTableDwords(m_pPSVRuntimeInfo1->SigPatchConstOrPrimVectors, m_pPSVRuntimeInfo1->SigOutputVectors[0])));
    }
  }

  if (mode == RWMode::CalcSize) {
    *pSize = rw.GetSize();
    m_pPSVRuntimeInfo1 = nullptr; // clear ptr to tempRuntimeInfo
  }
  return true;
}

namespace hlsl {

  class ViewIDValidator {
  public:
    enum class Result {
      Success = 0,
      SuccessWithViewIDDependentTessFactor,
      InsufficientInputSpace,
      InsufficientOutputSpace,
      InsufficientPCSpace,
      MismatchedSignatures,
      MismatchedPCSignatures,
      InvalidUsage,
      InvalidPSVVersion,
      InvalidPSV,
    };
    virtual ~ViewIDValidator() {}
    virtual Result ValidateStage(const DxilPipelineStateValidation &PSV,
                                 bool bFinalStage,
                                 bool bExpandInputOnly,
                                 unsigned &mismatchElementId) = 0;
  };

  ViewIDValidator* NewViewIDValidator(unsigned viewIDCount, unsigned gsRastStreamIndex);

}

#undef PSV_RETB

#endif  // __DXIL_PIPELINE_STATE_VALIDATION__H__
