///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilLibraryReflection.h                                                   //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Defines shader reflection for runtime usage.                              //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once
#include "dxc/DXIL/DxilConstants.h"

#define RDAT_NULL_REF ((uint32_t)0xFFFFFFFF)

namespace hlsl {
namespace RDAT {

// Data Layout:
// -start:
//  RuntimeDataHeader header;
//  uint32_t offsets[header.PartCount];
//  - for each i in header.PartCount:
//    - at &header + offsets[i]:
//      RuntimeDataPartHeader part;
//    - if part.Type is a Table (Function or Resource):
//      RuntimeDataTableHeader table;
//      byte TableData[table.RecordCount][table.RecordStride];
//    - else if part.Type is String:
//      byte UTF8Data[part.Size];
//    - else if part.Type is Index:
//      uint32_t IndexData[part.Size / 4];

enum RuntimeDataVersion {
  // Cannot be mistaken for part count from prerelease version
  RDAT_Version_10 = 0x10,
};

enum class RuntimeDataPartType : uint32_t {
  Invalid             = 0,
  StringBuffer        = 1,
  IndexArrays         = 2,
  ResourceTable       = 3,
  FunctionTable       = 4,
  Last_1_3 = FunctionTable,
  RawBytes            = 5,
  SubobjectTable      = 6,
  Last_1_4 = SubobjectTable,
  LastPlus1,
  LastExperimental = LastPlus1 - 1,
};

inline
RuntimeDataPartType MaxPartTypeForValVer(unsigned Major, unsigned Minor) {
  return DXIL::CompareVersions(Major, Minor, 1, 3) < 0
             ? RuntimeDataPartType::Invalid // No RDAT before 1.3
         : DXIL::CompareVersions(Major, Minor, 1, 4) < 0
             ? RuntimeDataPartType::Last_1_3
         : DXIL::CompareVersions(Major, Minor, 1, 7) <= 0
             ? RuntimeDataPartType::Last_1_4
             : RuntimeDataPartType::LastExperimental;
}

enum class RecordTableIndex : unsigned {
  ResourceTable,
  FunctionTable,
  SubobjectTable,
  RecordTableCount
};

///////////////////////////////////////
// Header Structures

struct RuntimeDataHeader {
  uint32_t Version;
  uint32_t PartCount;
  // Followed by uint32_t array of offsets to parts
  // offsets are relative to the beginning of this header
  // offsets must be 4-byte aligned
  //  uint32_t offsets[];
};
struct RuntimeDataPartHeader {
  RuntimeDataPartType Type;
  uint32_t Size;  // Not including this header.  Must be 4-byte aligned.
  // Followed by part data
  //  byte Data[ALIGN4(Size)];
};

// For tables of records, such as Function and Resource tables
// Stride allows for extending records, with forward and backward compatibility
struct RuntimeDataTableHeader {
  uint32_t RecordCount;
  uint32_t RecordStride;  // Must be 4-byte aligned.
  // Followed by recordCount records of recordStride size
  // byte TableData[RecordCount * RecordStride];
};

///////////////////////////////////////
// Raw Reader Classes

// General purpose strided table reader with casting Row() operation that
// returns nullptr if stride is smaller than type, for record expansion.
class TableReader {
  const char *m_table;
  uint32_t m_count;
  uint32_t m_stride;

public:
  TableReader() : TableReader(nullptr, 0, 0) {}
  TableReader(const char *table, uint32_t count, uint32_t stride)
    : m_table(table), m_count(count), m_stride(stride) {}
  void Init(const char *table, uint32_t count, uint32_t stride) {
    m_table = table; m_count = count; m_stride = stride;
  }
  const char *Data() const { return m_table; }
  uint32_t Count() const { return m_count; }
  uint32_t Stride() const { return m_stride; }

  template<typename T> const T *Row(uint32_t index) const {
    if (Valid() && index < m_count && sizeof(T) <= m_stride)
      return reinterpret_cast<const T*>(m_table + (m_stride * index));
    return nullptr;
  }
  bool Valid() const { return m_table && m_count && m_stride; }
  operator bool() { return Valid(); }
};

// Index table is a sequence of rows, where each row has a count as a first
// element followed by the count number of elements pre computing values
class IndexTableReader {
private:
  const uint32_t *m_table;
  uint32_t m_size;

public:
  class IndexRow {
  private:
    const uint32_t *m_values;
    const uint32_t m_count;

  public:
    IndexRow() : m_values(nullptr), m_count(0) {}
    IndexRow(const uint32_t *values, uint32_t count)
        : m_values(values), m_count(count) {}
    uint32_t Count() const { return m_count; }
    uint32_t At(uint32_t i) const {
      return (m_values && i < m_count) ? m_values[i] : 0;
    }
    const uint32_t &operator[](uint32_t i) const {
      if (m_values && i < m_count)
        return m_values[i];
      return m_values[0]; // If null, we should AV if value is read
    }
    bool empty() const { return !(m_values && m_count > 0); }
    operator bool() const { return !empty(); }
  };

  IndexTableReader() : IndexTableReader(nullptr, 0) {}
  IndexTableReader(const uint32_t *table, uint32_t size)
    : m_table(table), m_size(size) {}
  void Init(const uint32_t *table, uint32_t size) {
    m_table = table; m_size = size;
  }
  IndexRow getRow(uint32_t i) const {
    if (Valid() && i < m_size - 1 && m_table[i] + i < m_size) {
      return IndexRow(&m_table[i] + 1, m_table[i]);
    }
    return {};
  }
  const uint32_t *Data() const { return m_table; }
  uint32_t Count() const { return Valid() ? m_size : 0; }
  bool Valid() const { return m_table && m_size > 0; }
  operator bool() const { return Valid(); }
};

class StringTableReader {
  const char *m_table = nullptr;
  uint32_t m_size = 0;
public:
  void Init(const char *table, uint32_t size) {
    m_table = table;
    m_size = size;
  }
  const char *Get(uint32_t offset) const {
    _Analysis_assume_(offset < m_size && m_table &&
                      m_table[m_size - 1] == '\0');
    (void)m_size; // avoid unused private warning if use above is ignored.
    return m_table + offset;
  }
  uint32_t Size() const { return m_size; }
  const char *Data() const { return m_table; }
};

class RawBytesReader {
  const void *m_table;
  uint32_t m_size;
public:
  RawBytesReader(const void *table, uint32_t size)
      : m_table(table), m_size(size) {}
  RawBytesReader() : RawBytesReader(nullptr, 0) {}
  void Init(const void *table, uint32_t size) {
    m_table = table; m_size = size;
  }
  uint32_t Size() const { return m_size; }
  const void *Get(uint32_t offset) const {
    _Analysis_assume_(offset < m_size && m_table);
    (void)m_size; // avoid unused private warning if use above is ignored.
    return (const void*)(((const char*)m_table) + offset);
  }
};

///////////////////////////////////////
// Record Traits

template<typename _T>
class RecordTraits {
public:
  static constexpr const char *TypeName() {
#ifdef _WIN32
    static_assert(false, "");
#else
    assert(false);
#endif
    return nullptr;
  }
  // If the following static assert is hit, it means a structure defined with
  // RDAT_STRUCT is being used in ref type, which requires the struct to have
  // a table and be defined with RDAT_STRUCT_TABLE instead.
  static constexpr RecordTableIndex TableIndex() {
#ifdef _WIN32
    static_assert(false, "");
#else
    assert(false);
#endif
    return (RecordTableIndex)-1;
  }
  // RecordSize() is defined in order to allow for use of forward decl type in RecordRef
  static constexpr size_t RecordSize() { /*static_assert(false, "");*/ return sizeof(_T); }
};

///////////////////////////////////////
// RDATContext

struct RDATContext {
  StringTableReader StringBuffer;
  IndexTableReader IndexTable;
  RawBytesReader RawBytes;
  TableReader Tables[(unsigned)RecordTableIndex::RecordTableCount];
  const TableReader &Table(RecordTableIndex idx) const {
    if (idx < RecordTableIndex::RecordTableCount)
      return Tables[(unsigned)idx];
    return Tables[0]; // TODO: assert
  }
  TableReader &Table(RecordTableIndex idx) {
    return const_cast<TableReader &>(((const RDATContext *)this)->Table(idx));
  }
  template<typename RecordType>
  const TableReader &Table() const {
    static_assert(RecordTraits<RecordType>::TableIndex() < RecordTableIndex::RecordTableCount, "");
    return Table(RecordTraits<RecordType>::TableIndex());
  }
  template<typename RecordType>
  TableReader &Table() {
    return const_cast<TableReader &>(((const RDATContext *)this)->Table(RecordTraits<RecordType>::TableIndex()));
  }
};

///////////////////////////////////////
// Generic Reader Classes

class BaseRecordReader {
protected:
  const RDATContext *m_pContext = nullptr;
  const void *m_pRecord = nullptr;
  uint32_t m_Size = 0;

  template<typename _ReaderTy>
  const _ReaderTy asReader() const {
    if (*this && m_Size >= RecordTraits<typename _ReaderTy::RecordType>::RecordSize())
      return _ReaderTy(*this);
    return {};
  }

  template<typename _T>
  const _T *asRecord() const {
    return static_cast<const _T *>(
        (*this && m_Size >= RecordTraits<_T>::RecordSize()) ? m_pRecord
                                                            : nullptr);
  }

  void InvalidateReader() {
    m_pContext = nullptr;
    m_pRecord = nullptr;
    m_Size = 0;
  }

public:
  BaseRecordReader(const RDATContext *ctx, const void *record, uint32_t size)
    : m_pContext(ctx), m_pRecord(record), m_Size(size) {}
  BaseRecordReader() : BaseRecordReader(nullptr, nullptr, 0) {}

  // Is this a valid reader
  operator bool() const {
    return m_pContext != nullptr && m_pRecord != nullptr && m_Size != 0;
  }
  const RDATContext *GetContext() const { return m_pContext; }
};

template<typename _ReaderTy>
class RecordArrayReader {
  const RDATContext *m_pContext;
  const uint32_t m_IndexOffset;
public:
  RecordArrayReader(const RDATContext *ctx, uint32_t indexOffset)
      : m_pContext(ctx), m_IndexOffset(indexOffset) {
    typedef typename _ReaderTy::RecordType RecordType;
    const TableReader &Table = m_pContext->Table<RecordType>();
    // RecordArrays must be declared with the base record type,
    // with element reader upcast as necessary.
    if (Table.Stride() < RecordTraits<RecordType>::RecordSize())
      InvalidateReader();
  }
  RecordArrayReader() : RecordArrayReader(nullptr, 0) {}
  uint32_t Count() const {
    return *this ? m_pContext->IndexTable.getRow(m_IndexOffset).Count() : 0;
  }
  const _ReaderTy operator[](uint32_t idx) const {
    typedef typename _ReaderTy::RecordType RecordType;
    if (*this) {
      const TableReader &Table = m_pContext->Table<RecordType>();
      return _ReaderTy(BaseRecordReader(
          m_pContext,
          (const void *)Table.Row<RecordType>(
              m_pContext->IndexTable.getRow(m_IndexOffset).At(idx)),
          Table.Stride()));
    }
    return {};
  }
  // Is this a valid reader
  operator bool() const {
    return m_pContext != nullptr && m_IndexOffset < RDAT_NULL_REF;
  }
  void InvalidateReader() { m_pContext = nullptr; }
  const RDATContext *GetContext() const { return m_pContext; }
};

class StringArrayReader {
  const RDATContext *m_pContext;
  const uint32_t m_IndexOffset;
public:
  StringArrayReader(const RDATContext *pContext, uint32_t indexOffset)
    : m_pContext(pContext), m_IndexOffset(indexOffset) {}
  uint32_t Count() const {
    return *this ? m_pContext->IndexTable.getRow(m_IndexOffset).Count() : 0;
  }
  const char *operator[](uint32_t idx) const {
    return *this ? m_pContext->StringBuffer.Get(
                       m_pContext->IndexTable.getRow(m_IndexOffset).At(idx))
                 : 0;
  }
  // Is this a valid reader
  operator bool() const {
    return m_pContext != nullptr && m_IndexOffset < RDAT_NULL_REF;
  }
  void InvalidateReader() { m_pContext = nullptr; }
  const RDATContext *GetContext() const { return m_pContext; }
};

///////////////////////////////////////
// Field Helpers

template<typename _T>
struct RecordRef {
  uint32_t Index;

  template<typename RecordType = _T>
  const _T *Get(const RDATContext &ctx) const {
    return ctx.Table<_T>(). template Row<RecordType>(Index);
  }
  RecordRef &operator =(uint32_t index) { Index = index; return *this; }
  operator uint32_t&() { return Index; }
  operator const uint32_t&() const { return Index; }
  uint32_t *operator &() { return &Index; }
};

template<typename _T>
struct RecordArrayRef {
  uint32_t Index;

  RecordArrayReader<_T> Get(const RDATContext &ctx) const {
    return RecordArrayReader<_T>(ctx.IndexTable, ctx.Table<_T>(), Index);
  }
  RecordArrayRef &operator =(uint32_t index) { Index = index; return *this; }
  operator uint32_t&() { return Index; }
  operator const uint32_t&() const { return Index; }
  uint32_t *operator &() { return &Index; }
};

struct RDATString {
  uint32_t Offset;

  const char *Get(const RDATContext &ctx) const {
    return ctx.StringBuffer.Get(Offset);
  }
  RDATString &operator =(uint32_t offset) { Offset = offset; return *this; }
  operator uint32_t&() { return Offset; }
  operator const uint32_t&() const { return Offset; }
  uint32_t *operator &() { return &Offset; }
};

struct RDATStringArray {
  uint32_t Index;

  StringArrayReader Get(const RDATContext &ctx) const {
    return StringArrayReader(&ctx, Index);
  }
  operator bool() const { return Index == 0 ? false : true; }
  RDATStringArray &operator =(uint32_t index) { Index = index; return *this; }
  operator uint32_t&() { return Index; }
  operator const uint32_t&() const { return Index; }
  uint32_t *operator &() { return &Index; }
};

struct IndexArrayRef {
  uint32_t Index;

  IndexTableReader::IndexRow Get(const RDATContext &ctx) const {
    return ctx.IndexTable.getRow(Index);
  }
  IndexArrayRef &operator =(uint32_t index) { Index = index; return *this; }
  operator uint32_t&() { return Index; }
  operator const uint32_t&() const { return Index; }
  uint32_t *operator &() { return &Index; }
};

struct BytesRef {
  uint32_t Offset;
  uint32_t Size;

  const void *GetBytes(const RDATContext &ctx) const {
    return ctx.RawBytes.Get(Offset);
  }
  template<typename _T>
  const _T *GetAs(const RDATContext &ctx) const {
    return (sizeof(_T) > Size) ? nullptr :
      reinterpret_cast<const _T*>(ctx.RawBytes.Get(Offset));
  }
  uint32_t *operator &() { return &Offset; }
};

struct BytesPtr {
  const void *Ptr = nullptr;
  uint32_t Size = 0;

  BytesPtr(const void *ptr, uint32_t size) :
    Ptr(ptr), Size(size) {}
  BytesPtr() : BytesPtr(nullptr, 0) {}
  template<typename _T>
  const _T *GetAs() const {
    return (sizeof(_T) > Size) ? nullptr : reinterpret_cast<const _T*>(Ptr);
  }
};

///////////////////////////////////////
// Record Helpers

template<typename _RecordReader>
class RecordReader : public BaseRecordReader {
public:
  typedef _RecordReader ThisReaderType;
  RecordReader(const BaseRecordReader &base) : BaseRecordReader(base) {
    typedef typename _RecordReader::RecordType RecordType;
    if ((m_pContext || m_pRecord) && m_Size < RecordTraits<RecordType>::RecordSize())
      InvalidateReader();
  }
  RecordReader() : BaseRecordReader() {}
  template<typename _ReaderType> _ReaderType as() { _ReaderType(*this); }

protected:
  template<typename _FieldRecordReader>
  _FieldRecordReader GetField_RecordValue(const void *pField) const {
    if (*this) {
      return _FieldRecordReader(BaseRecordReader(
          m_pContext, pField, (uint32_t)RecordTraits<typename _FieldRecordReader::RecordType>::RecordSize()));
    }
    return {};
  }
  template<typename _FieldRecordReader>
  _FieldRecordReader GetField_RecordRef(const void *pIndex) const {
    typedef typename _FieldRecordReader::RecordType RecordType;
    if (*this) {
      const TableReader &Table = m_pContext->Table<RecordType>();
      return _FieldRecordReader(BaseRecordReader(
          m_pContext, (const void *)Table.Row<RecordType>(*(const uint32_t*)pIndex),
          Table.Stride()));
    }
    return {};
  }
  template<typename _FieldRecordReader>
  RecordArrayReader<_FieldRecordReader> GetField_RecordArrayRef(const void *pIndex) const {
    if (*this) {
      return RecordArrayReader<_FieldRecordReader>(m_pContext,
                                                   *(const uint32_t *)pIndex);
    }
    return {};
  }
  template<typename _T, typename _StorageTy>
  _T GetField_Value(const _StorageTy *value) const {
    _T result = {};
    if (*this)
      result = (_T)*value;
    return result;
  }
  IndexTableReader::IndexRow GetField_IndexArray(const void *pIndex) const {
    if (*this) {
      return m_pContext->IndexTable.getRow(*(const uint32_t *)pIndex);
    }
    return {};
  }
  // Would use std::array, but don't want this header dependent on that.
  // Array reference syntax is almost enough reason to abandon C++!!!
  template<typename _T, size_t _ArraySize>
  decltype(auto) GetField_ValueArray(_T const(&value)[_ArraySize])const {
    typedef _T ArrayType[_ArraySize];
    if (*this)
      return value;
    return *(const ArrayType*)nullptr;
  }
  const char *GetField_String(const void *pIndex) const {
    return *this ? m_pContext->StringBuffer.Get(*(const uint32_t*)pIndex) : nullptr;
  }
  StringArrayReader GetField_StringArray(const void *pIndex) const {
    return *this ? StringArrayReader(m_pContext, *(const uint32_t *)pIndex)
                 : StringArrayReader(nullptr, 0);
  }
  const void *GetField_Bytes(const void *pIndex) const {
    return *this ? m_pContext->RawBytes.Get(*(const uint32_t*)pIndex) : nullptr;
  }
  uint32_t GetField_BytesSize(const void *pIndex) const {
    return *this ? *(((const uint32_t*)pIndex) + 1) : 0;
  }
};

template<typename _RecordReader>
class RecordTableReader {
  const RDATContext *m_pContext;
public:
  RecordTableReader(const RDATContext *pContext) : m_pContext(pContext) {}
  template<typename RecordReaderType = _RecordReader>
  RecordReaderType Row(uint32_t index) const {
    typedef typename _RecordReader::RecordType RecordType;
    const TableReader &Table = m_pContext->Table<RecordType>();
    return RecordReaderType(BaseRecordReader(
        m_pContext, Table.Row<RecordType>(index), Table.Stride()));
  }
  uint32_t Count() const {
    return m_pContext->Table<typename _RecordReader::RecordType>().Count();
  }
  uint32_t size() const { return Count(); }
  const _RecordReader operator[](uint32_t index) const { return Row(index); }
  operator bool() { return m_pContext && Count(); }
};


/////////////////////////////
// All RDAT enums and types

#define DEF_RDAT_ENUMS DEF_RDAT_ENUM_CLASS
#define DEF_RDAT_TYPES DEF_RDAT_TYPES_FORWARD_DECL
#include "dxc/DxilContainer/RDAT_Macros.inl"

#define DEF_RDAT_TYPES DEF_RDAT_TYPES_USE_HELPERS
#include "dxc/DxilContainer/RDAT_Macros.inl"

#define DEF_RDAT_TYPES DEF_RDAT_TRAITS
#include "dxc/DxilContainer/RDAT_Macros.inl"

#define DEF_RDAT_TYPES DEF_RDAT_READER_DECL
#include "dxc/DxilContainer/RDAT_Macros.inl"

/////////////////////////////
/////////////////////////////

class DxilRuntimeData {
private:
  RDATContext m_Context;
  size_t m_DataSize = 0;

public:
  DxilRuntimeData();
  DxilRuntimeData(const void *ptr, size_t size);
  // initializing reader from RDAT. return true if no error has occured.
  bool InitFromRDAT(const void *pRDAT, size_t size);

  // Make sure data is well formed.
  bool Validate();

  size_t GetDataSize() const { return m_DataSize; }

  RDATContext &GetContext() { return m_Context; }
  const RDATContext &GetContext() const { return m_Context; }

#define RDAT_STRUCT_TABLE(type, table)                                         \
  RecordTableReader<type##_Reader> Get##table() const {                        \
    return RecordTableReader<type##_Reader>(&m_Context);                       \
  }
#define DEF_RDAT_TYPES DEF_RDAT_DEFAULTS
#include "dxc/DxilContainer/RDAT_Macros.inl"

};


//////////////////////////////////
/// structures for library runtime

struct DxilResourceDesc {
  uint32_t Class; // hlsl::DXIL::ResourceClass
  uint32_t Kind;  // hlsl::DXIL::ResourceKind
  uint32_t ID;    // id per class
  uint32_t Space;
  uint32_t UpperBound;
  uint32_t LowerBound;
  LPCWSTR Name;
  uint32_t Flags; // hlsl::RDAT::DxilResourceFlag
};

typedef const DxilResourceDesc *const *DxilResourceDescPtrArray;

struct DxilFunctionDesc {
  LPCWSTR Name;
  LPCWSTR UnmangledName;
  uint32_t NumResources;
  uint32_t NumFunctionDependencies;
  DxilResourceDescPtrArray Resources;
  const LPCWSTR *FunctionDependencies;
  DXIL::ShaderKind ShaderKind;
  uint32_t PayloadSizeInBytes;   // 1) hit, miss, or closest shader: payload count
                                 // 2) call shader: parameter size
  uint32_t AttributeSizeInBytes; // attribute size for closest hit and any hit
  uint32_t FeatureInfo1;         // first 32 bits of feature flag
  uint32_t FeatureInfo2;         // second 32 bits of feature flag
  uint32_t ShaderStageFlag;      // valid shader stage flag.
  uint32_t MinShaderTarget;      // minimum shader target.
};

struct DxilSubobjectDesc {
  LPCWSTR Name;
  DXIL::SubobjectKind Kind;         // D3D12_STATE_SUBOBJECT_TYPE

  struct StateObjectConfig_t {
    uint32_t Flags;   // DXIL::StateObjectFlags / D3D12_STATE_OBJECT_FLAGS
  };
  struct RootSignature_t {
    LPCVOID pSerializedSignature;
    uint32_t SizeInBytes;
  };    // GlobalRootSignature or LocalRootSignature
  struct SubobjectToExportsAssociation_t {
    LPCWSTR Subobject;
    uint32_t NumExports;
    const LPCWSTR* Exports;
  };
  struct RaytracingShaderConfig_t {
    uint32_t MaxPayloadSizeInBytes;
    uint32_t MaxAttributeSizeInBytes;
  };
  struct RaytracingPipelineConfig_t {
    uint32_t MaxTraceRecursionDepth;
  };
  struct HitGroup_t {
    DXIL::HitGroupType Type;        // D3D12_HIT_GROUP_TYPE
    LPCWSTR AnyHit;
    LPCWSTR ClosestHit;
    LPCWSTR Intersection;
  };

  struct RaytracingPipelineConfig1_t {
    uint32_t MaxTraceRecursionDepth;
    uint32_t Flags; // DXIL::RaytracingPipelineFlags / D3D12_RAYTRACING_PIPELINE_FLAGS
  };

  union {
    StateObjectConfig_t StateObjectConfig;
    RootSignature_t RootSignature;    // GlobalRootSignature or LocalRootSignature
    SubobjectToExportsAssociation_t SubobjectToExportsAssociation;
    RaytracingShaderConfig_t RaytracingShaderConfig;
    RaytracingPipelineConfig_t RaytracingPipelineConfig;
    HitGroup_t HitGroup;
    RaytracingPipelineConfig1_t RaytracingPipelineConfig1;
  };
};

struct DxilLibraryDesc {
  uint32_t NumFunctions;
  DxilFunctionDesc *pFunction;
  uint32_t NumResources;
  DxilResourceDesc *pResource;
  uint32_t NumSubobjects;
  DxilSubobjectDesc *pSubobjects;
};

class DxilRuntimeReflection {
public:
  virtual ~DxilRuntimeReflection() {}
  // This call will allocate memory for GetLibraryReflection call
  virtual bool InitFromRDAT(const void *pRDAT, size_t size) = 0;
  // DxilRuntimeReflection owns the memory pointed to by DxilLibraryDesc
  virtual const DxilLibraryDesc GetLibraryReflection() = 0;
};

DxilRuntimeReflection *CreateDxilRuntimeReflection();

} // namespace RDAT
} // namespace hlsl
