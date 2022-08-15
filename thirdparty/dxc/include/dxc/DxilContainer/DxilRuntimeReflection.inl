///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilRuntimeReflection.inl                                                 //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Defines shader reflection for runtime usage.                              //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/DxilContainer/DxilRuntimeReflection.h"
#include <unordered_map>
#include <vector>
#include <memory>
#include <cwchar>

namespace hlsl {
namespace RDAT {

#define DEF_RDAT_TYPES DEF_RDAT_READER_IMPL
#include "dxc/DxilContainer/RDAT_Macros.inl"

struct ResourceKey {
  uint32_t Class, ID;
  ResourceKey(uint32_t Class, uint32_t ID) : Class(Class), ID(ID) {}
  bool operator==(const ResourceKey& other) const {
    return other.Class == Class && other.ID == ID;
  }
};

// Size-checked reader
//  on overrun: throw buffer_overrun{};
//  on overlap: throw buffer_overlap{};
class CheckedReader {
  const char *Ptr;
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

  CheckedReader(const void *ptr, size_t size) :
    Ptr(reinterpret_cast<const char*>(ptr)), Size(size), Offset(0) {}
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
  const T *Cast(size_t size = 0) {
    if (0 == size) size = sizeof(T);
    CheckBounds(size);
    return reinterpret_cast<const T*>(Ptr + Offset);
  }
  template <typename T>
  const T &Read() {
    const size_t size = sizeof(T);
    const T* p = Cast<T>(size);
    Offset += size;
    return *p;
  }
  template <typename T>
  const T *ReadArray(size_t count = 1) {
    const size_t size = sizeof(T) * count;
    const T* p = Cast<T>(size);
    Offset += size;
    return p;
  }
};

DxilRuntimeData::DxilRuntimeData() : DxilRuntimeData(nullptr, 0) {}

DxilRuntimeData::DxilRuntimeData(const void *ptr, size_t size) {
  InitFromRDAT(ptr, size);
}

static void InitTable(RDATContext &ctx, CheckedReader &PR, RecordTableIndex tableIndex) {
  RuntimeDataTableHeader table = PR.Read<RuntimeDataTableHeader>();
  size_t tableSize = table.RecordCount * table.RecordStride;
  ctx.Table(tableIndex)
      .Init(PR.ReadArray<char>(tableSize), table.RecordCount,
            table.RecordStride);
}

// initializing reader from RDAT. return true if no error has occured.
bool DxilRuntimeData::InitFromRDAT(const void *pRDAT, size_t size) {
  if (pRDAT) {
    m_DataSize = size;
    try {
      CheckedReader Reader(pRDAT, size);
      RuntimeDataHeader RDATHeader = Reader.Read<RuntimeDataHeader>();
      if (RDATHeader.Version < RDAT_Version_10) {
        return false;
      }
      const uint32_t *offsets = Reader.ReadArray<uint32_t>(RDATHeader.PartCount);
      for (uint32_t i = 0; i < RDATHeader.PartCount; ++i) {
        Reader.Advance(offsets[i]);
        RuntimeDataPartHeader part = Reader.Read<RuntimeDataPartHeader>();
        CheckedReader PR(Reader.ReadArray<char>(part.Size), part.Size);
        switch (part.Type) {
        case RuntimeDataPartType::StringBuffer: {
          m_Context.StringBuffer.Init(PR.ReadArray<char>(part.Size), part.Size);
          break;
        }
        case RuntimeDataPartType::IndexArrays: {
          uint32_t count = part.Size / sizeof(uint32_t);
          m_Context.IndexTable.Init(PR.ReadArray<uint32_t>(count), count);
          break;
        }
        case RuntimeDataPartType::RawBytes: {
          m_Context.RawBytes.Init(PR.ReadArray<char>(part.Size), part.Size);
          break;
        }

// Once per table.
#define RDAT_STRUCT_TABLE(type, table) case RuntimeDataPartType::table: InitTable(m_Context, PR, RecordTableIndex::table); break;
#define DEF_RDAT_TYPES DEF_RDAT_DEFAULTS
#include "dxc/DxilContainer/RDAT_Macros.inl"

        default:
          continue; // Skip unrecognized parts
        }
      }
#ifndef NDEBUG
      return Validate();
#else  // NDEBUG
      return true;
#endif // NDEBUG
    } catch(CheckedReader::exception e) {
      // TODO: error handling
      //throw hlsl::Exception(DXC_E_MALFORMED_CONTAINER, e.what());
      return false;
    }
  }
  m_DataSize = 0;
  return false;
}

// PRERELEASE-TODO: Incorporate field names and report errors in error stream

// PRERELEASE-TODO: Low-pri: Check other things like that all the index, string,
// and binary buffer space is actually used.

template<typename _RecordType>
static bool ValidateRecordRef(const RDATContext &ctx, uint32_t id) {
  if (id == RDAT_NULL_REF)
    return true;
  // id should be a valid index into the appropriate table
  auto &table = ctx.Table(RecordTraits<_RecordType>::TableIndex());
  if (id >= table.Count())
    return false;
  return true;
}

static bool ValidateIndexArrayRef(const RDATContext &ctx, uint32_t id) {
  if (id == RDAT_NULL_REF)
    return true;
  uint32_t size = ctx.IndexTable.Count();
  // check that id < size of index array
  if (id >= size)
    return false;
  // check that array size fits in remaining index space
  if (id + ctx.IndexTable.Data()[id] >= size)
    return false;
  return true;
}

template<typename _RecordType>
static bool ValidateRecordArrayRef(const RDATContext &ctx, uint32_t id) {
  // Make sure index array is well-formed
  if (!ValidateIndexArrayRef(ctx, id))
    return false;
  // Make sure each record id is a valid record ref in the table
  auto ids = ctx.IndexTable.getRow(id);
  for (unsigned i = 0; i < ids.Count(); i++) {
    if (!ValidateRecordRef<_RecordType>(ctx, ids.At(i)))
      return false;
  }
  return true;
}
static bool ValidateStringRef(const RDATContext &ctx, uint32_t id) {
  if (id == RDAT_NULL_REF)
    return true;
  uint32_t size = ctx.StringBuffer.Size();
  if (id >= size)
    return false;
  return true;
}
static bool ValidateStringArrayRef(const RDATContext &ctx, uint32_t id) {
  if (id == RDAT_NULL_REF)
    return true;
  // Make sure index array is well-formed
  if (!ValidateIndexArrayRef(ctx, id))
    return false;
  // Make sure each index is valid in string buffer
  auto ids = ctx.IndexTable.getRow(id);
  for (unsigned i = 0; i < ids.Count(); i++) {
    if (!ValidateStringRef(ctx, ids.At(i)))
      return false;
  }
  return true;
}

// Specialized for each record type
template<typename _RecordType>
bool ValidateRecord(const RDATContext &ctx, const _RecordType *pRecord) {
  return false;
}

#define DEF_RDAT_TYPES DEF_RDAT_STRUCT_VALIDATION
#include "dxc/DxilContainer/RDAT_Macros.inl"

// This class ensures that all versions of record to latest one supported by
// table stride are validated
class RecursiveRecordValidator {
  const hlsl::RDAT::RDATContext &m_Context;
  uint32_t m_RecordStride;
public:
  RecursiveRecordValidator(const hlsl::RDAT::RDATContext &ctx, uint32_t recordStride)
      : m_Context(ctx), m_RecordStride(recordStride) {}
  template<typename _RecordType>
  bool Validate(const _RecordType *pRecord) const {
    if (pRecord && sizeof(_RecordType) <= m_RecordStride) {
      if (!ValidateRecord(m_Context, pRecord))
        return false;
      return ValidateDerived<_RecordType>(pRecord);
    }
    return true;
  }
  // Specialized for base type to recurse into derived
  template<typename _RecordType>
  bool ValidateDerived(const _RecordType *) const { return true; }
};

template<typename _RecordType>
static bool ValidateRecordTable(RDATContext &ctx, RecordTableIndex tableIndex) {
  // iterate through records, bounds-checking all refs and index arrays
  auto &table = ctx.Table(tableIndex);
  for (unsigned i = 0; i < table.Count(); i++) {
    RecursiveRecordValidator(ctx, table.Stride())
        .Validate<_RecordType>(table.Row<_RecordType>(i));
  }
  return true;
}

bool DxilRuntimeData::Validate() {
  if (m_Context.StringBuffer.Size()) {
    if (m_Context.StringBuffer.Data()[m_Context.StringBuffer.Size() - 1] != 0)
      return false;
  }

  // Once per table.
#define RDAT_STRUCT_TABLE(type, table) ValidateRecordTable<type>(m_Context, RecordTableIndex::table);
  // As an assumption of the way record types are versioned, derived record
  // types must always be larger than base record types.
#define RDAT_STRUCT_DERIVED(type, base)                                        \
  static_assert(sizeof(type) > sizeof(base),                                   \
                "otherwise, derived record type " #type                        \
                " is not larger than base record type " #base ".");
#define DEF_RDAT_TYPES DEF_RDAT_DEFAULTS
#include "dxc/DxilContainer/RDAT_Macros.inl"
  return true;
}

}} // hlsl::RDAT

using namespace hlsl;
using namespace RDAT;

namespace std {
template <> struct hash<ResourceKey> {
  size_t operator()(const ResourceKey &key) const throw() {
    return (hash<uint32_t>()(key.Class) * (size_t)16777619U) ^
           hash<uint32_t>()(key.ID);
  }
};
} // namespace std

namespace {

class DxilRuntimeReflection_impl : public DxilRuntimeReflection {
private:
  typedef std::unordered_map<const char *, std::unique_ptr<wchar_t[]>> StringMap;
  typedef std::unordered_map<const void *, std::unique_ptr<char[]>> BytesMap;
  typedef std::vector<const wchar_t *> WStringList;
  typedef std::vector<DxilResourceDesc> ResourceList;
  typedef std::vector<DxilResourceDesc *> ResourceRefList;
  typedef std::vector<DxilFunctionDesc> FunctionList;
  typedef std::vector<DxilSubobjectDesc> SubobjectList;

  DxilRuntimeData m_RuntimeData;
  StringMap m_StringMap;
  BytesMap m_BytesMap;
  std::vector<uint32_t> m_IndexData;
  ResourceList m_Resources;
  FunctionList m_Functions;
  SubobjectList m_Subobjects;
  std::unordered_map<ResourceKey, DxilResourceDesc *> m_ResourceMap;
  std::unordered_map<DxilFunctionDesc *, ResourceRefList> m_FuncToResMap;
  std::unordered_map<DxilFunctionDesc *, WStringList> m_FuncToDependenciesMap;
  std::unordered_map<DxilSubobjectDesc *, WStringList> m_SubobjectToExportsMap;
  bool m_initialized;

  const wchar_t *GetWideString(const char *ptr);
  void AddString(const char *ptr);
  const void *GetBytes(const void *ptr, size_t size);
  void InitializeReflection();
  const DxilResourceDesc * const*GetResourcesForFunction(DxilFunctionDesc &function,
                             const RuntimeDataFunctionInfo_Reader &functionReader);
  const wchar_t **GetDependenciesForFunction(DxilFunctionDesc &function,
                             const RuntimeDataFunctionInfo_Reader &functionReader);
  const wchar_t **GetExportsForAssociation(DxilSubobjectDesc &subobject,
                             const RuntimeDataSubobjectInfo_Reader &reader);
  void AddResources();
  void AddFunctions();
  void AddSubobjects();

public:
  // TODO: Implement pipeline state validation with runtime data
  // TODO: Update BlobContainer.h to recognize 'RDAT' blob
  DxilRuntimeReflection_impl()
      : m_RuntimeData(), m_StringMap(), m_BytesMap(), m_Resources(), m_Functions(),
        m_FuncToResMap(), m_FuncToDependenciesMap(), m_SubobjectToExportsMap(),
        m_initialized(false) {}
  virtual ~DxilRuntimeReflection_impl() {}
  // This call will allocate memory for GetLibraryReflection call
  bool InitFromRDAT(const void *pRDAT, size_t size) override;
  const DxilLibraryDesc GetLibraryReflection() override;
};

void DxilRuntimeReflection_impl::AddString(const char *ptr) {
  if (m_StringMap.find(ptr) == m_StringMap.end()) {
    auto state = std::mbstate_t();
    size_t size = std::mbsrtowcs(nullptr, &ptr, 0, &state);
    if (size != static_cast<size_t>(-1)) {
      std::unique_ptr<wchar_t[]> pNew(new wchar_t[size + 1]);
      auto pOldPtr = ptr;
      std::mbsrtowcs(pNew.get(), &ptr, size + 1, &state);
      m_StringMap[pOldPtr] = std::move(pNew);
    }
  }
}

const wchar_t *DxilRuntimeReflection_impl::GetWideString(const char *ptr) {
  if (m_StringMap.find(ptr) == m_StringMap.end()) {
    AddString(ptr);
  }
  return m_StringMap.at(ptr).get();
}

const void *DxilRuntimeReflection_impl::GetBytes(const void *ptr, size_t size) {
  if (!ptr || !size)
    return nullptr;

  auto it = m_BytesMap.find(ptr);
  if (it != m_BytesMap.end())
    return it->second.get();

  auto inserted = m_BytesMap.insert(std::make_pair(ptr, std::unique_ptr<char[]>(new char[size])));
  void *newPtr = inserted.first->second.get();
  memcpy(newPtr, ptr, size);
  return newPtr;
}

bool DxilRuntimeReflection_impl::InitFromRDAT(const void *pRDAT, size_t size) {
  assert(!m_initialized && "may only initialize once");
  m_initialized = m_RuntimeData.InitFromRDAT(pRDAT, size);
  if (!m_RuntimeData.Validate())
    return false;
  if (m_initialized)
    InitializeReflection();
  return m_initialized;
}

const DxilLibraryDesc DxilRuntimeReflection_impl::GetLibraryReflection() {
  DxilLibraryDesc reflection = {};
  if (m_initialized) {
    reflection.NumResources = m_Resources.size();
    reflection.pResource = m_Resources.data();
    reflection.NumFunctions = m_Functions.size();
    reflection.pFunction = m_Functions.data();
    reflection.NumSubobjects = m_Subobjects.size();
    reflection.pSubobjects = m_Subobjects.data();
  }
  return reflection;
}

void DxilRuntimeReflection_impl::InitializeReflection() {
  auto indexTable = m_RuntimeData.GetContext().IndexTable;
  m_IndexData.assign(indexTable.Data(), indexTable.Data() + indexTable.Count());

  // First need to reserve spaces for resources because functions will need to
  // reference them via pointers.
  AddResources();
  AddFunctions();
  AddSubobjects();
}

void DxilRuntimeReflection_impl::AddResources() {
  auto table = m_RuntimeData.GetResourceTable();
  m_Resources.assign(table.Count(), {});
  for (uint32_t i = 0; i < table.Count(); ++i) {
    auto reader = table[i];
    DxilResourceDesc &desc = m_Resources[i];
    desc.Class = (uint32_t)reader.getClass();
    desc.Kind = (uint32_t)reader.getKind();
    desc.Space = reader.getSpace();
    desc.LowerBound = reader.getLowerBound();
    desc.UpperBound = reader.getUpperBound();
    desc.ID = reader.getID();
    desc.Flags = reader.getFlags();
    desc.Name = GetWideString(reader.getName());
    ResourceKey key(desc.Class, desc.ID);
    m_ResourceMap[key] = &desc;
  }
}

const DxilResourceDesc * const*DxilRuntimeReflection_impl::GetResourcesForFunction(
    DxilFunctionDesc &function, const RuntimeDataFunctionInfo_Reader &functionReader) {
  auto resources = functionReader.getResources();
  if (!resources.Count())
    return nullptr;
  auto it = m_FuncToResMap.insert(std::make_pair(&function, ResourceRefList()));
  assert(it.second && "otherwise, collision");
  ResourceRefList &resourceList = it.first->second;
  resourceList.reserve(resources.Count());
  for (uint32_t i = 0; i < resources.Count(); ++i) {
    auto resourceReader = functionReader.getResources()[i];
    ResourceKey key((uint32_t)resourceReader.getClass(),
                    resourceReader.getID());
    auto it = m_ResourceMap.find(key);
    assert(it != m_ResourceMap.end() && it->second && "Otherwise, resource was not in map, or was null");
    resourceList.emplace_back(it->second);
  }
  return resourceList.data();
}

const wchar_t **DxilRuntimeReflection_impl::GetDependenciesForFunction(
    DxilFunctionDesc &function, const RuntimeDataFunctionInfo_Reader &functionReader) {
  auto it = m_FuncToDependenciesMap.insert(std::make_pair(&function, WStringList()));
  assert(it.second && "otherwise, collision");
  WStringList &wStringList = it.first->second;
  auto dependencies = functionReader.getFunctionDependencies();
  for (uint32_t i = 0; i < dependencies.Count(); ++i) {
    wStringList.emplace_back(GetWideString(dependencies[i]));
  }
  return wStringList.empty() ? nullptr : wStringList.data();
}

void DxilRuntimeReflection_impl::AddFunctions() {
  auto table = m_RuntimeData.GetFunctionTable();
  m_Functions.assign(table.Count(), {});
  for (uint32_t i = 0; i < table.Count(); ++i) {
    auto reader = table[i];
    auto &desc = m_Functions[i];
    desc.Name = GetWideString(reader.getName());
    desc.UnmangledName = GetWideString(reader.getUnmangledName());
    desc.NumResources = reader.getResources().Count();
    desc.Resources = GetResourcesForFunction(desc, reader);
    desc.NumFunctionDependencies = reader.getFunctionDependencies().Count();
    desc.FunctionDependencies = GetDependenciesForFunction(desc, reader);
    desc.ShaderKind = reader.getShaderKind();
    desc.PayloadSizeInBytes = reader.getPayloadSizeInBytes();
    desc.AttributeSizeInBytes = reader.getAttributeSizeInBytes();
    desc.FeatureInfo1 = reader.getFeatureInfo1();
    desc.FeatureInfo2 = reader.getFeatureInfo2();
    desc.ShaderStageFlag = reader.getShaderStageFlag();
    desc.MinShaderTarget = reader.getMinShaderTarget();
  }
}

const wchar_t **DxilRuntimeReflection_impl::GetExportsForAssociation(
  DxilSubobjectDesc &subobject, const RuntimeDataSubobjectInfo_Reader &reader) {
  auto it = m_SubobjectToExportsMap.insert(std::make_pair(&subobject, WStringList()));
  assert(it.second && "otherwise, collision");
  auto exports = reader.getSubobjectToExportsAssociation().getExports();
  WStringList &wStringList = it.first->second;
  for (uint32_t i = 0; i < exports.Count(); ++i) {
    wStringList.emplace_back(GetWideString(exports[i]));
  }
  return wStringList.empty() ? nullptr : wStringList.data();
}

void DxilRuntimeReflection_impl::AddSubobjects() {
  auto table = m_RuntimeData.GetSubobjectTable();
  m_Subobjects.assign(table.Count(), {});
  for (uint32_t i = 0; i < table.Count(); ++i) {
    auto reader = table[i];
    auto &desc = m_Subobjects[i];
    desc.Name = GetWideString(reader.getName());
    desc.Kind = reader.getKind();
    switch (reader.getKind()) {
    case DXIL::SubobjectKind::StateObjectConfig:
      desc.StateObjectConfig.Flags = reader.getStateObjectConfig().getFlags();
      break;
    case DXIL::SubobjectKind::GlobalRootSignature:
    case DXIL::SubobjectKind::LocalRootSignature:
      desc.RootSignature.SizeInBytes = reader.getRootSignature().sizeData();
      desc.RootSignature.pSerializedSignature = GetBytes(reader.getRootSignature().getData(), desc.RootSignature.SizeInBytes);
      break;
    case DXIL::SubobjectKind::SubobjectToExportsAssociation:
      desc.SubobjectToExportsAssociation.Subobject =
        GetWideString(reader.getSubobjectToExportsAssociation().getSubobject());
      desc.SubobjectToExportsAssociation.NumExports = reader.getSubobjectToExportsAssociation().getExports().Count();
      desc.SubobjectToExportsAssociation.Exports = GetExportsForAssociation(desc, reader);
      break;
    case DXIL::SubobjectKind::RaytracingShaderConfig:
      desc.RaytracingShaderConfig.MaxPayloadSizeInBytes = reader.getRaytracingShaderConfig().getMaxPayloadSizeInBytes();
      desc.RaytracingShaderConfig.MaxAttributeSizeInBytes = reader.getRaytracingShaderConfig().getMaxAttributeSizeInBytes();
      break;
    case DXIL::SubobjectKind::RaytracingPipelineConfig:
      desc.RaytracingPipelineConfig.MaxTraceRecursionDepth = reader.getRaytracingPipelineConfig().getMaxTraceRecursionDepth();
      break;
    case DXIL::SubobjectKind::HitGroup:
      desc.HitGroup.Type = reader.getHitGroup().getType();
      desc.HitGroup.Intersection = GetWideString(reader.getHitGroup().getIntersection());
      desc.HitGroup.AnyHit = GetWideString(reader.getHitGroup().getAnyHit());
      desc.HitGroup.ClosestHit = GetWideString(reader.getHitGroup().getClosestHit());
      break;
    case DXIL::SubobjectKind::RaytracingPipelineConfig1:
      desc.RaytracingPipelineConfig1.MaxTraceRecursionDepth = reader.getRaytracingPipelineConfig1().getMaxTraceRecursionDepth();
      desc.RaytracingPipelineConfig1.Flags = reader.getRaytracingPipelineConfig1().getFlags();
      break;
    default:
      // Ignore contents of unrecognized subobject type (forward-compat)
      break;
    }
  }
}

} // namespace anon

DxilRuntimeReflection *hlsl::RDAT::CreateDxilRuntimeReflection() {
  return new DxilRuntimeReflection_impl();
}
