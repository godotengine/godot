///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilSubobject.h                                                           //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Defines Subobject types for DxilModule.                                   //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <vector>
#include <memory>
#include <unordered_map>
#include <map>
#include "DxilConstants.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringRef.h"

namespace hlsl {

class DxilSubobjects;

namespace RDAT {
  class DxilRuntimeData;
}

class DxilSubobject {
public:
  using Kind = DXIL::SubobjectKind;

  DxilSubobject() = delete;
  DxilSubobject(const DxilSubobject &other) = delete;
  DxilSubobject(DxilSubobject &&other);
  ~DxilSubobject();

  DxilSubobject &operator=(const DxilSubobject &other) = delete;

  Kind GetKind() const { return m_Kind; }
  llvm::StringRef GetName() const { return m_Name; }

  // Note: strings and root signature data is owned by DxilModule
  // When creating subobjects, use canonical strings from module


  bool GetStateObjectConfig(uint32_t &Flags) const;
  bool GetRootSignature(bool local, const void * &Data, uint32_t &Size, 
                        const char **pText = nullptr) const;
  bool GetSubobjectToExportsAssociation(llvm::StringRef &Subobject,
                                        const char * const * &Exports,
                                        uint32_t &NumExports) const;
  bool GetRaytracingShaderConfig(uint32_t &MaxPayloadSizeInBytes,
                                 uint32_t &MaxAttributeSizeInBytes) const;
  bool GetRaytracingPipelineConfig(uint32_t &MaxTraceRecursionDepth) const;
  bool GetRaytracingPipelineConfig1(uint32_t &MaxTraceRecursionDepth, uint32_t &Flags) const;
  bool GetHitGroup(DXIL::HitGroupType &hitGroupType,
                   llvm::StringRef &AnyHit,
                   llvm::StringRef &ClosestHit,
                   llvm::StringRef &Intersection) const;

private:
  DxilSubobject(DxilSubobjects &owner, Kind kind, llvm::StringRef name);
  DxilSubobject(DxilSubobjects &owner, const DxilSubobject &other, llvm::StringRef name);
  void CopyUnionedContents(const DxilSubobject &other);
  void InternStrings();

  DxilSubobjects &m_Owner;
  Kind m_Kind;
  llvm::StringRef m_Name;

  std::vector<const char*> m_Exports;

  struct StateObjectConfig_t {
    uint32_t Flags;   // DXIL::StateObjectFlags
  };
  struct RootSignature_t {
    uint32_t Size;
    const void *Data;
    const char *Text; // can be null
  };
  struct SubobjectToExportsAssociation_t {
    const char *Subobject;
    // see m_Exports for export list
  };
  struct RaytracingShaderConfig_t {
    uint32_t MaxPayloadSizeInBytes;
    uint32_t MaxAttributeSizeInBytes;
  };
  struct RaytracingPipelineConfig_t {
    uint32_t MaxTraceRecursionDepth;
  };
  struct RaytracingPipelineConfig1_t {
    uint32_t MaxTraceRecursionDepth;
    uint32_t Flags; // DXIL::RaytracingPipelineFlags
  };
  struct HitGroup_t {
    DXIL::HitGroupType Type;
    const char *AnyHit;
    const char *ClosestHit;
    const char *Intersection;
  };

  union {
    StateObjectConfig_t StateObjectConfig;
    RootSignature_t RootSignature;
    SubobjectToExportsAssociation_t SubobjectToExportsAssociation;
    RaytracingShaderConfig_t RaytracingShaderConfig;
    RaytracingPipelineConfig_t RaytracingPipelineConfig;
    HitGroup_t HitGroup;
    RaytracingPipelineConfig1_t RaytracingPipelineConfig1;
  };

  friend class DxilSubobjects;
};

class DxilSubobjects {
public:
  typedef std::pair<std::unique_ptr<char[]>, size_t> StoredBytes;
  typedef llvm::MapVector< llvm::StringRef, StoredBytes > BytesStorage;
  typedef llvm::MapVector< llvm::StringRef, std::unique_ptr<DxilSubobject> > SubobjectStorage;
  using Kind = DXIL::SubobjectKind;

  DxilSubobjects();
  DxilSubobjects(const DxilSubobjects &other) = delete;
  DxilSubobjects(DxilSubobjects &&other);
  ~DxilSubobjects();

  DxilSubobjects &operator=(const DxilSubobjects &other) = delete;

  // Add/find string in owned subobject strings, returning canonical ptr
  llvm::StringRef InternString(llvm::StringRef value);
  // Add/find raw bytes, returning canonical ptr
  const void *InternRawBytes(const void *ptr, size_t size);
  DxilSubobject *FindSubobject(llvm::StringRef name);
  void RemoveSubobject(llvm::StringRef name);
  DxilSubobject &CloneSubobject(const DxilSubobject &Subobject, llvm::StringRef Name);
  const SubobjectStorage &GetSubobjects() const { return m_Subobjects;  }

  // Create DxilSubobjects

  DxilSubobject &CreateStateObjectConfig(llvm::StringRef Name,
                                         uint32_t Flags);
  // Local/Global RootSignature
  DxilSubobject &CreateRootSignature(llvm::StringRef Name,
                                     bool local,
                                     const void *Data,
                                     uint32_t Size,
                                     llvm::StringRef *pText = nullptr);
  DxilSubobject &CreateSubobjectToExportsAssociation(
    llvm::StringRef Name,
    llvm::StringRef Subobject, llvm::StringRef *Exports, uint32_t NumExports);
  DxilSubobject &CreateRaytracingShaderConfig(
    llvm::StringRef Name,
    uint32_t MaxPayloadSizeInBytes,
    uint32_t MaxAttributeSizeInBytes);
  DxilSubobject &CreateRaytracingPipelineConfig(
    llvm::StringRef Name,
    uint32_t MaxTraceRecursionDepth);
  DxilSubobject &CreateRaytracingPipelineConfig1(
    llvm::StringRef Name,
    uint32_t MaxTraceRecursionDepth,
    uint32_t Flags);
  DxilSubobject &CreateHitGroup(llvm::StringRef Name, 
                                DXIL::HitGroupType hitGroupType,
                                llvm::StringRef AnyHit,
                                llvm::StringRef ClosestHit,
                                llvm::StringRef Intersection);

private:
  DxilSubobject &CreateSubobject(Kind kind, llvm::StringRef Name);

  BytesStorage m_BytesStorage;
  SubobjectStorage m_Subobjects;
};

bool LoadSubobjectsFromRDAT(DxilSubobjects &subobjects, const RDAT::DxilRuntimeData &rdat);

} // namespace hlsl
