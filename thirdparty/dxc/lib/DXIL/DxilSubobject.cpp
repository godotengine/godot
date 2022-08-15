///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilSubobject.cpp                                                         //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/Support/Global.h"
#include "dxc/Support/Unicode.h"
#include "dxc/Support/WinIncludes.h"
#include "dxc/DXIL/DxilSubobject.h"
#include "dxc/DxilContainer/DxilRuntimeReflection.h"
#include "llvm/ADT/STLExtras.h"

namespace hlsl {

//------------------------------------------------------------------------------
//
// Subobject methods.
//

DxilSubobject::DxilSubobject(DxilSubobject &&other)
  : m_Owner(other.m_Owner),
    m_Kind(other.m_Kind),
    m_Name(m_Owner.InternString(other.m_Name)),
    m_Exports(std::move(other.m_Exports))
{
  DXASSERT_NOMSG(DXIL::IsValidSubobjectKind(m_Kind));
  CopyUnionedContents(other);
}

DxilSubobject::DxilSubobject(DxilSubobjects &owner, Kind kind, llvm::StringRef name)
  : m_Owner(owner),
    m_Kind(kind),
    m_Name(m_Owner.InternString(name)),
    m_Exports()
{
  DXASSERT_NOMSG(DXIL::IsValidSubobjectKind(m_Kind));
}

DxilSubobject::DxilSubobject(DxilSubobjects &owner, const DxilSubobject &other, llvm::StringRef name)
  : m_Owner(owner),
    m_Kind(other.m_Kind),
    m_Name(name),
    m_Exports(other.m_Exports.begin(), other.m_Exports.end())
{
  DXASSERT_NOMSG(DXIL::IsValidSubobjectKind(m_Kind));
  CopyUnionedContents(other);
  if (&m_Owner != &other.m_Owner)
    InternStrings();
}

void DxilSubobject::CopyUnionedContents(const DxilSubobject &other) {
  switch (m_Kind) {
  case Kind::StateObjectConfig:
    StateObjectConfig.Flags = other.StateObjectConfig.Flags;
    break;
  case Kind::GlobalRootSignature:
  case Kind::LocalRootSignature:
    RootSignature.Size = other.RootSignature.Size;
    RootSignature.Data = other.RootSignature.Data;
    break;
  case Kind::SubobjectToExportsAssociation:
    SubobjectToExportsAssociation.Subobject = other.SubobjectToExportsAssociation.Subobject;
    break;
  case Kind::RaytracingShaderConfig:
    RaytracingShaderConfig.MaxPayloadSizeInBytes = other.RaytracingShaderConfig.MaxPayloadSizeInBytes;
    RaytracingShaderConfig.MaxAttributeSizeInBytes = other.RaytracingShaderConfig.MaxAttributeSizeInBytes;
    break;
  case Kind::RaytracingPipelineConfig:
    RaytracingPipelineConfig.MaxTraceRecursionDepth = other.RaytracingPipelineConfig.MaxTraceRecursionDepth;
    break;
  case Kind::HitGroup:
    HitGroup.Type = other.HitGroup.Type;
    HitGroup.AnyHit = other.HitGroup.AnyHit;
    HitGroup.ClosestHit = other.HitGroup.ClosestHit;
    HitGroup.Intersection = other.HitGroup.Intersection;
    break;
  case Kind::RaytracingPipelineConfig1:
    RaytracingPipelineConfig1.MaxTraceRecursionDepth = other.RaytracingPipelineConfig1.MaxTraceRecursionDepth;
    RaytracingPipelineConfig1.Flags = other.RaytracingPipelineConfig1.Flags;
    break;
  default:
    DXASSERT(0, "invalid kind");
    break;
  }
}

void DxilSubobject::InternStrings() {
  // Transfer strings if necessary
  m_Name = m_Owner.InternString(m_Name).data();
  switch (m_Kind) {
  case Kind::SubobjectToExportsAssociation:
    SubobjectToExportsAssociation.Subobject = m_Owner.InternString(SubobjectToExportsAssociation.Subobject).data();
    for (auto &ptr : m_Exports)
      ptr = m_Owner.InternString(ptr).data();
    break;
  case Kind::HitGroup:
    HitGroup.AnyHit = m_Owner.InternString(HitGroup.AnyHit).data();
    HitGroup.ClosestHit = m_Owner.InternString(HitGroup.ClosestHit).data();
    HitGroup.Intersection = m_Owner.InternString(HitGroup.Intersection).data();
    break;
  default:
    break;
  }
}

DxilSubobject::~DxilSubobject() {
}


// StateObjectConfig
bool DxilSubobject::GetStateObjectConfig(uint32_t &Flags) const {
  if (m_Kind == Kind::StateObjectConfig) {
    Flags = StateObjectConfig.Flags;
    return true;
  }
  return false;
}

// Local/Global RootSignature
bool DxilSubobject::GetRootSignature(
    bool local, const void * &Data, uint32_t &Size, const char **pText) const {
  Kind expected = local ? Kind::LocalRootSignature : Kind::GlobalRootSignature;
  if (m_Kind == expected) {
    Data = RootSignature.Data;
    Size = RootSignature.Size;
    if (pText) 
      *pText = RootSignature.Text;
    return true;
  }
  return false;
}

// SubobjectToExportsAssociation
bool DxilSubobject::GetSubobjectToExportsAssociation(
    llvm::StringRef &Subobject,
    const char * const * &Exports,
    uint32_t &NumExports) const {
  if (m_Kind == Kind::SubobjectToExportsAssociation) {
    Subobject = SubobjectToExportsAssociation.Subobject;
    Exports = m_Exports.data();
    NumExports = (uint32_t)m_Exports.size();
    return true;
  }
  return false;
}

// RaytracingShaderConfig
bool DxilSubobject::GetRaytracingShaderConfig(uint32_t &MaxPayloadSizeInBytes,
                                              uint32_t &MaxAttributeSizeInBytes) const {
  if (m_Kind == Kind::RaytracingShaderConfig) {
    MaxPayloadSizeInBytes = RaytracingShaderConfig.MaxPayloadSizeInBytes;
    MaxAttributeSizeInBytes = RaytracingShaderConfig.MaxAttributeSizeInBytes;
    return true;
  }
  return false;
}

// RaytracingPipelineConfig
bool DxilSubobject::GetRaytracingPipelineConfig(
    uint32_t &MaxTraceRecursionDepth) const {
  if (m_Kind == Kind::RaytracingPipelineConfig) {
    MaxTraceRecursionDepth = RaytracingPipelineConfig.MaxTraceRecursionDepth;
    return true;
  }
  return false;
}

// HitGroup
bool DxilSubobject::GetHitGroup(DXIL::HitGroupType &hitGroupType, 
                                llvm::StringRef &AnyHit,
                                llvm::StringRef &ClosestHit,
                                llvm::StringRef &Intersection) const {
  if (m_Kind == Kind::HitGroup) {
    hitGroupType = HitGroup.Type;
    AnyHit = HitGroup.AnyHit;
    ClosestHit = HitGroup.ClosestHit;
    Intersection = HitGroup.Intersection;
    return true;
  }
  return false;
}

// RaytracingPipelineConfig1
bool DxilSubobject::GetRaytracingPipelineConfig1(
    uint32_t &MaxTraceRecursionDepth, uint32_t &Flags) const {
  if (m_Kind == Kind::RaytracingPipelineConfig1) {
    MaxTraceRecursionDepth = RaytracingPipelineConfig1.MaxTraceRecursionDepth;
    Flags = RaytracingPipelineConfig1.Flags;
    return true;
  }
  return false;
}
 
DxilSubobjects::DxilSubobjects()
  : m_BytesStorage()
  , m_Subobjects()
{}
DxilSubobjects::DxilSubobjects(DxilSubobjects &&other)
  : m_BytesStorage(std::move(other.m_BytesStorage))
  , m_Subobjects(std::move(other.m_Subobjects))
{}
DxilSubobjects::~DxilSubobjects() {}


llvm::StringRef DxilSubobjects::InternString(llvm::StringRef value) {
  auto it = m_BytesStorage.find(value);
  if (it != m_BytesStorage.end())
    return it->first;

  size_t size = value.size();
  StoredBytes stored(std::make_pair(std::unique_ptr<char[]>(new char[size + 1]), size + 1));
  memcpy(stored.first.get(), value.data(), size);
  stored.first[size] = 0;
  llvm::StringRef key(stored.first.get(), size);
  m_BytesStorage[key] = std::move(stored);
  return key;
}

const void *DxilSubobjects::InternRawBytes(const void *ptr, size_t size) {
  auto it = m_BytesStorage.find(llvm::StringRef((const char *)ptr, size));
  if (it != m_BytesStorage.end())
    return it->first.data();

  StoredBytes stored(std::make_pair(std::unique_ptr<char[]>(new char[size]), size));
  memcpy(stored.first.get(), ptr, size);
  llvm::StringRef key(stored.first.get(), size);
  m_BytesStorage[key] = std::move(stored);
  return key.data();
}

DxilSubobject *DxilSubobjects::FindSubobject(llvm::StringRef name) {
  auto it = m_Subobjects.find(name);
  if (it != m_Subobjects.end())
    return it->second.get();
  return nullptr;
}

void DxilSubobjects::RemoveSubobject(llvm::StringRef name) {
  auto it = m_Subobjects.find(name);
  if (it != m_Subobjects.end())
    m_Subobjects.erase(it);
}

DxilSubobject &DxilSubobjects::CloneSubobject(
    const DxilSubobject &Subobject, llvm::StringRef Name) {
  Name = InternString(Name);
  DXASSERT(FindSubobject(Name) == nullptr,
    "otherwise, name collision between subobjects");
  std::unique_ptr<DxilSubobject> ptr(new DxilSubobject(*this, Subobject, Name));
  DxilSubobject &ref = *ptr;
  m_Subobjects[Name] = std::move(ptr);
  return ref;
}

// Create DxilSubobjects

DxilSubobject &DxilSubobjects::CreateStateObjectConfig(
    llvm::StringRef Name, uint32_t Flags) {
  DXASSERT_NOMSG(0 == ((~(uint32_t)DXIL::StateObjectFlags::ValidMask) & Flags));
  auto &obj = CreateSubobject(Kind::StateObjectConfig, Name);
  obj.StateObjectConfig.Flags = Flags;
  return obj;
}

DxilSubobject &DxilSubobjects::CreateRootSignature(
    llvm::StringRef Name, bool local, const void *Data, uint32_t Size, llvm::StringRef *pText /*= nullptr*/) {
  auto &obj = CreateSubobject(local ? Kind::LocalRootSignature : Kind::GlobalRootSignature, Name);
  obj.RootSignature.Data = InternRawBytes(Data, Size);
  obj.RootSignature.Size = Size;
  obj.RootSignature.Text = (pText ? InternString(*pText).data() : nullptr);
  return obj;
}

DxilSubobject &DxilSubobjects::CreateSubobjectToExportsAssociation(
    llvm::StringRef Name,
    llvm::StringRef Subobject,
    llvm::StringRef *Exports,
    uint32_t NumExports) {
  auto &obj = CreateSubobject(Kind::SubobjectToExportsAssociation, Name);
  Subobject = InternString(Subobject);
  obj.SubobjectToExportsAssociation.Subobject = Subobject.data();
  for (unsigned i = 0; i < NumExports; i++) {
    obj.m_Exports.emplace_back(InternString(Exports[i]).data());
  }
  return obj;
}

DxilSubobject &DxilSubobjects::CreateRaytracingShaderConfig(
    llvm::StringRef Name,
    uint32_t MaxPayloadSizeInBytes,
    uint32_t MaxAttributeSizeInBytes) {
  auto &obj = CreateSubobject(Kind::RaytracingShaderConfig, Name);
  obj.RaytracingShaderConfig.MaxPayloadSizeInBytes = MaxPayloadSizeInBytes;
  obj.RaytracingShaderConfig.MaxAttributeSizeInBytes = MaxAttributeSizeInBytes;
  return obj;
}

DxilSubobject &DxilSubobjects::CreateRaytracingPipelineConfig(
    llvm::StringRef Name,
    uint32_t MaxTraceRecursionDepth) {
  auto &obj = CreateSubobject(Kind::RaytracingPipelineConfig, Name);
  obj.RaytracingPipelineConfig.MaxTraceRecursionDepth = MaxTraceRecursionDepth;
  return obj;
}

DxilSubobject &DxilSubobjects::CreateHitGroup(llvm::StringRef Name,
                                              DXIL::HitGroupType hitGroupType,
                                              llvm::StringRef AnyHit,
                                              llvm::StringRef ClosestHit,
                                              llvm::StringRef Intersection) {
  auto &obj = CreateSubobject(Kind::HitGroup, Name);
  AnyHit = InternString(AnyHit);
  ClosestHit = InternString(ClosestHit);
  Intersection = InternString(Intersection);
  obj.HitGroup.Type = hitGroupType;
  obj.HitGroup.AnyHit = AnyHit.data();
  obj.HitGroup.ClosestHit = ClosestHit.data();
  obj.HitGroup.Intersection = Intersection.data();
  return obj;
}

DxilSubobject &DxilSubobjects::CreateRaytracingPipelineConfig1(
    llvm::StringRef Name, uint32_t MaxTraceRecursionDepth, uint32_t Flags) {
  auto &obj = CreateSubobject(Kind::RaytracingPipelineConfig1, Name);
  obj.RaytracingPipelineConfig1.MaxTraceRecursionDepth = MaxTraceRecursionDepth;
  DXASSERT_NOMSG(
      0 == ((~(uint32_t)DXIL::RaytracingPipelineFlags::ValidMask) & Flags));
  obj.RaytracingPipelineConfig1.Flags = Flags;
  return obj;
}

DxilSubobject &DxilSubobjects::CreateSubobject(Kind kind, llvm::StringRef Name) {
  Name = InternString(Name);
  IFTBOOLMSG(FindSubobject(Name) == nullptr, DXC_E_GENERAL_INTERNAL_ERROR, "Subobject name collision");
  IFTBOOLMSG(!Name.empty(), DXC_E_GENERAL_INTERNAL_ERROR, "Empty Subobject name");
  std::unique_ptr<DxilSubobject> ptr(new DxilSubobject(*this, kind, Name));
  DxilSubobject &ref = *ptr;
  m_Subobjects[Name] = std::move(ptr);
  return ref;
}

} // namespace hlsl

