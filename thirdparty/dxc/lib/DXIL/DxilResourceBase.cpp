///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilResourceBase.cpp                                                      //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/DXIL/DxilResourceBase.h"
#include "dxc/Support/Global.h"
#include "llvm/IR/Constant.h"


namespace hlsl {


//------------------------------------------------------------------------------
//
// ResourceBase methods.
//
DxilResourceBase::DxilResourceBase(Class C)
: m_Class(C)
, m_Kind(Kind::Invalid)
, m_ID(UINT_MAX)
, m_SpaceID(0)
, m_LowerBound(0)
, m_RangeSize(0) 
, m_pSymbol(nullptr) 
, m_pHandle(nullptr)
, m_pHLSLTy(nullptr) {
}

DxilResourceBase::Class DxilResourceBase::GetClass() const { return m_Class; }
DxilResourceBase::Kind DxilResourceBase::GetKind() const { return m_Kind; }
void DxilResourceBase::SetKind(DxilResourceBase::Kind ResourceKind) {
  DXASSERT(ResourceKind > Kind::Invalid && ResourceKind < Kind::NumEntries, "otherwise the caller passed wrong resource type");
  m_Kind = ResourceKind;
}

unsigned DxilResourceBase::GetID() const          { return m_ID; }
unsigned DxilResourceBase::GetSpaceID() const     { return m_SpaceID; }
unsigned DxilResourceBase::GetLowerBound() const  { return m_LowerBound; }
unsigned DxilResourceBase::GetUpperBound() const  { return m_RangeSize != UINT_MAX ? m_LowerBound + m_RangeSize - 1 : UINT_MAX; }
unsigned DxilResourceBase::GetRangeSize() const   { return m_RangeSize; }
llvm::Constant *DxilResourceBase::GetGlobalSymbol() const { return m_pSymbol; }
const std::string &DxilResourceBase::GetGlobalName() const      { return m_Name; }
llvm::Value *DxilResourceBase::GetHandle() const { return m_pHandle; }
// If m_pHLSLTy is nullptr, HLSL type is the type of m_pSymbol.
// In sm6.6, type of m_pSymbol will be mutated to handleTy, m_pHLSLTy will save
// the original HLSL type.
llvm::Type *DxilResourceBase::GetHLSLType() const {
  return m_pHLSLTy == nullptr ? m_pSymbol->getType() : m_pHLSLTy;
}
bool DxilResourceBase::IsAllocated() const        { return m_LowerBound != UINT_MAX; }
bool DxilResourceBase::IsUnbounded() const        { return m_RangeSize == UINT_MAX; }

void DxilResourceBase::SetClass(Class C)                          { m_Class = C; }
void DxilResourceBase::SetID(unsigned ID)                         { m_ID = ID; }
void DxilResourceBase::SetSpaceID(unsigned SpaceID)               { m_SpaceID = SpaceID; }
void DxilResourceBase::SetLowerBound(unsigned LB)                 { m_LowerBound = LB; }
void DxilResourceBase::SetRangeSize(unsigned RangeSize)           { m_RangeSize = RangeSize; }
void DxilResourceBase::SetGlobalSymbol(llvm::Constant *pGV)       { m_pSymbol = pGV; }
void DxilResourceBase::SetGlobalName(const std::string &Name)     { m_Name = Name; }
void DxilResourceBase::SetHandle(llvm::Value *pHandle)            { m_pHandle = pHandle; }
void DxilResourceBase::SetHLSLType(llvm::Type *pTy)               { m_pHLSLTy = pTy; }

static const char *s_ResourceClassNames[] = {
    "texture", "UAV", "cbuffer", "sampler"
};
static_assert(_countof(s_ResourceClassNames) == (unsigned)DxilResourceBase::Class::Invalid,
  "Resource class names array must be updated when new resource class enums are added.");

const char *DxilResourceBase::GetResClassName() const {
  return s_ResourceClassNames[(unsigned)m_Class];
}

static const char *s_ResourceIDPrefixes[] = {
    "T", "U", "CB", "S"
};
static_assert(_countof(s_ResourceIDPrefixes) == (unsigned)DxilResourceBase::Class::Invalid,
  "Resource id prefixes array must be updated when new resource class enums are added.");

const char *DxilResourceBase::GetResIDPrefix() const {
  return s_ResourceIDPrefixes[(unsigned)m_Class];
}

static const char *s_ResourceBindPrefixes[] = {
    "t", "u", "cb", "s"
};
static_assert(_countof(s_ResourceBindPrefixes) == (unsigned)DxilResourceBase::Class::Invalid,
  "Resource bind prefixes array must be updated when new resource class enums are added.");

const char *DxilResourceBase::GetResBindPrefix() const {
  return s_ResourceBindPrefixes[(unsigned)m_Class];
}

static const char *s_ResourceDimNames[] = {
        "invalid", "1d",        "2d",      "2dMS",      "3d",
        "cube",    "1darray",   "2darray", "2darrayMS", "cubearray",
        "buf",     "rawbuf",    "structbuf", "cbuffer", "sampler",
        "tbuffer", "ras", "fbtex2d", "fbtex2darray",
};
static_assert(_countof(s_ResourceDimNames) == (unsigned)DxilResourceBase::Kind::NumEntries,
  "Resource dim names array must be updated when new resource kind enums are added.");

const char *DxilResourceBase::GetResDimName() const {
  return s_ResourceDimNames[(unsigned)m_Kind];
}

static const char *s_ResourceKindNames[] = {
        "invalid",     "Texture1D",        "Texture2D",        "Texture2DMS",      "Texture3D",
        "TextureCube", "Texture1DArray",   "Texture2DArray",   "Texture2DMSArray", "TextureCubeArray",
        "TypedBuffer", "RawBuffer",        "StructuredBuffer", "CBuffer",          "Sampler",
        "TBuffer",     "RTAccelerationStructure", "FeedbackTexture2D", "FeedbackTexture2DArray",
};
static_assert(_countof(s_ResourceKindNames) == (unsigned)DxilResourceBase::Kind::NumEntries,
  "Resource kind names array must be updated when new resource kind enums are added.");

const char *DxilResourceBase::GetResKindName() const {
  return GetResourceKindName(m_Kind);
}

const char *GetResourceKindName(DXIL::ResourceKind K) {
  return s_ResourceKindNames[(unsigned)K];
}

} // namespace hlsl
