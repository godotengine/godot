///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilResourceBase.h                                                        //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Base class to represent DXIL SRVs, UAVs, CBuffers, and Samplers.          //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <string>

#include "DxilConstants.h"

namespace llvm {
class Value;
class Constant;
class Type;
}


namespace hlsl {

/// Base class to represent HLSL SRVs, UAVs, CBuffers, and Samplers.
class DxilResourceBase {
public:
  using Class = DXIL::ResourceClass;
  using Kind = DXIL::ResourceKind;

public:
  DxilResourceBase(Class C);
  virtual ~DxilResourceBase() {}

  Class GetClass() const;
  DxilResourceBase::Kind GetKind() const;
  unsigned GetID() const;
  unsigned GetSpaceID() const;
  unsigned GetLowerBound() const;
  unsigned GetUpperBound() const;
  unsigned GetRangeSize() const;
  llvm::Constant *GetGlobalSymbol() const;
  llvm::Type *GetHLSLType() const;
  const std::string &GetGlobalName() const;
  llvm::Value *GetHandle() const;
  bool IsAllocated() const;
  bool IsUnbounded() const;

  void SetKind(DxilResourceBase::Kind ResourceKind);
  void SetSpaceID(unsigned SpaceID);
  void SetLowerBound(unsigned LB);
  void SetRangeSize(unsigned RangeSize);
  void SetGlobalSymbol(llvm::Constant *pGV);
  void SetGlobalName(const std::string &Name);
  void SetHandle(llvm::Value *pHandle);
  void SetHLSLType(llvm::Type *Ty);

  // TODO: check whether we can make this a protected method.
  void SetID(unsigned ID);

  const char *GetResClassName() const;
  const char *GetResDimName() const;
  const char *GetResIDPrefix() const;
  const char *GetResBindPrefix() const;
  const char *GetResKindName() const;

protected:
  void SetClass(Class C);

private:
  Class m_Class;                  // Resource class (SRV, UAV, CBuffer, Sampler).
  Kind m_Kind;                    // Detail resource kind( texture2D...).
  unsigned m_ID;                  // Unique ID within the class.
  unsigned m_SpaceID;             // Root signature space.
  unsigned m_LowerBound;          // Range lower bound.
  unsigned m_RangeSize;           // Range size in entries.
  llvm::Constant *m_pSymbol;      // Global variable.
  std::string m_Name;             // Unmangled name of the global variable.
  llvm::Value *m_pHandle;         // Cached resource handle for SM5.0- (and maybe SM5.1).
  llvm::Type *m_pHLSLTy;           // The original hlsl type for reflection.
};

const char *GetResourceKindName(DXIL::ResourceKind K);

} // namespace hlsl
