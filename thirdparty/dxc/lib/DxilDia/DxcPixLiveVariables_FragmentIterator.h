///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxcPixLiveVariables_FragmentIterator.h                                    //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Declares the FragmentIterator API. This API is used to traverse           //
// DIVariables and assign alloca registers to DIBasicTypes.                  //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <memory>

namespace llvm
{
class AllocaInst;
class DataLayout;
class DbgDeclareInst;
class DIExpression;
}  // namespace llvm

namespace dxil_debug_info
{
class MemberIterator
{
public:
  virtual ~MemberIterator() = default;
  virtual bool Next(unsigned *Index) = 0;
  virtual unsigned SizeInBits(unsigned Index) const = 0;
  virtual unsigned OffsetInBits(unsigned Index) = 0;
};

std::unique_ptr<MemberIterator> CreateMemberIterator
(
  llvm::DbgDeclareInst *DbgDeclare, 
  const llvm::DataLayout &DataLayout,
  llvm::AllocaInst *Alloca, 
  llvm::DIExpression *Expression
);
}  // namespace dxil_debug_info
