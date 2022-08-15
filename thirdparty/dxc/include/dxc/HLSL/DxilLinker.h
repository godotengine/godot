///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilLinker.h                                                              //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Representation of HLSL Linker.                                            //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <unordered_map>
#include <unordered_set>
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringMap.h"
#include <memory>
#include "llvm/Support/ErrorOr.h"
#include "dxc/HLSL/DxilExportMap.h"

namespace llvm {
class Function;
class GlobalVariable;
class Constant;
class Module;
class LLVMContext;
} // namespace llvm

namespace hlsl {
class DxilModule;
class DxilResourceBase;

// Linker for DxilModule.
class DxilLinker {
public:
  virtual ~DxilLinker() {}
  static DxilLinker *CreateLinker(llvm::LLVMContext &Ctx, unsigned valMajor, unsigned valMinor);

  void SetValidatorVersion(unsigned valMajor, unsigned valMinor) { m_valMajor = valMajor, m_valMinor = valMinor; }
  virtual bool HasLibNameRegistered(llvm::StringRef name) = 0;
  virtual bool RegisterLib(llvm::StringRef name,
                           std::unique_ptr<llvm::Module> pModule,
                           std::unique_ptr<llvm::Module> pDebugModule) = 0;
  virtual bool AttachLib(llvm::StringRef name) = 0;
  virtual bool DetachLib(llvm::StringRef name) = 0;
  virtual void DetachAll() = 0;

  virtual std::unique_ptr<llvm::Module>
  Link(llvm::StringRef entry, llvm::StringRef profile, dxilutil::ExportMap &exportMap) = 0;

protected:
  DxilLinker(llvm::LLVMContext &Ctx, unsigned valMajor, unsigned valMinor) : m_ctx(Ctx), m_valMajor(valMajor), m_valMinor(valMinor) {}
  llvm::LLVMContext &m_ctx;
  unsigned m_valMajor, m_valMinor;
};

} // namespace hlsl
