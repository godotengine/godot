///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// dxcfilesystem.h                                                           //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Provides helper file system for dxcompiler.                               //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "dxc/dxcapi.h"
#include "llvm/Support/MSFileSystem.h"
#include <string>

namespace clang {
class CompilerInstance;
}

namespace llvm {
class raw_string_ostream;
namespace sys {
namespace fs {
class MSFileSystem;
}
} // namespace sys
} // namespace llvm



namespace dxcutil {

class DxcArgsFileSystem : public ::llvm::sys::fs::MSFileSystem {
public:
  virtual ~DxcArgsFileSystem(){};
  virtual void SetupForCompilerInstance(clang::CompilerInstance &compiler) = 0;
  virtual void GetStdOutputHandleStream(IStream **ppResultStream) = 0;
  virtual void GetStdErrorHandleStream(IStream **ppResultStream) = 0;
  virtual void WriteStdErrToStream(llvm::raw_string_ostream &s) = 0;
  virtual void WriteStdOutToStream(llvm::raw_string_ostream &s) = 0;
  virtual void EnableDisplayIncludeProcess() = 0;
  virtual HRESULT CreateStdStreams(_In_ IMalloc *pMalloc) = 0;
  virtual HRESULT RegisterOutputStream(LPCWSTR pName, IStream *pStream) = 0;
  virtual HRESULT UnRegisterOutputStream() = 0;
};

DxcArgsFileSystem *
CreateDxcArgsFileSystem(_In_ IDxcBlobUtf8 *pSource, _In_ LPCWSTR pSourceName,
                        _In_opt_ IDxcIncludeHandler *pIncludeHandler,
                        _In_opt_ UINT32 defaultCodePage = CP_ACP);

void MakeAbsoluteOrCurDirRelativeW(LPCWSTR &Path, std::wstring &PathStorage);

} // namespace dxcutil