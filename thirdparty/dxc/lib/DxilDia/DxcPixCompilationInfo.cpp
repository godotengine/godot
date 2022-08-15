///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxcPixCompilationInfo.cpp                                                 //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Defines all of the entrypoints for DXC's PIX interfaces for returning     //
// compilation parameters such as target profile, entry point, etc.          //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/Support/WinIncludes.h"

#include "dxc/dxcapi.h"
#include "dxc/dxcpix.h"

#include "dxc/Support/FileIOHelper.h"
#include "dxc/Support/Global.h"
#include "dxc/Support/microcom.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MSFileSystem.h"

#include "DxcPixBase.h"
#include "DxcPixDxilDebugInfo.h"

#include <functional>

#include "dxc/Support/WinIncludes.h"

#include "DxcPixCompilationInfo.h"
#include "DxcPixDxilDebugInfo.h"

#include "dxc/Support/Global.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"

#include "DxilDiaSession.h"

#include <unordered_map>

namespace dxil_debug_info {

struct CompilationInfo : public IDxcPixCompilationInfo {
private:
  DXC_MICROCOM_TM_REF_FIELDS();

  dxil_dia::Session *m_pSession;
  llvm::NamedMDNode *m_contents;
  llvm::NamedMDNode *m_defines;
  llvm::NamedMDNode *m_mainFileName;
  llvm::NamedMDNode *m_arguments;

public:
  CompilationInfo(IMalloc *pMalloc, dxil_dia::Session *pSession);

  DXC_MICROCOM_TM_ADDREF_RELEASE_IMPL();
  DXC_MICROCOM_TM_ALLOC(CompilationInfo);

  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void **ppvObject) final {
    return DoBasicQueryInterface<IDxcPixCompilationInfo>(this, iid, ppvObject);
  }
  virtual STDMETHODIMP
  GetSourceFile(_In_ DWORD SourceFileOrdinal,
                _Outptr_result_z_ BSTR *pSourceName,
                _Outptr_result_z_ BSTR *pSourceContents) override;
  virtual STDMETHODIMP
  GetArguments(_Outptr_result_z_ BSTR *pArguments) override;
  virtual STDMETHODIMP
  GetMacroDefinitions(_Outptr_result_z_ BSTR *pMacroDefinitions) override;
  virtual STDMETHODIMP
  GetEntryPointFile(_Outptr_result_z_ BSTR *pEntryPointFile) override;
  virtual STDMETHODIMP
  GetHlslTarget(_Outptr_result_z_ BSTR *pHlslTarget) override;
  virtual STDMETHODIMP
  GetEntryPoint(_Outptr_result_z_ BSTR *pEntryPoint) override;
};

CompilationInfo::CompilationInfo(IMalloc *pMalloc, dxil_dia::Session *pSession)
    : m_pSession(pSession), m_pMalloc(pMalloc) {
  auto *Module = m_pSession->DxilModuleRef().GetModule();
  m_contents =
      Module->getNamedMetadata(hlsl::DxilMDHelper::kDxilSourceContentsMDName);
  if (!m_contents)
    m_contents = Module->getNamedMetadata("llvm.dbg.contents");

  m_defines =
      Module->getNamedMetadata(hlsl::DxilMDHelper::kDxilSourceDefinesMDName);
  if (!m_defines)
    m_defines = Module->getNamedMetadata("llvm.dbg.defines");

  m_mainFileName = Module->getNamedMetadata(
      hlsl::DxilMDHelper::kDxilSourceMainFileNameMDName);
  if (!m_mainFileName)
    m_mainFileName = Module->getNamedMetadata("llvm.dbg.mainFileName");

  m_arguments =
      Module->getNamedMetadata(hlsl::DxilMDHelper::kDxilSourceArgsMDName);
  if (!m_arguments)
    m_arguments = Module->getNamedMetadata("llvm.dbg.args");
}

static void MDStringOperandToBSTR(llvm::MDOperand const &mdOperand,
                                  BSTR *pBStr) {
  llvm::StringRef MetadataAsStringRef =
      llvm::dyn_cast<llvm::MDString>(mdOperand)->getString();
  std::string StringWithTerminator(MetadataAsStringRef.begin(),
                                   MetadataAsStringRef.size());
  CA2W cv(StringWithTerminator.c_str(), CP_UTF8);
  CComBSTR BStr;
  BStr.Append(cv);
  BStr.Append(L"\0", 1);
  *pBStr = BStr.Detach();
}

STDMETHODIMP
CompilationInfo::GetSourceFile(_In_ DWORD SourceFileOrdinal,
                               _Outptr_result_z_ BSTR *pSourceName,
                               _Outptr_result_z_ BSTR *pSourceContents) {
  *pSourceName = nullptr;
  *pSourceContents = nullptr;

  if (SourceFileOrdinal >= m_contents->getNumOperands()) {
    return E_INVALIDARG;
  }

  llvm::MDTuple *FileTuple =
      llvm::cast<llvm::MDTuple>(m_contents->getOperand(SourceFileOrdinal));

  MDStringOperandToBSTR(FileTuple->getOperand(0), pSourceName);
  MDStringOperandToBSTR(FileTuple->getOperand(1), pSourceContents);

  return S_OK;
}

STDMETHODIMP CompilationInfo::GetArguments(_Outptr_result_z_ BSTR *pArguments) {
  llvm::MDNode *argsNode = m_arguments->getOperand(0);

  // Don't return any arguments that denote things that are returned via
  // other methods in this class (and that PIX isn't expecting to see
  // in the arguments list):
  const char *specialCases[] = {
      "/T", "-T", "-D", "/D", "-E", "/E",
  };

  // Concatenate arguments into one string
  CComBSTR pBSTR;
  for (llvm::MDNode::op_iterator it = argsNode->op_begin();
       it != argsNode->op_end(); ++it) {
    llvm::StringRef strRef = llvm::dyn_cast<llvm::MDString>(*it)->getString();

    bool skip = false;
    bool skipTwice = false;
    for (unsigned i = 0; i < _countof(specialCases); i++) {
      // It's legal for users to specify, for example, /Emain or /E main:
      if (strRef == specialCases[i]) {
        skipTwice = true;
        skip = true;
        break;
      } else if (strRef.startswith(specialCases[i])) {
        skip = true;
        break;
      }
    }

    if (skip) {
      if (skipTwice)
        ++it;
      continue;
    }

    std::string str(strRef.begin(), strRef.size());
    CA2W cv(str.c_str(), CP_UTF8);
    pBSTR.Append(cv);
    pBSTR.Append(L" ", 1);
  }
  pBSTR.Append(L"\0", 1);
  *pArguments = pBSTR.Detach();
  return S_OK;
}

STDMETHODIMP CompilationInfo::GetMacroDefinitions(
    _Outptr_result_z_ BSTR *pMacroDefinitions) {
  llvm::MDNode *definesNode = m_defines->getOperand(0);
  // Concatenate definitions into one string separated by spaces
  CComBSTR pBSTR;
  for (llvm::MDNode::op_iterator it = definesNode->op_begin();
       it != definesNode->op_end(); ++it) {
    llvm::StringRef strRef = llvm::dyn_cast<llvm::MDString>(*it)->getString();
    std::string str(strRef.begin(), strRef.size());

    // PIX is expecting quoted strings as the definitions. So if no quotes were
    // given, add them now:
    auto findEquals = str.find_first_of("=");
    if (findEquals != std::string::npos && findEquals + 1 < str.length()) {
      std::string definition = str.substr(findEquals + 1);
      if (definition.front() != '\"') {
        definition.insert(definition.begin(), '\"');
      }
      if (definition.back() != '\"') {
        definition.push_back('\"');
      }
      std::string name = str.substr(0, findEquals);
      str = name + "=" + definition;
    }

    CA2W cv(str.c_str(), CP_UTF8);
    pBSTR.Append(L"-D", 2);
    pBSTR.Append(cv);
    pBSTR.Append(L" ", 1);
  }
  pBSTR.Append(L"\0", 1);
  *pMacroDefinitions = pBSTR.Detach();
  return S_OK;
}

STDMETHODIMP
CompilationInfo::GetEntryPointFile(_Outptr_result_z_ BSTR *pEntryPointFile) {
  llvm::StringRef strRef = llvm::dyn_cast<llvm::MDString>(
                               m_mainFileName->getOperand(0)->getOperand(0))
                               ->getString();
  std::string str(strRef.begin(),
                  strRef.size()); // To make sure str is null terminated
  CA2W cv(str.c_str(), CP_UTF8);
  CComBSTR pBSTR;
  pBSTR.Append(cv);
  *pEntryPointFile = pBSTR.Detach();
  return S_OK;
}

STDMETHODIMP
CompilationInfo::GetHlslTarget(_Outptr_result_z_ BSTR *pHlslTarget) {
  CA2W cv(m_pSession->DxilModuleRef().GetShaderModel()->GetName(), CP_UTF8);
  CComBSTR pBSTR;
  pBSTR.Append(cv);
  *pHlslTarget = pBSTR.Detach();
  return S_OK;
}

STDMETHODIMP
CompilationInfo::GetEntryPoint(_Outptr_result_z_ BSTR *pEntryPoint) {
  auto name = m_pSession->DxilModuleRef().GetEntryFunctionName();
  CA2W cv(name.c_str(), CP_UTF8);
  CComBSTR pBSTR;
  pBSTR.Append(cv);
  *pEntryPoint = pBSTR.Detach();
  return S_OK;
}

} // namespace dxil_debug_info

HRESULT
dxil_debug_info::CreateDxilCompilationInfo(IMalloc *pMalloc,
                                           dxil_dia::Session *pSession,
                                           IDxcPixCompilationInfo **ppResult) {
  return NewDxcPixDxilDebugInfoObjectOrThrow<CompilationInfo>(ppResult, pMalloc,
                                                              pSession);
}
