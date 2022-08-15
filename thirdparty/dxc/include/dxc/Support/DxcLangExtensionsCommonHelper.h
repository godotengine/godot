///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxcLangExtensionsCommonHelper.h                                           //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Provides a helper class to implement language extensions to HLSL.         //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "dxc/Support/Unicode.h"
#include "dxc/Support/FileIOHelper.h"
#include "dxc/dxcapi.internal.h"
#include <vector>

namespace llvm {
class raw_string_ostream;
class CallInst;
class Value;
}

namespace hlsl {

class DxcLangExtensionsCommonHelper {
private:
  llvm::SmallVector<std::string, 2> m_semanticDefines;
  llvm::SmallVector<std::string, 2> m_semanticDefineExclusions;
  llvm::SetVector<std::string> m_nonOptSemanticDefines;
  llvm::SmallVector<std::string, 2> m_defines;
  llvm::SmallVector<CComPtr<IDxcIntrinsicTable>, 2> m_intrinsicTables;
  CComPtr<IDxcSemanticDefineValidator> m_semanticDefineValidator;
  std::string m_semanticDefineMetaDataName;
  std::string m_targetTriple;
  HRESULT STDMETHODCALLTYPE RegisterIntoVector(LPCWSTR name, llvm::SmallVector<std::string, 2>& here)
  {
    try {
      IFTPTR(name);
      std::string s;
      if (!Unicode::WideToUTF8String(name, &s)) {
        throw ::hlsl::Exception(E_INVALIDARG);
      }
      here.push_back(s);
      return S_OK;
    }
    CATCH_CPP_RETURN_HRESULT();
  }

  HRESULT STDMETHODCALLTYPE RegisterIntoSet(LPCWSTR name, llvm::SetVector<std::string>& here)
  {
    try {
      IFTPTR(name);
      std::string s;
      if (!Unicode::WideToUTF8String(name, &s)) {
        throw ::hlsl::Exception(E_INVALIDARG);
      }
      here.insert(s);
      return S_OK;
    }
    CATCH_CPP_RETURN_HRESULT();
  }

public:
  const llvm::SmallVector<std::string, 2>& GetSemanticDefines() const { return m_semanticDefines; }
  const llvm::SmallVector<std::string, 2>& GetSemanticDefineExclusions() const { return m_semanticDefineExclusions; }
  const llvm::SetVector<std::string>& GetNonOptSemanticDefines() const { return m_nonOptSemanticDefines; }
  const llvm::SmallVector<std::string, 2>& GetDefines() const { return m_defines; }
  llvm::SmallVector<CComPtr<IDxcIntrinsicTable>, 2>& GetIntrinsicTables(){ return m_intrinsicTables; }
  const std::string &GetSemanticDefineMetadataName() { return m_semanticDefineMetaDataName; }
  const std::string &GetTargetTriple() { return m_targetTriple; }

  HRESULT STDMETHODCALLTYPE RegisterSemanticDefine(LPCWSTR name)
  {
    return RegisterIntoVector(name, m_semanticDefines);
  }

  HRESULT STDMETHODCALLTYPE RegisterSemanticDefineExclusion(LPCWSTR name)
  {
    return RegisterIntoVector(name, m_semanticDefineExclusions);
  }

  HRESULT STDMETHODCALLTYPE RegisterNonOptSemanticDefine(LPCWSTR name)
  {
    return RegisterIntoSet(name, m_nonOptSemanticDefines);
  }

  HRESULT STDMETHODCALLTYPE RegisterDefine(LPCWSTR name)
  {
    return RegisterIntoVector(name, m_defines);
  }

  HRESULT STDMETHODCALLTYPE RegisterIntrinsicTable(_In_ IDxcIntrinsicTable* pTable)
  {
    try {
      IFTPTR(pTable);
      LPCSTR tableName = nullptr;
      IFT(pTable->GetTableName(&tableName));
      IFTPTR(tableName);
      IFTARG(strcmp(tableName, "op") != 0);   // "op" is reserved for builtin intrinsics
      for (auto &&table : m_intrinsicTables) {
        LPCSTR otherTableName = nullptr;
        IFT(table->GetTableName(&otherTableName));
        IFTPTR(otherTableName);
        IFTARG(strcmp(tableName, otherTableName) != 0); // Added a duplicate table name
      }
      m_intrinsicTables.push_back(pTable);
      return S_OK;
    }
    CATCH_CPP_RETURN_HRESULT();
  }

  // Set the validator used to validate semantic defines.
  // Only one validator stored and used to run validation.
  HRESULT STDMETHODCALLTYPE SetSemanticDefineValidator(_In_ IDxcSemanticDefineValidator* pValidator) {
    if (pValidator == nullptr)
      return E_POINTER;

    m_semanticDefineValidator = pValidator;
    return S_OK;
  }

  HRESULT STDMETHODCALLTYPE SetSemanticDefineMetaDataName(LPCSTR name) {
    try {
      m_semanticDefineMetaDataName = name;
      return S_OK;
    }
    CATCH_CPP_RETURN_HRESULT();
  }

  HRESULT STDMETHODCALLTYPE SetTargetTriple(LPCSTR triple) {
    try {
      m_targetTriple = triple;
      return S_OK;
    }
    CATCH_CPP_RETURN_HRESULT();
  }

  // Get the name of the dxil intrinsic function.
  std::string GetIntrinsicName(UINT opcode) {
    LPCSTR pName = "";
    for (IDxcIntrinsicTable *table : m_intrinsicTables) {
      if (SUCCEEDED(table->GetIntrinsicName(opcode, &pName))) {
        return pName;
      }
    }

      return "";
  }

  // Get the dxil opcode for the extension opcode if one exists.
  // Return true if the opcode was mapped successfully.
  bool GetDxilOpCode(UINT opcode, UINT &dxilOpcode) {
    for (IDxcIntrinsicTable *table : m_intrinsicTables) {
      if (SUCCEEDED(table->GetDxilOpCode(opcode, &dxilOpcode))) {
        return true;
      }
    }
    return false;
  }

  // Result of validating a semantic define.
  // Stores any warning or error messages produced by the validator.
  // Successful validation means that there are no warning or error messages.
  struct SemanticDefineValidationResult {
    std::string Warning;
    std::string Error;

    bool HasError() { return Error.size() > 0; }
    bool HasWarning() { return Warning.size() > 0; }

    static SemanticDefineValidationResult Success() {
      return SemanticDefineValidationResult();
    }
  };

  // Use the contained semantice define validator to validate the given semantic define.
  SemanticDefineValidationResult ValidateSemanticDefine(const std::string &name, const std::string &value) {
    if (!m_semanticDefineValidator)
      return SemanticDefineValidationResult::Success();

    // Blobs for getting restul from validator. Strings for returning results to caller.
    CComPtr<IDxcBlobEncoding> pError;
    CComPtr<IDxcBlobEncoding> pWarning;
    std::string error;
    std::string warning;

    // Run semantic define validator.
    HRESULT result = m_semanticDefineValidator->GetSemanticDefineWarningsAndErrors(name.c_str(), value.c_str(), &pWarning, &pError);


    if (FAILED(result)) {
      // Failure indicates it was not able to even run validation so
      // we cannot say whether the define is invalid or not. Return a
      // generic error message about failure to run the valiadator.
      error = "failed to run semantic define validator for: ";
      error.append(name); error.append("="); error.append(value);
      return SemanticDefineValidationResult{ warning, error };
    }

    // Define a  little function to convert encoded blob into a string.
    auto GetErrorAsString = [&name](const CComPtr<IDxcBlobEncoding> &pBlobString) -> std::string {
      CComPtr<IDxcBlobUtf8> pUTF8BlobStr;
      if (SUCCEEDED(hlsl::DxcGetBlobAsUtf8(pBlobString, DxcGetThreadMallocNoRef(), &pUTF8BlobStr)))
        return std::string(pUTF8BlobStr->GetStringPointer(), pUTF8BlobStr->GetStringLength());
      else
        return std::string("invalid semantic define " + name);
    };

    // Check to see if any warnings or errors were produced.
    if (pError && pError->GetBufferSize()) {
      error = GetErrorAsString(pError);
    }
    if (pWarning && pWarning->GetBufferSize()) {
      warning = GetErrorAsString(pWarning);
    }

    return SemanticDefineValidationResult{ warning, error };
  }

  DxcLangExtensionsCommonHelper()
      : m_semanticDefineMetaDataName("hlsl.semdefs"),
        m_targetTriple("dxil-ms-dx") {}
};

// Use this macro to embed an implementation that will delegate to a field.
// Note that QueryInterface still needs to return the vtable.
#define DXC_LANGEXTENSIONS_HELPER_IMPL(_helper_field_) \
  HRESULT STDMETHODCALLTYPE RegisterIntrinsicTable(_In_ IDxcIntrinsicTable *pTable) override { \
    DxcThreadMalloc TM(m_pMalloc); \
    return (_helper_field_).RegisterIntrinsicTable(pTable); \
  } \
  HRESULT STDMETHODCALLTYPE RegisterSemanticDefine(LPCWSTR name) override { \
    DxcThreadMalloc TM(m_pMalloc); \
    return (_helper_field_).RegisterSemanticDefine(name); \
  } \
  HRESULT STDMETHODCALLTYPE RegisterSemanticDefineExclusion(LPCWSTR name) override { \
    DxcThreadMalloc TM(m_pMalloc); \
    return (_helper_field_).RegisterSemanticDefineExclusion(name); \
  } \
  HRESULT STDMETHODCALLTYPE RegisterNonOptSemanticDefine(LPCWSTR name) override { \
    DxcThreadMalloc TM(m_pMalloc); \
    return (_helper_field_).RegisterNonOptSemanticDefine(name); \
  } \
  HRESULT STDMETHODCALLTYPE RegisterDefine(LPCWSTR name) override { \
    DxcThreadMalloc TM(m_pMalloc); \
    return (_helper_field_).RegisterDefine(name); \
  } \
  HRESULT STDMETHODCALLTYPE SetSemanticDefineValidator(_In_ IDxcSemanticDefineValidator* pValidator) override { \
    DxcThreadMalloc TM(m_pMalloc); \
    return (_helper_field_).SetSemanticDefineValidator(pValidator); \
  } \
  HRESULT STDMETHODCALLTYPE SetSemanticDefineMetaDataName(LPCSTR name) override { \
    DxcThreadMalloc TM(m_pMalloc); \
    return (_helper_field_).SetSemanticDefineMetaDataName(name); \
  } \
  HRESULT STDMETHODCALLTYPE SetTargetTriple(LPCSTR name)  override { \
    DxcThreadMalloc TM(m_pMalloc);                                   \
    return (_helper_field_).SetTargetTriple(name);                   \
  } \

} // namespace hlsl
