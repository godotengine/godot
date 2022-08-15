///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// dxcapi.impl.h                                                             //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Provides support for DXC API implementations.                             //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#ifndef __DXCAPI_IMPL__
#define __DXCAPI_IMPL__

#include "dxc/dxcapi.h"
#include "dxc/Support/microcom.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/ArrayRef.h"

// Simple adaptor for IStream. Can probably do better.
class raw_stream_ostream : public llvm::raw_ostream {
private:
  CComPtr<hlsl::AbstractMemoryStream> m_pStream;
  void write_impl(const char *Ptr, size_t Size) override {
    ULONG cbWritten;
    IFT(m_pStream->Write(Ptr, Size, &cbWritten));
  }
  uint64_t current_pos() const override { return m_pStream->GetPosition(); }
public:
  raw_stream_ostream(hlsl::AbstractMemoryStream* pStream) : m_pStream(pStream) { }
  ~raw_stream_ostream() override {
    flush();
  }
};

namespace {
HRESULT TranslateUtf8StringForOutput(
    _In_opt_count_(size) LPCSTR pStr, SIZE_T size, UINT32 codePage, IDxcBlobEncoding **ppBlobEncoding) {
  CComPtr<IDxcBlobEncoding> pBlobEncoding;
  IFR(hlsl::DxcCreateBlobWithEncodingOnHeapCopy(pStr, size, DXC_CP_UTF8, &pBlobEncoding));
  if (codePage == DXC_CP_WIDE) {
    CComPtr<IDxcBlobWide> pBlobWide;
    IFT(hlsl::DxcGetBlobAsWide(pBlobEncoding, nullptr, &pBlobWide))
      pBlobEncoding = pBlobWide;
  }
  *ppBlobEncoding = pBlobEncoding.Detach();
  return S_OK;
}

HRESULT TranslateWideStringForOutput(
    _In_opt_count_(size) LPCWSTR pStr, SIZE_T size, UINT32 codePage, IDxcBlobEncoding **ppBlobEncoding) {
  CComPtr<IDxcBlobEncoding> pBlobEncoding;
  IFR(hlsl::DxcCreateBlobWithEncodingOnHeapCopy(pStr, size, DXC_CP_WIDE, &pBlobEncoding));
  if (codePage == DXC_CP_UTF8) {
    CComPtr<IDxcBlobUtf8> pBlobUtf8;
    IFT(hlsl::DxcGetBlobAsUtf8(pBlobEncoding, nullptr, &pBlobUtf8))
      pBlobEncoding = pBlobUtf8;
  }
  *ppBlobEncoding = pBlobEncoding.Detach();
  return S_OK;
}

HRESULT TranslateStringBlobForOutput(IDxcBlob *pBlob, UINT32 codePage, IDxcBlobEncoding **ppBlobEncoding) {
  CComPtr<IDxcBlobEncoding> pEncoding;
  IFR(pBlob->QueryInterface(&pEncoding));
  BOOL known;
  UINT32 inputCP;
  IFR(pEncoding->GetEncoding(&known, &inputCP));
  IFRBOOL(known, E_INVALIDARG);
  if (inputCP == DXC_CP_UTF8) {
    return TranslateUtf8StringForOutput((LPCSTR)pBlob->GetBufferPointer(), pBlob->GetBufferSize(), codePage, ppBlobEncoding);
  } else if (inputCP == DXC_CP_WIDE) {
    return TranslateWideStringForOutput((LPCWSTR)pBlob->GetBufferPointer(), pBlob->GetBufferSize(), codePage, ppBlobEncoding);
  }
  return E_INVALIDARG;
}
}

typedef enum DxcOutputType {
  DxcOutputType_None    = 0,
  DxcOutputType_Blob    = 1,
  DxcOutputType_Text    = 2,

  DxcOutputTypeForceDword = 0xFFFFFFFF
} DxcOutputType;

inline DxcOutputType DxcGetOutputType(DXC_OUT_KIND kind) {
  switch (kind) {
  case DXC_OUT_OBJECT:
  case DXC_OUT_PDB:
  case DXC_OUT_SHADER_HASH:
  case DXC_OUT_REFLECTION:
  case DXC_OUT_ROOT_SIGNATURE:
    return DxcOutputType_Blob;
  case DXC_OUT_ERRORS:
  case DXC_OUT_DISASSEMBLY:
  case DXC_OUT_HLSL:
  case DXC_OUT_TEXT:
  case DXC_OUT_REMARKS:
    return DxcOutputType_Text;
  }
  return DxcOutputType_None;
}

// Update when new results are allowed
static const unsigned kNumDxcOutputTypes = DXC_OUT_REMARKS;
static const SIZE_T kAutoSize = (SIZE_T)-1;
static const LPCWSTR DxcOutNoName = nullptr;

struct DxcOutputObject {
  CComPtr<IUnknown> object;
  CComPtr<IDxcBlobWide> name;
  DXC_OUT_KIND kind = DXC_OUT_NONE;

  /////////////////////////
  // Convenient set methods
  /////////////////////////

  HRESULT SetObject(IUnknown *pUnknown, UINT32 codePage = DXC_CP_UTF8) {
    DXASSERT_NOMSG(!object);
    if (!pUnknown)
      return S_OK;
    if (codePage && DxcGetOutputType(kind) == DxcOutputType_Text) {
      CComPtr<IDxcBlob> pBlob;
      IFR(pUnknown->QueryInterface(&pBlob));
      CComPtr<IDxcBlobEncoding> pEncoding;
      // If not blob encoding, assume utf-8 text
      if (FAILED(TranslateStringBlobForOutput(pBlob, codePage, &pEncoding)))
        IFR(TranslateUtf8StringForOutput(
          (LPCSTR)pBlob->GetBufferPointer(), pBlob->GetBufferSize(),
          codePage, &pEncoding));
      object = pEncoding;
    } else {
      object = pUnknown;
    }
    return S_OK;
  }
  HRESULT SetObjectData(_In_opt_bytecount_(size) LPCVOID pData, SIZE_T size) {
    DXASSERT_NOMSG(!object);
    if (!pData || !size)
      return S_OK;
    IDxcBlob *pBlob;
    IFR(hlsl::DxcCreateBlobOnHeapCopy(pData, size, &pBlob));
    object = pBlob;
    return S_OK;
  }
  HRESULT SetString(_In_ UINT32 codePage, _In_opt_count_(size) LPCWSTR pText, SIZE_T size = kAutoSize) {
    DXASSERT_NOMSG(!object);
    if (!pText)
      return S_OK;
    if (size == kAutoSize)
      size = wcslen(pText);
    CComPtr<IDxcBlobEncoding> pBlobEncoding;
    IFR(TranslateWideStringForOutput(pText, size, codePage, &pBlobEncoding));
    object = pBlobEncoding;
    return S_OK;
  }
  HRESULT SetString(_In_ UINT32 codePage, _In_opt_count_(size) LPCSTR pText, SIZE_T size = kAutoSize) {
    DXASSERT_NOMSG(!object);
    if (!pText)
      return S_OK;
    if (size == kAutoSize)
      size = strlen(pText);
    CComPtr<IDxcBlobEncoding> pBlobEncoding;
    IFR(TranslateUtf8StringForOutput(pText, size, codePage, &pBlobEncoding));
    object = pBlobEncoding;
    return S_OK;
  }
  HRESULT SetName(_In_opt_z_ IDxcBlobWide *pName) {
    DXASSERT_NOMSG(!name);
    name = pName;
    return S_OK;
  }
  HRESULT SetName(_In_opt_z_ LPCWSTR pName) {
    DXASSERT_NOMSG(!name);
    if (!pName)
      return S_OK;
    CComPtr<IDxcBlobEncoding> pBlobEncoding;
    IFR(hlsl::DxcCreateBlobWithEncodingOnHeapCopy(
          pName, (wcslen(pName) + 1) * sizeof(wchar_t), DXC_CP_WIDE, &pBlobEncoding));
    return pBlobEncoding->QueryInterface(&name);
  }
  HRESULT SetName(_In_opt_z_ LPCSTR pName) {
    DXASSERT_NOMSG(!name);
    if (!pName)
      return S_OK;
    CComPtr<IDxcBlobEncoding> pBlobEncoding;
    IFR(TranslateUtf8StringForOutput(pName, strlen(pName) + 1, DXC_CP_WIDE, &pBlobEncoding));
    return pBlobEncoding->QueryInterface(&name);
  }
  HRESULT SetName(_In_opt_z_ llvm::StringRef Name) {
    DXASSERT_NOMSG(!name);
    if (Name.empty())
      return S_OK;
    CComPtr<IDxcBlobEncoding> pBlobEncoding;
    IFR(TranslateUtf8StringForOutput(Name.data(), Name.size(), DXC_CP_WIDE, &pBlobEncoding));
    return pBlobEncoding->QueryInterface(&name);
  }

  /////////////////////////////
  // Static object constructors
  /////////////////////////////

  template<typename DataTy, typename NameTy>
  static DxcOutputObject StringOutput(_In_ DXC_OUT_KIND kind,
                                      _In_ UINT32 codePage,
                                      _In_opt_count_(size) DataTy pText, _In_ SIZE_T size,
                                      _In_opt_z_ NameTy pName) {
    DxcOutputObject output;
    output.kind = kind;
    IFT(output.SetString(codePage, pText, size));
    IFT(output.SetName(pName));
    return output;
  }
  template<typename DataTy, typename NameTy>
  static DxcOutputObject StringOutput(_In_ DXC_OUT_KIND kind,
                                      _In_ UINT32 codePage,
                                      _In_opt_ DataTy pText,
                                      _In_opt_z_ NameTy pName) {
    return StringOutput(kind, codePage, pText, kAutoSize, pName);
  }
  template<typename NameTy>
  static DxcOutputObject DataOutput(_In_ DXC_OUT_KIND kind,
                                    _In_opt_bytecount_(size) LPCVOID pData, _In_ SIZE_T size,
                                    _In_opt_z_ NameTy pName) {
    DxcOutputObject output;
    output.kind = kind;
    IFT(output.SetObjectData(pData, size));
    IFT(output.SetName(pName));
    return output;
  }
  template<typename NameTy>
  static DxcOutputObject DataOutput(_In_ DXC_OUT_KIND kind,
                                    _In_opt_ IDxcBlob *pBlob,
                                    _In_opt_z_ NameTy pName) {
    DxcOutputObject output;
    output.kind = kind;
    IFT(output.SetObject(pBlob));
    IFT(output.SetName(pName));
    return output;
  }
  static DxcOutputObject DataOutput(_In_ DXC_OUT_KIND kind,
                                    _In_opt_ IDxcBlob *pBlob) {
    return DataOutput(kind, pBlob, DxcOutNoName);
  }
  template<typename NameTy>
  static DxcOutputObject DataOutput(_In_ DXC_OUT_KIND kind,
                                    _In_ UINT32 codePage,
                                    _In_opt_ IDxcBlob *pBlob,
                                    _In_opt_z_ NameTy pName) {
    DxcOutputObject output;
    output.kind = kind;
    IFT(output.SetObject(pBlob, codePage));
    IFT(output.SetName(pName));
    return output;
  }
  static DxcOutputObject DataOutput(_In_ DXC_OUT_KIND kind,
                                    _In_ UINT32 codePage,
                                    _In_opt_ IDxcBlob *pBlob) {
    return DataOutput(kind, codePage, pBlob, DxcOutNoName);
  }
  static DxcOutputObject DataOutput(_In_ DXC_OUT_KIND kind,
                                    _In_ UINT32 codePage,
                                    _In_opt_ IUnknown *pBlob) {
    DxcOutputObject output;
    output.kind = kind;
    IFT(output.SetObject(pBlob, codePage));
    IFT(output.SetName(DxcOutNoName));
    return output;
  }

  template<typename DataTy>
  static DxcOutputObject ErrorOutput(UINT32 codePage, DataTy pText, SIZE_T size) {
    return StringOutput(DXC_OUT_ERRORS, codePage, pText, size, DxcOutNoName);
  }
  template<typename DataTy>
  static DxcOutputObject ErrorOutput(UINT32 codePage, DataTy pText) {
    return StringOutput(DXC_OUT_ERRORS, codePage, pText, DxcOutNoName);
  }
  template<typename NameTy>
  static DxcOutputObject ObjectOutput(LPCVOID pData, SIZE_T size, NameTy pName) {
    return DataOutput(DXC_OUT_OBJECT, pData, size, pName);
  }
  static DxcOutputObject ObjectOutput(LPCVOID pData, SIZE_T size) {
    return DataOutput(DXC_OUT_OBJECT, pData, size, DxcOutNoName);
  }
};

struct DxcExtraOutputObject {
  CComPtr<IDxcBlobWide> pType; // Custom name to identify the object
  CComPtr<IDxcBlobWide> pName; // The file path for the output
  CComPtr<IUnknown> pObject;    // The object itself
};

class DxcExtraOutputs : public IDxcExtraOutputs {
  DXC_MICROCOM_TM_REF_FIELDS()

  DxcExtraOutputObject *m_Objects = nullptr;
  UINT32 m_uCount = 0;

public:

  DXC_MICROCOM_TM_ADDREF_RELEASE_IMPL()
  DXC_MICROCOM_TM_CTOR(DxcExtraOutputs)

  ~DxcExtraOutputs() {
    Clear();
  }

  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void **ppvObject) override {
    return DoBasicQueryInterface<IDxcExtraOutputs>(this, iid, ppvObject);
  }

  /////////////////////
  // IDxcExtraOutputs
  /////////////////////

  UINT32 STDMETHODCALLTYPE GetOutputCount() override {
    return m_uCount;
  }

  HRESULT STDMETHODCALLTYPE GetOutput(_In_ UINT32 uIndex,
    _In_ REFIID iid, _COM_Outptr_opt_result_maybenull_ void **ppvObject,
    _COM_Outptr_opt_result_maybenull_ IDxcBlobWide **ppOutputType,
    _COM_Outptr_opt_result_maybenull_ IDxcBlobWide **ppOutputName) override
  {
    if (uIndex >= m_uCount)
      return E_INVALIDARG;

    DxcExtraOutputObject *pObject = &m_Objects[uIndex];

    if (ppOutputType) {
      *ppOutputType = nullptr;
      IFR(pObject->pType.CopyTo(ppOutputType));
    }

    if (ppOutputName) {
      *ppOutputName = nullptr;
      IFR(pObject->pName.CopyTo(ppOutputName));
    }

    if (ppvObject) {
      *ppvObject = nullptr;
      if (pObject->pObject) {
        IFR(pObject->pObject->QueryInterface(iid, ppvObject));
      }
    }

    return S_OK;
  }

  /////////////////////
  // Internal Interface
  /////////////////////
  void Clear() {
    m_uCount = 0;
    if (m_Objects) {
      delete[] m_Objects;
      m_Objects = nullptr;
    }
  }

  void SetOutputs(const llvm::ArrayRef<DxcExtraOutputObject> outputs) {
    Clear();

    m_uCount = outputs.size();
    if (m_uCount > 0) {
      m_Objects = new DxcExtraOutputObject[m_uCount];
      for (UINT32 i = 0; i < outputs.size(); i++)
        m_Objects[i] = outputs[i];
    }
  }
};

class DxcResult : public IDxcResult {
private:
  DXC_MICROCOM_TM_REF_FIELDS()
  HRESULT m_status = S_OK;
  DxcOutputObject m_outputs[kNumDxcOutputTypes];  // indexed by DXC_OUT_KIND enum - 1
  DXC_OUT_KIND m_resultType = DXC_OUT_NONE;       // result type for GetResult()
  UINT32 m_textEncoding = DXC_CP_UTF8;              // encoding for text outputs

public:
  DXC_MICROCOM_TM_ADDREF_RELEASE_IMPL()
  DXC_MICROCOM_TM_CTOR(DxcResult)

  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void **ppvObject) override {
    return DoBasicQueryInterface<IDxcResult, IDxcOperationResult>(this, iid, ppvObject);
  }

  //////////////////////
  // IDxcOperationResult
  //////////////////////

  HRESULT STDMETHODCALLTYPE GetStatus(_Out_ HRESULT *pStatus) override {
    if (pStatus == nullptr)
      return E_INVALIDARG;

    *pStatus = m_status;
    return S_OK;
  }

  HRESULT STDMETHODCALLTYPE
    GetResult(_COM_Outptr_result_maybenull_ IDxcBlob **ppResult) override {
    *ppResult = nullptr;
    if (m_resultType == DXC_OUT_NONE)
      return S_OK;
    DxcOutputObject *pObject = Output(m_resultType);
    if (pObject && pObject->object)
      return pObject->object->QueryInterface(ppResult);
    return S_OK;
  }

  HRESULT STDMETHODCALLTYPE
    GetErrorBuffer(_COM_Outptr_result_maybenull_ IDxcBlobEncoding **ppErrors) override {
    *ppErrors = nullptr;
    DxcOutputObject *pObject = Output(DXC_OUT_ERRORS);
    if (pObject && pObject->object)
      return pObject->object->QueryInterface(ppErrors);
    return S_OK;
  }

  /////////////
  // IDxcResult
  /////////////

  BOOL STDMETHODCALLTYPE HasOutput(_In_ DXC_OUT_KIND dxcOutKind) override {
    if (dxcOutKind <= DXC_OUT_NONE || (unsigned)dxcOutKind > kNumDxcOutputTypes)
      return FALSE;
    return m_outputs[(unsigned)dxcOutKind - 1].kind != DXC_OUT_NONE;
  }
  HRESULT STDMETHODCALLTYPE GetOutput(_In_ DXC_OUT_KIND dxcOutKind,
      _In_ REFIID iid, _COM_Outptr_opt_result_maybenull_ void **ppvObject,
      _COM_Outptr_ IDxcBlobWide **ppOutputName) override {
    if (ppvObject == nullptr)
      return E_INVALIDARG;
    if (dxcOutKind <= DXC_OUT_NONE || (unsigned)dxcOutKind > kNumDxcOutputTypes)
      return E_INVALIDARG;
    DxcOutputObject &object = m_outputs[(unsigned)dxcOutKind - 1];
    if (object.kind == DXC_OUT_NONE)
      return E_INVALIDARG;
    *ppvObject = nullptr;
    if (ppOutputName)
      *ppOutputName = nullptr;
    IFR(object.object->QueryInterface(iid, ppvObject));
    if (ppOutputName && object.name) {
      object.name.CopyTo(ppOutputName);
    }
    return S_OK;
  }

  UINT32 GetNumOutputs() override {
    UINT32 numOutputs = 0;
    for (unsigned i = 0; i < kNumDxcOutputTypes; ++i) {
      if (m_outputs[i].kind != DXC_OUT_NONE)
        numOutputs++;
    }
    return numOutputs;
  }
  DXC_OUT_KIND GetOutputByIndex(UINT32 Index) override {
    if (!(Index < kNumDxcOutputTypes))
      return DXC_OUT_NONE;
    UINT32 numOutputs = 0;
    unsigned i = 0;
    for (; i < kNumDxcOutputTypes; ++i) {
      if (Index == numOutputs)
        return m_outputs[i].kind;
      if (m_outputs[i].kind != DXC_OUT_NONE)
        numOutputs++;
    }
    return DXC_OUT_NONE;
  }
  DXC_OUT_KIND PrimaryOutput() override {
    return m_resultType;
  }

  /////////////////////
  // Internal Interface
  /////////////////////

  HRESULT SetEncoding(UINT32 textEncoding) {
    if (textEncoding != DXC_CP_ACP && textEncoding != DXC_CP_UTF8 && textEncoding != DXC_CP_WIDE)
      return E_INVALIDARG;
    m_textEncoding = textEncoding;
    return S_OK;
  }

  DxcOutputObject *Output(DXC_OUT_KIND kind) {
    if (kind <= DXC_OUT_NONE || (unsigned)kind > kNumDxcOutputTypes)
      return nullptr;
    return &(m_outputs[(unsigned)kind - 1]);
  }

  HRESULT ClearOutput(DXC_OUT_KIND kind) {
    if (kind <= DXC_OUT_NONE || (unsigned)kind > kNumDxcOutputTypes)
      return E_INVALIDARG;
    DxcOutputObject &output = m_outputs[(unsigned)kind - 1];
    output.kind = DXC_OUT_NONE;
    output.object.Release();
    output.name.Release();
    return S_OK;
  }

  void ClearAllOutputs() {
    for (unsigned i = DXC_OUT_NONE + 1; i <= kNumDxcOutputTypes; i++)
      ClearOutput((DXC_OUT_KIND)(i));
  }

  HRESULT SetStatusAndPrimaryResult(HRESULT status, DXC_OUT_KIND resultType = DXC_OUT_NONE) {
    if ((unsigned)resultType > kNumDxcOutputTypes)
      return E_INVALIDARG;
    m_status = status;
    m_resultType = resultType;
    return S_OK;
  }

  // Set output object and name for previously uninitialized entry
  HRESULT SetOutput(const DxcOutputObject &output) {
    if (output.kind <= DXC_OUT_NONE || (unsigned)output.kind > kNumDxcOutputTypes)
      return E_INVALIDARG;
    if (!output.object)
      return E_INVALIDARG;
    DxcOutputObject &internalOutput = m_outputs[(unsigned)output.kind - 1];
    // Must not be overwriting an existing output
    if (internalOutput.kind != DXC_OUT_NONE)
      return E_INVALIDARG;
    internalOutput = output;
    return S_OK;
  }

  // Set or overwrite output object and set the kind
  HRESULT SetOutputObject(DXC_OUT_KIND kind, IUnknown *pObject) {
    if (kind <= DXC_OUT_NONE || (unsigned)kind > kNumDxcOutputTypes)
      return E_INVALIDARG;
    DxcOutputObject &output = m_outputs[(unsigned)kind - 1];
    if (!pObject)
      kind = DXC_OUT_NONE;
    output.kind = kind;
    output.SetObject(pObject, m_textEncoding);
    return S_OK;
  }
  // Set or overwrite output string object and set the kind
  template<typename StringTy>
  HRESULT SetOutputString(DXC_OUT_KIND kind, StringTy pString, size_t size = kAutoSize) {
    if (kind <= DXC_OUT_NONE || (unsigned)kind > kNumDxcOutputTypes)
      return E_INVALIDARG;
    DxcOutputObject &output = m_outputs[(unsigned)kind - 1];
    if (!pString)
      kind = DXC_OUT_NONE;
    output.kind = kind;
    output.SetString(m_textEncoding, pString, size);
    return S_OK;
  }
  // Set or overwrite the output name.  This does not set kind,
  // since that indicates an active output, which must have an object.
  template<typename NameTy>
  HRESULT SetOutputName(DXC_OUT_KIND kind, NameTy Name) {
    if (kind <= DXC_OUT_NONE || (unsigned)kind > kNumDxcOutputTypes)
      return E_INVALIDARG;
    Output(kind)->SetName(Name);
    return S_OK;
  }

  HRESULT SetOutputs(const llvm::ArrayRef<DxcOutputObject> outputs) {
    for (unsigned i = 0; i < outputs.size(); i++) {
      const DxcOutputObject &output = outputs.data()[i];
      // Skip if DXC_OUT_NONE or no object to store
      if (output.kind == DXC_OUT_NONE || !output.object)
        continue;
      IFR(SetOutput(output));
    }
    return S_OK;
  }

  HRESULT CopyOutputsFromResult(IDxcResult *pResult) {
    if (!pResult)
      return E_INVALIDARG;
    for (unsigned i = 0; i < kNumDxcOutputTypes; i++) {
      DxcOutputObject &output = m_outputs[i];
      DXC_OUT_KIND kind = (DXC_OUT_KIND)(i + 1);
      if (pResult->HasOutput(kind)) {
        IFR(pResult->GetOutput(kind, IID_PPV_ARGS(&output.object), &output.name));
        output.kind = kind;
      }
    }
    return S_OK;
  }

  // All-in-one initialization
  HRESULT Init(_In_ HRESULT status, _In_ DXC_OUT_KIND resultType,
               const llvm::ArrayRef<DxcOutputObject> outputs) {
    m_status = status;
    m_resultType = resultType;
    return SetOutputs(outputs);
  }

  // All-in-one create functions

  static HRESULT Create(_In_ HRESULT status, _In_ DXC_OUT_KIND resultType,
                        _In_opt_count_(numOutputs) const DxcOutputObject *pOutputs,
                        _In_ unsigned numOutputs,
                        _COM_Outptr_ IDxcResult **ppResult) {
    *ppResult = nullptr;
    CComPtr<DxcResult> result =
      DxcResult::Alloc(DxcGetThreadMallocNoRef());
    IFROOM(result.p);
    IFR(result->Init(status, resultType, llvm::ArrayRef<DxcOutputObject>(pOutputs, numOutputs)));
    *ppResult = result.Detach();
    return S_OK;
  }
  static HRESULT Create(_In_ HRESULT status, _In_ DXC_OUT_KIND resultType,
                        const llvm::ArrayRef<DxcOutputObject> outputs,
                        _COM_Outptr_ IDxcResult **ppResult) {
    return Create(status, resultType, outputs.data(), outputs.size(), ppResult);
  }
  // For convenient use in legacy interface implementations
  static HRESULT Create(_In_ HRESULT status, _In_ DXC_OUT_KIND resultType,
                        const llvm::ArrayRef<DxcOutputObject> outputs,
                        _COM_Outptr_ IDxcOperationResult **ppResult) {
    IDxcResult *pResult;
    IFR(Create(status, resultType, outputs.data(), outputs.size(), &pResult));
    *ppResult = pResult;
    return S_OK;
  }
};

#endif
