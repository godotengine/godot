///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// dxcapi.use.cpp                                                            //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Provides support for DXC API users.                                       //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/Support/WinIncludes.h"
#include "dxc/Support/dxcapi.use.h"
#include "dxc/Support/Global.h"
#include "dxc/Support/Unicode.h"
#include "dxc/Support/FileIOHelper.h"
#include "dxc/Support/WinFunctions.h"

namespace dxc {

#ifdef _WIN32
static void TrimEOL(_Inout_z_ char *pMsg) {
  char *pEnd = pMsg + strlen(pMsg);
  --pEnd;
  while (pEnd > pMsg && (*pEnd == '\r' || *pEnd == '\n')) {
    --pEnd;
  }
  pEnd[1] = '\0';
}

static std::string GetWin32ErrorMessage(DWORD err) {
  char formattedMsg[200];
  DWORD formattedMsgLen =
      FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                    nullptr, err, 0, formattedMsg, _countof(formattedMsg), 0);
  if (formattedMsg > 0 && formattedMsgLen < _countof(formattedMsg)) {
    TrimEOL(formattedMsg);
    return std::string(formattedMsg);
  }
  return std::string();
}
#else
static std::string GetWin32ErrorMessage(DWORD err) {
  // Since we use errno for handling messages, we use strerror to get the error
  // message.
  return std::string(std::strerror(err));
}
#endif // _WIN32

void IFT_Data(HRESULT hr, LPCWSTR data) {
  if (SUCCEEDED(hr)) return;
  CW2A pData(data, CP_UTF8);
  std::string errMsg;
  if (HRESULT_IS_WIN32ERR(hr)) {
    DWORD err = HRESULT_AS_WIN32ERR(hr);
    errMsg.append(GetWin32ErrorMessage(err));
    if (data != nullptr) {
      errMsg.append(" ", 1);
    }
  }
  if (data != nullptr) {
    errMsg.append(pData);
  }
  throw ::hlsl::Exception(hr, errMsg);
}

void EnsureEnabled(DxcDllSupport &dxcSupport) {
  if (!dxcSupport.IsEnabled()) {
    IFT(dxcSupport.Initialize());
  }
}

void ReadFileIntoBlob(DxcDllSupport &dxcSupport, _In_ LPCWSTR pFileName,
                      _COM_Outptr_ IDxcBlobEncoding **ppBlobEncoding) {
  CComPtr<IDxcLibrary> library;
  IFT(dxcSupport.CreateInstance(CLSID_DxcLibrary, &library));
  IFT_Data(library->CreateBlobFromFile(pFileName, nullptr, ppBlobEncoding),
           pFileName);
}

void WriteOperationErrorsToConsole(_In_ IDxcOperationResult *pResult,
                                   bool outputWarnings) {
  HRESULT status;
  IFT(pResult->GetStatus(&status));
  if (FAILED(status) || outputWarnings) {
    CComPtr<IDxcBlobEncoding> pErrors;
    IFT(pResult->GetErrorBuffer(&pErrors));
    if (pErrors.p != nullptr) {
      WriteBlobToConsole(pErrors, STD_ERROR_HANDLE);
    }
  }
}

void WriteOperationResultToConsole(_In_ IDxcOperationResult *pRewriteResult,
                                   bool outputWarnings) {
  WriteOperationErrorsToConsole(pRewriteResult, outputWarnings);

  CComPtr<IDxcBlob> pBlob;
  IFT(pRewriteResult->GetResult(&pBlob));
  WriteBlobToConsole(pBlob, STD_OUTPUT_HANDLE);
}

static void WriteWideNullTermToConsole(_In_opt_count_(charCount) const wchar_t *pText,
                                 DWORD streamType) {
  if (pText == nullptr) {
    return;
  }

  bool lossy; // Note: even if there was loss,  print anyway
  std::string consoleMessage;
  Unicode::WideToConsoleString(pText, &consoleMessage, &lossy);
  if (streamType == STD_OUTPUT_HANDLE) {
    fprintf(stdout, "%s\n", consoleMessage.c_str());
  }
  else if (streamType == STD_ERROR_HANDLE) {
    fprintf(stderr, "%s\n", consoleMessage.c_str());
  }
  else {
    throw hlsl::Exception(E_INVALIDARG);
  }
}

static HRESULT BlobToUtf8IfText(_In_opt_ IDxcBlob *pBlob, IDxcBlobUtf8 **ppBlobUtf8) {
  CComPtr<IDxcBlobEncoding> pBlobEncoding;
  if (SUCCEEDED(pBlob->QueryInterface(&pBlobEncoding))) {
    BOOL known;
    UINT32 cp = 0;
    IFT(pBlobEncoding->GetEncoding(&known, &cp));
    if (known) {
      return hlsl::DxcGetBlobAsUtf8(pBlob, nullptr, ppBlobUtf8);
    }
  }
  return S_OK;
}

static HRESULT BlobToWideIfText(_In_opt_ IDxcBlob *pBlob, IDxcBlobWide **ppBlobWide) {
  CComPtr<IDxcBlobEncoding> pBlobEncoding;
  if (SUCCEEDED(pBlob->QueryInterface(&pBlobEncoding))) {
    BOOL known;
    UINT32 cp = 0;
    IFT(pBlobEncoding->GetEncoding(&known, &cp));
    if (known) {
      return hlsl::DxcGetBlobAsWide(pBlob, nullptr, ppBlobWide);
    }
  }
  return S_OK;
}

void WriteBlobToConsole(_In_opt_ IDxcBlob *pBlob, DWORD streamType) {
  if (pBlob == nullptr) {
    return;
  }

  // Try to get as UTF-16 or UTF-8
  BOOL known;
  UINT32 cp = 0;
  CComPtr<IDxcBlobEncoding> pBlobEncoding;
  IFT(pBlob->QueryInterface(&pBlobEncoding));
  IFT(pBlobEncoding->GetEncoding(&known, &cp));

  if (cp == DXC_CP_WIDE) {
    CComPtr<IDxcBlobWide> pWide;
    IFT(hlsl::DxcGetBlobAsWide(pBlob, nullptr, &pWide));
    WriteWideNullTermToConsole(pWide->GetStringPointer(), streamType);
  } else if (cp == CP_UTF8) {
    CComPtr<IDxcBlobUtf8> pUtf8;
    IFT(hlsl::DxcGetBlobAsUtf8(pBlob, nullptr, &pUtf8));
    WriteUtf8ToConsoleSizeT(pUtf8->GetStringPointer(), pUtf8->GetStringLength(), streamType);
  }
}

void WriteBlobToFile(_In_opt_ IDxcBlob *pBlob, _In_ LPCWSTR pFileName, _In_ UINT32 textCodePage) {
  if (pBlob == nullptr) {
    return;
  }

  CHandle file(CreateFileW(pFileName, GENERIC_WRITE, FILE_SHARE_READ, nullptr,
    CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr));
  if (file == INVALID_HANDLE_VALUE) {
    IFT_Data(HRESULT_FROM_WIN32(GetLastError()), pFileName);
  }

  WriteBlobToHandle(pBlob, file, pFileName, textCodePage);
}

void WriteBlobToHandle(_In_opt_ IDxcBlob *pBlob, _In_ HANDLE hFile, _In_opt_ LPCWSTR pFileName, _In_ UINT32 textCodePage) {
  if (pBlob == nullptr) {
    return;
  }

  LPCVOID pPtr = pBlob->GetBufferPointer();
  SIZE_T size = pBlob->GetBufferSize();

  std::string BOM;
  CComPtr<IDxcBlobUtf8> pBlobUtf8;
  CComPtr<IDxcBlobWide> pBlobWide;
  if (textCodePage == DXC_CP_UTF8) {
    IFT_Data(BlobToUtf8IfText(pBlob, &pBlobUtf8), pFileName);
    if (pBlobUtf8) {
      pPtr = pBlobUtf8->GetStringPointer();
      size = pBlobUtf8->GetStringLength();
      // TBD: Should we write UTF-8 BOM?
      //BOM = "\xef\xbb\xbf"; // UTF-8
    }
  } else if (textCodePage == DXC_CP_WIDE) {
    IFT_Data(BlobToWideIfText(pBlob, &pBlobWide), pFileName);
    if (pBlobWide) {
      pPtr = pBlobWide->GetStringPointer();
      size = pBlobWide->GetStringLength() * sizeof(wchar_t);
      BOM = "\xff\xfe"; // UTF-16 LE
    }
  }

  IFT_Data(size > (SIZE_T)UINT32_MAX ? E_OUTOFMEMORY : S_OK , pFileName);

  DWORD written;

  if (!BOM.empty()) {
    if (FALSE == WriteFile(hFile, BOM.data(), BOM.length(), &written, nullptr)) {
      IFT_Data(HRESULT_FROM_WIN32(GetLastError()), pFileName);
    }
  }

  if (FALSE == WriteFile(hFile, pPtr, (DWORD)size, &written, nullptr)) {
    IFT_Data(HRESULT_FROM_WIN32(GetLastError()), pFileName);
  }
}

void WriteUtf8ToConsole(_In_opt_count_(charCount) const char *pText,
                        int charCount, DWORD streamType) {
  if (charCount == 0 || pText == nullptr) {
    return;
  }

  std::string resultToPrint;
  wchar_t *wideMessage = nullptr;
  size_t wideMessageLen;
  Unicode::UTF8BufferToWideBuffer(pText, charCount, &wideMessage,
                                   &wideMessageLen);

  WriteWideNullTermToConsole(wideMessage, streamType);

  delete[] wideMessage;
}

void WriteUtf8ToConsoleSizeT(_In_opt_count_(charCount) const char *pText,
  size_t charCount, DWORD streamType) {
  if (charCount == 0) {
    return;
  }

  int charCountInt = 0;
  IFT(SizeTToInt(charCount, &charCountInt));
  WriteUtf8ToConsole(pText, charCountInt, streamType);
}

} // namespace dxc
