///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// FileIOHelper.h                                                            //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Provides utitlity functions to work with files.                           //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Global.h"

#ifndef _ATL_DECLSPEC_ALLOCATOR
#define _ATL_DECLSPEC_ALLOCATOR
#endif

// Forward declarations.
struct IDxcBlob;
struct IDxcBlobEncoding;
struct IDxcBlobUtf8;
struct IDxcBlobWide;

namespace hlsl {

IMalloc *GetGlobalHeapMalloc() throw();

class CDxcThreadMallocAllocator {
public:
  _Ret_maybenull_ _Post_writable_byte_size_(nBytes) _ATL_DECLSPEC_ALLOCATOR
  static void *Reallocate(_In_ void *p, _In_ size_t nBytes) throw() {
    return DxcGetThreadMallocNoRef()->Realloc(p, nBytes);
  }

  _Ret_maybenull_ _Post_writable_byte_size_(nBytes) _ATL_DECLSPEC_ALLOCATOR
  static void *Allocate(_In_ size_t nBytes) throw() {
    return DxcGetThreadMallocNoRef()->Alloc(nBytes);
  }

  static void Free(_In_ void *p) throw() {
    return DxcGetThreadMallocNoRef()->Free(p);
  }
};

// Like CComHeapPtr, but with CDxcThreadMallocAllocator.
template <typename T>
class CDxcTMHeapPtr :
  public CHeapPtr<T, CDxcThreadMallocAllocator>
{
public:
  CDxcTMHeapPtr() throw()
  {
  }

  explicit CDxcTMHeapPtr(_In_ T* pData) throw() :
    CHeapPtr<T, CDxcThreadMallocAllocator>(pData)
  {
  }
};

// Like CComHeapPtr, but with a stateful allocator.
template <typename T>
class CDxcMallocHeapPtr
{
private:
  CComPtr<IMalloc> m_pMalloc;
public:
  T *m_pData;

  CDxcMallocHeapPtr(IMalloc *pMalloc) throw()
      : m_pMalloc(pMalloc), m_pData(nullptr) {}

  ~CDxcMallocHeapPtr() {
    if (m_pData)
      m_pMalloc->Free(m_pData);
  }

  operator T *() const throw() { return m_pData; }

  IMalloc* GetMallocNoRef() const throw() { return m_pMalloc.p; }

  bool Allocate(_In_ SIZE_T ElementCount) throw() {
    ATLASSERT(m_pData == NULL);
    SIZE_T nBytes = ElementCount * sizeof(T);
    m_pData = static_cast<T *>(m_pMalloc->Alloc(nBytes));
    if (m_pData == NULL)
      return false;
    return true;
  }

  void AllocateBytes(_In_ SIZE_T ByteCount) throw() {
    if (m_pData)
      m_pMalloc->Free(m_pData);
    m_pData = static_cast<T *>(m_pMalloc->Alloc(ByteCount));
  }

  // Attach to an existing pointer (takes ownership)
  void Attach(_In_ T *pData) throw() {
    m_pMalloc->Free(m_pData);
    m_pData = pData;
  }

  // Detach the pointer (releases ownership)
  T *Detach() throw() {
    T *pTemp = m_pData;
    m_pData = NULL;
    return pTemp;
  }

  // Free the memory pointed to, and set the pointer to NULL
  void Free() throw() {
    m_pMalloc->Free(m_pData);
    m_pData = NULL;
  }
};

void ReadBinaryFile(_In_opt_ IMalloc *pMalloc,
                    _In_z_ LPCWSTR pFileName,
                    _Outptr_result_bytebuffer_(*pDataSize) void **ppData,
                    _Out_ DWORD *pDataSize);
void ReadBinaryFile(_In_z_ LPCWSTR pFileName,
                    _Outptr_result_bytebuffer_(*pDataSize) void **ppData,
                    _Out_ DWORD *pDataSize);
void WriteBinaryFile(_In_z_ LPCWSTR pFileName,
                     _In_reads_bytes_(DataSize) const void *pData,
                     _In_ DWORD DataSize);

///////////////////////////////////////////////////////////////////////////////
// Blob and encoding manipulation functions.

UINT32 DxcCodePageFromBytes(_In_count_(byteLen) const char *bytes,
                            size_t byteLen) throw();

// More general create blob functions, used by other functions
// Null pMalloc means use current thread malloc.
// bPinned will point to existing memory without managing it;
// bCopy will copy to heap; bPinned and bCopy are mutually exclusive.
// If encodingKnown, UTF-8 or wide, and null-termination possible,
// an IDxcBlobUtf8 or IDxcBlobWide will be constructed.
// If text, it's best if size includes null terminator when not copying,
// otherwise IDxcBlobUtf8 or IDxcBlobWide will not be constructed.
HRESULT DxcCreateBlob(
    LPCVOID pPtr, SIZE_T size, bool bPinned, bool bCopy,
    bool encodingKnown, UINT32 codePage,
    IMalloc *pMalloc, IDxcBlobEncoding **ppBlobEncoding) throw();
// Create from blob references original blob.
// Pass nonzero for offset or length for sub-blob reference.
HRESULT DxcCreateBlobEncodingFromBlob(
    IDxcBlob *pFromBlob, UINT32 offset, UINT32 length,
    bool encodingKnown, UINT32 codePage,
    IMalloc *pMalloc, IDxcBlobEncoding **ppBlobEncoding) throw();

// Load files
HRESULT
DxcCreateBlobFromFile(_In_opt_ IMalloc *pMalloc, LPCWSTR pFileName,
                      _In_opt_ UINT32 *pCodePage,
                      _COM_Outptr_ IDxcBlobEncoding **pBlobEncoding) throw();

HRESULT DxcCreateBlobFromFile(LPCWSTR pFileName, _In_opt_ UINT32 *pCodePage,
                              _COM_Outptr_ IDxcBlobEncoding **ppBlobEncoding) throw();

// Given a blob, creates a subrange view.
HRESULT DxcCreateBlobFromBlob(_In_ IDxcBlob *pBlob, UINT32 offset,
                              UINT32 length,
                              _COM_Outptr_ IDxcBlob **ppResult) throw();

// Creates a blob wrapping a buffer to be freed with the provided IMalloc
HRESULT
DxcCreateBlobOnMalloc(_In_bytecount_(size) LPCVOID pData, _In_ IMalloc* pIMalloc,
                      UINT32 size, _COM_Outptr_ IDxcBlob **ppResult) throw();

// Creates a blob with a copy of the provided data
HRESULT
DxcCreateBlobOnHeapCopy(_In_bytecount_(size) LPCVOID pData, UINT32 size,
                        _COM_Outptr_ IDxcBlob **ppResult) throw();

// Given a blob, creates a new instance with a specific code page set.
HRESULT
DxcCreateBlobWithEncodingSet(_In_ IDxcBlob *pBlob, UINT32 codePage,
                             _COM_Outptr_ IDxcBlobEncoding **ppBlobEncoding) throw();
HRESULT
DxcCreateBlobWithEncodingSet(
    _In_ IMalloc *pMalloc, _In_ IDxcBlob *pBlob, UINT32 codePage,
    _COM_Outptr_ IDxcBlobEncoding **ppBlobEncoding) throw();

// Creates a blob around encoded text without ownership transfer
HRESULT DxcCreateBlobWithEncodingFromPinned(
    _In_bytecount_(size) LPCVOID pText, UINT32 size, UINT32 codePage,
    _COM_Outptr_ IDxcBlobEncoding **pBlobEncoding) throw();

HRESULT DxcCreateBlobFromPinned(
    _In_bytecount_(size) LPCVOID pText, UINT32 size,
    _COM_Outptr_ IDxcBlob **pBlob) throw();

HRESULT
DxcCreateBlobWithEncodingFromStream(
    IStream *pStream, bool newInstanceAlways, UINT32 codePage,
    _COM_Outptr_ IDxcBlobEncoding **pBlobEncoding) throw();

// Creates a blob with a copy of the encoded text
HRESULT
DxcCreateBlobWithEncodingOnHeapCopy(
    _In_bytecount_(size) LPCVOID pText, UINT32 size, UINT32 codePage,
    _COM_Outptr_ IDxcBlobEncoding **pBlobEncoding) throw();

// Creates a blob wrapping encoded text to be freed with the provided IMalloc
HRESULT
DxcCreateBlobWithEncodingOnMalloc(
  _In_bytecount_(size) LPCVOID pText, _In_ IMalloc *pIMalloc, UINT32 size, UINT32 codePage,
  _COM_Outptr_ IDxcBlobEncoding **pBlobEncoding) throw();

// Creates a blob with a copy of encoded text, allocated using the provided IMalloc
HRESULT
DxcCreateBlobWithEncodingOnMallocCopy(
  _In_ IMalloc *pIMalloc, _In_bytecount_(size) LPCVOID pText, UINT32 size, UINT32 codePage,
  _COM_Outptr_ IDxcBlobEncoding **pBlobEncoding) throw();

HRESULT DxcGetBlobAsUtf8(_In_ IDxcBlob *pBlob, _In_ IMalloc *pMalloc,
                         _COM_Outptr_ IDxcBlobUtf8 **pBlobEncoding,
                         UINT32 defaultCodePage = CP_ACP) throw();
HRESULT
DxcGetBlobAsWide(_In_ IDxcBlob *pBlob, _In_ IMalloc *pMalloc,
                  _COM_Outptr_ IDxcBlobWide **pBlobEncoding) throw();

bool IsBlobNullOrEmpty(_In_opt_ IDxcBlob *pBlob) throw();

///////////////////////////////////////////////////////////////////////////////
// Stream implementations.
class AbstractMemoryStream : public IStream {
public:
  virtual LPBYTE GetPtr() throw() = 0;
  virtual ULONG GetPtrSize() throw() = 0;
  virtual LPBYTE Detach() throw() = 0;
  virtual UINT64 GetPosition() throw() = 0;
  virtual HRESULT Reserve(ULONG targetSize) throw() = 0;
};
HRESULT CreateMemoryStream(_In_ IMalloc *pMalloc, _COM_Outptr_ AbstractMemoryStream** ppResult) throw();
HRESULT CreateReadOnlyBlobStream(_In_ IDxcBlob *pSource, _COM_Outptr_ IStream** ppResult) throw();
HRESULT CreateFixedSizeMemoryStream(_In_ LPBYTE pBuffer, size_t size, _COM_Outptr_ AbstractMemoryStream** ppResult) throw();

template <typename T>
HRESULT WriteStreamValue(IStream *pStream, const T& value) {
  ULONG cb;
  return pStream->Write(&value, sizeof(value), &cb);
}

} // namespace hlsl
