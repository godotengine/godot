//===- llvm/Support/Windows/MSFileSystemBasic.cpp DXComplier Impl *- C++ -*-===//
///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// MSFileSystemBasic.cpp                                                     //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// This file implements the DXCompiler specific implementation of the Path API.//
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#ifdef _WIN32

#include "dxc/Support/WinIncludes.h"
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdint.h>
#include <errno.h>
#include <D3Dcommon.h>
#include <new>
#include <unordered_map>

#include "llvm/Support/MSFileSystem.h"
#include "dxc/Support/Global.h"

#include "dxc/dxcapi.internal.h"


///////////////////////////////////////////////////////////////////////////////////////////////////
// Externally visible functions.

/// <summary>Creates an implementation based on IDxcSystemAccess.</summary>
HRESULT CreateMSFileSystemForIface(_In_ IUnknown* pService, _COM_Outptr_ ::llvm::sys::fs::MSFileSystem** pResult) throw();

/// <summary>Creates an implementation with no access to system resources.</summary>
HRESULT CreateMSFileSystemBlocked(_COM_Outptr_ ::llvm::sys::fs::MSFileSystem** pResult) throw();

///////////////////////////////////////////////////////////////////////////////////////////////////
// Helper functions.

static
DWORD WIN32_FROM_HRESULT(HRESULT hr)
{
  if (SUCCEEDED(hr)) return ERROR_SUCCESS;
  if ((hr & 0xFFFF0000) == MAKE_HRESULT(SEVERITY_ERROR, FACILITY_WIN32, 0))
  {
    // Could have come from many values, but we choose this one
    return HRESULT_CODE(hr);
  }
  if (hr == E_OUTOFMEMORY) return ERROR_OUTOFMEMORY;
  if (hr == E_NOTIMPL) return ERROR_CALL_NOT_IMPLEMENTED;
  return ERROR_FUNCTION_FAILED;
}

static
HRESULT CopyStatStg(_In_ const STATSTG* statStg, _Out_ LPWIN32_FIND_DATAW lpFindFileData)
{
  HRESULT hr = S_OK;
  lpFindFileData->dwFileAttributes = FILE_ATTRIBUTE_NORMAL;
  lpFindFileData->ftCreationTime = statStg->ctime;
  lpFindFileData->ftLastAccessTime = statStg->atime;
  lpFindFileData->ftLastWriteTime = statStg->mtime;
  lpFindFileData->nFileSizeLow = statStg->cbSize.LowPart;
  lpFindFileData->nFileSizeHigh = statStg->cbSize.HighPart;
  if (statStg->pwcsName != nullptr)
  {
    IFC(StringCchCopyW(lpFindFileData->cFileName, _countof(lpFindFileData->cFileName), statStg->pwcsName));
  }

Cleanup:
  return hr;
}

static
void ClearStatStg(_Inout_ STATSTG* statStg)
{
  DXASSERT_NOMSG(statStg != nullptr);
  if (statStg->pwcsName != nullptr)
  {
    CoTaskMemFree(statStg->pwcsName);
    statStg->pwcsName = nullptr;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// IDxcSystemAccess-based MSFileSystem implementation.

static const int FirstAllocFD = 10;
static const int LastAllocFD = 8 * 1024;
static const HANDLE FirstAllocHandle = (HANDLE)(uintptr_t)FirstAllocFD;
static const HANDLE LastAllocHandle = (HANDLE)(uintptr_t)LastAllocFD;

struct MSFileSystemHandle
{
  enum MSFileSystemHandleKind
  {
    MSFileSystemHandleKind_FindHandle,
    MSFileSystemHandleKind_FileHandle,
    MSFileSystemHandleKind_FileMappingHandle
  };

  MSFileSystemHandleKind kind;
  CComPtr<IUnknown> storage;// For a file handle, the stream or directory handle.
                            // For a find handle, the IEnumSTATSTG associated.
  CComPtr<IStream> stream;  // For a file or console file handle, the stream interface.
  int fd;                   // For a file handle, its file descriptor.

  MSFileSystemHandle(int knownFD)
    : fd(knownFD)
    , kind(MSFileSystemHandleKind_FileHandle)
  {
  }

  MSFileSystemHandle(IUnknown* pMapping)
    : storage(pMapping)
    , kind(MSFileSystemHandleKind_FileMappingHandle)
    , fd(0)
  {
  }

  MSFileSystemHandle(IUnknown* pStorage, IStream* pStream)
    : storage(pStorage)
    , stream(pStream)
    , kind(MSFileSystemHandleKind_FileHandle)
    , fd(0)
  {
  }

  MSFileSystemHandle(IEnumSTATSTG* pEnumSTATG) : storage(pEnumSTATG), kind(MSFileSystemHandleKind_FindHandle)
  {
  }

  MSFileSystemHandle(MSFileSystemHandle&& other)
  {
    kind = other.kind;
    storage.p = other.storage.Detach();
    stream.p = other.stream.Detach();
  }
  
  HANDLE GetHandle() const { return (HANDLE)this; }

  IEnumSTATSTG* GetEnumStatStg()
  {
    DXASSERT(kind == MSFileSystemHandleKind_FindHandle, "otherwise caller didn't check");
    return (IEnumSTATSTG*)storage.p;
  }
};

namespace llvm {
namespace sys  {
namespace fs {

class MSFileSystemForIface : public MSFileSystem
{
private:
  CComPtr<IDxcSystemAccess> m_system;
  typedef std::unordered_multimap<LPCVOID, ID3D10Blob*> TViewMap;
  TViewMap m_mappingViews;
  MSFileSystemHandle m_knownHandle0;
  MSFileSystemHandle m_knownHandle1;
  MSFileSystemHandle m_knownHandle2;

  HRESULT AddFindHandle(_In_ IEnumSTATSTG* enumStatStg, _Out_ HANDLE* pResult) throw();
  HRESULT AddFileHandle(_In_ IUnknown* storage, _In_ IStream* stream, _Out_ HANDLE* pResult) throw();
  HRESULT AddMappingHandle(_In_ IUnknown* mapping, _Out_ HANDLE* pResult) throw();
  HRESULT AddMappingView(_In_ ID3D10Blob* blob) throw();
  HRESULT EnsureFDAvailable(int fd);
  HANDLE GetHandleForFD(int fd) throw();
  void GetFindHandle(HANDLE findHandle, _Outptr_ IEnumSTATSTG** enumStatStg) throw();
  int GetHandleFD(HANDLE fileHandle) throw();
  void GetHandleMapping(HANDLE fileHandle, _Outptr_ IUnknown** pResult) throw();
  void GetHandleStorage(HANDLE fileHandle, _Outptr_ IUnknown** pResult) throw();
  void GetHandleStream(HANDLE fileHandle, _Outptr_ IStream** pResult) throw();
  void CloseInternalHandle(HANDLE findHandle) throw();
  void RemoveMappingView(_In_ LPCVOID address) throw();

public:
  MSFileSystemForIface(_In_ IDxcSystemAccess* access);

  virtual BOOL FindNextFileW(_In_ HANDLE hFindFile, _Out_ LPWIN32_FIND_DATAW lpFindFileData) throw() override;
  virtual HANDLE FindFirstFileW(_In_ LPCWSTR lpFileName, _Out_ LPWIN32_FIND_DATAW lpFindFileData) throw() override;
  virtual void FindClose(HANDLE findHandle) throw() override;
  virtual HANDLE CreateFileW(_In_ LPCWSTR lpFileName, _In_ DWORD dwDesiredAccess, _In_ DWORD dwShareMode, _In_ DWORD dwCreationDisposition, _In_ DWORD dwFlagsAndAttributes) throw() override;
  virtual BOOL SetFileTime(_In_ HANDLE hFile, _In_opt_ const FILETIME *lpCreationTime, _In_opt_ const FILETIME *lpLastAccessTime, _In_opt_ const FILETIME *lpLastWriteTime) throw() override;
  virtual BOOL GetFileInformationByHandle(_In_ HANDLE hFile, _Out_ LPBY_HANDLE_FILE_INFORMATION lpFileInformation) throw() override;
  virtual DWORD GetFileType(_In_ HANDLE hFile) throw() override;
  virtual BOOL CreateHardLinkW(_In_ LPCWSTR lpFileName, _In_ LPCWSTR lpExistingFileName) throw() override;
  virtual BOOL MoveFileExW(_In_ LPCWSTR lpExistingFileName, _In_opt_ LPCWSTR lpNewFileName, _In_ DWORD dwFlags) throw() override;
  virtual DWORD GetFileAttributesW(_In_ LPCWSTR lpFileName) throw() override;
  virtual BOOL CloseHandle(_In_ HANDLE hObject) throw() override;
  virtual BOOL DeleteFileW(_In_ LPCWSTR lpFileName) throw() override;
  virtual BOOL RemoveDirectoryW(_In_ LPCWSTR lpFileName) throw() override;
  virtual BOOL CreateDirectoryW(_In_ LPCWSTR lpPathName) throw() override;
  _Success_(return != 0 && return < nBufferLength)
  virtual DWORD GetCurrentDirectoryW(_In_ DWORD nBufferLength, _Out_writes_to_opt_(nBufferLength, return +1) LPWSTR lpBuffer) throw() override;
  _Success_(return != 0 && return < nSize)
  virtual DWORD GetMainModuleFileNameW(__out_ecount_part(nSize, return +1) LPWSTR lpFilename, DWORD nSize) throw() override;
  virtual DWORD GetTempPathW(DWORD nBufferLength, _Out_writes_to_opt_(nBufferLength, return +1) LPWSTR lpBuffer) throw() override;
  virtual BOOLEAN CreateSymbolicLinkW(_In_ LPCWSTR lpSymlinkFileName, _In_ LPCWSTR lpTargetFileName, DWORD dwFlags) throw() override;
  virtual bool SupportsCreateSymbolicLink() throw() override;
  virtual BOOL ReadFile(_In_ HANDLE hFile, _Out_ LPVOID lpBuffer, _In_ DWORD nNumberOfBytesToRead, _Out_opt_ LPDWORD lpNumberOfBytesRead) throw() override;
  virtual HANDLE CreateFileMappingW(_In_ HANDLE hFile, _In_ DWORD flProtect, _In_ DWORD dwMaximumSizeHigh, _In_ DWORD dwMaximumSizeLow) throw() override;
  virtual LPVOID MapViewOfFile(_In_ HANDLE hFileMappingObject, _In_ DWORD dwDesiredAccess, _In_ DWORD dwFileOffsetHigh, _In_ DWORD dwFileOffsetLow, _In_ SIZE_T dwNumberOfBytesToMap) throw() override;
  virtual BOOL UnmapViewOfFile(_In_ LPCVOID lpBaseAddress) throw() override;
  
  // Console APIs.
  virtual bool FileDescriptorIsDisplayed(int fd) throw() override;
  virtual unsigned GetColumnCount(DWORD nStdHandle) throw() override;
  virtual unsigned GetConsoleOutputTextAttributes() throw() override;
  virtual void SetConsoleOutputTextAttributes(unsigned attributes) throw() override;
  virtual void ResetConsoleOutputTextAttributes() throw() override;

  // CRT APIs.
  virtual int open_osfhandle(intptr_t osfhandle, int flags) throw() override;
  virtual intptr_t get_osfhandle(int fd) throw() override;
  virtual int close(int fd) throw() override;
  virtual long lseek(int fd, long offset, int origin) throw() override;
  virtual int setmode(int fd, int mode) throw() override;
  virtual errno_t resize_file(_In_ LPCWSTR path, uint64_t size) throw() override;
  virtual int Read(int fd, _Out_bytecap_(count) void* buffer, unsigned int count) throw() override;
  virtual int Write(int fd, _In_bytecount_(count) const void* buffer, unsigned int count) throw() override;
#ifndef _WIN32
  virtual int Open(const char *lpFileName, int flags, mode_t mode) throw() override;
  virtual int Stat(const char *lpFileName, struct stat *Status) throw() override;
  virtual int Fstat(int FD, struct stat *Status) throw() override;
#endif
};

_Use_decl_annotations_
MSFileSystemForIface::MSFileSystemForIface(IDxcSystemAccess* systemAccess)
  : m_system(systemAccess)
  , m_knownHandle0(0)
  , m_knownHandle1(1)
  , m_knownHandle2(2)
{
}

_Use_decl_annotations_
HRESULT MSFileSystemForIface::AddMappingHandle(IUnknown* mapping, HANDLE* pResult) throw()
{
  DXASSERT_NOMSG(mapping != nullptr);
  DXASSERT_NOMSG(pResult != nullptr);

  HRESULT hr = S_OK;
  MSFileSystemHandle* handle = nullptr;
  *pResult = INVALID_HANDLE_VALUE;
  handle = new (std::nothrow)MSFileSystemHandle(mapping);
  IFCOOM(handle);
  *pResult = handle->GetHandle();

Cleanup:
  return hr;
}

_Use_decl_annotations_
HRESULT MSFileSystemForIface::AddMappingView(ID3D10Blob* blob) throw()
{
  DXASSERT_NOMSG(blob != nullptr);
  LPVOID address = blob->GetBufferPointer();
  try
  {
    m_mappingViews.insert(std::pair<LPVOID, ID3D10Blob*>(address, blob));
  }
  catch (std::bad_alloc&)
  {
    return E_OUTOFMEMORY;
  }
  blob->AddRef();
  return S_OK;
}

_Use_decl_annotations_
HRESULT MSFileSystemForIface::AddFindHandle(IEnumSTATSTG* enumStatStg, HANDLE* pResult) throw()
{
  DXASSERT_NOMSG(enumStatStg != nullptr);
  DXASSERT_NOMSG(pResult != nullptr);

  HRESULT hr = S_OK;
  MSFileSystemHandle* handle = nullptr;
  *pResult = INVALID_HANDLE_VALUE;
  handle = new (std::nothrow)MSFileSystemHandle(enumStatStg);
  IFCOOM(handle);
  *pResult = handle->GetHandle();

Cleanup:
  return hr;
}

_Use_decl_annotations_
HRESULT MSFileSystemForIface::AddFileHandle(IUnknown* storage, IStream* stream, HANDLE* pResult) throw()
{
  DXASSERT_NOMSG(storage != nullptr);
  DXASSERT_NOMSG(pResult != nullptr);

  HRESULT hr = S_OK;
  MSFileSystemHandle* handle = nullptr;
  *pResult = INVALID_HANDLE_VALUE;
  handle = new (std::nothrow)MSFileSystemHandle(storage, stream);
  IFCOOM(handle);
  *pResult = handle->GetHandle();

Cleanup:
  return hr;
}

void MSFileSystemForIface::CloseInternalHandle(HANDLE handle) throw()
{
  DXASSERT_NOMSG(handle != nullptr);
  DXASSERT_NOMSG(handle != INVALID_HANDLE_VALUE);
  MSFileSystemHandle* fsHandle = reinterpret_cast<MSFileSystemHandle*>(handle);
  if (fsHandle == &m_knownHandle0 || fsHandle == &m_knownHandle1 || fsHandle == &m_knownHandle2)
  {
    fsHandle->stream.Release();
    fsHandle->storage.Release();
  }
  else
  {
    delete fsHandle;
  }
}

_Use_decl_annotations_
void MSFileSystemForIface::RemoveMappingView(LPCVOID address) throw()
{
  TViewMap::iterator i = m_mappingViews.find(address);
  DXASSERT(i != m_mappingViews.end(), "otherwise pointer to view isn't in map");
  DXASSERT(i->second != nullptr, "otherwise blob is null and should not have been added");
  i->second->Release();
  m_mappingViews.erase(i);
}

_Use_decl_annotations_
void MSFileSystemForIface::GetFindHandle(HANDLE findHandle, IEnumSTATSTG** enumStatStg) throw()
{
  DXASSERT_NOMSG(findHandle != nullptr);
  DXASSERT_NOMSG(enumStatStg != nullptr);
  
  MSFileSystemHandle* fsHandle = reinterpret_cast<MSFileSystemHandle*>(findHandle);
  DXASSERT(fsHandle->kind == MSFileSystemHandle::MSFileSystemHandleKind_FindHandle, "otherwise caller is passing wrong handle to API");

  *enumStatStg = fsHandle->GetEnumStatStg();
  DXASSERT(*enumStatStg != nullptr, "otherwise it should not have been added to handle entry");

  (*enumStatStg)->AddRef();
}

int MSFileSystemForIface::GetHandleFD(HANDLE fileHandle) throw()
{
  DXASSERT_NOMSG(fileHandle != nullptr);

  MSFileSystemHandle* fsHandle = reinterpret_cast<MSFileSystemHandle*>(fileHandle);
  DXASSERT(fsHandle->kind == MSFileSystemHandle::MSFileSystemHandleKind_FileHandle, "otherwise caller is passing wrong handle to API");

  return fsHandle->fd;
}

_Use_decl_annotations_
void MSFileSystemForIface::GetHandleMapping(HANDLE mapping, _Outptr_ IUnknown** pResult) throw()
{
  DXASSERT_NOMSG(mapping != nullptr);
  DXASSERT_NOMSG(pResult != nullptr);

  MSFileSystemHandle* fsHandle = reinterpret_cast<MSFileSystemHandle*>(mapping);
  DXASSERT(fsHandle->kind == MSFileSystemHandle::MSFileSystemHandleKind_FileMappingHandle, "otherwise caller is passing wrong handle to API");

  *pResult = fsHandle->storage.p;
  DXASSERT(*pResult != nullptr, "otherwise it should not be requested through GetHandleMapping");

  (*pResult)->AddRef();
}

_Use_decl_annotations_
void MSFileSystemForIface::GetHandleStorage(HANDLE fileHandle, _Outptr_ IUnknown** pResult) throw()
{
  DXASSERT_NOMSG(fileHandle != nullptr);
  DXASSERT_NOMSG(pResult != nullptr);

  MSFileSystemHandle* fsHandle = reinterpret_cast<MSFileSystemHandle*>(fileHandle);
  DXASSERT(fsHandle->kind == MSFileSystemHandle::MSFileSystemHandleKind_FileHandle, "otherwise caller is passing wrong handle to API");

  *pResult = fsHandle->storage.p;
  DXASSERT(*pResult != nullptr, "otherwise it should not be requested through GetHandleStorage");

  (*pResult)->AddRef();
}

_Use_decl_annotations_
void MSFileSystemForIface::GetHandleStream(HANDLE fileHandle, _Outptr_ IStream** pResult) throw()
{
  DXASSERT_NOMSG(fileHandle != nullptr);
  DXASSERT_NOMSG(pResult != nullptr);

  MSFileSystemHandle* fsHandle = reinterpret_cast<MSFileSystemHandle*>(fileHandle);
  DXASSERT(fsHandle->kind == MSFileSystemHandle::MSFileSystemHandleKind_FileHandle, "otherwise caller is passing wrong handle to API");

  *pResult = fsHandle->stream.p;
  DXASSERT(*pResult != nullptr, "otherwise it should not be requested through GetHandleStream");

  (*pResult)->AddRef();
}


_Use_decl_annotations_
HANDLE MSFileSystemForIface::FindFirstFileW(LPCWSTR lpFileName, LPWIN32_FIND_DATAW lpFindFileData) throw()
{
  HRESULT hr = S_OK;
  CComPtr<IEnumSTATSTG> enumStatStg;
  HANDLE resultValue = INVALID_HANDLE_VALUE;
  STATSTG elt;
  ULONG fetched;

  ZeroMemory(&elt, sizeof(elt));
  ZeroMemory(lpFindFileData, sizeof(*lpFindFileData));
  fetched = 0;

  IFC(m_system->EnumFiles(lpFileName, &enumStatStg));
  IFC(enumStatStg->Next(1, &elt, &fetched));
  if (fetched == 0)
  {
    IFC(HRESULT_FROM_WIN32(ERROR_FILE_NOT_FOUND));
  }
  else
  {
    IFC(CopyStatStg(&elt, lpFindFileData));
    IFC(AddFindHandle(enumStatStg, &resultValue));
  }

Cleanup:
  ClearStatStg(&elt);
  if (FAILED(hr))
  {
    SetLastError(WIN32_FROM_HRESULT(hr));
    return INVALID_HANDLE_VALUE;
  }

  DXASSERT(resultValue != INVALID_HANDLE_VALUE, "otherwise AddFindHandle failed to return a valid handle");
  return resultValue;
}

_Use_decl_annotations_
BOOL MSFileSystemForIface::FindNextFileW(HANDLE hFindFile, LPWIN32_FIND_DATAW lpFindFileData) throw()
{
  HRESULT hr = S_OK;
  CComPtr<IEnumSTATSTG> enumStatStg;
  BOOL resultValue = FALSE;
  STATSTG elt;
  ULONG fetched;

  ZeroMemory(&elt, sizeof(elt));
  ZeroMemory(lpFindFileData, sizeof(*lpFindFileData));
  fetched = 0;

  GetFindHandle(hFindFile, &enumStatStg);
  IFC(enumStatStg->Next(1, &elt, &fetched));
  if (fetched == 0)
  {
    IFC(HRESULT_FROM_WIN32(ERROR_FILE_NOT_FOUND));
  }
  else
  {
    IFC(CopyStatStg(&elt, lpFindFileData));
    resultValue = TRUE;
  }

Cleanup:
  if (FAILED(hr))
  {
    SetLastError(WIN32_FROM_HRESULT(hr));
    return FALSE;
  }

  return TRUE;
}

void MSFileSystemForIface::FindClose(HANDLE findHandle) throw()
{
  CloseInternalHandle(findHandle);
}

_Use_decl_annotations_
HANDLE MSFileSystemForIface::CreateFileW(LPCWSTR lpFileName, DWORD dwDesiredAccess, DWORD dwShareMode, DWORD dwCreationDisposition, DWORD dwFlagsAndAttributes) throw()
{
  HRESULT hr = S_OK;
  CComPtr<IUnknown> storage;
  CComPtr<IStream> stream;
  HANDLE resultHandle = INVALID_HANDLE_VALUE;
  
  IFC(m_system->OpenStorage(lpFileName, dwDesiredAccess, dwShareMode, dwCreationDisposition, dwFlagsAndAttributes, &storage));
  IFC(storage.QueryInterface(&stream));
  IFC(AddFileHandle(storage, stream, &resultHandle));

Cleanup:
  if (FAILED(hr))
  {
    SetLastError(WIN32_FROM_HRESULT(hr));
    return INVALID_HANDLE_VALUE;
  }

  return resultHandle;
}

_Use_decl_annotations_
BOOL MSFileSystemForIface::SetFileTime(HANDLE hFile, _In_opt_ const FILETIME *lpCreationTime, _In_opt_ const FILETIME *lpLastAccessTime, _In_opt_ const FILETIME *lpLastWriteTime) throw()
{
  HRESULT hr = S_OK;
  CComPtr<IUnknown> storage;

  GetHandleStorage(hFile, &storage);
  IFC(m_system->SetStorageTime(storage, lpCreationTime, lpLastAccessTime, lpLastWriteTime));

Cleanup:
  if (FAILED(hr))
  {
    SetLastError(WIN32_FROM_HRESULT(hr));
    return FALSE;
  }

  return TRUE;
}

_Use_decl_annotations_
BOOL MSFileSystemForIface::GetFileInformationByHandle(HANDLE hFile, LPBY_HANDLE_FILE_INFORMATION lpFileInformation) throw()
{
  HRESULT hr = S_OK;
  CComPtr<IUnknown> storage;

  GetHandleStorage(hFile, &storage);
  IFC(m_system->GetFileInformationForStorage(storage, lpFileInformation));

Cleanup:
  if (FAILED(hr))
  {
    SetLastError(WIN32_FROM_HRESULT(hr));
    return FALSE;
  }

  return TRUE;
}

_Use_decl_annotations_
DWORD MSFileSystemForIface::GetFileType(HANDLE hFile) throw()
{
  HRESULT hr = S_OK;
  CComPtr<IUnknown> storage;
  DWORD fileType;

  GetHandleStorage(hFile, &storage);
  IFC(m_system->GetFileTypeForStorage(storage, &fileType));
  if (fileType == FILE_TYPE_UNKNOWN)
  {
    SetLastError(NO_ERROR);
  }

Cleanup:
  if (FAILED(hr))
  {
    SetLastError(WIN32_FROM_HRESULT(hr));
    fileType = FILE_TYPE_UNKNOWN;
  }

  return fileType;
}

_Use_decl_annotations_
BOOL MSFileSystemForIface::CreateHardLinkW(LPCWSTR lpFileName, LPCWSTR lpExistingFileName) throw()
{
  SetLastError(ERROR_FUNCTION_NOT_CALLED);
  return FALSE;
}

_Use_decl_annotations_
BOOL MSFileSystemForIface::MoveFileExW(LPCWSTR lpExistingFileName, LPCWSTR lpNewFileName, DWORD dwFlags) throw()
{
  SetLastError(ERROR_FUNCTION_NOT_CALLED);
  return FALSE;
}

_Use_decl_annotations_
DWORD MSFileSystemForIface::GetFileAttributesW(LPCWSTR lpFileName) throw()
{
  HRESULT hr = S_OK;
  DWORD attributes;

  IFC(m_system->GetFileAttributesForStorage(lpFileName, &attributes));

Cleanup:
  if (FAILED(hr))
  {
    SetLastError(WIN32_FROM_HRESULT(hr));
    attributes = INVALID_FILE_ATTRIBUTES;
  }

  return attributes;
}

_Use_decl_annotations_
BOOL MSFileSystemForIface::CloseHandle(HANDLE hObject) throw()
{
  this->CloseInternalHandle(hObject);
  return TRUE;
}

_Use_decl_annotations_
BOOL MSFileSystemForIface::DeleteFileW(LPCWSTR lpFileName) throw()
{
  SetLastError(ERROR_FUNCTION_NOT_CALLED);
  return FALSE;
}

_Use_decl_annotations_
BOOL MSFileSystemForIface::RemoveDirectoryW(LPCWSTR lpFileName) throw()
{
  SetLastError(ERROR_FUNCTION_NOT_CALLED);
  return FALSE;
}

_Use_decl_annotations_
BOOL MSFileSystemForIface::CreateDirectoryW(LPCWSTR lpPathName) throw()
{
  SetLastError(ERROR_FUNCTION_NOT_CALLED);
  return FALSE;
}

_Use_decl_annotations_
DWORD MSFileSystemForIface::GetCurrentDirectoryW(DWORD nBufferLength, LPWSTR lpBuffer) throw()
{
  DWORD written = 0;
  HRESULT hr = S_OK;

  IFC(m_system->GetCurrentDirectoryForStorage(nBufferLength, lpBuffer, &written));

Cleanup:
  if (FAILED(hr))
  {
    SetLastError(WIN32_FROM_HRESULT(hr));
    return 0;
  }

  return written;
}

_Use_decl_annotations_
DWORD MSFileSystemForIface::GetMainModuleFileNameW(LPWSTR lpFilename, DWORD nSize) throw()
{
  DWORD written = 0;
  HRESULT hr = S_OK;

  IFC(m_system->GetMainModuleFileNameW(nSize, lpFilename, &written));

Cleanup:
  if (FAILED(hr))
  {
    SetLastError(WIN32_FROM_HRESULT(hr));
    return 0;
  }

  return written;
}

_Use_decl_annotations_
DWORD MSFileSystemForIface::GetTempPathW(DWORD nBufferLength, LPWSTR lpBuffer) throw()
{
  DWORD written = 0;
  HRESULT hr = S_OK;

  IFC(m_system->GetTempStoragePath(nBufferLength, lpBuffer, &written));

Cleanup:
  if (FAILED(hr))
  {
    SetLastError(WIN32_FROM_HRESULT(hr));
    return 0;
  }

  return written;
}

_Use_decl_annotations_
BOOLEAN MSFileSystemForIface::CreateSymbolicLinkW(LPCWSTR lpSymlinkFileName, LPCWSTR lpTargetFileName, DWORD dwFlags) throw()
{
  SetLastError(ERROR_FUNCTION_NOT_CALLED);
  return FALSE;
}

bool MSFileSystemForIface::SupportsCreateSymbolicLink() throw()
{
  return false;
}

_Use_decl_annotations_
BOOL MSFileSystemForIface::ReadFile(HANDLE hFile, LPVOID lpBuffer, DWORD nNumberOfBytesToRead, _Out_opt_ LPDWORD lpNumberOfBytesRead) throw()
{
  HRESULT hr = S_OK;
  CComPtr<IStream> stream;
  GetHandleStream(hFile, &stream);
  ULONG cbRead;
  IFC(stream->Read(lpBuffer, nNumberOfBytesToRead, &cbRead));
  if (lpNumberOfBytesRead != nullptr)
  {
    *lpNumberOfBytesRead = cbRead;
  }

Cleanup:
  if (FAILED(hr))
  {
    SetLastError(WIN32_FROM_HRESULT(hr));
    return FALSE;
  }

  return TRUE;
}

_Use_decl_annotations_
HANDLE MSFileSystemForIface::CreateFileMappingW(HANDLE hFile, DWORD flProtect, DWORD dwMaximumSizeHigh, DWORD dwMaximumSizeLow) throw()
{
  HRESULT hr = S_OK;
  HANDLE result = INVALID_HANDLE_VALUE;
  CComPtr<IUnknown> storage;
  CComPtr<IUnknown> mapping;

  GetHandleStorage(hFile, &storage);
  IFC(m_system->CreateStorageMapping(storage, flProtect, dwMaximumSizeHigh, dwMaximumSizeLow, &mapping));
  IFC(AddMappingHandle(mapping, &result));

Cleanup:
  if (FAILED(hr))
  {
    SetLastError(WIN32_FROM_HRESULT(hr));
    return INVALID_HANDLE_VALUE;
  }

  return result;
}

_Use_decl_annotations_
LPVOID MSFileSystemForIface::MapViewOfFile(HANDLE hFileMappingObject, DWORD dwDesiredAccess, DWORD dwFileOffsetHigh, DWORD dwFileOffsetLow, SIZE_T dwNumberOfBytesToMap) throw()
{
  HRESULT hr = S_OK;
  CComPtr<IUnknown> mapping;
  CComPtr<ID3D10Blob> blob;

  GetHandleMapping(hFileMappingObject, &mapping);
  IFC(m_system->MapViewOfFile(mapping, dwDesiredAccess, dwFileOffsetHigh, dwFileOffsetLow, dwNumberOfBytesToMap, &blob));
  IFC(AddMappingView(blob));

Cleanup:
  if (FAILED(hr))
  {
    SetLastError(WIN32_FROM_HRESULT(hr));
    return INVALID_HANDLE_VALUE;
  }

  return blob->GetBufferPointer();
}

_Use_decl_annotations_
BOOL MSFileSystemForIface::UnmapViewOfFile(LPCVOID lpBaseAddress) throw()
{
  RemoveMappingView(lpBaseAddress);
  return TRUE;
}

bool MSFileSystemForIface::FileDescriptorIsDisplayed(int fd) throw()
{
  return false;
}

unsigned MSFileSystemForIface::GetColumnCount(DWORD nStdHandle) throw()
{
  return 0;
}

unsigned MSFileSystemForIface::GetConsoleOutputTextAttributes() throw()
{
  return 0;
}

void MSFileSystemForIface::SetConsoleOutputTextAttributes(unsigned attributes) throw()
{
  return;
}

void MSFileSystemForIface::ResetConsoleOutputTextAttributes() throw()
{
}

int MSFileSystemForIface::open_osfhandle(intptr_t osfhandle, int flags) throw()
{
  return GetHandleFD((HANDLE)osfhandle);
}

HRESULT MSFileSystemForIface::EnsureFDAvailable(int fd)
{
  MSFileSystemHandle* ptr;
  switch (fd)
  {
  case 0: ptr = &m_knownHandle0; break;
  case 1: ptr = &m_knownHandle1; break;
  case 2: ptr = &m_knownHandle2; break;
  default:
    return S_OK;
  }

  HRESULT hr = S_OK;
  if (ptr->storage == nullptr)
  {
    CComPtr<IUnknown> storage;
    CComPtr<IStream> stream;

    IFC(m_system->OpenStdStorage(fd, &storage));
    IFC(storage.QueryInterface(&stream));

    ptr->storage = storage;
    ptr->stream = stream;
  }

  DXASSERT(ptr->storage != nullptr, "otherwise we should have failed to initialize");
  DXASSERT(ptr->stream != nullptr, "otherwise we should have failed to initialize - input/output/error should support streams");

Cleanup:
  return hr;
}

HANDLE MSFileSystemForIface::GetHandleForFD(int fd) throw()
{
  MSFileSystemHandle* ptr;
  switch (fd)
  {
  case 0:  ptr = &m_knownHandle0; break;
  case 1:  ptr = &m_knownHandle1; break;
  case 2:  ptr = &m_knownHandle2; break;
  default: ptr = (MSFileSystemHandle *)(uintptr_t)fd; break;
  }
  return ptr->GetHandle();
}

intptr_t MSFileSystemForIface::get_osfhandle(int fd) throw() {
  if (FAILED(EnsureFDAvailable(fd))) {
    errno = EBADF;
    return -1;
  }

  return (intptr_t)GetHandleForFD(fd);
}

int MSFileSystemForIface::close(int fd) throw()
{
  HANDLE h = GetHandleForFD(fd);
  this->CloseInternalHandle(h);
  return 0;
}

long MSFileSystemForIface::lseek(int fd, long offset, int origin) throw()
{
  HRESULT hr = S_OK;
  CComPtr<IStream> stream;
  LARGE_INTEGER li;
  ULARGE_INTEGER uli;
  if (FAILED(EnsureFDAvailable(fd)))
  {
    errno = EBADF;
    return -1;
  }

  GetHandleStream(GetHandleForFD(fd), &stream);
  li.HighPart = 0;
  li.LowPart = offset;
  IFC(stream->Seek(li, origin, &uli));

Cleanup:
  if (FAILED(hr))
  {
    errno = EINVAL;
    return -1;
  }

  if (uli.HighPart > 0)
  {
    errno = EOVERFLOW;
    return -1;
  }

  return uli.LowPart;
}

int MSFileSystemForIface::setmode(int fd, int mode) throw()
{
  return 0;
}

_Use_decl_annotations_
errno_t MSFileSystemForIface::resize_file(LPCWSTR path, uint64_t size) throw()
{
  return EBADF;
}

_Use_decl_annotations_
int MSFileSystemForIface::Read(int fd, void* buffer, unsigned int count) throw()
{
  HRESULT hr = S_OK;
  CComPtr<IStream> stream;
  ULONG cbRead = 0;
  if (FAILED(EnsureFDAvailable(fd)))
  {
    errno = EBADF;
    return -1;
  }

  GetHandleStream(GetHandleForFD(fd), &stream);
  IFC(stream->Read(buffer, count, &cbRead));

Cleanup:
  if (FAILED(hr))
  {
    errno = EINVAL;
    return -1;
  }

  return (int)cbRead;
}

_Use_decl_annotations_
int MSFileSystemForIface::Write(int fd, const void* buffer, unsigned int count) throw()
{
  HRESULT hr = S_OK;
  CComPtr<IStream> stream;
  ULONG cbWritten = 0;
  if (FAILED(EnsureFDAvailable(fd)))
  {
    errno = EBADF;
    return -1;
  }

  GetHandleStream(GetHandleForFD(fd), &stream);
  IFC(stream->Write(buffer, count, &cbWritten));

Cleanup:
  if (FAILED(hr))
  {
    errno = EINVAL;
    return -1;
  }

  return (int)cbWritten;
}

#ifndef _WIN32
int MSFileSystemForIface::Open(const char *lpFileName, int flags, mode_t mode) throw() {
  SetLastError(ERROR_FUNCTION_NOT_CALLED);
  return FALSE;
}

int MSFileSystemForIface::Stat(const char *lpFileName, struct stat *Status) throw() {
  SetLastError(ERROR_FUNCTION_NOT_CALLED);
  return FALSE;
}

int MSFileSystemForIface::Fstat(int FD, struct stat *Status) throw() {
  SetLastError(ERROR_FUNCTION_NOT_CALLED);
  return FALSE;
}
#endif

} // end namespace fs
} // end namespace sys
} // end namespace llvm

///////////////////////////////////////////////////////////////////////////////////////////////////
// Blocked MSFileSystem implementation.

#ifndef NDEBUG
static void MSFileSystemBlockedCalled() { DebugBreak(); }
#else
static void MSFileSystemBlockedCalled() { }
#endif

static BOOL MSFileSystemBlockedErrWin32()
{
  MSFileSystemBlockedCalled();
  SetLastError(ERROR_FUNCTION_NOT_CALLED);
  return FALSE;
}

static HANDLE MSFileSystemBlockedHandle()
{
  MSFileSystemBlockedCalled();
  SetLastError(ERROR_FUNCTION_NOT_CALLED);
  return INVALID_HANDLE_VALUE;
}

static int MSFileSystemBlockedErrno()
{
  MSFileSystemBlockedCalled();
  errno = EBADF;
  return -1;
}

static int MSFileSystemBlockedErrnoT()
{
  MSFileSystemBlockedCalled();
  return EBADF;
}

namespace llvm {
namespace sys  {
namespace fs {

class MSFileSystemBlocked : public MSFileSystem
{
private:
public:
  MSFileSystemBlocked();

  virtual BOOL FindNextFileW(_In_ HANDLE , _Out_ LPWIN32_FIND_DATAW ) throw() override
  { return MSFileSystemBlockedErrWin32(); }

  virtual HANDLE FindFirstFileW(_In_ LPCWSTR lpFileName, _Out_ LPWIN32_FIND_DATAW lpFindFileData) throw() override
  { return MSFileSystemBlockedHandle(); }

  virtual void FindClose(HANDLE findHandle) throw() override
  { MSFileSystemBlockedCalled(); }

  virtual HANDLE CreateFileW(_In_ LPCWSTR lpFileName, _In_ DWORD dwDesiredAccess, _In_ DWORD dwShareMode, _In_ DWORD dwCreationDisposition, _In_ DWORD dwFlagsAndAttributes) throw() override
  { return MSFileSystemBlockedHandle(); }

  virtual BOOL SetFileTime(_In_ HANDLE hFile, _In_opt_ const FILETIME *lpCreationTime, _In_opt_ const FILETIME *lpLastAccessTime, _In_opt_ const FILETIME *lpLastWriteTime) throw() override
  { return MSFileSystemBlockedErrWin32(); }

  virtual BOOL GetFileInformationByHandle(_In_ HANDLE hFile, _Out_ LPBY_HANDLE_FILE_INFORMATION lpFileInformation) throw() override
  { return MSFileSystemBlockedErrWin32(); }

  virtual DWORD GetFileType(_In_ HANDLE hFile) throw() override
  { MSFileSystemBlockedErrWin32(); return FILE_TYPE_UNKNOWN; }

  virtual BOOL CreateHardLinkW(_In_ LPCWSTR lpFileName, _In_ LPCWSTR lpExistingFileName) throw() override
  { return MSFileSystemBlockedErrWin32(); }

  virtual BOOL MoveFileExW(_In_ LPCWSTR lpExistingFileName, _In_opt_ LPCWSTR lpNewFileName, _In_ DWORD dwFlags) throw() override
  { return MSFileSystemBlockedErrWin32(); }

  virtual DWORD GetFileAttributesW(_In_ LPCWSTR lpFileName) throw() override
  { MSFileSystemBlockedErrWin32(); return 0; }

  virtual BOOL CloseHandle(_In_ HANDLE hObject) throw() override
  { return MSFileSystemBlockedErrWin32(); }

  virtual BOOL DeleteFileW(_In_ LPCWSTR lpFileName) throw() override
  { return MSFileSystemBlockedErrWin32(); }

  virtual BOOL RemoveDirectoryW(LPCWSTR lpFileName) throw() override
  { return MSFileSystemBlockedErrWin32(); }

  virtual BOOL CreateDirectoryW(_In_ LPCWSTR lpPathName) throw() override
  { return MSFileSystemBlockedErrWin32(); }

  _Success_(return != 0 && return < nBufferLength)
  virtual DWORD GetCurrentDirectoryW(_In_ DWORD nBufferLength, _Out_writes_to_opt_(nBufferLength, return +1) LPWSTR lpBuffer) throw() override;
  virtual DWORD GetMainModuleFileNameW(__out_ecount_part(nSize, return +1) LPWSTR lpFilename, DWORD nSize) throw() override;
  _Success_(return != 0 && return < nBufferLength)
  virtual DWORD GetTempPathW(DWORD nBufferLength, _Out_writes_to_opt_(nBufferLength, return +1) LPWSTR lpBuffer) throw() override;

  virtual BOOLEAN CreateSymbolicLinkW(_In_ LPCWSTR lpSymlinkFileName, _In_ LPCWSTR lpTargetFileName, DWORD dwFlags) throw() override
  { return MSFileSystemBlockedErrWin32(); }

  virtual bool SupportsCreateSymbolicLink() throw() override
  { MSFileSystemBlockedErrWin32(); return false; }

  virtual BOOL ReadFile(_In_ HANDLE hFile, _Out_ LPVOID lpBuffer, _In_ DWORD nNumberOfBytesToRead, _Out_opt_ LPDWORD lpNumberOfBytesRead) throw() override
  { return MSFileSystemBlockedErrWin32(); }

  virtual HANDLE CreateFileMappingW(_In_ HANDLE hFile, _In_ DWORD flProtect, _In_ DWORD dwMaximumSizeHigh, _In_ DWORD dwMaximumSizeLow) throw() override
  { return MSFileSystemBlockedHandle(); }

  virtual LPVOID MapViewOfFile(_In_ HANDLE hFileMappingObject, _In_ DWORD dwDesiredAccess, _In_ DWORD dwFileOffsetHigh, _In_ DWORD dwFileOffsetLow, _In_ SIZE_T dwNumberOfBytesToMap) throw() override
  { MSFileSystemBlockedErrWin32(); return nullptr; }

  virtual BOOL UnmapViewOfFile(_In_ LPCVOID lpBaseAddress) throw() override
  { return MSFileSystemBlockedErrWin32(); }
  
  // Console APIs.
  virtual bool FileDescriptorIsDisplayed(int fd) throw() override
  { MSFileSystemBlockedCalled(); return false; }

  virtual unsigned GetColumnCount(DWORD nStdHandle) throw() override
  { MSFileSystemBlockedCalled(); return 80; }
  
  virtual unsigned GetConsoleOutputTextAttributes() throw() override
  { MSFileSystemBlockedCalled(); return 0; }

  virtual void SetConsoleOutputTextAttributes(unsigned attributes) throw() override
  { MSFileSystemBlockedCalled(); }
  
  virtual void ResetConsoleOutputTextAttributes() throw() override
  { MSFileSystemBlockedCalled(); }

  // CRT APIs.
  virtual int open_osfhandle(intptr_t osfhandle, int flags) throw() override
  { return MSFileSystemBlockedErrno(); }

  virtual intptr_t get_osfhandle(int fd) throw() override
  { MSFileSystemBlockedErrno(); return 0; }

  virtual int close(int fd) throw() override
  { return MSFileSystemBlockedErrno(); }

  virtual long lseek(int fd, long offset, int origin) throw() override
  { return MSFileSystemBlockedErrno(); }

  virtual int setmode(int fd, int mode) throw() override
  { return MSFileSystemBlockedErrno(); }

  virtual errno_t resize_file(_In_ LPCWSTR path, uint64_t size) throw() override
  { return MSFileSystemBlockedErrnoT(); }

  virtual int Read(int fd, void* buffer, unsigned int count) throw() override
  { return MSFileSystemBlockedErrno(); }

  virtual int Write(int fd, const void* buffer, unsigned int count) throw() override
  { return MSFileSystemBlockedErrno(); }

  // Unix interface
#ifndef _WIN32
  virtual int Open(const char *lpFileName, int flags, mode_t mode) throw() override
  { return MSFileSystemBlockedErrno(); }

  virtual int Stat(const char *lpFileName, struct stat *Status) throw() override
  { return MSFileSystemBlockedErrno(); }

  virtual int Fstat(int FD, struct stat *Status) throw() override
  { return MSFileSystemBlockedErrno(); }
#endif

};

MSFileSystemBlocked::MSFileSystemBlocked()
{
}

_Use_decl_annotations_
DWORD MSFileSystemBlocked::GetCurrentDirectoryW(DWORD nBufferLength, LPWSTR lpBuffer) throw()
{
  if (nBufferLength > 1)
  {
    lpBuffer[0] = L'.';
    lpBuffer[1] = L'\0';
  }
  return 1;
}

_Use_decl_annotations_
DWORD MSFileSystemBlocked::GetMainModuleFileNameW(LPWSTR lpFilename, DWORD nSize) throw()
{
  SetLastError(NO_ERROR);
  return 0;
}

_Use_decl_annotations_
DWORD MSFileSystemBlocked::GetTempPathW(DWORD nBufferLength, LPWSTR lpBuffer) throw()
{
  if (nBufferLength > 1)
  {
    lpBuffer[0] = L'.';
    lpBuffer[1] = L'\0';
  }
  return 1;
}

} // end namespace fs
} // end namespace sys
} // end namespace llvm


///////////////////////////////////////////////////////////////////////////////////////////////////
// Externally visible functions.

_Use_decl_annotations_
HRESULT CreateMSFileSystemForIface(IUnknown* pService, ::llvm::sys::fs::MSFileSystem** pResult) throw()
{
  DXASSERT_NOMSG(pService != nullptr);
  DXASSERT_NOMSG(pResult != nullptr);
  CComPtr<IDxcSystemAccess> systemAccess;

  HRESULT hr = pService->QueryInterface(__uuidof(IDxcSystemAccess), (void**)&systemAccess);
  if (FAILED(hr)) return hr;

  *pResult = new (std::nothrow) ::llvm::sys::fs::MSFileSystemForIface(systemAccess);
  return (*pResult != nullptr) ? S_OK : E_OUTOFMEMORY;
}

_Use_decl_annotations_
HRESULT CreateMSFileSystemBlocked(::llvm::sys::fs::MSFileSystem** pResult) throw()
{
  DXASSERT_NOMSG(pResult != nullptr);
  *pResult = new (std::nothrow) ::llvm::sys::fs::MSFileSystemBlocked();
  return (*pResult != nullptr) ? S_OK : E_OUTOFMEMORY;
}

#endif // _WIN32
