//===- llvm/Support/Windows/MSFileSystemImpl.cpp DXComplier Impl *- C++ -*-===//
///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// MSFileSystemImpl.cpp                                                      //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// This file implements the DXCompiler specific implementation of the Path API.//
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include <fcntl.h>

#ifdef _WIN32
#include <io.h>
#else
#include <sys/types.h>
#include <unistd.h>
#endif

#include <sys/stat.h>
#include <sys/types.h>
#include <stdint.h>
#include <errno.h>
#include <new>

#include "dxc/Support/WinIncludes.h"
#include "dxc/Support/WinAdapter.h"
#include "llvm/Support/MSFileSystem.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
// Externally visible functions.

HRESULT CreateMSFileSystemForDisk(_COM_Outptr_ ::llvm::sys::fs::MSFileSystem** pResult) throw();

///////////////////////////////////////////////////////////////////////////////////////////////////
// Win32-and-CRT-based MSFileSystem implementation with direct filesystem access.

namespace llvm {
namespace sys  {
namespace fs {

class MSFileSystemForDisk : public MSFileSystem
{
public:
  unsigned _defaultAttributes;
  MSFileSystemForDisk();

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
  virtual DWORD GetCurrentDirectoryW(_In_ DWORD nBufferLength, _Out_writes_to_opt_(nBufferLength, return + 1) LPWSTR lpBuffer) throw() override;
  _Success_(return != 0 && return < nSize)
  virtual DWORD GetMainModuleFileNameW(__out_ecount_part(nSize, return + 1) LPWSTR lpFilename, DWORD nSize) throw() override;
  virtual DWORD GetTempPathW(DWORD nBufferLength, _Out_writes_to_opt_(nBufferLength, return + 1) LPWSTR lpBuffer) throw() override;
  virtual BOOLEAN CreateSymbolicLinkW(_In_ LPCWSTR lpSymlinkFileName, _In_ LPCWSTR lpTargetFileName, DWORD dwFlags) throw() override;
  virtual bool SupportsCreateSymbolicLink() throw() override;
  virtual BOOL ReadFile(_In_ HANDLE hFile, _Out_bytecap_(nNumberOfBytesToRead) LPVOID lpBuffer, _In_ DWORD nNumberOfBytesToRead, _Out_opt_ LPDWORD lpNumberOfBytesRead) throw() override;
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

MSFileSystemForDisk::MSFileSystemForDisk()
{
  #ifdef _WIN32
  _defaultAttributes = GetConsoleOutputTextAttributes();
  #endif
}

_Use_decl_annotations_
BOOL MSFileSystemForDisk::FindNextFileW(HANDLE hFindFile, LPWIN32_FIND_DATAW lpFindFileData) throw()
{
  #ifdef _WIN32
  return ::FindNextFileW(hFindFile, lpFindFileData);
  #else
  assert(false && "Not implemented for Unix");
  return false;
  #endif
}

_Use_decl_annotations_
HANDLE MSFileSystemForDisk::FindFirstFileW(LPCWSTR lpFileName, LPWIN32_FIND_DATAW lpFindFileData) throw()
{
  #ifdef _WIN32
  return ::FindFirstFileW(lpFileName, lpFindFileData);
  #else
  assert(false && "Not implemented for Unix");
  return nullptr;
  #endif
}

void MSFileSystemForDisk::FindClose(HANDLE findHandle) throw()
{
  #ifdef _WIN32
  ::FindClose(findHandle);
  #else
  assert(false && "Not implemented for Unix");
  #endif
}

_Use_decl_annotations_
HANDLE MSFileSystemForDisk::CreateFileW(LPCWSTR lpFileName, DWORD dwDesiredAccess, DWORD dwShareMode, DWORD dwCreationDisposition, DWORD dwFlagsAndAttributes) throw()
{
  #ifdef _WIN32
  return ::CreateFileW(lpFileName, dwDesiredAccess, dwShareMode, nullptr, dwCreationDisposition, dwFlagsAndAttributes, nullptr);
  #else
  assert(false && "Not implemented for Unix");
  return nullptr;
  #endif
}

_Use_decl_annotations_
BOOL MSFileSystemForDisk::SetFileTime(HANDLE hFile, _In_opt_ const FILETIME *lpCreationTime, _In_opt_ const FILETIME *lpLastAccessTime, _In_opt_ const FILETIME *lpLastWriteTime) throw()
{
  #ifdef _WIN32
  return ::SetFileTime(hFile, lpCreationTime, lpLastAccessTime, lpLastWriteTime);
  #else
  assert(false && "Not implemented for Unix");
  return false;
  #endif
}

_Use_decl_annotations_
BOOL MSFileSystemForDisk::GetFileInformationByHandle(HANDLE hFile, LPBY_HANDLE_FILE_INFORMATION lpFileInformation) throw()
{
  #ifdef _WIN32
  return ::GetFileInformationByHandle(hFile, lpFileInformation);
  #else
  assert(false && "Not implemented for Unix");
  return false;
  #endif
}

_Use_decl_annotations_
DWORD MSFileSystemForDisk::GetFileType(HANDLE hFile) throw()
{
  #ifdef _WIN32
  return ::GetFileType(hFile);
  #else
  assert(false && "Not implemented for Unix");
  return 0;
  #endif
}

_Use_decl_annotations_
BOOL MSFileSystemForDisk::CreateHardLinkW(LPCWSTR lpFileName, LPCWSTR lpExistingFileName) throw()
{
  #ifdef _WIN32
  return ::CreateHardLinkW(lpFileName, lpExistingFileName, nullptr);
  #else
  assert(false && "Not implemented for Unix");
  return false;
  #endif
}

_Use_decl_annotations_
BOOL MSFileSystemForDisk::MoveFileExW(LPCWSTR lpExistingFileName, LPCWSTR lpNewFileName, DWORD dwFlags) throw()
{
  #ifdef _WIN32
  return ::MoveFileExW(lpExistingFileName, lpNewFileName, dwFlags);
  #else
  assert(false && "Not implemented for Unix");
  return false;
  #endif
}

_Use_decl_annotations_
DWORD MSFileSystemForDisk::GetFileAttributesW(LPCWSTR lpFileName) throw()
{
  #ifdef _WIN32
  return ::GetFileAttributesW(lpFileName);
  #else
  assert(false && "Not implemented for Unix");
  return 0;
  #endif
}

_Use_decl_annotations_
BOOL MSFileSystemForDisk::CloseHandle(HANDLE hObject) throw()
{
  #ifdef _WIN32
  return ::CloseHandle(hObject);
  #else
  assert(false && "Not implemented for Unix");
  return false;
  #endif
}

_Use_decl_annotations_
BOOL MSFileSystemForDisk::DeleteFileW(LPCWSTR lpFileName) throw()
{
  #ifdef _WIN32
  return ::DeleteFileW(lpFileName);
  #else
  assert(false && "Not implemented for Unix");
  return false;
  #endif
}

_Use_decl_annotations_
BOOL MSFileSystemForDisk::RemoveDirectoryW(LPCWSTR lpFileName) throw()
{
  #ifdef _WIN32
  return ::RemoveDirectoryW(lpFileName);
  #else
  assert(false && "Not implemented for Unix");
  return false;
  #endif
}

_Use_decl_annotations_
BOOL MSFileSystemForDisk::CreateDirectoryW(LPCWSTR lpPathName) throw()
{
  #ifdef _WIN32
  return ::CreateDirectoryW(lpPathName, nullptr);
  #else
  assert(false && "Not implemented for Unix");
  return false;
  #endif
}

_Use_decl_annotations_
DWORD MSFileSystemForDisk::GetCurrentDirectoryW(DWORD nBufferLength,  LPWSTR lpBuffer) throw()
{
  #ifdef _WIN32
  return ::GetCurrentDirectoryW(nBufferLength, lpBuffer);
  #else
  assert(false && "Not implemented for Unix");
  return 0;
  #endif
}

_Use_decl_annotations_
DWORD MSFileSystemForDisk::GetMainModuleFileNameW(LPWSTR lpFilename, DWORD nSize) throw()
{
  #ifdef _WIN32
  // Add some code to ensure that the result is null terminated.
  if (nSize <= 1)
  {
    ::SetLastError(ERROR_INSUFFICIENT_BUFFER);
    return 0;
  }

  DWORD result = ::GetModuleFileNameW(nullptr, lpFilename, nSize - 1);
  if (result == 0) return result;
  lpFilename[result] = L'\0';
  return result;
  #else
  assert(false && "Not implemented for Unix");
  return 0;
  #endif
}

_Use_decl_annotations_
DWORD MSFileSystemForDisk::GetTempPathW(DWORD nBufferLength, LPWSTR lpBuffer) throw()
{
  #ifdef _WIN32
  return ::GetTempPathW(nBufferLength, lpBuffer);
  #else
  assert(false && "Not implemented for Unix");
  return 0;
  #endif
}

#ifdef _WIN32
namespace {
  typedef BOOLEAN(WINAPI *PtrCreateSymbolicLinkW)(
    /*__in*/ LPCWSTR lpSymlinkFileName,
    /*__in*/ LPCWSTR lpTargetFileName,
    /*__in*/ DWORD dwFlags);

  PtrCreateSymbolicLinkW create_symbolic_link_api =
    PtrCreateSymbolicLinkW(::GetProcAddress(
    ::GetModuleHandleW(L"Kernel32.dll"), "CreateSymbolicLinkW"));
}
#endif

_Use_decl_annotations_
BOOLEAN MSFileSystemForDisk::CreateSymbolicLinkW(LPCWSTR lpSymlinkFileName, LPCWSTR lpTargetFileName, DWORD dwFlags) throw()
{
  #ifdef _WIN32
  return create_symbolic_link_api(lpSymlinkFileName, lpTargetFileName, dwFlags);
  #else
  assert(false && "Not implemented for Unix");
  return false;
  #endif
}

bool MSFileSystemForDisk::SupportsCreateSymbolicLink() throw()
{
  #ifdef _WIN32
  return create_symbolic_link_api != nullptr;
  #else
  assert(false && "Not implemented for Unix");
  return false;
  #endif
}

_Use_decl_annotations_
BOOL MSFileSystemForDisk::ReadFile(HANDLE hFile, LPVOID lpBuffer, DWORD nNumberOfBytesToRead, _Out_opt_ LPDWORD lpNumberOfBytesRead) throw()
{
  #ifdef _WIN32
  return ::ReadFile(hFile, lpBuffer, nNumberOfBytesToRead, lpNumberOfBytesRead, nullptr);
  #else
  assert(false && "Not implemented for Unix");
  return false;
  #endif
}

_Use_decl_annotations_
HANDLE MSFileSystemForDisk::CreateFileMappingW(HANDLE hFile, DWORD flProtect, DWORD dwMaximumSizeHigh, DWORD dwMaximumSizeLow) throw()
{
  #ifdef _WIN32
  return ::CreateFileMappingW(hFile, nullptr, flProtect, dwMaximumSizeHigh, dwMaximumSizeLow, nullptr);
  #else
  assert(false && "Not implemented for Unix");
  return nullptr;
  #endif
}

_Use_decl_annotations_
LPVOID MSFileSystemForDisk::MapViewOfFile(HANDLE hFileMappingObject, DWORD dwDesiredAccess, DWORD dwFileOffsetHigh, DWORD dwFileOffsetLow, SIZE_T dwNumberOfBytesToMap) throw()
{
  #ifdef _WIN32
  return ::MapViewOfFile(hFileMappingObject, dwDesiredAccess, dwFileOffsetHigh, dwFileOffsetLow, dwNumberOfBytesToMap);
  #else
  assert(false && "Not implemented for Unix");
  return nullptr;
  #endif
}

_Use_decl_annotations_
BOOL MSFileSystemForDisk::UnmapViewOfFile(LPCVOID lpBaseAddress) throw()
{
  #ifdef _WIN32
  return ::UnmapViewOfFile(lpBaseAddress);
  #else
  assert(false && "Not implemented for Unix");
  return false;
  #endif
}

bool MSFileSystemForDisk::FileDescriptorIsDisplayed(int fd) throw()
{
  #ifdef _WIN32
  DWORD Mode;  // Unused
  return (GetConsoleMode((HANDLE)_get_osfhandle(fd), &Mode) != 0);
  #else
  assert(false && "Not implemented for Unix");
  return false;
  #endif
}

unsigned MSFileSystemForDisk::GetColumnCount(DWORD nStdHandle) throw()
{
  #ifdef _WIN32
  unsigned Columns = 0;
  CONSOLE_SCREEN_BUFFER_INFO csbi;
  if (::GetConsoleScreenBufferInfo(GetStdHandle(nStdHandle), &csbi))
    Columns = csbi.dwSize.X;
  return Columns;
  #else
  assert(false && "Not implemented for Unix");
  return 0;
  #endif
}

unsigned MSFileSystemForDisk::GetConsoleOutputTextAttributes() throw()
{
  #ifdef _WIN32
  CONSOLE_SCREEN_BUFFER_INFO csbi;
  if (::GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi))
    return csbi.wAttributes;
  return 0;
  #else
  assert(false && "Not implemented for Unix");
  return 0;
  #endif
}

void MSFileSystemForDisk::SetConsoleOutputTextAttributes(unsigned attributes) throw()
{
  #ifdef _WIN32
  ::SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), attributes);
  #else
  assert(false && "Not implemented for Unix");
  #endif
}

void MSFileSystemForDisk::ResetConsoleOutputTextAttributes() throw()
{
  #ifdef _WIN32
  ::SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), _defaultAttributes);
  #else
  assert(false && "Not implemented for Unix");
  #endif
}

int MSFileSystemForDisk::open_osfhandle(intptr_t osfhandle, int flags) throw()
{
  #ifdef _WIN32
  return ::_open_osfhandle(osfhandle, flags);
  #else
  assert(false && "Not implemented for Unix");
  return 0;
  #endif
}

intptr_t MSFileSystemForDisk::get_osfhandle(int fd) throw()
{
  #ifdef _WIN32
  return ::_get_osfhandle(fd);
  #else
  assert(false && "Not implemented for Unix");
  return 0;
  #endif
}

int MSFileSystemForDisk::close(int fd) throw()
{
  #ifdef _WIN32
  return ::_close(fd);
  #else
  return ::close(fd);
  #endif
}

long MSFileSystemForDisk::lseek(int fd, long offset, int origin) throw()
{
  #ifdef _WIN32
  return ::_lseek(fd, offset, origin);
  #else
  return ::lseek(fd, offset, origin);
  #endif
}

int MSFileSystemForDisk::setmode(int fd, int mode) throw()
{
  #ifdef _WIN32
  return ::_setmode(fd, mode);
  #else
  assert(false && "Not implemented for Unix");
  return 0;
  #endif
}

_Use_decl_annotations_
errno_t MSFileSystemForDisk::resize_file(LPCWSTR path, uint64_t size) throw()
{
  #ifdef _WIN32
  int fd = ::_wopen(path, O_BINARY | _O_RDWR, S_IWRITE);
  if (fd == -1)
    return errno;
#ifdef HAVE__CHSIZE_S
  errno_t error = ::_chsize_s(fd, size);
#else
  errno_t error = ::_chsize(fd, size);
#endif
  ::_close(fd);
  return error;

  #else
  assert(false && "Not implemented for Unix");
  return 0;
  #endif
}

_Use_decl_annotations_
int MSFileSystemForDisk::Read(int fd, void* buffer, unsigned int count) throw()
{
  #ifdef _WIN32
  return ::_read(fd, buffer, count);
  #else
  return ::read(fd, buffer, count);
  #endif
}

_Use_decl_annotations_
int MSFileSystemForDisk::Write(int fd, const void* buffer, unsigned int count) throw()
{
  #ifdef _WIN32
  return ::_write(fd, buffer, count);
  #else
  return ::write(fd, buffer, count);
  #endif
}

#ifndef _WIN32
int MSFileSystemForDisk::Open(const char *lpFileName, int flags, mode_t mode) throw() {
  return ::open(lpFileName, flags, mode);
}

int MSFileSystemForDisk::Stat(const char *lpFileName, struct stat *Status) throw() {
  return ::stat(lpFileName, Status);
}

int MSFileSystemForDisk::Fstat(int FD, struct stat *Status) throw() {
  return ::fstat(FD, Status);
}
#endif

} // end namespace fs
} // end namespace sys
} // end namespace llvm

///////////////////////////////////////////////////////////////////////////////////////////////////
// Externally visible functions.

HRESULT CreateMSFileSystemForDisk(_COM_Outptr_ ::llvm::sys::fs::MSFileSystem** pResult) throw()
{
  *pResult = new (std::nothrow) ::llvm::sys::fs::MSFileSystemForDisk();
  return (*pResult != nullptr) ? S_OK : E_OUTOFMEMORY;
}
