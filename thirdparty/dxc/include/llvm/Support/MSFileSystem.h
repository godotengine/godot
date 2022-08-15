///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// MSFileSystem.h                                                            //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Provides error code values for the DirectX compiler.                      //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#ifndef LLVM_SUPPORT_MSFILESYSTEM_H
#define LLVM_SUPPORT_MSFILESYSTEM_H

///////////////////////////////////////////////////////////////////////////////////////////////////
// MSFileSystem interface.
struct stat;

namespace llvm {
namespace sys  {
namespace fs {

class MSFileSystem
{
public:
  virtual ~MSFileSystem() { };
  virtual BOOL FindNextFileW(
    _In_   HANDLE hFindFile,
    _Out_  LPWIN32_FIND_DATAW lpFindFileData) throw() = 0;
  virtual HANDLE FindFirstFileW(
    _In_   LPCWSTR lpFileName,
    _Out_  LPWIN32_FIND_DATAW lpFindFileData) throw() = 0;
  virtual void FindClose(HANDLE findHandle) throw() = 0;
  virtual HANDLE CreateFileW(
    _In_      LPCWSTR lpFileName,
    _In_      DWORD dwDesiredAccess,
    _In_      DWORD dwShareMode,
    _In_      DWORD dwCreationDisposition,
    _In_      DWORD dwFlagsAndAttributes) throw() = 0;
  virtual BOOL SetFileTime(_In_ HANDLE hFile,
    _In_opt_  const FILETIME *lpCreationTime,
    _In_opt_  const FILETIME *lpLastAccessTime,
    _In_opt_  const FILETIME *lpLastWriteTime) throw() = 0;
  virtual BOOL GetFileInformationByHandle(_In_ HANDLE hFile, _Out_ LPBY_HANDLE_FILE_INFORMATION lpFileInformation) throw() = 0;
  virtual DWORD GetFileType(_In_ HANDLE hFile) throw() = 0;
  virtual BOOL CreateHardLinkW(_In_ LPCWSTR lpFileName, _In_ LPCWSTR lpExistingFileName) throw() = 0;
  virtual BOOL MoveFileExW(_In_ LPCWSTR lpExistingFileName, _In_opt_ LPCWSTR lpNewFileName, _In_ DWORD dwFlags) throw() = 0;
  virtual DWORD GetFileAttributesW(_In_ LPCWSTR lpFileName) throw() = 0;
  virtual BOOL CloseHandle(_In_ HANDLE hObject) throw() = 0;
  virtual BOOL DeleteFileW(_In_ LPCWSTR lpFileName) throw() = 0;
  virtual BOOL RemoveDirectoryW(_In_ LPCWSTR lpFileName) throw() = 0;
  virtual BOOL CreateDirectoryW(_In_ LPCWSTR lpPathName) throw() = 0;
  _Success_(return != 0 && return < nBufferLength)
  virtual DWORD GetCurrentDirectoryW(_In_ DWORD nBufferLength, _Out_writes_to_opt_(nBufferLength, return + 1) LPWSTR lpBuffer) throw() = 0;
  _Success_(return != 0 && return < nSize)
  virtual DWORD GetMainModuleFileNameW(__out_ecount_part(nSize, return + 1) LPWSTR lpFilename, DWORD nSize) throw() = 0;
  virtual DWORD GetTempPathW(DWORD nBufferLength, _Out_writes_to_opt_(nBufferLength, return + 1) LPWSTR lpBuffer) throw() = 0;
  virtual BOOLEAN CreateSymbolicLinkW(_In_ LPCWSTR lpSymlinkFileName, _In_ LPCWSTR lpTargetFileName, DWORD dwFlags) throw() = 0;
  virtual bool SupportsCreateSymbolicLink() throw() = 0;
  virtual BOOL ReadFile(_In_ HANDLE hFile, _Out_bytecap_(nNumberOfBytesToRead) LPVOID lpBuffer, _In_ DWORD nNumberOfBytesToRead, _Out_opt_ LPDWORD lpNumberOfBytesRead) throw() = 0;
  virtual HANDLE CreateFileMappingW(
    _In_      HANDLE hFile,
    _In_      DWORD flProtect,
    _In_      DWORD dwMaximumSizeHigh,
    _In_      DWORD dwMaximumSizeLow) throw() = 0;
  virtual LPVOID MapViewOfFile(
    _In_  HANDLE hFileMappingObject,
    _In_  DWORD dwDesiredAccess,
    _In_  DWORD dwFileOffsetHigh,
    _In_  DWORD dwFileOffsetLow,
    _In_  SIZE_T dwNumberOfBytesToMap) throw() = 0;
  virtual BOOL UnmapViewOfFile(_In_ LPCVOID lpBaseAddress) throw() = 0;
  
  // Console APIs.
  virtual bool FileDescriptorIsDisplayed(int fd) throw() = 0;
  virtual unsigned GetColumnCount(DWORD nStdHandle) throw() = 0;
  virtual unsigned GetConsoleOutputTextAttributes() throw() = 0;
  virtual void SetConsoleOutputTextAttributes(unsigned) throw() = 0;
  virtual void ResetConsoleOutputTextAttributes() throw() = 0;

  // CRT APIs.
  virtual int open_osfhandle(intptr_t osfhandle, int flags) throw() = 0;
  virtual intptr_t get_osfhandle(int fd) throw() = 0;
  virtual int close(int fd) throw() = 0;
  virtual long lseek(int fd, long offset, int origin) throw() = 0;
  virtual int setmode(int fd, int mode) throw() = 0;
  virtual errno_t resize_file(_In_ LPCWSTR path, uint64_t size) throw() = 0; // A number of C calls.
  virtual int Read(int fd, _Out_bytecap_(count) void* buffer, unsigned int count) throw() = 0;
  virtual int Write(int fd, _In_bytecount_(count) const void* buffer, unsigned int count) throw() = 0;

  // Unix interface
#ifndef _WIN32
  virtual int Open(const char *lpFileName, int flags, mode_t mode = 0) throw() = 0;
  virtual int Stat(const char *lpFileName, struct stat *Status) throw() = 0;
  virtual int Fstat(int FD, struct stat *Status) throw() = 0;
#endif
};


} // end namespace fs
} // end namespace sys
} // end namespace llvm

/// <summary>Creates a Win32/CRT-based implementation with full fidelity for a console program.</summary>
/// <remarks>This requires the LLVM MS Support library to be linked in.</remarks>
HRESULT CreateMSFileSystemForDisk(_COM_Outptr_ ::llvm::sys::fs::MSFileSystem** pResult) throw();

struct IUnknown;

/// <summary>Creates an implementation based on IDxcSystemAccess.</summary>
HRESULT CreateMSFileSystemForIface(_In_ IUnknown* pService, _COM_Outptr_::llvm::sys::fs::MSFileSystem** pResult) throw();

#endif // LLVM_SUPPORT_MSFILESYSTEM_H
