//===-- WinFunctions.h - Windows Functions for other platforms --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines Windows-specific functions used in the codebase for
// non-Windows platforms.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_WINFUNCTIONS_H
#define LLVM_SUPPORT_WINFUNCTIONS_H

#ifndef _WIN32

#include "dxc/Support/WinAdapter.h"

HRESULT StringCchCopyEx(LPSTR pszDest, size_t cchDest, LPCSTR pszSrc,
                        LPSTR *ppszDestEnd, size_t *pcchRemaining, DWORD dwFlags);
HRESULT StringCchPrintfA(char *dst, size_t dstSize, const char *format, ...);
HRESULT UIntAdd(UINT uAugend, UINT uAddend, UINT *puResult);
HRESULT IntToUInt(int in, UINT *out);
HRESULT SizeTToInt(size_t in, INT *out);
HRESULT UInt32Mult(UINT a, UINT b, UINT *out);

int strnicmp(const char *str1, const char *str2, size_t count);
int _stricmp(const char *str1, const char *str2);
int _wcsicmp(const wchar_t *str1, const wchar_t *str2);
int _wcsnicmp(const wchar_t *str1, const wchar_t *str2, size_t n);
int wsprintf(wchar_t *wcs, const wchar_t *fmt, ...);
unsigned char _BitScanForward(unsigned long * Index, unsigned long Mask);
HRESULT CoGetMalloc(DWORD dwMemContext, IMalloc **ppMalloc);

HANDLE CreateFile2(_In_ LPCWSTR lpFileName, _In_ DWORD dwDesiredAccess,
                   _In_ DWORD dwShareMode, _In_ DWORD dwCreationDisposition,
                   _In_opt_ void *pCreateExParams);

HANDLE CreateFileW(_In_ LPCWSTR lpFileName, _In_ DWORD dwDesiredAccess,
                   _In_ DWORD dwShareMode, _In_opt_ void *lpSecurityAttributes,
                   _In_ DWORD dwCreationDisposition,
                   _In_ DWORD dwFlagsAndAttributes,
                   _In_opt_ HANDLE hTemplateFile);

BOOL GetFileSizeEx(_In_ HANDLE hFile, _Out_ PLARGE_INTEGER lpFileSize);

BOOL ReadFile(_In_ HANDLE hFile, _Out_ LPVOID lpBuffer,
              _In_ DWORD nNumberOfBytesToRead,
              _Out_opt_ LPDWORD lpNumberOfBytesRead,
              _Inout_opt_ void *lpOverlapped);
BOOL WriteFile(_In_ HANDLE hFile, _In_ LPCVOID lpBuffer,
               _In_ DWORD nNumberOfBytesToWrite,
               _Out_opt_ LPDWORD lpNumberOfBytesWritten,
               _Inout_opt_ void *lpOverlapped);

BOOL CloseHandle(_In_ HANDLE hObject);

// Windows-specific heap functions
HANDLE HeapCreate(DWORD flOptions, SIZE_T dwInitialSize , SIZE_T dwMaximumSize);
BOOL HeapDestroy(HANDLE heap);
LPVOID HeapAlloc(HANDLE hHeap, DWORD dwFlags, SIZE_T nBytes);
LPVOID HeapReAlloc(HANDLE hHeap, DWORD dwFlags, LPVOID lpMem, SIZE_T dwBytes);
BOOL HeapFree(HANDLE hHeap, DWORD dwFlags, LPVOID lpMem);
SIZE_T HeapSize(HANDLE hHeap, DWORD dwFlags, LPCVOID lpMem);
HANDLE GetProcessHeap();

#endif // _WIN32

#endif // LLVM_SUPPORT_WINFUNCTIONS_H
