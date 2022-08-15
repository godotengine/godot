//===- WindowsSupport.h - Common Windows Include File -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines things specific to Windows implementations.  In addition to
// providing some helpers for working with win32 APIs, this header wraps
// <windows.h> with some portability macros.  Always include WindowsSupport.h
// instead of including <windows.h> directly.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only generic Win32 code that
//===          is guaranteed to work on *all* Win32 variants.
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#ifndef LLVM_SUPPORT_WINDOWSSUPPORT_H
#define LLVM_SUPPORT_WINDOWSSUPPORT_H

/* HLSL Change - Moved to WinIncludes.h

These values need to be the same everywhere that windows headers are included.
// mingw-w64 tends to define it as 0x0502 in its headers.
#undef _WIN32_WINNT
#undef _WIN32_IE

// Require at least Windows XP(5.1) API.
#define _WIN32_WINNT 0x0501
#define _WIN32_IE    0x0600 // MinGW at it again.
*/

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Config/config.h" // Get build system configuration settings
#include "llvm/Support/Compiler.h"
#include "dxc/Support/WinIncludes.h"
#include <system_error>
#include <synchapi.h>
#include <wincrypt.h>
#include <cassert>
#include <string>
#include <vector>

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MSFileSystem.h"

inline bool MakeErrMsg(std::string* ErrMsg, const std::string& prefix) {
  if (!ErrMsg)
    return true;
  char *buffer = NULL;
  DWORD R = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER |
                          FORMAT_MESSAGE_FROM_SYSTEM,
                          NULL, GetLastError(), 0, (LPSTR)&buffer, 1, NULL);
  if (R)
    *ErrMsg = prefix + buffer;
  else
    *ErrMsg = prefix + "Unknown error";

  LocalFree(buffer);
  return R != 0;
}

template <typename HandleTraits>
class ScopedHandle {
  typedef typename HandleTraits::handle_type handle_type;
  handle_type Handle;

  ScopedHandle(const ScopedHandle &other); // = delete;
  void operator=(const ScopedHandle &other); // = delete;
public:
  ScopedHandle()
    : Handle(HandleTraits::GetInvalid()) {}

  explicit ScopedHandle(handle_type h)
    : Handle(h) {}

  ~ScopedHandle() {
    if (HandleTraits::IsValid(Handle))
      HandleTraits::Close(Handle);
  }

  handle_type take() {
    handle_type t = Handle;
    Handle = HandleTraits::GetInvalid();
    return t;
  }

  ScopedHandle &operator=(handle_type h) {
    if (HandleTraits::IsValid(Handle))
      HandleTraits::Close(Handle);
    Handle = h;
    return *this;
  }

  // True if Handle is valid.
  explicit operator bool() const {
    return HandleTraits::IsValid(Handle) ? true : false;
  }

  operator handle_type() const {
    return Handle;
  }
};

struct CommonHandleTraits {
  typedef HANDLE handle_type;

  static handle_type GetInvalid() {
    return INVALID_HANDLE_VALUE;
  }

  static void Close(handle_type h) {
    ::llvm::sys::fs::GetCurrentThreadFileSystem()->CloseHandle(h);
  }

  static bool IsValid(handle_type h) {
    return h != GetInvalid();
  }
};

struct JobHandleTraits : CommonHandleTraits {
  static handle_type GetInvalid() {
    return NULL;
  }
};

struct CryptContextTraits : CommonHandleTraits {
  typedef HCRYPTPROV handle_type;

  static handle_type GetInvalid() {
    return 0;
  }

  static void Close(handle_type h) {
    ::CryptReleaseContext(h, 0);
  }

  static bool IsValid(handle_type h) {
    return h != GetInvalid();
  }
};

struct FindHandleTraits : CommonHandleTraits {
  static void Close(handle_type h) {
    ::llvm::sys::fs::GetCurrentThreadFileSystem()->FindClose(h);
  }
};

struct FileHandleTraits : CommonHandleTraits {};

typedef ScopedHandle<CommonHandleTraits> ScopedCommonHandle;
typedef ScopedHandle<FileHandleTraits>   ScopedFileHandle;
typedef ScopedHandle<CryptContextTraits> ScopedCryptContext;
typedef ScopedHandle<FindHandleTraits>   ScopedFindHandle;
typedef ScopedHandle<JobHandleTraits>    ScopedJobHandle;

namespace llvm {
template <class T>
class SmallVectorImpl;

template <class T>
typename SmallVectorImpl<T>::const_pointer
c_str(SmallVectorImpl<T> &str) {
  str.push_back(0);
  str.pop_back();
  return str.data();
}

namespace sys {
namespace path {
std::error_code widenPath(const Twine &Path8,
                          SmallVectorImpl<wchar_t> &Path16);
} // end namespace path

namespace windows {
std::error_code UTF8ToUTF16(StringRef utf8, SmallVectorImpl<wchar_t> &utf16);
std::error_code UTF16ToUTF8(const wchar_t *utf16, size_t utf16_len,
                            SmallVectorImpl<char> &utf8);
/// Convert from UTF16 to the current code page used in the system
std::error_code UTF16ToCurCP(const wchar_t *utf16, size_t utf16_len,
                             SmallVectorImpl<char> &utf8);
std::error_code ACPToUTF8(const char *acp, size_t acp_len,
                       SmallVectorImpl<char> &utf8);
} // end namespace windows
} // end namespace sys
} // end namespace llvm.

#endif
